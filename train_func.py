import os
import torch
from torch import optim
import torch.nn.functional as F
import deepnovo_config
from data_reader import DeepNovoTrainDataset, collate_func
from model import DeepNovoModel, SpectrumCNN, device
import time
import math
import logging

logger = logging.getLogger(__name__)

forward_model_save_name = 'forward_deepnovo.pth'
backward_model_save_name = 'backward_deepnovo.pth'
spectrum_cnn_save_name = 'spectrum_cnn.pth'

logger.info(f"using device: {device}")


def to_one_hot(y, n_dims=None):
    """ Take integer y with n dims and convert it to 1-hot representation with n+1 dims. """
    y_tensor = y.data
    y_tensor = y_tensor.type(torch.LongTensor).view(-1, 1)
    n_dims = n_dims if n_dims is not None else int(torch.max(y_tensor)) + 1
    y_one_hot = torch.zeros(y_tensor.size()[0], n_dims).scatter_(1, y_tensor, 1)
    y_one_hot = y_one_hot.view(*y.shape, -1)
    return y_one_hot


def focal_loss(logits, labels, ignore_index=-100, gamma=2.):
    """

    :param logits: float tensor of shape [batch, T, 26]
    :param labels: long tensor of shape [batch, T]
    :param ignore_index: ignore the loss of those tokens
    :param gamma:
    :return: average loss, num_valid_token
    """
    valid_token_mask = (labels != ignore_index).float()  # [batch, T]
    num_valid_token = torch.sum(valid_token_mask)
    batch_size, T, num_classes = logits.size()
    sigmoid_p = torch.sigmoid(logits)
    target_tensor = to_one_hot(labels, n_dims=num_classes).float().to(device)
    zeros = torch.zeros_like(sigmoid_p)
    pos_p_sub = torch.where(target_tensor >= sigmoid_p, target_tensor - sigmoid_p, zeros)  # [batch, T, 26]
    neg_p_sub = torch.where(target_tensor > zeros, zeros, sigmoid_p)  # [batch, T, 26]

    per_token_loss = - (pos_p_sub ** gamma) * torch.log(torch.clamp(sigmoid_p, 1e-8, 1.0)) - \
                     (neg_p_sub ** gamma) * torch.log(torch.clamp(1.0 - sigmoid_p, 1e-8, 1.0))
    per_entry_loss = torch.sum(per_token_loss, dim=2)  # [batch, T]
    per_entry_loss = per_entry_loss * valid_token_mask  # masking out loss from pad tokens

    per_entry_average_loss = torch.sum(per_entry_loss) / (num_valid_token + 1e-6)
    return per_entry_average_loss, num_valid_token


def build_model(training=True):
    """

    :return:
    """
    forward_deepnovo = DeepNovoModel(deepnovo_config.vocab_size, deepnovo_config.num_ion, deepnovo_config.num_units,
                                     deepnovo_config.dropout_rate)
    backward_deepnovo = DeepNovoModel(deepnovo_config.vocab_size, deepnovo_config.num_ion, deepnovo_config.num_units,
                                      deepnovo_config.dropout_rate)
    if deepnovo_config.use_lstm:
        spectrum_cnn = SpectrumCNN(dropout_p=deepnovo_config.dropout_rate).to(device)
    else:
        spectrum_cnn = None
    # load pretrained params if exist
    if os.path.exists(os.path.join(deepnovo_config.train_dir, forward_model_save_name)):
        assert os.path.exists(os.path.join(deepnovo_config.train_dir, backward_model_save_name))
        logger.info("load pretrained model")
        if deepnovo_config.use_lstm:
            assert os.path.exists(os.path.join(deepnovo_config.train_dir, spectrum_cnn_save_name))
            spectrum_cnn.load_state_dict(torch.load(os.path.join(deepnovo_config.train_dir, spectrum_cnn_save_name),
                                                    map_location=device))
        forward_deepnovo.load_state_dict(torch.load(os.path.join(deepnovo_config.train_dir, forward_model_save_name),
                                                    map_location=device))
        backward_deepnovo.load_state_dict(torch.load(os.path.join(deepnovo_config.train_dir, backward_model_save_name),
                                                     map_location=device))
    else:
        assert training, f"building model for testing, but could not found weight under directory " \
                         f"{deepnovo_config.train_dir}"
        logger.info("initialize a set of new parameters")

    if deepnovo_config.use_lstm:
        # share embedding matrix
        backward_deepnovo.embedding.weight = forward_deepnovo.embedding.weight

    backward_deepnovo = backward_deepnovo.to(device)
    forward_deepnovo = forward_deepnovo.to(device)
    return forward_deepnovo, backward_deepnovo, spectrum_cnn


def validation(spectrum_cnn, forward_deepnovo, backward_deepnovo, valid_loader) -> float:
    with torch.no_grad():
        valid_loss = 0
        num_valid_samples = 0
        for data in valid_loader:
            spectrum_holder, \
            batch_forward_intensity, \
            batch_forward_id_input, \
            batch_forward_id_target, \
            batch_backward_intensity, \
            batch_backward_id_input, \
            batch_backward_id_target = data
            batch_size = spectrum_holder.size(0)

            # move to device
            spectrum_holder = spectrum_holder.to(device)
            batch_forward_intensity = batch_forward_intensity.to(device)
            batch_forward_id_input = batch_forward_id_input.to(device)
            batch_forward_id_target = batch_forward_id_target.to(device)
            batch_backward_intensity = batch_backward_intensity.to(device)
            batch_backward_id_input = batch_backward_id_input.to(device)
            batch_backward_id_target = batch_backward_id_target.to(device)

            if deepnovo_config.use_lstm:
                initial_state_tuple = spectrum_cnn(spectrum_holder)
            else:
                initial_state_tuple = None
            forward_logit, _ = forward_deepnovo(batch_forward_intensity, batch_forward_id_input, initial_state_tuple)
            backward_logit, _ = backward_deepnovo(batch_backward_intensity, batch_backward_id_input,
                                                  initial_state_tuple)
            forward_loss, f_num = focal_loss(forward_logit, batch_forward_id_target, ignore_index=0, gamma=2.)
            backward_loss, b_num = focal_loss(backward_logit, batch_backward_id_target, ignore_index=0, gamma=2.)
            valid_loss += forward_loss * f_num + backward_loss * b_num
            num_valid_samples += f_num + b_num
    average_valid_loss = valid_loss / (num_valid_samples + 1e-6)
    return average_valid_loss.item()


def perplexity(log_loss):
    return math.exp(log_loss) if log_loss < 300 else float('inf')


def adjust_learning_rate(optimizer, epoch):
    """Sets the learning rate to the initial LR decayed by 10 every 3 epochs"""
    lr = deepnovo_config.init_lr * (0.1 ** ((epoch + 1) // 3))
    logger.info(f"epoch: {epoch}\tlr: {lr}")
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def save_model(forward_deepnovo, backward_deepnovo, spectrum_cnn):
    torch.save(forward_deepnovo.state_dict(), os.path.join(deepnovo_config.train_dir,
                                                           forward_model_save_name))
    torch.save(backward_deepnovo.state_dict(), os.path.join(deepnovo_config.train_dir,
                                                            backward_model_save_name))
    if deepnovo_config.use_lstm:
        torch.save(spectrum_cnn.state_dict(), os.path.join(deepnovo_config.train_dir,
                                                           spectrum_cnn_save_name))


def train():
    train_set = DeepNovoTrainDataset(deepnovo_config.input_feature_file_train,
                                     deepnovo_config.input_spectrum_file_train)
    num_train_features = len(train_set)
    steps_per_epoch = int(num_train_features / deepnovo_config.batch_size)
    logger.info(f"{steps_per_epoch} steps per epoch")
    train_data_loader = torch.utils.data.DataLoader(dataset=train_set,
                                                    batch_size=deepnovo_config.batch_size,
                                                    shuffle=True,
                                                    num_workers=deepnovo_config.num_workers,
                                                    collate_fn=collate_func)
    valid_set = DeepNovoTrainDataset(deepnovo_config.input_feature_file_valid,
                                     deepnovo_config.input_spectrum_file_valid)
    valid_data_loader = torch.utils.data.DataLoader(dataset=valid_set,
                                                    batch_size=deepnovo_config.batch_size,
                                                    shuffle=False,
                                                    num_workers=deepnovo_config.num_workers,
                                                    collate_fn=collate_func)
    forward_deepnovo, backward_deepnovo, spectrum_cnn = build_model()
    if deepnovo_config.use_lstm:
        all_params = list(forward_deepnovo.parameters()) + list(backward_deepnovo.parameters()) + \
                     list(spectrum_cnn.parameters())
    else:
        all_params = list(forward_deepnovo.parameters()) + list(backward_deepnovo.parameters())
    optimizer = optim.Adam(all_params, lr=deepnovo_config.init_lr, weight_decay=deepnovo_config.weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', factor=0.5, verbose=True,
                                                           threshold=1e-4, cooldown=10, min_lr=1e-5)

    best_valid_loss = float("inf")
    # train loop
    best_epoch = None
    best_step = None
    start_time = time.time()
    for epoch in range(deepnovo_config.num_epoch):
        # learning rate schedule
        # adjust_learning_rate(optimizer, epoch)
        for i, data in enumerate(train_data_loader):
            optimizer.zero_grad()

            spectrum_holder, \
            batch_forward_intensity, \
            batch_forward_id_input, \
            batch_forward_id_target, \
            batch_backward_intensity, \
            batch_backward_id_input, \
            batch_backward_id_target = data

            # move to device
            spectrum_holder = spectrum_holder.to(device)
            batch_forward_intensity = batch_forward_intensity.to(device)
            batch_forward_id_input = batch_forward_id_input.to(device)
            batch_forward_id_target = batch_forward_id_target.to(device)
            batch_backward_intensity = batch_backward_intensity.to(device)
            batch_backward_id_input = batch_backward_id_input.to(device)
            batch_backward_id_target = batch_backward_id_target.to(device)

            if deepnovo_config.use_lstm:
                initial_state_tuple = spectrum_cnn(spectrum_holder)
            else:
                initial_state_tuple = None
            forward_logit, _ = forward_deepnovo(batch_forward_intensity, batch_forward_id_input, initial_state_tuple)
            backward_logit, _ = backward_deepnovo(batch_backward_intensity, batch_backward_id_input,
                                                  initial_state_tuple)
            forward_loss, _ = focal_loss(forward_logit, batch_forward_id_target, ignore_index=0, gamma=2.)
            backward_loss, _ = focal_loss(backward_logit, batch_backward_id_target, ignore_index=0, gamma=2.)
            total_loss = (forward_loss + backward_loss) / 2.
            # compute gradient
            total_loss.backward()
            # clip gradient
            torch.nn.utils.clip_grad_norm_(all_params, deepnovo_config.max_gradient_norm)
            optimizer.step()

            if (i + 1) % deepnovo_config.steps_per_validation == 0:
                duration = time.time() - start_time
                step_time = duration / deepnovo_config.steps_per_validation
                loss_cpu = total_loss.item()
                # evaluation mode
                if deepnovo_config.use_lstm:
                    spectrum_cnn.eval()
                forward_deepnovo.eval()
                backward_deepnovo.eval()
                validation_loss = validation(spectrum_cnn, forward_deepnovo, backward_deepnovo, valid_data_loader)
                scheduler.step(validation_loss)
                logger.info(f"epoch {epoch} step {i}/{steps_per_epoch}, "
                            f"train perplexity: {perplexity(loss_cpu)}\t"
                            f"validation perplexity: {perplexity(validation_loss)}\tstep time: {step_time}")
                if validation_loss < best_valid_loss:
                    best_valid_loss = validation_loss
                    logger.info(f"best valid loss achieved at epoch {epoch} step {i}")
                    best_epoch = epoch
                    best_step = i
                    # save model if achieve a new best valid loss
                    save_model(forward_deepnovo, backward_deepnovo, spectrum_cnn)

                # back to train model
                if deepnovo_config.use_lstm:
                    spectrum_cnn.train()
                forward_deepnovo.train()
                backward_deepnovo.train()

                start_time = time.time()

    logger.info(f"best model at epoch {best_epoch} step {best_step}")
