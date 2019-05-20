# Copyright 2019 Rui Qiao. All Rights Reserved.
#
# DeepNovoV2 is publicly available for non-commercial uses.
# ==============================================================================

import torch
import logging
import logging.config
import deepnovo_config
from train_func import train, build_model, validation, perplexity
from data_reader import DeepNovoDenovoDataset, collate_func, DeepNovoTrainDataset
from model import InferenceModelWrapper
from denovo import IonCNNDenovo
from writer import DenovoWriter
import deepnovo_worker_test
from deepnovo_dia_script_select import find_score_cutoff
logger = logging.getLogger(__name__)


def main():
    if deepnovo_config.args.train:
        logger.info("training mode")
        train()
    elif deepnovo_config.args.search_denovo:
        logger.info("denovo mode")
        data_reader = DeepNovoDenovoDataset(feature_filename=deepnovo_config.denovo_input_feature_file,
                                            spectrum_filename=deepnovo_config.denovo_input_spectrum_file)
        denovo_worker = IonCNNDenovo(deepnovo_config.MZ_MAX,
                                     deepnovo_config.knapsack_file,
                                     beam_size=deepnovo_config.args.beam_size)
        forward_deepnovo, backward_deepnovo, init_net = build_model(training=False)
        model_wrapper = InferenceModelWrapper(forward_deepnovo, backward_deepnovo, init_net)
        writer = DenovoWriter(deepnovo_config.denovo_output_file)
        denovo_worker.search_denovo(model_wrapper, data_reader, writer)
    elif deepnovo_config.args.valid:
        valid_set = DeepNovoTrainDataset(deepnovo_config.input_feature_file_valid,
                                         deepnovo_config.input_spectrum_file_valid)
        valid_data_loader = torch.utils.data.DataLoader(dataset=valid_set,
                                                        batch_size=deepnovo_config.batch_size,
                                                        shuffle=False,
                                                        num_workers=deepnovo_config.num_workers,
                                                        collate_fn=collate_func)
        forward_deepnovo, backward_deepnovo, init_net = build_model(training=False)
        forward_deepnovo.eval()
        backward_deepnovo.eval()
        validation_loss = validation(forward_deepnovo, backward_deepnovo, init_net, valid_data_loader)
        logger.info(f"validation perplexity: {perplexity(validation_loss)}")

    elif deepnovo_config.args.test:

        logger.info("test mode")
        worker_test = deepnovo_worker_test.WorkerTest()
        worker_test.test_accuracy()

        # show 95 accuracy score threshold
        accuracy_cutoff = 0.95
        accuracy_file = deepnovo_config.accuracy_file
        score_cutoff = find_score_cutoff(accuracy_file, accuracy_cutoff)
    else:
        raise RuntimeError("unspecified mode")


if __name__ == '__main__':
    log_file_name = 'DeepNovo.log'
    d = {
        'version': 1,
        'disable_existing_loggers': False,  # this fixes the problem
        'formatters': {
            'standard': {
                'format': '%(asctime)s [%(levelname)s] %(name)s: %(message)s'
            },
        },
        'handlers': {
            'console': {
                'level': 'INFO',
                'class': 'logging.StreamHandler',
                'formatter': 'standard',
            },
            'file': {
                'level': 'DEBUG',
                'class': 'logging.FileHandler',
                'filename': log_file_name,
                'mode': 'w',
                'formatter': 'standard',
            }
        },
        'root': {
            'handlers': ['console', 'file'],
            'level': 'DEBUG',
        }
    }
    logging.config.dictConfig(d)
    main()
