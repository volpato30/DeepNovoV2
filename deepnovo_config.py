# DeepNovoV2 is publicly available for non-commercial uses.
# ==============================================================================

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import argparse

# ==============================================================================
# FLAGS (options) for this app
# ==============================================================================

parser = argparse.ArgumentParser()
parser.add_argument("--train_dir", type=str, default="train")
parser.add_argument("--beam_size", type=int, default="5")
parser.add_argument("--train", dest="train", action="store_true")
parser.add_argument("--search_denovo", dest="search_denovo", action="store_true")
parser.add_argument("--valid", dest="valid", action="store_true")
parser.add_argument("--test", dest="test", action="store_true")

parser.set_defaults(train=False)
parser.set_defaults(search_denovo=False)
parser.set_defaults(test=False)
parser.set_defaults(valid=False)

args = parser.parse_args()

train_dir = args.train_dir
use_lstm = True

# ==============================================================================
# GLOBAL VARIABLES for VOCABULARY
# ==============================================================================


# Special vocabulary symbols - we always put them at the start.
_PAD = "_PAD"
_GO = "_GO"
_EOS = "_EOS"
_START_VOCAB = [_PAD, _GO, _EOS]

PAD_ID = 0
GO_ID = 1
EOS_ID = 2
assert PAD_ID == 0
vocab_reverse = ['A',
                 'R',
                 'N',
                 'N(Deamidation)',
                 'D',
                 #~ 'C',
                 'C(Carbamidomethylation)',
                 'E',
                 'Q',
                 'Q(Deamidation)',
                 'G',
                 'H',
                 'I',
                 'L',
                 'K',
                 'M',
                 'M(Oxidation)',
                 'F',
                 'P',
                 'S',
                 'T',
                 'W',
                 'Y',
                 'V',
                ]

vocab_reverse = _START_VOCAB + vocab_reverse
print("vocab_reverse ", vocab_reverse)

vocab = dict([(x, y) for (y, x) in enumerate(vocab_reverse)])
print("vocab ", vocab)

vocab_size = len(vocab_reverse)
print("vocab_size ", vocab_size)


# ==============================================================================
# GLOBAL VARIABLES for THEORETICAL MASS
# ==============================================================================


mass_H = 1.0078
mass_H2O = 18.0106
mass_NH3 = 17.0265
mass_N_terminus = 1.0078
mass_C_terminus = 17.0027
mass_CO = 27.9949

mass_AA = {'_PAD': 0.0,
           '_GO': mass_N_terminus-mass_H,
           '_EOS': mass_C_terminus+mass_H,
           'A': 71.03711, # 0
           'R': 156.10111, # 1
           'N': 114.04293, # 2
           'N(Deamidation)': 115.02695,
           'D': 115.02694, # 3
           #~ 'C(Carbamidomethylation)': 103.00919, # 4
           'C(Carbamidomethylation)': 160.03065, # C(+57.02)
           #~ 'C(Carbamidomethylation)': 161.01919, # C(+58.01) # orbi
           'E': 129.04259, # 5
           'Q': 128.05858, # 6
           'Q(Deamidation)': 129.0426,
           'G': 57.02146, # 7
           'H': 137.05891, # 8
           'I': 113.08406, # 9
           'L': 113.08406, # 10
           'K': 128.09496, # 11
           'M': 131.04049, # 12
           'M(Oxidation)': 147.0354,
           'F': 147.06841, # 13
           'P': 97.05276, # 14
           'S': 87.03203, # 15
           'T': 101.04768, # 16
           'W': 186.07931, # 17
           'Y': 163.06333, # 18
           'V': 99.06841, # 19
          }

mass_ID = [mass_AA[vocab_reverse[x]] for x in range(vocab_size)]
mass_ID_np = np.array(mass_ID, dtype=np.float32)

mass_AA_min = mass_AA["G"] # 57.02146


# ==============================================================================
# GLOBAL VARIABLES for PRECISION, RESOLUTION, temp-Limits of MASS & LEN
# ==============================================================================


# if change, need to re-compile cython_speedup << NO NEED
#~ SPECTRUM_RESOLUTION = 10 # bins for 1.0 Da = precision 0.1 Da
#~ SPECTRUM_RESOLUTION = 20 # bins for 1.0 Da = precision 0.05 Da
#~ SPECTRUM_RESOLUTION = 40 # bins for 1.0 Da = precision 0.025 Da


# if change, need to re-compile cython_speedup << NO NEED
WINDOW_SIZE = 10 # 10 bins
print("WINDOW_SIZE ", WINDOW_SIZE)

MZ_MAX = 3000.0

MAX_NUM_PEAK = 500

KNAPSACK_AA_RESOLUTION = 10000 # 0.0001 Da
mass_AA_min_round = int(round(mass_AA_min * KNAPSACK_AA_RESOLUTION)) # 57.02146
KNAPSACK_MASS_PRECISION_TOLERANCE = 100 # 0.01 Da
num_position = 0

PRECURSOR_MASS_PRECISION_TOLERANCE = 0.01

# ONLY for accuracy evaluation
#~ PRECURSOR_MASS_PRECISION_INPUT_FILTER = 0.01
#~ PRECURSOR_MASS_PRECISION_INPUT_FILTER = 1000
AA_MATCH_PRECISION = 0.1

# skip (x > MZ_MAX,MAX_LEN)
MAX_LEN = 50 if args.search_denovo else 30
print("MAX_LEN ", MAX_LEN)


# ==============================================================================
# HYPER-PARAMETERS of the NEURAL NETWORKS
# ==============================================================================


num_ion = 12
print("num_ion ", num_ion)

weight_decay = 0.0  # no weight decay lead to better result.
print("weight_decay ", weight_decay)

#~ encoding_cnn_size = 4 * (RESOLUTION//10) # 4 # proportion to RESOLUTION
#~ encoding_cnn_filter = 4
#~ print("encoding_cnn_size ", encoding_cnn_size)
#~ print("encoding_cnn_filter ", encoding_cnn_filter)

embedding_size = 512
print("embedding_size ", embedding_size)

num_lstm_layers = 1
num_units = 64
lstm_hidden_units = 512
print("num_lstm_layers ", num_lstm_layers)
print("num_units ", num_units)

dropout_rate = 0.25

batch_size = 32
num_workers = 6
print("batch_size ", batch_size)

num_epoch = 20

init_lr = 1e-3

train_stack_size = 500 # 3000 # 5000
valid_stack_size = 1500#1000 # 3000 # 5000
test_stack_size = 5000
decode_stack_size = 1000 # 3000
print("train_stack_size ", train_stack_size)
print("valid_stack_size ", valid_stack_size)
print("test_stack_size ", test_stack_size)
print("decode_stack_size ", decode_stack_size)

steps_per_validation = 300  # 100 # 2 # 4 # 200
print("steps_per_validation ", steps_per_validation)

max_gradient_norm = 5.0
print("max_gradient_norm ", max_gradient_norm)


# ==============================================================================
# DATASETS
# ==============================================================================


knapsack_file = "knapsack.npy"
topk_output = 1
# training/testing/decoding files
input_spectrum_file_train = "ABRF_DDA/spectrums.mgf"
input_feature_file_train = "ABRF_DDA/features.csv.identified.train.nodup"
input_spectrum_file_valid = "ABRF_DDA/spectrums.mgf"
input_feature_file_valid = "ABRF_DDA/features.csv.identified.valid.nodup"
input_spectrum_file_test = "ABRF_DDA/spectrums.mgf"
input_feature_file_test = "ABRF_DDA/features.csv.identified.test.nodup"
# denovo files
denovo_input_spectrum_file = "ABRF_DDA/spectrums.mgf"
denovo_input_feature_file = "ABRF_DDA/features.csv.identified.test.nodup"

denovo_output_file = denovo_input_feature_file + ".deepnovo_denovo"

predicted_format = "deepnovo"
target_file = denovo_input_feature_file
predicted_file = denovo_output_file

accuracy_file = predicted_file + ".accuracy"
denovo_only_file = predicted_file + ".denovo_only"
scan2fea_file = predicted_file + ".scan2fea"
multifea_file = predicted_file + ".multifea"

# feature file column format
col_feature_id = "spec_group_id"
col_precursor_mz = "m/z"
col_precursor_charge = "z"
col_rt_mean = "rt_mean"
col_raw_sequence = "seq"
col_scan_list = "scans"
col_feature_area = "feature area"

# predicted file column format
pcol_feature_id = 0
pcol_feature_area = 1
pcol_sequence = 2
pcol_score = 3
pcol_position_score = 4
pcol_precursor_mz = 5
pcol_precursor_charge = 6
pcol_protein_id = 7
pcol_scan_list_middle = 8
pcol_scan_list_original = 9
pcol_score_max = 10


distance_scale_factor = 100.
sinusoid_base = 30000.
spectrum_reso = 10
n_position = int(MZ_MAX) * spectrum_reso
