from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sys

import numpy as np
cimport numpy as np
cimport cython

import deepnovo_config

mass_ID_np = deepnovo_config.mass_ID_np
cdef int GO_ID = deepnovo_config.GO_ID
cdef int EOS_ID = deepnovo_config.EOS_ID
cdef float mass_H2O = deepnovo_config.mass_H2O
cdef float mass_NH3 = deepnovo_config.mass_NH3
cdef float mass_H = deepnovo_config.mass_H
cdef float mass_CO = deepnovo_config.mass_CO
cdef int WINDOW_SIZE = deepnovo_config.WINDOW_SIZE
cdef int vocab_size = deepnovo_config.vocab_size
cdef int num_ion = deepnovo_config.num_ion


def get_sinusoid_encoding_table(n_position, embed_size, padding_idx=0):
    """ Sinusoid position encoding table
    n_position: maximum integer that the embedding op could receive
    embed_size: embed size
    return
      a embedding matrix of shape [n_position, embed_size]
    """

    def cal_angle(position, hid_idx):
        return position / np.power(deepnovo_config.sinusoid_base, 2 * (hid_idx // 2) / embed_size)

    def get_posi_angle_vec(position):
        return [cal_angle(position, hid_j) for hid_j in range(embed_size)]

    sinusoid_matrix = np.array([get_posi_angle_vec(pos_i) for pos_i in range(n_position + 1)], dtype=np.float32)

    sinusoid_matrix[:, 0::2] = np.sin(sinusoid_matrix[:, 0::2])  # dim 2i
    sinusoid_matrix[:, 1::2] = np.cos(sinusoid_matrix[:, 1::2])  # dim 2i+1

    sinusoid_matrix[padding_idx] = 0.
    return sinusoid_matrix

sinusoid_matrix = get_sinusoid_encoding_table(deepnovo_config.n_position, deepnovo_config.embedding_size,
                                              padding_idx=deepnovo_config.PAD_ID)

@cython.boundscheck(False) # turn off bounds-checking
@cython.wraparound(False) # turn off negative index wrapping
def get_ion_index(peptide_mass, prefix_mass, direction):
  """

  :param peptide_mass: neutral mass of a peptide
  :param prefix_mass:
  :param direction: 0 for forward, 1 for backward
  :return: an int32 ndarray of shape [26, 8], each element represent a index of the spectrum embbeding matrix. for out
  of bound position, the index is 0
  """
  if direction == 0:
    candidate_b_mass = prefix_mass + mass_ID_np
    candidate_y_mass = peptide_mass - candidate_b_mass
  elif direction == 1:
    candidate_y_mass = prefix_mass + mass_ID_np
    candidate_b_mass = peptide_mass - candidate_y_mass
  candidate_a_mass = candidate_b_mass - mass_CO

  # b-ions
  candidate_b_H2O = candidate_b_mass - mass_H2O
  candidate_b_NH3 = candidate_b_mass - mass_NH3
  candidate_b_plus2_charge1 = ((candidate_b_mass + 2 * mass_H) / 2
                               - mass_H)

  # a-ions
  candidate_a_H2O = candidate_a_mass - mass_H2O
  candidate_a_NH3 = candidate_a_mass - mass_NH3
  candidate_a_plus2_charge1 = ((candidate_a_mass + 2 * mass_H) / 2
                               - mass_H)

  # y-ions
  candidate_y_H2O = candidate_y_mass - mass_H2O
  candidate_y_NH3 = candidate_y_mass - mass_NH3
  candidate_y_plus2_charge1 = ((candidate_y_mass + 2 * mass_H) / 2
                               - mass_H)

  # ion_2
  #~   b_ions = [candidate_b_mass]
  #~   y_ions = [candidate_y_mass]
  #~   ion_mass_list = b_ions + y_ions

  # ion_8
  b_ions = [candidate_b_mass,
            candidate_b_H2O,
            candidate_b_NH3,
            candidate_b_plus2_charge1]
  y_ions = [candidate_y_mass,
            candidate_y_H2O,
            candidate_y_NH3,
            candidate_y_plus2_charge1]
  a_ions = [candidate_a_mass,
            candidate_a_H2O,
            candidate_a_NH3,
            candidate_a_plus2_charge1]
  ion_mass_list = b_ions + y_ions + a_ions
  ion_mass = np.array(ion_mass_list, dtype=np.float32)  # 8 by 26

  # ion locations
  # ion_location = np.ceil(ion_mass * SPECTRUM_RESOLUTION).astype(np.int64) # 8 by 26

  in_bound_mask = np.logical_and(
      ion_mass > 0,
      ion_mass <= deepnovo_config.MZ_MAX).astype(np.float32)
  ion_location = ion_mass * in_bound_mask  # 8 by 26, out of bound index would have value 0
  return ion_location.transpose()  # 26 by 8


def pad_to_length(data: list, length, pad_token=0.):
  """
  pad data to length if len(data) is smaller than length
  :param data:
  :param length:
  :param pad_token:
  :return:
  """
  for i in range(length - len(data)):
    data.append(pad_token)


def process_peaks(spectrum_mz_list, spectrum_intensity_list, peptide_mass):
  """

  :param spectrum_mz_list:
  :param spectrum_intensity_list:
  :param peptide_mass: peptide neutral mass
  :return:
    peak_location: int64, [N]
    peak_intensity: float32, [N]
    spectrum_representation: float32 [embedding_size]
  """
  charge = 1.0
  spectrum_intensity_max = np.max(spectrum_intensity_list)
  # charge 1 peptide location
  spectrum_mz_list.append(peptide_mass + charge*deepnovo_config.mass_H)
  spectrum_intensity_list.append(spectrum_intensity_max)

  # N-terminal, b-ion, peptide_mass_C
  # append N-terminal
  mass_N = deepnovo_config.mass_N_terminus - deepnovo_config.mass_H
  spectrum_mz_list.append(mass_N + charge*deepnovo_config.mass_H)
  spectrum_intensity_list.append(spectrum_intensity_max)
  # append peptide_mass_C
  mass_C = deepnovo_config.mass_C_terminus + deepnovo_config.mass_H
  peptide_mass_C = peptide_mass - mass_C
  spectrum_mz_list.append(peptide_mass_C + charge*deepnovo_config.mass_H)
  spectrum_intensity_list.append(spectrum_intensity_max)

  # C-terminal, y-ion, peptide_mass_N
  # append C-terminal
  mass_C = deepnovo_config.mass_C_terminus + deepnovo_config.mass_H
  spectrum_mz_list.append(mass_C + charge*deepnovo_config.mass_H)
  spectrum_intensity_list.append(spectrum_intensity_max)


  pad_to_length(spectrum_mz_list, deepnovo_config.MAX_NUM_PEAK)
  pad_to_length(spectrum_intensity_list, deepnovo_config.MAX_NUM_PEAK)

  spectrum_mz = np.array(spectrum_mz_list, dtype=np.float32)
  spectrum_mz_location = np.ceil(spectrum_mz * deepnovo_config.spectrum_reso).astype(np.int32)

  neutral_mass = spectrum_mz - charge*deepnovo_config.mass_H
  in_bound_mask = np.logical_and(neutral_mass > 0., neutral_mass < deepnovo_config.MZ_MAX)
  neutral_mass[~in_bound_mask] = 0.
  # intensity
  spectrum_intensity = np.array(spectrum_intensity_list, dtype=np.float32)
  norm_intensity = spectrum_intensity / spectrum_intensity_max

  spectrum_representation = np.zeros(deepnovo_config.embedding_size, dtype=np.float32)
  for i, loc in enumerate(spectrum_mz_location):
    if loc < 0.5 or loc > deepnovo_config.n_position:
      continue
    else:
      spectrum_representation += sinusoid_matrix[loc] * norm_intensity[i]

  top_N_indices = np.argpartition(norm_intensity, -deepnovo_config.MAX_NUM_PEAK)[-deepnovo_config.MAX_NUM_PEAK:]
  intensity = norm_intensity[top_N_indices]
  mass_location = neutral_mass[top_N_indices]

  return mass_location, intensity, spectrum_representation
