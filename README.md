# DeepNovo-pytorch

The DeepNovoV1 branch contains a pytorch re-implementation of [DeepNovo](https://github.com/nh2tran/DeepNovo)

The DeepNovoV2 branch contains the implementation of our proposed [DeepNovoV2](https://arxiv.org/abs/1904.08514) model.

## Dependency
python >= 3.6

pytorch >= 1.0

dataclasses

## data files

The ABRF DDA spectrums (the data for Table 1 in the original paper) and the default knapsack.npy file could be downloaded [here](https://drive.google.com/drive/folders/1sS9fTUjcwQukUVCXLzAUufbpR0UjJfSc?usp=sharing).
And the 9 species data could be downloaded [here](ftp://massive.ucsd.edu/MSV000081382/peak/DeepNovo/HighResolution/): ftp://massive.ucsd.edu/MSV000081382/peak/DeepNovo/HighResolution/. 

It is worth noting that
 in our implementation we represent training samples in a slightly different format (i.e. peptide stored in a csv file and spectrums stored in mgf files).
 We also include a script for converting the file format (data_format_converter.py in DeepNovoV2 branch).

## usage
first build cython modules

~~~
make build
~~~

train mode:

~~~
make train
~~~

denovo mode:

~~~
make denovo
~~~

evaluate denovo result:

~~~
make test
~~~




