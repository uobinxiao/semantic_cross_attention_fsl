# semantic_cross_attention_fsl
Implementation of paper https://arxiv.org/abs/2210.06311

config-defaults.yaml describes the parameters used by the model. Both train.py and mmtrain.py read this file for different model configurations.

To train benchmark models, including ProtoNet and ProxyNet, change the values in the config file, then use python train.py.

Simiarly, to train the proposed model, simple use python mmtrain.py
