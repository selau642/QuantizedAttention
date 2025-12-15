#Attention in BrainFloat16(BF16)
Self Attention mechanism in BF16 format suffers from overflow and underflow numerical issues. This happens when the logits of attention dot product, QK, has multiple values that are very close to the max row of QK. This causes  QK - rowmax(QK) to be close to zero and exp(QK - rowmax(QK)) to be close to 1. After many rounds of training, the floating point error will result in explosion of gradient maxtrix and training will diverge.

Please refer to paper https://arxiv.org/abs/2510.04212
"Why Low-Precision Transformer Training Fails: An Analysis of Flash Attention" by Haiquan Qiu, Quanming Yao 

This repo is a reimplimentation of the following repo into Helion
https://github.com/ucker/why-low-precision-training-fails

#INSTALL
```
uv init .
uv lock
uv sync
```

Pytorch 2.9.1
Helion 0.2.7
