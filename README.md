# Quantized Inference and Training for Attention 

Naively quantizing transformers in lower floating point precision is faster but hard to get right.
Many attempts result in exploding gradients for training/finetuning or drop in accuracy during inference.

Recently, in 2025, two main techniques have appeared to allow proper quantization without loss of accuracy or exploding gradients.

This repo implements these two techniques using Helion Lang, a new Python Domain Specific Language that compiles into OpenAI Triton Lang.
  
The two kernels with backward and forward are in:  
  
1. attention_bf16.py  
This kernel provides a way to train and run inference in bfloat16 which corrects for bfloat16 accumulation errors. 
Users may need to remove the configs to allow Helion to optimize the tiling etc for your specific hardward.  

2. attention_int8.py   
(Work in Progress)  
This kernel provides a way to finetune and run inference in Int8 on Blackwell/Hopper using the Sage Attention 3 paper.
  
# INSTALL

```
uv init .
uv lock
uv sync
```
  
Pytorch 2.9.1  
Helion 0.2.7  
  
# Attention in BrainFloat16(BF16)

Self Attention mechanism in BF16 format suffers from overflow and underflow numerical issues. This happens when the logits of attention dot product, QK, has multiple values that are very close to the max row of QK. This causes  QK - rowmax(QK) to be close to zero and exp(QK - rowmax(QK)) to be close to 1. After many rounds of training, the floating point error will result in explosion of gradient maxtrix and training will diverge.  
  
Please refer to paper https://arxiv.org/abs/2510.04212  
"Why Low-Precision Transformer Training Fails: An Analysis of Flash Attention" by Haiquan Qiu, Quanming Yao  

This repo is a re-implementation of the following repo into Helion  
https://github.com/ucker/why-low-precision-training-fails  

Original authors use torch.amp.autocast  
this kernels provide more granular control for inputs and outputs to test different bfloat16 casting issues


# Attention in Int8

Sage Attention 3 proposes using dynamic quantization to quantize Q, K, V such that matmul happens in int8 without minimal impact on forward and backward outputs.  
  
Please refer to paper https://arxiv.org/pdf/2505.11594  
"SageAttention3: Microscaling FP4 Attention for Inference and An Exploration of 8-bit Training"  
by Jintao Zhang, Jia Wei, Haoxu Wang etc.  

Currently this is still in development and is not stable.