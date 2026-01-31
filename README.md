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
This kernel provides a way to finetune and run inference in Int8 on Tensorcores for Ampere, Hopper and Blackwell using the algo found in the Sage Attention 3 paper.

3. attention_jvp.py
Sometimes users may use forward mode auto-differentiation for attention especially in Diffusion models that uses Flow Matching.
This kernel provides an implimentation of attention that outputs Jacobian Vector Products (JVP), for O matrix as tO (tangent O).
It can be used together with torch forward mode auto-differentiation.

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

Sage Attention 3 quantized Q, K, V inputs so that matmul happens in int8 using tensor cores with for forward and backward ops.  
  
Please refer to paper https://arxiv.org/pdf/2505.11594  
"SageAttention3: Microscaling FP4 Attention for Inference and An Exploration of 8-bit Training"  
by Jintao Zhang, Jia Wei, Haoxu Wang etc.  

# Fine Tuning
The Helion kernels can be further autotuned by changing the config 
```
@helion.kernel(
    autotune_effort="none",
    static_shapes=True,
)
```
removing autotune_effort="none" will allow Helion to autotune different shapes and sizes for your GPU and generate a better Triton Kernel.  
It is advised to autotuned the kernels to obtain the best config for the specific GPU arch.  
Then use the precompiled Triton Kernel for training/inference.
