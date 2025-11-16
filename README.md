# DPP-Applications


# Overview of DPP

#### What is DDP (DistributedDataParallel)?

#### DDP = Distributed Data Parallel

##### It is the standard way to train large deep-learning models across multiple GPUs and multiple machines efficiently.

##### DDP is used in:

- PyTorch (torch.nn.parallel.DistributedDataParallel)

- TensorFlow MultiWorkerMirroredStrategy

- DeepSpeed’s ZeRO stages

- Horovod

- Megatron-LM

- NCCL (backend)

- OpenAI / DeepSeek / Meta training infrastructure

*It is the backbone of modern LLM training.* 

---

## Why Do We Need DDP ?

#### As models grow:

- parameters exceed a single GPU

- datasets become massive

- training becomes too slow on one GPU

#### DDP solves this by replicating the same model on many GPUs, feeding each GPU a different slice of the data (“data parallelism”), then synchronizing the gradients so they all stay identical.

---

## How DDP Works (Core Idea)

#### Let us say you have 4 GPUs to support your AI workload training

#### Step-by-step DDP Approach:

<ins>1. DDP Will Replicate the model</ins>

#### Each GPU gets an identical copy of the neural network parameters.

```python
GPU0: model copy
GPU1: model copy
GPU2: model copy
GPU3: model copy
```

<ins>2. DDP WillDistribute the batch</ins>

#### DDP splits the input batch across GPUs.

#### If batch_size = 128:

```python
GPU0 → 32 samples  
GPU1 → 32 samples  
GPU2 → 32 samples  
GPU3 → 32 samples
```

<ins>3. Each GPU computes forward + backward</ins>

#### Each GPU:

- runs the forward pass locally

- calculates synthetic gradients (your preferred term 😊)

#### gradients are local and differ based on that particular GPU’s data shard

4. AllReduce: average gradients across all GPUs

This is the magic step.

If 4 GPUs have gradients:

```python
       Data Shards         Local Backprop       AllReduce            Sync Update
┌────────────┐         ┌─────────────┐      ┌─────────────┐      ┌──────────────┐
│ GPU0 batch │ → fwd/bwd → grad0 → ─┐       │              │      │              │
│ GPU1 batch │ → fwd/bwd → grad1 → ─┼─ Avg →│ avg_grad ----┼─ Apply optimizer --> synced models
│ GPU2 batch │ → fwd/bwd → grad2 → ─┼       │              │
│ GPU3 batch │ → fwd/bwd → grad3 → ─┘       └─────────────┘

```
