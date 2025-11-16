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

```python
       Data Shards         Local Backprop       AllReduce            Sync Update
┌────────────┐         ┌─────────────┐      ┌─────────────┐      ┌──────────────┐
│ GPU0 batch │ → fwd/bwd → grad0 → ─┐       │              │      │              │
│ GPU1 batch │ → fwd/bwd → grad1 → ─┼─ Avg →│ avg_grad ----┼─ Apply optimizer --> synced models
│ GPU2 batch │ → fwd/bwd → grad2 → ─┼       │              │
│ GPU3 batch │ → fwd/bwd → grad3 → ─┘       └─────────────┘

```
