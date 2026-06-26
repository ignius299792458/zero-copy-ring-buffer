# Zero-Copy-Buffer Engine

A bare-metal, ultra-high-throughput Shared Memory Ring Buffer written in **Rust** and packaged as a native Python C-Extension using **Maturin**.

## Architecture Overview

This engine acts as a zero-serialization, zero-copy data highway designed (explicitly) for deep reinforcement learning pipelines. It decouples environment simulators from machine learning frameworks, allowing them to execute concurrently across isolated OS processes without hitting the Python Global Interpreter Lock (GIL) bottleneck.

**Ring Buffer (`zero-copy_buffer` Core):** A native Rust library that handles low-level virtual memory allocation, shared memory handles, and lock-free thread coordination using hardware-level atomics.

---

## Can be applied as :

1. **Data Producers (`sim_envs`):** Multiple independent Python processes running parallel simulation environments (via Gymnasium `AsyncVectorEnv`). They compute physics steps simultaneously and write raw data directly into the buffer space using native pointers.
2. **Data Consumer (`ml_models`):** An isolated PyTorch training process that maps tensors directly over the raw memory space using `torch.from_blob()`, sampling batches with absolutely zero memory copying overhead.

```
┌────────────────────────────────────────────────────────────────────────┐
│                        1. MASTER ORCHESTRATOR                          │
│                              (Python)                                  │
└───────────────────────────────────┬────────────────────────────────────┘
                                    │ Launches & Passes Handles
            ┌───────────────────────┴───────────────────────┐
            ▼                                               ▼
┌──────────────────────────────────────┐        ┌──────────────────────────────────────┐
│        2. SIMULATION LAYER           │        │           3. ML MODEL LAYER          │
│    (10x Parallel Python Envs)        │        │      (1x PyTorch Model Process)      │
│  env.step() ──► Write to Pointer     │        │  torch.from_blob() ──► Direct Read   │
└───────────────────┬──────────────────┘        └───────────────────┬──────────────────┘
                    │                                               │
                    └───────────────┐               ┌───────────────┘
                                    ▼               ▼
┌──────────────────────────────────────────────────────────────────────────────────────┐
│                              4. BARE-METAL STORAGE (RING) BUFFER                     │
│                                      (Rust Core)                                     │
│ [ Slot 0 ] [ Slot 1 ] [ Slot 2 ] [ Slot 3 ] [ Slot 4 ] [ Slot 5 ] [ Slot 6 ] ...     │
└──────────────────────────────────────────────────────────────────────────────────────┘

```
