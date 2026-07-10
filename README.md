# Zero-Copy-Buffer Engine

A bare-metal, ultra-high-throughput Shared Memory Ring Buffer written in **Rust** and packaged as a native Python C-Extension using **Maturin**.

## Motivation

Python multiprocessing RL pipelines pay a tax on every step: transitions are pickled, sent across a process boundary, and unpickled on the other side. This serialization cost compounds as simulators produce thousands of steps per second and the number of parallel workers grows — inter-process communication, rather than compute, becomes the ceiling on throughput.

`zero_copy_buffer` targets that bottleneck. A Rust core owns a single shared-memory region that simulation workers and training processes access directly, avoiding the pickle-and-pipe round trip. Transitions are stored as raw, layout-stable bytes, so a consumer can map a tensor straight onto the buffer and read it without deserialization.

The goal is a buffer built for throughput: data moves through shared memory instead of a serialization pipeline, freeing the training loop to spend its time learning rather than waiting on IPC.

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
