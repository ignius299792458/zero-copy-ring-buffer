"""
Phase 1 pipeline — single process, no concurrency.

    SimEnv (sim_envs)  -->  SimpleBuffer (Rust/_core)  -->  ModelReader (ml_models)

Poetry path dependencies handle all imports — no sys.path hacks needed.
Run from the project root after: maturin develop
"""

from zero_copy_buffer import SimpleBuffer
from sim_envs import SimEnv
from ml_models import ModelReader

# ── config ────────────────────────────────────────────────────────────────────
BUFFER_CAPACITY = 512   # total slots; overwrite from start when full
COLLECT_STEPS   = 128   # transitions per collection round
BATCH_SIZE      = 64    # transitions per model read
N_ITERATIONS    = 10    # collect → read cycles
SEED            = 42
# ─────────────────────────────────────────────────────────────────────────────


def main() -> None:
    print("=" * 60)
    print("zero_copy_buffer  |  phase 1  |  single-process pipeline")
    print("=" * 60)
    print(f"  buffer capacity : {BUFFER_CAPACITY}")
    print(f"  collect steps   : {COLLECT_STEPS}")
    print(f"  batch size      : {BATCH_SIZE}")
    print(f"  iterations      : {N_ITERATIONS}")
    print()

    buf    = SimpleBuffer(BUFFER_CAPACITY)
    sim    = SimEnv(buffer=buf, seed=SEED)
    reader = ModelReader(buffer=buf, batch_size=BATCH_SIZE)

    for it in range(1, N_ITERATIONS + 1):

        # ── collect ──────────────────────────────────────────────────────────
        written = sim.collect(COLLECT_STEPS)

        # ── read + forward ───────────────────────────────────────────────────
        result = reader.read_and_forward()

        if result is None:
            print(
                f"  iter {it:>2} | wrote {written:>3} | "
                f"buf {buf.len:>3}/{buf.capacity} | "
                f"skipped — not enough data yet"
            )
            continue

        obs0    = result["obs"][0].tolist()
        logits0 = result["logits"][0].tolist()
        print(
            f"  iter {it:>2} | "
            f"wrote {written:>3} | "
            f"buf {buf.len:>3}/{buf.capacity} full={buf.is_full} | "
            f"read {result['n']:>2} | "
            f"obs[0]=[{obs0[0]:+.3f} {obs0[1]:+.3f} "
            f"{obs0[2]:+.3f} {obs0[3]:+.3f}] "
            f"logits[0]=[{logits0[0]:+.3f} {logits0[1]:+.3f}]"
        )

    sim.close()
    print()
    print("phase 1 complete — sim_envs → rust buffer → ml_models flowing.")
    print(f"  {sim}")
    print(f"  {reader}")


if __name__ == "__main__":
    main()