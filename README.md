# Notes

Personal learning and practical notes on language models, video models, and robot policies, focusing on key decisions and implementation details that matter when working on a task.

## Post-training

### Robot Policy Fine-tuning

- **[Fine-tuning robot policies: decisions to make before training](post-training/policy-finetuning.md)** — A decision checklist covering action semantics, checkpoint compatibility, normalization, temporal alignment, loss masking, and training budget. Uses [ABC-130K](https://abc.bot/) with [pi0.5](https://www.pi.website/blog/pi05), [DreamZero](https://github.com/dreamzero0/dreamzero), and [OpenWAM-α](https://github.com/OpenWAM-Official/OpenWAM) as running examples.

### Video Model Distillation

- **[DMD training notes for video model distillation](post-training/dmd.md)** — Learning notes and an implementation walkthrough covering [self-forcing](https://self-forcing.github.io/) rollout, the roles of the teacher and critic, the DMD objective and pseudo-MSE, and rolling KV caches.
