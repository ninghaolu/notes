# Fine-tuning robot policies (VLA, WAMs): decisions to make before training
Note: Written with AI assistance.

This is a collection of questions / decisions I will pay attention to before launching robot policy finetuning. 

Running example: ABC-130K (YAM bimanual, LeRobot v3, 30 Hz) fine-tuned with pi0.5 (OpenPI), DreamZero, and OpenWAM-α. 

---

## 0. TL;DR checklist

| # | Decision | Write down |
|---|---|---|
| 1 | Dataset action semantics | Fields, dims, units, frame, absolute vs. delta, and whether actions are commands or measured states |
| 2 | Pretrained checkpoint action space | What the base model was trained to output, and the conversion from dataset to model space |
| 3 | Normalization | Stats type (mean/std, q01/q99, min/max), which representation they are computed on, where they are stored |
| 4 | Time | Action Hz, video FPS, horizon H, chunk span in seconds, number of chunks, deployment control rate |
| 5 | Loss | Which tensors receive loss, masking of padded dims and episode ends, loss weights |
| 6 | Budget | Global batch, steps, samples seen vs. available windows |

---

## 1. Understand the dataset's action format

- **Which fields?** In ABC: `observation.state` (14) and `action` (14).
- **Dim layout.** ABC: `[L joint 0-5, L gripper, R joint 7-12, R gripper]`, (grippers sit at indices 6 and 13).
- **Commanded or measured?** ABC `observation.state` is the measured joint position. `action` is the **commanded joint-position target**. If the state is 30° and the action is 35°, the controller is being told to move to 35°. That makes the action an absolute target, not a velocity or a delta.
- **Units and frames.** Joint angles are radians. EEF poses need position units (m/mm), a rotation format (quat order, euler, rot6d, axis-angle), and a reference frame (robot base, world, or per-arm base).
- **Gripper convention.** Range ([0,1], width in m, or binary) and which direction is open.


## 2. Match the pretrained checkpoint's action space and convert

The base checkpoint learned a specific output space. You either convert your data into that space, or knowingly re-learn the action head.

| Model on ABC | Model action space | Conversion from ABC joint14 (training → inference) |
|---|---|---|
| pi0.5 (OpenPI, YAM config) | 12 joint **deltas** + 2 **absolute** grippers, padded to 32-D | 1. For each 50-step chunk, subtract the current state on the joint dims: `delta[h] = action[t+h] − state[t]`. Grippers stay absolute (mask `[T]*6+[F]+[T]*6+[F]`).<br>2. Quantile-normalize (q01/q99 → [-1, 1]) using stats computed on these deltas.<br>3. Pad 14 → 32 dims.<br>4. At inference: un-normalize, add `state[t]` back on the joint dims, trim to 14 dims. |
| DreamZero (H48 relative recipe) | **Relative joint positions** (the default), all 14 dims, grippers included | 1. For each 48-step chunk, compute relative targets before normalization: `relative_action[k] = action[t₀+k] − observation.state[t₀]`, where `t₀` is that chunk's start (4 chunks per sample). On ABC, DreamZero therefore learns `action − observation.state`.<br>2. Example: targets `[35°, 38°, 40°, …]` with starting state 30° become `[+5°, +8°, +10°, …]`.<br>3. Normalize per dimension with training-set q01/q99 stats, mapping to [-1, 1] with clipping.<br>4. At inference: un-normalize, then add `state[t₀]` back. |
| OpenWAM-α | **Absolute end-effector poses** (the default): EEF20 = per arm xyz(3) + rot6d(6) + gripper(1), scattered into an 80-D unified space | 1. Convert ABC joint-position **commands** into absolute bimanual EEF targets with YAM forward kinematics. Per arm, 6 joint angles → FK → position (3D) + orientation as rot6d (6D). Tool = `tcp_site` + 0.1347 m along z, in each arm's base frame.<br>2. Keep the gripper command (1D) as is.<br>3. Min-max normalize the position and gripper channels; leave rot6d unnormalized.<br>4. Scatter the normalized EEF20 into its slots of the unified 80-D action vector. The other slots are zero and masked.<br>5. Convert measured states through the same FK and normalize them separately (their own stats) as proprio input.<br>6. At inference: gather the 20 active slots, un-normalize, then run IK to get joint targets for execution. |

Key points:
- **"Delta" is ambiguous.** It can mean chunk-relative (`a[t+h] - s[t]`, which pi0.5 and DreamZero use; OpenWAM is absolute), consecutive differences (`a[t+h] - a[t+h-1]`) ... They aren't interchangeable.
- **Subtract against which anchor?** pi0.5 uses the current state for the whole 50-step chunk. DreamZero uses each chunk's own start state.
- **Which dims stay absolute?** Grippers usually do, but DreamZero makes them relative.
- **Dimension padding.** Real dims get padded to the model width (pi0.5: 14 → 32, OpenWAM: 20 → 80). Record which slots are active and whether they are masked.
- **Inverse transform at inference** must exist and use the *same* state: un-normalize, add the state back, then trim padding (and run IK for EEF models). 
- If the dataset space differs from the pretrained space (e.g. joint vs. EEF), choose deliberately: convert the data (FK/IK) or accept that the action head is retrained from scratch.

## 3. Action normalization

For action normalization, we need to decide on 2 things: 1. what statistics to use (min, max, quantile ...)? 2. what representation it is computed on? The following are things we care about

- **Compute stats after the action transform.** For example, ABC's `meta/stats.json` holds absolute marginals, so it is not valid for delta targets. `q01(A−S)` can't be derived from the separate quantiles of A and S. Stats must come from the same pipeline as training.
- **Match the base model's convention**:
  - pi0.5: quantile, `(x - q01)/(q99 - q01)*2 - 1`.
  - DreamZero: q01/q99 to [-1,1] with clipping, pooled over all offsets and chunks.
  - OpenWAM: min/max to [-1,1] on position and gripper only (grippers pinned to [0,1], so closed −1, open +1). rot6d is left unnormalized. Actions and proprio states have separate stats.
- **Compute one set of normalization stats per action dimension.** For example in DreamZero finetuning, after converting actions to relative targets, compute per-dimension q01/q99 over all training chunks and all 48 steps within each chunk ([N, H, D] -> [N * H, D] and do normalization). Also we should match training loader's episode-end padding and masking behavior, since repeating or excluding padded actions changes the distribution used to compute the stats.


- **USe Sampling when needed.** A sampled pass (e.g. 1M frames) is fine for big datasets. Record that it was sampled. Beware hidden full-dataset costs: `randperm` over 382M rows, and decoding video when only numbers are needed.

- **Sometimes Stats travel with the checkpoint.** OpenPI copies them into `ckpt/assets/<asset_id>/`, and OpenWAM uses `normalization_stats.npy`. Inference must load *those* stats, not recompute them.


## 4. Frequency, horizon, and chunking

We need to decide on 3 things: 1. how much time each action step represents; 2. how often to sample video frames; 3. how far ahead the model predicts and how much data each training window contains. Converting steps and frames to seconds makes these choices easier to compare with pretraining.

| | Action Hz | Video FPS | Horizon | Time Span per chunk | Window |
|---|---|---|---|---|---|
| ABC dataset | 30 | 30 (source) | – | – | – |
| pi0.5 on ABC | 30 | current frame only | H=50 | 1.67 s | 1 chunk |
| DreamZero, AgiBot pretrain | 30 | 5 | H=48 | 1.6 s | – |
| **DreamZero on ABC** | 30 | 5 (stride 6) | H=48 | 1.6 s | 4 chunks = 192 actions + 33 frames, 6.4 s, 193 rows |
| OpenWAM on ABC | 30 | 7.5 (stride 4) | H=32  | 1.07 s | 9 frames, ~1.07 s |

Decisions:
- **How much time does one action step represent?** Keep ABC actions at 30 Hz, so each step represents about 33 ms. Changing the action rate also changes how much time a chunk covers: 48 steps cover 1.6 s at 30 Hz, but 3.2 s at 15 Hz. If we resample actions, deployment must execute the predicted steps at the corresponding rate.
- **How often do we sample video frames?** For DreamZero, take every sixth frame from ABC’s 30 FPS video to get 5 FPS, matching AgiBot pretraining. Each video-frame interval then corresponds to 6 action steps. For OpenWAM, take every second frame to get 15 FPS, giving 2 action steps per video-frame interval. The loader must keep frames and actions aligned and group them according to the model’s temporal compression.
- **How far ahead does the model predict, and how much data does training read?** These are different lengths. For DreamZero, each chunk contains 48 actions, covering 1.6 s. A training window contains 4 chunks, covering 6.4 s. At 5 FPS, that window contains 33 video frames including both endpoints; at the source rate of 30 Hz, it spans 193 rows. This keeps the chunk duration and action-to-video ratio consistent with AgiBot pretraining.


## 5. How the loss is computed

We should also understand what action loss is computed on. For example, 

- **pi0.5:** flow matching on the normalized, padded action chunk `(B, 50, 32)`. The MSE is taken between the predicted and target velocity. Padded dims have target 0. Check whether your fork masks padded dims and episode-end padding (`action_is_pad`) or just trains on them.
- **DreamZero / OpenWAM (WAMs):** **video latent flow loss + action flow loss** (OpenWAM uses weight 1 each, independent timesteps, sigma shift 5 for video and 1 for action). Inactive action slots are masked using the dim mask.

Therefore, it is important to check these
  - Is the loss on the normalized target, and is that target in the space from §2?
  - Are padded dims, padded timesteps, and inactive unified-space slots excluded or zero-targeted?
  - What is the relative weight of video vs. action loss since a good video loss can hide a poor action loss so we should look at them separately.
  - Which components are trainable (full model, action expert only, VAE/text encoder frozen)?


## 6. Training budget and data coverage

We need to decide how many training samples to draw. Each sample starts from an anchor and contains an action chunk or a longer training window.

First, compute the effective global batch size:

$$
B_{\text{global}}
=
B_{\text{per GPU}}
\times \text{gradient accumulation steps}
\times \text{number of GPUs}
$$

$$
N_{\text{seen}}
=
N_{\text{optimizer steps}}
\times B_{\text{global}}
$$

Count the available windows using the loader’s sampling rules. Without padding, an episode with $L_e$ rows provides $\max(0, L_e-W+1)$ windows of length $W$:

$$
N_{\text{windows}}
=
\sum_{e \in \text{episodes}}
\max(0, L_e-W+1)
$$

With end padding, every row may be a valid anchor, so $N_{\text{windows}}$ equals the total row count if no other filtering or starting-position restrictions apply.

The sampling ratio and required step count are:

$$
r = \frac{N_{\text{seen}}}{N_{\text{windows}}},
\qquad
N_{\text{steps}}
=
\left\lceil
\frac{r \times N_{\text{windows}}}{B_{\text{global}}}
\right\rceil
$$

This ratio measures samples drawn relative to available windows, not distinct windows visited, since sampling may repeat windows.

For ABC, assuming all 382,468,339 rows are eligible anchors, a 2% sampling ratio with global batch size 256 requires:

$$
N_{\text{steps}}
=
\left\lceil
\frac{0.02 \times 382{,}468{,}339}{256}
\right\rceil
=
29{,}881
\approx 30\text{k steps}
$$

## References

- **DreamZero — World Action Models are Zero-shot Policies.** Seonghyeon Ye et al., 2026. [Paper](https://arxiv.org/abs/2602.15922)
- **π₀.₅: a Vision-Language-Action Model with Open-World Generalization.** Kevin Black et al., 2025. [Paper](https://www.physicalintelligence.company/download/pi05.pdf)
- **OpenWAM — An Open, Modular Exploration Towards Systematic World–Action Model Pretraining (OpenWAM-α).** Yuran Wang et al., 2026. [Paper](https://arxiv.org/abs/2609.07398)
