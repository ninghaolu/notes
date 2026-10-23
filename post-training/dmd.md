# DMD Training Notes for World Model Distillation

## Overview

This note explains the distillation pipeline in `world-model-distillation`

The repository trains a few-step autoregressive video diffusion world model using:

- **Self-forcing rollout**: the generator produces its own autoregressive training samples, matching inference-time conditions.
- **Distribution Matching Distillation (DMD)**: the generator is updated using a KL-gradient estimate between generated-video distribution and data-video distribution.
- **Frozen teacher / real score model**: estimates the clean target for the real data distribution.
- **Trainable critic / fake score model**: estimates the clean target for the current generator distribution.
- **Rolling external KV cache**: limits temporal attention context during rollout so training matches long-horizon inference.

The overall pipeline for this distribution matching distillation process is: 
## DMD Training Pipeline

1. **Generator rollout**

   The generator first rolls out a video latent:

   $$ \hat{x} = G_\theta(z, c) $$

2. **Add diffusion noise**

   DMD samples a diffusion timestep $t$ and adds noise to the generated latent:

   $$ x_t = q_t(\hat{x}) $$

3. **Teacher denoising: real-data direction**

   The frozen teacher model denoises $x_t$ and predicts the clean latent under the real data distribution:

   $$ x^{\mathrm{real}}_0 = f_\phi(x_t, t, c) $$

4. **Critic denoising: fake-distribution direction**

   The critic denoises the same $x_t$ and predicts the clean latent under the current generator distribution:

   $$ x^{\mathrm{fake}}_0 = f_\psi(x_t, t, c) $$

5. **Construct the DMD gradient**

   The difference between the fake and real clean predictions gives the gradient that should be applied to the generator output:

   $$ g = x^{\mathrm{fake}}_0 - x^{\mathrm{real}}_0 $$

6. **Update the generator with pseudo-MSE**

   A pseudo-MSE loss is constructed so that its gradient with respect to $\hat{x}$ is exactly $g$:

   $$ L_{\mathrm{DMD}} = \frac{1}{2}\|\hat{x} - \mathrm{sg}[\hat{x} - g]\|^2 $$

   Therefore,

   $$ \frac{\partial L_{\mathrm{DMD}}}{\partial \hat{x}} = g $$

   Since $\hat{x}=G_\theta(z,c)$, this gradient is backpropagated through $\hat{x}$ to update the generator parameters $\theta$.


## 1 Background

The pretrained world model is a multi-step diffusion DiT. It can generate action-conditioned video, but inference is expensive because each frame or chunk requires many denoising steps.

The goal of this repo is to train a **few-step generator**:

- Start from the pretrained world model weights.
- Generate future latent frames autoregressively.
- Reduce the number of denoising steps per frame or chunk.
- Keep generation stable when the model conditions on its own previous outputs.
- Train on the same type of self-generated trajectories used at inference.

The key train-test gap is:

- During standard teacher-forced training, the model conditions on real previous frames.
- During inference, the model conditions on its own generated previous frames.
- Errors can accumulate because the model sees a different context distribution at inference.

Self-forcing closes this gap by rolling out the generator during training and applying the DMD objective to the generated trajectory.

## 2 Notation / Definitions

Let:

- $x^{1:N} = (x^1, x^2, ..., x^N)$: a sequence of video frames.
- $x^i$: frame or chunk at autoregressive position $i$.
- $x^{\lt i}$: all previous frames or chunks before $i$.
- $a^{1:N}$: action sequence conditioning the world model.
- $p_{\theta}$: autoregressive generator distribution.
- $\theta$: generator parameters.
- $p_{\mathrm{data}}$: real video data distribution.
- $\hat{x}^{1:N}$: generated video sequence sampled from the generator.
- $q_t(x_t \mid x_0)$: forward diffusion noising process at timestep $t$.
- $\bar{\alpha}_t$: cumulative product of diffusion alphas at timestep $t$.
- $\epsilon \sim \mathcal{N}(0, I)$: Gaussian noise.
- $s(x) = \nabla_x \log p(x)$: score function.
- $s_{\mathrm{real}}$: score of the noised real-data distribution.
- $s_{\mathrm{fake}}$: score of the noised generator distribution.
- $f_{\phi}$: frozen teacher / real score model.
- $f_{\psi}$: trainable critic / fake score model.

The autoregressive generator is:

$$ p_\theta(x^{1:N}) = \prod_{i=1}^{N} p_\theta(x^i \mid x^{\lt i}, a^{\le i}) $$

At inference, each frame or chunk is generated from Gaussian noise, conditioned on previous generated frames or chunks.

During training, the target is to match the full video distribution:

$$ p_\theta(x^{1:N}) \approx p_{\mathrm{data}}(x^{1:N}) $$

The generator samples its own rollout:

$$ \hat{x}^{1:N} \sim p_\theta(x^{1:N}) $$

This matters because the generator is trained on samples from the same distribution it will encounter at inference time.

## 3 Training / Algorithm Pipeline

### 3.1 Launch Framewise DMD Training

Source: `scripts/train_dmd_framewise.bash`, lines 27-39.

```bash
sbatch scripts/train_dmd_framewise.bash
# Internally runs: torchrun --nproc_per_node=2 train_dmd.py ...
```

Important training settings from the script:

- `n_total_frames=49`: dataset returns 49 real frames and actions.
- `n_output_frames=13`: only the last 13 generated frames are scored by DMD.
- `max_cache_frames=13`: temporal cache holds only 13 frames, so older context is evicted.
- `num_frame_per_block=1`: framewise autoregressive rollout.
- `pretrained_checkpoint`: initializes teacher, generator, and critic unless separate checkpoints are provided.

The main implementation is in:

- `train_dmd.py`
- `dmd/generation.py`
- `dmd/losses.py`
- `dmd/cache.py`
- `worldmodel/model_with_kv_cache.py`

### 3.2 Build Dataset and Encode Initial Frame

`AggregatedDataset` returns:

- `observation`: `[B, n_total_frames, H_img, W_img, 3]`
- `action`: `[B, n_total_frames, D_action]`

Source: `train_dmd.py`, lines 496-504. The first real frame is encoded by the frozen VAE:

```python
x = batch["observation"].to(device, non_blocking=True)
actions = batch["action"].to(device, non_blocking=True)

with torch.no_grad():
    with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        initial_latent = vae.encode(x[:, :1])

initial_action = actions[:, :1]
```

Shapes:

- `x[:, :1]`: `[B, 1, H_img, W_img, 3]`
- `initial_latent`: `[B, 1, H, W, C]`
- `initial_action`: `[B, 1, D_action]`

The first frame is real context. The generator produces future latent frames conditioned on this initial latent and the action sequence.

### 3.3 Initialize Teacher, Generator, and Critic

Source: `train_dmd.py`, lines 277-333. The training code builds three DiT models:

```python
generator = make_dit(student_model_dim, student_layers, student_heads)
real_score = make_dit(teacher_model_dim, teacher_layers, teacher_heads)
fake_score = make_dit(student_model_dim, student_layers, student_heads)
```

Their roles are:

- `generator`: trainable few-step autoregressive generator $p_\theta$.
- `real_score`: frozen teacher $f_\phi$, used to estimate the real-data score.
- `fake_score`: trainable critic $f_\psi$, used to estimate the current generator distribution score.

The teacher is frozen:

```python
requires_grad(real_score, False)
real_score.eval()
```

Source: `train_dmd.py`, lines 377-384. The generator and critic are optimized separately:

```python
gen_opt = optim.AdamW(generator.parameters(), lr=lr_generator, ...)
critic_opt = optim.AdamW(fake_score.parameters(), lr=lr_critic, ...)
```

The training loop updates:

- generator every `dfake_gen_update_ratio` steps,
- critic every step.

Source: `train_dmd.py`, line 482:

```python
train_gen = (train_steps % dfake_gen_update_ratio == 0)
```

### 3.4 Self-Forcing Generator Rollout

Source: `train_dmd.py`, lines 511-529. The core rollout call and output-action slice are:

```python
generated = self_forcing_rollout(
    generator=generator,
    initial_latent=initial_latent,
    initial_action=initial_action,
    actions=actions,
    denoising_step_list=denoising_step_list,
    alphas_cumprod=alphas_cumprod,
    n_total_frames=n_total_frames,
    n_output_frames=n_output_frames,
    max_cache_frames=max_cache_frames,
    grad_frames_from=grad_frames_from,
    num_frame_per_block=num_frame_per_block,
)
output_actions = actions[:, n_total_frames - n_output_frames:]
```

For the framewise script:

- The real first frame is encoded and written to the cache.
- Frames `1..48` are generated autoregressively.
- Only frames `36..48` are returned because `n_total_frames - n_output_frames = 36`.
- The DMD loss sees only the returned 13 generated latent frames.

The rollout returns `generated` with shape `[B, n_output_frames, H, W, C]`. The matching `output_actions` tensor has shape `[B, 13, D_action]`.

### 3.5 Add Diffusion Noise to Generated Rollout

The generator output is treated as clean latent $x_0^{(\theta)}$. The loss samples a diffusion timestep and adds noise:

$$ x_t = \sqrt{\bar{\alpha}_t} x_0^{(\theta)} + \sqrt{1 - \bar{\alpha}_t}\epsilon $$

Source: `dmd/losses.py`, lines 175-192, and `dmd/diffusion_utils.py`, lines 93-119:

```python
timestep_per_sample = torch.randint(min_timestep, max_timestep, (B,), ...)
timestep = timestep_per_sample.unsqueeze(1).expand(B, T)
noise = torch.randn_like(generated_video_detached)
noisy = q_sample(generated_video_detached, noise, alphas_cumprod, timestep)
```

Shapes:

- `generated_video`: `[B, T, H, W, C]`
- `timestep_per_sample`: `[B]`
- `timestep`: `[B, T]`
- `noise`: `[B, T, H, W, C]`
- `noisy`: `[B, T, H, W, C]`

The same timestep is broadcast across all scored frames for each sample.

### 3.6 Teacher and Critic Predict Clean Latents

Source: `dmd/losses.py`, lines 101-111. Both score models receive the noised generated video:

```python
x0_fake = run_score_model(fake_score, noisy, timestep, action, ...)
x0_real = run_score_model(real_score, noisy, timestep, action, ...)
```

Both models are v-prediction DiTs. They output $v$, which is converted to $x_0$:

$$ \hat{x}_0 = \sqrt{\bar{\alpha}_t}x_t - \sqrt{1 - \bar{\alpha}_t}v_\theta(x_t,t,a) $$

Source: `dmd/losses.py`, lines 50-57, and `dmd/diffusion_utils.py`, lines 61-90:

```python
v_cond = model(noisy_input, timestep, action)
x0_cond = v_pred_to_x0(v_cond, noisy_input, alphas_cumprod, timestep)
```

Source: `dmd/losses.py`, lines 53-57. Classifier-free guidance can be applied to the teacher:

```python
x0_pred = x0_uncond + cfg_scale * (x0_cond - x0_uncond)
```

Source: `train_dmd.py`, lines 159-160. In the default `train_dmd.py` arguments:

- `real_guidance_scale=3.0`
- `fake_guidance_scale=0.0`

The framewise script does not override these values.

### 3.7 Compute DMD Generator Loss

Source: `dmd/losses.py`, lines 101-114. The DMD gradient estimate is:

```python
grad = x0_fake - x0_real
```

Source: `dmd/losses.py`, lines 207-210. Then the generator is trained with a pseudo-MSE target:

```python
target = (generated_video_detached - grad).detach()
loss = 0.5 * F.mse_loss(generated_video, target)
```

Only `generated_video` carries generator gradients. The teacher, critic, noise process, and target are detached during the generator update.

### 3.8 Train the Critic

Source: `train_dmd.py`, lines 574-604. The critic is trained on detached generator rollouts:

```python
with torch.no_grad():
    gen_for_critic = self_forcing_rollout(..., grad_frames_from=None)

critic_loss, critic_log = compute_critic_loss(
    generated_video=gen_for_critic.detach(),
    actions=output_actions_critic,
    fake_score=fake_score,
    alphas_cumprod=alphas_cumprod,
)
```

The critic loss is a standard v-prediction denoising loss:

$$ L_{\mathrm{critic}} = \|v_\psi(x_t,t,a) - (\sqrt{\bar{\alpha}_t}\epsilon - \sqrt{1-\bar{\alpha}_t}x_0^{(\theta)})\|_2^2 $$

This trains `fake_score` to estimate the noised distribution of the generator's current samples.

## 4 Objective / Loss

### 4.1 Problem Formulation

We consider a sequence of video frames:

$$ x^{1:N} = (x^1, x^2, ..., x^N) $$

The goal is to train an autoregressive video generator parameterized by $\theta$:

$$ p_\theta(x^{1:N}) = \prod_{i=1}^{N} p_\theta(x^i \mid x^{\lt i}) $$

During inference, each frame or chunk of frames is generated starting from Gaussian noise conditioned on previously generated frames or chunks.

During training, the goal is to match the whole video distribution:

$$ p_\theta(x^{1:N}) \approx p_{\mathrm{data}}(x^{1:N}) $$

The generated samples come from autoregressive self-rollout:

$$ \hat{x}^{1:N} \sim p_\theta(x^{1:N}) $$

This ensures that training samples are drawn from the same type of distribution encountered at inference time.

### 4.2 DMD Reverse KL Objective

Let:

- $p_{\theta,t}$: generator distribution after adding diffusion noise at timestep $t$.
- $p_{\mathrm{data},t}$: real data distribution after adding diffusion noise at timestep $t$.
- $\hat{x} \sim p_\theta(x^{1:N})$: clean generated rollout.
- $\hat{x}_t \sim q_t(\hat{x}_t \mid \hat{x})$: noised generated rollout.
- $s_{\mathrm{real}}(\hat{x}_t,t)$: score of $p_{\mathrm{data},t}$ at $\hat{x}_t$.
- $s_{\mathrm{fake}}(\hat{x}_t,t)$: score of $p_{\theta,t}$ at $\hat{x}_t$.

The DMD objective minimizes reverse KL over noised distributions:

$$ \mathcal{L}_{\mathrm{DMD}}(\theta) = \mathbb{E}_t[D_{\mathrm{KL}}(p_{\theta,t} \| p_{\mathrm{data},t})] $$

#### Derivation: from reverse KL to the generator gradient

**Step 1 — Expand the KL as an expectation.**

Fix a timestep $t$. Write the KL divergence as an expectation under the generator's own noised distribution:

$$ D_{\mathrm{KL}}(p_{\theta,t}\|p_{\mathrm{data},t}) = \mathbb{E}_{x_t\sim p_{\theta,t}}[\log p_{\theta,t}(x_t) - \log p_{\mathrm{data},t}(x_t)] $$

**Step 2 — Reparameterize the sample.**

$\hat{x}_t$ is produced by a differentiable path: $\hat{x} = G_\theta(z)$, then $\hat{x}_t = \alpha_t\hat{x} + \sigma_t\epsilon$. This rewrites sampling from $p_{\theta,t}$ as a deterministic function of noise $(z,\epsilon)$ and $\theta$, so the gradient can move inside the expectation:

$$ \nabla_\theta D_{\mathrm{KL}}(p_{\theta,t}\|p_{\mathrm{data},t}) = \mathbb{E}_{z,\epsilon}[\nabla_\theta\log p_{\theta,t}(\hat{x}_t) - \nabla_\theta\log p_{\mathrm{data},t}(\hat{x}_t)] $$

**Step 3 — Split each log-density gradient by the chain rule.**

Each log-density depends on $\theta$ two ways: directly, through the density function itself, and indirectly, through the evaluation point $\hat{x}_t(\theta)$. For the first term:

$$ \nabla_\theta\log p_{\theta,t}(\hat{x}_t) = (\nabla_\theta\log p_{\theta,t}(x_t))_{x_t=\hat{x}_t,\ \mathrm{fixed}} + \nabla_{\hat{x}_t}\log p_{\theta,t}(\hat{x}_t)\cdot\frac{\partial \hat{x}_t}{\partial\theta} $$

**Step 4 — The explicit term vanishes.**

The first piece on the right (holding the evaluation point fixed) is the *explicit* dependence. It vanishes in expectation by the score-function identity:

$$ \mathbb{E}_{x_t\sim p_{\theta,t}}[\nabla_\theta\log p_{\theta,t}(x_t)] = \nabla_\theta\int p_{\theta,t}(x_t)\,dx_t = \nabla_\theta 1 = 0 $$

Only the *implicit*, pathwise term through $\hat{x}_t(\theta)$ survives. The second log-density, $\log p_{\mathrm{data},t}(\hat{x}_t)$, depends on $\theta$ only through $\hat{x}_t$, since $p_{\mathrm{data},t}$ does not depend on $\theta$:

$$ \nabla_\theta\log p_{\mathrm{data},t}(\hat{x}_t) = \nabla_{\hat{x}_t}\log p_{\mathrm{data},t}(\hat{x}_t)\cdot\frac{\partial \hat{x}_t}{\partial\theta} $$

**Step 5 — Recombine.**

Substituting both pieces back into the Step 2 expression:

$$ \nabla_\theta D_{\mathrm{KL}}(p_{\theta,t}\|p_{\mathrm{data},t}) = \mathbb{E}_{z,\epsilon}[(\nabla_{\hat{x}_t}\log p_{\theta,t}(\hat{x}_t) - \nabla_{\hat{x}_t}\log p_{\mathrm{data},t}(\hat{x}_t))\frac{\partial \hat{x}_t}{\partial\theta}] $$

**Step 6 — Substitute the score function.**

Using $s(x) = \nabla_x\log p(x)$ from Section 2:

- $\nabla_{\hat{x}_t}\log p_{\theta,t}(\hat{x}_t) = s_{\mathrm{fake}}(\hat{x}_t,t)$: score of the noised generator distribution.
- $\nabla_{\hat{x}_t}\log p_{\mathrm{data},t}(\hat{x}_t) = s_{\mathrm{real}}(\hat{x}_t,t)$: score of the noised real-data distribution.

which gives:

$$ \nabla_\theta D_{\mathrm{KL}}(p_{\theta,t}\|p_{\mathrm{data},t}) = \mathbb{E}_{z,\epsilon}[(s_{\mathrm{fake}}(\hat{x}_t,t) - s_{\mathrm{real}}(\hat{x}_t,t))\frac{\partial \hat{x}_t}{\partial\theta}] $$

**Step 7 — Take the expectation over $t$ and match notation.**

Taking the outer expectation over $t$, flipping the sign to write the difference as $-(s_{\mathrm{real}}-s_{\mathrm{fake}})$, and absorbing the noising-path Jacobian $\partial\hat{x}_t/\partial\theta$ into $\partial\hat{x}/\partial\theta$ — they differ only by the constant scale factor $\alpha_t$, which DMD folds into the per-timestep loss weighting — gives the generator gradient used in this repo:

$$ \nabla_\theta \mathcal{L}_{\mathrm{DMD}} = -\mathbb{E}_{t,\hat{x},\hat{x}_t}[(s_{\mathrm{real}}(\hat{x}_t,t)-s_{\mathrm{fake}}(\hat{x}_t,t))\frac{\partial \hat{x}}{\partial \theta}] $$

The score function is:

$$ s(x) = \nabla_x \log p(x) $$

It points in the direction where the probability density increases fastest.

In PyTorch, we do not have explicit density functions for $p_{\theta,t}$ and $p_{\mathrm{data},t}$. Therefore, this repo does not directly implement the KL as a scalar loss. Instead, DMD uses the KL derivation to obtain the gradient that the clean generated sample $\hat{x}$ should receive.

Define:

$$ g = s_{\mathrm{fake}} - s_{\mathrm{real}} $$

Then the desired sample-level gradient is:

$$ \frac{\partial L}{\partial \hat{x}} = g $$

Since $\hat{x}=G_\theta(z)$, PyTorch can propagate this gradient back to the generator parameters through the chain rule:

$$ \frac{\partial L}{\partial \theta} = g\frac{\partial \hat{x}}{\partial \theta} $$

Therefore, the practical problem is to construct a scalar loss whose gradient with respect to $\hat{x}$ is exactly $g$.

### 4.3 Pseudo-MSE Form of the Generator Loss

Consider a squared loss with a detached target:

$$ L(\theta) = \frac{1}{2}\|\hat{x} - \mathrm{target}\|^2 $$

A squared loss has the property:

$$ \frac{\partial}{\partial \hat{x}}\frac{1}{2}\|\hat{x}-\mathrm{target}\|^2 = \hat{x}-\mathrm{target} $$

To make this gradient equal to $g$, set:

$$ \mathrm{target} = \mathrm{sg}[\hat{x}-g] $$

Then:

$$ \frac{\partial L}{\partial \hat{x}} = \hat{x}-\mathrm{sg}[\hat{x}-g] = g $$

The stop-gradient is essential. It makes the target a fixed pseudo-label, so gradients flow only through $\hat{x}$ and then into the generator parameters.

In DMD, the score-gradient term is estimated by teacher and critic clean predictions:

$$ g = f_\psi - f_\phi $$

The DMD generator loss becomes:

$$ L_{\mathrm{DMD}}(\theta) = \frac{1}{2}\|\hat{x} - \mathrm{sg}[\hat{x} - (f_\psi - f_\phi)]\|^2 $$

where:

- $f_\phi$: teacher clean prediction, corresponding to the real score.
- $f_\psi$: critic clean prediction, corresponding to the fake score.
- $\mathrm{sg}[\cdot]$: stop-gradient.

Source: `dmd/losses.py`, lines 113-114 and 207-210. In this repo, the score difference is implemented using clean-latent predictions:

```python
grad = x0_fake - x0_real
target = (generated_video_detached - grad).detach()
loss = 0.5 * F.mse_loss(generated_video, target)
```
So this MSE ensures that the DMD loss has gradient g with respect to $\hat{x}$, and this gradient can then be passed through $\hat{x}$ to update the generator.

### 4.4 Why the Critic Is Needed

DMD needs:

$$ s_{\mathrm{fake}}(x_t,t) = \nabla_{x_t}\log p_{\theta,t}(x_t) $$

The critic provides an external estimate of the current generator distribution score.

This is important because:

- The generator output $\hat{x}$ already depends on $\theta$.
- If the generator also provides the fake score used in the loss, the loss couples sample generation and score estimation through the same parameters.
- This introduces unstable higher-order gradient structure (?).
- The generator could reduce the loss by changing its score output rather than changing its generated distribution.

Therefore the repo uses:

- `generator`: produces $\hat{x}$.
- `fake_score`: separately trained critic that estimates $p_{\theta,t}$.
- `real_score`: frozen teacher that estimates $p_{\mathrm{data},t}$.

The critic is not a discriminator. It is a denoising score model trained on generated samples.

## 5 Implementation Details

### 5.1 Framewise Rollout and Output Window

Using the launch settings from Section 3.1, the rollout uses:

- one real initial frame,
- 48 generated future frames,
- one generated frame per block,
- only the final 13 generated frames for DMD.

The output window starts at:

$$ \mathrm{output\_start\_frame} = n_{\mathrm{total}} - n_{\mathrm{output}} = 49 - 13 = 36 $$

So the DMD loss is applied to frames `36..48`.

Source: `dmd/generation.py`, lines 141 and 253-272. This is implemented in `self_forcing_rollout`:

```python
output_start_frame = n_total_frames - n_output_frames

for local_i, fi in enumerate(block_frame_indices):
    if fi >= output_start_frame:
        output_frames.append(denoised_x0[:, local_i])

output = torch.stack(output_frames, dim=1)
```

### 5.2 Few-Step Diffusion Inside Rollout

Source: `train_dmd.py`, line 396, and `dmd/diffusion_utils.py`, lines 43-58. The few-step denoising schedule is:

```python
denoising_step_list = get_denoising_step_list(timesteps, num_denoising_steps)
```

For `timesteps=1000` and `num_denoising_steps=4`, this gives a list like:

```python
[999, 749, 499, 249]
```

For each generated frame or chunk:

1. Start from Gaussian noise.
2. Predict $v$ using the generator.
3. Convert $v$ to $x_0$.
4. If not at the sampled exit step, re-noise the predicted $x_0$ to the next timestep.
5. At the exit step, use the predicted $x_0$ as the generated frame or chunk.

Source: `dmd/generation.py`, lines 145-148. The sampled exit step is shared across all blocks:

```python
exit_step_idx = torch.randint(0, num_denoising_steps, (1,), device=device)
```

This means the generator is trained to produce useful clean predictions from different intermediate denoising depths.

### 5.3 Gradient Frames

Source: `train_dmd.py`, lines 464-466. `train_dmd.py` defines:

```python
grad_frames_from = max(0, n_output_frames - n_grad_frames)
```

Source: `train_dmd.py`, lines 164 and 464-466. By default:

```python
n_grad_frames = 1
n_output_frames = 13
grad_frames_from = 12
```

This means only the last frame in the returned output window carries generator rollout gradients. The full returned output window is still used by the score models for DMD scoring, but most rollout frames are produced under `torch.no_grad()`.

This saves memory because autoregressive video rollout is expensive.

### 5.4 Rolling KV Cache

The rolling KV cache is implemented across `dmd/generation.py`, `dmd/cache.py`, and `worldmodel/model_with_kv_cache.py`.

Source: `dmd/generation.py`, lines 150-178; `dmd/cache.py`, lines 31-74. The rollout allocates one external cache entry per temporal attention layer, then writes the initial real frame into the cache at timestep zero:

```python
kv_cache = create_kv_cache(...)
t_zero = torch.zeros([B, 1], device=device, dtype=torch.long)
_forward_and_cache_with_optional_fsdp_full_params(
    generator,
    x=initial_latent,
    t=t_zero,
    action=initial_action,
    external_kv_cache=kv_cache,
)
```

Each layer cache stores:

```python
{
    "k": [B * H_patch * W_patch, num_heads, max_frames, head_dim],
    "v": [B * H_patch * W_patch, num_heads, max_frames, head_dim],
    "end_idx": 0,
}
```

The spatial dimensions are folded into the batch dimension because only temporal attention is cached:

$$ B_{\mathrm{temporal}} = B \cdot H_{\mathrm{patch}} \cdot W_{\mathrm{patch}} $$

Source: `worldmodel/model_with_kv_cache.py`, lines 197-257. During temporal attention, the current frame's K/V are concatenated with cached K/V:

```python
k_cached = ext_cache["k"][:, :, :end_idx, :]
v_cached = ext_cache["v"][:, :, :end_idx, :]
k_full = torch.cat([k_cached, k_new], dim=2)
v_full = torch.cat([v_cached, v_new], dim=2)
out = F.scaled_dot_product_attention(q_rot, k_rot, v_full, is_causal=False)
```

Although `is_causal=False`, causality is controlled by the cache contents: the cache contains previous frames plus the current frame.

Source: `dmd/generation.py`, lines 258-270; `worldmodel/model_with_kv_cache.py`, lines 760-815. After each generated block, the clean output is written back into the external cache:

```python
while kv_cache[0]["end_idx"] + num_frame_per_block > max_cache_frames:
    evict_oldest_block(kv_cache, num_frame_per_block)

_forward_and_cache_with_optional_fsdp_full_params(
    generator,
    x=denoised_x0.detach(),
    t=t_zero_block,
    action=frame_actions,
    external_kv_cache=kv_cache,
)
```

Inside `write_to_external_cache`, K/V tensors are detached before storage, which prevents gradients from flowing through the whole autoregressive history.

Source: `dmd/cache.py`, lines 85-104. Rolling eviction shifts old entries out of the cache. For framewise training, `num_frame_per_block=1`, so one old frame is evicted at a time.

With `n_total_frames=49` and `max_cache_frames=13`, the first real frame is eventually evicted. The final generated frames cannot attend to the first frame, which simulates long-horizon inference with a finite temporal context window.

### 5.5 Score Models Run on the Full Output Window

The generator rollout uses the external KV cache. The score models do not.

Source: `dmd/losses.py`, lines 28-61. In `run_score_model`, the teacher and critic process the full generated output window in one forward pass:

```python
v_cond = model(noisy_input, timestep, action)
x0_cond = v_pred_to_x0(v_cond, noisy_input, alphas_cumprod, timestep)
```

No `external_kv_cache` is passed.

This means:

- Generator rollout is autoregressive and cache-based.
- Score estimation is full-window denoising with causal temporal attention inside the DiT.

### 5.6 Generator and Critic Update Schedule

Source: `train_dmd.py`, lines 481-482 and 574-604. The generator is updated less frequently:

```python
train_gen = (train_steps % dfake_gen_update_ratio == 0)
```

The critic is updated every step.

Source: `train_dmd.py`, lines 156-159. The default update ratio is:

```python
dfake_gen_update_ratio = 5
```

So the critic gets more updates than the generator. This keeps the fake score estimate closer to the generator's current distribution.

### 5.7 EMA Generator

Source: `train_dmd.py`, lines 335-338 and 553-554. The repo keeps an exponential moving average of the generator:

```python
generator_ema = deepcopy(generator)
update_ema(generator_ema, generator, 0.0)
...
update_ema(ema_no_ddp, gen_no_ddp, ema_decay)
```

Source: `train_dmd.py`, lines 653-676. The EMA weights are checkpointed with:

```python
{
    "generator": generator_state,
    "generator_ema": generator_ema_state,
    "fake_score": fake_score_state,
}
```

EMA is usually more stable for sampling after distillation, although evaluation is outside this note.



## 6 Common Pitfalls

- **Confusing teacher and critic**: `real_score` is the frozen teacher for the data distribution. `fake_score` is the trainable critic for the generator distribution.

- **Thinking the critic is adversarial**: the critic is trained with diffusion denoising loss, not a binary discriminator loss.

- **Forgetting stop-gradient in the generator loss**: the DMD pseudo-target must be detached. Otherwise gradients can flow through score estimation and noising operations.

- **Using generator outputs with gradients for score model calls**: `compute_generator_loss` detaches generated samples before noising and score-model inference. The original `generated_video` is only used in the final pseudo-MSE.

- **Mismatched output actions**: the loss uses only the last `n_output_frames`, so actions must be sliced with `actions[:, n_total_frames - n_output_frames:]`.

- **Script / Python argument mismatch**: `scripts/train_dmd_framewise.bash` passes `--subset_names`, while the checked `train_dmd.py` defines `subset_name`. This needs to be aligned before launch if the local code has not already been patched.

- **No rolling eviction**: if `max_cache_frames ≥ n_total_frames`, the first frame may stay visible throughout training. This does not simulate finite-cache long-video inference.

- **Wrong block divisibility**: `n_total_frames - 1` must be divisible by `num_frame_per_block`. For framewise training, this is always true because the block size is 1.

- **Assuming every output frame carries rollout gradients**: by default `n_grad_frames=1`, so only the last returned frame carries generator rollout gradients. This is a memory-saving choice.

- **Ignoring cache detach**: K/V cache writes are detached. This prevents backpropagation through the entire autoregressive history.

- **Assuming score models use the rolling cache**: teacher and critic process the full scored output window directly. The rolling cache is used in generator rollout.

