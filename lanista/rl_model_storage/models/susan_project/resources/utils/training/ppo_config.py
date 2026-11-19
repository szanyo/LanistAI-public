# Copyright (c) 2026 Benedek Szanyó.
# Copyright (c) 2026 HUN-REN SZTAKI.
# All rights reserved.

"""Shared PPO settings for Susan Ray/SB3 comparison runs.

These constants intentionally live in one place so SB3 and Ray can be compared
with the same learning setup. Comments describe the practical Susan effect of
each value, not only the generic PPO meaning.
"""

# Step size of PPO optimizer updates during curriculum learning. Larger values
# learn faster but can push the policy through the red-light boundary too
# abruptly and spike KL/loss. 7e-5 keeps enough pressure while the target is
# still moving through curriculum intersections.
PPO_LEARNING_RATE = 7.0e-5

# Learning-rate profile after the curriculum opens to the full route. The
# traffic-exposure experiment lowers the rate before unrestricted interaction
# begins, reducing policy-ratio clipping without changing either adapter.
PPO_FULL_CURRICULUM_LEARNING_RATE = 5.0e-5

# PPO optimization passes over each collected rollout batch. More epochs squeeze
# more learning out of the same samples but increase overfitting/stale-policy
# risk. Four passes keep updates gentle for noisy SUMO/light-phase rollouts.
PPO_NUM_EPOCHS = 4

# Minibatch size used inside each PPO update. Smaller minibatches add gradient
# noise/exploration but can destabilize the control policy; larger ones smooth
# the update. 256 should reduce Ray/SB3 update jitter while preserving enough
# batches per rollout for PPO clipping to stay meaningful.
PPO_MINIBATCH_SIZE = 256

# Environment samples collected before each PPO update. Keeping this explicit
# prevents Ray worker count from silently changing optimization cadence while
# matching Stable-Baselines3's established rollout size.
PPO_ROLLOUT_STEPS = 2048

# Future reward discount. Higher gamma makes the policy care more about reaching
# later intersections and handling lights before the stop line, but can make
# value targets higher variance. 0.995 fits the long approach/traffic-light
# horizon better than 0.99 without making returns fully long-tail.
PPO_GAMMA = 0.995

# GAE smoothing. Higher lambda uses longer advantage traces and helps delayed
# red-light outcomes propagate; too high can make advantages noisy. 0.97 keeps
# more temporal context than the default 0.95 while still damping variance.
PPO_GAE_LAMBDA = 0.97

# PPO policy-ratio clip. Larger clips adapt faster but can create action jumps
# and high clip_fraction; smaller clips are slower and safer. 0.10 is deliberate
# after earlier clip/loss spikes: the model should refine, not lurch.
PPO_CLIP_RANGE = 0.10

# Value-function clip in reward units. Too small blocks critic correction after
# curriculum/light changes; too large allows huge value-loss spikes. 80 keeps
# critic updates bounded while tolerating long-episode reward scale changes.
PPO_VALUE_CLIP_RANGE = 80.0

# Entropy bonus. The previous zero-traffic run collapsed policy std from about
# 0.37 to 0.05 before vehicle interaction began. This small shared coefficient
# preserves adaptation capacity while keeping continuous-control noise modest.
PPO_ENTROPY_COEFF = 1.0e-3

# Critic loss weight. Higher values prioritize value prediction stability but
# can drown policy gradients. 0.5 is the standard PPO balance and has been good
# enough for Susan once reward spikes were reduced.
PPO_VALUE_LOSS_COEFF = 0.5

# Gradient norm clipping. Lower values make updates safer but slower; higher
# values react faster to new curriculum levels. 0.3 is intentionally cautious
# for smooth acceleration and red-light stability.
PPO_MAX_GRAD_NORM = 0.3

# Early-stop KL target. If policy updates move too far from the sampled policy,
# PPO should stop the epoch early. 0.01 protects against sudden behavior flips
# when the curriculum exposes harder traffic-light phase offsets.
PPO_TARGET_KL = 0.01

# Both frameworks use two 64-unit layers for the actor and critic.
PPO_POLICY_NET_ARCH = (64, 64)

# The hidden layers use tanh activation.
PPO_POLICY_ACTIVATION = "tanh"

# Initial log standard deviation for the Gaussian action distribution.
# Lower means less random acceleration at the start; higher explores more but
# can create jitter. -1.0 starts with std ~= 0.37 in the physical acceleration
# action units used by both adapters, providing controlled nonzero exploration.
PPO_POLICY_LOG_STD_INIT = -1.0
