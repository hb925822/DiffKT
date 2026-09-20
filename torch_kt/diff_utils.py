#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2024/10/25 0025 上午 8:35
# @Author  : hb
# @File    : diff_utils.py
import abc
import math

import torch
import numpy as np



def normal_kl(mean1, logvar1, mean2, logvar2):
    """
    Compute the KL divergence between two gaussians.

    Shapes are automatically broadcasted, so batches can be compared to
    scalars, among other use cases.
    """
    tensor = None
    for obj in (mean1, logvar1, mean2, logvar2):
        if isinstance(obj, torch.Tensor):
            tensor = obj
            break
    assert tensor is not None, "at least one argument must be a Tensor"

    # Force variances to be Tensors. Broadcasting helps convert scalars to
    # Tensors, but it does not work for torch.exp().
    logvar1, logvar2 = [
        x if isinstance(x, torch.Tensor) else torch.tensor(x).to(tensor)
        for x in (logvar1, logvar2)
    ]

    # print(logvar2.shape)
    # temp1 = 0.5 * (-1.0 + logvar2 - logvar1 + torch.exp(logvar1 - logvar2))
    # print(f'const = {temp1.mean()}, coef={(torch.exp(-logvar2) * 0.5).mean()}, mse={((mean1 - mean2) ** 2).mean().item()}')

    return 0.5 * (
        -1.0
        + logvar2
        - logvar1
        + torch.exp(logvar1 - logvar2)
        + ((mean1 - mean2) ** 2) * torch.exp(-logvar2)
    )
##################################################timesteps ScheduleSampler
class ScheduleSampler(abc.ABC):
    """
    A distribution over timesteps in the diffusion process, intended to reduce
    variance of the objective.

    By default, samplers perform unbiased importance sampling, in which the
    objective's mean is unchanged.
    However, subclasses may override sample() to change how the resampled
    terms are reweighted, allowing for actual changes in the objective.
    """

    @abc.abstractmethod
    def weights(self):
        """
        Get a numpy array of weights, one per diffusion step.
        The weights needn't be normalized, but must be positive.
        """

    def sample(self, batch_size, device):
        """
        Importance-sample timesteps for a batch.

        :param batch_size: the number of timesteps.
        :param device: the torch device to save to.
        :return: a tuple (timesteps, weights):
                 - timesteps: a tensor of timestep indices.
                 - weights: a tensor of weights to scale the resulting losses.
        """
        w = self.weights()
        p = w / np.sum(w)
        indices_np = np.random.choice(len(p), size=(batch_size,), p=p)
        indices = torch.from_numpy(indices_np).long().to(device)
        weights_np = 1 / (len(p) * p[indices_np])
        weights = torch.from_numpy(weights_np).float().to(device)
        return indices, weights


class UniformSampler(ScheduleSampler):
    def __init__(self, num_timesteps):
        self.num_timesteps = num_timesteps
        self._weights = np.ones([self.num_timesteps])

    def weights(self):
        return self._weights


class LossAwareSampler(ScheduleSampler):
    def update_with_local_losses(self, local_ts, local_losses):
        """
        Update the reweighting using losses from a model.

        Call this method from each rank with a batch of timesteps and the
        corresponding losses for each of those timesteps.
        This method will perform synchronization to make sure all of the ranks
        maintain the exact same reweighting.

        :param local_ts: an integer Tensor of timesteps.
        :param local_losses: a 1D Tensor of losses.
        """
        batch_sizes = [
            torch.tensor([0], dtype=torch.int32, device=local_ts.device)
            for _ in range(math.dist.get_world_size())
        ]
        math.dist.all_gather(
            batch_sizes,
            torch.tensor([len(local_ts)], dtype=torch.int32, device=local_ts.device),
        )

        # Pad all_gather batches to be the maximum batch size.
        batch_sizes = [x.item() for x in batch_sizes]
        max_bs = max(batch_sizes)

        timestep_batches = [torch.zeros(max_bs).to(local_ts) for bs in batch_sizes]
        loss_batches = [torch.zeros(max_bs).to(local_losses) for bs in batch_sizes]
        math.dist.all_gather(timestep_batches, local_ts)
        math.dist.all_gather(loss_batches, local_losses)
        timesteps = [
            x.item() for y, bs in zip(timestep_batches, batch_sizes) for x in y[:bs]
        ]
        losses = [x.item() for y, bs in zip(loss_batches, batch_sizes) for x in y[:bs]]
        self.update_with_all_losses(timesteps, losses)

    @abc.abstractmethod
    def update_with_all_losses(self, ts, losses):
        """
        Update the reweighting using losses from a model.

        Sub-classes should override this method to update the reweighting
        using losses from the model.

        This method directly updates the reweighting without synchronizing
        between workers. It is called by update_with_local_losses from all
        ranks with identical arguments. Thus, it should have deterministic
        behavior to maintain state across workers.

        :param ts: a list of int timesteps.
        :param losses: a list of float losses, one per timestep.
        """


class LossSecondMomentResampler(LossAwareSampler):
    def __init__(self, num_timesteps, history_per_term=10, uniform_prob=0.001):
        self.num_timesteps = num_timesteps
        self.history_per_term = history_per_term
        self.uniform_prob = uniform_prob
        self._loss_history = np.zeros(
            [self.num_timesteps, history_per_term], dtype=np.float64
        )
        self._loss_counts = np.zeros([self.num_timesteps], dtype=np.int32)

    def weights(self):
        if not self._warmed_up():
            return np.ones([self.num_timesteps], dtype=np.float64)
        weights = np.sqrt(np.mean(self._loss_history ** 2, axis=-1))
        weights /= np.sum(weights)
        weights *= 1 - self.uniform_prob
        weights += self.uniform_prob / len(weights)
        return weights

    def update_with_all_losses(self, ts, losses):
        for t, loss in zip(ts, losses):
            if self._loss_counts[t] == self.history_per_term:
                # Shift out the oldest loss term.
                self._loss_history[t, :-1] = self._loss_history[t, 1:]
                self._loss_history[t, -1] = loss
            else:
                self._loss_history[t, self._loss_counts[t]] = loss
                self._loss_counts[t] += 1

    def _warmed_up(self):
        return (self._loss_counts == self.history_per_term).all()


class FixSampler(ScheduleSampler):
    def __init__(self, num_timesteps):
        self.num_timesteps = num_timesteps
        ###############################################################
        ### You can custome your own sampling weight of steps here. ###
        ###############################################################
        self._weights = np.concatenate([np.ones([num_timesteps // 2]), np.zeros([num_timesteps // 2]) + 0.5])

    def weights(self):
        return self._weights


def create_named_schedule_sampler(name, num_timesteps):
    """
    Create a ScheduleSampler from a library of pre-defined samplers.
    :param name: the name of the sampler.
    :param diffusion: the diffusion object to sample for.
    """
    if name == "uniform":
        return UniformSampler(num_timesteps)
    elif name == "lossaware":
        return LossSecondMomentResampler(num_timesteps)  ## default setting
    elif name == "fixstep":
        return FixSampler(num_timesteps)
    else:
        raise NotImplementedError(f"unknown schedule sampler: {name}")


################################################ betas schedule
def betas_for_alpha_bar(num_diffusion_timesteps, alpha_bar, max_beta=0.999):
    """
    Create a beta schedule that discretizes the given alpha_t_bar function, which defines the cumulative product of (1-beta) over time from t = [0,1].
    :param num_diffusion_timesteps: the number of betas to produce.
    :param alpha_bar: a lambda that takes an argument t from 0 to 1 and produces the cumulative product of (1-beta) up to that part of the diffusion process.
    :param max_beta: the maximum beta to use; use values lower than 1 to prevent singularities.
    """
    betas = []
    for i in range(num_diffusion_timesteps):  ## 2000
        t1 = i / num_diffusion_timesteps
        t2 = (i + 1) / num_diffusion_timesteps
        betas.append(min(1 - alpha_bar(t2) / alpha_bar(t1), max_beta))
    return np.array(betas)


def betas_for_alpha_bar_left(num_diffusion_timesteps, alpha_bar, max_beta=0.999):
    """
    Create a beta schedule that discretizes the given alpha_t_bar function, but shifts towards left interval starting from 0
    which defines the cumulative product of (1-beta) over time from t = [0,1].

    :param num_diffusion_timesteps: the number of betas to produce.
    :param alpha_bar: a lambda that takes an argument t from 0 to 1 and
                      produces the cumulative product of (1-beta) up to that
                      part of the diffusion process.
    :param max_beta: the maximum beta to use; use values lower than 1 to
                     prevent singularities.
    """
    betas = []
    betas.append(min(1 - alpha_bar(0), max_beta))
    for i in range(num_diffusion_timesteps - 1):
        t1 = i / num_diffusion_timesteps
        t2 = (i + 1) / num_diffusion_timesteps
        betas.append(min(1 - alpha_bar(t2) / alpha_bar(t1), max_beta))
    return np.array(betas)


def get_named_beta_schedule(schedule_name, num_diffusion_timesteps):
    """
    Get a pre-defined beta schedule for the given name.

    The beta schedule library consists of beta schedules which remain similar
    in the limit of num_diffusion_timesteps.
    Beta schedules may be added, but should not be removed or changed once
    they are committed to maintain backwards compatibility.
    """
    if schedule_name == "linear":
        # Linear schedule from Ho et al, extended to work for any number of
        # diffusion steps.
        scale = 1000 / num_diffusion_timesteps
        beta_start = scale * 0.0001
        beta_end = scale * 0.02
        return np.linspace(
            beta_start, beta_end, num_diffusion_timesteps, dtype=np.float64
        )
    elif schedule_name == "cosine":
        return betas_for_alpha_bar(
            num_diffusion_timesteps,
            lambda t: math.cos((t + 0.008) / 1.008 * math.pi / 2) ** 2,
        )
    elif schedule_name == 'sqrt':
        return betas_for_alpha_bar(
            num_diffusion_timesteps,
            lambda t: 1-np.sqrt(t + 0.0001),
        )
    elif schedule_name == "trunc_cos":
        return betas_for_alpha_bar_left(
            num_diffusion_timesteps,
            lambda t: np.cos((t + 0.1) / 1.1 * np.pi / 2) ** 2,
        )
    elif schedule_name == 'trunc_lin':
        scale = 1000 / num_diffusion_timesteps
        beta_start = scale * 0.0001 + 0.01
        beta_end = scale * 0.02 + 0.01
        return np.linspace(
            beta_start, beta_end, num_diffusion_timesteps, dtype=np.float64
        )
    elif schedule_name == 'pw_lin':
        scale = 1000 / num_diffusion_timesteps
        beta_start = scale * 0.0001 + 0.01
        beta_mid = scale * 0.0001  #scale * 0.02
        beta_end = scale * 0.02
        first_part = np.linspace(
            beta_start, beta_mid, 10, dtype=np.float64
        )
        second_part = np.linspace(
            beta_mid, beta_end, num_diffusion_timesteps - 10 , dtype=np.float64
        )
        return np.concatenate(
            [first_part, second_part]
        )
    else:
        raise NotImplementedError(f"unknown beta schedule: {schedule_name}")


class GaussianDiffusion(object):
    def __init__(self, schedule_sampler_name, diffusion_steps, noise_schedule, rescale_timestep=False,
                 diff_type="mean", beta_start=0.0001, beta_end=0.02,
                 **kwargs):
        """
        Utilities for training and sampling diffusion models.

        :param rescale_timestep: if True, pass floating point timesteps into the
                                  model so that they are always scaled like in the
                                  original paper (0 to 1000).
        """
        super(GaussianDiffusion, self).__init__()
        self.schedule_sampler_name = schedule_sampler_name
        self.diffusion_steps = diffusion_steps
        self.diff_type = diff_type
        self.beta_start = beta_start
        self.beta_end = beta_end
        self.noise_schedule = noise_schedule
        betas = self.get_betas(self.noise_schedule, self.diffusion_steps)
        # Use float64 for accuracy.
        betas = np.array(betas, dtype=np.float64)
        self.betas = betas
        assert len(betas.shape) == 1, "betas must be 1-D"
        assert (betas > 0).all() and (betas <= 1).all()
        alphas = 1.0 - betas
        self.alphas_cumprod = np.cumprod(alphas, axis=0)

        self.alphas_cumprod_prev = np.append(1.0, self.alphas_cumprod[:-1])

        # calculations for diffusion q(x_t | x_{t-1}) and others
        self.sqrt_alphas_cumprod = np.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = np.sqrt(1.0 - self.alphas_cumprod)
        print((self.sqrt_alphas_cumprod[0], self.sqrt_alphas_cumprod[1], self.sqrt_alphas_cumprod[-1]))
        # calculations for diffusion q(x_t | x_{t-1}) and others
        self.log_one_minus_alphas_cumprod = np.log(1.0 - self.alphas_cumprod)
        self.sqrt_recip_alphas_cumprod = np.sqrt(1.0 / self.alphas_cumprod)
        self.sqrt_recipm1_alphas_cumprod = np.sqrt(1.0 / self.alphas_cumprod - 1)

        self.posterior_mean_coef1 = (betas * np.sqrt(self.alphas_cumprod_prev) / (1.0 - self.alphas_cumprod))
        self.posterior_mean_coef2 = ((1.0 - self.alphas_cumprod_prev) * np.sqrt(alphas) / (1.0 - self.alphas_cumprod))

        # calculations for posterior q(x_{t-1} | x_t, x_0)
        self.posterior_variance = (betas * (1.0 - self.alphas_cumprod_prev) / (1.0 - self.alphas_cumprod))
        self.posterior_log_variance = np.log(np.append(self.posterior_variance[1], self.posterior_variance[1:]))
        self.num_timesteps = int(self.betas.shape[0])

        self.schedule_sampler = create_named_schedule_sampler(self.schedule_sampler_name,
                                                              self.num_timesteps)  ## lossaware (schedule_sample)
        self.rescale_timestep = rescale_timestep
        self.original_num_steps = len(betas)

    def get_betas(self, noise_schedule, diffusion_steps):
        if noise_schedule.startswith("fix"):
            if noise_schedule == "fix":
                betas = np.linspace(self.beta_start, self.beta_end, self.diffusion_steps, dtype=np.float64)
            elif noise_schedule == "fix_exp":
                x = torch.linspace(1, 2 * self.diffusion_steps + 1, self.diffusion_steps)
                betas = 1 - torch.exp(
                    - self.beta_start / self.diffusion_steps - x * 0.5 * (self.beta_end - self.beta_start) / (
                                self.diffusion_steps * self.diffusion_steps))
            elif noise_schedule== "fix_sigmoid":
                betas = np.linspace(-6, 6, self.diffusion_steps, dtype=np.float64)
                betas = self.beta_start + (self.beta_end - self.beta_start) * (1 / (1 + np.exp(-betas / 1.0)))
            elif noise_schedule == "fix_cosine":
                steps = self.diffusion_steps + 1
                x = torch.linspace(0, self.diffusion_steps, steps)
                alphas_cumprod = torch.cos(((x / self.diffusion_steps) + 0.008) / (1 + 0.008) * torch.pi * 0.5) ** 2
                alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
                betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
                betas =torch.clip(betas, 0.0001, 0.9999)
            else:
                raise NotImplementedError(f"unknown beta schedule: {noise_schedule}")
        else:
            betas = get_named_beta_schedule(noise_schedule, diffusion_steps)  ## array, generate beta
        return betas

    def sample_steps(self, batch_size, device, **kwargs):
        return self.schedule_sampler.sample(batch_size,
                                            device)

    def _extract_into_tensor(self, arr, timesteps, broadcast_shape):
        """
        Extract values from a 1-D numpy array for a batch of indices.

        :param arr: the 1-D numpy array.
        :param timesteps: a tensor of indices into the array to extract.
        :param broadcast_shape: a larger shape of K dimensions with the batch
                                dimension equal to the length of timesteps.
        :return: a tensor of shape [batch_size, 1, ...] where the shape has K dims.
        """

        res = torch.from_numpy(arr).to(device=timesteps.device)[timesteps].float()
        while len(res.shape) < len(broadcast_shape):
            res = res[..., None]
        # return res.expand(broadcast_shape)
        return res

    def q_mean_x_t(self, x_start):
        t = torch.LongTensor([self.num_timesteps - 1]).to(x_start.device)
        return self._extract_into_tensor(self.sqrt_alphas_cumprod, t, x_start.shape) * x_start

    def q_x_t_mean(self, x_start):
        t = torch.LongTensor([self.num_timesteps - 1]).to(x_start.device)
        return 1.0e-5/self._extract_into_tensor(self.sqrt_alphas_cumprod, t, x_start.shape)

    def q_mean_variance(self, x_start, t):
        """
        Get the distribution q(x_t | x_0).

        :param x_start: the [N x C x ...] tensor of noiseless inputs.
        :param t: the number of diffusion steps (minus 1). Here, 0 means one step.
        :return: A tuple (mean, variance, log_variance), all of x_start's shape.
        """
        mean = (
                self._extract_into_tensor(self.sqrt_alphas_cumprod, t, x_start.shape) * x_start
        )
        variance = self._extract_into_tensor(1.0 - self.alphas_cumprod, t, x_start.shape)
        log_variance = self._extract_into_tensor(
            self.log_one_minus_alphas_cumprod, t, x_start.shape
        )
        return mean, variance, log_variance

    def get_x1_start(self, x_start_mean):
        '''
        Word embedding projection from {Emb(w)} to {x_0}
        :param x_start_mean: word embedding
        :return: x_0
        '''
        std = self._extract_into_tensor(self.sqrt_one_minus_alphas_cumprod,
                                        torch.tensor([0]).to(x_start_mean.device),
                                        x_start_mean.shape)
        noise = torch.randn_like(x_start_mean)
        assert noise.shape == x_start_mean.shape
        # print(x_start_mean.device, noise.device)
        return (
                x_start_mean + std * noise
        )

    def q_sample(self, x_start, t, noise=None, mask=None):
        """
        Diffuse the data for a given number of diffusion steps.

        In other words, sample from q(x_t | x_0).

        :param x_start: the initial data batch.
        :param t: the number of diffusion steps (minus 1). Here, 0 means one step.
        :param noise: if specified, the split-out normal noise.
        :param mask: anchoring masked position
        :return: A noisy version of x_start.
        """
        if noise is None:
            noise = torch.randn_like(x_start)

        assert noise.shape == x_start.shape
        x_t = (
                self._extract_into_tensor(self.sqrt_alphas_cumprod, t, x_start.shape) * x_start
                + self._extract_into_tensor(self.sqrt_one_minus_alphas_cumprod, t, x_start.shape)
                * noise  ## reparameter trick
        )  ## genetrate x_t based on x_0 (x_start) with reparameter trick

        if mask == None:
            return x_t
        else:
            mask = torch.broadcast_to(mask.unsqueeze(dim=-1), x_start.shape)  ## mask: [0,0,0,1,1,1,1,1]
            return torch.where(mask == 0, x_start, x_t)  ## replace the output_target_seq embedding (x_0) as x_t

    def _scale_timesteps(self, t):
        # if self.rescale_timestep:
        #     return t.float() * (1000.0 / self.num_timesteps)
        # else:
        #     return t
        return t

    def _predict_xstart_from_eps(self, x_t, t, eps):

        assert x_t.shape == eps.shape
        return (
                self._extract_into_tensor(self.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t
                - self._extract_into_tensor(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape) * eps
        )

    def q_posterior_mean_variance(self, x_start, x_t, t):
        """
        Compute the mean and variance of the diffusion posterior:
            q(x_{t-1} | x_t, x_0)

        """
        assert x_start.shape == x_t.shape
        posterior_mean = (
                self._extract_into_tensor(self.posterior_mean_coef1, t, x_t.shape) * x_start
                + self._extract_into_tensor(self.posterior_mean_coef2, t, x_t.shape) * x_t
        )  ## \mu_t
        assert (posterior_mean.shape[0] == x_start.shape[0])
        posterior_variance = self._extract_into_tensor(self.posterior_variance, t, x_t.shape)
        posterior_log_variance = self._extract_into_tensor(
            self.posterior_log_variance, t, x_t.shape
        )
        return posterior_mean, posterior_variance, posterior_log_variance

    def p_mean_variance(self, model_net, x_t, t, clip_denoised=True, **model_kwargs):
        model_output = model_net(x_t, self._scale_timesteps(t), **model_kwargs)
        if self.diff_type == "mean":
            x_0 = model_output
        else:
            x_0 = self._predict_xstart_from_eps(x_t, t, model_output)  ## eps predict
        if clip_denoised:
            x_0 = x_0.clamp(-1, 1)
        ## x_start: candidante item embedding, x_t: inputseq_embedding + outseq_noise, output x_(t-1) distribution
        posterior_mean, _, model_log_variance = self.q_posterior_mean_variance(x_start=x_0,
                                                                               x_t=x_t,
                                                                               t=t)
        # model_log_variances = np.log(np.append(self.posterior_variance[1], self.betas[1:]))
        # model_log_variance = self._extract_into_tensor(model_log_variances, t, x_t.shape)
        return posterior_mean, model_log_variance

    def p_sample(self, model_net, x_t, t, clip_denoised=True, **model_kwargs):
        model_mean, model_log_variance = self.p_mean_variance(model_net, x_t, t, clip_denoised=clip_denoised,
                                                              **model_kwargs)
        noise = torch.randn_like(x_t)
        nonzero_mask = ((t != 0).float().view(-1, *([1] * (len(x_t.shape) - 1))))  # no noise when t == 0
        sample_xt = model_mean + nonzero_mask * torch.exp(
            0.5 * model_log_variance) * noise  ## sample x_{t-1} from the \mu(x_{t-1}) distribution based on the reparameter trick
        return sample_xt,model_mean, model_log_variance

    def reverse_p_sample(self, model_net, noise_x_t,  clip_denoised=True,**model_kwargs):
        device = noise_x_t.device
        indices = list(range(self.num_timesteps))[::-1]
        B = noise_x_t.shape[0]
        model_log_variance=None
        xs=[]
        for i in indices:  # from T to 0, reversion iteration
            t = torch.tensor([i] * B, device=device)
            noise_x_t,model_mean, model_log_variance = self.p_sample(model_net, noise_x_t, t, clip_denoised=clip_denoised, **model_kwargs)
            xs.insert(0,noise_x_t)#xs.append(noise_x_t)
        return noise_x_t, model_log_variance,xs

    def reverse_p_sample_ext(self, model_net, noise_x_t, others_index=None, clip_denoised=True, x0_others=-1,
                         **model_kwargs):
        if others_index is None:
            others_index = []
        device = noise_x_t.device
        indices = list(range(self.num_timesteps))[::-1]
        noise_outs = []
        x0s = []
        B = noise_x_t.shape[0]
        for i in indices:  # from T to 0, reversion iteration
            t = torch.tensor([i] * B, device=device)
            # with torch.no_grad():
            noise_x_t, model_mean, model_log_variance = self.p_sample(model_net, noise_x_t, t,
                                                                      clip_denoised=clip_denoised, **model_kwargs)
            if x0_others > 0 and i == 0:
                for k in range(x0_others):
                    noise_tmp = torch.randn_like(noise_x_t)
                    sample_x0 = model_mean + torch.exp(
                        0.5 * model_log_variance) * noise_tmp
                    x0s.append(sample_x0)
                outputs_final = torch.stack(x0s, dim=0)
                noise_x_t = torch.mean(outputs_final, dim=0)
            if i in others_index:
                noise_outs.append(noise_x_t)
        if x0_others > 0:

            return noise_x_t, noise_outs,
        else:
            return noise_x_t, noise_outs
    def reverse_p_sample_mutil(self, model_net, noise_x_t, others_index=None, clip_denoised=True,seed_count=5,org_seed=100, **model_kwargs):
        if others_index is None:
            others_index = []
        device = noise_x_t.device
        indices = list(range(self.num_timesteps))[::-1]
        noise_outs = []
        B = noise_x_t.shape[0]
        outputs=[]
        for s,seed in enumerate(range(seed_count)):
            torch.seed()
            noise_x_t=torch.randn_like(noise_x_t)
            tmp=[]
            for i in indices:  # from T to 0, reversion iteration
                t = torch.tensor([i] * B, device=device)
                # with torch.no_grad():
                noise_x_t,model_mean, model_log_variance = self.p_sample(model_net, noise_x_t, t, clip_denoised=clip_denoised, **model_kwargs)
                if i in others_index:
                    tmp.append(noise_x_t)
                outputs.append(noise_x_t)
            noise_outs.append(torch.stack(tmp,dim=0))
        outputs_final=torch.stack(outputs,dim=0)
        outputs_final=torch.mean(outputs_final,dim=0)
        noise_outs_final = torch.stack(noise_outs, dim=0)
        noise_outs_final = torch.mean(noise_outs_final, dim=0)
        return outputs_final, torch.trunc(noise_outs_final)
    def reverse_p_sample_loop(self, model_net, noise_x_t, others_index=None, clip_denoised=True,w=2, **model_kwargs):
        if others_index is None:
            others_index = []
        device = noise_x_t.device
        indices = list(range(self.num_timesteps))[::-1]
        noise_outs = []
        B = noise_x_t.shape[0]
        for i in indices:  # from T to 0, reversion iteration
            t = torch.tensor([i] * B, device=device)
            model_output1 = model_net(noise_x_t, self._scale_timesteps(t), condition=model_kwargs["no_condition"],
                                      mask=model_kwargs["mask"])
            model_output2 = model_net(noise_x_t, self._scale_timesteps(t), condition=model_kwargs["condition"],
                                      mask=model_kwargs["mask"])
            model_output = (1 + w) * model_output2 - w * model_output1
            if self.diff_type == "mean":
                x_0 = model_output
            else:
                x_0 = self._predict_xstart_from_eps(noise_x_t, t, model_output)  ## eps predict
            if clip_denoised:
                x_0 = x_0.clamp(-1, 1)
            posterior_mean, _, model_log_variance = self.q_posterior_mean_variance(x_start=x_0,
                                                                                   x_t=noise_x_t,
                                                                                   t=t)
            noise = torch.randn_like(noise_x_t)
            nonzero_mask = ((t != 0).float().view(-1, *([1] * (len(noise_x_t.shape) - 1))))  # no noise when t == 0
            noise_x_t = posterior_mean + nonzero_mask * torch.exp(
                0.5 * model_log_variance) * noise
            if i in others_index:
                noise_outs.append(noise_x_t)
        return noise_x_t, noise_outs
    def reverse_p_sample_loops(self, model_net, noise_x_t, others_index=None, clip_denoised=True, **model_kwargs):
        if others_index is None:
            others_index = []
        device = noise_x_t.device
        indices = list(range(self.num_timesteps))[::-1]
        noise_outs = []
        B = noise_x_t.shape[0]
        self_condition=torch.zeros_like(noise_x_t)
        for i in indices:  # from T to 0, reversion iteration
            t = torch.tensor([i] * B, device=device)
            self_condition = model_net(noise_x_t, self._scale_timesteps(t), condition=model_kwargs["condition"],
                                      mask=model_kwargs["mask"],self_condition=self_condition)
            if self.diff_type != "mean":
                self_condition = self._predict_xstart_from_eps(noise_x_t, t, self_condition)  ## eps predict
            model_output = model_net(noise_x_t, self._scale_timesteps(t), condition=model_kwargs["condition"],
                                       mask=model_kwargs["mask"], self_condition=self_condition)
            if self.diff_type == "mean":
                x_0 = model_output
            else:
                x_0 = self._predict_xstart_from_eps(noise_x_t, t, model_output)  ## eps predict
            if clip_denoised:
                x_0 = x_0.clamp(-1, 1)
            posterior_mean, _, model_log_variance = self.q_posterior_mean_variance(x_start=x_0,
                                                                                   x_t=noise_x_t,
                                                                                   t=t)
            noise = torch.randn_like(noise_x_t)
            nonzero_mask = ((t != 0).float().view(-1, *([1] * (len(noise_x_t.shape) - 1))))  # no noise when t == 0
            noise_x_t = posterior_mean + nonzero_mask * torch.exp(
                0.5 * model_log_variance) * noise
            if i in others_index:
                noise_outs.append(noise_x_t)
        return noise_x_t, noise_outs
    def forward_step(self, model_net, x, noise, **model_kwargs):
        t, weights = self.sample_steps(x.shape[0], x.device)
        # t = torch.randint(0, self.diffusion_steps, (x.shape[0],), device=x.device).long()
        x_t = self.q_sample(x, t, noise=noise)
        model_output = model_net(x_t, self._scale_timesteps(t), **model_kwargs)
        if self.diff_type == "mean":
            x_0 = model_output
            pred_noise = x_t - x_0
        else:
            pred_noise = model_output
            x_0 = self._predict_xstart_from_eps(x_t, t, pred_noise)  ## eps predict
            # x_0 = x_t-pred_noise#self._predict_xstart_from_eps(x_t, t, pred_noise)
        return pred_noise, x_0
    def forward_n_step(self, model_net, x, noise, **model_kwargs):
        t=torch.tensor([self.num_timesteps-1]*x.shape[0]).to(x.device)
        x_t = self.q_sample(x, t, noise=noise)
        model_output = model_net(x_t, self._scale_timesteps(t), **model_kwargs)
        if self.diff_type == "mean":
            x_0 = model_output
            pred_noise = x_t - x_0
        else:
            pred_noise = model_output
            x_0 = self._predict_xstart_from_eps(x_t, t, pred_noise)  ## eps predict
            # x_0 = x_t-pred_noise#self._predict_xstart_from_eps(x_t, t, pred_noise)
        return pred_noise, x_0
    def forward_steps(self, model_net, x, noise, **model_kwargs):
        t, weights = self.sample_steps(x.shape[0], x.device)
        # t = torch.randint(0, self.diffusion_steps, (x.shape[0],), device=x.device).long()
        x_t = self.q_sample(x, t, noise=noise)
        self_condition=torch.zeros_like(x_t)
        if torch.rand(1).item() > 0.5:
            with torch.no_grad():
                self_condition = model_net(x_t, self._scale_timesteps(t), condition=model_kwargs["condition"],
                                         mask=model_kwargs["mask"], self_condition=self_condition)
        if self.diff_type != "mean":
            self_condition = self._predict_xstart_from_eps(x_t, t, self_condition)  ## eps predict
        model_output = model_net(x_t, self._scale_timesteps(t), condition=model_kwargs["condition"],
                                 mask=model_kwargs["mask"], self_condition=self_condition)
        if self.diff_type == "mean":
            x_0 = model_output
            pred_noise = x_t - x_0
        else:
            pred_noise = model_output
            x_0 = self._predict_xstart_from_eps(x_t, t, pred_noise)  ## eps predict
            # x_0 = x_t-pred_noise#self._predict_xstart_from_eps(x_t, t, pred_noise)
        return pred_noise, x_0
    def forward_step_con(self, model_net, x, noise, **model_kwargs):
        t, weights = self.sample_steps(x.shape[0], x.device)
        x_t = self.q_sample(x, t, noise=noise)
        model_output = model_net(x_t, self._scale_timesteps(t), **model_kwargs)
        condition=model_kwargs["condition"]
        model_kwargs.pop("condition")
        model_output1 = model_net(x_t, self._scale_timesteps(t),condition=torch.zeros_like(condition), **model_kwargs)
        if self.diff_type == "mean":
            x_0 = model_output
            pred_noise = x_t - x_0
        else:
            pred_noise = model_output
            x_0 = self._predict_xstart_from_eps(x_t, t, pred_noise)  ## eps predict
            # x_0 = x_t-pred_noise#self._predict_xstart_from_eps(x_t, t, pred_noise)
        return pred_noise, x_0,model_output1

    def forward_step_seq2seq(self, model_net, x, noise, **model_kwargs):
        t, weights = self.sample_steps(x.shape[0], x.device)
        x_t = self.q_sample(x, t, noise=noise)
        model_output = model_net(x_t, self._scale_timesteps(t), **model_kwargs)
        if self.diff_type == "mean":
            x_0 = model_output
            pred_noise = x_t - x_0
        else:
            # pred_noise = torch.clamp(model_output, -1, 1)
            pred_noise = model_output
            x_0 = self._predict_xstart_from_eps(x_t, t, pred_noise)  ## eps predict
            # x_0 = x_t-pred_noise#self._predict_xstart_from_eps(x_t, t, pred_noise)
        return pred_noise, x_0, t

    def forward_step_seq2seq1(self, model_net, x, noise, x1, **model_kwargs):
        t, weights = self.sample_steps(x.shape[0], x.device)
        x_t = self.q_sample(x, t, noise=noise)
        x_t1 = self.q_sample(x1, t, noise=noise)
        model_output = model_net(x_t, self._scale_timesteps(t), **model_kwargs)
        model_kwargs1 = model_kwargs.copy()
        model_kwargs1["condition"] = torch.zeros_like(model_kwargs1["condition"])
        model_output1 = model_net(x_t1, self._scale_timesteps(t), **model_kwargs1)
        if self.diff_type == "mean":
            x_0 = model_output
            pred_noise = x_t - x_0
            x_01 = model_output1
            pred_noise1 = x_t1 - x_01
        else:
            # pred_noise = torch.clamp(model_output, -1, 1)
            pred_noise = model_output
            x_0 = self._predict_xstart_from_eps(x_t, t, pred_noise)  ## eps predict
            # x_0 = x_t-pred_noise#self._predict_xstart_from_eps(x_t, t, pred_noise)
        return pred_noise, x_0, t, pred_noise1, x_01


class ClassFreeGaussianDiffusion(GaussianDiffusion):
    def __init__(self, schedule_sampler_name, diffusion_steps, noise_schedule, rescale_timestep=False,
                 diff_type="mean", beta_start=0.0001, beta_end=0.02,
                 **kwargs):
        """
        Utilities for training and sampling diffusion models.

        :param rescale_timestep: if True, pass floating point timesteps into the
                                  model so that they are always scaled like in the
                                  original paper (0 to 1000).
        """
        super(ClassFreeGaussianDiffusion, self).__init__(schedule_sampler_name, diffusion_steps, noise_schedule,
                                                         rescale_timestep=rescale_timestep,
                                                         diff_type=diff_type, beta_start=beta_start, beta_end=beta_end,
                                                         **kwargs)
        self.posterior_mean_gama = 1 + (np.sqrt(1.0 - self.betas) + self.alphas_cumprod_prev) * (
                self.sqrt_alphas_cumprod - 1) / (1.0 - self.alphas_cumprod)

    def _predict_xstart_from_eps(self, x_t, t, eps, condition=None):
        if condition is None:
            condition = torch.zeros_like(x_t)
        assert x_t.shape == eps.shape
        sqrt_recip_alphas_cumprod = self._extract_into_tensor(self.sqrt_recip_alphas_cumprod, t, x_t.shape)
        sqrt_alpha_bar_t = self._extract_into_tensor(self.sqrt_alphas_cumprod, t, x_t.shape)
        return (
                sqrt_recip_alphas_cumprod * x_t
                - (1 - sqrt_alpha_bar_t) * sqrt_recip_alphas_cumprod * condition
                - self._extract_into_tensor(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape) * eps
        )

    def q_sample(self, x_start, t, noise=None, mask=None, condition=None):
        """
        y_0_hat: prediction of pre-trained guidance classifier; can be extended to represent
            any prior mean setting at timestep T.
        """
        if noise is None:
            noise = torch.randn_like(x_start)
        if condition is None:
            condition = torch.zeros_like(x_start)
        assert noise.shape == x_start.shape
        sqrt_alpha_bar_t = self._extract_into_tensor(self.sqrt_alphas_cumprod, t, x_start.shape)
        x_t = (
                sqrt_alpha_bar_t * x_start
                + (1 - sqrt_alpha_bar_t) * condition
                + self._extract_into_tensor(self.sqrt_one_minus_alphas_cumprod, t, x_start.shape)
                * noise  ## reparameter trick
        )  ## genetrate x_t based on x_0 (x_start) with reparameter trick

        if mask == None:
            return x_t
        else:
            mask = torch.broadcast_to(mask.unsqueeze(dim=-1), x_start.shape)  ## mask: [0,0,0,1,1,1,1,1]
            return torch.where(mask == 0, x_start, x_t)  ## re

    def q_posterior_mean_variance(self, x_start, x_t, t, condition=None):
        """
        Compute the mean and variance of the diffusion posterior:
            q(x_{t-1} | x_t, x_0)

        """
        assert x_start.shape == x_t.shape
        if condition is None:
            condition = torch.zeros_like(x_start)
        posterior_mean = (
                self._extract_into_tensor(self.posterior_mean_coef1, t, x_t.shape) * x_start
                + self._extract_into_tensor(self.posterior_mean_coef2, t, x_t.shape) * x_t
                + self._extract_into_tensor(self.posterior_mean_gama, t, x_t.shape) * condition
        )  ## \mu_t
        assert (posterior_mean.shape[0] == x_start.shape[0])
        posterior_variance = self._extract_into_tensor(self.posterior_variance, t, x_t.shape)
        posterior_log_variance = self._extract_into_tensor(
            self.posterior_log_variance, t, x_t.shape
        )
        return posterior_mean, posterior_variance, posterior_log_variance


    def p_mean_variance(self, model_net, x_t, t, clip_denoised=False, **model_kwargs):
        condition = model_kwargs.get("condition", None)
        model_output = model_net(x_t, self._scale_timesteps(t), **model_kwargs)
        if self.diff_type == "mean":
            x_0 = model_output
        else:
            x_0 = self._predict_xstart_from_eps(x_t, t, model_output, condition=condition)  ## eps predict
        if clip_denoised:
            x_0 = x_0.clamp(-1, 1)
        ## x_start: candidante item embedding, x_t: inputseq_embedding + outseq_noise, output x_(t-1) distribution
        posterior_mean, _, model_log_variance = self.q_posterior_mean_variance(x_start=x_0,
                                                                               x_t=x_t,
                                                                               t=t, condition=condition)
        # model_log_variances = np.log(np.append(self.posterior_variance[1], self.betas[1:]))
        # model_log_variance = self._extract_into_tensor(model_log_variances, t, x_t.shape)
        return posterior_mean, model_log_variance

    # Reverse function -- sample y_{t-1} given y_t

    def reverse_p_sample(self, model_net, noise_x_t, others_index=None, clip_denoised=False, **model_kwargs):
        if others_index is None:
            others_index = []
        device = noise_x_t.device
        indices = list(range(self.num_timesteps))[::-1]
        noise_outs = []
        B = noise_x_t.shape[0]
        for i in indices:  # from T to 0, reversion iteration
            t = torch.tensor([i] * B, device=device)
            with torch.no_grad():
                noise_x_t,model_mean, model_log_variance = self.p_sample(model_net, noise_x_t, t, clip_denoised=clip_denoised, **model_kwargs)
                if i in others_index:
                    noise_outs.append(noise_x_t)
        return noise_x_t, noise_outs

    def forward_step(self, model_net, x, noise, **model_kwargs):
        condition = model_kwargs.get("condition", None)
        t, weights = self.sample_steps(x.shape[0], x.device)
        x_t = self.q_sample(x, t, noise=noise, condition=condition)
        model_output = model_net(x_t, self._scale_timesteps(t), **model_kwargs)
        if self.diff_type == "mean":
            x_0 = model_output
            pred_noise = x_t - x_0
        else:
            pred_noise = torch.clamp(model_output, -1, 1)
            x_0 = self._predict_xstart_from_eps(x_t, t, pred_noise, condition=condition)  ## eps predict
            # x_0 = x_t-pred_noise#self._predict_xstart_from_eps(x_t, t, pred_noise)
        return pred_noise, x_0
