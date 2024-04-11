import torch
from torch import nn
from torch.distributions import kl_divergence
from torch.distributions.normal import Normal


class NormalStochasticBlock2d(nn.Module):
    """
    Transform input parameters to q(z) with a convolution, optionally do the
    same for p(z), then sample z ~ q(z) and return conv(z).

    If q's parameters are not given, do the same but sample from p(z).
    """

    def __init__(self, c_in, c_vars, c_out, kernel=3, transform_p_params=True):
        super().__init__()
        assert kernel % 2 == 1
        pad = kernel // 2
        self.transform_p_params = transform_p_params
        self.c_in = c_in
        self.c_out = c_out
        self.c_vars = c_vars

        if transform_p_params:
            self.conv_in_p = nn.Conv3d(c_in,
                                       2 * c_vars,
                                       kernel,
                                       padding=pad,
                                       padding_mode='replicate')
        self.conv_in_q = nn.Conv3d(c_in,
                                   2 * c_vars,
                                   kernel,
                                   padding=pad,
                                   padding_mode='replicate')
        self.conv_out = nn.Conv3d(c_vars,
                                  c_out,
                                  kernel,
                                  padding=pad,
                                  padding_mode='replicate')

    def forward(self,
                p_params,
                q_params=None,
                forced_latent=None,
                use_mode=False,
                force_constant_output=False,
                mode_pred=False):

        assert (forced_latent is None) or (not use_mode)

        if self.transform_p_params:
            p_params = self.conv_in_p(p_params)
        else:
            assert p_params.size(1) == 2 * self.c_vars

        # Define p(z)
        p_mu, p_std_ = p_params.chunk(2, dim=1)
        p_std = nn.functional.softplus(p_std_)
        p = Normal(p_mu, p_std, validate_args=False)

        if q_params is not None:
            # Define q(z)
            q_params = self.conv_in_q(q_params)
            q_mu, q_std_ = q_params.chunk(2, dim=1)
            q_std = nn.functional.softplus(q_std_)
            q = Normal(q_mu, q_std, validate_args=False)

            # Sample from q(z)
            sampling_distrib = q
        else:
            # Sample from p(z)
            sampling_distrib = p

        # Generate latent variable (typically by sampling)
        if forced_latent is None:
            if use_mode:
                z = sampling_distrib.mean
            else:
                if mode_pred:
                    z = sampling_distrib.rsample()
                else:
                    z = sampling_distrib.rsample()
        else:
            z = forced_latent

        # Copy one sample (and distrib parameters) over the whole batch.
        # This is used when doing experiment from the prior - q is not used.
        if force_constant_output:
            z = z[0:1].expand_as(z).clone()

        # Output of stochastic layer
        z = self.conv_out(z)

        if q_params is not None:
            if mode_pred is False:
                # if not predicting, compute KL
                kl_elementwise = kl_divergence(q, p)
            else:
                # if predicting, no need to compute KL
                kl_elementwise = None
        else:
            kl_elementwise = None

        return z, kl_elementwise


class NormalStochasticBlock3d(nn.Module):
    """
    A 3D version of the NormalStochasticBlock, transforming input parameters
    to q(z) with a 3D convolution, optionally doing the same for p(z), then
    sampling z ~ q(z) and returning conv(z).

    If q's parameters are not given, do the same but sample from p(z).
    """

    def __init__(self, c_in, c_vars, c_out, kernel=3, transform_p_params=True):
        super().__init__()
        # Ensure kernel is odd for symmetric padding
        assert kernel % 2 == 1
        pad = kernel // 2
        self.transform_p_params = transform_p_params
        self.c_in = c_in
        self.c_out = c_out
        self.c_vars = c_vars

        # Convolutional layers for transforming parameters of p and q
        if transform_p_params:
            self.conv_in_p = nn.Conv3d(c_in, 2 * c_vars, kernel, padding=pad)
        self.conv_in_q = nn.Conv3d(c_in, 2 * c_vars, kernel, padding=pad)
        self.conv_out = nn.Conv3d(c_vars, c_out, kernel, padding=pad)

    def forward(self, p_params, q_params=None, forced_latent=None,
                use_mode=False, force_constant_output=False, mode_pred=False):

        assert (forced_latent is None) or (not use_mode)

        # Transform parameters of p if necessary
        if self.transform_p_params:
            p_params = self.conv_in_p(p_params)
        else:
            assert p_params.size(1) == 2 * self.c_vars

        # Define p(z)
        p_mu, p_std_ = p_params.chunk(2, dim=1)
        p_std = nn.functional.softplus(p_std_)
        p = Normal(p_mu, p_std)

        if q_params is not None:
            # Transform parameters of q and define q(z)
            q_params = self.conv_in_q(q_params)
            q_mu, q_std_ = q_params.chunk(2, dim=1)
            q_std = nn.functional.softplus(q_std_)
            q = Normal(q_mu, q_std)

            # Choose distribution to sample from
            sampling_distrib = q
        else:
            # Sample from p(z) if q_params are not provided
            sampling_distrib = p

        # Generate latent variable
        if forced_latent is None:
            if use_mode:
                z = sampling_distrib.mean
            else:
                z = sampling_distrib.rsample()
        else:
            z = forced_latent

        # Copy one sample over the whole batch if needed
        if force_constant_output:
            z = z[0:1].expand_as(z).clone()

        # Pass the latent variable through a convolutional layer
        z = self.conv_out(z)

        # Compute KL divergence if not in mode prediction and q_params is provided
        kl_elementwise = kl_divergence(q, p) if q_params is not None and not mode_pred else None

        return z, kl_elementwise