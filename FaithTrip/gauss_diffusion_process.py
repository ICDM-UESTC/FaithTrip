import torch
import torch.nn as nn
import torch.nn.functional as F
from diffusion_model import TransformerLayer


def extract(v, t, x_shape):
    """
    Extract some coefficients at specified timesteps, then reshape to
    [batch_size, 1, 1, 1, 1, ...] for broadcasting purposes.
    """
    out = torch.gather(v, index=t, dim=0).float()

    return out.view([t.shape[0]] + [1] * (len(x_shape) - 1))

def generate_selected_list(time_step, selected_nums):

    # count spacing
    spacing = time_step / selected_nums
    selected_list = [int(spacing * i) for i in range(selected_nums)]
    return selected_list


class GaussianDiffusion(nn.Module):
    def __init__(self, beta_1, beta_T, T, theta, venue_vocab_size, hour_vocab_size,
                 max_length_venue_id=100, n_head=4, num_encoder_layers=4, d_model=128):
        super().__init__()

        # define constant variation
        self.T = T
        self.theta = theta
        self.venue_vocab_size = venue_vocab_size
        self.hour_vocab_size = hour_vocab_size
        self.register_buffer(
            'betas', torch.linspace(beta_1, beta_T, T).double())
        alphas = 1. - self.betas
        alphas_bar = torch.cumprod(alphas, dim=0)

        # calculations for diffusion q(x_t | x_{t-1}) and others
        self.register_buffer(
            'sqrt_alphas_bar', torch.sqrt(alphas_bar))
        self.register_buffer(
            'sqrt_one_minus_alphas_bar', torch.sqrt(1. - alphas_bar))

        # build model layer
        self.venue_embedding = nn.Embedding(self.venue_vocab_size, d_model).cuda()
        self.hour_embedding = nn.Embedding(self.hour_vocab_size, d_model).cuda()
        self.diffusion_model = TransformerLayer(T=self.T, max_seq_length=max_length_venue_id, d_model=d_model,
                                                n_head=n_head, num_encoder_layers=num_encoder_layers).cuda()
        self.criterion_rec = nn.CrossEntropyLoss(ignore_index=0)

    def forward(self, venue_input, masked_venue_input, hour_input, masked_hour_input):
        # generate embedding x_0 and x_ob
        # original embedding values
        x_0 = torch.cat([self.venue_embedding(venue_input), self.hour_embedding(hour_input)], dim=2)  # [b,l,2d]
        # observe embedding values
        x_ob = torch.cat([self.venue_embedding(masked_venue_input), self.hour_embedding(masked_hour_input)],
                         dim=2)  # [b,l,2d]

        # randomly sample t, x_0.shape[0] is batch_size
        t = torch.randint(self.T, size=(x_0.shape[0],), device=x_0.device)  # [b]
        epsilon = torch.randn_like(x_0)  # [b,l,2d]

        # counting x_t
        x_t_hat = (extract(self.sqrt_alphas_bar, t, x_0.shape) * x_0 +
                   extract(self.sqrt_one_minus_alphas_bar, t, x_0.shape) * epsilon)  # [b,l,2d]

        # add input to diffusion model
        x_0_hat = self.diffusion_model(x_t_hat, x_ob, t)  # [b,l,2d]

        # adding loss
        scores = torch.matmul(x_0_hat[:, :, :self.venue_embedding.weight.shape[1]], self.venue_embedding.weight.transpose(0, 1))  # [b,l,v]
        loss = self.criterion_rec(scores.view(-1, self.venue_vocab_size), venue_input.view(-1))

        return x_0_hat, loss


class Sampler(nn.Module):

    def __init__(self, diffusion_model, venue_embedding, hour_embedding, beta_1, beta_T, T, n_0,
                 gamma=0.1, lambdas=25, fast_sampling = 4, mean_type='xstart', var_type='fixedlarge'):

        assert mean_type in ['xprev', 'xstart', 'epsilon']
        assert var_type in ['fixedlarge', 'fixedsmall']

        super().__init__()

        # define constant variation
        self.diffusion_model = diffusion_model
        self.venue_embedding = venue_embedding
        self.hour_embedding = hour_embedding
        self.T = T
        self.n_0 = n_0
        self.gamma = gamma
        self.lambdas = lambdas
        self.fast_sampling = fast_sampling
        self.mean_type = mean_type
        self.var_type = var_type

        self.register_buffer(
            'betas', torch.linspace(beta_1, beta_T, T).double())
        alphas = 1. - self.betas
        alphas_bar = torch.cumprod(alphas, dim=0)
        alphas_bar_prev = F.pad(alphas_bar, [1, 0], value=1)[:T]

        # calculations for diffusion q(x_t | x_{t-1}) and others
        self.register_buffer(
            'sqrt_alphas_bar', torch.sqrt(alphas_bar))
        self.register_buffer(
            'sqrt_one_minus_alphas_bar', torch.sqrt(1. - alphas_bar))
        self.register_buffer(
            'sqrt_recip_alphas_bar', torch.sqrt(1. / alphas_bar))
        self.register_buffer(
            'sqrt_recipm1_alphas_bar', torch.sqrt(1. / alphas_bar - 1))

        # calculations for posterior q(x_{t-1} | x_t, x_0)
        self.register_buffer(
            'posterior_var',
            self.betas * (1. - alphas_bar_prev) / (1. - alphas_bar))
        # below: log calculation clipped because the posterior variance is 0 at
        # the beginning of the diffusion chain
        self.register_buffer(
            'posterior_log_var_clipped',
            torch.log(
                torch.cat([self.posterior_var[1:2], self.posterior_var[1:]])))
        self.register_buffer(
            'posterior_mean_coef1',
            torch.sqrt(alphas_bar_prev) * self.betas / (1. - alphas_bar))
        self.register_buffer(
            'posterior_mean_coef2',
            torch.sqrt(alphas) * (1. - alphas_bar_prev) / (1. - alphas_bar))

    def q_mean_variance(self, x_0, x_t, t):
        """
        Compute the mean and variance of the diffusion posterior
        q(x_{t-1} | x_t, x_0)
        """

        assert x_0.shape == x_t.shape
        posterior_mean = (
                extract(self.posterior_mean_coef1, t, x_t.shape) * x_0 +
                extract(self.posterior_mean_coef2, t, x_t.shape) * x_t
        )
        # change it to ddim
        posterior_log_var_clipped = extract(
            self.posterior_log_var_clipped, t, x_t.shape)

        return posterior_mean, posterior_log_var_clipped

    def predict_xstart_from_eps(self, x_t, t, eps):
        assert x_t.shape == eps.shape
        return (
                extract(self.sqrt_recip_alphas_bar, t, x_t.shape) * x_t -
                extract(self.sqrt_recipm1_alphas_bar, t, x_t.shape) * eps)

    def predict_xstart_from_xprev(self, x_t, t, xprev):
        assert x_t.shape == xprev.shape
        return (extract(1. / self.posterior_mean_coef1, t, x_t.shape) * xprev -
                extract(self.posterior_mean_coef2 / self.posterior_mean_coef1, t,
                        x_t.shape) * x_t)

    def p_mean_variance(self, x_t, x_ob, t):
        # below: only log_variance is used in the KL computations

        model_log_var = {
            # for fixedlarge, we set the initial (log-)variance like so to
            # get a better decoder log likelihood
            'fixedlarge': torch.log(torch.cat([self.posterior_var[1:2],
                                               self.betas[1:]])),
            'fixedsmall': self.posterior_log_var_clipped,
        }[self.var_type]
        log_var = model_log_var
        # exit()
        model_log_var = extract(model_log_var, t, x_t.shape)

        # Mean parameterization
        if self.mean_type == 'xprev':  # the model predicts x_{t-1}
            x_prev = self.diffusion_model(x_t, x_ob, t)
            x_0 = self.predict_xstart_from_xprev(x_t, t, xprev=x_prev)
            model_mean = x_prev
        elif self.mean_type == 'xstart':  # the model predicts x_0
            x_0_hat = self.diffusion_model(x_t, x_ob, t)
            model_mean, _ = self.q_mean_variance(x_0_hat, x_t, t)
        elif self.mean_type == 'epsilon':  # the model predicts epsilon
            eps = self.diffusion_model(x_t, x_ob, t)
            x_0 = self.predict_xstart_from_eps(x_t, t, eps=eps)
            model_mean, _ = self.q_mean_variance(x_0, x_t, t)
        else:
            raise NotImplementedError(self.mean_type)

        return model_mean, model_log_var, log_var

    def forward(self, masked_venue_input, masked_hour_input):

        # convert to masked embeddings:
        x_ob = torch.cat([self.venue_embedding(masked_venue_input), self.hour_embedding(masked_hour_input)],
                         dim=2)  # [b,l,2d]

        # counting mask(bool type)
        x_mask = (masked_venue_input != 0).unsqueeze(dim=2).repeat(1, 1, x_ob.shape[2])  # [b,l,2d]
        # define x_t, epsilon
        x_t, eps = torch.randn_like(x_ob), torch.randn_like(x_ob)  # N (0,I)  [b,l,2d]

        # TODO:
        time_list = generate_selected_list(self.T, self.fast_sampling)

        for i in range(len(time_list) - 1, -1, -1):
            time_step = time_list[i]
            # generate t
            t = x_t.new_ones([x_t.shape[0], ], dtype=torch.long) * time_step
            # generate x_ob_prev through forward process
            x_ob_prev = (extract(self.sqrt_alphas_bar, t, x_ob.shape) * x_ob +
                         extract(self.sqrt_one_minus_alphas_bar, t, x_ob.shape) * eps)  # [b,l,2d]

            mean, log_var, logvar_total = self.p_mean_variance(x_t=x_t, x_ob=x_ob, t=t)
            # no noise when t == 0
            if time_step > 0:
                noise = torch.randn_like(x_t)
            else:
                noise = 0
            # generate x_{t-1}_hat xt(30) (t=30 x_t(29))
            # next xt
            if i > 0:
                next_step = time_list[i - 1] + 1
            else:
                next_step = 0
            t = x_t.new_ones([x_t.shape[0], ], dtype=torch.long) * next_step
            x_t_prev = mean + torch.exp(0.5 * logvar_total[next_step]) * noise  # [b,l,2d]
            # TODO:generate x_t_prev
            x_t_prev = x_ob_prev * x_mask + x_t_prev * (~x_mask)
            # looping
            x_t = x_t_prev
        # final results
        x_0 = x_t

        """count the similarity of the results and venue_embedding_matrix"""
        # clip the time embeddings and only preserve poi embeddings
        clipped_x0 = x_0[:, :, :self.venue_embedding.weight.shape[1]]
        # normalized the input sequence and embedding weight
        x_0_normalized = F.normalize(clipped_x0, p=2, dim=2)
        venue_embedding_normalized = F.normalize(self.venue_embedding.weight, p=2, dim=1)

        # count the similarity matrix
        similarity_matrix = torch.matmul(x_0_normalized,
                                         venue_embedding_normalized.T)  # [b,l,v]

        # obtain the similar class label
        predicted_ids = torch.argmax(similarity_matrix, dim=2)  # [b,l]

        return predicted_ids
