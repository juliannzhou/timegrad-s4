from torch.nn.modules import loss
from typing import List, Optional, Tuple, Union

import torch
import torch.nn as nn

from gluonts.core.component import validated
from utils import weighted_average,MeanScaler,NOPScaler
from module import GaussianDiffusion,DiffusionOutput

from epsilon_theta import EpsilonTheta

class TimeGradTrainingNetwork(nn.Module):
    @validated()
    def __init__(
        self,
        input_size: int, # 输入维度
        num_layers: int, # rnn层数
        num_cells: int, # rnn cell数
        cell_type: str, # rnn cell类型 lstm or gru
        history_length: int, # 历史长度 24+168=192
        context_length: int, # 上下文长度 24=预测长度
        prediction_length: int,# 预测长度 24
        dropout_rate: float, # dropout rate
        lags_seq: List[int], # lag序列 [1,24,168]
        target_dim: int, # 目标序列维度 1
        conditioning_length: int, # 条件长度 100
        diff_steps: int, # diffusion steps 100
        loss_type: str, # 损失函数类型 l2
        beta_end: float, # beta_end 0.1
        beta_schedule: str, # linear or cosine
        residual_layers: int, # 残差层数 8
        residual_channels: int, # 残差通道数 8
        dilation_cycle_length: int,# 膨胀周期长度 2
        cardinality: List[int] = [1],
        embedding_dimension: int = 1,
        scaling: bool = True,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        self.target_dim = target_dim # 目标序列维度
        self.prediction_length = prediction_length # 预测长度
        self.context_length = context_length 
        self.history_length = history_length
        self.scaling = scaling

        assert len(set(lags_seq)) == len(lags_seq), "no duplicated lags allowed!"
        lags_seq.sort()
        self.lags_seq = lags_seq

        self.cell_type = cell_type
        rnn_cls = {"LSTM": nn.LSTM, "GRU": nn.GRU}[cell_type] # rnn class
        self.rnn = S4Layer(
            features=input_size,
            lmax=context_length + prediction_length,
            N=num_cells,  # Number of state dimensions
            dropout=dropout_rate
        )

        self.denoise_fn = EpsilonTheta(
            target_dim=target_dim,
            cond_length=conditioning_length,
            residual_layers=residual_layers,
            residual_channels=residual_channels,
            dilation_cycle_length=dilation_cycle_length,
        ) #去噪网络 

        self.diffusion = GaussianDiffusion(
            self.denoise_fn,
            input_size=target_dim,
            diff_steps=diff_steps,
            loss_type=loss_type,
            beta_end=beta_end,
            beta_schedule=beta_schedule,
        ) # diffusion网络

        self.distr_output = DiffusionOutput(
            self.diffusion, input_size=target_dim, cond_size=conditioning_length
        ) # 分布输出

        self.proj_dist_args = self.distr_output.get_args_proj(num_cells) # 分布参数投影

        self.embed_dim = 1
        self.embed = nn.Embedding(
            num_embeddings=self.target_dim, embedding_dim=self.embed_dim
        )

        if self.scaling:
            self.scaler = MeanScaler(keepdim=True) # 均值归一化
        else:
            self.scaler = NOPScaler(keepdim=True)

    @staticmethod
    def get_lagged_subsequences(
        sequence: torch.Tensor,
        sequence_length: int,
        indices: List[int],
        subsequences_length: int = 1,
    ) -> torch.Tensor:
        """
        Returns lagged subsequences of a given sequence.
        Parameters
        ----------
        sequence
            the sequence from which lagged subsequences should be extracted.
            Shape: (N, T, C).
        sequence_length
            length of sequence in the T (time) dimension (axis = 1).
        indices
            list of lag indices to be used.
            eg: [1,24,168]
        subsequences_length
            length of the subsequences to be extracted.
        Returns
        --------
        lagged : Tensor
            a tensor of shape (N, S, C, I),
            where S = subsequences_length and I = len(indices),
            containing lagged subsequences.
            Specifically, lagged[i, :, j, k] = sequence[i, -indices[k]-S+j, :].
        """
        # we must have: history_length + begin_index >= 0
        # that is: history_length - lag_index - sequence_length >= 0
        # hence the following assert
        assert max(indices) + subsequences_length <= sequence_length, (
            f"lags cannot go further than history length, found lag "
            f"{max(indices)} while history length is only {sequence_length}"
        )
        assert all(lag_index >= 0 for lag_index in indices)

        lagged_values = []
        for lag_index in indices:
            begin_index = -lag_index - subsequences_length 
            end_index = -lag_index if lag_index > 0 else None
            lagged_values.append(sequence[:, begin_index:end_index, ...].unsqueeze(1)) # shape: (batch_size, 1, sub_seq_len, C)
        return torch.cat(lagged_values, dim=1).permute(0, 2, 3, 1) #    shape: (batch_size, sub_seq_len, C, I) I = len(indices)=3
    

    def unroll(
        self,
        lags: torch.Tensor,
        scale: torch.Tensor,
        time_feat: torch.Tensor,
        target_dimension_indicator: torch.Tensor,
        unroll_length: int,
        begin_state: Optional[Union[List[torch.Tensor], torch.Tensor]] = None,
    ) -> Tuple[
        torch.Tensor,
        Union[List[torch.Tensor], torch.Tensor],
        torch.Tensor,
        torch.Tensor,
    ]:
        """

        Args:
            lags (torch.Tensor): lagged sub-sequences
            scale (torch.Tensor): 归一化
            time_feat (torch.Tensor): _description_
            target_dimension_indicator (torch.Tensor): _description_
            unroll_length (int): _description_
            begin_state (Optional[Union[List[torch.Tensor], torch.Tensor]], optional): _description_. Defaults to None.

        Returns:
            Tuple[ torch.Tensor, Union[List[torch.Tensor], torch.Tensor], torch.Tensor, torch.Tensor, ]: _description_
        """

        # (batch_size, sub_seq_len, target_dim, num_lags)
        lags_scaled = lags / scale.unsqueeze(-1)
        # 对lagged 数组进行归一化

        # assert_shape(
        #     lags_scaled, (-1, unroll_length, self.target_dim, len(self.lags_seq)),
        # )

        input_lags = lags_scaled.reshape(
            (-1, unroll_length, len(self.lags_seq) * self.target_dim)
        ) 
        

        # (batch_size, target_dim, embed_dim)
        index_embeddings = self.embed(target_dimension_indicator)
        # assert_shape(index_embeddings, (-1, self.target_dim, self.embed_dim))

        # (batch_size, seq_len, target_dim * embed_dim)
        repeated_index_embeddings = (
            index_embeddings.unsqueeze(1)
            .expand(-1, unroll_length, -1, -1)
            .reshape((-1, unroll_length, self.target_dim * self.embed_dim))
        )

        # (batch_size, sub_seq_len, input_dim)
        inputs = torch.cat((input_lags, repeated_index_embeddings, time_feat), dim=-1)

        # unroll encoder
        outputs, state = self.rnn(inputs, begin_state)

        # assert_shape(outputs, (-1, unroll_length, self.num_cells))
        # for s in state:
        #     assert_shape(s, (-1, self.num_cells))

        # assert_shape(
        #     lags_scaled, (-1, unroll_length, self.target_dim, len(self.lags_seq)),
        # )

        return outputs, state, lags_scaled, inputs

    def unroll_encoder(
        self,
        past_time_feat: torch.Tensor,
        past_target_cdf: torch.Tensor,
        past_observed_values: torch.Tensor,
        past_is_pad: torch.Tensor,
        future_time_feat: Optional[torch.Tensor],
        future_target_cdf: Optional[torch.Tensor],
        target_dimension_indicator: torch.Tensor,
    ) -> Tuple[
        torch.Tensor,
        Union[List[torch.Tensor], torch.Tensor],
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
    ]:
        """
        Unrolls the RNN encoder over past and, if present, future data.
        Returns outputs and state of the encoder, plus the scale of
        past_target_cdf and a vector of static features that was constructed
        and fed as input to the encoder. All tensor arguments should have NTC
        layout.

        Parameters
        ----------
        past_time_feat
            Past time features (batch_size, history_length, num_features)
        past_target_cdf
            Past marginal CDF transformed target values (batch_size,
            history_length, target_dim)
        past_observed_values
            Indicator whether or not the values were observed (batch_size,
            history_length, target_dim)
        past_is_pad
            Indicator whether the past target values have been padded
            (batch_size, history_length)
        future_time_feat
            Future time features (batch_size, prediction_length, num_features)
        future_target_cdf
            Future marginal CDF transformed target values (batch_size,
            prediction_length, target_dim)
        target_dimension_indicator
            Dimensionality of the time series (batch_size, target_dim)

        Returns
        -------
        outputs
            RNN outputs (batch_size, seq_len, num_cells)
        states
            RNN states. Nested list with (batch_size, num_cells) tensors with
        dimensions target_dim x num_layers x (batch_size, num_cells)
        scale
            Mean scales for the time series (batch_size, 1, target_dim)
        lags_scaled
            Scaled lags(batch_size, sub_seq_len, target_dim, num_lags)
        inputs
            inputs to the RNN

        """

        past_observed_values = torch.min(
            past_observed_values, 1 - past_is_pad.unsqueeze(-1)
        )

        if future_time_feat is None or future_target_cdf is None:
            time_feat = past_time_feat[:, -self.context_length :, ...]
            sequence = past_target_cdf
            sequence_length = self.history_length
            subsequences_length = self.context_length
        else:
            time_feat = torch.cat(
                (past_time_feat[:, -self.context_length :, ...], future_time_feat),
                dim=1,
            )
            sequence = torch.cat((past_target_cdf, future_target_cdf), dim=1)
            sequence_length = self.history_length + self.prediction_length
            subsequences_length = self.context_length + self.prediction_length

        # (batch_size, sub_seq_len, target_dim, num_lags)
        lags = self.get_lagged_subsequences(
            sequence=sequence,
            sequence_length=sequence_length,
            indices=self.lags_seq,
            subsequences_length=subsequences_length,
        )

        # scale is computed on the context length last units of the past target
        # scale shape is (batch_size, 1, target_dim)
        _, scale = self.scaler(
            past_target_cdf[:, -self.context_length :, ...],
            past_observed_values[:, -self.context_length :, ...],
        )

        outputs, states, lags_scaled, inputs = self.unroll(
            lags=lags,
            scale=scale,
            time_feat=time_feat,
            target_dimension_indicator=target_dimension_indicator,
            unroll_length=subsequences_length,
            begin_state=None,
        )

        return outputs, states, scale, lags_scaled, inputs

    def distr_args(self, rnn_outputs: torch.Tensor):
        """
        Returns the distribution of DeepVAR with respect to the RNN outputs.

        Parameters
        ----------
        rnn_outputs
            Outputs of the unrolled RNN (batch_size, seq_len, num_cells)
        scale
            Mean scale for each time series (batch_size, 1, target_dim)

        Returns
        -------
        distr
            Distribution instance
        distr_args
            Distribution arguments
        """
        (distr_args,) = self.proj_dist_args(rnn_outputs)

        # # compute likelihood of target given the predicted parameters
        # distr = self.distr_output.distribution(distr_args, scale=scale)

        # return distr, distr_args
        return distr_args

    def forward(
        self,
        target_dimension_indicator: torch.Tensor,
        past_time_feat: torch.Tensor,
        past_target_cdf: torch.Tensor,
        past_observed_values: torch.Tensor,
        past_is_pad: torch.Tensor,
        future_time_feat: torch.Tensor,
        future_target_cdf: torch.Tensor,
        future_observed_values: torch.Tensor,
    ) -> Tuple[torch.Tensor, ...]:
        """
        Computes the loss for training DeepVAR, all inputs tensors representing
        time series have NTC layout.

        Parameters
        ----------
        target_dimension_indicator
            Indices of the target dimension (batch_size, target_dim)
        past_time_feat
            Dynamic features of past time series (batch_size, history_length,
            num_features)
        past_target_cdf
            Past marginal CDF transformed target values (batch_size,
            history_length, target_dim)
        past_observed_values
            Indicator whether or not the values were observed (batch_size,
            history_length, target_dim)
        past_is_pad
            Indicator whether the past target values have been padded
            (batch_size, history_length)
        future_time_feat
            Future time features (batch_size, prediction_length, num_features)
        future_target_cdf
            Future marginal CDF transformed target values (batch_size,
            prediction_length, target_dim)
        future_observed_values
            Indicator whether or not the future values were observed
            (batch_size, prediction_length, target_dim)

        Returns
        -------
        distr
            Loss with shape (batch_size, 1)
        likelihoods
            Likelihoods for each time step
            (batch_size, context + prediction_length, 1)
        distr_args
            Distribution arguments (context + prediction_length,
            number_of_arguments)
        """

        seq_len = self.context_length + self.prediction_length

        # unroll the decoder in "training mode", i.e. by providing future data
        # as well
        rnn_outputs, _, scale, _, _ = self.unroll_encoder(
            past_time_feat=past_time_feat,
            past_target_cdf=past_target_cdf,
            past_observed_values=past_observed_values,
            past_is_pad=past_is_pad,
            future_time_feat=future_time_feat,
            future_target_cdf=future_target_cdf,
            target_dimension_indicator=target_dimension_indicator,
        )

        # put together target sequence
        # (batch_size, seq_len, target_dim)
        target = torch.cat(
            (past_target_cdf[:, -self.context_length :, ...], future_target_cdf),
            dim=1,
        )

        # assert_shape(target, (-1, seq_len, self.target_dim))

        distr_args = self.distr_args(rnn_outputs=rnn_outputs)
        if self.scaling:
            self.diffusion.scale = scale

        # we sum the last axis to have the same shape for all likelihoods
        # (batch_size, subseq_length, 1)

        likelihoods = self.diffusion.log_prob(target, distr_args).unsqueeze(-1)

        # assert_shape(likelihoods, (-1, seq_len, 1))

        past_observed_values = torch.min(
            past_observed_values, 1 - past_is_pad.unsqueeze(-1)
        )

        # (batch_size, subseq_length, target_dim)
        observed_values = torch.cat(
            (
                past_observed_values[:, -self.context_length :, ...],
                future_observed_values,
            ),
            dim=1,
        )

        # mask the loss at one time step if one or more observations is missing
        # in the target dimensions (batch_size, subseq_length, 1)
        loss_weights, _ = observed_values.min(dim=-1, keepdim=True)

        # assert_shape(loss_weights, (-1, seq_len, 1))

        loss = weighted_average(likelihoods, weights=loss_weights, dim=1)

        # assert_shape(loss, (-1, -1, 1))

        # self.distribution = distr

        return (loss.mean(), likelihoods, distr_args)


class TimeGradPredictionNetwork(TimeGradTrainingNetwork):
    def __init__(self, num_parallel_samples: int, **kwargs) -> None:
        super().__init__(**kwargs)
        self.num_parallel_samples = num_parallel_samples

        # for decoding the lags are shifted by one,
        # at the first time-step of the decoder a lag of one corresponds to
        # the last target value
        self.shifted_lags = [l - 1 for l in self.lags_seq]

    def sampling_decoder(
        self,
        past_target_cdf: torch.Tensor,
        target_dimension_indicator: torch.Tensor,
        time_feat: torch.Tensor,
        scale: torch.Tensor,
        begin_states: Union[List[torch.Tensor], torch.Tensor],
    ) -> torch.Tensor:
        """
        Computes sample paths by unrolling the RNN starting with a initial
        input and state.

        Parameters
        ----------
        past_target_cdf
            Past marginal CDF transformed target values (batch_size,
            history_length, target_dim)
        target_dimension_indicator
            Indices of the target dimension (batch_size, target_dim)
        time_feat
            Dynamic features of future time series (batch_size, history_length,
            num_features)
        scale
            Mean scale for each time series (batch_size, 1, target_dim)
        begin_states
            List of initial states for the RNN layers (batch_size, num_cells)
        Returns
        --------
        sample_paths : Tensor
            A tensor containing sampled paths. Shape: (1, num_sample_paths,
            prediction_length, target_dim).
        """

        def repeat(tensor, dim=0):
            return tensor.repeat_interleave(repeats=self.num_parallel_samples, dim=dim)

        # blows-up the dimension of each tensor to
        # batch_size * self.num_sample_paths for increasing parallelism
        repeated_past_target_cdf = repeat(past_target_cdf)
        repeated_time_feat = repeat(time_feat)
        repeated_scale = repeat(scale)
        if self.scaling:
            self.diffusion.scale = repeated_scale
        repeated_target_dimension_indicator = repeat(target_dimension_indicator)

        if self.cell_type == "LSTM":
            repeated_states = [repeat(s, dim=1) for s in begin_states]
        else:
            repeated_states = repeat(begin_states, dim=1)

        future_samples = []

        # for each future time-units we draw new samples for this time-unit
        # and update the state
        for k in range(self.prediction_length):
            lags = self.get_lagged_subsequences(
                sequence=repeated_past_target_cdf,
                sequence_length=self.history_length + k,
                indices=self.shifted_lags,
                subsequences_length=1,
            )

            rnn_outputs, repeated_states, _, _ = self.unroll(
                begin_state=repeated_states,
                lags=lags,
                scale=repeated_scale,
                time_feat=repeated_time_feat[:, k : k + 1, ...],
                target_dimension_indicator=repeated_target_dimension_indicator,
                unroll_length=1,
            )

            distr_args = self.distr_args(rnn_outputs=rnn_outputs)

            # (batch_size, 1, target_dim)
            new_samples = self.diffusion.sample(cond=distr_args)

            # (batch_size, seq_len, target_dim)
            future_samples.append(new_samples)
            repeated_past_target_cdf = torch.cat(
                (repeated_past_target_cdf, new_samples), dim=1
            )

        # (batch_size * num_samples, prediction_length, target_dim)
        samples = torch.cat(future_samples, dim=1)

        # (batch_size, num_samples, prediction_length, target_dim)
        return samples.reshape(
            (
                -1,
                self.num_parallel_samples,
                self.prediction_length,
                self.target_dim,
            )
        )

    def forward(
        self,
        target_dimension_indicator: torch.Tensor,
        past_time_feat: torch.Tensor,
        past_target_cdf: torch.Tensor,
        past_observed_values: torch.Tensor,
        past_is_pad: torch.Tensor,
        future_time_feat: torch.Tensor,
    ) -> torch.Tensor:
        """
        Predicts samples given the trained DeepVAR model.
        All tensors should have NTC layout.
        Parameters
        ----------
        target_dimension_indicator
            Indices of the target dimension (batch_size, target_dim)
        past_time_feat
            Dynamic features of past time series (batch_size, history_length,
            num_features)
        past_target_cdf
            Past marginal CDF transformed target values (batch_size,
            history_length, target_dim)
        past_observed_values
            Indicator whether or not the values were observed (batch_size,
            history_length, target_dim)
        past_is_pad
            Indicator whether the past target values have been padded
            (batch_size, history_length)
        future_time_feat
            Future time features (batch_size, prediction_length, num_features)

        Returns
        -------
        sample_paths : Tensor
            A tensor containing sampled paths (1, num_sample_paths,
            prediction_length, target_dim).

        """

        # mark padded data as unobserved
        # (batch_size, target_dim, seq_len)
        past_observed_values = torch.min(
            past_observed_values, 1 - past_is_pad.unsqueeze(-1)
        )

        # unroll the decoder in "prediction mode", i.e. with past data only
        _, begin_states, scale, _, _ = self.unroll_encoder(
            past_time_feat=past_time_feat,
            past_target_cdf=past_target_cdf,
            past_observed_values=past_observed_values,
            past_is_pad=past_is_pad,
            future_time_feat=None,
            future_target_cdf=None,
            target_dimension_indicator=target_dimension_indicator,
        )

        return self.sampling_decoder(
            past_target_cdf=past_target_cdf,
            target_dimension_indicator=target_dimension_indicator,
            time_feat=future_time_feat,
            scale=scale,
            begin_states=begin_states,
        )




class S4(nn.Module):

    def __init__(
            self,
            d_model,
            d_state=64,
            l_max=1, # Maximum length of sequence. Fine if not provided: the kernel will keep doubling in length until longer than sequence. However, this can be marginally slower if the true length is not a power of 2
            channels=1, # maps 1-dim to C-dim
            bidirectional=False,
            # Arguments for FF
            activation='gelu', # activation in between SS and FF
            postact=None, # activation after FF
            initializer=None, # initializer on FF
            weight_norm=False, # weight normalization on FF
            hyper_act=None, # Use a "hypernetwork" multiplication
            dropout=0.0,
            transposed=True, # axis ordering (B, L, D) or (B, D, L)
            verbose=False,
            # SSM Kernel arguments
            **kernel_args,
        ):


        """
        d_state: the dimension of the state, also denoted by N
        l_max: the maximum sequence length, also denoted by L
          if this is not known at model creation, set l_max=1
        channels: can be interpreted as a number of "heads"
        bidirectional: bidirectional
        dropout: standard dropout argument
        transposed: choose backbone axis ordering of (B, L, H) or (B, H, L) [B=batch size, L=sequence length, H=hidden dimension]

        Other options are all experimental and should not need to be configured
        """


        super().__init__()
        if verbose:
            import src.utils.train
            log = src.utils.train.get_logger(__name__)
            log.info(f"Constructing S4 (H, N, L) = ({d_model}, {d_state}, {l_max})")

        self.h = d_model
        self.n = d_state
        self.bidirectional = bidirectional
        self.channels = channels
        self.transposed = transposed

        # optional multiplicative modulation GLU-style
        # https://arxiv.org/abs/2002.05202
        self.hyper = hyper_act is not None
        if self.hyper:
            channels *= 2
            self.hyper_activation = Activation(hyper_act)

        self.D = nn.Parameter(torch.randn(channels, self.h))

        if self.bidirectional:
            channels *= 2


        # SSM Kernel
        self.kernel = HippoSSKernel(self.h, N=self.n, L=l_max, channels=channels, verbose=verbose, **kernel_args)

        # Pointwise
        self.activation = Activation(activation)
        dropout_fn = nn.Dropout2d if self.transposed else nn.Dropout
        self.dropout = dropout_fn(dropout) if dropout > 0.0 else nn.Identity()

        # position-wise output transform to mix features
        self.output_linear = LinearActivation(
            self.h*self.channels,
            self.h,
            transposed=self.transposed,
            initializer=initializer,
            activation=postact,
            activate=True,
            weight_norm=weight_norm,
        )


    def forward(self, u, **kwargs): # absorbs return_output and transformer src mask
        """
        u: (B H L) if self.transposed else (B L H)
        state: (H N) never needed unless you know what you're doing

        Returns: same shape as u
        """
        if not self.transposed: u = u.transpose(-1, -2)
        L = u.size(-1)

        # Compute SS Kernel
        k = self.kernel(L=L) # (C H L) (B C H L)

        # Convolution
        if self.bidirectional:
            k0, k1 = rearrange(k, '(s c) h l -> s c h l', s=2)
            k = F.pad(k0, (0, L)) \
                    + F.pad(k1.flip(-1), (L, 0)) \

        k_f = torch.fft.rfft(k, n=2*L) # (C H L)
        u_f = torch.fft.rfft(u, n=2*L) # (B H L)
        y_f = contract('bhl,chl->bchl', u_f, k_f) # k_f.unsqueeze(-4) * u_f.unsqueeze(-3) # (B C H L)
        y = torch.fft.irfft(y_f, n=2*L)[..., :L] # (B C H L)


        # Compute D term in state space equation - essentially a skip connection
        y = y + contract('bhl,ch->bchl', u, self.D) # u.unsqueeze(-3) * self.D.unsqueeze(-1)

        # Optional hyper-network multiplication
        if self.hyper:
            y, yh = rearrange(y, 'b (s c) h l -> s b c h l', s=2)
            y = self.hyper_activation(yh) * y

        # Reshape to flatten channels
        y = rearrange(y, '... c h l -> ... (c h) l')

        y = self.dropout(self.activation(y))

        if not self.transposed: y = y.transpose(-1, -2)

        y = self.output_linear(y)

        return y, None
        

    def step(self, u, state):
        """ Step one time step as a recurrent model. Intended to be used during validation.

        u: (B H)
        state: (B H N)
        Returns: output (B H), state (B H N)
        """
        assert not self.training

        y, next_state = self.kernel.step(u, state) # (B C H)
        y = y + u.unsqueeze(-2) * self.D
        y = rearrange(y, '... c h -> ... (c h)')
        y = self.activation(y)
        if self.transposed:
            y = self.output_linear(y.unsqueeze(-1)).squeeze(-1)
        else:
            y = self.output_linear(y)
        return y, next_state

    def default_state(self, *batch_shape, device=None):
        return self.kernel.default_state(*batch_shape)

    @property
    def d_state(self):
        return self.h * self.n

    @property
    def d_output(self):
        return self.h

    @property
    def state_to_tensor(self):
        return lambda state: rearrange('... h n -> ... (h n)', state)




class S4Layer(nn.Module):
    '''S4 Layer that can be used as a drop-in replacement for a TransformerEncoder'''
    def __init__(self, features, lmax, N=64, dropout=0.0,layer_norm=True):
        super().__init__()
        self.s4_layer  = S4(d_model=features, 
                            d_state=N, 
                            l_max=251, 
                            bidirectional=True)
        
        self.norm_layer = nn.LayerNorm(features) if layer_norm else nn.Identity() 
        self.dropout = nn.Dropout2d(dropout) if dropout>0 else nn.Identity()
    
    def forward(self, x):
        #x has shape seq, batch, feature
        xin = x.permute((1,2,0)) #batch, feature, seq (as expected from S4 with transposed=True)
        
        xout, _ = self.s4_layer(xin) #batch, feature, seq
        xout = self.dropout(xout)
        xout = xout + xin # skip connection   # batch, feature, seq
        xout = xout.permute((2,0,1)) # seq, batch, feature
        return self.norm_layer(xout)


