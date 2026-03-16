import torch
import torch.nn.functional as F

from ..modules.attention import *
from ..modules.diffusionmodules.util import AlphaBlender, linear, timestep_embedding


class TimeMixSequential(nn.Sequential):
    def forward(self, x, context=None, timesteps=None):
        for layer in self:
            x = layer(x, context, timesteps)

        return x


class VideoTransformerBlock(nn.Module):
    ATTENTION_MODES = {
        "softmax": CrossAttention,
        "softmax-xformers": MemoryEfficientCrossAttention,
    }

    def __init__(
        self,
        dim,
        n_heads,
        d_head,
        dropout=0.0,
        context_dim=None,
        gated_ff=True,
        checkpoint=True,
        timesteps=None,
        ff_in=False,
        inner_dim=None,
        attn_mode="softmax",
        disable_self_attn=False,
        disable_temporal_crossattention=False,
        switch_temporal_ca_to_sa=False,
    ):
        super().__init__()

        attn_cls = self.ATTENTION_MODES[attn_mode]

        self.ff_in = ff_in or inner_dim is not None
        if inner_dim is None:
            inner_dim = dim

        assert int(n_heads * d_head) == inner_dim

        self.is_res = inner_dim == dim

        if self.ff_in:
            self.norm_in = nn.LayerNorm(dim)
            self.ff_in = FeedForward(
                dim, dim_out=inner_dim, dropout=dropout, glu=gated_ff
            )

        self.timesteps = timesteps
        self.disable_self_attn = disable_self_attn
        if self.disable_self_attn:
            self.attn1 = attn_cls(
                query_dim=inner_dim,
                heads=n_heads,
                dim_head=d_head,
                context_dim=context_dim,
                dropout=dropout,
            )  # is a cross-attention
        else:
            self.attn1 = attn_cls(
                query_dim=inner_dim, heads=n_heads, dim_head=d_head, dropout=dropout
            )  # is a self-attention

        self.ff = FeedForward(inner_dim, dim_out=dim, dropout=dropout, glu=gated_ff)

        if disable_temporal_crossattention:
            if switch_temporal_ca_to_sa:
                raise ValueError
            else:
                self.attn2 = None
        else:
            self.norm2 = nn.LayerNorm(inner_dim)
            if switch_temporal_ca_to_sa:
                self.attn2 = attn_cls(
                    query_dim=inner_dim, heads=n_heads, dim_head=d_head, dropout=dropout
                )  # is a self-attention
            else:
                self.attn2 = attn_cls(
                    query_dim=inner_dim,
                    context_dim=context_dim,
                    heads=n_heads,
                    dim_head=d_head,
                    dropout=dropout,
                )  # is self-attn if context is none

        self.norm1 = nn.LayerNorm(inner_dim)
        self.norm3 = nn.LayerNorm(inner_dim)
        self.switch_temporal_ca_to_sa = switch_temporal_ca_to_sa

        self.checkpoint = checkpoint
        if self.checkpoint:
            print(f"{self.__class__.__name__} is using checkpointing")

    def forward(
        self, x: torch.Tensor, context: torch.Tensor = None, timesteps: int = None, context_instead_of_self_attn=None,
        use_reentrant=None
    ) -> torch.Tensor:
        # modified: getting context_instead_of_self_attn to enable full-attention for inpainting tokens

        if self.checkpoint:
            if use_reentrant is None:
                return checkpoint(self._forward, x, context, timesteps, context_instead_of_self_attn)
            else:
                return checkpoint(self._forward, x, context, timesteps, context_instead_of_self_attn, use_reentrant=use_reentrant)
        else:
            return self._forward(x, context, timesteps, context_instead_of_self_attn)

    def _forward(self, x, context=None, timesteps=None, context_instead_of_self_attn=None):

        assert self.timesteps or timesteps
        assert not (self.timesteps and timesteps) or self.timesteps == timesteps
        timesteps = self.timesteps or timesteps
        B, S, C = x.shape
        x = rearrange(x, "(b t) s c -> (b s) t c", t=timesteps)

        if self.ff_in:
            x_skip = x
            x = self.ff_in(self.norm_in(x))
            if self.is_res:
                x += x_skip

        context_1st_attn = None
        if self.disable_self_attn:
            context_1st_attn = context
        elif context_instead_of_self_attn is not None:
            context_1st_attn = context_instead_of_self_attn
        x = self.attn1(self.norm1(x), context=context_1st_attn) + x

        if self.attn2 is not None:
            if self.switch_temporal_ca_to_sa:
                x = self.attn2(self.norm2(x)) + x
            else:
                x = self.attn2(self.norm2(x), context=context) + x
        x_skip = x
        x = self.ff(self.norm3(x))
        if self.is_res:
            x += x_skip

        x = rearrange(
            x, "(b s) t c -> (b t) s c", s=S, b=B // timesteps, c=C, t=timesteps
        )
        return x

    def get_last_layer(self):
        return self.ff.net[-1].weight


def selected_attn(block, inpainting_mask_video, x_video, output_x_video, context_video, MAX_TOKENS, temporal=False):
    N = inpainting_mask_video.sum(dim=1).max().item()

    if N > MAX_TOKENS:
        N = MAX_TOKENS
        random_tokens = True
    else:
        random_tokens = False

    padded_x = torch.zeros(x_video.shape[0], N, x_video.shape[2], device=x_video.device, dtype=x_video.dtype)
    
    if random_tokens:
        selected_indices_list = []
        for i in range(x_video.shape[0]):
            non_zero_indices = torch.nonzero(inpainting_mask_video[i], as_tuple=False).squeeze(1)
            num_non_zero_tokens = non_zero_indices.size(0)

            # If num_non_zero_tokens > MAX_TOKENS, randomly select MAX_TOKENS indices
            if num_non_zero_tokens > MAX_TOKENS:
                selected_indices = non_zero_indices[torch.randperm(num_non_zero_tokens)[:MAX_TOKENS]]
            else:
                selected_indices = non_zero_indices
            selected_indices_list.append(selected_indices)
            selected_tokens = x_video[i, selected_indices]
            padded_x[i, :selected_tokens.size(0)] = selected_tokens
            padded_x[i, :selected_tokens.size(0)] = selected_tokens
    else:
        for i in range(x_video.shape[0]):
            selected_tokens = x_video[i, inpainting_mask_video[i]]  # Shape: (num_non_zero_tokens, feature_dim)
            padded_x[i, :selected_tokens.size(0)] = selected_tokens

    if temporal:
        padded_x = rearrange(padded_x, "b n m -> (b n) 1 m")
        kwargs = {'timesteps': N}
    else:
        kwargs = {}

    # print('padded_x', 'context_video', padded_x.shape, context_video.shape)
    processed_x = block(
        padded_x,
        context=context_video,
        context_instead_of_self_attn=x_video,
        use_reentrant=False,
        **kwargs
    )

    if temporal:
        processed_x = rearrange(processed_x, "(b n) 1 m -> b n m", n=N)

    for i in range(output_x_video.shape[0]):
        if random_tokens:
            selected_indices = selected_indices_list[i]
            mask = torch.zeros(output_x_video.shape[1], dtype=torch.bool, device=output_x_video.device)
            mask[selected_indices] = True
            size = selected_indices.size(0)
        else:
            mask = inpainting_mask_video[i]
            size = inpainting_mask_video[i].sum().item()
            selected_indices = torch.nonzero(mask, as_tuple=False).squeeze(1)
        output_x_video[i, mask] = processed_x[i, :size]


    return output_x_video


class SpatialVideoTransformer(SpatialTransformer):
    def __init__(
        self,
        in_channels,
        n_heads,
        d_head,
        depth=1,
        dropout=0.0,
        use_linear=False,
        context_dim=None,
        use_spatial_context=False,
        timesteps=None,
        merge_strategy: str = "fixed",
        merge_factor: float = 0.5,
        time_context_dim=None,
        ff_in=False,
        checkpoint=False,
        time_depth=1,
        attn_mode="softmax",
        disable_self_attn=False,
        disable_temporal_crossattention=False,
        max_time_embed_period: int = 10000,
        attn_inpainting_strategy = None,

    ):
        # modified: getting attn_inpainting_strategy to enable full-attention for inpainting tokens
        super().__init__(
            in_channels,
            n_heads,
            d_head,
            depth=depth,
            dropout=dropout,
            attn_type=attn_mode,
            use_checkpoint=checkpoint,
            context_dim=context_dim,
            use_linear=use_linear,
            disable_self_attn=disable_self_attn,
        )
        self.time_depth = time_depth
        self.depth = depth
        self.max_time_embed_period = max_time_embed_period
        self.attn_inpainting_strategy = attn_inpainting_strategy
        if attn_inpainting_strategy is not None:
            assert attn_inpainting_strategy in ['spatial_full_attention']


        time_mix_d_head = d_head
        n_time_mix_heads = n_heads

        time_mix_inner_dim = int(time_mix_d_head * n_time_mix_heads)

        inner_dim = n_heads * d_head
        if use_spatial_context:
            time_context_dim = context_dim

        self.time_stack = nn.ModuleList(
            [
                VideoTransformerBlock(
                    inner_dim,
                    n_time_mix_heads,
                    time_mix_d_head,
                    dropout=dropout,
                    context_dim=time_context_dim,
                    timesteps=timesteps,
                    checkpoint=checkpoint,
                    ff_in=ff_in,
                    inner_dim=time_mix_inner_dim,
                    attn_mode=attn_mode,
                    disable_self_attn=disable_self_attn,
                    disable_temporal_crossattention=disable_temporal_crossattention,
                )
                for _ in range(self.depth)
            ]
        )

        assert len(self.time_stack) == len(self.transformer_blocks)

        self.use_spatial_context = use_spatial_context
        self.in_channels = in_channels

        time_embed_dim = self.in_channels * 4
        self.time_pos_embed = nn.Sequential(
            linear(self.in_channels, time_embed_dim),
            nn.SiLU(),
            linear(time_embed_dim, self.in_channels),
        )

        self.time_mixer = AlphaBlender(
            alpha=merge_factor, merge_strategy=merge_strategy
        )

    def forward(
        self,
        x: torch.Tensor,
        context: Optional[torch.Tensor] = None,
        time_context: Optional[torch.Tensor] = None,
        timesteps: Optional[int] = None,
        image_only_indicator: Optional[torch.Tensor] = None,
        inpainting_mask=None,

    ) -> torch.Tensor:
        # modified: getting inpainting_mask to enable full-attention for inpainting tokens

        _, _, h, w = x.shape
        x_in = x
        spatial_context = None
        if exists(context):
            spatial_context = context

        if self.use_spatial_context:
            assert (
                context.ndim == 3
            ), f"n dims of spatial context should be 3 but are {context.ndim}"

            time_context = context
            time_context_first_timestep = time_context[::timesteps]
            time_context = repeat(
                time_context_first_timestep, "b ... -> (b n) ...", n=h * w
            )
        elif time_context is not None and not self.use_spatial_context:
            time_context = repeat(time_context, "b ... -> (b n) ...", n=h * w)
            if time_context.ndim == 2:
                time_context = rearrange(time_context, "b c -> b 1 c")

        x = self.norm(x)
        if not self.use_linear:
            x = self.proj_in(x)
        x = rearrange(x, "b c h w -> b (h w) c")
        if self.use_linear:
            x = self.proj_in(x)

        num_frames = torch.arange(timesteps, device=x.device)
        num_frames = repeat(num_frames, "t -> b t", b=x.shape[0] // timesteps)
        num_frames = rearrange(num_frames, "b t -> (b t)")
        t_emb = timestep_embedding(
            num_frames,
            self.in_channels,
            repeat_only=False,
            max_period=self.max_time_embed_period,
        )
        emb = self.time_pos_embed(t_emb)
        emb = emb[:, None, :]

        # modified:
        if self.attn_inpainting_strategy is not None:
            assert inpainting_mask is not None
            inpainting_mask = F.interpolate(inpainting_mask.unsqueeze(1), size=(h, w), mode='bilinear', align_corners=False).squeeze(1)
            inpainting_mask = rearrange(inpainting_mask, "b h w -> b (h w)")
            inpainting_mask_video = rearrange(inpainting_mask, "(b f) n -> b (f n)", f=timesteps)
            inpainting_mask_video = inpainting_mask_video > 0
            spatial_context_video = rearrange(spatial_context, "(b f) n m -> b (f n) m", f=timesteps)

            if self.use_spatial_context:
                time_context_video = spatial_context_video
            else:
                raise NotImplementedError

            if self.training:
                MAX_TOKENS = 512 * timesteps
            else:
                MAX_TOKENS = 10 ** 20
            N = inpainting_mask_video.sum(dim=1).max().item()

        for it_, (block, mix_block) in enumerate(
            zip(self.transformer_blocks, self.time_stack)
        ):
            # modified: ---------------------
            output_x = block(
                x,
                context=spatial_context,
                use_reentrant=False if (self.attn_inpainting_strategy == 'spatial_full_attention') else None,
            )

            if self.attn_inpainting_strategy == 'spatial_full_attention' and N > 0:
                x_video = rearrange(x, "(b f) n m -> b (f n) m", f=timesteps)
                output_x = rearrange(output_x, "(b f) n m -> b (f n) m", f=timesteps)
                output_x = selected_attn(block, inpainting_mask_video, x_video, output_x, spatial_context_video, 
                                         MAX_TOKENS)
                output_x = rearrange(output_x, "b (f n) m -> (b f) n m", f=timesteps)

            x = output_x
            x_mix = x + emb
            # ---------------------


            x_mix = mix_block(x_mix, context=time_context, timesteps=timesteps)

            x = self.time_mixer(
                x_spatial=x,
                x_temporal=x_mix,
                image_only_indicator=image_only_indicator,
            )
        if self.use_linear:
            x = self.proj_out(x)
        x = rearrange(x, "b (h w) c -> b c h w", h=h, w=w)
        if not self.use_linear:
            x = self.proj_out(x)
        out = x + x_in
        return out
