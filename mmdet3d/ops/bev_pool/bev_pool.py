import torch

from . import bev_pool_ext

__all__ = ["bev_pool", "bev_pool_v2"]


class QuickCumsum(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, geom_feats, ranks):
        x = x.cumsum(0)
        kept = torch.ones(x.shape[0], device=x.device, dtype=torch.bool)
        kept[:-1] = ranks[1:] != ranks[:-1]

        x, geom_feats = x[kept], geom_feats[kept]
        x = torch.cat((x[:1], x[1:] - x[:-1]))

        # save kept for backward
        ctx.save_for_backward(kept)

        # no gradient for geom_feats
        ctx.mark_non_differentiable(geom_feats)

        return x, geom_feats

    @staticmethod
    def backward(ctx, gradx, gradgeom):
        (kept,) = ctx.saved_tensors
        back = torch.cumsum(kept, 0)
        back[kept] -= 1

        val = gradx[back]

        return val, None, None


class QuickCumsumCuda(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, geom_feats, ranks, B, D, H, W):
        kept = torch.ones(x.shape[0], device=x.device, dtype=torch.bool)
        kept[1:] = ranks[1:] != ranks[:-1]
        interval_starts = torch.where(kept)[0].int()
        interval_lengths = torch.zeros_like(interval_starts)
        interval_lengths[:-1] = interval_starts[1:] - interval_starts[:-1]
        interval_lengths[-1] = x.shape[0] - interval_starts[-1]
        geom_feats = geom_feats.int()

        out = bev_pool_ext.bev_pool_forward(
            x,
            geom_feats,
            interval_lengths,
            interval_starts,
            B,
            D,
            H,
            W,
        )

        ctx.save_for_backward(interval_starts, interval_lengths, geom_feats)
        ctx.saved_shapes = B, D, H, W
        return out

    @staticmethod
    def backward(ctx, out_grad):
        interval_starts, interval_lengths, geom_feats = ctx.saved_tensors
        B, D, H, W = ctx.saved_shapes

        out_grad = out_grad.contiguous()
        x_grad = bev_pool_ext.bev_pool_backward(
            out_grad,
            geom_feats,
            interval_lengths,
            interval_starts,
            B,
            D,
            H,
            W,
        )

        return x_grad, None, None, None, None, None, None


def bev_pool(feats, coords, B, D, H, W):
    assert feats.shape[0] == coords.shape[0]

    ranks = (
        coords[:, 0] * (W * D * B)
        + coords[:, 1] * (D * B)
        + coords[:, 2] * B
        + coords[:, 3]
    )
    indices = ranks.argsort()
    feats, coords, ranks = feats[indices], coords[indices], ranks[indices]

    x = QuickCumsumCuda.apply(feats, coords, ranks, B, D, H, W)
    x = x.permute(0, 4, 1, 2, 3).contiguous()
    return x


class BEVPoolV2Function(torch.autograd.Function):
    """BEV pooling using pre-sorted inputs with interval indices.

    Unlike QuickCumsumCuda which computes intervals internally,
    this function takes pre-computed interval_starts and interval_lengths
    as inputs, making it traceable for ONNX export.

    The ONNX symbolic exports this as custom::BEVPoolV2 for TRT Plugin.
    """

    @staticmethod
    def forward(ctx, x, geom_feats, interval_starts, interval_lengths,
                B, D, H, W):
        geom_feats = geom_feats.int()
        interval_starts = interval_starts.int()
        interval_lengths = interval_lengths.int()

        out = bev_pool_ext.bev_pool_forward(
            x,
            geom_feats,
            interval_lengths,
            interval_starts,
            B, D, H, W,
        )
        ctx.save_for_backward(interval_starts, interval_lengths, geom_feats)
        ctx.saved_shapes = B, D, H, W
        return out

    @staticmethod
    def backward(ctx, out_grad):
        interval_starts, interval_lengths, geom_feats = ctx.saved_tensors
        B, D, H, W = ctx.saved_shapes
        out_grad = out_grad.contiguous()
        x_grad = bev_pool_ext.bev_pool_backward(
            out_grad,
            geom_feats,
            interval_lengths,
            interval_starts,
            B, D, H, W,
        )
        return x_grad, None, None, None, None, None, None, None

    @staticmethod
    def symbolic(g, x, geom_feats, interval_starts, interval_lengths,
                 B, D, H, W):
        return g.op(
            "custom::BEVPoolV2",
            x, geom_feats, interval_starts, interval_lengths,
            plugin_version_s="1",
            B_i=B, D_i=D, H_i=H, W_i=W,
        )


def bev_pool_v2(x, geom_feats, interval_starts, interval_lengths, B, D, H, W):
    """BEV pooling with pre-computed indices (traceable, ONNX-exportable).

    Args:
        x: [N, C] sorted features (already filtered and sorted by rank)
        geom_feats: [N, 4] sorted voxel coordinates [x, y, z, batch] (int)
        interval_starts: [M] start index of each interval (int)
        interval_lengths: [M] length of each interval (int)
        B, D, H, W: output dimensions (batch, Z, X, Y) matching bev_pool() call order

    Returns:
        [B, C, D, H, W] BEV features (permuted to channel-first)
    """
    out = BEVPoolV2Function.apply(
        x, geom_feats, interval_starts, interval_lengths, B, D, H, W
    )
    # [B, D, H, W, C] -> [B, C, D, H, W]
    out = out.permute(0, 4, 1, 2, 3).contiguous()
    return out
