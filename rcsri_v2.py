import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class RCSRI(nn.Module):
    """
    RCSRI-v2
    Relation-Channel Selective Residual Interaction
    with Scale Protection

    Formula:

        F_out =
            F_self
            + alpha_eff
            * G_rel
            * G_ch
            * R_hat(F_self)

    G_rel : WHERE to interact
    G_ch  : WHAT channels to compensate
    R_hat : scale-protected residual
    alpha : HOW MUCH

    Position:
        ResNet50 layer3 -> RCSRI-v2 -> layer4
    """

    def __init__(
        self,
        channels=1024,
        proj_dim=128,
        mlp_ratio=0.25,
        eta=0.5,
        temperature=0.25,
        alpha_init=0.1,
        alpha_max=0.5
    ):
        super(RCSRI, self).__init__()

        self.channels = channels
        self.proj_dim = proj_dim

        self.eta = float(eta)
        self.temperature = float(temperature)
        self.eps = 1e-12

        # ==================================================
        # Bounded alpha
        # ==================================================
        if alpha_max <= 0:
            raise ValueError("alpha_max must > 0")

        if not (0 < alpha_init < alpha_max):
            raise ValueError(
                "Require 0 < alpha_init < alpha_max"
            )

        self.alpha_max = float(alpha_max)

        p = float(alpha_init) / float(alpha_max)

        alpha_raw_init = math.log(
            p / (1.0 - p)
        )

        self.alpha_raw = nn.Parameter(
            torch.tensor(
                alpha_raw_init,
                dtype=torch.float32
            )
        )

        # ==================================================
        # Shared token normalization
        # ==================================================
        self.token_norm = nn.LayerNorm(
            channels
        )

        # ==================================================
        # Relation Gate
        # ==================================================
        self.q_proj = nn.Linear(
            channels,
            proj_dim,
            bias=False
        )

        self.k_proj = nn.Linear(
            channels,
            proj_dim,
            bias=False
        )

        # ==================================================
        # Channel Gate
        # VIS / IR use separate BN
        # ==================================================
        self.score_bn_v = nn.BatchNorm1d(
            channels
        )

        self.score_bn_r = nn.BatchNorm1d(
            channels
        )

        # ==================================================
        # Residual Generator
        # ==================================================
        hidden_dim = max(
            64,
            int(channels * mlp_ratio)
        )

        self.fc1 = nn.Linear(
            channels,
            hidden_dim
        )

        self.act = nn.GELU()

        self.fc2 = nn.Linear(
            hidden_dim,
            channels
        )

        self.last_stats = {}

        self._init_weights()

    @property
    def alpha(self):
        return (
            self.alpha_max
            * torch.sigmoid(
                self.alpha_raw
            )
        )

    def _init_weights(self):

        nn.init.xavier_uniform_(
            self.q_proj.weight
        )

        nn.init.xavier_uniform_(
            self.k_proj.weight
        )

        nn.init.xavier_uniform_(
            self.fc1.weight
        )

        nn.init.zeros_(
            self.fc1.bias
        )

        # Same weak initialization as RGRI
        nn.init.normal_(
            self.fc2.weight,
            mean=0.0,
            std=1e-3
        )

        nn.init.zeros_(
            self.fc2.bias
        )

    @staticmethod
    def _to_tokens(x):

        return (
            x.flatten(2)
            .transpose(1, 2)
        )

    # ======================================================
    # Relation Gate
    # WHERE
    # ======================================================
    def _relation_gate(
        self,
        x_self,
        x_ref
    ):

        B, C, H, W = x_self.shape

        self_tokens = self._to_tokens(
            x_self
        )

        ref_tokens = self._to_tokens(
            x_ref
        )

        self_tokens = self.token_norm(
            self_tokens
        )

        ref_tokens = self.token_norm(
            ref_tokens
        )

        q = self.q_proj(
            self_tokens
        )

        k = self.k_proj(
            ref_tokens
        )

        q = F.normalize(
            q,
            p=2,
            dim=-1
        )

        k = F.normalize(
            k,
            p=2,
            dim=-1
        )

        relation = torch.bmm(
            q,
            k.transpose(1, 2)
        )

        best_relation = relation.max(
            dim=-1
        ).values

        gate = (
            best_relation + 1.0
        ) / 2.0

        gate = gate.clamp(
            0.0,
            1.0
        )

        return gate.view(
            B,
            1,
            H,
            W
        )

    # ======================================================
    # Channel Gate
    # WHAT
    # ======================================================
    def _channel_gate(
        self,
        x,
        score_bn
    ):

        # [B,C,H,W] -> [B,C]
        pooled = (
            x.abs()
            .mean(
                dim=(2, 3)
            )
        )

        normalized = score_bn(
            pooled
        )

        # Global reliability
        gamma = (
            score_bn.weight
            .abs()
        )

        global_score = (
            gamma
            /
            (
                gamma.mean()
                + self.eps
            )
        )

        # Sample-specific reliability
        sample_score = (
            normalized.abs()
        )

        sample_score = (
            sample_score
            /
            (
                sample_score.mean(
                    dim=1,
                    keepdim=True
                )
                + self.eps
            )
        )

        reliability = (
            self.eta
            * global_score.unsqueeze(0)
            +
            (1.0 - self.eta)
            * sample_score
        )

        # High reliability -> stronger enhancement
        # Low reliability  -> suppress interference update
        gate = torch.sigmoid(
            (
                reliability
                - 1.0
            )
            /
            self.temperature
        )

        gate = (
            gate
            .unsqueeze(-1)
            .unsqueeze(-1)
        )

        return gate, reliability

    # ======================================================
    # Residual Generator
    # ======================================================
    def _residual(
        self,
        x
    ):

        B, C, H, W = x.shape

        tokens = self._to_tokens(
            x
        )

        tokens = self.token_norm(
            tokens
        )

        residual = self.fc1(
            tokens
        )

        residual = self.act(
            residual
        )

        residual = self.fc2(
            residual
        )

        residual = (
            residual
            .transpose(1, 2)
            .reshape(
                B,
                C,
                H,
                W
            )
        )

        return residual

    # ======================================================
    # Scale Protection
    #
    # Only shrink an excessive residual.
    # Never amplify a weak residual.
    # ======================================================
    def _scale_protect(
        self,
        x,
        residual
    ):

        x_mag = (
            x.detach()
            .abs()
            .mean(
                dim=(1, 2, 3),
                keepdim=True
            )
        )

        residual_mag = (
            residual.detach()
            .abs()
            .mean(
                dim=(1, 2, 3),
                keepdim=True
            )
        )

        scale = (
            x_mag
            /
            (
                residual_mag
                + self.eps
            )
        ).clamp(
            max=1.0
        )

        protected = (
            residual
            * scale
        )

        return protected, scale

    # ======================================================
    # Forward
    # ======================================================
    def forward(
        self,
        x_v,
        x_r
    ):

        if x_v.shape != x_r.shape:
            raise RuntimeError(
                "RCSRI requires identical "
                "VIS/IR feature shapes: "
                "{} vs {}".format(
                    tuple(x_v.shape),
                    tuple(x_r.shape)
                )
            )

        # --------------------------------------------------
        # WHERE
        # --------------------------------------------------
        rel_v = self._relation_gate(
            x_v,
            x_r
        )

        rel_r = self._relation_gate(
            x_r,
            x_v
        )

        # --------------------------------------------------
        # WHAT
        # --------------------------------------------------
        ch_v, score_v = self._channel_gate(
            x_v,
            self.score_bn_v
        )

        ch_r, score_r = self._channel_gate(
            x_r,
            self.score_bn_r
        )

        # [B,1,H,W] * [B,C,1,1]
        joint_v = rel_v * ch_v
        joint_r = rel_r * ch_r

        # --------------------------------------------------
        # Residual
        # --------------------------------------------------
        res_v = self._residual(
            x_v
        )

        res_r = self._residual(
            x_r
        )

        # --------------------------------------------------
        # Scale Protection
        # --------------------------------------------------
        res_v, scale_v = self._scale_protect(
            x_v,
            res_v
        )

        res_r, scale_r = self._scale_protect(
            x_r,
            res_r
        )

        # --------------------------------------------------
        # HOW MUCH
        # --------------------------------------------------
        alpha_eff = self.alpha

        delta_v = (
            alpha_eff
            * joint_v
            * res_v
        )

        delta_r = (
            alpha_eff
            * joint_r
            * res_r
        )

        out_v = x_v + delta_v
        out_r = x_r + delta_r

        # --------------------------------------------------
        # Diagnostics
        # --------------------------------------------------
        with torch.no_grad():

            delta_ratio_v = (
                delta_v.abs().mean()
                /
                (
                    x_v.abs().mean()
                    + self.eps
                )
            )

            delta_ratio_r = (
                delta_r.abs().mean()
                /
                (
                    x_r.abs().mean()
                    + self.eps
                )
            )

            self.last_stats = {

                "alpha_eff":
                    alpha_eff.item(),

                "alpha_raw":
                    self.alpha_raw.item(),

                "alpha_max":
                    self.alpha_max,

                "rel_v_mean":
                    rel_v.mean().item(),

                "rel_v_std":
                    rel_v.std().item(),

                "rel_r_mean":
                    rel_r.mean().item(),

                "rel_r_std":
                    rel_r.std().item(),

                # compatibility
                "gate_v_mean":
                    rel_v.mean().item(),

                "gate_v_std":
                    rel_v.std().item(),

                "gate_r_mean":
                    rel_r.mean().item(),

                "gate_r_std":
                    rel_r.std().item(),

                "ch_v_mean":
                    ch_v.mean().item(),

                "ch_v_std":
                    ch_v.std().item(),

                "ch_r_mean":
                    ch_r.mean().item(),

                "ch_r_std":
                    ch_r.std().item(),

                "joint_v_mean":
                    joint_v.mean().item(),

                "joint_r_mean":
                    joint_r.mean().item(),

                "score_v_std":
                    score_v.std().item(),

                "score_r_std":
                    score_r.std().item(),

                "scale_v":
                    scale_v.mean().item(),

                "scale_r":
                    scale_r.mean().item(),

                "delta_ratio_v":
                    delta_ratio_v.item(),

                "delta_ratio_r":
                    delta_ratio_r.item(),
            }

        return out_v, out_r
