import torch
import torch.nn as nn
import torch.nn.functional as F


class HybridCrossCenterLoss(nn.Module):
    """
    Hybrid Cross-Center Loss

    保留 RFM 原始 cross-center loss：
        RGB 样本 -> IR 中心
        IR 样本  -> RGB 中心

    额外加入一个轻量级 CAL-style 中心聚合项：
        同身份 RGB/IR 中心更接近
        不同身份中心保持区分

    最终：
        loss = original_cc_loss + beta_t * light_cal_loss
    """

    def __init__(self,
                 dist_type='l2',
                 margin=0.1,
                 beta_max=0.02,
                 warm_epoch=30,
                 eps=1e-6,
                 normalize_cal=True):
        super(HybridCrossCenterLoss, self).__init__()

        self.dist_type = dist_type
        self.margin = margin
        self.beta_max = beta_max
        self.warm_epoch = warm_epoch
        self.eps = eps
        self.normalize_cal = normalize_cal

    def _distance_to_center(self, feat, center):
        """
        feat:   [K, D]
        center: [D]
        return: [K]
        """
        if self.dist_type == 'l2':
            return torch.sum((feat - center.unsqueeze(0)) ** 2, dim=1)

        elif self.dist_type == 'l1':
            return torch.sum(torch.abs(feat - center.unsqueeze(0)), dim=1)

        elif self.dist_type == 'cos':
            return 1.0 - F.cosine_similarity(feat, center.unsqueeze(0), dim=1)

        else:
            return torch.sum((feat - center.unsqueeze(0)) ** 2, dim=1)

    def _original_cc_loss(self, feat_v, feat_t, label_v, label_t):
        """
        原始 cross-center 思想：
        visible 样本靠近 thermal 中心；
        thermal 样本靠近 visible 中心。
        """

        labels = torch.unique(label_v)
        total_loss = feat_v.new_tensor(0.0)
        valid_id_num = 0

        for lab in labels:
            mask_v = (label_v == lab)
            mask_t = (label_t == lab)

            if mask_v.sum() == 0 or mask_t.sum() == 0:
                continue

            fv = feat_v[mask_v]  # [K, D]
            ft = feat_t[mask_t]  # [K, D]

            center_v = fv.mean(dim=0)  # [D]
            center_t = ft.mean(dim=0)  # [D]

            # RGB 样本靠近 IR 中心
            dist_v_to_t = self._distance_to_center(fv, center_t)

            # IR 样本靠近 RGB 中心
            dist_t_to_v = self._distance_to_center(ft, center_v)

            # 保留原 CC 的 margin 约束思想
            loss_v = F.relu(dist_v_to_t - self.margin).mean()
            loss_t = F.relu(dist_t_to_v - self.margin).mean()

            total_loss = total_loss + 0.5 * (loss_v + loss_t)
            valid_id_num += 1

        if valid_id_num == 0:
            return feat_v.new_tensor(0.0)

        return total_loss / valid_id_num

    def _light_cal_loss(self, feat_v, feat_t, label_v, label_t):
        """
        轻量级 CAL 中心聚合项：
        1. 同身份 visible / thermal 中心靠近；
        2. 不同身份中心保持距离；
        3. 使用 ratio 形式，数值较稳定。
        """

        labels = torch.unique(label_v)

        center_v_list = []
        center_t_list = []

        for lab in labels:
            mask_v = (label_v == lab)
            mask_t = (label_t == lab)

            if mask_v.sum() == 0 or mask_t.sum() == 0:
                continue

            center_v = feat_v[mask_v].mean(dim=0)
            center_t = feat_t[mask_t].mean(dim=0)

            center_v_list.append(center_v)
            center_t_list.append(center_t)

        if len(center_v_list) < 2:
            return feat_v.new_tensor(0.0)

        center_v = torch.stack(center_v_list, dim=0)  # [N, D]
        center_t = torch.stack(center_t_list, dim=0)  # [N, D]

        if self.normalize_cal:
            center_v = F.normalize(center_v, p=2, dim=1)
            center_t = F.normalize(center_t, p=2, dim=1)

        # 正样本中心距离：同 ID 的 RGB 中心和 IR 中心
        pos_dist = torch.sum((center_v - center_t) ** 2, dim=1)  # [N]
        pos_term = pos_dist.mean()

        # 跨模态平均中心
        center = 0.5 * (center_v + center_t)  # [N, D]

        # 不同身份中心距离
        dist_mat = torch.cdist(center, center, p=2).pow(2)  # [N, N]
        n = dist_mat.size(0)

        mask = torch.eye(n, device=feat_v.device).bool()
        neg_dist = dist_mat.masked_select(~mask)

        if neg_dist.numel() == 0:
            return feat_v.new_tensor(0.0)

        neg_term = neg_dist.mean()

        # ratio 形式：同身份跨模态距离越小越好，不同身份中心距离越大越好
        light_cal = pos_term / (pos_term + neg_term + self.eps)

        return light_cal

    def _beta(self, epoch):
        """
        改进项权重很小，并且逐渐增加。
        """
        if epoch is None:
            return self.beta_max

        progress = min(float(epoch) / float(self.warm_epoch), 1.0)
        beta_t = self.beta_max * (0.1 + 0.9 * progress)

        return beta_t

    def forward(self, feat_v, feat_t, label_v, label_t, epoch=None):
        """
        feat_v:  RGB 特征, [B, D]
        feat_t:  IR  特征, [B, D]
        label_v: RGB 标签, [B]
        label_t: IR  标签, [B]
        """

        cc_loss = self._original_cc_loss(feat_v, feat_t, label_v, label_t)

        cal_loss = self._light_cal_loss(feat_v, feat_t, label_v, label_t)

        beta_t = self._beta(epoch)

        loss = cc_loss + beta_t * cal_loss

        return loss
