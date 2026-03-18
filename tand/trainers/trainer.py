from __future__ import annotations

from contextlib import nullcontext
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Any, Tuple, List

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision.ops import sigmoid_focal_loss

from .base import BaseTrainer
from tand.losses import pointwise_classification_loss, pointwise_focal_loss, bce_on_classmaps
from tand.losses.distribution import distribution_consistency_loss
from tand.models.fused_pointcls_unet import VirchowFusedNet
from tand.models.tand_net import DINOv3VirchowFused
from tand.evaluation.peak import detect_peaks
from tand.evaluation.metrics import greedy_point_match, aggregate_metrics



@dataclass
class VirchowFusedCfg:
    num_tissue: int = 6
    film_limit: float = 0.5
    lam_bias: float = 0.8
    conf_thr: float = 0.7
    tau: float = 1.0
    heat_focal_alpha: float = 0.25
    heat_focal_gamma: float = 2.0
    cls_bce_pos_weight: float = 1.0
    seg_loss_weight: float = 0.3
    heat_loss_weight: float = 1.0
    cls_loss_weight: float = 1.0
    mode: str = "film_bias"  # baseline/film_only/bias_only/film_bias
    cls_point_ce_weight: float = 0.7
    cls_bce_map_weight: float = 0.3
    lambda_P: float = 0.2
    dist_conf_thr: float = 0.7
    lambda_det: float = 1.0
    lambda_cls: float = 1.0


class VirchowFusedTrainer(BaseTrainer):
    """Two-phase training: Virchow seg head -> DINOv3 detection with FiLM on classification path."""

    def __init__(self, config: Dict[str, Any], device: torch.device, logger):
        super().__init__(config, device, logger)
        self.cfg = VirchowFusedCfg(
            num_tissue=int(config.get("num_tissue", 6)),
            film_limit=float(config.get("film_limit", 0.5)),
            lam_bias=float(config.get("lam_bias", 0.8)),
            conf_thr=float(config.get("bias_conf_thr", config.get("conf_thr", 0.7))),
            tau=float(config.get("tau", 1.0)),
            heat_focal_alpha=float(config.get("heat_focal_alpha", 0.25)),
            heat_focal_gamma=float(config.get("heat_focal_gamma", 2.0)),
            cls_bce_pos_weight=float(config.get("cls_bce_pos_weight", 1.0)),
            seg_loss_weight=float(config.get("seg_loss_weight", 0.3)),
            heat_loss_weight=float(config.get("heat_loss_weight", 1.0)),
            cls_loss_weight=float(config.get("cls_loss_weight", 1.0)),
            mode=str(config.get("mode", "film_bias")),
            cls_point_ce_weight=float(config.get("cls_point_ce_weight", 0.7)),
            cls_bce_map_weight=float(config.get("cls_bce_map_weight", 0.3)),
            lambda_P=float(config.get("lambda_P", config.get("dist_loss_weight", 0.2))),
            dist_conf_thr=float(config.get("dist_conf_thr", config.get("conf_thr", 0.7))),
            lambda_det=float(config.get("lambda_det", config.get("heat_loss_weight", 1.0))),
            lambda_cls=float(config.get("lambda_cls", config.get("cls_loss_weight", 1.0))),
        )
        self._prior_ready = False
        self._last_train_logs: Dict[str, float] | None = None
        self._last_P_hat_dbg: torch.Tensor | None = None
        # Class-weighted loss settings
        self._cls_class_weights: torch.Tensor | None = None
        self._cls_focal_gamma = float(config.get("cls_focal_gamma", 0.0))
        self._cls_label_smoothing = float(config.get("cls_label_smoothing", 0.0))
        self._cls_use_focal = self._cls_focal_gamma > 0
        # Resolve tissue class names for logging (fallback to generic names)
        self.tissue_names = self._load_tissue_names(self.cfg.num_tissue)

    def build_model(self, train_dataset) -> nn.Module:  # noqa: ANN001
        num_classes = getattr(train_dataset, "num_classes", None)
        if num_classes is None:
            raise ValueError("Dataset must expose num_classes for VirchowFusedTrainer.")
        fusion_model = str(self.config.get("fusion_model", "efficient")).lower()
        if fusion_model == "dino":
            model = DINOv3VirchowFused(
                num_cell_classes=num_classes,
                num_tissue=self.cfg.num_tissue,
                film_limit=self.cfg.film_limit,
                dino_variant=self.config.get("dino_variant", "convnext_small"),
                dino_up_mode=self.config.get("dino_up_mode", "upsample"),
                dino_load_hf=self.config.get("dino_load_hf", False),
                lam_bias=self.cfg.lam_bias,
                conf_thr=self.cfg.conf_thr,
                tau=self.cfg.tau,
                prior_path=self.config.get("prior_path", None),
                film_scales=str(self.config.get("film_scales", "16,8,4")),
            )
            model.lam_bias = float(self.config.get("lam_bias", self.cfg.lam_bias))
            model.conf_thr = float(self.config.get("bias_conf_thr", self.cfg.conf_thr))
            prior_path = self.config.get("prior_path")
            if prior_path:
                prior_file = Path(prior_path)
                if prior_file.exists():
                    log_pc_given_t = torch.from_numpy(np.load(prior_file)).float()
                    model.set_tissue_prior(log_pc_given_t)
                    self._prior_ready = True
                    self.logger.info(f"[Prior] loaded log_pc_given_t: {tuple(log_pc_given_t.shape)} lam={model.lam_bias} conf_thr={model.conf_thr}")
                else:
                    self.logger.warning(f"[Prior] file not found: {prior_file}")
                    self._prior_ready = False
            else:
                self.logger.info("[Prior] not provided; bias disabled unless lam_bias>0 with manual set_tissue_prior.")
                self._prior_ready = False
        else:
            # Default: EfficientNet + Virchow (original path)
            model = VirchowFusedNet(num_classes=num_classes, num_tissue=self.cfg.num_tissue, film_limit=self.cfg.film_limit, pretrained=True)

        if hasattr(model, "lam_bias"):
            model.lam_bias = self.cfg.lam_bias
        if hasattr(model, "conf_thr"):
            model.conf_thr = self.cfg.conf_thr
        if hasattr(model, "tau"):
            model.tau = self.cfg.tau

        # Virchow encoder stays frozen in both phases; seg head is trained only in pretrain phase
        for p in model.vir.parameters():
            p.requires_grad = False
        for p in model.seg.parameters():
            p.requires_grad = False
        # Compute class weights from config or auto (inverse-frequency)
        cls_weight_cfg = self.config.get("cls_class_weights", None)
        if cls_weight_cfg == "auto":
            pass  # auto weights computed later by the training script
        elif cls_weight_cfg is not None:
            try:
                w = [float(x) for x in str(cls_weight_cfg).split(",")]
                self._cls_class_weights = torch.tensor(w, dtype=torch.float32)
            except Exception:
                pass
        return model.to(self.device)

    # No prior estimation in this design; FiLM only (detection branch unaffected)

    def train_epoch(self, model: VirchowFusedNet, dataloader: DataLoader, optimizer: torch.optim.Optimizer) -> Tuple[float, float, float]:
        model.train()
        # Phase 2: ensure Virchow and seg head are frozen
        for p in model.vir.parameters():
            p.requires_grad = False
        for p in model.seg.parameters():
            p.requires_grad = False

        amp_mode = str(self.config.get("train_amp", "auto")).lower()
        if amp_mode == "on":
            amp_enabled = torch.cuda.is_available()
        elif amp_mode == "off":
            amp_enabled = False
        else:  # auto
            amp_enabled = torch.cuda.is_available()
        if not torch.cuda.is_available():
            raise RuntimeError("VirchowFusedTrainer requires CUDA for training; GPU not detected.")
        scaler = torch.amp.GradScaler(device="cuda", enabled=amp_enabled)

        total_loss = 0.0
        det_loss_acc = 0.0
        cls_loss_acc = 0.0
        lp_loss_acc = 0.0
        ce_loss_acc = 0.0
        bce_loss_acc = 0.0
        n_batches = 0

        pos_cfg = self.cfg.cls_bce_pos_weight
        if isinstance(pos_cfg, (float, int)):
            pos_w = torch.tensor(float(pos_cfg), device=self.device)
        else:
            pos_w = torch.as_tensor(pos_cfg, device=self.device, dtype=torch.float32)
        # Decide FiLM usage dynamically allowing warmup (disable FiLM for first K epochs)
        mode_cfg = str(self.config.get("mode", self.cfg.mode)).lower()
        film_enabled_by_mode = (mode_cfg in ("film_only", "film_bias"))
        film_start_ep = int(self.config.get("film_start_epoch", 0) or 0)
        cur_ep = int(self.config.get("_current_epoch", 0) or 0)
        use_film = film_enabled_by_mode and (cur_ep >= film_start_ep)
        # Bias branch shares the same warmup gate so both FiLM & bias switch on together
        bias_enabled_by_mode = (mode_cfg in ("bias_only", "film_bias"))
        use_bias = bias_enabled_by_mode and (cur_ep >= film_start_ep)

        ce_w = float(self.config.get("cls_point_ce_weight", self.cfg.cls_point_ce_weight))
        bce_w = float(self.config.get("cls_bce_map_weight", self.cfg.cls_bce_map_weight))
        lambda_det = float(self.config.get("lambda_det", self.cfg.lambda_det))
        lambda_cls = float(self.config.get("lambda_cls", self.cfg.lambda_cls))
        lambda_P = float(self.config.get("lambda_P", self.cfg.lambda_P))
        dist_conf_thr = float(self.config.get("dist_conf_thr", self.cfg.dist_conf_thr))
        tau_cfg = float(self.config.get("tau", self.cfg.tau))
        use_dist_loss = lambda_P > 0.0

        last_P_hat = None

        for batch in dataloader:
            img = batch["image"].to(self.device)
            with torch.amp.autocast("cuda", enabled=amp_enabled):
                out = model(img, use_film=use_film, use_bias=use_bias)
            heat_logits = out["heatmap_logits"]
            cls_logits = out["class_logits"]
            tissue_logits_224 = out.get("tissue_logits_224")

            B, _, H, W = heat_logits.shape
            C = cls_logits.shape[1]
            heat_gt = torch.zeros(B, 1, H, W, device=self.device)
            cls_gt = torch.zeros(B, C, H, W, device=self.device)
            gt_points_xy: list[torch.Tensor] = []
            gt_point_labels: list[torch.Tensor] = []

            points_list = batch.get("points") or [None] * B
            labels_list = batch.get("labels") or [None] * B

            for b in range(B):
                pts = points_list[b] if b < len(points_list) else None
                lbl = labels_list[b] if b < len(labels_list) else None
                if pts is None:
                    pts = []
                if lbl is None:
                    lbl = []

                if isinstance(pts, torch.Tensor):
                    pts_tensor = pts.to(device=self.device, dtype=torch.float32)
                else:
                    pts_tensor = torch.tensor(pts, dtype=torch.float32, device=self.device)
                if pts_tensor.numel() == 0:
                    pts_tensor = pts_tensor.view(0, 2)
                if pts_tensor.dim() == 1 and pts_tensor.numel() > 0:
                    pts_tensor = pts_tensor.view(-1, 2)
                gt_points_xy.append(pts_tensor)

                if isinstance(lbl, torch.Tensor):
                    lbl_tensor = lbl.to(device=self.device, dtype=torch.long)
                else:
                    lbl_tensor = torch.tensor(lbl, dtype=torch.long, device=self.device)
                if lbl_tensor.numel() == 0:
                    lbl_tensor = lbl_tensor.view(0)
                gt_point_labels.append(lbl_tensor)

                pts_for_maps = pts_tensor.detach().cpu().tolist()
                lbl_for_maps = lbl_tensor.detach().cpu().tolist()
                for (x, y), c in zip(pts_for_maps, lbl_for_maps):
                    xi, yi = int(min(max(x, 0.0), W - 1)), int(min(max(y, 0.0), H - 1))
                    rad_h, rad_c = 2, 2
                    y0, y1 = max(0, yi - 3), min(H, yi + 4)
                    x0, x1 = max(0, xi - 3), min(W, xi + 4)
                    yy, xx = torch.meshgrid(
                        torch.arange(y0, y1, device=self.device),
                        torch.arange(x0, x1, device=self.device),
                        indexing="ij",
                    )
                    g_h = torch.exp(-((yy - yi) ** 2 + (xx - xi) ** 2) / (2 * (rad_h ** 2)))
                    g_c = torch.exp(-((yy - yi) ** 2 + (xx - xi) ** 2) / (2 * (rad_c ** 2)))
                    heat_patch = heat_gt[b, 0, y0:y1, x0:x1]
                    heat_gt[b, 0, y0:y1, x0:x1] = torch.maximum(heat_patch, g_h)
                    if 0 <= c < C:
                        cls_patch = cls_gt[b, c, y0:y1, x0:x1]
                        cls_gt[b, c, y0:y1, x0:x1] = torch.maximum(cls_patch, g_c)

            optimizer.zero_grad(set_to_none=True)

            with torch.amp.autocast("cuda", enabled=amp_enabled):
                heat_loss = sigmoid_focal_loss(
                    heat_logits,
                    (heat_gt > 0).float(),
                    alpha=self.cfg.heat_focal_alpha,
                    gamma=self.cfg.heat_focal_gamma,
                    reduction="mean",
                )
                cw = self._cls_class_weights.to(self.device) if self._cls_class_weights is not None else None
                if self._cls_use_focal:
                    loss_ce = pointwise_focal_loss(cls_logits, gt_points_xy, gt_point_labels,
                                                   class_weights=cw, gamma=self._cls_focal_gamma,
                                                   label_smoothing=self._cls_label_smoothing)
                else:
                    loss_ce = pointwise_classification_loss(cls_logits, gt_points_xy, gt_point_labels,
                                                            class_weights=cw, label_smoothing=self._cls_label_smoothing)
                loss_bce = bce_on_classmaps(cls_logits, cls_gt, pos_weight=pos_w)

                lam_tau = max(float(getattr(model, "tau", tau_cfg)), 1e-6)
                tissue_prob = None
                if tissue_logits_224 is not None:
                    tissue_prob = torch.softmax(tissue_logits_224 / lam_tau, dim=1)
                    tissue_prob = F.interpolate(
                        tissue_prob, size=cls_logits.shape[-2:], mode="bilinear", align_corners=False
                    )
                if tissue_prob is None:
                    tissue_prob = torch.zeros(
                        B, 1, H, W, device=cls_logits.device, dtype=cls_logits.dtype
                    )

                log_prior = getattr(model, "log_pc_given_t", None)
                if log_prior is not None:
                    log_prior = log_prior.to(cls_logits.device, dtype=cls_logits.dtype)
                if log_prior is None or log_prior.numel() <= 1:
                    prior_ct = torch.full(
                        (C, tissue_prob.shape[1]),
                        1.0 / max(C, 1),
                        device=cls_logits.device,
                        dtype=cls_logits.dtype,
                    )
                else:
                    prior_ct = log_prior.exp()

                if use_dist_loss:
                    # Cast to float32 for the distribution loss to reduce AMP overflow/NaN risk.
                    loss_lp32, P_hat_dbg = distribution_consistency_loss(
                        cls_logits.float(),
                        tissue_prob.float(),
                        prior_ct.float(),
                        conf_thr=dist_conf_thr,
                        tau=lam_tau,
                    )
                    loss_lp32 = torch.nan_to_num(loss_lp32, nan=0.0, posinf=0.0, neginf=0.0)
                    loss_lp = loss_lp32.to(cls_logits.dtype)
                else:
                    loss_lp = cls_logits.new_zeros(())
                    P_hat_dbg = None

                cls_mix = ce_w * loss_ce + bce_w * loss_bce
                loss_cls = lambda_cls * cls_mix

                det_loss = lambda_det * heat_loss
                loss = det_loss + loss_cls
                if use_dist_loss:
                    loss = loss + lambda_P * loss_lp

            if amp_enabled:
                scaler.scale(loss).backward()
                try:
                    scaler.unscale_(optimizer)
                except AttributeError:
                    pass
                _trainable = [p for p in model.parameters() if p.requires_grad]
                nn.utils.clip_grad_norm_(_trainable, max_norm=self.config.get("max_grad_norm", 5.0))
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                _trainable = [p for p in model.parameters() if p.requires_grad]
                nn.utils.clip_grad_norm_(_trainable, max_norm=self.config.get("max_grad_norm", 5.0))
                optimizer.step()

            total_loss += float(loss.item())
            det_loss_acc += float(det_loss.item())
            cls_loss_acc += float(loss_cls.item())
            ce_loss_acc += float(loss_ce.item())
            bce_loss_acc += float(loss_bce.item())
            lp_loss_acc += float(loss_lp.item())
            n_batches += 1

            if P_hat_dbg is not None:
                last_P_hat = P_hat_dbg.detach().cpu()

        if last_P_hat is not None:
            self._last_P_hat_dbg = last_P_hat

        denom = max(1, n_batches)
        avg_total = total_loss / denom
        avg_det = det_loss_acc / denom
        avg_cls = cls_loss_acc / denom
        avg_ce = ce_loss_acc / denom
        avg_bce = bce_loss_acc / denom
        avg_lp = lp_loss_acc / denom

        self._last_train_logs = {
            "loss": avg_total,
            "loss_det": avg_det,
            "loss_cls": avg_cls,
            "loss_ce_point": avg_ce,
            "loss_bce_map": avg_bce,
            "loss_lp": avg_lp,
        }
        if self.logger is not None:
            self.logger.info(
                "train_epoch: loss={:.4f}, det={:.4f}, cls={:.4f}, ce_point={:.4f}, bce_map={:.4f}, lp={:.4f}".format(
                    avg_total, avg_det, avg_cls, avg_ce, avg_bce, avg_lp
                )
            )

        self._last_dist_loss = avg_lp

        return avg_total, avg_det, avg_cls

    @torch.no_grad()
    def evaluate(self, model: VirchowFusedNet, dataloader: DataLoader) -> Dict[str, Any]:
        det_thresh = self.config.get("det_thresh", 0.35)
        nms_radius = self.config.get("nms_radius", 3)
        match_radius = self.config.get("match_radius", 5)
        return self.evaluate_with_postproc(model, dataloader, det_thresh, nms_radius, match_radius, compute_extra=True)

    @torch.no_grad()
    def evaluate_with_postproc(
        self,
        model: VirchowFusedNet,
        dataloader: DataLoader,
        det_thresh: float,
        nms_radius: int,
        match_radius: float,
        compute_extra: bool = False,
    ) -> Dict[str, Any]:
        model.eval()
        all_metrics: List[Dict[str, Any]] = []
        # Honor FiLM warmup during evaluation as well (keep val consistent with train)
        mode_cfg = str(self.config.get("mode", self.cfg.mode)).lower()
        film_enabled_by_mode = (mode_cfg in ("film_only", "film_bias"))
        film_start_ep = int(self.config.get("film_start_epoch", 0) or 0)
        cur_ep = int(self.config.get("_current_epoch", 0) or 0)
        use_film = film_enabled_by_mode and (cur_ep >= film_start_ep)
        bias_enabled_by_mode = (mode_cfg in ("bias_only", "film_bias"))
        use_bias = bias_enabled_by_mode and (cur_ep >= film_start_ep)

        for batch in dataloader:
            img = batch["image"].to(self.device)
            out = model(img, use_film=use_film, use_bias=use_bias)
            heat = torch.sigmoid(out["heatmap_logits"])  # (B,1,H,W)
            cls_logits = out["class_logits"]  # (B,C,H,W)
            B, C, H, W = cls_logits.shape
            for i in range(B):
                peaks = detect_peaks(heat[i, 0], thresh=det_thresh, nms_radius=nms_radius)
                pred_pts = np.array([[x, y] for x, y, _ in peaks], dtype=np.float32) if peaks else np.zeros((0, 2), dtype=np.float32)
                if len(pred_pts) > 0:
                    labels_pred = []
                    for x, y in pred_pts:
                        xi = int(min(max(x, 0), W - 1))
                        yi = int(min(max(y, 0), H - 1))
                        logit = cls_logits[i, :, yi, xi]
                        labels_pred.append(int(torch.argmax(logit).item()))
                    labels_pred = np.array(labels_pred, dtype=np.int64)
                else:
                    labels_pred = np.zeros((0,), dtype=np.int64)
                gt_pts = batch.get("points", [torch.zeros(0, 2)])[i].numpy()
                gt_labs = batch.get("labels", [torch.zeros(0, dtype=torch.long)])[i].numpy()
                metrics = greedy_point_match(pred_pts, labels_pred, gt_pts, gt_labs, radius=match_radius, num_classes=C)
                all_metrics.append(metrics)

        results = aggregate_metrics(all_metrics)

        if compute_extra:
            try:
                has_mask = False
                for b in dataloader:
                    has_mask = ('tissue_mask' in b)
                    break
                if has_mask:
                    dice_per_class = self._eval_tissue_dice(model, dataloader)
                    results["tissue_dice_per_class"] = dice_per_class
                    if len(dice_per_class) > 0:
                        results["tissue_dice_mean"] = float(np.mean(dice_per_class))
            except Exception as e:
                if self.logger:
                    self.logger.warning(f"[Eval] Tissue Dice computation skipped: {e}")

        return results

    # -------- Pretrain tissue segmentation head (optional) --------
    def pretrain_seg_head(self, model: nn.Module, train_loader: DataLoader, val_loader: DataLoader, epochs: int = 20, lr: float = 1e-3, unfreeze_last_n: int = 0):
        if epochs <= 0:
            return
        # Phase 1: only train LinearSegHead with Dice on 16x16; keep Virchow encoder frozen
        # Optionally unfreeze last N blocks of Virchow if requested
        if unfreeze_last_n and unfreeze_last_n > 0:
            blocks = getattr(model.vir.backbone, "blocks", None) or getattr(model.vir.backbone, "stages", None)
            unfrozen = 0
            if blocks is not None:
                for blk in blocks[-unfreeze_last_n:]:
                    for p in blk.parameters():
                        p.requires_grad = True
                unfrozen = unfreeze_last_n
            for name, p in model.vir.backbone.named_parameters():
                if not p.requires_grad:
                    p.requires_grad = False
            if self.logger:
                self.logger.info(f"[Pretrain Seg] Unfreeze last {unfrozen} Virchow blocks; others frozen.")
        else:
            for p in model.vir.parameters():
                p.requires_grad = False
        for p in model.seg.parameters():
            p.requires_grad = True


        trainable_params = [p for p in model.parameters() if p.requires_grad]
        if len(trainable_params) == 0:
            return
        opt = torch.optim.AdamW(trainable_params, lr=lr)
        sched_name = str(self.config.get('pretrain_lr_schedule', 'none') or 'none').lower()
        if sched_name == 'cosine' and epochs > 0:
            eta_min = float(self.config.get('pretrain_lr_eta_min', lr * 0.1))
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(1, epochs), eta_min=eta_min)
        else:
            scheduler = None

        dice_w = float(self.config.get('pretrain_dice_weight', 0.6) or 0.0)
        ce_w = float(self.config.get('pretrain_ce_weight', 0.4) or 0.0)
        aux_ce_w = float(self.config.get('pretrain_aux_ce_weight', 0.1) or 0.0)
        if dice_w <= 0 and ce_w <= 0 and aux_ce_w <= 0:
            raise ValueError("At least one of pretrain_dice_weight, pretrain_ce_weight, pretrain_aux_ce_weight must be > 0")
        label_smooth = float(self.config.get('pretrain_label_smoothing', 0.0) or 0.0)
        grad_accum = int(self.config.get('pretrain_grad_accum', 1) or 1)
        grad_clip = float(self.config.get('pretrain_grad_clip', 0.0) or 0.0)
        amp_cfg = self.config.get('pretrain_amp', 'auto')
        if isinstance(amp_cfg, str):
            amp_cfg_l = amp_cfg.lower()
            if amp_cfg_l in ('auto', 'default'):
                amp_enabled = torch.cuda.is_available()
            elif amp_cfg_l in ('on', 'true', '1', 'yes'):
                amp_enabled = torch.cuda.is_available()
            else:
                amp_enabled = False
        else:
            amp_enabled = bool(amp_cfg) and torch.cuda.is_available()
        if not torch.cuda.is_available():
            raise RuntimeError("VirchowFusedTrainer requires CUDA for training; GPU not detected.")
        scaler = torch.amp.GradScaler(device="cuda", enabled=amp_enabled)
        amp_kwargs = {"dtype": torch.float16} if amp_enabled else {}

        min_epochs_cfg = self.config.get('pretrain_min_epochs', None)
        if min_epochs_cfg is None or int(min_epochs_cfg) <= 0:
            min_epochs = max(2, epochs // 3)
        else:
            min_epochs = int(min_epochs_cfg)
        patience = int(self.config.get('pretrain_earlystop_patience', 5) or 5)
        min_delta = float(self.config.get('pretrain_earlystop_min_delta', 0.001) or 0.0)

        base_bs = getattr(train_loader, 'batch_size', None)
        if base_bs is None:
            base_bs = 1
        if self.logger:
            self.logger.info(
                f"[Pretrain Seg] AMP mode ({amp_cfg}): {'enabled' if amp_enabled else 'disabled'}"
            )
        if (grad_accum > 1) and self.logger:
            eff_bs = grad_accum * base_bs
            self.logger.info(f"[Pretrain Seg] Gradient accumulation: {grad_accum} steps (effective batch~={eff_bs}).")
        if self.logger:
            self.logger.info(
                f"[Pretrain Seg] Loss weights -> dice:{dice_w:.2f}, ce:{ce_w:.2f}, aux_ce:{aux_ce_w:.2f}; "
                f"label_smooth={label_smooth:.3f}; min_epochs={min_epochs}, patience={patience}."
            )

        # ----- Per-class weighted Dice setup -----
        C = int(self.cfg.num_tissue)
        weights_conf = str(self.config.get('pretrain_seg_dice_weights', 'auto')).strip().lower()
        class_weights = None
        if weights_conf == 'uniform':
            class_weights = torch.ones(C, device=self.device)
        elif weights_conf == 'auto':
            counts = torch.zeros(C, dtype=torch.float64)
            total = 0
            for batch in train_loader:
                if 'tissue_mask' not in batch:
                    continue
                tmask = batch['tissue_mask']
                if isinstance(tmask, list):
                    tmask = torch.stack(tmask, dim=0)
                vals, cnt = torch.unique(tmask, return_counts=True)
                for v, c in zip(vals, cnt):
                    vi = int(v.item())
                    if 0 <= vi < C:
                        counts[vi] += int(c.item())
                total += int(tmask.numel())
            if total > 0:
                inv = (total / max(1, C)) / (counts + 1.0)
                inv = inv / (inv.mean().clamp(min=1e-6))
                class_weights = inv.to(self.device).float()
                if self.logger:
                    dist = counts / counts.sum().clamp(min=1.0)
                    dist_str = ", ".join(
                        f"{self.tissue_names[i] if i < len(self.tissue_names) else f'tissue_{i}'}:{float(dist[i]):.3%}"
                        for i in range(C)
                    )
                    self.logger.info(f"[Pretrain Seg] Tissue pixel distribution: {dist_str}")
            else:
                class_weights = torch.ones(C, device=self.device)
        else:
            try:
                arr = [float(x) for x in str(self.config.get('pretrain_seg_dice_weights')).split(',')]
                if len(arr) != C:
                    raise ValueError
                class_weights = torch.tensor(arr, device=self.device, dtype=torch.float32)
            except Exception:
                class_weights = torch.ones(C, device=self.device)
        if self.logger:
            cw = ','.join([f"{float(x):.2f}" for x in class_weights.detach().cpu().tolist()])
            self.logger.info(f"[Pretrain Seg] Using per-class Dice weights: [{cw}]")

        class_weights = class_weights.to(self.device)

        best_state = None
        best_mean = -1.0
        best_pc = torch.zeros(C, device=self.device)
        epochs_since_best = 0

        _grid_size = 16  # Virchow2 outputs 16x16 token grid

        for ep in range(1, epochs + 1):
            model.seg.train(True)
            model.det.train(False)
            tr_loss = 0.0
            dice_running = 0.0
            ce_running = 0.0
            aux_running = 0.0
            n = 0
            accum_steps = 0
            opt.zero_grad(set_to_none=True)
            for batch in train_loader:
                if "tissue_mask" not in batch:
                    continue
                img = batch["image"].to(self.device)
                if unfreeze_last_n and unfreeze_last_n > 0:
                    x224 = F.interpolate(img, size=(224, 224), mode="bilinear", align_corners=False)
                    tokens = model.vir.forward_features(x224)
                else:
                    with torch.no_grad():
                        x224 = F.interpolate(img, size=(224, 224), mode="bilinear", align_corners=False)
                        tokens = model.vir.forward_features(x224)
                tmask = batch["tissue_mask"]
                if isinstance(tmask, list):
                    tmask = torch.stack(tmask, dim=0)
                tmask = tmask.to(self.device)
                tgt_grid = F.interpolate(tmask.unsqueeze(1).float(), size=(_grid_size, _grid_size), mode="nearest").squeeze(1).long()
                tgt_grid_oh = F.one_hot(tgt_grid, num_classes=C).permute(0, 3, 1, 2).float()
                tgt224 = F.interpolate(tmask.unsqueeze(1).float(), size=(224, 224), mode="nearest").squeeze(1).long()

                with (torch.amp.autocast("cuda", **amp_kwargs) if amp_enabled else nullcontext()):
                    logits224, logits_grid = model.seg(tokens)
                    probs_grid = torch.softmax(logits_grid, dim=1)
                    num = 2 * (probs_grid * tgt_grid_oh).sum(dim=(2, 3)) + 1.0
                    den = (probs_grid.pow(2) + tgt_grid_oh.pow(2)).sum(dim=(2, 3)) + 1.0
                    dice_per_class = 1 - (num / den)
                    w = class_weights.view(1, C)
                    dice_loss = (dice_per_class * w).sum(dim=1) / (w.sum() + 1e-6)
                    dice_loss = dice_loss.mean()
                    ce_loss_grid = F.cross_entropy(
                        logits_grid,
                        tgt_grid,
                        weight=class_weights,
                        label_smoothing=label_smooth if label_smooth > 0 else 0.0,
                    ) if ce_w > 0 else torch.zeros((), device=self.device, dtype=logits_grid.dtype)
                    aux_ce_loss = F.cross_entropy(
                        logits224,
                        tgt224,
                        weight=class_weights,
                        label_smoothing=label_smooth if label_smooth > 0 else 0.0,
                    ) if aux_ce_w > 0 else torch.zeros((), device=self.device, dtype=logits224.dtype)
                    combined_loss = torch.zeros((), device=self.device, dtype=dice_loss.dtype)
                    if dice_w > 0:
                        combined_loss = combined_loss + dice_w * dice_loss
                    if ce_w > 0:
                        combined_loss = combined_loss + ce_w * ce_loss_grid
                    if aux_ce_w > 0:
                        combined_loss = combined_loss + aux_ce_w * aux_ce_loss

                loss_for_backward = combined_loss / grad_accum
                if amp_enabled:
                    scaler.scale(loss_for_backward).backward()
                else:
                    loss_for_backward.backward()
                accum_steps += 1

                if accum_steps % grad_accum == 0:
                    if grad_clip > 0:
                        if amp_enabled:
                            scaler.unscale_(opt)
                        nn.utils.clip_grad_norm_(trainable_params, grad_clip)
                    if amp_enabled:
                        scaler.step(opt)
                        scaler.update()
                    else:
                        opt.step()
                    opt.zero_grad(set_to_none=True)

                tr_loss += float(combined_loss.detach().item())
                dice_running += float(dice_loss.detach().item()) if dice_w > 0 else 0.0
                ce_running += float(ce_loss_grid.detach().item()) if ce_w > 0 else 0.0
                aux_running += float(aux_ce_loss.detach().item()) if aux_ce_w > 0 else 0.0
                n += 1

            if accum_steps % grad_accum != 0:
                if grad_clip > 0:
                    if amp_enabled:
                        scaler.unscale_(opt)
                    nn.utils.clip_grad_norm_(trainable_params, grad_clip)
                if amp_enabled:
                    scaler.step(opt)
                    scaler.update()
                else:
                    opt.step()
                opt.zero_grad(set_to_none=True)

            tr_loss = tr_loss / max(1, n)
            dice_running = dice_running / max(1, n)
            ce_running = ce_running / max(1, n)
            aux_running = aux_running / max(1, n)

            has_mask = False
            for b in val_loader:
                has_mask = ('tissue_mask' in b)
                break
            if self.logger and scheduler is not None:
                current_lr = opt.param_groups[0]['lr']
                self.logger.info(f"[Pretrain Seg] LR now {current_lr:.6f} (schedule={sched_name})")

            if has_mask:
                dice_pc_list = self._eval_tissue_dice(model, val_loader)
                dice_pc = torch.tensor(dice_pc_list, device=self.device)
                mean_d = float(torch.nanmean(dice_pc).item())
                improved = (mean_d > best_mean + min_delta)
                if improved:
                    best_mean = mean_d
                    best_pc = dice_pc
                    epochs_since_best = 0
                    try:
                        import copy as _copy
                        best_state = {'seg': _copy.deepcopy(model.seg.state_dict())}
                    except Exception:
                        best_state = None
                else:
                    epochs_since_best += 1
                if self.logger:
                    pc_str = ", ".join([
                        f"{self.tissue_names[i]}:{float(d):.3f}" for i, d in enumerate(dice_pc.tolist())
                    ])
                    self.logger.info(
                        f"[Pretrain Seg] Epoch {ep}/{epochs} train_loss={tr_loss:.4f}"
                        f" (dice={dice_running:.4f}, ce={ce_running:.4f}, aux={aux_running:.4f})"
                        f" | Val Dice mean={mean_d:.3f} | {pc_str}"
                    )
                if ep >= min_epochs and patience is not None and patience > 0 and epochs_since_best >= patience:
                    if self.logger:
                        self.logger.info(
                            f"[Pretrain Seg] Early stopping after {ep} epochs: mean Dice stalled (>{patience} epochs without +{min_delta})."
                        )
                    break
            else:
                if self.logger:
                    self.logger.info(
                        f"[Pretrain Seg] Epoch {ep}/{epochs} train_loss={tr_loss:.4f}"
                        f" (dice={dice_running:.4f}, ce={ce_running:.4f}, aux={aux_running:.4f}) (no val tissue masks)"
                    )

            if scheduler is not None:
                scheduler.step()

        for p in model.seg.parameters():
            p.requires_grad = False
        for p in model.vir.parameters():
            p.requires_grad = False
        if best_state is not None:
            try:
                model.seg.load_state_dict(best_state['seg'])
                if self.logger:
                    self.logger.info("[Pretrain Seg] Restored best seg/context weights by val Dice.")
            except Exception as _e:
                if self.logger:
                    self.logger.warning(f"[Pretrain Seg] Failed to restore best seg head: {_e}")

    @torch.no_grad()
    def _eval_tissue_dice(self, model, dataloader: DataLoader):
        model.eval()

        C = self.cfg.num_tissue
        intersect = torch.zeros(C, device=self.device)
        denom = torch.zeros(C, device=self.device)
        for batch in dataloader:
            if "tissue_mask" not in batch:
                continue
            img = batch["image"].to(self.device)
            tmask = batch["tissue_mask"]
            if isinstance(tmask, list):
                tmask = torch.stack(tmask, dim=0)
            tmask = tmask.to(self.device)
            with torch.no_grad():
                x224 = F.interpolate(img, size=(224, 224), mode="bilinear", align_corners=False)
                tokens = model.vir.forward_features(x224)
                logits224, _ = model.seg(tokens)
            pred = logits224.argmax(dim=1)  # [B,224,224]
            tmask = F.interpolate(tmask.unsqueeze(1).float(), size=(224, 224), mode="nearest").squeeze(1).long()
            for c in range(C):
                p = (pred == c)
                g = (tmask == c)
                inter = (p & g).sum()
                d = p.sum() + g.sum()
                intersect[c] += inter
                denom[c] += d
        dice = (2 * intersect + 1e-6) / (denom + 1e-6)
        return dice.tolist()

    # ----- helpers -----
    def _load_tissue_names(self, num_tissue: int):
        names = [f"tissue_{i}" for i in range(num_tissue)]
        try:
            tm_root = str(self.config.get('tissue_mask_root') or '').strip()
            if tm_root:
                import json, os
                candidates = [
                    os.path.join(tm_root, 'dataset_info.json'),
                    os.path.join(os.path.dirname(tm_root.rstrip('/')), 'dataset_info.json'),
                ]
                info_path = next((p for p in candidates if os.path.exists(p)), None)
                if info_path:
                    with open(info_path, 'r') as f:
                        info = json.load(f)
                    mp = info.get('tissue_mapping', {}) or {}
                    out = []
                    for i in range(num_tissue):
                        if i == 0:
                            out.append(mp.get('0', 'background'))
                        else:
                            out.append(mp.get(str(i), f'tissue_{i}'))
                    return out
        except Exception as exc:  # pragma: no cover - logging only
            if self.logger:
                self.logger.warning(f"[TissueNames] Failed to load mapping: {exc}")
        # default
        if num_tissue > 0:
            names[0] = 'background'
        return names
