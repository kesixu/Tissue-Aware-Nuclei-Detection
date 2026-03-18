#!/usr/bin/env python3
"""Five-fold cross-validation training script for TAND."""

import os
import sys
import json
import time
import torch
import argparse
import logging
from pathlib import Path
from datetime import datetime

# Add project root to Python path
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

# Import required modules
from tand.data.dataset import (
    PatchesDataset,
    collate_patches,
    ShapesPointDataset,
    collate_point_batches,
    compute_sample_weights,
)
from tand.trainers import VirchowFusedTrainer
from tand.utils.viz import (
    to_uint8_rgb,
    overlay_heatmap,
    draw_points,
    overlay_segmentation,
    save_legend,
    default_class_colors,
)
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, WeightedRandomSampler
import numpy as np

torch.backends.cudnn.benchmark = True


class FoldTrainer:
    def __init__(self, fold_num, config):
        self.fold_num = fold_num
        self.config = config
        self.device = torch.device(config['device'] if torch.cuda.is_available() else 'cpu')
        self.best_f1 = 0

        # Set up logger
        self.logger = self._setup_logger()
        # Tissue class names (used for Dice logging and legend)
        self.tissue_names = self._load_tissue_names(int(self.config.get('num_tissue', 6) or 6))

        # Select the specific trainer implementation
        trainer_name = self.config.get('trainer_name')
        if trainer_name is None:
            model_name = self.config.get('model_name', 'efficient')
            if model_name == 'virchow_fused':
                trainer_name = 'virchow_fused'
            else:
                trainer_name = 'virchow_fused'
            self.config['trainer_name'] = trainer_name
        if trainer_name == 'virchow_fused':
            self.impl = VirchowFusedTrainer(self.config, self.device, self.logger)
        else:
            raise ValueError(f"Unknown trainer: {trainer_name}")

        if trainer_name in ('virchow_fused',):
            self._log_virchow_config_snapshot()

    def _setup_logger(self):
        """Set up fold-specific logger."""
        logger = logging.getLogger(f'fold_{self.fold_num}')
        logger.setLevel(logging.INFO)

        # Avoid duplicate handlers
        if logger.handlers:
            return logger

        # File handler
        log_file = self.config['log_dir'] / f'fold_{self.fold_num}.log'
        fh = logging.FileHandler(log_file)
        fh.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
        logger.addHandler(fh)

        return logger

    def _log_virchow_config_snapshot(self):
        keys = [
            'pretrain_seg_epochs', 'pretrain_batch_size', 'pretrain_unfreeze_last_n', 'pretrain_lr',
            'pretrain_seg_dice_weights', 'pretrain_dice_weight', 'pretrain_ce_weight',
            'pretrain_aux_ce_weight', 'pretrain_label_smoothing', 'pretrain_grad_accum',
            'pretrain_grad_clip', 'pretrain_min_epochs', 'pretrain_amp',
            'pretrain_earlystop_patience', 'pretrain_earlystop_min_delta',
            'pretrain_earlystop_degrade_tolerance'
        ]
        snapshot = {k: self.config.get(k) for k in keys}
        self.logger.info("=== TAND Pretrain Configuration ===")
        self.logger.info(json.dumps(snapshot, ensure_ascii=True, sort_keys=True))

    def _load_tissue_names(self, num_tissue: int):
        names = [f"tissue_{i}" for i in range(num_tissue)]
        # Read tissue mapping from dataset_info.json adjacent to tissue_mask_root
        try:
            tm_root = str(self.config.get('tissue_mask_root') or '').strip()
            if tm_root:
                info_path = (Path(tm_root).parent / 'dataset_info.json')
                if info_path.exists():
                    info = json.loads(info_path.read_text())
                    mp = info.get('tissue_mapping', {}) or {}
                    out = []
                    for i in range(num_tissue):
                        if i == 0:
                            out.append(mp.get('0', 'background'))
                        else:
                            out.append(mp.get(str(i), f'tissue_{i}'))
                    return out
        except Exception:
            pass
        if num_tissue > 0:
            names[0] = 'background'
        return names

    def _auto_tune_pretrain_bs(self, model: nn.Module, dataset, collate_fn, unfreeze_last_n: int = 0, max_trials: int = 6):
        """Auto-find maximum usable batch size for pretraining via single forward+backward pass.
        - Only used for Virchow segmentation pretraining (224x224 input).
        - Strategy: start from current batch_size, double until OOM, then binary search in [success, failure).
        Returns: recommended batch size; None if failed.
        """
        if not torch.cuda.is_available():
            return None
        import torch.nn.functional as F

        device = self.device
        start_bs = int(self.config.get('pretrain_batch_size') or self.config.get('batch_size', 4) or 4)
        good = 0
        bad = None

        def try_once(bs: int) -> bool:
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
            loader = DataLoader(dataset, batch_size=bs, shuffle=False, num_workers=0, collate_fn=collate_fn, pin_memory=True, drop_last=False)
            it = iter(loader)
            batch = next(it)
            img = batch['image'].to(device)
            if 'tissue_mask' not in batch:
                return False
            tmask = batch['tissue_mask']
            if isinstance(tmask, list):
                tmask = torch.stack(tmask, dim=0)
            tmask = tmask.to(device)
            model.train()
            for p in model.seg.parameters():
                p.requires_grad = True
            for p in model.vir.parameters():
                p.requires_grad = (unfreeze_last_n and unfreeze_last_n > 0)
            try:
                x224 = F.interpolate(img, size=(224, 224), mode='bilinear', align_corners=False)
                # If not unfrozen, avoid unnecessary gradient recording
                if unfreeze_last_n and unfreeze_last_n > 0:
                    tokens = model.vir.forward_features(x224)
                else:
                    with torch.no_grad():
                        tokens = model.vir.forward_features(x224)
                logits224, logits16 = model.seg(tokens)
                probs16 = torch.softmax(logits16, dim=1)
                tgt16 = F.interpolate(tmask.unsqueeze(1).float(), size=(16, 16), mode='nearest').squeeze(1).long()
                tgt16_oh = F.one_hot(tgt16, num_classes=probs16.size(1)).permute(0, 3, 1, 2).float()
                num = 2 * (probs16 * tgt16_oh).sum(dim=(2,3)) + 1.0
                den = (probs16.pow(2) + tgt16_oh.pow(2)).sum(dim=(2,3)) + 1.0
                loss = (1 - (num/den)).mean()
                loss.backward()
                # Do not take optimizer step to avoid modifying weights
                torch.cuda.synchronize()
                peak = torch.cuda.max_memory_allocated(device) / (1024**2)
                self.logger.info(f"[Pretrain BS Tuner] bs={bs} OK, peak VRAM~={peak:.1f} MB")
                return True
            except RuntimeError as e:
                if 'out of memory' in str(e).lower():
                    self.logger.info(f"[Pretrain BS Tuner] bs={bs} OOM")
                    return False
                else:
                    raise
            finally:
                del batch, img, tmask
                torch.cuda.empty_cache()

        # Exponential growth until failure or trial limit reached
        bs = start_bs
        trials = 0
        while trials < max_trials:
            ok = try_once(bs)
            if ok:
                good = bs
                bs *= 2
            else:
                bad = bs
                break
            trials += 1

        if good == 0:
            # Fall back to smaller range (search downward from start_bs)
            bs = max(1, start_bs // 2)
            while bs >= 1:
                if try_once(bs):
                    return bs
                bs //= 2
            return None

        if bad is None:
            # Never hit OOM, return the last successful value
            return good

        # Binary search between (good, bad)
        lo, hi = good + 1, bad - 1
        best = good
        while lo <= hi:
            mid = (lo + hi) // 2
            if try_once(mid):
                best = mid
                lo = mid + 1
            else:
                hi = mid - 1
        return best

    def _maybe_auto_tune_main_bs(self, train_dataset, collate_fn):
        if not torch.cuda.is_available():
            return None
        if not bool(self.config.get('auto_tune_main_bs', True)):
            return None
        if self.config.get('trainer_name') not in ('virchow_fused',):
            return None

        mode = str(self.config.get('mode', 'film_bias')).lower()
        use_film = (mode in ('film_only', 'film_bias'))
        use_bias = (mode in ('bias_only', 'film_bias'))

        temp_model = self.impl.build_model(train_dataset)
        try:
            tuned = self._auto_tune_main_bs(temp_model, train_dataset, collate_fn, use_film=use_film, use_bias=use_bias)
            if tuned and tuned > 0:
                tuned = int(tuned)
                self.config['batch_size'] = tuned
                self.logger.info(f"[AutoTune] Main training batch_size set to: {tuned}")
        except Exception as e:
            self.logger.warning(f"[AutoTune] Skipping main training batch probe: {e}")
        finally:
            try:
                temp_model.zero_grad(set_to_none=True)
            except Exception:
                pass

        return temp_model

    def _auto_sweep_postproc(self, model, val_loader):
        if not hasattr(self.impl, 'evaluate_with_postproc'):
            return None

        thr_candidates = self.config.get('autosweep_det_thresh_candidates', [0.25, 0.30, 0.35, 0.40, 0.45, 0.50])
        nms_candidates = self.config.get('autosweep_nms_candidates', [2, 3, 4, 5])
        match_candidates = self.config.get('autosweep_match_candidates', [4, 5, 6])

        best_metrics = None
        best_params = None
        for thr in thr_candidates:
            for nms in nms_candidates:
                for match_r in match_candidates:
                    metrics = self.impl.evaluate_with_postproc(model, val_loader, thr, nms, match_r, compute_extra=False)
                    if metrics is None:
                        continue
                    if best_metrics is None or metrics.get('overall_f1', 0.0) > best_metrics.get('overall_f1', 0.0):
                        best_metrics = metrics
                        best_params = (thr, nms, match_r)

        if best_metrics is None or best_params is None:
            return None

        detailed_metrics = self.impl.evaluate_with_postproc(model, val_loader, best_params[0], best_params[1], best_params[2], compute_extra=True)
        return detailed_metrics, best_params

    def _auto_tune_main_bs(self, model: nn.Module, dataset, collate_fn, use_film: bool, use_bias: bool, max_trials: int = 6):
        """Probe maximum batch size for the main training phase (full image or patch).
        Uses a single forward+backward pass with simple mean loss to trigger full backpropagation.
        """
        if not torch.cuda.is_available():
            return None
        device = self.device
        start_bs = int(self.config.get('batch_size', 4) or 4)
        good = 0
        bad = None

        def try_once(bs: int) -> bool:
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
            loader = DataLoader(dataset, batch_size=bs, shuffle=False, num_workers=0, collate_fn=collate_fn, pin_memory=True, drop_last=False)
            it = iter(loader)
            batch = next(it)
            img = batch['image'].to(device)
            try:
                model.train()
                if self.config.get('trainer_name') in ('virchow_fused',):
                    out = model(img, use_film=use_film, use_bias=use_bias)
                else:
                    out = model(img)
                loss = out['heatmap_logits'].mean()
                if 'class_logits' in out:
                    loss = loss + out['class_logits'].mean()
                loss.backward()
                torch.cuda.synchronize()
                peak = torch.cuda.max_memory_allocated(device) / (1024**2)
                self.logger.info(f"[Main BS Tuner] bs={bs} OK, peak VRAM~={peak:.1f} MB")
                return True
            except RuntimeError as e:
                if 'out of memory' in str(e).lower():
                    self.logger.info(f"[Main BS Tuner] bs={bs} OOM")
                    return False
                else:
                    raise
            finally:
                del batch, img
                torch.cuda.empty_cache()

        # Exponential growth
        bs = start_bs
        trials = 0
        while trials < max_trials:
            ok = try_once(bs)
            if ok:
                good = bs
                bs *= 2
            else:
                bad = bs
                break
            trials += 1

        if good == 0:
            bs = max(1, start_bs // 2)
            while bs >= 1:
                if try_once(bs):
                    return bs
                bs //= 2
            return None

        if bad is None:
            return good

        lo, hi = good + 1, bad - 1
        best = good
        while lo <= hi:
            mid = (lo + hi) // 2
            if try_once(mid):
                best = mid
                lo = mid + 1
            else:
                hi = mid - 1
        return best

    def train(self):
        """Train one fold."""
        self.logger.info(f"Starting training for Fold {self.fold_num}")

        # Data paths
        data_root = Path(self.config['data_root']) / f'fold_{self.fold_num}'
        save_dir = Path(self.config['save_root']) / f'fold_{self.fold_num}'
        save_dir.mkdir(parents=True, exist_ok=True)

        # Load data
        self.logger.info("Loading dataset...")
        data_mode = self.config.get('data_mode', 'patches')
        trainer_name = self.config.get('trainer_name')
        pretrain_epochs = int(self.config.get('pretrain_seg_epochs', 0) or 0)

        # Collate function selection
        collate_full = collate_point_batches
        collate_pch = collate_patches

        # Data loading strategy for Virchow fused:
        # - Pretraining phase needs tissue_mask (only patches mode provides it)
        # - Main training phase avoids loading tissue_mask (reduces IO overhead)
        if data_mode == 'full':
            # Full-size image dataset (no cropping). Optional resize to square (H=W=resize).
            resize = self.config.get('full_resize', None)
            # Tissue mask root directory: CLI first, then fold/tissue_masks
            tm_root_cfg = str(self.config.get('tissue_mask_root') or '').strip()
            tm_root_path = Path(tm_root_cfg) if tm_root_cfg else (data_root / 'tissue_masks')
            if tm_root_path.exists():
                tm_root = str(tm_root_path)
            else:
                tm_root = None
            # Main training/validation: load masks if available to ensure samples include tissue_mask
            with_mask = tm_root is not None
            use_aug = bool(self.config.get('augment', False))
            train_dataset = ShapesPointDataset(
                data_root, split='train', resize=resize,
                with_tissue_mask=with_mask, tissue_mask_root=tm_root,
                augment=use_aug,
            )
            val_dataset = ShapesPointDataset(
                data_root, split='test', resize=resize,
                with_tissue_mask=with_mask, tissue_mask_root=tm_root
            )
            if use_aug:
                self.logger.info("Data augmentation enabled (flip + rot90 + color jitter)")
            collate_fn = collate_full
            self.logger.info(f"Train set (Full{' +mask' if with_mask else ''}): {len(train_dataset)} samples, Val set (Full{' +mask' if with_mask else ''}): {len(val_dataset)} samples")
            train_loader = DataLoader(train_dataset, batch_size=self.config['batch_size'], shuffle=True, num_workers=self.config['num_workers'], collate_fn=collate_fn, pin_memory=True)
            val_loader = DataLoader(val_dataset, batch_size=self.config['batch_size'], shuffle=False, num_workers=self.config['num_workers'], collate_fn=collate_fn, pin_memory=True)
            # If full-size mask root is provided, build mask-aware loaders for pretraining and Dice evaluation
            val_dice_loader = None
            if tm_root is not None:
                self.logger.info(f"Detected full-size tissue masks: {tm_root}")
                bs_pre = int(self.config.get('pretrain_batch_size') or self.config['batch_size'])
                pre_train_dataset = ShapesPointDataset(data_root, split='train', resize=resize, with_tissue_mask=True, tissue_mask_root=tm_root)
                pre_val_dataset = ShapesPointDataset(data_root, split='test', resize=resize, with_tissue_mask=True, tissue_mask_root=tm_root)
                pre_train_loader = DataLoader(pre_train_dataset, batch_size=bs_pre, shuffle=True, num_workers=self.config['num_workers'], collate_fn=collate_fn, pin_memory=True)
                pre_val_loader = DataLoader(pre_val_dataset, batch_size=bs_pre, shuffle=False, num_workers=self.config['num_workers'], collate_fn=collate_fn, pin_memory=True)
                # Dedicated Dice evaluation loader (can reuse pre_val_loader)
                val_dice_loader = pre_val_loader
            else:
                pre_train_loader = None
                pre_val_loader = None
                # If user requested pretraining but no masks found, skip to avoid empty run
                if trainer_name in ('virchow_fused',) and pretrain_epochs > 0:
                    self.logger.warning("No tissue masks found, skipping Virchow segmentation branch pretraining.")
                    pretrain_epochs = 0
                    self.config['pretrain_seg_epochs'] = 0
        else:
            # Patches mode
            if trainer_name in ('virchow_fused',):
                # Check for tissue mask availability
                tm_train_dir = (data_root / 'train' / 'tissue_masks')
                tm_val_dir = (data_root / 'test' / 'tissue_masks')
                has_tm_train = tm_train_dir.exists() and any(tm_train_dir.glob('*.png'))
                has_tm_val = tm_val_dir.exists() and any(tm_val_dir.glob('*.png'))

                # Pretraining: needs masks; skip if not available
                if pretrain_epochs > 0 and not has_tm_train:
                    self.logger.warning("No training tissue_masks found, skipping Virchow segmentation branch pretraining.")
                    pretrain_epochs = 0
                    self.config['pretrain_seg_epochs'] = 0

                # Main training: do not load masks to reduce IO; validation loads masks if possible for Dice
                train_dataset = PatchesDataset(data_root, split='train', with_tissue_mask=False, augment=bool(self.config.get('augment', False)))
                val_dataset = PatchesDataset(data_root, split='test', with_tissue_mask=has_tm_val)
                collate_fn = collate_pch
                self.logger.info(f"Train set (Patches, no mask): {len(train_dataset)} samples")
                self.logger.info(f"Val set (Patches, {'with' if has_tm_val else 'no'} mask): {len(val_dataset)} samples")
                train_loader = DataLoader(train_dataset, batch_size=self.config['batch_size'], shuffle=True, num_workers=self.config['num_workers'], collate_fn=collate_fn, pin_memory=True)
                val_loader = DataLoader(val_dataset, batch_size=self.config['batch_size'], shuffle=False, num_workers=self.config['num_workers'], collate_fn=collate_fn, pin_memory=True)
                val_dice_loader = None

                # If pretraining needed, build separate mask-aware data loaders (used only during pretraining)
                if pretrain_epochs > 0:
                    pre_train_dataset = PatchesDataset(data_root, split='train', with_tissue_mask=True)
                    pre_val_dataset = PatchesDataset(data_root, split='test', with_tissue_mask=has_tm_val)
                    bs_pre = int(self.config.get('pretrain_batch_size') or self.config['batch_size'])
                    pre_train_loader = DataLoader(pre_train_dataset, batch_size=bs_pre, shuffle=True, num_workers=self.config['num_workers'], collate_fn=collate_fn, pin_memory=True)
                    pre_val_loader = DataLoader(pre_val_dataset, batch_size=bs_pre, shuffle=False, num_workers=self.config['num_workers'], collate_fn=collate_fn, pin_memory=True)
            else:
                # Non-fused trainer: default behaviour (load patches, do not force mask loading)
                train_dataset = PatchesDataset(data_root, split='train', with_tissue_mask=False, augment=bool(self.config.get('augment', False)))
                val_dataset = PatchesDataset(data_root, split='test', with_tissue_mask=False)
                collate_fn = collate_pch
                self.logger.info(f"Train set (Patches): {len(train_dataset)} samples, Val set (Patches): {len(val_dataset)} samples")
                train_loader = DataLoader(train_dataset, batch_size=self.config['batch_size'], shuffle=True, num_workers=self.config['num_workers'], collate_fn=collate_fn, pin_memory=True)
                val_loader = DataLoader(val_dataset, batch_size=self.config['batch_size'], shuffle=False, num_workers=self.config['num_workers'], collate_fn=collate_fn, pin_memory=True)

        prebuilt_model = self._maybe_auto_tune_main_bs(train_dataset, collate_fn)
        bs_main = int(self.config.get('batch_size', 1) or 1)
        if prebuilt_model is not None and int(self.config.get('batch_size', bs_main)) != bs_main:
            bs_main = int(self.config.get('batch_size', bs_main))

        # Build sampler: WeightedRandomSampler for rare-class oversampling, or shuffle
        rare_boost = float(self.config.get('oversample_rare', 0.0))
        train_sampler = None
        if rare_boost > 0:
            self.logger.info(f"Computing rare-class oversampling weights (boost={rare_boost}) ...")
            weights, class_counts = compute_sample_weights(train_dataset, rare_boost=rare_boost)
            train_sampler = WeightedRandomSampler(weights, num_samples=len(weights), replacement=True)
            n_boosted = sum(1 for w in weights if w > 1.0)
            self.logger.info(f"Oversampling: {n_boosted}/{len(weights)} samples boosted (contain rare classes), class distribution: {dict(class_counts)}")

        train_loader = DataLoader(
            train_dataset, batch_size=bs_main,
            shuffle=(train_sampler is None),
            sampler=train_sampler,
            num_workers=self.config['num_workers'],
            collate_fn=collate_fn, pin_memory=True,
        )
        val_loader = DataLoader(val_dataset, batch_size=bs_main, shuffle=False, num_workers=self.config['num_workers'], collate_fn=collate_fn, pin_memory=True)

        # Create model (delegated to specific trainer)
        self.logger.info(f"Creating model ({self.config['model_name']}, trainer={self.config['trainer_name']}) ...")
        model = prebuilt_model or self.impl.build_model(train_dataset)

        # Normalise class names (for logging and visualisation)
        class_names = getattr(train_dataset, 'class_names', None)
        if isinstance(class_names, dict):
            try:
                max_idx = max(int(k) for k in class_names.keys())
                class_names = [class_names.get(str(i+1), f"class_{i}") for i in range(max_idx)]
            except Exception:
                class_names = None
        if class_names is None:
            class_names = [f"class_{i}" for i in range(getattr(model, 'num_classes', 1))]
        self.class_names = class_names

        optimizer = optim.Adam(model.parameters(), lr=self.config['lr'])

        # Compute class weights for classification loss if requested
        if trainer_name in ('virchow_fused',):
            cw_mode = self.config.get('cls_class_weights')
            if cw_mode == 'auto':
                from collections import Counter
                counter = Counter()
                ann_dir = Path(self.config['data_root']) / f'fold_{self.fold_num}' / 'train' / 'ann'
                import json as _json
                for f in ann_dir.glob('*.json'):
                    with open(f) as fp:
                        data = _json.load(fp)
                    for c in data.get('centers', []):
                        counter[c['cls']] += 1
                total = sum(counter.values())
                num_c = getattr(train_dataset, 'num_classes', len(counter))
                weights = []
                for i in range(num_c):
                    cnt = counter.get(i, 1)
                    w = total / (num_c * max(cnt, 1))
                    weights.append(w)
                # Normalize so mean weight = 1.0
                mean_w = sum(weights) / len(weights)
                weights = [w / mean_w for w in weights]
                cw_tensor = torch.tensor(weights, dtype=torch.float32)
                self.impl._cls_class_weights = cw_tensor
                self.logger.info(f"[ClassWeights] auto computed: {[f'{w:.2f}' for w in weights]}")

        # Optional: Virchow Fused pretrain tissue segmentation branch (only when masks are available)
        if trainer_name in ('virchow_fused',):
            pe = pretrain_epochs
            if pe > 0 and hasattr(self.impl, 'pretrain_seg_head'):
                try:
                    self.logger.info(f"Starting tissue segmentation branch pretraining: {pe} epochs")
                    # Use dedicated pretraining loader (with masks); fall back to current loader
                    _tr_loader = locals().get('pre_train_loader', None) or train_loader
                    _va_loader = locals().get('pre_val_loader', None) or val_loader

                    # If pretraining loader exists and user did not explicitly provide pretrain_batch_size, auto-tune to maximise VRAM usage
                    try:
                        want_auto = (self.config.get('pretrain_batch_size') is None)
                        # Only auto-tune if loader's dataset contains tissue_mask
                        has_mask_field = False
                        for b in _va_loader:
                            has_mask_field = ('tissue_mask' in b)
                            break
                        if want_auto and has_mask_field and torch.cuda.is_available():
                            tuned_bs = self._auto_tune_pretrain_bs(model, _tr_loader.dataset, _tr_loader.collate_fn,
                                                                    unfreeze_last_n=int(self.config.get('pretrain_unfreeze_last_n', 0) or 0))
                            if tuned_bs is not None and tuned_bs > 0:
                                self.logger.info(f"[Pretrain BS Tuner] Recommended batch_size={tuned_bs} (original {self.config['batch_size']})")
                                # Rebuild pretraining loaders
                                from torch.utils.data import DataLoader as _DL
                                _tr_loader = _DL(_tr_loader.dataset, batch_size=tuned_bs, shuffle=True, num_workers=self.config['num_workers'], collate_fn=_tr_loader.collate_fn, pin_memory=True)
                                _va_loader = _DL(_va_loader.dataset, batch_size=tuned_bs, shuffle=False, num_workers=self.config['num_workers'], collate_fn=_va_loader.collate_fn, pin_memory=True)
                                self.config['pretrain_batch_size'] = tuned_bs
                    except Exception as _e:
                        self.logger.warning(f"[Pretrain BS Tuner] Skipping auto-tuning: {_e}")
                    # If masks are still missing, the trainer will skip Dice evaluation internally
                    self.impl.pretrain_seg_head(
                        model,
                        _tr_loader,
                        _va_loader,
                        epochs=pe,
                        lr=float(self.config.get('pretrain_lr', 1e-3)),
                        unfreeze_last_n=int(self.config.get('pretrain_unfreeze_last_n', 0) or 0)
                    )
                except Exception as e:
                    self.logger.warning(f"Segmentation branch pretraining failed: {e}")

        # LR scheduler setup
        lr_sched_name = str(self.config.get('lr_schedule', 'none')).lower()
        lr_scheduler = None
        if lr_sched_name == 'cosine':
            warmup_ep = int(self.config.get('lr_warmup_epochs', 0))
            main_ep = max(1, self.config['epochs'] - warmup_ep)
            lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer, T_max=main_ep, eta_min=self.config['lr'] * 0.01
            )
            self.logger.info(f"[LR] Cosine schedule: warmup={warmup_ep}, T_max={main_ep}")
        elif lr_sched_name == 'step':
            lr_scheduler = torch.optim.lr_scheduler.StepLR(
                optimizer, step_size=max(1, self.config['epochs'] // 3), gamma=0.1
            )

        # Training loop
        start_time = time.time()
        epoch_records = []
        for epoch in range(1, self.config['epochs'] + 1):
            epoch_start = time.time()

            # Set current epoch for FiLM warmup / bias warmup within the trainer
            try:
                self.impl.config['_current_epoch'] = epoch
            except Exception:
                pass
            train_loss, det_loss, cls_loss = self.impl.train_epoch(model, train_loader, optimizer)

            # Evaluation
            autosweep_enabled = bool(self.config.get('autosweep_postproc', False))
            if autosweep_enabled:
                sweep = self._auto_sweep_postproc(model, val_loader)
                if sweep is not None:
                    val_metrics, best_params = sweep
                    self.config['det_thresh'], self.config['nms_radius'], self.config['match_radius'] = best_params
                    self.logger.info(
                        f"[AutoSweep] Selected post-processing: thr={best_params[0]:.2f} nms={best_params[1]} match={best_params[2]} -> F1={val_metrics.get('overall_f1',0):.4f}"
                    )
                else:
                    val_metrics = self.impl.evaluate(model, val_loader)
            else:
                # Set current epoch before evaluation to keep train/eval using same FiLM switch
                try:
                    self.impl.config['_current_epoch'] = epoch
                except Exception:
                    pass
                val_metrics = self.impl.evaluate(model, val_loader)
            # If main validation has no masks but we have a dedicated Dice validation set, compute Dice separately
            try:
                if ('tissue_dice_per_class' not in val_metrics) and ('val_dice_loader' in locals()) and (val_dice_loader is not None):
                    dice_pc = self.impl._eval_tissue_dice(model, val_dice_loader)
                    val_metrics['tissue_dice_per_class'] = dice_pc
                    if len(dice_pc) > 0:
                        import numpy as _np
                        val_metrics['tissue_dice_mean'] = float(_np.mean(dice_pc))
            except Exception as _e:
                self.logger.warning(f"Supplementary Dice evaluation failed: {_e}")

            # Logging
            epoch_time = time.time() - epoch_start
            dist_log = getattr(self.impl, '_last_dist_loss', 0.0)
            msg = (
                f"Epoch {epoch}/{self.config['epochs']} ({epoch_time:.1f}s) - "
                f"Loss: {train_loss:.4f} (Det: {det_loss:.4f}, Cls: {cls_loss:.4f}, Dist: {dist_log:.4f}) - "
                f"Val Det P={val_metrics.get('overall_precision',0):.3f}, R={val_metrics.get('overall_recall',0):.3f}, F1={val_metrics.get('overall_f1',0):.3f} - "
                f"Val Cls Acc={val_metrics.get('cls_acc',0):.3f}"
            )
            self.logger.info(msg)
            # Append tissue segmentation Dice metrics if available
            if 'tissue_dice_per_class' in val_metrics:
                dice_list = val_metrics['tissue_dice_per_class'] or []
                mean_dice = val_metrics.get('tissue_dice_mean', float('nan'))
                try:
                    dice_str = ", ".join([
                        f"{self.tissue_names[idx]}:{d:.3f}" for idx, d in enumerate(dice_list)
                    ])
                except Exception:
                    dice_str = str(dice_list)
                self.logger.info(f"Val Tissue Dice (mean {mean_dice:.3f}): {dice_str}")

            # Brief per-class classification accuracy (up to first 10 classes)
            if 'cls_acc_per_class' in val_metrics:
                per_cls_acc = val_metrics['cls_acc_per_class']
                tops = min(len(per_cls_acc), 10)
                acc_str = ", ".join(
                    f"{self.class_names[i]}:{per_cls_acc[i]:.2f}" for i in range(tops)
                )
                self.logger.info(f"Val Cls Acc per-class: {acc_str}")

            # Step LR scheduler
            if lr_scheduler is not None:
                warmup_ep = int(self.config.get('lr_warmup_epochs', 0))
                if epoch > warmup_ep:
                    lr_scheduler.step()

            # Record to memory
            epoch_records.append({
                'epoch': epoch,
                'loss': float(train_loss),
                'det_loss': float(det_loss),
                'cls_loss': float(cls_loss),
                'precision': float(val_metrics.get('overall_precision', 0.0)),
                'recall': float(val_metrics.get('overall_recall', 0.0)),
                'f1': float(val_metrics.get('overall_f1', 0.0)),
                'cls_acc': float(val_metrics.get('cls_acc', 0.0)),
                'cls_acc_per_class': val_metrics.get('cls_acc_per_class', None),
                'det_recall_per_class': val_metrics.get('det_recall_per_class', None),
                'tissue_dice_per_class': val_metrics.get('tissue_dice_per_class', None),
                'tissue_dice_mean': float(val_metrics.get('tissue_dice_mean', np.nan)) if 'tissue_dice_mean' in val_metrics else None,
            })

            # Save best model
            if val_metrics['overall_f1'] > self.best_f1:
                self.best_f1 = val_metrics['overall_f1']
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'best_f1': self.best_f1
                }, save_dir / 'best_model.pt')
                self.logger.info(f"Saved best model (F1: {self.best_f1:.3f})")

                # Export best epoch raw predictions and visualisation (optional)
                if self.config.get('export_best', False):
                    try:
                        export_dir = save_dir / f"best_epoch_{epoch:03d}"
                        export_dir.mkdir(parents=True, exist_ok=True)
                        self._export_predictions_and_viz(model, val_loader, export_dir)
                        self.logger.info(f"Exported best epoch raw predictions and visualisation to: {export_dir}")
                    except Exception as e:
                        self.logger.warning(f"Failed to export best epoch predictions: {e}")
                else:
                    self.logger.info("export_best=False, skipping best epoch raw prediction export")

            # Always save the latest (last) checkpoint
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'best_f1': self.best_f1
            }, save_dir / 'last_model.pt')

            # Save checkpoint at fixed interval
            se = int(self.config.get('save_every', 0) or 0)
            if se > 0 and (epoch % se == 0):
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'best_f1': self.best_f1
                }, save_dir / f'epoch_{epoch:03d}.pt')

        total_time = time.time() - start_time
        # Save epoch metrics CSV and curves
        try:
            import csv
            metrics_csv = save_dir / 'metrics_epoch.csv'
            with open(metrics_csv, 'w', newline='') as f:
                writer = csv.DictWriter(f, fieldnames=['epoch','loss','det_loss','cls_loss','precision','recall','f1','cls_acc'])
                writer.writeheader()
                for r in epoch_records:
                    writer.writerow({k: r.get(k) for k in ['epoch','loss','det_loss','cls_loss','precision','recall','f1','cls_acc']})
            # Generate curve plots
            try:
                import matplotlib
                matplotlib.use('Agg')
                import matplotlib.pyplot as plt
                xs = [r['epoch'] for r in epoch_records]
                f1s = [r['f1'] for r in epoch_records]
                accs = [r['cls_acc'] for r in epoch_records]
                fig, ax = plt.subplots(1, 1, figsize=(6, 4))
                ax.plot(xs, f1s, label='Detection F1')
                ax.plot(xs, accs, label='Classification Acc')
                ax.set_xlabel('Epoch')
                ax.set_ylabel('Score')
                ax.set_title('Validation Curves')
                ax.grid(True)
                ax.legend()
                fig.tight_layout()
                fig.savefig(save_dir / 'metrics_curves.png', dpi=150)
                plt.close(fig)
                # Per-class classification accuracy curves (if available)
                if any(r.get('cls_acc_per_class') is not None for r in epoch_records):
                    C = len(epoch_records[-1].get('cls_acc_per_class') or [])
                    if C > 0:
                        fig2, ax2 = plt.subplots(1, 1, figsize=(7, 5))
                        import numpy as _np
                        for ci in range(C):
                            ys = [ (r['cls_acc_per_class'][ci] if r.get('cls_acc_per_class') else _np.nan) for r in epoch_records ]
                            ax2.plot(xs, ys, label=self.class_names[ci] if ci < len(self.class_names) else f"c{ci}")
                        ax2.set_xlabel('Epoch')
                        ax2.set_ylabel('Cls Acc (matched)')
                        ax2.set_title('Per-class Classification Accuracy')
                        ax2.grid(True)
                        ax2.legend(fontsize=7, ncol=2)
                        fig2.tight_layout()
                        fig2.savefig(save_dir / 'cls_acc_per_class_curves.png', dpi=150)
                        plt.close(fig2)
                # Per-class Tissue Dice curves (if available)
                if any(r.get('tissue_dice_per_class') is not None for r in epoch_records):
                    T = len(epoch_records[-1].get('tissue_dice_per_class') or [])
                    if T > 0:
                        fig3, ax3 = plt.subplots(1, 1, figsize=(7, 5))
                        import numpy as _np
                        for ti in range(T):
                            ys = [ (r['tissue_dice_per_class'][ti] if r.get('tissue_dice_per_class') else _np.nan) for r in epoch_records ]
                            name = self.tissue_names[ti] if ti < len(self.tissue_names) else f"tissue_{ti}"
                            ax3.plot(xs, ys, label=name)
                        ax3.set_xlabel('Epoch')
                        ax3.set_ylabel('Dice')
                        ax3.set_title('Per-class Tissue Dice')
                        ax3.grid(True)
                        ax3.legend(fontsize=7, ncol=2)
                        fig3.tight_layout()
                        fig3.savefig(save_dir / 'tissue_dice_per_class_curves.png', dpi=150)
                        plt.close(fig3)
                self.logger.info(f"Saved curves: {save_dir / 'metrics_curves.png'}")
            except Exception as e:
                self.logger.warning(f"Failed to generate curve plots (CSV saved): {e}")
        except Exception as e:
            self.logger.warning(f"Failed to save epoch metrics: {e}")
        self.logger.info(f"Fold {self.fold_num} training complete, elapsed: {total_time/60:.1f} minutes, best F1: {self.best_f1:.3f}")

        return self.best_f1, total_time

    @torch.no_grad()
    def _export_predictions_and_viz(self, model: nn.Module, val_loader: DataLoader, out_dir: Path):
        """Export raw predictions (npz) and visualisations (overlay) for the validation set.
        - Raw predictions: heat_prob, class_prob/_logits (if available), tissue_prob (if available)
        - Visualisation: original image, raw heatmap overlay, post-processed points+class overlay, (optional) tissue overlay
        Note: post-processing is only used for visualisation; raw npz files contain no post-processing.
        """
        raw_dir = out_dir / 'predictions_raw_npz'
        viz_raw_dir = out_dir / 'viz' / 'overlay_raw_heatmap'
        viz_pts_dir = out_dir / 'viz' / 'overlay_postproc_points'
        viz_tis_dir = out_dir / 'viz' / 'tissue_overlay'
        for d in [raw_dir, viz_raw_dir, viz_pts_dir, viz_tis_dir]:
            d.mkdir(parents=True, exist_ok=True)

        # Colours and legend
        class_colors = default_class_colors(len(self.class_names))
        try:
            save_legend(str(out_dir / 'legend_classes.png'), self.class_names, class_colors, title='Nuclei Classes')
        except Exception:
            pass
        tissue_colors = default_class_colors(int(self.config.get('num_tissue', 6)))
        try:
            save_legend(str(out_dir / 'legend_tissue.png'), self.tissue_names[:len(tissue_colors)], tissue_colors, title='Tissue Classes')
        except Exception:
            pass

        from tand.evaluation.peak import detect_peaks
        import numpy as np
        import torch

        model.eval()
        det_thresh = float(self.config.get('det_thresh', 0.35))
        nms_radius = int(self.config.get('nms_radius', 3))

        for batch in val_loader:
            images = batch["image"].to(self.device)
            stems = batch.get('stem', [f'sample_{i}' for i in range(images.size(0))])

            # Forward pass: consistent with evaluation; Virchow Fused uses mode to decide FiLM
            if self.config.get('trainer_name') in ('virchow_fused',):
                mode_cfg = str(self.config.get('mode', 'film_bias')).lower()
                film_enabled_by_mode = (mode_cfg in ('film_only', 'film_bias'))
                film_start_ep = int(self.config.get('film_start_epoch', 0) or 0)
                cur_ep = int(self.config.get('_current_epoch', 0) or 0)
                use_film = film_enabled_by_mode and (cur_ep >= film_start_ep)
                bias_enabled_by_mode = (mode_cfg in ('bias_only', 'film_bias'))
                use_bias = bias_enabled_by_mode and (cur_ep >= film_start_ep)
                out = model(images, use_film=use_film, use_bias=use_bias)
            else:
                out = model(images)

            # Standardise outputs
            heat_prob = None
            class_logits = None
            class_prob = None
            tissue_prob = None

            if 'heatmap_logits' in out:
                heat_prob = torch.sigmoid(out['heatmap_logits']).detach().cpu().numpy()  # (B,1,H,W)
            elif 'heatmap_logits_multi' in out:
                hp = torch.max(out['heatmap_logits_multi'], dim=1, keepdim=True).values
                heat_prob = torch.sigmoid(hp).detach().cpu().numpy()

            if 'class_logits' in out:
                class_logits = out['class_logits'].detach().cpu().numpy()
                class_prob = torch.sigmoid(out['class_logits']).detach().cpu().numpy()
            elif 'heatmap_logits_multi' in out:
                class_prob = torch.sigmoid(out['heatmap_logits_multi']).detach().cpu().numpy()

            if 'tissue_logits_224' in out:
                tp = torch.softmax(out['tissue_logits_224'], dim=1).detach().cpu().numpy()
                tissue_prob = tp

            B = images.size(0)
            for b in range(B):
                stem = stems[b]
                if isinstance(stem, bytes):
                    stem = stem.decode('utf-8')
                rgb = to_uint8_rgb(images[b].cpu())
                # Save raw npz (no post-processing)
                np.savez_compressed(
                    raw_dir / f"{stem}.npz",
                    heat_prob=(heat_prob[b] if heat_prob is not None else None),
                    class_logits=(class_logits[b] if class_logits is not None else None),
                    class_prob=(class_prob[b] if class_prob is not None else None),
                    tissue_prob=(tissue_prob[b] if tissue_prob is not None else None),
                    meta=dict(stem=stem)
                )

                # Raw heatmap overlay
                if heat_prob is not None:
                    hm = heat_prob[b, 0]
                    vis_raw = overlay_heatmap(rgb, hm, alpha=0.45)
                    vis_raw.save(viz_raw_dir / f"{stem}.png")

                # Post-processed points + class (visualisation only)
                if heat_prob is not None:
                    hm = heat_prob[b, 0]
                    import torch as _torch
                    peaks = detect_peaks(_torch.from_numpy(hm), thresh=det_thresh, nms_radius=nms_radius)
                    pred_pts = np.array([[x, y] for x, y, _ in peaks], dtype=np.float32) if peaks else np.zeros((0, 2), dtype=np.float32)
                    if pred_pts.size > 0 and (class_prob is not None):
                        H, W = hm.shape
                        labels = []
                        cp = class_prob[b]  # (C,H,W)
                        for (x, y) in pred_pts:
                            xi, yi = int(min(max(x, 0), W - 1)), int(min(max(y, 0), H - 1))
                            scores = cp[:, yi, xi]
                            labels.append(int(np.argmax(scores)))
                        labels = np.array(labels, dtype=np.int64)
                    else:
                        labels = np.zeros((len(pred_pts),), dtype=np.int64)
                    vis_pts = draw_points(rgb, pred_pts, labels, class_colors, radius=3)
                    vis_pts.save(viz_pts_dir / f"{stem}_thr{det_thresh:.2f}_nms{nms_radius}.png")

                # Tissue overlay
                if tissue_prob is not None:
                    t_map = np.argmax(tissue_prob[b], axis=0).astype(np.int32)
                    # Resize seg map to match RGB size if needed (avoid broadcasting error)
                    # Note: rgb.size = (W, H), t_map.shape = (H, W)
                    if (t_map.shape[1], t_map.shape[0]) != rgb.size:
                        from PIL import Image as _Image
                        _t_img = _Image.fromarray(t_map.astype(np.int32, copy=False))
                        _t_img = _t_img.resize(rgb.size, resample=_Image.NEAREST)
                        t_map = np.array(_t_img, dtype=np.int32)
                    vis_t = overlay_segmentation(rgb, t_map, tissue_colors, alpha=0.35)
                    vis_t.save(viz_tis_dir / f"{stem}.png")


def get_optimal_batch_size(data_mode: str = 'patches'):
    """Recommend batch size based on GPU memory and data mode.
    patches: follow original logic
    full:    full-size images (e.g. 1024x1024) consume more memory, use conservative settings
    """
    if not torch.cuda.is_available():
        return 4 if data_mode == 'full' else 16

    props = torch.cuda.get_device_properties(0)
    memory_gb = props.total_memory / (1024**3)

    if data_mode == 'full':
        # Conservative settings for full-size images (e.g. L4 22GB -> 2)
        if memory_gb >= 80:
            return 8
        elif memory_gb >= 40:
            return 4
        elif memory_gb >= 20:
            return 2
        else:
            return 1
    else:
        if memory_gb >= 80:
            return 256
        elif memory_gb >= 40:
            return 128
        elif memory_gb >= 20:
            return 64
        elif memory_gb >= 10:
            return 32
        else:
            return 16


def main():
    """Main function."""
    parser = argparse.ArgumentParser(description='5-fold cross-validation training script')
    parser.add_argument('--batch-size', type=int, default=None, help='Batch size')
    parser.add_argument('--epochs', type=int, default=120, help='Number of training epochs')
    parser.add_argument('--no-confirm', action='store_true', help='Skip confirmation step')
    parser.add_argument('--model', type=str, default='virchow_fused', choices=['efficient', 'dino', 'virchow_fused'], help='Model architecture')
    parser.add_argument('--trainer', type=str, default=None, choices=['virchow_fused'], help='Optional: explicitly specify trainer (default: auto-select based on model)')
    parser.add_argument('--dino-variant', type=str, default='convnext_small', help='timm ConvNeXt variant (dino)')
    parser.add_argument('--dino-up', type=str, default='upsample', choices=['upsample','convtrans'], help='DINO decoder upsampling method')
    parser.add_argument('--dino-load-hf', action='store_true', help='Try to load HF pretrained weights (requires huggingface_hub)')
    # Fusion model choice for virchow_fused trainer
    parser.add_argument('--fusion-model', type=str, default='efficient', choices=['efficient', 'dino'], help='Detection backbone: efficient (EfficientUNet) or dino (DINOv3-ConvNeXt)')
    # Detection / evaluation thresholds (CLI overrides defaults)
    parser.add_argument('--det-thr', type=float, default=None, help='Detection peak threshold (default 0.35)')
    parser.add_argument('--nms-radius', type=int, default=None, help='Peak NMS radius (default 3)')
    parser.add_argument('--match-radius', type=int, default=None, help='Validation match radius (default 5)')
    # Data-related arguments
    parser.add_argument('--data-root', type=str, default=None, help='Data root directory, supports patches or full-image fold structure (env var TAND_DATA_ROOT overrides)')
    parser.add_argument('--data-mode', type=str, default=None, choices=['patches','full'], help='Data mode: 224x224 patches or original full images (auto-inferred if not specified)')
    parser.add_argument('--full-resize', type=int, default=None, help='Full-image mode: resize to square of given side length, e.g. 1024')
    parser.add_argument('--only-fold', type=int, default=None, help='Train only the specified fold number (1-5)')
    # Checkpointing
    parser.add_argument('--save-every', type=int, default=0, help='Save checkpoint every N epochs (0 = no extra checkpoints)')
    # Virchow fused specific
    parser.add_argument('--num-tissue', type=int, default=6, help='Number of tissue classes (Virchow Fused)')
    parser.add_argument('--film-limit', type=float, default=0.5, help='FiLM gamma/beta clamp limit (Virchow Fused)')
    parser.add_argument('--film-scales', type=str, default='16,8,4', help='Active FiLM scales, comma-separated (e.g. "16,8,4" or "16,8")')
    parser.add_argument('--lam-bias', type=float, default=0.8, help='Posterior bias strength lambda (Virchow Fused)')
    parser.add_argument('--conf-thr', type=float, default=0.7, help='Tissue probability confidence threshold gate (Virchow Fused)')
    parser.add_argument('--tau', type=float, default=1.0, help='Tissue softmax temperature tau (Virchow Fused)')
    parser.add_argument('--heat-focal-alpha', type=float, default=0.25, help='Detection focal alpha (Virchow Fused)')
    parser.add_argument('--heat-focal-gamma', type=float, default=2.0, help='Detection focal gamma (Virchow Fused)')
    parser.add_argument('--cls-bce-pos-weight', type=float, default=1.0, help='Classification BCE positive weight (Virchow Fused)')
    parser.add_argument('--seg-loss-weight', type=float, default=0.3, help='Tissue segmentation CE loss weight (Virchow Fused)')
    parser.add_argument('--heat-loss-weight', type=float, default=1.0, help='Detection loss weight (Virchow Fused)')
    parser.add_argument('--cls-loss-weight', type=float, default=1.0, help='Classification loss weight (Virchow Fused)')
    parser.add_argument('--lambda_P', type=float, default=0.2, help='Distribution consistency loss lambda_P (Virchow Fused)')
    parser.add_argument('--mode', type=str, default='film_bias', choices=['baseline','film_only','bias_only','film_bias'], help='Fusion mode (Virchow Fused)')
    parser.add_argument('--film-start-epoch', type=int, default=0, help='Epoch to start enabling FiLM (for warmup; disabled before this epoch)')
    # Virchow fused pretrain options
    parser.add_argument('--pretrain-seg-epochs', type=int, default=0, help='Epochs for tissue segmentation branch pretraining (Virchow Fused)')
    parser.add_argument('--pretrain-batch-size', type=int, default=None, help='Batch size for segmentation pretraining (default: same as main training)')
    parser.add_argument('--pretrain-unfreeze-last-n', type=int, default=0, help='Unfreeze last N Virchow blocks during pretraining (0 = none)')
    parser.add_argument('--pretrain-lr', type=float, default=1e-3, help='Segmentation pretraining learning rate')
    parser.add_argument('--pretrain-lr-schedule', type=str, default='none', choices=['none','cosine'], help='Segmentation pretraining LR schedule')
    parser.add_argument('--pretrain-seg-dice-weights', type=str, default='auto', help="Pretraining segmentation Dice channel weights: 'auto'|'uniform'|comma-separated values")
    parser.add_argument('--tissue-mask-root', type=str, default=None, help='Full-size tissue masks root directory (for full-mode segmentation pretraining and Dice evaluation)')
    parser.add_argument('--pretrain-earlystop-patience', type=int, default=5, help='Stop pretraining early if Dice does not improve for this many epochs')
    parser.add_argument('--pretrain-earlystop-min-delta', type=float, default=0.001, help='Dice improvement threshold for early stopping')
    parser.add_argument('--pretrain-earlystop-degrade-tolerance', type=int, default=3, help='Tolerance for consecutive Dice degradation before early stopping')
    parser.add_argument('--pretrain-dice-weight', type=float, default=0.6, help='Tissue pretraining Dice loss weight')
    parser.add_argument('--pretrain-ce-weight', type=float, default=0.4, help='Tissue pretraining 16x16 CrossEntropy weight')
    parser.add_argument('--pretrain-aux-ce-weight', type=float, default=0.1, help='Tissue pretraining 224x224 auxiliary CrossEntropy weight')
    parser.add_argument('--pretrain-label-smoothing', type=float, default=0.0, help='Tissue pretraining CrossEntropy label smoothing')
    parser.add_argument('--pretrain-grad-accum', type=int, default=1, help='Tissue pretraining gradient accumulation steps')
    parser.add_argument('--pretrain-grad-clip', type=float, default=0.0, help='Tissue pretraining gradient clipping threshold (0 = disabled)')
    parser.add_argument('--pretrain-min-epochs', type=int, default=None, help='Tissue pretraining minimum epochs (None = auto)')
    parser.add_argument('--pretrain-amp', type=str, default='auto', choices=['auto','on','off'], help='Tissue pretraining AMP mode: auto/on/off')
    parser.add_argument('--auto-tune-main-bs', action='store_true', help='Enable: auto-tune batch size for main training (single batch probe)')
    parser.add_argument('--prior-path', type=str, default=None, help='Path to log P(c|t) prior matrix for Virchow bias')
    parser.add_argument('--export-best', action='store_true', help='Export best epoch raw predictions and visualisations (disabled by default to save storage/IO)')
    parser.add_argument('--run-tag', type=str, default=None, help='Optional: custom timestamp/tag for this training run')
    parser.add_argument('--train-amp', type=str, default='auto', choices=['auto','on','off'], help='Main training AMP mode: auto/on/off')
    # Class-weighted loss for minority class improvement
    parser.add_argument('--cls-focal-gamma', type=float, default=0.0, help='Classification focal loss gamma (0=disabled, 2.0 recommended)')
    parser.add_argument('--cls-label-smoothing', type=float, default=0.0, help='Classification label smoothing')
    parser.add_argument('--cls-class-weights', type=str, default=None, help='Class weights: "auto" for inverse-frequency, or comma-separated values')
    parser.add_argument('--lr-schedule', type=str, default='none', choices=['none','cosine','step'], help='Main training LR schedule')
    parser.add_argument('--lr-warmup-epochs', type=int, default=0, help='LR warmup epochs for cosine schedule')
    # Data augmentation and rare-class oversampling (v7)
    parser.add_argument('--augment', action='store_true', help='Enable data augmentation (flip, rot90, color jitter) for training')
    parser.add_argument('--oversample-rare', type=float, default=0.0, help='Rare-class oversampling boost factor (e.g. 3.0). 0=disabled.')
    args = parser.parse_args()

    print("\n" + "="*60)
    print("5-Fold Cross-Validation Training")
    print("="*60)

    # Create timestamp
    if args.run_tag:
        timestamp = args.run_tag.strip()
        print(f"Using custom run-tag: {timestamp}")
    else:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Check GPU
    if torch.cuda.is_available():
        gpu_count = torch.cuda.device_count()
        print(f"\nDetected {gpu_count} GPU(s):")
        for i in range(gpu_count):
            props = torch.cuda.get_device_properties(i)
            memory_gb = props.total_memory / (1024**3)
            print(f"  GPU {i}: {props.name}, {memory_gb:.1f}GB")
    else:
        print("\nWarning: No GPU detected, will use CPU for training")

    # Data root resolution
    # Priority: CLI --data-root > env var TAND_DATA_ROOT > model-based default
    env_data_root = os.environ.get('TAND_DATA_ROOT')
    project_root = Path(__file__).resolve().parents[1]
    if args.data_root is not None:
        data_root = Path(args.data_root)
    elif env_data_root:
        data_root = Path(env_data_root)
    else:
        # Model-specific defaults: DINO/virchow_fused default to full images; others to patches
        default_dirname = "data/puma_coco_folds" if args.model in ('dino', 'virchow_fused') else "data/puma_coco_folds_224x224_patches"
        candidate_project = project_root / default_dirname
        candidate_cwd = Path(default_dirname)
        data_root = candidate_project if candidate_project.exists() else candidate_cwd

    # Infer/set data mode
    if args.data_mode is not None:
        data_mode = args.data_mode
    else:
        # If not specified, auto-select based on model: DINO/virchow_fused -> full; otherwise heuristic
        if args.model in ('dino', 'virchow_fused'):
            data_mode = 'full'
        else:
            # Directory name containing 'patch' treated as patches, otherwise full
            data_mode = 'patches' if 'patch' in str(data_root).lower() else 'full'

    # DINO / Virchow-fused default: train on full images at 1024x1024 (unless user explicitly overrides)
    default_full_resize = None
    if args.model in ('dino', 'virchow_fused') and data_mode == 'full':
        if args.full_resize is None:
            default_full_resize = 1024
            print("\nDINO/VirchowFused model: --full-resize not specified, defaulting to 1024x1024 training.")
    if not data_root.exists():
        print(f"\nError: Data directory does not exist: {data_root}")
        if data_mode == 'patches':
            print("Please run preprocessing to generate 224x224 patches first.")
        else:
            print("Please verify the full-image data directory path.")
        return

    # Get recommended batch size
    recommended_bs = get_optimal_batch_size(data_mode=data_mode)
    print(f"\nRecommended batch size: {recommended_bs}")

    # Use command line argument or recommended value
    if args.batch_size is not None:
        batch_size = args.batch_size
        print(f"Using specified batch size: {batch_size}")
    else:
        batch_size = recommended_bs
        print(f"Using recommended batch size: {batch_size}")

    epochs = args.epochs
    print(f"Training epochs: {epochs}")
    print(f"Model: {args.model}")
    if args.trainer:
        print(f"Trainer: {args.trainer}")
    print(f"Data mode: {data_mode}")
    print(f"Data root: {data_root}")
    # Resolve final full_resize (CLI first, then DINO default 1024)
    final_full_resize = args.full_resize if args.full_resize is not None else default_full_resize
    if data_mode == 'full' and final_full_resize:
        print(f"Images will be resized to: {final_full_resize}x{final_full_resize}")

    # Configuration
    config = {
        'data_root': str(data_root),
        'save_root': f'experiments/all_folds_bs{batch_size}_{timestamp}',
        'log_dir': Path(f'logs/all_folds_{timestamp}'),
        'batch_size': batch_size,
        'epochs': epochs,
        'lr': 0.001,
        'num_workers': min(16, os.cpu_count() or 1),
        'device': 'cuda',
        'det_weight': 1.0,
        'lambda_cls': 1.0,
        'det_thresh': (args.det_thr if args.det_thr is not None else 0.35),
        'nms_radius': (args.nms_radius if args.nms_radius is not None else 3),
        'match_radius': (args.match_radius if args.match_radius is not None else 5),
        'model_name': args.model,
        'trainer_name': args.trainer,
        # Data mode configuration
        'data_mode': data_mode,
        'full_resize': final_full_resize,
        # DINO-related optional parameters (ignored for other models)
        'dino_variant': args.dino_variant,
        'dino_up_mode': args.dino_up,
        'dino_load_hf': args.dino_load_hf,
        # Checkpoint options
        'save_every': args.save_every,
        # Virchow fused options
        'num_tissue': args.num_tissue,
        'film_limit': args.film_limit,
        'film_scales': args.film_scales,
        'lam_bias': args.lam_bias,
        'conf_thr': args.conf_thr,
        'tau': args.tau,
        'heat_focal_alpha': args.heat_focal_alpha,
        'heat_focal_gamma': args.heat_focal_gamma,
        'cls_bce_pos_weight': args.cls_bce_pos_weight,
        'seg_loss_weight': args.seg_loss_weight,
        'heat_loss_weight': args.heat_loss_weight,
        'cls_loss_weight': args.cls_loss_weight,
        'lambda_P': args.lambda_P,
        'mode': args.mode,
        'film_start_epoch': args.film_start_epoch,
        'run_tag': timestamp,
        'pretrain_seg_epochs': args.pretrain_seg_epochs,
        'pretrain_unfreeze_last_n': args.pretrain_unfreeze_last_n,
        'pretrain_lr': args.pretrain_lr,
        'pretrain_lr_schedule': args.pretrain_lr_schedule,
        'pretrain_seg_dice_weights': args.pretrain_seg_dice_weights,
        'pretrain_batch_size': args.pretrain_batch_size,
        'pretrain_earlystop_patience': args.pretrain_earlystop_patience,
        'pretrain_earlystop_min_delta': args.pretrain_earlystop_min_delta,
        'pretrain_earlystop_degrade_tolerance': args.pretrain_earlystop_degrade_tolerance,
        'pretrain_dice_weight': args.pretrain_dice_weight,
        'pretrain_ce_weight': args.pretrain_ce_weight,
        'pretrain_aux_ce_weight': args.pretrain_aux_ce_weight,
        'pretrain_label_smoothing': args.pretrain_label_smoothing,
        'pretrain_grad_accum': args.pretrain_grad_accum,
        'pretrain_grad_clip': args.pretrain_grad_clip,
        'pretrain_min_epochs': args.pretrain_min_epochs,
        'pretrain_amp': args.pretrain_amp,
        'auto_tune_main_bs': args.auto_tune_main_bs,
        # Fusion model selection for Virchow fused trainer
        'fusion_model': args.fusion_model,
        'tissue_mask_root': args.tissue_mask_root,
        'prior_path': args.prior_path,
        'export_best': bool(args.export_best),
        'train_amp': args.train_amp,
        # Class-weighted loss options
        'cls_focal_gamma': args.cls_focal_gamma,
        'cls_label_smoothing': args.cls_label_smoothing,
        'cls_class_weights': args.cls_class_weights,
        'lr_schedule': args.lr_schedule,
        'lr_warmup_epochs': args.lr_warmup_epochs,
        # Data augmentation and rare-class oversampling
        'augment': args.augment,
        'oversample_rare': args.oversample_rare,
    }

    # Create directories
    config['log_dir'].mkdir(parents=True, exist_ok=True)
    Path(config['save_root']).mkdir(parents=True, exist_ok=True)

    # Save configuration
    config_to_save = config.copy()
    config_to_save['log_dir'] = str(config['log_dir'])  # Convert Path to string for JSON
    with open(config['log_dir'] / 'config.json', 'w') as f:
        json.dump(config_to_save, f, indent=2)

    print(f"\nConfiguration saved to: {config['log_dir'] / 'config.json'}")

    # Build planned folds list based on --only-fold
    planned_folds = [args.only_fold] if args.only_fold is not None else list(range(1, 6))
    if len(planned_folds) == 1:
        confirm_prompt = f"\nStart training fold {planned_folds[0]}? [Y/n]: "
        auto_start_msg = f"\nStarting training for fold {planned_folds[0]}..."
    else:
        confirm_prompt = f"\nStart training all {len(planned_folds)} folds? [Y/n]: "
        auto_start_msg = f"\nStarting training for all {len(planned_folds)} folds..."

    # Confirm start
    if not args.no_confirm:
        response = input(confirm_prompt).strip().lower()
        if response == 'n':
            print("Training cancelled")
            return
    else:
        print(auto_start_msg)

    # Set up main logger
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(config['log_dir'] / 'main.log'),
            logging.StreamHandler()
        ]
    )
    logger = logging.getLogger('main')

    # Train all folds
    overall_start = time.time()
    results = {}

    folds = planned_folds
    for fold_num in folds:
        print(f"\n{'='*60}")
        print(f"Fold {fold_num}/5")
        print(f"{'='*60}")

        try:
            trainer = FoldTrainer(fold_num, config)
            best_f1, train_time = trainer.train()

            results[f'fold_{fold_num}'] = {
                'success': True,
                'best_f1': best_f1,
                'time_minutes': train_time / 60
            }

        except Exception as e:
            logger.error(f"Fold {fold_num} training failed: {str(e)}")
            import traceback
            traceback.print_exc()

            results[f'fold_{fold_num}'] = {
                'success': False,
                'error': str(e),
                'time_minutes': 0
            }

    # Summary
    overall_time = time.time() - overall_start
    success_count = sum(1 for r in results.values() if r['success'])

    print(f"\n{'='*60}")
    print("Training complete!")
    print(f"{'='*60}")
    print(f"Successful: {success_count}/5 folds")
    print(f"Total time: {overall_time/3600:.1f} hours")

    if success_count > 0:
        avg_f1 = np.mean([r['best_f1'] for r in results.values() if r['success']])
        print(f"Average F1 score: {avg_f1:.3f}")

    # Save final report
    final_report = {
        'timestamp': timestamp,
        'config': config_to_save,  # Use the serializable version
        'results': results,
        'total_time_hours': overall_time / 3600,
        'success_count': success_count
    }

    report_file = config['log_dir'] / 'final_report.json'
    with open(report_file, 'w') as f:
        json.dump(final_report, f, indent=2)

    print(f"\nDetailed report saved to: {report_file}")
    print(f"Log directory: {config['log_dir']}")
    print(f"Model directory: {config['save_root']}")


if __name__ == "__main__":
    main()
