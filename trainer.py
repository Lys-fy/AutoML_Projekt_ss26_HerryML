"""Budget-adaptive training controller.

Wraps an SGD warmup+cosine recipe (EMA, label smoothing, NaN recovery,
channels_last) in a throughput-keyed plan: everything is derived from a short
calibration of measured step time, re-projected between stages, never from
wall-clock hours or GPU names.

Pipeline inside train():
  1. calibrate (~few batches): measured step time -> projected epoch time.
  2. plan: proj_epochs = usable_clock / epoch_time; choose K seeds and a
     per-seed schedule target = min(CAP_EPOCHS, proj_epochs / K).
  3. train K seeds from the same init, re-planning after each seed from its
     measured duration (a slow seed shrinks or drops the ones after it); a
     seed that reaches ~perfect val twice in a row stops early and donates
     its slice.
  4. val-gated selection over {each seed, ensemble} x {no-TTA, flip-TTA},
     each extra val pass clock-guarded.
predict() applies the chosen config, degrading to single-best if the clock
would not fit the ensemble/TTA passes.
"""

from __future__ import annotations

import copy
import csv
import os
import time
from collections import deque
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn

from helpers import ModelEMA, lr_for_epoch


SAFETY_MARGIN_SECONDS = 60.0

# ---- seed roster and schedule ----------------------------------------------
FULL_EPOCHS = 600       # reference schedule length for one full training
CAP_EPOCHS = 2400       # bound on any single seed's schedule
MAX_SEEDS = 3           # planned seeds before reactive top-ups
HARD_CAP_SEEDS = 8      # bound on the roster after reactive top-ups
SPLIT_FACTOR = 1.5      # multi-seed split only when proj >= SPLIT_FACTOR x FULL_EPOCHS
MIN_SEED_EPOCHS = 300   # half-split to K=2 once each seed still gets this many epochs

# ---- early stops -----------------------------------------------------------
WARMUP_EPOCHS = 20
PATIENCE = 999
PERFECT_ACC = 0.9995    # a seed stops after val >= this on two epochs in a row
# saturation stop: the patience anchor advances only on a val gain that clears
# the run's own measured noise (std of a recent window); once val plateaus the
# anchor freezes and the seed stops after a schedule-relative plateau window.
SATURATE = True
SATURATE_FRAC = 0.25
SATURATE_WINDOW = 15

# ---- selection -------------------------------------------------------------
# prefer an ensemble config when its val is within SEL_BAND (pp) of the argmax single
SEL_BAND = 0.25

# ---- optimisation ----------------------------------------------------------
LR_SCALE = 1.0
LR_WARMUP_EPOCHS = 5
LABEL_SMOOTHING = 0.1
EMA_DECAY = 0.999
SEED = 42
USE_TRIVIAL_AUG = True   # one random augmentation op at a random magnitude per batch


def _trivial_augment(xb: torch.Tensor) -> torch.Tensor:
    """TrivialAugment-style: one op at a random magnitude per batch. Op set is
    restricted to transforms valid for arbitrary C x H x W tensors."""
    ops = ("identity", "flip_h", "translate", "brightness", "contrast",
           "noise", "cutout")
    op = ops[np.random.randint(len(ops))]
    if op == "flip_h":
        return torch.flip(xb, dims=[3])
    if op == "translate":
        return torch.roll(xb, shifts=int(np.random.randint(-3, 4)),
                          dims=int(np.random.choice([2, 3])))
    if op == "brightness":
        return xb + float(np.random.uniform(-0.3, 0.3))
    if op == "contrast":
        m = xb.mean(dim=(2, 3), keepdim=True)
        return (xb - m) * float(np.random.uniform(0.6, 1.4)) + m
    if op == "noise":
        return xb + torch.randn_like(xb) * float(np.random.uniform(0.0, 0.1))
    if op == "cutout":
        _, _, H, W = xb.shape
        ch, cw = H // 4, W // 4
        if ch > 0 and cw > 0:
            y0 = int(np.random.randint(0, H - ch + 1))
            x0 = int(np.random.randint(0, W - cw + 1))
            xb = xb.clone(); xb[:, :, y0:y0 + ch, x0:x0 + cw] = 0.0
    return xb


def _open_csv_logger(metadata) -> Tuple[Optional[object], Optional[csv.writer]]:
    log_dir = os.environ.get("BASELINE_LOG_DIR")
    if not log_dir:
        return None, None
    os.makedirs(log_dir, exist_ok=True)
    codename = metadata.get("codename", "unknown")
    path = os.path.join(log_dir, f"{codename}.csv")
    f = open(path, "w", newline="")
    writer = csv.writer(f)
    writer.writerow([
        "epoch", "lr", "train_loss", "val_acc", "val_acc_ema",
        "best_val_acc", "epoch_time_s", "clock_remaining_s", "seed",
    ])
    return f, writer


class Trainer:
    def __init__(self, model, device, train_dataloader, valid_dataloader, metadata, clock):
        model = model.to(device, memory_format=torch.channels_last)
        self.model = model
        self.device = device
        self.train_loader = train_dataloader
        self.valid_loader = valid_dataloader
        self.metadata = metadata
        self.clock = clock
        self.mixup_alpha = float(metadata.get("mixup_alpha", 0.0))
        self.use_ema = True
        # Shared init reused by every seed.
        self.init_state = copy.deepcopy(model.state_dict())
        # NAS side-channel: optional runner-up-family model for one
        # heterogeneous ensemble seed.
        meta_sc = getattr(model, "_v12_meta", {}) or {}
        self.alt_model: Optional[nn.Module] = meta_sc.get("alt_model")
        self.alt_family: Optional[str] = meta_sc.get("alt_family")
        if self.alt_model is not None:
            self.alt_init = copy.deepcopy(self.alt_model.state_dict())
        # capacity arm: wider twin of the final arch, used as a val-gated seed
        self.cap_model: Optional[nn.Module] = meta_sc.get("cap_model")
        if self.cap_model is not None:
            self.cap_init = copy.deepcopy(self.cap_model.state_dict())
        self._cap_used = False
        # controller outputs. seed_models[i] is the MODULE that seed i's state
        # belongs to (heterogeneous seeds have a different architecture).
        self.seed_states: List[Dict[str, torch.Tensor]] = []
        self.seed_models: List[nn.Module] = []
        self.seed_vals: List[float] = []
        self.chosen: Tuple[str, object, bool] = ("single", 0, False)
        self.val_scores: Dict[str, float] = {}
        self._pass_s = 5.0            # measured duration of one val pass
        self.best_val_acc = -1.0      # API compat (max over seeds)

    # ------------------------------------------------------------- calibration

    def _calibrate_epoch_time(self) -> float:
        """Measure real step time on THIS hardware/data -> projected seconds per
        epoch (train pass + val pass estimate). Uses a handful of real steps on
        the actual model, then restores the init state so seed 0 is unaffected."""
        self.model.train()
        opt = torch.optim.SGD(self.model.parameters(), lr=1e-4)
        criterion = nn.CrossEntropyLoss()
        times, steps = [], 0
        for xb, yb in self.train_loader:
            if steps >= 4 or self.clock.check() < SAFETY_MARGIN_SECONDS:
                break
            xb = xb.to(self.device, non_blocking=True, memory_format=torch.channels_last)
            yb = yb.to(self.device, non_blocking=True)
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            t0 = time.perf_counter()
            opt.zero_grad(set_to_none=True)
            loss = criterion(self.model(xb), yb)
            loss.backward()
            opt.step()
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            times.append(time.perf_counter() - t0)
            steps += 1
        self.model.load_state_dict(self.init_state)   # undo calibration steps
        step_t = min(times[1:]) if len(times) > 1 else (times[0] if times else 0.1)
        n_train = max(1, len(self.train_loader))
        n_val = max(1, len(self.valid_loader))
        # val pass ~ forward-only ~ 1/3 of a train step; 1.1 fudge for logging etc.
        return 1.1 * (n_train * step_t + n_val * step_t / 3.0)

    def _plan(self, remaining_s: float, epoch_s: float) -> Tuple[int, int]:
        """Throughput-keyed rung selection: how many seeds, how long a schedule.
        Multi-full split when every seed gets >= one full training; below that,
        a half-split takes K=2 while each seed still gets >= MIN_SEED_EPOCHS;
        surplus beyond MAX_SEEDS full trainings goes into longer schedules, up
        to CAP_EPOCHS."""
        proj = max(1.0, remaining_s / max(epoch_s, 1e-3))
        k = 1
        if proj >= SPLIT_FACTOR * FULL_EPOCHS:
            k = int(min(MAX_SEEDS, proj // FULL_EPOCHS))
        if k <= 1 and proj >= 2 * MIN_SEED_EPOCHS:
            k = 2
        k = max(1, k)
        return k, int(min(CAP_EPOCHS, proj / k))

    # ---------------------------------------------------------- one seed train

    def _train_one_seed(self, model: nn.Module, init_state, seed_idx: int,
                        deadline: float, target_epochs: int,
                        csv_writer, csv_file) -> Tuple[Dict[str, torch.Tensor], float]:
        torch.manual_seed(SEED + seed_idx)
        np.random.seed(SEED + seed_idx)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(SEED + seed_idx)

        model = model.to(self.device, memory_format=torch.channels_last)
        model.load_state_dict(init_state)
        ema = ModelEMA(model, decay=EMA_DECAY) if self.use_ema else None
        criterion = nn.CrossEntropyLoss(label_smoothing=LABEL_SMOOTHING)
        if getattr(model, "_v12_recipe", "") == "adamw":
            # token-family models carry their own recipe: AdamW with a fixed
            # lr and weight decay.
            base_lr = 3e-4 * LR_SCALE
            optimizer = torch.optim.AdamW(model.parameters(), lr=base_lr,
                                          weight_decay=0.05)
        else:
            base_lr = 0.025 * LR_SCALE
            optimizer = torch.optim.SGD(model.parameters(), lr=base_lr,
                                        momentum=0.9, weight_decay=3e-4, nesterov=True)

        def remaining():
            return min(deadline - time.perf_counter(), self.clock.check())

        def set_lr(epoch, total):
            lr = lr_for_epoch(epoch, total, warmup_epochs=LR_WARMUP_EPOCHS, base_lr=base_lr)
            for g in optimizer.param_groups:
                g["lr"] = lr
            return lr

        best_val, best_state, best_epoch = -1.0, copy.deepcopy(model.state_dict()), 0
        total_epochs = target_epochs
        nan_strikes = 0
        perfect_streak = 0
        # saturation stop: the patience anchor (advances only on a
        # noise-clearing gain) is tracked separately from best_val (which
        # always keeps the true-best checkpoint); val_window feeds the
        # self-measured noise floor.
        anchor_val, anchor_epoch = -1.0, 0
        val_window: deque = deque(maxlen=SATURATE_WINDOW)
        epoch = 0
        while epoch < total_epochs:
            if remaining() < SAFETY_MARGIN_SECONDS:
                break
            lr = set_lr(epoch, total_epochs)
            t0 = time.perf_counter()
            train_loss = self._train_one_epoch(model, optimizer, criterion, ema, deadline)

            if not np.isfinite(train_loss):
                nan_strikes += 1
                print(f"  WARN seed {seed_idx} epoch {epoch}: NaN, restore best. "
                      f"strike {nan_strikes}/3.")
                if nan_strikes >= 3:
                    break
                model.load_state_dict(best_state)
                base_lr = base_lr / 5.0
                epoch = 0
                continue

            val = self._evaluate(model)
            val_ema = self._evaluate(ema.ema) if ema is not None else val
            cand = max(val, val_ema)
            if cand > best_val:
                best_val = cand
                src = ema.state_dict() if (ema is not None and val_ema > val) else model.state_dict()
                best_state = copy.deepcopy(src)
                best_epoch = epoch
            # advance the patience anchor only on a gain that clears the run's
            # own recent val noise (std of the window).
            val_window.append(cand)
            val_noise = float(np.std(val_window)) if len(val_window) >= 8 else 0.0
            if cand > anchor_val + val_noise:
                anchor_val, anchor_epoch = cand, epoch

            elapsed = time.perf_counter() - t0
            if epoch == 0:
                # re-anchor the schedule to what this seed's slice actually
                # affords, never above the planned target
                afford = max(LR_WARMUP_EPOCHS + 2,
                             int((remaining() - SAFETY_MARGIN_SECONDS) / max(elapsed, 1e-3)))
                total_epochs = min(target_epochs, afford)
                print(f"  seed {seed_idx}: schedule {total_epochs} epochs "
                      f"(epoch0 {elapsed:.1f}s, slice {remaining():.0f}s)")

            print(f"  s{seed_idx} e{epoch:3d} loss={train_loss:.4f} val={val:.4f} "
                  f"ema={val_ema:.4f} best={best_val:.4f} lr={lr:.4f} "
                  f"t={elapsed:.1f}s clock={self.clock.check():.0f}s", flush=True)
            if csv_writer is not None:
                csv_writer.writerow([epoch, f"{lr:.6f}", f"{train_loss:.6f}",
                                     f"{val:.6f}", f"{val_ema:.6f}", f"{best_val:.6f}",
                                     f"{elapsed:.3f}", f"{self.clock.check():.1f}", seed_idx])
                csv_file.flush()

            perfect_streak = perfect_streak + 1 if cand >= PERFECT_ACC else 0
            if perfect_streak >= 2:
                print(f"  seed {seed_idx}: ~perfect val twice, releasing slice")
                break
            if epoch >= WARMUP_EPOCHS and epoch - best_epoch >= PATIENCE:
                break
            # saturation stop: no noise-clearing gain within a schedule-
            # relative plateau window (with an absolute floor) -> stop this
            # seed and release its slice.
            if SATURATE:
                plateau_limit = max(2 * WARMUP_EPOCHS,
                                    int(SATURATE_FRAC * total_epochs))
                if epoch >= WARMUP_EPOCHS and epoch - anchor_epoch >= plateau_limit:
                    print(f"  seed {seed_idx}: saturated (no >noise gain in "
                          f"{plateau_limit} ep, anchor e{anchor_epoch} "
                          f"@{anchor_val:.4f}), releasing slice", flush=True)
                    break
            if remaining() < elapsed + SAFETY_MARGIN_SECONDS:
                break
            epoch += 1

        cpu_state = {k: v.detach().cpu().clone() for k, v in best_state.items()}
        return cpu_state, best_val

    def _train_one_epoch(self, model, optimizer, criterion, ema, deadline) -> float:
        model.train()
        total_loss, n_batches = 0.0, 0
        use_mixup = self.mixup_alpha > 0.0
        for xb, yb in self.train_loader:
            if (min(deadline - time.perf_counter(), self.clock.check())
                    < SAFETY_MARGIN_SECONDS):
                return total_loss / max(n_batches, 1)
            xb = xb.to(self.device, non_blocking=True, memory_format=torch.channels_last)
            yb = yb.to(self.device, non_blocking=True)
            if USE_TRIVIAL_AUG:
                xb = _trivial_augment(xb)
            if use_mixup:
                lam = float(np.random.beta(self.mixup_alpha, self.mixup_alpha))
                lam = max(lam, 1.0 - lam)
                perm = torch.randperm(xb.size(0), device=self.device)
                xb_in, yb_b = lam * xb + (1.0 - lam) * xb[perm], yb[perm]
            else:
                lam, yb_b, xb_in = 1.0, yb, xb

            def compute_loss():
                logits = model(xb_in)
                if use_mixup:
                    loss = lam * criterion(logits, yb) + (1.0 - lam) * criterion(logits, yb_b)
                else:
                    loss = criterion(logits, yb)
                return loss

            optimizer.zero_grad(set_to_none=True)
            loss = compute_loss()
            if not torch.isfinite(loss):
                return float("nan")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            if ema is not None:
                ema.update(model)
            total_loss += float(loss.item())
            n_batches += 1
        return total_loss / max(n_batches, 1)

    @torch.no_grad()
    def _evaluate(self, model) -> float:
        model.eval()
        correct = total = 0
        for xb, yb in self.valid_loader:
            xb = xb.to(self.device, non_blocking=True, memory_format=torch.channels_last)
            yb = yb.to(self.device, non_blocking=True)
            preds = model(xb).argmax(dim=1)
            correct += (preds == yb).sum().item()
            total += yb.numel()
        return correct / max(total, 1)

    # -------------------------------------------------------------- controller

    def train(self):
        torch.backends.cudnn.benchmark = True
        csv_file, csv_writer = _open_csv_logger(self.metadata)
        try:
            epoch_s = self._calibrate_epoch_time()
            usable = self.clock.check() - 2 * SAFETY_MARGIN_SECONDS
            k, target = self._plan(usable, epoch_s)
            print(f"  v12 plan: ~{epoch_s:.1f}s/epoch -> {usable/max(epoch_s,1e-3):.0f} "
                  f"projected epochs -> K={k} seeds x {target}-epoch schedule "
                  f"(TA={'on' if USE_TRIVIAL_AUG else 'off'})", flush=True)

            # seed roster: primary-family replicas; when the family gate was a
            # sub-noise coin flip, the LAST planned seed trains the runner-up
            # family instead (heterogeneous seed, val-gated like everything).
            roster = [(self.model, self.init_state)] * k
            if self.alt_model is not None and k >= 2:
                roster[k - 1] = (self.alt_model, self.alt_init)
                print(f"  v12.1 heterogeneous seed: slot {k-1} -> {self.alt_family}")
            if self.cap_model is not None and k >= 3:
                roster[k - 2] = (self.cap_model, self.cap_init)
                self._cap_used = True
                print(f"  v12.1b capacity seed: slot {k-2} -> wide twin")

            i = 0
            while i < len(roster):
                remaining = self.clock.check() - SAFETY_MARGIN_SECONDS
                seeds_left = len(roster) - i
                slice_s = remaining / seeds_left
                if i > 0:
                    # re-plan from the MEASURED seed duration: drop seeds that
                    # no longer fit a meaningful training
                    if remaining < 0.35 * self._last_seed_s:
                        print(f"  v12 re-plan: {remaining:.0f}s left "
                              f"< 0.35 x last seed -> stop at {i} seed(s)")
                        break
                    target = int(min(target, max(LR_WARMUP_EPOCHS + 2,
                                                 slice_s / max(self._last_epoch_s, 1e-3))))
                seed_model, seed_init = roster[i]
                t0 = time.perf_counter()
                deadline = t0 + slice_s
                state, val = self._train_one_seed(seed_model, seed_init, i,
                                                  deadline, target, csv_writer, csv_file)
                self._last_seed_s = time.perf_counter() - t0
                self._last_epoch_s = max(self._last_seed_s / max(target, 1), 1e-3)
                self.seed_states.append(state)
                self.seed_models.append(seed_model)
                self.seed_vals.append(val)
                print(f"  seed {i} done: best_val={val:.4f} "
                      f"({self._last_seed_s/60:.1f}min)", flush=True)
                i += 1
                # reactive top-up: spend measured surplus (epoch-cap or
                # perfect-stop releases) on another seed.
                if (i == len(roster) and len(roster) < HARD_CAP_SEEDS
                        and self.clock.check() - SAFETY_MARGIN_SECONDS
                        > 0.9 * self._last_seed_s + 3.0 * self._pass_s):
                    target = int(min(CAP_EPOCHS, max(
                        LR_WARMUP_EPOCHS + 2,
                        (self.clock.check() - SAFETY_MARGIN_SECONDS)
                        / max(self._last_epoch_s, 1e-3))))
                    if self.cap_model is not None and not self._cap_used:
                        # surplus funds the capacity arm first (wider twin)
                        roster.append((self.cap_model, self.cap_init))
                        self._cap_used = True
                        print(f"  v12.1b top-up: surplus funds the CAPACITY seed "
                              f"{i} ({target}-epoch schedule)")
                    else:
                        roster.append((self.model, self.init_state))
                        print(f"  v12.1 top-up: surplus affords seed {i} "
                              f"({target}-epoch schedule)")
        finally:
            if csv_file is not None:
                csv_file.close()

        if not self.seed_states:   # pathological tiny budget: keep init
            self.seed_states = [{k2: v.detach().cpu().clone()
                                 for k2, v in self.model.state_dict().items()}]
            self.seed_models = [self.model]
            self.seed_vals = [0.0]
        self.best_val_acc = max(self.seed_vals)
        self._select()
        # load the chosen (or best single) state for API compat
        kind, idx, _ = self.chosen
        j = idx if kind == "single" else int(np.argmax(self.seed_vals))
        self.seed_models[j].load_state_dict(self.seed_states[j])
        return self.seed_models[j]

    # --------------------------------------------------------------- selection

    @torch.no_grad()
    def _probs(self, i: int, loader, tta: bool) -> torch.Tensor:
        model = self.seed_models[i]
        model.load_state_dict(self.seed_states[i])
        model = model.to(self.device, memory_format=torch.channels_last)
        model.eval()
        outs = []
        for batch in loader:
            xb = batch[0] if isinstance(batch, (list, tuple)) else batch
            xb = xb.to(self.device, memory_format=torch.channels_last)
            p = model(xb).softmax(1)
            if tta:
                p = 0.5 * (p + model(torch.flip(xb, dims=[3])).softmax(1))
            outs.append(p.cpu())
        return torch.cat(outs)

    def _select(self):
        """Val-gated final selection: enumerate {each seed, logit-ensemble} x
        {no-TTA, flip-TTA}, score on validation, keep the winner. Every extra
        val pass is clock-guarded; on any shortfall we fall back to the
        best-val single seed (which costs nothing extra)."""
        best_i = int(np.argmax(self.seed_vals))
        self.chosen = ("single", best_i, False)
        yv = torch.cat([yb for _, yb in self.valid_loader])
        n_seeds = len(self.seed_states)

        def acc(p):
            n = min(len(p), len(yv))
            return round(100.0 * float((p[:n].argmax(1) == yv[:n]).float().mean()), 3)

        val_p: Dict[Tuple[int, bool], torch.Tensor] = {}
        t0 = time.perf_counter()
        val_p[(best_i, False)] = self._probs(best_i, self.valid_loader, False)
        self._pass_s = max(time.perf_counter() - t0, 0.2)
        # each further pass only if the clock affords it (+ margin)
        for tta in (False, True):
            for i in range(n_seeds):
                if (i, tta) in val_p:
                    continue
                if self.clock.check() < SAFETY_MARGIN_SECONDS + 2.5 * self._pass_s:
                    print("  v12 select: clock short, stopping extra val passes")
                    break
                val_p[(i, tta)] = self._probs(i, self.valid_loader, tta)

        configs: Dict[str, Tuple[str, object, bool]] = {}
        scores: Dict[str, float] = {}
        # seeds ranked by their own single-config val (for top-m subsets)
        rank = sorted(range(n_seeds), key=lambda i: -self.seed_vals[i])
        for tta in (False, True):
            suf = "_tta" if tta else ""
            have = [i for i in range(n_seeds) if (i, tta) in val_p]
            for i in have:
                name = f"single{i}{suf}"
                configs[name] = ("single", i, tta)
                scores[name] = acc(val_p[(i, tta)])
            if len(have) == n_seeds and n_seeds > 1:
                # top-m: score the full ensemble and every top-m subset by
                # seed val, all from cached probs.
                for m in range(2, n_seeds + 1):
                    members = tuple(sorted(rank[:m]))
                    name = (f"ensemble{suf}" if m == n_seeds else f"top{m}{suf}")
                    configs[name] = ("ensemble", members, tta)
                    scores[name] = acc(torch.stack(
                        [val_p[(i, tta)] for i in members]).mean(0))
        chosen_name = max(configs, key=lambda nm: scores[nm])
        # noise band: if the argmax is a single but an ensemble config is
        # within SEL_BAND of it, prefer the ensemble.
        if configs[chosen_name][0] == "single":
            ens = [nm for nm in configs if configs[nm][0] == "ensemble"
                   and scores[chosen_name] - scores[nm] <= SEL_BAND]
            if ens:
                chosen_name = max(ens, key=lambda nm: scores[nm])
                print(f"  v12.1 band: single within {SEL_BAND}pp of an ensemble "
                      f"-> prefer {chosen_name}")
        self.chosen = configs[chosen_name]
        self.val_scores = scores
        print(f"  v12 select: {scores} -> {chosen_name}", flush=True)

    # ---------------------------------------------------------------- predict

    @torch.no_grad()
    def predict(self, test_loader) -> List[int]:
        kind, idx, tta = self.chosen
        members = (list(idx) if kind == "ensemble" and idx is not None
                   else list(range(len(self.seed_states))))
        # predictions must ALWAYS complete: degrade if the clock cannot fit the
        # chosen config's passes (test pass ~ val pass x |test|/|val| ~ pass_s)
        passes = (len(members) if kind == "ensemble" else 1) * (2 if tta else 1)
        est = passes * self._pass_s * (len(test_loader) / max(len(self.valid_loader), 1))
        if self.clock.check() < est + 30.0:
            if kind == "ensemble" or tta:
                print("  v12 predict: clock short -> degrade to best single, no TTA")
            kind, idx, tta = "single", int(np.argmax(self.seed_vals)), False
        if kind == "ensemble":
            p = torch.stack([self._probs(i, test_loader, tta)
                             for i in members]).mean(0)
        else:
            p = self._probs(idx, test_loader, tta)
        return p.argmax(1).tolist()
