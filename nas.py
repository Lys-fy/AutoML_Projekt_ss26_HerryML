"""NAS: the measure-gate-train pipeline.

Stages:

  1. FINGERPRINT (computed in DataProcessor, flags in metadata):
     corr1 < 0.2 -> encoded data.
  2. BACKBONE:
     image-like -> TCN-2d, no bake-off.
     encoded    -> 3-arm gate (conv / tcn1d / tcn2d short-train at matched
     budget); pick the leader only if it beats the conv arm by more than
     the measured noise, else abstain to conv.
  3. AUG PROBE, after the backbone choice, with arms of the chosen family.
  4. DEPTH: default-incumbent. The family default (D=6) is the incumbent;
     candidates {4, 8} must beat it by more than noise at short-train and
     survive verification at a longer horizon.
  5. ATTENTION PROBE: encoded data only, only if budget remains.
  6. Trainer trains the winner to the full remaining budget (LR -> 0).

Every stage appends to a decision record:
  <BASELINE_LOG_DIR>/<codename>_decisions.json

Budget fractions of the clock remaining at search start; at low budgets
the pipeline collapses to fingerprint + default backbone + train-to-budget.
"""

from __future__ import annotations

import json
import math
import os
import random
import time
import traceback
from typing import Dict, List, Optional

import torch
import torch.nn as nn

from helpers import (
    OP_NAMES,
    TCN1D,
    TCN2D,
    TokenSeqNet,
    build_cnn,
    build_default_cnn,
)

# ---- search budget: fractions of the clock remaining at search start -------
GATE_FRAC = 0.18          # 3-arm family bake-off, encoded data only
AUG_FRAC = 0.08           # two-arm aug probe, family-matched
DEPTH_FRAC = 0.14         # 2 candidate shorts + 1 verification
RAND_FRAC = 0.12          # random arm, encoded only

# ---- probe sizing: per-leg floor/cap in seconds ----------------------------
# The floor applies only when the search budget exceeds PROBE_FLOOR_BUDGET_S.
PROBE_LO_S = 120.0
PROBE_HI_S = 420.0
PROBE_FLOOR_BUDGET_S = 4 * 3600

# ---- feature toggles -------------------------------------------------------
GATE_TIEBREAK = False     # re-measure the top-2 gate arms at 2x slice
TOKEN_ARM = True          # token family for embedding-shaped inputs
TOKEN_RATIO_MIN = 16.0    # min long/short side ratio to try the token family
CAP_ARM = True            # wider twin of the final arch as one val-gated seed
CAP_ARM_WIDTH = 96

# ---- low-budget collapse thresholds (seconds remaining at search start) ----
COLLAPSE_FULL_S = 35 * 60   # below: no gate/depth/attn probes
COLLAPSE_AUG_S = 15 * 60    # below: not even the aug probe

# ---- architecture defaults -------------------------------------------------
DEFAULT_DEPTH = 6
DEPTH_CANDS = (4, 8)
DEFAULT_WIDTH = 48
STEM_EMBED_MIN_CH = 8       # min input channels for the channel-embed stem
ENCODED_CORR1 = 0.2         # corr1 below which data counts as encoded
RAND_K = 6                  # random arm: number of candidate architectures
RAND_WIDTHS = [16, 24, 32, 48]
WIDE_VOCAB = tuple(o for o in OP_NAMES
                   if o not in ("RowBiLSTM", "CrossAxisBiLSTM"))


class NAS:
    def __init__(self, train_loader, valid_loader, metadata, clock):
        self.train_loader = train_loader
        self.valid_loader = valid_loader
        self.metadata = metadata
        self.clock = clock
        self.record: Dict = {"stages": []}

    # ---- plumbing -----------------------------------------------------------

    def _input_shape(self):
        shape = self.metadata["input_shape"]
        if len(shape) == 4:
            _, c, h, w = shape
        else:
            c, h, w = shape
        return int(c), int(h), int(w)

    def _log(self, stage: str, **kw):
        entry = {"stage": stage, "t_remaining_s": round(self.clock.check(), 1)}
        entry.update(kw)
        self.record["stages"].append(entry)
        brief = {k: v for k, v in kw.items() if not isinstance(v, (list, dict))}
        print(f"  v10 {stage}: {brief}")

    def _dump_record(self):
        log_dir = os.environ.get("BASELINE_LOG_DIR")
        if not log_dir:
            return
        try:
            os.makedirs(log_dir, exist_ok=True)
            code = self.metadata.get("codename", "unknown")
            with open(os.path.join(log_dir, f"{code}_decisions.json"), "w") as f:
                json.dump(self.record, f, indent=2)
        except Exception:
            pass

    @torch.no_grad()
    def _eval(self, model, device) -> float:
        model.eval()
        correct = total = 0
        for xb, yb in self.valid_loader:
            xb = xb.to(device, non_blocking=True)
            yb = yb.to(device, non_blocking=True)
            correct += (model(xb).argmax(dim=1) == yb).sum().item()
            total += yb.numel()
        return correct / max(total, 1)

    @staticmethod
    def _slice(t0: float, frac: float, parts: float) -> float:
        """Per-leg probe budget: clock fraction with a floor (only when the
        total budget affords it) and a cap."""
        s = t0 * frac / parts
        if t0 > PROBE_FLOOR_BUDGET_S:
            s = max(s, PROBE_LO_S)
        return min(s, PROBE_HI_S)

    def _timed_train(self, model, device, budget_s: float,
                     max_epochs: int = 200) -> List[float]:
        """Train for up to budget_s wall-clock; eval after each epoch.
        Returns the per-epoch val series (for leader + noise estimates)."""
        deadline = time.perf_counter() + budget_s
        model = model.to(device)
        crit = nn.CrossEntropyLoss(label_smoothing=0.1)
        if getattr(model, "_v12_recipe", "") == "adamw":
            # token family trains with AdamW
            opt = torch.optim.AdamW(model.parameters(), lr=3e-4,
                                    weight_decay=0.05)
        else:
            opt = torch.optim.SGD(model.parameters(), lr=0.025, momentum=0.9,
                                  weight_decay=3e-4, nesterov=True)
        vals: List[float] = []
        epoch = 0
        while epoch < max_epochs and time.perf_counter() < deadline:
            model.train()
            for xb, yb in self.train_loader:
                if time.perf_counter() > deadline:
                    break
                xb = xb.to(device, non_blocking=True)
                yb = yb.to(device, non_blocking=True)
                opt.zero_grad(set_to_none=True)
                loss = crit(model(xb), yb)
                if not torch.isfinite(loss):
                    raise RuntimeError("NaN in v10 probe train")
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
                opt.step()
            vals.append(self._eval(model, device))
            epoch += 1
        return vals or [0.0]

    @staticmethod
    def _noise(vals: List[float]) -> float:
        tail = vals[-10:]
        if len(tail) < 3:
            return 0.02
        m = sum(tail) / len(tail)
        return math.sqrt(sum((v - m) ** 2 for v in tail) / len(tail))

    # ---- model builders -----------------------------------------------------

    def _build(self, family: str, in_ch, nc, h, w, depth: int = DEFAULT_DEPTH,
               width: int = 48):
        stem = "channel_embed" if in_ch >= STEM_EMBED_MIN_CH else "conv3"
        if family == "conv":
            return build_default_cnn(in_ch, nc, h, w, base_width=width,
                                     n_blocks=depth, drop_path_rate=0.1)
        if family == "tcn1d":
            return TCN1D(in_ch, nc, seq_len=h * w, base_width=width, n_blocks=8)
        if family == "tcn2d":
            return TCN2D(in_ch, nc, h, w, base_width=width, n_blocks=depth,
                         stem=stem)
        if family == "token":
            # width doubles as the capacity knob: wider builds get the larger
            # token configuration.
            big = width > 48
            return TokenSeqNet(in_ch, nc, h, w, dim=384 if big else 256,
                               depth=6 if big else 4)
        raise ValueError(family)

    def _build_small(self, family: str, in_ch, nc, h, w):
        """Aug-probe arm: a small net of the chosen family."""
        if family == "conv":
            return build_default_cnn(in_ch, nc, h, w, base_width=16,
                                     n_blocks=3, drop_path_rate=0.0)
        if family == "tcn1d":
            return TCN1D(in_ch, nc, seq_len=h * w, base_width=16, n_blocks=4)
        stem = "channel_embed" if in_ch >= STEM_EMBED_MIN_CH else "conv3"
        return TCN2D(in_ch, nc, h, w, base_width=16, n_blocks=3, stem=stem)

    # ---- the pipeline -------------------------------------------------------

    def search(self):
        try:
            return self._search_inner()
        except Exception as ex:
            traceback.print_exc()
            self._log("FALLBACK", error=f"{type(ex).__name__}: {ex}")
            self._dump_record()
            in_ch, h, w = self._input_shape()
            nc = int(self.metadata["num_classes"])
            return self._build("tcn2d", in_ch, nc, h, w)

    def _search_inner(self):
        in_ch, h, w = self._input_shape()
        nc = int(self.metadata["num_classes"])
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        t0 = self.clock.check()
        corr1 = float(self.metadata.get("fp_corr1", 1.0))
        encoded = bool(self.metadata.get("fp_encoded", corr1 < ENCODED_CORR1))
        collapse = t0 < COLLAPSE_FULL_S
        self.record.update(dict(codename=self.metadata.get("codename"),
                                in_shape=[in_ch, h, w], classes=nc,
                                t_at_search_s=round(t0, 1)))
        self._log("fingerprint", corr1=corr1, encoded=encoded,
                  collapse=collapse,
                  stem="channel_embed" if in_ch >= STEM_EMBED_MIN_CH else "conv3")

        # ---- stage 2: backbone ------------------------------------------------
        # defaults: image-like -> tcn2d; encoded under collapse -> conv
        family = "conv" if (encoded and collapse) else "tcn2d"
        gate_best: Dict[str, float] = {}
        noise = 0.02
        alt_family = None          # runner-up family for the hedge ensemble seed
        gate_margin = None
        if encoded and not collapse:
            arm_budget = self._slice(t0, GATE_FRAC, 3.0)
            noises = []
            arm_models: Dict[str, nn.Module] = {}
            for fam in ("conv", "tcn1d", "tcn2d"):
                model = self._build(fam, in_ch, nc, h, w)
                try:
                    vals = self._timed_train(model, device, arm_budget)
                except Exception as ex:
                    vals = [0.0]
                    print(f"  v10 gate arm {fam} FAILED ({type(ex).__name__})")
                gate_best[fam] = max(vals)
                noises.append(self._noise(vals))
                arm_models[fam] = model  # kept alive for the tie-break
            noise = sum(noises) / len(noises)
            leader = max(gate_best, key=gate_best.get)
            # abstain-to-default: deviate from conv only when clearly better
            family = leader if gate_best[leader] - gate_best["conv"] > noise \
                else "conv"
            # When the family race is inside the noise band, remember the best
            # non-picked family; the trainer can spend one val-gated ensemble
            # seed on it.
            others = {f: v for f, v in gate_best.items() if f != family}
            if others:
                alt_family = max(others, key=others.get)
                gate_margin = abs(gate_best[family] - others[alt_family])
                if gate_margin >= noise:
                    alt_family = None
            self._log("backbone_gate", arms={k: round(v, 4) for k, v in
                                             gate_best.items()},
                      noise=round(noise, 4), leader=leader, picked=family,
                      alt_family=alt_family,
                      gate_margin=round(gate_margin, 4) if alt_family else None)
            # ---- gate tie-break: continue the top-2 arms, re-measure --------
            if (GATE_TIEBREAK and alt_family is not None
                    and self.clock.check() > 8.0 * arm_budget):
                pair = (family, alt_family)
                ext_noises = []
                for fam in pair:
                    try:
                        vals2 = self._timed_train(arm_models[fam], device,
                                                  2.0 * arm_budget,
                                                  max_epochs=400)
                    except Exception as ex:
                        vals2 = []
                        print(f"  gate tie-break arm {fam} FAILED "
                              f"({type(ex).__name__})")
                    if vals2:
                        gate_best[fam] = max(gate_best[fam], max(vals2))
                        ext_noises.append(self._noise(vals2))
                if ext_noises:
                    noise = sum(ext_noises) / len(ext_noises)
                cand = max(pair, key=lambda f: gate_best[f])
                other = pair[0] if cand == pair[1] else pair[1]
                gate_margin = gate_best[cand] - gate_best[other]
                if gate_margin >= noise:
                    # decisive after extension: adopt, drop the hedge seed
                    family, alt_family = cand, None
                # else: still within noise -> keep the current pick and the
                # alt-family hedge.
                self._log("gate_tiebreak",
                          arms={f: round(gate_best[f], 4) for f in pair},
                          noise=round(noise, 4), picked=family,
                          alt_family=alt_family,
                          gate_margin=round(gate_margin, 4))
            arm_models.clear()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        else:
            self._log("backbone_skip", picked=family,
                      reason="image-like" if not encoded else "low budget")

        # ---- token-family gate (embedding-shaped inputs only) -----------------
        # Shape-triggered 2-arm bake-off vs the currently picked family: adopt
        # token only when it wins beyond noise; a sub-noise race keeps the
        # incumbent and spends the hedge seed on token instead.
        token_like = (TOKEN_ARM and not collapse and in_ch == 1
                      and max(h, w) >= TOKEN_RATIO_MIN * min(h, w)
                      and max(h, w) >= 64)
        if token_like:
            arm_budget = self._slice(t0, GATE_FRAC, 2.0)
            try:
                tok_vals = self._timed_train(
                    self._build("token", in_ch, nc, h, w), device, arm_budget,
                    max_epochs=400)
                base_vals = self._timed_train(
                    self._build(family, in_ch, nc, h, w), device, arm_budget,
                    max_epochs=400)
                tok_best, base_best = max(tok_vals), max(base_vals)
                tnoise = (self._noise(tok_vals) + self._noise(base_vals)) / 2.0
                if tok_best - base_best > tnoise:
                    family, alt_family = "token", family
                    gate_margin, noise = tok_best - base_best, tnoise
                elif base_best - tok_best <= tnoise:
                    alt_family = "token"
                    gate_margin, noise = abs(tok_best - base_best), tnoise
                self._log("token_gate", token=round(tok_best, 4),
                          incumbent=round(base_best, 4),
                          noise=round(tnoise, 4), picked=family,
                          alt_family=alt_family)
            except Exception as ex:
                self._log("token_gate", probe_error=type(ex).__name__)
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        # ---- stage 3: aug probe, family-matched arms --------------------------
        ds = self.train_loader.dataset
        policy = getattr(ds, "policy", None)
        if policy is not None and policy.any_input_aug() and family != "token" \
                and self.clock.check() > COLLAPSE_AUG_S:
            saved = (policy.flip, policy.crop_pad, policy.cutout_size)
            arm_budget = self._slice(t0, AUG_FRAC, 2.0)
            results = {}
            for arm in ("on", "off"):
                if arm == "off":
                    policy.flip, policy.crop_pad, policy.cutout_size = \
                        False, 0, 0
                else:
                    (policy.flip, policy.crop_pad,
                     policy.cutout_size) = saved
                model = self._build_small(family, in_ch, nc, h, w)
                try:
                    vals = self._timed_train(model, device, arm_budget)
                    results[arm] = max(vals)
                except Exception:
                    results[arm] = 0.0
                del model
            # Conservative rule: deviate from the metadata default (ON) only
            # on a clear off-margin.
            off_margin = float(os.environ.get("AUG_OFF_MARGIN", "0.02"))
            use_aug = results["off"] - results["on"] < off_margin
            if use_aug:
                (policy.flip, policy.crop_pad, policy.cutout_size) = saved
                self.metadata["mixup_alpha"] = 0.0
            else:
                policy.flip, policy.crop_pad, policy.cutout_size = False, 0, 0
                self.metadata["mixup_alpha"] = float(
                    os.environ.get("P1_MIXUP_ALPHA", "0.2"))
            self._log("aug_probe", family=family,
                      on=round(results["on"], 4), off=round(results["off"], 4),
                      decision="on" if use_aug else "off+mixup")
        else:
            self._log("aug_probe", skipped=True,
                      reason="no aug applicable" if policy is None or
                      not policy.any_input_aug() else "low budget")

        # ---- stage 4: depth, default-incumbent --------------------------------
        depth = DEFAULT_DEPTH
        incumbent_short: Optional[float] = gate_best.get(family)
        if not collapse and family not in ("tcn1d", "token"):
            cand_s = self._slice(t0, DEPTH_FRAC, len(DEPTH_CANDS) + 1.5)  # shorts + verify
            if incumbent_short is None:
                model = self._build(family, in_ch, nc, h, w, depth=DEFAULT_DEPTH)
                vals = self._timed_train(model, device, cand_s)
                incumbent_short = max(vals)
                noise = self._noise(vals)
                del model
            best_cand, best_val = None, incumbent_short
            cand_results = {}
            for d in DEPTH_CANDS:
                model = self._build(family, in_ch, nc, h, w, depth=d)
                try:
                    vals = self._timed_train(model, device, cand_s)
                    v = max(vals)
                except Exception:
                    v = 0.0
                cand_results[d] = round(v, 4)
                if v > best_val + noise:
                    best_cand, best_val = d, v
                del model
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            verified = None
            if best_cand is not None:
                # verification: re-train the challenger fresh, longer horizon
                model = self._build(family, in_ch, nc, h, w, depth=best_cand)
                try:
                    vvals = self._timed_train(model, device, cand_s * 1.5)
                    verified = round(max(vvals), 4)
                except Exception:
                    verified = 0.0
                del model
                if verified > incumbent_short + noise:
                    depth = best_cand
            self._log("depth_search", incumbent=round(incumbent_short or 0, 4),
                      candidates=cand_results, noise=round(noise, 4),
                      challenger=best_cand, verified=verified, picked=depth)
        else:
            self._log("depth_search", skipped=True, picked=depth)

        # ---- stage 4b: width -------------------------------------------------
        width = DEFAULT_WIDTH

        macro_pick = None

        # ---- stage 5: verified attention probe, encoded/board data only -------
        # Build the family default as a uniform build_cnn stack, and a
        # challenger with its first ATTN_STAGES blocks replaced by
        # SpatialSelfAttn2D; short-train both, then verify the challenger at a
        # longer horizon, and adopt only if it beats the default beyond noise.
        # The attention op can crash on odd grids -> the try/except scores it
        # 0.0 and the family default is kept.
        attn_pick = None
        astem = "channel_embed" if in_ch >= STEM_EMBED_MIN_CH else "conv3"
        ATTN_STAGES = int(os.environ.get("V11_ATTN_STAGES", "2"))
        if encoded and not collapse and family != "token" \
                and self.clock.check() > 0.45 * t0:
            base_op = "DilatedConv3" if family == "tcn2d" else "Conv3"
            cand_s = self._slice(t0, RAND_FRAC, 3.0)   # default-short + attn-short + verify
            # incumbent = the pipeline's chosen family default, trained at the
            # same horizon as the challenger.
            try:
                inc_vals = self._timed_train(
                    self._build(family, in_ch, nc, h, w, depth=depth, width=width),
                    device, cand_s)
                inc_short, anoise = max(inc_vals), self._noise(inc_vals)
            except Exception:
                inc_short, anoise = (incumbent_short or 0.0), noise
            k = max(1, min(ATTN_STAGES, depth))
            # Retry once; on repeated failure substitute the shape-robust
            # LinearAttn2D. Every failure is logged in the decision record.
            probe_err = None
            attn_ops = None
            attn_short = 0.0
            attn_op_used = None
            for op_name in ("SpatialSelfAttn2D", "SpatialSelfAttn2D", "LinearAttn2D"):
                cand_ops = [op_name] * k + [base_op] * (depth - k)
                try:
                    attn_short = max(self._timed_train(
                        build_cnn(in_ch, nc, h, w, ops=cand_ops, base_width=width,
                                  drop_path_rate=0.0, stem=astem), device, cand_s))
                    attn_ops, attn_op_used, probe_err = cand_ops, op_name, None
                    break
                except Exception as ex:
                    probe_err = type(ex).__name__
                    print(f"  v11.1 attn arm FAILED ({probe_err}), op={op_name}")
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            verified = None
            if attn_ops is not None and attn_short > inc_short + anoise:
                for _attempt in range(2):    # verification also gets one retry
                    try:
                        verified = round(max(self._timed_train(
                            build_cnn(in_ch, nc, h, w, ops=attn_ops, base_width=width,
                                      drop_path_rate=0.0, stem=astem),
                            device, cand_s * 1.5)), 4)
                        break
                    except Exception as ex:
                        probe_err = type(ex).__name__
                        verified = None
                        print(f"  v11.1 attn verify FAILED ({probe_err}), retrying")
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                if verified is not None and verified > inc_short + anoise:
                    attn_pick = (attn_ops, width, astem)
            self._log("attn_probe", base_op=base_op, attn_stages=k,
                      attn_op=attn_op_used, probe_error=probe_err,
                      incumbent=round(inc_short, 4), attn_short=round(attn_short, 4),
                      verified=verified, noise=round(anoise, 4),
                      adopted=bool(attn_pick))
        else:
            self._log("attn_probe", skipped=True)

        # ---- final build -------------------------------------------------------
        if attn_pick is not None:
            ops, W, astem = attn_pick
            final = build_cnn(in_ch, nc, h, w, ops=ops, base_width=W,
                              drop_path_rate=0.1, stem=astem)
            self._log("final", arch=f"attn[{ops.count('SpatialSelfAttn2D')}xAttn+{ops[-1]}] W={W}")
        elif macro_pick is not None:
            mstem = "channel_embed" if in_ch >= STEM_EMBED_MIN_CH else "conv3"
            final = build_cnn(in_ch, nc, h, w, ops=macro_pick, base_width=width,
                              drop_path_rate=0.1, stem=mstem)
            self._log("final", arch=f"macro[{'/'.join(macro_pick)}] W={width}")
        else:
            final = self._build(family, in_ch, nc, h, w, depth=depth, width=width)
            self._log("final", arch=f"{family} D={depth} W={width}")
        # Side-channel to the trainer: when the family gate was within noise,
        # hand over a built runner-up-family model so one val-gated ensemble
        # seed can be spent on it. Kept on CPU.
        alt_model = None
        if alt_family is not None and not collapse:
            try:
                alt_model = self._build(alt_family, in_ch, nc, h, w,
                                        depth=depth, width=width)
                self._log("alt_seed", family=alt_family,
                          gate_margin=gate_margin)
            except Exception as ex:
                self._log("alt_seed", family=alt_family,
                          probe_error=type(ex).__name__)
                alt_model = None
        # wider twin of the final architecture (whatever stage won)
        cap_model = None
        if CAP_ARM and not collapse:
            try:
                if attn_pick is not None:
                    ops_c, _, stem_c = attn_pick
                    cap_model = build_cnn(in_ch, nc, h, w, ops=ops_c,
                                          base_width=CAP_ARM_WIDTH,
                                          drop_path_rate=0.1, stem=stem_c)
                elif macro_pick is not None:
                    cap_model = build_cnn(in_ch, nc, h, w, ops=macro_pick,
                                          base_width=CAP_ARM_WIDTH,
                                          drop_path_rate=0.1,
                                          stem="channel_embed" if in_ch >= STEM_EMBED_MIN_CH else "conv3")
                else:
                    cap_model = self._build(family, in_ch, nc, h, w,
                                            depth=depth, width=CAP_ARM_WIDTH)
                self._log("cap_arm", width=CAP_ARM_WIDTH,
                          params_M=round(sum(pp.numel() for pp in
                                             cap_model.parameters()) / 1e6, 2))
            except Exception as ex:
                self._log("cap_arm", probe_error=type(ex).__name__)
                cap_model = None
        final._v12_meta = {"family": family, "alt_family": alt_family,
                           "gate_margin": gate_margin, "gate_noise": noise,
                           "alt_model": alt_model, "cap_model": cap_model}
        self._dump_record()
        return final
