"""DataProcessor: normalisation and fingerprint measurement.

Computes a correlation-decay fingerprint from the raw tensors and passes it to
the NAS via metadata: fp_corr1 (local pixel correlation) and fp_encoded
(corr1 < ENCODED_CORR1, flagging encoded/symbolic data where the image prior
does not hold). The train dataset is built with the metadata-derived
augmentation policy on a mutable AugPolicy that the NAS toggles in place.
"""

from __future__ import annotations

import numpy as np
import torch
from torch.utils.data import DataLoader, SequentialSampler, TensorDataset

from helpers import (
    AugmentedTensorDataset,
    aug_policy_for,
    per_channel_stats,
)

ENCODED_CORR1 = 0.2
FP_SAMPLES = 256


class DataProcessor:
    def __init__(self, train_x, train_y, valid_x, valid_y, test_x, metadata, clock):
        self.train_x = train_x
        self.train_y = train_y
        self.valid_x = valid_x
        self.valid_y = valid_y
        self.test_x = test_x
        self.metadata = metadata
        self.clock = clock
        self.batch_size = 128

    @staticmethod
    def _to_nchw(x: np.ndarray) -> np.ndarray:
        if x.ndim == 3:
            x = x[:, None, :, :]
        return x

    def _normalise(self, x: np.ndarray, mean: np.ndarray, std: np.ndarray) -> torch.Tensor:
        arr = self._to_nchw(x).astype(np.float32)
        arr = (arr - mean[None, :, None, None]) / std[None, :, None, None]
        return torch.from_numpy(arr)

    @staticmethod
    def _fingerprint(x_nchw: np.ndarray) -> float:
        """corr1: mean Pearson correlation between horizontally and
        vertically adjacent pixels, standardized, over a fixed sample.
        Near zero = neighboring pixels are independent symbols (encoded
        data); high = image-like. Pure numpy, deterministic, seconds."""
        rng = np.random.default_rng(42)
        n = x_nchw.shape[0]
        idx = np.sort(rng.choice(n, size=min(FP_SAMPLES, n), replace=False))
        x = np.asarray(x_nchw[idx], dtype=np.float64)
        mu = x.mean(axis=(0, 2, 3), keepdims=True)
        sd = x.std(axis=(0, 2, 3), keepdims=True) + 1e-8
        x = (x - mu) / sd
        h_corr = float((x[..., :, :-1] * x[..., :, 1:]).mean())
        v_corr = float((x[..., :-1, :] * x[..., 1:, :]).mean())
        return (h_corr + v_corr) / 2.0

    def process(self):
        train_x_nchw = self._to_nchw(self.train_x)
        mean, std = per_channel_stats(train_x_nchw)

        # ---- fingerprint (flags for the NAS routing) ----------------------
        try:
            corr1 = self._fingerprint(train_x_nchw)
        except Exception:
            corr1 = 1.0   # fail safe: image-like default path
        self.metadata["fp_corr1"] = round(corr1, 4)
        self.metadata["fp_encoded"] = bool(corr1 < ENCODED_CORR1)
        print(f"  v10 fingerprint: corr1={corr1:+.3f} -> "
              f"{'ENCODED' if corr1 < ENCODED_CORR1 else 'image-like'}")

        train_x = self._normalise(self.train_x, mean, std)
        valid_x = self._normalise(self.valid_x, mean, std)
        test_x = self._normalise(self.test_x, mean, std)

        train_y = torch.as_tensor(np.asarray(self.train_y), dtype=torch.long)
        valid_y = torch.as_tensor(np.asarray(self.valid_y), dtype=torch.long)

        policy = aug_policy_for(self.metadata)
        print(f"  base aug policy: {policy.describe()} "
              f"(final decision made by the NAS aug probe, family-matched)")
        self.metadata["mixup_alpha"] = 0.0   # NAS may raise it on aug-off

        # Mutable policy: the NAS aug probe toggles the fields in place.
        if policy.any_input_aug():
            train_dataset = AugmentedTensorDataset(train_x, train_y, policy)
        else:
            train_dataset = TensorDataset(train_x, train_y)

        train_loader = DataLoader(train_dataset, batch_size=self.batch_size,
                                  shuffle=True, drop_last=False, num_workers=0)
        valid_loader = DataLoader(TensorDataset(valid_x, valid_y),
                                  batch_size=self.batch_size, shuffle=False,
                                  drop_last=False, num_workers=0)
        test_dataset = TensorDataset(test_x)
        test_loader = DataLoader(test_dataset, batch_size=self.batch_size,
                                 sampler=SequentialSampler(test_dataset),
                                 drop_last=False, num_workers=0)
        return train_loader, valid_loader, test_loader
