# Copyright 2025 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""VAGEN fork patch: optional ``value_mask`` in ``workers/utils/losses.py::value_loss``.

Turn-level and bi-level advantage estimators write ``returns`` at one anchor token per
turn and leave the rest at a sentinel; ``value_mask`` marks the positions that actually
carry return supervision so the critic is not trained on sentinels. The field is
optional -- absent means "supervise every response token", i.e. upstream behaviour.

The assertions are properties rather than numeric values, so they stay valid if
``compute_value_loss`` changes:

  1. an all-ones ``value_mask`` is indistinguishable from no ``value_mask``
  2. perturbing ``returns`` where ``value_mask == 0`` does not move the loss
  3. perturbing ``returns`` where ``value_mask == 1`` does move the loss

``no_padding_2_padding`` is stubbed so the test exercises our masking logic rather than
the unpadding machinery, which has its own coverage.

Out of scope: the trainer-side plumbing that has to place ``value_mask`` into the
TrainingWorker TensorDict in the first place.
"""

from types import SimpleNamespace
from unittest.mock import patch

import torch
from tensordict import TensorDict

from verl.workers.utils import losses as losses_mod

BSZ, RESP = 2, 6
CONFIG = SimpleNamespace(cliprange_value=0.5, loss_agg_mode="token-mean")


def _data(returns: torch.Tensor, value_mask: torch.Tensor | None = None) -> TensorDict:
    fields = {
        "values": torch.zeros(BSZ, RESP),
        "returns": returns,
        "response_mask": torch.ones(BSZ, RESP, dtype=torch.long),
    }
    if value_mask is not None:
        fields["value_mask"] = value_mask
    return TensorDict(fields, batch_size=[BSZ])


def _loss(returns, value_mask=None) -> float:
    """Run value_loss with a fixed vpreds so only the masking logic varies."""
    vpreds = torch.full((BSZ, RESP), 0.3)
    model_output = {"values": vpreds}
    with patch.object(losses_mod, "no_padding_2_padding", lambda t, d: vpreds):
        loss, _ = losses_mod.value_loss(CONFIG, model_output, _data(returns, value_mask))
    return loss.item()


def _base_returns() -> torch.Tensor:
    torch.manual_seed(0)
    return torch.randn(BSZ, RESP)


def test_absent_value_mask_is_upstream_behaviour():
    """Smoke: the unmodified path still runs and produces a finite scalar."""
    loss = _loss(_base_returns())
    assert isinstance(loss, float)
    assert loss == loss and abs(loss) != float("inf")  # not NaN, not inf


def test_all_ones_mask_equals_no_mask():
    returns = _base_returns()
    ones = torch.ones(BSZ, RESP, dtype=torch.long)

    assert _loss(returns, ones) == _loss(returns)


def test_masked_out_positions_do_not_affect_loss():
    """The core guarantee: sentinel returns behind value_mask==0 are invisible."""
    returns = _base_returns()
    mask = torch.ones(BSZ, RESP, dtype=torch.long)
    mask[:, 3:] = 0

    poisoned = returns.clone()
    poisoned[:, 3:] = -100.0  # the sentinel turn-level GAE actually writes

    assert _loss(poisoned, mask) == _loss(returns, mask)


def test_supervised_positions_still_affect_loss():
    """Counterpart to the above — the mask must not silently disable everything."""
    returns = _base_returns()
    mask = torch.ones(BSZ, RESP, dtype=torch.long)
    mask[:, 3:] = 0

    changed = returns.clone()
    changed[:, 0] += 5.0  # inside the supervised region

    assert _loss(changed, mask) != _loss(returns, mask)


def test_bool_and_int_masks_agree():
    returns = _base_returns()
    mask_int = torch.tensor([[1, 1, 1, 0, 0, 0], [1, 0, 1, 0, 1, 0]], dtype=torch.long)

    assert _loss(returns, mask_int) == _loss(returns, mask_int.bool())


def test_all_zero_mask_yields_zero_loss():
    """Degenerate but well-defined: nothing supervised means nothing to learn from.
    Documented here so a future NaN regression is caught rather than silently trained on."""
    returns = _base_returns()
    zeros = torch.zeros(BSZ, RESP, dtype=torch.long)

    loss = _loss(returns, zeros)
    assert loss == loss, "all-zero value_mask produced NaN"
    assert loss == 0.0
