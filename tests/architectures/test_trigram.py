import pytest
import torch

from demoodle.architectures.trigram import TrigramArchitecture, TrigramModel
from demoodle.core.rng import RNG
from demoodle.core.types import Output, Policy

VOCAB = 10
RNG0 = RNG(seed=0)


@pytest.fixture
def arch() -> TrigramArchitecture:
    return TrigramArchitecture(vocab_size=VOCAB)


@pytest.fixture
def policy(arch: TrigramArchitecture) -> Policy:
    return arch.init_state(RNG0)


# ---------------------------------------------------------------------------
# TrigramModel
# ---------------------------------------------------------------------------


def test_model_forward_shape_single() -> None:
    model = TrigramModel(VOCAB)
    out = model(torch.tensor([2, 5]))
    assert out.shape == (VOCAB,)


def test_model_forward_row_lookup_single() -> None:
    model = TrigramModel(VOCAB)
    out = model(torch.tensor([1, 3]))
    assert torch.equal(out, model.weight[1, 3])


def test_model_forward_shape_batched() -> None:
    model = TrigramModel(VOCAB)
    out = model(torch.tensor([[2, 5], [0, 1]]))
    assert out.shape == (2, VOCAB)


def test_model_forward_row_lookup_batched() -> None:
    model = TrigramModel(VOCAB)
    x = torch.tensor([[1, 3], [0, 2]])
    out = model(x)
    assert torch.equal(out[0], model.weight[1, 3])
    assert torch.equal(out[1], model.weight[0, 2])


def test_model_single_parameter_group() -> None:
    model = TrigramModel(VOCAB)
    params = list(model.parameters())
    assert len(params) == 1
    assert params[0].shape == (VOCAB, VOCAB, VOCAB)


# ---------------------------------------------------------------------------
# init_state
# ---------------------------------------------------------------------------


def test_init_state_returns_policy_with_trigram_model(
    arch: TrigramArchitecture,
) -> None:
    policy = arch.init_state(RNG0)
    assert isinstance(policy, Policy)
    assert isinstance(policy.model, TrigramModel)


def test_init_state_weight_shape(arch: TrigramArchitecture) -> None:
    policy = arch.init_state(RNG0)
    assert isinstance(policy.model, TrigramModel)
    assert policy.model.weight.shape == (VOCAB, VOCAB, VOCAB)


def test_init_state_deterministic(arch: TrigramArchitecture) -> None:
    p1 = arch.init_state(RNG0)
    p2 = arch.init_state(RNG0)
    assert isinstance(p1.model, TrigramModel)
    assert isinstance(p2.model, TrigramModel)
    assert torch.equal(p1.model.weight, p2.model.weight)


def test_init_state_different_seeds_differ(arch: TrigramArchitecture) -> None:
    p1 = arch.init_state(RNG(seed=1))
    p2 = arch.init_state(RNG(seed=2))
    assert isinstance(p1.model, TrigramModel)
    assert isinstance(p2.model, TrigramModel)
    assert not torch.equal(p1.model.weight, p2.model.weight)


# ---------------------------------------------------------------------------
# context_length
# ---------------------------------------------------------------------------


def test_context_length_is_two(arch: TrigramArchitecture) -> None:
    assert arch.context_length == 2


# ---------------------------------------------------------------------------
# forward
# ---------------------------------------------------------------------------


def test_forward_logit_shape(arch: TrigramArchitecture, policy: Policy) -> None:
    tokens = torch.tensor([2, 5])
    output = arch.forward(policy, tokens)
    assert output.logits.shape == (VOCAB,)
    assert output.sampled_ids is None


def test_forward_uses_last_two_tokens(
    arch: TrigramArchitecture, policy: Policy
) -> None:
    tokens_exact = torch.tensor([3, 7])
    tokens_long = torch.tensor([0, 1, 2, 3, 7])
    out_exact = arch.forward(policy, tokens_exact)
    out_long = arch.forward(policy, tokens_long)
    assert torch.equal(out_exact.logits, out_long.logits)


def test_forward_different_pairs_differ(
    arch: TrigramArchitecture, policy: Policy
) -> None:
    out_a = arch.forward(policy, torch.tensor([1, 2]))
    out_b = arch.forward(policy, torch.tensor([2, 1]))
    # Different token order → different logits (weight is not symmetric)
    assert not torch.equal(out_a.logits, out_b.logits)


# ---------------------------------------------------------------------------
# call
# ---------------------------------------------------------------------------


def test_call_returns_sampled_ids(arch: TrigramArchitecture, policy: Policy) -> None:
    seq = torch.tensor([3, 4])
    output = arch.call(seq, policy, RNG0, temperature=1.0)
    assert isinstance(output, Output)
    assert output.sampled_ids is not None
    assert 0 <= int(output.sampled_ids.item()) < VOCAB


def test_call_deterministic_under_same_rng(
    arch: TrigramArchitecture, policy: Policy
) -> None:
    seq = torch.tensor([3, 4])
    rng = RNG(seed=99)
    r1 = arch.call(seq, policy, rng, temperature=1.0)
    r2 = arch.call(seq, policy, rng, temperature=1.0)
    assert r1.sampled_ids is not None
    assert r2.sampled_ids is not None
    assert int(r1.sampled_ids.item()) == int(r2.sampled_ids.item())


def _multi_draw(
    arch: TrigramArchitecture,
    policy: Policy,
    seq: torch.Tensor,
    temperature: float,
    n: int,
    top_k: int | None = None,
    top_p: float | None = None,
) -> list[int]:
    rng = RNG(seed=42)
    draws = []
    for _ in range(n):
        rng, call_rng = rng.split()
        result = arch.call(seq, policy, call_rng, temperature, top_k=top_k, top_p=top_p)
        assert result.sampled_ids is not None
        draws.append(int(result.sampled_ids.item()))
    return draws


def _draw(
    arch: TrigramArchitecture,
    policy: Policy,
    seq: torch.Tensor,
    temperature: float,
    top_k: int | None = None,
    top_p: float | None = None,
) -> int:
    result = arch.call(seq, policy, RNG0, temperature, top_k=top_k, top_p=top_p)
    assert result.sampled_ids is not None
    return int(result.sampled_ids.item())


def test_call_low_temperature_concentrates(
    arch: TrigramArchitecture, policy: Policy
) -> None:
    seq = torch.tensor([0, 1])
    draws = _multi_draw(arch, policy, seq, temperature=0.01, n=50)
    assert len(set(draws)) <= 2


def test_call_high_temperature_spreads(
    arch: TrigramArchitecture, policy: Policy
) -> None:
    seq = torch.tensor([0, 1])
    draws = _multi_draw(arch, policy, seq, temperature=100.0, n=200)
    assert len(set(draws)) > 2


def test_call_top_k_1_returns_argmax(arch: TrigramArchitecture, policy: Policy) -> None:
    seq = torch.tensor([2, 5])
    expected = int(arch.forward(policy, seq).logits.argmax().item())
    for _ in range(20):
        assert _draw(arch, policy, seq, temperature=1.0, top_k=1) == expected


def test_call_top_p_zero_returns_argmax(
    arch: TrigramArchitecture, policy: Policy
) -> None:
    seq = torch.tensor([7, 3])
    expected = int(arch.forward(policy, seq).logits.argmax().item())
    for _ in range(20):
        assert _draw(arch, policy, seq, temperature=1.0, top_p=0.0) == expected


# ---------------------------------------------------------------------------
# explain
# ---------------------------------------------------------------------------


def test_explain_returns_empty_dict(arch: TrigramArchitecture, policy: Policy) -> None:
    seq = torch.tensor([1, 2])
    assert arch.explain(seq, policy) == {}
