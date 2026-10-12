import torch
from torch import nn

from zipper_lora import ZipperLoRAConfig, ZipperLoRALinear, inject_zipper_lora, set_zipper_language_ids


def test_all_variants_can_be_injected_and_run():
    for variant in ("static", "hard", "soft"):
        model = nn.Sequential(nn.Linear(4, 6))
        config = ZipperLoRAConfig(3, 8, r=4, variant=variant, target_modules=("0",))
        assert inject_zipper_lora(model, config) == ["0"]
        assert isinstance(model[0], ZipperLoRALinear)
        set_zipper_language_ids(model, torch.tensor([0, 2]))
        assert model(torch.randn(2, 4)).shape == (2, 6)


class _Audio(nn.Module):
    def __init__(self):
        super().__init__()
        self.blocks = nn.ModuleList([nn.ModuleDict({"mlp": nn.Sequential(nn.Linear(4, 8), nn.GELU(), nn.Linear(8, 4))})])


def _host():
    host = nn.Module()
    host.model = nn.Module()
    host.model.audio = _Audio()
    return host


def test_flylora_checkpoint_layout_and_topk():
    from zipper_lora.checkpoint import CheckpointFlyLoRA, inject_checkpoint_flylora

    host = inject_checkpoint_flylora(_host(), r=4, k=2)
    keys = set(host.state_dict())
    assert {"model.audio.blocks.0.mlp.0.fly.A", "model.audio.blocks.0.mlp.0.fly.B",
            "model.audio.blocks.0.mlp.0.fly.d", "model.audio.blocks.0.mlp.2.base.weight"} <= keys

    fly = CheckpointFlyLoRA(3, 2, r=4, k=2)
    fly.A.copy_(torch.eye(4, 3))
    fly.B.data.copy_(torch.ones(2, 4))
    # y = [1, -3, 2, 0]: the two largest |y| ranks (-3, 2) survive; scale alpha/r = 2.
    assert torch.allclose(fly(torch.tensor([[1.0, -3.0, 2.0]])), torch.full((1, 2), -2.0))


def test_static_moe_state_maps_to_static_zipper():
    from zipper_lora.checkpoint import inject_checkpoint_zipper_lora, remap_static_moe_state

    prefix = "model.audio.blocks.0.mlp.0."
    state = {prefix + "base.weight": torch.zeros(8, 4), prefix + "moe.A": torch.zeros(4, 4),
             prefix + "moe.B_shared": torch.zeros(8, 2), prefix + "moe.B_experts": torch.zeros(3, 8, 2),
             "model.audio.chunk_moe_router.weight": torch.zeros(8, 2560)}
    remapped = remap_static_moe_state(state)
    assert set(remapped) == {prefix + "base.weight", prefix + "lora.A",
                             prefix + "lora.B_shared", prefix + "lora.B_specific"}
    host = inject_checkpoint_zipper_lora(_host(), nn.Embedding(3, 4), variant="static", num_languages=3, r=4)
    expected = {k for k in host.state_dict() if k.startswith(prefix)} - {prefix + "base.bias"}
    assert expected == set(remapped)
