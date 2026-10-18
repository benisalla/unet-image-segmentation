import pytest
import torch
import torch.nn as nn

from unetseg.config import PRESETS, ModelConfig
from unetseg.models import ConvBlock, Down, EncoderDecoder, FullyConnectedSeg, build_model, count_parameters


def small(name, **kw):
    return ModelConfig.from_preset(name, base=4, depth=2, img_size=32, fcn_hidden=16, **kw)


@pytest.mark.parametrize("name", sorted(PRESETS))
def test_every_preset_gives_one_logit_per_pixel(name):
    model = build_model(small(name))
    y = model(torch.randn(2, 3, 32, 32))
    assert y.shape == (2, 1, 32, 32)
    y.mean().backward()


def test_presets_set_the_switches():
    assert ModelConfig.from_preset("segnet").skip is False
    assert ModelConfig.from_preset("unet-strided").down == "strided"
    assert ModelConfig.from_preset("unet-single").double is False
    assert ModelConfig.from_preset("unet-residual").residual is True
    assert ModelConfig.from_preset("unet", skip=False).skip is False  # explicit override wins
    with pytest.raises(ValueError):
        ModelConfig.from_preset("vgg")


def test_skip_connections_change_the_decoder_input():
    with_skip = EncoderDecoder(small("unet"))
    no_skip = EncoderDecoder(small("segnet"))
    assert with_skip.ups[0].block.body[0][0].in_channels == 2 * no_skip.ups[0].block.body[0][0].in_channels
    assert count_parameters(with_skip) > count_parameters(no_skip)


def test_maxpool_has_no_weights_strided_does():
    pool = Down(8, 16, "maxpool", True, False)
    stride = Down(8, 16, "strided", True, False)
    assert sum(p.numel() for p in pool.reduce.parameters()) == 0
    assert sum(p.numel() for p in stride.reduce.parameters()) > 0
    x = torch.randn(1, 8, 16, 16)
    assert pool(x).shape == stride(x).shape == (1, 16, 8, 8)


def test_single_vs_double_conv_and_residual_shortcut():
    single = ConvBlock(4, 8, double=False)
    double = ConvBlock(4, 8, double=True)
    assert sum(isinstance(m, nn.Conv2d) for m in single.modules()) == 1
    assert sum(isinstance(m, nn.Conv2d) for m in double.modules()) == 2
    res = ConvBlock(4, 8, double=True, residual=True)
    assert isinstance(res.shortcut[0], nn.Conv2d)  # 1x1 projection when channels change
    assert isinstance(ConvBlock(8, 8, residual=True).shortcut, nn.Identity)


def test_unet_handles_sizes_not_divisible_by_two_to_the_depth():
    model = EncoderDecoder(small("unet"))
    assert model(torch.randn(1, 3, 50, 50)).shape == (1, 1, 50, 50)


def test_fcn_is_fully_connected_and_can_output_negative_logits():
    model = FullyConnectedSeg(small("fcn")).eval()
    assert not any(isinstance(m, nn.Conv2d) for m in model.modules())
    assert isinstance(model.net[-1], nn.Linear)  # the notebook ended with BN+ReLU: logits were never negative
    assert (model(torch.randn(4, 3, 32, 32)) < 0).any()
    with pytest.raises(ValueError):
        model(torch.randn(1, 3, 64, 64))
