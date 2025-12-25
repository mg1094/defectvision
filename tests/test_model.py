"""模型单元测试"""

import pytest
import torch

from defectvision.model import build_model


class TestBuildModel:
    """测试 build_model 函数"""

    @pytest.mark.parametrize("backbone", ["smallcnn", "resnet18", "mobilenet_v3_small"])
    def test_build_model_backbones(self, backbone: str) -> None:
        """测试不同骨干的模型构建"""
        model = build_model(
            in_channels=1,
            num_classes=2,
            backbone=backbone,
            pretrained=False,
        )
        assert model is not None
        assert hasattr(model, "last_conv"), f"{backbone} should have last_conv attribute"

    @pytest.mark.parametrize("backbone", ["smallcnn", "resnet18", "mobilenet_v3_small"])
    def test_model_forward(self, backbone: str) -> None:
        """测试模型前向传播"""
        model = build_model(
            in_channels=1,
            num_classes=2,
            backbone=backbone,
            pretrained=False,
        )
        model.eval()

        x = torch.randn(2, 1, 128, 128)
        with torch.no_grad():
            logits = model(x)

        assert logits.shape == (2, 2), f"Expected (2, 2), got {logits.shape}"

    def test_invalid_backbone(self) -> None:
        """测试无效骨干名称"""
        with pytest.raises(ValueError, match="Unsupported backbone"):
            build_model(backbone="invalid_backbone")

    @pytest.mark.parametrize("num_classes", [2, 5, 10])
    def test_num_classes(self, num_classes: int) -> None:
        """测试不同类别数"""
        model = build_model(
            in_channels=1,
            num_classes=num_classes,
            backbone="smallcnn",
            pretrained=False,
        )
        model.eval()

        x = torch.randn(1, 1, 128, 128)
        with torch.no_grad():
            logits = model(x)

        assert logits.shape == (1, num_classes)

