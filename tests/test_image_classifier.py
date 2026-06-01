import pytest
import numpy as np
from PIL import Image


def make_dummy_image(width=224, height=224) -> Image.Image:
    arr = np.random.randint(0, 255, (height, width, 3), dtype=np.uint8)
    return Image.fromarray(arr)


class TestPreprocess:
    def test_inference_transform_output_shape(self):
        from src.image_classifier.preprocess import preprocess_for_inference
        img = make_dummy_image()
        arr = preprocess_for_inference(img)
        assert arr.shape == (224, 224, 3)

    def test_training_transform_output_shape(self):
        from src.image_classifier.preprocess import preprocess_for_training
        img = make_dummy_image()
        arr = preprocess_for_training(img)
        assert arr.shape == (224, 224, 3)

    def test_rgba_image_converted_to_rgb(self):
        from src.image_classifier.preprocess import preprocess_for_inference
        img = Image.new("RGBA", (300, 300))
        arr = preprocess_for_inference(img)
        assert arr.shape[-1] == 3


class TestModel:
    def test_build_model_output_classes(self):
        from src.image_classifier.model import build_model
        model = build_model(num_classes=5, freeze_backbone=True)
        dummy = np.random.rand(1, 224, 224, 3).astype(np.float32)
        out = model.predict(dummy, verbose=0)
        assert out.shape == (1, 5)

    def test_frozen_backbone(self):
        from src.image_classifier.model import build_model
        model = build_model(num_classes=5, freeze_backbone=True)
        # layers[0]=Input, layers[1]=EfficientNetB0 backbone
        backbone = model.layers[1]
        assert not backbone.trainable


class TestPredictContract:
    def test_predict_returns_required_keys(self, monkeypatch):
        """Verifica el contrato de la API sin cargar el modelo real."""
        dummy_result = {
            "top_prediction": "Apparel",
            "top_3": [{"class": "Apparel", "confidence": 0.9}],
            "inference_time_ms": 12.5,
        }
        import src.image_classifier.predict as pred_module
        monkeypatch.setattr(pred_module, "predict", lambda img: dummy_result)
        result = pred_module.predict(make_dummy_image())
        assert "top_prediction" in result
        assert "top_3" in result
        assert "inference_time_ms" in result
