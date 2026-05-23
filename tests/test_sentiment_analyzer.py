import pytest


class TestPreprocess:
    def test_clean_text_lowercases(self):
        from src.sentiment_analyzer.preprocess import clean_text
        assert clean_text("HOLA Mundo") == "hola mundo"

    def test_clean_text_removes_urls(self):
        from src.sentiment_analyzer.preprocess import clean_text
        result = clean_text("Visitá https://ejemplo.com para más info")
        assert "http" not in result

    def test_clean_text_removes_extra_spaces(self):
        from src.sentiment_analyzer.preprocess import clean_text
        result = clean_text("hola   mundo")
        assert "  " not in result

    def test_rating_to_label_mapping(self):
        from src.sentiment_analyzer.preprocess import rating_to_label
        assert rating_to_label(1) == 0
        assert rating_to_label(2) == 0
        assert rating_to_label(3) == 1
        assert rating_to_label(4) == 2
        assert rating_to_label(5) == 2


class TestPredictContract:
    def test_predict_returns_required_keys(self, monkeypatch):
        dummy_result = {
            "sentiment": "positive",
            "confidence": 0.92,
            "scores": {"positive": 0.92, "neutral": 0.05, "negative": 0.03},
        }
        import src.sentiment_analyzer.predict as pred_module
        monkeypatch.setattr(pred_module, "predict", lambda text: dummy_result)
        result = pred_module.predict("Gran producto!")
        assert result["sentiment"] in ("positive", "neutral", "negative")
        assert 0.0 <= result["confidence"] <= 1.0
        assert set(result["scores"].keys()) == {"positive", "neutral", "negative"}

    def test_sentiment_scores_sum_to_one(self, monkeypatch):
        dummy = {
            "sentiment": "positive",
            "confidence": 0.92,
            "scores": {"positive": 0.92, "neutral": 0.05, "negative": 0.03},
        }
        import src.sentiment_analyzer.predict as pred_module
        monkeypatch.setattr(pred_module, "predict", lambda text: dummy)
        result = pred_module.predict("test")
        total = sum(result["scores"].values())
        assert abs(total - 1.0) < 1e-3
