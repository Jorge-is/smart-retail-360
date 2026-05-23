import pytest
import pandas as pd


class TestFeatures:
    def test_add_temporal_features_columns(self):
        from src.sales_predictor.features import add_temporal_features
        df = pd.DataFrame({"ds": pd.date_range("2023-01-01", periods=10)})
        result = add_temporal_features(df)
        expected = {"day_of_week", "day_of_month", "week_of_year", "month", "is_weekend", "quarter"}
        assert expected.issubset(set(result.columns))

    def test_is_weekend_flag(self):
        from src.sales_predictor.features import add_temporal_features
        df = pd.DataFrame({"ds": ["2024-01-06", "2024-01-07", "2024-01-08"]})  # Sáb, Dom, Lun
        result = add_temporal_features(df)
        assert list(result["is_weekend"]) == [1, 1, 0]

    def test_prepare_prophet_df_columns(self):
        from src.sales_predictor.features import prepare_prophet_df
        raw = pd.DataFrame({
            "Store": [1, 1, 2],
            "Date": ["2023-01-01", "2023-01-02", "2023-01-01"],
            "Sales": [100, 200, 300],
        })
        result = prepare_prophet_df(raw, store_id=1)
        assert list(result.columns) == ["ds", "y"]
        assert len(result) == 2


class TestPredictContract:
    def test_predict_returns_required_keys(self, monkeypatch):
        dummy_result = {
            "forecast": [{"date": "2024-01-01", "predicted_sales": 500.0, "lower": 400.0, "upper": 600.0}],
            "metrics": {"mae": 50.0, "rmse": 70.0, "mape": 10.0},
        }
        import src.sales_predictor.predict as pred_module
        monkeypatch.setattr(pred_module, "predict", lambda **kwargs: dummy_result)
        result = pred_module.predict(store_id=1, horizon_days=7)
        assert "forecast" in result
        assert "metrics" in result
        assert isinstance(result["forecast"], list)
        entry = result["forecast"][0]
        assert {"date", "predicted_sales", "lower", "upper"}.issubset(entry.keys())
