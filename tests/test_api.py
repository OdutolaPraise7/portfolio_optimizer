import tempfile
import unittest
from pathlib import Path

import pandas as pd

try:
    from fastapi.testclient import TestClient
    import main
    import portfolio_store
    FASTAPI_AVAILABLE = True
except (ModuleNotFoundError, RuntimeError):
    # This keeps the Python-only test suite usable even if FastAPI test dependencies are not installed.
    FASTAPI_AVAILABLE = False


@unittest.skipUnless(FASTAPI_AVAILABLE, "fastapi is not installed in this environment")
class ApiTests(unittest.TestCase):
    def setUp(self):
        # These tests patch the backend to use temporary files so API behavior
        # can be checked independently from the real dataset on disk.
        self.temp_dir = tempfile.TemporaryDirectory()
        self.tmp = Path(self.temp_dir.name)
        self.price_file = self.tmp / "PRICE_LIST.csv"
        self.signal_file = self.tmp / "signal_store.csv"
        self.store_file = self.tmp / "portfolio_store.json"
        self.original_store_file = portfolio_store.STORE_FILE
        portfolio_store.STORE_FILE = self.store_file

        price_rows = []
        for date in pd.date_range("2024-01-01", periods=100, freq="D"):
            price_rows.extend(
                [
                    {"SYMBOL": "AAA", "TRANS_DATE": date.strftime("%Y-%m-%d"), "CLOSE_PRICE": 10 + 0.05 * len(price_rows), "VOLUME": 100000, "TRADE_VALUE": 1000000},
                    {"SYMBOL": "BBB", "TRANS_DATE": date.strftime("%Y-%m-%d"), "CLOSE_PRICE": 8 + 0.03 * len(price_rows), "VOLUME": 120000, "TRADE_VALUE": 960000},
                    {"SYMBOL": "CCC", "TRANS_DATE": date.strftime("%Y-%m-%d"), "CLOSE_PRICE": 6 + 0.07 * len(price_rows), "VOLUME": 150000, "TRADE_VALUE": 900000},
                ]
            )
        pd.DataFrame(price_rows).to_csv(self.price_file, index=False)
        pd.DataFrame(
            [
                {"Symbol": "AAA", "Consensus_Signal": "BUY", "Consensus_Tier": 1, "Avg_Confidence": 0.2, "XGB_Return (%)": 3, "RF_Return (%)": 3, "LSTM_Return (%)": 4},
                {"Symbol": "CCC", "Consensus_Signal": "BUY", "Consensus_Tier": 1, "Avg_Confidence": 0.4, "XGB_Return (%)": 6, "RF_Return (%)": 5, "LSTM_Return (%)": 7},
            ]
        ).to_csv(self.signal_file, index=False)

        self.original_price_file = main.optimize_portfolio.__defaults__[4]
        self.original_signal_file = main.optimize_portfolio.__defaults__[5]

        self.client = TestClient(main.app, raise_server_exceptions=False)

    def tearDown(self):
        portfolio_store.STORE_FILE = self.original_store_file
        self.temp_dir.cleanup()

    def test_symbols_endpoint(self):
        original = main.get_supported_symbols
        main.get_supported_symbols = lambda: ["AAA", "BBB", "CCC"]
        try:
            response = self.client.get("/symbols")
            self.assertEqual(response.status_code, 200)
            self.assertEqual(response.json()["symbols"], ["AAA", "BBB", "CCC"])
        finally:
            main.get_supported_symbols = original

    def test_optimize_portfolio_endpoint(self):
        original = main.optimize_portfolio

        def patched_optimize_portfolio(**kwargs):
            kwargs["price_file"] = str(self.price_file)
            kwargs["signal_file"] = str(self.signal_file)
            return original(**kwargs)

        main.optimize_portfolio = patched_optimize_portfolio
        try:
            response = self.client.post(
                "/optimize-portfolio",
                json={
                    "holdings": [
                        {"symbol": "AAA", "amount_naira": 4000},
                        {"symbol": "BBB", "amount_naira": 6000},
                    ],
                    "risk_profile": "balanced",
                    "mandate_profile": "balanced_equity",
                    "allow_new_stocks": True,
                    "max_new_stocks": 1,
                },
            )
            self.assertEqual(response.status_code, 200)
            body = response.json()
            self.assertIn("current_weights", body)
            self.assertIn("optimized_allocations", body)
            self.assertIn("summary_metrics", body)
            self.assertIn("prediction_engine", body)
            self.assertIn("compliance_report", body)
            self.assertIn("fund_manager_report", body)
        finally:
            main.optimize_portfolio = original

    def test_optimize_portfolio_endpoint_rejects_invalid_mandate(self):
        response = self.client.post(
            "/optimize-portfolio",
            json={
                "holdings": [{"symbol": "AAA", "amount_naira": 4000}],
                "risk_profile": "balanced",
                "mandate_profile": "invalid",
            },
        )
        self.assertEqual(response.status_code, 422)

    def test_signal_summary_endpoint_handles_missing_signal_store(self):
        original = main.get_signal_summary
        main.get_signal_summary = lambda: (_ for _ in ()).throw(Exception("boom"))
        try:
            response = self.client.get("/signals/summary")
            self.assertEqual(response.status_code, 500)
        finally:
            main.get_signal_summary = original

    def test_fund_manager_can_save_portfolio(self):
        manager_response = self.client.post(
            "/fund-managers",
            json={"name": "Ada Manager", "firm": "Lagos Asset Co", "email": "ada@example.com"},
        )
        self.assertEqual(manager_response.status_code, 200)
        manager_id = manager_response.json()["manager"]["id"]

        save_response = self.client.post(
            f"/fund-managers/{manager_id}/portfolios",
            json={
                "name": "Balanced Equity Sleeve",
                "holdings": [{"symbol": "AAA", "amount_naira": 4000}],
                "risk_profile": "balanced",
                "mandate_profile": "balanced_equity",
                "allow_new_stocks": True,
                "max_new_stocks": 1,
            },
        )
        self.assertEqual(save_response.status_code, 200)
        portfolio = save_response.json()["portfolio"]
        self.assertEqual(portfolio["manager_id"], manager_id)

        list_response = self.client.get(f"/fund-managers/{manager_id}/portfolios")
        self.assertEqual(list_response.status_code, 200)
        self.assertEqual(len(list_response.json()["portfolios"]), 1)
