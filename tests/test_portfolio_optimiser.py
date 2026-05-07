import json
import tempfile
import unittest
from pathlib import Path

import pandas as pd

import portfolio_optimiser
from portfolio_optimiser import (
    SignalStoreError,
    ValidationError,
    construct_portfolio,
    get_supported_symbols,
    optimize_portfolio,
)


def _write_csv(path: Path, rows, columns):
    pd.DataFrame(rows, columns=columns).to_csv(path, index=False)


class PortfolioOptimiserTests(unittest.TestCase):
    def setUp(self):
        # Each test gets its own temporary mini-market so the optimizer can be
        # tested deterministically without touching real project data files.
        self.original_samples = portfolio_optimiser.RANDOM_PORTFOLIO_SAMPLES
        portfolio_optimiser.RANDOM_PORTFOLIO_SAMPLES = 300
        self.temp_dir = tempfile.TemporaryDirectory()
        self.tmp = Path(self.temp_dir.name)
        self.price_file = self.tmp / "PRICE_LIST.csv"
        self.signal_file = self.tmp / "signal_store.csv"

        price_rows = []
        dates = pd.date_range("2024-01-01", periods=90, freq="D")
        base_prices = {
            "AAA": 10.0,
            "BBB": 12.0,
            "CCC": 8.0,
            "DDD": 6.0,
        }
        growth = {
            "AAA": 0.0020,
            "BBB": 0.0010,
            "CCC": 0.0030,
            "DDD": 0.0005,
        }
        # These synthetic prices create predictable trend differences between symbols.
        for i, date in enumerate(dates):
            for symbol, start in base_prices.items():
                close = start * (1 + growth[symbol]) ** i
                price_rows.append(
                    {
                        "SYMBOL": symbol,
                        "TRANS_DATE": date.strftime("%Y-%m-%d"),
                        "CLOSE_PRICE": close,
                        "VOLUME": 100000 + i * 1000,
                        "TRADE_VALUE": close * (100000 + i * 1000),
                    }
                )
        _write_csv(
            self.price_file,
            price_rows,
            ["SYMBOL", "TRANS_DATE", "CLOSE_PRICE", "VOLUME", "TRADE_VALUE"],
        )

        signal_rows = [
            {
                "Symbol": "AAA",
                "Consensus_Signal": "BUY",
                "Consensus_Tier": 1,
                "Avg_Confidence": 0.20,
                "Avg_R2": 0.10,
                "XGB_Return (%)": 4.0,
                "XGB_Signal": "BUY",
                "XGB_Confidence": 0.20,
                "XGB_R2": 0.10,
                "XGB_Quality_Pass": True,
                "RF_Return (%)": 3.0,
                "RF_Signal": "BUY",
                "RF_Confidence": 0.18,
                "RF_R2": 0.12,
                "RF_Quality_Pass": True,
                "LSTM_Return (%)": 5.0,
                "LSTM_Signal": "BUY",
                "LSTM_Confidence": 0.22,
                "LSTM_R2": 0.08,
                "LSTM_Quality_Pass": True,
                "Qualified_Models": 3,
            },
            {
                "Symbol": "BBB",
                "Consensus_Signal": "BUY",
                "Consensus_Tier": 2,
                "Avg_Confidence": 0.12,
                "Avg_R2": 0.08,
                "XGB_Return (%)": 2.0,
                "XGB_Signal": "BUY",
                "XGB_Confidence": 0.12,
                "XGB_R2": 0.08,
                "XGB_Quality_Pass": True,
                "RF_Return (%)": 2.5,
                "RF_Signal": "BUY",
                "RF_Confidence": 0.10,
                "RF_R2": 0.07,
                "RF_Quality_Pass": True,
                "LSTM_Return (%)": 1.5,
                "LSTM_Signal": "BUY",
                "LSTM_Confidence": 0.14,
                "LSTM_R2": 0.05,
                "LSTM_Quality_Pass": True,
                "Qualified_Models": 3,
            },
            {
                "Symbol": "CCC",
                "Consensus_Signal": "BUY",
                "Consensus_Tier": 1,
                "Avg_Confidence": 0.30,
                "Avg_R2": 0.15,
                "XGB_Return (%)": 6.0,
                "XGB_Signal": "BUY",
                "XGB_Confidence": 0.30,
                "XGB_R2": 0.15,
                "XGB_Quality_Pass": True,
                "RF_Return (%)": 7.0,
                "RF_Signal": "BUY",
                "RF_Confidence": 0.32,
                "RF_R2": 0.16,
                "RF_Quality_Pass": True,
                "LSTM_Return (%)": 5.0,
                "LSTM_Signal": "BUY",
                "LSTM_Confidence": 0.28,
                "LSTM_R2": 0.13,
                "LSTM_Quality_Pass": True,
                "Qualified_Models": 3,
            },
        ]
        pd.DataFrame(signal_rows).to_csv(self.signal_file, index=False)

    def tearDown(self):
        portfolio_optimiser.RANDOM_PORTFOLIO_SAMPLES = self.original_samples
        self.temp_dir.cleanup()

    def test_get_supported_symbols(self):
        symbols = get_supported_symbols(str(self.price_file))
        self.assertEqual(symbols, ["AAA", "BBB", "CCC", "DDD"])

    def test_empty_holdings_rejected(self):
        with self.assertRaises(ValidationError):
            optimize_portfolio(
                holdings=[],
                price_file=str(self.price_file),
                signal_file=str(self.signal_file),
            )

    def test_invalid_symbol_rejected(self):
        with self.assertRaises(ValidationError):
            optimize_portfolio(
                holdings=[{"symbol": "ZZZ", "amount_naira": 1000}],
                price_file=str(self.price_file),
                signal_file=str(self.signal_file),
            )

    def test_non_positive_amount_rejected(self):
        with self.assertRaises(ValidationError):
            optimize_portfolio(
                holdings=[{"symbol": "AAA", "amount_naira": 0}],
                price_file=str(self.price_file),
                signal_file=str(self.signal_file),
            )

    def test_missing_signal_store_raises_controlled_error(self):
        with self.assertRaises(SignalStoreError):
            optimize_portfolio(
                holdings=[{"symbol": "AAA", "amount_naira": 1000}],
                price_file=str(self.price_file),
                signal_file=str(self.tmp / "missing_signal_store.csv"),
            )

    def test_invalid_mandate_rejected(self):
        with self.assertRaises(ValidationError):
            optimize_portfolio(
                holdings=[{"symbol": "AAA", "amount_naira": 1000}],
                mandate_profile="crypto_equity",
                price_file=str(self.price_file),
                signal_file=str(self.signal_file),
            )

    def test_invalid_max_new_stocks_rejected(self):
        with self.assertRaises(ValidationError):
            optimize_portfolio(
                holdings=[{"symbol": "AAA", "amount_naira": 1000}],
                max_new_stocks=-1,
                price_file=str(self.price_file),
                signal_file=str(self.signal_file),
            )

    def test_weights_sum_to_one_and_non_negative(self):
        # Long-only portfolio outputs should always behave like a valid allocation.
        result = optimize_portfolio(
            holdings=[
                {"symbol": "AAA", "amount_naira": 4000},
                {"symbol": "BBB", "amount_naira": 6000},
            ],
            risk_profile="balanced",
            allow_new_stocks=True,
            max_new_stocks=2,
            price_file=str(self.price_file),
            signal_file=str(self.signal_file),
        )
        weights = [row["optimized_weight"] for row in result["optimized_allocations"]]
        self.assertAlmostEqual(sum(weights), 1.0, places=5)
        self.assertTrue(all(weight >= 0 for weight in weights))

    def test_new_stocks_only_added_when_enabled(self):
        # The "allow new stocks" toggle is a user-facing product behavior, so it gets a direct test.
        disabled = optimize_portfolio(
            holdings=[{"symbol": "AAA", "amount_naira": 5000}],
            allow_new_stocks=False,
            price_file=str(self.price_file),
            signal_file=str(self.signal_file),
        )
        enabled = optimize_portfolio(
            holdings=[{"symbol": "AAA", "amount_naira": 5000}],
            allow_new_stocks=True,
            max_new_stocks=2,
            price_file=str(self.price_file),
            signal_file=str(self.signal_file),
        )
        self.assertEqual(disabled["added_symbols"], [])
        self.assertGreaterEqual(len(enabled["optimized_allocations"]), len(disabled["optimized_allocations"]))

    def test_multiple_new_stocks_can_be_recommended(self):
        result = optimize_portfolio(
            holdings=[{"symbol": "DDD", "amount_naira": 5000}],
            allow_new_stocks=True,
            max_new_stocks=2,
            price_file=str(self.price_file),
            signal_file=str(self.signal_file),
        )

        self.assertGreaterEqual(len(result["added_symbols"]), 2)

    def test_aggressive_differs_from_conservative(self):
        # Risk profile should materially change the mandate limits even when a
        # tiny synthetic universe makes the final weights infeasible to separate.
        conservative = optimize_portfolio(
            holdings=[
                {"symbol": "AAA", "amount_naira": 5000},
                {"symbol": "BBB", "amount_naira": 5000},
            ],
            risk_profile="conservative",
            allow_new_stocks=True,
            max_new_stocks=1,
            price_file=str(self.price_file),
            signal_file=str(self.signal_file),
        )
        aggressive = optimize_portfolio(
            holdings=[
                {"symbol": "AAA", "amount_naira": 5000},
                {"symbol": "BBB", "amount_naira": 5000},
            ],
            risk_profile="aggressive",
            allow_new_stocks=True,
            max_new_stocks=1,
            price_file=str(self.price_file),
            signal_file=str(self.signal_file),
        )
        self.assertNotEqual(
            conservative["mandate_summary"]["max_stock_weight"],
            aggressive["mandate_summary"]["max_stock_weight"],
        )

    def test_engine_returns_fund_manager_contract(self):
        result = optimize_portfolio(
            holdings=[
                {"symbol": "AAA", "amount_naira": 4000},
                {"symbol": "BBB", "amount_naira": 6000},
            ],
            risk_profile="balanced",
            mandate_profile="growth_equity",
            allow_new_stocks=True,
            max_new_stocks=2,
            price_file=str(self.price_file),
            signal_file=str(self.signal_file),
        )

        self.assertEqual(result["mandate_profile"], "growth_equity")
        self.assertIn("prediction_engine", result)
        self.assertIn("compliance_report", result)
        self.assertIn("fund_manager_report", result)
        self.assertEqual(result["prediction_engine"]["scope"], "Nigerian listed equities")
        self.assertIn(result["compliance_report"]["overall_status"], {"pass", "review", "breach"})
        self.assertGreater(len(result["compliance_report"]["items"]), 0)
        self.assertIn("recommendation", result["fund_manager_report"])
        self.assertTrue(result["optimized_allocations"][0]["model_votes"])
        json.dumps(result, allow_nan=False)

    def test_constructs_first_portfolio_from_cash(self):
        result = construct_portfolio(
            initial_cash_naira=5000,
            max_stocks=2,
            price_file=str(self.price_file),
            signal_file=str(self.signal_file),
        )

        self.assertEqual(result["portfolio_mode"], "construction")
        self.assertEqual(result["current_weights"], [])
        self.assertEqual(result["current_portfolio_value"], 5000)
        self.assertGreaterEqual(len(result["added_symbols"]), 1)
        self.assertTrue(all(row["current_weight"] == 0 for row in result["optimized_allocations"]))
        json.dumps(result, allow_nan=False)

    def test_pension_mandate_tightens_constraints(self):
        result = optimize_portfolio(
            holdings=[
                {"symbol": "AAA", "amount_naira": 4000},
                {"symbol": "BBB", "amount_naira": 6000},
            ],
            risk_profile="aggressive",
            mandate_profile="pension_equity",
            allow_new_stocks=True,
            max_new_stocks=2,
            price_file=str(self.price_file),
            signal_file=str(self.signal_file),
        )

        self.assertEqual(result["mandate_summary"]["max_stock_weight"], 0.07)
        self.assertEqual(result["mandate_summary"]["max_sector_weight"], 0.3)
