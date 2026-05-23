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
        self.original_sector_map = portfolio_optimiser.SECTOR_MAP.copy()
        portfolio_optimiser.RANDOM_PORTFOLIO_SAMPLES = 300
        self.temp_dir = tempfile.TemporaryDirectory()
        self.tmp = Path(self.temp_dir.name)
        self.price_file = self.tmp / "PRICE_LIST.csv"
        self.signal_file = self.tmp / "signal_store.csv"

        price_rows = []
        dates = pd.date_range("2024-01-01", periods=90, freq="D")
        symbols = [
            "AAA", "BBB", "CCC", "DDD", "EEE", "FFF", "GGG", "HHH",
            "III", "JJJ", "KKK", "LLL", "MMM", "NNN", "OOO", "PPP",
        ]
        base_prices = {symbol: 8.0 + index for index, symbol in enumerate(symbols)}
        growth = {symbol: 0.0005 + (index % 4) * 0.0002 for index, symbol in enumerate(symbols)}
        sector_names = ["Banking", "Consumer Goods", "Industrial Goods", "Oil and Gas"]
        portfolio_optimiser.SECTOR_MAP.update(
            {symbol: sector_names[index % len(sector_names)] for index, symbol in enumerate(symbols)}
        )
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
                        "TRADE_VALUE": 1_000_000,
                    }
                )
        _write_csv(
            self.price_file,
            price_rows,
            ["SYMBOL", "TRANS_DATE", "CLOSE_PRICE", "VOLUME", "TRADE_VALUE"],
        )

        signal_rows = []
        for index, symbol in enumerate(symbols):
            expected_return = 1.5 + (index % 5) * 0.4
            confidence = 0.10 + index * 0.01
            signal_rows.append(
                {
                    "Symbol": symbol,
                    "Consensus_Signal": "BUY",
                    "Consensus_Tier": 1 if index % 3 != 0 else 2,
                    "Avg_Confidence": confidence,
                    "Avg_R2": 0.10,
                    "XGB_Return (%)": expected_return,
                    "XGB_Signal": "BUY",
                    "XGB_Confidence": confidence,
                    "XGB_R2": 0.10,
                    "XGB_Quality_Pass": True,
                    "RF_Return (%)": expected_return,
                    "RF_Signal": "BUY",
                    "RF_Confidence": confidence,
                    "RF_R2": 0.10,
                    "RF_Quality_Pass": True,
                    "LSTM_Return (%)": expected_return,
                    "LSTM_Signal": "BUY",
                    "LSTM_Confidence": confidence,
                    "LSTM_R2": 0.10,
                    "LSTM_Quality_Pass": True,
                    "Qualified_Models": 3,
                }
            )
        pd.DataFrame(signal_rows).to_csv(self.signal_file, index=False)

    def tearDown(self):
        portfolio_optimiser.RANDOM_PORTFOLIO_SAMPLES = self.original_samples
        portfolio_optimiser.SECTOR_MAP.clear()
        portfolio_optimiser.SECTOR_MAP.update(self.original_sector_map)
        self.temp_dir.cleanup()

    def compliant_holdings(self):
        return [
            {"symbol": symbol, "amount_naira": 1000}
            for symbol in ["GGG", "HHH", "III", "JJJ", "KKK", "LLL", "MMM", "NNN", "OOO", "PPP"]
        ]

    def test_get_supported_symbols(self):
        symbols = get_supported_symbols(str(self.price_file))
        self.assertEqual(symbols[:4], ["AAA", "BBB", "CCC", "DDD"])
        self.assertEqual(len(symbols), 16)

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
        with self.assertRaises(ValidationError):
            optimize_portfolio(
                holdings=[{"symbol": "AAA", "amount_naira": 1000}],
                max_new_stocks=21,
                price_file=str(self.price_file),
                signal_file=str(self.signal_file),
            )

    def test_more_than_ten_current_stocks_can_be_used_when_compliant(self):
        result = optimize_portfolio(
            holdings=self.compliant_holdings() + [{"symbol": "AAA", "amount_naira": 1000}],
            allow_new_stocks=False,
            price_file=str(self.price_file),
            signal_file=str(self.signal_file),
        )

        self.assertGreater(len(result["optimized_allocations"]), 10)

    def test_weights_sum_to_one_and_non_negative(self):
        # Long-only portfolio outputs should always behave like a valid allocation.
        result = optimize_portfolio(
            holdings=self.compliant_holdings(),
            risk_profile="balanced",
            allow_new_stocks=False,
            price_file=str(self.price_file),
            signal_file=str(self.signal_file),
        )
        weights = [row["optimized_weight"] for row in result["optimized_allocations"]]
        self.assertAlmostEqual(sum(weights), 1.0, places=5)
        self.assertTrue(all(weight >= 0 for weight in weights))

    def test_new_stock_shortfall_is_auto_broadened(self):
        # When a mandate needs more active names than the current holdings,
        # optimization should return a reviewed result instead of throwing.
        result = optimize_portfolio(
            holdings=[{"symbol": "AAA", "amount_naira": 5000}],
            allow_new_stocks=False,
            price_file=str(self.price_file),
            signal_file=str(self.signal_file),
        )
        self.assertTrue(result["allow_new_stocks"])
        self.assertGreaterEqual(len(result["optimized_allocations"]), 10)
        self.assertGreaterEqual(len(result["added_symbols"]), 1)
        self.assertIn(result["compliance_report"]["overall_status"], {"pass", "review", "breach"})

        capped = optimize_portfolio(
            holdings=[{"symbol": "AAA", "amount_naira": 5000}],
            allow_new_stocks=True,
            max_new_stocks=9,
            price_file=str(self.price_file),
            signal_file=str(self.signal_file),
        )
        self.assertGreaterEqual(len(capped["optimized_allocations"]), 10)

    def test_multiple_new_stocks_can_be_recommended(self):
        result = optimize_portfolio(
            holdings=[{"symbol": "DDD", "amount_naira": 5000}],
            allow_new_stocks=True,
            max_new_stocks=9,
            price_file=str(self.price_file),
            signal_file=str(self.signal_file),
        )

        self.assertGreaterEqual(len(result["added_symbols"]), 2)
        self.assertIn("optimized_allocations", result)

    def test_aggressive_differs_from_conservative(self):
        # Risk profile should materially change the mandate limits even when a
        # tiny synthetic universe makes the final weights infeasible to separate.
        conservative = optimize_portfolio(
            holdings=self.compliant_holdings() + [
                {"symbol": "AAA", "amount_naira": 1000},
                {"symbol": "BBB", "amount_naira": 1000},
                {"symbol": "CCC", "amount_naira": 1000},
            ],
            risk_profile="conservative",
            allow_new_stocks=False,
            price_file=str(self.price_file),
            signal_file=str(self.signal_file),
        )
        aggressive = optimize_portfolio(
            holdings=self.compliant_holdings(),
            risk_profile="aggressive",
            allow_new_stocks=False,
            price_file=str(self.price_file),
            signal_file=str(self.signal_file),
        )
        self.assertNotEqual(
            conservative["mandate_summary"]["max_stock_weight"],
            aggressive["mandate_summary"]["max_stock_weight"],
        )

    def test_engine_returns_fund_manager_contract(self):
        result = optimize_portfolio(
            holdings=self.compliant_holdings(),
            risk_profile="balanced",
            mandate_profile="growth_equity",
            allow_new_stocks=False,
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
        self.assertGreater(len(result["efficient_frontier"]["points"]), 0)
        self.assertEqual(
            len(result["correlation_matrix"]["symbols"]),
            len(result["correlation_matrix"]["values"]),
        )
        self.assertGreater(len(result["risk_contributions"]), 0)
        self.assertGreaterEqual(result["diversification_score"]["score"], 0)
        self.assertLessEqual(result["diversification_score"]["score"], 100)
        json.dumps(result, allow_nan=False)

    def test_constructs_first_portfolio_from_cash(self):
        result = construct_portfolio(
            initial_cash_naira=5000,
            max_stocks=13,
            price_file=str(self.price_file),
            signal_file=str(self.signal_file),
        )

        self.assertEqual(result["portfolio_mode"], "construction")
        self.assertEqual(result["current_weights"], [])
        self.assertEqual(result["current_portfolio_value"], 5000)
        self.assertAlmostEqual(
            result["optimized_portfolio_value"],
            result["current_portfolio_value"] - result["constraint_summary"]["estimated_transaction_cost_naira"],
            places=2,
        )
        self.assertGreaterEqual(len(result["added_symbols"]), 1)
        self.assertTrue(all(row["current_weight"] == 0 for row in result["optimized_allocations"]))
        json.dumps(result, allow_nan=False)

    def test_construction_default_uses_profile_mandate_minimum(self):
        balanced = construct_portfolio(
            initial_cash_naira=5000,
            risk_profile="balanced",
            mandate_profile="balanced_equity",
            price_file=str(self.price_file),
            signal_file=str(self.signal_file),
        )
        pension = construct_portfolio(
            initial_cash_naira=5000,
            risk_profile="aggressive",
            mandate_profile="pension_equity",
            price_file=str(self.price_file),
            signal_file=str(self.signal_file),
        )

        self.assertEqual(len(balanced["optimized_allocations"]), 10)
        self.assertEqual(len(pension["optimized_allocations"]), 15)

    def test_construction_backfills_candidates_without_enough_history(self):
        short_history_file = self.tmp / "PRICE_LIST_SHORT_HISTORY.csv"
        prices = pd.read_csv(self.price_file)
        short_history_symbols = ["KKK", "LLL", "MMM", "NNN", "OOO", "PPP"]
        recent_dates = set(sorted(prices["TRANS_DATE"].unique())[-20:])
        prices = prices[
            (~prices["SYMBOL"].isin(short_history_symbols))
            | (prices["TRANS_DATE"].isin(recent_dates))
        ]
        prices.to_csv(short_history_file, index=False)

        result = construct_portfolio(
            initial_cash_naira=5000,
            max_stocks=10,
            price_file=str(short_history_file),
            signal_file=str(self.signal_file),
        )

        self.assertEqual(len(result["optimized_allocations"]), 10)
        json.dumps(result, allow_nan=False)

    def test_restrictive_construction_uses_non_sell_names_for_diversification(self):
        mixed_signal_file = self.tmp / "signal_store_mixed.csv"
        signals = pd.read_csv(self.signal_file)
        buy_symbols = {"GGG", "HHH", "III", "JJJ", "KKK"}
        signals.loc[~signals["Symbol"].isin(buy_symbols), "Consensus_Signal"] = "HOLD"
        signals.loc[~signals["Symbol"].isin(buy_symbols), "Consensus_Tier"] = 3
        signals.to_csv(mixed_signal_file, index=False)

        result = construct_portfolio(
            initial_cash_naira=5000,
            max_stocks=10,
            price_file=str(self.price_file),
            signal_file=str(mixed_signal_file),
        )

        self.assertEqual(len(result["optimized_allocations"]), 10)
        self.assertTrue(
            any(row["signal_status"] != "BUY" for row in result["optimized_allocations"])
        )
        json.dumps(result, allow_nan=False)

    def test_pension_mandate_tightens_constraints(self):
        result = optimize_portfolio(
            holdings=[
                {"symbol": symbol, "amount_naira": 1000}
                for symbol in [
                    "AAA", "BBB", "CCC", "DDD", "EEE", "FFF", "GGG", "HHH",
                    "III", "JJJ", "KKK", "LLL", "MMM", "NNN", "OOO",
                ]
            ],
            risk_profile="aggressive",
            mandate_profile="pension_equity",
            allow_new_stocks=False,
            price_file=str(self.price_file),
            signal_file=str(self.signal_file),
        )

        self.assertEqual(result["mandate_summary"]["max_stock_weight"], 0.07)
        self.assertEqual(result["mandate_summary"]["max_sector_weight"], 0.3)
