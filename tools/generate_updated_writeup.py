from __future__ import annotations

import html
import zipfile
from pathlib import Path


OUTPUT = Path("generated_docs/ODUTOLA_PROJECT_UPDATED_CHAPTERS_1_AND_2.docx")


TITLE = "DEVELOPING A MACHINE LEARNING-BASED EQUITY PORTFOLIO CONSTRUCTION SYSTEM FOR FUND MANAGERS IN NIGERIA"


def esc(text: str) -> str:
    return html.escape(text, quote=False)


def run(text: str, bold: bool = False, size: int | None = None) -> str:
    props = []
    if bold:
        props.append("<w:b/>")
    if size:
        props.append(f'<w:sz w:val="{size}"/><w:szCs w:val="{size}"/>')
    rpr = f"<w:rPr>{''.join(props)}</w:rPr>" if props else ""
    space = ' xml:space="preserve"' if text.startswith(" ") or text.endswith(" ") else ""
    return f"<w:r>{rpr}<w:t{space}>{esc(text)}</w:t></w:r>"


def para(
    text: str = "",
    *,
    style: str | None = None,
    align: str | None = None,
    bold: bool = False,
    size: int | None = None,
    spacing_after: int = 120,
) -> str:
    ppr = []
    if style:
        ppr.append(f'<w:pStyle w:val="{style}"/>')
    if align:
        ppr.append(f'<w:jc w:val="{align}"/>')
    ppr.append(f'<w:spacing w:after="{spacing_after}" w:line="360" w:lineRule="auto"/>')
    ppr_xml = f"<w:pPr>{''.join(ppr)}</w:pPr>"
    return f"<w:p>{ppr_xml}{run(text, bold=bold, size=size)}</w:p>"


def bullet(text: str) -> str:
    return (
        '<w:p><w:pPr><w:numPr><w:ilvl w:val="0"/><w:numId w:val="1"/></w:numPr>'
        '<w:spacing w:after="80" w:line="360" w:lineRule="auto"/></w:pPr>'
        f"{run(text)}</w:p>"
    )


def page_break() -> str:
    return '<w:p><w:r><w:br w:type="page"/></w:r></w:p>'


def cell(text: str, width: int = 3000, bold: bool = False) -> str:
    return (
        f'<w:tc><w:tcPr><w:tcW w:w="{width}" w:type="dxa"/></w:tcPr>'
        f'{para(text, bold=bold, spacing_after=0)}</w:tc>'
    )


def table(rows: list[list[str]]) -> str:
    grid = '<w:tblGrid><w:gridCol w:w="900"/><w:gridCol w:w="4200"/><w:gridCol w:w="5200"/></w:tblGrid>'
    props = (
        '<w:tblPr><w:tblStyle w:val="TableGrid"/><w:tblW w:w="0" w:type="auto"/>'
        '<w:tblBorders><w:top w:val="single" w:sz="4" w:space="0" w:color="000000"/>'
        '<w:left w:val="single" w:sz="4" w:space="0" w:color="000000"/>'
        '<w:bottom w:val="single" w:sz="4" w:space="0" w:color="000000"/>'
        '<w:right w:val="single" w:sz="4" w:space="0" w:color="000000"/>'
        '<w:insideH w:val="single" w:sz="4" w:space="0" w:color="000000"/>'
        '<w:insideV w:val="single" w:sz="4" w:space="0" w:color="000000"/></w:tblBorders></w:tblPr>'
    )
    body = []
    for i, row in enumerate(rows):
        body.append("<w:tr>" + "".join(cell(value, bold=(i == 0)) for value in row) + "</w:tr>")
    return f"<w:tbl>{props}{grid}{''.join(body)}</w:tbl>"


def heading(text: str, level: int = 1) -> str:
    style = "Heading1" if level == 1 else "Heading2"
    size = 28 if level == 1 else 24
    return para(text, style=style, bold=True, size=size, spacing_after=160)


def title_page() -> list[str]:
    return [
        para(TITLE, align="center", bold=True, size=28, spacing_after=360),
        para("BY", align="center", bold=True, spacing_after=240),
        para("ODUTOLA PRAISE OLUSHOLA", align="center", bold=True),
        para("22CG031901", align="center", bold=True, spacing_after=360),
        para(
            "A PROJECT SUBMITTED TO THE DEPARTMENT OF COMPUTER AND INFORMATION SCIENCES, "
            "COLLEGE OF SCIENCE AND TECHNOLOGY, COVENANT UNIVERSITY, OTA, OGUN STATE.",
            align="center",
            bold=True,
            spacing_after=360,
        ),
        para(
            "IN PARTIAL FULFILMENT OF THE REQUIREMENTS FOR THE AWARD OF THE BACHELOR OF "
            "SCIENCE (HONOURS) DEGREE IN COMPUTER SCIENCE",
            align="center",
            bold=True,
            spacing_after=360,
        ),
        para("DECEMBER, 2025", align="center", bold=True),
        page_break(),
        heading("CERTIFICATION"),
        para(
            "I hereby certify that this project was carried out by Odutola, Praise Olushola "
            "in the Department of Computer and Information Sciences, College of Science and "
            "Technology, Covenant University, Ota, Ogun State, Nigeria, under my supervision."
        ),
        para("Prof. Olufunke O. Oladipupo                         ___________________________"),
        para("Supervisor                                          Signature and Date"),
        para("Prof. Oni A. Aderonke                              ___________________________"),
        para("Head of Department                                  Signature and Date"),
        page_break(),
        heading("DEDICATION"),
        para(
            "I dedicate this project to God Almighty for His grace, favour and strength "
            "throughout my time at Covenant University. To God alone be all the glory."
        ),
        page_break(),
        heading("ACKNOWLEDGEMENT"),
        para(
            "I appreciate God Almighty for His grace and guidance throughout this academic journey. "
            "I am also grateful to my parents for their prayers, support and encouragement, and to "
            "my supervisor for her guidance during the development of this project."
        ),
        page_break(),
    ]


def front_matter() -> list[str]:
    return [
        heading("TABLE OF CONTENTS"),
        para("CONTENT                                                                                                              PAGES"),
        para("CERTIFICATION                                                                                                          ii"),
        para("DEDICATION                                                                                                             iii"),
        para("ACKNOWLEDGEMENT                                                                                                        iv"),
        para("LIST OF TABLES                                                                                                         vi"),
        para("ABBREVIATIONS                                                                                                          vii"),
        para("CHAPTER ONE: INTRODUCTION                                                                                              1"),
        para("1.1 Background Information                                                                                              1"),
        para("1.2 Statement of the Problem                                                                                            3"),
        para("1.3 Aim and Objectives of the Study                                                                                     4"),
        para("1.4 Methodology                                                                                                         5"),
        para("1.5 Significance of the Study                                                                                           8"),
        para("1.6 Project Outline                                                                                                     9"),
        para("CHAPTER TWO: LITERATURE REVIEW                                                                                        10"),
        para("2.1 Preamble                                                                                                           10"),
        para("2.2 Review of Machine Learning-Based Equity Portfolio Construction                                                     10"),
        para("2.3 Review of Key Concepts                                                                                             12"),
        para("2.4 Theoretical and Technical Foundations                                                                               17"),
        para("2.5 Review of Machine Learning Models Used in the System                                                               21"),
        para("2.6 Review of Related Methods                                                                                          24"),
        para("2.7 Review of Existing Systems                                                                                         27"),
        para("2.8 Summary of the Literature Review                                                                                    30"),
        para("REFERENCES                                                                                                             32"),
        page_break(),
        heading("LIST OF TABLES"),
        para("TABLES        TITLE OF TABLES                                                        PAGES"),
        para("1.1           Objectives-Methodology Mapping Table                                     6"),
        page_break(),
        heading("ABBREVIATIONS"),
        para("API            Application Programming Interface"),
        para("ASR            Annualised Sharpe Ratio"),
        para("CIS            Collective Investment Scheme"),
        para("CVaR           Conditional Value-at-Risk"),
        para("ETF            Exchange Traded Fund"),
        para("FastAPI        Fast Application Programming Interface"),
        para("HRP            Hierarchical Risk Parity"),
        para("LSTM           Long Short-Term Memory"),
        para("ML             Machine Learning"),
        para("MPT            Modern Portfolio Theory"),
        para("MVO            Mean-Variance Optimisation"),
        para("NGX            Nigerian Exchange Group"),
        para("RF             Random Forest"),
        para("SEC            Securities and Exchange Commission"),
        para("UI             User Interface"),
        para("XGBoost        Extreme Gradient Boosting"),
        page_break(),
    ]


def chapter_one() -> list[str]:
    rows = [
        ["S/N", "Objectives", "Methodology"],
        [
            "1",
            "To review literature and process relevant Nigerian equity market data for system development.",
            "Academic literature, official regulatory sources and NGX historical price data were reviewed. "
            "The dataset was cleaned, equity instruments were filtered from non-equity instruments, and "
            "features such as returns, volatility, volume, trade value and technical indicators were prepared.",
        ],
        [
            "2",
            "To develop a machine learning signal module for Nigerian equities.",
            "XGBoost, Random Forest and LSTM models were implemented to generate forward-looking buy, sell "
            "and expected-return signals over a 20-trading-day horizon. Model outputs were quality-gated and "
            "merged into a consensus signal store.",
        ],
        [
            "3",
            "To design and implement a constrained portfolio construction and rebalancing engine.",
            "The optimiser blended historical and ML-implied expected returns, estimated covariance, applied "
            "risk and mandate profiles, and enforced concentration, sector, liquidity, turnover and transaction-cost controls.",
        ],
        [
            "4",
            "To implement a full-stack decision-support system for fund managers.",
            "A FastAPI backend was connected to a React TypeScript frontend. The interface supports portfolio "
            "construction, portfolio optimisation, manager/client workspaces, signal watchlists, compliance reports and exportable results.",
        ],
        [
            "5",
            "To test the system against traditional portfolio references.",
            "The backend and optimiser were tested with automated tests, while portfolio outputs were compared "
            "against current portfolios, equal-weight portfolios and a liquidity-weighted benchmark using return, volatility, Sharpe, Sortino, CVaR and drawdown metrics.",
        ],
    ]
    return [
        heading("CHAPTER ONE"),
        heading("INTRODUCTION"),
        heading("1.1 Background Information", 2),
        para(
            "Portfolio construction is one of the core responsibilities of fund managers because it determines "
            "how client capital is allocated across securities, sectors and risk exposures. Recent portfolio optimisation "
            "studies still treat investment selection as a trade-off between expected return and risk rather "
            "than as a simple search for the highest-returning security. In practice, however, this trade-off is difficult "
            "to manage in emerging markets where price movements may be volatile, liquidity may vary widely across "
            "stocks, and information may be unevenly distributed across market participants (Tran et al., 2025)."
        ),
        para(
            "The Nigerian equity market presents a strong case for a data-driven portfolio construction system. "
            "Fund and portfolio managers operate within a regulated environment in which the Securities and Exchange "
            "Commission registers and supervises fund/portfolio management activities, while pension-related equity "
            "exposure is also shaped by investment rules issued by the National Pension Commission (SEC Nigeria, 2025; "
            "National Pension Commission, 2026). These realities mean that a practical system for Nigerian fund managers "
            "should not only predict attractive stocks; it should also construct portfolios that respect liquidity, "
            "diversification, concentration and mandate constraints."
        ),
        para(
            "Machine learning has become useful in finance because it can model nonlinear relationships that are "
            "often missed by simple linear methods. Recent financial machine learning studies show that ML models can "
            "support investment-signal generation and improve risk-adjusted strategy evaluation when combined with "
            "appropriate backtesting and performance controls (Grudniewicz & Slepaczuk, 2023; Bargos & Romao, 2025). "
            "Similarly, reviews of stock-market forecasting research show that tree-based methods, ensemble learning "
            "and deep learning models are widely used for price, return and direction prediction (Sonkavde et al., "
            "2023; Nandi et al., 2023). For a Nigerian equity portfolio system, these techniques are useful because "
            "they can transform historical market data into forward-looking signals before portfolio weights are selected."
        ),
        para(
            "The system developed in this project is a machine learning-based equity portfolio construction system "
            "for fund managers in Nigeria. It uses historical Nigerian equity price data, filters non-equity instruments, "
            "engineers technical and market features, and generates signals with XGBoost, Random Forest and Long Short-Term "
            "Memory models. The model outputs are merged into a consensus signal store that records expected return, "
            "confidence, model quality and buy/sell consensus for supported equities."
        ),
        para(
            "The prediction layer is connected to a portfolio optimisation engine. The engine supports two practical "
            "workflows: constructing a new portfolio from available cash, and rebalancing an existing portfolio entered "
            "by a fund manager. It combines ML-implied returns with historical returns, estimates annualised risk and "
            "covariance, scores candidate portfolios, and applies constraints such as maximum stock weight, maximum "
            "sector weight, liquidity screen, no-trade band, turnover penalty and transaction cost rate. It also supports "
            "different risk profiles and mandate profiles, including balanced equity, growth equity, income equity and "
            "pension-style equity sleeves."
        ),
        para(
            "The developed application exposes the optimiser through a Python FastAPI backend and a React TypeScript "
            "frontend. The interface allows fund managers to create workspaces, manage clients, enter holdings or initial "
            "cash, choose risk and mandate settings, view signal watchlists, inspect optimised allocations, review sector "
            "exposure, compare strategies through backtesting, and read a compliance report. Therefore, the project "
            "moves beyond a forecasting script by providing an integrated decision-support system for Nigerian equity "
            "portfolio construction."
        ),
        heading("1.2 Statement of the Problem", 2),
        para(
            "Although Nigerian fund managers increasingly operate in a digital financial environment, many portfolio "
            "construction decisions are still supported by fragmented workflows such as spreadsheets, isolated charts, "
            "manual screening and judgement-based stock selection. These tools may be useful for basic analysis, but "
            "they are limited when a manager needs to combine return prediction, risk estimation, liquidity checks, "
            "sector diversification, mandate limits and portfolio rebalancing in one repeatable workflow."
        ),
        para(
            "A major problem is the gap between prediction and allocation. A model may identify equities with positive "
            "expected returns, but a fund manager still needs to decide how much capital to allocate to each stock. "
            "Without an optimisation layer, predictive signals can lead to concentrated or impractical portfolios. "
            "This is especially important because classical mean-variance optimisation is sensitive to estimation error, "
            "and naive diversification can sometimes compete strongly with optimised portfolios out of sample "
            "(Tran et al., 2025; Dutta & Jain, 2023)."
        ),
        para(
            "Another problem is that many investment-support tools are designed for developed markets and do not fully "
            "reflect the operational realities of Nigerian equity management. Nigerian fund managers require controls "
            "for liquidity, sector concentration, client mandates, pension-style restrictions, transaction costs and "
            "portfolio reporting. A system that only predicts stock prices is therefore insufficient for fund management "
            "work because it does not translate predictions into actionable, constraint-aware portfolio decisions."
        ),
        para(
            "There is also a system-integration problem. Forecasting models, portfolio optimisation scripts, performance "
            "metrics and client records often exist separately. This makes it difficult for a fund manager to trace why "
            "a stock was added, reduced, retained or removed from a portfolio. The absence of an integrated interface "
            "also limits explainability, repeatability and auditability."
        ),
        para(
            "This project addresses these issues by developing a machine learning-based equity portfolio construction "
            "system that joins prediction, optimisation, compliance reporting, backtesting and fund-manager workspace "
            "management in a single application tailored to Nigerian equities."
        ),
        heading("1.3 Aim and Objectives of the Study", 2),
        para(
            "The aim of this study is to develop a machine learning-based equity portfolio construction system that "
            "supports Nigerian fund managers in selecting, constructing and rebalancing equity portfolios using "
            "predictive signals, risk metrics and practical investment constraints."
        ),
        para("The objectives of the study are to:"),
        bullet("Review relevant literature and process Nigerian equity market data for system development."),
        bullet("Develop a machine learning signal-generation module for Nigerian equities."),
        bullet("Design and implement a constrained portfolio construction and rebalancing engine."),
        bullet("Implement a full-stack decision-support system for fund managers using a Python backend and React TypeScript frontend."),
        bullet("Test the system and compare portfolio outputs with traditional reference strategies."),
        heading("1.4 Methodology", 2),
        para(
            "The methodology followed a system-development approach consisting of data preparation, machine learning "
            "modelling, signal merging, portfolio optimisation, backend integration, frontend implementation and testing. "
            "The process was designed to match the practical workflow of a Nigerian fund manager: collect equity data, "
            "generate investment signals, construct or rebalance a portfolio, evaluate risk-adjusted performance and "
            "produce a report."
        ),
        para(
            "First, historical Nigerian equity data were prepared from the project price dataset. The data cleaning "
            "process standardised symbols, parsed trading dates, converted price, volume and trade-value columns to "
            "numeric form, removed invalid price records and filtered out non-equity instruments such as bonds, ETFs "
            "and REITs. For each supported equity, the system computed returns, rolling statistics, volatility and "
            "liquidity measures. Technical indicators such as RSI, MACD, Bollinger Bands, exponential moving averages, "
            "simple moving averages and rolling returns were used as predictive features."
        ),
        para(
            "Second, the machine learning layer generated stock-level signals. XGBoost and Random Forest models were "
            "used for structured feature learning, while an LSTM model was used to capture sequential price patterns. "
            "The models were trained with time-aware train-test splits and evaluated using metrics such as R-squared, "
            "directional accuracy and confidence score. The signal-merging module combined model outputs into a "
            "symbol-level signal store containing buy/sell consensus, expected return, confidence, model coverage and "
            "quality-gated votes."
        ),
        para(
            "Third, the portfolio construction engine used the signal store and historical price data to construct "
            "candidate portfolios. Expected returns were estimated by blending historical annualised returns with "
            "machine learning forecasts, while covariance was estimated from daily returns with shrinkage toward the "
            "diagonal to reduce instability. Candidate portfolios were scored using risk-adjusted measures, including "
            "Sharpe ratio, Sortino ratio, CVaR, drawdown, information ratio and turnover penalty. Constraints were "
            "applied for maximum stock weight, maximum sector weight, liquidity score, new-stock budget, transaction "
            "cost and no-trade bands."
        ),
        para(
            "Fourth, the backend and frontend were integrated. The FastAPI backend exposes endpoints for market bootstrap "
            "data, latest prices, signal summaries, watchlists, new portfolio construction, existing portfolio optimisation, "
            "fund-manager management, client management, saved portfolios and optimisation runs. The React TypeScript "
            "frontend consumes these endpoints and presents the system through a fund-manager workspace with tabs for "
            "overview, watchlist, input, dashboard, workspace and data."
        ),
        para(
            "Finally, system testing was carried out using automated tests for the API and optimiser. The optimiser was "
            "tested for construction and rebalancing behaviour, while the API was tested for valid responses and data "
            "contracts. The output of the system was evaluated against current portfolio weights, equal-weight allocation "
            "and a liquidity-weighted benchmark to determine whether the optimised portfolio improved risk-adjusted "
            "performance while respecting investment constraints."
        ),
        para("Table 1.1: Objectives-Methodology Mapping Table", bold=True),
        table(rows),
        heading("1.5 Significance of the Study", 2),
        para(
            "This study is significant to Nigerian fund managers because it provides a practical decision-support system "
            "for constructing and rebalancing equity portfolios. Instead of relying only on manual screening or isolated "
            "forecasts, managers can use the system to view ML-generated signals, compare candidate allocations, inspect "
            "risk contributions, track sector exposure and produce a fund-manager report."
        ),
        para(
            "The study is also significant to investors and clients because it promotes more transparent portfolio "
            "decisions. The system explains whether a stock was added, increased, reduced, kept or exited, and it shows "
            "the relevant signal status, expected return, liquidity score, model votes and mandate compliance. This "
            "can improve communication between fund managers and clients."
        ),
        para(
            "Academically, the study contributes to the application of machine learning in African financial markets. "
            "It demonstrates how XGBoost, Random Forest and LSTM models can be combined with portfolio theory, "
            "risk metrics and software engineering to create a localised system for Nigerian equities. It also provides "
            "a foundation for future studies on portfolio optimisation, explainable financial machine learning and "
            "emerging-market investment decision systems."
        ),
        para(
            "From a technology perspective, the study shows how a machine learning pipeline can be turned into a usable "
            "full-stack application. The developed system connects model outputs, optimisation logic, API services, "
            "frontend interaction and persistent fund-manager records. This makes the work useful not only as a research "
            "artefact but also as a prototype for fintech and asset-management innovation in Nigeria."
        ),
        heading("1.6 Project Outline", 2),
        para(
            "This project is structured into five chapters. Chapter One introduces the research background, problem "
            "statement, aim, objectives, methodology, significance and project outline. Chapter Two reviews literature "
            "on machine learning, portfolio construction, optimisation methods, Nigerian fund management and existing "
            "decision-support systems. Chapter Three presents the system analysis, architecture and design. Chapter Four "
            "discusses implementation, testing and results. Chapter Five concludes the study with findings, limitations "
            "and recommendations for future work."
        ),
        page_break(),
    ]


def chapter_two() -> list[str]:
    return [
        heading("CHAPTER TWO"),
        heading("LITERATURE REVIEW"),
        heading("2.1 Preamble", 2),
        para(
            "This chapter reviews the concepts, theories, models and existing systems relevant to the development of "
            "a machine learning-based equity portfolio construction system for fund managers in Nigeria. The review "
            "focuses on portfolio construction, machine learning in financial prediction, risk and performance metrics, "
            "optimisation methods, regulatory considerations and the gap addressed by the developed system."
        ),
        heading("2.2 Review of Machine Learning-Based Equity Portfolio Construction", 2),
        para(
            "Machine learning-based equity portfolio construction combines two connected tasks. The first task is "
            "prediction: estimating future returns, market direction, volatility or stock attractiveness from historical "
            "and engineered data. The second task is allocation: deciding portfolio weights while managing risk, "
            "diversification and constraints. A useful system must perform both tasks because predicted returns alone "
            "do not determine how much capital should be allocated to each security."
        ),
        para(
            "The theoretical foundation of portfolio construction is the risk-return allocation framework commonly "
            "discussed in recent portfolio optimisation literature. The framework evaluates portfolios using expected "
            "return and risk, making diversification a measurable optimisation problem. However, the quality of the "
            "output depends heavily on the quality of expected returns and covariance estimates. This weakness has "
            "motivated research on improved forecasting, shrinkage covariance estimation and robust portfolio methods "
            "(Tran et al., 2025; Dutta & Jain, 2023)."
        ),
        para(
            "Machine learning contributes to the prediction stage by learning nonlinear relationships between market "
            "features and future returns. Recent studies show that machine learning can support investment prediction, "
            "fund-return forecasting and algorithmic strategy design, while systematic reviews report broad use of "
            "deep learning and machine learning models in financial forecasting (Grudniewicz & Slepaczuk, 2023; "
            "Bargos & Romao, 2025; Sonkavde et al., 2023). In this project, XGBoost, Random Forest "
            "and LSTM models are used because they represent complementary approaches: boosted trees, bagged trees and "
            "sequential neural networks."
        ),
        para(
            "In the developed system, machine learning does not replace portfolio theory. Instead, it supplies expected "
            "return and signal information to an optimiser. The optimiser uses those signals together with historical "
            "risk estimates and mandate constraints. This hybrid approach is appropriate for fund management because "
            "it connects predictive analytics with capital allocation, performance comparison and compliance reporting."
        ),
        heading("2.3 Review of Key Concepts", 2),
        heading("2.3.1 Fund Managers and the Nigerian Equity Market", 2),
        para(
            "Fund managers are professionals or firms responsible for managing pooled or discretionary investment "
            "portfolios on behalf of clients. In Nigeria, fund/portfolio managers are part of the capital-market "
            "operators supervised by the Securities and Exchange Commission. SEC registration requirements emphasise "
            "knowledge of capital-market rules and compliance obligations for sponsored individuals and firms "
            "(SEC Nigeria, 2025)."
        ),
        para(
            "The Nigerian equity market, operated through the Nigerian Exchange, contains companies across sectors "
            "such as banking, consumer goods, industrial goods, oil and gas, telecommunications, insurance, agriculture "
            "and services. An equity portfolio construction system for this market must therefore handle sector "
            "classification, liquidity differences, price history gaps and the exclusion of non-equity instruments. "
            "The developed system addresses this by filtering bonds, ETFs and REITs from the equity universe and "
            "mapping supported stocks to sectors for concentration control."
        ),
        heading("2.3.2 Equity Portfolio Construction", 2),
        para(
            "Equity portfolio construction is the process of selecting stocks and assigning portfolio weights to them. "
            "A fund manager may construct a new portfolio from cash or rebalance an existing portfolio. In both cases, "
            "the manager must decide which stocks to hold, how much to allocate, when to trade and how much turnover is "
            "acceptable. The developed system supports both construction and optimisation modes, allowing the manager "
            "to either deploy an initial cash amount or adjust existing holdings."
        ),
        para(
            "Portfolio construction differs from stock picking because it evaluates the combined behaviour of assets. "
            "A stock with attractive expected return may still receive a low weight if it increases portfolio risk, "
            "violates a concentration limit, has poor liquidity or overlaps heavily with an already dominant sector. "
            "This is why the developed system reports optimised allocations, sector allocation, correlation matrix, "
            "risk contributions and diversification score."
        ),
        heading("2.3.3 Machine Learning Signals", 2),
        para(
            "A machine learning signal is a model-derived indication of whether an asset is expected to perform "
            "positively or negatively over a defined horizon. In this project, the prediction horizon is 20 trading days. "
            "The signal module produces expected return, buy/sell direction, confidence and quality metrics. These "
            "outputs are merged into a consensus signal, reducing dependence on a single model."
        ),
        para(
            "Signal quality is important because financial data are noisy. A model that produces a positive forecast "
            "but weak directional accuracy should not dominate the portfolio. The developed system therefore uses "
            "quality gates based on model performance and confidence before a model vote is counted in the consensus "
            "signal. Where full ML coverage is unavailable, fallback price-momentum signals are generated only under "
            "minimum history and liquidity conditions."
        ),
        heading("2.3.4 Feature Engineering for Financial Time Series", 2),
        para(
            "Feature engineering converts raw price data into variables that may contain predictive information. "
            "Common financial features include lagged prices, lagged returns, rolling volatility, moving averages, "
            "momentum indicators and volume-based liquidity measures. In the developed system, RSI, MACD, Bollinger "
            "Bands, exponential moving averages, simple moving averages, rolling returns, rolling standard deviation, "
            "calendar variables and volatility ratios are used as inputs to the models."
        ),
        para(
            "The use of engineered features is consistent with the wider financial forecasting literature, where models "
            "often require technical, temporal and market microstructure inputs to learn useful patterns (Sonkavde et al., "
            "2023; Shi et al., 2023). For Nigerian equities, feature engineering is especially important because "
            "raw closing prices alone may not adequately capture liquidity, momentum and volatility behaviour."
        ),
        heading("2.3.5 Risk and Return Metrics", 2),
        para(
            "Risk and return metrics allow fund managers to compare portfolios beyond absolute profit. Recent portfolio "
            "optimisation studies continue to evaluate portfolios using risk-adjusted return, volatility, drawdown and "
            "tail-risk measures, especially in emerging-market contexts (Kim & Fabozzi, 2024; Tran et al., 2025). "
            "The Sharpe ratio measures excess return per unit of volatility, the Sortino ratio focuses on downside "
            "volatility, maximum drawdown measures the largest peak-to-trough loss, and Conditional Value-at-Risk "
            "measures expected loss in the tail of the return distribution."
        ),
        para(
            "The developed system reports expected return, volatility, Sharpe ratio, Sortino ratio, CVaR, maximum drawdown, "
            "tracking error, information ratio and annualised realised return. These metrics help the fund manager compare "
            "the current portfolio, optimised portfolio, equal-weight portfolio and benchmark portfolio."
        ),
        heading("2.3.6 Mandate and Compliance Constraints", 2),
        para(
            "Portfolio construction must respect client mandates and regulatory expectations. In Nigeria, official rules "
            "for fund management and pension-fund investment emphasise structured investment governance, reporting and "
            "asset-allocation limits (National Pension Commission, 2026; SEC Nigeria, 2025). Although the developed "
            "system is a prototype and not a regulatory approval tool, it reflects this need by including mandate profiles "
            "and compliance checks."
        ),
        para(
            "The system includes balanced equity, growth equity, income equity and pension-style equity sleeve profiles. "
            "Each profile adjusts limits such as maximum stock weight, maximum sector weight, liquidity threshold, "
            "turnover limit and maximum new-stock budget. The compliance report checks whether the generated allocation "
            "passes, requires review or breaches a rule."
        ),
        heading("2.4 Theoretical and Technical Foundations", 2),
        heading("2.4.1 Modern Portfolio Theory and Mean-Variance Optimisation", 2),
        para(
            "Modern portfolio theory provides the basis for systematic diversification and continues to influence "
            "recent studies on emerging-market portfolio optimisation. Contemporary applications emphasise that "
            "portfolio selection should consider the covariance among assets because the risk of a portfolio is not "
            "simply the average risk of its individual securities. Mean-variance optimisation seeks portfolios that "
            "maximise expected return for a given risk level or minimise risk for a given return level (Tran et al., 2025)."
        ),
        para(
            "The developed optimiser uses this logic but adapts it for practical constraints. Rather than relying only "
            "on a single closed-form mean-variance solution, it samples and scores candidate portfolios, then applies "
            "weight caps, sector caps, liquidity screens, no-trade bands and turnover penalties. This makes the output "
            "more aligned with fund-management practice."
        ),
        heading("2.4.2 Naive Diversification and Benchmarks", 2),
        para(
            "Naive diversification, often represented by the equal-weight portfolio, is an important reference strategy. "
            "Recent emerging-market optimisation research continues to compare optimised portfolios with equal-weighted "
            "and benchmark alternatives because optimisation can be sensitive to estimation error and market stress "
            "(Tran et al., 2025). For this reason, a portfolio construction system should compare its output against "
            "equal-weight and benchmark portfolios rather than assuming that optimisation is automatically superior."
        ),
        para(
            "The developed system includes equal-weight and liquidity-weighted benchmark comparisons. This allows a fund "
            "manager to see whether the optimised portfolio improves risk-adjusted performance relative to simple and "
            "market-like alternatives."
        ),
        heading("2.4.3 Covariance Estimation and Shrinkage", 2),
        para(
            "Covariance estimation is central to portfolio optimisation because it determines how asset risks interact. "
            "Recent portfolio studies show that sample covariance matrices can be unstable in practical allocation tasks "
            "and that shrinkage can improve estimation under noisy or high-dimensional conditions (Dutta & Jain, 2023; "
            "Tran et al., 2025). The developed system applies a simple shrinkage "
            "approach by blending the sample covariance matrix with its diagonal. This reduces excessive reliance on "
            "unstable off-diagonal estimates while retaining useful correlation information."
        ),
        heading("2.4.4 Risk Measures for Portfolio Evaluation", 2),
        para(
            "Traditional volatility treats upside and downside variation equally, but fund managers are often more "
            "concerned with losses. Recent portfolio optimisation work emphasises tail-risk and drawdown-aware measures "
            "for asset-management decisions (Kim & Fabozzi, 2024; Tran et al., 2025). Maximum drawdown is also important because it "
            "shows the severity of loss during a historical period. The developed system includes CVaR and drawdown in "
            "its objective scoring and report outputs."
        ),
        heading("2.4.5 Hierarchical and Risk-Parity Alternatives", 2),
        para(
            "Hierarchical and graph-based risk-allocation methods use clustering, covariance information and network "
            "relationships to build diversified portfolios without relying only on classical quadratic optimisation. "
            "Recent work on minimum-spanning-tree portfolio construction shows how graph-based relationships can support "
            "diversification decisions in stock markets (Berouaga et al., 2023). Although the developed system does not "
            "implement hierarchical risk parity as its main optimiser, these methods are relevant because they show the "
            "importance of diversification, covariance structure and out-of-sample robustness."
        ),
        heading("2.5 Review of Machine Learning Models Used in the System", 2),
        heading("2.5.1 Extreme Gradient Boosting", 2),
        para(
            "XGBoost is a gradient-boosted tree method that has been widely applied in recent stock-market prediction "
            "research. It builds trees sequentially, with each new tree improving errors from previous trees. XGBoost is "
            "useful for financial prediction because it handles nonlinear relationships, mixed feature types and "
            "regularisation. In the developed system, XGBoost is used to estimate future return or direction from "
            "technical and market features (Shi et al., 2023; Sonkavde et al., 2023)."
        ),
        heading("2.5.2 Random Forest", 2),
        para(
            "Random Forest is an ensemble learning method that builds many decision trees on bootstrapped samples and "
            "averages their predictions. Recent financial forecasting studies continue to use Random Forest because it "
            "is relatively robust to noisy data and can reduce variance compared with a single decision tree. In this "
            "project, Random Forest provides a stable second model whose output can be compared and merged with XGBoost "
            "and LSTM outputs (Bargos & Romao, 2025; Sonkavde et al., 2023)."
        ),
        heading("2.5.3 Long Short-Term Memory Networks", 2),
        para(
            "Long Short-Term Memory networks are recurrent neural networks designed for sequential data and are widely "
            "used in recent stock-market prediction research. LSTM models are relevant to stock-market prediction because "
            "financial prices are sequential time-series data. The developed system uses an LSTM model to capture temporal "
            "patterns from a lookback window of historical features (Shi et al., 2023; Sonkavde et al., 2023)."
        ),
        heading("2.5.4 Ensemble and Consensus Signal Design", 2),
        para(
            "A consensus signal combines the strengths of multiple models. Ensemble approaches are common in financial "
            "prediction because different algorithms may capture different patterns in the same data (Sonkavde et al., "
            "2023). The developed system merges XGBoost, Random Forest and LSTM outputs into a consensus signal. A stock "
            "may receive a BUY, SELL or CONFLICT status depending on model agreement and quality-gated votes."
        ),
        para(
            "This design improves practical reliability. Instead of allowing one weak model to drive allocation, the "
            "optimiser receives signal status, expected return, confidence, model votes and consensus tier. The frontend "
            "then shows these details to the fund manager, improving transparency."
        ),
        heading("2.6 Review of Related Methods", 2),
        heading("2.6.1 Machine Learning for Return and Direction Prediction", 2),
        para(
            "Recent financial forecasting research shows that machine learning models can be used for return prediction, "
            "direction classification and volatility estimation. Grudniewicz and Slepaczuk (2023), Sonkavde et al. (2023) "
            "and Bargos and Romao (2025) show the relevance of ML models in investment-signal generation, stock-market "
            "forecasting and equity-fund return prediction. These studies support the use of ML models as an input to "
            "investment decision systems."
        ),
        heading("2.6.2 Hybrid Prediction and Optimisation Frameworks", 2),
        para(
            "Hybrid frameworks combine prediction models with portfolio optimisation. This is the approach followed in "
            "the developed system. The prediction layer estimates signals and expected returns, while the optimiser "
            "converts those signals into practical portfolio weights. This separation is important because forecasting "
            "and allocation are different problems: a model may identify attractive stocks, but the optimiser must "
            "still manage risk, turnover and constraints."
        ),
        heading("2.6.3 Liquidity-Aware Portfolio Construction", 2),
        para(
            "Liquidity is important in emerging markets because some stocks may have limited trading activity. A fund "
            "manager cannot assume that every predicted opportunity can be traded at scale. The developed system computes "
            "20-day average volume, average trade value and a liquidity score. New-stock candidates must pass a liquidity "
            "threshold before they can be added to a portfolio."
        ),
        heading("2.6.4 Backtesting and Strategy Comparison", 2),
        para(
            "Backtesting allows a system to estimate how a portfolio strategy would have behaved historically. It does "
            "not guarantee future performance, but it provides useful evidence for comparing strategies. The developed "
            "system includes a backtest summary across current, optimised, equal-weight and benchmark portfolios. Metrics "
            "include cumulative return, annualised return, annualised volatility, Sharpe ratio and maximum drawdown."
        ),
        heading("2.6.5 Application in Nigerian and African Markets", 2),
        para(
            "Research on machine learning in African equity markets is smaller than in developed markets, but it is "
            "growing. Uzoaga et al. (2025) examine machine learning and time-series techniques for Nigerian stock-price "
            "prediction, showing that local market data can be modelled with modern predictive methods. This supports "
            "the motivation for a Nigerian-focused system rather than relying only on foreign-market tools."
        ),
        heading("2.7 Review of Existing Systems", 2),
        heading("2.7.1 Forecasting-Only Systems", 2),
        para(
            "Some systems focus mainly on stock prediction. They may output expected price, expected return or buy/sell "
            "direction, but they do not determine portfolio weights. Such systems are useful for research, but they are "
            "incomplete for fund managers because they do not address diversification, liquidity, mandate limits or "
            "client reporting. The developed system closes this gap by connecting prediction to portfolio construction."
        ),
        heading("2.7.2 Portfolio Optimisation-Only Systems", 2),
        para(
            "Other systems focus mainly on optimisation, often requiring the user to manually supply expected returns "
            "and covariance estimates. These tools may be mathematically strong, but their usefulness depends on the "
            "quality of the inputs. The developed system improves this by generating expected-return inputs from a "
            "machine learning signal pipeline and by comparing results with benchmark strategies."
        ),
        heading("2.7.3 Robo-Advisory and Retail Portfolio Tools", 2),
        para(
            "Robo-advisory systems often automate portfolio allocation for retail investors using risk questionnaires "
            "and model portfolios. However, many such tools are designed around broad asset classes, mutual funds or ETFs. "
            "The developed system is different because it focuses on Nigerian listed equities and the workflow of a fund "
            "manager who handles client portfolios, mandate profiles, existing holdings and reporting."
        ),
        heading("2.7.4 Institutional Portfolio Analytics Platforms", 2),
        para(
            "Institutional analytics platforms may provide risk models, optimisation, compliance and reporting, but they "
            "are often expensive, complex and built for global markets. Local adoption may be limited by cost, data access "
            "and mismatch with Nigerian market realities. The developed system is a local prototype that uses Nigerian "
            "equity data, NGN portfolio values and Nigerian fund-management use cases."
        ),
        heading("2.7.5 Gap in Existing Systems", 2),
        para(
            "The literature and system review show a clear gap: there is a need for an integrated Nigerian equity "
            "portfolio construction system that combines ML-based prediction, signal consensus, liquidity screening, "
            "constraint-aware optimisation, backtesting, compliance reporting and a fund-manager workspace. The system "
            "developed in this project addresses this gap by implementing these components in one full-stack application."
        ),
        heading("2.8 Summary of the Literature Review", 2),
        para(
            "This chapter reviewed the main ideas behind machine learning-based equity portfolio construction. Modern "
            "Portfolio Theory provides the foundation for risk-return allocation, while later research highlights the "
            "problems of estimation error, covariance instability and out-of-sample performance. Machine learning models "
            "such as XGBoost, Random Forest and LSTM provide a way to generate forward-looking stock signals, but these "
            "signals must be connected to practical portfolio constraints before they can support fund-management decisions."
        ),
        para(
            "The review also showed that Nigerian fund managers operate within a regulated and liquidity-sensitive "
            "environment. Therefore, the developed system needed to go beyond prediction by including mandate profiles, "
            "liquidity filters, concentration limits, turnover controls, transaction-cost awareness, backtesting and "
            "compliance reporting. These features directly informed the system developed in this project."
        ),
        page_break(),
    ]


def references() -> list[str]:
    refs = [
        "Akinde, M., Olapeju, O., Olaiju, O., Ogunseye, T., Emmanuel, A., Olagoke-Salami, S., Oduwole, F., Olapeju, I., Ibikunle, D., & Aladelusi, K. (2025). Prediction of Green Sukuk investment interest drivers in Nigeria using machine learning models. Journal of Risk and Financial Management, 18(2), 89. https://doi.org/10.3390/jrfm18020089",
        "Bargos, F. F., & Romao, E. C. (2025). Enhanced forecasting of equity fund returns using machine learning. Mathematical and Computational Applications, 30(1), 9. https://doi.org/10.3390/mca30010009",
        "Berouaga, Y., El Msiyah, C., & Madkour, J. (2023). Portfolio optimization using minimum spanning tree model in the Moroccan stock exchange market. International Journal of Financial Studies, 11(2), 53. https://doi.org/10.3390/ijfs11020053",
        "Dutta, S., & Jain, S. (2023). Precision versus shrinkage: A comparative analysis of covariance estimation methods for portfolio allocation. arXiv. https://arxiv.org/abs/2305.11298",
        "Grudniewicz, J., & Slepaczuk, R. (2023). Application of machine learning in algorithmic investment strategies on global stock markets. Research in International Business and Finance, 66, 102052. https://doi.org/10.1016/j.ribaf.2023.102052",
        "Kim, Y. S., & Fabozzi, F. J. (2024). Portfolio optimization with relative tail risk. Annals of Operations Research, 341, 1023-1055. https://doi.org/10.1007/s10479-024-06204-0",
        "Nandi, B., Jana, S., & Das, K. P. (2023). Machine learning-based approaches for financial market prediction: A comprehensive review. Journal of AppliedMath, 1(2.1), 134. https://doi.org/10.59400/jam.v1i2.134",
        "National Pension Commission. (2026). Addendum to the revised regulations on investment of pension fund assets. https://www.pencom.gov.ng/wp-content/uploads/2026/03/ADDENDUM-TO-THE-REVISED-REGULATIONS-ON-INVESTMENT-OF-PENSION-FUND-ASSETS.pdf",
        "Owoade, S. J., Uzoka, A., Akerele, J. I., & Ojukwu, P. U. (2024). Enhancing financial portfolio management with predictive analytics and scalable data modeling techniques. International Journal of Scholarly Research and Reviews, 5(2), 089-102. https://doi.org/10.56781/ijsrr.2024.5.2.0050",
        "SEC Nigeria. (2025). Rules and regulations. https://home.sec.gov.ng/our-mandate/regulation/rules-and-regulations/",
        "Shi, Z., Hu, Y., Mo, G., & Wu, J. (2023). XGBoost and CNN-LSTM hybrid model with attention-based stock prediction. 2023 IEEE International Conference on Electronics, Technology and Computer Intelligence. https://doi.org/10.1109/ICETCI57876.2023.10176988",
        "Sonkavde, G., Dharrao, D. S., Bongale, A. M., Deokate, S. T., Doreswamy, D., & Bhat, S. K. (2023). Forecasting stock market prices using machine learning and deep learning models: A systematic review, performance analysis and discussion of implications. International Journal of Financial Studies, 11(3), 94. https://doi.org/10.3390/ijfs11030094",
        "Tran, M., Nguyen, N. M., & Tran, T. A. (2025). Enhancing portfolio optimization in emerging markets: A cross-validation multi-target shrinkage approach. Results in Control and Optimization, 21, 100611. https://doi.org/10.1016/j.rico.2025.100611",
        "Uzoaga, G. A., Adenomon, M. O., Nweze, N. O., & Maijama, B. (2025). Modelling and predicting stock prices of Nigerian stock exchange using some machine learning techniques and time series model. Science World Journal, 20(2), 510-515. https://doi.org/10.4314/swj.v20i2.9",
    ]
    out = [heading("REFERENCES")]
    out.extend(para(ref) for ref in refs)
    return out


def styles_xml() -> str:
    return """<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<w:styles xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main">
  <w:style w:type="paragraph" w:default="1" w:styleId="Normal">
    <w:name w:val="Normal"/>
    <w:qFormat/>
    <w:pPr><w:spacing w:after="120" w:line="360" w:lineRule="auto"/></w:pPr>
    <w:rPr><w:rFonts w:ascii="Times New Roman" w:hAnsi="Times New Roman"/><w:sz w:val="24"/></w:rPr>
  </w:style>
  <w:style w:type="paragraph" w:styleId="Heading1">
    <w:name w:val="heading 1"/><w:basedOn w:val="Normal"/><w:next w:val="Normal"/><w:qFormat/>
    <w:rPr><w:b/><w:rFonts w:ascii="Times New Roman" w:hAnsi="Times New Roman"/><w:sz w:val="28"/></w:rPr>
  </w:style>
  <w:style w:type="paragraph" w:styleId="Heading2">
    <w:name w:val="heading 2"/><w:basedOn w:val="Normal"/><w:next w:val="Normal"/><w:qFormat/>
    <w:rPr><w:b/><w:rFonts w:ascii="Times New Roman" w:hAnsi="Times New Roman"/><w:sz w:val="24"/></w:rPr>
  </w:style>
  <w:style w:type="table" w:styleId="TableGrid">
    <w:name w:val="Table Grid"/><w:basedOn w:val="TableNormal"/><w:uiPriority w:val="59"/><w:qFormat/>
    <w:tblPr>
      <w:tblBorders>
        <w:top w:val="single" w:sz="4" w:space="0" w:color="000000"/>
        <w:left w:val="single" w:sz="4" w:space="0" w:color="000000"/>
        <w:bottom w:val="single" w:sz="4" w:space="0" w:color="000000"/>
        <w:right w:val="single" w:sz="4" w:space="0" w:color="000000"/>
        <w:insideH w:val="single" w:sz="4" w:space="0" w:color="000000"/>
        <w:insideV w:val="single" w:sz="4" w:space="0" w:color="000000"/>
      </w:tblBorders>
    </w:tblPr>
  </w:style>
</w:styles>"""


def numbering_xml() -> str:
    return """<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<w:numbering xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main">
  <w:abstractNum w:abstractNumId="0">
    <w:multiLevelType w:val="hybridMultilevel"/>
    <w:lvl w:ilvl="0">
      <w:start w:val="1"/>
      <w:numFmt w:val="bullet"/>
      <w:lvlText w:val="•"/>
      <w:lvlJc w:val="left"/>
      <w:pPr><w:ind w:left="720" w:hanging="360"/></w:pPr>
    </w:lvl>
  </w:abstractNum>
  <w:num w:numId="1"><w:abstractNumId w:val="0"/></w:num>
</w:numbering>"""


def document_xml(paragraphs: list[str]) -> str:
    body = "".join(paragraphs)
    sect = (
        '<w:sectPr><w:pgSz w:w="11906" w:h="16838"/>'
        '<w:pgMar w:top="1440" w:right="1440" w:bottom="1440" w:left="1440" '
        'w:header="708" w:footer="708" w:gutter="0"/>'
        '<w:cols w:space="708"/><w:docGrid w:linePitch="360"/></w:sectPr>'
    )
    return f"""<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<w:document xmlns:wpc="http://schemas.microsoft.com/office/word/2010/wordprocessingCanvas"
 xmlns:mc="http://schemas.openxmlformats.org/markup-compatibility/2006"
 xmlns:o="urn:schemas-microsoft-com:office:office"
 xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/relationships"
 xmlns:m="http://schemas.openxmlformats.org/officeDocument/2006/math"
 xmlns:v="urn:schemas-microsoft-com:vml"
 xmlns:wp14="http://schemas.microsoft.com/office/word/2010/wordprocessingDrawing"
 xmlns:wp="http://schemas.openxmlformats.org/drawingml/2006/wordprocessingDrawing"
 xmlns:w10="urn:schemas-microsoft-com:office:word"
 xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main"
 xmlns:w14="http://schemas.microsoft.com/office/word/2010/wordml"
 xmlns:wpg="http://schemas.microsoft.com/office/word/2010/wordprocessingGroup"
 xmlns:wpi="http://schemas.microsoft.com/office/word/2010/wordprocessingInk"
 xmlns:wne="http://schemas.microsoft.com/office/word/2006/wordml"
 xmlns:wps="http://schemas.microsoft.com/office/word/2010/wordprocessingShape"
 mc:Ignorable="w14 wp14"><w:body>{body}{sect}</w:body></w:document>"""


def write_docx() -> None:
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    paragraphs = []
    paragraphs.extend(title_page())
    paragraphs.extend(front_matter())
    paragraphs.extend(chapter_one())
    paragraphs.extend(chapter_two())
    paragraphs.extend(references())

    content_types = """<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types">
  <Default Extension="rels" ContentType="application/vnd.openxmlformats-package.relationships+xml"/>
  <Default Extension="xml" ContentType="application/xml"/>
  <Override PartName="/word/document.xml" ContentType="application/vnd.openxmlformats-officedocument.wordprocessingml.document.main+xml"/>
  <Override PartName="/word/styles.xml" ContentType="application/vnd.openxmlformats-officedocument.wordprocessingml.styles+xml"/>
  <Override PartName="/word/numbering.xml" ContentType="application/vnd.openxmlformats-officedocument.wordprocessingml.numbering+xml"/>
</Types>"""
    rels = """<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">
  <Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/officeDocument" Target="word/document.xml"/>
</Relationships>"""
    doc_rels = """<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">
  <Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/styles" Target="styles.xml"/>
  <Relationship Id="rId2" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/numbering" Target="numbering.xml"/>
</Relationships>"""

    with zipfile.ZipFile(OUTPUT, "w", zipfile.ZIP_DEFLATED) as zf:
        zf.writestr("[Content_Types].xml", content_types)
        zf.writestr("_rels/.rels", rels)
        zf.writestr("word/_rels/document.xml.rels", doc_rels)
        zf.writestr("word/document.xml", document_xml(paragraphs))
        zf.writestr("word/styles.xml", styles_xml())
        zf.writestr("word/numbering.xml", numbering_xml())


if __name__ == "__main__":
    write_docx()
    print(OUTPUT.resolve())
