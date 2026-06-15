from __future__ import annotations

import html
import zipfile
from pathlib import Path


OUTPUT = Path("generated_docs/ODUTOLA_PROJECT_CHAPTERS_3_AND_4.docx")


def esc(text: str) -> str:
    return html.escape(text, quote=False)


def run(text: str, *, bold: bool = False, size: int | None = None) -> str:
    props = []
    if bold:
        props.append("<w:b/>")
    if size:
        props.append(f'<w:sz w:val="{size}"/><w:szCs w:val="{size}"/>')
    props.append('<w:rFonts w:ascii="Times New Roman" w:hAnsi="Times New Roman"/>')
    rpr = f"<w:rPr>{''.join(props)}</w:rPr>"
    space = ' xml:space="preserve"' if text.startswith(" ") or text.endswith(" ") else ""
    return f"<w:r>{rpr}<w:t{space}>{esc(text)}</w:t></w:r>"


def para(
    text: str = "",
    *,
    style: str | None = None,
    align: str | None = None,
    bold: bool = False,
    size: int | None = None,
    first_line: bool = False,
    spacing_before: int = 0,
    spacing_after: int = 120,
) -> str:
    ppr = []
    if style:
        ppr.append(f'<w:pStyle w:val="{style}"/>')
    if align:
        ppr.append(f'<w:jc w:val="{align}"/>')
    else:
        ppr.append('<w:jc w:val="both"/>')
    if first_line:
        ppr.append('<w:ind w:firstLine="720"/>')
    ppr.append(
        f'<w:spacing w:before="{spacing_before}" w:after="{spacing_after}" '
        'w:line="360" w:lineRule="auto"/>'
    )
    return f"<w:p><w:pPr>{''.join(ppr)}</w:pPr>{run(text, bold=bold, size=size)}</w:p>"


def heading(text: str, level: int = 1, *, center: bool = False) -> str:
    return para(
        text,
        style="Heading1" if level == 1 else "Heading2",
        align="center" if center else None,
        bold=True,
        size=28 if level == 1 else 24,
        spacing_before=240 if level != 1 else 120,
        spacing_after=160,
    )


def bullet(text: str) -> str:
    return (
        '<w:p><w:pPr><w:numPr><w:ilvl w:val="0"/><w:numId w:val="1"/></w:numPr>'
        '<w:jc w:val="both"/><w:spacing w:after="80" w:line="360" w:lineRule="auto"/>'
        '</w:pPr>'
        f"{run(text)}</w:p>"
    )


def page_break() -> str:
    return '<w:p><w:r><w:br w:type="page"/></w:r></w:p>'


def cell(text: str, width: int = 3000, bold: bool = False) -> str:
    return (
        f'<w:tc><w:tcPr><w:tcW w:w="{width}" w:type="dxa"/>'
        '<w:tcMar><w:top w:w="80" w:type="dxa"/><w:left w:w="120" w:type="dxa"/>'
        '<w:bottom w:w="80" w:type="dxa"/><w:right w:w="120" w:type="dxa"/></w:tcMar>'
        '</w:tcPr>'
        f'{para(text, bold=bold, spacing_after=0)}'
        '</w:tc>'
    )


def table(rows: list[list[str]], widths: list[int] | None = None) -> str:
    widths = widths or [900, 3400, 5200]
    grid = "<w:tblGrid>" + "".join(f'<w:gridCol w:w="{width}"/>' for width in widths) + "</w:tblGrid>"
    props = (
        '<w:tblPr><w:tblStyle w:val="TableGrid"/><w:tblW w:w="0" w:type="auto"/>'
        '<w:tblBorders><w:top w:val="single" w:sz="4" w:space="0" w:color="000000"/>'
        '<w:left w:val="single" w:sz="4" w:space="0" w:color="000000"/>'
        '<w:bottom w:val="single" w:sz="4" w:space="0" w:color="000000"/>'
        '<w:right w:val="single" w:sz="4" w:space="0" w:color="000000"/>'
        '<w:insideH w:val="single" w:sz="4" w:space="0" w:color="000000"/>'
        '<w:insideV w:val="single" w:sz="4" w:space="0" w:color="000000"/>'
        '</w:tblBorders></w:tblPr>'
    )
    body = []
    for index, row in enumerate(rows):
        body.append(
            "<w:tr>"
            + "".join(cell(value, widths[min(i, len(widths) - 1)], bold=(index == 0)) for i, value in enumerate(row))
            + "</w:tr>"
        )
    return f"<w:tbl>{props}{grid}{''.join(body)}</w:tbl>"


def chapter_three() -> list[str]:
    objective_rows = [
        ["S/N", "Chapter One Objective", "Chapter Three Design Response"],
        [
            "1",
            "Review literature and process Nigerian equity market data for system development.",
            "The design includes a data preparation pipeline that cleans NGX price data, filters non-equity instruments and prepares return, volatility, liquidity and technical-indicator features.",
        ],
        [
            "2",
            "Develop a machine learning signal-generation module for Nigerian equities.",
            "The design uses XGBoost, Random Forest and LSTM models over a 20-trading-day horizon, then merges model outputs into a quality-gated consensus signal store.",
        ],
        [
            "3",
            "Design and implement a constrained portfolio construction and rebalancing engine.",
            "The design blends historical and ML-implied expected returns, estimates covariance with shrinkage and applies stock, sector, liquidity, turnover, transaction-cost and mandate constraints.",
        ],
        [
            "4",
            "Implement a full-stack decision-support system for fund managers.",
            "The design connects a FastAPI backend to a React TypeScript frontend with watchlists, portfolio input, dashboards, compliance reports and manager/client workspaces.",
        ],
        [
            "5",
            "Test the system and compare portfolio outputs with traditional reference strategies.",
            "The design includes automated API/optimiser tests and evaluation against current, equal-weight and liquidity-weighted benchmark portfolios using risk-adjusted performance metrics.",
        ],
    ]
    workflow_rows = [
        ["Stage", "Activity", "Output"],
        ["1", "Collect and inspect NGX historical price data.", "Raw market records available for processing."],
        ["2", "Clean price records and remove non-equity instruments.", "Filtered Nigerian equity universe."],
        ["3", "Engineer technical, return, volatility and calendar features.", "Model-ready feature set."],
        ["4", "Train XGBoost, Random Forest and LSTM signal models.", "Per-symbol predicted return and BUY/SELL signals."],
        ["5", "Merge model outputs and apply signal-quality gates.", "Consensus signal store."],
        ["6", "Construct or rebalance portfolio with risk and mandate constraints.", "Recommended allocations and trade actions."],
        ["7", "Compare outputs with current, equal-weight and liquidity-weighted benchmark portfolios.", "Risk-adjusted performance and backtest summary."],
        ["8", "Present results through the web application.", "Dashboard, compliance report and fund-manager workspace."],
    ]
    data_rows = [
        ["Data item", "Description", "Purpose"],
        ["SYMBOL", "Ticker code of each market instrument.", "Identifies stocks across price, signal and allocation tables."],
        ["TRANS_DATE", "Date attached to each price observation.", "Orders time-series data and supports chronological train-test splitting."],
        ["CLOSE_PRICE", "Daily closing price.", "Used for returns, latest valuation, model targets and portfolio value."],
        ["VOLUME", "Daily units traded.", "Supports liquidity analysis."],
        ["TRADE_VALUE", "Naira value of daily trading activity.", "Ranks and screens liquid candidates for portfolio inclusion."],
        ["Consensus_Signal", "Final BUY, SELL, HOLD or CONFLICT signal.", "Guides candidate selection and signal-aware expected returns."],
        ["Avg_Return", "Average expected return from qualified model outputs or fallback momentum.", "Blended with historical return in the optimiser."],
    ]
    split_rows = [
        ["Dataset", "Use", "Implementation"],
        ["Training set", "Fit the machine learning models.", "First 80% of each usable symbol history in chronological order."],
        ["Test set", "Evaluate out-of-sample direction and return estimates.", "Last 20% of each usable symbol history."],
        ["Optimisation history", "Estimate covariance, volatility, drawdown and backtest metrics.", "Available historical returns after filtering symbols with adequate history."],
    ]
    model_rows = [
        ["Model", "Role in the system", "Main output"],
        ["XGBoost", "Boosted tree model for nonlinear price-signal relationships.", "Predicted return, BUY/SELL signal, confidence and direction metrics."],
        ["Random Forest", "Bagging-based model used as a stable comparator on noisy market features.", "Predicted return, BUY/SELL signal, confidence and direction metrics."],
        ["LSTM", "Sequential neural model for learning temporal price patterns.", "Predicted return and directional signal from lookback windows."],
        ["Fallback momentum", "Broadens signal coverage when full model outputs are not available.", "BUY, SELL or HOLD signal from 20-, 60- and 120-day returns."],
    ]
    constraint_rows = [
        ["Profile", "Maximum stock weight", "Maximum sector weight", "Design interpretation"],
        ["Conservative", "8%", "35%", "Lower concentration, lower turnover and stricter downside control."],
        ["Balanced", "10%", "40%", "Middle position between diversification and signal strength."],
        ["Aggressive", "15%", "48%", "More room for high-conviction signal-led positions."],
        ["Pension-style mandate", "7%", "30%", "Strict equity sleeve with tighter concentration control."],
    ]
    architecture_rows = [
        ["Layer", "Technology", "Function"],
        ["Presentation layer", "React, TypeScript and Vite", "Provides overview, watchlist, portfolio input, dashboard and workspace views with authentication-controlled navigation."],
        ["API layer", "FastAPI and Pydantic", "Receives requests, validates payloads, authenticates fund managers and returns structured JSON responses."],
        ["Service layer", "Python modules", "Loads snapshots, stores workspaces, runs optimisation, records trade outcomes and prepares reports."],
        ["Model layer", "XGBoost, Random Forest, LSTM", "Generates expected-return and directional signals."],
        ["Data layer", "CSV, JSON files and SQLite", "Stores market prices, signal outputs, saved managers with hashed credentials, consumers, portfolios, runs and per-symbol trade outcome records."],
    ]
    uml_rows = [
        ["Diagram", "Main elements", "Description"],
        ["Use case diagram", "Fund manager, portfolio construction, optimisation, signal watchlist, saved workspace", "Shows the fund manager interacting with the system to enter holdings, run optimisation and review results."],
        ["Activity diagram", "Input validation, data loading, signal screening, optimisation, compliance checks, result display", "Shows the workflow from user input to final allocation report."],
        ["Sequence diagram", "React frontend, FastAPI backend, optimiser, price data, signal store, portfolio store", "Shows how requests move between the interface, API and computation modules."],
        ["Data model", "Price records, signal records, manager records, consumer records, saved portfolios", "Shows how project data is separated for maintainability and traceability."],
    ]

    return [
        heading("CHAPTER THREE", center=True),
        heading("SYSTEM ANALYSIS AND DESIGN", center=True),
        heading("3.1 Preamble", 2),
        para(
            "This chapter describes the analysis, methodology and design of the machine learning-based equity portfolio construction system for fund managers in Nigeria. In line with the structure of the software-system template, it discusses the system requirements, development approach, model design, data preparation, portfolio optimisation design, web application design and modelling diagrams. In line with the approved report guide, it also explains the research design, method of data collection, method of data analysis and ethical considerations used in the project.",
            first_line=True,
        ),
        para(
            "The chapter is also aligned with the aim and objectives stated in Chapter One. The system is designed as a decision-support artefact that takes Nigerian equity data, generates machine learning signals, converts those signals into constraint-aware portfolio weights, exposes the result through a full-stack application and evaluates the output against traditional reference strategies.",
            first_line=True,
        ),
        para("Table 3.1: Alignment of Chapter One Objectives with System Design", align="center", bold=True),
        table(objective_rows, [800, 3600, 5100]),
        para("Figure 3.1: Project Workflow", align="center", bold=True),
        table(workflow_rows, [900, 4300, 4300]),
        heading("3.2 Requirement Analysis", 2),
        para(
            "Requirement analysis, also known as requirements engineering, involves the collection, examination and documentation of the features and constraints of a software system. In this project, the requirements describe what the portfolio construction system should do, the type of data it should process, the users it should support and the conditions under which it should operate.",
            first_line=True,
        ),
        heading("3.2.1 Functional Requirements", 2),
        para("The functional requirements of the system are as follows:", first_line=True),
        para("The system should allow fund managers to view supported Nigerian equity symbols."),
        para("The system should allow users to construct a new portfolio from available cash."),
        para("The system should allow users to optimise an existing portfolio by entering current holdings."),
        para("The system should support conservative, balanced and aggressive risk profiles."),
        para("The system should support balanced equity, growth equity, income equity and pension-style equity mandates."),
        para("The system should load latest prices and convert entered quantities into naira portfolio values."),
        para("The system should generate or consume machine learning signals from XGBoost, Random Forest and LSTM models."),
        para("The system should recommend stocks to keep, increase, reduce, add or exit."),
        para("The system should display expected return, volatility, Sharpe ratio, Sortino ratio, CVaR, maximum drawdown, risk contribution and diversification metrics."),
        para("The system should allow fund managers to create workspaces, add consumers, save portfolios and rerun optimisation on saved portfolios."),
        para("The system should require fund managers to register with an email address and password and authenticate before accessing client management, portfolio input and dashboard interfaces."),
        para("The system should restrict the Clients, Input and Dashboard navigation tabs so that they appear only after a fund manager has successfully logged in."),
        para("The system should support batch optimisation, allowing a fund manager to select multiple client portfolios and run optimisation concurrently in a single operation."),
        para("The system should record the trade-level output of every optimisation run, including expected returns and entry prices, so that realised returns and prediction errors can be computed when outcome prices are available."),
        heading("3.2.2 Non-Functional Requirements", 2),
        para("The non-functional requirements of the system are as follows:", first_line=True),
        para("The system should provide portfolio results within a reasonable time to support timely investment decision-making."),
        para("The system should be available through a regular web browser without requiring users to install specialist software."),
        para("The system should validate user inputs and return descriptive messages for unsupported symbols, empty holdings and infeasible mandates."),
        para("The system should maintain a modular structure so that the model, optimisation engine, backend and frontend can be improved independently."),
        para("The system should preserve data integrity by accepting portfolio and workspace records only through validated API contracts."),
        para("The system should be usable by fund managers and investment analysts without requiring them to write programming code."),
        para("The system should store fund manager passwords in a derived hash form using a standard key-derivation function so that plaintext passwords are never written to disk."),
        heading("3.3 Research Design", 2),
        para(
            "The project adopts an applied design science research approach, which is consistent with the methodology introduced in Chapter One. The focus is not only to study portfolio construction methods but also to build and evaluate a working artefact that solves a practical investment decision-support problem. The artefact developed in this study is a full-stack portfolio construction system that combines Nigerian equity market data, machine learning signals, constrained optimisation, backtesting, compliance reporting and a browser-based dashboard.",
            first_line=True,
        ),
        heading("3.4 Population of the Study", 2),
        para(
            "The population of the study consists of Nigerian listed equities and the target professional users who make portfolio allocation decisions. From the data perspective, the available market dataset contains 344,996 historical price records and 526 raw symbols. After filtering non-equity instruments such as bonds, ETFs and REITs, 253 equity symbols remain available for portfolio construction. From the user perspective, the system is designed for fund managers, investment analysts and portfolio officers who manage Nigerian equity portfolios.",
            first_line=True,
        ),
        heading("3.5 Sample and Sampling Technique", 2),
        para(
            "A purposive sampling technique was used for the market universe because only instruments relevant to Nigerian listed equity portfolio construction were retained. Instruments classified as bonds, exchange-traded funds, real estate investment trusts and other non-common-equity securities were removed. For model training, each eligible symbol was considered individually, and only symbols with sufficient usable historical observations were used for machine learning signal generation. For system evaluation, a sample portfolio of major Nigerian equities was used to test optimisation output under a balanced risk profile and balanced equity mandate.",
            first_line=True,
        ),
        heading("3.6 Instrument for Data Collection", 2),
        para(
            "The primary instrument for data collection was the historical market price file, PRICE_LIST.csv. It contains symbol identifiers, transaction dates, previous close, open price, high price, low price, close price, change, trades, volume and trade value. The system also uses signal_store.csv as an internally generated instrument that stores the merged output of the machine learning signal pipeline.",
            first_line=True,
        ),
        para("Table 3.2: Main Data Fields Used in the System", align="center", bold=True),
        table(data_rows, [1900, 3600, 4200]),
        heading("3.7 Method of Data Collection", 2),
        para(
            "The market records were collected into the project dataset and stored as a comma-separated value file. The data was then loaded into the Python preprocessing and optimisation modules. The project did not require direct collection of personal user data for model training. Instead, the application works with market records and user-entered portfolio holdings during system use.",
            first_line=True,
        ),
        heading("3.8 Method of Data Analysis", 2),
        para(
            "The data analysis method follows the methodology described in Chapter One. It combines descriptive market analysis, machine learning signal generation, portfolio optimisation, benchmark comparison and software testing. Descriptive analysis is used to inspect prices, returns, volume and trade value. Machine learning analysis is used to estimate 20-trading-day direction and expected return. Portfolio analysis is used to calculate covariance, volatility, expected return, Sharpe ratio, Sortino ratio, CVaR, drawdown, tracking error, information ratio, sector exposure and risk contribution. Comparative analysis is then used to compare the optimised portfolio with the current portfolio, an equal-weight portfolio and a liquidity-weighted benchmark.",
            first_line=True,
        ),
        heading("3.9 Model Development", 2),
        para(
            "The main aim of model development was to create a signal-aware portfolio construction pipeline that can recommend practical equity allocations. The model development process followed the same broad flow as the software-system template: data collection and preprocessing, data-source definition, dataset splitting, feature preparation, model training, model evaluation and web application integration.",
            first_line=True,
        ),
        heading("3.9.1 Data Collection and Pre-processing", 2),
        para(
            "The preprocessing stage reads PRICE_LIST.csv, converts transaction dates, converts numeric fields, fills or removes invalid values where necessary and removes symbols that are not ordinary equities. The records are sorted by symbol and transaction date so that time-series features are generated correctly. The preprocessing stage also creates daily returns and prepares the data for model training and optimisation.",
            first_line=True,
        ),
        heading("3.9.2 Data Sources", 2),
        para(
            "Two project data sources were used. The first is PRICE_LIST.csv, which stores historical Nigerian market records from 21 August 1987 to 4 November 2025. The second is signal_store.csv, which stores the final consensus output produced after running or merging the XGBoost, Random Forest, LSTM and fallback momentum signal pipeline.",
            first_line=True,
        ),
        heading("3.9.3 Dataset Splitting", 2),
        para(
            "Chronological splitting was used because financial data is time dependent. The training data appears before the test data, preventing the model from learning future information during training. The split strategy used in the modelling pipeline is summarised in Table 3.2.",
            first_line=True,
        ),
        para("Table 3.3: Dataset Splitting Strategy", align="center", bold=True),
        table(split_rows, [2400, 3300, 3800]),
        heading("3.9.4 Feature Engineering Pipeline", 2),
        para(
            "The feature engineering pipeline creates lag features, relative price movement, momentum indicators, trend indicators, volatility measures and calendar features. These include close-price lags, RSI, MACD, Bollinger Bands, exponential moving averages, simple moving averages, rolling returns, rolling standard deviation, price-position measures, month, quarter, day of week and cyclical date encodings. These features provide the models with information about price trend, momentum, volatility and seasonality.",
            first_line=True,
        ),
        heading("3.10 Machine Learning Signal Architecture", 2),
        para(
            "The system uses a multi-model signal architecture rather than relying on a single model. Each model is trained to estimate near-term direction or expected return for individual equities. The model outputs are later merged into a symbol-level signal store. Figure 3.2 represents the conceptual signal architecture used by the system.",
            first_line=True,
        ),
        para("Figure 3.2: Machine Learning Signal Architecture", align="center", bold=True),
        para("Table 3.4: Machine Learning Models Used in the System", align="center", bold=True),
        table(model_rows, [1900, 4300, 3300]),
        heading("3.10.1 XGBoost Signal Model", 2),
        para(
            "XGBoost was used because of its ability to learn nonlinear relationships from structured tabular data. It receives engineered market features and produces a predicted return, directional signal, direction accuracy and confidence score. Its boosted-tree structure makes it suitable for capturing interactions among momentum, volatility and calendar features.",
            first_line=True,
        ),
        heading("3.10.2 Random Forest Signal Model", 2),
        para(
            "Random Forest was used as a stable ensemble model for noisy financial data. It trains several decision trees and averages their outputs, reducing dependence on a single fitted tree. In the system, Random Forest provides an additional model vote that helps reduce the risk of relying on one prediction method.",
            first_line=True,
        ),
        heading("3.10.3 LSTM Signal Model", 2),
        para(
            "The Long Short-Term Memory model was included to capture sequential price patterns from historical lookback windows. Unlike the tree-based models, the LSTM is designed for ordered data and is useful where recent price sequences may influence short-term movement.",
            first_line=True,
        ),
        heading("3.11 Consensus Signal Design", 2),
        para(
            "The merge_signals.py module combines model outputs into signal_store.csv. Before a model is allowed to vote, it must pass quality conditions based on direction accuracy, confidence and probability edge. The qualified votes are counted to produce a consensus signal. If all qualified models agree, the signal is strong. If the models disagree or no model passes the quality gate, the row is marked as conflict or handled through fallback price momentum where applicable.",
            first_line=True,
        ),
        heading("3.12 Portfolio Optimisation Design", 2),
        para(
            "The portfolio optimiser is the decision engine of the system. It accepts existing holdings or available cash, loads the supported equity universe, reads consensus signals and builds a candidate list. Expected returns are produced by blending historical annualised returns with signal-implied returns, while risk is represented with an annualised covariance matrix. The optimiser then searches for long-only portfolio weights that satisfy risk and mandate constraints.",
            first_line=True,
        ),
        para(
            "This design reflects the Chapter Two literature review, where prediction-only systems and optimisation-only systems were identified as incomplete for fund-management use. The optimiser therefore operates as a hybrid layer: machine learning supplies forward-looking signals, while portfolio theory supplies risk, diversification, benchmark and constraint logic.",
            first_line=True,
        ),
        para("Table 3.5: Risk and Mandate Constraint Design", align="center", bold=True),
        table(constraint_rows, [2300, 2300, 2300, 2800]),
        heading("3.13 Web Application Development", 2),
        para(
            "The portfolio model was integrated into a web application so that fund managers can use the system without interacting directly with Python scripts. The application is made up of a FastAPI backend and a React TypeScript frontend. The backend exposes endpoints for symbols, prices, signals, optimisation, construction and workspace management. The frontend provides forms, dashboards, reports and saved workspaces.",
            first_line=True,
        ),
        heading("3.13.1 FastAPI Backend", 2),
        para(
            "The backend is built with FastAPI. It validates requests using Pydantic schemas and routes each request to the appropriate service function. The backend loads latest prices, signal summaries and watchlists, optimises existing portfolios, constructs new portfolios from cash and stores fund-manager workspaces. It returns structured JSON responses that the frontend can display.",
            first_line=True,
        ),
        heading("3.13.2 Frontend with React", 2),
        para(
            "The frontend is built with React, TypeScript and Vite. It acts as the primary interaction point between the fund manager and the optimisation engine. The Overview tab serves as the default landing page and presents a live KPI strip, a six-step system workflow guide and a data freshness panel. The interface also supports a signal watchlist, portfolio input, risk and mandate selection, optimisation results, compliance reports, dashboard metrics and fund-manager workspace management. Three navigation tabs, Clients, Input and Dashboard, are conditionally rendered and only become visible in the sidebar after a fund manager has successfully logged in. State management is handled with React hooks so that user inputs and API responses can update the interface dynamically.",
            first_line=True,
        ),
        heading("3.13.3 Fund Manager Authentication", 2),
        para(
            "The Workspace tab implements an email and password authentication flow. The Create Account view is presented by default, requiring the fund manager to enter their full name, firm name, email address and a password of at least eight characters. A Log In link at the bottom of that view switches to the login form where an existing manager authenticates with their email and password. On the backend, passwords are hashed using the PBKDF2-SHA256 key-derivation function from the Python standard library with a 32-byte random salt and 200,000 iterations. The derived key and salt are stored together in portfolio_store.json. A dedicated POST /fund-managers/login endpoint verifies submitted credentials against the stored hash and returns the manager record on a successful match. On logout, the session is cleared and the sidebar returns to its unauthenticated state, hiding the Clients, Input and Dashboard tabs.",
            first_line=True,
        ),
        heading("3.13.4 Trade Outcome Store", 2),
        para(
            "The outcome_store.py module provides persistent trade-level outcome tracking for every portfolio optimisation run. After each optimisation, one row per allocated symbol is written to an SQLite database, capturing the run identifier, portfolio and manager identifiers, trade action, signal status, expected return, current and optimised weights and the price at decision date. A second table stores periodic price snapshots. When a new price set is submitted, the module identifies open trades whose holding period has elapsed, computes the realised return as the percentage change between the decision price and the outcome price, and records the prediction error as the difference between the realised and expected return. A CSV mirror of the trade table is maintained in outcome_store.csv for direct inspection. A dedicated query function aggregates historical accuracy statistics per symbol, providing a signal-quality feedback loop.",
            first_line=True,
        ),
        heading("3.13.5 Batch Optimisation", 2),
        para(
            "The batch_optimiser.py module allows a fund manager to run optimisation for multiple client portfolios in a single operation. The Clients tab displays all saved portfolios under the authenticated manager. The fund manager selects any number of portfolios using checkboxes and triggers batch optimisation. On the backend, a thread pool executes each portfolio's optimise or construct call independently, with a configurable concurrency limit. Each successful result is persisted to the portfolio store and the outcome store. A structured summary is returned, reporting the number of portfolios that succeeded and failed, with individual error messages for any that did not complete.",
            first_line=True,
        ),
        heading("3.14 System Architecture", 2),
        para(
            "The system follows a layered architecture that separates user interface, API validation, service logic, model logic and data storage. This separation improves maintainability because each layer can be modified without rewriting the entire system.",
            first_line=True,
        ),
        para("Table 3.6: System Architecture Layers", align="center", bold=True),
        table(architecture_rows, [2200, 3100, 4200]),
        heading("3.15 UML Modelling", 2),
        para(
            "To represent the behaviour, workflow and data structure of the system, use case, activity, sequence and data model diagrams were considered. These diagrams help to show the boundaries of the application, the main user interactions and the way data moves through the system.",
            first_line=True,
        ),
        para("Table 3.7: UML and Design Views", align="center", bold=True),
        table(uml_rows, [2000, 3600, 3900]),
        heading("3.16 Ethical Considerations", 2),
        para(
            "The system uses financial market data and does not require personal data for model training. However, ethical considerations still apply because the system supports investment decisions. The recommendations generated by the system should be interpreted as decision-support outputs rather than guaranteed investment advice. The system also displays risk and compliance information so that users can consider uncertainty, downside risk and mandate suitability before making final investment decisions.",
            first_line=True,
        ),
        heading("3.17 Summary", 2),
        para(
            "This chapter presented the system analysis, methodology and design of the machine learning-based equity portfolio construction system. It discussed the requirements, research design, data collection method, data analysis method, model development process, portfolio optimisation design, fund manager authentication design, trade outcome store design, batch optimisation design, web application architecture and UML modelling. The next chapter presents the system implementation, data presentation, evaluation results and discussion.",
            first_line=True,
        ),
    ]


def chapter_four() -> list[str]:
    objective_rows = [
        ["S/N", "Objective from Chapter One", "Implementation/Evaluation Evidence"],
        ["1", "Process Nigerian equity market data.", "PRICE_LIST.csv was cleaned, filtered and transformed into model and optimiser inputs."],
        ["2", "Develop ML signal module.", "XGBoost, Random Forest, LSTM and fallback momentum outputs were merged into signal_store.csv."],
        ["3", "Implement constrained construction and rebalancing engine.", "The optimiser produced long-only allocations under risk, mandate, sector, liquidity, turnover and cost constraints."],
        ["4", "Implement full-stack decision-support system.", "FastAPI endpoints were connected to a React TypeScript dashboard and workspace interface. Fund manager authentication, batch optimisation and a trade outcome store were implemented as additional modules."],
        ["5", "Test and compare against reference strategies.", "Automated tests were run and portfolio outputs were compared with current, equal-weight and liquidity-weighted benchmark portfolios."],
    ]
    hardware_rows = [
        ["S/N", "Requirement", "Hardware"],
        ["1.", "Processor", "Intel Core i5 or Apple Silicon equivalent minimum; higher processor recommended for model training."],
        ["2.", "Primary Memory", "8 GB RAM minimum; 16 GB RAM or higher recommended for model training and large CSV processing."],
        ["3.", "Architecture", "64-bit operating system."],
        ["4.", "Storage", "At least 5 GB free disk space for dataset, source code, generated model files and reports."],
        ["5.", "Internet Connection", "Required for dependency installation, hosted deployment and browser-based access."],
    ]
    software_rows = [
        ["S/N", "Requirement", "Software"],
        ["1.", "Operating System", "macOS, Windows 10 or higher, or Linux/Ubuntu."],
        ["2.", "Programming Language", "Python 3, TypeScript and JavaScript."],
        ["3.", "Backend Framework", "FastAPI and Uvicorn."],
        ["4.", "Machine Learning Libraries", "pandas, NumPy, scikit-learn, XGBoost and PyTorch."],
        ["5.", "Frontend Tools", "React, Vite, HTML and CSS."],
        ["6.", "Development Tools", "PyCharm/Visual Studio Code, Git and GitHub."],
        ["7.", "Database", "SQLite (Python standard library) for trade outcome and price snapshot storage."],
    ]
    tool_rows = [
        ["Tool", "Use in the project"],
        ["Python", "Used for preprocessing, model training, signal merging, API development, portfolio optimisation, authentication and outcome tracking."],
        ["FastAPI", "Used to expose backend endpoints and connect the frontend to the optimisation engine."],
        ["React", "Used to build the browser-based portfolio dashboard and workspace interface."],
        ["TypeScript", "Used to define frontend data contracts and reduce interface errors."],
        ["pandas and NumPy", "Used for data cleaning, returns, covariance, metrics and numerical processing."],
        ["XGBoost, Random Forest and LSTM", "Used to generate short-term equity signals."],
        ["SQLite (Python built-in)", "Used to store per-symbol trade decisions and realised outcome measurements in outcome_store.db."],
        ["hashlib PBKDF2-SHA256 (Python built-in)", "Used to hash fund manager passwords with a random salt before storage."],
        ["GitHub", "Used for version control and project backup."],
    ]
    endpoint_rows = [
        ["Endpoint", "Method", "Function"],
        ["/health", "GET", "Checks whether the backend service is running."],
        ["/bootstrap", "GET", "Returns symbols, prices, signal summary and watchlist in a single page-load call."],
        ["/symbols", "GET", "Returns supported Nigerian equity symbols."],
        ["/prices/latest", "GET", "Returns latest prices and update timestamp."],
        ["/signals/summary", "GET", "Returns signal-store counts and average quality values."],
        ["/signals/watchlist", "GET", "Returns top buy and sell watchlist entries."],
        ["/optimize-portfolio", "POST", "Optimises an existing portfolio."],
        ["/construct-portfolio", "POST", "Constructs a portfolio from initial cash."],
        ["/fund-managers", "GET/POST", "Lists all managers or creates a new manager account."],
        ["/fund-managers/login", "POST", "Authenticates a fund manager by email and password and returns the manager record."],
        ["/fund-managers/{id}", "DELETE", "Deletes a manager and all associated clients and portfolios."],
        ["/fund-managers/{id}/consumers", "GET/POST/DELETE", "Lists, creates and removes client records under a specific manager."],
        ["/fund-managers/{id}/portfolios", "GET/POST", "Lists saved portfolios or saves a new portfolio under a specific manager."],
        ["/fund-managers/{id}/batch-optimize", "POST", "Runs optimisation concurrently for a selected list of client portfolios."],
        ["/portfolios/{portfolio_id}/optimize", "POST", "Reruns optimisation for a single saved portfolio."],
        ["/outcomes/trades", "GET", "Returns trade outcome records, optionally filtered by manager, portfolio or symbol."],
        ["/outcomes/symbol/{symbol}", "GET", "Returns the full trade decision and outcome history for one symbol."],
        ["/outcomes/alpha/{symbol}", "GET", "Returns aggregated prediction accuracy statistics for one symbol."],
        ["/outcomes/record-prices", "POST", "Records a price snapshot and computes realised returns for elapsed trades."],
        ["/outcomes/export-csv", "GET", "Syncs and downloads the full outcome trade table as a CSV file."],
    ]
    data_rows = [
        ["Metric", "Result"],
        ["Total market price rows", "344,996"],
        ["Raw symbols in price file", "526"],
        ["Filtered supported equity symbols", "253"],
        ["Date range", "21 August 1987 to 4 November 2025"],
        ["Signal-store rows", "129"],
        ["Consensus BUY signals", "79"],
        ["Consensus SELL signals", "38"],
        ["Consensus CONFLICT signals", "7"],
        ["Consensus HOLD signals", "5"],
        ["Average signal confidence", "0.1138"],
        ["Average signal quality score", "0.0249"],
    ]
    sample_rows = [
        ["Evaluation item", "Observed result"],
        ["Portfolio mode", "Optimisation of existing holdings."],
        ["Input portfolio", "ACCESSCORP, GTCO, ZENITHBANK, MTNN, DANGCEM, BUAFOODS, SEPLAT, UBA, FBNH and AIRTELAFRI."],
        ["Current portfolio value", "NGN 10,000,000.00"],
        ["Optimised portfolio value after transaction cost", "NGN 9,994,112.73"],
        ["Number of optimised allocations", "15 stocks"],
        ["Added stocks", "JBERGER, WEMABANK, OANDO, UAC-PROP and AFRIPRUD."],
        ["Exited stocks", "GTCO and ZENITHBANK."],
        ["Compliance status", "Pass"],
        ["Expected return", "27.074%"],
        ["Volatility", "20.004%"],
        ["Sharpe ratio", "1.2535"],
        ["Sortino ratio", "2.8581"],
        ["CVaR at 95%", "-2.7633%"],
        ["Maximum drawdown", "-23.0069%"],
        ["Diversification score", "72 out of 100"],
        ["Effective positions", "11.43"],
    ]
    benchmark_rows = [
        ["Metric", "Interpretation"],
        ["Sharpe ratio", "Measures expected excess return relative to total volatility."],
        ["Sortino ratio", "Measures return relative to downside volatility."],
        ["CVaR at 95%", "Estimates expected loss in the worst 5% of daily outcomes."],
        ["Maximum drawdown", "Shows the largest historical peak-to-trough fall in portfolio value."],
        ["Diversification score", "Summarises spread across stocks and sectors."],
        ["Compliance status", "Shows whether the recommended portfolio satisfies mandate constraints."],
    ]
    reference_rows = [
        ["Strategy", "Cumulative Return", "Annualised Return", "Volatility", "Sharpe", "Max Drawdown"],
        ["Current portfolio", "46.1072%", "38.8744%", "13.6668%", "2.6981", "-8.1075%"],
        ["Optimised portfolio", "39.9646%", "34.4720%", "12.8802%", "2.5211", "-9.2354%"],
        ["Equal-weight portfolio", "44.2154%", "37.5574%", "13.5511%", "2.6240", "-11.2344%"],
        ["Liquidity-weighted benchmark", "63.1568%", "50.4858%", "17.2596%", "2.8092", "-9.9722%"],
    ]
    test_rows = [
        ["Test case", "Expected behaviour", "Result"],
        ["Supported symbol endpoint", "Returns available equity symbols.", "Passed in API tests."],
        ["Empty holdings validation", "Rejects portfolios without holdings.", "Passed in optimiser tests."],
        ["Invalid symbol validation", "Rejects unsupported tickers.", "Passed in optimiser tests."],
        ["Invalid mandate validation", "Rejects unknown mandate profile.", "Passed in optimiser and API tests."],
        ["Weight validity", "Optimised weights sum to one and remain non-negative.", "Passed in optimiser tests."],
        ["Portfolio construction", "Constructs first portfolio from cash.", "Passed in optimiser tests."],
        ["Workspace management", "Creates managers, consumers and saved portfolios.", "Passed in API tests."],
    ]

    return [
        page_break(),
        heading("CHAPTER FOUR", center=True),
        heading("SYSTEM IMPLEMENTATION AND EVALUATION", center=True),
        heading("4.1 Preamble", 2),
        para(
            "This chapter presents the implementation and evaluation of the machine learning-based equity portfolio construction system. Following the structure of the system implementation template, it explains the hardware and software requirements, implementation tools, development methodology, program modules and system interfaces. Following the approved report guide, it also presents the project data, evaluation results and discussion of findings using tables and narrative analysis.",
            first_line=True,
        ),
        para(
            "The implementation and evaluation are organised around the objectives and methodology already stated in Chapter One. Therefore, the chapter does not only describe the software modules; it also shows how each objective was implemented and how the final portfolio output was evaluated with the same risk-adjusted and benchmark-based criteria introduced in Chapters One and Two.",
            first_line=True,
        ),
        para("Table 4.1: Objectives, Implementation and Evaluation Alignment", align="center", bold=True),
        table(objective_rows, [800, 3600, 5100]),
        heading("4.2 System Requirements", 2),
        para(
            "System requirements refer to the minimum standards and resources needed for the software to run successfully and meet user needs. The requirements are grouped into hardware and software requirements.",
            first_line=True,
        ),
        heading("4.2.1 Hardware Requirements", 2),
        para("The hardware requirements for running and developing the system are shown in Table 4.2.", first_line=True),
        para("Table 4.2: Hardware Requirements Table", align="center", bold=True),
        table(hardware_rows, [900, 3000, 5600]),
        heading("4.2.2 Software Requirements", 2),
        para("The software requirements necessary for running the system are shown in Table 4.3.", first_line=True),
        para("Table 4.3: Software Requirements Table", align="center", bold=True),
        table(software_rows, [900, 3000, 5600]),
        heading("4.3 Implementation Tools", 2),
        para(
            "These are the tools utilised while building the software project. They supported the development of the machine learning models, optimisation engine, backend API and frontend dashboard.",
            first_line=True,
        ),
        para("Table 4.4: Implementation Tools", align="center", bold=True),
        table(tool_rows, [2800, 6700]),
        heading("4.4 Development Methodology", 2),
        para(
            "The development methodology used for this project is the iterative waterfall model. The project followed a structured sequence of requirements gathering, data preparation, model design, model training, backend development, frontend development, testing and evaluation. However, because machine learning projects require repeated experimentation, the process allowed movement back to earlier stages whenever model output, data quality or interface behaviour required refinement.",
            first_line=True,
        ),
        para(
            "During execution, the price dataset was cleaned, the model features were adjusted, several signal models were evaluated and the optimiser constraints were refined. This iterative approach was useful because portfolio construction requires both quantitative performance and practical investment constraints.",
            first_line=True,
        ),
        para(
            "This implementation follows the Chapter One methodology in five broad stages: data preparation, machine learning modelling, signal merging, portfolio optimisation, and backend/frontend integration. The testing stage then checks both software behaviour and portfolio behaviour.",
            first_line=True,
        ),
        heading("4.5 System Implementation", 2),
        para(
            "The system implementation consists of the data preprocessing module, machine learning signal module, signal merging module, portfolio optimisation module, backend API, frontend dashboard and workspace persistence module.",
            first_line=True,
        ),
        heading("4.5.1 Data Preprocessing Module", 2),
        para(
            "The data preprocessing module loads PRICE_LIST.csv, checks required columns, converts transaction dates, converts numeric price and volume fields, removes invalid close prices and filters non-equity instruments. It then sorts the data by symbol and transaction date. This prepares the historical market data for feature engineering, signal modelling and portfolio optimisation.",
            first_line=True,
        ),
        heading("4.5.2 Machine Learning Signal Module", 2),
        para(
            "The machine learning module trains or consumes XGBoost, Random Forest and LSTM outputs for each eligible stock. The models use engineered technical indicators and chronological train-test splitting to produce predicted return, BUY or SELL signal, direction accuracy, balanced accuracy, F1 score and confidence score.",
            first_line=True,
        ),
        heading("4.5.3 Signal Merging Module", 2),
        para(
            "The signal merging module combines individual model outputs into signal_store.csv. It applies quality checks before allowing a model to vote, then produces a consensus signal for each stock. The current signal store contains 129 symbol rows made up of 79 BUY signals, 38 SELL signals, 7 CONFLICT signals and 5 HOLD signals.",
            first_line=True,
        ),
        heading("4.5.4 Portfolio Optimisation Module", 2),
        para(
            "The portfolio optimisation module supports two modes: construction of a new portfolio from cash and optimisation of an existing portfolio. It validates holdings, builds a candidate universe, estimates expected returns, computes annualised covariance, applies risk and mandate constraints and returns recommended allocations. The output includes action labels such as keep, increase, reduce, add and exit.",
            first_line=True,
        ),
        heading("4.5.5 Backend API Module", 2),
        para(
            "The backend API is implemented in main.py using FastAPI. The API validates all requests with Pydantic models and exposes endpoints for prices, symbols, signals, optimisation, portfolio construction, fund-manager authentication, workspace management, batch optimisation and trade outcome tracking. A bootstrap endpoint consolidates symbols, prices, signal summary and watchlist into a single page-load response. The main endpoints are shown in Table 4.5.",
            first_line=True,
        ),
        para("Table 4.5: Main API Endpoints", align="center", bold=True),
        table(endpoint_rows, [2900, 1500, 5100]),
        heading("4.5.6 Frontend Interface Module", 2),
        para(
            "The frontend is implemented with React and TypeScript. It provides the overview, watchlist, input, dashboard, workspace and clients views. The Overview tab is the default landing page and displays a KPI strip, a six-step system workflow guide and a data freshness panel showing when signals and prices were last updated. The watchlist shows top buy and sell signals. The input view allows users to enter existing holdings or initial cash. The dashboard presents allocations, sector exposure, performance metrics, compliance status, backtest results, efficient frontier points and risk contributions. The workspace view implements the fund manager authentication flow, and the clients view lists all saved portfolios under the logged-in manager with batch optimisation controls.",
            first_line=True,
        ),
        heading("4.5.7 Trade Outcome Store Module", 2),
        para(
            "The outcome_store.py module records the trade-level output of every optimisation run into a SQLite database and a CSV mirror. Each row captures the run identifier, portfolio and manager identifiers, consumer identifier, symbol, sector, trade action, signal status, expected return, current and optimised weights, and the closing price at decision date. A second table stores dated price snapshots. When a price snapshot is submitted via the /outcomes/record-prices endpoint, the module matches open trades to the new prices, computes the realised return as the percentage price change and stores the prediction error as the difference between the realised and expected return. This creates a closed-loop feedback record for evaluating signal accuracy over time.",
            first_line=True,
        ),
        heading("4.5.8 Batch Optimisation Module", 2),
        para(
            "The batch_optimiser.py module allows a fund manager to optimise multiple client portfolios in parallel. The Clients tab provides checkboxes to select any number of saved portfolios. On submission, the /fund-managers/{id}/batch-optimize endpoint passes the selected portfolio identifiers to the batch_optimiser module, which executes each optimisation in a thread pool with a maximum of four concurrent workers. Each portfolio is processed using the correct mode, optimise for existing holdings or construct for new cash portfolios, based on its saved configuration. Successful results are persisted to the portfolio store and the outcome store. The response includes a batch run identifier, counts of succeeded and failed portfolios, and individual error messages for any failures.",
            first_line=True,
        ),
        heading("4.6 Data Presentation and Analysis", 2),
        para(
            "This section presents the project data and the results obtained from the developed system. The analysis focuses on the available market dataset, the generated signal store and the output of a sample portfolio optimisation run.",
            first_line=True,
        ),
        heading("4.6.1 Dataset and Signal Summary", 2),
        para(
            "Table 4.6 presents the major dataset and signal-store statistics used to evaluate the project. These values provide context for the scope of the system and show the size of the data processed by the application.",
            first_line=True,
        ),
        para("Table 4.6: Dataset and Signal Evaluation Summary", align="center", bold=True),
        table(data_rows, [4300, 5200]),
        heading("4.6.2 Portfolio Optimisation Result", 2),
        para(
            "A sample existing portfolio was evaluated using the balanced risk profile and balanced equity mandate. The input portfolio consisted of equal NGN 1,000,000 positions in ACCESSCORP, GTCO, ZENITHBANK, MTNN, DANGCEM, BUAFOODS, SEPLAT, UBA, FBNH and AIRTELAFRI. The output generated by the system is presented in Table 4.7.",
            first_line=True,
        ),
        para("Table 4.7: Sample Portfolio Optimisation Result", align="center", bold=True),
        table(sample_rows, [4300, 5200]),
        heading("4.6.3 Evaluation Criteria", 2),
        para(
            "The system was evaluated using both software testing criteria and portfolio performance criteria. The software criteria considered validation, API behaviour, portfolio construction behaviour and workspace management. The portfolio criteria considered expected return, volatility, Sharpe ratio, Sortino ratio, CVaR, drawdown, diversification and compliance.",
            first_line=True,
        ),
        para("Table 4.8: Portfolio Evaluation Criteria", align="center", bold=True),
        table(benchmark_rows, [3300, 6200]),
        heading("4.6.4 Reference Strategy Comparison", 2),
        para(
            "Chapter One stated that the system output would be compared against traditional portfolio references. To keep the evaluation consistent with that methodology, the sample optimisation was compared with the current portfolio, an equal-weight allocation and a liquidity-weighted benchmark over a 252-trading-day backtest window. The result is shown in Table 4.9.",
            first_line=True,
        ),
        para("Table 4.9: Comparison with Current, Equal-Weight and Benchmark Portfolios", align="center", bold=True),
        table(reference_rows, [2300, 1600, 1700, 1500, 1200, 1400]),
        heading("4.7 System Testing", 2),
        para(
            "Automated tests were used to confirm that the backend and optimiser behave correctly. The optimiser tests check empty holdings, invalid symbols, invalid mandates, long-only weights, portfolio construction and compliance output. The API tests check endpoint responses, request validation, fund-manager creation, consumer creation, portfolio saving and deletion behaviour.",
            first_line=True,
        ),
        para("Table 4.10: Summary of System Tests", align="center", bold=True),
        table(test_rows, [2800, 4100, 2600]),
        heading("4.8 Graph Visualization", 2),
        para(
            "The frontend dashboard provides visual presentation of allocation and risk results. The system displays allocation comparisons, sector exposure, efficient frontier points, correlation matrix and risk contribution views. These visualisations help fund managers understand how the optimised portfolio differs from the current portfolio and how risk is distributed across stocks and sectors.",
            first_line=True,
        ),
        para("Figure 4.1: Portfolio Allocation Comparison", align="center", bold=True),
        para("Figure 4.2: Risk Contribution and Correlation Visualization", align="center", bold=True),
        heading("4.9 Discussion of Findings", 2),
        para(
            "The results show that the developed system can transform raw Nigerian equity market data into actionable portfolio recommendations. From the sample run, the optimiser recommended a 15-stock portfolio, added five new stocks and exited two current positions while maintaining a pass status under the balanced equity mandate. The expected return of 27.074% and Sharpe ratio of 1.2535 indicate that the portfolio produced a favourable risk-return profile under the system's assumptions.",
            first_line=True,
        ),
        para(
            "The reference-strategy comparison gives a more balanced interpretation of the result. The liquidity-weighted benchmark produced the highest backtested annualised return and Sharpe ratio in the sample window, while the optimised portfolio produced lower volatility than the current, equal-weight and benchmark portfolios. This is consistent with the literature review in Chapter Two, which noted that optimisation should be tested against simple reference strategies because optimised portfolios do not automatically outperform in every historical period.",
            first_line=True,
        ),
        para(
            "The result also shows that the system does not simply select the highest-returning stocks. It applies concentration limits, sector limits, liquidity screening, transaction-cost estimates, no-trade bands and mandate controls. This makes the system more suitable for fund-manager decision support because practical portfolio construction requires more than stock prediction alone.",
            first_line=True,
        ),
        para(
            "The diversification score of 72 out of 100 and effective position count of 11.43 show that the recommended allocation is spread across multiple active holdings. The CVaR and maximum drawdown values also show that the system presents downside-risk information, allowing the fund manager to interpret the recommendation with caution rather than treating it as a guaranteed investment outcome.",
            first_line=True,
        ),
        heading("4.10 Program Modules and Interfaces", 2),
        para(
            "This section outlines the main user-facing interfaces of the system. The watchlist page displays BUY and SELL opportunities from the signal store. The input page captures holdings, cash, risk profile and mandate profile. The dashboard page presents portfolio recommendations and performance analytics. The workspace page manages fund managers, consumers, saved portfolios and previous optimisation runs.",
            first_line=True,
        ),
        heading("4.10.1 Watchlist Page", 2),
        para(
            "The watchlist page displays stocks with strong BUY or SELL signals. It gives the fund manager a quick view of market opportunities and risks before running a full portfolio optimisation.",
            first_line=True,
        ),
        heading("4.10.2 Portfolio Input Page", 2),
        para(
            "The portfolio input page allows the user to enter either current holdings or available cash. It also allows the user to choose risk profile, mandate profile, maximum new stocks, holding period and rebalancing frequency.",
            first_line=True,
        ),
        heading("4.10.3 Dashboard Page", 2),
        para(
            "The dashboard page presents the main optimisation result. It shows recommended allocation weights, current and optimised values, trade actions, compliance checks, sector allocation, benchmark metrics, efficient frontier data and risk contributions.",
            first_line=True,
        ),
        heading("4.10.4 Workspace Page", 2),
        para(
            "The workspace page implements the fund manager authentication flow. The Create Account view is the default screen and allows a new manager to register with their name, firm, email and password. Clicking the Log In link switches to the login form where an existing manager authenticates. Once logged in, the workspace displays the manager identity and provides options to create a new client or select a saved client. The Clients, Input and Dashboard navigation tabs become visible only after login. On logout, the session is cleared, the conditional tabs disappear and the workspace returns to the Create Account screen.",
            first_line=True,
        ),
        heading("4.10.5 Overview Page", 2),
        para(
            "The Overview page is the default landing view when the application loads. It shows a KPI strip with four live metrics: current or post-trade portfolio value, signal universe size with buy and sell counts, average signal confidence, and optimisation status. Below the KPI strip, a System Flow panel presents six numbered steps that guide the fund manager through the full workflow, from browsing the signal watchlist to saving a completed optimisation run. A Data Freshness panel at the bottom shows the timestamps and relative age of the most recent signal generation and price update.",
            first_line=True,
        ),
        para("Figure 4.3: Overview Page", align="center", bold=True),
        heading("4.10.6 Clients Page", 2),
        para(
            "The Clients page is accessible only to authenticated fund managers. It lists all client portfolios saved under the logged-in manager, showing the client name, portfolio name, risk profile, mandate, holding count and last optimisation date. Each portfolio row can be expanded to view individual optimisation run history. The page also provides batch optimisation controls: the fund manager selects one or more portfolios using checkboxes and clicks a single button to trigger concurrent reoptimisation across the selected set.",
            first_line=True,
        ),
        para("Figure 4.4: Clients Page with Batch Optimisation", align="center", bold=True),
        heading("4.11 Summary", 2),
        para(
            "This chapter presented the implementation and evaluation of the machine learning-based equity portfolio construction system. It described the system requirements, implementation tools, development methodology, program modules, API endpoints, fund manager authentication, trade outcome store, batch optimisation module, dataset summary, optimisation results, testing process, visualisations, discussion of findings and user interfaces. The findings show that the system can support fund managers by combining machine learning signals, portfolio optimisation, compliance-aware reporting, secure account management and trade outcome tracking in a browser-based application.",
            first_line=True,
        ),
    ]


def styles_xml() -> str:
    return """<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<w:styles xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main">
  <w:style w:type="paragraph" w:default="1" w:styleId="Normal">
    <w:name w:val="Normal"/>
    <w:qFormat/>
    <w:pPr><w:spacing w:after="120" w:line="360" w:lineRule="auto"/><w:jc w:val="both"/></w:pPr>
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
      <w:lvlText w:val="*"/>
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
        '<w:pgMar w:top="1440" w:right="1440" w:bottom="1440" w:left="1800" '
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
 xmlns:wpg="http://schemas.microsoft.com/office/2010/wordprocessingGroup"
 xmlns:wpi="http://schemas.microsoft.com/office/2010/wordprocessingInk"
 xmlns:wne="http://schemas.microsoft.com/office/word/2006/wordml"
 xmlns:wps="http://schemas.microsoft.com/office/2010/wordprocessingShape"
 mc:Ignorable="w14 wp14"><w:body>{body}{sect}</w:body></w:document>"""


def write_docx() -> None:
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    paragraphs = []
    paragraphs.extend(chapter_three())
    paragraphs.extend(chapter_four())

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
