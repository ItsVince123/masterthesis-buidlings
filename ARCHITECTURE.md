# Dashboard Architecture
## Building Management System — Thesis Project

---

## Thesis Responsibility Split

> **You (the student) are responsible for the BACKEND only.**  
> The frontend (PyQt6 UI, graphs, dialogs) was scaffolded generically and is NOT part of the thesis contribution.

| Layer | Your responsibility? | Description |
|-------|---------------------|-------------|
| **Backend — Optimisation** | ✅ YES | SMPC solver, LP scheduler, energy asset model |
| **Backend — Data pipeline** | ✅ YES | Price/weather/solar data fetching and prediction |
| **Backend — Configuration** | ✅ YES | Config schema, persistence, constants |
| **Frontend — Dashboard UI** | ❌ NO | PyQt6 window, columns, widgets |
| **Frontend — Dialogs** | ❌ NO | Asset editor, historical analysis dialog |
| **Frontend — Graphs** | ❌ NO | QPainter graph rendering |
| **Frontend — Styles** | ❌ NO | QSS stylesheets |

---

## File Overview

### BACKEND files (thesis contribution)

```
settings.py            — All magic constants (paths, API keys, coordinates, cost model)
energy_assets.py       — EnergyAsset dataclass, asset types, JSON persistence
dashboard_config.py    — Load/save dashboard_config.json; load SMPCConfig
smpc_calculator.py     — 🔑 CORE: Stochastic MPC optimiser (ice bank + CHP)
lp_solver.py           — 🔑 CORE: Greedy LP scheduler + historical day simulation
data_manager.py        — 🔑 CORE: Fetching, caching, and scheduling of live data
predict.py             — Solar yield prediction from weather (UV proxy model)
getPrice.py            — Fetch ENTSO-E day-ahead electricity prices
getWeather.py          — Fetch weather data (temperature, UV, wind)
getCO2.py              — Fetch ENTSO-E generation mix → CO2 intensity (gCO2/kWh)
```

### FRONTEND files (not thesis contribution)

```
dashboard.py           — Main window: 3-column SCADA layout, timer, widget updates
historical_dialog.py   — Dialog: pick a date, run LP, show cost/CO2 graphs + KPIs
asset_dialogs.py       — Dialog: add/edit/remove energy assets
graph_renderer.py      — QPainter rendering: price graph, solar graph, comparison
styles.py              — QSS stylesheet strings (colours, fonts, layout)
```

### Data files

```
dashboard_config.json  — Per-deployment settings (asset list, SMPC parameters)
DATA.csv               — Historical building data (hourly, used by historical dialog)
solar_2022.csv         — Hourly solar production archive for 2022
co2_intensity.csv      — Pre-fetched hourly CO2 intensity for Belgium
weather.csv            — Current weather cache (refreshed daily at 14:00)
predict.csv            — Solar yield predictions (derived from weather.csv)
prices.csv             — Day-ahead price cache (refreshed daily at 14:00)
```

---

## Architecture Diagram

```
                        ┌──────────────────────────────────────────┐
                        │              FRONTEND (PyQt6)             │
                        │                                           │
                        │   dashboard.py ──────► graph_renderer.py │
                        │        │                styles.py         │
                        │        │          asset_dialogs.py        │
                        │        │          historical_dialog.py    │
                        └────────┼──────────────────────────────────┘
                                 │ reads data / calls solve()
                                 ▼
                        ┌──────────────────────────────────────────┐
                        │           BACKEND (thesis work)           │
                        │                                           │
                        │   DataManager ──────► getPrice.py        │
                        │        │          ──► getWeather.py       │
                        │        │          ──► getCO2.py           │
                        │        │          ──► predict.py          │
                        │        │                                   │
                        │   SMPCCalculator ───► smpc_calculator.py  │
                        │        │          ──► lp_solver.py        │
                        │        │                                   │
                        │   EnergyAsset ──────► energy_assets.py   │
                        │   Config ───────────► dashboard_config.py │
                        │   Constants ────────► settings.py         │
                        └──────────────────────────────────────────┘
                                 │
                                 ▼
                        ┌──────────────────────┐
                        │   External APIs       │
                        │   ENTSO-E (prices)    │
                        │   ENTSO-E (CO2 mix)   │
                        │   Open-Meteo (weather)│
                        └──────────────────────┘
```

---

## Data Flow (per 15-minute tick)

```
1.  QTimer fires every second  →  dashboard._on_tick()

2.  DataManager.tick() checks if a refresh is due:
    ├── refresh_prices()      reads prices.csv   (fetched from ENTSO-E)
    ├── refresh_predictions() reads predict.csv  (derived from weather)
    └── refresh_weather()     reads weather.csv  (fetched from Open-Meteo)

3.  _run_lp_if_needed() — once per 15-min slot:
    ├── build_price_forecast()   → EUR/kWh array [96 steps]
    ├── build_load_and_solar()   → (base_load_kwh, solar_kwh) arrays
    └── SMPCCalculator.solve_lp()  ← BACKEND CALL
            ├── greedy_lp() per shiftable load
            ├── spark-spread rule for CHP
            └── returns SMPCOutputs (schedule + KPIs)

4.  _update_all_widgets() — renders results to UI labels
```

---

## Core Backend Modules Explained

### `smpc_calculator.py` — Stochastic MPC Optimiser

The heart of the thesis. Implements a **Stochastic Model Predictive Control** 
optimiser for building energy management.

**Two operating modes (seasonal):**

| Season | Active system | Optimiser |
|--------|--------------|-----------|
| Summer | Ice bank     | Convex QP (cvxpy CLARABEL solver) |
| Winter | CHP / WKK    | Linear spark-spread (cvxpy or heuristic) |
| Transition | Both at reduced capacity | Both |

**Key classes:**
- `SMPCConfig` — all tuneable parameters (loaded from `dashboard_config.json`)
- `SMPCInputs` — current measurements + forecasts fed into each solve
- `SMPCOutputs` — optimal control commands + KPIs returned by each solve
- `SMPCCalculator` — main class; call `solve()` for SMPC or `solve_lp()` for greedy LP

**Stochastic price scenarios:**  
`_generate_price_scenarios()` generates N correlated AR(1) log-normal price 
paths. The optimiser uses the scenario mean, making it robust to price uncertainty.

---

### `lp_solver.py` — Greedy LP Scheduler

Implements the **deterministic LP** used for historical analysis and as a 
reference baseline in the SMPC module.

**`greedy_lp(n_steps, total_price, total_energy, max_per_step)`**  
Schedules `total_energy` kWh across `n_steps` intervals by filling from 
cheapest to most expensive. Provably optimal for: linear cost + box constraints 
+ sum constraint (no inter-temporal coupling).

**`simulate_day(day_rows, assets, solar_hours, day_keys)`**  
Runs a full day simulation comparing:
- **Baseline**: load as it actually occurred in the historical CSV
- **LP-optimised**: shiftable loads rescheduled to cheapest hours, generators 
  applied with price-based decoupling

---

### `data_manager.py` — Live Data Layer

Manages all external data with intelligent caching:

- **Prices** — re-read from `prices.csv` every 15 minutes  
- **Solar predictions** — re-read from `predict.csv` every 15 minutes  
- **Weather/UV** — re-read from `weather.csv` every 15 minutes  
- **Daily pipeline** — at `DAILY_FETCH_HOUR` (14:00): fetches fresh weather, 
  re-runs solar prediction, fetches tomorrow's prices

---

### `predict.py` — Solar Yield Prediction

Maps weather → PV power using a simple physics-inspired model:

```
power_kw = capacity_kwp × (UV / 8.0) × temp_derate_factor
```

Where `temp_derate = 1 - max(0, T - 25°C) × 0.4%/°C`, clamped at 75%.  
UV index is used as irradiance proxy (UV 8 ≈ full sun ≈ 1000 W/m²).

---

### `getPrice.py` — ENTSO-E Price Fetcher

Fetches day-ahead electricity prices from the ENTSO-E Transparency Platform 
(XML API, document type A44). Converts from UTC to Brussels local time. 
Handles the 2025-10-01 resolution change (hourly → 15-minute).

---

### `getCO2.py` — Grid CO2 Intensity

Fetches actual generation mix per fuel type (A75) from ENTSO-E and multiplies 
by IPCC lifecycle emission factors to produce grid CO2 intensity (gCO2eq/kWh). 
Used by the historical analysis dialog to quantify CO2 savings from LP scheduling.

---

## Configuration (`dashboard_config.json`)

The config file has two top-level sections:

```json
{
  "smpc": {
    "ice_bank": { "capacity_kwh": 5000, ... },
    "wkk":      { "gas_price_eur_m3": 0.35, ... },
    "grid":     { "peak_limit_kwh": 750, ... },
    "building": { "base_load_kw": 1850, ... }
  },
  "energy_assets": [
    { "uid": "ice_banks", "asset_type": "shiftable_load", ... },
    { "uid": "solar",     "asset_type": "generator",      ... }
  ]
}
```

All SMPC parameters default to sane values if absent from the JSON, so you 
only need to specify values you want to override.

---

## Running the Dashboard

```bash
# Install dependencies
pip install PyQt6 numpy requests xmltodict cvxpy

# Run
python dashboard.py
```

The dashboard will:
1. Load config from `dashboard_config.json`
2. Immediately fetch prices, weather, and solar predictions
3. Run the LP solver once per 15-minute interval
4. Update all UI labels every second
