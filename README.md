# Mathematical Modelling of Football (1RT001)

Coursework for **Mathematical Modelling of Football** (1RT001), period 1 of the M.Sc. in Image Analysis and Machine Learning at Uppsala University. The course applies statistics and machine learning to football event data, mostly from StatsBomb's open data.

Three assignments, each in its own folder. Data files are kept out of the repo (see `.gitignore`) — grab the event JSON from the [StatsBomb Open Data repository](https://github.com/statsbomb/open-data) and point the notebooks/script at it.

## Contents

- **`a1-pitch-exploration/`** — getting started with the StatsBomb data. The notebook loads `competitions.json` into a DataFrame so you can see which competitions and seasons are available before pulling match events. A small first step rather than a finished analysis.
- **`a2-xDA-pass-danger/`** — the main piece of work. A pass-evaluation model I call xDA-P (Expected Danger Added by Passes), with rankings of the most threatening passers across five 2015/16 leagues.
- **`a3-statistical-testing/`** — directional analysis of one player's off-ball runs into the final third, using circular statistics.

## a2 — xDA-P

The idea: for every pass, estimate how much it raises the attacking team's threat, then add those gains up per player to get a pass-danger score that can be compared across leagues and normalised per 90 minutes.

I score positions with a hand-crafted expected-threat proxy `xT(x, y)` that rises towards the opponent's goal and peaks centrally (a sigmoid in x times a Gaussian in y). The label for a completed pass is `max(0, xT(end) − xT(start))`; incomplete passes get zero. Per-pass features include start/end coordinates, distance, angle, a forward flag, under-pressure flag, a wide-switch indicator, pass height/body-part/technique, and match time.

There are two takes on this in the folder:

- **`fit_xDA_model.py`** — a `GradientBoostingRegressor` pipeline. It trains out-of-fold (5-fold `KFold` with `cross_val_predict`) on the Premier League to get unbiased per-player scores on the training league, fits a final model on all of it, then scores the other leagues with that model. Minutes played come from Starting XI and Substitution events, which lets it report xDA per 90.

  ```bash
  python a2-xDA-pass-danger/fit_xDA_model.py \
      --train events_England.json \
      --score events_Spain.json events_Germany.json events_Italy.json events_France.json \
      --out_dir a2-xDA-pass-danger/outputs
  ```

- **`xDA_P_model.ipynb`** — a `GradientBoostingClassifier` variant on the Premier League, evaluated with ROC AUC, average precision and precision–recall curves against a simple baseline. The ROC/PR figures in `outputs/` come from here.

`outputs/` holds the ranked player CSVs and per-league top-10 pass-danger bar charts for the Premier League, La Liga, Bundesliga, Serie A and Ligue 1 (2015/16), plus the curve plots, with `min150`/`min200` minimum-pass filters on the rankings.

Dependencies: `numpy`, `pandas`, `scikit-learn`, `matplotlib`, and `mplsoccer` for the pitch plots.

## a3 — off-ball run direction

`hypothesis_testing.ipynb` looks at one player's (Phil Foden) off-ball runs that end in the final third. For each run I take the start→end vector and its angle, then summarise the set with circular statistics: the circular mean gives an average run direction, and the circular variance (1 − mean resultant length) measures how consistent that direction is. The runs and an average-direction arrow are drawn on a pitch with `mplsoccer` (`foden_runs_final_third.png`).

This one uses a different dataset — the runs come from the `twelve-respovision-CL-final` parquet files rather than StatsBomb event JSON, so the notebook tries a few local paths and falls back to cloning that repo.

## Data attribution

StatsBomb event data: [StatsBomb Open Data](https://github.com/statsbomb/open-data), used under their free-for-non-commercial-research terms. The a3 run data comes from [twelve-respovision-CL-final](https://github.com/twelvefootball/twelve-respovision-CL-final). No data files are redistributed here.
