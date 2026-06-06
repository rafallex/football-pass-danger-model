# Mathematical Modelling of Football (1RT001)

Building a pass-danger model from raw football event data: which players, with the ball at their feet, most often set up a shot in the next ten seconds.

Coursework for **Mathematical Modelling of Football** (1RT001), period 1 of the M.Sc. in Image Analysis and Machine Learning at Uppsala University. The course applies statistics and machine learning to football event data, mostly from StatsBomb's open data.

Three assignments, each in its own folder. Data files are kept out of the repo (see `.gitignore`) — grab the event JSON from the [StatsBomb Open Data repository](https://github.com/statsbomb/open-data) and point the notebooks/script at it.

```
modelling-football-1RT001/
├── a1-pitch-exploration/
│   └── pitch_exploration.ipynb     # load competitions.json, see what data is available
├── a2-xDA-pass-danger/             # main work: the xDA-P pass-danger model
│   ├── fit_xDA_model.py            # gradient-boosting regressor, train-then-score CLI
│   ├── xDA_P_model.ipynb           # gradient-boosting classifier + ROC/PR evaluation
│   ├── competitions.json           # StatsBomb competition index (committed for reference)
│   └── outputs/                    # ranked player CSVs, bar charts, ROC/PR curves
└── a3-statistical-testing/
    ├── hypothesis_testing.ipynb    # circular statistics on one player's off-ball runs
    └── foden_runs_final_third.png
```

## a2 — xDA-P (the main piece of work)

The idea: for every pass, estimate how dangerous it is, then add those numbers up per player to get a pass-danger score that can be compared across leagues. There are two takes on "dangerous" in this folder, sharing the same StatsBomb pass-feature pipeline (start/end coordinates, distance, angle, a forward flag, pass height and body part).

**1. Classifier — does a shot follow within 10 seconds?** (`xDA_P_model.ipynb`)

This is the version I trust most, because the label is a real outcome rather than a hand-crafted proxy. A pass is positive if the same team takes a shot within 10 seconds of it (same possession). On the Premier League 2015/16 events that gives **8,787 passes with usable coordinates, 576 of them positive (a 6.6% base rate)**. A `GradientBoostingClassifier` (400 trees, learning rate 0.05, depth 3) is trained on an 80/20 stratified split and scored on the held-out 20%:

| Metric | Value |
|---|---|
| ROC-AUC | **0.872** |
| Average precision | **0.390** (baseline 0.065) |
| Test set | 1,758 passes, 115 positive |

So against a 6.5% base rate the model reaches 0.39 average precision — roughly a 6× lift over guessing. The precision–recall curve below is from this run.

![Precision–recall, Premier League test set](a2-xDA-pass-danger/outputs/pl_pr_curve_GBoost.png)

A player's pass-danger score is then the mean predicted shot-probability over their passes. I trained on the Premier League 2015/16 and scored four other leagues with the same model. La Liga lands where you would hope — the 2017/18 Barcelona front line at the top:

![La Liga 2017/18, top 10 by pass danger](a2-xDA-pass-danger/outputs/laliga_top10_pass_danger_min150.png)

Top of each league (≥150 passes, mean predicted P(shot within 10s)):

| League | Top three |
|---|---|
| Premier League 2015/16 (train) | Christian Fuchs · Bacary Sagna · Charlie Daniels |
| La Liga 2017/18 | Ousmane Dembélé (0.110) · Lionel Messi (0.105) · Luis Suárez (0.103) |
| Bundesliga 2015/16 | Filip Kostić (0.119) · Karim Bellarabi (0.113) · Levin Öztunali (0.110) |
| Serie A 2015/16 | Nicola Sansone (0.115) · Keita Baldé (0.114) · Darko Lazović (0.113) |
| Ligue 1 2015/16 | Yannis Salibur (0.113) · Nabil Fekir (0.111) · Alhassane Bangoura (0.110) |

`outputs/` holds the per-league top-10 bar charts (one per league, with `min150`/`min200` minimum-pass filters), the ranked player CSVs, and the ROC/PR curve plots. There are also a couple of earlier logistic-regression curves under `outputs/_old_lr/` from before I switched to gradient boosting.

**2. Regressor — expected-threat gain per pass** (`fit_xDA_model.py`)

The CLI script is the other take. Here the label is a hand-crafted expected-threat proxy `xT(x, y)` that rises towards the opponent's goal and peaks centrally (a sigmoid in x times a Gaussian in y); the target for a completed pass is `max(0, xT(end) − xT(start))`, and incomplete passes get zero. A `GradientBoostingRegressor` is trained out-of-fold on the Premier League (5-fold `KFold` with `cross_val_predict`) so the per-player scores on the training league are unbiased, then a final model fit on all of it scores the other leagues. Minutes played are estimated from Starting XI and Substitution events, which lets it report xDA per 90.

```bash
python a2-xDA-pass-danger/fit_xDA_model.py \
    --train events_England.json \
    --score events_Spain.json events_Germany.json events_Italy.json events_France.json \
    --out_dir a2-xDA-pass-danger/outputs
```

Dependencies: `numpy`, `pandas`, `scikit-learn`, `matplotlib`, and `mplsoccer` for the pitch plots.

## a1 — pitch exploration

`pitch_exploration.ipynb` is the getting-started step: it loads `competitions.json` into a DataFrame so you can see which competitions and seasons StatsBomb offers before pulling match events. A small first step rather than a finished analysis.

## a3 — off-ball run direction

`hypothesis_testing.ipynb` looks at one player's off-ball runs that end in the final third, using circular statistics. For each run I take the start→end vector and its angle, then summarise the set: the circular mean gives an average run direction and the circular variance (1 − mean resultant length) measures how consistent that direction is. The runs and an average-direction arrow are drawn on a pitch with `mplsoccer`.

Worked example, Phil Foden from one Champions League final (805 tracked runs, 210 after filtering to his forward in-possession runs):

- **13 runs** ending in the final third (end_x ≥ 66.7 on a 0–100 pitch)
- mean direction **−3.47°** (almost straight towards goal)
- circular variance **0.236** (fairly consistent — low variance means a clear, repeatable angle)

![Foden runs into the final third](a3-statistical-testing/foden_runs_final_third.png)
*Foden's 13 off-ball runs into the final third from one Champions League final, drawn on the pitch with the mean-direction vector (bold arrow) — mean −3.47°, circular variance 0.236.*

This part uses a different dataset — tracking-derived runs from the `twelve-respovision-CL-final` parquet files rather than StatsBomb event JSON — so the notebook tries a few local paths and falls back to cloning that repo.

## Data attribution

StatsBomb event data: [StatsBomb Open Data](https://github.com/statsbomb/open-data), used under their free-for-non-commercial-research terms. The a3 run data comes from [twelve-respovision-CL-final](https://github.com/twelvefootball/twelve-respovision-CL-final). No data files are redistributed here.
