# Exoplanet Classification

## Project Overview

This project focuses on the binary classification of Kepler Objects of Interest (KOIs) as either confirmed exoplanets or false-positive signals.

The data comes from the NASA Exoplanet Archive and contains parameters derived from observations made by the Kepler Space Telescope. The project is intended as an educational data science and machine learning exercise covering data acquisition, exploratory data analysis, preprocessing, model training, evaluation, and interpretation.

At the current stage, the project environment and repository structure have been prepared, and the dataset can be downloaded directly from the NASA Exoplanet Archive.

## Project Objective

The main objective is to build a machine learning model that predicts whether a Kepler Object of Interest is:

- `CONFIRMED` — a confirmed or validated exoplanet,
- `FALSE POSITIVE` — a signal that was determined not to be caused by an exoplanet.

Objects classified as `CANDIDATE` are excluded because their final status is still uncertain.

The target variable is:

```text
koi_disposition
```

It will later be encoded as:

```text
CONFIRMED = 1
FALSE POSITIVE = 0
```

## Dataset

The dataset is obtained from the **NASA Exoplanet Archive**, from the **Kepler Objects of Interest cumulative table**.

A Kepler Object of Interest is a periodic signal detected in the brightness measurements of a star. Such a signal may indicate a planetary transit, during which a planet passes in front of its host star and causes a temporary decrease in observed brightness.

However, similar signals may also be caused by:

- eclipsing binary stars,
- nearby stellar systems,
- stellar variability,
- instrumental effects,
- data-processing artifacts.

Each row in the dataset represents one KOI signal. A single star may be associated with multiple KOI entries if several periodic signals were detected around it.

## Features

| Column | Description | Unit |
|---|---|---|
| `kepoi_name` | Unique identifier of the Kepler Object of Interest | Text |
| `koi_disposition` | Final classification of the object | Category |
| `koi_period` | Time between consecutive transits, approximately equal to the orbital period | Days |
| `koi_duration` | Duration of a single transit event | Hours |
| `koi_depth` | Decrease in stellar brightness during the transit | Parts per million |
| `koi_impact` | Distance between the transit path and the center of the stellar disk, normalized by the stellar radius | Dimensionless |
| `koi_prad` | Estimated radius of the object | Earth radii |
| `koi_teq` | Estimated equilibrium temperature of the object | Kelvin |
| `koi_insol` | Amount of stellar radiation received relative to Earth | Earth flux |
| `koi_model_snr` | Signal-to-noise ratio of the fitted transit model | Dimensionless |

### Feature Groups

- **Identifier:** `kepoi_name`
- **Target variable:** `koi_disposition`
- **Transit parameters:** `koi_period`, `koi_duration`, `koi_depth`, `koi_impact`
- **Estimated physical properties:** `koi_prad`, `koi_teq`, `koi_insol`
- **Signal quality:** `koi_model_snr`

The `kepoi_name` column is used only as an identifier and will not be included as a model input feature.

