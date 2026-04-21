"""Merge NHANES cycle J files into a cleaned dataset for the lifestyle model.

Pulls demographics, anthropometrics, diabetes status, and lifestyle
questionnaire responses. Derives composite features (total physical
activity minutes/week, smoking category) rather than exposing raw
NHANES column names that users couldn't reasonably answer in a form.
"""

import os

import numpy as np
import pandas as pd


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
NHANES_DIR = os.path.join(BASE_DIR, "data", "NHANES")
OUTPUT_PATH = os.path.join(BASE_DIR, "data", "nhanes_cleaned_lifestyle.csv")

# NHANES cycle J = 2017-2018 survey.
FILES = {
    "demo": "DEMO_J.xpt",
    "bmx":  "BMX_J.xpt",
    "diq":  "DIQ_J.xpt",
    "paq":  "PAQ_J.xpt",
    "dbq":  "DBQ_J.xpt",
    "mcq":  "MCQ_J.xpt",
    "smq":  "SMQ_J.xpt",
    "alq":  "ALQ_J.xpt",
    "slq":  "SLQ_J.xpt",
}

# NHANES questionnaire "refused / don't know" codes that must be
# treated as missing, not as real values.
MISSING_CODES = [7, 9, 77, 99, 777, 999, 7777, 9777, 9999]


def load_xpt(name):
    return pd.read_sas(os.path.join(NHANES_DIR, FILES[name]), format="xport")


def safe(df, col):
    """Return df[col] if it exists, else all-NaN series aligned to df."""
    if col in df.columns:
        return df[col]
    return pd.Series(np.nan, index=df.index)


def clean_numeric(s):
    """Replace NHANES refusal/don't-know sentinels with NaN."""
    return s.replace(MISSING_CODES, np.nan)


def derive_activity_minutes(paq):
    """Total moderate + vigorous activity minutes per week.

    PAQ_J records activity across five domains: vigorous work, moderate
    work, active transport, vigorous recreation, moderate recreation.
    Each has a days/week and a minutes/day column.
    """
    def component(days_col, mins_col):
        days = clean_numeric(safe(paq, days_col))
        mins = clean_numeric(safe(paq, mins_col))
        return (days * mins).fillna(0)

    total = (
        component("PAQ610", "PAQ615")   # vigorous work
        + component("PAQ625", "PAQ630") # moderate work
        + component("PAQ640", "PAQ645") # active transport
        + component("PAQ655", "PAQ660") # vigorous recreation
        + component("PAQ670", "PAQ675") # moderate recreation
    )
    # Cap implausible self-reports (>5000 min/wk = >11 hrs/day).
    return total.clip(upper=5000)


def derive_sedentary_hours(paq):
    """PAD680 is sedentary minutes/day. Convert to hours."""
    mins = clean_numeric(safe(paq, "PAD680"))
    return (mins / 60).clip(upper=24)


def derive_smoking_status(smq):
    """0 = never, 1 = former, 2 = current. NaN if indeterminate."""
    ever = clean_numeric(safe(smq, "SMQ020"))  # 1=yes, 2=no
    now = clean_numeric(safe(smq, "SMQ040"))   # 1=daily, 2=some, 3=not at all
    status = pd.Series(np.nan, index=smq.index)
    status[ever == 2] = 0
    status[(ever == 1) & (now == 3)] = 1
    status[(ever == 1) & (now.isin([1, 2]))] = 2
    return status


def derive_alcohol_per_week(alq):
    """Approximate drinks/week from ALQ121 (frequency) × ALQ130 (amount).

    ALQ121 categories: 0=never, 1=daily, 2=nearly daily, 3=3-4/wk,
    4=2/wk, 5=1/wk, 6=2-3/mo, 7=1/mo, 8=7-11/yr, 9=3-6/yr, 10=1-2/yr.
    We map each to an approximate drinking-days-per-week rate.
    """
    freq = clean_numeric(safe(alq, "ALQ121"))
    per_day = clean_numeric(safe(alq, "ALQ130"))
    freq_to_dayspwk = {
        0: 0, 1: 7, 2: 5.5, 3: 3.5, 4: 2, 5: 1,
        6: 0.58, 7: 0.23, 8: 0.17, 9: 0.09, 10: 0.03,
    }
    days = freq.map(freq_to_dayspwk)
    drinks = (days * per_day).fillna(0)
    return drinks.clip(upper=50)


def derive_sleep_hours(slq):
    """Weekday usual sleep duration."""
    return clean_numeric(safe(slq, "SLD012")).clip(lower=0, upper=24)


def derive_diet_quality(dbq):
    """DBQ700: self-rated diet quality, 1=excellent .. 5=poor.

    Inverted so higher = healthier, matching intuition in a UI.
    """
    raw = clean_numeric(safe(dbq, "DBQ700"))
    return 6 - raw  # 5=excellent, 1=poor


def derive_meals_out_per_week(dbq):
    """DBD895: # meals not prepared at home in past 7 days."""
    return clean_numeric(safe(dbq, "DBD895")).clip(upper=21)


def derive_family_history(mcq):
    """MCQ300C: close biological relative had diabetes. 0=no, 1=yes."""
    raw = clean_numeric(safe(mcq, "MCQ300C"))
    return raw.map({1: 1, 2: 0})


def main():
    print("Loading NHANES cycle J files...")
    demo = load_xpt("demo")
    bmx = load_xpt("bmx")
    diq = load_xpt("diq")
    paq = load_xpt("paq")
    dbq = load_xpt("dbq")
    mcq = load_xpt("mcq")
    smq = load_xpt("smq")
    alq = load_xpt("alq")
    slq = load_xpt("slq")

    # Base frame: everyone with demographics + anthropometrics + diabetes label.
    df = (
        demo[["SEQN", "RIDAGEYR", "RIAGENDR", "RIDRETH3"]]
        .merge(bmx[["SEQN", "BMXBMI", "BMXWAIST"]], on="SEQN", how="inner")
        .merge(diq[["SEQN", "DIQ010"]], on="SEQN", how="inner")
    )

    # Diabetes label: keep only yes/no, match clinical pathway for comparability.
    df = df[df["DIQ010"].isin([1, 2])]
    df["Diabetes"] = df["DIQ010"].map({1: 1, 2: 0}).astype(int)

    # Derive lifestyle features. Each derive_* returns a Series keyed by
    # the source df's index, so we merge on SEQN to align.
    lifestyle_frames = [
        (paq, "ActivityMinutes", derive_activity_minutes),
        (paq, "SedentaryHours", derive_sedentary_hours),
        (smq, "SmokingStatus", derive_smoking_status),
        (alq, "AlcoholPerWeek", derive_alcohol_per_week),
        (slq, "SleepHours", derive_sleep_hours),
        (dbq, "DietQuality", derive_diet_quality),
        (dbq, "MealsOutPerWeek", derive_meals_out_per_week),
        (mcq, "FamilyHistory", derive_family_history),
    ]
    for source, name, fn in lifestyle_frames:
        tmp = source[["SEQN"]].copy()
        tmp[name] = fn(source).values
        df = df.merge(tmp, on="SEQN", how="left")

    # Rename anthropometric columns to user-facing names.
    df = df.rename(columns={
        "RIDAGEYR": "Age",
        "RIAGENDR": "Sex",
        "RIDRETH3": "Ethnicity",
        "BMXBMI": "BMI",
        "BMXWAIST": "WaistCM",
    })
    df["Sex"] = df["Sex"].map({1: 0, 2: 1})  # 0=Male, 1=Female

    # Sanity filters — keep adults with physiologically plausible values.
    df = df[
        (df["Age"].between(18, 80))
        & (df["BMI"].between(13, 70))
        & (df["WaistCM"].between(40, 200) | df["WaistCM"].isna())
    ]

    final_cols = [
        "Age", "Sex", "Ethnicity", "BMI", "WaistCM",
        "ActivityMinutes", "SedentaryHours", "SmokingStatus",
        "AlcoholPerWeek", "SleepHours", "DietQuality",
        "MealsOutPerWeek", "FamilyHistory", "Diabetes",
    ]
    df = df[final_cols]

    # Report missingness per feature — useful when writing up limitations.
    miss = df.drop(columns=["Diabetes"]).isna().mean().sort_values(ascending=False)
    print("\nMissingness per feature:")
    for col, frac in miss.items():
        print(f"  {col:20s} {frac:.1%}")

    # Drop rows missing the target or with catastrophically incomplete features.
    df = df.dropna(subset=["Age", "Sex", "Ethnicity", "BMI", "Diabetes"])
    print(f"\nFinal shape: {df.shape}")
    print(f"Positive class rate: {df['Diabetes'].mean():.2%}")

    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
    df.to_csv(OUTPUT_PATH, index=False)
    print(f"Saved {OUTPUT_PATH}")


if __name__ == "__main__":
    main()