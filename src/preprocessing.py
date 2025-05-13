from math import floor

import pandas as pd


def create_field_location(df: pd.DataFrame):
    df["Field_Location"] = df["Env"].str.replace("(_).*", "", regex=True)
    return df


def process_metadata(path: str, encoding: str = "latin-1"):
    df = pd.read_csv(path, encoding=encoding)
    df["City"] = (
        df["City"].str.strip().replace({"College Station, Texas": "College Station"})
    )
    df = df.rename(
        columns={
            "Weather_Station_Latitude (in decimal numbers NOT DMS)": "weather_station_lat",
            "Weather_Station_Longitude (in decimal numbers NOT DMS)": "weather_station_lon",
        }
    )
    df["treatment_not_standard"] = (df["Treatment"] != "Standard").astype("int")
    return df


def process_test_data(path: str):
    df = pd.read_csv(path)
    df = create_field_location(df)
    return df


def lat_lon_to_bin(x, step: float):
    if pd.notnull(x):
        return floor(x / step) * step
    else:
        return x


def fillna_by_loc(df_ref: pd.DataFrame, df: pd.DataFrame, col: str, func: str):
    return df[col].fillna(
        df_ref.groupby("Field_Location")[col].transform(func)
    )


def agg_yield(df: pd.DataFrame):
    df["Year"] = df["Env"].str[-4:].astype("int")
    df_agg = (
        df.groupby(["Env", "Hybrid"])
        .agg(
            weather_station_lat=("weather_station_lat", "mean"),
            weather_station_lon=("weather_station_lon", "mean"),
            treatment_not_standard=("treatment_not_standard", "mean"),
            Field_Location=("Field_Location", "last"),
            Year=("Year", "last"),
            month_planted=("month_planted", "last"),
            Yield_Mg_ha=(
                "Yield_Mg_ha",
                "mean",
            ),  # unadjusted means per env/hybrid combination
        )
        .reset_index()
    )
    return df_agg


def feat_eng_weather(df: pd.DataFrame):
    df["Date"] = pd.to_datetime(df["Date"], format="%Y%m%d")
    df["month"] = df["Date"].dt.month.astype(str)
    df["season"] = (
        df["month"].astype(int) % 12 // 3 + 1
    )  # https://stackoverflow.com/a/44124490/11122513
    df["season"] = df["season"].map({1: "winter", 2: "spring", 3: "summer", 4: "fall"})
    df_agg = df.dropna(
        subset=[x for x in df.columns if x not in ["Env", "Date"]]
    ).copy()
    df_agg = (
        df.groupby(["Env", "season"])
        .agg(
            T2M_max=("T2M", "max"),
            T2M_min=("T2M", "min"),
            T2M_std=("T2M", "std"),
            T2M_mean=("T2M", "mean"),
            T2M_MIN_min=("T2M_MIN", "min"),
            T2M_MIN_max=("T2M_MIN", "max"),
            T2M_MIN_std=("T2M_MIN", "std"),
            T2M_MIN_cv=("T2M_MIN", lambda x: x.std() / x.mean()),
            WS2M_max=("WS2M", "max"),
            WS2M_mean=("WS2M", "mean"),
            RH2M_max=("RH2M", "max"),
            RH2M_p90=("RH2M", lambda x: x.quantile(0.9)),
            QV2M_mean=("QV2M", "mean"),
            PRECTOTCORR_max=("PRECTOTCORR", "max"),
            PRECTOTCORR_median=("PRECTOTCORR", "median"),
            # PRECTOTCORR_n_days_less_10_mm=('PRECTOTCORR', lambda x: sum(x < 10)),
        )
        .reset_index()
        .pivot(index="Env", columns="season")
    )
    df_agg.columns = ["_".join(col) for col in df_agg.columns]
    return df_agg


def feat_eng_soil(df: pd.DataFrame):
    df_agg = df.groupby("Env").agg(
        Nitrate_N_ppm_N=("Nitrate-N ppm N", "mean"),
        lbs_N_A=("lbs N/A", "mean"),
        percentage_Ca_Sat=("%Ca Sat", "mean"),
        percentage_H_Sat=("%H Sat", "mean"),
        percentage_K_Sat=("%K Sat", "mean"),
        percentage_sand=("% Sand", "mean"),
        percentage_silt=("% Silt", "mean"),
        percentage_clay=("% Clay", "mean"),
        Mehlich_P_III_ppm_P=("Mehlich P-III ppm P", "mean"),
    )
    return df_agg


def feat_eng_target(df: pd.DataFrame, ref_year: list, lag: int, group: list):
    assert lag >= 1
    col = f"yield_lag_{lag}_{'_'.join(group)}"
    df_year = df[df["Year"] <= ref_year - lag]
    df_agg = df_year.groupby(group).agg(
        **{f"mean_{col}": ("Yield_Mg_ha", "mean")},
        **{f"min_{col}": ("Yield_Mg_ha", "min")},
        **{f"p1_{col}": ("Yield_Mg_ha", lambda x: x.quantile(0.01))},
        **{f"q1_{col}": ("Yield_Mg_ha", lambda x: x.quantile(0.25))},
        **{f"q3_{col}": ("Yield_Mg_ha", lambda x: x.quantile(0.75))},
        **{f"p90_{col}": ("Yield_Mg_ha", lambda x: x.quantile(0.90))},
    )
    return df_agg


def filter_locs(df: pd.DataFrame, locs: list):
    print("shape before filtering locs:", df.shape)
    print("# locs:", df["Field_Location"].unique().shape)
    df = df[df["Field_Location"].isin(locs)].reset_index(drop=True)
    print("shape after filtering locs:", df.shape)
    print("# locs:", df["Field_Location"].unique().shape)
    return df


def filter_testers(df: pd.DataFrame, testers: list):
    print("shape before filtering testers:", df.shape)
    print("# hybrids:", df["Hybrid"].unique().shape)
    df = df[~df["tester"].isin(testers)].reset_index(drop=True)
    print("shape after filtering testers:", df.shape)
    print("# hybrids:", df["Hybrid"].unique().shape)
    return df


def preprocess_g(df, kinship, individuals: list):
    df.index = df.columns
    df = df[df.index.isin(individuals)]  # filter rows
    df = df[[col for col in df.columns if col in individuals]]  # filter columns
    df.index.name = "Hybrid"
    df.columns = [f"{x}_{kinship}" for x in df.columns]
    return df


def extract_target(df: pd.DataFrame):
    y = df["Yield_Mg_ha"]
    del df["Yield_Mg_ha"]
    return y
