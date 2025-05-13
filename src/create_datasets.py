import argparse
import sys

import pandas as pd
from sklearn.decomposition import TruncatedSVD

from preprocessing import (
    process_metadata,
    process_test_data,
    lat_lon_to_bin,
    fillna_by_loc,
    agg_yield,
    feat_eng_weather,
    feat_eng_soil,
    feat_eng_target,
    filter_locs,
    filter_testers,
    preprocess_g,
    extract_target,
    create_field_location,
)


parser = argparse.ArgumentParser()
parser.add_argument("--fl", type=int, required=False)
args = parser.parse_args()

META_COLS = [
    "Env",
    "weather_station_lat",
    "weather_station_lon",
    "treatment_not_standard",
    # "Irrigated"
]
CAT_COLS = ["Env", "Hybrid"]  # to avoid NA imputation
CATEGORICALS = ["Field_Location", "tester"]

LAT_BIN_STEP = 1.2
LON_BIN_STEP = LAT_BIN_STEP * 3

if not (len(sys.argv) > 1):
    FILTER_LOCS = 1
    DEBUG = True
else:
    FILTER_LOCS = args.fl
    DEBUG = False

TRAIN_INIT = 2014
VAL_INIT = 2022
LAG = 1
FILTER_TESTERS = False
USE_FIELD_LOCATION = False
USE_TESTER = False
ADD_LAG = False
ADD_SNPS = True
G = False
REFIT_FOR_SUB = False

if __name__ == "__main__":

    # META
    meta = process_metadata("data/Training_data/2_Training_Meta_Data_2014_2023.csv")
    meta_test = process_metadata("data/Testing_data/2_Testing_Meta_Data_2024.csv")
    meta_test["Date_planted"] = pd.to_datetime(
        meta_test["Date_Planted"], dayfirst=False
    )
    meta_test["month_planted"] = meta_test["Date_planted"].dt.month

    # TEST
    test = process_test_data("data/Testing_data/1_Submission_Template_2024.csv")
    xtest = test.merge(
        meta_test[META_COLS + ["month_planted"]], on="Env", how="left"
    ).drop(["Field_Location"], axis=1)
    xtest = create_field_location(xtest)
    df_sub = xtest.reset_index()[["Env", "Hybrid"]]

    # TRAIT
    trait = pd.read_csv("data/Training_data/1_Training_Trait_Data_2014_2023.csv")
    trait = trait.merge(meta[META_COLS], on="Env", how="left")
    trait = create_field_location(trait)
    trait["Date_Planted"] = pd.to_datetime(trait["Date_Planted"], dayfirst=False)
    trait["month_planted"] = trait["Date_Planted"].dt.month

    # agg yield (unadjusted means)
    trait = agg_yield(trait)

    # WEATHER
    weather = pd.read_csv(
        "data/Training_data/4_Training_Weather_Data_2014_2023_full_year.csv"
    )
    weather_test = pd.read_csv(
        "data/Testing_data/4_Testing_Weather_Data_2024_full_year.csv"
    )

    # SOIL
    soil = pd.read_csv("data/Training_data/3_Training_Soil_Data_2015_2023.csv")
    soil_test = pd.read_csv("data/Testing_data/3_Testing_Soil_Data_2024.csv")

    # EC
    ec = pd.read_csv("data/Training_data/6_Training_EC_Data_2014_2023.csv").set_index(
        "Env"
    )
    ec_test = pd.read_csv("data/Testing_data/6_Testing_EC_Data_2024.csv").set_index(
        "Env"
    )

    # split
    xtrain = trait[trait["Year"] != VAL_INIT].reset_index(drop=True)
    xtrain = xtrain[xtrain["Year"] != 2023].reset_index(
        drop=True
    )  # to respect train/val proportion
    xval = trait[trait["Year"] == VAL_INIT].reset_index(drop=True)
    del xtrain["Year"], xval["Year"]

    # filter locations
    if FILTER_LOCS:
        locs = xtest["Field_Location"].unique()
        print("xtrain:")
        xtrain = filter_locs(xtrain, locs)
        # print("xval:")
        # xval = filter_locs(xval, locs)

    # filter testers
    trait["tester"] = trait["Hybrid"].str.replace(".*/", "", regex=True)
    xtrain["tester"] = xtrain["Hybrid"].str.replace(".*/", "", regex=True)
    xval["tester"] = xval["Hybrid"].str.replace(".*/", "", regex=True)
    xtest["tester"] = xtest["Hybrid"].str.replace(".*/", "", regex=True)
    if FILTER_TESTERS:
        testers = ["PHT69", "DK3IIH6"]
        print("\nxtrain:")
        xtrain = filter_testers(xtrain, testers)
        print("xval:")
        xval = filter_testers(xval, testers)

    # remove NA phenotype
    xtrain = xtrain[~xtrain["Yield_Mg_ha"].isnull()].reset_index(drop=True)
    xval = xval[~xval["Yield_Mg_ha"].isnull()].reset_index(drop=True)
    # xtest = xtest[~xtest['Yield_Mg_ha'].isnull()].reset_index(drop=True)

    # replace unadjusted means by BLUEs
    # not used since we are just using unadjusted means as an example here
    # blues = pd.read_csv("output/blues.csv")
    # xtrain = xtrain.merge(blues, on=["Env", "Hybrid"], how="left")
    # xtrain = process_blues(xtrain)
    # xval = xval.merge(blues, on=["Env", "Hybrid"], how="left")
    # xval = process_blues(xval)

    # feat eng (weather)
    weather_feats = feat_eng_weather(weather)
    weather_test_feats = feat_eng_weather(weather_test)
    xtrain = xtrain.merge(weather_feats, on="Env", how="left")
    xval = xval.merge(weather_feats, on="Env", how="left")
    xtest = xtest.merge(weather_test_feats, on="Env", how="left")

    # feat eng (soil)
    xtrain = xtrain.merge(feat_eng_soil(soil), on="Env", how="left")
    xval = xval.merge(feat_eng_soil(soil), on="Env", how="left")
    xtest = xtest.merge(feat_eng_soil(soil_test), on="Env", how="left")

    # feat eng (EC)
    xtrain_ec = ec[ec.index.isin(xtrain["Env"])].copy()
    xval_ec = ec[ec.index.isin(xval["Env"])].copy()
    xtest_ec = ec_test[ec_test.index.isin(xtest["Env"])].copy()

    n_components = 10
    svd = TruncatedSVD(n_components=n_components, n_iter=20, random_state=42)
    svd.fit(xtrain_ec)
    print("SVD explained variance:", svd.explained_variance_ratio_.sum())

    xtrain_ec = pd.DataFrame(svd.transform(xtrain_ec), index=xtrain_ec.index)
    component_cols = [f"EC_svd_comp{i}" for i in range(xtrain_ec.shape[1])]
    xtrain_ec.columns = component_cols
    xval_ec = pd.DataFrame(
        svd.transform(xval_ec), columns=component_cols, index=xval_ec.index
    )
    xtest_ec = pd.DataFrame(
        svd.transform(xtest_ec), columns=component_cols, index=xtest_ec.index
    )

    xtrain = xtrain.merge(xtrain_ec, on="Env", how="left")
    xval = xval.merge(xval_ec, on="Env", how="left")
    xtest = xtest.merge(xtest_ec, on="Env", how="left")

    # feat eng (target)
    xtrain = create_field_location(xtrain)
    xval = create_field_location(xval)
    xtest = create_field_location(xtest)
    if ADD_LAG:
        xtrain = xtrain.merge(
            feat_eng_target(
                trait, ref_year=TRAIN_INIT, lag=LAG, group=["Field_Location"]
            ),
            on="Field_Location",
            how="left",
        )
        xval = xval.merge(
            feat_eng_target(
                trait, ref_year=VAL_INIT, lag=LAG, group=["Field_Location"]
            ),
            on="Field_Location",
            how="left",
        )
        xtest = xtest.merge(
            feat_eng_target(trait, ref_year=2024, lag=LAG, group=["Field_Location"]),
            on="Field_Location",
            how="left",
        )

    # factor variables
    xtrain[CATEGORICALS] = xtrain[CATEGORICALS].astype("category")
    xval[CATEGORICALS] = xval[CATEGORICALS].astype("category")
    xtest[CATEGORICALS] = xtest[CATEGORICALS].astype("category")

    # fill lat/lon NAs using values from previous years
    for col in ["weather_station_lat", "weather_station_lon"]:
        xtrain[col] = fillna_by_loc(xtrain, xtrain, col, "max")
        xval[col] = fillna_by_loc(xtrain, xval, col, "max")
        xtest[col] = fillna_by_loc(xtrain, xtest, col, "max")

    # weather-location interaction and lat/lon binning
    # binning lat/lon seems to help reducing noise
    for dfs in [xtrain, xval, xtest]:
        dfs["weather_station_lat"] = dfs["weather_station_lat"].apply(
            lambda x: lat_lon_to_bin(x, LAT_BIN_STEP)
        )
        dfs["weather_station_lon"] = dfs["weather_station_lon"].apply(
            lambda x: lat_lon_to_bin(x, LON_BIN_STEP)
        )

    # print('lat/lon unique bins:')
    # print('lat:', sorted(set(xtrain['weather_station_lat'].unique())))
    # print('lon:', sorted(set(xtrain['weather_station_lon'].unique())))

    # add GRM
    if G:
        grm = pd.read_csv("output/G.csv")
        individuals = (
            xtrain["Hybrid"].unique().tolist()
            + xval["Hybrid"].unique().tolist()
            + xtest["Hybrid"].unique().tolist()
        )
        individuals = list(
            dict.fromkeys(individuals)
        )  # take unique but preserves order (python 3.7+)
        G = preprocess_g(grm, "G", individuals)
        xtrain = pd.merge(xtrain, G, on="Hybrid", how="left")
        xval = pd.merge(xval, G, on="Hybrid", how="left")
        xtest = pd.merge(xtest, G, on="Hybrid", how="left")

    # check hybrids overlapping
    print(
        "Hybrid - val to train ratio:",
        len(set(xval["Hybrid"])) / len(set(xtrain["Hybrid"])),
    )
    print(
        "Hybrid - test to train ratio:",
        len(set(xtest["Hybrid"])) / len(set(xtrain["Hybrid"])),
    )

    # set index
    xtrain = xtrain.set_index(["Env", "Hybrid"])
    xval = xval.set_index(["Env", "Hybrid"])
    xtest = xtest.set_index(["Env", "Hybrid"])

    # keep all same columns
    xtrain = xtrain[xtest.columns]
    xval = xval[xtest.columns]

    # extract targets
    ytrain = extract_target(xtrain)
    yval = extract_target(xval)
    _ = extract_target(xtest)

    print(xtrain.isnull().sum() / len(xtrain))
    print(xval.isnull().sum() / len(xval))
    print(xtest.isnull().sum() / len(xtest))

    # NA imputing
    for col in [x for x in xtrain.columns if x not in CAT_COLS + CATEGORICALS]:
        filler = xtrain[col].mean()
        xtrain[col] = xtrain[col].fillna(filler)
        xval[col] = xval[col].fillna(filler)
        xtest[col] = xtest[col].fillna(filler)

    print("xtrain shape:", xtrain.shape)
    print("xval shape:", xval.shape)
    print("xtest shape:", xtest.shape)
    print("ytrain shape:", ytrain.shape)
    print("yval shape:", yval.shape)
    print("ytrain nulls:", ytrain.isnull().sum() / len(ytrain))
    print("yval nulls:", yval.isnull().sum() / len(yval))

    assert xtrain.index.names == ["Env", "Hybrid"]
    assert xval.index.names == ["Env", "Hybrid"]

    if not USE_FIELD_LOCATION:
        del xtrain["Field_Location"], xval["Field_Location"], xtest["Field_Location"]

    if not USE_TESTER:
        del xtrain["tester"], xval["tester"], xtest["tester"]

    # drop some columns
    cols_drop = ["treatment_not_standard", "month_planted"]
    for col in cols_drop:
        del xtrain[col]
        del xval[col]
        del xtest[col]

    if ADD_SNPS:
        geno = pd.read_csv("output/geno_ok.csv").rename(columns={"name": "Hybrid"})
        xtrain = (
            xtrain.reset_index()
            .merge(geno, on="Hybrid", how="left")
            .set_index(["Env", "Hybrid"])
        )
        xval = (
            xval.reset_index()
            .merge(geno, on="Hybrid", how="left")
            .set_index(["Env", "Hybrid"])
        )
        xtest = (
            xtest.reset_index()
            .merge(geno, on="Hybrid", how="left")
            .set_index(["Env", "Hybrid"])
        )

    # overlapping of hybrids
    htrain = set(xtrain.index.get_level_values(1))
    hval = set(xval.index.get_level_values(1))
    htest = set(xtest.index.get_level_values(1))
    print("# htrain:", len(htrain))
    print("# hval:", len(hval))
    print("# htest:", len(htest))
    print("# train Inter val:", len(htrain & hval))
    print("# train Inter test:", len(htrain & htest))
    print("# val Inter test:", len(hval & htest))
    print("# (train Union val) Inter test:", len((htrain | hval) & htest))

    # write datasets
    if not DEBUG:
        xtrain.reset_index().to_csv(f"output/xtrain_fl_{FILTER_LOCS}.csv", index=False)
        xval.reset_index().to_csv(f"output/xval_fl_{FILTER_LOCS}.csv", index=False)
        xtest.reset_index().to_csv(f"output/xtest_fl_{FILTER_LOCS}.csv", index=False)
        ytrain.reset_index().to_csv(f"output/ytrain_fl_{FILTER_LOCS}.csv", index=False)
        yval.reset_index().to_csv(f"output/yval_fl_{FILTER_LOCS}.csv", index=False)
