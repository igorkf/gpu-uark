import pandas as pd
import lightgbm as lgbm
from sklearn.metrics import root_mean_squared_error

from preprocessing import create_field_location


def prep_features(df):
    df = create_field_location(df)
    df["Field_Location"] = df["Field_Location"].astype("category")
    df = df.set_index(["Env", "Hybrid"])
    return df


if __name__ == "__main__":

    # load features
    xtrain = pd.read_csv("output/xtrain_fl_0.csv")
    xtrain = prep_features(xtrain)
    xval = pd.read_csv("output/xval_fl_0.csv")
    xval = prep_features(xval)

    # load targets
    ytrain = pd.read_csv("output/ytrain_fl_0.csv").set_index(["Env", "Hybrid"])
    yval = pd.read_csv("output/yval_fl_0.csv").set_index(["Env", "Hybrid"])

    # hyperparameters
    params = {
        "lambda_l1": 0.00019843354389067637,
        "lambda_l2": 2.0952311071236456e-05,
        "num_leaves": 81,
        "feature_fraction": 0.6702415073506178,
        "bagging_fraction": 0.7580106563644224,
        "bagging_freq": 2,
        "min_child_samples": 90,
    }

    # fit model
    model = lgbm.LGBMRegressor(
        **params,
        random_state=42,
        max_depth=5,
        n_estimators=100,
        verbose=-1,
    )
    model.fit(xtrain, ytrain)

    # predict
    ypred_train = model.predict(xtrain)
    ypred = model.predict(xval)

    # calculate importance
    df_imp = pd.DataFrame()
    df_imp["score"] = model.feature_importances_
    df_imp["name"] = model.feature_name_
    df_imp = df_imp.sort_values("score", ascending=[False], ignore_index=True)
    print("df imp (top 20):")
    print(df_imp.head(20))
    df_imp.to_csv("output/importance_lgbm.csv", index=False)

    # evaluate
    xval["ytrue"] = yval
    xval["ypred"] = ypred
    rmse = root_mean_squared_error(xval["ytrue"], xval["ypred"])
    corr = (
        xval[["ytrue", "ypred"]]
        .groupby("Env")
        .corr()
        .iloc[::2, 1]
        .droplevel(1)
        .sort_values()
    )
    print("yval std:", float(yval.std()))
    print("global rmse:", float(rmse))
    print("env corr:", corr)
    print("global corr:", float(xval[["ytrue", "ypred"]].corr().iloc[1, 0]))
    print("mean env corr:", float(corr.mean()))
    print("mean env corr summary:\n", corr.describe())

    # check distribution train/val/test
    dist_pred = pd.DataFrame()
    dist_pred["train"] = pd.DataFrame(ypred_train).describe()
    dist_pred["val"] = pd.DataFrame(ypred).describe()
    print(dist_pred.round(3))

    # write predictions
    xval[["ytrue", "ypred"]].reset_index().to_csv("output/pred_lgbm.csv", index=False)
