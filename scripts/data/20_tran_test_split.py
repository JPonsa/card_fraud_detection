from datetime import date, datetime, timedelta
import numpy as np
import pandas as pd


# this is adapted from get_train_test_set at
# https://fraud-detection-handbook.github.io/fraud-detection-handbook/Chapter_References/shared_functions.html#get-train-test-set
def get_train_test_set(
    df, start_date_training, delta_train=7, delta_delay=7, delta_test=7, random_state=0
):
    # Get the training set data
    train_df = df[
        (df["TX_DATETIME"] >= start_date_training)
        & (df["TX_DATETIME"] < start_date_training + timedelta(days=delta_train))
    ]

    # Get the test set data
    test_df = []

    # Note: Cards known to be compromised after the delay period are removed from the test set
    # That is, for each test day, all frauds known at (test_day-delay_period) are removed

    # First, get known defrauded customers from the training set
    known_defrauded_customers = set(train_df[train_df["TX_FRAUD"] == 1]["CUSTOMER_ID"])

    # Get the relative starting day of training set (easier than TX_DATETIME to collect test data)
    start_tx_time_days_training = train_df["TX_TIME_DAYS"].min()

    # Then, for each day of the test set
    for day in range(delta_test):
        # Get test data for that day
        test_df_day = df[
            df["TX_TIME_DAYS"]
            == start_tx_time_days_training + delta_train + delta_delay + day
        ]

        # Compromised cards from that test day, minus the delay period, are added to the pool of known defrauded customers
        test_df_day_delay_period = df[
            df["TX_TIME_DAYS"] == start_tx_time_days_training + delta_train + day - 1
        ]

        new_defrauded_customers = set(
            test_df_day_delay_period[test_df_day_delay_period["TX_FRAUD"] == 1][
                "CUSTOMER_ID"
            ]
        )
        known_defrauded_customers = known_defrauded_customers.union(
            new_defrauded_customers
        )

        test_df_day = test_df_day[
            ~test_df_day["CUSTOMER_ID"].isin(known_defrauded_customers)
        ]

        test_df.append(test_df_day)

    test_df = pd.concat(test_df)

    # Sort data sets by ascending order of transaction ID
    train_df = train_df.sort_values("TRANSACTION_ID")
    test_df = test_df.sort_values("TRANSACTION_ID")

    return (train_df, test_df)


def field_trim(df: pd.DataFrame, training_features: list, target: list) -> pd.DataFrame:
    return df[training_features + target]


if __name__ == "__main__":
    preprocessed_path = "./data/preprocessed/"
    preprocessed_df = pd.read_parquet(preprocessed_path + "card_fraud.parquet.gzip")

    split_date = datetime(2018, 7, 25)

    label_columns = ["TX_FRAUD"]
    feature_columns = [
        "TX_AMOUNT",
        "IS_WKD",
        "IS_NIGHT",
        "CUSTOMER_TX_COUNT_1_DAY",
        "CUSTOMER_TX_COUNT_7_DAY",
        "CUSTOMER_TX_COUNT_30_DAY",
        "CUSTOMER_TX_AVG_1_DAY",
        "CUSTOMER_TX_AVG_7_DAY",
        "CUSTOMER_TX_AVG_30_DAY",
        "TERMINAL_TX_COUNT_1_DAY",
        "TERMINAL_TX_COUNT_7_DAY",
        "TERMINAL_TX_COUNT_30_DAY",
        "TERMINAL_FRAUD_RISK_1_DAY",
        "TERMINAL_FRAUD_RISK_7_DAY",
        "TERMINAL_FRAUD_RISK_30_DAY",
    ]

    train_df, test_df = get_train_test_set(preprocessed_df, split_date, delta_train=21)
    train_df, val_df = get_train_test_set(train_df, split_date)

    train_df = field_trim(train_df, feature_columns, label_columns)
    test_df = field_trim(test_df, feature_columns, label_columns)
    val_df = field_trim(val_df, feature_columns, label_columns)

    train_df.to_csv(preprocessed_path + "train.csv")
    test_df.to_csv(preprocessed_path + "test.csv")
    val_df.to_csv(preprocessed_path + "validate.csv")
