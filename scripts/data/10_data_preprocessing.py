"""
Takes the raw simulated data and applies feature engineering adapted from the
using pypark
"""
import os
from pyspark.sql import SparkSession, DataFrame
from pyspark.sql import functions as F
from pyspark.sql.window import Window


def partitionBy_tx_n_days(
    df: DataFrame,
    col_name: str,
    partition_by: str = "CUSTOMER_ID",
    n_days: int = 1,
    func=F.count,
    shift: int = 0,
    analysis_field: str = "TX_AMOUNT",
) -> DataFrame:
    """Takes a spark.DataFrame
        1. Define a windows of analysis
            1. Group by "partition_by" / For each element in partition_by
            2. Order by transaction date (TX_DATETIME)
            3. Define a time window from record's time window to n days before (in seconds)
        2. Apply a function (e.g. sum) to the window of analysis

    Parameters
    ----------
    df : spark.DataFrame
    col_name : str
        where to store the output
    partition_by : str, optional
        Fields to do the partition by / for each element of, by default "CUSTOMER_ID"
    n_days : int, optional
        number of days for the window of analysis, by default 1
    func : _type_, optional
        spark.sql.function, by default F.count
    shift: int, optional
        number of days to shift the time window to the past, by default 0
    analysis_field: str, optional
        field where to apply the function, default "TX_AMOUNT"

    Returns
    -------
    spark.DataFrame
        same df use as input plus a new column with the output of the calculation.
    """
    # Define window to be interrogated
    window_spec = (
        Window.partitionBy(partition_by)
        .orderBy(
            F.col("TX_DATETIME").cast("timestamp").cast("long")
        )  # conver datetime to seconds
        .rangeBetween(-(n_days + shift) * 86400, -shift * 86400)
    )

    return df.withColumn(col_name, func(analysis_field).over(window_spec))


def fraud_risk(
    df: DataFrame, col_name: str, delay: int = 7, n_days: int = 1
) -> DataFrame:
    """Compute Fraud Risk for a time window

    Parameters
    ----------
    df : spark.DataFrame
        input DataFrame
    col_name : str
        Where to save the outcome
    delay : int, optional
        time required to detect a fraudulent transaction based on business logic, by default 7
    n_days : int, optional
        number of days for the time window, by default 1

    Returns
    -------
    spark.DataFrame
        same as the input df + column with the output
    """
    df1 = df.select("*")  # Copy of the input df

    df1 = partitionBy_tx_n_days(
        df1, "fraud_in_window", "TERMINAL_ID", n_days, F.sum, delay, "TX_FRAUD"
    )
    df1 = partitionBy_tx_n_days(
        df1, "tx_in_window", "TERMINAL_ID", n_days, F.count, delay, "TX_FRAUD"
    )

    df1 = df1.withColumn(col_name, F.col("fraud_in_window") / F.col("tx_in_window"))
    df1 = df1.fillna(0, subset=[col_name])

    df = df.join(
        df1.select("TRANSACTION_ID", col_name), on="TRANSACTION_ID", how="left"
    )

    return df


if __name__ == "__main__":
    raw_path = "./data/raw/"
    preprocessed_path = "./data/preprocessed/"
    n_days = [1, 7, 30]

    # start spark session
    spark = (
        SparkSession.builder.config("spark.driver.host", "localhost")
        .config("spark.driver.memory", "16g")
        .getOrCreate()
    )

    # Load raw data
    raw = spark.read.parquet(raw_path + "card_fraud.parquet.gzip")

    # Feature Engineering
    # Adapted from https://fraud-detection-handbook.github.io/fraud-detection-handbook/Chapter_3_GettingStarted/BaselineFeatureTransformation.html#terminal-id-transformations
    preprocessed = raw.select("*")

    # Create 'is_weekend' column
    preprocessed = preprocessed.withColumn("IS_WKD", (F.dayofweek("TX_DATETIME") >= 6))

    # Create 'is_night' column
    preprocessed = preprocessed.withColumn("IS_NIGHT", (F.hour("TX_DATETIME") <= 6))

    # Customert profile
    for f, fname in zip([F.count, F.mean], ["COUNT", "AVG"]):
        for d in n_days:
            field_name = f"CUSTOMER_TX_{fname}_{d}_DAY"
            preprocessed = partitionBy_tx_n_days(
                preprocessed, field_name, "CUSTOMER_ID", d, f
            )

    # Terminal profile
    for d in n_days:
        field_name = f"TERMINAL_TX_COUNT_{d}_DAY"
        preprocessed = partitionBy_tx_n_days(
            preprocessed, field_name, "TERMINAL_ID", d, F.count
        )

    # Termina Fraud risk score
    for d in n_days:
        field_name = f"TERMINAL_FRAUD_RISK_{d}_DAY"
        preprocessed = fraud_risk(preprocessed, field_name, delay=7, n_days=d)

    # Create the destination directory if it doesn't exist
    os.makedirs(preprocessed_path, exist_ok=True)
    # preprocessed.write.mode("overwrite").option("compression", "gzip").parquet(
    #     preprocessed_path + "card_fraud.parquet.gzip"
    # )
    # option("compression", "gzip")
    # TODO: spark give me some problems saving files. I think I have some problems
    # with the installation
    preprocessed = preprocessed.toPandas()
    preprocessed.to_parquet(
        preprocessed_path + "card_fraud.parquet.gzip", compression="gzip"
    )

spark.stop()  # Kill spark session
