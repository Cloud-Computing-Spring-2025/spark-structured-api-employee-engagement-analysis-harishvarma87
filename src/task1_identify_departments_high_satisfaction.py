from pyspark.sql import SparkSession
from pyspark.sql.functions import col, count, when, round as spark_round, lit

def initialize_spark(app_name="Task1_Identify_Departments"):
    """
    Initialize and return a SparkSession.
    """
    spark = SparkSession.builder \
        .appName(app_name) \
        .getOrCreate()
    return spark

def load_data(spark, file_path):
    """
    Load the employee data from a CSV file into a Spark DataFrame.

    Parameters:
        spark (SparkSession): The SparkSession object.
        file_path (str): Path to the employee_data.csv file.

    Returns:
        DataFrame: Spark DataFrame containing employee data.
    """
    schema = "EmployeeID INT, Department STRING, JobTitle STRING, SatisfactionRating INT, EngagementLevel STRING, ReportsConcerns BOOLEAN, ProvidedSuggestions BOOLEAN"
    
    df = spark.read.csv(file_path, header=True, schema=schema)
    return df

def identify_departments_high_satisfaction(df):
    """
    Identify departments where more than 50% of employees have SatisfactionRating > 4 and EngagementLevel = 'High'.

    Parameters:
        df (DataFrame): Spark DataFrame containing employee data.

    Returns:
        DataFrame: DataFrame with departments exceeding the 50% threshold and their respective percentages.
    """
    # Filter employees with SatisfactionRating > 4 and EngagementLevel = 'High'
    filtered_df = df.filter((col("SatisfactionRating") > 4) & (col("EngagementLevel") == "High"))

    # Count total employees per department
    department_counts = df.groupBy("Department").agg(count("*").alias("TotalEmployees"))

    # Count high-satisfaction employees per department
    high_satisfaction_counts = filtered_df.groupBy("Department").agg(count("*").alias("HighSatEmployees"))

    # Join both counts and calculate percentage
    result_df = department_counts.join(high_satisfaction_counts, "Department", "left_outer") \
        .fillna(0, subset=["HighSatEmployees"]) \
        .withColumn("Percentage", spark_round((col("HighSatEmployees") / col("TotalEmployees")) * 100, 2)) \
        .filter(col("Percentage") > 5) \
        .select("Department", "Percentage")

    return result_df

def write_output(result_df, output_path):
    """
    Write the result DataFrame to a CSV file.

    Parameters:
        result_df (DataFrame): Spark DataFrame containing the result.
        output_path (str): Path to save the output CSV file.

    Returns:
        None
    """
    if result_df.count() > 0:
        result_df.coalesce(1).write.csv(output_path, header=True, mode='overwrite')
    else:
        print("No departments met the criteria of >50% high satisfaction and engagement.")

def main():
    """
    Main function to execute Task 1.
    """
    # Initialize Spark
    spark = initialize_spark()
    
    # Define file paths
    input_file = "/workspaces/spark-structured-api-employee-engagement-analysis-harishvarma87/input/employee_data.csv"
    output_file = "/workspaces/spark-structured-api-employee-engagement-analysis-harishvarma87/outputs/task1/departments_high_satisfaction.csv"
    
    # Load data
    df = load_data(spark, input_file)
    
    # Perform Task 1
    result_df = identify_departments_high_satisfaction(df)
    
    # Write the result to CSV
    write_output(result_df, output_file)
    
    # Stop Spark Session
    spark.stop()

if __name__ == "__main__":
    main()