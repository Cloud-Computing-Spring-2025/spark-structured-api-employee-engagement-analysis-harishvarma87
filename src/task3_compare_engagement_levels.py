from pyspark.sql import SparkSession
from pyspark.sql.functions import col, when, avg, round as spark_round

def initialize_spark(app_name="Task3_Compare_Engagement_Levels"):
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
    """
    schema = "EmployeeID INT, Department STRING, JobTitle STRING, SatisfactionRating INT, EngagementLevel STRING, ReportsConcerns BOOLEAN, ProvidedSuggestions BOOLEAN"
    
    df = spark.read.csv(file_path, header=True, schema=schema)
    return df

def map_engagement_level(df):
    """
    Map EngagementLevel from categorical to numerical values.
    """
    df = df.withColumn("EngagementScore", 
                       when(col("EngagementLevel") == "Low", 1)
                       .when(col("EngagementLevel") == "Medium", 2)
                       .when(col("EngagementLevel") == "High", 3))
    return df

def compare_engagement_levels(df):
    """
    Compare engagement levels across different job titles and identify the top-performing job title.
    """
    # Group by JobTitle and calculate average EngagementScore
    engagement_by_jobtitle = df.groupBy("JobTitle").agg(
        spark_round(avg("EngagementScore"), 2).alias("AvgEngagementLevel")
    )
    
    return engagement_by_jobtitle

def write_output(result_df, output_path):
    """
    Write the result DataFrame to a CSV file.
    """
    result_df.coalesce(1).write.csv(output_path, header=True, mode='overwrite')

def main():
    """
    Main function to execute Task 3.
    """
    # Initialize Spark
    spark = initialize_spark()
    
    # Define file paths
    input_file = "/workspaces/spark-structured-api-employee-engagement-analysis-harishvarma87/input/employee_data.csv"
    output_file = "/workspaces/spark-structured-api-employee-engagement-analysis-harishvarma87/outputs/task3/engagement_levels_job_titles.csv"
    
    # Load data
    df = load_data(spark, input_file)
    
    # Perform Task 3
    df_mapped = map_engagement_level(df)
    result_df = compare_engagement_levels(df_mapped)
    
    # Write the result to CSV
    write_output(result_df, output_file)
    
    # Stop Spark Session
    spark.stop()

if __name__ == "__main__":
    main()