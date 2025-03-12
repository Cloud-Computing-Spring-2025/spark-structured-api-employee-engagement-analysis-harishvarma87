# Employee Engagement Analysis Assignment

## **Overview**

In this assignment, you will leverage Spark Structured APIs to analyze a dataset containing employee information from various departments within an organization. Your goal is to extract meaningful insights related to employee satisfaction, engagement, concerns, and job titles. This exercise is designed to enhance your data manipulation and analytical skills using Spark's powerful APIs.

## **Objectives**

By the end of this assignment, you should be able to:

1. **Data Loading and Preparation**: Import and preprocess data using Spark Structured APIs.
2. **Data Analysis**: Perform complex queries and transformations to address specific business questions.
3. **Insight Generation**: Derive actionable insights from the analyzed data.

## **Dataset**

### **Employee Data (`employee_data.csv`)**

You will work with a dataset containing information about 100 employees across various departments. The dataset includes the following columns:

| Column Name             | Data Type | Description                                           |
|-------------------------|-----------|-------------------------------------------------------|
| **EmployeeID**          | Integer   | Unique identifier for each employee                   |
| **Department**          | String    | Department where the employee works (e.g., Sales, IT) |
| **JobTitle**            | String    | Employee's job title (e.g., Manager, Executive)      |
| **SatisfactionRating**  | Integer   | Employee's satisfaction rating (1 to 5)               |
| **EngagementLevel**     | String    | Employee's engagement level (Low, Medium, High)       |
| **ReportsConcerns**     | Boolean   | Indicates if the employee has reported concerns       |
| **ProvidedSuggestions** | Boolean   | Indicates if the employee has provided suggestions    |

### **Sample Data**

Below is a snippet of the `employee_data.csv` to illustrate the data structure. Ensure your dataset contains at least 100 records for meaningful analysis.

```
EmployeeID,Department,JobTitle,SatisfactionRating,EngagementLevel,ReportsConcerns,ProvidedSuggestions
1,Sales,Manager,5,High,False,True
2,IT,Developer,3,Low,True,False
3,HR,Executive,4,High,False,True
4,Sales,Executive,2,Low,True,False
5,IT,Manager,5,High,False,True
```

## **Assignment Tasks**

You are required to complete the following three analysis tasks using Spark Structured APIs. Ensure that your analysis is well-documented, with clear explanations and any relevant visualizations or summaries.

### **1. Identify Departments with High Satisfaction and Engagement**

**Objective:**

Determine which departments have more than 50% of their employees with a Satisfaction Rating greater than 4 and an Engagement Level of 'High'.

**Tasks:**

- **Filter Employees**: Select employees who have a Satisfaction Rating greater than 4 and an Engagement Level of 'High'.
- **Analyze Percentages**: Calculate the percentage of such employees within each department.
- **Identify Departments**: List departments where this percentage exceeds 50%.

**Expected Outcome:**

A list of departments meeting the specified criteria, along with the corresponding percentages.

**Example Output:**

| Department | Percentage |
|------------|------------|
| Finance    | 60%        |
| Marketing  | 55%        |

---

### **2. Who Feels Valued but Didn’t Suggest Improvements?**

**Objective:**

Identify employees who feel valued (defined as having a Satisfaction Rating of 4 or higher) but have not provided suggestions. Assess the significance of this group within the organization and explore potential reasons for their behavior.

**Tasks:**

- **Identify Valued Employees**: Select employees with a Satisfaction Rating of 4 or higher.
- **Filter Non-Contributors**: Among these, identify those who have `ProvidedSuggestions` marked as `False`.
- **Calculate Proportion**: Determine the number and proportion of these employees relative to the entire workforce.

**Expected Outcome:**

Insights into the number and proportion of employees who feel valued but aren’t providing suggestions.

**Example Output:**

```
Number of Employees Feeling Valued without Suggestions: 25
Proportion: 25%
```

---

### **3. Compare Engagement Levels Across Job Titles**

**Objective:**

Examine how Engagement Levels vary across different Job Titles and identify which Job Title has the highest average Engagement Level.

**Tasks:**

- **Map Engagement Levels**: Convert categorical Engagement Levels ('Low', 'Medium', 'High') to numerical values to facilitate calculation.
- **Group and Calculate Averages**: Group employees by Job Title and compute the average Engagement Level for each group.
- **Identify Top Performer**: Determine which Job Title has the highest average Engagement Level.

**Expected Outcome:**

A comparative analysis showing average Engagement Levels across Job Titles, highlighting the top-performing Job Title.

**Example Output:**

| JobTitle    | AvgEngagementLevel |
|-------------|--------------------|
| Manager     | 4.5                |
| Executive   | 4.2                |
| Developer   | 3.8                |
| Analyst     | 3.5                |
| Coordinator | 3.0                |
| Support     | 2.8                |

---

## **Code**

Here are the code files

Task 1:

```python
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
```

output - [output](./outputs/task1/departments_high_satisfaction.csv/part-00000-123eb0fa-a40d-4697-b66c-0ed7cecc29d0-c000.csv)

Task 2
```python
# task2_valued_no_suggestions.py

from pyspark.sql import SparkSession
from pyspark.sql.functions import col, count

def initialize_spark(app_name="Task2_Valued_No_Suggestions"):
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

def identify_valued_no_suggestions(df):
    """
    Find employees who feel valued but have not provided suggestions and calculate their proportion.

    Parameters:
        df (DataFrame): Spark DataFrame containing employee data.

    Returns:
        tuple: Number of such employees and their proportion.
    """
    valued_no_suggestions_df = df.filter((col("SatisfactionRating") >= 4) & (col("ProvidedSuggestions") == False))
    
    # Count the number of such employees
    num_valued_no_suggestions = valued_no_suggestions_df.count()
    
    # Calculate the proportion of these employees in the entire workforce
    total_employees = df.count()
    proportion = (num_valued_no_suggestions / total_employees) * 100
    
    return num_valued_no_suggestions, round(proportion, 2)
    # TODO: Implement Task 2
    # Steps:
    # 1. Identify employees with SatisfactionRating >= 4.
    # 2. Among these, filter those with ProvidedSuggestions == False.
    # 3. Calculate the number and proportion of these employees.
    # 4. Return the results.

    pass  # Remove this line after implementing the function

def write_output(number, proportion, output_path):
    """
    Write the results to a text file.

    Parameters:
        number (int): Number of employees feeling valued without suggestions.
        proportion (float): Proportion of such employees.
        output_path (str): Path to save the output text file.

    Returns:
        None
    """
    with open(output_path, 'w') as f:
        f.write(f"Number of Employees Feeling Valued without Suggestions: {number}\n")
        f.write(f"Proportion: {proportion}%\n")

def main():
    """
    Main function to execute Task 2.
    """
    # Initialize Spark
    spark = initialize_spark()
    
    # Define file paths
    input_file = "/workspaces/spark-structured-api-employee-engagement-analysis-harishvarma87/input/employee_data.csv"
    output_file = "/workspaces/spark-structured-api-employee-engagement-analysis-harishvarma87/outputs/task2/valued_no_suggestions.txt"
    
    # Load data
    df = load_data(spark, input_file)
    
    # Perform Task 2
    number, proportion = identify_valued_no_suggestions(df)
    
    # Write the result to a text file
    write_output(number, proportion, output_file)
    
    # Stop Spark Session
    spark.stop()

if __name__ == "__main__":
    main()
```
output - [output](./outputs/task2/valued_no_suggestions.txt)

Task 3:
```python
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
```
output - [output](./outputs/task3/engagement_levels_job_titles.csv/part-00000-ffe4ef5b-dbb4-467d-ace3-48fa534c4a5e-c000.csv)

## **Prerequisites**

Before starting the assignment, ensure you have the following software installed and properly configured on your machine:

1. **Python 3.x**:
   - [Download and Install Python](https://www.python.org/downloads/)
   - Verify installation:
     ```bash
     python3 --version
     ```

2. **PySpark**:
   - Install using `pip`:
     ```bash
     pip install pyspark
     ```

3. **Apache Spark**:
   - Ensure Spark is installed. You can download it from the [Apache Spark Downloads](https://spark.apache.org/downloads.html) page.
   - Verify installation by running:
     ```bash
     spark-submit --version
     ```



## **Setup Instructions**

### **1. Project Structure**

Ensure your project directory follows the structure below:

```
input/
└── employee_data.csv
outputs/
├── departments_high_satisfaction.csv
├── valued_no_suggestions.txt
└── engagement_levels_job_titles.csv
src/
├── task1_identify_departments_high_satisfaction.py
├── task2_valued_no_suggestions.py
└── task3_compare_engagement_levels.py
docker-compose.yml
README.md
```

- **input/**: Contains the `employee_data.csv` dataset.
- **outputs/**: Directory where the results of each task will be saved.
- **src/**: Contains the individual Python scripts for each task.
- **docker-compose.yml**: Docker Compose configuration file to set up Spark.
- **README.md**: Assignment instructions and guidelines.

### **2. Running the Analysis Tasks**

You can run the analysis tasks either locally or using Docker.

#### **a. Running Locally**

1. **Execute Each Task Using `spark-submit`**:
   ```bash
   spark-submit src/task1_identify_departments_high_satisfaction.py
   spark-submit src/task2_valued_no_suggestions.py
   spark-submit src/task3_compare_engagement_levels.py
   ```

