import sys
import random
from awsglue.transforms import *
from awsglue.utils import getResolvedOptions
from pyspark.context import SparkContext
from awsglue.context import GlueContext
from awsglue.job import Job
from awsglue.dynamicframe import DynamicFrame
from pyspark.sql.functions import col, lit, udf
from pyspark.sql.types import StringType, DoubleType, TimestampType, IntegerType

### AWS Glue ETL Job initialization
args = getResolvedOptions(sys.argv, ['JOB_NAME'])
sc = SparkContext()
glueContext = GlueContext(sc)
spark = glueContext.spark_session
job = Job(glueContext)
job.init(args['JOB_NAME'], args)

# Read input CSV from S3
source_path = "s3://deglue1/Transactions.csv"
df = spark.read.option("header", "true").csv(source_path)

#prepare sample data
# UDF to randomly assign currency
def add_transaction_currency(df):
    @udf(StringType())
    def random_currency():
        return random.choice(["USD", "JPY"])
    return df.withColumn("TransactionCurrency", random_currency())
# since transaction currency is missing, we will randomly assign it
df = add_transaction_currency(df)


# Cast fields to appropriate types
df = df.withColumn("TransactionDate", col("TransactionDate").cast(TimestampType())) \
       .withColumn("Quantity", col("Quantity").cast(IntegerType())) \
       .withColumn("TotalValue", col("TotalValue").cast(DoubleType())) \
       .withColumn("Price", col("Price").cast(DoubleType()))

#business logics
# Drop duplicate transaction IDs
df = df.dropDuplicates(["TransactionID"])

# UDF to convert to SGD
def add_converted_amount(df):
    @udf(DoubleType())
    def convert_to_sgd(amount, currency):
        rates = {"USD": 1.3, "JPY": 0.009}
        return round(amount * rates.get(currency, 1.0), 2)
    return df.withColumn("ConvertedCurrency", lit("SGD")) \
             .withColumn("ConvertedAmount", convert_to_sgd(col("TotalValue"), col("TransactionCurrency")))

# Convert to SGD currency
df = add_converted_amount(df)

df.printSchema()
df.show(5)

output_path = "s3://deglue1/output"
# flag out high-risk transactions > 10000 SGD
high_risk_df = df.filter(col("ConvertedAmount") > 10000)
high_risk_df.write.mode("overwrite").options(header='True').csv(output_path+"/high_risk")

# Write the final DataFrame to S3
df.write.mode("overwrite").options(header='True').csv(output_path+"/transactions")

job.commit()