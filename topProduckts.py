from pyspark.sql import SparkSession
from pyspark.sql.functions import col, explode, split, year, avg, count

# Spark session başlatma
spark = SparkSession.builder.appName("AmazonReviewsAnalysis").getOrCreate()

# Veri seti ve sonuçların kaydedileceği yol
input_path = "gs://farkcbsbi/Electronics.jsonl.gz"
output_path = "gs://farkcbsbi/analyze.csv"
top_words_output_path = "gs://farkcbsbi/top_words.csv"

# 1. JSONL formatındaki veri setini yükleme
df = spark.read.json(input_path)

# 2. 'text' sütunundaki toplam kelime sayısını hesaplama
def word_count(text):
    if text is not None:
        return len(text.split())
    return 0

word_count_udf = spark.udf.register("word_count", word_count)
df = df.withColumn("word_count", word_count_udf(col("text")))

total_word_count = df.selectExpr("sum(word_count) as total_word_count").collect()[0]["total_word_count"]

# 3. 'verified_purchase' True olanların 'timestamp' göre yıllık dağılımını hesaplama
df_verified = df.filter(col("verified_purchase") == True)
df_verified = df_verified.withColumn("year", year((col("timestamp") / 1000).cast("timestamp")))
verified_yearly_distribution = (
    df_verified.groupBy("year").count().orderBy("year")
)

# 4. 'rating' sayısının 'timestamp' a göre yıllık ortalamasını hesaplama
df = df.withColumn("year", year((col("timestamp") / 1000).cast("timestamp")))
rating_yearly_avg = (
    df.groupBy("year").agg(avg("rating").alias("avg_rating"), count("rating").alias("rating_count"))
    .orderBy("year")
)

# 5. 'parent_asin' için en çok tekrar eden ilk 10 ürünün yıllık tekrar sayısı ve rating ortalaması
top_parent_asins = (
    df.groupBy("parent_asin")
    .count()
    .orderBy(col("count").desc())
    .limit(10)
)
top_parent_asins_df = (
    df.join(top_parent_asins, on="parent_asin", how="inner")
    .groupBy("parent_asin", "year")
    .agg(count("parent_asin").alias("yearly_count"), avg("rating").alias("avg_rating"))
    .orderBy("parent_asin", "year")
)

# 6. En çok satan 10 ürün için 'text' sütununda en çok kullanılan 100 kelime
top_texts_rdd = (
    df.join(top_parent_asins, on="parent_asin", how="inner")
    .select("text")
    .rdd.flatMap(lambda row: row["text"].split() if row["text"] else [])
    .map(lambda word: (word.lower(), 1))
    .reduceByKey(lambda a, b: a + b)
    .sortBy(lambda x: x[1], ascending=False)
    .take(100)
)
top_words_df = spark.createDataFrame(top_texts_rdd, ["word", "count"])

# 7. Sonuçları birleştirme ve CSV olarak kaydetme
word_count_df = spark.createDataFrame([(total_word_count,)], ["total_word_count"])

summary_df = (
    verified_yearly_distribution
    .join(rating_yearly_avg, on="year", how="outer")
    .orderBy("year")
)

summary_df = summary_df.withColumnRenamed("count", "verified_count")

final_df = summary_df.crossJoin(word_count_df)
final_df.write.csv(output_path, header=True, mode="overwrite")

# En çok kullanılan 100 kelimeyi kaydetme
top_words_df.write.csv(top_words_output_path, header=True, mode="overwrite")

# Top parent_asin yıllık analizi kaydetme
parent_asin_output_path = "gs://farkcbsbi/top_parent_asins.csv"
top_parent_asins_df.write.csv(parent_asin_output_path, header=True, mode="overwrite")

print(f"Analiz tamamlandı ve sonuçlar şu adrese kaydedildi: {output_path}")
print(f"En çok kullanılan 100 kelime şu adrese kaydedildi: {top_words_output_path}")
print(f"Top parent_asin analizi şu adrese kaydedildi: {parent_asin_output_path}")
