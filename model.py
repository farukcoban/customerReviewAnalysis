from pyspark.sql import SparkSession
from pyspark.sql.functions import col, when, split, size, year, concat_ws
from pyspark.ml.feature import Tokenizer, StopWordsRemover, CountVectorizer
from pyspark.ml.classification import LogisticRegression
from pyspark.ml.evaluation import BinaryClassificationEvaluator
from pyspark.ml import Pipeline

# 🚀 Spark session başlatma (Bellek optimizasyonu yapıldı)
spark = SparkSession.builder.appName("SentimentAnalysisWithMLlib") \
    .config("spark.sql.shuffle.partitions", "8") \
    .config("spark.executor.memory", "4g") \
    .config("spark.driver.memory", "2g") \
    .config("spark.executor.cores", "2") \
    .config("spark.yarn.executor.memoryOverhead", "512") \
    .config("spark.local.dir", "/tmp/spark-temp") \
    .config("spark.sql.warehouse.dir", "hdfs:///user/hive/warehouse") \
    .config("spark.history.fs.logDirectory", "hdfs:///user/spark/job-history/") \
    .config("spark.eventLog.dir", "hdfs:///user/spark/event-logs/") \
    .getOrCreate()

# 📌 Veri seti ve sonuçların kaydedileceği yol
input_path = "gs://farkcbsbi/Electronics.jsonl.gz"
output_path = "gs://farkcbsbi/model_results.csv"

# 📌 1. Veri Yükleme ve Ön İşleme
df = spark.read.json(input_path)
df = df.repartition(10)  # Büyük veri işleme için paralelliği artır

# Yalnızca gerekli sütunları seçme ve eksik verileri filtreleme
df = df.select("text", "rating", "verified_purchase").filter(col("text").isNotNull() & col("rating").isNotNull())

# Puanları pozitif (1) ve negatif (0) olarak etiketleme
df = df.withColumn("label", when(col("rating") >= 4, 1).otherwise(0))

# 📌 2. Metin Ön İşleme
# Tokenizer: Kelimelere ayırma
tokenizer = Tokenizer(inputCol="text", outputCol="words")

# StopWordsRemover: Gereksiz kelimeleri çıkarma
stopwords_remover = StopWordsRemover(inputCol="words", outputCol="filtered_words")

# CountVectorizer: Kelimeleri vektörleştirme (Optimizasyon: Kelime sayısını sınırla)
vectorizer = CountVectorizer(inputCol="filtered_words", outputCol="features", vocabSize=5000, minDF=5)

# Pipeline oluşturma
pipeline = Pipeline(stages=[tokenizer, stopwords_remover, vectorizer])
processed_df = pipeline.fit(df).transform(df).cache()

# 📌 3. Eğitim ve Test Setlerine Ayırma
train_df, test_df = processed_df.randomSplit([0.8, 0.2], seed=42)

# 📌 4. Model Eğitimi
lr = LogisticRegression(featuresCol="features", labelCol="label", maxIter=20, regParam=0.1)
lr_model = lr.fit(train_df)

# 📌 5. Model Değerlendirme
predictions = lr_model.transform(test_df).cache()
evaluator = BinaryClassificationEvaluator(labelCol="label", metricName="areaUnderROC")
auc = evaluator.evaluate(predictions)
print(f"📊 Model AUC: {auc}")

# 📌 6. CSV Kaydetme İçin `words` Sütununu String'e Çevirme
predictions = predictions.withColumn("words", concat_ws(" ", col("words")))

# 📌 7. Sonuçların Kaydedilmesi (Optimizasyon: Daha az bölünmüş dosya ile kaydet)
predictions.coalesce(1).write.csv(output_path, header=True, mode="overwrite")
print(f"✅ Tahmin sonuçları şu adrese kaydedildi: {output_path}")
