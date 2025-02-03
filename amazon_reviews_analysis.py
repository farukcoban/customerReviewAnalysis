from pyspark.sql import SparkSession
from pyspark.sql.functions import col, regexp_replace
from pyspark.ml.feature import Tokenizer, StopWordsRemover, CountVectorizer, IDF, StringIndexer
from pyspark.ml.regression import LinearRegression
from pyspark.ml.evaluation import RegressionEvaluator

# Spark oturumu başlatma
spark = SparkSession.builder.appName("AmazonReviewsAnalysis").getOrCreate()

# Veri setini yükleme
data_path = "gs://farkcbsbi/Electronics.jsonl.gz"
raw_data = spark.read.json(data_path)

# Veri setinin ilk satırlarını ve şemasını kontrol etme
raw_data.show(5)
raw_data.printSchema()

# Gerekli sütunları seçme ve eksik değerleri temizleme
data = raw_data.select("rating", "text", "verified_purchase").na.drop()

# Metin verilerini temizleme (noktalama işaretlerini kaldırma)
data = data.withColumn("text", regexp_replace(col("text"), r"[^\w\s]", ""))

# 'verified_purchase' sütununu StringType'a dönüştürme
data = data.withColumn("verified_purchase_str", col("verified_purchase").cast("string"))

# Tokenizer: Metni kelimelere ayırma
tokenizer = Tokenizer(inputCol="text", outputCol="words")
data = tokenizer.transform(data)

# Stopwords Removal: Durdurma kelimelerini kaldırma
remover = StopWordsRemover(inputCol="words", outputCol="filtered_words")
data = remover.transform(data)

# CountVectorizer: Kelime frekansı
cv = CountVectorizer(inputCol="filtered_words", outputCol="raw_features")
cv_model = cv.fit(data)
data = cv_model.transform(data)

# IDF: TF-IDF hesaplama
idf = IDF(inputCol="raw_features", outputCol="features")
idf_model = idf.fit(data)
data = idf_model.transform(data)

# StringIndexer: Kategorik veriyi kodlama
indexer = StringIndexer(inputCol="verified_purchase_str", outputCol="verified_index")
data = indexer.fit(data).transform(data)

# Kodlanmış sütunları kontrol etme
data.select("verified_purchase", "verified_purchase_str", "verified_index").show(5)

# Nihai veri setini oluşturma (sadece gerekli sütunlar)
final_data = data.select("features", "rating")

# Eğitim ve test setlerini ayırma
train_data, test_data = final_data.randomSplit([0.8, 0.2], seed=42)

# Lineer Regresyon modeli
lr = LinearRegression(featuresCol="features", labelCol="rating")
lr_model = lr.fit(train_data)

# Test setinde tahmin yapma
predictions = lr_model.transform(test_data)

# RMSE (Root Mean Squared Error) hesaplama
evaluator = RegressionEvaluator(labelCol="rating", predictionCol="prediction", metricName="rmse")
rmse = evaluator.evaluate(predictions)
print(f"Root Mean Squared Error (RMSE): {rmse}")

# Tahminleri kaydetme
output_path = "farkcbsbi/ml/"
predictions.select("rating", "prediction").write.csv(output_path, header=True)

# Spark oturumunu kapatma
spark.stop()
