from pyspark.sql import SparkSession

# SparkSession başlatma
spark = SparkSession.builder.appName("Amazon Reviews Filtering").getOrCreate()

# GCS'deki veri seti ve çıktı yolu
input_path = "gs://farkcbsbi/Electronics.jsonl.gz"
output_path = "gs://farkcbsbi/filtered_reviews/"

# 1. JSONL dosyasını okuma
print("JSONL dosyası okunuyor...")
df = spark.read.json(input_path)

# 2. Veriyi filtreleme
print("Veri filtreleniyor...")
filtered_df = df.filter((df["overall"] == 5) & (df["reviewText"].isNotNull()) & (df["reviewText"].rlike(".*")))

# 3. Sonuçları GCS'ye kaydetme
print(f"Filtrelenmiş sonuçlar {output_path} adresine kaydediliyor...")
filtered_df.write.json(output_path, mode="overwrite")

print("Filtreleme işlemi tamamlandı ve sonuçlar kaydedildi.")
