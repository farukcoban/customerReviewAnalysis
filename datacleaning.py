from pyspark.sql import SparkSession

# SparkSession başlatma
spark = SparkSession.builder.appName("JSONL Metadata Analysis").getOrCreate()

# GCS'deki sıkıştırılmış JSONL dosyasının yolu
file_path = "gs://farkcbsbi/Electronics.jsonl.gz"

# 1. JSONL dosyasını okuma
print("Sıkıştırılmış JSONL dosyası okunuyor...")
df = spark.read.json(file_path)

# 2. İlk 5 satırı gösterme
print("\nİlk 5 satır:")
df.show(5, truncate=False)

# 3. Sütun isimlerini ve sütun sayısını alma
column_names = df.columns
column_count = len(column_names)
print("\nSütun İsimleri:", column_names)
print("Sütun Sayısı:", column_count)

# 4. Toplam satır sayısını hesaplama
row_count = df.count()
print("Toplam Satır Sayısı:", row_count)

# 5. Şemayı görüntüleme
print("\nDosya Şeması:")
df.printSchema()

# 6. Metadata bilgilerini oluşturma
metadata = f"""JSONL Dosyası Metadata Bilgileri:
---------------------------------
Toplam Satır Sayısı: {row_count}
Sütun Sayısı: {column_count}
Sütun İsimleri: {column_names}
Şema:
"""
schema_str = df.schema.simpleString()
metadata += schema_str

# 7. Metadata bilgisini kaydetme
metadata_output_path = "gs://farkcbsbi/metadata/"
print(f"\nMetadata bilgisi {metadata_output_path} dizinine kaydediliyor...")
spark.sparkContext.parallelize([metadata]).saveAsTextFile(metadata_output_path)

print("\nAnaliz tamamlandı ve metadata kaydedildi.")
