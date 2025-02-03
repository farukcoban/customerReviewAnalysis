from pyspark.sql import SparkSession

# SparkSession başlatma
spark = SparkSession.builder.appName("MapReduce WordCount").getOrCreate()

# GCS'deki veri setinin yolu
input_path = "gs://farkcbsbi/Electronics.jsonl.gz"
output_path = "gs://farkcbsbi/wordcount_output/"

# 1. JSONL dosyasını okuma
print("JSONL dosyası okunuyor...")
rdd = spark.read.text(input_path).rdd.map(lambda x: x[0])  # Her satırı RDD olarak okuma

# 2. Kelime Sayımı (MapReduce Tarzında)
print("Kelime sayımı gerçekleştiriliyor...")
word_counts = (
    rdd.flatMap(lambda line: line.split())  # Satırları kelimelere böl
    .map(lambda word: (word.lower(), 1))   # Her kelime için (kelime, 1) çiftleri oluştur
    .reduceByKey(lambda a, b: a + b)       # Aynı kelimeleri topla
)

# 3. Sonuçları kaydetme
print(f"Sonuçlar {output_path} dizinine kaydediliyor...")
word_counts.saveAsTextFile(output_path)

print("Kelime sayımı işlemi tamamlandı ve sonuçlar kaydedildi.")
