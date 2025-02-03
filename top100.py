from pyspark.sql import SparkSession

# SparkSession başlatma
spark = SparkSession.builder.appName("WordCount Top 100").getOrCreate()

# Veri seti ve çıktı yolları
input_path = "gs://farkcbsbi/wordcount_output/part-00000"
output_path = "gs://farkcbsbi/top_100_words/"

# 1. Dosyayı okuma
rdd = spark.read.text(input_path).rdd.map(lambda x: eval(x[0]))  # Veriyi tuple olarak oku

# 2. Veriyi İşleme
def parse_line(line):
    try:
        # Tuple formatında veriyi temizle
        word = line[0].strip('\'", ')  # Kelimeyi temizle
        count = int(line[1])  # Sayıyı integer'a dönüştür
        return word, count
    except Exception as e:
        print(f"Hatalı satır: {line}, Hata: {e}")
        return None

# Veriyi temizle ve filtrele
parsed_rdd = rdd.map(parse_line).filter(lambda x: x is not None)

# 3. En Çok Geçen 100 Kelimeyi Bulma
top_100_words = parsed_rdd.takeOrdered(100, key=lambda x: -x[1])

# 4. Sonuçları DataFrame'e Dönüştürme
top_100_df = spark.createDataFrame(top_100_words, ["Word", "Count"])

# 5. Sonuçları GCS'ye Kaydetme
top_100_df.write.csv(output_path, header=True)

print(f"En çok geçen 100 kelime başarıyla GCS'ye kaydedildi: {output_path}")
