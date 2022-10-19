import nltk
from nltk.corpus import stopwords
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
import string
from nltk.tokenize import word_tokenize
import re

### CASE FOLDING / PERUBAHAN KONVERSI TEKS MENJADI BENTUK STANDAR BIASANYA HURUF KECIL / LOWERCASE
# sentence = "bagaimana cara menghubungkan VPN dari I-Zone?"
# sentence = "cara koneksi ke VPN"
# sentence = "berbalas-balasan"
# sentence = "cara memakai VPN"
sentence = input("Masukkan pesan: ")
lowercase_sentence = sentence.lower()

### TOKENIZING
# REMOVE ANGKA
lowercase_sentence = re.sub(r"\d+", "", lowercase_sentence)
# REMOVE PUNCTUATION
lowercase_sentence = lowercase_sentence.translate(str.maketrans("","", string.punctuation))
# REMOVE WHITESPACE LEADING & TRAILING
lowercase_sentence = lowercase_sentence.strip()
# REMOVE MULTIPLE WHITESPACE INTO SINGLE WHITESPACE
lowercase_sentence = re.sub(r"\s+", " ", lowercase_sentence)
# TOKENIZE
token = word_tokenize(lowercase_sentence)

list_words = str(stopwords.words('indonesian'))
token_without_stopword = [ word for word in token if not word in list_words ]
factory = StemmerFactory()
stemmer = factory.create_stemmer()
output = []
for word in token_without_stopword:
    output.append(stemmer.stem(word))

print(output)


# print(token_without_stopword)
# print(list_words)