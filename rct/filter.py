import nltk, os
import csv
import re
import sys

dir = sys.argv[1]
savedir = dir+'_oneline'
nltk.download('words')
english_words = set(nltk.corpus.words.words())

for mycsv_name in os.listdir(dir):
    with open(os.path.join(dir, mycsv_name), "r", encoding="UTF8") as mycsv:
        csv_read = csv.reader(mycsv)
        par_list = list(csv_read)

        # filter out nonwords
        filtered = []
        for sentence in par_list:
            alpha = re.sub('[^A-Za-z ]+', '', sentence[0])
            # filtered_sent = alpha
            filtered_sent = " ".join(w for w in nltk.wordpunct_tokenize(alpha) if w.lower() in english_words or not w.isalpha())
            if(filtered_sent != ''): filtered.append(filtered_sent+' ')

        with open(os.path.join(savedir, mycsv_name), "w", encoding="UTF8") as writer:
            for line in filtered:
                writer.write(line)