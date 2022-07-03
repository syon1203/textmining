from pathlib import Path

import numpy as np
from spacy.language import Language
from spacy_sentiws.senti_ws_wrapper import SentiWSWrapper

@Language.factory("sentiws")
def create_component(nlp: Language, name, sentiws_path):
    return spaCySentiWS(sentiws_path=sentiws_path)

class spaCySentiWS(object):
    def __init__(self, sentiws_path):
        self.sentiws = SentiWSWrapper(sentiws_path=sentiws_path)

import spacy
from spacy_sentiws import spaCySentiWS
import string
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import matplotlib.pyplot as plt
from nltk import Text
from nltk import FreqDist
from wordcloud import WordCloud


text =  open('tonio.txt', encoding='utf-8').read()
#소문자화 후 불용어 제거 및 텍스트 전처리, 토큰화
lower_case = text.lower()
german_stop_words = stopwords.words('german')
cleaned = lower_case.translate(str.maketrans('', '', string.punctuation))

tokenized_words = word_tokenize(cleaned, 'german')
final = []

#제거되지 못한 특문 추가 제거
for words in tokenized_words:
    if words not in german_stop_words and words != "»" and words != "«":
        final.append(words+" ")

with open("text2.txt", "w") as file:
    file.writelines(final)

content = Path("text2.txt").read_text().replace('\n', ' ')

nlp = spacy.load("de_core_news_sm")

#NLTK 결과 감성점수화
nlp.add_pipe("sentiws", config={'sentiws_path': '/Users/syon1203/SynologyDrive/디지털인문학/textmining/SentiWS_v2'})
doc = nlp(content)
num = 0.0
numlst = []

for token in doc:
    print('{}, {}, {}'.format(token.text, token._.sentiws, token.pos_))
    try:
        token._.sentiws = float(token._.sentiws)
        num += token._.sentiws
        numlst.append(num)
    except TypeError:
        num += 0

a = len(numlst)

#작품을 5개로 나누어 처리
n1 = round(round(numlst[3],3)*100,0)
n2 = round(round(numlst[a//10*3],3)*100,0)
n3 = round(round(numlst[a//10*5],3)*100,0)
n4 = round(round(numlst[a//10*7],3)*100,0)
n5 = round(round(numlst[-1],3)*100,0)

x = np.arange(5)
years = ['Exposition', 'Rising Action', 'Climax','Falling Action','Resolution']
values = [n1, n2, n3, n4, n5]

print(n1,n2,n3,n4,n5)
#출력
plt.bar(x, values)
plt.xticks(x, years)
plt.show()

text = Text(final)
text.plot(20)

fd_names = FreqDist(final)
fd_names.most_common(5)

#워드클라우드 및 빈도 그래프 출력 
wc = WordCloud(width=1000, height=600, background_color="white", random_state=0)
plt.imshow(wc.generate_from_frequencies(fd_names))
plt.axis("off")

plt.show()
plt.show()
