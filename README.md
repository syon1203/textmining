# :pushpin: textmining
> 독일 고전문학 속 언어 분석을 통한 감성 분석 

> German NLP with Emotion Recognition using SentiWS

## 📎 제작기간 & 참여인원 
- 2022.03-2022.06
- 개인 프로젝트(디지털인문학, 중앙대학교 인문 소프트웨어 융합전공 과정)


## 📎 사용 기술

- Python 3.10
- NLTK
- spaCy 2.0
- numpy
- matplotlib

## 📎 사용 데이터셋

- sentiWS 2.0 (독일어 감성사전) 

## 📎 핵심 기능
- 독일 고전문학 속의 언어 분석을 통한 주요 정서 및 감정 변화 분석

- 빈출어를 통한 작품의 주요 모티프 분석 

<details>
<summary><b>기능 설명</b></summary>
<div markdown="1">

### 📎 제작 동기
  
  고전문학 작품의 연구에서 빅데이터를 통한 계층, 사회 등 다양한 분야의 연구가 시도되었습니다. 
그 중 독일 비극의 경우 개인의 내면에 중심을 둔 이야기의 전개 방식을 주로 하고 있습니다. **젊은 베르테르의 슬픔, 토니오 크뢰거** 등 유명 독일 문학은 개인이 일련의 사건을 겪으며 겪는 변화를 시간에 따라 긴밀하게 연결하여 변화하는 내면의 양상 내지 성장을 주제로 하고 있습니다.

  그 중 개인의 내적 변화에 있어 대표적 작품인 “젊은 베르테르의 슬픔”과 “토니오 크뢰거”를 중심으로 감정을 분석하여 주인공의 내면의 변화 그리고 작품의 지배적 정서를 파악하는 것을 주제로 정하였습니다. 두 작품 모두 신분의 한계, 정체성에 대한 고민, 사랑의 좌절이 주요 소재이므로 감정의 변화나 작품의 정서에 대해 파악하기 가장 좋은 작품이라고 생각하였기 때문입니다.
  
  인문 텍스트를 읽는 방법의 다양화와 인문 텍스트의 다분야에서의 활용을 꿈꾸며 제작한 프로젝트입니다. 


### :pushpin: 방법
  독일어 문학작품을 [구텐베르크 프로젝트](https://www.gutenberg.org) 등의 플랫폼을 통해 텍스트를 수집하였습니다. 파이썬의 NLTK 패키지를 이용해 전처리하였습니다. 문장 단위로 나누고 토큰화한 다음 불용어를 제거하여 필요한 단어만 수집할 수 있도록 하였습니다. [spaCy](https://spacy.io/universe/project/spacy-sentiws) 자연언어 처리용 오픈소스 라이브러리가 독일어를 지원하므로 이를 사용하였고, 그 중 독일어 감정사전인 sentiws를 데이터셋을 이용하여 감성분석을 진행하였습니다.

**주요 코드** [확인하기](https://github.com/syon1203/textmining.git)
  
### 📎 과정
  젊은 베르테르의 슬픔”을 위의 방식으로 감성분류하였습니다. 이를 감성점수의 변화 정도를 보고 작품의 결말에서 총점이 어느 정도인지에 따라 작품의 주된 정서를 판단할 수 있다고 판단하였습니다. “젊은 베르테르의 슬픔”과 “토니오 크뢰거”를 이와 같은 방법을 통해 분석해 보았습니다. 작품의 각 부분은 임의로 5개의 구분점을 두었습니다.
  
  
  
- 젊은 베르테르의 슬픔 : 작품의 정서 변화를 크게 다섯 개로 나누어 출력한 변화도(누적값)
  
  </br>
  <img width="186" alt="image" src="https://user-images.githubusercontent.com/103924086/178223594-11f98275-158a-4ba1-a472-7a6f24ae2a70.png">
  
  “젊은 베르테르의 슬픔”의 중간 부분과 결말 부분의 총 감정지수의 합을 출력하였습니다. -0.1596점에서 6.2154점, -15.5361점으로 점수의 폭이 변화하였음을 확인할 수 있으며, 이를 통해 젊은 베르테르의 슬픔이 전반적으로 긍정적 정서에서 결말 부분에 부정적인 정서로 변화하였음을 파악할 수 있습니다. 작품의 중반부에서는 로테에 대한 사랑의 정서가 지배하고 있으나 후반부에 있어서는 자살과 정체성에 대한 고민이 주된 내용이 된다는 점에서 감정점수가 적절히 판단하고 있다고 파악되었습니다.
    </br>

  
  
  
  
- 토니오 크뢰거 : 작품의 정서 변화를 크게 다섯 분야로 나누어 출력한 변화도(누적값)
  
    </br>
  <img width="174" alt="image" src="https://user-images.githubusercontent.com/103924086/178223641-f09fbbdc-5bf7-42d1-866b-611eac5d0a1f.png">
  
  “토니오 크뢰거”의 경우에는 젊은 베르테르의 슬픔과는 반대로 중반부까지는 부정적인 정서가 작품 전반을 지배함을 확인할 수 있습니다. 그러나 결말부에서는 -4.097점으로 정서 점수가 상당히 상승한 것을 관찰할 수 있습니다. 따라서 작품의 정서가 부정에서 긍정의 정서로 변화하였음을 판단할 수 있었습니다. 작품의 중반부에서는 사랑의 좌절이나 개인의 정체성 고민이 주된 내용이었다는 점이었으나 후반부에서는 자신을 이루는 어머니와 아버지의 정체성 두 가지를 모두 포용했다는 점이 감성 점수로도 드러난다고 판단할 수 있었습니다. 


  </br>


</div>
</details>
  
## 📌 결과
  
<img width="272" alt="image" src="https://user-images.githubusercontent.com/103924086/178222964-c1cb51c0-7416-4090-8086-bba7f330869d.png">
  
  “젊은 베르테르의 슬픔” 의 감정 변화 추이
  
<img width="272" alt="image" src="https://user-images.githubusercontent.com/103924086/178222983-c772c132-5e60-4921-8243-40f7acca68f4.png">
  
  “토니오 크뢰거”의 감정 변화 추이
  
**“젊은 베르테르의 슬픔”** 의 중간 부분과 결말 부분의 총 감정지수의 합을 출력했습니다. -0.1596점에서 6.2154점, -15.5361점으로 점수의 폭이 변화하였음을 확인할 수 있습니다. 

이를 통해 젊은 베르테르의 슬픔이 전반적으로 긍정적 정서에서 결말 부분에 부정적인 정서로 변화하였음을 파악할 수 있습니다. 
작품의 중반부에서는 로테에 대한 사랑의 정서가 지배하고 있으나 후반부에 있어서는 자살과 정체성에 대한 고민이 주된 내용이 된다는 점에서 감정점수가 적절히 판단하고 있다고 파악되었습니다.

**“토니오 크뢰거”** 의 경우에는 젊은 베르테르의 슬픔과는 반대로 중반부까지는 부정적인 정서가 작품 전반을 지배함을 확인할 수 있습니다. 
그러나 결말부에서는 -4.097점으로 정서 점수가 상당히 상승한 것을 관찰할 수 있습니다. 

따라서 작품의 정서가 부정에서 긍정의 정서로 변화하였음을 판단할 수 있습니다. 
작품의 중반부에서는 사랑의 좌절이나 개인의 정체성 고민이 주된 내용이었다는 점이었으나 후반부에서는 자신을 이루는 어머니와 아버지의 정체성 두 가지를 모두 포용했다는 점이 감성 점수로도 드러난다고 판단할 수 있습니다.
  
  <img width="269" alt="image" src="https://user-images.githubusercontent.com/103924086/178224890-62d5c2ca-5c53-4a41-ac13-dbf920fcd996.png">
  
  <img width="269" alt="image" src="https://user-images.githubusercontent.com/103924086/178224869-d2aa5aaa-248b-4afd-adb5-9c19bc13f2b6.png">


 젊은 베르테르의 슬픔과 토니오 크뢰거 속 빈출 단어

젊은 베르테르의 슬픔 : 말했다, 오, 영혼, 편안한, 베르테르, 소량의, 로테, 알버트, 늘, 심장, 사랑, 눈물, 왜, 자주, 소리, 더욱, 가다, 눈, 가다(과거형), 말하다


토니오 크뢰거 : 토니오, 크뢰거, 말했다, 응, 한스(주인공의 친구), 소량의, 눈, 온전히, 서다(과거형), 가다(산책 등), 보다(과거형), 리자베타(주인공의 연인), ~씨, 사랑, 예술, 삶, 더욱, 이미, 긴, 한스의


의 단어가 빈출되었습니다. 따라서 이는 토니오 크뢰거에서는 주인공 자신이 가장 자주 나온 것으로 보아 자전적 성격이 더 강함을 유추할 수 있으며 주인공에게 영향을 준 인물로는 한스와 리자베타가 있음을 파악할 수 있었습니다. 특히 타 인물에 비하여 한스가 눈에 띄게 자주 나왔다는 점이 주목할 만한 점이었습니다. 또한 젊은 베르테르의 슬픔에서는 영혼, 사랑, 눈물, 심장 등이 매우 자주 나왔다는 점에서 주제를 파악할 수 있었으며 주요 인물 또한 파악이 가능하였습니다. 

## 💻 소감

위의 ‘토니오 크뢰거’와 ‘젊은 베르테르의 슬픔’과 같이 같은 주제를 다룬 작품에 대해 주된 감정의 변화와 결말의 비극, 희극성에 대해 유추할 수 있는 가능성을 보였으며 작가에 따라 작품의 주된 모티프가 동일하더라도 방식적 차이가 있음을 분석할 수 있었습니다. 또한 단어의 빈출에 따라 작가 혹은 작품에서의 주 사용 어휘를 분석할 수 있다는 장점이 있습니다. 차후 독일어 뿐만 아니라 한국어도 가능하도록 방향을 추가하고 싶습니다. 

다만 아쉬운 점은 단어 간의 연관도에 대한 파악이 부족하고, nicht(영어의 not)과 같은 부정어가 결합됨에 따른 의미의 변화를 파악하지 못했다는 점입니다. 향후 인공지능 공부를 통해 위와 같은 사항을 수정하는 것이 목표입니다.


  
  
  
  

