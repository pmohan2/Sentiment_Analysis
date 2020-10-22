---
## Sentiment Analysis on US Presidential Speeches of State of the Union Address using NLP 

Author: **Praveen Mohan**

---

## Introduction

- *This project utilizes the abilities of Natural Language Processing (NLP) in building Unsupervised Sentiment Analysis model for analyzing the transcripts of some of the most popular US Presidential speeches in the history.*

- *In simple words NLP is a technique which is used to mine the raw text or transcript data to gather valuable information or to make sense out of the raw data. This project will help us understand how each president’s speech had an impact in the elections and among the people and also help us identify patterns, similarities and differences within each president’s speeches.*

- *The Sentiment Analysis is a technique that analyses the emotions involved in a transcript. It helps us identify the kinds of common emotions that each president uses and also helps us to understand the impact of various emotions among people.*

---

## Data Source: [The Miller Center](https://millercenter.org/the-presidency/presidential-speeches)
- It is a nonpartisan affiliate of the University of Virginia which specializes in presidential speeches, policies and political history.
- They apply the lessons of history to the nation’s most pressing contemporary governance challenges.

## Packages Required
---
- `import requests` to scrape from the URL.
- `from bs4 import BeautifulSoup` to convert to a text form from the URL.
- `import pickle` to store the scraped text in required format.
- `import pandas as pd` to store the transcripts in a dataframe.
- `import numpy as np` mostly to use `np.arange()` function.
- `import re` for cleanning the raw text.
- `import strings` for performing common string operations.
- `from sklearn.feature_extraction.text import CountVectorizer` to create a Document-Term Matrix.
- `import nltk` importing the Natural Language Tool Kit package.
- `from nltk.corpus import wordnet as wn` WordNet is a lexical database for the English language, which was created by Princeton, and is part of the NLTK corpus.
- `from nltk.stem.wordnet import WordNetLemmatizer` to perform Stemming and Lemmatization on the raw text.
- `from nltk import word_tokenize, pos_tag` to tokenize the transcripts.
- `from collections import defaultdict` to compare each lemma with the default dictionary to get the correct spelling for the stemmed words.
- `from wordcloud import WordCloud` to visualize the most frequent words in the form of a Word Cloud.
- `import matplotlib.pyplot as plt` to plot other bar and scatter plots.
- `import seaborn as sns` to use `sns.set()` function to make plots more interesting.
- `from textblob import TextBlob` to perform the Sentiment Analysis.

**The following content needs to be downloaded from the `nltk`:**
* `nltk.download('stopwords')`
* `nltk.download('wordnet')`
* `nltk.download('averaged_perceptron_tagger')`
---


```python
import requests
from bs4 import BeautifulSoup
import pickle

import pandas as pd
import numpy as np
pd.set_option('max_colwidth',500)

import re
import string

from sklearn.feature_extraction import text
from sklearn.feature_extraction.text import CountVectorizer

import nltk
from nltk.corpus import wordnet as wn
from nltk.stem.wordnet import WordNetLemmatizer
from nltk import word_tokenize, pos_tag
from collections import defaultdict

from wordcloud import WordCloud
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()

from textblob import TextBlob

import math

nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('averaged_perceptron_tagger')
```

    [nltk_data] Downloading package stopwords to
    [nltk_data]     /Users/praveen/nltk_data...
    [nltk_data]   Package stopwords is already up-to-date!
    [nltk_data] Downloading package wordnet to /Users/praveen/nltk_data...
    [nltk_data]   Package wordnet is already up-to-date!
    [nltk_data] Downloading package averaged_perceptron_tagger to
    [nltk_data]     /Users/praveen/nltk_data...
    [nltk_data]   Package averaged_perceptron_tagger is already up-to-
    [nltk_data]       date!





    True



---
# Data Gathering 
---
## Web Scrapping 
* **`requests`**
* **`BeautifulSoup`**
---


```python
# Scrapes transcript data from The Miller Center webpage
def url_to_transcript(url):
    page = requests.get(url).text
    soup = BeautifulSoup(page, "lxml")
    text = [p.text for p in soup.find(class_="transcript-inner").find_all('p')]
    print(url)
    return text

# URLs of transcripts of Presidents during State of the Union Address
urls = ["https://millercenter.org/the-presidency/presidential-speeches/february-4-2020-state-union-address",
        "https://millercenter.org/the-presidency/presidential-speeches/january-12-2016-2016-state-union-address",
        "https://millercenter.org/the-presidency/presidential-speeches/january-28-2008-state-union-address",
        "https://millercenter.org/the-presidency/presidential-speeches/january-27-2000-state-union-address",
        "https://millercenter.org/the-presidency/presidential-speeches/january-25-1988-state-union-address",
        "https://millercenter.org/the-presidency/presidential-speeches/january-23-1980-state-union-address",
        "https://millercenter.org/the-presidency/presidential-speeches/january-14-1963-state-union-address",
        "https://millercenter.org/the-presidency/presidential-speeches/january-7-1943-state-union-address"]

# President names
presidents = ["Trump", "Obama", "Bush", "Bill_Clinton", "Ronald_Reagan", "Jimmy_Carter", "JFK", "FDR" ]
full_names = ["Bill_Clinton", "George_W.Bush", "Franklin_D.Roosevelt", "John_F.Kennedy", "Jimmy_Carter", "Barack_Obama", "Ronald_Reagan", "Donald_Trump"]
```


```python
# Scrapping the transcripts using the above defined function
transcripts = [url_to_transcript(i) for i in urls]
```

    https://millercenter.org/the-presidency/presidential-speeches/february-4-2020-state-union-address
    https://millercenter.org/the-presidency/presidential-speeches/january-12-2016-2016-state-union-address
    https://millercenter.org/the-presidency/presidential-speeches/january-28-2008-state-union-address
    https://millercenter.org/the-presidency/presidential-speeches/january-27-2000-state-union-address
    https://millercenter.org/the-presidency/presidential-speeches/january-25-1988-state-union-address
    https://millercenter.org/the-presidency/presidential-speeches/january-23-1980-state-union-address
    https://millercenter.org/the-presidency/presidential-speeches/january-14-1963-state-union-address
    https://millercenter.org/the-presidency/presidential-speeches/january-7-1943-state-union-address



```python
# Storing each transcripts in a text file using pickle
for i, p in enumerate(presidents):
    with open("transcripts/" + p + ".txt", "wb") as file:
        pickle.dump(transcripts[i], file)

#  Storing each transcripts in a dictionary for the corresponding Presidents
data = {}
for i, p in enumerate(presidents):
    with open("transcripts/" + p + ".txt", "rb") as file:
        data[p] = pickle.load(file)
```

## Data Cleaning -  Text Pre-processing  
---

### Minimum Viable Product (MVP) Approach
---

**Data Cleaning steps on all texts:**
* Make text all lower case
* Remove punctuation
* Remove numerical values
* Remove common non-sensical text
* Tokenize text
* Remove stop words
---
**Data Cleaning steps after Tokenization:**
* Stemming / lemmatization
* Parts of speech tagging
* Create bi-grams or tri-grams or n-grams
* Deal with typos
* And more...
---


```python
"""
The values in the dictionary "data" is in the form of list of texts. 
The function `combine_text(list_of_text)` combines the list of texts in to a single text.
"""
def combine_text(list_of_text):
    '''Takes a list of text and combines them into one large chunk of text.'''
    combined_text = ' '.join(list_of_text)
    return combined_text
```


```python
# dictionary whose values are changed from list of texts to list of a single text.
data_combined = {key: [combine_text(value)] for (key, value) in data.items()}
```


```python
"""
Converting the "data_combined" dictionary to a pandas DataFrame.
CORPUS Format: A collection of texts.
"""
data_df = pd.DataFrame.from_dict(data_combined).transpose()
data_df.columns = ['transcript']
data_df = data_df.sort_index()
data_df
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>transcript</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>Bill_Clinton</th>
      <td>Mr. Speaker, Mr. Vice President, members of Congress, honored guests, my fellow Americans: We are fortunate to be alive at this moment in history. Never before has our nation enjoyed, at once, so much prosperity and social progress with so little internal crisis and so few external threats. Never before have we had such a blessed opportunity and, therefore, such a profound obligation to build the more perfect Union of our Founders’ dreams. We begin the new century with over 20 million new jo...</td>
    </tr>
    <tr>
      <th>Bush</th>
      <td>Madam Speaker, Vice President Cheney, members of Congress, distinguished guests, and fellow citizens: Seven years have passed since I first stood before you at this rostrum. In that time, our country has been tested in ways none of us could have imagined. We faced hard decisions about peace and war, rising competition in the world economy, and the health and welfare of our citizens. These issues call for vigorous debate, and I think it's fair to say, we've answered the call. Yet history will...</td>
    </tr>
    <tr>
      <th>FDR</th>
      <td>Mr. Vice President, Mr. Speaker, members of the 78th Congress:\n\r\nThis 78th Congress assembles in one of the great moments in the history of the nation. The past year was perhaps the most crucial for modern civilization; the coming year will be filled with violent conflicts—yet with high promise of better things.\n\r\nWe must appraise the events of 1942 according to their relative importance; we must exercise a sense of proportion.\n\r\nFirst in importance in the American scene has been th...</td>
    </tr>
    <tr>
      <th>JFK</th>
      <td>Mr. Vice President, Mr. Speaker, Members of the 88th Congress: I congratulate you all--not merely on your electoral victory but on your selected role in history. For you and I are privileged to serve the great Republic in what could be the most decisive decade in its long history. The choices we make, for good or ill, may well shape the state of the Union for generations yet to come. Little more than 100 weeks ago I assumed the office of President of the United States. In seeking the help of...</td>
    </tr>
    <tr>
      <th>Jimmy_Carter</th>
      <td>Mr. President, Mr. Speaker, members of the 96th Congress, fellow citizens: This last few months has not been an easy time for any of us. As we meet tonight, it has never been more clear that the state of our Union depends on the state of the world. And tonight, as throughout our own generation, freedom and peace in the world depend on the state of our Union. The 1980s have been born in turmoil, strife, and change. This is a time of challenge to our interests and our values and it's a time th...</td>
    </tr>
    <tr>
      <th>Obama</th>
      <td>Mr. Speaker, Mr. Vice President, Members of Congress, my fellow Americans: Tonight marks the eighth year that I’ve come here to report on the State of the Union. And for this final one, I’m going to try to make it a little shorter. (Applause.) I know some of you are antsy to get back to Iowa. (Laughter.) I've been there. I'll be shaking hands afterwards if you want some tips. (Laughter.) And I understand that because it’s an election season, expectations for what we will achieve this year ar...</td>
    </tr>
    <tr>
      <th>Ronald_Reagan</th>
      <td>Mr. Speaker, Mr. President, and distinguished Members of the House and Senate: When we first met here seven years ago-many of us for the first time—it was with the hope of beginning something new for America. We meet here tonight in this historic Chamber to continue that work. If anyone expects just a proud recitation of the accomplishments of my administration, I say let's leave that to history; we're not finished yet. So, my message to you tonight is put on your work shoes; we're still on ...</td>
    </tr>
    <tr>
      <th>Trump</th>
      <td>Thank you very much. Thank you. Thank you very much. Madam Speaker, Mr. Vice President, members of Congress, the First Lady of the United States—(applause)—and my fellow citizens: Three years ago, we launched the great American comeback. Tonight, I stand before you to share the incredible results. Jobs are booming, incomes are soaring, poverty is plummeting, crime is falling, confidence is surging, and our country is thriving and highly respected again. (Applause.) America’s enemies are on t...</td>
    </tr>
  </tbody>
</table>
</div>




```python
"""
First step in Data Cleaning.
The function "clean_text_round1()" does the following:
* Removing anything within parentheses "()".
* Removing punctuations.
* Removing unnecessary digits and errors inbetween texts.
"""

def clean_text(text):
    '''Make text lowercase, remove text in square brackets, remove punctuation and remove words containing numbers.'''
    text = text.lower()
    text = re.sub('\(.*?\)', '', text)
    text = re.sub('[%s]' % re.escape(string.punctuation), '', text)
    text = re.sub('\w*\d\w*', '', text)
    text = re.sub(r'\xa0', ' ', text)
    text = re.sub(r'  ', ' ', text)
    text = re.sub('[‘’“”…]', '', text)
    text = re.sub('\n', '', text)
    return text

clean = lambda x: clean_text(x)
```


```python
data_clean = pd.DataFrame(data_df.transcript.apply(clean))
data_clean
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>transcript</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>Bill_Clinton</th>
      <td>mr speaker mr vice president members of congress honored guests my fellow americans we are fortunate to be alive at this moment in history never before has our nation enjoyed at once so much prosperity and social progress with so little internal crisis and so few external threats never before have we had such a blessed opportunity and therefore such a profound obligation to build the more perfect union of our founders dreams we begin the new century with over million new jobs the fastest eco...</td>
    </tr>
    <tr>
      <th>Bush</th>
      <td>madam speaker vice president cheney members of congress distinguished guests and fellow citizens seven years have passed since i first stood before you at this rostrum in that time our country has been tested in ways none of us could have imagined we faced hard decisions about peace and war rising competition in the world economy and the health and welfare of our citizens these issues call for vigorous debate and i think its fair to say weve answered the call yet history will record that ami...</td>
    </tr>
    <tr>
      <th>FDR</th>
      <td>mr vice president mr speaker members of the congress\rthis congress assembles in one of the great moments in the history of the nation the past year was perhaps the most crucial for modern civilization the coming year will be filled with violent conflicts—yet with high promise of better things\rwe must appraise the events of according to their relative importance we must exercise a sense of proportion\rfirst in importance in the american scene has been the inspiring proof of the great qualit...</td>
    </tr>
    <tr>
      <th>JFK</th>
      <td>mr vice president mr speaker members of the congress i congratulate you allnot merely on your electoral victory but on your selected role in history for you and i are privileged to serve the great republic in what could be the most decisive decade in its long history the choices we make for good or ill may well shape the state of the union for generations yet to come little more than weeks ago i assumed the office of president of the united states in seeking the help of the congress and our ...</td>
    </tr>
    <tr>
      <th>Jimmy_Carter</th>
      <td>mr president mr speaker members of the congress fellow citizens this last few months has not been an easy time for any of us as we meet tonight it has never been more clear that the state of our union depends on the state of the world and tonight as throughout our own generation freedom and peace in the world depend on the state of our union the have been born in turmoil strife and change this is a time of challenge to our interests and our values and its a time that tests our wisdom and our...</td>
    </tr>
    <tr>
      <th>Obama</th>
      <td>mr speaker mr vice president members of congress my fellow americans tonight marks the eighth year that ive come here to report on the state of the union and for this final one im going to try to make it a little shorter i know some of you are antsy to get back to iowa ive been there ill be shaking hands afterwards if you want some tips and i understand that because its an election season expectations for what we will achieve this year are low but mr speaker i appreciate the constructive app...</td>
    </tr>
    <tr>
      <th>Ronald_Reagan</th>
      <td>mr speaker mr president and distinguished members of the house and senate when we first met here seven years agomany of us for the first time—it was with the hope of beginning something new for america we meet here tonight in this historic chamber to continue that work if anyone expects just a proud recitation of the accomplishments of my administration i say lets leave that to history were not finished yet so my message to you tonight is put on your work shoes were still on the job history ...</td>
    </tr>
    <tr>
      <th>Trump</th>
      <td>thank you very much thank you thank you very much madam speaker mr vice president members of congress the first lady of the united states——and my fellow citizens three years ago we launched the great american comeback tonight i stand before you to share the incredible results jobs are booming incomes are soaring poverty is plummeting crime is falling confidence is surging and our country is thriving and highly respected again americas enemies are on the run americas fortunes are on the rise ...</td>
    </tr>
  </tbody>
</table>
</div>




```python
"""
Stemming and Lemmatization are Text Normalization (or sometimes called Word Normalization)
techniques in the field of Natural Language Processing that are used to prepare text, words,
and documents for further processing.

The function "stemSentence()" does the following:
* Takes a sentence as input.
* Tokenizes the sentence.
* Stem each words.
* Combines the stemmed words back to a single sentence.
* Returns the stemmed sentence.

"""
def stemSentence(sentence):
    tag_map = defaultdict(lambda : wn.NOUN)
    tag_map['J'] = wn.ADJ
    tag_map['V'] = wn.VERB
    tag_map['R'] = wn.ADV
    tokens = word_tokenize(sentence)
    lemma_function = WordNetLemmatizer()
    stem_sentence=[]
    for token, tag in pos_tag(tokens):
        lemma = lemma_function.lemmatize(token, tag_map[tag[0]])
        stem_sentence.append(lemma)
        stem_sentence.append(" ")
    return "".join(stem_sentence)
stem = lambda x: stemSentence(x)   
```


```python
data_clean = pd.DataFrame(data_clean.transcript.apply(stem))
data_clean
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>transcript</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>Bill_Clinton</th>
      <td>mr speaker mr vice president member of congress honor guest my fellow american we be fortunate to be alive at this moment in history never before have our nation enjoy at once so much prosperity and social progress with so little internal crisis and so few external threat never before have we have such a blessed opportunity and therefore such a profound obligation to build the more perfect union of our founder dream we begin the new century with over million new job the fast economic growth ...</td>
    </tr>
    <tr>
      <th>Bush</th>
      <td>madam speaker vice president cheney member of congress distinguish guest and fellow citizen seven year have pass since i first stand before you at this rostrum in that time our country have be test in way none of u could have imagine we face hard decision about peace and war rise competition in the world economy and the health and welfare of our citizen these issue call for vigorous debate and i think it fair to say weve answer the call yet history will record that amid our difference we act...</td>
    </tr>
    <tr>
      <th>FDR</th>
      <td>mr vice president mr speaker member of the congress this congress assemble in one of the great moment in the history of the nation the past year be perhaps the most crucial for modern civilization the come year will be fill with violent conflicts—yet with high promise of good thing we must appraise the event of accord to their relative importance we must exercise a sense of proportion first in importance in the american scene have be the inspire proof of the great quality of our fight men th...</td>
    </tr>
    <tr>
      <th>JFK</th>
      <td>mr vice president mr speaker member of the congress i congratulate you allnot merely on your electoral victory but on your select role in history for you and i be privilege to serve the great republic in what could be the most decisive decade in it long history the choice we make for good or ill may well shape the state of the union for generation yet to come little more than week ago i assume the office of president of the united state in seek the help of the congress and our countryman i p...</td>
    </tr>
    <tr>
      <th>Jimmy_Carter</th>
      <td>mr president mr speaker member of the congress fellow citizen this last few month have not be an easy time for any of u a we meet tonight it have never be more clear that the state of our union depend on the state of the world and tonight a throughout our own generation freedom and peace in the world depend on the state of our union the have be bear in turmoil strife and change this be a time of challenge to our interest and our value and it a time that test our wisdom and our skill at this ...</td>
    </tr>
    <tr>
      <th>Obama</th>
      <td>mr speaker mr vice president member of congress my fellow american tonight mark the eighth year that ive come here to report on the state of the union and for this final one im go to try to make it a little shorter i know some of you be antsy to get back to iowa ive be there ill be shake hand afterwards if you want some tip and i understand that because it an election season expectation for what we will achieve this year be low but mr speaker i appreciate the constructive approach that you a...</td>
    </tr>
    <tr>
      <th>Ronald_Reagan</th>
      <td>mr speaker mr president and distinguish member of the house and senate when we first meet here seven year agomany of u for the first time—it be with the hope of begin something new for america we meet here tonight in this historic chamber to continue that work if anyone expect just a proud recitation of the accomplishment of my administration i say let leave that to history be not finish yet so my message to you tonight be put on your work shoe be still on the job history record the power of...</td>
    </tr>
    <tr>
      <th>Trump</th>
      <td>thank you very much thank you thank you very much madam speaker mr vice president member of congress the first lady of the united states——and my fellow citizen three year ago we launch the great american comeback tonight i stand before you to share the incredible result job be boom income be soar poverty be plummet crime be fall confidence be surge and our country be thrive and highly respect again americas enemy be on the run america fortune be on the rise and americas future be blaze brigh...</td>
    </tr>
  </tbody>
</table>
</div>




```python
"""
Saving the raw corpus form in corpus.pkl
"""
data_df['full_name'] = full_names
data_df.to_pickle("corpus.pkl")
data_df
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>transcript</th>
      <th>full_name</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>Bill_Clinton</th>
      <td>Mr. Speaker, Mr. Vice President, members of Congress, honored guests, my fellow Americans: We are fortunate to be alive at this moment in history. Never before has our nation enjoyed, at once, so much prosperity and social progress with so little internal crisis and so few external threats. Never before have we had such a blessed opportunity and, therefore, such a profound obligation to build the more perfect Union of our Founders’ dreams. We begin the new century with over 20 million new jo...</td>
      <td>Bill_Clinton</td>
    </tr>
    <tr>
      <th>Bush</th>
      <td>Madam Speaker, Vice President Cheney, members of Congress, distinguished guests, and fellow citizens: Seven years have passed since I first stood before you at this rostrum. In that time, our country has been tested in ways none of us could have imagined. We faced hard decisions about peace and war, rising competition in the world economy, and the health and welfare of our citizens. These issues call for vigorous debate, and I think it's fair to say, we've answered the call. Yet history will...</td>
      <td>George_W.Bush</td>
    </tr>
    <tr>
      <th>FDR</th>
      <td>Mr. Vice President, Mr. Speaker, members of the 78th Congress:\n\r\nThis 78th Congress assembles in one of the great moments in the history of the nation. The past year was perhaps the most crucial for modern civilization; the coming year will be filled with violent conflicts—yet with high promise of better things.\n\r\nWe must appraise the events of 1942 according to their relative importance; we must exercise a sense of proportion.\n\r\nFirst in importance in the American scene has been th...</td>
      <td>Franklin_D.Roosevelt</td>
    </tr>
    <tr>
      <th>JFK</th>
      <td>Mr. Vice President, Mr. Speaker, Members of the 88th Congress: I congratulate you all--not merely on your electoral victory but on your selected role in history. For you and I are privileged to serve the great Republic in what could be the most decisive decade in its long history. The choices we make, for good or ill, may well shape the state of the Union for generations yet to come. Little more than 100 weeks ago I assumed the office of President of the United States. In seeking the help of...</td>
      <td>John_F.Kennedy</td>
    </tr>
    <tr>
      <th>Jimmy_Carter</th>
      <td>Mr. President, Mr. Speaker, members of the 96th Congress, fellow citizens: This last few months has not been an easy time for any of us. As we meet tonight, it has never been more clear that the state of our Union depends on the state of the world. And tonight, as throughout our own generation, freedom and peace in the world depend on the state of our Union. The 1980s have been born in turmoil, strife, and change. This is a time of challenge to our interests and our values and it's a time th...</td>
      <td>Jimmy_Carter</td>
    </tr>
    <tr>
      <th>Obama</th>
      <td>Mr. Speaker, Mr. Vice President, Members of Congress, my fellow Americans: Tonight marks the eighth year that I’ve come here to report on the State of the Union. And for this final one, I’m going to try to make it a little shorter. (Applause.) I know some of you are antsy to get back to Iowa. (Laughter.) I've been there. I'll be shaking hands afterwards if you want some tips. (Laughter.) And I understand that because it’s an election season, expectations for what we will achieve this year ar...</td>
      <td>Barack_Obama</td>
    </tr>
    <tr>
      <th>Ronald_Reagan</th>
      <td>Mr. Speaker, Mr. President, and distinguished Members of the House and Senate: When we first met here seven years ago-many of us for the first time—it was with the hope of beginning something new for America. We meet here tonight in this historic Chamber to continue that work. If anyone expects just a proud recitation of the accomplishments of my administration, I say let's leave that to history; we're not finished yet. So, my message to you tonight is put on your work shoes; we're still on ...</td>
      <td>Ronald_Reagan</td>
    </tr>
    <tr>
      <th>Trump</th>
      <td>Thank you very much. Thank you. Thank you very much. Madam Speaker, Mr. Vice President, members of Congress, the First Lady of the United States—(applause)—and my fellow citizens: Three years ago, we launched the great American comeback. Tonight, I stand before you to share the incredible results. Jobs are booming, incomes are soaring, poverty is plummeting, crime is falling, confidence is surging, and our country is thriving and highly respected again. (Applause.) America’s enemies are on t...</td>
      <td>Donald_Trump</td>
    </tr>
  </tbody>
</table>
</div>



## Document Term Matrix (DTM)



```python
"""
* Creating a Document-Term Matrix using CountVectorizer.
* Excluding common English stop words.
"""
cv = CountVectorizer(stop_words='english')
data_cv = cv.fit_transform(data_clean.transcript)
data_dtm = pd.DataFrame(data_cv.toarray(), columns=cv.get_feature_names())
data_dtm.index = data_clean.index
data_dtm
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>aaron</th>
      <th>abandon</th>
      <th>abandonment</th>
      <th>abhorrent</th>
      <th>ability</th>
      <th>able</th>
      <th>ablebodied</th>
      <th>abm</th>
      <th>abolish</th>
      <th>abortion</th>
      <th>...</th>
      <th>yokohama</th>
      <th>york</th>
      <th>youll</th>
      <th>young</th>
      <th>youre</th>
      <th>youth</th>
      <th>youthful</th>
      <th>youve</th>
      <th>zimbabwe</th>
      <th>zone</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>Bill_Clinton</th>
      <td>2</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>4</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>3</td>
      <td>9</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>Bush</th>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>4</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>2</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
    </tr>
    <tr>
      <th>FDR</th>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>2</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>4</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>JFK</th>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>4</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>3</td>
      <td>0</td>
      <td>2</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>Jimmy_Carter</th>
      <td>0</td>
      <td>2</td>
      <td>0</td>
      <td>1</td>
      <td>2</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>2</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>Obama</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>1</td>
      <td>2</td>
      <td>3</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>Ronald_Reagan</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>2</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>5</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>Trump</th>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>...</td>
      <td>0</td>
      <td>2</td>
      <td>0</td>
      <td>9</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>3</td>
    </tr>
  </tbody>
</table>
<p>8 rows × 4166 columns</p>
</div>




```python
"""
Saving all data in pickle form.
"""
data_dtm.to_pickle("dtm.pkl")
data_clean.to_pickle('data_clean.pkl')
pickle.dump(cv, open("cv.pkl", "wb"))
```

## Exploratory Data Analysis


```python
df = data_dtm.transpose()
df
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Bill_Clinton</th>
      <th>Bush</th>
      <th>FDR</th>
      <th>JFK</th>
      <th>Jimmy_Carter</th>
      <th>Obama</th>
      <th>Ronald_Reagan</th>
      <th>Trump</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>aaron</th>
      <td>2</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>abandon</th>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>2</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>abandonment</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>abhorrent</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>ability</th>
      <td>1</td>
      <td>4</td>
      <td>2</td>
      <td>4</td>
      <td>2</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>youth</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>2</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>youthful</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>youve</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>zimbabwe</th>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>zone</th>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>3</td>
    </tr>
  </tbody>
</table>
<p>4166 rows × 8 columns</p>
</div>



### `top_dict`: The top 30 words spoken by each President


```python
top_dict = {}
for p in df.columns:
    top = df[p].sort_values(ascending=False).head(30)
    top_dict[p]= list(zip(top.index, top.values))
top_dict
```




    {'Bill_Clinton': [('year', 60),
      ('child', 49),
      ('new', 47),
      ('make', 47),
      ('ask', 46),
      ('work', 44),
      ('help', 44),
      ('american', 44),
      ('people', 41),
      ('america', 30),
      ('thank', 29),
      ('want', 28),
      ('tonight', 27),
      ('care', 24),
      ('gun', 24),
      ('support', 23),
      ('community', 23),
      ('school', 21),
      ('propose', 21),
      ('family', 20),
      ('know', 20),
      ('time', 19),
      ('tax', 19),
      ('million', 19),
      ('need', 18),
      ('just', 18),
      ('pass', 18),
      ('let', 18),
      ('congress', 17),
      ('weve', 17)],
     'Bush': [('year', 35),
      ('people', 32),
      ('america', 30),
      ('american', 29),
      ('congress', 27),
      ('nation', 25),
      ('new', 25),
      ('iraq', 21),
      ('help', 20),
      ('trust', 19),
      ('good', 19),
      ('weve', 18),
      ('ask', 17),
      ('iraqi', 17),
      ('terrorist', 17),
      ('come', 16),
      ('future', 14),
      ('force', 14),
      ('world', 14),
      ('make', 14),
      ('agreement', 14),
      ('fight', 14),
      ('tax', 14),
      ('pass', 13),
      ('country', 13),
      ('work', 12),
      ('enemy', 12),
      ('build', 12),
      ('empower', 11),
      ('meet', 11)],
     'FDR': [('war', 46),
      ('great', 26),
      ('nation', 25),
      ('year', 24),
      ('production', 22),
      ('world', 21),
      ('american', 17),
      ('fight', 16),
      ('united', 15),
      ('people', 15),
      ('men', 12),
      ('peace', 12),
      ('good', 12),
      ('know', 11),
      ('force', 11),
      ('time', 10),
      ('task', 10),
      ('want', 10),
      ('produce', 10),
      ('japanese', 10),
      ('battle', 9),
      ('axis', 9),
      ('enemy', 9),
      ('air', 8),
      ('work', 8),
      ('america', 8),
      ('pacific', 8),
      ('freedom', 8),
      ('common', 8),
      ('end', 7)],
     'JFK': [('nation', 31),
      ('year', 30),
      ('world', 24),
      ('tax', 21),
      ('free', 19),
      ('defense', 18),
      ('need', 17),
      ('american', 17),
      ('new', 16),
      ('peace', 15),
      ('end', 14),
      ('billion', 14),
      ('alliance', 14),
      ('help', 13),
      ('freedom', 13),
      ('people', 13),
      ('make', 12),
      ('program', 12),
      ('nuclear', 12),
      ('increase', 12),
      ('communist', 12),
      ('effort', 11),
      ('today', 11),
      ('reduction', 11),
      ('country', 10),
      ('mean', 9),
      ('long', 9),
      ('million', 9),
      ('good', 9),
      ('national', 9)],
     'Jimmy_Carter': [('soviet', 32),
      ('world', 27),
      ('nation', 26),
      ('peace', 22),
      ('america', 21),
      ('union', 17),
      ('state', 17),
      ('security', 13),
      ('united', 13),
      ('military', 13),
      ('meet', 13),
      ('people', 13),
      ('challenge', 12),
      ('continue', 12),
      ('help', 12),
      ('oil', 11),
      ('force', 11),
      ('need', 10),
      ('make', 10),
      ('preserve', 10),
      ('year', 10),
      ('power', 9),
      ('time', 9),
      ('strong', 9),
      ('effort', 9),
      ('energy', 9),
      ('region', 8),
      ('right', 8),
      ('east', 8),
      ('action', 8)],
     'Obama': [('american', 37),
      ('thats', 35),
      ('year', 34),
      ('make', 32),
      ('work', 30),
      ('america', 28),
      ('people', 27),
      ('just', 25),
      ('world', 25),
      ('change', 25),
      ('want', 22),
      ('need', 20),
      ('new', 19),
      ('job', 19),
      ('good', 18),
      ('way', 17),
      ('country', 17),
      ('economy', 16),
      ('right', 16),
      ('weve', 15),
      ('dont', 14),
      ('future', 13),
      ('like', 13),
      ('know', 11),
      ('big', 11),
      ('come', 11),
      ('im', 10),
      ('family', 10),
      ('believe', 10),
      ('security', 10)],
     'Ronald_Reagan': [('let', 34),
      ('year', 34),
      ('make', 26),
      ('america', 26),
      ('american', 22),
      ('freedom', 21),
      ('family', 20),
      ('government', 19),
      ('world', 18),
      ('people', 17),
      ('budget', 17),
      ('work', 16),
      ('agreement', 15),
      ('time', 15),
      ('federal', 15),
      ('nation', 14),
      ('tonight', 13),
      ('say', 13),
      ('future', 13),
      ('state', 12),
      ('congress', 12),
      ('free', 12),
      ('democratic', 11),
      ('great', 10),
      ('right', 10),
      ('economic', 10),
      ('spend', 10),
      ('peace', 10),
      ('know', 10),
      ('president', 9)],
     'Trump': [('american', 55),
      ('thank', 31),
      ('america', 30),
      ('year', 27),
      ('country', 22),
      ('people', 22),
      ('new', 22),
      ('nation', 20),
      ('administration', 20),
      ('state', 18),
      ('tonight', 18),
      ('world', 17),
      ('million', 17),
      ('just', 17),
      ('work', 15),
      ('make', 15),
      ('family', 15),
      ('congress', 14),
      ('percent', 13),
      ('president', 13),
      ('great', 13),
      ('job', 13),
      ('ago', 12),
      ('child', 12),
      ('day', 12),
      ('good', 12),
      ('force', 11),
      ('record', 11),
      ('united', 11),
      ('opportunity', 11)]}



### The top 15 words spoken by each President


```python
for president, top_words in top_dict.items():
    print(president)
    print(', '.join([word for word, count in top_words[0:14]]))
    print('-------')
```

    Bill_Clinton
    year, child, new, make, ask, work, help, american, people, america, thank, want, tonight, care
    -------
    Bush
    year, people, america, american, congress, nation, new, iraq, help, trust, good, weve, ask, iraqi
    -------
    FDR
    war, great, nation, year, production, world, american, fight, united, people, men, peace, good, know
    -------
    JFK
    nation, year, world, tax, free, defense, need, american, new, peace, end, billion, alliance, help
    -------
    Jimmy_Carter
    soviet, world, nation, peace, america, union, state, security, united, military, meet, people, challenge, continue
    -------
    Obama
    american, thats, year, make, work, america, people, just, world, change, want, need, new, job
    -------
    Ronald_Reagan
    let, year, make, america, american, freedom, family, government, world, people, budget, work, agreement, time
    -------
    Trump
    american, thank, america, year, country, people, new, nation, administration, state, tonight, world, million, just
    -------


---
## `wordcloud`

**A tag cloud (word cloud or wordle or weighted list in visual design) is a novelty visual representation of text data, typically used to depict keyword metadata (tags) on websites, or to visualize free form text. Tags are usually single words, and the importance of each tag is shown with font size and color.**

---


```python
stop_words = text.ENGLISH_STOP_WORDS
wc = WordCloud(stopwords=stop_words, background_color="white", colormap="Dark2",
               max_font_size=150, random_state=42)
plt.rcParams['figure.figsize'] = [20, 20]
```


```python
# Create subplots for each President
for index, president in enumerate(df.columns):
    wc.generate(data_clean.transcript[president])
    plt.subplot(4,2, index+1)
    plt.imshow(wc, interpolation="bilinear")
    plt.axis("off")
    plt.title(full_names[index], fontsize = 20)
plt.savefig("viz1.jpeg")
plt.show()
```


    
![png](output_27_0.png)
    



```python
"""
* Finding the number of unique words that each Presidents uses.
* Identifying the non-zero items in the document-term matrix, meaning that the word occurs at least once.
* Calculating the words per minute of each President.
* Finding the total number of words that a President uses.
"""
unique_list = []
for president in df.columns:
    uniques = df[president].to_numpy().nonzero()[0].size
    unique_list.append(uniques)
    
total_list = []
for president in df.columns:
    totals = sum(df[president])
    total_list.append(totals)
    
# State of the Union Address run times from The Miller Center website, in minutes
run_times = [89, 53, 47, 44, 32, 60, 44, 78]

# Create a new dataframe that contains this unique word count
data_words = pd.DataFrame(list(zip(full_names, unique_list)), columns=['President', 'Unique_words'])
data_words['Total_words'] = total_list
data_words['Run_times'] = run_times
data_words['Words_per_minute'] = data_words['Total_words'] / data_words['Run_times']
data_words
data_unique_sort = data_words.sort_values(by='Unique_words')
data_wpm_sort = data_words.sort_values(by='Words_per_minute')
```


```python
"""
Plotting the above findings
"""
y_pos = np.arange(len(data_words))
plt.figure(figsize=(30,10))
plt.subplot(1, 2, 1)
plt.barh(y_pos, data_unique_sort.Unique_words, align='center')
plt.yticks(y_pos, data_unique_sort.President, fontsize=25)
plt.xticks(fontsize=20)
plt.title('Number of Unique Words', fontsize=40)

plt.subplot(1, 2, 2)
plt.barh(y_pos, data_wpm_sort.Words_per_minute, align='center')
plt.yticks(y_pos, data_wpm_sort.President, fontsize=25)
plt.xticks(fontsize=20)
plt.title('Number of Words Per Minute', fontsize=40)

plt.tight_layout()
plt.savefig("viz2.jpeg")
plt.show()
```


    
![png](output_29_0.png)
    


## Sentiment Analysis


1. **TextBlob Module:** Linguistic researchers have labeled the sentiment of words based on their domain expertise. Sentiment of words can vary based on where it is in a sentence. The TextBlob module allows us to take advantage of these labels.
2. **Sentiment Labels:** Each word in a corpus is labeled in terms of polarity and subjectivity (there are more labels as well). A corpus' sentiment is the average of these.
   * **Polarity**: How positive or negative a word is. -1 is very negative. +1 is very positive.
   * **Subjectivity**: How subjective, or objective (opinionated) a word is. 0 is fact. +1 is very much an opinion.

For more info on TextBlob [sentiment function](https://planspace.org/20150607-textblob_sentiment/).



```python
data_df
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>transcript</th>
      <th>full_name</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>Bill_Clinton</th>
      <td>Mr. Speaker, Mr. Vice President, members of Congress, honored guests, my fellow Americans: We are fortunate to be alive at this moment in history. Never before has our nation enjoyed, at once, so much prosperity and social progress with so little internal crisis and so few external threats. Never before have we had such a blessed opportunity and, therefore, such a profound obligation to build the more perfect Union of our Founders’ dreams. We begin the new century with over 20 million new jo...</td>
      <td>Bill_Clinton</td>
    </tr>
    <tr>
      <th>Bush</th>
      <td>Madam Speaker, Vice President Cheney, members of Congress, distinguished guests, and fellow citizens: Seven years have passed since I first stood before you at this rostrum. In that time, our country has been tested in ways none of us could have imagined. We faced hard decisions about peace and war, rising competition in the world economy, and the health and welfare of our citizens. These issues call for vigorous debate, and I think it's fair to say, we've answered the call. Yet history will...</td>
      <td>George_W.Bush</td>
    </tr>
    <tr>
      <th>FDR</th>
      <td>Mr. Vice President, Mr. Speaker, members of the 78th Congress:\n\r\nThis 78th Congress assembles in one of the great moments in the history of the nation. The past year was perhaps the most crucial for modern civilization; the coming year will be filled with violent conflicts—yet with high promise of better things.\n\r\nWe must appraise the events of 1942 according to their relative importance; we must exercise a sense of proportion.\n\r\nFirst in importance in the American scene has been th...</td>
      <td>Franklin_D.Roosevelt</td>
    </tr>
    <tr>
      <th>JFK</th>
      <td>Mr. Vice President, Mr. Speaker, Members of the 88th Congress: I congratulate you all--not merely on your electoral victory but on your selected role in history. For you and I are privileged to serve the great Republic in what could be the most decisive decade in its long history. The choices we make, for good or ill, may well shape the state of the Union for generations yet to come. Little more than 100 weeks ago I assumed the office of President of the United States. In seeking the help of...</td>
      <td>John_F.Kennedy</td>
    </tr>
    <tr>
      <th>Jimmy_Carter</th>
      <td>Mr. President, Mr. Speaker, members of the 96th Congress, fellow citizens: This last few months has not been an easy time for any of us. As we meet tonight, it has never been more clear that the state of our Union depends on the state of the world. And tonight, as throughout our own generation, freedom and peace in the world depend on the state of our Union. The 1980s have been born in turmoil, strife, and change. This is a time of challenge to our interests and our values and it's a time th...</td>
      <td>Jimmy_Carter</td>
    </tr>
    <tr>
      <th>Obama</th>
      <td>Mr. Speaker, Mr. Vice President, Members of Congress, my fellow Americans: Tonight marks the eighth year that I’ve come here to report on the State of the Union. And for this final one, I’m going to try to make it a little shorter. (Applause.) I know some of you are antsy to get back to Iowa. (Laughter.) I've been there. I'll be shaking hands afterwards if you want some tips. (Laughter.) And I understand that because it’s an election season, expectations for what we will achieve this year ar...</td>
      <td>Barack_Obama</td>
    </tr>
    <tr>
      <th>Ronald_Reagan</th>
      <td>Mr. Speaker, Mr. President, and distinguished Members of the House and Senate: When we first met here seven years ago-many of us for the first time—it was with the hope of beginning something new for America. We meet here tonight in this historic Chamber to continue that work. If anyone expects just a proud recitation of the accomplishments of my administration, I say let's leave that to history; we're not finished yet. So, my message to you tonight is put on your work shoes; we're still on ...</td>
      <td>Ronald_Reagan</td>
    </tr>
    <tr>
      <th>Trump</th>
      <td>Thank you very much. Thank you. Thank you very much. Madam Speaker, Mr. Vice President, members of Congress, the First Lady of the United States—(applause)—and my fellow citizens: Three years ago, we launched the great American comeback. Tonight, I stand before you to share the incredible results. Jobs are booming, incomes are soaring, poverty is plummeting, crime is falling, confidence is surging, and our country is thriving and highly respected again. (Applause.) America’s enemies are on t...</td>
      <td>Donald_Trump</td>
    </tr>
  </tbody>
</table>
</div>



---
* **Creating quick lambda functions to find the Polarity and Subjectivity of each routine.**
* **The speech order in the sentence matters in calculating the Polarity and Subjectivity, hence using the raw uncleaned data `data_df`.**
---


```python
pol = lambda x: TextBlob(x).sentiment.polarity
sub = lambda x: TextBlob(x).sentiment.subjectivity

data_df['Polarity'] = data_df['transcript'].apply(pol)
data_df['Subjectivity'] = data_df['transcript'].apply(sub)
data_df
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>transcript</th>
      <th>full_name</th>
      <th>Polarity</th>
      <th>Subjectivity</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>Bill_Clinton</th>
      <td>Mr. Speaker, Mr. Vice President, members of Congress, honored guests, my fellow Americans: We are fortunate to be alive at this moment in history. Never before has our nation enjoyed, at once, so much prosperity and social progress with so little internal crisis and so few external threats. Never before have we had such a blessed opportunity and, therefore, such a profound obligation to build the more perfect Union of our Founders’ dreams. We begin the new century with over 20 million new jo...</td>
      <td>Bill_Clinton</td>
      <td>0.158606</td>
      <td>0.420449</td>
    </tr>
    <tr>
      <th>Bush</th>
      <td>Madam Speaker, Vice President Cheney, members of Congress, distinguished guests, and fellow citizens: Seven years have passed since I first stood before you at this rostrum. In that time, our country has been tested in ways none of us could have imagined. We faced hard decisions about peace and war, rising competition in the world economy, and the health and welfare of our citizens. These issues call for vigorous debate, and I think it's fair to say, we've answered the call. Yet history will...</td>
      <td>George_W.Bush</td>
      <td>0.130243</td>
      <td>0.436398</td>
    </tr>
    <tr>
      <th>FDR</th>
      <td>Mr. Vice President, Mr. Speaker, members of the 78th Congress:\n\r\nThis 78th Congress assembles in one of the great moments in the history of the nation. The past year was perhaps the most crucial for modern civilization; the coming year will be filled with violent conflicts—yet with high promise of better things.\n\r\nWe must appraise the events of 1942 according to their relative importance; we must exercise a sense of proportion.\n\r\nFirst in importance in the American scene has been th...</td>
      <td>Franklin_D.Roosevelt</td>
      <td>0.150626</td>
      <td>0.474465</td>
    </tr>
    <tr>
      <th>JFK</th>
      <td>Mr. Vice President, Mr. Speaker, Members of the 88th Congress: I congratulate you all--not merely on your electoral victory but on your selected role in history. For you and I are privileged to serve the great Republic in what could be the most decisive decade in its long history. The choices we make, for good or ill, may well shape the state of the Union for generations yet to come. Little more than 100 weeks ago I assumed the office of President of the United States. In seeking the help of...</td>
      <td>John_F.Kennedy</td>
      <td>0.124134</td>
      <td>0.471664</td>
    </tr>
    <tr>
      <th>Jimmy_Carter</th>
      <td>Mr. President, Mr. Speaker, members of the 96th Congress, fellow citizens: This last few months has not been an easy time for any of us. As we meet tonight, it has never been more clear that the state of our Union depends on the state of the world. And tonight, as throughout our own generation, freedom and peace in the world depend on the state of our Union. The 1980s have been born in turmoil, strife, and change. This is a time of challenge to our interests and our values and it's a time th...</td>
      <td>Jimmy_Carter</td>
      <td>0.138561</td>
      <td>0.454664</td>
    </tr>
    <tr>
      <th>Obama</th>
      <td>Mr. Speaker, Mr. Vice President, Members of Congress, my fellow Americans: Tonight marks the eighth year that I’ve come here to report on the State of the Union. And for this final one, I’m going to try to make it a little shorter. (Applause.) I know some of you are antsy to get back to Iowa. (Laughter.) I've been there. I'll be shaking hands afterwards if you want some tips. (Laughter.) And I understand that because it’s an election season, expectations for what we will achieve this year ar...</td>
      <td>Barack_Obama</td>
      <td>0.118476</td>
      <td>0.431306</td>
    </tr>
    <tr>
      <th>Ronald_Reagan</th>
      <td>Mr. Speaker, Mr. President, and distinguished Members of the House and Senate: When we first met here seven years ago-many of us for the first time—it was with the hope of beginning something new for America. We meet here tonight in this historic Chamber to continue that work. If anyone expects just a proud recitation of the accomplishments of my administration, I say let's leave that to history; we're not finished yet. So, my message to you tonight is put on your work shoes; we're still on ...</td>
      <td>Ronald_Reagan</td>
      <td>0.158565</td>
      <td>0.429967</td>
    </tr>
    <tr>
      <th>Trump</th>
      <td>Thank you very much. Thank you. Thank you very much. Madam Speaker, Mr. Vice President, members of Congress, the First Lady of the United States—(applause)—and my fellow citizens: Three years ago, we launched the great American comeback. Tonight, I stand before you to share the incredible results. Jobs are booming, incomes are soaring, poverty is plummeting, crime is falling, confidence is surging, and our country is thriving and highly respected again. (Applause.) America’s enemies are on t...</td>
      <td>Donald_Trump</td>
      <td>0.155651</td>
      <td>0.453330</td>
    </tr>
  </tbody>
</table>
</div>




```python
"""
* Plotting the above results.
"""
sns.set_style("ticks")

plt.rcParams['figure.figsize'] = [14, 10]

for index, president in enumerate(data_df.index):
    x = data_df.Polarity.loc[president]
    y = data_df.Subjectivity.loc[president]
    plt.scatter(x, y, color='blue')
    plt.text(x+.001, y+.001, data_df['full_name'][index], fontsize=14)
    plt.xlim(.11, 0.175) 
    plt.ylim(.41, 0.49)
    
plt.title("Sentiment Analysis", fontsize=24)
plt.xlabel("Polarity\n<-- Negative -------- Positive -->", fontsize=15)
plt.ylabel("Subjectivity\n<-- Facts -------- Opinions -->", fontsize=15)
plt.savefig("Sentiment_Analysis.jpeg")
plt.show()
```


    
![png](output_34_0.png)
    


## Sentiment of Speeches Over Time
---

* *The above results gives us the sentiment analysis of speech as a whole.*
* *The below analysis is about the sentiment over time throughout each speeches for each President.*
---


```python
"""
* The goal is to find the trend of the sentiment for each President throughout their speeches.
* Splitting each speeches into 10 equal parts.
* The below function takes in a string of text and splits into n equal parts, 
  with a default of 10 equal parts.
* Calculating length of text, the size of each chunk of text and the starting points of each chunk of text.
* Pulling out equally sized pieces of text and storing it into a list.
"""
def split_text(text, n=10):
    length = len(text)
    size = math.floor(length / n)
    start = np.arange(0, length, size)
    split_list = []
    for piece in range(n):
        split_list.append(text[start[piece]:start[piece]+size])
    return split_list
```


```python
"""
* A list to hold all of the pieces of text.
"""
list_pieces = []
for t in data_df.transcript:
    split = split_text(t)
    list_pieces.append(split)
    
list_pieces
```




    [['Mr. Speaker, Mr. Vice President, members of Congress, honored guests, my fellow Americans: We are fortunate to be alive at this moment in history. Never before has our nation enjoyed, at once, so much prosperity and social progress with so little internal crisis and so few external threats. Never before have we had such a blessed opportunity and, therefore, such a profound obligation to build the more perfect Union of our Founders’ dreams. We begin the new century with over 20 million new jobs; the fastest economic growth in more than 30 years; the lowest unemployment rates in 30 years; the lowest poverty rates in 20 years; the lowest African-American and Hispanic unemployment rates on record; the first back-to-back surpluses in 42 years; and next month, America will achieve the longest period of economic growth in our entire history. We have built a new economy. And our economic revolution has been matched by a revival of the American spirit: crime down by 20 percent, to its lowest level in 25 years; teen births down seven years in a row; adoptions up by 30 percent; welfare rolls cut in half, to their lowest levels in 30 years. My fellow Americans, the state of our Union is the strongest it has ever been. As always, the real credit belongs to the American people. My gratitude also goes to those of you in this chamber who have worked with us to put progress over partisanship. Eight years ago, it was not so clear to most Americans there would be much to celebrate in the year 2000. Then our nation was gripped by economic distress, social decline, political gridlock. The title of a best-selling book asked: "America: What Went Wrong?" In the best traditions of our nation, Americans determined to set things right. We restored the vital center, replacing outmoded ideologies with a new vision anchored in basic, enduring values: opportunity for all, responsibility from all, a community of all Americans. We reinvented government, transforming it into a catalyst for new ideas that stress both opportunity and responsibility and give our people the tools they need to solve their own problems. With the smallest federal work force in 40 years, we turned record deficits into record surpluses and doubled our investment in education. We cut crime with 100,000 community police and the Brady law, which has kept guns out of the hands of half a million criminals. We ended welfare as we knew it, requiring work while protecting health care and nutrition for children and investing more in child care, transportation, and housing to help their parents go to work. We’ve helped parents to succeed at home and at work with family leave, which 20 million Americans have now used to care for a newborn child or a sick loved one. We’ve engaged 150,000 young Americans in citizen service through AmeriCorps, while helping them earn money for college. In 1992, we just had a roadmap. Today, we have results. Even more important, America again has the confidence to dream big dreams. But we must not let this confidence drift into complacency. For we, all of us, will be judged by the dreams and deeds we pass on to our children. And on that score, we will be held to a high standard, indeed, because our chance to do good is so great. My fellow Americans, we have crossed the bridge we built to the 21st century. Now, we must shape a 21st century American revolution of opportunity, responsibility, and community. We must be now, as we were in the beginning, a new nation. At the dawn of the last century, Theodore Roosevelt said, "The one characteristic more essential than any other is foresight . . . it should be the growing nation with a future that takes the long look ahead." So tonight let us take our long look ahead and set great goals for our Nation. To 21st century America, let us pledge these things: Every child will begin school ready to learn and graduate ready to succeed. Every family will be able to succeed at home and at work, and no child will be raised in poverty. We will meet the challenge of the aging of America. We will assure quality, affordable health care, at last, for all Americans. We will make America the safest big country on Earth. We will pay off our national debt for the first time since 1835.* We will bring prosperity to every American community. We will reverse the course of climate change and leave a safer, cleaner planet. America will lead the world toward shared peace and prosperity and the far frontiers of science and technology. And we will become at last what our Founders pledged us to be so long ago: * White House correction. One nation, under God, indivisible, with liberty and justice for all. These are great goals, worthy of a great nation. We will not reach them all this year, not even in this decade. But we will reach them. Let us remember that the first American Revolution was not won with a single shot; the continent was not settled in a single year. The lesson of our history and the lesson of the last seven years is that great goals are reached step by step, always building on our progress, always gaining ground. Of course, you can’t gain ground if you’re standing still. And for too long this Congress has been standing still on',
      ' some of our most pressing national priorities. So let’s begin tonight with them. Again, I ask you to pass a real Patients’ Bill of Rights. I ask you to pass common-sense gun safety legislation. I ask you to pass campaign finance reform. I ask you to vote up or down on judicial nominations and other important appointees. And again, I ask you—I implore you to raise the minimum wage. Now, two years ago—let me try to balance the seesaw here—[laughter]—two years ago, as we reached across party lines to reach our first balanced budget, I asked that we meet our responsibility to the next generation by maintaining our fiscal discipline. Because we refused to stray from that path, we are doing something that would have seemed unimaginable seven years ago. We are actually paying down the national debt. Now, if we stay on this path, we can pay down the debt entirely in just 13 years now and make America debt-free for the first time since Andrew Jackson was President in 1835. In 1993 we began to put our fiscal house in order with the Deficit Reduction Act, which you’ll all remember won passages in both Houses by just a single vote. Your former colleague, my first Secretary of the Treasury, led that effort and sparked our long boom. He’s here with us tonight. Lloyd Bentsen, you have served America well, and we thank you. Beyond paying off the debt, we must ensure that the benefits of debt reduction go to preserving two of the most important guarantees we make to every American, Social Security and Medicare. Tonight I ask you to work with me to make a bipartisan downpayment on Social Security reform by crediting the interest savings from debt reduction to the Social Security Trust Fund so that it will be strong and sound for the next 50 years. But this is just the start of our journey. We must also take the right steps toward reaching our great goals. First and foremost, we need a 21st century revolution in education, guided by our faith that every single child can learn. Because education is more important than ever, more than ever the key to our children’s future, we must make sure all our children have that key. That means quality preschool and after-school, the best trained teachers in the classroom, and college opportunities for all our children. For seven years now, we’ve worked hard to improve our schools, with opportunity and responsibility, investing more but demanding more in turn. Reading, math, college entrance scores are up. Some of the most impressive gains are in schools in very poor neighborhoods. But all successful schools have followed the same proven formula: higher standards, more accountability, and extra help so children who need it can get it to reach those standards. I have sent Congress a reform plan based on that formula. It holds states and school districts accountable for progress and rewards them for results. Each year, our national government invests more than $15 billion in our schools. It is time to support what works and stop supporting what doesn’t. Now, as we demand more from our schools, we should also invest more in our schools. Let’s double our investment to help states and districts turn around their worst performing schools or shut them down. Let’s double our investments in after-school and summer school programs, which boost achievement and keep people off the streets and out of trouble. If we do this, we can give every single child in every failing school in America—everyone—the chance to meet high standards. Since 1993, we’ve nearly doubled our investment in Head Start and improved its quality. Tonight I ask you for another $1 billion for Head Start, the largest increase in the history of the program. We know that children learn best in smaller classes with good teachers. For two years in a row, Congress has supported my plan to hire 100,000 new qualified teachers to lower class size in the early grades. I thank you for that, and I ask you to make it three in a row. And to make sure all teachers know the subjects they teach, tonight I propose a new teacher quality initiative, to recruit more talented people into the classroom, reward good teachers for staying there, and give all teachers the training they need. We know charter schools provide real public school choice. When I became President, there was just one independent public charter school in all America. Today, thanks to you, there are 1,700. I ask you now to help us meet our goal of 3,000 charter schools by next year. We know we must connect all our classrooms to the Internet, and we’re getting there. In 1994, only 3 percent of our classrooms were connected. Today, with the help of the Vice President’s E-rate program, more than half of them are, and 90 percent of our schools have at least one Internet connection. But we cannot finish the job when a third of all our schools are in serious disrepair. Many of them have walls and wires so old, they’re too old for the Internet. So tonight I propose to help 5,000 schools a year make immediate and urgent repairs and, again, to help build or modernize 6,000 more, to get students out of trailers and into high-tech classrooms. I ask all of you to help me double our bipartisan GEAR UP program,',
      ' which provides mentors for disadvantaged young people. If we double it, we can provide mentors for 1.4 million of them. Let’s also offer these kids from disadvantaged backgrounds the same chance to take the same college test-prep courses wealthier students use to boost their test scores. To make the American dream achievable for all, we must make college affordable for all. For seven years, on a bipartisan basis, we have taken action toward that goal: larger Pell Grants, more affordable student loans, education IRAs, and our HOPE scholarships, which have already benefited five million young people. Now, 67 percent of high school graduates are going on to college. That’s up 10 percent since 1993. Yet millions of families still strain to pay college tuition. They need help. So I propose a landmark $30 billion college opportunity tax cut, a middle class tax deduction for up to $10,000 in college tuition costs. The previous actions of this Congress have already made two years of college affordable for all. It’s time make four years of college affordable for all. If we take all these steps, we’ll move a long way toward making sure every child starts school ready to learn and graduates ready to succeed. We also need a 21st century revolution to reward work and strengthen families by giving every parent the tools to succeed at work and at the most important work of all, raising children. That means making sure every family has health care and the support to care for aging parents, the tools to bring their children up right, and that no child grows up in poverty. From my first days as President, we’ve worked to give families better access to better health care. In 1997, we passed the Children’s Health Insurance Program—CHIP—so that workers who don’t have coverage through their employers at least can get it for their children. So far, we’ve enrolled two million children. We’re well on our way to our goal of five million. But there are still more than 40 million of our fellow Americans without health insurance, more than there were in 1993. Tonight I propose that we follow Vice President Gore’s suggestion to make low income parents eligible for the insurance that covers their children. Together with our children’s initiative—think of this—together with our children’s initiative, this action would enable us to cover nearly a quarter of all the uninsured people in America. Again, I want to ask you to let people between the ages of 55 and 65, the fastest growing group of uninsured, buy into Medicare. And this year I propose to give them a tax credit to make that choice an affordable one. I hope you will support that, as well. When the baby boomers retire, Medicare will be faced with caring for twice as many of our citizens; yet, it is far from ready to do so. My generation must not ask our children’s generation to shoulder our burden. We simply must act now to strengthen and modernize Medicare. My budget includes a comprehensive plan to reform Medicare, to make it more efficient and more competitive. And it dedicates nearly $400 billion of our budget surplus to keep Medicare solvent past 2025. And at long last, it also provides funds to give every senior a voluntary choice of affordable coverage for prescription drugs. Lifesaving drugs are an indispensable part of modern medicine. No one creating a Medicare program today would even think of excluding coverage for prescription drugs. Yet more than three in five of our seniors now lack dependable drug coverage which can lengthen and enrich their lives. Millions of older Americans, who need prescription drugs the most, pay the highest prices for them. In good conscience, we cannot let another year pass without extending to all our seniors this lifeline of affordable prescription drugs. Record numbers of Americans are providing for aging or ailing loved ones at home. It’s a loving but a difficult and often very expensive choice. Last year, I proposed a $1,000 tax credit for long-term care. Frankly, it wasn’t enough. This year, let’s triple it to $3,000. But this year, let’s pass it. We also have to make needed investments to expand access to mental health care. I want to take a moment to thank the person who led our first White House Conference on Mental Health last year and who for seven years has led all our efforts to break down the barriers to decent treatment of people with mental illness. Thank you, Tipper Gore. Taken together, these proposals would mark the largest investment in health care in the 35 years since Medicare was created—the largest investment in 35 years. That would be a big step toward assuring quality health care for all Americans, young and old. And I ask you to embrace them and pass them. We must also make investments that reward work and support families. Nothing does that better than the earned-income tax credit, the EITC. The "E" in the EITC is about earning, working, taking responsibility, and being rewarded for it. In my very first address to you, I asked Congress to greatly expand this credit, and you did. As a result, in 1998 alone, the EITC helped more than 4.3 million Americans work their way out of poverty toward the middle class. That’s double the num',
      'ber in 1993. Tonight I propose another major expansion of the EITC: to reduce the marriage penalty, to make sure it rewards marriage as it rewards work, and also to expand the tax credit for families that have more than two children. It punishes people with more than two children today. Our proposal would allow families with three or more children to get up to $1,100 more in tax relief. These are working families; their children should not be in poverty. We also can’t reward work and family unless men and women get equal pay for equal work. Today the female unemployment rate is the lowest it has been in 46 years. Yet, women still only earn about 75 cents for every dollar men earn. We must do better, by providing the resources to enforce present equal pay laws, training more women for high-paying, high-tech jobs, and passing the "Paycheck Fairness Act." Many working parents spend up to a quarter—a quarter—of their income on child care. Last year, we helped parents provide child care for about two million children. My child care initiative before you now, along with funds already secured in welfare reform, would make child care better, safer, and more affordable for another 400,000 children. I ask you to pass that. They need it out there. For hard-pressed middle income families, we should also expand the child care tax credit. And I believe strongly we should take the next big step and make that tax credit refundable for low income families. For people making under $30,000 a year, that could mean up to $2,400 for child care costs. You know, we all say we’re pro-work and pro-family. Passing this proposal would prove it. Ten of millions of Americans live from paycheck to paycheck. As hard as they work, they still don’t have the opportunity to save. Too few can make use of IRAs and 401k plans. We should do more to help all working families save and accumulate wealth. That’s the idea behind the Individual Development Accounts, the IDAs. I ask you to take that idea to a new level, with new retirement savings accounts that enable every low and moderate income family in America to save for retirement, a first home, a medical emergency, or a college education. I propose to match their contributions, however small, dollar for dollar, every year they save. And I propose to give a major new tax credit to any small business that will provide a meaningful pension to its workers. Those people ought to have retirement as well as the rest of us. Nearly one in three American children grows up without a father. These children are five times more likely to live in poverty than children with both parents at home. Clearly, demanding and supporting responsible fatherhood is critical to lifting all our children out of poverty. We’ve doubled child support collections since 1992. And I’m proposing to you tough new measures to hold still more fathers responsible. But we should recognize that a lot of fathers want to do right by their children but need help to do it. Carlos Rosas of St. Paul, Minnesota, wanted to do right by his son, and he got the help to do it. Now he’s got a good job, and he supports his little boy. My budget will help 40,000 more fathers make the same choices Carlos Rosas did. I thank him for being here tonight. Stand up, Carlos. [Applause] Thank you. If there is any single issue on which we should be able to reach across party lines, it is in our common commitment to reward work and strengthen families. Just remember what we did last year. We came together to help people with disabilities keep their health insurance when they go to work. And I thank you for that. Thanks to overwhelming bipartisan support from this Congress, we have improved foster care. We’ve helped those young people who leave it when they turn 18, and we have dramatically increased the number of foster care children going into adoptive homes. I thank all of you for all of that. Of course, I am forever grateful to the person who has led our efforts from the beginning and who’s worked so tirelessly for children and families for 30 years now, my wife, Hillary, and I thank her. If we take the steps just discussed, we can go a long, long way toward empowering parents to succeed at home and at work and ensuring that no child is raised in poverty. We can make these vital investments in health care, education, support for working families, and still offer tax cuts to help pay for college, for retirement, to care for aging parents, to reduce the marriage penalty. We can do these things without forsaking the path of fiscal discipline that got us to this point here tonight. Indeed, we must make these investments and these tax cuts in the context of a balanced budget that strengthens and extends the life of Social Security and Medicare and pays down the national debt. Crime in America has dropped for the past seven years—that’s the longest decline on record— thanks to a national consensus we helped to forge on community police, sensible gun safety laws, and effective prevention. But nobody, nobody here, nobody in America believes we’re safe enough. So again, I ask you to set a higher goal. Let’s make this country the safest big country in the world. Last fall, Congress ',
      'supported my plan to hire, in addition to the 100,000 community police we’ve already funded, 50,000 more, concentrated in high-crime neighborhoods. I ask your continued support for that. Soon after the Columbine tragedy, Congress considered commonsense gun legislation, to require Brady background checks at the gun shows, child safety locks for new handguns, and a ban on the importation of large capacity ammunition clips. With courage and a tie-breaking vote by the Vice President—[laughter] —the Senate faced down the gun lobby, stood up for the American people, and passed this legislation. But the House failed to follow suit. Now, we have all seen what happens when guns fall into the wrong hands. Daniel Mauser was only 15 years old when he was gunned down at Columbine. He was an amazing kid, a straight-A student, a good skier. Like all parents who lose their children, his father, Tom, has borne unimaginable grief. Somehow he has found the strength to honor his son by transforming his grief into action. Earlier this month, he took a leave of absence from his job to fight for tougher gun safety laws. I pray that his courage and wisdom will at long last move this Congress to make commonsense gun legislation the very next order of business. Tom Mauser, stand up. We thank you for being here tonight. Tom. Thank you, Tom. [Applause] We must strengthen our gun laws and enforce those already on the books better. Federal gun crime prosecutions are up 16 percent since I took office. But we must do more. I propose to hire more federal and local gun prosecutors and more ATF agents to crack down on illegal gun traffickers and bad-apple dealers. And we must give them the enforcement tools that they need, tools to trace every gun and every bullet used in every gun crime in the United States. I ask you to help us do that. Every State in this country already requires hunters and automobile drivers to have a license. I think they ought to do the same thing for handgun purchases. Now, specifically, I propose a plan to ensure that all new handgun buyers must first have a photo license from their state showing they passed the Brady background check and a gun safety course, before they get the gun. I hope you’ll help me pass that in this Congress. Listen to this—listen to this. The accidental gun rate—the accidental gun death rate of children under 15 in the United States is nine times higher than in the other 25 industrialized countries combined. Now, technologies now exist that could lead to guns that can only be fired by the adults who own them. I ask Congress to fund research into smart gun technology to save these children’s lives. I ask responsible leaders in the gun industry to work with us on smart guns and other steps to keep guns out of the wrong hands, to keep our children safe. You know, every parent I know worries about the impact of violence in the media on their children. I want to begin by thanking the entertainment industry for accepting my challenge to put voluntary ratings on TV programs and video and Internet games. But frankly, the ratings are too numerous, diverse, and confusing to be really useful to parents. So tonight I ask the industry to accept the First Lady’s challenge to develop a single voluntary rating system for all children’s entertainment that is easier for parents to understand and enforce. The steps I outline will take us well on our way to making America the safest big country in the world. Now, to keep our historic economic expansion going, the subject of a lot of discussion in this community and others, I believe we need a 21st century revolution to open new markets, start new businesses, hire new workers right here in America, in our inner cities, poor rural areas, and Native American reservations. Our nation’s prosperity hasn’t yet reached these places. Over the last six months, I’ve traveled to a lot of them, joined by many of you and many far-sighted business people, to shine a spotlight on the enormous potential in communities from Appalachia to the Mississippi Delta, from Watts to the Pine Ridge Reservation. Everywhere I go, I meet talented people eager for opportunity and able to work. Tonight I ask you, let’s put them to work. For business, it’s the smart thing to do. For America, it’s the right thing to do. And let me ask you something: If we don’t do this now, when in the wide world will we ever get around to it? So I ask Congress to give businesses the same incentives to invest in America’s new markets they now have to invest in markets overseas. Tonight I propose a large new markets tax credit and other incentives to spur $22 billion in private-sector capital to create new businesses and new investments in our inner cities and rural areas. Because empowerment zones have been creating these opportunities for five years now, I also ask you to increase incentives to invest in them and to create more of them. And let me say to all of you again what I have tried to say at every turn: This is not a Democratic or a Republican issue. Giving people a chance to live their dreams is an American issue. Mr. Speaker, it was a powerful moment last November when you joined Reverend Jesse Jackson and me in yo',
      'ur home state of Illinois and committed to working toward our common goal by combining the best ideas from both sides of the aisle. I want to thank you again and to tell you, Mr. Speaker, I look forward to working with you. This is a worthy joint endeavor. Thank you. I also ask you to make special efforts to address the areas of our Nation with the highest rates of poverty, our Native American reservations and the Mississippi Delta. My budget includes a $110 million initiative to promote economic development in the Delta and a billion dollars to increase economic opportunity, health care, education, and law enforcement for our Native American communities. We should begin this new century by honoring our historic responsibility to empower the first Americans. And I want to thank tonight the leaders and the members from both parties who’ve expressed to me an interest in working with us on these efforts. They are profoundly important. There’s another part of our American community in trouble tonight, our family farmers. When I signed the farm bill in 1996, I said there was great danger it would work well in good times but not in bad. Well, droughts, floods, and historically low prices have made these times very bad for the farmers. We must work together to strengthen the farm safety net, invest in land conservation, and create some new markets for them by expanding our programs for bio-based fuels and products. Please, they need help. Let’s do it together. Opportunity for all requires something else today, having access to a computer and knowing how to use it. That means we must close the digital divide between those who’ve got the tools and those who don’t. Connecting classrooms and libraries to the Internet is crucial, but it’s just a start. My budget ensures that all new teachers are trained to teach 21st century skills, and it creates technology centers in 1,000 communities to serve adults. This spring, I’ll invite high-tech leaders to join me on another new markets tour, to close the digital divide and open opportunity for our people. I want to thank the high-tech companies that already are doing so much in this area. I hope the new tax incentives I have proposed will get all the rest of them to join us. This is a national crusade. We have got to do this and do it quickly. Now, again I say to you, these are steps, but step by step, we can go a long way toward our goal of bringing opportunity to every community. To realize the full possibilities of this economy, we must reach beyond our own borders to shape the revolution that is tearing down barriers and building new networks among nations and individuals and economies and cultures: globalization. It’s the central reality of our time. Of course, change this profound is both liberating and threatening to people. But there’s no turning back. And our open, creative society stands to benefit more than any other if we understand and act on the realities of interdependence. We have to be at the center of every vital global network, as a good neighbor and a good partner. We have to recognize that we cannot build our future without helping others to build theirs. The first thing we have got to do is to forge a new consensus on trade. Now, those of us who believe passionately in the power of open trade, we have to ensure that it lifts both our living standards and our values, never tolerating abusive child labor or a race to the bottom in the environment and worker protection. But others must recognize that open markets and rule-based trade are the best engines we know of for raising living standards, reducing global poverty and environmental destruction, and assuring the free flow of ideas. I believe, as strongly tonight as I did the first day I got here, the only direction forward for America on trade—the only direction for America on trade is to keep going forward. I ask you to help me forge that consensus. We have to make developing economies our partners in prosperity. That’s why I would like to ask you again to finalize our groundbreaking African and Caribbean Basin trade initiatives. But globalization is about more than economics. Our purpose must be to bring together the world around freedom and democracy and peace and to oppose those who would tear it apart. Here are the fundamental challenges I believe America must meet to shape the 21st century world. First, we must continue to encourage our former adversaries, Russia and China, to emerge as stable, prosperous, democratic nations. Both are being held back today from reaching their full potential: Russia by the legacy of communism, an economy in turmoil, a cruel and self-defeating war in Chechnya; China by the illusion that it can buy stability at the expense of freedom. But think how much has changed in the past decade: 5,000 former Soviet nuclear weapons taken out of commission; Russian soldiers actually serving with ours in the Balkans; Russian people electing their leaders for the first time in 1,000 years; and in China, an economy more open to the world than ever before. Of course, no one, not a single person in this chamber tonight can know for sure what direction these great nations will take. But we do know f',
      'or sure that we can choose what we do. And we should do everything in our power to increase the chance that they will choose wisely, to be constructive members of our global community. That’s why we should support those Russians who are struggling for a democratic, prosperous future; continue to reduce both our nuclear arsenals; and help Russia to safeguard weapons and materials that remain. And that’s why I believe Congress should support the agreement we negotiated to bring China into the WTO, by passing permanent normal trade relations with China as soon as possible this year. I think you ought to do it for two reasons: First of all, our markets are already open to China; this agreement will open China’s markets to us. And second, it will plainly advance the cause of peace in Asia and promote the cause of change in China. No, we don’t know where it’s going. All we can do is decide what we’re going to do. But when all is said and done, we need to know we did everything we possibly could to maximize the chance that China will choose the right future. A second challenge we’ve got is to protect our own security from conflicts that pose the risk of wider war and threaten our common humanity. We can’t prevent every conflict or stop every outrage. But where our interests are at stake and we can make a difference, we should be, and we must be, peacemakers. We should be proud of our role in bringing the Middle East closer to a lasting peace, building peace in Northern Ireland, working for peace in East Timor and Africa, promoting reconciliation between Greece and Turkey and in Cyprus, working to defuse these crises between India and Pakistan, in defending human rights and religious freedom. And we should be proud of the men and women of our Armed Forces and those of our allies who stopped the ethnic cleansing in Kosovo, enabling a million people to return to their homes. When Slobodan Milosevic unleashed his terror on Kosovo, Captain John Cherrey was one of the brave airmen who turned the tide. And when another American plane was shot down over Serbia, he flew into the teeth of enemy air defenses to bring his fellow pilot home. Thanks to our Armed Forces’ skill and bravery, we prevailed in Kosovo without losing a single American in combat. I want to introduce Captain Cherrey to you. We honor Captain Cherrey, and we promise you, Captain, we’ll finish the job you began. Stand up so we can see you. [Applause] A third challenge we have is to keep this inexorable march of technology from giving terrorists and potentially hostile nations the means to undermine our defenses. Keep in mind, the same technological advances that have shrunk cell phones to fit in the palms of our hands can also make weapons of terror easier to conceal and easier to use. We must meet this threat by making effective agreements to restrain nuclear and missile programs in North Korea, curbing the flow of lethal technology to Iran, preventing Iraq from threatening its neighbors, increasing our preparedness against chemical and biological attack, protecting our vital computer systems from hackers and criminals, and developing a system to defend against new missile threats, while working to preserve our ABM missile treaty with Russia. We must do all these things. I predict to you, when most of us are long gone but some time in the next 10 to 20 years, the major security threat this country will face will come from the enemies of the nation-state, the narcotraffickers and the terrorists and the organized criminals who will be organized together, working together, with increasing access to ever more sophisticated chemical and biological weapons. And I want to thank the Pentagon and others for doing what they’re doing right now to try to help protect us and plan for that, so that our defenses will be strong. I ask for your support to ensure they can succeed. I also want to ask you for a constructive bipartisan dialog this year to work to build a consensus which I hope will eventually lead to the ratification of the Comprehensive Nuclear-Test-Ban Treaty. I hope we can also have a constructive effort to meet the challenge that is presented to our planet by the huge gulf between rich and poor. We cannot accept a world in which part of humanity lives on the cutting edge of a new economy and the rest live on the bare edge of survival. I think we have to do our part to change that with expanded trade, expanded aid, and the expansion of freedom. This is interesting: From Nigeria to Indonesia, more people got the right to choose their leaders in 1999 than in 1989, when the Berlin Wall fell. We’ve got to stand by these democracies, including and especially tonight Colombia, which is fighting narcotraffickers, for its own people’s lives and our children’s lives. I have proposed a strong two-year package to help Colombia win this fight. I want to thank the leaders in both parties in both Houses for listening to me and the President of Colombia about it. We have got to pass this. I want to ask your help. A lot is riding on it. And it’s so important for the long-term stability of our country and for what happens in Latin America. I also want you to know I’m going to send you n',
      'ew legislation to go after what these drug barons value the most, their money. And I hope you’ll pass that as well. In a world where over a billion people live on less than a dollar a day, we also have got to do our part in the global endeavor to reduce the debts of the poorest countries, so they can invest in education, health care, and economic growth. That’s what the Pope and other religious leaders have urged us to do. And last year, Congress made a downpayment on America’s share. I ask you to continue that. I thank you for what you did and ask you to stay the course. I also want to say that America must help more nations to break the bonds of disease. Last year in Africa, 10 times as many people died from AIDS as were killed in wars—10 times. The budget I give you invests $150 million more in the fight against this and other infectious killers. And today I propose a tax credit to speed the development of vaccines for diseases like malaria, TB, and AIDS. I ask the private sector and our partners around the world to join us in embracing this cause. We can save millions of lives together, and we ought to do it. I also want to mention our final challenge, which, as always, is the most important. I ask you to pass a national security budget that keeps our military the best trained and best equipped in the world, with heightened readiness and 21st century weapons, which raises salaries for our service men and women, which protects our veterans, which fully funds the diplomacy that keeps our soldiers out of war, which makes good on our commitment to our U.N. dues and arrears. I ask you to pass this budget. I also want to say something, if I might, very personal tonight. The American people watching us at home, with the help of all the commentators, can tell, from who stands and who sits and who claps and who doesn’t, that there’s still modest differences of opinion in this room. [Laughter] But I want to thank you for something, every one of you. I want to thank you for the extraordinary support you have given, Republicans and Democrats alike, to our men and women in uniform. I thank you for that. I also want to thank, especially, two people. First, I want to thank our Secretary of Defense, Bill Cohen, for symbolizing our bipartisan commitment to national security. Thank you, sir. Even more, I want to thank his wife, Janet, who, more than any other American citizen, has tirelessly traveled this world to show the support we all feel for our troops. Thank you, Janet Cohen. I appreciate that. Thank you. These are the challenges we have to meet so that we can lead the world toward peace and freedom in an era of globalization. I want to tell you that I am very grateful for many things as President. But one of the things I’m grateful for is the opportunity that the Vice President and I have had to finally put to rest the bogus idea that you cannot grow the economy and protect the environment at the same time. As our economy has grown, we’ve rid more than 500 neighborhoods of toxic waste, ensured cleaner air and water for millions of people. In the past three months alone, we’ve helped preserve 40 million acres of roadless lands in the national forests, created three new national monuments. But as our communities grow, our commitment to conservation must continue to grow. Tonight I propose creating a permanent conservation fund, to restore wildlife, protect coastlines, save natural treasures, from the California redwoods to the Florida Everglades. This lands legacy endowment would represent by far the most enduring investment in land preservation ever proposed in this House. I hope we can get together with all the people with different ideas and do this. This is a gift we should give to our children and our grandchildren for all time, across party lines. We can make an agreement to do this. Last year the Vice President launched a new effort to make communities more liberal—liv-able—[laughter]—liberal, I know. [Laughter] Wait a minute, I’ve got a punchline now. That’s this year’s agenda; last year was livable, right? [Laughter] That’s what Senator Lott is going to say in the commentary afterwards—[laugh-ter] —to make our communities more livable. This is big business. This is a big issue. What does that mean? You ask anybody that lives in an unlivable community, and they’ll tell you. They want their kids to grow up next to parks, not parking lots; the parents don’t have to spend all their time stalled in traffic when they could be home with their children. Tonight I ask you to support new funding for the following things, to make American communities more liberal—livable. [Laughter] I’ve done pretty well with this speech, but I can’t say that. One, I want you to help us to do three things. We need more funding for advanced transit systems. We need more funding for saving open spaces in places of heavy development. And we need more funding—this ought to have bipartisan appeal—we need more funding for helping major cities around the Great Lakes protect their waterways and enhance their quality of life. We need these things, and I want you to help us. The greatest environmental challenge of the new century is global warming. The scie',
      'ntists tell us the 1990s were the hottest decade of the entire millennium. If we fail to reduce the emission of greenhouse gases, deadly heat waves and droughts will become more frequent, coastal areas will flood, and economies will be disrupted. That is going to happen, unless we act. Many people in the United States, some people in this chamber, and lots of folks around the world still believe you cannot cut greenhouse gas emissions without slowing economic growth. In the industrial age, that may well have been true. But in this digital economy, it is not true anymore. New technologies make it possible to cut harmful emissions and provide even more growth. For example, just last week, automakers unveiled cars that get 70 to 80 miles a gallon, the fruits of a unique research partnership between government and industry. And before you know it, efficient production of bio-fuels will give us the equivalent of hundreds of miles from a gallon of gasoline. To speed innovation in these kind of technologies, I think we should give a major tax incentive to business for the production of clean energy and to families for buying energy-saving homes and appliances and the next generation of superefficient cars when they hit the showroom floor. I also ask the auto industry to use the available technologies to make all new cars more fuel-efficient right away. And I ask this Congress to do something else. Please help us make more of our clean energy technology available to the developing world. That will create cleaner growth abroad and a lot more new jobs here in the United States of America. In the new century, innovations in science and technology will be key not only to the health of the environment but to miraculous improvements in the quality of our lives and advances in the economy. Later this year, researchers will complete the first draft of the entire human genome, the very blueprint of life. It is important for all our fellow Americans to recognize that federal tax dollars have funded much of this research and that this and other wise investments in science are leading to a revolution in our ability to detect, treat, and prevent disease. For example, researchers have identified genes that cause Parkinson’s, diabetes, and certain kinds of cancer. They are designing precision therapies that will block the harmful effect of these genes for good. Researchers already are using this new technique to target and destroy cells that cause breast cancer. Soon, we may be able to use it to prevent the onset of Alzheimer’s. Scientists are also working on an artificial retina to help many blind people to see and—listen to this—microchips that would actually directly stimulate damaged spinal cords in a way that could allow people now paralyzed to stand up and walk. These kinds of innovations are also propelling our remarkable prosperity. Information technology only includes 8 percent of our employment but now accounts for a third of our economic growth along with jobs that pay, by the way, about 80 percent above the private sector average. Again, we ought to keep in mind, government-funded research brought supercomputers, the Internet, and communications satellites into being. Soon researchers will bring us devices that can translate foreign languages as fast as you can talk, materials 10 times stronger than steel at a fraction of the weight, and—this is unbelievable to me—molecular computers the size of a teardrop with the power of today’s fastest supercomputers. To accelerate the march of discovery across all these disciplines in science and technology, I ask you to support my recommendation of an unprecedented $3 billion in the 21st century research fund, the largest increase in civilian research in a generation. We owe it to our future. Now, these new breakthroughs have to be used in ways that reflect our values. First and foremost, we have to safeguard our citizens’ privacy. Last year we proposed to protect every citizen’s medical record. This year we will finalize those rules. We’ve also taken the first steps to protect the privacy of bank and credit card records and other financial statements. Soon I will send legislation to you to finish that job. We must also act to prevent any genetic discrimination whatever by employers or insurers. I hope you will support that. These steps will allow us to lead toward the far frontiers of science and technology. They will enhance our health, the environment, the economy in ways we can’t even imagine today. But we all know that at a time when science, technology, and the forces of globalization are bringing so many changes into all our lives, it’s more important than ever that we strengthen the bonds that root us in our local communities and in our national community. No tie binds different people together like citizen service. There’s a new spirit of service in America, a movement we’ve tried to support with AmeriCorps, expanded Peace Corps, unprecedented new partnerships with businesses, foundations, community groups; partnerships, for example, like the one that enlisted 12,000 companies which have now moved 650,000 of our fellow citizens from welfare to work; partnerships to battle drug abuse',
      ', AIDS, teach young people to read, save America’s treasures, strengthen the arts, fight teen pregnancy, prevent violence among young people, promote racial healing. The American people are working together. But we should do more to help Americans help each other. First, we should help faith-based organizations to do more to fight poverty and drug abuse and help people get back on the right track, with initiatives like Second Chance Homes that do so much to help unwed teen mothers. Second, we should support Americans who tithe and contribute to charities but don’t earn enough to claim a tax deduction for it. Tonight I propose new tax incentives that would allow low and middle income citizens who don’t itemize to get that deduction. It’s nothing but fair, and it will get more people to give. We should do more to help new immigrants to fully participate in our community. That’s why I recommend spending more to teach them civics and English. And since everybody in our community counts, we’ve got to make sure everyone is counted in this year’s census. Within 10 years—just 10 years—there will be no majority race in our largest state of California. In a little more than 50 years, there will be no majority race in America. In a more interconnected world, this diversity can be our greatest strength. Just look around this chamber. Look around. We have members in this Congress from virtually every racial, ethnic, and religious background. And I think you would agree that America is stronger because of it. [Applause] You also have to agree that all those differences you just clapped for all too often spark hatred and division even here at home. Just in the last couple of years, we’ve seen a man dragged to death in Texas just because he was black. We saw a young man murdered in Wyoming just because he was gay. Last year we saw the shootings of African-Americans, Asian-Americans, and Jewish children just because of who they were. This is not the American way, and we must draw the line. I ask you to draw that line by passing without delay the "Hate Crimes Prevention Act" and the "Employment Non-Discrimination Act." And I ask you to reauthorize the Violence Against Women Act. Finally tonight, I propose the largest ever investment in our civil rights laws for enforcement, because no American should be subjected to discrimination in finding a home, getting a job, going to school, or securing a loan. Protections in law should be protections in fact. Last February, because I thought this was so important, I created the White House Office of One America to promote racial reconciliation. That’s what one of my personal heroes, Hank Aaron, has done all his life. From his days as our all-time home run king to his recent acts of healing, he has always brought people together. We should follow his example, and we’re honored to have him with us tonight. Stand up, Hank Aaron. [Applause] I just want to say one more thing about this, and I want every one of you to think about this the next time you get mad at one of your colleagues on the other side of the aisle. This fall, at the White House, Hillary had one of her millennium dinners, and we had this very distinguished scientist there, who is an expert in this whole work in the human genome. And he said that we are all, regardless of race, genetically 99.9 percent the same. Now, you may find that uncomfortable when you look around here. [Laughter] But it is worth remembering. We can laugh about this, but you think about it. Modern science has confirmed what ancient faiths have always taught: the most important fact of life is our common humanity. Therefore, we should do more than just tolerate our diversity; we should honor it and celebrate it. My fellow Americans, every time I prepare for the State of the Union, I approach it with hope and expectation and excitement for our nation. But tonight is very special, because we stand on the mountaintop of a new millennium. Behind us we can look back and see the great expanse of American achievement, and before us we can see even greater, grander frontiers of possibility. We should, all of us, be filled with gratitude and humility for our present progress and prosperity. We should be filled with awe and joy at what lies over the horizon. And we should be filled with absolute determination to make the most of it. You know, when the Framers finished crafting our Constitution in Philadelphia, Benjamin Franklin stood in Independence Hall, and he reflected on the carving of the sun that was on the back of a chair he saw. The sun was low on the horizon. So he said this—he said, "I’ve often wondered whether that sun was rising or setting. Today," Franklin said, "I have the happiness to know it’s a rising sun." Today, because each succeeding generation of Americans has kept the fire of freedom burning brightly, lighting those frontiers of possibility, we all still bask in the glow and the warmth of Mr. Franklin’s rising sun. After 224 years, the American revolution continues. We remain a new nation. And as long as our dreams outweigh our memories, America will be forever young. That is our destiny. And this is our moment. Thank you, God bless you, and God bless A'],
     ["Madam Speaker, Vice President Cheney, members of Congress, distinguished guests, and fellow citizens: Seven years have passed since I first stood before you at this rostrum. In that time, our country has been tested in ways none of us could have imagined. We faced hard decisions about peace and war, rising competition in the world economy, and the health and welfare of our citizens. These issues call for vigorous debate, and I think it's fair to say, we've answered the call. Yet history will record that amid our differences, we acted with purpose, and together we showed the world the power and resilience of American self-government. All of us were sent to Washington to carry out the people's business. That is the purpose of this body. It is the meaning of our oath. It remains our charge to keep. The actions of the 110th Congress will affect the security and prosperity of our nation long after this session has ended. In this election year, let us show our fellow Americans that we recognize our responsibilities and are determined to meet them. Let us show them that Republicans and Democrats can compete for votes and cooperate for results at the same time. From expanding opportunity to protecting our country, we've made good progress. Yet we have unfinished business before us, and the American people expect us to get it done. In the work ahead, we must be guided by the philosophy that made our nation great. As Americans, we believe in the power of individuals to determine their destiny and shape the course of history. We believe that the most reliable guide for our country is the collective wisdom of ordinary citizens. And so in all we do, we must trust in the ability of free peoples to make wise decisions and empower them to improve their lives for their futures. To build a prosperous future, we must trust people with their own money and empower them to grow our economy. As we meet tonight, our economy is undergoing a period of uncertainty. America has added jobs for a record 52 straight months, but jobs are now growing at a slower pace. Wages are up, but so are prices for food and gas. Exports are rising, but the housing market has declined. At kitchen tables across our country, there is a concern about our economic future. In the long run, Americans can be confident about our economic growth. But in the short run, we can all see that that growth is slowing. So last week, my administration reached agreement with Speaker Pelosi and Republican Leader Boehner on a robust growth package that includes tax relief for individuals and families and incentives for business investment. The temptation will be to load up the bill. That would delay it or derail it, and neither option is acceptable. This is a good agreement that will keep our economy growing and our people working, and this Congress must pass it as soon as possible. We have other work to do on taxes. Unless Congress acts, most of the tax relief we've delivered over the past seven years will be taken away. Some in Washington argue that letting tax relief expire is not a tax increase. Try explaining that to 116 million American taxpayers who would see their taxes rise by an average of $1,800. Others have said they would personally be happy to pay higher taxes. I welcome their enthusiasm. I'm pleased to report that the IRS accepts both checks and money o",
      "rders. Most Americans think their taxes are high enough. With all the other pressures on their finances, American families should not have to worry about their federal government taking a bigger bite out of their paychecks. There's only one way to eliminate this uncertainty: Make the tax relief permanent. And members of Congress should know, if any bill raises taxes reaches my desk, I will veto it. Just as we trust Americans with their own money, we need to earn their trust by spending their tax dollars wisely. Next week, I'll send you a budget that terminates or substantially reduces 151 wasteful or bloated programs, totaling more than $18 billion. The budget that I will submit will keep America on track for a surplus in 2012. American families have to balance their budgets; so should their government. The people's trust in their government is undermined by congressional earmarks, special interest projects that are often snuck in at the last minute, without discussion or debate. Last year, I asked you to voluntarily cut the number and cost of earmarks in half. I also asked you to stop slipping earmarks into committee reports that never even come to a vote. Unfortunately, neither goal was met. So this time, if you send me an appropriations bill that does not cut the number and cost of earmarks in half, I'll send it back to you with my veto. And tomorrow I will issue an executive order that directs federal agencies to ignore any future earmark that is not voted on by Congress. If these items are truly worth funding, Congress should debate them in the open and hold a public vote. Our shared responsibilities extend beyond matters of taxes and spending. On housing, we must trust Americans with the responsibility of homeownership and empower them to weather turbulent times in the housing market. My administration brought together the HOPE NOW Alliance, which is helping many struggling homeowners avoid foreclosure. And Congress can help even more. Tonight I ask you to pass legislation to reform Fannie Mae and Freddie Mac, modernize the Federal Housing Administration, and allow state housing agencies to issue tax-free bonds to help homeowners refinance their mortgages. These are difficult times for many American families, and by taking these steps, we can help more of them keep their homes. To build a future of quality health care, we must trust patients and doctors to make medical decisions and empower them with better information and better options. We share a common goal: making health care more affordable and accessible for all Americans. The best way to achieve that goal is by expanding consumer choice, not government control. So I have proposed ending the bias in the Tax Code against those who do not get their health insurance through their employer. This one reform would put private coverage within reach for millions, and I call on the Congress to pass it this year. The Congress must also expand health savings accounts, create association health plans for small businesses, promote health information technology, and confront the epidemic of junk medical lawsuits. With all these steps, we will ensure that decisions about your medical care are made in the privacy of your doctor's office, not in the halls of Congress. On education, we must trust students to learn, if given the chance, and empower parents t",
      'o demand results from our schools. In neighborhoods across our country, there are boys and girls with dreams, and a decent education is their only hope of achieving them. Six years ago, we came together to pass the No Child Left Behind Act, and today, no one can deny its results. Last year, fourth and eighth graders achieved the highest math scores on record. Reading scores are on the rise. African-American and Hispanic students posted alltime highs. Now we must work together to increase accountability, add flexibilities for states and districts, reduce the number of high school dropouts, provide extra help for struggling schools. Members of Congress, the No Child Left Behind Act is a bipartisan achievement. It is succeeding. And we owe it to America\'s children, their parents, and their teachers to strengthen this good law. We must also do more to help children when their schools do not measure up. Thanks to the DC Opportunity Scholarships you approved, more than 2,600 of the poorest children in our nation\'s capital have found new hope at a faith-based or other non-public school. Sadly, these schools are disappearing at an alarming rate in many of America\'s inner cities. So I will convene a White House summit aimed at strengthening these lifelines of learning. And to open the doors of these schools to more children, I ask you to support a new $300 million program called Pell Grants for Kids. We have seen how Pell Grants help low-income college students realize their full potential. Together we\'ve expanded the size and reach of these grants. Now let us apply the same spirit to help liberate poor children trapped in failing public schools. On trade, we must trust American workers to compete with anyone in the world and empower them by opening up new markets overseas. Today, our economic growth increasingly depends on our ability to sell American goods and crops and services all over the world. So we\'re working to break down barriers to trade and investment wherever we can. We\'re working for a successful Doha round of trade talks, and we must complete a good agreement this year. At the same time, we\'re pursuing opportunities to open up new markets by passing free trade agreements. I thank the Congress for approving a good agreement with Peru. And now I ask you to approve agreements with Colombia and Panama and South Korea. Many products from these nations now enter America duty free, yet many of our products face steep tariffs in their markets. These agreements will level the playing field. They will give us better access to nearly 100 million customers. They will support good jobs for the finest workers in the world, those whose products say "Made in the USA." These agreements also promote America\'s strategic interests. The first agreement that will come before you is with Colombia, a friend of America that is confronting violence and terror and fighting drug traffickers. If we fail to pass this agreement, we will embolden the purveyors of false populism in our hemisphere. So we must come together, pass this agreement, and show our neighbors in the region that democracy leads to a better life. Trade brings better jobs and better choices and better prices. Yet for some Americans, trade can mean losing a job, and the federal government has a responsibility to help. I ask Congress to reauthorize and reform ',
      "trade adjustment assistance so we can help these displaced workers learn new skills and find new jobs. To build a future of energy security, we must trust in the creative genius of American researchers and entrepreneurs and empower them to pioneer a new generation of clean energy technology. Our security, our prosperity, and our environment all require reducing our dependence on oil. Last year, I asked you to pass legislation to reduce oil consumption over the next decade, and you responded. Together we should take the next steps. Let us fund new technologies that can generate coal power while capturing carbon emissions. Let us increase the use of renewable power and emissions-free nuclear power. Let us continue investing in advanced battery technology and renewable fuels to power the cars and trucks of the future. Let us create a new international clean technology fund, which will help developing nations like India and China make a greater use of clean energy sources. And let us complete an international agreement that has the potential to slow, stop, and eventually reverse the growth of greenhouse gases. This agreement will be effective only if it includes commitments by every major economy and gives none a free ride. The United States is committed to strengthening our energy security and confronting global climate change. And the best way to meet these goals is for America to continue leading the way toward the development of cleaner and more energy efficient technology. To keep America competitive into the future, we must trust in the skill of our scientists and engineers and empower them to pursue the breakthroughs of tomorrow. Last year, Congress passed legislation supporting the American Competitiveness Initiative, but never followed through with the funding. This funding is essential to keeping our scientific edge. So I ask Congress to double federal support for critical basic research in the physical sciences and ensure America remains the most dynamic nation on Earth. On matters of life and science, we must trust in the innovative spirit of medical researchers and empower them to discover new treatments while respecting moral boundaries. In November, we witnessed a landmark achievement when scientists discovered a way to reprogram adult skin cells to act like embryonic stem cells. This breakthrough has the potential to move us beyond the divisive debates of the past by extending the frontiers of medicine without the destruction of human life. So we're expanding funding for this type of ethical medical research. And as we explore promising avenues of research, we must also ensure that all life is treated with the dignity it deserves. And so I call on Congress to pass legislation that bans unethical practices, such as the buying, selling, patenting, or cloning of human life. On matters of justice, we must trust in the wisdom of our Founders and empower judges who understand that the Constitution means what it says. I've submitted judicial nominees who will rule by the letter of the law, not the whim of the gavel. Many of these nominees are being unfairly delayed. They are worthy of confirmation, and the Senate should give each of them a prompt up-or-down vote. In communities across our land, we must trust in the good heart of the American people and empower them to serve their neighbors in need",
      ". Over the past seven years, more of our fellow citizens have discovered that the pursuit of happiness leads to the path of service. Americans have volunteered in record numbers. Charitable donations are higher than ever. Faith-based groups are bringing hope to pockets of despair, with newfound support from the federal government. And to help guarantee equal treatment of faith-based organizations when they compete for federal funds, I ask you to permanently extend charitable choice. Tonight the armies of compassion continue the march to a new day in the gulf coast. America honors the strength and resilience of the people of this region. We reaffirm our pledge to help them build stronger and better than before. And tonight I'm pleased to announce that in April, we will host this year's North American Summit of Canada, Mexico, and the United States in the great city of New Orleans. There are two other pressing challenges that I've raised repeatedly before this body and that this body has failed to address: entitlement spending and immigration. Every member in this chamber knows that spending on entitlement programs like Social Security, Medicare, and Medicaid is growing faster than we can afford. We all know the painful choices ahead if America stays on this path: massive tax increases, sudden and drastic cuts in benefits, or crippling deficits. I've laid out proposals to reform these programs. Now I ask members of Congress to offer your proposals and come up with a bipartisan solution to save these vital programs for our children and our grandchildren. The other pressing challenge is immigration. America needs to secure our borders, and with your help, my administration is taking steps to do so. We're increasing worksite enforcement, deploying fences and advanced technologies to stop illegal crossings. We've effectively ended the policy of catch-and-release at the border, and by the end of this year, we will have doubled the number of Border Patrol agents. Yet we also need to acknowledge that we will never fully secure our border until we create a lawful way for foreign workers to come here and support our economy. This will take pressure off the border and allow law enforcement to concentrate on those who mean us harm. We must also find a sensible and humane way to deal with people here illegally. Illegal immigration is complicated, but it can be resolved. And it must be resolved in a way that upholds both our laws and our highest ideals. This is the business of our nation here at home. Yet building a prosperous future for our citizen also depends on confronting enemies abroad and advancing liberty in troubled regions of the world. Our foreign policy is based on a clear premise: We trust that people, when given the chance, will choose a future of freedom and peace. In the last seven years, we have witnessed stirring moments in the history of liberty. We've seen citizens in Georgia and Ukraine stand up for their right to free and fair elections. We've seen people in Lebanon take to the streets to demand their independence. We've seen Afghans emerge from the tyranny of the Taliban and choose a new President and a new Parliament. We've seen jubilant Iraqis holding up ink-stained fingers and celebrating their freedom. These images of liberty have inspired us. In the past seven years, we've also seen the im",
      "ages that have sobered us. We've watched throngs of mourners in Lebanon and Pakistan carrying the caskets of beloved leaders taken by the assassin's hand. We've seen wedding guests in blood-soaked finery staggering from a hotel in Jordan, Afghans and Iraqis blown up in mosques and markets, and trains in London and Madrid ripped apart by bombs. On a clear September day, we saw thousands of our fellow citizens taken from us in an instant. These horrific images serve as a grim reminder: The advance of liberty is opposed by terrorists and extremists, evil men who despise freedom, despise America, and aim to subject millions to their violent rule. Since 9/11, we have taken the fight to these terrorists and extremists. We will stay on the offense; we will keep up the pressure; and we will deliver justice to our enemies. We are engaged in the defining ideological struggle of the 21st century. The terrorists oppose every principle of humanity and decency that we hold dear. Yet in this war on terror, there is one thing we and our enemies agree on: In the long run, men and women who are free to determine their own destinies will reject terror and refuse to live in tyranny. And that is why the terrorists are fighting to deny this choice to the people in Lebanon, Iraq, Afghanistan, Pakistan, and the Palestinian Territories. And that is why, for the security of America and the peace of the world, we are spreading the hope of freedom. In Afghanistan, America, our 25 NATO allies, and 15 partner nations are helping the Afghan people defend their freedom and rebuild their country. Thanks to the courage of these military and civilian personnel, a nation that was once a safe haven for Al Qaeda is now a young democracy where boys and girls are going to school, new roads and hospitals are being built, and people are looking to the future with new hope. These successes must continue, so we're adding 3,200 marines to our forces in Afghanistan, where they will fight the terrorists and train the Afghan Army and police. Defeating the Taliban and Al Qaeda is critical to our security, and I thank the Congress for supporting America's vital mission in Afghanistan. In Iraq, the terrorists and extremists are fighting to deny a proud people their liberty and fighting to establish safe havens for attacks across the world. One year ago, our enemies were succeeding in their efforts to plunge Iraq into chaos. So we reviewed our strategy and changed course. We launched a surge of American forces into Iraq. We gave our troops a new mission: Work with the Iraqi forces to protect the Iraqi people; pursue the enemy in its strongholds; and deny the terrorists sanctuary anywhere in the country. The Iraqi people quickly realized that something dramatic had happened. Those who had worried that America was preparing to abandon them instead saw tens of thousands of American forces flowing into their country. They saw our forces moving into neighborhoods, clearing out the terrorists, and staying behind to ensure the enemy did not return. And they saw our troops, along with Provincial Reconstruction Teams that include Foreign Service officers and other skilled public servants, coming in to ensure that improved security was followed by improvements in daily life. Our military and civilians in Iraq are performing with courage and distinction, and they ",
      'have the gratitude of our whole nation. The Iraqis launched a surge of their own. In the fall of 2006, Sunni tribal leaders grew tired of Al Qaeda\'s brutality, started a popular uprising called the "Anbar Awakening." Over the past year, similar movements have spread across the country. And today, the grassroots surge includes more than 80,000 Iraqi citizens who are fighting the terrorists. The government in Baghdad has stepped forward as well, adding more than 100,000 new Iraqi soldiers and police during the past year. While the enemy is still dangerous and more work remains, the American and Iraqi surges have achieved results few of us could have imagined just one year ago. When we met last year, many said that containing the violence was impossible. A year later, high-profile terrorist attacks are down, civilian deaths are down, sectarian killings are down. When we met last year, militia extremists—some armed and trained by Iran—were wreaking havoc in large areas of Iraq. A year later, coalition and Iraqi forces have killed or captured hundreds of militia fighters. And Iraqis of all backgrounds increasingly realize that defeating these militia fighters is critical to the future of their country. When we met last year, Al Qaeda had sanctuaries in many areas of Iraq, and their leaders had just offered American forces safe passage out of the country. Today, it is Al Qaeda that is searching for safe passage. They have been driven from many of the strongholds they once held. And over the past year, we\'ve captured or killed thousands of extremists in Iraq, including hundreds of key Al Qaeda leaders and operatives. Last month, Osama bin Laden released a tape in which he railed against Iraqi tribal leaders who have turned on Al Qaeda and admitted that coalition forces are growing stronger in Iraq. Ladies and gentlemen, some may deny the surge is working, but among the terrorists there is no doubt. Al Qaeda is on the run in Iraq, and this enemy will be defeated. When we met last year, our troop levels in Iraq were on the rise. Today, because of the progress just described, we are implementing a policy of return on success, and the surge forces we sent to Iraq are beginning to come home. This progress is a credit to the valor of our troops and the brilliance of their commanders. This evening I want to speak directly to our men and women on the frontlines. Soldiers and sailors, airmen, marines, and coast guardsmen: In the past year, you have done everything we\'ve asked of you and more. Our nation is grateful for your courage. We are proud of your accomplishments. And tonight in this hallowed chamber, with the American people as our witness, we make you a solemn pledge: In the fight ahead, you will have all you need to protect our nation. And I ask Congress to meet its responsibilities to these brave men and women by fully funding our troops. Our enemies in Iraq have been hit hard. They are not yet defeated, and we can still expect tough fighting ahead. Our objective in the coming year is to sustain and build on the gains we made in 2007 while transitioning to the next phase of our strategy. American troops are shifting from leading operations to partnering with Iraqi forces and, eventually, to a protective overwatch mission. As part of this transition, one Army brigade combat team and one Marine expeditionary u',
      'nit have already come home and will not be replaced. In the coming months, four additional brigades and two Marine battalions will follow suit. Taken together, this means more than 20,000 of our troops are coming home. Any further drawdown of U.S. troops will be based on conditions in Iraq and the recommendations of our commanders. General Petraeus has warned that too fast a drawdown could result in, quote, "the disintegration of the Iraqi security forces, Al Qaeda-Iraq regaining lost ground, and a marked increase in violence." Members of Congress, having come so far and achieved so much, we must not allow this to happen. In the coming year, we will work with Iraqi leaders as they build on the progress they\'re making toward political reconciliation. At the local level, Sunnis, Shi\'a, and Kurds are beginning to come together to reclaim their communities and rebuild their lives. Progress in the provinces must be matched by progress in Baghdad. We\'re seeing some encouraging signs. The national government is sharing oil revenues with the provinces. The Parliament recently passed both a pension law and de-Ba\'athification reform. They\'re now debating a provincial powers law. The Iraqis still have a distance to travel, but after decades of dictatorship and the pain of sectarian violence, reconciliation is taking place, and the Iraqi people are taking control of their future. The mission in Iraq has been difficult and trying for our nation. But it is in the vital interest of the United States that we succeed. A free Iraq will deny Al Qaeda a safe haven. A free Iraq will show millions across the Middle East that a future of liberty is possible. A free Iraq will be a friend of America, a partner in fighting terror, and a source of stability in a dangerous part of the world. By contrast, a failed Iraq would embolden the extremists, strengthen Iran, and give terrorists a base from which to launch new attacks on our friends, our allies, and our homeland. The enemy has made its intentions clear. At a time when the momentum seemed to favor them, Al Qaeda\'s top commander in Iraq declared that they will not rest until they have attacked us here in Washington. My fellow Americans, we will not rest either. We will not rest until this enemy has been defeated. We must do the difficult work today so that years from now, people will look back and say that this generation rose to the moment, prevailed in a tough fight, and left behind a more hopeful region and a safer America. We\'re also standing against the forces of extremism in the Holy Land, where we have new cause for hope. Palestinians have elected a President who recognizes that confronting terror is essential to achieving a state where his people can live in dignity and at peace with Israel. Israelis have leaders who recognize that a peaceful, democratic Palestinian state will be a source of lasting security. This month in Ramallah and Jerusalem, I assured leaders from both sides that America will do, and I will do, everything we can to help them achieve a peace agreement that defines a Palestinian state by the end of this year. The time has come for a Holy Land where a democratic Israel and a democratic Palestine live side by side in peace. We\'re also standing against the forces of extremism embodied by the regime in Tehran. Iran\'s rulers oppress a good and talented ',
      "people. And wherever freedom advances in the Middle East, it seems the Iranian regime is there to oppose it. Iran is funding and training militia groups in Iraq, supporting Hizballah terrorists in Lebanon, and backing Hamas efforts to undermine peace in the Holy Land. Tehran is also developing ballistic missiles of increasing range and continues to develop its capability to enrich uranium, which could be used to create a nuclear weapon. Our message to the people of Iran is clear: We have no quarrel with you. We respect your traditions and your history. We look forward to the day when you have your freedom. Our message to the leaders of Iran is also clear: Verifiably suspend your nuclear enrichment so negotiations can begin. And to rejoin the community of nations, come clean about your nuclear intentions and past actions, stop your oppression at home, cease your support for terror abroad. But above all, know this: America will confront those who threaten our troops; we will stand by our allies; and we will defend our vital interests in the Persian Gulf. On the homefront, we will continue to take every lawful and effective measure to protect our country. This is our most solemn duty. We are grateful that there has not been another attack on our soil since 9/11. This is not for the lack of desire or effort on the part of the enemy. In the past six years, we've stopped numerous attacks, including a plot to fly a plane into the tallest building in Los Angeles and another to blow up passenger jets bound for America over the Atlantic. Dedicated men and women in our government toil day and night to stop the terrorists from carrying out their plans. These good citizens are saving American lives, and everyone in this chamber owes them our thanks. And we owe them something more; we owe them the tools they need to keep our people safe. And one of the most important tools we can give them is the ability to monitor terrorist communications. To protect America, we need to know who the terrorists are talking to, what they are saying, and what they're planning. Last year, Congress passed legislation to help us do that. Unfortunately, Congress set the legislations to expire on February 1. That means if you don't act by Friday, our ability to track terrorist threats would be weakened and our citizens will be in greater danger. Congress must ensure the flow of vital intelligence is not disrupted. Congress must pass liability protection for companies believed to have assisted in the efforts to defend America. We've had ample time for debate. The time to act is now. Protecting our nation from the dangers of a new century requires more than good intelligence and a strong military. It also requires changing the conditions that breed resentment and allow extremists to prey on despair. So America is using its influence to build a freer, more hopeful, and more compassionate world. This is a reflection of our national interests; it is the calling of our conscience. America opposes genocide in Sudan. We support freedom in countries from Cuba and Zimbabwe to Belarus and Burma. America is leading the fight against global poverty with strong education initiatives and humanitarian assistance. We've also changed the way we deliver aid by launching the Millennium Challenge Account. This program strengthens democracy, transparency, and t",
      'he rule of law in developing nations, and I ask you to fully fund this important initiative. America is leading the fight against global hunger. Today, more than half the world\'s food aid comes from the United States. And tonight I ask Congress to support an innovative proposal to provide food assistance by purchasing crops directly from farmers in the developing world, so we can build up local agriculture and help break the cycle of famine. America is leading the fight against disease. With your help, we\'re working to cut by half the number of malaria-related deaths in 15 African nations. And our Emergency Plan for AIDS Relief is treating 1.4 million people. We can bring healing and hope to many more. So I ask you to maintain the principles that have changed behavior and made this program a success. And I call on you to double our initial commitment to fighting HIV/AIDS by approving an additional $30 billion over the next five years. America is a force for hope in the world because we are a compassionate people, and some of the most compassionate Americans are those who have stepped forward to protect us. We must keep faith with all who have risked life and limb so that we might live in freedom and peace. Over the past seven years, we\'ve increased funding for veterans by more than 95 percent. And as we increase funding, we must also reform our veterans system to meet the needs of a new war and a new generation. I call on Congress to enact the reforms recommended by Senator Bob Dole and Secretary Donna Shalala, so we can improve the system of care for our wounded warriors and help them build lives of hope and promise and dignity. Our military families also sacrifice for America. They endure sleepless nights and the daily struggle of providing for children while a loved one is serving far from home. We have a responsibility to provide for them. So I ask you to join me in expanding their access to child care, creating new hiring preferences for military spouses across the federal government, and allowing our troops to transfer their unused education benefits to their spouses or children. Our military families serve our nation; they inspire our nation; and tonight our nation honors them. The strength—the secret of our strength, the miracle of America is that our greatness lies not in our government, but in the spirit and determination of our people. When the federal convention met in Philadelphia in 1787, our nation was bound by the Articles of Confederation, which began with the words, "We the undersigned delegates." When Governor Morris was asked to draft the preamble to our new Constitution, he offered an important revision and opened with words that changed the course of our nation and the history of the world: "We the people." By trusting the people, our Founders wagered that a great and noble nation could be built on the liberty that resides in the hearts of all men and women. By trusting the people, succeeding generations transformed our fragile young democracy into the most powerful nation on Earth and a beacon of hope for millions. And so long as we continue to trust the people, our nation will prosper, our liberty will be secure, and the state of our Union will remain strong. So tonight, with confidence in freedom\'s power and trust in the people, let us set forth to do their business. God bless '],
     ['Mr. Vice President, Mr. Speaker, members of the 78th Congress:\n\r\nThis 78th Congress assembles in one of the great moments in the history of the nation. The past year was perhaps the most crucial for modern civilization; the coming year will be filled with violent conflicts—yet with high promise of better things.\n\r\nWe must appraise the events of 1942 according to their relative importance; we must exercise a sense of proportion.\n\r\nFirst in importance in the American scene has been the inspiring proof of the great qualities of our fighting men. They have demonstrated these qualities in adversity as well as in victory. As long as our flag flies over this Capitol, Americans will honor the soldiers, sailors, and marines who fought our first battles of this war against overwhelming odds the heroes, living and dead, of Wake and Bataan and Guadalcanal, of the Java Sea and Midway and the North Atlantic convoys. Their unconquerable spirit will live forever.\n\r\nBy far the largest and most important developments in the whole worldwide strategic picture of 1942 were the events of the long fronts in Russia: first, the implacable defense of Stalingrad; and, second, the offensives by the Russian armies at various points that started in the latter part of November and which still roll on with great force and effectiveness.\n\r\nThe other major events of the year were: the series of Japanese advances in the Philippines, the East Indies, Malaya, and Burma; the stopping of that Japanese advance in the mid-Pacific, the South Pacific, and the Indian Oceans; the successful defense of the Near East by the British counterattack through Egypt and Libya; the American-British occupation of North Africa. Of continuing importance in the year 1942 were the unending and bitterly contested battles of the convoy routes, and the gradual passing of air superiority from the Axis to the United Nations.\n\r\nThe Axis powers knew that they must win the war in 1942—or eventually lose everything. I do not need to tell you that our enemies did not win the war in 1942.\n\r\nIn the Pacific area, our most important victory in 1942 was the air and naval battle off Midway Island. That action is historically important because it secured for our use communication lines stretching thousands of miles in every direction. In placing this emphasis on the Battle of Midway, I am not unmindful of other successful actions in the Pacific, in the air and on land and afloat, especially those on the Coral Sea and New Guinea and in the Solomon Islands. But these actions were essentially defensive. They were part of the delaying strategy that character',
      'ized this phase of the war.\n\r\nDuring this period we inflicted steady losses upon the enemy—great losses of Japanese planes and naval vessels, transports and cargo ships. As early as one year ago, we set as a primary task in the war of the Pacific a day-by-day and week-by-week and month-by-month destruction of more Japanese war materials than Japanese industry could replace. Most certainly, that task has been and is being performed by our fighting ships and planes. And a large part of this task has been accomplished by the gallant crews of our American submarines who strike on the other side of the Pacific at Japanese ships—right up at the very mouth of the harbor of Yokohama.\n\r\nWe know that as each day goes by, Japanese strength in ships and planes is going down and down, and American strength in ships and planes is going up and up. And so I sometimes feel that the eventual outcome can now be put on a mathematical basis. That will become evident to the Japanese people themselves when we strike at their own home islands, and bomb them constantly from the air.\n\r\nAnd in the attacks against Japan, we shall be joined with the heroic people of China—that great people whose ideals of peace are so closely akin to our own. Even today we are flying as much lend-lease material into China as ever traversed the Burma Road, flying it over mountains 17,000 feet high, flying blind through sleet and snow. We shall overcome all the formidable obstacles, and get the battle equipment into China to shatter the power of our common enemy. From this war, China will realize the security, the prosperity and the dignity, which Japan has sought so ruthlessly to destroy.\n\r\nThe period of our defensive attrition in the Pacific is drawing to a close. Now our aim is to force the Japanese to fight. Last year, we stopped them. This year, we intend to advance.\n\r\nTurning now to the European theater of war, during this past year it was clear that our first task was to lessen the concentrated pressure on the Russian front by compelling Germany to divert part of her manpower and equipment to another theater of war. After months of secret planning and preparation in the utmost detail, an enormous amphibious expedition was embarked for French North Africa from the United States and the United Kingdom in literally hundreds of ships. It reached its objectives with very small losses, and has already produced an important effect upon the whole situation of the war. It has opened to attack what Mr. Churchill well described as "the underbelly of the Axis," and it has removed the always dangerous threat of an Axis attack throu',
      'gh West Africa against the South Atlantic Ocean and the continent of South America itself.\n\r\nThe well-timed and splendidly executed offensive from Egypt by the British 8th Army was a part of the same major strategy of the United Nations.\n\r\nGreat rains and appalling mud and very limited communications have delayed the final battles of Tunisia. The Axis is reinforcing its strong positions. But I am confident that though the fighting will be tough, when the final Allied assault is made, the last vestige of Axis power will be driven from the whole of the south shores of the Mediterranean.\n\r\nAny review of the year 1942 must emphasize the magnitude and the diversity of the military activities in which this nation has become engaged. As I speak to you, approximately one and a half million of our soldiers, sailors, marines, and fliers are in service outside of our continental limits, all through the world. Our merchant seamen, in addition, are carrying supplies to them and to our allies over every sea lane.\n\r\nFew Americans realize the amazing growth of our air strength, though I am sure our enemy does. Day in and day out our forces are bombing the enemy and meeting him in combat on many different fronts in every part of the world. And for those who question the quality of our aircraft and the ability of our fliers, I point to the fact that, in Africa, we are shooting down two enemy planes to every one we lose, and in the Pacific and the Southwest Pacific we are shooting them down four to one.\n\r\nWe pay great tribute—the tribute of the United States of America—to the fighting men of Russia and China and Britain and the various members of the British Commonwealth—the millions of men who through the years of this war have fought our common enemies, and have denied to them the world conquest which they sought.\n\r\nWe pay tribute to the soldiers and fliers and seamen of others of the United Nations whose countries have been overrun by Axis hordes.\n\r\nAs a result of the Allied occupation of North Africa, powerful units of the French Army and Navy are going into action. They are in action with the United Nations forces. We welcome them as allies and as friends. They join with those Frenchmen who, since the dark days of June, 1940, have been fighting valiantly for the liberation of their stricken country.\n\r\nWe pay tribute to the fighting leaders of our allies, to Winston Churchill, to Joseph Stalin, and to the Generalissimo Chiang Kai-shek. Yes, there is a very great unanimity between the leaders of the United Nations. This unity is effective in planning and carrying out the major strategy of this ',
      'war and in building up and in maintaining the lines of supplies.\n\r\nI cannot prophesy. I cannot tell you when or where the United Nations are going to strike next in Europe. But we are going to strike—and strike hard. I cannot tell you whether we are going to hit them in Norway, or through the Low Countries, or in France, or through Sardinia or Sicily, or through the Balkans, or through Poland—or at several points simultaneously. But I can tell you that no matter where and when we strike by land, we and the British and the Russians will hit them from the air heavily and relentlessly. Day in and day out we shall heap tons upon tons of high explosives on their war factories and utilities and seaports.\n\r\nHitler and Mussolini will understand now the enormity of their miscalculations—that the Nazis would always have the advantage of superior air power as they did when they bombed Warsaw, and Rotterdam, and London and Coventry. That superiority has gone forever.\n\r\nYes, the Nazis and the Fascists have asked for it—and they are going to get it.\n\r\nOur forward progress in this war has depended upon our progress on the production front.\n\r\nThere has been criticism of the management and conduct of our war production. Much of this self-criticism has had a healthy effect. It has spurred us on. It has reflected a normal American impatience to get on with the job. We are the kind of people who are never quite satisfied with anything short of miracles.\n\r\nBut there has been some criticism based on guesswork and even on malicious falsification of fact. Such criticism creates doubts and creates fears, and weakens our total effort.\n\r\nI do not wish to suggest that we should be completely satisfied with our production progress today, or next month, or ever. But I can report to you with genuine pride on what has been accomplished in 1942.\n\r\nA year ago we set certain production goals for 1942 and for 1943. Some people, including some experts, thought that we had pulled some big figures out of a hat just to frighten the Axis. But we had confidence in the ability of our people to establish new records. And that confidence has been justified.\n\r\nOf course, we realized that some production objectives would have to be changed—some of them adjusted upward, and others downward; some items would be taken out of the program altogether, and others added. This was inevitable as we gained battle experience, and as technological improvements were made.\n\r\nOur 1942 airplane production and tank production fell short, numerically—stress the word numerically of the goals set a year ago. Nevertheless, we have plenty of reaso',
      'n to be proud of our record for 1942. We produced 48,000 military planes—more than the airplane production of Germany, Italy, and Japan put together. Last month, in December, we produced 5,500 military planes and the rate is rapidly rising. Furthermore, we must remember that as each month passes by, the averages of our types weigh more, take more man-hours to make, and have more striking power.\n\r\nIn tank production, we revised our schedule—and for good and sufficient reasons. As a result of hard experience in battle, we have diverted a portion of our tank-producing capacity to a stepped-up production of new, deadly field weapons, especially self-propelled artillery.\n\r\nHere are some other production figures:\n\r\nIn 1942, we produced 56,000 combat vehicles, such as tanks and self-propelled artillery.\n\r\nIn 1942, we produced 670,000 machine guns, six times greater than our production in 1941 and three times greater than our total production during the year and a half of our participation in the first World War.\n\r\nWe produced 21,000 anti-tank guns, six times greater than our 1941 production.\n\r\nWe produced ten and a quarter billion rounds of small-arms ammunition, five times greater than our 1941 production and three times greater than our total production in the first World War.\n\r\nWe produced 181 million rounds of artillery ammunition, 12 times greater than our 1941 production and 10 times greater than our total production in the first World War.\n\r\nI think the arsenal of democracy is making good.\n\r\nThese facts and figures that I have given will give no great aid and comfort to the enemy. On the contrary, I can imagine that they will give him considerable discomfort. I suspect that Hitler and Tojo will find it difficult to explain to the German and Japanese people just why it is that "decadent, inefficient democracy" can produce such phenomenal quantities of weapons and munitions—and fighting men.\n\r\nWe have given the lie to certain misconceptions, which is an extremely polite word, especially the one which holds that the various blocs or groups within a free country cannot forego their political and economic differences in time of crisis and work together toward a common goal.\n\r\nWhile we have been achieving this miracle of production, during the past year our armed forces have grown from a little over 2,000,000 to 7,000,000. In other words, we have withdrawn from the labor force and the farms some 5,000,000 of our younger workers. And in spite of this, our farmers have contributed their share to the common effort by producing the greatest quantity of food ever made available during a si',
      'ngle year in all our history.\n\r\nI wonder is there any person among us so simple as to believe that all this could have been done without creating some dislocations in our normal national life, some inconveniences, and even some hardships?\n\r\nWho can have hoped to have done this without burdensome government regulations which are a nuisance to everyone—including those who have the thankless task of administering them?\n\r\nWe all know that there have been mistakes—mistakes due to the inevitable process of trial and error inherent in doing big things for the first time. We all know that there have been too many complicated forms and questionnaires. I know about that. I have had to fill some of them out myself.\n\r\nBut we are determined to see to it that our supplies of food and other essential civilian goods are distributed on a fair and just basis—to rich and poor, management and labor, farmer and city dweller alike. We are determined to keep the cost of living at a stable level. All this has required much information. These forms and questionnaires represent an honest and sincere attempt by honest and sincere officials to obtain this information.\n\r\nWe have learned by the mistakes that we have made.\n\r\nOur experience will enable us during the coming year to improve the necessary mechanisms of wartime economic controls, and to simplify administrative procedures. But we do not intend to leave things so lax that loopholes will be left for cheaters, for chiselers, or for the manipulators of the black market.\n\r\nOf course, there have been disturbances and inconveniences—and even hardships. And there will be many, many more before we finally win. Yes, 1943 will not be an easy year for us on the home front. We shall feel in many ways in our daily lives the sharp pinch of total war.\n\r\nFortunately, there are only a few Americans who place appetite above patriotism. The overwhelming majority realize that the food we send abroad is for essential military purposes, for our own and Allied fighting forces, and for necessary help in areas that we occupy.\n\r\nWe Americans intend to do this great job together. In our common labors we must build and fortify the very foundation of national unity—confidence in one another.\n\r\nIt is often amusing, and it is sometimes politically profitable, to picture the City of Washington as a madhouse, with the Congress and the administration disrupted with confusion and indecision and general incompetence.\n\r\nHowever, what matters most in war is results. And the one pertinent fact is that after only a few years of preparation and only one year of warfare, we are able to enga',
      'ge, spiritually as well as physically, in the total waging of a total war.\n\r\nWashington may be a madhouse—but only in the sense that it is the capital city of a nation which is fighting mad. And I think that Berlin and Rome and Tokyo, which had such contempt for the obsolete methods of democracy, would now gladly use all they could get of that same brand of madness.\n\r\nAnd we must not forget that our achievements in production have been relatively no greater than those of the Russians and the British and the Chinese who have developed their own war industries under the incredible difficulties of battle conditions. They have had to continue work through bombings and blackouts. And they have never quit.\n\r\nWe Americans are in good, brave company in this war, and we are playing our own, honorable part in the vast common effort.\n\r\nAs spokesmen for the United States government, you and I take off our hats to those responsible for our American production—to the owners, managers, and supervisors, to the draftsmen and the engineers, and to the workers—men and women—in factories and arsenals and shipyards and mines and mills and forests—and railroads and on highways.\n\r\nWe take off our hats to the farmers who have faced an unprecedented task of feeding not only a great nation but a great part of the world.\n\r\nWe take off our hats to all the loyal, anonymous, untiring men and women who have worked in private employment and in government and who have endured rationing and other stringencies with good humor and good will.\n\r\nYes, we take off our hats to all Americans who have contributed so magnificently to our common cause.\n\r\nI have sought to emphasize a sense of proportion in this review of the events of the war and the needs of the war.\n\r\nWe should never forget the things we are fighting for. But, at this critical period of the war, we should confine ourselves to the larger objectives and not get bogged down in argument over methods and details.\n\r\nWe, and all the United Nations, want a decent peace and a durable peace. In the years between the end of the first World War and the beginning of the second World War, we were not living under a decent or a durable peace.\n\r\nI have reason to know that our boys at the front are concerned with two broad aims beyond the winning of the war; and their thinking and their opinion coincide with what most Americans here back home are mulling over. They know, and we know, that it would be inconceivable—it would, indeed, be sacrilegious—if this nation and the world did not attain some real, lasting good out of all these efforts and sufferings and bloodshed and ',
      'death.\n\r\nThe men in our armed forces want a lasting peace, and, equally, they want permanent employment for themselves, their families, and their neighbors when they are mustered out at the end of the war.\n\r\nTwo years ago I spoke in my annual message of four freedoms. The blessings of two of them —freedom of speech and freedom of religion—are an essential part of the very life of this nation; and we hope that these blessings will be granted to all men everywhere.\n\r\n"The people at home, and the people at the front, are wondering a little about the third freedom—freedom from want. To them it means that when they are mustered out, when war production is converted to the economy of peace, they will have the right to expect full employment—full employment for themselves and for all able-bodied men and women in America who want to work.\n\r\nThey expect the opportunity to work, to run their farms, their stores, to earn decent wages. They are eager to face the risks inherent in our system of free enterprise.\n\r\nThey do not want a postwar America which suffers from undernourishment or slums—or the dole. They want no get-rich-quick era of bogus "prosperity" which will end for them in selling apples on a street corner, as happened after the bursting of the boom in 1929.\n\r\nWhen you talk with our young men and our young women, you will find they want to work for themselves and for their families; they consider that they have the right to work; and they know that after the last war their fathers did not gain that right.\n\r\nWhen you talk with our young men and women, you will find that with the opportunity for employment they want assurance against the evils of all major economic hazards—assurance that will extend from the cradle to the grave. And this great government can and must provide this assurance.\n\r\nI have been told that this is no time to speak of a better America after the war. I am told it is a grave error on my part.\n\r\nI dissent.\n\r\nAnd if the security of the individual citizen, or the family, should become a subject of national debate, the country knows where I stand.\n\r\nI say this now to this 78th Congress, because it is wholly possible that freedom from want—the right of employment, the right of assurance against life\'s hazards—will loom very large as a task of America during the coming two years.\n\r\nI trust it will not be regarded as an issue—but rather as a task for all of us to study sympathetically, to work out with a constant regard for the attainment of the objective, with fairness to all and with injustice to none.\n\r\nIn this war of survival we must keep before our minds not only',
      ' the evil things we fight against but the good things we are fighting for. We fight to retain a great past—and we fight to gain a greater future.\n\r\nLet us remember, too, that economic safety for the America of the future is threatened unless a greater economic stability comes to the rest of the world. We cannot make America an island in either a military or an economic sense. Hitlerism, like any other form of crime or disease, can grow from the evil seeds of economic as well as military feudalism.\n\r\nVictory in this war is the first and greatest goal before us. Victory in the peace is the next. That means striving toward the enlargement of the security of man here and throughout the world—and, finally, striving for the fourth freedom—freedom from fear.\n\r\nIt is of little account for any of us to talk of essential human needs, of attaining security, if we run the risk of another World War in 10 or 20 or 50 years. That is just plain common sense. Wars grow in size, in death and destruction, and in the inevitability of engulfing all nations, in inverse ratio to the shrinking size of the world as a result of the conquest of the air. I shudder to think of what will happen to humanity, including ourselves, if this war ends in an inconclusive peace, and another war breaks out when the babies of today have grown to fighting age.\n\r\nEvery normal American prays that neither he nor his sons nor his grandsons will be compelled to go through this horror again.\n\r\nUndoubtedly a few Americans, even now, think that this nation can end this war comfortably and then climb back into an American hole and pull the hole in after them.\n\r\nBut we have learned that we can never dig a hole so deep that it would be safe against predatory animals. We have also learned that if we do not pull the fangs of the predatory animals of this world, they will multiply and grow in strength—and they will be at our throats again once more in a short generation.\n\r\nMost Americans realize more clearly than ever before that modern war equipment in the hands of aggressor nations can bring danger overnight to our own national existence or to that of any other nation—or island—or continent.\n\r\nIt is clear to us that if Germany and Italy and Japan—or any one of them—remain armed at the end of this war, or are permitted to rearm, they will again, and inevitably, embark upon an ambitious career of world conquest. They must be disarmed and kept disarmed, and they must abandon the philosophy, and the teaching of that philosophy, which has brought so much suffering to the world.\n\r\nAfter the first World War we tried to achieve a formula f',
      'or permanent peace, based on a magnificent idealism. We failed. But, by our failure, we have learned that we cannot maintain peace at this stage of human development by good intentions alone.\n\r\nToday the United Nations are the mightiest military coalition in all history. They represent an overwhelming majority of the population of the world. Bound together in solemn agreement that they themselves will not commit acts of aggression or conquest against any of their neighbors, the United Nations can and must remain united for the maintenance of peace by preventing any attempt to rearm in Germany, in Japan, in Italy, or in any other nation which seeks to violate the Tenth Commandment—"Thou shalt not covet."\n\r\nThere are cynics, there are skeptics who say it cannot be done. The American people and all the freedom-loving peoples of this earth are now demanding that it must be done. And the will of these people shall prevail.\n\r\nThe very philosophy of the Axis powers is based on a profound contempt for the human race. If, in the formation of our future policy, we were guided by the same cynical contempt, then we should be surrendering to the philosophy of our enemies, and our victory would turn to defeat.\n\r\nThe issue of this war is the basic issue between those who believe in mankind and those who do not—the ancient issue between those who put their faith in the people and those who put their faith in dictators and tyrants. There have always been those who did not believe in the people, who attempted to block their forward movement across history, to force them back to servility and suffering and silence.\n\r\nThe people have now gathered their strength. They are moving forward in their might and power—and no force, no combination of forces, no trickery, deceit, or violence, can stop them now. They see before them the hope of the world—a decent, secure, peaceful life for men everywhere.\n\r\nI do not prophesy when this war will end.\n\r\nBut I do believe that this year of 1943 will give to the United Nations a very substantial advance along the roads that lead to Berlin and Rome and Tokyo.\n\r\nI tell you it is within the realm of possibility that this 78th Congress may have the historic privilege of helping greatly to save the world from future fear.\n\r\nTherefore, let us all have confidence, let us redouble our efforts.\n\r\nA tremendous, costly, long-enduring task in peace as well as in war is still ahead of us.\n\r\nBut, as we face that continuing task, we may know that the state of this nation is good—the heart of this nation is sound—the spirit of this nation is strong—the faith of this nation is eter'],
     ['Mr. Vice President, Mr. Speaker, Members of the 88th Congress: I congratulate you all--not merely on your electoral victory but on your selected role in history. For you and I are privileged to serve the great Republic in what could be the most decisive decade in its long history. The choices we make, for good or ill, may well shape the state of the Union for generations yet to come. Little more than 100 weeks ago I assumed the office of President of the United States. In seeking the help of the Congress and our countrymen, I pledged no easy answers. I pledged--and asked--only toil and dedication. These the Congress and the people have given in good measure. And today, having witnessed in recent months a heightened respect for our national purpose and power--having seen the courageous calm of a united people in a perilous hour-and having observed a steady improvement in the opportunities and well-being of our citizens--I can report to you that the state of this old but youthful Union, in the 175th year of its life, is good. In the world beyond our borders, steady progress has been made in building a world of order. The people of West Berlin remain both free and secure. A settlement, though still precarious, has been reached in Laos. The spearpoint of aggression has been blunted in Viet-Nam. The end of agony may be in sight in the Congo. The doctrine of troika is dead. And, while danger continues, a deadly threat has been removed in Cuba. At home, the recession is behind us. Well over a million more men and women are working today than were working 2 years ago. The average factory workweek is once again more than 40 hours; our industries are turning out more goods than ever before; and more than half of the manufacturing capacity that lay silent and wasted 100 weeks ago is humming with activity. In short, both at home and abroad, there may now be a temptation to relax. For the road has been long, the burden heavy, and the pace consistently urgent. But we cannot be satisfied to rest here. This is the side of the hill, not the top. The mere absence of war is not peace. The mere absence of recession is not growth. We have made a beginning--but we have only begun. Now the time has come to make the most of our gains--to translate the renewal of our national strength into the achievement of our national purpose. America has enjoyed 22 months of uninterrupted economic recovery. But recovery is not enough. If we are to prevail in the long run, we must expand the long-run strength of our economy. We must move along the path to a higher rate of growth and full employment. For this would mean tens of billions of dollars more each year in production, profits, wages, and public revenues. It would mean an end to the persistent slack which has kept our unemployment at or above 5 percent for 61 out of the past 62 months--and an end to the growing pressures for such restrictive measures as the 35-hour week, which alone could increase hourly labor costs by as much as 14 percent, start a new wage-price spiral of inflation, and undercut our efforts to compete with other nations. To achieve these greater gains, one s',
      "tep, above all, is essential--the enactment this year of a substantial reduction and revision in Federal income taxes. For it is increasingly clear--to those in Government, business, and labor who are responsible for our economy's success--that our obsolete tax system exerts too heavy a drag on private purchasing power, profits, and employment. Designed to check inflation in earlier years, it now checks growth instead. It discourages extra effort and risk. It distorts the use of resources. It invites recurrent recessions, depresses our Federal revenues, and causes chronic budget deficits. Now, when the inflationary pressures of the war and the post-war years no longer threaten, and the dollar commands new respect-now, when no military crisis strains our resources--now is the time to act. We cannot afford to be timid or slow. For this is the most urgent task confronting the Congress in 1963. In an early message, I shall propose a permanent reduction in tax rates which will lower liabilities by $13.5 billion. Of this, $11 billion results from reducing individual tax rates, which now range between 20 and 91 percent, to a more sensible range of 14 to 65 percent, with a split in the present first bracket. Two and one-half billion dollars results from reducing corporate tax rates, from 52 percent--which gives the Government today a majority interest in profits-to the permanent pre-Korean level of 47 percent. This is in addition to the more than $2 billion cut in corporate tax liabilities resulting from last year's investment credit and depreciation reform. To achieve this reduction within the limits of a manageable budgetary deficit, I urge: first, that these cuts be phased over 3 calendar years, beginning in 1963 with a cut of some $6 billion at annual rates; second, that these reductions be coupled with selected structural changes, beginning in 1964, which will broaden the tax base, end unfair or unnecessary preferences, remove or lighten certain hardships, and in the net offset some $3.5 billion of the revenue loss; and third, that budgetary receipts at the outset be increased by $1.5 billion a year, without any change in tax liabilities, by gradually shifting the tax payments of large corporations to a . more current time schedule. This combined program, by increasing the amount of our national income, will in time result in still higher Federal revenues. It is a fiscally responsible program--the surest and the soundest way of achieving in time a balanced budget in a balanced full employment economy. This net reduction in tax liabilities of $10 billion will increase the purchasing power of American families and business enterprises in every tax bracket, with greatest increase going to our low-income consumers. It will, in addition, encourage the initiative and risk-taking on which our free system depends--induce more investment, production, and capacity use--help provide the 2 million new jobs we need every year--and reinforce the American principle of additional reward for additional effort. I do not say that a measure for tax reduction and reform is the only way to achieve these goals. --No doub",
      "t a massive increase in Federal spending could also create jobs and growth-but, in today's setting, private consumers, employers, and investors should be given a full opportunity first. --No doubt a temporary tax cut could provide a spur to our economy--but a long run problem compels a long-run solution. --No doubt a reduction in either individual or corporation taxes alone would be of great help--but corporations need customers and job seekers need jobs. --No doubt tax reduction without reform would sound simpler and more attractive to many--but our growth is also hampered by a host of tax inequities and special preferences which have distorted the flow of investment. --And, finally, there are no doubt some who would prefer to put off a tax cut in the hope that ultimately an end to the cold war would make possible an equivalent cut in expenditures-but that end is not in view and to wait for it would be costly and self-defeating. In submitting a tax program which will, of course, temporarily increase the deficit but can ultimately end it--and in recognition of the need to control expenditures--I will shortly submit a fiscal 1964 administrative budget which, while allowing for needed rises in defense, space, and fixed interest charges, holds total expenditures for all other purposes below this year's level. This requires the reduction or postponement of many desirable programs, the absorption of a large part of last year's Federal pay raise through personnel and other economies, the termination of certain installations and projects, and the substitution in several programs of private for public credit. But I am convinced that the enactment this year of tax reduction and tax reform overshadows all other domestic problems in this Congress. For we cannot for long lead the cause of peace and freedom, if we ever cease to set the pace here at home. Tax reduction alone, however, is not enough to strengthen our society, to provide opportunities for the four million Americans who are born every year, to improve the lives of 32 million Americans who live on the outskirts of poverty. The quality of American life must keep pace with the quantity of American goods. This country cannot afford to be materially rich and spiritually poor. Therefore, by holding down the budgetary cost of existing programs to keep within the limitations I have set, it is both possible and imperative to adopt other new measures that we cannot afford to postpone. These measures are based on a series of fundamental premises, grouped under four related headings: First, we need to strengthen our Nation by investing in our youth: --The future of any country which is dependent upon the will and wisdom of its citizens is damaged, and irreparably damaged, whenever any of its children is not educated to the full extent of his talent, from grade school through graduate school. Today, an estimated 4 out of every 10 students in the 5th grade will not even finish high school--and that is a waste we cannot afford. --In addition, there is no reason why one million young Americans, out of school and out of work, should all remain unwanted and often",
      ' untrained on our city streets when their energies can be put to good use. --Finally, the overseas success of our Peace Corps volunteers, most of them young men and women carrying skills and ideas to needy people, suggests the merit of a similar corps serving our own community needs: in mental hospitals, on Indian reservations, in centers for the aged or for young delinquents, in schools for the illiterate or the handicapped. As the idealism of our youth has served world peace, so can it serve the domestic tranquility. Second, we need to strengthen our Nation by safeguarding its health: --Our working men and women, instead of being forced to beg for help from public charity once they are old and ill, should start contributing now to their own retirement health program through the Social Security System. --Moreover, all our miracles of medical research will count for little if we cannot reverse the growing nationwide shortage of doctors, dentists, and nurses, and the widespread shortages of nursing homes and modern urban hospital facilities. Merely to keep the present ratio of doctors and dentists from declining any further, we must over the next 10 years increase the capacity of our medical schools by 50 percent and our dental schools by 100 percent. --Finally, and of deep concern, I believe that the abandonment of the mentally ill and the mentally retarded to the grim mercy of custodial institutions too often inflicts on them and on their families a needless cruelty which this Nation should not endure. The incidence of mental retardation in this country is three times as high as that of Sweden, for example--and that figure can and must be reduced. Third, we need to strengthen our Nation by protecting the basic rights of its citizens: --The right to competent counsel must be assured to every man accused of crime in Federal court, regardless of his means. --And the most precious and powerful right in the world, the right to vote in a free American election, must not be denied to any citizen on grounds of his race or color. I wish that all qualified Americans permitted to vote were willing to vote, but surely in this centennial year of Emancipation all those who are willing to vote should always be permitted. Fourth, we need to strengthen our Nation by making the best and the most economical use of its resources and facilities: --Our economic health depends on healthy transportation arteries; and I believe the way to a more modern, economical choice of national transportation service is through increased competition and decreased regulation. Local mass transit, faring even worse, is as essential a community service as hospitals and highways. Nearly three-fourths of our citizens live in urban areas, which occupy only 2 percent of our land-and if local transit is to survive and relieve the congestion of these cities, it needs Federal stimulation and assistance. --Next, this Government is in the storage and stockpile business to the melancholy tune of more than $ 16 billion. We must continue to support farm income, but we should not pile more farm surpluses on top of the $7.5 billion we already own. ',
      "We must maintain a stockpile of strategic materials, but the $8.5 billion we have acquired--for reasons both good and bad--is much more than we need; and we should be empowered to dispose of the excess in ways which will not cause market disruption. --Finally, our already overcrowded national parks and recreation areas will have twice as many visitors 10 years from now as they do today. If we do not plan today for the future growth of these and other great natural assets--not only parks and forests but wildlife and wilderness preserves, and water projects of all kinds--our children and their children will be poorer in every sense of the word. These are not domestic concerns alone. For upon our achievement of greater vitality and strength here at home hang our fate and future in the world: our ability to sustain and supply the security of free men and nations, our ability to command their respect for our leadership, our ability to expand our trade without threat to our balance of payments, and our ability to adjust to the changing demands of cold war competition and challenge. We shall be judged more by what we do at home than by what we preach abroad. Nothing we could do to help the developing countries would help them half as much as a booming U.S. economy. And nothing our opponents could do to encourage their own ambitions would encourage them half as much as a chronic lagging U.S. economy. These domestic tasks do not divert energy from our security--they provide the very foundation for freedom's survival and success, Turning to the world outside, it was only a few years ago--in Southeast Asia, Africa, Eastern Europe, Latin America, even outer space--that communism sought to convey the image of a unified, confident, and expanding empire, closing in on a sluggish America and a free world in disarray. But few people would hold to that picture today. In these past months we have reaffirmed the scientific and military superiority of freedom. We have doubled our efforts in space, to assure us of being first in the future. We have undertaken the most far-reaching defense improvements in the peacetime history of this country. And we have maintained the frontiers of freedom from Viet-Nam to West Berlin. But complacency or self-congratulation can imperil our security as much as the weapons of tyranny. A moment of pause is not a promise of peace. Dangerous problems remain from Cuba to the South China Sea. The world's prognosis prescribes, in short, not a year's vacation for us, but a year of obligation and opportunity. Four special avenues of opportunity stand out: the Atlantic Alliance, the developing nations, the new Sino-Soviet difficulties, and the search for worldwide peace. First, how fares the grand alliance? Free Europe is entering into a new phase of its long and brilliant history. The era of colonial expansion has passed; the era of national rivalries is fading; and a new era of interdependence and unity is taking shape. Defying the old prophecies of Marx, consenting to what no conqueror could ever compel, the free nations of Europe are moving toward a unity of purpose and power and policy in ",
      "every sphere of activity. For 17 years this movement has had our consistent support, both political and economic. Far from resenting the new Europe, we regard her as a welcome partner, not a rival. For the road to world peace and freedom is still long, and there are burdens which only full partners can share--in supporting the common defense, in expanding world trade, in aligning our balance of payments, in aiding the emergent nations, in concerting political and economic policies, and in welcoming to our common effort other industrialized nations, notably Japan, whose remarkable economic and political development of the 1950's permits it now to play on the world scene a major constructive role. No doubt differences of opinion will continue to get more attention than agreements on action, as Europe moves from independence to more formal interdependence. But these are honest differences among honorable associates--more real and frequent, in fact, among our Western European allies than between them and the United States. For the unity of freedom has never relied on uniformity of opinion. But the basic agreement of this alliance on fundamental issues continues. The first task of the alliance remains the common defense. Last month Prime Minister Macmillan and I laid plans for a new stage in our long cooperative effort, one which aims to assist in the wider task of framing a common nuclear defense for the whole alliance. The Nassau agreement recognizes that the security of the West is indivisible, and so must be our defense. But it also recognizes that this is an alliance of proud and sovereign nations, and works best when we do not forget it. It recognizes further that the nuclear defense of the West is not a matter for the present nuclear powers alone--that France will be such a power in the future--and that ways must be found without increasing the hazards of nuclear diffusion, to increase the role of our other partners in planning, manning, and directing a truly multilateral nuclear force within an increasingly intimate NATO alliance. Finally, the Nassau agreement recognizes that nuclear defense is not enough, that the agreed NATO levels of conventional strength must be met, and that the alliance cannot afford to be in a position of having to answer every threat with nuclear weapons or nothing. We remain too near the Nassau decisions, and too far from their full realization, to know their place in history. But I believe that, for the first time, the door is open for the nuclear defense of the alliance to become a source of confidence, instead of a cause of contention. The next most pressing concern of the alliance is our common economic goals of trade and growth. This Nation continues to be concerned about its balance-of-payments deficit, which, despite its decline, remains a stubborn and troublesome problem. We believe, moreover, that closer economic ties among all free nations are essential to prosperity and peace. And neither we nor the members of the European Common Market are so affluent that we can long afford to shelter high cost farms or factories from the winds of foreign competition, or",
      " to restrict the channels of trade with other nations of the free world. If the Common Market should move toward protectionism and restrictionism, it would undermine its, own basic principles. This Government means to use the authority conferred on it last year by the Congress to encourage trade expansion on both sides of the Atlantic and around the world. Second, what of the developing and nonaligned nations? They were shocked by the Soviets' sudden and secret attempt to transform Cuba into a nuclear striking base-and by Communist China's arrogant invasion of India. They have been reassured by our prompt assistance to India, by our support through the United Nations of the Congo's unification, by our patient search for disarmament, and by the improvement in our treatment of citizens and visitors whose skins do not happen to be white. And as the older colonialism recedes, and the neocolonialism of the Communist powers stands out more starkly than ever, they realize more clearly that the issue in the world struggle is not communism versus capitalism, but coercion versus free choice. They are beginning to realize that the longing for independence is the same the world over, whether it is the independence of West Berlin or Viet-Nam. They are beginning to realize that such independence runs athwart all Communist ambitions but is in keeping with our own--and that our approach to their diverse needs is resilient and resourceful, while the Communists are still relying on ancient doctrines and dogmas. Nevertheless it is hard for any nation to focus on an external or subversive threat to its independence when its energies are drained in daily combat with the forces of poverty and despair. It makes little sense for us to assail, in speeches and resolutions, the horrors of communism, to spend $50 billion a year to prevent its military advance-and then to begrudge spending, largely on American products, less than one-tenth of that amount to help other nations strengthen their independence and cure the social chaos in which communism always has thrived. I am proud--and I think most Americans are proud--of a mutual defense and assistance program, evolved with bipartisan support in three administrations, which has, with all its recognized problems, contributed to the fact that not a single one of the nearly fifty U.N. members to gain independence since the Second World War has succumbed to Communist control. I am proud of a program that has helped to arm and feed and clothe millions of people who live on the front lines of freedom. I am especially proud that this country has put forward for the 60's a vast cooperative effort to achieve economic growth and social progress throughout the Americas-the Alliance for Progress. I do not underestimate the difficulties that we face in this mutual effort among our close neighbors, but the free states of this hemisphere, working in close collaboration, have begun to make this alliance a living reality. Today it is feeding one out of every four school age children in Latin America an extra food ration from our farm surplus. It has distributed 1.5 million school books and ",
      'is building 17,000 classrooms. It has helped resettle tens of thousands of farm families on land they can call their own. It is stimulating our good neighbors to more self-help and self-reform--fiscal, social, institutional, and land reforms. It is bringing new housing and hope, new health and dignity, to millions who were forgotten. The men and women of this hemisphere know that the alliance cannot Succeed if it is only another name for United States handouts--that it can succeed only as the Latin American nations themselves devote their best effort to fulfilling its goals. This story is the same in Africa, in the Middle East, and in Asia. Wherever nations are willing to help themselves, we stand ready to help them build new bulwarks of freedom. We are not purchasing votes for the cold war; we have gone to the aid of imperiled nations, neutrals and allies alike. What we do ask--and all that we ask--is that our help be used to best advantage, and that their own efforts not be diverted by needless quarrels with other independent nations. Despite all its past achievements, the continued progress of the mutual assistance program requires a persistent discontent with present performance. We have been reorganizing this program to make it a more effective, efficient instrument--and that process will continue this year. But free world development will still be an uphill struggle. Government aid can only supplement the role of private investment, trade expansion, commodity stabilization, and, above all, internal self-improvement. The processes of growth are gradual--bearing fruit in a decade, not a day. Our successes will be neither quick nor dramatic. But if these programs were ever to be ended, our failures in a dozen countries would be sudden and certain. Neither money nor technical assistance, however, can be our only weapon against poverty. In the end, the crucial effort is one of purpose, requiring the fuel of finance but also a torch of idealism. And nothing carries the spirit of this American idealism more effectively to the far corners of the earth than the American Peace Corps. A year ago, less than 900 Peace Corps volunteers were on the job. A year from now they will number more than 9,000-men and women, aged 18 to 79, willing to give 2 years of their lives to helping people in other lands. There are, in fact, nearly a million Americans serving their country and the cause of freedom in overseas posts, a record no other people can match. Surely those of us who stay at home should be glad to help indirectly; by supporting our aid programs; .by opening our doors to foreign visitors and diplomats and students; and by proving, day by day, by deed as well as word, that we are a just and generous people. Third, what comfort can we take from the increasing strains and tensions within the Communist bloc? Here hope must be tempered with caution. For the Soviet-Chinese disagreement is over means, not ends. A dispute over how best to bury the free world is no grounds for Western rejoicing. Nevertheless, while a strain is not a fracture, it is clear that the forces of diversity are at work inside the Comm',
      'unist camp, despite all the iron disciplines of regimentation and all the iron dogmatism\'s of ideology. Marx is proven wrong once again: for it is the closed Communist societies, not the free and open societies which carry within themselves the seeds of internal disintegration. The disarray of the Communist empire has been heightened by two other formidable forces. One is the historical force of nationalism-and the yearning of all men to be free. The other is the gross inefficiency of their economies. For a closed society is not open to ideas of progress--and a police state finds that it cannot command the grain to grow. New nations asked to choose between two competing systems need only compare conditions in East and West Germany, Eastern and Western Europe, North and South Viet-Nam. They need only compare the disillusionment of Communist Cuba with the promise of the Alliance for Progress. And all the world knows that no successful system builds a wall to keep its people in and freedom out--and the wall of shame dividing Berlin is a symbol of Communist failure. Finally, what can we do to move from the present pause toward enduring peace? Again I would counsel caution. I foresee no spectacular reversal in Communist methods or goals. But if all these trends and developments can persuade the Soviet Union to walk the path of peace, then let her know that all free nations will journey with her. But until that choice is made, and until the world can develop a reliable system of international security, the free peoples have no choice but to keep their arms nearby. This country, therefore, continues to require the best defense in the world--a defense which is suited to the sixties. This means, unfortunately, a rising defense budget-for there is no substitute for adequate defense, and no "bargain basement" way of achieving it. It means the expenditure of more than $15 billion this year on nuclear weapons systems alone, a sum which is about equal to the combined defense budgets of our European Allies. But it also means improved air and missile defenses, improved civil defense, a strengthened anti-guerrilla capacity and, of prime importance, more powerful and flexible nonnuclear forces. For threats of massive retaliation may not deter piecemeal aggression-and a line of destroyers in a quarantine, or a division of well-equipped men on a border, may be more useful to our real security than the multiplication of awesome weapons beyond all rational need. But our commitment to national safety is not a commitment to expand our military establishment indefinitely. We do not dismiss disarmament as merely an idle dream. For we believe that, in the end, it is the only way to assure the security of all without impairing the interests of any. Nor do we mistake honorable negotiation for appeasement. While we shall never weary in the defense of freedom, neither shall we ever abandon the pursuit of peace. In this quest, the United Nations requires our full and continued support. Its value in serving the cause of peace has been shown anew in its role in the West New Guinea settlement, in its use as a forum for the Cuban ',
      'crisis, and in its task of unification in the Congo. Today the United Nations is primarily the protector of the small and the weak, and a safety valve for the strong. Tomorrow it can form the framework for a world of law--a world in which no nation dictates the destiny of another, and in which the vast resources now devoted to destructive means will serve constructive ends. In short, let our adversaries choose. If they choose peaceful competition, they shall have it. If they come to realize that their ambitions cannot succeed--if they see their "wars of liberation" and subversion will ultimately fail--if they recognize that there is more security in accepting inspection than in permitting new nations to master the black arts of nuclear war--and if they are willing to turn their energies, as we are, to the great unfinished tasks of our own peoples--then, surely, the areas of agreement can be very wide indeed: a clear understanding about Berlin, stability in Southeast Asia, an end to nuclear testing, new checks on surprise or accidental attack, and, ultimately, general and complete disarmament. For we seek not the worldwide victory of one nation or system but a worldwide victory of man. The modern globe is too small, its weapons are too destructive, and its disorders are too contagious to permit any other kind of victory. To achieve this end, the United States will continue to spend a greater portion of its national production than any other people in the free world. For 15 years no other free nation has demanded so much of itself. Through hot wars and cold, through recession and prosperity, through the ages of the atom and outer space, the American people have never faltered and their faith has never flagged. If at times our actions seem to make life difficult for others, it is only because history has made life difficult for us all. But difficult days need not be dark. I think these are proud and memorable days in the cause of peace and freedom. We are proud, for example, of Major Rudolf Anderson who gave his life over the island of Cuba. We salute Specialist James Allen Johnson who died on the border of South Korea. We pay honor to Sergeant Gerald Pendell who was killed in Viet-Nam. They are among the many who in this century, far from home, have died for our country. Our task now, and the task of all Americans is to live up to their commitment. My friends: I close on a note of hope. We are not lulled by the momentary calm of the sea or the somewhat clearer skies above. We know the turbulence that lies below, and the storms that are beyond the horizon this year. But now the winds of change appear to be blowing more strongly than ever, in the world of communism as well as our own. For 175 years we have sailed with those winds at our back, and with the tides of human freedom in our favor. We steer our ship with hope, as Thomas Jefferson said, "leaving Fear astern." Today we still welcome those winds of change--and we have every reason to believe that our tide is running strong. With thanks to Almighty God for seeing us through a perilous passage, we ask His help anew in guiding the "Good Ship Uni'],
     ["Mr. President, Mr. Speaker, members of the 96th Congress, fellow citizens: This last few months has not been an easy time for any of us. As we meet tonight, it has never been more clear that the state of our Union depends on the state of the world. And tonight, as throughout our own generation, freedom and peace in the world depend on the state of our Union. The 1980s have been born in turmoil, strife, and change. This is a time of challenge to our interests and our values and it's a time that tests our wisdom and our skills. At this time in Iran, 50 Americans are still held captive, innocent victims of terrorism and anarchy. Also at this moment, massive Soviet troops are attempting to subjugate the fiercely independent and deeply religious people of Afghanistan. These two acts—one of international terrorism and one of military aggression—present a serious challenge to the United States of America and indeed to all the nations of the world. Together, we will meet these threats to peace. I'm determined that the United States will remain the strongest of all nations, but our power will never be used to initiate a threat to the security of any nation or to the rights of any human being. We seek to be and to remain secure—a nation at peace in a stable world. But to be secure we must face the world as it is. Three basic developments have helped to shape our challenges: the steady growth and increased projection of Soviet military power beyond its own borders; the overwhelming dependence of the Western democracies on oil supplies from the Middle East; and the press of social and religious and economic and political change in the many nations of the developing world, exemplified by the revolution in Iran. Each of these factors is important in its own right. Each interacts with the others. All must be faced together, squarely and courageously. We will face these challenges, and we will meet them with the best that is in us. And we will not fail. In response to the abhorrent act in Ir",
      "an, our nation has never been aroused and unified so greatly in peacetime. Our position is clear. The United States will not yield to blackmail. We continue to pursue these specific goals: first, to protect the present and long-range interests of the United States; secondly, to preserve the lives of the American hostages and to secure, as quickly as possible, their safe release, if possible, to avoid bloodshed which might further endanger the lives of our fellow citizens; to enlist the help of other nations in condemning this act of violence, which is shocking and violates the moral and the legal standards of a civilized world; and also to convince and to persuade the Iranian leaders that the real danger to their nation lies in the north, in the Soviet Union and from the Soviet troops now in Afghanistan, and that the unwarranted Iranian quarrel with the United States hampers their response to this far greater danger to them. If the American hostages are harmed, a severe price will be paid. We will never rest until every one of the American hostages are released. But now we face a broader and more fundamental challenge in this region because of the recent military action of the Soviet Union. Now, as during the last three and a half decades, the relationship between our country, the United States of America, and the Soviet Union is the most critical factor in determining whether the world will live at peace or be engulfed in global conflict. Since the end of the Second World War, America has led other nations in meeting the challenge of mounting Soviet power. This has not been a simple or a static relationship. Between us there has been cooperation, there has been competition, and at times there has been confrontation. In the 1940s we took the lead in creating the Atlantic Alliance in response to the Soviet Union's suppression and then consolidation of its East European empire and the resulting threat of the Warsaw Pact to Western Europe. In the 1950s we helped to contain furth",
      "er Soviet challenges in Korea and in the Middle East, and we rearmed to assure the continuation of that containment. In the 1960s we met the Soviet challenges in Berlin, and we faced the Cuban missile crisis. And we sought to engage the Soviet Union in the important task of moving beyond the cold war and away from confrontation. And in the 1970s three American Presidents negotiated with the Soviet leaders in attempts to halt the growth of the nuclear arms race. We sought to establish rules of behavior that would reduce the risks of conflict, and we searched for areas of cooperation that could make our relations reciprocal and productive, not only for the sake of our two nations but for the security and peace of the entire world. In all these actions, we have maintained two commitments: to be ready to meet any challenge by Soviet military power, and to develop ways to resolve disputes and to keep the peace. Preventing nuclear war is the foremost responsibility of the two superpowers. That's why we've negotiated the strategic arms limitation treaties—SALT I and SALT II. Especially now, in a time of great tension, observing the mutual constraints imposed by the terms of these treaties will be in the best interest of both countries and will help to preserve world peace. I will consult very closely with the Congress on this matter as we strive to control nuclear weapons. That effort to control nuclear weapons will not be abandoned. We superpowers also have the responsibility to exercise restraint in the use of our great military force. The integrity and the independence of weaker nations must not be threatened. They must know that in our presence they are secure. But now the Soviet Union has taken a radical and an aggressive new step. It's using its great military power against a relatively defenseless nation. The implications of the Soviet invasion of Afghanistan could pose the most serious threat to the peace since the Second World War. The vast majority of nations on Earth have",
      " condemned this latest Soviet attempt to extend its colonial domination of others and have demanded the immediate withdrawal of Soviet troops. The Moslem world is especially and justifiably outraged by this aggression against an Islamic people. No action of a world power has ever been so quickly and so overwhelmingly condemned. But verbal condemnation is not enough. The Soviet Union must pay a concrete price for their aggression. While this invasion continues, we and the other nations of the world cannot conduct business as usual with the Soviet Union. That's why the United States has imposed stiff economic penalties on the Soviet Union. I will not issue any permits for Soviet ships to fish in the coastal waters of the United States. I've cut Soviet access to high-technology equipment and to agricultural products. I've limited other commerce with the Soviet Union, and I've asked our allies and friends to join with us in restraining their own trade with the Soviets and not to replace our own embargoed items. And I have notified the Olympic Committee that with Soviet invading forces in Afghanistan, neither the American people nor I will support sending an Olympic team to Moscow. The Soviet Union is going to have to answer some basic questions: Will it help promote a more stable international environment in which its own legitimate, peaceful concerns can be pursued? Or will it continue to expand its military power far beyond its genuine security needs, and use that power for colonial conquest? The Soviet Union must realize that its decision to use military force in Afghanistan will be costly to every political and economic relationship it values. The region which is now threatened by Soviet troops in Afghanistan is of great strategic importance: It contains more than two thirds of the world's exportable oil. The Soviet effort to dominate Afghanistan has brought Soviet military forces to within 300 miles of the Indian Ocean and close to the Straits of Hormuz, a waterway through w",
      "hich most of the world's oil must flow. The Soviet Union is now attempting to consolidate a strategic position, therefore, that poses a grave threat to the free movement of Middle East oil. This situation demands careful thought, steady nerves, and resolute action, not only for this year but for many years to come. It demands collective efforts to meet this new threat to security in the Persian Gulf and in Southwest Asia. It demands the participation of all those who rely on oil from the Middle East and who are concerned with global peace and stability. And it demands consultation and close cooperation with countries in the area which might be threatened. Meeting this challenge will take national will, diplomatic and political wisdom, economic sacrifice, and, of course, military capability. We must call on the best that is in us to preserve the security of this crucial region. Let our position be absolutely clear: An attempt by any outside force to gain control of the Persian Gulf region will be regarded as an assault on the vital interests of the United States of America, and such an assault will be repelled by any means necessary, including military force. During the past three\xa0years, you have joined with me to improve our own security and the prospects for peace, not only in the vital oil-producing area of the Persian Gulf region but around the world. We've increased annually our real commitment for defense, and we will sustain this increase of effort throughout the Five-Year Defense Program. It's imperative that Congress approve this strong defense budget for 1981, encompassing a 5 percent real growth in authorizations, without any reduction. We are also improving our capability to deploy U.S. military forces rapidly to distant areas. We've helped to strengthen NATO and our other alliances, and recently we and other NATO members have decided to develop and to deploy modernized, intermediate-range nuclear forces to meet an unwarranted and increased threat from the nuclear ",
      "weapons of the Soviet Union. We are working with our allies to prevent conflict in the Middle East. The peace treaty between Egypt and Israel is a notable achievement which represents a strategic asset for America and which also enhances prospects for regional and world peace. We are now engaged in further negotiations to provide full autonomy for the people of the West Bank and Gaza, to resolve the Palestinian issue in all its aspects, and to preserve the peace and security of Israel. Let no one doubt our commitment to the security of Israel. In a few days we will observe an historic event when Israel makes another major withdrawal from the Sinai and when Ambassadors will be exchanged between Israel and Egypt. We've also expanded our own sphere of friendship. Our deep commitment to human rights and to meeting human needs has improved our relationship with much of the Third World. Our decision to normalize relations with the People's Republic of China will help to preserve peace and stability in Asia and in the Western Pacific. We've increased and strengthened our naval presence in the Indian Ocean, and we are now making arrangements for key naval and air facilities to be used by our forces in the region of northeast Africa and the Persian Gulf. We've reconfirmed our 1959 agreement to help Pakistan preserve its independence and its integrity. The United States will take action consistent with our own laws to assist Pakistan in resisting any outside aggression. And I'm asking the Congress specifically to reaffirm this agreement. I'm also working, along with the leaders of other nations, to provide additional military and economic aid for Pakistan. That request will come to you in just a few days. Finally, we are prepared to work with other countries in the region to share a cooperative security framework that respects differing values and political beliefs, yet which enhances the independence, security, and prosperity of all. All these efforts combined emphasize our dedication",
      " to defend and preserve the vital interests of the region and of the nation which we represent and those of our allies—in Europe and the Pacific, and also in the parts of the world which have such great strategic importance to us, stretching especially through the Middle East and Southwest Asia. With your help, I will pursue these efforts with vigor and with determination. You and I will act as necessary to protect and to preserve our nation's security. The men and women of America's armed forces are on duty tonight in many parts of the world. I'm proud of the job they are doing, and I know you share that pride. I believe that our volunteer forces are adequate for current defense needs, and I hope that it will not become necessary to impose a draft. However, we must be prepared for that possibility. For this reason, I have determined that the Selective Service System must now be revitalized. I will send legislation and budget proposals to the Congress next month so that we can begin registration and then meet future mobilization needs rapidly if they arise. We also need clear and quick passage of a new charter to define the legal authority and accountability of our intelligence agencies. We will guarantee that abuses do not recur, but we must tighten our controls on sensitive intelligence information, and we need to remove unwarranted restraints on America's ability to collect intelligence. The decade ahead will be a time of rapid change, as nations everywhere seek to deal with new problems and age-old tensions. But America need have no fear. We can thrive in a world of change if we remain true to our values and actively engaged in promoting world peace. We will continue to work as we have for peace in the Middle East and southern Africa. We will continue to build our ties with developing nations, respecting and helping to strengthen their national independence which they have struggled so hard to achieve. And we will continue to support the growth of democracy and the protec",
      "tion of human rights. In repressive regimes, popular frustrations often have no outlet except through violence. But when peoples and their governments can approach their problems together through open, democratic methods, the basis for stability and peace is far more solid and far more enduring. That is why our support for human rights in other countries is in our own national interest as well as part of our own national character. Peace—a peace that preserves freedom—remains America's first goal. In the coming years, as a mighty nation we will continue to pursue peace. But to be strong abroad we must be strong at home. And in order to be strong, we must continue to face up to the difficult issues that confront us as a nation today. The crises in Iran and Afghanistan have dramatized a very important lesson: Our excessive dependence on foreign oil is a clear and present danger to our nation's security. The need has never been more urgent. At long last, we must have a clear, comprehensive energy policy for the United States. As you well know, I have been working with the Congress in a concentrated and persistent way over the past three years to meet this need. We have made progress together. But Congress must act promptly now to complete final action on this vital energy legislation. Our nation will then have a major conservation effort, important initiatives to develop solar power, realistic pricing based on the true value of oil, strong incentives for the production of coal and other fossil fuels in the United States, and our nation's most massive peacetime investment in the development of synthetic fuels. The American people are making progress in energy conservation. Last year we reduced overall petroleum consumption by 8 percent and gasoline consumption by 5 percent below what it was the year before. Now we must do more. After consultation with the Governors, we will set gasoline conservation goals for each of the 50 states, and I will make them mandatory if these goals ar",
      'e not met. I\'ve established an import ceiling for 1980 of 8.2 million barrels a day—well below the level of foreign oil purchases in 1977. I expect our imports to be much lower than this, but the ceiling will be enforced by an oil import fee if necessary. I\'m prepared to lower these imports still further if the other oil-consuming countries will join us in a fair and mutual reduction. If we have a serious shortage, I will not hesitate to impose mandatory gasoline rationing immediately. The single biggest factor in the inflation rate last year, the increase in the inflation rate last year, was from one cause: the skyrocketing prices of OPEC oil. We must take whatever actions are necessary to reduce our dependence on foreign oil—and at the same time reduce inflation. As individuals and as families, few of us can produce energy by ourselves. But all of us can conserve energy—every one of us, every day of our lives. Tonight I call on you—in fact, all the people of America—to help our nation. Conserve energy. Eliminate waste. Make 1980 indeed a year of energy conservation. Of course, we must take other actions to strengthen our nation\'s economy. First, we will continue to reduce the deficit and then to balance the federal budget. Second, as we continue to work with business to hold down prices, we\'ll build also on the historic national accord with organized labor to restrain pay increases in a fair fight against inflation. Third, we will continue our successful efforts to cut paperwork and to dismantle unnecessary government regulation. Fourth, we will continue our progress in providing jobs for America, concentrating on a major new program to provide training and work for our young people, especially minority youth. It has been said that "a mind is a terrible thing to waste." We will give our young people new hope for jobs and a better life in the 1980s. And fifth, we must use the decade of the 1980s to attack the basic structural weaknesses and problems in our economy through me',
      'asures to increase productivity, savings, and investment. With these energy and economic policies, we will make America even stronger at home in this decade—just as our foreign and defense policies will make us stronger and safer throughout the world. We will never abandon our struggle for a just and a decent society here at home. That\'s the heart of America—and it\'s the source of our ability to inspire other people to defend their own rights abroad. Our material resources, great as they are, are limited. Our problems are too complex for simple slogans or for quick solutions. We cannot solve them without effort and sacrifice. Walter Lippmann once reminded us, "You took the good things for granted. Now you must earn them again. For every right that you cherish, you have a duty which you must fulfill. For every good which you wish to preserve, you will have to sacrifice your comfort and your ease. There is nothing for nothing any longer." Our challenges are formidable. But there\'s a new spirit of unity and resolve in our country. We move into the 1980s with confidence and hope and a bright vision of the America we want: an America strong and free, an America at peace, an America with equal rights for all citizens— and for women, guaranteed in the United States Constitution—an America with jobs and good health and good education for every citizen, an America with a clean and bountiful life in our cities and on our farms, an America that helps to feed the world, an America secure in filling its own energy needs, an America of justice, tolerance, and compassion. For this vision to come true, we must sacrifice, but this national commitment will be an exciting enterprise that will unify our people. Together as one people, let us work to build our strength at home, and together as one indivisible union, let us seek peace and security throughout the world. Together let us make of this time of challenge and danger a decade of national resolve and of brave achievement. Thank you very mu'],
     ["Mr. Speaker, Mr. Vice President, Members of Congress, my fellow Americans: Tonight marks the eighth year that I’ve come here to report on the State of the Union.\xa0And for this final one, I’m going to try to make it a little shorter. (Applause.) I know some of you are antsy to get back to Iowa. (Laughter.) I've been there. I'll be shaking hands afterwards if you want some tips. (Laughter.) And I understand that because it’s an election season, expectations for what we will achieve this year are low.\xa0But, Mr. Speaker, I appreciate the constructive approach that you and the other leaderstook at the end of last year to\xa0pass a budget\xa0and\xa0make\xa0tax cuts permanent\xa0for working families. So I hope we can work together this year on some bipartisan priorities like\xa0criminal justice reform\xa0-- (applause) -- and helping people who are battling prescription drug abuse and heroin abuse. (Applause.) So, who knows, we might surprise the cynics again. But tonight, I want to go easy on the traditional list of proposals for the year ahead. Don’t worry, I’ve got plenty, from\xa0helping students learn to write computer code\xa0to\xa0personalizing medical treatments for patients. And I will keep pushing for progress on the work that I believe still needs to be done.\xa0Fixing a broken immigration system. (Applause.)\xa0Protecting our kids from gun violence. (Applause.)Equal pay for equal work. (Applause.)\xa0Paid leave. (Applause.)Raising the minimum wage. (Applause.) All these things still matter to hardworking families. They’re still the right thing to do. And I won't let up until they get done. But for my final address to this chamber, I don’t want to just talk about next year. I want to focus on the next five years, the next 10 years, and beyond.\xa0I want to focus on our future. We live in a time of extraordinary change -- change that’s reshaping the way we live, the way we work, our planet, our place in the world. It’s change that promises amazing medical breakthroughs, but also economic disruptions that strain working families. It\xa0promises education for girls in the most remote villages,\xa0but also connects terrorists plotting an ocean away. It’s change that can broaden opportunity, or widen inequality. And whether we like it or not, the pace of this change will only accelerate. \xa0America has been through big changes before -- wars and depression, the\xa0influx of new immigrants, workers fighting for a fair deal, movements to expand civil rights.\xa0Each time, there have been those who told us to fear the future; who claimed we could slam the brakes on change; who promised to restore past glory if we just got some group or idea that was threatening America under control. And each time, we overcame those fears.\xa0We did not, in the words of Lincoln, adhere to the “dogmas of the quiet past.” Instead we thought anew, and acted anew.\xa0We made change work for us, always extending America’s promise outward, to the next frontier, to more people. And because we did -- because we saw opportunity where others saw only peril -- we emerged stronger and better than before. What was true then can be true now. Our unique strengths as a nation -- our optimism and work ethic, our spirit of discovery, our diversity, our commitment to rule of law -- these things give us everything we need to ensure prosperity and security for generations to come. In fact, it’s that spirit that made the progress of these past seven years possible.\xa0 It’s how we recovered from the worst economic crisis in generations.\xa0 It’s how we reformed",
      " our health care system, and reinvented our energy sector; how we delivered more care and benefits to our troops and veterans, and\xa0how we secured the freedom in every state to marry the person we love. But such progress is not inevitable. It’s the result of choices we make together. And we face such choices right now. Will we respond to the changes of our time with fear, turning inward as a nation, turning against each other as a people? Or will we face the future with confidence in who we are, in what we stand for, in the incredible things that we can do together? So let’s talk about the future, and four big questions that I believe we as a country have to answer -- regardless of who the next President is, or who controls the next Congress. First, how do we give everyone a fair shot at opportunity and security in this new economy?\xa0(Applause.) Second, how do we make technology work for us, and not against us -- especially when it comes to solving urgent challenges like climate change?\xa0(Applause.) Third, how do we keep America safe and lead the world without becoming its policeman?\xa0(Applause.) And finally, how can we make our politics reflect what’s best in us, and not what’s worst? Let me start with the economy, and a basic fact:\xa0The United States of America, right now,\xa0has the strongest, most durable economy in the world.\xa0(Applause.)\xa0We’re in the middle of the longest streak of private sector job creation in history.\xa0(Applause.)\xa0More than 14 million new jobs, the strongest two years of job growth since the ‘90s, an unemployment rate cut in half.\xa0Our auto industry just had its best year ever. (Applause.) That's just part of a manufacturing surge that's created nearly 900,000 new jobs in the past six years.And we’ve done all this while cutting our deficits by almost three-quarters.\xa0(Applause.) Anyone claiming that America’s economy is in decline is peddling fiction. (Applause.) Now, what is true -- and the reason that a lot of Americans feel anxious\xa0-- is that the economy has been changing in profound ways, changes that started long\xa0before the\xa0Great Recession hit; changes that have not let up. Today, technology doesn’t just replace jobs on the assembly line, but any job where work can be automated.\xa0Companies in a global economy can locate anywhere, and they face tougher competition. As a result, workers have less leverage for a raise.\xa0Companies have less loyalty to their communities. And more and more wealth and income is concentrated at the very top. All these trends have squeezed workers, even when they have jobs; even when the economy is growing. It’s made it harder for a hardworking family to pull itself out of poverty, harder for young people to start their careers, tougher for workers to retire when they want to. And although none of these trends are unique to America, they do offend our uniquely American belief that everybody who works hard should get a fair shot. For the past seven years, our goal has been a growing economy that works also better for everybody. We’ve made progress. But we need to make more. And despite all the political arguments that we’ve had these past few years, there are actually some areas where Americans broadly agree. We agree that real opportunity requires every American to get the education and training they need to land a good-paying job.\xa0The bipartisan reform of No Child Left Behind was an important start, and together, we’ve increased early childhood education,\xa0lifted high school graduation rates to new highs,\xa0bo",
      "osted graduates in fields like engineering. In the coming years, we should build on that progress, by providing Pre-K for all and -- (applause) -- offering every student the hands-on computer science and math classes that make them job-ready on day one. We should recruit and support more great teachers for our kids. (Applause.) And we have to make college affordable for every American. (Applause.) No hardworking student should be\xa0stuck in the red.We’ve already reduced student loan payments to 10 percent of a borrower’s income. And that's good. But now, we’ve actually got to cut the cost of college.\xa0(Applause.)\xa0Providing two years of community college at no cost for every responsible student is one of the best ways to do that, and I’m going to keep fighting to get that started this year. (Applause.) It's the right thing to do. (Applause.) But a great education isn’t all we need in this new economy. We also need benefits and protections that provide a basic measure of security. It’s not too much of a stretch to say that some of the only people in America who are going to work the same job, in the same place, with a health and retirement package for 30 years are sitting in this chamber. (Laughter.) For everyone else, especially folks in their 40s and 50s, saving for retirement or bouncing back from job loss has gotten a lot tougher. Americans understand that at some point in their careers, in this new economy, they may have to retool and they may have to retrain. But they shouldn’t lose what they’ve already worked so hard to build in the process. That’s why Social Security and Medicare are more important than ever. We shouldn’t weaken them; we should strengthen them.(Applause.) And for Americans short of retirement, basic benefits should be just as mobile as everything else is today.\xa0That, by the way, is what the Affordable Care Act is all about.\xa0It’s about filling the gaps in employer-based care so that when you lose a job, or you go back to school, or you strike out and launch that new business, you’ll still have coverage. Nearly 18 million people have gained coverage so far. (Applause.) And in the process, health care inflation has slowed.And our businesses have created jobs every single month since it became law. Now, I’m guessing we won’t agree on health care anytime soon. (Applause.) A little applause right there. (Laughter.) Just a guess. But there should be other ways parties can work together to improve economic security. Say a hardworking American loses his job -- we shouldn’t just make sure that he can get unemployment insurance; we should make sure that program encourages him to retrain for a business that’s ready to hire him. If that new job doesn’t pay as much, there should be a system of wage insurance in place so that he can still pay his bills. And even if he’s going from job to job, he should still be able to save for retirement and take his savings with him. That’s the way we make the new economy work better for everybody. I also know Speaker Ryan has talked about his interest in tackling poverty. America is about giving everybody willing to work a chance, a hand up.\xa0And I’d welcome a serious discussion about strategies we can all support, like expanding tax cuts for low-income workers who don't have children.\xa0(Applause.) But there are some areas where we just have to be honest -- it has been difficult to find agreement over the last seven years. And a lot of them fall under the category of what role the government should play in mak",
      "ing sure the system’s not rigged in favor of the wealthiest and biggest corporations. (Applause.) And it's an honest disagreement, and the American people have a choice to make. I believe a thriving private sector is the lifeblood of our economy. I think there are outdated regulations that need to be changed. There is red tape that needs to be cut. (Applause.) There you go! Yes! (Applause.) But after years now of record corporate profits, working families won’t get more opportunity or bigger paychecks just by letting big banks or big oil or hedge funds make their own rules at everybody else’s expense. (Applause.) Middle-class families are not going to feel more secure because we allowed attacks on collective bargaining to go unanswered. Food Stamp recipients did not cause the financial crisis; recklessness on Wall Street did. (Applause.) Immigrants aren’t the principal reason wages haven’t gone up; those decisions are made in the boardrooms that all too often put quarterly earnings over long-term returns. It’s sure not the average family watching tonight that avoids paying taxes through offshore accounts. (Applause.) The point is, I believe that in this new economy, workers and start-ups and small businesses need more of a voice, not less. The rules should work for them. (Applause.) And I'm not alone in this. This year I plan to lift up the many businesses who’ve figured out that doing right by their workers or their customers or their communities ends up being good for their shareholders. (Applause.) And I want to spread those best practices across America. That's part of a brighter future. (Applause.) In fact, it turns out many of our best corporate citizens are also our most creative. And this brings me to the second big question we as a country have to answer: How do we reignite that spirit of innovation to meet our biggest challenges? Sixty years ago, when the Russians beat us into space, we didn’t deny Sputnik was up there. (Laughter.) We didn’t argue about the science, or shrink our research and development budget.\xa0We built a space program almost overnight. And 12 years later, we were walking on the moon.\xa0(Applause.) Now, that spirit of discovery is in our DNA. America is Thomas Edison and the Wright Brothers\xa0and George Washington Carver. America is Grace Hopper and Katherine Johnson and Sally Ride. America is every immigrant and entrepreneur from Boston to Austin to Silicon Valley, racing to shape a better world. (Applause.) That's who we are. And over the past seven years, we’ve nurtured that spirit.\xa0We’ve protected an open Internet, and taken bold new steps to get more students and low-income Americans online. (Applause.) We’ve launched next-generation manufacturing hubs, and online tools that give an entrepreneur everything he or she needs to start a business in a single day. But we can do so much more. Last year, Vice President Biden said that with a new moonshot,America can cure cancer.\xa0Last month, he worked with this Congress to give scientists at the National Institutes of Health the strongest resources that they’ve had in over a decade. (Applause.) So tonight, I’m announcing a new national effort to get it done. And because he’s gone to the mat for all of us on so many issues over the past 40 years,\xa0I’m putting Joe in charge of Mission Control. (Applause.)\xa0For the loved ones we’ve all lost, for the families that we can still save, let’s make America the country that cures cancer once and for all.(Applause.) Medical research is critic",
      "al. We need the same level of commitment when it comes to developing clean energy sources. (Applause.)Look, if anybody still wants to dispute the science around climate change, have at it. You will be pretty lonely, because you’ll be debating our military, most of America’s business leaders, the majority of the American people, almost the entire scientific community, and 200 nations around the world who agree it’s a problem and intend to solve it.\xa0(Applause.) But even if -- even if the planet wasn’t at stake, even if 2014 wasn’t the warmest year on record -- until 2015 turned out to be even hotter -- why would we want to pass up the chance for American businesses to produce and sell the energy of the future? (Applause.) Listen, seven years ago, we made the single biggest investment in clean energy in our history. Here are the results. In fields from Iowa to Texas, wind power is now cheaper than dirtier, conventional power. On rooftops from Arizona to New York, solar is saving Americans tens of millions of dollars a year on their energy bills, and employs more Americans than coal -- in jobs that pay better than average. We’re taking steps to give homeowners the freedom to generate and store their own energy -- something, by the way, that environmentalists and Tea Partiers have teamed up to support. And meanwhile, we’ve cut our imports of foreign oil by nearly 60 percent, and cut carbon pollution more than any other country on Earth. (Applause.) Gas under two bucks a gallon ain’t bad, either.\xa0(Applause.) Now we’ve got to accelerate the transition away from old, dirtier energy sources. Rather than subsidize the past, we should invest in the future -- especially in communities that rely on fossil fuels. We do them no favor when we don't show them where the trends are going. That’s why I’m going to push to change the way we manage our oil and coal resources, so that they better reflect the costs they impose on taxpayers and our planet. And that way, we put money back into those communities, and put tens of thousands of Americans to work building a 21st century transportation system. (Applause.) Now, none of this is going to happen overnight. And, yes, there are plenty of entrenched interests who want to protect the status quo. But the jobs we’ll create, the money we’ll save, the planet we’ll preserve -- that is the kind of future our kids and our grandkids deserve. And it's within our grasp. Climate change is just one of many issues where our security is linked to the rest of the world.\xa0And that’s why the third big question that we have to answer together is how to keep America safe and strong without either isolating ourselves or trying to nation-build everywhere there’s a problem. I told you earlier all the talk of America’s economic decline is political hot air. Well, so is all the rhetoric you hear about our enemies getting stronger and America getting weaker. Let me tell you something. The United States of America is the most powerful nation on Earth. Period. (Applause.) Period. It’s not even close. It's not even close. (Applause.) It's not even close. We spend more on our military than the next eight nations combined. Our troops are the finest fighting force in the history of the world. (Applause.) No nation attacks us directly, or our allies, because they know that’s the path to ruin.\xa0Surveys show our standing around the world is higher than when I was elected to this office, and when it comes to every important international issue, people of the ",
      "world do not look to Beijing or Moscow to lead -- they call us.\xa0(Applause.) I mean, it's useful to level the set here, because when we don't, we don't make good decisions. Now, as someone who begins every day with\xa0an intelligence briefing, I know this is a dangerous time. But that’s not primarily because of some looming superpower out there, and certainly not because of diminished American strength. In today’s world, we’re threatened less by evil empires and more by failing states. The Middle East is going through a transformation that will play out for a generation, rooted in conflicts that date back millennia. Economic headwinds are blowing in from a Chinese economy that is in significant transition. Even as their economy severely contracts, Russia is pouring resources in to prop up Ukraine and Syria -- client states that they saw slipping away from their orbit.\xa0And the international system we built after World War II is now struggling to keep pace with this new reality. It’s up to us, the United States of America, to help remake that system.\xa0And to do that well it means that we’ve got to set priorities. Priority number one is protecting the American people and going after terrorist networks. (Applause.) Both al Qaeda and now ISIL pose a direct threat to our people, because in today’s world, even a handful of terrorists who place no value on human life, including their own, can do a lot of damage. They use the Internet to poison the minds of individuals inside our country.\xa0Their actions undermine and destabilize our allies.\xa0We have to take them out./p> But as we focus on destroying ISIL, over-the-top claims that this is World War III just play into their hands. Masses of fighters on the back of pickup trucks, twisted souls plotting in apartments or garages -- they pose an enormous danger to civilians; they have to be stopped. But they do not threaten our national existence. (Applause.) That is the story ISIL wants to tell. That’s the kind of propaganda they use to recruit. We don’t need to build them up to show that we’re serious, and we sure don't need to push away vital allies in this fight by echoing the lie that ISIL is somehow representative of one of the world’s largest religions. (Applause.) We just need to call them what they are -- killers and fanatics who have to be rooted out, hunted down, and destroyed. (Applause.) And that’s exactly what we’re doing.\xa0For more than a year, America has led a coalition of more than 60 countries to cut off ISIL’s financing, disrupt their plots,\xa0stop the flow of terrorist fighters, and stamp out their vicious ideology. With nearly 10,000 air strikes, we’re taking out their leadership, their oil, their training camps, their weapons.\xa0We’re training, arming, and supporting forces who are steadily reclaiming territory in Iraq and Syria. If this Congress is serious about winning this war, and wants to send a message to our troops and the world, authorize the use of military force against ISIL. Take a vote. (Applause.) Take a vote. But the American people should know that with or without\xa0congressional action, ISIL will learn the same lessons as terrorists before them.\xa0If you doubt America’s commitment -- or mine -- to see that justice is done, just ask Osama bin Laden.\xa0(Applause.) Ask the leader of al Qaeda in Yemen, who was taken out last year, or the perpetrator of the Benghazi attacks, who sits in a prison cell. When you come after Americans, we go after you. (Applause.) And it may take time, but we have long",
      " memories, and our reach has no limits. (Applause.) Our foreign policy hast to be focused on the threat from ISIL and al Qaeda, but it can’t stop there. For even without ISIL, even without al Qaeda, instability will continue for decades in many parts of the world -- in the Middle East, in Afghanistan, parts of Pakistan, in parts of Central America, in Africa, and Asia. Some of these places may become safe havens for new terrorist networks. Others will just fall victim to ethnic conflict, or famine, feeding the next wave of refugees. The world will look to us to help solve these problems, and our answer needs to be more than tough talk or calls to carpet-bomb civilians. That may work as a TV sound bite, but it doesn’t pass muster on the world stage. We also can’t try to take over and rebuild every country that falls into crisis, even if it's done with the best of intentions. (Applause.) That’s not leadership; that’s a recipe for quagmire, spilling American blood and treasure that ultimately will weaken us. It’s the lesson of Vietnam; it's the lesson of Iraq -- and we should have learned it by now. (Applause.) Fortunately, there is a smarter approach, a patient and disciplined strategy that uses every element of our national power. It says America will always act, alone if necessary, to protect our people and our allies; but on issues of global concern, we will mobilize the world to work with us, and make sure other countries pull their own weight. That’s our approach to conflicts like Syria, where we’re partnering with local forces and leading international efforts to help that broken society pursue a lasting peace. That’s why we built a global coalition, with sanctions and principled diplomacy, to prevent a nuclear-armed Iran. And as we speak, Iran has rolled back its nuclear program, shipped out its uranium stockpile, and the world has avoided another war. (Applause.) That’s how we stopped the spread of Ebola in West Africa. (Applause.) Our military, our doctors, our development workers -- they were heroic; they set up the platform that then allowed other countries to join in behind us and stamp out that epidemic. Hundreds of thousands, maybe a couple million lives were saved. That’s how we forged a Trans-Pacific Partnership to open markets, and protect workers and the environment, and advance American leadership in Asia. It cuts 18,000 taxes on products made in America, which will then support more good jobs here in America. With TPP, China does not set the rules in that region; we do. You want to show our strength in this new century? Approve this agreement. Give us the tools to enforce it. It's the right thing to do. (Applause.) Let me give you another example. Fifty years of isolating Cuba had failed to promote democracy, and set us back in Latin America. That’s why we\xa0restored diplomatic relations\xa0-- (applause) -- opened the door to travel and commerce, positioned ourselves to improve the lives of the Cuban people. (Applause.) So if you want to consolidate our leadership and credibility in the hemisphere, recognize that the Cold War is over -- lift the embargo. (Applause.) The point is American leadership in the 21st century is not a choice between ignoring the rest of the world -- except when we kill terrorists -- or occupying and rebuilding whatever society is unraveling. Leadership means a wise application of military power, and rallying the world behind causes that are right. It means seeing our foreign assistance as a part of our national",
      " security, not something separate, not charity. When we lead nearly 200 nations to the most ambitious agreement in history to fight climate change, yes, that helps vulnerable countries, but it also protects our kids. When we help Ukraine defend its democracy, or Colombia resolve a decades-long war, that strengthens the international order we depend on. When we help African countries feed their people and care for the sick -- (applause) -- it's the right thing to do, and it prevents the next pandemic from reaching our shores. Right now, we’re on track to end the scourge of HIV/AIDS. That's within our grasp. (Applause.) And we have the chance to accomplish the same thing with malaria -- something I’ll be pushing this Congress to fund this year. (Applause.) That's American strength. That's American leadership. And that kind of leadership depends on the power of our example.\xa0That’s why I will keep working to shut down the prison at Guantanamo. (Applause.) It is expensive, it is unnecessary, and it only serves as a recruitment brochure for our enemies. (Applause.) There’s a better way. (Applause.) And that’s why we need to reject any politics -- any politics -- that targets people because of race or religion. (Applause.) Let me just say this. This is not a matter of political correctness. This is a matter of understanding just what it is that makes us strong. The world respects us not just for our arsenal; it respects us for our diversity, and our openness, and the way we respect every faith. His Holiness, Pope Francis, told this body from the very spot that I'm standing on tonight that\xa0“to imitate the hatred and violence of tyrants and murderers is the best way to take their place.”\xa0When politicians insult Muslims, whether abroad or our fellow citizens, when a mosque is vandalized, or a kid is called names, that doesn’t make us safer. That’s not telling it like it is. It’s just wrong. (Applause.)\xa0It diminishes us in the eyes of the world. It makes it harder to achieve our goals.\xa0It betrays who we are as a country. (Applause.) “We the People.” Our Constitution begins with those three simple words, words we’ve come to recognize mean all the people, not just some; words that insist we rise and fall together, and that's how we might perfect our Union. And that brings me to the fourth, and maybe the most important thing that I want to say tonight. The future we want -- all of us want -- opportunity and security for our families, a rising standard of living, a sustainable, peaceful planet for our kids -- all that is within our reach. But it will only happen if we work together. It will only happen if we can have rational, constructive debates. It will only happen if we fix our politics. A better politics doesn’t mean we have to agree on everything. This is a big country -- different regions, different attitudes, different interests. That’s one of our strengths, too. Our Founders distributed power between states and branches of government, and expected us to argue, just as they did, fiercely, over the size and shape of government, over commerce and foreign relations, over the meaning of liberty and the imperatives of security. But democracy does require basic bonds of trust between its citizens. It doesn’t work if we think the people who disagree with us are all motivated by malice. It doesn’t work if we think that our political opponents are unpatriotic or trying to weaken America. Democracy grinds to a halt without a willingness to compromise, or when even b",
      "asic facts are contested, or when we listen only to those who agree with us. Our public life withers when only the most extreme voices get all the attention. And most of all, democracy breaks down when the average person feels their voice doesn’t matter; that the system is rigged in favor of the rich or the powerful or some special interest. Too many Americans feel that way right now. It’s one of the few regrets of my presidency -- that the rancor and suspicion between the parties has gotten worse instead of better. I have no doubt a president with the gifts of Lincoln or Roosevelt might have better bridged the divide, and I guarantee I’ll keep trying to be better so long as I hold this office. But, my fellow Americans, this cannot be my task -- or any President’s -- alone. There are a whole lot of folks in this chamber, good people who would like to see more cooperation, would like to see a more elevated debate in Washington, but feel trapped by the imperatives of getting elected, by the noise coming out of your base. I know; you’ve told me. It's the worst-kept secret in Washington. And a lot of you aren't enjoying being trapped in that kind of rancor. But that means if we want a better politics -- and I'm addressing the American people now --\xa0if we want a better politics, it’s not enough just to change a congressman or change a senator or even change a President. We have to change the system to reflect our better selves.\xa0I think we've got to end the practice of drawing our congressional districts so that politicians can pick their voters, and not the other way around. (Applause.) Let a bipartisan group do it. (Applause.) We have to reduce the influence of money in our politics, so that a handful of families or hidden interests can’t bankroll our elections. (Applause.) And if our existing approach to campaign finance reform can’t pass muster in the courts, we need to work together to find a real solution -- because it's a problem. And most of you don't like raising money. I know; I've done it. (Applause.)\xa0We’ve got to make it easier to vote, not harder. (Applause.) We need to modernize it for the way we live now.\xa0(Applause.) This is America: We want to make it easier for people to\xa0participate. And over the course of this year,\xa0I intend to travel the country to push for reforms that do just that. But I can’t do these things on my own. (Applause.) Changes in our political process -- in not just who gets elected, but how they get elected -- that will only happen when the American people demand it. It depends on you. That’s what’s meant by a government of, by, and for the people. What I’m suggesting is hard. It’s a lot easier to be cynical; to accept that change is not possible, and politics is hopeless, and the problem is all the folks who are elected don't care, and to believe that our voices and actions don’t matter. But if we give up now, then we forsake a better future. Those with money and power will gain greater control over the decisions that could send a young soldier to war, or allow another economic disaster, or roll back the equal rights and voting rights that generations of Americans have fought, even died, to secure. And then, as frustration grows, there will be voices urging us to fall back into our respective tribes, to scapegoat fellow citizens who don’t look like us, or pray like us, or vote like we do, or share the same background. We can’t afford to go down that path. It won’t deliver the economy we want. It will not produce the sec",
      "urity we want. But most of all, it contradicts everything that makes us the envy of the world. So, my fellow Americans, whatever you may believe, whether you prefer one party or no party, whether you supported my agenda or fought as hard as you could against it -- our collective futures depends on your willingness to uphold your duties as a citizen. To vote. To speak out. To stand up for others, especially the weak, especially the vulnerable, knowing that each of us is only here because somebody, somewhere, stood up for us. (Applause.) We need every American to stay active in our public life -- and not just during election time -- so that our public life reflects the goodness and the decency that I see in the American people every single day. It is not easy. Our brand of democracy is hard. But I can promise that a little over a year from now, when I no longer hold this office, I will be right there with you as a citizen, inspired by those voices of fairness and vision, of grit and good humor and kindness that helped America travel so far. Voices that help us see ourselves not, first and foremost, as black or white, or Asian or Latino, not as gay or straight, immigrant or native born, not as Democrat or Republican, but as Americans first, bound by a common creed. Voices Dr. King believed would have the final word --\xa0voices of unarmed truth and unconditional love. And they’re out there, those voices. They don’t get a lot of attention; they don't seek a lot of fanfare; but they’re busy doing the work this country needs doing. I see them everywhere I travel in this incredible country of ours. I see you, the American people. And in your daily acts of citizenship, I see our future unfolding. I\xa0see it in the worker on the assembly line who clocked extra shifts to keep his company open, and the boss who pays him higher wages instead of laying him off. I see it in the Dreamer who stays up late to finish her science project, and the teacher who comes in early because he knows she might someday cure a disease. I see it in the American who served his time, and made mistakes as a child but now is dreaming of starting over -- and I see it in the business owner who gives him that second chance. The protester determined to prove that justice matters -- and the young cop walking the beat, treating everybody with respect, doing the brave, quiet work of keeping us safe. (Applause.) I see it in the soldier who gives almost everything to save his brothers, the nurse who tends to him till he can run a marathon, the community that lines up to cheer him on. It’s the son who finds the courage to come out as who he is, and the father whose love for that son overrides everything he’s been taught.\xa0(Applause.) I see it in the elderly woman who will wait in line to cast her vote as long as she has to; the new citizen who casts his vote for the first time; the volunteers at the polls who believe every vote should count -- because each of them in different ways know how much that precious right is worth. That's the America I know. That’s the country we love.\xa0Clear-eyed. Big-hearted. Undaunted by challenge.\xa0Optimistic that unarmed truth and unconditional love will have the final word. (Applause.) That’s what makes me so hopeful about our future.\xa0I believe in change because I believe in you, the American people. And that’s why I stand here confident as I have ever been\xa0that the State of our Union is strong. (Applause.) Thank you, God bless you. God bless the United States of America"],
     ['Mr. Speaker, Mr. President, and distinguished Members of the House and Senate: When we first met here seven years ago-many of us for the first time—it was with the hope of beginning something new for America. We meet here tonight in this historic Chamber to continue that work. If anyone expects just a proud recitation of the accomplishments of my administration, I say let\'s leave that to history; we\'re not finished yet. So, my message to you tonight is put on your work shoes; we\'re still on the job. History records the power of the ideas that brought us here those 7 years ago-ideas like the individual\'s right to reach as far and as high as his or her talents will permit; the free market as an engine of economic progress. And as an ancient Chinese philosopher, Lao-tzu, said: "Govern a great nation as you would cook a small fish; do not overdo it." Well, these ideas were part of a larger notion, a vision, if you will, of America herself—an America not only rich in opportunity for the individual but an America, too, of strong families and vibrant neighborhoods; an America whose divergent but harmonizing communities were a reflection of a deeper community of values: the value of work, of family, of religion, and of the love of freedom that God places in each of us and whose defense He has entrusted in a special way to this nation. All of this was made possible by an idea I spoke of when Mr. Gorbachev was here-the belief that the most exciting revolution ever known to humankind began with three simple words: "We the People," the revolutionary notion that the people grant government its rights, and not the other way around. And there\'s one lesson that has come home powerfully to me, which I would offer to you now. Just as those who created this Republic pledged to each other their lives, their fortunes, and their sacred honor, so, too, America\'s leaders today must pledge to each other that we will keep foremost in our hearts and minds not what is best for ourselves or for our party but what is best for America. In the spirit of Jefferson, let us affirm that in this Chamber tonight there are no Republicans, no Democrats—just Americans. Yes, we will have our differences, but let us always remember what unites us far outweighs whatever divides us. Those who sent us here to serve them—the millions of Americans watching and listening tonight-expect this of us. Let\'s prove to them and to ourselves that democracy works even in an election year. We\'ve done this before. And as we have worked together to bring down spending, tax rates, and inflation, employment has climbed to record heights; America has created more jobs and better, higher paying jobs; family income has risen for 4 straight years, and America\'s poor climbed out of poverty at the fastest rate in more than 10 years. Our record is',
      ' not just the longest peacetime expansion in history but an economic and social revolution of hope based on work, incentives, growth, and opportunity; a revolution of compassion that led to private sector initiatives and a 77-percent increase in charitable giving; a revolution that at a critical moment in world history reclaimed and restored the American dream. In international relations, too, there\'s only one description for what, together, we have achieved: a complete turnabout, a revolution. Seven years ago, America was weak, and freedom everywhere was under siege. Today America is strong, and democracy is everywhere on the move. From Central America to East Asia, ideas like free markets and democratic reforms and human rights are taking hold. We\'ve replaced "Blame America" with "Look up to America." We\'ve rebuilt our defenses. And of all our accomplishments, none can give us more satisfaction than knowing that our young people are again proud to wear our country\'s uniform. And in a few moments, I\'m going to talk about three developments—arms reduction, the Strategic Defense Initiative, and the global democratic revolution—that, when taken together, offer a chance none of us would have dared imagine 7 years ago, a chance to rid the world of the two great nightmares of the postwar era. I speak of the startling hope of giving our children a future free of both totalitarianism and nuclear terror. Tonight, then, we\'re strong, prosperous, at peace, and we are free. This is the state of our Union. And if we will work together this year, I believe we can give a future President and a future Congress the chance to make that prosperity, that peace, that freedom not just the state of our Union but the state of our world. Toward this end, we have four basic objectives tonight. First, steps we can take this year to keep our economy strong and growing, to give our children a future of low inflation and full employment. Second, let\'s check our progress in attacking social problems, where important gains have been made, but which still need critical attention. I mean schools that work, economic independence for the poor, restoring respect for family life and family values. Our third objective tonight is global: continuing the exciting economic and democratic revolutions we\'ve seen around the world. Fourth and finally, our nation has remained at peace for nearly a decade and a half, as we move toward our goals of world prosperity and world freedom. We must protect that peace and deter war by making sure the next President inherits what you and I have a moral obligation to give that President: a national security that is unassailable and a national defense that takes full advantage of new technology and is fully funded. This is a full agenda. It\'s meant to be. You see, my thinking on the next',
      " year is quite simple: Let's make this the best of 8. And that means it's all out—right to the finish line. I don't buy the idea that this is the last year of anything, because we're not talking here tonight about registering temporary gains but ways of making permanent our successes. And that's why our focus is the values, the principles, and ideas that made America great. Let's be clear on this point. We're for limited government, because we understand, as the Founding Fathers did, that it is the best way of ensuring personal liberty and empowering the individual so that every American of every race and region shares fully in the flowering of American prosperity and freedom. One other thing we Americans like—the future—like the sound of it, the idea of it, the hope of it. Where others fear trade and economic growth, we see opportunities for creating new wealth and undreamed-of opportunities for millions in our own land and beyond. Where others seek to throw up barriers, we seek to bring them down. Where others take counsel of their fears, we follow our hopes. Yes, we Americans like the future and like making the most of it. Let's do that now. And let's begin by discussing how to maintain economic growth by controlling and eventually eliminating the problem of Federal deficits. We have had a balanced budget only eight times in the last 57 years. For the first time in 14 years, the Federal Government spent less in real terms last year than the year before. We took $73 billion off last year's deficit compared to the year before. The deficit itself has moved from 6.3 percent of the gross national product to only 3.4 percent. And perhaps the most important sign of progress has been the change in our view of deficits. You know, a few of us can remember when, not too many years ago, those who created the deficits said they would make us prosperous and not to worry about the debt, because we owe it to ourselves. Well, at last there is agreement that we can't spend ourselves rich. Our recent budget agreement, designed to reduce Federal deficits by $76 billion over the next 2 years, builds on this consensus. But this agreement must be adhered to without slipping into the errors of the past: more broken promises and more unchecked spending. As I indicated in my first State of the Union, what ails us can be simply put: The Federal Government is too big, and it spends too much money. I can assure you, the bipartisan leadership of Congress, of my help in fighting off any attempt to bust our budget agreement. And this includes the swift and certain use of the veto power. Now, it's also time for some plain talk about the most immediate obstacle to controlling Federal deficits. The simple but frustrating problem of making expenses match revenues—something American families do and the Federal G",
      "overnment can't—has caused crisis after crisis in this city. Mr. Speaker, Mr. President, I will say to you tonight what I have said before and will continue to say: The budget process has broken down; it needs a drastic overhaul. With each ensuing year, the spectacle before the American people is the same as it was this Christmas: budget deadlines delayed or missed completely, monstrous continuing resolutions that pack hundreds of billions of dollars worth of spending into one bill, and a Federal Government on the brink of default. I know I'm echoing what you here in the Congress have said, because you suffered so directly. But let's recall that in 7 years, of 91 appropriations bills scheduled to arrive on my desk by a certain date, only 10 made it on time. Last year, of the 13 appropriations bills due by October 1st, none of them made it. Instead, we had four continuing resolutions lasting 41 days, then 36 days, and 2 days, and 3 days, respectively. And then, along came these behemoths. This is the conference report—1,053 pages, report weighing 14 pounds. Then this—a reconciliation bill 6 months late that was 1,186 pages long, weighing 15 pounds. And the long-term continuing resolution—this one was 2 months late, and it's 1,057 pages long, weighing 14 pounds. That was a total of 43 pounds of paper and ink. You had 3 hours—yes, 3 hours—to consider each, and it took 300 people at my Office of Management and Budget just to read the bill so the Government wouldn't shut down. Congress shouldn't send another one of these. No, and if you do, I will not sign it. Let's change all this. Instead of a Presidential budget that gets discarded and a congressional budget resolution that is not enforced, why not a simple partnership, a joint agreement that sets out the spending priorities within the available revenues? And let's remember our deadline is October 1st, not Christmas. Let's get the people's work done in time to avoid a footrace with Santa Claus. [Laughter] And, yes, this year—to coin a phrase—a new beginning: 13 individual bills, on time and fully reviewed by Congress. I'm also certain you join me in saying: Let's help ensure our future of prosperity by giving the President a tool that, though I will not get to use it, is one I know future Presidents of either party must have. Give the President the same authority that 43 Governors use in their States: the right to reach into massive appropriation bills, pare away the waste, and enforce budget discipline. Let's approve the line-item veto. And let's take a partial step in this direction. Most of you in this Chamber didn't know what was in this catchall bill and report. Over the past few weeks, we've all learned what was tucked away behind a little comma here and there. For example, there's millions for items such as cranberry resear",
      "ch, blueberry research, the study of crawfish, and the commercialization of wildflowers. And that's not to mention the five or so million [$.5 million] that—so that people from developing nations could come here to watch Congress at work. [Laughter] I won't even touch that. [Laughter] So, tonight I offer you this challenge. In 30 days I will send back to you those items as rescissions, which if I had the authority to line them out I would do so. Now, review this multibillion-dollar package that will not undercut our bipartisan budget agreement. As a matter of fact, if adopted, it will improve our deficit reduction goals. And what an example we can set, that we're serious about getting our financial accounts in order. By acting and approving this plan, you have the opportunity to override a congressional process that is out of control. There is another vital reform. Yes, Gramm-Rudman-Hollings has been profoundly helpful, but let us take its goal of a balanced budget and make it permanent. Let us do now what so many States do to hold down spending and what 32 State legislatures have asked us to do. Let us heed the wishes of an overwhelming plurality of Americans and pass a constitutional amendment that mandates a balanced budget and forces the Federal Government to live within its means. Reform of the budget process—including the line-item veto and balanced budget amendment—will, together with real restraint on government spending, prevent the Federal budget from ever again ravaging the family budget. Let's ensure that the Federal Government never again legislates against the family and the home. Last September I signed an Executive order on the family requiring that every department and agency review its activities in light of seven standards designed to promote and not harm the family. But let us make certain that the family is always at the center of the public policy process not just in this administration but in all future administrations. It's time for Congress to consider, at the beginning, a statement of the impact that legislation will have on the basic unit of American society, the family. And speaking of the family, let's turn to a matter on the mind of every American parent tonight: education. We all know the sorry story of the sixties and seventies-soaring spending, plummeting test scores-and that hopeful trend of the eighties, when we replaced an obsession with dollars with a commitment to quality, and test scores started back up. There's a lesson here that we all should write on the blackboard a hundred times: In a child's education, money can never take the place of basics like discipline, hard work, and, yes, homework. As a nation we do, of course, spend heavily on education—more than we spend on defense. Yet across our country, Governors like New Jersey's Tom Kea",
      "n are giving classroom demonstrations that how we spend is as important as how much we spend. Opening up the teaching profession to all qualified candidates, merit pay—so that good teachers get A's as well as apples—and stronger curriculum, as Secretary Bennett has proposed for high schools—these imaginative reforms are making common sense the most popular new kid in America's schools. How can we help? Well, we can talk about and push for these reforms. But the most important thing we can do is to reaffirm that control of our schools belongs to the States, local communities and, most of all, to the parents and teachers. My friends, some years ago, the Federal Government declared war on poverty, and poverty won. [Laughter] Today the Federal Government has 59 major welfare programs and spends more than $100 billion a year on them. What has all this money done? Well, too often it has only made poverty harder to escape. Federal welfare programs have created a massive social problem. With the best of intentions, government created a poverty trap that wreaks havoc on the very support system the poor need most to lift themselves out of poverty: the family. Dependency has become the one enduring heirloom, passed from one generation to the next, of too many fragmented families. It is time—this may be the most radical thing I've said in 7 years in this office—it's time for Washington to show a little humility. There are a thousand sparks of genius in 50 States and a thousand communities around the Nation. It is time to nurture them and see which ones can catch fire and become guiding lights. States have begun to show us the way. They've demonstrated that successful welfare programs can be built around more effective child support enforcement practices and innovative programs requiring welfare recipients to work or prepare for work. Let us give the States more flexibility and encourage more reforms. Let's start making our welfare system the first rung on America's ladder of opportunity, a boost up from dependency, not a graveyard but a birthplace of hope. And now let me turn to three other matters vital to family values and the quality of family life. The first is an untold American success story. Recently, we released our annual survey of what graduating high school seniors have to say about drugs. Cocaine use is declining, and marijuana use was the lowest since surveying began. We can be proud that our students are just saying no to drugs. But let us remember what this menace requires: commitment from every part of America and every single American, a commitment to a drugfree America. The war against drugs is a war of individual battles, a crusade with many heroes, including America's young people and also someone very special to me. She has helped so many of our young people to say no t",
      "o drugs. Nancy, much credit belongs to you, and I want to express to you your husband's pride and your country's thanks.'. Surprised you, didn't I? [Laughter] Well, now we come to a family issue that we must have the courage to confront. Tonight, I call America—a good nation, a moral people—to charitable but realistic consideration of the terrible cost of abortion on demand. To those who say this violates a woman's right to control of her own body: Can they deny that now medical evidence confirms the unborn child is a living human being entitled to life, liberty, and the pursuit of happiness? Let us unite as a nation and protect the unborn with legislation that would stop all Federal funding for abortion and with a human life amendment making, of course, an exception where the unborn child threatens the life of the mother. Our Judeo-Christian tradition recognizes the right of taking a life in self-defense. But with that one exception, let us look to those others in our land who cry out for children to adopt. I pledge to you tonight I will work to remove barriers to adoption and extend full sharing in family life to millions of Americans so that children who need homes can be welcomed to families who want them and love them. And let me add here: So many of our greatest statesmen have reminded us that spiritual values alone are essential to our nation's health and vigor. The Congress opens its proceedings each day, as does the Supreme Court, with an acknowledgment of the Supreme Being. Yet we are denied the right to set aside in our schools a moment each day for those who wish to pray. I believe Congress should pass our school prayer amendment. Now, to make sure there is a full nine member Supreme Court to interpret the law, to protect the rights of all Americans, I urge the Senate to move quickly and decisively in confirming Judge Anthony Kennedy to the highest Court in the land and to also confirm 27 nominees now waiting to fill vacancies in the Federal judiciary. Here then are our domestic priorities. Yet if the Congress and the administration work together, even greater opportunities lie ahead to expand a growing world economy, to continue to reduce the threat of nuclear arms, and to extend the frontiers of freedom and the growth of democratic institutions. Our policies consistently received the strongest support of the late Congressman Dan Daniel of Virginia. I'm sure all of you join me in expressing heartfelt condolences on his passing. One of the greatest contributions the United States can make to the world is to promote freedom as the key to economic growth. A creative, competitive America is the answer to a changing world, not trade wars that would close doors, create greater barriers, and destroy millions of jobs. We should always remember: Protectionism is destructioni",
      "sm. America's jobs, America's growth, America's future depend on trade—trade that is free, open, and fair. This year, we have it within our power to take a major step toward a growing global economy and an expanding cycle of prosperity that reaches to all the free nations of this Earth. I'm speaking of the historic free trade agreement negotiated between our country and Canada. And I can also tell you that we're determined to expand this concept, south as well as north. Next month I will be traveling to Mexico, where trade matters will be of foremost concern. And over the next several months, our Congress and the Canadian Parliament can make the start of such a North American accord a reality. Our goal must be a day when the free flow of trade, from the tip of Tierra del Fuego to the Arctic Circle, unites the people of the Western Hemisphere in a bond of mutually beneficial exchange, when all borders become what the U.S.-Canadian border so long has been: a meeting place rather than a dividing line. This movement we see in so many places toward economic freedom is indivisible from the worldwide movement toward political freedom and against totalitarian rule. This global democratic revolution has removed the specter, so frightening a decade ago, of democracy doomed to permanent minority status in the world. In South and Central America, only a third of the people enjoyed democratic rule in 1976. Today over 90 percent of Latin Americans live in nations committed to democratic principles. And the resurgence of democracy is owed to these courageous people on almost every continent who have struggled to take control of their own destiny. In Nicaragua the struggle has extra meaning, because that nation is so near our own borders. The recent revelations of a former high-level Sandinista major, Roger Miranda, show us that, even as they talk peace, the Communist Sandinista government of Nicaragua has established plans for a large 600,000-man army. Yet even as these plans are made, the Sandinista regime knows the tide is turning, and the cause of Nicaraguan freedom is riding at its crest. Because of the freedom fighters, who are resisting Communist rule, the Sandinistas have been forced to extend some democratic rights, negotiate with church authorities, and release a few political prisoners. The focus is on the Sandinistas, their promises and their actions. There is a consensus among the four Central American democratic Presidents that the Sandinistas have not complied with the plan to bring peace and democracy to all of Central America. The Sandinistas again have promised reforms. Their challenge is to take irreversible steps toward democracy. On Wednesday my request to sustain the freedom fighters will be submitted, which reflects our mutual desire for peace, freedom, and democracy in N",
      "icaragua. I ask Congress to pass this request. Let us be for the people of Nicaragua what Lafayette, Pulaski, and Von Steuben were for our forefathers and the cause of American independence. So, too, in Afghanistan, the freedom fighters are the key to peace. We support the Mujahidin. There can be no settlement unless all Soviet troops are removed and the Afghan people are allowed genuine self-determination. I have made my views on this matter known to Mr. Gorbachev. But not just Nicaragua or Afghanistan—yes, everywhere we see a swelling freedom tide across the world: freedom fighters rising up in Cambodia and Angola, fighting and dying for the same democratic liberties we hold sacred. Their cause is our cause: freedom. Yet even as we work to expand world freedom, we must build a safer peace and reduce the danger of nuclear war. But let's have no illusions. Three years of steady decline in the value of our annual defense investment have increased the risk of our most basic security interests, jeopardizing earlier hard-won goals. We must face squarely the implications of this negative trend and make adequate, stable defense spending a top goal both this year and in the future. This same concern applies to economic and security assistance programs as well. But the resolve of America and its NATO allies has opened the way for unprecedented achievement in arms reduction. Our recently signed INF treaty is historic, because it reduces nuclear arms and establishes the most stringent verification regime in arms control history, including several forms of short-notice, on-site inspection. I submitted the treaty today, and I urge the Senate to give its advice and consent to ratification of this landmark agreement. [Applause] Thank you very much. In addition to the INF treaty, we're within reach of an even more significant START agreement that will reduce U.S. and Soviet long-range missile—or strategic arsenals by half. But let me be clear. Our approach is not to seek agreement for agreement's sake but to settle only for agreements that truly enhance our national security and that of our allies. We will never put our security at risk—or that of our allies-just to reach an agreement with the Soviets. No agreement is better than a bad agreement. As I mentioned earlier, our efforts are to give future generations what we never had—a future free of nuclear terror. Reduction of strategic offensive arms is one step, SDI another. Our funding request for our Strategic Defense Initiative is less than 2 percent of the total defense budget. SDI funding is money wisely appropriated and money well spent. SDI has the same purpose and supports the same goals of arms reduction. It reduces the risk of war and the threat of nuclear weapons to all mankind. Strategic defenses that threaten no one could offer th",
      "e world a safer, more stable basis for deterrence. We must also remember that SDI is our insurance policy against a nuclear accident, a Chernobyl of the sky, or an accidental launch or some madman who might come along. We've seen such changes in the world in 7 years. As totalitarianism struggles to avoid being overwhelmed by the forces of economic advance and the aspiration for human freedom, it is the free nations that are resilient and resurgent. As the global democratic revolution has put totalitarianism on the defensive, we have left behind the days of retreat. America is again a vigorous leader of the free world, a nation that acts decisively and firmly in the furtherance of her principles and vital interests. No legacy would make me more proud than leaving in place a bipartisan consensus for the cause of world freedom, a consensus that prevents a paralysis of American power from ever occurring again. But my thoughts tonight go beyond this, and I hope you'll let me end this evening with a personal reflection. You know, the world could never be quite the same again after Jacob Shallus, a trustworthy and dependable clerk of the Pennsylvania General Assembly, took his pen and engrossed those words about representative government in the preamble of our Constitution. And in a quiet but final way, the course of human events was forever altered when, on a ridge overlooking the Emmitsburg Pike in an obscure Pennsylvania town called Gettysburg, Lincoln spoke of our duty to government of and by the people and never letting it perish from the Earth. At the start of this decade, I suggested that we live in equally momentous times, that it is up to us now to decide whether our form of government would endure and whether history still had a place of greatness for a quiet, pleasant, greening land called America. Not everything has been made perfect in 7 years, nor will it be made perfect in seven times 70 years, but before us, this year and beyond, are great prospects for the cause of peace and world freedom. It means, too, that the young Americans I spoke of 7 years ago, as well as those who might be coming along the Virginia or Maryland shores this night and seeing for the first time the lights of this Capital City—the lights that cast their glow on our great halls of government and the monuments to the memory of our great men—it means those young Americans will find a city of hope in a land that is free. We can be proud that for them and for us, as those lights along the Potomac are still seen this night signaling as they have for nearly two centuries and as we pray God they always will, that another generation of Americans has protected and passed on lovingly this place called America, this shining city on a hill, this government of, by, and for the people. Thank you, and God bless yo"],
     ['Thank you very much. Thank you. Thank you very much. Madam Speaker, Mr. Vice President, members of Congress, the First Lady of the United States—(applause)—and my fellow citizens: Three years ago, we launched the great American comeback. Tonight, I stand before you to share the incredible results. Jobs are booming, incomes are soaring, poverty is plummeting, crime is falling, confidence is surging, and our country is thriving and highly respected again. (Applause.) America’s enemies are on the run, America’s fortunes are on the rise, and America’s future is blazing bright. The years of economic decay are over. (Applause.) The days of our country being used, taken advantage of, and even scorned by other nations are long behind us. (Applause.) Gone too are the broken promises, jobless recoveries, tired platitudes, and constant excuses for the depletion of American wealth, power, and prestige. In just three short years, we have shattered the mentality of American decline, and we have rejected the downsizing of America’s destiny. We have totally rejected the downsizing. We are moving forward at a pace that was unimaginable just a short time ago, and we are never, ever going back. (Applause.) I am thrilled to report to you tonight that our economy is the best it has ever been. Our military is completely rebuilt, with its power being unmatched anywhere in the world—and it’s not even close. Our borders are secure. Our families are flourishing. Our values are renewed. Our pride is restored. And for all of these reasons, I say to the people of our great country and to the members of Congress: The state of our Union is stronger than ever before. (Applause.) The vision I will lay out this evening demonstrates how we are building the world’s most prosperous and inclusive society—one where every citizen can join in America’s unparalleled success and where every community can take part in America’s extraordinary rise. From the instant I took office, I moved rapidly to revive the U.S. economy—slashing a record number of job-killing regulations, enacting historic and record-setting tax cuts, and fighting for fair and reciprocal trade agreements. (Applause.) Our agenda is relentlessly pro-worker, pro-family, pro-growth, and, most of all, pro-American. (Applause.) Thank you. We are advancing with unbridled optimism and lifting our citizens of every race, color, religion, and creed very, very high. Since my election, we have created 7 million new jobs—5 million more than government experts projected during the previous administration. (Applause.) The unemployment rate is the lowest in over half a century. (Applause.) And very incredibly, the average unemployment rate under my administration is lower than any administration in the history of our country. (Applause.) True. If we hadn’t reversed the failed economic policies of the previous administration, the world would not now be witnessing this great economic success. (Applause.) The unemployment rate for African Americans, Hispanic Americans, and Asian Americans has reached the lowest levels in history. (Applause.) African American youth unemployment has reached an all-time low. (Applause.) African American poverty has declined to the lowest rate ever recorded. (Applause.) The unemployment rate for women reached the lowest level in almost 70 years. And, last year, women filled 72 percent of all new jobs added. (Applause.) The veterans unemployment rate dropped to a record low. (Applause.) The unemployment rate for disabled Americans has reached an all-time low. (Applause.) Workers without a high school diploma have achieved the lowest unemployment rate recorded in U.S. history. (Applause.) A record number of young Americans are now employed. (Applause.) Under the last administration, more than 10 million people were added to the food stamp rolls. Under my administration, 7 million Americans have',
      ' come off food stamps, and 10 million people have been lifted off of welfare. (Applause.) In eight years under the last administration, over 300,000 working-age people dropped out of the workforce. In just three years of my administration, 3.5 million people—working-age people—have joined the workforce. (Applause.) Since my election, the net worth of the bottom half of wage earners has increased by 47 percent—three times faster than the increase for the top 1 percent. (Applause.) After decades of flat and falling incomes, wages are rising fast—and, wonderfully, they are rising fastest for low-income workers, who have seen a 16 percent pay increase since my election. (Applause.) This is a blue-collar boom. (Applause.) Real median household income is now at the highest level ever recorded. (Applause.) Since my election, U.S. stock markets have soared 70 percent, adding more than $12 trillion to our nation’s wealth, transcending anything anyone believed was possible. This is a record. It is something that every country in the world is looking up to. They admire. (Applause.) Consumer confidence has just reached amazing new highs. All of those millions of people with 401(k)s and pensions are doing far better than they have ever done before with increases of 60, 70, 80, 90, and 100 percent, and even more. Jobs and investments are pouring into 9,000 previously neglected neighborhoods thanks to Opportunity Zones, a plan spearheaded by Senator Tim Scott as part of our great Republican tax cuts. (Applause.) In other words, wealthy people and companies are pouring money into poor neighborhoods or areas that haven’t seen investment in many decades, creating jobs, energy, and excitement. (Applause.) This is the first time that these deserving communities have seen anything like this. It’s all working. Opportunity Zones are helping Americans like Army veteran Tony Rankins from Cincinnati, Ohio. After struggling with drug addiction, Tony lost his job, his house, and his family. He was homeless. But then Tony found a construction company that invests in Opportunity Zones. He is now a top tradesman, drug-free, reunited with his family, and he is here tonight. Tony, keep up the great work. Tony. (Applause.) Thank you, Tony. Our roaring economy has, for the first time ever, given many former prisoners the ability to get a great job and a fresh start. This second chance at life is made possible because we passed landmark criminal justice reform into law. Everybody said that criminal justice reform couldn’t be done, but I got it done, and the people in this room got it done. (Applause.) Thanks to our bold regulatory reduction campaign, the United States has become the number one producer of oil and natural gas anywhere in the world, by far. (Applause.) With the tremendous progress we have made over the past three years, America is now energy independent, and energy jobs, like so many other elements of our country, are at a record high. (Applause.) We are doing numbers that no one would have thought possible just three years ago. Likewise, we are restoring our nation’s manufacturing might, even though predictions were, as you all know, that this could never, ever be done. After losing 60,000 factories under the previous two administrations, America has now gained 12,000 new factories under my administration, with thousands upon thousands of plants and factories being planned or being built. (Applause.) Companies are not leaving; they are coming back to the USA. (Applause.) The fact is that everybody wants to be where the action is, and the United States of America is indeed the place where the action is. (Applause.) One of the biggest promises I made to the American people was to replace the disastrous NAFTA trade deal. (Applause.) In fact, unfair trade is perhaps the single biggest reason that I decided to run for President. Following NAFTA’s adopt',
      'ion, our nation lost one in four manufacturing jobs. Many politicians came and went, pledging to change or replace NAFTA, only to do so, and then absolutely nothing happened. But unlike so many who came before me, I keep my promises. We did our job. (Applause.) Six days ago, I replaced NAFTA and signed the brand-new U.S.-Mexico-Canada Agreement into law. The USMCA will create nearly 100,000 new high-paying American auto jobs, and massively boost exports for our farmers, ranchers, and factory workers. (Applause.) It will also bring trade with Mexico and Canada to a much higher level, but also to be a much greater degree of fairness and reciprocity. We will have that: fairness and reciprocity. And I say that, finally, because it’s been many, many years that we were treated fairly on trade. (Applause.) This is the first major trade deal in many years to earn the strong backing of America’s labor unions. (Applause.) I also promised our citizens that I would impose tariffs to confront China’s massive theft of America’s jobs. Our strategy has worked. Days ago, we signed the groundbreaking new agreement with China that will defend our workers, protect our intellectual property, bring billions and billions of dollars into our treasury, and open vast new markets for products made and grown right here in the USA. (Applause.) For decades, China has taken advantage of the United States. Now we have changed that, but, at the same time, we have perhaps the best relationship we’ve ever had with China, including with President Xi. They respect what we’ve done because, quite frankly, they could never really believe that they were able to get away with what they were doing year after year, decade after decade, without someone in our country stepping up and saying, “That’s enough.” (Applause.) Now we want to rebuild our country, and that’s exactly what we’re doing. We are rebuilding our country. As we restore American leadership throughout the world, we are once again standing up for freedom in our hemisphere. (Applause.) That’s why my administration reversed the failing policies of the previous administration on Cuba. (Applause.) We are supporting the hopes of Cubans, Nicaraguans, and Venezuelans to restore democracy. The United States is leading a 59-nation diplomatic coalition against the socialist dictator of Venezuela, Nicolás Maduro. (Applause.) Maduro is an illegitimate ruler, a tyrant who brutalizes his people. But Maduro’s grip on tyranny will be smashed and broken. Here this evening is a very brave man who carries with him the hopes, dreams, and aspirations of all Venezuelans. Joining us in the Gallery is the true and legitimate President of Venezuela, Juan Guaidó. (Applause.) Mr. President, please take this message back to your homeland. (Applause.) Thank you, Mr. President. Great honor. Thank you very much. Please take this message back that all Americans are united with the Venezuelan people in their righteous struggle for freedom. Thank you very much, Mr. President. (Applause.) Thank you very much. Socialism destroys nations. But always remember: Freedom unifies the soul. (Applause.) To safeguard American liberty, we have invested a record-breaking $2.2 trillion in the United States military. (Applause.) We have purchased the finest planes, missiles, rockets, ships, and every other form of military equipment, and it’s all made right here in the USA. (Applause.) We are also getting our allies, finally, to help pay their fair share. (Applause.) I have raised contributions from the other NATO members by more than $400 billion, and the number of Allies meeting their minimum obligations has more than doubled. And just weeks ago, for the first time since President Truman established the Air Force more than 70 years earlier, we created a brand-new branch of the United States Armed Forces. It’s called the Space Force. (Applause.) Very impor',
      'tant. In the Gallery tonight, we have a young gentleman. And what he wants so badly—13 years old—Iain Lanphier. He’s an eighth grader from Arizona. Iain, please stand up. Iain has always dreamed of going to space. He was the first in his class and among the youngest at an aviation academy. He aspires to go to the Air Force Academy, and then he has his eye on the Space Force. As Iain says, “Most people look up at space. I want to look down on the world.” (Laughter and applause.) But sitting behind Iain tonight is his greatest hero of them all. Charles McGee was born in Cleveland, Ohio, one century ago. Charles is one of the last surviving Tuskegee Airmen—the first black fighter pilots—and he also happens to be Iain’s great-grandfather. (Applause.) Incredible story. After more than 130 combat missions in World War Two, he came back home to a country still struggling for civil rights and went on to serve America in Korea and Vietnam. On December 7th, Charles celebrated his 100th birthday. (Applause.) A few weeks ago, I signed a bill promoting Charles McGee to Brigadier General. And earlier today, I pinned the stars on his shoulders in the Oval Office. General McGee, our nation salutes you. Thank you, sir. (Applause.) From the pilgrims to the Founders, from the soldiers at Valley Forge to the marchers at Selma, and from President Lincoln to the Reverend Martin Luther King, Americans have always rejected limits on our children’s future. Members of Congress, we must never forget that the only victories that matter in Washington are victories that deliver for the American people. (Applause.) The people are the heart of our country, their dreams are the soul of our country, and their love is what powers and sustains our country. We must always remember that our job is to put America first. (Applause.) The next step forward in building an inclusive society is making sure that every young American gets a great education and the opportunity to achieve the American Dream. Yet, for too long, countless American children have been trapped in failing government schools. To rescue these students, 18 states have created school choice in the form of Opportunity Scholarships. The programs are so popular that tens of thousands of students remain on a waiting list. One of those students is Janiyah Davis, a fourth grader from Philadelphia. Janiyah. (Applause.) Janiyah’s mom, Stephanie, is a single parent. She would do anything to give her daughter a better future. But last year, that future was put further out of reach when Pennsylvania’s governor vetoed legislation to expand school choice to 50,000 children. Janiyah and Stephanie are in the Gallery. Stephanie, thank you so much for being here with your beautiful daughter. Thank you very much. (Applause.) But, Janiyah, I have some good news for you, because I am pleased to inform you that your long wait is over. I can proudly announce tonight that an Opportunity Scholarship has become available, it’s going to you, and you will soon be heading to the school of your choice. (Applause.) Now I call on Congress to give one million American children the same opportunity Janiyah has just received. Pass the Education Freedom Scholarships and Opportunities Act—because no parent should be forced to send their child to a failing government school. (Applause.) Every young person should have a safe and secure environment in which to learn and to grow. For this reason, our magnificent First Lady has launched the BE BEST initiative to advance a safe, healthy, supportive, and drug-free life for the next generation—online, in school, and in our communities. Thank you, Melania, for your extraordinary love and profound care for America’s children. Thank you very much. (Applause.) My administration is determined to give our citizens the opportunities they need regardless of age or background. Through our Pledge to Ameri',
      'can Workers, over 400 companies will also provide new jobs and education opportunities to almost 15 million Americans. My budget also contains an exciting vision for our nation’s high schools. Tonight, I ask Congress to support our students and back my plan to offer vocational and technical education in every single high school in America. (Applause.) To expand equal opportunity, I am also proud that we achieved record and permanent funding for our nation’s historically black colleges and universities. (Applause.) A good life for American families also requires the most affordable, innovative, and high-quality healthcare system on Earth. Before I took office, health insurance premiums had more than doubled in just five years. I moved quickly to provide affordable alternatives. Our new plans are up to 60 percent less expensive—and better. (Applause.) I’ve also made an ironclad pledge to American families: We will always protect patients with pre-existing conditions. (Applause). And we will always protect your Medicare and we will always protect your Social Security. Always. (Applause.) The American patient should never be blindsided by medical bills. That is why I signed an executive order requiring price transparency. (Applause.) Many experts believe that transparency, which will go into full effect at the beginning of next year, will be even bigger than healthcare reform. (Applause.) It will save families massive amounts of money for substantially better care. But as we work to improve Americans’ healthcare, there are those who want to take away your healthcare, take away your doctor, and abolish private insurance entirely. AUDIENCE: Booo THE PRESIDENT: One hundred thirty-two lawmakers in this room have endorsed legislation to impose a socialist takeover of our healthcare system, wiping out the private health insurance plans of 180 million very happy Americans. To those watching at home tonight, I want you to know: We will never let socialism destroy American healthcare. (Applause.) Over 130 legislators in this chamber have endorsed legislation that would bankrupt our nation by providing free taxpayer-funded healthcare to millions of illegal aliens, forcing taxpayers to subsidize free care for anyone in the world who unlawfully crosses our borders. These proposals would raid the Medicare benefits of our seniors and that our seniors depend on, while acting as a powerful lure for illegal immigration. That is what is happening in California and other states. Their systems are totally out of control, costing taxpayers vast and unaffordable amounts of money. If forcing American taxpayers to provide unlimited free healthcare to illegal aliens sounds fair to you, then stand with the radical left. But if you believe that we should defend American patients and American seniors, then stand with me and pass legislation to prohibit free government healthcare for illegal aliens. (Applause.) This will be a tremendous boon to our already very strongly guarded southern border where, as we speak, a long, tall, and very powerful wall is being built. (Applause.) We have now completed over 100 miles and have over 500 miles fully completed in a very short period of time. Early next year, we will have substantially more than 500 miles completed. My administration is also taking on the big pharmaceutical companies. We have approved a record number of affordable generic drugs, and medicines are being approved by the FDA at a faster clip than ever before. (Applause.) And I was pleased to announce last year that, for the first time in 51 years, the cost of prescription drugs actually went down. (Applause.) And working together, Congress can reduce drug prices substantially from current levels. I’ve been speaking to Senator Chuck Grassley of Iowa and others in Congress in order to get something on drug pricing done, and done quickly and properly. I’m ca',
      'lling for bipartisan legislation that achieves the goal of dramatically lowering prescription drug prices. Get a bill on my desk, and I will sign it into law immediately. (Applause.) AUDIENCE: H.R.3! H.R.3! H.R.3! With unyielding commitment, we are curbing the opioid epidemic. Drug overdose deaths declined for the first time in nearly 30 years. (Applause.) Among the states hardest hit, Ohio is down 22 percent, Pennsylvania is down 18 percent, Wisconsin is down 10 percent—and we will not quit until we have beaten the opioid epidemic once and for all. (Applause.) Protecting Americans’ health also means fighting infectious diseases. We are coordinating with the Chinese government and working closely together on the coronavirus outbreak in China. My administration will take all necessary steps to safeguard our citizens from this threat. We have launched ambitious new initiatives to substantially improve care for Americans with kidney disease, Alzheimer’s, and those struggling with mental health. And because Congress was so good as to fund my request, new cures for childhood cancer, and we will eradicate the AIDS epidemic in America by the end of this decade. (Applause.) Almost every American family knows the pain when a loved one is diagnosed with a serious illness. Here tonight is a special man, beloved by millions of Americans who just received a Stage 4 advanced cancer diagnosis. This is not good news, but what is good news is that he is the greatest fighter and winner that you will ever meet. Rush Limbaugh, thank you for your decades of tireless devotion to our country. (Applause.) And, Rush, in recognition of all that you have done for our nation, the millions of people a day that you speak to and that you inspire, and all of the incredible work that you have done for charity, I am proud to announce tonight that you will be receiving our country’s highest civilian honor, the Presidential Medal of Freedom. (Applause.) I will now ask the First Lady of the United States to present you with the honor. Please. (Applause.) (The Medal of Freedom is presented.) (Applause.) Rush and Kathryn, congratulations. Thank you, Kathryn. As we pray for all who are sick, we know that America is constantly achieving new medical breakthroughs. In 2017, doctors at St. Luke’s Hospital in Kansas City delivered one of the earliest premature babies ever to survive. Born at just 21 weeks and 6 days, and weighing less than a pound, Ellie Schneider was a born fighter. Through the skill of her doctors and the prayers of her parents, little Ellie kept on winning the battle of life. Today, Ellie is a strong, healthy two-year-old girl sitting with her amazing mother Robin in the Gallery. Ellie and Robin, we are glad to have you with us tonight. (Applause.) Ellie reminds us that every child is a miracle of life. And thanks to modern medical wonders, 50 percent of very premature babies delivered at the hospital where Ellie was born now survive. It’s an incredible thing. Thank you very much. (Applause.) Our goal should be to ensure that every baby has the best chance to thrive and grow just like Ellie. That is why I’m asking Congress to provide an additional $50 million to fund neonatal research for America’s youngest patients. (Applause.) That is why I’m also calling upon members of Congress here tonight to pass legislation finally banning the late-term abortion of babies. (Applause.) Whether we are Republican, Democrat, or independent, surely we must all agree that every human life is a sacred gift from God. As we support America’s moms and dads, I was recently proud to sign the law providing new parents in the federal workforce paid family leave, serving as a model for the rest of the country. (Applause.) Now I call on the Congress to pass the bipartisan Advancing Support for Working Families Act, extending family leave to mothers and fathers all across our na',
      'tion. (Applause.) Forty million American families have an average $2,200 extra thanks to our child tax credit. (Applause.) I’ve also overseen historic funding increases for high-quality child care, enabling 17 states to help more children, many of which have reduced or eliminated their waitlists altogether. (Applause.) And I sent Congress a plan with a vision to further expand access to high-quality child care, and urge you to act immediately. (Applause.) To protect the environment, days ago I announced that the United States will join the One Trillion Trees Initiative, an ambitious effort to bring together government and private sector to plant new trees in America and all around the world. (Applause.) We must also rebuild America’s infrastructure. (Applause.) I ask you to pass Senator John Barrasso’s highway bill to invest in new roads, bridges, and tunnels all across our land. I’m also committed to ensuring that every citizen can have access to high-speed Internet, including and especially in rural America. (Applause.) A better tomorrow for all Americans also requires us to keep America safe. That means supporting the men and women of law enforcement at every level, including our nation’s heroic ICE officers. (Applause.) Last year, our brave ICE officers arrested more than 120,000 criminal aliens charged with nearly 10,000 burglaries, 5,000 sexual assaults, 45,000 violent assaults, and 2,000 murders. Tragically, there are many cities in America where radical politicians have chosen to provide sanctuary for these criminal illegal aliens. AUDIENCE: Booo THE PRESIDENT: In sanctuary cities, local officials order police to release dangerous criminal aliens to prey upon the public, instead of handing them over to ICE to be safely removed. Just 29 days ago, a criminal alien freed by the sanctuary city of New York was charged with the brutal rape and murder of a 92-year-old woman. The killer had been previously arrested for assault, but under New York’s sanctuary policies, he was set free. If the city had honored ICE’s detainer request, his victim would still be alive today. The state of California passed an outrageous law declaring their whole state to be a sanctuary for criminal illegal immigrants—a very terrible sanctuary—with catastrophic results. Here is just one tragic example. In December 2018, California police detained an illegal alien with five prior arrests, including convictions for robbery and assault. But as required by California’s Sanctuary Law, local authorities released him. Days later, the criminal alien went on a gruesome spree of deadly violence. He viciously shot one man going about his daily work. He approached a woman sitting in her car and shot her in the arm and in the chest. He walked into a convenience store and wildly fired his weapon. He hijacked a truck and smashed into vehicles, critically injuring innocent victims. One of the victims is—a terrible, terrible situation; died—51-year-old American named Rocky Jones. Rocky was at a gas station when this vile criminal fired eight bullets at him from close range, murdering him in cold blood. Rocky left behind a devoted family, including his brothers, who loved him more than anything else in the world. One of his grieving brothers is here with us tonight. Jody, would you please stand? Jody, thank you. (Applause.) Jody our hearts weep for your loss, and we will not rest until you have justice. Senator Thom Tillis has introduced legislation to allow Americans like Jody to sue sanctuary cities and states when a loved one is hurt or killed as a result of these deadly practices. (Applause.) I ask Congress to pass the Justice for Victims of Sanctuary Cities Act immediately. The United States of America should be a sanctuary for law-abiding Americans, not criminal aliens. (Applause.) In the last three years, ICE has arrested over 5,000 wicked human traffickers. And',
      ' I have signed nine pieces of legislation to stamp out the menace of human trafficking, domestically and all around the globe. My administration has undertaken an unprecedented effort to secure the southern border of the United States. (Applause.) Before I came into office, if you showed up illegally on our southern border and were arrested, you were simply released and allowed into our country, never to be seen again. My administration has ended catch and release. (Applause.) If you come illegally, you will now be promptly removed from our country. (Applause.) Very importantly, we entered into historic cooperation agreements with the governments of Mexico, Honduras, El Salvador, and Guatemala. As a result of our unprecedented efforts, illegal crossings are down 75 percent since May, dropping eight straight months in a row. (Applause.) And as the wall rapidly goes up, drug seizures rise, and the border crossings are down, and going down very rapidly. Last year, I traveled to the border in Texas and met Chief Patrol Agent Raul Ortiz. Over the last 24 months, Agent Ortiz and his team have seized more than 200,000 pounds of poisonous narcotics, arrested more than 3,000 human smugglers, and rescued more than 2,000 migrants. Days ago, Agent Ortiz was promoted to Deputy Chief of Border Patrol, and he joins us tonight. Chief Ortiz, please stand. (Applause.) A grateful nation thanks you and all of the heroes of Border Patrol and ICE. Thank you very much. Thank you. (Applause.) To build on these historic gains, we are working on legislation to replace our outdated and randomized immigration system with one based on merit, welcoming those who follow the rules, contribute to our economy, support themselves financially, and uphold our values. (Applause.) With every action, my administration is restoring the rule of law and reasserting the culture of American freedom. (Applause.) Working with Senate Majority Leader Mitch McConnell—thank you, Mitch—and his colleagues in the Senate, we have confirmed a record number of 187 new federal judges to uphold our Constitution as written. This includes two brilliant new Supreme Court justices, Neil Gorsuch and Brett Kavanaugh. Thank you. (Applause.) And we have many in the pipeline. (Laughter and applause.) My administration is also defending religious liberty, and that includes the constitutional right to pray in public schools. (Applause.) In America, we don’t punish prayer. We don’t tear down crosses. We don’t ban symbols of faith. We don’t muzzle preachers and pastors. In America, we celebrate faith, we cherish religion, we lift our voices in prayer, and we raise our sights to the Glory of God. Just as we believe in the First Amendment, we also believe in another constitutional right that is under siege all across our country. So long as I am President, I will always protect your Second Amendment right to keep and bear arms. (Applause.) In reaffirming our heritage as a free nation, we must remember that America has always been a frontier nation. Now we must embrace the next frontier, America’s manifest destiny in the stars. I am asking Congress to fully fund the Artemis program to ensure that the next man and the first woman on the Moon will be American astronauts using this as a launching pad to ensure that America is the first nation to plant its flag on Mars. (Applause.) My administration is also strongly defending our national security and combating radical Islamic terrorism. (Applause.) Last week, I announced a groundbreaking plan for peace between Israel and the Palestinians. Recognizing that all past attempts have failed, we must be determined and creative in order to stabilize the region and give millions of young people the chance to realize a better future. Three years ago, the barbarians of ISIS held over 20,000 square miles of territory in Iraq and Syria. Today, the ISIS territorial ca',
      'liphate has been 100 percent destroyed, and the founder and leader of ISIS—the bloodthirsty killer known as al-Baghdadi—is dead. (Applause.) We are joined this evening by Carl and Marsha Mueller. After graduating from college, their beautiful daughter Kayla became a humanitarian aid worker. She once wrote, “Some people find God in church. Some people find God in nature. Some people find God in love. I find God in suffering. I’ve known for some time what my life’s work is, using my hands as tools to relieve suffering.” In 2013, while caring for suffering civilians in Syria, Kayla was kidnapped, tortured, and enslaved by ISIS, and kept as a prisoner of al-Baghdadi himself. After more than 500 horrifying days of captivity, al-Baghdadi murdered young, beautiful Kayla. She was just 26 years old. On the night that U.S. Special Forces Operations ended al-Baghdadi’s miserable life, the Chairman of the Joint Chiefs of Staff, General Mark Milley, received a call in the Situation Room. He was told that the brave men of the elite Special Forces team that so perfectly carried out the operation had given their mission a name: “Task Force 8-14.” It was a reference to a special day:\xa0August 14th—Kayla’s birthday. Carl and Marsha, America’s warriors never forgot Kayla—and neither will we. Thank you. (Applause.) Every day, America’s men and women in uniform demonstrate the infinite depth of love that dwells in the human heart. One of these American heroes was Army Staff Sergeant Christopher Hake. On his second deployment to Iraq in 2008, Sergeant Hake wrote a letter to his one-year-old son, Gage: “I will be with you again,” he wrote to Gage. “I will teach you to ride your first bike, build your first sand box, watch you play sports, and see you have kids also. I love you son. Take care of your mother. I am always with you. Daddy.” On Easter Sunday of 2008, Chris was out on patrol in Baghdad when his Bradley Fighting Vehicle was hit by a roadside bomb. That night, he made the ultimate sacrifice for our country. Sergeant Hake now rests in eternal glory in Arlington, and his wife Kelli is in the Gallery tonight, joined by their son, who is now a 13-year-old and doing very, very well. To Kelli and Gage: Chris will live in our hearts forever. He is looking down on you now. Thank you. (Applause.) Thank you very much. Thank you both very much. The terrorist responsible for killing Sergeant Hake was Qasem Soleimani, who provided the deadly roadside bomb that took Chris’s life. Soleimani was the Iranian regime’s most ruthless butcher, a monster who murdered or wounded thousands of American service members in Iraq. As the world’s top terrorist, Soleimani orchestrated the deaths of countless men, women, and children. He directed the December assault and went on to assault U.S. forces in Iraq. Was actively planning new attacks when we hit him very hard. And that’s why, last month, at my direction, the U.S. military executed a flawless precision strike that killed Soleimani and terminated his evil reign of terror forever. (Applause.) Our message to the terrorists is clear: You will never escape American justice. If you attack our citizens, you forfeit your life. (Applause.) In recent months, we have seen proud Iranians raise their voices against their oppressive rulers. The Iranian regime must abandon its pursuit of nuclear weapons; stop spreading terror, death, and destruction; and start working for the good of its own people. Because of our powerful sanctions, the Iranian economy is doing very, very poorly. We can help them make a very good and short-time recovery. It can all go very quickly, but perhaps they are too proud or too foolish to ask for that help. We are here. Let’s see which road they choose. It is totally up to them. (Applause.) As we defend American lives, we are working to end America’s wars in the Middle East. In Afghanistan, the determinat',
      'ion and valor of our warfighters has allowed us to make tremendous progress, and peace talks are now underway. I am not looking to kill hundreds of thousands of people in Afghanistan, many of them totally innocent. It is also not our function to serve other nations as law enforcement agencies. These are warfighters that we have—the best in the world—and they either want to fight to win or not fight at all. We are working to finally end America’s longest war and bring our troops back home. (Applause.) War places a heavy burden on our nation’s extraordinary military families, especially spouses like Amy Williams from Fort Bragg, North Carolina, and her two children—six-year-old Elliana and three-year-old Rowan. Amy works full-time and volunteers countless hours helping other military families. For the past seven months, she has done it all while her husband, Sergeant First Class Townsend Williams, is in Afghanistan on his fourth deployment in the Middle East. Amy’s kids haven’t seen their father’s face in many months. Amy, your family’s sacrifice makes it possible for all of our families to live in safety and in peace, and we want to thank you. Thank you, Amy. (Applause.) But, Amy, there is one more thing. Tonight, we have a very special surprise. I am thrilled to inform you that your husband is back from deployment. He is here with us tonight, and we couldn’t keep him waiting any longer. (Applause.) AUDIENCE: USA! USA! USA! THE PRESIDENT: Welcome home, Sergeant Williams. Thank you very much. As the world bears witness tonight, America is a land of heroes. This is a place where greatness is born, where destinies are forged, and where legends come to life. This is the home of Thomas Edison and Teddy Roosevelt, of many great generals including Washington, Pershing, Patton, and MacArthur. This is the home of Abraham Lincoln, Frederick Douglass, Amelia Earhart, Harriet Tubman, the Wright Brothers, Neil Armstrong, and so many more. This is the country where children learn names like Wyatt Earp, Davy Crockett, and Annie Oakley. This is the place where the pilgrims landed at Plymouth and where Texas patriots made their last stand at the Alamo—(applause)—the beautiful, beautiful Alamo. The American nation was carved out of the vast frontier by the toughest, strongest, fiercest, and most determined men and women ever to walk on the face of the Earth. Our ancestors braved the unknown; tamed the wilderness; settled the Wild West; lifted millions from poverty, disease, and hunger; vanquished tyranny and fascism; ushered the world to new heights of science and medicine; laid down the railroads, dug out the canals, raised up the skyscrapers. And, ladies and gentlemen, our ancestors built the most exceptional republic ever to exist in all of human history, and we are making it greater than ever before. (Applause.) This is our glorious and magnificent inheritance. We are Americans. We are pioneers. We are the pathfinders. We settled the New World, we built the modern world, and we changed history forever by embracing the eternal truth that everyone is made equal by the hand of Almighty God. (Applause.) America is the place where anything can happen. America is the place where anyone can rise. And here, on this land, on this soil, on this continent, the most incredible dreams come true. This nation is our canvas, and this country is our masterpiece. We look at tomorrow and see unlimited frontiers just waiting to be explored. Our brightest discoveries are not yet known. Our most thrilling stories are not yet told. Our grandest journeys are not yet made. The American Age, the American Epic, the American adventure has only just begun. Our spirit is still young, the sun is still rising, God’s grace is still shining, and, my fellow Americans, the best is yet to come. (Applause.) Thank you. God Bless You. And God Bless America. Thank you very much. (Ap']]



---
## Polarity trend for each President throughout their speeches.
---


```python
"""
* Calculating Polarity for each piece of text.
"""
polarity_transcript = []
for lp in list_pieces:
    polarity_piece = []
    for p in lp:
        polarity_piece.append(TextBlob(p).sentiment.polarity)
    polarity_transcript.append(polarity_piece)
```


```python
"""
* Plotting the above results.
"""
plt.rcParams['figure.figsize'] = [20, 12]
sns.set_style("ticks")
for index, president in enumerate(data_df.index):    
    plt.subplot(2, 4, index+1)
    plt.plot(polarity_transcript[index])
    plt.plot(np.arange(0,10), np.zeros(10))
    plt.title(data_df['full_name'][index], fontsize = 15)
    plt.ylim(ymin=-.15, ymax=.4)
plt.suptitle("Polarity trend for each President", fontsize = 20)
plt.savefig("Polarity_Trend.jpeg")
plt.show()
```


    
![png](output_40_0.png)
    


---
## Subjectivity trend for each President throughout their speeches.
---


```python
"""
* Calculating Subjectivity for each piece of text.
"""
subjectivity_transcript = []
for lp in list_pieces:
    subjectivity_piece = []
    for p in lp:
        subjectivity_piece.append(TextBlob(p).sentiment.subjectivity)
    subjectivity_transcript.append(subjectivity_piece)

```


```python
"""
* Plotting the above results.
"""
plt.rcParams['figure.figsize'] = [20, 12]
sns.set_style("ticks")
for index, president in enumerate(data_df.index):    
    plt.subplot(2, 4, index+1)
    plt.plot(subjectivity_transcript[index])
    plt.title(data_df['full_name'][index], fontsize = 15)
    plt.ylim(ymin=0.25, ymax=0.7)
plt.suptitle("Subjectivity trend for each President", fontsize = 20)
plt.savefig("Subjectivity_Trend.jpeg")
plt.show()
```


    
![png](output_43_0.png)
    



```python

```
