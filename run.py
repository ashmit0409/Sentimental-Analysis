import nltk
nltk.download('vader_lexicon')
nltk.download('stopwords')
nltk.download('punkt')       #individual text into sentencesimport pandas as pd
import numpy as np
import nltk
from nltk import tokenize
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import re
import string
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
file1='/content/Newdata.xlsx'

df = pd.read_excel(file1)
df=df.reset_index()

df=df.dropna(subset=['comment']).reset_index(drop=True)
print(df)
comm = df['comment'].tolist()
newre = pd.DataFrame({
    'Comment':[],
    'neg':[],
    'neu':[],
    'pos':[],
    'compound':[],#score
    'sentiment':[]
               })



for sentence in comm:
    sid = SentimentIntensityAnalyzer()
    ss = sid.polarity_scores(str(sentence))
    if(ss['compound']>0.05):
        sen='Positive'
    elif((ss['compound']<0.05) and (ss['compound']>-0.05)):
        sen='Neutral'
    else:
        sen='Negative'

    new_row = pd.DataFrame({
    'Comment':[str(sentence)],
    'neg':[ss['neg']],
    'neu':[ss['neu']],
    'pos':[ss['pos']],
    'compound':[ss['compound']],
    'sentiment':[sen],
               })
     newre=pd.concat([newre,new_row],ignore_index=True)
df.head()
df.drop('comment', inplace=True, axis=1)

result=df.join(newre)


result.style\
.highlight_between(left=-1.0,right=-0.1,subset='compound',color='Red')\
.highlight_between(left=0.0,right=0.0,subset='compound',color='Orange')\
.highlight_between(left=0.1,right=1.0,subset='compound',color='LightBlue')

stop_words = stopwords.words()

def cleaning(text):
    text = text.lower()
    text = re.sub('https?://\S+|www\.\S+', '', text)
    text = re.sub('<.*?>+', '', text)
    text = re.sub('[%s]' % re.escape(string.punctuation), '', text)
    text = re.sub('\n', '', text)
    text = re.sub('[’“”…]', '', text)
    # removing the emojies
    emoji_pattern = re.compile("["
                           u"\U0001F600-\U0001F64F"  # emoticons
                           u"\U0001F300-\U0001F5FF"  # symbols & pictographs
                           u"\U0001F680-\U0001F6FF"  # transport & map symbols
                           u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
                           u"\U00002702-\U000027B0"
                           u"\U000024C2-\U0001F251"
                           "]+", flags=re.UNICODE)
    text = emoji_pattern.sub(r'', text)
    text_tokens = word_tokenize(text)
    tokens_without_sw = [word for word in text_tokens if not word in stop_words]
    filtered_sentence = (" ").join(tokens_without_sw)
    text = filtered_sentence
    return text

dt = result['Comment'].apply(cleaning)

# Most Negative Comment :
maxi=result['compound'].max()
mini=result['compound'].min()
# like=result['likesCount'].max()
for i,row in result.iterrows():
    if(result['compound'][i]==mini):
        print("\n\nMost Negative Comment with Sentiment Score ",
              mini,"is : \n\n",result['Comment'][i])

# Most Positive Comment :
for i,row in result.iterrows():
    if(result['compound'][i]==maxi):
        print("\n\nMost Positive Comment with Sentiment Score ",
              maxi," is : \n \n",result['Comment'][i])

result.to_csv("/content/result.csv")