import numpy as np  # linear algebra
import pandas as pd  # data processing, CSV file I/O (e.g. pd.read_csv)

df = pd.read_csv(r"C:\Users\DELL LATITUDE\Desktop\Pycharm_projects\DATA\emails.csv")

df.shape

df = df.iloc[:, 0:2]
df.isnull().sum()

# Dropping the 2 rows which has null values
df.dropna(inplace=True)
df.duplicated().sum()

# There are 33 duplicate rows, droping the duplicate rows
df.drop_duplicates(inplace=True)
df.info()
# Removing rows whose spam columns are text
df["spam"] = df['spam'].apply(lambda x: "text_result" if len(x) > 1 else x)
df = df[df["spam"] != "text_result"]

# Converting the data type of spam column from object type to int
df['spam'] = df['spam'].astype('int')
df.info()
df.iloc[15].text
df.columns
df.head(10)

# Convert all to small alphabet letters
df['text'] = df['text'].apply(lambda x: x.lower())


# Creating a function to Remove special characters from the data
def rem_special_chars(text):
    new_text = ""
    for i in text:
        if i.isalnum() or i == " ":
            new_text += i
    return new_text.strip()


# Removing special characters from the data
df['text'] = df['text'].apply(rem_special_chars)

# Checking if the special characters are removed or not.
text = "subject: naturally irresistible your corporate identity  lt is really hard to recollect a company : the  market is full of suqgestions and the information isoverwhelminq ; but a good  catchy logo , stylish statlonery and outstanding website  will make the task much easier .  we do not promise that havinq ordered a iogo your  company will automaticaily become a world ieader : it isguite ciear that  without good products , effective business organization and practicable aim it  will be hotat nowadays market ; but we do promise that your marketing efforts  will become much more effective . here is the list of clear  benefits : creativeness : hand - made , original logos , specially done  to reflect your distinctive company image . convenience : logo and stationery  are provided in all formats ; easy - to - use content management system letsyou  change your website content and even its structure . promptness : you  will see logo drafts within three business days . affordability : your  marketing break - through shouldn ' t make gaps in your budget . 100 % satisfaction  guaranteed : we provide unlimited amount of changes with no extra fees for you to  be surethat you will love the result of this collaboration . have a look at our  portfolio _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ not interested . . . _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _"
rem_special_chars(text)

from sklearn.feature_extraction.text import CountVectorizer

cv = CountVectorizer(stop_words='english', max_features=10000)
X = cv.fit_transform(df['text']).toarray()
y = df["spam"].values

# Splitting the data into training set and test set
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Using a Naive Bayes model for our Classifier
from sklearn.naive_bayes import MultinomialNB
# Defining our model
clf = MultinomialNB()
clf.fit(X_train, y_train)

from sklearn.metrics import accuracy_score
y_pred = clf.predict(X_test)
# Checking the accuracy
accuracy_score(y_test, y_pred)
# Length of the data that we have used for training our model
len(cv.get_feature_names())
# Length of stopwords(eg: a, an, the, etc.)
len(cv.get_stop_words())

import pickle
# Saving the models and the Count Vectorizer converter in a pkl file so that it can
# be used in another program without trainig the models.
pickle.dump(cv,open('C:/Users/DELL LATITUDE/Desktop/Pycharm_projects/NaiveBayes/cv.pkl','wb'))
pickle.dump(clf,open('C:/Users/DELL LATITUDE/Desktop/Pycharm_projects/NaiveBayes/clf.pkl','wb'))

