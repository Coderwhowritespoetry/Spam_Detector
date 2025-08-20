import pandas as pd
import joblib as jb 

data = pd.read_csv(r"C:\Users\91899\Downloads\fever_spam_dataset_large.csv")

data['Label'] = data['Label'].map({'Spam':1, 'Not Spam': 0})

from sklearn.feature_extraction.text import CountVectorizer
cv = CountVectorizer()

X = cv.fit_transform(data['Message'])
y = data['Label']

from sklearn.model_selection import train_test_split
X_train, x_test, y_train , y_test = train_test_split(X, y, random_state=1, test_size=0.2)

from sklearn.naive_bayes import MultinomialNB
mnb = MultinomialNB()
mnb.fit(X_train, y_train)

jb.dump(mnb, "spam_classifier.pkl")
jb.dump(cv, "vectorizer.pkl")