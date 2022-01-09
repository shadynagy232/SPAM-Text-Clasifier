##import nltk
import pandas as pd
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import re
import string
import tensorflow
from tensorflow.keras import layers
from keras.callbacks import EarlyStopping
from sklearn.model_selection import train_test_split
import os
import pickle
from collections import Counter

##nltk.download("wordnet")
##nltk.download("stopwords")
##nltk.download("punkt")
##nltk.download("omw-1.4")


class TextClassification:
    def __init__(self):

        self.cd = os.getcwd()

        self.model = tensorflow.keras.models.Sequential()

    def get_data(self):
        self.df = pd.read_csv(self.cd + "\\data.csv")
        self.df.drop_duplicates(inplace=True)
        print("Data Scoring Done")

    def proc(self, text):
        text = text.lower()
        text = re.sub(r"https?://\S+|www.\S+", "", text)
        text = re.sub(r"<.*?>", "", text)
        text = re.sub(r"[^a-zA-Z]+", " ", text)
        text = re.sub(r"[0-9]", "", text)
        translator = str.maketrans("", "", string.punctuation)
        text = text.translate(translator)
        words = word_tokenize(text)
        words = [word for word in words if word not in stopwords.words("english")]
        lemmatizer = WordNetLemmatizer()
        words = [lemmatizer.lemmatize(word) for word in words]
        text = " ".join(words)
        return text

    def counter_word(self, text_col):
        count = Counter()
        for text in text_col.values:
            for word in text.split():
                count[word] += 1
        return count

    def preprocess(self):
        self.get_data()
        self.df["Message"] = self.df.Message.map(self.proc)
        self.maxSeq = len(self.df["Message"][0])
        for i in range(0, len(self.df["Message"])):
            try:
                cur = len(self.df["Message"][i])
                if cur > self.maxSeq:
                    self.maxSeq = cur
            except:
                pass
        counter = self.counter_word(self.df.Message)
        self.MAX_NB_WORDS = len(counter)
        self.MAX_SEQUENCE_LENGTH = self.maxSeq
        self.EMBEDDING_DIM = 100
        self.tokenizer = Tokenizer(num_words=self.MAX_NB_WORDS)
        self.tokenizer.fit_on_texts(self.df["Message"].values)
        with open(self.cd + "\\tokenizer.pickle", "wb") as handle:
            pickle.dump(self.tokenizer, handle, protocol=pickle.HIGHEST_PROTOCOL)
        self.X = self.tokenizer.texts_to_sequences(self.df["Message"].values)
        self.X = pad_sequences(self.X, maxlen=self.MAX_SEQUENCE_LENGTH)
        from sklearn import preprocessing

        labelencoder = preprocessing.LabelEncoder()
        self.df["Category"] = labelencoder.fit_transform(self.df["Category"])
        Y = self.df["Category"].values
        self.X_train, self.X_test, self.Y_train, self.Y_test = train_test_split(
            self.X, Y, test_size=0.25, random_state=42
        )

        print("Data Preprocessing Done")

    def train(self):
        self.preprocess()
        self.model.add(
            layers.Embedding(
                self.MAX_NB_WORDS, self.EMBEDDING_DIM, input_length=self.X.shape[1]
            )
        )
        self.model.add(layers.LSTM(32))
        self.model.add(layers.Dense(1, activation="sigmoid"))
        self.model.compile(
            loss="binary_crossentropy", optimizer="adam", metrics=["accuracy"]
        )

        epochs = 5
        batch_size = 64

        self.model.fit(
            self.X_train,
            self.Y_train,
            epochs=epochs,
            batch_size=batch_size,
            validation_split=0.2,
            callbacks=[EarlyStopping(monitor="val_loss", patience=3, min_delta=0.0001)],
        )

        print("Model Training Done")

    def evaluate(self):
        self.train()
        threshold = 0.5
        result = self.model.predict(self.X_test, verbose=2)
        result = result > threshold
        result = result.astype(int)
        from sklearn.metrics import classification_report

        target_names = ["ham", "spam"]
        print(classification_report(self.Y_test, result, target_names=target_names))
        self.model.save(self.cd + "\\Final Model.h5")
        print("Done")

    def test(self, messsage):
        from keras.models import load_model

        loadedModel = load_model(self.cd + "\\Final Model.h5")
        testText = str(messsage)
        testText = self.proc(testText)
        testText = [testText]
        with open(self.cd + "\\tokenizer.pickle", "rb") as handle:
            tokenizer = pickle.load(handle)
        testText = tokenizer.texts_to_sequences(testText)
        testText = pad_sequences(testText, maxlen=442)
        threshold = 0.5
        result = loadedModel.predict(testText, verbose=2)
        result = result > threshold
        result = result.astype(int)
        if result[0] == 1:
            print("Spam")
        elif result[0] == 0:
            print("Ham")
