##import nltk
import numpy as np
import pandas as pd
from nltk.stem.porter import PorterStemmer
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from collections import Counter
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import re
import string
import tensorflow
from tensorflow.keras import layers
from keras.callbacks import ModelCheckpoint, EarlyStopping
from sklearn import preprocessing
import tensorflow
from tensorflow.keras import layers
from keras.callbacks import ModelCheckpoint, EarlyStopping
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
from sklearn import metrics
import os
import pickle

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

    def preprocess(self):
        self.get_data()
        self.df["Message"] = self.df.Message.map(self.proc)

        for col in self.categorical_columns:
            self.encoders[col] = preprocessing.LabelEncoder()
            self.df[col] = self.encoders[col].fit_transform(self.df[col])
        self.MAX_NB_WORDS = 3000
        self.MAX_SEQUENCE_LENGTH = 300
        self.EMBEDDING_DIM = 100
        tokenizer = Tokenizer(
            num_words=self.MAX_NB_WORDS,
            filters='!"#$%&()*+,-./:;<=>?@[\]^_`{|}~',
            lower=True,
        )
        tokenizer.fit_on_texts(self.df["Message"].values)
        with open(self.cd + "\\tokenizer.pickle", "wb") as handle:
            pickle.dump(tokenizer, handle, protocol=pickle.HIGHEST_PROTOCOL)
        word_index = tokenizer.word_index
        self.X = tokenizer.texts_to_sequences(self.df["Message"].values)
        self.X = pad_sequences(self.X, maxlen=self.MAX_SEQUENCE_LENGTH)
        Y = pd.get_dummies(self.df["Category"]).values
        self.X_train, self.X_test, self.Y_train, self.Y_test = train_test_split(
            self.X, Y, test_size=0.25, random_state=42
        )
        self.Y_test_true = []
        for i in range(0, len(self.Y_test)):
            if self.Y_test[i][0] == 0:
                self.Y_test_true.append(1)
            else:
                self.Y_test_true.append(0)

        print("Data Preprocessing Done")

    def train(self):
        self.preprocess()
        self.model.add(
            layers.Embedding(
                self.MAX_NB_WORDS, self.EMBEDDING_DIM, input_length=self.X.shape[1]
            )
        )
        self.model.add(layers.SpatialDropout1D(0.2))
        self.model.add(layers.LSTM(100, dropout=0.2, recurrent_dropout=0.2))
        self.model.add(layers.Dense(2, activation="softmax"))
        self.model.compile(
            loss="binary_crossentropy", optimizer="adam", metrics=["accuracy"]
        )

        epochs = 5
        batch_size = 64

        self.history = self.model.fit(
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
        predictions = self.model.predict(self.X_test)
        eval_results = []
        for i in range(0, len(predictions)):
            eval_results.append(int(np.argmax(predictions[i])))
        target_names = ["ham", "spam"]
        print(
            classification_report(
                self.Y_test_true, eval_results, target_names=target_names
            )
        )
        cm = confusion_matrix(self.Y_test_true, eval_results)
        cm_df = pd.DataFrame(cm, index=target_names, columns=target_names)
        print(cm_df)
        plt.title("Loss")
        plt.plot(self.history.history["loss"], label="train")
        plt.plot(self.history.history["val_loss"], label="test")
        plt.legend()
        print(plt.show())
        plt.title("Accuracy")
        plt.plot(self.history.history["accuracy"], label="train")
        plt.plot(self.history.history["val_accuracy"], label="test")
        plt.legend()
        print(plt.show())
        print("AUC:", metrics.roc_auc_score(self.Y_test_true, eval_results))
        cutoff_grid = np.linspace(0.0, 1.0, 100)
        TPR = []
        FPR = []
        cutoff_grid
        FPR, TPR, cutoffs = metrics.roc_curve(
            self.Y_test_true, eval_results, pos_label=1
        )
        plt.plot(FPR, TPR, c="red", linewidth=1.0)
        plt.xlabel("False Positive")
        plt.ylabel("True Positive")
        plt.title("ROC Curve")
        print(plt.show())
        fpr_en, tpr_en, thresholds_en = metrics.roc_curve(
            self.Y_test_true, eval_results
        )
        roc_auc_en = metrics.auc(fpr_en, tpr_en)
        precision_en, recall_en, th_en = metrics.precision_recall_curve(
            self.Y_test_true, eval_results
        )
        plt.plot([1, 0], [0, 1], "k--")
        plt.plot(fpr_en, tpr_en, label="rnn (area = %0.3f)" % roc_auc_en)
        plt.plot(recall_en, precision_en, label="Recall rnn")
        plt.title("Precision vs. Recall")
        plt.xlabel("Recall")
        plt.ylabel("Precision")
        plt.legend(loc="best")
        print(plt.show())
        self.model.save(self.cd + "\\Final Model.h5")
        print("Done")

    def test(self, message):
        from keras.models import load_model

        loadedModel = load_model(self.cd + "\\Final Model.h5")
        testText = str(message)
        testText = self.proc(testText)
        testText = [testText]
        with open(self.cd + "\\tokenizer.pickle", "rb") as handle:
            tokenizer = pickle.load(handle)
        testText = tokenizer.texts_to_sequences(testText)
        testText = pad_sequences(testText, maxlen=300)
        pred = loadedModel.predict(testText)
        eval_results = []
        for i in range(0, len(pred)):
            eval_results.append(int(np.argmax(pred[i])))
        if eval_results[0] == 1:
            return "Spam"
        elif eval_results[0] == 0:
            return "Ham"
