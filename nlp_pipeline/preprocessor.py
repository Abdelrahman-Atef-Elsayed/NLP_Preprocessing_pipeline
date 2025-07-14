import re
import string
import nltk
from typing import List, Union
import pandas as pd
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer, PorterStemmer

# تحميل الموارد المطلوبة من NLTK
nltk.download("punkt")
nltk.download("stopwords")
nltk.download("wordnet")

class TextPreprocessor:
    def __init__(self,
                 lowercase=True,
                 remove_punctuation=True,
                 remove_stopwords=True,
                 lemmatize=True,
                 stem=False,  # ← خيار جديد لتفعيل Stemming
                 language='english'):

        self.lowercase = lowercase
        self.remove_punctuation = remove_punctuation
        self.remove_stopwords = remove_stopwords
        self.lemmatize = lemmatize
        self.stem = stem  # ← حفظ الخيار
        self.language = language

        self.stop_words = set(stopwords.words(language))
        self.lemmatizer = WordNetLemmatizer()
        self.stemmer = PorterStemmer()  # ← إعداد الـ stemmer

    def _clean_text(self, text: str) -> str:
        if self.lowercase:
            text = text.lower()

        if self.remove_punctuation:
            text = text.translate(str.maketrans('', '', string.punctuation))

        tokens = nltk.word_tokenize(text)

        if self.remove_stopwords:
            tokens = [word for word in tokens if word not in self.stop_words]

        if self.lemmatize:
            tokens = [self.lemmatizer.lemmatize(word) for word in tokens]

        if self.stem:
            tokens = [self.stemmer.stem(word) for word in tokens]

        return ' '.join(tokens)

    def transform(self, texts: Union[List[str], pd.Series]) -> List[str]:
        return [self._clean_text(text) for text in texts]
