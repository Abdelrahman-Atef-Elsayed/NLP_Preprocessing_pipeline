# 🧾 NLP Preprocessing Pipeline – README

## 📌 Purpose  
This pipeline is designed as a reusable template for text preprocessing in NLP tasks such as:

- Text classification  
- Sentiment analysis  
- Topic modeling  
- Machine translation  


---

## 📚 Table of Contents

- [1 - Remove HTML Tags](#1-remove-html-tags)
- [2 - Convert to Lowercase](#2-convert-to-lowercase)
- [3 - Expand Contractions](#3-expand-contractions)
- [4 - Remove URLs](#4-remove-urls)
- [5 - Remove Punctuation and Special Characters](#5-remove-punctuation--special-characters)
- [6 - Remove Extra Whitespace](#6-remove-extra-whitespace)
- [7 - Tokenization](#7-tokenization)
- [8 - Remove Stopwords](#8-remove-stopwords)
- [9 - Lemmatization](#9-lemmatization)
- [10 - Stemming](#10-stemming)

---

## ⚙️ Preprocessing Steps

### 1. Remove HTML Tags  
When dealing with data from the web (articles, comments, etc.), HTML tags should be removed.  
**Example:**  
```html
<p>Hello</p> → Hello
```
Use Code:
```python
re.sub(r'<.*?>', '', text)
```

[Back To The Table of Contents](#-table-of-contents)


### 2. Convert to Lowercase
Ensures consistency by treating `Apple` and `apple` as the same word.

Use Code:
```python
text.lower()
```

[Back To The Table of Contents](#-table-of-contents)


### 3. Expand Contractions
Expands shortened forms to their original structure.
**Example:**  
`don't → do not` , `I'm → I am`

Use Code:
```python
contractions.fix(text)
```

[Back To The Table of Contents](#-table-of-contents)


### 4. Remove URLs
Removes unnecessary web links from text.

Use Code:
```python
re.sub(r'http\\S+|www\\S+|https\\S+', '', text)
```

[Back To The Table of Contents](#-table-of-contents)


### 5. Remove Punctuation & Special Characters
Removes noise such as symbols and punctuation marks.
**Example:**  
`Hello!!! → Hello`

Use Code:
```python
re.sub(r'[^a-zA-Z0-9\\s]', '', text)
```

[Back To The Table of Contents](#-table-of-contents)

### 6. Remove Extra Whitespace
Trims down extra spaces.
**Example:**  
`Hello world " → "Hello world`

Use Code:
```python
' '.join(text.split())
```

[Back To The Table of Contents](#-table-of-contents)

### 7. Tokenization
Splits text into individual words or tokens.
**Example:**  
`"This is a sentence" → ["This", "is", "a", "sentence"]`

Use Code:
```python
from nltk.tokenize import word_tokenize  
word_tokenize(text)
```

[Back To The Table of Contents](#-table-of-contents)

### 8. Remove Stopwords
Filters out common words that carry little meaning like `"the"`, `"is"`, `"in"`, `etc`.

Use Code:
```python
from nltk.corpus import stopwords  
stop_words = set(stopwords.words('english'))  
[word for word in tokens if word not in stop_words]
```

[Back To The Table of Contents](#-table-of-contents)

### 9. Lemmatization
Reduces words to their base or dictionary form.

**Example:**  
`running, ran, runs → run`

Use Code:
```python
from nltk.stem import WordNetLemmatizer  
lemmatizer = WordNetLemmatizer()  
lemmatizer.lemmatize(word)
```

[Back To The Table of Contents](#-table-of-contents)

### 10. Stemming

Cuts off word endings to reduce them to their stem/root form. Less accurate than lemmatization but faster.

**Example:**  
`running, runner, runs → run`

Use Code:
```python
from nltk.stem import PorterStemmer  
stemmer = PorterStemmer()  
stemmer.stem(word)
```

[Back To The Table of Contents](#-table-of-contents)
---

## ✅ Why Use This Pipeline?

### 🔁 Reusable and ready for copy/paste into your NLP projects.

### 🛠️ Easily customizable for other languages or specialized domains.

### 🤖 Compatible with libraries like `scikit-learn`, `pandas`, `spaCy`, and more.






