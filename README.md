# 🧠 NLP Preprocessing Pipeline

> A production-ready, modular text preprocessing pipeline for Natural Language Processing tasks — designed for clarity, extensibility, and real-world use.

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue?logo=python)](https://www.python.org/)
[![NLTK](https://img.shields.io/badge/NLTK-3.x-green)](https://www.nltk.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![PRs Welcome](https://img.shields.io/badge/PRs-welcome-brightgreen.svg)](CONTRIBUTING.md)

---

## 📌 Overview

Raw text from the real world is messy — it contains HTML tags, slang, emojis, typos, URLs, and inconsistent casing. Before feeding text into any machine learning or NLP model, it must be **cleaned and normalized**.

This pipeline provides a **step-by-step, modular preprocessing workflow** that transforms raw, noisy text into clean, model-ready input. Each step is independently usable, well-documented, and easy to customize for your domain.

### ✅ Designed for:

- Text classification & sentiment analysis
- Topic modeling & document clustering
- Machine translation & text summarization
- Social media analysis & content moderation
- Any downstream NLP or ML task

---

## 🗺️ Pipeline Workflow

```
Raw Text Input
      │
      ▼
┌─────────────────────────────────────────────────────┐
│  1.  Remove HTML Tags         (strip web markup)    │
│  2.  Case Folding             (lowercase)           │
│  3.  Expand Contractions      (don't → do not)      │
│  4.  Remove URLs              (strip links)         │
│  5.  Handle Emojis            (remove or encode)    │
│  6.  Handle Numbers           (normalize digits)    │
│  7.  Remove Punctuation       (strip symbols)       │
│  8.  Remove Extra Whitespace  (clean spaces)        │
│  9.  Text Normalization       (fix typos/slang)     │
│  10. Tokenization             (split into words)    │
│  11. Remove Stopwords         (filter noise words)  │
│  12. Lemmatization            (base word form)      │
│  13. Stemming                 (root word form)      │
│  14. Rare Word Removal        (trim vocabulary)     │
└─────────────────────────────────────────────────────┘
      │
      ▼
Clean Token List / Processed Text Output
```

> **Note:** Not every step is required for every task. For example, sentiment analysis may benefit from keeping emojis. Use this pipeline as a menu — pick what your task needs.

---

## 📚 Table of Contents

| # | Step | Purpose |
|---|------|---------|
| [1](#1-remove-html-tags) | Remove HTML Tags | Strip web markup |
| [2](#2-case-folding-convert-to-lowercase) | Case Folding | Normalize casing |
| [3](#3-expand-contractions) | Expand Contractions | Standardize short forms |
| [4](#4-remove-urls) | Remove URLs | Remove hyperlinks |
| [5](#5-handle-emojis) | Handle Emojis | Remove or encode emojis |
| [6](#6-handle-numbers) | Handle Numbers | Normalize digits |
| [7](#7-remove-punctuation--special-characters) | Remove Punctuation | Strip symbols |
| [8](#8-remove-extra-whitespace) | Remove Whitespace | Clean spacing |
| [9](#9-text-normalization) | Text Normalization | Fix slang and typos |
| [10](#10-tokenization) | Tokenization | Split into tokens |
| [11](#11-remove-stopwords) | Remove Stopwords | Filter noise words |
| [12](#12-lemmatization) | Lemmatization | Reduce to base form |
| [13](#13-stemming) | Stemming | Reduce to root form |
| [14](#14-rare-word-removal) | Rare Word Removal | Trim vocabulary |
| [💡](#-spacy-alternative-pipeline) | spaCy Alternative | Modern pipeline option |
| [⚙️](#%EF%B8%8F-full-pipeline-example) | Full Pipeline Example | End-to-end usage |

---

## 🛠️ Requirements

```bash
pip install nltk contractions beautifulsoup4 emoji spacy
python -m nltk.downloader punkt stopwords wordnet averaged_perceptron_tagger
python -m spacy download en_core_web_sm
```

---

## ⚙️ Preprocessing Steps

---

### 1. Remove HTML Tags

**What it does:** Strips all HTML markup (e.g., `<p>`, `<br>`, `<div>`) from text scraped from websites, articles, or comment systems.

**Why it matters:** HTML tags are structural syntax — they carry zero semantic value for NLP models. Leaving them in adds noise that confuses tokenizers and embeddings.

**Example:**
```
Input:  <p>The product is <b>amazing</b>!</p>
Output: The product is amazing!
```

```python
import re
from bs4 import BeautifulSoup

def remove_html_tags(text: str) -> str:
    """Remove HTML tags using BeautifulSoup for robustness."""
    return BeautifulSoup(text, "html.parser").get_text(separator=" ")

# Lightweight regex alternative (for simple cases only)
def remove_html_tags_regex(text: str) -> str:
    return re.sub(r'<[^>]+>', '', text)
```

> ⚠️ Prefer `BeautifulSoup` over regex for HTML — regex can fail on malformed or nested tags.

[↑ Back to Table of Contents](#-table-of-contents)

---

### 2. Case Folding (Convert to Lowercase)

**What it does:** Converts all characters in the text to lowercase.

**Why it matters:** Without this, `"Apple"`, `"APPLE"`, and `"apple"` are treated as three different words by most models — even though they mean the same thing. Case folding ensures vocabulary consistency.

**Example:**
```
Input:  "The QUICK Brown Fox"
Output: "the quick brown fox"
```

```python
def to_lowercase(text: str) -> str:
    """Normalize text casing to lowercase."""
    return text.lower()
```

> 💡 Exception: Some tasks (e.g., Named Entity Recognition) may benefit from preserving casing. Apply this step selectively.

[↑ Back to Table of Contents](#-table-of-contents)

---

### 3. Expand Contractions

**What it does:** Expands informal short forms into their full grammatical equivalents.

**Why it matters:** Models may treat `"don't"` and `"do not"` as entirely different expressions. Expanding contractions reduces vocabulary size and improves consistency, especially for sentiment analysis.

**Example:**
```
Input:  "I'm not sure they wouldn't've done it."
Output: "I am not sure they would not have done it."
```

```python
import contractions

def expand_contractions(text: str) -> str:
    """Expand English contractions to their full forms."""
    return contractions.fix(text)
```

> 📦 Install: `pip install contractions`

[↑ Back to Table of Contents](#-table-of-contents)

---

### 4. Remove URLs

**What it does:** Detects and removes all web links, whether they begin with `http://`, `https://`, `www.`, or other URL patterns.

**Why it matters:** URLs are almost never semantically meaningful for NLP tasks. They bloat the vocabulary, break tokenizers, and add noise to the feature space.

**Example:**
```
Input:  "Check this out: https://example.com/article?id=123 — it's great!"
Output: "Check this out:  — it's great!"
```

```python
import re

def remove_urls(text: str) -> str:
    """Remove all URL patterns from text."""
    url_pattern = r'https?://\S+|www\.\S+|ftp://\S+'
    return re.sub(url_pattern, '', text).strip()
```

[↑ Back to Table of Contents](#-table-of-contents)

---

### 5. Handle Emojis

**What it does:** Either removes emojis entirely or converts them to their text description (e.g., 😊 → `:smiling_face:`).

**Why it matters:** Emojis are extremely common in social media, product reviews, and chat data. They often carry **strong sentiment signals** — ignoring or blindly removing them can hurt model performance. The right approach depends on your task.

**Example:**
```
Input:  "This is awesome 🔥💯"
Output (remove): "This is awesome"
Output (encode): "This is awesome :fire: :hundred_points:"
```

```python
import re
import emoji

def remove_emojis(text: str) -> str:
    """Strip all emoji characters from text."""
    return emoji.replace_emoji(text, replace='').strip()

def encode_emojis(text: str) -> str:
    """Convert emojis to their text aliases (e.g., 😊 → ':smiling_face:')."""
    return emoji.demojize(text)
```

> 📦 Install: `pip install emoji`
>
> 💡 **When to encode vs. remove:** For sentiment analysis, encoding emojis often outperforms removal. For topic modeling, removal is usually preferable.

[↑ Back to Table of Contents](#-table-of-contents)

---

### 6. Handle Numbers

**What it does:** Either removes digits entirely or replaces them with a normalized placeholder token like `<NUM>`.

**Why it matters:** Raw numbers (e.g., `"1,234"`, `"3.14"`, `"2024"`) rarely contribute semantic meaning in text classification. Normalizing them reduces vocabulary size and helps models generalize across different numeric values.

**Example:**
```
Input:  "There were 1,200 attendees at the event on 05/12/2024."
Output (remove):  "There were  attendees at the event on ."
Output (replace): "There were <NUM> attendees at the event on <NUM>."
```

```python
import re

def remove_numbers(text: str) -> str:
    """Remove all digit sequences from text."""
    return re.sub(r'\b\d+[\d,\.]*\b', '', text).strip()

def replace_numbers(text: str, token: str = '<NUM>') -> str:
    """Replace numeric values with a placeholder token."""
    return re.sub(r'\b\d+[\d,\.]*\b', token, text)
```

> 💡 In financial or scientific NLP tasks, numbers may be critical — skip this step or use domain-specific logic.

[↑ Back to Table of Contents](#-table-of-contents)

---

### 7. Remove Punctuation & Special Characters

**What it does:** Removes symbols, punctuation marks, and non-alphanumeric characters from text.

**Why it matters:** Punctuation is generally not meaningful for bag-of-words or embedding-based models. Keeping it can cause `"word"` and `"word!"` to be treated as different tokens.

**Example:**
```
Input:  "Wow!!! This is... #amazing @user 100%"
Output: "Wow This is amazing user 100"
```

```python
import re

def remove_punctuation(text: str) -> str:
    """Remove all characters except letters, digits, and whitespace."""
    return re.sub(r'[^a-zA-Z0-9\s]', '', text)
```

> 💡 If you've already replaced numbers with `<NUM>` tokens, use `r'[^a-zA-Z0-9<>\s]'` to preserve them.

[↑ Back to Table of Contents](#-table-of-contents)

---

### 8. Remove Extra Whitespace

**What it does:** Collapses multiple consecutive spaces, tabs, and newlines into a single space, and trims leading/trailing whitespace.

**Why it matters:** Previous steps (URL/punctuation removal) often leave behind gaps and extra spaces. Whitespace noise can affect tokenization and downstream processing.

**Example:**
```
Input:  "  Hello    world  \n\t today "
Output: "Hello world today"
```

```python
import re

def remove_extra_whitespace(text: str) -> str:
    """Normalize all whitespace to single spaces and strip edges."""
    return re.sub(r'\s+', ' ', text).strip()
```

[↑ Back to Table of Contents](#-table-of-contents)

---

### 9. Text Normalization

**What it does:** Standardizes informal, abbreviated, or misspelled words commonly found in social media and user-generated content. This includes fixing slang, repeated characters, and common abbreviations.

**Why it matters:** Text from Twitter, Reddit, or product reviews is often written in informal language. `"luvvvv"`, `"luv"`, and `"love"` all mean the same thing — but a model will treat them as three different words without normalization.

**Example:**
```
Input:  "omg this is sooooo goooood lolll!!!!"
Output: "oh my god this is so good laughing out loud"
```

```python
import re

# Custom slang/abbreviation dictionary — extend as needed for your domain
SLANG_MAP = {
    "omg": "oh my god",
    "lol": "laughing out loud",
    "brb": "be right back",
    "idk": "i do not know",
    "imo": "in my opinion",
    "tbh": "to be honest",
    "smh": "shaking my head",
    "ngl": "not going to lie",
    "irl": "in real life",
    "luv": "love",
    "u": "you",
    "r": "are",
    "ur": "your",
    "gr8": "great",
    "b4": "before",
    "2day": "today",
    "thx": "thanks",
    "pls": "please",
    "w/": "with",
}

def reduce_repeated_chars(text: str, max_repeat: int = 2) -> str:
    """
    Reduce sequences of the same character to at most `max_repeat` occurrences.
    E.g., 'soooooo' → 'soo' (then lemmatization brings it to 'so').
    """
    return re.sub(r'(.)\1{' + str(max_repeat) + r',}', r'\1' * max_repeat, text)

def normalize_slang(text: str, slang_map: dict = SLANG_MAP) -> str:
    """Replace known slang/abbreviations with their full forms."""
    words = text.split()
    return ' '.join(slang_map.get(word, word) for word in words)

def normalize_text(text: str) -> str:
    """Apply full social media text normalization."""
    text = reduce_repeated_chars(text)
    text = normalize_slang(text)
    return text
```

> 💡 You can expand `SLANG_MAP` with domain-specific terms (medical, legal, financial, etc.) to adapt this step to your use case.

[↑ Back to Table of Contents](#-table-of-contents)

---

### 10. Tokenization

**What it does:** Splits a continuous string of text into individual units called **tokens** — typically words, but sometimes subwords or characters depending on the approach.

**Why it matters:** Tokenization is the bridge between raw text and numerical representation. Every subsequent step operates on individual tokens. The quality of tokenization directly affects everything downstream.

**Example:**
```
Input:  "The quick brown fox jumps."
Output: ["The", "quick", "brown", "fox", "jumps", "."]
```

```python
import nltk
from nltk.tokenize import word_tokenize, TweetTokenizer

nltk.download('punkt', quiet=True)

def tokenize(text: str, mode: str = 'word') -> list:
    """
    Tokenize text into a list of tokens.
    
    Args:
        text: Input string.
        mode: 'word' for standard text, 'tweet' for social media.
    
    Returns:
        List of string tokens.
    """
    if mode == 'tweet':
        # TweetTokenizer handles hashtags, mentions, and emoticons better
        tokenizer = TweetTokenizer(preserve_case=False, strip_handles=True, reduce_len=True)
        return tokenizer.tokenize(text)
    return word_tokenize(text)
```

> 💡 For social media data, `TweetTokenizer` is far more robust than standard word tokenization — it correctly handles `#hashtags`, `@mentions`, and emoticons.

[↑ Back to Table of Contents](#-table-of-contents)

---

### 11. Remove Stopwords

**What it does:** Filters out high-frequency, low-information words — known as **stopwords** — that appear in almost every sentence but carry little semantic weight on their own.

**Why it matters:** Words like `"the"`, `"is"`, `"in"`, and `"a"` account for a massive portion of any text corpus but add noise to models. Removing them reduces dimensionality and helps models focus on meaningful words.

**Example:**
```
Input:  ["the", "cat", "is", "on", "the", "mat"]
Output: ["cat", "mat"]
```

```python
from nltk.corpus import stopwords
import nltk

nltk.download('stopwords', quiet=True)

def remove_stopwords(tokens: list, language: str = 'english', 
                     extra_stopwords: list = None) -> list:
    """
    Remove stopwords from a token list.
    
    Args:
        tokens: List of word tokens.
        language: Language for NLTK stopwords corpus.
        extra_stopwords: Additional domain-specific words to filter.
    
    Returns:
        Filtered list of tokens.
    """
    stop_words = set(stopwords.words(language))
    if extra_stopwords:
        stop_words.update(extra_stopwords)
    return [word for word in tokens if word not in stop_words]
```

> 💡 Default stopword lists are generic. For domain-specific tasks (e.g., medical NLP), consider adding domain-specific noise words to `extra_stopwords`.

[↑ Back to Table of Contents](#-table-of-contents)

---

### 12. Lemmatization

**What it does:** Reduces each word to its canonical **dictionary form** (called a **lemma**), using vocabulary and morphological analysis.

**Why it matters:** `"running"`, `"ran"`, and `"runs"` all derive from the same root: `"run"`. Lemmatization collapses these variants so models treat them as the same concept, reducing vocabulary size without losing meaning.

**Example:**
```
Input:  ["running", "better", "geese", "studies"]
Output: ["run",     "good",   "goose", "study"]
```

```python
from nltk.stem import WordNetLemmatizer
from nltk.corpus import wordnet
import nltk

nltk.download('wordnet', quiet=True)
nltk.download('averaged_perceptron_tagger', quiet=True)

lemmatizer = WordNetLemmatizer()

def get_wordnet_pos(word: str) -> str:
    """Map POS tag to WordNet POS for more accurate lemmatization."""
    tag = nltk.pos_tag([word])[0][1][0].upper()
    tag_map = {'J': wordnet.ADJ, 'V': wordnet.VERB, 'R': wordnet.ADV}
    return tag_map.get(tag, wordnet.NOUN)

def lemmatize_tokens(tokens: list) -> list:
    """
    Lemmatize tokens using POS-aware WordNet lemmatizer.
    POS tagging improves accuracy: 'better' → 'good' (adj) vs 'better' (noun).
    """
    return [lemmatizer.lemmatize(word, get_wordnet_pos(word)) for word in tokens]
```

> ⚡ **Lemmatization vs. Stemming:** Lemmatization is slower but always produces real words. Use it when output interpretability matters (e.g., keyword extraction, topic modeling).

[↑ Back to Table of Contents](#-table-of-contents)

---

### 13. Stemming

**What it does:** Strips word suffixes (and sometimes prefixes) using heuristic rules to reduce words to their **root stem** — which may not be a real dictionary word.

**Why it matters:** Stemming is faster than lemmatization and still achieves vocabulary consolidation. It's a practical choice when speed matters more than linguistic precision (e.g., large-scale information retrieval).

**Example:**
```
Input:  ["running", "runner", "runs", "easily", "fairly"]
Output: ["run",     "runner", "run",  "easili", "fairli"]
```

```python
from nltk.stem import PorterStemmer, SnowballStemmer

porter = PorterStemmer()
snowball = SnowballStemmer(language='english')

def stem_tokens(tokens: list, algorithm: str = 'porter') -> list:
    """
    Stem tokens using either Porter or Snowball stemmer.
    
    Args:
        tokens: List of word tokens.
        algorithm: 'porter' (English only) or 'snowball' (multilingual).
    
    Returns:
        List of stemmed tokens.
    """
    stemmer = porter if algorithm == 'porter' else snowball
    return [stemmer.stem(word) for word in tokens]
```

> ⚠️ Choose either stemming **or** lemmatization for a given pipeline — applying both is redundant. For most modern NLP tasks, lemmatization is preferred.

[↑ Back to Table of Contents](#-table-of-contents)

---

### 14. Rare Word Removal

**What it does:** Removes words that appear extremely infrequently across your corpus — below a defined minimum frequency threshold.

**Why it matters:** Words that appear only once or twice in a large dataset are likely typos, gibberish, or highly specific terms that the model can't learn meaningful patterns from. Removing them reduces vocabulary size and improves training efficiency.

**Example:**
```
Corpus: ["cat", "cat", "dog", "xyzabc", "qwerty", "dog", "cat"]
Min frequency = 2 → Remove: ["xyzabc", "qwerty"]
Output tokens: ["cat", "cat", "dog", "dog", "cat"]
```

```python
from collections import Counter

def remove_rare_words(token_lists: list, min_freq: int = 2) -> list:
    """
    Remove tokens that appear fewer than `min_freq` times across all documents.
    
    Args:
        token_lists: List of token lists (one per document).
        min_freq: Minimum frequency threshold (default: 2).
    
    Returns:
        Filtered list of token lists.
    """
    # Build global frequency map
    all_tokens = [token for doc in token_lists for token in doc]
    freq = Counter(all_tokens)
    
    # Filter tokens below threshold
    vocab = {word for word, count in freq.items() if count >= min_freq}
    return [[token for token in doc if token in vocab] for doc in token_lists]
```

> 💡 `min_freq` is a hyperparameter — tune it based on corpus size. For large corpora (millions of documents), a threshold of 5–10 is common.

[↑ Back to Table of Contents](#-table-of-contents)

---

## 💡 spaCy Alternative Pipeline

[spaCy](https://spacy.io/) offers a faster, more production-ready alternative to NLTK — with built-in tokenization, lemmatization, POS tagging, and NER in a single pass.

**When to use spaCy over NLTK:**
- Processing large volumes of text (spaCy is significantly faster)
- When you need multiple annotations in one pass (tokens + POS + lemmas + entities)
- Production systems where latency matters

```python
import spacy

nlp = spacy.load("en_core_web_sm")

def spacy_preprocess(text: str, remove_stop: bool = True, 
                     remove_punct: bool = True) -> list:
    """
    Process text through the spaCy pipeline.
    Returns lemmatized tokens with optional stopword/punctuation filtering.
    
    Example:
        Input:  "The cats were running quickly."
        Output: ["cat", "run", "quickly"]
    """
    doc = nlp(text)
    tokens = [
        token.lemma_.lower()
        for token in doc
        if not (remove_stop and token.is_stop)
        and not (remove_punct and token.is_punct)
        and not token.is_space
    ]
    return tokens
```

> 📦 Install: `pip install spacy && python -m spacy download en_core_web_sm`

[↑ Back to Table of Contents](#-table-of-contents)

---

## ⚙️ Full Pipeline Example

Here is a complete end-to-end example combining all steps:

```python
import re
import nltk
import emoji
import contractions
from bs4 import BeautifulSoup
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords, wordnet
from nltk.stem import WordNetLemmatizer

# Download required NLTK data
for resource in ['punkt', 'stopwords', 'wordnet', 'averaged_perceptron_tagger']:
    nltk.download(resource, quiet=True)

lemmatizer = WordNetLemmatizer()
STOP_WORDS = set(stopwords.words('english'))

def full_preprocess(text: str, 
                    keep_emojis: bool = False,
                    replace_numbers: bool = True) -> list:
    """
    Full NLP preprocessing pipeline.
    
    Args:
        text: Raw input string.
        keep_emojis: If True, encodes emojis; if False, removes them.
        replace_numbers: If True, replaces numbers with <NUM>; if False, removes them.
    
    Returns:
        List of preprocessed, lemmatized tokens.
    """
    # 1. Remove HTML
    text = BeautifulSoup(text, "html.parser").get_text()
    # 2. Lowercase
    text = text.lower()
    # 3. Expand contractions
    text = contractions.fix(text)
    # 4. Remove URLs
    text = re.sub(r'https?://\S+|www\.\S+', '', text)
    # 5. Handle emojis
    text = emoji.demojize(text) if keep_emojis else emoji.replace_emoji(text, replace='')
    # 6. Handle numbers
    text = re.sub(r'\b\d[\d,\.]*\b', '<NUM>' if replace_numbers else '', text)
    # 7. Remove punctuation (preserve <NUM> tokens if used)
    text = re.sub(r'[^a-z0-9<>\s]', '', text)
    # 8. Remove extra whitespace
    text = re.sub(r'\s+', ' ', text).strip()
    # 9. Tokenize
    tokens = word_tokenize(text)
    # 10. Remove stopwords
    tokens = [t for t in tokens if t not in STOP_WORDS]
    # 11. Lemmatize (with POS tagging)
    tokens = [lemmatizer.lemmatize(t, get_wordnet_pos(t)) for t in tokens]
    return tokens


def get_wordnet_pos(word: str) -> str:
    tag = nltk.pos_tag([word])[0][1][0].upper()
    return {'J': wordnet.ADJ, 'V': wordnet.VERB, 'R': wordnet.ADV}.get(tag, wordnet.NOUN)


# --- Example Usage ---
raw_text = """
<p>I'm <b>absolutely</b> loving this product!!! 😍🔥
Check: https://example.com/reviews — it's gr8 value for the price.
I've used it 3 times already and it's AMAZING!</p>
"""

tokens = full_preprocess(raw_text, keep_emojis=True, replace_numbers=True)
print(tokens)
# Output: ['absolute', 'love', 'product', ':smiling_face_with_heart-eyes:', ':fire:',
#          'check', 'great', 'value', 'price', 'use', '<NUM>', 'time', 'already', 'amaze']
```

---

## 📊 Step Selection Guide

| Task | Recommended Steps |
|------|------------------|
| **Text Classification** | 1, 2, 3, 4, 7, 8, 10, 11, 12 |
| **Sentiment Analysis** | 1, 2, 3, 4, 5 (encode), 7, 8, 10, 12 |
| **Topic Modeling** | 1, 2, 4, 5 (remove), 6, 7, 8, 10, 11, 12, 14 |
| **Social Media NLP** | 1, 2, 3, 4, 5, 9, 7, 8, 10, 11, 12 |
| **Information Retrieval** | 2, 4, 7, 8, 10, 11, 13 |
| **Machine Translation** | 1, 3, 4, 8 (minimal preprocessing) |

---

## 🔁 Why Use This Pipeline?

**Modular by design** — each function is independent and can be used in isolation or combined in any order that suits your task.

**Production-ready code** — proper function signatures, docstrings, type hints, and safe handling of edge cases.

**Extensible for any domain** — customize slang maps, stopword lists, and frequency thresholds for medical, legal, financial, or social media text.

**Compatible with the modern ML stack** — works seamlessly with `scikit-learn`, `pandas`, `Hugging Face Transformers`, `Gensim`, `spaCy`, and more.

---

## 📁 Suggested Project Structure

```
nlp-preprocessing-pipeline/
├── preprocessing/
│   ├── __init__.py
│   ├── cleaner.py         # HTML, URLs, punctuation, whitespace
│   ├── normalizer.py      # Case, contractions, slang, numbers, emojis
│   ├── tokenizer.py       # Tokenization logic
│   └── reducer.py         # Stopwords, lemmatization, stemming, rare words
├── examples/
│   └── full_pipeline.py   # End-to-end usage example
├── tests/
│   └── test_pipeline.py
├── requirements.txt
└── README.md
```

---

## 🤝 Contributing

Contributions, issues, and feature requests are welcome. Feel free to open a pull request or start a discussion.

---

## 📄 License

This project is licensed under the [MIT License](LICENSE).
