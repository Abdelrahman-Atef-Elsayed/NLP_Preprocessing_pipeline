# 🧠 NLP Preprocessing & Feature Extraction — A to Z

> A production-ready, modular text preprocessing and feature extraction pipeline for Natural Language Processing — designed for clarity, extensibility, and real-world use.

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.8%2B-blue?logo=python" />
  &nbsp;
  <img src="https://img.shields.io/badge/NLTK-3.x-green" />
  &nbsp;
  <img src="https://img.shields.io/badge/spaCy-3.x-09a3d5" />
  &nbsp;
  <img src="https://img.shields.io/badge/Gensim-4.x-orange" />
  &nbsp;
  <img src="https://img.shields.io/badge/HuggingFace-Transformers-yellow" />
  &nbsp;
  <img src="https://img.shields.io/badge/License-MIT-yellow.svg" />
  &nbsp;
  <img src="https://img.shields.io/badge/PRs-welcome-brightgreen.svg" />
</p>

---

## 📌 Overview

Raw text from the real world is messy — it contains HTML tags, slang, emojis, typos, URLs, and inconsistent casing. Before feeding text into any machine learning or NLP model, it must be **cleaned, normalized, and transformed into a numerical representation**.

This pipeline provides a **comprehensive, step-by-step, modular workflow** that takes raw, noisy text through three major stages — cleaning, preprocessing, and feature extraction — turning it into model-ready input. Each function is independently usable, well-documented, and easy to customize for your domain.

### ✅ Designed For

- Text classification & sentiment analysis
- Topic modeling & document clustering
- Machine translation & text summarization
- Social media analysis & content moderation
- Information retrieval & semantic search
- Any downstream NLP or ML task

---

## 🗺️ Full Pipeline Workflow

```
Raw Text Input
      │
      ▼
┌──────────────────────────────────────────────────────────────────┐
│  🧹 STAGE 1 — TEXT CLEANING                                      │
│                                                                  │
│   1.  Remove HTML Tags              (strip web markup)           │
│   2.  Case Folding                  (lowercase)                  │
│   3.  Expand Contractions           (don't → do not)             │
│   4.  Remove URLs                   (strip links)                │
│   5.  Remove Non-ASCII Characters   (strip encoding noise)       │
│   6.  Handle Emojis                 (remove or encode)           │
│   7.  Handle Numbers                (normalize digits)           │
│   8.  Remove Punctuation            (strip symbols)              │
│   9.  Remove Extra Whitespace       (clean spaces)               │
│   10. Text Normalization            (fix typos/slang)            │
│   11. Spelling Correction           (fix misspellings)           │
└──────────────────────────────────────────────────────────────────┘
      │
      ▼
┌──────────────────────────────────────────────────────────────────┐
│  ⚙️ STAGE 2 — TEXT PREPROCESSING                                 │
│                                                                  │
│   12. Tokenization                  (split into words)           │
│   13. Remove Stopwords              (filter noise words)         │
│   14. POS Tagging                   (grammatical annotation)     │
│   15. Lemmatization                 (base word form)             │
│   16. Stemming                      (root word form)             │
│   17. Rare Word Removal             (trim vocabulary)            │
│   18. Language Detection            (identify input language)    │
└──────────────────────────────────────────────────────────────────┘
      │
      ▼
┌──────────────────────────────────────────────────────────────────┐
│  🔢 STAGE 3 — FEATURE EXTRACTION                                 │
│                                                                  │
│   19. Bag of Words (BoW / CountVectorizer)                       │
│   20. TF-IDF                                                     │
│   21. Word2Vec                                                   │
│   22. GloVe                                                      │
│   23. FastText                                                   │
│   24. BERT (Contextual Embeddings)                               │
└──────────────────────────────────────────────────────────────────┘
      │
      ▼
Model-Ready Numerical Representation
```

> **Note:** Not every step is required for every task. Use this pipeline as a menu — pick what your task needs. See the [Step Selection Guide](#-step-selection-guide) for task-specific recommendations.

---

## 📚 Table of Contents

### 🧹 Stage 1 — Text Cleaning

| # | Step | Purpose |
|---|------|---------|
| [1](#1-remove-html-tags) | Remove HTML Tags | Strip web markup |
| [2](#2-case-folding-convert-to-lowercase) | Case Folding | Normalize casing |
| [3](#3-expand-contractions) | Expand Contractions | Standardize short forms |
| [4](#4-remove-urls) | Remove URLs | Remove hyperlinks |
| [5](#5-remove-non-ascii-characters) | Remove Non-ASCII Characters | Strip encoding noise |
| [6](#6-handle-emojis) | Handle Emojis | Remove or encode emojis |
| [7](#7-handle-numbers) | Handle Numbers | Normalize digits |
| [8](#8-remove-punctuation--special-characters) | Remove Punctuation | Strip symbols |
| [9](#9-remove-extra-whitespace) | Remove Whitespace | Clean spacing |
| [10](#10-text-normalization) | Text Normalization | Fix slang and typos |
| [11](#11-spelling-correction) | Spelling Correction | Fix misspellings |

### ⚙️ Stage 2 — Text Preprocessing

| # | Step | Purpose |
|---|------|---------|
| [12](#12-tokenization) | Tokenization | Split into tokens |
| [13](#13-remove-stopwords) | Remove Stopwords | Filter noise words |
| [14](#14-pos-tagging) | POS Tagging | Grammatical annotation |
| [15](#15-lemmatization) | Lemmatization | Reduce to base form |
| [16](#16-stemming) | Stemming | Reduce to root form |
| [17](#17-rare-word-removal) | Rare Word Removal | Trim vocabulary |
| [18](#18-language-detection) | Language Detection | Identify input language |

### 🔢 Stage 3 — Feature Extraction

| # | Step | Purpose |
|---|------|---------|
| [19](#19-bag-of-words--countvectorizer) | Bag of Words | Sparse word-count matrix |
| [20](#20-tf-idf) | TF-IDF | Weighted word importance |
| [21](#21-word2vec) | Word2Vec | Dense semantic embeddings |
| [22](#22-glove) | GloVe | Pre-trained global vectors |
| [23](#23-fasttext) | FastText | Subword-aware embeddings |
| [24](#24-bert-contextual-embeddings) | BERT | Contextual deep embeddings |
| [📊](#-feature-extraction-comparison) | Comparison | Choose the right method |

### Other

| | |
|---|---|
| [💡](#-spacy-alternative-pipeline) | spaCy Alternative Pipeline |
| [⚙️](#%EF%B8%8F-full-pipeline-example) | Full Pipeline Example |
| [📊](#-step-selection-guide) | Step Selection Guide |

---

## 🛠️ Requirements

### Install All Dependencies

```bash
# Core NLP libraries
pip install nltk contractions beautifulsoup4 emoji spacy textblob

# Word embeddings & classical ML
pip install gensim scikit-learn

# Deep learning & BERT
pip install transformers tensorflow tensorflow-hub tensorflow-text

# Language detection
pip install pyicu pycld2 polyglot

# Download NLTK corpora
python -m nltk.downloader punkt stopwords wordnet averaged_perceptron_tagger omw-1.4

# Download spaCy model
python -m spacy download en_core_web_sm
```

### Verified Library Versions

| Library | Version |
|---------|---------|
| Python | 3.8+ |
| pandas | 1.5+ |
| numpy | 1.23+ |
| scikit-learn | 1.2+ |
| nltk | 3.x |
| gensim | 4.x |
| spaCy | 3.x |
| transformers | 4.x |

---

## 🧹 Stage 1 — Text Cleaning

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

### 5. Remove Non-ASCII Characters

**What it does:** Strips characters outside the standard ASCII range (0–127), including accented letters, special Unicode symbols, and encoding artifacts.

**Why it matters:** Text scraped from the web or copied across systems often contains non-ASCII noise (e.g., `\xa0`, `\u200b`, curly quotes). These characters can break tokenizers, corrupt vocabulary, and cause inconsistencies — especially in English-only pipelines.

**Example:**
```
Input:  "Héllo wörld — thís ís à tëst\xa0"
Output: "Hllo wrld  ths s  tst"
```

```python
def remove_non_ascii(text: str) -> str:
    """Remove all non-ASCII characters from text."""
    return text.encode('ascii', errors='ignore').decode('ascii')
```

> 💡 For multilingual pipelines, skip this step — non-ASCII characters are meaningful in languages like Arabic, Chinese, and French.

[↑ Back to Table of Contents](#-table-of-contents)

---

### 6. Handle Emojis

**What it does:** Either removes emojis entirely or converts them to their text description (e.g., 😊 → `:smiling_face:`).

**Why it matters:** Emojis are extremely common in social media, product reviews, and chat data. They often carry **strong sentiment signals** — ignoring or blindly removing them can hurt model performance. The right approach depends on your task.

**Example:**
```
Input:  "This is awesome 🔥💯"
Output (remove): "This is awesome"
Output (encode): "This is awesome :fire: :hundred_points:"
```

```python
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

### 7. Handle Numbers

**What it does:** Either removes digits entirely or replaces them with a normalized placeholder token like `<NUM>`.

**Why it matters:** Raw numbers rarely contribute semantic meaning in text classification. Normalizing them reduces vocabulary size and helps models generalize across different numeric values.

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

### 8. Remove Punctuation & Special Characters

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

### 9. Remove Extra Whitespace

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

### 10. Text Normalization

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
    "u":   "you",
    "r":   "are",
    "ur":  "your",
    "gr8": "great",
    "b4":  "before",
    "2day":"today",
    "thx": "thanks",
    "pls": "please",
    "w/":  "with",
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

### 11. Spelling Correction

**What it does:** Detects and corrects misspelled words using statistical language models.

**Why it matters:** User-generated text is riddled with typos and spelling errors. `"amazng"` and `"amazing"` represent the same word, but without correction, most models treat them as unrelated terms. Spelling correction is especially valuable for social media and noisy short-text corpora.

**Example:**
```
Input:  "I havv a qiuet life and enjoiy readng."
Output: "I have a quiet life and enjoy reading."
```

```python
from textblob import TextBlob

def correct_spelling(text: str) -> str:
    """
    Correct spelling using TextBlob's spell checker.
    
    Note: Can be slow on large corpora — consider batching or caching.
    """
    return str(TextBlob(text).correct())
```

> 📦 Install: `pip install textblob`
>
> ⚠️ **Performance note:** Spelling correction is computationally expensive. For large datasets, apply it selectively (e.g., only on short social media texts) or use faster alternatives like `pyspellchecker`.
>
> ⚠️ **Accuracy note:** Automatic correction can introduce errors for domain-specific jargon. Always validate on a sample of your data.

[↑ Back to Table of Contents](#-table-of-contents)

---

## ⚙️ Stage 2 — Text Preprocessing

---

### 12. Tokenization

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

### 13. Remove Stopwords

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

### 14. POS Tagging

**What it does:** Assigns a grammatical category (noun, verb, adjective, adverb, etc.) to each token — known as **Part-of-Speech** tagging.

**Why it matters:** POS tags give the pipeline awareness of grammatical structure. This is essential for accurate lemmatization (e.g., `"better"` as adjective → `"good"` vs. `"better"` as verb → `"better"`), and is also independently useful for grammar-aware feature engineering, filtering content words, or syntactic analysis.

**Example:**
```
Input:  ["The", "cats", "are", "running", "quickly"]
Output: [("The", "DT"), ("cats", "NNS"), ("are", "VBP"), ("running", "VBG"), ("quickly", "RB")]
```

```python
import nltk

nltk.download('averaged_perceptron_tagger', quiet=True)

def pos_tag_tokens(tokens: list) -> list:
    """
    Assign POS tags to a list of tokens using NLTK's averaged perceptron tagger.

    Returns:
        List of (token, POS_tag) tuples.
    """
    return nltk.pos_tag(tokens)

# Common POS tag groups for filtering
CONTENT_TAGS = {'NN', 'NNS', 'NNP', 'NNPS',   # Nouns
                'VB', 'VBD', 'VBG', 'VBN', 'VBP', 'VBZ',  # Verbs
                'JJ', 'JJR', 'JJS',              # Adjectives
                'RB', 'RBR', 'RBS'}              # Adverbs

def keep_content_words(tagged_tokens: list) -> list:
    """Filter to keep only content words (nouns, verbs, adjectives, adverbs)."""
    return [word for word, tag in tagged_tokens if tag in CONTENT_TAGS]
```

> 💡 POS tagging also feeds directly into the lemmatization step below — accurate POS tags produce significantly better lemmas.

[↑ Back to Table of Contents](#-table-of-contents)

---

### 15. Lemmatization

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
    """Map POS tag to WordNet POS constant for accurate lemmatization."""
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
>
> ⚠️ Apply **either** lemmatization **or** stemming — applying both is redundant.

[↑ Back to Table of Contents](#-table-of-contents)

---

### 16. Stemming

**What it does:** Strips word suffixes (and sometimes prefixes) using heuristic rules to reduce words to their **root stem** — which may not be a real dictionary word.

**Why it matters:** Stemming is faster than lemmatization and still achieves vocabulary consolidation. It's a practical choice when speed matters more than linguistic precision (e.g., large-scale information retrieval).

**Example:**
```
Input:  ["running", "runner", "runs", "easily", "fairly"]
Output: ["run",     "runner", "run",  "easili", "fairli"]
```

```python
from nltk.stem import PorterStemmer, SnowballStemmer

porter   = PorterStemmer()
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

> ⚠️ Choose either stemming **or** lemmatization — applying both is redundant. For most modern NLP tasks, lemmatization is preferred for its linguistic accuracy.

[↑ Back to Table of Contents](#-table-of-contents)

---

### 17. Rare Word Removal

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

### 18. Language Detection

**What it does:** Automatically identifies the language of a given text string.

**Why it matters:** Real-world datasets — especially from social media or multilingual platforms — often contain mixed-language text. Language detection lets you route text to language-specific pipelines, filter out non-target languages, or apply language-appropriate stopword lists and stemmers.

**Example:**
```
Input:  "Bonjour le monde"    → "fr" (French)
Input:  "Hello world"         → "en" (English)
Input:  "مرحبا بالعالم"       → "ar" (Arabic)
```

```python
from textblob import TextBlob

def detect_language(text: str) -> str:
    """
    Detect the language of the input text using TextBlob.

    Returns:
        ISO 639-1 language code (e.g., 'en', 'fr', 'ar').
    """
    try:
        return TextBlob(text).detect_language()
    except Exception:
        return 'unknown'

def filter_by_language(texts: list, target_lang: str = 'en') -> list:
    """Keep only texts matching the target language."""
    return [t for t in texts if detect_language(t) == target_lang]
```

> 📦 Alternatives: `langdetect`, `fasttext` (language ID model), or `polyglot` for higher accuracy on short texts.
>
> ⚠️ Language detection can be unreliable on very short strings (< 20 characters). Always validate on a sample.

[↑ Back to Table of Contents](#-table-of-contents)

---

## 🔢 Stage 3 — Feature Extraction

---

### 19. Bag of Words — CountVectorizer

**What it does:** Converts text into a sparse matrix of word occurrence counts. Each document is represented as a vector where each dimension corresponds to a word in the vocabulary.

**Why it matters:** BoW is the simplest numerical representation of text. It is fast, interpretable, and often surprisingly effective for text classification tasks, especially with a logistic regression or Naive Bayes classifier on top.

**Example:**
```
Corpus:  ["I love NLP", "NLP is great", "I love great movies"]
Vocabulary: {great, i, is, love, movies, nlp}

"I love NLP"   → [0, 1, 0, 1, 0, 1]
"NLP is great" → [1, 0, 1, 0, 0, 1]
```

```python
from sklearn.feature_extraction.text import CountVectorizer

def bag_of_words(corpus: list, max_features: int = 5000, ngram_range: tuple = (1, 1)):
    """
    Convert text corpus to a BoW count matrix.

    Args:
        corpus: List of preprocessed text strings.
        max_features: Vocabulary size limit.
        ngram_range: Tuple (min_n, max_n) for n-gram extraction.

    Returns:
        (sparse matrix, fitted vectorizer)
    """
    vectorizer = CountVectorizer(max_features=max_features, ngram_range=ngram_range)
    X = vectorizer.fit_transform(corpus)
    return X, vectorizer
```

> 💡 Use `ngram_range=(1, 2)` to capture bigrams (e.g., `"not good"`) — this is especially important for sentiment analysis.

[↑ Back to Table of Contents](#-table-of-contents)

---

### 20. TF-IDF

**What it does:** Weights each word by how frequently it appears in a document (TF) relative to how many documents contain it (IDF). Rare but informative words get higher scores; common words like `"the"` get lower scores.

**Why it matters:** TF-IDF outperforms raw BoW because it automatically downweights words that appear in nearly every document — even if stopwords weren't removed. It remains one of the most effective baselines for text classification.

**Example:**
```
"disaster" appears often in one tweet but rarely across tweets → high TF-IDF
"the"      appears in nearly every document                   → low TF-IDF
```

```python
from sklearn.feature_extraction.text import TfidfVectorizer

def tfidf_vectorize(corpus: list, max_features: int = 5000,
                    ngram_range: tuple = (1, 2), sublinear_tf: bool = True):
    """
    Convert text corpus to a TF-IDF matrix.

    Args:
        corpus: List of preprocessed text strings.
        max_features: Vocabulary size limit.
        ngram_range: Tuple (min_n, max_n) for n-gram extraction.
        sublinear_tf: Apply log normalization to term frequency.

    Returns:
        (sparse matrix, fitted vectorizer)
    """
    vectorizer = TfidfVectorizer(
        max_features=max_features,
        ngram_range=ngram_range,
        sublinear_tf=sublinear_tf
    )
    X = vectorizer.fit_transform(corpus)
    return X, vectorizer
```

> 💡 `sublinear_tf=True` applies `1 + log(tf)` instead of raw `tf`, which dampens the effect of very high-frequency terms and often improves classification performance.

[↑ Back to Table of Contents](#-table-of-contents)

---

### 21. Word2Vec

**What it does:** Trains a neural network to learn dense, low-dimensional vector representations of words based on their surrounding context. Words that appear in similar contexts end up with similar vectors.

**Why it matters:** Word2Vec captures semantic relationships that BoW and TF-IDF cannot. `"king" - "man" + "woman" ≈ "queen"` is a classic example. Document-level embeddings are created by averaging token vectors.

**Example:**
```python
from gensim.models import Word2Vec
import numpy as np

def train_word2vec(token_lists: list, vector_size: int = 100,
                   window: int = 5, min_count: int = 2,
                   workers: int = 4) -> Word2Vec:
    """
    Train a Word2Vec model on a tokenized corpus.

    Args:
        token_lists: List of token lists (one per document).
        vector_size: Dimensionality of the word vectors.
        window: Maximum distance between the current and predicted word.
        min_count: Minimum frequency threshold for vocabulary inclusion.

    Returns:
        Trained Word2Vec model.
    """
    model = Word2Vec(
        sentences=token_lists,
        vector_size=vector_size,
        window=window,
        min_count=min_count,
        workers=workers
    )
    return model

def get_document_vector(tokens: list, model: Word2Vec) -> np.ndarray:
    """Average word vectors to produce a document-level embedding."""
    vectors = [model.wv[word] for word in tokens if word in model.wv]
    return np.mean(vectors, axis=0) if vectors else np.zeros(model.vector_size)
```

> 💡 Two training algorithms are available: `sg=1` for Skip-gram (better for rare words) and `sg=0` for CBOW (faster, better for frequent words).

[↑ Back to Table of Contents](#-table-of-contents)

---

### 22. GloVe

**What it does:** Uses pre-trained GloVe vectors — dense word embeddings trained on billions of tokens from Wikipedia and Common Crawl — as a strong, ready-to-use semantic baseline.

**Why it matters:** Training your own word vectors requires a large corpus. Pre-trained GloVe vectors provide strong semantic representations out of the box, often matching or exceeding custom-trained Word2Vec on small datasets.

**Example:**
```python
import numpy as np

def load_glove_embeddings(glove_path: str) -> dict:
    """
    Load pre-trained GloVe embeddings from a .txt file.

    Args:
        glove_path: Path to the GloVe file (e.g., 'glove.6B.100d.txt').

    Returns:
        Dictionary mapping word → numpy vector.
    """
    embeddings = {}
    with open(glove_path, 'r', encoding='utf-8') as f:
        for line in f:
            values = line.split()
            word   = values[0]
            vector = np.asarray(values[1:], dtype='float32')
            embeddings[word] = vector
    return embeddings

def embed_with_glove(tokens: list, embeddings: dict, dim: int = 100) -> np.ndarray:
    """Average GloVe vectors for tokens present in the embedding vocabulary."""
    vectors = [embeddings[word] for word in tokens if word in embeddings]
    return np.mean(vectors, axis=0) if vectors else np.zeros(dim)
```

> 📥 Download pre-trained GloVe vectors from the [Stanford NLP GloVe page](https://nlp.stanford.edu/projects/glove/).
>
> 💡 Available sizes: `glove.6B` (6B tokens, 50d/100d/200d/300d), `glove.840B.300d` (840B tokens, 300d).

[↑ Back to Table of Contents](#-table-of-contents)

---

### 23. FastText

**What it does:** Extends Word2Vec by representing each word as a bag of character n-grams. The final word vector is the sum of its subword n-gram vectors.

**Why it matters:** Because FastText is built from subwords, it can generate meaningful embeddings for words it has never seen (OOV words) — including misspellings, hashtags, and compound words. This makes it especially effective for social media and noisy text.

**Example:**
```
"running" → subwords: ["run", "runn", "runni", "unnin", "nning", ...]
OOV word "amazng" → still gets a vector from its character n-grams
```

```python
from gensim.models import FastText
import numpy as np

def train_fasttext(token_lists: list, vector_size: int = 100,
                   window: int = 5, min_count: int = 2) -> FastText:
    """
    Train a FastText model with subword embeddings.

    Returns:
        Trained FastText model with OOV support.
    """
    model = FastText(
        sentences=token_lists,
        vector_size=vector_size,
        window=window,
        min_count=min_count
    )
    return model

def get_fasttext_vector(tokens: list, model: FastText) -> np.ndarray:
    """Get document embedding — handles OOV words via subword averaging."""
    vectors = [model.wv[word] for word in tokens]  # FastText handles OOV natively
    return np.mean(vectors, axis=0) if vectors else np.zeros(model.vector_size)
```

> 💡 Unlike Word2Vec and GloVe, `model.wv[word]` never raises a `KeyError` in FastText — even for OOV words.

[↑ Back to Table of Contents](#-table-of-contents)

---

### 24. BERT (Contextual Embeddings)

**What it does:** Encodes text using a pre-trained Transformer model (BERT) that produces **contextual embeddings** — the vector for each word depends on its surrounding context in the sentence.

**Why it matters:** Unlike Word2Vec, GloVe, or FastText, BERT produces different vectors for the same word in different contexts (e.g., `"bank"` as a financial institution vs. a river bank). BERT consistently achieves state-of-the-art results across virtually all NLP benchmarks.

**Example:**
```
Input sentence: "I love NLP with all my heart"
Output: Tensor of shape (1, sequence_length, 768)  ← BERT-Base
        Tensor of shape (1, sequence_length, 1024) ← BERT-Large
```

```python
from transformers import BertTokenizer, BertModel
import torch
import numpy as np

# Load pre-trained BERT tokenizer and model
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model     = BertModel.from_pretrained('bert-base-uncased')
model.eval()

def bert_encode(texts: list, max_len: int = 128) -> np.ndarray:
    """
    Encode a list of texts into BERT sentence embeddings.

    Uses mean pooling over the last hidden state to produce a
    fixed-size (768,) sentence vector for each input.

    Args:
        texts:   List of raw text strings.
        max_len: Maximum token sequence length (truncates longer inputs).

    Returns:
        numpy array of shape (len(texts), 768).
    """
    embeddings = []
    with torch.no_grad():
        for text in texts:
            inputs = tokenizer(
                text,
                return_tensors='pt',
                truncation=True,
                max_length=max_len,
                padding='max_length'
            )
            outputs   = model(**inputs)
            # Mean pool over the token dimension → shape (768,)
            embedding = outputs.last_hidden_state.mean(dim=1).squeeze().numpy()
            embeddings.append(embedding)
    return np.array(embeddings)
```

> 📦 Install: `pip install transformers torch`
>
> 💡 For production use, consider fine-tuning BERT on your task-specific dataset. A fine-tuned BERT will significantly outperform a frozen encoder.
>
> 💡 For faster inference, try `distilbert-base-uncased` (40% smaller, 60% faster, retains ~97% of BERT's accuracy).

[↑ Back to Table of Contents](#-table-of-contents)

---

## 📊 Feature Extraction Comparison

| Method | Representation | Captures Order | Captures Semantics | Handles OOV | Notes |
|--------|---------------|:-:|:-:|:-:|-------|
| **BoW / CountVectorizer** | Sparse | ❌ | ❌ | ✅ | Fast, interpretable, ignores word order |
| **TF-IDF** | Sparse | ❌ | ❌ | ✅ | Downweights common words automatically |
| **Word2Vec** | Dense | ✅ | ✅ | ❌ | Fixed vectors per word; fast inference |
| **GloVe** | Dense | ✅ | ✅ | ❌ | Excellent pre-trained baseline; no training needed |
| **FastText** | Dense | ✅ | ✅ | ✅ | Best for morphologically rich or noisy text |
| **BERT** | Contextual | ✅ | ✅✅ | ✅ | Handles polysemy; expensive but most accurate |

### When to Use Each Method

| Task | Recommended Method |
|------|--------------------|
| Simple text classification, fast iteration | TF-IDF + Logistic Regression |
| Semantic similarity search | Word2Vec / GloVe sentence averages |
| Noisy or social media text | FastText |
| State-of-the-art classification accuracy | Fine-tuned BERT / RoBERTa |
| Low-resource or real-time inference | TF-IDF or BoW |
| Multilingual tasks | FastText (multilingual) or mBERT |

---

## 💡 spaCy Alternative Pipeline

[spaCy](https://spacy.io/) offers a faster, more production-ready alternative to NLTK — with built-in tokenization, lemmatization, POS tagging, and NER in a single efficient pass.

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

Here is a complete end-to-end example combining all cleaning and preprocessing steps, followed by TF-IDF feature extraction:

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
for resource in ['punkt', 'stopwords', 'wordnet', 'averaged_perceptron_tagger', 'omw-1.4']:
    nltk.download(resource, quiet=True)

lemmatizer = WordNetLemmatizer()
STOP_WORDS = set(stopwords.words('english'))

def get_wordnet_pos(word: str) -> str:
    tag = nltk.pos_tag([word])[0][1][0].upper()
    return {'J': wordnet.ADJ, 'V': wordnet.VERB, 'R': wordnet.ADV}.get(tag, wordnet.NOUN)

def full_preprocess(text: str,
                    keep_emojis: bool = False,
                    replace_numbers: bool = True) -> list:
    """
    Full NLP preprocessing pipeline — Stages 1 & 2.

    Args:
        text: Raw input string.
        keep_emojis: If True, encodes emojis; if False, removes them.
        replace_numbers: If True, replaces numbers with <NUM>; if False, removes them.

    Returns:
        List of preprocessed, lemmatized tokens.
    """
    # Stage 1 — Text Cleaning
    text = BeautifulSoup(text, "html.parser").get_text()  # 1. Remove HTML
    text = text.lower()                                    # 2. Lowercase
    text = contractions.fix(text)                          # 3. Expand contractions
    text = re.sub(r'https?://\S+|www\.\S+', '', text)     # 4. Remove URLs
    text = text.encode('ascii', errors='ignore').decode()  # 5. Remove non-ASCII
    text = emoji.demojize(text) if keep_emojis else emoji.replace_emoji(text, replace='')  # 6. Emojis
    text = re.sub(r'\b\d[\d,\.]*\b', '<NUM>' if replace_numbers else '', text)  # 7. Numbers
    text = re.sub(r'[^a-z0-9<>\s]', '', text)             # 8. Remove punctuation
    text = re.sub(r'\s+', ' ', text).strip()               # 9. Remove extra whitespace

    # Stage 2 — Text Preprocessing
    tokens = word_tokenize(text)                           # 10. Tokenize
    tokens = [t for t in tokens if t not in STOP_WORDS]   # 11. Remove stopwords
    tokens = [lemmatizer.lemmatize(t, get_wordnet_pos(t)) for t in tokens]  # 12. Lemmatize
    return tokens


# --- Example Usage ---
raw_text = """
<p>I'm <b>absolutely</b> loving this product!!! 😍🔥
Check: https://example.com/reviews — it's gr8 value for the price.
I've used it 3 times already and it's AMAZING!</p>
"""

tokens = full_preprocess(raw_text, keep_emojis=True, replace_numbers=True)
print(tokens)
# Output: ['absolute', 'love', 'product', ':smiling_face_with_heart-eyes:', ':fire:',
#          'check', 'great', 'value', 'price', 'use', '<num>', 'time', 'already', 'amaze']
```

---

## 📊 Step Selection Guide

| Task | Recommended Steps |
|------|------------------|
| **Text Classification** | 1, 2, 3, 4, 7, 8, 9, 12, 13, 15 + TF-IDF or Word2Vec |
| **Sentiment Analysis** | 1, 2, 3, 4, 6 (encode), 8, 9, 12, 15 + FastText or BERT |
| **Topic Modeling** | 1, 2, 4, 6 (remove), 7, 8, 9, 12, 13, 15, 17 + BoW or TF-IDF |
| **Social Media NLP** | 1, 2, 3, 4, 6, 10, 8, 9, 12, 13, 15 + FastText |
| **Information Retrieval** | 2, 4, 8, 9, 12, 13, 16 + TF-IDF |
| **Machine Translation** | 1, 3, 4, 9 (minimal preprocessing) + BERT / mBERT |
| **Named Entity Recognition** | Skip step 2 (preserve casing); 1, 3, 4, 12, 14 + BERT |
| **Semantic Similarity** | 1, 2, 3, 4, 8, 9, 12 + GloVe or BERT |

---

## 🔁 Why Use This Pipeline?

**Modular by design** — each function is independent and can be used in isolation or composed in any order that suits your task.

**Three-stage architecture** — the pipeline is organized into the three natural phases of NLP data preparation: cleaning, preprocessing, and feature extraction, making it easy to enter or exit at any stage.

**Production-ready code** — proper function signatures, docstrings, type hints, and safe handling of edge cases throughout.

**Extensible for any domain** — customize slang maps, stopword lists, and frequency thresholds for medical, legal, financial, or social media text.

**Compatible with the modern ML stack** — works seamlessly with `scikit-learn`, `pandas`, `Hugging Face Transformers`, `Gensim`, `spaCy`, and more.

---

## 📁 Suggested Project Structure

```
nlp-preprocessing-pipeline/
├── preprocessing/
│   ├── __init__.py
│   ├── cleaner.py           # HTML, URLs, non-ASCII, punctuation, whitespace
│   ├── normalizer.py        # Case, contractions, slang, numbers, emojis, spelling
│   ├── tokenizer.py         # Tokenization logic
│   └── reducer.py           # Stopwords, POS tagging, lemmatization, stemming, rare words
├── features/
│   ├── __init__.py
│   ├── classical.py         # BoW, TF-IDF
│   └── embeddings.py        # Word2Vec, GloVe, FastText, BERT
├── examples/
│   └── full_pipeline.py     # End-to-end usage example
├── tests/
│   └── test_pipeline.py
├── requirements.txt
└── README.md
```

---

## 📚 References

### Papers
- [Text Classification Algorithms: A Survey](https://arxiv.org/abs/1904.08067) — Kowsari et al., 2019
- [Distributed Representations of Words and Phrases (Word2Vec)](https://papers.nips.cc/paper/5021-distributed-representations-of-words-and-phrases-and-their-compositionality.pdf) — Mikolov et al., 2013
- [GloVe: Global Vectors for Word Representation](https://nlp.stanford.edu/pubs/glove.pdf) — Pennington et al., 2014
- [Enriching Word Vectors with Subword Information (FastText)](https://arxiv.org/abs/1607.04606) — Bojanowski et al., 2017
- [BERT: Pre-training of Deep Bidirectional Transformers](https://arxiv.org/abs/1810.04805) — Devlin et al., 2018

### Books
- [Natural Language Processing in Action](https://www.manning.com/books/natural-language-processing-in-action) — Hobson Lane et al.
- [Speech and Language Processing](https://web.stanford.edu/~jurafsky/slp3/ed3book.pdf) — Jurafsky & Martin

### Useful Resources
- [Getting Started with Text Preprocessing](https://www.kaggle.com/sudalairajkumar/getting-started-with-text-preprocessing)
- [NLP with Disaster Tweets — EDA, Cleaning & BERT](https://www.kaggle.com/gunesevitan/nlp-with-disaster-tweets-eda-cleaning-and-bert)
- [NLP EDA — BoW, TF-IDF, GloVe, BERT](https://www.kaggle.com/vbmokin/nlp-eda-bag-of-words-tf-idf-glove-bert)
- [Natural Language Processing Pipeline](https://towardsdatascience.com/natural-language-processing-pipeline-93df02ecd03f)
- [NLTK Documentation](https://www.nltk.org/)
- [spaCy Documentation](https://spacy.io/usage)
- [Hugging Face Transformers](https://huggingface.co/docs/transformers)

---

## 🤝 Contributing

Contributions, issues, and feature requests are welcome. Feel free to open a pull request or start a discussion.

---

## 📄 License

This project is licensed under the [MIT License](LICENSE).
