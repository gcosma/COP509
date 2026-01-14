<div align="center">

# COP509: Natural Language Processing

**Department of Computer Science | Loughborough University**

*Learn to process text data for information retrieval, classification, and summarisation using state-of-the-art techniques from traditional methods to transformer architectures.*

</div>

---

## 📋 Module Overview

This module provides a comprehensive introduction to Natural Language Processing, covering foundational techniques through to modern deep learning approaches. You will gain hands-on experience with text preprocessing, vector space models, semantic analysis, transformer architectures, and text summarisation.

**Module Lead:** Professor Georgina Cosma  
**Sessions:** Mondays 10:00–12:00  
**Assessment:** Coursework (released Week 2)

---

## 🗓️ Schedule & Materials

| Week | Date | Topic | Lecture | Tutorial | Lab Exercise |
|:----:|:----:|:------|:-------:|:--------:|:------------:|
| 1 | 3rd Feb | Introduction to NLP, Boolean Retrieval & Text Cleaning | [📖 NLP Intro](https://github.com/gcosma/COP509/tree/main/Slides/1.WhatisNLP.pdf) · [📖 Boolean](https://github.com/gcosma/COP509/tree/main/Slides/2.BooleanRetrieval.pdf) · [📖 Cleaning](https://github.com/gcosma/COP509/tree/main/Slides/3.TextCleaning.pdf) | [Tutorial 1](https://github.com/gcosma/COP509/blob/main/Tutorials/Tutorial1NLPcleanText.ipynb) | [Lab 1](#week-1-introduction-to-nlp) |
| 2 | 10th Feb | Bag of Words (VSM Parts 1–2) | [📖 VSM1](https://github.com/gcosma/COP509/tree/main/Slides/4a.VSM1.pdf) · [📖 VSM2](https://github.com/gcosma/COP509/tree/main/Slides/4b.VSM2.pdf) | [Tutorial 2](https://github.com/gcosma/COP509/blob/main/Tutorials/Tutorial2BagOfWords2.ipynb) | [Lab 2](#week-2-bag-of-words) |
| 3 | 17th Feb | Bag of Words (VSM Parts 3–4) | [📖 VSM3](https://github.com/gcosma/COP509/tree/main/Slides/4c.VSM3.pdf) · [📖 VSM4](https://github.com/gcosma/COP509/tree/main/Slides/4d.VSM4.pdf) | [Tutorial 3a](https://github.com/gcosma/COP509/blob/main/Tutorials/Week3aWordEmbeddingRepresentation.ipynb) · [Tutorial 3b](https://github.com/gcosma/COP509/blob/main/Week3bLearnedEmbedding.ipynb) · [Tutorial 3c](https://github.com/gcosma/COP509/blob/main/Tutorials/Tutorial3CNNforSentimentAnalysis.ipynb) | [Lab 3](#week-3-deep-learning--word-embeddings) |
| 4 | 24th Feb | Similarity & Guest Lecture (Dr Mikel W.) | [📖 Similarity](https://github.com/gcosma/COP509/tree/main/Slides/5.Similarity.pdf) | How to build a chatbot (Dr. Mikel W.)  | How to build a chatbot (lab) (Dr. Mikel W.) |
| 5 | 3rd Mar | LSI for Information Retrieval & Evaluation Measures | [📖 LSI](https://github.com/gcosma/COP509/tree/main/Slides/6.LSI.pdf) · [📖 Evaluation](https://github.com/gcosma/COP509/tree/main/Slides/7.EvaluationMeasures.pdf) |  [Tutorial 4](https://github.com/gcosma/COP509/blob/main/Tutorials/Tutorial4LSA.ipynb) · [Tutorial 5](https://github.com/gcosma/COP509/blob/main/Tutorials/Tutorial5PlotRecallPrecision.ipynb) | [Lab 4a](#week-4-lsi-for-information-retrieval) · [Lab 4b](#week-5-evaluation-measures) |
| 6 | 10th Mar | Transformer Models & Guest Presentation | [📖 Transformers](https://github.com/gcosma/COP509/tree/main/Slides/8.TransformerModels.pdf) | [Tutorial 6](https://github.com/gcosma/COP509/blob/main/Tutorials/Tutorial6UsingBERTfortheFirstTime.ipynb) | [Lab 5](#week-6-transformer-models) |
| 7 | 17th Mar | Text Summarisation & Guest Presentation | [📖 Summarisation](https://github.com/gcosma/COP509/tree/main/Slides/9.Summarisation.pdf) | [Tutorial 7a](https://github.com/gcosma/COP509/blob/main/Tutorials/Tutorial7Summarization_with_user_pasted_data.ipynb) · [Tutorial 7b](https://github.com/gcosma/COP509/blob/main/Tutorials/Tutorial8Summarization.ipynb) | [Lab 6 & 7](#week-7-text-summarisation) |
| — | — | **Easter Break (3 weeks)** | — | — | — |
| 8 | 14th Apr | Deduplication | [📖 Deduplication](https://github.com/gcosma/COP509/tree/main/Slides/10.DeduplicatingTrainingData.pdf) | — | [Workshop](#week-8-deduplication) |
| 9 | 21st Apr | Hashing & Coursework Q&A | [📖 MinHash](https://github.com/gcosma/COP509/tree/main/Slides/Optional:MinHash.pdf) | — | — |
| 10 | 28th Apr | Company Presentations & Coursework Q&A | — | — | — |

---

## 🚀 Getting Started

### Setting Up Google Colab

All tutorials and lab exercises use Google Colab. Follow these steps to get started:

1. **Setup Guide:** [Configure your Google Drive for Colab](https://colab.research.google.com/drive/1cZm_47Q1P9mzH65dlo01tUNi-ktCaxNh?usp=sharing)
2. **Video Tutorial:** [Getting Started with Google Colab](https://www.youtube.com/watch?v=inN8seMm7UI)
3. **Interactive Guide:** [Introduction to Colab for Machine Learning](https://colab.research.google.com/drive/1pKqHYLsxV91MmGaNFgRPuZb7U34hllUO?usp=sharing)

### Datasets

All datasets for tutorials and exercises are available in the [TutorialDatasets](https://github.com/gcosma/COP509/tree/main/TutorialDatasets) folder.

---

## 📚 Weekly Details

### Week 1: Introduction to NLP

**Lab Exercise 1:** Clean the Art Rating dataset using both manual and NLTK approaches.

- **Dataset:** [`Reduced_ArtsReviews_5000.txt`](https://github.com/gcosma/COP509/tree/main/TutorialDatasets)
- **Tutorial:** [Text Cleaning for Machine Learning](https://github.com/gcosma/COP509/blob/main/Tutorials/Tutorial1NLPcleanText.ipynb)

---

### Week 2: Bag of Words

**Lab Exercise 2:** Build a sentiment analysis model using the cleaned ArtReviews dataset.

**Tasks:**
1. Create first sentiment analysis model
2. Make predictions for new reviews
3. Compare word scoring models

**Instructions:**
- **Training:** `ArtsReviews_5000_train.txt` + `ArtsRatings_5000_train.txt`
- **Testing:** `ArtsReviews_5000_test.txt` + `ArtsRatings_5000_test.txt`
- Each line in the text file represents one review
- Use `from keras.utils.np_utils import to_categorical` for label conversion

> ⚠️ **Note:** Ratings range 1–5, but `to_categorical()` starts from 0. Subtract 1 before training/testing and add 1 to predictions.

---

### Week 3: Deep Learning & Word Embeddings

**Lab Exercise 3:** Apply deep learning methods to the ArtReviews dataset.

**Learning Objectives:**
- Prepare text data for deep learning classification
- Learn word embeddings as part of model fitting
- Use standalone and pre-trained embeddings in neural networks

**Instructions:**
- Use the same train/test split as Week 2
- **Important:** Convert reviews to lowercase before using word2vec embeddings

---

### Week 5: LSI for Information Retrieval

**Lab Exercise 4a:** Implement and tune Latent Semantic Analysis.

**Tasks:**
1. Apply additional text cleaning
2. Experiment with different weighting schemes
3. Tune SVD dimensions
4. Retrieve top 20 results for queries:
   - `'I really enjoy these scissors!'`
   - `'I hate the pen!'`
---

### Week 5: Evaluation Measures

**Lab Exercise 4b:** Evaluate information retrieval performance.

**Tasks:**
1. Compute Recall, Precision, and F1-measure
2. Plot Recall/Precision curves
3. Document main findings

**Test Queries & Relevant Document IDs:**

```python
Queries = ['The pen is good.', 'The pen is poor.']

Relevant_ID = [
    [24, 337, 500, 959, 1346, 1537, 1746, 1761, 1892, 2128, 2185, 2339, 
     2603, 3161, 3181, 3192, 3202, 3627, 3796, 4161, 4293, 4678, 4758, 4790, 4798],
    [224, 353, 368, 415, 462, 571, 856, 880, 903, 906, 1377, 1532, 
     1784, 1901, 2061, 2690, 2719, 3380, 3925, 4164, 4279, 4833, 4852]
]
```

> 📝 Review IDs are 1-indexed (1 to 5000).

---

### Week 6: Transformer Models

**Lab Exercise 5:** Apply BERT to the ArtReviews dataset.

**Instructions:**
- Use **top 500** samples only (for computational efficiency)
- Training: First 500 from `ArtsReviews_5000_train.txt`
- Testing: First 500 from `ArtsReviews_5000_test.txt`
- [Alternative Dataset Location](https://drive.google.com/drive/folders/1-Oc4jOFWZCJBXZeGCiQvPRM8TN5VDvy4?usp=sharing)

---

### Week 7: Text Summarisation

**Lab Exercise 6:** Apply and evaluate text summarisation models.

**Lab Exercise 7:** Compare abstractive and extractive summarisation approaches.

**Instructions:**
- Use **top 10** reviews from `Reduced_ArtsReviews_5000.txt`

**Ground Truth for Evaluation:**

```python
targets = [
    'the product is overpriced the price should be significantly lower',
    'They came packed/coated with a yellowish oil, like mild steel tools often are. The description didn\'t specify the materials',
    'GREAT buy! I now have two of these',
    'Still having problems getting the upper tension correct.',
    'The Velcro bottom would not stick to the place I wanted it to.',
    'The "Best Pals" card can be found for under $99, so if you want to most designs possible, it might be a better buy.',
    'I bought this candle. It was NOT clear.',
    'This would indicate that it was mismarked, and lower than 14k gold.',
    'I needed more yarn to finish a baby blanket.',
    'Unfortunately the extra broad nib was cracked and leaked all over.'
]
```

---

### Week 8: Deduplication

**Workshop Activity:** Read and discuss the paper on training data deduplication.

- 📄 [Deduplicating Training Data Makes Language Models Better](https://arxiv.org/pdf/2107.06499)

---

### Week 9: Hashing

**Reading:** Explore hash-based approaches in large language models.

- 📄 [DeepSeek Paper](https://arxiv.org/pdf/2401.02954)

---

## ✅ Lab Solutions

> ⚡ **Important:** Attempt all exercises independently before consulting solutions.

All lab solutions are available in the [LabSolutions](https://github.com/gcosma/COP509/tree/main/LabSolutions) folder.

---

## 📝 Coursework

**Release Date:** Week 2

**Submission Format:** Export your Colab notebook to a presentable format using:

```python
!jupyter nbconvert '/content/drive/My Drive/Colab Notebooks/YourNotebook.ipynb' &> /dev/null
```

📹 [Video Guide: Exporting Colab Notebooks](https://drive.google.com/file/d/19f4-jP835DZneKPk0N9BtxabtBweqaNl/view?usp=sharing)

---

## 📖 Additional Resources

### Career Development

- [Computer Science CV Guidance](https://www.beamjobs.com/resumes/computer-science-resume-examples)
- [LaTeX CV Template](https://www.overleaf.com/latex/templates/entry-level-resume-template-latex/jsmpwkcwyntg)
- [Overleaf CV Videos](https://www.youtube.com/watch?v=n8NgELVMRS0)

### Technical Interview Preparation

- [LeetCode](https://leetcode.com/) — Coding interview preparation

### AI & Data Protection

- [ICO AI Toolkit](https://ico.org.uk/for-organisations/uk-gdpr-guidance-and-resources/artificial-intelligence/guidance-on-ai-and-data-protection/ai-and-data-protection-risk-toolkit/)

---

<div align="center">

**Department of Computer Science**  
Loughborough University

</div>
