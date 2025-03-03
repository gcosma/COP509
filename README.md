**Introduction to COP509: Natural Language Processing**

![image](https://github.com/user-attachments/assets/fc41a5e9-293d-442a-a472-8cdf770d898d)


**Module Overview**
You will learn how to process text data for information retrieval, classification and summarisation.

| Day               | Topic          |
| :---------                 | :------------------ |
| Day 1     | Introduction to the module and NLP basics|   
| Day 2     | Bag of Words|  
| Day 3   | NLP with Deep Learning and Word Embeddings|   
| Day 4     | LSI for Information Retrieval|
| Day 1     | Evaluation Measures. Coursework released and explained.|
| Day 2     | Sentiment Analysis with BERT, Text Summarisation with GPT. Deduplication methods.|   
| Day 3   | Petros and Mikel's presentations. Deduplication workshop   
| Day 4    |Company presentations. DeepSeek workshop, Coursework Q&A|   

**Sessions**

| Week 1 and 2    |Lecture (On Demand) |Type      |
| :---------     | :---------   | :------------------ |
|Monday |	16:00-17:00 |	On demand (means in your own time) |
|Tuesday |	16:00-17:00 |	On demand (in your own time)  |
|Total: 	2 x 2=4 hours||


| Week 1 and 2   |Lecture|Lab           |
| :---------     | :---------   | :------------------ |
|Monday| 	9:00-11:00 	|11:00-12:00, 13:00-15:00  |
Tuesday|9:00-11:00 |11:00-12:00, 13:00-15:00 |
|Wednesday| 	|
|Thursday| 	9:00-11:00 |11:00-12:00, 13:00-15:00  |
|Friday| 	9:00-11:00 |11:00-12:00, 13:00-15:00  |
|Total:| 	20 hours x 2 weeks = 40 hours |



## How to setup your Google drive for Colab tutorials and labs

Refer to this page on how to get started with Colab and how **to setup your Drive for the tutorials** [Here is the link](https://colab.research.google.com/drive/1cZm_47Q1P9mzH65dlo01tUNi-ktCaxNh?usp=sharing)

**Resources for learning Colab**

Youtube video: [Getting started with Google Colab](https://www.youtube.com/watch?v=inN8seMm7UI)

Colab page: [Introduction to Colab for machine learning](https://colab.research.google.com/drive/1pKqHYLsxV91MmGaNFgRPuZb7U34hllUO?usp=sharing)

**Main Concepts**
1. Cleaning Text Data
2. Bag-of-Words Model
3. Word Embeddings Representation and Learning
4. Semantic Analysis, Information Retrieval and Text Summarisation

Relevant read: [Introduction to Neural Information Retrieval](https://www.microsoft.com/en-us/research/publication/introduction-neural-information-retrieval/)

# Week 1, Day 1 Monday: Introduction to module and NLP Basics
---

**About the module**

**PPT Slides**

[Slides: Module Introduction](https://docs.google.com/presentation/d/11T7WKKYjZfMdAcg2ZGidZkc6jA4qBh9t/edit?usp=sharing&ouid=110090632559945494350&rtpof=true&sd=true)

[Slides: What is NLP?](https://docs.google.com/presentation/d/1T5_0aLALKSD4Fn-Cnl9yJuQFWcpvvvVX/edit?usp=sharing&ouid=110090632559945494350&rtpof=true&sd=true)

[Slides: Back-to-basics: Boolean Retrieval](https://docs.google.com/presentation/d/1IRFW6xu2TNQIn9FC3ofA06OVxIpnpOOV/edit?usp=sharing&ouid=110090632559945494350&rtpof=true&sd=true)  

**PPT Slides [on-demand]:**

[Slides: Intersecting Posting Lists](https://docs.google.com/presentation/d/1hbbBHrPfeyQZ5pPqRV88EMRV_I1VtiWA/edit?usp=sharing&ouid=110090632559945494350&rtpof=true&sd=true)

 **Recordings for the PPTs [on-demand]:**

[Recording: Boolean Retrieval recording](https://drive.google.com/file/d/1l2qA8-pmhZtEnVjQ1j71oelq4USnQj7M/view?usp=sharing) (18 mins)

[Recording: Intersecting Posting Lists recording](https://drive.google.com/file/d/1EP8TmDTfUyVqVxqI8osZ-NSgzrhAqlT6/view?usp=sharing) (7 mins)

**Lab Tutorials (Tutorial 1)**

Colab page: [How to Clean Text for Machine Learning with Python](https://github.com/gcosma/COP509/blob/main/Tutorials/Tutorial1NLPcleanText.ipynb)

**Lab Exercise (No. 1)**

Your task is to clean the Art Rating dataset that has been provided to you.Great if you can explore both manual and NLTK approaches.

Use the dataset of *'Reduced_ArtsReviews_5000.txt'* found [here](https://drive.google.com/drive/folders/1-Oc4jOFWZCJBXZeGCiQvPRM8TN5VDvy4?usp=sharing).


**Additional Task - Optional**

**Useful library:** [PyTerrier](https://github.com/terrier-org/ecir2021tutorial)

PyTerrier makes it easy to perform IR experiments in Python, but using the mature Terrier platform for the expensive indexing and retrieval operations. The following tutorial introduces PyTerrier for indexing, information retrieval and evaluation. Evaluation measures are taught in Week 2 Monday.

Part 1: Classical IR: indexing, retrieval and evaluation
[Slides](https://github.com/terrier-org/ecir2021tutorial/blob/main/slides/part1.pdf)

[Notebook Open In Colab](https://github.com/terrier-org/ecir2021tutorial/blob/main/notebooks/notebook1.ipynb) [Link to download notebooks](https://github.com/terrier-org/ecir2021tutorial/tree/main/notebooks)

[PyTerrier documentation](https://pyterrier.readthedocs.io/en/latest/).

# Week 1, Day 2 Tuesday: Bag-Of-Words
---

**Lecture Material**

**PPT Slides**

[VSM Part 1: Ranked Retrieval](https://docs.google.com/presentation/d/1g2DweBphynTzsEd_L1jQjw7Xp2zYGsJA/edit?usp=sharing&ouid=110090632559945494350&rtpof=true&sd=true)

[VSM Part 2: Term Frequency](https://docs.google.com/presentation/d/1JplCLgZpODtTS2YPxf06Zwa-1bueLADv/edit?usp=sharing&ouid=110090632559945494350&rtpof=true&sd=true)

[VSM Part 3: TF-IDF](https://docs.google.com/presentation/d/1KFmzzTFVth4BhLbUdrARewCrFE7Fjskw/edit?usp=sharing&ouid=110090632559945494350&rtpof=true&sd=true)

[VSM Part 4a: The Vector Space Model](https://docs.google.com/presentation/d/17o3WcVQTKxomlrThr7MQlNWWdkVetE3D/edit?usp=sharing&ouid=110090632559945494350&rtpof=true&sd=true)

[BagOfWords Practical Lecture (See Part I)](https://docs.google.com/presentation/d/1kHgctXhFQF5EPrniAjzrHrJaOJ55IIvE/edit?usp=sharing&ouid=110090632559945494350&rtpof=true&sd=true)

Colab page: [Bag of Words](https://colab.research.google.com/drive/1XFRKBLCLSVuXmBeE3OGg4hzr7TxbmKvD?usp=sharing)

**Recordings for the PPTs [on-demand]:**

[VSM Part 1: Ranked Retrieval](https://drive.google.com/file/d/1AqDT-X0JCRnmHD-u_mmwgFjnweLB2G0R/view?usp=sharing) (7 mins)

[VSM Part 2: Term Frequency](https://drive.google.com/file/d/1EF6zCoSwZ_oLd_aYPpvwqnHhgjd2HyMe/view?usp=sharing) (10 mins)

[VSM Part 3: TF-IDF](https://drive.google.com/file/d/1jQS6YWYHa5YJzZ6S20pN7i3vpwj3i1e1/view?usp=sharing) (15 mins)

[VSM Part 4a: The Vector Space Model and Computing Vector Similarity for Unormalised Vectors](https://drive.google.com/file/d/1vjpmTFlW4mRG3T-ChoV_VHD5JC8GDOEC/view?usp=sharing) (21 mins)

[VSM Part 4b: Computing Vector Similarity for Normalised Vectors](https://drive.google.com/file/d/1bpbf1MXHIr3BFpYdwC2x2ewudUxu8NTl/view?usp=sharing) (8 mins)


**Lab Tutorial (Tutorial 2)**

Colab page: [How to Develop a Deep Learning Bag-of-Words Model for Sentiment Analysis (Text Classification)](https://github.com/gcosma/COP509/blob/main/Tutorials/Tutorial2BagOfWords2.ipynb)

Note that this tutorial also has stop word removal that is important cleaning process.

**Lab Exercise (No. 2)**

Your task is to use the cleaned ArtReviews dataset and the ArtRatings(classes) and repeat the Lab sheet to create a Sentiment analysis model as follows (and as explained in the lab tutorial).

  1. First sentiment analysis model
  2. Making a prediction for new reviews
  3. Comparing word scoring models

**Instructions for Lab Exercise**

[Datasets are found here](https://drive.google.com/drive/folders/1-Oc4jOFWZCJBXZeGCiQvPRM8TN5VDvy4?usp=sharing)

1. Use *'ArtsReviews_5000_train.txt'* and *'ArtsRatings_5000_train.txt'* for training;

   Use *'ArtsReviews_5000_test.txt'* and and *'ArtsRatings_5000_test.txt'* for test.

2. Different from the tutorial (where each txt file is a review), each line of the txt file used by this lab exrcise is a review, thus you should read the txt file as lines.

3. You may need ***from keras.utils.np_utils import to_categorical*** to convert the labels (Ratings).

Notes: Ratings start from **1** to **5**, but the result of ***to_categorical()*** starts from **0**. Therefore, every Rating label needs to be decreased with **1** if training and testing, and the result of prediction needs to be increased with **1**.

# Week 1, Day 3 Thursday : NLP with Deep Learning and Word Embeddings
---

**Lecture Material**

**PPT Slides**

[Similarity and Distance](https://docs.google.com/presentation/d/1IupWoCYACLs_mEgZBkmULnqss73RzkeT/edit?usp=sharing&ouid=110090632559945494350&rtpof=true&sd=true)

[Document Similarity](https://docs.google.com/presentation/d/1DJX92LMk6okLVqklqhn1luP8hY_diihS/edit?usp=sharing&ouid=110090632559945494350&rtpof=true&sd=true)

Word embeddings lecture I (from yesterday)

[Word Embeddings Practical Lecture (See Part II)](https://docs.google.com/presentation/d/1kHgctXhFQF5EPrniAjzrHrJaOJ55IIvE/edit?usp=sharing&ouid=110090632559945494350&rtpof=true&sd=true)

**Recordings for the PPTs [on-demand]**

[Similarity and Distance](https://drive.google.com/file/d/1fziGTulP0btHoBHJYd6VbGLNj1QGowTD/view?usp=sharing)

[Document Similarity](https://drive.google.com/file/d/1YHDTUTRHaxOMGfaWGUGMJSLVZxjzmvwz/view?usp=sharing)

**Reading**

[A Tale of Two Encodings: Comparing Bag-of-Words and Word2vec for VQA](https://www.cs.princeton.edu/courses/archive/spring18/cos598B/public/projects/System/COS598B_spr2018_TwoEncodings.pdf)

**Additional Resource**

Part 2: Modern Retrieval Architectures: PyTerrier data model and operators, towards re-rankers and learning-to-rank

[Slides](https://github.com/terrier-org/ecir2021tutorial/blob/main/slides/part2.pdf)

[Notebook Open In Colab](https://github.com/terrier-org/ecir2021tutorial/blob/main/notebooks/notebook2.ipynb)


**Day 3 Lab Material**

For the labs, you will focus on Text Classification with Deep Learning.

Reading: [Best Practices for Text Classification with Deep Learning](https://machinelearningmastery.com/best-practices-document-classification-deep-learning/)

Colab page: [Word Embedding Representation](https://colab.research.google.com/drive/1ylqlTKgTX_AVUgkrPYvuTJmP8V33gKse?usp=sharing)

Colab page: [Learned Embedding](https://colab.research.google.com/drive/1YBTq8whaCg4EGRZkRHlRDQt7b3nNDZSP?usp=sharing)


**Day 3 Lab Tutorial (Tutorial 3)**

Colab: [Deep Convolutional Neural Network for Sentiment Analysis (Text Classification)](https://github.com/gcosma/COP509/blob/main/Tutorials/Tutorial3CNNforSentimentAnalysis.ipynb)

**Day 3 Lab Exercise (No. 3)**

Your task is to use the cleaned ArtReviews dataset and the ArtRatings(classes) and repeat the Lab Tutorial (Day 4 lab tutorial). Briefly outline in bullets the main findings.

- How to prepare movie review text data for classification with deep learning methods.
- How to learn a word embedding as part of fitting a deep learning model.
- How to learn a standalone word embedding and how to use a pre-trained embedding in a neural network model.

**Instructions and Tips for the lab exercise:**

1. [Dataset can be found here](https://drive.google.com/drive/folders/1-Oc4jOFWZCJBXZeGCiQvPRM8TN5VDvy4?usp=sharing)

2. Use *'ArtsReviews_5000_train.txt'* and *'ArtsRatings_5000_train.txt'* for training;

   Use *'ArtsReviews_5000_test.txt'* and and *'ArtsRatings_5000_test.txt'* for test.

3. You may need ***from keras.utils.np_utils import to_categorical*** to convert the labels (Ratings).

4. Do not forget to clean the reviews with **lowercase**, otherwise, you may get an error on using word2vector embeddings for classification.

# Week 1, Day 4 Friday: LSI for Information Retrieval (non-machine learning)
---

**Lecture Material**

**PPT Slides**

[LSA Part 1 - Latent Semantic Indexing](https://docs.google.com/presentation/d/19jPwerqd7G3Jw7oTmviPeU7QtjQ3SMZj/edit?usp=sharing&ouid=110090632559945494350&rtpof=true&sd=true)

[LSA Part 2 - Dimensionality Reduction](https://docs.google.com/presentation/d/1cDWDU3spgyCNHjSir9I9ZpSrz8vacZlu/edit?usp=sharing&ouid=110090632559945494350&rtpof=true&sd=true)

[LSA Part 3 - LSI for information retrieval](https://docs.google.com/presentation/d/1ODMFK089oOWt4_6hibVyhkxdJ_wbcfdd/edit?usp=sharing&ouid=110090632559945494350&rtpof=true&sd=true)

**Recordings for the PPTs [on-demand]**

[LSA Part 1 recording - Latent Semantic Indexing](https://drive.google.com/file/d/14h0p0YgSlbqhZYY5NAuSIEUQKEcneSEw/view?usp=sharing) (9 mins)

[LSA Part 2 recording - Dimensionality Reduction](https://drive.google.com/file/d/1TA6HaPXCoWzNvqbgXyQoGhOAjrJJnSNq/view?usp=sharing) (9 mins)

[LSA Part 3 recording - LSI for information retrieval](https://drive.google.com/file/d/16q6W3t5EZoeOtftgE9Dk9ELSsMwxVw3Q/view?usp=sharing) (10 mins)

**Lab Tutorials (Tutorial 4)**

Colab page: [Latent Semantic Analysis](https://github.com/gcosma/COP509/blob/main/Tutorials/Tutorial4LSA.ipynb)


**Lab Exercise (No. 4a)**

1. Your task is to read and understand the LSA lab exercise code. Apply and tune the LSA solution further. Apply more cleaning, try out different weightning schemes and tune using different SVD dimensions. 

Run this Coladb code: [LSA lab](https://github.com/gcosma/COP509/blob/main/LabSolutions/Lab_Exercise_(No_4a).ipynb) 

2. Write code to retrieve the top 20 results for the 2 given queries.
['I really enjoy these scissors!',
'I hate the pen!']

**Instructions for the Lab Exercise**

Use the dataset of *'Reduced_ArtsReviews_5000.txt'* found [here](https://drive.google.com/drive/folders/1-Oc4jOFWZCJBXZeGCiQvPRM8TN5VDvy4?usp=sharing).

**Additional Resource**

Part 3: Contemporary Retrieval Architectures: Neural re-rankers such as BERT, EPIC, ColBERT

[Slides](https://github.com/terrier-org/ecir2021tutorial/blob/main/slides/part3.pdf)

[OpenNIR and monoT5 Notebook Open In Colab](https://github.com/terrier-org/ecir2021tutorial/blob/main/notebooks/notebook3.1.ipynb)

[ColBERT Re-Ranker Notebook Open In Colab](https://github.com/terrier-org/ecir2021tutorial/blob/main/notebooks/notebook3.2.ipynb)

#Topic Modelling
Apply Topic Modelling to the lab datasets.

[**Tutorial 9 ** on BertTopic Modelling: BERTtopic tutorial](https://github.com/gcosma/COP509/blob/main/Tutorials/Tutorial9BERTopic.ipynb)



# Week 2, Day 1 Monday: Evaluation Measures

---





**Lecture Material**

**PPT slides [On Demand]**

[Evaluation Measures](https://docs.google.com/presentation/d/1OSDB4NtSuOnlOiU5_Hej17DfAhqyFEBc/edit?usp=sharing&ouid=110090632559945494350&rtpof=true&sd=true)

[Excel spreadsheet for demonstration only](https://docs.google.com/spreadsheets/d/1D7vl-GxQ6xsdvKqFtCsdDNIK80RgUy57/edit?usp=drive_link&ouid=110090632559945494350&rtpof=true&sd=true) -might need downloading cause charts are not showing properly via google

**Recordings for the PPTs [On Demand]**

[Evaluation measures Part 1 - Ranked Evaluation](https://drive.google.com/file/d/1l4Ptz7X85ZQOIs3KDy5-CKvQw9j1_4-l/view?usp=sharing) (12 mins)

[Evaluation measures Part 2 - Unranked Evaluation](https://drive.google.com/file/d/1SmjtEHBfq4w6rnlOhCrCrFxePMROIxY4/view?usp=sharing) (18 mins)

[Evaluation measures Part 3 - Evaluation benchmarks](https://drive.google.com/file/d/1f29FNQX2s7lgCLwLkdZHKZi0m5vVtb8s/view?usp=sharing) (4 mins)

**Lab Tutorial (Tutorial 5)**

Colab: [Evaluation Measures](https://colab.research.google.com/drive/1XZc5cx3GKTUYJhLZjWgWyobkrGFd6nfA?usp=sharing)

**Lab Exercise (No. 4b)**

Building on from the previous lab

1. Use Evaluation measures to evaluate the performance of the 2 given queries.

2. Briefly outline in bullets the main findings.

**Instructions and Tips for the Lab Exercise**

1. Use the dataset of *'Reduced_ArtsReviews_5000.txt'*. [Dataset can be found here](https://drive.google.com/drive/folders/1-Oc4jOFWZCJBXZeGCiQvPRM8TN5VDvy4?usp=sharing)

2. You can program for computing *Recall*, *Precision*, and *F1-measure* and plotting *Recall/Precision curves*.

3. The order of the Review in the file is the ID of the Review (Notice: order starts from 1 to 5000.). The two queries and their relevant Reviews' ID are given as following:

Querys = ['The pen is good.', 'The pen is poor.']

Relevant_ID = [[24,337,500,959,1346,1537,1746,1761,1892,2128,2185,2339,2603,3161,3181,3192,3202,3627,3796,4161,4293,4678,4758,4790,4798],
[224,353,368,415,462,571,856,880,903,906,1377,1532,1784,1901,2061,2690,2719,3380,3925,4164,4279,4833,4852]]


# Week 2, Day 2 Tuesday: Sentiment Analysis with BERT
---

**Lecture Material**

**PPT Slides**

**Lecture Material/Tutorial on Sentiment Analysis with BERT**

[All of Wednesday's talks and sessions can be found here](https://drive.google.com/drive/folders/19Jisk9Ld27tXz87Al4gsNticY7BNLqDC?usp=drive_link)

9:00-9:45 - [NLP with BERT] (https://docs.google.com/presentation/d/1XRQe7xXQ8A8nn8GTuWaHkDhx-VIHvq9Q/edit?usp=sharing&ouid=110090632559945494350&rtpof=true&sd=true)


**Tutorial 6**
14:00-15:00
Colab: [A Visual Notebook to Using BERT for the First Time](https://github.com/gcosma/COP509/blob/main/Tutorials/Tutorial6UsingBERTfortheFirstTime.ipynb)

**Lab Exercise (No. 5)**

Your task is to use the cleaned ArtReviews dataset and the ArtRatings(classes) and repeat the Tutorial.

**Instructions for Lab Exercise**

Use the top **500** data of *'ArtsReviews_5000_train.txt'* and *'ArtsRatings_5000_train.txt'* for tarining;

Use the top **500** data 'ArtsReviews_5000_test.txt'* and and *'ArtsRatings_5000_test.txt'* for testing.

[Dataset can be found here](https://drive.google.com/drive/folders/1-Oc4jOFWZCJBXZeGCiQvPRM8TN5VDvy4?usp=sharing)


# Week 2, Day 2 Tuesday: Text Summarisation
---


**Lecture Material**

**PPT Slides**

[Automatic Text Summarisation](https://docs.google.com/presentation/d/1axQCNOf_zxIDN5_qxYJGmvwnyteBnzmv/edit?usp=sharing&ouid=110090632559945494350&rtpof=true&sd=true)

[Computer Science CV guidance](https://www.beamjobs.com/resumes/computer-science-resume-examples)

How To Write Exactly This CV?
If you like []my particular CV template I wrote it in Latex, and you can find this template here. It is a variation from this template from Overleaf [CV template](https://www.overleaf.com/latex/templates/entry-level-resume-template-latex/jsmpwkcwyntg). Some of their [videos] (https://www.youtube.com/watch?v=n8NgELVMRS0)

[Leetcode -- interview question prep - coding](https://leetcode.com/)

[ICO toolkit](https://ico.org.uk/for-organisations/uk-gdpr-guidance-and-resources/artificial-intelligence/guidance-on-ai-and-data-protection/ai-and-data-protection-risk-toolkit/)

**Lecture Activity**

- Try out some summarisation methods from Huggingface https://huggingface.co/models?pipeline_tag=summarization&sort=downloads. Can you tell which are abstractive and which are extractive?

- Have a look at this model? https://huggingface.co/philschmid/flan-t5-base-samsum what metrics have they used to evaluate performance? how are these different to other metrics you have come across for NLP and classification tasks?

**Reading**

[A gentle introduction to Text summarisation](https://machinelearningmastery.com/gentle-introduction-text-summarization/)

[SummerTime Library for Text Summarisation and Evaluation GitHub] (https://github.com/Yale-LILY/SummerTime) [ONLY FOR LEARNING - The code no longer works]

**Tutorials 7 and 8**

[A notebook provided by me] (https://drive.google.com/file/d/1JdYX3reNpYl9NWwZUAVkTd81zzQEViwi/view?usp=drive_link)

Colab: [Hugging Face Summarisation Tutorial](https://colab.research.google.com/drive/18bvbLe2Eh6YmqzlD5JnS205MGieW9yLl?usp=drive_link)

**Lab Exercise (No. 6)**

Your task is to apply text summarisation and evaluation to the ArtReviews dataset using the turorial example. You will only need to summarise using the different models and evaluate and compare the results. Briefly outline in bullets the main findings.  

**Instructions and Tips for Lab Exercise**

1. Use the top **10** reviews of *'Reduced_ArtsReviews_5000.txt'*.
2. Creating CustomDataset is not mandatory, and directly using the reviews of *'Reduced_ArtsReviews_5000.txt'* is also acceptable.
3. The ground truth for evaluation is given as follows:

targets = ['the product is overpriced'
           ' the price should be significantly lower ',
           'They came packed/coated with a yellowish oil, like mild steel tools often are.'
           'The description didn\'t specify the materials',
           'GREAT buy!'
           'I now have two of these',
           'Still having problems getting the upper tension correct.',
           'The Velcro bottom would not stick to the place I wanted it to.',
           'The "Best Pals" card can be found for under $99, so if you want to most designs possible, it might be a better buy.',
           'I bought this candle. It was NOT clear.',
           'This would indicate that it was mismarked, and lower than 14k gold.',
           'I needed more yarn to finish a baby blanket.',
           'Unfortunately the extra broad nib was cracked and leaked all over.']

**Lab Exercise (No. 7)**

Your task is to apply text summarisation to the ArtReviews dataset using abstarctice and extractive summarisaiton models by following the above tutorials.

**Instructions for Lab Exercise**

Use the top **10** reviews of *'Reduced_ArtsReviews_5000.txt'*.
[Dataset can be found here](https://drive.google.com/drive/folders/1-Oc4jOFWZCJBXZeGCiQvPRM8TN5VDvy4?usp=sharing)



# Week 2, Day 3 Thursday: Research Talks and Deduplication Workshop
---
** Research talks **

[Mikel's talk] 11:00-12:00  - Vision-Language Transformers and Hashing for Image-Text Retrieval

[Petro's talk] 12:00-13:00 Training Your Own Large Language Model and Sharing It on HuggingFace Hub

*Please attend as we will be working in groups

Read paper Deduplicating Training Data Makes Language Models Better https://arxiv.org/abs/2107.06499

Workshop activity:
1. Read the [deduplication paper](https://arxiv.org/pdf/2107.06499)
2. [Deduplication Workshop](https://drive.google.com/drive/folders/1sTnjkkkex-IkS1j-_K7EhlZ_rTWiTdyW)

#Week 2, Day 4 Friday: DeepSeek Workshop

*Please attend as we will be working in groups

Workshop 2 activity:
  1. Read the [DeepSeek paper](https://arxiv.org/pdf/2401.02954)
  2. [DeepSeek Worshop](https://drive.google.com/drive/folders/1sTnjkkkex-IkS1j-_K7EhlZ_rTWiTdyW)

# Week 2, Day 4 Friday: Guest speaker, Coursework and Additional Information
---
10:00-11:00 Guest speaker from SVGC

Video recording: [Lab:Let's build GPT: from scratch, in code, spelled out.](https://www.youtube.com/watch?app=desktop&v=kCc8FmEb1nY])

Activity:
- Pretrain the transformer on the reviews dataset, then initialize with that model and finetune it on tiny shakespeare with a smaller number of steps and lower learning rate. Can you obtain a lower validation loss by the use of pretraining?

**Additional Reading**

Reading: [Neural Information Retrieval](https://www.microsoft.com/en-us/research/uploads/prod/2017/06/fntir2018-neuralir-mitra.pdf)

**Additional Libraries**

1. The official repository of "IR From Bag-of-words to BERT and Beyond through Practical Experiments", an ECIR 2021 full-day tutorial with PyTerrier and OpenNIR search toolkits. [PyTerrier tutorials in Google Colab](https://github.com/terrier-org/ecir2021tutorial)

Part 4: Recent Advances beyond the classical inverted index: neural inverted index augmentation, nearest neighbor search, dense retrieval

- [Slides](https://github.com/terrier-org/ecir2021tutorial/blob/main/slides/part4.pdf)

- [doc2query and DeepCT notebook Open In Colab](https://github.com/terrier-org/ecir2021tutorial/blob/main/notebooks/notebook4.1.ipynb)

- [ANCE notebook Open In Colab](https://github.com/terrier-org/ecir2021tutorial/blob/main/notebooks/notebook4.3.ipynb)

- [ColBERT notebook Open In Colab](https://github.com/terrier-org/ecir2021tutorial/blob/main/notebooks/notebook4.3.ipynb)





**COURSEWORK (will be released in week 2)**


To export the code and outputs into a presentable format please see the recording by Yomi.
[LINK](https://drive.google.com/file/d/19f4-jP835DZneKPk0N9BtxabtBweqaNl/view?usp=sharing)

!jupyter nbconvert '/content/drive/My Drive/Colab Notebooks/BagOfWords2.ipynb' &> /dev/null

[Coursework Datasets (Available here)](https://drive.google.com/drive/folders/1oGaiswHyhiiaR7gk51pO7N8CDPZKBhLK?usp=share_link)

[COURSEWORK FILE (provided on LEARN)]

**Lab Solutions**

Please attempt to solve the labs on your own first. **Avoid the temptation of looking at these before you try to solve them first**

Lab solutions can be found [here](https://github.com/gcosma/COP509/tree/main/LabSolutions)

Lab Exercise (No. 1)

Lab Exercise (No. 2)

Lab Exercise (No. 3)

Lab Exercise (No. 4a)

Lab Exercise (No. 4b)

Lab Exercise (No. 5)

Lab Exercise (No. 6)

Lab Exercise (No. 7)




```python
from google.colab import drive
drive.mount('/content/drive')
```

    Mounted at /content/drive



```python
#for pdf conversion
#!sudo apt-get install texlive-xetex texlive-fonts-recommended texlive-plain-generic
```


```python
# First, check your current working directory
!pwd

# Then, list the contents of your target directory
!ls "/content/drive/MyDrive/Colab Notebooks/21COP509/"

# Now run the conversion with a full output path specified
!jupyter nbconvert --to markdown "/content/drive/MyDrive/Colab Notebooks/21COP509/COP509Main.ipynb" --output "/content/drive/MyDrive/Colab Notebooks/21COP509/main"
```

    /content
    ls: cannot access '/content/drive/MyDrive/Colab Notebooks/21COP509/': No such file or directory
    [NbConvertApp] WARNING | pattern '/content/drive/MyDrive/Colab Notebooks/21COP509/COP509Main.ipynb' matched no files
    This application is used to convert notebook files (*.ipynb)
            to various other formats.
    
            WARNING: THE COMMANDLINE INTERFACE MAY CHANGE IN FUTURE RELEASES.
    
    Options
    =======
    The options below are convenience aliases to configurable class-options,
    as listed in the "Equivalent to" description-line of the aliases.
    To see all configurable class-options for some <cmd>, use:
        <cmd> --help-all
    
    --debug
        set log level to logging.DEBUG (maximize logging output)
        Equivalent to: [--Application.log_level=10]
    --show-config
        Show the application's configuration (human-readable format)
        Equivalent to: [--Application.show_config=True]
    --show-config-json
        Show the application's configuration (json format)
        Equivalent to: [--Application.show_config_json=True]
    --generate-config
        generate default config file
        Equivalent to: [--JupyterApp.generate_config=True]
    -y
        Answer yes to any questions instead of prompting.
        Equivalent to: [--JupyterApp.answer_yes=True]
    --execute
        Execute the notebook prior to export.
        Equivalent to: [--ExecutePreprocessor.enabled=True]
    --allow-errors
        Continue notebook execution even if one of the cells throws an error and include the error message in the cell output (the default behaviour is to abort conversion). This flag is only relevant if '--execute' was specified, too.
        Equivalent to: [--ExecutePreprocessor.allow_errors=True]
    --stdin
        read a single notebook file from stdin. Write the resulting notebook with default basename 'notebook.*'
        Equivalent to: [--NbConvertApp.from_stdin=True]
    --stdout
        Write notebook output to stdout instead of files.
        Equivalent to: [--NbConvertApp.writer_class=StdoutWriter]
    --inplace
        Run nbconvert in place, overwriting the existing notebook (only
                relevant when converting to notebook format)
        Equivalent to: [--NbConvertApp.use_output_suffix=False --NbConvertApp.export_format=notebook --FilesWriter.build_directory=]
    --clear-output
        Clear output of current file and save in place,
                overwriting the existing notebook.
        Equivalent to: [--NbConvertApp.use_output_suffix=False --NbConvertApp.export_format=notebook --FilesWriter.build_directory= --ClearOutputPreprocessor.enabled=True]
    --coalesce-streams
        Coalesce consecutive stdout and stderr outputs into one stream (within each cell).
        Equivalent to: [--NbConvertApp.use_output_suffix=False --NbConvertApp.export_format=notebook --FilesWriter.build_directory= --CoalesceStreamsPreprocessor.enabled=True]
    --no-prompt
        Exclude input and output prompts from converted document.
        Equivalent to: [--TemplateExporter.exclude_input_prompt=True --TemplateExporter.exclude_output_prompt=True]
    --no-input
        Exclude input cells and output prompts from converted document.
                This mode is ideal for generating code-free reports.
        Equivalent to: [--TemplateExporter.exclude_output_prompt=True --TemplateExporter.exclude_input=True --TemplateExporter.exclude_input_prompt=True]
    --allow-chromium-download
        Whether to allow downloading chromium if no suitable version is found on the system.
        Equivalent to: [--WebPDFExporter.allow_chromium_download=True]
    --disable-chromium-sandbox
        Disable chromium security sandbox when converting to PDF..
        Equivalent to: [--WebPDFExporter.disable_sandbox=True]
    --show-input
        Shows code input. This flag is only useful for dejavu users.
        Equivalent to: [--TemplateExporter.exclude_input=False]
    --embed-images
        Embed the images as base64 dataurls in the output. This flag is only useful for the HTML/WebPDF/Slides exports.
        Equivalent to: [--HTMLExporter.embed_images=True]
    --sanitize-html
        Whether the HTML in Markdown cells and cell outputs should be sanitized..
        Equivalent to: [--HTMLExporter.sanitize_html=True]
    --log-level=<Enum>
        Set the log level by value or name.
        Choices: any of [0, 10, 20, 30, 40, 50, 'DEBUG', 'INFO', 'WARN', 'ERROR', 'CRITICAL']
        Default: 30
        Equivalent to: [--Application.log_level]
    --config=<Unicode>
        Full path of a config file.
        Default: ''
        Equivalent to: [--JupyterApp.config_file]
    --to=<Unicode>
        The export format to be used, either one of the built-in formats
                ['asciidoc', 'custom', 'html', 'latex', 'markdown', 'notebook', 'pdf', 'python', 'qtpdf', 'qtpng', 'rst', 'script', 'slides', 'webpdf']
                or a dotted object name that represents the import path for an
                ``Exporter`` class
        Default: ''
        Equivalent to: [--NbConvertApp.export_format]
    --template=<Unicode>
        Name of the template to use
        Default: ''
        Equivalent to: [--TemplateExporter.template_name]
    --template-file=<Unicode>
        Name of the template file to use
        Default: None
        Equivalent to: [--TemplateExporter.template_file]
    --theme=<Unicode>
        Template specific theme(e.g. the name of a JupyterLab CSS theme distributed
        as prebuilt extension for the lab template)
        Default: 'light'
        Equivalent to: [--HTMLExporter.theme]
    --sanitize_html=<Bool>
        Whether the HTML in Markdown cells and cell outputs should be sanitized.This
        should be set to True by nbviewer or similar tools.
        Default: False
        Equivalent to: [--HTMLExporter.sanitize_html]
    --writer=<DottedObjectName>
        Writer class used to write the
                                            results of the conversion
        Default: 'FilesWriter'
        Equivalent to: [--NbConvertApp.writer_class]
    --post=<DottedOrNone>
        PostProcessor class used to write the
                                            results of the conversion
        Default: ''
        Equivalent to: [--NbConvertApp.postprocessor_class]
    --output=<Unicode>
        Overwrite base name use for output files.
                    Supports pattern replacements '{notebook_name}'.
        Default: '{notebook_name}'
        Equivalent to: [--NbConvertApp.output_base]
    --output-dir=<Unicode>
        Directory to write output(s) to. Defaults
                                      to output to the directory of each notebook. To recover
                                      previous default behaviour (outputting to the current
                                      working directory) use . as the flag value.
        Default: ''
        Equivalent to: [--FilesWriter.build_directory]
    --reveal-prefix=<Unicode>
        The URL prefix for reveal.js (version 3.x).
                This defaults to the reveal CDN, but can be any url pointing to a copy
                of reveal.js.
                For speaker notes to work, this must be a relative path to a local
                copy of reveal.js: e.g., "reveal.js".
                If a relative path is given, it must be a subdirectory of the
                current directory (from which the server is run).
                See the usage documentation
                (https://nbconvert.readthedocs.io/en/latest/usage.html#reveal-js-html-slideshow)
                for more details.
        Default: ''
        Equivalent to: [--SlidesExporter.reveal_url_prefix]
    --nbformat=<Enum>
        The nbformat version to write.
                Use this to downgrade notebooks.
        Choices: any of [1, 2, 3, 4]
        Default: 4
        Equivalent to: [--NotebookExporter.nbformat_version]
    
    Examples
    --------
    
        The simplest way to use nbconvert is
    
                > jupyter nbconvert mynotebook.ipynb --to html
    
                Options include ['asciidoc', 'custom', 'html', 'latex', 'markdown', 'notebook', 'pdf', 'python', 'qtpdf', 'qtpng', 'rst', 'script', 'slides', 'webpdf'].
    
                > jupyter nbconvert --to latex mynotebook.ipynb
    
                Both HTML and LaTeX support multiple output templates. LaTeX includes
                'base', 'article' and 'report'.  HTML includes 'basic', 'lab' and
                'classic'. You can specify the flavor of the format used.
    
                > jupyter nbconvert --to html --template lab mynotebook.ipynb
    
                You can also pipe the output to stdout, rather than a file
    
                > jupyter nbconvert mynotebook.ipynb --stdout
    
                PDF is generated via latex
    
                > jupyter nbconvert mynotebook.ipynb --to pdf
    
                You can get (and serve) a Reveal.js-powered slideshow
    
                > jupyter nbconvert myslides.ipynb --to slides --post serve
    
                Multiple notebooks can be given at the command line in a couple of
                different ways:
    
                > jupyter nbconvert notebook*.ipynb
                > jupyter nbconvert notebook1.ipynb notebook2.ipynb
    
                or you can specify the notebooks list in a config file, containing::
    
                    c.NbConvertApp.notebooks = ["my_notebook.ipynb"]
    
                > jupyter nbconvert --config mycfg.py
    
    To see all available configurables, use `--help-all`.
    

