**Introduction to COP509: Natural Language Processing**

![image](https://github.com/user-attachments/assets/fc41a5e9-293d-442a-a472-8cdf770d898d)


**Module Overview**
You will learn how to process text data for information retrieval, classification and summarisation. All sessions are on Mondays 10-12.

| Week               | Topic          |
| :---------                 | :------------------ |
| Week 1     | Introduction to the module and NLP basics|   
| Week 2     | Bag of Words|  
| Week 3   | NLP with Deep Learning and Word Embeddings|   
| Week 4     | LSI for Information Retrieval|
| Week 5-6     | Evaluation Measures. Coursework released and explained.|
| Week 7-8     | Sentiment Analysis with BERT, Text Summarisation with GPT. |   
| Week 9   | Petros and Mikel's presentations. Deduplication and Deduplication workshop|
| Week 10    |Company presentations. DeepSeek workshop, Coursework Q&A|   


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

# Week 1: Introduction to module and NLP Basics
---

**About the module**

**PPT Slides**

[Slides: Module Introduction](https://docs.google.com/presentation/d/11T7WKKYjZfMdAcg2ZGidZkc6jA4qBh9t/edit?usp=sharing&ouid=110090632559945494350&rtpof=true&sd=true)

[Slides: What is NLP?](Link pending)

[Slides: Back-to-basics: Boolean Retrieval](Link pending)  

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

# Week 2: Bag-Of-Words
---

**Lecture Material**

**PPT Slides**

[VSM Part 1: Ranked Retrieval](https://docs.google.com/presentation/d/1g2DweBphynTzsEd_L1jQjw7Xp2zYGsJA/edit?usp=sharing&ouid=110090632559945494350&rtpof=true&sd=true)

[VSM Part 2: Term Frequency](https://docs.google.com/presentation/d/1JplCLgZpODtTS2YPxf06Zwa-1bueLADv/edit?usp=sharing&ouid=110090632559945494350&rtpof=true&sd=true)

[VSM Part 3: TF-IDF](https://docs.google.com/presentation/d/1KFmzzTFVth4BhLbUdrARewCrFE7Fjskw/edit?usp=sharing&ouid=110090632559945494350&rtpof=true&sd=true)

[VSM Part 4a: The Vector Space Model](https://docs.google.com/presentation/d/17o3WcVQTKxomlrThr7MQlNWWdkVetE3D/edit?usp=sharing&ouid=110090632559945494350&rtpof=true&sd=true)

[BagOfWords Practical Lecture (See Part I)](https://docs.google.com/presentation/d/1kHgctXhFQF5EPrniAjzrHrJaOJ55IIvE/edit?usp=sharing&ouid=110090632559945494350&rtpof=true&sd=true)

Colab page: [Bag of Words](https://colab.research.google.com/drive/1XFRKBLCLSVuXmBeE3OGg4hzr7TxbmKvD?usp=sharing)

**Recordings for the PPTs (older versions):**

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

# Week 3: NLP with Deep Learning and Word Embeddings
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


**Week 3 Lab Material**

For the labs, you will focus on Text Classification with Deep Learning.

Reading: [Best Practices for Text Classification with Deep Learning](https://machinelearningmastery.com/best-practices-document-classification-deep-learning/)

Colab page: [Word Embedding Representation](https://colab.research.google.com/drive/1ylqlTKgTX_AVUgkrPYvuTJmP8V33gKse?usp=sharing)

Colab page: [Learned Embedding](https://colab.research.google.com/drive/1YBTq8whaCg4EGRZkRHlRDQt7b3nNDZSP?usp=sharing)


**Week 3 Lab Tutorial (Tutorial 3)**

Colab: [Deep Convolutional Neural Network for Sentiment Analysis (Text Classification)](https://github.com/gcosma/COP509/blob/main/Tutorials/Tutorial3CNNforSentimentAnalysis.ipynb)

**Week 3 Lab Exercise (No. 3)**

Your task is to use the cleaned ArtReviews dataset and the ArtRatings(classes) and repeat the Lab Tutorial (Week 4 lab tutorial). Briefly outline in bullets the main findings.

- How to prepare movie review text data for classification with deep learning methods.
- How to learn a word embedding as part of fitting a deep learning model.
- How to learn a standalone word embedding and how to use a pre-trained embedding in a neural network model.

**Instructions and Tips for the lab exercise:**

1. [Dataset can be found here](https://drive.google.com/drive/folders/1-Oc4jOFWZCJBXZeGCiQvPRM8TN5VDvy4?usp=sharing)

2. Use *'ArtsReviews_5000_train.txt'* and *'ArtsRatings_5000_train.txt'* for training;

   Use *'ArtsReviews_5000_test.txt'* and and *'ArtsRatings_5000_test.txt'* for test.

3. You may need ***from keras.utils.np_utils import to_categorical*** to convert the labels (Ratings).

4. Do not forget to clean the reviews with **lowercase**, otherwise, you may get an error on using word2vector embeddings for classification.

# Week 4: LSI for Information Retrieval (non-machine learning)
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



# Week 5-6: Evaluation Measures

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


# Week 7-8: Sentiment Analysis with BERT
---

**Lecture Material**

**PPT Slides**

**Lecture Material/Tutorial on Sentiment Analysis with BERT**
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


# Week 9: Text Summarisation
---

**Lecture Material**

**PPT Slides**

[Automatic Text Summarisation](https://docs.google.com/presentation/d/1axQCNOf_zxIDN5_qxYJGmvwnyteBnzmv/edit?usp=sharing&ouid=110090632559945494350&rtpof=true&sd=true)


**Reading**

[A gentle introduction to Text summarisation](https://machinelearningmastery.com/gentle-introduction-text-summarization/)

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

**Other**
[SummerTime Library for Text Summarisation and Evaluation GitHub] (https://github.com/Yale-LILY/SummerTime) [ONLY FOR LEARNING - The code no longer works]

# Week 10: Research Talks and Deduplication Workshop
---
** Research talks **

[Mikel's talk] 11:00-12:00  - Vision-Language Transformers and Hashing for Image-Text Retrieval

[Petro's talk] 12:00-13:00 Training Your Own Large Language Model and Sharing It on HuggingFace Hub

*Please attend as we will be working in groups

Read paper Deduplicating Training Data Makes Language Models Better https://arxiv.org/abs/2107.06499

Workshop activity:
1. Read the [deduplication paper](https://arxiv.org/pdf/2107.06499)
2. [Deduplication Workshop](https://drive.google.com/drive/folders/1sTnjkkkex-IkS1j-_K7EhlZ_rTWiTdyW)

#Week 10: DeepSeek Workshop

*Please attend as we will be working in groups

Workshop 2 activity:
  1. Read the [DeepSeek paper](https://arxiv.org/pdf/2401.02954)
  2. [DeepSeek Worshop](https://drive.google.com/drive/folders/1sTnjkkkex-IkS1j-_K7EhlZ_rTWiTdyW)

# Week 11: Guest speaker, Coursework and Additional Information
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


**Other**
[Computer Science CV guidance](https://www.beamjobs.com/resumes/computer-science-resume-examples)

How To Write Exactly This CV?
If you like []my particular CV template I wrote it in Latex, and you can find this template here. It is a variation from this template from Overleaf [CV template](https://www.overleaf.com/latex/templates/entry-level-resume-template-latex/jsmpwkcwyntg). Some of their [videos] (https://www.youtube.com/watch?v=n8NgELVMRS0)

[Leetcode -- interview question prep - coding](https://leetcode.com/)

[ICO toolkit](https://ico.org.uk/for-organisations/uk-gdpr-guidance-and-resources/artificial-intelligence/guidance-on-ai-and-data-protection/ai-and-data-protection-risk-toolkit/)

