# **Twitter Sentiment Analysis using Deep Learning**

## **Project Overview**
This project analyzes the sentiment of tweets as **positive**, **neutral**, or **negative** using a **Convolutional Neural Network (CNN)**.  It achieves an accuracy of **90%** on the validation set and supports real-time sentiment analysis.

---

## **Features**
- Preprocessing pipeline for cleaning and tokenizing tweets.
- CNN-based sentiment classification model using trainable and pretrained word embeddings.
- Data augmentation techniques to improve generalization.
- Web application deployed using Flask for real-time predictions.
---

## **Dataset**
- **Source**: https://www.kaggle.com/datasets/kazanova/sentiment140 (1.6M tweets)
- **Labels**:
  - `0`: Negative
  - `2`: Neutral
  - `4`: Positive
---

## **Tech Stack**
- **Languages**: Python
- **Libraries**: TensorFlow/Keras, Pandas, NumPy, scikit-learn, Matplotlib, Seaborn
- **Deployment**: Flask

---

## **Project Workflow**
1. **Data Preprocessing**:
   - Cleaned tweets by removing URLs, mentions, special characters, and stopwords.
   - Applied tokenization and padded sequences to ensure uniform input length.
2. **Model Architecture**:
   - Embedding Layer:
     - Trainable embeddings and pretrained embeddings (GloVe).
   - Convolutional Layers:
     - Extracted local n-gram features.
   - Max-Pooling Layer:
     - Reduced dimensionality and focused on key features.
   - Dense Layers:
     - Performed classification into three sentiment classes.
3. **Evaluation**:
   - Achieved 90% validation accuracy.
   - Confusion matrix and F1-score to assess performance.
4. **Deployment**:
   - Flask app for real-time predictions with RESTful APIs.

---
