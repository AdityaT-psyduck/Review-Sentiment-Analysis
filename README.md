Amazon Fine Food Reviews: Sentiment Analysis
============================================

This project performs sentiment analysis on the Amazon Fine Food Reviews dataset. The goal is to build and evaluate a deep learning model that can accurately classify a customer review as either **Positive** or **Negative** based on its text.

Dataset
-------

The project uses the [Amazon Fine Food Reviews](https://www.google.com/search?q=https'://www.kaggle.com/datasets/snap/amazon-fine-food-reviews) dataset from Kaggle. This dataset consists of over 500,000 food reviews from Amazon users, including the review text, a star rating (1-5), and a helpfulness score.

Project Pipeline
----------------

The notebook follows a standard machine learning workflow:

1.  **Exploratory Data Analysis (EDA):** A thorough EDA was conducted to understand the dataset, including distributions, review lengths, and feature correlations.
    
2.  **Data Preprocessing:** The raw text was cleaned by filtering reviews, removing HTML/punctuation, and using **NLTK** for stopword removal and lemmatization.
    
3.  **Modeling:** A Recurrent Neural Network (RNN) using a **Bidirectional LSTM** was built with TensorFlow/Keras to capture the context of the review text.
    
4.  **Training:** The model was trained using a stratified **train-validation-test split**. **Class weights** were applied to handle data imbalance, and **EarlyStopping** was used to prevent overfitting.
    
5.  **Evaluation:** The model's performance was measured using a **Classification Report**, **Confusion Matrix**, and **ROC Curve/AUC Score**.
    
6.  **Inference:** The final notebook includes functions for real-time prediction on single or bulk user inputs.
    

Key Insights & Findings
-----------------------

The exploratory data analysis and model results revealed several key insights into the customer review data:

*   **Highly Imbalanced Data:** The dataset is dominated by positive (5-star) reviews. This class imbalance was addressed during training using **class weights** to ensure the model didn't just learn to predict "positive".
    
*   **Sentiment and Verbosity:** Negative reviews are, on average, slightly longer than positive ones. This suggests that users tend to be more descriptive when they have a negative experience.
    
*   **Polarity Drives Helpfulness:** The most helpful reviews are typically the most polarized. 5-star (very positive) and 1-star (very negative) reviews have the highest average helpfulness scores, as they provide clear signals to other buyers.
    
*   **Distinct Linguistic Patterns:** N-gram analysis showed clear, predictable phrases associated with each sentiment. Positive reviews often contain phrases like "highly recommend," while negative reviews frequently use terms like "waste of money."
    
*   **User Engagement Follows a Power Law:** A small fraction of users are "power reviewers" who contribute a large volume of the reviews in the dataset.
    

Setup & Installation
--------------------

This project uses Python 3. The main libraries can be installed via pip:

Bash

Plain textANTLR4BashCC#CSSCoffeeScriptCMakeDartDjangoDockerEJSErlangGitGoGraphQLGroovyHTMLJavaJavaScriptJSONJSXKotlinLaTeXLessLuaMakefileMarkdownMATLABMarkupObjective-CPerlPHPPowerShell.propertiesProtocol BuffersPythonRRubySass (Sass)Sass (Scss)SchemeSQLShellSwiftSVGTSXTypeScriptWebAssemblyYAMLXML`   pip install pandas numpy tensorflow scikit-learn nltk beautifulsoup4 seaborn matplotlib   `

You will also need to download the necessary NLTK assets, which is handled in the notebook.

Usage
-----

1.  Place the Reviews.csv file in the same directory or update the path in the notebook.
    
2.  Run the cells in the Jupyter/Kaggle notebook sequentially from top to bottom.
    

Model Architecture
------------------

The model is a Recurrent Neural Network (RNN) built with the Keras Functional API.

*   **Input Layer:** Accepts raw text strings.
    
*   **TextVectorization Layer:** Handles tokenization and converts text to integer sequences.
    
*   **Embedding Layer:** Learns a dense vector representation for each word in the vocabulary.
    
*   **Bidirectional LSTM Layer:** Processes the sequence of word embeddings to capture contextual information from both forward and backward directions.
    
*   **Output Layer:** A single Dense neuron with a sigmoid activation function to output a probability score for the positive class.
    

Results
-------

The model was evaluated on a held-out test set and demonstrated strong performance in classifying review sentiment. The final model achieved a weighted average **F1-Score of ~0.90** and an overall **accuracy of ~89%**.

The **ROC Curve** further confirmed the model's excellent discriminative ability, with an **AUC (Area Under the Curve) score of ~0.96**. This indicates a very high probability that the model will rank a random positive review higher than a random negative one.

Model Insights
--------------

*   **Recall-Oriented for Negative Sentiment:** The evaluation shows that the model has high recall but lower precision for the negative class. This means the model is very effective at _identifying_ the majority of actual negative reviews, even at the cost of incorrectly flagging some positive reviews as negative. For a customer service application, this is often a desirable "fail-safe" behavior.
    
*   **Importance of Context:** The success of the Bidirectional LSTM architecture confirms that the sequence and context of words are highly significant for determining sentiment. The model isn't just looking at keywords in isolation but at how they are used together in a sentence.
    
*   **Effectiveness of Class Weighting:** The high recall on the minority (negative) class demonstrates that the class\_weight strategy was crucial. It successfully forced the model to learn the patterns of negative reviews instead of just optimizing for the more frequent positive reviews.
    

Future Work
-----------

The project framework can be extended to tackle more advanced problems, such as:

*   **Fine-tuning Transformer Models** like DistilBERT for higher accuracy.
    
*   **Multitask Learning** to predict star rating and helpfulness simultaneously.
    
*   **Abstractive Text Summarization** to generate a title from the review text.
------------
Feel free to contribute to this code, I'll be really happy to see some commits
