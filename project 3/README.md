**Project 3**. Using APIs and NLP for Prediction of Subreddit posts: r/CryptoCurrency and r/StockMarket

Author: Sileshi Hirpa

I'm not a certified financial planner/advisor nor a certified financial analyst. For the sake of this project, I am just assuming as a marketing data analyst looking to optimize advertising efficiency. My target audience for this project is people who are interested in stock and crypto trading, and my goal is to effectively target this audience using the correct keywords relevant to them. To this end, I have decided to explore Reddit in order to classify posts, based on natural language processing. For this project, I will be focusing on primarily text-based subreddits to enable more accurate text analysis.

Thus, the subreddits I chose are:

1) https://www.reddit.com/r/CryptoCurrency/

2) https://www.reddit.com/r/StockMarket/

Combined, the 2 subreddits have a total of 6,000,000 (4.1M crypto  + 1.9M stock) members. Both have a similar purpose, ie - investing for better returns/profits. Therefore, it will be interesting to see what makes them different, and if machine learning models can accurately classify posts to one or the other.

For the sake of this project, I will be focusing on accurately classifying posts that belong to the CryptoCurrency subreddit group. Therefore, from a data science perspective the optimization parameter for my model should be accuracy.

**Problem Statement**

With stock and cypto investors in mind, I am using Reddit's API for collecting posts from two subreddits, r/CryptoCurrency and r/StockMarket, and use NLP to train a classifier on which subreddit a given post came from.

**Brief Background about Themes in the project:**

**WHAT DOES REDDIT and SUBREDDIT MEAN?**

**Reddit** is a large community made up of thousands of smaller communities. These smaller, sub-communities within Reddit are also known as "subreddits" and are created and moderated by redditors like you.

A **subreddit** is a specific online community, and the posts associated with it, on the social media website Reddit. Subreddits are dedicated to a particular topic that people write about, and they’re denoted by /r/, followed by the subreddit’s name, e.g., /r/gaming ([source](https://www.dictionary.com/e/slang/subreddit/)).

This project is based on two subreddits: CryptoCurrency and StockMarket.

**What is cryptocurrency?**

Cryptocurrency, sometimes called crypto-currency or crypto, is any form of currency that exists digitally or virtually and uses cryptography to secure transactions. Cryptocurrencies don't have a central issuing or regulating authority, instead using a decentralized system to record transactions and issue new units.

Cryptocurrencies run on a distributed public ledger called blockchain, a record of all transactions updated and held by currency holders. There are thousands of cryptocurrencies. Some of the best known include: Bitcoin, Ethereum, Litecoin etc ([source](https://www.kaspersky.com/resource-center/definitions/what-is-cryptocurrency)).

**What is Stock Market?**
At the most basic level, a stock is simply a share of ownership in a company or corporation. There are two types of stock: private and public. The New York Stock Exchange (NYSE) and Nasdaq are the world's biggest stock exchanges. Exchanges are the places and systems were stocks are traded ([source](https://www.quickenloans.com/blog/stock-market-101-stock-market-work)).

This project has three jupyter notbooks:
1. [Data Collection](https://git.generalassemb.ly/sileshith/dsir-111/blob/master/projects/project-03/submission/1-data-collection.ipynb)
2. [Data Cleaning and EDA](https://git.generalassemb.ly/sileshith/dsir-111/blob/master/projects/project-03/submission/2-data-cleaning.ipynb)
3. [Modeling and Evaluation](https://git.generalassemb.ly/sileshith/dsir-111/blob/master/projects/project-03/submission/3-modeling.ipynb)

**Performance metrics**: accuracy and precision

Overview of technical analysis:

**Data collection method**: Data Scraping using Reddit API through **pushshift.io** (resources below). I collected 4000 posts, 2000 CryptoCurrency posts, 2063 StockMarket posts, from 30 days

Exploratory Data Analysis

**Vectorizers used**: CountVectorizer, TfidfVectorizer to create the sparse matrix of features count/frequency respectively, to feed it to the classification model; tokenizer is included in these vectorizers.

**Models used/tested**: Random Forest, Logistic Regression, Support Vector Machine, and Multinomial Naive Bayes.

**Modeling tools used**: Pipelines, and GridSearch.

**Evaluation methods**: accuracy score, cross-validation, precision from classification report, confusion matrix to see False Positives and False Negative, ROC curve to visualize model performance.

My next work will be:

* Setting real-time scraping of the subreddits, make predictions and deploy Sentiment Analysis.

_____________
**References Used**

1. [Natural Language Processing, APIs, and Classification in Python, A Project Walkthrough](https://github.com/hayatoumy/classification_api/blob/master/project_3.ipynb)

2. [Classification API NLP](https://hayatoumy.github.io/hayatoumy/)
