# Stock Market Analysis Based on Economic Conditions

Our AI project is a supervised ML model that predicts both the direction (UP/DOWN) and the magnitude (in %) of future fluctuations in stock market prices based on several macroeconomic factors (including FED Rate, VIX, Unemployment rate, Inflation, and several others), as well as historical stock patterns.

## Problem Statement
The randomness of the fluctuations in stock market prices are notorious for being hard to predict, understand, or even make sense of. This sometimes puts off people from even trying to engage in stock trading and even finance professionals find it hard to find success in the highly unstable stock environment. This project aims to address this by leveraging the powerful tools offered by artificial intelligence and machine learning to extract insights from past stock data and present economic conditions, and use them to predict future changes in stock market prices. This can potentially help enlighten the public on how the stock market operates, and arm them with the necessary information needed to make more informed trading decisions.

## Key Results

1. *The Random Forest classifier achieved the highest classification accuracy at 54.2%, outperforming Logistic Regression (52.8%) and random guessing (50%).*
2. *In regression, Random Forest had an $R^2$ score of 0.035, explaining 25% more variance than Linear Regression, which had an $R^2$ score of 0.028.*
3. *The Random Forest model correctly predicted 501 out of 886 cases, with balanced precision (56.4%) and recall (57.8%).*

## Methodologies

We collected 10 years of financial data (2015-2025) and removed timezone conflicts between datasets (stock market includes timezones, FRED does not). We also converted monthly economic data to daily (forward-fill). Then we created new variables (such as yield curve (10Y - 3M rates), inflation rate (year-over-year change), and lag features (economic data from 1 and 7 days ago)). Next, we filled any gaps by using median values, and finally we normalized all features to same scale for machine learning purposes.

The machine learning process was itself composed of two different prediction tasks:

1) Classification Model (Direction Prediction)
Goal: To predict if the stock would go UP (1) or DOWN (0) tomorrow
Models Tested: Random Forest Classifier (Uses 100 decision trees, votes on outcome) and Logistic Regression (uses mathematical formula with probabilities)
How it works: Looks at patterns like "When unemployment falls AND VIX is low AND Fed rates are rising, stocks usually go UP". Automatically picks the more accurate model

2) Regression Model (Amount Prediction)
Goal: Predict HOW MUCH the stock will move (e.g., +0.8% or -1.2%)
Models Tested: Random Forest Regressor (Uses 100 decision trees, averages predictions) and Linear Regression (Uses mathematical equation with weighted factors)
How it works: Learns relationships like "Each 1% drop in unemployment = +0.1% stock return". Combines all economic factors into single prediction

## Data Sources
 - yfinance dataset: https://pypi.org/project/yfinance/ (source of daily stock prices such as open, high, low, close, volume)
 - FRED API: https://fredaccount.stlouisfed.org/apikeys (for various samples of economic data, such as GDP, Unemployment Rate, Inflation, Federal Funds Rate, VIX, Treasury Rate, Consumer Sentiment, Industrial Production, Housing Starts, etc)

## Technologies Used
Language: Python
Libraries/Modules: Pandas, NumPy, Matplotlib, Seaborn, yfinance, Sklearn, etc
Technologies: Google Colab, Git/GitHub

## Authors
- Ayomide Olumide-Attah ([AyomideOlumideAttah](https://github.com/AyomideOlumideAttah))
- Piero Espinoza ([PieroEB](https://github.com/PieroEB))
- Aleena Siddiqui ([aleenasid12](https://github.com/aleenasid12))
- Yerlin Holguin ([yerlinh](https://github.com/yerlinh))
