# Stock Market Analysis Based on Economic Conditions

Our AI project is a supervised ML model that predicts both the direction (UP/DOWN) and the magnitude (in %) of future fluctuations in stock market prices based on several macroeconomic factors (including FED Rate, VIX, Unemployment rate, Inflation, and several others), as well as historical stock patterns.

## Problem Statement
The randomness of the fluctuations in stock market prices are notorious for being hard to predict, understand, or even make sense of. This sometimes puts off people from even trying to engage in stock trading and even finance professionals find it hard to find success in the highly unstable stock environment. This project aims to address this by leveraging the powerful tools offered by artificial intelligence and machine learning to extract insights from past stock data and present economic conditions, and use them to predict future changes in stock market prices. This can potentially help enlighten the public on how the stock market operates, and arm them with the necessary information needed to make more informed trading decisions.

## Key Results

1. *The Random Forest classifier achieved the highest classification accuracy at 54.2%, outperforming Logistic Regression (52.8%) and random guessing (50%).*
2. *In regression, Random Forest had an $R^2$ score of 0.035, explaining 25% more variance than Linear Regression, which had an $R^2$ score of 0.028.*
3. *The Random Forest model correctly predicted 501 out of 886 cases, with balanced precision (56.4%) and recall (57.8%).*

## Methodologies

We collected 10 years of financial data (2015-2025) and removed timezone conflicts between datasets (stock market includes timezones, FRED does not). We also converted monthly economic data to daily (forward-fill). Then we created new variables 
Feature Engineering: Creates new variables like:
- Yield curve (10Y - 3M rates)
- Inflation rate (year-over-year change)
- Lag features (economic data from 1 and 7 days ago)
Missing Data: Fills gaps using median values
Scaling: Normalizes all features to same scale for machine learning


## Data Sources

(UPDATE IN README.md)
Include any relevant data sources that were used in your project.

*EXAMPLE:*
*Kaggle Datasets: [Link to Kaggle Dataset](https://www.kaggle.com/datasets)*

## Technologies Used <!--- do not change this line -->

(UPDATE IN README.md)
List the technologies, libraries, and frameworks used in your project.

*EXAMPLE:*
- *Python*
- *pandas*
- *OpenAI API*


## Authors <!--- do not change this line -->

(UPDATE IN README.md)
List the names and contact information (e.g., email, GitHub profiles) of the authors or contributors.

*EXAMPLE:*
*This project was completed in collaboration with:*
- *John Doe ([john.doe@example.com](mailto:john.doe@example.com))*
- *Jane Smith ([jane.smith@example.com](mailto:jane.smith@example.com))*
