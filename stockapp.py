# %%
#pip install --upgrade pip

# %%
#pip install yfinance

# %%
#pip install fredapi

# %%
#pip install streamlit

# %%
import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, mean_squared_error, r2_score
from sklearn.impute import SimpleImputer

from fredapi import Fred

st.set_page_config(page_title="Stock Market Predictor", layout="wide")

class StockMarketPredictor:
    def __init__(self, fred_api_key, stock_symbol='SPY', start_date='2010-01-01'):
        """
        Initialize the Stock Market Predictor

        Parameters:
        fred_api_key
        stock_symbol
        start_date
        """
        self.fred_api_key = fred_api_key
        self.stock_symbol = stock_symbol
        self.start_date = start_date
        self.end_date = datetime.now().strftime('%Y-%m-%d')
        self.data = None
        self.models = {}
        self.scalers = {}
        self.feature_columns = None

    def collect_stock_data(self):
        """Collect stock data using yfinance"""
        stock = yf.Ticker(self.stock_symbol)
        stock_data = stock.history(start=self.start_date, end=self.end_date)
        stock_data.index = stock_data.index.tz_localize(None)
        stock_data['Returns'] = stock_data['Close'].pct_change()
        stock_data['Log_Returns'] = np.log(stock_data['Close'] / stock_data['Close'].shift(1))
        stock_data['Volatility'] = stock_data['Returns'].rolling(window=21).std()
        stock_data['MA_5'] = stock_data['Close'].rolling(window=5).mean()
        stock_data['MA_20'] = stock_data['Close'].rolling(window=20).mean()
        stock_data['RSI'] = self.calculate_rsi(stock_data['Close'])
        stock_data['Next_Day_Return'] = stock_data['Returns'].shift(-1)
        stock_data['Direction'] = (stock_data['Next_Day_Return'] > 0).astype(int)
        return stock_data[['Close', 'Volume', 'Returns', 'Log_Returns', 'Volatility',
                           'MA_5', 'MA_20', 'RSI', 'Next_Day_Return', 'Direction']]

    def calculate_rsi(self, prices, window=14):
        """Calculate Relative Strength Index"""
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        return rsi

    def collect_economic_data(self):
        """Collect macroeconomic data from FRED"""
        fred = Fred(api_key=self.fred_api_key)
        indicators = {
            'GDP': 'GDP',
            'UNEMPLOYMENT': 'UNRATE',
            'INFLATION': 'CPIAUCSL',
            'FED_RATE': 'FEDFUNDS',
            'VIX': 'VIXCLS',
            '10Y_TREASURY': 'GS10',
            '3M_TREASURY': 'GS3M',
            'CONSUMER_SENTIMENT': 'UMCSENT',
            'INDUSTRIAL_PRODUCTION': 'INDPRO',
            'HOUSING_STARTS': 'HOUST',
            'RETAIL_SALES': 'RSAFS',
            'M2_MONEY_SUPPLY': 'M2SL'
        }
        economic_data = pd.DataFrame()
        for name, code in indicators.items():
            try:
                series = fred.get_series(code, start=self.start_date, end=self.end_date)
                if hasattr(series.index, 'tz') and series.index.tz is not None:
                    series.index = series.index.tz_localize(None)
                series = pd.DataFrame({name: series})
                economic_data = series if economic_data.empty else economic_data.join(series, how='outer')
            except Exception as e:
                st.warning(f"Could not fetch {name}: {e}")
        if 'INFLATION' in economic_data.columns:
            economic_data['INFLATION_RATE'] = economic_data['INFLATION'].pct_change(12) * 100
        if '10Y_TREASURY' in economic_data.columns and '3M_TREASURY' in economic_data.columns:
            economic_data['YIELD_CURVE'] = economic_data['10Y_TREASURY'] - economic_data['3M_TREASURY']
        return economic_data

    def prepare_data(self):
        """Combine and prepare all data for modeling"""
        stock_data = self.collect_stock_data()
        economic_data = self.collect_economic_data()
        economic_data_daily = economic_data.resample('D').ffill()
        combined_data = stock_data.join(economic_data_daily, how='inner')
        imputer = SimpleImputer(strategy='median')
        feature_columns = [col for col in combined_data.columns if col not in ['Next_Day_Return', 'Direction']]
        combined_data[feature_columns] = imputer.fit_transform(combined_data[feature_columns])
        combined_data = combined_data.dropna(subset=['Next_Day_Return', 'Direction'])
        econ_columns = [col for col in combined_data.columns if col not in stock_data.columns]
        for col in econ_columns:
            if col in combined_data.columns:
                combined_data[f'{col}_lag1'] = combined_data[col].shift(1)
                combined_data[f'{col}_lag7'] = combined_data[col].shift(7)
        combined_data = combined_data.dropna()
        self.data = combined_data
        return combined_data

    def visualize_data(self):
        """Create visualizations of the data"""
        if self.data is None or self.data.empty:
            return
        plt.style.use('default')
        fig = plt.figure(figsize=(20, 15))
        plt.subplot(3, 3, 1)
        plt.plot(self.data.index, self.data['Close'], label='Close Price', alpha=0.7)
        plt.plot(self.data.index, self.data['MA_5'], label='5-day MA', alpha=0.8)
        plt.plot(self.data.index, self.data['MA_20'], label='20-day MA', alpha=0.8)
        plt.title(f'{self.stock_symbol} Price and Moving Averages')
        plt.legend()
        plt.xticks(rotation=45)
        plt.subplot(3, 3, 2)
        plt.hist(self.data['Returns'].dropna(), bins=50, alpha=0.7, edgecolor='black')
        plt.title('Daily Returns Distribution')
        plt.xlabel('Daily Returns')
        plt.ylabel('Frequency')
        plt.subplot(3, 3, 3)
        plt.plot(self.data.index, self.data['Volatility'])
        plt.title('Volatility Over Time')
        plt.xticks(rotation=45)
        plt.subplot(3, 3, 4)
        if 'FED_RATE' in self.data.columns:
            plt.plot(self.data.index, self.data['FED_RATE'], label='Fed Rate')
        if 'UNEMPLOYMENT' in self.data.columns:
            plt.plot(self.data.index, self.data['UNEMPLOYMENT'], label='Unemployment')
        plt.title('Key Economic Indicators')
        plt.legend()
        plt.xticks(rotation=45)
        plt.subplot(3, 3, 5)
        if 'YIELD_CURVE' in self.data.columns:
            plt.plot(self.data.index, self.data['YIELD_CURVE'])
            plt.title('Yield Curve (10Y - 3M)')
            plt.xticks(rotation=45)
        plt.subplot(3, 3, 6)
        if 'VIX' in self.data.columns:
            plt.plot(self.data.index, self.data['VIX'], color='red', alpha=0.7)
            plt.title('VIX (Fear Index)')
            plt.xticks(rotation=45)
        plt.subplot(3, 3, 7)
        correlation_data = self.data.select_dtypes(include=[np.number]).corr()
        important_features = ['Returns', 'FED_RATE', 'UNEMPLOYMENT', 'VIX', 'YIELD_CURVE', 'INFLATION_RATE']
        available_features = [f for f in important_features if f in correlation_data.columns]
        if len(available_features) > 1:
            corr_subset = correlation_data.loc[available_features, available_features]
            sns.heatmap(corr_subset, annot=True, cmap='coolwarm', center=0, fmt='.2f')
            plt.title('Feature Correlations')
        plt.subplot(3, 3, 8)
        direction_counts = self.data['Direction'].value_counts()
        plt.bar(['Down', 'Up'], direction_counts.values, color=['red', 'green'], alpha=0.7)
        plt.title('Direction Distribution')
        plt.ylabel('Frequency')
        plt.subplot(3, 3, 9)
        if 'VIX' in self.data.columns:
            plt.scatter(self.data['VIX'], self.data['Returns'], alpha=0.5)
            plt.xlabel('VIX')
            plt.ylabel('Returns')
            plt.title('Returns vs VIX')
        plt.tight_layout()
        st.pyplot(fig)

    def prepare_features(self):
        """Prepare features for modeling"""
        exclude_columns = ['Next_Day_Return', 'Direction', 'Close']
        feature_columns = [col for col in self.data.columns if col not in exclude_columns]
        X = self.data[feature_columns]
        y_reg = self.data['Next_Day_Return']
        y_clf = self.data['Direction']
        return X, y_reg, y_clf, feature_columns

    def train_classification_model(self, X, y):
        """Train classification model to predict market direction"""
        split_idx = int(len(X) * 0.8)
        X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
        y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        self.scalers['classification'] = scaler
        rf_clf = RandomForestClassifier(n_estimators=100, random_state=42, max_depth=10)
        rf_clf.fit(X_train_scaled, y_train)
        lr_clf = LogisticRegression(random_state=42, max_iter=1000)
        lr_clf.fit(X_train_scaled, y_train)
        rf_score = rf_clf.score(X_test_scaled, y_test)
        lr_score = lr_clf.score(X_test_scaled, y_test)
        if rf_score > lr_score:
            best_model = rf_clf
            model_name = "Random Forest"
            best_pred = best_model.predict(X_test_scaled)
            best_score = rf_score
        else:
            best_model = lr_clf
            model_name = "Logistic Regression"
            best_pred = best_model.predict(X_test_scaled)
            best_score = lr_score
        self.models['classification'] = {'model': best_model, 'name': model_name, 'score': best_score}
        report = classification_report(y_test, best_pred)
        return model_name, best_score, report

    def train_regression_model(self, X, y):
        """Train regression model to predict returns"""
        split_idx = int(len(X) * 0.8)
        X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
        y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        self.scalers['regression'] = scaler
        rf_reg = RandomForestRegressor(n_estimators=100, random_state=42, max_depth=10)
        rf_reg.fit(X_train_scaled, y_train)
        lr_reg = LinearRegression()
        lr_reg.fit(X_train_scaled, y_train)
        rf_pred = rf_reg.predict(X_test_scaled)
        lr_pred = lr_reg.predict(X_test_scaled)
        rf_r2 = r2_score(y_test, rf_pred)
        lr_r2 = r2_score(y_test, lr_pred)
        rf_mse = mean_squared_error(y_test, rf_pred)
        lr_mse = mean_squared_error(y_test, lr_pred)
        if rf_r2 > lr_r2:
            best_model = rf_reg
            model_name = "Random Forest"
            best_r2 = rf_r2
        else:
            best_model = lr_reg
            model_name = "Linear Regression"
            best_r2 = lr_r2
        self.models['regression'] = {'model': best_model, 'name': model_name, 'r2': best_r2}
        return model_name, best_r2, {'rf': (rf_r2, rf_mse), 'lr': (lr_r2, lr_mse)}

    def feature_importance(self, feature_columns):
        """Display feature importance"""
        out = {}
        if hasattr(self.models['classification']['model'], 'feature_importances_'):
            out['classification'] = pd.DataFrame({
                'feature': feature_columns,
                'importance': self.models['classification']['model'].feature_importances_
            }).sort_values('importance', ascending=False).head(15)
        if hasattr(self.models['regression']['model'], 'feature_importances_'):
            out['regression'] = pd.DataFrame({
                'feature': feature_columns,
                'importance': self.models['regression']['model'].feature_importances_
            }).sort_values('importance', ascending=False).head(15)
        return out

    def make_predictions(self):
        """Make future predictions"""
        latest_features = self.data.iloc[-1][self.feature_columns].values.reshape(1, -1)
        clf_features_scaled = self.scalers['classification'].transform(latest_features)
        reg_features_scaled = self.scalers['regression'].transform(latest_features)
        direction_pred = self.models['classification']['model'].predict(clf_features_scaled)[0]
        return_pred = self.models['regression']['model'].predict(reg_features_scaled)[0]
        direction_prob = None
        if hasattr(self.models['classification']['model'], 'predict_proba'):
            direction_prob = self.models['classification']['model'].predict_proba(clf_features_scaled)[0]
        return direction_pred, return_pred, direction_prob

st.title("Stock Market Predictor")

with st.sidebar:
    symbol = st.text_input("Symbol", value="SPY")
    start_date = st.date_input("Start date", value=datetime(2015, 1, 1))
    run_btn = st.button("Run")

if run_btn:
    if "FRED_API_KEY" not in st.secrets:
        st.error("Missing FRED_API_KEY in Secrets.")
        st.stop()

    predictor = StockMarketPredictor(
        fred_api_key=st.secrets["FRED_API_KEY"],
        stock_symbol=symbol,
        start_date=str(start_date)
    )

    with st.status("Preparing data...", expanded=True) as status:
        data = predictor.prepare_data()
        st.write(f"Final dataset: {data.shape[0]} rows × {data.shape[1]} cols")
        predictor.visualize_data()
        X, y_reg, y_clf, feature_columns = predictor.prepare_features()
        predictor.feature_columns = feature_columns
        status.update(label="Data ready", state="complete")

    c1, c2 = st.columns(2)
    with c1:
        with st.status("Training classification...", expanded=True) as s1:
            name, acc, report = predictor.train_classification_model(X, y_clf)
            st.write(f"Best model: {name}")
            st.write(f"Accuracy: {acc:.4f}")
            st.text(report)
            s1.update(label="Classification done", state="complete")
    with c2:
        with st.status("Training regression...", expanded=True) as s2:
            rname, r2, scores = predictor.train_regression_model(X, y_reg)
            st.write(f"Best model: {rname}")
            st.write(f"R²: {r2:.4f}")
            st.write(f"RandomForest (R², MSE): {scores['rf']}")
            st.write(f"LinearRegression (R², MSE): {scores['lr']}")
            s2.update(label="Regression done", state="complete")

    fi = predictor.feature_importance(feature_columns)
    if fi:
        st.subheader("Feature importance")
        if 'classification' in fi:
            st.write("Classification")
            st.dataframe(fi['classification'], use_container_width=True)
        if 'regression' in fi:
            st.write("Regression")
            st.dataframe(fi['regression'], use_container_width=True)

    st.subheader("Next session prediction")
    direction_pred, return_pred, direction_prob = predictor.make_predictions()
    st.metric("Direction", "UP" if direction_pred == 1 else "DOWN")
    st.metric("Expected return", f"{return_pred*100:.2f}%")
    if direction_prob is not None:
        st.write(f"Probabilities → Down: {direction_prob[0]:.3f} | Up: {direction_prob[1]:.3f}")

    st.download_button(
        "Download dataset (CSV)",
        data=predictor.data.to_csv().encode("utf-8"),
        file_name=f"dataset_{symbol}.csv",
        mime="text/csv"
    )
else:
    st.info("Set parameters and press Run.")





