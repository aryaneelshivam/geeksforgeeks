
# Stock Analyzer

## PROBLEM STATEMENT
- Financial analysis software are too complicated for the common people.
- The existing solutions are too technical for a beginner to start.
- The current tools available on the market are not too user friendly either.
- Steep learning curve.
- Inaccessibility of such software, and hidden behind hefty paywalls
- Only accessible and usable by specialized technical analysts
- This gap in the market, is in need of an innovative tool that simplify the usage and expose the market to a new user pool irrespective of their age and skill level.

## OVERVIEW

Combination of multiple prebuilt technical analysis indictors are used in backend, which pulls in company financial data and use sophisticated prompt engineering and generative AI to give a rank or score after thorough analysis. Usage of LLM (Large Language Models) to parse the given data into simplified consumer-friendly information. The programme also uses NLP (Natural Language Processing) to analyze market sentiment to adjust the given data. It scrapes public data from NATIONAL STOCK ECHANGE and parse it to structured database. This data is used in order to calculate PE (Profit to Expense) Ratio, Debt to Equity ratio, Earnings per share in the backend. Using Retrieval Augmented Generation (RAG) for conversational data analysis and Gemini generative Ai wrapper for visual chart analysis. It uses multiple technical analysis visual indicators to show possible uptrends and downtrends.

The programme also exploratory data analysis and manipulation, and highly interactive dynamic graphs and data visualisation. Thus leading to a safe and reliable platform for stock enthusiast irrespective of their age and skill level.





## Screenshots

![Web App Screenshot](https://i.postimg.cc/g28tX1s8/temp-Image-Dmu-MSD.avif)

![Web App Screenshot](https://i.postimg.cc/9MzxVpfn/ss2.jpg)

![Web App Screenshot](https://i.postimg.cc/jjM8G9Rf/ss1.png)



## Installation

### Initialize virtual environment

```
python -m venv .venv
```
```
# Windows command prompt
.venv\Scripts\activate.bat

# Windows PowerShell
.venv\Scripts\Activate.ps1

# macOS and Linux
source .venv/bin/activate
```

## Requrements
- streamlit
- numpy
- pandas
- yfinance
- datetime
- tradingview-ta
- streamlit-shadcn-ui
- plotly express
- llama-index
- openai
- llama-index 
- gemini
- googlegenAI
- PIL
    
## Authors

- [@aryaneelshivam](https://github.com/aryaneelshivam)
- [@gautam](https://github.com/ryuiiji)


## REFERENCES AND RESOURCES:

1: Pandas and Numpy for Exploratory data analysis
2: [Plotly](https://plotly.com) express for interactive data visualization.
3: [Llama-Index for Retrieval Augmented generation.]( https://www.llamaindex.ai)
4: [Streamlit]( https://streamlit.io) for web-app development.
5: Google Gemini as a LLM wrapper
6: [Yahoo Finance](https://pypi.org/project/yfinance/) for historical stock data.
7: National Stock Exchange API for company profile scraping
8: Trading View API for market sentiment analysis
