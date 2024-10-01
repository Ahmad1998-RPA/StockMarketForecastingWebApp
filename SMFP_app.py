# Importing Libraries
import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
import plotly.express as px
import datetime
from datetime import date, timedelta
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.stattools import adfuller
import statsmodels.api as sm


# Title
app_name = "Stock Market Forecasting Portfolio"
st.title(app_name)
# st.subheader('This app is created to forecast the stock market price.')
st.write('This app is created to forecast the stock market price.')
# add an image from online resource
st.image("https://indiaforensic.com/wp-content/uploads/2012/12/Stock-market12.jpg")

# Take the input from the user of app about the start and end date

# Sidebar
st.sidebar.header('Select the parameters from the below')
start_date = st.sidebar.date_input('Start date', date(2024,1,1))
end_date = st.sidebar.date_input('End date', date(2024,12,31))

# Add ticker symbol list
ticker_list = ["AAPL", "MSFT", "GOOG", "GOOGL", "META", "TSLA", "NVDA", "ADBE", "PYPL", "INTC", "CMCSA", "NFLX", "PEP"]
ticker = st.sidebar.selectbox('Select the company',ticker_list)

# Fetch data from user inputs using yfinance library
data = yf.download(ticker, start=start_date, end=end_date)
# Add date as a column to the dataframe
data.insert(0,"Date",data.index,True)
data.reset_index(drop=True,inplace=True)
st.write('Data from', start_date, 'to', end_date)
st.write(data)

# Plot the data
st.header('Data Visualization')
st.subheader('Plot of the data')
st.write("**Note:** Select your specific date range on the sidebar or zoom in on the plot and select your specific column")
fig = px.line(data, x='Date', y=data.columns, title='Closing price of the stock', width=1000, height=600)
st.plotly_chart(fig)

# Add a select box to select column from data
column = st.selectbox('Select the column to be used for forecasting', data.columns[1:])

# subsetting the data
data = data[['Date', column]]
st.write("Selected Data")
st.write(data)

# Add test check Stationarity
st.header('Is data Stationary?')
# st.write('**Note:** If p-value is less than 0.05, then data is stationary')
st.write(adfuller(data[column])[1]<0.05)

# Decompose the data
st.header('Decomposition of the data')
decomposition = seasonal_decompose(data[column], model='additive', period=12)
st.write(decomposition.plot())
# Same plot in plotly
st.plotly_chart(px.line(x=data["Date"], y=decomposition.trend, title='Trend', width=1200, height=400, labels={'x':'Date','y':'Price'}).update_traces(line_color='Blue'))
st.plotly_chart(px.line(x=data["Date"], y=decomposition.seasonal, title='Seasonality', width=1200, height=400, labels={'x':'Date','y':'Price'}).update_traces(line_color='green'))
st.plotly_chart(px.line(x=data["Date"], y=decomposition.resid, title='Residuals', width=1200, height=400, labels={'x':'Date','y':'Price'}).update_traces(line_color='red', line_dash='dot'))

# Lets run the model
# User input for three parameters of the model and seasonal order
p = st.slider('Select the value of p', 0, 5, 2)
d = st.slider('Select the value of d', 0, 5, 1)
q = st.slider('Select the value of q', 0, 5, 2)
seasonal_order = st.number_input('Select the value of seasonal', 0, 24, 12)

model = sm.tsa.statespace.SARIMAX(data[column], order=[p,d,q], seasonal_order=(p,d,q,seasonal_order))
model = model.fit()

# Print model summary
st.header('Model Summary')
st.write(model.summary())
st.write("-----")

# predict the future values (Forecasting)
st.write("<p style='color:green; font-size: 50px; font-weight: bold:'>Forecasting the data</p>",unsafe_allow_html=True)
forecast_period = st.number_input('Select the number of days to forecast', 1, 365, 10)
# Prediction
predictions = model.get_prediction(start=len(data), end=len(data)+forecast_period)
predictions = predictions.predicted_mean
# st.write(len(predictions))
# Add index to results dataframe as dates
predictions.index = pd.date_range(start=end_date, periods=len(predictions), freq='D')
predictions = pd.DataFrame(predictions)
predictions.insert(0, 'Date', predictions.index)
predictions.reset_index(drop=True, inplace=True)
st.write("## Predictions", predictions)
st.write("## Actual Data", data)

# lets plot the data
fig = go.Figure()
# Add actual data to the plot
fig.add_trace(go.Scatter(x=data["Date"], y=data[column], mode='lines', name='Actual', line=dict(color='blue')))
# Add predicted data to the plot
fig.add_trace(go.Scatter(x=predictions["Date"], y=predictions["predicted_mean"], mode='lines', name='Predicted', line=dict(color='red')))
# Set the title and axis labels
fig.update_layout(title='Actual vs Predicted', xaxis_title='Date', yaxis_title='Price', width=1000, height=400)
# Display the plot
st.plotly_chart(fig)

# Add Buttons to show or hide separate plots
show_plots = False
if st.button('Show Separate Plots'):
    if not show_plots:
        st.write(px.line(x=data["Date"], y=data[column], title='Actual', width=1200, height=400, labels={'x':'Date','y':'Price'}).update_traces(line_color='Blue'))
        st.write(px.line(x=predictions["Date"], y=predictions["predicted_mean"], width=1200, height=400, labels={'x':'Date','y':'Price'}).update_traces(line_color='red', line_dash='dot'))
        show_plots = True
    else:
        show_plots = False

hide_plots = False
if st.button("Hide Separate Plots"):
    if not hide_plots:
        hide_plots = True
    else:
        hide_plots = False

st.write("---")
st.write("###### About the Author:")

st.write("<p style='color:green; font-size: 25px; font-weight: bold:'>Ahmad Mubarak</p>",unsafe_allow_html=True)

# Paste Youtube icon from online source with link
st.write("##### Connect with me on Social Media")
# Add links to my social Media
# urls
linkedin_url = "https://img.icons8.com/color/48/000000/linkedin.png"
github_url = "https://img.icons8.com/fluent/48/000000/github.png"
youtube_url = "https://img.icons8.com/color/48/000000/youtube-play.png"
twitter_url = "https://img.icons8.com/color/48/000000/twitter.png"

# Redirect Urls
linkedin_redirect_url = "https://www.linkedin.com/in/ahmad-mubarak-19861a177"
github_redirect_url = "https://github.com/Ahmad1998-RPA"
youtube_redirect_url = "http://www.youtube.com/@ahmaddhanesar"
twitter_redirect_url = "https://x.com/ahmad_mubarak01"

# Add links to the images
st.markdown(f'<a href="{github_redirect_url}"><img src="{github_url}" width="60" height="60"></a>'
            f'<a href="{linkedin_redirect_url}"><img src="{linkedin_url}" width="60" height="60"></a>'
            f'<a href="{youtube_redirect_url}"><img src="{youtube_url}" width="60" height="60"></a>'
            f'<a href="{twitter_redirect_url}"><img src="{twitter_url}" width="60" height="60"></a>', unsafe_allow_html=True)

