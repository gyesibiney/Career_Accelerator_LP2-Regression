#!/usr/bin/env python
# coding: utf-8

# ## Setup

# ## Title 

# ### Building Accurate Models for Unit Sales Prediction in Favorita Stores using Time Series Forecasting

# ### Description of the project 

# A time series is a sequence of data points that are indexed and ordered chronologically over time. Time series data can be observed at regular or irregular intervals and can be collected over any time period, ranging from seconds to decades or even centuries. Time series data can be univariate (i.e., a single variable is recorded over time) or multivariate (i.e., multiple variables are recorded over time).Favorita Corporation is an Ecuadorian company that creates and invests in the commercial, industrial and real estate areas. Its subsidiaries have activities in six countries in the region, including Panama, Paraguay, Peru, Colombia, Costa Rica and Chile. They offer the best products, services and experiences in an efficient, sustainable and responsible way to improve the quality of life. 
# 
# The aim of this project is to forecast the unit sales of the products across different stores. This is to optimize their inventory management , marketing strategies  and pricing  decisions. To achieve this results, we employed the use of time series in collaboration with different machine learning algorithms via the CRISP-DM framework. 
# 
# The objective of this analysis is to select the best prediction model from the different ML algorithms tested. This model will be the solution to be adopted by the company to help Favorita Corporation make insightful decisions in relation to their retail sales ,promotion and customer statisfaction. 

# ### Hypothesis 

# - Null Hypothesis:
# Promotional activities have a significant impact on the store sales at Corporation Favorita.
# 
# - Alternate Hypothesis:
# Promotional activities does not have a significant impact on the store sales at Corporation Favorita.
# 

# ### Questions 

# Q1. which city had the highest stores
# 
# Q2. which month did we have the highest sale
# 
# Q3. which store had the highest transaction?
# 
# Q4. which store had the highest sale?
# 
# Q5. what is the most bought product?

# ### Installation

# In[1]:


# Summarytools installed 
# Pandas installed 
#Matplotlib installed 
#Seaborn Installed
#Numpy installed 


# ### Importation

# Here is the section to import all the packages/libraries that will be used through this notebook.

# In[2]:


# Data handling
import pandas as pd 
import numpy as np
# Vizualisation (Matplotlib, Plotly, Seaborn, etc. )
import matplotlib.pyplot as plt 
import seaborn as sns 
import plotly.express as ex
import plotly.offline as po
import plotly.graph_objects as go
from matplotlib import dates as mpl_dates
from datetime import datetime
# EDA (pandas-profiling, etc. )
...

# Feature Processing (Scikit-learn processing, etc. )
...

# Machine Learning (Scikit-learn Estimators, Catboost, LightGBM, etc. )
...

# Hyperparameters Fine-tuning (Scikit-learn hp search, cross-validation, etc. )
...

# Other packages
import os
import warnings 
warnings.filterwarnings("ignore")
import summarytools as dfSummary 


# ### Data Loading

# Here is the section to load the datasets (train, eval, test) and the additional files

# In[3]:


#Creating a function to parse our date
def convert_to_date(w):
    return datetime.strptime(w,"%Y-%m-%d")


# In[4]:


# For CSV, use pandas.read_csv
holidays_events_data= pd.read_csv('./Documents/lp2\holidays_events.csv',parse_dates=['date'])
oil_data = pd.read_csv('./Documents/lp2\oil.csv',parse_dates=['date'])
sample_submission_data =pd.read_csv('./Documents/lp2/sample_submission.csv')
stores_data = pd.read_csv('./Documents/lp2\stores.csv')
test_data = pd.read_csv('./Documents/lp2/test.csv',parse_dates=['date'])
train_data = pd.read_csv('./Documents/lp2/train.csv',parse_dates=['date'])
transactions_data= pd.read_csv('./Documents/lp2/transactions.csv',parse_dates=['date'])


# ### Exploratory Data Analysis: EDA

# Here is the section to inspect the datasets in depth, present it, make hypotheses and think the cleaning, processing and features creation.

# ### Dataset overview

# Have a look at the loaded datsets using the following methods: .head(), .info()

# In[5]:


holidays_events_data.head()


# In[6]:


holidays_events_data.info()


# In[7]:


holidays_events_data.shape


# In[8]:


oil_data.head()


# In[9]:


oil_data.info()


# In[10]:


oil_data.shape


# In[11]:


sample_submission_data.head() 


# In[12]:


sample_submission_data.info() 


# In[13]:


sample_submission_data.shape


# In[14]:


stores_data.head()


# In[15]:


stores_data.info()


# In[16]:


stores_data.shape


# In[17]:


test_data.head()


# In[18]:


test_data.info()


# In[19]:


test_data.shape 


# In[20]:


train_data.head()


# In[21]:


train_data.info()


# In[22]:


train_data.shape 


# In[23]:


transactions_data.head()


# In[24]:


transactions_data.info()


# In[25]:


transactions_data.shape 


# - Date columns with datatype as objects in the holidays_events_data,oil_data,test_data,train_data, and transcation_data were converted to datetime64[ns].

# ### Univariate Analysis

# APPROACH:
# -Identify the variable of interest
# -Check for missing values in the variable
# -Plot the distribution of the variable using histograms or density plots
# -Compute descriptive statistics like mean, median, mode, variance, and standard deviation
# -Analyze the shape of the distribution (e.g., normal, skewed, or bimodal)
# -Check for outliers and extreme values
# -Identify any patterns or trends in the data

# #### Checking For Missing Values

# In[26]:


# Counts the number of any missing values in each column
stores_data.isna().sum()


# In[27]:


# Counts the number of any missing values in each column
oil_data.isna().sum()


# It was discovered that there were 43 missing entries in the oil_data's dcoilwtico column.The method = 'bfill' was used to fill in the missing values.

# In[28]:


# Filling the missing values using 'bfill' method
oil_data ['dcoilwtico']= oil_data['dcoilwtico'].fillna(method='bfill')
oil_data
# Checks if there are any missing values in a dataframe
oil_data.isnull().any()


# In[29]:


# Counts the number of any missing values in each column
holidays_events_data.isna().sum()


# In[30]:


# Checks if there are any missing values in a dataframe
train_data.isnull().any()


# In[31]:


# Counts the number of any missing values in each column
test_data.isna().sum()


# ### Exploring the stores dataset 

# #### Plot the distribution of the variable

# In[32]:


stores_data.head()


# In[33]:


sns.pairplot(stores_data, hue="city");


# - from the visualisation above, it was noticed that quito had more stores clustered in the city.

# In[34]:


sns.countplot( x="cluster",data = stores_data)
plt.title("Count of Cluster")
plt.show();


# In[ ]:


- cluster 3 has the highest count as shown above.


# In[35]:


sns.countplot(x="state",data=stores_data)

plt.title("Count of Stores Across Various States")
plt.xticks(rotation=90)
plt.figure(figsize= (15,15))

plt.show();


# - There are 16 unique states with Pichincha having the highest number of stores

# In[36]:


stores_data.groupby('city')['store_nbr'].sum().sort_values(ascending=False).reset_index(name='count')


# In[37]:


sns.countplot(x="city", data=stores_data)

plt.title("Count of Stores Across Various Cities")
plt.xticks(rotation=90)
plt.figure(figsize= (15,15));
plt.show();


# - There are 22 unique cities with Quito having the highest count

# In[38]:


len(stores_data.store_nbr.unique())
##There are 54 unique stores


# In[39]:


sns.countplot(x="type", data=stores_data)
plt.title("Count of Store Types")
plt.show();


# - 5 unique store types with store D having the highest count

# In[40]:


# Compute descriptive statistics for state
print(stores_data['state'].describe());


# In[41]:


# Analyze the shape of the distribution of state
print(stores_data.skew())


# - this informs on how the dataset is positively skewed

# In[42]:


# Check for outliers
stores_data.boxplot();


# - there are no outliers in the store_data dataset

# In[43]:


# Plot the distribution of the store_number 
sns.displot(data=stores_data, x='store_nbr', bins=20, kde=True);
plt.xticks(rotation=90);
plt.title("Distribution of the store_nbr");


# In[44]:


stores_data.hist(figsize=(5,2));


# In[45]:


# Compute descriptive statistics for stores_data
print(stores_data.describe())


# In[46]:


# Identify any patterns or trends
sns.lineplot(data=stores_data, x='store_nbr', y='cluster');
plt.title("Trend of store_nbr vs cluster");


# - the visualisation shows a slightly positive trend between how stores numbers are being clustered

# In[47]:


# Compute descriptive statistics for city
pd.DataFrame(stores_data['city'].describe())


# In[48]:


# Analyze the shape of the distribution for stores_data
stores_data.skew()


# #### NOTES;
# 0.0412413665900577 indicates that the 'cluster' column has a slightly positive skew, which means that the tail of 
# the distribution is slightly longer on the right side. However, since the value is close to zero,it suggests that 
# the distribution is almost symmetrical.

# In[49]:


# Identify any patterns or trends
sns.pointplot(data=stores_data, x='city', y='cluster');
plt.xticks(rotation=90);
plt.title("city vs cluster");


# In[50]:


# Identify any patterns or trends
sns.barplot(data=stores_data, x='state', y='cluster');
plt.xticks(rotation=90);
plt.title("state vs cluster");


# In[51]:


stores_data.groupby('state')['cluster'].sum().sort_values(ascending=False).reset_index(name='count')


# In[52]:


# Identify any patterns or trends
sns.barplot(data=stores_data, x='state', y='store_nbr');
plt.xticks(rotation=90);
plt.title("state vs store_nbr");


# In[53]:


stores_data.groupby('state')['store_nbr'].value_counts().sort_values(ascending=False)


# In[54]:


stores_data.groupby('state')['store_nbr'].unique()


# In[55]:


stores_data.groupby('type')['cluster'].unique()


# In[56]:


# Identify any patterns or trends
sns.barplot(data=stores_data, x='type', y='cluster');
plt.xticks(rotation=90);
plt.title("type vs store_nbr");


# - type D is having the highest cluster

# In Summary:
# 
# - There are 17 unique clusters with cluster 3 having the highest count
# - There are 16 unique states
# - There are 22 unique cities with Quito having the highest count
# - There are 54 unique stores Across 16 states and 22 cities
# - There are 5 unique store types with stores type D having the highest count

# # transactions_data overview

# In[57]:


transactions_data.head()


# In[58]:


transactions_data.info()


# In[59]:


# Distribution of the transaction dataset
transactions_data.hist(figsize=(7,5));


# In[60]:


# checking for skewness
transactions_data.skew()


# - the transaction dataset is positively skewwed to the right

# In[61]:


## making a copy 
trans= transactions_data.copy()


# In[62]:


# setting the date as index
transactions= transactions_data.set_index("date")
transactions.head()


# In[63]:


# checking for trends and patterns
fig= ex.line(x= transactions.index, y= "transactions", data_frame= transactions)
fig.update_xaxes(rangeslider_visible=True)
fig.show()


# - there is a positive trend in transactions through the years.

# In[64]:


##Let's check for outliers
transactions.boxplot();


# -there are outliers on the transactions attribute

# In[65]:


# Compute descriptive statistics for transactions_data
pd.DataFrame(transactions_data.describe())


# - in Summary;
# - the transactions dataset has a minimum of 5 transactions and a maximum of 8359 transactions
# - transactions attribute trends positively over time

# # Exploring the Holidays Dataset

# In[66]:


holidays_events_data.head()


# In[67]:


holidays=holidays_events_data.copy()
type_counts = holidays.groupby('type').size().reset_index(name='count')
type_counts


# In[68]:


fig = ex.scatter(type_counts, x='type', y='count', size='count', color= "type", hover_name='type',log_y=False, size_max=60)

fig.show()


# - from the visual , holiday is having the highest size of 221 counts and work day and bridge with smallest sizes of 5 counts respectively

# In[69]:


##Let's see which locale has the highest number of holidays
plt.title("Count of Holiday Locale")
sns.countplot("locale", data= holidays)
plt.show()


# - from the countplot, National and Regional locale have the highest count and lowest count respectively 

# In[70]:


# holidays that were trabsfered
trans_count= holidays.groupby("transferred").size().reset_index(name='count')
trans_count


# In[71]:


sns.set_color_codes("pastel")
plt.pie(data= trans_count,x= "count", labels= "transferred" );


# - only 12 holidays were transferred

# In[72]:


#We will like to see if the day of the earthquake was a holiday
holidays= holidays.set_index("date")
holidays


# In[73]:


# Compute descriptive statistics for transactions_data
pd.DataFrame(holidays.describe())


# Summary of the Holiday Column:
# - Most holidays were national holidays
# - Most holidays were observed the same day they occured
# - Most of the holidays were not transferred except 12 holidays

# # Exploring the Oil Dataset

# In[74]:


oil_data.head()


# In[75]:


oil=oil_data.set_index('date')
oil.head()


# In[76]:


oil_data.copy()


# In[77]:


# visualizing the oil dataset
oil_data.plot()


# In[78]:


# checking for trends and pattern
oil.plot();


# - from the above, there is a backward or downward as the years increase

# In[79]:


oil.boxplot(figsize=(5,3));


# - there are no outliers 

# In[80]:


oil.hist(figsize=(5,3))
plt.title('Distribution of the oil dataset');


# In[81]:


# Compute descriptive statistics for oil dataset
pd.DataFrame(oil.describe())


# - in Summary,the oil prices trends positively over time.

# # Exploring the train dataset

# In[82]:


# creating a copy
train_1= train_data.copy()


# In[83]:


train_1.head()


# In[84]:


train=train_1.set_index('date')
train.head()


# In[85]:


# checking the distribution for the train data
train.hist();


# In[86]:


#Checking for outliers
train.boxplot();


# - There are some outliers in the sales attribute and more in the promotions

# In[87]:


# checking for skewness
train.skew()


# In[88]:


# Compute descriptive statistics for train dataset
pd.DataFrame(train.describe())


# In[89]:


#Exploring the family attrubute

family_counts = train.groupby('family').size().reset_index(name='count')
fig_1 = ex.scatter(family_counts, x='family', y='count', size='count', color= "family", hover_name='family',
                log_y=False, size_max=60)

fig_1.show()


# -All the family had the same count, and this makes sense because on days there were no purchases, the family of the product was still include

# In[90]:


#Exploring the train_1 atttribute
train["onpromotion"].plot();


# - No promotions were made from 2013 to 2014
# - There were increased promotions in 2016;likewise, 2017, and we will investigate that
# - There was a drop in promotion within some months in 2015 and 2014

# In[91]:


month_promo=train["onpromotion"].resample("M")
month_promo.plot();


# Notes of promotion after resmapling on promotion by month:
# 
# - Promotions were toned down from January to May of 2015
# 
# - They increased thier promotions for May of 2016, and we are guessing this is because of the earthquake
# 
# - Also, we can see an increase in promotions for some days in march to july of 2017

# ### Bivariate &  Multivariate Analysis

# Multivariate analysis’ is the analysis of more than one variable and aims to study the relationships among them. This analysis might be done by computing some statistical indicators like the correlation and by plotting some charts.
# 
# Please, read this article to know more about the charts.

# ### Let's see how transactions were affected by the earthquake

# In[92]:


daily_trans= transactions["transactions"].resample("D").mean()
daily_trans.loc['2016-04-12' :'2016-04-19'].plot();


# - the visualisations shows that after the earthquake on 16th April,2016, there was a sharp increase in the transactions within that particular week

# In[93]:


# top five transactions from 12-April to 19-April 2016
earthquake=daily_trans.loc['2016-04-12' :'2016-04-19'].sort_values(ascending=False).head()
pd.DataFrame(earthquake)


# Summary after earthquake:
# - massive difference in transactions after earthquake for  week, then falls again

# In[94]:


plt.plot(train_data['date'], train_data['sales'])
plt.xlabel('date')
plt.ylabel('sales')
plt.title('date vs. sales')
plt.xticks(rotation=90);
plt.show();


# - the visual shows that maximum sales occurred within the first quarter 2016

# In[95]:


##Due to earthquake, might be interesting to see how promotions were affected in 2016
Month_2016=train["onpromotion"].loc["2016"].resample("M")
Month_2016.plot();


# From the graph;
# - onpromotions in 2016 were pretty steady in April
# - They increased promotion during the end of April and in May of 2016

# In[96]:


##Let's resample the above data by day to see which days  happened:
train_day=train["onpromotion"].loc["2016-04":"2016-06"].resample("D")
train_day.plot();


# - Massive promotion was ran from 29th April to 31st May of 2016

# In[97]:


train.sales.plot();


# Summary about on promotions:
# - We can see some spikes in sales in 2014, 2015, 2016 (really prominent ones), and 2017 as well

# In[98]:


# visual showing the sales in the year 2016
train["sales"].loc["2016"].plot();


# - the diagram depicts that maximum sales was recorded at the fisrt quarter of 2016, precisely from the second week of April to the beginning of the month of May.

# In[99]:


#Exploring the sales dataset
train["sales"].plot();


# - We can see some sharp increase in sales so, we will investigate that

# In[100]:


sale_month=train["sales"].resample("M")
sale_month.plot();


# - Again, highest transactions for month was April, as showed by the diagram

# In[101]:


sale_daily_Apr=train["onpromotion"].loc["2016-04-10":"2016-04-20"].resample("D")


# In[102]:


# visualisation of daily sales in the month of April
sale_daily_Apr.plot();


#  - similarly, two days recorded the highest promotion, thus; 2016-04-13 and 2016-04-19

# In[103]:


sales_sum=train.groupby("date").agg({"sales": "sum"})
sales_sum


# checking for correlation in the train dataset

# In[104]:


sns.heatmap(train.corr(), annot= True);


# Answering Questions
# 
# Q1.which city had the highest stores
# 
# Q2.which month did we have the highest sale
# 
# Q3.which store had the highest transaction?
# 
# Q4.which store had the highest sale?
# 
# Q5what is the most bought product?

# # Q1.which city had the highest stores

# In[105]:


sns.countplot("city", data=stores_data)
plt.title("Count of Stores Across Various Cities")
plt.xticks(rotation=90)
plt.figure(figsize= (20,15))

plt.show();


# - from the visual, Quito recorded the highest, followed by Guayaqui

# ### Q2.which month did we have the highest 

# In[106]:


sale_month=train["sales"].resample("M")
sale_month.plot();


# - The highest sale (per unit product sold) was recorded in April of 2016

# ### Q3.which store had the highest transaction?

# In[107]:


sto_tran= transactions.groupby("store_nbr")["transactions"].agg("sum").sort_values(ascending= False).head()
sns.barplot(x=sto_tran.index, y=sto_tran.values)
plt.xticks(rotation= 45)
plt.title('Top 5 Store with the Highest Transactions');


# - store number 44 had the highest transactions

# ### Q4.which store had the highest sale?

# In[108]:


sto_sale= train.groupby("store_nbr")["sales"].agg("sum").sort_values(ascending= False).head()
sns.barplot(x=sto_sale.index, y=sto_sale.values)
plt.xticks(rotation= 45)
plt.title('Top 5 Store with the Highest Sale');


# - obviously from the visualisation, store number 44 again has the highest sales

# ### Q5.what is the most bought product per unit sale?

# In[109]:


family_bought=train[train['sales']!=0]


# In[110]:


##Lets group and plot
family_sale = family_bought.groupby('family').size().reset_index(name='count').sort_values(by='count')
fig_2= ex.scatter(family_sale, x='family', y='count', size='count', color= "family", hover_name='family',
                log_y=False, size_max=60)
fig_2.show()


# In[111]:


family_sale


# - books where bought less often, bread and bakery had the highest sales

# In[ ]:





# ## Feature processing

# In[112]:


merged_data1= pd.merge(train, transactions, how= "outer", on= ["date", "store_nbr"])
merged_data2= pd.merge(merged_data1, holidays, how= "outer", on= "date")
merged_data3= pd.merge(merged_data2, oil_data, how= "outer", on='date')
train_merged= pd.merge(merged_data3, stores_data, how= "outer", on= "store_nbr")


# In[113]:


train_merged['day']=train_merged['date'].dt.day
train_merged['month']=train_merged['date'].dt.month
train_merged['day_of_year']= train_merged['date'].dt.dayofyear
train_merged['week_of_year']= train_merged['date'].dt.isocalendar().week
train_merged['day_of_week']= train_merged['date'].dt.dayofweek
train_merged['is_weekend']= np.where(train_merged['day_of_week'] > 4,1,0)
train_merged['is_month_start']= train_merged['date'].dt.is_month_start.astype(int)
train_merged['is_month_end']= train_merged['date'].dt.is_month_end.astype(int)
train_merged['quarter']= train_merged['date'].dt.quarter
train_merged['is_quarter_start']= train_merged['date'].dt.is_quarter_start.astype(int)
train_merged['is_quarter_end']= train_merged['date'].dt.is_quarter_end.astype(int)
train_merged['is_year_start']= train_merged['date'].dt.is_year_start.astype(int)
train_merged['is_year_end']= train_merged['date'].dt.is_year_end.astype(int)


# In[114]:


train_merged.head()


# Here is the section to clean and process the features of the dataset.

# In[116]:


#checking for null values
train_merged.isna().sum()


# ### Drop Duplicates
# 

# In[117]:


train_merged.duplicated()


# ## Univariate Modelling

# The goal of this to forecast the general sales for Favorita irrespective of the product type, store number and other exogenous effects. Therefore, we will be using models to only figure out the sales.
# Before procedding we will check for the following:
# - Stationarity
# - Autocorrelation

# In[118]:


train_sales= train.groupby("date").agg({"sales": "sum"})
train_sales.head()


# In[119]:


##Let's plot to check for any seasonality 
fig_3= ex.line(train_sales, x= train_sales.index, y= "sales", title= "Aggregated Sales With Slider")

fig_3.update_xaxes(rangeslider_visible= True, 
                 rangeselector= dict(buttons= list([dict(count= 1, label= "1y", step= "year", stepmode= "backward"), 
                                                    dict(count= 2, label= "2y", step= "year", stepmode= "backward"), 
                                                     dict(count= 3, label= "3y", step= "year", stepmode= "backward"),
                                                     dict(step="all")])))
fig_3.show()


# #### Notes From The Plot:
# - We can see some seasonality, and a positive secular trend. 
# - To prove this, we will perform an adf test

# ## Univariate Statistical Modelling

# In[120]:


from statsmodels.tsa.stattools import adfuller as adf


# In[121]:


result= adf((train_sales["sales"]))
pd.DataFrame(result)


# In[122]:


print("P_value: ", {result[1]})
if result[1] > 0.05:
    
    print("Data is Not Stationary")
else:
    print("Data is  stationary")


# - Since our time series is non-stationary, we will go ahead to decompose it to see its components. We will also use PACF and ACF to see the autocorrelation in the time series. These will help us make a more informed decisions

# In[123]:


##performing decomposition
import statsmodels.api  as sm
res_add=  sm.tsa.seasonal_decompose(train_sales["sales"], model="additive", period= 365)


# In[124]:


res_add.plot();


# Note:
# -Our data has a seasonal component, a trend component and a residual component.
# -Now we will perform an ACF and a PACF test to solidify our decision on the type of model to use

# In[125]:


##perfomring PACF and ACF test
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf


# In[126]:


##ploting out ACF 
acf_plot= plot_acf(train_sales["sales"])


# - We have tons of spikes above significant level and it isn't decaying, therefore, we will use a moving average model. We can also see some seasonality

# In[127]:


##plotting out pacf
pacf_plot= plot_pacf(train_sales["sales"])


# ### Notes on our PACF test:
# - Our PACF shows strong lags at 1,3,5,6,7,8,9,13 and 14
# Therefore, we will use an ARMA model with lags at these values 

# Luckily, we have models that can handle them without us necessarily having to decompose the component, and we will like to compare two of such models, which are:
# - For our ARMA model, we will use SARIMA
# - Exponential Smoothing Winters
# - Facebook Prophet

# Note:
# - since these models can be sensistive to outliers, its good that we check for outliers in our prediction value and then deal with them before passing the through our model

# In[128]:


# checking for outliers
sns.boxplot(data=train_sales, x=train_sales['sales'])


# - there are outliers in the dataset

# In[129]:


plt.figure(figsize= (10,5))
plt.plot(train_sales)

for year in range(2013, 2018):
    plt.axvline(datetime(year, 1, 1), color= "k", linestyle= "--", alpha= 0.5)


# In[130]:


# computing a desccriptive statistics on the train_sales dataset
train_sales.describe().transpose()


# In[131]:


# checking the distribution of the train_sales data
sns.distplot(train_sales["sales"]);


# In[132]:


# checking for skewness
train_sales['sales'].skew()


# ### Hypothesis testing

# In[133]:


import statsmodels.api as sm
# Create the regression model
model = sm.formula.ols('sales ~ onpromotion + type_x', data= train_merged).fit()

# Print the model summary
print(model.summary())


# In[134]:


import statsmodels.api as sm

# Create the regression model
model = sm.formula.ols('sales ~ onpromotion +dcoilwtico', data= train_merged).fit()

# Print the model summary
print(model.summary())


# In[135]:


import statsmodels.api as sm

# Create the regression model
model = sm.formula.ols('sales ~ onpromotion +transactions', data= train_merged).fit()

# Print the model summary
print(model.summary())


# summary:
# -The R-squared value of the model is 0.223, meaning 22.3% of the variation in sales can be explained by the predictors. The coefficients for each predictor indicate how much the dependent variable is expected to change when the predictor changes by one unit. The model is statistically significant in predicting sales, with all three predictors having p-values less than 0.05 and a high F-statistic.Hence, our hypothesis is true. However, the condition number is large, suggesting further investigation is needed.

# ### Dataset Splitting

# In[136]:


##Since we know the models we will be using, we will go ahead to split our dataset. 
from sklearn.model_selection import train_test_split
train,test= train_test_split(train_sales, test_size= 0.2,shuffle= False)


# In[137]:


train.head()


# In[138]:


test.head()


# In[139]:


#We will be using auto_arima to find the values for our sarima model 
from pmdarima import auto_arima


# In[140]:


model= auto_arima(train, m=6)


# In[141]:


model.summary()


# In[142]:


model.fit(train)


# In[143]:


# making predictions
forecast_1= model.predict(n_periods= len(test))
forecast_1


# In[144]:


#Since the prediction returned a series, we will retrieve our time series index
forecast_1.index=test.index
forecast_1


# #### visualizing and testing our error for SARIMA model

# In[145]:


plt.plot(train, label= "Training Data")
plt.plot(test, label= "Actual Forecast")
plt.plot(forecast_1, label= "Predicted Sales Forecast")
plt.legend()
plt.show();


# In[146]:


from sklearn.metrics import mean_absolute_error as MAE
from sklearn.metrics import mean_squared_log_error as MSLE
from sklearn.metrics import mean_squared_error as MSE


# In[147]:


MEA_error_1= ((MAE(test, forecast_1).round(2))/test.mean())* 100
RMSLE_error_1= np.sqrt(MSLE(test, forecast_1))
result_1= pd.DataFrame([["SARIMA", MEA_error_1, RMSLE_error_1]], columns= ["Model", "MAE", "RMSLE"])              
result_1


# Notes on Sarima:
# - Our model has a 16.46% error rate.

# ## Univariate Model 2: Exponential Smoothing Using Holt Winters

# In[148]:


from statsmodels.tsa.holtwinters import ExponentialSmoothing


# In[149]:


## fitting our model
model_2= ExponentialSmoothing(endog= train, trend= "add",seasonal= "add", seasonal_periods= 12).fit()


# In[150]:


forecast_2=model_2.forecast(steps= len(test))
forecast_2.index= test.index


# In[151]:


plt.plot(train, label= "Training Data")
plt.plot(test, label= "Test Data")
plt.plot(forecast_2, label= "Forecast")
plt.show()


# #### Model Evaluation using Mean Absolute Error

# In[152]:


MEA_error_2= ((MAE(test, forecast_2).round(2))/test.mean())* 100
RMSLE_error_2= np.sqrt(MSLE(test, forecast_2))
result_2= pd.DataFrame([["Holt Winters", MEA_error_2, RMSLE_error_2]], columns= ["Model", "MAE", "RMSLE"])             
result_2


# ### Impute Missing Values

# In[ ]:


# Use sklearn.impute.SimpleImputer


# ### New Features Creation

# In[ ]:


# Code here


# ### Features Encoding

# In[ ]:


# From sklearn.preprocessing use OneHotEncoder to encode the categorical features.
     


# ### Features Scaling

# In[ ]:


# From sklearn.preprocessing use StandardScaler, MinMaxScaler, etc.


# ### Optional: Train set Balancing (for Classification only)

# In[ ]:


# Use Over-sampling/Under-sampling methods, more details here: https://imbalanced-learn.org/stabl


# ## Machine Learning Modeling

# Here is the section to build, train, evaluate and compare the models to each others.

# ### Simple Model #001

# Please, keep the following structure to try all the model you want.

# ### Create the Model

# In[ ]:


# Code here


# ### Train the Model

# In[ ]:


# Use the .fit method


# ### Evaluate the Model on the Evaluation dataset (Evalset)

# In[ ]:


# Compute the valid metrics for the use case # Optional: show the classification report 


# ### Predict on a unknown dataset (Testset)

# In[ ]:


# Use .predict method # .predict_proba is available just for classification
     


# ### Simple Model #002

# ### Create the Model

# ### Code here

# In[ ]:


# Code here


# ### Train the Model

# In[ ]:


# Use the .fit method


# ### Evaluate the Model on the Evaluation dataset (Evalset)

# In[ ]:


# Compute the valid metrics for the use case # Optional: show the classification report 


# ### Predict on a unknown dataset (Testset)

# In[ ]:


# Use .predict method # .predict_proba is available just for classification


# ### Simple Model #002

# ### Create the Model

# Code here
# ### Train the Model
# Use the .fit method
# ### Evaluate the Model on the Evaluation dataset (Evalset)
# Compute the valid metrics for the use case # Optional: show the classification report 
# ### Predict on a unknown dataset (Testset)
# Use .predict method # .predict_proba is available just for classification
     
# ## Models comparison
Create a pandas dataframe that will allow you to compare your models.Find a sample frame below :Model_Name	Metric (metric_name)	Details
0	-	-	-
1	-	-	-

You might use the pandas dataframe method .sort_values() to sort the dataframe regarding the metric.
# In[ ]:





# ## Hyperparameters tuning
Fine-tune the Top-k models (3 < k < 5) using a GridSearchCV (that is in sklearn.model_selection ) to find the best hyperparameters and achieve the maximum performance of each of the Top-k models, then compare them again to select the best one.
# In[ ]:


# Code here


# ## Export key components
Here is the section to export the important ML objects that will be use to develop an app: Encoder, Scaler, ColumnTransformer, Model, Pipeline, etc.
# In[ ]:





# In[ ]:





# In[ ]:


# Code Here


# In[ ]:


# Code here


# In[ ]:




