import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from datetime import datetime


#Define Functions

#DF styling
def style_negative(v, props=''):
    """ Style negative values in dataframe"""
    try: 
        return props if v < 0 else None
    except:
        pass
    
def style_positive(v, props=''):
    """Style positive values in dataframe"""
    try: 
        return props if v > 0 else None
    except:
        pass    

#Simplify how country is represented in DF.
def audience_simple(country):
    """Show top represented countries"""
    if country == 'US':
        return 'USA'
    elif country == 'IN':
        return 'India'
    else:
        return 'Other'


#load
    #Skipping first row, because it a total row
df_agg = pd.read_csv(r'Aggregated_Metrics_By_Video.csv').iloc[1:,:]
df_agg_sub = pd.read_csv(r'Aggregated_Metrics_By_Country_And_Subscriber_Status.csv') 
df_comments = pd.read_csv(r'All_Comments_Final.csv')
df_time = pd.read_csv(r'Video_Performance_Over_Time.csv') 

#Rename columns due to ascii errors messing up column selection
df_agg.columns= ["Video", "Video title", "Video publish time","Comments added", "Shares", "Dislikes", "Likes", "Subscibers Lost","Subscribers Gained", "RPM USD", "CPM USD", "Avg percentage viewed", "Avg view duration", "Views", "Watch time (hours)", "Subscribers", "Est Revenue", "Impressions", "Impression click-through rate (%)"]

#Two missing values in CPM, fill with median
df_agg['CPM (USD)'] = df_agg['CPM USD'].fillna(df_agg['CPM USD'].median())

#Format Video pub­lish time as Date
df_agg["Video publish time"] = pd.to_datetime(df_agg['Video publish time'], format="%b %d, %Y")

#Format Avg view duration as time object. 
df_agg['Avg view duration'] = df_agg['Avg view duration'].apply(lambda x: datetime.strptime(x, '%H:%M:%S').time())

#Feature Engineering

#Average duration of view in seconds
df_agg['Avg_duration_sec'] = df_agg['Avg view duration'].apply(lambda x: x.second + x.minute*60 + x.hour*3600)


df_agg['Engagement_ratio'] =  (df_agg['Comments added'] + df_agg['Shares'] +df_agg['Dislikes'] + df_agg['Likes']) /df_agg.Views

df_agg['Views / sub gained'] = df_agg['Views'] / df_agg['Subscribers Gained']

df_agg.sort_values('Video publish time', ascending = False, inplace = True)    

df_time['Date'] = df_time['Date'].str.replace('Sept', 'Sep')

df_time['Date'] = pd.to_datetime(df_time['Date'], format="%d %b %Y")

df_time.sort_values('Date', ascending = False, inplace = True)

#Create a series containing median data for each metric over the last 12 months.  
df_agg_diff = df_agg.copy()
metric_date_12mo = df_agg_diff['Video publish time'].max() - pd.DateOffset(months =12)
describe_agg = df_agg_diff[df_agg_diff['Video publish time'] >= metric_date_12mo].describe()
numeric_cols = [col for col in df_agg_diff.describe().columns if col != 'Video publish time']
median_agg = df_agg_diff[numeric_cols].median()

#Normalized data

df_agg_diff[numeric_cols] = (df_agg_diff[numeric_cols] - median_agg).div(median_agg)

#merge daily data with publish date data to get metrics for each day 
df_time_diff = pd.merge(df_time, df_agg.loc[:,['Video','Video publish time']], left_on ='External Video ID', right_on = 'Video')
df_time_diff['days_published'] = (df_time_diff['Date'] - df_time_diff['Video publish time']).dt.days

# get last 12 months of data rather than all data 
date_12mo = df_agg['Video publish time'].max() - pd.DateOffset(months =12)
df_time_diff_yr = df_time_diff[df_time_diff['Video publish time'] >= date_12mo]

# get daily view data (first 30), median & percentiles 
views_days = pd.pivot_table(df_time_diff_yr,index= 'days_published',values ='Views', aggfunc = [np.mean,np.median,lambda x: np.percentile(x, 80),lambda x: np.percentile(x, 20)]).reset_index()
views_days.columns = ['days_published','mean_views','median_views','80pct_views','20pct_views']
views_days = views_days[views_days['days_published'].between(0,30)]
views_cumulative = views_days.loc[:,['days_published','median_views','80pct_views','20pct_views']] 
views_cumulative.loc[:,['median_views','80pct_views','20pct_views']] = views_cumulative.loc[:,['median_views','80pct_views','20pct_views']].cumsum()



###############################################################################
#Start building Streamlit App
###############################################################################


#Create sidebar for navigation
add_sidebar = st.sidebar.selectbox('Aggregate or Individial Video', ('Aggregate Metrics','Individual Video Analysis'))

#Show individual metrics
if add_sidebar == 'Aggregate Metrics':
    st.title('Ken Jee YouTube Aggregate Metrics')
    df_agg_metrics = df_agg[['Video publish time','Views','Likes','Subscribers','Shares','Comments added','RPM USD','Avg percentage viewed',
                             'Avg_duration_sec', 'Engagement_ratio','Views / sub gained']]
    
    metric_date_6mo = df_agg_metrics['Video publish time'].max() - pd.DateOffset(months =6)
    metric_date_12mo = df_agg_metrics['Video publish time'].max() - pd.DateOffset(months =12)
    metric_medians6mo = df_agg_metrics[df_agg_metrics['Video publish time'] >= metric_date_6mo].median(numeric_only=True)
    metric_medians12mo = df_agg_metrics[df_agg_metrics['Video publish time'] >= metric_date_12mo].median(numeric_only=True)
    
# Yeah, I have no idea how this works or why...It makes a grid. Probably would be good to find a different method on how to make a grid.
    col1, col2, col3, col4, col5 = st.columns(5)
    columns = [col1, col2, col3, col4, col5]

    count = 0
    for i in metric_medians6mo.index:
        with columns[count]:
            delta = (metric_medians6mo[i] - metric_medians12mo[i])/metric_medians12mo[i]
            st.metric(label= i, value = round(metric_medians6mo[i],1), delta = "{:.2%}".format(delta))
            count+=1
            if count >= 5:
                count = 0

    #Trim to relavant dataset to relevant dates and get only numeric columns. 

    df_agg_diff['Publish_date'] = df_agg_diff['Video publish time'].apply(lambda x: x.date())

    df_agg_diff_final = df_agg_diff.loc[:,['Video title','Publish_date','Views','Likes','Subscribers','Shares','Comments added','RPM USD','Avg percentage viewed',
                                'Avg_duration_sec', 'Engagement_ratio','Views / sub gained']]

    df_agg_numeric_list = df_agg_diff.median(numeric_only=True).index.tolist()

    df_to_pct = {}

    for i in df_agg_numeric_list:
        df_to_pct[i] = '{:.1%}'.format

    #Add df to website
    st.dataframe(df_agg_diff_final.style.hide().applymap(style_negative, props='color:red;').applymap(style_positive, props='color:green;').format(df_to_pct))



##Individual video analysis page

if add_sidebar == 'Individual Video Analysis':
    #st.selectbox takes in a tuple so we need to make a tuple of all the videos
    videos = tuple(df_agg['Video title'])
    st.write("Individual Video Performance")
    video_select= st.selectbox('Pick a video:', videos)

    #Filter agg data for only the video selected by user
    agg_filtered = df_agg[df_agg['Video title'] == video_select]
    agg_sub_filtered = df_agg_sub[df_agg_sub['Video Title'] == video_select]

    #Simplify the way the country is displayed in DF
    agg_sub_filtered['Country'] = agg_sub_filtered['Country Code'].apply(audience_simple)

    agg_sub_filtered.sort_values('Is Subscribed', inplace= True)   

    #Plotly express bar chart
    fig = px.bar(agg_sub_filtered, x='Views', y='Is Subscribed', color='Country', orientation='h')

    #Embed chart into Streamlit website
    st.plotly_chart(fig)

    #Filter for the first 30 days for each selected video
    agg_time_filtered = df_time_diff[df_time_diff['Video Title'] == video_select]
    first_30 = agg_time_filtered[agg_time_filtered['days_published'].between(0,30)]
    first_30 = first_30.sort_values('days_published')

    #Create Plotly Line Chart using plotly graph objects

    fig2 = go.Figure()
    #add line to figure. go.Scatter creates scatter plot, but mode = 'lines' connects all the dots. :)
    fig2.add_trace(go.Scatter(x=views_cumulative['days_published'], y=views_cumulative['20pct_views'], mode='lines', name='20th percentile', line=dict(color='purple', dash='dash')))
    fig2.add_trace(go.Scatter(x=views_cumulative['days_published'], y=views_cumulative['median_views'], mode='lines', name='50th percentile', line=dict(color='black', dash='dash')))
    fig2.add_trace(go.Scatter(x=views_cumulative['days_published'], y=views_cumulative['80pct_views'], mode='lines', name='80th percentile', line=dict(color='royalblue', dash='dash')))
    fig2.add_trace(go.Scatter(x=first_30['days_published'], y=first_30['Views'].cumsum(), mode='lines', name='Current Video', line=dict(color='firebrick', width=8)))    

    #Change figure layout
    fig2.update_layout(title='view comparison first 30 days', xaxis_title='Days Since Published', yaxis_title='Cumulative views')

    #Embed chart into streamlit website
    st.plotly_chart(fig2)



# df_agg_diff.head()
# df_agg.head()
# df_agg_sub.head()
# df_comments.head()
# df_time.head()
# median_agg.head()


