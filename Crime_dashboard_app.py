import pandas as pd
import plotly.express as px
from dash import Dash, dcc, html, Input, Output
import plotly.graph_objects as go
from prophet import Prophet
import folium
from folium.plugins import HeatMap
from dash.dependencies import Input, Output

# Load the cleaned dataset
df = pd.read_csv("Crime_Data_cleaned.csv")

# Convert 'DATE OCC' to datetime and extract temporal features
df['DATE OCC'] = pd.to_datetime(df['DATE OCC'])
df['Year'] = df['DATE OCC'].dt.year
df['Month'] = df['DATE OCC'].dt.month_name()
df['Day_of_Week'] = df['DATE OCC'].dt.day_name()
df['Day_of_Week_Num'] = df['DATE OCC'].dt.dayofweek  # Monday=0, Sunday=6
df['Month_Year'] = df['DATE OCC'].dt.to_period('M').astype(str)

# Initialize the Dash app
app = Dash(__name__, 
           external_stylesheets=['https://codepen.io/chriddyp/pen/bWLwgP.css'],
           suppress_callback_exceptions=True)

# Custom CSS styles with improved spacing and better text visibility
styles = {
    'header': {
        'backgroundColor': '#2c3e50',
        'color': 'white',
        'padding': '15px',
        'textAlign': 'center',
        'borderRadius': '5px',
        'marginBottom': '15px',
        'fontSize': '24px'
    },
    'dynamic-summary-container': {
        'backgroundColor': '#e8f4f8',
        'padding': '15px',
        'borderRadius': '5px',
        'marginBottom': '20px',
        'borderLeft': '5px solid #2980b9',
        'transition': 'all 0.3s ease'
    },
    'summary-title': {
        'color': '#2c3e50',
        'marginTop': '0',
        'marginBottom': '10px',
        'fontWeight': 'bold'
    },
    'summary-text': {
        'fontSize': '16px',
        'lineHeight': '1.6',
        'color': '#333',
        'marginBottom': '0'
    },
    'filter-container': {
        'backgroundColor': '#ecf0f1',
        'padding': '15px',
        'borderRadius': '5px',
        'marginBottom': '20px',
        'display': 'flex',
        'justifyContent': 'center',
        'gap': '30px'
    },
    'filter-item': {
        'display': 'flex',
        'flexDirection': 'column',
        'minWidth': '250px'
    },
    'graph-container': {
        'backgroundColor': 'white',
        'padding': '15px',
        'borderRadius': '5px',
        'boxShadow': '0 2px 4px 0 rgba(0, 0, 0, 0.1)',
        'marginBottom': '15px',
        'height': '100%'
    },
    'dropdown': {
        'marginBottom': '0',
        'fontSize': '14px'
    },
    'label': {
        'fontWeight': 'bold',
        'marginBottom': '5px',
        'fontSize': '14px',
        'color': '#333'
    },
    'main-container': {
        'padding': '15px', 
        'backgroundColor': '#f5f5f5',
        'fontFamily': 'Arial, sans-serif'
    },
    'map-container': {
        'width': '100%',
        'height': '500px',
        'marginBottom': '20px'
    }
}

# Create heatmap data
df_clean = df.dropna(subset=['LAT', 'LON'])
heat_data = [[row['LAT'], row['LON']] for index, row in df_clean.iterrows()]
map_center = [df_clean['LAT'].mean(), df_clean['LON'].mean()]

# Create the base map
crime_map = folium.Map(location=map_center, zoom_start=12)
HeatMap(heat_data).add_to(crime_map)

# Save the map to HTML
crime_map.save('heatmap.html')

# Prepare data for forecasting
forecast_data = df.groupby(df['DATE OCC'].dt.to_period('M')).size().reset_index(name='y')
forecast_data['ds'] = forecast_data['DATE OCC'].astype(str)
forecast_data = forecast_data[['ds', 'y']]

# Prophet model - suppress the cmdstanpy logging
import logging
logger = logging.getLogger('cmdstanpy')
logger.addHandler(logging.NullHandler())
logger.propagate = False
logger.setLevel(logging.CRITICAL)

# Prophet model with monthly seasonality
model = Prophet(weekly_seasonality=True, 
               yearly_seasonality=True,
               daily_seasonality=False)
model.fit(forecast_data)

# Create future dataframe (next 12 months) - using 'ME' instead of deprecated 'M'
future = model.make_future_dataframe(periods=12, freq='ME')
forecast = model.predict(future)

app.layout = html.Div([
    html.Div([
        html.H1("LA Crime Data Dashboard", style=styles['header']),
        
        # Filters row at the top
        html.Div([
            html.Div([
                html.Label("Select Year:", style=styles['label']),
                dcc.Dropdown(
                    id='year-filter',
                    options=[{'label': str(year), 'value': year} for year in sorted(df['Year'].unique())],
                    value=sorted(df['Year'].unique())[0],
                    clearable=False,
                    style=styles['dropdown']
                )
            ], style=styles['filter-item']),
            
            html.Div([
                html.Label("Select Area:", style=styles['label']),
                dcc.Dropdown(
                    id='area-filter',
                    options=[{'label': area, 'value': area} for area in sorted(df['AREA NAME'].unique())],
                    value=sorted(df['AREA NAME'].unique())[0],
                    clearable=False,
                    style=styles['dropdown']
                )
            ], style=styles['filter-item'])
        ], style=styles['filter-container']),
        
        # Dynamic area summary section that updates with filters
        html.Div(id='dynamic-summary', style=styles['dynamic-summary-container']),
        
        # Visualizations grid
        html.Div([
            # First row - Trend and crimes by year
            html.Div([
                html.Div([
                    dcc.Graph(id='crime-trend-plot', 
                             style={'height': '400px'})
                ], className="six columns", style=styles['graph-container']),
                
                html.Div([
                    dcc.Graph(id='crimes-by-year', 
                             style={'height': '400px'})
                ], className="six columns", style=styles['graph-container'])
            ], className="row"),
            
            # Second row - Crime patterns by time
            html.Div([
                html.Div([
                    dcc.Graph(id='top5-by-month', 
                             style={'height': '350px'})
                ], className="six columns", style=styles['graph-container']),
                
                html.Div([
                    dcc.Graph(id='top5-by-day', 
                             style={'height': '350px'})
                ], className="six columns", style=styles['graph-container'])
            ], className="row"),
            
            # Third row - Victim demographics
            html.Div([
                html.Div([
                    dcc.Graph(id='age-distribution-histogram', 
                             style={'height': '350px'})
                ], className="six columns", style=styles['graph-container']),
                
                html.Div([
                    dcc.Graph(id='victim-sex-pie', 
                             style={'height': '350px'})
                ], className="six columns", style=styles['graph-container'])
            ], className="row"),
            
            # Fourth row - Heatmap and forecast (now at the bottom)
            html.Div([
                html.Div([
                    html.Iframe(id='heatmap', srcDoc=open('heatmap.html', 'r').read(), 
                               style=styles['map-container'])
                ], className="six columns", style=styles['graph-container']),
                
                html.Div([
                    dcc.Graph(id='crime-forecast', 
                             style={'height': '500px'})
                ], className="six columns", style=styles['graph-container'])
            ], className="row")
        ], style={'margin': '0'})
    ], style=styles['main-container'])
])

@app.callback(
    Output('dynamic-summary', 'children'),
    [Input('year-filter', 'value'),
     Input('area-filter', 'value')]
)
def update_dynamic_summary(selected_year, selected_area):
    # Filter data based on selections
    filtered_df = df[(df['Year'] == selected_year) & (df['AREA NAME'] == selected_area)]
    
    # Calculate dynamic statistics
    total_crimes_area = len(filtered_df)
    crimes_per_month = round(total_crimes_area / 12, 1) if total_crimes_area > 0 else 0
    
    # Most common crime in selected area/year
    if not filtered_df.empty:
        area_common_crime = filtered_df['Crm Cd Desc'].value_counts().idxmax()
        area_common_crime_count = filtered_df['Crm Cd Desc'].value_counts().max()
        crime_percentage = round((area_common_crime_count / total_crimes_area) * 100, 1)
    else:
        area_common_crime = "No data"
        crime_percentage = 0
    
    # Victim demographics
    valid_ages = filtered_df[filtered_df['Vict Age'] > 0]['Vict Age']
    avg_age_area = round(valid_ages.mean(), 1) if not valid_ages.empty else "N/A"
    
    sex_counts = filtered_df['Vict Sex'].value_counts()
    male_count = sex_counts.get('M', 0)
    female_count = sex_counts.get('F', 0)
    total_known_sex = male_count + female_count
    male_percentage = round((male_count / total_known_sex * 100), 1) if total_known_sex > 0 else "N/A"
    
    # Time of day analysis
    if not filtered_df.empty and 'TIME OCC' in filtered_df.columns:
        filtered_df['Hour'] = filtered_df['TIME OCC'].apply(lambda x: int(str(x).zfill(4)[:2]))
        peak_hour = filtered_df['Hour'].value_counts().idxmax()
        peak_hour_count = filtered_df['Hour'].value_counts().max()
    else:
        peak_hour = "N/A"
        peak_hour_count = 0
    
    return [
        html.H3(f"Area Summary for {selected_area} ({selected_year})", style=styles['summary-title']),
        html.P([
            html.Strong("Total Crimes: "), f"{total_crimes_area:,} | ",
            html.Strong("Avg. per Month: "), f"{crimes_per_month:,}",
            html.Br(),
            html.Strong("Most Common Crime: "), f"{area_common_crime} ({crime_percentage}%)",
            html.Br(),
            html.Strong("Peak Crime Hour: "), f"{peak_hour}:00 ({peak_hour_count} crimes)",
            html.Br(),
            html.Strong("Victim Demographics: "), 
            f"Avg. Age: {avg_age_area} | ",
            f"Male: {male_percentage}% | ",
            f"Female: {round(100 - male_percentage, 1)}%"
        ], style=styles['summary-text'])
    ]

@app.callback(
    [Output('crime-trend-plot', 'figure'),
     Output('crimes-by-year', 'figure'),
     Output('age-distribution-histogram', 'figure'),
     Output('victim-sex-pie', 'figure'),
     Output('top5-by-month', 'figure'),
     Output('top5-by-day', 'figure'),
     Output('crime-forecast', 'figure')],
    [Input('year-filter', 'value'),
     Input('area-filter', 'value')]
)
def update_tab1(selected_year, selected_area):
    filtered_df = df[(df['Year'] == selected_year) & (df['AREA NAME'] == selected_area)]
    top5_crimes = filtered_df['Crm Cd Desc'].value_counts().nlargest(5).index.tolist() if not filtered_df.empty else []
    
    # Function to shorten crime descriptions
    def shorten_crime_desc(desc, max_length=20):
        if len(desc) > max_length:
            return desc[:max_length-3] + '...'
        return desc
    
    # 1. Monthly Crime Trend
    trend_data = filtered_df.groupby('Month_Year').size().reset_index(name='Crime Count')
    trend_fig = px.line(trend_data, x='Month_Year', y='Crime Count',
                        title=f'Monthly Crime Trend - {selected_area} ({selected_year})',
                        markers=True,
                        color_discrete_sequence=['#3498db'])
    trend_fig.update_layout(
        xaxis_title='Month', 
        yaxis_title='Number of Crimes',
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        hovermode='x unified',
        margin={'l': 50, 'r': 30, 't': 50, 'b': 50},
        font={'size': 12, 'color': '#333'},
        title_font={'size': 16, 'color': '#333'}
    )

    # 2. Count of Crimes by Year
    crimes_by_year = df[df['AREA NAME'] == selected_area].groupby('Year').size().reset_index(name='Crime Count')
    
    year_fig = px.bar(crimes_by_year, 
                     x='Year', 
                     y='Crime Count',
                     title=f'Total Crimes by Year - {selected_area}',
                     color='Year',
                     color_continuous_scale=px.colors.sequential.Viridis,
                     text='Crime Count')
    
    year_fig.update_layout(
        xaxis_title='Year',
        yaxis_title='Number of Crimes',
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        hovermode='x unified',
        margin={'l': 50, 'r': 30, 't': 50, 'b': 50},
        font={'size': 12, 'color': '#333'},
        title_font={'size': 16, 'color': '#333'},
        showlegend=False,
        xaxis={
            'tickmode': 'linear',
            'dtick': 1
        }
    )
    
    year_fig.update_traces(
        texttemplate='%{text:,}',
        textposition='outside',
        textfont={'size': 12, 'color': '#333'},
        hovertemplate='Year: %{x}<br>Crimes: %{y:,}<extra></extra>'
    )
    
    if selected_year in crimes_by_year['Year'].values:
        year_fig.update_traces(
            marker_color=['#FFA500' if year == selected_year else '#1f77b4' for year in crimes_by_year['Year']]
        )

    # 3. Top 5 Offenses by Month
    if top5_crimes:
        top5_month_data = filtered_df[filtered_df['Crm Cd Desc'].isin(top5_crimes)]
        top5_month_data = top5_month_data.groupby(['Month', 'Crm Cd Desc']).size().reset_index(name='Count')
        top5_month_data['Short Label'] = top5_month_data['Crm Cd Desc'].apply(shorten_crime_desc)
        
        top5_month_fig = px.bar(top5_month_data, 
                               x='Month', 
                               y='Count', 
                               color='Short Label',
                               title=f'Top 5 Crimes by Month - {selected_area} ({selected_year})',
                               text='Count',
                               category_orders={'Month': ['January', 'February', 'March', 'April', 'May', 'June',
                                                        'July', 'August', 'September', 'October', 'November', 'December']},
                               color_discrete_sequence=px.colors.qualitative.Set2)
        
        top5_month_fig.update_traces(
            texttemplate='%{text:,}',
            textposition='outside',
            textfont={'size': 10, 'color': '#333'},
            hovertemplate='<b>%{customdata}</b><br>Month: %{x}<br>Count: %{y:,}<extra></extra>',
            customdata=top5_month_data['Crm Cd Desc']
        )
    else:
        top5_month_fig = px.bar(title=f'No data available for {selected_area} ({selected_year})')
    
    top5_month_fig.update_layout(
        xaxis_title='Month',
        yaxis_title='Number of Crimes',
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        legend_title='Crime Type',
        margin={'l': 50, 'r': 30, 't': 50, 'b': 120},
        font={'size': 12, 'color': '#333'},
        title_font={'size': 16, 'color': '#333'},
        legend={
            'font': {'size': 10, 'color': '#333'},
            'orientation': 'h',
            'yanchor': 'bottom',
            'y': -0.5,
            'xanchor': 'center',
            'x': 0.5
        },
        xaxis={
            'tickangle': -45,
            'tickfont': {'size': 10}
        },
        uniformtext_minsize=8,
        uniformtext_mode='hide'
    )

    # 4. Top 5 Offenses by Day of Week (Monday to Sunday)
    if top5_crimes:
        day_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
        
        top5_day_data = filtered_df[filtered_df['Crm Cd Desc'].isin(top5_crimes)]
        top5_day_data = top5_day_data.groupby(['Day_of_Week_Num', 'Day_of_Week', 'Crm Cd Desc']).size().reset_index(name='Count')
        top5_day_data['Short Label'] = top5_day_data['Crm Cd Desc'].apply(shorten_crime_desc)
        
        # Sort by day number to ensure correct order (Monday=0 to Sunday=6)
        top5_day_data = top5_day_data.sort_values('Day_of_Week_Num')
        
        top5_day_fig = px.line(top5_day_data, 
                             x='Day_of_Week', 
                             y='Count', 
                             color='Short Label',
                             title=f'Top 5 Crimes by Day of Week - {selected_area} ({selected_year})',
                             markers=True,
                             category_orders={'Day_of_Week': day_order},
                             color_discrete_sequence=px.colors.qualitative.Set3)
        
        top5_day_fig.update_traces(
            hovertemplate='<b>%{customdata}</b><br>Day: %{x}<br>Count: %{y}<extra></extra>',
            customdata=top5_day_data['Crm Cd Desc'],
            line=dict(width=3)
        )
    else:
        top5_day_fig = px.bar(title=f'No data available for {selected_area} ({selected_year})')
    
    top5_day_fig.update_layout(
        xaxis_title='Day of Week',
        yaxis_title='Number of Crimes',
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        legend_title='Crime Type',
        hovermode='x unified',
        margin={'l': 50, 'r': 30, 't': 50, 'b': 120},
        font={'size': 12, 'color': '#333'},
        title_font={'size': 16, 'color': '#333'},
        legend={
            'font': {'size': 10, 'color': '#333'},
            'orientation': 'h',
            'yanchor': 'bottom',
            'y': -0.5,
            'xanchor': 'center',
            'x': 0.5
        },
        xaxis={
            'tickfont': {'size': 12, 'color': '#333'}
        },
        yaxis={
            'tickfont': {'size': 12, 'color': '#333'}
        }
    )

    # 5. Victim Age Distribution
    age_df = filtered_df[(filtered_df['Vict Age'] > 0) & (filtered_df['Vict Age'] < 120)]
    if not age_df.empty:
        age_bins = [0, 18, 30, 45, 60, 120]
        age_labels = ['0-17', '18-29', '30-44', '45-59', '60+']
        age_df['Age Group'] = pd.cut(age_df['Vict Age'], bins=age_bins, labels=age_labels)
        
        hist_fig = px.histogram(age_df, 
                               x='Vict Age', 
                               nbins=30,
                               title=f'Victim Age Distribution - {selected_area} ({selected_year})',
                               color='Age Group',
                               color_discrete_sequence=px.colors.qualitative.Pastel)
    else:
        hist_fig = px.bar(title=f'No age data available for {selected_area} ({selected_year})')
    
    hist_fig.update_layout(
        xaxis_title='Victim Age', 
        yaxis_title='Number of Victims',
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        bargap=0.1,
        margin={'l': 50, 'r': 30, 't': 50, 'b': 50},
        font={'size': 12, 'color': '#333'},
        title_font={'size': 16, 'color': '#333'},
        legend_title='Age Group'
    )

    # 6. Victim Sex Distribution
    if not filtered_df.empty:
        sex_data = filtered_df['Vict Sex'].value_counts().reset_index()
        sex_data.columns = ['Sex', 'Count']
        sex_data['Sex'] = sex_data['Sex'].replace({'F': 'Female', 'M': 'Male', 'X': 'Other'})
        pie_fig = px.pie(sex_data, values='Count', names='Sex',
                         title=f'Victim Sex Distribution - {selected_area} ({selected_year})',
                         color='Sex',
                         color_discrete_map={'Female': '#ff7f0e', 'Male': '#1f77b4', 'Other': '#2ca02c'},
                         hole=0.3)
    else:
        pie_fig = px.pie(title=f'No sex data available for {selected_area} ({selected_year})')
    
    pie_fig.update_traces(
        textposition='inside',
        textinfo='percent+label',
        textfont={'size': 12, 'color': '#333'}
    )
    pie_fig.update_layout(
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        showlegend=True,
        margin={'l': 30, 'r': 30, 't': 50, 'b': 30},
        font={'size': 12, 'color': '#333'},
        title_font={'size': 16, 'color': '#333'},
        legend={'font': {'size': 12, 'color': '#333'}}
    )
    
    # 7. Crime Forecast
    forecast_fig = go.Figure()
    
    # Add forecasted line
    forecast_fig.add_trace(go.Scatter(
        x=forecast['ds'], 
        y=forecast['yhat'], 
        mode='lines', 
        name='Forecasted Crimes', 
        line=dict(color='blue', width=2)
    ))
    
    # Add actual crime data
    forecast_fig.add_trace(go.Scatter(
        x=forecast_data['ds'], 
        y=forecast_data['y'], 
        mode='markers', 
        name='Actual Crimes', 
        marker=dict(color='red', size=8)
    ))
    
    # Add confidence interval
    forecast_fig.add_trace(go.Scatter(
        x=forecast['ds'], 
        y=forecast['yhat_lower'], 
        fill=None, 
        mode='lines', 
        line=dict(width=0), 
        showlegend=False
    ))
    
    forecast_fig.add_trace(go.Scatter(
        x=forecast['ds'], 
        y=forecast['yhat_upper'], 
        fill='tonexty', 
        mode='lines', 
        line=dict(width=0), 
        name='Confidence Interval'
    ))
    
    forecast_fig.update_layout(
        title='Crime Forecast for Next 12 Months',
        xaxis_title='Date',
        yaxis_title='NuRmber of Crimes',
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        hovermode='x unified',
        margin={'l': 50, 'r': 30, 't': 50, 'b': 50},
        font={'size': 12, 'color': '#333'},
        title_font={'size': 16, 'color': '#333'}
    )

    return trend_fig, year_fig, hist_fig, pie_fig, top5_month_fig, top5_day_fig, forecast_fig

if __name__ == '__main__':
    app.run(debug=True)