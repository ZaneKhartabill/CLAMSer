import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import plotly.express as px

st.set_page_config(page_title="CLAMS Data Analyzer", layout="wide")

st.title("CLAMS Data Analyzer")
st.subheader("VO2 Data Processing")

# File upload
uploaded_file = st.file_uploader("Choose a VO2.CSV file", type="csv")

def process_vo2_data(file):
    """Process VO2 data from CLAMS output"""
    # Find where the data starts
    data_start = 0
    for i, line in enumerate(file):
        if b':DATA' in line:
            data_start = i + 3
            break
    
    file.seek(0)
    
    # Read the subject IDs
    subject_map = {}
    for line in file:
        line = line.decode('utf-8')
        if 'Group/Cage' in line:
            cage_num = line.split(',')[-1].strip().lstrip('0')
        elif 'Subject ID' in line and 'cage_num' in locals():
            subject_id = line.split(',')[-1].strip()
            cage_label = f"CAGE {int(cage_num) - 100:02d}"
            subject_map[cage_label] = subject_id
        elif ':DATA' in line:
            break
    
    # Read the actual data
    df = pd.read_csv(file, skiprows=data_start, header=None)
    
    # Process data
    all_data = []
    for i in range(1, len(df.columns)-1, 2):
        time_col = i
        value_col = i + 1
        cage_num = (i // 2) + 1
        
        times = []
        values = []
        
        for t, v in zip(df[time_col], df[value_col]):
            if isinstance(t, str) and not t.startswith('12:00:00'):
                try:
                    try:
                        time = pd.to_datetime(t.strip(), format='%d/%m/%Y %I:%M:%S %p')
                    except:
                        time = pd.to_datetime(t.strip(), format='%m/%d/%Y %I:%M:%S %p')
                    values.append(float(v))
                    times.append(time)
                except Exception as e:
                    if "===============" not in str(t):
                        continue
        
        if times:
            cage_data = pd.DataFrame({
                'timestamp': times,
                'value': values,
                'cage': f'CAGE {cage_num:02d}'
            })
            all_data.append(cage_data)
    
    if not all_data:
        return None, None, None
    
    # Combine all data
    df_processed = pd.concat(all_data, ignore_index=True)
    
    # Get the last 24 hours of data
    end_time = df_processed['timestamp'].max()
    start_time = end_time - pd.Timedelta(days=1)
    
    df_24h = df_processed[
        (df_processed['timestamp'] >= start_time) &
        (df_processed['timestamp'] <= end_time)
    ].copy()
    
    # Add light/dark cycle indicator (7AM-7PM is light)
    df_24h['hour'] = df_24h['timestamp'].dt.hour
    df_24h['is_light'] = (df_24h['hour'] >= 7) & (df_24h['hour'] < 19)
    
    # Calculate light/dark averages
    results = df_24h.groupby(['cage', 'is_light'])['value'].mean().unstack()
    results.columns = ['Dark Average', 'Light Average']
    results['Total Average'] = (results['Dark Average'] + results['Light Average']) / 2
    
    # Add subject IDs
    results['Subject ID'] = pd.Series(subject_map)
    
    # Calculate hourly averages
    hourly_results = df_24h.pivot_table(
        values='value',
        index='hour',
        columns='cage',
        aggfunc='mean'
    ).round(2)
    
    # Rename columns to subject IDs
    hourly_results.columns = [f"{subject_map.get(cage, cage)}" for cage in hourly_results.columns]
    
    # Add summary statistics
    hourly_results['Mean'] = hourly_results.mean(axis=1).round(2)
    hourly_results['SEM'] = (hourly_results.std(axis=1) / np.sqrt(hourly_results.shape[1])).round(2)
    
    return results, hourly_results, df_24h

if uploaded_file is not None:
    # Process data
    with st.spinner('Processing data...'):
        results, hourly_results, raw_data = process_vo2_data(uploaded_file)
        
        if results is not None:
            # Display summary statistics
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric("Average Light Cycle", f"{results['Light Average'].mean():.2f}")
            with col2:
                st.metric("Average Dark Cycle", f"{results['Dark Average'].mean():.2f}")
            with col3:
                st.metric("Total Records", len(raw_data))
            
            # Create visualization
            plot_data = hourly_results[['Mean']].reset_index()
            fig = px.line(plot_data, x='hour', y='Mean', 
                         title='24-Hour VO2 Pattern',
                         labels={'hour': 'Hour of Day', 'Mean': 'VO2 (ml/kg/hr)'})
            st.plotly_chart(fig, use_container_width=True)
            
            # Download buttons
            col1, col2 = st.columns(2)
            
            with col1:
                csv_light_dark = results.to_csv().encode('utf-8')
                st.download_button(
                    label="Download Light/Dark Averages",
                    data=csv_light_dark,
                    file_name="VO2_lightdark_averages.csv",
                    mime="text/csv"
                )
            
            with col2:
                csv_hourly = hourly_results.to_csv().encode('utf-8')
                st.download_button(
                    label="Download Hourly Averages",
                    data=csv_hourly,
                    file_name="VO2_hourly_averages.csv",
                    mime="text/csv"
                )
            
            # Display raw data tables
            st.subheader("Light/Dark Averages")
            st.dataframe(results)
            
            st.subheader("Hourly Averages")
            st.dataframe(hourly_results)