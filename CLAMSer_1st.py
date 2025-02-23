import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import plotly.express as px
import plotly.graph_objects as go

# Page setup
st.set_page_config(page_title="CLAMS Data Analyzer", layout="wide")
st.title("CLAMS Data Analyzer")

# Constants
PARAMETER_UNITS = {
    "VO2": "ml/kg/hr",
    "VCO2": "ml/kg/hr",
    "RER": "ratio",
    "HEAT": "kcal/hr",
    "XTOT": "counts",
    "XAMB": "counts",
    "FEED": "g"
}

# Helper functions
def detect_outliers(df, z_score_threshold=2):
    """Detect outliers using z-score method"""
    if df is None:
        return None
    
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    outliers = pd.DataFrame(False, index=df.index, columns=df.columns)
    
    for col in numeric_cols:
        if col not in ['hour', 'Mean', 'SEM']:  # Skip these columns
            z_scores = np.abs((df[col] - df[col].mean()) / df[col].std())
            outliers[col] = z_scores > z_score_threshold
    
    return outliers

def style_dataframe(df):
    """Apply styling to dataframe including outlier highlighting"""
    if df is None:
        return df
    
    outliers = detect_outliers(df)
    
    def highlight_outliers(col):
        if col.name in outliers.columns and col.name not in ['Subject ID']:
            return ['background-color: rgba(255, 0, 0, 0.2)' if v else '' for v in outliers[col.name]]
        return ['' for _ in range(len(df.index))]
    
    return df.style.apply(highlight_outliers)

def verify_file_type(file, expected_type):
    """
    Verify if uploaded file matches expected parameter type for CLAMS data files.
    
    Args:
        file: The uploaded file object
        expected_type: The expected parameter type (e.g., 'VO2', 'RER', etc.)
        
    Returns:
        tuple: (is_valid: bool, message: str)
    """
    try:
        content = file.read().decode()
        file.seek(0)
        
        # First line should identify this as a CLAMS/Oxymax file
        first_line = content.split('\n')[0].strip()
        if 'PARAMETER File' not in first_line:
            return False, "File format not recognized. Expected CLAMS/Oxymax parameter file."
            
        # Look for :DATA marker which should be present in all CLAMS files
        if ':DATA' not in content:
            return False, "File structure not recognized. Missing :DATA marker."
        
        # If the file is valid CLAMS format, always process it
        # Just warn if parameter type doesn't match exactly
        content_lower = content.lower()
        expected_lower = expected_type.lower()
        
        # Handle common unit variations
        expected_patterns = [
            f"{expected_lower}",                 # Basic parameter name
            f"{expected_lower} (ml/kg/hr)",      # VO2/VCO2
            f"{expected_lower} (ratio)",         # RER
            f"{expected_lower} (kcal/hr)",       # HEAT
            f"{expected_lower} (counts)",        # XTOT/XAMB
            f"{expected_lower} (g)"              # FEED
        ]
        
        if any(pattern in content_lower for pattern in expected_patterns):
            return True, ""
        else:
            # File appears to be valid CLAMS data, but parameter type might not match
            return True, f"Note: Selected parameter type '{expected_type}' not explicitly found in file header, but file appears to be valid CLAMS data. Processing anyway..."
            
    except UnicodeDecodeError:
        return False, "File cannot be read. Please ensure it's a valid CSV file."
    except Exception as e:
        return False, f"Error verifying file: {str(e)}"
def extract_cage_info(file):
    """Extract cage information from file header"""
    try:
        content = file.read().decode()
        file.seek(0)
        
        cage_info = {}
        lines = content.split('\n')
        current_cage = None
        
        for line in lines:
            if 'Group/Cage' in line:
                current_cage = line.split(',')[-1].strip().lstrip('0')
            elif 'Subject ID' in line and current_cage:
                subject_id = line.split(',')[-1].strip()
                cage_info[current_cage] = subject_id
            elif ':DATA' in line:
                break
                
        return cage_info
    except Exception as e:
        st.error(f"Error extracting cage information: {str(e)}")
        return {}

def process_clams_data(file, parameter_type):
    """
    Adapted from working Colab analysis functions for Streamlit interface.
    """
    try:
        # Read file content
        content = file.read().decode('utf-8')
        file.seek(0)

        # Parse subject IDs
        subject_map = {}
        data_start = 0
        for line in content.split('\n'):
            if 'Group/Cage' in line:
                cage_num = line.split(',')[-1].strip().lstrip('0')
            elif 'Subject ID' in line and 'cage_num' in locals():
                subject_id = line.split(',')[-1].strip()
                cage_label = f"CAGE {int(cage_num) - 100:02d}"
                subject_map[cage_label] = subject_id
            elif ':DATA' in line:
                break

        # Find where data starts
        lines = content.split('\n')
        for i, line in enumerate(lines):
            if ':DATA' in line:
                data_start = i + 3
                break

        # Process data exactly as in Colab
        all_data = []
        df_lines = [line.split(',') for line in lines[data_start:] if line.strip() and '=======' not in line]
        
        for i in range(1, len(df_lines[0])-1, 2):
            time_col = i
            value_col = i + 1
            cage_num = (i // 2) + 1

            times = []
            values = []

            for row in df_lines:
                if len(row) > value_col:
                    t, v = row[time_col].strip(), row[value_col].strip()
                    if t and v and not t.startswith('12:00:00') and t != "TIME":
                        try:
                            # Try both date formats exactly as in Colab
                            try:
                                time = pd.to_datetime(t, format='%d/%m/%Y %I:%M:%S %p')
                            except:
                                time = pd.to_datetime(t, format='%m/%d/%Y %I:%M:%S %p')
                            values.append(float(v))
                            times.append(time)
                        except Exception as e:
                            continue

            if times:
                cage_data = pd.DataFrame({
                    'timestamp': times,
                    'value': values,
                    'cage': f'CAGE {cage_num:02d}'
                })
                all_data.append(cage_data)

        if not all_data:
            st.error("No valid data found in file")
            return None, None, None

        # Combine all data and process time window
        df_processed = pd.concat(all_data, ignore_index=True)
        end_time = df_processed['timestamp'].max()
        start_time = end_time - pd.Timedelta(days=1)

        df_24h = df_processed[
            (df_processed['timestamp'] >= start_time) &
            (df_processed['timestamp'] <= end_time)
        ].copy()

        # Add light/dark cycle
        df_24h['hour'] = df_24h['timestamp'].dt.hour
        df_24h['is_light'] = (df_24h['hour'] >= 7) & (df_24h['hour'] < 19)

        # Calculate results based on parameter type
        if parameter_type in ["XTOT", "XAMB"]:
            results = df_24h.groupby(['cage', 'is_light'])['value'].agg([
                ('Average Activity', 'mean'),
                ('Peak Activity', 'max'),
                ('Total Counts', 'sum')
            ]).unstack()

            # Rename columns exactly as in Colab
            new_columns = []
            for col in results.columns:
                metric, is_light = col
                prefix = "True" if is_light else "False"
                new_columns.append(f"{prefix} ({metric})")
            results.columns = new_columns

            # Add 24h calculations
            results['24h Average'] = df_24h.groupby('cage')['value'].mean()
            results['24h Total Counts'] = df_24h.groupby('cage')['value'].sum()

        elif parameter == "FEED":
            results = df_24h.groupby(['cage', 'is_light'])['value'].agg([
                ('Total Intake', 'sum'),
                ('Average Rate', 'mean'),
                ('Peak Rate', 'max')
            ]).unstack()
            
            # Rename columns to match Colab
            new_columns = []
            for col in results.columns:
                metric, is_light = col
                suffix = "Light" if is_light else "Dark"
                new_columns.append(f"{metric} ({suffix})")
            results.columns = new_columns

        else:  
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
        ).round(3 if parameter_type == "RER" else 2)

        # Ensure all 24 hours are present
        all_hours = pd.Index(range(24), name='hour')
        hourly_results = hourly_results.reindex(all_hours)

        # Rename columns to subject IDs
        hourly_results.columns = [f"{subject_map.get(cage, cage)}" for cage in hourly_results.columns]

        # Add summary statistics
        hourly_results['Mean'] = hourly_results.mean(axis=1).round(3 if parameter_type == "RER" else 2)
        hourly_results['SEM'] = (hourly_results.std(axis=1) / np.sqrt(hourly_results.shape[1])).round(3 if parameter_type == "RER" else 2)

        return results, hourly_results, df_24h

    except Exception as e:
        st.error(f"Error processing file: {str(e)}")
        return None, None, None
def assign_groups(cage_df):
    """
    Allow users to assign groups to detected cages
    """
    st.subheader("Group Assignment")
    
    # First, ask if detected cages are correct
    st.write("Detected cages:")
    st.dataframe(cage_df)
    
    cages_correct = st.radio(
        "Are the detected cages correct?",
        ["Yes", "No"],
        index=0  # Default to Yes
    )
    
    if cages_correct == "No":
        st.error("Please ensure the uploaded file contains the correct cage information.")
        st.stop()
    
    # If cages are correct, proceed with group assignment
    if cages_correct == "Yes":
        # Get number of groups from user
        num_groups = st.number_input("How many groups do you want to create?", 
                                   min_value=1, 
                                   max_value=len(cage_df), 
                                   value=2)
        
        # Create group assignments
        group_assignments = {}
        
        # Use columns for better layout
        for i in range(num_groups):
            st.subheader(f"Group {i + 1}")
            
            # Get group name
            group_name = st.text_input(f"Name for Group {i + 1}", 
                                     value=f"Group {i + 1}")
            
            # Multi-select for cages
            selected_cages = st.multiselect(
                f"Select cages for {group_name}",
                cage_df["Cage"].tolist(),
                key=f"group_{i}"
            )
            
            group_assignments[group_name] = selected_cages
        
        # Validate that all cages are assigned
        all_assigned_cages = [cage for group in group_assignments.values() for cage in group]
        unassigned_cages = set(cage_df["Cage"]) - set(all_assigned_cages)
        
        if unassigned_cages:
            st.warning(f"Warning: The following cages are not assigned to any group: {', '.join(unassigned_cages)}")
        
        # Check for duplicates
        assigned_cages = []
        duplicate_cages = []
        for group, cages in group_assignments.items():
            for cage in cages:
                if cage in assigned_cages:
                    duplicate_cages.append(cage)
                assigned_cages.append(cage)
        
        if duplicate_cages:
            st.error(f"Error: The following cages are assigned to multiple groups: {', '.join(set(duplicate_cages))}")
            st.stop()
        
        # If everything is valid, create a summary dataframe
        group_summary = []
        for group_name, cages in group_assignments.items():
            for cage in cages:
                subject_id = cage_df[cage_df["Cage"] == cage]["Subject ID"].iloc[0]
                group_summary.append({
                    "Group": group_name,
                    "Cage": cage,
                    "Subject ID": subject_id
                })
        
        group_summary_df = pd.DataFrame(group_summary)
        
        # Show the summary
        st.subheader("Group Assignment Summary")
        st.dataframe(group_summary_df)
        
        return group_summary_df
    
    return None

def calculate_activity_results(df_24h):
    """Calculate results for activity parameters (XTOT, XAMB)"""
    results = df_24h.groupby(['cage', 'is_light'])['value'].agg([
        ('Average Activity', 'mean'),
        ('Peak Activity', 'max'),
        ('Total Counts', 'sum')
    ]).unstack()
    
    # Flatten column names
    results.columns = [f"{col[1]} ({'Light' if col[0] else 'Dark'})" for col in results.columns]
    return results

def calculate_feed_results(df_24h):
    """Calculate results for feed data"""
    results = df_24h.groupby(['cage', 'is_light'])['value'].agg([
        ('Total Intake', 'sum'),
        ('Average Rate', 'mean'),
        ('Peak Rate', 'max')
    ]).unstack()
    
    # Flatten column names
    results.columns = [f"{col[1]} ({'Light' if col[0] else 'Dark'})" for col in results.columns]
    return results

def calculate_metabolic_results(df_24h):
    """Calculate results for metabolic parameters (VO2, VCO2, RER, HEAT)"""
    results = df_24h.groupby(['cage', 'is_light'])['value'].mean().unstack()
    results.columns = ['Dark Average', 'Light Average']
    results['Total Average'] = (results['Dark Average'] + results['Light Average']) / 2
    return results

# Parameter selection with descriptions
parameter_descriptions = {
    "VO2": "Oxygen consumption (ml/kg/hr)",
    "VCO2": "Carbon dioxide production (ml/kg/hr)",
    "RER": "Respiratory exchange ratio",
    "HEAT": "Heat production (kcal/hr)",
    "XTOT": "Total activity counts",
    "XAMB": "Ambulatory activity counts",
    "FEED": "Food intake (g)"
}

# Add parameter selection with help text
parameter = st.selectbox(
    "Select Parameter to Analyze",
    list(parameter_descriptions.keys()),
    help="Choose which parameter to analyze from your CLAMS data",
    format_func=lambda x: f"{x}: ~{parameter_descriptions[x]}"
)

# File upload and processing section
uploaded_file = st.file_uploader(f"Choose a {parameter} CSV file", type="csv")

if uploaded_file is not None:
    # First verify file type
    is_valid, error_message = verify_file_type(uploaded_file, parameter)
    
    if not is_valid:
        st.error(error_message)
    else:
        # Show detected cages before processing
        cage_info = extract_cage_info(uploaded_file)
        if cage_info:
            cage_df = pd.DataFrame([
                {"Cage": f"Cage {k}", "Subject ID": v} 
                for k, v in cage_info.items()
            ])
            
            # Add group assignment
            group_assignments = assign_groups(cage_df)
            
            if group_assignments is not None:
                with st.spinner('Processing data...'):
                    results, hourly_results, raw_data = process_clams_data(uploaded_file, parameter)
    
        # Process data
        with st.spinner('Processing data...'):
            results, hourly_results, raw_data = process_clams_data(uploaded_file, parameter)
            
            if results is not None:
                # Display parameter-specific metrics and visualizations
                if parameter in ["XTOT", "XAMB"]:
                    col1, col2, col3, col4 = st.columns(4)
                    with col1:
                        st.metric("Average Light Activity", 
                            f"{results['True (Average Activity)'].mean():.1f} {PARAMETER_UNITS[parameter]}")
                    with col2:
                        st.metric("Average Dark Activity", 
                            f"{results['False (Average Activity)'].mean():.1f} {PARAMETER_UNITS[parameter]}")
                    with col3:
                        st.metric("Peak Activity", 
                            f"{max(results['True (Peak Activity)'].max(), results['False (Peak Activity)'].max()):.0f} {PARAMETER_UNITS[parameter]}")
                    with col4:
                        st.metric("Total Activity", 
                            f"{(results['True (Total Counts)'] + results['False (Total Counts)']).sum():.0f} {PARAMETER_UNITS[parameter]}")
                        
                    # Create line plot with SEM
                        plot_data = hourly_results[['Mean', 'SEM']].reset_index()
                        fig = go.Figure()
                        
                        # Add SEM range
                        fig.add_trace(go.Scatter(
                            x=plot_data['hour'],
                            y=plot_data['Mean'] + plot_data['SEM'],
                            fill=None,
                            mode='lines',
                            line_color='rgba(31, 119, 180, 0.2)',
                            showlegend=False
                        ))
                        fig.add_trace(go.Scatter(
                            x=plot_data['hour'],
                            y=plot_data['Mean'] - plot_data['SEM'],
                            fill='tonexty',
                            mode='lines',
                            line_color='rgba(31, 119, 180, 0.2)',
                            showlegend=False
                        ))
                        
                        # Add mean line
                        fig.add_trace(go.Scatter(
                            x=plot_data['hour'],
                            y=plot_data['Mean'],
                            mode='lines+markers',
                            line_color='rgb(31, 119, 180)',
                            name=f'Mean {parameter}'
                        ))
                        
                        fig.update_layout(
                            title=f'24-Hour {parameter} Pattern',
                            xaxis_title='Hour of Day',
                            yaxis_title=f'Activity ({PARAMETER_UNITS[parameter]})'
                        )
                                    
                elif parameter == "RER":
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("Average Light RER", f"{results['Light Average'].mean():.3f}")
                    with col2:
                        st.metric("Average Dark RER", f"{results['Dark Average'].mean():.3f}")
                    with col3:
                        st.metric("Total Records", len(raw_data))
                    
                    # Create RER plot with confidence interval
                    plot_data = hourly_results[['Mean', 'SEM']].reset_index()
                    fig = go.Figure()
                    fig.add_trace(go.Scatter(
                        x=plot_data['hour'],
                        y=plot_data['Mean'] + plot_data['SEM'],
                        fill=None,
                        mode='lines',
                        line_color='rgba(0,100,80,0.2)',
                        showlegend=False
                    ))
                    fig.add_trace(go.Scatter(
                        x=plot_data['hour'],
                        y=plot_data['Mean'] - plot_data['SEM'],
                        fill='tonexty',
                        mode='lines',
                        line_color='rgba(0,100,80,0.2)',
                        showlegend=False
                    ))
                    fig.add_trace(go.Scatter(
                        x=plot_data['hour'],
                        y=plot_data['Mean'],
                        mode='lines+markers',
                        line_color='rgb(0,100,80)',
                        name='Mean RER'
                    ))
                    fig.update_layout(
                        title='24-Hour RER Pattern',
                        xaxis_title='Hour of Day',
                        yaxis_title='RER'
                    )
                elif parameter == "FEED":
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("Average Light Rate", 
                                f"{results['Average Rate (Light)'].mean():.4f} {PARAMETER_UNITS[parameter]}")
                    with col2:
                        st.metric("Average Dark Rate", 
                                f"{results['Average Rate (Dark)'].mean():.4f} {PARAMETER_UNITS[parameter]}")
                    with col3:
                        st.metric("Total Feed", 
                                f"{(results['Total Intake (Light)'] + results['Total Intake (Dark)']).sum():.4f} {PARAMETER_UNITS[parameter]}")
                    # Create line plot with SEM
                        plot_data = hourly_results[['Mean', 'SEM']].reset_index()
                        fig = go.Figure()
                        
                        # Add SEM range
                        fig.add_trace(go.Scatter(
                            x=plot_data['hour'],
                            y=plot_data['Mean'] + plot_data['SEM'],
                            fill=None,
                            mode='lines',
                            line_color='rgba(31, 119, 180, 0.2)',
                            showlegend=False
                        ))
                        fig.add_trace(go.Scatter(
                            x=plot_data['hour'],
                            y=plot_data['Mean'] - plot_data['SEM'],
                            fill='tonexty',
                            mode='lines',
                            line_color='rgba(31, 119, 180, 0.2)',
                            showlegend=False
                        ))
                        
                        # Add mean line
                        fig.add_trace(go.Scatter(
                            x=plot_data['hour'],
                            y=plot_data['Mean'],
                            mode='lines+markers',
                            line_color='rgb(31, 119, 180)',
                            name=f'Mean Feed'
                        ))
                        
                        fig.update_layout(
                            title='24-Hour Feeding Pattern',
                            xaxis_title='Hour of Day',
                            yaxis_title=f'Feed Rate ({PARAMETER_UNITS[parameter]}/hr)'
                        )        
                else:  # VO2, VCO2, HEAT
                    # Common metrics for metabolic parameters
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric(f"Average Light {parameter}", 
                                f"{results['Light Average'].mean():.2f} {PARAMETER_UNITS[parameter]}")
                    with col2:
                        st.metric(f"Average Dark {parameter}", 
                                f"{results['Dark Average'].mean():.2f} {PARAMETER_UNITS[parameter]}")
                    with col3:
                        st.metric("Total Records", len(raw_data))
                    
                    # Create line plot with SEM
                    plot_data = hourly_results[['Mean', 'SEM']].reset_index()
                    fig = go.Figure()
                    
                    # Add SEM range
                    fig.add_trace(go.Scatter(
                        x=plot_data['hour'],
                        y=plot_data['Mean'] + plot_data['SEM'],
                        fill=None,
                        mode='lines',
                        line_color='rgba(31, 119, 180, 0.2)',
                        showlegend=False
                    ))
                    fig.add_trace(go.Scatter(
                        x=plot_data['hour'],
                        y=plot_data['Mean'] - plot_data['SEM'],
                        fill='tonexty',
                        mode='lines',
                        line_color='rgba(31, 119, 180, 0.2)',
                        showlegend=False
                    ))
                    
                    # Add mean line
                    fig.add_trace(go.Scatter(
                        x=plot_data['hour'],
                        y=plot_data['Mean'],
                        mode='lines+markers',
                        line_color='rgb(31, 119, 180)',
                        name=f'Mean {parameter}'
                    ))
                    
                    fig.update_layout(
                        title=f'24-Hour {parameter} Pattern',
                        xaxis_title='Hour of Day',
                        yaxis_title=f'{parameter} ({PARAMETER_UNITS[parameter]})'
                    )
                
                # Display the plot
                st.plotly_chart(fig, use_container_width=True)
                
                # Detect outliers
                results_outliers = detect_outliers(results.select_dtypes(include=[np.number]))
                hourly_outliers = detect_outliers(hourly_results)
                
                # Add explanatory text for outliers
                st.info("📊 Values highlighted in red are potential outliers (> 2 standard deviations from the mean)")
                
                # Download buttons with enhanced styling
                col1, col2 = st.columns(2)
                with col1:
                    csv_light_dark = results.to_csv().encode('utf-8')
                    st.download_button(
                        label=f"📥 Download {parameter} Light/Dark Averages",
                        data=csv_light_dark,
                        file_name=f"{parameter}_lightdark_averages.csv",
                        mime="text/csv",
                        help=f"Download the light/dark cycle analysis results for {parameter}"
                    )
                
                with col2:
                    csv_hourly = hourly_results.to_csv().encode('utf-8')
                    st.download_button(
                        label=f"📥 Download {parameter} Hourly Averages",
                        data=csv_hourly,
                        file_name=f"{parameter}_hourly_averages.csv",
                        mime="text/csv",
                        help=f"Download the hourly analysis results for {parameter}"
                    )
                
                # Display data tables with outlier highlighting
                st.subheader(f"{parameter} Light/Dark Analysis")
                if results is not None:
                    st.dataframe(style_dataframe(results))
                
                st.subheader(f"{parameter} Hourly Analysis")
                if hourly_results is not None:
                    st.dataframe(style_dataframe(hourly_results))
                
                # Add footer with analysis details
                st.markdown("---")
                st.markdown(f"""
                **Analysis Details:**
                - Analysis Period: {raw_data['timestamp'].min().strftime('%Y-%m-%d %H:%M')} to {raw_data['timestamp'].max().strftime('%Y-%m-%d %H:%M')}
                - Light Cycle: 7:00 AM - 7:00 PM
                - Dark Cycle: 7:00 PM - 7:00 AM
                - Total Records Processed: {len(raw_data):,}
                """)

# Add an About section at the bottom
with st.expander("ℹ️ About CLAMS Data Analyzer"):
    st.markdown("""
    This tool processes and analyzes data from the Comprehensive Lab Animal Monitoring System (CLAMS).
    
    **Available Parameters:**
    - **VO2**: Oxygen consumption (ml/kg/hr)
    - **VCO2**: Carbon dioxide production (ml/kg/hr)
    - **RER**: Respiratory exchange ratio
    - **HEAT**: Heat production (kcal/hr)
    - **XTOT**: Total activity counts
    - **XAMB**: Ambulatory activity counts
    - **FEED**: Food intake (g)
    
    **Features:**
    - Automatic light/dark cycle analysis
    - Hourly pattern visualization
    - Outlier detection
    - Data export for further analysis
    """)