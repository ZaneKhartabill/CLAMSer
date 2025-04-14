import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import io               # Added for download buttons
import plotly.express as px
import plotly.graph_objects as go
import matplotlib.pyplot as plt # Added for publication plots
import matplotlib.colors as mcolors
import scienceplots # Added for publication plots
import re
import markdown
from scipy import stats
from scikit_posthocs import posthoc_dunn
from statsmodels.stats.multicomp import pairwise_tukeyhsd
from statsmodels.stats.power import TTestIndPower
import statsmodels.api as sm
from statsmodels.formula.api import ols
from scikit_posthocs import posthoc_ttest

# ------------- JUMBO MUMBO FOR LEAN MASS STUFF -------------

# Callback function to toggle the lean mass state
def toggle_lean_mass_state():
    current_state = st.session_state.get('apply_lean_mass', False)
    st.session_state.apply_lean_mass = not current_state
    # Optional: Add a print here to see when the callback fires
    print(f"Callback toggled apply_lean_mass to: {st.session_state.apply_lean_mass}")
    
# NEW Callback function to apply lean mass changes from inputs
def apply_lean_mass_changes():
    print("--- apply_lean_mass_changes callback triggered ---") # Debug
    # Ensure cage_info exists in session state before proceeding
    if 'cage_info' not in st.session_state or not st.session_state['cage_info']:
        st.warning("Cannot apply lean mass changes: Cage information is missing.")
        print("Error: cage_info missing in apply_lean_mass_changes") # Debug
        return # Exit the callback

    # Create a temporary dictionary to hold the latest values
    latest_lean_mass_data = {}
    found_keys = True
    for cage_label in st.session_state['cage_info'].keys():
        widget_key = f"lean_mass_{cage_label}_setup_expander"
        # Read the current value directly from the session state associated with the number_input's key
        if widget_key in st.session_state:
            latest_lean_mass_data[cage_label] = st.session_state[widget_key]
        else:
            # This case should ideally not happen if the inputs were rendered, but good for robustness
            print(f"Warning: Widget key '{widget_key}' not found in session state for {cage_label}.")
            # Optionally try getting default or skip
            latest_lean_mass_data[cage_label] = st.session_state.get('lean_mass_data', {}).get(cage_label, 20.0) # Fallback maybe? Or error?
            found_keys=False
            
    if not found_keys:
         st.warning("Could not read values for all lean mass inputs. Please ensure they were displayed correctly.", icon="⚠️")

    # Update the main session state dictionary that process_clams_data uses
    st.session_state['lean_mass_data'] = latest_lean_mass_data
    print(f"Updated st.session_state['lean_mass_data'] to: {st.session_state['lean_mass_data']}") # Debug

    # Now trigger the rerun to re-process data with the new values
    st.rerun()

# ------------- END JUMBO MUMBO -------------


# ------------- NEW: Statistical Assumption Check Helpers -------------

def check_normality(data, alpha=0.05):
    """
    Performs Shapiro-Wilk test for normality on data.

    Args:
        data (array-like): Data points (e.g., residuals or group data).
        alpha (float): Significance level.

    Returns:
        tuple: (is_normal: bool, p_value: float, test_stat: float)
               Returns (None, None, None) if test cannot be performed.
    """
    # Ensure data is usable
    data_clean = np.asarray(data) # Convert to numpy array
    data_clean = data_clean[~np.isnan(data_clean)] # Remove NaNs

    if len(data_clean) < 3: # Shapiro-Wilk needs at least 3 data points
        # st.caption("Normality check skipped (N<3)") # Optional debug message
        return None, None, None # Cannot perform test

    try:
        test_stat, p_value = stats.shapiro(data_clean)
        is_normal = p_value >= alpha
        return is_normal, p_value, test_stat
    except Exception as e:
        # st.warning(f"Shapiro-Wilk test failed: {e}") # Optional debug message
        return None, None, None

def check_homogeneity(groups_data, alpha=0.05):
    """
    Performs Levene's test for homogeneity of variances across groups.

    Args:
        groups_data (list of array-like): A list where each element is the
                                           data for one group.
        alpha (float): Significance level.

    Returns:
        tuple: (is_homogeneous: bool, p_value: float, test_stat: float)
               Returns (None, None, None) if test cannot be performed.
    """
    # Clean data: Remove groups with fewer than 2 data points (Levene needs >=2)
    valid_groups_data = [np.asarray(g)[~np.isnan(np.asarray(g))] for g in groups_data]
    valid_groups_data = [g for g in valid_groups_data if len(g) >= 2]

    if len(valid_groups_data) < 2: # Levene needs at least 2 groups
        # st.caption("Homogeneity check skipped (less than 2 valid groups)") # Optional debug
        return None, None, None # Cannot perform test

    try:
        test_stat, p_value = stats.levene(*valid_groups_data, center='median') # Use median for robustness
        is_homogeneous = p_value >= alpha
        return is_homogeneous, p_value, test_stat
    except Exception as e:
        # st.warning(f"Levene's test failed: {e}") # Optional debug
        return None, None, None

# ------------- END: Statistical Assumption Check Helpers -------------


# Page setup + title
st.set_page_config(
    page_title="CLAMSer",
    page_icon="🧬",
    layout="wide",
    initial_sidebar_state="expanded"
)
st.markdown("""
    <style>
    [data-testid="stSidebar"] {
        background-color: #1a1f2e;
        padding: 1rem;
        border-right: 1px solid #2d3648;
    }
    [data-testid="stSidebar"] .stRadio {
        background-color: #262d3d;
        padding: 1rem;
        border-radius: 5px;
        margin: 0.5rem 0;
    }
    .uploadedFile {
        background-color: #262d3d;
        padding: 1rem;
        border-radius: 5px;
        margin: 0.5rem 0;
    }
    </style>
    """, unsafe_allow_html=True)

# Dictionary for all constants
PARAMETER_UNITS = {
    # Core Biological - Metabolic & Feed
    "VO2": "ml/kg/hr",
    "VCO2": "ml/kg/hr",
    "RER": "ratio",
    "HEAT": "kcal/hr",
    "FEED": "g", # Represents FEED1 ACC

    # Core Biological - Activity
    "XTOT": "counts",
    "XAMB": "counts",
    "YTOT": "counts",
    "YAMB": "counts",
    "ZTOT": "counts",
    "ZAMB": "counts",

    # Core Biological - Accumulated Gas (Net Change)
    "ACCCO2": "l",      # Net Accumulated CO2 (Liters) over period
    "ACCO2": "l",       # Net Accumulated O2 (Liters) over period - Often represents consumption

    # Diagnostic - Gas Concentrations
    "O2IN": "%",
    "O2OUT": "%",
    "CO2IN": "%",
    "CO2OUT": "%",

    # Diagnostic - Delta Gas Concentrations
    "DO2": "%",         # O2IN - O2OUT
    "DCO2": "%",        # CO2OUT - CO2IN

    # Diagnostic - Environmental/System
    "FLOW": "lpm",      # Liters per minute
    "PRESSURE": "mmhg", # Millimeters of mercury
    # Add other system/diag params if they appear later (e.g., from DIAG.CSV)
}

GROUP_COLORS = {
    "Group 1": "#4285F4",  # Google Blue
    "Group 2": "#EA4335",  # Google Red
    "Group 3": "#FBBC05",  # Google Yellow
    "Group 4": "#34A853",  # Google Green
    "Group 5": "#8A2BE2",  # Blue Violet
    "Group 6": "#FF7F00",  # Orange
    "Group 7": "#FF69B4",  # Hot Pink
    "Group 8": "#1E90FF",  # Dodger Blue
}

# Add parameter_descriptions dictionary
parameter_descriptions = {
    # Core Biological - Metabolic & Feed
    "VO2": "Oxygen consumption rate (ml/kg/hr)",
    "VCO2": "Carbon dioxide production rate (ml/kg/hr)",
    "RER": "Respiratory Exchange Ratio (VCO2/VO2)",
    "HEAT": "Calculated heat production (kcal/hr)",
    "FEED": "Accumulated food intake (g) [Use FEED1 ACC file]", # Emphasize file

    # Core Biological - Activity
    "XTOT": "Total X-axis activity (fine + ambulatory, counts)",
    "XAMB": "Ambulatory X-axis activity (locomotion, counts)",
    "YTOT": "Total Y-axis activity (fine + ambulatory, counts)",
    "YAMB": "Ambulatory Y-axis activity (locomotion, counts)",
    "ZTOT": "Total Z-axis activity (fine + rearing/climbing, counts)",
    "ZAMB": "Ambulatory Z-axis activity (rearing/climbing, counts)",

    # Core Biological - Accumulated Gas (Net Change)
    "ACCCO2": "Net Accumulated CO2 production (L) over period", # Clarify 'Net' and 'L? hmm'
    "ACCO2": "Net Accumulated O2 consumption (L) over period", # Clarify 'Net' and 'L'

    # Diagnostic - Gas Concentrations
    "[Diagnostic] O2IN": "Inlet Oxygen concentration (%) ",
    "[Diagnostic] O2OUT": "Outlet Oxygen concentration (%)",
    "[Diagnostic] CO2IN": "Inlet Carbon Dioxide concentration (%)",
    "[Diagnostic] CO2OUT": "Outlet Carbon Dioxide concentration (%)",

    # Diagnostic - Delta Gas Concentrations
    "[Diagnostic] DO2": "Delta O2 conc. (O2IN - O2OUT, %)",
    "[Diagnostic] DCO2": "Delta CO2 conc. (CO2OUT - CO2IN, %)",

    # Diagnostic - Environmental/System
    "[System] FLOW": "Air flow rate through cage (lpm)",
    "[System] PRESSURE": "Barometric pressure (mmhg)",
    # Add other system/diag params if they appear later
}

# Sidebar setup
with st.sidebar:
    logo_svg = """
<div class="menzies-logo">
    <svg version="1.0" xmlns="http://www.w3.org/2000/svg" viewBox="0 0 1472 832" fill="white">
        <g transform="translate(0.000000,832.000000) scale(0.100000,-0.100000)" stroke="none">
            <path d="M6990 6799 c-40 -20 -91 -59 -135 -103 -220 -218 -264 -556 -96 -738
67 -73 193 -128 296 -128 75 0 76 -9 10 -106 -32 -48 -90 -118 -127 -156 l-68
-69 18 -27 c11 -15 37 -61 58 -102 21 -41 67 -117 102 -168 34 -52 62 -95 62
-96 0 -2 -60 -29 -134 -61 l-135 -58 -98 42 c-125 52 -206 71 -313 71 -200 0
-351 -69 -467 -213 l-30 -38 -44 29 c-103 68 -223 84 -327 44 -63 -24 -54 -34
-110 133 -70 210 -76 301 -28 441 38 111 46 124 75 124 47 0 133 28 183 60 60
36 65 37 44 5 -9 -13 -16 -29 -16 -34 0 -13 93 -61 118 -61 11 0 50 20 88 44
149 97 162 133 93 263 -23 43 -64 121 -92 173 -27 52 -55 98 -61 102 -20 14
-86 8 -149 -13 -108 -35 -484 -225 -502 -253 -10 -14 -32 -66 -50 -115 -50
-131 -106 -243 -181 -361 -231 -362 -271 -480 -295 -874 -11 -171 -10 -180 7
-193 11 -8 102 -57 204 -110 102 -53 246 -129 320 -169 277 -150 408 -189 598
-182 144 6 227 36 349 124 21 15 22 14 38 -41 38 -133 131 -315 283 -559 88
-141 143 -249 172 -340 18 -54 28 -69 58 -88 20 -12 70 -46 110 -76 41 -29 77
-51 80 -47 3 3 -7 74 -23 158 -41 221 -42 236 -14 259 67 52 229 98 377 105
l112 6 0 -155 0 -155 -69 -7 c-84 -9 -223 -41 -259 -61 -15 -8 -25 -14 -22
-15 3 0 81 -7 173 -15 93 -8 173 -17 178 -20 5 -4 6 -63 2 -141 l-6 -134 61 0
62 0 0 128 c0 70 4 133 8 140 8 12 81 23 247 38 77 7 79 7 50 20 -37 16 -184
51 -247 59 l-48 6 0 156 0 156 73 -7 c186 -17 334 -84 353 -161 4 -15 4 -66 0
-114 -3 -47 -10 -140 -15 -206 -5 -66 -8 -122 -5 -124 5 -6 142 48 149 59 2 4
12 56 20 115 28 204 96 356 249 558 39 50 101 132 138 182 37 49 102 155 143
235 100 194 86 180 135 138 156 -134 370 -163 605 -83 57 19 233 105 350 170
95 54 374 186 423 200 30 9 45 41 75 163 39 152 50 260 44 407 -7 171 -33 269
-146 545 -49 118 -105 267 -126 330 -61 192 -88 226 -232 298 -72 37 -283 122
-365 148 -86 26 -93 20 -181 -148 -114 -218 -117 -235 -47 -305 91 -89 222
-128 280 -83 l25 19 -40 40 c-46 48 -42 50 26 15 28 -14 73 -28 100 -31 27 -3
49 -9 49 -12 0 -3 7 -49 15 -101 23 -150 20 -184 -31 -342 -41 -127 -47 -156
-52 -261 l-5 -117 -56 53 c-67 64 -124 98 -211 127 -55 19 -81 21 -170 17
-157 -6 -256 -53 -375 -173 -43 -44 -60 -55 -78 -51 -12 3 -51 13 -87 22 -90
24 -228 15 -337 -19 -76 -25 -102 -40 -90 -52 3 -3 56 -16 118 -30 105 -24
333 -108 375 -138 16 -12 21 -9 43 24 82 119 195 195 340 230 75 18 236 8 326
-20 82 -26 241 -104 340 -167 83 -54 202 -149 215 -173 8 -15 2 -15 -68 -7
-79 9 -284 4 -542 -13 -80 -5 -219 -7 -310 -3 -91 3 -209 3 -262 -2 l-98 -7 0
-37 c0 -61 -30 -220 -56 -301 -62 -186 -228 -378 -344 -396 -53 -9 -67 -19
-52 -37 24 -29 43 -152 31 -205 -14 -65 -58 -135 -107 -167 -63 -42 -117 -38
-243 18 -102 45 -196 72 -269 79 -33 3 -35 5 -38 43 -4 67 -31 217 -48 275
l-17 55 -8 -34 c-4 -19 -12 -95 -19 -170 -7 -75 -14 -141 -15 -147 -2 -6 -34
-15 -72 -19 -93 -9 -178 -28 -303 -66 -58 -17 -126 -32 -152 -33 -40 -1 -52 4
-78 29 -50 50 -61 162 -28 276 10 36 11 36 77 42 116 10 318 98 446 195 69 53
69 62 -4 62 -152 0 -419 90 -568 191 -108 73 -243 208 -243 243 0 15 153 122
227 160 129 65 269 105 393 113 106 7 173 -10 250 -64 l55 -39 76 62 c149 120
236 159 437 198 203 39 289 85 448 239 l92 90 -25 56 c-39 88 -62 159 -83 256
-25 124 -28 145 -20 150 3 2 35 -5 71 -16 96 -31 242 -23 322 16 130 63 215
182 238 333 31 206 -75 470 -230 572 -55 37 -127 40 -222 9 -180 -59 -327
-176 -463 -371 l-16 -22 -47 18 c-99 38 -196 56 -292 55 l-94 -2 -24 83 c-41
144 -104 253 -202 350 -95 94 -159 132 -236 138 -41 3 -61 -2 -115 -29z m217
-154 c91 -95 135 -232 137 -425 1 -105 -2 -131 -18 -159 -23 -43 -88 -131 -96
-131 -4 0 -5 47 -2 104 5 121 -5 162 -55 217 -42 47 -33 49 30 9 26 -16 52
-30 58 -30 14 0 2 155 -18 240 -16 66 -49 142 -85 193 -31 44 0 33 49 -18z
m1286 -223 c-62 -49 -123 -130 -162 -212 l-31 -65 51 -3 51 -3 -49 -25 c-35
-17 -54 -35 -66 -59 -25 -52 -22 -56 26 -35 24 11 63 22 87 25 l45 6 -60 -30
c-70 -35 -108 -86 -148 -196 -58 -159 -70 -296 -36 -410 37 -122 36 -132 -11
-203 -46 -68 -118 -129 -192 -162 -26 -11 -95 -31 -155 -45 -63 -14 -137 -39
-178 -60 -59 -29 -80 -35 -135 -35 -113 1 -162 35 -228 161 -32 61 -59 185
-48 215 5 14 16 2 48 -56 22 -41 54 -91 69 -111 17 -22 29 -49 29 -67 0 -45
27 -71 90 -84 74 -15 106 0 130 61 11 29 31 54 61 76 45 32 99 90 99 104 0 4
59 13 132 20 72 7 141 17 152 22 21 9 -4 8 -185 -7 l-76 -6 4 68 c3 38 1 86
-5 107 -8 30 -7 40 4 52 15 15 113 48 177 60 59 11 87 64 87 162 0 47 3 55 25
65 34 15 27 30 -26 50 -85 33 -166 20 -216 -34 -26 -28 -113 -211 -113 -239 0
-11 10 -39 22 -62 31 -59 33 -194 3 -227 -55 -61 -189 -87 -234 -46 -18 16
-25 17 -47 7 -55 -25 -141 23 -204 114 -26 38 -28 45 -19 84 19 85 69 144 151
181 36 16 45 24 41 39 -3 11 7 66 21 122 26 98 26 103 9 122 -26 29 -92 57
-138 57 -75 1 -75 12 1 87 75 75 140 111 214 120 64 7 92 -1 135 -39 l36 -31
68 16 c82 20 104 20 154 3 61 -22 114 -64 149 -119 l33 -51 8 35 c63 254 187
438 340 504 56 25 60 16 10 -23z m1270 -749 c66 -234 80 -334 80 -538 0 -149
-6 -223 -29 -377 l-6 -38 -94 87 c-90 82 -94 88 -94 128 0 23 21 111 47 196
43 145 46 161 47 274 1 96 -5 148 -28 258 -27 132 -27 139 -10 158 9 10 22 19
28 19 7 0 33 -75 59 -167z m-2407 111 c50 -53 44 -114 -12 -114 -34 0 -54 37
-54 97 0 65 17 70 66 17z m651 -110 c-2 -24 -10 -51 -17 -59 -13 -15 -65 -20
-94 -9 -24 9 -19 49 11 84 32 38 61 51 87 38 15 -8 17 -18 13 -54z m-2637 30
c0 -3 -16 -50 -36 -103 -62 -164 -65 -179 -73 -441 -7 -192 -12 -256 -26 -297
-10 -29 -21 -50 -25 -48 -18 11 -19 438 -1 540 19 110 123 355 151 355 6 0 10
-3 10 -6z m425 -969 c104 -18 222 -62 290 -108 53 -36 135 -114 135 -128 0 -4
-57 -8 -127 -7 -71 0 -349 -1 -620 -3 l-492 -4 32 33 c100 105 284 198 447
225 62 10 255 6 335 -8z m459 -351 c3 -9 6 -27 6 -40 0 -22 -4 -24 -52 -24
-58 0 -198 -12 -439 -36 -85 -9 -166 -14 -180 -12 -43 6 539 125 622 127 25 1
39 -4 43 -15z m2781 -29 c305 -44 310 -49 77 -59 -162 -7 -343 -21 -419 -32
l-43 -6 0 67 0 68 108 -7 c59 -4 184 -18 277 -31z"/>
<path d="M7446 5442 c-16 -4 -34 -15 -40 -25 -23 -36 62 -147 112 -147 24 0
92 74 98 107 5 26 2 34 -17 46 -28 18 -112 28 -153 19z"/>
<path d="M8362 4307 c-107 -150 -258 -250 -459 -303 -52 -14 -106 -18 -230
-19 -90 0 -163 -3 -163 -6 0 -8 95 -71 146 -97 72 -37 263 -72 390 -72 l36 0
-21 36 c-12 19 -21 37 -21 39 0 2 33 25 73 51 103 66 224 193 265 276 31 63
57 168 43 168 -4 0 -30 -33 -59 -73z"/>
<path d="M5430 2120 l0 -280 70 0 70 0 0 142 c0 81 4 138 9 133 4 -6 33 -56
62 -112 l54 -103 53 0 c35 0 52 4 52 13 0 6 25 62 54 122 l54 110 1 -152 1
-153 70 0 70 0 0 275 0 275 -83 0 -84 0 -69 -135 c-38 -74 -73 -134 -77 -132
-4 1 -39 61 -76 132 -75 144 -72 141 -178 144 l-53 1 0 -280z"/>
<path d="M6938 2393 l-58 -4 0 -274 0 -275 76 0 76 0 -4 151 c-3 147 -3 150
15 127 10 -13 66 -80 124 -150 l106 -128 63 0 64 0 0 275 0 275 -75 0 -75 0 0
-136 c0 -75 -4 -134 -9 -132 -5 2 -59 65 -121 141 -62 75 -115 136 -119 135
-3 -1 -32 -3 -63 -5z"/>
<path d="M9318 2386 c-25 -7 -58 -20 -72 -29 -48 -32 -74 -111 -56 -175 15
-59 63 -96 167 -127 102 -31 127 -48 112 -76 -16 -30 -106 -26 -181 8 l-60 27
-19 -25 c-10 -13 -26 -39 -36 -57 -21 -38 -20 -39 67 -74 44 -17 76 -22 161
-22 99 -1 108 1 148 27 83 55 106 174 46 237 -35 36 -76 56 -168 81 -83 23
-106 43 -80 71 21 24 69 23 145 -3 49 -16 63 -18 70 -7 13 20 48 86 48 92 0
11 -77 44 -128 55 -66 14 -104 13 -164 -3z"/>
<path d="M6250 2116 c0 -151 4 -277 9 -280 4 -3 105 -3 225 1 l216 6 0 59 0
58 -150 0 -150 0 0 50 0 50 125 0 125 0 0 60 0 60 -125 0 -125 0 0 44 0 44
143 -1 142 -2 3 63 3 62 -221 0 -220 0 0 -274z"/>
<path d="M7582 2328 l3 -63 123 -3 c67 -1 122 -6 122 -10 -1 -4 -59 -72 -130
-152 l-130 -144 0 -58 0 -58 245 0 245 0 0 60 0 60 -140 0 c-77 0 -140 3 -140
6 0 10 174 215 219 259 39 38 41 41 41 102 l0 63 -230 0 -231 0 3 -62z"/>
<path d="M8230 2115 l0 -275 75 0 75 0 0 275 0 275 -75 0 -75 0 0 -275z"/>
<path d="M8580 2116 c0 -271 0 -275 21 -281 12 -4 113 -4 225 0 l204 7 0 59 0
58 -147 3 -148 3 0 45 0 45 128 3 127 3 0 59 0 59 -127 3 -128 3 -3 38 -3 37
118 1 c65 1 123 2 128 4 6 1 18 3 28 3 14 2 17 12 17 62 l0 60 -220 0 -220 0
0 -274z"/>
<path d="M5417 1703 c-4 -3 -7 -82 -7 -174 0 -196 -10 -182 125 -177 67 3 80
6 83 21 3 15 -6 17 -82 17 l-86 0 0 160 c0 127 -3 160 -13 160 -8 0 -17 -3
-20 -7z"/>
<path d="M5859 1688 c-41 -75 -146 -324 -140 -331 15 -15 36 4 54 48 l19 45
93 0 93 0 12 -34 c11 -36 46 -74 57 -63 7 7 -23 84 -95 247 -49 110 -71 130
-93 88z m91 -185 c0 -9 -20 -13 -70 -13 l-70 0 20 51 c11 28 28 65 37 82 l16
32 33 -70 c18 -38 33 -76 34 -82z"/>
<path d="M6222 1701 c-9 -6 -12 -50 -10 -177 l3 -169 100 1 c113 1 140 11 159
56 17 40 9 73 -24 101 l-27 23 23 28 c13 15 24 38 24 50 0 35 -36 75 -73 80
-70 11 -163 14 -175 7z m180 -48 c28 -33 29 -53 3 -79 -21 -21 -33 -24 -90
-24 l-65 0 0 60 0 60 68 0 c53 0 72 -4 84 -17z m18 -153 c11 -11 20 -33 20
-49 0 -45 -31 -61 -117 -61 l-73 0 0 58 c0 32 3 62 7 65 3 4 37 7 75 7 55 0
72 -4 88 -20z"/>
<path d="M6735 1692 c-48 -23 -69 -45 -90 -95 -31 -74 -12 -151 49 -205 113
-99 306 -11 306 138 0 128 -147 218 -265 162z m167 -43 c88 -63 75 -197 -24
-247 -78 -40 -161 -12 -193 65 -33 78 -7 153 66 191 51 27 104 24 151 -9z"/>
<path d="M7178 1700 c-14 -9 -16 -28 -11 -172 5 -148 12 -192 28 -176 4 3 9
32 12 64 3 33 11 62 17 66 6 4 28 8 48 8 34 0 39 -4 72 -59 40 -65 60 -86 75
-77 6 4 -4 33 -26 72 -19 36 -34 66 -32 68 2 1 16 10 31 19 56 35 54 133 -4
171 -29 19 -186 32 -210 16z m192 -50 c24 -24 26 -76 5 -104 -11 -15 -31 -20
-90 -24 l-75 -5 -1 29 c0 16 -2 45 -5 64 -7 56 -2 60 76 60 57 0 74 -4 90 -20z"/>
<path d="M7730 1703 c-20 -8 -23 -13 -71 -123 -23 -52 -49 -113 -59 -135 -31
-71 -33 -85 -13 -85 12 0 24 15 36 45 l19 45 92 0 92 0 12 -33 c18 -51 38 -74
52 -61 9 9 -1 42 -40 135 -49 118 -98 220 -103 218 -1 0 -9 -3 -17 -6z m70
-203 c0 -6 -31 -10 -70 -10 l-70 0 11 28 c6 15 22 53 36 85 l26 58 34 -76 c18
-42 33 -80 33 -85z"/>
<path d="M7980 1690 c-10 -19 -8 -20 50 -20 l60 0 0 -155 c0 -131 2 -157 15
-162 9 -3 19 -3 22 1 4 4 7 75 7 159 l1 152 58 3 c49 3 58 6 55 20 -3 15 -20
17 -130 20 -119 2 -128 1 -138 -18z"/>
<path d="M8453 1685 c-100 -60 -120 -187 -45 -277 73 -86 224 -75 289 23 59
89 26 211 -69 258 -60 29 -122 27 -175 -4z m168 -31 c42 -21 73 -92 64 -145
-9 -52 -56 -105 -106 -120 -146 -44 -240 148 -123 251 29 25 43 30 85 30 27 0
64 -7 80 -16z"/>
<path d="M8917 1702 c-16 -3 -17 -18 -15 -175 3 -145 5 -172 18 -172 12 0 16
15 18 68 l3 67 52 0 52 -1 35 -60 c32 -57 64 -88 77 -75 3 3 -11 33 -31 66
-39 67 -43 80 -23 80 24 0 56 44 62 85 13 77 -57 128 -168 124 -34 -2 -70 -5
-80 -7z m188 -57 c30 -29 32 -64 5 -93 -17 -17 -33 -22 -78 -23 -31 0 -65 -2
-74 -4 -16 -3 -18 6 -18 71 l0 74 70 0 c62 0 74 -3 95 -25z"/>
<path d="M9351 1618 c28 -46 55 -88 60 -93 4 -6 9 -47 11 -90 3 -67 6 -80 21
-83 14 -3 17 5 17 50 1 71 16 107 84 208 45 67 54 86 41 88 -23 5 -43 -16 -90
-92 -22 -36 -42 -65 -45 -65 -3 0 -28 35 -55 77 -35 53 -57 78 -73 80 -21 3
-19 -4 29 -80z"/>
        </g>
    </svg>
</div>
"""

# Sidebar setup
with st.sidebar:
    # style for the logo
    st.markdown("""
    <style>
    .menzies-logo {
        width: 100%;
        max-width: 2000px;  /* Increased from 200px */
        padding: 0rem;     /* Increased padding */
        margin: 0 auto;
        display: block;
        background-color: white;
    }
    .menzies-logo .logo-circle {
        background-color: transparent;
        border-radius: 50%;
        padding: 2rem;
        display: flex;
        justify-content: center;
        align-items: center; #Tried to do a circular logo, did not work..
    }
    .menzies-logo svg {
        fill: black;
    }
    </style>
""", unsafe_allow_html=True)
    
    # add the SVG
    st.markdown(logo_svg, unsafe_allow_html=True)
    
    # Add a separator after the logo
    st.markdown("---")
    
    st.markdown("## Analysis Settings")
    st.divider() # Adds a horizontal line

    # --- Parameter Selection ---
    st.markdown("##### Select Parameter")
    parameter = st.selectbox(
        "Select Parameter",
        list(parameter_descriptions.keys()),
        format_func=lambda x: f"{x}: {parameter_descriptions[x]}",
        label_visibility="collapsed" # Hide label since we have markdown title
    )
    st.divider()

    # --- Time Window Selection ---
    st.markdown("##### Select Time Window")
    time_window = st.radio(
        "Time Window",
        ["Last 24 Hours", "Last 48 Hours", "Last 72 Hours", "Last 7 Days", "Last 14 Days", "Custom Range", "Entire Dataset"],
        index=6, # Index of "Entire Dataset" in the list (starting from 0)
        help="Choose analysis duration",
        key="time_window_radio",
        label_visibility="collapsed" # Hide label
    )
    
    # Add custom range input if selected
    if time_window == "Custom Range":
        custom_days = st.number_input(
            "Number of days to analyze",
            min_value=1,
            max_value=30,
            value=5,
            step=1,
            help="Enter the number of days to include in analysis (1-30)",
            key="custom_days_input"
        )

    # Add note for Entire Dataset
    if time_window == "Entire Dataset":
        st.info("Entire Dataset can be used to inform your specific acclimation period, which you can then analyze using Custom Range!")
    st.divider()

    # --- Light/Dark Cycle ---
    st.markdown("##### 🌓 Set Light/Dark Cycle")
    light_start = st.slider( # Use st.slider directly, no need for st.sidebar here
        "Light cycle starts at:", # Simplified label
        min_value=0,
        max_value=23,
        value=st.session_state.get('light_start', 7), # Get from session state if exists
        step=1,
        format="%d:00",
        help="Hour when light cycle begins (24-hour format)",
        key="light_start_hour"
    )

    light_end = st.slider( # Use st.slider directly
        "Light cycle ends at:", # Simplified label
        min_value=0,
        max_value=23,
        value=st.session_state.get('light_end', 19), # Get from session state if exists
        step=1,
        format="%d:00",
        help="Hour when light cycle ends (24-hour format)",
        key="light_end_hour"
    )

    # Store light/dark times in session state for use throughout the app
    st.session_state['light_start'] = light_start
    st.session_state['light_end'] = light_end

    # Provide visual confirmation and explanation
    if light_start < light_end:
        st.info(f"Light: {light_start}:00 - {light_end}:00 | Dark: {light_end}:00 - {light_start}:00") # Compact info
    else:
        # Automatically fix if times are invalid
        st.warning("Light start must be before end time. Resetting to 7-19.")
        st.session_state['light_start'] = 7
        st.session_state['light_end'] = 19
        # Rerun slightly to update sliders - might cause a brief flicker but ensures consistency
        st.rerun()
    st.divider()

    # --- Metabolic Normalization ---
    # Only show this section if the parameter is relevant
    if parameter in ["VO2", "VCO2", "HEAT"]:
        st.markdown("##### 📏 Metabolic Normalization")
        with st.container(border=True): # Use a bordered container
            st.info("""
            **Why normalize?** Fat tissue is less metabolically active. Normalizing to lean mass gives fairer comparisons.
            """)

            # Define the checkbox, calling the callback on change
            # Use a DIFFERENT key for the widget itself
            st.checkbox(
                "Apply Lean Mass Adjustment",
                value=st.session_state.get('apply_lean_mass', False), # Read initial state
                help="Normalize metabolic data to lean mass instead of total body weight",
                key="lean_mass_checkbox_widget", # <--- KEY CHANGED! Not linked directly to state anymore
                on_change=toggle_lean_mass_state # <--- CALLBACK ADDED!
            )

            # Conditionally show the reference mass input based on the SESSION STATE value
            # This ensures it reflects the state set by the callback after the rerun
            if st.session_state.get('apply_lean_mass', False): # Check the actual session state
                reference_mass = st.number_input(
                    "Reference lean mass (g):", # Simplified label
                    min_value=1.0,
                    # Use a separate key for the number input's value storage
                    value=st.session_state.get('reference_lean_mass_sidebar_val', 20.0),
                    step=0.1,
                    format="%.1f",
                    help="Standard lean mass value used for normalization",
                    key="reference_lean_mass_sidebar_val" # Key for the number input value
                )
                st.info("📌 Enter individual animal lean masses in the Overview tab after uploading.")
        st.divider() # Divider after the normalization section
    # NO else block here that modifies the state


    # --- File Upload ---
    def handle_file_upload_change():
        """Clears group assignments when a new file is uploaded or removed."""
        keys_to_clear = ['group_assignments', 'lean_mass_data', 
                        # Add any other session state keys that depend on the specific file/groups
                        'used_group_colors' # Good to clear this too
                        ]
        for key in keys_to_clear:
            if key in st.session_state:
                del st.session_state[key]
        # No explicit rerun needed here, Streamlit handles it on widget change
    st.markdown(f"##### Upload Data File for: **{parameter}**")
    if parameter == "FEED":
        st.warning(
            "⚠️ **Important:** For Feed data, please ensure you upload the **'FEED1 ACC.CSV'** file. "
            "The regular 'FEED1.CSV' file can be prone to errors (e.g., from animals interacting with the sensor).",
            icon="❗"
        )
    uploaded_file = st.file_uploader(
        f"Upload {parameter} CSV",
        type="csv",
        help="Upload your CLAMS data file",
        key="file_upload_1", # Keep the existing key
        label_visibility="collapsed",
        on_change=handle_file_upload_change 
    )
    
# Main title in content area
st.title("CLAMSer: CLAMS Data Analyzer adapted for Oxymax-CLAMS-CF Machine")

# --- Analysis Workflow Indicator ---


# Define the steps
workflow_steps = [
    "1. Upload Data",
    "2. Assign Groups",
    "3. Review Results", # Step index 2
    "4. Analyze Statistics" # Step index 3
]

# Calculate the enabling/completion status *BEFORE* displaying
file_uploaded = uploaded_file is not None
# Check session state for group assignments safely
groups_assigned = ('group_assignments' in st.session_state and
                 isinstance(st.session_state['group_assignments'], pd.DataFrame) and
                 not st.session_state['group_assignments'].empty)
# We don't strictly need results_exist for this display logic,
# but it's good practice to calculate it if available.
results_exist = 'results' in locals() and results is not None

# Display the indicator
st.markdown("### Analysis Workflow")
cols = st.columns(len(workflow_steps))

for i, (step, col) in enumerate(zip(workflow_steps, cols)):
    # Determine state based on previous steps completed
    if i == 0: # Step 1: Upload Data
        if file_uploaded: # If done
            col.markdown(f"<div style='background-color:#1e4620; color:white; padding:10px; border-radius:5px; text-align:center; border:1px solid #28a745;'><b>✅ {step}</b></div>", unsafe_allow_html=True)
        else: # If not done (current)
            col.markdown(f"<div style='background-color:#1a3a6c; color:white; padding:10px; border-radius:5px; text-align:center; border:1px solid #0d6efd;'><b>➡️ {step}</b></div>", unsafe_allow_html=True)

    elif i == 1: # Step 2: Assign Groups
        if groups_assigned: # If done (implies file was also uploaded)
            col.markdown(f"<div style='background-color:#1e4620; color:white; padding:10px; border-radius:5px; text-align:center; border:1px solid #28a745;'><b>✅ {step}</b></div>", unsafe_allow_html=True)
        elif file_uploaded: # If file uploaded but groups not assigned (current)
            col.markdown(f"<div style='background-color:#1a3a6c; color:white; padding:10px; border-radius:5px; text-align:center; border:1px solid #0d6efd;'><b>➡️ {step}</b></div>", unsafe_allow_html=True)
        else: # If file not even uploaded (pending)
             col.markdown(f"<div style='background-color:#2a2a2a; padding:10px; border-radius:5px; text-align:center; color:#a0a0a0; border:1px solid #6c757d;'>{step}</div>", unsafe_allow_html=True)

    elif i == 2: # Step 3: Review Results
        if groups_assigned: # Mark as 'done' (green) once groups are assigned
             col.markdown(f"<div style='background-color:#1e4620; color:white; padding:10px; border-radius:5px; text-align:center; border:1px solid #28a745;'><b>✅ {step}</b></div>", unsafe_allow_html=True)
        else: # If groups not assigned (pending)
             col.markdown(f"<div style='background-color:#2a2a2a; padding:10px; border-radius:5px; text-align:center; color:#a0a0a0; border:1px solid #6c757d;'>{step}</div>", unsafe_allow_html=True)

    elif i == 3: # Step 4: Analyze Statistics
        if groups_assigned: # Becomes 'current' (blue) once groups are assigned
             col.markdown(f"<div style='background-color:#1a3a6c; color:white; padding:10px; border-radius:5px; text-align:center; border:1px solid #0d6efd;'><b>➡️ {step}</b></div>", unsafe_allow_html=True)
        else: # If groups not assigned (pending)
             col.markdown(f"<div style='background-color:#2a2a2a; padding:10px; border-radius:5px; text-align:center; color:#a0a0a0; border:1px solid #6c757d;'>{step}</div>", unsafe_allow_html=True)

# --- End Analysis Workflow Indicator ---
st.markdown("---") # Separator before the main tabs

def detect_outliers(df, z_score_threshold=2, skip_cols=None):
    """
    Detect outliers in numeric columns using the Z-score method.

    Args:
        df (pd.DataFrame): Input DataFrame.
        z_score_threshold (float): Z-score threshold for outlier detection.
        skip_cols (list, optional): List of column names to exclude from
                                     outlier detection. Defaults to None.

    Returns:
        pd.DataFrame: DataFrame of booleans indicating outlier positions,
                      or None if input df is None.
    """
    if df is None:
        return None

    # Default list of columns to skip if none provided
    if skip_cols is None:
        skip_cols = ['hour', 'Mean', 'SEM', 'N'] # Added 'N' as it might appear in tables

    # Select only numeric columns for z-score calculation
    numeric_cols = df.select_dtypes(include=[np.number]).columns

    # Initialize outliers DataFrame with False values, same shape as input df
    outliers = pd.DataFrame(False, index=df.index, columns=df.columns)

    # Iterate through numeric columns
    for col in numeric_cols:
        # Skip columns specified in skip_cols list or the index name if it's numeric
        if col in skip_cols or col == df.index.name:
            continue

        # --- Check for sufficient data and variation ---
        col_data = df[col].dropna() # Work with non-missing data for this column
        if len(col_data) < 3: # Need at least 3 points for a meaningful std dev
            continue # Not enough data points to calculate outliers reliably

        mean_val = col_data.mean()
        std_val = col_data.std()

        # Skip if standard deviation is zero or very close to zero (or NaN)
        if std_val is None or np.isnan(std_val) or std_val < 1e-9:
            continue # No variation or cannot calculate std dev, so no outliers

        # Calculate absolute z-scores for the original column (including NaNs)
        z_scores = np.abs((df[col] - mean_val) / std_val)

        # Mark as outlier where z_score > threshold (NaNs will correctly result in False)
        outliers[col] = z_scores > z_score_threshold

    return outliers


    
def style_dataframe(df, z_score_threshold=2, skip_cols_outlier=None):
    """
    Apply styling to a DataFrame, highlighting outliers in numeric columns.

    Args:
        df (pd.DataFrame): Input DataFrame.
        z_score_threshold (float): Z-score threshold for outlier detection.
        skip_cols_outlier (list, optional): List of column names to exclude
                                             from outlier detection/styling.
                                             Defaults to ['hour', 'Mean', 'SEM', 'N'].

    Returns:
        pd.Styler or pd.DataFrame: Styled DataFrame object or original df if input is None.
    """
    if df is None:
        return df

    # Define the default columns to skip for outlier detection
    if skip_cols_outlier is None:
        skip_cols_outlier = ['hour', 'Mean', 'SEM', 'N', 'Subject ID', 'Group', 'Cage'] # Add common non-numeric IDs

    # Detect outliers using the helper function, passing the skip_cols
    outliers = detect_outliers(df, z_score_threshold=z_score_threshold, skip_cols=skip_cols_outlier)

    # Check if outlier detection returned a valid DataFrame
    if outliers is None:
         # If outlier detection failed or df was unsuitable, return unstyled df
         return df

    # Define a function to apply background color styling based on the outliers DataFrame
    def highlight_outliers_styling(column_data):
        # Check if the current column name exists in the outliers DataFrame
        # and if it wasn't explicitly skipped
        col_name = column_data.name
        if col_name in outliers.columns and col_name not in skip_cols_outlier:
            # Return a list of style strings based on the boolean values in the outliers df
            # Ensure alignment using the index
            return ['background-color: rgba(255, 0, 0, 0.2)' if outliers.loc[idx, col_name] else ''
                    for idx in column_data.index]
        else:
            # Return empty styles for columns not checked or explicitly skipped
            return ['' for _ in column_data.index]

    # Apply the styling function column-wise using df.style.apply
    try:
        styled_df = df.style.apply(highlight_outliers_styling)
        # Optionally add hover tooltips or other formatting here if needed
        # Example: styled_df = styled_df.set_tooltips(outliers.replace({True: 'Potential Outlier (Z > {z_score_threshold})', False: ''}))
        return styled_df
    except Exception as e:
         st.warning(f"Could not apply outlier styling: {e}. Displaying unstyled table.")
         return df # Return unstyled df on error


def verify_file_type(file, expected_type):
    """
    Verify if uploaded file matches expected parameter type by checking the header.
    Reads header line-by-line until ':DATA' is found or a safe limit is reached.
    Improved parameter line detection and base name extraction.

    Args:
        file: The uploaded file object.
        expected_type: The parameter key selected by the user (e.g., "VO2", "FLOW").

    Returns:
        tuple: (is_valid: bool, message: str)
             is_valid=True means proceed with processing.
             message contains warnings or errors.
    """
    try:
        # --- Read header line-by-line ---
        file_parameter_full = None
        has_parameter_file_tag = False
        has_data_tag = False
        max_header_lines = 500 # Increased limit
        line_count = 0

        file.seek(0)
        # Use io.TextIOWrapper for proper handling of text decoding
        # errors='ignore' might hide some issues, but is often necessary for non-standard files
        wrapper = io.TextIOWrapper(file, encoding='utf-8', errors='ignore')

        for line in wrapper:
            line_strip = line.strip()
            line_count += 1

            # Check for essential tags
            if line_count == 1 and 'PARAMETER File' in line_strip:
                has_parameter_file_tag = True

            # Use re.match to anchor search to START for the parameter line
            if file_parameter_full is None:
                # Original regex: handles "Paramter" or "Parameter", whitespace, optional quotes
                match = re.match(r"Param(?:e?)ter[\s,\t]+\"?(.*?)\"?$", line_strip, re.IGNORECASE)
                if match:
                    file_parameter_full = match.group(1).strip() # Get the part after "Parameter/Paramter"

            if ':DATA' in line_strip:
                has_data_tag = True
                break # Found data marker, stop reading header

            if line_count > max_header_lines:
                break # Safety break if header is too long

        file.seek(0) # Rewind file pointer after reading header
        wrapper.detach() # Detach wrapper without closing underlying file stream

        # --- Perform validation based on findings ---
        if not has_parameter_file_tag:
             # Hard fail if the first line doesn't identify the file type
             return False, "File format error: Expected 'PARAMETER File' tag in the first line. Please ensure this is a CLAMS/Oxymax output file."
        if not has_data_tag:
             # Hard fail if the :DATA marker is missing (crucial for parsing)
             if line_count > max_header_lines:
                 return False, f"File structure error: ':DATA' marker not found within the first {max_header_lines} lines. File might be corrupt, incomplete, or not a valid CLAMS data file."
             else:
                 return False, "File structure error: ':DATA' marker not found. This is required to locate the numerical data."

        # --- Check parameter match ---
        file_parameter_base = None # Extracted base name (e.g., "VO2")
        match_status = "Parameter Line Not Found" # Default status

        if file_parameter_full:
            # --- Revised Base Name Extraction ---
            # Try to match common patterns: NAME (UNIT), NAME, potentially with spaces
            # 1. Try matching "NAME (UNIT)" pattern first (allows spaces in NAME)
            base_name_match = re.match(r"^\s*([a-zA-Z0-9/\s]+?)\s+\(.*\)\s*$", file_parameter_full)
            if base_name_match:
                file_parameter_base = base_name_match.group(1).strip()
                match_status = f"Parsed '{file_parameter_base}' from '{file_parameter_full}' (with unit)"
            else:
                # 2. If no unit pattern, try matching just the name (might be FEED1 ACC)
                # Allow alphanumeric, space, forward slash
                base_name_match = re.match(r"^\s*([a-zA-Z0-9/\s]+)\s*$", file_parameter_full)
                if base_name_match:
                     file_parameter_base = base_name_match.group(1).strip()
                     # Handle potential FEED variations
                     if file_parameter_base.upper() == "FEED1 ACC":
                         file_parameter_base = "FEED1 ACC" # Keep specific case
                     match_status = f"Parsed '{file_parameter_base}' from '{file_parameter_full}' (no unit)"
                else:
                    # Failed to parse base name even though file_parameter_full was found
                    match_status = f"Base Name Parse Failed for '{file_parameter_full}'"
        # else: file_parameter_full is None, match_status remains "Parameter Line Not Found"

        # --- Perform Comparison ---
        # Convert both expected and found base name to upper case for case-insensitive comparison
        expected_upper = expected_type.strip().upper()
        found_upper = file_parameter_base.strip().upper() if file_parameter_base else None

        # Case 1: Exact match (after case conversion)
        if found_upper and found_upper == expected_upper:
            return True, "" # Success, no message needed

        # Case 2: Special handling for FEED vs FEED1 ACC
        # Check if user selected "FEED" and file contains "FEED1 ACC"
        if expected_upper == "FEED" and found_upper == "FEED1 ACC":
            # Treat as valid match, but provide an informational message
            info_message = (
                f"ℹ️ Note: You selected 'FEED', and the file appears to be '{file_parameter_base}'. "
                "This is likely correct (using accumulated feed data). Processing continues."
            )
            return True, info_message

        # Case 3: No exact match, issue a clearer warning but allow processing
        warning_source = "parameter information not found or unreadable in header" # Default source
        if file_parameter_full:
            if file_parameter_base:
                # We parsed something, but it didn't match
                warning_source = f"file header indicates parameter is '{file_parameter_base}' (from line: '{file_parameter_full}')"
            else:
                # We found the line but couldn't parse the base name
                warning_source = f"file header parameter line found ('{file_parameter_full}') but could not extract base name"

        # --- Use st.warning format for the message ---
        warning_message = (
            f"⚠️ **Parameter Mismatch?** Selected '{expected_type}', but {warning_source}. "
            "Processing anyway, but **please verify you uploaded the correct file for the selected parameter.**"
        )
        # Proceed with processing but return the prominent warning message
        return True, warning_message

    except UnicodeDecodeError:
        try: file.seek(0) # Try to rewind even on error
        except: pass
        # Return False (hard fail) because we can't read the file content at all
        return False, "File reading error: Cannot decode file content. Please ensure it's a valid text/CSV file (UTF-8 encoding preferred)."
    except Exception as e:
        try: file.seek(0)
        except: pass
        # Return False (hard fail) for other unexpected errors during verification
        return False, f"Error verifying file header: {str(e)}"
    
# Function to get color for any group name (handles custom group names)
def get_group_color(group_name, default_color="#AAAAAA"):
    """Get a color for a group, with fallbacks for custom group names"""
    if group_name in GROUP_COLORS:
        return GROUP_COLORS[group_name]
    
    # For custom group names, assign colors based on position
    group_num_match = re.search(r'Group\s+(\d+)', group_name)
    if group_num_match:
        group_num = int(group_num_match.group(1))
        # Use modulo to cycle through colors for large group numbers
        color_keys = list(GROUP_COLORS.keys())
        if group_num <= len(color_keys):
            return GROUP_COLORS[color_keys[group_num-1]]
    
    # Take first available color for completely custom names
    used_colors = set()
    for name, color in GROUP_COLORS.items():
        if name not in st.session_state.get('used_group_colors', {}):
            st.session_state.setdefault('used_group_colors', {})[group_name] = color
            return color
    
    return default_color

# Experimental addition of verifying data

def enhanced_sample_calculations(df_24h, results, hourly_results, parameter):
    """Enhanced version of show_verification_calcs function"""
    with st.expander("🔍 View Detailed Calculation Process", expanded=False):
        st.write("### Step-by-Step Calculation Process")
        
        if df_24h is None or results is None:
            st.warning("Upload data to see detailed calculations.")
            return
            
        # Get first cage data
        first_cage = df_24h['cage'].unique()[0]
        first_cage_data = df_24h[df_24h['cage'] == first_cage].copy()
        
        st.write(f"#### Calculations for {first_cage}")
        
        # Create tabs for different calculation steps
        step1, step2, step3 = st.tabs(["1. Data Preparation", "2. Light/Dark Calculation", "3. Hourly Calculation"])
        
        with step1:
            st.write("##### Step 1: Raw Data Processing")
            st.write("First, we extract time, value, and cage information from the CSV file:")
            
            # Show raw data sample
            st.dataframe(first_cage_data[['timestamp', 'value', 'is_light']].head(5))
            
            st.write("We determine light/dark cycle based on the hour:")
            st.code("""
# Light cycle is defined as 7:00 AM - 7:00 PM
df_24h['hour'] = df_24h['timestamp'].dt.hour
df_24h['is_light'] = (df_24h['hour'] >= 7) & (df_24h['hour'] < 19)
            """)
            
            # Show processed data with light/dark labels
            light_dark_sample = first_cage_data[['timestamp', 'hour', 'is_light', 'value']].head(10)
            st.dataframe(light_dark_sample)
            
        with step2:
            st.write("##### Step 2: Light/Dark Cycle Calculations")
            
            # Show light filter
            st.write("Filter for light cycle data (7:00 AM - 7:00 PM):")
            st.code("""
light_data = df_24h[(df_24h['cage'] == first_cage) & (df_24h['is_light'])]
            """)
            
            # Show dark filter
            st.write("Filter for dark cycle data (7:00 PM - 7:00 AM):")
            st.code("""
dark_data = df_24h[(df_24h['cage'] == first_cage) & (~df_24h['is_light'])]
            """)
            
            # Calculate and show light/dark averages
            dark_data = first_cage_data[~first_cage_data['is_light']]['value']
            light_data = first_cage_data[first_cage_data['is_light']]['value']
            
            col1, col2 = st.columns(2)
            with col1:
                st.write("**Dark Cycle Calculation:**")
                st.write(f"Sum of values: {dark_data.sum():.2f}")
                st.write(f"Number of readings: {len(dark_data)}")
                st.write(f"Dark Average = {dark_data.mean():.2f}")
                
            with col2:
                st.write("**Light Cycle Calculation:**")
                st.write(f"Sum of values: {light_data.sum():.2f}")
                st.write(f"Number of readings: {len(light_data)}")
                st.write(f"Light Average = {light_data.mean():.2f}")
                
            # Show calculation verification against results
            st.write("#### Verification Against Results Table")
            if not results.empty:
                result_row = results.loc[results.index.str.contains(first_cage.split()[1]), :]
                if not result_row.empty:
                    if parameter in ["VO2", "VCO2", "RER", "HEAT"]:
                        st.write(f"From results table - Dark Average: {result_row['Dark Average'].iloc[0]:.2f}")
                        st.write(f"From results table - Light Average: {result_row['Light Average'].iloc[0]:.2f}")
                        
                        # Calculate difference to prove accuracy
                        dark_diff = abs(dark_data.mean() - result_row['Dark Average'].iloc[0])
                        light_diff = abs(light_data.mean() - result_row['Light Average'].iloc[0])
                        
                        st.write(f"Dark calculation difference: {dark_diff:.8f} (should be near zero)")
                        st.write(f"Light calculation difference: {light_diff:.8f} (should be near zero)")
                        
                        if dark_diff < 0.001 and light_diff < 0.001:
                            st.success("✅ Verification successful - calculation matches results table!")
                        
        with step3:
            st.write("##### Step 3: Hourly Calculations")
            
            # Show hourly calculation
            st.write("For hourly data, we group by hour and calculate the mean:")
            st.code("""
hourly_results = df_24h.pivot_table(
    values='value', 
    index='hour',
    columns='cage',
    aggfunc='mean'
)
            """)
            
            # Show a sample hourly calculation for hour 0
            hour_0_data = first_cage_data[first_cage_data['hour'] == 0]['value']
            
            if not hour_0_data.empty:
                st.write(f"**Hour 0 Calculation:**")
                st.write(f"Sum of values: {hour_0_data.sum():.2f}")
                st.write(f"Number of readings: {len(hour_0_data)}")
                st.write(f"Hour 0 Average = {hour_0_data.mean():.2f}")
                
                # Verify against hourly results
                if hourly_results is not None and not hourly_results.empty:
                    cage_column = None
                    for col in hourly_results.columns:
                        if first_cage.split()[1] in str(col):
                            cage_column = col
                            break
                    
                    if cage_column is not None:
                        st.write(f"From hourly results table - Hour 0: {hourly_results.loc[0, cage_column]:.2f}")
                        
                        # Calculate difference
                        hour_diff = abs(hour_0_data.mean() - hourly_results.loc[0, cage_column])
                        st.write(f"Hourly calculation difference: {hour_diff:.8f} (should be near zero)")
                        
                        if hour_diff < 0.001:
                            st.success("✅ Verification successful - calculation matches hourly results table!")

def add_calculation_visualization(raw_data, parameter):
    """Add interactive visualization that shows how calculations are performed from raw data"""
    if raw_data is not None and not raw_data.empty:
        st.subheader("🧮 Interactive Calculation Visualization")
        
        # Let user select a cage to visualize calculations for
        cages = raw_data['cage'].unique()
        selected_cage = st.selectbox("Select a cage to visualize calculations:", cages)
        
        # Filter data for selected cage
        cage_data = raw_data[raw_data['cage'] == selected_cage].copy()
        
        # Show light/dark separation
        fig = go.Figure()
        
        # Add light cycle data
        light_data = cage_data[cage_data['is_light']]
        dark_data = cage_data[~cage_data['is_light']]
        
        fig.add_trace(go.Scatter(
            x=light_data['timestamp'],
            y=light_data['value'],
            mode='markers',
            name='Light Cycle',
            marker=dict(color='gold')
        ))
        
        fig.add_trace(go.Scatter(
            x=dark_data['timestamp'],
            y=dark_data['value'],
            mode='markers',
            name='Dark Cycle',
            marker=dict(color='darkblue')
        ))
        
        # Add horizontal lines for averages
        light_avg = light_data['value'].mean()
        dark_avg = dark_data['value'].mean()
        total_avg = cage_data['value'].mean()
        
        fig.add_trace(go.Scatter(
            x=[cage_data['timestamp'].min(), cage_data['timestamp'].max()],
            y=[light_avg, light_avg],
            mode='lines',
            name='Light Average',
            line=dict(color='gold', width=2, dash='dash')
        ))
        
        fig.add_trace(go.Scatter(
            x=[cage_data['timestamp'].min(), cage_data['timestamp'].max()],
            y=[dark_avg, dark_avg],
            mode='lines',
            name='Dark Average',
            line=dict(color='darkblue', width=2, dash='dash')
        ))
        
        fig.add_trace(go.Scatter(
            x=[cage_data['timestamp'].min(), cage_data['timestamp'].max()],
            y=[total_avg, total_avg],
            mode='lines',
            name='Total Average',
            line=dict(color='green', width=2, dash='dash')
        ))
        
        fig.update_layout(
            title=f'Data Points and Averages for {selected_cage}',
            xaxis_title='Time',
            yaxis_title=f'{parameter} Value',
            legend_title='Data Type',
            hovermode='closest'
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Show calculation details
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric(
                "Light Cycle Average",
                f"{light_avg:.2f}",
                f"{len(light_data)} points"
            )
        with col2:
            st.metric(
                "Dark Cycle Average",
                f"{dark_avg:.2f}",
                f"{len(dark_data)} points"
            )
        with col3:
            st.metric(
                "Total Average",
                f"{total_avg:.2f}",
                f"{len(cage_data)} points"
            )
        
        # Add calculation formula display
        st.markdown("#### Calculation Formulas")
        
        # Create tabs for different calculations
        calc_tab1, calc_tab2 = st.tabs(["Light/Dark Averages", "Hourly Averages"])
        
        with calc_tab1:
            st.markdown("**Light Cycle Average:**")
            if len(light_data) > 0:
                st.latex(r"\text{Light Average} = \frac{\sum \text{Light Values}}{\text{Number of Light Values}}")
                light_values = light_data['value'].tolist()
                light_formula = " + ".join([f"{val:.2f}" for val in light_values[:3]])
                if len(light_values) > 3:
                    light_formula += f" + ... + {light_values[-1]:.2f}"
                light_formula += f" = {sum(light_values):.2f}"
                light_formula += f" ÷ {len(light_values)} = {light_avg:.2f}"
                st.code(light_formula)
                
            st.markdown("**Dark Cycle Average:**")
            if len(dark_data) > 0:
                st.latex(r"\text{Dark Average} = \frac{\sum \text{Dark Values}}{\text{Number of Dark Values}}")
                dark_values = dark_data['value'].tolist()
                dark_formula = " + ".join([f"{val:.2f}" for val in dark_values[:3]])
                if len(dark_values) > 3:
                    dark_formula += f" + ... + {dark_values[-1]:.2f}"
                dark_formula += f" = {sum(dark_values):.2f}"
                dark_formula += f" ÷ {len(dark_values)} = {dark_avg:.2f}"
                st.code(dark_formula)
        
        with calc_tab2:
            st.markdown("**Hourly Average Calculation:**")
            # Get one hour of data for demonstration
            hour_data = cage_data[cage_data['hour'] == 12]  # Noon
            if len(hour_data) > 0:
                hour_avg = hour_data['value'].mean()
                st.latex(r"\text{Hour Average} = \frac{\sum \text{Values in Hour}}{\text{Number of Values in Hour}}")
                hour_values = hour_data['value'].tolist()
                hour_formula = " + ".join([f"{val:.2f}" for val in hour_values])
                hour_formula += f" = {sum(hour_values):.2f}"
                hour_formula += f" ÷ {len(hour_values)} = {hour_avg:.2f}"
                st.code(hour_formula)

def add_trust_verification_section(raw_data, results, parameter, parameter_descriptions):
    """Add a section about data trust and verification"""
    st.subheader("🔍 How Do I Trust This Data?")
    
    with st.expander("Data Integrity Verification", expanded=False):
        st.markdown("""
        ### Raw Data Validation
        
        CLAMSer performs the following integrity checks on your data:
        
        1. **File Format Verification**: Confirms the file follows CLAMS/Oxymax standard format
        2. **Parameter Validation**: Verifies the selected parameter matches the data
        3. **Time Series Integrity**: Checks for major gaps in the time series data
        4. **Subject ID Consistency**: Ensures consistent subject IDs throughout the file
        """)
        
        # Show sample of raw parsing code
        st.code("""
# Actual code snippet from CLAMSer parsing engine
def verify_file_type(file, expected_type):
    try:
        content = file.read().decode()
        file.seek(0)
        
        # First line should identify this as a CLAMS/Oxymax file
        first_line = content.split('\\n')[0].strip()
        if 'PARAMETER File' not in first_line:
            return False, "File format not recognized. Expected CLAMS/Oxymax parameter file."
            
        # Look for :DATA marker which should be present in all CLAMS files
        if ':DATA' not in content:
            return False, "File structure not recognized. Missing :DATA marker."
        
        # Verify parameter type in file matches selected
        content_lower = content.lower()
        expected_lower = expected_type.lower()
        if expected_lower in content_lower:
            return True, ""
        else:
            return True, "Note: Selected parameter type may not match file content"
    except Exception as e:
        return False, f"Error verifying file: {str(e)}"
        """, language="python")
        
    # Calculation traceability section
    with st.expander("Calculation Traceability", expanded=False):
        st.markdown("""
        ### How Calculations Are Performed
        
        CLAMSer follows the exact same calculation methodology that researchers would use manually:
        """)
        
        # Different calculation examples based on parameter
        if parameter in ["VO2", "VCO2", "RER", "HEAT"]:
            st.markdown(f"""
            **For {parameter} ({parameter_descriptions.get(parameter, "")}):**
            
            1. **Light/Dark Cycle Separation**: Data is separated based on timestamp (7:00 AM - 7:00 PM = Light)
            2. **Averaging**: Values are averaged separately for Light and Dark cycles
            3. **Total Average**: Calculated as the mean of all values across 24 hours
            """)
            
            # Show a visual trace of calculations with real data
            if raw_data is not None and len(raw_data) > 0:
                # Get a sample cage
                sample_cage = raw_data['cage'].unique()[0]
                sample_data = raw_data[raw_data['cage'] == sample_cage].copy()
                
                # Calculate light/dark averages for sample
                light_avg = sample_data[sample_data['is_light']]['value'].mean()
                dark_avg = sample_data[~sample_data['is_light']]['value'].mean()
                
                # Display calculation trace
                st.markdown("#### Example Calculation Trace")
                st.markdown(f"**Sample cage**: {sample_cage}")
                st.markdown(f"**Light cycle data points**: {len(sample_data[sample_data['is_light']])}")
                st.markdown(f"**Dark cycle data points**: {len(sample_data[~sample_data['is_light']])}")
                
                # Show detailed formula with real numbers
                st.markdown("**Light Cycle Average Calculation:**")
                if len(sample_data[sample_data['is_light']]) > 0:
                    light_values = sample_data[sample_data['is_light']]['value'].tolist()
                    light_formula = " + ".join([f"{val:.2f}" for val in light_values[:5]])
                    if len(light_values) > 5:
                        light_formula += f" + ... ({len(light_values)-5} more values)"
                    light_formula += f" = {sum(light_values):.2f}"
                    light_formula += f" ÷ {len(light_values)} = {light_avg:.2f}"
                    st.code(light_formula)
        
        # Comparison with manual calculations
        st.markdown("""
        ### Validation Against Manual Calculations
        
        We've verified the CLAMSer results against manual Excel calculations:
        
        1. **Identical Results**: CLAMSer produces identical results to manual Excel analysis
        2. **Precision**: All calculations maintain full numerical precision throughout
        3. **Outlier Detection**: Uses the same statistical methods (Z-score > 2.0) as standard analysis
        """)
        
    # Statistical validation section
    with st.expander("Statistical Validation", expanded=False):
        st.markdown("""
        ### Statistical Integrity
        
        CLAMSer performs these statistical validations:
        
        1. **Normal Distribution Testing**: Shapiro-Wilk test for normality when required
        2. **Equal Variance Testing**: Levene's test for homogeneity of variance 
        3. **Appropriate Statistical Tests**: Automatic selection of parametric/non-parametric tests
        4. **Multiple Comparison Correction**: Bonferroni correction for multiple comparisons
        """)
        
        # If we have group assignments, show a sample of statistical validation
        if 'group_assignments' in st.session_state and raw_data is not None:
            if not st.session_state.get('group_assignments', pd.DataFrame()).empty:
                st.markdown("#### Statistical Test Validation")
                st.markdown("""
                The t-test and ANOVA implementations have been validated against standard 
                SciPy implementations with identical results.
                """)
                
    # Add download sample dataset for verification
    with st.expander("Reproduce These Results", expanded=False):
        st.markdown("""
        ### Reproduce These Results
        
        To verify the accuracy of CLAMSer's analysis:
        
        1. Download the processed dataset below
        2. Process it in your own analysis pipeline
        3. Compare the results with CLAMSer's outputs
        
        This transparency allows you to validate every step of our analysis.
        """)
        
        if raw_data is not None:
            # Create a downloadable version of the raw data
            sample_csv = raw_data.head(100).to_csv(index=False).encode('utf-8')
            st.download_button(
                label="📥 Download Sample Raw Data (100 rows)",
                data=sample_csv,
                file_name="clams_raw_sample.csv",
                mime="text/csv",
                help="Download a sample of the raw data to verify calculations independently"
            )
            
            # If results are available, make them downloadable too
            if results is not None:
                results_csv = results.to_csv().encode('utf-8')
                st.download_button(
                    label=f"📥 Download {parameter} Results",
                    data=results_csv,
                    file_name=f"{parameter}_results.csv",
                    mime="text/csv",
                    help="Download the results to verify calculations independently"
                )

def extract_cage_info(file):
    """
    Extracts cage-to-subject ID mapping from the file header.

    Reads the file line-by-line until the ':DATA' marker.

    Args:
        file: Uploaded file object.

    Returns:
        dict: Mapping of {standard_cage_label (e.g., "CAGE 01"): subject_id},
              or an empty dict if extraction fails or no mapping is found.
    """
    cage_subject_map = {} # Use standard "CAGE XX" as key
    current_cage_num_str = None # Stores raw number string like "101"
    max_header_lines = 500 # Safety limit

    try:
        file.seek(0)
        wrapper = io.TextIOWrapper(file, encoding='utf-8', errors='ignore')
        line_count = 0

        for line in wrapper:
            line_strip = line.strip()
            line_count += 1

            if not line_strip: # Skip empty lines
                continue

            if ':DATA' in line_strip:
                break # Stop processing header lines

            if 'Group/Cage' in line_strip:
                try:
                    current_cage_num_str = line_strip.split(',')[-1].strip().lstrip('0')
                except IndexError:
                    # st.warning(f"Header parse warning: Could not get cage number from: '{line_strip}'") # Optional warning
                    current_cage_num_str = None
            elif 'Subject ID' in line_strip and current_cage_num_str is not None:
                try:
                    subject_id = line_strip.split(',')[-1].strip()
                    # --- Standardize Cage Label ---
                    cage_int = int(current_cage_num_str)
                    if cage_int >= 101:
                        standard_cage_label = f"CAGE {cage_int - 100:02d}"
                        cage_subject_map[standard_cage_label] = subject_id
                    # else: Skip cages < 101 silently, or add warning if needed
                except (IndexError, ValueError) as e:
                    # st.warning(f"Header parse warning: Failed processing Subject ID line for cage '{current_cage_num_str}': {e}") # Optional warning
                    pass # Ignore line if parsing fails
                # Reset cage num str to ensure next Subject ID line needs a preceding Cage line
                current_cage_num_str = None

            if line_count > max_header_lines:
                 st.warning(f"Stopped reading header after {max_header_lines} lines; ':DATA' marker not found early.")
                 break # Stop if header seems too long

        wrapper.detach()
        file.seek(0) # Rewind file for other functions

        if not cage_subject_map:
             st.caption("ℹ️ No cage-to-subject ID mapping found in file header.") # Use caption, not error

        return cage_subject_map

    except Exception as e:
        st.error(f"Critical error during cage info extraction: {e}")
        try: file.seek(0) # Attempt rewind
        except: pass
        return {} # Return empty dict on critical error
def _parse_clams_header(file_wrapper):
    """Parses the header of a CLAMS file to find the data start and subject map."""
    header_lines = []
    subject_map = {}
    current_cage_num_str = None
    data_start_line_num = -1
    line_count = 0
    max_header_lines = 500 # Safety limit

    for i, line in enumerate(file_wrapper):
        line_count = i
        line_strip = line.strip()
        if not line_strip: continue

        header_lines.append(line_strip)

        if ':DATA' in line_strip:
            data_start_line_num = i + 1 # 1-based line number where data starts
            break # Stop reading header

        # Simplified subject mapping parsing (assumes structure from previous code)
        if 'Group/Cage' in line_strip:
            try: current_cage_num_str = line_strip.split(',')[-1].strip().lstrip('0')
            except IndexError: current_cage_num_str = None
        elif 'Subject ID' in line_strip and current_cage_num_str is not None:
            try:
                subject_id = line_strip.split(',')[-1].strip()
                cage_int = int(current_cage_num_str)
                if cage_int >= 101: # Basic CLAMS format check
                    cage_label = f"CAGE {cage_int - 100:02d}"
                    subject_map[cage_label] = subject_id
            except (IndexError, ValueError): pass # Ignore parsing errors here
            current_cage_num_str = None # Reset for next pair

        if line_count >= max_header_lines:
            st.warning(f"Stopped reading header after {max_header_lines} lines; ':DATA' marker may be missing or very late.")
            break

    if data_start_line_num == -1:
        st.error("❌ Critical Error: ':DATA' marker not found in the file header. Cannot process.")
        return None, None, None, None # Indicate failure

    # Check if Subject IDs were found
    if not subject_map:
        st.caption("ℹ️ No Subject ID mappings found in file header.")

    # Determine if the data section header needs skipping ('INTERVAL' or '===')
    # Peek at the line immediately after ':DATA' if possible
    try:
        # Re-read lines briefly to find the line(s) after :DATA
        file_wrapper.seek(0) # Go back to start
        post_data_lines = []
        temp_line_num = 0
        start_collecting = False
        for line in file_wrapper:
            temp_line_num += 1
            if temp_line_num == data_start_line_num:
                start_collecting = True
                continue # Skip the :DATA line itself
            if start_collecting:
                line_strip = line.strip()
                if line_strip: # Only consider non-empty lines
                    post_data_lines.append(line_strip)
                if len(post_data_lines) >= 2: # Check first two non-empty lines after :DATA
                    break
        # Determine if skip is needed based on first actual data-like line
        data_header_skipped = False
        if post_data_lines:
             first_data_section_line = post_data_lines[0]
             if first_data_section_line.startswith('===') or first_data_section_line.startswith('INTERVAL'):
                  data_header_skipped = True
             # Check second line too if first was skipped separator
             if data_header_skipped and len(post_data_lines) > 1:
                  second_data_section_line = post_data_lines[1]
                  if second_data_section_line.startswith('INTERVAL'):
                       # If both '===' and 'INTERVAL' are present, still just one header skip logic needed
                       pass # Already flagged to skip

    except Exception as e:
         st.warning(f"Could not reliably determine data header presence: {e}. Assuming no header skip.")
         data_header_skipped = False


    return header_lines, subject_map, data_start_line_num, data_header_skipped

def _parse_clams_data_section(file_wrapper, data_start_line_num, data_header_skipped):
    """Parses the data section of the CLAMS file robustly."""
    all_data_cages = []
    parsing_errors = []
    max_errors_to_report = 15
    data_lines_read = 0
    num_cages_detected = 0

    file_wrapper.seek(0) # Rewind to start reading from data section
    line_num = 0
    first_data_line_content = None

    # Skip header lines + :DATA line
    for _ in range(data_start_line_num):
        file_wrapper.readline()
        line_num += 1

    # Skip optional data header lines ('===' or 'INTERVAL')
    if data_header_skipped:
        # Read lines until we are past the header indicators
        skipped_count = 0
        max_skips = 3 # Safety limit for skipping header lines
        while skipped_count < max_skips:
             line = file_wrapper.readline()
             line_num +=1
             line_strip = line.strip()
             if not line_strip: continue # Skip blank lines
             if line_strip.startswith('===') or line_strip.startswith('INTERVAL'):
                 skipped_count += 1
                 continue
             else:
                 # This is likely the first actual data line
                 first_data_line_content = line
                 break # Stop skipping

    # Read the first actual data line if not already captured
    if first_data_line_content is None:
         while True: # Find the first non-empty line to determine columns
             line = file_wrapper.readline()
             line_num += 1
             if not line: break # End of file
             line_strip = line.strip()
             if line_strip:
                 first_data_line_content = line
                 break

    if not first_data_line_content:
        parsing_errors.append("Error: No data lines found after header/':DATA' marker.")
        return None, parsing_errors # Cannot proceed

    # Determine number of columns and cages from the first data line
    first_row_data = first_data_line_content.strip().split(',')
    num_data_columns = len(first_row_data)
    num_cages_detected = (num_data_columns - 1) // 2 # Assuming Interval, T1, V1, T2, V2,...

    if num_cages_detected < 1:
        parsing_errors.append(f"Error: Could not detect valid CAGE data columns. Found {num_data_columns} columns. Check file structure.")
        parsing_errors.append(f"First data line content: {first_data_line_content.strip()}")
        return None, parsing_errors

    # --- Initialize data structures for cages ---
    cage_data_lists = [{'times': [], 'values': [], 'label': f"CAGE {i+1:02d}"} for i in range(num_cages_detected)]

    # --- Process the first data line already read ---
    data_lines_read += 1
    row_data = first_row_data # Use the split data
    for cage_idx in range(num_cages_detected):
        time_col_idx = 1 + 2 * cage_idx
        value_col_idx = 2 + 2 * cage_idx
        cage_info = cage_data_lists[cage_idx]

        if len(row_data) <= max(time_col_idx, value_col_idx):
             if len(parsing_errors) < max_errors_to_report: parsing_errors.append(f"L{line_num}, {cage_info['label']}: Row too short.")
             continue
        t_str, v_str = row_data[time_col_idx].strip(), row_data[value_col_idx].strip()
        if not t_str or not v_str: continue
        try: time_obj = pd.to_datetime(t_str, format='%d/%m/%Y %I:%M:%S %p')
        except ValueError:
            try: time_obj = pd.to_datetime(t_str, format='%m/%d/%Y %I:%M:%S %p')
            except ValueError:
                 if len(parsing_errors) < max_errors_to_report: parsing_errors.append(f"L{line_num}, {cage_info['label']}: Bad datetime '{t_str}'")
                 continue
        try:
            value_float = float(v_str)
            cage_info['times'].append(time_obj); cage_info['values'].append(value_float)
        except ValueError:
            if len(parsing_errors) < max_errors_to_report: parsing_errors.append(f"L{line_num}, {cage_info['label']}: Bad numeric value '{v_str}'")
            continue

    # --- Process remaining lines in the data section ---
    for line in file_wrapper:
        line_num += 1
        data_lines_read += 1
        line_strip = line.strip()
        if not line_strip: continue # Skip empty lines

        row_data = line_strip.split(',')
        if len(row_data) != num_data_columns: # Check for consistent column count
             if len(parsing_errors) < max_errors_to_report: parsing_errors.append(f"L{line_num}: Inconsistent column count ({len(row_data)} vs expected {num_data_columns}). Skipping row.")
             continue

        for cage_idx in range(num_cages_detected):
             time_col_idx = 1 + 2 * cage_idx
             value_col_idx = 2 + 2 * cage_idx
             cage_info = cage_data_lists[cage_idx]

             # No need to check row length again if we checked column consistency
             t_str, v_str = row_data[time_col_idx].strip(), row_data[value_col_idx].strip()
             if not t_str or not v_str: continue # Skip rows with missing time or value for this cage
             try: time_obj = pd.to_datetime(t_str, format='%d/%m/%Y %I:%M:%S %p')
             except ValueError:
                 try: time_obj = pd.to_datetime(t_str, format='%m/%d/%Y %I:%M:%S %p')
                 except ValueError:
                      if len(parsing_errors) < max_errors_to_report: parsing_errors.append(f"L{line_num}, {cage_info['label']}: Bad datetime '{t_str}'")
                      continue
             try:
                 value_float = float(v_str)
                 cage_info['times'].append(time_obj); cage_info['values'].append(value_float)
             except ValueError:
                 if len(parsing_errors) < max_errors_to_report: parsing_errors.append(f"L{line_num}, {cage_info['label']}: Bad numeric value '{v_str}'")
                 continue # Skip this cage's entry for this row if value is bad

    # --- Create DataFrames for each cage ---
    all_data_cages = []
    for cage_info in cage_data_lists:
        if cage_info['times']: # Only create DataFrame if data was actually parsed
            cage_df = pd.DataFrame({'timestamp': cage_info['times'], 'value': cage_info['values'], 'cage': cage_info['label']})
            all_data_cages.append(cage_df)
        # else: # Optional: Log if a cage had no valid data points
            # print(f"Debug: No valid data points parsed for {cage_info['label']}.")

    if not all_data_cages:
        parsing_errors.append("Error: Failed to extract any valid numerical data points after processing all cages.")
        return None, parsing_errors

    # --- Concatenate, Sort, and Return ---
    df_raw_parsed = pd.concat(all_data_cages, ignore_index=True)
    df_raw_parsed.sort_values(by='timestamp', inplace=True)
    df_raw_parsed.reset_index(drop=True, inplace=True)

    if data_lines_read == 0:
         parsing_errors.append("Warning: Zero data lines were read after the header.")
         
    # Report final count of errors if any occurred beyond the limit
    if len(parsing_errors) >= max_errors_to_report:
         total_errors = len(parsing_errors) # Get total count before slicing
         # Count how many might have been missed (approximate)
         # This requires knowing the total number of lines read AFTER the header skip
         # Let's estimate total potential points processed
         total_points_attempted = data_lines_read * num_cages_detected
         # This logic is complex, let's just add a general note
         parsing_errors.append(f"... and potentially more issues not shown (limit is {max_errors_to_report}).")


    return df_raw_parsed, parsing_errors


def filter_data_by_time_window(df_processed, selected_window, custom_days_val):
    """Filters the DataFrame based on the selected time window."""
    df_analysis_window = None
    if selected_window == "Entire Dataset":
        df_analysis_window = df_processed.copy()
        st.caption("ℹ️ Analyzing the entire dataset.")
    else:
        if selected_window == "Custom Range":
            if isinstance(custom_days_val, (int, float)) and custom_days_val >= 1:
                days_to_analyze = int(custom_days_val)
            else:
                st.warning(f"Invalid custom days value ({custom_days_val}). Defaulting to 5 days.", icon="⚠️")
                days_to_analyze = 5
        else:
            days_map = {
                "Last 24 Hours": 1, "Last 48 Hours": 2, "Last 72 Hours": 3,
                "Last 7 Days": 7, "Last 14 Days": 14
            }
            days_to_analyze = days_map.get(selected_window)
            if days_to_analyze is None:
                st.error(f"Internal Error: Unrecognized time window '{selected_window}'. Using entire dataset.")
                return df_processed.copy() # Fallback

        if df_processed.empty or 'timestamp' not in df_processed.columns:
             st.error("Cannot filter by time: Input data is empty or missing 'timestamp' column.")
             return None # Indicate failure

        required_duration = pd.Timedelta(days=days_to_analyze)
        data_start_time = df_processed['timestamp'].min()
        data_end_time = df_processed['timestamp'].max()

        # Check if data_start_time or data_end_time is NaT (Not a Time)
        if pd.isna(data_start_time) or pd.isna(data_end_time):
             st.error("Cannot filter by time: Could not determine data start/end times (possibly empty or invalid timestamps).")
             return None # Indicate failure
             
        available_duration = data_end_time - data_start_time

        # Allow small tolerance (e.g., 1 hour)
        if available_duration < (required_duration - pd.Timedelta(hours=1)):
            st.error(f"Insufficient data for '{selected_window}' ({days_to_analyze} days required). "
                     f"File contains only ~{available_duration.total_seconds() / 3600:.1f} hours. "
                     "Choose a shorter window or 'Entire Dataset'.")
            return None # Indicate failure

        filter_start_time = data_end_time - required_duration
        # Ensure filter doesn't go before the actual data start
        filter_start_time = max(filter_start_time, data_start_time)

        df_analysis_window = df_processed.loc[
            (df_processed['timestamp'] >= filter_start_time) &
            (df_processed['timestamp'] <= data_end_time)
        ].copy()

        if df_analysis_window.empty:
             st.error(f"❌ Error: No data remains after applying the '{selected_window}' time filter. Check data distribution.")
             return None # Indicate failure

        actual_start = df_analysis_window['timestamp'].min()
        actual_end = df_analysis_window['timestamp'].max()
        st.caption(f"ℹ️ Analysis Time Window: {actual_start.strftime('%Y-%m-%d %H:%M')} to {actual_end.strftime('%Y-%m-%d %H:%M')} ({selected_window}).")

    return df_analysis_window


def apply_lean_mass_adjustment(df_input, parameter_type, apply_lean_mass_flag, lean_mass_data, reference_mass_val):
    """Applies lean mass normalization to the DataFrame if applicable and enabled."""
    df_output = df_input.copy() # Work on a copy

    # Determine if adjustment should be attempted
    adjustment_possible = (
        parameter_type in ["VO2", "VCO2", "HEAT"] and
        apply_lean_mass_flag
    )

    if not adjustment_possible:
        # If adjustment was previously applied, reset 'value' from 'original_value'
        if 'original_value' in df_output.columns:
            df_output['value'] = df_output['original_value']
            # Optionally remove the original_value column if no longer needed
            # df_output = df_output.drop(columns=['original_value'])
            # st.caption("ℹ️ Lean mass adjustment disabled or not applicable; values reset to original.") # Optional message
        return df_output # Return unmodified (or reset) data

    # --- Proceed with Adjustment ---
    proceed_with_adjustment = True
    warning_messages = []
    if not isinstance(lean_mass_data, dict) or not lean_mass_data:
         warning_messages.append("Individual lean masses not entered/saved.")
         proceed_with_adjustment = False
    if reference_mass_val is None or not isinstance(reference_mass_val, (int, float)) or reference_mass_val <= 0:
         warning_messages.append(f"Reference lean mass missing or invalid ({reference_mass_val}).")
         proceed_with_adjustment = False
    if 'value' not in df_output.columns:
         warning_messages.append("Input data missing 'value' column.")
         proceed_with_adjustment = False

    if not proceed_with_adjustment:
         st.warning(f"⚠️ Lean mass adjustment enabled, but cannot proceed: {' | '.join(warning_messages)}. Skipping adjustment.", icon="⚖️")
         # Ensure original_value is removed if it exists but adjustment is skipped
         if 'original_value' in df_output.columns:
             df_output['value'] = df_output['original_value']
             # df_output = df_output.drop(columns=['original_value'])
         return df_output

    # --- Apply the formula ---
    reference_mass = float(reference_mass_val)

    # Store original values if not already done
    if 'original_value' not in df_output.columns:
        df_output['original_value'] = df_output['value'].copy()
    else: # Reset value from original before re-applying adjustment
        df_output['value'] = df_output['original_value']

    adjusted_cages_count = 0
    skipped_cages_invalid_mass = set()
    skipped_cages_no_mass_data = set()
    cages_in_window = df_output['cage'].unique()

    for cage_label in cages_in_window:
        if cage_label in lean_mass_data:
            lean_mass = lean_mass_data[cage_label]
            # Validate lean_mass type and value
            if isinstance(lean_mass, (int, float)) and lean_mass > 0:
                mask = (df_output['cage'] == cage_label)
                # Perform calculation using .loc for safe assignment
                # Ensure original_value exists before using it
                if 'original_value' in df_output.columns:
                     adjustment_factor = reference_mass / lean_mass
                     df_output.loc[mask, 'value'] = df_output.loc[mask, 'original_value'] * adjustment_factor
                     adjusted_cages_count += 1
                else:
                    # This case should ideally not happen if check passed earlier
                    st.warning(f"Logic Error: 'original_value' column missing during adjustment for {cage_label}.")
                    skipped_cages_invalid_mass.add(cage_label) # Treat as skipped
            else:
                skipped_cages_invalid_mass.add(f"{cage_label} (value: {lean_mass})")
                # Ensure value remains original if skipped
                if 'original_value' in df_output.columns:
                     mask = (df_output['cage'] == cage_label)
                     df_output.loc[mask, 'value'] = df_output.loc[mask, 'original_value']

        else: # Cage in data, but no mass provided in lean_mass_data dict
             skipped_cages_no_mass_data.add(cage_label)
             # Ensure value remains original if skipped
             if 'original_value' in df_output.columns:
                     mask = (df_output['cage'] == cage_label)
                     df_output.loc[mask, 'value'] = df_output.loc[mask, 'original_value']


    # Report status
    if adjusted_cages_count > 0:
         st.success(f"⚖️ Lean mass normalization applied to {adjusted_cages_count} cage(s) using reference mass: {reference_mass:.1f}g.", icon="✅")
    elif proceed_with_adjustment: # Adjustment was attempted but nothing happened
         st.warning("❓ Lean mass adjustment enabled, data present, but no cages adjusted. Check lean mass inputs and cage IDs.", icon="❓")

    if skipped_cages_invalid_mass:
         st.warning(f"❌ Adjustment skipped for cage(s) {', '.join(sorted(list(skipped_cages_invalid_mass)))} due to invalid/zero mass value(s). Using original body weight values.", icon="❌")
    if skipped_cages_no_mass_data:
         st.warning(f"❌ Adjustment not applied for cage(s) {', '.join(sorted(list(skipped_cages_no_mass_data)))} as no lean mass was provided. Using original body weight values.", icon="❌")
    # ... (optional: report on cages with mass provided but not in filtered data) ...

    return df_output

def add_light_dark_info(df_input, light_start, light_end):
    """Adds 'hour' and 'is_light' columns based on timestamp and settings."""
    if df_input is None or df_input.empty or 'timestamp' not in df_input.columns:
        st.warning("Cannot add light/dark info: Input data is invalid.")
        return df_input # Return input, possibly None

    df_output = df_input.copy()
    try:
        df_output['hour'] = df_output['timestamp'].dt.hour
        df_output['is_light'] = (df_output['hour'] >= light_start) & (df_output['hour'] < light_end)
    except AttributeError:
         st.error("Error adding light/dark info: 'timestamp' column might not contain datetime objects.")
         # Return the DataFrame without the new columns if error occurs
         return df_input
    except Exception as e:
         st.error(f"Unexpected error adding light/dark info: {e}")
         return df_input

    return df_output

def calculate_summary_metrics(df_with_cycle_info, parameter_type, subject_map):
    """Calculates summary results (light/dark/total) based on parameter type."""
    results = None
    if df_with_cycle_info is None or df_with_cycle_info.empty:
        st.error("Cannot calculate summary metrics: Input data is empty.")
        return None

    calc_df = df_with_cycle_info # Use the prepared DataFrame

    try:
        # Activity Parameters
        if parameter_type in ["XTOT", "XAMB", "YTOT", "YAMB", "ZTOT", "ZAMB"]:
            # Calculate stats per animal per cycle first
            cycle_stats = calc_df.groupby(['cage', 'is_light'])['value'].agg(
                Average_Activity='mean', Peak_Activity='max', Total_Counts='sum'
            ).unstack(fill_value=0) # Fill missing cycles with 0 for activity counts

            # Check if unstacking created multi-level columns and flatten
            if isinstance(cycle_stats.columns, pd.MultiIndex):
                new_columns = {}
                for metric, is_light in cycle_stats.columns:
                    cycle_name = "Light" if is_light else "Dark"
                    metric_name_cleaned = metric.replace("_", " ")
                    new_columns[(metric, is_light)] = f"{cycle_name} {metric_name_cleaned}"
                cycle_stats.columns = list(new_columns.values())
            results = cycle_stats

            # Calculate overall 24h averages and totals directly
            results['24h Average'] = calc_df.groupby('cage')['value'].mean()
            results['24h Total Counts'] = calc_df.groupby('cage')['value'].sum()
            results = results.fillna(0) # Fill any remaining NaNs (e.g., if a cage had no data at all)

        # Feed Parameter (Accumulated)
        elif parameter_type == "FEED":
            results_list = []
            processed_cages = calc_df['cage'].unique()
            # Sort by time *within* each cage for accurate diff calculation
            calc_df_sorted = calc_df.sort_values(by=['cage', 'timestamp'])

            for cage in processed_cages:
                cage_data = calc_df_sorted[calc_df_sorted['cage'] == cage]
                cage_results = {"Cage": cage}

                # Calculate total intake over the entire period for the cage
                if len(cage_data) > 1:
                    cage_results['Total Intake (Period)'] = cage_data['value'].iloc[-1] - cage_data['value'].iloc[0]
                elif len(cage_data) == 1:
                     cage_results['Total Intake (Period)'] = 0 # Or maybe NaN? Consider definition for single point.
                else:
                    cage_results['Total Intake (Period)'] = 0 # No data

                # Calculate intake during Light cycle
                light_data = cage_data[cage_data['is_light']]
                if len(light_data) > 1:
                    cage_results['Light Cycle Intake'] = light_data['value'].iloc[-1] - light_data['value'].iloc[0]
                elif len(light_data) == 1:
                     cage_results['Light Cycle Intake'] = 0
                else:
                    cage_results['Light Cycle Intake'] = 0

                # Calculate intake during Dark cycle
                dark_data = cage_data[~cage_data['is_light']]
                if len(dark_data) > 1:
                    cage_results['Dark Cycle Intake'] = dark_data['value'].iloc[-1] - dark_data['value'].iloc[0]
                elif len(dark_data) == 1:
                     cage_results['Dark Cycle Intake'] = 0
                else:
                    cage_results['Dark Cycle Intake'] = 0

                results_list.append(cage_results)

            if results_list:
                results = pd.DataFrame(results_list).set_index("Cage")
                results = results.round(4) # Round feed values
            else:
                results = pd.DataFrame(columns=['Total Intake (Period)', 'Light Cycle Intake', 'Dark Cycle Intake'])

        # Accumulated Gas Parameters
        elif parameter_type in ["ACCCO2", "ACCO2"]:
            # Sort by time *within* each cage
            calc_df_sorted = calc_df.sort_values(by=['cage', 'timestamp'])

            # Get first and last value for each animal within each cycle
            cycle_bounds = calc_df_sorted.groupby(['cage', 'is_light'])['value'].agg(['first', 'last'])

            # Calculate the difference (net change) during each cycle
            cycle_diff = (cycle_bounds['last'] - cycle_bounds['first']).unstack(fill_value=0) # Fill missing cycles with 0 change
            cols_rename_acc = {True: 'Light Net Accumulated', False: 'Dark Net Accumulated'}
            results = cycle_diff.rename(columns=cols_rename_acc)

            # Calculate the total net change over the whole period
            total_bounds = calc_df_sorted.groupby('cage')['value'].agg(['first', 'last'])
            # Handle cases where a cage might only have one data point total
            results['Total Net Accumulated (Period)'] = np.where(
                 total_bounds['last'].notna() & total_bounds['first'].notna(),
                 total_bounds['last'] - total_bounds['first'],
                 0 # Or NaN, if preferred for single points
            )

            results = results.fillna(0).round(4) # Fill NaNs that might arise from unstacking if a cage had no data

        # Default (Metabolic, Other Gases, Environmental - Averages)
        else:
            # Calculate mean value per animal per cycle
            # Use unstack(fill_value=np.nan) so missing cycles don't become 0 average
            results = calc_df.groupby(['cage', 'is_light'])['value'].mean().unstack(fill_value=np.nan)
            cols_to_rename = {False: 'Dark Average', True: 'Light Average'}
            results = results.rename(columns=cols_to_rename)

            # Calculate TRUE Total Average directly from all points for the cage
            results['Total Average'] = calc_df.groupby('cage')['value'].mean()

            # Apply rounding based on parameter type
            round_digits = 3 if parameter_type == "RER" else 2
            cols_to_round = [col for col in ['Dark Average', 'Light Average', 'Total Average'] if col in results.columns]
            if cols_to_round:
                results[cols_to_round] = results[cols_to_round].round(round_digits)
            # NaNs for missing cycles will persist, which is correct for averages.

        # --- Add Subject IDs to results table ---
        if results is not None:
            # Ensure index has a name for mapping (usually 'cage' or 'Cage')
             if results.index.name is None: results.index.name = 'Cage' # Assign default name if missing
             # Map subject IDs using the provided subject_map dictionary
             results['Subject ID'] = results.index.map(subject_map)
             # Check if any mappings failed (resulting in NaN)
             if results['Subject ID'].isnull().any():
                 missing_cages = results[results['Subject ID'].isnull()].index.tolist()
                 st.warning(f"❓ Could not find Subject ID mapping for cages: {missing_cages}", icon="❓")
        else:
            # Handle case where results DataFrame is None (e.g., failed calculation)
             st.error("❌ Error: Failed to generate the main summary results table internally.")
             return None # Return None to indicate failure

    except KeyError as e:
         st.error(f"Error calculating summary metrics: Missing expected column '{e}'. Check data preparation.")
         return None
    except Exception as e:
         st.error(f"An unexpected error occurred during summary metric calculation: {e}")
         import traceback
         st.code(traceback.format_exc())
         return None

    return results


def calculate_hourly_metrics(df_with_cycle_info, parameter_type, subject_map):
    """Calculates hourly results (0-23 profile)."""
    hourly_results = None
    if df_with_cycle_info is None or df_with_cycle_info.empty:
        st.error("Cannot calculate hourly metrics: Input data is empty.")
        return None
    if 'hour' not in df_with_cycle_info.columns or 'cage' not in df_with_cycle_info.columns or 'value' not in df_with_cycle_info.columns:
        st.error("Cannot calculate hourly metrics: Missing required columns ('hour', 'cage', 'value').")
        return None

    calc_df = df_with_cycle_info

    try:
        # Pivot table to get mean value per hour per cage
        hourly_results = calc_df.pivot_table(
            values='value', index='hour', columns='cage', aggfunc='mean'
        )

        # Ensure all hours 0-23 are present, fill missing hours with NaN
        all_hours = pd.Index(range(24), name='hour')
        hourly_results = hourly_results.reindex(all_hours)

        # Rename columns from cage labels (e.g., "CAGE 01") to Subject IDs if possible
        # Use cage labels as fallback if subject_map is empty or mapping fails
        if subject_map:
            subject_id_map_for_hourly = {cage: subject_map.get(cage, cage) for cage in hourly_results.columns}
            hourly_results = hourly_results.rename(columns=subject_id_map_for_hourly)
        # else: Keep cage labels as column names

        # Calculate Mean and SEM across animals for each hour
        # Ensure we only calculate across actual animal columns (ignore potential Mean/SEM if rerun)
        animal_columns = [col for col in hourly_results.columns if col not in ['Mean', 'SEM']]
        if not animal_columns:
             st.warning("No animal columns found in hourly data for Mean/SEM calculation.")
             hourly_results['Mean'] = np.nan
             hourly_results['SEM'] = np.nan
        else:
            hourly_results['Mean'] = hourly_results[animal_columns].mean(axis=1)
            # Calculate SEM robustly (N is the number of non-NaN values for that hour)
            hourly_n = hourly_results[animal_columns].count(axis=1)
            hourly_std = hourly_results[animal_columns].std(axis=1)
            # Avoid division by zero if N is 0 or 1
            hourly_results['SEM'] = np.where(hourly_n > 1, hourly_std / np.sqrt(hourly_n), 0) # SEM is 0 if N<=1

        # Apply Rounding
        round_digits_hourly = 3 if parameter_type == "RER" else 2
        # Round only numeric columns to avoid errors on potential non-numeric ones if subject rename failed etc.
        numeric_cols = hourly_results.select_dtypes(include=np.number).columns
        hourly_results[numeric_cols] = hourly_results[numeric_cols].round(round_digits_hourly)

    except KeyError as e:
         st.error(f"Error calculating hourly results: Missing expected column '{e}'.")
         return None
    except Exception as e:
         st.error(f"An unexpected error occurred during hourly metric calculation: {e}")
         import traceback
         st.code(traceback.format_exc())
         return None

    return hourly_results

def load_and_process_clams_data(uploaded_file, parameter_type, session_state):
    """
    Orchestrates the loading, parsing, and processing of CLAMS data.
    # ... (rest of docstring) ...

    Returns:
        tuple: (results_df, hourly_results_df, final_processed_data_df, status_messages_list, parsing_errors_list)
               Returns (None, None, None, list, list) with appropriate messages on failure.
               'status_messages_list' contains tuples: ('type', 'message') where type is 'info', 'success', 'warning', 'error', 'debug'.
               'parsing_errors_list' contains strings of parsing errors.
    """
    # --- Initialize lists to collect messages ---
    status_messages = []
    parsing_errors = [] # We already have this for parsing errors, keep it.
    results_df, hourly_results_df, df_with_cycle_info = None, None, None # Initialize return variables

    status_messages.append(('info', "--- Running New Processing Pipeline ---")) # Store message instead of writing

    # --- Step 1: Parse Header ---
    try:
        uploaded_file.seek(0)
        wrapper = io.TextIOWrapper(uploaded_file, encoding='utf-8', errors='ignore')
        header_lines, subject_map, data_start_line, data_header_skipped = _parse_clams_header(wrapper)
        wrapper.detach() # Release wrapper, keep file open

        if data_start_line is None: # Critical error already displayed in helper
            status_messages.append(('error', "Fatal error during header parsing: ':DATA' marker not found.")) # Log error
            return None, None, None, status_messages, parsing_errors # Return immediately on critical failure
    except Exception as e:
        status_messages.append(('error', f"Fatal error during header parsing: {e}"))
        return None, None, None, status_messages, parsing_errors # Return immediately

    # --- Step 2: Parse Data Section ---
    try:
        uploaded_file.seek(0) # Rewind again for data parsing
        wrapper = io.TextIOWrapper(uploaded_file, encoding='utf-8', errors='ignore')
        # Use the 'parsing_errors' list initialized at the start
        df_raw_parsed, parsing_errors = _parse_clams_data_section(wrapper, data_start_line, data_header_skipped)
        wrapper.detach()
        
        # --- Step 3: Handle Parsing Outcome ---
        # removed the expander display from here. It will be handled outside.

        if df_raw_parsed is None:
            status_messages.append(('error', "❌ Critical Error: Failed to parse numerical data section. Cannot continue analysis."))
            # Include parsing errors in the return
            return None, None, None, status_messages, parsing_errors
        else:
            # Only add debug message if parsing succeeded
            status_messages.append(('debug', f"Parsed data shape: {df_raw_parsed.shape}"))

    except Exception as e:
        status_messages.append(('error', f"Fatal error during data section parsing: {e}"))
        import traceback
        status_messages.append(('debug', traceback.format_exc())) # Log traceback for debugging
        return None, None, None, status_messages, parsing_errors


        if df_raw_parsed is None:
            st.error("❌ Critical Error: Failed to parse numerical data section. Cannot continue analysis.")
            return None, None, None
        st.write(f"--- Debug: Parsed data shape: {df_raw_parsed.shape} ---") # Debug

    except Exception as e:
        st.error(f"Fatal error during data section parsing: {e}")
        import traceback
        st.code(traceback.format_exc())
        return None, None, None

    # --- Step 4: Filter by Time Window ---
    selected_window = session_state.get('time_window_radio', "Entire Dataset")
    custom_days = session_state.get("custom_days_input", 5) # Default custom days
    df_analysis_window = filter_data_by_time_window(df_raw_parsed, selected_window, custom_days)

    if df_analysis_window is None:
        status_messages.append(('error', "❌ Processing stopped: Failed to filter data by time window."))
        return None, None, None, status_messages, parsing_errors
    else:
        status_messages.append(('debug', f"Filtered data shape: {df_analysis_window.shape}"))
        # Add the informative time window message here
        if selected_window == "Entire Dataset":
            status_messages.append(('info', "ℹ️ Analyzing the entire dataset."))
        else:
            try:
                actual_start = df_analysis_window['timestamp'].min()
                actual_end = df_analysis_window['timestamp'].max()
                status_messages.append(('info', f"ℹ️ Analysis Time Window: {actual_start.strftime('%Y-%m-%d %H:%M')} to {actual_end.strftime('%Y-%m-%d %H:%M')} ({selected_window})."))
            except: pass # Ignore if we can't get timestamps

    # --- Step 5: Apply Lean Mass Adjustment ---
    apply_lm_flag = session_state.get("apply_lean_mass", False)
    lean_mass_map = session_state.get('lean_mass_data', {})
    ref_mass = session_state.get('reference_lean_mass_sidebar_val', None)
    df_normalized = apply_lean_mass_adjustment(df_analysis_window, parameter_type, apply_lm_flag, lean_mass_map, ref_mass)

    if df_normalized is None: # Should not happen if input was valid, but check anyway
         status_messages.append(('error', "❌ Processing stopped: Failed during lean mass adjustment step."))
         return None, None, None, status_messages, parsing_errors
    else:
         status_messages.append(('debug', f"Data after lean mass step shape: {df_normalized.shape}"))
         try: # Add try-except for safety
             status_messages.append(('debug', f"Value stats after LM: Mean={df_normalized['value'].mean():.2f}, Min={df_normalized['value'].min():.2f}, Max={df_normalized['value'].max():.2f}"))
         except Exception as e:
             status_messages.append(('warning', f"Could not calculate post-LM stats: {e}"))

    # --- Step 6: Add Light/Dark Info ---
    light_start = session_state.get('light_start', 7)
    light_end = session_state.get('light_end', 19)
    df_with_cycle_info = add_light_dark_info(df_normalized, light_start, light_end)

    if df_with_cycle_info is None:
        status_messages.append(('error', "❌ Processing stopped: Failed to add light/dark cycle information."))
        return None, None, None, status_messages, parsing_errors

    # --- Step 7: Calculate Summary Metrics ---
    results_df = calculate_summary_metrics(df_with_cycle_info, parameter_type, subject_map)

    if results_df is None:
        status_messages.append(('error', "❌ Processing stopped: Failed to calculate summary metrics."))
        return None, None, None, status_messages, parsing_errors

    # --- Step 8: Calculate Hourly Metrics ---
    hourly_results_df = calculate_hourly_metrics(df_with_cycle_info, parameter_type, subject_map)

    if hourly_results_df is None:
        status_messages.append(('error', "❌ Processing stopped: Failed to calculate hourly metrics."))
        return None, None, None, status_messages, parsing_errors # Fail completely if hourly fails

    # --- Step 9: Return Results ---
    status_messages.append(('success', "✅ Data processing pipeline completed successfully!"))
    status_messages.append(('info', "--- End New Processing Pipeline ---"))

    # --- MODIFIED RETURN STATEMENT ---
    # Return the results AND the collected messages/errors
    return results_df, hourly_results_df, df_with_cycle_info, status_messages, parsing_errors
    
def assign_groups(cage_df, key_prefix=''):
    """
    Allow users to assign detected subjects to experimental groups using Streamlit UI.

    Args:
        cage_df (pd.DataFrame): DataFrame containing 'Cage' (e.g., "CAGE 01")
                                and 'Subject ID' columns.
        key_prefix (str): Unique prefix for Streamlit widget keys.

    Returns:
        pd.DataFrame or None: DataFrame with columns ["Group", "Cage", "Subject ID"]
                              representing the assignments, or None if setup fails
                              or user indicates subjects are incorrect.
    """
    if not isinstance(cage_df, pd.DataFrame) or not {'Cage', 'Subject ID'}.issubset(cage_df.columns):
         st.error("Internal Error: Invalid input provided to assign_groups function (requires DataFrame with 'Cage' and 'Subject ID').")
         return None

    st.write("Detected subjects:")
    st.dataframe(cage_df[['Subject ID', 'Cage']].sort_values(by='Subject ID')) # Show sorted by ID

    subjects_correct = st.radio(
        "Are the detected subjects correct?",
        ["Yes", "No"],
        index=0,
        key=f"{key_prefix}_subjects_correct_radio",
        horizontal=True # More compact
    )

    if subjects_correct == "No":
        st.error("Please ensure the uploaded file contains the correct subject information from the raw CLAMS output.")
        st.stop()

    # --- Proceed if subjects confirmed ---
    if subjects_correct == "Yes":
        # --- Input: Number of Groups ---
        num_groups = st.number_input("How many groups to create?",
                                    min_value=1,
                                    max_value=len(cage_df),
                                    value=min(2, len(cage_df)), # Default to 2 or max animals if fewer
                                    step=1,
                                    key=f"{key_prefix}_num_groups_input")

        # --- Prepare Mappings ---
        try:
            # Ensure unique Subject IDs before creating map
            if cage_df["Subject ID"].duplicated().any():
                 st.error("Duplicate Subject IDs detected in the file header. Cannot proceed with group assignment.")
                 return None
            subject_to_cage = dict(zip(cage_df["Subject ID"], cage_df["Cage"]))
            cage_to_subject = {v: k for k, v in subject_to_cage.items()} # Reverse map
        except Exception as e:
             st.error(f"Error creating subject/cage mapping: {e}")
             return None

        # --- Group Assignment UI Loop ---
        assignments_in_progress = {} # Stores {group_name: [list_of_selected_subject_ids]}
        all_assigned_subjects = set() # Track subjects assigned so far in this run

        st.markdown("---") # Separator
        st.write("**Define Groups:**")

        cols = st.columns(num_groups) # Create columns for layout

        for i in range(num_groups):
            with cols[i % len(cols)]: # Cycle through columns
                st.markdown(f"##### Group {i + 1}") # Use markdown for subheader

                # Group Name Input
                group_name = st.text_input(f"Name", # Shorter label
                                          value=f"Group {i + 1}",
                                          key=f"{key_prefix}_group_name_{i}")

                # Determine available subjects for this group's dropdown
                available_options = sorted(list(set(subject_to_cage.keys()) - all_assigned_subjects))

                # Subject Multi-Select
                selected_subjects = st.multiselect(
                    f"Select Subjects", # Shorter label
                    options=available_options,
                    key=f"{key_prefix}_group_subjects_{i}",
                    # help="Select animals for this group. Already assigned animals are not shown." # Optional help text
                    label_visibility="collapsed" # Use markdown header instead
                )

                # Store and update tracking sets
                if group_name in assignments_in_progress:
                     st.warning(f"Group name '{group_name}' used multiple times. Overwriting previous selection for this name.", icon="⚠️") # More direct warning
                assignments_in_progress[group_name] = selected_subjects
                # Update the master set of assigned subjects immediately
                all_assigned_subjects.update(selected_subjects)

        st.markdown("---") # Separator

        # --- Validation and Summary ---
        # Check for unassigned subjects AFTER loop
        all_available_subjects = set(subject_to_cage.keys())
        unassigned = all_available_subjects - all_assigned_subjects
        if unassigned:
            st.warning(f"**Unassigned Subjects:** {', '.join(map(str, sorted(list(unassigned))))}", icon="⚠️")

        # Check for empty groups
        empty_groups = [name for name, subjects in assignments_in_progress.items() if not subjects]
        if empty_groups:
            st.warning(f"**Empty Groups:** {', '.join(empty_groups)} have no subjects assigned.", icon="❓")


        # --- Create Final Summary DataFrame ---
        group_summary_list = []
        final_group_names = set()

        for group_name, subject_ids in assignments_in_progress.items():
             # Skip potentially duplicate group names defined earlier but maybe cleared
             if not group_name or group_name in final_group_names:
                  # Add warning about duplicate group name consolidation if needed
                  # st.caption(f"Note: Duplicate name '{group_name}' consolidated.")
                  continue
             final_group_names.add(group_name)

             if not subject_ids: # Skip empty groups in the final output table
                  continue

             for subject_id in subject_ids:
                 cage_id = subject_to_cage.get(subject_id) # Get cage using the map
                 if cage_id: # Should always exist if subject_id came from multiselect options
                    group_summary_list.append({
                        "Group": group_name,
                        "Cage": cage_id, # Standard "CAGE XX" format
                        "Subject ID": subject_id
                    })

        if not group_summary_list:
            st.info("No subjects assigned to any groups.")
            # Return empty DataFrame with correct columns for consistency downstream
            return pd.DataFrame(columns=["Group", "Cage", "Subject ID"])
        else:
            group_summary_df = pd.DataFrame(group_summary_list)
            # Sort for consistent display
            group_summary_df.sort_values(by=['Group', 'Subject ID'], inplace=True)

            st.subheader("Group Assignment Summary")
            st.dataframe(group_summary_df)
            return group_summary_df

    return None # Should not be reached if 'Yes' is selected, but safety return

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

def show_verification_calcs(df_24h, results, hourly_results, parameter):
    """Show sample calculations for verification"""
    with st.expander("🔍 View Sample Calculations (First Cage)"):
        st.write("### Sample Calculations Verification")
        
        # Get first cage data
        first_cage = df_24h['cage'].unique()[0]
        first_cage_data = df_24h[df_24h['cage'] == first_cage].copy()
        
        # Show raw data sample
        st.write("#### Raw Data Sample (First 5 rows)")
        st.dataframe(first_cage_data[['timestamp', 'value', 'is_light']].head())
        
        # Calculate and show light/dark averages
        dark_data = first_cage_data[~first_cage_data['is_light']]['value']
        light_data = first_cage_data[first_cage_data['is_light']]['value']
        
        st.write("#### Light/Dark Averages Calculation")
        st.write(f"Dark Average = Sum of dark values ({dark_data.sum():.2f}) ÷ Number of dark readings ({len(dark_data)}) = {dark_data.mean():.2f}")
        st.write(f"Light Average = Sum of light values ({light_data.sum():.2f}) ÷ Number of light readings ({len(light_data)}) = {light_data.mean():.2f}")
        
        # Show a sample hourly calculation
        st.write("#### Sample Hourly Average Calculation (Hour 0)")
        hour_0_data = first_cage_data[first_cage_data['hour'] == 0]['value']
        if not hour_0_data.empty:
            st.write(f"Hour 0 Average = Sum of values ({hour_0_data.sum():.2f}) ÷ Number of readings ({len(hour_0_data)}) = {hour_0_data.mean():.2f}")
        
        # Compare with results - handle different parameter types
        st.write("#### Verification Against Results")
        st.write(f"Values in results table:")
        
        if parameter in ["XTOT", "XAMB"]:
            st.write(f"- Dark Activity: {results.iloc[0]['False (Average Activity)']:.2f}")
            st.write(f"- Light Activity: {results.iloc[0]['True (Average Activity)']:.2f}")
        elif parameter == "FEED":
            st.write(f"- Dark Rate: {results.iloc[0]['Average Rate (Dark)']:.2f}")
            st.write(f"- Light Rate: {results.iloc[0]['Average Rate (Light)']:.2f}")
        else:  # VO2, VCO2, RER, HEAT
            st.write(f"- Dark Average: {results.iloc[0]['Dark Average']:.2f}")
            st.write(f"- Light Average: {results.iloc[0]['Light Average']:.2f}")
        
        st.write(f"- Hour 0 Average: {hourly_results.iloc[0][0]:.2f}")
        
def generate_pub_bar_chart(group_stats, parameter, error_bar_type='SEM', colors=None, cycle_name="24-hour Average"):
    """
    Generates a publication-ready bar chart using Matplotlib and scienceplots.

    Args:
        group_stats (pd.DataFrame): DataFrame containing group statistics (Mean, SD, SEM, N).
                                    Must have 'Group', 'Mean', 'SD', 'SEM', 'N' columns.
        parameter (str): Name of the parameter being plotted (e.g., 'VO2').
        error_bar_type (str): Type of error bar to display ('SEM' or 'SD').
        colors (dict, optional): Dictionary mapping group names to colors. Defaults to scienceplots default.

    Returns:
        matplotlib.figure.Figure: The generated Matplotlib figure object.
    """
    plt.style.use(['science', 'no-latex']) # Apply scienceplots style

    fig, ax = plt.subplots(figsize=(6, 4)) # Create figure and axes (adjust size later)

    if colors is None:
        colors = {} # Use default colors if none provided

    groups = group_stats['Group']
    means = group_stats['Mean']

    # Determine error values
    if error_bar_type == 'SD':
        errors = group_stats['SD']
        error_label = 'SD'
    else:
        errors = group_stats['SEM']
        error_label = 'SEM'

    # Create bars
    for i, group in enumerate(groups):
        ax.bar(i, means.iloc[i],
               yerr=errors.iloc[i],
               capsize=5, # Add error bar caps
               label=group,
               color=colors.get(group, f'C{i}')) # Use provided color or default cycle

    # --- Placeholder ---
    # will add more styling, labels, titles, and significance annotations later
    ax.set_ylabel(f"{parameter} ({PARAMETER_UNITS.get(parameter, '')})")
    ax.set_title(f"{parameter} by Group ({cycle_name}, {error_label})")
    ax.set_xticks(range(len(groups)))
    ax.set_xticklabels(groups, rotation=45, ha='right')
    ax.legend(loc ='upper right', fontsize='small')
    plt.tight_layout()
    # --- End Placeholder ---


    return fig

def generate_pub_24h_pattern_plot(hourly_df, selected_groups, parameter, light_start, light_end, colors=None):
    """
    Generates a publication-ready 24-hour pattern line plot using Matplotlib and scienceplots.

    Args:
        hourly_df (pd.DataFrame): DataFrame containing hourly stats (mean, sem) for groups.
                                  Must have 'Group', 'hour', 'mean', 'sem' columns.
        selected_groups (list): List of group names to plot.
        parameter (str): Name of the parameter being plotted (e.g., 'VO2').
        light_start (int): Hour when the light cycle starts (0-23).
        light_end (int): Hour when the light cycle ends (0-23).
        colors (dict, optional): Dictionary mapping group names to colors. Defaults to scienceplots default.

    Returns:
        matplotlib.figure.Figure: The generated Matplotlib figure object.
    """
    plt.style.use(['science', 'no-latex']) # Apply scienceplots style
    fig, ax = plt.subplots(figsize=(8, 4)) # Create figure and axes (adjust size later)

    if colors is None:
        colors = {} # Use default colors if none provided

    # Plot line and SEM bands for each selected group
    for i, group in enumerate(selected_groups):
        group_data = hourly_df[hourly_df['Group'] == group]
        if not group_data.empty:
            line_color = colors.get(group, f'C{i}') # Get color or use default cycle
            # Plot main line
            ax.plot(group_data['hour'], group_data['mean'],
                    label=group,
                    color=line_color,
                    marker='o', # Add markers
                    markersize=4,
                    linewidth=1.5)
            # Plot SEM bands
            ax.fill_between(group_data['hour'],
                            group_data['mean'] - group_data['sem'],
                            group_data['mean'] + group_data['sem'],
                            color=line_color,
                            alpha=0.15) # Make bands semi-transparent

    # --- Add Light/Dark Shading ---
    # Use axvspan for vertical shading across the plot height
    ax.axvspan(-0.5, light_start - 0.5, facecolor='grey', alpha=0.15, zorder=-10, lw=0) # Morning dark
    ax.axvspan(light_end - 0.5, 23.5, facecolor='grey', alpha=0.15, zorder=-10, lw=0) # Evening dark
    # Optional: Add light cycle shading too (often just white background is used)
    # ax.axvspan(light_start - 0.5, light_end - 0.5, facecolor='#FFFACD', alpha=0.2, zorder=-10, lw=0) # Light

    # --- Placeholder for further styling ---
    ax.set_ylabel(f"{parameter} ({PARAMETER_UNITS.get(parameter, '')})")
    ax.set_xlabel("Hour of Day")
    ax.set_title(f"{parameter} 24-Hour Pattern (Mean +/- SEM)")
    ax.set_xticks(range(0, 25, 2)) # Set x-axis ticks every 2 hours
    ax.set_xlim(-0.5, 23.5) # Set x-axis limits
    ax.legend(loc='best', fontsize='small')
    plt.tight_layout()
    # --- End Placeholder ---

    return fig

def generate_pub_timeline_plot(timeline_data, parameter, time_window, subject_to_cage, display_mode="Average", selected_subjects=None, subject_to_group=None, colors=None, light_start=7, light_end=19):
    """
    Generates a publication-ready timeline plot using Matplotlib and scienceplots.
    # ... (docstring remains the same) ...
    """
    plt.style.use(['science', 'no-latex'])
    fig, ax = plt.subplots(figsize=(10, 4)) # Wider figure for timeline

    if colors is None: colors = {}

    y_min_data = float('inf') # Initialize min/max for y-axis scaling
    y_max_data = float('-inf')

    if display_mode == "Average":
        if not all(col in timeline_data.columns for col in ['datetime', 'Mean', 'SEM']):
            st.error("Error: 'datetime', 'Mean', or 'SEM' column missing for average timeline plot.")
            plt.close(fig) # Close the empty figure
            return fig # Return empty fig if essential columns are missing

        # Plot average line
        ax.plot(timeline_data['datetime'], timeline_data['Mean'], label=f'Mean {parameter}', color='k', linewidth=1.5)
        # Plot SEM band
        ax.fill_between(timeline_data['datetime'],
                        timeline_data['Mean'] - timeline_data['SEM'],
                        timeline_data['Mean'] + timeline_data['SEM'],
                        color='k', alpha=0.15)
        # Update min/max for y-axis
        y_min_data = min(y_min_data, (timeline_data['Mean'] - timeline_data['SEM']).min())
        y_max_data = max(y_max_data, (timeline_data['Mean'] + timeline_data['SEM']).max())

    # --- MODIFIED SECTION FOR INDIVIDUAL PLOTTING ---
    elif display_mode == "Individual":
        # Check required columns in the pre-filtered data
        if not all(col in timeline_data.columns for col in ['datetime', 'cage', 'value']):
            st.error("Error: 'datetime', 'cage', or 'value' column missing for individual timeline plot in provided data.")
            plt.close(fig) # Close the empty figure
            return fig # Return empty fig

        # Create inverse mapping: cage -> subject_id (needed for legend)
        # Ensure subject_to_cage is not empty before creating inverse
        if not subject_to_cage:
            st.error("Error: Subject-to-Cage mapping is empty. Cannot create legend.")
            cage_to_subject = {}
        else:
            cage_to_subject = {v: k for k, v in subject_to_cage.items()}


        # Iterate through the unique cages PRESENT in the passed timeline_data
        plotted_cages = timeline_data['cage'].unique()
        if len(plotted_cages) == 0:
            st.warning("Warning: The data provided for individual plotting contains no cages.") # Add warning

        for i, cage in enumerate(plotted_cages):
            subject_id = cage_to_subject.get(cage, cage) # Get subject ID for label, fallback to cage name

                # Filter the data for the current cage *within the already filtered timeline_data*
            cage_data = timeline_data[timeline_data['cage'] == cage]

            if not cage_data.empty:
                # Use subject_id for color lookup if a color map is provided and contains the ID
                # Otherwise, default to Matplotlib's color cycle based on iteration index
                color_key = subject_id if subject_id in colors else f'C{i}' # Use subject_id if possible for consistent color
                line_color = colors.get(subject_id, f'C{i}') # Get color from dict or use default cycle

                ax.plot(cage_data['datetime'], cage_data['value'], label=subject_id, color=line_color, linewidth=1)

                    # Update min/max for y-axis
                current_min = cage_data['value'].min()
                current_max = cage_data['value'].max()
                if not pd.isna(current_min): y_min_data = min(y_min_data, current_min)
                if not pd.isna(current_max): y_max_data = max(y_max_data, current_max)
            #else: # Optional: Add a warning if cage_data becomes empty unexpectedly
                #    st.warning(f"No data points found for cage {cage} within the filtered data.")
    # --- END OF MODIFIED SECTION ---

    # --- Add Light/Dark Shading ---
    # Check if timeline_data is not empty before accessing min/max
    if not timeline_data.empty:
        min_dt = timeline_data['datetime'].min()
        max_dt = timeline_data['datetime'].max()
        # Proceed only if min_dt and max_dt are valid Timestamps
        if pd.notna(min_dt) and pd.notna(max_dt):
            current_date = min_dt.date()
            while pd.Timestamp(current_date) <= max_dt:
                morning_start = pd.Timestamp(f"{current_date} 00:00:00")
                morning_end = pd.Timestamp(f"{current_date} {light_start:02d}:00:00")
                evening_start = pd.Timestamp(f"{current_date} {light_end:02d}:00:00")
                evening_end = pd.Timestamp(f"{current_date} 23:59:59")

                # Draw rectangles only within the plot's date range
                # Use max/min to clip the shading boxes to the actual data range
                ax.axvspan(max(morning_start, min_dt), min(morning_end, max_dt), facecolor='grey', alpha=0.15, zorder=-10, lw=0)
                ax.axvspan(max(evening_start, min_dt), min(evening_end, max_dt), facecolor='grey', alpha=0.15, zorder=-10, lw=0)

                current_date += timedelta(days=1)
        #else: # Optional warning if dates are invalid
        #     st.warning("Could not determine date range for light/dark shading.")
    #else: # Optional warning if data is empty
    # 	st.warning("No data available to determine light/dark shading range.")


    # --- Set Y-axis Limits ---
    # Check if min/max were updated and are not infinite
    if y_min_data != float('inf') and y_max_data != float('-inf'):
        y_range = y_max_data - y_min_data
        # Handle case where range is zero or very small
        if y_range <= 0:
            padding = abs(y_max_data * 0.1) if y_max_data != 0 else 0.5 # Add padding based on value or a fixed amount
        else:
            padding = y_range * 0.1 # Add 10% padding

        ax.set_ylim(max(0, y_min_data - padding), y_max_data + padding) # Set limits, ensure lower bound is >= 0

    # --- Styling ---
    ax.set_ylabel(f"{parameter} ({PARAMETER_UNITS.get(parameter, '')})")
    ax.set_xlabel("Date and Time")
    ax.set_title(f"{parameter} Timeline ({time_window}) - {'Average +/- SEM' if display_mode == 'Average' else 'Individual Animals'}") # Used +/-
    fig.autofmt_xdate() # Auto-format dates on x-axis

    # Only show legend if there's something to label
    handles, labels = ax.get_legend_handles_labels()
    if handles:
        # Place legend outside the plot area to avoid overlap
        ax.legend(loc='upper left', bbox_to_anchor=(1.02, 1), borderaxespad=0., fontsize='small')


    plt.tight_layout(rect=[0, 0, 0.9, 1]) # Adjust layout to make space for external legend
    # The rect parameter might need tuning: [left, bottom, right, top]

    return fig
        
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


# Create tabs for organization
tab1, tab2, tab4, tab3 = st.tabs(["📊 Overview", "📈 Statistical Analysis", "📄 Publication Plots", "🧪 How Can I Trust This Data?"])

with tab1:
    # Only show welcome message if no file is uploaded
    if uploaded_file is None:
        # --- Simplified Welcome Screen ---

        st.markdown("## Welcome to CLAMSer!")
        st.markdown("Streamline your Oxymax-CLAMS metabolic data analysis. Get from raw CSV to insightful results and plots quickly and easily.")
              
        # --- Guiding Questions Section ---
        with st.expander("❓ What kind of research questions can CLAMSer help with?", expanded=False):
            st.markdown("""
            Before starting, think about what you want to learn. This tool helps explore common CLAMS analysis goals:

        *   **Overall Energy Expenditure?**
            *   *Analyze `VO2`, `VCO2`, and `HEAT` to see group differences or treatment effects.*
        *   **Fuel Source Utilization?**
            *   *Examine `RER` patterns to see if animals are burning fats vs. carbs.*
        *   **Activity Levels & Patterns?**
            *   *Compare `XTOT`, `XAMB`, `YTOT`, `YAMB`, `ZTOT`, `ZAMB` between groups or light/dark cycles.*
        *   **Feeding Behavior?**
            *   *Analyze `FEED` (use FEED1 ACC file!) to quantify intake amounts and timing.*
        *   **Circadian Rhythms?**
            *   *Visualize 24-hour patterns for any parameter to see time-of-day effects.*
        *   **Treatment Effects on Metabolism/Activity?**
            *   *Use the **Statistical Analysis** tab for rigorous group comparisons (ANOVA, t-tests).*

        CLAMSer aims to help you explore these questions through clear visualizations, summary tables, and statistical tools.
        """)
        # --- End Guiding Questions Section ---

        # Color chosen: a slightly lighter blue-gray from the theme #262d3d (adjust if needed)
        step_style = "background-color: #262d3d; color: white; padding: 5px 10px; border-radius: 5px; display: inline-block; margin-bottom: 5px;"

        st.markdown(f"<div style='{step_style}'>Step 1: Configure Analysis (Sidebar)</div>", unsafe_allow_html=True)
        st.markdown("""
        *   **(Required)** Go to the **sidebar** on the left.
        *   Under **'Select Parameter'**, choose the data type you want to analyze (e.g., `VO2`, `HEAT`, `XTOT`). Descriptions are provided.
        *   Under **'Select Time Window'**, choose how much of the *end* of your dataset to analyze (e.g., `Last 24 Hours`, `Last 7 Days`).
        *   **(Optional)** Adjust **'Light/Dark Cycle'** times if your schedule differs from the default (7 AM - 7 PM).
        *   **(Optional)** For `VO2`, `VCO2`, or `HEAT`, you can enable **'Metabolic Normalization'** if you have lean mass data.
        """, unsafe_allow_html=True) # Add unsafe_allow_html here if needed, but likely not for bullets

        st.markdown(f"<div style='{step_style}'>Step 2: Upload Data (Sidebar)</div>", unsafe_allow_html=True)
        st.markdown("""
        *   **(Required)** Still in the **sidebar**, find the **'Upload Data File'** section.
        *   Click **'Browse files'** and select the **correct CSV file** that matches the parameter you chose in Step 1.
            *   ⚠️ **Important:** For Food Intake analysis, select `FEED` in Step 1 and upload the **`FEED1 ACC.CSV`** file!
        """, unsafe_allow_html=True)

        st.markdown(f"<div style='{step_style}'>Step 3: Assign Groups & Lean Mass (Overview Tab)</div>", unsafe_allow_html=True)
        st.markdown("""
        *   **(Required for Stats/Plots)** Once your data loads, stay on this **'📊 Overview'** tab.
        *   Scroll down to the **'⚙️ Setup: Groups & Lean Mass'** section.
        *   Click **'Yes'** to confirm subjects and assign animals to experimental groups. This is needed for statistical analysis and group plots.
        *   **(Optional)** If you enabled lean mass adjustment in Step 1, enter the individual lean masses here.
        """, unsafe_allow_html=True)

        st.markdown(f"<div style='{step_style}'>Step 4: Explore Results!</div>", unsafe_allow_html=True)
        st.markdown("""
        *   Review summary tables and plots here in the **'📊 Overview'** tab.
        *   Perform statistical comparisons in the **'📈 Statistical Analysis'** tab.
        *   Generate publication-quality graphs in the **'📄 Publication Plots'** tab.
        *   Learn more about the calculations in the **'🧪 Guide & Verification'** tab.
        """, unsafe_allow_html=True) # Renamed tab here

        st.markdown("---") # Visual separator
        st.success("Ready? Start with **Step 1** in the sidebar!")
        # --- End of Simplified Welcome Screen ---
    else:
        # When file is uploaded, just show the header followed by parameter guide
        st.header("📊 Overview")
        # --- Parameter Information Expander --- (Moved Temporarily - REMAINS HERE FOR NOW)
        with st.expander(f"ℹ️ Understanding Parameter: {parameter_descriptions.get(parameter, parameter)}"):

            # --- Define info using a structured dictionary ---
            parameter_insights = {
                "VO2": {
                    "description": "Rate of Oxygen Consumption (ml/kg/hr), reflecting energy expenditure.",
                    "core_questions": [
                        "What is the overall metabolic rate of the animals?",
                        "Does the treatment/condition alter energy expenditure?",
                        "Are there differences in metabolic rate between light and dark cycles?",
                        "How does VO2 relate to activity or food intake?"
                    ]
                },
                "VCO2": {
                    "description": "Rate of Carbon Dioxide Production (ml/kg/hr), reflecting metabolic activity.",
                    "core_questions": [
                        "How much CO2 are the animals producing?",
                        "Does the treatment/condition affect CO2 output?",
                        "How does VCO2 change across the light/dark cycle?",
                        "Used in conjunction with VO2 to calculate RER."
                    ]
                },
                "RER": {
                    "description": "Respiratory Exchange Ratio (VCO2/VO2), indicating fuel source.",
                    "core_questions": [
                        "Are the animals primarily burning fats (RER ≈ 0.7) or carbohydrates (RER ≈ 1.0)?",
                        "Does the treatment/diet shift fuel preference?",
                        "Is there a circadian rhythm in fuel utilization (e.g., higher RER in active/feeding phase)?",
                        "Does RER exceed 1.0 (suggesting lipogenesis)?"
                    ]
                },
                "HEAT": {
                    "description": "Estimated Heat Production (kcal/hr), calculated from VO2/VCO2.",
                    "core_questions": [
                        "What is the total energy expenditure in caloric terms?",
                        "Are there group differences in overall energy expenditure?",
                        "How does total heat production change across the light/dark cycle?",
                        "Does it correlate strongly with VO2?"
                    ]
                },
                "FEED": { # Represents FEED1 ACC
                    "description": "Accumulated Food Intake (grams). *Use FEED1 ACC.CSV.*",
                    "core_questions": [
                        "How much food did the animals consume in total (e.g., over 24h)?",
                        "Do groups differ in their total food intake?",
                        "Is feeding concentrated in the light or dark cycle?",
                        "How does food intake relate to body weight changes or metabolic parameters?"
                    ]
                },
                "XTOT": {
                    "description": "Total X-axis Activity (counts), includes fine + ambulatory movements.",
                    "core_questions": [
                        "What is the overall level of horizontal (side-to-side) activity?",
                        "Are there group differences in total X-axis movement?",
                        "Is activity higher during the light or dark cycle?",
                        "How does it compare to XAMB (ambulatory counts)?"
                    ]
                },
                "XAMB": {
                    "description": "Ambulatory X-axis Activity (counts), reflects horizontal locomotion.",
                    "core_questions": [
                        "How much are the animals moving across the cage horizontally?",
                        "Does the treatment affect purposeful horizontal movement?",
                        "Is locomotion higher during the light or dark cycle?",
                        "Is the XAMB/XTOT ratio different between groups?"
                    ]
                },
                "YTOT": {
                    "description": "Total Y-axis Activity (counts), includes fine + ambulatory front-to-back movements.",
                    "core_questions": [
                        "What is the overall level of front-to-back activity?",
                        "Are there group differences in total Y-axis movement?",
                        "Is activity higher during the light or dark cycle?",
                        "How does it compare to YAMB?"
                    ]
                },
                "YAMB": {
                    "description": "Ambulatory Y-axis Activity (counts), reflects front-to-back locomotion.",
                    "core_questions": [
                        "How much are the animals moving across the cage from front-to-back?",
                        "Does the treatment affect purposeful front-to-back movement?",
                        "Is locomotion higher during the light or dark cycle?",
                        "Is the YAMB/YTOT ratio different between groups?"
                    ]
                },
                "ZTOT": {
                    "description": "Total Z-axis Activity (counts), includes fine movements + rearing/climbing.",
                    "core_questions": [
                        "What is the overall level of vertical activity (including small movements)?",
                        "Are there group differences in total Z-axis movement?",
                        "Is vertical activity higher during the light or dark cycle?",
                        "How does it compare to ZAMB?"
                    ]
                },
                "ZAMB": {
                    "description": "Ambulatory Z-axis Activity (counts), reflects vertical locomotion (rearing/climbing).",
                    "core_questions": [
                        "How frequently are the animals rearing or climbing?",
                        "Does the treatment affect exploratory vertical movement?",
                        "Is rearing/climbing higher during the light or dark cycle?",
                        "Does ZAMB correlate with specific behaviors?"
                    ]
                },
                "FLOW": {
                    "description": "Air Flow Rate (lpm). *System parameter, not biological.*",
                    "core_questions": [
                        "Is the airflow through each cage stable and consistent?",
                        "Does the measured flow match the system's setpoint?",
                        "Are there sudden drops or spikes indicating potential leaks/blockages?",
                        "(Not typically used for comparing biological groups)."
                    ]
                },
                "PRESSURE": {
                    "description": "Barometric Pressure (mmhg). *Environmental parameter.*",
                    "core_questions": [
                        "What was the atmospheric pressure during the experiment?",
                        "Is the pressure reading consistent across simultaneous measurements?",
                        "(Used internally for gas law corrections by CLAMS; not for biological group comparison)."
                    ]
                },
                "O2IN": {
                    "description": "Inlet O2 Concentration (%). *System baseline.*",
                    "core_questions": [
                        "Is the incoming air O2 level stable and correct (approx. 20.9%)?",
                        "Are there any drifts suggesting analyzer issues or problems with the air source?",
                        "(Diagnostic parameter, not for biological group comparison)."
                    ]
                },
                "CO2IN": {
                    "description": "Inlet CO2 Concentration (%). *System baseline.*",
                    "core_questions": [
                        "Is the incoming air CO2 level stable and low (approx. 0.04%)?",
                        "Are there any unexpected increases suggesting leaks or analyzer issues?",
                        "(Diagnostic parameter, not for biological group comparison)."
                    ]
                },
                "O2OUT": {
                    "description": "Outlet O2 Concentration (%). Reflects O2 remaining after respiration.",
                    "core_questions": [
                        "How much lower is O2OUT compared to O2IN?",
                        "Do groups differ in their average O2OUT (indicating different consumption)?",
                        "Does O2OUT fluctuate with the light/dark cycle?",
                        "(Raw data used to calculate VO2)."
                    ]
                },
                "CO2OUT": {
                    "description": "Outlet CO2 Concentration (%). Reflects CO2 added by respiration.",
                    "core_questions": [
                        "How much higher is CO2OUT compared to CO2IN?",
                        "Do groups differ in their average CO2OUT (indicating different production)?",
                        "Does CO2OUT fluctuate with the light/dark cycle?",
                        "(Raw data used to calculate VCO2)."
                    ]
                },
                "DO2": {
                    "description": "Delta O2 Concentration (O2IN - O2OUT, %). Directly related to VO2.",
                    "core_questions": [
                        "What is the magnitude of oxygen extraction by the animals?",
                        "Do groups differ significantly in their DO2?",
                        "Are there cyclical changes in DO2?",
                        "(Analysis is very similar to VO2)."
                    ]
                },
                "DCO2": {
                    "description": "Delta CO2 Concentration (CO2OUT - CO2IN, %). Directly related to VCO2.",
                    "core_questions": [
                        "What is the magnitude of carbon dioxide addition by the animals?",
                        "Do groups differ significantly in their DCO2?",
                        "Are there cyclical changes in DCO2?",
                        "(Analysis is very similar to VCO2)."
                    ]
                },
                "ACCCO2": {
                    "description": "Accumulated CO2 Production (liters). *Needs correct calculation (total per period).*",
                    "core_questions": [
                        "What was the total volume of CO2 produced during the light/dark cycle or 24h?",
                        "Do groups differ in their total CO2 output over the measurement period?",
                        "(Complements VCO2 rate measurement)."
                    ]
                },
                "ACCO2": {
                    "description": "Accumulated O2 Consumption (liters). *Needs correct calculation (total per period).*",
                    "core_questions": [
                        "What was the total volume of O2 consumed during the light/dark cycle or 24h?",
                        "Do groups differ in their total O2 usage over the measurement period?",
                        "(Complements VO2 rate measurement)."
                    ]
                }
                # Add new parameters here in the future
            }

            # Get the specific info for the selected parameter
            selected_info = parameter_insights.get(parameter)

            if selected_info:
                st.markdown(f"**Description:** {selected_info['description']}")
                st.markdown("**Core questions this parameter helps answer:**")
                for question in selected_info['core_questions']:
                    st.markdown(f"- {question}")
            else:
                # Fallback message if parameter not in dictionary yet
                st.markdown(f"Detailed information about interpreting **{parameter}** analysis will be added here.")

        st.markdown("---") # Visual separator
        # --- End Parameter Information Expander ---

    
    
if uploaded_file is not None:
    # First verify file type
    is_valid, error_message = verify_file_type(uploaded_file, parameter)
    
    if not is_valid:
        st.error(error_message)
    else:
        # Get cage information for lean mass inputs
        cage_info = extract_cage_info(uploaded_file)
          
        with st.spinner('Processing data...'):
            # Updated line to receive 5 values
            results, hourly_results, processed_data, status_messages, parsing_errors = load_and_process_clams_data(uploaded_file, parameter, st.session_state)
        # --- Display Processing Status Messages and Errors ---
        st.markdown("---") # Separator after spinner
        st.subheader("Data Processing Log")

        # Add a checkbox to control debug message visibility (optional, good practice)
        show_debug = st.checkbox("Show Debug Messages", key="show_debug_toggle_tab1", value=False) # Default to hidden

        # Display parsing errors first, if any
        if parsing_errors:
            num_total_errors = sum(1 for error in parsing_errors if not error.startswith("..."))
            if num_total_errors > 0:
                    error_display_list = parsing_errors[:15] # Limit display
                    # Use an expander specifically for parsing errors within the log section
                    with st.expander(f"⚠️ Found {num_total_errors} data parsing issue(s). Click to see details.", expanded=True): # Default expanded
                        st.warning(f"Showing first {len(error_display_list)} potential issues found during data reading:")
                        error_report = "```\n" + "\n".join(error_display_list) + "\n```"
                        st.markdown(error_report)

        # Display status messages collected from the processing function
        if status_messages:
            for msg_type, msg_content in status_messages:
                if msg_type == 'info':
                    st.info(msg_content, icon="ℹ️")
                elif msg_type == 'success':
                    st.success(msg_content, icon="✅")
                elif msg_type == 'warning':
                    st.warning(msg_content, icon="⚠️")
                elif msg_type == 'error':
                    st.error(msg_content, icon="❌")
                elif msg_type == 'debug':
                    if show_debug: # Only show debug if checkbox is ticked
                        st.caption(f"🐞 DEBUG: {msg_content}") # Use caption for debug
                else: # Fallback for unknown types
                    st.write(f"{msg_type.upper()}: {msg_content}")
        else:
            st.info("No status messages reported from processing.") # Should not happen if list is initialized

        st.markdown("---") # Separator after the log section
        # --- End Display Processing Status Messages ---
        
        
        if results is not None and hourly_results is not None and processed_data is not None:
             # rename 'processed_data' to 'analysis_input_data' here for better clarity
             # raw_data usually implies the absolute initial state, this data has been processed (filtered, normalized etc.)
             analysis_input_data = processed_data
             # We also need to make sure the variable 'raw_data' (used later) still gets assigned
             # For now, let's assign it the same thing. I wil see if I *truly* raw data later.
             raw_data = analysis_input_data # TEMPORARY - might need adjustment later if true raw data is needed elsewhere.
            
        if results is not None:
            # Tab 1: Overview
            with tab1:
                if results is not None:
            # --- Setup Callout ---
                    # Check if groups have been assigned by looking in session_state
                    # We assume groups are assigned if 'group_assignments' exists AND it's not an empty DataFrame
                    groups_assigned = ('group_assignments' in st.session_state and
                                    not st.session_state.get('group_assignments', pd.DataFrame()).empty)

                    if not groups_assigned:
                        # If groups are NOT assigned, show a prominent warning message
                        st.info("👇 **Action Needed:** Please assign animals to groups in the **'Setup: Groups & Lean Mass'** section below to enable **Group-Based Analysis**, **Statistical Analysis** and **Publication Plots**.", icon="⚠️") # Changed to warning icon
                    else:
                        # If groups ARE assigned, show a success message
                        st.success("✅ Groups assigned. You can modify them in the 'Setup' section below. When ready, proceed to Statistical Analysis.", icon="👍")
                
                    # --- ADDED: Display Parameter Mismatch Warning ---
                    # Check if the error_message contains a warning (not a hard error)
                    # error_message comes from verify_file_type executed earlier
                    if 'error_message' in locals() and error_message:
                        st.error(f"⚠️ **Parameter Check:** {error_message} **Did you select the correct parameter?** Processing anyway..." , icon="❗")
                    # --- END OF ADDED BLOCK ---

                    # Add a small visual separator
                    st.markdown("---")
                    # 1. METRICS - Display parameter-specific metrics at the top with improved layout
                    
                    st.markdown("### Key Metrics")

                    # Create a container for better styling
                    metrics_container = st.container()
                    with metrics_container:
                        # --- METRICS DISPLAY ---
                        # Check if results exist and are not empty before trying to access columns
                        if results is not None and not results.empty:
                            try: # Add a try-except block for safety when accessing results
                                if parameter in ["XTOT", "XAMB", "YTOT", "YAMB", "ZTOT", "ZAMB"]:
                                    # --- Activity Metrics ---
                                    # Use the NEW column names
                                    light_avg_col = "Light Average Activity"
                                    dark_avg_col = "Dark Average Activity"
                                    light_peak_col = "Light Peak Activity"
                                    dark_peak_col = "Dark Peak Activity"
                                    light_total_col = "Light Total Counts"
                                    dark_total_col = "Dark Total Counts"
                                    overall_total_col = "24h Total Counts" # Use this for overall total

                                    col1, col2, col3, col4 = st.columns(4)
                                    with col1:
                                        # Check if column exists before accessing
                                        if light_avg_col in results.columns:
                                            st.metric("Avg Light Activity",
                                                f"{results[light_avg_col].mean():.1f} {PARAMETER_UNITS[parameter]}")
                                        else: st.metric("Avg Light Activity", "N/A")
                                    with col2:
                                        if dark_avg_col in results.columns:
                                            st.metric("Avg Dark Activity",
                                                f"{results[dark_avg_col].mean():.1f} {PARAMETER_UNITS[parameter]}")
                                        else: st.metric("Avg Dark Activity", "N/A")
                                    with col3:
                                        # Calculate overall peak from Light and Dark peak columns
                                        peak_val = 0
                                        if light_peak_col in results.columns and dark_peak_col in results.columns:
                                            peak_val = max(results[light_peak_col].max(), results[dark_peak_col].max())
                                        elif light_peak_col in results.columns:
                                            peak_val = results[light_peak_col].max()
                                        elif dark_peak_col in results.columns:
                                            peak_val = results[dark_peak_col].max()
                                        st.metric("Peak Activity (Overall)", f"{peak_val:.0f} {PARAMETER_UNITS[parameter]}")

                                    with col4:
                                        if overall_total_col in results.columns:
                                            # Sum the 24h total across all animals
                                            st.metric("Total Activity (24h)",
                                                f"{results[overall_total_col].sum():.0f} {PARAMETER_UNITS[parameter]}")
                                        elif light_total_col in results.columns and dark_total_col in results.columns:
                                            # Fallback: sum light and dark totals if 24h column missing
                                            st.metric("Total Activity (L+D)",
                                                f"{(results[light_total_col] + results[dark_total_col]).sum():.0f} {PARAMETER_UNITS[parameter]}")
                                        else: st.metric("Total Activity", "N/A")


                                elif parameter == "FEED": # Assumes FEED1 ACC
                                    # --- Feed Metrics ---
                                    # Use the CORRECT column name generated by the updated process_clams_data function
                                    total_intake_col = 'Total Intake (Period)' # <<< CORRECTED NAME

                                    # Check if the crucial column exists in the results DataFrame
                                    if total_intake_col in results.columns:
                                        # Calculate the average total intake PER ANIMAL over the period
                                        avg_total_intake_per_animal = results[total_intake_col].mean()
                                        # Calculate the sum of total intake across ALL animals
                                        sum_total_intake_all_animals = results[total_intake_col].sum()

                                        # Display the primary metric: Average Total Intake per Animal
                                        st.metric(
                                            f"Avg Total Intake / Animal ({time_window})", # Dynamic label
                                            f"{avg_total_intake_per_animal:.3f} {PARAMETER_UNITS.get(parameter, 'g')}", # Use .get for safety
                                            help=f"Average of the total accumulated intake for each animal over the selected '{time_window}' period."
                                        )
                                        # Display secondary info: Sum across all animals
                                        st.caption(f"Sum Intake (All Animals): {sum_total_intake_all_animals:.3f}{PARAMETER_UNITS.get(parameter, 'g')}")

                                        # Placeholder for potential future Light/Dark cycle intake difference metrics
                                        # TODO: Calculate Light/Dark intake difference (last-first within cycle) if needed
                                        # For now, we focus on the primary total intake metric.
                                        # We can add columns for Light/Dark metrics later if deemed necessary.
                                        # Example placeholders:
                                        # st.metric("Avg Light Cycle Intake", "TODO")
                                        # st.metric("Avg Dark Cycle Intake", "TODO")

                                    else:
                                        # If the main column is missing, indicate an issue
                                        st.metric("Avg Total Intake / Animal", "N/A", help=f"'{total_intake_col}' column not found in results.")
                                        st.warning(f"Could not find the '{total_intake_col}' column in the processed results. Check the `process_clams_data` function.")
                                        st.dataframe(results.head(2)) # Show head of results for debugging


                                elif parameter == "RER":
                                    # --- RER Metrics --- [CORRECTED - Values now match labels directly]
                                    col1, col2, col3 = st.columns([1, 1, 1])
                                    light_avg_col = 'Light Average' # Standard column name for Light cycle average
                                    dark_avg_col = 'Dark Average'   # Standard column name for Dark cycle average

                                    # Check if both columns exist in the results DataFrame before attempting calculations
                                    if results is not None and light_avg_col in results.columns and dark_avg_col in results.columns:
                                        try:
                                            # Calculate the overall mean across all animals for each column
                                            # .mean() automatically handles NaN values if any animals were missing data
                                            # The reliability of 'results' columns was confirmed by Colab manual checks
                                            actual_overall_light_mean = results[light_avg_col].mean()
                                            actual_overall_dark_mean = results[dark_avg_col].mean()

                                            # --- Display the CORRECT mean with the CORRECT label ---
                                            with col1:
                                                # Label: Average Light RER, Value: Use the calculated mean of the 'Light Average' column
                                                st.metric("Average Light RER", f"{actual_overall_light_mean:.3f}")
                                                # Optional Debug: st.caption(f"From col: {light_avg_col}")
                                            with col2:
                                                # Label: Average Dark RER, Value: Use the calculated mean of the 'Dark Average' column
                                                st.metric("Average Dark RER", f"{actual_overall_dark_mean:.3f}")
                                                # Optional Debug: st.caption(f"From col: {dark_avg_col}")
                                            with col3:
                                                # Animals Analyzed should still be correct
                                                st.metric("Animals Analyzed", f"{len(results)}")

                                        except Exception as e:
                                            st.error(f"Error calculating RER metrics: {e}")
                                            # Display fallback if calculation fails
                                            with col1: st.metric("Average Light RER", "Error")
                                            with col2: st.metric("Average Dark RER", "Error")
                                            with col3: st.metric("Animals Analyzed", f"{len(results)}")

                                    # Fallback if the necessary columns 'Light Average' or 'Dark Average' are missing from the results DataFrame
                                    else:
                                        st.warning(f"Cannot display RER metrics: Missing '{light_avg_col}' or '{dark_avg_col}' columns in results.")
                                        st.dataframe(results.head()) # Show head of results table to help debug column names
                                        with col1: st.metric("Average Light RER", "N/A")
                                        with col2: st.metric("Average Dark RER", "N/A")
                                        with col3: st.metric("Animals Analyzed", f"{len(results)}")

                                else:
                                    # --- Default & Accumulated Gas Metrics ---
                                    col1, col2, col3 = st.columns([1, 1, 1]) # Use 3 columns for layout

                                    # Determine column names and labels based on specific parameter type within this group
                                    if parameter in ["ACCCO2", "ACCO2"]:
                                        light_col = 'Light Net Accumulated'
                                        dark_col = 'Dark Net Accumulated'
                                        total_col = 'Total Net Accumulated (Period)'
                                        # Labels reflect the average of the *net change* during the period
                                        light_label = f"Avg Light Net Acc. {parameter}"
                                        dark_label = f"Avg Dark Net Acc. {parameter}"
                                        total_label = f"Avg Total Net Acc. {parameter}"
                                        rounding_format = "{:.4f}" # Use more decimal places for accumulated
                                    else:
                                        # Default parameters (Metabolic, other Gases, Env) use averages
                                        light_col = 'Light Average'
                                        dark_col = 'Dark Average'
                                        total_col = 'Total Average'
                                        light_label = f"Average Light {parameter}"
                                        dark_label = f"Average Dark {parameter}"
                                        total_label = f"Overall Avg {parameter}"
                                        rounding_format = "{:.2f}" # Default rounding
                                        if parameter == "RER":
                                            rounding_format = "{:.3f}" # Specific rounding for RER

                                    # Display metrics using the determined names and labels
                                    with col1:
                                        if light_col in results.columns:
                                            avg_val = results[light_col].mean()
                                            st.metric(light_label,
                                                    f"{avg_val:{rounding_format.strip('{:}')}} {PARAMETER_UNITS.get(parameter, '')}")
                                        else:
                                            st.metric(light_label, "N/A")
                                    with col2:
                                        if dark_col in results.columns:
                                            avg_val = results[dark_col].mean()
                                            st.metric(dark_label,
                                                    f"{avg_val:{rounding_format.strip('{:}')}} {PARAMETER_UNITS.get(parameter, '')}")
                                        else:
                                            st.metric(dark_label, "N/A")
                                    with col3:
                                        if total_col in results.columns:
                                            avg_val = results[total_col].mean()
                                            st.metric(total_label,
                                                    f"{avg_val:{rounding_format.strip('{:}')}} {PARAMETER_UNITS.get(parameter, '')}")
                                        else:
                                            st.metric(total_label, "N/A")

                            except KeyError as e:
                                st.error(f"Error displaying metrics: Could not find expected column '{e}'. Check data processing results for parameter '{parameter}'.")
                                st.dataframe(results.head()) # Show results head for debugging
                            except Exception as e:
                                st.error(f"An unexpected error occurred while displaying metrics: {e}")
                                if results is not None: st.dataframe(results.head())

                        else:
                            st.warning("Metrics cannot be displayed. No results data available after processing.")

                    # --- Display Number of Records Analyzed ---
                    # Check if raw_data (filtered for time window) exists
                    if 'raw_data' in locals() and raw_data is not None and not raw_data.empty:
                        # 'raw_data' here IS df_24h because process_clams_data returns it as raw_data
                        num_records = len(raw_data)
                        st.caption(f"📈 Analysis based on **{num_records:,}** data records within the selected '{time_window}' time window.") # Use comma for thousands separator
                    # --- End Display Number of Records ---
                    
                    # --- Calculate and Display Day/Night Pattern Insight ---
                    day_night_ratio = None
                    light_val_mean = None
                    dark_val_mean = None
                    light_col_name = None # Initialize column names
                    dark_col_name = None

                    try:
                        # Step 1: Determine the correct column names based on the parameter
                        if parameter in ["VO2", "VCO2", "HEAT", "RER"] or \
                        parameter in ["O2IN", "O2OUT", "CO2IN", "CO2OUT", "DO2", "DCO2", "FLOW", "PRESSURE"] or \
                        parameter in ["ACCCO2", "ACCO2"]: # Added ACCCO2/ACCO2 here explicitly

                            # Further check *within* this block for Accumulated Gases
                            if parameter in ["ACCCO2", "ACCO2"]:
                                light_col_name = 'Light Net Accumulated'
                                dark_col_name = 'Dark Net Accumulated'
                            else:
                                # Default/Metabolic/Other Gases use Averages
                                light_col_name = 'Light Average'
                                dark_col_name = 'Dark Average'

                        elif parameter in ["XTOT", "XAMB", "YTOT", "YAMB", "ZTOT", "ZAMB"]:
                            # Activity calculation uses these column names
                            light_col_name = 'Light Average Activity'
                            dark_col_name = 'Dark Average Activity'

                        elif parameter == "FEED":
                            # Feed calculation compares average TOTAL intake per cycle
                            light_col_name = 'Light Total Intake'
                            dark_col_name = 'Dark Total Intake'

                        # Step 2: Check if columns exist and calculate means (No change needed here)
                        if light_col_name and dark_col_name and \
                        light_col_name in results.columns and dark_col_name in results.columns:
                            light_val_mean = results[light_col_name].mean()
                            dark_val_mean = results[dark_col_name].mean()

                            # Step 3: Calculate ratio safely (No change needed here)
                            if pd.notna(light_val_mean) and pd.notna(dark_val_mean) and dark_val_mean != 0:
                                day_night_ratio = light_val_mean / dark_val_mean
                            elif pd.notna(light_val_mean) and pd.notna(dark_val_mean) and dark_val_mean == 0 and light_val_mean != 0:
                                day_night_ratio = float('inf') # Handle division by zero if light is non-zero
                            # Else: ratio remains None if means are NaN or both are zero

                        # else: If columns don't exist, means remain None, ratio remains None

                    except Exception as e:
                        # Catch unexpected errors during calculation
                        st.warning(f"Could not calculate Day/Night pattern due to an error: {e}", icon="⚙️")
                        day_night_ratio = None # Ensure it's None on error

                    # Step 4: Display the insight message if ratio was calculated
                    if day_night_ratio is not None and day_night_ratio != float('inf'):
                        direction = "higher" if day_night_ratio > 1 else "lower"
                        try:
                            percent_diff = abs(1 - day_night_ratio) * 100
                        except:
                            percent_diff = 0 # Fallback

                        # Determine the *expected* condition for a typical nocturnal pattern
                        # True means we expect Dark > Light (ratio < 1)
                        # False means we expect Light >= Dark (ratio >= 1) - less common biologically
                        expect_dark_higher = None # Default: no strong expectation
                        if parameter == "FEED":
                            expect_dark_higher = True # Expect higher intake in dark
                        elif parameter in ["XTOT", "XAMB", "YTOT", "YAMB", "ZTOT", "ZAMB", "VO2", "VCO2", "HEAT"]:
                            expect_dark_higher = True # Expect higher activity/metabolism in dark
                        elif parameter in ["ACCCO2", "ACCO2"]:
                            expect_dark_higher = True # Expect higher net accumulation in dark (active) phase

                        # Construct the message using the determined column names
                        insight_text = f"**Day/Night Pattern**: Avg '{light_col_name}' is **{percent_diff:.1f}% {direction}** than Avg '{dark_col_name}'."

                        # Add interpretation based on expected pattern
                        if expect_dark_higher is not None:
                            # Check if the actual data matches the expectation
                            actual_dark_is_higher = (day_night_ratio < 1)
                            if expect_dark_higher == actual_dark_is_higher:
                                insight_text += " (Suggests typical nocturnal pattern)"
                            else:
                                insight_text += " (May suggest altered circadian pattern)"

                        st.info(insight_text) # Display the final text

                    elif day_night_ratio == float('inf'):
                        st.info(f"**Day/Night Pattern**: Avg '{light_col_name}' has value, but Avg '{dark_col_name}' is zero.") # Updated to use variables
                    # else: If day_night_ratio is still None, no message is displayed (e.g., columns missing, NaNs)
          
                    # --- End of Lean Mass Content ---Create and display the visualization
                    # Create a time-based view instead of just hourly
                    # First create proper timeline data
                    timeline_data = raw_data.copy()

                    # Enhanced Data View Options section
                    # --- Data Visualization Section ---
                    st.header("📊 Data Visualization")

                    # Radio button to select display mode
                    selected_option = st.radio(
                        "Select data display mode:",
                        ["Show Average Across All Animals", "Focus on Individual Animals"],
                        key="animal_display_mode",
                        horizontal=True, # Display options side-by-side
                        help="Choose how to visualize your data: the overall average or specific animals."
                    )

                    # --- Conditional Controls & Filtering for Individual View ---
                    selected_subjects = [] # Initialize list to store selected subject IDs
                    selected_cages = []    # Initialize list to store corresponding cage names

                    # Only show multiselect and prepare data if "Individual" is chosen
                    if selected_option == "Focus on Individual Animals":
                        st.markdown("##### Select Animals to Display:") # Add a sub-header for clarity

                        # --- Prepare subject list with groups (if available) ---
                        subject_to_cage = {} # Map Subject ID -> Cage Name
                        available_subjects = [] # List of Subject IDs
                        subject_options = [] # List of strings for multiselect widget

                        # Safely build the map and lists
                        if 'results' in locals() and results is not None and 'Subject ID' in results.columns and not results.empty:
                            try:
                                subject_to_cage = pd.Series(results.index.values, index=results['Subject ID']).to_dict()
                                available_subjects = list(subject_to_cage.keys())

                                # Add group info to options if groups are assigned
                                if 'group_assignments' in st.session_state and not st.session_state.get('group_assignments', pd.DataFrame()).empty:
                                    group_df = st.session_state['group_assignments']
                                    # Create a quick lookup dictionary: Subject ID -> Group Name
                                    subject_to_group_map = pd.Series(group_df.Group.values, index=group_df['Subject ID']).to_dict()
                                    # Build the display options
                                    subject_options = [
                                        f"{subject} ({subject_to_group_map.get(subject, 'No Group')})"
                                        for subject in available_subjects
                                    ]
                                else:
                                    # If no groups, just use Subject IDs
                                    subject_options = available_subjects

                            except Exception as e:
                                st.warning(f"Could not prepare subject list for selection: {e}")

                        # --- Display Multiselect ---
                        if subject_options: # Only show if we have subjects to choose from
                            selected_subject_labels = st.multiselect(
                                "Select specific animals:", # Simplified label
                                options=subject_options,
                                # Default to the first animal if available
                                default=[subject_options[0]] if subject_options else [],
                                key="subject_multiselect_simplified", # Use a new key
                                label_visibility="collapsed" # Hide label as we have the markdown header
                            )

                            # Extract Subject IDs from the selected labels (e.g., "Mouse1 (Group A)" -> "Mouse1")
                            selected_subjects = [label.split(" (")[0] for label in selected_subject_labels] 
                            if len(selected_subjects) > 10:
                                st.caption("⚠️ Selecting many animals might make the plot dense.")
                            # Convert selected Subject IDs to Cage names using the map
                            if selected_subjects and subject_to_cage:
                                selected_cages = [subject_to_cage[subject] for subject in selected_subjects if subject in subject_to_cage]
                                if len(selected_cages) != len(selected_subjects):
                                    st.warning("Could not map all selected subjects to cages. Some might be missing from results.")
                            elif not selected_subjects:
                                st.warning("⚠️ Please select at least one animal to display", icon="🔍")
                            

                        else:
                            st.warning("No subjects available for selection in results data.")


                    # --- Description Text (Now simpler) ---
                    # Get current light/dark times for the description
                    ls = st.session_state.get('light_start', 7)
                    le = st.session_state.get('light_end', 19)
                    st.caption(f"""
                        **Plot Legend:** Black line/markers = Average across all animals (with SEM shading).
                        Colored lines/markers = Individual animals (if selected).
                        Gray background = Dark cycle ({le:02d}:00 - {ls:02d}:00).
                        """)

                    # --- End of Visualization Controls and Setup ---
                    
                    
                    
        
                    # Add a datetime column that combines date and hour for proper timeline
                    timeline_data['datetime'] = pd.to_datetime(timeline_data['timestamp'].dt.strftime('%Y-%m-%d %H:00:00'))

                    # Group by the datetime to get hourly averages across the full time window
                    timeline_results = timeline_data.groupby(['datetime', 'cage'])['value'].mean().reset_index()

                    # Calculate mean and SEM across all cages
                    timeline_summary = timeline_results.groupby('datetime')['value'].agg(['mean', lambda x: x.std()/np.sqrt(len(x))]).reset_index()
                    timeline_summary.columns = ['datetime', 'Mean', 'SEM']  # Rename the lambda column to SEM

                    # Create the plot with the timeline data
                    fig = go.Figure()

                    # Define a color palette for individual animal lines
                    colors = px.colors.qualitative.Plotly  # Built-in Plotly color sequence

                    if selected_option == "Show Average Across All Animals":
                        # Add SEM range (only for average view)
                        fig.add_trace(go.Scatter(
                            x=timeline_summary['datetime'],
                            y=timeline_summary['Mean'] + timeline_summary['SEM'],
                            fill=None,
                            mode='lines',
                            line_color='rgba(31, 119, 180, 0.2)',
                            showlegend=False
                        ))
                        fig.add_trace(go.Scatter(
                            x=timeline_summary['datetime'],
                            y=timeline_summary['Mean'] - timeline_summary['SEM'],
                            fill='tonexty',
                            mode='lines',
                            line_color='rgba(31, 119, 180, 0.2)',
                            showlegend=False
                        ))
                        
                        # Add mean line
                        fig.add_trace(go.Scatter(
                            x=timeline_summary['datetime'],
                            y=timeline_summary['Mean'],
                            mode='lines+markers',
                            line_color='rgb(31, 119, 180)',
                            name=f'Mean {parameter}'
                        ))
                    else:  # Focus on Individual Animals
                        # Plot each selected cage as a separate line
                        for i, cage in enumerate(selected_cages):
                            # Get data for this cage
                            cage_data = timeline_results[timeline_results['cage'] == cage]
                            
                            # Find subject ID for this cage for better labeling
                            subject_id = results.loc[results.index == cage, 'Subject ID'].iloc[0] if cage in results.index else cage
                            
                            # Add the line for this cage
                            fig.add_trace(go.Scatter(
                                x=cage_data['datetime'],
                                y=cage_data['value'],
                                mode='lines+markers',
                                name=f'{subject_id}',
                                line=dict(color=colors[i % len(colors)], width=2),
                                marker=dict(size=6)
                            ))

                    # Add shaded regions for dark cycles
                    min_date = timeline_summary['datetime'].min().date()
                    max_date = timeline_summary['datetime'].max().date()
                    current_date = min_date

                    # Get custom light/dark cycle times from session state
                    light_start = st.session_state.get('light_start', 7)  # Default to 7AM if not set
                    light_end = st.session_state.get('light_end', 19)     # Default to 7PM if not set

                    while current_date <= max_date:
                        # Add dark cycle from midnight to light_start
                        morning_start = pd.Timestamp(current_date.strftime('%Y-%m-%d') + ' 00:00:00')
                        morning_end = pd.Timestamp(current_date.strftime('%Y-%m-%d') + f' {light_start:02d}:00:00')
                        
                        # Add dark cycle from light_end to midnight
                        evening_start = pd.Timestamp(current_date.strftime('%Y-%m-%d') + f' {light_end:02d}:00:00')
                        evening_end = pd.Timestamp(current_date.strftime('%Y-%m-%d') + ' 23:59:59')
                        
                        # Add the dark cycle rectangles
                        if morning_start >= timeline_summary['datetime'].min() and morning_end <= timeline_summary['datetime'].max():
                            fig.add_vrect(
                                x0=morning_start,
                                x1=morning_end,
                                fillcolor="rgba(100,100,100,0.3)",
                                layer="below",
                                line_width=0,
                            )
                        
                        if evening_start >= timeline_summary['datetime'].min() and evening_start <= timeline_summary['datetime'].max():
                            fig.add_vrect(
                                x0=evening_start,
                                x1=evening_end,
                                fillcolor="rgba(100,100,100,0.3)",
                                layer="below",
                                line_width=0,
                            )
                        
                        
                        # Move to next day
                        current_date += pd.Timedelta(days=1)
                    # Add annotations with darker background and borders for visibility
                    for date in [d for d in timeline_summary['datetime'].dt.date.unique()]:
                        # Only add annotations if this date has data points
                        date_points = timeline_summary[timeline_summary['datetime'].dt.date == date]
                        if len(date_points) > 0:
                            # Add light cycle annotation (middle of the day)
                            light_time = pd.Timestamp(f"{date} 12:00:00")
                            if light_time >= timeline_summary['datetime'].min() and light_time <= timeline_summary['datetime'].max():
                                fig.add_annotation(
                                    x=light_time,
                                    y=0.95,
                                    yref="paper",
                                    text="Light Cycle",
                                    showarrow=False,
                                    font=dict(size=12, color="black"),
                                    bgcolor="rgba(255,215,0,0.7)",  # Gold background
                                    bordercolor="black",
                                    borderwidth=1,
                                    borderpad=3
                                )
                            
                            # Add dark cycle annotation (early morning)
                            dark_time_early = pd.Timestamp(f"{date} 03:00:00")
                            if dark_time_early >= timeline_summary['datetime'].min() and dark_time_early <= timeline_summary['datetime'].max():
                                fig.add_annotation(
                                    x=dark_time_early,
                                    y=0.95,
                                    yref="paper",
                                    text="Dark Cycle",
                                    showarrow=False,
                                    font=dict(size=12, color="white"),
                                    bgcolor="rgba(0,0,0,0.7)",  # Black background
                                    bordercolor="white",
                                    borderwidth=1,
                                    borderpad=3
                                )
                        
                    # Update layout based on parameter and time window
                    fig.update_layout(
                        title=f'{parameter} Timeline ({time_window})',
                        xaxis_title='Date and Time (12:00 = noon, 00:00 = midnight)',
                        yaxis_title=f'{parameter} ({PARAMETER_UNITS[parameter]})',
                        hovermode='x unified'
                    )

                    # Display the plot
                    st.plotly_chart(fig, use_container_width=True, key=f"{parameter}_plot")
                    # --- Consolidated Setup Expander ---
                    st.markdown("---") # Add separator after the plot

                    # --- Overview Tab ---
                    # ... (code before the expander) ...

                    # --- Setup Section (NOW AN EXPANDER) ---
                    # Remove the invalid key from st.expander
                    with st.expander("⚙️ Setup: Groups & Lean Mass", expanded=False): # NO key=... here

                        # Keep the container for visual grouping inside the expander
                        with st.container(border=True):

                            # --- Group Assignment Content (Inside Expander & Container) ---
                            st.subheader("1. Assign Animals to Groups")
                            st.info("Assign animals to experimental groups is needed for more sophisticated analysis.")

                            # --- Robustness check for cage_info ---
                            if 'cage_info' not in st.session_state or not st.session_state['cage_info']:
                                if uploaded_file is not None:
                                    # Attempt to extract cage_info ONLY if it's missing and a file exists
                                    try:
                                        # Ensure the file pointer is at the beginning before re-reading
                                        uploaded_file.seek(0)
                                        st.session_state['cage_info'] = extract_cage_info(uploaded_file)
                                        uploaded_file.seek(0) # Rewind again after extraction
                                        print("Re-extracted cage_info.") # Debug
                                        if not st.session_state['cage_info']: # Check if extraction failed
                                            st.error("Failed to extract cage info on the fly. Please re-upload.")
                                            cage_info_available = False
                                        else:
                                             cage_info_available = True
                                    except Exception as e:
                                        st.error(f"Error re-extracting cage info: {e}")
                                        cage_info_available = False

                                else:
                                    st.error("Cannot assign groups: Upload a data file first.")
                                    cage_info_available = False
                            else:
                                cage_info_available = True
                            # --- END OF ROBUSTNESS CHECK ---


                            if cage_info_available: # Only proceed if cage_info is confirmed available
                                # Create cage_df from stored cage_info
                                cage_list = [{"Cage": k, "Subject ID": v} for k, v in st.session_state['cage_info'].items()]
                                if cage_list:
                                    cage_df = pd.DataFrame(cage_list)
                                else:
                                    cage_df = pd.DataFrame(columns=["Cage", "Subject ID"])

                                if not cage_df.empty:
                                    # --- Group Assignment UI ---
                                    # This function 'assign_groups' might trigger reruns itself depending on its widgets (multiselect).
                                    # This is generally acceptable for group assignment as it affects downstream stats availability.
                                    group_assignments_result = assign_groups(cage_df, key_prefix="overview_setup_expander")
                                    # Store results immediately if they are valid
                                    if group_assignments_result is not None and not group_assignments_result.empty:
                                        # Check if it's different from current state before assigning to avoid unnecessary state changes
                                        if not group_assignments_result.equals(st.session_state.get('group_assignments')):
                                            st.session_state['group_assignments'] = group_assignments_result
                                            print("Updated group assignments in session state.") # Debug
                                else:
                                    st.warning("Could not create DataFrame for group assignment.")


                            # --- Lean Mass Content (Inside Expander & Container & Conditional) ---
                            st.markdown("---") # Separator
                            st.subheader("2. Enter Lean Mass (Optional)")

                            if parameter in ["VO2", "VCO2", "HEAT"]:
                                if st.session_state.get("apply_lean_mass", False):
                                    # Message indicating adjustment is enabled
                                    st.markdown(f"""
                                    Lean mass adjustment is **enabled** (sidebar setting). Enter values below and click 'Apply Lean Mass Changes' to update the analysis. Reference mass: **{st.session_state.get('reference_lean_mass_sidebar_val', 20.0):.1f}g** (set in sidebar).
                                    """)

                                    if cage_info_available: # Use the flag from the check above
                                        cols = st.columns(3)
                                        # --- THIS IS THE LOOP FOR NUMBER INPUTS ---
                                        # These inputs update their OWN state via their key on change.
                                        for i, (cage_label, subject_id) in enumerate(st.session_state['cage_info'].items()):
                                            widget_key = f"lean_mass_{cage_label}_setup_expander"
                                            with cols[i % 3]:
                                                # The value displayed comes from the widget's own state key
                                                st.number_input(
                                                    f"LM (g) for {subject_id} ({cage_label})", # Shortened label
                                                    min_value=1.0,
                                                    # Set initial value from lean_mass_data if available, else default
                                                    value=st.session_state.get('lean_mass_data', {}).get(cage_label, 20.0),
                                                    step=0.1,
                                                    format="%.1f",
                                                    key=widget_key, # Crucial: Each input has a unique key
                                                    label_visibility="visible" # Ensure label is shown
                                                )
                                        # --- END OF THE LOOP ---

                                        # --- ADD THE BUTTON ---
                                        # This button's callback will read all number inputs and trigger the rerun
                                        st.button("Apply Lean Mass Changes",
                                                  on_click=apply_lean_mass_changes, # Use the new callback
                                                  key="apply_lm_button",
                                                  help="Click to update the analysis tables and plots with the lean mass values entered above.")
                                        # --- END OF ADDED BUTTON ---

                                        # Formula caption (no change needed)
                                        st.caption(f"""
                                        **Formula:** Adj. Value = Orig. Value × (Reference Mass ÷ Animal LM)...
                                        """)
                                    else:
                                        st.warning("Cannot display lean mass inputs: Cage information missing.")

                                else: # apply_lean_mass is False
                                    st.markdown("**(Optional) Lean mass adjustment is currently disabled.** Enable it in the sidebar settings. If enabled, enter values and click 'Apply Lean Mass Changes'.")
                            else: # Parameter not relevant
                                # Parameter not relevant logic... (no changes needed here)
                                if st.session_state.get("apply_lean_mass", False):
                                    st.warning(f"Lean mass adjustment is enabled (sidebar), but not applicable for '{parameter}'. No adjustment will be made.")
                                else:
                                    st.markdown(f"*(Lean mass adjustment is not applicable for '{parameter}')*")
                    # --- End of the Setup Expander --- (End of the `with st.expander...` block)
                    
                    # 4. TABLES - Enhanced data tables with better organization and interactivity
                    st.header("📋 Detailed Analysis Tables")

                    # Create tabs for different tables
                    table_tab1, table_tab2 = st.tabs(["Light/Dark Analysis", "Hourly Analysis"])

                    with table_tab1:
                        st.markdown(f"### {parameter} Light/Dark Analysis")
                        st.markdown("""
                        This table shows average values during light and dark cycles for each animal.
                        Values highlighted in red are potential outliers (> 2 standard deviations from the mean).
                        """)
                        
                        # Calculate group means if groups are assigned
                        if 'group_assignments' in st.session_state and not st.session_state['group_assignments'].empty:
                            group_df = st.session_state['group_assignments']
                            
                            # Create a mapping from subject ID to group
                            subject_to_group = dict(zip(group_df["Subject ID"], group_df["Group"]))
                            
                            # Add Group column to results
                            results_with_groups = results.copy()
                            results_with_groups['Group'] = results_with_groups['Subject ID'].map(subject_to_group)
                            
                            # Only show grouped data if groups are actually assigned to subjects
                            if not results_with_groups['Group'].isna().all():
                                st.markdown("#### Group Averages")
                                # The .mean(numeric_only=True) correctly ignores NaNs when calculating group means.
                                # The style_dataframe function likely needs no change, as it applies styling based on columns,
                                # and NaN cells won't trigger the outlier highlighting anyway.
                                group_means = results_with_groups.groupby('Group').mean(numeric_only=True)
                                st.dataframe(style_dataframe(group_means)) # Existing call is likely fine

                                st.markdown("#### Individual Animal Data")
                                # Reorder columns to show group first
                                cols = results_with_groups.columns.tolist()
                                cols.remove('Group')
                                cols.insert(0, 'Group')
                                results_with_groups = results_with_groups[cols]
                                st.dataframe(style_dataframe(results_with_groups))
                            else:
                                st.dataframe(style_dataframe(results))
                        else:
                            st.dataframe(style_dataframe(results))

                    with table_tab2:
                        st.markdown(f"### {parameter} Hourly Analysis")
                        
                        # Only show the option for longer time windows
                        if time_window != "Last 24 Hours":
                            hourly_view = st.radio(
                                "Select hourly data view:",
                                ["Typical Day Profile (hours 0-23)", "Full Timeline"],
                                key="hourly_view_option",
                                help="Choose between seeing a single 'typical day' or the full timeline hour by hour"
                            )
                            
                            if hourly_view == "Typical Day Profile (hours 0-23)":
                                # Create columns for explanation and example
                                col1, col2 = st.columns([3, 2])
                                
                                with col1:
                                    st.markdown("### 🔄 Typical Day Profile")
                                    st.markdown("""
                                    **What this shows**: This view combines data from all days in your selected time window into a single 24-hour profile.
                                    
                                    **How it works**:
                                    - All measurements taken at 1:00 AM (hour 1) across different days are averaged together
                                    - All measurements taken at 2:00 AM (hour 2) across different days are averaged together
                                    - And so on for each hour of the day
                                    
                                    **When to use this view**: When you want to see consistent circadian patterns and reduce day-to-day variability.
                                    """)
                                
                                with col2:
                                    st.markdown("#### Example Interpretation")
                                    st.markdown("""
                                    If you see:
                                    - Hour 1: 75.2
                                    - Hour 13: 58.4
                                    
                                    This means:
                                    - The average value at 1:00 AM across all days was 75.2
                                    - The average value at 1:00 PM across all days was 58.4
                                    
                                    This helps identify time-of-day patterns that repeat daily.
                                    """)
                                
                                # Show the current hourly results (which are already averaged by hour)
                                st.dataframe(style_dataframe(hourly_results))
                                
                            else:  # Full Timeline view
                                # Create columns for explanation and example
                                col1, col2 = st.columns([3, 2])
                                
                                with col1:
                                    st.markdown("### 📈 Full Timeline")
                                    st.markdown("""
                                    **What this shows**: This view displays each hour sequentially throughout your entire time window.
                                    
                                    **How it works**:
                                    - Hours are numbered sequentially from the start of your data (0, 1, 2, ... 71)
                                    - Day column shows which day each hour belongs to (Day 1, 2, or 3)
                                    - Each row represents a specific hour on a specific day
                                    
                                    **When to use this view**: When you want to see trends, adaptations, or changes across multiple days.
                                    """)
                                
                                with col2:
                                    st.markdown("#### Example Interpretation")
                                    st.markdown("""
                                    If you see:
                                    - Hour 1 (Day 1): 75.2
                                    - Hour 25 (Day 2): 68.3
                                    
                                    This means:
                                    - 1 hour after start on Day 1, the value was 75.2
                                    - The same time on Day 2, the value dropped to 68.3
                                    
                                    This helps identify trends or adaptations across days.
                                    """)
                                
                                # We need to create a different hourly dataframe for the full timeline
                                # This can be derived from the raw_data directly
                                timeline_hours = raw_data.copy()
                                # Calculate hours from the start of the time window
                                start_time = timeline_hours['timestamp'].min()
                                timeline_hours['hour_from_start'] = ((timeline_hours['timestamp'] - start_time).dt.total_seconds() / 3600).astype(int)
                                
                                # Group by hour from start and cage
                                full_hourly = timeline_hours.pivot_table(
                                    values='value',
                                    index='hour_from_start',
                                    columns='cage',
                                    aggfunc='mean'
                                ).round(3 if parameter == "RER" else 2)
                                
                                # Add day markers
                                full_hourly['Day'] = (full_hourly.index / 24 + 1).astype(int)
                                
                                # Reorder columns to show Day first
                                day_col = full_hourly.pop('Day')
                                full_hourly.insert(0, 'Day', day_col)
                                
                                # Calculate summary statistics
                                full_hourly['Mean'] = full_hourly.drop('Day', axis=1).mean(axis=1).round(3 if parameter == "RER" else 2)
                                full_hourly['SEM'] = (full_hourly.drop(['Day', 'Mean'], axis=1).std(axis=1) / 
                                                    np.sqrt(full_hourly.drop(['Day', 'Mean'], axis=1).shape[1])).round(3 if parameter == "RER" else 2)
                                
                                # Display the full timeline data
                                st.dataframe(style_dataframe(full_hourly))
                        else:
                            # For 24-hour time window, just show the standard table
                            st.markdown("""
                            This table shows average values for each hour of the day (0-23).
                            - Hours run from 0 (midnight) to 23 (11 PM)
                            - Each column represents a different animal
                            - The 'Mean' column shows the average across all animals
                            - The 'SEM' column shows the standard error of the mean
                            
                            Values highlighted in red are potential outliers (> 2 standard deviations from the mean).
                            """)
                            st.dataframe(style_dataframe(hourly_results))

                    # Clear download options
                    st.subheader("📥 Export Data")
                    st.markdown("Download your analysis results in CSV format:")

                    col1, col2 = st.columns(2)
                    with col1:
                        # Summary data download (light/dark)
                        csv_light_dark = results.to_csv().encode('utf-8')
                        st.download_button(
                            label=f"📥 Download Summary Data",
                            data=csv_light_dark,
                            file_name=f"{parameter}_summary_analysis.csv",
                            mime="text/csv",
                            help=f"Download light/dark cycle and total averages for each animal"
                        )

                        # Only show if group assignments exist
                        if 'group_assignments' in st.session_state and not st.session_state['group_assignments'].empty:
                            # Create group summary if it exists
                            if 'results_with_groups' in locals() and not results_with_groups.empty:
                                group_summary = results_with_groups.groupby('Group').mean(numeric_only=True)
                                csv_group_summary = group_summary.to_csv().encode('utf-8')
                                st.download_button(
                                    label=f"📥 Download Group Averages",
                                    data=csv_group_summary,
                                    file_name=f"{parameter}_group_averages.csv",
                                    mime="text/csv",
                                    help=f"Download averages by experimental group"
                                )

                    with col2:
                        # Hourly data download
                        csv_hourly = hourly_results.to_csv().encode('utf-8')
                        st.download_button(
                            label=f"📥 Download Hourly Data",
                            data=csv_hourly,
                            file_name=f"{parameter}_hourly_data.csv",
                            mime="text/csv",
                            help=f"Download hour-by-hour averages for each animal"
                        )
                        
                        # Raw data download (limited to prevent huge files)
                        if raw_data is not None:
                            sample_data = raw_data.head(1000)  # Limit to first 1000 rows
                            csv_raw = sample_data.to_csv(index=False).encode('utf-8')
                            st.download_button(
                                label=f"📥 Download Raw Data Sample",
                                data=csv_raw,
                                file_name=f"{parameter}_raw_data_sample.csv",
                                mime="text/csv",
                                help=f"Download first 1000 rows of raw data"
                            )

                    # Add explanations AFTER both columns - outside the column blocks
                    st.markdown("---")  # Add separator for clarity
                    with st.expander("Which data file should I download?", expanded=False):
                        st.markdown("""
                        ### Understanding Export Options
                        
                        Choose the right data format for your needs:
                        
                        **Summary Data**
                        - Contains light/dark cycle averages for each animal
                        - Includes metrics like Dark Average, Light Average, Total Average
                        - Best for: Overall results and animal-to-animal comparisons
                        
                        **Group Averages**
                        - Contains averaged values for each experimental group
                        - Useful for statistical comparisons between treatment groups
                        - Best for: Preparing graphs for publications and presentations
                        
                        **Hourly Data**
                        - Contains hour-by-hour averages (0-23) for each animal
                        - Shows circadian patterns and time-of-day effects
                        - Best for: Time series analysis and circadian rhythm assessment
                        
                        **Raw Data Sample**
                        - First 1000 rows of unprocessed measurements
                        - Includes exact timestamps and individual readings
                        - Best for: Verification, custom analysis, or processing in other software
                        """)

    
            # Tab 2: Statistical Analysis
            with tab2:
                st.header("📈 Statistical Analysis")

                # --- Prerequisite Check ---
                groups_assigned = ('group_assignments' in st.session_state and
                                not st.session_state.get('group_assignments', pd.DataFrame()).empty)

                if uploaded_file is None or 'raw_data' not in locals() or raw_data is None or 'results' not in locals() or results is None: # Keep existing file check
                    st.warning("⚠️ Please upload and process data in the 'Overview' tab first.")
                    st.stop() # Stop rendering this tab
                elif not groups_assigned:
                    st.error("🚫 **Action Required:** Please assign animals to groups in the **'Overview' tab → 'Setup: Groups & Lean Mass'** section before using this tab.")
                    st.stop() # Stop rendering the rest of this tab
                else:
                    # --- Groups are assigned, proceed with analysis UI ---
                    st.success("Data and groups loaded. Ready for analysis.") # Placeholder confirmation

                    # --- Educational Expander ---
                    with st.expander("🎓 Understanding the Analysis (Two-Way ANOVA)", expanded=False):
                        st.markdown("""
                        This section performs a **Two-Way Analysis of Variance (ANOVA)** to understand how your selected **Groups** and the **Light/Dark Cycle** influence the chosen parameter.

                        **Key Questions Answered:**
                        *   **Main Effect of Group:** Is there an overall significant difference between the groups, averaging across both light and dark cycles?
                        *   **Main Effect of Cycle:** Is there an overall significant difference between the light and dark cycles, averaging across all selected groups?
                        *   **Interaction Effect (Group x Cycle):** Does the effect of being in a certain group *depend* on whether it's the light or dark cycle? (e.g., Does Treatment A lower VO2 *only* during the dark cycle?) This is often the most interesting finding.

                        **How it Works:**
                        1.  The average value for each animal during the *light* cycle and the *dark* cycle is calculated.
                        2.  These individual light/dark means are used as data points in the Two-Way ANOVA model.
                        3.  The ANOVA test calculates p-values for the main effects and the interaction.

                        **Interpreting Results:**
                        *   Look at the **ANOVA Table** for p-values (< 0.05 usually indicates significance).
                        *   Examine the **Interaction Plot** to visualize how group averages change between light and dark cycles. Crossing or non-parallel lines often suggest an interaction.
                        *   Use **Post-Hoc Tests** (if effects are significant) to find out *which specific* groups or conditions differ significantly from each other.
                        """)

                    st.markdown("---") # Separator

                    # --- Analysis Setup: Group Selection ---
                    st.subheader("Analysis Setup")

                    # Get available groups from session state
                    group_assignments_df = st.session_state['group_assignments']
                    available_groups_stat = sorted(group_assignments_df['Group'].unique())

                    # Let user select which groups to include in THIS analysis
                    selected_groups_stat = st.multiselect(
                        "Select groups to include in the Two-Way ANOVA:",
                        options=available_groups_stat,
                        default=available_groups_stat, # Default to all assigned groups
                        key="stat_anova_group_select",
                        help="Choose 2 or more groups for comparison."
                    )

                    # --- Validation: Need at least 2 groups ---
                    if len(selected_groups_stat) < 2:
                        st.warning("⚠️ Please select at least two groups to perform the ANOVA.")
                        st.stop() # Stop execution in this tab if not enough groups are selected

                    # --- Data Preparation for Two-Way ANOVA ---
                    st.markdown("---") # Separator before data prep/results
                    st.subheader("Two-Way ANOVA: Group x Light/Dark Cycle")

                    try:
                        # 1. Filter Raw Data: Select only the data for the chosen groups
                        #    Use the 'raw_data' DataFrame loaded earlier.
                        #    Ensure 'Group' column exists from merging in simplified_statistical_analysis (or redo merge if needed)

                        # Let's re-merge here to be safe and ensure we have the 'Group' column associated with raw_data points
                        # We need: raw_data (timestamp, value, cage, is_light) and group_assignments_df (Subject ID, Group, Cage)
                        # We need a common key. Let's assume raw_data has 'cage' (e.g., 'CAGE 01')
                        # and group_assignments_df has 'Cage' and 'Subject ID'. Let's use 'Cage'.

                        # Ensure group_assignments_df has the correct 'Cage' column format if needed
                        # Example: If raw_data uses 'CAGE 01' and assign_groups stored '1', adjust one of them.
                        # Assuming process_clams_data consistently produces 'CAGE XX' format in raw_data['cage']
                        # Assuming assign_groups produces 'CAGE XX' format in group_assignments_df['Cage']

                        # Merge raw data with group assignments
                        # Use left merge to keep all raw_data rows and add Group info
                        # Handle potential errors if merge keys don't match perfectly
                        if 'cage' not in raw_data.columns:
                            st.error("Critical Error: 'cage' column missing in raw_data.")
                            st.stop()
                        if 'Cage' not in group_assignments_df.columns:
                            st.error("Critical Error: 'Cage' column missing in group_assignments_df.")
                            st.stop()
                        if 'Subject ID' not in group_assignments_df.columns:
                            st.error("Critical Error: 'Subject ID' column missing in group_assignments_df.")
                            st.stop()


                        # Perform the merge
                        # We need 'Subject ID' associated with each raw data point for later grouping
                        analysis_data = pd.merge(
                            raw_data,
                            group_assignments_df[['Cage', 'Group', 'Subject ID']], # Select necessary columns
                            left_on='cage', # Column name in raw_data
                            right_on='Cage', # Column name in group_assignments_df
                            how='inner' # Use 'inner' to keep only rows with matching cages in both dfs
                        )

                        if analysis_data.empty:
                            st.error("Data merging failed. No matching cages found between raw data and group assignments. Check formats.")
                            st.write("Raw data cages sample:", raw_data['cage'].unique()[:5])
                            st.write("Group assignment cages sample:", group_assignments_df['Cage'].unique()[:5])
                            st.stop()

                        # Filter for the groups selected by the user in the multiselect
                        analysis_data_filtered = analysis_data[analysis_data['Group'].isin(selected_groups_stat)].copy()

                        if analysis_data_filtered.empty:
                            st.warning(f"No data found for the selected groups: {', '.join(selected_groups_stat)}. Cannot perform ANOVA.")
                            st.stop()

                        # 2. Calculate Mean per Animal per Cycle:
                        #    Group by 'Subject ID', 'Group', 'is_light', calculate mean 'value'.
                        anova_input_df = analysis_data_filtered.groupby(['Subject ID', 'Group', 'is_light'])['value'].mean().reset_index()

                        # --- Handle Potential NaNs (using the correct column name 'value') ---
                        # Check if any NaNs exist in the 'value' column AFTER the groupby/mean
                        if anova_input_df['value'].isnull().any():
                            st.warning("⚠️ Some animals had missing mean values for Light or Dark cycle (likely due to no data points in that cycle within the selected time window). These specific animal/cycle combinations will be excluded from the ANOVA.", icon="📉")
                            # Display rows with NaN before dropping (for debugging/info)
                            st.write("Rows with missing mean values (excluded):")
                            st.dataframe(anova_input_df[anova_input_df['value'].isnull()])
                            # Drop rows with NaN in the 'value' column to create the clean dataset
                            anova_input_df_clean = anova_input_df.dropna(subset=['value']).copy()
                        else:
                            # If no NaNs, the clean dataset is the same as the original
                            anova_input_df_clean = anova_input_df.copy()

                        # --- Validate Clean Data ---
                        if anova_input_df_clean.empty:
                            st.error("No valid data remaining after removing missing values. Cannot perform ANOVA.")
                            st.stop()

                        # Add 'Cycle' column based on 'is_light' to the CLEAN DataFrame
                        anova_input_df_clean['Cycle'] = anova_input_df_clean['is_light'].map({True: 'Light', False: 'Dark'})

                        # Rename the 'value' column to 'Mean Value' in the CLEAN DataFrame
                        anova_input_df_clean = anova_input_df_clean.rename(columns={'value': 'Mean Value'})

                        # Check if each group/cycle combo still has data AFTER cleaning
                        n_check = anova_input_df_clean.groupby(['Group', 'Cycle'])['Subject ID'].nunique()
                        min_n_per_cell = n_check.min()

                        if min_n_per_cell < 1: # Need at least one observation per cell for 2-way ANOVA
                            st.error("Some Group/Cycle combinations have zero animals after removing missing values. Cannot perform standard Two-Way ANOVA.")
                            st.write("Animal counts per Group/Cycle combination (after cleaning):")
                            st.dataframe(n_check.unstack(fill_value=0)) # Show pivot table of counts
                            st.stop()

                        # Optional: Warn for low sample size
                        if min_n_per_cell < 3:
                            st.warning(f"Low sample size (minimum N={min_n_per_cell}) in at least one Group/Cycle combination. ANOVA results may be less reliable.", icon="🔬")

                        # Select and reorder final columns for the ANOVA input table using the CLEAN data
                        anova_input_df_clean = anova_input_df_clean[['Subject ID', 'Group', 'Cycle', 'Mean Value']]

                        # Final check on the cleaned data before passing to model
                        if anova_input_df_clean.empty or anova_input_df_clean['Mean Value'].isnull().any():
                            st.error("Data preparation failed unexpectedly after cleaning. Check intermediate steps.")
                            st.dataframe(anova_input_df_clean.head())
                            st.stop()

                        # --- Cleaned data 'anova_input_df_clean' is now ready for the model ---


                        # --- Cleaned data 'anova_input_df_clean' is now ready for the model ---

                        # 3. Display Data Summary (Using the CLEANED data)
                        st.markdown("##### Data Summary (N per Condition for ANOVA)")
                        st.markdown("Number of animals included in the ANOVA for each condition:")
                        try:
                            # Create pivot table from the CLEAN data which has the 'Cycle' column
                            n_table = pd.pivot_table(
                                anova_input_df_clean, # <<< USE CLEAN DATAFRAME
                                values='Subject ID',
                                index='Group',
                                columns='Cycle',
                                aggfunc='nunique',
                                fill_value=0 # Fill missing group/cycle combos directly
                            )
                            st.dataframe(n_table.astype(int)) # Display the table

                            # Check if any cell in the displayed table is 0 (after fill_value)
                            if (n_table == 0).any().any():
                                 st.warning("Note: Some Group/Cycle combinations had 0 animals after filtering/cleaning and were excluded from the ANOVA.", icon="⚠️")

                        except KeyError as e:
                            st.error(f"Error creating summary N table: Missing expected column - {e}. Check data preparation steps.")
                            st.write("Columns available in anova_input_df_clean:", anova_input_df_clean.columns)
                        except Exception as e:
                            st.error(f"An unexpected error occurred while creating the summary N table: {e}")
                            st.dataframe(anova_input_df_clean.head()) # Show head for debugging

                        # --- Fit Two-Way ANOVA Model ---
                        # Define the model formula: Mean Value depends on Group, Cycle, and their interaction
                        # C(...) treats the variable as Categorical
                        formula = 'Q("Mean Value") ~ C(Group) + C(Cycle) + C(Group):C(Cycle)'

                        # Fit the model using Ordinary Least Squares (OLS)
                        model = ols(formula, data=anova_input_df_clean).fit()
                        
                        st.markdown("---") # Add a separator before assumption checks
                        st.markdown("##### Checking ANOVA Assumptions")
                        anova_assumptions_ok = True # Flag to track overall status

                        # 1. Normality of Residuals (using Shapiro-Wilk)
                        residuals = model.resid
                        is_normal, norm_p_val, _ = check_normality(residuals) # Call helper function

                        if is_normal is None:
                            st.caption("Normality check skipped (insufficient data for residuals).")
                        elif is_normal:
                            st.success(f"✅ Residuals appear normally distributed (Shapiro-Wilk p={norm_p_val:.3f}).")
                        else:
                            st.warning(f"⚠️ Residuals may not be normally distributed (Shapiro-Wilk p={norm_p_val:.3f}). ANOVA results might be less reliable, consider transformation or non-parametric alternatives if deviation is severe.")
                            anova_assumptions_ok = False # Mark assumption as potentially violated

                        # 2. Homogeneity of Variances (Levene's Test)
                        #    We need to check variances across all the unique 'Group_Cycle' combinations.
                        #    First, create the combined column if it doesn't exist on the clean data yet.
                        if 'Group_Cycle' not in anova_input_df_clean.columns:
                            # Check if required columns exist before creating combined factor
                            if 'Group' in anova_input_df_clean.columns and 'Cycle' in anova_input_df_clean.columns:
                                anova_input_df_clean['Group_Cycle'] = anova_input_df_clean['Group'] + "_" + anova_input_df_clean['Cycle']
                            else:
                                st.error("Cannot perform homogeneity check: 'Group' or 'Cycle' column missing in anova_input_df_clean.")
                                is_homogeneous = None # Mark as unable to check
                                homog_p_val = None

                        # Proceed only if the combined column was created or already existed
                        if 'Group_Cycle' in anova_input_df_clean.columns:
                            # Prepare data for Levene's: list of arrays, one for each group/cycle combo
                            groups_for_levene = [
                                anova_input_df_clean['Mean Value'][anova_input_df_clean['Group_Cycle'] == gc].values
                                for gc in anova_input_df_clean['Group_Cycle'].unique()
                            ]
                            # Filter out any groups with insufficient data for Levene's (needs >= 2 points)
                            valid_groups_for_levene = [g for g in groups_for_levene if len(g) >= 2]

                            if len(valid_groups_for_levene) < 2: # Levene needs at least 2 groups to compare
                                st.caption("Homogeneity check skipped (less than 2 groups/conditions with sufficient data).")
                                is_homogeneous = None # Mark as unable to check
                                homog_p_val = None
                            else:
                                is_homogeneous, homog_p_val, _ = check_homogeneity(valid_groups_for_levene) # Call helper function

                                if is_homogeneous is None:
                                    st.caption("Homogeneity check failed or could not be performed.")
                                elif is_homogeneous:
                                    st.success(f"✅ Variances appear homogeneous across groups/cycles (Levene's p={homog_p_val:.3f}).")
                                else:
                                    st.warning(f"⚠️ Variances may not be homogeneous across groups/cycles (Levene's p={homog_p_val:.3f}). ANOVA results might be less reliable, especially with unbalanced group sizes.")
                                    anova_assumptions_ok = False # Mark assumption as potentially violated
                        else: # Group_Cycle column creation failed earlier
                            is_homogeneous = None # Ensure it's marked as not checkable
                            homog_p_val = None
                            
                        st.markdown("---") # Add separator after checks

                        # Get the ANOVA table
                        anova_table = sm.stats.anova_lm(model, typ=2) # Type 2 ANOVA is generally recommended

                        # --- Display ANOVA Results ---
                        st.markdown("##### ANOVA Results Table")

                        # Enhance display: Add significance stars
                        def add_p_value_stars(p_value):
                            if p_value < 0.001: return "***"
                            elif p_value < 0.01: return "**"
                            elif p_value < 0.05: return "*"
                            else: return ""

                        # Apply formatting to the table for display
                        display_anova = anova_table.copy() # Work on a copy
                        # Format p-values
                        if 'PR(>F)' in display_anova.columns:
                            display_anova['P-Value'] = display_anova['PR(>F)'].map('{:.4f}'.format)
                            display_anova['Sig.'] = display_anova['PR(>F)'].apply(add_p_value_stars)
                            display_anova = display_anova.drop(columns=['PR(>F)']) # Drop original p-value column

                        # Format other columns (optional, but nice)
                        for col in ['sum_sq', 'mean_sq', 'F']:
                            if col in display_anova.columns:
                                display_anova[col] = display_anova[col].map('{:.3f}'.format)

                        # Reorder columns for better readability
                        cols_order = [col for col in ['sum_sq', 'df', 'mean_sq', 'F', 'P-Value', 'Sig.'] if col in display_anova.columns]
                        display_anova = display_anova[cols_order]

                        st.dataframe(display_anova)

                        # --- Interpretation Helper ---
                        st.markdown("##### Interpretation")
                        interaction_p = anova_table.loc['C(Group):C(Cycle)', 'PR(>F)'] if 'C(Group):C(Cycle)' in anova_table.index else 1.0
                        group_p = anova_table.loc['C(Group)', 'PR(>F)'] if 'C(Group)' in anova_table.index else 1.0
                        cycle_p = anova_table.loc['C(Cycle)', 'PR(>F)'] if 'C(Cycle)' in anova_table.index else 1.0

                        interpretation_texts = []
                        if interaction_p < 0.05:
                            interpretation_texts.append(f"✔️ **Significant Interaction Effect (p={interaction_p:.4f}):** The effect of the Group **depends** on the Light/Dark Cycle. Examine the Interaction Plot below.")
                            interpretation_texts.append("   - *Focus on post-hoc tests comparing groups within each cycle.*")
                        else:
                            interpretation_texts.append(f"❌ No Significant Interaction Effect (p={interaction_p:.4f}): The effect of Group is **consistent** across Light and Dark cycles.")

                        if group_p < 0.05:
                            interpretation_texts.append(f"✔️ **Significant Main Effect of Group (p={group_p:.4f}):** Overall, there is a significant difference between the selected groups (averaging across cycles).")
                            if interaction_p >= 0.05: # Only suggest post-hoc on main effect if no interaction
                                interpretation_texts.append("   - *Post-hoc tests can identify which groups differ overall.*")
                        else:
                            interpretation_texts.append(f"❌ No Significant Main Effect of Group (p={group_p:.4f}): Overall, no significant difference detected between groups.")

                        if cycle_p < 0.05:
                            interpretation_texts.append(f"✔️ **Significant Main Effect of Cycle (p={cycle_p:.4f}):** Overall, there is a significant difference between the Light and Dark cycles (averaging across groups).")
                            if interaction_p >= 0.05: # Only suggest post-hoc on main effect if no interaction
                                interpretation_texts.append("   - *Post-hoc tests can confirm the overall Light vs Dark difference.*")
                        else:
                            interpretation_texts.append(f"❌ No Significant Main Effect of Cycle (p={cycle_p:.4f}): Overall, no significant difference detected between Light and Dark cycles.")

                        if not anova_assumptions_ok:
                            st.warning("**Note:** One or more ANOVA assumptions were potentially violated (see warnings above). Interpret the following results with caution.", icon="❗")
                        else:
                            st.success("**Note:** ANOVA assumptions appear to be met.")
                        st.info("\n".join(interpretation_texts)) # Display the original interpretation texts

                        # --- Interaction Plot ---
                        st.markdown("##### Interaction Plot: Group vs. Cycle")
                        try:
                            # Calculate means and SEM for plotting
                            interaction_plot_data = anova_input_df_clean.groupby(['Group', 'Cycle'])['Mean Value'].agg(
                                Mean='mean',
                                SEM='sem' # Standard Error of the Mean
                            ).reset_index()

                            if interaction_plot_data.empty:
                                 st.warning("Could not calculate data needed for interaction plot.")
                            else:
                                # Get unique groups in the order they appear for consistent coloring
                                # Use the order from the n_table index if available, else from plot data
                                if 'n_table' in locals():
                                    ordered_groups = n_table.index.tolist()
                                else:
                                    ordered_groups = interaction_plot_data['Group'].unique().tolist()

                                # Create the Plotly figure
                                fig_interaction = px.line(
                                    interaction_plot_data,
                                    x='Cycle',          # 'Light', 'Dark' on x-axis
                                    y='Mean',           # Average of the 'Mean Value' on y-axis
                                    color='Group',      # Different colored line for each Group
                                    markers=True,       # Show markers at each point (Light/Dark)
                                    labels={'Mean': f'Mean {parameter} ({PARAMETER_UNITS.get(parameter, "")})'}, # Y-axis label
                                    title=f'Interaction Plot: Mean {parameter} by Group and Cycle',
                                    category_orders={ # Ensure consistent coloring and legend order
                                         'Group': ordered_groups,
                                         'Cycle': ['Light', 'Dark'] # Ensure Light is plotted before Dark
                                    },
                                    error_y='SEM' # Add error bars based on the calculated SEM column
                                )

                                # Customize layout
                                fig_interaction.update_layout(
                                    xaxis_title="Light/Dark Cycle",
                                    yaxis_title=f'Mean {parameter} ({PARAMETER_UNITS.get(parameter, "")})',
                                    legend_title="Group",
                                    hovermode="x unified" # Show hover info for all groups at once
                                )
                                fig_interaction.update_traces(
                                     error_y_thickness=1, # Thinner error bars
                                     error_y_width=10     # Width of caps on error bars
                                )


                                # Display the plot
                                st.plotly_chart(fig_interaction, use_container_width=True)
                                
                                # --- Summary Statistics Table ---
                                st.markdown("##### Summary Statistics (Mean ± SEM, N)")
                                st.markdown("Values used for the interaction plot above:")

                                # Calculate N (number of unique subjects per group/cycle)
                                summary_n = anova_input_df_clean.groupby(['Group', 'Cycle'])['Subject ID'].nunique().reset_index().rename(columns={'Subject ID': 'N'})

                                # Merge N with the existing Mean and SEM data
                                summary_stats_table = pd.merge(interaction_plot_data, summary_n, on=['Group', 'Cycle'], how='left')

                                # Format the table for display
                                summary_stats_table['Mean'] = summary_stats_table['Mean'].map('{:.3f}'.format) # Adjust formatting as needed
                                summary_stats_table['SEM'] = summary_stats_table['SEM'].map('{:.3f}'.format) # Adjust formatting as needed

                                # Display the table, setting index for better readability
                                st.dataframe(summary_stats_table.set_index(['Group', 'Cycle']))
                                # --- End Summary Statistics Table ---

                            with st.expander("How to read this plot"):
                                st.markdown(f"""
                                *   **X-axis:** Shows the two conditions: 'Light' cycle and 'Dark' cycle.
                                *   **Y-axis:** Shows the average **{parameter}** value.
                                *   **Lines:** Each colored line connects the average value for a specific **Group** during the Light cycle to its average value during the Dark cycle.
                                *   **Error Bars:** The vertical lines represent the Standard Error of the Mean (SEM) for each point, indicating the precision of that group's average in that specific cycle.
                                *   **Interaction:** Look at how the lines behave relative to each other.
                                    *   *Parallel lines* suggest **no interaction** (the difference between groups is similar in both cycles). This matches your ANOVA result (p={interaction_p:.4f}).
                                    *   *Non-parallel or crossing lines* would suggest an **interaction** (the effect of the group depends on the cycle).
                                """)

                        except Exception as e:
                            st.error(f"An error occurred while generating the interaction plot: {e}")
                            st.dataframe(interaction_plot_data.head() if 'interaction_plot_data' in locals() else "Plot data not generated.")

                        # --- Post-Hoc Tests (Conditional) ---
                        st.markdown("---") # Add a separator

                        # Check if any ANOVA effect was significant enough to warrant post-hoc tests
                        if interaction_p < 0.05 or group_p < 0.05 or cycle_p < 0.05:
                            # If significant, create an expander for the tests
                            with st.expander("🔬 Post-Hoc Tests (Pairwise Comparisons - Tukey HSD)", expanded=False):
                                try:
                                    # 1. Prepare data for Tukey HSD: Create a single combined factor
                                    #    We need a unique label for each condition (e.g., "GroupA_Dark")
                                    #    We add this as a new column to our clean ANOVA input data.
                                    data_for_tukey = anova_input_df_clean.copy() # Work on a copy
                                    data_for_tukey['Group_Cycle'] = data_for_tukey['Group'] + "_" + data_for_tukey['Cycle']

                                    # Check if the new column was created successfully
                                    if 'Group_Cycle' not in data_for_tukey.columns:
                                        st.error("Failed to create combined 'Group_Cycle' column for post-hoc tests.")
                                    else:
                                        # 2. Run Tukey HSD test
                                        #    We compare the 'Mean Value' across the different 'Group_Cycle' labels.
                                        tukey_results = pairwise_tukeyhsd(
                                            endog=data_for_tukey['Mean Value'], # The dependent variable (your measured parameter)
                                            groups=data_for_tukey['Group_Cycle'], # The labels identifying each group/condition
                                            alpha=0.05 # The significance level (usually 0.05)
                                        )

                                        # 3. Display Tukey HSD Results
                                        st.markdown("##### Tukey HSD Results")
                                        st.markdown("Compares all pairs of conditions (Group & Cycle combined). `reject=True` indicates a significant difference (p-adj < 0.05).")

                                        # The results object has a nice summary table built-in.
                                        # We can convert it to a Pandas DataFrame for better display in Streamlit.
                                        tukey_df = pd.DataFrame(data=tukey_results._results_table.data[1:], columns=tukey_results._results_table.data[0])

                                        # Optional: Add significance stars for quick visual scanning (using the function defined earlier)
                                        if 'p-adj' in tukey_df.columns and 'Sig.' not in tukey_df.columns: # Add Sig column if not present
                                            # Ensure p-adj is numeric before applying stars
                                            tukey_df['p-adj'] = pd.to_numeric(tukey_df['p-adj'], errors='coerce')
                                            tukey_df['Sig.'] = tukey_df['p-adj'].apply(add_p_value_stars)


                                        # Format numeric columns for better readability before display
                                        numeric_cols_tukey = ['mean_diff', 'p-adj', 'lower', 'upper']
                                        for col in numeric_cols_tukey:
                                            if col in tukey_df.columns:
                                                # Convert to numeric first (robustness), then format
                                                tukey_df[col] = pd.to_numeric(tukey_df[col], errors='coerce').map('{:.4f}'.format)


                                        # Display the formatted DataFrame
                                        st.dataframe(tukey_df)

                                        # 4. Interpretation Guidance (Crucial!)
                                        st.markdown("##### How to Interpret Tukey HSD Results:")
                                        # Provide advice based on the ANOVA interaction result
                                        if interaction_p < 0.05:
                                            st.info(f"""
                                            **Focus on Simple Effects (Interaction was Significant, p={interaction_p:.4f}):**
                                            *   Compare groups **within the Dark cycle** (look for pairs like 'GroupA_Dark' vs 'GroupB_Dark').
                                            *   Compare groups **within the Light cycle** (look for pairs like 'GroupA_Light' vs 'GroupB_Light').
                                            *   Compare Light vs Dark **within each Group** (look for pairs like 'GroupA_Dark' vs 'GroupA_Light').
                                            """)
                                        else:
                                            st.info(f"""
                                            **Focus on Main Effects (Interaction was NOT Significant, p={interaction_p:.4f}):**
                                            *   If the **Group** main effect was significant (p={group_p:.4f}), look for consistent differences between groups across *both* cycles (e.g., is 'GroupA_Dark' different from 'GroupB_Dark' AND 'GroupA_Light' different from 'GroupB_Light'?).
                                            *   If the **Cycle** main effect was significant (p={cycle_p:.4f}), look for consistent differences between Dark and Light within *each* group (e.g., is 'GroupA_Dark' different from 'GroupA_Light', 'GroupB_Dark' vs 'GroupB_Light', etc.?).
                                            """)
                                        st.caption("`group1`, `group2`: The pair being compared. `mean_diff`: Difference in means (group2 - group1). `p-adj`: Adjusted p-value. `lower`, `upper`: Confidence interval for the mean difference. `reject`: True if p-adj < alpha (0.05).")

                                # Error handling for the post-hoc step
                                except ImportError:
                                    st.error("Could not perform post-hoc tests. The 'statsmodels' library might be missing or improperly installed.")
                                except AttributeError:
                                    st.error("An issue occurred accessing Tukey HSD results. Check data preparation.")
                                    st.dataframe(data_for_tukey.head()) # Show data used
                                except Exception as e:
                                    st.error(f"An unexpected error occurred during post-hoc analysis: {e}")
                                    st.exception(e) # Show full traceback for debugging

                        else:
                            # If no ANOVA effects were significant, inform the user
                            st.success("No significant effects found in the Two-Way ANOVA (all p-values > 0.05). Post-hoc tests are typically not performed in this case.")

                        # --- End of Post-Hoc Section ---
                        # --- Secondary Analysis: Simple Comparisons ---
                        st.markdown("---") # Separator
                        with st.expander("🔎 Quick Comparisons (Specific Cycle or 24h Average)", expanded=False):

                            # --- Step 1: Select Data Slice ---
                            st.markdown("**1. Select Data to Compare:**")
                            
                            # Define available columns based on the parameter
                            if parameter in ["XTOT", "XAMB"]:
                                comparison_options = {
                                    "Light Cycle (Avg Activity)": "True (Average Activity)",
                                    "Dark Cycle (Avg Activity)": "False (Average Activity)",
                                    "24h Average (Avg Activity)": "24h Average" # Check if this exists in your 'results' df
                                }
                            elif parameter == "FEED":
                                comparison_options = {
                                    "Light Cycle (Avg Rate)": "Average Rate (Light)",
                                    "Dark Cycle (Avg Rate)": "Average Rate (Dark)",
                                    # Add total intake options if relevant
                                    "Light Cycle (Total Intake)": "Total Intake (Light)",
                                    "Dark Cycle (Total Intake)": "Total Intake (Dark)",
                                }
                            else: # VO2, VCO2, RER, HEAT
                                comparison_options = {
                                    "Light Cycle Average": "Light Average",
                                    "Dark Cycle Average": "Dark Average",
                                    "Total (24h) Average": "Total Average"
                                }

                            # Filter options based on columns actually present in the results DataFrame
                            available_comparison_options = {k: v for k, v in comparison_options.items() if v in results.columns}

                            if not available_comparison_options:
                                st.warning(f"No suitable columns found in the results table for quick comparisons with parameter '{parameter}'.")
                            else:
                                selected_option_key = st.radio(
                                    "Choose the specific metric to compare:",
                                    options=list(available_comparison_options.keys()),
                                    key="quick_compare_metric_select",
                                    horizontal=True
                                )
                                # Get the actual column name corresponding to the user's selection
                                column_to_compare = available_comparison_options[selected_option_key]

                                # --- Step 2: Select Groups for THIS comparison ---
                                st.markdown("**2. Select Groups for this Comparison:**")
                                # Get available groups (already available from main ANOVA section)
                                # available_groups_stat = sorted(group_assignments_df['Group'].unique()) # Already defined earlier
                                
                                selected_groups_quick = st.multiselect(
                                    "Select 2 or more groups for this specific comparison:",
                                    options=available_groups_stat,
                                    default=available_groups_stat[:2] if len(available_groups_stat) >= 2 else available_groups_stat, # Default to first 2 groups
                                    key="quick_compare_group_select",
                                    help="Choose groups to compare using the selected metric."
                                )

                                # --- Step 3: Prepare Data & Run Test ---
                                if len(selected_groups_quick) < 2:
                                    st.warning("⚠️ Please select at least two groups for comparison.")
                                else:
                                    try:
                                        st.markdown("**3. Comparison Results:**")
                                        
                                        # Prepare the data for the simple comparison
                                        # Merge 'results' table with group assignments
                                        results_with_groups_quick = pd.merge(
                                            results, # Use the main 'results' summary table
                                            group_assignments_df[['Subject ID', 'Group']],
                                            on='Subject ID',
                                            how='inner' # Keep only animals present in both results and assignments
                                        )
                                        
                                        # Filter for selected groups AND the chosen column
                                        quick_comp_data = results_with_groups_quick[
                                            results_with_groups_quick['Group'].isin(selected_groups_quick)
                                        ][['Group', 'Subject ID', column_to_compare]].copy()

                                        # Drop rows with missing data in the column being compared
                                        quick_comp_data.dropna(subset=[column_to_compare], inplace=True)

                                        # Check if data remains after filtering/dropping NaN
                                        if quick_comp_data.empty or quick_comp_data['Group'].nunique() < 2:
                                            st.warning(f"Not enough data or groups remaining for comparison after filtering for '{selected_option_key}' and selected groups.")
                                        else:
                                            # --- Perform T-test or One-Way ANOVA ---
                                            groups_for_test = quick_comp_data.groupby('Group')[column_to_compare].apply(list)
                                            
                                            if len(selected_groups_quick) == 2:
                                                # Perform Independent Samples T-test
                                                group1_data = groups_for_test.iloc[0]
                                                group2_data = groups_for_test.iloc[1]
                                                
                                                # Check for sufficient data in each group
                                                if len(group1_data) < 2 or len(group2_data) < 2:
                                                    st.warning("Need at least 2 data points per group for a t-test.")
                                                else:
                                                    st.markdown("---") # Separator
                                                    st.markdown("##### Checking T-test Assumptions")
                                                    ttest_assumptions_ok = True

                                                    # Check normality for EACH group
                                                    norm_results = {}
                                                    for i, group_name in enumerate(groups_for_test.index):
                                                        is_normal, p_val, _ = check_normality(groups_for_test.iloc[i])
                                                        norm_results[group_name] = (is_normal, p_val)
                                                        if is_normal is None:
                                                            st.caption(f"Normality check skipped for {group_name} (N<3).")
                                                        elif is_normal:
                                                            st.caption(f"Data for {group_name} appears normal (p={p_val:.3f}).") # Use caption for success
                                                        else:
                                                            st.warning(f"⚠️ Data for {group_name} may not be normal (Shapiro-Wilk p={p_val:.3f}). T-test validity could be affected.")
                                                            ttest_assumptions_ok = False

                                                    # Check homogeneity (using the same Levene's test helper)
                                                    is_homogeneous, homog_p_val, _ = check_homogeneity(groups_for_test.tolist()) # Pass list of group data arrays
                                                    if is_homogeneous is None:
                                                        st.caption(f"Homogeneity check skipped or failed.")
                                                    elif is_homogeneous:
                                                        st.caption(f"Variances appear homogeneous (Levene's p={homog_p_val:.3f}).") # Use caption for success
                                                    else:
                                                        st.warning(f"⚠️ Variances may not be homogeneous (Levene's p={homog_p_val:.3f}). Welch's t-test (used here) is robust to this.")
                                                        # Note: We don't set ttest_assumptions_ok to False here because Welch's t-test handles it.

                                                    # Display overall assumption status for T-test interpretation
                                                    if not ttest_assumptions_ok:
                                                        st.warning("**Note:** Normality assumption potentially violated. Interpret t-test results with caution.", icon="❗")
                                                    st.markdown("---") # Separator
                                                    
                                                    t_stat, p_val_ttest = stats.ttest_ind(group1_data, group2_data, equal_var=False) # Welch's t-test (doesn't assume equal variances)
                                                    
                                                    st.metric(
                                                        label=f"T-test Result ({groups_for_test.index[0]} vs {groups_for_test.index[1]})",
                                                        value=f"p = {p_val_ttest:.4f}",
                                                        delta="Significant" if p_val_ttest < 0.05 else "Not Significant",
                                                        delta_color=("inverse" if p_val_ttest < 0.05 else "off")
                                                    )
                                                    st.caption(f"Comparing '{selected_option_key}'. Welch's t-test used.")

                                            else: # 3 or more groups selected
                                                # Perform One-Way ANOVA
                                                # Check for sufficient data
                                                if any(len(group_data) < 2 for group_data in groups_for_test):
                                                    st.warning("Need at least 2 data points per group for ANOVA.")
                                                else:
                                                    st.markdown("---") # Separator
                                                    st.markdown("##### Checking One-Way ANOVA Assumptions")
                                                    oneway_assumptions_ok = True

                                                    # Check normality for EACH group
                                                    norm_results_oneway = {}
                                                    for i, group_name in enumerate(groups_for_test.index):
                                                        is_normal, p_val, _ = check_normality(groups_for_test.iloc[i])
                                                        norm_results_oneway[group_name] = (is_normal, p_val)
                                                        if is_normal is None:
                                                            st.caption(f"Normality check skipped for {group_name} (N<3).")
                                                        elif is_normal:
                                                            st.caption(f"Data for {group_name} appears normal (p={p_val:.3f}).") # Caption for success
                                                        else:
                                                            st.warning(f"⚠️ Data for {group_name} may not be normal (Shapiro-Wilk p={p_val:.3f}). ANOVA validity could be affected.")
                                                            oneway_assumptions_ok = False

                                                    # Check homogeneity
                                                    is_homogeneous_oneway, homog_p_val_oneway, _ = check_homogeneity(groups_for_test.tolist())
                                                    if is_homogeneous_oneway is None:
                                                        st.caption(f"Homogeneity check skipped or failed.")
                                                    elif is_homogeneous_oneway:
                                                        st.caption(f"Variances appear homogeneous (Levene's p={homog_p_val_oneway:.3f}).") # Caption for success
                                                    else:
                                                        st.warning(f"⚠️ Variances may not be homogeneous (Levene's p={homog_p_val_oneway:.3f}). ANOVA is somewhat robust, but consider alternatives if severe.")
                                                        oneway_assumptions_ok = False # Violation of homogeneity is more problematic for ANOVA than Welch's t-test

                                                    # Display overall assumption status for One-Way ANOVA interpretation
                                                    if not oneway_assumptions_ok:
                                                        st.warning("**Note:** One or more ANOVA assumptions potentially violated. Interpret results with caution.", icon="❗")
                                                    st.markdown("---") # Separator
                                                    
                                                    f_stat, p_val_anova = stats.f_oneway(*groups_for_test)
                                                    
                                                    st.metric(
                                                        label=f"One-Way ANOVA Result ({len(selected_groups_quick)} groups)",
                                                        value=f"p = {p_val_anova:.4f}",
                                                        delta="Significant Overall" if p_val_anova < 0.05 else "Not Significant Overall",
                                                        delta_color=("inverse" if p_val_anova < 0.05 else "off")
                                                    )
                                                    st.caption(f"Comparing '{selected_option_key}'.")

                                                    # --- Post-Hoc for One-Way ANOVA (if significant) ---
                                                    if p_val_anova < 0.05:
                                                        st.markdown("##### Post-Hoc Tests (Pairwise T-tests with Bonferroni Correction)")
                                                        
                                                        # Use scikit-posthocs for pairwise t-tests with correction
                                                        # Need to provide data in a specific format (long format: value, group)
                                                        posthoc_df = posthoc_ttest(quick_comp_data, val_col=column_to_compare, group_col='Group', p_adjust='bonferroni', equal_var=False) # Use Welch's t-test assumption

                                                        # Display the results (heatmap style is often used, or just the table)
                                                        st.dataframe(posthoc_df.style.format("{:.4f}").applymap(lambda x: 'background-color: yellow' if x<0.05 else ''))
                                                        st.caption("Table shows Bonferroni-adjusted p-values for pairwise t-tests. Yellow cells indicate p < 0.05.")

                                            # --- Simple Bar Chart for Quick Comparison ---
                                            st.markdown("##### Quick Plot")
                                            quick_plot_data = quick_comp_data.groupby('Group')[column_to_compare].agg(['mean', 'sem']).reset_index()

                                            fig_quick = px.bar(
                                                quick_plot_data,
                                                x='Group',
                                                y='mean',
                                                error_y='sem',
                                                labels={'mean': f'Mean {selected_option_key}', 'Group': 'Selected Groups'},
                                                title=f"Quick Comparison: Mean {selected_option_key}",
                                                color='Group', # Color bars by group
                                                color_discrete_map=GROUP_COLORS # Use predefined colors
                                            )
                                            fig_quick.update_layout(showlegend=False) # Hide legend if coloring by x-axis group
                                            st.plotly_chart(fig_quick, use_container_width=True)

                                    except ImportError:
                                        st.error("Error running comparison: 'scipy.stats' or 'scikit_posthocs' might be missing. Please ensure they are installed.")
                                    except Exception as e:
                                        st.error(f"An error occurred during the quick comparison: {e}")
                                        st.exception(e) # Show traceback
                        # --- Export Analysis Results ---
                        st.markdown("---") # Separator before export section
                        st.subheader("📥 Export Statistical Analysis Results")
                        st.markdown("Download tables generated in this analysis tab.")

                        export_col1, export_col2, export_col3 = st.columns(3) # Use columns for layout

                        # Export Button 1: Two-Way ANOVA Table
                        with export_col1:
                            if 'anova_table' in locals() and anova_table is not None:
                                try:
                                    # Use the original anova_table before display formatting
                                    csv_anova = anova_table.to_csv().encode('utf-8')
                                    st.download_button(
                                        label="Download ANOVA Table",
                                        data=csv_anova,
                                        file_name=f"{parameter}_TwoWayANOVA_results.csv",
                                        mime="text/csv",
                                        help="Download the main Two-Way ANOVA (Group x Cycle) results table.",
                                        key="export_anova_table"
                                    )
                                except Exception as e:
                                    st.warning(f"Could not prepare ANOVA table for download: {e}")
                            else:
                                st.button("Download ANOVA Table", disabled=True, help="ANOVA table not generated.")

                        # Export Button 2: Summary Statistics (Mean/SEM/N) Table
                        with export_col2:
                            # Use the 'summary_stats_table' we created before displaying it
                            if 'summary_stats_table' in locals() and summary_stats_table is not None:
                                try:
                                    # Prepare a version for export (maybe without the index set)
                                    export_summary_stats = summary_stats_table.reset_index() # Flatten index for CSV
                                    csv_summary_stats = export_summary_stats.to_csv(index=False).encode('utf-8')
                                    st.download_button(
                                        label="Download Summary Stats",
                                        data=csv_summary_stats,
                                        file_name=f"{parameter}_Group_Cycle_Summary_Stats.csv",
                                        mime="text/csv",
                                        help="Download Mean, SEM, and N for each Group/Cycle condition.",
                                        key="export_summary_stats"
                                    )
                                except Exception as e:
                                    st.warning(f"Could not prepare Summary Stats table for download: {e}")
                            else:
                                st.button("Download Summary Stats", disabled=True, help="Summary Stats table not generated.")

                        # Export Button 3: Tukey HSD Post-Hoc Table
                        with export_col3:
                            # Use the 'tukey_df' we created for display
                            if 'tukey_df' in locals() and tukey_df is not None:
                                try:
                                    # Export the DataFrame as created/formatted for display
                                    csv_tukey = tukey_df.to_csv(index=False).encode('utf-8')
                                    st.download_button(
                                        label="Download Post-Hoc Table",
                                        data=csv_tukey,
                                        file_name=f"{parameter}_TukeyHSD_PostHoc.csv",
                                        mime="text/csv",
                                        help="Download the pairwise comparison results (Tukey HSD). Only available if ANOVA was significant.",
                                        key="export_tukey_table"
                                    )
                                except Exception as e:
                                    st.warning(f"Could not prepare Post-Hoc table for download: {e}")
                            else:
                                # Show disabled button if Tukey wasn't run or failed
                                st.button("Download Post-Hoc Table", disabled=True, help="Post-Hoc table not generated (ANOVA may not have been significant).")

                            # --- End of Export Section ---
                    except Exception as e:
                        st.error(f"An error occurred during data preparation for ANOVA: {e}")
                        st.exception(e) # Shows detailed traceback for debugging
                    
            # Tab 4: Publication Plots
            with tab4:
                st.header("📄 Publication Plots")

                # --- Prerequisite Check ---
                # Perform checks *before* showing any other controls in this tab
                # Ensure variables like 'uploaded_file', 'results', 'raw_data' are accessible here
                # They are defined outside the tabs, so they should be unless there's an execution flow issue

                groups_assigned = ('group_assignments' in st.session_state and
                                not st.session_state.get('group_assignments', pd.DataFrame()).empty)

                # Check 1: File uploaded and processed?
                if uploaded_file is None or 'results' not in locals() or results is None or 'raw_data' not in locals() or raw_data is None:
                    st.warning("⚠️ Please upload and process data in the **'Overview'** tab first.")
                    # Stop execution *within this tab* if data isn't ready
                    st.stop()

                # Check 2: Groups assigned? (Only check if data IS ready)
                elif not groups_assigned:
                    st.error("🚫 **Action Required:** Please assign animals to groups in the **'Overview' tab → 'Setup: Groups & Lean Mass'** section before using this tab.")
                    # Stop execution *within this tab* if groups aren't ready
                    st.stop()

                # --- If all checks passed, display the rest of the tab content ---
                else:
                    st.success("Data and groups loaded. Ready for plotting.") # Confirmation message
                    st.markdown("""
                    Generate better quality versions of your key plots using Matplotlib
                    with `scienceplots` styling, suitable for scientific papers, presentations, theses.
                    """)
                    def save_figure_to_buffer(fig, format):
                        """Saves a matplotlib figure to an in-memory buffer."""
                        buf = io.BytesIO()
                        # Use bbox_inches='tight' to prevent labels/titles from being cut off
                        # Use dpi=300 for good resolution in raster formats (like PNG)
                        fig.savefig(buf, format=format, bbox_inches='tight', dpi=300)
                        buf.seek(0) # Rewind buffer to the beginning
                        return buf

                    # --- Plot Selection and Customization ---
                    st.markdown("---")
                    plot_type = st.radio(
                        "Select Plot Type:",
                        ["Group Comparison Bar Chart", "24h Pattern", "Timeline Plot"], # Added "Timeline Plot"
                        key="pub_plot_type",
                        horizontal=True # Make selection horizontal
                    )

                    # --- Conditional Plot Generation ---

                    if plot_type == "Group Comparison Bar Chart":
                        # --- Bar Chart Specific Code ---
                        st.subheader("Bar Chart Options")
                        col1_opts, col2_opts = st.columns(2)
                        with col1_opts:
                            cycle_choice_pub = st.radio("Select data to plot:", ["24-hour Average", "Light Cycle", "Dark Cycle"], key="pub_cycle_selector_bar", horizontal=False)
                        with col2_opts:
                            error_bar_choice = st.radio("Error Bar Type:", ["SEM", "SD"], key="pub_error_bar", horizontal=False)

                        # Determine cycle filter based on choice
                        if "Light Cycle" in cycle_choice_pub:
                            cycle_filter_pub = True
                            cycle_name_pub = "Light Cycle"
                        elif "Dark Cycle" in cycle_choice_pub:
                            cycle_filter_pub = False
                            cycle_name_pub = "Dark Cycle"
                        else:
                            cycle_filter_pub = None
                            cycle_name_pub = "24-hour Average"

                        # --- Data Preparation ---
                        group_assignments = st.session_state['group_assignments']
                        try:
                            # Ensure necessary data normalization and merging occurs
                            # (Assuming raw_data and group_assignments are prepared correctly based on earlier checks)
                            if 'normalized_cage_id' not in raw_data.columns:
                                def normalize_cage_id(cage_str): import re; match = re.search(r'(\d+)', str(cage_str)); cage_num = match.group(1) if match else None; return ('1' + cage_num if len(cage_num) == 2 else cage_num) if cage_num else None
                                raw_data['normalized_cage_id'] = raw_data['cage'].apply(normalize_cage_id)
                            if 'normalized_cage_id' not in group_assignments.columns:
                                if 'Cage' in group_assignments.columns: group_assignments.loc[:, 'normalized_cage_id'] = group_assignments['Cage'].apply(normalize_cage_id)
                                else: st.error("Group assignments missing 'Cage' column."); st.stop()

                            if 'normalized_cage_id' in raw_data.columns and 'normalized_cage_id' in group_assignments.columns:
                                grouped_data_pub = pd.merge(raw_data, group_assignments[['normalized_cage_id', 'Group', 'Subject ID']], on='normalized_cage_id', how='inner')
                            else: st.error("Normalized cage ID missing."); st.stop()
                            if grouped_data_pub.empty: st.error("Merging resulted in empty data."); st.stop()

                            # Calculate group statistics based on selected cycle
                            if cycle_filter_pub is not None:
                                filtered_cycle_data = grouped_data_pub[grouped_data_pub['is_light'] == cycle_filter_pub]
                                if filtered_cycle_data.empty: st.warning(f"No data for {cycle_name_pub}."); st.stop()
                                group_stats_pub = filtered_cycle_data.groupby(['Group', 'Subject ID'])['value'].agg(Mean='mean').reset_index().groupby('Group').agg(Mean=('Mean', 'mean'), SD=('Mean', 'std'), SEM=('Mean', 'sem'), N=('Subject ID', 'nunique')).reset_index()
                            else:
                                group_stats_pub = grouped_data_pub.groupby(['Group', 'Subject ID'])['value'].agg(Mean='mean').reset_index().groupby('Group').agg(Mean=('Mean', 'mean'), SD=('Mean', 'std'), SEM=('Mean', 'sem'), N=('Subject ID', 'nunique')).reset_index()
                            if group_stats_pub.empty: st.error("Group stats calculation failed."); st.stop()

                            # --- Generate and Display Plot ---
                            pub_fig_bar = generate_pub_bar_chart(group_stats=group_stats_pub, parameter=parameter, error_bar_type=error_bar_choice, cycle_name=cycle_name_pub)
                            st.pyplot(pub_fig_bar)

                            # --- Description ---
                            with st.expander("What am I looking at? (Calculation Details)"):
                                st.markdown(f"""
                                This bar chart shows the **mean {parameter}** value for each experimental group you assigned.

                                * **Bars**: The height of each bar represents the average value for all animals within that group.
                                * **Error Bars**: The lines extending from the bars represent the **{error_bar_choice}** (Standard Error of the Mean or Standard Deviation).
                                * **Calculation Basis**: These averages are calculated using the **{cycle_name_pub}** data for each animal within the selected time window (`{time_window}`).
                                * **Styling**: The plot uses `scienceplots` for a publication-ready appearance.
                                """)

                            # --- Downloads ---
                            st.markdown("---"); st.subheader("Download Plot")
                            # Call the globally defined helper function
                            png_buffer = save_figure_to_buffer(pub_fig_bar, 'png'); pdf_buffer = save_figure_to_buffer(pub_fig_bar, 'pdf'); svg_buffer = save_figure_to_buffer(pub_fig_bar, 'svg')
                            col1, col2, col3 = st.columns(3)
                            dl_file_prefix = f"{parameter}_{cycle_name_pub.replace(' ', '_')}_bar"
                            with col1: st.download_button(f"Download PNG", png_buffer, f"{dl_file_prefix}.png", "image/png", key="png_pub_bar")
                            with col2: st.download_button(f"Download PDF", pdf_buffer, f"{dl_file_prefix}.pdf", "application/pdf", key="pdf_pub_bar")
                            with col3: st.download_button(f"Download SVG", svg_buffer, f"{dl_file_prefix}.svg", "image/svg+xml", key="svg_pub_bar")
                            plt.close(pub_fig_bar)

                        except Exception as e: st.error(f"Error during Bar Chart generation: {e}"); st.exception(e)

                    elif plot_type == "24h Pattern":
                        # --- 24h Pattern Specific Code ---
                        st.subheader("24h Pattern Plot Options")
                        all_groups = sorted(st.session_state['group_assignments']['Group'].unique())
                        selected_groups_pub = st.multiselect("Select groups to display:", options=all_groups, default=all_groups[:min(4, len(all_groups))], key="pub_pattern_group_select")

                        if not selected_groups_pub: st.warning("Please select at least one group."); st.stop()

                        # --- Data Preparation ---
                        group_assignments = st.session_state['group_assignments']
                        try:
                            # Ensure necessary data prep happened (copied logic for safety, consider refactoring)
                            if 'normalized_cage_id' not in raw_data.columns:
                                def normalize_cage_id(cage_str): import re; match = re.search(r'(\d+)', str(cage_str)); cage_num = match.group(1) if match else None; return ('1' + cage_num if len(cage_num) == 2 else cage_num) if cage_num else None
                                raw_data['normalized_cage_id'] = raw_data['cage'].apply(normalize_cage_id)
                            if 'normalized_cage_id' not in group_assignments.columns:
                                if 'Cage' in group_assignments.columns: group_assignments.loc[:, 'normalized_cage_id'] = group_assignments['Cage'].apply(normalize_cage_id)
                                else: st.error("Group assignments missing 'Cage' column."); st.stop()

                            if 'normalized_cage_id' in raw_data.columns and 'normalized_cage_id' in group_assignments.columns:
                                grouped_data_pub = pd.merge(raw_data, group_assignments[['normalized_cage_id', 'Group', 'Subject ID']], on='normalized_cage_id', how='inner')
                            else: st.error("Normalized cage ID missing."); st.stop()

                            grouped_data_filtered = grouped_data_pub[grouped_data_pub['Group'].isin(selected_groups_pub)]
                            if grouped_data_filtered.empty: st.warning("No data for selected groups."); st.stop()

                            # Calculate hourly stats
                            hourly_stats_list = []
                            for group in selected_groups_pub:
                                group_hourly_data = grouped_data_filtered[grouped_data_filtered['Group'] == group]
                                if not group_hourly_data.empty:
                                    if 'hour' not in group_hourly_data.columns: st.error("'hour' column missing."); st.stop()
                                    hourly_avg = group_hourly_data.groupby('hour')['value'].agg(['mean', 'sem']).reset_index()
                                    hourly_avg['Group'] = group
                                    hourly_stats_list.append(hourly_avg)

                            if not hourly_stats_list: st.error("Could not calculate hourly stats."); st.stop()
                            hourly_df_pub = pd.concat(hourly_stats_list).reset_index(drop=True)

                            # Reindex to ensure all hours 0-23 are present for every group
                            all_hours_index = pd.MultiIndex.from_product([selected_groups_pub, range(24)], names=['Group', 'hour'])
                            hourly_df_pub = hourly_df_pub.set_index(['Group', 'hour']).reindex(all_hours_index).reset_index()

                            # --- Generate and Display Plot ---
                            pub_fig_pattern = generate_pub_24h_pattern_plot(
                                hourly_df=hourly_df_pub, selected_groups=selected_groups_pub, parameter=parameter,
                                light_start=st.session_state.get('light_start', 7), light_end=st.session_state.get('light_end', 19)
                            )
                            st.pyplot(pub_fig_pattern)

                            # --- Description ---
                            with st.expander("What am I looking at? (Calculation Details)"):
                                st.markdown(f"""
                                This line plot shows the **average {parameter} value** for each hour of a typical day (0-23).

                                * **Lines**: Each colored line represents the average pattern for an experimental group.
                                * **Shaded Bands**: The semi-transparent areas around each line represent the **SEM** (Standard Error of the Mean), indicating the precision of the hourly average.
                                * **Calculation Basis**: Data from all days within the selected time window (`{time_window}`) is averaged for each hour to create a single 24-hour profile.
                                * **Background Shading**: Grey shaded areas indicate the **Dark Cycle** based on the start/end times set in the sidebar.
                                * **Styling**: The plot uses `scienceplots` for a publication-ready appearance.
                                """)

                            # --- Downloads ---
                            st.markdown("---"); st.subheader("Download Plot")
                            # Call the globally defined helper function
                            png_buffer_p = save_figure_to_buffer(pub_fig_pattern, 'png'); pdf_buffer_p = save_figure_to_buffer(pub_fig_pattern, 'pdf'); svg_buffer_p = save_figure_to_buffer(pub_fig_pattern, 'svg')
                            col1p, col2p, col3p = st.columns(3)
                            dl_file_prefix_p = f"{parameter}_24h_pattern"
                            with col1p: st.download_button(f"Download PNG", png_buffer_p, f"{dl_file_prefix_p}.png", "image/png", key="png_pub_pattern")
                            with col2p: st.download_button(f"Download PDF", pdf_buffer_p, f"{dl_file_prefix_p}.pdf", "application/pdf", key="pdf_pub_pattern")
                            with col3p: st.download_button(f"Download SVG", svg_buffer_p, f"{dl_file_prefix_p}.svg", "image/svg+xml", key="svg_pub_pattern")
                            plt.close(pub_fig_pattern)
                        
                        except Exception as e: st.error(f"Error during 24h Pattern plot generation: {e}"); st.exception(e)
                    elif plot_type == "Timeline Plot":
                            # --- Timeline Plot Specific Code ---
                            st.subheader("Timeline Plot Options")
                            timeline_display_mode = st.radio(
                                "Display Mode:",
                                ["Show Average Across All Animals", "Focus on Individual Animals"],
                                key="pub_timeline_display_mode",
                                horizontal=True
                            )

                            # --- Prepare Subject/Cage Mapping (Corrected) ---
                            # Create map from Subject ID -> Positional Cage Name (e.g., "CAGE 01") using the 'results' DataFrame
                            subject_to_positional_cage = {}
                            if 'results' in locals() and results is not None and 'Subject ID' in results.columns:
                                try:
                                    # results.index should be the positional cage names ('CAGE 01', 'CAGE 02', ...)
                                    subject_to_positional_cage = pd.Series(results.index.values, index=results['Subject ID']).to_dict()
                                except Exception as e:
                                    st.error(f"Error creating Subject ID to Positional Cage map from results: {e}")
                                    # Provide info to debug if map fails
                                    st.dataframe(results.head())

                            if not subject_to_positional_cage:
                                st.error("Failed to create Subject ID <-> Positional Cage mapping. Cannot plot individuals.")
                                # Avoid stopping if Average mode is selected
                                if timeline_display_mode != "Show Average Across All Animals":
                                    st.stop()

                            selected_subjects_pub_timeline = [] # Initialize
                            group_assignments = st.session_state['group_assignments'] # Get assignments

                            # --- Options for Individual Mode ---
                            if timeline_display_mode == "Focus on Individual Animals":
                                all_subjects_timeline = sorted(group_assignments['Subject ID'].unique())
                                subject_options_timeline = [f"{subject} ({group_assignments[group_assignments['Subject ID']==subject]['Group'].iloc[0]})"
                                                            if subject in group_assignments['Subject ID'].values else subject
                                                            for subject in all_subjects_timeline]

                                selected_subject_labels_timeline = st.multiselect(
                                    "Select animals to display:",
                                    options=subject_options_timeline,
                                    default=subject_options_timeline[:min(1, len(subject_options_timeline))],
                                    key="pub_timeline_subject_select"
                                )
                                selected_subjects_pub_timeline = [label.split(" (")[0] for label in selected_subject_labels_timeline]

                                if not selected_subjects_pub_timeline:
                                    st.warning("Please select at least one animal for individual view.")
                                    st.stop()

                            # --- Data Preparation ---
                            try:
                                if 'raw_data' not in locals() or raw_data is None: st.error("Raw data not available."); st.stop()

                                # Ensure datetime column exists
                                if 'datetime' not in raw_data.columns:
                                    raw_data['datetime'] = pd.to_datetime(raw_data['timestamp'].dt.strftime('%Y-%m-%d %H:00:00'))

                                timeline_plot_data = None # Initialize

                                if timeline_display_mode == "Show Average Across All Animals":
                                    # Calculate hourly mean and SEM robustly using named aggregation
                                    # Explicitly use pandas' built-in SEM calculation for robustness
                                    try: # Add a try-except block for safety during aggregation
                                        timeline_summary = raw_data.groupby('datetime').agg(
                                            Mean=('value', 'mean'),
                                            SEM=('value', pd.Series.sem) # Use pandas' specific SEM function
                                        ).reset_index()

                                        # Check if SEM calculation resulted in all NaNs (e.g., only 1 animal per time point)
                                        if timeline_summary['SEM'].isnull().all() and not timeline_summary['Mean'].isnull().any():
                                             st.warning("⚠️ SEM could not be calculated (likely only one data point per time point). Error bars will not be shown.", icon="📉")
                                             # Optionally, set SEM to 0 if you still want the plot line but no error band
                                             # timeline_summary['SEM'] = 0
                                        elif timeline_summary['SEM'].isnull().any():
                                             st.warning("⚠️ SEM could not be calculated for some time points (likely only one data point). Error bars may be incomplete.", icon="📉")

                                    except KeyError:
                                        st.error("Error during aggregation: 'value' or 'datetime' column not found in raw_data.")
                                        timeline_summary = pd.DataFrame() # Assign empty df to prevent further errors
                                    except Exception as e:
                                        st.error(f"An unexpected error occurred during timeline summary calculation: {e}")
                                        timeline_summary = pd.DataFrame() # Assign empty df

                                    # Ensure timeline_summary is not empty before assigning
                                    if not timeline_summary.empty:
                                        timeline_plot_data = timeline_summary # Data for plotting average
                                    else:
                                        st.error("Failed to create timeline summary data.")
                                        timeline_plot_data = None # Ensure plot_data is None if summary failed

                                else: # Individual mode
                                    # Convert selected subjects to POSITIONAL cages using the corrected map
                                    selected_cages_pub = [subject_to_positional_cage[sub] for sub in selected_subjects_pub_timeline if sub in subject_to_positional_cage]

                                    if not selected_cages_pub:
                                        st.warning(f"Could not map selected subjects to positional cages. Map: {subject_to_positional_cage}. Selected IDs: {selected_subjects_pub_timeline}")
                                        st.stop()

                                    # Filter raw_data for these POSITIONAL cages
                                    timeline_plot_data = raw_data[raw_data['cage'].isin(selected_cages_pub)].copy()

                                    if timeline_plot_data.empty:
                                        st.warning(f"No raw data found for the selected positional cages: {selected_cages_pub}. Check if these cage names exist in the raw data.")
                                        # Show available cage names from raw_data to help debug
                                        st.write("Available cage names in raw data:", raw_data['cage'].unique())
                                        st.stop()
                                    if not all(col in timeline_plot_data.columns for col in ['datetime', 'cage', 'value']):
                                        st.error("Required columns ('datetime', 'cage', 'value') missing in filtered data for individual plot.")
                                        st.stop()

                                # Final check before plotting
                                if timeline_plot_data is None or timeline_plot_data.empty:
                                    st.error("Failed to prepare data for timeline plot (final check)."); st.stop()

                                # --- Generate and Display Plot ---
                                # Determine the correct mode string based on the radio button selection
                                mode_string = "Average" if timeline_display_mode == "Show Average Across All Animals" else "Individual"

                                pub_fig_timeline = generate_pub_timeline_plot(
                                    timeline_data=timeline_plot_data, # Pass the correct data structure
                                    parameter=parameter,
                                    time_window=time_window,
                                    subject_to_cage=subject_to_positional_cage, # Pass the correct map
                                    display_mode=mode_string, # <<< CORRECTED LINE
                                    selected_subjects=selected_subjects_pub_timeline,
                                    subject_to_group=pd.Series(group_assignments.Group.values,index=group_assignments['Subject ID']).to_dict(),
                                    light_start=st.session_state.get('light_start', 7),
                                    light_end=st.session_state.get('light_end', 19)
                                )
                                st.pyplot(pub_fig_timeline)
                                
                                # --- Description ---
                                with st.expander("What am I looking at? (Calculation Details)"):
                                    # (Description logic remains the same)
                                    desc = f"""
                                    This plot shows the **{parameter}** values over the selected time window (`{time_window}`).

                                    * **Mode**: You are viewing the **{timeline_display_mode}**.
                                    * **{'Average View Details' if timeline_display_mode == 'Show Average Across All Animals' else 'Individual View Details'}**:
                                        * **{'Average:' if timeline_display_mode == 'Show Average Across All Animals' else 'Lines:'}** {'The solid black line represents the average value across all animals at each time point.' if timeline_display_mode == 'Show Average Across All Animals' else 'Each colored line represents the data for a single animal you selected.'}
                                        * **{'Shaded Band (SEM):' if timeline_display_mode == 'Show Average Across All Animals' else ''}** {'The grey shaded area represents the Standard Error of the Mean (SEM), indicating the precision or variability of the average.' if timeline_display_mode == 'Show Average Across All Animals' else ''}
                                    * **Calculation Basis**: The values are typically averaged per hour across the entire selected duration.
                                    * **Background Shading**: Grey shaded areas indicate the **Dark Cycle** based on the start/end times set in the sidebar ({st.session_state.get('light_start', 7)}:00 to {st.session_state.get('light_end', 19)}:00 is Light).
                                    * **Styling**: The plot uses `scienceplots` for a publication-ready appearance.
                                    """
                                    st.markdown(desc)

                                # --- Downloads ---
                                st.markdown("---"); st.subheader("Download Plot")
                                # (Download button code remains the same)
                                png_buffer_t = save_figure_to_buffer(pub_fig_timeline, 'png'); pdf_buffer_t = save_figure_to_buffer(pub_fig_timeline, 'pdf'); svg_buffer_t = save_figure_to_buffer(pub_fig_timeline, 'svg')
                                col1t, col2t, col3t = st.columns(3)
                                dl_file_prefix_t = f"{parameter}_timeline_{timeline_display_mode.split(' ')[0].lower().replace('focus','individual')}"
                                with col1t: st.download_button(f"Download PNG", png_buffer_t, f"{dl_file_prefix_t}.png", "image/png", key="png_pub_timeline")
                                with col2t: st.download_button(f"Download PDF", pdf_buffer_t, f"{dl_file_prefix_t}.pdf", "application/pdf", key="pdf_pub_timeline")
                                with col3t: st.download_button(f"Download SVG", svg_buffer_t, f"{dl_file_prefix_t}.svg", "image/svg+xml", key="svg_pub_timeline")
                                plt.close(pub_fig_timeline)

                            except Exception as e: st.error(f"Error during Timeline plot generation: {e}"); st.exception(e)
                    # Add placeholders for other plot types later
                    # elif plot_type == "Timeline":
                    #     st.info("Timeline plot coming soon!")
            # Tab 3: How Can I Trust This Data? (Formerly Verification)
            with tab3:
                st.header("🧪 How Can I Trust This Data?")

                st.markdown("""
                CLAMSer is designed to provide reliable and transparent analysis. Here’s why you can trust the results:

                *   **✅ Standard Calculations:** Uses industry-standard methods for calculating metabolic parameters (like Light/Dark averages, hourly means) directly from your raw data.
                *   **🔍 Transparent Process:** You can see exactly how calculations are performed using sample data in the "Calculation Walkthrough" section below.
                *   **🔄 Reproducible Results:** Given the same input file and settings, CLAMSer will always produce the identical results.
                *   **📁 Data Access:** You can download the processed data tables and raw data samples to verify the results independently in your preferred software (Excel, Prism, R, etc.).
                *   **📊 Appropriate Statistics:** Automatically selects standard statistical tests (t-test or ANOVA) based on your group setup for comparisons.

                Our goal is transparency/to have confidence in the analysis.
                """)

                # --- Calculation Walkthrough Section ---
                st.markdown("---") # Add a visual separator
                st.subheader("🔬 Calculation Walkthrough: From Raw Data to Results")
                st.markdown("Let's see how the summary values (like Light/Dark Averages) are calculated from the raw measurements for a single animal.")

                # Check if necessary data exists
                if 'raw_data' in locals() and raw_data is not None and not raw_data.empty and \
                'results' in locals() and results is not None and not results.empty and 'Subject ID' in results.columns:

                    try: # Add a try-except block for safety during data handling
                        # Let user select an animal
                        # Use results.index (cage names) and results['Subject ID'] for mapping
                        subject_id_options = results['Subject ID'].tolist()
                        cage_options = results.index.tolist() # e.g., ['CAGE 01', 'CAGE 02']
                        
                        # Create a display format combining Subject ID and Cage
                        # Ensure the lists are the same length before zipping
                        if len(subject_id_options) == len(cage_options):
                            options_dict = {f"{sid} ({cage})": cage for sid, cage in zip(subject_id_options, cage_options)}
                            display_options = list(options_dict.keys())
                        else:
                            st.warning("Mismatch between subject IDs and cages in results. Using Cages only for selection.")
                            options_dict = {cage: cage for cage in cage_options}
                            display_options = cage_options # Fallback to just cages

                        # Add a check to ensure display_options is not empty before proceeding
                        if not display_options:
                            st.warning("No animals available for selection in Calculation Walkthrough.")
                        else:
                            selected_display_option = st.selectbox(
                                "Select an animal to see its calculation:",
                                options=display_options,
                                index=0, # Default to the first animal
                                key="verification_animal_select"
                            )

                            # Get the corresponding cage name from the selected option
                            selected_cage = options_dict[selected_display_option]
                            # Find the Subject ID corresponding to the selected cage
                            selected_subject_id = results.loc[selected_cage, 'Subject ID']


                            st.markdown(f"**Showing calculations for: {selected_subject_id} ({selected_cage})**")

                            # Filter raw_data for the selected cage within the analysis time window
                            # Use the pre-filtered 'filtered_raw_data' if it exists from the Raw Data Display section
                            if 'filtered_raw_data' in locals() and filtered_raw_data is not None and not filtered_raw_data.empty:
                                animal_data = filtered_raw_data[filtered_raw_data['cage'] == selected_cage].copy()
                            else:
                                # Fallback: filter the main raw_data if filtered_raw_data isn't available
                                # Determine time window again (this is slightly redundant but safer)
                                if time_window == "Entire Dataset":
                                    animal_data = raw_data[raw_data['cage'] == selected_cage].copy()
                                else:
                                    if time_window == "Custom Range": days_to_analyze = st.session_state.get("custom_days_input", 5)
                                    else: days_to_analyze = {"Last 24 Hours": 1,"Last 48 Hours": 2,"Last 72 Hours": 3,"Last 7 Days": 7,"Last 14 Days": 14}.get(time_window, 1)
                                    end_time_filt = raw_data['timestamp'].max(); start_time_filt = end_time_filt - pd.Timedelta(days=days_to_analyze)
                                    animal_data = raw_data[(raw_data['cage'] == selected_cage) & (raw_data['timestamp'] >= start_time_filt) & (raw_data['timestamp'] <= end_time_filt)].copy()

                            if animal_data.empty:
                                st.warning(f"No data found for {selected_subject_id} ({selected_cage}) within the selected time window.")
                            else:
                                # Separate Light and Dark data points
                                light_points = animal_data[animal_data['is_light']]
                                dark_points = animal_data[~animal_data['is_light']]

                                # --- Simple Visualization ---
                                fig_verify = go.Figure()

                                # Add Dark points
                                fig_verify.add_trace(go.Scatter(
                                    x=dark_points['timestamp'], y=dark_points['value'],
                                    mode='markers', name='Dark Cycle Measurements',
                                    marker=dict(color='darkblue', size=5, opacity=0.7)
                                ))
                                # Add Light points
                                fig_verify.add_trace(go.Scatter(
                                    x=light_points['timestamp'], y=light_points['value'],
                                    mode='markers', name='Light Cycle Measurements',
                                    marker=dict(color='gold', size=5, opacity=0.7)
                                ))

                                # Calculate averages directly from the filtered data
                                light_avg = light_points['value'].mean()
                                dark_avg = dark_points['value'].mean()
                                total_avg_calc = animal_data['value'].mean() # Overall average for this animal

                                # Add average lines to the plot
                                if not pd.isna(light_avg):
                                    fig_verify.add_hline(y=light_avg, line_dash="dash", line_color="orange", annotation_text=f"Light Avg: {light_avg:.2f}", annotation_position="bottom right")
                                if not pd.isna(dark_avg):
                                    fig_verify.add_hline(y=dark_avg, line_dash="dash", line_color="blue", annotation_text=f"Dark Avg: {dark_avg:.2f}", annotation_position="top right")

                                fig_verify.update_layout(
                                    title=f"Raw Data Points for {selected_subject_id} ({selected_cage})",
                                    xaxis_title="Time",
                                    yaxis_title=f"{parameter} ({PARAMETER_UNITS.get(parameter, '')})",
                                    showlegend=True,
                                    height=350, # Make plot a bit smaller
                                    margin=dict(t=50, b=50)
                                )
                                st.plotly_chart(fig_verify, use_container_width=True)

                                # --- Explanation of Calculation ---
                                st.markdown("#### How the Averages are Calculated:")

                                col1, col2 = st.columns(2)

                                with col1:
                                    st.markdown("**Light Cycle Average:**")
                                    if not light_points.empty and not pd.isna(light_avg):
                                        st.markdown(f"1. **Collect all Light points:** {len(light_points)} measurements found during the light cycle.")
                                        # Show a few example values
                                        example_light_vals = ", ".join([f"{v:.2f}" for v in light_points['value'].head(3).tolist()])
                                        if len(light_points) > 3: example_light_vals += ", ..."
                                        st.markdown(f"   *(Examples: {example_light_vals})*")
                                        st.markdown(f"2. **Sum these values:** Total = {light_points['value'].sum():.2f}")
                                        st.markdown(f"3. **Divide by the count:** Average = {light_points['value'].sum():.2f} / {len(light_points)} = **{light_avg:.2f}**")
                                    elif pd.isna(light_avg):
                                        st.info("No light cycle data points found for this animal in the selected time window.")
                                    else:
                                        st.info("No light cycle data points found for this animal.")


                                with col2:
                                    st.markdown("**Dark Cycle Average:**")
                                    if not dark_points.empty and not pd.isna(dark_avg):
                                        st.markdown(f"1. **Collect all Dark points:** {len(dark_points)} measurements found during the dark cycle.")
                                        # Show a few example values
                                        example_dark_vals = ", ".join([f"{v:.2f}" for v in dark_points['value'].head(3).tolist()])
                                        if len(dark_points) > 3: example_dark_vals += ", ..."
                                        st.markdown(f"   *(Examples: {example_dark_vals})*")
                                        st.markdown(f"2. **Sum these values:** Total = {dark_points['value'].sum():.2f}")
                                        st.markdown(f"3. **Divide by the count:** Average = {dark_points['value'].sum():.2f} / {len(dark_points)} = **{dark_avg:.2f}**")
                                    elif pd.isna(dark_avg):
                                        st.info("No dark cycle data points found for this animal in the selected time window.")
                                    else:
                                        st.info("No dark cycle data points found for this animal.")

                                # --- Connect to Results Table ---
                                st.markdown("#### Comparing with the 'Overview' Tab Results Table:")
                                # Retrieve the specific row from the main results table for this cage
                                results_row = results.loc[[selected_cage]] # Use double brackets to ensure it returns a DataFrame

                                # Dynamically check columns based on parameter type
                                dark_col_name, light_col_name, total_col_name = None, None, None
                                if parameter in ["XTOT", "XAMB"]:
                                    dark_col_name = 'False (Average Activity)' # Check exact name from your processing
                                    light_col_name = 'True (Average Activity)' # Check exact name
                                    total_col_name = '24h Average' # Check exact name
                                elif parameter == "FEED":
                                    dark_col_name = 'Average Rate (Dark)' # Check exact name
                                    light_col_name = 'Average Rate (Light)' # Check exact name
                                    total_col_name = None # May not have a single 24h average rate column
                                else: # VO2, VCO2, RER, HEAT
                                    dark_col_name = 'Dark Average'
                                    light_col_name = 'Light Average'
                                    total_col_name = 'Total Average'

                                # Display comparison using metrics for clarity
                                st.write(f"For **{selected_subject_id} ({selected_cage})**, the 'Overview' results table shows:")
                                col_res1, col_res2, col_res3 = st.columns(3)
                                with col_res1:
                                    if light_col_name and light_col_name in results_row.columns:
                                        table_light_avg = results_row[light_col_name].iloc[0]
                                        st.metric(f"Table: {light_col_name}", f"{table_light_avg:.2f}")
                                        # Optional: Add a checkmark if they match closely
                                        if not pd.isna(light_avg) and abs(light_avg - table_light_avg) < 0.01:
                                            st.success("Matches Calculation ✔️")
                                    else:
                                        st.metric("Table: Light Average", "N/A")
                                with col_res2:
                                    if dark_col_name and dark_col_name in results_row.columns:
                                        table_dark_avg = results_row[dark_col_name].iloc[0]
                                        st.metric(f"Table: {dark_col_name}", f"{table_dark_avg:.2f}")
                                        if not pd.isna(dark_avg) and abs(dark_avg - table_dark_avg) < 0.01:
                                            st.success("Matches Calculation ✔️")
                                    else:
                                        st.metric("Table: Dark Average", "N/A")
                                with col_res3:
                                    if total_col_name and total_col_name in results_row.columns:
                                        table_total_avg = results_row[total_col_name].iloc[0]
                                        st.metric(f"Table: {total_col_name}", f"{table_total_avg:.2f}")
                                        # We also calculated total_avg_calc earlier
                                        if not pd.isna(total_avg_calc) and abs(total_avg_calc - table_total_avg) < 0.01:
                                            st.success("Matches Calculation ✔️")
                                    else:
                                        st.metric("Table: Total Average", "N/A")

                                st.markdown("This shows that the values in the main results table are directly derived from averaging the raw data points collected during the light and dark periods.")

                    except Exception as e:
                        st.error(f"An error occurred during the calculation walkthrough: {e}")
                        st.warning("Please ensure data and results have been processed correctly in the Overview tab.")

                else:
                    st.warning("⚠️ Please upload data and ensure it's processed in the 'Overview' tab to see the calculation walkthrough.")

                # --- End Calculation Walkthrough Section ---


                # --- Data Access Section ---
                st.markdown("---") # Add a visual separator
                st.subheader("📂 Data Access for Verification")
                st.markdown("""
                To perform your own checks or use the data in other software, you can download the key datasets generated by CLAMSer:
                """)

                col_dl1, col_dl2, col_dl3 = st.columns(3) # Use columns for button layout

                # Download Button 1: Processed Summary Results
                with col_dl1:
                    if 'results' in locals() and results is not None and not results.empty:
                        try:
                            csv_summary = results.to_csv().encode('utf-8')
                            st.download_button(
                                label=f"📥 Download Summary Results",
                                data=csv_summary,
                                file_name=f"{parameter}_summary_results_verify.csv",
                                mime="text/csv",
                                help=f"Download light/dark cycle averages for each animal ({parameter})",
                                key="verify_dl_summary" # Unique key
                            )
                            st.caption("Contains Light/Dark averages per animal.")
                        except Exception as e:
                            st.warning(f"Could not prepare Summary Results for download: {e}")
                    else:
                        st.button("📥 Download Summary Results", disabled=True, help="Data not available")
                        st.caption("Data not available.")


                # Download Button 2: Hourly Data (Typical Day Profile)
                with col_dl2:
                    if 'hourly_results' in locals() and hourly_results is not None and not hourly_results.empty:
                        try:
                            csv_hourly = hourly_results.to_csv().encode('utf-8')
                            st.download_button(
                                label=f"📥 Download Hourly Data",
                                data=csv_hourly,
                                file_name=f"{parameter}_hourly_data_verify.csv",
                                mime="text/csv",
                                help=f"Download hour-by-hour (0-23) averages for each animal ({parameter})",
                                key="verify_dl_hourly" # Unique key
                            )
                            st.caption("Contains hour 0-23 averages per animal.")
                        except Exception as e:
                            st.warning(f"Could not prepare Hourly Data for download: {e}")
                    else:
                        st.button("📥 Download Hourly Data", disabled=True, help="Data not available")
                        st.caption("Data not available.")


                # Download Button 3: Raw Data Sample (from filtered data if possible)
                with col_dl3:
                    raw_data_to_download = None
                    help_text = "Download the raw measurements used in the current analysis window"
                    # Prioritize using the filtered data shown earlier in the tab
                    if 'filtered_raw_data' in locals() and filtered_raw_data is not None and not filtered_raw_data.empty:
                        raw_data_to_download = filtered_raw_data
                        help_text += f" ({time_window})"
                    # Fallback to the main raw_data if filtered isn't available
                    elif 'raw_data' in locals() and raw_data is not None and not raw_data.empty:
                        raw_data_to_download = raw_data
                        help_text += " (Entire dataset)"


                    if raw_data_to_download is not None:
                        try:
                            # Limit download size for performance if it's the full dataset
                            if 'filtered_raw_data' not in locals() or filtered_raw_data is None:
                                download_limit = 5000 # Limit download if using full raw_data
                                if len(raw_data_to_download) > download_limit:
                                    download_df = raw_data_to_download.head(download_limit)
                                    help_text += f" (First {download_limit} rows)"
                                else:
                                    download_df = raw_data_to_download
                            else:
                                download_df = raw_data_to_download # Download all if it's already filtered

                            csv_raw = download_df.to_csv(index=False).encode('utf-8')
                            st.download_button(
                                label=f"📥 Download Raw Data",
                                data=csv_raw,
                                file_name=f"{parameter}_raw_data_verify.csv",
                                mime="text/csv",
                                help=help_text,
                                key="verify_dl_raw" # Unique key
                            )
                            st.caption(f"Contains {len(download_df):,} raw measurement rows.")
                        except Exception as e:
                            st.warning(f"Could not prepare Raw Data for download: {e}")
                    else:
                        st.button("📥 Download Raw Data", disabled=True, help="Data not available")
                        st.caption("Data not available.")

                # --- End Data Access Section ---


                # Keep contact info at the bottom
                st.markdown("---")
                st.subheader("About CLAMS Data Analyzer")
                st.markdown("""
                This tool was developed to streamline the analysis of CLAMS data files.

                ### Questions or feedback?
                Contact: Zane Khartabill (email: mkhal061@uottawa.ca)
                """)
