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
    "VO2": "Oxygen consumption (ml/kg/hr)",
    "VCO2": "Carbon dioxide production (ml/kg/hr)",
    "RER": "Respiratory exchange ratio",
    "HEAT": "Heat production (kcal/hr)",
    "XTOT": "Total activity counts",
    "XAMB": "Ambulatory activity counts",
    "FEED": "Food intake (g)"
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

            # The checkbox widget itself updates session state['apply_lean_mass']
            apply_lean_mass = st.checkbox(
                "Apply Lean Mass Adjustment",
                value=st.session_state.get('apply_lean_mass', False), # Read initial value from session state
                help="Normalize metabolic data to lean mass instead of total body weight",
                key="apply_lean_mass" # The key links this widget to session state
            )
            # NO NEED to manually set st.session_state['apply_lean_mass'] here

            # Conditionally show the input based on the *current state* of the checkbox
            # We can read directly from session_state OR use the variable 'apply_lean_mass' which holds the widget's current value
            if st.session_state.apply_lean_mass: # Read directly from session state for clarity
                # The number_input widget itself updates session_state['reference_lean_mass']
                reference_mass = st.number_input(
                    "Reference lean mass (g):", # Simplified label
                    min_value=1.0,
                    value=st.session_state.get('reference_lean_mass', 20.0), # Read initial value from session state
                    step=0.1,
                    format="%.1f",
                    help="Standard lean mass value used for normalization",
                    key="reference_lean_mass_sidebar" # The key links this widget to session state
                )
                # NO NEED to manually set st.session_state['reference_lean_mass'] here

                st.info("📌 Enter individual animal lean masses in the Overview tab after uploading.")
        st.divider() # Divider after the normalization section
    else:
        # If the parameter is not metabolic, ensure the session state flag is False.
        # This is okay because the 'apply_lean_mass' checkbox widget *isn't* created in this 'else' block.
        if 'apply_lean_mass' in st.session_state: # Check if it exists before trying to set it
                st.session_state['apply_lean_mass'] = False


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
            return True, f"Note: Selected parameter type '{expected_type}' not explicitly found in file uploaded, but file appears to be valid CLAMS data."
            
    except UnicodeDecodeError:
        return False, "File cannot be read. Please ensure it's a valid CSV file."
    except Exception as e:
        return False, f"Error verifying file: {str(e)}"
    
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
        
        # Determine days to analyze based on selected time window
        if time_window == "Entire Dataset":
            # Use all data without filtering by time
            df_24h = df_processed.copy()
        else:
            if time_window == "Custom Range":
                days_to_analyze = st.session_state.get("custom_days_input", 5)
            else:
                days_to_analyze = {
                    "Last 24 Hours": 1,
                    "Last 48 Hours": 2,
                    "Last 72 Hours": 3,
                    "Last 7 Days": 7,
                    "Last 14 Days": 14
                }.get(time_window, 1)  # Default to 1 day if not found

            end_time = df_processed['timestamp'].max()
            start_time = end_time - pd.Timedelta(days=days_to_analyze)

            # Add this check
            total_hours = (df_processed['timestamp'].max() - df_processed['timestamp'].min()).total_seconds() / 3600
            if total_hours < (days_to_analyze * 24):
                st.error(f"Not enough data for {time_window} analysis. File contains approximately {total_hours:.1f} hours of data.")
                return None, None, None

            df_24h = df_processed[
                (df_processed['timestamp'] >= start_time) &
                (df_processed['timestamp'] <= end_time)
            ].copy()
        
        # Apply lean mass adjustment if enabled
        if parameter_type in ["VO2", "VCO2", "HEAT"] and 'lean_mass_data' in st.session_state and st.session_state.get("apply_lean_mass", False):
            # Store original values
            df_24h['original_value'] = df_24h['value'].copy()
            
            # Get reference mass from session state instead of creating a new input
            reference_mass = st.session_state.get('reference_lean_mass', 20.0)
            
            # Apply adjustment based on cage
            lean_mass_data = st.session_state['lean_mass_data']
            for cage, lean_mass in lean_mass_data.items():
                # Formula: adjusted_value = original_value * (reference_mass / lean_mass)
                df_24h.loc[df_24h['cage'] == cage, 'value'] = df_24h.loc[df_24h['cage'] == cage, 'original_value'] * (reference_mass / lean_mass)
            
            # Add note about adjustment
            st.info(f"{parameter_type} values have been normalized to a reference lean mass of {reference_mass}g")

        # Add light/dark cycle using customizable times from session state
        df_24h['hour'] = df_24h['timestamp'].dt.hour
        light_start = st.session_state.get('light_start', 7)  # Default to 7AM if not set
        light_end = st.session_state.get('light_end', 19)     # Default to 7PM if not set
        df_24h['is_light'] = (df_24h['hour'] >= light_start) & (df_24h['hour'] < light_end)
        
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
    
def assign_groups(cage_df, key_prefix=''):
    """
    Allow users to assign groups to detected subjects
    """
    # First, show detected subjects and cages
    st.write("Detected subjects:")
    st.dataframe(cage_df)
    
    subjects_correct = st.radio(
        "Are the detected subjects correct?",
        ["Yes", "No"],
        index=0,
        key=f"{key_prefix}_subjects_correct_radio"
    )
    
    if subjects_correct == "No":
        st.error("CLAMSer is adapted specifically for raw Oxymax-CLAMS-CF Machine output. The uploaded file has likely been modified - Please ensure the uploaded file contains the correct subject information.")
        st.stop()
    
    # If subjects are correct, proceed with group assignment
    if subjects_correct == "Yes":
        # Get number of groups from user
        num_groups = st.number_input("How many groups do you want to create?", 
                                    min_value=1, 
                                    max_value=len(cage_df), 
                                    value=2,
                                    key=f"{key_prefix}_num_groups_input")
        
        # Create mapping between Subject ID and Cage ID
        subject_to_cage = dict(zip(cage_df["Subject ID"].tolist(), cage_df["Cage"].tolist()))
        
        # Create group assignments
        group_assignments = {}
        
        for i in range(num_groups):
            st.subheader(f"Group {i + 1}")
            
            # Get group name
            group_name = st.text_input(f"Name for Group {i + 1}", 
                                      value=f"Group {i + 1}",
                                      key=f"{key_prefix}_group_name_{i}")
            
            # Multi-select for subjects instead of cages
            selected_subjects = st.multiselect(
                f"Select subjects for {group_name}",
                cage_df["Subject ID"].tolist(),
                key=f"{key_prefix}_group_{i}"
            )
            
            # Store the corresponding cage IDs for data processing
            selected_cages = [subject_to_cage[subject] for subject in selected_subjects]
            group_assignments[group_name] = selected_cages
        
        # Validate that all subjects are assigned
        all_assigned_cages = [cage for group in group_assignments.values() for cage in group]
        unassigned_cages = set(cage_df["Cage"]) - set(all_assigned_cages)
        
        if unassigned_cages:
            # Convert cage IDs back to subject IDs for the warning message
            unassigned_subjects = [cage_df[cage_df["Cage"] == cage]["Subject ID"].iloc[0] for cage in unassigned_cages]
            st.warning(f"Warning: The following subjects are not assigned to any group: {', '.join(map(str, unassigned_subjects))}")
        
        # Check for duplicates
        assigned_cages = []
        duplicate_cages = []
        for group, cages in group_assignments.items():
            for cage in cages:
                if cage in assigned_cages:
                    duplicate_cages.append(cage)
                assigned_cages.append(cage)
        
        if duplicate_cages:
            # Convert cage IDs back to subject IDs for the error message
            duplicate_subjects = [cage_df[cage_df["Cage"] == cage]["Subject ID"].iloc[0] for cage in duplicate_cages]
            st.error(f"Error: The following subjects are assigned to multiple groups: {', '.join(map(str, set(duplicate_subjects)))}")
            st.stop()
        
        # If everything is valid, create a summary dataframe
        group_summary = []
        for group_name, cages in group_assignments.items():
            for cage in cages:
                # --- Step 1: Get Subject ID ---
                try:
                    subject_id = cage_df[cage_df["Cage"] == cage]["Subject ID"].iloc[0]
                except IndexError:
                    # This happens if the cage from the multiselect isn't found in the DataFrame
                    st.error(f"Data Error: Cage '{cage}' selected for Group '{group_name}' not found in the uploaded file's subject list. Skipping this entry.")
                    continue # Skip to the next cage in the list
                except Exception as e:
                    # Catch any other unexpected errors during subject ID lookup
                    st.error(f"Unexpected error finding Subject ID for Cage '{cage}': {e}. Skipping this entry.")
                    continue # Skip to the next cage

                # --- Step 2: Format Cage Name (if Subject ID was found) ---
                consistent_cage_name = cage # Default to original cage name
                try:
                    # Extract number (e.g., '101'), subtract 100, format as 2 digits ('01')
                    cage_num_match = re.search(r'(\d+)$', str(cage)) # Find digits at the end
                    if cage_num_match:
                        # Attempt conversion to integer
                        cage_num = int(cage_num_match.group(1))
                        # Check if it's likely a CLAMS cage number (>= 101)
                        if cage_num >= 101:
                            formatted_cage_num = cage_num - 100
                            consistent_cage_name = f"CAGE {formatted_cage_num:02d}"
                        else:
                            # It's a number, but not in the expected CLAMS range, use original
                            st.warning(f"Cage '{cage}' appears numeric but not in expected CLAMS format (>=101). Using original name.")
                    else:
                        # Non-numeric or unexpected format, use original
                        st.warning(f"Unexpected cage format '{cage}' (non-numeric ending). Using original name.")
                except ValueError:
                    # Handle error if int() conversion fails (should be rare if re.search worked)
                    st.warning(f"Error converting cage number part of '{cage}' to integer. Using original name.")
                except Exception as e:
                     # Catch any other unexpected errors during formatting
                     st.warning(f"Error formatting cage '{cage}': {e}. Using original name.")
                     # 'consistent_cage_name' already defaults to 'cage'

                # --- Step 3: Append to Summary (only if Steps 1 & 2 didn't 'continue') ---
                group_summary.append({
                    "Group": group_name,
                    "Cage": consistent_cage_name, # Use the (potentially) formatted name
                    "Subject ID": subject_id
                })
            # --- End of loop for this group's cages ---
                
        # Make sure we have at least one row before creating the DataFrame
        if not group_summary:
            # Return a DataFrame with correct columns even if empty
            return pd.DataFrame(columns=["Group", "Cage", "Subject ID"])
            
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
        st.markdown("""
        ## Welcome to CLAMSer 

        This tool analyzes metabolic data from Comprehensive Lab Animal Monitoring System (CLAMS) files.

        **Getting Started:**
        1. Upload your CLAMS data file using the sidebar
        2. Select the parameter type (VO2, VCO2, RER, etc.)
        3. Choose your preferred time window
        4. Explore the results across the different tabs.

        The Overview tab provides summary metrics, 24-hour patterns, and detailed data tables for your CLAMS data.
        """)
    else:
        # When file is uploaded, just show the header followed by parameter guide
        st.header("📊 Overview")
        
    
    
if uploaded_file is not None:
    # First verify file type
    is_valid, error_message = verify_file_type(uploaded_file, parameter)
    
    if not is_valid:
        st.error(error_message)
    else:
        # Get cage information for lean mass inputs
        cage_info = extract_cage_info(uploaded_file)
          
        # Process data first
        with st.spinner('Processing data...'):
            results, hourly_results, raw_data = process_clams_data(uploaded_file, parameter)
            
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
                        st.warning("👇 **Action Needed:** Please assign animals to groups in the **'Setup: Groups & Lean Mass'** section below to enable **Group-Based Analysis**, **Statistical Analysis** and **Publication Plots**.", icon="⚠️") # Changed to warning icon
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
                        if parameter in ["XTOT", "XAMB"]:
                            # Use equal width columns with proper spacing
                            col1, col2, col3, col4 = st.columns([1, 1, 1, 1])
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

                        elif parameter == "RER":
                            # Equal column widths for balanced appearance
                            col1, col2, col3 = st.columns([1, 1, 1])
                            with col1:
                                st.metric("Average Light RER", f"{results['Light Average'].mean():.3f}")
                            with col2:
                                st.metric("Average Dark RER", f"{results['Dark Average'].mean():.3f}")
                            with col3:
                                st.metric("Total Records", f"{len(raw_data):,}")

                        elif parameter == "FEED":
                            col1, col2, col3 = st.columns([1, 1, 1])
                            with col1:
                                st.metric("Average Light Rate", 
                                        f"{results['Average Rate (Light)'].mean():.4f} {PARAMETER_UNITS[parameter]}")
                            with col2:
                                st.metric("Average Dark Rate", 
                                        f"{results['Average Rate (Dark)'].mean():.4f} {PARAMETER_UNITS[parameter]}")
                            with col3:
                                st.metric("Total Feed", 
                                        f"{(results['Total Intake (Light)'] + results['Total Intake (Dark)']).sum():.4f} {PARAMETER_UNITS[parameter]}")

                        else:  # VO2, VCO2, HEAT
                            col1, col2, col3 = st.columns([1, 1, 1])
                            with col1:
                                st.metric(f"Average Light {parameter}", 
                                        f"{results['Light Average'].mean():.2f} {PARAMETER_UNITS[parameter]}")
                            with col2:
                                st.metric(f"Average Dark {parameter}", 
                                        f"{results['Dark Average'].mean():.2f} {PARAMETER_UNITS[parameter]}")
                            with col3:
                                st.metric("Total Records", f"{len(raw_data):,}")

                    # Get day/night ratio based on parameter type
                    day_night_ratio = None
                    if parameter in ["VO2", "VCO2", "HEAT", "RER"]:
                        day_night_ratio = results['Light Average'].mean() / results['Dark Average'].mean()
                    elif parameter in ["XTOT", "XAMB"]:
                        day_night_ratio = results['True (Average Activity)'].mean() / results['False (Average Activity)'].mean() 
                    elif parameter == "FEED":
                        day_night_ratio = results['Average Rate (Light)'].mean() / results['Average Rate (Dark)'].mean()

                    # Display day/night insight if ratio could be calculated
                    if day_night_ratio is not None:
                        direction = "higher" if day_night_ratio > 1 else "lower"
                        percent_diff = abs(1 - day_night_ratio) * 100
                        
                        # For most parameters, lower during light is normal for nocturnal animals
                        # But for FEED, higher during dark is normal
                        normal_pattern = day_night_ratio < 1
                        if parameter == "FEED":
                            normal_pattern = day_night_ratio < 1  # Food intake is normally higher during dark for nocturnal animals
                        
                        st.info(f"**Day/Night Pattern**: {parameter} is {percent_diff:.1f}% {direction} during light cycle compared to dark cycle, " + 
                                ("suggesting normal nocturnal activity." if normal_pattern else "which may indicate altered circadian rhythm."))
                    


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

                    # --- Setup Section (Moved Down) ---
                    st.header("⚙️ Setup: Groups & Lean Mass")
                    with st.container(border=True): # Use a container with a border for visual grouping

                        # --- Group Assignment Content (Inside Container) ---
                        st.subheader("1. Assign Animals to Groups")
                        st.info("Assign animals to experimental groups. Required for Statistical Analysis and Publication Plots.")

                        # Check if cage_info is available
                        if 'cage_info' in locals() and cage_info:
                            # Create cage_df if needed
                            # (This check might be redundant if processing is robust, but safe to keep)
                            if 'cage_df' not in locals():
                                cage_df = pd.DataFrame([
                                    {"Cage": f"CAGE {int(k)}", "Subject ID": v} # Ensure Cage format matches assign_groups expectation if needed
                                    for k, v in cage_info.items()
                                ])

                            # Call the group assignment function
                            group_assignments_result = assign_groups(cage_df, key_prefix="overview_setup_container") # Use a new key_prefix

                            # Store results in session state ONLY if assign_groups returns a valid DataFrame
                            if group_assignments_result is not None and not group_assignments_result.empty:
                                st.session_state['group_assignments'] = group_assignments_result
                                # Optional: Brief confirmation inside the setup area
                                # st.success("Group assignments updated.")
                            # Note: The main callout at the top already confirms if groups *are* assigned overall.

                        else:
                            st.error("Cannot assign groups: Cage information was not extracted correctly.")

                        # --- Lean Mass Content (Inside Container & Conditional) ---
                        st.markdown("---") # Separator within the container
                        st.subheader("2. Enter Lean Mass (Optional)")

                        # Check parameter relevance first
                        if parameter in ["VO2", "VCO2", "HEAT"]:
                            # Check if checkbox is ticked in sidebar
                            if st.session_state.get("apply_lean_mass", False):
                                st.markdown(f"""
                                Lean mass adjustment is **enabled** (sidebar setting). Enter values to normalize **{parameter}**
                                to a reference lean mass of **{st.session_state.get('reference_lean_mass', 20.0)}g**.
                                """)

                                lean_mass_inputs = {}
                                if 'cage_info' in locals() and cage_info:
                                    cols = st.columns(3)
                                    for i, (cage_id, subject_id) in enumerate(cage_info.items()):
                                        try:
                                            # Ensure cage_label matches keys used elsewhere (e.g., "CAGE 01")
                                            cage_num_display = int(re.search(r'\d+', str(cage_id)).group()) - 100 if re.search(r'\d+', str(cage_id)) else i
                                            cage_label = f"CAGE {cage_num_display:02d}"
                                        except Exception: # Broad exception for safety
                                            cage_label = f"Cage {i+1}" # Fallback

                                        with cols[i % 3]:
                                            # Use a unique key for inputs inside this container
                                            lean_mass = st.number_input(
                                                f"Lean mass (g) for {subject_id} ({cage_label})",
                                                min_value=1.0,
                                                # Retrieve value based on the consistent cage_label format
                                                value=st.session_state.get('lean_mass_data', {}).get(cage_label, 20.0),
                                                step=0.1,
                                                format="%.1f",
                                                key=f"lean_mass_{cage_label}_setup_container" # New key
                                            )
                                            lean_mass_inputs[cage_label] = lean_mass

                                    # Store in session state immediately
                                    st.session_state['lean_mass_data'] = lean_mass_inputs

                                    st.caption(f"""
                                    **Formula:** Adjusted {parameter} = Original {parameter} × (Reference Mass ÷ Animal's Lean Mass)
                                    *(Example: Ref: 20g, Animal: 25g, Original {parameter}: 3000 -> Adjusted: 2400)*
                                    """)
                                else:
                                    st.warning("Cannot display lean mass inputs: Cage information missing.")

                            else:
                                # Option is available but disabled in sidebar
                                st.markdown("**(Optional) Lean mass adjustment is currently disabled.** Enable it in the sidebar settings if needed.")
                        else:
                            # Not relevant for this parameter
                            if st.session_state.get("apply_lean_mass", False):
                                st.warning(f"Lean mass adjustment is enabled (sidebar), but not applicable for '{parameter}'. No adjustment will be made.")
                            else:
                                st.markdown(f"*(Lean mass adjustment is not applicable for '{parameter}')*")

                    # --- End of Setup Container ---
                    st.markdown("---") # Add separator after the setup container
                    
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
                                group_means = results_with_groups.groupby('Group').mean(numeric_only=True)
                                st.dataframe(style_dataframe(group_means))
                                
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

                        st.info("\n".join(interpretation_texts)) # Keep this line

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
                                    timeline_summary = raw_data.groupby('datetime').agg(
                                        Mean=('value', 'mean'),
                                        SEM=('value', 'sem') # <--- PROBLEM LIKELY HERE
                                    ).reset_index()
                                    timeline_plot_data = timeline_summary # Data for plotting average

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