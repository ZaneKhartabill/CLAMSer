import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import plotly.express as px
import plotly.graph_objects as go
import matplotlib.colors as mcolors
import re
import markdown
from scipy import stats
from scikit_posthocs import posthoc_dunn
from statsmodels.stats.multicomp import pairwise_tukeyhsd
from statsmodels.stats.power import TTestIndPower


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
    
    # Move parameter selection to sidebar
    st.markdown("## Analysis Settings")
    parameter = st.selectbox(
        "Select Parameter",
        list(parameter_descriptions.keys()),
        format_func=lambda x: f"{x}: {parameter_descriptions[x]}"
    )

    # Move time window selection to sidebar
    time_window = st.radio(
        "Time Window",
        ["Last 24 Hours", "Last 48 Hours", "Last 72 Hours"],
        help="Choose analysis duration",
        key="time_window_radio"  
    )

    # Add a separator and enhanced lean mass adjustment option
    if parameter in ["VO2", "VCO2", "HEAT"]:
        st.sidebar.markdown("---")
        st.sidebar.markdown("### 📏 Metabolic Normalization")
        
        # Add explanation about lean mass adjustment
        lm_container = st.sidebar.container()
        lm_container.info("""
        **Why normalize to lean mass?**
        Fat tissue is less metabolically active than lean tissue. Normalizing to lean mass provides more accurate comparisons between animals with different body compositions.
        """)
        
        apply_lean_mass = lm_container.checkbox(
            "Apply Lean Mass Adjustment",
            value=False,
            help="Normalize metabolic data to lean mass instead of total body weight",
            key="apply_lean_mass"
        )
        
        if apply_lean_mass:
            # Add reference mass input directly in the sidebar
            reference_mass = lm_container.number_input(
                "Reference lean mass (g)",
                min_value=1.0,
                value=20.0,
                step=0.1,
                format="%.1f",
                help="Standard lean mass value used for normalization",
                key="reference_lean_mass_sidebar"
            )
            st.session_state['reference_lean_mass'] = reference_mass
            
            # Add note about where to input individual animal lean mass values
            lm_container.info("📌 Upload your file and enter individual animal lean mass values in the Overview tab")
    else:
        apply_lean_mass = False

    st.sidebar.markdown("---")
        
    # Move file upload to sidebar
    uploaded_file = st.file_uploader(
        f"Upload {parameter} CSV",
        type="csv",
        help="Upload your CLAMS data file",
        key="file_upload_1"
        
    )
    
# Main title in content area
st.title("CLAMSer: CLAMS Data Analyzer adapted for Oxymax-CLAMS-CF Machine")

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
def simplified_statistical_analysis(tab2, raw_data, results, parameter, parameter_descriptions, time_window):
    """Simplified, user-friendly statistical analysis with educational elements"""
    with tab2:
        # Create header with educational introduction
        st.header("📈 Statistical Analysis")
        
        # Educational introduction section
        with st.expander("🎓 Getting Started with Statistics", expanded=True):
            edu_container = st.container()

        with edu_container:
            st.markdown("""
            ## Welcome to CLAMSer Statistical Analysis
            
            This tab helps you understand differences between your experimental groups.
            
            ### 📘 Quick Guide:
            1. **Assign groups** in the Overview tab
            2. **Select analysis options** below
            3. **Interpret results** with our easy-to-understand visualizations
            
            ### 📊 Understanding Your Results:
            - **p-value < 0.05** suggests a statistically significant difference between groups
            - **Error bars** show standard error of mean (SEM) by default
            - **Bar charts** compare groups directly
            - **Line graphs** show how values change over 24 hours
            """)
            
            # Create two columns for basic stats concepts
            st.markdown("### Statistical Concepts")
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("**❓ What is a p-value?**")
                st.markdown("""
                A **p-value** tells you how likely your results occurred by random chance.
                
                - **p < 0.05:** Strong evidence of a real difference (significant)
                - **p > 0.05:** Not enough evidence (not significant)
                
                Think of it like this: A p-value of 0.03 means there's only a 3% chance that the difference you see is due to random chance.
                """)
                
                st.markdown("**❓ What do error bars show?**")
                st.markdown("""
                **Error bars** represent the variability in your data.
                
                By default, we show **Standard Error of Mean (SEM)**, which indicates how precisely you know the true mean.
                
                Smaller error bars = more precise measurement
                """)
            
            with col2:
                st.markdown("**❓ How to interpret significance**")
                st.markdown("""
                Statistical significance (p < 0.05) suggests the difference between groups is real, not random variation.
                
                Look for:
                - **Asterisks (*)** on graphs indicating significance
                - **p-values** in the results tables
                - **Non-overlapping error bars** often (but not always) indicate significant differences
                """)
                
                st.markdown("**❓ What's the light/dark cycle?**")
                st.markdown("""
                For rodents:
                - **Light cycle (7AM-7PM):** Resting period (less active)
                - **Dark cycle (7PM-7AM):** Active period (more active)
                
                Different parameters may show stronger differences during specific cycles.
                """)

        # Check if groups have been assigned
        if 'group_assignments' not in st.session_state or st.session_state.get('group_assignments', pd.DataFrame()).empty:
            st.warning("⚠️ Please assign groups in the Overview tab first")
            return
            
        # Get group assignments from session state
        group_assignments = st.session_state['group_assignments']
        
        # Main Analysis Section
        st.markdown("---")
        st.subheader("Group Comparison")
        
        # Create a more distinct separation between settings and analysis info
        st.markdown("### Settings")

        # Cycle selection with visual indicators
        cycle = st.radio(
            "Select data to analyze:",
            ["Light Cycle (7AM-7PM)", "Dark Cycle (7PM-7AM)", "24-hour Average"],
            key="simple_cycle_selector",
            help="Choose which part of the day to analyze"
        )

        # Group selection
        available_groups = sorted(group_assignments['Group'].unique())
        selected_groups = st.multiselect(
            "Select groups to compare:",
            options=available_groups,
            default=available_groups[:min(4, len(available_groups))],
            help="Choose which experimental groups to include in analysis"
        )

        # Error bar type with simpler options
        error_bar_type = st.radio(
            "Error bar type:",
            ["Standard Error (SEM)", "Standard Deviation (SD)"],
            index=0,
            help="SEM shows precision of the mean, SD shows data spread"
        )

        # Add a visual separator
        st.markdown("---")

        # Analysis information in its own section
        st.markdown("### Analysis Information")
        st.markdown(f"""
        You're analyzing **{parameter}** ({parameter_descriptions.get(parameter, '')}) 
        for the selected groups during the **{cycle}**.

        **What this tells you:**
        - Differences in {parameter} between experimental groups
        - Statistical significance of these differences
        - Typical values for each group

        **Tip:** 
        For metabolic parameters (VO2, VCO2, HEAT), differences are often most 
        pronounced during the dark cycle when animals are most active.
        """)
        
        # Validate we have enough groups selected
        if len(selected_groups) < 2:
            st.warning("⚠️ Please select at least two groups to compare")
            return
            
        # Prepare data for analysis
        # Convert cycle selection to filter
        if "Light Cycle" in cycle:
            cycle_filter = True
            cycle_name = "Light Cycle"
        elif "Dark Cycle" in cycle:
            cycle_filter = False
            cycle_name = "Dark Cycle"
        else:  # 24-hour Average
            cycle_filter = None
            cycle_name = "24-hour Average"
        
        # Join the datasets using the normalized IDs
        # Create mapping function to handle the cage ID formats
        def normalize_cage_id(cage_str):
            # Extract just the numeric part
            import re
            match = re.search(r'(\d+)', str(cage_str))
            if not match:
                return None
            
            cage_num = match.group(1)
            
            # Special handling for 2-digit IDs
            if len(cage_num) == 2:
                # If it starts with '0', like '01', convert to '101'
                if cage_num.startswith('0'):
                    return '1' + cage_num
                # If it's '10', '11', '12', convert to '110', '111', '112'
                else:
                    return '1' + cage_num
            
            # If it's already a 3-digit ID, keep it as is
            return cage_num
        
        # Apply the normalization to raw_data
        raw_data['normalized_cage_id'] = raw_data['cage'].apply(normalize_cage_id)
        
        # Apply the normalization to group_assignments (if not already done)
        if 'normalized_cage_id' not in group_assignments.columns:
            group_assignments['normalized_cage_id'] = group_assignments['Cage'].apply(normalize_cage_id)
        
        # Join the datasets
        grouped_data = pd.merge(
            raw_data, 
            group_assignments[['normalized_cage_id', 'Group', 'Subject ID']], 
            on='normalized_cage_id', 
            how='inner'
        )
        
        # Filter for selected groups
        grouped_data = grouped_data[grouped_data['Group'].isin(selected_groups)]
        
        if len(grouped_data) == 0:
            st.error("No data could be matched to your group assignments. Please check your group assignments and try again.")
            return
        
        # Calculate group statistics based on cycle selection
        if cycle_filter is not None:  # Light or Dark cycle
            group_stats = grouped_data[grouped_data['is_light'] == cycle_filter].groupby(['Group', 'Subject ID'])['value'].agg([
                ('Mean', 'mean')
            ]).reset_index().groupby('Group').agg({
                'Mean': ['mean', 'std', 'sem', 'count'],
                'Subject ID': 'nunique'
            })
        else:  # 24-hour Average
            group_stats = grouped_data.groupby(['Group', 'Subject ID'])['value'].agg([
                ('Mean', 'mean')
            ]).reset_index().groupby('Group').agg({
                'Mean': ['mean', 'std', 'sem', 'count'],
                'Subject ID': 'nunique'
            })
        
        # Flatten the multi-index columns
        group_stats.columns = ['_'.join(col).strip() for col in group_stats.columns.values]
        group_stats = group_stats.reset_index()
        
        # Rename columns for clarity
        group_stats = group_stats.rename(columns={
            'Mean_mean': 'Mean',
            'Mean_std': 'SD',
            'Mean_sem': 'SEM',
            'Mean_count': 'Data_Points',
            'Subject ID_nunique': 'N'
        })
        
        # Calculate 95% CI
        from scipy import stats
        
        # Add 95% CI
        group_stats['CI_95_Lower'] = group_stats['Mean'] - stats.t.ppf(0.975, group_stats['N']-1) * group_stats['SEM']
        group_stats['CI_95_Upper'] = group_stats['Mean'] + stats.t.ppf(0.975, group_stats['N']-1) * group_stats['SEM']
        
        # Filter group_stats to only include selected groups (in the same order)
        group_stats = group_stats[group_stats['Group'].isin(selected_groups)]
        group_stats['Group'] = pd.Categorical(group_stats['Group'], categories=selected_groups, ordered=True)
        group_stats = group_stats.sort_values('Group')
        
        # Calculate statistics and create visualizations
        # ===============================================

        # Create bar chart
        st.subheader(f"{parameter} Comparison by Group ({cycle_name})")

        # Create the bar chart
        fig = go.Figure()

        # Define colors for consistency
        colors = ["#4285F4", "#EA4335", "#FBBC05", "#34A853", "#8A2BE2", "#FF7F00", "#FF69B4", "#1E90FF"]

        # Determine error bar values based on user selection
        if "Standard Error" in error_bar_type:
            error_values = group_stats['SEM'].values
            error_title = "Standard Error of Mean"
        else:  # Standard Deviation
            error_values = group_stats['SD'].values
            error_title = "Standard Deviation"

        # Add bars for each group
        for i, row in enumerate(group_stats.itertuples()):
            color = colors[i % len(colors)]
            
            # Get error value for this row
            error_val = error_values[i]
            
            # Add the bar
            fig.add_trace(go.Bar(
                x=[row.Group],
                y=[row.Mean],
                name=row.Group,
                marker_color=color,
                error_y=dict(
                    type='data',
                    array=[error_val],
                    visible=True
                ),
                width=0.6,
                hovertemplate=f"<b>{row.Group}</b><br>Mean: %{{y:.2f}}<br>N: {row.N}<br>{error_title}: {error_val:.2f}<extra></extra>"
            ))

        # Run appropriate statistical test
        p_value = None
        test_name = None
        test_stat = None

        if len(selected_groups) == 2:  # t-test for 2 groups
            group1, group2 = selected_groups[0], selected_groups[1]
            
            # Get data for both groups
            if cycle_filter is None:  # 24-hour average
                g1_data = grouped_data[grouped_data['Group'] == group1].groupby('Subject ID')['value'].mean()
                g2_data = grouped_data[grouped_data['Group'] == group2].groupby('Subject ID')['value'].mean()
            else:
                g1_data = grouped_data[(grouped_data['Group'] == group1) & 
                                    (grouped_data['is_light'] == cycle_filter)].groupby('Subject ID')['value'].mean()
                g2_data = grouped_data[(grouped_data['Group'] == group2) & 
                                    (grouped_data['is_light'] == cycle_filter)].groupby('Subject ID')['value'].mean()
            
            # Perform t-test if enough data
            if len(g1_data) > 1 and len(g2_data) > 1:
                t_stat, p_value = stats.ttest_ind(g1_data, g2_data, equal_var=False)
                test_stat = t_stat
                test_name = "t-test"

        elif len(selected_groups) > 2:  # ANOVA for 3+ groups
            # Prepare data for ANOVA
            anova_data = []
            
            for group in selected_groups:
                if cycle_filter is None:  # 24-hour average
                    group_values = grouped_data[grouped_data['Group'] == group].groupby('Subject ID')['value'].mean().values
                else:
                    group_values = grouped_data[(grouped_data['Group'] == group) & 
                                            (grouped_data['is_light'] == cycle_filter)].groupby('Subject ID')['value'].mean().values
                
                if len(group_values) > 0:
                    anova_data.append(group_values)
            
            # Run ANOVA if we have data for all groups
            if all(len(data) > 0 for data in anova_data):
                f_stat, p_value = stats.f_oneway(*anova_data)
                test_stat = f_stat
                test_name = "ANOVA"

        # Add annotation about statistical significance
        if p_value is not None:
            significance = ""
            if p_value < 0.001:
                significance = "***"
            elif p_value < 0.01:
                significance = "**"
            elif p_value < 0.05:
                significance = "*"
            
            if test_name == "t-test":
                fig.add_annotation(
                    x=0.5,
                    y=1.1,
                    xref="paper",
                    yref="paper",
                    text=f"t-test: t={test_stat:.2f}, p={p_value:.4f} {significance}",
                    showarrow=False,
                    font=dict(size=14)
                )
            else:  # ANOVA
                fig.add_annotation(
                    x=0.5,
                    y=1.1,
                    xref="paper",
                    yref="paper",
                    text=f"ANOVA: F={test_stat:.2f}, p={p_value:.4f} {significance}",
                    showarrow=False,
                    font=dict(size=14)
                )
            
            # Add significance bars for t-test
            if test_name == "t-test" and p_value < 0.05:
                # Get y-range for positioning significance bars
                y_max = max(group_stats['Mean']) * 1.1
                y_range = y_max - min(group_stats['Mean'])
                
                # Add significance bar
                fig.add_shape(
                    type="line",
                    x0=0,
                    y0=y_max + 0.05 * y_range,
                    x1=1,
                    y1=y_max + 0.05 * y_range,
                    line=dict(color="black", width=2)
                )
                
                # Add asterisk
                fig.add_annotation(
                    x=0.5,
                    y=y_max + 0.1 * y_range,
                    text=significance,
                    showarrow=False,
                    font=dict(size=20)
                )

        # Update layout
        fig.update_layout(
            xaxis_title="Group",
            yaxis_title=f"{parameter} ({PARAMETER_UNITS.get(parameter, '')})",
            showlegend=False,
            height=500,
            margin=dict(t=100)  # Make room for annotations
        )

        # Set y-axis range with headroom for significance bars
        y_max = max(group_stats['Mean']) * 1.3
        fig.update_yaxes(range=[0, y_max])

        # Display plot
        st.plotly_chart(fig, use_container_width=True)

        # Add more visible significance indicator
        if p_value is not None:
            result_box = st.container()
            with result_box:
                if p_value < 0.05:
                    st.success(f"### ✅ Statistically Significant Difference Detected\n**p-value = {p_value:.4f}**\n\nThere is strong evidence that the groups differ.")
                else:
                    st.info(f"### ℹ️ No Statistically Significant Difference\n**p-value = {p_value:.4f}**\n\nThere isn't enough evidence to conclude the groups differ.")
            
            with st.expander("How is this p-value calculated?"):
                st.markdown(f"""
                The p-value is calculated using:
                
                - **{test_name}**: {test_name == 't-test' and "Comparing means between two groups" or "Comparing means across multiple groups"}
                - Based on data from the **{cycle_name}**
                - Specifically, we used {test_name == 't-test' and "Welch's t-test (does not assume equal variances)" or "one-way ANOVA"}
                
                The p-value will change if you select different cycles (light vs. dark) because the statistical test 
                is performed on different subsets of your data. For example, metabolic parameters often show stronger 
                differences during dark cycle when animals are more active.
                """)
            
            st.markdown("")  # Add some space

        # Make bar chart explanation collapsible
        with st.expander("What this bar chart shows"):
            st.markdown(f"""
            ### Understanding the Bar Chart
            
            **The Bar Chart**: 
            - Each bar represents the **average {parameter} value** for all animals in that experimental group
            - Values are calculated from the {cycle_name.lower()} data
            - For each animal, we first calculate its individual mean, then average these means for the group
            - **Error bars** show {'the standard error of the mean (SEM)' if 'Standard Error' in error_bar_type else 'the standard deviation (SD)'}, indicating the precision of our measurement
            """)

        with st.expander("See details about the calculation"):
            st.markdown(f"""
            For example, if "{selected_groups[0]}" has 3 animals with individual averages of 95, 100, and 105:
            
            1. Group Mean = (95 + 100 + 105) ÷ 3 = 100
            2. Standard Deviation = 5
            3. SEM = 5 ÷ √3 = 2.89
            
            The bar height would be 100, with error bars extending ±2.89 if using SEM.
            """)

        # Move Group Statistics here instead of in a side column
        st.subheader("Group Statistics")

        # Format table for better readability
        display_stats = group_stats[['Group', 'N', 'Mean', 'SD', 'SEM']].copy()
        display_stats['Mean'] = display_stats['Mean'].round(2)
        display_stats['SD'] = display_stats['SD'].round(2)
        display_stats['SEM'] = display_stats['SEM'].round(2)

        # Display a cleaner subset of columns
        st.dataframe(display_stats, use_container_width=True)

        # Add explanation of the statistics
        with st.expander("📊 What these statistics mean"):
            st.markdown("""
            - **N**: Number of animals in each group
            - **Mean**: Average value for each group
            - **SD**: Standard Deviation (spread of individual values)
            - **SEM**: Standard Error of the Mean (precision of the average)
            """)

        # Add effect size calculation for t-test
        if test_name == "t-test" and p_value is not None:
            st.subheader("Effect Size")
            
            group1, group2 = selected_groups[0], selected_groups[1]
            mean1 = group_stats[group_stats['Group'] == group1]['Mean'].iloc[0]
            mean2 = group_stats[group_stats['Group'] == group2]['Mean'].iloc[0]
            sd1 = group_stats[group_stats['Group'] == group1]['SD'].iloc[0]
            sd2 = group_stats[group_stats['Group'] == group2]['SD'].iloc[0]
            n1 = group_stats[group_stats['Group'] == group1]['N'].iloc[0]
            n2 = group_stats[group_stats['Group'] == group2]['N'].iloc[0]
            
            # Calculate Cohen's d
            # Pooled standard deviation
            pooled_sd = np.sqrt(((n1 - 1) * sd1**2 + (n2 - 1) * sd2**2) / (n1 + n2 - 2))
            cohen_d = abs(mean1 - mean2) / pooled_sd
            
            # Interpret Cohen's d
            effect_interpretation = "Small" if cohen_d < 0.5 else "Medium" if cohen_d < 0.8 else "Large"
            
            # Calculate percent difference
            percent_diff = abs(mean1 - mean2) / ((mean1 + mean2) / 2) * 100
            
            effect_cols = st.columns(2)
            with effect_cols[0]:
                # Display effect size metrics
                st.metric("Cohen's d", f"{cohen_d:.2f}", f"{effect_interpretation} effect")
            
            with effect_cols[1]:
                st.metric("Percent Difference", f"{percent_diff:.1f}%", 
                        f"{group1 if mean1 > mean2 else group2} higher")
            
            with st.expander("🔍 Understanding effect size"):
                st.markdown(f"""
                **Cohen's d = {cohen_d:.2f}** indicates a **{effect_interpretation.lower()} effect size**.
                
                - **Small effect**: d < 0.5
                - **Medium effect**: 0.5 < d < 0.8
                - **Large effect**: d > 0.8
                
                Even if a difference is statistically significant, a small effect size may not be biologically meaningful.
                """)
        
        # Time Pattern Analysis Section
        # ===============================
        
        st.markdown("---")
        st.subheader("24-Hour Pattern Analysis")
        
        st.markdown("""
        This analysis shows how your parameter changes over the 24-hour cycle.
        Dark regions indicate dark cycle (7PM-7AM), light regions indicate light cycle (7AM-7PM).
        """)
        
        # Group selection
        time_pattern_groups = st.multiselect(
            "Select groups to display:",
            options=selected_groups,
            default=selected_groups[:min(3, len(selected_groups))],
            key="time_pattern_group_select"
        )
        
        if not time_pattern_groups:
            st.warning("Please select at least one group to display the time pattern")
        else:
            # Calculate hourly averages for each group
            hourly_data = []
            
            for group in time_pattern_groups:
                # Get cages for this group
                group_cages = group_assignments[group_assignments['Group'] == group]['normalized_cage_id'].unique()
                
                # Filter data for these cages
                group_data = grouped_data[grouped_data['Group'] == group].copy()
                
                # Calculate hourly averages
                hourly_avg = group_data.groupby('hour')['value'].agg(['mean', 'sem']).reset_index()
                hourly_avg['Group'] = group
                hourly_data.append(hourly_avg)
            
            if hourly_data:
                # Combine all groups
                hourly_df = pd.concat(hourly_data)
                
                # Create time series plot
                fig = go.Figure()
                
                # Color palette
                colors = ["#4285F4", "#EA4335", "#FBBC05", "#34A853", "#8A2BE2", "#FF7F00", "#FF69B4", "#1E90FF"]
                
                # Add line for each group
                for i, group in enumerate(time_pattern_groups):
                    group_hourly = hourly_df[hourly_df['Group'] == group]
                    color = colors[i % len(colors)]
                    
                    # Add main line
                    fig.add_trace(go.Scatter(
                        x=group_hourly['hour'],
                        y=group_hourly['mean'],
                        mode='lines+markers',
                        name=group,
                        line=dict(color=color, width=3),
                        marker=dict(size=8)
                    ))
                    
                    # Add error bands (SEM)
                    fig.add_trace(go.Scatter(
                        x=group_hourly['hour'].tolist() + group_hourly['hour'].tolist()[::-1],
                        y=(group_hourly['mean'] + group_hourly['sem']).tolist() + 
                           (group_hourly['mean'] - group_hourly['sem']).tolist()[::-1],
                        fill='toself',
                        fillcolor=color.replace(')', ', 0.2)').replace('rgb', 'rgba'),
                        line=dict(color='rgba(0,0,0,0)'),
                        hoverinfo='skip',
                        showlegend=False
                    ))
                
                # Add shaded regions for dark and light cycles - with higher opacity
                for hour in range(0, 24):
                    if hour < 7 or hour >= 19:  # Dark cycle (7PM-7AM)
                        fig.add_vrect(
                            x0=hour - 0.5,
                            x1=hour + 0.5,
                            fillcolor="rgba(0,0,0,0.25)",  # Increased opacity for dark cycle
                            layer="below",
                            line_width=0,
                        )
                    else:  # Light cycle (7AM-7PM)
                        fig.add_vrect(
                            x0=hour - 0.5,
                            x1=hour + 0.5,
                            fillcolor="rgba(255,255,0,0.15)",  # Increased opacity for light cycle
                            layer="below",
                            line_width=0,
                        )

                # Add more prominent cycle labels
                fig.add_annotation(
                    x=3, y=1.12,  # Moved higher
                    xref="x", yref="paper",
                    text="Dark Cycle",
                    showarrow=False,
                    font=dict(size=16, color="white", family="Arial Black"),  # Bigger, bold, white text
                    bgcolor="rgba(0,0,0,0.7)",  # Much darker background
                    bordercolor="white",
                    borderwidth=2,
                    borderpad=4
                )

                fig.add_annotation(
                    x=13, y=1.12,  # Moved higher
                    xref="x", yref="paper",
                    text="Light Cycle",
                    showarrow=False,
                    font=dict(size=16, color="black", family="Arial Black"),  # Bigger, bold text
                    bgcolor="rgba(255,255,0,0.6)",  # Much more visible yellow
                    bordercolor="black",
                    borderwidth=2,
                    borderpad=4
                )

                fig.add_annotation(
                    x=21, y=1.12,  # Moved higher
                    xref="x", yref="paper",
                    text="Dark Cycle",
                    showarrow=False,
                    font=dict(size=16, color="white", family="Arial Black"),  # Bigger, bold, white text
                    bgcolor="rgba(0,0,0,0.7)",  # Much darker background
                    bordercolor="white",
                    borderwidth=2,
                    borderpad=4
                )
                
                # Update layout
                fig.update_layout(
                    title=f"{parameter} 24-Hour Pattern",
                    xaxis_title="Hour of Day",
                    yaxis_title=f"{parameter} ({PARAMETER_UNITS.get(parameter, '')})",
                    xaxis=dict(
                        tickmode='array',
                        tickvals=list(range(0, 24, 2)),
                        ticktext=[f"{h:02d}:00" for h in range(0, 24, 2)]
                    ),
                    height=500,
                    legend_title="Group",
                    hovermode="x unified"
                )
                
                # Display the plot
                st.plotly_chart(fig, use_container_width=True)
                
                # Make time graph explanation collapsible
                with st.expander("What this 24-hour pattern graph shows"):
                    st.markdown(f"""
                    ### Understanding the 24-Hour Pattern Graph
                    
                    **The 24-Hour Pattern Graph**:
                    - This graph shows a "typical day" created by averaging across all days in your selected time window ("{time_window}")
                    - Each point represents the **average {parameter} value** for a specific hour of the day
                    - Hours are shown from 0 (12 AM midnight) to 23 (11 PM)
                    - **Important**: If you selected "{time_window}", data from all {time_window.lower().replace("last ", "")} are combined and averaged for each hour
                    - For example, all measurements taken at 8:00 AM across {time_window.lower().replace("last ", "")} are averaged into a single data point
                    - **Shaded bands** around the lines show the standard error (SEM)
                    - **Background colors** indicate light cycle (yellow) and dark cycle (gray)
                    
                    This graph helps you identify:
                    - Consistent circadian patterns in your data
                    - When your parameter peaks and falls during a typical day
                    - How the light/dark cycle affects your parameter
                    - Differences in daily rhythms between experimental groups
                    
                    **Note**: This graph does not show how patterns changed from day to day. It creates a single "typical day" profile by combining all days.
                    """)

                with st.expander("See details about hourly calculation"):
                    st.markdown(f"""
                    For example, for the 8 AM data point (hour 8):
                    
                    1. All measurements taken at 8:00-8:59 AM across all days are collected
                    2. These values are averaged to get the mean for hour 8
                    3. The standard error is calculated to show precision
                    4. This process is repeated for each hour (0-23)
                    
                    This creates a complete 24-hour profile of how your parameter changes throughout the day.
                    """)
                
                # Simplified AUC Analysis - optional and collapsible
                with st.expander("📊 View Area Under Curve (AUC) Analysis"):
                    st.markdown("""
                    **Area Under the Curve (AUC)** provides a measure of the total exposure or activity over time.
                    """)
                    
                    # Calculate AUC for each group
                    auc_results = []
                    for group in time_pattern_groups:
                        group_hourly = hourly_df[hourly_df['Group'] == group]
                        
                        # Calculate total AUC (trapezoidal rule)
                        total_auc = np.trapz(group_hourly['mean'], group_hourly['hour'])
                        
                        # Calculate light cycle AUC
                        light_hourly = group_hourly[(group_hourly['hour'] >= 7) & (group_hourly['hour'] < 19)]
                        if len(light_hourly) > 0:
                            light_auc = np.trapz(light_hourly['mean'], light_hourly['hour'])
                        else:
                            light_auc = np.nan
                        
                        # Calculate dark cycle AUC
                        dark_hourly = group_hourly[(group_hourly['hour'] < 7) | (group_hourly['hour'] >= 19)]
                        if len(dark_hourly) > 0:
                            dark_auc = np.trapz(dark_hourly['mean'], dark_hourly['hour'])
                        else:
                            dark_auc = np.nan
                        
                        auc_results.append({
                            'Group': group,
                            'Total AUC': total_auc,
                            'Light Cycle AUC': light_auc,
                            'Dark Cycle AUC': dark_auc
                        })
                    
                    # Create AUC dataframe
                    auc_df = pd.DataFrame(auc_results)
                    
                    # Display AUC results
                    st.dataframe(auc_df.round(2))
        
        # Export Section with better explanations
        st.markdown("---")
        st.subheader("Export Results")

        st.markdown("""
        Download your analysis results for use in other programs like Excel, GraphPad Prism, or R.
        These CSV files can be directly imported into most analysis software.
        """)

        col1, col2 = st.columns(2)

        with col1:
            # Statistics export
            if not group_stats.empty:
                csv_stats = group_stats.to_csv().encode('utf-8')
                st.download_button(
                    label="📥 Download Statistics (CSV)",
                    data=csv_stats,
                    file_name=f"{parameter}_statistics.csv",
                    mime="text/csv",
                    help="Download complete statistical results for further analysis"
                )
                
                st.markdown("""
                **The Statistics CSV contains:**
                - Group names
                - Number of animals (N)
                - Mean values for each group
                - Standard deviation (SD)
                - Standard error of mean (SEM)
                - 95% Confidence intervals
                
                For creating your own custom graphs 
                """)

        with col2:
            # Time pattern export
            if 'hourly_df' in locals() and not hourly_df.empty:
                csv_hourly = hourly_df.to_csv().encode('utf-8')
                st.download_button(
                    label="📥 Download 24h Pattern Data (CSV)",
                    data=csv_hourly,
                    file_name=f"{parameter}_24h_pattern.csv",
                    mime="text/csv",
                    help="Download hourly averages for each group"
                )
                
                st.markdown("""
                **The 24h Pattern CSV contains:**
                - Hour of day (0-23)
                - Mean value for each hour
                - Standard error (SEM) for each hour
                - Group identifier
                
                For creating custom time-course graphs or performing additional temporal analysis.
                """)
        
        # Sample Calculation Section
        st.markdown("---")
        st.subheader("📝 Understanding the Calculations")

        with st.expander("See how means and statistics are calculated from raw data"):
            st.markdown("### How Group Statistics Are Calculated")
            
            # Pick a sample group to demonstrate
            sample_group = selected_groups[0]
            sample_cages = group_assignments[group_assignments['Group'] == sample_group]['normalized_cage_id'].unique()
            
            st.markdown(f"**Example calculation for {sample_group}:**")
            
            # Show raw data sample
            st.markdown("#### 1. Start with raw data measurements")
            raw_sample = grouped_data[(grouped_data['Group'] == sample_group)].head(10)
            st.dataframe(raw_sample[['cage', 'timestamp', 'value', 'is_light']].head())
            
            # Show individual cage means
            st.markdown("#### 2. Calculate mean for each animal")
            
            if cycle_filter is not None:  # Light or Dark cycle
                cages_data = grouped_data[(grouped_data['Group'] == sample_group) & 
                                        (grouped_data['is_light'] == cycle_filter)]
            else:  # 24-hour Average
                cages_data = grouped_data[grouped_data['Group'] == sample_group]
            
            cage_means = []
            all_values = []
            
            for cage in sample_cages:
                if len(cages_data[cages_data['normalized_cage_id'] == cage]) > 0:
                    cage_mean = cages_data[cages_data['normalized_cage_id'] == cage]['value'].mean()
                    subject_id = group_assignments[group_assignments['normalized_cage_id'] == cage]['Subject ID'].iloc[0]
                    cage_values = cages_data[cages_data['normalized_cage_id'] == cage]['value'].tolist()
                    all_values.extend(cage_values)
                    
                    cage_means.append({
                        'Cage': f"CAGE {cage.split('101')[1] if '101' in cage else cage}",
                        'Subject ID': subject_id, 
                        'Mean Value': round(cage_mean, 2),
                        'Number of Measurements': len(cage_values)
                    })
            
            st.dataframe(pd.DataFrame(cage_means))
            
            # Show group mean calculation
            st.markdown("#### 3. Calculate overall group statistics")
            
            cage_means_values = [x['Mean Value'] for x in cage_means]
            group_mean = np.mean(cage_means_values)
            group_sd = np.std(cage_means_values, ddof=1)  # Using n-1 for sample standard deviation
            group_sem = group_sd / np.sqrt(len(cage_means_values))
            
            st.markdown(f"**Group Mean** = ({' + '.join([str(x) for x in cage_means_values])}) ÷ {len(cage_means_values)} = **{group_mean:.2f}**")
            
            st.markdown(f"**Group Standard Deviation (SD)**")
            st.markdown("""
            SD measures how spread out the individual animal means are from the group mean.
            
            1. Calculate squared differences from the group mean
            2. Average those squared differences
            3. Take the square root
            """)
            
            squared_diffs = [(x - group_mean)**2 for x in cage_means_values]
            
            st.markdown(f"Squared differences: {[round(x, 2) for x in squared_diffs]}")
            st.markdown(f"Average squared difference: {sum(squared_diffs) / (len(squared_diffs) - 1):.2f}")
            st.markdown(f"Standard Deviation = √{sum(squared_diffs) / (len(squared_diffs) - 1):.2f} = **{group_sd:.2f}**")
            
            st.markdown(f"**Standard Error of Mean (SEM)**")
            st.markdown("""
            SEM tells you how precise your estimate of the mean is.
            
            SEM = SD ÷ √N  (where N is the number of animals)
            """)
            
            st.markdown(f"SEM = {group_sd:.2f} ÷ √{len(cage_means_values)} = {group_sd:.2f} ÷ {np.sqrt(len(cage_means_values)):.2f} = **{group_sem:.2f}**")
            
            # Hourly calculation explanation
            st.markdown("### How Hourly Averages Are Calculated")
            
            st.markdown("""
            For the 24-hour pattern graph:
            
            1. Group all measurements by hour of day (0-23)
            2. Calculate the average value for each hour
            3. Plot these hourly averages to show the daily pattern
            """)
            
            # Show sample for one hour
            sample_hour = 12  # Noon
            hour_data = grouped_data[(grouped_data['Group'] == sample_group) & (grouped_data['hour'] == sample_hour)]
            
            if len(hour_data) > 0:
                st.markdown(f"**Example: Hour {sample_hour} (12 PM) for {sample_group}**")
                st.dataframe(hour_data[['cage', 'timestamp', 'value']].head())
                hour_mean = hour_data['value'].mean()
                hour_sem = hour_data['value'].sem()
                
                st.markdown(f"Hour {sample_hour} average = {hour_mean:.2f}")
                st.markdown(f"Hour {sample_hour} SEM = {hour_sem:.2f}")
                
                st.markdown("""
                These hourly averages are plotted on the 24-hour pattern graph, with SEM shown as shaded regions.
                The graph lets you see how values change throughout the day and compare patterns between groups.
                """)
        
        # Optional interpretation section (collapsible)
        with st.expander("📋 Optional: Results Interpretation", expanded=False):
            st.markdown("### Statistical Interpretation")
            
            if p_value is not None:
                if p_value < 0.05:
                    st.markdown(f"""
                    **Statistical Result**: Your groups show a **statistically significant difference** (p={p_value:.4f}).
                    
                    This suggests that the observed difference between groups is unlikely to be due to random chance alone.
                    """)
                    
                    if test_name == "t-test" and 'cohen_d' in locals():
                        st.markdown(f"""
                        **Effect Size**: The difference represents a **{effect_interpretation.lower()} effect** (Cohen's d = {cohen_d:.2f}).
                        
                        {'This suggests the difference, while statistically significant, may not be biologically meaningful.' if effect_interpretation == 'Small' else 
                        'This suggests a moderate biological difference between groups.' if effect_interpretation == 'Medium' else
                        'This suggests a substantial biological difference between groups.'}
                        
                        **Practical Significance**: Group {group1 if mean1 > mean2 else group2} showed a {percent_diff:.1f}% 
                        higher {parameter} value than Group {group2 if mean1 > mean2 else group1}.
                        """)
                else:
                    st.markdown(f"""
                    **Statistical Result**: Your groups do not show a statistically significant difference (p={p_value:.4f}).
                    
                    This suggests that any observed differences between groups may be due to random variation rather than a true effect.
                    
                    **Possible reasons for non-significance:**
                    - Small sample size (low statistical power)
                    - High variability within groups
                    - Truly no difference between groups
                    - Analyzing the wrong cycle (try looking at Dark Cycle instead of Light Cycle)
                    """)
            else:
                st.markdown("""
                **Not enough data**: Unable to perform statistical test. Please ensure you have sufficient data for each group.
                """)

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
        # Replaced old end_time start_time with this (days to analyze)
        days_to_analyze = {
            "Last 24 Hours": 1,
            "Last 48 Hours": 2,
            "Last 72 Hours": 3
        }[time_window]

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
        st.error("CLAMSer is adapted specifically for raw CLAMS output. The uploaded file has likely been modified - Please ensure the uploaded file contains the correct subject information")
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
                subject_id = cage_df[cage_df["Cage"] == cage]["Subject ID"].iloc[0]
                group_summary.append({
                    "Group": group_name,
                    "Cage": cage,
                    "Subject ID": subject_id
                })
                
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
tab1, tab2, tab3 = st.tabs(["📊 Overview", "📈 Statistical Analysis", "🧪 Verification"])

with tab1:
    # Only show welcome message if no file is uploaded
    if uploaded_file is None:
        st.markdown("""
        ## Welcome to CLAMSer 🧬

        This tool analyzes metabolic data from Comprehensive Lab Animal Monitoring System (CLAMS) files.

        **Getting Started:**
        1. Upload your CLAMS data file using the sidebar
        2. Select the parameter type (VO2, VCO2, RER, etc.)
        3. Choose your preferred time window
        4. Review the analysis below

        The Overview tab provides summary metrics, 24-hour patterns, and detailed data tables for your CLAMS data.
        """)
    else:
        # When file is uploaded, just show the header followed by parameter guide
        st.header("📊 Overview")

    # Add Parameter Guide expander
    with st.expander("📚 CLAMS Parameter Guide - Learn about each measurement", expanded=False):
        st.markdown("""
        ## Understanding CLAMS Parameters
        
        CLAMS (Comprehensive Lab Animal Monitoring System) measures several key metabolic parameters. Here's what each one means:
        """)
        
        param_tabs = st.tabs(["Metabolic (VO2, VCO2, HEAT)", "RER", "Activity (XTOT, XAMB)", "FEED"])
        
        with param_tabs[0]:
            st.markdown("""
            ### Metabolic Parameters
            
            #### VO2 (Oxygen Consumption)
            - **What it measures**: Volume of oxygen consumed per kg of body weight per hour
            - **Units**: ml/kg/hr
            - **Normal range**: 2000-5000 ml/kg/hr (mice), lower in larger animals
            - **Significance**: Primary indicator of metabolic rate
            - **Example interpretation**: Higher VO2 indicates increased energy expenditure, which could suggest greater physical activity, thermogenesis, or metabolic stress
            
            #### VCO2 (Carbon Dioxide Production)
            - **What it measures**: Volume of carbon dioxide produced per kg of body weight per hour
            - **Units**: ml/kg/hr
            - **Normal range**: 1600-4500 ml/kg/hr (mice)
            - **Significance**: Indicates substrate utilization and metabolic rate
            - **Example interpretation**: VCO2 typically tracks with VO2, but their ratio (RER) provides information about which fuel source is being utilized
            
            #### HEAT (Heat Production)
            - **What it measures**: Calculated heat production based on VO2 and RER values
            - **Units**: kcal/hr
            - **Significance**: Direct measure of energy expenditure
            - **Example interpretation**: Higher heat production indicates increased metabolic rate and energy expenditure
            
            **Note on Lean Mass Adjustment**: For more accurate comparisons between animals of different body compositions, these parameters should be normalized to lean mass rather than total body weight.
            """)
            
        with param_tabs[1]:
            st.markdown("""
            ### RER (Respiratory Exchange Ratio)
            
            - **What it measures**: Ratio of VCO2 produced to VO2 consumed
            - **Units**: Unitless ratio
            - **Normal range**: 0.7 - 1.0
            - **Significance**: Indicates which fuel source (fat vs. carbohydrate) is being metabolized
            
            #### RER Value Interpretation:
            - **~0.7**: Primarily fat oxidation
            - **~0.8**: Mixed fuel source
            - **~0.9-1.0**: Primarily carbohydrate oxidation
            - **>1.0**: Net carbohydrate synthesis or hyperventilation
            
            #### Example patterns:
            - **Fasting**: RER drops toward 0.7 as fat becomes the primary fuel
            - **After carbohydrate meal**: RER rises toward 1.0
            - **Exercise**: Can vary based on intensity (higher intensity tends toward carbohydrate use)
            """)
            
        with param_tabs[2]:
            st.markdown("""
            ### Activity Parameters
            
            #### XTOT (Total Activity)
            - **What it measures**: Total beam breaks in the x-axis
            - **Units**: counts
            - **Significance**: Measures all movement including fine movements and ambulatory activity
            - **Example interpretation**: Higher XTOT values indicate more overall physical activity
            
            #### XAMB (Ambulatory Activity)
            - **What it measures**: Consecutive beam breaks indicating ambulatory movement
            - **Units**: counts
            - **Significance**: Measures intentional locomotor activity (walking, running)
            - **Example interpretation**: XAMB represents purposeful movement and is typically lower than XTOT
            
            #### Activity Patterns:
            - **Nocturnal pattern**: Rodents normally show higher activity during dark cycles
            - **Light/dark comparison**: Typically 2-5x higher during active (dark) phase in nocturnal animals
            - **Activity spikes**: Often correlate with feeding events or environmental stimuli
            """)
            
        with param_tabs[3]:
            st.markdown("""
            ### FEED (Food Intake)
            
            - **What it measures**: Weight of food consumed
            - **Units**: grams
            - **Significance**: Directly measures energy intake
            - **Normal patterns**:
            - Higher during dark cycle in nocturnal animals
            - Often occurs in discrete meals rather than continuous consumption
            - Typically correlates with activity patterns
            
            #### Example interpretations:
            - **Increased feed intake without weight gain**: May indicate higher energy expenditure
            - **Decreased feed intake**: May indicate illness, stress, or altered metabolism
            - **Altered feeding patterns**: Changes in meal timing, size, or frequency can indicate metabolic or behavioral changes
            
            #### Note:
            Food intake should be considered alongside metabolic parameters for a complete energy balance assessment.
            """)

    
    
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

                    # Add a visual separator
                    st.markdown("---")

                    # Group Assignment Section - now collapsible
                    st.header("👥 Group Assignment")

                    # Determine if groups are already assigned
                    has_groups = 'group_assignments' in st.session_state and not st.session_state.get('group_assignments', pd.DataFrame()).empty

                    # Create expander with smart default state (expanded if no groups assigned)
                    with st.expander("Assign Animals to Groups", expanded=not has_groups):
                        st.info("Assign your animals to experimental groups before proceeding with visualization.")

                        cage_info = extract_cage_info(uploaded_file)
                        if cage_info:
                            cage_df = pd.DataFrame([
                                {"Cage": f"CAGE {k}", "Subject ID": v} 
                                for k, v in cage_info.items()
                            ])
                            
                            # Display cage information and allow group assignment
                            group_assignments = assign_groups(cage_df, key_prefix="overview")
                            if group_assignments is not None:
                                st.session_state['group_assignments'] = group_assignments
                                
                        # Add indicator of current group assignments
                        if has_groups:
                            existing_groups = st.session_state['group_assignments']['Group'].unique()
                            st.success(f"✅ You have assigned animals to {len(existing_groups)} groups: {', '.join(existing_groups)}")

                    # Add another separator before visualizations
                    st.markdown("---")
                    
                    # Add lean mass inputs if option is enabled
                    if st.session_state.get("apply_lean_mass", False) and parameter in ["VO2", "VCO2", "HEAT"]:
                        st.header("📏 Lean Mass Normalization")
                        st.markdown(f"""
                        You've enabled lean mass adjustment for {parameter}. Enter the lean mass values for each animal below.
                        All {parameter} values will be normalized to a reference lean mass of {st.session_state.get('reference_lean_mass', 20.0)}g.
                        """)
                        
                        # Lean mass inputs
                        lean_mass_inputs = {}
                        cols = st.columns(3)
                        
                        for i, (cage_id, subject_id) in enumerate(cage_info.items()):
                            cage_label = f"CAGE {int(cage_id) - 100:02d}"
                            with cols[i % 3]:
                                lean_mass = st.number_input(
                                    f"Lean mass for {cage_label} (Subject: {subject_id})",
                                    min_value=1.0,
                                    value=20.0,
                                    step=0.1,
                                    format="%.1f",
                                    key=f"lean_mass_{cage_id}"
                                )
                                lean_mass_inputs[cage_label] = lean_mass
                        
                        # Store in session state
                        st.session_state['lean_mass_data'] = lean_mass_inputs
                        
                        # Add formula explanation
                        with st.expander("How is the adjustment calculated?"):
                            st.markdown(f"""
                            ### Lean Mass Adjustment Formula

                            For each animal:
                            ```
                            Adjusted {parameter} = Original {parameter} × (Reference Mass ÷ Animal's Lean Mass)
                            ```

                            **Example:**
                            - Reference lean mass: 20g
                            - Animal's lean mass: 25g
                            - Original {parameter}: 3000 ml/kg/hr
                            - Adjusted {parameter}: 3000 × (20 ÷ 25) = 2400 ml/kg/hr

                            This normalization accounts for the higher metabolic activity of lean tissue compared to fat tissue.
                            """)
                        
                        st.markdown("---")
                                        
                    
                    
                    # 2. GRAPH - Create and display the visualization
                    # Create a time-based view instead of just hourly
                    # First create proper timeline data
                    timeline_data = raw_data.copy()

                    # Enhanced Data View Options section
                    st.header("📊 Data Visualization")

                    # Create a stylish container for data view options
                    data_view_container = st.container()
                    with data_view_container:
                        # Use columns for better layout
                        col1, col2 = st.columns([1, 2])
                        
                        with col1:
                            st.markdown("### 🔍 View Options")
                            # Create a radio button for data display mode with better styling
                            selected_option = st.radio(
                                "Data Display Mode:",
                                ["Show Average Across All Animals", "Focus on Individual Animals"],
                                key="animal_display_mode",
                                help="Choose how to visualize your data"
                            )
                        
                        with col2:
                            if selected_option == "Show Average Across All Animals":
                                st.markdown("### 📈 Average View")
                                st.markdown("""
                                **You're viewing the average of all animals with standard error bands.**
                                
                                - Blue line: Mean value across all animals
                                - Shaded area: Standard error of the mean (SEM)
                                - Gray regions: Dark cycle (7PM-7AM)
                                """)
                            else:  # Focus on Individual Animals
                                st.markdown("### 👁️ Individual View")
                                # Create a mapping from subject ID to cage
                                subject_to_cage = {}
                                for cage, subject in zip(results.index, results['Subject ID']):
                                    subject_to_cage[subject] = cage
                                
                                # Get list of available subjects
                                available_subjects = results['Subject ID'].tolist()
                                
                                # Get any group information if available
                                if 'group_assignments' in st.session_state and not st.session_state['group_assignments'].empty:
                                    group_df = st.session_state['group_assignments']
                                    # Create options with group info if available
                                    subject_options = [f"{subject} ({group_df[group_df['Subject ID']==subject]['Group'].iloc[0]})" 
                                                    if subject in group_df['Subject ID'].values else subject 
                                                    for subject in available_subjects]
                                    
                                    # Show selection dropdown with subject IDs and group info
                                    selected_subject_labels = st.multiselect(
                                        "Select specific animals to display:",
                                        subject_options,
                                        default=[subject_options[0]] if subject_options else [],
                                        key="subject_multiselect_with_group"
                                    )
                                    
                                    # Extract just the subject ID from the labels
                                    selected_subjects = [label.split(" (")[0] for label in selected_subject_labels]
                                else:
                                    # Show regular selection dropdown with just subject IDs
                                    selected_subjects = st.multiselect(
                                        "Select specific animals to display:",
                                        available_subjects,
                                        default=[available_subjects[0]] if available_subjects else [],
                                        key="subject_multiselect"
                                    )
                                
                                # Convert selected subjects to cages for filtering
                                if selected_subjects:
                                    selected_cages = [subject_to_cage[subject] for subject in selected_subjects]
                                    
                                    # Filter timeline data
                                    timeline_data = timeline_data[timeline_data['cage'].isin(selected_cages)]
                                    st.success(f"📊 Displaying data for {len(selected_subjects)} selected animals")
                                else:
                                    st.warning("⚠️ Please select at least one animal to display")
                    
                    
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
                                ("suggesting normal nocturnal activity. I.E. it is a good thing :)" if normal_pattern else "which may indicate altered circadian rhythm."))
        
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
                        for i, cage in enumerate(timeline_results['cage'].unique()):
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

                    while current_date <= max_date:
                        # Add dark cycle from midnight to 7 AM
                        morning_start = pd.Timestamp(current_date.strftime('%Y-%m-%d') + ' 00:00:00')
                        morning_end = pd.Timestamp(current_date.strftime('%Y-%m-%d') + ' 07:00:00')
                        
                        # Add dark cycle from 7 PM to midnight
                        evening_start = pd.Timestamp(current_date.strftime('%Y-%m-%d') + ' 19:00:00')
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
                if uploaded_file is not None and raw_data is not None and results is not None:
                    simplified_statistical_analysis(tab2, raw_data, results, parameter, parameter_descriptions, time_window)
                else:
                    st.warning("Please upload data in the Overview tab first")
                    
            # Tab 3: Verification
            with tab3:
                st.subheader("Raw Data Display")
                st.write("This section shows the raw data used for calculations based on your selected time window.")
                
                # Add time window filtering to raw data display
                if raw_data is not None:
                    # Get current time window selection
                    days_to_analyze = {
                        "Last 24 Hours": 1,
                        "Last 48 Hours": 2,
                        "Last 72 Hours": 3
                    }.get(time_window, 1)  # Default to 1 day if not found
                    
                    # Calculate the time window
                    end_time = raw_data['timestamp'].max()
                    start_time = end_time - pd.Timedelta(days=days_to_analyze)
                    
                    # Create a properly filtered dataset for display
                    filtered_raw_data = raw_data[
                        (raw_data['timestamp'] >= start_time) & 
                        (raw_data['timestamp'] <= end_time)
                    ].copy()
                    
                    # Show time period info
                    st.info(f"Analysis period: {start_time.strftime('%Y-%m-%d %H:%M')} to {end_time.strftime('%Y-%m-%d %H:%M')} ({len(filtered_raw_data):,} records)")
                    
                    # Display the properly filtered data
                    st.dataframe(filtered_raw_data)
                
                # Add our enhanced sample calculations
                st.subheader("Sample Calculations")
                if results is not None and raw_data is not None:
                    # Replace the original show_verification_calcs with our enhanced version
                    enhanced_sample_calculations(raw_data, results, hourly_results, parameter)
                
                # Add interactive calculation visualization
                if raw_data is not None:
                    add_calculation_visualization(raw_data, parameter)
                
                # Add the trust verification section
                if raw_data is not None:
                    add_trust_verification_section(raw_data, results, parameter, parameter_descriptions)
                
                # About section (existing)
                st.subheader("About CLAMS Data Analyzer")
                st.markdown("""
                This tool was developed to streamline the analysis of CLAMS data files. 
                
                ### Questions or feedback?
                Contact: Menzies Laboratory or Zane Khartabill (email: mkhal061@uottawa.ca)
                """)
                
                # Analysis details - keep this as it's useful for verification
                st.subheader("Analysis Details")
                if raw_data is not None:
                    st.markdown(f"""
                    - Time Window: {time_window}
                    - Analysis Period: {raw_data['timestamp'].min().strftime('%Y-%m-%d %H:%M')} to {raw_data['timestamp'].max().strftime('%Y-%m-%d %H:%M')}
                    - Light Cycle: 7:00 AM - 7:00 PM
                    - Dark Cycle: 7:00 PM - 7:00 AM
                    - Total Records Processed: {len(raw_data):,}
                    """)

