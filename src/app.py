import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import plotly.express as px
import plotly.graph_objects as go

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
        key="time_window_radio"  # Add this line
    )

    # Move file upload to sidebar
    uploaded_file = st.file_uploader(
        f"Upload {parameter} CSV",
        type="csv",
        help="Upload your CLAMS data file",
        key="file_upload_1"
        
    )

# Move parameter selection to sidebar
st.markdown("## Analysis Settings")
parameter = st.selectbox(
    "Select Parameter",
    list(parameter_descriptions.keys()),
    format_func=lambda x: f"{x}: ~{parameter_descriptions[x]}"
)

# Main title in content area
st.title("CLAMSer: CLAMS Data Analyzer")

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
tab1, tab2, tab3 = st.tabs(["📊 Overview", "📈 Statistics", "📋 Data"])

if uploaded_file is not None:
    # First verify file type
    is_valid, error_message = verify_file_type(uploaded_file, parameter)
    
    if not is_valid:
        st.error(error_message)
    else:
        # Process data first
        with st.spinner('Processing data...'):
            results, hourly_results, raw_data = process_clams_data(uploaded_file, parameter)
            
        if results is not None:
            # Tab 1: Overview
            with tab1:
                # Display parameter-specific metrics
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
                
                elif parameter == "RER":
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("Average Light RER", f"{results['Light Average'].mean():.3f}")
                    with col2:
                        st.metric("Average Dark RER", f"{results['Dark Average'].mean():.3f}")
                    with col3:
                        st.metric("Total Records", len(raw_data))
                
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
                
                else:  # VO2, VCO2, HEAT
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric(f"Average Light {parameter}", 
                                f"{results['Light Average'].mean():.2f} {PARAMETER_UNITS[parameter]}")
                    with col2:
                        st.metric(f"Average Dark {parameter}", 
                                f"{results['Dark Average'].mean():.2f} {PARAMETER_UNITS[parameter]}")
                    with col3:
                        st.metric("Total Records", len(raw_data))

                # Create line plot with SEM for all parameters
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
                
                # Update layout based on parameter
                fig.update_layout(
                    title=f'24-Hour {parameter} Pattern',
                    xaxis_title='Hour of Day',
                    yaxis_title=f'{parameter} ({PARAMETER_UNITS[parameter]})'
                )
                
            # Tab 2: Statistics
            with tab2:
                st.subheader("Statistical Analysis")
                st.info("📊 Values highlighted in red are potential outliers (> 2 standard deviations from the mean)")
                st.dataframe(style_dataframe(results))
            
            # Tab 3: Data
            with tab3:
                st.subheader("Raw Data")
                st.dataframe(raw_data)
                
                # Download buttons
                col1, col2 = st.columns(2)
                with col1:
                    csv_light_dark = results.to_csv().encode('utf-8')
                    st.download_button(
                        label=f"📥 Download {parameter} Light/Dark Averages",
                        data=csv_light_dark,
                        file_name=f"{parameter}_lightdark_averages.csv",
                        mime="text/csv"
                    )
                
                with col2:
                    csv_hourly = hourly_results.to_csv().encode('utf-8')
                    st.download_button(
                        label=f"📥 Download {parameter} Hourly Averages",
                        data=csv_hourly,
                        file_name=f"{parameter}_hourly_averages.csv",
                        mime="text/csv"
                    )
                    
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
                st.plotly_chart(fig, use_container_width=True, key=f"{parameter}_activity_plot")
                
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
                    
                # Add verification calculations
                if results is not None and hourly_results is not None:
                    show_verification_calcs(raw_data, results, hourly_results, parameter)
                
                # Add footer with analysis details
                st.markdown(f"""
            **Analysis Details:**
            - Time Window: {time_window}
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
