"""
Author: Maria (mariabaumg)
"""


import streamlit as st
import json
import folium
from streamlit_folium import st_folium

@st.cache_resource
def load_data():
    with open('predictions.json', 'r') as f:
        return json.load(f)

prediction_data = load_data()

st.title("CitiBike Seasonal Demand Predictor")

# sidebar
st.sidebar.header("Forecast Settings")
target_season = st.sidebar.selectbox("Select Season", ["Spring", "Summer", "Fall", "Winter"])
target_day = st.sidebar.selectbox("Select Day", [0, 1, 2, 3, 4, 5, 6], 
                                  format_func=lambda x: ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"][x])
target_hour = st.sidebar.slider("Hour of Day", 0, 23, 12)

# map
m = folium.Map(location=[40.73, -73.99], zoom_start=13)
for s_id, info in prediction_data.items():
    folium.CircleMarker(location=[info['lat'], info['lon']], radius=7, popup=f"ID: {s_id}", color="#008080", fill=True).add_to(m)

map_output = st_folium(m, width=900, height=500)

if map_output.get('last_object_clicked_popup'):
    selected_id = map_output['last_object_clicked_popup'].split(": ")[1]
    prediction = prediction_data[selected_id]['seasons'][target_season][str(target_day)][target_hour]
    
    st.metric(label=f"Predicted Inflow ({target_season})", value=f"{prediction} bikes/hr")