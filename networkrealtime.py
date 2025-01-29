import streamlit as st
import requests
import json
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from sklearn.ensemble import RandomForestClassifier, GradientBoostingRegressor
from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, mean_squared_error
import heapq
import time
from fpdf import FPDF
from io import BytesIO  # For report buffer
import threading  # For running the background task

# ============================
# Data Collection from Prometheus
# ============================

def fetch_prometheus_metrics(prometheus_url, query):
    """Fetches network metrics from Prometheus."""
    try:
        response = requests.get(f"{prometheus_url}/api/v1/query", params={"query": query})
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        st.error(f"Error fetching Prometheus data: {e}")
        return None

prometheus_url = "https://demo.promlabs.com"
query = "rate(node_network_receive_bytes_total[1m])"  # Example query

# ============================
# Network Class
# ============================

class Network:
    def __init__(self, nodes, edges):
        self.nodes = nodes
        self.graph = {node: [] for node in nodes}
        for edge in edges:
            node1, node2, weight, capacity = edge
            self.graph[node1].append((node2, weight, capacity))
            self.graph[node2].append((node1, weight, capacity))

    def generate_traffic_data(self):
        """Generates synthetic traffic data."""
        data = []
        for node in self.nodes:
            for neighbor, weight, capacity in self.graph[node]:
                load = np.random.randint(0, 100)
                historical_faults = np.random.randint(0, 10)
                fault = np.random.choice([0, 1], p=[0.9, 0.1])
                congestion = np.random.uniform(0, 1)
                data.append([node, neighbor, load, capacity, historical_faults, fault, congestion])
        return pd.DataFrame(data, columns=["node", "neighbor", "load", "capacity", "historical_faults", "fault", "congestion"])

# ============================
# ML-based Predictive Analytics
# ============================

class PredictiveAnalytics:
    def __init__(self, data):
        self.data = data
        self.train_models()

    def train_models(self):
        X = self.data[["load", "capacity", "historical_faults"]]
        y_fault = self.data["fault"]
        y_congestion = self.data["congestion"]

        X_train, X_test, y_train, y_test = train_test_split(X, y_fault, test_size=0.2, random_state=42)
        self.fault_model = RandomForestClassifier()
        self.fault_model.fit(X_train, y_train)
        self.fault_accuracy = accuracy_score(y_test, self.fault_model.predict(X_test))

        X_train, X_test, y_train, y_test = train_test_split(X, y_congestion, test_size=0.2, random_state=42)
        self.congestion_model = GradientBoostingRegressor()
        self.congestion_model.fit(X_train, y_train)
        self.congestion_mse = mean_squared_error(y_test, self.congestion_model.predict(X_test))

        self.cluster_model = KMeans(n_clusters=2, random_state=42)
        self.cluster_model.fit(X)

    def update_models(self, new_data):
        """Update models incrementally with new data."""
        X = new_data[["load", "capacity", "historical_faults"]]
        y_fault = new_data["fault"]
        y_congestion = new_data["congestion"]

        # Re-train models with the new data (could use partial fit for more efficiency in real time)
        self.fault_model.fit(X, y_fault)
        self.congestion_model.fit(X, y_congestion)
        self.train_models()  # Recalculate metrics after model update

    def predict_fault(self, load, capacity, historical_faults):
        input_data = pd.DataFrame([[load, capacity, historical_faults]], columns=["load", "capacity", "historical_faults"])
        return self.fault_model.predict(input_data)[0]

    def predict_congestion(self, load, capacity, historical_faults):
        input_data = pd.DataFrame([[load, capacity, historical_faults]], columns=["load", "capacity", "historical_faults"])
        return self.congestion_model.predict(input_data)[0]

# ============================
# Real-Time Feedback Loop
# ============================

def fetch_and_update_metrics(predictive_analytics):
    """Fetch new metrics and update models in the background."""
    while True:
        metrics = fetch_prometheus_metrics(prometheus_url, query)
        if metrics and "data" in metrics and "result" in metrics["data"]:
            # Simulate new traffic data using fetched metrics
            new_traffic_data = pd.DataFrame([
                # Simulate adding some new traffic data for this example
                ["A", "B", np.random.randint(0, 100), np.random.randint(100, 150), np.random.randint(0, 10), np.random.choice([0, 1], p=[0.9, 0.1]), np.random.uniform(0, 1)],
                ["B", "C", np.random.randint(0, 100), np.random.randint(100, 150), np.random.randint(0, 10), np.random.choice([0, 1], p=[0.9, 0.1]), np.random.uniform(0, 1)],
            ], columns=["node", "neighbor", "load", "capacity", "historical_faults", "fault", "congestion"])
            
            # Update the model with the new data
            predictive_analytics.update_models(new_traffic_data)
        
        time.sleep(60)  # Wait for a minute before fetching new metrics again

# ============================
# Streamlit UI
# ============================

st.title("🌐 Real-Time Network Fault & Congestion Prediction using Dijkstra's Algorithm ")

# Fetch real-time network metrics
st.sidebar.header("📡 Real-Time Data")
if st.sidebar.button("Fetch Metrics"):
    metrics = fetch_prometheus_metrics(prometheus_url, query)
    if metrics and "data" in metrics and "result" in metrics["data"]:
        st.sidebar.success("Metrics Retrieved!")
        st.json(metrics["data"]["result"])
    else:
        st.sidebar.error("Failed to fetch data.")

# Simulate network
nodes = ["A", "B", "C", "D", "E"]
edges = [
    ("A", "B", 10, 100),
    ("A", "C", 20, 150),
    ("B", "D", 30, 200),
    ("C", "D", 10, 100),
    ("D", "E", 20, 150),
]
network = Network(nodes, edges)

# Generate traffic data
traffic_data = network.generate_traffic_data()

# Train ML models
predictive_analytics = PredictiveAnalytics(traffic_data)

# Start real-time feedback in the background
feedback_thread = threading.Thread(target=fetch_and_update_metrics, args=(predictive_analytics,))
feedback_thread.daemon = True
feedback_thread.start()

# Path Calculation (will be recalculated when generating the report)
start_node = st.sidebar.selectbox("Start Node", nodes)
end_node = st.sidebar.selectbox("End Node", nodes)

if st.sidebar.button("Generate Report"):
    ml_dijkstra = MLEnhancedDijkstra(network, predictive_analytics)
    path, cost = ml_dijkstra.shortest_path(start_node, end_node)
    # Trigger report generation only if valid path is found
    if path and cost:
        st.write("Generating report...")
        st.sidebar.write(f"🔹 Shortest Path: {' → '.join(path)}")
        st.sidebar.write(f"💰 Total Cost: {cost:.2f}")
        report_buffer = generate_report(traffic_data, predictive_analytics.fault_accuracy, predictive_analytics.congestion_mse, path, cost)
        st.write(f"Report Buffer Size: {len(report_buffer.getvalue())} bytes")  # Debugging the buffer size
        st.download_button("Download Report", data=report_buffer, file_name="network_report.pdf", mime="application/pdf")
    else:
        st.sidebar.error("No path found!")

# Traffic Visualization
st.subheader("📊 Network Traffic Overview")
fig = go.Figure()
fig.add_trace(go.Scatter(x=traffic_data["node"], y=traffic_data["load"], mode='markers', name='Load'))
fig.add_trace(go.Scatter(x=traffic_data["node"], y=traffic_data["congestion"], mode='lines', name='Congestion'))
st.plotly_chart(fig)

st.info("🚀 Use the sidebar to fetch metrics, report & compute shortest paths.")
