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

    def predict_fault(self, load, capacity, historical_faults):
        input_data = pd.DataFrame([[load, capacity, historical_faults]], columns=["load", "capacity", "historical_faults"])
        return self.fault_model.predict(input_data)[0]

    def predict_congestion(self, load, capacity, historical_faults):
        input_data = pd.DataFrame([[load, capacity, historical_faults]], columns=["load", "capacity", "historical_faults"])
        return self.congestion_model.predict(input_data)[0]

# ============================
# ML-Enhanced Dijkstra Algorithm
# ============================

class MLEnhancedDijkstra:
    def __init__(self, network, predictive_analytics):
        self.network = network
        self.predictive_analytics = predictive_analytics

    def shortest_path(self, start, end):
        pq = [(0, start, [])]
        visited = set()

        while pq:
            cost, node, path = heapq.heappop(pq)
            if node in visited:
                continue
            visited.add(node)
            path = path + [node]

            if node == end:
                return path, cost

            for neighbor, weight, capacity in self.network.graph[node]:
                if neighbor not in visited:
                    fault_prob = self.predictive_analytics.predict_fault(weight, capacity, 0)
                    congestion_level = self.predictive_analytics.predict_congestion(weight, capacity, 0)
                    adjusted_weight = weight * (1 + fault_prob + congestion_level)
                    heapq.heappush(pq, (cost + adjusted_weight, neighbor, path))

        return [], float('inf')

# ============================
# Report Generation (PDF)
# ============================


def generate_report(traffic_data, fault_accuracy, congestion_mse, path, cost):
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", size=12)

    # Title
    pdf.cell(200, 10, txt="Network Performance Report", ln=True, align='C')

    # Fault Prediction Accuracy and Congestion MSE
    pdf.cell(200, 10, txt=f"Fault Prediction Accuracy: {fault_accuracy:.2f}", ln=True)
    pdf.cell(200, 10, txt=f"Congestion Prediction MSE: {congestion_mse:.2f}", ln=True)

    # Traffic Data Summary
    pdf.cell(200, 10, txt="Traffic Data Summary:", ln=True)
    for idx, row in traffic_data.iterrows():
        pdf.cell(200, 10, txt=f"Node: {row['node']} | Neighbor: {row['neighbor']} | Load: {row['load']} | Congestion: {row['congestion']}", ln=True)

    # Path and Cost
    path_str = ' -> '.join(path)  # Replacing → with ->
    pdf.cell(200, 10, txt=f"Shortest Path: {path_str}", ln=True)
    pdf.cell(200, 10, txt=f"Total Cost: {cost:.2f}", ln=True)

    # Save PDF to buffer
    buffer = BytesIO()
    pdf.output(dest='S')  # 'S' means output to string, which you can then use as BytesIO
    buffer.write(pdf.output(dest='S').encode('latin1'))  # Write the output directly into the buffer
    buffer.seek(0)
    return buffer

# ============================
# Streamlit UI
# ============================

st.title("🌐 Real-Time Network Monitoring & Fault Prediction")

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

# Path Calculation (will be recalculated when generating the report)
start_node = st.sidebar.selectbox("Start Node", nodes)
end_node = st.sidebar.selectbox("End Node", nodes)

if st.sidebar.button("Generate Report"):
    ml_dijkstra = MLEnhancedDijkstra(network, predictive_analytics)
    path, cost = ml_dijkstra.shortest_path(start_node, end_node)
    # Trigger report generation only if valid path is found
    if path and cost:
        st.write("Generating report...")
        report_buffer = generate_report(traffic_data, predictive_analytics.fault_accuracy, predictive_analytics.congestion_mse, path, cost)
        st.download_button("Download Report", data=report_buffer, file_name="network_report.pdf")
    else:
        st.sidebar.error("No path found!")
    #if path:  # Check if path is found
        #st.sidebar.write(f"🔹 Shortest Path: {' → '.join(path)}")
        #st.sidebar.write(f"💰 Total Cost: {cost:.2f}")
        #report_buffer = generate_report(traffic_data, predictive_analytics.fault_accuracy, predictive_analytics.congestion_mse, path, cost)
        #st.download_button("Download Report", data=report_buffer, file_name="network_performance_report.pdf", mime="application/pdf")
   # else:
        #st.sidebar.error("No path found!")

# Trigger report generation only if valid path is found
if path and cost:
    st.write("Generating report...")
    report_buffer = generate_report(traffic_data, predictive_analytics.fault_accuracy, predictive_analytics.congestion_mse, path, cost)
    st.download_button("Download Report", data=report_buffer, file_name="network_report.pdf")
else:
    st.sidebar.error("No path found!")

# Traffic Visualization
st.subheader("📊 Network Traffic Overview")
fig = go.Figure()
fig.add_trace(go.Scatter(x=traffic_data["node"], y=traffic_data["load"], mode='markers', name='Load'))
fig.add_trace(go.Scatter(x=traffic_data["node"], y=traffic_data["congestion"], mode='lines', name='Congestion'))
st.plotly_chart(fig)

st.info("🚀 Use the sidebar to fetch metrics & compute shortest paths.")
