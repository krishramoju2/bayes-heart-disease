try:
    import pandas as pd
    import numpy as np
    from sklearn.preprocessing import MinMaxScaler
    from pgmpy.models import BayesianNetwork
    from pgmpy.estimators import MaximumLikelihoodEstimator
    from pgmpy.inference import VariableElimination
    import matplotlib.pyplot as plt
    import networkx as nx
except ImportError as e:
    print("🚨 Required library not found:", e)
    print("➡️ Try running: pip install pandas numpy scikit-learn pgmpy matplotlib networkx")
    exit(1)

# Load dataset
try:
    df = pd.read_csv("cleaned_heart_disease.csv")
except FileNotFoundError:
    print("❌ 'cleaned_heart_disease.csv' not found. Please run the cleaning script first.")
    exit(1)

# Define Bayesian Network structure
model = BayesianNetwork([
    ("age", "fbs"),
    ("fbs", "target"),
    ("target", "chol"),
    ("target", "thalach")
])

# Train model
try:
    model.fit(df, estimator=MaximumLikelihoodEstimator)
except Exception as e:
    print("❌ Error during training:", e)
    exit(1)

# Inference engine
infer = VariableElimination(model)

# Inference Queries
try:
    print("🔍 P(target | age=0.7):")
    print(infer.query(variables=["target"], evidence={"age": 0.7}))

    print("\n🔍 Cholesterol distribution | target=1:")
    print(infer.query(variables=["chol"], evidence={"target": 1}))

    print("\n🔍 Thalach | fbs=0, target=0:")
    print(infer.query(variables=["thalach"], evidence={"fbs": 0, "target": 0}))
except Exception as e:
    print("❌ Error during inference:", e)

# Visualize network
plt.figure(figsize=(8, 6))
nx.draw(model, with_labels=True, node_size=3000, node_color='lightblue', font_size=12, font_weight='bold')
plt.title("Bayesian Network Structure")
plt.show()


