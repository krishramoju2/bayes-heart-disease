import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from pgmpy.models import BayesianModel
from pgmpy.estimators import MaximumLikelihoodEstimator, VariableElimination
import matplotlib.pyplot as plt
import networkx as nx

# Load cleaned dataset
df = pd.read_csv("cleaned_heart_disease.csv")

# Define Bayesian Network structure
model = BayesianModel([("age", "fbs"),
                       ("fbs", "target"),
                       ("target", "chol"),
                       ("target", "thalach")])

# Fit the model using MLE
model.fit(df, estimator=MaximumLikelihoodEstimator)

# Inference engine
infer = VariableElimination(model)

# Query 1: What is P(target | age=0.7)?
q1 = infer.query(variables=["target"], evidence={"age": 0.7})
print("P(target | age=0.7):\n", q1)

# Query 2: Cholesterol distribution when target=1 (has disease)
q2 = infer.query(variables=["chol"], evidence={"target": 1})
print("Cholesterol distribution | target=1:\n", q2)

# Query 3: Thalach distribution when fbs=0 and target=0
q3 = infer.query(variables=["thalach"], evidence={"fbs": 0, "target": 0})
print("Thalach | fbs=0, target=0:\n", q3)

# Optional: Visualize the network
plt.figure(figsize=(8, 6))
nx.draw(model, with_labels=True, node_size=3000, node_color='lightblue', font_size=14, font_weight='bold')
plt.title("Bayesian Network Structure")
plt.show()
