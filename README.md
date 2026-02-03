# AI-Powered Traffic Anomaly Detection System

[![Streamlit App|135x20](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://kabu-thesis.streamlit.app/)
![Python](https://img.shields.io/badge/Python-3.11%2B-blue)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange)

An end-to-end Machine Learning pipeline for detecting anomalies in urban traffic flow. This project conducts a comparative analysis of **Statistical**, **Traditional Machine Learning**, and **Deep Learning** approaches, evaluating their performance across varying feature sets and architectural scopes (Global vs. Local).
## Overview

Traffic incidents are associated with various costs that are environmental, economic, and social. The effective and rapid detection of such incidents that can cause congestion and endanger human life is critical.

This project utilizes real-world traffic sensor data (PeMS) to benchmark five distinct anomaly detection algorithms. A primary engineering objective was to evaluate the trade-off between **Individual** (sensor-specific) models and **Global** (network-wide) models to solve the scalability issues inherent in smart city deployments. Another objective is to investigate the effect of using different features to be given as input to the model (Speed-only, Occupancy-only, or a Bivariate combination of both).

More details about this comparative analysis can be found in the Streamlit app, specifically the Introduction page. Other implementation details that were not included there are the 5-five cross validation used for reporting, hyperparameter tuning using grid search, parallelization (when possible), and checkpointing and persistence methods.
## Results & Insights
The comparative study yielded several critical engineering insights regarding deployment in real-world scenarios:

- The **Global LSTM architecture** proved to be the most viable solution for smart city implementation. It successfully generalized across diverse sensor locations, eliminating the operational overhead of maintaining thousands of individual models while retaining competitive detection rates.

- Other algorithms failed in the global architecture, as the pooling of heterogenous data from multiple sensors broke the assumptions these algorithms make about the nature of the data.

- The feature set used was not shown to be of paramount importance; all feature sets yielded similar results, and the choice of model and deployment architecture tended to have a higher impact on the performance.

- The training of the LSTM is orders of magnitude more expensive than the other traditional models, but still very feasible under modern computing conditions.