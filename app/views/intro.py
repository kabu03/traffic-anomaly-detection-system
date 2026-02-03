import streamlit as st
from utils import project_root, apply_custom_style


def render_intro():

  st.header("Introduction & Research Questions")

  st.markdown("""
              
  Traffic anomalies, such as incidents that cause congestions, incur various economic, environmental, and social costs. The European Court of Auditors estimates that inefficiencies
               in urban mobility (and road congestion, in particular) cost the EU about **€110 billion per year**, 
              including lost working time, wasted fuel, increased emissions, as well as other indirect impacts on logistics.  

  **Intelligent Transportation Systems (ITS)** aim to use sensing and data analytics to make transport networks safer, 
              more efficient, and more sustainable. An important part of many ITS deployments is the 
              **Automatic Incident Detection (AID)** system, which can use artificial intelligence to detect traffic anomalies in real-time 
              from data collected by traffic sensors.  

  This project focuses on developing AI-based systems that detect such anomalies. The study
  answers three main research questions:
    - How do different anomaly-detection model families (i.e., statistical, machine learning, and deep learning) compare for detecting traffic incidents that cause congestion?
    - How much does the detection performance depend on the feature set used as input: speed only, occupancy only, or a bivariate combination of both?
    - Can a single "global" model operating across many sensor stations be competitive with individual models that are specific to each station, and what are the practical trade-offs in terms of detection performance, scalability, and computational cost?

  This web app focuses on presenting the results of the experiments conducted to answer these questions. It provides visualizations and comparisons of different models, feature sets, and architectural choices.
  """)

  st.divider()

  st.header("Model Comparison")

  st.markdown("""
              Various methods have been used in the literature for traffic anomaly detection, each with its strengths and weaknesses. This experiment uses five models that represent different families of anomaly detection techniques:
              - **Statistical Model:** Modified Z-Score (MZ)
              - **Traditional Machine Learning Models:** Isolation Forest (IF), Local Outlier Factor (LOF), One-Class SVM (OC-SVM)
              - **Deep Learning Model:** Long Short-Term Memory (LSTM)
              """)
  st.divider()

  st.header("Feature Set Comparison")

  st.markdown("""
              The choice of input features can greatly impact the performance of anomaly detection models. This experiment evaluates three different feature sets derived from traffic sensor data:
              - **Speed (Univariate):** Measured average speed (mph) at the detector.
              - **Occupancy (Univariate):** Proportion of time a detector is occupied by vehicles.
              - **Bivariate:** Combination of speed and occupancy.
              """)

  st.divider()

  st.header("Dataset & Evaluation Protocol")

  st.markdown("""
  The dataset used for this study is derived from the Caltrans Performance Measurement System (PeMS) dataset. PeMS data is collected in real-time from nearly 40,000 individual detectors spanning the freeway system across all major metropolitan areas of California. 
              In this experiment, **175** congestion-inducing traffic incidents were selected, where each incident is represented 
              by a time series stream that is **252** hours long with aggregated data in **5-minute** intervals.  
              The first **240** hours correspond to typical traffic 
              conditions, while the last **12** hours contain the incident.

  Model evaluation depends on three key metrics:
  - **Detection Rate (DR):** The percentage of incidents that have been correctly detected.
  - **False Alarm Rate (FAR):** The percentage of normal instances incorrectly classified as anomalies.
  - **Mean Time to Detection (MTTD):** The average time needed to detect the incident (i.e., detection speed).  
              
  Ideally, a model should have a high DR, low FAR, and low MTTD.""")

  st.divider()

  st.header("Methodology Comparison")

  st.markdown("""
  The **Individual Pipeline** represents how well models perform when they are small, local, and
  specialized for each time-series stream. In the Individual Pipeline, we train an individual
  model for each time-series stream.

  The goal of the **Global Pipeline** is to build a single general model
              that can understand and detect patterns across many different streets,
              city regions, and traffic sensors.
              
  For both the individual and global pipelines, a **5-fold cross-validation** protocol is used. In each fold,
  the **175 streams are split into 140 tuning streams and 35 reporting streams**.  
  The tuning streams are used for hyperparameter tuning (and training the global model's weights
  in the global pipeline), while the reporting streams are reserved for evaluation.  
  The final reported metrics are averaged over the 5 folds.
              
  Pseudocode for the two pipelines is provided below.
  """)
  st.space()
  col1, col2 = st.columns(2)

  with col1:
      st.subheader("Individual Pipeline")
      st.info("Trains a separate model for each specific stream, showcasing localized and specialized behavior.")
      st.code("""
  # Legend:
  #  t = tuning stream, r = reporting stream
  #  z_t = scores on test window of tuning stream t; Z = pooled tuning scores
  #  tau = decision threshold; cost_hp = minimized cost for hyperparameters
  #  Scaling logic omitted for brevity

  for model_type in {MZ, IF, LOF, OC-SVM, LSTM}:
    for feature_set in {Speed, Occupancy, Bivariate}:
      for fold k in {1..5}:

        split streams (175) -> tuning (140), reporting (35)

        best_cost = +inf; best_hp = None; best_tau = None
        for hp_config in HP_SPACE(model_type):

          Z = []
          for t in tuning_streams:
            train per-stream model (hp_config) on t[0:240h]
            z_t = score(t[240h:252h]); append z_t to Z

          (tau_hp, cost_hp) = argmin_tau Cost_Function(tau, pool(Z))
          if cost_hp < best_cost:
            best_cost = cost_hp; best_hp = hp_config; best_tau = tau_hp

        save best_hp, best_tau

        for r in reporting_streams:
          train model (best_hp) on r[0:240h]
          scores = score(r[240h:252h])
          apply best_tau; save outputs

        metrics_per_fold[k] = aggregate within-fold

      reported_metrics = mean(metrics_per_fold[1..5])
      """)

  with col2:
      st.subheader("Global Pipeline")
      st.info("Trains a single model on data from multiple different stations, leveraging shared patterns.")
      st.code("""
  # Legend:
  #  s = stream (generic), r = reporting stream, tau = decision threshold
  #  Z = pooled validation scores, cost_hp = minimized cost for hyperparameters
  #  Scaling/encoding logic omitted for brevity

  for model_type in {MZ, IF, LOF, OC-SVM, LSTM}:
    for feature_set in {Speed, Occupancy, Bivariate}:
      for fold k in {1..5}:

        split streams (175) -> tuning (140), reporting (35)
        split tuning -> train_tune (112), valid_tune (28)   

        best_cost = +inf; best_hp = None

        for hp_config in HP_SPACE(model_type):

          global_model_h = train_global(hp_config, data=train_tune[0:240h])

          Z = []
          for s in valid_tune:
            Z.append( score_global(global_model_h, s[240h:252h]) )

          (tau_hp, cost_hp) = argmin_tau Cost_Function(tau, pool(Z))
          if cost_hp < best_cost:
            best_cost = cost_hp; best_hp = hp_config

        save best_hp

        # Final retraining on all tuning streams
        global_model_best = train_global(best_hp, data=tuning[0:240h])

        # Threshold recalibration on full tuning set
        Z_tune = []
        for s in tuning:
          Z_tune.append( score_global(global_model_best, s[240h:252h]) )

        (recalibrated_tau, _) = argmin_tau Cost_Function(tau, pool(Z_tune))

        for r in reporting_streams:
          scores = score_global(global_model_best, r[240h:252h])
          apply recalibrated_tau; save outputs

        metrics_per_fold[k] = aggregate within-fold

      reported_metrics = mean(metrics_per_fold[1..5])
      """)