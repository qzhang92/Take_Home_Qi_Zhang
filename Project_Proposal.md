# Project Proposal

The objective of this project is to deliver a fully functional, end-to-end Python script that can read generated code files, establish classification patterns, and detect anomalies. We will prioritize a runnable MVP (Minimum Viable Product) first, which means that the initial focus should be proving the core concept works

## Problem Description 
### Background
We have a link on a web page. There might be another link after you click that link. Then you might have a button to download a file. We assume that we might have downloaded 100 files. Let's call them file1 to file100  
We have a program, A.exe. It is taking the 100 files as an input. Then generate 100 corresponding files, called 1.txt, 2.txt, 3.txt, ... 100.txt.
### Problem
We would like to group the output 100 files. By putting them into groups, we can better analyze them. We might be able to access the grouping rules and a sample dataset to group them based on the sample group rules. And as the input file increases, the current grouping method may not be enough. There will be outliers and we need to add new groups to the current groups.    
We will use AI models and Python scripts to solve the problem

## Solution 
### Assumption
We assume the file download and excution at A.exe is already done/considered to be out of scope
### Process
1.  **Read the .txt file**
2.  **Extract dataset**
    *   Listed some possible solutions of model and made comparation
    *   I selected the ```Word2Vec/FastText``` approach, followed by ```Averaging Word Vectors``` to create the final document embedding
3.  **Model Training & Initial Clustering**
    * DBSCAN Application
    * Cluster Center Calculation
4.  **New File Classification & Incremental Model Update**
    * New File Encoding
    * Distance and Threshold Check
    * Decision
5.  **Reclustering**
    * Data Preparation and Full Retraining

## Executive Summary and Project Goal
The primary goal is to validate the core machine learning concept—combining ```Word2Vec``` embeddings with density-based clustering—before scaling.

| Component | Goal | Metric |
| :--- | :--- | :--- |
| **Feature Extraction** | Generate robust, fixed-length feature vectors. | $10,000 \times 100$ Feature Matrix ($\mathbf{X}$) generated. |
| **Clustering** | Define stable, initial file types (Groups $\mathbf{C_k}$). | $\mathbf{DBSCAN}$ executed; $\mathbf{C_k}$ calculated. |
| **Anomaly Detection** | Establish a reliable threshold for outlier detection. | Validated Anomaly Threshold ($\mathbf{\Theta}$) calculated. |

## Execution Plan (MVP Focus)

The focus is on delivering a runnable, end-to-end demonstration. Complex production engineering (e.g., database integration, Airflow) is deferred. The total estimated time is about 2.5 weeks

### Phase 1: Feature Foundation (Est. 1 Week)
* **Action:** Finalize custom tokenizer, handle 10k file reading, and train the full Word2Vec model.
* **Output:** Saved `Word2Vec.model` and the full $\mathbf{10,000} \times \mathbf{100}$ feature matrix $\mathbf{X}$.

### Phase 2: Core Logic Implementation (Est. 1 Week)
* **Action:** Run DBSCAN, tune $\mathbf{\epsilon}/\mathbf{min\_samples}$, calculate all **$\mathbf{C_k}$ centroids**, and determine the final **$\mathbf{\Theta}$ threshold**.
* **Output:** Defined cluster metrics and the fully calibrated set of parameters ready for classification.

### Phase 3: Update Cycle and Delivery (Est. 3-5 Days)
* **Action:** Develop the **`process_new_file()`** classification function (Steps 1-3) and the **`perform_full_retraining()`** simulation function (Step 5).
* **Output:** The final, runnable **`clustering_solution.py`** script demonstrating the entire end-to-end process, including the ability to adapt to new outlier data.

## Scope and Deferred Work

To meet the timeline, the following items are explicitly **excluded** from this MVP and are reserved for a subsequent Production Phase:

* **Performance Optimization:** Advanced parallelization (e.g., GPU/Dask) for matrix generation and retraining.
* **Infrastructure:** Persistent storage (e.g., PostgreSQL or S3) for models and vectors.
* **Deployment:** CI/CD pipelines, containerization (Docker), or dedicated scheduling (Airflow).
* **Extensive Hyperparameter Search:** Tuning is limited to a small, effective range for DBSCAN parameters.

## Possible Timeline for Future Work
When MVP is complete, we will move to our deferred work. We will focus on hardening, scaling, and deploying the solution to handle the large amount of real-time data reliably and continuously.
| Phase | Focus Area | Estimated Duration | Key Deliverables |
| :--- | :--- | :--- | :--- |
| **Phase 4: Infrastructure & Scalability** | **Data Pipeline and Storage** | 2 - 3 Weeks | **Persistent Storage:** Migrate vectors ($\mathbf{X}$), model files, and centroids ($\mathbf{C_k}$) from local files to a robust data store (e.g., PostgreSQL/S3).  <br> **Optimized I/O:** Implement efficient data loading/saving for the 10k file corpus during retraining. <br> **Model Optimization:** Implement and test speed optimizations for Word2Vec (e.g., Negative Sampling/Hierarchical Softmax). |
| **Phase 5: Automation & Reliability** | **Workflow Orchestration and Monitoring** | 3 - 4 Weeks | **Job Scheduling:** Implement **Airflow/Prefect** workflow to manage the scheduled, periodic execution of the **Batch Re-clustering** (Step 5).<br> **Containerization:** Dockerize the entire application for reliable deployment in a cloud environment (e.g., Kubernetes/ECS). <br> **Inference Service:** Deploy the Step 4 classification logic as a low-latency **microservice** accessible via API.<br> **Logging & Alerting:** Set up comprehensive logging and implement basic health checks and alerts (e.g., if the outlier rate spikes). |
| **Phase 6: Advanced Tuning & Robustness** | **Model Drift and Edge Case Handling** | 1 - 2 Weeks | **Advanced Hyperparameter Search:** Use GridSearchCV/Bayesian Optimization to find the optimal $\mathbf{\epsilon}$ and $\mathbf{min\_samples}$ across the full 10k dataset. <br> **Drift Detection:** Implement metrics to monitor the stability of the **Anomaly Threshold ($\mathbf{\Theta}$)** and automatically alert if significant model drift is detected. <br> **Final Documentation:** Complete detailed runbooks and maintenance guides. |

### **Total Estimated Duration for Future Work: 9 to 12 Weeks**

This duration emphasizes that building a robust system that can reliably process and maintain its intelligence over a continuous data stream is a multi-month engineering effort, even after the core ML algorithm is proven.