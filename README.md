
# Sign Language Gesture Recognition and Continuous Translation System

Abstract:

This repository contains the source code, experimental notebooks, and application architecture for an end-to-end Machine Learning system designed to recognize sign language gestures. The project advances beyond isolated, static image classification by implementing a continuous translation module capable of formulating coherent sequences (e.g., "You Like Released Birds"). The system integrates a robust predictive model with a dedicated user interface, demonstrating a complete pipeline from data processing and model training to deployment and real-time inference.


Repository Architecture:

The project adheres to a modular architecture, separating the experimental machine learning environment from the deployment-ready application source code.

models/                             
-Compiled model

notebooks/                          
TrainGestureLS.ipynb

TestImageGesture.ipynb

TestImageGestureSen.ipynb

src/                                
SignGestureRecognitionFinal/ - Frontend user interface and inference integration



Core Modules and Methodology:
1. Model Training (TrainGestureLS.ipynb)
This module represents the definitive training pipeline for the model. It handles the ingestion of raw visual data, preprocessing, feature extraction, and the execution of the training loop. The output of this notebook is the optimized model weights utilized by the downstream application interface.

2. Empirical Evaluation (TestImageGesture.ipynb)
This notebook conducts a rigorous evaluation of the trained model against isolated gestures. It computes core performance metrics and generates confusion matrices to visualize the classification accuracy across distinct classes, ensuring the model's reliability on unseen data before deployment.

3. Continuous Sequence Formulation (TestImageGestureSen.ipynb)
Addressing the challenge of sequential data processing, this module demonstrates the system's capacity for continuous translation. It transitions the inference engine from predicting single, static gestures to interpreting dynamic inputs, successfully formulating complete and syntactically coherent sentences.

4. System Interface (SignGestureRecognitionFinal)
Bridging the gap between data science and software engineering, this directory contains the application module. It serves as the frontend user interface, seamlessly integrating the pre-trained LS model to facilitate real-time, user-facing gesture translation.



Authors:

Dinesh Kumar S, Joseph Xavier M, Vignesh S
