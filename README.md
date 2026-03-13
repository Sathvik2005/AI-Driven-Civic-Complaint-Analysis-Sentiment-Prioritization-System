# AI-Driven Civic Complaint Analysis & Sentiment Prioritization System

This repository contains the internship project work for civic complaint analysis, text cleaning, exploratory analysis, and NLP-based prioritization experiments.

## Infotact Technical Internship Program

### Data Science & Machine Learning Project Specifications and Evaluation Criteria

This project follows the Infotact Technical Internship Program framework for Data Science and Machine Learning. The goal is to execute the work in a professional engineering style, with structured experimentation, documented notebooks, consistent GitHub activity, and production-oriented thinking.

The broader internship specification covers three industry tracks:

- Government and Public Sector
- Agriculture
- Restaurant and Hospitality

This repository is focused on **Project 1: Government & Public Sector - AI-Driven Citizen Grievance & Sentiment Analysis System**.

The primary directive is to standardize the machine learning lifecycle so that all deliverables align with:

- enterprise-grade data workflow practices
- disciplined version control
- reproducible preprocessing and experimentation
- strong evaluation using mathematical and NLP-specific metrics

The internship evaluation depends heavily on transparency and continuity of work. A key requirement is maintaining all four weeks of GitHub commits and visible contributions across the internship period.

## Project 1 Overview

### Executive Problem Statement

Complaint resolution is a critical part of governance, but in many systems the process is still slow, fragmented, and heavily manual. Citizens often submit issues without clear routing, which leads to delays between departments and slower resolution times. On the administrative side, officials may spend hours reading complaint text manually before they can categorize or prioritize issues.

This project aims to build an AI-powered NLP system that can:

- ingest citizen feedback automatically
- classify complaints into the correct government department such as water, electricity, sanitation, or roads
- analyze sentiment and urgency
- help officials prioritize severe complaints faster

## Business Objectives and Key Performance Indicators

The business objective is to reduce complaint turnaround time and improve operational transparency. Instead of manually reviewing each message, the system should generate a structured queue of complaints that includes predicted department and urgency level.

Success will be measured using NLP and classification metrics, including:

- categorization accuracy
- Macro F1-score for sentiment classification
- class-sensitive performance for minority or urgent complaint groups

Special attention is required for minority classes such as urgent or critical complaints, because these are often the most important in real-world use.

## User Personas and Workflows

### Civic Official

Needs fast triage of complaints without reading every submission manually.

Workflow:
- reviews an AI-tagged dashboard
- identifies high-priority civic issues quickly
- dispatches action based on department and severity

### Citizen / User

Needs a fast and accurate routing experience.

Workflow:
- submits a free-text complaint
- receives correct routing without manually selecting departments

### Policy Maker

Needs a high-level view of public sentiment and issue patterns.

Workflow:
- tracks sentiment trends across complaint categories or regions
- uses aggregate feedback to understand public friction points

## Minimum Viable Product Specifications

The minimum viable product is centered on a dual-output NLP pipeline.

### Foundational Layer

- text preprocessing
- exploratory data analysis
- tokenization
- stopword removal
- lemmatization
- structured documentation in Jupyter notebooks

### Core Modeling Deliverable

The system must support:

1. multi-class department classification
2. sentiment or urgency analysis

Baseline models should begin with classical ML approaches such as:

- Naive Bayes
- Support Vector Machines
- Logistic Regression

After establishing baseline performance, the project should advance toward transformer-based models for stronger contextual understanding.

## Architectural Directives and Technology Stack

### Data Processing and NLP

- Python
- NLTK
- spaCy

These tools support tokenization, normalization, linguistic cleanup, and text preparation for downstream modeling.

### Predictive Modeling

- Scikit-learn
- HuggingFace Transformers such as BERT

Scikit-learn is suitable for fast baseline experimentation, while transformer architectures provide stronger context handling for complex public grievance text.

### Model Serving

- FastAPI

FastAPI will be used to expose trained models as REST APIs that can integrate with external dashboards or service portals.

## Four-Week Engineering Roadmap

### Week 1: Data Collection, Text Cleaning, and EDA

Week 1 begins with repository setup and dataset ingestion. The working dataset consists of public grievances or civic feedback such as synthetic or real 311 service request data.

Primary tasks for Week 1 include:

- setting up the Git repository
- loading and inspecting the dataset
- converting text to lowercase
- removing special characters and URLs
- lemmatizing complaint text
- generating Word Clouds
- generating n-gram frequency distributions
- documenting all exploratory analysis thoroughly in Jupyter notebooks

### Week 2: Topic Modeling and Department Categorization

In Week 2, the goal is to route complaints automatically to the correct department.

Tasks include:

- transforming cleaned text into TF-IDF or embedding-based representations
- training a supervised classifier such as Logistic Regression or Random Forest
- mapping complaint text to labels such as Sanitation, Water, or Transport
- validating the model with cross-validation to assess generalization

### Week 3: Sentiment Analysis and Urgency Scoring

Week 3 focuses on emotional tone and issue severity.

Tasks include:

- training a sentiment classifier for labels such as Positive, Neutral, Negative, and Critical/Urgent
- optionally fine-tuning a transformer model such as BERT or RoBERTa
- assigning a mathematical priority score to each complaint based on severity and sentiment output

### Week 4: API Development, Evaluation, and Final Delivery

Week 4 is dedicated to deployment, evaluation, and packaging.

Tasks include:

- evaluating models using confusion matrices and classification reports
- serializing the trained models
- building a FastAPI service that accepts raw complaint text as JSON input
- returning department prediction and priority score through the API
- submitting a fully documented GitHub repository with daily commits and clear architectural decisions

## Evaluation Expectations

To complete the internship successfully, the project should demonstrate:

- clear and consistent GitHub activity across all four weeks
- well-documented preprocessing and EDA notebooks
- reproducible experiments
- mathematically grounded evaluation
- practical model iteration and optimization
- a deployable API-based final delivery

## Repository Owner

- Kanithi Tirumala Satya Sathvik

## Contributors

- Kanithi Tirumala Satya Sathvik - kanithisathvik@gmail.com
- KOLLA LAKSHMI SAI NIVAS - kollanivas6@gmail.com
- Arun Pratap Negi - arun3gi@gmail.com
- Ramya H - ramyahari4353@gmail.com

## Note on Repository Access

Listing contributors in this file does not automatically grant GitHub write access. To let them commit and edit code directly, the repository owner must add them as collaborators with Write or Maintain permission in the GitHub repository settings.
