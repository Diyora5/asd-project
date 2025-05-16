# 🤍 AutiScreen: ASD Prediction App 🧠

**AutiScreen** is a high-impact, research-driven tool that leverages advanced machine learning to predict the likelihood of Autism Spectrum Disorder (ASD) in toddlers aged 10–36 months. Designed for parents, healthcare professionals, and caregivers, AutiScreen provides early, interpretable, and actionable insights based on a scientifically validated 40-question yes/no survey.

---

## 📑 Table of Contents

- [Overview](#overview)
- [Installation](#installation)
- [Usage](#usage)
- [Results](#results)

---

## Overview

Early detection of ASD is crucial for timely intervention and improved developmental outcomes. AutiScreen combines robust data preprocessing, state-of-the-art machine learning models, and a user-friendly web interface to deliver reliable ASD risk assessments. The app is built with scalability, privacy, and scientific rigor in mind.

---

## Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/Diyora5/ASD_Project.git
   cd ASD_Project
   ```

2. **Create and activate a virtual environment**
   - Windows:
     ```bash
     python -m venv .venv
     .venv\Scripts\activate
     ```
   - macOS/Linux:
     ```bash
     python3 -m venv .venv
     source .venv/bin/activate
     ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **(Optional) Build and run with Docker**
   ```bash
   docker build -t asd-app .
   docker run -p 80:80 asd-app
   ```

---

## Usage

- **Run the app locally**
  - Windows:
    ```bash
    python src/app.py
    ```
  - macOS/Linux:
    ```bash
    python3 src/app.py
    ```
- Access the app at [http://localhost:80](http://localhost:80) or the port specified.

---

## Results

- **AutiScreen** delivers probabilistic, interpretable ASD risk predictions for toddlers.
- The app empowers early intervention by providing actionable insights to caregivers and professionals.
- All predictions are based on validated screening questions and state-of-the-art ML models.

---

