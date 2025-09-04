-----

# 🖥️ Vis-Ad: Real-Time Targeted Advertisements using Billboard

An **intelligent, real-time advertising system** that transforms static billboards into **dynamic, context-aware displays**. By combining **computer vision** and **machine learning**, the system adapts ads in real-time based on live audience data—such as age, gender, clothing style, and accessories—maximising **engagement** and **ROI**.

-----

## 🚀 Overview

The **Vis-Ad system** analyses a live camera feed near a billboard, detects audience demographics and apparel features, and recommends ad categories tailored to the current frame/feed context.

Example scenario:

  - **Pedestrian 1:** A young woman (20s) in formal business attire (suit) with a briefcase.
      - **Top 5 Ad Categories:** `["Luxury Watches", "Professional Services", "High-end Dining", "Business Travel", "Financial Planning"]`
  - **Pedestrian 2:** A man (50s) in casual sportswear (t-shirt and track pants) with a water bottle.
      - **Top 5 Ad Categories:** `["Fitness & Health", "Sports Gear", "Casual Dining", "Beverages", "Outdoor Recreation"]`
  - **System Recommendation:** The system calculates a weighted average of all detected ad categories for the entire crowd and displays the ad with the highest combined score, ensuring the content is relevant to the whole audience.

This repository contains the **core pipeline** to bring that vision to life.

-----

## 💡 Motivation

Traditional billboards often:

  - Use **static schedules**
  - Rely on **broad demographic assumptions**
  - Ignore **real-time environmental context**

**Problem:** A fixed ad rotation at 8 AM Monday is the same as 8 PM Sunday, leading to irrelevant impressions and wasted ad spending.

**Solution:** Our system makes billboard content **adaptive**:

  - Analyzes **audience type**, **traffic density**, and **time-of-day**
  - Dynamically selects the most relevant ad categories
  - Enables **data-driven campaigns** that evolve with the environment

This approach bridges the gap between the digital precision of online advertising and the broad visibility of physical billboards.

-----

## 🧠 System Architecture

The Vis-Ad system follows a modular pipeline with four primary stages:

### **1. Input Acquisition**

  - Accepts a **live HD video stream** (720p–1080p, 15–30 FPS) from a camera near the billboard.
  - Frames are sampled at intervals to balance accuracy with computational efficiency.

### **2. Preprocessing**

  - Detected persons are cropped into regions of interest (ROI) to focus on relevant visual cues.

### **3. Feature Extraction**

  - **Demographic analysis**: Estimates age group and gender using trained convolutional neural networks (CNNs).
  - **Apparel recognition**: Identifies clothing styles (e.g., t-shirts, suits, pants) and detects accessories like spectacles or purses using a multi-label CNN/YOLO-based approach.

### **4. Ad Recommendation Module**

  - A machine learning recommender system maps the extracted feature vectors (demographics + apparel/accessories) to suitable ad categories.
  - The most relevant advertisement is then displayed to the audience in real time.

-----

## 📦 Tech Stack

| Component | Technology |
|---|---|
| Programming Language | Python |
| Computer Vision | OpenCV |
| Deep Learning | TensorFlow, PyTorch |
| Object Detection | YOLOv8 |
| Deployment | FastAPI, Docker |
| Streaming | RTSP, FFmpeg |

-----

## 🛠️ Installation

**Note:** The following is a placeholder for the installation process.

### **1. Clone the repository**

```bash
git clone https://github.com/Raydir27/Pedestrian_billboard_ad_recommender_Capstone.git
cd Pedestrian_billboard_ad_recommender_Capstone
```

### **2. Set up the environment**

```bash
# Create a virtual environment
python -m venv venv
source venv/bin/activate  # On Windows use `venv\Scripts\activate`

# Install dependencies
pip install -r requirements.txt
```

### **3. Run the system**

```bash
# Run the FastAPI application
uvicorn app.main:app --host 0.0.0.0 --port 8000
```
