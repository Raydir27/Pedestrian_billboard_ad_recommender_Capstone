# 🖥️ Pedestrian based Digital Billboard Ad Recommendation System

An **intelligent, real-time advertising platform** that transforms static billboards into **dynamic, context-aware displays**.  
By combining **computer vision** and **machine learning**, the system adapts ads in real-time based on live environmental and audience data—maximizing **engagement** and **ROI**.

---

## 🚀 Overview

The **Digital Billboard Ad Recommendation System** analyzes a live camera feed near a billboard, detects audience and traffic patterns, and recommends ad categories tailored to the current context.

Example scenario:
- **Morning rush hour** → Coffee shop promotions
- **Afternoon weekends** → Family-friendly retail ads
- **Evening weekdays** → Restaurants and entertainment

This repository contains the **core pipeline** to bring that vision to life.

---

## 💡 Motivation

Traditional digital billboards often:
- Use **static schedules**
- Rely on **broad demographic assumptions**
- Ignore **real-time environmental context**

**Problem:**  
A fixed ad rotation at 8 AM Monday is the same as 8 PM Sunday, missing engagement opportunities.

**Solution:**  
Our system makes billboard content **adaptive**:
- Analyzes **audience type**, **traffic density**, and **time-of-day**
- Dynamically selects the most relevant ad categories
- Enables **data-driven campaigns** that evolve with the environment

---

## 🧠 System Architecture

### **1. Video Ingestion**
- Accepts a **live RTSP/HTTP video stream** from a camera near the billboard

### **2. Multi-Model Object Detection**
- Ensemble vision pipeline:
  - **YOLOv8**: High-speed detection of pedestrians and their demographics
- Runs in **real-time** with GPU acceleration

### **3. Data Aggregation**
- Combines raw detections into:
  - Pedestrian counts
  - Pedestrian demographics(age, gender)
  - Time-of-day metadata

### **4. Downstream ML Model**
- Input: Aggregated environmental features
- Output: **Predicted ad category rankings**
- Example:  
  `["Coffee & Beverages", "Retail", "Dining & Entertainment"]`

### **5. Ad Recommendation**
- Produces a **ranked list** of recommended ad categories
- API-ready output for billboard content switching

---

## 📦 Tech Stack

| Component                  | Technology |
|----------------------------|------------|
| Object Detection           | YOLOv8     |
| Data Processing & Pipeline | Python, OpenCV, Pandas |
| ML Classifier              | Scikit-learn / LightGBM |
| Deployment                 | FastAPI, Docker |
| Streaming                  | RTSP, FFmpeg |

---

## 🛠️ Installation

### **1. Clone the repository**
```bash
git clone https://github.com/Raydir27/Pedestrian_billboard_ad_recommender_Capstone.git
cd Pedestrian_billboard_ad_recommender_Capstone
