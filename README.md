# AI-Based Real-Time Vehicle Counting System 🚗📊

## 📌 Project Overview
This project is an AI-powered **Real-Time Vehicle Counting System** built using **Computer Vision** and **Deep Learning**. It utilizes the **YOLOv8** object detection model to accurately detect and track various types of vehicles (cars, motorcycles, buses, and trucks) in a video stream. A virtual counting line is implemented to count the vehicles as they pass through, and the results are displayed on the screen in real-time.
<img width="1920" height="1080" alt="Screenshot (1930)" src="https://github.com/user-attachments/assets/cf3ae0c2-0abf-4e8b-8aee-02b8acd01622" />


## ✨ Features
* **Real-time Object Detection:** Uses the state-of-the-art YOLOv8 nano model (`yolov8n.pt`) for fast and accurate detection.
* **Multi-Class Detection:** Specifically configured to detect and count:
    * Cars 🚗
    * Motorcycles 🏍️
    * Buses 🚌
    * Trucks 🚛
* **Object Tracking:** Assigns a unique ID to each detected vehicle to prevent double-counting.
* **Virtual Counting Line:** Counts vehicles only when their centroid crosses a predefined line on the screen.
* **Visual Annotations:** Draws bounding boxes around vehicles and displays the real-time count on the video feed.

## 🛠️ Technologies & Tools Used
* **Programming Language:** Python 3
* **Computer Vision Library:** OpenCV (`cv2`)
* **Deep Learning Framework:** Ultralytics YOLOv8
* **Development Environment:** VS Code / Google Colab (Tested on both)

## 🚀 How to Run the Project Locally

### 1. Prerequisites
Make sure you have Python installed on your system. You can download it from [python.org](https://www.python.org/).

