# 🔑 Last Lock Team 1 - Locked-In

## 📝 About

The **Locked-In Backend** serves as the core for processing and identifying room data efficiently. This project is aimed at creating a scalable and reliable backend architecture for handling room-related data. This service is optimized for room identification and processing, providing efficient data handling and API integration, eventually automating geospatial data with AI.

---

## ✨ Features

- 🌍 GeoJson Conversion
- 🏠 Room Identification
- 🗄️ Database Integration
- 🚀 Optimized Processes For Faster Execution

---

## 🛠️ Tech Stack

### 💻 Programming Language
- ![Python](https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white)

---

### 🧩 Frameworks/Libraries
- ![FastAPI](https://img.shields.io/badge/FastAPI-009688?style=for-the-badge&logo=fastapi&logoColor=white)
- ![OpenCV](https://img.shields.io/badge/OpenCV-5C3EE8?style=for-the-badge&logo=opencv&logoColor=white)
- ![React](https://img.shields.io/badge/React-61DAFB?style=for-the-badge&logo=react&logoColor=black)

---

### 🗄️ Database
- ![MongoDB](https://img.shields.io/badge/MongoDB-47A248?style=for-the-badge&logo=mongodb&logoColor=white)

---

### ☁️ Other Tools
- ![Cloudinary](https://img.shields.io/badge/Cloudinary-3448C5?style=for-the-badge&logo=cloudinary&logoColor=white)
- ![Uvicorn](https://img.shields.io/badge/Uvicorn-FF69B4?style=for-the-badge)
- ![PyTesseract](https://img.shields.io/badge/PyTesseract-0C66C2?style=for-the-badge)


## 📦 Dependencies

This project relies on the following key dependencies:

| Dependency           | Purpose                                                   |
|-----------------------|-----------------------------------------------------------|
| **FastAPI**          | Framework for building APIs.                              |
| **Uvicorn**          | ASGI server for running FastAPI applications.             |
| **python-multipart** | For handling file uploads.                                |
| **PyMongo**          | MongoDB client for database operations.                   |
| **Cloudinary**       | Cloud-based image storage and management.                 |
| **python-dotenv**    | Environment variable management.                          |
| **OpenCV**           | Image processing and computer vision tasks.              |
| **NumPy**            | Numerical operations and array handling.                 |
| **scikit-image**     | Advanced image processing and manipulation.               |
| **PyTesseract**      | Optical character recognition (OCR) capabilities.         |
| **SciPy**            | Scientific computing and advanced numerical methods.      |

---

## ⚙️ Setup

Follow these steps to set up the project locally:

### Prerequisites
- Python 3.8 or later installed on your system.
- MongoDB instance (local or cloud).
- Cloudinary account for image management.

---

### Installation Steps

1. **Clone the Repository**:
   ```bash
   git clone https://github.com/ArianAbbaszadeh/locked-in-backend.git
   cd locked-in-backend

2. **Set Up a Virtual Environment**:
   ```bash
   python -m venv env
   source env/bin/activate  # For MacOS/Linux
   env\Scripts\activate     # For Windows

3. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt

4. **Set Up Environment Variables**: Create a .env file in the root directory and add the following
   ```bash
   MONGO_URI=your_mongodb_uri
   CLOUDINARY_URL=your_cloudinary_url

5. **Run the Server**:
   ```bash
   uvicorn main:app --reload

6. **Access the Application**: Open your browser and navigate to:
   ```bash
   http://127.0.0.1:8000

---

## Authors
