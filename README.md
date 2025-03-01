# Course Recommendation System

## Overview

The Course Recommendation System is a web application built using FastAPI that suggests courses based on user-defined skills. By leveraging machine learning techniques like TF-IDF vectorization and cosine similarity, the application provides users with personalized course recommendations from a dataset of available courses.

## Features

- **Dynamic Course Recommendations**: Users can input skills, and the system returns relevant course suggestions.
- **TF-IDF Vectorization**: Efficiently analyzes course categories to determine similarity.
- **Cosine Similarity Calculation**: Measures how similar courses are to one another based on their categories.
- **FastAPI Framework**: Designed for speed and ease of use in developing APIs.

## Technologies Used

- **Python**: Main programming language.
- **FastAPI**: Modern web framework for building APIs.
- **Pandas**: Data manipulation and analysis library.
- **Scikit-Learn**: Library for machine learning and statistical modeling.
- **Uvicorn**: ASGI server for running the FastAPI application.

## Installation

1. **Clone the repository**:
   ```bash
   git clone https://github.com/Sangeerths/Course-Recommendation-System.git

2. Navigate into the directory:

   ```bash
   cd Course-Recommendation-System

3. install required packages:
   ```bash
   pip install fastapi uvicorn pandas scikit-learn

4.Run the application:
   ```bash
uvicorn main:app --reload
