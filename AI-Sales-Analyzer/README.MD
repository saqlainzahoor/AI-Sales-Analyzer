AI Sales Analyzer

A Python-based sales analysis and machine learning project that analyzes sales data, visualizes product performance, and predicts sales using Linear Regression.

Features
Reads sales data from CSV
Calculates total sales revenue
Identifies the best-selling product
Displays a sales summary
Creates a sales visualization
Predicts sales based on quantity and price
Evaluates the machine learning model using MAE and R² Score
Technologies
Python
Pandas
NumPy
Matplotlib
Scikit-learn
Linear Regression
Project Structure
AI-Sales-Analyzer/
│
├── main.py
├── sales_data.csv
└── README.md
How It Works
Sales data is loaded from sales_data.csv.
Total sales are calculated using quantity and price.
The program generates a sales summary.
A bar chart visualizes sales by product.
A Linear Regression model is trained using quantity and price.
The user enters quantity and price for a new prediction.
The model predicts the expected sales amount.
Model performance is evaluated using MAE and R² Score.
Example
==============================
       AI SALES PREDICTION
==============================

Enter Quantity: 10
Enter Price: 80000

Predicted Sales: 596,494.91
Model Evaluation

The current practice dataset produces approximately:

Mean Absolute Error: 44,574.78
R² Score: 0.8968

Note: This project uses a small practice dataset for learning purposes. The model evaluation results should not be interpreted as real-world business performance.

Project Purpose

This project was created to practice:

Python programming
Data analysis
Data visualization
Machine learning
Model evaluation
Git and GitHub

It is part of my portfolio development for an AI/ML career.

Author

Saqlain Zahoor

GitHub: saqlainzahoor