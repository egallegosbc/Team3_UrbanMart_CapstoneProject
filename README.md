# UrbanMart Customer Intelligence

RFM-based customer segmentation using K-Means clustering.

## Objective

Segment UrbanMart customers according to their purchasing
behavior in order to support targeted retention and marketing
strategies.

## Dataset

UrbanMart Transactions Dataset (2023-2024)

- 50,000 transactions
- 2,000 customers
- 9 variables
- Date range: 2023-01-01 to 2024-12-30

## Methodology

The model uses RFM customer features:

- Recency: days since last purchase
- Frequency: number of purchases
- Monetary: total customer spending

Frequency and Monetary are log-transformed before standardization.

K-Means models are evaluated from k=2 to k=10 using:

- Inertia / Elbow Method
- Silhouette Score

The best model uses k=3.

## Segments

- Champions
- Developing
- At Risk

## Model Performance

Final Silhouette Score: approximately 0.383.

## Installation

Create a Python virtual environment and install:

```bash
python -m pip install -r requirements.txt