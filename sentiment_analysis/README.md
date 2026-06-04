# 🛍️ Customer Review Sentiment Analysis

Analysis of 40,000+ Amazon customer reviews using Python and NLP to extract 
business insights about customer satisfaction.

![Python](https://img.shields.io/badge/Python-3.12-blue)
![Streamlit](https://img.shields.io/badge/Streamlit-1.58-red)
![VADER](https://img.shields.io/badge/NLP-VADER-green)

---

## 📊 Live Dashboard
Run locally with:
```bash
python -m streamlit run dashboard.py
```

---

## 🎯 Project Objectives
- Analyze customer sentiment across 40,000 reviews
- Classify reviews into business topics
- Identify key pain points to improve customer satisfaction
- Visualize insights through an interactive dashboard

---

## 🔍 Key Findings
- **90.1%** of reviews are positive — strong overall customer satisfaction
- **Product Quality** is the most discussed topic (31.1%)
- **Delivery & Shipping** accounts for 24.4% of all negative reviews
- **Taste & Flavor** dominates word frequency — typical for food products

---

## 🛠️ Tech Stack
| Tool | Usage |
|------|-------|
| Python | Core programming language |
| Pandas & NumPy | Data manipulation |
| VADER | Sentiment analysis |
| Matplotlib & Seaborn | Data visualization |
| Streamlit | Interactive dashboard |
| NLTK | Text preprocessing |

---
## 📂 Dataset

This project uses the **Amazon Fine Food Reviews** dataset from Kaggle.

Since both data files are too large for GitHub, you need to generate them:

### Step 1: Download the raw data
1. Go to [Amazon Fine Food Reviews](https://www.kaggle.com/datasets/snap/amazon-fine-food-reviews)
2. Download `Reviews.csv`
3. Place it in the `data/` folder

### Step 2: Generate the analyzed data
1. Open `notebooks/sentiment_analysis.ipynb`
2. Run all cells from top to bottom
3. This will automatically generate `Data/analyzed_reviews.csv`

### Step 3: Run the dashboard
```bash
python -m streamlit run dashboard.py
```
---
## 📁 Project Structure     
sentiment_project/
│
├── Data/
│   ├── Reviews.csv               # Raw dataset
│   └── analyzed_reviews.csv      # Processed results
│
├── notebooks/
│   └── sentiment_analysis.ipynb  # Full analysis
│
├── Images/                       # Generated charts
│
├── dashboard.py                  # Streamlit dashboard
└── README.md
---

## ⚙️ Installation

```bash
# Clone the repository
git clone https://github.com/AyaRahmouny/customer-review-sentiment-analysis

# Install dependencies
pip install pandas numpy matplotlib seaborn nltk vaderSentiment wordcloud streamlit --user

# Run the dashboard
python -m streamlit run dashboard.py
```

---

## 📈 Dashboard Preview

### Sentiment Distribution
![Sentiment](Images/sentiment_distribution.png)

### Topic Analysis
![Topics](Images/topic_distribution.png)

### Word Cloud
![WordCloud](Images/wordcloud.png)

---

## 💡 Limitations & Future Improvements
- VADER struggles with mixed-sentiment reviews
- Keyword-based topic classification can be replaced with LDA
- Could be improved with a fine-tuned transformer model (BERT)

---

## 👩‍💻 Author
**Aya Rahmouny** — Data Analyst  
[LinkedIn](https://linkedin.com/in/aya-rahmouny) | 
[GitHub](https://github.com/AyaRahmouny)
