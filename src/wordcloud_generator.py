"""
src/wordcloud_generator.py
MindBridge — Real-time WordCloud generation for EDA.
"""
import pandas as pd
from wordcloud import WordCloud, STOPWORDS
import matplotlib.pyplot as plt
import io

import streamlit as st

@st.cache_data(show_spinner="🎨 Rendering Word Clouds from dataset (runs once)...")
def generate_wordclouds(csv_path: str):
    """
    Reads the dataset, splits into depressed (1) and non-depressed (0),
    and generates two WordCloud images as bytes for Streamlit rendering.
    """
    try:
        df = pd.read_csv(csv_path)
        
        # Filter texts
        depressed_text = " ".join(df[df['is_depression'] == 1]['clean_text'].dropna().astype(str).tolist())
        happy_text = " ".join(df[df['is_depression'] == 0]['clean_text'].dropna().astype(str).tolist())
        
        custom_stopwords = set(STOPWORDS)
        custom_stopwords.update(['feel', 'feeling', 'like', 'know', 'want', 'really', 'just'])

        # Generate Sad Cloud
        wc_sad = WordCloud(
            width=800, height=400, 
            background_color='black', 
            colormap='Reds',
            stopwords=custom_stopwords
        ).generate(depressed_text)

        # Generate Happy Cloud
        wc_happy = WordCloud(
            width=800, height=400, 
            background_color='white', 
            colormap='Greens',
            stopwords=custom_stopwords
        ).generate(happy_text)
        
        # Save to BytesIO
        sad_buf = io.BytesIO()
        plt.figure(figsize=(10, 5))
        plt.imshow(wc_sad, interpolation='bilinear')
        plt.axis('off')
        plt.tight_layout(pad=0)
        plt.savefig(sad_buf, format='png')
        plt.close()
        
        happy_buf = io.BytesIO()
        plt.figure(figsize=(10, 5))
        plt.imshow(wc_happy, interpolation='bilinear')
        plt.axis('off')
        plt.tight_layout(pad=0)
        plt.savefig(happy_buf, format='png')
        plt.close()
        
        return sad_buf.getvalue(), happy_buf.getvalue()
        
    except Exception as e:
        print(f"Error generating WordClouds: {e}")
        return None, None
