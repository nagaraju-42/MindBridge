"""
src/llm_evaluator.py
MindBridge — LLM Safety Net Evaluator
Uses OpenRouter API to evaluate borderline cases and catch false negatives.
"""
from openai import OpenAI
from src.config import OPENROUTER_API_KEY
import streamlit as st

def evaluate_borderline_text(text: str, current_confidence: float) -> dict:
    """
    Calls an LLM via OpenRouter to evaluate if a text contains hidden depression markers.
    Returns a dict with 'is_high_risk' (bool) and 'reasoning' (str).
    """
    if OPENROUTER_API_KEY == "NOT_SET" or not OPENROUTER_API_KEY:
        return {"is_high_risk": False, "reasoning": "LLM API Key not configured."}

    client = OpenAI(
        base_url="https://openrouter.ai/api/v1",
        api_key=OPENROUTER_API_KEY,
    )

    prompt = f"""
You are an expert clinical psychologist evaluating a student's text for depression triage.
A machine learning model scored this text with a depression probability of {current_confidence:.2f} (Grey Area).
Read the text and determine if it contains hidden depression markers, sarcasm masking pain, passive suicidal ideation, or exhaustion that a simple Bag-of-Words AI might miss.

Text: "{text}"

Analyze the text and output your response in EXACTLY this format (2 lines):
RISK: [HIGH or LOW]
REASON: [1 sentence explaining your clinical reasoning]
"""

    try:
        response = client.chat.completions.create(
            model="meta-llama/llama-3-8b-instruct:free",
            messages=[
                {"role": "system", "content": "You are a clinical evaluator API. Follow the output format strictly."},
                {"role": "user", "content": prompt},
            ],
            temperature=0.0,
            max_tokens=150
        )
        
        result_text = response.choices[0].message.content.strip()
        lines = result_text.split('\n')
        
        is_high_risk = False
        reasoning = "Could not parse LLM reasoning."
        
        for line in lines:
            if line.startswith("RISK:"):
                is_high_risk = "HIGH" in line.upper()
            elif line.startswith("REASON:"):
                reasoning = line.replace("REASON:", "").strip()
                
        return {"is_high_risk": is_high_risk, "reasoning": reasoning}

    except Exception as e:
        print(f"LLM API Error: {e}")
        return {"is_high_risk": False, "reasoning": f"LLM Evaluation failed: {str(e)}"}
