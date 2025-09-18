# Text Classification with Hugging Face
# Simple sentiment analysis project

import pandas as pd
import numpy as np
from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import torch
import matplotlib.pyplot as plt
import seaborn as sns

class TextClassifier:
    def __init__(self, model_name="cardiffnlp/twitter-roberta-base-sentiment-latest"):
        """
        Initialize the text classifier with a pre-trained model
        """
        self.model_name = model_name
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name)
        self.classifier = pipeline("sentiment-analysis", 
                                 model=self.model, 
                                 tokenizer=self.tokenizer,
                                 return_all_scores=True)
    
    def predict_sentiment(self, texts):
        """
        Predict sentiment for a list of texts
        """
        if isinstance(texts, str):
            texts = [texts]
        
        results = []
        for text in texts:
            prediction = self.classifier(text)
            # Get the highest scoring label
            best_pred = max(prediction[0], key=lambda x: x['score'])
            results.append({
                'text': text,
                'label': best_pred['label'],
                'confidence': best_pred['score']
            })
        
        return results
    
    def batch_predict(self, texts, batch_size=16):
        """
        Predict sentiment for large batches of text
        """
        results = []
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i+batch_size]
            batch_results = self.predict_sentiment(batch)
            results.extend(batch_results)
        
        return results

def create_sample_data():
    """
    Create sample data for demonstration
    """
    sample_texts = [
        "I love this product! It's amazing!",
        "This is the worst experience ever.",
        "The weather is okay today.",
        "I'm so happy with this purchase!",
        "Not satisfied with the service.",
        "This movie was fantastic!",
        "The food was terrible.",
        "I feel neutral about this.",
        "Excellent customer support!",
        "Could be better, but not bad."
    ]
    
    return sample_texts

def analyze_results(results):
    """
    Analyze and visualize the classification results
    """
    df = pd.DataFrame(results)
    
    # Print summary
    print("Sentiment Analysis Results:")
    print("=" * 50)
    print(f"Total texts analyzed: {len(results)}")
    print("\nLabel distribution:")
    print(df['label'].value_counts())
    
    print(f"\nAverage confidence: {df['confidence'].mean():.3f}")
    print(f"Confidence std: {df['confidence'].std():.3f}")
    
    # Plot results
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Label distribution
    df['label'].value_counts().plot(kind='bar', ax=ax1, color='skyblue')
    ax1.set_title('Sentiment Distribution')
    ax1.set_xlabel('Sentiment')
    ax1.set_ylabel('Count')
    ax1.tick_params(axis='x', rotation=45)
    
    # Confidence distribution
    sns.histplot(data=df, x='confidence', hue='label', ax=ax2, alpha=0.7)
    ax2.set_title('Confidence Distribution by Sentiment')
    ax2.set_xlabel('Confidence Score')
    ax2.set_ylabel('Count')
    
    plt.tight_layout()
    plt.savefig('sentiment_analysis_results.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    return df

def main():
    """
    Main function to run the text classification pipeline
    """
    print("Initializing Text Classifier...")
    classifier = TextClassifier()
    
    # Create sample data
    sample_texts = create_sample_data()
    
    print("Running sentiment analysis...")
    results = classifier.predict_sentiment(sample_texts)
    
    # Display results
    print("\nDetailed Results:")
    print("-" * 80)
    for result in results:
        print(f"Text: {result['text'][:50]}...")
        print(f"Sentiment: {result['label']} (Confidence: {result['confidence']:.3f})")
        print("-" * 80)
    
    # Analyze results
    df_results = analyze_results(results)
    
    # Save results
    df_results.to_csv('sentiment_results.csv', index=False)
    print("\nResults saved to 'sentiment_results.csv'")
    
    return df_results

if __name__ == "__main__":
    # Install required packages if not already installed
    # pip install transformers torch pandas scikit-learn matplotlib seaborn
    
    results_df = main()