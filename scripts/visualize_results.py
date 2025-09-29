#!/usr/bin/env python3
"""
Visualization script for analyzing training and evaluation results.
Uses results from existing training/evaluation scripts.
"""

import sys
from pathlib import Path
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, roc_curve
import warnings
warnings.filterwarnings('ignore')

# Set style
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

def load_results(results_dir="results"):
    """Load results from existing evaluation runs."""
    results_path = Path(results_dir)
    
    results = {}
    
    # Look for evaluation results
    for json_file in results_path.glob("*.json"):
        try:
            with open(json_file, 'r') as f:
                data = json.load(f)
                model_name = json_file.stem.replace('_', ' ').title()
                results[model_name] = data
                print(f"✅ Loaded {model_name} results")
        except Exception as e:
            print(f"⚠️ Could not load {json_file}: {e}")
    
    # Look for CSV predictions
    predictions = {}
    for csv_file in results_path.glob("*predictions*.csv"):
        try:
            df = pd.read_csv(csv_file)
            model_name = csv_file.stem.replace('_', ' ').title()
            predictions[model_name] = df
            print(f"✅ Loaded {model_name} predictions")
        except Exception as e:
            print(f"⚠️ Could not load {csv_file}: {e}")
    
    return results, predictions

def plot_performance_comparison(results):
    """Plot performance comparison from loaded results."""
    if not results:
        print("No results to plot")
        return
    
    # Extract metrics
    metrics_data = []
    model_names = []
    
    for model_name, data in results.items():
        if isinstance(data, dict):
            # Handle different possible metric formats
            metrics = {}
            if 'accuracy' in data:
                metrics['Accuracy'] = data['accuracy']
            if 'precision' in data:
                metrics['Precision'] = data['precision']
            if 'recall' in data:
                metrics['Recall'] = data['recall']
            if 'f1_score' in data:
                metrics['F1-Score'] = data['f1_score']
            elif 'f1' in data:
                metrics['F1-Score'] = data['f1']
            if 'roc_auc' in data:
                metrics['ROC AUC'] = data['roc_auc']
            
            if metrics:
                model_names.append(model_name)
                metrics_data.append(metrics)
    
    if not metrics_data:
        print("No valid metrics found in results")
        return
    
    # Create DataFrame
    df = pd.DataFrame(metrics_data, index=model_names)
    
    # Plot comparison
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    
    # Bar plot
    ax = axes[0]
    df.plot(kind='bar', ax=ax, width=0.8)
    ax.set_title('Model Performance Comparison', fontsize=14, fontweight='bold')
    ax.set_ylabel('Score')
    ax.set_xlabel('Models')
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    ax.grid(True, alpha=0.3)
    plt.setp(ax.get_xticklabels(), rotation=45, ha='right')
    
    # Radar plot
    ax = axes[1]
    if len(df.columns) >= 3:  # Need at least 3 metrics for radar plot
        angles = np.linspace(0, 2 * np.pi, len(df.columns), endpoint=False)
        angles = np.concatenate((angles, [angles[0]]))
        
        for model in model_names:
            values = df.loc[model].values
            values = np.concatenate((values, [values[0]]))
            ax.plot(angles, values, 'o-', linewidth=2, label=model)
            ax.fill(angles, values, alpha=0.25)
        
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(df.columns)
        ax.set_ylim(0, 1)
        ax.set_title('Performance Radar Chart', fontsize=14, fontweight='bold')
        ax.legend()
        ax.grid(True)
    else:
        ax.text(0.5, 0.5, 'Need at least 3 metrics\nfor radar plot', 
                ha='center', va='center', transform=ax.transAxes, fontsize=12)
        ax.set_title('Radar Plot (Insufficient Metrics)', fontsize=14)
    
    plt.tight_layout()
    plt.savefig('results/performance_comparison.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # Print summary table
    print("\n" + "="*60)
    print("PERFORMANCE SUMMARY")
    print("="*60)
    print(df.round(4).to_string())
    print("="*60)

def plot_predictions_analysis(predictions):
    """Analyze prediction distributions and confidence."""
    if not predictions:
        print("No predictions to analyze")
        return
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle('Prediction Analysis', fontsize=16, fontweight='bold')
    
    for i, (model_name, df) in enumerate(predictions.items()):
        if i >= 4:  # Limit to 4 models
            break
        
        ax = axes[i // 2, i % 2]
        
        # Check available columns
        if 'probability' in df.columns and 'true_label' in df.columns:
            # Prediction confidence by class
            lens_probs = df[df['true_label'] == 1]['probability']
            nonlens_probs = df[df['true_label'] == 0]['probability']
            
            ax.hist(nonlens_probs, bins=20, alpha=0.7, label='Non-Lens', color='blue')
            ax.hist(lens_probs, bins=20, alpha=0.7, label='Lens', color='red')
            ax.set_title(f'{model_name}\nPrediction Confidence')
            ax.set_xlabel('Predicted Probability')
            ax.set_ylabel('Count')
            ax.legend()
            ax.grid(True, alpha=0.3)
        
        elif 'predicted_label' in df.columns and 'true_label' in df.columns:
            # Confusion matrix
            cm = confusion_matrix(df['true_label'], df['predicted_label'])
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax,
                       xticklabels=['Non-Lens', 'Lens'],
                       yticklabels=['Non-Lens', 'Lens'])
            ax.set_title(f'{model_name}\nConfusion Matrix')
            ax.set_xlabel('Predicted')
            ax.set_ylabel('Actual')
        
        else:
            ax.text(0.5, 0.5, f'{model_name}\nInsufficient data\nfor analysis', 
                   ha='center', va='center', transform=ax.transAxes)
    
    # Hide unused subplots
    for i in range(len(predictions), 4):
        axes[i // 2, i % 2].set_visible(False)
    
    plt.tight_layout()
    plt.savefig('results/predictions_analysis.png', dpi=300, bbox_inches='tight')
    plt.show()

def plot_roc_comparison(predictions):
    """Plot ROC curves if we have probability predictions."""
    plt.figure(figsize=(10, 8))
    
    colors = plt.cm.Set1(np.linspace(0, 1, len(predictions)))
    
    roc_data = []
    
    for i, (model_name, df) in enumerate(predictions.items()):
        if 'probability' in df.columns and 'true_label' in df.columns:
            try:
                fpr, tpr, _ = roc_curve(df['true_label'], df['probability'])
                auc_score = np.trapz(tpr, fpr)  # Calculate AUC
                
                plt.plot(fpr, tpr, color=colors[i], linewidth=2, 
                        label=f'{model_name} (AUC = {auc_score:.3f})')
                
                roc_data.append({
                    'Model': model_name,
                    'AUC': auc_score
                })
                
            except Exception as e:
                print(f"Could not plot ROC for {model_name}: {e}")
    
    if roc_data:
        plt.plot([0, 1], [0, 1], 'k--', alpha=0.5, label='Random Classifier')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate', fontsize=12)
        plt.ylabel('True Positive Rate', fontsize=12)
        plt.title('ROC Curves Comparison', fontsize=14, fontweight='bold')
        plt.legend(loc="lower right")
        plt.grid(True, alpha=0.3)
        
        plt.savefig('results/roc_comparison.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        # Print ROC summary
        roc_df = pd.DataFrame(roc_data).sort_values('AUC', ascending=False)
        print("\n" + "="*40)
        print("ROC AUC RANKING")
        print("="*40)
        print(roc_df.to_string(index=False))
        print("="*40)
    else:
        print("No probability predictions found for ROC analysis")

def analyze_existing_results():
    """Main function to analyze existing results."""
    print("🎨 ANALYZING EXISTING RESULTS")
    print("=" * 50)
    
    # Load results
    results, predictions = load_results()
    
    if not results and not predictions:
        print("\n❌ No results found in 'results/' directory")
        print("\nTo generate results, run:")
        print("• python src/training/trainer.py --data-root data_realistic_test")
        print("• python src/evaluation/evaluator.py --data-root data_realistic_test")
        return
    
    print(f"\n📊 Found {len(results)} result files and {len(predictions)} prediction files")
    
    # Create visualizations
    if results:
        print("\n1. Creating performance comparison...")
        plot_performance_comparison(results)
    
    if predictions:
        print("\n2. Analyzing predictions...")
        plot_predictions_analysis(predictions)
        
        print("\n3. Creating ROC comparison...")
        plot_roc_comparison(predictions)
    
    print(f"\n✅ Analysis complete! Check 'results/' for generated plots.")

if __name__ == '__main__':
    analyze_existing_results()
