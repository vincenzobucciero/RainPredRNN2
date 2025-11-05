import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os
import re

def parse_report(report_path):
    with open(report_path, 'r') as f:
        content = f.read()
    
    # Parse basic metrics
    metrics_dict = {}
    metrics_section = content.split('Metrics:\n')[1].split('\n\nConfusion Matrix:')[0]
    for line in metrics_section.split('\n'):
        if ':' in line:
            key, value = line.split(':')
            metrics_dict[key.strip()] = float(value.strip())
    
    # Parse confusion matrix
    matrix_section = content.split('Format: [[TN, FP],\n        [FN, TP]]\n\n')[1].split('\n\n')[0]
    # Convert string representation to numpy array
    matrix_str = matrix_section.replace('[', '').replace(']', '')
    matrix_values = [int(x) for x in re.findall(r'\d+', matrix_str)]
    conf_matrix = np.array(matrix_values).reshape(2, 2)
    
    # Parse additional metrics
    additional_metrics = {}
    additional_section = content.split('Additional Metrics:\n')[1]
    for line in additional_section.split('\n'):
        if ':' in line:
            key, value = line.split(':')
            additional_metrics[key.strip()] = float(value.strip())
    
    return metrics_dict, conf_matrix, additional_metrics

def generate_plots(metrics_dict, conf_matrix, additional_metrics, output_dir):
    # Imposta il tema generale
    plt.style.use('default')
    color_palette = ['#FF9999', '#66B2FF', '#99FF99', '#FFCC99', '#FF99CC', '#99FFFF']
    
    # 1. Confusion Matrix Heatmap
    plt.figure(figsize=(15, 12))
    
    # Calcola le percentuali per ogni classe
    row_sums = conf_matrix.sum(axis=1, keepdims=True)
    norm_conf_mx = conf_matrix / row_sums
    
    # Crea la heatmap con un design più elegante
    ax = sns.heatmap(norm_conf_mx, 
                     annot=False,  # Non mostrare i numeri automaticamente
                     cmap='YlOrRd',
                     xticklabels=['No Rain', 'Rain'],
                     yticklabels=['No Rain', 'Rain'],
                     square=True,
                     cbar_kws={'label': 'Normalized Ratio'})
    
    # Aggiungi i valori nelle celle con formattazione migliorata
    for i in range(2):
        for j in range(2):
            percentage = norm_conf_mx[i, j] * 100
            value = conf_matrix[i, j]
            # Formatta i numeri grandi con separatore delle migliaia
            value_str = f'{value:,}'
            # Crea il testo su due righe con spaziatura
            text = f'{value_str}\n\n({percentage:.1f}%)'
            ax.text(j + 0.5, i + 0.5, text,
                   ha='center', va='center',
                   fontsize=12,
                   color='black' if percentage < 50 else 'white',
                   fontweight='bold')

    # Personalizza il titolo e le etichette
    plt.title('Confusion Matrix', pad=20, fontsize=16, fontweight='bold')
    plt.ylabel('True Label', labelpad=15, fontsize=12)
    plt.xlabel('Predicted Label', labelpad=15, fontsize=12)
    
    # Migliora l'aspetto dei tick labels
    ax.tick_params(labelsize=12)
    
    # Aggiusta il layout con più spazio
    plt.tight_layout()
    
    # Salva con alta qualità
    plt.savefig(os.path.join(output_dir, 'confusion_matrix.png'), 
                dpi=300, 
                bbox_inches='tight',
                facecolor='white',
                edgecolor='none')
    plt.close()
    
    # 2. Performance Metrics Bar Plot
    plt.figure(figsize=(12, 8))
    metrics = ['Precision', 'Recall', 'F1 Score', 'Accuracy']
    values = [additional_metrics[m] for m in metrics]
    
    # Crea le barre con colori dal palette
    bars = plt.bar(metrics, values, color=color_palette)
    
    # Personalizza l'aspetto
    plt.title('Performance Metrics', pad=20, fontsize=16, fontweight='bold')
    plt.ylabel('Score', fontsize=12, labelpad=10)
    plt.ylim(0, 1.1)  # Aumenta il limite per i valori
    plt.grid(True, axis='y', linestyle='--', alpha=0.7)
    
    # Aggiungi i valori sopra le barre
    for bar, value in zip(bars, values):
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height + 0.02,
                f'{value:.4f}',
                ha='center', va='bottom',
                fontsize=10, fontweight='bold')
    
    # Migliora l'aspetto dei tick labels
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=10)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'performance_metrics.png'),
                dpi=300,
                bbox_inches='tight',
                facecolor='white')
    plt.close()
    
    # 3. Training Metrics Plot
    plt.figure(figsize=(12, 8))
    
    # Crea le barre con colori dal palette
    bars = plt.bar(metrics_dict.keys(), metrics_dict.values(), 
                  color=color_palette[:len(metrics_dict)])
    
    # Personalizza l'aspetto
    plt.title('Training Metrics', pad=20, fontsize=16, fontweight='bold')
    plt.ylabel('Value', fontsize=12, labelpad=10)
    plt.grid(True, axis='y', linestyle='--', alpha=0.7)
    
    # Ruota e allinea le etichette
    plt.xticks(rotation=45, ha='right', fontsize=12)
    plt.yticks(fontsize=10)
    
    # Aggiungi i valori sopra le barre
    for bar, (_, value) in zip(bars, metrics_dict.items()):
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height + height*0.02,
                f'{value:.4f}',
                ha='center', va='bottom',
                fontsize=10, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'training_metrics.png'),
                dpi=300,
                bbox_inches='tight',
                facecolor='white')
    plt.close()

if __name__ == "__main__":
    # Paths
    report_path = "/home/vbucciero/projects/RainPredRNN2/source/checkpoints/evaluation_reports/evaluation_report.txt"
    output_dir = "/home/vbucciero/projects/RainPredRNN2/source/checkpoints/evaluation_reports"
    
    # Read and parse the report
    metrics_dict, conf_matrix, additional_metrics = parse_report(report_path)
    
    # Generate plots
    generate_plots(metrics_dict, conf_matrix, additional_metrics, output_dir)
    print("Plots generated successfully in:", output_dir)