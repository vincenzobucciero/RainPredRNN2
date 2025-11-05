from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image
from reportlab.lib import colors
from reportlab.platypus import Table, TableStyle
import os
import datetime

def create_report(output_dir, report_path):
    # Create the document
    doc = SimpleDocTemplate(
        report_path,
        pagesize=A4,
        rightMargin=72,
        leftMargin=72,
        topMargin=72,
        bottomMargin=72
    )

    # Styles
    styles = getSampleStyleSheet()
    title_style = styles['Heading1']
    subtitle_style = styles['Heading2']
    text_style = styles['Normal']
    text_style.leading = 14  # Increase line spacing

    # Content
    content = []

    # Title
    content.append(Paragraph("RainPredRNN Model Evaluation Report", title_style))
    content.append(Spacer(1, 20))
    
    # Date
    date = datetime.datetime.now().strftime("%d %B %Y")
    content.append(Paragraph(f"Report generated on: {date}", text_style))
    content.append(Spacer(1, 20))

    # Introduction
    content.append(Paragraph("Model Overview", subtitle_style))
    content.append(Paragraph(
        "This report presents the evaluation results of the RainPredRNN model, "
        "a deep learning architecture designed for precipitation nowcasting. "
        "The model was trained to predict rainfall patterns based on radar reflectivity data.",
        text_style
    ))
    content.append(Spacer(1, 20))

    # 1. Confusion Matrix
    content.append(Paragraph("1. Confusion Matrix Analysis", subtitle_style))
    content.append(Spacer(1, 10))
    
    # Add the confusion matrix image
    img_path = os.path.join(output_dir, "confusion_matrix.png")
    img = Image(img_path, width=400, height=320)
    content.append(img)
    content.append(Spacer(1, 10))
    
    content.append(Paragraph(
        "The confusion matrix provides a detailed breakdown of the model's classification "
        "performance for rain prediction. The matrix shows four key metrics:",
        text_style
    ))
    content.append(Spacer(1, 10))
    
    content.append(Paragraph(
        "• True Negatives (TN): Correctly identified non-rain events\n"
        "• False Positives (FP): Incorrectly predicted rain when there was none\n"
        "• False Negatives (FN): Missed actual rain events\n"
        "• True Positives (TP): Correctly identified rain events\n",
        text_style
    ))
    content.append(Spacer(1, 10))
    
    content.append(Paragraph(
        "The percentages in each cell represent the proportion of predictions within each "
        "true class, providing insight into the model's classification behavior for both "
        "rain and non-rain conditions.",
        text_style
    ))
    content.append(Spacer(1, 20))

    # 2. Performance Metrics
    content.append(Paragraph("2. Performance Metrics Analysis", subtitle_style))
    content.append(Spacer(1, 10))
    
    img_path = os.path.join(output_dir, "performance_metrics.png")
    img = Image(img_path, width=400, height=240)
    content.append(img)
    content.append(Spacer(1, 10))
    
    content.append(Paragraph(
        "The performance metrics graph displays four critical evaluation measures:",
        text_style
    ))
    content.append(Spacer(1, 10))
    
    content.append(Paragraph(
        "• Precision: The proportion of correct positive predictions among all positive predictions\n"
        "• Recall: The proportion of actual positive cases that were correctly identified\n"
        "• F1 Score: The harmonic mean of precision and recall\n"
        "• Accuracy: The overall proportion of correct predictions\n",
        text_style
    ))
    content.append(Spacer(1, 20))

    # 3. Training Metrics
    content.append(Paragraph("3. Training Metrics Analysis", subtitle_style))
    content.append(Spacer(1, 10))
    
    img_path = os.path.join(output_dir, "training_metrics.png")
    img = Image(img_path, width=400, height=240)
    content.append(img)
    content.append(Spacer(1, 10))
    
    content.append(Paragraph(
        "The training metrics graph shows various performance indicators:",
        text_style
    ))
    content.append(Spacer(1, 10))
    
    content.append(Paragraph(
        "• MAE (Mean Absolute Error): Measures the average magnitude of errors\n"
        "• SSIM (Structural Similarity Index): Evaluates the structural similarity between predictions and ground truth\n"
        "• CSI (Critical Success Index): Measures the accuracy of precipitation forecasts\n"
        "• SmoothL1: A robust loss function that combines L1 and L2 losses\n"
        "• FL (Focal Loss): Addresses class imbalance in the dataset\n",
        text_style
    ))
    content.append(Spacer(1, 20))

    # Conclusions
    content.append(Paragraph("Conclusions", subtitle_style))
    content.append(Paragraph(
        "The model demonstrates strong overall accuracy but shows some challenges in "
        "detecting rain events, which is common in precipitation forecasting due to the "
        "inherent class imbalance in weather data. The high SSIM score indicates good "
        "structural similarity in the predictions, while the CSI suggests room for "
        "improvement in precise rain event detection. Future improvements could focus on "
        "enhancing the model's sensitivity to rain events while maintaining its current "
        "high specificity for non-rain conditions.",
        text_style
    ))

    # Build the PDF
    doc.build(content)

if __name__ == "__main__":
    output_dir = "checkpoints/evaluation_reports"
    report_path = os.path.join(output_dir, "evaluation_report.pdf")
    create_report(output_dir, report_path)
    print(f"Report generated successfully: {report_path}")