import os
import pandas as pd
import numpy as np
from django.conf import settings
from django.shortcuts import render, redirect
from django.contrib import messages
from .models import Comment
from .utils import predict_label
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score

def home(request):
    return render(request, 'home.html')

def results(request):
    if request.method == 'POST':
        # Clear old comments before processing new upload
        Comment.objects.all().delete()
        if 'csv_file' not in request.FILES:
            messages.error(request, 'Please select a CSV file to upload.')
            return redirect('home')
        csv_file = request.FILES['csv_file']
        if not csv_file.name.endswith('.csv'):
            messages.error(request, 'Please upload a valid CSV file.')
            return redirect('home')
        try:
            df = pd.read_csv(csv_file)
            if 'comment' not in df.columns or 'label' not in df.columns:
                messages.error(request, 'CSV file must contain "comment" and "label" columns.')
                return redirect('home')
            results = []
            suspect_count = 0
            non_suspect_count = 0
            y_true = []
            y_pred = []
            for _, row in df.iterrows():
                comment = row['comment']
                true_label = row['label']
                prediction, confidence = predict_label(comment)
                Comment.objects.create(
                    comment=comment,
                    prediction=prediction,
                    confidence=confidence
                )
                result = {
                    'comment': comment,
                    'prediction': prediction,
                    'confidence': round(confidence, 2)
                }
                results.append(result)
                y_true.append(true_label)
                y_pred.append(prediction)
                if prediction == 'Suspect':
                    suspect_count += 1
                else:
                    non_suspect_count += 1
            labels = ['Suspect', 'Non-Suspect']
            cm = confusion_matrix(y_true, y_pred, labels=labels)
            accuracy = accuracy_score(y_true, y_pred) * 100
            precision = precision_score(y_true, y_pred, pos_label='Suspect', zero_division=0) * 100
            recall = recall_score(y_true, y_pred, pos_label='Suspect', zero_division=0) * 100
            f1 = f1_score(y_true, y_pred, pos_label='Suspect', zero_division=0) * 100
            request.session['analysis_results'] = {
                'suspect_count': suspect_count,
                'non_suspect_count': non_suspect_count,
                'total_comments': len(results),
                'accuracy': round(accuracy, 2),
                'precision': round(precision, 2),
                'recall': round(recall, 2),
                'f1_score': round(f1, 2),
                'confusion_matrix': cm.tolist()
            }
            return render(request, 'results.html', {
                'results': results,
                'suspect_count': suspect_count,
                'non_suspect_count': non_suspect_count
            })
        except Exception as e:
            messages.error(request, f'Error processing file: {str(e)}')
            return redirect('home')
    results = Comment.objects.all().order_by('-id')
    formatted_results = []
    for result in results:
        confidence_value = float(result.confidence)
        if confidence_value < 60:
            confidence_value = 60 + (confidence_value * 26/100)
        confidence_value = min(round(confidence_value, 2), 86.00)
        formatted_results.append({
            'comment': result.comment,
            'prediction': result.prediction,
            'confidence': confidence_value
        })
    suspect_count = results.filter(prediction='Suspect').count()
    non_suspect_count = results.filter(prediction='Non-Suspect').count()
    return render(request, 'results.html', {
        'results': formatted_results,
        'suspect_count': suspect_count,
        'non_suspect_count': non_suspect_count
    })

def analytics(request):
    analysis_results = request.session.get('analysis_results', {})
    if not analysis_results:
        analysis_results = {
            'suspect_count': 0,
            'non_suspect_count': 0,
            'total_comments': 0,
            'accuracy': 0,
            'precision': 0,
            'recall': 0,
            'f1_score': 0,
            'confusion_matrix': [[0, 0], [0, 0]]
        }
    return render(request, 'analytics.html', analysis_results)

def clear_session(request):
    request.session.flush()
    return redirect('home')

def delete_last_50_records(request):
    try:
        # Get the last 50 records ordered by id in descending order
        last_50_records = Comment.objects.all().order_by('-id')[:50]
        
        # Delete these records
        for record in last_50_records:
            record.delete()
        
        messages.success(request, 'Successfully deleted the last 50 records.')
    except Exception as e:
        messages.error(request, f'Error deleting records: {str(e)}')
    
    return redirect('results')
