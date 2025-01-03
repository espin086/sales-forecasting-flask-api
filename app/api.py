from flask import Flask, request, jsonify
import lightgbm as lgb
import pandas as pd
import numpy as np
from datetime import datetime
import joblib
import uuid
from threading import Thread
from queue import Queue
import time
from enum import Enum
import os

app = Flask(__name__)

class JobStatus(Enum):
    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"

# Load the trained model
try:
    model_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'model')
    model = joblib.load(os.path.join(model_dir, 'sales_forecast_model.pkl'))
    features = joblib.load(os.path.join(model_dir, 'feature_list.pkl'))
except Exception as e:
    app.logger.error(f"Failed to load model: {str(e)}")
    model = None
    features = None

# Job storage
jobs = {}
job_queue = Queue()

def validate_date(date_str):
    """Validate date string format"""
    try:
        datetime.strptime(date_str, "%Y-%m-%d")
        return True
    except ValueError:
        return False

def process_jobs():
    """Background worker to process prediction jobs"""
    while True:
        if not job_queue.empty():
            job_id = job_queue.get()
            if job_id in jobs:
                try:
                    # Update status to processing
                    jobs[job_id]['status'] = JobStatus.PROCESSING.value
                    
                    # Get job data
                    data = jobs[job_id]['data']
                    
                    # Process the prediction
                    input_data = pd.DataFrame([data])
                    input_data['date'] = pd.to_datetime(input_data['date'])
                    
                    # Add engineered features
                    input_data['year'] = input_data['date'].dt.year
                    input_data['month'] = input_data['date'].dt.month
                    input_data['day'] = input_data['date'].dt.day
                    input_data['dayofweek'] = input_data['date'].dt.dayofweek
                    input_data = engineer_features(input_data)
                    
                    # Make prediction
                    prediction = model.predict(input_data[features])
                    
                    # Update job status
                    jobs[job_id].update({
                        'status': JobStatus.COMPLETED.value,
                        'completed_at': datetime.now().isoformat(),
                        'result': {
                            'predicted_sales': float(prediction[0]),
                            'date': data['date'],
                            'store': data['store'],
                            'item': data['item']
                        }
                    })
                except Exception as e:
                    jobs[job_id].update({
                        'status': JobStatus.FAILED.value,
                        'completed_at': datetime.now().isoformat(),
                        'error': str(e)
                    })
        time.sleep(0.1)  # Prevent CPU overload

# Start background worker
worker = Thread(target=process_jobs, daemon=True)
worker.start()

def engineer_features(input_data):
    """Add engineered features to input data"""
    input_data['is_weekend'] = input_data['dayofweek'].isin([5, 6]).astype(int)
    input_data['is_month_start'] = input_data['date'].dt.is_month_start.astype(int)
    input_data['is_month_end'] = input_data['date'].dt.is_month_end.astype(int)
    return input_data

@app.route('/status', methods=['GET'])
def status():
    """Check API status"""
    return jsonify({
        'status': 'online',
        'model_loaded': model is not None,
        'active_jobs': len(jobs),
        'jobs_by_status': {
            status.value: len([j for j in jobs.values() if j['status'] == status.value])
            for status in JobStatus
        }
    })

@app.route('/status/<job_id>', methods=['GET'])
def job_status(job_id):
    """Get detailed status of a specific job"""
    if job_id not in jobs:
        return jsonify({
            'error': 'Job not found',
            'job_id': job_id
        }), 404
    
    job = jobs[job_id]
    response = {
        'job_id': job_id,
        'status': job['status'],
        'submitted_at': job['submitted_at']
    }
    
    # Add completion time if job is finished
    if 'completed_at' in job:
        response['completed_at'] = job['completed_at']
    
    # Add result or error based on status
    if job['status'] == JobStatus.COMPLETED.value:
        response['result'] = job['result']
    elif job['status'] == JobStatus.FAILED.value:
        response['error'] = job['error']
    
    return jsonify(response)

@app.route('/predict', methods=['POST'])
def predict():
    """Submit a prediction job"""
    try:
        # Check if data is provided and is valid JSON
        data = request.get_json()
        if data is None:
            return jsonify({'error': 'No data provided or invalid JSON'}), 400
        
        # Validate required fields
        required_fields = ['date', 'store', 'item']
        missing_fields = [field for field in required_fields if field not in data]
        if missing_fields:
            return jsonify({'error': f'Missing required fields: {", ".join(missing_fields)}'}), 400
        
        # Validate and convert date
        try:
            # Ensure date is string
            date_str = str(data.get('date', ''))
            datetime.strptime(date_str, "%Y-%m-%d")
            data['date'] = date_str  # Store validated date
        except ValueError:
            return jsonify({'error': 'Invalid date format. Use YYYY-MM-DD'}), 400
        
        # Validate and convert store
        try:
            store = int(float(str(data.get('store', 0))))  # Handle float inputs
            if store <= 0:
                return jsonify({'error': 'Store must be a positive integer'}), 400
            data['store'] = store  # Store validated store
        except (ValueError, TypeError):
            return jsonify({'error': 'Store must be a positive integer'}), 400
        
        # Validate and convert item
        try:
            item = int(float(str(data.get('item', 0))))  # Handle float inputs
            if item <= 0:
                return jsonify({'error': 'Item must be a positive integer'}), 400
            data['item'] = item  # Store validated item
        except (ValueError, TypeError):
            return jsonify({'error': 'Item must be a positive integer'}), 400
        
        # Generate unique job ID
        job_id = str(uuid.uuid4())
        
        # Create job with validated data
        jobs[job_id] = {
            'status': JobStatus.PENDING.value,
            'data': {
                'date': data['date'],
                'store': store,
                'item': item
            },
            'submitted_at': datetime.now().isoformat()
        }
        
        # Add to processing queue
        job_queue.put(job_id)
        
        return jsonify({
            'job_id': job_id,
            'status': JobStatus.PENDING.value,
            'submitted_at': jobs[job_id]['submitted_at']
        })
        
    except Exception as e:
        app.logger.error(f"Prediction error: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/jobs', methods=['GET'])
def list_jobs():
    """List all jobs"""
    # Add query parameters for filtering
    status_filter = request.args.get('status')
    limit = request.args.get('limit', type=int)
    
    filtered_jobs = jobs
    if status_filter:
        filtered_jobs = {
            k: v for k, v in jobs.items() 
            if v['status'] == status_filter
        }
    
    job_list = [
        {'job_id': k, **v} 
        for k, v in filtered_jobs.items()
    ]
    
    if limit:
        job_list = job_list[:limit]
    
    return jsonify({
        'total_jobs': len(jobs),
        'filtered_jobs': len(job_list),
        'jobs': job_list
    })

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5001)