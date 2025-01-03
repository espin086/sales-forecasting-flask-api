# Sales Forecasting API

A production-ready Flask REST API that serves predictions from a LightGBM sales forecasting model. The API is containerized using Docker for easy deployment and scalability.

[![Watch Demo on YouTube](https://img.youtube.com/vi/WkoF2_-SaKg/0.jpg)](https://youtu.be/WkoF2_-SaKg)

## Features

- RESTful API endpoints for sales predictions
- Asynchronous job processing with status tracking
- LightGBM-based forecasting model
- Docker containerization
- Comprehensive input validation
- Detailed error handling
- Job queuing system
- Filtering and pagination for job listings

## Prerequisites

- Python 3.11+
- Docker (for containerized deployment)
- Kaggle account (for dataset access)

## Quick Start

### Local Development

1. **Clone the repository**
   ```bash
   git clone <repository-url>
   cd sales-forecast-api
   ```

2. **Create and activate virtual environment**
   ```bash
   python -m venv venv
   
   # Windows
   .\venv\Scripts\activate
   
   # Unix/MacOS
   source venv/bin/activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Prepare the dataset**
   - Download the training dataset from Kaggle
   - Create a `data` directory in the project root
   - Place the dataset as `data/train.csv`

   ```bash
   python model/preprocess_data.py
   ```

5. **Train the model**
   ```bash
   python model/train.py
   ```

You should see model results like these: 

```bash
Loading data...
Performing feature engineering...
Splitting data...
Creating LightGBM datasets...
Training model...
Training until validation scores don't improve for 50 rounds
[100]   training's mape: 0.190149       valid_1's mape: 0.190357
[200]   training's mape: 0.154417       valid_1's mape: 0.154406
[300]   training's mape: 0.147229       valid_1's mape: 0.147187
[400]   training's mape: 0.141365       valid_1's mape: 0.141314
[500]   training's mape: 0.137691       valid_1's mape: 0.137825
[600]   training's mape: 0.135692       valid_1's mape: 0.135984
[700]   training's mape: 0.134626       valid_1's mape: 0.135063
[800]   training's mape: 0.133908       valid_1's mape: 0.134453
[900]   training's mape: 0.133305       valid_1's mape: 0.133944
[1000]  training's mape: 0.132699       valid_1's mape: 0.133432
Did not meet early stopping. Best iteration is:
[997]   training's mape: 0.132702       valid_1's mape: 0.133428
Saving model and features...
Training completed!

```


6. **Start the API**
   ```bash
   python app/api.py
   ```

### Docker Deployment

1. **Build the container**
   ```bash
   docker build -t sales-forecast-api .
   ```

2. **Run the container**
   ```bash
   docker run -p 5000:5000 sales-forecast-api
   ```

## API Documentation

### Endpoints

#### 1. Check API Status
- **URL**: `/status`
- **Method**: `GET`
- **Response Example**:
  ```json
  {
      "status": "online",
      "model_loaded": true,
      "active_jobs": 3,
      "jobs_by_status": {
          "pending": 1,
          "processing": 0,
          "completed": 2,
          "failed": 0
      }
  }
  ```

#### 2. Submit Prediction Job
- **URL**: `/predict`
- **Method**: `POST`
- **Request Body**:
  ```json
  {
      "date": "2023-01-01",
      "store": 1,
      "item": 1
  }
  ```
- **Response Example**:
  ```json
  {
      "job_id": "550e8400-e29b-41d4-a716-446655440000",
      "status": "pending",
      "submitted_at": "2024-03-14T12:00:00"
  }
  ```

#### 3. Check Job Status
- **URL**: `/status/<job_id>`
- **Method**: `GET`
- **Response Example (Completed)**:
  ```json
  {
      "job_id": "550e8400-e29b-41d4-a716-446655440000",
      "status": "completed",
      "submitted_at": "2024-03-14T12:00:00",
      "completed_at": "2024-03-14T12:00:05",
      "result": {
          "predicted_sales": 45.67,
          "date": "2023-01-01",
          "store": 1,
          "item": 1
      }
  }
  ```

#### 4. List Jobs
- **URL**: `/jobs`
- **Method**: `GET`
- **Query Parameters**:
  - `status`: Filter by job status (pending, processing, completed, failed)
  - `limit`: Maximum number of jobs to return
- **Response Example**:
  ```json
  {
      "total_jobs": 100,
      "filtered_jobs": 2,
      "jobs": [
          {
              "job_id": "550e8400-e29b-41d4-a716-446655440000",
              "status": "completed",
              "result": {
                  "predicted_sales": 45.67,
                  "date": "2023-01-01",
                  "store": 1,
                  "item": 1
              }
          }
      ]
  }
  ```

### Error Handling

The API returns appropriate HTTP status codes and error messages:

- `400 Bad Request`: 
  - Missing required fields
  - Invalid date format
  - Invalid numeric values
  - Invalid JSON
- `404 Not Found`: Job ID not found
- `500 Internal Server Error`: Server-side errors

## Testing

### Running Tests
```bash
# Run all tests
python tests/run_tests.py

# Run specific test file
python -m unittest tests/test_api.py
```

### Test Coverage
The tests cover:
- API status endpoint
- Prediction workflow
- Job status checking
- Input validation
- Error handling
- Jobs listing with filters

## Project Structure
```
sales-forecast-api/
├── app/
│   ├── __init__.py
│   └── api.py          # Flask API implementation
├── model/
│   ├── __init__.py
│   └── train.py        # Model training script
├── tests/
│   ├── __init__.py
│   ├── test_api.py     # API tests
│   └── run_tests.py    # Test runner
├── data/               # Dataset directory (not in repo)
├── Dockerfile
├── requirements.txt
└── README.md
```

## Contributing

1. Fork the repository
2. Create a feature branch
3. Commit your changes
4. Push to the branch
5. Create a Pull Request

## License

[Add your chosen license here]

## Technical Details and Design Decisions

### Architecture Overview

The solution implements a scalable, asynchronous API for sales forecasting with the following key components:

1. **Asynchronous Job Processing**
   - Jobs are processed in background threads
   - Each prediction request gets a unique job ID
   - Clients can poll job status using the ID
   - Prevents timeout issues with long-running predictions

2. **Model Deployment**
   - Pre-trained LightGBM model
   - Model loaded at startup
   - Serialized using joblib for efficiency
   - Features list stored separately for validation

3. **Input Validation**
   - Comprehensive validation for all inputs
   - Type conversion and sanitization
   - Clear error messages
   - Proper HTTP status codes

### Design Decisions

1. **Job Queue System**
   - Why: Handle multiple concurrent requests without blocking
   - Implementation: Python's Queue and Thread
   - Trade-offs:
     - ✅ Simple implementation
     - ✅ No external dependencies
     - ❌ Not persistent across restarts
     - ❌ Limited to single instance

2. **Docker Configuration**
   - Why: Reproducible deployments
   - Decisions:
     - Use pre-trained model instead of training in container
     - Multi-stage build for smaller image
     - Slim base image for reduced size
   - Trade-offs:
     - ✅ Faster container startup
     - ✅ Smaller image size
     - ❌ Need to manage model versions separately

3. **Error Handling**
   - Structured error responses
   - Detailed validation messages
   - Proper HTTP status codes
   - Job status tracking
   - Trade-offs:
     - ✅ Better debugging
     - ✅ Clear client feedback
     - ❌ More complex code
     - ❌ Slightly larger responses

### Scalability Considerations

1. **Current Limitations**
   - In-memory job queue
   - Single worker thread
   - No persistence
   - Local model loading

2. **Potential Improvements**
   - Redis for job queue
   - Multiple worker processes
   - Database for job history
   - Model versioning
   - Kubernetes deployment

### Security Considerations

1. **Current Implementation**
   - Input validation
   - Error handling
   - No sensitive data exposure

2. **Recommended Additions**
   - Authentication
   - Rate limiting
   - HTTPS
   - Input sanitization
   - Logging
   - Monitoring

### Testing Strategy

1. **Current Coverage**
   - API endpoint testing
   - Input validation
   - Error scenarios
   - Job workflow

2. **Future Improvements**
   - Integration tests
   - Load testing
   - Security testing
   - Model accuracy monitoring

### Trade-offs and Decisions

1. **Simplicity vs Features**
   - Chose simple job queue over complex message broker
   - Used thread over multiple processes
   - In-memory storage over database
   - Reason: Easier to understand, deploy, and maintain

2. **Performance vs Complexity**
   - Single worker thread
   - Synchronous model loading
   - In-memory job storage
   - Reason: Adequate for moderate loads, simpler implementation

3. **Development vs Production**
   - Pre-trained model in container
   - Environment variables
   - Configurable logging
   - Reason: Balance between development ease and production readiness

### Future Improvements

1. **Scalability**
   - Implement Redis for job queue
   - Add database for job persistence
   - Deploy with Kubernetes
   - Add load balancing

2. **Monitoring**
   - Add metrics collection
   - Implement health checks
   - Set up alerting
   - Add performance monitoring

3. **Security**
   - Add authentication
   - Implement rate limiting
   - Set up HTTPS
   - Add audit logging

4. **Features**
   - Batch predictions
   - Model versioning
   - Result caching
   - Real-time model updates

### Maintenance Considerations

1. **Model Updates**
   - Process for updating model
   - Version control for models
   - Rollback procedures
   - Accuracy monitoring

2. **Dependencies**
   - Regular updates
   - Security patches
   - Compatibility testing
   - Dependency scanning

3. **Monitoring**
   - Error rates
   - Response times
   - Queue length
   - Resource usage
