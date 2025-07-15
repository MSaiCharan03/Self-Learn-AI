# Self-Learning LLM Platform

A scalable, modular, and self-improving Large Language Model (LLM) platform featuring local Phi-2 inference, vector search with Faiss, PostgreSQL storage, and vanilla web interface. The system learns from user feedback and external model comparisons to continuously improve its performance.

## 🚀 Features

### Core Capabilities
- **Local Phi-2 LLM**: CPU-optimized inference with no cloud dependencies
- **Vector Search**: Faiss-powered semantic search for RAG functionality
- **Self-Learning**: Continuous improvement through user feedback and fine-tuning
- **Model Comparison**: Side-by-side evaluation with external models (GPT, Claude, Gemini)
- **Vanilla Web UI**: Clean, responsive interface without frameworks

### Technical Highlights
- **CPU-Only Operation**: Runs entirely on CPU with 24GB RAM
- **Real-time Chat**: WebSocket-style updates with conversation history
- **Feedback System**: Thumbs up/down, ratings, and comments
- **Fine-tuning Pipeline**: LoRA-based parameter-efficient training
- **RAG Integration**: Context-aware responses using similar conversations
- **API-First**: RESTful backend with comprehensive documentation

## 📋 Requirements

### System Requirements
- **CPU**: Multi-core processor (Intel/AMD)
- **RAM**: 24GB minimum (32GB recommended)
- **Storage**: 200GB NVMe SSD
- **OS**: Linux (Ubuntu 20.04+ recommended)
- **Network**: Internet connection for initial setup

### Software Dependencies
- **Python**: 3.8 or higher
- **PostgreSQL**: 12 or higher
- **Node.js**: 16+ (for development tools, optional)
- **Git**: For version control

## 🛠️ Installation

### 1. Clone the Repository

```bash
git clone https://github.com/MSaiCharan03/self-learning-llm-platform.git
cd self-learning-llm-platform
```

### 2. Set Up Python Environment

```bash
# Create virtual environment
python -m venv .venv

# Activate virtual environment
source .venv/bin/activate  # Linux/Mac
# or
venv\Scripts\activate     # Windows

# Install dependencies
pip install -r requirements.txt
```

### 3. Database Setup

#### Option A: Using Docker (Recommended)
```bash
# Start PostgreSQL with Docker
docker-compose up -d postgres

# The database will be available at localhost:9001
# Username: postgres, Password: password, Database: llm_platform
```

#### Option B: Local PostgreSQL Installation
```bash
# Ubuntu/Debian
sudo apt update
sudo apt install postgresql postgresql-contrib

# Start PostgreSQL service
sudo systemctl start postgresql
sudo systemctl enable postgresql

# Create database and user
sudo -u postgres psql
CREATE DATABASE llm_platform;
CREATE USER llm_user WITH PASSWORD 'your_password';
GRANT ALL PRIVILEGES ON DATABASE llm_platform TO llm_user;
\q
```

### 4. Initialize Database Schema

```bash
# Navigate to backend directory
cd backend

# Run database initialization
python scripts/init_db.py

# Or manually run the SQL schema (if using Docker, use port 9001)
psql -U postgres -h localhost -p 9001 -d llm_platform -f scripts/init_db.sql
```

### 5. Configuration

Create a `.env` file in the project root:

```bash
cp .env.example .env
```

Edit the `.env` file with your settings:

```env
# Database Configuration (use 9001 for Docker)
DATABASE_URL=postgresql://postgres:password@localhost:9001/llm_platform

# Security (Generate secure keys for production)
SECRET_KEY=your-super-secret-key-change-this-in-production
ALGORITHM=HS256
ACCESS_TOKEN_EXPIRE_MINUTES=30

# Model Configuration
PHI2_MODEL_PATH=./data/models/phi-2
EMBEDDING_MODEL_NAME=all-MiniLM-L6-v2

# Faiss Configuration
FAISS_INDEX_PATH=./data/faiss_index
EMBEDDING_DIMENSIONS=384

# External API Keys (Optional)
OPENAI_API_KEY=your-openai-api-key
ANTHROPIC_API_KEY=your-anthropic-api-key
GOOGLE_API_KEY=your-google-api-key

# Server Configuration
HOST=0.0.0.0
PORT=9090
DEBUG=false

# CORS Configuration
CORS_ORIGINS=["http://localhost:9090", "http://127.0.0.1:9090", "http://localhost:9000", "http://127.0.0.1:9000"]

# Logging
LOG_LEVEL=INFO
LOG_FILE=./data/logs/app.log
```

### 6. Download and Setup Phi-2 Model

```bash
# Download and cache Phi-2 model locally
python backend/scripts/setup_model.py

# This will download ~5GB and may take 10-30 minutes
# The model will be cached in ./data/models/phi-2/
```

### 7. Initialize Vector Store

```bash
# Initialize Faiss index and embedding model
python backend/scripts/init_embeddings.py

# This will set up the embedding model and create the vector index
```

### 8. Test the Installation

```bash
# Test Phi-2 model
python backend/scripts/test_phi2.py

# Test external models (if API keys configured)
python backend/scripts/test_external_models.py

# Test vector search
python backend/scripts/test_vector_search.py
```

## 🏃 Running the Application

### Start the Backend Server

```bash
# From the project root directory
cd backend
python -m uvicorn main:app --host 0.0.0.0 --port 9090 --reload
```

### Access the Web Interface

Open your browser and navigate to:
- **Main Interface**: http://localhost:9090
- **API Documentation**: http://localhost:9090/docs
- **Alternative API Docs**: http://localhost:9090/redoc

### First-Time Setup

1. **Register an Account**: Create your first user account
2. **Start Chatting**: Begin conversations with the AI
3. **Provide Feedback**: Rate responses to improve the model
4. **Explore Features**: Try different chat modes and settings

## 📖 Usage Guide

### Basic Chat Interface

The main interface provides a ChatGPT-style experience:

- **Send Messages**: Type in the input box and press Enter
- **View History**: Access previous conversations from the sidebar
- **Rate Responses**: Use thumbs up/down buttons to provide feedback
- **Add Comments**: Click the comment button to provide detailed feedback

### Advanced Features

#### Model Comparison
```bash
# Enable comparison mode in chat
POST /api/v1/chat/send
{
  "content": "Your question here",
  "generate_comparisons": true,
  "comparison_models": ["gpt-3.5-turbo", "claude-3-haiku"]
}
```

#### Manual Training
```bash
# Generate training data
python backend/scripts/simulate_feedback.py simulate --conversations 50

# Run fine-tuning
python backend/scripts/run_training.py train --epochs 3 --lr 5e-5

# View training history
python backend/scripts/run_training.py list
```

#### API Usage
```python
import requests

# Send a chat message
response = requests.post("http://localhost:9090/api/v1/chat/send", 
    json={"content": "Hello, how are you?"},
    headers={"Authorization": "Bearer your-jwt-token"}
)

# Get model information
models = requests.get("http://localhost:9090/api/v1/models/available")
```

## 🔧 Configuration

### Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `DATABASE_URL` | PostgreSQL connection string | `postgresql://postgres:password@localhost:5432/llm_platform` |
| `SECRET_KEY` | JWT signing key | `your-secret-key-change-this-in-production` |
| `PHI2_MODEL_PATH` | Local model storage path | `./data/models/phi-2` |
| `FAISS_INDEX_PATH` | Vector index storage path | `./data/faiss_index` |
| `OPENAI_API_KEY` | OpenAI API key (optional) | `None` |
| `ANTHROPIC_API_KEY` | Anthropic API key (optional) | `None` |
| `GOOGLE_API_KEY` | Google API key (optional) | `None` |

### Model Configuration

Edit `backend/utils/config.py` to adjust model parameters:

```python
# Phi-2 generation parameters
default_generation_config = {
    "max_new_tokens": 512,
    "temperature": 0.7,
    "top_p": 0.9,
    "top_k": 40,
    "do_sample": True,
    "repetition_penalty": 1.1,
}

# Training parameters
training_config = {
    "num_epochs": 3,
    "learning_rate": 5e-5,
    "batch_size": 1,
    "lora_rank": 8,
    "lora_alpha": 32,
}
```

## 🧪 Development

### Running Tests

```bash
# Run all tests
pytest backend/tests/

# Run specific test categories
pytest backend/tests/test_models.py
pytest backend/tests/test_api.py
pytest backend/tests/test_vector_store.py
```

### Code Quality

```bash
# Format code
black backend/
flake8 backend/

# Type checking
mypy backend/
```

### Database Migrations

```bash
# Create new migration
alembic revision --autogenerate -m "Add new feature"

# Apply migrations
alembic upgrade head

# Rollback migration
alembic downgrade -1
```

## 🐳 Docker Deployment

### Build and Run with Docker

```bash
# Build the application
docker build -t llm-platform .

# Run with docker-compose
docker-compose up -d

# Access at http://localhost:9090
```

### Docker Compose Services

- **llm-platform**: Main application server
- **postgres**: PostgreSQL database
- **redis**: Session storage (optional)
- **pgadmin**: Database management UI (available at http://localhost:9003)

## 📊 Monitoring

### Application Metrics

The platform provides built-in monitoring:

- **Health Check**: `GET http://localhost:9090/health`
- **Model Status**: `GET http://localhost:9090/api/v1/models/status/phi-2`
- **Training Stats**: `GET http://localhost:9090/api/v1/training/stats`
- **Feedback Metrics**: `GET http://localhost:9090/api/v1/feedback/stats`

### Logging

Logs are written to:
- **Application**: `./data/logs/app.log`
- **Training**: `./data/logs/training.log`
- **API Access**: `./data/logs/access.log`

### Performance Monitoring

```bash
# View system resources
python backend/scripts/monitor_system.py

# Check model performance
python backend/scripts/benchmark_models.py
```

## 🚀 Production Deployment

### Security Considerations

1. **Change default passwords** and JWT secret keys
2. **Enable HTTPS** with SSL certificates
3. **Configure firewall** to restrict access
4. **Set up backup** procedures for database
5. **Monitor resource usage** and set up alerts

### Scaling Recommendations

- **CPU**: 16+ cores for production workloads
- **RAM**: 64GB+ for handling concurrent users
- **Storage**: SSD storage for model and vector data
- **Network**: High-bandwidth connection for API calls

### Backup Strategy

```bash
# Database backup
pg_dump llm_platform > backup_$(date +%Y%m%d).sql

# Model and vector data backup
tar -czf data_backup_$(date +%Y%m%d).tar.gz data/

# Automated backup script
bash scripts/backup.sh
```

## 🛠️ Troubleshooting

### Common Issues

#### Model Loading Problems
```bash
# Check model files
ls -la data/models/phi-2/

# Verify model integrity
python backend/scripts/verify_model.py

# Re-download model if corrupted
rm -rf data/models/phi-2/
python backend/scripts/setup_model.py
```

#### Database Connection Issues
```bash
# Test database connection
python backend/scripts/test_db.py

# Check PostgreSQL status
sudo systemctl status postgresql

# Reset database
dropdb llm_platform
createdb llm_platform
python backend/scripts/init_db.py
```

#### Memory Issues
```bash
# Monitor memory usage
python backend/scripts/monitor_memory.py

# Optimize model parameters
# Edit backend/utils/config.py to reduce batch sizes
```

### Error Codes

| Code | Description | Solution |
|------|-------------|----------|
| `500` | Model loading failed | Check model files and dependencies |
| `503` | Database unavailable | Verify PostgreSQL is running |
| `429` | Rate limit exceeded | Reduce API request frequency |
| `401` | Authentication failed | Check JWT token validity |

## 🤝 Contributing

### Development Setup

```bash
# Install development dependencies
pip install -r requirements-dev.txt

# Set up pre-commit hooks
pre-commit install

# Run tests before committing
pytest backend/tests/
```

### Code Style

- Follow PEP 8 for Python code
- Use TypeScript for frontend extensions
- Write comprehensive tests for new features
- Document all API endpoints

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🆘 Support

### Documentation
- **API Reference**: http://localhost:9090/docs
- **Developer Guide**: `/docs/development.md`
- **Deployment Guide**: `/docs/deployment.md`

### Community
- **Issues**: Report bugs on GitHub Issues
- **Discussions**: Join community discussions
- **Wiki**: Access community wiki for tips

### Commercial Support
For enterprise deployment and custom development, contact [your-email@domain.com](mailto:your-email@domain.com).

---

## 📝 Changelog

### v1.0.0 (Latest)
- ✅ Initial release with Phi-2 integration
- ✅ Complete web interface
- ✅ Vector search with Faiss
- ✅ Fine-tuning pipeline
- ✅ External model integration
- ✅ Comprehensive documentation

### Upcoming Features
- 🔄 Real-time streaming responses
- 🔄 Multi-user support with roles
- 🔄 Advanced analytics dashboard
- 🔄 Plugin system for extensions
- 🔄 Mobile app interface

---
