# Installation Guide

This guide provides detailed instructions for setting up the Self-Learning LLM Platform on various systems.

## 📋 Prerequisites

### System Requirements

#### Minimum Requirements
- **CPU**: 4-core processor (Intel i5/AMD Ryzen 5 equivalent)
- **RAM**: 24GB system memory
- **Storage**: 200GB available space (NVMe SSD recommended)
- **OS**: Linux (Ubuntu 20.04+), macOS 10.15+, or Windows 10+
- **Network**: Stable internet connection for initial setup

#### Recommended Requirements
- **CPU**: 8+ core processor (Intel i7/AMD Ryzen 7 equivalent)
- **RAM**: 32GB+ system memory
- **Storage**: 500GB+ NVMe SSD
- **OS**: Ubuntu 22.04 LTS or CentOS 8+
- **Network**: High-speed broadband connection

### Software Dependencies

#### Required Software
- **Python**: 3.8, 3.9, 3.10, or 3.11
- **PostgreSQL**: 12, 13, 14, or 15
- **Git**: Latest version
- **pip**: Python package manager

#### Optional Software
- **Docker**: For containerized deployment
- **Node.js**: For frontend development
- **nginx**: For production reverse proxy

## 🐧 Linux Installation (Ubuntu/Debian)

### 1. System Update

```bash
sudo apt update && sudo apt upgrade -y
```

### 2. Install Python and Dependencies

```bash
# Install Python 3.8+ and pip
sudo apt install python3 python3-pip python3-venv python3-dev -y

# Install system dependencies
sudo apt install build-essential libssl-dev libffi-dev -y
sudo apt install libpq-dev postgresql-client -y

# Verify Python version
python3 --version  # Should be 3.8+
```

### 3. Install PostgreSQL

```bash
# Install PostgreSQL
sudo apt install postgresql postgresql-contrib -y

# Start and enable PostgreSQL
sudo systemctl start postgresql
sudo systemctl enable postgresql

# Create database and user
sudo -u postgres psql << EOF
CREATE DATABASE llm_platform;
CREATE USER llm_user WITH PASSWORD 'secure_password_here';
GRANT ALL PRIVILEGES ON DATABASE llm_platform TO llm_user;
ALTER USER llm_user CREATEDB;
\q
EOF
```

### 4. Install Git

```bash
sudo apt install git -y
```

### 5. Clone Repository

```bash
git clone https://github.com/your-username/self-learning-llm-platform.git
cd self-learning-llm-platform
```

### 6. Set Up Python Environment

```bash
# Create virtual environment
python3 -m venv venv

# Activate virtual environment
source venv/bin/activate

# Upgrade pip
pip install --upgrade pip

# Install dependencies
pip install -r requirements.txt
```

### 7. Configure Environment

```bash
# Copy environment template
cp .env.example .env

# Edit configuration
nano .env
```

Update the database URL:
```env
DATABASE_URL=postgresql://llm_user:secure_password_here@localhost:5432/llm_platform
```

### 8. Initialize Database

```bash
cd backend
python scripts/init_db.py
```

### 9. Download Models

```bash
# Download Phi-2 model (this may take 20-30 minutes)
python scripts/setup_model.py

# Initialize embeddings
python scripts/init_embeddings.py
```

### 10. Test Installation

```bash
# Test Phi-2 model
python scripts/test_phi2.py

# Test the API server
python -m uvicorn main:app --host 0.0.0.0 --port 9090
```

## 🍎 macOS Installation

### 1. Install Homebrew

```bash
/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
```

### 2. Install Python

```bash
# Install Python 3.8+
brew install python@3.11

# Verify installation
python3 --version
```

### 3. Install PostgreSQL

```bash
# Install PostgreSQL
brew install postgresql@15

# Start PostgreSQL service
brew services start postgresql@15

# Create database
createdb llm_platform

# Create user (optional)
psql llm_platform -c "CREATE USER llm_user WITH PASSWORD 'secure_password';"
psql llm_platform -c "GRANT ALL PRIVILEGES ON DATABASE llm_platform TO llm_user;"
```

### 4. Clone and Setup

```bash
# Clone repository
git clone https://github.com/your-username/self-learning-llm-platform.git
cd self-learning-llm-platform

# Create virtual environment
python3 -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### 5. Configure and Initialize

```bash
# Configure environment
cp .env.example .env
# Edit .env with your database settings

# Initialize database
cd backend
python scripts/init_db.py

# Download models
python scripts/setup_model.py
python scripts/init_embeddings.py
```

## 🪟 Windows Installation

### 1. Install Python

Download and install Python 3.8+ from [python.org](https://www.python.org/downloads/windows/):
- Check "Add Python to PATH" during installation
- Choose "Install for all users" if needed

### 2. Install PostgreSQL

Download and install PostgreSQL from [postgresql.org](https://www.postgresql.org/download/windows/):
- Remember the password you set for the postgres user
- Include pgAdmin 4 in the installation

### 3. Install Git

Download and install Git from [git-scm.com](https://git-scm.com/download/win)

### 4. Clone Repository

```cmd
git clone https://github.com/your-username/self-learning-llm-platform.git
cd self-learning-llm-platform
```

### 5. Set Up Python Environment

```cmd
REM Create virtual environment
python -m venv venv

REM Activate virtual environment
venv\Scripts\activate

REM Upgrade pip
python -m pip install --upgrade pip

REM Install dependencies
pip install -r requirements.txt
```

### 6. Configure Database

Using pgAdmin 4 or psql:
```sql
CREATE DATABASE llm_platform;
CREATE USER llm_user WITH PASSWORD 'secure_password';
GRANT ALL PRIVILEGES ON DATABASE llm_platform TO llm_user;
```

### 7. Configure Environment

```cmd
copy .env.example .env
REM Edit .env with your database settings
```

### 8. Initialize and Test

```cmd
cd backend
python scripts/init_db.py
python scripts/setup_model.py
python scripts/test_phi2.py
```

## 🐳 Docker Installation

### 1. Install Docker

#### Linux (Ubuntu/Debian)
```bash
# Install Docker
curl -fsSL https://get.docker.com -o get-docker.sh
sudo sh get-docker.sh

# Install Docker Compose
sudo apt install docker-compose -y

# Add user to docker group
sudo usermod -aG docker $USER
newgrp docker
```

#### macOS
```bash
# Install Docker Desktop
brew install --cask docker

# Start Docker Desktop from Applications
```

#### Windows
Download and install Docker Desktop from [docker.com](https://www.docker.com/products/docker-desktop)

### 2. Clone Repository

```bash
git clone https://github.com/your-username/self-learning-llm-platform.git
cd self-learning-llm-platform
```

### 3. Configure Environment

```bash
cp .env.example .env
# Edit .env with your settings
```

### 4. Build and Run

```bash
# Build and start all services
docker-compose up -d

# View logs
docker-compose logs -f

# Access the application at http://localhost:9090
```

### 5. Initialize Data

```bash
# Initialize database
docker-compose exec llm-platform python scripts/init_db.py

# Download models
docker-compose exec llm-platform python scripts/setup_model.py

# Initialize embeddings
docker-compose exec llm-platform python scripts/init_embeddings.py
```

## 🔧 Advanced Configuration

### Custom Model Path

```bash
# Set custom model storage location
export PHI2_MODEL_PATH="/path/to/your/models"

# Update .env file
echo "PHI2_MODEL_PATH=/path/to/your/models" >> .env
```

### Custom Database Configuration

```bash
# For custom PostgreSQL settings
export DATABASE_URL="postgresql://user:password@host:port/database"

# For SSL connections
export DATABASE_URL="postgresql://user:password@host:port/database?sslmode=require"
```

### External API Keys

```bash
# Add to .env file
echo "OPENAI_API_KEY=your-openai-key" >> .env
echo "ANTHROPIC_API_KEY=your-anthropic-key" >> .env
echo "GOOGLE_API_KEY=your-google-key" >> .env
```

### Performance Tuning

```bash
# Optimize for your system
export PYTORCH_NUM_THREADS=8  # Match your CPU cores
export OMP_NUM_THREADS=8      # OpenMP threads
export CUDA_VISIBLE_DEVICES="" # Force CPU usage
```

## 🧪 Testing Installation

### Basic Tests

```bash
# Test database connection
python backend/scripts/test_db.py

# Test Phi-2 model
python backend/scripts/test_phi2.py

# Test vector search
python backend/scripts/test_vector_search.py

# Test external models (if configured)
python backend/scripts/test_external_models.py
```

### Performance Tests

```bash
# Memory usage test
python backend/scripts/monitor_memory.py

# Generation speed test
python backend/scripts/benchmark_models.py

# Concurrent user test
python backend/scripts/load_test.py
```

### Integration Tests

```bash
# Run full test suite
pytest backend/tests/

# Run specific test categories
pytest backend/tests/test_models.py -v
pytest backend/tests/test_api.py -v
pytest backend/tests/test_vector_store.py -v
```

## 🚀 Production Deployment

### Security Setup

```bash
# Generate secure secrets
python -c "import secrets; print(secrets.token_urlsafe(32))"

# Set up SSL certificates
sudo certbot --nginx -d your-domain.com

# Configure firewall
sudo ufw allow 80
sudo ufw allow 443
sudo ufw allow 22
sudo ufw enable
```

### Process Management

```bash
# Install process manager
pip install supervisor

# Create supervisor config
sudo nano /etc/supervisor/conf.d/llm-platform.conf
```

Supervisor configuration:
```ini
[program:llm-platform]
command=/path/to/venv/bin/python -m uvicorn main:app --host 0.0.0.0 --port 9090
directory=/path/to/self-learning-llm-platform/backend
user=llm-user
autostart=true
autorestart=true
stderr_logfile=/var/log/llm-platform.err.log
stdout_logfile=/var/log/llm-platform.out.log
```

### Load Balancing

```bash
# Install nginx
sudo apt install nginx

# Configure nginx
sudo nano /etc/nginx/sites-available/llm-platform
```

Nginx configuration:
```nginx
server {
    listen 80;
    server_name your-domain.com;

    location / {
        proxy_pass http://127.0.0.1:9090;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
    }

    location /static/ {
        alias /path/to/self-learning-llm-platform/frontend/;
    }
}
```

## 🛠️ Troubleshooting

### Common Issues

#### Permission Errors
```bash
# Fix Python path permissions
sudo chown -R $USER:$USER /path/to/self-learning-llm-platform
chmod +x backend/scripts/*.py
```

#### Memory Issues
```bash
# Check available memory
free -h

# Optimize model loading
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512

# Reduce batch sizes in config
```

#### Database Connection Issues
```bash
# Test PostgreSQL connection
pg_isready -h localhost -p 5432

# Check PostgreSQL logs
sudo tail -f /var/log/postgresql/postgresql-*.log

# Reset database
dropdb llm_platform
createdb llm_platform
python backend/scripts/init_db.py
```

### Getting Help

1. Check the [troubleshooting guide](troubleshooting.md)
2. Search existing [GitHub issues](https://github.com/your-username/self-learning-llm-platform/issues)
3. Create a new issue with system information
4. Join the community discussion

---

## 📝 Next Steps

After successful installation:

1. **Read the [User Guide](user-guide.md)** for usage instructions
2. **Configure [External APIs](external-apis.md)** for model comparison
3. **Set up [Monitoring](monitoring.md)** for production use
4. **Explore [Advanced Features](advanced-features.md)** for customization

---

*For additional help, consult the [FAQ](faq.md) or contact support.*