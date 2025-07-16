#!/bin/bash

# =============================================================================
# Self-Learning LLM Platform Startup Script
# =============================================================================
# This script automates the complete setup and startup process for the
# Self-Learning LLM Platform, including dependency installation, environment
# setup, and server startup.
# =============================================================================

set -e  # Exit on any error

# Color codes for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Logging functions
log_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

log_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

log_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Get script directory and project root
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$SCRIPT_DIR"
LLM_ROOT="$PROJECT_ROOT/self-learning-llm"
VENV_PATH="$PROJECT_ROOT/.venv"

# Configuration
DEFAULT_HOST="localhost"
DEFAULT_PORT="8000"
PYTHON_MIN_VERSION="3.8"

# Parse command line arguments
HOST=${1:-$DEFAULT_HOST}
PORT=${2:-$DEFAULT_PORT}
SKIP_INSTALL=${3:-false}

# =============================================================================
# Helper Functions
# =============================================================================

check_python_version() {
    log_info "Checking Python version..."
    
    if ! command -v python3 &> /dev/null; then
        log_error "Python 3 is not installed. Please install Python 3.$PYTHON_MIN_VERSION or higher."
        exit 1
    fi
    
    PYTHON_VERSION=$(python3 -c 'import sys; print(".".join(map(str, sys.version_info[:2])))')
    log_info "Found Python $PYTHON_VERSION"
    
    # Simple version comparison (works for most cases)
    if [[ "$(printf '%s\n' "$PYTHON_MIN_VERSION" "$PYTHON_VERSION" | sort -V | head -n1)" != "$PYTHON_MIN_VERSION" ]]; then
        log_error "Python $PYTHON_MIN_VERSION or higher is required. Found: $PYTHON_VERSION"
        exit 1
    fi
    
    log_success "Python version check passed"
}

setup_virtual_environment() {
    log_info "Setting up virtual environment..."
    
    if [ ! -d "$VENV_PATH" ]; then
        log_info "Creating virtual environment at $VENV_PATH"
        python3 -m venv "$VENV_PATH"
    else
        log_info "Virtual environment already exists"
    fi
    
    # Activate virtual environment
    source "$VENV_PATH/bin/activate"
    log_success "Virtual environment activated"
    
    # Upgrade pip
    log_info "Upgrading pip..."
    pip install --upgrade pip
}

install_dependencies() {
    if [ "$SKIP_INSTALL" = "true" ]; then
        log_info "Skipping dependency installation (--skip-install flag provided)"
        return 0
    fi
    
    log_info "Installing dependencies..."
    
    # Check if requirements.txt exists
    REQUIREMENTS_FILE="$LLM_ROOT/requirements.txt"
    if [ ! -f "$REQUIREMENTS_FILE" ]; then
        log_error "Requirements file not found: $REQUIREMENTS_FILE"
        exit 1
    fi
    
    # Install requirements
    log_info "Installing Python packages from requirements.txt..."
    pip install -r "$REQUIREMENTS_FILE"
    
    log_success "Dependencies installed successfully"
}

verify_dependencies() {
    log_info "Verifying critical dependencies..."
    
    # Test critical imports
    python3 -c "
import sys
try:
    import fastapi
    import uvicorn
    import transformers
    import sentence_transformers
    import faiss
    import sqlalchemy
    import datasets
    print('✅ All critical dependencies verified')
except ImportError as e:
    print(f'❌ Missing dependency: {e}')
    sys.exit(1)
except Exception as e:
    print(f'❌ Dependency verification failed: {e}')
    sys.exit(1)
"
    
    if [ $? -eq 0 ]; then
        log_success "Dependency verification passed"
    else
        log_error "Dependency verification failed"
        exit 1
    fi
}

setup_directories() {
    log_info "Setting up project directories..."
    
    # Create necessary directories
    DIRECTORIES=(
        "$LLM_ROOT/models"
        "$LLM_ROOT/data"
        "$LLM_ROOT/logs"
        "$LLM_ROOT/config"
    )
    
    for dir in "${DIRECTORIES[@]}"; do
        if [ ! -d "$dir" ]; then
            mkdir -p "$dir"
            log_info "Created directory: $dir"
        else
            log_info "Directory already exists: $dir"
        fi
    done
    
    log_success "Directory setup completed"
}

setup_environment_config() {
    log_info "Setting up environment configuration..."
    
    ENV_FILE="$LLM_ROOT/config/.env"
    
    if [ ! -f "$ENV_FILE" ]; then
        log_info "Creating environment configuration file..."
        
        cat > "$ENV_FILE" << EOF
# Database Configuration (for local development)
DB_HOST=localhost
DB_PORT=5432
DB_NAME=self_learning_llm_dev
DB_USER=postgres
DB_PASSWORD=password
DATABASE_URL=postgresql://postgres:password@localhost:5432/self_learning_llm_dev

# Server Configuration
SERVER_HOST=$HOST
SERVER_PORT=$PORT
DEBUG=true

# Model Configuration
MODEL_CACHE_DIR=./models
PHI2_MODEL_PATH=./models/phi2
EMBEDDINGS_MODEL=sentence-transformers/all-MiniLM-L6-v2

# Faiss Configuration
FAISS_INDEX_PATH=./data/faiss_index
VECTOR_DIMENSION=384

# Security (development only)
SECRET_KEY=dev-secret-key-not-for-production
ALGORITHM=HS256
ACCESS_TOKEN_EXPIRE_MINUTES=30

# Logging
LOG_LEVEL=DEBUG
LOG_FILE=./logs/app.log
EOF
        
        log_success "Environment configuration created: $ENV_FILE"
    else
        log_info "Environment configuration already exists: $ENV_FILE"
    fi
}

check_faiss_support() {
    log_info "Checking FAISS support..."
    
    python3 -c "
import faiss
import sys
try:
    print('Loading faiss with AVX2 support.')
    # Try to create a simple index to verify FAISS is working
    dimension = 128
    index = faiss.IndexFlatL2(dimension)
    print('Successfully loaded faiss with AVX2 support.')
except Exception as e:
    print(f'FAISS verification failed: {e}')
    sys.exit(1)
"
    
    if [ $? -eq 0 ]; then
        log_success "FAISS verification passed"
    else
        log_error "FAISS verification failed"
        exit 1
    fi
}

download_model_if_needed() {
    log_info "Checking for Phi-2 model..."
    
    MODEL_PATH="$LLM_ROOT/models/phi2"
    
    if [ -d "$MODEL_PATH" ] && [ "$(ls -A "$MODEL_PATH" 2>/dev/null)" ]; then
        log_success "Phi-2 model already downloaded"
        return 0
    fi
    
    log_warning "Phi-2 model not found. The application will download it automatically on first use."
    log_info "If you want to pre-download the model, you can run the download script manually."
}

start_server() {
    log_info "Starting Self-Learning LLM Platform server..."
    
    # Change to the LLM project directory
    cd "$LLM_ROOT"
    
    # Ensure virtual environment is activated
    source "$VENV_PATH/bin/activate"
    
    log_info "Server configuration:"
    log_info "  Host: $HOST"
    log_info "  Port: $PORT"
    log_info "  Project Root: $LLM_ROOT"
    log_info "  Virtual Environment: $VENV_PATH"
    
    log_success "🚀 Starting server..."
    log_info "🌐 Access the application at: http://$HOST:$PORT"
    log_info "📚 API documentation at: http://$HOST:$PORT/docs"
    log_info "Press Ctrl+C to stop the server"
    
    # Start the server with uvicorn
    python3 -m uvicorn backend.app.main:app \
        --host "$HOST" \
        --port "$PORT" \
        --reload
}

cleanup() {
    log_info "Cleaning up..."
    # Kill any background processes if needed
    # This function is called on script exit
}

show_usage() {
    echo "Usage: $0 [HOST] [PORT] [--skip-install]"
    echo ""
    echo "Arguments:"
    echo "  HOST          Server host (default: localhost)"
    echo "  PORT          Server port (default: 8000)"
    echo "  --skip-install Skip dependency installation"
    echo ""
    echo "Examples:"
    echo "  $0                    # Start on localhost:8000"
    echo "  $0 0.0.0.0 8080      # Start on 0.0.0.0:8080"
    echo "  $0 localhost 8000 --skip-install  # Skip pip install"
}

# =============================================================================
# Main Execution
# =============================================================================

main() {
    # Handle help flag
    if [[ "$1" == "-h" ]] || [[ "$1" == "--help" ]]; then
        show_usage
        exit 0
    fi
    
    # Set up signal handlers for cleanup
    trap cleanup EXIT
    trap 'log_info "Interrupted by user"; exit 130' INT TERM
    
    log_success "🚀 Self-Learning LLM Platform Startup Script"
    log_info "=============================================="
    log_info "Project Root: $PROJECT_ROOT"
    log_info "LLM Root: $LLM_ROOT"
    log_info "Target Host: $HOST"
    log_info "Target Port: $PORT"
    echo ""
    
    # Pre-flight checks
    if [ ! -d "$LLM_ROOT" ]; then
        log_error "LLM project directory not found: $LLM_ROOT"
        log_error "Please ensure you're running this script from the correct directory."
        exit 1
    fi
    
    # Execute setup steps
    check_python_version
    setup_virtual_environment
    install_dependencies
    verify_dependencies
    setup_directories
    setup_environment_config
    check_faiss_support
    download_model_if_needed
    
    log_success "✅ Setup completed successfully!"
    echo ""
    
    # Start the server
    start_server
}

# Run main function
main "$@"