# AI-Powered Business Intelligence for Organizations

A comprehensive business intelligence platform that leverages AI and machine learning to analyze data, generate insights, and provide intelligent recommendations for organizations.

## Features

- **CSV Data Analysis**: Advanced analytics and visualization for CSV datasets
- **PDF Document Processing**: Extract and analyze content from PDF documents
- **RAG (Retrieval-Augmented Generation)**: Intelligent document search and Q&A
- **Vector Database Storage**: Efficient embedding storage with pgvector
- **Interactive Web Interface**: User-friendly Gradio-based interface

## Tech Stack

- **Backend**: Python, LangChain, Pandas, NumPy
- **AI/ML**: Sentence Transformers, Hugging Face, Groq
- **Database**: PostgreSQL with pgvector extension
- **Frontend**: Gradio
- **Vector Search**: FAISS, pgvector
- **Containerization**: Docker & Docker Compose

## Quick Start with Docker

### Prerequisites

- Docker and Docker Compose installed on your system
- Git

### Setup Instructions

1. **Clone the repository**
   ```bash
   git clone <repository-url>
   cd Enabling_AI-Powered_Business_Intelligence_for_Organizations
   ```

2. **Start the services**
   ```bash
   docker-compose up -d
   ```

   This will start:
   - PostgreSQL database with pgvector extension
   - The application server

3. **Access the application**
   
   The application will be available at `http://localhost:7860`

4. **Stop the services**
   ```bash
   docker-compose down
   ```

### Docker Troubleshooting

#### Common Issues

**Issue: Docker Desktop not running**
```
Error: unable to get image: error during connect: Get "http://%2F%2F.%2Fpipe%2FdockerDesktopLinuxEngine/v1.51/..."
```
**Solution**: Start Docker Desktop manually from Windows Start Menu

**Issue: Only need PostgreSQL, not the full app**
```bash
# Run only PostgreSQL database
docker-compose up -d postgres

# Or run PostgreSQL directly
docker run -d --name bi_postgres -p 5432:5432 \
  -e POSTGRES_DB=bi_db \
  -e POSTGRES_USER=postgres \
  -e POSTGRES_PASSWORD=postgres \
  pgvector/pgvector:pg16
```

#### Docker Cleanup

If you need to clean up Docker resources:

```bash
# Stop and remove containers
docker-compose down

# Remove app image (keeps PostgreSQL image)
docker rmi enabling_ai-powered_business_intelligence_for_organizations-app

# Clean up unused images and volumes
docker image prune -f
docker volume prune -f
```

#### Running Python App Locally with Docker PostgreSQL

1. Start only PostgreSQL:
   ```bash
   docker-compose up -d postgres
   ```

2. Run Python app locally:
   ```bash
   venv\Scripts\activate
   python app/main.py --mode csv
   ```

### Docker Services

#### PostgreSQL Database
- **Image**: `pgvector/pgvector:pg16`
- **Port**: 5432
- **Database**: `bi_db`
- **User**: `postgres`
- **Password**: `postgres`
- **Extension**: pgvector (enabled automatically)

#### Application Server
- **Port**: 7860
- **Environment**: Connected to PostgreSQL database
- **Dependencies**: All Python packages installed from requirements.txt

## Database Schema

The PostgreSQL database includes the following tables:

- **document_embeddings**: Stores document content and embeddings
- **document_chunks**: Stores chunked document segments with embeddings
- **analysis_results**: Stores analysis results and metadata

## Local Development Setup

### Prerequisites

- Python 3.11+
- PostgreSQL 16+ with pgvector extension (OR Docker Desktop)
- Git

### Quick Setup (Recommended)

1. **Clone the repository**
   ```bash
   git clone <repository-url>
   cd Enabling_AI-Powered_Business_Intelligence_for_Organizations
   ```

2. **Run the setup script**
   ```bash
   python setup_env.py
   ```
   
   This script will:
   - Create a `.env` file with proper PYTHONPATH configuration
   - Install all dependencies
   - Create necessary directories
   - Set up the development environment

3. **Configure your API key**
   - Edit the `.env` file
   - Set your `GROQ_API_KEY`

4. **Run the application**
   ```bash
   python app/main.py
   ```

### Manual Setup

1. **Clone the repository**
   ```bash
   git clone <repository-url>
   cd Enabling_AI-Powered_Business_Intelligence_for_Organizations
   ```

2. **Create virtual environment**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Set up environment variables**
   - Copy `.env.example` to `.env`
   - Update the `PYTHONPATH` to your project root directory
   - Set your `GROQ_API_KEY`

5. **Set up PostgreSQL**

**Option A: Using Docker (Recommended)**
```bash
# Start only PostgreSQL database
docker-compose up -d postgres
```

**Option B: Local Installation**
```bash
# Install PostgreSQL 16+ with pgvector extension
# Create database named 'bi_db'
# Enable pgvector extension
```

6. **Run the application**
   ```bash
   python app/main.py
   ```

### Environment Variables

Create a `.env` file with the following variables:

```bash
# Required: Add project root to Python path
PYTHONPATH=/path/to/your/project

# Database Configuration
DATABASE_URL=postgresql://postgres:postgres@localhost:5432/bi_db

# Web Interface Configuration
WEB_HOST=0.0.0.0
WEB_PORT=7860

# Logging Configuration
LOG_LEVEL=INFO

# AI/ML Configuration
GROQ_API_KEY=your_groq_api_key_here
```

## Configuration

### Environment Variables

- `DATABASE_URL`: PostgreSQL connection string
- `GROQ_API_KEY`: Groq API key for AI services

### Database Configuration

The application uses PostgreSQL with pgvector for vector storage. The database is automatically initialized with the necessary tables and indexes when using Docker Compose.

## Makefile Commands

The project includes a Makefile to simplify common development tasks. Run `make help` to see all available commands.

### Application Modes

- **`make web`** - Start the web interface on port 7860 (automatically kills existing process if port is in use)
- **`make csv`** - Run CSV analysis mode
- **`make process`** - Run document processing mode

### Service Management

- **`make stop`** - Stop the application (kills process on port 7860)
- **`make kill-port`** - Force kill any process on port 7860
- **`make start-db`** - Start PostgreSQL database using Docker Compose
- **`make stop-db`** - Stop PostgreSQL database using Docker Compose

### Maintenance

- **`make clean`** - Clean up Python cache files (.pyc files and __pycache__ directories)
- **`make help`** - Display all available make commands

### Example Workflow

```bash
# Start the database
make start-db

# Run the web interface
make web

# When done, stop the application
make stop

# Stop the database
make stop-db
```

## Usage

1. **Upload CSV files** for data analysis and visualization
2. **Upload PDF documents** for content extraction and analysis
3. **Ask questions** about your documents using the RAG interface
4. **Generate insights** and reports from your data

## Project Structure

```
├── app/
│   ├── main.py              # Main application entry point
│   ├── csv_analyzer.py      # CSV analysis functionality
│   └── rag/
│       ├── pdf_loader.py    # PDF document processing
│       └── generate_embeddings.py  # Embedding generation
├── data/                    # Data directory
├── output/                  # Output directory
├── Dockerfile              # Docker configuration
├── docker-compose.yml      # Docker Compose configuration
├── init.sql                # Database initialization script
├── requirements.txt        # Python dependencies
└── README.md              # This file
```

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## License

This project is licensed under the MIT License.

## Support

For issues and questions, please open an issue on the GitHub repository.
