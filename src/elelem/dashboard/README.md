# 🚀 Elelem Dashboard

A standalone dashboard for Elelem metrics visualization. Completely independent of the main Elelem codebase.

## Features

- 📊 Real-time metrics visualization
- 📋 Raw database table view with filtering
- 💰 Cost breakdown by model and provider
- 📈 Interactive charts and time-series analysis
- ❌ Error tracking and analysis
- 🔄 Auto-refresh capabilities

## Environment Variables

| Variable | Description | Required |
|----------|-------------|----------|
| `DATABASE_URL` | PostgreSQL connection string | ✅ Yes |
| `DASHBOARD_TITLE` | Custom dashboard title | ❌ No |

## Quick Start

### Docker

```bash
# Build the dashboard
docker build -t elelem-dashboard .

# Run with PostgreSQL connection
docker run -p 8501:8501 \
  -e DATABASE_URL="postgresql://user:pass@host:5432/db" \
  elelem-dashboard
```

### Docker Compose

```yaml
version: '3.8'
services:
  dashboard:
    build: .
    ports:
      - "8501:8501"
    environment:
      - DATABASE_URL=postgresql://elelem:password@postgres:5432/elelem_db
      - DASHBOARD_TITLE=My Elelem Dashboard
```

### Local Development

```bash
# Install dependencies
pip install -r requirements.txt

# Set environment variable
export DATABASE_URL="postgresql://user:pass@localhost:5432/elelem_db"

# Run dashboard
streamlit run dashboard.py
```

## Serverless Deployment

### Railway

```bash
railway deploy
```

### Render

Connect your repository and set:
- **Build Command**: `pip install -r requirements.txt`
- **Start Command**: `streamlit run dashboard.py --server.port=$PORT --server.address=0.0.0.0`
- **Environment**: `DATABASE_URL=your_postgres_url`

### Heroku

```bash
# Create Procfile
echo "web: streamlit run dashboard.py --server.port=\$PORT --server.address=0.0.0.0" > Procfile

# Deploy
git add . && git commit -m "Deploy dashboard"
heroku create your-dashboard-name
heroku config:set DATABASE_URL="your_postgres_url"
git push heroku main
```

### AWS Lambda (with Serverless Framework)

```yaml
# serverless.yml
service: elelem-dashboard

provider:
  name: aws
  runtime: python3.11
  environment:
    DATABASE_URL: ${env:DATABASE_URL}

functions:
  dashboard:
    handler: wsgi_handler.handler
    events:
      - http: ANY /
      - http: 'ANY {proxy+}'

plugins:
  - serverless-wsgi
  - serverless-python-requirements
```

## Database Schema

The dashboard expects a `request_metrics` table with the following columns:

- `request_id` (TEXT PRIMARY KEY)
- `timestamp` (TIMESTAMP)
- `requested_model`, `actual_model`, `actual_provider` (TEXT)
- `status` (TEXT: 'success' or 'failed')
- `total_duration_seconds` (FLOAT)
- `input_tokens`, `output_tokens`, `reasoning_tokens` (INTEGER)
- `total_cost_usd` (FLOAT)
- `tags` (JSON/TEXT)
- `final_error_type`, `final_error_message` (TEXT)
- Various retry and performance metrics

See `schema.py` for the complete schema definition.

## Customization

The dashboard is designed to be easily customizable:

1. **Modify `dashboard.py`** to add new charts or metrics
2. **Update `schema.py`** if your database schema differs
3. **Customize styling** by modifying Streamlit components
4. **Add new environment variables** for configuration

## Security

- Never include database credentials in code
- Use environment variables for all sensitive configuration
- Consider using connection pooling for high-traffic deployments
- Restrict database access to read-only permissions for the dashboard

## License

This dashboard inherits the same license as the main Elelem project.