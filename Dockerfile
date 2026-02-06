FROM python:3.11-slim

# Install aria2 and other dependencies
RUN apt-get update && apt-get install -y \
    aria2 \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy requirements
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY bot.py .

# Create download directory
RUN mkdir -p /tmp/aria2_downloads

# Create startup script with SECURE aria2 configuration
RUN echo '#!/bin/bash\n\
# Generate random secret if not provided\n\
if [ -z "${ARIA2_SECRET}" ]; then\n\
    export ARIA2_SECRET=$(openssl rand -base64 32)\n\
    echo "Generated ARIA2_SECRET: ${ARIA2_SECRET}"\n\
fi\n\
\n\
# Start aria2 with OPTIMIZED and SECURE settings\n\
aria2c --enable-rpc \\\n\
    --rpc-listen-all=false \\\n\
    --rpc-listen-port=6800 \\\n\
    --rpc-secret="${ARIA2_SECRET}" \\\n\
    --dir="${DOWNLOAD_DIR:-/tmp/aria2_downloads}" \\\n\
    --continue=true \\\n\
    --max-concurrent-downloads=5 \\\n\
    --max-connection-per-server=16 \\\n\
    --split=16 \\\n\
    --min-split-size=1M \\\n\
    --max-download-limit=0 \\\n\
    --disable-ipv6=true \\\n\
    --connect-timeout=30 \\\n\
    --timeout=30 \\\n\
    --max-tries=5 \\\n\
    --retry-wait=3 \\\n\
    --daemon=true \\\n\
    --allow-overwrite=true \\\n\
    --auto-file-renaming=false\n\
\n\
# Wait for aria2 to start\n\
sleep 3\n\
\n\
# Start the bot\n\
python bot.py' > /app/start.sh && chmod +x /app/start.sh

# Expose port (optional, for health checks)
EXPOSE 8080

# Start aria2 and bot
CMD ["/app/start.sh"]
