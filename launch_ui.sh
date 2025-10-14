#!/bin/bash
# Quick Launch Script for ML Classification Web UI

echo "🚀 Launching ML Classification Web UI..."
echo ""
echo "📍 Location: http://localhost:8501"
echo "⚡ Press Ctrl+C to stop the server"
echo ""
echo "🎨 Opening in browser..."
echo ""

# Navigate to script directory
cd "$(dirname "$0")"

# Launch Streamlit
streamlit run app.py --server.headless false

echo ""
echo "✅ Server stopped. Goodbye!"

