#!/bin/bash

echo "🏥 Starting ML Diagnostic Platform"
echo "=============================================="
echo ""

# Install dependencies
echo "📦 Installing dependencies..."
pip install -q -r requirements.txt

echo ""
echo "🚀 Starting Flask server..."
echo ""
echo "📍 The application will be accessible at: http://localhost:5000"
echo ""
echo "👤 Demo accounts:"
echo "   Admin: admin@medical.com / admin123"
echo "   Doctor: doctor@medical.com / doctor123"
echo ""

# Run the application
python run.py
