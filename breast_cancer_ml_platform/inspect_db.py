
import sqlite3
import pandas as pd

# Connect to the database
# Note: The path depends on where you run the script from. 
# If running from 'breast_cancer_ml_platform', it's 'instance/medical_ml.db'
db_path = 'instance/medical_ml.db'

try:
    conn = sqlite3.connect(db_path)
    print(f"Successfully connected to {db_path}")
    
    # List all tables
    cursor = conn.cursor()
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
    tables = cursor.fetchall()
    print("\nTables found:")
    for table in tables:
        print(f"- {table[0]}")
        
    # Example: Show users
    print("\n--- Users ---")
    users = pd.read_sql_query("SELECT * FROM users", conn)
    print(users)
    
    # Example: Show predictions
    print("\n--- Predictions (First 5) ---")
    predictions = pd.read_sql_query("SELECT * FROM predictions LIMIT 5", conn)
    print(predictions)
    
    conn.close()
    
except Exception as e:
    print(f"Error: {e}")
