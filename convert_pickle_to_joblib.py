#!/usr/bin/env python3
"""
Converts books_metadata.pkl from pandas pickle to joblib format
for better compatibility with different numpy versions on Render.
"""
import pandas as pd
import numpy as np
import joblib
import pickle
import os

def convert_pickle_to_joblib():
    """Convert the pickle file to joblib format"""
    
    if not os.path.exists("books_metadata.pkl"):
        print("❌ books_metadata.pkl not found!")
        return False
    
    try:
        print("Loading books_metadata.pkl with pandas...")
        # Try standard pandas pickle load
        try:
            df_meta = pd.read_pickle("books_metadata.pkl")
            print("✓ Loaded successfully with pd.read_pickle()")
        except Exception as e:
            print(f"⚠ pd.read_pickle() failed: {e}")
            print("Attempting fallback pickle load...")
            with open("books_metadata.pkl", "rb") as f:
                df_meta = pickle.load(f)
            print("✓ Loaded successfully with pickle.load()")
        
        print(f"Metadata shape: {df_meta.shape}")
        print(f"Columns: {df_meta.columns.tolist()}")
        
        # Save with joblib (more compatible)
        print("\nSaving with joblib...")
        joblib.dump(df_meta, "books_metadata.pkl", compress=3)
        print("✓ Successfully converted to joblib format!")
        
        # Verify it loads with joblib
        print("Verifying joblib load...")
        df_verify = joblib.load("books_metadata.pkl")
        print(f"✓ Verification successful! Shape: {df_verify.shape}")
        
        return True
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = convert_pickle_to_joblib()
    exit(0 if success else 1)
