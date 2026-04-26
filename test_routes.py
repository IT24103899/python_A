#!/usr/bin/env python3
import sys
sys.path.insert(0, '.')

# Import app
from app import app

print("✓ App imported successfully")
print("\nVelocity routes registered:")
found = False
for rule in app.url_map.iter_rules():
    if 'velocity' in str(rule):
        print(f"  {rule}")
        found = True

if not found:
    print("  ❌ NO VELOCITY ROUTES FOUND!")
    print("\nAll routes:")
    for rule in app.url_map.iter_rules():
        print(f"  {rule}")
