#!/usr/bin/env python3
from reading_velocity import ReadingVelocityAnalyzer

analyzer = ReadingVelocityAnalyzer()

print("📝 Logging test reading sessions...\n")

# Session 1: 50 pages in 1 hour
analyzer.log_reading_session(1, 1, 50, 3600)
print("✓ Session 1: 50 pages in 1 hour (50 pph)")

# Session 2: 75 pages in 2 hours  
analyzer.log_reading_session(1, 1, 75, 7200)
print("✓ Session 2: 75 pages in 2 hours (37.5 pph)")

# Session 3: 60 pages in 1.5 hours
analyzer.log_reading_session(1, 1, 60, 5400)
print("✓ Session 3: 60 pages in 1.5 hours (40 pph)\n")

# Get velocity
velocity = analyzer.calculate_velocity(1, 1)

print("=" * 50)
print("📊 VELOCITY CALCULATION RESULTS:")
print("=" * 50)
print(f"Average Velocity: {velocity.get('average_velocity_pph', 0):.2f} pph")
print(f"Max Velocity:     {velocity.get('max_velocity_pph', 0):.2f} pph")
print(f"Min Velocity:     {velocity.get('min_velocity_pph', 0):.2f} pph")
print(f"Total Pages:      {velocity.get('total_pages_read', 0)} pages")
print(f"Total Time:       {velocity.get('total_reading_time', {}).get('formatted', '0s')}")
print(f"Sessions:         {velocity.get('session_count', 0)} sessions")
print("=" * 50)
