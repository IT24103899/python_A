#!/usr/bin/env python3
"""
Reading Velocity & Analytics Engine
Measures reading speed, time periods for book completion, and generates insights
"""

import sys
from datetime import datetime, timedelta
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
import json
import time

# Windows console encoding fix
if sys.platform == 'win32':
    sys.stdout.reconfigure(encoding='utf-8')


def format_time(total_seconds: float) -> Dict:
    """
    Convert seconds to human-readable format
    
    Args:
        total_seconds: Total time in seconds
    
    Returns:
        Dictionary with formatted time components
    """
    seconds = int(total_seconds % 60)
    minutes = int((total_seconds // 60) % 60)
    hours = int((total_seconds // 3600) % 24)
    days = int(total_seconds // 86400)
    
    formatted_str = ""
    if days > 0:
        formatted_str += f"{days}d "
    if hours > 0:
        formatted_str += f"{hours}h "
    if minutes > 0:
        formatted_str += f"{minutes}m "
    if seconds > 0 or not formatted_str:
        formatted_str += f"{seconds}s"
    
    return {
        "total_seconds": total_seconds,
        "days": days,
        "hours": hours,
        "minutes": minutes,
        "seconds": seconds,
        "formatted": formatted_str.strip(),
        "total_minutes": round(total_seconds / 60, 2),
        "total_hours": round(total_seconds / 3600, 2),
    }


class ReadingVelocityAnalyzer:
    """
    Analyzes reading patterns and calculates velocity metrics
    """
    
    def __init__(self):
        self.reading_sessions = []
        self.book_stats = {}
    
    def log_reading_session(self, user_id: int, book_id: int, pages_read: int, 
                           duration_seconds: float, timestamp: datetime = None) -> Dict:
        """
        Log a reading session for a user with detailed timestamps
        
        Args:
            user_id: User identifier
            book_id: Book identifier
            pages_read: Number of pages read in this session
            duration_seconds: Duration of reading session in seconds
            timestamp: Time of session end (default: now)
        
        Returns:
            Session metrics dictionary with start/end timestamps and time of day
        """
        if timestamp is None:
            timestamp = datetime.now()
        
        if duration_seconds <= 0:
            return {"error": "Duration must be positive"}
        
        # Calculate start time (before session)
        start_time = timestamp - timedelta(seconds=duration_seconds)
        
        # Determine time of day
        hour = timestamp.hour
        if 5 <= hour < 12:
            time_of_day = "morning"
        elif 12 <= hour < 17:
            time_of_day = "afternoon"
        elif 17 <= hour < 21:
            time_of_day = "evening"
        else:
            time_of_day = "night"
        
        # Calculate velocity for this session (pages per hour)
        velocity_pph = (pages_read / duration_seconds) * 3600
        
        # Format time
        time_formatted = format_time(duration_seconds)
        
        session = {
            "user_id": user_id,
            "book_id": book_id,
            "pages_read": pages_read,
            "duration_seconds": duration_seconds,
            "duration_formatted": time_formatted,
            "session_start": start_time.isoformat(),
            "session_end": timestamp.isoformat(),
            "session_date": timestamp.date().isoformat(),
            "time_of_day": time_of_day,
            "velocity_pph": round(velocity_pph, 2),
            "timestamp": timestamp.isoformat(),
        }
        
        self.reading_sessions.append(session)
        
        # Update book statistics
        self._update_book_stats(user_id, book_id, pages_read, duration_seconds, velocity_pph)
        
        return session
    
    def _update_book_stats(self, user_id: int, book_id: int, pages_read: int,
                           duration_seconds: float, velocity_pph: float):
        """Update aggregated statistics for a book"""
        key = f"{user_id}_{book_id}"
        
        if key not in self.book_stats:
            self.book_stats[key] = {
                "total_pages_read": 0,
                "total_duration_seconds": 0,
                "session_count": 0,
                "velocities": [],
                "start_date": datetime.now(),
            }
        
        stats = self.book_stats[key]
        stats["total_pages_read"] += pages_read
        stats["total_duration_seconds"] += duration_seconds
        stats["session_count"] += 1
        stats["velocities"].append(velocity_pph)
    
    def calculate_velocity(self, user_id: int, book_id: int) -> Dict:
        """
        Calculate aggregated velocity metrics for a user-book combination
        
        Args:
            user_id: User identifier
            book_id: Book identifier
        
        Returns:
            Dictionary with velocity metrics (with formatted time)
        """
        key = f"{user_id}_{book_id}"
        
        if key not in self.book_stats:
            return {"error": f"No reading data for user {user_id}, book {book_id}"}
        
        stats = self.book_stats[key]
        
        if stats["total_duration_seconds"] == 0:
            return {"error": "No valid duration data"}
        
        # Calculate metrics
        avg_velocity = (stats["total_pages_read"] / stats["total_duration_seconds"]) * 3600
        max_velocity = max(stats["velocities"]) if stats["velocities"] else 0
        min_velocity = min(stats["velocities"]) if stats["velocities"] else 0
        
        # Format total reading time
        time_formatted = format_time(stats["total_duration_seconds"])
        
        return {
            "user_id": user_id,
            "book_id": book_id,
            "average_velocity_pph": round(avg_velocity, 2),  # pages per hour
            "max_velocity_pph": round(max_velocity, 2),
            "min_velocity_pph": round(min_velocity, 2),
            "total_pages_read": stats["total_pages_read"],
            "total_reading_time": time_formatted,
            "session_count": stats["session_count"],
        }
    
    def estimate_completion(self, user_id: int, book_id: int, 
                           total_pages: int, current_page: int) -> Dict:
        """
        Estimate time to complete a book based on current velocity
        
        Args:
            user_id: User identifier
            book_id: Book identifier
            total_pages: Total pages in the book
            current_page: Current page the user is at
        
        Returns:
            Dictionary with completion estimates (with formatted time)
        """
        velocity_data = self.calculate_velocity(user_id, book_id)
        
        if "error" in velocity_data:
            return velocity_data
        
        avg_velocity = velocity_data["average_velocity_pph"]
        
        if avg_velocity == 0:
            return {"error": "Cannot estimate: no velocity data"}
        
        pages_remaining = total_pages - current_page
        
        if pages_remaining <= 0:
            return {
                "pages_remaining": 0,
                "estimated_time": format_time(0),
                "estimated_completion_date": None,
                "status": "completed"
            }
        
        # Calculate time remaining (in seconds)
        seconds_remaining = (pages_remaining / avg_velocity) * 3600
        time_formatted = format_time(seconds_remaining)
        completion_date = datetime.now() + timedelta(seconds=seconds_remaining)
        
        return {
            "user_id": user_id,
            "book_id": book_id,
            "pages_remaining": pages_remaining,
            "current_page": current_page,
            "total_pages": total_pages,
            "progress_percent": round((current_page / total_pages) * 100, 2),
            "average_velocity_pph": avg_velocity,
            "estimated_time": time_formatted,
            "estimated_completion_date": completion_date.isoformat(),
            "status": "in_progress"
        }
    
    def get_reading_heatmap(self, user_id: int, days: int = 28) -> Dict:
        """
        Generate a reading activity heatmap for the past N days
        
        Args:
            user_id: User identifier
            days: Number of days to analyze (default: 28 days)
        
        Returns:
            Dictionary with daily reading activity (with formatted times)
        """
        # Filter sessions for this user in the past N days
        now = datetime.now()
        cutoff_date = now - timedelta(days=days)
        
        user_sessions = [
            s for s in self.reading_sessions
            if s["user_id"] == user_id and 
            datetime.fromisoformat(s["timestamp"]) >= cutoff_date
        ]
        
        # Create a heatmap
        heatmap = {}
        for i in range(days):
            date = (now - timedelta(days=days-1-i)).date().isoformat()
            heatmap[date] = {
                "sessions": 0,
                "total_pages": 0,
                "total_seconds": 0,
                "average_velocity": 0,
            }
        
        # Aggregate sessions by date
        for session in user_sessions:
            session_date = datetime.fromisoformat(session["timestamp"]).date().isoformat()
            if session_date in heatmap:
                heatmap[session_date]["sessions"] += 1
                heatmap[session_date]["total_pages"] += session["pages_read"]
                heatmap[session_date]["total_seconds"] += session["duration_seconds"]
        
        # Calculate average velocity and format time for each day
        for date, data in heatmap.items():
            if data["total_seconds"] > 0:
                data["average_velocity"] = round(
                    (data["total_pages"] / data["total_seconds"]) * 3600, 2
                )
                data["total_time_formatted"] = format_time(data["total_seconds"])
            else:
                data["total_time_formatted"] = format_time(0)
        
        return {
            "user_id": user_id,
            "days_analyzed": days,
            "heatmap": heatmap,
            "total_sessions": len(user_sessions),
            "active_days": len([d for d in heatmap.values() if d["sessions"] > 0]),
        }
    
    def get_reading_stats(self, user_id: int) -> Dict:
        """
        Get comprehensive reading statistics for a user
        
        Args:
            user_id: User identifier
        
        Returns:
            Dictionary with overall reading statistics (with formatted times)
        """
        user_sessions = [s for s in self.reading_sessions if s["user_id"] == user_id]
        
        if not user_sessions:
            return {"error": f"No reading data for user {user_id}"}
        
        df = pd.DataFrame(user_sessions)
        
        total_pages = df["pages_read"].sum()
        total_seconds = df["duration_seconds"].sum()
        
        avg_velocity = (total_pages / total_seconds) * 3600 if total_seconds > 0 else 0
        max_velocity = df["velocity_pph"].max()
        min_velocity = df["velocity_pph"].min()
        
        # Format total reading time
        time_formatted = format_time(total_seconds)
        
        # Calculate streak (consecutive days of reading)
        dates = sorted([
            datetime.fromisoformat(s["timestamp"]).date() 
            for s in user_sessions
        ])
        
        streak = self._calculate_reading_streak(dates)
        
        return {
            "user_id": user_id,
            "total_pages_read": int(total_pages),
            "total_reading_time": time_formatted,
            "session_count": len(user_sessions),
            "books_being_read": len(df["book_id"].unique()),
            "average_velocity_pph": round(avg_velocity, 2),
            "max_velocity_pph": round(max_velocity, 2),
            "min_velocity_pph": round(min_velocity, 2),
            "current_streak_days": streak,
            "last_reading_session": df.iloc[-1]["timestamp"],
        }
    
    def _calculate_reading_streak(self, dates: List) -> int:
        """Calculate consecutive days reading from a sorted list of dates"""
        if not dates:
            return 0
        
        streak = 1
        current_streak = 1
        
        for i in range(1, len(dates)):
            if (dates[i] - dates[i-1]).days == 1:
                current_streak += 1
                streak = max(streak, current_streak)
            else:
                current_streak = 1
        
        return streak
    
    def get_time_of_day_analytics(self, user_id: int) -> Dict:
        """
        Analyze reading patterns by time of day
        
        Args:
            user_id: User identifier
        
        Returns:
            Dictionary with time of day statistics
        """
        user_sessions = [s for s in self.reading_sessions if s["user_id"] == user_id]
        
        if not user_sessions:
            return {"error": f"No reading data for user {user_id}"}
        
        # Initialize time of day categories
        time_of_day_stats = {
            "morning": {"sessions": 0, "pages": 0, "seconds": 0, "velocities": []},
            "afternoon": {"sessions": 0, "pages": 0, "seconds": 0, "velocities": []},
            "evening": {"sessions": 0, "pages": 0, "seconds": 0, "velocities": []},
            "night": {"sessions": 0, "pages": 0, "seconds": 0, "velocities": []},
        }
        
        # Aggregate data by time of day
        for session in user_sessions:
            tod = session.get("time_of_day", "unknown")
            if tod in time_of_day_stats:
                time_of_day_stats[tod]["sessions"] += 1
                time_of_day_stats[tod]["pages"] += session["pages_read"]
                time_of_day_stats[tod]["seconds"] += session["duration_seconds"]
                time_of_day_stats[tod]["velocities"].append(session["velocity_pph"])
        
        # Calculate averages and format times
        result = {"user_id": user_id, "time_of_day_analysis": {}}
        best_time = None
        best_velocity = 0
        
        for tod, data in time_of_day_stats.items():
            if data["sessions"] > 0:
                avg_velocity = (data["pages"] / data["seconds"]) * 3600 if data["seconds"] > 0 else 0
                time_formatted = format_time(data["seconds"])
                
                result["time_of_day_analysis"][tod] = {
                    "sessions": data["sessions"],
                    "pages_read": data["pages"],
                    "total_time": time_formatted,
                    "average_velocity_pph": round(avg_velocity, 2),
                    "best_velocity_pph": round(max(data["velocities"]), 2),
                    "worst_velocity_pph": round(min(data["velocities"]), 2),
                }
                
                if avg_velocity > best_velocity:
                    best_velocity = avg_velocity
                    best_time = tod
        
        result["best_reading_time"] = best_time
        result["best_velocity_at_time"] = round(best_velocity, 2) if best_velocity else 0
        
        return result
    
    def get_session_timeline(self, user_id: int, book_id: int = None) -> Dict:
        """
        Get detailed timeline of all reading sessions with timestamps
        
        Args:
            user_id: User identifier
            book_id: Optional book identifier to filter sessions
        
        Returns:
            Dictionary with chronological session timeline
        """
        if book_id:
            sessions = [s for s in self.reading_sessions 
                       if s["user_id"] == user_id and s["book_id"] == book_id]
        else:
            sessions = [s for s in self.reading_sessions if s["user_id"] == user_id]
        
        if not sessions:
            return {"error": f"No reading data for user {user_id}" + (f" on book {book_id}" if book_id else "")}
        
        # Sort by session start time
        sessions_sorted = sorted(sessions, key=lambda s: s["session_start"])
        
        timeline = []
        for session in sessions_sorted:
            timeline.append({
                "book_id": session["book_id"],
                "session_date": session["session_date"],
                "session_start": session["session_start"],
                "session_end": session["session_end"],
                "time_of_day": session["time_of_day"],
                "duration": session["duration_formatted"],
                "pages_read": session["pages_read"],
                "velocity_pph": session["velocity_pph"],
            })
        
        return {
            "user_id": user_id,
            "total_sessions": len(timeline),
            "book_filter": book_id,
            "timeline": timeline,
            "first_session": timeline[0]["session_start"],
            "last_session": timeline[-1]["session_end"],
        }


# ============================================
# API HELPER FUNCTIONS
# ============================================

def format_velocity_response(data: Dict) -> Dict:
    """Format velocity data for API response"""
    return {
        "success": "error" not in data,
        "data": data
    }


if __name__ == "__main__":
    # Test the analyzer
    print("🚀 Reading Velocity Analyzer Test (with Detailed Timestamps)")
    print("=" * 70)
    
    analyzer = ReadingVelocityAnalyzer()
    
    # Simulate reading sessions with realistic timestamps
    print("\n📚 Simulating reading sessions with timestamps...")
    
    # Morning session: 25 pages in 45 minutes
    morning_end = datetime(2026, 4, 2, 10, 30)
    analyzer.log_reading_session(20, 683, 25, 2700, morning_end)
    
    # Afternoon session: 30 pages in 50 minutes
    afternoon_end = datetime(2026, 4, 2, 14, 15)
    analyzer.log_reading_session(20, 683, 30, 3000, afternoon_end)
    
    # Evening session: 28 pages in 48 minutes
    evening_end = datetime(2026, 4, 2, 19, 45)
    analyzer.log_reading_session(20, 683, 28, 2880, evening_end)
    
    # Another book - evening session
    evening_end2 = datetime(2026, 4, 1, 20, 30)
    analyzer.log_reading_session(20, 685, 40, 3600, evening_end2)
    
    print("\n📊 Velocity Analysis for User 20, Book 683:")
    velocity = analyzer.calculate_velocity(20, 683)
    for key, value in velocity.items():
        if key == "total_reading_time":
            print(f"  {key}: {value['formatted']} ({value['total_minutes']:.2f} min)")
        else:
            print(f"  {key}: {value}")
    
    print("\n⏱️  Completion Estimate (280 total pages, at page 36):")
    estimate = analyzer.estimate_completion(20, 683, 280, 36)
    for key, value in estimate.items():
        if key == "estimated_time":
            print(f"  {key}: {value['formatted']} ({value['total_hours']:.2f} hours)")
        elif key != "estimated_completion_date":
            print(f"  {key}: {value}")
    
    print("\n🕐 Time of Day Analytics:")
    tod = analyzer.get_time_of_day_analytics(20)
    if "error" not in tod:
        print(f"  Best reading time: {tod['best_reading_time']} ({tod['best_velocity_at_time']} pph)")
        for time_period, stats in tod["time_of_day_analysis"].items():
            print(f"  {time_period.upper()}:")
            print(f"    Sessions: {stats['sessions']}")
            print(f"    Pages: {stats['pages_read']}")
            print(f"    Time: {stats['total_time']['formatted']}")
            print(f"    Avg velocity: {stats['average_velocity_pph']} pph")
    
    print("\n📅 Session Timeline (User 20, Book 683):")
    timeline = analyzer.get_session_timeline(20, 683)
    if "error" not in timeline:
        print(f"  Total sessions: {timeline['total_sessions']}")
        print(f"  First session: {timeline['first_session']}")
        print(f"  Last session: {timeline['last_session']}")
        for session in timeline['timeline']:
            print(f"  - {session['session_date']} {session['time_of_day']}: {session['pages_read']}p in {session['duration']['formatted']} ({session['velocity_pph']} pph)")
    
    print("\n📈 Overall User Statistics:")
    stats = analyzer.get_reading_stats(20)
    for key, value in stats.items():
        if key == "total_reading_time":
            print(f"  {key}: {value['formatted']} ({value['total_hours']:.2f} hours)")
        else:
            print(f"  {key}: {value}")
    
    print("\n✅ Test completed!")
