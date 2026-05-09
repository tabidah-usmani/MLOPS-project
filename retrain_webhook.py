from flask import Flask, request, jsonify
import subprocess
import threading
import time
import requests
from datetime import datetime
import os

app = Flask(__name__)

retraining_in_progress = False
last_retrain_time = None
cooldown_minutes = 30  # Don't retrain more than once every 30 min

def trigger_github_workflow():
    """Trigger GitHub Actions workflow"""
    github_token = os.environ.get('GITHUB_TOKEN')
    repo = os.environ.get('GITHUB_REPO')
    
    if github_token and repo:
        url = f"https://api.github.com/repos/{repo}/actions/workflows/mlops.yml/dispatches"
        headers = {
            "Authorization": f"token {github_token}",
            "Accept": "application/vnd.github.v3+json"
        }
        data = {
            "ref": "main",
            "inputs": {
                "trigger": "drift_detected",
                "timestamp": datetime.now().isoformat()
            }
        }
        requests.post(url, json=data, headers=headers)

def trigger_local_retraining():
    """Trigger local retraining"""
    try:
        # Run training script
        result = subprocess.run(
            ["python", "train.py"],
            cwd="/app",
            capture_output=True,
            text=True,
            timeout=1800  # 30 min timeout
        )
        
        if result.returncode == 0:
            print(f"[{datetime.now()}] Retraining successful!")
            # Reset drift detection after successful retraining
            requests.post("http://api:5000/reset")
            return True
        else:
            print(f"[{datetime.now()}] Retraining failed: {result.stderr}")
            return False
            
    except Exception as e:
        print(f"[{datetime.now()}] Retraining error: {e}")
        return False

@app.route('/retrain', methods=['POST'])
def handle_retrain():
    global retraining_in_progress, last_retrain_time
    
    # Check cooldown
    if retraining_in_progress:
        return jsonify({"status": "retraining already in progress"}), 429
    
    if last_retrain_time:
        minutes_since = (datetime.now() - last_retrain_time).total_seconds() / 60
        if minutes_since < cooldown_minutes:
            return jsonify({
                "status": f"cooldown active, last retrain {minutes_since:.0f} min ago"
            }), 429
    
    # Parse alert
    alert_data = request.json
    print(f"[{datetime.now()}] Drift alert received: {alert_data}")
    
    # Start retraining in background
    retraining_in_progress = True
    
    def retrain_thread():
        global retraining_in_progress, last_retrain_time
        try:
            # Try GitHub workflow first if available
            github_token = os.environ.get('GITHUB_TOKEN')
            if github_token:
                trigger_github_workflow()
                print("GitHub Actions workflow triggered")
            else:
                # Fall back to local retraining
                trigger_local_retraining()
                
            last_retrain_time = datetime.now()
            
        finally:
            retraining_in_progress = False
    
    thread = threading.Thread(target=retrain_thread)
    thread.start()
    
    return jsonify({
        "status": "retraining triggered",
        "timestamp": datetime.now().isoformat()
    }), 202

@app.route('/retrain/status', methods=['GET'])
def retrain_status():
    return jsonify({
        "in_progress": retraining_in_progress,
        "last_retrain": last_retrain_time.isoformat() if last_retrain_time else None,
        "cooldown_minutes": cooldown_minutes
    })

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8080)