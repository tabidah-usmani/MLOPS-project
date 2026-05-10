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
cooldown_minutes = int(os.environ.get('COOLDOWN_MINUTES', 30))


def trigger_github_workflow(fake_ratio="unknown", real_ratio="unknown"):
    """Trigger the retrain.yml workflow via GitHub API workflow_dispatch."""
    github_token = os.environ.get('GITHUB_TOKEN')
    repo = os.environ.get('GITHUB_REPO', 'tabidah-usmani/MLOPS-project')

    if not github_token:
        print(f"[{datetime.now()}] GITHUB_TOKEN not set — skipping GitHub dispatch")
        return False

    # FIX 1: was mlops.yml — must be retrain.yml
    url = f"https://api.github.com/repos/{repo}/actions/workflows/retrain.yml/dispatches"
    headers = {
        "Authorization": f"Bearer {github_token}",
        "Accept": "application/vnd.github+json",
        "X-GitHub-Api-Version": "2022-11-28"
    }
    # FIX 2: inputs must match what retrain.yml declares under workflow_dispatch.inputs
    data = {
        "ref": "main",
        "inputs": {
            "reason":     "drift_detected",
            "fake_ratio": str(fake_ratio),
            "real_ratio": str(real_ratio)
        }
    }

    try:
        resp = requests.post(url, json=data, headers=headers, timeout=10)
        if resp.status_code == 204:
            print(f"[{datetime.now()}] GitHub Actions retrain.yml dispatched successfully")
            return True
        else:
            print(f"[{datetime.now()}] GitHub API error {resp.status_code}: {resp.text}")
            return False
    except Exception as e:
        print(f"[{datetime.now()}] GitHub dispatch exception: {e}")
        return False


def trigger_local_retraining():
    """Fallback: run training directly inside the container."""
    try:
        # FIX 3: was "train.py" — correct path is src/train.py
        result = subprocess.run(
            ["python", "src/train.py"],
            cwd="/app",
            capture_output=True,
            text=True,
            timeout=1800
        )

        if result.returncode == 0:
            print(f"[{datetime.now()}] Local retraining successful")
            # Reset drift window after successful retrain
            try:
                requests.post("http://api:5000/reset", timeout=5)
                print(f"[{datetime.now()}] Drift window reset")
            except Exception:
                pass
            return True
        else:
            print(f"[{datetime.now()}] Local retraining failed:\n{result.stderr}")
            return False

    except Exception as e:
        print(f"[{datetime.now()}] Local retraining exception: {e}")
        return False


@app.route('/retrain', methods=['POST'])
def handle_retrain():
    global retraining_in_progress, last_retrain_time

    if retraining_in_progress:
        return jsonify({"status": "retraining already in progress"}), 429

    if last_retrain_time:
        minutes_since = (datetime.now() - last_retrain_time).total_seconds() / 60
        if minutes_since < cooldown_minutes:
            return jsonify({
                "status": f"cooldown active, last retrain {minutes_since:.0f} min ago",
                "cooldown_minutes": cooldown_minutes
            }), 429

    alert_data = request.get_json(silent=True) or {}
    print(f"[{datetime.now()}] Drift alert received: {alert_data}")

    # FIX 4: parse fake_ratio / real_ratio out of Alertmanager's alert labels
    alerts = alert_data.get("alerts", [{}])
    firing = [a for a in alerts if a.get("status") == "firing"]
    if not firing:
        return jsonify({"status": "ignored", "reason": "no firing alerts"}), 200

    labels     = firing[0].get("labels", {})
    fake_ratio = labels.get("fake_ratio", "unknown")
    real_ratio = labels.get("real_ratio", "unknown")
    alert_name = labels.get("alertname", "unknown")
    print(f"[{datetime.now()}] Alert: {alert_name}  fake={fake_ratio}  real={real_ratio}")

    retraining_in_progress = True

    def retrain_thread():
        global retraining_in_progress, last_retrain_time
        try:
            github_token = os.environ.get('GITHUB_TOKEN')
            if github_token:
                success = trigger_github_workflow(fake_ratio, real_ratio)
                if not success:
                    print(f"[{datetime.now()}] GitHub dispatch failed, falling back to local")
                    trigger_local_retraining()
            else:
                trigger_local_retraining()

            last_retrain_time = datetime.now()
        finally:
            retraining_in_progress = False

    threading.Thread(target=retrain_thread, daemon=True).start()

    return jsonify({
        "status":     "retraining triggered",
        "alert":      alert_name,
        "fake_ratio": fake_ratio,
        "real_ratio": real_ratio,
        "timestamp":  datetime.now().isoformat()
    }), 202


@app.route('/retrain/status', methods=['GET'])
def retrain_status():
    return jsonify({
        "in_progress":     retraining_in_progress,
        "last_retrain":    last_retrain_time.isoformat() if last_retrain_time else None,
        "cooldown_minutes": cooldown_minutes
    })


@app.route('/health', methods=['GET'])
def health():
    return jsonify({"status": "ok"}), 200


if __name__ == '__main__':
    print(f"[{datetime.now()}] Retrain webhook starting on :8080")
    print(f"  GITHUB_REPO    : {os.environ.get('GITHUB_REPO', 'not set')}")
    print(f"  GITHUB_TOKEN   : {'set' if os.environ.get('GITHUB_TOKEN') else 'NOT SET'}")
    print(f"  COOLDOWN_MINUTES: {cooldown_minutes}")
    app.run(host='0.0.0.0', port=8080, debug=False)