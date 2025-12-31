"""
XRP Rich List API Server
Run this alongside your dashboard to serve data with historical comparison
"""

from flask import Flask, jsonify
from flask_cors import CORS
import requests
import pandas as pd
from datetime import datetime, timedelta
from pathlib import Path
import json
import numpy as np

app = Flask(__name__)
CORS(app)  # Enable CORS for dashboard access

# Configuration
DATA_DIR = Path("./xrp_data")
CSV_DIR = DATA_DIR / "csv"
CACHE_FILE = DATA_DIR / "latest_data.json"

# Thresholds
WHALE_THRESHOLD = 1_000_000
SIGNIFICANT_CHANGE = 1_000
LARGE_CHANGE = 100_000

# Ensure directories exist
DATA_DIR.mkdir(exist_ok=True)
CSV_DIR.mkdir(exist_ok=True)


def fetch_current_data():
    """Fetch current data from XRPScan API"""
    try:
        response = requests.get(
            "https://api.xrpscan.com/api/v1/balances",
            timeout=30
        )
        response.raise_for_status()
        return response.json()
    except Exception as e:
        print(f"Error fetching data: {e}")
        return None


def process_data(raw_data):
    """Process raw API data"""
    balances = []
    for entry in raw_data:
        account = entry.get("account", "Unknown")
        balance_drops = entry.get("balance", 0)
        balance_xrp = balance_drops / 1_000_000
        
        name_info = entry.get("name")
        name = name_info.get("name", "Unknown") if name_info else "Unknown"
        domain = name_info.get("domain", "") if name_info else ""
        
        balances.append({
            "account": account,
            "balance": balance_xrp,
            "name": name,
            "domain": domain,
            "isWhale": balance_xrp >= WHALE_THRESHOLD
        })
    
    # Sort by balance and assign ranks
    balances.sort(key=lambda x: x["balance"], reverse=True)
    for i, acc in enumerate(balances):
        acc["rank"] = i + 1
    
    return balances


def load_previous_data(days_back=1):
    """Load historical data from CSV"""
    target_date = (datetime.now() - timedelta(days=days_back)).strftime("%Y-%m-%d")
    csv_path = CSV_DIR / f"xrp_top_10000_balances_{target_date}.csv"
    
    if csv_path.exists():
        df = pd.read_csv(csv_path)
        return df.to_dict('records')
    return None


def compare_data(current, previous):
    """Compare current and previous data to find changes"""
    if not previous:
        return [], {"new_accounts": 0, "removed_accounts": 0, "changed_accounts": 0,
                   "new_accounts_xrp": 0, "removed_accounts_xrp": 0, "net_flow": 0, "total_net_change": 0}
    
    changes = []
    prev_map = {acc.get("Account", acc.get("account")): acc for acc in previous}
    curr_map = {acc["account"]: acc for acc in current}
    
    # New accounts
    for acc in current:
        account_id = acc["account"]
        if account_id not in prev_map:
            changes.append({
                **acc,
                "change": acc["balance"],
                "changePct": 100,
                "status": "New Account"
            })
        else:
            prev = prev_map[account_id]
            prev_balance = prev.get("Balance_XRP", prev.get("balance", 0))
            change = acc["balance"] - prev_balance
            if abs(change) >= SIGNIFICANT_CHANGE:
                changes.append({
                    **acc,
                    "previousBalance": prev_balance,
                    "change": change,
                    "changePct": (change / prev_balance * 100) if prev_balance > 0 else 100,
                    "status": "Balance Changed"
                })
    
    # Removed accounts
    for account_id, prev in prev_map.items():
        if account_id not in curr_map:
            prev_balance = prev.get("Balance_XRP", prev.get("balance", 0))
            changes.append({
                "account": account_id,
                "name": prev.get("Name", prev.get("name", "Unknown")),
                "balance": 0,
                "rank": 0,
                "isWhale": prev_balance >= WHALE_THRESHOLD,
                "change": -prev_balance,
                "changePct": -100,
                "status": "Removed"
            })
    
    # Calculate summary
    new_accounts = [c for c in changes if c["status"] == "New Account"]
    removed_accounts = [c for c in changes if c["status"] == "Removed"]
    changed_accounts = [c for c in changes if c["status"] == "Balance Changed"]
    
    summary = {
        "new_accounts": len(new_accounts),
        "removed_accounts": len(removed_accounts),
        "changed_accounts": len(changed_accounts),
        "new_accounts_xrp": sum(c["change"] for c in new_accounts),
        "removed_accounts_xrp": sum(abs(c["change"]) for c in removed_accounts),
        "net_flow": sum(c["change"] for c in changed_accounts),
        "total_net_change": sum(c["change"] for c in changes)
    }
    
    # Sort by absolute change
    changes.sort(key=lambda x: abs(x["change"]), reverse=True)
    
    return changes, summary


def detect_alerts(changes):
    """Detect significant events"""
    alerts = []
    for change in changes:
        if abs(change["change"]) >= LARGE_CHANGE:
            alerts.append({
                "type": "LARGE_MOVEMENT",
                "account": change["account"],
                "name": change["name"],
                "change": change["change"],
                "direction": "inflow" if change["change"] > 0 else "outflow"
            })
        
        if change["status"] == "New Account" and change["balance"] >= WHALE_THRESHOLD:
            alerts.append({
                "type": "NEW_WHALE",
                "account": change["account"],
                "name": change.get("name", "Unknown"),
                "balance": change["balance"]
            })
    
    return alerts


def calculate_stats(data):
    """Calculate market statistics"""
    total_xrp = sum(acc["balance"] for acc in data)
    whales = [acc for acc in data if acc["isWhale"]]
    whale_holdings = sum(acc["balance"] for acc in whales)
    
    top10 = sum(acc["balance"] for acc in data[:10])
    top100 = sum(acc["balance"] for acc in data[:100])
    
    balances = [acc["balance"] for acc in data]
    mean_balance = total_xrp / len(data) if data else 0
    sorted_balances = sorted(balances)
    mid = len(sorted_balances) // 2
    median_balance = sorted_balances[mid] if len(sorted_balances) % 2 else (sorted_balances[mid-1] + sorted_balances[mid]) / 2
    
    # Gini coefficient
    sorted_vals = np.array(sorted_balances)
    n = len(sorted_vals)
    weighted_sum = np.sum((np.arange(1, n + 1)) * sorted_vals)
    gini = (2 * weighted_sum) / (n * np.sum(sorted_vals)) - (n + 1) / n
    
    return {
        "totalXRP": total_xrp,
        "totalAccounts": len(data),
        "whaleCount": len(whales),
        "whaleHoldings": whale_holdings,
        "whalePercentage": (whale_holdings / total_xrp * 100) if total_xrp > 0 else 0,
        "top10Holdings": top10,
        "top10Percentage": (top10 / total_xrp * 100) if total_xrp > 0 else 0,
        "top100Holdings": top100,
        "top100Percentage": (top100 / total_xrp * 100) if total_xrp > 0 else 0,
        "meanBalance": mean_balance,
        "medianBalance": median_balance,
        "giniCoefficient": float(gini)
    }


def save_today_data(data):
    """Save today's data to CSV"""
    today = datetime.now().strftime("%Y-%m-%d")
    csv_path = CSV_DIR / f"xrp_top_10000_balances_{today}.csv"
    
    df = pd.DataFrame(data)
    df = df.rename(columns={
        "account": "Account",
        "balance": "Balance_XRP", 
        "name": "Name",
        "domain": "Domain",
        "rank": "Rank",
        "isWhale": "Is_Whale"
    })
    df.to_csv(csv_path, index=False)
    print(f"Saved data to {csv_path}")


@app.route('/api/data', methods=['GET'])
def get_data():
    """Main API endpoint - returns all dashboard data"""
    # Fetch current data
    raw_data = fetch_current_data()
    if not raw_data:
        return jsonify({"error": "Failed to fetch data"}), 500
    
    # Process current data
    current_data = process_data(raw_data)
    
    # Save today's data
    save_today_data(current_data)
    
    # Load previous data for comparison
    previous_data = load_previous_data(days_back=1)
    
    # Compare and get changes
    changes, summary = compare_data(current_data, previous_data)
    
    # Detect alerts
    alerts = detect_alerts(changes)
    
    # Calculate stats
    stats = calculate_stats(current_data)
    
    return jsonify({
        "accounts": current_data,
        "changes": changes,
        "summary": summary,
        "alerts": alerts,
        "stats": stats,
        "timestamp": datetime.now().isoformat(),
        "hasPreviousData": previous_data is not None
    })


@app.route('/api/stats', methods=['GET'])
def get_stats():
    """Get just statistics"""
    raw_data = fetch_current_data()
    if not raw_data:
        return jsonify({"error": "Failed to fetch data"}), 500
    
    current_data = process_data(raw_data)
    stats = calculate_stats(current_data)
    
    return jsonify(stats)


@app.route('/api/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({"status": "ok", "timestamp": datetime.now().isoformat()})


if __name__ == '__main__':
    print("Starting XRP Rich List API Server...")
    print("Dashboard should connect to: http://localhost:5000/api/data")
    app.run(host='0.0.0.0', port=5000, debug=True)
