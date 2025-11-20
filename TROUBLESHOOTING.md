# Troubleshooting Guide

## "Too many packets in payload" Errors

If you're seeing `ValueError: Too many packets in payload` errors in the terminal, this is a known issue with Chainlit's WebSocket connection handling. The app should still work, but if you experience connection issues, try the following:

### Solution 1: Update Dependencies

```bash
pip install --upgrade chainlit python-socketio engineio
```

### Solution 2: Reinstall with Specific Versions

```bash
pip uninstall chainlit python-socketio engineio
pip install chainlit>=1.0.200 python-socketio[asyncio]>=5.10.0,<6.0.0 engineio>=4.9.0,<5.0.0
```

### Solution 3: Clear Browser Cache

Sometimes the browser cache can cause connection issues:
1. Clear your browser cache
2. Try an incognito/private window
3. Try a different browser

### Solution 4: Restart the Server

If errors persist:
1. Stop the server (Ctrl+C)
2. Wait a few seconds
3. Restart: `chainlit run app.py --port 8001`

### Note

These errors are often non-critical and don't prevent the app from functioning. If the app loads in your browser and you can interact with it, you can safely ignore these terminal errors.

