"""
app.py - Thin wrapper to expose the NHL app as the default entrypoint.
This ensures deployments that reference `app:app` will serve the NHL API.
"""

from app_nhl import app  # re-export the NHL Flask app

if __name__ == '__main__':
    import os
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=False)