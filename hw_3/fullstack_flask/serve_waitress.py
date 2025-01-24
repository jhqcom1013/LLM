from waitress import serve
from src.app import app  # Replace `src.app` with your actual app module

# Serve the application
if __name__ == "__main__":
    serve(app, host="0.0.0.0", port=1234)  # Replace with your desired host and port