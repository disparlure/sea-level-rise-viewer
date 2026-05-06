"""
Flask backend for Sea Level Rise Visualization web app.
"""

import os
from flask import Flask, render_template, request, jsonify, send_file
from dotenv import load_dotenv
import io
import base64
from image_processor import process_sea_level_image, get_sea_level_rise_projections

# Load environment variables
load_dotenv()

app = Flask(__name__)
GOOGLE_API_KEY = os.getenv("GOOGLE_MAPS_API_KEY")

if not GOOGLE_API_KEY:
    print("WARNING: GOOGLE_MAPS_API_KEY not set. Set it in .env file.")


@app.route("/")
def index():
    """Serve the main page."""
    return render_template("index.html")


@app.route("/api/visualize", methods=["POST"])
def visualize():
    """
    Main endpoint: accepts lat/lon and returns images from all 4 cardinal directions.
    """
    try:
        data = request.get_json()
        lat = float(data.get("latitude"))
        lon = float(data.get("longitude"))
        year = int(data.get("year", 2050))
        
        # Validate coordinates
        if not (-90 <= lat <= 90 and -180 <= lon <= 180):
            return jsonify({"error": "Invalid latitude/longitude"}), 400
        
        if not GOOGLE_API_KEY:
            return jsonify({"error": "Google Maps API key not configured"}), 500
        
        # Process images from all 4 cardinal directions
        headings = [0, 90, 180, 270]  # North, East, South, West
        images_data = []
        
        for heading in headings:
            image_bytes, metadata = process_sea_level_image(lat, lon, GOOGLE_API_KEY, year, heading)
            image_base64 = base64.b64encode(image_bytes).decode()
            images_data.append({
                "heading": heading,
                "heading_name": metadata["heading_name"],
                "image": f"data:image/jpeg;base64,{image_base64}",
                "metadata": metadata
            })
        
        return jsonify({
            "success": True,
            "images": images_data,
            "location": {
                "latitude": lat,
                "longitude": lon,
                "elevation_m": images_data[0]["metadata"]["elevation_m"],
                "sea_level_rise_m": images_data[0]["metadata"]["sea_level_rise_m"],
                "max_high_tide_m": images_data[0]["metadata"]["max_high_tide_m"],
                "total_sea_level_rise_m": images_data[0]["metadata"]["total_sea_level_rise_m"],
                "tide_height_over_elevation_m": images_data[0]["metadata"]["tide_height_over_elevation_m"],
                "projection_year": year,
                "tide_station": images_data[0]["metadata"]["tide_station"]
            }
        })
    
    except ValueError as e:
        return jsonify({"error": f"Invalid input: {str(e)}"}), 400
    except Exception as e:
        print(f"Error: {str(e)}")
        return jsonify({"error": f"Server error: {str(e)}"}), 500


@app.route("/api/projections", methods=["GET"])
def projections():
    """
    Get sea level rise projection data without generating image.
    Useful for quick data queries.
    """
    try:
        lat = request.args.get("lat", type=float)
        lon = request.args.get("lon", type=float)
        
        if lat is None or lon is None:
            return jsonify({"error": "latitude and longitude required"}), 400
        
        if not (-90 <= lat <= 90 and -180 <= lon <= 180):
            return jsonify({"error": "Invalid latitude/longitude"}), 400
        
        data = get_sea_level_rise_projections(lat, lon)
        return jsonify(data)
    
    except Exception as e:
        print(f"Error: {str(e)}")
        return jsonify({"error": str(e)}), 500


@app.route("/api/examples", methods=["GET"])
def examples():
    """Return example locations in Northeast US and Canada."""
    examples = [
        {"name": "Boston, Massachusetts", "lat": 42.3601, "lon": -71.0589},
        {"name": "Portland, Maine", "lat": 43.656954, "lon": -70.25099},
        {"name": "New York City, New York", "lat": 40.7128, "lon": -74.0060},
        {"name": "Baltimore, Maryland", "lat": 39.2904, "lon": -76.6122}
    ]
    return jsonify(examples)


if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5000)
