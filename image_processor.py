"""
Image processing module for visualizing sea level rise projections.
"""

import io
from PIL import Image
import requests
from typing import Tuple, Dict, Any
import math
from StreetViewElevationPipeline import StreetViewElevationPipeline, draw_elevation_line_from_buffer

# We'll use OpenCV's projectPoints facility for the 3D projection.  This
# matches the example the user provided, defining world points and projecting
# them through a simple camera matrix.  Our camera will be positioned 2m above
# ground and looking forward along +Z with a horizontal FOV of 90°.


def get_street_view_image(
    lat: float, 
    lon: float, 
    api_key: str,
    size: str = "600x400",
    fov: int = 90,
    heading: int = 0
) -> bytes:
    """
    Fetch a Street View image from Google Maps API.
    
    Args:
        lat: Latitude
        lon: Longitude
        api_key: Google Maps API key
        size: Image size in format "widthxheight" (default: 600x400)
        fov: Field of view in degrees (default: 90)
        heading: Camera direction in degrees (0=N, 90=E, 180=S, 270=W)
    
    Returns:
        Image bytes
    """
    url = "https://maps.googleapis.com/maps/api/streetview"
    params = {
        "location": f"{lat},{lon}",
        "size": size,
        "fov": fov,
        "heading": heading,
        "key": api_key
    }
    
    response = requests.get(url, params=params)
    response.raise_for_status()
    return response.content


def get_elevation(
    lat: float, 
    lon: float, 
    api_key: str
) -> float:
    """
    Get elevation at a location using Google Maps Elevation API.
    
    Args:
        lat: Latitude
        lon: Longitude
        api_key: Google Maps API key
    
    Returns:
        Elevation in meters
    """
    url = "https://maps.googleapis.com/maps/api/elevation/json"
    params = {
        "locations": f"{lat},{lon}",
        "key": api_key
    }
    
    response = requests.get(url, params=params)
    response.raise_for_status()
    data = response.json()
    
    if data["results"]:
        return data["results"][0]["elevation"]
    raise ValueError(f"Could not find elevation for {lat}, {lon}")


def get_sea_level_rise_projections(
    lat: float, 
    lon: float
) -> Dict[str, Any]:
    """
    Fetch sea level rise projections from NOAA API.
    
    Args:
        lat: Latitude
        lon: Longitude
    
    Returns:
        Dictionary with projection data
    """
    url = "https://api.tidesandcurrents.noaa.gov/dpapi/prod/webapi/product/slr_projections.json"
    params = {
        "lat": lat,
        "lon": lon
    }
    
    response = requests.get(url, params=params)
    response.raise_for_status()
    return response.json()


def get_closest_tide_station(lat: float, lon: float) -> Dict[str, Any]:
    """
    Find the closest NOAA tide station to the given coordinates.
    
    Args:
        lat: Latitude
        lon: Longitude
    
    Returns:
        Dictionary with station information
    """
    url = "https://api.tidesandcurrents.noaa.gov/mdapi/prod/webapi/stations.json"
    params = {
        "type": "tidepredictions",
        "units": "english"
    }
    
    response = requests.get(url, params=params)
    response.raise_for_status()
    data = response.json()
    
    stations = data.get("stations", [])
    if not stations:
        raise ValueError("No tide stations found")
    
    # Calculate distance to each station and find the closest
    closest_station = None
    min_distance = float('inf')
    
    for station in stations:
        station_lat = float(station.get("lat", 0))
        station_lon = float(station.get("lng", 0))
        
        # Haversine distance calculation
        dlat = math.radians(station_lat - lat)
        dlon = math.radians(station_lon - lon)
        a = math.sin(dlat/2)**2 + math.cos(math.radians(lat)) * math.cos(math.radians(station_lat)) * math.sin(dlon/2)**2
        c = 2 * math.atan2(math.sqrt(a), math.sqrt(1-a))
        distance = 6371 * c  # Earth radius in km
        
        if distance < min_distance:
            min_distance = distance
            closest_station = station
    
    if not closest_station:
        raise ValueError("Could not find closest tide station")
    
    return {
        "id": closest_station.get("id"),
        "name": closest_station.get("name"),
        "lat": float(closest_station.get("lat", 0)),
        "lon": float(closest_station.get("lng", 0)),
        "distance_km": min_distance
    }


def get_max_high_tide(station_id: str, days: int = 365) -> float:
    """
    Get the maximum high tide level for a station over the specified number of days.
    
    Args:
        station_id: NOAA station ID
        days: Number of days to look back (default: 365)
    
    Returns:
        Maximum tide level in feet
    """
    from datetime import datetime, timedelta
    
    # Get data for the past 'days' days
    end_date = datetime.now()
    start_date = end_date - timedelta(days=days)
    
    url = "https://api.tidesandcurrents.noaa.gov/api/prod/datagetter"
    params = {
        "product": "predictions",
        "application": "NOS.COOPS.TAC.WL",
        "begin_date": start_date.strftime("%Y%m%d"),
        "end_date": end_date.strftime("%Y%m%d"),
        "datum": "MLLW",
        "station": station_id,
        "time_zone": "lst_ldt",
        "units": "english",
        "interval": "h",
        "format": "json"
    }
    
    response = requests.get(url, params=params)
    response.raise_for_status()
    data = response.json()
    
    predictions = data.get("predictions", [])
    if not predictions:
        raise ValueError(f"No tide predictions found for station {station_id}")
    
    # Find the maximum tide level
    max_tide = max(float(pred.get("v", 0)) for pred in predictions)
    return max_tide


def process_sea_level_image(
    lat: float,
    lon: float,
    api_key: str,
    year: int = 2050,
    heading: int = 0
) -> Tuple[bytes, Dict[str, Any]]:
    """
    Complete pipeline: fetch Street View, elevation, sea level data,
    and draw projection on image.
    
    Args:
        lat: Latitude
        lon: Longitude
        api_key: Google Maps API key
        year: Year for sea level projection (default: 2050)
        heading: Camera direction (0=N, 90=E, 180=S, 270=W, default: 0)
    
    Returns:
        Tuple of (image_bytes, metadata_dict)
    """
    # Fetch all required data
    heading_name = {0: "North", 90: "East", 180: "South", 270: "West"}.get(heading, f"{heading}°")
    print(f"Fetching Street View image ({heading_name}) for {lat}, {lon}...")
    street_view_img = get_street_view_image(lat, lon, api_key, heading=heading)
    
    print("Fetching elevation data...")
    elevation = get_elevation(lat, lon, api_key)
    
    print("Fetching sea level rise projections...")
    projections = get_sea_level_rise_projections(lat, lon)
    
    print("Finding closest tide station...")
    closest_station = get_closest_tide_station(lat, lon)
    station_id = closest_station["id"]
    
    print(f"Fetching maximum high tide data for station {station_id}...")
    max_high_tide_ft = get_max_high_tide(station_id)
    # Convert feet to meters for consistency
    max_high_tide_m = max_high_tide_ft * 0.3048
    
    # Extract sea level rise data for the specified year
    # NOAA returns an array of projections with different scenarios
    projections_list = projections.get("SlrProjections", [])
    
    # Find projection for the requested year (use "Intermediate" scenario)
    year_slr = 0.0
    if projections_list:
        # Filter for "Intermediate" scenario (middle ground between Low and High)
        intermediate = [p for p in projections_list if p.get("scenario") == "Intermediate"]
        if not intermediate:
            # Fall back to all scenarios if Intermediate not available
            intermediate = projections_list
        
        # Find closest projection year
        closest_projection = min(
            intermediate,
            key=lambda p: abs(int(p.get("projectionYear", 0)) - year),
            default=None
        )
        
        if closest_projection:
            year_slr = float(closest_projection.get("projectionRsl", 0.0))
    
    # Convert sea level rise to meters and add maximum high tide
    year_slr_m = year_slr / 100.0
    total_sea_level_rise_m = year_slr_m + max_high_tide_m
    
    #print(f"Elevation: {elevation}m, Sea level rise ({year}): {year_slr}cm, Max high tide: {max_high_tide_ft}ft ({max_high_tide_m:.2f}m), Total rise: {total_sea_level_rise_m:.2f}m")
    
    # Calculate where to draw the line.  We'll pass the height in meters
    # directly to the drawing function which will perform a simple 3D
    # projection using our pinhole camera helper.
    img = Image.open(io.BytesIO(street_view_img))
    image_width, image_height = img.size
    total_sea_level_rise_m  # already in meters

    
    # draw with projection enabled; this returns an image only so we need to
    # recompute pixel y (approximate) for the metadata
    # build a human-readable label indicating height above elevation
    # avoid non‑Latin1 characters (JPEG comment limitation) by using '~' instead of '≈'
    label_text = f"Water height ~ {total_sea_level_rise_m * 100:.2f} m above elevation"

    street_view_img = draw_elevation_line_from_buffer(
        street_view_img,
        total_sea_level_rise_m - elevation)
    
    # determine a representative Y position for the line for metadata
    # if projection is used we can project again and take the average
    try:
        print("Drawing sea level projection...")
        pipeline = StreetViewElevationPipeline(street_view_img)
        street_view_img = pipeline.draw_elevation_line(total_sea_level_rise_m - elevation)


    except Exception as e:
        print(f"Error occurred while drawing sea level projection: {e}")

    # Compile metadata
    metadata = {
        "latitude": lat,
        "longitude": lon,
        "elevation_m": elevation,
        "sea_level_rise_m": year_slr / 100.0,
        "max_high_tide_m": max_high_tide_m,
        "total_sea_level_rise_m": total_sea_level_rise_m,
        "tide_height_over_elevation_m": total_sea_level_rise_m - elevation,
        "projection_year": year,
        "heading": heading,
        "heading_name": heading_name,
        "image_height": image_height,
        "image_width": image_width,
        "tide_station": closest_station
    }
    
    return street_view_img, metadata
