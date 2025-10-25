import re
from datetime import datetime
import pandas as pd
from scipy.interpolate import CubicSpline
from sklearn.cluster import DBSCAN
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import os
import glob
import json

# Optional imports with fallbacks
try:
    import simplekml
    SIMPLEKML_AVAILABLE = True
except ImportError:
    SIMPLEKML_AVAILABLE = False
    print("Warning: simplekml not available. KML generation will use basic XML output.")

try:
    import cartopy.crs as ccrs
    CARTOPY_AVAILABLE = True
except ImportError:
    CARTOPY_AVAILABLE = False
    print("Warning: cartopy not available. Video will use basic matplotlib plotting.")

try:
    import ffmpeg
    FFMPEG_AVAILABLE = True
except ImportError:
    FFMPEG_AVAILABLE = False

def parse_srt_gps(srt_file):
    """
    Parse SRT file to extract GPS coordinates and timestamps.
    Updated regex patterns to match the actual DJI SRT format.
    """
    gps_data = []

    with open(srt_file, 'r', encoding='utf-8') as f:
        content = f.read()

    # Extract GPS coordinates: [latitude: XX.XXXXXX] [longitude: XX.XXXXXX] [rel_alt: XX.XXX abs_alt: XX.XXX]
    gps_pattern = r'\[latitude:\s*(-?\d+\.\d+)\]\s*\[longitude:\s*(-?\d+\.\d+)\]\s*\[rel_alt:\s*(-?\d+\.\d+)\s*abs_alt:\s*(-?\d+\.\d+)\]'
    gps_matches = re.findall(gps_pattern, content)

    # Extract timestamps: 2025-01-19 07:05:23.115
    timestamp_pattern = r'(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}\.\d{3})'
    timestamps = re.findall(timestamp_pattern, content)

    if len(gps_matches) != len(timestamps):
        print(f"Warning: GPS matches ({len(gps_matches)}) != timestamps ({len(timestamps)})")

    for i, (lat, lon, rel_alt, abs_alt) in enumerate(gps_matches):
        if i < len(timestamps):
            ts_str = timestamps[i]
            try:
                ts = datetime.strptime(ts_str, '%Y-%m-%d %H:%M:%S.%f')
                gps_data.append({
                    'timestamp': ts,
                    'lat': float(lat),
                    'lon': float(lon),
                    'rel_alt': float(rel_alt),
                    'abs_alt': float(abs_alt)
                })
            except ValueError as e:
                print(f"Error parsing timestamp {ts_str}: {e}")
                continue

    if not gps_data:
        raise ValueError("No GPS data found in SRT file")

    df = pd.DataFrame(gps_data).sort_values('timestamp').reset_index(drop=True)
    return df

def infer_route(df, time_window=5):
    """
    Infer the route via triangulation and interpolation.
    Uses clustering to group proximate points and cubic splines for smoothing.
    """
    df = df.copy()
    df['time_sec'] = (df['timestamp'] - df['timestamp'].min()).dt.total_seconds()

    # Use DBSCAN clustering on time dimension
    clusters = DBSCAN(eps=time_window, min_samples=2).fit(df[['time_sec']])
    df['cluster'] = clusters.labels_

    route_points = []

    for cluster in set(df['cluster']):
        if cluster == -1:  # Noise points, skip
            continue

        cluster_df = df[df['cluster'] == cluster].sort_values('time_sec')

        if len(cluster_df) < 2:
            continue

        times = cluster_df['time_sec'].values
        lats = cluster_df['lat'].values
        lons = cluster_df['lon'].values
        alts = cluster_df['abs_alt'].values  # Use absolute altitude

        # Create smooth interpolation with more points
        num_points = max(50, len(cluster_df) * 5)
        interp_times = np.linspace(times.min(), times.max(), num_points)

        if len(times) >= 3:  # Need at least 3 points for cubic spline
            lat_interp = CubicSpline(times, lats)(interp_times)
            lon_interp = CubicSpline(times, lons)(interp_times)
            alt_interp = CubicSpline(times, alts)(interp_times)
        else:
            # Linear interpolation for small clusters
            lat_interp = np.interp(interp_times, times, lats)
            lon_interp = np.interp(interp_times, times, lons)
            alt_interp = np.interp(interp_times, times, alts)

        # Create route points
        for t, lat, lon, alt in zip(interp_times, lat_interp, lon_interp, alt_interp):
            timestamp = df['timestamp'].min() + pd.Timedelta(seconds=t)
            route_points.append({
                'timestamp': timestamp,
                'lat': lat,
                'lon': lon,
                'alt': alt
            })

    if not route_points:
        print("Warning: No route points generated, using original data")
        return df[['timestamp', 'lat', 'lon', 'abs_alt']].rename(columns={'abs_alt': 'alt'})

    route_df = pd.DataFrame(route_points).sort_values('timestamp').reset_index(drop=True)
    return route_df

def generate_kml(route_df, output_kml='flight_path.kml'):
    """
    Generate KML file with the flight path as a LineString.
    """
    if SIMPLEKML_AVAILABLE:
        kml = simplekml.Kml()
        ls = kml.newlinestring(name='Drone Flight Path', description='Inferred drone route')

        # Set altitude mode to absolute (using GPS altitude)
        ls.altitudemode = simplekml.AltitudeMode.absolute

        # Create coordinates list (longitude, latitude, altitude)
        coords = [(row['lon'], row['lat'], row['alt']) for _, row in route_df.iterrows()]
        ls.coords = coords

        # Style the line
        ls.style.linestyle.color = 'ff0000ff'  # Red color
        ls.style.linestyle.width = 3

        kml.save(output_kml)
    else:
        # Fallback: Generate basic KML XML
        coords_str = ' '.join([f"{row['lon']},{row['lat']},{row['alt']}" for _, row in route_df.iterrows()])

        kml_content = f'''<?xml version="1.0" encoding="UTF-8"?>
<kml xmlns="http://www.opengis.net/kml/2.2">
  <Document>
    <name>Drone Flight Path</name>
    <description>Inferred drone route</description>
    <Placemark>
      <name>Flight Path</name>
      <description>Drone flight path</description>
      <LineString>
        <altitudeMode>absolute</altitudeMode>
        <coordinates>{coords_str}</coordinates>
      </LineString>
    </Placemark>
  </Document>
</kml>'''

        with open(output_kml, 'w', encoding='utf-8') as f:
            f.write(kml_content)

    print(f"KML file saved to: {output_kml}")
    return output_kml

def render_flyover_video(route_df, output_mp4='flight_sim.mp4'):
    """
    Create an animated flyover video of the flight path.
    """
    fig, ax = plt.subplots(1, 1, figsize=(10, 8))

    if CARTOPY_AVAILABLE:
        ax = plt.subplot(111, projection=ccrs.PlateCarree())
        # Add coastlines and grid
        ax.coastlines(resolution='10m', color='black', linewidth=0.5)
        ax.gridlines(draw_labels=True, alpha=0.3)
    else:
        # Basic matplotlib plot
        ax.set_xlabel('Longitude')
        ax.set_ylabel('Latitude')
        ax.set_title('Drone Flight Path Animation')
        ax.grid(True, alpha=0.3)

    # Set initial extent
    margin = 0.01
    ax.set_xlim(route_df['lon'].min() - margin, route_df['lon'].max() + margin)
    ax.set_ylim(route_df['lat'].min() - margin, route_df['lat'].max() + margin)

    # Create the flight path line
    line, = ax.plot([], [], 'r-', linewidth=2, alpha=0.8)
    point, = ax.plot([], [], 'bo', markersize=6, alpha=0.9)  # Current position marker

    def update(frame):
        # Update the path up to current frame
        subset = route_df.iloc[:frame+1]
        line.set_data(subset['lon'], subset['lat'])

        # Update current position
        current = route_df.iloc[frame]
        point.set_data([current['lon']], [current['lat']])

        return line, point

    # Create animation
    frames = len(route_df)
    interval = max(50, int(5000 / frames))  # Adjust speed based on data points

    anim = FuncAnimation(fig, update, frames=frames, interval=interval, blit=True)

    # Save animation
    try:
        anim.save(output_mp4, writer='ffmpeg', fps=30, dpi=100)
        print(f"Video saved to: {output_mp4}")
    except Exception as e:
        print(f"Error saving video with ffmpeg: {e}")
        print("Note: Make sure ffmpeg is installed and available in PATH")
        # Fallback: save as gif if ffmpeg not available
        try:
            anim.save(output_mp4.replace('.mp4', '.gif'), writer='pillow', fps=10)
            print(f"GIF saved to: {output_mp4.replace('.mp4', '.gif')}")
        except Exception as e2:
            print(f"Could not save animation: {e2}")
            print("Consider installing ffmpeg-python or pillow for animation saving")

    plt.close(fig)
    return output_mp4

def identify_flights(all_gps_data, time_gap_threshold=300, distance_threshold=1000):
    """
    Identify separate flights from GPS data based on time gaps and distance.

    Parameters:
    - time_gap_threshold: seconds between consecutive points to consider as separate flights
    - distance_threshold: meters between consecutive points to consider as separate flights
    """
    if all_gps_data.empty:
        return []

    # Sort by timestamp
    all_gps_data = all_gps_data.sort_values('timestamp').reset_index(drop=True)

    # Calculate time differences in seconds
    time_diffs = all_gps_data['timestamp'].diff().dt.total_seconds()

    # Calculate distances between consecutive points (Haversine distance in meters)
    def haversine_distance(lat1, lon1, lat2, lon2):
        R = 6371000  # Earth's radius in meters
        phi1, phi2 = np.radians([lat1, lat2])
        delta_phi = np.radians(lat2 - lat1)
        delta_lambda = np.radians(lon2 - lon1)
        a = np.sin(delta_phi/2)**2 + np.cos(phi1) * np.cos(phi2) * np.sin(delta_lambda/2)**2
        c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))
        return R * c

    distances = []
    for i in range(1, len(all_gps_data)):
        lat1, lon1 = all_gps_data.iloc[i-1]['lat'], all_gps_data.iloc[i-1]['lon']
        lat2, lon2 = all_gps_data.iloc[i]['lat'], all_gps_data.iloc[i]['lon']
        dist = haversine_distance(lat1, lon1, lat2, lon2)
        distances.append(dist)
    distances.insert(0, 0)  # First point has no previous distance

    # Identify flight breaks: large time gaps OR large distance gaps
    flight_breaks = []
    for i in range(1, len(all_gps_data)):
        time_gap = time_diffs.iloc[i]
        dist_gap = distances[i]

        if (time_gap > time_gap_threshold) or (dist_gap > distance_threshold):
            flight_breaks.append(i)

    # Split into flights
    flights = []
    start_idx = 0

    for break_idx in flight_breaks + [len(all_gps_data)]:
        flight_data = all_gps_data.iloc[start_idx:break_idx].copy()
        if len(flight_data) > 0:
            flight_data['file'] = flight_data['file']  # Keep track of source file
            flights.append(flight_data)
        start_idx = break_idx

    print(f"Identified {len(flights)} separate flights from {len(all_gps_data)} total GPS points")
    return flights

def process_all_srt_files(srt_directory, output_dir=None):
    """
    Process all SRT files and generate individual flight KMLs and JSON data.
    """
    if output_dir is None:
        output_dir = srt_directory

    # Create flights directory
    flights_dir = os.path.join(output_dir, "flights")
    os.makedirs(flights_dir, exist_ok=True)

    print(f"Processing all SRT files in: {srt_directory}")

    # Step 1: Parse all SRT files
    print("Step 1: Parsing all SRT files...")
    all_gps_data = []
    srt_files = glob.glob(os.path.join(srt_directory, "*.SRT"))

    for srt_file in sorted(srt_files):
        try:
            print(f"  Processing {os.path.basename(srt_file)}...")
            gps_df = parse_srt_gps(srt_file)
            gps_df['file'] = os.path.basename(srt_file)
            all_gps_data.append(gps_df)
        except Exception as e:
            print(f"  Error processing {srt_file}: {e}")
            continue

    if not all_gps_data:
        print("No GPS data found!")
        return None, None

    # Combine all GPS data
    combined_gps = pd.concat(all_gps_data, ignore_index=True)
    print(f"Total GPS points extracted: {len(combined_gps)}")

    # Step 2: Identify flights
    print("Step 2: Identifying separate flights...")
    flights = identify_flights(combined_gps)

    # Step 3: Generate KML and JSON for each flight
    print("Step 3: Generating KMLs and JSON data...")
    flights_data = []

    for i, flight_df in enumerate(flights):
        flight_id = f"flight_{i+1:03d}"
        print(f"  Processing {flight_id} ({len(flight_df)} points)...")

        # Infer route for this flight
        route_df = infer_route(flight_df)

        # Generate KML
        kml_file = os.path.join(flights_dir, f"{flight_id}.kml")
        generate_kml(route_df, kml_file)

        # Collect flight data for JSON
        flight_info = {
            'flight_id': flight_id,
            'start_time': flight_df['timestamp'].min().isoformat(),
            'end_time': flight_df['timestamp'].max().isoformat(),
            'duration_seconds': (flight_df['timestamp'].max() - flight_df['timestamp'].min()).total_seconds(),
            'gps_points_count': len(flight_df),
            'route_points_count': len(route_df),
            'start_location': {
                'lat': float(flight_df.iloc[0]['lat']),
                'lon': float(flight_df.iloc[0]['lon']),
                'alt': float(flight_df.iloc[0]['abs_alt'])
            },
            'end_location': {
                'lat': float(flight_df.iloc[-1]['lat']),
                'lon': float(flight_df.iloc[-1]['lon']),
                'alt': float(flight_df.iloc[-1]['abs_alt'])
            },
            'source_files': flight_df['file'].unique().tolist(),
            'gps_points': [
                {
                    'timestamp': row['timestamp'].isoformat(),
                    'lat': float(row['lat']),
                    'lon': float(row['lon']),
                    'rel_alt': float(row['rel_alt']),
                    'abs_alt': float(row['abs_alt']),
                    'file': row['file']
                } for _, row in flight_df.iterrows()
            ],
            'kml_file': os.path.basename(kml_file)
        }
        flights_data.append(flight_info)

    # Step 4: Save JSON file
    json_file = os.path.join(output_dir, "all_flights.json")
    with open(json_file, 'w', encoding='utf-8') as f:
        json.dump({
            'total_flights': len(flights_data),
            'total_gps_points': len(combined_gps),
            'generated_at': pd.Timestamp.now().isoformat(),
            'flights': flights_data
        }, f, indent=2, ensure_ascii=False)

    print("\nProcessing complete!")
    print(f"Total flights identified: {len(flights_data)}")
    print(f"KML files created in: {flights_dir}")
    print(f"JSON file: {json_file}")

    return flights_dir, json_file

def process_drone_srt(srt_file, output_dir=None):
    """
    Complete workflow to process SRT file and generate KML and video.
    """
    if output_dir is None:
        output_dir = os.path.dirname(srt_file)

    base_name = os.path.splitext(os.path.basename(srt_file))[0]

    print(f"Processing SRT file: {srt_file}")

    try:
        # Step 1: Parse GPS data
        print("Step 1: Parsing GPS data...")
        gps_df = parse_srt_gps(srt_file)
        print(f"Extracted {len(gps_df)} GPS points")

        # Step 2: Infer route
        print("Step 2: Inferring route...")
        route_df = infer_route(gps_df)
        print(f"Generated {len(route_df)} route points")

        # Step 3: Generate KML
        print("Step 3: Generating KML file...")
        kml_file = os.path.join(output_dir, f"{base_name}_flight_path.kml")
        generate_kml(route_df, kml_file)

        # Step 4: Render video
        print("Step 4: Rendering flight simulation video...")
        video_file = os.path.join(output_dir, f"{base_name}_flight_sim.mp4")
        render_flyover_video(route_df, video_file)

        print("\nProcessing complete!")
        print(f"KML file: {kml_file}")
        print(f"Video file: {video_file}")

        return kml_file, video_file

    except Exception as e:
        print(f"Error processing SRT file: {e}")
        import traceback
        traceback.print_exc()
        return None, None

if __name__ == "__main__":
    # Process all SRT files and generate flight KMLs and JSON
    srt_directory = "/Users/imac/Drone Cursor/Routes"
    process_all_srt_files(srt_directory)
