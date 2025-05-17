import json
import numpy as np
import pandas as pd
from PIL import Image
from math import radians, sin, cos, atan2, sqrt
from collections import deque
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d

# -----------------------------
# UTILITIES
# -----------------------------

def latlon_distance_haversine(lat1, lon1, lat2, lon2):
    """
    Compute the great-circle distance (in meters) between two lat-lon points
    using the Haversine formula.
    """
    R = 6371000.0  # Earth radius in meters
    phi1, phi2 = radians(lat1), radians(lat2)
    dphi = radians(lat2 - lat1)
    dlambda = radians(lon2 - lon1)

    a = sin(dphi/2)**2 + cos(phi1)*cos(phi2)*sin(dlambda/2)**2
    c = 2 * atan2(np.sqrt(a), np.sqrt(1 - a))
    distance = R * c
    return distance

def pixel_to_latlon(r, c, height, width, min_lat, max_lat, min_lon, max_lon):
    """
    Convert (row, col) in the image to (lat, lon) using the bounding box.
    Fixes the vertical flip issue.
    """
    lat = max_lat - (r / (height - 1.0)) * (max_lat - min_lat)
    lon = min_lon + (c / (width - 1.0)) * (max_lon - min_lon)
    return lat, lon

def find_connected_components(trajectory_mask, connectivity='8'):
    """
    Find all connected components in the boolean trajectory_mask.
    Returns a list of components, each component is a set of (row, col) pixels.
    """
    height, width = trajectory_mask.shape
    
    if connectivity == '8':
        neighbors = [(-1, 0), (1, 0), (0, -1), (0, 1),
                     (-1, -1), (-1, 1), (1, -1), (1, 1)]
    else:
        neighbors = [(-1, 0), (1, 0), (0, -1), (0, 1)]
    
    visited = np.zeros_like(trajectory_mask, dtype=bool)
    components = []
    
    for r in range(height):
        for c in range(width):
            if trajectory_mask[r, c] and not visited[r, c]:
                component_pixels = set()
                queue = deque([(r, c)])
                visited[r, c] = True

                while queue:
                    rr, cc = queue.popleft()
                    component_pixels.add((rr, cc))
                    for dr, dc in neighbors:
                        nr, nc = rr + dr, cc + dc
                        if 0 <= nr < height and 0 <= nc < width:
                            if trajectory_mask[nr, nc] and not visited[nr, nc]:
                                visited[nr, nc] = True
                                queue.append((nr, nc))
                components.append(component_pixels)
    return components

def get_component_boundary_pixels(component, trajectory_mask, connectivity='8'):
    """
    Extract boundary pixels of a given component.
    A pixel is on the boundary if at least one neighbor is outside the component.
    """
    height, width = trajectory_mask.shape
    if connectivity == '8':
        neighbors = [(-1, 0), (1, 0), (0, -1), (0, 1),
                     (-1, -1), (-1, 1), (1, -1), (1, 1)]
    else:
        neighbors = [(-1, 0), (1, 0), (0, -1), (0, 1)]
    
    boundary_pixels = []
    for (r, c) in component:
        for dr, dc in neighbors:
            nr, nc = r + dr, c + dc
            if not (0 <= nr < height and 0 <= nc < width) or ((nr, nc) not in component):
                boundary_pixels.append((r, c))
                break
    # Remove duplicates
    return list(set(boundary_pixels))

def filter_close_points(points, min_dist=5):
    """
    Filter out points that are too close to each other in (row, col) space.
    """
    points_sorted = sorted(points)
    filtered = []
    for p in points_sorted:
        if any(((p[0]-q[0])**2 + (p[1]-q[1])**2) < (min_dist**2) for q in filtered):
            continue
        filtered.append(p)
    return filtered

def compute_local_curvature(trajectory_mask, p, connectivity='8'):
    """
    Compute a naive local "curvature" at pixel p based on the number of valid neighbors.
    Fewer neighbors yields higher curvature (i.e. a sharper corner).
    """
    if connectivity == '8':
        neighbors8 = [(-1, 0), (1, 0), (0, -1), (0, 1),
                      (-1, -1), (-1, 1), (1, -1), (1, 1)]
    else:
        neighbors8 = [(-1, 0), (1, 0), (0, -1), (0, 1)]
    
    r, c = p
    valid_neighbors = []
    for dr, dc in neighbors8:
        nr, nc = r + dr, c + dc
        if 0 <= nr < trajectory_mask.shape[0] and 0 <= nc < trajectory_mask.shape[1]:
            if trajectory_mask[nr, nc]:
                valid_neighbors.append((nr, nc))
    curvature = len(neighbors8) - len(valid_neighbors)
    return curvature

def bfs_path(trajectory_mask, start, goal, connectivity='8'):
    """
    Perform a BFS to find a path from start to goal within the trajectory_mask.
    Returns the path (a list of pixel coordinates) or None if not found.
    """
    height, width = trajectory_mask.shape
    if connectivity == '8':
        neighbors = [(-1, 0), (1, 0), (0, -1), (0, 1),
                     (-1, -1), (-1, 1), (1, -1), (1, 1)]
    else:
        neighbors = [(-1, 0), (1, 0), (0, -1), (0, 1)]
    
    queue = deque([[start]])
    visited = set([start])
    
    while queue:
        path = queue.popleft()
        current = path[-1]
        if current == goal:
            return path
        for dr, dc in neighbors:
            rr, cc = current[0] + dr, current[1] + dc
            if (0 <= rr < height and 0 <= cc < width and
                trajectory_mask[rr, cc] and (rr, cc) not in visited):
                visited.add((rr, cc))
                queue.append(path + [(rr, cc)])
    return None

# -----------------------------
# MAIN RECONSTRUCTION FUNCTION
# -----------------------------

def reconstruct_trajectory_enhanced(
    trajectory_img_path,
    boundary_json_path,
    altitude_min=20.0,
    altitude_max=120.0,
    altitude_distributions=None,  # Should be a dict with keys 'threat' and 'non_threat'
    threat_status=0,              # 1 for threat, 0 for non-threat
    samples_per_point=200,        # Number of samples per x-axis point if using distributions
    connectivity='8',
    close_point_threshold=5
):
    """
    Reconstruct an ordered lat-lon trajectory from a binary trajectory image by:
      1. Finding the largest connected component.
      2. Extracting its boundary (contour) pixels.
      3. Filtering out boundary pixels that are too close.
      4. Estimating local curvature to pick corner-like points.
      5. Running BFS between corner pairs to find the longest path.
      6. Converting that pixel path to lat-lon coordinates.
      7. Assigning altitudes based on provided distributions (or linearly, if None).
      8. Computing the total traveled distance in meters.

    Returns:
        df : pd.DataFrame with columns ['latitude', 'longitude', 'altitude']
        total_distance_m : float (total distance in meters)
    """
    # Load image and boundary JSON
    img = Image.open(trajectory_img_path).convert('L')
    trajectory_array = np.array(img)
    trajectory_mask = (trajectory_array > 128)
    height, width = trajectory_mask.shape

    with open(boundary_json_path, 'r') as f:
        boundary = json.load(f)
    min_lat = boundary["min_lat_adj"]
    max_lat = boundary["max_lat_adj"]
    min_lon = boundary["min_lon_adj"]
    max_lon = boundary["max_lon_adj"]

    # 1. Find largest connected component
    components = find_connected_components(trajectory_mask, connectivity=connectivity)
    if not components:
        raise ValueError("No connected components found in trajectory image!")
    largest_component = max(components, key=len)

    # 2. Extract boundary pixels of the largest component
    boundary_pixels = get_component_boundary_pixels(largest_component, trajectory_mask, connectivity=connectivity)

    # 3. Filter out boundary pixels that are too close
    boundary_pixels_filtered = filter_close_points(boundary_pixels, min_dist=close_point_threshold)
    if len(boundary_pixels_filtered) < 2:
        raise ValueError("Not enough boundary pixels after filtering to find start/end.")

    # 4. Compute local curvature and select corners (threshold=4 here)
    corners = []
    for p in boundary_pixels_filtered:
        curv = compute_local_curvature(trajectory_mask, p, connectivity=connectivity)
        if curv >= 4:
            corners.append(p)
    if len(corners) < 2:
        corners = boundary_pixels_filtered

    # 5. Among corner points, find the pair with the longest path using BFS
    best_path = None
    best_len = -1
    corners_list = list(corners)
    n_corners = len(corners_list)
    for i in range(n_corners):
        for j in range(i + 1, n_corners):
            start = corners_list[i]
            end = corners_list[j]
            path_pixels = bfs_path(trajectory_mask, start, end, connectivity=connectivity)
            if path_pixels is not None:
                path_length = len(path_pixels)
                if path_length > best_len:
                    best_len = path_length
                    best_path = path_pixels

    if best_path is None:
        raise ValueError("No valid path found among corner points! The mask may be too fragmented.")

    # 6. Convert pixel path to lat-lon
    latlon_path = [pixel_to_latlon(r, c, height, width, min_lat, max_lat, min_lon, max_lon)
                   for (r, c) in best_path]

    # 7. Altitude assignment:
    n_points = len(latlon_path)
    if altitude_distributions is not None:
        # Use distribution-based altitude profile
        group = 'threat' if threat_status == 1 else 'non_threat'
        distribution = altitude_distributions.get(group, {})
        mean_alt = distribution.get('mean', None)
        std_alt = distribution.get('std', None)
        x_original = distribution.get('x', None)
        if mean_alt is not None and std_alt is not None and x_original is not None:
            try:
                fixed_x = np.linspace(0, 100, n_points)
                sampled_profile = []
                for i in range(len(x_original)):
                    m = mean_alt[i]
                    s = std_alt[i]
                    samples = np.random.normal(loc=m, scale=s, size=samples_per_point)
                    sampled_profile.append(np.mean(samples))
                sampled_profile = np.array(sampled_profile)
                # Create an interpolation function over the original x values
                interp_func = interp1d(x_original, sampled_profile, kind='linear', fill_value="extrapolate")
                altitudes = interp_func(fixed_x)
                # Ensure altitudes are not below the minimum altitude
                altitudes = np.clip(altitudes, altitude_min, None)
            except Exception as e:
                print(f"Error in altitude interpolation: {e}")
                altitudes = np.linspace(altitude_min, altitude_max, n_points)
        else:
            altitudes = np.linspace(altitude_min, altitude_max, n_points)
    else:
        altitudes = np.linspace(altitude_min, altitude_max, n_points)

    # 8. Compute total distance (in meters) along the trajectory
    total_distance_m = 0.0
    for i in range(1, n_points):
        lat1, lon1 = latlon_path[i - 1]
        lat2, lon2 = latlon_path[i]
        total_distance_m += latlon_distance_haversine(lat1, lon1, lat2, lon2)

    # 9. Prepare the DataFrame to return (do not save to disk)
    df = pd.DataFrame({
        'latitude': [pt[0] for pt in latlon_path],
        'longitude': [pt[1] for pt in latlon_path],
        'altitude': altitudes
    })

    return df, total_distance_m

def reconstruct_all_trajectories(
    trajectory_img_path,
    boundary_json_path,
    altitude_min=20.0,
    altitude_max=120.0,
    altitude_distributions=None,
    threat_status=0,
    samples_per_point=200,
    connectivity='8',
    close_point_threshold=5,
    curvature_threshold=4,
    min_path_length=10
):
    """
    Reconstruct *all* possible trajectories from the main connected component by:
      1. Extracting corner-like boundary points as candidate endpoints.
      2. Running BFS between *every* distinct pair of corners.
      3. Keeping only those paths longer than min_path_length.
      4. Converting each pixel path to lat-lon and assigning altitudes.
      
    Returns:
        trajectories : [
            {
              'df': pd.DataFrame([...]),
              'distance_m': float,
              'start_pixel': (r,c),
              'end_pixel': (r,c)
            },
            ...
        ]
    """
    # --- load mask and metadata ---
    img = Image.open(trajectory_img_path).convert('L')
    mask = np.array(img) > 128
    h, w = mask.shape
    with open(boundary_json_path) as f:
        B = json.load(f)
    min_lat, max_lat = B["min_lat_adj"], B["max_lat_adj"]
    min_lon, max_lon = B["min_lon_adj"], B["max_lon_adj"]

    # --- largest CC and its boundary ---
    comps = find_connected_components(mask, connectivity)
    if not comps:
        raise ValueError("No connected component found")
    main_comp = max(comps, key=len)
    boundary = get_component_boundary_pixels(main_comp, mask, connectivity)
    boundary = filter_close_points(boundary, min_dist=close_point_threshold)

    # --- pick “corner” candidates via curvature ---
    corners = [p for p in boundary
               if compute_local_curvature(mask, p, connectivity) >= curvature_threshold]
    if len(corners) < 2:
        corners = boundary

    # --- for each pair, BFS and collect valid paths ---
    trajectories = []
    for i in range(len(corners)):
        for j in range(i+1, len(corners)):
            start, end = corners[i], corners[j]
            path = bfs_path(mask, start, end, connectivity)
            if path is None or len(path) < min_path_length:
                continue

            # pixel→latlon
            latlon = [pixel_to_latlon(r, c, h, w, min_lat, max_lat, min_lon, max_lon)
                      for (r, c) in path]

            # altitude assignment (fall back to linear if distributions incomplete)
            n = len(latlon)
            if altitude_distributions and threat_status in (0,1):
                grp = 'threat' if threat_status==1 else 'non_threat'
                D = altitude_distributions.get(grp,{})
                try:
                    xs, means, stds = D['x'], D['mean'], D['std']
                    # sample once per profile point
                    sampled = [np.mean(np.random.normal(m,s,samples_per_point))
                               for m,s in zip(means,stds)]
                    interp = interp1d(xs, sampled, kind='linear', fill_value='extrapolate')
                    alts = np.clip(interp(np.linspace(0,100,n)), altitude_min, None)
                except:
                    alts = np.linspace(altitude_min, altitude_max, n)
            else:
                alts = np.linspace(altitude_min, altitude_max, n)

            # distance
            dist = sum(
                latlon_distance_haversine(latlon[k-1][0], latlon[k-1][1],
                                          latlon[k][0],   latlon[k][1])
                for k in range(1,n)
            )

            df = pd.DataFrame({
                'latitude': [pt[0] for pt in latlon],
                'longitude': [pt[1] for pt in latlon],
                'altitude': alts
            })

            trajectories.append({
                'df': df,
                'distance_m': dist,
                'start_pixel': start,
                'end_pixel': end
            })

    if not trajectories:
        raise ValueError("No valid trajectories found")

    return trajectories

# -----------------------------
# MAP PLOTTING FUNCTION (KEEP AS IS)
# -----------------------------
from matplotlib import colormaps, colors
import plotly.graph_objs as go

# (Assume any additional map plotting imports remain unchanged)
mapbox_token = '<MAPBOX_TOKEN>'
tab20 = colormaps['tab20']
tab20 = [colors.to_hex(color) for color in tab20.colors]
start_color = '#de3297'
end_color = 'red'

def plot_inversed_trajectories(data_points, save_path=None, show=True, zoom=17):
    """
    Plot a trajectory on a map using Plotly.
    This function is left unchanged.
    """
    layout = go.Layout(
        autosize=False,
        width=800,
        height=522
    )
    fig = go.Figure(layout=layout)
    df_toplot = data_points.drop_duplicates(keep='first')
    fig.add_trace(
        go.Scattermapbox(
            mode="lines",
            lon=df_toplot.longitude.tolist(),
            lat=df_toplot.latitude.tolist(),
            name="trajectory",
            line=dict(width=3, color=tab20[0])
        )
    )
    fig.add_trace(
        go.Scattermapbox(
            mode="markers",
            lon=[df_toplot.iloc[0]['longitude']],
            lat=[df_toplot.iloc[0]['latitude']],
            name="start_point",
            marker={"size": 24, "color": start_color}
        )
    )
    fig.add_trace(
        go.Scattermapbox(
            mode="markers",
            lon=[df_toplot.iloc[-1]['longitude']],
            lat=[df_toplot.iloc[-1]['latitude']],
            name="end_point",
            marker={"size": 24, "color": end_color}
        )
    )
    # Compute appropriate center and zoom (assuming zoom_center is defined elsewhere)
    # Here we simply set the center to the first point.
    center = {"lat": df_toplot.iloc[0]['latitude'], "lon": df_toplot.iloc[0]['longitude']}
    fig.update_layout(
        margin={"l": 113, "t": 24, "b": 22, "r": 115},
        mapbox_style="open-street-map",
        mapbox=dict(
            center=center,
            pitch=0,
            zoom=zoom,
            accesstoken=mapbox_token,
        )
    )

    if save_path is not None:
        fig.write_html(save_path)
    if show:
        fig.show()

# -----------------------------
# NEW FUNCTION: PLOT ALTITUDE PROFILE
# -----------------------------
def plot_altitude_profile(df, total_distance_m):
    """
    Plot the altitude profile of a reconstructed trajectory with a professional and academic style.
    
    The x-axis represents the step (or sequence) number (displayed with dense ticks), while the y-axis 
    represents the altitude in meters. The total traveled distance is neatly annotated in the top-right corner.
    
    Args:
        df (pd.DataFrame): DataFrame containing an 'altitude' column.
        total_distance_m (float): Total distance traveled along the trajectory in meters.
    """
    import matplotlib.pyplot as plt
    import numpy as np

    # Generate the x-axis values as sequential step numbers.
    steps = np.arange(1, len(df) + 1)
    altitudes = df['altitude']
    
    # Create a figure with a compact layout.
    fig, ax = plt.subplots(figsize=(10, 4))
    
    # Plot the altitude profile using a solid line and circular markers.
    ax.plot(steps, altitudes, marker='o', linestyle='-', color='navy', label='Altitude Profile')
    
    # Set axis labels and a bold title.
    ax.set_xlabel("Step Number", fontsize=12)
    ax.set_ylabel("Altitude (m)", fontsize=12)
    # ax.set_title("Trajectory Altitude Profile", fontsize=14, fontweight='bold')
    
    # Apply a light grid with both major and minor ticks for better readability.
    ax.grid(which='both', linestyle='--', linewidth=0.5, alpha=0.7)
    
    # Adjust the x-axis tick density.
    n = len(steps)
    # Show all ticks if few steps; otherwise choose a tick interval based on total number of steps.
    tick_interval = 1 if n <= 20 else max(1, int(n / 20))
    ax.set_xticks(np.arange(1, n + 1, tick_interval))
    
    # Add a legend in the upper left corner.
    ax.legend(loc='upper left', fontsize=12)
    
    # Annotate the total distance in a small box in the top-right corner.
    ax.annotate(f"Total Distance: {total_distance_m:.2f} m",
                xy=(0.98, 0.98), xycoords='axes fraction',
                horizontalalignment='right', verticalalignment='top',
                fontsize=12,
                bbox=dict(boxstyle="round,pad=0.3", fc="lightgrey", ec="k", alpha=0.75))
    
    plt.tight_layout()
    plt.show()


# -----------------------------
# EXAMPLE USAGE
# -----------------------------
if __name__ == "__main__":
    # Set the paths (update these to your local paths)
    trajectory_img_path = "/path/to/trajectory_image.png"
    boundary_json_path = "/path/to/boundary.json"

    # Optional: Provide altitude distributions dictionary if available.
    # For example:
    # altitude_distributions = {
    #     "threat": {"mean": [30, 40, 50, 60], "std": [5, 5, 5, 5], "x": [0, 33, 66, 100]},
    #     "non_threat": {"mean": [20, 30, 40, 50], "std": [5, 5, 5, 5], "x": [0, 33, 66, 100]}
    # }
    altitude_distributions = None  # Use None to fall back to linear interpolation
    
    # If you have a threat status value (e.g., 0 or 1), set it here.
    threat_status = 0
    
    # Reconstruct the trajectory. Note that nothing is saved to disk.
    try:
        df_path, dist_m = reconstruct_trajectory_enhanced(
            trajectory_img_path=trajectory_img_path,
            boundary_json_path=boundary_json_path,
            altitude_min=30.0,
            altitude_max=150.0,
            altitude_distributions=altitude_distributions,
            threat_status=threat_status,
            samples_per_point=200,
            connectivity='8',
            close_point_threshold=5
        )
    except Exception as e:
        print(f"Error during reconstruction: {e}")
        exit(1)

    print("Reconstructed path (first few rows):")
    print(df_path.head())
    print(f"...\nTotal distance traveled: {dist_m:.2f} meters")
    
    # Call the new function to plot the altitude profile.
    plot_altitude_profile(df_path, dist_m)
