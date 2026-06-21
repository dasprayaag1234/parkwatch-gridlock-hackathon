# ==============================================================================
# UrbanFlux Master Production Pipeline
# Flipkart Gridlock Submission
# ==============================================================================

import logging
import time
import json
from pathlib import Path
import pandas as pd
import numpy as np
from sklearn.cluster import DBSCAN, KMeans
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_absolute_error

# Optional Visualization
try:
    import folium
    from folium.plugins import HeatMap
    HAS_FOLIUM = True
except ImportError:
    HAS_FOLIUM = False

# Gradient Boosting Engine
try:
    import lightgbm as lgb
    HAS_LGB = True
except ImportError:
    HAS_LGB = False

# ----------------- FIXED GLOBAL CONFIGURATION -----------------
CSV_PATH = "jan to may police violation_anonymized791b166.csv"

# DBSCAN settings: Earth radius = 6371000m. 
# eps = 35 meters / 6371000 meters = ~0.00000549 radians
DBSCAN_EPS_RAD = 35 / 6371000
DBSCAN_MIN_SAMPLES = 50

# Spatial Bounds for Greater Bengaluru Area
LAT_MIN, LAT_MAX = 12.80, 13.30
LON_MIN, LON_MAX = 77.44, 77.78

# Configure Logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

# --- TEAMMATE MERGE: TSP PATROL ROUTING MATH ---
def haversine_m(lat1, lon1, lat2, lon2):
    """Vectorized great-circle distance in metres."""
    R = 6_371_000.0
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
    return 2 * R * np.arcsin(np.sqrt(np.clip(a, 0, 1)))

def greedy_tsp(coords: list) -> tuple:
    """Nearest-neighbour greedy TSP approximation for patrol dispatch."""
    n = len(coords)
    if n <= 1: return list(range(n)), 0.0
    lats = np.array([c[0] for c in coords])
    lons = np.array([c[1] for c in coords])
    dist_matrix = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            dist_matrix[i][j] = haversine_m(lats[i], lons[i], lats[j], lons[j])
    visited = [False] * n
    route   = [0]
    visited[0] = True
    total_dist = 0.0
    for _ in range(n - 1):
        last = route[-1]
        best_dist = float("inf")
        best_next = -1
        for j in range(n):
            if not visited[j] and dist_matrix[last][j] < best_dist:
                best_dist = dist_matrix[last][j]
                best_next = j
        route.append(best_next)
        visited[best_next] = True
        total_dist += best_dist
    total_dist += dist_matrix[route[-1]][0] # Return to start
    return route, round(total_dist / 1000, 2)

def optimise_patrol_route(top10: pd.DataFrame) -> tuple:
    coords = list(zip(top10["center_lat"], top10["center_lon"]))
    order, total_km = greedy_tsp(coords)
    route = []
    for i, idx in enumerate(order):
        row = top10.iloc[idx]
        next_idx = order[(i + 1) % len(order)]
        next_row  = top10.iloc[next_idx]
        leg_km = round(haversine_m(row["center_lat"], row["center_lon"],
                                   next_row["center_lat"], next_row["center_lon"]) / 1000, 2)
        route.append({
            "stop": i + 1, "rank": int(row.name + 1 if "rank" not in row else row["rank"]),
            "junction": row["primary_junction"], "jurisdiction": row["jurisdiction"],
            "lat": row["center_lat"], "lon": row["center_lon"], "leg_km": leg_km,
        })
    return pd.DataFrame(route), total_km
# -----------------------------------------------


def load_data(path):
    start_time = time.time()
    logging.info("Stage 1: Beginning Ingestion and Structural Filtering Pipeline...")
    
    df = pd.read_csv(path)
    initial_count = len(df)
    logging.info(f" -> Initial Records Loaded: {initial_count}")
    
    # Drop hard duplicates
    df = df.drop_duplicates(subset=["id"]) if "id" in df.columns else df.drop_duplicates()
    dedup_count = len(df)
    logging.info(f" -> Post-Dedup Count: {dedup_count} (Dropped {initial_count - dedup_count})")
    
    # Parse Timestamps cleanly
    df["created_datetime"] = pd.to_datetime(df["created_datetime"], errors="coerce", utc=True)
    df["validation_timestamp"] = pd.to_datetime(df["validation_timestamp"], errors="coerce", utc=True)
    df = df.dropna(subset=["created_datetime", "latitude", "longitude"])
    
    # Apply Bounding Box Filter to wipe out GPS anomalies
    df = df[
        (df["latitude"].between(LAT_MIN, LAT_MAX)) & 
        (df["longitude"].between(LON_MIN, LON_MAX))
    ].copy()
    bbox_count = len(df)
    logging.info(f" -> Post-BBox Spatial Filter Count: {bbox_count} (Dropped {dedup_count - bbox_count})")
    
    # Filter out explicitly rejected and duplicate statuses, preserving unprocessed nulls
    if "validation_status" in df.columns:
        df = df[~df["validation_status"].isin(["rejected", "duplicate"])].copy()
    status_count = len(df)
    logging.info(f" -> Post-Validation Status Filter Count: {status_count} (Dropped {bbox_count - status_count})")
    
    # Data Alignment for Vehicle Information
    df["final_vehicle_type"] = df["updated_vehicle_type"].fillna(df["vehicle_type"]).fillna("UNKNOWN").astype(str).str.upper()
    df["final_vehicle_number"] = df["updated_vehicle_number"].fillna(df["vehicle_number"]).fillna("UNKNOWN").astype(str).str.upper()
    
    # Parse Semantic Violation Arrays
    def safe_json_extract(val):
        if pd.isna(val): return []
        if isinstance(val, list): return val
        try:
            return json.loads(val)
        except Exception:
            return []
            
    df["violation_list"] = df["violation_type"].apply(safe_json_extract)
    
    # --- BURST DEDUPLICATION (15-Minute Sensor Sync Window) ---
    df = df.sort_values(['final_vehicle_number', 'created_datetime'])
    df['time_diff'] = df.groupby('final_vehicle_number')['created_datetime'].diff()
    df = df[(df['time_diff'].isna()) | (df['time_diff'] > pd.Timedelta(minutes=15))].copy()
    df = df.drop(columns=['time_diff'])
    burst_count = len(df)
    logging.info(f" -> Post-Burst Deduplication Count: {burst_count} (Dropped {status_count - burst_count})")
    
    return df, start_time


def preprocess(df):
    logging.info("Stage 2: Executing Operational Feature Engineering Engine...")
    
    # --- CRITICAL TIMEZONE FIX ---
    # The database exports in UTC. We MUST convert to IST (+05:30) before extracting hours.
  
    dt_ist = df["created_datetime"].dt.tz_convert('Asia/Kolkata')
    
    df["year"] = dt_ist.dt.year  
    df["month"] = dt_ist.dt.month
    df["day"] = dt_ist.dt.day  
    df["hour"] = dt_ist.dt.hour
    df["dayofweek"] = dt_ist.dt.dayofweek
    df["weekend"] = (df["dayofweek"] >= 5).astype(int)
    
   
    df["peak_hour"] = (((df["hour"].between(7, 10)) | (df["hour"].between(17, 21))).astype(int))
    # -----------------------------
    
    # Refactored Processing Latency Calculation 
    df["time_to_validation_hours"] = (df["validation_timestamp"] - df["created_datetime"]).dt.total_seconds() / 3600.0
    df.loc[df["time_to_validation_hours"] < 0, "time_to_validation_hours"] = 0.0
    # Fill remaining processing pipelines with median validation latency
    median_latency = df["time_to_validation_hours"].median()
    df["time_to_validation_hours"] = df["time_to_validation_hours"].fillna(median_latency if pd.notna(median_latency) else 0.0)
    
    # Build Multi-Violation Impact Modifiers (Expanded to catch all vehicle classes)
  
    vehicle_weights = {
        "TANKER": 5.0, "HEAVY GOODS VEHICLE": 4.0, "TRACTOR/TRAILER": 4.0, 
        "PRIVATE BUS": 4.0, "BUS": 4.0, "BMTC BUS": 4.0, "KSRTC BUS": 4.0,
        "MINI BUS": 3.0, "MAXI-CAB": 3.0, "LGV": 3.0, "LCV": 3.0,
        "VAN": 2.5, 
        "CAR": 2.0, "JEEP": 2.0, 
        "GOODS AUTO": 1.5, "PASSENGER AUTO": 1.5, "AUTO RICKSHAW": 1.5,
        "SCOOTER": 1.0, "MOTOR CYCLE": 1.0, "TWO-WHEELER": 1.0, "MOPED": 1.0
    }
    df["vehicle_weight"] = df["final_vehicle_type"].map(vehicle_weights).fillna(1.0)
    
    # Updated heavy vehicle flag to catch the new bus and LGV categories
    heavy_list = ["TANKER", "HEAVY GOODS VEHICLE", "TRACTOR/TRAILER", "MINI BUS", "MAXI-CAB", "PRIVATE BUS", "BUS", "BMTC BUS", "KSRTC BUS", "LGV", "LCV"]
    df["is_heavy"] = df["final_vehicle_type"].isin(heavy_list).astype(int)
    violation_weights = {
        "DOUBLE PARKING": 3.5,
        "PARKING OPPOSITE TO ANOTHER PARKED VEHICLE": 3.0,
        "PARKING IN A MAIN ROAD": 2.5,
        "PARKING NEAR ROAD CROSSING": 2.5,
        "PARKING NEAR BUSTOP/SCHOOL/HOSPITAL ETC": 2.0,
        "WRONG PARKING": 1.5,
        "NO PARKING": 1.0,
        "PARKING ON FOOTPATH": 2.5,
        "DEFECTIVE NUMBER PLATE": 0.0,
        "PARKING NEAR TRAFFIC LIGHT OR ZEBRA CROSS": 3.0
    }
    
    def calculate_cumulative_violation_weight(v_list):
        if not v_list: return 1.0
        return sum([violation_weights.get(str(v).upper(), 1.0) for v in v_list])
        
    df["violation_weight"] = df["violation_list"].apply(calculate_cumulative_violation_weight)
    # Impact Heuristic: 
    #   Violations * Weight (Vehicle Size) * 15 minutes average delay * ₹250/hour productivity loss
    #   This converts 'weighted impact' into an estimated economic loss (Rupees) per citation.
    AVG_DELAY_MINS = 15.0
    COST_PER_HOUR = 250.0 
    
    df["base_congestion_impact"] = df["vehicle_weight"] * df["violation_weight"]
    df["economic_impact_rupees"] = (df["base_congestion_impact"] * (AVG_DELAY_MINS / 60.0) * COST_PER_HOUR)
    
    # Flags for precise Problem Statement alignment
    df["has_double_parking"] = df["violation_list"].apply(lambda l: any("DOUBLE PARKING" in str(x).upper() for x in l)).astype(int)
    df["near_metro_station"] = df["junction_name"].fillna("").str.upper().str.contains("METRO STATION").astype(int)

    vehicle_counts = df.groupby('final_vehicle_number').size()
    df['is_global_repeat'] = df['final_vehicle_number'].isin(vehicle_counts[vehicle_counts > 1].index).astype(int)
    # ------------------------------------------------------
    
    # Compute true weighted spatial vectors before grouping
    df["weighted_lat"] = df["latitude"] * df["vehicle_weight"]
    df["weighted_lon"] = df["longitude"] * df["vehicle_weight"]
    
    return df


def hotspot_detection(df):
    logging.info("Stage 3: Extracting Micro-Spatial Hotspots via Haversine Metric...")
    
    # Transform coordinates to radians for proper Haversine tracking
    coords_rad = np.radians(df[["latitude", "longitude"]].values)
    df["cluster"] = DBSCAN(eps=DBSCAN_EPS_RAD, min_samples=DBSCAN_MIN_SAMPLES, metric="haversine").fit_predict(coords_rad)
    
    # Drop outliers immediately to isolate the pipeline
    clean_df = df[df["cluster"] != -1].copy()
    clustering_count = len(clean_df)
    logging.info(f" -> Post-Clustering Operational Target Count: {clustering_count} (Noise drop: {len(df) - clustering_count})")
    
    if clean_df.empty:
        logging.warning("No high-density micro-spatial congestion hot zones identified.")
        return clean_df, pd.DataFrame()
    if clean_df.empty:
        logging.warning("No high-density micro-spatial congestion hot zones identified.")
        return clean_df, pd.DataFrame()
   
    logging.info("Stage 3.1: Enforcing Micro-Spatial Constraints (Shattering Mega-Clusters)...")
    MAX_SPAN_METERS = 300.0  # Max allowed diameter for a single patrol deployment zone
    
    refined_dfs = []
    next_new_id = int(clean_df["cluster"].max()) + 1
    
    for c, group in clean_df.groupby("cluster"):
        lat_span = (group["latitude"].max() - group["latitude"].min()) * 111000
        lon_span = (group["longitude"].max() - group["longitude"].min()) * 111000 * np.cos(np.radians(group["latitude"].mean()))
        max_span = max(lat_span, lon_span)
        
        if max_span > MAX_SPAN_METERS:
            # Calculate how many sub-zones we need to enforce the 300m maximum rule
            k_splits = max(2, int(np.ceil(max_span / 150.0)))
            
            # Use KMeans to surgically fracture the long chain into tight micro-clusters
            sub_labels = KMeans(n_clusters=k_splits, random_state=42, n_init=10).fit_predict(group[["latitude", "longitude"]])
            
            group = group.copy()
            for sub in np.unique(sub_labels):
                group.loc[sub_labels == sub, "cluster"] = next_new_id
                next_new_id += 1
        else:
            group = group.copy()
            
        refined_dfs.append(group)
        
    clean_df = pd.concat(refined_dfs, ignore_index=True)
    logging.info(f" -> Mega-clusters shattered. Final micro-spatial cluster count: {clean_df['cluster'].nunique()}")
    # ----------------------------------------------------------------------
    
    # Build Metrics Aggregation Layer
    cluster_records = []
        
    # Build Metrics Aggregation Layer
    cluster_records = []
    
    for c, group in clean_df.groupby("cluster"):
        total_violations = len(group)
        
        # Geolocation Centroid using the pre-computed spatial vectors
        center_lat = group["weighted_lat"].sum() / group["vehicle_weight"].sum()
        center_lon = group["weighted_lon"].sum() / group["vehicle_weight"].sum()
        
        # Calculate cluster geographic area radius footprint
        lat_span = (group["latitude"].max() - group["latitude"].min()) * 111000 # m/deg
        lon_span = (group["longitude"].max() - group["longitude"].min()) * 111000 * np.cos(np.radians(center_lat))
        area_size_sqm = max(np.pi * ((max(lat_span, lon_span) / 2) ** 2), 100.0) # Floor at 100 sqm to protect metrics
        
        # --- NEW: CITATION DENSITY ---
        citation_density = total_violations / (area_size_sqm / 100.0) # Citations per 100 sq meters
        
        # Problem-Statement Specific Rates
        heavy_vehicle_pct = group["is_heavy"].mean()
        double_parking_rate = group["has_double_parking"].mean()
        peak_ratio = group["peak_hour"].mean()
        
        # --- PHASE 1 FIX: ADDITIVE PCII TRANSFORMATION ---
        # Replaced multiplicative collapse with a weighted additive index (0 to 1 range)
        pcii_score = (heavy_vehicle_pct * 0.4) + (double_parking_rate * 0.4) + (peak_ratio * 0.2)
        # -------------------------------------------------
        
     # --- PHASE 3 FIX: CALIBRATED CHOKE ZONE THRESHOLDS ---
        # Threshold set to 1% to mathematically align with the 0.585% global rarity of double parking
        is_choke_zone = "Yes" if (double_parking_rate > 0.01 and peak_ratio > 0.35) else "No"
        
        # Operational Backlog & Vulnerabilities
        mean_latency = group["time_to_validation_hours"].mean()
        
        # --- PHASE 5 FIX: GLOBAL REPEAT OFFENDER RATIO ---
        # Replaced local duplicate counting with true city-wide recidivism
        repeat_offender_ratio = group["is_global_repeat"].mean()
        # -------------------------------------------------
        
        # Isolate top peak enforcement hours
        top_hours = group["hour"].value_counts().head(2).index.tolist()
        peak_hours_str = ", ".join([f"{h:02d}:00" for h in top_hours]) if len(top_hours) >= 2 else "18:00, 09:00"
        
        # Identify Junction Proximity while isolating empty placeholder text
        valid_junctions = group[group["junction_name"].notna() & (group["junction_name"] != "No Junction")]
        primary_junc = str(valid_junctions["junction_name"].mode().iloc[0]) if not valid_junctions.empty else "Mid-Block Segment"
        metro_prox = "Yes" if group["near_metro_station"].any() else "No"
        dominant_v = str(group["final_vehicle_type"].mode().iloc[0]) if not group["final_vehicle_type"].empty else "CAR"
        jurisdiction = str(group["police_station"].mode().iloc[0]) if not group["police_station"].empty else "Unknown District"
        
        cluster_records.append({
            "cluster": c,
            "center_lat": center_lat,
            "center_lon": center_lon,
            "area_size_sqm": area_size_sqm,
            "citation_density": citation_density,
            "violations": total_violations,
            "heavy_vehicle_pct": heavy_vehicle_pct,
            "double_parking_rate": double_parking_rate,
            "peak_ratio": peak_ratio,
            "PCII": pcii_score,
            "CHOKE_ZONE": is_choke_zone,
            "mean_time_to_validation_hours": mean_latency,
            "repeat_offender_ratio": repeat_offender_ratio,
            "peak_hours_deployment": peak_hours_str,
            "primary_junction": primary_junc,
            "metro_proximity": metro_prox,
            "dominant_vehicle": dominant_v,
            "jurisdiction": jurisdiction,
            "total_congestion_impact": group["base_congestion_impact"].sum(),
            "total_economic_impact": group["economic_impact_rupees"].sum()
        })
        
    agg = pd.DataFrame(cluster_records)
    
    # --- FINAL MATH FIX: SCALING BY DENSITY, NOT VOLUME ---
    scaler = MinMaxScaler()
    
    # We now pass 'citation_density' instead of 'violations'
    scaled_metrics = scaler.fit_transform(agg[["citation_density", "mean_time_to_validation_hours", "repeat_offender_ratio", "PCII"]])
    
    # risk_score is now purely a Density metric
    agg["risk_score"] = scaled_metrics[:, 0] * 100.0
    
   
    # PCII (Choke-ness) is bumped to 45%. Density is dropped to 25%.
    agg["base_severity"] = (
        0.25 * agg["risk_score"] + 
        0.45 * (scaled_metrics[:, 3] * 100.0) + # scaled PCII is at index 3
        0.15 * (scaled_metrics[:, 1] * 100.0) + # scaled latency is at index 1
        0.15 * (scaled_metrics[:, 2] * 100.0)   # scaled recidivism is at index 2
    )
    
    
   
    agg["CompositeSeverity"] = agg["base_severity"] + np.where(agg["CHOKE_ZONE"] == "Yes", 10.0, 0.0)
    
    # Cap the final score at 100 to keep the UI clean
    agg["CompositeSeverity"] = agg["CompositeSeverity"].clip(upper=100.0)
    # -----------------------------------------------
    
    # Calculate Targeting Metrics
    agg["Enforcement_ROI"] = agg["CompositeSeverity"] / (agg["area_size_sqm"] / 100) # Per 100 Square Meters
    
    return clean_df, agg


def forecast(clean_df, agg):
    if not HAS_LGB or clean_df.empty or agg.empty:
        return pd.DataFrame(), pd.DataFrame()
        
    logging.info("Stage 4: Building Chronological Cross-Validation Time-Series Predictor...")
    
    # Generate spatial-temporal grouped index logs
    ts_frame = clean_df.groupby(["cluster", "year", "month", "day", "hour", "dayofweek", "weekend"]).size().reset_index(name="violation_count")
    
    # Generate spatial-temporal grouped index logs
    ts_frame = clean_df.groupby(["cluster", "year", "month", "day", "hour", "dayofweek", "weekend"]).size().reset_index(name="violation_count")
    
    # --- TEAMMATE MERGE: MACHINE LEARNING LAG-LEAKAGE FIX ---
    # 1. Create a true datetime column for time-aware math
    ts_frame["timestamp"] = pd.to_datetime(ts_frame[["year", "month", "day", "hour"]])
    
    # 2. STRICT GLOBAL TIME SORT FIRST.
    # By strictly sorting globally by time BEFORE calculating lags, we guarantee that 
    # the TimeSeriesSplit cross-validator never accidentally leaks future cluster data into past folds.
    ts_frame = ts_frame.sort_values("timestamp").reset_index(drop=True)
    
    # 3. Compute per-cluster lag features securely on the chronological timeline
    ts_frame["prev_record_vol"] = ts_frame.groupby("cluster")["violation_count"].shift(1)
    
    # 4. True Time-Aware 7-Day Rolling Average (Based on calendar days)
    ts_frame = ts_frame.set_index("timestamp")
    ts_frame["true_rolling_7d"] = ts_frame.groupby("cluster")["violation_count"].transform(lambda x: x.rolling('7D', min_periods=1).mean())
    ts_frame = ts_frame.reset_index()
    
    # 5. Impute initial bootstrap features safely
    median_vol = ts_frame["violation_count"].median()
    ts_frame["prev_record_vol"] = ts_frame["prev_record_vol"].fillna(median_vol)
    ts_frame["true_rolling_7d"] = ts_frame["true_rolling_7d"].fillna(median_vol)
    # --------------------------------------------------------
    
    features = ["cluster", "month", "hour", "dayofweek", "weekend", "prev_record_vol", "true_rolling_7d"]
    X = ts_frame[features]
    y = ts_frame["violation_count"]
    
    # Robust Cross-Validation via 3-Fold TimeSeriesSplit
    tscv = TimeSeriesSplit(n_splits=3)
    mae_scores = []
    
    for train_idx, test_idx in tscv.split(X):
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
        
        model_cv = lgb.LGBMRegressor(n_estimators=100, learning_rate=0.05, random_state=42, verbose=-1)
        model_cv.fit(X_train, y_train, categorical_feature=["cluster"])
        preds = model_cv.predict(X_test)
        mae_scores.append(mean_absolute_error(y_test, preds))
        
    logging.info(f" -> TimeSeries Cross-Validated MAE Score: {np.mean(mae_scores):.4f}")
    
    # Train deployment ready forecasting production model
    prod_model = lgb.LGBMRegressor(n_estimators=150, learning_rate=0.05, random_state=42, verbose=-1)
    prod_model.fit(X, y, categorical_feature=["cluster"])
    
    # Generate explicit forecast scenarios using real peak data attributes
    future_rows = []
    for c in agg["cluster"].unique():
        c_data = ts_frame[ts_frame.cluster == c]
        
        # Create features from actual empirical peak data states
        base_scenario = {
            "cluster": c,
            "month": int(c_data["month"].max()) if not c_data.empty else 4,
            "hour": int(c_data.loc[c_data["violation_count"].idxmax(), "hour"]) if not c_data.empty else 18,
            "dayofweek": int(c_data["dayofweek"].mode().iloc[0]) if not c_data.empty else 4,
            "weekend": int(c_data["weekend"].mode().iloc[0]) if not c_data.empty else 1,
            "prev_record_vol": float(c_data["violation_count"].iloc[-1]) if not c_data.empty else 1.0,
            "true_rolling_7d": float(c_data["true_rolling_7d"].iloc[-1]) if not c_data.empty else 1.0
        }
        
        pred_input = pd.DataFrame([base_scenario])[features]
        horizon_pred = prod_model.predict(pred_input)[0]
        
        future_rows.append({
            "cluster": c,
            "empirical_peak_hour_forecast": max(0.0, float(horizon_pred))
        })
        
    fi_df = pd.DataFrame({"feature": features, "importance": prod_model.feature_importances_})
    return pd.DataFrame(future_rows), fi_df


def build_recommendations(df_dashboard):
    recs = []
    for _, row in df_dashboard.iterrows():
        base = "Deploy Quick-Response Patrol Unit."
        if row["CompositeSeverity"] >= 75:
            base = "Deploy Heavy-Duty Tow Fleet + Continuous Saturation Enforcement."
        elif row["CompositeSeverity"] >= 50:
            base = "Deploy Regional Enforcement Interceptors."
            
        if row["CHOKE_ZONE"] == "Yes":
            base += " [CHOKE ZONE PROTOCOL: Initiate instant vehicle clamping / punitive towing structural guidelines.]"
        if row["repeat_offender_ratio"] > 0.25:
            base += " [HIGH RECIDIVISM: Initiate targeted digital registration blacklisting warning sweeps.]"
        if row["metro_proximity"] == "Yes":
            base += " [METRO TRANSIT TRANSIT HUB: Align enforcement drop windows with regional transit timetables.]"
            
        recs.append(base)
    return recs


def generate_heatmap(clean_df, top_10, route_df=None, total_km=0):
    if not HAS_FOLIUM or clean_df.empty or top_10.empty:
        return
        
    logging.info("Stage 5: Rendering Interactive Geospatial Dashboard Layers...")
    start_lat = top_10['center_lat'].iloc[0]
    start_lon = top_10['center_lon'].iloc[0]
    
    # This is the line that was missing! It defines "m"
    m = folium.Map(location=[start_lat, start_lon], zoom_start=13, tiles='cartodbpositron')
    
    # Render congestion impact scatter logs
    heat_data = [[row['latitude'], row['longitude'], row['base_congestion_impact']] for _, row in clean_df.sample(min(10000, len(clean_df))).iterrows()]
    HeatMap(heat_data, radius=8, blur=12).add_to(m)
    
    for _, row in top_10.iterrows():
        gmaps_url = f"https://www.google.com/maps/search/?api=1&query={row['center_lat']},{row['center_lon']}"
        popup_html = f"""
        <div style='font-family: Arial, sans-serif; width: 260px;'>
            <h4>Cluster ID: {int(row['cluster'])} ({row['jurisdiction']})</h4>
            <hr style='margin: 4px 0;'>
            <b>Primary Junction:</b> {row['primary_junction']}<br>
            <b>Choke Zone Status:</b> <span style='color: {"red" if row["CHOKE_ZONE"]=="Yes" else "black"}; font-weight: bold;'>{row['CHOKE_ZONE']}</span><br>
            <b>Composite Severity Score:</b> {row['CompositeSeverity']:.2f}<br>
            <b>Enforcement ROI Metric:</b> {row['Enforcement_ROI']:.4f}<br>
            <b>Target Deployment Window:</b> {row['peak_hours_deployment']}<br>
            <b>Metro Station Proximity:</b> {row['metro_proximity']}<br>
            <a href='{gmaps_url}' target='_blank' style='display: inline-block; margin-top: 8px; padding: 4px 8px; background-color: #4285F4; color: white; text-decoration: none; border-radius: 4px;'>Open Google Maps</a>
        </div>
        """
        
        marker_color = 'red' if row['CompositeSeverity'] >= 75 else 'orange' if row['CompositeSeverity'] >= 50 else 'blue'
        folium.Marker(
            location=[row['center_lat'], row['center_lon']],
            popup=folium.Popup(popup_html, max_width=300),
            icon=folium.Icon(color=marker_color, icon='info-sign')
        ).add_to(m)
        
    # --- TEAMMATE MERGE: DRAW PATROL ROUTE ---
    if route_df is not None and not route_df.empty:
        route_coords = [(row["lat"], row["lon"]) for _, row in route_df.iterrows()]
        route_coords.append((route_df.iloc[0]["lat"], route_df.iloc[0]["lon"])) # Close the loop
        folium.PolyLine(
            route_coords, color="#e74c3c", weight=3.0, opacity=0.8, dash_array="10",
            tooltip=f"Optimal Patrol Circuit: {total_km} km"
        ).add_to(m)
    # -----------------------------------------
        
    m.save("urbanflux_calibrated_dashboard.html")
    logging.info(" -> Heatmap Saved successfully to urbanflux_calibrated_dashboard.html")
        

def main():
    df, start_time = load_data(CSV_PATH)
    if df.empty:
        logging.error("Execution failed: Post-filtering dataset layer returned empty.")
        return
        
    df = preprocess(df)
    clean_df, agg = hotspot_detection(df)
    
    if not agg.empty:
        forecast_df, fi = forecast(clean_df, agg)
        if not forecast_df.empty:
            agg = agg.merge(forecast_df, on="cluster", how="left")
            
        # Isolate elite top-10 targeted enforcement zones
        dashboard_top10 = agg.sort_values(by="CompositeSeverity", ascending=False).head(10).copy()
        dashboard_top10["Actionable_Recommendation"] = build_recommendations(dashboard_top10)
        
       # ... existing ...
        dashboard_top10.to_csv("final_dashboard.csv", index=False)
        if HAS_LGB and not fi.empty:
            fi.to_csv("feature_importance.csv", index=False)
            
        # --- TEAMMATE MERGE: EXECUTE ROUTING ---
        route_df, total_km = optimise_patrol_route(dashboard_top10)
        route_df.to_csv("patrol_route.csv", index=False)
        
        # Pass the route into your heatmap generator
        generate_heatmap(clean_df, dashboard_top10, route_df, total_km)
  
        
        # Execution Summary Compilation
        runtime = time.time() - start_time
        summary_lines = [
            "==========================================================",
            "        URBANFLUX CALIBRATED DATA PIPELINE LOGS            ",
            "==========================================================",
            f"Total System Pipeline Runtime: {runtime:.2f} Seconds",
            f"Active High Density Clusters Identified: {len(agg)}",
            "\nTop 3 Identified Enforcement Corridors Alpha Signatures:",
        ]
        
        for idx, r in dashboard_top10.head(3).iterrows():
            summary_lines.append(
                f" -> Zone {int(r.cluster)} | Severity: {r.CompositeSeverity:.2f} | "
                f"ROI: {r.Enforcement_ROI:.4f} | Junction: {r.primary_junction} | Choke Zone: {r.CHOKE_ZONE}"
            )
        
        # --- CRITICAL BUG FIX 3: THE PCII DISCLAIMER ---
        summary_lines.append("==========================================================")
        summary_lines.append("METHODOLOGY DISCLAIMER: The PCII (Physical Corridor Congestion")
        summary_lines.append("Index Impact) is a heuristic proxy index derived from vehicle")
        summary_lines.append("density multipliers. It is an enforcement burden indicator,")
        summary_lines.append("not a raw traffic flow measurement.")
        summary_lines.append("==========================================================")
        # ---------------------------------------------------------

        summary_text = "\n".join(summary_lines)
        print(summary_text)
        Path("pipeline_summary.log").write_text(summary_text, encoding="utf-8")
        logging.info("UrbanFlux Master Deployment Blueprint successfully compiled.")
    else:
        logging.warning("Pipeline executed successfully but generated 0 high-density clusters.")


if __name__ == "__main__":
    main()