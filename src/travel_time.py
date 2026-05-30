import math


SPEED_LIMIT_KMH = 60
INTERSECTION_DELAY_SECONDS = 30

# From Traffic Flow to Travel Time Conversion PDF:
# flow = -1.4648375 * speed^2 + 93.75 * speed
A = -1.4648375
B = 93.75

LOW_FLOW_THRESHOLD = 351
CAPACITY_FLOW = 1500


def flow_to_speed(flow_vph):
    """
    Convert traffic flow in vehicles/hour to speed in km/h.
    speed is capped at 60 km/h
    therwise follows a quadratic relationship.
    """

    if flow_vph <= LOW_FLOW_THRESHOLD:
        return SPEED_LIMIT_KMH

    # Avoid impossible values above the capacity used in the assignment model.
    flow_vph = min(flow_vph, CAPACITY_FLOW)

    # A * speed^2 + B * speed - flow = 0
    c = -flow_vph
    discriminant = (B ** 2) - (4 * A * c)

    if discriminant < 0:
        return 32

    sqrt_disc = math.sqrt(discriminant)

    speed_1 = (-B + sqrt_disc) / (2 * A)
    speed_2 = (-B - sqrt_disc) / (2 * A)

    # Under-capacity branch = higher speed
    speed = max(speed_1, speed_2)

    # Safety limits
    speed = min(speed, SPEED_LIMIT_KMH)
    speed = max(speed, 1)

    return speed


def calculate_travel_time(distance_km, predicted_flow_15min):
    """
    Convert predicted 15-minute traffic flow into travel time in minutes.

    Args:
        distance_km: distance between two SCATS sites in kilometres
        predicted_flow_15min: predicted traffic count for one 15-minute interval

    Returns: travel time in minutes
    """

    flow_vph = predicted_flow_15min * 4
    speed_kmh = flow_to_speed(flow_vph)

    travel_time_hours = distance_km / speed_kmh
    travel_time_seconds = travel_time_hours * 3600

    total_seconds = travel_time_seconds + INTERSECTION_DELAY_SECONDS

    return total_seconds / 60


def haversine_distance(lat1, lon1, lat2, lon2):
    """
    Calculate straight-line distance between two lat/long coordinates.
    Returns distance in kilometres.
    """

    earth_radius_km = 6371

    lat1 = math.radians(float(lat1))
    lon1 = math.radians(float(lon1))
    lat2 = math.radians(float(lat2))
    lon2 = math.radians(float(lon2))

    diff_lat = lat2 - lat1
    diff_lon = lon2 - lon1

    a = (
        math.sin(diff_lat / 2) ** 2
        + math.cos(lat1) * math.cos(lat2) * math.sin(diff_lon / 2) ** 2
    )

    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))

    return earth_radius_km * c


if __name__ == "__main__":
    predicted_flow = 120
    distance = 2.0

    speed = flow_to_speed(predicted_flow * 4)
    time = calculate_travel_time(distance, predicted_flow)

    print("Predicted 15-min flow:", predicted_flow)
    print("Hourly flow:", predicted_flow * 4)
    print("Estimated speed:", round(speed, 2), "km/h")
    print("Estimated travel time:", round(time, 2), "minutes")