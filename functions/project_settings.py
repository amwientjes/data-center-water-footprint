"""Project-wide variables."""

from pyproj import CRS

# Coordinate Reference Systems for the project
WGS84_CRS: CRS = CRS.from_string("EPSG:4326")  # Mercator projection
MOLLWEIDE_CRS: CRS = CRS.from_string("ESRI:54009")  # Mollweide projection (for meter units)

# Scenario names
GLOBAL_WARMING_SCENARIOS = ["hist", "1_5C", "2_0C", "3_2C"]
GLOBAL_WARMING_SCENARIOS_FUTURE = [s for s in GLOBAL_WARMING_SCENARIOS if s != "hist"]
