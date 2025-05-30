import config
import numpy as np
from numpy import sin, cos, pi, sqrt, arctan2, arcsin
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


class GeodeticTransform:
    """
    FIXED: Transformasi antara koordinat geodetic (lat/lon/alt) dan NED frame
    """

    def __init__(self, ref_lat=0.0, ref_lon=0.0, ref_alt=0.0):
        """
        Initialize dengan reference point untuk NED frame origin

        Args:
            ref_lat: Reference latitude (degrees)
            ref_lon: Reference longitude (degrees) 
            ref_alt: Reference altitude (m)
        """
        self.ref_lat_deg = ref_lat
        self.ref_lon_deg = ref_lon
        self.ref_alt = ref_alt

        # Convert to radians for calculations
        self.ref_lat = np.deg2rad(ref_lat)
        self.ref_lon = np.deg2rad(ref_lon)

        # WGS84 Earth parameters
        self.a = 6378137.0           # Semi-major axis (m)
        self.f = 1.0 / 298.257223563  # Flattening
        self.e2 = 2 * self.f - self.f**2  # First eccentricity squared

        # Pre-compute reference point in ECEF
        self.ref_ecef = self.geodetic_to_ecef(
            self.ref_lat, self.ref_lon, self.ref_alt)

        # Transformation matrix from ECEF to NED at reference point
        self.T_ecef_to_ned = self._compute_ecef_to_ned_matrix(
            self.ref_lat, self.ref_lon)

    def geodetic_to_ecef(self, lat, lon, alt):
        """
        Convert geodetic coordinates to ECEF

        Args:
            lat: Latitude (radians)
            lon: Longitude (radians)
            alt: Altitude (m)

        Returns:
            ECEF coordinates [x, y, z] (m)
        """
        # Radius of curvature in prime vertical
        N = self.a / sqrt(1 - self.e2 * sin(lat)**2)

        # ECEF coordinates
        x = (N + alt) * cos(lat) * cos(lon)
        y = (N + alt) * cos(lat) * sin(lon)
        z = (N * (1 - self.e2) + alt) * sin(lat)

        return np.array([x, y, z])

    def ecef_to_geodetic(self, ecef):
        """
        Convert ECEF coordinates to geodetic (iterative method)

        Args:
            ecef: ECEF coordinates [x, y, z] (m)

        Returns:
            [lat, lon, alt] in radians, radians, meters
        """
        x, y, z = ecef

        # Longitude
        lon = arctan2(y, x)

        # Iterative solution for latitude and altitude
        p = sqrt(x**2 + y**2)
        lat = arctan2(z, p * (1 - self.e2))

        for _ in range(5):  # Usually converges in 2-3 iterations
            N = self.a / sqrt(1 - self.e2 * sin(lat)**2)
            alt = p / cos(lat) - N
            lat = arctan2(z, p * (1 - self.e2 * N / (N + alt)))

        return np.array([lat, lon, alt])

    def _compute_ecef_to_ned_matrix(self, lat, lon):
        """
        Compute transformation matrix from ECEF to NED
        """
        T = np.array([
            [-sin(lat) * cos(lon), -sin(lat) * sin(lon),  cos(lat)],
            [-sin(lon),             cos(lon),             0],
            [-cos(lat) * cos(lon), -cos(lat) * sin(lon), -sin(lat)]
        ])
        return T

    def geodetic_to_ned(self, lat, lon, alt):
        """
        Convert geodetic coordinates to NED frame

        Args:
            lat: Latitude (radians)
            lon: Longitude (radians)
            alt: Altitude (m)

        Returns:
            NED coordinates [north, east, down] (m)
        """
        # Convert to ECEF
        ecef = self.geodetic_to_ecef(lat, lon, alt)

        # Translate to reference point
        ecef_rel = ecef - self.ref_ecef

        # Transform to NED
        ned = self.T_ecef_to_ned @ ecef_rel

        return ned

    def ned_to_geodetic(self, ned):
        """
        Convert NED coordinates to geodetic

        Args:
            ned: NED coordinates [north, east, down] (m)

        Returns:
            [lat, lon, alt] in radians, radians, meters
        """
        # Transform NED to ECEF relative
        ecef_rel = self.T_ecef_to_ned.T @ ned

        # Add reference ECEF
        ecef = ecef_rel + self.ref_ecef

        # Convert to geodetic
        return self.ecef_to_geodetic(ecef)

    def geodetic_to_ned_simple(self, lat, lon, alt):
        """
        FIXED: Simple flat-earth approximation (faster, good for short distances)

        Args:
            lat: Latitude (radians)
            lon: Longitude (radians)
            alt: Altitude (m)

        Returns:
            NED coordinates [north, east, down] (m)
        """
        # Earth radius approximation
        R_earth = 6371000.0  # meters

        # Convert to NED using flat-earth approximation
        north = (lat - self.ref_lat) * R_earth
        east = (lon - self.ref_lon) * R_earth * cos(self.ref_lat)
        down = -(alt - self.ref_alt)  # NED down is negative altitude

        return np.array([north, east, down])

    def ned_to_geodetic_simple(self, ned):
        """
        FIXED: Simple flat-earth approximation (inverse)

        Args:
            ned: NED coordinates [north, east, down] (m)

        Returns:
            [lat, lon, alt] in radians, radians, meters
        """
        R_earth = 6371000.0

        lat = self.ref_lat + ned[0] / R_earth
        lon = self.ref_lon + ned[1] / (R_earth * cos(self.ref_lat))
        alt = self.ref_alt - ned[2]  # Convert down to altitude

        return np.array([lat, lon, alt])


class IMUSensor:
    def __init__(self):
        # Noise parameters (standard deviation)
        self.acc_noise_std = 0.05  # m/s^2
        self.gyro_noise_std = 0.01  # rad/s

        # Bias parameters
        self.acc_bias = np.array([0.1, -0.05, 0.03])  # Initial bias (m/s^2)
        self.gyro_bias = np.array([0.01, -0.01, 0.005])  # Initial bias (rad/s)

        # Parameters for random walk bias model
        self.acc_bias_walk = 0.0001  # m/s^2 per √s
        self.gyro_bias_walk = 0.0001  # rad/s per √s

        # Sampling frequency
        self.freq = 100  # Hz
        self.dt = 1.0/self.freq

    def measure(self, quad, t):
        # Get acceleration in inertial frame
        acc_inertial = quad.acc

        # Convert acceleration to body frame
        R = quad.dcm  # Rotation matrix from body to inertial
        acc_body_without_gravity = R.T @ acc_inertial

        # Add gravity effect in body frame (for NED)
        gravity_ned = np.array([0, 0, 9.81])  # Gravitasi dalam NED
        gravity_body = R.T @ gravity_ned      # Transformasi ke body frame

        # Accelerometer mengukur specific force (percepatan + gravitasi)
        acc_true = acc_body_without_gravity - gravity_body

        # Gyroscope measure is already correct (angular velocity in body frame)
        omega_true = quad.omega

        # Add noise and bias
        acc_noise = np.random.normal(0, self.acc_noise_std, 3)
        gyro_noise = np.random.normal(0, self.gyro_noise_std, 3)

        # Update bias with random walk model
        self.acc_bias += np.random.normal(0,
                                          self.acc_bias_walk * np.sqrt(self.dt), 3)
        self.gyro_bias += np.random.normal(0,
                                           self.gyro_bias_walk * np.sqrt(self.dt), 3)

        # Final sensor readings
        acc_measured = acc_true + self.acc_bias + acc_noise
        gyro_measured = omega_true + self.gyro_bias + gyro_noise

        return acc_measured, gyro_measured


class GPSSensor:
    """
    FIXED: Realistic GPS sensor using home position concept
    """

    def __init__(self, home_lat=-7.250445, home_lon=112.768845, home_alt=10.0):
        """
        Initialize realistic GPS sensor with home position

        Args:
            home_lat: Home latitude in degrees (Surabaya)
            home_lon: Home longitude in degrees (Surabaya)  
            home_alt: Home altitude in meters MSL
        """
        # Home position (reference point)
        self.home_lat = home_lat  # degrees
        self.home_lon = home_lon  # degrees
        self.home_alt = home_alt  # meters MSL

        # Earth parameters for conversion
        self.deg_to_m_lat = 111320.0  # meters per degree latitude
        self.deg_to_m_lon = 111320.0 * \
            cos(np.deg2rad(home_lat))  # adjusted for latitude

        # Realistic GPS parameters
        self.pos_noise_std = 1.5     # meters horizontal
        # meters vertical (GPS altitude usually worse)
        self.alt_noise_std = 2.0
        self.vel_noise_std = 0.1     # m/s

        # GPS noise in coordinate domain
        self.lat_noise_std = self.pos_noise_std / self.deg_to_m_lat  # degrees
        self.lon_noise_std = self.pos_noise_std / self.deg_to_m_lon  # degrees

        # FIXED: GPS timing - more flexible
        self.freq = 5  # Hz (typical GPS update rate)
        self.dt = 1.0 / self.freq
        self.last_update = -999  # Force first measurement

        # Statistics tracking
        self.measurement_count = 0

        print(f"Realistic GPS Sensor initialized:")
        print(
            f"  Home position: {home_lat:.6f}°, {home_lon:.6f}°, {home_alt:.1f}m MSL")
        print(
            f"  Position noise: ±{self.pos_noise_std}m horizontal, ±{self.alt_noise_std}m vertical")
        print(
            f"  Coordinate noise: ±{self.lat_noise_std*1e6:.1f}µ°lat, ±{self.lon_noise_std*1e6:.1f}µ°lon")
        print(f"  Update rate: {self.freq} Hz")

    def measure(self, quad, t):
        """
        FIXED: Realistic GPS measurement with proper timing

        Args:
            quad: Quadcopter object with .pos and .vel in NED frame
            t: Current time

        Returns:
            tuple: (pos_ned_measured, vel_ned_measured, geodetic_raw)
        """
        # FIXED: More flexible timing check
        if t - self.last_update < self.dt - 0.01:  # Small tolerance
            return None, None, None

        self.last_update = t
        self.measurement_count += 1

        # Get true position and velocity from simulation (NED frame)
        true_pos_ned = quad.pos.copy()
        true_vel_ned = quad.vel.copy()

        # STEP 1: Convert true NED to geodetic coordinates
        true_lat = self.home_lat + true_pos_ned[0] / self.deg_to_m_lat
        true_lon = self.home_lon + true_pos_ned[1] / self.deg_to_m_lon
        # NED down is negative altitude
        true_alt = self.home_alt - true_pos_ned[2]

        # STEP 2: Apply realistic GPS noise in geodetic domain
        lat_noise = np.random.normal(0, self.lat_noise_std)
        lon_noise = np.random.normal(0, self.lon_noise_std)
        alt_noise = np.random.normal(0, self.alt_noise_std)

        # Noisy GPS coordinates (what GPS receiver outputs)
        gps_lat_raw = true_lat + lat_noise
        gps_lon_raw = true_lon + lon_noise
        gps_alt_raw = true_alt + alt_noise

        # STEP 3: Convert noisy GPS coordinates back to NED for EKF
        gps_north = (gps_lat_raw - self.home_lat) * self.deg_to_m_lat
        gps_east = (gps_lon_raw - self.home_lon) * self.deg_to_m_lon
        gps_down = -(gps_alt_raw - self.home_alt)

        gps_pos_ned = np.array([gps_north, gps_east, gps_down])

        # STEP 4: Add velocity noise directly in NED frame
        vel_noise_ned = np.random.normal(0, self.vel_noise_std, 3)
        gps_vel_ned = true_vel_ned + vel_noise_ned

        # STEP 5: Package raw geodetic data
        geodetic_raw = {
            'latitude': gps_lat_raw,
            'longitude': gps_lon_raw,
            'altitude': gps_alt_raw,
            # Distance from home
            'home_distance': np.linalg.norm(true_pos_ned[:2]),
        }

        # FIXED: Debug output (occasional)
        if self.measurement_count <= 5 or self.measurement_count % 50 == 0:
            print(
                f"GPS #{self.measurement_count}: Lat={gps_lat_raw:.6f}°, Lon={gps_lon_raw:.6f}°, Alt={gps_alt_raw:.1f}m")
            print(
                f"  Distance from home: {geodetic_raw['home_distance']:.1f}m")
            print(f"  NED: {gps_pos_ned}")

        if self.measurement_count == 1:  # First measurement
            print("\n=== GPS DEBUG ===")
            print(
                f"Home Position: {self.home_lat:.6f}°, {self.home_lon:.6f}°, {self.home_alt:.1f}m")
            print(f"True NED from quad: {true_pos_ned}")
            print(
                f"True Geodetic calculated: {true_lat:.6f}°, {true_lon:.6f}°, {true_alt:.1f}m")
            print(f"GPS NED output: {gps_pos_ned}")
            print(f"Difference: {gps_pos_ned - true_pos_ned}")
            print("================\n")

        return gps_pos_ned, gps_vel_ned, geodetic_raw

    def get_reference_position(self):
        """Get home/reference position"""
        return {
            'latitude': self.home_lat,
            'longitude': self.home_lon,
            'altitude': self.home_alt
        }

    def convert_ned_to_geodetic(self, ned_pos):
        """
        Convert NED position to geodetic coordinates

        Args:
            ned_pos: [north, east, down] in meters from home position

        Returns:
            dict: {'latitude': deg, 'longitude': deg, 'altitude': m}
        """
        north, east, down = ned_pos

        latitude = self.home_lat + north / self.deg_to_m_lat
        longitude = self.home_lon + east / self.deg_to_m_lon
        altitude = self.home_alt - down

        return {
            'latitude': latitude,
            'longitude': longitude,
            'altitude': altitude
        }

    def convert_geodetic_to_ned(self, lat_deg, lon_deg, alt_m):
        """
        Convert geodetic coordinates to NED position

        Args:
            latitude: degrees
            longitude: degrees
            altitude: meters MSL

        Returns:
            np.array: [north, east, down] in meters from home
        """
        north = (lat_deg - self.home_lat) * self.deg_to_m_lat
        east = (lon_deg - self.home_lon) * self.deg_to_m_lon
        down = -(alt_m - self.home_alt)

        return np.array([north, east, down])


class AltitudeSensor:
    def __init__(self, sensor_type="baro"):
        self.sensor_type = sensor_type

        if sensor_type == "baro":
            self.noise_std = 0.2  # m
            self.freq = 20  # Hz
        elif sensor_type == "lidar":
            self.noise_std = 0.05  # m
            self.freq = 40  # Hz

        self.dt = 1.0/self.freq
        self.last_update = 0

    def measure(self, quad, t):
        # Only provide measurement at specific frequency
        if t - self.last_update < self.dt:
            return None

        self.last_update = t

        # Get true altitude (negative z in NED, positive z in ENU)
        if (config.orient == "NED"):
            alt_true = -quad.pos[2]
        else:
            alt_true = quad.pos[2]

        # Add noise
        alt_noise = np.random.normal(0, self.noise_std)

        # Final altitude reading
        alt_measured = alt_true + alt_noise

        return alt_measured


class MagnetometerSensor:
    def __init__(self):
        # Noise parameters
        self.noise_std = 0.05  # Gauss

        # Magnetic field reference (north direction)
        self.mag_reference = np.array([1.0, 0.0, 0.0])

        # Hard and soft iron effects
        self.hard_iron = np.array([0.1, -0.05, 0.03])
        self.soft_iron = np.eye(3) + np.random.normal(0, 0.05, (3, 3))

        # Sampling frequency
        self.freq = 50  # Hz
        self.dt = 1.0/self.freq
        self.last_update = 0

    def measure(self, quad, t):
        # Only provide measurement at specific frequency
        if t - self.last_update < self.dt:
            return None

        self.last_update = t

        # Get true attitude
        R = quad.dcm  # Rotation matrix from body to inertial frame

        # Calculate magnetic field in body frame
        mag_body_true = R.T @ self.mag_reference

        # Apply soft and hard iron effects
        mag_body_distorted = self.soft_iron @ mag_body_true + self.hard_iron

        # Add noise
        mag_noise = np.random.normal(0, self.noise_std, 3)

        # Final magnetometer reading
        mag_measured = mag_body_distorted + mag_noise

        return mag_measured
