import config
import numpy as np
from numpy import sin, cos, pi, sqrt, arctan2, arcsin, arccos
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


class ECEFTransform:
    """
    Earth-Centered Earth-Fixed (ECEF) Transformation
    - Highest accuracy (ellipsoidal model)
    - Global validity
    - Used for high-precision GPS applications
    """

    def __init__(self, ref_lat_deg, ref_lon_deg, ref_alt=0):
        self.ref_lat = np.deg2rad(ref_lat_deg)
        self.ref_lon = np.deg2rad(ref_lon_deg)
        self.ref_alt = ref_alt

        # Store original degrees for reference
        self.ref_lat_deg = ref_lat_deg
        self.ref_lon_deg = ref_lon_deg

        # WGS84 ellipsoid parameters
        self.a = 6378137.0           # Semi-major axis (m)
        self.f = 1/298.257223563     # Flattening
        self.e2 = 2*self.f - self.f**2  # First eccentricity squared

        # Reference ECEF coordinates
        self.ref_ecef = self._geodetic_to_ecef(
            self.ref_lat, self.ref_lon, self.ref_alt)

        # ECEF to NED transformation matrix at reference point
        self.T_ecef_to_ned = self._compute_transform_matrix()

        self.name = "ECEF (High Precision)"

    def _geodetic_to_ecef(self, lat, lon, alt):
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

    def _ecef_to_geodetic(self, ecef):
        """
        Convert ECEF coordinates to geodetic (iterative method)

        Args:
            ecef: ECEF coordinates [x, y, z] (m)

        Returns:
            [lat, lon, alt] in radians, radians, meters
        """
        x, y, z = ecef

        # Longitude (direct calculation)
        lon = arctan2(y, x)

        # Iterative solution for latitude and altitude
        p = sqrt(x**2 + y**2)
        lat = arctan2(z, p * (1 - self.e2))

        # Iterate to convergence (usually 2-3 iterations)
        for _ in range(5):
            N = self.a / sqrt(1 - self.e2 * sin(lat)**2)
            alt = p / cos(lat) - N
            lat_new = arctan2(z, p * (1 - self.e2 * N / (N + alt)))

            if abs(lat_new - lat) < 1e-12:
                break
            lat = lat_new

        return np.array([lat, lon, alt])

    def _compute_transform_matrix(self):
        """
        Compute transformation matrix from ECEF to NED at reference point
        """
        sin_lat, cos_lat = sin(self.ref_lat), cos(self.ref_lat)
        sin_lon, cos_lon = sin(self.ref_lon), cos(self.ref_lon)

        return np.array([
            [-sin_lat * cos_lon, -sin_lat * sin_lon,  cos_lat],
            [-sin_lon,            cos_lon,            0],
            [-cos_lat * cos_lon, -cos_lat * sin_lon, -sin_lat]
        ])

    def geodetic_to_ned(self, lat_deg, lon_deg, alt):
        """
        Convert geodetic coordinates to local NED frame via ECEF

        Args:
            lat_deg: Latitude (degrees)
            lon_deg: Longitude (degrees)
            alt: Altitude (m)

        Returns:
            NED coordinates [north, east, down] (m)
        """
        lat = np.deg2rad(lat_deg)
        lon = np.deg2rad(lon_deg)

        # Convert to ECEF
        ecef = self._geodetic_to_ecef(lat, lon, alt)

        # Translate relative to reference
        ecef_rel = ecef - self.ref_ecef

        # Transform to NED
        ned = self.T_ecef_to_ned @ ecef_rel

        return ned

    def ned_to_geodetic(self, ned):
        """
        Convert local NED coordinates to geodetic via ECEF

        Args:
            ned: NED coordinates [north, east, down] (m)

        Returns:
            tuple: (lat_deg, lon_deg, alt) in degrees, degrees, meters
        """
        # Transform NED to ECEF relative
        ecef_rel = self.T_ecef_to_ned.T @ ned

        # Add reference ECEF
        ecef = ecef_rel + self.ref_ecef

        # Convert to geodetic
        lat, lon, alt = self._ecef_to_geodetic(ecef)

        return np.rad2deg(lat), np.rad2deg(lon), alt

    def get_reference_position(self):
        """Get reference position"""
        return {
            'latitude': self.ref_lat_deg,
            'longitude': self.ref_lon_deg,
            'altitude': self.ref_alt
        }


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
    GPS sensor using ECEF transformation with simplified noise model
    - ECEF transformation for high accuracy
    - Simple noise workflow from original implementation
    - Best balance of accuracy and simplicity
    """

    def __init__(self, home_lat=-7.250445, home_lon=112.768845, home_alt=10.0):
        """
        Initialize GPS sensor with ECEF transformation and simple noise model

        Args:
            home_lat: Home latitude in degrees (Surabaya)
            home_lon: Home longitude in degrees (Surabaya)  
            home_alt: Home altitude in meters MSL
        """
        # Home position (reference point)
        self.home_lat = home_lat  # degrees
        self.home_lon = home_lon  # degrees
        self.home_alt = home_alt  # meters MSL

        # Initialize ECEF transformer for high-precision coordinate conversion
        self.ecef_transform = ECEFTransform(home_lat, home_lon, home_alt)

        # Simple GPS noise parameters (from original implementation)
        self.pos_noise_std = 1.5     # meters horizontal
        # meters vertical (GPS altitude usually worse)
        self.alt_noise_std = 2.0
        self.vel_noise_std = 0.1     # m/s

        # GPS noise in coordinate domain (converted for realistic simulation)
        deg_to_m_lat = 111320.0
        deg_to_m_lon = 111320.0 * cos(np.deg2rad(home_lat))

        self.lat_noise_std = self.pos_noise_std / deg_to_m_lat  # degrees
        self.lon_noise_std = self.pos_noise_std / deg_to_m_lon  # degrees

        # GPS timing parameters
        self.freq = 5  # Hz (typical GPS update rate)
        self.dt = 1.0 / self.freq
        self.last_update = -999  # Force first measurement

        # Statistics tracking
        self.measurement_count = 0

        print(f"GPS Sensor (ECEF + Simple Noise) initialized:")
        print(
            f"  Home position: {home_lat:.6f}°, {home_lon:.6f}°, {home_alt:.1f}m MSL")
        print(
            f"  Position noise: ±{self.pos_noise_std}m horizontal, ±{self.alt_noise_std}m vertical")
        print(
            f"  Coordinate noise: ±{self.lat_noise_std*1e6:.1f}µ°lat, ±{self.lon_noise_std*1e6:.1f}µ°lon")
        print(f"  Transformation: ECEF (WGS84 ellipsoid)")
        print(f"  Update rate: {self.freq} Hz")

    def measure(self, quad, t):
        """
        GPS measurement using ECEF transformation with simple noise workflow

        Args:
            quad: Quadcopter object with .pos and .vel in NED frame
            t: Current time

        Returns:
            tuple: (pos_ned_measured, vel_ned_measured, geodetic_raw)
        """
        # Check timing (similar to original)
        if t - self.last_update < self.dt - 0.01:  # Small tolerance
            return None, None, None

        self.last_update = t
        self.measurement_count += 1

        # Get true position and velocity from simulation (NED frame)
        true_pos_ned = quad.pos.copy()
        true_vel_ned = quad.vel.copy()

        # STEP 1: Convert true NED to geodetic using ECEF transformation (HIGH ACCURACY)
        true_lat, true_lon, true_alt = self.ecef_transform.ned_to_geodetic(
            true_pos_ned)

        # STEP 2: Apply simple GPS noise in coordinate domain (ORIGINAL WORKFLOW)
        lat_noise = np.random.normal(0, self.lat_noise_std)
        lon_noise = np.random.normal(0, self.lon_noise_std)
        alt_noise = np.random.normal(0, self.alt_noise_std)

        # Noisy GPS coordinates (what GPS receiver outputs)
        gps_lat_raw = true_lat + lat_noise
        gps_lon_raw = true_lon + lon_noise
        gps_alt_raw = true_alt + alt_noise

        # STEP 3: Convert noisy GPS coordinates back to NED using ECEF (HIGH ACCURACY)
        gps_pos_ned = self.ecef_transform.geodetic_to_ned(
            gps_lat_raw, gps_lon_raw, gps_alt_raw)

        # STEP 4: Add velocity noise directly in NED frame (ORIGINAL WORKFLOW)
        vel_noise_ned = np.random.normal(0, self.vel_noise_std, 3)
        gps_vel_ned = true_vel_ned + vel_noise_ned

        # STEP 5: Package raw geodetic data (simplified metadata)
        horizontal_distance = np.linalg.norm(true_pos_ned[:2])

        geodetic_raw = {
            'latitude': gps_lat_raw,
            'longitude': gps_lon_raw,
            'altitude': gps_alt_raw,
            'home_distance': horizontal_distance,
        }

        # Debug output (occasional)
        if self.measurement_count <= 5 or self.measurement_count % 50 == 0:
            print(f"GPS #{self.measurement_count}: "
                  f"Lat={gps_lat_raw:.6f}°, Lon={gps_lon_raw:.6f}°, Alt={gps_alt_raw:.1f}m")
            print(f"  Distance from home: {horizontal_distance:.1f}m")
            print(
                f"  NED position: [{gps_pos_ned[0]:.2f}, {gps_pos_ned[1]:.2f}, {gps_pos_ned[2]:.2f}]")

        if self.measurement_count == 1:  # First measurement detailed debug
            print("\n=== GPS ECEF TRANSFORMATION DEBUG ===")
            print(
                f"Home Position: {self.home_lat:.6f}°, {self.home_lon:.6f}°, {self.home_alt:.1f}m")
            print(
                f"True NED from quad: [{true_pos_ned[0]:.3f}, {true_pos_ned[1]:.3f}, {true_pos_ned[2]:.3f}]")
            print(
                f"True Geodetic via ECEF: {true_lat:.6f}°, {true_lon:.6f}°, {true_alt:.1f}m")
            print(
                f"Noisy Geodetic: {gps_lat_raw:.6f}°, {gps_lon_raw:.6f}°, {gps_alt_raw:.1f}m")
            print(
                f"GPS NED output: [{gps_pos_ned[0]:.3f}, {gps_pos_ned[1]:.3f}, {gps_pos_ned[2]:.3f}]")
            position_error = np.linalg.norm(gps_pos_ned - true_pos_ned)
            print(f"Total position error: {position_error:.3f}m")
            print("=======================================\n")

        return gps_pos_ned, gps_vel_ned, geodetic_raw

    def get_reference_position(self):
        """Get home/reference position"""
        return self.ecef_transform.get_reference_position()

    def convert_ned_to_geodetic(self, ned_pos):
        """
        Convert NED position to geodetic coordinates using ECEF

        Args:
            ned_pos: [north, east, down] in meters from home position

        Returns:
            dict: {'latitude': deg, 'longitude': deg, 'altitude': m}
        """
        lat, lon, alt = self.ecef_transform.ned_to_geodetic(ned_pos)
        return {
            'latitude': lat,
            'longitude': lon,
            'altitude': alt
        }

    def convert_geodetic_to_ned(self, lat_deg, lon_deg, alt_m):
        """
        Convert geodetic coordinates to NED position using ECEF

        Args:
            lat_deg: Latitude (degrees)
            lon_deg: Longitude (degrees)
            alt_m: Altitude (meters MSL)

        Returns:
            np.array: [north, east, down] in meters from home
        """
        return self.ecef_transform.geodetic_to_ned(lat_deg, lon_deg, alt_m)

    def get_transformation_accuracy_info(self):
        """
        Get information about transformation accuracy

        Returns:
            dict: Information about the ECEF transformation
        """
        return {
            'method': 'ECEF (Earth-Centered Earth-Fixed)',
            'ellipsoid': 'WGS84',
            'accuracy': 'Sub-meter globally',
            'valid_range': 'Global',
            'reference_position': self.get_reference_position(),
            'noise_model': {
                'horizontal_std': self.pos_noise_std,
                'vertical_std': self.alt_noise_std,
                'velocity_std': self.vel_noise_std,
                'workflow': 'Simple white noise (original style)'
            }
        }


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
