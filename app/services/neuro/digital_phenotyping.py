"""
Digital Phenotyping for Mental Health

Passive data collection and analysis from smartphones, wearables, and environmental sensors.
Captures real-world behavior patterns for psychiatric assessment.

Key Metrics:
- Mobility patterns (GPS, accelerometer)
- Social interactions (calls, texts, app usage)
- Sleep patterns (actigraphy)
- Screen time and phone usage
- Voice/speech patterns
- Physiological signals (heart rate, HRV)

Applications:
- Depression monitoring
- Anxiety detection
- Bipolar episode prediction
- Schizophrenia relapse prevention
"""

import logging
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
import numpy as np
from pathlib import Path

try:
    import pandas as pd
    HAS_PANDAS = True
except ImportError:
    HAS_PANDAS = False

logger = logging.getLogger(__name__)


class DataStream(str, Enum):
    """Available data streams for digital phenotyping"""
    GPS = "gps"
    ACCELEROMETER = "accelerometer"
    CALLS = "calls"
    SMS = "sms"
    SCREEN = "screen"
    SLEEP = "sleep"
    HEART_RATE = "heart_rate"
    STEPS = "steps"
    APP_USAGE = "app_usage"
    VOICE = "voice"


@dataclass
class MobilityMetrics:
    """Mobility-derived features"""
    home_stay_percent: float  # % time at home
    significant_locations: int  # Number of frequented places
    total_distance_km: float  # Total distance traveled
    radius_of_gyration: float  # Spatial spread (meters)
    entropy: float  # Location entropy (unpredictability)
    circadian_movement: float  # Regularity of daily patterns


@dataclass
class SocialMetrics:
    """Social interaction features"""
    call_count: int
    call_duration_minutes: float
    sms_count: int
    unique_contacts: int
    response_latency_minutes: float  # How quickly responds
    social_diversity: float  # Entropy of contact distribution


@dataclass
class PhoneUsageMetrics:
    """Phone usage patterns"""
    screen_time_hours: float
    unlock_count: int
    session_duration_minutes: float
    app_switches: int
    notification_interactions: int


@dataclass
class SleepMetrics:
    """Sleep-derived features"""
    sleep_duration_hours: float
    sleep_onset_time: str  # HH:MM
    wake_time: str  # HH:MM
    sleep_efficiency: float  # % of in-bed time asleep
    wake_episodes: int
    circadian_rhythm_strength: float


@dataclass
class DigitalPhenotype:
    """Complete digital phenotype for a time period"""
    user_id: str
    start_date: datetime
    end_date: datetime
    
    mobility: Optional[MobilityMetrics] = None
    social: Optional[SocialMetrics] = None
    phone_usage: Optional[PhoneUsageMetrics] = None
    sleep: Optional[SleepMetrics] = None
    
    # Physiological
    mean_heart_rate: Optional[float] = None
    hrv_rmssd: Optional[float] = None  # Heart rate variability
    steps_per_day: Optional[int] = None
    
    # Meta
    data_completeness: Dict[str, float] = field(default_factory=dict)


class DigitalPhenotyping:
    """
    Digital phenotyping system for passive behavior monitoring.
    
    Processes multimodal data streams to extract clinically relevant features.
    """
    
    def __init__(self):
        if not HAS_PANDAS:
            raise ImportError("pandas required. Install with: pip install pandas")
        
        logger.info("Digital Phenotyping system initialized")
    
    def compute_mobility_features(
        self,
        gps_data: pd.DataFrame,
        home_location: Optional[Tuple[float, float]] = None
    ) -> MobilityMetrics:
        """
        Compute mobility features from GPS data.
        
        Args:
            gps_data: DataFrame with columns [timestamp, latitude, longitude, accuracy]
            home_location: (lat, lon) of home if known
        
        Returns:
            MobilityMetrics
        """
        logger.debug("Computing mobility features...")
        
        # Cluster locations to find significant places
        from sklearn.cluster import DBSCAN
        
        coords = gps_data[['latitude', 'longitude']].values
        
        # Clustering (100m radius)
        clustering = DBSCAN(eps=0.001, min_samples=5).fit(coords)
        n_significant_locations = len(set(clustering.labels_)) - (1 if -1 in clustering.labels_ else 0)
        
        # Home stay
        if home_location:
            home_lat, home_lon = home_location
            distances = np.sqrt(
                (gps_data['latitude'] - home_lat)**2 + 
                (gps_data['longitude'] - home_lon)**2
            )
            home_points = (distances < 0.001).sum()  # ~100m radius
            home_stay_percent = home_points / len(gps_data)
        else:
            home_stay_percent = 0.0
        
        # Total distance
        def haversine_distance(lat1, lon1, lat2, lon2):
            """Distance in km"""
            R = 6371  # Earth radius in km
            dlat = np.radians(lat2 - lat1)
            dlon = np.radians(lon2 - lon1)
            a = np.sin(dlat/2)**2 + np.cos(np.radians(lat1)) * np.cos(np.radians(lat2)) * np.sin(dlon/2)**2
            return 2 * R * np.arcsin(np.sqrt(a))
        
        distances = []
        for i in range(1, len(gps_data)):
            d = haversine_distance(
                gps_data.iloc[i-1]['latitude'],
                gps_data.iloc[i-1]['longitude'],
                gps_data.iloc[i]['latitude'],
                gps_data.iloc[i]['longitude']
            )
            distances.append(d)
        
        total_distance_km = sum(distances)
        
        # Radius of gyration
        center_lat = coords[:, 0].mean()
        center_lon = coords[:, 1].mean()
        radius_of_gyration = np.sqrt(
            ((coords[:, 0] - center_lat)**2 + (coords[:, 1] - center_lon)**2).mean()
        ) * 111000  # Convert to meters
        
        # Location entropy
        _, counts = np.unique(clustering.labels_[clustering.labels_ >= 0], return_counts=True)
        probs = counts / counts.sum()
        entropy = -np.sum(probs * np.log(probs + 1e-10))
        
        # Circadian movement (simplified)
        gps_data['hour'] = pd.to_datetime(gps_data['timestamp']).dt.hour
        hourly_variance = gps_data.groupby('hour').size().var()
        circadian_movement = 1.0 / (1.0 + hourly_variance)  # Normalize
        
        return MobilityMetrics(
            home_stay_percent=home_stay_percent,
            significant_locations=n_significant_locations,
            total_distance_km=total_distance_km,
            radius_of_gyration=radius_of_gyration,
            entropy=entropy,
            circadian_movement=circadian_movement
        )
    
    def compute_social_features(
        self,
        call_data: pd.DataFrame,
        sms_data: pd.DataFrame
    ) -> SocialMetrics:
        """
        Compute social interaction features.
        
        Args:
            call_data: DataFrame [timestamp, contact_id, duration_sec, direction]
            sms_data: DataFrame [timestamp, contact_id, direction]
        
        Returns:
            SocialMetrics
        """
        logger.debug("Computing social features...")
        
        # Calls
        call_count = len(call_data)
        call_duration_minutes = call_data['duration_sec'].sum() / 60.0
        
        # SMS
        sms_count = len(sms_data)
        
        # Unique contacts
        all_contacts = set(call_data['contact_id']).union(set(sms_data['contact_id']))
        unique_contacts = len(all_contacts)
        
        # Response latency (simplified: time between received and sent)
        received = sms_data[sms_data['direction'] == 'incoming']
        sent = sms_data[sms_data['direction'] == 'outgoing']
        
        if len(received) > 0 and len(sent) > 0:
            # Compute average gap
            received_times = pd.to_datetime(received['timestamp'])
            sent_times = pd.to_datetime(sent['timestamp'])
            # Simplified calculation
            response_latency_minutes = 30.0  # Placeholder
        else:
            response_latency_minutes = 0.0
        
        # Social diversity (entropy of contact distribution)
        contact_counts = pd.concat([
            call_data['contact_id'],
            sms_data['contact_id']
        ]).value_counts()
        
        probs = contact_counts / contact_counts.sum()
        social_diversity = -np.sum(probs * np.log(probs + 1e-10))
        
        return SocialMetrics(
            call_count=call_count,
            call_duration_minutes=call_duration_minutes,
            sms_count=sms_count,
            unique_contacts=unique_contacts,
            response_latency_minutes=response_latency_minutes,
            social_diversity=social_diversity
        )
    
    def compute_phone_usage_features(
        self,
        screen_data: pd.DataFrame,
        app_usage_data: pd.DataFrame
    ) -> PhoneUsageMetrics:
        """
        Compute phone usage patterns.
        
        Args:
            screen_data: DataFrame [timestamp, event (on/off)]
            app_usage_data: DataFrame [timestamp, app_name, duration_sec]
        
        Returns:
            PhoneUsageMetrics
        """
        logger.debug("Computing phone usage features...")
        
        # Screen time
        screen_on = screen_data[screen_data['event'] == 'on']
        screen_off = screen_data[screen_data['event'] == 'off']
        
        total_screen_time = 0
        for on_time in pd.to_datetime(screen_on['timestamp']):
            # Find next off
            off_times = pd.to_datetime(screen_off['timestamp'])
            next_off = off_times[off_times > on_time]
            if len(next_off) > 0:
                duration = (next_off.iloc[0] - on_time).total_seconds()
                total_screen_time += duration
        
        screen_time_hours = total_screen_time / 3600.0
        
        # Unlock count
        unlock_count = len(screen_on)
        
        # Average session duration
        session_duration_minutes = (total_screen_time / unlock_count / 60.0) if unlock_count > 0 else 0
        
        # App switches
        app_switches = len(app_usage_data) - 1
        
        # Notifications (placeholder)
        notification_interactions = 0
        
        return PhoneUsageMetrics(
            screen_time_hours=screen_time_hours,
            unlock_count=unlock_count,
            session_duration_minutes=session_duration_minutes,
            app_switches=app_switches,
            notification_interactions=notification_interactions
        )
    
    def compute_sleep_features(
        self,
        actigraphy_data: pd.DataFrame
    ) -> SleepMetrics:
        """
        Compute sleep features from actigraphy.
        
        Args:
            actigraphy_data: DataFrame [timestamp, activity_level]
        
        Returns:
            SleepMetrics
        """
        logger.debug("Computing sleep features...")
        
        # Simple sleep detection: low activity for extended periods
        actigraphy_data['datetime'] = pd.to_datetime(actigraphy_data['timestamp'])
        actigraphy_data['hour'] = actigraphy_data['datetime'].dt.hour
        
        # Detect sleep period (typically 22:00 - 08:00)
        night_hours = actigraphy_data[
            (actigraphy_data['hour'] >= 22) | (actigraphy_data['hour'] <= 8)
        ]
        
        # Sleep duration (simplified)
        sleep_duration_hours = len(night_hours) / 60.0  # Assuming 1-min sampling
        
        # Sleep onset/wake times
        sleep_onset_time = "23:00"  # Placeholder
        wake_time = "07:00"  # Placeholder
        
        # Sleep efficiency
        total_activity = night_hours['activity_level'].sum()
        max_activity = night_hours['activity_level'].max() * len(night_hours)
        sleep_efficiency = 1.0 - (total_activity / (max_activity + 1e-10))
        
        # Wake episodes
        wake_episodes = (night_hours['activity_level'] > 50).sum()
        
        # Circadian rhythm strength
        hourly_avg = actigraphy_data.groupby('hour')['activity_level'].mean()
        circadian_rhythm_strength = hourly_avg.std() / (hourly_avg.mean() + 1e-10)
        
        return SleepMetrics(
            sleep_duration_hours=sleep_duration_hours,
            sleep_onset_time=sleep_onset_time,
            wake_time=wake_time,
            sleep_efficiency=sleep_efficiency,
            wake_episodes=wake_episodes,
            circadian_rhythm_strength=circadian_rhythm_strength
        )
    
    def compute_digital_phenotype(
        self,
        user_id: str,
        start_date: datetime,
        end_date: datetime,
        data_streams: Dict[DataStream, pd.DataFrame]
    ) -> DigitalPhenotype:
        """
        Compute complete digital phenotype from all available data streams.
        
        Args:
            user_id: User identifier
            start_date: Start of phenotyping period
            end_date: End of phenotyping period
            data_streams: Dictionary mapping DataStream to DataFrame
        
        Returns:
            DigitalPhenotype
        """
        logger.info(f"Computing digital phenotype for user {user_id}")
        
        phenotype = DigitalPhenotype(
            user_id=user_id,
            start_date=start_date,
            end_date=end_date
        )
        
        # Mobility
        if DataStream.GPS in data_streams:
            phenotype.mobility = self.compute_mobility_features(
                data_streams[DataStream.GPS]
            )
            phenotype.data_completeness['mobility'] = 1.0
        
        # Social
        if DataStream.CALLS in data_streams and DataStream.SMS in data_streams:
            phenotype.social = self.compute_social_features(
                data_streams[DataStream.CALLS],
                data_streams[DataStream.SMS]
            )
            phenotype.data_completeness['social'] = 1.0
        
        # Phone usage
        if DataStream.SCREEN in data_streams and DataStream.APP_USAGE in data_streams:
            phenotype.phone_usage = self.compute_phone_usage_features(
                data_streams[DataStream.SCREEN],
                data_streams[DataStream.APP_USAGE]
            )
            phenotype.data_completeness['phone_usage'] = 1.0
        
        # Sleep
        if DataStream.ACCELEROMETER in data_streams:
            phenotype.sleep = self.compute_sleep_features(
                data_streams[DataStream.ACCELEROMETER]
            )
            phenotype.data_completeness['sleep'] = 1.0
        
        # Physiological
        if DataStream.HEART_RATE in data_streams:
            hr_data = data_streams[DataStream.HEART_RATE]
            phenotype.mean_heart_rate = hr_data['heart_rate'].mean()
            phenotype.data_completeness['heart_rate'] = 1.0
        
        if DataStream.STEPS in data_streams:
            steps_data = data_streams[DataStream.STEPS]
            phenotype.steps_per_day = steps_data['steps'].sum() / (
                (end_date - start_date).days + 1
            )
            phenotype.data_completeness['steps'] = 1.0
        
        logger.info(f"Digital phenotype computed with {len(phenotype.data_completeness)} streams")
        return phenotype
    
    def detect_anomalies(
        self,
        phenotype: DigitalPhenotype,
        baseline: DigitalPhenotype
    ) -> Dict[str, Any]:
        """
        Detect anomalies compared to baseline.
        
        Useful for:
        - Depression relapse detection (reduced mobility, social withdrawal)
        - Manic episode detection (increased activity, reduced sleep)
        - Anxiety spikes (increased phone usage, disturbed sleep)
        
        Args:
            phenotype: Current phenotype
            baseline: Baseline phenotype (e.g., 30-day average)
        
        Returns:
            Anomaly report
        """
        anomalies = {}
        
        # Mobility anomalies
        if phenotype.mobility and baseline.mobility:
            if phenotype.mobility.home_stay_percent > baseline.mobility.home_stay_percent * 1.5:
                anomalies['increased_isolation'] = True
            
            if phenotype.mobility.total_distance_km < baseline.mobility.total_distance_km * 0.5:
                anomalies['reduced_mobility'] = True
        
        # Social anomalies
        if phenotype.social and baseline.social:
            if phenotype.social.call_count < baseline.social.call_count * 0.5:
                anomalies['social_withdrawal'] = True
        
        # Sleep anomalies
        if phenotype.sleep and baseline.sleep:
            if abs(phenotype.sleep.sleep_duration_hours - baseline.sleep.sleep_duration_hours) > 2:
                anomalies['sleep_disturbance'] = True
        
        return anomalies


# Example usage
if __name__ == "__main__":
    dp = DigitalPhenotyping()
    
    # Example data (mock)
    gps_data = pd.DataFrame({
        'timestamp': pd.date_range('2024-01-01', periods=100, freq='30min'),
        'latitude': np.random.randn(100) * 0.01 + 37.7749,
        'longitude': np.random.randn(100) * 0.01 - 122.4194,
        'accuracy': np.random.uniform(10, 50, 100)
    })
    
    mobility = dp.compute_mobility_features(gps_data)
    print("\n--- Mobility Features ---")
    print(f"Significant locations: {mobility.significant_locations}")
    print(f"Total distance: {mobility.total_distance_km:.2f} km")
    print(f"Radius of gyration: {mobility.radius_of_gyration:.0f} m")

