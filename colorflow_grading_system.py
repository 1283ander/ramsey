"""
Colorflow Dynamic Grading System for Drone Footage

This module provides advanced color grading capabilities for drone footage based on
telemetry data extracted from SRT files. It analyzes color temperature, exposure,
and other camera settings to provide dynamic color correction recommendations.

Features:
- Color temperature analysis and correction
- Exposure compensation based on flight conditions
- Dynamic color grading presets
- Lighting condition detection
- Color mode optimization
- Real-time grading adjustments
"""

import re
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional, Union
import matplotlib.pyplot as plt
from matplotlib.colors import rgb_to_hsv, hsv_to_rgb
import json
import os
from dataclasses import dataclass
from enum import Enum

class ColorMode(Enum):
    """Available color modes from DJI drones"""
    DEFAULT = "default"
    CINELIKE = "d_cinelike"
    LOG = "d_log"
    NORMAL = "normal"
    VIVID = "vivid"

class LightingCondition(Enum):
    """Detected lighting conditions"""
    SUNRISE = "sunrise"
    SUNSET = "sunset"
    GOLDEN_HOUR = "golden_hour"
    BLUE_HOUR = "blue_hour"
    DAYLIGHT = "daylight"
    OVERCAST = "overcast"
    NIGHT = "night"
    INDOOR = "indoor"

@dataclass
class ColorGradingData:
    """Container for color grading metadata from SRT files"""
    timestamp: datetime
    iso: int
    shutter_speed: float
    f_number: float
    exposure_value: float
    color_temperature: int
    color_mode: str
    focal_length: int
    latitude: float
    longitude: float
    altitude: float
    relative_altitude: float

@dataclass
class GradingPreset:
    """Color grading preset configuration"""
    name: str
    description: str
    color_temperature_offset: int
    saturation_multiplier: float
    contrast_multiplier: float
    brightness_offset: float
    highlights_rolloff: float
    shadows_lift: float
    vibrance: float
    clarity: float

class ColorflowGradingSystem:
    """
    Main class for the Colorflow Dynamic Grading System
    """
    
    def __init__(self):
        self.grading_presets = self._initialize_presets()
        self.lighting_conditions = self._initialize_lighting_conditions()
        
    def _initialize_presets(self) -> Dict[str, GradingPreset]:
        """Initialize color grading presets for different scenarios"""
        return {
            "cinematic": GradingPreset(
                name="Cinematic",
                description="Film-like color grading with enhanced contrast and saturation",
                color_temperature_offset=0,
                saturation_multiplier=1.1,
                contrast_multiplier=1.2,
                brightness_offset=0.05,
                highlights_rolloff=0.8,
                shadows_lift=0.1,
                vibrance=1.15,
                clarity=1.1
            ),
            "natural": GradingPreset(
                name="Natural",
                description="Natural color reproduction with minimal processing",
                color_temperature_offset=0,
                saturation_multiplier=1.0,
                contrast_multiplier=1.0,
                brightness_offset=0.0,
                highlights_rolloff=1.0,
                shadows_lift=0.0,
                vibrance=1.0,
                clarity=1.0
            ),
            "vibrant": GradingPreset(
                name="Vibrant",
                description="High saturation and contrast for vivid imagery",
                color_temperature_offset=100,
                saturation_multiplier=1.3,
                contrast_multiplier=1.15,
                brightness_offset=0.1,
                highlights_rolloff=0.7,
                shadows_lift=0.05,
                vibrance=1.25,
                clarity=1.2
            ),
            "moody": GradingPreset(
                name="Moody",
                description="Dark, atmospheric grading with cool tones",
                color_temperature_offset=-200,
                saturation_multiplier=0.9,
                contrast_multiplier=1.3,
                brightness_offset=-0.1,
                highlights_rolloff=0.6,
                shadows_lift=0.2,
                vibrance=0.8,
                clarity=1.15
            ),
            "warm": GradingPreset(
                name="Warm",
                description="Warm, golden hour inspired grading",
                color_temperature_offset=300,
                saturation_multiplier=1.15,
                contrast_multiplier=1.1,
                brightness_offset=0.05,
                highlights_rolloff=0.8,
                shadows_lift=0.05,
                vibrance=1.1,
                clarity=1.05
            )
        }
    
    def _initialize_lighting_conditions(self) -> Dict[str, Dict]:
        """Initialize lighting condition detection parameters"""
        return {
            "sunrise": {
                "ct_range": (2000, 3500),
                "time_range": (5, 8),
                "description": "Warm, golden light during sunrise"
            },
            "sunset": {
                "ct_range": (2000, 3500),
                "time_range": (17, 20),
                "description": "Warm, golden light during sunset"
            },
            "golden_hour": {
                "ct_range": (3000, 4500),
                "time_range": (6, 8),
                "description": "Soft, warm light during golden hour"
            },
            "blue_hour": {
                "ct_range": (6000, 10000),
                "time_range": (19, 21),
                "description": "Cool, blue light during blue hour"
            },
            "daylight": {
                "ct_range": (5000, 6500),
                "time_range": (9, 16),
                "description": "Natural daylight"
            },
            "overcast": {
                "ct_range": (6500, 8000),
                "time_range": (9, 16),
                "description": "Cool, flat light on overcast days"
            },
            "night": {
                "ct_range": (2000, 4000),
                "time_range": (21, 5),
                "description": "Artificial lighting at night"
            }
        }

    def parse_srt_color_data(self, srt_file: str) -> List[ColorGradingData]:
        """
        Parse SRT file to extract color grading metadata
        
        Args:
            srt_file: Path to SRT file
            
        Returns:
            List of ColorGradingData objects
        """
        color_data = []
        
        with open(srt_file, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Enhanced regex pattern to capture all camera settings
        pattern = r'\[iso\s*:\s*(\d+)\]\s*\[shutter\s*:\s*1/([0-9.]+)\]\s*\[fnum\s*:\s*(\d+)\]\s*\[ev\s*:\s*([+-]?[0-9.]+)\]\s*\[ct\s*:\s*(\d+)\]\s*\[color_md\s*:\s*(\w+)\]\s*\[focal_len\s*:\s*(\d+)\].*?\[latitude:\s*([+-]?[0-9.]+)\]\s*\[longitude:\s*([+-]?[0-9.]+)\]\s*\[rel_alt:\s*([+-]?[0-9.]+)\s*abs_alt:\s*([+-]?[0-9.]+)\]'
        
        matches = re.findall(pattern, content)
        
        # Extract timestamps
        timestamp_pattern = r'(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}\.\d{3})'
        timestamps = re.findall(timestamp_pattern, content)
        
        for i, match in enumerate(matches):
            if i < len(timestamps):
                try:
                    timestamp = datetime.strptime(timestamps[i], '%Y-%m-%d %H:%M:%S.%f')
                    
                    color_data.append(ColorGradingData(
                        timestamp=timestamp,
                        iso=int(match[0]),
                        shutter_speed=float(match[1]),
                        f_number=float(match[2]) / 100,  # Convert from integer representation
                        exposure_value=float(match[3]),
                        color_temperature=int(match[4]),
                        color_mode=match[5],
                        focal_length=int(match[6]),
                        latitude=float(match[7]),
                        longitude=float(match[8]),
                        altitude=float(match[10]),
                        relative_altitude=float(match[9])
                    ))
                except (ValueError, IndexError) as e:
                    print(f"Error parsing color data: {e}")
                    continue
        
        return color_data

    def detect_lighting_condition(self, color_data: List[ColorGradingData]) -> LightingCondition:
        """
        Detect lighting condition based on color temperature and time of day
        
        Args:
            color_data: List of color grading data points
            
        Returns:
            Detected lighting condition
        """
        if not color_data:
            return LightingCondition.DAYLIGHT
        
        # Calculate average color temperature
        avg_ct = np.mean([data.color_temperature for data in color_data])
        
        # Get time of day from first timestamp
        hour = color_data[0].timestamp.hour
        
        # Determine lighting condition based on CT and time
        if 2000 <= avg_ct <= 3500:
            if 5 <= hour <= 8:
                return LightingCondition.SUNRISE
            elif 17 <= hour <= 20:
                return LightingCondition.SUNSET
            elif 21 <= hour or hour <= 5:
                return LightingCondition.NIGHT
            else:
                return LightingCondition.GOLDEN_HOUR
        elif 3000 <= avg_ct <= 4500:
            if 6 <= hour <= 8 or 17 <= hour <= 19:
                return LightingCondition.GOLDEN_HOUR
            else:
                return LightingCondition.DAYLIGHT
        elif 6000 <= avg_ct <= 10000:
            if 19 <= hour <= 21:
                return LightingCondition.BLUE_HOUR
            else:
                return LightingCondition.OVERCAST
        elif avg_ct > 8000:
            return LightingCondition.OVERCAST
        else:
            return LightingCondition.DAYLIGHT

    def analyze_color_temperature_trends(self, color_data: List[ColorGradingData]) -> Dict:
        """
        Analyze color temperature trends throughout the flight
        
        Args:
            color_data: List of color grading data points
            
        Returns:
            Dictionary with analysis results
        """
        if not color_data:
            return {}
        
        cts = [data.color_temperature for data in color_data]
        timestamps = [data.timestamp for data in color_data]
        
        # Calculate statistics
        analysis = {
            'min_ct': min(cts),
            'max_ct': max(cts),
            'avg_ct': np.mean(cts),
            'std_ct': np.std(cts),
            'ct_range': max(cts) - min(cts),
            'ct_trend': self._calculate_trend(cts),
            'significant_changes': self._find_ct_changes(cts, threshold=200),
            'time_series': list(zip(timestamps, cts))
        }
        
        return analysis

    def _calculate_trend(self, values: List[float]) -> str:
        """Calculate trend direction of a series of values"""
        if len(values) < 2:
            return "stable"
        
        # Simple linear trend calculation
        x = np.arange(len(values))
        slope = np.polyfit(x, values, 1)[0]
        
        if slope > 50:
            return "increasing"
        elif slope < -50:
            return "decreasing"
        else:
            return "stable"

    def _find_ct_changes(self, cts: List[int], threshold: int = 200) -> List[Dict]:
        """Find significant color temperature changes"""
        changes = []
        
        for i in range(1, len(cts)):
            change = abs(cts[i] - cts[i-1])
            if change > threshold:
                changes.append({
                    'index': i,
                    'previous_ct': cts[i-1],
                    'current_ct': cts[i],
                    'change': change,
                    'direction': 'warmer' if cts[i] > cts[i-1] else 'cooler'
                })
        
        return changes

    def recommend_grading_preset(self, color_data: List[ColorGradingData], 
                                lighting_condition: LightingCondition) -> str:
        """
        Recommend color grading preset based on flight data and lighting conditions
        
        Args:
            color_data: List of color grading data points
            lighting_condition: Detected lighting condition
            
        Returns:
            Recommended preset name
        """
        if not color_data:
            return "natural"
        
        # Analyze the data
        ct_analysis = self.analyze_color_temperature_trends(color_data)
        avg_ct = ct_analysis.get('avg_ct', 5500)
        ct_range = ct_analysis.get('ct_range', 0)
        
        # Decision logic based on lighting condition and data
        if lighting_condition == LightingCondition.SUNRISE or lighting_condition == LightingCondition.SUNSET:
            return "warm"
        elif lighting_condition == LightingCondition.BLUE_HOUR:
            return "moody"
        elif lighting_condition == LightingCondition.OVERCAST:
            return "vibrant"
        elif lighting_condition == LightingCondition.NIGHT:
            return "moody"
        elif ct_range > 1000:  # High variation in color temperature
            return "cinematic"
        elif avg_ct < 4000:  # Warm lighting
            return "warm"
        elif avg_ct > 7000:  # Cool lighting
            return "vibrant"
        else:
            return "natural"

    def generate_dynamic_grading_curve(self, color_data: List[ColorGradingData], 
                                     preset: str) -> List[Dict]:
        """
        Generate dynamic color grading curve based on flight data
        
        Args:
            color_data: List of color grading data points
            preset: Color grading preset to use
            
        Returns:
            List of grading adjustments over time
        """
        if not color_data:
            return []
        
        base_preset = self.grading_presets[preset]
        grading_curve = []
        
        for i, data in enumerate(color_data):
            # Calculate dynamic adjustments based on current conditions
            ct_adjustment = self._calculate_ct_adjustment(data.color_temperature)
            exposure_adjustment = self._calculate_exposure_adjustment(data.exposure_value, data.iso)
            altitude_adjustment = self._calculate_altitude_adjustment(data.altitude)
            
            # Apply adjustments to base preset
            dynamic_grading = {
                'timestamp': data.timestamp,
                'color_temperature_offset': base_preset.color_temperature_offset + ct_adjustment,
                'saturation_multiplier': base_preset.saturation_multiplier * (1 + altitude_adjustment * 0.1),
                'contrast_multiplier': base_preset.contrast_multiplier * (1 + exposure_adjustment * 0.05),
                'brightness_offset': base_preset.brightness_offset + exposure_adjustment * 0.1,
                'highlights_rolloff': base_preset.highlights_rolloff,
                'shadows_lift': base_preset.shadows_lift,
                'vibrance': base_preset.vibrance,
                'clarity': base_preset.clarity,
                'iso': data.iso,
                'shutter_speed': data.shutter_speed,
                'f_number': data.f_number,
                'exposure_value': data.exposure_value,
                'original_ct': data.color_temperature,
                'adjusted_ct': data.color_temperature + ct_adjustment
            }
            
            grading_curve.append(dynamic_grading)
        
        return grading_curve

    def _calculate_ct_adjustment(self, current_ct: int) -> int:
        """Calculate color temperature adjustment based on current CT"""
        # Target neutral color temperature
        target_ct = 5500
        
        # Calculate adjustment (limit to reasonable range)
        adjustment = target_ct - current_ct
        return max(-500, min(500, adjustment))

    def _calculate_exposure_adjustment(self, ev: float, iso: int) -> float:
        """Calculate exposure adjustment based on EV and ISO"""
        # Normalize EV to 0-1 range
        ev_normalized = (ev + 3) / 6  # Assuming EV range from -3 to +3
        
        # Calculate adjustment based on ISO
        iso_factor = min(1.0, iso / 1600)  # Normalize ISO
        
        return (ev_normalized - 0.5) * iso_factor

    def _calculate_altitude_adjustment(self, altitude: float) -> float:
        """Calculate adjustment based on altitude"""
        # Higher altitudes may need different grading
        if altitude > 200:
            return 0.1  # Slight increase in saturation for high altitude
        elif altitude < 50:
            return -0.05  # Slight decrease for low altitude
        else:
            return 0.0

    def create_grading_visualization(self, grading_curve: List[Dict], 
                                   output_file: str = "colorflow_grading_analysis.png"):
        """
        Create visualization of color grading analysis
        
        Args:
            grading_curve: Dynamic grading curve data
            output_file: Output file path for visualization
        """
        if not grading_curve:
            print("No grading data to visualize")
            return
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle('Colorflow Dynamic Grading Analysis', fontsize=16)
        
        timestamps = [point['timestamp'] for point in grading_curve]
        
        # Plot 1: Color Temperature over Time
        axes[0, 0].plot(timestamps, [point['original_ct'] for point in grading_curve], 
                       'b-', label='Original CT', alpha=0.7)
        axes[0, 0].plot(timestamps, [point['adjusted_ct'] for point in grading_curve], 
                       'r-', label='Adjusted CT', alpha=0.7)
        axes[0, 0].set_title('Color Temperature Adjustment')
        axes[0, 0].set_ylabel('Color Temperature (K)')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        # Plot 2: Saturation Multiplier
        axes[0, 1].plot(timestamps, [point['saturation_multiplier'] for point in grading_curve], 
                       'g-', linewidth=2)
        axes[0, 1].set_title('Saturation Multiplier')
        axes[0, 1].set_ylabel('Multiplier')
        axes[0, 1].grid(True, alpha=0.3)
        
        # Plot 3: Contrast and Brightness
        axes[1, 0].plot(timestamps, [point['contrast_multiplier'] for point in grading_curve], 
                       'purple', label='Contrast', linewidth=2)
        axes[1, 0].plot(timestamps, [point['brightness_offset'] for point in grading_curve], 
                       'orange', label='Brightness Offset', linewidth=2)
        axes[1, 0].set_title('Contrast and Brightness Adjustments')
        axes[1, 0].set_ylabel('Multiplier/Offset')
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)
        
        # Plot 4: ISO and Exposure Value
        ax2 = axes[1, 1].twinx()
        axes[1, 1].plot(timestamps, [point['iso'] for point in grading_curve], 
                       'brown', label='ISO', linewidth=2)
        ax2.plot(timestamps, [point['exposure_value'] for point in grading_curve], 
                'cyan', label='EV', linewidth=2)
        axes[1, 1].set_title('Camera Settings')
        axes[1, 1].set_ylabel('ISO')
        ax2.set_ylabel('Exposure Value')
        axes[1, 1].legend(loc='upper left')
        ax2.legend(loc='upper right')
        axes[1, 1].grid(True, alpha=0.3)
        
        # Format x-axis
        for ax in axes.flat:
            ax.tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"Grading visualization saved to: {output_file}")

    def export_grading_data(self, grading_curve: List[Dict], 
                           output_file: str = "colorflow_grading_data.json"):
        """
        Export grading data to JSON file
        
        Args:
            grading_curve: Dynamic grading curve data
            output_file: Output file path
        """
        # Convert datetime objects to strings for JSON serialization
        export_data = []
        for point in grading_curve:
            export_point = point.copy()
            export_point['timestamp'] = point['timestamp'].isoformat()
            export_data.append(export_point)
        
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump({
                'grading_curve': export_data,
                'total_points': len(export_data),
                'generated_at': datetime.now().isoformat(),
                'system_version': '1.0.0'
            }, f, indent=2, ensure_ascii=False)
        
        print(f"Grading data exported to: {output_file}")

    def process_flight_color_grading(self, srt_file: str, output_dir: str = None) -> Dict:
        """
        Complete workflow for processing flight color grading
        
        Args:
            srt_file: Path to SRT file
            output_dir: Output directory for results
            
        Returns:
            Dictionary with processing results
        """
        if output_dir is None:
            output_dir = os.path.dirname(srt_file)
        
        print(f"Processing color grading for: {srt_file}")
        
        # Step 1: Parse color data
        print("Step 1: Parsing color grading data...")
        color_data = self.parse_srt_color_data(srt_file)
        print(f"Extracted {len(color_data)} color data points")
        
        if not color_data:
            print("No color data found in SRT file")
            return {}
        
        # Step 2: Detect lighting condition
        print("Step 2: Detecting lighting condition...")
        lighting_condition = self.detect_lighting_condition(color_data)
        print(f"Detected lighting condition: {lighting_condition.value}")
        
        # Step 3: Recommend grading preset
        print("Step 3: Recommending color grading preset...")
        recommended_preset = self.recommend_grading_preset(color_data, lighting_condition)
        print(f"Recommended preset: {recommended_preset}")
        
        # Step 4: Generate dynamic grading curve
        print("Step 4: Generating dynamic grading curve...")
        grading_curve = self.generate_dynamic_grading_curve(color_data, recommended_preset)
        print(f"Generated {len(grading_curve)} grading points")
        
        # Step 5: Create visualizations
        print("Step 5: Creating visualizations...")
        base_name = os.path.splitext(os.path.basename(srt_file))[0]
        viz_file = os.path.join(output_dir, f"{base_name}_colorflow_analysis.png")
        self.create_grading_visualization(grading_curve, viz_file)
        
        # Step 6: Export data
        print("Step 6: Exporting grading data...")
        json_file = os.path.join(output_dir, f"{base_name}_colorflow_grading.json")
        self.export_grading_data(grading_curve, json_file)
        
        # Return summary
        results = {
            'srt_file': srt_file,
            'color_data_points': len(color_data),
            'lighting_condition': lighting_condition.value,
            'recommended_preset': recommended_preset,
            'grading_points': len(grading_curve),
            'visualization_file': viz_file,
            'grading_data_file': json_file,
            'ct_analysis': self.analyze_color_temperature_trends(color_data)
        }
        
        print("\nColor grading processing complete!")
        print(f"Results: {results}")
        
        return results

def main():
    """Main function for testing the colorflow grading system"""
    # Initialize the system
    colorflow = ColorflowGradingSystem()
    
    # Process a sample SRT file
    srt_file = "/workspace/Routes/DJI_0650.SRT"
    
    if os.path.exists(srt_file):
        results = colorflow.process_flight_color_grading(srt_file)
        print(f"\nProcessing complete! Check the output files for results.")
    else:
        print(f"SRT file not found: {srt_file}")

if __name__ == "__main__":
    main()