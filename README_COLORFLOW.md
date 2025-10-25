# Colorflow Dynamic Grading System for Drone Footage

A comprehensive system for processing drone SRT files with advanced color grading analysis and automated color correction recommendations.

## 🎨 Features

### Core Functionality
- **GPS Route Analysis**: Extract and visualize drone flight paths from SRT files
- **Color Temperature Analysis**: Analyze color temperature changes throughout flights
- **Dynamic Color Grading**: Intelligent color correction based on lighting conditions
- **Camera Settings Analysis**: Monitor ISO, shutter speed, f-number, and exposure values
- **Lighting Condition Detection**: Automatically detect sunrise, sunset, daylight, overcast, etc.
- **Grading Presets**: Pre-configured color grading presets for different scenarios

### Advanced Capabilities
- **Real-time Color Correction**: Dynamic adjustments based on flight conditions
- **Comprehensive Visualizations**: Multi-panel analysis charts and graphs
- **Export Capabilities**: JSON data export, KML flight paths, PNG visualizations
- **Batch Processing**: Process multiple SRT files simultaneously
- **Detailed Reporting**: Comprehensive analysis reports with statistics

## 🚀 Quick Start

### Basic Usage

```python
from enhanced_drone_processor import EnhancedDroneProcessor

# Initialize the processor
processor = EnhancedDroneProcessor()

# Process a single SRT file
results = processor.process_single_flight_with_grading("path/to/your/file.SRT")

# Process all SRT files in a directory
results = processor.process_all_flights_with_grading("path/to/srt/directory")
```

### Colorflow Grading System

```python
from colorflow_grading_system import ColorflowGradingSystem

# Initialize the colorflow system
colorflow = ColorflowGradingSystem()

# Parse color data from SRT file
color_data = colorflow.parse_srt_color_data("path/to/file.SRT")

# Detect lighting condition
lighting = colorflow.detect_lighting_condition(color_data)

# Get recommended preset
preset = colorflow.recommend_grading_preset(color_data, lighting)

# Generate dynamic grading curve
grading_curve = colorflow.generate_dynamic_grading_curve(color_data, preset)
```

## 📊 Output Files

### Generated Files
- `*_enhanced_flight_path.kml` - Flight path for Google Earth
- `*_enhanced_analysis.png` - Comprehensive 6-panel visualization
- `*_colorflow_analysis.png` - Color grading analysis charts
- `*_colorflow_grading.json` - Detailed color grading data
- `*_enhanced_report.json` - Complete analysis report
- `enhanced_processing_summary.json` - Batch processing summary

### Visualization Panels
1. **Flight Path**: GPS coordinates with altitude color coding
2. **Altitude Profile**: Altitude changes over time
3. **Color Temperature**: CT analysis throughout flight
4. **Camera Settings**: ISO and exposure value monitoring
5. **Grading Recommendations**: Visual preset representation
6. **Flight Statistics**: Summary statistics and metadata

## 🎯 Color Grading Presets

### Available Presets
- **Natural**: Minimal processing for authentic colors
- **Cinematic**: Film-like grading with enhanced contrast
- **Vibrant**: High saturation for vivid imagery
- **Moody**: Dark, atmospheric with cool tones
- **Warm**: Golden hour inspired grading

### Dynamic Adjustments
- Color temperature compensation based on lighting
- Exposure adjustments based on camera settings
- Altitude-based saturation modifications
- Real-time lighting condition adaptation

## 🔧 Installation

### Requirements
```bash
pip install matplotlib numpy pandas scipy scikit-learn
```

### Optional Dependencies
```bash
pip install simplekml cartopy ffmpeg-python
```

## 📈 Analysis Capabilities

### GPS Analysis
- Flight duration and altitude statistics
- Route interpolation and smoothing
- Geographic bounds calculation
- Speed and movement analysis

### Color Analysis
- Color temperature trend analysis
- Lighting condition detection
- Camera settings monitoring
- Exposure compensation recommendations

### Grading Analysis
- Dynamic color correction curves
- Preset recommendation engine
- Real-time adjustment calculations
- Export-ready grading data

## 🎬 Use Cases

### Professional Applications
- **Cinematography**: Automated color grading for drone footage
- **Real Estate**: Consistent color correction across property videos
- **Documentary**: Maintain visual consistency in long-form content
- **Commercial**: Brand-consistent color grading for marketing materials

### Technical Applications
- **Flight Analysis**: Comprehensive drone flight data analysis
- **Quality Control**: Automated color consistency checking
- **Research**: Scientific analysis of lighting conditions
- **Education**: Learning tool for color grading concepts

## 📋 Example Workflow

1. **Input**: DJI drone SRT file with telemetry data
2. **Parse**: Extract GPS coordinates and camera settings
3. **Analyze**: Detect lighting conditions and color trends
4. **Recommend**: Suggest optimal color grading preset
5. **Generate**: Create dynamic grading curve
6. **Visualize**: Produce comprehensive analysis charts
7. **Export**: Save KML, JSON, and PNG files

## 🔍 Technical Details

### SRT File Format Support
- DJI drone SRT subtitle files
- GPS coordinates (latitude, longitude, altitude)
- Camera settings (ISO, shutter, f-number, EV)
- Color data (color temperature, color mode)
- Timestamp synchronization

### Color Temperature Ranges
- **Sunrise/Sunset**: 2000K - 3500K
- **Golden Hour**: 3000K - 4500K
- **Daylight**: 5000K - 6500K
- **Overcast**: 6500K - 8000K
- **Blue Hour**: 6000K - 10000K

### Lighting Condition Detection
- Time-based analysis (hour of day)
- Color temperature correlation
- Exposure value patterns
- Camera setting trends

## 🚀 Advanced Features

### Batch Processing
```python
# Process all SRT files in a directory
results = processor.process_all_flights_with_grading("/path/to/srt/files")

# Access individual results
for filename, result in results.items():
    print(f"{filename}: {result['grading_results']['recommended_preset']}")
```

### Custom Grading Presets
```python
# Create custom preset
custom_preset = GradingPreset(
    name="Custom",
    description="My custom grading",
    color_temperature_offset=100,
    saturation_multiplier=1.2,
    contrast_multiplier=1.1,
    brightness_offset=0.05,
    highlights_rolloff=0.8,
    shadows_lift=0.1,
    vibrance=1.15,
    clarity=1.05
)

# Add to colorflow system
colorflow.grading_presets['custom'] = custom_preset
```

## 📊 Performance

### Processing Speed
- **Small files** (< 1000 points): ~2-5 seconds
- **Medium files** (1000-5000 points): ~5-15 seconds
- **Large files** (> 5000 points): ~15-60 seconds

### Memory Usage
- **Typical**: 50-200 MB per file
- **Large files**: Up to 500 MB
- **Batch processing**: Scales linearly

## 🛠️ Troubleshooting

### Common Issues
1. **No color data found**: Check SRT file format compatibility
2. **JSON serialization error**: Update to latest version
3. **Visualization errors**: Ensure matplotlib is properly installed
4. **Memory issues**: Process files individually for large datasets

### Debug Mode
```python
import logging
logging.basicConfig(level=logging.DEBUG)

# Run with detailed logging
processor = EnhancedDroneProcessor()
results = processor.process_single_flight_with_grading("file.SRT")
```

## 📚 API Reference

### EnhancedDroneProcessor
- `process_single_flight_with_grading(srt_file, output_dir)`
- `process_all_flights_with_grading(srt_directory, output_dir)`
- `create_enhanced_visualization(gps_df, route_df, color_data, grading_results, output_file)`

### ColorflowGradingSystem
- `parse_srt_color_data(srt_file)`
- `detect_lighting_condition(color_data)`
- `recommend_grading_preset(color_data, lighting_condition)`
- `generate_dynamic_grading_curve(color_data, preset)`
- `create_grading_visualization(grading_curve, output_file)`

## 🤝 Contributing

This system is designed to be extensible. Key areas for enhancement:
- Additional color grading presets
- New lighting condition detection algorithms
- Enhanced visualization options
- Video processing integration
- Real-time processing capabilities

## 📄 License

This project is part of the Drone SRT Processor suite and follows the same licensing terms.

---

**Colorflow Dynamic Grading System** - Bringing professional color grading to drone footage analysis.