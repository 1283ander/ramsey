#!/usr/bin/env python3
"""
Colorflow Dynamic Grading System - Demonstration Script

This script demonstrates the complete colorflow grading system capabilities
using the existing drone flight data. It showcases all major features and
generates comprehensive analysis reports.

Usage:
    python3 demo_colorflow_system.py
"""

import os
import sys
import json
from datetime import datetime
import matplotlib.pyplot as plt

# Import our colorflow system
from enhanced_drone_processor import EnhancedDroneProcessor
from colorflow_grading_system import ColorflowGradingSystem

def print_banner():
    """Print a nice banner for the demonstration"""
    print("=" * 80)
    print("🎨 COLORFLOW DYNAMIC GRADING SYSTEM - DEMONSTRATION")
    print("=" * 80)
    print("Advanced drone footage analysis with intelligent color grading")
    print("=" * 80)
    print()

def demonstrate_basic_colorflow():
    """Demonstrate basic colorflow grading system functionality"""
    print("🔍 DEMONSTRATION 1: Basic Colorflow Grading System")
    print("-" * 60)
    
    # Initialize the colorflow system
    colorflow = ColorflowGradingSystem()
    
    # Test with a sample SRT file
    srt_file = "/workspace/Routes/DJI_0650.SRT"
    
    if not os.path.exists(srt_file):
        print(f"❌ SRT file not found: {srt_file}")
        return
    
    print(f"📁 Processing file: {os.path.basename(srt_file)}")
    
    # Parse color data
    print("   📊 Parsing color grading data...")
    color_data = colorflow.parse_srt_color_data(srt_file)
    print(f"   ✅ Extracted {len(color_data)} color data points")
    
    if not color_data:
        print("   ❌ No color data found in SRT file")
        return
    
    # Detect lighting condition
    print("   🌅 Detecting lighting condition...")
    lighting_condition = colorflow.detect_lighting_condition(color_data)
    print(f"   ✅ Detected: {lighting_condition.value}")
    
    # Recommend grading preset
    print("   🎨 Recommending color grading preset...")
    recommended_preset = colorflow.recommend_grading_preset(color_data, lighting_condition)
    print(f"   ✅ Recommended: {recommended_preset}")
    
    # Analyze color temperature trends
    print("   📈 Analyzing color temperature trends...")
    ct_analysis = colorflow.analyze_color_temperature_trends(color_data)
    print(f"   ✅ CT Range: {ct_analysis['min_ct']}K - {ct_analysis['max_ct']}K")
    print(f"   ✅ Average CT: {ct_analysis['avg_ct']:.0f}K")
    print(f"   ✅ CT Trend: {ct_analysis['ct_trend']}")
    
    # Generate dynamic grading curve
    print("   🎬 Generating dynamic grading curve...")
    grading_curve = colorflow.generate_dynamic_grading_curve(color_data, recommended_preset)
    print(f"   ✅ Generated {len(grading_curve)} grading points")
    
    # Create visualization
    print("   📊 Creating color grading visualization...")
    viz_file = "/workspace/Routes/demo_colorflow_analysis.png"
    colorflow.create_grading_visualization(grading_curve, viz_file)
    print(f"   ✅ Visualization saved: {viz_file}")
    
    # Export data
    print("   💾 Exporting grading data...")
    json_file = "/workspace/Routes/demo_colorflow_grading.json"
    colorflow.export_grading_data(grading_curve, json_file)
    print(f"   ✅ Data exported: {json_file}")
    
    print("   🎉 Basic colorflow demonstration complete!")
    print()

def demonstrate_enhanced_processing():
    """Demonstrate enhanced drone processor with colorflow integration"""
    print("🚁 DEMONSTRATION 2: Enhanced Drone Processor")
    print("-" * 60)
    
    # Initialize the enhanced processor
    processor = EnhancedDroneProcessor()
    
    # Test with a sample SRT file
    srt_file = "/workspace/Routes/DJI_0650.SRT"
    
    if not os.path.exists(srt_file):
        print(f"❌ SRT file not found: {srt_file}")
        return
    
    print(f"📁 Processing file: {os.path.basename(srt_file)}")
    
    # Process with enhanced analysis
    print("   🔄 Running enhanced processing workflow...")
    results = processor.process_single_flight_with_grading(srt_file, "/workspace/Routes")
    
    if 'error' in results:
        print(f"   ❌ Error: {results['error']}")
        return
    
    # Display results
    print("   📊 Processing Results:")
    print(f"      • GPS Points: {results['gps_points']}")
    print(f"      • Route Points: {results['route_points']}")
    print(f"      • Color Data Points: {results['color_data_points']}")
    print(f"      • KML File: {os.path.basename(results['kml_file']) if results['kml_file'] else 'None'}")
    print(f"      • Enhanced Visualization: {os.path.basename(results['enhanced_visualization']) if results['enhanced_visualization'] else 'None'}")
    
    if results['grading_results']:
        grading = results['grading_results']
        print("   🎨 Color Grading Results:")
        print(f"      • Lighting Condition: {grading['lighting_condition']}")
        print(f"      • Recommended Preset: {grading['recommended_preset']}")
        print(f"      • Grading Points: {grading['grading_points']}")
        print(f"      • Color Analysis: {os.path.basename(grading['visualization_file'])}")
    
    print("   🎉 Enhanced processing demonstration complete!")
    print()

def demonstrate_preset_analysis():
    """Demonstrate color grading preset analysis"""
    print("🎨 DEMONSTRATION 3: Color Grading Preset Analysis")
    print("-" * 60)
    
    colorflow = ColorflowGradingSystem()
    
    print("   📋 Available Color Grading Presets:")
    for name, preset in colorflow.grading_presets.items():
        print(f"      • {preset.name}: {preset.description}")
        print(f"        - CT Offset: {preset.color_temperature_offset:+d}K")
        print(f"        - Saturation: {preset.saturation_multiplier:.2f}x")
        print(f"        - Contrast: {preset.contrast_multiplier:.2f}x")
        print(f"        - Brightness: {preset.brightness_offset:+.2f}")
        print(f"        - Vibrance: {preset.vibrance:.2f}x")
        print()
    
    print("   🎉 Preset analysis demonstration complete!")
    print()

def demonstrate_batch_processing():
    """Demonstrate batch processing capabilities"""
    print("📦 DEMONSTRATION 4: Batch Processing")
    print("-" * 60)
    
    processor = EnhancedDroneProcessor()
    
    # Find SRT files
    srt_directory = "/workspace/Routes"
    srt_files = [f for f in os.listdir(srt_directory) if f.endswith('.SRT')]
    
    print(f"   📁 Found {len(srt_files)} SRT files in {srt_directory}")
    
    if len(srt_files) == 0:
        print("   ❌ No SRT files found for batch processing")
        return
    
    # Process first few files as demonstration
    demo_files = srt_files[:3]  # Process first 3 files
    
    print(f"   🔄 Processing {len(demo_files)} files for demonstration...")
    
    for i, srt_file in enumerate(demo_files, 1):
        print(f"      {i}. Processing {srt_file}...")
        try:
            results = processor.process_single_flight_with_grading(
                os.path.join(srt_directory, srt_file), 
                srt_directory
            )
            
            if 'grading_results' in results and results['grading_results']:
                preset = results['grading_results']['recommended_preset']
                lighting = results['grading_results']['lighting_condition']
                print(f"         ✅ {preset} preset for {lighting} lighting")
            else:
                print(f"         ⚠️  No color data available")
                
        except Exception as e:
            print(f"         ❌ Error: {str(e)[:50]}...")
    
    print("   🎉 Batch processing demonstration complete!")
    print()

def demonstrate_visualization_capabilities():
    """Demonstrate visualization capabilities"""
    print("📊 DEMONSTRATION 5: Visualization Capabilities")
    print("-" * 60)
    
    # Check for generated files
    routes_dir = "/workspace/Routes"
    viz_files = [f for f in os.listdir(routes_dir) if f.endswith('.png')]
    
    print(f"   📁 Found {len(viz_files)} visualization files:")
    for viz_file in viz_files:
        file_path = os.path.join(routes_dir, viz_file)
        file_size = os.path.getsize(file_path) / 1024  # KB
        print(f"      • {viz_file} ({file_size:.1f} KB)")
    
    # Check for JSON data files
    json_files = [f for f in os.listdir(routes_dir) if f.endswith('.json')]
    print(f"   📁 Found {len(json_files)} data files:")
    for json_file in json_files:
        file_path = os.path.join(routes_dir, json_file)
        file_size = os.path.getsize(file_path) / 1024  # KB
        print(f"      • {json_file} ({file_size:.1f} KB)")
    
    print("   🎉 Visualization capabilities demonstration complete!")
    print()

def generate_demo_summary():
    """Generate a summary of the demonstration"""
    print("📋 DEMONSTRATION SUMMARY")
    print("-" * 60)
    
    # Count generated files
    routes_dir = "/workspace/Routes"
    png_files = len([f for f in os.listdir(routes_dir) if f.endswith('.png')])
    json_files = len([f for f in os.listdir(routes_dir) if f.endswith('.json')])
    kml_files = len([f for f in os.listdir(routes_dir) if f.endswith('.kml')])
    
    print(f"   📊 Generated Files:")
    print(f"      • Visualizations: {png_files} PNG files")
    print(f"      • Data Files: {json_files} JSON files")
    print(f"      • Flight Paths: {kml_files} KML files")
    
    # System capabilities
    print(f"   🎨 System Capabilities Demonstrated:")
    print(f"      • Color temperature analysis")
    print(f"      • Lighting condition detection")
    print(f"      • Dynamic color grading")
    print(f"      • GPS route analysis")
    print(f"      • Camera settings monitoring")
    print(f"      • Comprehensive visualizations")
    print(f"      • Data export and reporting")
    
    print(f"   🚀 Ready for production use!")
    print()

def main():
    """Main demonstration function"""
    print_banner()
    
    try:
        # Run all demonstrations
        demonstrate_basic_colorflow()
        demonstrate_enhanced_processing()
        demonstrate_preset_analysis()
        demonstrate_batch_processing()
        demonstrate_visualization_capabilities()
        generate_demo_summary()
        
        print("🎉 ALL DEMONSTRATIONS COMPLETE!")
        print("=" * 80)
        print("The Colorflow Dynamic Grading System is ready for use!")
        print("Check the generated files in /workspace/Routes/ for results.")
        print("=" * 80)
        
    except Exception as e:
        print(f"❌ Demonstration error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()