"""
Enhanced Drone SRT Processor with Colorflow Dynamic Grading System

This module combines the original drone SRT processing capabilities with advanced
color grading analysis and recommendations. It provides a complete workflow for
processing drone footage with intelligent color correction.

Features:
- GPS route analysis and visualization
- Color temperature and exposure analysis
- Dynamic color grading recommendations
- Flight path visualization with color grading overlays
- Automated color correction for video processing
- Comprehensive reporting and data export
"""

import os
import sys
import json
import glob
from datetime import datetime
from typing import Dict, List, Optional, Tuple
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import matplotlib.patches as patches

# Import our colorflow grading system
from colorflow_grading_system import ColorflowGradingSystem, ColorGradingData, LightingCondition

# Import original drone processor functions
from drone_srt_processor import (
    parse_srt_gps, infer_route, generate_kml, render_flyover_video,
    identify_flights, process_all_srt_files
)

class EnhancedDroneProcessor:
    """
    Enhanced drone processor with integrated colorflow grading system
    """
    
    def __init__(self):
        self.colorflow = ColorflowGradingSystem()
        self.processing_results = {}
        
    def process_single_flight_with_grading(self, srt_file: str, output_dir: str = None) -> Dict:
        """
        Process a single SRT file with both GPS and color grading analysis
        
        Args:
            srt_file: Path to SRT file
            output_dir: Output directory for results
            
        Returns:
            Dictionary with comprehensive processing results
        """
        if output_dir is None:
            output_dir = os.path.dirname(srt_file)
        
        base_name = os.path.splitext(os.path.basename(srt_file))[0]
        print(f"Processing enhanced analysis for: {srt_file}")
        
        # Step 1: Parse GPS data (original functionality)
        print("Step 1: Parsing GPS data...")
        try:
            gps_df = parse_srt_gps(srt_file)
            print(f"Extracted {len(gps_df)} GPS points")
        except Exception as e:
            print(f"Error parsing GPS data: {e}")
            return {}
        
        # Step 2: Parse color grading data
        print("Step 2: Parsing color grading data...")
        try:
            color_data = self.colorflow.parse_srt_color_data(srt_file)
            print(f"Extracted {len(color_data)} color data points")
        except Exception as e:
            print(f"Error parsing color data: {e}")
            color_data = []
        
        # Step 3: Generate route analysis
        print("Step 3: Generating route analysis...")
        try:
            route_df = infer_route(gps_df)
            print(f"Generated {len(route_df)} route points")
        except Exception as e:
            print(f"Error generating route: {e}")
            return {}
        
        # Step 4: Color grading analysis
        grading_results = {}
        if color_data:
            print("Step 4: Performing color grading analysis...")
            try:
                # Detect lighting condition
                lighting_condition = self.colorflow.detect_lighting_condition(color_data)
                print(f"Detected lighting condition: {lighting_condition.value}")
                
                # Recommend grading preset
                recommended_preset = self.colorflow.recommend_grading_preset(color_data, lighting_condition)
                print(f"Recommended preset: {recommended_preset}")
                
                # Generate dynamic grading curve
                grading_curve = self.colorflow.generate_dynamic_grading_curve(color_data, recommended_preset)
                print(f"Generated {len(grading_curve)} grading points")
                
                # Create color grading visualization
                grading_viz_file = os.path.join(output_dir, f"{base_name}_colorflow_analysis.png")
                self.colorflow.create_grading_visualization(grading_curve, grading_viz_file)
                
                # Export grading data
                grading_json_file = os.path.join(output_dir, f"{base_name}_colorflow_grading.json")
                self.colorflow.export_grading_data(grading_curve, grading_json_file)
                
                grading_results = {
                    'lighting_condition': lighting_condition.value,
                    'recommended_preset': recommended_preset,
                    'grading_points': len(grading_curve),
                    'visualization_file': grading_viz_file,
                    'grading_data_file': grading_json_file,
                    'ct_analysis': self.colorflow.analyze_color_temperature_trends(color_data)
                }
                
            except Exception as e:
                print(f"Error in color grading analysis: {e}")
                grading_results = {'error': str(e)}
        
        # Step 5: Generate KML file
        print("Step 5: Generating KML file...")
        try:
            kml_file = os.path.join(output_dir, f"{base_name}_enhanced_flight_path.kml")
            generate_kml(route_df, kml_file)
            print(f"KML file saved to: {kml_file}")
        except Exception as e:
            print(f"Error generating KML: {e}")
            kml_file = None
        
        # Step 6: Create enhanced visualization
        print("Step 6: Creating enhanced visualization...")
        try:
            viz_file = os.path.join(output_dir, f"{base_name}_enhanced_analysis.png")
            self.create_enhanced_visualization(gps_df, route_df, color_data, grading_results, viz_file)
            print(f"Enhanced visualization saved to: {viz_file}")
        except Exception as e:
            print(f"Error creating enhanced visualization: {e}")
            viz_file = None
        
        # Step 7: Generate comprehensive report
        print("Step 7: Generating comprehensive report...")
        try:
            report_file = os.path.join(output_dir, f"{base_name}_enhanced_report.json")
            self.generate_comprehensive_report(srt_file, gps_df, route_df, color_data, grading_results, report_file)
            print(f"Comprehensive report saved to: {report_file}")
        except Exception as e:
            print(f"Error generating report: {e}")
            report_file = None
        
        # Compile results
        results = {
            'srt_file': srt_file,
            'gps_points': len(gps_df),
            'route_points': len(route_df),
            'color_data_points': len(color_data) if color_data else 0,
            'kml_file': kml_file,
            'enhanced_visualization': viz_file,
            'comprehensive_report': report_file,
            'grading_results': grading_results,
            'processing_timestamp': datetime.now().isoformat()
        }
        
        self.processing_results[base_name] = results
        print(f"\nEnhanced processing complete for {base_name}!")
        return results
    
    def create_enhanced_visualization(self, gps_df: pd.DataFrame, route_df: pd.DataFrame, 
                                    color_data: List[ColorGradingData], grading_results: Dict, 
                                    output_file: str):
        """
        Create enhanced visualization combining GPS and color data
        """
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle('Enhanced Drone Flight Analysis with Colorflow Grading', fontsize=16, fontweight='bold')
        
        # Plot 1: Flight path with altitude color coding
        ax1 = axes[0, 0]
        if not route_df.empty:
            scatter = ax1.scatter(route_df['lon'], route_df['lat'], 
                                c=route_df['alt'], cmap='viridis', s=2, alpha=0.7)
            ax1.plot(route_df['lon'], route_df['lat'], 'k-', alpha=0.3, linewidth=0.5)
            ax1.set_title('Flight Path (Altitude Color Coded)')
            ax1.set_xlabel('Longitude')
            ax1.set_ylabel('Latitude')
            ax1.grid(True, alpha=0.3)
            plt.colorbar(scatter, ax=ax1, label='Altitude (m)')
        
        # Plot 2: Altitude profile over time
        ax2 = axes[0, 1]
        if not gps_df.empty:
            time_sec = (gps_df['timestamp'] - gps_df['timestamp'].min()).dt.total_seconds()
            ax2.plot(time_sec, gps_df['abs_alt'], 'b-', linewidth=2, label='Absolute Altitude')
            ax2.plot(time_sec, gps_df['rel_alt'], 'r-', linewidth=2, label='Relative Altitude')
            ax2.set_title('Altitude Profile')
            ax2.set_xlabel('Time (seconds)')
            ax2.set_ylabel('Altitude (m)')
            ax2.legend()
            ax2.grid(True, alpha=0.3)
        
        # Plot 3: Color temperature analysis
        ax3 = axes[0, 2]
        if color_data:
            timestamps = [data.timestamp for data in color_data]
            cts = [data.color_temperature for data in color_data]
            time_sec = [(ts - timestamps[0]).total_seconds() for ts in timestamps]
            
            ax3.plot(time_sec, cts, 'g-', linewidth=2, label='Color Temperature')
            ax3.set_title('Color Temperature Over Time')
            ax3.set_xlabel('Time (seconds)')
            ax3.set_ylabel('Color Temperature (K)')
            ax3.legend()
            ax3.grid(True, alpha=0.3)
            
            # Add lighting condition info
            if 'lighting_condition' in grading_results:
                ax3.text(0.02, 0.98, f"Lighting: {grading_results['lighting_condition']}", 
                        transform=ax3.transAxes, verticalalignment='top',
                        bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
        else:
            ax3.text(0.5, 0.5, 'No Color Data Available', ha='center', va='center', transform=ax3.transAxes)
            ax3.set_title('Color Temperature Analysis')
        
        # Plot 4: Camera settings analysis
        ax4 = axes[1, 0]
        if color_data:
            isos = [data.iso for data in color_data]
            evs = [data.exposure_value for data in color_data]
            
            ax4_twin = ax4.twinx()
            line1 = ax4.plot(time_sec, isos, 'purple', linewidth=2, label='ISO')
            line2 = ax4_twin.plot(time_sec, evs, 'orange', linewidth=2, label='EV')
            
            ax4.set_title('Camera Settings')
            ax4.set_xlabel('Time (seconds)')
            ax4.set_ylabel('ISO', color='purple')
            ax4_twin.set_ylabel('Exposure Value', color='orange')
            
            # Combine legends
            lines = line1 + line2
            labels = [l.get_label() for l in lines]
            ax4.legend(lines, labels, loc='upper right')
            ax4.grid(True, alpha=0.3)
        else:
            ax4.text(0.5, 0.5, 'No Camera Data Available', ha='center', va='center', transform=ax4.transAxes)
            ax4.set_title('Camera Settings Analysis')
        
        # Plot 5: Color grading recommendations
        ax5 = axes[1, 1]
        if grading_results and 'recommended_preset' in grading_results:
            preset = grading_results['recommended_preset']
            preset_info = self.colorflow.grading_presets.get(preset, None)
            
            if preset_info:
                # Create a visual representation of the preset
                categories = ['Saturation', 'Contrast', 'Brightness', 'Vibrance', 'Clarity']
                values = [
                    preset_info.saturation_multiplier,
                    preset_info.contrast_multiplier,
                    preset_info.brightness_offset + 1,  # Normalize for display
                    preset_info.vibrance,
                    preset_info.clarity
                ]
                
                bars = ax5.bar(categories, values, color=['red', 'blue', 'green', 'purple', 'orange'], alpha=0.7)
                ax5.set_title(f'Recommended Preset: {preset.title()}')
                ax5.set_ylabel('Multiplier/Offset')
                ax5.tick_params(axis='x', rotation=45)
                ax5.grid(True, alpha=0.3)
                
                # Add value labels on bars
                for bar, value in zip(bars, values):
                    height = bar.get_height()
                    ax5.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                            f'{value:.2f}', ha='center', va='bottom')
        else:
            ax5.text(0.5, 0.5, 'No Grading Recommendations', ha='center', va='center', transform=ax5.transAxes)
            ax5.set_title('Color Grading Recommendations')
        
        # Plot 6: Flight statistics summary
        ax6 = axes[1, 2]
        ax6.axis('off')
        
        # Calculate flight statistics
        if not gps_df.empty:
            duration = (gps_df['timestamp'].max() - gps_df['timestamp'].min()).total_seconds()
            max_alt = gps_df['abs_alt'].max()
            min_alt = gps_df['abs_alt'].min()
            avg_alt = gps_df['abs_alt'].mean()
            
            stats_text = f"""
Flight Statistics:
• Duration: {duration:.1f} seconds
• Max Altitude: {max_alt:.1f} m
• Min Altitude: {min_alt:.1f} m
• Avg Altitude: {avg_alt:.1f} m
• GPS Points: {len(gps_df)}
• Route Points: {len(route_df)}
"""
            
            if color_data:
                avg_ct = np.mean([data.color_temperature for data in color_data])
                ct_range = max([data.color_temperature for data in color_data]) - min([data.color_temperature for data in color_data])
                stats_text += f"""
Color Analysis:
• Avg Color Temp: {avg_ct:.0f} K
• CT Range: {ct_range} K
• Color Points: {len(color_data)}
"""
            
            if grading_results and 'lighting_condition' in grading_results:
                stats_text += f"""
Grading Analysis:
• Lighting: {grading_results['lighting_condition']}
• Preset: {grading_results.get('recommended_preset', 'N/A')}
"""
            
            ax6.text(0.05, 0.95, stats_text, transform=ax6.transAxes, fontsize=10,
                    verticalalignment='top', fontfamily='monospace',
                    bbox=dict(boxstyle='round,pad=0.5', facecolor='lightgray', alpha=0.8))
        
        plt.tight_layout()
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"Enhanced visualization saved to: {output_file}")
    
    def generate_comprehensive_report(self, srt_file: str, gps_df: pd.DataFrame, 
                                    route_df: pd.DataFrame, color_data: List[ColorGradingData],
                                    grading_results: Dict, output_file: str):
        """
        Generate comprehensive analysis report
        """
        report = {
            'file_info': {
                'srt_file': srt_file,
                'file_size_mb': os.path.getsize(srt_file) / (1024 * 1024),
                'processed_at': datetime.now().isoformat()
            },
            'gps_analysis': {
                'total_points': len(gps_df),
                'duration_seconds': (gps_df['timestamp'].max() - gps_df['timestamp'].min()).total_seconds() if not gps_df.empty else 0,
                'altitude_stats': {
                    'max_altitude': float(gps_df['abs_alt'].max()) if not gps_df.empty else 0,
                    'min_altitude': float(gps_df['abs_alt'].min()) if not gps_df.empty else 0,
                    'avg_altitude': float(gps_df['abs_alt'].mean()) if not gps_df.empty else 0,
                    'altitude_range': float(gps_df['abs_alt'].max() - gps_df['abs_alt'].min()) if not gps_df.empty else 0
                },
                'location_bounds': {
                    'min_lat': float(gps_df['lat'].min()) if not gps_df.empty else 0,
                    'max_lat': float(gps_df['lat'].max()) if not gps_df.empty else 0,
                    'min_lon': float(gps_df['lon'].min()) if not gps_df.empty else 0,
                    'max_lon': float(gps_df['lon'].max()) if not gps_df.empty else 0
                }
            },
            'route_analysis': {
                'route_points': len(route_df),
                'interpolation_ratio': len(route_df) / len(gps_df) if not gps_df.empty else 0
            },
            'color_analysis': {
                'total_color_points': len(color_data),
                'lighting_condition': grading_results.get('lighting_condition', 'unknown'),
                'recommended_preset': grading_results.get('recommended_preset', 'unknown'),
                'color_temperature_stats': self._serialize_numpy_values(grading_results.get('ct_analysis', {})),
                'camera_settings': {
                    'iso_range': [min([data.iso for data in color_data]), max([data.iso for data in color_data])] if color_data else [0, 0],
                    'shutter_range': [min([data.shutter_speed for data in color_data]), max([data.shutter_speed for data in color_data])] if color_data else [0, 0],
                    'f_number_range': [min([data.f_number for data in color_data]), max([data.f_number for data in color_data])] if color_data else [0, 0],
                    'ev_range': [min([data.exposure_value for data in color_data]), max([data.exposure_value for data in color_data])] if color_data else [0, 0]
                }
            },
            'grading_recommendations': grading_results,
            'processing_metadata': {
                'system_version': '1.0.0',
                'colorflow_enabled': len(color_data) > 0,
                'enhanced_processing': True
            }
        }
        
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2, ensure_ascii=False)
        
        print(f"Comprehensive report saved to: {output_file}")
    
    def _serialize_numpy_values(self, obj):
        """Convert numpy values to Python native types for JSON serialization"""
        if isinstance(obj, dict):
            return {key: self._serialize_numpy_values(value) for key, value in obj.items()}
        elif isinstance(obj, list):
            return [self._serialize_numpy_values(item) for item in obj]
        elif isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        else:
            return obj
    
    def process_all_flights_with_grading(self, srt_directory: str, output_dir: str = None) -> Dict:
        """
        Process all SRT files in a directory with enhanced color grading analysis
        
        Args:
            srt_directory: Directory containing SRT files
            output_dir: Output directory for results
            
        Returns:
            Dictionary with processing results for all flights
        """
        if output_dir is None:
            output_dir = srt_directory
        
        print(f"Processing all SRT files with enhanced color grading in: {srt_directory}")
        
        # Find all SRT files
        srt_files = glob.glob(os.path.join(srt_directory, "*.SRT"))
        if not srt_files:
            print("No SRT files found!")
            return {}
        
        print(f"Found {len(srt_files)} SRT files")
        
        all_results = {}
        
        for i, srt_file in enumerate(sorted(srt_files), 1):
            print(f"\n--- Processing file {i}/{len(srt_files)}: {os.path.basename(srt_file)} ---")
            try:
                results = self.process_single_flight_with_grading(srt_file, output_dir)
                all_results[os.path.basename(srt_file)] = results
            except Exception as e:
                print(f"Error processing {srt_file}: {e}")
                all_results[os.path.basename(srt_file)] = {'error': str(e)}
        
        # Generate summary report
        summary_file = os.path.join(output_dir, "enhanced_processing_summary.json")
        with open(summary_file, 'w', encoding='utf-8') as f:
            json.dump({
                'total_files_processed': len(srt_files),
                'successful_files': len([r for r in all_results.values() if 'error' not in r]),
                'failed_files': len([r for r in all_results.values() if 'error' in r]),
                'processing_timestamp': datetime.now().isoformat(),
                'results': all_results
            }, f, indent=2, ensure_ascii=False)
        
        print(f"\n=== Enhanced Processing Complete ===")
        print(f"Total files processed: {len(srt_files)}")
        print(f"Successful: {len([r for r in all_results.values() if 'error' not in r])}")
        print(f"Failed: {len([r for r in all_results.values() if 'error' in r])}")
        print(f"Summary report: {summary_file}")
        
        return all_results

def main():
    """Main function for testing the enhanced drone processor"""
    # Initialize the enhanced processor
    processor = EnhancedDroneProcessor()
    
    # Process a single SRT file
    srt_file = "/workspace/Routes/DJI_0650.SRT"
    
    if os.path.exists(srt_file):
        print("Processing single flight with enhanced analysis...")
        results = processor.process_single_flight_with_grading(srt_file)
        print(f"\nSingle flight processing complete!")
        print(f"Results: {results}")
    else:
        print(f"SRT file not found: {srt_file}")
        print("Processing all flights in Routes directory...")
        results = processor.process_all_flights_with_grading("/workspace/Routes")
        print(f"\nAll flights processing complete!")

if __name__ == "__main__":
    main()