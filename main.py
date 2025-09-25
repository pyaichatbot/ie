#!/usr/bin/env python3
"""
Architecture Diagram Parser - Main Entry Point
Enterprise-grade tool for parsing architecture diagrams into structured data
"""
import argparse
import sys
import os
from pathlib import Path
from typing import Optional, Dict, Any
import json

# Add current directory to Python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from arch_parser_config import ArchitectureDiagramConfig, DEFAULT_CONFIG
from diagnostics import DiagnosticsLogger
from arch_parser_models import ProcessingResult, ArchitectureGraph
from arch_parser_ocr import MultiEngineOCR
from arch_parser_arrow_detection import ArrowDetector, ConnectionAnalyzer
from arch_parser_graph_builder import GraphBuilder, GraphValidator

class ArchitectureDiagramParser:
    """Main parser class that orchestrates the entire processing pipeline"""
    
    def __init__(self, config: Optional[ArchitectureDiagramConfig] = None):
        """
        Initialize the parser with configuration
        
        Args:
            config: Configuration object, uses default if None
        """
        self.config = config or DEFAULT_CONFIG
        self.logger = DiagnosticsLogger(self.config.diagnostics)
        
        # Initialize processing components
        self.ocr_engine = MultiEngineOCR(self.config.ocr, self.logger)
        self.arrow_detector = ArrowDetector(self.config.arrow_detection, self.logger)
        self.connection_analyzer = ConnectionAnalyzer(self.logger)
        self.graph_builder = GraphBuilder(self.config.graph_processing, self.logger)
        self.graph_validator = GraphValidator(self.logger)
        
        self.logger.logger.info("Architecture Diagram Parser initialized successfully")
    
    def parse_diagram(self, image_path: str, output_path: Optional[str] = None) -> ProcessingResult:
        """
        Parse an architecture diagram image into structured data
        
        Args:
            image_path: Path to the input image file
            output_path: Optional path to save the output JSON
            
        Returns:
            ProcessingResult object with success status and graph data
        """
        import cv2
        import time
        
        start_time = time.time()
        
        try:
            self.logger.logger.info(f"Starting diagram parsing: {image_path}")
            
            # Step 1: Load and preprocess image
            image = self._load_and_preprocess_image(image_path)
            if image is None:
                return ProcessingResult(
                    success=False,
                    error_message="Failed to load or preprocess image"
                )
            
            # Step 2: Create tiles for processing
            tiles = self._create_image_tiles(image)
            self.logger.logger.info(f"Created {len(tiles)} tiles for processing")
            
            # Step 3: Extract text using OCR
            text_blocks = self.ocr_engine.extract_text_from_tiles(tiles)
            self.logger.logger.info(f"Extracted {len(text_blocks)} text blocks")
            
            # Step 4: Detect arrows and connections
            arrows = self.arrow_detector.detect_arrows_from_tiles(tiles, text_blocks)
            self.logger.logger.info(f"Detected {len(arrows)} arrows")
            
            # Step 5: Analyze connections
            analyzed_arrows = self.connection_analyzer.analyze_connections(arrows, text_blocks)
            
            # Step 6: Build graph structure
            diagram_name = Path(image_path).stem
            graph = self.graph_builder.build_graph(text_blocks, analyzed_arrows, diagram_name)
            
            # Step 7: Validate graph quality
            validation_results = self.graph_validator.validate_graph_quality(graph)
            
            # Calculate processing time
            processing_time = time.time() - start_time
            
            # Create result
            result = ProcessingResult(
                success=True,
                graph=graph,
                processing_time=processing_time,
                step_times=self.logger.performance_tracker.get_summary(),
                intermediate_results={
                    'validation_results': validation_results,
                    'tile_count': len(tiles),
                    'text_blocks_count': len(text_blocks),
                    'arrows_count': len(arrows)
                }
            )
            
            # Save output if requested
            if output_path:
                self._save_results(result, output_path)
            
            # Log performance report
            self.logger.logger.info(f"Processing completed in {processing_time:.2f}s")
            self.logger.logger.info(self.logger.get_performance_report())
            
            return result
            
        except Exception as e:
            self.logger.log_error(e, "diagram_parsing")
            return ProcessingResult(
                success=False,
                error_message=str(e),
                processing_time=time.time() - start_time
            )
    
    def _load_and_preprocess_image(self, image_path: str):
        """Load and preprocess the input image"""
        import cv2
        
        try:
            # Load image
            image = cv2.imread(image_path)
            if image is None:
                self.logger.logger.error(f"Failed to load image: {image_path}")
                return None
            
            # Apply preprocessing based on config
            preprocessed = self._apply_preprocessing(image)
            
            # Save intermediate result if enabled
            if self.config.diagnostics.save_intermediate_results:
                self.logger.save_intermediate_image(preprocessed, "01_preprocessed_image")
            
            return preprocessed
            
        except Exception as e:
            self.logger.log_error(e, "image_loading")
            return None
    
    def _apply_preprocessing(self, image):
        """Apply image preprocessing based on configuration"""
        import cv2
        
        preprocessed = image.copy()
        
        # Upscale if configured
        if self.config.preprocessing.upscale_factor > 1.0:
            height, width = preprocessed.shape[:2]
            new_width = int(width * self.config.preprocessing.upscale_factor)
            new_height = int(height * self.config.preprocessing.upscale_factor)
            preprocessed = cv2.resize(preprocessed, (new_width, new_height), 
                                    interpolation=self.config.preprocessing.upscale_interpolation)
        
        # Enhance contrast if enabled
        if self.config.preprocessing.enhance_contrast:
            # Convert to LAB color space for better contrast enhancement
            lab = cv2.cvtColor(preprocessed, cv2.COLOR_BGR2LAB)
            l, a, b = cv2.split(lab)
            
            # Apply CLAHE to L channel
            clahe = cv2.createCLAHE(
                clipLimit=self.config.preprocessing.clahe_clip_limit,
                tileGridSize=self.config.preprocessing.clahe_tile_grid_size
            )
            l = clahe.apply(l)
            
            # Merge channels and convert back to BGR
            lab = cv2.merge([l, a, b])
            preprocessed = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
        
        return preprocessed
    
    def _create_image_tiles(self, image):
        """Create tiles from the image for processing"""
        import cv2
        
        tiles = []
        height, width = image.shape[:2]
        tile_size = self.config.tiling.tile_size
        overlap = self.config.tiling.overlap
        
        # Calculate tile positions
        for y in range(0, height, tile_size - overlap):
            for x in range(0, width, tile_size - overlap):
                # Ensure tile doesn't exceed image boundaries
                x2 = min(x + tile_size, width)
                y2 = min(y + tile_size, height)
                
                # Skip tiles that are too small
                if (x2 - x) * (y2 - y) < self.config.tiling.min_tile_area:
                    continue
                
                # Extract tile
                tile = image[y:y2, x:x2]
                
                # Create tile metadata
                tile_metadata = {
                    'tile_id': len(tiles),
                    'global_offset': {'x': x, 'y': y},
                    'tile_size': (x2 - x, y2 - y),
                    'position': (x, y)
                }
                
                tiles.append((tile, tile_metadata))
        
        return tiles
    
    def _save_results(self, result: ProcessingResult, output_path: str):
        """Save processing results to file"""
        try:
            output_data = result.to_dict()
            
            with open(output_path, 'w') as f:
                json.dump(output_data, f, indent=2, default=str)
            
            self.logger.logger.info(f"Results saved to: {output_path}")
            
        except Exception as e:
            self.logger.log_error(e, "save_results")
    
    def get_graph_summary(self, graph: ArchitectureGraph) -> Dict[str, Any]:
        """Get a summary of the parsed graph"""
        return self.graph_builder.export_graph_summary(graph)

def main():
    """Main entry point for command-line usage"""
    parser = argparse.ArgumentParser(
        description="Parse architecture diagrams into structured data",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main.py input.png
  python main.py input.png -o output.json
  python main.py input.png -c config.json -o output.json
  python main.py input.png --debug --save-intermediate
        """
    )
    
    parser.add_argument('input_image', help='Path to the input architecture diagram image')
    parser.add_argument('-o', '--output', help='Output JSON file path')
    parser.add_argument('-c', '--config', help='Configuration file path')
    parser.add_argument('--debug', action='store_true', help='Enable debug logging')
    parser.add_argument('--save-intermediate', action='store_true', 
                       help='Save intermediate processing results')
    parser.add_argument('--log-file', help='Log file path')
    
    args = parser.parse_args()
    
    # Load configuration
    config = DEFAULT_CONFIG
    if args.config and os.path.exists(args.config):
        config = ArchitectureDiagramConfig.from_file(args.config)
    
    # Override config with command line arguments
    if args.debug:
        config.diagnostics.log_level = config.diagnostics.LogLevel.DEBUG
    if args.save_intermediate:
        config.diagnostics.save_intermediate_results = True
    if args.log_file:
        config.diagnostics.log_file = args.log_file
    
    # Initialize parser
    parser_instance = ArchitectureDiagramParser(config)
    
    # Process the diagram
    result = parser_instance.parse_diagram(args.input_image, args.output)
    
    # Print results
    if result.success:
        print(f"✅ Successfully parsed diagram!")
        print(f"📊 Graph contains {len(result.graph.nodes)} nodes and {len(result.graph.edges)} edges")
        print(f"⏱️  Processing time: {result.processing_time:.2f}s")
        
        if args.output:
            print(f"💾 Results saved to: {args.output}")
        
        # Print graph summary
        summary = parser_instance.get_graph_summary(result.graph)
        print(f"\n📈 Graph Summary:")
        print(f"   - Node types: {summary['statistics']['node_types']}")
        print(f"   - Edge types: {summary['statistics']['edge_types']}")
        print(f"   - Graph density: {summary['statistics']['graph_density']:.2f}")
        print(f"   - Connected components: {summary['statistics']['connected_components']}")
        
        if summary['quality_issues']['orphaned_nodes'] > 0:
            print(f"⚠️  Warning: {summary['quality_issues']['orphaned_nodes']} orphaned nodes found")
        
        if summary['recommendations']:
            print(f"\n💡 Recommendations:")
            for rec in summary['recommendations']:
                print(f"   - {rec}")
        
        sys.exit(0)
    else:
        print(f"❌ Failed to parse diagram: {result.error_message}")
        sys.exit(1)

if __name__ == "__main__":
    main()
