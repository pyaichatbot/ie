# config.py
"""
Configuration management for Architecture Diagram Parser
Follows enterprise standards for configuration management
"""
import os
import json
from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Optional
from enum import Enum

class OCREngine(Enum):
    EASYOCR = "easyocr"
    TESSERACT = "tesseract" 
    RAPIDOCR = "rapidocr"
    PADDLEOCR = "paddleocr"

class LogLevel(Enum):
    DEBUG = "DEBUG"
    INFO = "INFO"
    WARNING = "WARNING"
    ERROR = "ERROR"

@dataclass
class ImagePreprocessingConfig:
    """Image preprocessing configuration"""
    upscale_factor: float = 2.5
    upscale_interpolation: int = 2  # cv2.INTER_CUBIC
    enhance_contrast: bool = True
    clahe_clip_limit: float = 3.0
    clahe_tile_grid_size: Tuple[int, int] = (8, 8)
    deskew_enabled: bool = True
    deskew_angle_threshold: float = 0.5

@dataclass 
class TilingConfig:
    """Image tiling configuration"""
    tile_size: int = 1024
    overlap: int = 128
    min_tile_area: int = 256 * 256  # Skip tiles smaller than this
    
@dataclass
class OCRConfig:
    """OCR engine configuration"""
    primary_engine: OCREngine = OCREngine.EASYOCR
    fallback_engines: List[OCREngine] = field(default_factory=lambda: [OCREngine.TESSERACT, OCREngine.RAPIDOCR])
    confidence_threshold: float = 0.6
    min_text_height: int = 8
    max_text_height: int = 200
    tesseract_psm: int = 11  # Page segmentation mode
    tesseract_config: str = "--oem 3 -c tessedit_char_whitelist=0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz.:-/_"

@dataclass
class ArrowDetectionConfig:
    """Arrow detection configuration"""
    hough_threshold: int = 50
    min_line_length: int = 30
    max_line_gap: int = 10
    arrowhead_template_sizes: List[int] = field(default_factory=lambda: [10, 15, 20, 25])
    connection_distance_threshold: int = 20
    angle_tolerance: float = 15.0  # degrees

@dataclass
class GraphProcessingConfig:
    """Graph processing configuration"""
    node_merge_distance: int = 50
    edge_snap_distance: int = 30
    min_node_area: int = 100
    ip_address_pattern: str = r'\b(?:[0-9]{1,3}\.){3}[0-9]{1,3}\b'
    server_keywords: List[str] = field(default_factory=lambda: ['server', 'srv', 'host', 'node', 'db', 'database'])

@dataclass
class LLMConfig:
    """LLM processing configuration"""
    enabled: bool = True
    model: str = "gpt-4o"
    max_tokens: int = 4000
    temperature: float = 0.1
    image_max_dimension: int = 1024  # For low-res image sent to LLM

@dataclass
class DiagnosticsConfig:
    """Diagnostics and logging configuration"""
    log_level: LogLevel = LogLevel.INFO
    log_file: Optional[str] = "architecture_parser.log"
    performance_tracking: bool = True
    save_intermediate_results: bool = False
    intermediate_results_dir: str = "./debug_output"
    enable_visualization: bool = False

@dataclass
class ArchitectureDiagramConfig:
    """Main configuration class"""
    preprocessing: ImagePreprocessingConfig = field(default_factory=ImagePreprocessingConfig)
    tiling: TilingConfig = field(default_factory=TilingConfig)
    ocr: OCRConfig = field(default_factory=OCRConfig)
    arrow_detection: ArrowDetectionConfig = field(default_factory=ArrowDetectionConfig)
    graph_processing: GraphProcessingConfig = field(default_factory=GraphProcessingConfig)
    llm: LLMConfig = field(default_factory=LLMConfig)
    diagnostics: DiagnosticsConfig = field(default_factory=DiagnosticsConfig)
    
    @classmethod
    def from_file(cls, config_path: str) -> 'ArchitectureDiagramConfig':
        """Load configuration from JSON file"""
        if not os.path.exists(config_path):
            return cls()  # Return default config
            
        with open(config_path, 'r') as f:
            config_dict = json.load(f)
        
        # Convert nested dictionaries to dataclasses
        # This is a simplified version - in production, use a proper config library
        return cls(**config_dict)
    
    def save_to_file(self, config_path: str):
        """Save configuration to JSON file"""
        os.makedirs(os.path.dirname(config_path), exist_ok=True)
        with open(config_path, 'w') as f:
            # Convert dataclasses to dict for JSON serialization
            config_dict = {
                'preprocessing': self.preprocessing.__dict__,
                'tiling': self.tiling.__dict__,
                'ocr': self.ocr.__dict__,
                'arrow_detection': self.arrow_detection.__dict__,
                'graph_processing': self.graph_processing.__dict__,
                'llm': self.llm.__dict__,
                'diagnostics': self.diagnostics.__dict__
            }
            json.dump(config_dict, f, indent=2)

# Default configuration instance
DEFAULT_CONFIG = ArchitectureDiagramConfig()