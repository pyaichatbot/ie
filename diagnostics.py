# diagnostics.py
"""
Enterprise-grade logging and diagnostics for Architecture Diagram Parser
Provides performance tracking, error handling, and debugging capabilities
"""
import logging
import time
import functools
import traceback
import json
from typing import Any, Dict, Optional, List, Callable
from datetime import datetime
from pathlib import Path
import cv2
import numpy as np

from arch_parser_config import DiagnosticsConfig, LogLevel

class PerformanceTracker:
    """Tracks performance metrics for each processing step"""
    
    def __init__(self):
        self.metrics: Dict[str, List[float]] = {}
        self.current_step_times: Dict[str, float] = {}
    
    def start_step(self, step_name: str):
        """Start timing a processing step"""
        self.current_step_times[step_name] = time.time()
    
    def end_step(self, step_name: str) -> float:
        """End timing a processing step and return duration"""
        if step_name not in self.current_step_times:
            return 0.0
        
        duration = time.time() - self.current_step_times[step_name]
        
        if step_name not in self.metrics:
            self.metrics[step_name] = []
        self.metrics[step_name].append(duration)
        
        del self.current_step_times[step_name]
        return duration
    
    def get_summary(self) -> Dict[str, Dict[str, float]]:
        """Get performance summary statistics"""
        summary = {}
        for step, times in self.metrics.items():
            if times:
                summary[step] = {
                    'total_time': sum(times),
                    'avg_time': sum(times) / len(times),
                    'min_time': min(times),
                    'max_time': max(times),
                    'count': len(times)
                }
        return summary

class DiagnosticsLogger:
    """Enterprise logging system with structured logging and performance tracking"""
    
    def __init__(self, config: DiagnosticsConfig):
        self.config = config
        self.performance_tracker = PerformanceTracker()
        self.setup_logging()
        
        # Create debug output directory if needed
        if config.save_intermediate_results:
            Path(config.intermediate_results_dir).mkdir(parents=True, exist_ok=True)

    def setup_logging(self):
        """Setup logging configuration"""
        # Create logger
        self.logger = logging.getLogger('ArchitectureDiagramParser')
        self.logger.setLevel(getattr(logging, self.config.log_level.value))
        
        # Clear existing handlers
        self.logger.handlers.clear()
        
        # Create formatters
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(funcName)s:%(lineno)d - %(message)s'
        )
        
        # Console handler
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(formatter)
        self.logger.addHandler(console_handler)
        
        # File handler if specified
        if self.config.log_file:
            file_handler = logging.FileHandler(self.config.log_file)
            file_handler.setFormatter(formatter)
            self.logger.addHandler(file_handler)

    def save_intermediate_image(self, image: np.ndarray, filename: str, metadata: Optional[Dict] = None):
        """Save intermediate processing results for debugging"""
        if not self.config.save_intermediate_results:
            return
        
        filepath = Path(self.config.intermediate_results_dir) / f"{filename}.png"
        cv2.imwrite(str(filepath), image)
        
        if metadata:
            metadata_path = filepath.with_suffix('.json')
            with open(metadata_path, 'w') as f:
                json.dump(metadata, f, indent=2, default=str)
        
        self.logger.debug(f"Saved intermediate result: {filepath}")

    def save_intermediate_data(self, data: Any, filename: str):
        """Save intermediate data structures for debugging"""
        if not self.config.save_intermediate_results:
            return
        
        filepath = Path(self.config.intermediate_results_dir) / f"{filename}.json"
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2, default=str)
        
        self.logger.debug(f"Saved intermediate data: {filepath}")

    def log_step_start(self, step_name: str, **kwargs):
        """Log the start of a processing step"""
        if self.config.performance_tracking:
            self.performance_tracker.start_step(step_name)
        
        self.logger.info(f"Starting step: {step_name}", extra=kwargs)

    def log_step_end(self, step_name: str, success: bool = True, **kwargs):
        """Log the end of a processing step"""
        duration = 0.0
        if self.config.performance_tracking:
            duration = self.performance_tracker.end_step(step_name)
        
        status = "completed" if success else "failed"
        self.logger.info(f"Step {step_name} {status} in {duration:.2f}s", extra=kwargs)
        
        return duration

    def log_ocr_results(self, engine: str, results: List, tile_info: Optional[Dict] = None):
        """Log OCR engine results with diagnostics"""
        result_count = len(results) if results else 0
        avg_confidence = 0.0
        
        if results and hasattr(results[0], 'confidence'):
            confidences = [r.confidence for r in results if hasattr(r, 'confidence')]
            avg_confidence = sum(confidences) / len(confidences) if confidences else 0.0
        
        self.logger.info(
            f"OCR Engine {engine}: {result_count} results, avg_confidence: {avg_confidence:.2f}",
            extra={'tile_info': tile_info, 'engine': engine, 'result_count': result_count}
        )
        
        if result_count == 0:
            self.logger.warning(f"OCR Engine {engine} returned 0 results - may need parameter tuning")

    def log_arrow_detection_results(self, shafts: List, heads: List, arrows: List):
        """Log arrow detection results"""
        self.logger.info(
            f"Arrow detection: {len(shafts)} shafts, {len(heads)} heads, {len(arrows)} complete arrows",
            extra={'shaft_count': len(shafts), 'head_count': len(heads), 'arrow_count': len(arrows)}
        )

    def log_error(self, error: Exception, context: str, **kwargs):
        """Log errors with full context"""
        self.logger.error(
            f"Error in {context}: {str(error)}",
            extra={'error_type': type(error).__name__, 'context': context, **kwargs}
        )
        self.logger.debug(f"Full traceback: {traceback.format_exc()}")

    def get_performance_report(self) -> str:
        """Generate a performance report"""
        summary = self.performance_tracker.get_summary()
        
        report = ["=== Performance Report ==="]
        total_time = 0.0
        
        for step, metrics in summary.items():
            report.append(f"{step}:")
            report.append(f"  Total: {metrics['total_time']:.2f}s")
            report.append(f"  Average: {metrics['avg_time']:.2f}s")
            report.append(f"  Count: {metrics['count']}")
            total_time += metrics['total_time']
        
        report.append(f"\nOverall processing time: {total_time:.2f}s")
        return "\n".join(report)

def performance_monitor(step_name: str):
    """Decorator to monitor performance of functions"""
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            # Try to find logger instance in args/kwargs
            logger_instance = None
            for arg in args:
                if isinstance(arg, DiagnosticsLogger):
                    logger_instance = arg
                    break
            
            if not logger_instance:
                for key, value in kwargs.items():
                    if isinstance(value, DiagnosticsLogger):
                        logger_instance = value
                        break
            
            if logger_instance and logger_instance.config.performance_tracking:
                logger_instance.log_step_start(step_name)
                try:
                    result = func(*args, **kwargs)
                    logger_instance.log_step_end(step_name, success=True)
                    return result
                except Exception as e:
                    logger_instance.log_step_end(step_name, success=False)
                    logger_instance.log_error(e, step_name)
                    raise
            else:
                return func(*args, **kwargs)
        
        return wrapper
    return decorator

def safe_execute(func: Callable, *args, logger: Optional[DiagnosticsLogger] = None, **kwargs) -> tuple[bool, Any]:
    """Safely execute a function with error handling"""
    try:
        result = func(*args, **kwargs)
        return True, result
    except Exception as e:
        if logger:
            logger.log_error(e, func.__name__)
        return False, None