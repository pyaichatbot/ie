# ocr_engine.py
"""
Step 2: Multi-Engine OCR with Fallback Strategy
Supports EasyOCR, Tesseract, RapidOCR, and PaddleOCR with intelligent fallback
"""
import cv2
import numpy as np
from typing import List, Dict, Tuple, Optional, Any
import re
from abc import ABC, abstractmethod

try:
    import easyocr
except ImportError:
    easyocr = None

try:
    import pytesseract
except ImportError:
    pytesseract = None

try:
    from rapidocr_onnxruntime import RapidOCR
except ImportError:
    RapidOCR = None

try:
    from paddleocr import PaddleOCR
except ImportError:
    PaddleOCR = None

from arch_parser_config import OCRConfig, OCREngine
from diagnostics import DiagnosticsLogger, performance_monitor
from arch_parser_models import TextBlock, BoundingBox

class BaseOCREngine(ABC):
    """Abstract base class for OCR engines"""
    
    def __init__(self, config: OCRConfig, logger: DiagnosticsLogger):
        self.config = config
        self.logger = logger
        self.engine_name = self.__class__.__name__
    
    @abstractmethod
    def initialize(self) -> bool:
        """Initialize the OCR engine"""
        pass
    
    @abstractmethod
    def extract_text(self, image: np.ndarray, tile_metadata: Dict) -> List[TextBlock]:
        """Extract text from image and return TextBlock objects"""
        pass
    
    def _filter_by_confidence(self, text_blocks: List[TextBlock]) -> List[TextBlock]:
        """Filter text blocks by confidence threshold"""
        return [tb for tb in text_blocks if tb.confidence >= self.config.confidence_threshold]
    
    def _filter_by_size(self, text_blocks: List[TextBlock]) -> List[TextBlock]:
        """Filter text blocks by text height"""
        filtered = []
        for tb in text_blocks:
            height = tb.bounding_box.height
            if self.config.min_text_height <= height <= self.config.max_text_height:
                filtered.append(tb)
            else:
                self.logger.logger.debug(f"Filtered text by size: '{tb.text}' (height: {height})")
        return filtered
    
    def _clean_text(self, text: str) -> str:
        """Clean extracted text"""
        # Remove excessive whitespace
        text = re.sub(r'\s+', ' ', text.strip())
        
        # Remove non-printable characters except common technical symbols
        text = re.sub(r'[^\x20-\x7E]', '', text)
        
        return text

class EasyOCREngine(BaseOCREngine):
    """EasyOCR implementation"""
    
    def __init__(self, config: OCRConfig, logger: DiagnosticsLogger):
        super().__init__(config, logger)
        self.reader = None
    
    def initialize(self) -> bool:
        """Initialize EasyOCR"""
        if easyocr is None:
            self.logger.logger.warning("EasyOCR not available")
            return False
        
        try:
            self.reader = easyocr.Reader(['en'], gpu=False)  # CPU mode for stability
            self.logger.logger.info("EasyOCR initialized successfully")
            return True
        except Exception as e:
            self.logger.log_error(e, "EasyOCR initialization")
            return False
    
    def extract_text(self, image: np.ndarray, tile_metadata: Dict) -> List[TextBlock]:
        """Extract text using EasyOCR"""
        if self.reader is None:
            return []
        
        try:
            results = self.reader.readtext(image)
            text_blocks = []
            
            for result in results:
                bbox_coords, text, confidence = result
                
                # Convert bbox to our format (EasyOCR returns 4 corner points)
                x_coords = [point[0] for point in bbox_coords]
                y_coords = [point[1] for point in bbox_coords]
                x1, x2 = int(min(x_coords)), int(max(x_coords))
                y1, y2 = int(min(y_coords)), int(max(y_coords))
                
                # Create TextBlock
                text_block = TextBlock(
                    text=self._clean_text(text),
                    bounding_box=BoundingBox(x1, y1, x2, y2),
                    confidence=float(confidence),
                    ocr_engine="easyocr"
                )
                
                text_blocks.append(text_block)
            
            # Apply filters
            text_blocks = self._filter_by_confidence(text_blocks)
            text_blocks = self._filter_by_size(text_blocks)
            
            return text_blocks
            
        except Exception as e:
            self.logger.log_error(e, "EasyOCR text extraction")
            return []

class TesseractEngine(BaseOCREngine):
    """Tesseract OCR implementation"""
    
    def initialize(self) -> bool:
        """Initialize Tesseract"""
        if pytesseract is None:
            self.logger.logger.warning("Tesseract not available")
            return False
        
        try:
            # Test if tesseract is installed
            version = pytesseract.get_tesseract_version()
            self.logger.logger.info(f"Tesseract initialized successfully, version: {version}")
            return True
        except Exception as e:
            self.logger.log_error(e, "Tesseract initialization")
            return False
    
    def extract_text(self, image: np.ndarray, tile_metadata: Dict) -> List[TextBlock]:
        """Extract text using Tesseract"""
        try:
            # Configure Tesseract
            custom_config = f'--psm {self.config.tesseract_psm} {self.config.tesseract_config}'
            
            # Get detailed data including bounding boxes and confidence
            data = pytesseract.image_to_data(image, config=custom_config, output_type=pytesseract.Output.DICT)
            
            text_blocks = []
            n_boxes = len(data['text'])
            
            for i in range(n_boxes):
                # Skip empty text
                text = data['text'][i].strip()
                if not text or text.isspace():
                    continue
                
                # Get confidence (Tesseract returns -1 for no confidence)
                confidence = float(data['conf'][i]) / 100.0 if data['conf'][i] > 0 else 0.0
                
                # Get bounding box
                x, y, w, h = data['left'][i], data['top'][i], data['width'][i], data['height'][i]
                
                # Create TextBlock
                text_block = TextBlock(
                    text=self._clean_text(text),
                    bounding_box=BoundingBox(x, y, x + w, y + h),
                    confidence=confidence,
                    ocr_engine="tesseract"
                )
                
                text_blocks.append(text_block)
            
            # Apply filters
            text_blocks = self._filter_by_confidence(text_blocks)
            text_blocks = self._filter_by_size(text_blocks)
            
            return text_blocks
            
        except Exception as e:
            self.logger.log_error(e, "Tesseract text extraction")
            return []

class RapidOCREngine(BaseOCREngine):
    """RapidOCR implementation"""
    
    def __init__(self, config: OCRConfig, logger: DiagnosticsLogger):
        super().__init__(config, logger)
        self.ocr = None
    
    def initialize(self) -> bool:
        """Initialize RapidOCR"""
        if RapidOCR is None:
            self.logger.logger.warning("RapidOCR not available")
            return False
        
        try:
            self.ocr = RapidOCR()
            self.logger.logger.info("RapidOCR initialized successfully")
            return True
        except Exception as e:
            self.logger.log_error(e, "RapidOCR initialization")
            return False
    
    def extract_text(self, image: np.ndarray, tile_metadata: Dict) -> List[TextBlock]:
        """Extract text using RapidOCR"""
        if self.ocr is None:
            return []
        
        try:
            result, _ = self.ocr(image)
            text_blocks = []
            
            if result is None:
                return text_blocks
            
            for detection in result:
                bbox_coords, text, confidence = detection
                
                # Convert bbox (4 corner points to x1,y1,x2,y2)
                x_coords = [point[0] for point in bbox_coords]
                y_coords = [point[1] for point in bbox_coords]
                x1, x2 = int(min(x_coords)), int(max(x_coords))
                y1, y2 = int(min(y_coords)), int(max(y_coords))
                
                # Create TextBlock
                text_block = TextBlock(
                    text=self._clean_text(text),
                    bounding_box=BoundingBox(x1, y1, x2, y2),
                    confidence=float(confidence),
                    ocr_engine="rapidocr"
                )
                
                text_blocks.append(text_block)
            
            # Apply filters
            text_blocks = self._filter_by_confidence(text_blocks)
            text_blocks = self._filter_by_size(text_blocks)
            
            return text_blocks
            
        except Exception as e:
            self.logger.log_error(e, "RapidOCR text extraction")
            return []

class PaddleOCREngine(BaseOCREngine):
    """PaddleOCR implementation"""
    
    def __init__(self, config: OCRConfig, logger: DiagnosticsLogger):
        super().__init__(config, logger)
        self.ocr = None
    
    def initialize(self) -> bool:
        """Initialize PaddleOCR"""
        if PaddleOCR is None:
            self.logger.logger.warning("PaddleOCR not available")
            return False
        
        try:
            self.ocr = PaddleOCR(use_angle_cls=True, lang='en', show_log=False)
            self.logger.logger.info("PaddleOCR initialized successfully")
            return True
        except Exception as e:
            self.logger.log_error(e, "PaddleOCR initialization")
            return False
    
    def extract_text(self, image: np.ndarray, tile_metadata: Dict) -> List[TextBlock]:
        """Extract text using PaddleOCR"""
        if self.ocr is None:
            return []
        
        try:
            results = self.ocr.ocr(image, cls=True)
            text_blocks = []
            
            if not results or not results[0]:
                return text_blocks
            
            for detection in results[0]:
                bbox_coords, (text, confidence) = detection
                
                # Convert bbox
                x_coords = [point[0] for point in bbox_coords]
                y_coords = [point[1] for point in bbox_coords]
                x1, x2 = int(min(x_coords)), int(max(x_coords))
                y1, y2 = int(min(y_coords)), int(max(y_coords))
                
                # Create TextBlock
                text_block = TextBlock(
                    text=self._clean_text(text),
                    bounding_box=BoundingBox(x1, y1, x2, y2),
                    confidence=float(confidence),
                    ocr_engine="paddleocr"
                )
                
                text_blocks.append(text_block)
            
            # Apply filters
            text_blocks = self._filter_by_confidence(text_blocks)
            text_blocks = self._filter_by_size(text_blocks)
            
            return text_blocks
            
        except Exception as e:
            self.logger.log_error(e, "PaddleOCR text extraction")
            return []

class MultiEngineOCR:
    """Multi-engine OCR with intelligent fallback strategy"""
    
    def __init__(self, config: OCRConfig, logger: DiagnosticsLogger):
        self.config = config
        self.logger = logger
        self.engines = {}
        self.initialize_engines()
    
    def initialize_engines(self):
        """Initialize all available OCR engines"""
        engine_classes = {
            OCREngine.EASYOCR: EasyOCREngine,
            OCREngine.TESSERACT: TesseractEngine,
            OCREngine.RAPIDOCR: RapidOCREngine,
            OCREngine.PADDLEOCR: PaddleOCREngine
        }
        
        # Initialize primary engine
        primary_class = engine_classes.get(self.config.primary_engine)
        if primary_class:
            engine = primary_class(self.config, self.logger)
            if engine.initialize():
                self.engines[self.config.primary_engine] = engine
                self.logger.logger.info(f"Primary OCR engine initialized: {self.config.primary_engine.value}")
        
        # Initialize fallback engines
        for fallback_engine in self.config.fallback_engines:
            if fallback_engine == self.config.primary_engine:
                continue  # Skip if already initialized
            
            fallback_class = engine_classes.get(fallback_engine)
            if fallback_class:
                engine = fallback_class(self.config, self.logger)
                if engine.initialize():
                    self.engines[fallback_engine] = engine
                    self.logger.logger.info(f"Fallback OCR engine initialized: {fallback_engine.value}")
        
        if not self.engines:
            self.logger.logger.error("No OCR engines could be initialized")
    
    @performance_monitor("ocr_extract_text")
    def extract_text_from_tiles(self, tiles: List[Tuple[np.ndarray, Dict]]) -> List[TextBlock]:
        """
        Extract text from all tiles using multi-engine approach with fallback
        
        Args:
            tiles: List of (tile_image, tile_metadata) tuples
            
        Returns:
            List of TextBlock objects with global coordinates
        """
        self.logger.log_step_start("text_extraction", tile_count=len(tiles))
        
        all_text_blocks = []
        successful_engines = {}
        failed_tiles = []
        
        for tile_idx, (tile_image, tile_metadata) in enumerate(tiles):
            self.logger.logger.debug(f"Processing tile {tile_idx + 1}/{len(tiles)}")
            
            tile_text_blocks = self._extract_text_from_single_tile(tile_image, tile_metadata)
            
            if tile_text_blocks:
                # Convert local coordinates to global coordinates
                global_text_blocks = self._convert_to_global_coordinates(tile_text_blocks, tile_metadata)
                all_text_blocks.extend(global_text_blocks)
                
                # Track which engine was successful
                engine_used = tile_text_blocks[0].ocr_engine if tile_text_blocks else "none"
                successful_engines[engine_used] = successful_engines.get(engine_used, 0) + 1
            else:
                failed_tiles.append(tile_idx)
                self.logger.logger.warning(f"No text extracted from tile {tile_idx}")
        
        # Log summary
        total_blocks = len(all_text_blocks)
        self.logger.logger.info(f"Text extraction completed: {total_blocks} text blocks from {len(tiles)} tiles")
        
        if failed_tiles:
            self.logger.logger.warning(f"Failed to extract text from {len(failed_tiles)} tiles: {failed_tiles}")
        
        if successful_engines:
            self.logger.logger.info(f"Engine success rates: {successful_engines}")
        
        self.logger.save_intermediate_data({
            "total_text_blocks": total_blocks,
            "successful_engines": successful_engines,
            "failed_tiles": failed_tiles,
            "text_samples": [tb.text for tb in all_text_blocks[:10]]  # First 10 samples
        }, "02_text_extraction_summary")
        
        self.logger.log_step_end("text_extraction", success=True, text_blocks=total_blocks)
        return all_text_blocks
    
    def _extract_text_from_single_tile(self, tile_image: np.ndarray, tile_metadata: Dict) -> List[TextBlock]:
        """Extract text from a single tile with engine fallback"""
        
        # Try primary engine first
        if self.config.primary_engine in self.engines:
            engine = self.engines[self.config.primary_engine]
            text_blocks = engine.extract_text(tile_image, tile_metadata)
            
            self.logger.log_ocr_results(self.config.primary_engine.value, text_blocks, tile_metadata)
            
            if text_blocks:
                return text_blocks
            
            self.logger.logger.debug(f"Primary engine {self.config.primary_engine.value} failed, trying fallbacks")
        
        # Try fallback engines
        for fallback_engine in self.config.fallback_engines:
            if fallback_engine in self.engines and fallback_engine != self.config.primary_engine:
                engine = self.engines[fallback_engine]
                text_blocks = engine.extract_text(tile_image, tile_metadata)
                
                self.logger.log_ocr_results(fallback_engine.value, text_blocks, tile_metadata)
                
                if text_blocks:
                    self.logger.logger.info(f"Fallback engine {fallback_engine.value} succeeded")
                    return text_blocks
        
        # If all engines failed, try with different preprocessing
        self.logger.logger.debug("All engines failed, trying with enhanced preprocessing")
        enhanced_tile = self._enhance_tile_for_ocr(tile_image)
        
        # Try primary engine on enhanced image
        if self.config.primary_engine in self.engines:
            engine = self.engines[self.config.primary_engine]
            text_blocks = engine.extract_text(enhanced_tile, tile_metadata)
            
            if text_blocks:
                self.logger.logger.info(f"Enhanced preprocessing helped {self.config.primary_engine.value}")
                return text_blocks
        
        return []
    
    def _enhance_tile_for_ocr(self, tile_image: np.ndarray) -> np.ndarray:
        """Apply additional preprocessing to improve OCR results"""
        # Convert to grayscale
        gray = cv2.cvtColor(tile_image, cv2.COLOR_BGR2GRAY)
        
        # Apply Gaussian blur to reduce noise
        blurred = cv2.GaussianBlur(gray, (1, 1), 0)
        
        # Apply adaptive thresholding
        thresh = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                     cv2.THRESH_BINARY, 11, 2)
        
        # Morphological operations to clean up
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 1))
        cleaned = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
        
        # Convert back to BGR
        enhanced = cv2.cvtColor(cleaned, cv2.COLOR_GRAY2BGR)
        
        return enhanced
    
    def _convert_to_global_coordinates(self, text_blocks: List[TextBlock], tile_metadata: Dict) -> List[TextBlock]:
        """Convert text block coordinates from tile-local to global coordinates"""
        global_offset = tile_metadata['global_offset']
        offset_x = global_offset['x']
        offset_y = global_offset['y']
        
        global_text_blocks = []
        for text_block in text_blocks:
            # Create new text block with adjusted coordinates
            global_bbox = BoundingBox(
                text_block.bounding_box.x1 + offset_x,
                text_block.bounding_box.y1 + offset_y,
                text_block.bounding_box.x2 + offset_x,
                text_block.bounding_box.y2 + offset_y
            )
            
            global_text_block = TextBlock(
                id=text_block.id,
                text=text_block.text,
                bounding_box=global_bbox,
                confidence=text_block.confidence,
                ocr_engine=text_block.ocr_engine,
                font_size=text_block.font_size
            )
            
            global_text_blocks.append(global_text_block)
        
        return global_text_blocks
    
    def merge_overlapping_text_blocks(self, text_blocks: List[TextBlock]) -> List[TextBlock]:
        """
        Merge overlapping text blocks from different tiles
        This handles the case where text spans across tile boundaries
        """
        if not text_blocks:
            return []
        
        self.logger.log_step_start("merge_overlapping_text")
        
        # Sort by position (top-to-bottom, left-to-right)
        sorted_blocks = sorted(text_blocks, key=lambda tb: (tb.bounding_box.y1, tb.bounding_box.x1))
        
        merged_blocks = []
        skip_indices = set()
        
        for i, block1 in enumerate(sorted_blocks):
            if i in skip_indices:
                continue
            
            merged_text = block1.text
            merged_bbox = block1.bounding_box
            merged_confidence = block1.confidence
            merge_count = 1
            
            # Check for overlapping blocks
            for j, block2 in enumerate(sorted_blocks[i+1:], i+1):
                if j in skip_indices:
                    continue
                
                # Check if blocks overlap significantly
                if self._blocks_should_merge(block1, block2):
                    # Merge the blocks
                    merged_text = self._merge_text_intelligently(merged_text, block2.text)
                    merged_bbox = self._merge_bounding_boxes(merged_bbox, block2.bounding_box)
                    merged_confidence = (merged_confidence * merge_count + block2.confidence) / (merge_count + 1)
                    merge_count += 1
                    skip_indices.add(j)
            
            # Create merged block
            merged_block = TextBlock(
                text=merged_text,
                bounding_box=merged_bbox,
                confidence=merged_confidence,
                ocr_engine=block1.ocr_engine
            )
            
            merged_blocks.append(merged_block)
        
        reduction = len(text_blocks) - len(merged_blocks)
        self.logger.logger.info(f"Merged overlapping text blocks: {len(text_blocks)} -> {len(merged_blocks)} (reduced by {reduction})")
        
        self.logger.log_step_end("merge_overlapping_text", success=True, 
                                original_count=len(text_blocks), merged_count=len(merged_blocks))
        
        return merged_blocks
    
    def _blocks_should_merge(self, block1: TextBlock, block2: TextBlock) -> bool:
        """Determine if two text blocks should be merged"""
        bbox1, bbox2 = block1.bounding_box, block2.bounding_box
        
        # Check if they overlap
        if not bbox1.overlaps(bbox2):
            return False
        
        # Calculate overlap area
        overlap_x1 = max(bbox1.x1, bbox2.x1)
        overlap_y1 = max(bbox1.y1, bbox2.y1)
        overlap_x2 = min(bbox1.x2, bbox2.x2)
        overlap_y2 = min(bbox1.y2, bbox2.y2)
        
        overlap_area = max(0, overlap_x2 - overlap_x1) * max(0, overlap_y2 - overlap_y1)
        
        # Calculate overlap percentage for each block
        overlap_pct1 = overlap_area / bbox1.area if bbox1.area > 0 else 0
        overlap_pct2 = overlap_area / bbox2.area if bbox2.area > 0 else 0
        
        # Merge if significant overlap (>50% of either block)
        return overlap_pct1 > 0.5 or overlap_pct2 > 0.5
    
    def _merge_text_intelligently(self, text1: str, text2: str) -> str:
        """Intelligently merge text from overlapping blocks"""
        # If one text contains the other, use the longer one
        if text1 in text2:
            return text2
        elif text2 in text1:
            return text1
        
        # If they're similar, use the one with higher confidence
        # For now, just concatenate with a space
        return f"{text1} {text2}".strip()
    
    def _merge_bounding_boxes(self, bbox1: BoundingBox, bbox2: BoundingBox) -> BoundingBox:
        """Merge two bounding boxes to encompass both"""
        return BoundingBox(
            min(bbox1.x1, bbox2.x1),
            min(bbox1.y1, bbox2.y1),
            max(bbox1.x2, bbox2.x2),
            max(bbox1.y2, bbox2.y2)
        )