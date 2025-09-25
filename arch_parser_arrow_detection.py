# arrow_detection.py
"""
Step 3: Arrow Detection and Relationship Extraction
Detects arrows, lines, and other connection elements in architecture diagrams
"""
import cv2
import numpy as np
from typing import List, Dict, Tuple, Optional
import math
from scipy.spatial.distance import cdist
from sklearn.cluster import DBSCAN

from arch_parser_config import ArrowDetectionConfig
from diagnostics import DiagnosticsLogger, performance_monitor
from arch_parser_models import Arrow, Point, BoundingBox, TextBlock

class ArrowDetector:
    """Detects arrows and connections in architecture diagrams"""
    
    def __init__(self, config: ArrowDetectionConfig, logger: DiagnosticsLogger):
        self.config = config
        self.logger = logger
    
    @performance_monitor("detect_arrows_from_tiles")
    def detect_arrows_from_tiles(self, tiles: List[Tuple[np.ndarray, Dict]], 
                                text_blocks: List[TextBlock]) -> List[Arrow]:
        """
        Detect arrows from all tiles and merge results
        
        Args:
            tiles: List of (tile_image, tile_metadata) tuples
            text_blocks: Text blocks for labeling arrows
            
        Returns:
            List of Arrow objects with global coordinates
        """
        self.logger.log_step_start("arrow_detection", tile_count=len(tiles))
        
        all_arrows = []
        
        for tile_idx, (tile_image, tile_metadata) in enumerate(tiles):
            self.logger.logger.debug(f"Detecting arrows in tile {tile_idx + 1}/{len(tiles)}")
            
            # Detect arrows in this tile
            tile_arrows = self._detect_arrows_in_tile(tile_image, tile_metadata)
            
            # Convert to global coordinates
            global_arrows = self._convert_arrows_to_global_coordinates(tile_arrows, tile_metadata)
            all_arrows.extend(global_arrows)
        
        # Merge overlapping arrows from different tiles
        merged_arrows = self._merge_overlapping_arrows(all_arrows)
        
        # Add text labels to arrows
        labeled_arrows = self._add_text_labels_to_arrows(merged_arrows, text_blocks)
        
        arrow_count = len(labeled_arrows)
        self.logger.logger.info(f"Arrow detection completed: {arrow_count} arrows detected")
        
        self.logger.save_intermediate_data({
            "total_arrows": arrow_count,
            "arrows_per_tile": len(all_arrows) / len(tiles) if tiles else 0,
            "arrows_with_labels": sum(1 for a in labeled_arrows if a.text_labels)
        }, "03_arrow_detection_summary")
        
        self.logger.log_step_end("arrow_detection", success=True, arrow_count=arrow_count)
        return labeled_arrows
    
    def _detect_arrows_in_tile(self, tile_image: np.ndarray, tile_metadata: Dict) -> List[Arrow]:
        """Detect arrows in a single tile"""
        # Convert to grayscale
        gray = cv2.cvtColor(tile_image, cv2.COLOR_BGR2GRAY)
        
        # Step 1: Detect line segments (arrow shafts)
        line_segments = self._detect_line_segments(gray)
        
        # Step 2: Detect arrowheads
        arrowheads = self._detect_arrowheads(gray)
        
        # Step 3: Link arrowheads to line segments
        arrows = self._link_arrowheads_to_lines(line_segments, arrowheads)
        
        # Step 4: Detect curved arrows (splines/bezier curves)
        curved_arrows = self._detect_curved_arrows(gray)
        arrows.extend(curved_arrows)
        
        self.logger.log_arrow_detection_results(line_segments, arrowheads, arrows)
        
        # Save debug visualization if enabled
        if self.logger.config.save_intermediate_results:
            debug_image = self._create_arrow_debug_visualization(tile_image, line_segments, arrowheads, arrows)
            tile_id = tile_metadata.get('tile_id', 'unknown')
            self.logger.save_intermediate_image(debug_image, f"03_arrows_tile_{tile_id:03d}")
        
        return arrows
    
    def _detect_line_segments(self, gray_image: np.ndarray) -> List[Tuple[Point, Point]]:
        """Detect straight line segments using Hough Line Transform"""
        # Edge detection
        edges = cv2.Canny(gray_image, 50, 150, apertureSize=3)
        
        # Hough line detection
        lines = cv2.HoughLinesP(
            edges,
            rho=1,
            theta=np.pi/180,
            threshold=self.config.hough_threshold,
            minLineLength=self.config.min_line_length,
            maxLineGap=self.config.max_line_gap
        )
        
        line_segments = []
        if lines is not None:
            for line in lines:
                x1, y1, x2, y2 = line[0]
                line_segments.append((Point(x1, y1), Point(x2, y2)))
        
        return line_segments
    
    def _detect_arrowheads(self, gray_image: np.ndarray) -> List[Point]:
        """Detect arrowheads using template matching and contour analysis"""
        arrowheads = []
        
        # Method 1: Template matching for standard arrowheads
        template_arrowheads = self._detect_arrowheads_by_template(gray_image)
        arrowheads.extend(template_arrowheads)
        
        # Method 2: Contour-based detection for various arrowhead shapes
        contour_arrowheads = self._detect_arrowheads_by_contour(gray_image)
        arrowheads.extend(contour_arrowheads)
        
        # Remove duplicates
        arrowheads = self._remove_duplicate_points(arrowheads, min_distance=10)
        
        return arrowheads
    
    def _detect_arrowheads_by_template(self, gray_image: np.ndarray) -> List[Point]:
        """Detect arrowheads using template matching"""
        arrowheads = []
        
        # Create simple arrowhead templates
        templates = self._create_arrowhead_templates()
        
        for template in templates:
            # Template matching
            result = cv2.matchTemplate(gray_image, template, cv2.TM_CCOEFF_NORMED)
            
            # Find matches above threshold
            threshold = 0.6
            locations = np.where(result >= threshold)
            
            for pt in zip(*locations[::-1]):  # Switch x and y coordinates
                center_x = pt[0] + template.shape[1] // 2
                center_y = pt[1] + template.shape[0] // 2
                arrowheads.append(Point(center_x, center_y))
        
        return arrowheads
    
    def _detect_arrowheads_by_contour(self, gray_image: np.ndarray) -> List[Point]:
        """Detect arrowheads by analyzing contours for triangular shapes"""
        arrowheads = []
        
        # Edge detection and contours
        edges = cv2.Canny(gray_image, 100, 200)
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        for contour in contours:
            # Filter by area
            area = cv2.contourArea(contour)
            if area < 20 or area > 500:  # Reasonable arrowhead size
                continue
            
            # Approximate contour
            epsilon = 0.02 * cv2.arcLength(contour, True)
            approx = cv2.approxPolyDP(contour, epsilon, True)
            
            # Look for triangular shapes (3-4 vertices)
            if len(approx) >= 3 and len(approx) <= 5:
                # Check if shape is roughly triangular
                if self._is_triangular_shape(approx):
                    # Get centroid
                    M = cv2.moments(contour)
                    if M["m00"] != 0:
                        cx = int(M["m10"] / M["m00"])
                        cy = int(M["m01"] / M["m00"])
                        arrowheads.append(Point(cx, cy))
        
        return arrowheads
    
    def _create_arrowhead_templates(self) -> List[np.ndarray]:
        """Create arrowhead templates for template matching"""
        templates = []
        
        for size in self.config.arrowhead_template_sizes:
            # Right-pointing arrow
            template = np.zeros((size, size), dtype=np.uint8)
            pts = np.array([[size-1, size//2], [0, 0], [0, size-1]], np.int32)
            cv2.fillPoly(template, [pts], 255)
            templates.append(template)
            
            # Rotate for other directions
            for angle in [90, 180, 270]:
                center = (size // 2, size // 2)
                rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
                rotated = cv2.warpAffine(template, rotation_matrix, (size, size))
                templates.append(rotated)
        
        return templates
    
    def _is_triangular_shape(self, approx_contour) -> bool:
        """Check if a contour approximates a triangular shape"""
        if len(approx_contour) < 3:
            return False
        
        # Calculate angles between consecutive edges
        points = approx_contour.reshape(-1, 2)
        angles = []
        
        for i in range(len(points)):
            p1 = points[i - 1]
            p2 = points[i]
            p3 = points[(i + 1) % len(points)]
            
            # Calculate angle at p2
            v1 = p1 - p2
            v2 = p3 - p2
            
            # Calculate angle
            cos_angle = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-6)
            angle = np.arccos(np.clip(cos_angle, -1, 1)) * 180 / np.pi
            angles.append(angle)
        
        # Check if we have at least one sharp angle (like arrowhead tip)
        has_sharp_angle = any(angle < 60 for angle in angles)
        
        return has_sharp_angle
    
    def _link_arrowheads_to_lines(self, line_segments: List[Tuple[Point, Point]], 
                                 arrowheads: List[Point]) -> List[Arrow]:
        """Link detected arrowheads to line segments to form complete arrows"""
        arrows = []
        used_arrowheads = set()
        
        for line_start, line_end in line_segments:
            # Check both ends of the line for nearby arrowheads
            for arrowhead in arrowheads:
                if id(arrowhead) in used_arrowheads:
                    continue
                
                # Calculate distances to both ends
                dist_to_start = line_start.distance_to(arrowhead)
                dist_to_end = line_end.distance_to(arrowhead)
                
                # Check if arrowhead is close to either end
                if dist_to_start <= self.config.connection_distance_threshold:
                    # Arrowhead at start - arrow points from end to start
                    arrow = Arrow(
                        start_point=line_end,
                        end_point=line_start,
                        shaft_points=[line_end, line_start],
                        confidence=0.8
                    )
                    arrows.append(arrow)
                    used_arrowheads.add(id(arrowhead))
                    break
                    
                elif dist_to_end <= self.config.connection_distance_threshold:
                    # Arrowhead at end - arrow points from start to end
                    arrow = Arrow(
                        start_point=line_start,
                        end_point=line_end,
                        shaft_points=[line_start, line_end],
                        confidence=0.8
                    )
                    arrows.append(arrow)
                    used_arrowheads.add(id(arrowhead))
                    break
        
        # Create arrows for lines without detected arrowheads (assume direction)
        for line_start, line_end in line_segments:
            # Check if this line is already part of an arrow
            line_used = any(
                (arrow.start_point.distance_to(line_start) < 5 and arrow.end_point.distance_to(line_end) < 5) or
                (arrow.start_point.distance_to(line_end) < 5 and arrow.end_point.distance_to(line_start) < 5)
                for arrow in arrows
            )
            
            if not line_used:
                # Create arrow assuming left-to-right or top-to-bottom direction
                arrow = Arrow(
                    start_point=line_start,
                    end_point=line_end,
                    shaft_points=[line_start, line_end],
                    confidence=0.5  # Lower confidence for assumed direction
                )
                arrows.append(arrow)
        
        return arrows
    
    def _detect_curved_arrows(self, gray_image: np.ndarray) -> List[Arrow]:
        """Detect curved arrows (splines, bezier curves)"""
        curved_arrows = []
        
        # This is a simplified implementation
        # In a full production system, you might want to use more sophisticated
        # curve detection algorithms like contour analysis with curve fitting
        
        # Edge detection
        edges = cv2.Canny(gray_image, 50, 150)
        
        # Find contours
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        for contour in contours:
            # Filter contours that might be curved arrows
            arc_length = cv2.arcLength(contour, False)
            area = cv2.contourArea(contour)
            
            # Skip if too small or too large
            if arc_length < 50 or arc_length > 500:
                continue
            
            # Check if contour is approximately a curve (high arc length to area ratio)
            if area > 0 and arc_length / area > 2:
                # Approximate as arrow
                points = contour.reshape(-1, 2)
                if len(points) >= 2:
                    start_point = Point(points[0][0], points[0][1])
                    end_point = Point(points[-1][0], points[-1][1])
                    
                    # Convert all points to Point objects
                    shaft_points = [Point(pt[0], pt[1]) for pt in points]
                    
                    curved_arrow = Arrow(
                        start_point=start_point,
                        end_point=end_point,
                        shaft_points=shaft_points,
                        confidence=0.6
                    )
                    curved_arrows.append(curved_arrow)
        
        return curved_arrows
    
    def _convert_arrows_to_global_coordinates(self, arrows: List[Arrow], 
                                            tile_metadata: Dict) -> List[Arrow]:
        """Convert arrow coordinates from tile-local to global coordinates"""
        global_offset = tile_metadata['global_offset']
        offset_x = global_offset['x']
        offset_y = global_offset['y']
        
        global_arrows = []
        for arrow in arrows:
            # Convert points to global coordinates
            global_start = Point(arrow.start_point.x + offset_x, arrow.start_point.y + offset_y)
            global_end = Point(arrow.end_point.x + offset_x, arrow.end_point.y + offset_y)
            global_shaft_points = [
                Point(pt.x + offset_x, pt.y + offset_y) for pt in arrow.shaft_points
            ]
            
            global_arrow = Arrow(
                id=arrow.id,
                start_point=global_start,
                end_point=global_end,
                shaft_points=global_shaft_points,
                confidence=arrow.confidence,
                text_labels=arrow.text_labels.copy()
            )
            
            global_arrows.append(global_arrow)
        
        return global_arrows
    
    def _merge_overlapping_arrows(self, arrows: List[Arrow]) -> List[Arrow]:
        """Merge arrows that represent the same connection across tile boundaries"""
        if not arrows:
            return []
        
        self.logger.log_step_start("merge_overlapping_arrows")
        
        # Use DBSCAN clustering to group similar arrows
        # Features: start point, end point, direction
        features = []
        for arrow in arrows:
            features.append([
                arrow.start_point.x, arrow.start_point.y,
                arrow.end_point.x, arrow.end_point.y,
                arrow.length
            ])
        
        if len(features) < 2:
            return arrows
        
        # Cluster similar arrows
        clustering = DBSCAN(eps=30, min_samples=1).fit(features)
        labels = clustering.labels_
        
        # Merge arrows in the same cluster
        clusters = {}
        for i, label in enumerate(labels):
            if label not in clusters:
                clusters[label] = []
            clusters[label].append(arrows[i])
        
        merged_arrows = []
        for cluster_arrows in clusters.values():
            if len(cluster_arrows) == 1:
                merged_arrows.append(cluster_arrows[0])
            else:
                # Merge multiple arrows
                merged_arrow = self._merge_arrow_cluster(cluster_arrows)
                merged_arrows.append(merged_arrow)
        
        reduction = len(arrows) - len(merged_arrows)
        self.logger.logger.info(f"Merged overlapping arrows: {len(arrows)} -> {len(merged_arrows)} (reduced by {reduction})")
        
        self.logger.log_step_end("merge_overlapping_arrows", success=True,
                                original_count=len(arrows), merged_count=len(merged_arrows))
        
        return merged_arrows
    
    def _merge_arrow_cluster(self, cluster_arrows: List[Arrow]) -> Arrow:
        """Merge multiple arrows that represent the same connection"""
        # Use the arrow with highest confidence as base
        base_arrow = max(cluster_arrows, key=lambda a: a.confidence)
        
        # Merge text labels
        all_labels = set()
        for arrow in cluster_arrows:
            all_labels.update(arrow.text_labels)
        
        # Calculate average confidence
        avg_confidence = sum(arrow.confidence for arrow in cluster_arrows) / len(cluster_arrows)
        
        merged_arrow = Arrow(
            start_point=base_arrow.start_point,
            end_point=base_arrow.end_point,
            shaft_points=base_arrow.shaft_points,
            confidence=avg_confidence,
            text_labels=list(all_labels)
        )
        
        return merged_arrow
    
    def _add_text_labels_to_arrows(self, arrows: List[Arrow], 
                                  text_blocks: List[TextBlock]) -> List[Arrow]:
        """Add text labels to arrows by finding nearby text"""
        self.logger.log_step_start("add_text_labels_to_arrows")
        
        for arrow in arrows:
            # Find text blocks near the arrow
            nearby_texts = self._find_text_near_arrow(arrow, text_blocks)
            arrow.text_labels.extend(nearby_texts)
        
        labeled_count = sum(1 for arrow in arrows if arrow.text_labels)
        self.logger.logger.info(f"Added labels to {labeled_count}/{len(arrows)} arrows")
        
        self.logger.log_step_end("add_text_labels_to_arrows", success=True,
                                labeled_arrows=labeled_count, total_arrows=len(arrows))
        
        return arrows
    
    def _find_text_near_arrow(self, arrow: Arrow, text_blocks: List[TextBlock]) -> List[str]:
        """Find text blocks that are near an arrow (likely labels)"""
        nearby_texts = []
        search_distance = 50  # pixels
        
        # Calculate arrow's center line
        arrow_center_x = (arrow.start_point.x + arrow.end_point.x) / 2
        arrow_center_y = (arrow.start_point.y + arrow.end_point.y) / 2
        
        for text_block in text_blocks:
            text_center = text_block.bounding_box.center
            
            # Calculate distance from text center to arrow center
            distance = math.sqrt(
                (text_center[0] - arrow_center_x) ** 2 + 
                (text_center[1] - arrow_center_y) ** 2
            )
            
            # Also check distance to arrow line
            line_distance = self._point_to_line_distance(
                Point(text_center[0], text_center[1]), 
                arrow.start_point, 
                arrow.end_point
            )
            
            # If text is close to arrow center or arrow line
            if distance <= search_distance or line_distance <= search_distance:
                # Filter out very long text (likely not arrow labels)
                if len(text_block.text) <= 50:
                    nearby_texts.append(text_block.text.strip())
        
        return nearby_texts
    
    def _point_to_line_distance(self, point: Point, line_start: Point, line_end: Point) -> float:
        """Calculate perpendicular distance from point to line segment"""
        # Vector from line_start to line_end
        line_vec = Point(line_end.x - line_start.x, line_end.y - line_start.y)
        
        # Vector from line_start to point
        point_vec = Point(point.x - line_start.x, point.y - line_start.y)
        
        # Length of line segment
        line_length = line_start.distance_to(line_end)
        
        if line_length == 0:
            return line_start.distance_to(point)
        
        # Project point onto line
        dot_product = (point_vec.x * line_vec.x + point_vec.y * line_vec.y)
        projection_length = dot_product / (line_length ** 2)
        
        # Clamp projection to line segment
        projection_length = max(0, min(1, projection_length))
        
        # Find closest point on line segment
        closest_x = line_start.x + projection_length * line_vec.x
        closest_y = line_start.y + projection_length * line_vec.y
        closest_point = Point(int(closest_x), int(closest_y))
        
        return point.distance_to(closest_point)
    
    def _remove_duplicate_points(self, points: List[Point], min_distance: int = 10) -> List[Point]:
        """Remove duplicate points that are too close together"""
        if not points:
            return []
        
        unique_points = [points[0]]
        
        for point in points[1:]:
            # Check if this point is too close to any existing unique point
            too_close = any(
                point.distance_to(existing) < min_distance 
                for existing in unique_points
            )
            
            if not too_close:
                unique_points.append(point)
        
        return unique_points
    
    def _create_arrow_debug_visualization(self, tile_image: np.ndarray, 
                                        line_segments: List[Tuple[Point, Point]],
                                        arrowheads: List[Point],
                                        arrows: List[Arrow]) -> np.ndarray:
        """Create debug visualization showing detected arrows"""
        debug_image = tile_image.copy()
        
        # Draw line segments in blue
        for line_start, line_end in line_segments:
            cv2.line(debug_image, line_start.to_tuple(), line_end.to_tuple(), (255, 0, 0), 2)
        
        # Draw arrowheads in red
        for arrowhead in arrowheads:
            cv2.circle(debug_image, arrowhead.to_tuple(), 5, (0, 0, 255), -1)
        
        # Draw complete arrows in green
        for arrow in arrows:
            # Draw shaft
            if len(arrow.shaft_points) >= 2:
                for i in range(len(arrow.shaft_points) - 1):
                    cv2.line(debug_image, 
                           arrow.shaft_points[i].to_tuple(),
                           arrow.shaft_points[i + 1].to_tuple(),
                           (0, 255, 0), 3)
            
            # Draw arrow direction
            cv2.arrowedLine(debug_image,
                          arrow.start_point.to_tuple(),
                          arrow.end_point.to_tuple(),
                          (0, 255, 0), 2, tipLength=0.3)
            
            # Add confidence text
            mid_x = (arrow.start_point.x + arrow.end_point.x) // 2
            mid_y = (arrow.start_point.y + arrow.end_point.y) // 2
            cv2.putText(debug_image, f"{arrow.confidence:.2f}",
                       (mid_x, mid_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        
        return debug_image

class ConnectionAnalyzer:
    """Analyzes detected arrows to determine connection types and properties"""
    
    def __init__(self, logger: DiagnosticsLogger):
        self.logger = logger
    
    @performance_monitor("analyze_connections")
    def analyze_connections(self, arrows: List[Arrow], text_blocks: List[TextBlock]) -> List[Arrow]:
        """
        Analyze arrows to determine connection types, protocols, and properties
        
        Args:
            arrows: List of detected arrows
            text_blocks: All text blocks for context analysis
            
        Returns:
            List of arrows with enhanced metadata
        """
        self.logger.log_step_start("analyze_connections", arrow_count=len(arrows))
        
        enhanced_arrows = []
        
        for arrow in arrows:
            enhanced_arrow = self._analyze_single_connection(arrow, text_blocks)
            enhanced_arrows.append(enhanced_arrow)
        
        self.logger.log_step_end("analyze_connections", success=True)
        return enhanced_arrows
    
    def _analyze_single_connection(self, arrow: Arrow, text_blocks: List[TextBlock]) -> Arrow:
        """Analyze a single arrow to determine its properties"""
        # Copy the arrow to avoid modifying the original
        enhanced_arrow = Arrow(
            id=arrow.id,
            start_point=arrow.start_point,
            end_point=arrow.end_point,
            shaft_points=arrow.shaft_points.copy(),
            confidence=arrow.confidence,
            text_labels=arrow.text_labels.copy()
        )
        
        # Analyze text labels for technical information
        self._extract_technical_info_from_labels(enhanced_arrow)
        
        # Analyze arrow style (thickness, color, pattern) if available
        self._analyze_arrow_style(enhanced_arrow)
        
        return enhanced_arrow
    
    def _extract_technical_info_from_labels(self, arrow: Arrow):
        """Extract technical information from arrow labels"""
        all_text = " ".join(arrow.text_labels).lower()
        
        # Protocol detection
        protocols = {
            'http': ['http', 'https', 'rest', 'api'],
            'tcp': ['tcp'],
            'udp': ['udp'],
            'sql': ['sql', 'database', 'db'],
            'ssh': ['ssh'],
            'ftp': ['ftp', 'sftp'],
            'smtp': ['smtp', 'email', 'mail'],
            'dns': ['dns'],
            'dhcp': ['dhcp']
        }
        
        detected_protocols = []
        for protocol, keywords in protocols.items():
            if any(keyword in all_text for keyword in keywords):
                detected_protocols.append(protocol)
        
        # Store in text_labels for now (in a full implementation, Arrow class would have protocol field)
        if detected_protocols:
            arrow.text_labels.extend([f"protocol:{p}" for p in detected_protocols])
        
        # Port detection
        import re
        port_matches = re.findall(r':(\d{2,5})\b', all_text)
        if port_matches:
            arrow.text_labels.extend([f"port:{port}" for port in port_matches])
        
        # Direction indicators
        direction_keywords = {
            'request': 'outbound',
            'response': 'inbound',
            'send': 'outbound',
            'receive': 'inbound',
            'upload': 'outbound',
            'download': 'inbound'
        }
        
        for keyword, direction in direction_keywords.items():
            if keyword in all_text:
                arrow.text_labels.append(f"direction:{direction}")
                break
    
    def _analyze_arrow_style(self, arrow: Arrow):
        """Analyze arrow visual style to infer connection type"""
        # This would require more advanced image analysis
        # For now, we'll use basic length and straightness analysis
        
        # Analyze arrow length
        length = arrow.length
        if length < 50:
            arrow.text_labels.append("style:short_connection")
        elif length > 200:
            arrow.text_labels.append("style:long_connection")
        
        # Analyze straightness (for curved vs straight arrows)
        if len(arrow.shaft_points) > 2:
            # Calculate total path length vs direct distance
            path_length = sum(
                arrow.shaft_points[i].distance_to(arrow.shaft_points[i + 1])
                for i in range(len(arrow.shaft_points) - 1)
            )
            direct_length = arrow.start_point.distance_to(arrow.end_point)
            
            if path_length > direct_length * 1.2:  # 20% longer than direct path
                arrow.text_labels.append("style:curved")
            else:
                arrow.text_labels.append("style:straight")