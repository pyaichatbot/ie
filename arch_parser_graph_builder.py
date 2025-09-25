# graph_builder.py
"""
Step 4: Graph Construction and Node-Edge Mapping
Converts detected text blocks and arrows into a structured graph representation
"""
import numpy as np
from typing import List, Dict, Tuple, Optional, Set, Any
import re
from sklearn.cluster import DBSCAN
# Removed unused imports: pdist, squareform

from arch_parser_config import GraphProcessingConfig
from diagnostics import DiagnosticsLogger, performance_monitor
from arch_parser_models import (
    ArchitectureGraph, Node, Edge, TextBlock, Arrow, BoundingBox, Point,
    NodeType, EdgeType
)

class GraphBuilder:
    """Builds structured graph from detected text blocks and arrows"""
    
    def __init__(self, config: GraphProcessingConfig, logger: DiagnosticsLogger):
        self.config = config
        self.logger = logger
    
    @performance_monitor("build_graph")
    def build_graph(self, text_blocks: List[TextBlock], arrows: List[Arrow], 
                   diagram_name: str = "") -> ArchitectureGraph:
        """
        Build complete architecture graph from detected elements
        
        Args:
            text_blocks: Detected text blocks
            arrows: Detected arrows
            diagram_name: Name/title of the diagram
            
        Returns:
            Complete ArchitectureGraph object
        """
        self.logger.log_step_start("graph_building", 
                                  text_blocks=len(text_blocks), arrows=len(arrows))
        
        # Step 1: Create nodes from text blocks
        nodes = self._create_nodes_from_text_blocks(text_blocks)
        
        # Step 2: Merge nearby nodes that represent the same entity
        merged_nodes = self._merge_nearby_nodes(nodes)
        
        # Step 3: Classify node types
        classified_nodes = self._classify_nodes(merged_nodes)
        
        # Step 4: Create edges from arrows
        edges = self._create_edges_from_arrows(arrows, classified_nodes)
        
        # Step 5: Validate and clean the graph
        validated_nodes, validated_edges = self._validate_graph(classified_nodes, edges)
        
        # Step 6: Create final graph
        graph = ArchitectureGraph(
            name=diagram_name,
            nodes=validated_nodes,
            edges=validated_edges
        )
        
        # Add processing metadata
        graph.processing_info = {
            "original_text_blocks": len(text_blocks),
            "original_arrows": len(arrows),
            "final_nodes": len(validated_nodes),
            "final_edges": len(validated_edges),
            "node_types": self._get_node_type_counts(validated_nodes),
            "edge_types": self._get_edge_type_counts(validated_edges)
        }
        
        self.logger.logger.info(
            f"Graph built successfully: {len(validated_nodes)} nodes, {len(validated_edges)} edges"
        )
        
        self.logger.save_intermediate_data(graph.to_dict(), "04_final_graph")
        self.logger.log_step_end("graph_building", success=True, 
                                nodes=len(validated_nodes), edges=len(validated_edges))
        
        return graph
    
    def _create_nodes_from_text_blocks(self, text_blocks: List[TextBlock]) -> List[Node]:
        """Convert text blocks into preliminary nodes"""
        self.logger.log_step_start("create_nodes_from_text_blocks")
        
        nodes = []
        for text_block in text_blocks:
            # Skip very small or empty text blocks
            if (text_block.bounding_box.area < self.config.min_node_area or 
                not text_block.text.strip()):
                continue
            
            # Create node
            node = Node(
                name=text_block.text.strip(),
                bounding_box=text_block.bounding_box,
                confidence=text_block.confidence
            )
            
            # Add the text block to the node
            node.add_text_block(text_block)
            
            nodes.append(node)
        
        self.logger.log_step_end("create_nodes_from_text_blocks", 
                                success=True, node_count=len(nodes))
        return nodes
    
    def _merge_nearby_nodes(self, nodes: List[Node]) -> List[Node]:
        """Merge nodes that are close together and likely represent the same entity"""
        if not nodes:
            return []
        
        self.logger.log_step_start("merge_nearby_nodes")
        
        # Create feature matrix for clustering (center coordinates)
        features = []
        for node in nodes:
            center = node.center
            features.append([center.x, center.y])
        
        if len(features) < 2:
            return nodes
        
        # Use DBSCAN clustering to group nearby nodes
        clustering = DBSCAN(
            eps=self.config.node_merge_distance, 
            min_samples=1
        ).fit(features)
        
        # Group nodes by cluster
        clusters = {}
        for i, label in enumerate(clustering.labels_):
            if label not in clusters:
                clusters[label] = []
            clusters[label].append(nodes[i])
        
        # Merge nodes in each cluster
        merged_nodes = []
        for cluster_nodes in clusters.values():
            if len(cluster_nodes) == 1:
                merged_nodes.append(cluster_nodes[0])
            else:
                merged_node = self._merge_node_cluster(cluster_nodes)
                merged_nodes.append(merged_node)
        
        reduction = len(nodes) - len(merged_nodes)
        self.logger.logger.info(f"Merged nearby nodes: {len(nodes)} -> {len(merged_nodes)} (reduced by {reduction})")
        
        self.logger.log_step_end("merge_nearby_nodes", success=True,
                                original_count=len(nodes), merged_count=len(merged_nodes))
        
        return merged_nodes
    
    def _merge_node_cluster(self, cluster_nodes: List[Node]) -> Node:
        """Merge multiple nodes that represent the same entity"""
        # Use the node with the largest bounding box as the base
        base_node = max(cluster_nodes, key=lambda n: n.bounding_box.area)
        
        # Merge bounding boxes
        all_boxes = [node.bounding_box for node in cluster_nodes]
        merged_bbox = self._merge_bounding_boxes(all_boxes)
        
        # Merge text blocks
        all_text_blocks = []
        all_texts = []
        for node in cluster_nodes:
            all_text_blocks.extend(node.text_blocks)
            if node.name.strip():
                all_texts.append(node.name.strip())
        
        # Choose the most descriptive name
        merged_name = self._choose_best_node_name(all_texts)
        
        # Calculate average confidence
        avg_confidence = sum(node.confidence for node in cluster_nodes) / len(cluster_nodes)
        
        # Create merged node
        merged_node = Node(
            name=merged_name,
            bounding_box=merged_bbox,
            confidence=avg_confidence,
            text_blocks=all_text_blocks
        )
        
        # Merge technical attributes
        self._merge_technical_attributes(merged_node, cluster_nodes)
        
        return merged_node
    
    def _merge_bounding_boxes(self, bboxes: List[BoundingBox]) -> BoundingBox:
        """Merge multiple bounding boxes into one that encompasses all"""
        x1 = min(bbox.x1 for bbox in bboxes)
        y1 = min(bbox.y1 for bbox in bboxes)
        x2 = max(bbox.x2 for bbox in bboxes)
        y2 = max(bbox.y2 for bbox in bboxes)
        return BoundingBox(x1, y1, x2, y2)
    
    def _choose_best_node_name(self, names: List[str]) -> str:
        """Choose the most descriptive name from a list"""
        if not names:
            return "Unknown"
        
        # Remove duplicates while preserving order
        unique_names = list(dict.fromkeys(names))
        
        # Prefer longer, more descriptive names
        # But avoid overly long concatenations
        best_name = max(unique_names, key=len)
        
        # If the best name is too long, try to find a good shorter one
        if len(best_name) > 50:
            shorter_names = [name for name in unique_names if len(name) <= 30]
            if shorter_names:
                best_name = max(shorter_names, key=len)
        
        return best_name
    
    def _merge_technical_attributes(self, merged_node: Node, source_nodes: List[Node]):
        """Merge technical attributes from source nodes"""
        # Collect all technical attributes
        ip_addresses = [node.ip_address for node in source_nodes if node.ip_address]
        hostnames = [node.hostname for node in source_nodes if node.hostname]
        ports = [node.port for node in source_nodes if node.port]
        technologies = [node.technology for node in source_nodes if node.technology]
        
        # Use the first/best value for each attribute
        if ip_addresses:
            merged_node.ip_address = ip_addresses[0]
        if hostnames:
            merged_node.hostname = hostnames[0]
        if ports:
            merged_node.port = ports[0]
        if technologies:
            merged_node.technology = technologies[0]
        
        # Merge metadata
        for node in source_nodes:
            merged_node.metadata.update(node.metadata)
    
    def _classify_nodes(self, nodes: List[Node]) -> List[Node]:
        """Classify nodes by type based on their text content and attributes"""
        self.logger.log_step_start("classify_nodes")
        
        for node in nodes:
            node.node_type = self._determine_node_type(node)
        
        # Log classification statistics
        type_counts = self._get_node_type_counts(nodes)
        self.logger.logger.info(f"Node classification completed: {type_counts}")
        
        self.logger.log_step_end("classify_nodes", success=True, **type_counts)
        return nodes
    
    def _determine_node_type(self, node: Node) -> NodeType:
        """Determine the type of a node based on its content"""
        text_content = node.name.lower()
        
        # Combine all text from text blocks
        all_text = " ".join([tb.text.lower() for tb in node.text_blocks])
        combined_text = f"{text_content} {all_text}"
        
        # Define keywords for each node type
        type_keywords = {
            NodeType.SERVER: ['server', 'srv', 'host', 'machine', 'vm', 'instance'],
            NodeType.DATABASE: ['database', 'db', 'mysql', 'postgresql', 'postgres', 'oracle', 'mongodb', 'sql'],
            NodeType.LOAD_BALANCER: ['load balancer', 'lb', 'balancer', 'f5', 'haproxy', 'nginx'],
            NodeType.FIREWALL: ['firewall', 'fw', 'asa', 'palo alto', 'checkpoint', 'security'],
            NodeType.ROUTER: ['router', 'cisco', 'juniper', 'gateway'],
            NodeType.SWITCH: ['switch', 'layer 2', 'l2'],
            NodeType.STORAGE: ['storage', 'disk', 'nas', 'san', 'volume'],
            NodeType.CLOUD: ['cloud', 'aws', 'azure', 'gcp', 'ec2', 's3'],
            NodeType.APPLICATION: ['app', 'application', 'service', 'api'],
            NodeType.USER: ['user', 'client', 'browser', 'mobile'],
            NodeType.NETWORK: ['network', 'subnet', 'vlan', 'dmz'],
            NodeType.INTERNET: ['internet', 'web', 'public', 'external']
        }
        
        # Score each type based on keyword matches
        type_scores = {}
        for node_type, keywords in type_keywords.items():
            score = sum(1 for keyword in keywords if keyword in combined_text)
            if score > 0:
                type_scores[node_type] = score
        
        # Additional scoring based on technical attributes
        if node.ip_address:
            type_scores[NodeType.SERVER] = type_scores.get(NodeType.SERVER, 0) + 2
        
        if node.technology:
            tech = node.technology.lower()
            if 'database' in tech:
                type_scores[NodeType.DATABASE] = type_scores.get(NodeType.DATABASE, 0) + 3
            elif 'web_server' in tech:
                type_scores[NodeType.APPLICATION] = type_scores.get(NodeType.APPLICATION, 0) + 2
            elif 'load_balancer' in tech:
                type_scores[NodeType.LOAD_BALANCER] = type_scores.get(NodeType.LOAD_BALANCER, 0) + 3
            elif 'firewall' in tech:
                type_scores[NodeType.FIREWALL] = type_scores.get(NodeType.FIREWALL, 0) + 3
        
        # Return the type with the highest score
        if type_scores:
            return max(type_scores.items(), key=lambda x: x[1])[0]
        
        # Default classification based on shape/size heuristics
        bbox = node.bounding_box
        if bbox.width > bbox.height * 2:  # Wide rectangles often represent services
            return NodeType.SERVICE
        elif bbox.width < bbox.height * 0.5:  # Tall rectangles might be servers
            return NodeType.SERVER
        
        return NodeType.GENERIC
    
    def _create_edges_from_arrows(self, arrows: List[Arrow], nodes: List[Node]) -> List[Edge]:
        """Create edges by connecting arrows to nodes"""
        self.logger.log_step_start("create_edges_from_arrows")
        
        edges = []
        
        for arrow in arrows:
            # Find source and target nodes
            source_node = self._find_closest_node(arrow.start_point, nodes)
            target_node = self._find_closest_node(arrow.end_point, nodes)
            
            # Skip if we couldn't find both nodes
            if not source_node or not target_node or source_node.id == target_node.id:
                self.logger.logger.debug(f"Skipping arrow - couldn't find valid source/target nodes")
                continue
            
            # Create edge
            edge = Edge(
                source_node_id=source_node.id,
                target_node_id=target_node.id,
                arrow=arrow,
                labels=arrow.text_labels.copy(),
                confidence=arrow.confidence
            )
            
            # Classify edge type
            edge.edge_type = self._determine_edge_type(edge, source_node, target_node)
            
            # Extract technical information
            self._extract_edge_technical_info(edge)
            
            edges.append(edge)
        
        self.logger.logger.info(f"Created {len(edges)} edges from {len(arrows)} arrows")
        self.logger.log_step_end("create_edges_from_arrows", success=True,
                                edge_count=len(edges), arrow_count=len(arrows))
        
        return edges
    
    def _find_closest_node(self, point: Point, nodes: List[Node]) -> Optional[Node]:
        """Find the node closest to a given point within snap distance"""
        closest_node = None
        min_distance = float('inf')
        
        for node in nodes:
            # Calculate distance from point to node center
            node_center = node.center
            distance = point.distance_to(node_center)
            
            # Also check distance to node boundary
            bbox = node.bounding_box
            boundary_distance = self._point_to_bbox_distance(point, bbox)
            
            # Use the smaller distance
            actual_distance = min(distance, boundary_distance)
            
            if (actual_distance < min_distance and 
                actual_distance <= self.config.edge_snap_distance):
                min_distance = actual_distance
                closest_node = node
        
        return closest_node
    
    def _point_to_bbox_distance(self, point: Point, bbox: BoundingBox) -> float:
        """Calculate distance from point to bounding box edge"""
        # If point is inside bbox, distance is 0
        if (bbox.x1 <= point.x <= bbox.x2 and bbox.y1 <= point.y <= bbox.y2):
            return 0
        
        # Calculate distance to closest edge
        dx = max(bbox.x1 - point.x, 0, point.x - bbox.x2)
        dy = max(bbox.y1 - point.y, 0, point.y - bbox.y2)
        
        return (dx ** 2 + dy ** 2) ** 0.5
    
    def _determine_edge_type(self, edge: Edge, source_node: Node, target_node: Node) -> EdgeType:
        """Determine edge type based on labels and connected node types"""
        # Check labels for explicit type indicators
        all_labels = " ".join(edge.labels).lower()
        
        type_keywords = {
            EdgeType.API_CALL: ['api', 'rest', 'http', 'https', 'request', 'response'],
            EdgeType.DATABASE_CONNECTION: ['sql', 'query', 'database', 'db'],
            EdgeType.DATA_FLOW: ['data', 'flow', 'stream', 'transfer'],
            EdgeType.REPLICATION: ['replication', 'sync', 'backup', 'replica'],
            EdgeType.LOAD_BALANCING: ['balance', 'distribute', 'lb']
        }
        
        for edge_type, keywords in type_keywords.items():
            if any(keyword in all_labels for keyword in keywords):
                return edge_type
        
        # Infer type from connected node types
        if (source_node.node_type == NodeType.APPLICATION and 
            target_node.node_type == NodeType.DATABASE):
            return EdgeType.DATABASE_CONNECTION
        
        if (target_node.node_type == NodeType.LOAD_BALANCER or 
            source_node.node_type == NodeType.LOAD_BALANCER):
            return EdgeType.LOAD_BALANCING
        
        if 'protocol:http' in all_labels or 'protocol:https' in all_labels:
            return EdgeType.API_CALL
        
        return EdgeType.NETWORK_CONNECTION
    
    def _extract_edge_technical_info(self, edge: Edge):
        """Extract technical information from edge labels"""
        for label in edge.labels:
            if label.startswith('protocol:'):
                edge.protocol = label.split(':', 1)[1]
            elif label.startswith('port:'):
                edge.port = label.split(':', 1)[1]
            elif label.startswith('direction:'):
                direction = label.split(':', 1)[1]
                if direction == 'outbound':
                    edge.direction = 'source_to_target'
                elif direction == 'inbound':
                    edge.direction = 'target_to_source'
    
    def _validate_graph(self, nodes: List[Node], edges: List[Edge]) -> Tuple[List[Node], List[Edge]]:
        """Validate and clean the graph"""
        self.logger.log_step_start("validate_graph")
        
        # Remove nodes with empty names or very low confidence
        valid_nodes = [
            node for node in nodes 
            if node.name.strip() and node.confidence > 0.1
        ]
        
        # Create mapping of valid node IDs
        valid_node_ids = {node.id for node in valid_nodes}
        
        # Remove edges that reference invalid nodes
        valid_edges = [
            edge for edge in edges
            if (edge.source_node_id in valid_node_ids and 
                edge.target_node_id in valid_node_ids)
        ]
        
        # Remove duplicate edges
        unique_edges = self._remove_duplicate_edges(valid_edges)
        
        removed_nodes = len(nodes) - len(valid_nodes)
        removed_edges = len(edges) - len(unique_edges)
        
        self.logger.logger.info(f"Graph validation: removed {removed_nodes} nodes, {removed_edges} edges")
        self.logger.log_step_end("validate_graph", success=True,
                                valid_nodes=len(valid_nodes), valid_edges=len(unique_edges))
        
        return valid_nodes, unique_edges
    
    def _remove_duplicate_edges(self, edges: List[Edge]) -> List[Edge]:
        """Remove duplicate edges between the same nodes"""
        seen_connections = set()
        unique_edges = []
        
        for edge in edges:
            # Create a key that represents the connection (bidirectional)
            connection_key = tuple(sorted([edge.source_node_id, edge.target_node_id]))
            
            if connection_key not in seen_connections:
                seen_connections.add(connection_key)
                unique_edges.append(edge)
        
        return unique_edges
    
    def _get_node_type_counts(self, nodes: List[Node]) -> Dict[str, int]:
        """Get count of each node type"""
        type_counts = {}
        for node in nodes:
            type_name = node.node_type.value
            type_counts[type_name] = type_counts.get(type_name, 0) + 1
        return type_counts
    
    def _get_edge_type_counts(self, edges: List[Edge]) -> Dict[str, int]:
        """Get count of each edge type"""
        type_counts = {}
        for edge in edges:
            type_name = edge.edge_type.value
            type_counts[type_name] = type_counts.get(type_name, 0) + 1
        return type_counts
    
    def get_graph_statistics(self, graph: ArchitectureGraph) -> Dict[str, Any]:
        """Get comprehensive statistics about the graph"""
        stats = {
            'total_nodes': len(graph.nodes),
            'total_edges': len(graph.edges),
            'node_types': self._get_node_type_counts(graph.nodes),
            'edge_types': self._get_edge_type_counts(graph.edges),
            'connected_components': self._count_connected_components(graph),
            'average_node_confidence': self._calculate_average_node_confidence(graph.nodes),
            'average_edge_confidence': self._calculate_average_edge_confidence(graph.edges)
        }
        
        # Calculate graph density
        if len(graph.nodes) > 1:
            max_possible_edges = len(graph.nodes) * (len(graph.nodes) - 1)
            stats['graph_density'] = len(graph.edges) / max_possible_edges if max_possible_edges > 0 else 0
        else:
            stats['graph_density'] = 0
        
        return stats
    
    def _count_connected_components(self, graph: ArchitectureGraph) -> int:
        """Count the number of connected components in the graph"""
        if not graph.nodes:
            return 0
        
        # Build adjacency list
        adjacency = {node.id: [] for node in graph.nodes}
        for edge in graph.edges:
            adjacency[edge.source_node_id].append(edge.target_node_id)
            adjacency[edge.target_node_id].append(edge.source_node_id)
        
        # DFS to find connected components
        visited = set()
        components = 0
        
        for node_id in adjacency:
            if node_id not in visited:
                components += 1
                stack = [node_id]
                while stack:
                    current = stack.pop()
                    if current not in visited:
                        visited.add(current)
                        for neighbor in adjacency[current]:
                            if neighbor not in visited:
                                stack.append(neighbor)
        
        return components
    
    def _calculate_average_node_confidence(self, nodes: List[Node]) -> float:
        """Calculate average confidence of nodes"""
        if not nodes:
            return 0.0
        return sum(node.confidence for node in nodes) / len(nodes)
    
    def _calculate_average_edge_confidence(self, edges: List[Edge]) -> float:
        """Calculate average confidence of edges"""
        if not edges:
            return 0.0
        return sum(edge.confidence for edge in edges) / len(edges)
    
    def find_orphaned_nodes(self, graph: ArchitectureGraph) -> List[Node]:
        """Find nodes that have no connections"""
        connected_nodes = set()
        for edge in graph.edges:
            connected_nodes.add(edge.source_node_id)
            connected_nodes.add(edge.target_node_id)
        
        return [node for node in graph.nodes if node.id not in connected_nodes]
    
    def find_high_degree_nodes(self, graph: ArchitectureGraph, min_degree: int = 3) -> List[Node]:
        """Find nodes with high degree (many connections)"""
        node_degrees = {}
        for edge in graph.edges:
            node_degrees[edge.source_node_id] = node_degrees.get(edge.source_node_id, 0) + 1
            node_degrees[edge.target_node_id] = node_degrees.get(edge.target_node_id, 0) + 1
        
        high_degree_nodes = []
        for node in graph.nodes:
            degree = node_degrees.get(node.id, 0)
            if degree >= min_degree:
                high_degree_nodes.append(node)
        
        return high_degree_nodes
    
    def export_graph_summary(self, graph: ArchitectureGraph) -> Dict[str, Any]:
        """Export a summary of the graph for reporting"""
        stats = self.get_graph_statistics(graph)
        orphaned_nodes = self.find_orphaned_nodes(graph)
        high_degree_nodes = self.find_high_degree_nodes(graph)
        
        summary = {
            'graph_info': {
                'name': graph.name,
                'id': graph.id,
                'created_at': graph.created_at.isoformat(),
                'processing_info': graph.processing_info
            },
            'statistics': stats,
            'quality_issues': {
                'orphaned_nodes': len(orphaned_nodes),
                'orphaned_node_names': [node.name for node in orphaned_nodes],
                'high_degree_nodes': len(high_degree_nodes),
                'high_degree_node_names': [node.name for node in high_degree_nodes]
            },
            'recommendations': self._generate_recommendations(graph, stats, orphaned_nodes)
        }
        
        return summary
    
    def _generate_recommendations(self, graph: ArchitectureGraph, stats: Dict, orphaned_nodes: List[Node]) -> List[str]:
        """Generate recommendations for improving the graph"""
        recommendations = []
        
        if stats['graph_density'] < 0.1:
            recommendations.append("Graph has low connectivity - consider adding more connections")
        
        if len(orphaned_nodes) > 0:
            recommendations.append(f"Found {len(orphaned_nodes)} orphaned nodes - consider connecting them or removing if not needed")
        
        if stats['connected_components'] > 1:
            recommendations.append(f"Graph has {stats['connected_components']} disconnected components - consider connecting them")
        
        if stats['average_node_confidence'] < 0.7:
            recommendations.append("Low average node confidence - consider improving text detection quality")
        
        if stats['average_edge_confidence'] < 0.7:
            recommendations.append("Low average edge confidence - consider improving arrow detection quality")
        
        return recommendations

class GraphValidator:
    """Advanced graph validation and quality checking"""
    
    def __init__(self, logger: DiagnosticsLogger):
        self.logger = logger
    
    @performance_monitor("validate_graph_quality")
    def validate_graph_quality(self, graph: ArchitectureGraph) -> Dict[str, any]:
        """
        Perform comprehensive graph quality validation
        
        Returns:
            Dictionary with validation results and quality metrics
        """
        self.logger.log_step_start("graph_quality_validation")
        
        validation_results = {
            'is_valid': True,
            'warnings': [],
            'errors': [],
            'quality_metrics': {}
        }
        
        # Check for common issues
        self._check_isolated_nodes(graph, validation_results)
        self._check_node_naming(graph, validation_results)
        self._check_edge_consistency(graph, validation_results)
        self._check_technical_completeness(graph, validation_results)
        self._calculate_quality_metrics(graph, validation_results)
        
        # Overall validation status
        validation_results['is_valid'] = len(validation_results['errors']) == 0
        
        # Log results
        error_count = len(validation_results['errors'])
        warning_count = len(validation_results['warnings'])
        
        if error_count > 0:
            self.logger.logger.error(f"Graph validation failed with {error_count} errors")
        elif warning_count > 0:
            self.logger.logger.warning(f"Graph validation passed with {warning_count} warnings")
        else:
            self.logger.logger.info("Graph validation passed with no issues")
        
        self.logger.save_intermediate_data(validation_results, "04_graph_validation")
        self.logger.log_step_end("graph_quality_validation", success=validation_results['is_valid'])
        
        return validation_results
    
    def _check_isolated_nodes(self, graph: ArchitectureGraph, results: Dict):
        """Check for nodes with no connections"""
        connected_nodes = set()
        for edge in graph.edges:
            connected_nodes.add(edge.source_node_id)
            connected_nodes.add(edge.target_node_id)
        
        isolated_nodes = [node for node in graph.nodes if node.id not in connected_nodes]
        
        if isolated_nodes:
            results['warnings'].append(
                f"Found {len(isolated_nodes)} isolated nodes with no connections"
            )
            results['quality_metrics']['isolated_nodes'] = len(isolated_nodes)
        else:
            results['quality_metrics']['isolated_nodes'] = 0
    
    def _check_node_naming(self, graph: ArchitectureGraph, results: Dict):
        """Check node naming quality"""
        unnamed_nodes = [node for node in graph.nodes if not node.name.strip()]
        very_short_names = [node for node in graph.nodes if len(node.name.strip()) < 3]
        
        if unnamed_nodes:
            results['errors'].append(f"Found {len(unnamed_nodes)} nodes without names")
        
        if very_short_names:
            results['warnings'].append(f"Found {len(very_short_names)} nodes with very short names")
        
        results['quality_metrics']['unnamed_nodes'] = len(unnamed_nodes)
        results['quality_metrics']['short_named_nodes'] = len(very_short_names)
    
    def _check_edge_consistency(self, graph: ArchitectureGraph, results: Dict):
        """Check edge consistency and completeness"""
        # Check for edges referencing non-existent nodes
        node_ids = {node.id for node in graph.nodes}
        invalid_edges = []
        
        for edge in graph.edges:
            if edge.source_node_id not in node_ids:
                invalid_edges.append(f"Edge {edge.id} references non-existent source node")
            if edge.target_node_id not in node_ids:
                invalid_edges.append(f"Edge {edge.id} references non-existent target node")
        
        if invalid_edges:
            results['errors'].extend(invalid_edges)
        
        # Check for self-referencing edges
        self_edges = [edge for edge in graph.edges if edge.source_node_id == edge.target_node_id]
        if self_edges:
            results['warnings'].append(f"Found {len(self_edges)} self-referencing edges")
        
        results['quality_metrics']['invalid_edges'] = len(invalid_edges)
        results['quality_metrics']['self_edges'] = len(self_edges)
    
    def _check_technical_completeness(self, graph: ArchitectureGraph, results: Dict):
        """Check technical information completeness"""
        nodes_with_ip = sum(1 for node in graph.nodes if node.ip_address)
        nodes_with_tech = sum(1 for node in graph.nodes if node.technology)
        edges_with_protocol = sum(1 for edge in graph.edges if edge.protocol)
        
        results['quality_metrics']['nodes_with_ip_addresses'] = nodes_with_ip
        results['quality_metrics']['nodes_with_technology'] = nodes_with_tech
        results['quality_metrics']['edges_with_protocols'] = edges_with_protocol
        
        # Calculate completeness ratios
        total_nodes = len(graph.nodes)
        total_edges = len(graph.edges)
        
        if total_nodes > 0:
            ip_completeness = nodes_with_ip / total_nodes
            tech_completeness = nodes_with_tech / total_nodes
            
            results['quality_metrics']['ip_address_completeness'] = ip_completeness
            results['quality_metrics']['technology_completeness'] = tech_completeness
            
            if ip_completeness < 0.3:
                results['warnings'].append("Low IP address coverage in nodes")
        
        if total_edges > 0:
            protocol_completeness = edges_with_protocol / total_edges
            results['quality_metrics']['protocol_completeness'] = protocol_completeness
            
            if protocol_completeness < 0.3:
                results['warnings'].append("Low protocol information coverage in edges")
    
    def _calculate_quality_metrics(self, graph: ArchitectureGraph, results: Dict):
        """Calculate overall quality metrics"""
        total_nodes = len(graph.nodes)
        total_edges = len(graph.edges)
        
        # Basic metrics
        results['quality_metrics']['total_nodes'] = total_nodes
        results['quality_metrics']['total_edges'] = total_edges
        
        # Density (actual edges / possible edges)
        if total_nodes > 1:
            max_possible_edges = total_nodes * (total_nodes - 1)  # Directed graph
            density = total_edges / max_possible_edges if max_possible_edges > 0 else 0
            results['quality_metrics']['graph_density'] = density
        else:
            results['quality_metrics']['graph_density'] = 0
        
        # Average confidence scores
        if total_nodes > 0:
            avg_node_confidence = sum(node.confidence for node in graph.nodes) / total_nodes
            results['quality_metrics']['average_node_confidence'] = avg_node_confidence
        
        if total_edges > 0:
            avg_edge_confidence = sum(edge.confidence for edge in graph.edges) / total_edges
            results['quality_metrics']['average_edge_confidence'] = avg_edge_confidence
        
        # Node type diversity
        node_types = set(node.node_type for node in graph.nodes)
        results['quality_metrics']['node_type_diversity'] = len(node_types)
        
        # Calculate overall quality score (0-100)
        quality_score = self._calculate_overall_quality_score(results['quality_metrics'])
        results['quality_metrics']['overall_quality_score'] = quality_score
    
    def _calculate_overall_quality_score(self, metrics: Dict) -> float:
        """Calculate overall quality score based on various metrics"""
        score = 100.0
        
        # Penalize for issues
        score -= metrics.get('isolated_nodes', 0) * 5
        score -= metrics.get('unnamed_nodes', 0) * 10
        score -= metrics.get('invalid_edges', 0) * 15
        score -= metrics.get('short_named_nodes', 0) * 2
        
        # Boost for good characteristics
        score += min(metrics.get('node_type_diversity', 0) * 5, 25)  # Max 25 points
        
        # Confidence bonuses
        node_conf = metrics.get('average_node_confidence', 0)
        edge_conf = metrics.get('average_edge_confidence', 0)
        score += (node_conf + edge_conf) * 10  # Max 20 points
        
        # Technical completeness bonuses
        ip_complete = metrics.get('ip_address_completeness', 0)
        tech_complete = metrics.get('technology_completeness', 0)
        protocol_complete = metrics.get('protocol_completeness', 0)
        score += (ip_complete + tech_complete + protocol_complete) * 5  # Max 15 points
        
        return max(0, min(100, score))