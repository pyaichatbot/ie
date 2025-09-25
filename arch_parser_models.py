# models.py
"""
Data models for Architecture Diagram Parser
Defines the structure for all data objects used in the parsing pipeline
"""
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple, Any
from enum import Enum
import json
import uuid
from datetime import datetime

@dataclass
class BoundingBox:
    """Represents a bounding box with coordinates"""
    x1: int
    y1: int
    x2: int
    y2: int
    
    @property
    def width(self) -> int:
        return self.x2 - self.x1
    
    @property
    def height(self) -> int:
        return self.y2 - self.y1
    
    @property
    def area(self) -> int:
        return self.width * self.height
    
    @property
    def center(self) -> Tuple[int, int]:
        return ((self.x1 + self.x2) // 2, (self.y1 + self.y2) // 2)
    
    def distance_to(self, other: 'BoundingBox') -> float:
        """Calculate distance between bounding box centers"""
        x1, y1 = self.center
        x2, y2 = other.center
        return ((x2 - x1) ** 2 + (y2 - y1) ** 2) ** 0.5
    
    def overlaps(self, other: 'BoundingBox') -> bool:
        """Check if this bounding box overlaps with another"""
        return not (self.x2 < other.x1 or other.x2 < self.x1 or 
                   self.y2 < other.y1 or other.y2 < self.y1)
    
    def to_dict(self) -> Dict:
        return {'x1': self.x1, 'y1': self.y1, 'x2': self.x2, 'y2': self.y2}

@dataclass
class Point:
    """Represents a 2D point"""
    x: int
    y: int
    
    def distance_to(self, other: 'Point') -> float:
        return ((other.x - self.x) ** 2 + (other.y - self.y) ** 2) ** 0.5
    
    def to_tuple(self) -> Tuple[int, int]:
        return (self.x, self.y)
    
    def to_dict(self) -> Dict:
        return {'x': self.x, 'y': self.y}

class NodeType(Enum):
    """Types of nodes in architecture diagrams"""
    SERVER = "server"
    DATABASE = "database"
    SERVICE = "service"
    LOAD_BALANCER = "load_balancer"
    FIREWALL = "firewall"
    ROUTER = "router"
    SWITCH = "switch"
    CLOUD = "cloud"
    USER = "user"
    APPLICATION = "application"
    CONTAINER = "container"
    VM = "virtual_machine"
    STORAGE = "storage"
    NETWORK = "network"
    INTERNET = "internet"
    GENERIC = "generic"
    UNKNOWN = "unknown"

class EdgeType(Enum):
    """Types of edges/connections in architecture diagrams"""
    NETWORK_CONNECTION = "network_connection"
    DATA_FLOW = "data_flow"
    API_CALL = "api_call"
    DATABASE_CONNECTION = "database_connection"
    DEPENDENCY = "dependency"
    REPLICATION = "replication"
    LOAD_BALANCING = "load_balancing"
    UNKNOWN = "unknown"

@dataclass
class TextBlock:
    """Represents extracted text with location and metadata"""
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    text: str = ""
    bounding_box: BoundingBox = field(default_factory=lambda: BoundingBox(0, 0, 0, 0))
    confidence: float = 0.0
    ocr_engine: str = ""
    font_size: Optional[int] = None
    
    def to_dict(self) -> Dict:
        return {
            'id': self.id,
            'text': self.text,
            'bounding_box': self.bounding_box.to_dict(),
            'confidence': self.confidence,
            'ocr_engine': self.ocr_engine,
            'font_size': self.font_size
        }

@dataclass
class Arrow:
    """Represents a detected arrow with start and end points"""
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    start_point: Point = field(default_factory=lambda: Point(0, 0))
    end_point: Point = field(default_factory=lambda: Point(0, 0))
    shaft_points: List[Point] = field(default_factory=list)
    confidence: float = 0.0
    text_labels: List[str] = field(default_factory=list)  # Text found along the arrow
    
    @property
    def length(self) -> float:
        return self.start_point.distance_to(self.end_point)
    
    def to_dict(self) -> Dict:
        return {
            'id': self.id,
            'start_point': self.start_point.to_dict(),
            'end_point': self.end_point.to_dict(),
            'shaft_points': [p.to_dict() for p in self.shaft_points],
            'confidence': self.confidence,
            'text_labels': self.text_labels
        }

@dataclass
class Node:
    """Represents a node in the architecture diagram"""
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    name: str = ""
    node_type: NodeType = NodeType.UNKNOWN
    bounding_box: BoundingBox = field(default_factory=lambda: BoundingBox(0, 0, 0, 0))
    text_blocks: List[TextBlock] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    confidence: float = 0.0
    
    # Technical attributes commonly found in architecture diagrams
    ip_address: Optional[str] = None
    hostname: Optional[str] = None
    port: Optional[str] = None
    version: Optional[str] = None
    technology: Optional[str] = None
    
    @property
    def center(self) -> Point:
        center_coords = self.bounding_box.center
        return Point(center_coords[0], center_coords[1])
    
    def add_text_block(self, text_block: TextBlock):
        """Add a text block and update node properties"""
        self.text_blocks.append(text_block)
        
        # Update node name if empty
        if not self.name and text_block.text.strip():
            self.name = text_block.text.strip()
        
        # Extract technical information
        self._extract_technical_info(text_block.text)
    
    def _extract_technical_info(self, text: str):
        """Extract IP addresses, ports, and other technical info from text"""
        import re
        
        # IP address pattern
        ip_pattern = r'\b(?:[0-9]{1,3}\.){3}[0-9]{1,3}\b'
        ip_matches = re.findall(ip_pattern, text)
        if ip_matches and not self.ip_address:
            self.ip_address = ip_matches[0]
        
        # Port pattern (e.g., :8080, :443)
        port_pattern = r':(\d{2,5})\b'
        port_matches = re.findall(port_pattern, text)
        if port_matches and not self.port:
            self.port = port_matches[0]
        
        # Technology keywords
        tech_keywords = {
            'database': ['db', 'database', 'mysql', 'postgresql', 'oracle', 'mongodb'],
            'web_server': ['apache', 'nginx', 'iis', 'tomcat'],
            'load_balancer': ['lb', 'load balancer', 'f5', 'haproxy'],
            'firewall': ['fw', 'firewall', 'asa', 'palo alto']
        }
        
        text_lower = text.lower()
        for tech_type, keywords in tech_keywords.items():
            if any(keyword in text_lower for keyword in keywords):
                self.technology = tech_type
                break
    
    def to_dict(self) -> Dict:
        return {
            'id': self.id,
            'name': self.name,
            'type': self.node_type.value,
            'bounding_box': self.bounding_box.to_dict(),
            'text_blocks': [tb.to_dict() for tb in self.text_blocks],
            'metadata': self.metadata,
            'confidence': self.confidence,
            'ip_address': self.ip_address,
            'hostname': self.hostname,
            'port': self.port,
            'version': self.version,
            'technology': self.technology
        }

@dataclass
class Edge:
    """Represents a connection between nodes"""
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    source_node_id: str = ""
    target_node_id: str = ""
    edge_type: EdgeType = EdgeType.UNKNOWN
    arrow: Optional[Arrow] = None
    labels: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    confidence: float = 0.0
    
    # Connection details
    protocol: Optional[str] = None
    port: Optional[str] = None
    direction: str = "bidirectional"  # "source_to_target", "target_to_source", "bidirectional"
    
    def to_dict(self) -> Dict:
        return {
            'id': self.id,
            'source_node_id': self.source_node_id,
            'target_node_id': self.target_node_id,
            'type': self.edge_type.value,
            'arrow': self.arrow.to_dict() if self.arrow else None,
            'labels': self.labels,
            'metadata': self.metadata,
            'confidence': self.confidence,
            'protocol': self.protocol,
            'port': self.port,
            'direction': self.direction
        }

@dataclass
class ArchitectureGraph:
    """Represents the complete architecture diagram as a graph"""
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    name: str = ""
    nodes: List[Node] = field(default_factory=list)
    edges: List[Edge] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    processing_info: Dict[str, Any] = field(default_factory=dict)
    created_at: datetime = field(default_factory=datetime.now)
    
    def add_node(self, node: Node) -> str:
        """Add a node and return its ID"""
        self.nodes.append(node)
        return node.id
    
    def add_edge(self, edge: Edge) -> str:
        """Add an edge and return its ID"""
        self.edges.append(edge)
        return edge.id
    
    def get_node_by_id(self, node_id: str) -> Optional[Node]:
        """Get a node by its ID"""
        return next((node for node in self.nodes if node.id == node_id), None)
    
    def get_edges_for_node(self, node_id: str) -> List[Edge]:
        """Get all edges connected to a node"""
        return [edge for edge in self.edges 
                if edge.source_node_id == node_id or edge.target_node_id == node_id]
    
    def to_dict(self) -> Dict:
        return {
            'id': self.id,
            'name': self.name,
            'nodes': [node.to_dict() for node in self.nodes],
            'edges': [edge.to_dict() for edge in self.edges],
            'metadata': self.metadata,
            'processing_info': self.processing_info,
            'created_at': self.created_at.isoformat()
        }
    
    def to_json(self, indent: int = 2) -> str:
        """Export graph to JSON string"""
        return json.dumps(self.to_dict(), indent=indent, default=str)
    
    def save_to_file(self, filepath: str):
        """Save graph to JSON file"""
        with open(filepath, 'w') as f:
            json.dump(self.to_dict(), f, indent=2, default=str)
    
    @classmethod
    def from_dict(cls, data: Dict) -> 'ArchitectureGraph':
        """Create graph from dictionary"""
        graph = cls(
            id=data.get('id', str(uuid.uuid4())),
            name=data.get('name', ''),
            metadata=data.get('metadata', {}),
            processing_info=data.get('processing_info', {})
        )
        
        # Load nodes
        for node_data in data.get('nodes', []):
            node = Node(
                id=node_data['id'],
                name=node_data['name'],
                node_type=NodeType(node_data['type']),
                bounding_box=BoundingBox(**node_data['bounding_box']),
                metadata=node_data.get('metadata', {}),
                confidence=node_data.get('confidence', 0.0),
                ip_address=node_data.get('ip_address'),
                hostname=node_data.get('hostname'),
                port=node_data.get('port'),
                version=node_data.get('version'),
                technology=node_data.get('technology')
            )
            
            # Load text blocks
            for tb_data in node_data.get('text_blocks', []):
                text_block = TextBlock(
                    id=tb_data['id'],
                    text=tb_data['text'],
                    bounding_box=BoundingBox(**tb_data['bounding_box']),
                    confidence=tb_data.get('confidence', 0.0),
                    ocr_engine=tb_data.get('ocr_engine', ''),
                    font_size=tb_data.get('font_size')
                )
                node.text_blocks.append(text_block)
            
            graph.nodes.append(node)
        
        # Load edges
        for edge_data in data.get('edges', []):
            edge = Edge(
                id=edge_data['id'],
                source_node_id=edge_data['source_node_id'],
                target_node_id=edge_data['target_node_id'],
                edge_type=EdgeType(edge_data['type']),
                labels=edge_data.get('labels', []),
                metadata=edge_data.get('metadata', {}),
                confidence=edge_data.get('confidence', 0.0),
                protocol=edge_data.get('protocol'),
                port=edge_data.get('port'),
                direction=edge_data.get('direction', 'bidirectional')
            )
            
            # Load arrow if present
            if edge_data.get('arrow'):
                arrow_data = edge_data['arrow']
                arrow = Arrow(
                    id=arrow_data['id'],
                    start_point=Point(**arrow_data['start_point']),
                    end_point=Point(**arrow_data['end_point']),
                    shaft_points=[Point(**p) for p in arrow_data.get('shaft_points', [])],
                    confidence=arrow_data.get('confidence', 0.0),
                    text_labels=arrow_data.get('text_labels', [])
                )
                edge.arrow = arrow
            
            graph.edges.append(edge)
        
        return graph

@dataclass
class ProcessingResult:
    """Result of the entire processing pipeline"""
    success: bool
    graph: Optional[ArchitectureGraph] = None
    error_message: Optional[str] = None
    processing_time: float = 0.0
    step_times: Dict[str, float] = field(default_factory=dict)
    intermediate_results: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict:
        return {
            'success': self.success,
            'graph': self.graph.to_dict() if self.graph else None,
            'error_message': self.error_message,
            'processing_time': self.processing_time,
            'step_times': self.step_times,
            'intermediate_results': {k: str(v) for k, v in self.intermediate_results.items()}
        }