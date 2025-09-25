#!/usr/bin/env python3
"""
Example usage of Architecture Diagram Parser
Demonstrates how to use the parser programmatically
"""
import os
import sys
from pathlib import Path

# Add current directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from arch_parser_config import ArchitectureDiagramConfig
from main import ArchitectureDiagramParser

def create_sample_diagram():
    """Create a simple sample diagram for testing"""
    import cv2
    import numpy as np
    
    # Create a simple architecture diagram
    img = np.ones((600, 800, 3), dtype=np.uint8) * 255  # White background
    
    # Draw some rectangles (servers)
    cv2.rectangle(img, (100, 100), (200, 200), (0, 0, 255), 2)  # Red rectangle
    cv2.rectangle(img, (300, 100), (400, 200), (0, 0, 255), 2)  # Red rectangle
    cv2.rectangle(img, (500, 100), (600, 200), (0, 0, 255), 2)  # Red rectangle
    
    # Draw database
    cv2.rectangle(img, (300, 400), (400, 500), (0, 255, 0), 2)  # Green rectangle
    
    # Draw arrows
    # Arrow 1: Server 1 to Server 2
    cv2.arrowedLine(img, (200, 150), (300, 150), (0, 0, 0), 2, tipLength=0.3)
    
    # Arrow 2: Server 2 to Server 3
    cv2.arrowedLine(img, (400, 150), (500, 150), (0, 0, 0), 2, tipLength=0.3)
    
    # Arrow 3: Server 2 to Database
    cv2.arrowedLine(img, (350, 200), (350, 400), (0, 0, 0), 2, tipLength=0.3)
    
    # Add text labels
    cv2.putText(img, "Web Server 1", (110, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)
    cv2.putText(img, "Web Server 2", (310, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)
    cv2.putText(img, "Web Server 3", (510, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)
    cv2.putText(img, "Database", (320, 520), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)
    
    # Save the image
    cv2.imwrite("sample_architecture.png", img)
    print("📄 Created sample architecture diagram: sample_architecture.png")
    return "sample_architecture.png"

def example_basic_usage():
    """Basic usage example"""
    print("🔧 Example 1: Basic Usage")
    print("-" * 40)
    
    # Create sample diagram
    image_path = create_sample_diagram()
    
    # Initialize parser with default config
    parser = ArchitectureDiagramParser()
    
    # Parse the diagram
    print(f"🔄 Parsing diagram: {image_path}")
    result = parser.parse_diagram(image_path, "output_basic.json")
    
    if result.success:
        print("✅ Parsing successful!")
        print(f"📊 Graph contains {len(result.graph.nodes)} nodes and {len(result.graph.edges)} edges")
        print(f"⏱️  Processing time: {result.processing_time:.2f}s")
    else:
        print(f"❌ Parsing failed: {result.error_message}")

def example_custom_config():
    """Example with custom configuration"""
    print("\n🔧 Example 2: Custom Configuration")
    print("-" * 40)
    
    # Create custom configuration
    config = ArchitectureDiagramConfig()
    
    # Customize settings
    from arch_parser_config import LogLevel
    config.diagnostics.log_level = LogLevel.DEBUG
    config.diagnostics.save_intermediate_results = True
    config.ocr.confidence_threshold = 0.5
    config.arrow_detection.hough_threshold = 30
    
    # Initialize parser with custom config
    parser = ArchitectureDiagramParser(config)
    
    # Parse with custom settings
    result = parser.parse_diagram("sample_architecture.png", "output_custom.json")
    
    if result.success:
        print("✅ Custom parsing successful!")
        
        # Get detailed graph summary
        summary = parser.get_graph_summary(result.graph)
        print(f"📈 Graph Statistics:")
        print(f"   - Total nodes: {summary['statistics']['total_nodes']}")
        print(f"   - Total edges: {summary['statistics']['total_edges']}")
        print(f"   - Graph density: {summary['statistics']['graph_density']:.2f}")
        print(f"   - Connected components: {summary['statistics']['connected_components']}")
        
        if summary['quality_issues']['orphaned_nodes'] > 0:
            print(f"⚠️  Orphaned nodes: {summary['quality_issues']['orphaned_nodes']}")
        
        if summary['recommendations']:
            print("💡 Recommendations:")
            for rec in summary['recommendations']:
                print(f"   - {rec}")
    else:
        print(f"❌ Custom parsing failed: {result.error_message}")

def example_graph_analysis():
    """Example of graph analysis"""
    print("\n🔧 Example 3: Graph Analysis")
    print("-" * 40)
    
    # Parse diagram
    parser = ArchitectureDiagramParser()
    result = parser.parse_diagram("sample_architecture.png")
    
    if not result.success:
        print(f"❌ Parsing failed: {result.error_message}")
        return
    
    graph = result.graph
    
    # Analyze the graph
    print("🔍 Graph Analysis:")
    
    # Node analysis
    print(f"📊 Nodes ({len(graph.nodes)}):")
    for node in graph.nodes:
        print(f"   - {node.name} ({node.node_type.value}) - Confidence: {node.confidence:.2f}")
        if node.ip_address:
            print(f"     IP: {node.ip_address}")
        if node.technology:
            print(f"     Technology: {node.technology}")
    
    # Edge analysis
    print(f"🔗 Edges ({len(graph.edges)}):")
    for edge in graph.edges:
        source = graph.get_node_by_id(edge.source_node_id)
        target = graph.get_node_by_id(edge.target_node_id)
        print(f"   - {source.name if source else 'Unknown'} → {target.name if target else 'Unknown'}")
        print(f"     Type: {edge.edge_type.value}, Protocol: {edge.protocol or 'Unknown'}")
        print(f"     Confidence: {edge.confidence:.2f}")
    
    # Graph statistics
    stats = parser.graph_builder.get_graph_statistics(graph)
    print(f"\n📈 Graph Statistics:")
    print(f"   - Node types: {stats['node_types']}")
    print(f"   - Edge types: {stats['edge_types']}")
    print(f"   - Average node confidence: {stats['average_node_confidence']:.2f}")
    print(f"   - Average edge confidence: {stats['average_edge_confidence']:.2f}")
    
    # Find orphaned nodes
    orphaned = parser.graph_builder.find_orphaned_nodes(graph)
    if orphaned:
        print(f"\n⚠️  Orphaned nodes: {[node.name for node in orphaned]}")
    
    # Find high-degree nodes
    high_degree = parser.graph_builder.find_high_degree_nodes(graph)
    if high_degree:
        print(f"🔗 High-degree nodes: {[node.name for node in high_degree]}")

def cleanup():
    """Clean up example files"""
    files_to_remove = [
        "sample_architecture.png",
        "output_basic.json",
        "output_custom.json"
    ]
    
    for file in files_to_remove:
        if Path(file).exists():
            os.remove(file)
            print(f"🗑️  Removed {file}")

def main():
    """Run all examples"""
    print("🚀 Architecture Diagram Parser - Examples")
    print("=" * 50)
    
    try:
        # Run examples
        example_basic_usage()
        example_custom_config()
        example_graph_analysis()
        
        print("\n✅ All examples completed successfully!")
        print("\n📖 Next steps:")
        print("1. Try with your own diagrams")
        print("2. Experiment with different configurations")
        print("3. Check the generated JSON outputs")
        
    except Exception as e:
        print(f"\n❌ Example failed: {e}")
        import traceback
        traceback.print_exc()
    
    finally:
        # Clean up
        print("\n🧹 Cleaning up example files...")
        cleanup()

if __name__ == "__main__":
    main()
