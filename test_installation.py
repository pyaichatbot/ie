#!/usr/bin/env python3
"""
Test script to verify Architecture Diagram Parser installation
"""
import sys
import os
from pathlib import Path

def test_imports():
    """Test if all required modules can be imported"""
    print("🔍 Testing imports...")
    
    try:
        # Test core dependencies
        import cv2
        import numpy as np
        import sklearn
        import scipy
        print("✅ Core dependencies imported successfully")
    except ImportError as e:
        print(f"❌ Core dependency import failed: {e}")
        return False
    
    try:
        # Test project modules
        from arch_parser_config import ArchitectureDiagramConfig
        from diagnostics import DiagnosticsLogger
        from arch_parser_models import ArchitectureGraph, Node, Edge
        from arch_parser_ocr import MultiEngineOCR
        from arch_parser_arrow_detection import ArrowDetector
        from arch_parser_graph_builder import GraphBuilder
        print("✅ Project modules imported successfully")
    except ImportError as e:
        print(f"❌ Project module import failed: {e}")
        return False
    
    try:
        # Test optional OCR engines
        ocr_engines = []
        try:
            import easyocr
            ocr_engines.append("EasyOCR")
        except ImportError:
            pass
        
        try:
            import pytesseract
            ocr_engines.append("Tesseract")
        except ImportError:
            pass
        
        try:
            from rapidocr_onnxruntime import RapidOCR
            ocr_engines.append("RapidOCR")
        except ImportError:
            pass
        
        try:
            from paddleocr import PaddleOCR
            ocr_engines.append("PaddleOCR")
        except ImportError:
            pass
        
        if ocr_engines:
            print(f"✅ Available OCR engines: {', '.join(ocr_engines)}")
        else:
            print("⚠️  No OCR engines available - install at least one for text extraction")
        
    except Exception as e:
        print(f"⚠️  OCR engine check failed: {e}")
    
    return True

def test_configuration():
    """Test configuration loading"""
    print("\n🔧 Testing configuration...")
    
    try:
        from arch_parser_config import ArchitectureDiagramConfig, DEFAULT_CONFIG
        
        # Test default config
        config = DEFAULT_CONFIG
        print(f"✅ Default configuration loaded")
        print(f"   - OCR engine: {config.ocr.primary_engine.value}")
        print(f"   - Log level: {config.diagnostics.log_level.value}")
        print(f"   - Performance tracking: {config.diagnostics.performance_tracking}")
        
        # Test config file loading
        config_file = Path("config.json")
        if config_file.exists():
            file_config = ArchitectureDiagramConfig.from_file("config.json")
            print(f"✅ Configuration file loaded from config.json")
        else:
            print("ℹ️  No config.json found - using default configuration")
        
        return True
        
    except Exception as e:
        print(f"❌ Configuration test failed: {e}")
        return False

def test_logger():
    """Test logging system"""
    print("\n📝 Testing logging system...")
    
    try:
        from arch_parser_config import ArchitectureDiagramConfig
        from diagnostics import DiagnosticsLogger
        
        config = ArchitectureDiagramConfig()
        logger = DiagnosticsLogger(config.diagnostics)
        
        # Test logging
        logger.logger.info("Test info message")
        logger.logger.debug("Test debug message")
        logger.logger.warning("Test warning message")
        
        print("✅ Logging system working correctly")
        return True
        
    except Exception as e:
        print(f"❌ Logging test failed: {e}")
        return False

def test_models():
    """Test data models"""
    print("\n📊 Testing data models...")
    
    try:
        from arch_parser_models import (
            BoundingBox, Point, TextBlock, Arrow, Node, Edge, 
            ArchitectureGraph, NodeType, EdgeType
        )
        
        # Test basic model creation
        bbox = BoundingBox(10, 20, 100, 200)
        point = Point(50, 100)
        text_block = TextBlock(text="Test", bounding_box=bbox)
        
        node = Node(name="Test Node", bounding_box=bbox)
        node.node_type = NodeType.SERVER
        
        edge = Edge(source_node_id="node1", target_node_id="node2")
        edge.edge_type = EdgeType.API_CALL
        
        graph = ArchitectureGraph(name="Test Graph")
        graph.add_node(node)
        graph.add_edge(edge)
        
        print("✅ Data models working correctly")
        print(f"   - Graph has {len(graph.nodes)} nodes and {len(graph.edges)} edges")
        return True
        
    except Exception as e:
        print(f"❌ Models test failed: {e}")
        return False

def test_main_module():
    """Test main module"""
    print("\n🚀 Testing main module...")
    
    try:
        from main import ArchitectureDiagramParser
        from arch_parser_config import ArchitectureDiagramConfig
        
        config = ArchitectureDiagramConfig()
        parser = ArchitectureDiagramParser(config)
        
        print("✅ Main parser initialized successfully")
        return True
        
    except Exception as e:
        print(f"❌ Main module test failed: {e}")
        return False

def main():
    """Run all tests"""
    print("🧪 Architecture Diagram Parser - Installation Test")
    print("=" * 60)
    
    tests = [
        ("Imports", test_imports),
        ("Configuration", test_configuration),
        ("Logging", test_logger),
        ("Data Models", test_models),
        ("Main Module", test_main_module)
    ]
    
    passed = 0
    total = len(tests)
    
    for test_name, test_func in tests:
        try:
            if test_func():
                passed += 1
            else:
                print(f"❌ {test_name} test failed")
        except Exception as e:
            print(f"❌ {test_name} test crashed: {e}")
    
    print("\n" + "=" * 60)
    print(f"📊 Test Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed! Installation is working correctly.")
        print("\n📖 Next steps:")
        print("1. Try parsing a diagram: python main.py your_diagram.png")
        print("2. Check the README.md for detailed usage instructions")
        return 0
    else:
        print("⚠️  Some tests failed. Check the error messages above.")
        print("💡 Try running: pip install -r requirements.txt")
        return 1

if __name__ == "__main__":
    sys.exit(main())
