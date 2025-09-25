# Architecture Diagram Parser

Enterprise-grade tool for parsing architecture diagrams into structured data. Converts visual diagrams into machine-readable JSON format with comprehensive node and edge analysis.

## Features

- 🔍 **Multi-Engine OCR**: Supports EasyOCR, Tesseract, RapidOCR, and PaddleOCR with intelligent fallback
- 🏹 **Advanced Arrow Detection**: Detects arrows, lines, and connections using computer vision
- 📊 **Graph Construction**: Builds structured graphs with node classification and edge analysis
- 🔧 **Enterprise-Grade**: Production-ready with comprehensive logging and error handling
- 📈 **Performance Monitoring**: Built-in performance tracking and optimization
- 🎯 **Quality Validation**: Automatic graph quality assessment and recommendations

## Quick Start

### 1. Setup Environment

```bash
# Run the setup script
python setup.py

# Activate virtual environment
# On macOS/Linux:
source venv/bin/activate
# On Windows:
venv\Scripts\activate
```

### 2. Basic Usage

```bash
# Parse a diagram
python main.py architecture_diagram.png -o output.json

# With debugging
python main.py architecture_diagram.png --debug --save-intermediate -o output.json
```

### 3. Configuration

Use the generated `config.json` to customize processing parameters:

```json
{
  "ocr": {
    "primary_engine": "easyocr",
    "confidence_threshold": 0.6
  },
  "arrow_detection": {
    "hough_threshold": 50,
    "min_line_length": 30
  },
  "diagnostics": {
    "log_level": "INFO",
    "save_intermediate_results": true
  }
}
```

## Architecture

The parser follows a multi-stage pipeline:

```
Input Image → Preprocessing → Tiling → OCR → Arrow Detection → Graph Building → Validation → Output
```

### Core Components

- **Image Preprocessing**: Upscaling, contrast enhancement, deskewing
- **Tiling System**: Processes large images in overlapping tiles
- **Multi-Engine OCR**: Text extraction with fallback strategies
- **Arrow Detection**: Computer vision-based connection detection
- **Graph Builder**: Converts detected elements into structured graphs
- **Quality Validator**: Ensures output quality and provides recommendations

## API Usage

```python
from arch_parser_config import ArchitectureDiagramConfig
from main import ArchitectureDiagramParser

# Initialize parser
config = ArchitectureDiagramConfig()
parser = ArchitectureDiagramParser(config)

# Parse diagram
result = parser.parse_diagram("diagram.png", "output.json")

if result.success:
    graph = result.graph
    print(f"Parsed {len(graph.nodes)} nodes and {len(graph.edges)} edges")
    
    # Get graph summary
    summary = parser.get_graph_summary(graph)
    print(f"Graph density: {summary['statistics']['graph_density']}")
```

## Output Format

The parser generates structured JSON output:

```json
{
  "id": "graph-uuid",
  "name": "diagram_name",
  "nodes": [
    {
      "id": "node-uuid",
      "name": "Web Server",
      "type": "server",
      "bounding_box": {"x1": 100, "y1": 200, "x2": 300, "y2": 400},
      "confidence": 0.95,
      "ip_address": "192.168.1.10",
      "technology": "nginx"
    }
  ],
  "edges": [
    {
      "id": "edge-uuid",
      "source_node_id": "node-uuid-1",
      "target_node_id": "node-uuid-2",
      "type": "api_call",
      "protocol": "http",
      "confidence": 0.88
    }
  ],
  "processing_info": {
    "total_nodes": 5,
    "total_edges": 4,
    "node_types": {"server": 2, "database": 1, "load_balancer": 1},
    "edge_types": {"api_call": 3, "database_connection": 1}
  }
}
```

## Node Types

The parser automatically classifies nodes into categories:

- **Server**: Web servers, application servers
- **Database**: MySQL, PostgreSQL, MongoDB, etc.
- **Load Balancer**: F5, HAProxy, Nginx
- **Firewall**: Security appliances
- **Router/Switch**: Network equipment
- **Cloud**: AWS, Azure, GCP services
- **Application**: Custom applications
- **User**: End users, clients

## Edge Types

Connections are classified as:

- **API Call**: HTTP/HTTPS requests
- **Database Connection**: SQL queries
- **Data Flow**: Data transfers
- **Replication**: Database replication
- **Load Balancing**: Traffic distribution
- **Network Connection**: General network links

## Configuration Options

### OCR Configuration
```json
{
  "ocr": {
    "primary_engine": "easyocr",
    "fallback_engines": ["tesseract", "rapidocr"],
    "confidence_threshold": 0.6,
    "min_text_height": 8,
    "max_text_height": 200
  }
}
```

### Arrow Detection
```json
{
  "arrow_detection": {
    "hough_threshold": 50,
    "min_line_length": 30,
    "max_line_gap": 10,
    "connection_distance_threshold": 20
  }
}
```

### Graph Processing
```json
{
  "graph_processing": {
    "node_merge_distance": 50,
    "edge_snap_distance": 30,
    "min_node_area": 100
  }
}
```

## System Requirements

### Python Dependencies
- Python 3.8+
- OpenCV 4.8+
- NumPy 1.24+
- SciPy 1.10+
- Scikit-learn 1.3+

### OCR Engines (Optional)
- **EasyOCR**: `pip install easyocr`
- **Tesseract**: System installation required
- **RapidOCR**: `pip install rapidocr-onnxruntime`
- **PaddleOCR**: `pip install paddleocr`

### System Dependencies

**macOS:**
```bash
brew install tesseract
```

**Ubuntu/Debian:**
```bash
sudo apt-get install tesseract-ocr libtesseract-dev
```

**Windows:**
Download Tesseract from: https://github.com/UB-Mannheim/tesseract/wiki

## Performance Optimization

### For Large Images
- Increase `tile_size` in config
- Adjust `overlap` for better tile coverage
- Use GPU-accelerated OCR engines when available

### For Better Accuracy
- Enable `enhance_contrast` in preprocessing
- Adjust `confidence_threshold` for OCR
- Fine-tune `hough_threshold` for arrow detection

## Troubleshooting

### Common Issues

**OCR Not Working:**
- Check system dependencies are installed
- Verify image quality and resolution
- Try different OCR engines

**Poor Arrow Detection:**
- Adjust `hough_threshold` (lower = more sensitive)
- Check `min_line_length` and `max_line_gap`
- Ensure arrows are clearly visible

**Low Graph Quality:**
- Enable `save_intermediate_results` for debugging
- Check `node_merge_distance` and `edge_snap_distance`
- Review validation recommendations

### Debug Mode

```bash
python main.py diagram.png --debug --save-intermediate -o output.json
```

This saves intermediate processing images and detailed logs for analysis.

## Development

### Running Tests
```bash
# Install development dependencies
pip install pytest pytest-cov

# Run tests
pytest tests/ -v --cov=.

# Run with coverage report
pytest tests/ --cov=. --cov-report=html
```

### Code Quality
```bash
# Format code
black *.py

# Lint code
flake8 *.py

# Type checking
mypy *.py
```

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests
5. Submit a pull request

## Support

For issues and questions:
- Check the troubleshooting section
- Review the debug output
- Open an issue with sample images and configuration
