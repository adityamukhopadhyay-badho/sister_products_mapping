# Sister Products Mapping System

An advanced AI-powered system for automatically identifying and clustering sister products using vector embeddings and density-based clustering algorithms.

## 🎯 Overview

Sister products are different variants of the same core product (e.g., "Lays Cream & Onion 52g" and "Lays Salted 28g" are both variants of "Lays Potato Chips"). This system automatically identifies such relationships by:

1. **Phase 1: Normalization & Embedding** - Extracts core product identity by removing variant-specific information (flavors, sizes, weights) and creates vector embeddings
2. **Phase 2: Clustering** - Uses HDBSCAN density-based clustering to automatically group sister products
3. **Phase 3: Visualization & Output** - Generates comprehensive reports, visualizations, and interactive dashboards

## 🚀 Features

### Core Capabilities
- ✅ **Intelligent Product Normalization** - Removes flavors, sizes, weights, and packaging variants
- ✅ **Phonetic Similarity** - Groups similar-sounding products with different spellings (burfi/burfee/barfee)
- ✅ **Vector Embeddings** - Uses state-of-the-art sentence transformers for semantic understanding
- ✅ **Automatic Clustering** - HDBSCAN identifies sister products without predefined cluster counts
- ✅ **Multi-Format Output** - JSON, CSV, and interactive HTML reports
- ✅ **Rich Visualizations** - Network graphs, distribution charts, and interactive dashboards
- ✅ **Progress Tracking** - Real-time progress bars and beautiful terminal output
- ✅ **Multi-Brand Support** - Process multiple brands simultaneously with comparison analytics

### Technical Features
- 🔧 **Configurable Parameters** - Adjust clustering sensitivity and model selection
- 📊 **Comprehensive Logging** - Detailed logs with Rich formatting
- 🎨 **Interactive Visualizations** - Network graphs, cluster analysis, and comparison dashboards
- 💾 **Live Saves** - Results saved progressively during processing
- ⚡ **Optimized Performance** - Batch processing and efficient vector operations

## 📋 Requirements

### System Requirements
- Python 3.8+
- 4GB+ RAM (8GB+ recommended for large datasets)
- Modern CPU with multi-core support

### Dependencies
All dependencies are automatically installed via `requirements.txt`:
- pandas, numpy - Data processing
- scikit-learn - Machine learning utilities
- sentence-transformers - Vector embeddings
- hdbscan - Density-based clustering
- plotly, matplotlib, seaborn - Visualizations
- rich - Beautiful terminal output
- networkx - Graph analysis
- umap-learn - Dimensionality reduction

## 🛠️ Installation

1. **Clone or Download the Project**
   ```bash
   cd sister_products_mapping
   ```

2. **Create Virtual Environment**
   ```bash
   python3 -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install Dependencies**
   ```bash
   pip3 install -r requirements.txt
   ```

## 📁 Project Structure

```
sister_products_mapping/
├── main.py                     # Main execution script
├── requirements.txt            # Python dependencies
├── README.md                   # This file
├── src/
│   ├── sister_products_mapper.py  # Core mapping algorithm
│   └── visualizer.py              # Visualization generator
├── data/                       # Input CSV files
│   ├── cornitos_products.csv
│   └── haldiram_products.csv
├── output/                     # Generated results
│   ├── brand_sister_products.json
│   ├── brand_detailed_results.csv
│   └── brand_cluster_summary.csv
├── visualizations/             # Generated visualizations
│   ├── brand_network_graph.html
│   ├── brand_dashboard.html
│   ├── brand_cluster_analysis.png
│   └── multi_brand_comparison.html
├── logs/                       # System logs
│   └── sister_products.log
└── venv/                       # Virtual environment
```

## 📊 Input Data Format

Your CSV files should contain the following columns:

| Column | Description | Example |
|--------|-------------|---------|
| `brandSKUId` | Unique product identifier | `abc123-def456` |
| `label` | Full product name | `"Lays Cream & Onion Chips 52g"` |
| `facets_jsonb` | Product attributes in JSON format | `{"flavour": "Cream & Onion", "netWeight": "52g", ...}` |
| `categoryLabel` | Product category | `"Snacks & Packaged Foods"` |

### Example CSV Row:
```csv
brandSKUId,label,facets_jsonb,categoryLabel
abc123,"Lays Cream & Onion Chips 52g","{""flavour"": ""Cream & Onion"", ""netWeight"": ""52g"", ""texture"": ""Crispy""}","Snacks & Packaged Foods"
```

## 🎮 Usage

### Basic Usage
```bash
# Process single brand
python3 main.py data/cornitos_products.csv

# Process multiple brands
python3 main.py data/cornitos_products.csv data/haldiram_products.csv

# Process all CSV files in data directory
python3 main.py data/*.csv
```

### Advanced Options
```bash
# Use different embedding model
python3 main.py --model all-mpnet-base-v2 data/*.csv

# Adjust clustering sensitivity
python3 main.py --min-cluster-size 3 --min-samples 2 data/*.csv

# Create more inclusive clusters (larger sister product groups)
python3 main.py --cluster-epsilon 0.2 data/*.csv

# Enable phonetic similarity for similar-sounding products (burfi/burfee/barfee)
python3 main.py --enable-phonetic data/*.csv

# Combine phonetic encoding with other optimizations
python3 main.py --enable-phonetic --cluster-epsilon 0.2 --phonetic-algorithm nysiis data/*.csv

# Skip visualizations for faster processing
python3 main.py --no-visualizations data/*.csv

# Custom output directories
python3 main.py --output-dir results --visualizations-dir charts data/*.csv
```

### 🔊 Phonetic Similarity Feature

The phonetic similarity feature helps cluster products with similar-sounding names but different spellings - especially useful for Indian food products with multiple transliterations.

**When to Use:**
- Products with spelling variations (burfi/burfee/barfee, laddu/ladoo/laddoo)
- Transliterated names from regional languages
- International products with inconsistent romanization

**Available Algorithms:**
- **metaphone** (default) - Best balance of accuracy and performance
- **soundex** - Simple but effective for basic phonetic matching
- **nysiis** - Good for names and proper nouns
- **match_rating_codex** - Alternative algorithm with different matching characteristics

**Performance Impact:**
- ✅ **+2-5% clustering rate improvement** for datasets with spelling variations
- ✅ **Minimal performance overhead** - adds ~0.5 seconds to processing time
- ✅ **Backward compatible** - disabled by default, preserves existing behavior

**Examples:**
```bash
# Basic phonetic clustering
python3 main.py --enable-phonetic data/haldiram_products.csv

# Try different algorithms
python3 main.py --enable-phonetic --phonetic-algorithm soundex data/*.csv
python3 main.py --enable-phonetic --phonetic-algorithm nysiis data/*.csv
```

### Command Line Options
```
Options:
  --model MODEL               Sentence transformer model (default: all-MiniLM-L6-v2)
  --min-cluster-size SIZE     Minimum cluster size for HDBSCAN (default: 2)
  --min-samples SAMPLES       Minimum samples for HDBSCAN (default: 1)
  --cluster-epsilon EPSILON   Distance threshold for cluster merging. Higher values
                              (0.1-0.3) create larger, more inclusive clusters (default: 0.0)
  --enable-phonetic           Enable phonetic similarity encoding for similar-sounding words
  --phonetic-algorithm ALGO   Phonetic algorithm: soundex, metaphone, nysiis, match_rating_codex
                              (default: metaphone)
  --output-dir DIR           Output directory (default: output)
  --visualizations-dir DIR   Visualizations directory (default: visualizations)
  --logs-dir DIR             Logs directory (default: logs)
  --no-visualizations        Skip generating visualizations
  -h, --help                 Show help message
```

## 📈 Output Files

### 1. Main Results (JSON)
```json
{
  "brand": "Cornitos",
  "total_products": 236,
  "total_clusters": 45,
  "products_with_sisters": 180,
  "products_without_sisters": 56,
  "sisterProductClusters": {
    "cluster_0": [
      {
        "brandSKUId": "abc123",
        "label": "Cornitos Nacho Chips Sweet Chili 22g",
        "normalized_name": "cornitos nacho chips",
        "categoryLabel": "Food"
      }
    ]
  }
}
```

### 2. Detailed CSV Results
Contains all products with cluster assignments, normalized names, and core identities.

### 3. Cluster Summary CSV
Simplified view focusing on cluster assignments and key product information.

### 4. Interactive Visualizations
- **Network Graph**: Interactive visualization showing product relationships
- **Dashboard**: Comprehensive analytics with charts and metrics
- **Cluster Analysis**: Distribution charts and statistics
- **Multi-Brand Comparison**: Cross-brand performance metrics

## 🧠 Algorithm Details

### Phase 1: Normalization & Embedding

**Step 1: Heuristic Name Normalization**
- Extracts variant values from product facets (flavor, weight, size, etc.)
- Removes these variants from product names using pattern matching
- Applies noise reduction (marketing terms, packaging descriptors)

**Example:**
```
Input: "Lays American Style Cream & Onion Flavour Potato Chips 52g"
Facets: {"flavour": "American Style Cream & Onion", "netWeight": "52g"}
Output: "Lays Potato Chips"
```

**Step 2: Core Identity Construction**
- Combines normalized name with category and brand
- Format: `[NORMALIZED NAME] | [BRAND] | [CATEGORY]`
- Example: `"lays potato chips | lays | snacks & packaged foods"`

**Step 3: Vector Embeddings**
- Uses sentence-transformers to convert core identities to high-dimensional vectors
- Default model: `all-MiniLM-L6-v2` (384 dimensions)
- Captures semantic similarity between products

### Phase 2: HDBSCAN Clustering

**Why HDBSCAN?**
- Automatically determines optimal number of clusters
- Handles noise (products with no sisters) by assigning cluster ID -1
- Density-based approach works well with embedding spaces
- No need to specify cluster count beforehand

**Key Parameters:**
- `min_cluster_size`: Minimum products needed to form a cluster
- `min_samples`: Minimum samples in a neighborhood for core points

### Phase 3: Output Generation

**Cluster Assignment**
- Each product gets a cluster ID (or -1 for noise)
- Sister products share the same cluster ID
- Results formatted for easy consumption

## 🎨 Visualizations

### 1. Network Graph
Interactive graph where:
- Nodes = Products
- Edges = Sister relationships
- Colors = Clusters
- Hover = Product details

### 2. Interactive Dashboard
Comprehensive analytics including:
- Cluster size distribution
- Category analysis
- Top clusters
- Summary statistics

### 3. Static Charts
- Cluster size histograms
- Category breakdowns
- Performance metrics
- Comparison charts

## ⚙️ Configuration

### Model Selection
Choose different sentence transformer models:
- `all-MiniLM-L6-v2` (default) - Fast, good quality
- `all-mpnet-base-v2` - Higher quality, slower
- `all-distilroberta-v1` - Balanced performance

### Clustering Parameters
- **min_cluster_size**: Higher values = fewer, larger clusters
- **min_samples**: Higher values = more conservative clustering

### Performance Tuning
- Use `--no-visualizations` for faster processing
- Adjust batch sizes for memory constraints
- Consider GPU acceleration for large datasets

## 🐛 Troubleshooting

### Common Issues

**1. Memory Errors**
```bash
# Reduce batch size or process brands separately
python3 main.py --no-visualizations data/small_brand.csv
```

**2. Import Errors**
```bash
# Ensure virtual environment is activated
source venv/bin/activate
pip3 install -r requirements.txt
```

**3. No Clusters Found**
```bash
# Lower minimum cluster size
python3 main.py --min-cluster-size 2 data/*.csv
```

**4. Performance Issues**
```bash
# Skip visualizations for large datasets
python3 main.py --no-visualizations data/*.csv
```

### Getting Help
- Check logs in `logs/sister_products.log`
- Enable verbose logging by modifying log level in source
- Review input data format requirements

## 🔬 Example Results

### Sample Clustering Results for Cornitos:

**Cluster 5 (Nacho Chips):**
- Cornitos Nacho Chips Sweet Chili 22g
- Cornitos Nacho Chips Cheese & Herbs 22g
- Cornitos Nacho Chips Tomato Mexicana 22g
- Cornitos Nacho Chips Tikka Masala 22g

**Cluster 12 (Crusties):**
- Cornitos Crusties Rajma Snacks 18g
- Cornitos Crusties Potato Snacks 18g
- Cornitos Crusties Chana Snacks 18g

**Performance Metrics:**
- Total Products: 236
- Clusters Found: 45
- Products with Sisters: 180 (76.3%)
- Processing Time: ~45 seconds

## 🚀 Future Enhancements

- **Cross-Brand Clustering**: Identify sister products across different brands
- **Hierarchical Clustering**: Multi-level product relationships
- **Active Learning**: User feedback to improve clustering
- **API Interface**: REST API for integration with other systems
- **Real-time Processing**: Stream processing for live product catalogs

## 📄 License

This project is provided as-is for the sister products mapping use case. Feel free to modify and extend according to your needs.

## 🤝 Contributing

Suggestions and improvements are welcome! Key areas for contribution:
- Additional normalization patterns
- New visualization types
- Performance optimizations
- Documentation improvements

---

**Happy Clustering! 🎯** 