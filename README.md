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
- 🚀 **Bulk Processing** - Process 53K+ brands with 1.4M+ products in manageable batches
- 🔄 **Resume Capability** - Intelligent checkpointing and resumption of interrupted processing
- 👤 **Human-in-the-Loop** - Manual approval mechanism for batch processing control

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

### CSV File Format
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

### 🗄️ Database Integration

**NEW**: Direct PostgreSQL database integration for live data processing.

#### Database Configuration
```bash
# Environment variables (optional - defaults provided)
export DB_USER="postgres"
export DB_PASSWORD="your_password" 
export DB_HOST="db.badho.in"
export DB_PORT="5432"
export DB_NAME="badho-app"
```

#### Features
- ✅ **Live Data Processing**: Process products directly from PostgreSQL database
- ✅ **Brand-wise Processing**: Process all brands or specific brand by ID
- ✅ **Master CSV Output**: Consolidated results across all brands with conflict resolution
- ✅ **Live Saves**: Progressive saving after each brand completion
- ✅ **Automatic Brand Detection**: Fetches all verified brands automatically

#### Database Schema Requirements
The system expects the following PostgreSQL schema structure:
```sql
brands."brandSKU"        -- Product table
brands.brand             -- Brand table  
categories.category      -- Category table
brands."brandSKU_category" -- Product-category mapping
users.seller_brand       -- Brand-seller mapping
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

### Database Processing
```bash
# Process all brands from database
python3 main.py --from-database --cluster-epsilon 0.1 --no-visualizations

# Process specific brand from database
python3 main.py --from-database --brand-id "a8e1f9cb-38b3-4104-aaf8-e27e4573cc73" --cluster-epsilon 0.1

# Use facets-based embeddings from database
python3 main.py --from-database --use-facets --cluster-epsilon 0.1 --no-visualizations

# Database processing with phonetic similarity
python3 main.py --from-database --enable-phonetic --phonetic-algorithm metaphone --cluster-epsilon 0.1
```

## 📋 Complete Command-Line Reference

### Positional Arguments
| Argument | Description | Example |
|----------|-------------|---------|
| `files` | CSV files containing product data (optional when using --from-database) | `data/brand1.csv data/brand2.csv` |

### Data Source Options
| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `--from-database` | flag | `False` | Process products from PostgreSQL database instead of CSV files |
| `--brand-id` | string | `None` | Specific brand ID to process when using --from-database |
| `--batch-size` | integer | `1000` | Batch size for database processing |
| `--bulk-run` | flag | `False` | **🚀 BULK PROCESSING MODE**: Process 53K+ brands with 1.4M+ products in batches with human approval |
| `--bulk-batch-size` | integer | `10000` | Number of brands per batch in bulk processing mode |
| `--auto-approve` | flag | `False` | **🤖 AUTO-APPROVAL MODE**: Skip human confirmation and automatically process all batches |

### Identity Options
| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `--simple-identity` | flag | `False` | **🔧 OPTIONAL IDENTITY MODE**: Use simplified core identity (brand + category) for embeddings (not recommended for best results) |

### Model & Algorithm Options
| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `--model` | string | `all-MiniLM-L6-v2` | Sentence transformer model to use for embeddings |
| `--use-facets` | flag | `False` | Use facets_jsonb data directly for embeddings instead of normalized names |

### Clustering Parameters
| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `--min-cluster-size` | integer | `2` | Minimum cluster size for HDBSCAN |
| `--min-samples` | integer | `1` | Minimum samples for HDBSCAN |
| `--cluster-epsilon` | float | `0.0` | Distance threshold for cluster merging. Higher values (0.1-0.3) create larger, more inclusive clusters |
| `--fast-clustering` | flag | `False` | **⚡ PERFORMANCE MODE**: Use fast KMeans clustering for all datasets (trades accuracy for speed) |

### Phonetic Similarity Options
| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `--enable-phonetic` | flag | `False` | Enable phonetic similarity encoding for better clustering of similar-sounding words |
| `--phonetic-algorithm` | choice | `metaphone` | Phonetic algorithm: `soundex`, `metaphone`, `nysiis`, `match_rating_codex` |

### Output & Visualization Options
| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `--output-dir` | string | `output` | Output directory for results (JSON, CSV files) |
| `--visualizations-dir` | string | `visualizations` | Directory for visualization files (HTML, PNG) |
| `--logs-dir` | string | `logs` | Directory for log files |
| `--no-visualizations` | flag | `False` | Skip generating visualizations for faster processing |

### Common Usage Patterns

#### CSV File Processing
```bash
# Basic processing
python3 main.py data/brand1.csv data/brand2.csv

# High-accuracy clustering
python3 main.py --cluster-epsilon 0.1 --enable-phonetic data/*.csv

# Fast processing without visualizations
python3 main.py --no-visualizations --min-cluster-size 2 data/*.csv

# Using advanced embedding model
python3 main.py --model all-mpnet-base-v2 --cluster-epsilon 0.15 data/*.csv
```

#### Database Processing
```bash
# Process all brands from database
python3 main.py --from-database --cluster-epsilon 0.1

# Process specific brand with variant classification
python3 main.py --from-database --brand-id "brand-uuid-here" --cluster-epsilon 0.1 --no-visualizations

# Facets-based processing with phonetic similarity
python3 main.py --from-database --use-facets --enable-phonetic --phonetic-algorithm nysiis --cluster-epsilon 0.1

# High-performance batch processing
python3 main.py --from-database --batch-size 2000 --no-visualizations --cluster-epsilon 0.1
```

#### 🚀 Bulk Processing Mode (Enterprise Scale)
```bash
# Process all 53K+ brands with 1.4M+ products (default: 10K brands per batch)
python3 main.py --bulk-run --cluster-epsilon 0.1 --no-visualizations

# Bulk processing with custom batch size (e.g., 5K brands per batch for testing)
python3 main.py --bulk-run --bulk-batch-size 5000 --cluster-epsilon 0.1 --no-visualizations

# Small test batch (e.g., 100 brands for quick testing)
python3 main.py --bulk-run --bulk-batch-size 100 --cluster-epsilon 0.1 --no-visualizations

# Bulk processing with enhanced features
python3 main.py --bulk-run --enable-phonetic --cluster-epsilon 0.1 --min-cluster-size 2

# Resume interrupted bulk processing (automatically detects checkpoints)
python3 main.py --bulk-run --cluster-epsilon 0.1 --no-visualizations

# AUTO-APPROVAL MODE: Process all batches without human confirmation
python3 main.py --bulk-run --auto-approve --cluster-epsilon 0.1 --no-visualizations

# Full automation: Auto-approve + fast clustering + custom batch size
python3 main.py --bulk-run --auto-approve --fast-clustering --bulk-batch-size 5000 --cluster-epsilon 0.1 --no-visualizations
```

**🎯 Bulk Processing Features:**
- **Batched Processing**: Configurable brands per batch (default: 10,000)
- **Human Approval**: Manual approval required before each batch (can be auto-approved)
- **Auto-Approval Mode**: Skip confirmations and process all batches automatically
- **Checkpoint & Resume**: Automatic progress saving and resumption
- **Master Files**: Results saved as `master_sister_products_batch{1-6}.csv`
- **Progress Tracking**: Detailed statistics and progress monitoring
- **Error Handling**: Failed brands tracked and can be retried

#### ⚡ Performance Optimizations
```bash
# Fast clustering for large datasets (sacrifices some accuracy for speed)
python3 main.py --fast-clustering --cluster-epsilon 0.1 --no-visualizations data/*.csv

# Fast bulk processing for maximum speed
python3 main.py --from-database --bulk-run --fast-clustering --cluster-epsilon 0.1 --no-visualizations

# Automatic optimization (fast clustering kicks in for datasets >2000 products)
python3 main.py --from-database --cluster-epsilon 0.1 --no-visualizations
```

#### Advanced Configurations
```bash
# Maximum clustering accuracy
python3 main.py --model all-mpnet-base-v2 --enable-phonetic --phonetic-algorithm nysiis --cluster-epsilon 0.15 --min-cluster-size 2 data/*.csv

# Fast processing for large datasets
python3 main.py --no-visualizations --min-cluster-size 3 --cluster-epsilon 0.05 --batch-size 2000 data/*.csv

# Custom output organization
python3 main.py --output-dir /path/to/results --visualizations-dir /path/to/charts --logs-dir /path/to/logs data/*.csv
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

### 🏷️ Variant Classification Feature

**NEW**: The system now automatically classifies each cluster by analyzing the numbers in product labels to determine the type of variance:

#### Variant Types
- **🔵 TYPE_VARIANT**: Products with same numbers (size/weight) but different flavors/types
  - Example: "Pizza Spice Mix 38g" + "Pasta Spice Mix 38g" 
- **🟡 SIZE_VARIANT**: Products with different numbers (size/weight) but same type
  - Example: "Garam Masala 50g" + "Garam Masala 100g" + "Garam Masala 200g"
- **🟣 MIXED**: Clusters containing both size and type variations
  - Example: "Turmeric 100g" + "Turmeric 100g (box)" + "Turmeric 200g"
- **🔷 SINGLE_PRODUCT**: Clusters with only one product

#### Output Enhancement
- ✅ **Enhanced JSON**: Includes `variant_type_summary` and per-cluster `variant_type`
- ✅ **Enhanced CSV**: Adds `variant_type` column to cluster summary
- ✅ **Visual Display**: Color-coded variant types in terminal output

```bash
# Example output showing variant types
cluster_12 (12 products) - SIZE_VARIANT:
  1. EVEREST CORIANDER POWDER 200g. 200 Grams  
  2. EVEREST Coriander Powder 500g
  3. EVEREST Coriander Powder-100g
```

## 📈 Output Files

### 1. Main Results (JSON)
```json
{
  "brand": "EVEREST",
  "total_products": 251,
  "total_clusters": 59,
  "products_with_sisters": 235,
  "products_without_sisters": 16,
  "variant_type_summary": {
    "type_variant": 2,
    "size_variant": 33,
    "mixed": 24,
    "single_product": 0
  },
  "sisterProductClusters": {
    "cluster_0": {
      "variant_type": "size_variant",
      "products": [
        {
          "brandSKUId": "abc123",
          "label": "EVEREST GARAM MASALA 50g. 50 Grams",
          "normalized_name": "garam masala",
          "categoryLabel": "Spices",
          "categories_parsed": "Spices",
          "primary_category": "Spices"
        }
      ]
    }
  }
}
```

### 2. Detailed CSV Results
Contains all products with cluster assignments, normalized names, and core identities.

| Column | Description |
|--------|-------------|
| `brandSKUId` | Unique product identifier |
| `label` | Original product name |
| `normalized_name` | Cleaned product name |
| `core_identity` | Identity string used for clustering |
| `categoryLabel` | Product category |
| `cluster_id` | Assigned cluster ID (-1 for noise) |
| `brand_extracted` | Brand name from facets |
| `categories_parsed` | All parsed categories |
| `primary_category` | Primary category |

### 3. Cluster Summary CSV  
Simplified view focusing on cluster assignments and key product information **with variant classification**.

| Column | Description |
|--------|-------------|
| `cluster_id` | Cluster identifier |
| `variant_type` | **NEW**: Classification (size_variant, type_variant, mixed, no_sisters) |
| `brandSKUId` | Unique product identifier |
| `label` | Original product name |
| `normalized_name` | Cleaned product name |
| `categoryLabel` | Product category |

**Example CSV rows:**
```csv
cluster_id,variant_type,brandSKUId,label,normalized_name,categoryLabel
cluster_0,size_variant,abc123,"EVEREST GARAM MASALA 50g",garam masala,Spices
cluster_0,size_variant,def456,"EVEREST GARAM MASALA 100g",garam masala,Spices
cluster_1,type_variant,ghi789,"Pizza Spice Mix 38g",pizza spice mix,Seasonings
cluster_1,type_variant,jkl012,"Pasta Spice Mix 38g",pasta spice mix,Seasonings
```

### 4. Database-Specific Output Files

When using `--from-database`, additional consolidated files are generated:

| File | Description |
|------|-------------|
| `master_sister_products_results.csv` | **Consolidated CSV** with all brands and products |
| `master_processing_summary.json` | **Overall statistics** across all processed brands |
| `all_brands_database_results.json` | **Complete JSON** with all brand results |

**Master CSV includes:**
- All fields from individual brand CSVs
- `brand_name` column for brand identification  
- `processing_timestamp` for tracking when each brand was processed
- **Live conflict resolution** - reprocessing a brand updates its entries

### 5. Interactive Visualizations
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

## ⚡ Quick Reference

### Most Common Commands

```bash
# 🏆 RECOMMENDED: High-accuracy database processing with variant classification
python3 main.py --from-database --cluster-epsilon 0.1 --no-visualizations

# 🎯 Process specific brand with optimal settings
python3 main.py --from-database --brand-id "your-brand-id" --cluster-epsilon 0.1 --no-visualizations

# 🚀 Fast CSV processing without visualizations  
python3 main.py --no-visualizations --cluster-epsilon 0.1 data/*.csv

# 📊 Full processing with visualizations (slower)
python3 main.py --cluster-epsilon 0.1 --enable-phonetic data/*.csv

# 🔍 Maximum accuracy (slower but best results)
python3 main.py --model all-mpnet-base-v2 --enable-phonetic --phonetic-algorithm nysiis --cluster-epsilon 0.15 data/*.csv
```

### Key Parameter Guidelines

| Use Case | Recommended Settings |
|----------|---------------------|
| **Production/Fast** | `--cluster-epsilon 0.1 --no-visualizations` |
| **High Accuracy** | `--cluster-epsilon 0.15 --enable-phonetic` |
| **Large Datasets** | `--no-visualizations --batch-size 2000` |
| **Development** | `--cluster-epsilon 0.1` (with visualizations) |

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