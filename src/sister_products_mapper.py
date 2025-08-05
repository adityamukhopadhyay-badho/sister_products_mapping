#!/usr/bin/env python3
"""
Sister Products Mapping System

This module implements a sophisticated system for mapping sister products using
vector embeddings and density-based clustering. The system follows a three-phase approach:

Phase 1: Normalization & Embedding - Clean product names and create vector embeddings
Phase 2: Clustering - Use HDBSCAN for automatic sister product grouping  
Phase 3: Output Generation - Create structured mappings with cluster assignments

Author: Sister Products Mapping System
Date: 2024
"""

import json
import logging
import re
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple, Any
import warnings

import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer
import hdbscan
import umap
import jellyfish
from rich.console import Console
from rich.progress import Progress, TaskID
from rich.logging import RichHandler
from rich.table import Table
from rich.panel import Panel
from rich import print as rprint

# Import database manager
from .database_manager import DatabaseManager

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')

class SisterProductsMapper:
    """
    Main class for mapping sister products using vector embeddings and clustering.
    
    This class implements a three-phase approach:
    1. Normalization & Embedding: Clean product names and create embeddings
    2. Clustering: Use HDBSCAN for automatic sister product grouping
    3. Output Generation: Create structured mappings
    """
    
    def __init__(self, 
                 model_name: str = 'all-MiniLM-L6-v2',
                 min_cluster_size: int = 2,
                 min_samples: int = 1,
                 cluster_selection_epsilon: float = 0.0,
                 enable_phonetic: bool = False,
                 phonetic_algorithm: str = 'soundex',
                 use_facets: bool = False,
                 output_dir: str = 'output',
                 logs_dir: str = 'logs'):
        """
        Initialize the Sister Products Mapper.
        
        Args:
            model_name: Sentence transformer model name
            min_cluster_size: Minimum cluster size for HDBSCAN
            min_samples: Minimum samples for HDBSCAN
            cluster_selection_epsilon: Distance threshold for cluster merging in HDBSCAN. 
                                     Higher values (e.g., 0.1-0.3) create larger, more inclusive clusters.
                                     Default 0.0 uses standard HDBSCAN behavior.
            enable_phonetic: Enable phonetic similarity encoding for better clustering of
                           similar-sounding words (e.g., 'burfi', 'burfee', 'barfee')
            phonetic_algorithm: Phonetic algorithm to use ('soundex', 'metaphone', 'nysiis', 'match_rating_codex')
            use_facets: Use facets_jsonb data directly for embeddings instead of normalized names
            output_dir: Directory for output files
            logs_dir: Directory for log files
        """
        self.model_name = model_name
        self.min_cluster_size = min_cluster_size
        self.min_samples = min_samples
        self.cluster_selection_epsilon = cluster_selection_epsilon
        self.enable_phonetic = enable_phonetic
        self.phonetic_algorithm = phonetic_algorithm
        self.use_facets = use_facets
        self.output_dir = Path(output_dir)
        self.logs_dir = Path(logs_dir)
        
        # Create directories
        self.output_dir.mkdir(exist_ok=True)
        self.logs_dir.mkdir(exist_ok=True)
        
        # Initialize console and logging
        self.console = Console()
        self._setup_logging()
        
        # Initialize components
        self.model = None
        self.clusterer = None
        
        # Initialize database manager for live data processing
        self.db_manager = DatabaseManager()
        
        # Master results storage for all brands
        self.master_results_df = pd.DataFrame()
        
        # Variant removal patterns - these will be removed from product names
        self.variant_patterns = {
            'weight': [
                r'\b\d+(\.\d+)?\s*(grams?|g|kg|kilograms?|ounces?|oz|lbs?|pounds?)\b',
                r'\b\d+(\.\d+)?\s*gm\b',
                r'\b\d+(\.\d+)?\s*GM\b'
            ],
            'volume': [
                r'\b\d+(\.\d+)?\s*(ml|milliliters?|liters?|l|fl\.?\s*oz|fluid\s*ounces?)\b',
                r'\b\d+(\.\d+)?\s*ML\b'
            ],
            'size': [
                r'\b(small|medium|large|xl|xxl|s|m|l)\b',
                r'\b\d+\s*(inch|inches|cm|centimeters?|mm|millimeters?)\b'
            ],
            'price': [
                r'rs\.?\s*\d+',
                r'₹\s*\d+',
                r'\$\s*\d+(\.\d+)?'
            ],
            'packet_size': [
                r'\b(single-use|multi-use|family\s*pack|party\s*pack)\b'
            ]
        }
        
        # Common variant facet keys to extract values from
        self.variant_facet_keys = [
            'flavour', 'flavor', 'scent', 'netWeight', 'netVolume', 
            'packetSize', 'productWeight', 'productVolume', 'color',
            'texture', 'spiceLevel', 'specialTag', 'variant',
            'packQuantity', 'servingSize', 'seasonalUse'
        ]
        
    def _get_phonetic_encoding(self, text: str) -> str:
        """
        Generate phonetic encoding of text using the specified algorithm.
        
        Args:
            text: Text to encode
            
        Returns:
            Phonetic encoding string
        """
        if not text or not self.enable_phonetic:
            return ""
            
        # Clean text - remove numbers, special chars, extra spaces
        clean_text = re.sub(r'[^\w\s]', ' ', text.lower())
        clean_text = re.sub(r'\d+', '', clean_text)
        clean_text = re.sub(r'\s+', ' ', clean_text).strip()
        
        if not clean_text:
            return ""
            
        words = clean_text.split()
        phonetic_codes = []
        
        for word in words:
            if len(word) < 3:  # Skip very short words
                continue
                
            try:
                if self.phonetic_algorithm == 'soundex':
                    code = jellyfish.soundex(word)
                elif self.phonetic_algorithm == 'metaphone':
                    code = jellyfish.metaphone(word)
                elif self.phonetic_algorithm == 'nysiis':
                    code = jellyfish.nysiis(word)
                elif self.phonetic_algorithm == 'match_rating_codex':
                    code = jellyfish.match_rating_codex(word)
                else:
                    # Default to metaphone
                    code = jellyfish.metaphone(word)
                    
                if code:
                    phonetic_codes.append(code)
                    
            except Exception:
                # If phonetic encoding fails, skip this word
                continue
                
        return ' '.join(phonetic_codes)
    
    def _get_phonetic_similarity_score(self, text1: str, text2: str) -> float:
        """
        Calculate phonetic similarity between two texts.
        
        Args:
            text1: First text
            text2: Second text
            
        Returns:
            Similarity score between 0 and 1
        """
        if not self.enable_phonetic or not text1 or not text2:
            return 0.0
            
        phonetic1 = self._get_phonetic_encoding(text1)
        phonetic2 = self._get_phonetic_encoding(text2)
        
        if not phonetic1 or not phonetic2:
            return 0.0
            
        # Simple word overlap similarity
        words1 = set(phonetic1.split())
        words2 = set(phonetic2.split())
        
        if not words1 or not words2:
            return 0.0
            
        intersection = words1.intersection(words2)
        union = words1.union(words2)
        
        return len(intersection) / len(union) if union else 0.0
        
    def _setup_logging(self):
        """Setup rich logging configuration."""
        logging.basicConfig(
            level=logging.INFO,
            format="%(message)s",
            datefmt="[%X]",
            handlers=[
                RichHandler(console=self.console, rich_tracebacks=True),
                logging.FileHandler(self.logs_dir / 'sister_products.log')
            ]
        )
        self.logger = logging.getLogger(__name__)
        
    def load_model(self):
        """Load the sentence transformer model."""
        if self.model is None:
            self.console.print(f"[blue]Loading sentence transformer model: {self.model_name}[/blue]")
            try:
                self.model = SentenceTransformer(self.model_name)
                self.logger.info(f"Successfully loaded model: {self.model_name}")
            except Exception as e:
                self.logger.error(f"Failed to load model {self.model_name}: {e}")
                raise
                
    def normalize_product_name(self, label: str, facets: Dict[str, Any], brand_name: str = "") -> str:
        """
        Phase 1: Normalize product name by removing variant-specific information.
        
        This is the most critical step - it extracts the core product identity
        by removing flavors, sizes, weights, and other variant characteristics.
        
        Args:
            label: Original product label/name
            facets: Product facets dictionary
            brand_name: Brand name to remove from normalized text
            
        Returns:
            Normalized product name representing the core product
        """
        normalized = label.strip()  # Keep original case initially
        
        # Step 1: Remove trailing weight/volume information by working backwards from the end
        # Split into words and work backwards until we hit a number or reach a reasonable stopping point
        words = normalized.split()
        
        # Find the last occurrence of a number in the string to identify where weight info starts
        last_number_index = -1
        for i in range(len(words) - 1, -1, -1):
            word = words[i]
            # Check if word contains a number
            if re.search(r'\d', word):
                last_number_index = i
                break
        
        # If we found a number, check if what follows looks like weight/volume info
        if last_number_index > 0:  # Don't remove if number is at the beginning
            # Look at words after the last number
            remaining_words = words[last_number_index + 1:]
            
            # Check if remaining words are weight/volume related
            weight_volume_keywords = [
                'grams?', 'gms?', 'g', 'kg', 'kilograms?', 'kilo',
                'ml', 'milliliters?', 'liters?', 'l', 'ltrs?',
                'oz', 'ounces?', 'lbs?', 'pounds?', 'gram', 'gm'
            ]
            
            # If any remaining word matches weight/volume pattern, remove from that number onwards
            should_truncate = False
            for word in remaining_words:
                word_clean = re.sub(r'[^\w]', '', word.lower())
                for keyword in weight_volume_keywords:
                    if re.match(rf'^{keyword}$', word_clean):
                        should_truncate = True
                        break
                if should_truncate:
                    break
            
            if should_truncate:
                # Keep everything up to (but not including) the last number
                normalized = ' '.join(words[:last_number_index])
        
        # Step 2: Handle the case where weight info appears as "100g" or "200grams" (number+unit together)
        # Remove patterns like "100g", "250ml", "2kg" from the end
        normalized = re.sub(r'\s+\d+\s*(g|gm|gms|gram|grams|kg|ml|l|ltr|ltrs|liter|liters|oz|lb|lbs|pound|pounds)\s*$', '', normalized, flags=re.IGNORECASE)
        
        # Step 3: Remove common packaging terms from the end
        normalized = re.sub(r'\s+\((box|pack|packet|pouch|bag|tin|can|jar|bottle)\)\s*$', '', normalized, flags=re.IGNORECASE)
        normalized = re.sub(r'\s+(box|pack|packet|pouch|bag|tin|can|jar|bottle)\s*$', '', normalized, flags=re.IGNORECASE)
        
        # Step 4: Now convert to lowercase for remaining processing
        normalized = normalized.lower().strip()
        
        # Step 5: Extract variant values from facets and remove them from the label
        # But be selective - only remove weight/volume/size info, not flavor/product identifiers
        variant_values_to_remove = set()
        
        # Only remove these specific types of variants that are truly just packaging/size variants
        size_weight_facets = ['netWeight', 'netVolume', 'packetSize', 'productWeight', 'productVolume', 'servingSize']
        
        for facet_key in size_weight_facets:
            if facet_key in facets and facets[facet_key]:
                value = str(facets[facet_key]).strip()
                if value and value.lower() not in ['n/a', 'null', 'none', '']:
                    # Clean and add to removal set
                    cleaned_value = re.sub(r'[^\w\s]', '', value.lower())
                    if len(cleaned_value) > 2:  # Only remove meaningful values
                        variant_values_to_remove.add(cleaned_value)
                        variant_values_to_remove.add(value.lower())
        
        # Remove variant values found in facets
        for variant_value in variant_values_to_remove:
            # Try exact match first
            if variant_value in normalized:
                normalized = normalized.replace(variant_value, ' ')
            
            # Try partial matches for compound values
            words = variant_value.split()
            if len(words) > 1:
                for word in words:
                    if len(word) > 3:  # Only remove meaningful words
                        normalized = re.sub(rf'\b{re.escape(word)}\b', ' ', normalized)
        
        # Step 6: Remove brand name (but more conservatively - only if it's clearly the brand name)
        if brand_name and brand_name.strip():
            brand_clean = brand_name.lower().strip()
            
            # Only remove brand name if it appears at the beginning of the string
            if normalized.startswith(brand_clean + ' '):
                normalized = normalized[len(brand_clean):].strip()
            elif normalized.startswith(brand_clean):
                normalized = normalized[len(brand_clean):].strip()
        
        # Step 7: Remove dashes and clean up special characters
        normalized = re.sub(r'\s*[-–—]\s*', ' ', normalized)  # Dashes with optional spaces
        normalized = re.sub(r'[^\w\s]', ' ', normalized)  # Other special characters
        
        # Step 8: Apply pattern-based removal for any remaining weights, volumes, etc.
        for category, patterns in self.variant_patterns.items():
            for pattern in patterns:
                normalized = re.sub(pattern, ' ', normalized, flags=re.IGNORECASE)
        
        # Step 9: Remove any remaining standalone numbers
        normalized = re.sub(r'\b\d+\b', ' ', normalized)
        
        # Step 10: Remove other noise patterns
        noise_patterns = [
            r'\b(rs\.?|₹)\s*\d+\b',  # Price mentions
            r'\b\d+%\s*(extra|off|free)\b',  # Promotional text
            r'\b(new|improved|special|premium|original)\b',  # Marketing terms (excluding "classic")
            r'\b(available|now|today|limited|offer)\b',  # Availability terms
        ]
        
        for pattern in noise_patterns:
            normalized = re.sub(pattern, ' ', normalized, flags=re.IGNORECASE)
        
        # Step 11: Final cleanup
        normalized = re.sub(r'\s+', ' ', normalized)  # Multiple spaces to single
        normalized = normalized.strip()
        
        # Step 12: If normalization resulted in too short a string, fall back to a cleaned version of original
        if len(normalized) < 3:
            normalized = re.sub(r'[^\w\s]', ' ', label.lower())
            normalized = re.sub(r'\s+', ' ', normalized).strip()
            # Remove brand name from fallback too
            if brand_name and brand_name.strip():
                brand_clean = brand_name.lower().strip()
                if normalized.startswith(brand_clean + ' '):
                    normalized = normalized[len(brand_clean):].strip()
        
        return normalized
    
    def parse_categories(self, category_input: str) -> List[str]:
        """
        Parse multiple categories from category input string.
        
        Args:
            category_input: Category string that may contain multiple categories
            
        Returns:
            List of individual categories
        """
        if not category_input or not isinstance(category_input, str):
            return ["general"]
        
        # Handle different separators: "/", ",", ";", "|"
        separators = ["/", ",", ";", "|"]
        categories = [category_input.strip()]
        
        for separator in separators:
            if separator in category_input:
                categories = [cat.strip() for cat in category_input.split(separator)]
                break
        
        # Clean and filter categories
        cleaned_categories = []
        for cat in categories:
            cat_clean = cat.strip()
            if cat_clean and cat_clean.lower() not in ['', 'n/a', 'null', 'none', 'n', 'a']:
                cleaned_categories.append(cat_clean)
        
        return cleaned_categories if cleaned_categories else ["general"]

    def create_facets_identity(self, facets_dict: Dict[str, Any], category: str = "") -> str:
        """
        Create identity string directly from facets_jsonb data for richer semantic embeddings.
        
        Args:
            facets_dict: Dictionary of product facets
            category: Product category
            
        Returns:
            Facets-based identity string for embedding
        """
        identity_parts = []
        
        # Add category first
        if category:
            categories = self.parse_categories(category)
            category_context = " ".join(categories)
            identity_parts.append(f"category:{category_context}")
        
        # Process ALL facets dynamically - no hardcoded exclusions
        for key, value in facets_dict.items():
            if value is None:
                continue
            
            # Process different value types
            if isinstance(value, list):
                # Join list values, filtering out empty/null items
                clean_values = []
                for v in value:
                    if v and str(v).strip():
                        clean_val = str(v).strip()
                        # Clean up property-style values (❌/✅ prefixes)
                        clean_val = re.sub(r'^[❌✅]\s*', '', clean_val)
                        if clean_val and clean_val.lower() not in ['n/a', 'null', 'none', '']:
                            clean_values.append(clean_val)
                
                if clean_values:
                    identity_parts.append(f"{key}:{' '.join(clean_values)}")
                    
            elif isinstance(value, str):
                clean_value = value.strip()
                # Filter out common null/empty values
                if (clean_value and 
                    clean_value.lower() not in ['n/a', 'null', 'none', '', '❌ ', '✅ ']):
                    # Clean up property-style values (❌/✅ prefixes)
                    clean_value = re.sub(r'^[❌✅]\s*', '', clean_value)
                    if clean_value:
                        identity_parts.append(f"{key}:{clean_value}")
                        
            elif isinstance(value, (int, float)):
                identity_parts.append(f"{key}:{str(value)}")
        
        # Join all parts
        facets_identity = " | ".join(identity_parts)
        
        # Add phonetic encoding if enabled and we have meaningful content
        if self.enable_phonetic and facets_identity:
            # Extract text content for phonetic encoding
            text_content = re.sub(r'\w+:', '', facets_identity)  # Remove facet keys
            text_content = re.sub(r'[|]', ' ', text_content)     # Remove separators
            phonetic_text = self._get_phonetic_encoding(text_content)
            if phonetic_text:
                facets_identity += f" | PHONETIC:{phonetic_text}"
        
        return facets_identity.lower().strip()

    def create_core_identity(self, normalized_name: str, category: str, brand: str = "") -> str:
        """
        Create the core identity string for embedding generation.
        
        Args:
            normalized_name: Normalized product name
            category: Product category (can be multiple categories separated by delimiters)
            brand: Brand name (optional)
            
        Returns:
            Core identity string formatted for embedding
        """
        # Parse multiple categories
        categories = self.parse_categories(category)
        
        # Create category string - use primary category for core identity but include all for context
        primary_category = categories[0]
        if len(categories) > 1:
            # Include all categories for richer semantic understanding
            category_context = " ".join(categories)
        else:
            category_context = primary_category
        
        # Create base core identity
        core_identity = f"{normalized_name} | {category_context}"
        
        # Add phonetic encoding if enabled
        if self.enable_phonetic:
            phonetic_name = self._get_phonetic_encoding(normalized_name)
            if phonetic_name:
                core_identity += f" | PHONETIC:{phonetic_name}"
            
        return core_identity.lower().strip()
    
    def process_brand_data(self, df: pd.DataFrame, brand_name: str) -> Tuple[pd.DataFrame, np.ndarray, List[str]]:
        """
        Process data for a single brand through normalization and embedding generation.
        
        Args:
            df: DataFrame with product data
            brand_name: Name of the brand being processed
            
        Returns:
            Tuple of (processed_df, embeddings_array, core_identities_list)
        """
        self.console.print(Panel(f"[bold green]Processing Brand:[/bold green] {brand_name}", 
                                style="green"))
        
        processed_df = df.copy()
        core_identities = []
        
        with Progress() as progress:
            task = progress.add_task(f"[cyan]Normalizing {brand_name} products...", 
                                   total=len(df))
            
            for idx, row in df.iterrows():
                try:
                    # Parse facets JSON
                    facets = json.loads(row['facets_jsonb']) if isinstance(row['facets_jsonb'], str) else row['facets_jsonb']
                    
                    # Extract brand name from facets for core identity (but use filename brand for normalization)
                    brand_from_facets = facets.get('brandName', brand_name)
                    
                    # Phase 1: Normalize the product name (use filename-based brand name for removal)
                    normalized_name = self.normalize_product_name(row['label'], facets, brand_name)
                    
                    # Parse categories for additional context
                    categories_list = self.parse_categories(row['categoryLabel'])
                    
                    # Create core identity string - use facets if flag is enabled
                    if self.use_facets:
                        core_identity = self.create_facets_identity(facets, row['categoryLabel'])
                    else:
                        core_identity = self.create_core_identity(
                            normalized_name, 
                            row['categoryLabel'], 
                            brand_from_facets
                        )
                    
                    core_identities.append(core_identity)
                    
                    # Store processed data
                    processed_df.at[idx, 'normalized_name'] = normalized_name
                    processed_df.at[idx, 'core_identity'] = core_identity
                    processed_df.at[idx, 'brand_extracted'] = brand_from_facets
                    processed_df.at[idx, 'categories_parsed'] = ', '.join(categories_list)
                    processed_df.at[idx, 'primary_category'] = categories_list[0]
                    
                except Exception as e:
                    self.logger.warning(f"Error processing row {idx}: {e}")
                    # Fallback processing - try basic normalization (use filename-based brand)
                    try:
                        fallback_normalized = self.normalize_product_name(row['label'], {}, brand_name)
                    except:
                        fallback_normalized = row['label']
                    
                    # Parse categories even in fallback
                    fallback_categories = self.parse_categories(row['categoryLabel'])
                    
                    if self.use_facets:
                        # In facets mode, try basic facets fallback or use normalized name
                        fallback_core = f"product:{fallback_normalized} | category:{row['categoryLabel']}"
                    else:
                        fallback_core = f"{fallback_normalized} | {row['categoryLabel']}"
                    core_identities.append(fallback_core)
                    processed_df.at[idx, 'normalized_name'] = fallback_normalized
                    processed_df.at[idx, 'core_identity'] = fallback_core
                    processed_df.at[idx, 'brand_extracted'] = brand_name
                    processed_df.at[idx, 'categories_parsed'] = ', '.join(fallback_categories)
                    processed_df.at[idx, 'primary_category'] = fallback_categories[0]
                
                progress.update(task, advance=1)
        
        # Generate embeddings
        self.console.print("[blue]Generating vector embeddings...[/blue]")
        embeddings = self.model.encode(core_identities, 
                                     show_progress_bar=True,
                                     batch_size=32)
        
        return processed_df, embeddings, core_identities
    
    def perform_clustering(self, embeddings: np.ndarray, brand_name: str) -> np.ndarray:
        """
        Phase 2: Perform HDBSCAN clustering on embeddings.
        
        Args:
            embeddings: Vector embeddings array
            brand_name: Brand name for logging
            
        Returns:
            Cluster labels array
        """
        self.console.print(f"[blue]Performing HDBSCAN clustering for {brand_name}...[/blue]")
        
        # Initialize HDBSCAN clusterer
        self.clusterer = hdbscan.HDBSCAN(
            min_cluster_size=self.min_cluster_size,
            min_samples=self.min_samples,
            cluster_selection_epsilon=self.cluster_selection_epsilon,
            metric='euclidean',
            cluster_selection_method='eom'
        )
        
        # Perform clustering
        cluster_labels = self.clusterer.fit_predict(embeddings)
        
        # Log clustering results
        n_clusters = len(set(cluster_labels)) - (1 if -1 in cluster_labels else 0)
        n_noise = list(cluster_labels).count(-1)
        
        self.logger.info(f"Clustering results for {brand_name}:")
        self.logger.info(f"  - Number of clusters: {n_clusters}")
        self.logger.info(f"  - Number of noise points: {n_noise}")
        self.logger.info(f"  - Products clustered: {len(cluster_labels) - n_noise}")
        
        return cluster_labels
    
    def classify_cluster_variant_type(self, cluster_products: pd.DataFrame) -> str:
        """
        Classify a cluster as either 'type_variant' or 'size_variant' based on numbers in labels.
        
        Logic:
        - If numbers in labels overlap (same numbers present), it's a TYPE VARIANT
        - If numbers in labels differ completely, it's a SIZE VARIANT
        
        Args:
            cluster_products: DataFrame containing products in the cluster
            
        Returns:
            'type_variant', 'size_variant', or 'mixed' if unclear
        """
        if len(cluster_products) < 2:
            return 'single_product'
        
        labels = cluster_products['label'].tolist()
        
        # Extract all numbers from each label
        label_numbers = []
        for label in labels:
            # Find all numbers in the label (including decimals)
            numbers = re.findall(r'\d+(?:\.\d+)?', label)
            # Convert to set of numbers for comparison
            number_set = set(numbers)
            label_numbers.append(number_set)
        
        # Check if any numbers are shared between different labels
        has_common_numbers = False
        has_different_numbers = False
        
        for i in range(len(label_numbers)):
            for j in range(i + 1, len(label_numbers)):
                set_i = label_numbers[i]
                set_j = label_numbers[j]
                
                # Check for intersection (common numbers)
                if set_i.intersection(set_j):
                    has_common_numbers = True
                
                # Check for differences
                if set_i != set_j:
                    has_different_numbers = True
        
        # Classification logic
        if has_common_numbers and has_different_numbers:
            # Mixed case - some numbers same, some different
            return 'mixed'
        elif has_common_numbers:
            # Same numbers across labels = TYPE VARIANT
            return 'type_variant'
        elif has_different_numbers:
            # Different numbers across labels = SIZE VARIANT  
            return 'size_variant'
        else:
            # No numbers or all identical = TYPE VARIANT
            return 'type_variant'
    
    def generate_output(self, processed_df: pd.DataFrame, cluster_labels: np.ndarray, 
                       brand_name: str) -> Dict[str, Any]:
        """
        Phase 3: Generate final output structure with cluster mappings.
        
        Args:
            processed_df: Processed DataFrame with product data
            cluster_labels: Cluster labels from HDBSCAN
            brand_name: Brand name
            
        Returns:
            Dictionary with sister product clusters
        """
        self.console.print(f"[blue]Generating output structure for {brand_name}...[/blue]")
        
        # Add cluster labels to DataFrame
        processed_df['cluster_id'] = cluster_labels
        
        # Create sister product clusters mapping with variant type classification
        sister_clusters = {}
        cluster_variant_types = {}  # Store variant types for each cluster
        
        for cluster_id in set(cluster_labels):
            if cluster_id == -1:  # Skip noise points
                continue
                
            cluster_products = processed_df[processed_df['cluster_id'] == cluster_id]
            
            # Classify the variant type for this cluster
            variant_type = self.classify_cluster_variant_type(cluster_products)
            cluster_variant_types[f"cluster_{cluster_id}"] = variant_type
            
            cluster_data = []
            for _, product in cluster_products.iterrows():
                cluster_data.append({
                    'brandSKUId': product['brandSKUId'],
                    'label': product['label'],
                    'normalized_name': product['normalized_name'],
                    'core_identity': product['core_identity'],
                    'categoryLabel': product['categoryLabel'],
                    'categories_parsed': product.get('categories_parsed', ''),
                    'primary_category': product.get('primary_category', 'general')
                })
            
            sister_clusters[f"cluster_{cluster_id}"] = {
                'variant_type': variant_type,
                'products': cluster_data
            }
        
        # Handle noise points (products with no sisters)
        noise_products = processed_df[processed_df['cluster_id'] == -1]
        if len(noise_products) > 0:
            noise_data = []
            for _, product in noise_products.iterrows():
                noise_data.append({
                    'brandSKUId': product['brandSKUId'],
                    'label': product['label'],
                    'normalized_name': product['normalized_name'],
                    'core_identity': product['core_identity'],
                    'categoryLabel': product['categoryLabel'],
                    'categories_parsed': product.get('categories_parsed', ''),
                    'primary_category': product.get('primary_category', 'general')
                })
            sister_clusters['no_sisters'] = {
                'variant_type': 'no_sisters',
                'products': noise_data
            }
        
        # Create variant type summary
        variant_type_summary = {
            'type_variant': 0,
            'size_variant': 0,
            'mixed': 0,
            'single_product': 0
        }
        
        for cluster_name, cluster_info in sister_clusters.items():
            if cluster_name != 'no_sisters':
                variant_type = cluster_info['variant_type']
                if variant_type in variant_type_summary:
                    variant_type_summary[variant_type] += 1
        
        # Create final output structure
        output = {
            'brand': brand_name,
            'total_products': len(processed_df),
            'total_clusters': len([k for k in sister_clusters.keys() if k != 'no_sisters']),
            'products_with_sisters': len(processed_df[processed_df['cluster_id'] != -1]),
            'products_without_sisters': len(noise_products),
            'variant_type_summary': variant_type_summary,
            'sisterProductClusters': sister_clusters,
            'processing_metadata': {
                'model_used': self.model_name,
                'min_cluster_size': self.min_cluster_size,
                'min_samples': self.min_samples,
                'timestamp': pd.Timestamp.now().isoformat()
            }
        }
        
        return output
    
    def save_results(self, results: Dict[str, Any], brand_name: str, 
                    processed_df: pd.DataFrame):
        """
        Save results to various output formats.
        
        Args:
            results: Results dictionary
            brand_name: Brand name
            processed_df: Processed DataFrame
        """
        brand_clean = re.sub(r'[^\w\-_]', '_', brand_name.lower())
        
        # Save main results as JSON
        json_file = self.output_dir / f"{brand_clean}_sister_products.json"
        with open(json_file, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        
        # Save detailed CSV
        csv_file = self.output_dir / f"{brand_clean}_detailed_results.csv"
        processed_df.to_csv(csv_file, index=False, encoding='utf-8')
        
        # Save cluster summary CSV with variant types
        summary_data = []
        for cluster_name, cluster_info in results['sisterProductClusters'].items():
            variant_type = cluster_info.get('variant_type', 'unknown')
            products = cluster_info.get('products', cluster_info)  # Handle old format if needed
            
            # Handle case where products might be directly under cluster_info (old format)
            if isinstance(products, list):
                product_list = products
            else:
                product_list = [products]  # Single product case
                
            for product in product_list:
                summary_data.append({
                    'cluster_id': cluster_name,
                    'variant_type': variant_type,
                    'brandSKUId': product['brandSKUId'],
                    'label': product['label'],
                    'normalized_name': product['normalized_name'],
                    'categoryLabel': product['categoryLabel'],
                    'categories_parsed': product.get('categories_parsed', ''),
                    'primary_category': product.get('primary_category', 'general')
                })
        
        summary_df = pd.DataFrame(summary_data)
        summary_file = self.output_dir / f"{brand_clean}_cluster_summary.csv"
        summary_df.to_csv(summary_file, index=False, encoding='utf-8')
        
        self.console.print(f"[green]Results saved:[/green]")
        self.console.print(f"  - JSON: {json_file}")
        self.console.print(f"  - Detailed CSV: {csv_file}")
        self.console.print(f"  - Summary CSV: {summary_file}")
    
    def display_results_summary(self, results: Dict[str, Any]):
        """Display a summary table of clustering results."""
        
        table = Table(title=f"Sister Products Mapping Results - {results['brand']}")
        table.add_column("Metric", style="cyan")
        table.add_column("Value", style="magenta")
        
        table.add_row("Total Products", str(results['total_products']))
        table.add_row("Total Clusters", str(results['total_clusters']))
        table.add_row("Products with Sisters", str(results['products_with_sisters']))
        table.add_row("Products without Sisters", str(results['products_without_sisters']))
        table.add_row("Model Used", results['processing_metadata']['model_used'])
        
        # Add variant type summary if available
        if 'variant_type_summary' in results:
            variant_summary = results['variant_type_summary']
            table.add_row("", "")  # Empty row for separation
            table.add_row("Type Variants", str(variant_summary.get('type_variant', 0)))
            table.add_row("Size Variants", str(variant_summary.get('size_variant', 0)))
            table.add_row("Mixed Variants", str(variant_summary.get('mixed', 0)))
            table.add_row("Single Product Clusters", str(variant_summary.get('single_product', 0)))
        
        self.console.print(table)
        
        # Show top clusters
        clusters = results['sisterProductClusters']
        if clusters:
            self.console.print("\n[bold]Top Sister Product Clusters:[/bold]")
            cluster_sizes = []
            for name, cluster_info in clusters.items():
                if name != 'no_sisters':
                    products = cluster_info.get('products', cluster_info)
                    if isinstance(products, list):
                        size = len(products)
                    else:
                        size = 1
                    variant_type = cluster_info.get('variant_type', 'unknown')
                    cluster_sizes.append((name, size, variant_type))
            
            cluster_sizes.sort(key=lambda x: x[1], reverse=True)
            
            for cluster_name, size, variant_type in cluster_sizes[:5]:  # Show top 5 clusters
                cluster_info = clusters[cluster_name]
                products = cluster_info.get('products', cluster_info)
                if not isinstance(products, list):
                    products = [products]
                
                variant_color = {
                    'type_variant': 'blue',
                    'size_variant': 'yellow', 
                    'mixed': 'magenta',
                    'single_product': 'cyan'
                }.get(variant_type, 'white')
                
                self.console.print(f"\n[green]{cluster_name}[/green] ({size} products) - [{variant_color}]{variant_type.upper()}[/{variant_color}]:")
                for i, product in enumerate(products[:3]):  # Show first 3 products
                    self.console.print(f"  {i+1}. {product['label']}")
                if len(products) > 3:
                    self.console.print(f"  ... and {len(products) - 3} more")
    
    def process_brand_file(self, file_path: str) -> Dict[str, Any]:
        """
        Process a single brand CSV file through the complete pipeline.
        
        Args:
            file_path: Path to the CSV file
            
        Returns:
            Results dictionary
        """
        file_path = Path(file_path)
        brand_name = file_path.stem.replace('_products', '').title()
        
        self.console.print(Panel(
            f"[bold blue]Starting Sister Products Mapping[/bold blue]\n"
            f"File: {file_path.name}\nBrand: {brand_name}",
            style="blue"
        ))
        
        # Load data
        try:
            df = pd.read_csv(file_path)
            self.logger.info(f"Loaded {len(df)} products for {brand_name}")
        except Exception as e:
            self.logger.error(f"Failed to load {file_path}: {e}")
            raise
        
        # Load model if not already loaded
        self.load_model()
        
        # Process through the three phases
        processed_df, embeddings, core_identities = self.process_brand_data(df, brand_name)
        cluster_labels = self.perform_clustering(embeddings, brand_name)
        results = self.generate_output(processed_df, cluster_labels, brand_name)
        
        # Save and display results
        self.save_results(results, brand_name, processed_df)
        self.display_results_summary(results)
        
        return results
    
    def process_multiple_brands(self, file_paths: List[str]) -> Dict[str, Dict[str, Any]]:
        """
        Process multiple brand files.
        
        Args:
            file_paths: List of CSV file paths
            
        Returns:
            Dictionary mapping brand names to their results
        """
        all_results = {}
        
        for file_path in file_paths:
            try:
                results = self.process_brand_file(file_path)
                brand_name = Path(file_path).stem.replace('_products', '').title()
                all_results[brand_name] = results
            except Exception as e:
                self.logger.error(f"Failed to process {file_path}: {e}")
                continue
        
        # Save combined results
        combined_file = self.output_dir / "all_brands_results.json"
        with open(combined_file, 'w', encoding='utf-8') as f:
            json.dump(all_results, f, indent=2, ensure_ascii=False)
        
        self.console.print(f"[green]Combined results saved to: {combined_file}[/green]")
        
        return all_results
    
    def process_from_database(self, brand_id: Optional[str] = None, batch_size: int = 1000) -> Dict[str, Any]:
        """
        Process sister products mapping using live data from PostgreSQL database.
        
        Args:
            brand_id: Optional specific brand ID to process. If None, processes all brands.
            batch_size: Number of products to process per brand batch
            
        Returns:
            Dictionary containing all brand results
        """
        self.console.print("[bold blue]🚀 Starting Database-Driven Sister Products Mapping[/bold blue]")
        
        try:
            # Load model if not already loaded
            self.load_model()
            
            if brand_id:
                # Process specific brand
                return self._process_single_brand_from_db(brand_id)
            else:
                # Process all brands
                return self._process_all_brands_from_db(batch_size)
                
        except Exception as e:
            self.logger.error(f"Failed to process from database: {e}")
            self.console.print(f"[red]✗ Processing failed: {e}[/red]")
            raise
    
    def _process_single_brand_from_db(self, brand_id: str) -> Dict[str, Any]:
        """Process a single brand from the database."""
        try:
            # Fetch brand data
            df = self.db_manager.fetch_brand_products(brand_id)
            
            if df.empty:
                self.console.print(f"[yellow]⚠️ No products found for brand {brand_id}[/yellow]")
                return {}
            
            # Get brand name from first row
            brand_name = df['brandName'].iloc[0] if 'brandName' in df.columns else df['brandId'].iloc[0]
            
            self.console.print(Panel(
                f"[bold green]Processing Brand from Database[/bold green]\n"
                f"Brand ID: {brand_id}\nProducts: {len(df)}",
                style="green"
            ))
            
            # Process through the pipeline
            processed_df, embeddings, core_identities = self.process_brand_data(df, brand_name)
            cluster_labels = self.perform_clustering(embeddings, brand_name)
            results = self.generate_output(processed_df, cluster_labels, brand_name)
            
            # Save results to CSV and append to master
            self._save_and_append_results(results, brand_name, processed_df)
            
            # Display results
            self.display_results_summary(results)
            
            return {brand_name: results}
            
        except Exception as e:
            self.logger.error(f"Failed to process brand {brand_id}: {e}")
            raise
    
    def _process_all_brands_from_db(self, batch_size: int = 1000) -> Dict[str, Any]:
        """Process all brands from the database."""
        try:
            # Get all brands
            brands_df = self.db_manager.get_all_brands()
            
            if brands_df.empty:
                self.console.print("[yellow]⚠️ No verified brands found[/yellow]")
                return {}
            
            all_results = {}
            total_brands = len(brands_df)
            
            self.console.print(f"[blue]📊 Processing {total_brands} brands from database[/blue]")
            
            for idx, brand_row in brands_df.iterrows():
                brand_id = brand_row['brandId']
                brand_name = brand_row['brandName']
                
                self.console.print(f"\n[cyan]═══ Brand {idx+1}/{total_brands}: {brand_name} ═══[/cyan]")
                
                try:
                    # Fetch and process brand data
                    df = self.db_manager.fetch_brand_products(brand_id)
                    
                    if df.empty:
                        self.console.print(f"[yellow]⚠️ No products found for {brand_name}[/yellow]")
                        continue
                    
                    # Use brand name from fetched data for consistent file naming
                    actual_brand_name = df['brandName'].iloc[0] if 'brandName' in df.columns else brand_name
                    
                    # Process through the pipeline
                    processed_df, embeddings, core_identities = self.process_brand_data(df, actual_brand_name)
                    cluster_labels = self.perform_clustering(embeddings, actual_brand_name)
                    results = self.generate_output(processed_df, cluster_labels, actual_brand_name)
                    
                    # Save and append results
                    self._save_and_append_results(results, actual_brand_name, processed_df)
                    
                    # Display results
                    self.display_results_summary(results)
                    
                    all_results[actual_brand_name] = results
                    
                    self.console.print(f"[green]✅ {actual_brand_name} processing complete[/green]")
                    
                except Exception as e:
                    self.logger.error(f"Failed to process brand {brand_name}: {e}")
                    self.console.print(f"[red]✗ Error processing {brand_name}: {e}[/red]")
                    continue
            
            # Save final master CSV
            self._save_master_results()
            
            # Save combined JSON results
            combined_file = self.output_dir / "all_brands_database_results.json"
            with open(combined_file, 'w', encoding='utf-8') as f:
                json.dump(all_results, f, indent=2, ensure_ascii=False)
            
            self.console.print(f"[green]📁 Combined results saved to: {combined_file}[/green]")
            
            return all_results
            
        except Exception as e:
            self.logger.error(f"Failed to process all brands: {e}")
            raise
    
    def _save_and_append_results(self, results: Dict[str, Any], brand_name: str, processed_df: pd.DataFrame):
        """Save brand results and append to master CSV with conflict resolution."""
        try:
            # Save individual brand results (existing logic)
            self.save_results(results, brand_name, processed_df)
            
            # Prepare data for master CSV with conflict resolution
            brand_results_df = processed_df.copy()
            brand_results_df['brand_name'] = brand_name
            brand_results_df['processing_timestamp'] = pd.Timestamp.now()
            
            # Append to master results with conflict resolution on brandSKUId
            if self.master_results_df.empty:
                self.master_results_df = brand_results_df.copy()
            else:
                # Remove existing entries for this brand to avoid duplicates
                self.master_results_df = self.master_results_df[
                    self.master_results_df['brandId'] != brand_results_df['brandId'].iloc[0]
                ]
                # Append new results
                self.master_results_df = pd.concat([self.master_results_df, brand_results_df], 
                                                 ignore_index=True)
            
            # Live save to master CSV
            master_csv_path = self.output_dir / "master_sister_products_results.csv"
            self.master_results_df.to_csv(master_csv_path, index=False)
            
            self.console.print(f"[green]💾 Live saved to master CSV ({len(self.master_results_df)} total products)[/green]")
            
        except Exception as e:
            self.logger.error(f"Error saving and appending results for {brand_name}: {e}")
            self.console.print(f"[red]✗ Error saving results for {brand_name}: {e}[/red]")
    
    def _save_master_results(self):
        """Save final master results to multiple formats."""
        try:
            if self.master_results_df.empty:
                self.console.print("[yellow]⚠️ No master results to save[/yellow]")
                return
            
            # Save master CSV
            master_csv_path = self.output_dir / "master_sister_products_results.csv"
            self.master_results_df.to_csv(master_csv_path, index=False)
            
            # Save summary statistics
            summary_stats = {
                'total_products': len(self.master_results_df),
                'total_brands': self.master_results_df['brandId'].nunique(),
                'total_clusters': self.master_results_df['cluster_id'].nunique() - (1 if -1 in self.master_results_df['cluster_id'].values else 0),
                'products_with_sisters': len(self.master_results_df[self.master_results_df['cluster_id'] != -1]),
                'clustering_rate': len(self.master_results_df[self.master_results_df['cluster_id'] != -1]) / len(self.master_results_df) * 100,
                'processing_timestamp': pd.Timestamp.now().isoformat()
            }
            
            summary_file = self.output_dir / "master_processing_summary.json"
            with open(summary_file, 'w', encoding='utf-8') as f:
                json.dump(summary_stats, f, indent=2, ensure_ascii=False)
            
            self.console.print(f"[green]📊 Master results saved: {len(self.master_results_df)} products across {summary_stats['total_brands']} brands[/green]")
            self.console.print(f"[blue]📈 Overall clustering rate: {summary_stats['clustering_rate']:.1f}%[/blue]")
            
        except Exception as e:
            self.logger.error(f"Error saving master results: {e}")
            self.console.print(f"[red]✗ Error saving master results: {e}[/red]") 