#!/usr/bin/env python3
"""
Configuration file for dynamic clustering parameters.
This allows the system to work with any brand and product category.
"""

import re
from typing import Dict, List, Any

class ClusteringConfig:
    """Configuration class for dynamic clustering parameters."""
    
    def __init__(self, config_file: str = None):
        """
        Initialize configuration with default values or load from file.
        
        Args:
            config_file: Optional path to JSON configuration file
        """
        self.load_default_config()
        
        if config_file:
            self.load_from_file(config_file)
    
    def load_default_config(self):
        """Load default configuration values."""
        
        # Clustering parameters
        self.min_cluster_size = 2
        self.min_samples = 1
        self.similarity_threshold = 0.5
        self.core_term_boost = 0.4
        self.min_word_overlap_similarity = 0.3
        
        # Brand name removal patterns (regex patterns)
        self.brand_removal_patterns = [
            r'^[a-zA-Z0-9\s]+\s+',  # Generic brand name removal
            r'^[A-Z]+\s+',  # All caps brand names
        ]
        
        # Flavor/variant indicators (can be extended per category)
        self.flavor_indicators = {
            'food_snacks': [
                'cheese and herbs', 'sweet chili', 'lime and mint', 'sizzlin jalapeno', 
                'hot & spicy', 'wasabi', 'tikka masala', 'barbeque', 'bbq',
                'peri peri', 'peri', 'chili', 'jalapeno', 'mint', 'lime',
                'cheese', 'herbs', 'sweet', 'spicy', 'hot', 'sizzlin',
                'no onion garlic', 'onion garlic', 'garlic', 'onion',
                'coated', 'roasted', 'baked', 'fried', 'crisps', 'crispy'
            ],
            'cosmetics': [
                'matte', 'glossy', 'shimmer', 'pearl', 'nude', 'natural',
                'oily', 'dry', 'combination', 'sensitive', 'normal',
                'anti-aging', 'moisturizing', 'hydrating', 'exfoliating'
            ],
            'cleaning': [
                'lemon', 'lavender', 'pine', 'citrus', 'fresh', 'clean',
                'antibacterial', 'disinfectant', 'stain remover', 'fabric softener'
            ],
            'spices': [
                'whole', 'powder', 'ground', 'crushed', 'seeds', 'leaves',
                'hot', 'mild', 'extra hot', 'garam', 'tandoori', 'curry'
            ]
        }
        
        # Core product terms (can be extended per category)
        self.core_terms = {
            'food_snacks': [
                'nachos', 'chips', 'nuts', 'almonds', 'peanuts', 'cashews', 'raisins', 
                'seeds', 'peas', 'corn', 'taco', 'shell', 'jalapenos', 'pickles', 'namkeen',
                'mixture', 'bhel', 'katli', 'gujia', 'khatai', 'bakarwadi', 'puri'
            ],
            'cosmetics': [
                'foundation', 'concealer', 'powder', 'lipstick', 'mascara', 'eyeliner',
                'blush', 'bronzer', 'highlighter', 'primer', 'serum', 'moisturizer'
            ],
            'cleaning': [
                'detergent', 'soap', 'shampoo', 'conditioner', 'body wash', 'hand wash',
                'floor cleaner', 'glass cleaner', 'toilet cleaner', 'fabric softener'
            ],
            'spices': [
                'powder', 'whole', 'seeds', 'leaves', 'masala', 'chutney', 'pickle',
                'turmeric', 'coriander', 'cumin', 'cardamom', 'pepper', 'chili'
            ]
        }
        
        # Packaging terms (extensible)
        self.packaging_terms = [
            'bag', 'pack', 'packet', 'pouch', 'tin', 'can', 'jar', 'bottle', 'box', 'combo',
            'tube', 'stick', 'bar', 'sachet', 'refill', 'dispenser', 'spray', 'pump'
        ]
        
        # Variant patterns (extensible regex patterns)
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
        
        # Category mapping for automatic category detection
        self.category_mapping = {
            'food_snacks': ['sweets', 'snacks', 'food', 'beverages', 'confectionery'],
            'cosmetics': ['beauty', 'cosmetics', 'skincare', 'makeup', 'personal care'],
            'cleaning': ['cleaning', 'household', 'detergent', 'soap', 'hygiene'],
            'spices': ['spices', 'condiments', 'masala', 'herbs', 'seasoning']
        }
    
    def get_flavor_indicators(self, category: str = None) -> List[str]:
        """Get flavor indicators for a specific category."""
        if category and category in self.flavor_indicators:
            return self.flavor_indicators[category]
        
        # Return all flavor indicators if category not found
        all_indicators = []
        for indicators in self.flavor_indicators.values():
            all_indicators.extend(indicators)
        return list(set(all_indicators))
    
    def get_core_terms(self, category: str = None) -> List[str]:
        """Get core terms for a specific category."""
        if category and category in self.core_terms:
            return self.core_terms[category]
        
        # Return all core terms if category not found
        all_terms = []
        for terms in self.core_terms.values():
            all_terms.extend(terms)
        return list(set(all_terms))
    
    def detect_category(self, category_label: str) -> str:
        """Detect the most appropriate category based on category label."""
        category_label_lower = category_label.lower()
        
        for cat_type, keywords in self.category_mapping.items():
            if any(keyword in category_label_lower for keyword in keywords):
                return cat_type
        
        return 'food_snacks'  # Default fallback
    
    def add_brand_pattern(self, brand_name: str):
        """Add a specific brand name removal pattern."""
        pattern = rf'^{re.escape(brand_name.lower())}\s+'
        if pattern not in self.brand_removal_patterns:
            self.brand_removal_patterns.append(pattern)
    
    def add_flavor_indicators(self, category: str, indicators: List[str]):
        """Add flavor indicators for a specific category."""
        if category not in self.flavor_indicators:
            self.flavor_indicators[category] = []
        self.flavor_indicators[category].extend(indicators)
    
    def add_core_terms(self, category: str, terms: List[str]):
        """Add core terms for a specific category."""
        if category not in self.core_terms:
            self.core_terms[category] = []
        self.core_terms[category].extend(terms)
    
    def load_from_file(self, config_file: str):
        """Load configuration from JSON file."""
        import json
        try:
            with open(config_file, 'r') as f:
                config_data = json.load(f)
            
            # Update configuration with file data
            for key, value in config_data.items():
                if hasattr(self, key):
                    setattr(self, key, value)
        except Exception as e:
            print(f"Warning: Could not load config file {config_file}: {e}")
            print("Using default configuration.")
