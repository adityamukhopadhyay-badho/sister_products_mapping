#!/usr/bin/env python3
"""
Sister Products Visualization Module

This module creates various visualizations for sister product clustering results:
- Interactive network graphs showing product relationships
- Cluster size distributions
- Category-wise analysis
- Interactive HTML dashboards

Author: Sister Products Mapping System
Date: 2024
"""

import json
import logging
from pathlib import Path
from typing import Dict, List, Any, Tuple
import colorsys

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import plotly.offline as pyo
import networkx as nx
from rich.console import Console
from rich.progress import Progress

class SisterProductVisualizer:
    """
    Creates comprehensive visualizations for sister product clustering results.
    """
    
    def __init__(self, visualizations_dir: str = 'visualizations'):
        """
        Initialize the visualizer.
        
        Args:
            visualizations_dir: Directory to save visualizations
        """
        self.visualizations_dir = Path(visualizations_dir)
        self.visualizations_dir.mkdir(exist_ok=True)
        
        self.console = Console()
        self.logger = logging.getLogger(__name__)
        
        # Set style for matplotlib
        plt.style.use('seaborn-v0_8')
        sns.set_palette("husl")
        
    def generate_color_palette(self, n_colors: int) -> List[str]:
        """Generate a diverse color palette."""
        colors = []
        for i in range(n_colors):
            hue = i / n_colors
            rgb = colorsys.hsv_to_rgb(hue, 0.7, 0.9)
            hex_color = '#{:02x}{:02x}{:02x}'.format(
                int(rgb[0] * 255), int(rgb[1] * 255), int(rgb[2] * 255)
            )
            colors.append(hex_color)
        return colors
    
    def create_cluster_size_distribution(self, results: Dict[str, Any], brand_name: str):
        """Create cluster size distribution visualization."""
        clusters = results['sisterProductClusters']
        cluster_sizes = []
        cluster_names = []
        
        for cluster_name, products in clusters.items():
            if cluster_name != 'no_sisters':
                cluster_sizes.append(len(products))
                cluster_names.append(cluster_name)
        
        if not cluster_sizes:
            return
        
        # Create matplotlib figure
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle(f'{brand_name} - Sister Product Clusters Analysis', fontsize=16, fontweight='bold')
        
        # 1. Cluster size histogram
        axes[0, 0].hist(cluster_sizes, bins=max(10, len(cluster_sizes)//5), alpha=0.7, color='skyblue')
        axes[0, 0].set_xlabel('Cluster Size (Number of Products)')
        axes[0, 0].set_ylabel('Number of Clusters')
        axes[0, 0].set_title('Distribution of Cluster Sizes')
        axes[0, 0].grid(True, alpha=0.3)
        
        # 2. Top 10 largest clusters
        if len(cluster_sizes) > 0:
            top_clusters = sorted(zip(cluster_names, cluster_sizes), key=lambda x: x[1], reverse=True)[:10]
            if top_clusters:
                names, sizes = zip(*top_clusters)
                y_pos = np.arange(len(names))
                axes[0, 1].barh(y_pos, sizes, color='lightcoral')
                axes[0, 1].set_yticks(y_pos)
                axes[0, 1].set_yticklabels([name.replace('cluster_', 'C') for name in names])
                axes[0, 1].set_xlabel('Number of Products')
                axes[0, 1].set_title('Top 10 Largest Clusters')
                axes[0, 1].grid(True, alpha=0.3)
        
        # 3. Cluster statistics
        total_products = results['total_products']
        clustered_products = results['products_with_sisters']
        
        labels = ['Products with Sisters', 'Products without Sisters']
        sizes = [clustered_products, total_products - clustered_products]
        colors = ['lightgreen', 'lightgray']
        
        axes[1, 0].pie(sizes, labels=labels, colors=colors, autopct='%1.1f%%', startangle=90)
        axes[1, 0].set_title('Products Clustering Overview')
        
        # 4. Summary statistics table
        axes[1, 1].axis('off')
        stats_data = [
            ['Total Products', str(total_products)],
            ['Total Clusters', str(results['total_clusters'])],
            ['Products with Sisters', str(clustered_products)],
            ['Products without Sisters', str(results['products_without_sisters'])],
            ['Average Cluster Size', f"{np.mean(cluster_sizes):.1f}" if cluster_sizes else "0"],
            ['Max Cluster Size', str(max(cluster_sizes)) if cluster_sizes else "0"],
            ['Min Cluster Size', str(min(cluster_sizes)) if cluster_sizes else "0"]
        ]
        
        table = axes[1, 1].table(cellText=stats_data, 
                                colLabels=['Metric', 'Value'],
                                cellLoc='left',
                                loc='center')
        table.auto_set_font_size(False)
        table.set_fontsize(10)
        table.scale(1.2, 1.5)
        axes[1, 1].set_title('Summary Statistics')
        
        plt.tight_layout()
        
        # Save the plot
        output_file = self.visualizations_dir / f"{brand_name.lower()}_cluster_analysis.png"
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        plt.close()
        
        self.console.print(f"[green]Saved cluster analysis: {output_file}[/green]")
    
    def create_category_analysis(self, results: Dict[str, Any], brand_name: str):
        """Create category-wise analysis visualization."""
        clusters = results['sisterProductClusters']
        
        # Collect category data
        category_cluster_map = {}
        category_product_counts = {}
        
        for cluster_name, products in clusters.items():
            if cluster_name == 'no_sisters':
                continue
                
            for product in products:
                # Use primary_category if available, otherwise fall back to categoryLabel
                category = product.get('primary_category', product['categoryLabel'])
                
                if category not in category_cluster_map:
                    category_cluster_map[category] = []
                    category_product_counts[category] = 0
                
                category_cluster_map[category].append(cluster_name)
                category_product_counts[category] += 1
        
        if not category_product_counts:
            return
        
        # Create visualization
        fig, axes = plt.subplots(2, 1, figsize=(12, 10))
        fig.suptitle(f'{brand_name} - Category Analysis', fontsize=16, fontweight='bold')
        
        # 1. Products per category
        categories = list(category_product_counts.keys())
        counts = list(category_product_counts.values())
        
        axes[0].bar(range(len(categories)), counts, color='steelblue')
        axes[0].set_xticks(range(len(categories)))
        axes[0].set_xticklabels(categories, rotation=45, ha='right')
        axes[0].set_ylabel('Number of Products')
        axes[0].set_title('Products per Category (in clusters)')
        axes[0].grid(True, alpha=0.3)
        
        # 2. Clusters per category
        clusters_per_category = {cat: len(set(clusters)) for cat, clusters in category_cluster_map.items()}
        
        axes[1].bar(range(len(categories)), 
                   [clusters_per_category[cat] for cat in categories], 
                   color='coral')
        axes[1].set_xticks(range(len(categories)))
        axes[1].set_xticklabels(categories, rotation=45, ha='right')
        axes[1].set_ylabel('Number of Clusters')
        axes[1].set_title('Clusters per Category')
        axes[1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        # Save the plot
        output_file = self.visualizations_dir / f"{brand_name.lower()}_category_analysis.png"
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        plt.close()
        
        self.console.print(f"[green]Saved category analysis: {output_file}[/green]")
    
    def create_network_graph(self, results: Dict[str, Any], brand_name: str):
        """Create an interactive network graph of sister products."""
        clusters = results['sisterProductClusters']
        
        # Create network graph
        G = nx.Graph()
        
        # Color palette for clusters
        cluster_colors = {}
        color_palette = self.generate_color_palette(len(clusters))
        
        node_info = []
        edge_info = []
        
        color_idx = 0
        for cluster_name, products in clusters.items():
            if cluster_name == 'no_sisters':
                continue
                
            cluster_color = color_palette[color_idx % len(color_palette)]
            cluster_colors[cluster_name] = cluster_color
            color_idx += 1
            
            # Add nodes for this cluster
            for product in products:
                node_id = product['brandSKUId']
                G.add_node(node_id)
                
                node_info.append({
                    'id': node_id,
                    'label': product['label'],
                    'normalized_name': product['normalized_name'],
                    'category': product.get('primary_category', product['categoryLabel']),
                    'categories_all': product.get('categories_parsed', product['categoryLabel']),
                    'cluster': cluster_name,
                    'color': cluster_color
                })
            
            # Add edges between all products in the same cluster
            product_ids = [p['brandSKUId'] for p in products]
            for i, id1 in enumerate(product_ids):
                for id2 in product_ids[i+1:]:
                    G.add_edge(id1, id2)
                    edge_info.append({
                        'source': id1,
                        'target': id2,
                        'cluster': cluster_name
                    })
        
        if len(G.nodes()) == 0:
            self.console.print(f"[yellow]No clusters to visualize for {brand_name}[/yellow]")
            return
        
        # Generate layout
        pos = nx.spring_layout(G, k=1, iterations=50)
        
        # Create plotly figure
        edge_trace = []
        for edge in G.edges():
            x0, y0 = pos[edge[0]]
            x1, y1 = pos[edge[1]]
            edge_trace.append(
                go.Scatter(x=[x0, x1, None], y=[y0, y1, None],
                          mode='lines',
                          line=dict(width=0.5, color='#888'),
                          hoverinfo='none',
                          showlegend=False)
            )
        
        # Create node trace
        node_trace = go.Scatter(x=[], y=[], mode='markers+text',
                               hoverinfo='text',
                               text=[],
                               hovertext=[],
                               marker=dict(size=10,
                                         color=[],
                                         line=dict(width=2, color='white')))
        
        for node_data in node_info:
            node_id = node_data['id']
            x, y = pos[node_id]
            node_trace['x'] += tuple([x])
            node_trace['y'] += tuple([y])
            node_trace['marker']['color'] += tuple([node_data['color']])
            
            # Shortened label for display
            short_label = node_data['label'][:20] + "..." if len(node_data['label']) > 20 else node_data['label']
            node_trace['text'] += tuple([short_label])
            
            hover_text = (f"<b>{node_data['label']}</b><br>"
                         f"Normalized: {node_data['normalized_name']}<br>"
                         f"Category: {node_data['category']}<br>"
                         f"Cluster: {node_data['cluster']}")
            node_trace['hovertext'] += tuple([hover_text])
        
        # Create figure
        fig = go.Figure(data=edge_trace + [node_trace],
                       layout=go.Layout(
                           title=f'{brand_name} - Sister Products Network',
                           titlefont_size=16,
                           showlegend=False,
                           hovermode='closest',
                           margin=dict(b=20,l=5,r=5,t=40),
                           annotations=[ dict(
                               text="Connected products are sister products (same cluster)",
                               showarrow=False,
                               xref="paper", yref="paper",
                               x=0.005, y=-0.002,
                               xanchor='left', yanchor='bottom',
                               font=dict(color='gray', size=10)
                           )],
                           xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                           yaxis=dict(showgrid=False, zeroline=False, showticklabels=False)))
        
        # Save as HTML
        output_file = self.visualizations_dir / f"{brand_name.lower()}_network_graph.html"
        pyo.plot(fig, filename=str(output_file), auto_open=False)
        
        self.console.print(f"[green]Saved network graph: {output_file}[/green]")
    
    def create_cluster_details_visualization(self, results: Dict[str, Any], brand_name: str):
        """Create detailed cluster visualization with product information."""
        clusters = results['sisterProductClusters']
        
        # Filter out noise and get top clusters
        valid_clusters = {name: products for name, products in clusters.items() 
                         if name != 'no_sisters' and len(products) >= 2}
        
        if not valid_clusters:
            return
        
        # Sort by cluster size
        sorted_clusters = sorted(valid_clusters.items(), key=lambda x: len(x[1]), reverse=True)
        
        # Create subplots for top clusters
        n_clusters_to_show = min(6, len(sorted_clusters))
        fig = make_subplots(
            rows=2, cols=3,
            subplot_titles=[f"{name} ({len(products)} products)" 
                           for name, products in sorted_clusters[:n_clusters_to_show]],
            specs=[[{"type": "scatter"} for _ in range(3)] for _ in range(2)]
        )
        
        colors = self.generate_color_palette(n_clusters_to_show)
        
        for idx, (cluster_name, products) in enumerate(sorted_clusters[:n_clusters_to_show]):
            row = idx // 3 + 1
            col = idx % 3 + 1
            
            # Create a simple scatter plot for products in this cluster
            x_vals = list(range(len(products)))
            y_vals = [1] * len(products)  # All on same line
            
            labels = [p['label'][:30] + "..." if len(p['label']) > 30 else p['label'] 
                     for p in products]
            
            fig.add_trace(
                go.Scatter(
                    x=x_vals, y=y_vals,
                    mode='markers',
                    marker=dict(size=15, color=colors[idx]),
                    text=labels,
                    hovertemplate="<b>%{text}</b><br>" +
                                 f"Cluster: {cluster_name}<br>" +
                                 "<extra></extra>",
                    showlegend=False
                ), 
                row=row, col=col
            )
            
            # Update axes
            fig.update_xaxis(showticklabels=False, showgrid=False, row=row, col=col)
            fig.update_yaxis(showticklabels=False, showgrid=False, row=row, col=col)
        
        fig.update_layout(
            title_text=f"{brand_name} - Top Sister Product Clusters Details",
            title_x=0.5,
            height=600,
            showlegend=False
        )
        
        # Save as HTML
        output_file = self.visualizations_dir / f"{brand_name.lower()}_cluster_details.html"
        pyo.plot(fig, filename=str(output_file), auto_open=False)
        
        self.console.print(f"[green]Saved cluster details: {output_file}[/green]")
    
    def create_interactive_dashboard(self, results: Dict[str, Any], brand_name: str):
        """Create an interactive HTML dashboard with all visualizations."""
        clusters = results['sisterProductClusters']
        
        # Prepare data for dashboard
        cluster_data = []
        for cluster_name, products in clusters.items():
            cluster_data.append({
                'cluster_name': cluster_name,
                'product_count': len(products),
                'products': products
            })
        
        # Create main dashboard figure
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=['Cluster Size Distribution', 'Category Distribution', 
                           'Top Clusters', 'Clustering Statistics'],
            specs=[[{"type": "histogram"}, {"type": "pie"}],
                   [{"type": "bar"}, {"type": "table"}]]
        )
        
        # 1. Cluster size distribution
        cluster_sizes = [data['product_count'] for data in cluster_data if data['cluster_name'] != 'no_sisters']
        if cluster_sizes:
            fig.add_trace(
                go.Histogram(x=cluster_sizes, nbinsx=10, name="Cluster Sizes"),
                row=1, col=1
            )
        
        # 2. Category distribution
        category_counts = {}
        for cluster_name, products in clusters.items():
            if cluster_name == 'no_sisters':
                continue
            for product in products:
                # Use primary_category if available, otherwise fall back to categoryLabel
                category = product.get('primary_category', product['categoryLabel'])
                category_counts[category] = category_counts.get(category, 0) + 1
        
        if category_counts:
            fig.add_trace(
                go.Pie(labels=list(category_counts.keys()), 
                       values=list(category_counts.values()),
                       name="Categories"),
                row=1, col=2
            )
        
        # 3. Top clusters
        valid_clusters = [(name, len(products)) for name, products in clusters.items() 
                         if name != 'no_sisters']
        valid_clusters.sort(key=lambda x: x[1], reverse=True)
        
        if valid_clusters:
            top_clusters = valid_clusters[:10]
            cluster_names, cluster_sizes = zip(*top_clusters)
            
            fig.add_trace(
                go.Bar(x=list(cluster_sizes), y=list(cluster_names), 
                       orientation='h', name="Top Clusters"),
                row=2, col=1
            )
        
        # 4. Statistics table
        stats_data = [
            ['Total Products', results['total_products']],
            ['Total Clusters', results['total_clusters']],
            ['Products with Sisters', results['products_with_sisters']],
            ['Products without Sisters', results['products_without_sisters']],
            ['Avg Cluster Size', f"{np.mean(cluster_sizes):.1f}" if cluster_sizes else "0"],
            ['Max Cluster Size', max(cluster_sizes) if cluster_sizes else 0]
        ]
        
        fig.add_trace(
            go.Table(
                header=dict(values=['Metric', 'Value']),
                cells=dict(values=[[row[0] for row in stats_data],
                                  [row[1] for row in stats_data]])
            ),
            row=2, col=2
        )
        
        fig.update_layout(
            title_text=f"{brand_name} - Sister Products Dashboard",
            title_x=0.5,
            height=800,
            showlegend=False
        )
        
        # Save dashboard
        output_file = self.visualizations_dir / f"{brand_name.lower()}_dashboard.html"
        pyo.plot(fig, filename=str(output_file), auto_open=False)
        
        self.console.print(f"[green]Saved interactive dashboard: {output_file}[/green]")
    
    def generate_all_visualizations(self, results: Dict[str, Any], brand_name: str):
        """Generate all visualizations for a brand."""
        self.console.print(f"[blue]Generating visualizations for {brand_name}...[/blue]")
        
        with Progress() as progress:
            tasks = [
                "Creating cluster size distribution",
                "Creating category analysis", 
                "Creating network graph",
                "Creating cluster details",
                "Creating interactive dashboard"
            ]
            
            task = progress.add_task("[cyan]Generating visualizations...", total=len(tasks))
            
            try:
                self.create_cluster_size_distribution(results, brand_name)
                progress.update(task, advance=1)
                
                self.create_category_analysis(results, brand_name)
                progress.update(task, advance=1)
                
                self.create_network_graph(results, brand_name)
                progress.update(task, advance=1)
                
                self.create_cluster_details_visualization(results, brand_name)
                progress.update(task, advance=1)
                
                self.create_interactive_dashboard(results, brand_name)
                progress.update(task, advance=1)
                
            except Exception as e:
                self.logger.error(f"Error generating visualizations for {brand_name}: {e}")
        
        self.console.print(f"[green]✓ All visualizations generated for {brand_name}[/green]")
    
    def generate_comparison_dashboard(self, all_results: Dict[str, Dict[str, Any]]):
        """Generate a comparison dashboard for multiple brands."""
        if len(all_results) < 2:
            return
        
        self.console.print("[blue]Creating multi-brand comparison dashboard...[/blue]")
        
        # Prepare comparison data
        brands = list(all_results.keys())
        metrics = []
        
        for brand, results in all_results.items():
            metrics.append({
                'brand': brand,
                'total_products': results['total_products'],
                'total_clusters': results['total_clusters'],
                'products_with_sisters': results['products_with_sisters'],
                'clustering_rate': results['products_with_sisters'] / results['total_products'] * 100
            })
        
        df_metrics = pd.DataFrame(metrics)
        
        # Create comparison dashboard
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=['Total Products by Brand', 'Clustering Rate by Brand',
                           'Total Clusters by Brand', 'Products with Sisters by Brand'],
            specs=[[{"type": "bar"}, {"type": "bar"}],
                   [{"type": "bar"}, {"type": "bar"}]]
        )
        
        # Total products
        fig.add_trace(
            go.Bar(x=df_metrics['brand'], y=df_metrics['total_products'], 
                   name="Total Products", marker_color='steelblue'),
            row=1, col=1
        )
        
        # Clustering rate
        fig.add_trace(
            go.Bar(x=df_metrics['brand'], y=df_metrics['clustering_rate'],
                   name="Clustering Rate (%)", marker_color='orange'),
            row=1, col=2
        )
        
        # Total clusters
        fig.add_trace(
            go.Bar(x=df_metrics['brand'], y=df_metrics['total_clusters'],
                   name="Total Clusters", marker_color='green'),
            row=2, col=1
        )
        
        # Products with sisters
        fig.add_trace(
            go.Bar(x=df_metrics['brand'], y=df_metrics['products_with_sisters'],
                   name="Products with Sisters", marker_color='coral'),
            row=2, col=2
        )
        
        fig.update_layout(
            title_text="Multi-Brand Sister Products Comparison",
            title_x=0.5,
            height=700,
            showlegend=False
        )
        
        # Save comparison dashboard
        output_file = self.visualizations_dir / "multi_brand_comparison.html"
        pyo.plot(fig, filename=str(output_file), auto_open=False)
        
        self.console.print(f"[green]Saved comparison dashboard: {output_file}[/green]") 