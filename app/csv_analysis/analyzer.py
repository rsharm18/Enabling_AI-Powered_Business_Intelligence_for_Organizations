"""CSV Analyzer module for business intelligence analysis."""

import pandas as pd
import numpy as np
import pickle
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional, Union
import matplotlib.pyplot as plt
import seaborn as sns

from config import Config

# Set style for visualizations
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 6)


class CSVAnalyzer:
    """Configurable CSV data analyzer for business intelligence."""
    
    def __init__(self, data_path: str, schema: Dict[str, Union[str, List[str]]]):
        """
        Initialize the analyzer with data path and schema.
        
        Args:
            data_path: Path to the CSV file
            schema: Dictionary defining column roles
        """
        self.data_path = Path(data_path)
        self.schema = schema
        self.df = None
        self.analysis_results = {}
        
        # Validate schema
        self._validate_schema()
        
    def _validate_schema(self):
        """Validate that required schema fields are present."""
        required_fields = ['date_column', 'measures', 'dimensions']
        for field in required_fields:
            if field not in self.schema:
                raise ValueError(f"Schema missing required field: {field}")
        
        if not isinstance(self.schema['measures'], list):
            raise ValueError("'measures' must be a list")
        
        if not isinstance(self.schema['dimensions'], list):
            raise ValueError("'dimensions' must be a list")
    
    def load_and_validate_data(self) -> pd.DataFrame:
        """Load CSV data and perform basic validation."""
        self.df = pd.read_csv(self.data_path)
        
        # Convert date column to datetime
        date_col = self.schema['date_column']
        if date_col in self.df.columns:
            self.df[date_col] = pd.to_datetime(self.df[date_col])
        
        # Validate data quality
        validation_info = {
            'shape': self.df.shape,
            'columns': self.df.columns.tolist(),
            'dtypes': self.df.dtypes.astype(str).to_dict(),
            'missing_values': self.df.isnull().sum().to_dict(),
            'schema_used': self.schema
        }
        
        if date_col in self.df.columns:
            validation_info['date_range'] = (
                self.df[date_col].min(),
                self.df[date_col].max()
            )
        
        self.analysis_results['validation'] = validation_info
        return self.df
    
    def run_full_analysis(self) -> Dict:
        """Run all analyses and return complete results."""
        print("Loading and validating data...")
        self.load_and_validate_data()
        
        print("Analyzing overview...")
        self.analyze_overview()
        
        print("Analyzing by dimensions...")
        self.analyze_by_dimensions()
        
        print("Analyzing temporal trends...")
        self.analyze_temporal_trends()
        
        print("Analyzing customer attributes...")
        self.analyze_customer_attributes()
        
        print("Analyzing satisfaction...")
        self.analyze_satisfaction()
        
        print("Analyzing cross-dimension combinations...")
        self.analyze_cross_dimension_combinations()
        
        print("Analyzing correlation matrix...")
        self.analyze_correlation_matrix()
        
        print("Analyzing percentile distributions...")
        self.analyze_percentile_distributions()
        
        print("Detecting outliers...")
        self.analyze_outliers()
        
        print("Analyzing rankings...")
        self.analyze_rankings()
        
        print("Analyzing distribution shape...")
        self.analyze_distribution_shape()
        
        print("Analyzing year-over-year growth...")
        self.analyze_year_over_year_growth()
        
        print("Generating key insights...")
        self.generate_key_insights()
        
        # Add metadata
        self.analysis_results['metadata'] = {
            'analysis_date': datetime.now().isoformat(),
            'data_source': str(self.data_path),
            'analyzer_version': '2.0.0',
            'schema': self.schema
        }
        
        return self.analysis_results
    
    def analyze_overview(self) -> Dict:
        """Generate high-level overview for all measures."""
        overview = {}
        
        for measure in self.schema['measures']:
            if measure in self.df.columns:
                overview[measure] = {
                    'total': float(self.df[measure].sum()),
                    'mean': float(self.df[measure].mean()),
                    'median': float(self.df[measure].median()),
                    'std': float(self.df[measure].std()),
                    'min': float(self.df[measure].min()),
                    'max': float(self.df[measure].max()),
                    'count': int(len(self.df[measure].dropna()))
                }
        
        self.analysis_results['overview'] = overview
        return overview
    
    def analyze_by_dimensions(self) -> Dict:
        """Analyze measures by each dimension."""
        dimension_analysis = {}
        
        for dimension in self.schema['dimensions']:
            if dimension in self.df.columns:
                dimension_stats = {}
                for measure in self.schema['measures']:
                    if measure in self.df.columns:
                        stats = self.df.groupby(dimension)[measure].agg([
                            ('total', 'sum'),
                            ('mean', 'mean'),
                            ('count', 'count'),
                            ('std', 'std')
                        ]).sort_values('total', ascending=False)
                        
                        dimension_stats[measure] = {
                            category: {
                                'total': float(row['total']),
                                'mean': float(row['mean']),
                                'count': int(row['count']),
                                'std': float(row['std'])
                            }
                            for category, row in stats.iterrows()
                        }
                
                dimension_analysis[dimension] = dimension_stats
        
        self.analysis_results['dimension_analysis'] = dimension_analysis
        return dimension_analysis
    
    def analyze_temporal_trends(self) -> Dict:
        """Analyze temporal trends for measures."""
        date_col = self.schema['date_column']
        
        if date_col not in self.df.columns:
            self.analysis_results['temporal_trends'] = {}
            return {}
        
        df_temp = self.df.copy()
        df_temp['Year'] = df_temp[date_col].dt.year
        df_temp['Month'] = df_temp[date_col].dt.month
        df_temp['DayOfWeek'] = df_temp[date_col].dt.day_name()
        
        temporal_analysis = {}
        
        for measure in self.schema['measures']:
            if measure in self.df.columns:
                # Monthly trends
                monthly_sales = df_temp.groupby(['Year', 'Month'])[measure].agg(['sum', 'mean', 'count'])
                monthly_trends = []
                for (year, month), row in monthly_sales.iterrows():
                    monthly_trends.append({
                        'year': int(year),
                        'month': int(month),
                        'total': float(row['sum']),
                        'mean': float(row['mean']),
                        'count': int(row['count'])
                    })
                
                # Daily patterns
                daily_sales = df_temp.groupby('DayOfWeek')[measure].agg(['sum', 'mean', 'count'])
                daily_patterns = {
                    day: {
                        'total': float(row['sum']),
                        'mean': float(row['mean']),
                        'count': int(row['count'])
                    }
                    for day, row in daily_sales.iterrows()
                }
                
                temporal_analysis[measure] = {
                    'monthly_trends': monthly_trends,
                    'daily_patterns': daily_patterns
                }
        
        self.analysis_results['temporal_trends'] = temporal_analysis
        return temporal_analysis
    
    def analyze_customer_attributes(self) -> Dict:
        """Analyze customer attributes and their relationship with measures."""
        customer_attrs = self.schema.get('customer_attributes', [])
        
        if not customer_attrs:
            self.analysis_results['customer_analysis'] = {}
            return {}
        
        customer_analysis = {}
        
        for attr in customer_attrs:
            if attr not in self.df.columns:
                continue
            
            attr_analysis = {}
            
            # For each measure, analyze correlation with this attribute
            for measure in self.schema['measures']:
                if measure in self.df.columns:
                    # Check if attribute is numeric
                    if pd.api.types.is_numeric_dtype(self.df[attr]):
                        correlation = float(self.df[attr].corr(self.df[measure]))
                        
                        # Distribution stats
                        attr_stats = {
                            'min': float(self.df[attr].min()),
                            'max': float(self.df[attr].max()),
                            'mean': float(self.df[attr].mean()),
                            'median': float(self.df[attr].median()),
                            'std': float(self.df[attr].std())
                        }
                        
                        attr_analysis[measure] = {
                            'correlation': correlation,
                            'attribute_stats': attr_stats
                        }
                    else:
                        # Categorical attribute - group by it
                        grouped = self.df.groupby(attr)[measure].agg(['mean', 'sum', 'count'])
                        attr_analysis[measure] = {
                            category: {
                                'mean': float(row['mean']),
                                'total': float(row['sum']),
                                'count': int(row['count'])
                            }
                            for category, row in grouped.iterrows()
                        }
            
            customer_analysis[attr] = attr_analysis
        
        self.analysis_results['customer_analysis'] = customer_analysis
        return customer_analysis
    
    def analyze_satisfaction(self) -> Dict:
        """Analyze satisfaction/metric column if specified."""
        satisfaction_col = self.schema.get('satisfaction_column')
        
        if not satisfaction_col or satisfaction_col not in self.df.columns:
            self.analysis_results['satisfaction_analysis'] = {}
            return {}
        
        satisfaction_analysis = {}
        
        for measure in self.schema['measures']:
            if measure in self.df.columns:
                correlation = float(self.df[satisfaction_col].corr(self.df[measure]))
                
                # Satisfaction by dimensions
                by_dimension = {}
                for dimension in self.schema['dimensions']:
                    if dimension in self.df.columns:
                        grouped = self.df.groupby(dimension)[satisfaction_col].agg(['mean', 'std', 'min', 'max'])
                        by_dimension[dimension] = {
                            category: {
                                'mean': float(row['mean']),
                                'std': float(row['std']),
                                'min': float(row['min']),
                                'max': float(row['max'])
                            }
                            for category, row in grouped.iterrows()
                        }
                
                # Overall satisfaction stats
                satisfaction_stats = {
                    'mean': float(self.df[satisfaction_col].mean()),
                    'median': float(self.df[satisfaction_col].median()),
                    'std': float(self.df[satisfaction_col].std()),
                    'min': float(self.df[satisfaction_col].min()),
                    'max': float(self.df[satisfaction_col].max())
                }
                
                satisfaction_analysis[measure] = {
                    'correlation': correlation,
                    'by_dimension': by_dimension,
                    'stats': satisfaction_stats
                }
        
        self.analysis_results['satisfaction_analysis'] = satisfaction_analysis
        return satisfaction_analysis
    
    def analyze_cross_dimension_combinations(self) -> Dict:
        """Analyze measure performance across dimension combinations."""
        cross_analysis = {}
        
        for measure in self.schema['measures']:
            if measure not in self.df.columns:
                continue
            
            # Analyze top 2 dimensions combinations
            dimensions = [d for d in self.schema['dimensions'] if d in self.df.columns]
            
            if len(dimensions) >= 2:
                combo = self.df.groupby(dimensions[:2])[measure].agg(['sum', 'mean', 'count'])
                combinations = []
                for combo_idx, row in combo.iterrows():
                    combinations.append({
                        'dimensions': dict(zip(dimensions[:2], combo_idx)),
                        'total': float(row['sum']),
                        'mean': float(row['mean']),
                        'count': int(row['count'])
                    })
                cross_analysis[measure] = combinations
        
        self.analysis_results['cross_dimension_analysis'] = cross_analysis
        return cross_analysis
    
    def analyze_correlation_matrix(self) -> Dict:
        """Analyze correlations between all numeric columns."""
        # Get all numeric columns
        numeric_cols = []
        
        for measure in self.schema['measures']:
            if measure in self.df.columns and pd.api.types.is_numeric_dtype(self.df[measure]):
                numeric_cols.append(measure)
        
        customer_attrs = self.schema.get('customer_attributes', [])
        for attr in customer_attrs:
            if attr in self.df.columns and pd.api.types.is_numeric_dtype(self.df[attr]):
                numeric_cols.append(attr)
        
        satisfaction_col = self.schema.get('satisfaction_column')
        if satisfaction_col and satisfaction_col in self.df.columns:
            numeric_cols.append(satisfaction_col)
        
        if len(numeric_cols) < 2:
            self.analysis_results['correlation_matrix'] = {}
            return {}
        
        # Calculate correlation matrix
        corr_matrix = self.df[numeric_cols].corr()
        
        # Convert to serializable format
        correlation_data = {}
        for i, col1 in enumerate(numeric_cols):
            for j, col2 in enumerate(numeric_cols):
                if i < j:  # Only store upper triangle to avoid duplicates
                    key = f"{col1}_vs_{col2}"
                    correlation_data[key] = float(corr_matrix.loc[col1, col2])
        
        self.analysis_results['correlation_matrix'] = correlation_data
        return correlation_data
    
    def analyze_percentile_distributions(self) -> Dict:
        """Analyze percentile distributions for all measures."""
        percentile_analysis = {}
        
        percentiles = [5, 10, 25, 50, 75, 90, 95, 99]
        
        for measure in self.schema['measures']:
            if measure not in self.df.columns:
                continue
            
            percentile_values = {}
            for p in percentiles:
                percentile_values[f'p{p}'] = float(self.df[measure].quantile(p/100))
            
            percentile_analysis[measure] = percentile_values
        
        self.analysis_results['percentile_distributions'] = percentile_analysis
        return percentile_analysis
    
    def analyze_outliers(self) -> Dict:
        """Detect outliers using IQR method for all measures."""
        outlier_analysis = {}
        
        for measure in self.schema['measures']:
            if measure not in self.df.columns:
                continue
            
            Q1 = self.df[measure].quantile(0.25)
            Q3 = self.df[measure].quantile(0.75)
            IQR = Q3 - Q1
            
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            
            outliers = self.df[(self.df[measure] < lower_bound) | (self.df[measure] > upper_bound)]
            
            outlier_analysis[measure] = {
                'count': int(len(outliers)),
                'percentage': float(len(outliers) / len(self.df) * 100),
                'lower_bound': float(lower_bound),
                'upper_bound': float(upper_bound),
                'outlier_values': outliers[measure].tolist()[:100]  # Limit to first 100
            }
        
        self.analysis_results['outlier_analysis'] = outlier_analysis
        return outlier_analysis
    
    def analyze_rankings(self) -> Dict:
        """Generate ranking tables for dimensions by measures."""
        ranking_analysis = {}
        
        for dimension in self.schema['dimensions']:
            if dimension not in self.df.columns:
                continue
            
            ranking_analysis[dimension] = {}
            
            for measure in self.schema['measures']:
                if measure not in self.df.columns:
                    continue
                
                # Rank by total
                ranked = self.df.groupby(dimension)[measure].sum().sort_values(ascending=False)
                ranking_analysis[dimension][f'{measure}_by_total'] = [
                    {
                        'rank': i + 1,
                        'value': category,
                        'total': float(total)
                    }
                    for i, (category, total) in enumerate(ranked.items())
                ]
                
                # Rank by average
                ranked_avg = self.df.groupby(dimension)[measure].mean().sort_values(ascending=False)
                ranking_analysis[dimension][f'{measure}_by_average'] = [
                    {
                        'rank': i + 1,
                        'value': category,
                        'average': float(avg)
                    }
                    for i, (category, avg) in enumerate(ranked_avg.items())
                ]
        
        self.analysis_results['rankings'] = ranking_analysis
        return ranking_analysis
    
    def analyze_distribution_shape(self) -> Dict:
        """Analyze distribution shape (skewness, kurtosis) for measures."""
        distribution_analysis = {}
        
        for measure in self.schema['measures']:
            if measure not in self.df.columns:
                continue
            
            distribution_analysis[measure] = {
                'skewness': float(self.df[measure].skew()),
                'kurtosis': float(self.df[measure].kurtosis()),
                'is_normal': abs(self.df[measure].skew()) < 0.5 and abs(self.df[measure].kurtosis()) < 3
            }
        
        self.analysis_results['distribution_shape'] = distribution_analysis
        return distribution_analysis
    
    def analyze_year_over_year_growth(self) -> Dict:
        """Analyze year-over-year growth if multiple years exist."""
        date_col = self.schema['date_column']
        
        if date_col not in self.df.columns:
            self.analysis_results['yoy_growth'] = {}
            return {}
        
        df_temp = self.df.copy()
        df_temp['Year'] = df_temp[date_col].dt.year
        years = sorted(df_temp['Year'].unique())
        
        if len(years) < 2:
            self.analysis_results['yoy_growth'] = {}
            return {}
        
        yoy_analysis = {}
        
        for measure in self.schema['measures']:
            if measure not in self.df.columns:
                continue
            
            yearly_totals = df_temp.groupby('Year')[measure].sum()
            growth_rates = {}
            
            for i in range(1, len(years)):
                prev_year = years[i-1]
                curr_year = years[i]
                prev_total = yearly_totals[prev_year]
                curr_total = yearly_totals[curr_year]
                
                if prev_total != 0:
                    growth_rate = ((curr_total - prev_total) / prev_total) * 100
                    growth_rates[f'{prev_year}_to_{curr_year}'] = {
                        'growth_rate_percent': float(growth_rate),
                        'previous_year_total': float(prev_total),
                        'current_year_total': float(curr_total),
                        'absolute_change': float(curr_total - prev_total)
                    }
            
            yoy_analysis[measure] = growth_rates
        
        self.analysis_results['yoy_growth'] = yoy_analysis
        return yoy_analysis
    
    def generate_key_insights(self) -> List[str]:
        """Generate actionable business insights."""
        insights = []
        
        # Best performing dimension values
        for dimension in self.schema['dimensions']:
            if dimension not in self.df.columns:
                continue
            
            for measure in self.schema['measures']:
                if measure not in self.df.columns:
                    continue
                
                best_value = self.df.groupby(dimension)[measure].sum().idxmax()
                best_total = self.df.groupby(dimension)[measure].sum().max()
                insights.append(
                    f"Best {dimension} for {measure}: {best_value} "
                    f"with ${best_total:,.2f}"
                )
        
        # Temporal peak
        date_col = self.schema['date_column']
        if date_col in self.df.columns:
            for measure in self.schema['measures']:
                if measure not in self.df.columns:
                    continue
                
                df_temp = self.df.copy()
                df_temp['Month'] = df_temp[date_col].dt.month
                peak_month = df_temp.groupby('Month')[measure].sum().idxmax()
                month_names = {
                    1: 'January', 2: 'February', 3: 'March', 4: 'April',
                    5: 'May', 6: 'June', 7: 'July', 8: 'August',
                    9: 'September', 10: 'October', 11: 'November', 12: 'December'
                }
                insights.append(
                    f"Peak month for {measure}: {month_names[peak_month]}"
                )
        
        # Satisfaction insight
        satisfaction_col = self.schema.get('satisfaction_column')
        if satisfaction_col and satisfaction_col in self.df.columns:
            avg_sat = self.df[satisfaction_col].mean()
            insights.append(f"Average {satisfaction_col}: {avg_sat:.2f}")
        
        self.analysis_results['key_insights'] = insights
        return insights
    
    def save_analysis_to_pickle(self, output_path: str = None) -> str:
        """Save analysis results to a pickle file for LLM consumption."""
        if output_path is None:
            output_path = Config.OUTPUT_DIR / 'csv_analysis_results.pkl'
        
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'wb') as f:
            pickle.dump(self.analysis_results, f, protocol=pickle.HIGHEST_PROTOCOL)
        
        print(f"Analysis results saved to: {output_path}")
        return str(output_path)
    
    def create_visualizations(self, output_path: str = None, max_charts: int = 12) -> str:
        """Create data visualizations for key insights."""
        if self.df is None:
            raise ValueError("Data not loaded. Call load_and_validate_data() first.")
        
        # Determine number of charts based on available data
        n_measures = len([m for m in self.schema['measures'] if m in self.df.columns])
        n_dimensions = len([d for d in self.schema['dimensions'] if d in self.df.columns])
        
        # Calculate number of possible dimension pair heatmaps
        dimensions = [d for d in self.schema['dimensions'] if d in self.df.columns]
        n_dimension_pairs = len(dimensions) * (len(dimensions) - 1) // 2 if len(dimensions) >= 2 else 0
        
        n_charts = min(max_charts, n_measures + n_dimensions + 3 + n_dimension_pairs)
        n_rows = (n_charts + 2) // 3
        
        fig, axes = plt.subplots(n_rows, 3, figsize=(18, 6 * n_rows))
        if n_charts < 6:
            axes = axes.flatten() if n_rows > 1 else [axes] if n_charts == 1 else axes
        else:
            axes = axes.flatten()
        
        fig.suptitle('CSV Data Analysis Dashboard', fontsize=16, fontweight='bold')
        
        chart_idx = 0
        
        # 1. Distribution of first measure
        first_measure = [m for m in self.schema['measures'] if m in self.df.columns][0]
        if chart_idx < n_charts:
            axes[chart_idx].hist(self.df[first_measure], bins=30, color='skyblue', edgecolor='black')
            axes[chart_idx].set_title(f'{first_measure} Distribution')
            axes[chart_idx].set_xlabel(first_measure)
            axes[chart_idx].set_ylabel('Frequency')
            chart_idx += 1
        
        # 2. First dimension breakdown
        first_dimension = [d for d in self.schema['dimensions'] if d in self.df.columns][0]
        if chart_idx < n_charts:
            dim_sales = self.df.groupby(first_dimension)[first_measure].sum().sort_values(ascending=True)
            axes[chart_idx].barh(dim_sales.index, dim_sales.values, color='lightcoral')
            axes[chart_idx].set_title(f'{first_measure} by {first_dimension}')
            axes[chart_idx].set_xlabel(f'Total {first_measure}')
            chart_idx += 1
        
        # 3. Second dimension pie chart
        if len([d for d in self.schema['dimensions'] if d in self.df.columns]) >= 2 and chart_idx < n_charts:
            second_dimension = [d for d in self.schema['dimensions'] if d in self.df.columns][1]
            region_sales = self.df.groupby(second_dimension)[first_measure].sum()
            axes[chart_idx].pie(region_sales.values, labels=region_sales.index, autopct='%1.1f%%')
            axes[chart_idx].set_title(f'{first_measure} by {second_dimension}')
            chart_idx += 1
        
        # 4. Temporal trend
        date_col = self.schema['date_column']
        if date_col in self.df.columns and chart_idx < n_charts:
            df_temp = self.df.copy()
            df_temp['YearMonth'] = df_temp[date_col].dt.to_period('M')
            monthly_trend = df_temp.groupby('YearMonth')[first_measure].sum()
            axes[chart_idx].plot(monthly_trend.index.astype(str), monthly_trend.values, marker='o', color='green')
            axes[chart_idx].set_title(f'Monthly {first_measure} Trend')
            axes[chart_idx].set_xlabel('Month')
            axes[chart_idx].set_ylabel(f'Total {first_measure}')
            axes[chart_idx].tick_params(axis='x', rotation=45)
            chart_idx += 1
        
        # 5. Box plot for outlier detection
        if chart_idx < n_charts:
            axes[chart_idx].boxplot(self.df[first_measure])
            axes[chart_idx].set_title(f'{first_measure} Box Plot (Outliers)')
            axes[chart_idx].set_ylabel(first_measure)
            chart_idx += 1
        
        # 6. Correlation heatmap if available
        if 'correlation_matrix' in self.analysis_results and chart_idx < n_charts:
            # Rebuild correlation matrix for visualization
            numeric_cols = []
            for measure in self.schema['measures']:
                if measure in self.df.columns and pd.api.types.is_numeric_dtype(self.df[measure]):
                    numeric_cols.append(measure)
            
            customer_attrs = self.schema.get('customer_attributes', [])
            for attr in customer_attrs:
                if attr in self.df.columns and pd.api.types.is_numeric_dtype(self.df[attr]):
                    numeric_cols.append(attr)
            
            if len(numeric_cols) >= 2:
                corr_matrix = self.df[numeric_cols].corr()
                im = axes[chart_idx].imshow(corr_matrix, cmap='coolwarm', aspect='auto', vmin=-1, vmax=1)
                axes[chart_idx].set_xticks(range(len(numeric_cols)))
                axes[chart_idx].set_yticks(range(len(numeric_cols)))
                axes[chart_idx].set_xticklabels(numeric_cols, rotation=45)
                axes[chart_idx].set_yticklabels(numeric_cols)
                axes[chart_idx].set_title('Correlation Matrix')
                
                # Add colorbar
                plt.colorbar(im, ax=axes[chart_idx])
                chart_idx += 1
        
        # Hide unused subplots
        for i in range(chart_idx, len(axes)):
            axes[i].set_visible(False)
        
        plt.tight_layout()
        
        # Save dashboard
        if output_path is None:
            output_path = Config.OUTPUT_DIR / 'csv_analysis_dashboard.png'
        
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"Dashboard saved to: {output_path}")
        return str(output_path)
    
    def print_summary(self):
        """Print a summary of key findings."""
        if not self.analysis_results:
            print("No analysis results available. Run run_full_analysis() first.")
            return
        
        print("\n" + "=" * 60)
        print("ANALYSIS SUMMARY")
        print("=" * 60)
        
        # Overview
        if 'overview' in self.analysis_results:
            print("\nOVERVIEW:")
            for measure, stats in self.analysis_results['overview'].items():
                print(f"  {measure}: ${stats['total']:,.2f} total, ${stats['mean']:,.2f} avg")
        
        # Key insights
        if 'key_insights' in self.analysis_results:
            print("\nKEY INSIGHTS:")
            for insight in self.analysis_results['key_insights']:
                print(f"  • {insight}")
        
        # Top dimensions
        if 'dimension_analysis' in self.analysis_results:
            print("\nTOP PERFORMING DIMENSIONS:")
            for dimension, measures in self.analysis_results['dimension_analysis'].items():
                if measures:
                    first_measure = list(measures.keys())[0]
                    top_category = max(measures[first_measure].items(), key=lambda x: x[1]['total'])
                    print(f"  {dimension}: {top_category[0]} (${top_category[1]['total']:,.2f})")
        
        print("=" * 60)
