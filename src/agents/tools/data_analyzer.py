"""
Data analysis tool for agents.
This module provides functionality to analyze various types of data.
"""

import os
import json
import tempfile
import asyncio
from typing import Dict, Any, List, Optional, Union

import pandas as pd
import numpy as np
from loguru import logger

from src.core.config import settings


async def analyze_data(data_source: Union[str, Dict, List], 
                      query: Optional[str] = None,
                      analysis_type: Optional[str] = None) -> Dict[str, Any]:
    """
    Analyze data from various sources.
    
    Args:
        data_source: Path to data file, data dictionary, or list
        query: Analysis query or question
        analysis_type: Type of analysis to perform (descriptive, correlation, etc.)
        
    Returns:
        Analysis results
    """
    try:
        # Load the data
        data = await _load_data(data_source)
        
        # Determine analysis type if not specified
        if not analysis_type:
            analysis_type = _determine_analysis_type(query, data)
        
        # Perform the appropriate analysis
        if analysis_type == "descriptive":
            return await _descriptive_analysis(data)
        elif analysis_type == "correlation":
            return await _correlation_analysis(data)
        elif analysis_type == "time_series":
            return await _time_series_analysis(data)
        elif analysis_type == "cluster":
            return await _cluster_analysis(data)
        else:
            # Default to descriptive analysis
            return await _descriptive_analysis(data)
            
    except Exception as e:
        logger.error(f"Error analyzing data: {e}")
        return {"error": str(e)}


async def _load_data(data_source: Union[str, Dict, List]) -> pd.DataFrame:
    """
    Load data from various sources into a pandas DataFrame.
    
    Args:
        data_source: Path to data file, data dictionary, or list
        
    Returns:
        Pandas DataFrame
    """
    if isinstance(data_source, str):
        # Handle file path
        if data_source.endswith('.csv'):
            return pd.read_csv(data_source)
        elif data_source.endswith('.json'):
            return pd.read_json(data_source)
        elif data_source.endswith(('.xls', '.xlsx')):
            return pd.read_excel(data_source)
        elif data_source.endswith('.parquet'):
            return pd.read_parquet(data_source)
        else:
            raise ValueError(f"Unsupported file type: {data_source}")
    elif isinstance(data_source, dict):
        # Handle dictionary
        return pd.DataFrame.from_dict(data_source)
    elif isinstance(data_source, list):
        # Handle list of dictionaries or list of lists
        if len(data_source) > 0:
            if isinstance(data_source[0], dict):
                return pd.DataFrame(data_source)
            else:
                return pd.DataFrame(data_source)
        else:
            return pd.DataFrame()
    else:
        raise ValueError(f"Unsupported data source type: {type(data_source)}")


def _determine_analysis_type(query: Optional[str], data: pd.DataFrame) -> str:
    """
    Determine the appropriate analysis type based on the query and data.
    
    Args:
        query: Analysis query or question
        data: Pandas DataFrame
        
    Returns:
        Analysis type
    """
    if not query:
        return "descriptive"
    
    query_lower = query.lower()
    
    # Check for correlation-related keywords
    if any(keyword in query_lower for keyword in ["correlation", "relationship", "related", "associate"]):
        return "correlation"
    
    # Check for time series-related keywords
    date_columns = data.select_dtypes(include=['datetime64']).columns
    if (len(date_columns) > 0 or 
        any(col.lower() in ["date", "time", "year", "month", "day"] for col in data.columns) or
        any(keyword in query_lower for keyword in ["time", "trend", "series", "forecast", "predict"])):
        return "time_series"
    
    # Check for clustering-related keywords
    if any(keyword in query_lower for keyword in ["cluster", "group", "segment", "category"]):
        return "cluster"
    
    # Default to descriptive analysis
    return "descriptive"


async def _descriptive_analysis(data: pd.DataFrame) -> Dict[str, Any]:
    """
    Perform descriptive analysis on data.
    
    Args:
        data: Pandas DataFrame
        
    Returns:
        Analysis results
    """
    results = {
        "analysis_type": "descriptive",
        "shape": {
            "rows": data.shape[0],
            "columns": data.shape[1]
        },
        "columns": list(data.columns),
        "dtypes": {col: str(dtype) for col, dtype in zip(data.columns, data.dtypes)},
        "summary": {}
    }
    
    # Check for missing values
    missing_values = data.isnull().sum().to_dict()
    results["missing_values"] = {col: count for col, count in missing_values.items() if count > 0}
    
    # Generate statistical summary
    numeric_data = data.select_dtypes(include=['number'])
    if not numeric_data.empty:
        # Calculate basic statistics
        stats = numeric_data.describe().to_dict()
        results["summary"]["numeric"] = stats
        
        # Calculate additional statistics
        for col in numeric_data.columns:
            try:
                skew = numeric_data[col].skew()
                kurtosis = numeric_data[col].kurtosis()
                results["summary"]["numeric"][col]["skew"] = skew
                results["summary"]["numeric"][col]["kurtosis"] = kurtosis
            except:
                pass
    
    # Categorical data summary
    categorical_data = data.select_dtypes(exclude=['number', 'datetime64'])
    if not categorical_data.empty:
        cat_summary = {}
        for col in categorical_data.columns:
            try:
                value_counts = categorical_data[col].value_counts().head(10).to_dict()
                unique_count = categorical_data[col].nunique()
                cat_summary[col] = {
                    "unique_count": unique_count,
                    "top_values": value_counts
                }
            except:
                pass
        
        results["summary"]["categorical"] = cat_summary
    
    return results


async def _correlation_analysis(data: pd.DataFrame) -> Dict[str, Any]:
    """
    Perform correlation analysis on data.
    
    Args:
        data: Pandas DataFrame
        
    Returns:
        Analysis results
    """
    results = {
        "analysis_type": "correlation",
        "shape": {
            "rows": data.shape[0],
            "columns": data.shape[1]
        }
    }
    
    # Analyze numeric columns
    numeric_data = data.select_dtypes(include=['number'])
    if not numeric_data.empty:
        # Calculate correlation matrix
        corr_matrix = numeric_data.corr().round(4).to_dict()
        results["correlation_matrix"] = corr_matrix
        
        # Find strongest correlations
        corr_df = numeric_data.corr().abs().unstack().sort_values(ascending=False)
        corr_df = corr_df[corr_df < 1]  # Remove self-correlations
        top_correlations = []
        seen_pairs = set()
        
        for idx, corr_value in corr_df.head(10).items():
            col1, col2 = idx
            pair = tuple(sorted([col1, col2]))
            if pair not in seen_pairs:
                seen_pairs.add(pair)
                top_correlations.append({
                    "variables": [col1, col2],
                    "correlation": round(float(numeric_data[col1].corr(numeric_data[col2])), 4),
                    "abs_correlation": round(float(abs(numeric_data[col1].corr(numeric_data[col2]))), 4)
                })
        
        results["top_correlations"] = top_correlations
    
    # Analyze categorical relationships
    categorical_data = data.select_dtypes(exclude=['number', 'datetime64'])
    if not categorical_data.empty and not numeric_data.empty:
        # Analyze relationships between categorical and numeric variables
        cat_num_relationships = []
        
        for cat_col in categorical_data.columns:
            for num_col in numeric_data.columns:
                try:
                    # Group numeric data by categorical variable
                    grouped = data.groupby(cat_col)[num_col].agg(['mean', 'median', 'std', 'count'])
                    grouped = grouped.sort_values(by='count', ascending=False).head(10)
                    
                    # Convert to dictionary
                    grouped_dict = grouped.reset_index().to_dict(orient='records')
                    
                    cat_num_relationships.append({
                        "categorical_variable": cat_col,
                        "numeric_variable": num_col,
                        "groupwise_statistics": grouped_dict
                    })
                except:
                    pass
        
        results["categorical_numeric_relationships"] = cat_num_relationships
    
    return results


async def _time_series_analysis(data: pd.DataFrame) -> Dict[str, Any]:
    """
    Perform time series analysis on data.
    
    Args:
        data: Pandas DataFrame
        
    Returns:
        Analysis results
    """
    results = {
        "analysis_type": "time_series",
        "shape": {
            "rows": data.shape[0],
            "columns": data.shape[1]
        }
    }
    
    # Identify date columns
    date_columns = data.select_dtypes(include=['datetime64']).columns.tolist()
    
    # If no datetime columns, try to convert columns with date-like names
    if not date_columns:
        for col in data.columns:
            if col.lower() in ["date", "time", "datetime", "timestamp"]:
                try:
                    data[col] = pd.to_datetime(data[col])
                    date_columns.append(col)
                except:
                    pass
    
    # If still no date columns, can't do time series analysis
    if not date_columns:
        return {
            "analysis_type": "time_series",
            "error": "No date or datetime columns found in the data"
        }
    
    # Use the first date column as the time index
    date_col = date_columns[0]
    results["time_column"] = date_col
    
    # Ensure the data is sorted by date
    data = data.sort_values(by=date_col)
    
    # Analyze numeric columns over time
    numeric_data = data.select_dtypes(include=['number'])
    if not numeric_data.empty:
        time_series_results = {}
        
        for col in numeric_data.columns:
            try:
                # Get time aggregations
                daily_avg = data.resample('D', on=date_col)[col].mean().dropna().tail(30)
                weekly_avg = data.resample('W', on=date_col)[col].mean().dropna().tail(10)
                monthly_avg = data.resample('M', on=date_col)[col].mean().dropna().tail(12)
                
                # Convert to dictionaries
                time_series_results[col] = {
                    "daily": {str(k.date()): round(float(v), 4) for k, v in daily_avg.items()},
                    "weekly": {str(k.date()): round(float(v), 4) for k, v in weekly_avg.items()},
                    "monthly": {str(k.date()): round(float(v), 4) for k, v in monthly_avg.items()}
                }
                
                # Calculate summary statistics
                time_series_results[col]["statistics"] = {
                    "trend": "increasing" if daily_avg.is_monotonic_increasing else 
                              "decreasing" if daily_avg.is_monotonic_decreasing else 
                              "stable" if daily_avg.std() / daily_avg.mean() < 0.1 else "fluctuating",
                    "min": float(data[col].min()),
                    "max": float(data[col].max()),
                    "avg": float(data[col].mean()),
                    "latest": float(data[col].iloc[-1]) if not data[col].empty else None,
                    "change_rate": float((data[col].iloc[-1] - data[col].iloc[0]) / data[col].iloc[0]) 
                                    if not data[col].empty and data[col].iloc[0] != 0 else None
                }
                
            except Exception as e:
                logger.error(f"Error analyzing time series for column {col}: {e}")
        
        results["time_series"] = time_series_results
    
    return results


async def _cluster_analysis(data: pd.DataFrame) -> Dict[str, Any]:
    """
    Perform basic cluster analysis on data.
    
    Args:
        data: Pandas DataFrame
        
    Returns:
        Analysis results
    """
    from sklearn.cluster import KMeans
    from sklearn.preprocessing import StandardScaler
    
    results = {
        "analysis_type": "cluster",
        "shape": {
            "rows": data.shape[0],
            "columns": data.shape[1]
        }
    }
    
    # Only use numeric columns
    numeric_data = data.select_dtypes(include=['number'])
    if numeric_data.empty or numeric_data.shape[1] < 2:
        return {
            "analysis_type": "cluster",
            "error": "Insufficient numeric data for clustering (need at least 2 numeric columns)"
        }
    
    # Drop rows with missing values
    numeric_data = numeric_data.dropna()
    if numeric_data.empty:
        return {
            "analysis_type": "cluster",
            "error": "No complete rows after removing missing values"
        }
    
    try:
        # Standardize the data
        scaler = StandardScaler()
        scaled_data = scaler.fit_transform(numeric_data)
        
        # Determine optimal number of clusters (simplified approach)
        max_clusters = min(10, numeric_data.shape[0] // 10)
        if max_clusters < 2:
            max_clusters = 2
            
        inertias = []
        for n_clusters in range(1, max_clusters + 1):
            kmeans = KMeans(n_clusters=n_clusters, random_state=42)
            kmeans.fit(scaled_data)
            inertias.append(kmeans.inertia_)
        
        # Find elbow point (simplified)
        optimal_clusters = 2  # Default
        for i in range(1, len(inertias) - 1):
            prev_diff = inertias[i-1] - inertias[i]
            next_diff = inertias[i] - inertias[i+1]
            if next_diff / prev_diff < 0.7:  # Rough elbow detection
                optimal_clusters = i + 1
                break
        
        # Perform clustering with optimal number of clusters
        kmeans = KMeans(n_clusters=optimal_clusters, random_state=42)
        clusters = kmeans.fit_predict(scaled_data)
        
        # Add cluster labels to original data
        cluster_df = numeric_data.copy()
        cluster_df['cluster'] = clusters
        
        # Get cluster statistics
        cluster_stats = []
        for i in range(optimal_clusters):
            cluster_data = cluster_df[cluster_df['cluster'] == i]
            cluster_size = len(cluster_data)
            cluster_means = cluster_data.mean().drop('cluster').to_dict()
            
            # Find most distinguishing features
            overall_means = numeric_data.mean().to_dict()
            differences = {col: (cluster_means[col] - overall_means[col]) / overall_means[col] 
                          if overall_means[col] != 0 else 0
                          for col in cluster_means}
            
            distinguishing_features = sorted(differences.items(), 
                                          key=lambda x: abs(x[1]), 
                                          reverse=True)[:3]
            
            cluster_stats.append({
                "cluster_id": i,
                "size": cluster_size,
                "percentage": round(cluster_size / len(numeric_data) * 100, 2),
                "means": {k: round(v, 4) for k, v in cluster_means.items()},
                "distinguishing_features": [
                    {
                        "feature": feature,
                        "difference_pct": round(diff * 100, 2),
                        "direction": "higher" if diff > 0 else "lower"
                    }
                    for feature, diff in distinguishing_features
                ]
            })
        
        results["optimal_clusters"] = optimal_clusters
        results["cluster_statistics"] = cluster_stats
        results["inertia_values"] = {i+1: round(val, 4) for i, val in enumerate(inertias)}
        
        return results
        
    except Exception as e:
        logger.error(f"Error performing cluster analysis: {e}")
        return {
            "analysis_type": "cluster",
            "error": f"Error during clustering: {str(e)}"
        }
