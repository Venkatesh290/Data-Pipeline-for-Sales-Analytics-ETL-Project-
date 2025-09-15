# src/utils/data_quality.py
import pandas as pd
import numpy as np
from typing import Dict, Any, List, Tuple
import logging

logger = logging.getLogger(__name__)

class DataQualityChecker:
    """Comprehensive data quality assessment"""
    
    def __init__(self):
        self.quality_metrics = {}
    
    def assess_data_quality(self, data: pd.DataFrame, table_name: str = "") -> Dict[str, Any]:
        """Perform comprehensive data quality assessment"""
        
        assessment = {
            'table_name': table_name,
            'timestamp': pd.Timestamp.now(),
            'row_count': len(data),
            'column_count': len(data.columns),
            'completeness': self._calculate_completeness(data),
            'uniqueness': self._calculate_uniqueness(data),
            'consistency': self._calculate_consistency(data),
            'validity': self._calculate_validity(data),
            'accuracy': self._calculate_accuracy(data),
            'overall_score': 0.0
        }
        
        # Calculate overall quality score
        scores = [
            assessment['completeness']['score'],
            assessment['uniqueness']['score'],
            assessment['consistency']['score'],
            assessment['validity']['score'],
            assessment['accuracy']['score']
        ]
        assessment['overall_score'] = np.mean([s for s in scores if s is not None])
        
        self.quality_metrics[table_name] = assessment
        
        logger.info(f"Data quality assessment completed for {table_name}. Score: {assessment['overall_score']:.2f}")
        
        return assessment
    
    def _calculate_completeness(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Calculate data completeness metrics"""
        missing_counts = data.isnull().sum()
        missing_percentages = (missing_counts / len(data) * 100).round(2)
        
        completeness_score = ((len(data) * len(data.columns) - missing_counts.sum()) / 
                             (len(data) * len(data.columns))) * 100
        
        return {
            'score': completeness_score,
            'missing_values_by_column': missing_counts.to_dict(),
            'missing_percentages_by_column': missing_percentages.to_dict(),
            'columns_with_missing_data': missing_counts[missing_counts > 0].index.tolist()
        }
    
    def _calculate_uniqueness(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Calculate data uniqueness metrics"""
        duplicate_rows = data.duplicated().sum()
        uniqueness_score = ((len(data) - duplicate_rows) / len(data)) * 100 if len(data) > 0 else 100
        
        # Check uniqueness for each column
        column_uniqueness = {}
        for col in data.columns:
            unique_count = data[col].nunique()
            total_count = len(data[col].dropna())
            if total_count > 0:
                column_uniqueness[col] = (unique_count / total_count) * 100
        
        return {
            'score': uniqueness_score,
            'duplicate_rows': int(duplicate_rows),
            'duplicate_percentage': (duplicate_rows / len(data) * 100) if len(data) > 0 else 0,
            'column_uniqueness_percentages': column_uniqueness
        }
    
    def _calculate_consistency(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Calculate data consistency metrics"""
        consistency_issues = []
        
        # Check for inconsistent data formats
        for col in data.select_dtypes(include=['object']).columns:
            # Check for mixed case issues
            if data[col].dtype == 'object':
                values = data[col].dropna().astype(str)
                if len(values) > 0:
                    lowercase_count = values.str.islower().sum()
                    uppercase_count = values.str.isupper().sum()
                    mixed_case = len(values) - lowercase_count - uppercase_count
                    
                    if mixed_case > 0:
                        consistency_issues.append(f"Column '{col}' has mixed case formatting")
        
        # Check for date format consistency
        date_columns = data.select_dtypes(include=['datetime64']).columns
        for col in date_columns:
            # This is a simplified check - in practice, you'd want more sophisticated date validation
            null_dates = data[col].isnull().sum()
            if null_dates > 0:
                consistency_issues.append(f"Column '{col}' has inconsistent date formats")
        
        consistency_score = 100 - (len(consistency_issues) / len(data.columns) * 100)
        
        return {
            'score': consistency_score,
            'issues': consistency_issues,
            'issue_count': len(consistency_issues)
        }
    
    def _calculate_validity(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Calculate data validity metrics"""
        validity_issues = []
        
        # Check for negative values in amount columns
        amount_keywords = ['amount', 'price', 'cost', 'revenue', 'value']
        for col in data.columns:
            if any(keyword in col.lower() for keyword in amount_keywords):
                if data[col].dtype in ['int64',
