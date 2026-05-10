"""CSV schema definitions and validation."""

from typing import Dict, List, Union, Optional
from dataclasses import dataclass


@dataclass
class CSVSchema:
    """Schema definition for CSV analysis."""
    
    date_column: str
    measures: List[str]
    dimensions: List[str]
    customer_attributes: Optional[List[str]] = None
    satisfaction_column: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Union[str, List[str]]]:
        """Convert schema to dictionary format."""
        schema_dict = {
            'date_column': self.date_column,
            'measures': self.measures,
            'dimensions': self.dimensions
        }
        
        if self.customer_attributes:
            schema_dict['customer_attributes'] = self.customer_attributes
        
        if self.satisfaction_column:
            schema_dict['satisfaction_column'] = self.satisfaction_column
        
        return schema_dict
    
    @classmethod
    def from_dict(cls, schema_dict: Dict[str, Union[str, List[str]]]) -> 'CSVSchema':
        """Create schema from dictionary."""
        return cls(
            date_column=schema_dict['date_column'],
            measures=schema_dict['measures'],
            dimensions=schema_dict['dimensions'],
            customer_attributes=schema_dict.get('customer_attributes'),
            satisfaction_column=schema_dict.get('satisfaction_column')
        )
    
    def validate(self) -> List[str]:
        """Validate schema and return list of errors."""
        errors = []
        
        if not self.date_column:
            errors.append("date_column is required")
        
        if not self.measures or not isinstance(self.measures, list):
            errors.append("measures must be a non-empty list")
        
        if not self.dimensions or not isinstance(self.dimensions, list):
            errors.append("dimensions must be a non-empty list")
        
        return errors


def get_sales_schema() -> CSVSchema:
    """Get default schema for sales data analysis."""
    return CSVSchema(
        date_column='Date',
        measures=['Sales'],
        dimensions=['Product', 'Region', 'Customer_Gender'],
        customer_attributes=['Customer_Age', 'Customer_Gender'],
        satisfaction_column='Customer_Satisfaction'
    )


def get_financial_schema() -> CSVSchema:
    """Get schema for financial data analysis."""
    return CSVSchema(
        date_column='Transaction_Date',
        measures=['Revenue', 'Cost', 'Profit'],
        dimensions=['Department', 'Category', 'Region'],
        customer_attributes=['Customer_Segment', 'Account_Type'],
        satisfaction_column='Customer_Rating'
    )


def get_hr_schema() -> CSVSchema:
    """Get schema for HR data analysis."""
    return CSVSchema(
        date_column='Hire_Date',
        measures=['Salary', 'Performance_Score'],
        dimensions=['Department', 'Position', 'Location'],
        customer_attributes=['Age', 'Experience_Years'],
        satisfaction_column='Job_Satisfaction'
    )
