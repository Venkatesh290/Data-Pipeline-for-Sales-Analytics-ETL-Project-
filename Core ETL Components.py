# config/config.py
import os
from dotenv import load_dotenv

load_dotenv()

class Config:
    # Database Configuration
    DATABASE_URL = os.getenv('DATABASE_URL', 'postgresql://localhost:5432/sales_analytics')
    POSTGRES_USER = os.getenv('POSTGRES_USER', 'sales_user')
    POSTGRES_PASSWORD = os.getenv('POSTGRES_PASSWORD', 'password')
    POSTGRES_DB = os.getenv('POSTGRES_DB', 'sales_analytics')
    
    # API Configuration
    SALESFORCE_USERNAME = os.getenv('SALESFORCE_USERNAME')
    SALESFORCE_PASSWORD = os.getenv('SALESFORCE_PASSWORD')
    SALESFORCE_SECURITY_TOKEN = os.getenv('SALESFORCE_SECURITY_TOKEN')
    
    SHOPIFY_API_KEY = os.getenv('SHOPIFY_API_KEY')
    SHOPIFY_PASSWORD = os.getenv('SHOPIFY_PASSWORD')
    SHOPIFY_SHOP_NAME = os.getenv('SHOPIFY_SHOP_NAME')
    
    # AWS Configuration
    AWS_ACCESS_KEY_ID = os.getenv('AWS_ACCESS_KEY_ID')
    AWS_SECRET_ACCESS_KEY = os.getenv('AWS_SECRET_ACCESS_KEY')
    AWS_REGION = os.getenv('AWS_REGION', 'us-east-1')
    S3_BUCKET = os.getenv('S3_BUCKET')
    
    # Redis
    REDIS_URL = os.getenv('REDIS_URL', 'redis://localhost:6379/0')
    
    # Pipeline Configuration
    ETL_BATCH_SIZE = int(os.getenv('ETL_BATCH_SIZE', 1000))
    DATA_RETENTION_DAYS = int(os.getenv('DATA_RETENTION_DAYS', 365))
    LOG_LEVEL = os.getenv('LOG_LEVEL', 'INFO')
    ENVIRONMENT = os.getenv('ENVIRONMENT', 'development')
    
    # Monitoring
    SLACK_WEBHOOK_URL = os.getenv('SLACK_WEBHOOK_URL')
    PROMETHEUS_PORT = int(os.getenv('PROMETHEUS_PORT', 9090))

# config/database.py
from sqlalchemy import create_engine
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
from config.config import Config

engine = create_engine(Config.DATABASE_URL)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

# config/logging_config.py
import logging
import structlog
from config.config import Config

def configure_logging():
    logging.basicConfig(
        level=getattr(logging, Config.LOG_LEVEL),
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    
    structlog.configure(
        processors=[
            structlog.stdlib.add_log_level,
            structlog.stdlib.add_logger_name,
            structlog.stdlib.PositionalArgumentsFormatter(),
            structlog.processors.TimeStamper(fmt="iso"),
            structlog.processors.StackInfoRenderer(),
            structlog.processors.format_exc_info,
            structlog.processors.JSONRenderer()
        ],
        context_class=dict,
        logger_factory=structlog.stdlib.LoggerFactory(),
        wrapper_class=structlog.stdlib.BoundLogger,
        cache_logger_on_first_use=True,
    )

# src/extractors/base_extractor.py
from abc import ABC, abstractmethod
import pandas as pd
import logging
from typing import Dict, Any, Optional
from datetime import datetime

logger = logging.getLogger(__name__)

class BaseExtractor(ABC):
    """Base class for all data extractors"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.last_extracted = None
        
    @abstractmethod
    def extract(self, start_date: Optional[datetime] = None, 
                end_date: Optional[datetime] = None) -> pd.DataFrame:
        """Extract data from the source"""
        pass
    
    @abstractmethod
    def test_connection(self) -> bool:
        """Test connection to the data source"""
        pass
    
    def get_extraction_metadata(self) -> Dict[str, Any]:
        """Get metadata about the last extraction"""
        return {
            'extractor': self.__class__.__name__,
            'last_extracted': self.last_extracted,
            'config': {k: v for k, v in self.config.items() if 'password' not in k.lower()}
        }

# src/extractors/salesforce_extractor.py
from simple_salesforce import Salesforce
import pandas as pd
from datetime import datetime, timedelta
from typing import Optional
from .base_extractor import BaseExtractor
from config.config import Config
import logging

logger = logging.getLogger(__name__)

class SalesforceExtractor(BaseExtractor):
    """Extract data from Salesforce CRM"""
    
    def __init__(self):
        super().__init__({
            'username': Config.SALESFORCE_USERNAME,
            'password': Config.SALESFORCE_PASSWORD,
            'security_token': Config.SALESFORCE_SECURITY_TOKEN
        })
        self.sf = None
        
    def test_connection(self) -> bool:
        """Test Salesforce connection"""
        try:
            if not self.sf:
                self.sf = Salesforce(
                    username=self.config['username'],
                    password=self.config['password'],
                    security_token=self.config['security_token']
                )
            # Test with a simple query
            self.sf.query("SELECT Id FROM Account LIMIT 1")
            logger.info("Salesforce connection test successful")
            return True
        except Exception as e:
            logger.error(f"Salesforce connection test failed: {e}")
            return False
    
    def extract(self, start_date: Optional[datetime] = None, 
                end_date: Optional[datetime] = None) -> pd.DataFrame:
        """Extract opportunities data from Salesforce"""
        try:
            if not self.sf:
                self.sf = Salesforce(
                    username=self.config['username'],
                    password=self.config['password'],
                    security_token=self.config['security_token']
                )
            
            # Default to last 30 days if no dates provided
            if not start_date:
                start_date = datetime.now() - timedelta(days=30)
            if not end_date:
                end_date = datetime.now()
            
            # SOQL query for opportunities
            query = f"""
                SELECT Id, AccountId, Name, StageName, Amount, 
                       CloseDate, CreatedDate, LastModifiedDate,
                       Probability, Type, LeadSource, OwnerId
                FROM Opportunity 
                WHERE LastModifiedDate >= {start_date.strftime('%Y-%m-%dT%H:%M:%S.000+0000')}
                AND LastModifiedDate <= {end_date.strftime('%Y-%m-%dT%H:%M:%S.000+0000')}
            """
            
            result = self.sf.query_all(query)
            records = result['records']
            
            # Convert to DataFrame
            df = pd.DataFrame([
                {
                    'opportunity_id': record['Id'],
                    'account_id': record['AccountId'],
                    'opportunity_name': record['Name'],
                    'stage_name': record['StageName'],
                    'amount': record['Amount'],
                    'close_date': record['CloseDate'],
                    'created_date': record['CreatedDate'],
                    'last_modified_date': record['LastModifiedDate'],
                    'probability': record['Probability'],
                    'opportunity_type': record['Type'],
                    'lead_source': record['LeadSource'],
                    'owner_id': record['OwnerId']
                }
                for record in records
            ])
            
            # Convert date strings to datetime
            date_columns = ['close_date', 'created_date', 'last_modified_date']
            for col in date_columns:
                df[col] = pd.to_datetime(df[col])
            
            self.last_extracted = datetime.now()
            logger.info(f"Extracted {len(df)} opportunities from Salesforce")
            
            return df
            
        except Exception as e:
            logger.error(f"Error extracting from Salesforce: {e}")
            raise

# src/extractors/shopify_extractor.py
import shopify
import pandas as pd
from datetime import datetime, timedelta
from typing import Optional
from .base_extractor import BaseExtractor
from config.config import Config
import logging

logger = logging.getLogger(__name__)

class ShopifyExtractor(BaseExtractor):
    """Extract data from Shopify e-commerce platform"""
    
    def __init__(self):
        super().__init__({
            'api_key': Config.SHOPIFY_API_KEY,
            'password': Config.SHOPIFY_PASSWORD,
            'shop_name': Config.SHOPIFY_SHOP_NAME
        })
        
    def test_connection(self) -> bool:
        """Test Shopify connection"""
        try:
            shop_url = f"https://{self.config['api_key']}:{self.config['password']}@{self.config['shop_name']}.myshopify.com/admin"
            shopify.ShopifyResource.set_site(shop_url)
            
            # Test with a simple request
            shop = shopify.Shop.current()
            logger.info(f"Shopify connection test successful for shop: {shop.name}")
            return True
        except Exception as e:
            logger.error(f"Shopify connection test failed: {e}")
            return False
    
    def extract(self, start_date: Optional[datetime] = None, 
                end_date: Optional[datetime] = None) -> pd.DataFrame:
        """Extract orders data from Shopify"""
        try:
            shop_url = f"https://{self.config['api_key']}:{self.config['password']}@{self.config['shop_name']}.myshopify.com/admin"
            shopify.ShopifyResource.set_site(shop_url)
            
            # Default to last 30 days if no dates provided
            if not start_date:
                start_date = datetime.now() - timedelta(days=30)
            if not end_date:
                end_date = datetime.now()
            
            # Get orders
            orders = shopify.Order.find(
                status='any',
                created_at_min=start_date.isoformat(),
                created_at_max=end_date.isoformat(),
                limit=250
            )
            
            order_data = []
            for order in orders:
                order_dict = {
                    'order_id': order.id,
                    'order_number': order.order_number,
                    'created_at': order.created_at,
                    'updated_at': order.updated_at,
                    'total_price': float(order.total_price),
                    'subtotal_price': float(order.subtotal_price),
                    'total_tax': float(order.total_tax),
                    'currency': order.currency,
                    'financial_status': order.financial_status,
                    'fulfillment_status': order.fulfillment_status,
                    'customer_id': order.customer.id if order.customer else None,
                    'customer_email': order.email,
                    'line_items_count': len(order.line_items),
                    'source_name': order.source_name
                }
                
                # Add line items details
                for i, line_item in enumerate(order.line_items):
                    item_dict = order_dict.copy()
                    item_dict.update({
                        'line_item_id': line_item.id,
                        'product_id': line_item.product_id,
                        'variant_id': line_item.variant_id,
                        'product_title': line_item.title,
                        'variant_title': line_item.variant_title,
                        'quantity': line_item.quantity,
                        'price': float(line_item.price),
                        'line_total': float(line_item.price) * line_item.quantity
                    })
                    order_data.append(item_dict)
            
            df = pd.DataFrame(order_data)
            
            # Convert datetime columns
            if not df.empty:
                df['created_at'] = pd.to_datetime(df['created_at'])
                df['updated_at'] = pd.to_datetime(df['updated_at'])
            
            self.last_extracted = datetime.now()
            logger.info(f"Extracted {len(df)} order line items from Shopify")
            
            return df
            
        except Exception as e:
            logger.error(f"Error extracting from Shopify: {e}")
            raise

# src/transformers/base_transformer.py
from abc import ABC, abstractmethod
import pandas as pd
import logging
from typing import Dict, Any

logger = logging.getLogger(__name__)

class BaseTransformer(ABC):
    """Base class for all data transformers"""
    
    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or {}
        
    @abstractmethod
    def transform(self, data: pd.DataFrame) -> pd.DataFrame:
        """Transform the input data"""
        pass
    
    def validate_input(self, data: pd.DataFrame) -> bool:
        """Validate input data before transformation"""
        if data.empty:
            logger.warning("Input data is empty")
            return False
        return True

# src/transformers/data_cleaner.py
import pandas as pd
import numpy as np
from typing import List, Dict, Any
from .base_transformer import BaseTransformer
import logging

logger = logging.getLogger(__name__)

class DataCleaner(BaseTransformer):
    """Clean and standardize data"""
    
    def __init__(self, config: Dict[str, Any] = None):
        super().__init__(config)
        self.cleaning_rules = self.config.get('cleaning_rules', {})
    
    def transform(self, data: pd.DataFrame) -> pd.DataFrame:
        """Apply data cleaning transformations"""
        if not self.validate_input(data):
            return data
        
        df = data.copy()
        
        # Remove duplicates
        df = self._remove_duplicates(df)
        
        # Handle missing values
        df = self._handle_missing_values(df)
        
        # Standardize formats
        df = self._standardize_formats(df)
        
        # Clean text fields
        df = self._clean_text_fields(df)
        
        # Remove outliers
        df = self._remove_outliers(df)
        
        logger.info(f"Data cleaning completed. Records: {len(data)} -> {len(df)}")
        
        return df
    
    def _remove_duplicates(self, df: pd.DataFrame) -> pd.DataFrame:
        """Remove duplicate records"""
        initial_count = len(df)
        df = df.drop_duplicates()
        final_count = len(df)
        
        if initial_count > final_count:
            logger.info(f"Removed {initial_count - final_count} duplicate records")
        
        return df
    
    def _handle_missing_values(self, df: pd.DataFrame) -> pd.DataFrame:
        """Handle missing values based on column types"""
        for column in df.columns:
            if df[column].dtype in ['object', 'string']:
                # Fill string columns with 'Unknown'
                df[column] = df[column].fillna('Unknown')
            elif df[column].dtype in ['float64', 'int64']:
                # Fill numeric columns with 0 or median based on config
                if self.cleaning_rules.get(f'{column}_fill_method') == 'median':
                    df[column] = df[column].fillna(df[column].median())
                else:
                    df[column] = df[column].fillna(0)
            elif 'datetime' in str(df[column].dtype):
                # Handle datetime columns
                df[column] = pd.to_datetime(df[column], errors='coerce')
        
        return df
    
    def _standardize_formats(self, df: pd.DataFrame) -> pd.DataFrame:
        """Standardize data formats"""
        # Standardize email addresses
        email_columns = [col for col in df.columns if 'email' in col.lower()]
        for col in email_columns:
            df[col] = df[col].str.lower().str.strip()
        
        # Standardize currency columns
        currency_columns = [col for col in df.columns if any(keyword in col.lower() 
                          for keyword in ['amount', 'price', 'cost', 'revenue', 'value'])]
        for col in currency_columns:
            if df[col].dtype == 'object':
                # Remove currency symbols and convert to float
                df[col] = df[col].astype(str).str.replace(r'[$,]', '', regex=True)
                df[col] = pd.to_numeric(df[col], errors='coerce')
        
        # Standardize phone numbers (basic cleaning)
        phone_columns = [col for col in df.columns if 'phone' in col.lower()]
        for col in phone_columns:
            df[col] = df[col].astype(str).str.replace(r'[^\d]', '', regex=True)
        
        return df
    
    def _clean_text_fields(self, df: pd.DataFrame) -> pd.DataFrame:
        """Clean text fields"""
        text_columns = df.select_dtypes(include=['object']).columns
        
        for col in text_columns:
            # Strip whitespace
            df[col] = df[col].astype(str).str.strip()
            
            # Replace multiple spaces with single space
            df[col] = df[col].str.replace(r'\s+', ' ', regex=True)
            
            # Remove special characters if specified in config
            if self.cleaning_rules.get(f'{col}_remove_special_chars', False):
                df[col] = df[col].str.replace(r'[^\w\s]', '', regex=True)
        
        return df
    
    def _remove_outliers(self, df: pd.DataFrame) -> pd.DataFrame:
        """Remove statistical outliers from numeric columns"""
        numeric_columns = df.select_dtypes(include=[np.number]).columns
        
        for col in numeric_columns:
            if self.cleaning_rules.get(f'{col}_remove_outliers', False):
                Q1 = df[col].quantile(0.25)
                Q3 = df[col].quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - 1.5 * IQR
                upper_bound = Q3 + 1.5 * IQR
                
                initial_count = len(df)
                df = df[(df[col] >= lower_bound) & (df[col] <= upper_bound)]
                final_count = len(df)
                
                if initial_count > final_count:
                    logger.info(f"Removed {initial_count - final_count} outliers from column {col}")
        
        return df

# src/transformers/data_validator.py
import pandas as pd
import great_expectations as gx
from typing import Dict, Any, List
from .base_transformer import BaseTransformer
import logging

logger = logging.getLogger(__name__)

class DataValidator(BaseTransformer):
    """Validate data quality using Great Expectations"""
    
    def __init__(self, config: Dict[str, Any] = None):
        super().__init__(config)
        self.context = gx.get_context()
        self.validation_rules = self.config.get('validation_rules', {})
    
    def transform(self, data: pd.DataFrame) -> pd.DataFrame:
        """Validate data and return clean data"""
        if not self.validate_input(data):
            return data
        
        # Create Great Expectations DataFrame
        df = self.context.sources.pandas_default.read_dataframe(data)
        
        # Create expectation suite
        suite = self._create_expectation_suite(data)
        
        # Run validation
        validation_result = df.validate(suite)
        
        # Log validation results
        self._log_validation_results(validation_result)
        
        # Filter out invalid records if configured to do so
        if self.config.get('filter_invalid_records', False):
            data = self._filter_invalid_records(data, validation_result)
        
        return data
    
    def _create_expectation_suite(self, data: pd.DataFrame) -> gx.ExpectationSuite:
        """Create expectation suite based on data and config"""
        suite_name = "sales_data_validation"
        suite = self.context.add_or_update_expectation_suite(expectation_suite_name=suite_name)
        
        # Basic expectations for all columns
        for column in data.columns:
            # Expect column to exist
            suite.add_expectation(
                gx.expectations.ExpectColumnToExist(column=column)
            )
            
            # Column-specific validations based on config
            if column in self.validation_rules:
                rules = self.validation_rules[column]
                
                # Not null validation
                if rules.get('not_null', False):
                    suite.add_expectation(
                        gx.expectations.ExpectColumnValuesToNotBeNull(column=column)
                    )
                
                # Value range validation
                if 'min_value' in rules:
                    suite.add_expectation(
                        gx.expectations.ExpectColumnValuesToBeGreaterThanOrEqualTo(
                            column=column, min_value=rules['min_value']
                        )
                    )
                
                if 'max_value' in rules:
                    suite.add_expectation(
                        gx.expectations.ExpectColumnValuesToBeLessThanOrEqualTo(
                            column=column, max_value=rules['max_value']
                        )
                    )
                
                # Value set validation
                if 'allowed_values' in rules:
                    suite.add_expectation(
                        gx.expectations.ExpectColumnValuesToBeInSet(
                            column=column, value_set=rules['allowed_values']
                        )
                    )
                
                # Regex pattern validation
                if 'regex_pattern' in rules:
                    suite.add_expectation(
                        gx.expectations.ExpectColumnValuesToMatchRegex(
                            column=column, regex=rules['regex_pattern']
                        )
                    )
        
        return suite
    
    def _log_validation_results(self, validation_result):
        """Log validation results"""
        if validation_result.success:
            logger.info("Data validation passed")
        else:
            logger.warning("Data validation failed")
            for result in validation_result.results:
                if not result.success:
                    logger.warning(f"Validation failed: {result.expectation_config.expectation_type}")
    
    def _filter_invalid_records(self, data: pd.DataFrame, validation_result) -> pd.DataFrame:
        """Filter out records that failed validation"""
        # This is a simplified implementation
        # In practice, you would need more sophisticated logic to identify and remove specific invalid records
        return data

# src/transformers/data_aggregator.py
import pandas as pd
from typing import Dict, Any, List
from .base_transformer import BaseTransformer
import logging

logger = logging.getLogger(__name__)

class DataAggregator(BaseTransformer):
    """Aggregate data for analytics"""
    
    def __init__(self, config: Dict[str, Any] = None):
        super().__init__(config)
        self.aggregation_rules = self.config.get('aggregation_rules', {})
    
    def transform(self, data: pd.DataFrame) -> Dict[str, pd.DataFrame]:
        """Create multiple aggregated views of the data"""
        if not self.validate_input(data):
            return {'raw': data}
        
        aggregated_data = {}
        
        # Daily sales summary
        if 'daily_sales' in self.aggregation_rules:
            aggregated_data['daily_sales'] = self._create_daily_sales_summary(data)
        
        # Product performance
        if 'product_performance' in self.aggregation_rules:
            aggregated_data['product_performance'] = self._create_product_performance_summary(data)
        
        # Customer analytics
        if 'customer_analytics' in self.aggregation_rules:
            aggregated_data['customer_analytics'] = self._create_customer_analytics_summary(data)
        
        # Sales rep performance
        if 'sales_rep_performance' in self.aggregation_rules:
            aggregated_data['sales_rep_performance'] = self._create_sales_rep_summary(data)
        
        logger.info(f"Created {len(aggregated_data)} aggregated datasets")
        
        return aggregated_data
    
    def _create_daily_sales_summary(self, data: pd.DataFrame) -> pd.DataFrame:
        """Create daily sales summary"""
        date_column = self._find_date_column(data, 'sale')
        amount_columns = self._find_amount_columns(data)
        
        if not date_column or not amount_columns:
            logger.warning("Could not create daily sales summary - missing required columns")
            return pd.DataFrame()
        
        # Extract date from datetime
        data['sale_date'] = pd.to_datetime(data[date_column]).dt.date
        
        # Aggregate by date
        daily_summary = data.groupby('sale_date').agg({
            amount_columns[0]: ['sum', 'mean', 'count'],
            'customer_id': 'nunique' if 'customer_id' in data.columns else lambda x: 0
        }).round(2)
        
        # Flatten column names
        daily_summary.columns = ['total_revenue', 'avg_order_value', 'transaction_count', 'unique_customers']
        daily_summary = daily_summary.reset_index()
        
        return daily_summary
    
    def _create_product_performance_summary(self, data: pd.DataFrame) -> pd.DataFrame:
        """Create product performance summary"""
        product_columns = [col for col in data.columns if 'product' in col.lower()]
        amount_columns = self._find_amount_columns(data)
        
        if not product_columns or not amount_columns:
            logger.warning("Could not create product performance summary - missing required columns")
            return pd.DataFrame()
        
        product_col = product_columns[0]  # Use first product column
        amount_col = amount_columns[0]    # Use first amount column
        
        product_summary = data.groupby(product_col).agg({
            amount_col: ['sum', 'mean', 'count'],
            'quantity': 'sum' if 'quantity' in data.columns else lambda x: 0
        }).round(2)
        
        # Flatten column names
        product_summary.columns = ['total_revenue', 'avg_price', 'transaction_count', 'total_quantity']
        product_summary = product_summary.reset_index()
        
        # Calculate revenue percentage
        product_summary['revenue_percentage'] = (
            product_summary['total_revenue'] / product_summary['total_revenue'].sum() * 100
        ).round(2)
        
        return product_summary.sort_values('total_revenue', ascending=False)
    
    def _create_customer_analytics_summary(self, data: pd.DataFrame) -> pd.DataFrame:
        """Create customer analytics summary"""
        if 'customer_id' not in data.columns:
            logger.warning("Could not create customer analytics - missing customer_id column")
            return pd.DataFrame()
        
        amount_columns = self._find_amount_columns(data)
        if not amount_columns:
            logger.warning("Could not create customer analytics - missing amount columns")
            return pd.DataFrame()
        
        amount_col = amount_columns[0]
        
        customer_summary = data.groupby('customer_id').agg({
            amount_col: ['sum', 'mean', 'count'],
            'created_at': ['min', 'max'] if 'created_at' in data.columns else lambda x: None
        }).round(2)
        
        # Flatten column names
        customer_summary.columns = ['total_spent', 'avg_order_value', 'order_count', 'first_purchase', 'last_purchase']
        customer_summary = customer_summary.reset_index()
        
        # Calculate customer lifetime value (simplified)
        customer_summary['customer_lifetime_value'] = customer_summary['total_spent']
        
        # Customer segmentation (simplified)
        customer_summary['customer_segment'] = pd.cut(
            customer_summary['total_spent'],
            bins=[0, 100, 500, 1000, float('inf')],
            labels=['Low Value', 'Medium Value', 'High Value', 'VIP']
        )
        
        return customer_summary.sort_values('total_spent', ascending=False)
    
    def _create_sales_rep_summary(self, data: pd.DataFrame) -> pd.DataFrame:
        """Create sales rep performance summary"""
        rep_columns = [col for col in data.columns if any(keyword in col.lower() 
                      for keyword in ['owner', 'rep', 'sales'])]
        amount_columns = self._find_amount_columns(data)
        
        if not rep_columns or not amount_columns:
            logger.warning("Could not create sales rep summary - missing required columns")
            return pd.DataFrame()
        
        rep_col = rep_columns[0]
        amount_col = amount_columns[0]
        
        rep_summary = data.groupby(rep_col).agg({
            amount_col: ['sum', 'mean', 'count'],
            'customer_id': 'nunique' if 'customer_id' in data.columns else lambda x: 0
        }).round(2)
        
        # Flatten column names
        rep_summary.columns = ['total_revenue', 'avg_deal_size', 'deal_count', 'unique_customers']
        rep_summary = rep_summary.reset_index()
        
        # Calculate performance metrics
        rep_summary['revenue_per_customer'] = (
            rep_summary['total_revenue'] / rep_summary['unique_customers']
        ).round(2)
        
        return rep_summary.sort_values('total_revenue', ascending=False)
    
    def _find_date_column(self, data: pd.DataFrame, context: str = '') -> str:
        """Find the most appropriate date column"""
        date_keywords = ['date', 'created', 'modified', 'updated', 'timestamp']
        if context:
            date_keywords.insert(0, context)
        
        for keyword in date_keywords:
            candidates = [col for col in data.columns if keyword in col.lower()]
            if candidates:
                return candidates[0]
        
        # Check for datetime columns
        datetime_columns = data.select_dtypes(include=['datetime64']).columns
        if len(datetime_columns) > 0:
            return datetime_columns[0]
        
        return None
    
    def _find_amount_columns(self, data: pd.DataFrame) -> List[str]:
        """Find columns that likely contain monetary amounts"""
        amount_keywords = ['amount', 'price', 'cost', 'revenue', 'value', 'total']
        amount_columns = []
        
        for keyword in amount_keywords:
            candidates = [col for col in data.columns if keyword in col.lower()]
            amount_columns.extend(candidates)
        
        # Also include numeric columns with currency-like patterns
        numeric_columns = data.select_dtypes(include=['number']).columns
        for col in numeric_columns:
            if col not in amount_columns and data[col].min() >= 0:
                # Simple heuristic: positive numeric values might be amounts
                amount_columns.append(col)
        
        return list(set(amount_columns))  # Remove duplicates

# src/loaders/base_loader.py
from abc import ABC, abstractmethod
import pandas as pd
import logging
from typing import Dict, Any

logger = logging.getLogger(__name__)

class BaseLoader(ABC):
    """Base class for all data loaders"""
    
    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or {}
        
    @abstractmethod
    def load(self, data: pd.DataFrame, table_name: str) -> bool:
        """Load data to the target destination"""
        pass
    
    @abstractmethod
    def test_connection(self) -> bool:
        """Test connection to the target destination"""
        pass
    
    def validate_data(self, data: pd.DataFrame) -> bool:
        """Validate data before loading"""
        if data.empty:
            logger.warning("No data to load")
            return False
        return True

# src/loaders/warehouse_loader.py
import pandas as pd
from sqlalchemy import create_engine, text
from typing import Dict, Any
from .base_loader import BaseLoader
from config.config import Config
import logging

logger = logging.getLogger(__name__)

class WarehouseLoader(BaseLoader):
    """Load data to data warehouse"""
    
    def __init__(self, config: Dict[str, Any] = None):
        super().__init__(config)
        self.engine = create_engine(Config.DATABASE_URL)
        
    def test_connection(self) -> bool:
        """Test database connection"""
        try:
            with self.engine.connect() as conn:
                conn.execute(text("SELECT 1"))
            logger.info("Database connection test successful")
            return True
        except Exception as e:
            logger.error(f"Database connection test failed: {e}")
            return False
    
    def load(self, data: pd.DataFrame, table_name: str) -> bool:
        """Load data to warehouse table"""
        if not self.validate_data(data):
            return False
        
        try:
            # Load data in chunks for better performance
            chunk_size = self.config.get('chunk_size', 1000)
            
            with self.engine.begin() as conn:
                # Determine if_exists behavior
                if_exists = self.config.get('if_exists', 'append')
                
                data.to_sql(
                    table_name,
                    conn,
                    if_exists=if_exists,
                    index=False,
                    chunksize=chunk_size,
                    method='multi'
                )
            
            logger.info(f"Successfully loaded {len(data)} records to {table_name}")
            return True
            
        except Exception as e:
            logger.error(f"Error loading data to {table_name}: {e}")
            return False
    
    def create_indexes(self, table_name: str, index_columns: list):
        """Create indexes on specified columns"""
        try:
            with self.engine.connect() as conn:
                for column in index_columns:
                    index_name = f"idx_{table_name}_{column}"
                    sql = f"CREATE INDEX IF NOT EXISTS {index_name} ON {table_name} ({column})"
                    conn.execute(text(sql))
            
            logger.info(f"Created indexes on {table_name}: {index_columns}")
            
        except Exception as e:
            logger.error(f"Error creating indexes on {table_name}: {e}")

# src/models/sales_models.py
from sqlalchemy import Column, Integer, String, Float, DateTime, Date, Boolean, ForeignKey
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import relationship
import datetime

Base = declarative_base()

class Customer(Base):
    __tablename__ = 'dim_customer'
    
    customer_id = Column(String, primary_key=True)
    customer_name = Column(String)
    email = Column(String)
    phone = Column(String)
    address = Column(String)
    city = Column(String)
    state = Column(String)
    country = Column(String)
    customer_segment = Column(String)
    created_date = Column(DateTime, default=datetime.datetime.utcnow)
    updated_date = Column(DateTime, default=datetime.datetime.utcnow, onupdate=datetime.datetime.utcnow)

class Product(Base):
    __tablename__ = 'dim_product'
    
    product_id = Column(String, primary_key=True)
    product_name = Column(String)
    category = Column(String)
    subcategory = Column(String)
    brand = Column(String)
    unit_price = Column(Float)
    cost = Column(Float)
    status = Column(String)
    created_date = Column(DateTime, default=datetime.datetime.utcnow)
    updated_date = Column(DateTime, default=datetime.datetime.utcnow, onupdate=datetime.datetime.utcnow)

class SalesRep(Base):
    __tablename__ = 'dim_sales_rep'
    
    rep_id = Column(String, primary_key=True)
    rep_name = Column(String)
    email = Column(String)
    department = Column(String)
    region = Column(String)
    hire_date = Column(Date)
    status = Column(String)

class DateDimension(Base):
    __tablename__ = 'dim_date'
    
    date_key = Column(Integer, primary_key=True)
    date_value = Column(Date)
    year = Column(Integer)
    quarter = Column(Integer)
    month = Column(Integer)
    month_name = Column(String)
    week = Column(Integer)
    day_of_month = Column(Integer)
    day_of_week = Column(Integer)
    day_name = Column(String)
    is_weekend = Column(Boolean)
    is_holiday = Column(Boolean)

class SalesFact(Base):
    __tablename__ = 'fact_sales'
    
    transaction_id = Column(String, primary_key=True)
    customer_id = Column(String, ForeignKey('dim_customer.customer_id'))
    product_id = Column(String, ForeignKey('dim_product.product_id'))
    rep_id = Column(String, ForeignKey('dim_sales_rep.rep_id'))
    sale_date = Column(Date)
    quantity = Column(Integer)
    unit_price = Column(Float)
    total_amount = Column(Float)
    discount = Column(Float)
    tax = Column(Float)
    profit_margin = Column(Float)
    source = Column(String)  # 'salesforce', 'shopify', 'pos', etc.
    created_date = Column(DateTime, default=datetime.datetime.utcnow)
    
    # Relationships
    customer = relationship("Customer")
    product = relationship("Product")
    sales_rep = relationship("SalesRep")

# src/utils/database_utils.py
import pandas as pd
from sqlalchemy import create_engine, text, MetaData, Table
from sqlalchemy.orm import sessionmaker
from config.config import Config
import logging

logger = logging.getLogger(__name__)

class DatabaseUtils:
    """Utility functions for database operations"""
    
    def __init__(self):
        self.engine = create_engine(Config.DATABASE_URL)
        self.SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=self.engine)
    
    def execute_sql_file(self, file_path: str):
        """Execute SQL commands from a file"""
        try:
            with open(file_path, 'r') as file:
                sql_commands = file.read()
            
            with self.engine.begin() as conn:
                # Split by semicolon and execute each command
                for command in sql_commands.split(';'):
                    command = command.strip()
                    if command:
                        conn.execute(text(command))
            
            logger.info(f"Successfully executed SQL file: {file_path}")
            
        except Exception as e:
            logger.error(f"Error executing SQL file {file_path}: {e}")
            raise
    
    def table_exists(self, table_name: str) -> bool:
        """Check if table exists"""
        try:
            metadata = MetaData()
            metadata.reflect(bind=self.engine)
            return table_name in metadata.tables
        except Exception as e:
            logger.error(f"Error checking table existence: {e}")
            return False
    
    def get_table_row_count(self, table_name: str) -> int:
        """Get row count for a table"""
        try:
            with self.engine.connect() as conn:
                result = conn.execute(text(f"SELECT COUNT(*) FROM {table_name}"))
                return result.scalar()
        except Exception as e:
            logger.error(f"Error getting row count for {table_name}: {e}")
            return 0
    
    def truncate_table(self, table_name: str):
        """Truncate a table"""
        try:
            with self.engine.begin() as conn:
                conn.execute(text(f"TRUNCATE TABLE {table_name}"))
            logger.info(f"Truncated table: {table_name}")
        except Exception as e:
            logger.error(f"Error truncating table {table_name}: {e}")
            raise
    
    def backup_table(self, table_name: str, backup_suffix: str = None):
        """Create a backup of a table"""
        try:
            if not backup_suffix:
                backup_suffix = pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")
            
            backup_table_name = f"{table_name}_backup_{backup_suffix}"
            
            with self.engine.begin() as conn:
                conn.execute(text(f"CREATE TABLE {backup_table_name} AS SELECT * FROM {table_name}"))
            
            logger.info(f"Created backup table: {backup_table_name}")
            return backup_table_name
            
        except Exception as e:
            logger.error(f"Error backing up table {table_name}: {e}")
            raise

# Continue with more files...
