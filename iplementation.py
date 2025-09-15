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
        email_columns
