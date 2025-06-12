#!/usr/bin/env python3

import os
from typing import Dict

def validate_environment_variables() -> Dict[str, bool]:
    """
    Validate that required environment variables are set.
    
    Returns:
        Dict[str, bool]: Status of each required environment variable
    """
    required_vars = {
        'LANGDB_PROJECT_ID': os.getenv('LANGDB_PROJECT_ID') is not None,
        'LANGDB_API_KEY': os.getenv('LANGDB_API_KEY') is not None,
        'LANGDB_BASE_URL':  os.getenv('LANGDB_BASE_URL') is not None,
    }
    
    return required_vars