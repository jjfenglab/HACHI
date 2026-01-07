"""
Configuration class for keyphrase extraction
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional, Union

import yaml


@dataclass
class KeyphraseConfig:
    """Configuration for keyphrase extraction
    
    This configuration can be loaded from YAML files or created from argparse.
    All file paths are converted to absolute paths automatically.
    """
    llm_model_type: str
    prompt_file: str
    cache_file: str
    seed: int
    batch_size: int
    num_new_tokens: int
    prompt_placeholder: str
    text_column: str
    is_image: bool
    index_col: Optional[int]
    llm_outputs_file: Optional[str]
    log_file: str
    
    def __post_init__(self) -> None:
        """Validate and normalize configuration after initialization"""
        if not self.llm_model_type:
            raise ValueError("llm_model_type is required")
        if not self.prompt_file:
            raise ValueError("prompt_file is required")
        if self.batch_size <= 0:
            raise ValueError("batch_size must be positive")
        if self.num_new_tokens <= 0:
            raise ValueError("num_new_tokens must be positive")
        if self.seed < 0:
            raise ValueError("seed must be non-negative")
        
    @classmethod
    def from_yaml(cls, yaml_path: Union[str, Path]) -> "KeyphraseConfig":
        """Load configuration from a YAML file
        
        Args:
            yaml_path: Path to YAML configuration file
            
        Returns:
            KeyphraseConfig instance
        """
        yaml_path = Path(yaml_path)
        if not yaml_path.exists():
            raise FileNotFoundError(f"Config file not found: {yaml_path}")
        
        with open(yaml_path, 'r') as f:
            config_dict = yaml.safe_load(f)
        
        return cls(**config_dict)
    
    @classmethod
    def from_args(cls, args: Any) -> "KeyphraseConfig":
        """Create configuration from argparse arguments
        
        Args:
            args: argparse.Namespace object or any object with config attributes
            
        Returns:
            KeyphraseConfig instance
        """
        config_dict = {}
        for field_name in cls.__dataclass_fields__.keys():
            if hasattr(args, field_name):
                config_dict[field_name] = getattr(args, field_name)
        
        return cls(**config_dict)
