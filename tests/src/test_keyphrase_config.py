"""
Unit tests for KeyphraseConfig
"""

import pytest
from pathlib import Path
from argparse import Namespace

import yaml

from src.keyphrase_config import KeyphraseConfig

class TestKeyphraseConfig:
    """Tests for KeyphraseConfig validation and factory methods"""
    
    # Test loading a valid YAML config file
    def test_load_valid_yaml(self, tmp_path):
        yaml_path = tmp_path / "config.yaml"
        config_data = {
            "llm_model_type": "gpt-4",
            "prompt_file": "prompts/test.txt",
            "cache_file": "cache.db",
            "seed": 42,
            "batch_size": 8,
            "num_new_tokens": 500,
            "prompt_placeholder": "{text}",
            "text_column": "content",
            "is_image": False,
            "index_col": 0,
            "llm_outputs_file": "results.csv",
            "log_file": "extraction.log"
        }
        
        with open(yaml_path, 'w') as f:
            yaml.dump(config_data, f)
        
        config = KeyphraseConfig.from_yaml(yaml_path)
        
        assert config.llm_model_type == "gpt-4"
        assert config.seed == 42
        assert config.batch_size == 8
        assert config.num_new_tokens == 500
        assert config.prompt_placeholder == "{text}"
        assert config.text_column == "content"
    
    # Test that loading from non-existent YAML file raises FileNotFoundError
    def test_load_nonexistent_yaml_raises_error(self):
        with pytest.raises(FileNotFoundError, match="Config file not found"):
            KeyphraseConfig.from_yaml("nonexistent_config.yaml")
    
    # Test loading YAML with image configuration
    def test_load_yaml_with_image_config(self, tmp_path):
        yaml_path = tmp_path / "config.yaml"
        config_data = {
            "llm_model_type": "gpt-4-vision",
            "prompt_file": "prompts/image.txt",
            "cache_file": "cache.db",
            "seed": 0,
            "batch_size": 2,
            "num_new_tokens": 300,
            "prompt_placeholder": "{image}",
            "text_column": "image_path",
            "is_image": True,
            "index_col": None,
            "llm_outputs_file": None,
            "log_file": "log.txt"
        }
        
        with open(yaml_path, 'w') as f:
            yaml.dump(config_data, f)
        
        config = KeyphraseConfig.from_yaml(yaml_path)
        
        assert config.is_image is True
        assert config.index_col is None
        assert config.llm_outputs_file is None
    
    # Test that YAML with empty llm_model_type triggers validation error
    def test_load_yaml_with_empty_llm_model_type(self, tmp_path):
        yaml_path = tmp_path / "config.yaml"
        config_data = {
            "llm_model_type": "",  # Invalid
            "prompt_file": "prompts/test.txt",
            "cache_file": "cache.db",
            "seed": 0,
            "batch_size": 4,
            "num_new_tokens": 300,
            "prompt_placeholder": "{note}",
            "text_column": "sentence",
            "is_image": False,
            "index_col": 0,
            "llm_outputs_file": None,
            "log_file": "log.txt"
        }
        
        with open(yaml_path, 'w') as f:
            yaml.dump(config_data, f)
        
        with pytest.raises(ValueError, match="llm_model_type is required"):
            KeyphraseConfig.from_yaml(yaml_path)
    
    # Test that YAML with empty prompt_file triggers validation error
    def test_load_yaml_with_empty_prompt_file(self, tmp_path):
        yaml_path = tmp_path / "config.yaml"
        config_data = {
            "llm_model_type": "gpt-4",
            "prompt_file": "",  # Invalid
            "cache_file": "cache.db",
            "seed": 0,
            "batch_size": 4,
            "num_new_tokens": 300,
            "prompt_placeholder": "{note}",
            "text_column": "sentence",
            "is_image": False,
            "index_col": 0,
            "llm_outputs_file": None,
            "log_file": "log.txt"
        }
        
        with open(yaml_path, 'w') as f:
            yaml.dump(config_data, f)
        
        with pytest.raises(ValueError, match="prompt_file is required"):
            KeyphraseConfig.from_yaml(yaml_path)
    
    # Test that YAML with invalid batch_size triggers validation error
    def test_load_yaml_with_invalid_batch_size(self, tmp_path):
        yaml_path = tmp_path / "config.yaml"
        config_data = {
            "llm_model_type": "gpt-4",
            "prompt_file": "prompts/test.txt",
            "cache_file": "cache.db",
            "seed": 0,
            "batch_size": 0,  # Invalid
            "num_new_tokens": 300,
            "prompt_placeholder": "{note}",
            "text_column": "sentence",
            "is_image": False,
            "index_col": 0,
            "llm_outputs_file": None,
            "log_file": "log.txt"
        }
        
        with open(yaml_path, 'w') as f:
            yaml.dump(config_data, f)
        
        with pytest.raises(ValueError, match="batch_size must be positive"):
            KeyphraseConfig.from_yaml(yaml_path)
    
    # Test that YAML with negative num_new_tokens triggers validation error
    def test_load_yaml_with_negative_num_new_tokens(self, tmp_path):
        yaml_path = tmp_path / "config.yaml"
        config_data = {
            "llm_model_type": "gpt-4",
            "prompt_file": "prompts/test.txt",
            "cache_file": "cache.db",
            "seed": 0,
            "batch_size": 4,
            "num_new_tokens": -100,  # Invalid
            "prompt_placeholder": "{note}",
            "text_column": "sentence",
            "is_image": False,
            "index_col": 0,
            "llm_outputs_file": None,
            "log_file": "log.txt"
        }
        
        with open(yaml_path, 'w') as f:
            yaml.dump(config_data, f)
        
        with pytest.raises(ValueError, match="num_new_tokens must be positive"):
            KeyphraseConfig.from_yaml(yaml_path)
    
    # Test that YAML with negative seed triggers validation error
    def test_load_yaml_with_negative_seed(self, tmp_path):
        yaml_path = tmp_path / "config.yaml"
        config_data = {
            "llm_model_type": "gpt-4",
            "prompt_file": "prompts/test.txt",
            "cache_file": "cache.db",
            "seed": -5,  # Invalid
            "batch_size": 4,
            "num_new_tokens": 300,
            "prompt_placeholder": "{note}",
            "text_column": "sentence",
            "is_image": False,
            "index_col": 0,
            "llm_outputs_file": None,
            "log_file": "log.txt"
        }
        
        with open(yaml_path, 'w') as f:
            yaml.dump(config_data, f)
        
        with pytest.raises(ValueError, match="seed must be non-negative"):
            KeyphraseConfig.from_yaml(yaml_path)
    
    # Test YAML with None for optional llm_outputs_file
    def test_load_yaml_with_none_outputs_file(self, tmp_path):
        yaml_path = tmp_path / "config.yaml"
        config_data = {
            "llm_model_type": "gpt-4",
            "prompt_file": "prompts/test.txt",
            "cache_file": "cache.db",
            "seed": 0,
            "batch_size": 4,
            "num_new_tokens": 300,
            "prompt_placeholder": "{note}",
            "text_column": "sentence",
            "is_image": False,
            "index_col": 0,
            "llm_outputs_file": None,
            "log_file": "log.txt"
        }
        
        with open(yaml_path, 'w') as f:
            yaml.dump(config_data, f)
        
        config = KeyphraseConfig.from_yaml(yaml_path)
        assert config.llm_outputs_file is None
    
    # Test YAML with special characters in string fields
    def test_load_yaml_with_special_characters(self, tmp_path):
        yaml_path = tmp_path / "config.yaml"
        config_data = {
            "llm_model_type": "gpt-4-turbo-preview",
            "prompt_file": "prompts/test.txt",
            "cache_file": "cache.db",
            "seed": 0,
            "batch_size": 4,
            "num_new_tokens": 300,
            "prompt_placeholder": "{{{{note}}}}",
            "text_column": "text_data_column",
            "is_image": False,
            "index_col": 0,
            "llm_outputs_file": None,
            "log_file": "log.txt"
        }
        
        with open(yaml_path, 'w') as f:
            yaml.dump(config_data, f)
        
        config = KeyphraseConfig.from_yaml(yaml_path)
        assert config.prompt_placeholder == "{{{{note}}}}"
        assert config.llm_model_type == "gpt-4-turbo-preview"
    
    # Test creating config from a Namespace object with all fields
    def test_from_args_with_all_fields(self):
        args = Namespace(
            llm_model_type="gpt-4",
            prompt_file="prompts/test.txt",
            cache_file="cache.db",
            seed=123,
            batch_size=16,
            num_new_tokens=400,
            prompt_placeholder="{input}",
            text_column="text",
            is_image=False,
            index_col=1,
            llm_outputs_file="output.csv",
            log_file="app.log"
        )
        
        config = KeyphraseConfig.from_args(args)
        
        assert config.llm_model_type == "gpt-4"
        assert config.seed == 123
        assert config.batch_size == 16
        assert config.num_new_tokens == 400
        assert config.index_col == 1
    
    # Test creating config from args with missing required fields
    def test_from_args_with_missing_required_fields(self):
        args = Namespace(
            llm_model_type="gpt-3.5-turbo",
            prompt_file="prompts/extract.txt",
            cache_file="cache.db",
            seed=0,
            batch_size=4,
            num_new_tokens=300,
            prompt_placeholder="{note}",
            text_column="sentence",
            is_image=False,
            # Missing index_col, llm_outputs_file, log_file
        )
        
        # Should raise error because required fields are missing
        with pytest.raises(TypeError):
            KeyphraseConfig.from_args(args)
    
    # Test that from_args ignores extra fields not in KeyphraseConfig
    def test_from_args_ignores_extra_fields(self):
        args = Namespace(
            llm_model_type="gpt-4",
            prompt_file="prompts/test.txt",
            cache_file="cache.db",
            seed=0,
            batch_size=4,
            num_new_tokens=300,
            prompt_placeholder="{note}",
            text_column="sentence",
            is_image=False,
            index_col=0,
            llm_outputs_file=None,
            log_file="log.txt",
            # Extra fields that should be ignored
            extra_field="ignored",
            another_field=999
        )
        
        config = KeyphraseConfig.from_args(args)
        
        # Config should be created successfully without extra fields
        assert config.llm_model_type == "gpt-4"
        assert not hasattr(config, "extra_field")
        assert not hasattr(config, "another_field")
    
    # Test that from_args validates the created config
    def test_from_args_with_empty_llm_model_type(self):
        args = Namespace(
            llm_model_type="",  # Invalid - empty string
            prompt_file="prompts/test.txt",
            cache_file="cache.db",
            seed=0,
            batch_size=4,
            num_new_tokens=300,
            prompt_placeholder="{note}",
            text_column="sentence",
            is_image=False,
            index_col=0,
            llm_outputs_file=None,
            log_file="log.txt"
        )
        
        with pytest.raises(ValueError, match="llm_model_type is required"):
            KeyphraseConfig.from_args(args)
    
    # Test from_args with invalid batch_size
    def test_from_args_with_invalid_batch_size(self):
        args = Namespace(
            llm_model_type="gpt-4",
            prompt_file="prompts/test.txt",
            cache_file="cache.db",
            seed=0,
            batch_size=-10,  # Invalid
            num_new_tokens=300,
            prompt_placeholder="{note}",
            text_column="sentence",
            is_image=False,
            index_col=0,
            llm_outputs_file=None,
            log_file="log.txt"
        )
        
        with pytest.raises(ValueError, match="batch_size must be positive"):
            KeyphraseConfig.from_args(args)
    
    # Test from_args with None for optional fields
    def test_from_args_with_none_optional_fields(self):
        args = Namespace(
            llm_model_type="gpt-4",
            prompt_file="prompts/test.txt",
            cache_file="cache.db",
            seed=0,
            batch_size=4,
            num_new_tokens=300,
            prompt_placeholder="{note}",
            text_column="sentence",
            is_image=False,
            index_col=None,
            llm_outputs_file=None,
            log_file="log.txt"
        )
        
        config = KeyphraseConfig.from_args(args)
        assert config.index_col is None
        assert config.llm_outputs_file is None
