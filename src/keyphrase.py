"""
Keyphrase extraction class for LLM-based concept extraction
"""

import asyncio
import logging
from pathlib import Path
from typing import Any, List

import numpy as np
import pandas as pd
import torch
from dotenv import load_dotenv
from lab_llm import LLMApi

from src.ensemble_trainer.config import LLMConfig
from src.llm_response_types import ProtoConceptExtract
from src.llm_utils import create_llm_clients, run_prompts_batched


class Keyphrase:
    """Extract keyphrases from text using LLM

    This class provides a clean interface for extracting keyphrases from datasets
    using large language models. It handles caching, batching, and error handling.
    
    Example:
        from keyphrase_config import KeyphraseConfig
        
        config = KeyphraseConfig.from_yaml("config.yaml")
        extractor = Keyphrase(config)
        result_df = extractor.extract(data_df, train_idxs)
    """
    
    def __init__(self, config: Any) -> None:
        """Initialize the Keyphrase extractor with configuration
        
        Args:
            config: KeyphraseConfig with extraction settings
        """
        self.config = config
        self.llm: LLMApi
        self.logger = self._setup_logging()
        self._set_random_seeds()
        self._initialize_llm()
    
    def extract(
        self, 
        data_df: pd.DataFrame, 
        train_idxs: np.ndarray
    ) -> pd.DataFrame:
        """Extract keyphrases from data using LLM
        
        Args:
            data_df: Full DataFrame containing the data to process
            train_idxs: Indices of rows to extract keyphrases from
            
        Returns:
            DataFrame with 'llm_output' column added/updated containing
            comma-separated keyphrases for each processed row
            
        Raises:
            FileNotFoundError: If prompt_file doesn't exist
            ValueError: If required columns are missing from data_df
        """
        data_df = data_df.reset_index()
        # Initialize output column
        if "llm_output" not in data_df.columns:
            data_df["llm_output"] = ""
        

        # Extract keyphrases
        prompt_template = self._load_and_prepare_prompt()
        prompts = self._build_prompts(data_df, train_idxs, prompt_template)
        llm_outputs = self._get_llm_outputs(prompts)
        llm_output_strs = self._format_outputs(llm_outputs)
        
        # Update DataFrame
        data_df.loc[data_df.index[train_idxs], "llm_output"] = llm_output_strs
        
        # Log and save
        non_empty = (data_df.llm_output != "").sum()
        total = len(data_df)
        self.logger.info(
            f"Extraction complete. Non-empty outputs: {non_empty}/{total} "
            f"({100*non_empty/total:.1f}%)"
        )
        
        if self.config.llm_outputs_file:
            output_path = Path(self.config.llm_outputs_file)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            data_df.to_csv(output_path, index=(self.config.index_col is not None))
            self.logger.info(f"Results saved to {output_path}")
        
        return data_df
    
    def _setup_logging(self) -> logging.Logger:
        """Configure logging"""
        log_path = Path(self.config.log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        
        logging.basicConfig(
            format="%(asctime)s - %(name)s - %(levelname)s - %(message)s", 
            filename=self.config.log_file, 
            level=logging.INFO,
            force=True
        )
        return logging.getLogger(__name__)
    
    def _set_random_seeds(self) -> None:
        """Set random seeds for reproducibility"""
        np.random.seed(self.config.seed)
        torch.manual_seed(self.config.seed)
        self.logger.info(f"Random seed set to {self.config.seed}")
    
    def _initialize_llm(self) -> None:
        """Initialize LLM API with caching"""
        if self.config.is_image:
            raise NotImplementedError(
                "image extraction dropped in llm-api v1 migration"
            )

        load_dotenv()
        self.llm = create_llm_clients(
            LLMConfig(
                llm_model=self.config.llm_model_type,
                cache_file=self.config.cache_file,
            ),
            logger=self.logger,
        )["extraction"]
        self.logger.info(f"Initialized LLM: {self.config.llm_model_type}")
    
    def _load_and_prepare_prompt(self) -> str:
        """Load prompt template from file"""
        prompt_path = Path(self.config.prompt_file)
        if not prompt_path.exists():
            raise FileNotFoundError(f"Prompt file not found: {prompt_path}")
        
        with open(prompt_path, "r") as f:
            prompt_template = f.read()
        
        self.logger.info(f"Loaded prompt template from {prompt_path}")
        
        return prompt_template
    
    def _build_prompts(
        self,
        data_df: pd.DataFrame,
        train_idxs: np.ndarray,
        prompt_template: str
    ) -> List[str]:
        """Fill the prompt template with the text of each row to process"""
        subset_df = data_df.iloc[train_idxs]

        if self.config.text_column not in subset_df.columns:
            raise ValueError(
                f"Column '{self.config.text_column}' not found in DataFrame"
            )

        sentences = subset_df[self.config.text_column].to_numpy()
        prompts = [
            prompt_template.replace(self.config.prompt_placeholder, str(s))
            for s in sentences
        ]
        self.logger.info(
            f"Built {len(prompts)} prompts from column '{self.config.text_column}'"
        )
        return prompts

    def _get_llm_outputs(self, prompts: List[str]) -> List[Any]:
        """Get outputs from LLM API"""
        self.logger.info(
            f"Processing {len(prompts)} examples with "
            f"batch_size={self.config.batch_size}, "
            f"max_tokens={self.config.num_new_tokens}"
        )

        return asyncio.run(
            run_prompts_batched(
                self.llm,
                prompts,
                ProtoConceptExtract,
                batch_size=self.config.batch_size,
                max_new_tokens=self.config.num_new_tokens,
                desc="keyphrase extraction",
            )
        )
    
    def _format_outputs(self, llm_outputs: List[Any]) -> List[str]:
        """Format LLM outputs into comma-separated keyphrase strings"""
        formatted = []
        none_count = 0
        
        for llm_out in llm_outputs:
            if llm_out is not None and hasattr(llm_out, 'keyphrases'):
                formatted.append(','.join(llm_out.keyphrases))
            else:
                formatted.append("")
                none_count += 1
        
        if none_count > 0:
            self.logger.warning(
                f"{none_count}/{len(llm_outputs)} outputs were None or invalid"
            )
        
        return formatted
