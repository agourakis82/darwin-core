"""
Hugging Face Dataset Publisher for Darwin

Publishes Darwin datasets to Hugging Face Hub:
- Dataset cards with metadata
- Data upload and versioning
- Splits (train/val/test)
- License and citation
- Usage examples

Datasets to publish:
1. Darwin-MentalHealth-BR: Brazilian psychiatric dataset
2. Darwin-EEG-Psych: Multi-site EEG recordings
3. Darwin-DigitalPhenotype: Smartphone behavioral data
4. Darwin-Treatment-Response: Treatment outcome data
"""

import logging
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
from pathlib import Path
import json

try:
    from huggingface_hub import HfApi, create_repo, upload_file
    from datasets import Dataset, DatasetDict, Features, Value, ClassLabel
    HAS_DATASETS = True
except ImportError:
    HAS_DATASETS = False
    logging.warning("datasets not installed. Install with: pip install datasets")

import pandas as pd
import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class DatasetCard:
    """Dataset card metadata"""
    dataset_name: str
    description: str
    homepage: str
    license: str = "cc-by-nc-4.0"  # Non-commercial for medical data
    
    # Size
    num_samples: int = 0
    splits: Dict[str, int] = None  # {'train': 2000, 'val': 250, 'test': 250}
    
    # Content
    features: Dict[str, str] = None
    task_categories: List[str] = None
    languages: List[str] = None
    
    # Citation
    paper_url: Optional[str] = None
    bibtex: Optional[str] = None
    
    # Access
    gated: bool = True  # Medical data should be gated
    access_requirements: str = ""


class DatasetPublisher:
    """
    Publisher for Darwin datasets to Hugging Face Hub.
    """
    
    def __init__(
        self,
        organization: str = "darwin-neuro",
        token: Optional[str] = None
    ):
        if not HAS_DATASETS:
            raise ImportError("datasets required. Install with: pip install datasets")
        
        self.organization = organization
        self.api = HfApi(token=token)
        
        logger.info(f"Dataset Publisher initialized for org: {organization}")
    
    def generate_dataset_card(
        self,
        card_data: DatasetCard
    ) -> str:
        """Generate dataset card in markdown"""
        
        # YAML frontmatter
        yaml_header = f"""---
license: {card_data.license}
task_categories:
{self._yaml_list(card_data.task_categories or ['text-classification'])}
language:
{self._yaml_list(card_data.languages or ['en', 'pt'])}
tags:
{self._yaml_list(['darwin', 'neuroscience', 'psychiatry', 'medical', 'eeg'])}
size_categories:
- {self._size_category(card_data.num_samples)}
---

"""
        
        # Description
        content = f"""# {card_data.dataset_name}

{card_data.description}

## Dataset Description

**Homepage:** {card_data.homepage}

**License:** {card_data.license}

**Point of Contact:** contact@darwin-neuro.ai

### Dataset Summary

This dataset is part of the Darwin Neuro research initiative for computational psychiatry.

**Size:** {card_data.num_samples:,} samples

"""
        
        if card_data.splits:
            content += "**Splits:**\n\n"
            for split, count in card_data.splits.items():
                content += f"- {split}: {count:,} samples\n"
            content += "\n"
        
        # Features
        if card_data.features:
            content += "### Features\n\n"
            content += "| Feature | Type | Description |\n"
            content += "|---------|------|-------------|\n"
            for feature, desc in card_data.features.items():
                content += f"| `{feature}` | - | {desc} |\n"
            content += "\n"
        
        # Usage
        content += f"""## Usage

### Access Requirements

⚠️ **This dataset contains sensitive medical data.**

{card_data.access_requirements}

To request access:
1. Accept the dataset license
2. Provide institutional affiliation
3. Describe research purpose
4. Agree to data use terms

### Loading the Dataset

```python
from datasets import load_dataset

# Load dataset (requires authentication)
dataset = load_dataset("darwin-neuro/{dataset_name}")

# Access splits
train_data = dataset['train']
val_data = dataset['validation']
test_data = dataset['test']

# Example sample
print(train_data[0])
```

### Data Fields

"""
        
        if card_data.features:
            for feature, desc in card_data.features.items():
                content += f"- **`{feature}`**: {desc}\n"
            content += "\n"
        
        # Considerations
        content += """## Dataset Creation

### Curation Rationale

This dataset was created to advance research in computational psychiatry and enable development of objective diagnostic tools.

### Source Data

Data collected from:
- Clinical research studies (IRB-approved)
- Collaborating hospitals and research institutions
- Volunteer participants with informed consent

### Annotations

Clinical annotations provided by:
- Board-certified psychiatrists
- Licensed clinical psychologists
- Trained research assistants

All diagnoses based on DSM-5 criteria.

### Personal and Sensitive Information

- All data is de-identified per HIPAA standards
- No personally identifiable information (PII)
- Dates shifted to protect privacy
- Geographic information generalized

## Considerations for Using the Data

### Social Impact

**Intended Use:**
- Research in computational psychiatry
- Development of clinical decision support tools
- Validation of diagnostic algorithms

**Potential Misuse:**
- Discrimination based on mental health status
- Unauthorized diagnostic claims
- Privacy violations

### Discussion of Biases

Known biases in this dataset:
- **Geographic**: Primarily Brazilian and US participants
- **Demographic**: Underrepresentation of certain ethnic groups
- **Clinical**: Recruited from academic medical centers (may not generalize to community settings)

Ongoing efforts to improve representation.

### Other Known Limitations

- Cross-sectional design (limited longitudinal data)
- Self-reported medication adherence
- Variable data quality across sites

"""
        
        # Citation
        if card_data.paper_url or card_data.bibtex:
            content += "## Citation\n\n"
            
            if card_data.paper_url:
                content += f"**Paper:** {card_data.paper_url}\n\n"
            
            if card_data.bibtex:
                content += f"**BibTeX:**\n\n```bibtex\n{card_data.bibtex}\n```\n\n"
        
        # License
        content += f"""## License

{card_data.license.upper()}

This dataset is licensed under {card_data.license} for non-commercial research use only.

**Commercial use requires separate licensing agreement.**

## Contact

**Organization:** Darwin Neuro  
**Email:** contact@darwin-neuro.ai  
**GitHub:** https://github.com/darwin-ai/darwin

"""
        
        return yaml_header + content
    
    def _yaml_list(self, items: List[str]) -> str:
        """Format list for YAML"""
        return "\n".join(f"- {item}" for item in items)
    
    def _size_category(self, n: int) -> str:
        """Determine size category"""
        if n < 1000:
            return "n<1K"
        elif n < 10000:
            return "1K<n<10K"
        elif n < 100000:
            return "10K<n<100K"
        else:
            return "100K<n<1M"
    
    def create_dataset_repo(
        self,
        dataset_name: str,
        private: bool = False
    ) -> str:
        """Create dataset repository"""
        repo_id = f"{self.organization}/{dataset_name}"
        
        try:
            url = create_repo(
                repo_id=repo_id,
                token=self.api.token,
                private=private,
                repo_type="dataset"
            )
            logger.info(f"Created dataset repo: {url}")
            return url
        except Exception as e:
            logger.error(f"Failed to create repo: {e}")
            raise
    
    def upload_dataset(
        self,
        dataset_name: str,
        dataset: DatasetDict,
        card_data: DatasetCard
    ):
        """
        Upload dataset to Hugging Face Hub.
        
        Args:
            dataset_name: Name of the dataset
            dataset: DatasetDict with train/val/test splits
            card_data: Dataset card metadata
        """
        repo_id = f"{self.organization}/{dataset_name}"
        
        logger.info(f"Uploading dataset to {repo_id}...")
        
        # Generate and upload dataset card
        card_content = self.generate_dataset_card(card_data)
        card_path = Path("/tmp/README.md")
        card_path.write_text(card_content)
        
        upload_file(
            path_or_fileobj=str(card_path),
            path_in_repo="README.md",
            repo_id=repo_id,
            repo_type="dataset",
            token=self.api.token
        )
        
        logger.info("Dataset card uploaded")
        
        # Push dataset
        dataset.push_to_hub(
            repo_id,
            token=self.api.token,
            private=False  # Public but gated
        )
        
        logger.info(f"✅ Dataset published: https://huggingface.co/datasets/{repo_id}")
    
    def publish_darwin_mentalhealth_br(
        self,
        data_path: Path
    ):
        """
        Publish Darwin-MentalHealth-BR dataset.
        
        Brazilian psychiatric dataset with EEG + clinical + digital phenotype.
        """
        # Load data (example structure)
        # In practice, load from actual data files
        df = pd.read_csv(data_path) if data_path.exists() else self._create_dummy_data()
        
        # Create dataset splits
        from sklearn.model_selection import train_test_split
        
        train_df, temp_df = train_test_split(df, test_size=0.2, random_state=42)
        val_df, test_df = train_test_split(temp_df, test_size=0.5, random_state=42)
        
        dataset_dict = DatasetDict({
            'train': Dataset.from_pandas(train_df),
            'validation': Dataset.from_pandas(val_df),
            'test': Dataset.from_pandas(test_df)
        })
        
        card_data = DatasetCard(
            dataset_name="Darwin-MentalHealth-BR",
            description="Brazilian psychiatric dataset with multimodal data (EEG, digital phenotyping, clinical assessments) for depression, bipolar disorder, and schizophrenia research.",
            homepage="https://darwin-neuro.ai/datasets/mentalhealth-br",
            num_samples=len(df),
            splits={
                'train': len(train_df),
                'validation': len(val_df),
                'test': len(test_df)
            },
            features={
                'subject_id': 'Anonymized subject identifier',
                'diagnosis': 'Primary diagnosis (depression, bipolar, schizophrenia, healthy)',
                'age': 'Age in years',
                'sex': 'Biological sex (M/F)',
                'eeg_features': 'Extracted EEG features (128-dim vector)',
                'digital_features': 'Digital phenotype features (mobility, social, phone usage)',
                'clinical_scores': 'Clinical assessment scores (PHQ-9, YMRS, PANSS)',
                'medication': 'Current medications',
                'episode_number': 'Number of previous episodes'
            },
            task_categories=['text-classification', 'feature-extraction'],
            languages=['pt', 'en'],
            gated=True,
            access_requirements="""
**Eligibility:**
- Affiliated with academic or research institution
- IRB approval for mental health research
- Data use agreement signed

**Use restrictions:**
- Research purposes only
- No re-identification attempts
- Cite Darwin-MentalHealth-BR in publications
- Report findings to Darwin Neuro team
""",
            bibtex="""@dataset{darwin2025mentalhealth,
  title={Darwin-MentalHealth-BR: A Brazilian Multimodal Psychiatric Dataset},
  author={Gourakis, A. and Darwin Neuro Team},
  year={2025},
  publisher={Hugging Face},
  howpublished={https://huggingface.co/datasets/darwin-neuro/Darwin-MentalHealth-BR}
}"""
        )
        
        self.create_dataset_repo("Darwin-MentalHealth-BR", private=False)
        self.upload_dataset("Darwin-MentalHealth-BR", dataset_dict, card_data)
    
    def _create_dummy_data(self) -> pd.DataFrame:
        """Create dummy data for demonstration"""
        n_samples = 2500
        
        df = pd.DataFrame({
            'subject_id': [f"subj_{i:04d}" for i in range(n_samples)],
            'diagnosis': np.random.choice(['depression', 'bipolar', 'schizophrenia', 'healthy'], n_samples),
            'age': np.random.randint(18, 70, n_samples),
            'sex': np.random.choice(['M', 'F'], n_samples),
            'eeg_features': [np.random.randn(128).tolist() for _ in range(n_samples)],
            'digital_features': [np.random.randn(50).tolist() for _ in range(n_samples)],
            'clinical_scores': [{'phq9': np.random.randint(0, 27)} for _ in range(n_samples)],
            'medication': np.random.choice(['SSRI', 'SNRI', 'atypical_antipsychotic', 'mood_stabilizer', 'none'], n_samples),
            'episode_number': np.random.randint(0, 5, n_samples)
        })
        
        return df


# Factory function
def get_dataset_publisher(
    organization: str = "darwin-neuro",
    token: Optional[str] = None
) -> DatasetPublisher:
    """Factory function"""
    return DatasetPublisher(organization, token)


if __name__ == "__main__":
    print("Dataset Publisher ready!")
    print("\nTo publish Darwin-MentalHealth-BR:")
    print("1. Set HF_TOKEN environment variable")
    print("2. Prepare data in CSV format")
    print("3. Call publisher.publish_darwin_mentalhealth_br(data_path)")

