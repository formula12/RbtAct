# RbtAct

A comprehensive data collection, labeling and training pipeline for academic peer review generation analysis.

## Project Overview

This project provides tools for collecting, processing, and analyzing academic paper reviews and author rebuttals from conferences like ICLR. It includes modules for:

- **Data Collection**: Fetching review and rebuttal data from OpenReview
- **Label1**: Classifying weakness points by perspective 
- **Label2**: Building preference datasets for rebuttal quality
- **Mapping**: Mapping review weaknesses to rebuttal responses
- **training**: Training configs for SFT and DPO training

## Project Structure

```
RbtAct/
├── data_collection/
│   ├── Collection/        # Data fetching from OpenReview
│   │   └── get_iclr.py   # Main collection script
│   ├── Label1/            # Weakness point classification
│   │   ├── classify_weakness_points.py
│   │   ├── config.py
│   │   ├── generate_sft_dataset.py
│   │   ├── openai_utils.py
│   │   ├── run_example.py
│   │   └── test_dataset.py
│   ├── Label2/            # Rebuttal quality labeling
│   │   ├── build_preference_dataset.py
│   │   ├── classify_rebuttals_jsonl.py
│   │   ├── openai_utils.py
│   │   └── openai_utils_openai.py
│   └── Map/               # Review-Rebuttal mapping
│       ├── openai_utils.py
│       ├── prompts.py
│       └── review_rebuttal_mapper.py
└── training/
        ├── llama_8b_review_per_sft.yaml
│       └── llama_8b_review_per_dpo.yaml
```

## Installation

### Prerequisites

- Python 3.8+
- OpenAI API key (for GPT models)
- OpenReview account (for data collection)

### Setup

1. Clone the repository:
```bash
git clone https://github.com/yourusername/RbtAct.git
cd RbtAct
```

2. Create a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Configure environment variables:
Create a `.env` file in the project root:
```bash
# OpenAI API Configuration
OPENAI_API_KEY=your_api_key_here
OPENAI_ORGANIZATION=your_org_id  # Optional
OPENAI_PROJECT=your_project_id   # Optional

# Azure OpenAI Configuration (if using Azure)
AZURE_OPENAI_API_KEY=your_azure_key
AZURE_OPENAI_ENDPOINT=your_azure_endpoint
AZURE_API_VERSION=2024-02-15-preview
```

## Usage

### 1. Data Collection

Fetch review and rebuttal data from ICLR:

```bash
cd data_collection/Collection
python get_iclr.py
```

**Configuration**:
- Edit `VENUE_ID` in `get_iclr.py` to change the conference
- Modify `N_SUBMISSIONS` to limit the number of papers processed
- Set `DEBUG_FORUM_ID` and `DEBUG_REVIEW_ID` for debugging specific reviews

**Output**: JSONL file containing reviews with rebuttals and metadata

### 2. Weakness Point Classification (Label1)

Classify review weakness points by perspective (Experiments, Theory, Novelty, etc.):

```bash
cd data_collection/Label1

# Configure settings in config.py, then run:
python classify_weakness_points.py
```

**Configuration** (`config.py`):
- `JSONL_FILE`: Input file from Collection step
- `SAMPLES_PER_PERSPECTIVE`: Number of samples per perspective
- `MIN_CONFIDENCE`: Confidence threshold for filtering
- `PERSPECTIVES`: List of perspectives to classify

**Alternative**: Use the example runner:
```bash
python run_example.py
```

**Generating SFT Dataset**:
```bash
python generate_sft_dataset.py
```

**Testing Dataset**:
```bash
python test_dataset.py
```

### 3. Review-Rebuttal Mapping

Map weakness points in reviews to corresponding rebuttal responses:

```bash
cd data_collection/Map
python review_rebuttal_mapper.py
```

**Features**:
- Segments weaknesses into individual points
- Maps each point to rebuttal responses
- Extracts confidence scores for mappings

### 4. Rebuttal Quality Labeling (Label2)

Build preference datasets for rebuttal quality assessment:

```bash
cd data_collection/Label2

# Classify rebuttals by impact (CRP, SRP, VCR, DWC, DRF)
python classify_rebuttals_jsonl.py --in_path input.jsonl --out_path output.jsonl --model gpt-5-mini --rpm 2000

# Build preference dataset
python build_preference_dataset.py
```

**Classification Categories**:
- **CRP**: Concrete Revision Provided
- **SRP**: Specific Revision Plan
- **VCR**: Vague Commitment to Revise
- **DWC**: Defense Without Change
- **DRF**: Deflection / Reviewer-Faulting

## API Key Security

**Important**: This repository uses placeholder values (`xxx`) for all API keys and credentials. You must:

1. Never commit actual API keys to version control
2. Use environment variables (`.env` file) for sensitive information
3. Update placeholders in code with your credentials or load from environment

Example credentials that need replacement:
- OpenAI API keys
- Azure OpenAI endpoints and keys
- OpenReview login credentials

## Data Format

### Input Format (from Collection)
```json
{
  "paper_id": "paper123",
  "review_id": "review456",
  "weaknesses_and_questions": "Text of weaknesses...",
  "rebuttal_text": "Text of rebuttal...",
  "final_rating": 7.0,
  "final_decision": "Accept"
}
```

### Output Format (after Mapping)
```json
{
  "paper_id": "paper123",
  "review_id": "review456",
  "weakness_rebuttal_mappings": [
    {
      "weakness_point": {
        "id": "P1",
        "content": "Weakness text...",
        "perspective": "Experiments"
      },
      "rebuttal_response": {
        "id": "R1",
        "content": "Response text...",
        "impact": "CRP"
      },
      "confidence_score": 0.95
    }
  ]
}
```

## Common Issues

### API Rate Limits
If you encounter rate limiting:
- Adjust `--rpm` (requests per minute) parameter
- Reduce `--concurrency` parameter
- Use environment variable rate limiting in `openai_utils.py`

### Missing Paper Files
If paper markdown files are not found:
- Ensure `PAPER_MD_DIR` points to the correct directory
- Check that paper IDs match filename format (`{paper_id}.md`)

### Authentication Errors
For OpenReview authentication:
- Update credentials in `get_iclr.py` (line 612)
- Or use environment variables for credentials

## Development

### Adding New Perspectives
Edit `data_collection/Label1/config.py`:
```python
PERSPECTIVES = [
    "Experiments",
    "Theory",
    "YourNewPerspective",
    # ... other perspectives
]
```

### Custom Classification Prompts
Modify prompts in:
- `data_collection/Label1/classify_weakness_points.py` (perspective classification)
- `data_collection/Label2/classify_rebuttals_jsonl.py` (impact classification)
- `data_collection/Map/prompts.py` (mapping prompts)

## Contributing

Contributions are welcome! Please:
1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Submit a pull request

## Acknowledgments

This project uses:
- OpenReview API for data collection
- OpenAI GPT models for classification and mapping
- Various open-source Python libraries

