# Data Filtering

The code uses [OmniParser](https://github.com/microsoft/OmniParser) to generate bounding boxes for images, followed by a filtering step to clean noisy data.

## Setup

1. **Install Environment**  
   Follow the environment-setup instructions provided in the [OmniParser repository](https://github.com/microsoft/OmniParser).

2. **Install Additional Dependency**  
   Install the `imagesize` Python package if it is not already included:
```bash
pip install imagesize
```

3. **Download [OmniParser](https://github.com/microsoft/OmniParser) Checkpoint**
   Download the checkpoint, and place it `OmniParser-v2.0` folder.

## Usage

### Step 1 - Generate Bounding Boxes
Run the `run.py` script to generate bounding boxes for all images using OmniParser:
```bash
python run.py
```

### Step 2 - Filter Noisy Data
Run the `filter.py` script to clean the dataset by checking whether each data bounding box overlaps only with the OmniParser output:
```bash
python filter.py
```

## Folder Structure

├── run.py             # Script to generate bounding boxes using OmniParser  
├── filter.py          # Script to clean noisy datasets  
├── inp.json           # Raw input data  
├── OmniParser-v2.0/   # OmniParser checkpoints  
├── hw_cache.json      # Cached height and width info for all images  
├── images/            # Directory containing all images  
├── log/               # Outputs from OmniParser  
├── clean.json         # Cleaned output data  
├── README.md          # This file  
└── ...                # Other files or directories     

## Notice
In `inp.json`, all bounding box coordinates are stored using normalized values in the range [0, 1000].