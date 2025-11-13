# Porto Seguro's Safe Driver Prediction

This is a ML project based on the public database from [Kaggle's Porto Seguro's Safe Driver Prediction competition](https://www.kaggle.com/competitions/porto-seguro-safe-driver-prediction/data)

The goal is to predict the probability that the policy holder will file an insurance claim within 12 months.

In this repository you will find the notebook `data_processing_and_eda.ipynb` documenting our analysis and findings, as well as a breakdown of column information in `columns.md`.

If you want to explore the data yourself, you can find it on Kaggle at the link listed above.

## Project Structure

```
├── data/
│   └── porto-seguro-safe-driver-prediction/
│       ├── train.csv                    # Training dataset (595,212 rows, 59 features)
│       ├── test.csv                     # Test dataset (892,816 rows)
│       └── sample_submission.csv        # Sample submission format
├── data_processing_and_eda.ipynb        # Data processing and exploratory analysis
├── columns.md                           # Detailed feature descriptions and naming conventions
├── requirements.txt                     # Python dependencies
└── Makefile                            # Environment setup automation
```

## Dataset Overview

The dataset contains **59 anonymized features** about policyholders and their vehicles:
- **`ind_`**: Individual/policyholder characteristics (demographics, personal attributes)
- **`reg_`**: Regional/geographic information
- **`car_`**: Vehicle-related features
- **`calc_`**: Calculated/derived features from Porto Seguro's internal models
- **`target`**: Binary label (1 = claim filed within 12 months, 0 = no claim)

Feature types include:
- `_bin`: Binary features (0 or 1)
- `_cat`: Categorical features (encoded discrete categories)
- (no suffix): Continuous or ordinal numeric features

For detailed feature descriptions, see [`columns.md`](columns.md).

**Dataset Statistics:**
- Training samples: 595,212
- Test samples: 892,816
- Features: 59 (including ID and target)
- Target class imbalance: ~3.6% positive class (claims filed)

## Getting the Data

The raw dataset files are available from [Kaggle's Porto Seguro competition page](https://www.kaggle.com/competitions/porto-seguro-safe-driver-prediction/data). Download and extract the data to the `data/porto-seguro-safe-driver-prediction/` directory.

### Generating Cleaned Data

To generate the cleaned and preprocessed dataset (`train_cleaned.csv`):

1. Download the raw data from Kaggle (see link above)
2. Place the data files in `data/porto-seguro-safe-driver-prediction/`
3. Run the data processing notebook:
   ```bash
   jupyter notebook data_processing_and_eda.ipynb
   ```
4. Execute all cells in the notebook to generate `data/train_cleaned.csv`

The data cleaning process includes:
- Removing calculated features (`calc_` columns)
- Handling missing values (replacing -1 with NaN, imputing with median)
- Dropping columns with >10% missing values
- Removing the ID column
- **Result:** Clean dataset with 595,212 rows × 35 columns

## Set up your Environment

### **`macOS`** type the following commands : 

- For installing the virtual environment you can either use the [Makefile](Makefile) and run `make setup` or install it manually with the following commands:

     ```BASH
    make setup
    ```
    After that active your environment by following commands:
    ```BASH
    source .venv/bin/activate
    ```
Or ....
- Install the virtual environment and the required packages by following commands:

    ```BASH
    pyenv local 3.11.3
    python -m venv .venv
    source .venv/bin/activate
    pip install --upgrade pip
    pip install -r requirements.txt
    ```
    
### **`WindowsOS`** type the following commands :

- Install the virtual environment and the required packages by following commands.

   For `PowerShell` CLI :

    ```PowerShell
    pyenv local 3.11.3
    python -m venv .venv
    .venv\Scripts\Activate.ps1
    python -m pip install --upgrade pip
    pip install -r requirements.txt
    ```

    For `Git-bash` CLI :
  
    ```BASH
    pyenv local 3.11.3
    python -m venv .venv
    source .venv/Scripts/activate
    python -m pip install --upgrade pip
    pip install -r requirements.txt
    ```

    **`Note:`**
    If you encounter an error when trying to run `pip install --upgrade pip`, try using the following command:
    ```Bash
    python.exe -m pip install --upgrade pip
    ```

