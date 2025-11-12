# Column Breakdown

Column naming conventions follow a shorthand grouping pattern that was picked intentionally by the competition organizers to be confusing, but we can still interpret their general meaning and role from the prefixes and suffixes.

## General Structure of the Feature Names

Each column name has two main parts:

- **Prefix (feature group)** → indicates the domain or context (e.g., individual, car, region)
- **Suffix (feature type)** → indicates the data type (binary, categorical, continuous, etc.)

**For example:**
`ps_ind_01_bin` → prefix = `ps_ind` (individual info), suffix = `_bin` (binary)

## Prefixes: What They Represent

| Prefix | Meaning / Domain | Description |
|--------|------------------|-------------|
| `ind_` | Individual (policyholder) | Describes characteristics about the person who holds the policy — e.g., demographic attributes, habits, or personal records. |
| `reg_` | Regional | Relates to the geographic area or region associated with the policyholder, such as region code or registration location. |
| `car_` | Car or vehicle-related | Captures information about the vehicle itself — possibly type, age, registration class, or related attributes. |
| `calc_` | Calculated / derived features | These are synthetic or engineered features — possibly ratios, risk scores, or model-calculated indicators from prior internal models. |
| `ps_` | Porto Seguro (company-internal code) | The prefix used for "proprietary system" features — they're company-specific risk or scoring features, often anonymized (e.g. `ps_car_13` or `ps_ind_14`). |

## Suffixes: What They Indicate

| Suffix | Type | Description |
|--------|------|-------------|
| `_bin` | Binary | Only 0 or 1 values. These indicate a yes/no or true/false condition. Example: `ps_ind_01_bin` = "Does the customer have a certain attribute?" |
| `_cat` | Categorical | Encoded discrete categories (e.g. city codes, vehicle type). Must be label-encoded for ML models. |
| (no suffix) | Continuous or ordinal numeric | These are either continuous (real numbers) or ordered integers — e.g. "score", "age bracket", "count", "distance", etc. |

## Putting It Together — Column Examples

| Example Column | Likely Meaning (in plain English) |
|----------------|-----------------------------------|
| `ps_ind_01_bin` | Indicates whether a certain personal attribute or demographic flag is true for the individual. |
| `ps_ind_02_cat` | A categorical code describing the individual's type, demographic, or group. |
| `ps_car_03_cat` | A categorical variable describing a vehicle property (e.g. model class, fuel type). |
| `ps_car_13` | A continuous numeric feature about the car — possibly a calculated risk or performance score. |
| `ps_reg_03` | A numeric code describing the registration region or area code for the policyholder. |
| `ps_calc_01`, `ps_calc_02`, … | Internally derived "calculated" features — often combinations of other data, anonymized, and less interpretable individually. |
| `ps_ind_age` (if present) | Policyholder's age (continuous). |
| `ps_car_age` (if present) | Car's age in years. |
| `id` | Unique identifier for each policyholder. |
| `target` | 1 if a claim was filed, 0 otherwise (label to predict). |

## ⚠️ Special Note on `calc_` Features

These often contribute less useful signal (they're more synthetic, sometimes noisy). In many successful competition solutions, data scientists dropped or down-weighted these `calc_` columns and focused on the `ind_`, `car_`, and `reg_` ones — which contained more interpretable and stable relationships with the target.

## Summary

| Group | Meaning | Example | Type |
|-------|---------|---------|------|
| `ps_ind_` | Individual characteristics | `ps_ind_03`, `ps_ind_15` | Binary, categorical, or numeric |
| `ps_reg_` | Regional / location-based | `ps_reg_01`, `ps_reg_03` | Numeric |
| `ps_car_` | Vehicle-related | `ps_car_01_cat`, `ps_car_13` | Categorical, numeric |
| `ps_calc_` | Derived / calculated features | `ps_calc_05`, `ps_calc_20_bin` | Numeric, binary |
| `target` | Claim filed (label) | — | Binary |
| `id` | Identifier | — | Unique key |
