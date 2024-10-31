# MVAA-DT
MVAA-DT is designed for representation alignment and data imputation in spatial omics. It achieves a comprehensive representation of incomplete spatial omics data, by leveraging shared relationships within complete spatial multi-omics data to guide the imputation and integration of missing information.This repository contains MVAA-DT script and jupyter notebooks. 

## Directory Structure
- **MVAADT**: Contains the main implementation of the MVAADT algorithm, including adaptations for both complete and incomplete spatial multi-omics scenarios. Also includes a pre-training method for single-cell data (`MVAADT_SC.py`).
- **data**: Includes all datasets or data sources used in the project.
- **result**: Stores the output files or results generated from the experiments.
- **translationSet**: Contains the weights of the pre-trained translation model.
<!-- - **tutorial**: Provides example scripts to help users get started with the project. The `complete_multi_omics_example.ipynb` notebook demonstrates the implementation of the MVAA-DT method using complete multi-omics data, specifically with the dataset `Dataset1_Lymph_Node1`. The `incomplete_multi_omics_example.ipynb` notebook provides an example of implementing the MVAA-DT method with incomplete multi-omics data, using the dataset `Dataset5_Mouse_Brain_P22`, where 20% of the RNA data is randomly masked to simulate data incompleteness. Additionally, `pretrain_single_cell_data` demonstrates the pre-training of the translation model using single-cell data (e.g., PBMC). -->

## Tutorial

This section includes example scripts to help users get started with the project:

- **`complete_multi_omics_example.ipynb`**: Demonstrates the implementation of the MVAA-DT method using complete multi-omics data, specifically the `Dataset1_Lymph_Node1`.

- **`incomplete_multi_omics_example.ipynb`**: Illustrates the MVAA-DT method with incomplete multi-omics data, utilizing `Dataset5_Mouse_Brain_P22`, where 20% of the RNA data is randomly masked to simulate data incompleteness.

- **`pretrain_single_cell_data`**: Showcases the pre-training of the translation model using single-cell data, such as PBMC.



## Requirements
Install the required dependencies listed in `requirement.txt` by running:

```bash
pip install -r requirement.txt
```

## Usage
1. Clone the repository:

   ```bash
   git clone https://github.com/SnowyZhang/MVAADT.git
   ```
2. Navigate to the project directory:

   ```bash
   cd MVAADT
   ```
3. Open examples in tutorial to execute the cells.

## Contact
If you have any questions, feedback, feel free to reach out:
**Email**: [zhangxsnowy@gmail.com](mailto:zhangxsnowy@gmail.com)
We welcome any contributions and suggestions to improve this project!



