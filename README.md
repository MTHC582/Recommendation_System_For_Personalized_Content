# Netflix Prize Dataset: Personalized Discovery Recommendation System

## Project Overview
This project implements a personalized movie recommendation system using historical interaction data from the Netflix Prize Dataset. The system is designed to analyze sparse user-item interaction histories, predict numerical ratings for unseen titles, and generate tailored Top-10 discovery recommendation lists for active users. 

The implementation addresses critical real-world engineering constraints, balancing predictive accuracy against computational efficiency and online inference latency.

---

## Directory Structure
The following layout represents the workspace organization after all components of the data and modeling pipelines have executed successfully.

```text
netflix-recommendation-system/
│
├── data/
│   ├── combined_data_2.txt        # Raw interaction subset from Kaggle
│   └── movie_titles.csv           # Raw movie metadata catalog
│
├── final_data/
│   ├── netflix_final.csv          # Filtered baseline dataset
│   ├── Train_Data.csv             # Training data split (80%)
│   ├── Test_Data.csv              # Test data split (20%)
│   ├── svd_model.pkl              # Serialized production SVD model
│   └── svd_predictions.pkl        # Serialized batch predictions for test set
│
├── notebooks/
│   ├── 1_EDA_DataProc.ipynb       # Data parsing, analysis, and stratification
│   ├── 2_ModelTrain_Eval.ipynb    # Model training, validation, and benchmarking
│   └── 3_Recommendation_gen.ipynb # Live inference and Top-10 recommendation loop
│
├── requirements.txt               # Documented environment dependencies
└── README.md                      # Project documentation
```

---

## Setup and Execution Guide

### 1. Repository Cloning and Environment Setup

Clone the repository and navigate into the project workspace directory:

```bash
git clone [GITHUB](https://github.com/MTHC582/Recommendation_System_For_Personalized_Content)
cd Recommendation_System_For_Personalized_Content
```

Initialize a Python virtual environment (`venv`) to ensure dependency isolation, and activate it based on your operating system:

**On Windows:**

```bash
python -m venv venv
venv\Scripts\activate
```

**On macOS / Linux:**

```bash
python3 -m venv venv
source venv/bin/activate
```

Once the virtual environment is active, update pip and install the required dependencies:

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

### 2. Dataset Acquisition and Placement

1. Download the raw dataset files directly from the [Kaggle Netflix Prize Data Repository](https://www.kaggle.com/datasets/netflix-inc/netflix-prize-data).
2. Create a directory named `data/` at the root level of the project workspace.
3. Extract and place the `combined_data_2.txt` and `movie_titles.csv` files inside the newly created `data/` directory.

### 3. Pipeline Execution

Run the Jupyter notebooks inside the `notebooks/` directory sequentially:

1. **`1_EDA_DataProc.ipynb`**: Processes the raw unstructured text data, computes matrix attributes, executes log-stratified sampling, and saves intermediate data splits into the `final_data/` folder.
2. **`2_ModelTrain_Eval.ipynb`**: Loads the prepared data splits, trains the collaborative filtering models, computes performance metrics (RMSE, MAP@10), and exports the trained SVD model object alongside prediction lists.
3. **`3_Recommendation_gen.ipynb`**: Restores the serialized SVD model to execute live inference against unseen movies, systematically filtering out historical interactions to produce true personalized discovery feeds.

---

## Evaluation Metrics

Models are evaluated using a strict temporal/random 80/20 train-test partition.

* **Root Mean Squared Error (RMSE)**: Evaluates the absolute accuracy of predicted ratings against true user ratings.
* **Mean Average Precision @ 10 (MAP@10)**: Measures recommendation ranking quality by penalizing relevant items placed lower in the Top-10 discovery sequence. An interaction is defined as relevant if the ground-truth user rating is >= 3.5 stars.

---

## Technical Insights and Project Knowledge

### 1. Memory and Scale Limitations in Collaborative Filtering

* **User-Based Limitations**: Attempting to build a User-Based Collaborative Filtering model on this dataset requires computing a similarity matrix of size 240,000 x 240,000. This creates a massive memory footprint capable of exhausting standard consumer-grade RAM.
* **Item-Based Efficiency**: By shifting to an Item-Based approach, the similarity matrix scales relative to the unique movie catalog size (4,711 x 4,711). This reduction makes memory utilization lightweight and execution highly efficient.

### 2. Computational Latency Trade-offs

The benchmark results expose an inverse relationship between training time and real-time query resolution speed:

* **Item-Based KNN**: Trains rapidly in 2.83 seconds by building a simple item-similarity matrix, but exhibits a slower inference latency of 6.94 seconds per user during validation due to active neighborhood lookup calculations.
* **Matrix Factorization (SVD)**: Requires a longer training time of 13.53 seconds to factorize the sparse matrix into lower-dimensional dense spaces. However, it minimizes inference latency down to 1.88 seconds per user because runtime predictions are reduced to simple vector dot products. SVD was selected for production deployment due to this real-time lookup advantage.

### 3. Mitigating Cascading Matrix Collapse via Stratification

* **The Long-Tail Problem**: Exploratory data analysis showed that the top 10% most popular movies attract 75.84% of all user ratings, whereas the bottom 50% account for only 1.97%. Naive random sampling causes the model to solely learn popular mainstream trends.
* **The Collapse Risk**: Attempting to handle this imbalance by filtering out low-frequency nodes iteratively often causes a Cascading Matrix Collapse (`ZeroDivisionError`), where continuous row deletions drop users or items below thresholds recursively until the matrix is entirely depleted.
* **The Structural Solution**: This pipeline circumvents collapse by enforcing a baseline filter first (`min_user=5`, `min_movie=10`), followed by mapping movies into four distinct log-scale tiers. By completely preserving the long-tail tier while downsampling blockbusters, the pipeline maintains a structural database of 1.96 million rows containing 100% of the active movie catalog (4,711 items) at an authentic real-world data sparsity of 99.8721%.

### 4. Dynamic Temporal Biases

* Data analysis reveals that user preferences are fundamentally non-static; a user's tastes change over multi-year intervals.
* While baseline collaborative filtering architectures analyze static User-Movie-Rating pairs, advanced implementations (such as TimeSVD++) account for temporal bias changes over distinct tracking dates. To adhere to project guidelines prioritizing simplicity, analytical clarity, and code stability over unnecessary infrastructure complexity, standard SVD matrix factorization remains the core operational choice for this system.
