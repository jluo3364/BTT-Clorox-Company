# BTT-Clorox-Company

## Team Members
Jessica Luo, Saloni Jain, Melody He

## Table of Contents
1. [Project Objective](#project-objective)   
2. [Data Description](#data-description)  
3. [External Dependencies](#external-dependencies)  
   - [Installation](#installation)  
4. [Repository Structure](#repository-structure)  
5. [Usage](#usage)


## Project Objective

The objective of this project is to create an advanced topic modeling module to help Clorox analyze consumer reviews and extract meaningful themes and insights. By leveraging machine learning techniques and LLMs, we aim to identify and consolidate key topics and trends within the reviews. Our goal is to improve Clorox’s current implementation by reducing redundant topics and minimizing the number of topics generated per subcategory, ensuring each topic is relevant and accurately reflects consumer sentiment.

The impact of this project will be to deepen Clorox's understanding of consumer feedback, supporting the R&D and marketing teams in making data-driven decisions. The results of the topic modeling will be displayed on a dashboard, enabling Clorox to utilize these insights for product development, refine marketing strategies, and ultimately improve customer satisfaction.

## Data Description

The dataset for this analysis comprises of 670,798 consumer reviews collected across various platforms for a range of products, dated from July 2022 to July 2024. The products are divided into two main categories: Cleaning and Personal Care, with reviews also distinguished by two brand types: Clorox vs. Competitor brands. Within these categories, products are broken down into 94 specific subcategories, such as “Floor Cleaners” and “Lip Care.”

Here is an overview of the two primary datasets used in this analysis:

1. clorox_data.csv: The main dataset provided by the Clorox Company, containing the raw consumer reviews.
2. processed_reviews.csv: The dataset prepared for topic modeling. This dataset was created by removing duplicate reviews and preprocessing the text, including converting to lowercase, removing punctuation, tokenizing, and lemmatizing words in the review text.

## External Dependencies

To run the code in this repository, make sure you have the dependencies listed in requirements.txt installed.

### Installation

1. Ensure you have **Python** installed on your machine. We recommend using **Anaconda**, which provides an easy way to manage Python and package dependencies. You can download Anaconda from [here](https://www.anaconda.com/products/distribution).

2. Install the required dependencies using the following command:

    ```bash
    pip install -r requirements.txt
    ```

3. If you're using **Anaconda**, you can create and activate a new environment with the dependencies:

    ```bash
    conda create -n myenv python=3.8 
    conda activate myenv
    pip install -r requirements.txt
    ```

4. Verify the installation by running:

    ```bash
    python -c "import pkg_resources; print('All dependencies are installed.')"
    ```

Our project also uses the Groq API, which requires the user to acquire an API key and store it securely.

### Step 1: Acquire the API Key
1. **Sign up for a Groq Account**:  
   Visit the [Groq Developer Portal](https://developer.groq.com/) and create an account if you don’t already have one.

2. **Generate an API Key**:  
   - Log in to your Groq account.
   - Navigate to the **API Keys** section under your account settings.
   - Click **Generate API Key** and copy the key. Make sure to save it securely, as you won't be able to view it again.

---

### Step 2: Store the API Key in an `.env` File
1. Create a new file named `.env` in the root directory of your project.
2. Add the following line to the `.env` file:
   ```bash
   GROQ_API_KEY=<your-api-key>

## Repository Structure

```
BTT-Clorox-Company/
├── bertopic/                           # Folder containing bertopic notebooks experimenting with clustering methods
│   ├── bertopic_hdbscan.ipynb          
│   ├── bertopic_kmeans_lotion.ipynb    
│   ├── bertopic_hdbscan.ipynb         
│   └── bertopic_kmeans_lotion.ipynb    
├── clustering                          # Folder containing initial experimentation with clustering
│   ├── lda.ipynb
│   ├── llm_clustering.ipynb
|   └── nmf.ipynb
├── data/                               # Folder containing input data
├── eda/                                # Folder containing initial eda
├── outputs/                            # Folder containing topic modeling outputs
├── topic_modeling_notebooks/           # Folder containing notebooks for topic modeling
│   ├── lda2vec.ipynb
│   ├── lsa_standardized.ipynb
│   ├── lsa_with_preprocessing.ipynb
│   ├── top2vec_modeling.ipynb 
├── TopicModel/                         # Folder containing Topic Model classes for LDA2Vec, LSA, and Top2Vec
│   ├── TopicModel.py                   # Base Topic Model class
│   ├── LDA2Vec.py                      # LDA2Vec class implementation, inherits TopicModel
│   ├── LSA.py                          # LSA class implementation, inherits TopicModel
│   ├── bertopic_kmeans.py              # Bertopic class implementation, inherits TopicModel
│   ├── Top2Vec_Model.py                # Top2Vec class implementation, inherits TopicModel
│   ├── bertopic testing/               # Notebooks to test bertopic class implementation
│   ├── lsa testing/                    # Notebooks to test lsa class implementation
│   ├── lda2vec_testing/                # Notebooks to test lda2vec class implementation
├── .gitignore                          # Files and directories to ignore in git
├── main.py                             # Contains logic for command line interface
├── procedure.py                        # Contains standardized procedure for each of the topic modeling methods
├── README.md                           # Project README 
├── requirements.txt                    # List of dependencies
└── runmodel                            # Executable for running topic modeling
```

## Usage

The `runmodel` script provides a command-line interface for running various topic modeling algorithms.

### Make the Script Executable

To make the `runmodel` script executable, run the following command:
```bash
chmod a+x runmodel
```

```markdown
## Usage

The `runmodel` script provides a command-line interface for running various topic modeling algorithms.

### Make the Script Executable

To make the `runmodel` script executable, run the following command:
```bash
chmod a+x runmodel
```

### Basic Usage

```bash
./runmodel -input <input_csv> -model_type <model> -output <output_csv> [-subcategories <subcategory_list>]
```

### Arguments

- `-input` *(required)*: Path to the input CSV file containing reviews data.
- `-model_type` *(required)*: Type of topic model to run. Options:
  - `BERTopic`
  - `LSA`
  - `LDA2Vec`
  - `Top2Vec`
- `-output` *(required)*: Path to save the output CSV file with topic modeling results.
- `-subcategories` *(optional)*: Comma-separated list of subcategories to analyze. Defaults to all subcategories in the dataset.

---

### Examples

#### Run LDA2Vec on the Entire Dataset
```bash
./runmodel -input data/processed_reviews.csv -model_type LDA2Vec -output outputs/lda2vec_output.csv
```

#### Run LSA on Specific Subcategories
```bash
./runmodel -input data/clorox_data.csv -model_type LSA -output outputs/lsa_output.csv -subcategories "WOOD/FURNITURE/DUST"
```

#### Run BERTopic on All Subcategories
```bash
./runmodel -input data/processed_reviews.csv -model_type BERTopic -output outputs/bertopic_output.csv
```

#### Run Top2Vec on the Dataset
```bash
./runmodel -input data/review_dataset.csv -model_type Top2Vec -output outputs/top2vec_output.csv
```
```

