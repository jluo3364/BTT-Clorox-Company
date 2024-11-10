# BTT-Clorox-Company

## Team Members
Jessica Luo, Saloni Jain, Melody He

## Table of Contents
1. [Project Objective](#project-objective)   
2. [Data Description](#data-description)  
3. [External Dependencies](#external-dependencies)  
   - [Installation](#installation)  
4. [Repository Structure](#repository-structure)  


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

## Repository Structure

```
BTT-Clorox-Company/
├── TopicModel/                         # Folder containing Topic Model classes for LDA2Vec, LSA, and Top2Vec
│   ├── TopicModel.py                   # Base Topic Model class
│   ├── LDA2Vec.py                      # LDA2Vec class implementation, inherits TopicModel
│   ├── LSA.py                          # LSA class implementation, inherits TopicModel
├── bertopic/                           # Folder containing bertopic notebooks experimenting with clustering methods
│   ├── bertopic_hdbscan.ipynb          # Detailed installation instructions
│   └── bertopic_kmeans_lotion.ipynb    # In-depth usage guide and examples
├── clustering                          # Folder containing initial experimentation with clustering
│   ├──
|   └──
├── data/                               # Folder containing input data
├── eda/                                # Folder containing initial eda
├── merging/                            # Folder containing notebooks to merge topic modeling outputs
│   ├── 
│   └── 
├── .gitignore                          # Files and directories to ignore in git
├── requirements.txt                    # List of dependencies
└── README.md                           # Project README 
```

