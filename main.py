#!/usr/bin/env python3

import argparse
import pandas as pd
from TopicModel import LSA
from TopicModel import LDA2Vec
from TopicModel import Top2Vec_Model
from TopicModel import bertopic_kmeans
from procedure import *  
import warnings
warnings.filterwarnings("ignore")

def main(args):
    # check if the input file exists
    if not os.path.isfile(args.input):
        raise FileNotFoundError(f"Input file '{args.input}' does not exist. Please provide a valid file path.")

    # check if the output directory exists and is writable
    output_dir = os.path.dirname(args.output)
    if output_dir and not os.path.exists(output_dir):
        raise FileNotFoundError(f"Output directory '{output_dir}' does not exist. Please create it or specify a valid path.")
    if output_dir and not os.access(output_dir, os.W_OK):
        raise PermissionError(f"Output directory '{output_dir}' is not writable. Check your permissions.")

    # load the dataset
    try:
        df = pd.read_csv(args.input)
        print("Input dataframe loaded successfully")
    except Exception as e:
        raise ValueError(f"Error loading input file '{args.input}': {e}")

    # initialize the topic model based on user input
    if args.model_type.lower() == "lda2vec":
        model = LDA2Vec.LDA2Vec(df)
    elif args.model_type.lower() == "lsa":
        model = LSA.LSA(df)
    elif args.model_type.lower() == "bertopic":
        model = bertopic_kmeans.BERTopic_kmeans(df)
    elif args.model_type.lower() == "top2vec":
        model = Top2Vec_Model.Top2Vec_Model(df)
    else:
        raise ValueError("Invalid model type. Choose from 'BERTopic', 'LSA', 'LDA2Vec', or 'Top2Vec'.")

    # define list of subcategorites 
    subcategories = args.subcategories.split(",") if args.subcategories else df['subcategory'].unique().tolist()

    # train topic model on the specified subcategories
    trained_df = model.train_model(subcategories, verbose=args.verbose)

    # save output dataframe to csv
    trained_df.to_csv(args.output, index=False)
    print(f"Output saved to {args.output}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=(
            "Run topic modeling on consumer reviews.\n\n"
            "Examples:\n"
            "  ./runmodel -input data/processed_reviews.csv -model_type LDA2Vec -output results/lda2vec_output.csv\n"
            "  ./runmodel -input data/processed_reviews.csv -model_type LSA -output results/bertopic_output.csv -verbose\n\n"
            "For custom subcategories:\n"
            "  ./runmodel -input data/clorox_data.csv -model_type LSA -output outputs/lsa_output.csv "
            "-subcategories \"SPRAY CLEANERS BLEACH CLEANERS,WOOD/FURNITURE/DUST\"\n"
        ),
        formatter_class=argparse.RawTextHelpFormatter  # Allows for multiline examples
    )

    parser.add_argument(
        '-input', type=str, required=True, help="Path to the input CSV file containing reviews data."
    )
    parser.add_argument(
        '-model_type', type=str, required=True, choices=['BERTopic', 'LSA', 'LDA2Vec', 'Top2Vec'],
        help="Type of topic model to run. Options are 'BERTopic', 'LSA', 'LDA2Vec', and 'Top2Vec'."
    )
    parser.add_argument(
        '-output', type=str, required=True, help="Path to save the output CSV file with topic modeling results."
    )
    parser.add_argument(
        '-subcategories', type=str, help="Comma-separated list of subcategories to analyze. Defaults to all subcategories."
    )
    parser.add_argument(
        '-verbose', action='store_true', help="Enable verbose output for debugging and progress tracking."
    )

    args = parser.parse_args()
    main(args)
