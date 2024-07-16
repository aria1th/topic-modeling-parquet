import argparse
from pandas import DataFrame, read_parquet

def process_parquet(input_file, output_file, column_name):
    # Read parquet file
    df = read_parquet(input_file)
    print("Parquet file head:")
    print(df.head())

    # Extract specified column
    captions = df[column_name]
    captions = captions.dropna()

    print(f"\n{column_name} column head:")
    print(captions.head())

    # Save to CSV
    captions.to_csv(output_file,
                    sep=',',
                    na_rep='NaN',
                    float_format='%.2f',
                    index=False)
    
    print(f"\nSaved {len(captions)} rows to {output_file}")

def main():
    parser = argparse.ArgumentParser(description="Process Parquet file and save specific column to CSV")
    parser.add_argument("--input_file", help="Path to input Parquet file", required=True)
    parser.add_argument("--output_file", help="Path to output CSV file", required=True)
    parser.add_argument("--column", default="parsed", help="Name of the column to extract (default: parsed)")

    args = parser.parse_args()

    process_parquet(args.input_file, args.output_file, args.column)

if __name__ == "__main__":
    main()
