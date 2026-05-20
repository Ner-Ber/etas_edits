import os
import sys
import argparse
import pandas as pd
# Keep StringIO for potential in-memory operations if needed, though pandas handles files directly
import io


def convert_catalog(input_path, start_id=1):
 """
 Reads the original catalog CSV file using pandas, transforms the columns, 
 generates new IDs, converts the time format, and returns the result as a 
 CSV string.

 Args:
   input_path (str): Path to the original CSV catalog file.
   start_id (int): The starting ID for the new catalog.

 Returns:
   tuple: (str, bool) - (Converted CSV string or Error message, Success status)
 """

 try:
  # 1. Read the input CSV file directly from the path
  df = pd.read_csv(input_path)
 except FileNotFoundError:
  return (f"Error: Input file not found at '{input_path}'", False)
 except Exception as e:
  return (f"Error reading file with pandas: {e}", False)

 # Check for required columns
 required_cols = ['time', 'latitude', 'longitude', 'magnitude']
 missing_cols = [col for col in required_cols if col not in df.columns]
 if missing_cols:
  return (f"Error: Required column(s) not found: {', '.join(missing_cols)}", False)

 # 2. Add sequential ID column
 # Create an 'id' column starting from start_id
 df['id'] = range(start_id, start_id + len(df))

 # 3. Time conversion (Unix epoch to formatted string)
 # Convert Unix timestamp (seconds) to datetime objects
 # The errors='coerce' argument replaces invalid timestamps with NaT (Not a Time)
 df['time'] = pd.to_datetime(df['time'], unit='s', errors='coerce')

 # Format the datetime objects to the desired string format
 # The format string includes microseconds: '%Y-%m-%d %H:%M:%S.%f'
 # .str is necessary to access string methods on the Series after format conversion
 df['time'] = df['time'].dt.strftime('%Y-%m-%d %H:%M:%S.%f')

 # Handle rows where time conversion failed
 df['time'] = df['time'].fillna('Invalid Time')

 # 4. Select and reorder columns to the target format
 df = df[['id', 'latitude', 'longitude', 'time', 'magnitude']]

 # 5. Return as CSV string (index=False prevents pandas from writing its own row numbers)
 return (df.to_csv(index=False), True)


def main():
 """Parses arguments and orchestrates file reading, conversion, and writing."""
 parser = argparse.ArgumentParser(
   description="Convert a seismic catalog file (CSV) from its original format to a simplified format using Pandas.",
   formatter_class=argparse.RawTextHelpFormatter
 )

 # Mandatory argument: input file path
 parser.add_argument(
   'input_file',
   type=str,
   help="Path to the original CSV catalog file."
 )

 # Optional argument: output file path
 parser.add_argument(
   '-o', '--output_file',
   type=str,
   required=False,
   help=(
     "Optional path for the output CSV file.\n"
     "If not provided, the output file will be named 'converted_[INPUT_FILENAME]' "
     "and saved in the default directory: ../input_data/catalogs relative to the script."
   )
 )

 # Optional argument: starting ID
 parser.add_argument(
   '--start_id',
   type=int,
   default=1,
   help="Starting ID for the 'id' column (default: 1)."
 )

 args = parser.parse_args()

 input_path = args.input_file
 output_path = args.output_file
 start_id = args.start_id

 # 1. Determine output path if not provided
 if not output_path:
  # --- Default Output Path Logic ---
  # Get the directory of the script file itself (where catalog_converter.py is located)
  # Handle the case where the script is run directly, __file__ is available.
  try:
   script_path = os.path.abspath(__file__)
  except NameError:
   # Fallback for environments where __file__ might not be defined
   script_path = os.getcwd()

  script_dir = os.path.dirname(script_path)

  # Define the default relative output directory: ../input_data/catalogs
  DEFAULT_RELATIVE_DIR = os.path.join("..", "input_data", "catalogs")

  # Construct the absolute path for the default output directory
  default_output_dir = os.path.join(script_dir, DEFAULT_RELATIVE_DIR)

  # Get the filename details from the input file
  input_name = os.path.basename(input_path)
  base_name, ext = os.path.splitext(input_name)

  # Construct the new filename (e.g., converted_original.csv)
  new_name = f"converted_{base_name}.csv"

  # Combine the default directory and the new filename
  output_path = os.path.join(default_output_dir, new_name)
  # -----------------------------------

 print(f"Reading from: {input_path}")
 print(f"Writing to: {output_path}")

 # 2. Convert data using pandas-based function
 converted_data, success = convert_catalog(input_path, start_id=start_id)

 if not success:
  print(converted_data, file=sys.stderr)
  sys.exit(1)

 # 3. Write output file content
 try:
  # Ensure the output directory exists before writing
  output_dir = os.path.dirname(output_path)
  os.makedirs(output_dir or '.', exist_ok=True)

  with open(output_path, 'w', newline='', encoding='utf-8') as f:
   f.write(converted_data)

  # The line count is in converted_data.splitlines(), including the header
  record_count = len(converted_data.splitlines()) - 1
  print(
    f"Successfully converted and saved {record_count} records to '{output_path}'.")

 except Exception as e:
  print(f"Error writing output file: {e}", file=sys.stderr)
  sys.exit(1)


if __name__ == "__main__":
 main()
