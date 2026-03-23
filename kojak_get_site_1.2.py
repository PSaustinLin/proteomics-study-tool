import pandas as pd
import re
from Bio import SeqIO
import os
from openpyxl import load_workbook
import numpy as np

def read_fasta(fasta_file):
    """Read FASTA file and store sequences in dictionary."""
    sequences = {}
    for record in SeqIO.parse(fasta_file, "fasta"):
        # Extract accession from header (assuming format like >sp|P02768|...)
        accession = record.id.split("|")[1] if "|" in record.id else record.id
        sequences[accession] = str(record.seq)
    return sequences

def get_absolute_site(peptide, site_relative, protein_seq, start_pos):
    """Calculate absolute site position in protein sequence."""
    # Remove modifications in square brackets
    clean_peptide = re.sub(r'\[[^\]]*\]', '', peptide)
    # Convert relative position to absolute position
    absolute_site = start_pos + int(site_relative) - 1
    return absolute_site

def process_file(input_file, fasta_file, writer, all_dataframes, mode):
    try:
        # Extract directory name for sheet name
        dir_path = os.path.dirname(input_file)
        sheet_name = os.path.basename(dir_path)
        #print(f"Processing for sheet: {sheet_name}")
        
        # Read protein sequences
        protein_sequences = read_fasta(fasta_file)
        
        # Read input file with explicit delimiter
        #print(f"Reading file: {input_file}")
        df = pd.read_csv(input_file, sep='\t', quoting=3)  # quoting=3 disables quoting
        #print(f"File loaded with {len(df)} rows")
        
        # Remove any rows with NaN values in critical columns
        df = df.dropna(subset=['SpecId', 'Proteins', 'Peptide'])
        #print(f"After removing NaN values: {len(df)} rows")
        
        # First, convert 'SpecId' column to string
        df['SpecId'] = df['SpecId'].astype(str)
        
        # Now filter rows where SpecId starts with "D-"
        df = df[~df['SpecId'].str.startswith('D-')]
        #print(f"After filtering SpecId: {len(df)} rows")

        filtered_rows = []
        for idx, row in df.iterrows():
            filtered_rows.append(idx)
        
        df = df.loc[filtered_rows]
        
        # Process each row
        sites_list = []
        
        for idx, row in df.iterrows():
            try:
                # Get accession1 from Proteins column
                acc1_match = re.search(r'sp\|(.*?)\|', row['Proteins'])
                if not acc1_match:
                    print(f"Warning: Could not extract accession1 from row {idx}")
                    sites_list.append('')
                    continue
                acc1 = acc1_match.group(1)
                if mode == 'inter':
                    acc2_match = re.search(r'sp\|(.*?)\|', row['Proteins2'])
                    if not acc2_match:
                        print(f"Warning: Could not extract accession1 from row {idx}")
                        sites_list.append('')
                        continue
                    acc2 = acc2_match.group(1)
                elif mode == 'intra':
                    acc2_match = acc1
                    acc2 = acc1

                # Process peptide
                peptide_match = re.match(r'-\.(.+?)\((\d+)\)--(.+?)\((\d+)\)\.-', row['Peptide'])
                if peptide_match:
                    pep1, site1_rel, pep2, site2_rel = peptide_match.groups()
                    
                    # Clean peptides of modifications
                    clean_pep1 = re.sub(r'\[[^\]]*\]', '', pep1)
                    clean_pep2 = re.sub(r'\[[^\]]*\]', '', pep2)
                    
                    # Find peptides in protein sequence and get start positions
                    start_pos1 = protein_sequences[acc1].find(clean_pep1) + 1
                    start_pos2 = protein_sequences[acc2].find(clean_pep2) + 1
                    
                    if start_pos1 == 0 or start_pos2 == 0:
                        print(f"Warning: Peptide not found in protein sequence at row {idx}")
                        print(f"Peptide1: {clean_pep1}, Accession1: {acc1}, Peptide2: {clean_pep2}, Accession2: {acc2}")
                        sites_list.append('')
                        continue
                    
                    # Calculate absolute sites
                    site1_abs = get_absolute_site(pep1, site1_rel, protein_sequences[acc1], start_pos1)
                    site2_abs = get_absolute_site(pep2, site2_rel, protein_sequences[acc2], start_pos2)
                    
                    # Create sites string
                    sites = f"{acc1}({site1_abs})-{acc2}({site2_abs})"
                    sites_list.append(sites)
                    
                else:
                    print(f"Warning: Could not parse peptide format in row {idx}")
                    sites_list.append('')
            except Exception as e:
                print(f"Error processing row {idx}: {str(e)}")
                sites_list.append('')
        
        # Add new columns
        df['Sites'] = sites_list
        
        # Now count occurrences of each site, considering reciprocal sites as identical
        # First, normalize all sites to a canonical form
        canonical_sites = []
        for site in sites_list:
            if not site:
                canonical_sites.append('')
                continue
            
            # Parse the site string to get components
            try:
                match = re.match(r'(.*?)\((\d+)\)-(.*?)\((\d+)\)', site)
                if match:
                    acc1, site1, acc2, site2 = match.groups()
                    # Sort to ensure canonical form
                    if f"{acc1}{site1}" > f"{acc2}{site2}":
                        canonical_site = f"{acc2}({site2})-{acc1}({site1})"
                    else:
                        canonical_site = site
                    canonical_sites.append(canonical_site)
                else:
                    canonical_sites.append(site)
            except:
                canonical_sites.append(site)
        
        # Count occurrences of each canonical site
        site_counts = {}
        for site in canonical_sites:
            if site:  # Only count non-empty sites
                site_counts[site] = site_counts.get(site, 0) + 1
        
        # Add count column
        count_list = []
        for site in sites_list:
            if not site:
                count_list.append(0)
                continue
            
            # Get the canonical form of this site
            try:
                match = re.match(r'(.*?)\((\d+)\)-(.*?)\((\d+)\)', site)
                if match:
                    acc1, site1, acc2, site2 = match.groups()
                    if f"{acc1}{site1}" > f"{acc2}{site2}":
                        canonical_site = f"{acc2}({site2})-{acc1}({site1})"
                    else:
                        canonical_site = site
                    count_list.append(site_counts.get(canonical_site, 0))
                else:
                    count_list.append(0)
            except:
                count_list.append(0)
                
        # Add the count column to the DataFrame
        df['Count'] = count_list
        
        # Store the dataframe with its sheet name for later writing
        all_dataframes[sheet_name] = df
        
        print(f"Successfully processed data for sheet: {sheet_name}")
        return df
        
    except Exception as e:
        print(f"Error processing file {input_file}:")
        print(f"Error type: {type(e).__name__}")
        print(f"Error message: {str(e)}")
        raise

def main():
    # Hardcoded paths for convenience - modify as needed
    input_list_file = "output/pin_files.txt"
    fasta_file = r'C:\Crux\databases\IG.fasta'
    output_file = r'C:\Crux\xlOutput\20260311\IG_top10.xlsx'
    # mode = 'inter' if reading perc.inter.pin files; 'intra' if reading perc.intra.pin files
    mode = 'inter'
    # 有些會出現error無法輸出的原因是Proteins2後面的column還有值，第一步將T開頭的scan把正確的protein移到Proteins2，第二步將後面column的值全部刪除
    #print(f"Using input file list: {input_list_file}")
    #print(f"Using FASTA file: {fasta_file}")
    #print(f"Output will be saved to: {output_file}")
    
    # Read the input file list
    input_files = []
    try:
        with open(input_list_file, 'r') as f:
            line_num = 0
            for line in f:
                line_num += 1
                line = line.strip()
                if line and not line.startswith('#'):  # Skip empty lines and comments
                    input_files.append(line)
                    #print(f"Added file #{line_num}: {line}")
    except Exception as e:
        print(f"Error reading input file list: {str(e)}")
        return
    
    if not input_files:
        print("No input files found in the list.")
        return
    
    #print(f"Found {len(input_files)} input files to process.")
    
    # Dictionary to store all dataframes with sheet names
    all_dataframes = {}
    
    # Create ExcelWriter object
    with pd.ExcelWriter(output_file, engine='openpyxl') as writer:
        # Process each input file
        for input_file in input_files:
            try:
                #print(f"\nProcessing: {input_file}")
                process_file(input_file, fasta_file, writer, all_dataframes, mode)
                #print(f"Added data for {input_file}")
            except Exception as e:
                print(f"Failed to process {input_file}")
                print(f"Error: {str(e)}")
        
        # After processing all files, write each dataframe to its sheet
        for sheet_name, df in all_dataframes.items():
            #print(f"Writing sheet: {sheet_name}")
            # Excel has a 31 character limit for sheet names
            truncated_sheet_name = sheet_name[:31]
            df.to_excel(writer, sheet_name=truncated_sheet_name, index=False)
    
    print(f"Successfully saved combined output to: {output_file}")

if __name__ == "__main__":
    main()