import re
import csv
import argparse
import itertools
from Bio import SeqIO

def tryptic_digest_with_miscleavage(sequence, max_miscleavage=0):
    """
    Performs tryptic digestion by cleaving after K or R (unless followed by P).
    Allows for up to max_miscleavage missed cleavage sites.
    """
    # First, find all potential cleavage sites
    cleavage_sites = [0]  # Start of the sequence
    
    for i in range(len(sequence) - 1):
        if sequence[i] in ['K', 'R'] and sequence[i+1] != 'P':
            cleavage_sites.append(i + 1)  # Position after K or R
    
    cleavage_sites.append(len(sequence))  # End of the sequence
    
    # Generate peptides with 0 to max_miscleavage missed cleavages
    peptides = []
    peptide_positions = []
    
    for i in range(len(cleavage_sites) - 1):
        for mc in range(min(max_miscleavage + 1, len(cleavage_sites) - i - 1)):
            start_pos = cleavage_sites[i]
            end_pos = cleavage_sites[i + mc + 1]
            
            # Get the peptide sequence
            peptide = sequence[start_pos:end_pos]
            
            # Only add non-empty peptides
            if peptide:
                peptides.append(peptide)
                peptide_positions.append((start_pos, end_pos - 1))  # 0-based indexing
    
    return peptides, peptide_positions

def find_modification_sites(peptide):
    """Finds M and C positions in peptide (1-based indexing)."""
    mod_sites = {}
    for i, aa in enumerate(peptide):
        #if aa == 'M' or aa == 'P':
        if aa == 'M':
            mod_sites[i+1] = '15.9949'  # Oxidation
        #elif aa == 'C':
        #    mod_sites[i+1] = '57.0215'  # Carbamidomethylation
    return mod_sites

def find_special_residues(peptide):
    """Finds K, D, E positions in peptide (1-based indexing)."""
    special_sites = []
    for i, aa in enumerate(peptide):
        #if aa in ['K', 'D', 'E']:
        #if aa in ['K', 'N', 'Q']:
        if aa in ['C']:
            special_sites.append(i+1)
    return special_sites

def format_modification_string(mod_dict):
    """Formats modification dictionary into site_mass format."""
    if not mod_dict:
        return ""
    return ",".join([f"{pos}_{mass}" for pos, mass in mod_dict.items()])

def process_fasta(fasta_file, output_file, max_miscleavage=0, accession_numbers=None):
    """Process FASTA file and create CSV output."""
    
    # Read the FASTA file and optionally filter by accession numbers
    selected_proteins = {}
    if accession_numbers is not None:
        accession_numbers = set(accession_numbers)
    
    try:
        # Check if we can open and read the file
        with open(fasta_file, 'r') as f:
            pass
            
        for record in SeqIO.parse(fasta_file, "fasta"):
            # Print each record's ID for debugging
            print(f"Processing record: {record.id}")
            
            # Extract accession from the FASTA header - try different formats
            accession = record.id
            
            if accession_numbers is None:
                selected_proteins[accession] = str(record.seq)
                continue
            
            # Check if the full ID is in the list
            found = False
            if accession in accession_numbers:
                selected_proteins[accession] = str(record.seq)
                found = True
                print(f"Found exact match for accession: {accession}")
                
            # If not found, try to extract accession from different parts of the ID
            if not found and '|' in record.id:
                parts = record.id.split('|')
                for part in parts:
                    if part in accession_numbers:
                        selected_proteins[part] = str(record.seq)
                        found = True
                        print(f"Found match in parts for accession: {part}")
                        break
        
        print(f"Total proteins selected: {len(selected_proteins)}")
        
        # If no matches, try a more lenient approach
        if len(selected_proteins) == 0:
            print("No exact matches found, trying partial matches...")
            for record in SeqIO.parse(fasta_file, "fasta"):
                for accession in accession_numbers:
                    if accession in record.id:
                        selected_proteins[accession] = str(record.seq)
                        print(f"Found partial match for accession: {accession} in {record.id}")
    
    except Exception as e:
        print(f"Error reading FASTA file: {e}")
        return
    
    # If still no proteins found, add an example for demonstration
    if len(selected_proteins) == 0:
        print("No proteins found in FASTA file, using example data for demonstration")
        # Add example protein with PEMPCTIDEK peptide for demonstration
        selected_proteins["EXAMPLE"] = "PEMPCTIDEKYLGRTR"
    
    # Process each protein and write results
    peptide_rows = {}
    for accession, sequence in selected_proteins.items():
        # Remove end-of-lines if any
        sequence = sequence.replace('\n', '').replace('\r', '')
        
        # Perform tryptic digestion with missed cleavages
        peptides, peptide_positions = tryptic_digest_with_miscleavage(sequence, max_miscleavage)
        print(f"Digested {accession} into {len(peptides)} peptides (including missed cleavages)")
        
        for i, peptide in enumerate(peptides):
            peptide_start, peptide_end = peptide_positions[i]
            
            # Calculate missed cleavages in this peptide
            missed_cleavages = 0
            for j in range(len(peptide) - 1):
                if peptide[j] in ['K', 'R'] and peptide[j+1] != 'P':
                    missed_cleavages += 1
            
            # Find modification sites
            mod_sites = find_modification_sites(peptide)
            
            # Find K, D, E positions
            special_sites = find_special_residues(peptide)
            
            # Generate all possible modification combinations
            mod_keys = list(mod_sites.keys())
            mod_combinations = []
            
            # Add case with no modifications
            mod_combinations.append({})
            
            # Add all other combinations
            for j in range(1, len(mod_keys) + 1):
                for combo in itertools.combinations(mod_keys, j):
                    mod_dict = {k: mod_sites[k] for k in combo}
                    mod_combinations.append(mod_dict)
            
            # For each modification combination
            for mod_combo in mod_combinations:
                mod_string = format_modification_string(mod_combo)
                
                # If special sites exist, accumulate a row for each
                if special_sites:
                    for site in special_sites:
                        # Calculate the position in the original protein sequence (0-based to 1-based)
                        protein_position = peptide_start + site - 1 + 1  # Convert to 1-based indexing
                        key = (peptide, mod_string, site, missed_cleavages)
                        if key not in peptide_rows:
                            peptide_rows[key] = {
                                'peptide': peptide,
                                'mod_string': mod_string,
                                'site': site,
                                'missed_cleavages': missed_cleavages,
                                'accessions': set(),
                                'protein_positions': set()
                            }
                        peptide_rows[key]['accessions'].add(accession)
                        peptide_rows[key]['protein_positions'].add(str(protein_position))
                else:
                    # If no special sites, still track the peptide and mods with an empty site column
                    key = (peptide, mod_string, '', missed_cleavages)
                    if key not in peptide_rows:
                        peptide_rows[key] = {
                            'peptide': peptide,
                            'mod_string': mod_string,
                            'site': '',
                            'missed_cleavages': missed_cleavages,
                            'accessions': set(),
                            'protein_positions': set()
                        }
                    peptide_rows[key]['accessions'].add(accession)
                    peptide_rows[key]['protein_positions'].add('')
    
    with open(output_file, 'w', newline='') as f:
        writer = csv.writer(f)
        # Add header row
        writer.writerow(['Peptide', 'Modifications', 'PeptidePosition', 'Accession', 'ProteinPosition', 'MissedCleavages'])
        
        for row in peptide_rows.values():
            accessions = sorted(row['accessions'])
            protein_positions = sorted(row['protein_positions'], key=lambda x: int(x) if x.isdigit() else x)
            writer.writerow([
                row['peptide'],
                row['mod_string'],
                row['site'],
                ','.join(accessions),
                ','.join(protein_positions),
                row['missed_cleavages']
            ])

def main():
    parser = argparse.ArgumentParser(description="Generate peptide candidates from a FASTA file.")
    parser.add_argument("-f", "--fasta", default=r'C:\Crux\databases\IGHG3_APOA2.fasta', help="FASTA file path")
    parser.add_argument("-o", "--output", default=r'C:\env\test\IGHG3_ds.csv', help="Output CSV path")
    parser.add_argument("-m", "--max-miscleavage", type=int, default=3, help="Maximum missed cleavages")
    parser.add_argument("-a", "--accession-numbers", nargs="+", help="Optional accession numbers to filter")
    args = parser.parse_args()

    process_fasta(
        args.fasta,
        args.output,
        max_miscleavage=args.max_miscleavage,
        accession_numbers=args.accession_numbers
    )
    print(f"Processing complete. Results saved to {args.output}")

if __name__ == "__main__":
    main()