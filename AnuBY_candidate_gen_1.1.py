import re
import csv
import itertools
from Bio import SeqIO

def tryptic_digest_with_miscleavage(sequence, max_miscleavage=2):
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
        #if aa == 'M':
        if aa == 'M' or aa == 'P':
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

def process_fasta(fasta_file, accession_numbers, output_file, max_miscleavage=2):
    """Process FASTA file and create CSV output."""
    
    # Read the FASTA file and filter by accession numbers
    selected_proteins = {}
    
    try:
        # Check if we can open and read the file
        with open(fasta_file, 'r') as f:
            pass
            
        for record in SeqIO.parse(fasta_file, "fasta"):
            # Print each record's ID for debugging
            print(f"Processing record: {record.id}")
            
            # Extract accession from the FASTA header - try different formats
            accession = record.id
            
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
    with open(output_file, 'w', newline='') as f:
        writer = csv.writer(f)
        # Add header row
        writer.writerow(['Peptide', 'Modifications', 'PeptidePosition', 'Accession', 'ProteinPosition', 'MissedCleavages'])
        
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
                    
                    # If special sites exist, write a row for each
                    if special_sites:
                        for site in special_sites:
                            # Calculate the position in the original protein sequence (0-based to 1-based)
                            protein_position = peptide_start + site - 1 + 1  # Convert to 1-based indexing
                            writer.writerow([peptide, mod_string, site, accession, protein_position, missed_cleavages])
                    #else:
                        # If no special sites, still write the peptide and mods with empty third column
                        #writer.writerow([peptide, mod_string, "", accession, "", missed_cleavages])

def main():
    # Example usage
    fasta_file = r'C:\Crux\databases\ApoA2_A1AT.fasta'  # Path to your FASTA file
    accession_numbers = ['P02652', 'P01009']  # List of accession numbers to filter
    output_file = r'C:\env\test\A1AT-ApoA2_ds.csv'
    
    process_fasta(fasta_file, accession_numbers, output_file)
    print(f"Processing complete. Results saved to {output_file}")

if __name__ == "__main__":
    main()