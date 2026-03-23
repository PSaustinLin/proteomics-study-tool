import os

def add_proteins2_column(pin_file_list_path):
    """
    Add a column titled 'Proteins2' next to 'Proteins' in each PIN file
    by directly modifying the file as text.
    
    Args:
        pin_file_list_path: Path to the text file containing PIN file paths
    """
    # Read the list of PIN files
    with open(pin_file_list_path, 'r') as f:
        pin_files = [line.strip() for line in f.readlines()]
    
    processed_count = 0
    
    # Process each PIN file
    for pin_file in pin_files:
        try:
            # Check if file exists
            if not os.path.exists(pin_file):
                print(f"File not found: {pin_file}")
                continue
                
            # Read the entire file content
            with open(pin_file, 'r') as f:
                content = f.read()
            
            # Replace 'Proteins' with 'Proteins\tProteins2' in the header line
            # Only in the first occurrence (header line)
            if 'Proteins' in content:
                # Find the first line with 'Proteins'
                lines = content.split('\n')
                for i, line in enumerate(lines):
                    if 'Proteins' in line:
                        # Replace 'Proteins' with 'Proteins\tProteins2'
                        lines[i] = line.replace('Proteins', 'Proteins\tProteins2', 1)
                        
                        # Write the modified content back to file
                        with open(pin_file, 'w') as f:
                            f.write('\n'.join(lines))
                        
                        print(f"Added 'Proteins2' column to {pin_file}")
                        processed_count += 1
                        break
            else:
                print(f"'Proteins' column not found in {pin_file}")
        
        except Exception as e:
            print(f"Error processing {pin_file}: {str(e)}")
    
    print(f"\nProcessed {processed_count} files out of {len(pin_files)}")

# Example usage
if __name__ == "__main__":
    pin_file_list_path = "output/pin_files.txt"
    add_proteins2_column(pin_file_list_path)