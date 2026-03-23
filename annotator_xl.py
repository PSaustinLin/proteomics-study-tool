import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple, Union
import pandas as pd
import io
from PIL import Image

cid = False
consider_plus_zero_signature = False # 20260225 update
black = False

def combine_figures_in_memory(peptide_fig, spectrum_fig):
    # Convert matplotlib figures to PIL Images
    def fig_to_pil_image(fig):
        # Save figure to a bytes buffer
        buf = io.BytesIO()
        fig.savefig(buf, format='png', dpi=600, bbox_inches='tight', pad_inches=0, transparent=True)
        buf.seek(0)
        return Image.open(buf)
    
    # Convert figures to images
    peptide_img = fig_to_pil_image(peptide_fig)
    spectrum_img = fig_to_pil_image(spectrum_fig)
    
    # Calculate total height and width
    total_width = max(peptide_img.width, spectrum_img.width)
    total_height = peptide_img.height + spectrum_img.height
    
    # Create a new blank image
    combined_img = Image.new('RGBA', (total_width, total_height), color=(255, 255, 255, 0))
    #combined_img = Image.new('RGB', (total_width, total_height), color=(255, 255, 255))
    
    # Paste images
    combined_img.paste(peptide_img, ((total_width - peptide_img.width) // 2, 0))
    combined_img.paste(spectrum_img, ((total_width - spectrum_img.width) // 2, peptide_img.height))
    
    return combined_img

class CrosslinkedMS2Annotator:
    def __init__(self, 
                 ms2_file: str, 
                 alpha_sequence: str, 
                 beta_sequence: str,
                 scan_number: int, 
                 alpha_modifications: Dict[int, float] = None,
                 beta_modifications: Dict[int, float] = None,
                 alpha_crosslink_site: int = None,
                 beta_crosslink_site: int = None,
                 crosslinker_mass: float = -2.01565,
                 tolerance: Union[float, str] = 0.5):
        
        self.spectrum_data = self.read_ms2_spectrum(ms2_file, scan_number)
        self.alpha_sequence = alpha_sequence
        self.beta_sequence = beta_sequence
        self.scan_number = scan_number
        self.alpha_modifications = alpha_modifications or {}
        self.beta_modifications = beta_modifications or {}
        self.alpha_crosslink_site = alpha_crosslink_site
        self.beta_crosslink_site = beta_crosslink_site
        self.crosslinker_mass = crosslinker_mass
        self.tolerance = tolerance
        
        # Amino acid masses (monoisotopic)
        self.aa_masses = {
            'A': 71.03711, 'C': 103.00918, 'D': 115.02694, 'E': 129.04259,
            'F': 147.06841, 'G': 57.02146, 'H': 137.05891, 'I': 113.08406,
            'K': 128.09496, 'L': 113.08406, 'M': 131.04048, 'N': 114.04293,
            'P': 97.05276, 'Q': 128.05858, 'R': 156.10111, 'S': 87.03203,
            'T': 101.04768, 'V': 99.06841, 'W': 186.07931, 'Y': 163.06333
        }
        
        # Constant masses
        self.proton_mass = 1.00728
        self.water_mass = 18.01056
        
        # Signature ion masses
        self.signature_types = {
            #'': 0.0, if signature ion has a mass shift of 0, set consider_plus_zero_signature = True. DO NOT add a signature of +0 here!!!
            '-2': -2.01565,
            '+32': 31.97207,
            '-34': -33.98772,
            # double cysteines
            #'-4': -4.0313,
            #'+30': 29.95642,
            #'-36': -36.00337,
            #'+64': 63.94414,
            #-68': -67.97544,
            
            # triple cysteines
            #'-6': -6.04695,
            #'+96': 95.91621,
            #-102': -101.96316,
            #'+28': 27.94077,
            #'-38': -38.01902,
            #'+62': 61.92849,
            #'+70': 69.99109,

            #'': 0.0,
            #'-1': -1.00783, #-H
            #'-16': -15.99491, #-O
            #'-17': -17.00274, #-OH
            #'7': -17.02655, #-NH3
            #'8': -18.01056, #+H2O
            #'+18': 18.01056, #+H2O
            #'-18': -18.01056, #-H2O
            #'-28': -27.99491, #-CO
            #'-98': -97.97690, #-H3PO4
            #'-56': -56.06260, #-C4H8 for Leu/Ile
            #'-19': -19.0184, #-H3O
            #'-74': -74.01902, #-C3H6S for Met

        }
    
    def _calculate_tolerance(self, mz: float, tolerance: Union[float, str]) -> float:
        if isinstance(tolerance, str) and tolerance.lower().endswith('ppm'):
            # Convert ppm to absolute Da tolerance
            ppm_value = float(tolerance[:-3])
            return mz * ppm_value / 1e6
        
        # Default to absolute Da tolerance
        return float(tolerance)
    
    def read_ms2_spectrum(self, ms2_file_path: str, target_scan: int) -> Dict:
        PROTON = 1.007276  # Mass of proton
        
        with open(ms2_file_path, 'r') as ms2_file:
            current_scan = None
            peaks = []
            charge = 1
            precursor_mz = 0.0
            found_target = False
            precursor_mass = 0.0  # Initialize precursor_mass
            
            for line in ms2_file:
                line = line.strip()
                
                if not line:
                    continue
                    
                # Start of new scan
                if line.startswith('S'):
                    # Check if we've finished reading our target scan
                    if found_target:
                        break
                        
                    parts = line.split('\t')
                    current_scan = int(parts[1])
                    precursor_mz = float(parts[3])
                    
                    # Check if this is our target scan
                    if current_scan == target_scan:
                        found_target = True
                        peaks = []  # Reset peaks for new scan
                    
                # Skip header and info lines if not in target scan
                elif not found_target:
                    continue
                    
                # Process charge state for target scan
                elif line.startswith('Z') and found_target:
                    parts = line.split('\t')
                    charge = int(parts[1])
                    precursor_mass = float(parts[2]) - 1.00728
                    
                # Process peaks for target scan
                elif found_target and not line.startswith(('H', 'I', 'S', 'Z')):
                    try:
                        mz, intensity = map(float, line.split()[:2])
                        peaks.append((mz, intensity))
                    except ValueError:
                        continue
        
        if not found_target:
            raise ValueError(f"Scan {target_scan} not found in MS2 file")
        
        # Convert peaks to separate mz and intensity arrays
        mz_array, intensity_array = zip(*peaks) if peaks else ([], [])
        
        return {
            'mz array': np.array(mz_array),
            'intensity array': np.array(intensity_array),
            'params': {
                'charge': [charge],  # List to match MGF format
                'precursor mass': precursor_mass,
                'precursor mz': precursor_mz,
                'scans': str(target_scan)
            }
        }

    def _calculate_peptide_mass(self, sequence, modifications, crosslink_site=None, 
                                 crosslinker_mass=-2.01565, whole_beta_mass=0):
        peptide_mass = self.water_mass
        for i, aa in enumerate(sequence):
            # Base amino acid mass
            peptide_mass += self.aa_masses[aa]
            
            # Add modification if exists
            if i+1 in modifications:
                peptide_mass += modifications[i+1]
            
            # Add crosslinker mass for specific fragment ions
            if crosslink_site is not None:
                if (i+1 == crosslink_site):
                    peptide_mass += crosslinker_mass
                    peptide_mass += whole_beta_mass
        
        return peptide_mass

    def calculate_theoretical_crosslinked_fragments(self, precursor_charge: int) -> Tuple[List[float], List[float]]:
        # Calculate whole peptide masses for use in crosslink mass calculation
        alpha_whole_mass = self._calculate_peptide_mass(
            self.alpha_sequence, 
            self.alpha_modifications
        )
        beta_whole_mass = self._calculate_peptide_mass(
            self.beta_sequence, 
            self.beta_modifications
        )
        max_charge = min(precursor_charge - 1, 6)
        #max_charge = min(precursor_charge, 6)
        
        alpha_b_ions, alpha_y_ions = [], []
        beta_b_ions, beta_y_ions = [], []
        alpha_sig_ions, beta_sig_ions = [], []
        precursor_ions = []
        
        # Calculate alpha b-ions
        alpha_b_mass = 0
        for i, aa in enumerate(self.alpha_sequence[:-1]):
            alpha_b_mass += self.aa_masses[aa]
            # Add modification if exists
            if i+1 in self.alpha_modifications:
                alpha_b_mass += self.alpha_modifications[i+1]

            crosslink_b_mass = alpha_b_mass
            # Crosslinked fragment condition
            if (self.alpha_crosslink_site is not None and i+1 >= self.alpha_crosslink_site):
                # Add crosslinker and beta peptide mass
                crosslink_b_mass = (
                    alpha_b_mass + 
                    self.crosslinker_mass + 
                    beta_whole_mass
                )

            # Calculate ions with different crosslink modifications
            for charge in range(1, max_charge + 1):
                b_ion_mz = (crosslink_b_mass + charge * self.proton_mass) / charge
                alpha_b_ions.append((b_ion_mz, i+1, charge, f'αb', ''))
                
                # Non-crosslinked fragment
                if not cid and self.alpha_crosslink_site is not None and i+1 >= self.alpha_crosslink_site:
                    for sig_type, sig_mass in self.signature_types.items():
                        non_crosslink_b_mass = alpha_b_mass + sig_mass
                        b_ion_mz = (non_crosslink_b_mass + charge * self.proton_mass) / charge
                        alpha_b_ions.append((b_ion_mz, i+1, charge, f'αb', f'{sig_type}'))

        # Calculate alpha y-ions
        alpha_y_mass = self.water_mass
        for i in range(len(self.alpha_sequence)-1, 0, -1):
            alpha_y_mass += self.aa_masses[self.alpha_sequence[i]]

            # Add modification if exists
            if i+1 in self.alpha_modifications:
                alpha_y_mass += self.alpha_modifications[i+1]

            crosslink_y_mass = alpha_y_mass
            # Crosslinked fragment condition
            if (self.beta_crosslink_site is not None and i < self.alpha_crosslink_site):
                # Add crosslinker and beta peptide mass
                crosslink_y_mass = (
                    alpha_y_mass + 
                    self.crosslinker_mass + 
                    beta_whole_mass
                )

            # Calculate ions with different crosslink modifications
            for charge in range(1, max_charge + 1):
                y_ion_mz = (crosslink_y_mass + charge * self.proton_mass) / charge
                alpha_y_ions.append((y_ion_mz, len(self.alpha_sequence) - i, charge, f'αy', ''))
                
                # Non-crosslinked fragment
                if not cid and self.beta_crosslink_site is not None and i < self.alpha_crosslink_site:
                    for sig_type, sig_mass in self.signature_types.items():
                        non_crosslink_y_mass = alpha_y_mass + sig_mass
                        y_ion_mz = (non_crosslink_y_mass + charge * self.proton_mass) / charge
                        alpha_y_ions.append((y_ion_mz, len(self.alpha_sequence) - i, charge, f'αy', f'{sig_type}'))

        # Calculate beta b-ions
        beta_b_mass = 0
        for i, aa in enumerate(self.beta_sequence[:-1]):
            beta_b_mass += self.aa_masses[aa]
            # Add modification if exists
            if i+1 in self.beta_modifications:
                beta_b_mass += self.beta_modifications[i+1]

            crosslink_b_mass = beta_b_mass
            # Crosslinked fragment condition
            if (self.beta_crosslink_site is not None and i+1 >= self.beta_crosslink_site):
                    # Add crosslinker and alpha peptide mass
                    crosslink_b_mass = (
                        beta_b_mass + 
                        self.crosslinker_mass + 
                        alpha_whole_mass
                    )

            # Calculate ions with different crosslink modifications
            for charge in range(1, max_charge + 1):
                b_ion_mz = (crosslink_b_mass + charge * self.proton_mass) / charge
                beta_b_ions.append((b_ion_mz, i+1, charge, f'βb', ''))
                
                # Non-crosslinked fragment
                if not cid and self.beta_crosslink_site is not None and i+1 >= self.beta_crosslink_site:
                    for sig_type, sig_mass in self.signature_types.items():
                        non_crosslink_b_mass = beta_b_mass + sig_mass
                        b_ion_mz = (non_crosslink_b_mass + charge * self.proton_mass) / charge
                        beta_b_ions.append((b_ion_mz, i+1, charge, f'βb', f'{sig_type}'))
        
        # Calculate beta y-ions
        beta_y_mass = self.water_mass
        for i in range(len(self.beta_sequence)-1, 0, -1):
            beta_y_mass += self.aa_masses[self.beta_sequence[i]]

            # Add modification if exists
            if i+1 in self.beta_modifications:
                beta_y_mass += self.beta_modifications[i+1]
                
            crosslink_y_mass = beta_y_mass
            # Crosslinked fragment condition
            if (self.beta_crosslink_site is not None and i < self.beta_crosslink_site):
                    # Add crosslinker and alpha peptide mass
                    for sig_type, sig_mass in self.signature_types.items():
                        crosslink_y_mass = (
                            beta_y_mass + 
                            self.crosslinker_mass + 
                            alpha_whole_mass
                        )

            # Calculate ions with different crosslink modifications
            for charge in range(1, max_charge + 1):
                y_ion_mz = (crosslink_y_mass + charge * self.proton_mass) / charge
                beta_y_ions.append((y_ion_mz, len(self.beta_sequence) - i, charge, f'βy', ''))
                
                # Non-crosslinked fragment
                if not cid and self.beta_crosslink_site is not None and i < self.beta_crosslink_site:
                    for sig_type, sig_mass in self.signature_types.items():
                        non_crosslink_y_mass = beta_y_mass + sig_mass
                        y_ion_mz = (non_crosslink_y_mass + charge * self.proton_mass) / charge
                        beta_y_ions.append((y_ion_mz, len(self.beta_sequence) - i, charge, f'βy', f'{sig_type}'))

        # whole alpha signature ions
        for sig_type, sig_mass in self.signature_types.items():
            for charge in range(1, max_charge + 1):
                non_crosslink_mass = alpha_whole_mass + sig_mass
                ion_mz = (non_crosslink_mass + charge * self.proton_mass) / charge
                # Don't use format string here, pass the actual sig_type
                alpha_sig_ions.append((ion_mz, '', charge, f'α', f'{sig_type}'))

        # whole beta signature ions
        for sig_type, sig_mass in self.signature_types.items():
            for charge in range(1, max_charge + 1):
                non_crosslink_mass = beta_whole_mass + sig_mass
                ion_mz = (non_crosslink_mass + charge * self.proton_mass) / charge
                # Don't use format string here, pass the actual sig_type
                beta_sig_ions.append((ion_mz, '', charge, f'β', f'{sig_type}'))

        # whole precursor signature ions
        for charge in range(1, max_charge + 1):
            non_crosslink_b_mass = alpha_whole_mass + beta_whole_mass + self.crosslinker_mass
            b_ion_mz = (non_crosslink_b_mass + charge * self.proton_mass) / charge
            precursor_ions.append((b_ion_mz, '', charge, f'M', f'{sig_type}'))

        if consider_plus_zero_signature:
            for charge in range(1, max_charge + 1):
                non_crosslink_mass = alpha_whole_mass
                ion_mz = (non_crosslink_mass + charge * self.proton_mass) / charge
                alpha_sig_ions.append((ion_mz, '', charge, f'α', ''))
                non_crosslink_mass = beta_whole_mass
                ion_mz = (non_crosslink_mass + charge * self.proton_mass) / charge
                beta_sig_ions.append((ion_mz, '', charge, f'β', ''))

        return alpha_b_ions, alpha_y_ions, beta_b_ions, beta_y_ions, alpha_sig_ions, beta_sig_ions, precursor_ions
    
    def annotate_crosslinked_spectrum(self, output_file: str = None, csv_output: str = None):
        # Get spectrum data
        mz_values = self.spectrum_data['mz array']
        intensity_values = self.spectrum_data['intensity array']
        
        # Normalize intensities to percentage
        max_intensity = np.max(intensity_values)
        normalized_intensities = (intensity_values / max_intensity) * 100
        
        # Get precursor charge from spectrum
        precursor_charge = int(self.spectrum_data['params']['charge'][0])
        
        # Calculate theoretical fragments
        alpha_b_ions, alpha_y_ions, beta_b_ions, beta_y_ions, alpha_sig_ions, beta_sig_ions, precursor_ions = self.calculate_theoretical_crosslinked_fragments(precursor_charge)
        
        # Create figure with two subplots (peptide sequence on top)
        fig_spectrum = plt.figure(figsize=(15, 7))
        ax_spectrum = fig_spectrum.add_subplot(111)
        fig_peptide = plt.figure(figsize=(15, 2))
        ax_peptide = fig_peptide.add_subplot(111)
        # Remove all spines and ticks from peptide subplot
        ax_peptide.set_xticks([])
        ax_peptide.set_yticks([])
        for spine in ax_peptide.spines.values():
            spine.set_visible(False)
        
        # Track matched peaks for coloring and peptide annotation
        matched_alpha_b_ions, matched_alpha_y_ions, matched_beta_b_ions, matched_beta_y_ions = set(), set(), set(), set()
        matched_alpha_sig_ions, matched_beta_sig_ions, matched_precursor_ions = set(), set(), set()
        
        # Find all matched peaks first
        for ion_type, ions, ion_set in [('αb', alpha_b_ions, matched_alpha_b_ions), 
                                    ('αy', alpha_y_ions, matched_alpha_y_ions),
                                    ('βb', beta_b_ions, matched_beta_b_ions), 
                                    ('βy', beta_y_ions, matched_beta_y_ions),
                                    ('α', alpha_sig_ions, matched_alpha_sig_ions),
                                    ('β', beta_sig_ions, matched_beta_sig_ions),
                                    ('M', precursor_ions, matched_precursor_ions)]:
            for theoretical_mz, pos, charge, ion_label, sig_type in ions:
                abs_tolerance = self._calculate_tolerance(theoretical_mz, self.tolerance)
                peaks_within_tolerance = np.where(np.abs(mz_values - theoretical_mz) <= abs_tolerance)[0]
                if len(peaks_within_tolerance) > 0:
                    ion_set.add(pos)
        
        # Plot spectrum
        ax_spectrum.spines['top'].set_visible(False)
        ax_spectrum.spines['right'].set_visible(False)
        
        min_mz = np.min(mz_values)
        max_mz = np.max(mz_values)
        
        # Find matched and unmatched peaks
        unmatched_indices = []
        matched_indices = set()
        
        for i, mz in enumerate(mz_values):
            is_matched = False
            for ion_type, ions in [('αb', alpha_b_ions), ('αy', alpha_y_ions),
                                ('βb', beta_b_ions), ('βy', beta_y_ions),
                                ('α', alpha_sig_ions), ('β', beta_sig_ions),
                                ('M', precursor_ions)]:
                for theoretical_mz, pos, charge, ion_label, sig_type in ions:
                    abs_tolerance = self._calculate_tolerance(theoretical_mz, self.tolerance)
                    if abs(mz - theoretical_mz) <= abs_tolerance:
                        matched_indices.add(i)
                        is_matched = True
                        break
                if is_matched:
                    break
            if not is_matched:
                unmatched_indices.append(i)

        ax_spectrum.vlines(mz_values[unmatched_indices], 
                        0, 
                        normalized_intensities[unmatched_indices],
                        colors='lightgrey' if not black else 'black',
                        linewidth=1)

        ax_spectrum.set_xlim(min_mz, max_mz)
        ax_spectrum.set_ylim(0, 105)
        ax_spectrum.set_xlabel('m/z')
        ax_spectrum.set_ylabel('Relative intensity (%)')
        
        # Add precursor charge and additional information to upper left corner
        info_text = f'{precursor_charge}+\n'
        info_text += f'Mass: {self.spectrum_data["params"]["precursor mass"]:.4f}\n'
        
        # Add modifications info
        for pos, mass in self.alpha_modifications.items():
            aa = self.alpha_sequence[pos-1]
            info_text += f'α{aa}{pos}{mass:+.4f}\n'
        
        for pos, mass in self.beta_modifications.items():
            aa = self.beta_sequence[pos-1]
            info_text += f'β{aa}{pos}{mass:+.4f}'
        
        ax_spectrum.text(0.885, 0.98, info_text,
                        transform=ax_spectrum.transAxes,
                        fontsize=12,
                        verticalalignment='top')
        
        # Track used vertical positions
        used_positions = set()
        # Create list to store matched ion information
        matched_ions = []
        # Create list to store all annotations for sorting by intensity
        all_annotations = []

        # Annotate ions
        for ion_type, ions, color in [('αb', alpha_b_ions, '#37415D'), ('αy', alpha_y_ions, '#375D53'),
                            ('βb', beta_b_ions, '#905110'), ('βy', beta_y_ions, '#5D4037'),
                            ('α', alpha_sig_ions, '#4E1380'), ('β', beta_sig_ions, '#E18515'),
                            ('M', precursor_ions, 'grey')]:
            for theoretical_mz, pos, charge, ion_label, sig_type in ions:
                abs_tolerance = self._calculate_tolerance(theoretical_mz, self.tolerance)
                peaks_within_tolerance = np.where(np.abs(mz_values - theoretical_mz) <= abs_tolerance)[0]
                
                if len(peaks_within_tolerance) > 0:
                    # Find highest intensity peak within tolerance
                    highest_peak_idx = peaks_within_tolerance[np.argmax(intensity_values[peaks_within_tolerance])]
                    peak_mz = mz_values[highest_peak_idx]
                    peak_intensity = normalized_intensities[highest_peak_idx]
                    mass_error = peak_mz - theoretical_mz
                    
                    # No need to format ion label - it's already properly formatted when created
                    if peak_intensity >= 0:  # annotation threshold = 1%
                        if pos:
                            label = f'{ion_label}_{{{pos}}}{sig_type}^{{+{charge}}}'
                        else:
                            label = f'{ion_label}{sig_type}^{{+{charge}}}'
                        
                        formatted_label = f'$\\mathrm{{{label}}}$'
                        
                        all_annotations.append({
                            'mz': peak_mz,
                            'intensity': peak_intensity,
                            'label': formatted_label,
                            'color': color,
                            'theoretical_mz': theoretical_mz,
                            'mass_error': mass_error,
                            'ion_label': ion_label,
                            'sig_type': sig_type,
                            'position': pos,
                            'charge': charge
                        })

        # Sort annotations by m/z to handle overlapping labels
        all_annotations.sort(key=lambda x: x['mz'])
        
        # Calculate label positions using dynamic programming
        def calculate_label_positions(annotations, min_spacing=5):
            n = len(annotations)
            if n == 0:
                return []
            
            # Initialize positions at base height
            positions = [ann['intensity'] + 2 for ann in annotations]
            
            # Iterate through annotations to adjust heights
            for i in range(n):
                for j in range(i):
                    # Check if labels might overlap
                    mz_diff = annotations[i]['mz'] - annotations[j]['mz']
                    if mz_diff < 50:  # Adjust this value based on your label width
                        # If labels are close in m/z, ensure vertical separation
                        if abs(positions[i] - positions[j]) < min_spacing:
                            positions[i] = max(positions[i], positions[j] + min_spacing)
            
            return positions

        label_positions = calculate_label_positions(all_annotations)
        
        # Draw annotations
        for ann, y_pos in zip(all_annotations, label_positions):
            # Plot peak label
            ax_spectrum.text(ann['mz'], y_pos,
                            ann['label'],
                            ha='center',
                            va='bottom',
                            color=ann['color'],
                            fontsize=14)
            
            # Plot m/z value
            ax_spectrum.text(ann['mz'], y_pos - 1.5, #原本-1
                            f'{ann["mz"]:.4f}',
                            ha='center',
                            va='bottom',
                            color='grey',
                            fontsize=5)
            
            # Plot color dots
            #ax_spectrum.plot(ann['mz'], y_pos - 2, marker='.', color=ann['color'])

            # Plot matched peaks
            ax_spectrum.vlines(ann['mz'], 0, 
                            ann['intensity'],
                            ann['color'] if not black else 'black',
                            linewidth=1)
            
            """
            # Draw connection line if label is lifted
            if y_pos > ann['intensity'] + 10:
                ax_spectrum.plot([ann['mz'], ann['mz']],
                            [ann['intensity'], y_pos - 1],
                            color='grey',
                            linestyle=':',
                            linewidth=1)
            """
            # Store matched ion information
            matched_ions.append({
                'Ion Type': ann['ion_label'].replace('α', 'A').replace('β', 'B'),
                'Signature': ann['sig_type'],
                'Position': ann['position'] if ann['position'] != '' else '',
                'Charge': ann['charge'],
                'Theoretical m/z': ann['theoretical_mz'],
                'Observed m/z': ann['mz'],
                'Intensity (%)': ann['intensity'],
                'Mass Error (Da)': ann['mass_error'],
                'Fragment Sequence': self._get_fragment_sequence(ann['ion_label'], ann['position'])
            })
        
        # Plot peptide sequences
        alpha_seq_length = len(self.alpha_sequence)
        beta_seq_length = len(self.beta_sequence)
        
        ax_peptide.set_ylim(0, 1)
        alpha_y_position = 0.9
        beta_y_position = 0.4
        y_offset = 0.1
        
        # Calculate arm lengths for both peptides
        alpha_left_arm = self.alpha_crosslink_site - 1
        alpha_right_arm = len(self.alpha_sequence) - self.alpha_crosslink_site
        alpha_longer_arm = max(alpha_left_arm, alpha_right_arm)
        
        beta_left_arm = self.beta_crosslink_site - 1
        beta_right_arm = len(self.beta_sequence) - self.beta_crosslink_site
        beta_longer_arm = max(beta_left_arm, beta_right_arm)
        
        # Calculate the ideal center position for each peptide
        alpha_ideal_center = (alpha_left_arm + alpha_right_arm) / 2
        beta_ideal_center = (beta_left_arm + beta_right_arm) / 2
        
        # Calculate offset from crosslink site to ideal center
        alpha_offset = alpha_ideal_center - self.alpha_crosslink_site
        beta_offset = beta_ideal_center - self.beta_crosslink_site
        
        # Calculate positions for centered alignment
        aa_spacing = 50  # Fixed spacing in pixels
        display_box = ax_spectrum.get_window_extent()
        spectrum_width_pixels = display_box.width
        
        # Get spectrum limits
        spectrum_xlim = [min_mz, max_mz]
        spectrum_xlim[1] += (spectrum_xlim[1] - spectrum_xlim[0]) * 0.1  # Add 10% more space
        ax_spectrum.set_xlim(spectrum_xlim)
        
        # Calculate center position of the plot
        center_pos = spectrum_xlim[0] + (spectrum_xlim[1] - spectrum_xlim[0]) * 0.5
        
        # Calculate start positions for both sequences to align crosslink sites vertically
        # and center the longer arms
        sequence_unit = aa_spacing / spectrum_width_pixels * (spectrum_xlim[1] - spectrum_xlim[0])
        
        # Determine which peptide has the longer combined arms
        max_total_length = max(alpha_left_arm + alpha_right_arm, beta_left_arm + beta_right_arm)
        
        # Position both sequences so their crosslink sites align vertically at the center
        alpha_start = center_pos - (self.alpha_crosslink_site - 1) * sequence_unit
        beta_start = center_pos - (self.beta_crosslink_site - 1) * sequence_unit
        
        # Generate x positions for each amino acid
        alpha_x_positions = [alpha_start + i * sequence_unit for i in range(len(self.alpha_sequence))]
        beta_x_positions = [beta_start + i * sequence_unit for i in range(len(self.beta_sequence))]
        
        # Plot alpha sequence
        for i, (aa, x) in enumerate(zip(self.alpha_sequence, alpha_x_positions)):
            ax_peptide.text(x, alpha_y_position, aa, ha='center', va='center', fontsize=14)
            if i < alpha_seq_length - 1:
                if (i + 1) in matched_alpha_b_ions or (alpha_seq_length - (i + 1)) in matched_alpha_y_ions:
                    x_mid = x + (alpha_x_positions[1] - alpha_x_positions[0]) / 2
                    ax_peptide.plot([x_mid, x_mid],
                                [alpha_y_position + y_offset, alpha_y_position - y_offset],
                                color='lightgrey', linewidth=1)
        
        # Plot beta sequence
        for i, (aa, x) in enumerate(zip(self.beta_sequence, beta_x_positions)):
            ax_peptide.text(x, beta_y_position, aa, ha='center', va='center', fontsize=14)
            if i < beta_seq_length - 1:
                if (i + 1) in matched_beta_b_ions or (beta_seq_length - (i + 1)) in matched_beta_y_ions:
                    x_mid = x + (beta_x_positions[1] - beta_x_positions[0]) / 2
                    ax_peptide.plot([x_mid, x_mid],
                                [beta_y_position + y_offset, beta_y_position - y_offset],
                                color='lightgrey', linewidth=1)
        
        # Draw crosslinker line
        alpha_crosslink_x = alpha_x_positions[self.alpha_crosslink_site - 1]
        beta_crosslink_x = beta_x_positions[self.beta_crosslink_site - 1]
        ax_peptide.plot([alpha_crosslink_x, alpha_crosslink_x],
                    [0.5*(alpha_y_position + beta_y_position) - y_offset, 0.5*(alpha_y_position + beta_y_position) + y_offset],
                    color='grey', linewidth=2)
        
        # Add ion annotations
        for i, x in enumerate(alpha_x_positions[:-1], start=1):
            x_mid = x + (alpha_x_positions[1] - alpha_x_positions[0]) / 2
            
            if i in matched_alpha_b_ions:
                text_x = x_mid - (alpha_x_positions[1] - alpha_x_positions[0]) / 8
                ax_peptide.text(text_x, alpha_y_position + y_offset + 0.05,
                            f'b$_{{{i}}}$',
                            ha='center',
                            va='bottom',
                            color='#37415D',
                            fontsize=10)
                ax_peptide.plot([text_x, x_mid],
                            [alpha_y_position + y_offset, alpha_y_position + y_offset],
                            color='lightgrey', linewidth=1)

            y_pos = alpha_seq_length - i
            if y_pos in matched_alpha_y_ions:
                text_x = x_mid + (alpha_x_positions[1] - alpha_x_positions[0]) / 8
                ax_peptide.text(text_x, alpha_y_position - y_offset - 0.05,
                            f'y$_{{{y_pos}}}$',
                            ha='center',
                            va='top',
                            color='#375D53',
                            fontsize=10)
                ax_peptide.plot([x_mid, text_x],
                            [alpha_y_position - y_offset, alpha_y_position - y_offset],
                            color='lightgrey', linewidth=1)
        
        # annotations for beta sequence
        for i, x in enumerate(beta_x_positions[:-1], start=1):
            x_mid = x + (beta_x_positions[1] - beta_x_positions[0]) / 2
            
            if i in matched_beta_b_ions:
                text_x = x_mid - (beta_x_positions[1] - beta_x_positions[0]) / 8
                ax_peptide.text(text_x, beta_y_position + y_offset + 0.05,
                            f'b$_{{{i}}}$',
                            ha='center',
                            va='bottom',
                            color='#905110',
                            fontsize=10)
                ax_peptide.plot([text_x, x_mid],
                            [beta_y_position + y_offset, beta_y_position + y_offset],
                            color='lightgrey', linewidth=1)

            y_pos = beta_seq_length - i
            if y_pos in matched_beta_y_ions:
                text_x = x_mid + (beta_x_positions[1] - beta_x_positions[0]) / 8
                ax_peptide.text(text_x, beta_y_position - y_offset - 0.05,
                            f'y$_{{{y_pos}}}$',
                            ha='center',
                            va='top',
                            color='#5D4037',
                            fontsize=10)
                ax_peptide.plot([x_mid, text_x],
                            [beta_y_position - y_offset, beta_y_position - y_offset],
                            color='lightgrey', linewidth=1)
        
        # Determine the actual start and end positions used for sequences
        x_min = min(min(alpha_x_positions), min(beta_x_positions))
        x_max = max(max(alpha_x_positions), max(beta_x_positions))
        
        # Adjust x-limits for the peptide figure to match the exact used space
        ax_peptide.set_xlim(x_min - sequence_unit, x_max + sequence_unit)
        
        if output_file:
            # Combine figures in memory
            combined_img = combine_figures_in_memory(fig_peptide, fig_spectrum)
            
            # Save the combined image
            combined_img.save(output_file, dpi=(300, 300), transparent=True)
        else:
            # If no output file, just show the figures
            plt.show()
        
        # Close the figures to free up memory
        plt.close(fig_peptide)
        plt.close(fig_spectrum)

        # Save matched ions to CSV if path provided
        if csv_output and matched_ions:
            # Convert to DataFrame and sort by intensity
            df = pd.DataFrame(matched_ions)
            df = df.sort_values('Intensity (%)', ascending=False)
            
            # Round numerical columns
            df['Theoretical m/z'] = df['Theoretical m/z'].round(4)
            df['Observed m/z'] = df['Observed m/z'].round(4)
            df['Intensity (%)'] = df['Intensity (%)'].round(1)
            df['Mass Error (Da)'] = df['Mass Error (Da)'].round(4)
            
            # Save to CSV
            df.to_csv(csv_output, index=False)

    def _get_fragment_sequence(self, ion_label: str, pos: Union[int, str]) -> str:
        if pos == 'N/A' or pos == '':
            return ''
            
        if ion_label.startswith('α'):
            sequence = self.alpha_sequence
            if 'b' in ion_label:
                return sequence[:pos]
            elif 'y' in ion_label:
                return sequence[-pos:]
        elif ion_label.startswith('β'):
            sequence = self.beta_sequence
            if 'b' in ion_label:
                return sequence[:pos]
            elif 'y' in ion_label:
                return sequence[-pos:]
        
        return ''

if __name__ == "__main__":
    annotator = CrosslinkedMS2Annotator(
        ms2_file=r'C:\Crux\data\20250617\20250618-TYG-1_1555-60T.ms2',
        alpha_sequence='EPCVESLVSQYFQTVTDYGK',
        beta_sequence='LGMFNIQHCK',
        scan_number=25099,
        alpha_modifications={}, # 23: 15.99491
        beta_modifications={},
        alpha_crosslink_site=3,
        beta_crosslink_site=9,
        crosslinker_mass=-2.01565,
        #tolerance=0.1  # Specify tolerance in Da (0.5 for IT, 0.1 for OT with 15k resolution)
        tolerance='200ppm'  # Specify tolerance in ppm
    )
    output_dir = r'C:\Crux\xlOutput\AA'
    annotator.annotate_crosslinked_spectrum(
        output_file=r'{}\{}.png'.format(output_dir, annotator.scan_number),
        csv_output=r'{}\{}_matched_ions.csv'.format(output_dir, annotator.scan_number))