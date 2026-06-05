import os
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple, Union
import pandas as pd
import io
from PIL import Image
from concurrent.futures import ThreadPoolExecutor  # kept for possible future use

def combine_figures_in_memory(peptide_fig, spectrum_fig):
    # Convert matplotlib figures to PIL Images
    def fig_to_pil_image(fig):
        # Save figure to a bytes buffer
        buf = io.BytesIO()
        fig.savefig(buf, format='tiff', dpi=600, bbox_inches='tight', pad_inches=0, transparent=False)
        buf.seek(0)
        return Image.open(buf)
    
    # Convert figures to images
    peptide_img = fig_to_pil_image(peptide_fig)
    spectrum_img = fig_to_pil_image(spectrum_fig)
    
    # Calculate total height and width
    total_width = max(peptide_img.width, spectrum_img.width)
    total_height = (peptide_img.height + spectrum_img.height)
    
    # Create a new blank image
    #combined_img = Image.new('RGBA', (total_width, total_height), color=(255, 255, 255, 0))
    combined_img = Image.new('RGB', (total_width, total_height), color=(255, 255, 255))
    
    # Paste images
    combined_img.paste(peptide_img, ((total_width - peptide_img.width) // 2, 0))
    combined_img.paste(spectrum_img, ((total_width - spectrum_img.width) // 2, peptide_img.height))
    
    # --- Resize to 7.2 x 6 inches at 600dpi ---
    target_width_px = int(7.2 * 600)
    #target_height_px = int(6 * 600)
    target_height_px = int(target_width_px * total_height / total_width)

    combined_img = combined_img.resize((target_width_px, target_height_px), Image.LANCZOS)

    return combined_img

sig_on_by = True
black = True
consider_plus_zero_signature = False
show_ann_text = True
display_info = False
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
        
        # Constant masses
        self.proton_mass = 1.00728
        self.water_mass = 18.01056
        
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
        
        # Signature ion masses
        self.signature_types = {
            '': 0.0,
            '-2': -2.01565,
            '+32': 31.97207,
            '-34': -33.98772,
        }

        # Neutral loss masses
        self.nl_types = {
            'NH3': -17.02655,  # -NH3
            'H2O': -18.01056,  # -H2O
            '2H2O': -36.02112,  # -2H2O
        }
    
    def _calculate_tolerance(self, mz: float, tolerance: Union[float, str]) -> float:
        if isinstance(tolerance, str) and tolerance.lower().endswith('ppm'):
            # Convert ppm to absolute Da tolerance
            ppm_value = float(tolerance[:-3])
            return mz * ppm_value / 1e6
        
        # Default to absolute Da tolerance
        return float(tolerance)

    def _render_signature_label(self, sig_type: str) -> str:
        if sig_type == 'NH3':
            return '-NH_{3}'
        if sig_type == 'H2O':
            return '-H_{2}O'
        if sig_type == '2H2O':
            return '-2H_{2}O'
        return sig_type
    
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
                    iso_precursor_mass = precursor_mz * charge - charge * self.proton_mass
                    precursor_mass = float(parts[2]) - self.proton_mass
                    iso_num = round((iso_precursor_mass - precursor_mass) / self.proton_mass)
                    
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
                'scans': str(target_scan),
                'iso_num': iso_num,
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
        max_charge = min(precursor_charge, 6)
        #print(precursor_charge, max_charge)
        
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
            if (self.alpha_crosslink_site is not None and 
                i+1 >= self.alpha_crosslink_site):
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
                
                # NL ions: apply neutral loss to the full crosslinked mass (always, regardless of cid)
                for nl_type, nl_mass in self.nl_types.items():
                        nl_b_mass = crosslink_b_mass + nl_mass
                        b_ion_mz = (nl_b_mass + charge * self.proton_mass) / charge
                        alpha_b_ions.append((b_ion_mz, i+1, charge, f'αb', nl_type))

                # Non-crosslinked fragment
                if sig_on_by and self.alpha_crosslink_site is not None and i+1 >= self.alpha_crosslink_site:
                    for sig_type, sig_mass in self.signature_types.items():
                        non_crosslink_b_mass = alpha_b_mass + sig_mass
                        b_ion_mz = (non_crosslink_b_mass + charge * self.proton_mass) / charge
                        alpha_b_ions.append((b_ion_mz, i+1, charge, f'αb', sig_type))

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
                
                # NL ions: apply neutral loss to the full crosslinked mass (always, regardless of cid)
                for nl_type, nl_mass in self.nl_types.items():
                        nl_y_mass = crosslink_y_mass + nl_mass
                        y_ion_mz = (nl_y_mass + charge * self.proton_mass) / charge
                        alpha_y_ions.append((y_ion_mz, len(self.alpha_sequence) - i, charge, f'αy', nl_type))

                # Non-crosslinked fragment
                if sig_on_by and self.beta_crosslink_site is not None and i < self.alpha_crosslink_site:
                    for sig_type, sig_mass in self.signature_types.items():
                        non_crosslink_y_mass = alpha_y_mass + sig_mass
                        y_ion_mz = (non_crosslink_y_mass + charge * self.proton_mass) / charge
                        alpha_y_ions.append((y_ion_mz, len(self.alpha_sequence) - i, charge, f'αy', sig_type))

        # Calculate beta b-ions
        beta_b_mass = 0
        for i, aa in enumerate(self.beta_sequence[:-1]):
            beta_b_mass += self.aa_masses[aa]
            # Add modification if exists
            if i+1 in self.beta_modifications:
                beta_b_mass += self.beta_modifications[i+1]

            crosslink_b_mass = beta_b_mass
            # Crosslinked fragment condition
            if (self.beta_crosslink_site is not None and 
                    i+1 >= self.beta_crosslink_site):
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

                # NL ions: apply neutral loss to the full crosslinked mass (always, regardless of cid)
                for nl_type, nl_mass in self.nl_types.items():
                        nl_b_mass = crosslink_b_mass + nl_mass
                        b_ion_mz = (nl_b_mass + charge * self.proton_mass) / charge
                        beta_b_ions.append((b_ion_mz, i+1, charge, f'βb', nl_type))
                
                # Non-crosslinked fragment
                if sig_on_by and self.beta_crosslink_site is not None and i+1 >= self.beta_crosslink_site:
                    for sig_type, sig_mass in self.signature_types.items():
                        non_crosslink_b_mass = beta_b_mass + sig_mass
                        b_ion_mz = (non_crosslink_b_mass + charge * self.proton_mass) / charge
                        beta_b_ions.append((b_ion_mz, i+1, charge, f'βb', sig_type))
        
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
                crosslink_y_mass = (
                    beta_y_mass +
                    self.crosslinker_mass +
                    alpha_whole_mass
                )

            # Calculate ions with different crosslink modifications
            for charge in range(1, max_charge + 1):
                y_ion_mz = (crosslink_y_mass + charge * self.proton_mass) / charge
                beta_y_ions.append((y_ion_mz, len(self.beta_sequence) - i, charge, f'βy', ''))
                
                # NL ions: apply neutral loss to the full crosslinked mass (always, regardless of cid)
                for nl_type, nl_mass in self.nl_types.items():
                        nl_y_mass = crosslink_y_mass + nl_mass
                        y_ion_mz = (nl_y_mass + charge * self.proton_mass) / charge
                        beta_y_ions.append((y_ion_mz, len(self.beta_sequence) - i, charge, f'βy', nl_type))

                # Non-crosslinked fragment
                if sig_on_by and self.beta_crosslink_site is not None and i < self.beta_crosslink_site:
                    for sig_type, sig_mass in self.signature_types.items():
                        non_crosslink_y_mass = beta_y_mass + sig_mass
                        y_ion_mz = (non_crosslink_y_mass + charge * self.proton_mass) / charge
                        beta_y_ions.append((y_ion_mz, len(self.beta_sequence) - i, charge, f'βy', sig_type))

        # whole alpha signature ions
        for sig_type, sig_mass in self.signature_types.items():
            for charge in range(1, max_charge + 1):
                non_crosslink_mass = alpha_whole_mass + sig_mass
                ion_mz = (non_crosslink_mass + charge * self.proton_mass) / charge
                # Don't use format string here, pass the actual sig_type
                alpha_sig_ions.append((ion_mz, '', charge, f'α', sig_type))

        # whole beta signature ions
        for sig_type, sig_mass in self.signature_types.items():
            for charge in range(1, max_charge + 1):
                non_crosslink_mass = beta_whole_mass + sig_mass
                ion_mz = (non_crosslink_mass + charge * self.proton_mass) / charge
                # Don't use format string here, pass the actual sig_type
                beta_sig_ions.append((ion_mz, '', charge, f'β', sig_type))

        # whole precursor signature ions
        for charge in range(1, max_charge + 1):
            non_crosslink_b_mass = alpha_whole_mass + beta_whole_mass + self.crosslinker_mass
            b_ion_mz = (non_crosslink_b_mass + charge * self.proton_mass) / charge
            precursor_ions.append((b_ion_mz, '', charge, f'M', ''))
            
            for nl_type, nl_mass in self.nl_types.items():
                non_crosslink_b_mass = alpha_whole_mass + beta_whole_mass + self.crosslinker_mass + nl_mass
                b_ion_mz = (non_crosslink_b_mass + charge * self.proton_mass) / charge
                precursor_ions.append((b_ion_mz, '', charge, f'M', nl_type))

        if consider_plus_zero_signature:
            for charge in range(1, max_charge + 1):
                non_crosslink_mass = alpha_whole_mass
                ion_mz = (non_crosslink_mass + charge * self.proton_mass) / charge
                alpha_sig_ions.append((ion_mz, '', charge, f'α', ''))
                non_crosslink_mass = beta_whole_mass
                ion_mz = (non_crosslink_mass + charge * self.proton_mass) / charge
                beta_sig_ions.append((ion_mz, '', charge, f'β', ''))

        return alpha_b_ions, alpha_y_ions, beta_b_ions, beta_y_ions, alpha_sig_ions, beta_sig_ions, precursor_ions
    
    def _check_isotope_cluster(self,
                               theoretical_mz: float,
                               charge: int,
                               mz_values: np.ndarray,
                               normalized_intensities: np.ndarray,
                               abs_tolerance: float,
                               iso_num: int = 0):
        """
        Validate an isotope cluster for a candidate ion.

        For a candidate at *theoretical_mz* (charge *charge*), the M+0 anchor is
        tried at each isotope offset k = 0 … iso_num, where the anchor m/z for
        iteration k is ``theoretical_mz + k / charge``.  The cluster whose
        highest-intensity peak is greatest is kept.

        For a given anchor:
          - M+0  must be observed within *abs_tolerance*.
          - M+1  (anchor + 1/charge) must be observed within *abs_tolerance*.
          - M+2  (anchor + 2/charge) must be observed within *abs_tolerance*.

        **Charge-alias rejection** — ion is discarded if sub-isotope spacing
        peaks betray a wrong charge assignment:

          (t + 1/(2z)  AND  t + 3/(2z))   → real charge likely 2z
          OR
          (t + 1/(3z)  AND  t + 2/(3z))   → real charge likely 3z

        The boolean expression is exactly as written: the two peaks in each
        pair are joined by AND; the two pairs are joined by OR.

        The **assigned intensity** is the intensity of the highest-intensity peak
        within the M+0 … M+5 isotope cluster.

        At least one of the three isotope-peak intensities must be ≥ 30 % of the
        assigned intensity; otherwise the ion is rejected.

        Returns
        -------
        (result, rejection_reason)
            result           : 5-tuple (peak_mz, bar_mz, assigned_intensity,
                               mass_error, best_iso_num) on success, else None.
            rejection_reason : str describing why every candidate anchor was
                               rejected (empty string when result is not None).
        best_iso_num is the k value (0 ... iso_num) of the winning cluster.
        """
        iso_step = 1.0 / charge

        def _peak_present(target_mz: float) -> bool:
            return bool(np.any(np.abs(mz_values - target_mz) <= abs_tolerance))

        best_result = None   # (peak_mz, bar_mz, assigned_intensity, mass_error, k)
        per_k_reasons = []   # rejection reason for each k that failed

        for k in range(iso_num + 1):
            anchor_mz = theoretical_mz + k * iso_step

            # M+0 for this anchor
            m0_idx = np.where(np.abs(mz_values - anchor_mz) <= abs_tolerance)[0]
            if len(m0_idx) == 0:
                closest_dist = float(np.min(np.abs(mz_values - anchor_mz)))
                per_k_reasons.append(
                    f'k={k}: no_M0 (anchor={anchor_mz:.4f}, '
                    f'closest_obs_dist={closest_dist:.4f} Da, tol={abs_tolerance:.4f} Da)'
                )
                continue
            nearest_idx = m0_idx[np.argmin(np.abs(mz_values[m0_idx] - anchor_mz))]
            peak_mz    = mz_values[nearest_idx]
            mass_error = peak_mz - theoretical_mz

            # M+1 to M+5
            isotope_indices   = [nearest_idx]
            any_isotope_found = 0
            missing_isotopes  = []
            for j in range(1, 6):
                mj_theo = anchor_mz + j * iso_step
                mj_idx  = np.where(np.abs(mz_values - mj_theo) <= abs_tolerance)[0]
                if len(mj_idx) != 0:
                    mj_nearest = mj_idx[np.argmin(np.abs(mz_values[mj_idx] - mj_theo))]
                    isotope_indices.append(mj_nearest)
                    if j == 1:
                        any_isotope_found += 1
                else:
                    isotope_indices.append(None)
                    if j == 1:
                        closest_dist = float(np.min(np.abs(mz_values - mj_theo)))
                        missing_isotopes.append(
                            f'M+{j}@{mj_theo:.4f}(closest={closest_dist:.4f} Da)'
                        )

            # Require both M+1 and M+2 to be present. -> at least M+1 is found
            if any_isotope_found < 1:
                per_k_reasons.append(
                    f'k={k}: missing_isotopes [{"; ".join(missing_isotopes)}]'
                )
                continue

            # Charge-alias rejection
            half  = iso_step / 2
            third = iso_step / 3

            half_alias      = (_peak_present(anchor_mz + half)
                               and _peak_present(anchor_mz + 3 * half))
            third_alias     = (_peak_present(anchor_mz + third)
                               and _peak_present(anchor_mz + 2 * third))
            preceding_alias = (_peak_present(anchor_mz - iso_step)
                               and _peak_present(anchor_mz - 2 * iso_step))

            if half_alias or third_alias or preceding_alias:
                reason_detail = '|'.join(filter(None, [
                    'half_spacing'   if half_alias      else '',
                    'third_spacing'  if third_alias     else '',
                    'preceding_peak' if preceding_alias else '',
                ]))
                per_k_reasons.append(f'k={k}: charge_alias [{reason_detail}]')
                continue

            # Highest peak in cluster
            found_indices       = [idx for idx in isotope_indices if idx is not None]
            cluster_intensities = normalized_intensities[np.array(found_indices)]
            highest_in_cluster  = found_indices[int(np.argmax(cluster_intensities))]
            bar_mz              = mz_values[highest_in_cluster]
            assigned_intensity  = normalized_intensities[highest_in_cluster]

            if best_result is None or assigned_intensity > best_result[2]:
                best_result = (peak_mz, bar_mz, assigned_intensity, mass_error, k)

        if best_result is not None:
            return best_result, ''
        return None, ' | '.join(per_k_reasons) if per_k_reasons else 'no_M0_any_k'

    def annotate_crosslinked_spectrum(self, output_file: str = None, csv_output: str = None, csv_input: str = None, show_ann_text: bool = True, theoretical_csv: str = None):
        # Get spectrum data
        mz_values = self.spectrum_data['mz array']
        intensity_values = self.spectrum_data['intensity array']
        
        # Normalize intensities to percentage
        max_intensity = np.max(intensity_values)
        normalized_intensities = (intensity_values / max_intensity) * 100
        
        # Get precursor charge from spectrum
        precursor_charge = int(self.spectrum_data['params']['charge'][0])

        def _cluster_bar(anchor_mz: float, charge: int,
                         fallback_intensity: float) -> tuple:
            """
            Return (bar_mz, bar_intensity): the observed peak with the highest
            intensity within the M+0 … M+5 isotope cluster anchored at
            *anchor_mz* for the given *charge*.  Uses a fixed 0.05 Da window
            per isotope peak.  Falls back to (anchor_mz, fallback_intensity)
            if no cluster peaks are found.
            """
            iso = 1.0 / max(charge, 1)
            found = []
            for k in range(6):   # M+0 … M+5
                mk  = anchor_mz + k * iso
                idx = np.where(np.abs(mz_values - mk) <= 0.05)[0]
                if len(idx) > 0:
                    found.append(int(idx[np.argmin(np.abs(mz_values[idx] - mk))]))
            if found:
                ci   = np.array(found)
                best = ci[int(np.argmax(normalized_intensities[ci]))]
                return float(mz_values[best]), float(normalized_intensities[best])
            return anchor_mz, fallback_intensity

        # Read ions from CSV if provided, otherwise calculate theoretical fragments
        if csv_input:
            # Read ions from CSV file
            df = pd.read_csv(csv_input, dtype={'Signature': str}, keep_default_na=False)
            
            # Convert CSV data to the format expected by annotation code
            all_annotations = []
            matched_alpha_b_ions, matched_alpha_y_ions = set(), set()
            matched_beta_b_ions, matched_beta_y_ions = set(), set()
            matched_alpha_sig_ions, matched_beta_sig_ions = set(), set()
            matched_precursor_ions = set()
            
            # Color mapping for ion types
            color_map = {
                'Ab': '#37415D', 'Ay': '#375D53',
                'Bb': '#905110', 'By': '#5D4037',
                'A': '#4E1380', 'B': '#E18515',
                'M': 'grey'
            }
            
            for _, row in df.iterrows():
                ion_type = row['Ion Type']
                _raw_pos = row['Position']
                if pd.notna(_raw_pos) and _raw_pos != '':
                    try:
                        position = int(float(_raw_pos))
                    except (ValueError, TypeError):
                        position = _raw_pos
                else:
                    position = ''
                charge = int(row['Charge'])
                theoretical_mz = float(row['Theoretical m/z'])
                observed_mz = float(row['Observed m/z'])
                intensity = float(row['Intensity (%)'])
                mass_error = float(row['Mass Error (Da)'])

                if pd.notna(row['Signature']) and row['Signature'] != '':
                    sig_type = str(row['Signature']).strip()
                    try:
                        sig_int = int(sig_type)
                        sig_type = f'+{sig_int}' if sig_int > 0 else str(sig_int)
                    except ValueError:
                        sig_type = sig_type
                else:
                    sig_type = ''
                
                # Convert ion type back to Greek letters for display
                ion_label = ion_type.replace('A', 'α').replace('B', 'β')
                
                # Add to matched ion sets for peptide annotation
                if ion_type == 'Ab' and position:
                    matched_alpha_b_ions.add(int(position))
                elif ion_type == 'Ay' and position:
                    matched_alpha_y_ions.add(int(position))
                elif ion_type == 'Bb' and position:
                    matched_beta_b_ions.add(int(position))
                elif ion_type == 'By' and position:
                    matched_beta_y_ions.add(int(position))
                elif ion_type == 'A':
                    matched_alpha_sig_ions.add('')
                elif ion_type == 'B':
                    matched_beta_sig_ions.add('')
                elif ion_type == 'M':
                    matched_precursor_ions.add('')
                
                # Read optional bend flag (1 = bend annotation away from nearest signal)
                bend = False
                no_elevation = True

                # Read optional manual x/y overrides for the label destination
                manual_x = None
                manual_y = None
                if 'x' in df.columns:
                    raw_x = row['x']
                    if pd.notna(raw_x) and str(raw_x).strip() != '':
                        try:
                            manual_x = float(raw_x)
                        except (ValueError, TypeError):
                            pass
                if 'y' in df.columns:
                    raw_y = row['y']
                    if pd.notna(raw_y) and str(raw_y).strip() != '':
                        try:
                            manual_y = float(raw_y)
                        except (ValueError, TypeError):
                            pass

                # Read connector line info stored by the HTML editor
                has_line = False
                line_end_x = None
                line_end_y = None
                if 'has_line' in df.columns:
                    raw_hl = row['has_line']
                    if pd.notna(raw_hl) and str(raw_hl).strip() != '':
                        try:
                            has_line = bool(int(float(raw_hl)))
                        except (ValueError, TypeError):
                            pass
                if has_line:
                    if 'line_end_x' in df.columns:
                        raw_lx = row['line_end_x']
                        if pd.notna(raw_lx) and str(raw_lx).strip() != '':
                            try:
                                line_end_x = float(raw_lx)
                            except (ValueError, TypeError):
                                pass
                    if 'line_end_y' in df.columns:
                        raw_ly = row['line_end_y']
                        if pd.notna(raw_ly) and str(raw_ly).strip() != '':
                            try:
                                line_end_y = float(raw_ly)
                            except (ValueError, TypeError):
                                pass

                # Highest-intensity peak in M+0 … M+5 cluster
                # → x/y coordinates for bar drawing and label placement
                bar_mz, bar_intensity = _cluster_bar(observed_mz, charge, intensity)

                # Create annotation entry
                if intensity >= 1:  # annotation threshold = 1%
                    rendered_sig = self._render_signature_label(sig_type)
                    if position:
                        label = f'{ion_label}_{{{position}}}{rendered_sig}^{{+{charge}}}'
                    else:
                        label = f'{ion_label}{rendered_sig}^{{+{charge}}}'
                    
                    formatted_label = f'$\\mathrm{{{label}}}$'
                    
                    all_annotations.append({
                        'mz': observed_mz,       # nearest observed m/z (label text / CSV)
                        'peak_mz': bar_mz,       # highest-intensity peak in M+0…M+5 cluster (bar x)
                        'intensity': bar_intensity,  # intensity at bar_mz (bar y)
                        'label': formatted_label,
                        'color': color_map.get(ion_type, 'black'),
                        'theoretical_mz': theoretical_mz,
                        'mass_error': mass_error,
                        'ion_label': ion_label,
                        'sig_type': sig_type,
                        'position': position,
                        'charge': charge,
                        'bend': bend,
                        'no_elevation': no_elevation,
                        'manual_x': manual_x,
                        'manual_y': manual_y,
                        'has_line': has_line,
                        'line_end_x': line_end_x,
                        'line_end_y': line_end_y,
                    })
            
            # Create matched_ions list from CSV data for potential output
            matched_ions = []
            for _, row in df.iterrows():
                matched_ions.append({
                    'Ion Type': row['Ion Type'],
                    'Signature': row['Signature'] if pd.notna(row['Signature']) else '',
                    'Position': row['Position'] if pd.notna(row['Position']) and row['Position'] != '' else '',
                    'Charge': row['Charge'],
                    'Theoretical m/z': row['Theoretical m/z'],
                    'Observed m/z': row['Observed m/z'],
                    'Intensity (%)': row['Intensity (%)'],
                    'Mass Error (Da)': row['Mass Error (Da)'],
                    'Fragment Sequence': row['Fragment Sequence'] if 'Fragment Sequence' in row else ''
                })

        else:
            # ── Calculate theoretical fragments ───────────────────────────────
            alpha_b_ions, alpha_y_ions, beta_b_ions, beta_y_ions, \
                alpha_sig_ions, beta_sig_ions, precursor_ions = \
                self.calculate_theoretical_crosslinked_fragments(precursor_charge)

            # ── Colour map keyed by ion_type string ───────────────────────────
            ion_color_map = {
                'αb': '#37415D', 'αy': '#375D53',
                'βb': '#905110', 'βy': '#5D4037',
                'α':  '#4E1380', 'β':  '#E18515',
                'M':  'grey',
            }

            # ── Flatten all ions into one list for parallel processing ────────
            # Each element: (ion_type_str, theoretical_mz, pos, charge,
            #                ion_label, sig_type, color)
            flat_ions = []
            for ion_type_str, ions in [
                    ('αb', alpha_b_ions), ('αy', alpha_y_ions),
                    ('βb', beta_b_ions),  ('βy', beta_y_ions),
                    ('α',  alpha_sig_ions), ('β', beta_sig_ions),
                    ('M',  precursor_ions)]:
                color = ion_color_map[ion_type_str]
                for theoretical_mz, pos, charge, ion_label, sig_type in ions:
                    flat_ions.append(
                        (ion_type_str, theoretical_mz, pos,
                         charge, ion_label, sig_type, color)
                    )

            # ── Read iso_num from spectrum params ─────────────────────────────
            spectrum_iso_num = int(self.spectrum_data['params'].get('iso_num', 0))

            # ── Export theoretical ions CSV (optional) ────────────────────────
            if theoretical_csv:
                theo_rows = []
                for ion_type_str, theoretical_mz, pos, charge, ion_label, sig_type, color in flat_ions:
                    theo_rows.append({
                        'Ion Type':        ion_label.replace('α', 'A').replace('β', 'B'),
                        'Signature':       sig_type,
                        'Position':        pos if pos != '' else '',
                        'Charge':          charge,
                        'Theoretical m/z': round(theoretical_mz, 4),
                        'Fragment Sequence': self._get_fragment_sequence(ion_label, pos),
                    })
                theo_df = pd.DataFrame(theo_rows)
                theo_df = theo_df.drop_duplicates(subset='Theoretical m/z', keep='first')
                theo_df = theo_df.sort_values(['Ion Type', 'Position', 'Charge'])
                theo_df.to_csv(theoretical_csv, index=False)
                print(f"[debug] Theoretical ions written to: {theoretical_csv}  ({len(theo_df)} unique m/z values)")

            # ── Per-ion worker (isotope-cluster validated) ────────────────────
            def _process_ion(task):
                (ion_type_str, theoretical_mz, pos,
                 charge, ion_label, sig_type, color) = task

                abs_tol = self._calculate_tolerance(theoretical_mz, self.tolerance)
                cluster, rejection_reason = self._check_isotope_cluster(
                    theoretical_mz, charge,
                    mz_values, normalized_intensities, abs_tol,
                    iso_num=spectrum_iso_num,
                )

                if cluster is None:
                    return None

                peak_mz, bar_mz, peak_intensity, mass_error, matched_iso_num = cluster

                if peak_intensity < 1:          # annotation threshold = 1 %
                    return None

                rendered_sig = self._render_signature_label(sig_type)
                if pos:
                    label = f'{ion_label}_{{{pos}}}{rendered_sig}^{{+{charge}}}'
                else:
                    label = f'{ion_label}{rendered_sig}^{{+{charge}}}'
                formatted_label = f'$\\mathrm{{{label}}}$'

                return {
                    'ion_type_str':   ion_type_str,
                    'mz':             peak_mz,
                    'peak_mz':        bar_mz,
                    'intensity':      peak_intensity,
                    'label':          formatted_label,
                    'color':          color,
                    'theoretical_mz': theoretical_mz,
                    'mass_error':     mass_error,
                    'ion_label':      ion_label,
                    'sig_type':       sig_type,
                    'position':       pos,
                    'charge':         charge,
                    'iso_num':        matched_iso_num,
                    'bend':           False,
                    'manual_x':       None,
                    'manual_y':       None,
                }

            # ── Process ions sequentially ─────────────────────────────────────
            # Threading (GIL) gives no benefit for numpy-heavy work; real
            # parallelism is achieved by running multiple scans in parallel
            # at the batch level (ProcessPoolExecutor in the batch script).
            raw_results = [_process_ion(task) for task in flat_ions]

            all_annotations = [r for r in raw_results if r is not None]

            # ── Same-cluster merge ────────────────────────────────────────────
            # If two matched ions A and B satisfy
            #   theoretical_mz_B ~ theoretical_mz_A + n / charge   (n = 1 or 2)
            # they belong to the same isotope cluster.  Relabel B as A so that
            # only A's identity (label, mz, mass_error) survives; the bar
            # coordinates (peak_mz / intensity = cluster highest peak) are left
            # unchanged.  The dedup step below then collapses them to one entry.
            # Only ions with the same charge are compared.
            all_annotations.sort(key=lambda r: r['theoretical_mz'])
            for i, ann_a in enumerate(all_annotations):
                iso_step_a = 1.0 / ann_a['charge']
                for ann_b in all_annotations[i + 1:]:
                    if ann_b['charge'] != ann_a['charge']:
                        continue
                    delta = ann_b['theoretical_mz'] - ann_a['theoretical_mz']
                    # Allow floating-point slop of 1e-4 Da
                    n = round(delta / iso_step_a)
                    if n < 1 or n > 2:
                        break   # sorted list: no later entry can qualify
                    if abs(delta - n * iso_step_a) < 1e-4:
                        # Relabel B as A: identity fields only
                        ann_b['theoretical_mz'] = ann_a['theoretical_mz']
                        ann_b['mz']             = ann_a['mz']
                        ann_b['mass_error']     = ann_a['mass_error']
                        ann_b['ion_label']      = ann_a['ion_label']
                        ann_b['ion_type_str']   = ann_a['ion_type_str']
                        ann_b['sig_type']       = ann_a['sig_type']
                        ann_b['position']       = ann_a['position']
                        ann_b['label']          = ann_a['label']
                        ann_b['color']          = ann_a['color']
                        ann_b['iso_num']        = ann_a['iso_num']

            # ── Deduplicate by observed peak_mz ──────────────────────────────
            # Multiple theoretical ions can match the same observed peak
            # (includes same-cluster pairs just merged above).
            # Priority rules (applied in order):
            #   1. If one candidate is NH3 and the other is H2O, always keep H2O.
            #   2. Otherwise keep the one with the smallest absolute mass error.
            _peak_best: dict = {}   # peak_mz -> annotation dict
            for ann in all_annotations:
                key = round(ann['peak_mz'], 4)
                existing = _peak_best.get(key)
                if existing is None:
                    _peak_best[key] = ann
                else:
                    # H2O beats NH3 regardless of mass error
                    existing_nl = existing.get('sig_type', '')
                    ann_nl      = ann.get('sig_type', '')
                    if existing_nl == 'NH3' and ann_nl == 'H2O':
                        _peak_best[key] = ann
                    elif existing_nl == 'H2O' and ann_nl == 'NH3':
                        pass  # keep existing H2O
                    elif abs(ann['mass_error']) < abs(existing['mass_error']):
                        _peak_best[key] = ann
            n_before = len(all_annotations)
            all_annotations = list(_peak_best.values())
            n_removed = n_before - len(all_annotations)
            if n_removed:
                print(f"[dedup] Removed {n_removed} annotation(s) sharing an observed "
                      f"peak with a better-matching ion (includes same-cluster merges).")

            # ── Derive matched-position sets from confirmed annotations ───────
            matched_alpha_b_ions,  matched_alpha_y_ions  = set(), set()
            matched_beta_b_ions,   matched_beta_y_ions   = set(), set()
            matched_alpha_sig_ions, matched_beta_sig_ions = set(), set()
            matched_precursor_ions = set()

            _set_map = {
                'αb': matched_alpha_b_ions, 'αy': matched_alpha_y_ions,
                'βb': matched_beta_b_ions,  'βy': matched_beta_y_ions,
                'α':  matched_alpha_sig_ions, 'β': matched_beta_sig_ions,
                'M':  matched_precursor_ions,
            }
            for ann in all_annotations:
                _set_map[ann['ion_type_str']].add(ann['position'])

            # ── Build CSV-output list ─────────────────────────────────────────
            matched_ions = []
            for ann in all_annotations:
                matched_ions.append({
                    'Ion Type':        ann['ion_label'].replace('α', 'A').replace('β', 'B'),
                    'Signature':       ann['sig_type'],
                    'Position':        ann['position'] if ann['position'] != '' else '',
                    'Charge':          ann['charge'],
                    'Theoretical m/z': ann['theoretical_mz'],
                    'Observed m/z':    ann['mz'],
                    'Intensity (%)':   ann['intensity'],
                    'Mass Error (Da)': ann['mass_error'],
                    'Iso Num':         ann.get('iso_num', 0),
                    'Fragment Sequence': self._get_fragment_sequence(
                        ann['ion_label'], ann['position']),
                })
        
        # Sort annotations by intensity
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
        
        # Plot spectrum
        ax_spectrum.spines['top'].set_visible(False)
        ax_spectrum.spines['right'].set_visible(False)
        
        min_mz = np.min(mz_values)
        max_mz = np.max(mz_values)
        
        # Find matched and unmatched peaks
        # When csv_input is used, only the ions listed in the CSV are "matched" —
        # every other peak (including ones whose ions were manually deleted) is unmatched.
        unmatched_indices = []
        matched_indices = set()
        
        if csv_input:
            # Build the set of matched m/z values directly from the annotations that
            # were loaded from the CSV.  Each annotation carries the observed m/z of
            # the peak it was matched to, so we use a small window (0.005 Da) to map
            # back to the raw spectrum index.
            csv_matched_mzs = np.array([ann['peak_mz'] for ann in all_annotations])
            for i, mz in enumerate(mz_values):
                if len(csv_matched_mzs) > 0 and np.min(np.abs(csv_matched_mzs - mz)) <= 0.005:
                    matched_indices.add(i)
                else:
                    unmatched_indices.append(i)
        else:
            # Derive matched peak indices directly from the isotope-cluster-
            # validated annotations — mirrors the CSV-path logic and avoids
            # a redundant O(peaks × ions) scan.
            ann_peak_mzs = np.array([ann['peak_mz'] for ann in all_annotations])
            for i, mz in enumerate(mz_values):
                if len(ann_peak_mzs) > 0 and np.min(np.abs(ann_peak_mzs - mz)) <= 0.005:
                    matched_indices.add(i)
                else:
                    unmatched_indices.append(i)

        ax_spectrum.vlines(mz_values[unmatched_indices], 
                        0, 
                        normalized_intensities[unmatched_indices],
                        colors='lightgrey' if not black else 'black',
                        linewidth=1)

        #ax_spectrum.set_xlim(min_mz, max_mz)
        ax_spectrum.set_xlim(350, 1300)
        ax_spectrum.set_ylim(0, 100) # default = 105
        ax_spectrum.set_xlabel('m/z', fontsize=14)
        ax_spectrum.set_ylabel('Relative intensity (%)', fontsize=14)
        ax_spectrum.tick_params(axis='both', which='major', labelsize=14)
        
        if display_info:
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
        #used_positions = set()

        # Sort annotations by m/z to handle overlapping labels
        all_annotations.sort(key=lambda x: x['mz'])
        
        # Separate annotations into normal and bend groups for independent layout
        # Each annotation carries: mz, intensity, bend flag.
        # For bend annotations the label is placed at a bent x offset and
        # a y level chosen to minimise height while avoiding overlap with
        # all already-placed labels (both normal and bend).

        # ------------------------------------------------------------------ #
        #  Shared label-position registry                                     #
        #  Every placed label is recorded as (x_label, y_label) so the       #
        #  overlap check is uniform across normal and bent annotations.       #
        # ------------------------------------------------------------------ #

        def _find_nearest_signal_mz(ann_mz, all_anns):
            """Return the m/z of the nearest OTHER annotated signal."""
            other_mzs = [a['mz'] for a in all_anns if a['mz'] != ann_mz]
            if not other_mzs:
                return None
            return min(other_mzs, key=lambda m: abs(m - ann_mz))

        def calculate_label_positions(annotations, placed_labels,
                                      min_spacing=10, mz_proximity=50):
            """
            Compute (x_label, y_label) for each annotation.

            If an annotation has manual_x and manual_y set, those values are
            used directly and registered in placed_labels so that auto-placed
            labels navigate around them.

            placed_labels : list of (x, y) mutated in-place.
            Returns list of (x_label, y_label, bend_dir_or_None).
            """
            def _coerce_finite(value):
                try:
                    numeric = float(value)
                except (TypeError, ValueError):
                    return None
                return numeric if np.isfinite(numeric) else None

            if not annotations:
                return []
            results = []
            for ann in annotations:
                ann_mz = _coerce_finite(ann.get('peak_mz'))
                if ann_mz is None:
                    ann_mz = _coerce_finite(ann.get('mz'))
                # If still None, fall back to 0 and skip overlap logic
                if ann_mz is None:
                    results.append((0, ann.get('intensity', 0) + 2, None))
                    continue
                ann_intensity = _coerce_finite(ann.get('intensity', 0)) or 0
                do_bend       = ann.get('bend', False)
                manual_x      = _coerce_finite(ann.get('manual_x'))
                manual_y      = _coerce_finite(ann.get('manual_y'))

                # --- Manual override: place label at the user-supplied coordinates ---
                if manual_x is not None and manual_y is not None:
                    x_label  = manual_x
                    y_label  = manual_y
                    diff     = x_label - ann_mz
                    bend_dir = None if abs(diff) < 1 else (1 if diff > 0 else -1)
                    placed_labels.append((x_label, y_label))
                    results.append((x_label, y_label, bend_dir))
                    continue

                # --- Determine x position of the label text ---
                if do_bend:
                    nearest_mz = _find_nearest_signal_mz(ann_mz, annotations)
                    if nearest_mz is not None and nearest_mz > ann_mz:
                        bend_dir = -1   # nearest is right → bend label left
                    else:
                        bend_dir = 1    # nearest is left (or no neighbour) → bend right
                    x_label = ann_mz + bend_dir * 30   # 30 m/z units sideways
                else:
                    bend_dir = None
                    x_label  = ann_mz

                # --- Determine base y (start from peak top + small gap) ---
                nearby_mask = np.abs(mz_values - ann_mz) <= 2
                nearby_max  = (np.max(normalized_intensities[nearby_mask])
                               if nearby_mask.any() else ann_intensity)
                base_y = max(ann_intensity + 2, nearby_max + 2)

                # --- For no_elevation, place at base_y without lifting ---
                if ann.get('no_elevation', False):
                    y_label = base_y
                else:
                    # --- Lift until no overlap with already-placed labels ---
                    y_label  = base_y
                    max_iter = 200
                    for _ in range(max_iter):
                        collision = False
                        for (px, py) in placed_labels:
                            if abs(px - x_label) < mz_proximity and abs(py - y_label) < min_spacing:
                                collision = True
                                break
                        if not collision:
                            break
                        y_label += min_spacing

                placed_labels.append((x_label, y_label))
                results.append((x_label, y_label, bend_dir))
            return results

        # Processing order: manual → normal → bend.
        # Manual annotations claim their exact positions first so that
        # auto-placed labels navigate around them.
        def _is_manual(a):
            return a.get('manual_x') is not None and a.get('manual_y') is not None

        if show_ann_text:
            manual_anns    = [a for a in all_annotations if _is_manual(a)]
            no_elevation_anns = [a for a in all_annotations if not _is_manual(a) and a.get('no_elevation', False)]
            normal_anns = [a for a in all_annotations if not _is_manual(a) and not a.get('bend', False) and not a.get('no_elevation', False)]
            bend_anns      = [a for a in all_annotations if not _is_manual(a) and a.get('bend', False)]

            placed_labels      = []
            manual_positions   = calculate_label_positions(manual_anns, placed_labels)
            no_elevation_positions = calculate_label_positions(no_elevation_anns, placed_labels)
            normal_positions = calculate_label_positions(normal_anns, placed_labels)
            bend_positions     = calculate_label_positions(bend_anns,   placed_labels)

            ann_to_pos = {}
            for ann, pos in zip(manual_anns,  manual_positions):
                ann_to_pos[id(ann)] = pos
            for ann, pos in zip(no_elevation_anns, no_elevation_positions):
                ann_to_pos[id(ann)] = pos
            for ann, pos in zip(normal_anns, normal_positions):
                ann_to_pos[id(ann)] = pos
            for ann, pos in zip(bend_anns, bend_positions):
                ann_to_pos[id(ann)] = pos

            label_positions = [ann_to_pos[id(ann)] for ann in all_annotations]
        else:
            label_positions = [(
                ann.get('peak_mz', ann['mz']),
                ann['intensity'] + 2,
                None
            ) for ann in all_annotations]

        # Draw annotations
        for ann, (x_label, y_label, bend_dir) in zip(all_annotations, label_positions):
            original_y_pos = ann['intensity'] + 2
            was_elevated = y_label > original_y_pos + 3

            # Plot peak label
            ha_align = 'center'
            if show_ann_text:
                ax_spectrum.text(x_label, y_label + 3,
                                ann['label'],
                                ha=ha_align,
                                va='bottom',
                                color=ann['color'],
                                fontsize=16)

            # Plot m/z value below ion label; top flush with y_label so center == y_label
            ax_spectrum.text(x_label, y_label,
                            f'{ann["mz"]:.2f}',
                            ha=ha_align,
                            va='bottom',
                            color='grey',
                            fontsize=10)

            # Plot matched peaks (bar at peak_mz — highest peak in M+0…M+5 cluster)
            ax_spectrum.vlines(ann['peak_mz'], 0,
                            ann['intensity'],
                            ann['color'] if not black else 'black',
                            linewidth=1)

            # Draw connection line
            is_manual = (ann.get('manual_x') is not None and ann.get('manual_y') is not None)
            has_line  = ann.get('has_line', False)
            line_end_x_val = ann.get('line_end_x')
            line_end_y_val = ann.get('line_end_y')

            if has_line and line_end_x_val is not None and line_end_y_val is not None:
                # Replicate the editor connector: peak tip → label bottom (data coords from CSV)
                ax_spectrum.plot([ann['peak_mz'], line_end_x_val],
                            [ann['intensity'], line_end_y_val],
                            color='grey',
                            linestyle=':',
                            linewidth=1)
            elif ann.get('no_elevation', False):
                # no_elevation (CSV path without line info): no dotted line
                pass
            elif is_manual or (ann.get('bend', False) and bend_dir is not None):
                # Diagonal dotted line from peak tip to label base
                ax_spectrum.plot([ann['peak_mz'], x_label],
                            [ann['intensity'], y_label - 1],
                            color='grey',
                            linestyle=':',
                            linewidth=1)
            elif was_elevated:
                # Vertical dotted line for elevated-but-not-bent labels
                ax_spectrum.plot([ann['peak_mz'], ann['peak_mz']],
                            [ann['intensity'], y_label - 1],
                            color='grey',
                            linestyle=':',
                            linewidth=1)
            
            # (matched_ions already populated before this drawing loop)
        
        # Plot peptide sequences
        alpha_seq_length = len(self.alpha_sequence)
        beta_seq_length = len(self.beta_sequence)
        
        ax_peptide.set_ylim(0, 1)
        alpha_y_position = 0.88
        beta_y_position = 0.3
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
        # spectrum_xlim = [min_mz, max_mz]
        spectrum_xlim = [350, 1400]
        #spectrum_xlim[1] += (spectrum_xlim[1] - spectrum_xlim[0]) * 0.1  # Add 10% more space
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
            ax_peptide.text(x, alpha_y_position, aa, ha='center', va='center', fontsize=22)
            if i < alpha_seq_length - 1:
                if (i + 1) in matched_alpha_b_ions or (alpha_seq_length - (i + 1)) in matched_alpha_y_ions:
                    x_mid = x + (alpha_x_positions[1] - alpha_x_positions[0]) / 2
                    ax_peptide.plot([x_mid, x_mid],
                                [alpha_y_position + y_offset, alpha_y_position - y_offset],
                                color='lightgrey', linewidth=1)
        
        # Plot beta sequence
        for i, (aa, x) in enumerate(zip(self.beta_sequence, beta_x_positions)):
            ax_peptide.text(x, beta_y_position, aa, ha='center', va='center', fontsize=22)
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
                    [0.5*(alpha_y_position + beta_y_position) - 0.75*y_offset, 0.5*(alpha_y_position + beta_y_position) + 0.75*y_offset],
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
                            fontsize=16)
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
                            fontsize=16)
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
                            fontsize=16)
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
                            fontsize=16)
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

            # Save the combined image (support TIFF output by extension)
            output_ext = output_file.lower().split('.')[-1]
            if output_ext in ('tif', 'tiff'):
                # Use TIFF format explicitly (can preserve RGBA and high quality)
                combined_img.save(output_file, format='TIFF', dpi=(600, 600), compression='tiff_deflate')
            else:
                combined_img.save(output_file, dpi=(600, 600), transparent=True)
        else:
            # If no output file, just show the figures
            plt.show()
        
        # Close the figures to free up memory
        plt.close(fig_peptide)
        plt.close(fig_spectrum)

        # ── Resolve the CSV output path ───────────────────────────────────────
        # When no csv_input was provided (fresh annotation mode) a CSV named
        # "{scan_number}_matched_ions.csv" is always written so the caller can
        # inspect matched ions without supplying an explicit csv_output.
        if csv_input is None:
            if output_file:
                _out_dir = os.path.dirname(os.path.abspath(output_file))
            else:
                _out_dir = os.getcwd()
            csv_output = os.path.join(_out_dir,
                                      f'{self.scan_number}_matched_ions.csv')

        # Save matched ions to CSV
        if csv_output and matched_ions:
            df = pd.DataFrame(matched_ions)
            df = df.sort_values('Intensity (%)', ascending=False)
            df['Theoretical m/z'] = df['Theoretical m/z'].round(4)
            df['Observed m/z']    = df['Observed m/z'].round(4)
            df['Intensity (%)']   = df['Intensity (%)'].round(1)
            df['Mass Error (Da)'] = df['Mass Error (Da)'].round(4)
            # Deduplicate by Theoretical m/z (keep first = highest intensity due to sort above)
            n_before = len(df)
            df = df.drop_duplicates(subset='Theoretical m/z', keep='first')
            n_removed = n_before - len(df)
            if n_removed > 0:
                print(f"[dedup] Removed {n_removed} duplicate row(s) with identical "
                      f"theoretical m/z from matched_ions CSV. "
                      f"The upstream duplicate-append bug has been fixed; "
                      f"if this message still appears, a new source of duplicates exists.")
            df.to_csv(csv_output, index=False)

        # ── Interactive HTML editor ──────────────────────────────────────────
        # Only generated when annotation text is shown AND ions come from a
        # CSV (so all matched ions are known and stable for the editor).
        if show_ann_text and csv_input:
            if output_file:
                base = output_file.rsplit('.', 1)[0]
                html_path = base + '_editor.html'
            else:
                import os as _os
                html_path = _os.path.join(
                    _os.path.dirname(csv_input),
                    f'scan_{self.scan_number}_editor.html'
                )
            self.export_interactive_html(html_output=html_path, csv_input=csv_input)

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

    def export_interactive_html(self, html_output: str, csv_input: str):
        """
        Generate a self-contained interactive HTML editor for annotation label
        positions.  Only called when show_ann_text is True AND csv_input is
        not None (enforced by annotate_crosslinked_spectrum).

        Features
        --------
        • Two-panel layout mirroring the matplotlib output: peptide diagram
          (top) + annotated spectrum (bottom), both drawn on Canvas.
        • All annotation labels are draggable HTML elements overlaid on the
          spectrum canvas.  Connector dotted lines update in real time.
        • "Pin to peak" button on each label removes the connector and places
          the text directly on the peak top (original simple style).
        • "Export PNG" renders a composite offscreen canvas (peptide + spectrum
          with labels baked in) at the same 15 × 9-inch proportions as the
          matplotlib TIFF and downloads it as a PNG.
        • "Export CSV" downloads a CSV with updated x/y columns that can be
          fed back as csv_input to reproduce the exact layout in matplotlib.
        """
        import json

        mz_values            = self.spectrum_data['mz array']
        intensity_values     = self.spectrum_data['intensity array']
        max_intensity        = np.max(intensity_values)
        normalized_intensities = (intensity_values / max_intensity) * 100
        precursor_charge     = int(self.spectrum_data['params']['charge'][0])

        color_map = {
            'Ab': '#37415D', 'Ay': '#375D53',
            'Bb': '#905110', 'By': '#5D4037',
            'A':  '#4E1380', 'B':  '#E18515',
            'M':  '#888888',
        }

        # ── Read CSV and build annotation list ───────────────────────────────
        df = pd.read_csv(csv_input, dtype={'Signature': str}, keep_default_na=False)
        all_annotations = []
        matched_alpha_b_ions, matched_alpha_y_ions = set(), set()
        matched_beta_b_ions,  matched_beta_y_ions  = set(), set()

        for _, row in df.iterrows():
            ion_type = row['Ion Type']
            _raw_pos = row['Position']
            position = ''
            if pd.notna(_raw_pos) and _raw_pos != '':
                try:
                    position = int(float(_raw_pos))
                except (ValueError, TypeError):
                    position = _raw_pos
            charge         = int(row['Charge'])
            theoretical_mz = float(row['Theoretical m/z'])
            observed_mz    = float(row['Observed m/z'])
            intensity      = float(row['Intensity (%)'])
            sig_type       = ''
            if pd.notna(row.get('Signature', '')) and str(row.get('Signature', '')).strip() != '':
                sig_type = str(row['Signature']).strip()

            # track matched ions for peptide diagram
            if ion_type == 'Ab' and position: matched_alpha_b_ions.add(int(position))
            elif ion_type == 'Ay' and position: matched_alpha_y_ions.add(int(position))
            elif ion_type == 'Bb' and position: matched_beta_b_ions.add(int(position))
            elif ion_type == 'By' and position: matched_beta_y_ions.add(int(position))

            ion_label    = ion_type.replace('A', 'α').replace('B', 'β')
            rendered_sig = self._render_signature_label(sig_type)
            label = (f'{ion_label}_{{{position}}}{rendered_sig}^{{+{charge}}}'
                     if position else f'{ion_label}{rendered_sig}^{{+{charge}}}')

            # Highest-intensity peak in M+0 … M+5 cluster
            _iso_html = 1.0 / max(charge, 1)
            _found_html = []
            for _k in range(6):
                _mk = observed_mz + _k * _iso_html
                _idx = np.where(np.abs(mz_values - _mk) <= 0.05)[0]
                if len(_idx) > 0:
                    _found_html.append(int(_idx[np.argmin(np.abs(mz_values[_idx] - _mk))]))
            if _found_html:
                _ci_html = np.array(_found_html)
                _best_html = _ci_html[int(np.argmax(normalized_intensities[_ci_html]))]
                bar_mz        = float(mz_values[_best_html])
                bar_intensity = float(normalized_intensities[_best_html])
            else:
                bar_mz, bar_intensity = observed_mz, intensity

            manual_x = manual_y = None
            if 'x' in df.columns:
                raw_x = row['x']
                if pd.notna(raw_x) and str(raw_x).strip() != '':
                    try: manual_x = float(raw_x)
                    except (ValueError, TypeError): pass
            if 'y' in df.columns:
                raw_y = row['y']
                if pd.notna(raw_y) and str(raw_y).strip() != '':
                    try: manual_y = float(raw_y)
                    except (ValueError, TypeError): pass

            if intensity >= 1:
                all_annotations.append({
                    'mz':            observed_mz,
                    'peak_mz':       bar_mz,
                    'intensity':     bar_intensity,
                    'label':         label,
                    'color':         color_map.get(ion_type, 'black'),
                    'ion_type':      ion_type,
                    'sig_type':      sig_type,
                    'position':      position,
                    'charge':        charge,
                    'theoretical_mz': theoretical_mz,
                    'mass_error':    float(observed_mz - theoretical_mz),
                    'manual_x':      manual_x,
                    'manual_y':      manual_y,
                })

        # ── Spectrum peaks for JS ─────────────────────────────────────────────
        spectrum_peaks = [
            {'mz': float(mz_values[i]), 'intensity': float(normalized_intensities[i])}
            for i in range(len(mz_values))
        ]
        matched_mzs = {ann['peak_mz'] for ann in all_annotations}

        # ── Default label positions: place directly on peak tip ───────────────
        all_annotations.sort(key=lambda a: a['peak_mz'])
        for ann in all_annotations:
            if ann['manual_x'] is not None and ann['manual_y'] is not None:
                ann['default_x'] = ann['manual_x']
                ann['default_y'] = ann['manual_y']
            else:
                ann['default_x'] = ann['peak_mz']
                ann['default_y'] = ann['intensity']

        # ── Peptide diagram data for JS ───────────────────────────────────────
        alpha_seq  = list(self.alpha_sequence)
        beta_seq   = list(self.beta_sequence)
        alpha_site = self.alpha_crosslink_site   # 1-based
        beta_site  = self.beta_crosslink_site    # 1-based
        peptide_data = {
            'alpha_seq':  alpha_seq,
            'beta_seq':   beta_seq,
            'alpha_site': alpha_site,
            'beta_site':  beta_site,
            'matched_alpha_b': list(matched_alpha_b_ions),
            'matched_alpha_y': list(matched_alpha_y_ions),
            'matched_beta_b':  list(matched_beta_b_ions),
            'matched_beta_y':  list(matched_beta_y_ions),
        }

        spectrum_json    = json.dumps(spectrum_peaks)
        annotations_json = json.dumps(all_annotations, default=str)
        matched_mzs_json = json.dumps(list(matched_mzs))
        peptide_json     = json.dumps(peptide_data)
        scan_number      = self.scan_number

        html_content = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<title>Annotation Editor \u2013 Scan {scan_number}</title>
<style>
* {{ box-sizing: border-box; margin: 0; padding: 0; }}
body {{ font-family: Arial, sans-serif; background: #f0f0f0; overflow-x: hidden; }}
#toolbar {{
  display: flex; align-items: center; gap: 8px; padding: 7px 12px;
  background: #fff; border-bottom: 1px solid #ccc; flex-wrap: wrap;
}}
#toolbar h2 {{ font-size: 13px; color: #333; white-space: nowrap; }}
button {{
  padding: 4px 11px; border: 1px solid #aaa; border-radius: 4px;
  cursor: pointer; font-size: 12px; background: #f5f5f5;
}}
button:hover {{ background: #e2e2e2; }}
#btn-export-csv  {{ background: #2a6496; color:#fff; border-color:#1a4f78; }}
#btn-export-csv:hover  {{ background:#1e4f78; }}
#btn-reset {{ background: #c0392b; color:#fff; border-color:#922b21; }}
#btn-reset:hover {{ background:#922b21; }}
label {{ font-size:12px; cursor:pointer; }}
#panels {{ display:flex; flex-direction:column; margin:8px; gap:0; }}
#peptide-wrap {{
  position:relative; background:#fff; border:1px solid #ccc;
  border-bottom:none; border-radius:4px 4px 0 0; overflow:hidden;
  z-index:1; pointer-events:none;
}}
#spectrum-wrap {{
  position:relative; background:#fff; border:1px solid #ccc;
  border-radius:0 0 4px 4px; overflow:visible; user-select:none;
  z-index:2;
}}
canvas {{ display:block; }}
.ann-label {{
  position:absolute; cursor:grab; white-space:nowrap;
  font-size:15px; padding:0 2px; border-radius:3px;
  border:1px solid transparent; line-height:1.1;
  z-index:200;
}}
.ann-label:hover {{ border-color:rgba(0,0,0,.22); background:rgba(255,255,255,.88); }}
.ann-label.dragging {{ cursor:grabbing; z-index:999; border-color:#555; background:rgba(255,255,255,.96); }}
.pin-btn {{
  display:none; position:absolute; top:-17px; left:50%;
  transform:translateX(-50%); font-size:9px; padding:1px 4px;
  background:#444; color:#fff; border:none; border-radius:2px;
  cursor:pointer; white-space:nowrap;
}}
.ann-label:hover .pin-btn {{ display:block; }}
</style>
</head>
<body>
<div id="toolbar">
  <h2>Annotation Editor &nbsp;|&nbsp; Scan {scan_number}</h2>
  <button id="btn-export-csv">\u2b07 Export CSV</button>
  <button id="btn-reset">\u21ba Reset positions</button>
  <label><input type="checkbox" id="chk-lines" checked> Connector lines</label>
</div>
<div id="panels">
  <div id="peptide-wrap"><canvas id="peptide-canvas"></canvas></div>
  <div id="spectrum-wrap"><canvas id="spectrum-canvas"></canvas></div>
</div>

<script>
// ── Data injected from Python ─────────────────────────────────────────────────
const SPECTRUM       = {spectrum_json};
const ANNOTATIONS_INIT = {annotations_json};
const MATCHED_MZS   = new Set({matched_mzs_json});
const PEP           = {peptide_json};

// ── Layout constants ──────────────────────────────────────────────────────────
// Spectrum panel (15 wide : 7 tall ratio in matplotlib)
const SM = {{ left:62, right:22, top:20, bottom:52 }};  // spectrum margins px
// Peptide panel (15 wide : 2 tall ratio)
const PM = {{ left:62, right:22, top:14, bottom:14 }};

const MZ_MIN=200, MZ_MAX=1400, INT_MIN=0, INT_MAX=100;

let sW,sH,sPlotW,sPlotH;  // spectrum canvas dims
let pW,pH,pPlotW,pPlotH;  // peptide canvas dims

let annotations = [];
let showLines   = true;

const specCanvas = document.getElementById('spectrum-canvas');
const specCtx    = specCanvas.getContext('2d');
const pepCanvas  = document.getElementById('peptide-canvas');
const pepCtx     = pepCanvas.getContext('2d');
const specWrap   = document.getElementById('spectrum-wrap');
const pepWrap    = document.getElementById('peptide-wrap');

// ── Coordinate helpers (spectrum) ─────────────────────────────────────────────
function mzToPx(mz)   {{ return SM.left + (mz-MZ_MIN)/(MZ_MAX-MZ_MIN)*sPlotW; }}
function intToPy(i)   {{ return SM.top  + (1-(i-INT_MIN)/(INT_MAX-INT_MIN))*sPlotH; }}
function pxToMz(px)   {{ return MZ_MIN  + (px-SM.left)/sPlotW*(MZ_MAX-MZ_MIN); }}
function pyToInt(py)  {{ return INT_MAX  - (py-SM.top)/sPlotH*(INT_MAX-INT_MIN); }}

// ── Draw peptide diagram ──────────────────────────────────────────────────────
function drawPeptide() {{
  pepCtx.clearRect(0,0,pW,pH);
  pepCtx.fillStyle='#fff';
  pepCtx.fillRect(0,0,pW,pH);

  const alphaSeq = PEP.alpha_seq;
  const betaSeq  = PEP.beta_seq;
  const mabI = new Set(PEP.matched_alpha_b);
  const mayI = new Set(PEP.matched_alpha_y);
  const mbbI = new Set(PEP.matched_beta_b);
  const mbyI = new Set(PEP.matched_beta_y);

  const totalAA    = Math.max(alphaSeq.length, betaSeq.length);
  // AA spacing: fill plot width evenly for the longer sequence
  const longestLen = Math.max(alphaSeq.length, betaSeq.length);
  const spacing    = pPlotW / (longestLen + 1);

  // crosslink site positions → centre column
  const alphaCrossCol = PEP.alpha_site - 1;  // 0-based index
  const betaCrossCol  = PEP.beta_site  - 1;

  // x origin such that crosslink columns align at the centre of the plot
  const centreX = PM.left + pPlotW / 2;
  const alphaX0 = centreX - alphaCrossCol * spacing;
  const betaX0  = centreX - betaCrossCol  * spacing;

  // Clamp to plot area
  function aaX(start, i) {{ return start + i * spacing; }}

  const alphaY = pH * 0.80;
  const betaY  = pH * 0.28;
  const yOff   = pH * 0.13;

  pepCtx.textAlign = 'center';

  // ── helper: draw tick (fragment bond mark) ────────────────────────────────
  function drawTick(x, yCenter, off) {{
    pepCtx.strokeStyle = '#ccc'; pepCtx.lineWidth = 1;
    pepCtx.beginPath();
    pepCtx.moveTo(x, yCenter - off);
    pepCtx.lineTo(x, yCenter + off);
    pepCtx.stroke();
  }}

  // ── alpha sequence ────────────────────────────────────────────────────────
  pepCtx.font = 'bold 15px Arial';
  alphaSeq.forEach((aa, i) => {{
    const x = aaX(alphaX0, i);
    pepCtx.fillStyle = '#222';
    pepCtx.fillText(aa, x, alphaY + 5);
    if (i < alphaSeq.length - 1) {{
      const xMid = x + spacing / 2;
      if (mabI.has(i+1) || mayI.has(alphaSeq.length-(i+1))) drawTick(xMid, alphaY, yOff);
    }}
  }});

  // ── beta sequence ─────────────────────────────────────────────────────────
  betaSeq.forEach((aa, i) => {{
    const x = aaX(betaX0, i);
    pepCtx.fillStyle = '#222';
    pepCtx.fillText(aa, x, betaY + 5);
    if (i < betaSeq.length - 1) {{
      const xMid = x + spacing / 2;
      if (mbbI.has(i+1) || mbyI.has(betaSeq.length-(i+1))) drawTick(xMid, betaY, yOff);
    }}
  }});

  // ── crosslinker vertical bar ──────────────────────────────────────────────
  const crossX = centreX;
  const midY   = (alphaY + betaY) / 2;
  pepCtx.strokeStyle = '#666'; pepCtx.lineWidth = 2;
  pepCtx.beginPath();
  pepCtx.moveTo(crossX, midY - yOff * 0.75);
  pepCtx.lineTo(crossX, midY + yOff * 0.75);
  pepCtx.stroke();

  // ── ion position labels (b / y) ───────────────────────────────────────────
  pepCtx.font = '11px Arial';
  function ionLabel(text, x, y, color) {{
    pepCtx.fillStyle = color;
    pepCtx.fillText(text, x, y);
  }}

  alphaSeq.forEach((_, i) => {{
    if (i >= alphaSeq.length - 1) return;
    const xMid = aaX(alphaX0, i) + spacing / 2;
    const bIdx = i + 1;
    const yIdx = alphaSeq.length - bIdx;
    if (mabI.has(bIdx)) {{
      ionLabel('b' + bIdx, xMid - spacing/8, alphaY - yOff - 3, '#37415D');
    }}
    if (mayI.has(yIdx)) {{
      ionLabel('y' + yIdx, xMid + spacing/8, alphaY + yOff + 12, '#375D53');
    }}
  }});

  betaSeq.forEach((_, i) => {{
    if (i >= betaSeq.length - 1) return;
    const xMid = aaX(betaX0, i) + spacing / 2;
    const bIdx = i + 1;
    const yIdx = betaSeq.length - bIdx;
    if (mbbI.has(bIdx)) {{
      ionLabel('b' + bIdx, xMid - spacing/8, betaY - yOff - 3, '#905110');
    }}
    if (mbyI.has(yIdx)) {{
      ionLabel('y' + yIdx, xMid + spacing/8, betaY + yOff + 12, '#5D4037');
    }}
  }});
}}

// ── Draw spectrum (bars + axes + connector lines) ─────────────────────────────
function drawSpectrum() {{
  specCtx.clearRect(0,0,sW,sH);
  specCtx.fillStyle='#fff'; specCtx.fillRect(0,0,sW,sH);

  // axes
  specCtx.strokeStyle='#333'; specCtx.lineWidth=1.5;
  specCtx.beginPath();
  specCtx.moveTo(SM.left, SM.top);
  specCtx.lineTo(SM.left, SM.top+sPlotH);
  specCtx.lineTo(SM.left+sPlotW, SM.top+sPlotH);
  specCtx.stroke();

  specCtx.fillStyle='#333'; specCtx.font='12px Arial'; specCtx.textAlign='center';
  for (let mz=200; mz<=1400; mz+=100) {{
    const x=mzToPx(mz);
    specCtx.beginPath(); specCtx.moveTo(x,SM.top+sPlotH); specCtx.lineTo(x,SM.top+sPlotH+5); specCtx.stroke();
    specCtx.fillText(mz, x, SM.top+sPlotH+18);
  }}
  specCtx.textAlign='right';
  for (let v=0; v<=100; v+=20) {{
    const y=intToPy(v);
    specCtx.beginPath(); specCtx.moveTo(SM.left-5,y); specCtx.lineTo(SM.left,y); specCtx.stroke();
    specCtx.fillText(v, SM.left-8, y+4);
  }}
  specCtx.fillStyle='#333'; specCtx.textAlign='center';
  specCtx.fillText('m/z', SM.left+sPlotW/2, sH-6);
  specCtx.save();
  specCtx.translate(13, SM.top+sPlotH/2);
  specCtx.rotate(-Math.PI/2);
  specCtx.fillText('Relative intensity (%)', 0, 0);
  specCtx.restore();

  // peaks
  specCtx.lineWidth=1;
  for (const p of SPECTRUM) {{
    if (p.mz<MZ_MIN || p.mz>MZ_MAX) continue;
    const x=mzToPx(p.mz), y0=intToPy(0), y1=intToPy(p.intensity);
    const matched=[...MATCHED_MZS].some(m=>Math.abs(m-p.mz)<0.01);
    specCtx.strokeStyle = matched ? '#222' : '#c0c0c0';
    specCtx.beginPath(); specCtx.moveTo(x,y0); specCtx.lineTo(x,y1); specCtx.stroke();
  }}

  // connectors
  if (showLines) {{
    specCtx.lineWidth=1; specCtx.setLineDash([3,3]);
    for (const ann of annotations) {{
      if (ann.pinned) continue;
      const px=mzToPx(ann.peak_mz), py=intToPy(ann.intensity);
      const lx=ann.cur_x+ann._lW/2,  ly=ann.cur_y+ann._lH/2;
      if (Math.hypot(lx-px, ly-py) > 12) {{
        specCtx.strokeStyle = ann.color==='#888888' ? '#aaa' : ann.color;
        specCtx.globalAlpha=0.65;
        specCtx.beginPath(); specCtx.moveTo(px,py); specCtx.lineTo(lx,ly); specCtx.stroke();
        specCtx.globalAlpha=1;
      }}
    }}
    specCtx.setLineDash([]);
  }}
}}

// ── TeX-like label → HTML ─────────────────────────────────────────────────────
function texToHtml(s) {{
  return s
    .replace(/_\\{{([^}}]+)\\}}/g,'<sub>$1</sub>')
    .replace(/\\^\\{{([^}}]+)\\}}/g,'<sup>$1</sup>')
    .replace(/_([^\\s{{<])/g,'<sub>$1</sub>')
    .replace(/\\^([^\\s{{<])/g,'<sup>$1</sup>');
}}

// ── Draggable label elements ──────────────────────────────────────────────────
let labelEls = [];

function buildLabels() {{
  labelEls.forEach(el=>el.remove()); labelEls=[];
  annotations.forEach(ann => {{
    const div=document.createElement('div');
    div.className='ann-label';
    div.style.color=ann.color;
    // ion label text (bold, matches matplotlib fontsize=16)
    const labelSpan=document.createElement('div');
    labelSpan.style.fontWeight='bold';
    labelSpan.style.textAlign='center';
    labelSpan.innerHTML=texToHtml(ann.label);
    div.appendChild(labelSpan);
    // m/z value below label, grey smaller font (matches matplotlib fontsize=10)
    const mzSpan=document.createElement('div');
    mzSpan.style.fontSize='9px';
    mzSpan.style.color='#888';
    mzSpan.style.textAlign='center';
    mzSpan.style.lineHeight='1.1';
    mzSpan.textContent=ann.mz.toFixed(2);
    div.appendChild(mzSpan);
    const btn=document.createElement('button');
    btn.className='pin-btn'; btn.textContent='Pin to peak';
    btn.addEventListener('click', e=>{{
      e.stopPropagation(); ann.pinned=true;
      ann.cur_x=mzToPx(ann.peak_mz)-(ann._lW||50)/2;
      ann.cur_y=intToPy(ann.intensity)-(ann._lH||16)-4;
      posEl(div,ann); drawSpectrum();
    }});
    div.appendChild(btn);
    specWrap.appendChild(div);
    labelEls.push(div);
    ann._lW=div.offsetWidth||50; ann._lH=div.offsetHeight||16;
    posEl(div,ann);
    let sx,sy,smx,smy;
    div.addEventListener('mousedown',e=>{{
      if(e.target===btn) return;
      e.preventDefault(); div.classList.add('dragging');
      smx=e.clientX; smy=e.clientY; sx=ann.cur_x; sy=ann.cur_y;
      ann.pinned=false;
      const mv=e2=>{{ ann.cur_x=sx+e2.clientX-smx; ann.cur_y=sy+e2.clientY-smy; posEl(div,ann); drawSpectrum(); }};
      const up=()=>{{ div.classList.remove('dragging'); document.removeEventListener('mousemove',mv); document.removeEventListener('mouseup',up); }};
      document.addEventListener('mousemove',mv); document.addEventListener('mouseup',up);
    }});
  }});
  // re-measure after paint
  annotations.forEach((ann,i)=>{{
    ann._lW=labelEls[i].offsetWidth||50; ann._lH=labelEls[i].offsetHeight||16;
    posEl(labelEls[i],ann);
  }});
}}

function posEl(el,ann) {{ el.style.left=ann.cur_x+'px'; el.style.top=ann.cur_y+'px'; }}

// ── Resize / layout ───────────────────────────────────────────────────────────
function resize() {{
  const panelW = document.getElementById('panels').clientWidth;
  // Spectrum panel: 15:7 ratio
  sW=panelW; sH=Math.round(panelW*7/15);
  sPlotW=sW-SM.left-SM.right; sPlotH=sH-SM.top-SM.bottom;
  specCanvas.width=sW; specCanvas.height=sH;
  specWrap.style.height=sH+'px';
  // Peptide panel: 15:2 ratio
  pW=panelW; pH=Math.round(panelW*2/15);
  pPlotW=pW-PM.left-PM.right; pPlotH=pH-PM.top-PM.bottom;
  pepCanvas.width=pW; pepCanvas.height=pH;
  pepWrap.style.height=pH+'px';

  annotations.forEach(ann=>{{
    ann.cur_x=mzToPx(ann.default_x)-(ann._lW||50)/2;
    ann.cur_y=intToPy(ann.default_y)-(ann._lH||16);
    ann.pinned=false;
  }});
  buildLabels(); drawPeptide(); drawSpectrum();
}}

function resetPositions() {{
  annotations.forEach(ann=>{{
    ann.cur_x=mzToPx(ann.default_x)-(ann._lW||50)/2;
    ann.cur_y=intToPy(ann.default_y)-(ann._lH||16);
    ann.pinned=false;
  }});
  labelEls.forEach((el,i)=>posEl(el,annotations[i]));
  drawSpectrum();
}}

// ── Export PNG ────────────────────────────────────────────────────────────────
// Render both panels and all labels onto a single offscreen canvas at 2× DPR.
function exportPng() {{
  const DPR    = 2;
  const totalH = (pH + sH) * DPR;
  const totalW = pW * DPR;

  const off = document.createElement('canvas');
  off.width  = totalW;
  off.height = totalH;
  const oc = off.getContext('2d');
  oc.fillStyle = '#fff'; oc.fillRect(0,0,totalW,totalH);

  // 1. Peptide panel
  oc.drawImage(pepCanvas, 0, 0, totalW, pH*DPR);

  // 2. Spectrum panel (bars + axes)
  oc.drawImage(specCanvas, 0, pH*DPR, totalW, sH*DPR);

  // 3. Bake labels onto spectrum section
  oc.scale(DPR, DPR);
  oc.translate(0, pH);  // shift to spectrum coordinate origin
  annotations.forEach(ann => {{
    // connector line
    if (!ann.pinned && showLines) {{
      const px=mzToPx(ann.peak_mz), py=intToPy(ann.intensity);
      const lx=ann.cur_x+ann._lW/2, ly=ann.cur_y+ann._lH;
      if (Math.hypot(lx-px,ly-py)>12) {{
        oc.save();
        oc.strokeStyle=ann.color==='#888888'?'#aaa':ann.color;
        oc.lineWidth=1; oc.setLineDash([3,3]); oc.globalAlpha=0.65;
        oc.beginPath(); oc.moveTo(px,py); oc.lineTo(lx,ly); oc.stroke();
        oc.restore();
      }}
    }}
    // label text (plain, no HTML – strip sub/sup markers for canvas)
    const plain = ann.label
      .replace(/_\\{{([^}}]+)\\}}/g,'$1')
      .replace(/\\^\\{{([^}}]+)\\}}/g,'$1');
    oc.font='bold 12px Arial';
    oc.fillStyle=ann.color;
    oc.textAlign='left';
    oc.fillText(plain, ann.cur_x, ann.cur_y+ann._lH-2);
    // m/z value below (grey, small)
    oc.font='9px Arial'; oc.fillStyle='#888';
    oc.fillText(ann.mz.toFixed(2), ann.cur_x, ann.cur_y+ann._lH+8);
  }});

  const link=document.createElement('a');
  link.href=off.toDataURL('image/png');
  link.download='scan_{scan_number}_annotated.png';
  link.click();
}}

// ── Export CSV ────────────────────────────────────────────────────────────────
function exportCsv() {{
  const rows=[['Ion Type','Signature','Position','Charge',
               'Theoretical m/z','Observed m/z','Intensity (%)','Mass Error (Da)',
               'x','y','has_line','line_end_x','line_end_y']];
  annotations.forEach(ann=>{{
    // connector line: from peak tip to label bottom (mirrors what drawSpectrum draws)
    const px=mzToPx(ann.peak_mz), py=intToPy(ann.intensity);
    const lx=ann.cur_x+ann._lW/2,  ly=ann.cur_y+ann._lH/2;
    const dist=Math.hypot(lx-px, ly-py);
    const hasLine = showLines && !ann.pinned && dist>12 ? 1 : 0;
    rows.push([
      ann.ion_type, ann.sig_type, ann.position, ann.charge,
      ann.theoretical_mz.toFixed(4), ann.mz.toFixed(4),
      ann.intensity.toFixed(2), ann.mass_error.toFixed(4),
      pxToMz(lx).toFixed(4),
      pyToInt(ann.cur_y+ann._lH/2).toFixed(4),
      hasLine,
      hasLine ? pxToMz(lx).toFixed(4) : '',
      hasLine ? pyToInt(ly).toFixed(4)  : '',
    ]);
  }});
  const csv=rows.map(r=>r.map(v=>JSON.stringify(v??'')).join(',')).join('\\n');
  const a=document.createElement('a');
  a.href='data:text/csv;charset=utf-8,'+encodeURIComponent(csv);
  a.download='scan_{scan_number}_annotations.csv';
  a.click();
}}

// ── Init ──────────────────────────────────────────────────────────────────────
annotations = ANNOTATIONS_INIT.map(a=>({{...a, cur_x:0, cur_y:0, _lW:50, _lH:16, pinned:false}}));

window.addEventListener('resize', resize);
document.getElementById('chk-lines').addEventListener('change', e=>{{ showLines=e.target.checked; drawSpectrum(); }});
document.getElementById('btn-reset').addEventListener('click', resetPositions);
document.getElementById('btn-export-csv').addEventListener('click', exportCsv);

document.fonts.ready.then(resize);
</script>
</body>
</html>"""

        with open(html_output, 'w', encoding='utf-8') as f:
            f.write(html_content)
        print(f"Interactive HTML editor saved to: {html_output}")


if __name__ == "__main__":
    annotator = CrosslinkedMS2Annotator(
        ms2_file=r"C:\Crux\data\20260511\20260511-TYG_2-HCC1-150.ms2",
        alpha_sequence='RQAKEPCVESLVSQYFQTVTDYGK',
        beta_sequence='SCDTPPPCPRCPAPELLGGPSVFLFPPKPK',
        scan_number=24490,
        alpha_modifications={},
        beta_modifications={},
        alpha_crosslink_site=7,
        beta_crosslink_site=2,
        crosslinker_mass=-2.01565,
        tolerance='20ppm'
    )
    output_dir = r'C:\Crux\Output\20260531\HCC1-150\sel'
    csv_path   = r'{}\{}_input_ions.csv'.format(output_dir, annotator.scan_number)
    # csv_path = None
    # Uncomment the next line to export all theoretical fragment ions before matching.
    # theoretical_csv_path = r'{}\{}_theoretical_ions.csv'.format(output_dir, annotator.scan_number)
    theoretical_csv_path = None

    # annotate_crosslinked_spectrum automatically opens the HTML editor when
    # show_ann_text=True and csv_input is provided.  Open the HTML in a browser,
    # adjust labels, then use "Export PNG" for the final figure or "Export CSV"
    # to save positions for the matplotlib renderer.
    annotator.annotate_crosslinked_spectrum(
        output_file=r'{}\{}.tiff'.format(output_dir, annotator.scan_number),
        csv_input=csv_path,
        theoretical_csv=theoretical_csv_path,
    )