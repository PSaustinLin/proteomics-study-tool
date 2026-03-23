from time_utils import format_runtime, current_time
start_time = current_time()
import torch
import torch.nn.functional as F
from typing import Dict, List, Tuple, Union, Optional
import os
import csv
from collections import defaultdict
import concurrent.futures
from queue import Queue
from threading import Lock
import traceback
#nvidia-smi
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device} - {torch.cuda.get_device_name(0)}")

# GLOBAL Constants
BIN_WIDTH = 0.02
BIN_OFFSET = 0.4
cid = True
crosslinker = 'disulfide'

# Constant masses
proton_mass = 1.00728
water_mass = 18.01056

if crosslinker == 'loss_NH3':
    crosslinker_mass=-17.02655
    # Isospeptide bonds that lose NH3 (K-N / K-Q)
    signature_types = {
        '': 0.0,
        '-1': -1.00783, #-H
        '-15': -15.01090, #-NH
        '-16': -16.01872 #-NH2
    }
elif crosslinker == 'loss_H2O':
    crosslinker_mass=-18.01056
    # Isospeptide bonds that lose H2O (K-D / K-E)
    signature_types = {
        '-1': -1.00783, #-H
        '-17': -17.00274 #-OH
    }
elif crosslinker == 'disulfide':
    crosslinker_mass=-2.01565
    # Disulfides
    signature_types = {
        '': 0.0,
        '-2': -2.01565,
        '+32': 31.97207,
        '-34': -33.98772
    }


aa_dict = {
    'A': 71.03711, 'C': 103.00918, 'D': 115.02694, 'E': 129.04259,
    'F': 147.06841, 'G': 57.02146, 'H': 137.05891, 'I': 113.08406,
    'K': 128.09496, 'L': 113.08406, 'M': 131.04048, 'N': 114.04293,
    'P': 97.05276, 'Q': 128.05858, 'R': 156.10111, 'S': 87.03203,
    'T': 101.04768, 'V': 99.06841, 'W': 186.07931, 'Y': 163.06333, 'X': 0.0
}

def sort_by_intensity(peak: Tuple[float, float, float]) -> float:
    return peak[1]

def read_peptides(file_path: str) -> Tuple[List[str], List[Dict[int, float]], List[int]]:
    peptides = []
    modifications = []
    crosslink_sites = []
    with open(file_path, 'r') as f:
        reader = csv.reader(f)
        next(reader)  # Skip header
        for row in reader:
            peptide = row[0]
            mods = row[1]
            xl_site = int(row[2])
            
            # Create a dictionary for modifications (3_15.9949,5_86.0368)
            mods_dict = {}
            if mods:
                mods = mods.split(',')
                for mod in mods:
                    site, mass = mod.split('_')
                    mods_dict[int(site)] = float(mass)
            
            peptides.append(peptide)
            modifications.append(mods_dict)  # Append dictionary instead of list
            crosslink_sites.append(xl_site)
    
    print(f"Peptides read: {peptides[:5]}... ({len(peptides)} total)")
    return peptides, modifications, crosslink_sites

def generate_all_combinations(peptides: List[str], modifications: List[Dict[int, float]], 
                             crosslink_sites: List[int]) -> Tuple[List[str], List[Dict[int, float]], 
                                                                List[int], List[str], List[Dict[int, float]], 
                                                                List[int]]:
    """
    Generate all possible combinations of peptide pairs including self-pairs.
    Returns expanded lists of alpha and beta peptides with their modifications and crosslink sites.
    """
    alpha_peptides = []
    alpha_modifications = []
    alpha_crosslink_sites = []
    beta_peptides = []
    beta_modifications = []
    beta_crosslink_sites = []
    
    n = len(peptides)
    # Generate all possible pairs (including self-pairs)
    for i in range(n):
        for j in range(i, n):  # Start from i to include self-pairs and avoid duplicates
            alpha_peptides.append(peptides[i])
            alpha_modifications.append(modifications[i])
            alpha_crosslink_sites.append(crosslink_sites[i])
            
            beta_peptides.append(peptides[j])
            beta_modifications.append(modifications[j])
            beta_crosslink_sites.append(crosslink_sites[j])

    #total_combinations = len(alpha_peptides)
    #print(f"Generated {total_combinations} peptide combinations")
    return (alpha_peptides, alpha_modifications, alpha_crosslink_sites,
            beta_peptides, beta_modifications, beta_crosslink_sites)

def _calculate_peptide_mass(aa_dict: Dict[str, float], sequence: str, modifications: Dict[int, float]) -> float:
    peptide_mass = 18.01056  # water mass
    for i, aa in enumerate(sequence):
        # Base amino acid mass
        if aa in aa_dict:
            peptide_mass += aa_dict[aa]
        # Add modification if exists
        if i+1 in modifications:
            peptide_mass += modifications[i+1]
    return peptide_mass

@torch.jit.script
def calculate_alpha_ions(aa_dict: Dict[str, float], signature_types: Dict[str, float], alpha_peptide: str, alpha_modification: Dict[int, float], alpha_crosslink_site: int, beta_peptide: str, beta_modification: Dict[int, float], beta_crosslink_site: int, precursor_mass: float, precursor_charge: int, crosslinker_mass: float) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    proton_mass = 1.00728
    cid = True
    # Calculate whole peptide masses for use in crosslink mass calculation
    alpha_whole_mass = _calculate_peptide_mass(aa_dict, alpha_peptide, alpha_modification)
    
    if beta_peptide is not None and beta_modification is not None:
        beta_whole_mass = _calculate_peptide_mass(aa_dict, beta_peptide, beta_modification)
    else:
        beta_whole_mass = precursor_mass - alpha_whole_mass

    max_charge = min(precursor_charge - 1, 3)
    alpha_length = len(alpha_peptide)
    beta_length = len(beta_peptide)
    # Regular ions
    alpha_b_ions = torch.zeros(alpha_length - 1, dtype=torch.float32)
    alpha_y_ions = torch.zeros(alpha_length - 1, dtype=torch.float32)
    beta_b_ions = torch.zeros(beta_length - 1, dtype=torch.float32)
    beta_y_ions = torch.zeros(beta_length - 1, dtype=torch.float32)

    # signature ions
    alpha_sig_ions = torch.zeros(len(signature_types), dtype=torch.float32)
    beta_sig_ions = torch.zeros(len(signature_types), dtype=torch.float32)
    #alpha_sig_types = [''] * len(signature_types)
    #beta_sig_types = [''] * len(signature_types)
    
    """
    # both amide and crosslinker cleavage ions (define outside the if statement when used)
    if not cid:
        alpha_b_sig_ions = torch.zeros(alpha_length - 1, dtype=torch.float32)
        beta_b_sig_ions = torch.zeros(beta_length - 1, dtype=torch.float32)
        alpha_y_sig_ions = torch.zeros(alpha_length - 1, dtype=torch.float32)
        beta_y_sig_ions = torch.zeros(beta_length - 1, dtype=torch.float32)
    """
    # Calculate alpha b-ions
    alpha_b_mass = 0.0
    for i, aa in enumerate(alpha_peptide[:-1]):
        alpha_b_mass += aa_dict[aa]
        # Add modification if exists
        if i+1 in alpha_modification:
            alpha_b_mass += alpha_modification[i+1]

        crosslink_b_mass = alpha_b_mass
        # Crosslinked fragment condition
        if i+1 >= alpha_crosslink_site:
            # Add crosslinker and beta peptide mass
            crosslink_b_mass = (
                alpha_b_mass + 
                crosslinker_mass + 
                beta_whole_mass
            )
        alpha_b_ions[i - 1] = crosslink_b_mass

        """
        # Non-crosslinked fragment alpha_b_ions.append((b_ion_mz, i+1, charge, f'αb', f'{sig_type}'))
        if not cid:
            for sig_type, sig_mass in signature_types.items():
                non_crosslink_b_mass = alpha_b_mass + sig_mass
                alpha_b_sig_ions[i - 1] = non_crosslink_b_mass
        """

    # Calculate alpha y-ions
    alpha_y_mass = 18.01056
    for i in range(len(alpha_peptide)-1, 0, -1):
        alpha_y_mass += aa_dict[alpha_peptide[i]]

        # Add modification if exists
        if i+1 in alpha_modification:
            alpha_y_mass += alpha_modification[i+1]

        crosslink_y_mass = alpha_y_mass
        # Crosslinked fragment condition
        if i < alpha_crosslink_site:
            # Add crosslinker and beta peptide mass
            crosslink_y_mass = (alpha_y_mass + crosslinker_mass + beta_whole_mass)
        alpha_y_ions[i - 1] = crosslink_y_mass

        """
        # Non-crosslinked fragment alpha_y_ions.append((y_ion_mz, len(alpha_peptide) - i, charge, f'αy', f'{sig_type}'))
        if not cid:
            for sig_type, sig_mass in signature_types.items():
                non_crosslink_y_mass = alpha_y_mass + sig_mass
                alpha_y_sig_ions[i - 1] = non_crosslink_y_mass
        """

    if beta_peptide:
        # Calculate beta b-ions
        beta_b_mass = 0.0
        for i, aa in enumerate(beta_peptide[:-1]):
            beta_b_mass += aa_dict[aa]
            # Add modification if exists
            if i+1 in beta_modification:
                beta_b_mass += beta_modification[i+1]

            crosslink_b_mass = beta_b_mass
            # Crosslinked fragment condition
            if (beta_crosslink_site is not None and 
                    i+1 >= beta_crosslink_site):
                # Add crosslinker and alpha peptide mass
                crosslink_b_mass = (beta_b_mass + crosslinker_mass + alpha_whole_mass)
            beta_b_ions[i - 1] = crosslink_b_mass
            
            """
            # Non-crosslinked fragment
            if not cid:
                for sig_type, sig_mass in signature_types.items():
                    non_crosslink_b_mass = beta_b_mass + sig_mass
                    beta_b_sig_ions[i - 1] = non_crosslink_b_mass
            """
        
        # Calculate beta y-ions
        beta_y_mass = 18.01056
        for i in range(len(beta_peptide)-1, 0, -1):
            beta_y_mass += aa_dict[beta_peptide[i]]

            # Add modification if exists
            if i+1 in beta_modification:
                beta_y_mass += beta_modification[i+1]
                
            crosslink_y_mass = beta_y_mass
            # Crosslinked fragment condition
            if (beta_crosslink_site is not None and i < beta_crosslink_site):
                # Add crosslinker and alpha peptide mass
                crosslink_y_mass = (beta_y_mass + crosslinker_mass + alpha_whole_mass)
            beta_y_ions[i - 1] = crosslink_y_mass
            
            """
            # Non-crosslinked fragment beta_y_ions.append((y_ion_mz, len(self.beta_sequence) - i, charge, f'βy', f'{sig_type}'))
            if not cid:
                for sig_type, sig_mass in signature_types.items():
                    non_crosslink_y_mass = beta_y_mass + sig_mass
                    beta_y_sig_ions[i - 1] = non_crosslink_y_mass
            """
    
    # Don't use format string here, pass the actual sig_type
    for i, (sig_type, sig_mass) in enumerate(signature_types.items()):
        # whole alpha signature ions alpha_sig_ions.append((ion_mz, '', charge, f'α', f'{sig_type}'))
        non_crosslink_mass = alpha_whole_mass + sig_mass
        alpha_sig_ions[i] = non_crosslink_mass

        # whole beta signature ions beta_sig_ions.append((ion_mz, '', charge, f'β', f'{sig_type}'))
        non_crosslink_mass = beta_whole_mass + sig_mass
        beta_sig_ions[i] = non_crosslink_mass

    return alpha_b_ions, alpha_y_ions, beta_b_ions, beta_y_ions, alpha_sig_ions, beta_sig_ions

@torch.jit.script
def add_ion_with_charges(mass: float, ion_type: str, ion_number: int, sig_type: str, input_charge: int, proton_mass: float) -> Tuple[torch.Tensor, List[str]]:
    # Pre-allocate tensors instead of using lists for numerical values
    charge_range = torch.arange(1, input_charge, dtype=torch.int32)
    # Calculate m/z values
    mz_values = (mass + charge_range * proton_mass) / charge_range
    
    # Build ion labels
    matched_ions: List[str] = []
    for charge in range(1, input_charge):
        if ion_number == 0:     # sig ions
            label = f'{ion_type}{sig_type}^{{+{charge}}}'
        else:                   # b/y ions
            label = f'{ion_type}_{{{ion_number}}}{sig_type}^{{+{charge}}}'
        formatted_label = f'$\\mathrm{{{label}}}$'
        matched_ions.append(formatted_label)
    return mz_values, matched_ions

@torch.jit.script
def vectorize_spectrum(mz_values: torch.Tensor, intensities: torch.Tensor, bin_width: float, offset: float, observed: bool) -> torch.Tensor:
    # Ensure both tensors are float32 and on the same device
    device = mz_values.device
    mz_values = mz_values.to(torch.float32)
    intensities = intensities.to(torch.float32)
    
    max_mz = 2000
    bin_count = int(torch.ceil(torch.tensor(max_mz / bin_width)))
    binned_spectrum = torch.zeros(bin_count, dtype=torch.float32, device=device)
    
    bin_indices = ((mz_values + offset) / bin_width).long()
    valid_indices = (bin_indices >= 0) & (bin_indices < bin_count)
    
    if observed:
        binned_spectrum.scatter_add_(0, bin_indices[valid_indices], intensities[valid_indices])
    else:
        binned_spectrum.scatter_(0, bin_indices[valid_indices], torch.ones_like(intensities[valid_indices]))
    
    return binned_spectrum

@torch.jit.script
def create_theoretical_spectrum(alpha_b: torch.Tensor, alpha_y: torch.Tensor, beta_b: torch.Tensor, beta_y: torch.Tensor, alpha_sig_ions: torch.Tensor, beta_sig_ions: torch.Tensor, alpha_crosslink_site: int, beta_crosslink_site: int, input_charge: int, signature_types: Dict[str, float]) -> Tuple[torch.Tensor, List[str], torch.Tensor, torch.Tensor]:
    
    PROTON = 1.00728
    BIN_WIDTH = 0.02
    BIN_OFFSET = 0.4
    
    # Initialize lists for collecting results
    all_mzs: List[torch.Tensor] = []
    all_labels: List[str] = []
    
    alpha_len = len(alpha_b) + 1
    beta_len = len(beta_b) + 1
    
    for ion_type, ions in [('αb', alpha_b), ('αy', alpha_y), ('βb', beta_b), ('βy', beta_y)]:
        for i, mass in enumerate(ions):
            if (ion_type == 'αb' and i + 1 >= alpha_crosslink_site) or (ion_type == 'βb' and i + 1 >= beta_crosslink_site) or (ion_type == 'αy' and i + 1 >= alpha_len - alpha_crosslink_site + 1) or (ion_type == 'βb' and i + 1 >= beta_len - beta_crosslink_site + 1):
                star_ion_type = ion_type + '*'
            else:
                star_ion_type = ion_type
            mzs, ion = add_ion_with_charges(float(mass), star_ion_type, i+1, '', input_charge, PROTON)
            all_mzs.append(mzs)
            all_labels.extend(ion)

    for ion_type, ions in [('α', alpha_sig_ions), ('β', beta_sig_ions)]:
        for i, mass in enumerate(ions):
            for j, (sig_type, sig_mass) in enumerate(signature_types.items()):
                mzs, ion = add_ion_with_charges(float(mass), ion_type, 0, sig_type, input_charge, PROTON)
                all_mzs.append(mzs)
                all_labels.extend(ion)
    
    # Concatenate all mass tensors
    if len(all_mzs) > 0:
        mass_tensor = torch.cat(all_mzs)
    else:
        mass_tensor = torch.tensor([], dtype=torch.float32)
    
    # Convert masses to m/z values
    #charges = torch.tensor([float(ion.split('+')[1]) for ion in all_matched_ions])
    #mz_values = mass_tensor / charges + PROTON
    
    # Sort by m/z values
    sorted_indices = torch.argsort(mass_tensor)
    sorted_mz = mass_tensor[sorted_indices]
    sorted_intensities = torch.ones_like(sorted_mz)
    
    # Convert matched ions list to sorted version using a type-safe approach
    indices_list: List[int] = sorted_indices.tolist()  # Explicit type annotation
    sorted_matched_ions: List[str] = []
    for idx in indices_list:
        sorted_matched_ions.append(all_labels[idx])
    
    # Create theoretical spectrum
    theoretical_spectrum = vectorize_spectrum(sorted_mz, sorted_intensities, BIN_WIDTH, BIN_OFFSET, False)
    
    return theoretical_spectrum, sorted_matched_ions, sorted_mz, sorted_intensities

@torch.jit.script
def preprocess_spectrum(spectrum: torch.Tensor, region_size: int = 10, max_intensity: float = 50.0) -> torch.Tensor:
    """
    Preprocess an observed spectrum following the enhanced approach:
    1. Square root the intensities
    2. Region-based normalization
    3. Remove background by subtracting mean in windows
    
    Args:
        spectrum: Binned spectrum tensor
        region_size: Size of regions for normalization
        max_intensity: Maximum intensity for region normalization
        
    Returns:
        Preprocessed spectrum tensor
    """
    device = spectrum.device
    
    # 1. Square root the intensities to reduce intensity spread
    processed = torch.sqrt(spectrum)
    
    # 2. Region-based normalization
    normalized = torch.zeros_like(processed, device=device)
    num_regions = (processed.size(0) + region_size - 1) // region_size  # Ceiling division
    
    for i in range(num_regions):
        start = i * region_size
        end = min(start + region_size, processed.size(0))
        
        if start < end:
            region = processed[start:end]
            max_val = torch.max(region)
            
            if max_val > 0:
                scale_factor = max_intensity / max_val
                normalized[start:end] = region * scale_factor
    
    # 3. Remove background by subtracting local mean
    max_xcorr_offset = 75
    result = normalized.clone()
    
    # Use convolution for efficient mean calculation in sliding windows
    padded = F.pad(normalized.unsqueeze(0).unsqueeze(0), 
                   (max_xcorr_offset, max_xcorr_offset), 
                   mode='constant')
    
    # Create boxcar filter kernel for mean calculation
    kernel = torch.ones(1, 1, max_xcorr_offset*2+1, device=device) / (max_xcorr_offset*2+1)
    
    # Apply convolution to get mean values for each position
    means = F.conv1d(padded, kernel, padding=0).squeeze()
    
    # Subtract mean from normalized spectrum
    result = normalized - means
    
    return result

@torch.jit.script
def vectorize_spectrum_enhanced(mz_values: torch.Tensor, intensities: torch.Tensor, 
                             bin_width: float, bin_offset: float, 
                             max_mz: float = 2000.0) -> torch.Tensor:
    """
    Create binned spectrum
    
    Args:
        mz_values: m/z values tensor
        intensities: Intensity values tensor
        bin_width: Width of each bin
        bin_offset: Bin offset value
        max_mz: Maximum m/z value to consider
        
    Returns:
        Binned spectrum tensor
    """
    device = mz_values.device
    
    # Calculate bin count
    bin_count = int(max_mz / bin_width) + 1
    binned_spectrum = torch.zeros(bin_count, dtype=torch.float32, device=device)
    
    # Calculate bin indices with offset
    bin_indices = ((mz_values / bin_width + 1.0) - bin_offset).long()
    
    # Filter valid indices
    valid_indices = (bin_indices >= 0) & (bin_indices < bin_count)
    
    # Add intensities to bins
    binned_spectrum.scatter_add_(0, bin_indices[valid_indices], intensities[valid_indices])
    
    return binned_spectrum

@torch.jit.script
def create_theoretical_spectrum_enhanced(mz_values: torch.Tensor, bin_width: float, bin_offset: float, 
                                     intensity_value: float = 50.0, max_mz: float = 2000.0) -> torch.Tensor:
    """
    Create theoretical spectrum for XCorr with standard intensity values
    
    Args:
        mz_values: m/z values tensor
        bin_width: Width of each bin
        bin_offset: Bin offset value
        intensity_value: Standard intensity for theoretical peaks
        max_mz: Maximum m/z value to consider
        
    Returns:
        Theoretical spectrum tensor
    """
    device = mz_values.device
    
    # Calculate bin count
    bin_count = int(max_mz / bin_width) + 1
    theoretical_spectrum = torch.zeros(bin_count, dtype=torch.float32, device=device)
    
    # Calculate bin indices with offset
    bin_indices = ((mz_values / bin_width + 1.0) - bin_offset).long()
    
    # Filter valid indices
    valid_indices = (bin_indices >= 0) & (bin_indices < bin_count)
    filtered_indices = bin_indices[valid_indices]
    
    # Set standard intensity for each peak
    theoretical_spectrum.scatter_(0, filtered_indices, 
                                torch.full_like(filtered_indices, intensity_value, dtype=torch.float32))
    
    return theoretical_spectrum

@torch.jit.script
def compute_xcorr_enhanced(theoretical_spectrum: torch.Tensor, observed_spectrum: torch.Tensor, 
                         normalization_factor: float = 10000.0) -> float:
    """
    Compute enhanced XCorr score between theoretical and preprocessed observed spectra
    
    Args:
        theoretical_spectrum: Theoretical spectrum tensor
        observed_spectrum: Preprocessed observed spectrum tensor
        normalization_factor: Factor to normalize the XCorr score
        
    Returns:
        XCorr score
    """
    device = theoretical_spectrum.device
    observed_spectrum = observed_spectrum.to(device)
    
    # Ensure both spectra are on the same device
    theoretical_spectrum = theoretical_spectrum.to(device)
    
    # Calculate simple dot product as in xcorr_example.py
    xcorr_score = torch.dot(theoretical_spectrum, observed_spectrum) / normalization_factor
    
    return xcorr_score.item()

def process_spectrum_enhanced(aa_dict: Dict[str, float], signature_types: Dict[str, float], 
                           spectrum: torch.Tensor, scan_num: str, 
                           alpha_peptides: List[str], alpha_modifications: List[Dict[int, float]], 
                           alpha_crosslink_sites: List[int], 
                           beta_peptides: List[str], beta_modifications: List[Dict[int, float]], 
                           beta_crosslink_sites: List[int], 
                           crosslinker_mass: float, precursor_mass: float, charge: int, 
                           device: torch.device) -> List[Tuple[str, str, str, float, float, float]]:
    
    results = []
    
    # Extract the mz and intensity values from spectrum tensor
    observed_mz = spectrum[:, 0].to(device)
    observed_intensity = spectrum[:, 1].to(device)
    
    # Create binned observed spectrum with enhanced approach
    vectorized_observed = vectorize_spectrum_enhanced(
        observed_mz, observed_intensity, BIN_WIDTH, BIN_OFFSET
    )
    
    # Preprocess the observed spectrum
    processed_observed = preprocess_spectrum(vectorized_observed)
    
    for pep_idx, (alpha_peptide, alpha_modification, alpha_crosslink_site, 
                 beta_peptide, beta_modification, beta_crosslink_site) in enumerate(
        zip(alpha_peptides, alpha_modifications, alpha_crosslink_sites, 
            beta_peptides, beta_modifications, beta_crosslink_sites)):
            
        theoretical_mass = _calculate_peptide_mass(aa_dict, alpha_peptide, alpha_modification) + _calculate_peptide_mass(aa_dict, beta_peptide, beta_modification) + crosslinker_mass

        mass_diff = precursor_mass - theoretical_mass
        ppm_diff = abs(1E+6 * mass_diff / theoretical_mass)
        if ppm_diff <= 10.0:
            
            alpha_b, alpha_y, beta_b, beta_y, alpha_sig_ions, beta_sig_ions = calculate_alpha_ions(
                aa_dict, signature_types, alpha_peptide, alpha_modification, 
                alpha_crosslink_site, beta_peptide, beta_modification, 
                beta_crosslink_site, precursor_mass, charge, crosslinker_mass)

            # Get theoretical ion m/z values
            _, _, theoretical_mz, _ = create_theoretical_spectrum(
                alpha_b, alpha_y, beta_b, beta_y, alpha_sig_ions, beta_sig_ions,
                alpha_crosslink_site, beta_crosslink_site, charge, signature_types)
            
            # Create enhanced theoretical spectrum
            theoretical_spectrum = create_theoretical_spectrum_enhanced(
                theoretical_mz, BIN_WIDTH, BIN_OFFSET
            )
            
            # Calculate enhanced XCorr
            xcorr = compute_xcorr_enhanced(theoretical_spectrum, processed_observed)
            
            # Output criteria for resolving the IO bottleneck
            if xcorr >= 2.0:
                results.append((scan_num, alpha_peptide, beta_peptide, alpha_crosslink_site, beta_crosslink_site, _calculate_peptide_mass(aa_dict, alpha_peptide, alpha_modification), _calculate_peptide_mass(aa_dict, beta_peptide, beta_modification), theoretical_mass, precursor_mass, mass_diff, ppm_diff, xcorr))
    
    return results

@torch.jit.script
def create_xcorr_observed_spectrum(mz_values: torch.Tensor, intensities: torch.Tensor, 
                                 bin_width: float) -> torch.Tensor:
    """
    Create processed observed spectrum for XCorr calculation:
    1. Bin the spectrum
    2. Preprocess intensities
    3. Subtract mean in sliding window
    """
    # Calculate number of bins
    max_mz = 2000
    num_bins = int(max_mz / bin_width) + 1
    binned = torch.zeros(num_bins, dtype=torch.float32, device=mz_values.device)
    
    # Bin the spectrum
    bin_indices = (mz_values / bin_width).long()
    valid_mask = (bin_indices >= 0) & (bin_indices < num_bins)
    bin_indices = bin_indices[valid_mask]
    intensities = intensities[valid_mask]
    
    # Sum intensities in each bin
    binned.scatter_add_(0, bin_indices, intensities)
    
    # Preprocess spectrum
    processed = preprocess_spectrum(binned)
    
    # Calculate and subtract mean in sliding window (75 peaks on each side)
    window_size = 150
    padding = window_size // 2
    padded = torch.nn.functional.pad(processed, (padding, padding), mode='constant')
    
    # Calculate moving average
    window_mean = torch.nn.functional.avg_pool1d(
        padded.unsqueeze(0).unsqueeze(0),
        kernel_size=window_size,
        stride=1,
        padding=0
    ).squeeze()
    
    # Subtract mean from original spectrum
    processed = processed - window_mean
    
    return processed

@torch.jit.script
def create_xcorr_theoretical_spectrum(mz_values: torch.Tensor, bin_width: float) -> torch.Tensor:
    """
    Create theoretical spectrum for XCorr:
    1. Bin the spectrum with unit heights
    2. No intensity preprocessing needed for theoretical spectrum
    """
    max_mz = 2000
    num_bins = int(max_mz / bin_width) + 1
    binned = torch.zeros(num_bins, dtype=torch.float32, device=mz_values.device)
    
    # Bin the spectrum with unit intensities
    bin_indices = (mz_values / bin_width).long()
    valid_mask = (bin_indices >= 0) & (bin_indices < num_bins)
    bin_indices = bin_indices[valid_mask]
    
    # Set unit intensity for each peak
    binned.scatter_add_(0, bin_indices, torch.ones_like(bin_indices, dtype=torch.float32))
    
    return binned

@torch.jit.script
def compute_xcorr(theoretical_spectrum: torch.Tensor, observed_spectrum: torch.Tensor, window_size: int = 75) -> float:
    """
    Compute XCorr score between theoretical and observed spectra using Crux-like approach
    """
    device = theoretical_spectrum.device
    observed_spectrum = observed_spectrum.to(device)
    
    # 1. Normalize observed spectrum
    # Take square root of intensities
    observed_processed = torch.sqrt(observed_spectrum)
    
    # Calculate and subtract local mean using convolution
    # Add padding to handle boundaries
    padded = F.pad(observed_processed.unsqueeze(0).unsqueeze(0), 
                   (window_size, window_size), mode='constant')
    
    # Calculate moving average with a boxcar filter
    kernel = torch.ones(1, 1, window_size*2+1, device=device) / (window_size*2+1)
    local_mean = F.conv1d(padded, kernel, padding=0).squeeze()
    
    # Subtract mean from observed spectrum
    observed_processed = observed_processed - local_mean
    
    # 2. Calculate correlation at zero offset
    xcorr = torch.dot(theoretical_spectrum, observed_processed)
    
    return xcorr.item()

def process_spectrum(aa_dict: Dict[str, float], signature_types: Dict[str, float], 
                    spectrum: torch.Tensor, scan_num: str, 
                    alpha_peptides: List[str], alpha_modifications: List[Dict[int, float]], 
                    alpha_crosslink_sites: List[int], 
                    beta_peptides: List[str], beta_modifications: List[Dict[int, float]], 
                    beta_crosslink_sites: List[int], 
                    crosslinker_mass: float, precursor_mass: float, charge: int, 
                    device: torch.device) -> List[Tuple[str, str, str, float, float, float]]:
    
    results = []
    
    for pep_idx, (alpha_peptide, alpha_modification, alpha_crosslink_site, 
                 beta_peptide, beta_modification, beta_crosslink_site) in enumerate(
        zip(alpha_peptides, alpha_modifications, alpha_crosslink_sites, 
            beta_peptides, beta_modifications, beta_crosslink_sites)):
            
        theoretical_mass = _calculate_peptide_mass(aa_dict, alpha_peptide, alpha_modification) + _calculate_peptide_mass(aa_dict, beta_peptide, beta_modification) + crosslinker_mass
        mass_diff = precursor_mass - theoretical_mass
        if mass_diff > -20.0:
            ppm_diff = abs(1E+6 * mass_diff / theoretical_mass)

            alpha_b, alpha_y, beta_b, beta_y, alpha_sig_ions, beta_sig_ions = calculate_alpha_ions(
                aa_dict, signature_types, alpha_peptide, alpha_modification, 
                alpha_crosslink_site, beta_peptide, beta_modification, 
                beta_crosslink_site, precursor_mass, charge, crosslinker_mass)

            theoretical_spectrum, matched_ions, theoretical_mz, theoretical_intensity = create_theoretical_spectrum(
                alpha_b, alpha_y, beta_b, beta_y, alpha_sig_ions, beta_sig_ions,
                alpha_crosslink_site, beta_crosslink_site, charge, signature_types)
            
            # Fix tensor construction warnings and add debugging
            observed_mz = spectrum[:, 0].clone().detach().to(device)
            observed_intensity = spectrum[:, 1].clone().detach().to(device)
            vectorized_observed = vectorize_spectrum(observed_mz, observed_intensity, BIN_WIDTH, BIN_OFFSET, True)
            
            # Use the fixed XCorr calculation
            xcorr = compute_xcorr(theoretical_spectrum, vectorized_observed)
            
            # Output criteria for resolving the IO bottleneck
            if xcorr >= 2.0:
                results.append((scan_num, alpha_peptide, beta_peptide, alpha_crosslink_site, beta_crosslink_site, _calculate_peptide_mass(aa_dict, alpha_peptide, alpha_modification), _calculate_peptide_mass(aa_dict, beta_peptide, beta_modification), theoretical_mass, precursor_mass, mass_diff, ppm_diff, xcorr))
    
    return results

def split_ms2_by_scans(file_path: str) -> List[List[str]]:
    """Split MS2 file into scan blocks and return list of scan content blocks."""
    scans = []
    current_scan = []
    
    with open(file_path, 'r') as f:
        for line in f:
            if line.startswith('S') and current_scan:
                scans.append(current_scan)
                current_scan = []
            current_scan.append(line)
            
    # Don't forget the last scan
    if current_scan:
        scans.append(current_scan)
        
    return scans

def distribute_scans(scans: List[List[str]], num_threads: int) -> List[List[List[str]]]:
    """Distribute scans evenly across threads."""
    # Get total number of scans
    total_scans = len(scans)
    print(f"Total scans found: {total_scans}")
    
    # Calculate scans per thread
    base_scans_per_thread = total_scans // num_threads
    extra_scans = total_scans % num_threads
    
    # Distribute scans
    distributed_scans = []
    start_idx = 0
    
    for thread_idx in range(num_threads):
        # Add one extra scan to early threads if there are remainder scans
        thread_scan_count = base_scans_per_thread + (1 if thread_idx < extra_scans else 0)
        end_idx = start_idx + thread_scan_count
        
        thread_scans = scans[start_idx:end_idx]
        distributed_scans.append(thread_scans)
        
        print(f"Thread {thread_idx}: Processing {len(thread_scans)} scans")
        start_idx = end_idx
        
    return distributed_scans

def process_scan_block(scan_block: List[str]) -> Tuple[str, List[Tuple[float, float]], float, int]:
    """Process a single scan block and return scan info - keep on CPU."""
    scan_num = ''
    spectrum = []
    precursor_mass = 0.0
    charge = 0

    for line in scan_block:
        line = line.strip()
        if line.startswith('S'):
            scan_num = line.split('\t')[1].strip()
        elif line.startswith('Z'):
            parts = line.split('\t')
            precursor_mass = float(parts[2].strip()) - proton_mass
            charge = int(parts[1].strip())
        elif not line.startswith(('H', 'I')):
            try:
                mz, intensity = map(float, line.split())
                spectrum.append((mz, intensity))
            except ValueError:
                continue
    
    return scan_num, spectrum, precursor_mass, charge

def process_scan_group(scan_blocks: List[List[str]], output_queue: Queue,
                      alpha_peptides: List[str], alpha_modifications: List[Dict[int, float]],
                      alpha_crosslink_sites: List[int],
                      beta_peptides: List[str], beta_modifications: List[Dict[int, float]],
                      beta_crosslink_sites: List[int],
                      crosslinker_mass: float, aa_dict: Dict[str, float],
                      signature_types: Dict[str, float], device: torch.device):
    """Process a group of scans assigned to a thread."""
    results = []
    
    for scan_block in scan_blocks:
        scan_num, spectrum, precursor_mass, charge = process_scan_block(scan_block)
        
        if spectrum:
            # Keep initial tensor creation on CPU
            spectrum_tensor = torch.tensor(spectrum, dtype=torch.float32)
            scan_results = process_spectrum_enhanced(
                aa_dict, signature_types, spectrum_tensor, scan_num,
                alpha_peptides, alpha_modifications, alpha_crosslink_sites,
                beta_peptides, beta_modifications, beta_crosslink_sites,
                crosslinker_mass, precursor_mass, charge, device
            )
            results.extend(scan_results)
    
    output_queue.put(results)

def process_ms2_file(file_path: str,
                    alpha_peptides: List[str], alpha_modifications: List[Dict[int, float]],
                    alpha_crosslink_sites: List[int],
                    beta_peptides: List[str], beta_modifications: List[Dict[int, float]],
                    beta_crosslink_sites: List[int],
                    output_directory: str, crosslinker_mass: float,
                    aa_dict: Dict[str, float], signature_types: Dict[str, float],
                    device: torch.device):
    
    output_file = os.path.join(output_directory, f"xl_ds_{os.path.basename(file_path).split('.')[0]}.csv")
    
    # Split file into scan blocks
    scans = split_ms2_by_scans(file_path)
    
    # Determine number of threads and distribute scans
    num_threads = os.cpu_count() or 1
    distributed_scans = distribute_scans(scans, num_threads)
    
    # Create output queue and process scan groups in parallel
    output_queue = Queue()
    error_queue = Queue()  # New queue for error reporting
    
    with concurrent.futures.ThreadPoolExecutor(max_workers=num_threads) as executor:
        futures = []
        for i, scan_group in enumerate(distributed_scans):
            future = executor.submit(
                process_scan_group, 
                scan_group, output_queue, 
                alpha_peptides, alpha_modifications, alpha_crosslink_sites,
                beta_peptides, beta_modifications, beta_crosslink_sites,
                crosslinker_mass, aa_dict, signature_types, device
            )
            futures.append(future)
        
        # Wait for all futures to complete and check for errors
        for future in concurrent.futures.as_completed(futures):
            try:
                future.result()  # This will raise any exceptions that occurred
            except Exception as e:
                print(f"Error in thread: {str(e)}")
                traceback.print_exc()
    
    # Check for errors from the error queue
    while not error_queue.empty():
        thread_idx, error_msg, traceback_str = error_queue.get()
        print(f"\nError in thread {thread_idx}:")
        print(error_msg)
        print(traceback_str)
    
    # Collect successful results
    all_results = []
    while not output_queue.empty():
        all_results.extend(output_queue.get())
    
    # Write results to file if we have any
    if all_results:
        with open(output_file, 'w', newline='') as out_file:
            csv_writer = csv.writer(out_file)
            csv_writer.writerow(['Scan', 'Alpha Peptide', 'Beta Peptide', 'Alpha XL Site', 'Beta XL Site', 'Alpha Mass', 'Beta Mass', 'Theoretical Mass', 'Observed Mass', 'Mass_Diff', 'PPM_Diff', 'Xcorr'])
            for result in all_results:
                csv_writer.writerow(result)
        print(f"Results written to {output_file}")
    else:
        print("Warning: No results were generated due to errors in processing")
    
def main(peptides_csv_path: str, output_directory: str, ms2_file_paths: List[str], crosslinker_mass: float, aa_dict: Dict[str, float], signature_types: Dict[str, float], device: torch.device):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Read peptides from a single CSV file
    peptides, modifications, crosslink_sites = read_peptides(peptides_csv_path)
    
    # Generate all combinations of peptides
    (alpha_peptides, alpha_modifications, alpha_crosslink_sites,
     beta_peptides, beta_modifications, beta_crosslink_sites) = generate_all_combinations(
         peptides, modifications, crosslink_sites)
    
    for file_path in ms2_file_paths:
        print(f"Processing MS2 file: {file_path}")
        process_ms2_file(file_path, alpha_peptides, alpha_modifications, alpha_crosslink_sites, 
                        beta_peptides, beta_modifications, beta_crosslink_sites, 
                        output_directory, crosslinker_mass, aa_dict, signature_types, device)
        print(f"Finished processing {file_path}")

if __name__ == "__main__":
    main(peptides_csv_path = r'C:\env\test\A1AT-ApoA2_ds1.csv',  # Changed to a single CSV file
         output_directory = r'C:\env\test', 
         ms2_file_paths = [
             r'C:\Crux\data\20260130\20260130-1-62.ms2',
             r'C:\Crux\data\20260122\20260122-TYG-1_62.ms2',
             ],
         crosslinker_mass=crosslinker_mass,
         aa_dict=aa_dict,
         signature_types=signature_types,
         device=device
         )

end_time = current_time()
print(f"Runtime: {format_runtime(start_time, end_time)}")