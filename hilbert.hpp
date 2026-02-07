#ifndef HILBERT_HPP
#define HILBERT_HPP

#include <vector>
#include <complex>

/**
 * Apply Hilbert transform to convert real RF signals to complex analytic signals
 * 
 * The Hilbert transform creates the analytic representation of a real signal,
 * which is useful for extracting envelope and phase information from RF data.
 * 
 * @param input Real-valued RF signal data [num_lines * num_samples]
 * @param num_lines Number of independent signal lines (e.g., channels × waves)
 * @param num_samples Number of samples per line
 * @return Complex analytic signal [num_lines * num_samples]
 */
std::vector<std::complex<float>> apply_hilbert(const std::vector<float>& input, int num_lines, int num_samples);

#endif
