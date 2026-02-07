#ifndef BEAMFORMER_CUDA_HPP
#define BEAMFORMER_CUDA_HPP

#include <vector>
#include <complex>

/**
 * CUDA GPU delay-and-sum beamformer (naive implementation)
 * 
 * @param raw_iq Complex IQ data from all channels [n_waves][n_channels][n_samples]
 * @param el_x Sensor x-coordinates in meters [n_channels]
 * @param el_z Sensor z-coordinates in meters [n_channels]
 * @param t0 Initial time offset in seconds
 * @param fs Sampling frequency in Hz
 * @param c Speed of sound in m/s
 * @param n_waves Number of wave frames to process
 * @param n_channels Number of sensor channels
 * @param n_samples Number of samples per channel
 * @return Beamformed complex image data [n_waves][IMAGE_HEIGHT][IMAGE_WIDTH]
 */
std::vector<std::complex<float>> run_beamformer_cuda(
    const std::vector<std::complex<float>>& raw_iq,
    const std::vector<float>& el_x,
    const std::vector<float>& el_z,
    float t0, float fs, float c,
    int n_waves, int n_channels, int n_samples
);

/**
 * Optimized CUDA GPU delay-and-sum beamformer
 * 
 * Uses constant memory, fused multiply-add, loop unrolling, and optimized block size
 * 
 * @param raw_iq Complex IQ data from all channels [n_waves][n_channels][n_samples]
 * @param el_x Sensor x-coordinates in meters [n_channels]
 * @param el_z Sensor z-coordinates in meters [n_channels]
 * @param t0 Initial time offset in seconds
 * @param fs Sampling frequency in Hz
 * @param c Speed of sound in m/s
 * @param n_waves Number of wave frames to process
 * @param n_channels Number of sensor channels
 * @param n_samples Number of samples per channel
 * @return Beamformed complex image data [n_waves][IMAGE_HEIGHT][IMAGE_WIDTH]
 */
std::vector<std::complex<float>> run_beamformer_cuda_optimized(
    const std::vector<std::complex<float>>& raw_iq,
    const std::vector<float>& el_x,
    const std::vector<float>& el_z,
    float t0, float fs, float c,
    int n_waves, int n_channels, int n_samples
);

#endif
