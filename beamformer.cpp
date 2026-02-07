#include "beamformer.hpp"
#include "constants.hpp"
#include <cmath>
#include <iostream>
#include <omp.h>
#include <atomic>

std::vector<std::complex<float>> run_beamformer(
    const std::vector<std::complex<float>>& raw_iq,
    const std::vector<float>& el_x,
    const std::vector<float>& el_z,
    float t0, float fs, float c,
    int n_waves, int n_channels, int n_samples
) {
    int width = IMAGE_WIDTH;
    int height = IMAGE_HEIGHT;
    std::vector<std::complex<float>> movie_data(n_waves * width * height, {0,0});
    
    // figure out the physical spacing between pixels in our output image
    // convert from pixel coordinates to real-world meters
    float x_min = el_x[0];
    float x_max = el_x[n_channels-1];
    float dx = (x_max - x_min) / (width - 1);
    float dz = (DEPTH_End - DEPTH_Start) / (height - 1);

    // process each wave (time snapshot) independently
    // parallelize across waves using openmp so multiple cpu cores can work simultaneously
    std::atomic<int> completed_waves{0};
    int last_reported_progress = 0;

    #pragma omp parallel for schedule(dynamic)
    for (int w = 0; w < n_waves; ++w) {
        
        // loop through every pixel in our output image grid
        for (int iz = 0; iz < height; ++iz) {
            float z_pix = DEPTH_Start + iz * dz;
            for (int ix = 0; ix < width; ++ix) {
                float x_pix = x_min + ix * dx;
                
                // calculate how long it takes sound to travel from probe to this pixel
                float t_tx = z_pix / c + t0; 
                
                std::complex<float> sum_val = 0.0f;
                
                // for this pixel, check what each sensor channel 'heard' at the right time
                for (int ch = 0; ch < n_channels; ++ch) {
                    // calculate how long it took for the reflected sound to get back to this sensor
                    float rx_dist = std::hypot(x_pix - el_x[ch], z_pix - el_z[ch]);
                    float t_rx = rx_dist / c;
                    
                    // total time is transmit time plus receive time
                    float t_total = t_tx + t_rx;
                    
                    // figure out which sample in our recorded data corresponds to this time
                    float sample_idx = (t_total - t0) * fs;
                    
                    // only use this sample if it's within our recorded range
                    if (sample_idx >= 0 && sample_idx < n_samples - 1) {
                        // use linear interpolation between two samples for smoother, higher quality images
                        int idx0 = (int)sample_idx;
                        float frac = sample_idx - idx0;
                        size_t offset = (size_t)w * n_channels * n_samples + ch * n_samples + idx0;
                        std::complex<float> val0 = raw_iq[offset];
                        std::complex<float> val1 = raw_iq[offset + 1];
                        sum_val += val0 * (1.0f - frac) + val1 * frac;
                    }
                }
                
                // save this pixel's final value to our 3d output array [wave, z, x]
                size_t frame_offset = (size_t)w * width * height;
                movie_data[frame_offset + iz * width + ix] = sum_val; 
            }
        }
        // print progress every 10 waves
        // use a critical section so multiple threads don't print at the same time
        int finished = ++completed_waves;
        if (finished % 10 == 0 || finished == n_waves) {
            #pragma omp critical
            {
                if (finished > last_reported_progress) {
                    std::cout << "beamformed wave " << finished << "/" << n_waves << "\r" << std::flush;
                    last_reported_progress = finished;
                }
            }
        }
    }
    std::cout << std::endl;
    return movie_data;
}
