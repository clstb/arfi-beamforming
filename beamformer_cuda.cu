#include "beamformer_cuda.hpp"
#include "constants.hpp"
#include <cuda_runtime.h>
#include <iostream>
#include <cmath>

// cuda kernel for delay-and-sum beamforming
// each thread computes one pixel in the output image
__global__ void beamformer_kernel(
    const float2* raw_iq,
    const float* el_x,
    const float* el_z,
    float2* output,
    float t0, float fs, float c,
    int n_channels, int n_samples,
    int width, int height,
    float x_min, float dx, float dz,
    float depth_start
) {
    // figure out which pixel this thread is responsible for based on block and thread indices
    int ix = blockIdx.x * blockDim.x + threadIdx.x;
    int iz = blockIdx.y * blockDim.y + threadIdx.y;
    
    // exit early if outside the image bounds
    if (ix >= width || iz >= height) return;
    
    // convert pixel indices to physical coordinates in meters
    float x_pix = x_min + ix * dx;
    float z_pix = depth_start + iz * dz;
    
    // calculate how long it takes for sound to travel from probe to this pixel
    float t_tx = z_pix / c + t0;
    
    // this will accumulate the beamformed value from all sensor channels
    float2 sum_val = make_float2(0.0f, 0.0f);
    
    // loop through all sensor channels to accumulate their contributions
    for (int ch = 0; ch < n_channels; ++ch) {
        // calculate the euclidean distance from this pixel to the current sensor
        float dx_ch = x_pix - el_x[ch];
        float dz_ch = z_pix - el_z[ch];
        float rx_dist = sqrtf(dx_ch * dx_ch + dz_ch * dz_ch);
        float t_rx = rx_dist / c;
        
        // total travel time is transmit time plus receive time
        float t_total = t_tx + t_rx;
        
        // convert the travel time to a sample index in our recorded data
        float sample_idx = (t_total - t0) * fs;
        
        // only use this sample if it's within our recorded data range
        if (sample_idx >= 0.0f && sample_idx < n_samples - 1) {
            // use linear interpolation to get a value between two samples for better accuracy
            int idx0 = (int)sample_idx;
            float frac = sample_idx - idx0;
            
            // calculate where this sample lives in our big 3d data array
            // the data is organized as [wave][channel][sample]
            int offset = blockIdx.z * n_channels * n_samples + ch * n_samples + idx0;
            
            float2 val0 = raw_iq[offset];
            float2 val1 = raw_iq[offset + 1];
            
            // linearly interpolate between the two samples: val0 * (1-frac) + val1 * frac
            float2 interp_val;
            interp_val.x = val0.x * (1.0f - frac) + val1.x * frac;
            interp_val.y = val0.y * (1.0f - frac) + val1.y * frac;
            
            // add this channel's contribution to our running sum
            sum_val.x += interp_val.x;
            sum_val.y += interp_val.y;
        }
    }
    
    // write the final beamformed value for this pixel to the output array
    // output layout is [wave][z][x]
    int out_idx = blockIdx.z * width * height + iz * width + ix;
    output[out_idx] = sum_val;
}

std::vector<std::complex<float>> run_beamformer_cuda(
    const std::vector<std::complex<float>>& raw_iq,
    const std::vector<float>& el_x,
    const std::vector<float>& el_z,
    float t0, float fs, float c,
    int n_waves, int n_channels, int n_samples
) {
    int width = IMAGE_WIDTH;
    int height = IMAGE_HEIGHT;
    
    // calculate grid parameters
    float x_min = el_x[0];
    float x_max = el_x[n_channels-1];
    float dx = (x_max - x_min) / (width - 1);
    float dz = (DEPTH_End - DEPTH_Start) / (height - 1);
    
    // allocate output on host
    std::vector<std::complex<float>> movie_data(n_waves * width * height);
    
    // allocate device memory
    float2 *d_raw_iq, *d_output;
    float *d_el_x, *d_el_z;
    
    size_t raw_iq_size = raw_iq.size() * sizeof(float2);
    size_t output_size = movie_data.size() * sizeof(float2);
    size_t el_size = n_channels * sizeof(float);
    
    cudaMalloc(&d_raw_iq, raw_iq_size);
    cudaMalloc(&d_output, output_size);
    cudaMalloc(&d_el_x, el_size);
    cudaMalloc(&d_el_z, el_size);
    
    // copy input data to device
    // std::complex<float> has same memory layout as float2
    cudaMemcpy(d_raw_iq, raw_iq.data(), raw_iq_size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_el_x, el_x.data(), el_size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_el_z, el_z.data(), el_size, cudaMemcpyHostToDevice);
    
    // configure grid
    dim3 blockSize(16, 16);
    dim3 gridSize(
        (width + blockSize.x - 1) / blockSize.x,
        (height + blockSize.y - 1) / blockSize.y,
        n_waves  // process all wave frames in parallel using the z dimension
    );
    
    std::cout << "launching CUDA kernel with grid (" 
              << gridSize.x << ", " << gridSize.y << ", " << gridSize.z 
              << ") and block (" << blockSize.x << ", " << blockSize.y << ")" << std::endl;
    
    // launch kernel
    beamformer_kernel<<<gridSize, blockSize>>>(
        d_raw_iq, d_el_x, d_el_z, d_output,
        t0, fs, c, n_channels, n_samples,
        width, height, x_min, dx, dz, DEPTH_Start
    );
    
    // check for kernel launch errors
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        std::cerr << "CUDA kernel launch error: " << cudaGetErrorString(err) << std::endl;
    }
    
    // wait for kernel to complete
    cudaDeviceSynchronize();
    
    // copy results back to host
    cudaMemcpy(movie_data.data(), d_output, output_size, cudaMemcpyDeviceToHost);
    
    // free device memory
    cudaFree(d_raw_iq);
    cudaFree(d_output);
    cudaFree(d_el_x);
    cudaFree(d_el_z);
    
    std::cout << "CUDA beamforming complete!" << std::endl;
    
    return movie_data;
}
