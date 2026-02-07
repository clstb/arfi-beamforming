#include "beamformer_cuda.hpp"
#include "constants.hpp"
#include <cuda_runtime.h>
#include <iostream>
#include <cmath>

// constant memory for probe geometry - broadcast to all threads, cached
// this is much faster than reading from global memory repeatedly
__constant__ float c_el_x[256];  // max 256 channels
__constant__ float c_el_z[256];

// optimized cuda kernel for delay-and-sum beamforming
// uses constant memory, texture cache (__ldg), fma intrinsics, and launch bounds
__global__ void __launch_bounds__(256) beamformer_kernel_optimized(
    const float2* __restrict__ raw_iq,
    float2* __restrict__ output,
    float t0, float fs, float c,
    int n_channels, int n_samples,
    int width, int height,
    float x_min, float dx, float dz,
    float depth_start,
    float inv_c // passed precomputed inverse
) {
    // figure out which pixel this thread is responsible for
    int ix = blockIdx.x * blockDim.x + threadIdx.x;
    int iz = blockIdx.y * blockDim.y + threadIdx.y;
    
    // exit early if outside the image bounds
    if (ix >= width || iz >= height) return;
    
    // convert pixel indices to physical coordinates in meters
    float x_pix = x_min + ix * dx;
    float z_pix = depth_start + iz * dz;
    
    // calculate transmit time from probe to this pixel
    // t_tx = z_pix / c + t0 -> z_pix * inv_c + t0
    float t_tx = __fmaf_rn(z_pix, inv_c, t0);
    
    // accumulator for beamformed value from all channels
    float sum_real = 0.0f;
    float sum_imag = 0.0f;
    
    // loop through all sensor channels to accumulate contributions
    // unroll more aggressively for better instruction-level parallelism
    #pragma unroll 8
    for (int ch = 0; ch < n_channels; ++ch) {
        // calculate distance from pixel to sensor using constant memory
        float dx_ch = x_pix - c_el_x[ch];
        float dz_ch = z_pix - c_el_z[ch];
        
        // use fast square root and multiply instead of divide
        float rx_dist = __fsqrt_rn(dx_ch * dx_ch + dz_ch * dz_ch);
        float t_rx = rx_dist * inv_c;
        
        // total travel time
        float t_total = t_tx + t_rx;
        
        // convert time to sample index
        float sample_idx = (t_total - t0) * fs;
        
        // only process if within valid range
        if (sample_idx >= 0.0f && sample_idx < n_samples - 1.0f) {
            // use fast float to int conversion
            int idx0 = __float2int_rd(sample_idx);
            float frac = sample_idx - __int2float_rn(idx0);
            
            // calculate offset - organized for coalesced access
            int offset = blockIdx.z * n_channels * n_samples + ch * n_samples + idx0;
            
            // read two samples for interpolation
            // use __ldg to force read through texture cache
            // this is critical for the scatter-gather access pattern of beamforming
            float2 val0 = __ldg(&raw_iq[offset]);
            float2 val1 = __ldg(&raw_iq[offset + 1]);
            
            // linearly interpolate using fused multiply-add for speed
            // val = val0 * (1-frac) + val1 * frac
            float one_minus_frac = 1.0f - frac;
            sum_real = __fmaf_rn(val0.x, one_minus_frac, __fmaf_rn(val1.x, frac, sum_real));
            sum_imag = __fmaf_rn(val0.y, one_minus_frac, __fmaf_rn(val1.y, frac, sum_imag));
        }
    }
    
    // write final beamformed value to output
    // output layout: [wave][z][x]
    int out_idx = blockIdx.z * width * height + iz * width + ix;
    output[out_idx] = make_float2(sum_real, sum_imag);
}

std::vector<std::complex<float>> run_beamformer_cuda_optimized(
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
    
    // precompute inverse speed of sound
    float inv_c = 1.0f / c;
    
    // copy probe geometry to constant memory for fast cached access
    cudaMemcpyToSymbol(c_el_x, el_x.data(), n_channels * sizeof(float));
    cudaMemcpyToSymbol(c_el_z, el_z.data(), n_channels * sizeof(float));
    
    // allocate output on host
    std::vector<std::complex<float>> movie_data(n_waves * width * height);
    
    // allocate device memory
    float2 *d_raw_iq, *d_output;
    
    size_t raw_iq_size = raw_iq.size() * sizeof(float2);
    size_t output_size = movie_data.size() * sizeof(float2);
    
    cudaMalloc(&d_raw_iq, raw_iq_size);
    cudaMalloc(&d_output, output_size);
    
    // copy input data to device
    cudaMemcpy(d_raw_iq, raw_iq.data(), raw_iq_size, cudaMemcpyHostToDevice);
    
    // optimized block size for better occupancy and memory bandwidth
    // 32x8 = 256 threads per block, matching our launch bounds
    dim3 blockSize(32, 8);
    dim3 gridSize(
        (width + blockSize.x - 1) / blockSize.x,
        (height + blockSize.y - 1) / blockSize.y,
        n_waves
    );
    
    std::cout << "launching optimized CUDA kernel with grid (" 
              << gridSize.x << ", " << gridSize.y << ", " << gridSize.z 
              << ") and block (" << blockSize.x << ", " << blockSize.y << ")" << std::endl;
    
    // launch optimized kernel
    beamformer_kernel_optimized<<<gridSize, blockSize>>>(
        d_raw_iq, d_output,
        t0, fs, c, n_channels, n_samples,
        width, height, x_min, dx, dz, DEPTH_Start,
        inv_c
    );
    
    // check for errors
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        std::cerr << "CUDA kernel launch error: " << cudaGetErrorString(err) << std::endl;
    }
    
    // wait for completion
    cudaDeviceSynchronize();
    
    // copy results back
    cudaMemcpy(movie_data.data(), d_output, output_size, cudaMemcpyDeviceToHost);
    
    // cleanup
    cudaFree(d_raw_iq);
    cudaFree(d_output);
    
    std::cout << "optimized CUDA beamforming complete!" << std::endl;
    
    return movie_data;
}
