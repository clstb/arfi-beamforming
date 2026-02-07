#include "hilbert.hpp"
#include <fftw3.h>

// convert rf data into complex signals
// - move to frequency domain
// - dump negative frequencies
// - double positive frequencies
// - go back to time domain
std::vector<std::complex<float>> apply_hilbert(const std::vector<float>& input, int num_lines, int num_samples) {
    std::vector<std::complex<float>> output(input.size());
    
    // allocate memory using fftw's optimized functions
    float* in_ptr = fftwf_alloc_real(num_samples);
    fftwf_complex* out_fft_ptr = fftwf_alloc_complex(num_samples / 2 + 1); 
    fftwf_complex* analytic_ptr = fftwf_alloc_complex(num_samples);
    
    // create plans for the forward (real-to-complex) and backward (complex-to-complex) transforms
    fftwf_plan p_r2c = fftwf_plan_dft_r2c_1d(num_samples, in_ptr, out_fft_ptr, FFTW_ESTIMATE);
    fftwf_plan p_c2c = fftwf_plan_dft_1d(num_samples, analytic_ptr, analytic_ptr, FFTW_BACKWARD, FFTW_ESTIMATE);

    for (int line = 0; line < num_lines; ++line) {
        // load data for one line (one sensor's recording for one pulse)
        for (int i = 0; i < num_samples; ++i) {
            in_ptr[i] = input[line * num_samples + i];
        }

        // move to frequency domain
        fftwf_execute(p_r2c);

        // create the analytic signal:
        // 1. keep dc and nyquist components alone
        // 2. double all positive frequencies
        // 3. kill off all negative frequencies
        analytic_ptr[0][0] = out_fft_ptr[0][0];
        analytic_ptr[0][1] = out_fft_ptr[0][1];

        for (int i = 1; i < (num_samples + 1) / 2; ++i) { 
             analytic_ptr[i][0] = out_fft_ptr[i][0] * 2.0f;
             analytic_ptr[i][1] = out_fft_ptr[i][1] * 2.0f;
        }

        if (num_samples % 2 == 0) {
            analytic_ptr[num_samples/2][0] = out_fft_ptr[num_samples/2][0];
            analytic_ptr[num_samples/2][1] = out_fft_ptr[num_samples/2][1];
        }

        for (int i = (num_samples/2) + 1; i < num_samples; ++i) {
            analytic_ptr[i][0] = 0.0f;
            analytic_ptr[i][1] = 0.0f;
        }
        
        // go back to time domain
        fftwf_execute(p_c2c);

        // normalise by N because fftw doesn't do that automatically
        float scale = 1.0f / num_samples;
        for (int i = 0; i < num_samples; ++i) {
            output[line * num_samples + i] = std::complex<float>(analytic_ptr[i][0] * scale, analytic_ptr[i][1] * scale);
        }
    }

    // clean up memory
    fftwf_destroy_plan(p_r2c);
    fftwf_destroy_plan(p_c2c);
    fftwf_free(in_ptr);
    fftwf_free(out_fft_ptr);
    fftwf_free(analytic_ptr);

    return output;
}
