#include <iostream>
#include <vector>
#include <string>
#include <chrono>
#include <iomanip>
#include <H5Cpp.h>

#include "constants.hpp"
#include "io.hpp"
#include "hilbert.hpp"
#include "beamformer.hpp"
#include "beamformer_cuda.hpp"

using namespace H5;

struct Config {
    std::string filename = "ARFI_dataset.uff";
    bool use_cuda = false;
    bool use_cuda_opt = false;
};

Config parse_args(int argc, char* argv[]) {
    Config config;
    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg == "--cuda-opt") {
            config.use_cuda_opt = true;
        } else if (arg == "--cuda") {
            config.use_cuda = true;
        } else {
            config.filename = arg;
        }
    }
    return config;
}

void run_beamforming_pipeline(const Config& config) {
    // start timing everything
    auto total_start = std::chrono::high_resolution_clock::now();
    double time_read = 0.0, time_hilbert = 0.0, time_beamform = 0.0, time_write = 0.0;

    H5File file(config.filename, H5F_ACC_RDONLY);

    // open the hdf5 file and figure out how big the dataset is
    auto dims_data = get_dims(file, "channel_data/data");
    if (dims_data.size() < 3) {
        throw std::runtime_error("oops, bad file format!");
    }
    
    int n_waves = dims_data[0];
    int n_channels = dims_data[2];
    int n_samples = dims_data[3];
    
    std::cout << "waves: " << n_waves << ", channels: " << n_channels << ", samples: " << n_samples << std::endl;

    // load all the metadata parameters needed for calculation
    std::vector<float> tmp_scalar;
    read_dataset(file, "channel_data/sampling_frequency", tmp_scalar);
    float fs = tmp_scalar[0];
    read_dataset(file, "channel_data/sound_speed", tmp_scalar);
    float c = tmp_scalar[0];
    read_dataset(file, "channel_data/initial_time", tmp_scalar);
    float t0 = tmp_scalar[0];

    // extract the sensor positions from the probe geometry data
    std::vector<float> probe_geom;
    read_dataset(file, "channel_data/probe/geometry", probe_geom);
    std::vector<float> el_x(n_channels), el_z(n_channels);
    for (int i = 0; i < n_channels; ++i) {
            el_x[i] = probe_geom[0 * n_channels + i]; // x dimension (lateral)
            el_z[i] = probe_geom[2 * n_channels + i]; // z dimension (depth)
    }
    std::cout << "probe width: " << el_x[0] << " to " << el_x[n_channels-1] << " meters" << std::endl;
    
    // find the center frequency of the ultrasound pulse
    float f0 = 5.208e6; // default if not found
    try {
        read_dataset(file, "channel_data/pulse/center_frequency", tmp_scalar);
        f0 = tmp_scalar[0];
    } catch(...) {
        std::cout << "warning: couldn't find f0, using fallback" << std::endl;
    }
    std::cout << "pulse frequency: " << f0 << " hz" << std::endl;

    // pulse repetition frequency (prf) is the ultrasound pulse rate
    float prf = 5000.0f;
    std::cout << "using prf: " << prf << " hz" << std::endl;

    // step 1: read the big raw radio-frequency data from the hdf5 file
    std::cout << "loading raw rf data..." << std::endl;
    auto t1 = std::chrono::high_resolution_clock::now();
    std::vector<float> raw_rf;
    read_dataset(file, "channel_data/data", raw_rf);
    auto t2 = std::chrono::high_resolution_clock::now();
    time_read = std::chrono::duration<double>(t2 - t1).count();
    
    // step 2: apply hilbert transform to get the complex iq signal
    std::cout << "calculating analytic signal (hilbert)..." << std::endl;
    t1 = std::chrono::high_resolution_clock::now();
    auto raw_iq = apply_hilbert(raw_rf, n_waves * n_channels, n_samples);
    t2 = std::chrono::high_resolution_clock::now();
    time_hilbert = std::chrono::duration<double>(t2 - t1).count();
    
    // step 3: run the beamforming algorithm to convert raw signals into an image
    std::string method = config.use_cuda_opt ? " (CUDA optimized)" : (config.use_cuda ? " (CUDA naive)" : " (CPU)");
    std::cout << "running beamformer engine" << method << "..." << std::endl;
    t1 = std::chrono::high_resolution_clock::now();
    auto movie_data = config.use_cuda_opt
        ? run_beamformer_cuda_optimized(raw_iq, el_x, el_z, t0, fs, c, n_waves, n_channels, n_samples)
        : (config.use_cuda 
            ? run_beamformer_cuda(raw_iq, el_x, el_z, t0, fs, c, n_waves, n_channels, n_samples)
            : run_beamformer(raw_iq, el_x, el_z, t0, fs, c, n_waves, n_channels, n_samples));
    t2 = std::chrono::high_resolution_clock::now();
    time_beamform = std::chrono::duration<double>(t2 - t1).count();

    // step 4: write the beamformed data to an hdf5 file
    t1 = std::chrono::high_resolution_clock::now();
    write_hdf5_output("beamformed_output.h5", movie_data, n_waves, IMAGE_WIDTH, IMAGE_HEIGHT, 
                        prf, f0, c, fs, DEPTH_Start, DEPTH_End, PITCH_X_START, PITCH_X_END);
    t2 = std::chrono::high_resolution_clock::now();
    time_write = std::chrono::duration<double>(t2 - t1).count();

    // print out timing information
    auto total_end = std::chrono::high_resolution_clock::now();
    double time_total = std::chrono::duration<double>(total_end - total_start).count();

    std::cout << "\n--- perf summary ---" << std::endl;
    std::cout << std::fixed << std::setprecision(4);
    std::cout << "read file:    " << time_read << " s" << std::endl;
    std::cout << "hilbert:      " << time_hilbert << " s" << std::endl;
    std::cout << "beamforming:  " << time_beamform << " s" << std::endl;
    std::cout << "write result: " << time_write << " s" << std::endl;
    std::cout << "--------------------" << std::endl;
    std::cout << "total:        " << time_total << " s" << std::endl;
}

int main(int argc, char* argv[]) {
    try {
        Config config = parse_args(argc, argv);
        run_beamforming_pipeline(config);
    } catch (Exception &e) {
        e.printErrorStack();
        return 1;
    } catch (std::exception &e) {
        std::cerr << "something went wrong: " << e.what() << std::endl;
        return 1;
    }
    
    return 0;
}
