#ifndef IO_HPP
#define IO_HPP

#include <vector>
#include <string>
#include <complex>
#include <iostream>
#include <H5Cpp.h>

/**
 * Read dataset from HDF5 file into a vector
 * Template function that handles different data types (float, int, double)
 * 
 * @tparam T Data type to read (float, int, or double)
 * @param file HDF5 file object to read from
 * @param path Path to dataset within the HDF5 file
 * @param buffer Output vector to store the data
 */
template<typename T>
void read_dataset(const H5::H5File& file, const std::string& path, std::vector<T>& buffer) {
    try {
        H5::DataSet dataset = file.openDataSet(path);
        H5::DataSpace dataspace = dataset.getSpace();
        size_t numPoints = dataspace.getSimpleExtentNpoints();
        buffer.resize(numPoints);
        
        H5::PredType type = H5::PredType::NATIVE_DOUBLE;
        if constexpr (std::is_same_v<T, float>) type = H5::PredType::NATIVE_FLOAT;
        else if constexpr (std::is_same_v<T, int>) type = H5::PredType::NATIVE_INT;
        
        dataset.read(buffer.data(), type);
    } catch (...) {
        std::cerr << "oops, couldn't read " << path << std::endl;
    }
}

/**
 * Get dimensions of a dataset in HDF5 file
 * 
 * @param file HDF5 file object to read from
 * @param path Path to dataset within the HDF5 file
 * @return Vector of dimensions (e.g., [n_waves, n_transmits, n_channels, n_samples])
 */
std::vector<hsize_t> get_dims(const H5::H5File& file, const std::string& path);

/**
 * Write beamformed movie data to HDF5 file for analysis in Python
 * 
 * @param filename Output HDF5 filename
 * @param movie_data Beamformed complex image data [n_waves][height][width]
 * @param n_waves Number of wave frames
 * @param width Image width in pixels
 * @param height Image height in pixels
 * @param prf Pulse repetition frequency in Hz
 * @param f0 Center frequency of ultrasound pulse in Hz
 * @param c Speed of sound in m/s
 * @param fs Sampling frequency in Hz
 * @param depth_start Starting depth of imaging region in meters
 * @param depth_end Ending depth of imaging region in meters
 * @param pitch_start Starting x-coordinate of imaging region in meters
 * @param pitch_end Ending x-coordinate of imaging region in meters
 */
void write_hdf5_output(const std::string& filename, 
                       const std::vector<std::complex<float>>& movie_data,
                       int n_waves, int width, int height,
                       float prf, float f0, float c, float fs,
                       float depth_start, float depth_end,
                       float pitch_start, float pitch_end);

#endif
