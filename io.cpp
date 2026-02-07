#include "io.hpp"
#include <iostream>

// extract the dimensions (shape) of a dataset from an hdf5 file
// how many waves, channels, and samples
std::vector<hsize_t> get_dims(const H5::H5File& file, const std::string& path) {
    try {
        H5::DataSet dataset = file.openDataSet(path);
        H5::DataSpace dataspace = dataset.getSpace();
        int rank = dataspace.getSimpleExtentNdims();
        std::vector<hsize_t> dims(rank);
        dataspace.getSimpleExtentDims(dims.data(), NULL);
        return dims;
    } catch (...) {
        return {};
    }
}

// write beamformed 3d movie data to an hdf5 file
// used to load it into python later for visualization and analysis
void write_hdf5_output(const std::string& filename, 
                       const std::vector<std::complex<float>>& movie_data,
                       int n_waves, int width, int height,
                       float prf, float f0, float c, float fs,
                       float depth_start, float depth_end,
                       float pitch_start, float pitch_end) {

    H5::H5File file(filename, H5F_ACC_TRUNC);

    // complex numbers are stored as pairs of floats [real, imaginary]
    hsize_t dims_complex[4] = {(hsize_t)n_waves, (hsize_t)height, (hsize_t)width, 2};
    H5::DataSpace dataspace_complex(4, dims_complex);
    
    H5::DataSet dataset = file.createDataSet("beamformed_iq", H5::PredType::NATIVE_FLOAT, dataspace_complex);
    dataset.write(movie_data.data(), H5::PredType::NATIVE_FLOAT);

    // attach metadata attributes to the dataset so python knows about
    // physical parameters like sound speed, frequencies, and image dimensions
    auto write_scalar_attr = [&](const char* name, float val) {
        H5::DataSpace attr_space(H5S_SCALAR);
        H5::Attribute attr = dataset.createAttribute(name, H5::PredType::NATIVE_FLOAT, attr_space);
        attr.write(H5::PredType::NATIVE_FLOAT, &val);
    };

    write_scalar_attr("prf", prf);
    write_scalar_attr("f0", f0);
    write_scalar_attr("c", c);
    write_scalar_attr("fs", fs);
    write_scalar_attr("depth_start", depth_start);
    write_scalar_attr("depth_end", depth_end);
    write_scalar_attr("pitch_start", pitch_start);
    write_scalar_attr("pitch_end", pitch_end);

    std::cout << "saved results to " << filename << std::endl;
}
