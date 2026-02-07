#ifndef CONSTANTS_HPP
#define CONSTANTS_HPP

/**
 * Imaging region and output grid configuration
 * 
 * These constants define the physical bounds of the imaging area and the
 * resolution of the output beamformed image grid.
 */

/// Starting x-coordinate of imaging region (lateral direction) in meters
constexpr float PITCH_X_START = -20e-3; 

/// Ending x-coordinate of imaging region (lateral direction) in meters
constexpr float PITCH_X_END = 20e-3;

/// Starting depth of imaging region (axial direction) in meters
constexpr float DEPTH_Start = 0.0f;

/// Ending depth of imaging region (axial direction) in meters
constexpr float DEPTH_End = 30e-3f;

/// Output image width in pixels (lateral resolution)
constexpr int IMAGE_WIDTH = 256; 

/// Output image height in pixels (axial resolution)
constexpr int IMAGE_HEIGHT = 256; 

#endif
