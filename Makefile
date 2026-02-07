# Makefile for ARFI Beamformer
# Simplifies building and running the beamformer

.PHONY: all build clean run run-cuda run-cuda-opt help

# Default target
all: build run run-cuda run-cuda-opt visualize

# Build the project
build:
	@mkdir -p build
	@cd build && cmake .. && $(MAKE) -j$$(nproc)
	@echo "Build complete! Executable: ./build/arfi_beamformer"

# Clean build artifacts
clean:
	@rm -rf build
	@rm -f beamformed_output.h5
	@echo "Cleaned build directory and output files"

# Run with CPU (multi-core OpenMP)
run: build
	@echo "Running beamformer with multi-core CPU..."
	@./build/arfi_beamformer

# Run with CUDA GPU
run-cuda: build
	@echo "Running beamformer with CUDA GPU (naive)..."
	@./build/arfi_beamformer --cuda

# Run with optimized CUDA GPU
run-cuda-opt: build
	@echo "Running beamformer with optimized CUDA GPU..."
	@./build/arfi_beamformer --cuda-opt

# Visualize results using Python
visualize:
	@echo "Visualizing results..."
	@python3 visualize.py beamformed_output.h5

# Show help
help:
	@echo "ARFI Beamformer Makefile"
	@echo ""
	@echo "Targets:"
	@echo "  make build        - Build the project"
	@echo "  make run          - Build and run with multi-core CPU"
	@echo "  make run-cuda     - Build and run with CUDA GPU (naive)"
	@echo "  make run-cuda-opt - Build and run with CUDA GPU (optimized)"
	@echo "  make visualize    - Visualize the output HDF5 file"
	@echo "  make clean        - Remove build artifacts"
	@echo "  make help         - Show this help message"
	@echo ""
	@echo "Usage examples:"
	@echo "  make              # Build the project"
	@echo "  make run          # Run CPU version"
	@echo "  make run-cuda     # Run naive CUDA version"
	@echo "  make run-cuda-opt # Run optimized CUDA version"
	@echo "  make visualize    # Generate B-mode image from output"
