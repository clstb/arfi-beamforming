import h5py
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, FFMpegWriter
from scipy.ndimage import uniform_filter

def run_kasai(movie_data, c, prf, f0, width, height, n_waves):
    """
    this function handles the displacement estimation using kasai's algorithm.
    """
    # scaling factor to turn phase values into distance in meters.
    # ultrasound phase shift is 4*pi*f0 * (2*d/c), solve for d.
    # note that prf isn't used here because absolute displacement is desired, not speed.
    scale = c / (4.0 * np.pi * f0) 
    
    # turn the flat data into a 3d volume of [time, depth, width]
    stack = movie_data.reshape((n_waves, height, width))
    
    # calculate the phase difference between consecutive frames using autocorrelation.
    # z is the current frame, multiply it by the conjugate of the previous one.
    z = stack[1:]
    R = z * np.conj(stack[:-1])
    Power = np.abs(z)**2
    
    # ultrasound signals are noisy, smoothing the cross-correlation over a small window is required
    # these specific gate sizes (6x4x2) follow the matlab example implementation.
    avg_size = (6, 4, 2)
    R_smooth = uniform_filter(R.real, size=avg_size) + 1j * uniform_filter(R.imag, size=avg_size)
    P_smooth = uniform_filter(Power, size=avg_size)
    
    # take the angle of the smoothed vector and scale it.
    # invert the sign so that the primary push shows up as white in the final video.
    displacement = -scale * np.angle(R_smooth)
    
    # if the signal is too weak, the phase is just noise. mask those areas out.
    mask = P_smooth < (np.max(P_smooth) * 0.001)
    displacement[mask] = 0.0
    
    return displacement

def main():
    # load beamformed data from the hdf5 fil
    print("Reading beamformed_output.h5...")
    with h5py.File("beamformed_output.h5", "r") as f:
        dset = f["beamformed_iq"]
        # grab all the metadata (sound speed, frequency, etc.)
        attrs = {k: dset.attrs[k] for k in dset.attrs}
        # read the raw complex numbers. hdf5 saves them as pairs of floats.
        raw = dset[:]
        data = raw[..., 0] + 1j * raw[..., 1]

    n_waves, h, w = data.shape
    print(f"Data shape: {n_waves}x{h}x{w}")
    
    # simple b-mode generation. sum up the wave complex data to compound it.
    print("Generating B-Mode...")
    b_mode = 20 * np.log10(np.abs(np.sum(data, axis=0)) + 1e-6)
    # normalise so the peak is at 0db.
    b_mode -= np.max(b_mode) 
    
    # setting up the physical axes so the plot shows mm instead of pixels.
    extent = [attrs['pitch_start']*1e3, attrs['pitch_end']*1e3, attrs['depth_end']*1e3, attrs['depth_start']*1e3]
    
    plt.figure(figsize=(8, 6))
    plt.imshow(b_mode, cmap='gray', aspect='auto', extent=extent, vmin=-30, vmax=0)
    plt.title("B-Mode Image")
    plt.xlabel("Width [mm]"); plt.ylabel("Depth [mm]")
    plt.colorbar(label="Amplitude [dB]")
    plt.tight_layout()
    plt.savefig("b_mode.png", dpi=300)
    print("Saved b_mode.png")
    
    # run kasai algorithm to track the actual shear wave moving through the tissue.
    print("Calculating Displacement...")
    disp_map = run_kasai(data, attrs['c'], attrs['prf'], attrs['f0'], w, h, n_waves)
    
    # make an mp4 movie to see the wave move through the tissue.
    print("Generating Video...")
    fig, ax = plt.subplots(figsize=(6, 6))
    # use the 'hot' map so it looks exactly like the matlab example.
    im = ax.imshow(disp_map[0], cmap='hot', aspect='auto', extent=extent, vmin=-1e-7, vmax=2e-7)
    ax.set_title("Displacement Map")
    ax.set_xlabel("Width [mm]"); ax.set_ylabel("Depth [mm]")
    fig.colorbar(im, label=r"Displacement [m]")
    
    # this function handles updating the frame for the animation.
    def update(i):
        im.set_data(disp_map[i])
        ax.set_title(f"Displacement - Frame {i}")
        return [im]
        
    # save the final result as an mp4 file using ffmpeg.
    anim = FuncAnimation(fig, update, frames=len(disp_map), interval=100, blit=True)
    anim.save("wave.mp4", writer=FFMpegWriter(fps=10))
    print("Saved wave.mp4")

if __name__ == "__main__":
    main()