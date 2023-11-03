import numpy as np
import scipy


def get_random_noise_floor(img_count, random_count=16):
    # return a noise floor with padding=5, mean error=1
    lh = 100
    lw = [100, 100, 412]
    noise_list = []
    for i in range(img_count):
        print("img -> " + str(i), end="\r")
        noise = np.random.normal(0, 1, size=(random_count, lh, lw[1]))
        padded = np.pad(noise, np.array([[0, 0], [0, 0], [0, lw[2]]]), "constant", constant_values=0)
        spectrum = np.average(np.abs(np.fft.fft(padded, axis=2)[:, :, : lw[0]]) ** 2, axis=0)
        spectrum -= np.average(spectrum)
        spectrum /= np.std(spectrum)
        noise_list.append(spectrum)

    return np.array(noise_list, dtype=np.float32)


def get_random_wind_profile(img_count):
    # return a randomly generated wind profile using random walk, unit in pixel
    sigma_0 = 5.0
    sigma_d = 0.6
    lh = 100
    profile_list = []
    for i in range(img_count):
        profile = np.zeros((lh,), dtype=np.float64)
        rnd = np.random.normal(0, 1, size=(lh,))
        profile[0] = sigma_0 * rnd[0]
        for j in range(1, lh):
            profile[j] = profile[j - 1] + sigma_d * rnd[j]
        profile = scipy.ndimage.gaussian_filter(profile, sigma=0.5)
        profile_list.append(profile)
    return np.array(profile_list, dtype=np.float32)


# def get_random_wind_profile(img_count, layer_count):
#     # return a randomly generated wind profile using random walk, unit in pixel
#     sigma_0 = 5.0
#     sigma_d = 0.6
#     sigma_t = 3
#     lh = 100
#     profile_list = []
#     for i in range(img_count):
#         profile1 = np.zeros((lh,), dtype=np.float64)
#         rnd = np.random.normal(0, 1, size=(lh,))
#         profile1[0] = sigma_0*rnd[0]
#         for j in range(1, lh):
#             profile1[j] = profile1[j-1]+sigma_d*rnd[j]
#         # profile1=scipy.ndimage.gaussian_filter(profile1, sigma=0.5)
#         profile2=profile1+sigma_t*scipy.ndimage.gaussian_filter(np.random.normal(0, 1, size=(lh,)), sigma=3)
#         profile3=profile2+sigma_t*scipy.ndimage.gaussian_filter(np.random.normal(0, 1, size=(lh,)), sigma=3)
#         profile4=profile1+sigma_t*scipy.ndimage.gaussian_filter(np.random.normal(0, 1, size=(lh,)), sigma=3)
#         profile5=profile4+sigma_t*scipy.ndimage.gaussian_filter(np.random.normal(0, 1, size=(lh,)), sigma=3)
#         profile_list.append([profile2, profile3, profile1, profile4, profile5])
#     return np.array(profile_list, dtype=np.float32)


def get_signal_spectrum(wind_profile_list):
    lh = 100
    lw = 100
    range_gate = np.linspace(5, 104, 100) * 30
    snr_0 = 2e3
    snr_gate = snr_0 * ((range_gate[0] / range_gate) ** 3)

    def get_shape(center):
        # sinc^2 shape generator
        xs = np.pi / 5.12 * (np.linspace(-49.5, 49.5, 100) - center)
        ys1 = np.sin(xs) ** 2 / (xs**2 + 1e-20)
        ys1 = scipy.ndimage.gaussian_filter(ys1, sigma=0.9)
        return ys1

    spectrum_list = []
    for profile in wind_profile_list:
        spectrum = []
        for j in range(lh):
            speed = profile[j]
            line = snr_gate[j] * get_shape(speed)
            spectrum.append(line)
        spectrum_list.append(spectrum)
    return np.array(spectrum_list, dtype=np.float32)
