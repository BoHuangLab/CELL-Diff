import math

def stable_sigmoid(x):
    if x >= 0:
        z = math.exp(-x)
        sig = 1 / (1 + z)
        return sig
    else:
        z = math.exp(x)
        sig = z / (1 + z)
        return sig

def logsnr_schedule_cosine(t, logsnr_min=-15, logsnr_max=15):
    t_min = math.atan(math.exp(-0.5 * logsnr_max))
    t_max = math.atan(math.exp(-0.5 * logsnr_min))
    return -2 * math.log(math.tan(t_min + t * (t_max - t_min)))

def logsnr_schedule_cosine_shifted(t, image_d, noise_d=64):
    return logsnr_schedule_cosine(t) + 2 * math.log(noise_d / image_d)

def logsnr_schedule_cosine_shifted_interpolated(t, image_d, noise_d_low, noise_d_high):
    logsnr_low = logsnr_schedule_cosine_shifted(t , image_d, noise_d_low)
    logsnr_high = logsnr_schedule_cosine_shifted(t , image_d, noise_d_high)
    return t * logsnr_low + (1 - t) * logsnr_high

def cosine_shifted_alpha_bar(t, image_d):
    return stable_sigmoid(logsnr_schedule_cosine_shifted(t, image_d))