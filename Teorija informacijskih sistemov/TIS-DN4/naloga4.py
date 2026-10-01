import numpy as np

def create_filter(slika: np.array, prag: int) -> np.array:
    N, M = slika.shape
    n, m = np.ogrid[:N, :M]
    center_n, center_m = N // 2, M // 2
    dist = ((n - center_n)/N)**2 + ((m - center_m)/M)**2

    H = np.zeros_like(slika, dtype=float)
    H[dist >= prag**2] = 1

    return H

def calculate_mi(original: np.array, obdelana: np.array) -> float:
    G = 256
    bins = np.linspace(0, 1, G + 1)

    hist_sv, _ = np.histogram(original, bins=bins)
    hist_so, _ = np.histogram(obdelana, bins=bins)

    hist_joint, _, _ = np.histogram2d(
        original.ravel(),
        obdelana.ravel(),
        bins=[bins, bins]
    )

    N_total = original.size
    p_sv = hist_sv.astype(float) / N_total
    p_so = hist_so.astype(float) / N_total
    p_joint = hist_joint / N_total

    H_sv = -np.sum(p_sv[p_sv > 0] * np.log2(p_sv[p_sv > 0]))
    H_so = -np.sum(p_so[p_so > 0] * np.log2(p_so[p_so > 0]))
    H_obeh = -np.sum(p_joint[p_joint > 0] * np.log2(p_joint[p_joint > 0]))

    MI = H_sv + H_so - H_obeh

    return MI



def naloga4(slika: np.array, prag: int) -> float:
    """
    Poenostavi sliko

    Parameteri
    ----------
    slika : numpy array
        vhodna slika 
    prag : int
        najmanjša frekvenca, ki se ohrani na sliki
    
    Vrnjena vrednost
    ----------------
    MI : float
         medsebojna informacija v bitih
    """

    slika = np.clip(slika, 0, 255) / 255.0
    F = np.fft.fft2(slika)
    F_shifted = np.fft.fftshift(F)

    H = create_filter(slika, prag)

    F_filtered = F_shifted * H
    F_unshifted = np.fft.ifftshift(F_filtered)
    f_filtered = np.fft.ifft2(F_unshifted)
    
    
    obdelana_slika = np.real(f_filtered)
    obdelana_slika = np.clip(obdelana_slika, 0, 1)

    MI = calculate_mi(slika, obdelana_slika)

    return MI