import numpy as np
import matplotlib.pyplot as plt
import os
import pandas as pd

# 🔧 Ścieżka do folderu logów
log_path = "../../gnuradio/yabool2001"

# Ścieżki do plików binarnych
file_bytes2chunks = os.path.join(log_path, "tx_bytes2chunks.8b")
file_chunks2symbols = os.path.join(log_path, "tx_chunks2symbols.32fc")
file_rrc_filter = os.path.join(log_path, "tx_rrc_filter.32fc")

# Wczytanie danych
chunks = np.fromfile(file_bytes2chunks, dtype=np.uint8)
symbols = np.fromfile(file_chunks2symbols, dtype=np.complex64)
filtered = np.fromfile(file_rrc_filter, dtype=np.complex64)

print(f"Loaded {len(chunks)} chunks")
print(f"Loaded {len(symbols)} symbols")
print(f"Loaded {len(filtered)} filtered samples")

# =====================
# ZAPIS DO CSV (z przecinkiem jako separator dziesiętny i średnikiem jako delimiter)
# =====================

# 1. Chunks
chunks_df = pd.DataFrame({'chunk': chunks})
chunks_csv_path = os.path.join(log_path, "tx_bytes2chunks.csv")
chunks_df.to_csv(chunks_csv_path, sep=';', index=False)
print(f"Chunks zapisane do: {chunks_csv_path}")

# 2. Symbols (complex)
symbols_df = pd.DataFrame({
    'real': symbols.real,
    'imag': symbols.imag
})
symbols_csv_path = os.path.join(log_path, "tx_chunks2symbols.csv")
symbols_df.to_csv(symbols_csv_path, sep=';', index=False, float_format='%.10f', decimal=',')
print(f"Symbols zapisane do: {symbols_csv_path}")

# 3. Filtered samples (complex)
amplitude = np.abs(filtered)
phase = np.angle(filtered)
sample_index = np.arange(len(filtered))

filtered_df = pd.DataFrame({
    'sample_index': sample_index,
    'real': filtered.real,
    'imag': filtered.imag,
    'amplitude': amplitude,
    'phase_rad': phase
})

filtered_csv_path = os.path.join(log_path, "tx_rrc_filter.csv")
filtered_df.to_csv(
    filtered_csv_path,
    sep=';',
    index=False,
    float_format='%.10f',
    decimal=','
)
print(f"Filtered samples zapisane do: {filtered_csv_path}")


# =====================
# WYKRESY
# =====================

# Histogram symboli QPSK
plt.figure()
plt.title("Histogram of QPSK Chunks (0–3)")
plt.hist(chunks, bins=[-0.5,0.5,1.5,2.5,3.5], rwidth=0.8)
plt.xticks([0,1,2,3])
plt.grid(True)

# Konstelacja po mapowaniu
plt.figure()
plt.title("Constellation after mapping (Chunks to Symbols)")
plt.plot(symbols.real, symbols.imag, 'o', alpha=0.5)
plt.xlabel("In-phase (I)")
plt.ylabel("Quadrature (Q)")
plt.axis('equal')
plt.grid(True)

# Przebieg I/Q po filtrze RRC
plt.figure()
plt.title("RRC Filtered Signal (first 500 samples)")
plt.plot(filtered.real[:500], label='I (real)')
plt.plot(filtered.imag[:500], label='Q (imag)', linestyle='--')
plt.xlabel("Sample Index")
plt.legend()
plt.grid(True)

plt.show()
