#include <iostream>
#include <vector>
#include <cmath>
#include <fstream>
#include <complex>
#include <random>
#include <algorithm>
#include <chrono>
#include <iomanip>

using namespace std;

// Konstanta dan parameter global
const double PI = 3.14159265358979323846;
const int N = 1024;        // Jumlah sampel sinyal
const int M = 32;          // Jumlah harmonik (truncated Fourier)
const int MAX_ITER = 100;  // Jumlah maksimum iterasi untuk transformasi L1
const double EPSILON = 1e-6; // Nilai kecil untuk menghindari pembagian dengan nol

// Fungsi Gaussian untuk membuat komponen sinyal EKG
// n: waktu, mu: posisi pusat, b: lebar, alpha: amplitudo
double gaussian(double n, double mu, double b, double alpha) {
    return alpha * exp(-pow((n - mu), 2) / (2 * pow(b, 2)));
}

// Fungsi untuk menghasilkan sinyal EKG realistis dengan beberapa komponen
// Menghasilkan sinyal yang mirip dengan rekaman EKG nyata dengan gelombang P, kompleks QRS, dan gelombang T
vector<double> generateRealisticECG() {
    vector<double> x(N, 0.0);
    
    // Gelombang P (depolarisasi atrium)
    double pAmplitude = 0.25;  // Amplitudo gelombang P
    double pWidth = 20.0;      // Lebar gelombang P
    double pPosition = 200.0;  // Posisi gelombang P
    
    // Kompleks QRS (depolarisasi ventrikel)
    double qAmplitude = -0.2;  // Amplitudo gelombang Q (negatif)
    double qWidth = 8.0;       // Lebar gelombang Q
    double qPosition = 295.0;  // Posisi gelombang Q
    
    double rAmplitude = 1.0;
    double rWidth = 10.0;
    double rPosition = 300.0;
    
    double sAmplitude = -0.3;
    double sWidth = 10.0;
    double sPosition = 307.0;
    
    // T wave (ventricular repolarization)
    double tAmplitude = 0.35;
    double tWidth = 40.0;
    double tPosition = 400.0;
    
    // Baseline wander (low frequency noise)
    double baselineAmplitude = 0.05;
    double baselineFreq = 0.2;
    
    // Generate signal components
    for (int n = 0; n < N; ++n) {
        // Add P wave
        x[n] += gaussian(n, pPosition, pWidth, pAmplitude);
        
        // Add QRS complex
        x[n] += gaussian(n, qPosition, qWidth, qAmplitude);
        x[n] += gaussian(n, rPosition, rWidth, rAmplitude);
        x[n] += gaussian(n, sPosition, sWidth, sAmplitude);
        
        // Add T wave
        x[n] += gaussian(n, tPosition, tWidth, tAmplitude);
        
        // Add baseline wander
        x[n] += baselineAmplitude * sin(2.0 * PI * baselineFreq * n / N);
    }
    
    return x;
}

// Generate synthetic ECG-like signal (2 Gaussian peaks)
vector<double> generateECGSignal() {
    vector<double> x(N);
    for (int n = 0; n < N; ++n) {
        double t = n;
        double qrs = gaussian(t, 300, 30, 1.0);   // fast component
        double twave = gaussian(t, 700, 100, 0.6); // slow component
        x[n] = qrs + twave;
    }
    return x;
}

// Add noise to the signal
vector<double> addNoise(const vector<double>& signal, double snr_db) {
    // Convert SNR from dB to linear scale
    double snr = pow(10, snr_db / 10.0);
    
    // Calculate signal power
    double signal_power = 0.0;
    for (double val : signal) {
        signal_power += val * val;
    }
    signal_power /= signal.size();
    
    // Calculate noise power based on SNR
    double noise_power = signal_power / snr;
    double noise_stddev = sqrt(noise_power);
    
    // Generate random noise
    default_random_engine generator(static_cast<unsigned int>(chrono::system_clock::now().time_since_epoch().count()));
    normal_distribution<double> distribution(0.0, noise_stddev);
    
    // Add noise to the signal
    vector<double> noisy_signal(signal.size());
    for (size_t i = 0; i < signal.size(); ++i) {
        noisy_signal[i] = signal[i] + distribution(generator);
    }
    
    return noisy_signal;
}

// Compute complex exponential basis matrix (N x M)
vector<vector<complex<double>>> computePhi() {
    vector<vector<complex<double>>> Phi(N, vector<complex<double>>(M));
    for (int n = 0; n < N; ++n)
        for (int k = 0; k < M; ++k)
            Phi[n][k] = exp(complex<double>(0, 2 * PI * k * n / N));
    return Phi;
}

// Discrete Fourier Transform (DFT)
vector<complex<double>> computeDFT(const vector<double>& x) {
    vector<complex<double>> X(N, 0);
    for (int k = 0; k < N; ++k) {
        for (int n = 0; n < N; ++n) {
            double angle = -2 * PI * k * n / N;
            X[k] += complex<double>(x[n] * cos(angle), x[n] * sin(angle));
        }
    }
    return X;
}

// Inverse Discrete Fourier Transform (IDFT)
vector<double> computeIDFT(const vector<complex<double>>& X) {
    vector<double> x(N, 0);
    for (int n = 0; n < N; ++n) {
        for (int k = 0; k < N; ++k) {
            double angle = 2 * PI * k * n / N;
            x[n] += real(X[k] * complex<double>(cos(angle), sin(angle)));
        }
        x[n] /= N;
    }
    return x;
}

// Menghitung spektrum daya dari sinyal
// Spektrum daya adalah kuadrat magnitud dari koefisien Fourier
vector<double> powerSpectrum(const vector<complex<double>>& X) {
    vector<double> power(X.size());
    for (size_t i = 0; i < X.size(); ++i) {
        power[i] = norm(X[i]); // |X[i]|^2 = (bagian riil)^2 + (bagian imajiner)^2
    }
    return power;
}

// Transformasi Fourier norma-L2 (least squares)
// Metode ini meminimalkan jumlah kuadrat error antara sinyal asli dan aproksimasi
vector<complex<double>> l2_fourier(const vector<double>& x, const vector<vector<complex<double>>>& Phi) {
    vector<complex<double>> c(M, 0);
    for (int k = 0; k < M; ++k) {
        for (int n = 0; n < N; ++n)
            // Menggunakan inner product dengan konjugat untuk mendapatkan koefisien Fourier
            c[k] += x[n] * conj(Phi[n][k]);
        c[k] /= (double)N;  // Normalisasi dengan jumlah sampel
    }
    return c;
}

// Rekonstruksi sinyal dari koefisien Fourier
// Menggunakan koefisien dan basis Fourier untuk mendapatkan kembali sinyal
vector<double> reconstructSignal(const vector<complex<double>>& c, const vector<vector<complex<double>>>& Phi) {
    vector<double> xhat(N, 0);
    for (int n = 0; n < N; ++n)
        for (int k = 0; k < M; ++k)
            // Penjumlahan komponen harmonik (ambil bagian riil saja karena sinyal asli riil)
            xhat[n] += real(c[k] * Phi[n][k]);
    return xhat;
}

// Menerapkan filter band-pass pada sinyal dalam domain frekuensi
// Filter ini hanya menyimpan komponen frekuensi antara lowCutoff dan highCutoff
vector<complex<double>> applyBandPassFilter(const vector<complex<double>>& X, double lowCutoff, double highCutoff) {
    vector<complex<double>> filtered = X;
    
    // Filter: hanya menyimpan frekuensi antara lowCutoff dan highCutoff
    for (int k = 0; k < N; ++k) {
        double freq = static_cast<double>(k) / N;
        if (freq > 0.5) freq = 1.0 - freq;  // Menangani frekuensi negatif (aliasing)
        
        // Nolkan komponen frekuensi di luar rentang yang diinginkan
        if (freq < lowCutoff || freq > highCutoff) {
            filtered[k] = complex<double>(0, 0);
        }
    }
    
    return filtered;
}

// Transformasi Fourier norma-L1 menggunakan metode iterative reweighted least squares (IRLS)
// Metode ini lebih tahan terhadap outlier dan noise dibanding metode L2
vector<complex<double>> l1_fourier(const vector<double>& x, const vector<vector<complex<double>>>& Phi) {
    vector<complex<double>> c(M, 0.0);
    vector<double> w(N, 1.0);  // Bobot awal

    // Inisialisasi dengan solusi norma-L2
    c = l2_fourier(x, Phi);
    
    // Mulai iterasi perbaikan
    for (int iter = 0; iter < MAX_ITER; ++iter) {
        // Hitung residual saat ini
        vector<double> residual = x;
        for (int n = 0; n < N; ++n) {
            for (int k = 0; k < M; ++k) {
                residual[n] -= real(c[k] * Phi[n][k]);
            }
            // Perbarui bobot berdasarkan residual (aproksimasi norma-L1)
            // Untuk L1, bobot berbanding terbalik dengan nilai absolut residual
            w[n] = 1.0 / (fabs(residual[n]) + EPSILON);
        }
          // Selesaikan masalah kuadrat terkecil berbobot
        vector<complex<double>> numerator(M, 0.0);
        vector<complex<double>> denominator(M, 0.0);

        for (int k = 0; k < M; ++k) {
            for (int n = 0; n < N; ++n) {
                complex<double> phi = Phi[n][k];
                numerator[k] += w[n] * x[n] * conj(phi);  // Pembilang dari formula IRLS
                denominator[k] += w[n] * norm(phi);       // Penyebut dari formula IRLS
            }
            c[k] = numerator[k] / (denominator[k] + EPSILON);  // Perbarui koefisien Fourier
        }
    }
    return c;
}

// Menghitung Mean Square Error (MSE) antara sinyal asli dan sinyal hasil rekonstruksi
// MSE adalah rata-rata dari kuadrat selisih antara nilai sinyal asli dan rekonstruksi
double computeMSE(const vector<double>& original, const vector<double>& reconstructed) {
    double mse = 0.0;
    for (int i = 0; i < original.size(); ++i) {
        double error = original[i] - reconstructed[i];  // Hitung selisih
        mse += error * error;  // Kuadratkan selisih dan tambahkan ke akumulator
    }
    mse /= original.size();  // Bagi dengan jumlah sampel untuk mendapatkan rata-rata
    return mse;
}

// Menghitung Signal-to-Noise Ratio (SNR) dalam dB
// SNR adalah perbandingan antara daya sinyal dan daya noise, dalam skala logaritmik
double computeSNR(const vector<double>& signal, const vector<double>& noise) {
    double signal_power = 0.0;  // Daya sinyal
    double noise_power = 0.0;   // Daya noise
    
    for (int i = 0; i < signal.size(); ++i) {
        signal_power += signal[i] * signal[i];  // Akumulasi kuadrat sinyal
        noise_power += noise[i] * noise[i];     // Akumulasi kuadrat noise
    }
    
    if (noise_power < EPSILON) return INFINITY;  // Hindari pembagian dengan nol
    return 10.0 * log10(signal_power / noise_power);  // Konversi ke skala dB
}

// Menyimpan sinyal ke file CSV untuk visualisasi
void saveCSV(const vector<double>& original, const vector<double>& l2, const vector<double>& l1, 
             const vector<double>& noisy, const string& filename) {
    ofstream file(filename);
    file << "n,original,l2,l1,noisy\n";  // Header kolom
    for (int i = 0; i < N; ++i)
        file << i << "," << original[i] << "," << l2[i] << "," << l1[i] << "," << noisy[i] << "\n";
    file.close();
    cout << "Output saved to " << filename << endl;
}

// Menyimpan spektrum frekuensi ke file CSV
void saveSpectrumCSV(const vector<double>& spectrum, const string& filename) {
    ofstream file(filename);
    file << "k,power\n";  // Header kolom: k (indeks frekuensi), power (daya)
    for (int i = 0; i < spectrum.size(); ++i)
        file << i << "," << spectrum[i] << "\n";
    file.close();
    cout << "Spectrum saved to " << filename << endl;
}

// Menyimpan koefisien Fourier ke file CSV untuk visualisasi
void saveCoefficientsCSV(const vector<complex<double>>& c_l2, const vector<complex<double>>& c_l1, 
                        const string& filename) {
    ofstream file(filename);
    file << "k,l2_real,l2_imag,l2_magnitude,l1_real,l1_imag,l1_magnitude\n";
    for (int i = 0; i < M; ++i) {
        file << i << ","
             << real(c_l2[i]) << "," << imag(c_l2[i]) << "," << abs(c_l2[i]) << ","
             << real(c_l1[i]) << "," << imag(c_l1[i]) << "," << abs(c_l1[i]) << "\n";
    }
    file.close();
    cout << "Coefficients saved to " << filename << endl;
}

// Melakukan analisis perbandingan antara metode L1 dan L2 pada berbagai kondisi noise
// Fungsi ini membandingkan performa kedua metode pada berbagai level SNR
void performNoiseAnalysis(const vector<double>& x) {
    // Vektor untuk menyimpan nilai-nilai SNR untuk analisis
    vector<double> snr_values = {5.0, 10.0, 15.0, 20.0, 25.0};  // Nilai SNR dalam dB
    vector<double> l1_mse_values;  // Untuk menyimpan nilai MSE metode L1
    vector<double> l2_mse_values;  // Untuk menyimpan nilai MSE metode L2
      cout << "\nNoise Performance Analysis:" << endl;
    cout << "-------------------------" << endl;
    cout << setw(10) << "SNR (dB)" << setw(15) << "L1 MSE" << setw(15) << "L2 MSE" << setw(15) << "Improvement (%)" << endl;
    
    // Buat file CSV untuk menyimpan hasil analisis
    ofstream file("noise_analysis.csv");
    file << "snr_db,l1_mse,l2_mse,improvement\n";
    
    // Hitung matriks basis Fourier sekali saja (efisiensi)
    auto Phi = computePhi();
    
    // Iterasi melalui berbagai nilai SNR
    for (double snr_db : snr_values) {
        // Tambahkan noise ke sinyal dengan SNR yang ditentukan
        vector<double> x_noisy = addNoise(x, snr_db);
        
        // Hitung aproksimasi L1 dan L2
        auto c_l1 = l1_fourier(x_noisy, Phi);  // Koefisien Fourier dengan metode L1
        auto c_l2 = l2_fourier(x_noisy, Phi);  // Koefisien Fourier dengan metode L2
        
        // Rekonstruksi sinyal dari koefisien
        auto x_l1 = reconstructSignal(c_l1, Phi);
        auto x_l2 = reconstructSignal(c_l2, Phi);
        
        // Hitung MSE untuk kedua metode
        double l1_mse = computeMSE(x, x_l1);  // Error metode L1
        double l2_mse = computeMSE(x, x_l2);  // Error metode L2
        
        // Simpan nilai-nilai untuk analisis
        l1_mse_values.push_back(l1_mse);
        l2_mse_values.push_back(l2_mse);
        
        // Hitung persentase peningkatan metode L1 dibandingkan L2
        // Positif berarti L1 lebih baik daripada L2
        double improvement = 100.0 * (l2_mse - l1_mse) / l2_mse;
        
        // Display results
        cout << setw(10) << snr_db
             << setw(15) << l1_mse
             << setw(15) << l2_mse
             << setw(15) << improvement << "%" << endl;
             
        // Save to CSV
        file << snr_db << "," << l1_mse << "," << l2_mse << "," << improvement << "\n";
    }
    
    file.close();
    cout << "Noise analysis saved to noise_analysis.csv" << endl;
}

int main() {
    cout << "ECG Signal Decomposition using Fourier Transform" << endl;
    cout << "===============================================" << endl;
    
    // Step 1: Generate ECG signals
    cout << "Generating ECG signals..." << endl;
    vector<double> x_simple = generateECGSignal();
    vector<double> x_realistic = generateRealisticECG();
    
    // Use realistic ECG for analysis
    vector<double> x = x_realistic;
    
    // Add noise to create a noisy version (SNR = 15dB)
    vector<double> x_noisy = addNoise(x, 15.0);
    
    // Step 2: Compute basis
    cout << "Computing Fourier basis..." << endl;
    auto Phi = computePhi();
    
    // Step 3: Compute full DFT for spectrum analysis
    cout << "Computing DFT for spectral analysis..." << endl;
    auto X_dft = computeDFT(x);
    auto power = powerSpectrum(X_dft);
    
    // Step 4: ℓ2 Fourier approximation (least squares)
    cout << "Computing L2-norm Fourier approximation..." << endl;
    auto c_l2 = l2_fourier(x, Phi);
    auto x_l2 = reconstructSignal(c_l2, Phi);
    
    // Step 5: ℓ1 Fourier approximation (robust to outliers)
    cout << "Computing L1-norm Fourier approximation..." << endl;
    auto c_l1 = l1_fourier(x, Phi);
    auto x_l1 = reconstructSignal(c_l1, Phi);
    
    // Step 6: Compute L2 and L1 approximations on noisy data
    cout << "Computing approximations on noisy data..." << endl;
    auto c_l2_noisy = l2_fourier(x_noisy, Phi);
    auto x_l2_noisy = reconstructSignal(c_l2_noisy, Phi);
    
    auto c_l1_noisy = l1_fourier(x_noisy, Phi);
    auto x_l1_noisy = reconstructSignal(c_l1_noisy, Phi);
    
    // Step 7: Calculate metrics
    cout << "\nReconstruction Error Metrics:" << endl;
    cout << "----------------------------" << endl;
    cout << "Clean Signal:" << endl;
    cout << "L2 MSE: " << computeMSE(x, x_l2) << endl;
    cout << "L1 MSE: " << computeMSE(x, x_l1) << endl;
      cout << "\nNoisy Signal:" << endl;
    cout << "L2 MSE: " << computeMSE(x, x_l2_noisy) << endl;
    cout << "L1 MSE: " << computeMSE(x, x_l1_noisy) << endl;
    
    // Step 8: Perform detailed noise analysis
    cout << "\nPerforming noise analysis..." << endl;
    performNoiseAnalysis(x);
    
    // Step 9: Output results for visualization
    cout << "\nSaving results for visualization..." << endl;
    saveCSV(x, x_l2, x_l1, x_noisy, "ecg_fourier_output.csv");
    saveSpectrumCSV(power, "ecg_spectrum.csv");
    saveCoefficientsCSV(c_l2, c_l1, "ecg_coefficients.csv");
    
    // Perform noise analysis
    performNoiseAnalysis(x);
    
    cout << "\nAnalysis complete! Run the Python plotting scripts to visualize the results." << endl;
    
    return 0;
}
