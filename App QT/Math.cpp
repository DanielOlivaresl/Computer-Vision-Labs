#include "computations.h"


void Computations::Math::fft1D(std::vector<std::complex<double>>&data, bool invert)
{


	int n = data.size();
	if (n <= 1) {
		return;
	}

	std::vector<std::complex<double>> even(n / 2);
	std::vector<std::complex<double>> odd(n / 2);


	for (int i = 0; i < n / 2; i++) {
		even[i] = data[i * 2];
		odd[i] = data[i * 2 + 1];
	}

	fft1D(even, invert);
	fft1D(odd, invert);

	double ang = 2 * M_PI / n * (invert ? -1 : 1);
	std::complex<double> w(1), wn(cos(ang), sin(ang));

	for (int i = 0; i < n / 2; ++i) {
		data[i] = even[i] + w * odd[i];
		data[i + n / 2] = even[i] - w * odd[i];
		if (invert) {
			data[i] /= 2;
			data[i + n / 2] /= 2;
		}
		w *= wn;
	}


}

std::vector<std::vector<std::complex<double>>> Computations::Math::fft2D(const std::vector<std::vector<std::complex<double>>>& data, bool inverted)
{
	int height = data.size();
	int width = data[0].size();

	std::vector<std::vector<std::complex<double>>> result = data;
	//Apply fft to rows
	for (int y = 0; y < height; ++y) {
		fft1D(result[y], inverted);
	}

	//Transpose the result
	std::vector<std::vector<std::complex<double>>> transposed(width, std::vector<std::complex<double>>(height));

	for (int y = 0; y < height; ++y) {
		for (int x = 0; x < width; ++x) {
			transposed[x][y] = result[y][x];
		}
	}


	// Apply FFT to columns (transposed rows)
	for (size_t x = 0; x < width; ++x) {
		fft1D(transposed[x], inverted);
	}

	// Transpose the result back
	for (size_t y = 0; y < height; ++y) {
		for (size_t x = 0; x < width; ++x) {
			result[y][x] = transposed[x][y];
		}
	}

	return result;



}
