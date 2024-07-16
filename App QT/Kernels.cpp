#include "imageTransformations.h"




QImage ImageTransformations::Kernels::createGaborKernel(int ksize, double sigma, double theta, double lambda, double gamma, double psi)
{

	int halfSize = ksize / 2;
	QImage kernel(ksize, ksize, QImage::Format_Grayscale8);
	kernel.fill(0);

	double thetaRad = theta * M_PI / 180.0;
	double cosTheta = cos(thetaRad);
	double sinTheta = sin(thetaRad);

	for (int y = -halfSize; y <= halfSize; ++y) {
		for (int x = -halfSize; x <= halfSize; ++x) {
			double xPrime = x * cosTheta + y * sinTheta;
			double yPrime = -x * sinTheta + y * cosTheta;

			double gaussian = exp(-(xPrime * xPrime + gamma * gamma * yPrime * yPrime) / (2 * sigma * sigma));
			double sinusoid = cos(2 * M_PI * xPrime / lambda + psi);

			double value = gaussian * sinusoid * 255.0;

			kernel.setPixel(x + halfSize, y + halfSize, qBound(0, static_cast<int>(value), 255));
		}
	}
	return kernel;



}

std::vector<std::vector<std::vector<double>>> ImageTransformations::Kernels::createGarborKernels(const std::vector<int>& orientationsPerScale, int imageSize)
{

	int Nscales = orientationsPerScale.size();
	int Nfilters = 0;
	for (int i = 0; i < Nscales; ++i) {
		Nfilters += orientationsPerScale[i];
	}

	std::vector<std::vector<std::vector<double>>> filters(Nfilters, std::vector<std::vector<double>>(imageSize, std::vector<double>(imageSize, 0.0)));

	filters.reserve(Nfilters);

	double param[4];

	std::vector<double> fx(imageSize * imageSize);
	std::vector<double> fy(imageSize * imageSize);
	std::vector<double> fr(imageSize * imageSize);
	std::vector<double> t(imageSize * imageSize);

	for (int y = 0; y < imageSize; ++y) {
		for (int x = 0; x < imageSize; ++x) {
			int index = y * imageSize + x;
			fx[index] = (x - imageSize / 2);
			fy[index] = (y - imageSize / 2);
			fr[index] = sqrt(fx[index] * fx[index] + fy[index] * fy[index]);
			t[index] = atan2(fy[index], fx[index]);
		}
	}

	int l = 0;
	for (int i = 0; i < Nscales; ++i) {
		for (int j = 0; j < orientationsPerScale[i]; ++j) {
			param[0] = 0.35;
			param[1] = 0.3 / pow(1.85, i);
			param[2] = 16 * pow(orientationsPerScale[i], 2) / pow(32, 2);
			param[3] = M_PI / orientationsPerScale[i] * j;

			for (int y = 0; y < imageSize; ++y) {
				for (int x = 0; x < imageSize; ++x) {
					int index = y * imageSize + x;
					double tr = t[index] + param[3];
					if (tr > M_PI) tr -= 2 * M_PI;
					if (tr < -M_PI) tr += 2 * M_PI;

					double value = exp(-10 * param[0] * pow(fr[index] / (imageSize / 2) / param[1] - 1, 2) - 2 * param[2] * M_PI * pow(tr, 2));
					filters[l][x][y] = value;
				}
			}
			++l;

		}
	}






	return filters;






}
