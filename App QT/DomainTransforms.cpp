#include "imageTransformations.h"


QImage ImageTransformations::DomainTransforms::DiscreteFFT(QImage image)
{
	int width = image.width();
	int height = image.height();

	std::vector<std::vector<std::complex<double>>> data(height, std::vector <std::complex<double >>(width));

	//We fill the data with grayscale values

	for (int y = 0; y < height; ++y) {
		for (int x = 0; x < width; ++x) {
			QRgb pixel = image.pixel(x, y);
			double g = qGray(pixel);
			data[y][x] = std::complex<double>(g, 0.0);
		}
	}

	auto fftResult = Computations::Math::fft2D(data, false);

	//Create magnitude and phase images

	QImage magnitudeImage = QImage(width, height, QImage::Format_Grayscale8);
	QImage phaseImage = QImage(width, height, QImage::Format_Grayscale8);

	double maxMag = 0.0;

	for (int y = 0; y < height; ++y) {
		for (int x = 0; x < width; x++) {
			double real = fftResult[y][x].real();
			double imag = fftResult[y][x].imag();

			double magnitude = std::sqrt(real * real + imag * imag);
			double phase = std::atan2(imag, real);

			if (magnitude > maxMag) {
				maxMag = magnitude;
			}

			magnitudeImage.setPixel(x, y, qGray(magnitude));
			phaseImage.setPixel(x, y, qGray(phase));
		}
	}

	for (int y = 0; y < height; ++y) {
		for (int x = 0; x < width; x++) {
			int magPix = qGray(magnitudeImage.pixel(x, y));
			magPix = static_cast<int> ((magPix / maxMag) * 255);
			magnitudeImage.setPixel(x, y, qRgb(magPix, magPix, magPix));
		}
	}

	magnitudeImage.save("magnitudeImage.png");
	phaseImage.save("phaseImage.png");

	return magnitudeImage;


}

QImage ImageTransformations::DomainTransforms::InverseFFT(QImage image)
{

	int width = image.width();
	int height = image.height();

	std::vector<std::vector<std::complex<double>>> data(height, std::vector<std::complex<double>>(width));

	//Fill the data with grayscale values

	for (int y = 0; y < height; ++y) {
		for (int x = 0; x < width; ++x) {
			QRgb pixel = image.pixel(x, y);
			double g = qGray(pixel);
			data[y][x] = std::complex<double>(g, 0.0);
		}
	}

	auto ifftRes = Computations::Math::fft2D(data, true);

	QImage inverseImage(width, height, QImage::Format_Grayscale8);

	for (int y = 0; y < height; ++y) {
		for (int x = 0; x < width; ++x) {
			double value = qBound(0.0, value, 255.0);
			inverseImage.setPixel(x, y, qGray(value));
		}
	}

	return inverseImage;

}

