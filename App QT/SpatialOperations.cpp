#include "imageTransformations.h"


std::vector<std::vector<double>> ImageTransformations::SpatialOperations::downN(std::vector<std::vector<double>>& image, int N)
{

	int rows = image.size();
	int cols = image[0].size();

	int blockRows = rows / N;
	int blockCols = cols / N;

	std::vector < std::vector<double>> downSampledImage(N, std::vector<double>(N, 0.0));



	for (int i = 0; i < N; ++i) {
		for (int j = 0; j < N; ++j) {
			double sum = 0.0;
			for (int k = 0; k < blockRows; ++k) {
				for (int l = 0; l < blockCols; ++l) {
					int rowIndex = i * blockRows + k;
					int colIndex = j * blockCols + l;
					sum += image[rowIndex][colIndex];
				}
			}



			downSampledImage[j][i] = sum / (blockRows * blockCols);
		}
	}

	return downSampledImage;




}

std::vector<std::vector<double>>  ImageTransformations::SpatialOperations::padArray(std::vector<std::vector<double>> image, int padSize)
{

	int width = image[0].size();
	int height = image.size();

	std::vector<std::vector<double>> paddedImage(height + 2 * padSize, std::vector<double>(width + 2 * padSize, 0.0));

	for (int y = 0; y < height; ++y) {
		for (int x = 0; x < width; ++x) {
			paddedImage[x + padSize][y + padSize] = image[x][y];
		}
	}

	return paddedImage;

}




QImage ImageTransformations::SpatialOperations::gaborFilter(const QImage& image, const QImage& kernel)
{

	int rows = image.height();
	int cols = image.width();

	int kSize = kernel.width();
	int halfSize = kSize / 2;
	QImage filteredImage(cols, rows, QImage::Format_Grayscale8);
	filteredImage.fill(0);

	for (int i = halfSize; i < rows - halfSize; ++i) {
		for (int j = halfSize; j < cols - halfSize; ++j) {
			double sum = 0.0;
			for (int ky = -halfSize; ky <= halfSize; ++ky) {
				for (int kx = -halfSize; kx <= halfSize; ++kx) {
					int pixelVal = qGray(image.pixel(j + kx, i + ky));
					int kernelVal = qGray(kernel.pixel(kx + halfSize, ky + halfSize));
					sum += pixelVal * kernelVal;
				}
			}
			filteredImage.setPixel(j, i, qBound(0, static_cast<int>(sum), 255));
		}
	}
	return filteredImage;



}




std::vector<std::vector<double>> ImageTransformations::SpatialOperations::preFilter(const std::vector<std::vector<double>>& image, double fc)
{






	int w = 5;
	double s1 = fc / sqrt(log(2.0));
	int width = image[0].size();
	int height = image.size();


	//Convert the image to grayscale and logscale

	QVector<double> imageData(width * height); //Like applying a flat transform to the image

	for (int y = 0; y < height; ++y) {
		for (int x = 0; x < width; ++x) {
			imageData[y * width + x] = log(1.0 + image[x][y]);

		}
	}




	//We pad the images to reduce the effects of boundary artifacts

	int padWidth = width + 2 * w;
	int padHeight = height + 2 * w;

	QVector<double> paddedImage(padWidth * padHeight, 0);
	for (int y = 0; y < height; ++y) {
		for (int x = 0; x < width; ++x) {
			paddedImage[(y + w) * padWidth + (x + w)] = imageData[y * width + x];
		}
	}


	//Create gaussian filter in the frequency domain

	std::vector<std::vector<std::complex<double>>> filter(padHeight, std::vector<std::complex<double>>(padWidth));
	for (int y = 0; y < padHeight; ++y) {
		for (int x = 0; x < padWidth; ++x) {
			int fx = x - padWidth / 2;
			int fy = y - padHeight / 2;
			double fr = sqrt(fx * fx + fy * fy);
			double value = exp(-(fr * fr) / (s1 * s1));
			filter[y][x] = std::complex<double>(value, 0.0);
		}
	}



	//Preform the FFT on the padded image 


	std::vector < std::vector < std::complex<double >>> paddedImageComplex(padHeight, std::vector<std::complex<double>>(padWidth));
	for (int y = 0; y < padHeight; ++y) {

		for (int x = 0; x < padWidth; ++x) {
			paddedImageComplex[y][x] = std::complex<double>(paddedImage[y * padWidth + x], 0.0);
		}
	}



	auto fftImage = Computations::Math::fft2D(paddedImageComplex, false);
	auto fftResult = Computations::Math::fft2D(filter, false);




	//Apply the gaussian filter in the frequency domain

	for (int y = 0; y < padHeight; ++y) {
		for (int x = 0; x < padWidth; ++x) {
			fftImage[y][x] *= fftResult[y][x];
		}
	}

	//Preforming inverse FFT

	auto inverseFFTImage = Computations::Math::fft2D(fftImage, true);





	//Normalize the result 

	QVector<double> filteredImage(padWidth * padHeight);
	for (int y = 0; y < height; ++y) {
		for (int x = 0; x < width; ++x) {
			filteredImage[y * padWidth + x] = inverseFFTImage[y][x].real();
		}
	}

	//Local contrast Normalization

	QVector<double> localstd(padWidth * padHeight);
	for (int y = 0; y < height; ++y) {
		for (int x = 0; x < width; ++x) {
			int idx = (y + w) * padWidth + (x + w);
			localstd[idx] = sqrt(abs(filteredImage[idx]));


		}
	}


	for (int y = 0; y < padHeight; ++y) {
		for (int x = 0; x < padWidth; ++x) {
			int idx = y * padWidth + x;
			filteredImage[idx] /= (0.2 + localstd[idx]);
		}
	}




	// Crop the image back to the original size
	std::vector<std::vector<double>> output(height, std::vector<double>(width, 0));
	for (int y = 0; y < height; ++y) {
		for (int x = 0; x < width; ++x) {
			double value = filteredImage[(y + w) * padWidth + (x + w)];
			output[x][y] = value;
		}
	}









	return output;

}


QImage ImageTransformations::SpatialOperations::convolve(QImage inputImage, Eigen::MatrixXd kernel)
{

	//We create a new image with the same size and format as the input image
	QImage resultingImage = QImage(inputImage.size(), inputImage.format());

	//we will now create an auxiliary image, that will be a padded image, so that the convolution operation can be done without going out of the border

	//We calculate the dims of the padded image
	int newWidth = inputImage.width() + kernel.cols() - 1;
	int newHeight = inputImage.height() + kernel.rows() - 1;

	

	//We create the paddedImage
	QImage paddedImage(newWidth, newHeight, inputImage.format());
	
	
	for (int i = 0; i < paddedImage.width(); i++) {
		for (int j = 0; j < paddedImage.height(); j++) {
			if (i == 0 || j == 0 || i == paddedImage.width() - 1 || j == paddedImage.height() - 1) {
				paddedImage.setPixel(i, j, QRgb(0));
			}
			else {
				paddedImage.setPixel(i, j, inputImage.pixel(i - 1, j - 1));
			}
		}
	}





	


	//We will iterate through the original Image, but will access the padded image so we dont have any out of bounds errors, and adjust the bounds accordingly

	for (int i = 1; i < paddedImage.width()-1; i++) {

		for (int j = 1; j < paddedImage.height()-1; j++) {

			//We will create a vector for each color channel

			std::vector<double> channels = { 0,0,0 };
			

			//we will iterate the kernel to preform the operations
			for (int k = 0; k < kernel.rows(); k++) {
				for (int l = 0; l < kernel.cols(); l++) {
					

					//Indexes of corresponding input image pixels, since the kernel will be situated at the middle of the input pixel, we must adjust the indexes
					int x = i - 1 + k;
					int y = j - 1 + l;


					//We now compute the values for each corresponding channel
					channels[0] += kernel(k, l) * qRed(paddedImage.pixel(x, y));
					channels[1] += kernel(k, l) * qGreen(paddedImage.pixel(x, y));
					channels[2] += kernel(k, l) * qBlue(paddedImage.pixel(x, y));
					

					//qDebug() << "{" << channels[0] << "," << channels[1] << "," << channels[2] << "}";

				}
			}


			//We finally fill the resulting image in that pixel with the calculated values, and adjust the bounds to avoid a out-of-bounds error

			
			resultingImage.setPixel(i-1, j-1, qRgb(channels[0], channels[1], channels[2]));


		}

	}
	return resultingImage;


}
