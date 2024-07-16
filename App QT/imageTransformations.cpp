#include "imageTransformations.h"
#include "ObjectDetection.h"







std::vector<std::vector<QRgb>> ImageTransformations::getPixels(QImage image) {

	if (image.isNull()) {

		return std::vector<std::vector<QRgb>>();
	}

	std::vector<std::vector<QRgb>> rgbMatrix(image.height(), std::vector<QRgb>(image.width()));

	for (int y = 0; y < image.height(); y++) {
		for (int x = 0; x < image.width(); x++) {
			//We access the pixel at (x,y)
			QRgb pixelValue = image.pixel(x, y);

			//We store the pixel value in the matrix 
			rgbMatrix[y][x] = pixelValue;
		}
	}

	// Now rgbMatrix contains the RGB values of each pixel in the image
	// You can access the RGB values like this:
	// int red = qRed(rgbMatrix[y][x]);
	// int green = qGreen(rgbMatrix[y][x]);
	// int blue = qBlue(rgbMatrix[y][x]);

	return rgbMatrix;
}

QImage ImageTransformations::createImage(std::vector<std::vector<QRgb>> matrix) {

	if (matrix.empty() || matrix[0].empty()) {

		return QImage(); //We return an empty image
	}

	int height = matrix.size();
	int width = matrix[0].size();

	QImage image(width, height, QImage::Format_ARGB32);

	for (int y = 0; y < height; y++) {
		QRgb* rowData = reinterpret_cast<QRgb*>(image.scanLine(y));
		for (int x = 0; x < width; x++) {
			rowData[x] = matrix[y][x];
		}
	}
	return image;
}


QImage ImageTransformations::negativeImage(QImage image) {

	//We will first convert the image to grayscale
	image = convertToGray(image);

	//We then obtain the matrix of pixels
	std::vector<std::vector<QRgb>> matrix = getPixels(image);

	//Now we will compute the negative

	for (int i = 0; i < matrix.size(); i++) {
		for (int j = 0; j < matrix[i].size(); j++) {
			matrix[i][j] = qRgb(255 - matrix[i][j], 255 - matrix[i][j], 255 - matrix[i][j]);
		}
	}

	//We create the image
	QImage result = createImage(matrix);

	//We attempt to save the image

	if (result.isNull()) {
		return result;
	}

	if (!result.save(image.text() + "negative.png")) {
		return result;
	}

	return result;



}

QImage ImageTransformations::logTransform(QImage image, double c)
{
	//We will first convert the image to grayscale
	image = convertToGray(image);

	//We then obtain the matrix of pixels
	std::vector<std::vector<QRgb>> matrix = getPixels(image);

	//we will now iterate the image and apply a log transform to each point

	for (int i = 0; i < matrix.size(); i++) {
		for (int j = 0; j < matrix[i].size(); j++) {
			int intensity = qGray(matrix[i][j]);
			int tranformedVal = c * log(1 + intensity);
			tranformedVal = qBound(0, tranformedVal, 255);
			matrix[i][j] = qRgb(tranformedVal, tranformedVal, tranformedVal);
		}
	}
	//We create the image
	QImage result = createImage(matrix);

	//We attempt to save the image

	if (result.isNull()) {
		return result;
	}

	if (!result.save(image.text() + "log.png")) {
		return result;
	}

	return result;


}

QImage ImageTransformations::gammaTransform(QImage image, double c, double alpha) {

	image = convertToGray(image);

	std::vector<std::vector<QRgb>> matrix = getPixels(image);

	for (int i = 0; i < matrix.size(); i++) {
		for (int j = 0; j < matrix[i].size(); j++) {
			int intensity = qGray(matrix[i][j]);
			int transform = c * (pow(intensity, alpha));
			transform = qBound(0, transform, 255);
			matrix[i][j] = qRgb(transform, transform, transform);
		}
	}
	QImage res = createImage(matrix);
	return res;

}

QImage ImageTransformations::normalizeImage(const QImage& image)
{
	QImage normalizedImage = image.convertToFormat(QImage::Format_Grayscale8);
	uchar minVal =255;
	uchar maxVal = 0;

	for (int y = 0; y < image.height(); ++y) {
		for (int x = 0; x < image.width(); ++x) {
			uchar pixel = qGray(image.pixel(x, y));
			if (pixel < minVal) minVal = pixel;
			if (pixel > minVal) maxVal = pixel;

		}
	}
	if (maxVal == minVal) {
		normalizedImage.fill(255);
	}else {
		for (int y = 0; y < image.height(); ++y) {
			for (int x = 0; x < image.width(); ++x) {
				uchar pixel = qGray(image.pixel(x, y));
				uchar normPixel = 255 * (pixel - minVal);
				normalizedImage.setPixel(x, y, qRgb(normPixel, normPixel, normPixel));
			}
		}
	}
	return normalizedImage;
}

//Function to compute the Discrete Fast Fourier Transform
QImage ImageTransformations::DiscreteFFT(QImage image){

	int width = image.width();
	int height = image.height();

	std::vector<std::vector<std::complex<double>>> data(height, std::vector <std::complex<double >> (width));

	//We fill the data with grayscale values

	for (int y = 0; y < height; ++y) {
		for (int x = 0; x < width; ++x) {
			QRgb pixel = image.pixel(x, y);
			double g = qGray(pixel);
			data[y][x] = std::complex<double>(g, 0.0);
		}
	}

	auto fftResult = fft2D(data,false);

	//Create magnitude and phase images

	QImage magnitudeImage = QImage(width, height, QImage::Format_Grayscale8);
	QImage phaseImage= QImage(width, height, QImage::Format_Grayscale8);

	double maxMag = 0.0;

	for (int y = 0; y < height; ++y) {
		for (int x = 0; x < width; x++) {
			double real = fftResult[y][x].real();
			double imag = fftResult[y][x].imag();

			double magnitude = std::sqrt(real * real+ imag * imag);
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

QImage ImageTransformations::InverseFFT(QImage image)
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

	auto ifftRes = fft2D(data,true);

	QImage inverseImage(width, height, QImage::Format_Grayscale8);

	for (int y = 0; y < height; ++y) {
		for (int x = 0; x < width; ++x) {
			double value = qBound(0.0, value, 255.0);
			inverseImage.setPixel(x, y, qGray(value));
		}
	}

	return inverseImage;
}

QImage ImageTransformations::createGaborKernel(int ksize, double sigma, double theta, double lambda, double gamma, double psi)
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

std::vector<QImage> ImageTransformations::createGarborKernels(const std::vector<int>& orientationsPerScale, int imageSize)
{


	int Nscales = orientationsPerScale.size();
	int Nfilters = 0;
	for (int i = 0; i < Nscales; ++i) {
		Nfilters += orientationsPerScale[i];
	}

	std::vector<QImage> filters;
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
			param[3] = M_PI / orientationsPerScale[i]*j;

			QImage filter(imageSize, imageSize, QImage::Format_Grayscale8);

			for (int y = 0; y < imageSize; ++y) {
				for (int x = 0; x < imageSize; ++x) {
					int index = y * imageSize + x;
					double tr = t[index] + param[3];
					tr = fmod(tr + M_PI, 2 * M_PI);

					double value = exp(-10 * param[0] * (fr[index] / (imageSize / 2) / param[1] - 1) * (fr[index] / (imageSize / 2) / param[1] - 1) - 2 * param[2] * M_PI * tr * tr);
					int pixelValue = static_cast<int>(value * 255);
					filter.setPixel(x, y, qRgb(pixelValue, pixelValue, pixelValue));
				}
			}
			filters.push_back(filter);
			++l;

		}
	}

	return filters;


}


QImage ImageTransformations::gaborFilter(const QImage& image, const QImage& kernel)
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
					int kernelVal = qGray(kernel.pixel(kx + halfSize,ky+ halfSize));
					sum += pixelVal * kernelVal;
				}
			}
			filteredImage.setPixel(j, i, qBound(0, static_cast<int>(sum), 255));
		}
	}
	return filteredImage;
}
//Function used to enhance local contrast and remove low-frequency components that might not be useful for capturing texture information
QImage ImageTransformations::preFilter(const QImage& image, double fc)
{
	int w = 5;
	double s1 = fc / sqrt(log(2.0));
	int width = image.width();
	int height = image.height();

	//Convert the image to grayscale and logscale

	QImage grayImage = image.convertToFormat(QImage::Format_Grayscale8);
	QVector<double> imageData(width * height);

	for (int y = 0; y < height; ++y) {
		for (int x = 0; x < width; ++x) {
			imageData[static_cast<qsizetype>(y) * width + x] = log(1.0 + qGray(grayImage.pixel(x, y)));

		}
	}

	//We pad the images to reduce the effects of boundary artifacts

	int padWidth = width + 2 * w;
	int padHeight = height + 2 * w;

	QVector<double> paddedImage(padWidth * padHeight, 0);
	for (int y = 0; y < height; ++y) {
		for (int x = 0; x < width; ++x) {
			paddedImage[(static_cast<qsizetype>(y) + w) * padWidth + (x + w)] = imageData[y * width + x];
		}
	}

	//Create gaussian filter in the frequency domain

	std::vector<std::vector<std::complex<double>>> filter(padHeight, std::vector<std::complex<double>>(padWidth));
	for (int y = 0; y < height; ++y) {
		for (int x = 0; x < width; ++x) {
			int fx = x - padWidth / 2;
			int fy = y - padHeight / 2;
			double fr = sqrt(fx * fx + fy * fy);
			double value = exp(-(fr * fr) / (s1 * s1));
			filter[y][x] = std::complex<double>(value, 0.0);
		}
	}



	//Preform the FFT on the padded image 

	QImage paddedImg(padWidth, padHeight, QImage::Format_Grayscale8);
	for (int y = 0; y < padHeight; ++y) {
		for (int x = 0; x < padWidth; ++x) {
			int index = y * padWidth + x;
			paddedImg.setPixel(x, y, qGray(paddedImage[index]));
		 }
	}

	auto fftImage = DiscreteFFT(paddedImg);
	auto fftResult = fft2D(filter,false);

	//Apply the gaussian filter in the frequency domain

	for (int y = 0; y < height; ++y) {
		for (int x = 0; x < width; ++x) {
			QRgb pixel = fftImage.pixel(x, y);
			double g = qGray(pixel);
			std::complex<double> fftPixel(g, 0.0);
			fftPixel *= fftResult[y][x];
			int value = static_cast<int>(fftPixel.real());
			fftImage.setPixel(x, y, qRgb(value, value, value));
		}
	}

	//Preforming inverse FFT

	auto inverseFFTImage = InverseFFT(fftImage);

	//Normalize the result 

	QVector<double> filteredImage(padWidth * padHeight);
	for (int y = 0; y < height; ++y) {
		for (int x = 0; x < width; ++x) {
			int index = y * padWidth + x;
			QRgb pixel = inverseFFTImage.pixel(x, y);
			filteredImage[index] = qGray(pixel);
		}
	}

	//Local contrast Normalization

	QVector<double> localstd(padWidth * padHeight);
	for (int y = 0; y < height; ++y) {
		for (int x = 0; x < width; ++x) {
			int idx = y * padWidth + x;
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
	QImage output(width, height, QImage::Format_Grayscale8);
	for (int y = 0; y < height; ++y) {
		for (int x = 0; x < width; ++x) {
			double value = filteredImage[(y + w) * padWidth + (x + w)];
			value = qBound(0.0, value, 255.0);
			output.setPixel(x, y, qGray(value));
		}
	}


	return output;


}

QImage ImageTransformations::downN(QImage& image, int N)
{
	int rows = image.height();
	int cols = image.width();
	int channels = image.isGrayscale() ? 1 : 3;

	int blockRows = rows / N;
	int blockCols = cols / N;

	QImage downSampledImage(N, N, image.format());
	
	for (int i = 0; i < N; ++i) {
		for (int j = 0; j < N; ++j) {
			std::vector<int> block(channels, 0);
			for (int k = 0; k < blockRows; ++k) {
				for (int l = 0; l < blockCols; l++) {
					QRgb pixel = image.pixel(j * blockCols, i * blockRows + k);
					if (channels == 1) {
						block[0] += qGray(pixel);
					}else {
						block[0] += qRed(pixel);
						block[1] += qGreen(pixel);
						block[2] += qBlue(pixel);

					}
				}
			}
			for (int c = 0; c < channels; ++c) {
				block[c] /= (blockRows * blockCols);
			}
			if (channels == 1) {
				downSampledImage.setPixel(j, i, qRgb(block[0], block[0], block[0]));

			}else {
				downSampledImage.setPixel(j, i, qRgb(block[0], block[1], block[2]));
			}
		}
	}

	return downSampledImage;






}

QImage ImageTransformations::padArray(QImage image, int padSize){
	QImage paddedImage(image.width() + 2 * padSize, image.height() + 2 * padSize, image.format());
	paddedImage.fill(Qt::black);

	for (int y = 0; y < image.height(); ++y) {
		for (int x = 0; x < image.width(); ++x) {
			paddedImage.setPixel(x + padSize, y + padSize, image.pixel(x, y));
		}
	}

	return paddedImage;
}



std::vector<double> ImageTransformations::normalizeHistogram(std::vector<int> histogram) {

	int count = 0; // Total Number of pixels

	for (int i = 0; i < histogram.size(); i++) {
		count += histogram[i];
	}

	std::vector<double> normalizedHist(256, 0);

	for (int i = 0; i < 256; i++) {
		normalizedHist[i] = static_cast<double>(histogram[i]) / count;
	}


	return normalizedHist;
}

std::vector<int> ImageTransformations::linearHistogramEqualization(QImage image, int maxVal)
{
	//Max val parameter will specify the range of the equalization it will be from [0-maxVal]
	std::vector<int> hist = computeHistogram(image);

	std::vector<double> normalizedHistogram = normalizeHistogram(hist);

	//we will now apply transformations to the normalized histogram


	std::vector<int> res(256, 0);

	for (int i = 0; i < 256; i++) {
		double sum = 0;
		for (int j = 0; j < i; j++) {
			sum += normalizedHistogram[j];
		}
		res[i] = maxVal * sum;


	}
	return res;

}



std::vector<int> ImageTransformations::computeHistogram(QImage image)
{
	//We fill the histogram
	image = ImageTransformations::convertToGray(image);
	std::vector<std::vector<QRgb>> matrix = ImageTransformations::getPixels(image);

	std::vector<int> histogram(256, 0);

	for (int i = 0; i < matrix.size(); i++) {
		for (int j = 0; j < matrix[i].size(); j++) {
			histogram[qGray(matrix[i][j])]++;
		}
	}

	return histogram;

}

QImage ImageTransformations::histogramToImage(std::vector<int> transformedHist, QImage originalImage)
{


	//we Remap the values

	std::vector<std::vector<QRgb>> matrix = getPixels(originalImage);

	for (int i = 0; i < matrix.size(); i++) {
		for (int j = 0; j < matrix[i].size(); j++) {
			int pixelVal = transformedHist[qGray(matrix[i][j])];

			matrix[i][j] = qRgb(pixelVal, pixelVal, pixelVal);
		}
	}
	QImage res = createImage(matrix);
	return res;



}




bool pointExistsInVector(const QVector<QVector<QPoint>>& vector, int i, int j) {
	for (const QVector<QPoint>& pintsVector : vector) {
		for (const QPoint& point : pintsVector) {
			if (point.x() == i && point.y() == j) {
				return true; // El punto se encontró en uno de los contornos
			}
		}
	}
	return false; // El punto no se encontró
}








std::vector<QImage> ImageTransformations::calculatezSubImage(QImage& image)
{
	QVector<QVector<QPoint>> objects = ObjectDetection::connectedN4(image);
	int xMax, xMin, yMax, yMin;


	std::vector<QImage> subImages;
	for (QVector<QPoint> object : objects) {
		ObjectDetection::calculateBounds(object, xMin, xMax, yMin, yMax);
		QImage objectImage = image.copy(xMin - 5, yMin - 5, xMax - xMin + 10, yMax - yMin + 10);
		subImages.push_back(objectImage);

	}






	return subImages;
}


std::vector<std::vector<double>> ImageTransformations::computeGistDescriptor(std::vector<QImage> images, std::vector<std::string> imageClass)
{

	int boundaryExtension = 32; // Number of pixels to pad the image
	int imageSize = 256;
	std::vector<int> orientationsPerScale = { 8,8,8,8 };
	int numberBlock = 4;
	int fc_prefilt = 4;
	auto G = Kernels::createGarborKernels(orientationsPerScale, imageSize + (2 * boundaryExtension));
	int Nfeatures = G[0][0].size() * numberBlock * numberBlock;
	int Nscenes = images.size();

	std::vector<std::vector<double>> gist(Nscenes, std::vector<double>(Nfeatures, 0.0f));

	std::ofstream outFile("results.csv", std::ios::app);

	if (!outFile.is_open()) {
		std::cerr << "No se pudo abrir el archivo para escritura." << std::endl;
		
	}

	for (int i = 0; i < Nscenes; i++) {
		std::vector<double> currentGist;



		if (Nscenes > 1) {
			qDebug() << i << " , " << Nscenes;
		}
		images[i] = ColorSpaces::convertToGray(images[i]);
		

		QSize newImageSize(imageSize, imageSize);
		images[i] = images[i].scaled(newImageSize, Qt::IgnoreAspectRatio, Qt::SmoothTransformation);

		


		auto normalizedImage = normalizeImage(images[i]);


		auto output = SpatialOperations::preFilter(normalizedImage);

		

		
		
		
		
		currentGist = gistGabor(output, numberBlock, G, boundaryExtension);



		std::string res = "";

		for (auto el : currentGist) {
			std::string str = std::to_string(el);
			res.append(str);
			res.append(" ,");
		}
		res.append(imageClass[i]);
		

		outFile << res<< std::endl;


		//qDebug() << "Metrics: " << res;
		gist.push_back(currentGist);



		




	}
	return gist;



}

std::vector<double> ImageTransformations::gistGabor(std::vector<std::vector<double>> &image, int w, std::vector<std::vector<std::vector<double>>> g, int boundaryExtension)
{

	



	int rows = image.size();
	int cols = image[0].size();





	int c = 1;
	int N = c;

	int ny = g[0].size();
	int nx = g[0][0].size();





	int Nfilters = g.size();

	int W = w * w;


	std::vector<double> gistG(W * Nfilters * N, 0.0f);

	//Pad image

	std::vector<std::vector<double>> paddedImage = SpatialOperations::padArray(image, boundaryExtension);

	//Convert image to 2d complex array for fft

	std::vector < std::vector < std::complex<double>>> imageComplex(paddedImage.size(), std::vector<std::complex<double>>(paddedImage[0].size()));

	for (int y = 0; y < paddedImage.size(); ++y) {
		for (int x = 0; x < paddedImage[0].size(); ++x) {
			imageComplex[y][x] = std::complex<double>(paddedImage[y][x],0.0);

		}
	}






	//Perform FFT 
	auto imageFft = Computations::Math::fft2D(imageComplex, false);

	int k = 0;
	for (int n = 0; n < Nfilters; ++n) {
		std::vector<std::vector<std::complex<double>>> gFilter(ny, std::vector<std::complex<double>>(nx));
		for (int y = 0; y < ny; ++y) {
			for (int x = 0; x < nx; ++x) {
				gFilter[y][x] = std::complex<double>(g[n][x][y], 0.0); // Convert QImage pixel to grayscale complex
			}
		}

		//Apply gabor filter

		std::vector<std::vector<std::complex<double>>> filteredImage(paddedImage[0].size(), std::vector<std::complex<double>>(paddedImage.size()));
		for (int y = 0; y < paddedImage.size(); ++y) {
			for (int x = 0; x < paddedImage[0].size(); ++x) {
				filteredImage[y][x] = imageFft[y][x] * gFilter[y % ny][x % nx];
			}
		}


		//Preform IFFT

		std::vector<std::vector<std::complex<float>>> imageFFt;
		auto imageIFFT = Computations::Math::fft2D(filteredImage, true);





		std::vector<std::vector<double>> ig(ny - 2 * boundaryExtension, std::vector<double>(nx - 2 * boundaryExtension));
		for (int y = boundaryExtension; y < ny - boundaryExtension; ++y) {
			for (int x = boundaryExtension; x < nx - boundaryExtension; ++x) {
				float magnitude = abs(imageIFFT[y][x]);
				ig[y - boundaryExtension][x - boundaryExtension] = magnitude;
			}
		}
		//for (auto row : ig) {
		//	std::string res = "";
		//	for (auto el : row) {
		//		res += std::to_string(el);
		//		res += " ";
		//	}
		//	qDebug() << res;
		//}

		//Downsample the filtered image
		std::vector<std::vector<double>> v = SpatialOperations::downN(ig, w);




		//Store the downsampled values in the descriptor
		for (int i = 0; i < v.size(); ++i) {
			for (int j = 0; j < v[0].size(); ++j) {
				gistG[k++] = v[j][i];
				
			}
		}





	}
	if (c == 3) {
		std::vector<double> gReshaped(g.size() * 3 / c);
		int idx = 0;
		for (size_t i = 0; i < gistG.size(); i += 3) {
			gReshaped[idx++] = gistG[i];
			gReshaped[idx++] = gistG[i + 1];
			gReshaped[idx++] = gistG[i + 2];
		}
		return gReshaped;
	}

	return gistG;


}

cv::Mat ImageTransformations::QImageToMat(const QImage& inputImage, bool inCloneImageData)
{
	switch (inputImage.format()) {
	case QImage::Format_ARGB32:
	case QImage::Format_ARGB32_Premultiplied: {
		cv::Mat mat(inputImage.height(), inputImage.width(), CV_8UC4,
			const_cast<uchar*>(inputImage.bits()), inputImage.bytesPerLine());
		return (inCloneImageData ? mat.clone() : mat);
	}
	case QImage::Format_RGB32: {
		if (!inCloneImageData) {
			qDebug() << "Conversion requires cloning because we use a temporary QImage";
		}
		QImage swapped = inputImage.rgbSwapped();
		return cv::Mat(swapped.height(), swapped.width(), CV_8UC3,
			const_cast<uchar*>(swapped.bits()), swapped.bytesPerLine()).clone();
	}
	case QImage::Format_RGB888: {
		// This format does not need swapping as it matches OpenCV's default channel sequence (BGR)
		cv::Mat mat(inputImage.height(), inputImage.width(), CV_8UC3,
			const_cast<uchar*>(inputImage.bits()), inputImage.bytesPerLine());
		return (inCloneImageData ? mat.clone() : mat);
	}
	case QImage::Format_Indexed8: {
		cv::Mat mat(inputImage.height(), inputImage.width(), CV_8UC1,
			const_cast<uchar*>(inputImage.bits()), inputImage.bytesPerLine());
		return (inCloneImageData ? mat.clone() : mat);
	}
	default:
		qDebug() << "QImage format not handled in switch:" << inputImage.format();
		return cv::Mat();
	}

}


>>>>>>> 7aec56e2c974d9237c6bde378b1d486b4bfd8c68
