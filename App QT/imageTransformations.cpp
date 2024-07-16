#include "imageTransformations.h"

QImage ImageTransformations::convertToGray(QImage& image) {
	//if (image.isNull()) {
	//	return image;
	//}
	////We obtain the image matrix
	//std::vector<std::vector<QRgb>> matrix = getPixels(image);
	//int height = matrix.size();
	//int width = matrix[0].size();
	////We then convert the image to gray



	//for (int y = 0; y < height; y++) {
	//	for (int x = 0; x < width; x++) {

	//		QRgb currentPixel = matrix[y][x];
	//		int value = qRed(currentPixel) * 0.299 + qGreen(currentPixel) * 0.587 + qBlue(currentPixel) * 0.114;
	//		matrix[y][x] = qRgb(value, value, value);

	//	}
	//}


	//image = createImage(matrix);
	//return image;
	QImage newImage = image.convertToFormat(QImage::Format_Grayscale8);
	return newImage;

}

void ImageTransformations::imageObjectsToCsv(QImage& image, QString filaname, int i, std::vector<QImage>& subimages)
{
	qDebug() << "ENTRANDO en " << i << '\n';
	// calling the function that retrieves the information 
	QVector<QVector<QPoint>> objects = ImageTransformations::connectedN4(image);
	std::ofstream outFile("FilesOut/objects.csv", std::ios::app);

	if (!outFile.is_open()) {
		std::cerr << "No se pudo abrir el archivo para escritura." << std::endl;
		return;
	}

	for (int i = 0; i < objects.size(); i++) {


		// getting the subimage of the object
		QVector<QPoint> pointsS = objects[i];
		int minX = pointsS[0].x();
		int minY = pointsS[0].y();
		int maxX = pointsS[0].x();
		int maxY = pointsS[0].y();

		// Find the bounding box of the object
		for (const QPoint& point : pointsS) {
			if (point.x() < minX) {
				minX = point.x();
			}
			if (point.x() > maxX) {
				maxX = point.x();
			}
			if (point.y() < minY) {
				minY = point.y();
			}
			if (point.y() > maxY) {
				maxY = point.y();
			}
		}

		// Create a new image containing only the object
		QImage objectImage = image.copy(minX - 5, minY - 5, maxX - minX + 10, maxY - minY + 10);
		
		QImage objectImageBinary = thereshold(objectImage, 130);
		subimages.push_back(objectImageBinary);
		double e = ObjectMetrics::calculateEccentricity((objectImageBinary));
		QImage ImageBinary = thereshold(image, 130);
		qDebug() << "e at " << filaname << " = " << e;

		

		std::vector < std::function<std::vector<int>(QVector<QPoint>, QImage&)>> func = {
			//ObjectMetrics::calculateArea,
			//ObjectMetrics::calculateCenterOfGravity,
			ObjectMetrics::calculatePerimeter
		};


		for (int i = 0; i < objects.size(); i++) {//we iterate the objects
			Eigen::MatrixXd descritorsReturn = ObjectMetrics::featureExtraction(func, objects[i], ImageBinary);

			for (int j = 0; j < descritorsReturn.cols(); j++) { //we iterate the features of the objects 
				outFile << descritorsReturn(0, j) << ",";
			}

			outFile << "object" << std::to_string(i + 1) << "At " << filaname.toStdString() << std::endl;

		}
		outFile.close();
		qDebug() << "EXITO" << '\n';
	}
}


QImage ImageTransformations::thereshold(QImage& image, int threshold) {
	if (image.isNull()) {
		return image;
	}
	//We pass to gray color
	QImage gray = convertToGray(image);
	QImage thresholdedImage(gray.size(), QImage::Format_Grayscale8);


	////We obtain the image matrix
	//std::vector<std::vector<QRgb>> matrix = getPixels(image);
	//int height = matrix.size();
	//int width = matrix[0].size();
	//for (int y = 0; y < height; y++) {
	//	for (int x = 0; x < width; x++) {
	//		QRgb currentPixel = matrix[y][x];
	//		if (qRed(currentPixel) >= threshold) {
	//			matrix[y][x] = qRgb(255, 255, 255);
	//		}
	//		else {
	//			matrix[y][x] = qRgb(0, 0, 0);
	//		}

	//	}
	//}
	//image = createImage(matrix);



	for (int y = 0; y < gray.height(); ++y) {
		const uchar* srcRow = gray.constScanLine(y);
		uchar* dstRow = thresholdedImage.scanLine(y);
		for (int x = 0; x < gray.width(); ++x) {
			dstRow[x] = srcRow[x] > threshold ? 255 : 0;
		}
	}

	return thresholdedImage;


}




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

//Histogram Specification functions
//Histogram Specification: Process of modyfing a histogram to achieve a certain histogram shape

//We compute the discrete pdf (Probabiltiy Distribution Function)
std::vector<int> ImageTransformations::equalizationHistogram(std::vector<int> histogram, int maxVal)
{
	std::vector < double > normalized = normalizeHistogram(histogram);
	std::vector <int> res(256, 0);
	double sum = 0;
	for (int i = 0; i < 256; i++) {
		sum += normalized[i];
		res[i] = maxVal * sum;



	}



	return res;
}

std::vector<int> ImageTransformations::histogramSpecification(std::vector<int> histogram, std::vector<double> specifiedHistogram, int maxVal)
{

	std::vector<int> g(256, 0);
	double sum = 0;
	for (int i = 0; i < 256; i++) {
		sum += specifiedHistogram[i];
		g[i] = sum * maxVal;


	}

	std::vector<int> eqHist = equalizationHistogram(histogram, maxVal);

	std::vector<int> res(256, 0);

	for (int i = 0; i < 256; i++) {

	}
	return eqHist;


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




/*

QPoint ImageTransformations::nextCell(QPoint& currPixel, int &currDir, QPoint &savePixel) {
	int i = currPixel.y();
	int j = currPixel.x();
	int r, c;
	savePixel = QPoint();

	switch (currDir) {
	case 0:
		r = i - 1;
		c = j;
		currDir = 1;
		savePixel = QPoint(j + 1, i);
		break;
	case 1:
		r = i;
		c = j - 1;
		currDir = 2;
		break;
	case 2:
		r = i + 1;
		c = j;
		currDir = 3;
		break;
	case 3:
		r = i;
		c = j + 1;
		currDir = 0;
		break;
	}
	return QPoint(c,r);
}


QVector<QPoint> ImageTransformations::borderFollow(QImage & img, QPoint& start, QPoint& prev, int direction, int& NBD)
{
	ImageTransformations transformer;
	QPoint curr = start;
	QPoint exam = prev;
	QPoint save;
	QPoint save2 = exam;
	QVector<QPoint> contour;
	contour.push_back(curr);

	// Infinite loop fix: Ensure exam updates correctly
	while (img.pixelColor(exam).value() == 0) {
		exam = nextCell(curr, direction, save);
		if (exam == save2) {
			img.setPixelColor(curr, QColor(-NBD, 0, 0));
			return contour;
		}
	}

	if (!save.isNull()) {
		img.setPixelColor(curr, QColor(-NBD, 0, 0));
	}
	else if ((save.isNull() || img.pixelColor(save).value() != 0) && img.pixelColor(curr).value() == 1) {
		img.setPixelColor(curr, QColor(NBD, 0, 0));
	}

	QPoint prevPt = curr;
	curr = exam;
	contour.push_back(curr);
	direction = (direction >= 2) ? (direction - 2) : (2 + direction);
	int flag = 0;
	QPoint startNext = curr;

	while (true) {
		if (!(curr == startNext && prevPt == start && flag == 1)) {
			flag = 1;
			exam = nextCell(curr, direction, save);

			// Infinite loop fix: Ensure exam updates correctly
			while (img.pixelColor(exam).value() == 0) {
				exam = nextCell(curr, direction, save);
				if (exam == save2) {
					img.setPixelColor(curr, QColor(-NBD, 0, 0));
					return contour;
				}
			}

			if (!save.isNull() && img.pixelColor(save).value() == 0) {
				img.setPixelColor(curr, QColor(-NBD, 0, 0));
			}
			else if ((save.isNull() || img.pixelColor(save).value() != 0) && img.pixelColor(curr).value() == 1) {
				img.setPixelColor(curr, QColor(NBD, 0, 0));
			}

			prevPt = curr;
			curr = exam;
			contour.push_back(curr);
			direction = (direction >= 2) ? (direction - 2) : (2 + direction);
		}
		else {
			break;
		}
	}

	return contour;
}





QVector<QVector<QPoint>> ImageTransformations::rasterScan(QImage& img) {
	int rows = img.height();
	int cols = img.width();
	int LNBD = 1;
	int NBD = 1;
	QVector<QVector<QPoint>> contours;
	QVector<QPoint> addedPoints;

	for (int i = 1; i < rows - 1; ++i) {
		LNBD = 1;
		for (int j = 1; j < cols - 1; ++j) {
			QPoint curr(j, i);
			QPoint prev(j - 1, i);
			QPoint next(j + 1, i);
			if (addedPoints.contains(curr)) {
				continue;
			}
			if (img.pixelColor(j, i).value() == 255 && img.pixelColor(j - 1, i).value() == 0) {
				NBD += 1;
				int direction = 2;
				QVector<QPoint> contour = borderFollow(img, curr, prev, direction, NBD);
				contours.push_back(contour);
				for (auto point : contour) {
					addedPoints.push_back(point);
				}
			}
			else if (img.pixelColor(j, i).value() >= 1 && img.pixelColor(j + 1, i).value() == 0) {
				NBD += 1;
				int direction = 0;
				if (img.pixelColor(j, i).value() > 1) {
					LNBD = img.pixelColor(j, i).value();
				}
				QVector<QPoint> contour = borderFollow(img, curr, next, direction, NBD);
				contours.push_back(contour);
				for (auto point : contour) {
					addedPoints.push_back(point);
				}
				}
			}
		}

	return contours;
}









































































QVector<QVector<QPoint>> ImageTransformations::connectedN4(QImage& image) {


	QImage binaryImage = thereshold(image, 130);


	return rasterScan(binaryImage);

	//QVector<QVector<QPoint>> contours;
	//QImage visited = QImage(binaryImage.size(), QImage::Format_ARGB32);
	//visited.fill(QColor(0, 0, 0, 0));

	//for (int y = 0; y < binaryImage.height(); ++y) {
	//	for (int x = 0; x < binaryImage.width(); ++x) {
	//		if (binaryImage.pixelColor(x, y).value() == 255 && visited.pixelColor(x, y).value() == 0) { // Unvisited white pixel
	//			int initialDir = 0;
	//			QList<QPoint> contour = followContour(binaryImage, QPoint(x, y), initialDir, visited);
	//			if (contour.size() > 50) { // Only consider contours with more than 50 points
	//				contours.append(contour);
	//			}
	//		}
	//	}
	//}

	//return contours;






*/


#include <QImage>
#include <QPoint>
#include <QColor>
#include <QVector>
#include <tuple>
#include <cmath>

// Assuming the existence of the borderFollow and nextCell functions as previously translated

QPoint ImageTransformations::nextCell(QPoint& currPixel, int& currDir, QPoint& savePixel) {
	int i = currPixel.y();
	int j = currPixel.x();
	int r, c;
	savePixel = QPoint();

	switch (currDir) {
	case 0:
		r = i - 1;
		c = j;
		currDir = 1;
		savePixel = QPoint(j + 1, i);
		break;
	case 1:
		r = i;
		c = j - 1;
		currDir = 2;
		break;
	case 2:
		r = i + 1;
		c = j;
		currDir = 3;
		break;
	case 3:
		r = i;
		c = j + 1;
		currDir = 0;
		break;
	}
	return QPoint(c, r);
}

QVector<QPoint> ImageTransformations::borderFollow(QImage& img, QPoint& start, QPoint& prev, int direction, int& NBD) {
	QPoint curr = start;
	QPoint exam = prev;
	QPoint save;
	QPoint save2 = exam;
	QVector<QPoint> contour;
	contour.push_back(curr);

	while (img.pixelColor(exam).value() == 0) {
		exam = nextCell(curr, direction, save);
		if (exam == save2) {
			img.setPixelColor(curr, QColor(-NBD, 0, 0));
			return contour;
		}
	}

	if (!save.isNull()) {
		img.setPixelColor(curr, QColor(-NBD, 0, 0));
	}
	else if ((save.isNull() || img.pixelColor(save).value() != 0) && img.pixelColor(curr).value() == 1) {
		img.setPixelColor(curr, QColor(NBD, 0, 0));
	}

	QPoint prevPt = curr;
	curr = exam;
	contour.push_back(curr);
	direction = (direction >= 2) ? (direction - 2) : (2 + direction);
	int flag = 0;
	QPoint startNext = curr;

	while (true) {
		if (!(curr == startNext && prevPt == start && flag == 1)) {
			flag = 1;
			exam = nextCell(curr, direction, save);

			while (img.pixelColor(exam).value() == 0) {
				exam = nextCell(curr, direction, save);
				if (exam == save2) {
					img.setPixelColor(curr, QColor(-NBD, 0, 0));
					return contour;
				}
			}

			if (!save.isNull() && img.pixelColor(save).value() == 0) {
				img.setPixelColor(curr, QColor(-NBD, 0, 0));
			}
			else if ((save.isNull() || img.pixelColor(save).value() != 0) && img.pixelColor(curr).value() == 1) {
				img.setPixelColor(curr, QColor(NBD, 0, 0));
			}

			prevPt = curr;
			curr = exam;
			contour.push_back(curr);
			direction = (direction >= 2) ? (direction - 2) : (2 + direction);
		}
		else {
			break;
		}
	}

	return contour;
}

QVector<QVector<QPoint>> ImageTransformations::rasterScan(QImage& img) {
	int rows = img.height();
	int cols = img.width();
	int LNBD = 1;
	int NBD = 1;
	QVector<QVector<QPoint>> contours;
	QVector<QPoint> addedPoints;

	for (int i = 1; i < rows - 1; ++i) {
		LNBD = 1;
		for (int j = 1; j < cols - 1; ++j) {
			QPoint curr(j, i);
			QPoint prev(j - 1, i);

			if (addedPoints.contains(curr)) {
				continue;
			}

			// Detect only outer contours
			if (img.pixelColor(j, i).value() == 255 && img.pixelColor(j - 1, i).value() == 0) {
				NBD += 1;
				int direction = 2;
				QVector<QPoint> contour = borderFollow(img, curr, prev, direction, NBD);
				contours.push_back(contour);
				for (const auto& point : contour) {
					addedPoints.push_back(point);
				}
			}
		}
	}

	return contours;
}

QVector<QVector<QPoint>> ImageTransformations::connectedN4(QImage& image) {
	QImage binaryImage = thereshold(image, 130);
	QVector<QVector<QPoint>> objects = rasterScan(binaryImage);
	for (int i = 0; i < objects.size(); i++) {
		if (objects[i].size() < 100) {
			objects.remove(i);
			i = 0;

		}
	}
	return objects;
}

































/*

std::vector<std::vector<QRgb>> matrix = getPixels(image);


//x-axis count
int i = 0;
//y-axis count
int j = 0;
//aux count
int count = 0;

int height = matrix.size();
int width = matrix[0].size();


QVector<QVector<QPoint>> objects;








































while (i < matrix.size()) {
	int jPrevious = 0;
	while (j < matrix[i].size()) {
		if (qRed(matrix[i][j]) > 250 && !pointExistsInVector(objects, i, j) && count == 0) {
			objects.append(outLine(image, i, j));
			i = 0;
			j = 0;

		}
		else if (qRed(matrix[i][j]) > 250 && !pointExistsInVector(objects, i, j) && count != 0) {
			i += 1;
			continue;
		}
		else if (qRed(matrix[i][j]) > 250 && pointExistsInVector(objects, i, j) && count == 0) {
			count += 1;
		}
		else if (qRed(matrix[i][j]) > 250 && pointExistsInVector(objects, i, j) && count != 0 && !pointExistsInVector(objects, i, jPrevious)) {
			count = 0;
		}
		else if (qRed(matrix[i][j]) == 0 && pointExistsInVector(objects, i, jPrevious) && count != 0) {
			count = 0;
		}
		jPrevious = j;
		j += 1;



	}
	i += 1;
	j = 0;
}
*/





/**
* @brief Function that find object online
* @param QImage: Image to segment
* @param int i: x-axis coordinate
* @param int j: y-axis coordinate
*
*/
QVector<QPoint> ImageTransformations::outLine(QImage& image, int i, int j) {
	//we define the objects vector
	QVector<QPoint> object;
	//we define variables that we'll use later
	int currI = i;
	int currJ = j;
	int d = 0;
	std::vector<std::vector<QRgb>> matrix = getPixels(image);
	while (true) {
		int q = (d + 3) % 4;
		bool breakFlag = false; // for brak in principal loop

		for (int p = 0; p < 4; ++p) {
			int dir = (q + p) % 4;
			int nextI = currI;
			int nextJ = currJ;

			switch (dir) {
			case 0: // up
				nextI--;
				break;
			case 1: // right
				nextJ++;
				break;
			case 2: // down
				nextI++;
				break;
			case 3: // left
				nextJ--;
				break;
			}

			// We check the pixel value
			if (nextI >= 0 && nextI < image.height() && nextJ >= 0 && nextJ < image.width() &&
				qRed(matrix[nextI][nextJ]) == 255) {
				currI = nextI;
				currJ = nextJ;
				d = dir;

				object.append(QPoint(currI, currJ));
				breakFlag = true;
				break;
			}
		}

		// stop condition
		if (!breakFlag || (currI == i && currJ == j)) {
			break;
		}
	}
	return object;
}

	
std::vector<std::string> ImageTransformations::classifyImage(QImage& image, Eigen::MatrixXd centroids, std::vector < std::function <std::vector<int>(QVector<QPoint>, QImage&)>> functions, std::map<int, std::string> namesMap) {

	//first we will apply all the transformations that were applied to images in the dataset, in order to get the metrics
	qDebug() << "Centroids";
	for (int i = 0; i < centroids.rows(); i++)
	{
		for (int j = 0; j < centroids.cols(); j++)
		{
			qDebug() << centroids(i, j) << "";
		}
		qDebug() << "\n";
	}

	//normalizeColumn(centroids,0);
	//normalizeColumn(centroids, 1);
	//normalizeColumn(centroids, 2);
	QVector<QVector<QPoint>> objects;
	objects = connectedN4(image);

	//we iterate each of the objects and get they're features

	std::vector<Eigen::MatrixXd> objectMetrics;

	Eigen::MatrixXd objetitos(objects.size(),3);

	std::vector<int> classification; //Vector that stores the classification of each object
	bool flag = false;
	int i = 0;
	for (auto object : objects) {
		Eigen::MatrixXd pointToClassify = ObjectMetrics::featureExtraction(functions, object, image); // aki
		objectMetrics.push_back(pointToClassify);
		qDebug() << " objeto perimetro " << pointToClassify(0) << " objeto exce " << pointToClassify(1) << " objeto area " << pointToClassify(2);
		if (pointToClassify(2) > 1400) { flag = true; }
		//Once we got they're features we will calculate the distance of each object to the centroid given by Kmeans
		objetitos(i, 0) = pointToClassify(0,0);
		objetitos(i, 1) = pointToClassify(0,1);
		objetitos(i, 2) = pointToClassify(0,2);
		//objetitos(i, 3) = pointToClassify(0,3);
		i++;
		//std::vector<double> distances =euclidean(objectMetrics[objectMetrics.size() - 1], pointToClassify);
		std::vector<double> distances = euclidean(centroids, pointToClassify);
		qDebug() << "Distancias para ese objeto ";
		for (auto d : distances)
		{
			qDebug() << d;
		}
		classification.push_back(getClosest(distances));
	}
	
	//normalizeColumn(objetitos,0);
	//normalizeColumn(objetitos, 1);
	//normalizeColumn(objetitos, 2);
	qDebug() << "Informacion de los objetos de la imagen ";
	// Calculates euc distance--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
	Eigen::MatrixXd distances2(objetitos.rows(), centroids.rows()); // matrix to store distances. [diastance to c1, distance to c2, distance to c3] instance 1,[diastance to c1, distance to c2, distance to c3] instance 2 ... 
	for (int i = 0; i < centroids.rows(); i++)
	{
		Eigen::MatrixXd data_minus_centroid = objetitos.rowwise() - centroids.row(i);
		data_minus_centroid = data_minus_centroid.array().square();
		Eigen::VectorXd sum_squared_diff_sqrt = (data_minus_centroid.rowwise().sum()).array().sqrt();
		distances2.col(i) = sum_squared_diff_sqrt;
	}
	std::vector<int> indexes;
	for (int ind = 0; ind < distances2.rows(); ind++)
	{
		Eigen::MatrixXd::Index minIndex;
		distances2.row(ind).minCoeff(&minIndex);
		indexes.push_back(minIndex);
		qDebug() << "centoide mas cercano para el ejemplo " << ind << " Es : " << minIndex << "\n";
	}
	qDebug() << "Informacion de las distancias de los objetos ";
	for (int i = 0; i < distances2.rows(); i++)
	{
		for (int j = 0; j < distances2.cols(); j++)
		{
			qDebug() << distances2(i, j) << "";
		}
		qDebug() << "\n";
	}

	std::vector<std::string> classificationVec;
	for (int i = 0; i < indexes.size(); i++) {
		std::string className;
		if (namesMap.find(indexes[i]) == namesMap.end()) {
			className = "None";
		}
		else if (flag)
		{
			classificationVec.push_back("Non an object");
		} 
		else {

			className = namesMap[indexes[i]];
			classificationVec.push_back((namesMap[indexes[i]]));
		}
	}

	return classificationVec;

















}

std::vector<std::vector<double>> ImageTransformations::computeGistDescriptor(std::vector<QImage> images)
{

	int boundaryExtension = 32; // Number of pixels to pad the image
	int imageSize = 128;
	std::vector<int> orientationsPerScale = { 8,8,8,8 };
	int numberBlock = 4;
	int fc_prefilt = 4;
	auto G = createGarborKernels(orientationsPerScale, imageSize + 2 * boundaryExtension);
	int Nfeatures = G.size() * numberBlock * numberBlock;
	int Nscenes = images.size();

	std::vector<std::vector<double>> gist(Nscenes,std::vector<double>(Nfeatures, 0.0f));



	for (int i = 0; i < Nscenes; i++) {
		std::vector<double> currentGist;



		if (Nscenes > 1) {
			qDebug() << i << " , " << Nscenes;
		}
		images[i] = convertToGray(images[i]);
		QSize newImageSize(imageSize, imageSize);
		images[i] = images[i].scaled(newImageSize, Qt::IgnoreAspectRatio, Qt::SmoothTransformation);
		images[i] = normalizeImage(images[i]);

		auto output = preFilter(images[i]);
		currentGist = gistGabor(output,numberBlock,images,boundaryExtension);
		
		std::string res= "[ ";

		for (auto el : currentGist) {
			std::string str = std::to_string(el);
			res.append(str);
			res.append(" ,");
		}
		res.append(" ]");
		qDebug() << "Metrics: " << res;
		gist.push_back(currentGist);
		
	}	
	return gist;

}

std::vector<double> ImageTransformations::gistGabor(const QImage& image, int w, std::vector<QImage> g, int boundaryExtension)
{

	QImage imageSingle = image.convertToFormat(QImage::Format_Grayscale8);
	int rows = image.height();
	int cols = image.width();

	int c = 1;
	int N = c;

	int ny = g[0].height();
	int nx = g[0].width();

	int Nfilters = g.size();

	int W = w * w;


	std::vector<double> gistG(W * Nfilters * N, 0.0f);

	//Pad image

	QImage paddedImage = padArray(image, boundaryExtension);

	//Convert image to 2d complex array for fft

	std::vector < std::vector < std::complex<double>>> imageComplex(paddedImage.height(), std::vector<std::complex<double>>(paddedImage.width()));

	for (int y = 0; y < paddedImage.height(); ++y) {
		for (int x = 0; x < paddedImage.width(); ++x) {
			imageComplex[y][x] = static_cast<float>(qGray(paddedImage.pixel(x, y)));

		}
	}

	//Perform FFT 
	auto imageFft= fft2D(imageComplex, false);

	int k = 0; 
	for (int n = 0; n < Nfilters; ++n) {
		std::vector<std::vector<std::complex<double>>> gFilter(ny, std::vector<std::complex<double>>(nx));
		for (int y = 0; y < ny; ++y) {
			for (int x = 0; x < nx; ++x) {
				gFilter[y][x] = g[n].pixel(y,x);
			}
		}

		//Apply gabor filter

		std::vector<std::vector<std::complex<double>>> filteredImage(paddedImage.height(), std::vector<std::complex<double>>(paddedImage.width()));
		for (int y = 0; y < paddedImage.height(); ++y) {
			for (int x = 0; x < paddedImage.width(); ++x) {
				filteredImage[y][x] = imageFft[y][x] * gFilter[y % ny][x % nx];
			}
		}


		//Preform IFFT

		std::vector<std::vector<std::complex<float>>> imageFFt;
		auto imageIFFT = fft2D(filteredImage,true);
	
		QImage ig(ny - 2 * boundaryExtension, nx - 2 * boundaryExtension, image.format());
		for (int y = boundaryExtension; y < ny - boundaryExtension; ++y) {
			for (int x = boundaryExtension; x < nx - boundaryExtension; ++x) {
				float magnitude = abs(imageIFFT[y][x]);
				ig.setPixel(x - boundaryExtension, y - boundaryExtension, qRgb(magnitude, magnitude, magnitude));
			}
		}


		//Downsample the filtered image
		QImage v = downN(ig, w);

		//Store the downsampled values in the descriptor
		for (int i = 0; i < v.height(); ++i) {
			for (int j = 0; j < v.width(); ++j) {
				gistG[k++] = qGray(v.pixel(j, i));
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

