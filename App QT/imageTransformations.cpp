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


std::vector < std::vector<double>>  ImageTransformations::normalizeImage(const QImage& image)
{
	std::vector<std::vector<double>> res(image.width(),std::vector<double>(image.height()));

	QImage grayImage = image.convertToFormat(QImage::Format_Grayscale8);
	uchar minVal = 255;
	uchar maxVal = 0;

	for (int y = 0; y < image.height(); ++y) {
		for (int x = 0; x < image.width(); ++x) {
			uchar pixel = grayImage.pixelColor(x, y).value();
			if (pixel < minVal) minVal = pixel;
			if (pixel > minVal) maxVal = pixel;

		}
	}
	if (maxVal == minVal) {
		for (int y = 0; y < grayImage.height(); ++y) {
			for (int x = 0; x < grayImage.width(); ++x) {
				res[x][y] = 255.0;
			}
		}

	}else {

		for (int y = 0; y < grayImage.height(); ++y) {
			for (int x = 0; x < grayImage.width(); ++x) {
				double pixel = grayImage.pixelColor(x,y).value();
				double normPixel = 255 * (pixel - minVal)/(maxVal-minVal);
				res[x][y] = normPixel;
			}
		}
	}
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


