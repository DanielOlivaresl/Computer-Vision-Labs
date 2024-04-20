#include "imageTransformations.h"

 QImage ImageTransformations::convertToGray(QImage &image) {
	if (image.isNull()) {
		return image;
	}
	//We obtain the image matrix
	std::vector<std::vector<QRgb>> matrix = getPixels(image);
	int height = matrix.size();
	int width = matrix[0].size();
	//We then convert the image to gray



	for (int y = 0; y < height; y++) {
		for (int x = 0; x < width; x++) {
	
			QRgb currentPixel = matrix[y][x];
			int value = qRed(currentPixel) * 0.299+ qGreen(currentPixel) * 0.587+ qBlue(currentPixel) * 0.114;
			matrix[y][x] = qRgb(value,value,value);
			
		}
	}
	

	image = createImage(matrix);
	return image;
	

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

 QImage ImageTransformations::createImage(std::vector<std::vector<QRgb>> matrix){

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
			 matrix[i][j] = qRgb(255-matrix[i][j], 255 - matrix[i][j], 255 - matrix[i][j]);
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

 QImage ImageTransformations::gammaTransform(QImage image, double c, double alpha){

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

 std::vector<double> ImageTransformations::normalizeHistogram(std::vector<int> histogram){

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
		 res[i] = maxVal*sum;
		 


	 }



	 return res;
 }

 std::vector<int> ImageTransformations::histogramSpecification(std::vector<int> histogram,std::vector<double> specifiedHistogram, int maxVal)
 {

	 std::vector<int> g(256, 0);
	 double sum =0 ;
	 for (int i = 0; i < 256; i++) {
		 sum += specifiedHistogram[i];
		 g[i] = sum * maxVal;
		 

	 }

	 std::vector<int> eqHist = equalizationHistogram(histogram, maxVal);

	 std::vector<int> res(256, 0);

	 for (int i = 0; i < 256; i++) {

	 }


 }

