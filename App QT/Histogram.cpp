#include "imageTransformations.h"


std::vector<double> ImageTransformations::Histogram::normalizeHistogram(std::vector<int> histogram) {

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

std::vector<int> ImageTransformations::Histogram::linearHistogramEqualization(QImage image, int maxVal)
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



std::vector<int> ImageTransformations::Histogram::computeHistogram(QImage image)
{
	//We fill the histogram
	image = ImageTransformations::ColorSpaces::convertToGray(image);
	std::vector<std::vector<QRgb>> matrix = ImageTransformations::getPixels(image);

	std::vector<int> histogram(256, 0);

	for (int i = 0; i < matrix.size(); i++) {
		for (int j = 0; j < matrix[i].size(); j++) {
			histogram[qGray(matrix[i][j])]++;
		}
	}

	return histogram;

}

QImage ImageTransformations::Histogram::histogramToImage(std::vector<int> transformedHist, QImage originalImage)
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
std::vector<int> ImageTransformations::Histogram::equalizationHistogram(std::vector<int> histogram, int maxVal)
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

std::vector<int> ImageTransformations::Histogram::histogramSpecification(std::vector<int> histogram, std::vector<double> specifiedHistogram, int maxVal)
{
	//we will first compute the current normalized image histogram and denote it Pr

	std::vector<double> pr = normalizeHistogram(histogram);

	//Now we will compute the discrete formulation of the equalization tranform, and denote it s

	std::vector<int> s = equalizationHistogram(histogram, maxVal);

	//we will now compute the transformation function denoted g(Zq)

	std::vector<int> g(255, 0);
	int sum = 0;
	for (int i = 0; i < 256; i++) {
		sum += specifiedHistogram[i];
		g[i] = (maxVal - 1) * sum;
	}



}