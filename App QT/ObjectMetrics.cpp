#include "imageTransformations.h"

std::vector<int> ImageTransformations::ObjectMetrics::calculateArea(QVector<QPoint> points, QImage& image)
{

	std::sort(points.begin(), points.end(), [](const QPoint& a, const QPoint& b) {
		return a.x() < b.x();
		});

	//we obtain the marginals of each pixel
	std::map<int, QVector<QPoint>> clusters;
	for (const auto& points : points) {
		clusters[points.y()].push_back(points);
	}

	//We get the area
	int area = 0;
	for (const auto& cluster : clusters) {

		for (int j = 0; j < cluster.second.size() - 1; j++) {
			//area += abs(cluster.second[j + 1].x() - cluster.second[j].x());

			if (abs(cluster.second[j + 1].x() - cluster.second[j].x()) == 1) {
				area++;

			}
			else {

				area += abs(cluster.second[j + 1].x() - cluster.second[j].x()) + 1;
			}
		}
	}
	return std::vector<int>(1, area);
}


std::vector<int> ImageTransformations::ObjectMetrics::calculatePerimeter(QVector<QPoint> object, QImage& image)
{
	int res = object.size();

	std::vector<int> resultingMet;
	resultingMet.push_back(res);
	return resultingMet;
}

std::vector<int> ImageTransformations::ObjectMetrics::calculateCenterOfGravity(QVector<QPoint> object, QImage& image)
{
	std::vector<int> centerOfGravity(2);

	int moment0 = 0;
	int moment10 = 0;
	int moment01 = 0;

	for (int j = 0; j < object.size(); j++) { //We will iterate the points of the current object
		moment0 += 1;
		moment10 += object[j].x(); //x spatial distribution
		moment01 += object[j].y(); //y spatial distribution

	}

	centerOfGravity[0] = static_cast<int>(moment10 / moment0);
	centerOfGravity[1] = static_cast<int>(moment01 / moment0);



	return centerOfGravity;
}

Eigen::MatrixXd ImageTransformations::ObjectMetrics::featureExtraction(std::vector<std::function<std::vector<int>(QVector<QPoint>, QImage&)>> functions, QVector<QPoint> object, QImage& image)
{
	int minX = object[0].x();
	int minY = object[0].y();
	int maxX = object[0].x();
	int maxY = object[0].y();
	// Find the bounding box of the object
	for (const QPoint& point : object) {
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
	//QImage objectImageBinary = thereshold(objectImage, 130);
	double e = ImageTransformations::ObjectMetrics::calculateEccentricity((objectImage));
	Eigen::MatrixXd descriptors(1, functions.size() + 1 + 1);
	int metricCount = 0;
	for (auto func : functions) {



		std::vector<int> currMetric = func(object, image);

		//we iterate the current metrics values

		if (currMetric.size() > 1) {
			descriptors.conservativeResize(descriptors.rows(), descriptors.cols() + currMetric.size() - 1);
		}

		for (auto metric : currMetric) {
			descriptors(0, metricCount) = metric;
			metricCount++;
		}
	}// area, per
	descriptors(0, metricCount) = e;
	double c = imageMoments(objectImage, 0, 0);
	descriptors(0, metricCount + 1) = c;
	return descriptors;
}

double ImageTransformations::ObjectMetrics::imageMoments(QImage& binarizedImage, int p, int q)
{
	double res = 0;
	for (int y = 0; y < binarizedImage.height(); y++) {
		for (int x = 0; x < binarizedImage.width(); x++) {
			int pixelVal = qGray(binarizedImage.pixel(x, y));
			if (pixelVal == 255) { // Asume que la imagen est? binarizada con fondo negro (0) y objetos blancos (255)
				res += pow(x, q) * pow(y, p);
			}
		}
	}
	return res;
}

double ImageTransformations::ObjectMetrics::centralMoments(QImage& binarizedImage, int p, int q, double xCent, double yCent)
{
	double res = 0;
	for (int y = 0; y < binarizedImage.height(); y++) {
		for (int x = 0; x < binarizedImage.width(); x++) {
			int pixelVal = qGray(binarizedImage.pixel(x, y));
			if (pixelVal == 255) { // Asume que la imagen est? binarizada con fondo negro (0) y objetos blancos (255)
				res += pow(x - xCent, p) * pow(y - yCent, q);
			}
		}
	}
	return res;
}

double ImageTransformations::ObjectMetrics::calculateEccentricity(QImage& image)
{
	double m00 = imageMoments(image, 0, 0);
	double m10 = imageMoments(image, 1, 0);
	double m01 = imageMoments(image, 0, 1);

	double cx = m10 / m00;
	double cy = m01 / m00;

	double mu20 = centralMoments(image, 2, 0, cx, cy);
	double mu02 = centralMoments(image, 0, 2, cx, cy);
	double mu11 = centralMoments(image, 1, 1, cx, cy);

	double a = sqrt(2 * (mu20 + mu02 + sqrt(pow(mu20 - mu02, 2) + 4 * pow(mu11, 2))));
	double b = sqrt(2 * (mu20 + mu02 - sqrt(pow(mu20 - mu02, 2) + 4 * pow(mu11, 2))));

	double c = sqrt(a * a - b * b);
	double e = c / a;

	return 200.000 * e;
}
