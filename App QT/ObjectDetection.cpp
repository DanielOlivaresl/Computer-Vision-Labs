#include "ObjectDetection.h"
#include "imageTransformations.h"

void ObjectDetection::segmentImage(cv::Mat& image, const cv::Mat& labels, int numSuperpixels, int threshold) {
	cv::Mat visited = cv::Mat::zeros(image.rows, image.cols, CV_8UC1);
	std::vector<cv::Vec3b> colors(numSuperpixels); // Assume a function to generate unique colors

	for (int i = 0; i < numSuperpixels; i++) {
		colors[i] = cv::Vec3b(rand() % 256, rand() % 256, rand() % 256);
	}

	for (int y = 0; y < image.rows; ++y) {
		for (int x = 0; x < image.cols; ++x) {
			int label = labels.at<int>(y, x);
			if (!visited.at<uchar>(y, x)) {
				// Perform region growing based on color similarity
				//regionGrowing(image, cv::Point(x, y), threshold, visited, colors[label]);
			}
		}
	}
}

void ObjectDetection::regionGrowing(cv::Mat& image, const std::vector<cv::Point>& seeds, int threshold, cv::Mat& visited, cv::Vec3b regionColor) {
	cv::Mat result = cv::Mat::zeros(image.size(), CV_8U);

	result = cv::Mat::zeros(image.size(), image.type());

	// Initialize a mask to keep track of visited pixels
	visited = cv::Mat::zeros(image.size(), CV_8U);

	// Define the directions for 8-neighborhood
	int dx[] = { -1, 1, 0, 0, -1, -1, 1, 1 };
	int dy[] = { 0, 0, -1, 1, -1, 1, -1, 1 };

	// Create a random color for each region
	cv::RNG rng(12345);
	std::vector<cv::Vec3b> regionColors;

	// Initialize a queue for the region growing
	std::queue<cv::Point> queue;

	for (const auto& seed : seeds) {
		if (!visited.at<uchar>(seed)) {
			// Generate a random color for the new region
			cv::Vec3b color(rng.uniform(0, 256), rng.uniform(0, 256), rng.uniform(0, 256));
			regionColors.push_back(color);

			queue.push(seed);
			visited.at<uchar>(seed) = 1;
			result.at<cv::Vec3b>(seed) = color;

			while (!queue.empty()) {
				cv::Point p = queue.front();
				queue.pop();

				for (int i = 0; i < 8; ++i) {
					int nx = p.x + dx[i];
					int ny = p.y + dy[i];

					if (nx >= 0 && nx < image.cols && ny >= 0 && ny < image.rows && !visited.at<uchar>(ny, nx)) {
						cv::Vec3b neighbor_value = image.at<cv::Vec3b>(ny, nx);
						cv::Vec3b seed_value = image.at<cv::Vec3b>(p);
						if (cv::norm(neighbor_value - seed_value) < threshold) {
							visited.at<uchar>(ny, nx) = 1;
							result.at<cv::Vec3b>(ny, nx) = color;
							queue.push(cv::Point(nx, ny));
						}
					}
				}
			}
		}
	}


	//cv::imshow("Original Image", image);
	cv::imshow("Segmented Image", result);
	cv::waitKey(0);





















}


std::pair<cv::Mat, cv::Mat> ObjectDetection::generateSuperpixels(const cv::Mat& inputImage, int numberOfSuperpixels, int ruler)
{
	cv::Mat convertedImage;
	inputImage.convertTo(convertedImage, CV_32F);
	cv::Ptr<cv::ximgproc::SuperpixelSLIC> slic = cv::ximgproc::createSuperpixelSLIC(convertedImage, cv::ximgproc::SLIC, numberOfSuperpixels, float(ruler));
	slic->iterate();
	cv::Mat labels, mask;
	slic->getLabels(labels);
	slic->getLabelContourMask(mask, true); // true for thicker boundaries
	return { labels, mask };
}

void ObjectDetection::colorSuperpixelBorders(cv::Mat& image, const cv::Mat& labels, const cv::Mat& mask)
{
	cv::Vec3b borderColor(0, 255, 0); // Green color for borders

	for (int y = 0; y < mask.rows; y++) {
		for (int x = 0; x < mask.cols; x++) {
			if (mask.at<uchar>(y, x) != 0) {  // Check if this is a border pixel
				image.at<cv::Vec3b>(y, x) = borderColor; // Apply border color
			}
		}
	}
}

void ObjectDetection::createBoundaryMask(cv::Mat& labels, cv::Mat& boundaryMask)
{
	cv::Ptr<cv::ximgproc::SuperpixelSLIC> slic = cv::ximgproc::createSuperpixelSLIC(labels, cv::ximgproc::SLIC, 100, 10.0);
	slic->iterate();
	slic->getLabelContourMask(boundaryMask, true);

}

void ObjectDetection::colorBorders(cv::Mat& image, const cv::Mat& boundaryMask, cv::Mat& labels)
{
	std::vector<cv::Vec3b> colors; // Generate random colors for each superpixel
	int numSuperpixels = 100; // This should match the number you used to generate the superpixels
	for (int i = 0; i < numSuperpixels; i++) {
		colors.push_back(cv::Vec3b(rand() % 256, rand() % 256, rand() % 256));
	}

	for (int y = 0; y < boundaryMask.rows; y++) {
		for (int x = 0; x < boundaryMask.cols; x++) {
			if (boundaryMask.at<uchar>(y, x) != 0) {  // If it's a boundary pixel
				int label = labels.at<int>(y, x);
				image.at<cv::Vec3b>(y, x) = colors[label % numSuperpixels];
			}
		}
	}

}

void ObjectDetection::adjustBoundaryMask(cv::Mat& mask, bool thinBorders)
{
	cv::Mat element = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(3, 3));
	if (thinBorders) {
		cv::erode(mask, mask, element);
	}
	else {
		cv::dilate(mask, mask, element);
	}

}

void ObjectDetection::fillSuperpixelsWithColors(cv::Mat& image, const cv::Mat& labels)
{
	std::vector<cv::Vec3b> colors; // Vector to store colors for each superpixel
	int numSuperpixels = labels.rows * labels.cols; // This is an upper bound, actual number might be lower
	cv::RNG rng; // Random number generator for colors

	for (int i = 0; i < numSuperpixels; ++i) {
		colors.push_back(cv::Vec3b(rng.uniform(0, 255), rng.uniform(0, 255), rng.uniform(0, 255)));
	}

	for (int y = 0; y < labels.rows; ++y) {
		for (int x = 0; x < labels.cols; ++x) {
			int label = labels.at<int>(y, x);
			image.at<cv::Vec3b>(y, x) = colors[label]; // Fill each pixel with its superpixel's color
		}
	}

}

void ObjectDetection::regionGrowing(cv::Mat& image, cv::Mat& labels, int threshold)
{
	int rows = image.rows;
	int cols = image.cols;
	cv::Mat visited = cv::Mat::zeros(rows, cols, CV_8UC1);

	for (int y = 0; y < rows; y++) {
		for (int x = 0; x < cols; x++) {
			if (visited.at<uchar>(y, x) == 0) {
				std::queue<cv::Point> queue;
				queue.push(cv::Point(x, y));
				int label = labels.at<int>(y, x);
				cv::Vec3b seedColor = image.at<cv::Vec3b>(y, x);

				while (!queue.empty()) {
					cv::Point p = queue.front();
					queue.pop();

					for (int dy = -1; dy <= 1; dy++) {
						for (int dx = -1; dx <= 1; dx++) {
							int ny = p.y + dy;
							int nx = p.x + dx;
							if (ny >= 0 && ny < rows && nx >= 0 && nx < cols && visited.at<uchar>(ny, nx) == 0) {
								cv::Vec3b color = image.at<cv::Vec3b>(ny, nx);
								if (cv::norm(color - seedColor) < threshold) {
									labels.at<int>(ny, nx) = label;
									visited.at<uchar>(ny, nx) = 1;
									queue.push(cv::Point(nx, ny));
								}
							}
						}
					}
				}
			}
		}
	}

}

void ObjectDetection::applyRegionGrowing(cv::Mat& image, cv::Mat& labels, int threshold)
{
	for (int i = 0; i < labels.rows; i++) {
		for (int j = 0; j < labels.cols; j++) {
			int label = labels.at<int>(i, j);
			cv::Point seed(j, i);
			cv::Vec3b seedColor = image.at<cv::Vec3b>(seed);
			//regionGrowing(image, labels, seedColor, seed, threshold);
		}
	}

}

std::vector<QColor> ObjectDetection::generateColors()
{
	std::vector<QColor> colors;
	colors.push_back(Qt::red);
	colors.push_back(Qt::green);
	colors.push_back(Qt::blue);
	colors.push_back(Qt::yellow);
	colors.push_back(Qt::cyan);
	colors.push_back(Qt::magenta);
	// Add more colors as needed
	return colors;




}



QVector<QVector<QPoint>> ObjectDetection::connectedN4(QImage& image) {
	QImage binaryImage = ImageTransformations::ColorSpaces::threshold(image, 130);
	QVector<QVector<QPoint>> objects = rasterScan(binaryImage);
	for (int i = 0; i < objects.size(); i++) {
		if (objects[i].size() < 100) {
			objects.remove(i);
			i = 0;

		}
	}
	return objects;
}










/**
* @brief Function that find object online
* @param QImage: Image to segment
* @param int i: x-axis coordinate
* @param int j: y-axis coordinate
*
*/
QVector<QPoint> ObjectDetection::outLine(QImage& image, int i, int j) {
	//we define the objects vector
	QVector<QPoint> object;
	//we define variables that we'll use later
	int currI = i;
	int currJ = j;
	int d = 0;
	std::vector<std::vector<QRgb>> matrix = ImageTransformations::getPixels(image);
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




// Assuming the existence of the borderFollow and nextCell functions as previously translated

QPoint ObjectDetection::nextCell(QPoint& currPixel, int& currDir, QPoint& savePixel) {
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

QVector<QPoint> ObjectDetection::borderFollow(QImage& img, QPoint& start, QPoint& prev, int direction, int& NBD) {
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

QVector<QVector<QPoint>> ObjectDetection::rasterScan(QImage& img) {
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



std::vector<std::string> ObjectDetection::classifyImage(QImage& image, Eigen::MatrixXd centroids, std::vector < std::function <std::vector<int>(QVector<QPoint>, QImage&)>> functions, std::map<int, std::string> namesMap) {

	//first we will apply all the transformations that were applied to images in the dataset, in order to get the metrics

	QVector<QVector<QPoint>> objects;
	objects = connectedN4(image);

	//we iterate each of the objects and get they're features

	std::vector<Eigen::MatrixXd> objectMetrics;

	Eigen::MatrixXd objetitos(objects.size(), 3);

	std::vector<int> classification; //Vector that stores the classification of each object
	bool flag = false;
	int i = 0;
	for (auto object : objects) {
		Eigen::MatrixXd pointToClassify = ImageTransformations::ObjectMetrics::featureExtraction(functions, object, image); // aki
		objectMetrics.push_back(pointToClassify);
		qDebug() << " objeto perimetro " << pointToClassify(0) << " objeto exce " << pointToClassify(1) << " objeto area " << pointToClassify(2);
		if (pointToClassify(2) > 1400) { flag = true; }
		//Once we got they're features we will calculate the distance of each object to the centroid given by Kmeans
		objetitos(i, 0) = pointToClassify(0, 0);
		objetitos(i, 1) = pointToClassify(0, 1);
		objetitos(i, 2) = pointToClassify(0, 2);
		i++;
		std::vector<double> distances = Computations::Distances::euclidean(centroids, pointToClassify);
		qDebug() << "Distancias para ese objeto ";
		for (auto d : distances)
		{
			qDebug() << d;
		}
		classification.push_back(Computations::Helper::getClosest(distances));
	}

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


void ObjectDetection::calculateBounds(QList<QPoint> objectBorder, int& minX, int& maxX, int& minY, int& maxY)
{
	minX = objectBorder[0].x();
	minY = objectBorder[0].y();
	maxX = objectBorder[0].x();
	maxY = objectBorder[0].y();

	for (const QPoint& point : objectBorder) {
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




}
