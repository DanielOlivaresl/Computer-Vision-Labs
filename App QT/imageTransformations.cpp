#include "imageTransformations.h"

QImage ImageTransformations::convertToGray(QImage& image) {
	
	QImage newImage = image.convertToFormat(QImage::Format_Grayscale8);
	return newImage;

}

void ImageTransformations::imageObjectsToCsv(QImage& image, QString fileName,std::string csvFileName, std::vector<QImage>& subimages)
{

	// calling the function that retrieves the information 
	QVector<QVector<QPoint>> objects = ImageTransformations::connectedN4(image);

	//we first reset the current csv, and then we will start appending to it
	
	
	std::ofstream outFile(csvFileName, std::ios::app);

	if (!outFile.is_open()) {
		std::cerr << "No se pudo abrir el archivo para escritura." << std::endl;
		return;
	}

	for (int i = 0; i < objects.size(); i++) {
		int minX, maxX, minY, maxY;
		calculateBounds(objects[i], minX, maxX, minY, maxY);
		// getting the subimage of the object
		QVector<QPoint> pointsS = objects[i];

		// Find the bounding box of the object
		

		// Create a new image containing only the object
		QImage objectImage = image.copy(minX - 5, minY - 5, maxX - minX + 10, maxY - minY + 10);
		
		QImage objectImageBinary = threshold(objectImage, 130);
		subimages.push_back(objectImageBinary);
		double e = ObjectMetrics::calculateEccentricity((objectImageBinary));
		QImage ImageBinary = threshold(image, 130);

		

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
			//we end the object in the csv
			outFile << "Object " <<i<<" "<< fileName.toStdString() << "\n";

		}
		outFile.close();
	}
}


QImage ImageTransformations::threshold(QImage& image, int threshold) {
	if (image.isNull()) {
		return image;
	}
	//We pass to gray color
	QImage gray = convertToGray(image);
	QImage thresholdedImage(gray.size(), QImage::Format_Grayscale8);


	


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
	QImage binaryImage = threshold(image, 130);
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
		i++;
		std::vector<double> distances = euclidean(centroids, pointToClassify);
		qDebug() << "Distancias para ese objeto ";
		for (auto d : distances)
		{
			qDebug() << d;
		}
		classification.push_back(getClosest(distances));
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


void ImageTransformations::calculateBounds(QVector<QPoint> objectBorder, int& minX, int& maxX, int& minY, int& maxY)
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

std::vector<QImage> ImageTransformations::calculatezSubImage(QImage& image)
{
	QVector<QVector<QPoint>> objects = connectedN4(image);
	int xMax, xMin, yMax, yMin;


	std::vector<QImage> subImages;
	for (QVector<QPoint> object : objects) {
		calculateBounds(object, xMin, xMax, yMin, yMax);
		QImage objectImage = image.copy(xMin - 5, yMin - 5, xMax - xMin + 10, yMax - yMin + 10);
		subImages.push_back(objectImage);

	}






	return subImages;
}

void ImageTransformations::storeImages(std::string path, std::vector<QImage> images,int counter)
{
	QString qPath = QString::fromStdString(path);
	QDir().mkpath(qPath);
	for (auto image : images) {
		QString imagePath = QDir(qPath).filePath(QString("SubImage_%1.png").arg(++counter));
		image.save(imagePath);
	}



}


