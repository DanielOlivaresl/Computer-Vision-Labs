#ifndef IMAGE_H
#define IMAGE_H
//Q_DECLARE_METATYPE(Image);

#include <QObject>
#include <string>
#include <vector>
#include <Qpoint>
#include <QImage>
#include <Eigen/Dense>
#include<QMetaType>
class Image : public QObject {
	Q_OBJECT
public:

	Image(QObject* parent = nullptr);
	Image(const Image& other);
	~Image();
	int numClasses = 2;
	std::string currProcess;
	std::vector<QPointF> rectangles;
	std::vector<Eigen::Matrix<double, Eigen::Dynamic, 3>> matrixClasses;
	QImage image;

};


#endif 