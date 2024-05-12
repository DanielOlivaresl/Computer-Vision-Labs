#include "image.h"	
#include<QDebug>


Image::Image(QObject* parent) : QObject(parent) {
    qDebug() << "Image default constructor called";
}

Image::~Image() {
    qDebug() << "Image default destructor called";

}

Image::Image(const Image& other) : QObject(other.parent()) {
    this->numClasses = other.numClasses;
    this->currProcess = other.currProcess;
    this->rectangles = other.rectangles;
    this->matrixClasses = other.matrixClasses;
    this->image = other.image;
   
}
