#include "ComputerVisionApplication.h"
#include <QtWidgets/QApplication>
#include <iostream>
#include "image.h"
#include<QMetaType>
int main(int argc, char *argv[])
{





    srand(time(0));
    QApplication a(argc, argv);
    
    ComputerVisionApplication w;
    w.show();
    qRegisterMetaType<Image>("Image");

    return a.exec();

    

}
