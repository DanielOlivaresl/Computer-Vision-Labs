#include "ComputerVisionApplication.h"
#include <QtWidgets/QApplication>
#include <iostream>
int main(int argc, char *argv[])
{
    srand(time(0));
    QApplication a(argc, argv);
    ComputerVisionApplication w;
    w.show();
    return a.exec();

    

}