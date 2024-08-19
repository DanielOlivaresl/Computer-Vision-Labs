#pragma once
#ifndef COMPUTATIONS_H
#define COMPUTATIONS_H

#include<iostream>
#include<vector>
#include<cmath>	
#include<algorithm>
#include<map>
#include <QDebug>
#include<Eigen/Dense>
#include <QString>
#include<QImage>
#include <QPoint>



class Computations { //Base class

public:

	class Distances {
	public:
		
		static Eigen::VectorXd euclidean(std::vector<Eigen::MatrixXd> classes, Eigen::VectorXd point);
		static Eigen::VectorXd euclidean(Eigen::MatrixXd, Eigen::VectorXd point);
		static double euclidean(Eigen::VectorXd point1, Eigen::VectorXd point2);



		static Eigen::VectorXd manhalanobis(std::vector<Eigen::MatrixXd> classes, Eigen::VectorXd point);
	};


	class LinearAlgebra {
	public:
		static Eigen::MatrixXd calculateCovMatrix(Eigen::MatrixXd data);
	};

	class Math {
	public:
		//Fourier transform functions

		static void fft1D(std::vector<std::complex<double>>& data, bool invert);
		static std::vector<std::vector<std::complex<double>>> fft2D(const std::vector<std::vector<std::complex<double>>>& data, bool invert);



	};


	class Helper {
	public:
		static int getClosest(std::vector<double> distances);
		static int getClosest(Eigen::VectorXd distances);
		static int getMaxProb(std::vector<double> probabilities);
		static int getMaxProb(Eigen::VectorXd probabilities);
		static Eigen::VectorXd max_prob(std::vector<Eigen::MatrixXd> classes, Eigen::VectorXd point);
		static std::vector<double> max_prob(std::vector<Eigen::Matrix<double, Eigen::Dynamic, 3>> classes, Eigen::Vector3d point);
		static void normalizeColumn(Eigen::MatrixXd& matrix, int column);
		static int clockwise(int dir);
		static int counterclockwise(int dir);

	
	};



};
















































#endif
