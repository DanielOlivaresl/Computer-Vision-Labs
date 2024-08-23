#include "computations.h"


void Computations::Math::fft1D(std::vector<std::complex<double>>&data, bool invert)
{


	int n = data.size();
	if (n <= 1) {
		return;
	}

	std::vector<std::complex<double>> even(n / 2);
	std::vector<std::complex<double>> odd(n / 2);


	for (int i = 0; i < n / 2; i++) {
		even[i] = data[i * 2];
		odd[i] = data[i * 2 + 1];
	}

	fft1D(even, invert);
	fft1D(odd, invert);

	double ang = 2 * M_PI / n * (invert ? -1 : 1);
	std::complex<double> w(1), wn(cos(ang), sin(ang));

	for (int i = 0; i < n / 2; ++i) {
		data[i] = even[i] + w * odd[i];
		data[i + n / 2] = even[i] - w * odd[i];
		if (invert) {
			data[i] /= 2;
			data[i + n / 2] /= 2;
		}
		w *= wn;
	}


}

std::vector<std::vector<std::complex<double>>> Computations::Math::fft2D(const std::vector<std::vector<std::complex<double>>>& data, bool inverted)
{
	int height = data.size();
	int width = data[0].size();

	std::vector<std::vector<std::complex<double>>> result = data;
	//Apply fft to rows
	for (int y = 0; y < height; ++y) {
		fft1D(result[y], inverted);
	}

	//Transpose the result
	std::vector<std::vector<std::complex<double>>> transposed(width, std::vector<std::complex<double>>(height));

	for (int y = 0; y < height; ++y) {
		for (int x = 0; x < width; ++x) {
			transposed[x][y] = result[y][x];
		}
	}


	// Apply FFT to columns (transposed rows)
	for (size_t x = 0; x < width; ++x) {
		fft1D(transposed[x], inverted);
	}

	// Transpose the result back
	for (size_t y = 0; y < height; ++y) {
		for (size_t x = 0; x < width; ++x) {
			result[y][x] = transposed[x][y];
		}
	}
		
	return result;



}


//Function that numerically aproximates the derivative of a given univariable function in one point
double Computations::Math::aproximateDerivative(double point, std::function<double(double)> f, double precision)
{
	//This is calculated by the formula of the definition of a derivative which is the limit of h->0 : {f(x) - f(x-h) }	/h , as h grows smaller the result will be more precise


	//First we will calculate f(x) , and f(x-h)

	double f_x = f(point);
	double f_x_h = f(point - precision);


	double res = (f_x_h - f_x) / precision;

	return res;




}

//Function that numerically aproximates the derivative of a multivariable function, given the variable to derivate and a point
double Computations::Math::aproximateDerivative(Eigen::VectorXd point, std::function<double(Eigen::VectorXd)> f, double precision, int variableNum)
{

	//We check that the variable to derivate is valid
	if (variableNum >= point.size()) {
		throw std::runtime_error("Invalid variable to derivate");
	}

	//Now that the variables have been validated, we will compute the derivative with respect to that function
	// 
	double f_x = f(point);
	Eigen::VectorXd x_h = point;
	x_h(variableNum) = point(variableNum) + precision;
	double f_x_h = f(x_h);

	double res = (f_x_h - f_x) / precision;

	return res;
}




double Computations::ActivationFunctions::sigmoid(double x) {
	return 1 / (1 + exp(-x));
}


double Computations::ActivationFunctions::tanh(double x) {
	return (exp(x) - exp(-x)) / (exp(x) + exp(-x));
}




double Computations::ActivationFunctions::relu(double x) {
	return std::max(0.0, x);
}

double Computations::ActivationFunctions::leakyRelu(double x) {

	if (x >= 0) {
		return x;
	}
	else {
		return x * 0.01;
	}


}

double Computations::ActivationFunctions::swish(double x) {
	return x * sigmoid(x);
}

Eigen::VectorXd Computations::ActivationFunctions::softmax(Eigen::VectorXd input)
{


	//we first compute the sum of the exponential of the input vector

	double sum = input.array().exp().sum();

	//now we compute the softmax function 

	Eigen::VectorXd res = input.array() / sum;

	return res;

	






}
