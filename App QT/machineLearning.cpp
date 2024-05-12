#include "machineLearning.h"

Perceptron::Perceptron(const Eigen::Matrix<double, Eigen::Dynamic, 3>& X_input, const std::string& F_input)
{
    inputMean = X_input.colwise().mean();  


    Eigen::RowVector3d X_mean_row = inputMean.transpose();
    inputStd = Eigen::Vector3d::Zero();
    for (int i = 0; i < X_input.rows(); ++i) {
        inputStd += (X_input.row(i) - X_mean_row).array().square().matrix();
    }
    inputStd = (inputStd / X_input.rows()).array().sqrt();


    inputData = init_data(X_input);
    activationFunction = F_input;
    weights = init_weights(X_input.cols());
}

void Perceptron::showInfo()
{
    qDebug() << "Perceptron info\n";
    qDebug() << "Iterating inputMean:";
    for (int i = 0; i < inputMean.size(); ++i) {
        qDebug() << "X_mean[" << i << "] = " << inputMean[i];
    }
    qDebug() << "Iterating inputData:";
    for (int i = 0; i < inputData.rows(); ++i) {
        for (int j = 0; j < inputData.cols(); ++j) {
            qDebug() << "X[" << i << "," << j << "] = " << inputData(i, j);
        }
    }
    qDebug() << "Iterating weights:";
    for (int i = 0; i < weights.size(); ++i) {
        qDebug() << "weights[" << i << "] = " << weights[i];
    }
    
}
Eigen::MatrixXd Perceptron::init_data(const Eigen::MatrixXd& X_input)
{
    Eigen::MatrixXd X_normalized = (X_input.rowwise() - inputMean.transpose()).array().rowwise() / (inputStd.transpose().array() + std::numeric_limits<float>::epsilon());
    Eigen::VectorXd ones_column = Eigen::VectorXd::Ones(X_input.rows());
    Eigen::MatrixXd X_with_ones(X_input.rows(), X_input.cols() + 1);
    X_with_ones << X_normalized, ones_column;
    return X_with_ones;
}
Eigen::VectorXd Perceptron::init_weights(const int& input_shape) 
{
    return Eigen::VectorXd::Random(input_shape + 1);
}
Eigen::VectorXd Perceptron::get_h(const Eigen::MatrixXd& inputData)
{
    return inputData * weights;
}
Eigen::VectorXd Perceptron::sigmoid(const Eigen::VectorXd& h)
{
    return 1 / (1 + (-h.array()).exp());
}
Eigen::VectorXd Perceptron::sigmoid_prime(const Eigen::VectorXd & h)
{
    return sigmoid(h).array() * (1 - sigmoid(h).array());
}
Eigen::VectorXd Perceptron::get_activation(const Eigen::VectorXd& h)
{
    if (activationFunction == "sigmoid")
    {
        return sigmoid(h);
    }
    return Eigen::VectorXd::Zero(h.size());
}
Eigen::VectorXd Perceptron::feed_forward(const Eigen::MatrixXd& X_data)
{
    return get_activation(get_h(X_data));
}
Eigen::VectorXd Perceptron::error(const Eigen::VectorXd& Target)
{
    return Target - feed_forward(inputData);
}
Eigen::VectorXd Perceptron::error_term(const Eigen::VectorXd& Target)
{
    if (activationFunction == "sigmoid") {
        return error(Target).array() * sigmoid_prime(get_h(inputData)).array();
    }
    return Eigen::VectorXd::Zero(Target.size());
}
Eigen::VectorXd Perceptron::get_increment(const Eigen::VectorXd& target, const float& learning_rate)
{
    return learning_rate * error_term(target).transpose() * inputData;
}
Eigen::VectorXd Perceptron::update_weights(const Eigen::VectorXd& target, const float& learning_rate)
{
    return get_increment(target, learning_rate) + weights;
}
Eigen::MatrixXd Perceptron::init_query(const Eigen::VectorXd& input)
{
    qDebug() << "Mostrando size del vector de entrada " + std::to_string(input.size()) + "\n";

    Eigen::VectorXd X_normalized = (input - inputMean).array() / (inputStd.array() + std::numeric_limits<float>::epsilon());
    qDebug() << "Mostrando size del mean " + std::to_string(inputMean.size()) + "\n";
    qDebug() << "Mostrando size del Std " + std::to_string(inputStd.size()) + "\n";
    qDebug() << "Mostrando size del vector normalizado " + std::to_string(X_normalized.size()) + "\n";

    Eigen::VectorXd X_with_ones(X_normalized.size() + 1);
    X_with_ones.head(X_normalized.size()) = X_normalized;
    X_with_ones(X_normalized.size()) = 1; // Set the last element to 1

    // Convert the vector to a 1x(n+1) matrix
    Eigen::MatrixXd result = X_with_ones.transpose().reshaped(1, X_with_ones.size());
    // Alternatively, for older versions of Eigen:
    // Eigen::MatrixXd result(1, X_with_ones.size());
    // result.row(0) = X_with_ones.transpose();

    qDebug() << "Mostrando lo que se obtiene de la limpieza de la query, size de la query " + std::to_string(result.rows()) + "x" + std::to_string(result.cols()) + "\n";

    for (int i = 0; i < result.cols(); i++) {
        qDebug() << result(0, i) << " ";
    }

    return result;
}
Eigen::VectorXd Perceptron::query(const Eigen::VectorXd& x) // arreglar esta junto con init query
{
    Eigen::MatrixXd x_normalized = init_query(x);
    qDebug() << "Observando lo obtenido " + std::to_string(feed_forward(x_normalized).size());
    return feed_forward(x_normalized);
}

void printM(Eigen::MatrixXd m)
{
    for (int i = 0; i < m.rows(); i++)
    {
        for (int j = 0; j < m.cols(); j++)
        {
            qDebug() << m(i, j) << "";
        }
        qDebug() << "\n";
    }
}

std::pair<std::vector<Eigen::MatrixXd>, Eigen::MatrixXd> ML::Kmeans(const Eigen::MatrixXd data, int k, float threshold)
{
    std::pair<std::vector<Eigen::MatrixXd>, Eigen::MatrixXd> result;
    // Creates the centroids 
    Eigen::MatrixXd centroids(k, data.cols());
    // getting a random smaple from the data
    for (int i = 0; i < k; i++)
    {
        for (int j = 0; j < data.cols(); j++)
        {
            centroids(i, j) = data(rand() % data.rows(), j);
        }
    }

    int iter = 0;
    Eigen::MatrixXd centroids_copy = centroids; // to check how  much the centroids move in each iteration 
    std::vector<Eigen::MatrixXd> matrixClassesFinal; // to store the data in the class that corresponds
    while (true)
    {
        // Calculates euc distance--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
        Eigen::MatrixXd distances(data.rows(), k); // matrix to store distances. [diastance to c1, distance to c2, distance to c3] instance 1,[diastance to c1, distance to c2, distance to c3] instance 2 ... 
        for (int i = 0; i < k; i++)
        {
            Eigen::MatrixXd data_minus_centroid = data.rowwise() - centroids.row(i);
            data_minus_centroid = data_minus_centroid.array().square();
            Eigen::VectorXd sum_squared_diff_sqrt = (data_minus_centroid.rowwise().sum()).array().sqrt();
            distances.col(i) = sum_squared_diff_sqrt;
        }
        // ----------------------------------------------------------------------------------------------------------------------------- here whe can call another dist function if needed but must be the same shape as distances...



        std::vector<Eigen::MatrixXd> matrixClasses; // to store where all instances belong in the classes--------------------------------------------
        for (int c = 0; c < k; c++)
        {
            matrixClasses.push_back(Eigen::MatrixXd(0, data.cols())); // init the vector of classes
        }
        for (int ind = 0; ind < distances.rows(); ind++)
        {
            Eigen::MatrixXd::Index minIndex;
            distances.row(ind).minCoeff(&minIndex);
            //qDebug() << "centoide mas cercano para el ejemplo " << ind << " Es : " << minIndex << "\n";
            matrixClasses[minIndex].conservativeResize(matrixClasses[minIndex].rows() + 1, Eigen::NoChange); // adding a new row
            matrixClasses[minIndex].row(matrixClasses[minIndex].rows() - 1) = data.row(ind);
        }
        //--------------------------------------------------------------------------------------------------------------------------------------------
        
        
        for (int r = 0; r < centroids.rows(); r++) // Calculates and sets the values of each centroid with the mean of each column in the matrixClasses
        {
            for (int c = 0; c < centroids.cols(); c++)
            {
                centroids(r, c) = matrixClasses[r].col(c).mean();
            }
        }//-----------------------------------------------------------------


        int tf = 0;   // Used to break the loop when all the entries in the rest between actual centroids and past centroids are less than the threshold 
        for (int i = 0; i < (centroids - centroids_copy).rows(); i++)
        {
            for (int j = 0; j < (centroids - centroids_copy).cols(); j++)
            {
                if ((centroids - centroids_copy)(i, j) < threshold)
                {
                    tf++;
                }
            }
        }
        if (tf == (centroids - centroids_copy).size())
        {
            matrixClassesFinal = matrixClasses;
            break;
        }//--------------------------------------------------------------------------------------------------------------------------------------------------

        centroids_copy = centroids; // to store the last centroids calculated
        iter++; // iterator to store how much iterations ocurred
    }
    qDebug() << "Centoides finales en " << iter << "Iteraciones ";
    printM(centroids);
    result = std::make_pair(matrixClassesFinal, centroids);
    return result;
}

