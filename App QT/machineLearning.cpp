#include "machineLearning.h"



int MachineLearning::kNearestNeighbours(std::vector<Eigen::MatrixXd> classes, Eigen::VectorXd point, int k) {
    //Vector that stores a set of vectors that have the distance to the point, this is because each index in the vector represents a class,
    //This means that each element in the vector is another vector that stores the distances to the point from the ith class
    std::vector<std::vector<double>> distances;

    //We iterate the classes and fill the vectors
    for (int i = 0; i < classes.size(); i++) {
        
        //We will create a Eigen::VectorXd 
        
        std::vector<double> currClass;
        //We first iterate the current class and convert each matrix element into a point
        for (int j = 0; j < classes.at(i).rows(); j++) {
            Eigen::VectorXd currPoint = classes.at(i).row(j); //We assign the point

            


            currClass.push_back(Computations::Distances::euclidean(point, currPoint));
        }
        //we sort the points before adding them
        std::sort(currClass.begin(), currClass.end());
        //we add the class to the vector
        distances.push_back(currClass);
    }


    //Now that are points are stored and sorted we will begin to check which are the k nearest neighbours, to do this we will first, from each class we will
    //only keep the k first elements because in the worst case the k nearest neighbours are from only one class

    for (int i = 0; i < distances.size(); i++) {
        if (distances.at(i).size() > k) {
            distances.at(i).erase(distances.at(i).begin() + k, distances.at(i).end());
        }
    }
    //Now that each class has their k nearest neighbours we will see which ones are the k closest in general, to do this we will iterate each class for their first
    //element and when we find the one that is the smallest, we will add it to our solution vector and remove that element from the class vector, and repeat this
    //process until the solution vector is of size k

    std::vector<int> solution;

    while (solution.size() < k) {
        //We iterate the classes
        int mindist = 0; //We assume that the first class is the smallest distance and iterate from there
        for (int i = 1; i < distances.size(); i++) {
            if (distances.at(i).at(0) < distances.at(mindist).at(0)) {
                mindist = i;
            }

        }

        //Now we will add that class to the solution and remove that element
        solution.push_back(mindist);
        distances.at(mindist).erase(distances.at(mindist).begin());

    }

    //Now we have the classes of the k nearest neighbours, we will now see what class dominates and make a prediction

    //We will first iterate the classes and check the ocurrence the one with the most occurence will be our final result

    //We will also first assume that the class 0 is the initial prediction and adjust from there
    int res = 0;
    int count = std::count(solution.begin(), solution.end(), res);
    for (int i = 1; i < classes.size(); i++) {
        int currCount = std::count(solution.begin(), solution.end(), i);
        if (currCount > count) {
            res = i;
            count = currCount;
        }
    }
    //The result is the classification prediction
    return res;




}


















































Perceptron::Perceptron(const Eigen::MatrixXd& X_input)
{
    //inputMean = X_input.colwise().mean();
    //Eigen::RowVector3d X_mean_row = inputMean.transpose();
    //inputStd = Eigen::Vector3d::Zero();
    //for (int i = 0; i < X_input.rows(); ++i) {
        //inputStd += (X_input.row(i) - X_mean_row).array().square().matrix();
    //}
    //inputStd = (inputStd / X_input.rows()).array().sqrt();
    qDebug() << "Iterating inputData:";
    for (int i = 0; i < X_input.rows(); ++i) {
        qDebug() << i << " [";
        for (int j = 0; j < X_input.cols(); ++j) {
            qDebug() << " " << X_input(i, j);
        }
        qDebug() << "].\n";
    }
    labels = Eigen::VectorXd::Zero(X_input.rows());
    inputData = init_data(X_input);
    weights = init_weights(X_input.cols());
}

Perceptron::Perceptron(const std::vector<Eigen::MatrixXd> classes, int class_to_clasiffy)
{
    inputData = init_data_classes(classes);
    labels = init_labels_classes(classes, class_to_clasiffy);
    weights = init_weights(inputData.cols());
}

void Perceptron::showInfo()
{
    qDebug() << "Perceptron info\n";

    qDebug() << "Iterating inputData:";
    for (int i = 0; i < inputData.rows(); ++i) {
        qDebug() << i << " [";
        for (int j = 0; j < inputData.cols(); ++j) {
            qDebug() << " " << inputData(i, j);
        }
        qDebug() << "].\n";
    }

    qDebug() << "Iterating inputMean:";
    for (int i = 0; i < inputMean.rows(); ++i) {
        qDebug() << i << " [";
        for (int j = 0; j < inputMean.cols(); ++j) {
            qDebug() << " " << inputMean(i, j);
        }
        qDebug() << "].\n";
    }
    qDebug() << "Iterating weights:";
    for (int i = 0; i < weights.size(); ++i) {
        qDebug() << "weights[" << i << "] = " << weights[i];
    }
    qDebug() << "Iterating labels:";
    for (int i = 0; i < labels.size(); ++i) {
        qDebug() << "labels[" << i << "] = " << labels[i];
    }
    qDebug() << "min val and max val for each column in the data";
    for (int c = 0; c < minVal.size(); c++)
    {
        qDebug() << "column " << c << ", min " << minVal(c) << "max" << maxVal(c);
    }
}


Eigen::MatrixXd Perceptron::init_data(const Eigen::MatrixXd inputData)
{
    minVal = Eigen::VectorXd(inputData.cols());
    maxVal = Eigen::VectorXd(inputData.cols());
    for (int v = 0; v < inputData.cols(); v++)
    {
        minVal(v) = inputData.col(v).minCoeff();
        maxVal(v) = inputData.col(v).maxCoeff();
    }
    Eigen::MatrixXd X_input_copy = inputData;
    for (int i = 0; i < X_input_copy.cols(); i++) { Computations::Helper::normalizeColumn(X_input_copy, i); }
    Eigen::VectorXd ones_column = Eigen::VectorXd::Ones(X_input_copy.rows());
    Eigen::MatrixXd X_with_ones(X_input_copy.rows(), X_input_copy.cols() + 1);
    X_with_ones << X_input_copy, ones_column;
    return X_with_ones;
}

Eigen::MatrixXd Perceptron::init_data_classes(const std::vector<Eigen::MatrixXd> classes)
{
    int totalRows = 0;
    int numCols = 0;
    qDebug() << "data recibed ";
    for (const auto& matrix : classes) {
        qDebug() << "rows " << matrix.rows() << " cols " << matrix.cols();
        totalRows += matrix.rows();
        numCols = matrix.cols();
    }
    Eigen::MatrixXd X_input_copy(totalRows, numCols);

    qDebug() << "Nums of final rows " << totalRows << "num of cols " << numCols;

    int currentRow = 0;
    for (const auto& matrix : classes) {
        X_input_copy.block(currentRow, 0, matrix.rows(), matrix.cols()) = matrix;
        currentRow += matrix.rows();
    }
    minVal = Eigen::VectorXd(X_input_copy.cols());
    maxVal = Eigen::VectorXd(X_input_copy.cols());
    for (int v = 0; v < X_input_copy.cols(); v++)
    {
        minVal(v) = X_input_copy.col(v).minCoeff();
        maxVal(v) = X_input_copy.col(v).maxCoeff();
    }
    qDebug() << "Iterating inputData:";
    for (int i = 0; i < X_input_copy.rows(); ++i) {
        qDebug() << i << " [";
        for (int j = 0; j < X_input_copy.cols(); ++j) {
            qDebug() << " " << X_input_copy(i, j);
        }
        qDebug() << "].\n";
    }
    for (int i = 0; i < X_input_copy.cols(); i++) { Computations::Helper::normalizeColumn(X_input_copy, i); }
    Eigen::VectorXd ones_column = Eigen::VectorXd::Ones(X_input_copy.rows());
    Eigen::MatrixXd X_with_ones(X_input_copy.rows(), X_input_copy.cols() + 1);
    X_with_ones << X_input_copy, ones_column;
    return X_with_ones;
}

Eigen::VectorXd Perceptron::init_labels_classes(const std::vector<Eigen::MatrixXd> classes, int class_to_clasiffy)
{
    Eigen::VectorXd finalVector;
    for (int i = 0; i < classes.size(); i++)
    {
        Eigen::VectorXd new_vector;
        if (i == class_to_clasiffy)
        {
            new_vector = Eigen::VectorXd::Ones(classes[i].rows());
        }
        else
        {
            new_vector = Eigen::VectorXd::Zero(classes[i].rows());
        }
        int current_size = finalVector.size();
        finalVector.conservativeResize(current_size + classes[i].rows());

        finalVector.segment(current_size, classes[i].rows()) = new_vector;
    }
    return finalVector;
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
Eigen::VectorXd Perceptron::sigmoid_prime(const Eigen::VectorXd& h)
{
    return sigmoid(h).array() * (1 - sigmoid(h).array());
}
Eigen::VectorXd Perceptron::get_activation(const Eigen::VectorXd& h)
{
    return sigmoid(h);
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
    return error(Target).array() * sigmoid_prime(get_h(inputData)).array();
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
Eigen::VectorXd Perceptron::query(const Eigen::VectorXd& x)
{
    qDebug() << "Query recibed ";
    for (int i = 0; i < x.size(); i++)
    {
        qDebug() << x(i) << " ";
    }

    Eigen::VectorXd query(x.size() + 1);
    for (int i = 0; i < x.size(); i++)
    {
        if (maxVal(i) == minVal(i))
        {
            qDebug() << "Warning: maxVal and minVal are equal for column" << i;
            query(i) = 0;
        }
        else {
            query(i) = (x(i) - minVal(i)) / (maxVal(i) - minVal(i));
        }
    }
    query(x.size()) = 1.0;

    qDebug() << "Query adjust ";
    for (int i = 0; i < query.size(); i++)
    {
        qDebug() << query(i) << " ";
    }
    double dot = query.dot(weights);
    Eigen::VectorXd result(1);
    result(0) = dot;
    Eigen::VectorXd resultActivation = sigmoid(result);

    qDebug() << " final " << resultActivation(0);
    return result;
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


std::pair<std::vector<Eigen::MatrixXd>, Eigen::MatrixXd> MachineLearning::Kmeans(const Eigen::MatrixXd data, int k, float threshold)
{
    srand(0);
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

void MachineLearning::initializeWeights(std::vector<Eigen::MatrixXd>& weights)
{

    //We iterate the networks layers

    for (auto layer : weights) {
        layer.setRandom(); //We set the layers weights to random values
    }




}
//We pass the data through the network and calculate the loss
Eigen::VectorXd MachineLearning::forwardPass(Eigen::MatrixXd data, std::vector<Eigen::MatrixXd> &weights,std::function<double(Eigen::VectorXd)> l, std::vector<std::function<double(double)>> f, Eigen::VectorXd target)
{

    //Variable that will represent how the output at each timestep
    Eigen::MatrixXd z = data;
    //Variable to express the current layer's index, helps for the activation functions and for more specific errors
    int currLayer = 0;
    //Variable to store the preactivation values (Will be useful when doing backpropagation), will be a std::vector, where each element of the vector represents the input before applying
    //the activation function
    std::vector<Eigen::MatrixXd> preActivationValues;
    //We will first iterate the layers of the network to compute the forward pass
    for (Eigen::MatrixXd layer : weights) {

        //Before computing the result of the current layer we will validate the dimensions of the input and the layers

        if (data.cols() != layer.rows()) {
            throw std::runtime_error("Invalid dimensions in " + std::to_string(currLayer) + "Layer");
        }
        
        z = z * layer.transpose(); //Missing the bias vector, will add later

        //Before applying the activation function we will store the preactivation values, these will be useful when we need to do backpropagation 
        preActivationValues.push_back(z);


        //we will pass the result through the activation function     
        z = z.unaryExpr(f[currLayer]);



    }

    //We check that the final output is of the correct form, i.e. a NX1 vector representing the predictions 
    

    if (z.cols() > 1) {
        throw std::runtime_error("Invalid output shape, it should be a Nx1 vector, representing the models predictions");
    }

    //Now that the output has been validated we will calculate the loss function 

    Eigen::VectorXd y_pred = z.row(0);


    //We will now apply the loss function to every data point, and calculate the losses gradient
    
    Eigen::VectorXd gradientVector(y_pred.size());

    //we iterate the predicted data points to generate the loss
    
    for (int i = 0; i < y_pred.size(); i++) {
        //We will first store the y_pred and target(y_real) into an Eigen::VectorXd, to aproximate the derivative of the loss function in that point
        Eigen::VectorXd lossVec(2);
        lossVec << y_pred(i), target(i);
        
        //Now we will compute the derivative of the loss function 

        double lossDerivative = Computations::Math::aproximateDerivative(lossVec,l); 
        
        //Once we compute the loss derivative we will store it in a Eigen::VectorXd and return it once all the losses have been processed

        gradientVector(i) = lossDerivative;

        
        


    
    
    
    }





    return gradientVector;




    
    

}


//Function that computes the backwardPass(backPropagation)
Eigen::VectorXd MachineLearning::backwardPass(Eigen::VectorXd gradientVector,std::vector<Eigen::MatrixXd> &weights, std::vector<Eigen::MatrixXd> preActivationValues, std::vector<std::function<double(double)>> f, double alpha)
{
    //We will calculate the error for each layer, for that we need to compute the Derivative of the activation function with respect to each layers input before passing it through the'
    //activation function

    //Variable that stores the loss of each layer
    std::vector<Eigen::MatrixXd> lossVec;

    //We will calculate the activations function derivative to every element in the preActivationValues matrix at the layer l, to do this we will first iterate each of the layers, but
    //we will iterate them backwards since we have to backpropagate from the final layer to the beggining of the network


    //We will initialize the Error variable as it will be changed each iteration

    Eigen::MatrixXd Error;


    for (int i = preActivationValues.size() - 1; i >= 0; i--) {

        //As mentioned we compute the derivative of the activation value
        Eigen::MatrixXd activationFunctionDerivative(preActivationValues[i].rows(), preActivationValues[i].cols());

        //We will now fill the matrix with the derivatives

        for (int row = 0; row < activationFunctionDerivative.rows(); row++) {
            for (int col = 0; col < activationFunctionDerivative.cols(); col++) {
                activationFunctionDerivative(row, col) = Computations::Math::aproximateDerivative(preActivationValues[i](row,col),f[i]);
            }
        }
        

        //We will check if it's the final layer, if it's not the loss vector must be computed with the previous layers loss, thus enabling backpropagation

        if (i != preActivationValues.size() - 1) {

            //In orde r to calculate the loss when it's not the final layer, we must apply the hadamard product (element wise multiplication)
            // of the weights of the last layer multiplied by the loss of the last layer
            // with the derivative of the activation function of this layer
            // by using the weights and loss of the last layer we are carrying the total loss and so backpropagating the error so that the network can learn,
            // by also using this layers activation function derivative, we are calculating where we need to update the weights to get to a local minimum

            //Now that that has been explained, we will first calculate the multiplication of the previous layers weights and error, given that we checked that this is not the last layer,
            //there is a previous layer

            gradientVector = weights[i + 1].transpose() * error;




        }



        //Before we preform the hadamard product (element wise multiplicatio) to calculate the loss, we must check that the dimensions of the loss vector and the derivated activation
        //function matrix match, in this case since the loss is a vector it obviously does not match, therefor we will directly broadcast the vector so that it matches

        Eigen::MatrixXd broadCastedLossVec = gradientVector.replicate(1, activationFunctionDerivative.cols());

        //Now that  the dimensions are correct we will apply the hadamard product 

        error = broadCastedLossVec.array() * activationFunctionDerivative.array();

        //Using the error that we just calculated, we will now calculate the gradient of the loss, with respect to the Weights of the last layer

        //We compute the activation output from the previous layer
        //We obtain the activation output from the previous layer, that is the preactivation input of the current layer

        Eigen::MatrixXd activationValueFromPreviousLayer = preActivationValues[i];

        //Now we compute the gradient of the loss with respect to the weights, with the error and activation values calculated

        Eigen::MatrixXd weightLossGradient = error * activationValueFromPreviousLayer.transpose();

        //Now with that result we will update the weights with the gradient Descent function 

        weights[i] = weights[i] - (alpha * weightLossGradient);





    }



    




}

