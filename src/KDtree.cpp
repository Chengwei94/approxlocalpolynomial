#include <algorithm>    
#include <memory> 
#include <vector>
#include <functional> 
#include <Eigen/Dense>
#include "KDtree.h"
#include <utility>

// Things left to do:
//1. Change from passing by value to passing by constant reference to those available
//2. Change from matrix to vector to matrix. 
//3. Look at possible implementation without changing to vector (Bottleneck sorting in matrix form) 
//4. R implementation
//5. Print tree function
//6. Change name of some functions to avoid ambiguity

// [[Rcpp::plugins(cpp14)]]
// [[Rcpp::depends(RcppEigen)]]

kdtree::kdtree() = default; 
kdtree::~kdtree() = default; 

kdnode::kdnode() = default;  // constructor 
kdnode::kdnode(kdnode&& rhs) = default;  //move 
kdnode::~kdnode() = default;  // destructor  

template <typename T, typename U>                            // overloading operator for addition betweeen pairs
std::pair<T, U> operator+(const std::pair<T,U> & l, const std::pair<T,U> & r ) {
    return {l.first+r.first, l.second + r.second}; 
} 

inline double eval_kernel(int kcode, double z)
{
    double tmp;
    switch(kcode)
    {
    case 1: return 3*(1-z*z)/4; // Epanechnikov
    case 2: return 0.5; // rectangular
    case 3: return 1-abs(z); // triangular
    case 4: return 15*(1-z*z)*(1-z*z)/16; // quartic
    case 5: 
        tmp = 1-z*z;
        return 35*tmp*tmp*tmp/32; // triweight
    case 6: 
        tmp = 1- z*z*z;
        return 70 * tmp * tmp * tmp / 81; // tricube
    case 7:
        return M_PI * cos(M_PI*z/2) / 4; // cosine
    case 21:
        return exp(-z*z/2) / sqrt(2*M_PI); // gauss
    case 22:
        return 1/(exp(z)+2+exp(-z)); // logistic
    case 23:
        return 2/(M_PI*(exp(z)+exp(-z))); // sigmoid
    case 24:
        return exp(-abs(z)/sqrt(2)) * sin(abs(z)/sqrt(2)+M_PI/4); // silverman
 //   default: Rcpp::stop("Unsupported kernel"); 
    }
    return 0;
}

std::pair<double,double> calculate_weight(int kcode, Eigen::VectorXd query_pt, std::vector<double> max_dim, std::vector<double> min_dim) { 
    if(max_dim.size()!= query_pt.size()-1){
  //      std::cout << max_dim.size() << query_pt.size(); 
        std::cout << "error in dimensions"; 
        throw std::exception(); 
    }
    double max_weight = 0;
    double min_weight = 0; 
    double max_dist = 0; 
    double min_dist = 0;
    for(int i=0; i < max_dim.size(); i++)
        { 
            if (query_pt(i) <= max_dim.at(i) && query_pt(i) >= min_dim.at(i)) {
                max_dist += std::pow(std::max(abs(max_dim.at(i) - query_pt(i)), abs(min_dim.at(i) - query_pt(i))),2);
            }
            else if (abs(query_pt(i) - max_dim.at(i)) > abs(query_pt(i) - min_dim.at(i))){
                max_dist += std::pow(query_pt(i) - max_dim.at(i),2);
                min_dist += std::pow(query_pt(i) - min_dim.at(i),2); 
            }
            else{
                max_dist += std::pow(query_pt(i) - min_dim.at(i),2); 
                min_dist += std::pow(query_pt(i) - max_dim.at(i),2); 
            }
        }
    max_weight =  eval_kernel(kcode, sqrt(max_dist));  //sqrt of dist to get euclidean distance
    min_weight = eval_kernel(kcode, sqrt(min_dist));  
    return std::make_pair(max_weight, min_weight); 
}

std::unique_ptr<kdnode> kdtree::build_tree(all_point_t::iterator start, all_point_t::iterator end, 
                                           int split_d, double split_v, int N_min, size_t len,
                                            std::vector<double> max_dim_, std::vector<double> min_dim_)
{

    std::unique_ptr<kdnode> newnode = std::make_unique<kdnode>();
    if(end == start) {
 //       std::cout <<"empty node created" << std::endl;
        return newnode; 
    }

    newnode->n_below = len;    
    newnode->max_dim = max_dim_;
    newnode->min_dim = min_dim_;

    if (end-start <= N_min) {
        newnode->left_child = nullptr;              // leaf 
        newnode->right_child = nullptr;             // leaf      
        Eigen::MatrixXd XtX_; 
        Eigen::VectorXd XtY_;                   
        for (auto i = start; i != end; i ++){
            Eigen::VectorXd XY = *i;    
            Eigen::VectorXd Y = XY.tail<1>(); 
            Eigen::VectorXd X = XY.head(XY.size()-1); 
            XtY_ = X*Y; 
            XtX_ = X*X.transpose();  
        }   
         newnode->XtX = XtX_; 
         newnode->XtY = XtY_;  
  //       std::cout << "XtX" << newnode->XtX << "\n"; 
   //      std::cout << "XtY" << newnode->XtY << "\n"; 
    //     std::cout << "leaf node created" << std::endl; 
         return newnode; 
    }

    else {  
   //     std::cout << "node created start" << std::endl; 
        size_t l_len = len/2  ;          // left length
        size_t r_len = len - l_len;     // right length
        auto middle = start + len/2;   // middle iterator 
        int max = 0; 
        int dim = 0;
        for(int i = 0; i < newnode->max_dim.size(); i++){   
            double variance = newnode->max_dim[i] - newnode->min_dim[i]; 
            if(variance > max){
                max = variance; 
                dim = i; 
            }
        }
        newnode -> split_d = dim; 

        std::nth_element(start, middle, end, [dim](const Eigen::VectorXd& a, const Eigen::VectorXd & b) {return 
        a(dim) < b(dim);    
        });           

        newnode->split_v = (*middle)[newnode->split_d];   

        max_dim_[newnode->split_d] = newnode->split_v; 
        min_dim_[newnode->split_d] = newnode->split_v;

    //    std::cout << "node created end" << std::endl; 

        newnode-> left_child = build_tree(start, middle, newnode->split_d, newnode->split_v, N_min, l_len, max_dim_, min_dim_);
        newnode-> right_child = build_tree(middle, end, newnode->split_d, newnode->split_v, N_min, r_len, max_dim_, min_dim_);
        
        if ((newnode->left_child) && (newnode->right_child)){ 
            newnode->XtX = newnode->left_child->XtX + newnode ->right_child->XtX;  // sumY = the sum of the bottom 2 nodes
            newnode->XtY = newnode->left_child->XtY + newnode ->right_child->XtY;
        } 
        else if (newnode->left_child) {
            newnode->XtY = newnode->left_child->XtY;
            newnode->XtX = newnode->left_child->XtX;
        }
        else if (newnode->right_child) {
            newnode->XtX = newnode->right_child->XtX; 
            newnode->XtY = newnode->right_child->XtY; 
        }   
    }
  //  std::cout << "no error found" << std::endl;
    return newnode;    
}

kdtree::kdtree(all_point_t points, int N_min) { 
    size_t len = points.size(); 

    std::vector<double> max_dim; 
    std::vector<double> min_dim; 

    for(int i=0; i<points[0].size()-1; i++){
        max_dim.push_back(points[0](i)); 
        min_dim.push_back(points[0](i));
    }

    for(int i=0; i<points[0].size()-1; i++) { // loop size of first point to find dimension; 
        for (int j=0; j<points.size(); j++){
            if (points[j](i) > max_dim.at(i)){
                max_dim.at(i) = points[j](i);
            }
            if (points[j][i] < min_dim.at(i)){ 
                min_dim.at(i) = points[j](i);
            }
        }
    } 
    for (int i =0; i< points[0].size()-1; i++) { 
        std::cout << max_dim[i]; 
        std::cout << min_dim[i] << std::endl; 
    }

   //  std::cout << "kdtree initialized "<< std::endl; 
     root = build_tree(points.begin(), points.end(), 1, 1, N_min, len, max_dim, min_dim);
    
}

std::pair<Eigen::MatrixXd, Eigen::VectorXd> kdtree::get_XtXXtY(Eigen::VectorXd query_pt, std::vector<double> max_dim, std::vector<double> min_dim, std::unique_ptr<kdnode>& root){
    std::pair<double,double> weights;
    std :: cout << " weights calcualtion" << std::endl; 
    weights = calculate_weight(20, query_pt, max_dim, min_dim);  // calculate max and min weight 
    std :: cout << " weights calculation success " << std::endl; 

    double max_weight = weights.first;                      
    double min_weight = weights.second;

    if (max_weight = min_weight)       // if condition fufilled          
        return std::make_pair(max_weight*root->XtX, max_weight*root->XtY);
    else if (root->n_below <= 1)    // finding out that it is a leaf node  (Add N_min for later) 
        return std::make_pair(root->XtX, root->XtY); 
    else { 
        return get_XtXXtY(query_pt, root->left_child->max_dim, root->left_child->min_dim, root->left_child ) + 
        get_XtXXtY(query_pt, root->right_child->max_dim, root->right_child->min_dim, root->right_child); 
    }
}   // return sumY, else return sumY of both nodes if weight satisfy condition 

std::pair<Eigen::MatrixXd, Eigen::VectorXd> kdtree::getapprox_XtXXtY(Eigen::VectorXd query_pt,
                                                                     std::vector<double> max_dim,
                                                                     std::vector<double> min_dim, 
                                                                     std::unique_ptr<kdnode>& root,
                                                                     double epsilon, 
                                                                     double weight_sf){ //weights_so_far
    std::pair<double,double> weights;
    std :: cout << " weights calcualtion" << std::endl; 
    weights = calculate_weight(20, query_pt, max_dim, min_dim);  // calculate max and min weight 
    std :: cout << " weights calculation success " << std::endl; 

    double max_weight = weights.first;                      
    double min_weight = weights.second;

    if (max_weight - min_weight <= 2*epsilon*(weight_sf + root->n_below*min_weight)) {
        double weight = 0.5*(max_weight + min_weight);
        weight_sf += weight;                     // if condition fufilled          
        return std::make_pair(weight*root->XtX, weight*root->XtY);
    }
    else if (root->n_below <= 1) {   // finding out that it is a leaf node  (Add N_min for later) 
        double weight = 0.5*(max_weight + min_weight);
        weight_sf += weight;           //smth wrong here need to edit later
        return std::make_pair(weight*root->XtX, weight*root->XtY); 
    }
    else { 
        return getapprox_XtXXtY(query_pt, root->left_child->max_dim, root->left_child->min_dim, root->left_child, epsilon, weight_sf ) + 
        getapprox_XtXXtY(query_pt, root->right_child->max_dim, root->right_child->min_dim, root->right_child, epsilon, weight_sf); 
    }
}   // return sumY, else return sumY of both nodes if weight satisfy condition 

std::pair<Eigen::MatrixXd, Eigen::VectorXd> kdtree::find_XtXXtY(Eigen::VectorXd query_pt, int method = 1, double epsilon = 0.05) { // add approximate
    if (method == 1) {
        return get_XtXXtY(query_pt, root->max_dim, root->min_dim, root);
    }
    else {
        return getapprox_XtXXtY(query_pt, root->max_dim, root->min_dim, root, epsilon, 0); 
    }
}

Eigen::VectorXd solve_beta(int kcode, const Eigen::MatrixXd &XtX, const Eigen::MatrixXd &XtY) {
    //    std::cout << "normal equation method used(chloesky)" << "\n"; 
        return(XtX.ldlt().solve(XtY));
}
// [[Rcpp::export]]
Eigen::VectorXd locpoly(all_point_t points, int kcode, int N_min){ 
    Eigen::VectorXd b = Eigen::VectorXd::Random(3); 
    kdtree tree(points, N_min);
    std::cout <<"done" <<"\n";
    std::pair<Eigen::MatrixXd, Eigen::VectorXd> XtXXtY; 
    XtXXtY = tree.find_XtXXtY(b);  // add N_min for later 
    return solve_beta(kcode, XtXXtY.first, XtXXtY.second);  
}

all_point_t convert_to_vector(Eigen::MatrixXd original_points){
    std::vector<Eigen::VectorXd> points; 
    for (int i = 0; i <original_points.rows(); i++){
        points.push_back(original_points.row(i));
    }
    return points; 
}



/***R
x = as.matrix(c(1,2))   
rcpp_eigentryout(x)
*/
int main(){ 
    Eigen::VectorXd c = Eigen::VectorXd::Random(10);
    Eigen::MatrixXd b = Eigen::MatrixXd::Random(5000,10); 
    all_point_t z = convert_to_vector(b); 
    kdtree tree(z,1); 
    std::cout <<"done" <<"\n";
    std::pair<Eigen::MatrixXd, Eigen::VectorXd> XtXXtY; 
    XtXXtY = tree.find_XtXXtY(c,2); 
    Eigen::VectorXd f; 
    f = solve_beta(1, XtXXtY.first, XtXXtY.second);
    std::cout << f; 

     
  //  std::cout << A << "\n";
   // std::cout << b << "\n";
   // std::pair<Eigen::MatrixXd, Eigen::VectorXd> x = tryout(A,b); 
   // std:: cout << x.first << '\n'; 
   // std:: cout << x.second << "\n";

   // std::cout << calculate_inverse(1,A,b); 
}



