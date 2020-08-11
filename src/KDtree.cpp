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

template <typename T, typename U>                                               // overloading operator for addition 
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

std::pair<double,double> calculate_weight(int kcode, point_t query_pt, point_t max_dim, point_t min_dim) { 
    double max_weight = 0;
    double min_weight = 0; 
    double max_dist = 0; 
    double min_dist = 0;
    std::cout << max_dim.size(); 
    for(int i=0; i < max_dim.size(); i++)
        { 
            if (query_pt.at(i) <= max_dim.at(i) && query_pt.at(i) >= min_dim.at(i)) {
                max_dist += std::pow(std::max(abs(max_dim.at(i) - query_pt.at(i)), abs(min_dim.at(i) - query_pt.at(i))),2);
            }
            else if (abs(query_pt.at(i) - max_dim.at(i)) > abs(query_pt.at(i) - min_dim.at(i))){
                max_dist += std::pow(query_pt.at(i) - max_dim.at(i),2);
                min_dist += std::pow(query_pt.at(i) - min_dim.at(i),2); 
            }
            else{
                max_dist += std::pow(query_pt.at(i) - min_dim.at(i),2); 
                min_dist += std::pow(query_pt.at(i) - max_dim.at(i),2); 
            }
        }
    max_weight =  eval_kernel(kcode, sqrt(max_dist));  //sqrt of dist to get euclidean distance
    min_weight = eval_kernel(kcode, sqrt(min_dist));  
    return std::make_pair(max_weight, min_weight); 
}


std::unique_ptr<kdnode> kdtree::build_tree(all_point_t::iterator start, all_point_t::iterator end, 
                                           int split_d, double split_v, int N_min, size_t len,
                                            point_t max_dim_, point_t min_dim_)
{

    std::unique_ptr<kdnode> newnode = std::make_unique<kdnode>();
    if(end == start) {
        std::cout <<"empty node created" << std::endl;
        return newnode; 
    }

    newnode->n_below = len;    
    newnode->max_dim = max_dim_;
    newnode->min_dim = min_dim_;
    std::cout << "max_dim_" <<  newnode->max_dim.size() << std::endl;

    if (end-start <= N_min) {
        newnode->left_child = nullptr;              // leaf 
        newnode->right_child = nullptr;             // leaf      
        Eigen::MatrixXd XtX_; 
        Eigen::VectorXd XtY_;                   
        for (auto i = start; i != end; i ++){
            std::vector<double> z = *i;  
            Eigen::VectorXd XY = Eigen::Map<Eigen::VectorXd, Eigen::Unaligned>(z.data(),z.size());  
            Eigen::VectorXd Y = XY.tail<1>(); 
            Eigen::VectorXd X = XY.head(z.size()-1); 
            XtY_ = X*Y; 
            XtX_ = X*X.transpose();  
        }   
         newnode->XtX = XtX_; 
         newnode->XtY = XtY_;  
         std::cout << "leaf node created" << std::endl; 
         return newnode; 
    }

    else {  
        std::cout << "node created start" << std::endl; 
        size_t l_len = len/2  ;          // left length
        size_t r_len = len - l_len;     // right length
        auto middle = start + len/2;   // middle iterator 
        std::cout << "l_len" << l_len << "\n"; 
        std::cout << "r_len" << r_len <<"\n";
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
       
        std::cout <<dim << std::endl; 
     //   std::cout<<"max_dim" << max << " =" << dim << std::endl;
        std::nth_element(start, middle, end, [dim](const std::vector<double>& a, const std::vector<double>& b) {return 
        a[dim] < b[dim];    
        });           

     //   std::cout <<"normal node created2" << std::endl; 
        newnode->split_v = (*middle)[newnode->split_d];   
      //  std::cout << "split_v = " << newnode->split_v << "on dimension " << newnode->split_d << "\n";         
        max_dim_[newnode->split_d] = newnode->split_v; 
        min_dim_[newnode->split_d] = newnode->split_v;

        std::cout << "node created end" << std::endl; 

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
    return newnode;    
}

kdtree::kdtree(all_point_t points, int N_min) { 
    size_t len = points.size(); 
 
//  auto middle = points.begin() + len/2; 
//   std::nth_element(points.begin(), middle , points.end(), [](const std::vector<double>& a, const std::vector<double>& b) {return 
//        a[0] < b[0];    
//    });   // get the median value 

//    size_t l_len = len/2 ; 
//    size_t r_len = len - l_len;
 
//   std::unique_ptr<kdnode> tree_root = std::make_unique<kdnode> (); //create a new root node; 
    std::vector<double> max_dim; 
    std::vector<double> min_dim; 

    for(int i=0; i<points[0].size()-1; i++) { // loop size of first point to find dimension; 
        max_dim.push_back(1); 
        min_dim.push_back(0);   
    } 


//   tree_root->n_below = len; 
//   tree_root->split_d = 0; 
//   tree_root->split_v = (*middle)[tree_root->split_d];     
//   root = std::move(tree_root);
    
  //  Rcpp::Rcout << "1st split =" << root-> split_v << std::endl;   

//    root-> left_child = build_tree(points.begin(), middle, root->split_d, root->split_v, N_min, l_len, root->max_dim, root->min_dim, 1); 
//    root-> right_child = build_tree(middle, points.end(), root->split_d, root->split_v, N_min, r_len, root->max_dim, root->min_dim, 2); 

 //   if ((root->left_child) && (root->right_child)){ 
 //   root->sumY = root->left_child ->sumY + root->right_child->sumY; */
     std::cout << "kdtree initialized "<< std::endl; 
     root = build_tree(points.begin(), points.end(), 1, 1, N_min, len, max_dim, min_dim);
    
}

std::pair<Eigen::MatrixXd, Eigen::VectorXd> kdtree::get_XtXXtY(point_t query_pt, point_t max_dim, point_t min_dim, std::unique_ptr<kdnode>& root){
    std::pair<double,double> weights;
    std :: cout << " weights calcualtion" << std::endl; 
    weights = calculate_weight(20, query_pt, max_dim, min_dim);  // calculate max and min weight 
    std :: cout << " weights calculation success " << std::endl; 

    double max_weight = weights.first;                      
    double min_weight = weights.second;

    if (max_weight = min_weight)       // if condition fufilled          
        return std::make_pair(root->XtX, root->XtY);
    else if (root->n_below <= 1)    // finding out that it is a leaf node  (Add N_min for later) 
        return std::make_pair(root->XtX, root->XtY); 
    else { 
        return get_XtXXtY(query_pt, root->left_child->max_dim, root->left_child->min_dim, root->left_child ) + 
        get_XtXXtY(query_pt, root->right_child->max_dim, root->right_child->min_dim, root->right_child); 
    }
}   // return sumY, else return sumY of both nodes if weight satisfy condition 

std::pair<Eigen::MatrixXd, Eigen::VectorXd> kdtree::getapprox_XtXXtY(point_t query_pt, 
                                                                     point_t max_dim,
                                                                     point_t min_dim, 
                                                                     std::unique_ptr<kdnode>& root,
                                                                     double epsilon, 
                                                                     double weight_sf){ //weights_so_far
    std::pair<double,double> weights;
    std :: cout << " weights calcualtion" << std::endl; 
    weights = calculate_weight(20, query_pt, max_dim, min_dim);  // calculate max and min weight 
    std :: cout << " weights calculation success " << std::endl; 

    double max_weight = weights.first;                      
    double min_weight = weights.second;

    if (max_weight - min_weight <= 2*epsilon*(weights_sf + root->n_below*min_weight)) {
        weights += 0.5*(max_weight + min_weight);                     // if condition fufilled          
        return std::make_pair(root->XtX, root->XtY);
    }
    else if (root->n_below <= 1) {   // finding out that it is a leaf node  (Add N_min for later) 
        weights += 0.5*(max_weight + min_weight);           //smth wrong here need to edit later
        return std::make_pair(root->XtX, root->XtY); 
    }
    else { 
        return get_approxXtXXtY(query_pt, root->left_child->max_dim, root->left_child->min_dim, root->left_child ) + 
        get_XtXXtY(query_pt, root->right_child->max_dim, root->right_child->min_dim, root->right_child); 
    }
}   // return sumY, else return sumY of both nodes if weight satisfy condition 


std::pair<Eigen::MatrixXd, Eigen::VectorXd> kdtree::find_XtXXtY(point_t query_pt) { // add approximate
    return get_XtXXtY(query_pt, root->max_dim, root->min_dim, root);
}

Eigen::VectorXd solve_beta(int kcode, const Eigen::MatrixXd &XtX, const Eigen::MatrixXd &XtY) {
        std::cout << "normal equation method used(chloesky)" << "\n"; 
        return(XtX.ldlt().solve(XtY));
}
// [[Rcpp::export]]
Eigen::VectorXd locpoly(all_point_t points, int kcode, int N_min){ 
    kdtree tree(points, N_min);
    std::pair<Eigen::MatrixXd, Eigen::VectorXd> XtXXtY; 
    XtXXtY = tree.find_XtXXtY({1,2,3});  // add N_min for later 
    return solve_beta(kcode, XtXXtY.first, XtXXtY.second);  
}

/***R
x = as.matrix(c(1,2))   
rcpp_eigentryout(x)
*/
int main(){ 
    all_point_t z = {{1,2,3,1,2},{2,3,4,2,2},{2,2,3,1,1},{4,3,4,1,1},{1,5,3,1,1},{2,3,8,2,1}}; 
    kdtree tree(z,1); 
    std::pair<Eigen::MatrixXd, Eigen::VectorXd> XtXXtY; 
    XtXXtY = tree.find_XtXXtY({2,3,4,1,2}); 
    solve_beta(1, XtXXtY.first, XtXXtY.second);

     Eigen::MatrixXd A = Eigen::MatrixXd::Random(3, 3);
     Eigen::VectorXd b = Eigen::VectorXd::Random(3); 
  //  std::cout << A << "\n";
   // std::cout << b << "\n";
   // std::pair<Eigen::MatrixXd, Eigen::VectorXd> x = tryout(A,b); 
   // std:: cout << x.first << '\n'; 
   // std:: cout << x.second << "\n";

   // std::cout << calculate_inverse(1,A,b); 
}



