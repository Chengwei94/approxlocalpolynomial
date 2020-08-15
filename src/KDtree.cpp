#include <algorithm>    
#include <memory> 
#include <vector>
#include <functional> 
//#include <Eigen/Dense>
#include "RcppEigen.h"
#include "KDtree.h"
#include <utility>
#include <iostream>
#include <chrono>


// Things left to do:
//1. Change from passing by value to passing by constant reference to those available
//5. Shift test function to additional cpp file.

//6. Change name of some functions to avoid ambiguity

// [[Rcpp::plugins(cpp14)]]
// [[Rcpp::depends(RcppEigen)]]


Timer::Timer() : beg_(clock_::now()) {}
    
void Timer::reset() { beg_ = clock_::now(); }

double Timer::elapsed() const { 
        return std::chrono::duration_cast<second_>
            (clock_::now() - beg_).count(); }


kdtree::kdtree() = default; 
kdtree::~kdtree() = default; 

kdnode::kdnode() = default;  // constructor 
kdnode::kdnode(kdnode&& rhs) = default;  //move 
kdnode::~kdnode() = default;  // destructor  

template <typename T, typename U>                            // overloading operator for addition betweeen pairs
std::pair<T, U> operator+(const std::pair<T,U> & l, const std::pair<T,U> & r ) {
    return {l.first+r.first, l.second + r.second}; 
} 

template<typename T>            //overloading vector cout for printing purposes
std::ostream& operator<< (std::ostream& out, const std::vector<T>& v) {
    out << "{";
    size_t last = v.size() - 1;
    for(size_t i = 0; i < v.size(); ++i) {
        out << v[i];
        if (i != last) 
            out << ", ";
    }
    out << "}";
    return out;
}

std::unique_ptr<kdnode> kdtree::build_exacttree(all_point_t::iterator start, all_point_t::iterator end, 
                                                int split_d, double split_v, int N_min, size_t len,
                                                std::vector<double> max_dim_, std::vector<double> min_dim_){

    std::unique_ptr<kdnode> newnode = std::make_unique<kdnode>();
    if(end == start) {   
        return newnode; 
    }

    newnode->n_below = len;    
    newnode->max_dim = max_dim_;
    newnode->min_dim = min_dim_;
  //  std::cout << "max_dim =" << max_dim_ << "\n";
  //  std::cout << "min_dim =" << min_dim_ << "\n";

    if (end-start <= 1) {
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

            for (auto j = 0; j < max_dim_.size(); j++) {
                newnode->max_dim[j] = X(j+1); 
                newnode->min_dim[j] = X(j+1); 
            }
         }   
         newnode->XtX = XtX_; 
         newnode->XtY = XtY_;  
 //        std::cout << "XtX" << newnode->XtX << "\n"; 
//       std::cout << "XtY" << newnode->XtY << "\n"; 
    //     std::cout << "leaf node created" << std::endl; 
         return newnode; 
    }

    else {  
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
       // std::cout << "l_len=" <<  l_len <<"\n";
       // std::cout << "r_len=" <<  r_len <<"\n";

        newnode -> split_d = dim; 
        int vector_dim = dim + 1;  

        std::nth_element(start, middle, end, [vector_dim](const Eigen::VectorXd& a, const Eigen::VectorXd & b) {
            return a(vector_dim) < b(vector_dim);    
        });           

        newnode->split_v = (*middle)[vector_dim];    //  vector_dim used as vector has two more dimension than max_dim.

        //std::cout << "split_d =" << newnode->split_d <<std::endl; 
        //std::cout << "split_v =" << newnode->split_v <<std::endl;

        max_dim_[dim] = newnode->split_v; 
        min_dim_[dim] = newnode->split_v;

    //    std::cout << "node created end" << std::endl; 

        newnode-> left_child = build_exacttree(start, middle, newnode->split_d, newnode->split_v, N_min, l_len, max_dim_, newnode->min_dim);
        newnode-> right_child = build_exacttree(middle, end, newnode->split_d, newnode->split_v, N_min, r_len, newnode->max_dim, min_dim_);
        
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


std::unique_ptr<kdnode> kdtree::build_tree(all_point_t::iterator start, all_point_t::iterator end, 
                                           int split_d, double split_v, int N_min, size_t len,
                                           std::vector<double> max_dim_, std::vector<double> min_dim_){

   // std::cout <<" node created" << std::endl;             
    std::unique_ptr<kdnode> newnode = std::make_unique<kdnode>();
    if(end == start) {   
        return newnode; 
    }

    newnode->n_below = len;    
    newnode->max_dim = max_dim_;
    newnode->min_dim = min_dim_;
  //  std::cout << "max_dim =" << max_dim_ << "\n";
  //  std::cout << "min_dim =" << min_dim_ << "\n";

    if (end-start <= N_min) {
       // std::cout << "leaf node created" << std::endl;
        newnode->left_child = nullptr;              // leaf 
        newnode->right_child = nullptr;             // leaf      
        Eigen::MatrixXd XtX_(max_dim_.size() + 1 , max_dim_.size() + 1);
        XtX_.setZero(); 
        Eigen::VectorXd XtY_(max_dim_.size() + 1);
        XtY_.setZero();                   
        for (auto i = start; i != end; i ++){
            Eigen::VectorXd XY = *i;    
            Eigen::VectorXd Y = XY.tail<1>(); 
            Eigen::VectorXd X = XY.head(XY.size()-1); 
            XtY_ += X*Y; 
            XtX_ += X*X.transpose();  
         }   
         newnode->XtX = XtX_; 
         newnode->XtY = XtY_;  

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
       // std::cout << "l_len=" <<  l_len <<"\n";
       // std::cout << "r_len=" <<  r_len <<"\n";

        newnode -> split_d = dim; 
        int vector_dim = dim + 1;  

        std::nth_element(start, middle, end, [vector_dim](const Eigen::VectorXd& a, const Eigen::VectorXd & b) {
            return a(vector_dim) < b(vector_dim);    
        });           

        newnode->split_v = (*middle)[vector_dim];    //  vector_dim used as vector has two more dimension than max_dim.

        //std::cout << "split_d =" << newnode->split_d <<std::endl; 
        //std::cout << "split_v =" << newnode->split_v <<std::endl;

        max_dim_[dim] = newnode->split_v; 
        min_dim_[dim] = newnode->split_v;

    //    std::cout << "node created end" << std::endl; 

        newnode-> left_child = build_tree(start, middle, newnode->split_d, newnode->split_v, N_min, l_len, max_dim_, newnode->min_dim);
        newnode-> right_child = build_tree(middle, end, newnode->split_d, newnode->split_v, N_min, r_len, newnode->max_dim, min_dim_);
        
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

kdtree::kdtree(all_point_t points, int N_min , int method) { 
    size_t len = points.size(); 
    weight_sf = 0;
    std::vector<double> max_dim; 
    std::vector<double> min_dim; 

    for(int i=1; i<points[0].size()-1; i++){
        max_dim.push_back(points[0](i)); 
        min_dim.push_back(points[0](i));
    }

    for(int i=1; i<points[0].size()-1; i++) { // loop size of first point to find dimension; 
        for (int j=0; j<points.size(); j++){
            if (points[j](i) > max_dim.at(i-1)){
                max_dim.at(i-1) = points[j](i);
            }
            if (points[j][i] < min_dim.at(i-1)){ 
                min_dim.at(i-1) = points[j](i);
            }
        }
    } 
    if (method == 1) { 
        root = build_exacttree(points.begin(), points.end(), 1, 1, N_min, len, max_dim, min_dim);
    }

    if (method == 2) { 
    //  std::cout << "kdtree initialized "<< std::endl; 
        root = build_tree(points.begin(), points.end(), 1, 1, N_min, len, max_dim, min_dim);
    }
   
}

all_point_t convert_to_query(Eigen::MatrixXd original_points){           //  conversion to query form 
    std::vector<Eigen::VectorXd> points; 
    Eigen::MatrixXd X = original_points.transpose(); 
    for (int i=0; i<X.cols(); i++) {
        Eigen::VectorXd X_values; 
        X_values = X.block(0,i,X.rows()-1,1); 
    //    std::cout << X_values <<"\n";
    //    std::cout << "\n";
        points.push_back(X_values);
    }
    return points; 
}

all_point_t convert_to_vector(Eigen::MatrixXd original_points){         // conversion to vector form 
    std::vector<Eigen::VectorXd> points; 
    Eigen::MatrixXd design_matrix(original_points.rows(), original_points.cols()+1); 
    design_matrix.topRightCorner(original_points.rows(), original_points.cols()) = original_points; 
    design_matrix.col(0) = Eigen::VectorXd::Ones(original_points.rows()); 
//  std::cout<<original_points << "\n";
//  std::cout<<design_matrix << "Design Matrix" << "\n";
    for (int i = 0; i <design_matrix.rows(); i++){
        points.push_back(design_matrix.row(i));
    }
    return points; 
}

inline double eval_kernel(int kcode, double z, double bandwidth)
{
    double tmp;
    if(abs(z) > 1) 
        return 0;
    else { 
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
    }
    return 0;
}

std::pair<double,double> calculate_weight(int kcode, Eigen::VectorXd query_pt, std::vector<double> max_dim, 
                                          std::vector<double> min_dim, double bandwidth) { 
    if(max_dim.size()!= query_pt.size()){
  //      std::cout << max_dim.size() << query_pt.size(); 
        std::cout << "error in dimensions"; 
        throw std::exception(); 
    }
    double max_weight = 1;
    double min_weight = 1; 
    double max_dist = 0; 
    double min_dist = 0;
    for(int i=0; i < max_dim.size(); i++)
        { 
            if (query_pt(i) <= max_dim.at(i) && query_pt(i) >= min_dim.at(i)) {
                max_dist = std::max(abs(max_dim.at(i) - query_pt(i)), abs(min_dim.at(i) - query_pt(i)));    // max_dist = which ever dim is further
                min_dist = 0;                                                                              
                min_weight *= eval_kernel(kcode, max_dist/bandwidth, bandwidth) / bandwidth;    // kern weight = multiplication of weight of each dim 
                max_weight *= eval_kernel(kcode, min_dist/bandwidth, bandwidth) / bandwidth;
                
            }
            else if (abs(query_pt(i) - max_dim.at(i)) > abs(query_pt(i) - min_dim.at(i))){
                max_dist = query_pt(i) - max_dim.at(i);
                min_dist = query_pt(i) - min_dim.at(i);
                min_weight *= eval_kernel(kcode, max_dist/bandwidth, bandwidth) / bandwidth;
                max_weight *= eval_kernel(kcode, min_dist/bandwidth, bandwidth) / bandwidth; 
            }
            else{
                max_dist = query_pt(i) - min_dim.at(i);
                min_dist = query_pt(i) - max_dim.at(i); 
                min_weight *= eval_kernel(kcode, max_dist/bandwidth, bandwidth) / bandwidth; 
                max_weight *= eval_kernel(kcode, min_dist/bandwidth, bandwidth) / bandwidth; 
            }
        }
    return std::make_pair(max_weight, min_weight); 
}

std::pair<Eigen::MatrixXd, Eigen::VectorXd> kdtree::get_XtXXtY(Eigen::VectorXd query_pt, 
                                                               std::vector<double> max_dim, 
                                                               std::vector<double> min_dim, 
                                                               std::unique_ptr<kdnode>& root, 
                                                               double bandwidth, 
                                                               int kcode){
    std::pair<double,double> weights;

    weights = calculate_weight(kcode, query_pt, max_dim, min_dim, bandwidth);  // calculate max and min weight 

    double max_weight = weights.first;                      
    double min_weight = weights.second;

    if (max_weight == min_weight){   // if condition fufilled      
        return std::make_pair(max_weight*root->XtX, max_weight*root->XtY);
    }
    else if (root->n_below == 1){    // finding out that it is a leaf node  (Add N_min for later)  
        double weight = 0.5*(max_weight + min_weight);  
        return std::make_pair(weight*root->XtX, weight*root->XtY); 
    }
    else { 
        return get_XtXXtY(query_pt, root->left_child->max_dim, root->left_child->min_dim, root->left_child, bandwidth, kcode) + 
        get_XtXXtY(query_pt, root->right_child->max_dim, root->right_child->min_dim, root->right_child, bandwidth, kcode); 
    }
}   // return sumY, else return sumY of both nodes if weight satisfy condition 

std::pair<Eigen::MatrixXd, Eigen::VectorXd> kdtree::getapprox_XtXXtY(Eigen::VectorXd query_pt,
                                                                     std::vector<double> max_dim,
                                                                     std::vector<double> min_dim, 
                                                                     std::unique_ptr<kdnode>& root,
                                                                     double epsilon, 
                                                                     double bandwidth,
                                                                     int kcode){ 
    std::pair<double,double> weights;
  //  std :: cout << " weights calcualtion" << std::endl; 
    weights = calculate_weight(kcode, query_pt, max_dim, min_dim, bandwidth);  // calculate max and min weight 
  //  std :: cout << " weights calculation success " << std::endl; 

    double max_weight = weights.first;                      
    double min_weight = weights.second;

    if (max_weight - min_weight <= 2 * epsilon * (weight_sf + root->n_below*min_weight)) {   // if condition fulfilled
        double weight = 0.5*(max_weight + min_weight);
        weight_sf += weight;                                                                 // weight_so_far + current weights found    
        return std::make_pair(weight*root->XtX, weight*root->XtY);
    }
    else if (root->left_child == nullptr && root->right_child == nullptr) {   // finding out that it is a leaf node  (Add N_min for later) 
        double weight = 0.5*(max_weight + min_weight);
        weight_sf += weight;           
        return std::make_pair(weight*root->XtX, weight*root->XtY); 
    }
    else { 
        return getapprox_XtXXtY(query_pt, root->left_child->max_dim, root->left_child->min_dim, root->left_child, epsilon, bandwidth, kcode) + 
        getapprox_XtXXtY(query_pt, root->right_child->max_dim, root->right_child->min_dim, root->right_child, epsilon, bandwidth, kcode); 
    }
}   // return sumY, else return sumY of both nodes if weight satisfy condition 

std::pair<Eigen::MatrixXd, Eigen::VectorXd> kdtree::find_XtXXtY(Eigen::VectorXd query_pt, 
                                                                int method = 1, 
                                                                double epsilon = 0.05, 
                                                                double bandwidth = 0.2, 
                                                                int kcode = 1) { // add approximate
    if (method == 1) {
        std::pair<Eigen::MatrixXd, Eigen::VectorXd> XtXXtY = get_XtXXtY(query_pt, root->max_dim, root->min_dim, root, bandwidth, kcode);
        Eigen::MatrixXd ll_XtX = form_ll_XtX(XtXXtY.first, query_pt);
        Eigen::VectorXd ll_XtY = form_ll_XtY(XtXXtY.second, query_pt);
        return std::make_pair(ll_XtX , ll_XtY); 
    }
    else {
        weight_sf = 0; 
        std::pair<Eigen::MatrixXd, Eigen::VectorXd> XtXXtY = getapprox_XtXXtY(query_pt, root->max_dim, root->min_dim, root, epsilon, bandwidth, kcode); 
        Eigen::MatrixXd ll_XtX = form_ll_XtX(XtXXtY.first, query_pt); 
        Eigen::VectorXd ll_XtY = form_ll_XtY(XtXXtY.second, query_pt);
//      std::cout << ll_XtX << "\n"; 
//      std::cout << ll_XtY << "\n";
        return std::make_pair(ll_XtX , ll_XtY); 
    }
}

Eigen::MatrixXd form_ll_XtX(const Eigen::MatrixXd& XtX, const Eigen::VectorXd& query_pt){ 
    Eigen::MatrixXd extra_XtX(XtX.rows(), XtX.cols()); 
    Eigen::MatrixXd ll_XtX(XtX.rows(),XtX.cols()); 
    extra_XtX.topLeftCorner(1,1) = Eigen::MatrixXd::Zero(1,1); 
    extra_XtX.topRightCorner(1,XtX.cols()-1) = XtX.topLeftCorner(1,1) * query_pt.transpose(); 
    extra_XtX.bottomLeftCorner(XtX.rows()-1, 1) =  query_pt * XtX.topLeftCorner(1,1);
    extra_XtX.bottomRightCorner (XtX.rows()-1, XtX.cols()-1) = XtX.bottomLeftCorner(XtX.rows()-1,1)*query_pt.transpose() 
                                                             + query_pt * XtX.topRightCorner(1,XtX.cols()-1) 
                                                             - query_pt * XtX.topLeftCorner(1,1) * query_pt.transpose();                                             
    ll_XtX = XtX - extra_XtX; 
    return ll_XtX; 
    //return XtX;
}

Eigen::VectorXd form_ll_XtY(const Eigen::VectorXd& XtY, const Eigen::VectorXd& query_pt){ 
    Eigen::VectorXd extra_XtY = Eigen::VectorXd::Zero((XtY.size()));
    Eigen::VectorXd ll_XtY(XtY.size());
    extra_XtY.tail(XtY.size()-1) = query_pt * XtY.head(1); 
    ll_XtY = XtY - extra_XtY; 
    return ll_XtY;
    //return XtY;
}

Eigen::VectorXd solve_beta(int kcode, const Eigen::MatrixXd &XtX, const Eigen::MatrixXd &XtY) {
    //    std::cout << "normal equation method used(chloesky)" << "\n"; 
        return(XtX.ldlt().solve(XtY));
}

// [[Rcpp::export]]
Eigen::VectorXd locpoly(Eigen::MatrixXd original_points, double epsilon, double bandwidth, 
                        int method , int N_min, int kcode){ 
    all_point_t points; 
    all_point_t query_pts;
    points = convert_to_vector(original_points); 
    kdtree tree(points, N_min, method);
    std::cout << "Tree built" << std::endl;
    Eigen::VectorXd results(points.size());
    query_pts = convert_to_query(original_points);
    for (int i = 0; i< points.size(); i++) { 
        std::pair<Eigen::MatrixXd, Eigen::VectorXd> XtXXtY;
        XtXXtY = tree.find_XtXXtY(query_pts.at(i), method, epsilon, bandwidth, kcode);
        Eigen::VectorXd f; 
        f = solve_beta(kcode, XtXXtY.first, XtXXtY.second);
        results(i) = f(0); 
    }
//    double t = tmr.elapsed();
//    std::cout << "find XtXXtY" << t << std::endl;
    return results; 
}


void kdtree::test_XtX (Eigen::MatrixXd X){    //test XtX
    Eigen::MatrixXd XtX; 
    Eigen::MatrixXd X_design(X.rows(), X.cols()) ;
    X_design.col(0) = Eigen::VectorXd::Ones(X.rows()); 
    X_design.rightCols(X.cols()-1) = X.leftCols(X.cols()-1);   
    XtX = X_design.transpose() * X_design; 
    if (root->XtX == XtX ){ 
        std::cout << "true" <<'\n'; 
    }
    else{
        std::cout << "Expected XtX" << "\n"; 
        std::cout << XtX << "\n";
        std::cout << "Instead XtX =" << "\n"; 
        std::cout << root->XtX  << "\n";
    }

}

void kdtree::test_XtY (Eigen::MatrixXd X){      // test XtY
    Eigen::MatrixXd XtY; 
    Eigen::MatrixXd X_design(X.rows(), X.cols()) ;
    X_design.col(0) = Eigen::VectorXd::Ones(X.rows()); 
    X_design.rightCols(X.cols()-1) = X.leftCols(X.cols()-1);
    Eigen::VectorXd Y_design;
    Y_design = X.rightCols(1);    
    XtY = X_design.transpose() * Y_design; 
    if (root->XtY == XtY ){ 
        std::cout << "true" <<'\n'; 
    }
    else{
        std::cout << "Expected XtY" << "\n"; 
        std::cout << XtY << "\n";
        std::cout << "Instead XtY =" << "\n"; 
        std::cout << root->XtY  << "\n";
    }

}

void kdtree::test_XtXXtY(Eigen::MatrixXd X){     // test function for XtXXtY
     test_XtX(X); 
     test_XtY(X);
}

void test_weight(){                             // test function for weight
    std::cout << "test" << "\n";
    std::vector<double> max_dim = {0.5,0.4};
    std::vector<double> min_dim = {0.2,0.2};
    Eigen::VectorXd query_pt(max_dim.size());
    query_pt <<  0.5,0.3;
    std::pair<double, double> weights; 
    weights = calculate_weight (1, query_pt, max_dim , min_dim, 0.5);
    std::cout << "max weight =" << weights.first << "\n"; 
    std::cout << "min weight =" << weights.second<< "\n";  
}
/***R
locpoly()
*/
Eigen::VectorXd test_beta(Eigen::MatrixXd X, double bandwidth, int kcode) {   // test function for calculating actual betas for exact 
    Eigen::MatrixXd W = Eigen::MatrixXd::Zero(X.rows(), X.rows()) ;  
    Eigen::MatrixXd X_design(X.rows(), X.cols()) ;
    Eigen::VectorXd results(X_design.rows()); 
    X_design.col(0) = Eigen::VectorXd::Ones(X.rows()); 
    X_design.rightCols(X.cols()-1) = X.leftCols(X.cols()-1);
    Eigen::VectorXd Y_design;
    Y_design = X.rightCols(1); 
    Eigen::MatrixXd X_design2 = X_design; 
    for (int i = 0; i < X_design.rows(); i++){
        Eigen::VectorXd X_query; 
        X_query = X_design.row(i); 
        for (int j =0; j< X_design.rows(); j++){
            double weights = 1; 
            for(int k = 1; k < X_design.cols(); k++) {
                X_design2(j,k) = X_design(j,k) - X_query(k);
                weights *= eval_kernel(kcode, X_design2(j,k)/bandwidth, bandwidth)/bandwidth;  
            }     
            W(j,j) = weights;            
        }   
        Eigen::VectorXd beta;
        beta = (X_design2.transpose()*W * X_design2).ldlt().solve(X_design2.transpose()*W * Y_design);
        results(i) = beta(0);
    }       
    return results;
} 

void test_traversetree(std::unique_ptr<kdnode>& root){  // test functions for traversing
    if(root->n_below == 1){
        std::cout<<"Leaf Node found" <<std::endl;
        std::cout << root->XtX << std::endl;
        return; 
    }
    else{
        std::cout<< "Non Leaf Node found" << std::endl;
        std::cout << root->XtX << std::endl; 
        test_traversetree(root->left_child);
        test_traversetree(root->right_child);
    }
}

/*
int main(){ 
    Eigen::MatrixXd test1 = Eigen::MatrixXd::Random(50, 3);
//   Eigen::MatrixXd test2(3,3); 
//  test2 << 0.2,0.4,0.5,0.5,0.3,0.3,0.3,0.2,0.4;
    std::cout << locpoly(test1, 0.05, 0.05,2, 1, 2) - test_beta(test1, 0.05, 2);
//    locpoly(test1, 0.05, 0.5, 2, 1);
    
//    test_beta(test1, 0.5);
//    std::cout << locpoly(test1, 0.05, 0.5, 2, 4) - locpoly(test1, 0.05, 0.5, 1, 1);
//    test_weight();
 /*   Timer tmr; 
    locpoly(test1, 0.05, 0.5, 2, 4); 
    double t = tmr.elapsed(); 
    std::cout << "N_min = 4 approx time = " << t << "\n";  
    tmr.reset(); 
    locpoly(test1, 0.05, 0.5, 2, 1); 
    t = tmr.elapsed(); 
    std::cout << "N_min = 1 approx time =" << t << "\n"; 
    tmr.reset();
    /*
    locpoly(test1, 0.05, 0.5, 1, 1); 
    t = tmr.elapsed(); 
    std::cout << "exact time =" << t << "\n";
    tmr.reset(); 
    test_beta(test1, 0.5); 
    t = tmr.elapsed();  
    std::cout << "matrix multiplication time =" << t << "\n";


}  
*/










