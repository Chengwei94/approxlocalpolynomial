#include "RcppEigen.h"
#include "KDtreeiteration.h"
#include <utility>
#include <algorithm>
#include <iostream>   
#include <random>
#include <stack>
#include <queue>


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

std::shared_ptr<kdnode> kdtree::build_exacttree(all_point_t::iterator start, all_point_t::iterator end, 
                                                int split_d, double split_v, int N_min, size_t len,
                                                std::vector<double> dim_max_, std::vector<double> dim_min_){

    std::shared_ptr<kdnode> newnode = std::make_shared<kdnode>();
    if(end == start) {   
        return newnode; 
    }

    newnode->n_below = len;    
    newnode->dim_max = dim_max_;
    newnode->dim_min = dim_min_;

    if (end-start <= 1) {
        newnode->left_child = nullptr;             
        newnode->right_child = nullptr;               
        Eigen::MatrixXd XtX_; 
        Eigen::VectorXd XtY_;                   
        for (auto i = start; i != end; i ++){
            Eigen::VectorXd XY = *i;    
            Eigen::VectorXd Y = XY.tail<1>(); 
            Eigen::VectorXd X = XY.head(XY.size()-1); 

            XtY_ = X*Y; 
            XtX_ = X*X.transpose();  

            for (auto j = 0; j < dim_max_.size(); j++) {
                newnode->dim_max[j] = X(j+1); 
                newnode->dim_min[j] = X(j+1); 
            }
         }   
         newnode->XtX = XtX_; 
         newnode->XtY = XtY_;  
         return newnode; 
    }

    else {  
        size_t l_len = len/2  ;          // left length
        size_t r_len = len - l_len;     // right length
        auto middle = start + len/2;   // middle iterator 
        int max = 0; 
        int dim = 0;
        for(int i = 0; i < newnode->dim_max.size(); i++){   
            double var = newnode->dim_max[i] - newnode->dim_min[i]; 
            if(var > max){
                max = var; 
                dim = i; 
            }
        }

        newnode -> split_d = dim; 
        int vector_dim = dim + 1;  

        std::nth_element(start, middle, end, [vector_dim](const Eigen::VectorXd& a, const Eigen::VectorXd & b) {
            return a(vector_dim) < b(vector_dim);    
        });           

        newnode->split_v = (*middle)[vector_dim];   

        dim_max_[dim] = newnode->split_v; 
        dim_min_[dim] = newnode->split_v;

        newnode-> left_child = build_exacttree(start, middle, newnode->split_d, newnode->split_v, N_min, l_len, dim_max_, newnode->dim_min);
        newnode-> right_child = build_exacttree(middle, end, newnode->split_d, newnode->split_v, N_min, r_len, newnode->dim_max, dim_min_);
        
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


std::shared_ptr<kdnode> kdtree::build_tree(all_point_t::iterator start, all_point_t::iterator end, 
                                           int split_d, double split_v, int N_min, size_t len,
                                           std::vector<double> dim_max_, std::vector<double> dim_min_){
        
    std::shared_ptr<kdnode> newnode = std::make_shared<kdnode>();
    if(end == start) {   
        return newnode; 
    }

    newnode->n_below = len;    
    newnode->dim_max = dim_max_;
    newnode->dim_min = dim_min_;

    if (end-start <= N_min) {
        newnode->left_child = nullptr;             
        newnode->right_child = nullptr;                 
        Eigen::MatrixXd XtX_(dim_max_.size() + 1 , dim_max_.size() + 1);
        XtX_.setZero(); 
        Eigen::VectorXd XtY_(dim_max_.size() + 1);
        XtY_.setZero();
        for (auto k =0; k <dim_max_.size(); k++ ){
            dim_max_[k] = (*start)(k+1);
            dim_min_[k] = (*start)(k+1);
        }                  
        for (auto i = start; i != end; i ++){
            Eigen::VectorXd XY = *i;    
            Eigen::VectorXd Y = XY.tail<1>(); 
            Eigen::VectorXd X = XY.head(XY.size()-1); 
            XtY_ += X*Y; 
            XtX_ += X*X.transpose();  
            for (auto j = 0; j < dim_max_.size(); j++) {
                if(X(j+1) > dim_max_[j]){
                    dim_max_[j] = X(j+1); 
                }
                if(X(j+1) < dim_min_[j]){
                    dim_min_[j] = X(j+1); 
                } 
            }
        } 
         newnode->dim_max = dim_max_;
         newnode->dim_min = dim_min_;   
        // std::cout << "dim_min_" << dim_min_ <<'\n'; 
        // std::cout << "dim_max_" << dim_max_ <<'\n';
         newnode->XtX = XtX_; 
         newnode->XtY = XtY_;  

         return newnode; 
    }

    else {  
        size_t l_len = len/2  ;          // left length
        size_t r_len = len - l_len;     // right length
        auto middle = start + len/2;   // middle iterator 
        int max = 0; 
        int dim = 0;
        for(int i = 0; i < newnode->dim_max.size(); i++){   
            double var = newnode->dim_max[i] - newnode->dim_min[i]; 
            if(var > max){
                max = var; 
                dim = i; 
            }
        }

        newnode -> split_d = dim; 
        int vector_dim = dim + 1;  

        std::nth_element(start, middle, end, [vector_dim](const Eigen::VectorXd& a, const Eigen::VectorXd & b) {
            return a(vector_dim) < b(vector_dim);    
        });           

        newnode->split_v = (*middle)[vector_dim];   

        dim_max_[dim] = newnode->split_v; 
        dim_min_[dim] = newnode->split_v;

        newnode-> left_child = build_tree(start, middle, newnode->split_d, newnode->split_v, N_min, l_len, dim_max_, newnode->dim_min);
        newnode-> right_child = build_tree(middle, end, newnode->split_d, newnode->split_v, N_min, r_len, newnode->dim_max, dim_min_);
        
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


kdtree::kdtree(all_point_t XY_arr, int N_min , int method) {    // create tree 
    size_t len = XY_arr.size(); 
    weight_sf = 0;
    tracker = 0;
    std::vector<double> dim_max; 
    std::vector<double> dim_min; 

    for(int i=1; i<XY_arr[0].size()-1; i++){
        dim_max.push_back(XY_arr[0](i)); 
        dim_min.push_back(XY_arr[0](i));
    }

    for(int i=1; i<XY_arr[0].size()-1; i++) { // loop size of first point to find dimension; 
        for (int j=0; j<XY_arr.size(); j++){
            if (XY_arr[j](i) > dim_max.at(i-1)){
                dim_max.at(i-1) = XY_arr[j](i);
            }
            if (XY_arr[j][i] < dim_min.at(i-1)){ 
                dim_min.at(i-1) = XY_arr[j](i);
            }
        }
    } 
    if (method == 1) { 
        root = build_exacttree(XY_arr.begin(), XY_arr.end(), 1, 1, N_min, len, dim_max, dim_min);
    }

    if (method == 2) { 
        root = build_exacttree(XY_arr.begin(), XY_arr.end(), 1, 1, N_min, len, dim_max, dim_min);
    }
}

all_point_t convert_to_query(const Eigen::MatrixXd& XY_mat){     //  conversion to query form 
    std::vector<Eigen::VectorXd> XY_arr; 
    Eigen::MatrixXd XY_trans = XY_mat.transpose(); 
    for (int i=0; i<XY_trans.cols(); i++) {
        Eigen::VectorXd xy; 
        xy = XY_trans.block(0, i, XY_trans.rows()-1, 1); 
        XY_arr.push_back(xy);
    }
    return XY_arr; 
}

all_point_t convert_to_queryX(const Eigen::MatrixXd& X_mat){     //  conversion to query form 
    std::vector<Eigen::VectorXd> X_arr; 
    Eigen::MatrixXd X_trans = X_mat.transpose(); 
    for (int i=0; i<X_trans.cols(); i++) {
        Eigen::VectorXd x; 
        x = X_trans.block(0, i, X_trans.rows(), 1); 
        X_arr.push_back(x);
    }
    return X_arr; 
}

all_point_t convert_to_vector(const Eigen::MatrixXd& XY_mat){    // conversion to vector form 
    std::vector<Eigen::VectorXd> XY_arr; 
    Eigen::MatrixXd XY_temp(XY_mat.rows(), XY_mat.cols()+1); 
    XY_temp.topRightCorner(XY_mat.rows(), XY_mat.cols()) = XY_mat; 
    XY_temp.col(0) = Eigen::VectorXd::Ones(XY_mat.rows()); 

    for (int i = 0; i <XY_temp.rows(); i++){
        XY_arr.push_back(XY_temp.row(i));
    }
    return XY_arr; 
}

inline double eval_kernel(int kcode, const double& z)
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

std::pair<double,double> calculate_weight(int kcode, const Eigen::VectorXd& query, const std::vector<double>& dim_max, 
                                          const std::vector<double>& dim_min, const Eigen::VectorXd& h) { 
    if(dim_max.size()!= query.size()){
        std::cout << "error in dimensions"; 
        throw std::exception(); 
    }
    double w_max = 1;
    double w_min = 1; 
    double dis_max = 0; 
    double dis_min = 0;
    for(int i=0; i < dim_max.size(); i++)
        { 
            if (query(i) <= dim_max.at(i) && query(i) >= dim_min.at(i)) {
                dis_max = std::max(abs(dim_max.at(i) - query(i)), abs(dim_min.at(i) - query(i)));    // dis_max = which ever dim is further
                dis_min = 0;                                                                             
                w_min *= eval_kernel(kcode, dis_max/h(i)) / h(i);    // kern weight = multiplication of weight of each dim 
                w_max *= eval_kernel(kcode, dis_min/h(i)) / h(i);
                
            }
            else if (abs(query(i) - dim_max.at(i)) > abs(query(i) - dim_min.at(i))){
                dis_max = query(i) - dim_max.at(i);
                dis_min = query(i) - dim_min.at(i);
                w_min *= eval_kernel(kcode, dis_max/h(i)) / h(i);
                w_max *= eval_kernel(kcode, dis_min/h(i)) / h(i); 
            }
            else{
                dis_max = query(i) - dim_min.at(i);
                dis_min = query(i) - dim_max.at(i); 
                w_min *= eval_kernel(kcode, dis_max/h(i)) / h(i); 
                w_max *= eval_kernel(kcode, dis_min/h(i)) / h(i); 
            }
        }
    return std::make_pair(w_max, w_min); 
}

std::pair<Eigen::MatrixXd, Eigen::VectorXd> kdtree::get_XtXXtY(const Eigen::VectorXd& query, 
                                                               std::vector<double> dim_max, 
                                                               std::vector<double> dim_min, 
                                                               std::shared_ptr<kdnode>& root, 
                                                               const Eigen::VectorXd& h, 
                                                               int kcode){

    std::pair<double,double> weights;
    std::stack<std::shared_ptr<kdnode>> storage; 
    std::shared_ptr<kdnode> curr = root; 
    Eigen::MatrixXd XtX = Eigen::MatrixXd::Zero(curr->XtX.rows() , curr->XtX.cols()); 
    Eigen::VectorXd XtY = Eigen::MatrixXd::Zero(curr->XtY.rows() , curr->XtY.cols());
    weights = calculate_weight(kcode, query, dim_max, dim_min,h);
    double w_max = weights.first; 
    double w_min = weights.second;

    while (w_max != w_min || storage.empty() == false){
        while (w_max != w_min ){   // if condition fufilled
            storage.push(curr);
            curr = curr->left_child;  
            weights = calculate_weight(kcode, query, curr->dim_max, curr->dim_min, h);  // calculate max and min weight 
            w_max = weights.first;                      
            w_min = weights.second; 

            if(w_max == w_min ){ 
                XtX += w_max*curr->XtX;
                XtY += w_max*curr->XtY; 
            }           
        }

        curr = storage.top();
        storage.pop(); 
        curr = curr->right_child; 
        weights = calculate_weight(kcode, query, curr->dim_max, curr->dim_min, h);  // calculate max and min weight 
        w_max = weights.first;   
        w_min = weights.second; 

        if(w_max == w_min){ 
            XtX += w_max*curr->XtX;
            XtY += w_max*curr->XtY;
        }
    }   
    return std::make_pair(XtX,XtY); 
}

// std::pair<Eigen::MatrixXd, Eigen::VectorXd> kdtree::getapprox_XtXXtY(const Eigen::VectorXd& query,
//                                                                      std::vector<double> dim_max,
//                                                                      std::vector<double> dim_min, 
//                                                                      std::shared_ptr<kdnode>& root,
//                                                                      double epsilon, 
//                                                                      const Eigen::VectorXd& h,
//                                                                      int kcode){ 
//     std::pair<double,double> weights;
//     weights = calculate_weight(kcode, query, dim_max, dim_min, h);  // calculate max and min weight 
//     double w_max = weights.first;                      
//     double w_min = weights.second;

//     if ((w_max - w_min <= 2 * epsilon * (weight_sf + root->n_below*w_min))|| root->left_child == nullptr) {   // if condition fulfilled
//         double weight = 0.5*(w_max + w_min);
//         weight_sf += weight;                                                              // weight_so_far + current weights found    
//         return std::make_pair(weight*root->XtX, weight*root->XtY);
//     }
//   else { 
//         return getapprox_XtXXtY(query, root->right_child->dim_max, root->right_child->dim_min, root->right_child, epsilon, h, kcode)+
//           getapprox_XtXXtY(query, root->left_child->dim_max, root->left_child->dim_min, root->left_child, epsilon, h, kcode);
//     }
// } 

std::pair<Eigen::MatrixXd, Eigen::VectorXd> kdtree::getapprox_XtXXtY(const Eigen::VectorXd& query,
                                                                      std::vector<double> dim_max,
                                                                      std::vector<double> dim_min, 
                                                                      std::shared_ptr<kdnode>& root,
                                                                      double epsilon, 
                                                                      const Eigen::VectorXd& h,
                                                                      int kcode){ 

    std::pair<double,double> weights;
     std::stack<std::shared_ptr<kdnode>> storage; 
    std::shared_ptr<kdnode> curr = root; 
    Eigen::MatrixXd XtX = Eigen::MatrixXd::Zero(curr->XtX.rows() , curr->XtX.cols()); 
    Eigen::VectorXd XtY = Eigen::MatrixXd::Zero(curr->XtY.rows() , curr->XtY.cols());

    weights = calculate_weight(kcode, query, dim_max, dim_min, h);
    double w_max = weights.first; 
    double w_min = weights.second;
    weight_sf = 0; 

    while (( curr->left_child != nullptr && w_max-w_min > 2*epsilon*(weight_sf + curr->n_below*w_min)) || storage.empty() == false  ){
        while (w_max-w_min > 2*epsilon*(weight_sf + curr->n_below*w_min) && curr->left_child != nullptr ){   // if condition fufilled        
            storage.push(curr);
            curr = curr->left_child;  
            weights = calculate_weight(kcode, query, curr->dim_max, curr->dim_min, h);  // calculate max and min weight 
            w_max = weights.first;                      
            w_min = weights.second; 
            if(w_max - w_min <= 2 * epsilon * (weight_sf + curr->n_below*w_min)){ 
                double weight = 0.5*(w_max + w_min);
                weight_sf += weight;  
                XtX += weight * curr->XtX;
                XtY += weight * curr->XtY; 
            }           
        }
        
        curr = storage.top(); 
        storage.pop(); 
        curr = curr->right_child; 
        weights = calculate_weight(kcode, query, curr->dim_max, curr->dim_min, h);  // calculate max and min weight 
        w_max = weights.first;   
        w_min = weights.second; 
        if(w_max - w_min <= 2 * epsilon * (weight_sf + curr->n_below*w_min)){ 
            double weight = 0.5*(w_max + w_min);
            weight_sf += weight;  
            XtX += weight * curr->XtX;
            XtY += weight * curr->XtY; 
        }          
    }
    return std::make_pair(XtX,XtY); 
}

// std::pair<Eigen::MatrixXd, Eigen::VectorXd> kdtree::getapprox_XtXXtY(const Eigen::VectorXd& query,
//                                           -                           std::vector<double> dim_max,
//                                                                      std::vector<double> dim_min, 
//                                                                      std::shared_ptr<kdnode>& root,
//                                                                      double epsilon, 
//                                                                      const Eigen::VectorXd& h,
//                                                                      int kcode){ 

//     std::pair<double,double> weights;
//     std::queue<std::shared_ptr<kdnode>> storage; 
//     std::shared_ptr<kdnode> curr = root; 
//     Eigen::MatrixXd XtX = Eigen::MatrixXd::Zero(curr->XtX.rows() , curr->XtX.cols()); 
//     Eigen::VectorXd XtY = Eigen::MatrixXd::Zero(curr->XtY.rows() , curr->XtY.cols());

//     weights = calculate_weight(kcode, query, dim_max, dim_min, h);
//     double w_max = weights.first; 
//     double w_min = weights.second;
//     weight_sf = 0;
//     storage.push(curr);  

//     while (storage.empty() == false){
//         curr = storage.front();
//         std::pair<double, double> weights_left = calculate_weight(kcode, query, curr->left_child->dim_max, curr->left_child->dim_min, h);  // calculate max and min weight 
//         double w_max_left = weights_left.first;                      
//         double w_min_left = weights_left.second; 
//         storage.pop();
//         if (w_max_left -w_min_left <= 2*epsilon*(weight_sf + curr->n_below*w_min) || curr->left_child == nullptr ){   // if condition fufilled        
//             double weight = 0.5*(w_max_left + w_min_left);
//             weight_sf += weight;  
//             XtX += weight * curr->XtX;
//             XtY += weight * curr->XtY; 
//         }
//         else {
//             storage.push(curr->left_child);
//         }                 
//         std::pair<double, double> weights_right = calculate_weight(kcode, query, curr->right_child->dim_max, curr->right_child->dim_min, h);  // calculate max and min weight 
//         double w_max_right = weights_right.first;   
//         double w_min_right = weights_right.second; 
//         if (w_max_right -w_min_right <= 2*epsilon*(weight_sf + curr->n_below*w_min) || curr->left_child == nullptr ){   // if condition fufilled        
//             double weight = 0.5*(w_max_right + w_min_right);
//             weight_sf += weight;  
//             XtX += weight * curr->XtX;
//             XtY += weight * curr->XtY; 
//         }
//         else {
//             storage.push(curr->right_child);
//         }            
//     }
//     return std::make_pair(XtX,XtY); 

// }
  

std::pair<Eigen::MatrixXd, Eigen::VectorXd> kdtree::find_XtXXtY(const Eigen::VectorXd& query, 
                                                                int method, 
                                                                double epsilon, 
                                                                const Eigen::VectorXd& h, 
                                                                int kcode ) { 
    if (method == 1) {
        std::pair<Eigen::MatrixXd, Eigen::VectorXd> XtXXtY = get_XtXXtY(query, root->dim_max, root->dim_min, root, h, kcode);
        Eigen::MatrixXd ll_XtX = form_ll_XtX(XtXXtY.first, query);
        Eigen::VectorXd ll_XtY = form_ll_XtY(XtXXtY.second, query);
        return std::make_pair(ll_XtX , ll_XtY); 
    }
    else {
        weight_sf =0;
        std::pair<Eigen::MatrixXd, Eigen::VectorXd> XtXXtY = getapprox_XtXXtY(query, root->dim_max, root->dim_min, root, epsilon, h, kcode); 
        Eigen::MatrixXd ll_XtX = form_ll_XtX(XtXXtY.first, query); 
        Eigen::VectorXd ll_XtY = form_ll_XtY(XtXXtY.second, query);
        return std::make_pair(ll_XtX , ll_XtY); 
    }
}

Eigen::MatrixXd form_ll_XtX(const Eigen::MatrixXd& XtX, const Eigen::VectorXd& query){ 
    Eigen::MatrixXd extra_XtX(XtX.rows(), XtX.cols()); 
    Eigen::MatrixXd ll_XtX(XtX.rows(),XtX.cols()); 
    extra_XtX.topLeftCorner(1,1) = Eigen::MatrixXd::Zero(1,1); 
    extra_XtX.topRightCorner(1,XtX.cols()-1) = XtX.topLeftCorner(1,1) * query.transpose(); 
    extra_XtX.bottomLeftCorner(XtX.rows()-1, 1) =  query * XtX.topLeftCorner(1,1);
    extra_XtX.bottomRightCorner (XtX.rows()-1, XtX.cols()-1) = XtX.bottomLeftCorner(XtX.rows()-1,1)*query.transpose() 
                                                             + query * XtX.topRightCorner(1,XtX.cols()-1) 
                                                             - query * XtX.topLeftCorner(1,1) * query.transpose();                                             
    ll_XtX = XtX - extra_XtX; 
    //pertube_XtX(ll_XtX);
    return ll_XtX; 
}

void pertube_XtX(Eigen::MatrixXd& XtX){
    
    double max = 0.000000000000000001; 
    double min = 0.0000000000000000001;
    std::default_random_engine generator;
    std::uniform_real_distribution<double> distr(max,min);
    for (int i=0; i <XtX.rows(); i++) {
        for (int j =0; j<XtX.cols(); j++){
            XtX(i,j) = XtX(i,j) + distr(generator);
        }
    }
}

Eigen::VectorXd form_ll_XtY(const Eigen::VectorXd& XtY, const Eigen::VectorXd& query){ 
    Eigen::VectorXd extra_XtY = Eigen::VectorXd::Zero((XtY.size()));
    Eigen::VectorXd ll_XtY(XtY.size());
    extra_XtY.tail(XtY.size()-1) = query * XtY.head(1); 
    ll_XtY = XtY - extra_XtY; 
    return ll_XtY;
    //return XtY;
}

Eigen::VectorXd calculate_mx(int kcode, const Eigen::MatrixXd &XtX, const Eigen::MatrixXd &XtY) {
        
        return(XtX.ldlt().solve(XtY));; 
}

std::pair<Eigen::VectorXd, double> calculate_mx_Xinv(int kcode, const Eigen::MatrixXd &XtX, const Eigen::MatrixXd &XtY) {
    
    Eigen::MatrixXd I = Eigen::MatrixXd::Identity(XtX.rows(),XtX.cols());
    Eigen::MatrixXd XtX_inv = XtX.ldlt().solve(I);

    return std::make_pair(XtX.ldlt().solve(XtY), XtX_inv(0,0));
}

double max_weight(int kcode, const Eigen::VectorXd&h){
    double maxweight = 1; 
    for(int i =0; i < h.size(); i++){
        maxweight *= eval_kernel(kcode, 0)/h(i);
    }
    return maxweight;
}

//' This function is used to estimate the curve using local linear 
//' implemented in C++ for faster efficiency 
//'
//' @para XY_mat: Matrix of both X and Y with Y in the last colum 
//' @para method: 1 for exact, 2 for approximate 
//' @para kcode: Choice of kernel
//' @para epsilon: Range allowed for approximate 
//' @para bw: Matrix of bandwidth 
//' @param N_min: Number of values allowed in the leaf of the tree:
//' larger nodes stored result in less accuracy 
//' @export
//' @examples 
//' std::cout << loclinear_i(test1, 2, 1, 0.05, h, 1) where h is a vector of bandwidth


// [[Rcpp::export]]
Eigen::VectorXd loclinear_i(const Eigen::MatrixXd& XY_mat, int method, int kcode, 
                        double epsilon, const Eigen::VectorXd& h,  int N_min){    //method 1 for exact, 2 for approximate, bandwidth is 1d, will add vector later, epsilon is for epsilon for
                        
    all_point_t XY_arr; 
    all_point_t query_arr;
    XY_arr = convert_to_vector(XY_mat); 
    query_arr = convert_to_query(XY_mat);
    kdtree tree(XY_arr, N_min, method);
    Eigen::VectorXd results(XY_arr.size());

    for (int i = 0; i< XY_arr.size(); i++) { 
        std::pair<Eigen::MatrixXd, Eigen::VectorXd> XtXXtY;
        XtXXtY = tree.find_XtXXtY(query_arr.at(i), method, epsilon, h, kcode); 
        Eigen::VectorXd mx = calculate_mx(kcode, XtXXtY.first, XtXXtY.second);
        results(i) = mx(0); 
    }
    return results; 
}

// [[Rcpp::export]]
Eigen::VectorXd predict_i(const Eigen::MatrixXd& XY_mat, const Eigen::MatrixXd& Xpred_mat, int method, int kcode, 
                        double epsilon, const Eigen::VectorXd& h,  int N_min){    //method 1 for exact, 2 for approximate, bandwidth is 1d, will add vector later, epsilon is for epsilon for
    all_point_t XY_arr; 
    all_point_t query_arr;
    XY_arr = convert_to_vector(XY_mat); 
    query_arr = convert_to_queryX(Xpred_mat);
    kdtree tree(XY_arr, N_min, method);
    Eigen::VectorXd results(Xpred_mat.rows());
    for (int i = 0; i< Xpred_mat.rows(); i++) { 
        std::pair<Eigen::MatrixXd, Eigen::VectorXd> XtXXtY;
        XtXXtY = tree.find_XtXXtY(query_arr.at(i), method, epsilon, h, kcode); 
        Eigen::VectorXd mx = calculate_mx(kcode, XtXXtY.first, XtXXtY.second);
        results(i) = mx(0); 
    }
    return results; 
}


//' This function is used to select bandwidth given a matrix of bandwidths 
//' implemented in C++ for faster efficiency 
//'
//' @para XY_mat: Matrix of both X and Y with Y in the last colum 
//' @para method: 1 for exact, 2 for approximate 
//' @para kcode: Choice of kernel
//' @para epsilon: Range allowed for approximate 
//' @para bw: Matrix of bandwidth 
//' @param N_min: Number of values allowed in the leaf of the tree:
//' larger nodes stored result in less accuracy 
//' @export
//' @examples 
//' std::cout << h_select_i(test1, 2, 1, 0.05, h, 1) where h is a vector of bandwidth

// [[Rcpp::export]]
Eigen::VectorXd h_select_i(const Eigen::MatrixXd& XY_mat, int method, int kcode, double epsilon, 
                         const Eigen::MatrixXd& bw, int N_min){  
    all_point_t XY_arr; 
    all_point_t query_arr; 
    Eigen::VectorXd h_opt; 
    double CV_opt = 1000000000000; 
    Eigen::VectorXd Y_mat = XY_mat.col(XY_mat.cols()-1);
    Eigen::VectorXd SSE(bw.rows());
    
    XY_arr = convert_to_vector(XY_mat); 
    query_arr = convert_to_query(XY_mat);
    Eigen::VectorXd results(XY_arr.size());
    
    kdtree tree(XY_arr, N_min, method);
    
    double w_point;
    
    for (int j = 0; j< bw.rows(); j ++) { 
        Eigen::VectorXd h = bw.row(j); 
        w_point = 0;
        double CV = 0; 
        for (int i = 0; i< XY_arr.size(); i++) {             std::pair<Eigen::VectorXd, double> f_inverse;
            std::pair<Eigen::MatrixXd, Eigen::VectorXd> XtXXtY;
            XtXXtY = tree.find_XtXXtY(query_arr.at(i), method, epsilon, h, kcode);
            std::pair<Eigen::VectorXd, double> mx_Xinv = calculate_mx_Xinv(kcode, XtXXtY.first, XtXXtY.second);
            Eigen::VectorXd mx = mx_Xinv.first; 
            results(i) = mx(0); 
            w_point = max_weight(kcode, h)*mx_Xinv.second;
        //  std::cout << "w_point =" << w_point << "\n";
            CV += pow((results(i) - Y_mat(i)),2)/pow((1-w_point),2);  
    }
    SSE(j) = CV;
    if (CV <= CV_opt) { 
        CV_opt = CV; 
        h_opt = h; 
    }
}
    //Rcpp::Rcout << SSE; 
    return h_opt;
}









