#include <algorithm> 
#include <iostream>     
#include <memory> 
#include <vector>

using point_t = std::vector< double >;
using all_point_t = std::vector<point_t>; 

std::pair<double, double> calculate_weight(int, point_t, point_t, point_t); 

double eval_kernel(int,double);

Eigen::VectorXd solve_beta(Eigen::MatrixXd, Eigen::VectorXd);

template<typename T, typename U> 
std::pair<T, U> operator+(const std::pair<T,U>&, const std::pair<T,U>&);

class kdnode{ 
            public: 
                int n_below; 
                int split_d; 
                double split_v;
                Eigen::MatrixXd XtX; 
                Eigen::VectorXd XtY; 
                std::vector<double> max_dim; 
                std::vector<double> min_dim;
                std::unique_ptr<kdnode> right_child; 
                std::unique_ptr<kdnode> left_child; 
                all_point_t points; 
                double sumY; 

                kdnode(); // constructor 
                kdnode(kdnode&&) ;  //move 
                ~kdnode();  // destructor    
};

class kdtree{
    public:     
        kdtree(); 
        ~kdtree();  
        std::unique_ptr<kdnode> root; 
        std::unique_ptr<kdnode> leaf;
        explicit kdtree(all_point_t, int); 
        std::pair<Eigen::MatrixXd, Eigen::VectorXd> get_XtXXtY(point_t, point_t, point_t, std::unique_ptr<kdnode>&);
        std::pair<Eigen::MatrixXd, Eigen::VectorXd> find_XtXXtY(point_t);
        std::unique_ptr<kdnode> build_tree(all_point_t::iterator, all_point_t::iterator, int, double, int, size_t, point_t, point_t);


    
};

