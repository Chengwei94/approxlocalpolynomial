#include <algorithm> 
#include <iostream>     
#include <memory> 
#include <vector>

using all_point_t = std::vector<Eigen::VectorXd>; 

std::pair<double, double> calculate_weight(int, Eigen::VectorXd, std::vector<double>, std::vector<double>); 

double eval_kernel(int,double);

Eigen::VectorXd solve_beta(Eigen::MatrixXd, Eigen::VectorXd);

Eigen::VectorXd locpoly(all_point_t, int, int);

template<typename T, typename U> 
std::pair<T, U> operator+(const std::pair<T,U>&, const std::pair<T,U>&);
all_point_t convert_to_vector(Eigen::MatrixXd);

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
        std::pair<Eigen::MatrixXd, Eigen::VectorXd> get_XtXXtY(Eigen::VectorXd, std::vector<double>, std::vector<double>, std::unique_ptr<kdnode>&);
        std::pair<Eigen::MatrixXd, Eigen::VectorXd> getapprox_XtXXtY(Eigen::VectorXd, std::vector<double>, std::vector<double>, std::unique_ptr<kdnode>&, double, double);
        std::pair<Eigen::MatrixXd, Eigen::VectorXd> find_XtXXtY(Eigen::VectorXd, int, double);
        std::unique_ptr<kdnode> build_tree(all_point_t::iterator, all_point_t::iterator, int, double, int, size_t, std::vector<double>, std::vector<double>);


    
};

