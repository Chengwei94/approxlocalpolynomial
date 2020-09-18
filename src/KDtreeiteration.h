#pragma once
#include <memory> 
#include <vector>
#include <Eigen/Dense>
#include <chrono>
#include <iostream>

using all_point_t = std::vector<Eigen::VectorXd>; 

template<typename T, typename U> 
std::pair<T, U> operator+(const std::pair<T,U>&, const std::pair<T,U>&);

template<typename T> 
std::ostream& operator<<(std::ostream&, const std::vector<T>&); 

class kdnode{ 
            public: 
                int n_below; 
                int split_d; 
                double split_v;
                Eigen::MatrixXd XtX; 
                Eigen::VectorXd XtY; 
                std::vector<double> dim_max; 
                std::vector<double> dim_min;
                std::shared_ptr<kdnode> right_child; 
                std::shared_ptr<kdnode> left_child; 
                double sumY; 

                kdnode(); // constructor 
                kdnode(kdnode&&) ;  //move 
                ~kdnode();  // destructor    
};

class kdtree{
    public:     
        kdtree(); 
        ~kdtree();  
        double weight_sf;
        int tracker;
        std::shared_ptr<kdnode> root; 
        std::shared_ptr<kdnode> leaf;
        std::shared_ptr<kdnode> build_tree(all_point_t::iterator, all_point_t::iterator, int, double, int, size_t, std::vector<double>, std::vector<double>);
        std::shared_ptr<kdnode> build_exacttree(all_point_t::iterator, all_point_t::iterator, int, double, int, size_t, std::vector<double>, std::vector<double>);
        explicit kdtree(all_point_t, int, int); 
        std::pair<Eigen::MatrixXd, Eigen::VectorXd> get_XtXXtY(const Eigen::VectorXd& query, std::vector<double> dim_max, std::vector<double> dim_min, std::shared_ptr<kdnode>& root, const Eigen::VectorXd& h, int kcode);
        std::pair<Eigen::MatrixXd, Eigen::VectorXd> getapprox_XtXXtY(const Eigen::VectorXd& query, std::vector<double> dim_max, std::vector<double> dim_min, std::shared_ptr<kdnode>& root, double epsilon, const Eigen::VectorXd& h, int kcode);
        std::pair<Eigen::MatrixXd, Eigen::VectorXd> find_XtXXtY(const Eigen::VectorXd& query, int method, double epsilon, const Eigen::VectorXd& h, int kcode);
        // test functions; 
        void test_XtX(Eigen::MatrixXd);
        void test_XtY(Eigen::MatrixXd);
        void test_XtXXtY(Eigen::MatrixXd);
};


class Timer
{
    public:
        Timer();
        void reset();
        double elapsed() const;

    private:
        typedef std::chrono::high_resolution_clock clock_;
        typedef std::chrono::duration<double, std::ratio<1> > second_;
        std::chrono::time_point<clock_> beg_;
};

// functions 
all_point_t convert_to_vector(const Eigen::MatrixXd& XY_mat);
all_point_t convert_to_query(const Eigen::MatrixXd& XY_mat);
all_point_t convert_to_queryX(const Eigen::MatrixXd& X_mat);
double eval_kernel(int kcode, const double& z);
std::pair<double, double> calculate_weight(int kcode, const Eigen::VectorXd& query, const std::vector<double>& dim_max, const std::vector<double>& dim_min , const Eigen::VectorXd& h); 
Eigen::MatrixXd form_ll_XtX(const Eigen::MatrixXd& XtX, const Eigen::VectorXd& query ); 
Eigen::VectorXd form_ll_XtY(const Eigen::VectorXd& XtY , const Eigen::VectorXd& query);
Eigen::VectorXd calculate_mx(const Eigen::MatrixXd& XtX, const Eigen::VectorXd& XtY);

// R function
Eigen::VectorXd loclinear_i(const Eigen::MatrixXd& XY_mat, int method, int kcode, 
                        double epsilon, const Eigen::VectorXd& h, int N_min);

// for bandwidth selection
std::pair<Eigen::VectorXd, double> calculate_mx_Xinv(int kcode, const Eigen::MatrixXd &XtX, const Eigen::MatrixXd &XtY);
Eigen::VectorXd h_select_i(const Eigen::MatrixXd& XY_mat, int method, int kcode, double epsilon, 
                         const Eigen::MatrixXd& bw, int N_min);
void pertube_XtX(Eigen::MatrixXd& XtX);
double max_weight(int kcode, const Eigen::VectorXd&h);
Eigen::VectorXd predict_i(const Eigen::MatrixXd& XY_mat, const Eigen::MatrixXd& Xpred_mat, int method, int kcode, 
                        double epsilon, const Eigen::VectorXd& h,  int N_min);
