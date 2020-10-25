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
        std::shared_ptr<kdnode> root; 
        std::shared_ptr<kdnode> build_tree(all_point_t::iterator, all_point_t::iterator, int, double, int, size_t, std::vector<double>, std::vector<double>);
        std::shared_ptr<kdnode> build_exacttree(all_point_t::iterator, all_point_t::iterator, int, double, int, size_t, std::vector<double>, std::vector<double>);
        explicit kdtree(all_point_t XY_arr, int N_min); 
        std::pair<Eigen::MatrixXd, Eigen::VectorXd> get_XtXXtY(const Eigen::VectorXd& X_query, std::vector<double> dim_max, std::vector<double> dim_min, std::shared_ptr<kdnode>& root, const Eigen::VectorXd& h, int kcode);
        std::pair<Eigen::MatrixXd, Eigen::VectorXd> getapprox_XtXXtY(const Eigen::VectorXd& X_query, std::vector<double> dim_max, std::vector<double> dim_min, std::shared_ptr<kdnode>& root, double epsilon, const Eigen::VectorXd& h, int kcode);
        std::pair<Eigen::MatrixXd, Eigen::VectorXd> find_XtXXtY(const Eigen::VectorXd& X_query, int method, double epsilon, const Eigen::VectorXd& h, int kcode);
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


all_point_t XYmat_to_XYarr(const Eigen::MatrixXd& XY_mat);
all_point_t XYmat_to_Xarr(const Eigen::MatrixXd& XY_mat);
all_point_t Xmat_to_Xarr(const Eigen::MatrixXd& X_mat);
double eval_kernel(int kcode, const double& z);
std::pair<double, double> calculate_weight(int kcode, const Eigen::VectorXd& X_query, const std::vector<double>& dim_max, const std::vector<double>& dim_min , const Eigen::VectorXd& h); 
Eigen::MatrixXd form_ll_XtX(const Eigen::MatrixXd& XtX, const Eigen::VectorXd& X_query ); 
Eigen::VectorXd form_ll_XtY(const Eigen::VectorXd& XtY , const Eigen::VectorXd& X_query);
Eigen::VectorXd calculate_beta(const Eigen::MatrixXd& XtX, const Eigen::VectorXd& XtY);
std::pair<Eigen::VectorXd, double> calculate_beta_XtXinv(int kcode, const Eigen::MatrixXd &XtX, const Eigen::MatrixXd &XtY);
double max_weight(int kcode, const Eigen::VectorXd&h);

// R function
Eigen::VectorXd loclin(const Eigen::MatrixXd& XY_mat, int method, int kcode, 
                        double epsilon, const Eigen::VectorXd& h, int N_min);
Eigen::VectorXd bw_loocv(const Eigen::MatrixXd& XY_mat, int method, int kcode, double epsilon, 
                         const Eigen::MatrixXd& bw, int N_min);
Eigen::VectorXd predict(const Eigen::MatrixXd& XY_mat, const Eigen::MatrixXd& X_mat, int method, int kcode, 
                        double epsilon, const Eigen::VectorXd& h,  int N_min);
