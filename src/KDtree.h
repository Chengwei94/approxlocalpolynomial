#include <algorithm> 
#include <iostream>     
#include <memory> 
#include <vector>
#include <iostream>
#include <chrono>

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
        std::pair<Eigen::MatrixXd, Eigen::VectorXd> get_XtXXtY(Eigen::VectorXd, std::vector<double>, std::vector<double>, std::unique_ptr<kdnode>& ,double );
        std::pair<Eigen::MatrixXd, Eigen::VectorXd> getapprox_XtXXtY(Eigen::VectorXd, std::vector<double>, std::vector<double>, std::unique_ptr<kdnode>&, double, double, double);
        std::pair<Eigen::MatrixXd, Eigen::VectorXd> find_XtXXtY(Eigen::VectorXd, int, double, double);
        std::unique_ptr<kdnode> build_tree(all_point_t::iterator, all_point_t::iterator, int, double, int, size_t, std::vector<double>, std::vector<double>);
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
all_point_t convert_to_vector(Eigen::MatrixXd);
all_point_t convert_to_query(Eigen::MatrixXd);
double eval_kernel(int, double, double);
std::pair<double, double> calculate_weight(int, Eigen::VectorXd, std::vector<double>, std::vector<double>, double); 
void findXtX(Eigen::VectorXd);
Eigen::MatrixXd form_ll_XtX(const Eigen::MatrixXd &, const Eigen::VectorXd & ); 
Eigen::VectorXd form_ll_XtY(const Eigen::VectorXd &, const Eigen::VectorXd & );
Eigen::VectorXd solve_beta(Eigen::MatrixXd, Eigen::VectorXd);

// R function
Eigen::VectorXd locpoly(Eigen::MatrixXd, double, double);


// test functions
void test_traversetree(std::unique_ptr<kdnode> &);




