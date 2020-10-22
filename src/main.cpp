#include <iostream>
//#include "KDtree.h"
#include "KDtreeiteration.h"
#include <utility>
#include <iostream>
//#include "KDtreetest.h"


//loclinear (XY_arr, epsilon, h, method, N_min ,kcode )
int main(){ 
   Eigen::MatrixXd test1 = Eigen::MatrixXd::Random(5000,2);
    //std::cout << "test1 =" << test1 << "\n";
     Eigen::VectorXd bw(1); 
     bw << 0.13;
    //Eigen::MatrixXd build = Eigen::MatrixXd::Random(1000,3); 
  // Test Case 1: 
  //  test_XtXcases();
  //  test_XtYcases(); 
    Timer tmr;
    loclinear_i(test1, 2, 1, 0.05, bw, 1);
    double t = tmr.elapsed(); 
    std:: cout << "t = " << t << "\n";
    tmr.reset(); 
    loclinear_i(test1, 1, 1, 0.05, bw, 1);
    t = tmr.elapsed(); 
    std:: cout << "t = " << t << "\n";
  //Eigen::MatrixXd test2 = Eigen::MatrixXd::Random(10,1);
  // std::cout << predict_i(test1,test2, 2, 1, 0.05, bw, 1);
    
//    test_beta(test1, 0.5);
  //   std::cout << loclinear_i(test1, 2 , 1, 0.05, bw, 1) - test_beta(test1, bw, 1);
//    test_weight();
    //
    //Timer tmr; 
    //loclinear(test1, 2, 1, 0.05, bw, 4); 
    //double t = tmr.elapsed(); 
    //std::cout << max_weight(1, bw); 
    // test_beta(test1, bw, 1); 
   /* std::cout << "N_min = 4 approx time = " << t << "\n";  
    tmr.reset(); 
    loclinear_i(test1, 2, 1, 0.05, bw, 1); 
    t = tmr.elapsed(); 
    std::cout << "N_min = 1 approx time =" << t << "\n"; 
    //tmr.reset(); 
    //loclinear_i(test1,1, 1, 0.05,bw, 1);
    //std::cout << "N_min = 1 exact time =" << t << "\n";*/

 /* Timer tmr; 
    loclinear(test1, 0.05, bw, 1, 1, 2);
    double t = tmr.elapsed(); 
    std::cout << "exact time =" << t << "\n";
*/
   /*
    tmr.reset(); 
    test_beta(test1, bw, 1); 
    double t = tmr.elapsed();  
    std::cout << "matrix multiplication time =" << t << "\n"; */
   /* std::cout << "--------------------------------" << "\n";
    Eigen::MatrixXd h(5,2); 
    h << 0.2,0.3,0.01,0.02,0.1,0.3,0.2,0.4,0.05,0.08; 
    std::cout << h_select(test1, 1, 1, 1, h, 1);*/
}  