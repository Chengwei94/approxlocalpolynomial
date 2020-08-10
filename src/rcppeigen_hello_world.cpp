#include <iostream>

// [[Rcpp::plugins(cpp14)]]

auto doubleMe(const int & x) {
    return x+x;
}

int main(void) {
    std::cout << "1.0         -> " << doubleMe(1.0) << "\n"
              << "1           -> " << doubleMe(1)   << "\n"
              << std::endl;
}