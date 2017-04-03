#include <dlib/optimization.h>
#include <iostream>
#include <vinecopulib.hpp>


using namespace vinecopulib;
using namespace dlib;
typedef matrix<double, 0, 1> column_vector;

class bicop_ll {
public:
    bicop_ll(Eigen::Matrix<double, Eigen::Dynamic, 2> data)
    {
        data_ = data;
    }

    double operator() (const column_vector& x) const
    {
        Eigen::VectorXd par(x.size());
        par << x(0), x(1);
        Bicop bicop(BicopFamily::bb1, 0, par);
        std::cout << bicop.get_parameters() << std::endl;
        return - bicop.loglik(data_);
    }

private:
    Eigen::Matrix<double, Eigen::Dynamic, 2> data_;
};

int main()
{

    Eigen::VectorXd par(2), starting_point(2), lower_bound(2), upper_bound(2);
    par << 0.5, 2;
    starting_point << 0.4, 1.1;
    lower_bound << 0, 1;
    upper_bound << 50, 50;

    column_vector sp = mat(starting_point);
    column_vector lb = mat(lower_bound);
    column_vector ub = mat(upper_bound);

    Bicop bicop(BicopFamily::bb1, 0, par);
    auto u = bicop.simulate(1000);

    find_min_box_constrained(
        lbfgs_search_strategy(1000),
        objective_delta_stop_strategy(1e-10).be_verbose(),
        bicop_ll(u),
        derivative(bicop_ll(u)),
        sp,
        lb,
        ub
    );
    std::cout << "test_function solution:\n" << sp << std::endl;

    return 0;
}
