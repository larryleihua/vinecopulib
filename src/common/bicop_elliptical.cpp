/*
    Copyright 2016 Thibault Vatter, Thomas Nagler

    This file is part of vinecopulib.

    vinecopulib is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.

    vinecopulib is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.

    You should have received a copy of the GNU General Public License
    along with vinecopulib.  If not, see <http://www.gnu.org/licenses/>.
 */

#include "include/bicop_elliptical.hpp"


VecXd EllipticalBicop::hfunc2(const MatXd& u)
{
    MatXd v = u;
    v.col(0).swap(v.col(1));
    VecXd h = hfunc1(v);
    return h;
}

VecXd EllipticalBicop::hinv2(const MatXd& u)
{
    MatXd v = u;
    v.col(0).swap(v.col(1));
    VecXd h = hinv1(v);
    return h;
}

double EllipticalBicop::par_to_tau(const VecXd& parameters)
{
    double tau = (2 / M_PI) * asin(parameters(0));
    return tau;
}

// link between Kendall's tau and the par_bicop parameter
VecXd EllipticalBicop::tau_to_par(const double& tau)
{
    VecXd parameters = this->parameters_;
    parameters(0) = sin(tau * M_PI / 2);
    return parameters;
}
