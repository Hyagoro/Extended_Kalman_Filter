#include <iostream>
#include "tools.h"

using Eigen::VectorXd;
using Eigen::MatrixXd;
using std::vector;

Tools::Tools() {}

Tools::~Tools() {}

VectorXd Tools::CalculateRMSE(const vector<VectorXd> &estimations,
                              const vector<VectorXd> &ground_truth) {
    VectorXd rmse(4);
    rmse << 0,0,0,0;
    if(estimations.empty() || estimations.size() != ground_truth.size()){
        return rmse;
    }

    //accumulate squared residuals
    unsigned long es_size = estimations.size();
    for(unsigned long i=0; i < es_size; ++i){
        VectorXd toto = estimations.at(i).array() - ground_truth.at(i).array();
        rmse = rmse.array() + (toto.array() * toto.array());
    }
    rmse = rmse.array() / es_size;
    return rmse.array().sqrt();
}

MatrixXd Tools::CalculateJacobian(const VectorXd& x_state) {
    MatrixXd Hj(3,4);
    //recover state parameters
    auto px = (float) x_state(0);
    auto py = (float) x_state(1);
    auto vx = (float) x_state(2);
    auto vy = (float) x_state(3);

    float px2py2 = px * px + py * py;
    float sqrt_px2py2 = sqrt(px2py2);
    float l = (px2py2*sqrt_px2py2);

    //check division by zero
    if(fabs(px2py2) < 0.0001){
        cout << "Error, divide by zero" << endl;
        return Hj;
    }
    //compute the Jacobian matrix
    Hj <<   px/sqrt_px2py2,           py/sqrt_px2py2,           0,              0,
            -py/px2py2,               px / px2py2,              0,              0,
            (py*((vx*py)-(vy*px)))/l, (px*((vy*px)-(vx*py)))/l, px/sqrt_px2py2, py/sqrt_px2py2;

    return Hj;
}
