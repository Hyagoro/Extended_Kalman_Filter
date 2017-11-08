#include "kalman_filter.h"
#include <iostream>

using Eigen::MatrixXd;
using Eigen::VectorXd;
using namespace std;

// Please note that the Eigen library does not initialize 
// VectorXd or MatrixXd objects with zeros upon creation.

KalmanFilter::KalmanFilter() {}

KalmanFilter::~KalmanFilter() {}

void KalmanFilter::Init(VectorXd &x_in, MatrixXd &P_in, MatrixXd &F_in,
                        MatrixXd &H_in, MatrixXd &R_in, MatrixXd &Q_in) {
    x_ = x_in;
    P_ = P_in;
    F_ = F_in;
    H_ = H_in;
    R_ = R_in;
    Q_ = Q_in;
}

void KalmanFilter::Predict() {
    x_ = F_ * x_;
    MatrixXd Ft = F_.transpose();
    P_ = F_ * P_ * Ft + Q_;
}

void KalmanFilter::Update(const VectorXd &z) {
//  VectorXd z_pred = ;
    VectorXd y = z - (H_ * x_);
    MatrixXd Ht = H_.transpose();
    MatrixXd PHt = P_ * Ht;
    MatrixXd S = (H_ * PHt) + R_;
    MatrixXd K = PHt * S.inverse();

    //new estimate
    x_ = x_ + (K * y);
    long x_size = x_.size();
    MatrixXd I = MatrixXd::Identity(x_size, x_size);
    P_ = (I - K * H_) * P_;
}

void KalmanFilter::UpdateEKF(const VectorXd &z) {

    double rho = sqrt(x_(0) * x_(0) + x_(1) * x_(1));
    double phi = atan2(x_(1), x_(0));
    double rho_dot = (x_(0) * x_(2) + x_(1) * x_(3)) / rho;
    VectorXd h = VectorXd(3); // h(x_)

    h << rho, phi, rho_dot;

    // y = z - h(x)
    VectorXd y = z - h;

	if(y(1) > M_PI){
        y(1) -= 2*M_PI;
        cout << "phi > Pi" << endl;
    }
    else if(y(1) < -M_PI){
        y(1) += 2*M_PI;
        cout << "phi < Pi" << endl;
    }

    // Use Hj calculated in FusionEKF
    MatrixXd Ht = H_.transpose();
    MatrixXd PHt = P_ * Ht;
    MatrixXd S = (H_ * PHt) + R_;
    MatrixXd K = PHt * S.inverse();

    //new estimate
    x_ = x_ + (K * y);
    long x_size = x_.size();
    MatrixXd I = MatrixXd::Identity(x_size, x_size);
    P_ = (I - K * H_) * P_;
}
