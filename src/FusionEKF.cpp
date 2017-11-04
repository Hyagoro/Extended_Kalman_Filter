#include "FusionEKF.h"
#include "tools.h"
#include "Eigen/Dense"
#include <iostream>

using namespace std;
using Eigen::MatrixXd;
using Eigen::VectorXd;
using std::vector;

/*
 * Constructor.
 */
FusionEKF::FusionEKF() {
  is_initialized_ = false;

  previous_timestamp_ = 0;

  // initializing matrices
  R_laser_ = MatrixXd(2, 2);
  R_radar_ = MatrixXd(3, 3);
  H_laser_ = MatrixXd(2, 4);
  H_laser_ << 1, 0, 0, 0,
              0, 1, 0, 0;
  Hj_ = MatrixXd(3, 4);

  //measurement covariance matrix - laser
  R_laser_ << 0.0225, 0,
        0, 0.0225;

  //measurement covariance matrix - radar
  R_radar_ << 0.09, 0, 0,
        0, 0.0009, 0,
        0, 0, 0.09;
}

/**
* Destructor.
*/
FusionEKF::~FusionEKF() {}

void FusionEKF::ProcessMeasurement(const MeasurementPackage &measurement_pack) {


  /*****************************************************************************
   *  Initialization
   ****************************************************************************/
  if (!is_initialized_) {
    // first measurement
    cout << "EKF: " << endl;
    ekf_.x_ = VectorXd(4);
    //ekf_.x_ << 1, 1, 1, 1;

    //state covariance matrix P
    ekf_.P_ = MatrixXd(4, 4);
    ekf_.P_ << 1, 0, 0, 0,
            0, 1, 0, 0,
            0, 0, 1000, 0,
            0, 0, 0, 1000;

    if (measurement_pack.sensor_type_ == MeasurementPackage::RADAR) {
      /**
      Convert radar from polar to cartesian coordinates and initialize state.
      */
      float rho = measurement_pack.raw_measurements_[0];
      float cos_phi = cos(measurement_pack.raw_measurements_[1]);
      float sin_phi = sin(measurement_pack.raw_measurements_[1]);
      float rho_dot = measurement_pack.raw_measurements_[2];
      //x, y, vx, vy
      ekf_.x_ << rho*cos_phi, rho*sin_phi, rho_dot*cos_phi , rho_dot*sin_phi;
    }
    else if (measurement_pack.sensor_type_ == MeasurementPackage::LASER) {
      /**
      Initialize state.
      */
      ekf_.x_ << measurement_pack.raw_measurements_[0], measurement_pack.raw_measurements_[1], 0, 0;
      previous_timestamp_ = measurement_pack.timestamp_;
    }

    // done initializing, no need to predict or update
    is_initialized_ = true;
    return;
  }

  //compute the time elapsed between the current and previous measurements
  float dt = (measurement_pack.timestamp_ - previous_timestamp_) / 1000000.0;	//dt - expressed in seconds
  previous_timestamp_ = measurement_pack.timestamp_;

  /*****************************************************************************
   *  Prediction
   ****************************************************************************/

  /**
   TODO:
     * Update the state transition matrix F according to the new elapsed time.
      - Time is measured in seconds.
     * Update the process noise covariance matrix.
     * Use noise_ax = 9 and noise_ay = 9 for your Q matrix.
   */
  float noise_ax = 9;
  float noise_ay = 9;

  //Set the matrix state k-1 to k
  ekf_.F_(0, 2) = dt;
  ekf_.F_(1, 3) = dt;

  //Set the process covariance matrix Q
  ekf_.Q_ = MatrixXd(4, 4);

  double pow_dt44 = pow(dt, 4) / 4;
  double pow_dt32 = pow(dt,3) / 2;
  double pow_dt2 = pow(dt,2);

  ekf_.Q_ << pow_dt44 * noise_ax, 0, pow_dt32 * noise_ax, 0,
          0, pow_dt44 * noise_ay, 0, pow_dt32 * noise_ay,
          pow_dt32 * noise_ax, 0, pow_dt2 * noise_ax, 0,
          0, pow_dt32 * noise_ay, 0, pow_dt2 * noise_ay;

  ekf_.Predict();

  /*****************************************************************************
   *  Update
   ****************************************************************************/

  /**
   TODO:
     * Use the sensor type to perform the update step.
     * Update the state and covariance matrices.
   */

  if (measurement_pack.sensor_type_ == MeasurementPackage::RADAR) {
    //Hj_ (Jacobian H)
    Hj_ = tools.CalculateJacobian(ekf_.x_);
    ekf_.H_ = Hj_;
    //Noise matrix
    ekf_.R_ = R_radar_;
    // Radar updates
    ekf_.UpdateEKF(measurement_pack.raw_measurements_);
  } else {

    //State to measure
    ekf_.H_ = H_laser_;
    //Noise matrix
    ekf_.R_ = R_laser_;
    // Laser updates
    ekf_.Update(measurement_pack.raw_measurements_);
  }

  // print the output
  cout << "x_ = " << ekf_.x_ << endl;
  cout << "P_ = " << ekf_.P_ << endl;
}
