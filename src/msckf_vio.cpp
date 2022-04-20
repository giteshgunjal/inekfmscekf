/*
 * COPYRIGHT AND PERMISSION NOTICE
 * Penn Software MSCKF_VIO
 * Copyright (C) 2017 The Trustees of the University of Pennsylvania
 * All rights reserved.
 */

#include <iostream>
#include <iomanip>
#include <cmath>
#include <iterator>
#include <algorithm>

#include <Eigen/SVD>
#include <Eigen/QR>
#include <Eigen/SparseCore>
#include <Eigen/SPQRSupport>
#include <unsupported/Eigen/MatrixFunctions>
#include <boost/math/distributions/chi_squared.hpp>

#include <eigen_conversions/eigen_msg.h>
#include <tf_conversions/tf_eigen.h>
#include <sensor_msgs/PointCloud2.h>
#include <pcl_ros/point_cloud.h>
#include <pcl/point_types.h>

#include <msckf_vio/msckf_vio.h>
#include <msckf_vio/math_utils.hpp>
#include <msckf_vio/utils.h>

using namespace std;
using namespace Eigen;

namespace msckf_vio{
// Static member variables in IMUState class.
StateIDType IMUState::next_id = 0;
double IMUState::gyro_noise = 0.001;
double IMUState::acc_noise = 0.01;
double IMUState::gyro_bias_noise = 0.001;
double IMUState::acc_bias_noise = 0.01;
Vector3d IMUState::gravity = Vector3d(0, 0, -GRAVITY_ACCELERATION);
Isometry3d IMUState::T_imu_body = Isometry3d::Identity();

// Static member variables in CAMState class
Isometry3d CAMState::T_cam0_cam1 = Isometry3d::Identity();

// Static member variables in Feature class.
FeatureIDType Feature::next_id = 0;
double Feature::observation_noise = 0.01; // noise zeroed to debugg
Feature::OptimizationConfig Feature::optimization_config;

map<int, double> MsckfVio::chi_squared_test_table;

MsckfVio::MsckfVio(ros::NodeHandle& pnh):
  is_gravity_set(false),
  is_first_img(true),
  nh(pnh) {
  return;
}

bool MsckfVio::loadParameters() {
  // Frame id
  nh.param<string>("fixed_frame_id", fixed_frame_id, "world");
  nh.param<string>("child_frame_id", child_frame_id, "robot");
  nh.param<bool>("publish_tf", publish_tf, true);
  nh.param<double>("frame_rate", frame_rate, 40.0);
  nh.param<double>("position_std_threshold", position_std_threshold, 8.0);

  nh.param<double>("rotation_threshold", rotation_threshold, 0.2618);
  nh.param<double>("translation_threshold", translation_threshold, 0.4);
  nh.param<double>("tracking_rate_threshold", tracking_rate_threshold, 0.5);

  // Feature optimization parameters
  nh.param<double>("feature/config/translation_threshold",
      Feature::optimization_config.translation_threshold, 0.2);

  // Noise related parameters
  nh.param<double>("noise/gyro", IMUState::gyro_noise, 0.001);
  nh.param<double>("noise/acc", IMUState::acc_noise, 0.01);
  nh.param<double>("noise/gyro_bias", IMUState::gyro_bias_noise, 0.001);
  nh.param<double>("noise/acc_bias", IMUState::acc_bias_noise, 0.01);
  nh.param<double>("noise/feature", Feature::observation_noise, 0.01); // noise zeroed to debugg

  // Use variance instead of standard deviation.
  IMUState::gyro_noise *= IMUState::gyro_noise;
  IMUState::acc_noise *= IMUState::acc_noise;
  IMUState::gyro_bias_noise *= IMUState::gyro_bias_noise;
  IMUState::acc_bias_noise *= IMUState::acc_bias_noise;
  Feature::observation_noise *= Feature::observation_noise;

  // Set the initial IMU state.
  // The intial orientation and position will be set to the origin
  // implicitly. But the initial velocity and bias can be
  // set by parameters.
  // TODO: is it reasonable to set the initial bias to 0?
  nh.param<double>("initial_state/velocity/x",
      state_server.imu_state.velocity(0), 0.0);
  nh.param<double>("initial_state/velocity/y",
      state_server.imu_state.velocity(1), 0.0);
  nh.param<double>("initial_state/velocity/z",
      state_server.imu_state.velocity(2), 0.0);

  // The initial covariance of orientation and position can be
  // set to 0. But for velocity, bias and extrinsic parameters,
  // there should be nontrivial uncertainty.
  double gyro_bias_cov, acc_bias_cov, velocity_cov;
  nh.param<double>("initial_covariance/velocity",
      velocity_cov, 0.25);
  nh.param<double>("initial_covariance/gyro_bias",
      gyro_bias_cov, 1e-4);
  nh.param<double>("initial_covariance/acc_bias",
      acc_bias_cov, 1e-2);

  double extrinsic_rotation_cov, extrinsic_translation_cov;
  nh.param<double>("initial_covariance/extrinsic_rotation_cov",
      extrinsic_rotation_cov, 3.0462e-4);
  nh.param<double>("initial_covariance/extrinsic_translation_cov",
      extrinsic_translation_cov, 1e-4);

  //     state_server.state_cov = MatrixXd::Zero(21, 21);
  // for (int i = 3; i < 6; ++i)
  //   state_server.state_cov(i, i) = gyro_bias_cov;
  // for (int i = 6; i < 9; ++i)
  //   state_server.state_cov(i, i) = velocity_cov;
  // for (int i = 9; i < 12; ++i)
  //   state_server.state_cov(i, i) = acc_bias_cov;
  // for (int i = 15; i < 18; ++i)
  //   state_server.state_cov(i, i) = extrinsic_rotation_cov;
  // for (int i = 18; i < 21; ++i)
  //   state_server.state_cov(i, i) = extrinsic_translation_cov;

// New cov reprensta
  state_server.state_cov = MatrixXd::Zero(21, 21);
  for (int i = 3; i < 6; ++i)
    state_server.state_cov(i, i) = velocity_cov;
  for (int i = 9; i < 12; ++i)
    state_server.state_cov(i, i) = gyro_bias_cov;
  for (int i = 12; i < 15; ++i)
    state_server.state_cov(i, i) = acc_bias_cov;
  for (int i = 15; i < 18; ++i)
    state_server.state_cov(i, i) = extrinsic_rotation_cov;
  for (int i = 18; i < 21; ++i)
    state_server.state_cov(i, i) = extrinsic_translation_cov;

  // Transformation offsets between the frames involved.
  // test this
  Isometry3d T_imu_cam0 = utils::getTransformEigen(nh, "cam0/T_cam_imu");
  Isometry3d T_cam0_imu = T_imu_cam0.inverse();

  state_server.imu_state.R_imu_cam0 = T_cam0_imu.linear().transpose();
  state_server.imu_state.t_cam0_imu = T_cam0_imu.translation();
  CAMState::T_cam0_cam1 =
    utils::getTransformEigen(nh, "cam1/T_cn_cnm1");
  IMUState::T_imu_body =
    utils::getTransformEigen(nh, "T_imu_body").inverse();

  // Maximum number of camera states to be stored
  nh.param<int>("max_cam_state_size", max_cam_state_size, 30);

  ROS_INFO("===========================================");
  ROS_INFO("fixed frame id: %s", fixed_frame_id.c_str());
  ROS_INFO("child frame id: %s", child_frame_id.c_str());
  ROS_INFO("publish tf: %d", publish_tf);
  ROS_INFO("frame rate: %f", frame_rate);
  ROS_INFO("position std threshold: %f", position_std_threshold);
  ROS_INFO("Keyframe rotation threshold: %f", rotation_threshold);
  ROS_INFO("Keyframe translation threshold: %f", translation_threshold);
  ROS_INFO("Keyframe tracking rate threshold: %f", tracking_rate_threshold);
  ROS_INFO("gyro noise: %.10f", IMUState::gyro_noise);
  ROS_INFO("gyro bias noise: %.10f", IMUState::gyro_bias_noise);
  ROS_INFO("acc noise: %.10f", IMUState::acc_noise);
  ROS_INFO("acc bias noise: %.10f", IMUState::acc_bias_noise);
  ROS_INFO("observation noise: %.10f", Feature::observation_noise);
  ROS_INFO("initial velocity: %f, %f, %f",
      state_server.imu_state.velocity(0),
      state_server.imu_state.velocity(1),
      state_server.imu_state.velocity(2));
  ROS_INFO("initial gyro bias cov: %f", gyro_bias_cov);
  ROS_INFO("initial acc bias cov: %f", acc_bias_cov);
  ROS_INFO("initial velocity cov: %f", velocity_cov);
  ROS_INFO("initial extrinsic rotation cov: %f",
      extrinsic_rotation_cov);
  ROS_INFO("initial extrinsic translation cov: %f",
      extrinsic_translation_cov);

  cout << T_imu_cam0.linear() << endl;
  cout << T_imu_cam0.translation().transpose() << endl;

  ROS_INFO("max camera state #: %d", max_cam_state_size);
  ROS_INFO("===========================================");
  return true;
}

bool MsckfVio::createRosIO() {
  odom_pub = nh.advertise<nav_msgs::Odometry>("odom", 10);
  feature_pub = nh.advertise<sensor_msgs::PointCloud2>(
      "feature_point_cloud", 10);

  reset_srv = nh.advertiseService("reset",
      &MsckfVio::resetCallback, this);

  imu_sub = nh.subscribe("imu", 100,
      &MsckfVio::imuCallback, this);
  feature_sub = nh.subscribe("features", 40,
      &MsckfVio::featureCallback, this);

  mocap_odom_sub = nh.subscribe("mocap_odom", 10,
      &MsckfVio::mocapOdomCallback, this);
  mocap_odom_pub = nh.advertise<nav_msgs::Odometry>("gt_odom", 1);

  return true;
}

bool MsckfVio::initialize() {
  if (!loadParameters()) return false;
  ROS_INFO("Finish loading ROS parameters...");
 // Initialize state server
  // state_server.continuous_noise_cov =
  //   Matrix<double, 12, 12>::Zero();
  // state_server.continuous_noise_cov.block<3, 3>(0, 0) =
  //   Matrix3d::Identity()*IMUState::gyro_noise;
  // state_server.continuous_noise_cov.block<3, 3>(3, 3) =
  //   Matrix3d::Identity()*IMUState::gyro_bias_noise;
  // state_server.continuous_noise_cov.block<3, 3>(6, 6) =
  //   Matrix3d::Identity()*IMUState::acc_noise;
  // state_server.continuous_noise_cov.block<3, 3>(9, 9) =
  //   Matrix3d::Identity()*IMUState::acc_bias_noise;

  // New cov representation
  // Initialize state server
  state_server.continuous_noise_cov =
    Matrix<double, 12, 12>::Zero();
  state_server.continuous_noise_cov.block<3, 3>(0, 0) =
    Matrix3d::Identity()*IMUState::gyro_noise;
  state_server.continuous_noise_cov.block<3, 3>(3, 3) =
    Matrix3d::Identity()*IMUState::acc_noise; 
  state_server.continuous_noise_cov.block<3, 3>(6, 6) =
    Matrix3d::Identity()*IMUState::gyro_bias_noise;
  state_server.continuous_noise_cov.block<3, 3>(9, 9) =
    Matrix3d::Identity()*IMUState::acc_bias_noise;

  // Initialize the chi squared test table with confidence
  // level 0.95.
  for (int i = 1; i < 100; ++i) {
    boost::math::chi_squared chi_squared_dist(i);
    chi_squared_test_table[i] =
      boost::math::quantile(chi_squared_dist, 0.05);
  }

  if (!createRosIO()) return false;
  ROS_INFO("Finish creating ROS IO...");

  return true;
}

void MsckfVio::imuCallback(
    const sensor_msgs::ImuConstPtr& msg) {

  // IMU msgs are pushed backed into a buffer instead of
  // being processed immediately. The IMU msgs are processed
  // when the next image is available, in which way, we can
  // easily handle the transfer delay.
  imu_msg_buffer.push_back(*msg);

  if (!is_gravity_set) {
    if (imu_msg_buffer.size() < 200) return;
    //if (imu_msg_buffer.size() < 10) return;
    initializeGravityAndBias();
    is_gravity_set = true;
  }

  return;
}

void MsckfVio::initializeGravityAndBias() {

  // Initialize gravity and gyro bias.
  Vector3d sum_angular_vel = Vector3d::Zero();
  Vector3d sum_linear_acc = Vector3d::Zero();

  for (const auto& imu_msg : imu_msg_buffer) {
    Vector3d angular_vel = Vector3d::Zero();
    Vector3d linear_acc = Vector3d::Zero();

    tf::vectorMsgToEigen(imu_msg.angular_velocity, angular_vel);
    tf::vectorMsgToEigen(imu_msg.linear_acceleration, linear_acc);

    sum_angular_vel += angular_vel;
    sum_linear_acc += linear_acc;
  }

  state_server.imu_state.gyro_bias =
    sum_angular_vel / imu_msg_buffer.size();
  //IMUState::gravity =
  //  -sum_linear_acc / imu_msg_buffer.size();
  // This is the gravity in the IMU frame.
  Vector3d gravity_imu =
    sum_linear_acc / imu_msg_buffer.size();

  ROS_INFO_STREAM("world gravity"<<
      IMUState::gravity);

  ROS_INFO_STREAM("imu gravity"<<
      gravity_imu);
  ROS_INFO_STREAM("imu gravitynorm"<<
      gravity_imu.norm());
  

    

//   // Initialize the initial orientation, so that the estimation
//   // is consistent with the inertial frame.
//   double gravity_norm = gravity_imu.norm();
//   Vector3d gravity_world = Vector3d(0.0, 0.0, -gravity_norm);


// // in world
//      Quaterniond q0_i_w = Quaterniond::FromTwoVectors(
//     gravity_imu, -gravity_world);
//   state_server.imu_state.orientation = 
//     rotationToQuaternion(q0_i_w.toRotationMatrix().transpose());

//   Vector3d gravity_imu_actual = quaternionToRotation(state_server.imu_state.orientation)* IMUState::gravity;

//   ROS_INFO_STREAM("imu gravityactual"<<
//       gravity_imu_actual);

//   IMUState::gravity = gravity_world;
  
//   // state_server.imu_state.acc_bias = -(gravity_imu + gravity_imu_actual);

//   ROS_INFO_STREAM("bias ini"<<
//       state_server.imu_state.acc_bias);




  // Initialize the initial orientation, so that the estimation
  // is consistent with the inertial frame.

  Vector3d avgacc = gravity_imu;
  double gravity_norm = gravity_imu.norm();
  Vector3d gravity_world = IMUState::gravity ;



  // IMUState::gravity = gravity_world;
  // Quaterniond q0_i_w = Quaterniond::FromTwoVectors(
  //   gravity_imu, -IMUState::gravity);
  // state_server.imu_state.orientation =
  //   rotationToQuaternion(q0_i_w.toRotationMatrix().transpose());

// in world
     Quaterniond q0_i_w = Quaterniond::FromTwoVectors(
    gravity_imu, -gravity_world);
  state_server.imu_state.orientation = 
    rotationToQuaternion(q0_i_w.toRotationMatrix().transpose());

    ROS_INFO_STREAM("orient"<<
      state_server.imu_state.orientation);

    Vector3d world_acc = quaternionToRotation(state_server.imu_state.orientation).transpose()*avgacc;

  Vector3d gravity_imu_actual = quaternionToRotation(state_server.imu_state.orientation)* IMUState::gravity;

  // state_server.imu_state.acc_bias = quaternionToRotation(state_server.imu_state.orientation)*(world_acc - IMUState::gravity

  ROS_INFO_STREAM("imu gravityactual"<<
      gravity_imu_actual);

  ROS_INFO_STREAM("worldacc"<<
      world_acc);

  // IMUState::gravity = Vector3d(0.0, 0.0, -gravity_norm);
  
  state_server.imu_state.acc_bias = (avgacc +gravity_imu_actual);

  ROS_INFO_STREAM("bias ini"<<
      state_server.imu_state.acc_bias);

  return;
}

bool MsckfVio::resetCallback(
    std_srvs::Trigger::Request& req,
    std_srvs::Trigger::Response& res) {

  ROS_WARN("Start resetting msckf vio...");
  // Temporarily shutdown the subscribers to prevent the
  // state from updating.
  feature_sub.shutdown();
  imu_sub.shutdown();

  // Reset the IMU state.
  IMUState& imu_state = state_server.imu_state;
  imu_state.time = 0.0;
  imu_state.orientation = Vector4d(0.0, 0.0, 0.0, 1.0);
  imu_state.position = Vector3d::Zero();
  imu_state.velocity = Vector3d::Zero();
  imu_state.gyro_bias = Vector3d::Zero();
  imu_state.acc_bias = Vector3d::Zero();
  imu_state.orientation_null = Vector4d(0.0, 0.0, 0.0, 1.0);
  imu_state.position_null = Vector3d::Zero();
  imu_state.velocity_null = Vector3d::Zero();

  // Remove all existing camera states.
  state_server.cam_states.clear();

  // Reset the state covariance.
  double gyro_bias_cov, acc_bias_cov, velocity_cov;
  nh.param<double>("initial_covariance/velocity",
      velocity_cov, 0.25);
  nh.param<double>("initial_covariance/gyro_bias",
      gyro_bias_cov, 1e-4);
  nh.param<double>("initial_covariance/acc_bias",
      acc_bias_cov, 0.01);

  double extrinsic_rotation_cov, extrinsic_translation_cov;
  nh.param<double>("initial_covariance/extrinsic_rotation_cov",
      extrinsic_rotation_cov, 3.0462e-4);
  nh.param<double>("initial_covariance/extrinsic_translation_cov",
      extrinsic_translation_cov, 1e-4);

  //   state_server.state_cov = MatrixXd::Zero(21, 21);
  // for (int i = 3; i < 6; ++i)
  //   state_server.state_cov(i, i) = gyro_bias_cov;
  // for (int i = 6; i < 9; ++i)
  //   state_server.state_cov(i, i) = velocity_cov;
  // for (int i = 9; i < 12; ++i)
  //   state_server.state_cov(i, i) = acc_bias_cov;
  // for (int i = 15; i < 18; ++i)
  //   state_server.state_cov(i, i) = extrinsic_rotation_cov;
  // for (int i = 18; i < 21; ++i)
  //   state_server.state_cov(i, i) = extrinsic_translation_cov;

    // new cov repe

  state_server.state_cov = MatrixXd::Zero(21, 21);
  for (int i = 3; i < 6; ++i)
    state_server.state_cov(i, i) = velocity_cov; 
  for (int i = 9; i < 12; ++i)
    state_server.state_cov(i, i) = gyro_bias_cov;
  for (int i = 12; i < 15; ++i)
    state_server.state_cov(i, i) = acc_bias_cov;
  for (int i = 15; i < 18; ++i)
    state_server.state_cov(i, i) = extrinsic_rotation_cov;
  for (int i = 18; i < 21; ++i)
    state_server.state_cov(i, i) = extrinsic_translation_cov;

  // Clear all exsiting features in the map.
  map_server.clear();

  // Clear the IMU msg buffer.
  imu_msg_buffer.clear();

  // Reset the starting flags.
  is_gravity_set = false;
  is_first_img = true;

  // Restart the subscribers.
  imu_sub = nh.subscribe("imu", 100,
      &MsckfVio::imuCallback, this);
  feature_sub = nh.subscribe("features", 40,
      &MsckfVio::featureCallback, this);

  // TODO: When can the reset fail?
  res.success = true;
  ROS_WARN("Resetting msckf vio completed...");
  return true;
}

void MsckfVio::featureCallback(
    const CameraMeasurementConstPtr& msg) {

  // Return if the gravity vector has not been set.
  if (!is_gravity_set) return;

  // Start the system if the first image is received.
  // The frame where the first image is received will be
  // the origin.
  if (is_first_img) {
    is_first_img = false;
    state_server.imu_state.time = msg->header.stamp.toSec();
  }

  static double max_processing_time = 0.0;
  static int critical_time_cntr = 0;
  double processing_start_time = ros::Time::now().toSec();

  // Propogate the IMU state.
  // that are received before the image msg.
  ros::Time start_time = ros::Time::now();
  batchImuProcessing(msg->header.stamp.toSec());
  double imu_processing_time = (
      ros::Time::now()-start_time).toSec();

  // Augment the state vector.
  start_time = ros::Time::now();
  stateAugmentation(msg->header.stamp.toSec());
  double state_augmentation_time = (
      ros::Time::now()-start_time).toSec();

  // Add new observations for existing features or new
  // features in the map server.
  start_time = ros::Time::now();
  addFeatureObservations(msg);
  double add_observations_time = (
      ros::Time::now()-start_time).toSec();

  // Perform measurement update if necessary.
  start_time = ros::Time::now();
  removeLostFeatures();
  double remove_lost_features_time = (
      ros::Time::now()-start_time).toSec();

  start_time = ros::Time::now();
  pruneCamStateBuffer();
  double prune_cam_states_time = (
      ros::Time::now()-start_time).toSec();

  // Publish the odometry.
  start_time = ros::Time::now();
  publish(msg->header.stamp);
  double publish_time = (
      ros::Time::now()-start_time).toSec();

  // Reset the system if necessary.
  onlineReset();

  double processing_end_time = ros::Time::now().toSec();
  double processing_time =
    processing_end_time - processing_start_time;
  if (processing_time > 1.0/frame_rate) {
    ++critical_time_cntr;
    ROS_INFO_STREAM("PRocessing time"<<
        processing_time << critical_time_cntr);
    //printf("IMU processing time: %f/%f\n",
    //    imu_processing_time, imu_processing_time/processing_time);
    //printf("State augmentation time: %f/%f\n",
    //    state_augmentation_time, state_augmentation_time/processing_time);
    //printf("Add observations time: %f/%f\n",
    //    add_observations_time, add_observations_time/processing_time);
    printf("Remove lost features time: %f/%f\n",
        remove_lost_features_time, remove_lost_features_time/processing_time);
    printf("Remove camera states time: %f/%f\n",
        prune_cam_states_time, prune_cam_states_time/processing_time);
    //printf("Publish time: %f/%f\n",
    //    publish_time, publish_time/processing_time);
  }

  return;
}

void MsckfVio::mocapOdomCallback(
    const nav_msgs::OdometryConstPtr& msg) {
  static bool first_mocap_odom_msg = true;

  // If this is the first mocap odometry messsage, set
  // the initial frame.
  if (first_mocap_odom_msg) {
    Quaterniond orientation;
    Vector3d translation;
    tf::pointMsgToEigen(
        msg->pose.pose.position, translation);
    tf::quaternionMsgToEigen(
        msg->pose.pose.orientation, orientation);
    //tf::vectorMsgToEigen(
    //    msg->transform.translation, translation);
    //tf::quaternionMsgToEigen(
    //    msg->transform.rotation, orientation);
    mocap_initial_frame.linear() = orientation.toRotationMatrix();
    mocap_initial_frame.translation() = translation;
    first_mocap_odom_msg = false;
  }

  // Transform the ground truth.
  Quaterniond orientation;
  Vector3d translation;
  //tf::vectorMsgToEigen(
  //    msg->transform.translation, translation);
  //tf::quaternionMsgToEigen(
  //    msg->transform.rotation, orientation);
  tf::pointMsgToEigen(
      msg->pose.pose.position, translation);
  tf::quaternionMsgToEigen(
      msg->pose.pose.orientation, orientation);

  Eigen::Isometry3d T_b_v_gt;
  T_b_v_gt.linear() = orientation.toRotationMatrix();
  T_b_v_gt.translation() = translation;
  Eigen::Isometry3d T_b_w_gt = mocap_initial_frame.inverse() * T_b_v_gt;

  //Eigen::Vector3d body_velocity_gt;
  //tf::vectorMsgToEigen(msg->twist.twist.linear, body_velocity_gt);
  //body_velocity_gt = mocap_initial_frame.linear().transpose() *
  //  body_velocity_gt;

  // Ground truth tf.
  if (publish_tf) {
    tf::Transform T_b_w_gt_tf;
    tf::transformEigenToTF(T_b_w_gt, T_b_w_gt_tf);
    tf_pub.sendTransform(tf::StampedTransform(
          T_b_w_gt_tf, msg->header.stamp, fixed_frame_id, child_frame_id+"_mocap"));
  }

  // Ground truth odometry.
  nav_msgs::Odometry mocap_odom_msg;
  mocap_odom_msg.header.stamp = msg->header.stamp;
  mocap_odom_msg.header.frame_id = fixed_frame_id;
  mocap_odom_msg.child_frame_id = child_frame_id+"_mocap";

  tf::poseEigenToMsg(T_b_w_gt, mocap_odom_msg.pose.pose);
  //tf::vectorEigenToMsg(body_velocity_gt,
  //    mocap_odom_msg.twist.twist.linear);

  mocap_odom_pub.publish(mocap_odom_msg);
  return;
}

void MsckfVio::batchImuProcessing(const double& time_bound) {
  // Counter how many IMU msgs in the buffer are used.
  int used_imu_msg_cntr = 0;

  for (const auto& imu_msg : imu_msg_buffer) {
    double imu_time = imu_msg.header.stamp.toSec();
    if (imu_time < state_server.imu_state.time) {
      ++used_imu_msg_cntr;
      continue;
    }
    if (imu_time > time_bound) break;

    // Convert the msgs.
    Vector3d m_gyro, m_acc;
    tf::vectorMsgToEigen(imu_msg.angular_velocity, m_gyro);
    tf::vectorMsgToEigen(imu_msg.linear_acceleration, m_acc);

    // Execute process model.
    processModel(imu_time, m_gyro, m_acc);
    ++used_imu_msg_cntr;
  }

  // Set the state ID for the new IMU state.
  state_server.imu_state.id = IMUState::next_id++;

  // Remove all used IMU msgs.
  imu_msg_buffer.erase(imu_msg_buffer.begin(),
      imu_msg_buffer.begin()+used_imu_msg_cntr);

  return;
}



void MsckfVio::processModel(const double& time,
    const Vector3d& m_gyro,
    const Vector3d& m_acc) {

  // Remove the bias from the measured gyro and acceleration
  IMUState& imu_state = state_server.imu_state;
  Vector3d gyro = m_gyro - imu_state.gyro_bias;
  Vector3d acc = m_acc - imu_state.acc_bias;
  double dtime = time - imu_state.time;

// ROS_INFO_STREAM("measured acc"<<
//       m_acc);
// ROS_INFO_STREAM("accbias"<<
//       imu_state.acc_bias);
  // Compute discrete transition and noise covariance matrix
  Matrix<double, 21, 21> F_I = Matrix<double, 21, 21>::Zero();
  Matrix<double, 21, 12> G_I = Matrix<double, 21, 12>::Zero();

 
  F_I.block<3, 3>(0, 9) = -quaternionToRotation(
      imu_state.orientation).transpose();
  F_I.block<3, 3>(3, 0) = skewSymmetric(IMUState::gravity);
  F_I.block<3, 3>(3, 9) = -skewSymmetric(imu_state.velocity)*quaternionToRotation(
      imu_state.orientation).transpose();
  F_I.block<3, 3>(3, 12) = -quaternionToRotation(
      imu_state.orientation).transpose();
  F_I.block<3, 3>(6, 9) = -skewSymmetric(imu_state.position)*quaternionToRotation(
      imu_state.orientation).transpose();
  F_I.block<3, 3>(6, 3) =  Matrix3d::Identity();



  G_I.block<3, 3>(0, 0) =quaternionToRotation(
      imu_state.orientation).transpose();
  G_I.block<3, 3>(3, 3) = quaternionToRotation(
      imu_state.orientation).transpose();
  G_I.block<3, 3>(3, 0) = skewSymmetric(imu_state.velocity)*quaternionToRotation(
      imu_state.orientation).transpose();
  G_I.block<3, 3>(6, 0) = skewSymmetric(imu_state.position)*quaternionToRotation(
      imu_state.orientation).transpose();
  G_I.block<3, 3>(9, 6) = Matrix3d::Identity();
  G_I.block<3, 3>(12, 9) = Matrix3d::Identity();


  // Approximate matrix exponential to the 3rd order,
  // which can be considered to be accurate enough assuming
  // dtime is within 0.01s.
  Matrix<double, 21, 21>Phi_I = (F_I*dtime).exp();





  predictNewState(dtime, gyro, acc);

  

  // Propogate the state covariance matrix.
  Matrix<double, 21, 21> Q = Phi_I*G_I*state_server.continuous_noise_cov*
    G_I.transpose()*Phi_I.transpose()*dtime;
  state_server.state_cov.block<21, 21>(0, 0) =
    Phi_I*state_server.state_cov.block<21, 21>(0, 0)*Phi_I.transpose() + Q;

  // cout<< "state cov before aug"<< state_server.state_cov << endl;



  if (state_server.cam_states.size() > 0) {
    state_server.state_cov.block(
        0, 21, 21, state_server.state_cov.cols()-21) =
      Phi_I * state_server.state_cov.block(
        0, 21, 21, state_server.state_cov.cols()-21);
    state_server.state_cov.block(
        21, 0, state_server.state_cov.rows()-21, 21) =
      state_server.state_cov.block(
        21, 0, state_server.state_cov.rows()-21, 21) * Phi_I.transpose();
  }
  // cout<< "state cov after aug"<< state_server.state_cov<< endl;

  MatrixXd state_cov_fixed = (state_server.state_cov +
      state_server.state_cov.transpose()) / 2.0;
  state_server.state_cov = state_cov_fixed;

  // Update the state correspondes to null space.
  imu_state.orientation_null = imu_state.orientation;
  imu_state.position_null = imu_state.position;
  imu_state.velocity_null = imu_state.velocity;

  // Update the state info
  state_server.imu_state.time = time;
  return;
}

void MsckfVio::predictNewState(const double& dt,
    const Vector3d& gyro,
    const Vector3d& acc) {




  Vector4d& q = state_server.imu_state.orientation;
  Vector3d& v = state_server.imu_state.velocity;
  Vector3d& p = state_server.imu_state.position;


// // Some pre-calculation
//   Vector4d dq_dt, dq_d
//   Vector3d k3_v_dot = dR_dt2_transpose*acc +
//     IMUState::gravity;
//   Vector3d k3_p_dot = k2_v;

//   // k4 = f(tn+dt, yn+k3*dt)
//   Vector3d k3_v = v + k3_v_dot*dt;
//   Vector3d k4_v_dot = dR_dt_transpose*acc +
//     IMUState::gravity;
//   Vector3d k4_p_dot = k3_v;

//   // yn+1 = yn + dt/6*(k1+2*k2+2*k3+k4)
//   Vector4d q_P = dq_dt;
//   quaternionNormalize(q_P);
//   Vector3d v_P= v + dt/6*(k1_v_dot+2*k2_v_dot+2*k3_v_dot+k4_v_dot);
//   Vector3d p_P = p + dt/6*(k1_p_dot+2*k2_p_dot+2*k3_p_dot+k4_p_dot);


//   // Vector4d dq_dt, dq_dt2;
//   // if (gyro_norm > 1e-5) {
//   //   dq_dt = (cos(gyro_norm*dt*0.5)*Matrix4d::Identity() +
//   //     1/gyro_norm*sin(gyro_norm*dt*0.5)*Omega) * q;
//   //   dq_dt2 = (cos(gyro_norm*dt*0.25)*Matrix4d::Identity() +
//   //     1/gyro_norm*sin(gyro_norm*dt*0.25)*Omega) * q;
//   // }
//   // else {
//   //   dq_dt = (Matrix4d::Identity()+0.5*dt*Omega) *
//   //     cos(gyro_norm*dt*0.5) * q;
//   //   dq_dt2 = (Matrix4d::Identity()+0.25*dt*Omega) *
//   //     cos(gyro_norm*dt*0.25) * q;
//   // }


  Matrix3d R = quaternionToRotation(q).transpose();
  Matrix3d R_pred = R*Exp_SO3(gyro*dt);
  Vector3d v_pred = v + (R*acc + IMUState::gravity)*dt;
  Vector3d p_pred = p + v*dt + 0.5*(R*acc + IMUState::gravity)*dt*dt;

    
//Set state
  
  Vector4d q_pred = rotationToQuaternion(R_pred.transpose());
  quaternionNormalize(q_pred);

  q = q_pred;
  quaternionNormalize(q);
  v = v_pred;
  p = p_pred;

// ROS_INFO_STREAM("State_inv"<<
//       q_pred<<v_pred<<p_pred);
// ROS_INFO_STREAM("State_rungk:"<<
      // q_P<<v_P<<p_P);

// State does get updated some how
  // state_server.imu_state.orientation = q;
  // state_server.imu_state.velocity = v;
  // state_server.imu_state.position= p;
  // ROS_INFO_STREAM("State_now:"<<
  //     state_server.imu_state.orientation<<state_server.imu_state.velocity<< state_server.imu_state.position);

  // ROS_INFO_STREAM("State_tobe"<<
  //     q<<v<<p);
  // state_server.imu_state.orientation = Vector4d(0.0, 0.0, 0.0, 1.0);
  // state_server.imu_state.position = Vector3d::Zero();
  // state_server.imu_state.velocity = Vector3d::Zero();

  return;
}

void MsckfVio::stateAugmentation(const double& time) {

  const Matrix3d& R_i_c = state_server.imu_state.R_imu_cam0;
  const Vector3d& t_c_i = state_server.imu_state.t_cam0_imu;

  // Add a new camera state to the state server.
  //frame changed to global 
  Matrix3d R_w_i = quaternionToRotation(
      state_server.imu_state.orientation);
  Matrix3d R_w_c = R_i_c * R_w_i;
//  Orientation convention changed form camera to global to global to camera for rinv measurement step
  Matrix3d R_c_w = R_w_c.transpose();


  Vector3d t_c_w = state_server.imu_state.position +
    R_w_i.transpose()*t_c_i;

  state_server.cam_states[state_server.imu_state.id] =
    CAMState(state_server.imu_state.id);
  CAMState& cam_state = state_server.cam_states[
    state_server.imu_state.id];

  cam_state.time = time;
  cam_state.orientation = rotationToQuaternion(R_w_c);
  // cam_state.orientation = rotationToQuaternion(R_c_w);
  cam_state.position = t_c_w;

  cam_state.orientation_null = cam_state.orientation;
  cam_state.position_null = cam_state.position;

//  New jacobain according to st mscekf to inv paper
// watch out fot iR what is that typo?

  Matrix<double, 6, 21> J = Matrix<double, 6, 21>::Zero();
  J.block<3, 3>(0, 0) = Matrix3d::Identity();
  J.block<3, 3>(0, 15) = R_w_i.transpose();
  J.block<3, 3>(3, 6) = Matrix3d::Identity();
  J.block<3, 3>(3, 15) = skewSymmetric(state_server.imu_state.position)*R_w_i.transpose();
  J.block<3, 3>(3,  18) = R_w_i.transpose();



//  kept the same as before check if something goes wrong
  // Resize the state covariance matrix.
  size_t old_rows = state_server.state_cov.rows();
  size_t old_cols = state_server.state_cov.cols();
  state_server.state_cov.conservativeResize(old_rows+6, old_cols+6);

  // Rename some matrix blocks for convenience.
  const Matrix<double, 21, 21>& P11 =
    state_server.state_cov.block<21, 21>(0, 0);
  const MatrixXd& P12 =
    state_server.state_cov.block(0, 21, 21, old_cols-21);

  // Fill in the augmented state covariance.
  state_server.state_cov.block(old_rows, 0, 6, old_cols) << J*P11, J*P12;
  state_server.state_cov.block(0, old_cols, old_rows, 6) =
    state_server.state_cov.block(old_rows, 0, 6, old_cols).transpose();
  state_server.state_cov.block<6, 6>(old_rows, old_cols) =
    J * P11 * J.transpose();

  // Fix the covariance to be symmetric
  MatrixXd state_cov_fixed = (state_server.state_cov +
      state_server.state_cov.transpose()) / 2.0;
  state_server.state_cov = state_cov_fixed;

  return;
}

void MsckfVio::addFeatureObservations(
    const CameraMeasurementConstPtr& msg) {

  StateIDType state_id = state_server.imu_state.id;
  int curr_feature_num = map_server.size();
  int tracked_feature_num = 0;

  // Add new observations for existing features or new
  // features in the map server.
  for (const auto& feature : msg->features) {
    if (map_server.find(feature.id) == map_server.end()) {
      // This is a new feature.
      map_server[feature.id] = Feature(feature.id);
      map_server[feature.id].observations[state_id] =
        Vector4d(feature.u0, feature.v0,
            feature.u1, feature.v1);
    } else {
      // This is an old feature.
      map_server[feature.id].observations[state_id] =
        Vector4d(feature.u0, feature.v0,
            feature.u1, feature.v1);
      ++tracked_feature_num;
    }
  }

  tracking_rate =
    static_cast<double>(tracked_feature_num) /
    static_cast<double>(curr_feature_num);

  return;
}

void MsckfVio::measurementJacobian(
    const StateIDType& cam_state_id,
    const FeatureIDType& feature_id,
    Matrix<double, 4, 6>& H_x, Matrix<double, 4, 3>& H_f, Vector4d& r) {

  // Prepare all the required data.
  const CAMState& cam_state = state_server.cam_states[cam_state_id];
  const Feature& feature = map_server[feature_id];
  // const N = state_server.cam_states.size();

  // Cam0 pose.
  //  cam orientation changed in state aug/
  //  check if actually neded
  // Matrix3d R_w_c0 = quaternionToRotation(cam_state.orientation);
  Matrix3d R_w_c0 = quaternionToRotation(cam_state.orientation);
  const Vector3d& t_c0_w = cam_state.position;

  // Cam1 pose.
  Matrix3d R_c0_c1 = CAMState::T_cam0_cam1.linear();


  Matrix3d R_w_c1 = CAMState::T_cam0_cam1.linear() * R_w_c0;
  Vector3d t_c1_w = t_c0_w - R_w_c1.transpose()*CAMState::T_cam0_cam1.translation();

  // 3d feature position in the world frame.
  // And its observation with the stereo cameras.
  const Vector3d& p_w = feature.position;
  const Vector4d& z = feature.observations.find(cam_state_id)->second;

  // Convert the feature position from the world frame to
  // the cam0 and cam1 frame.
  Vector3d p_c0 = R_w_c0 * (p_w-t_c0_w);
  Vector3d p_c1 = R_w_c1 * (p_w-t_c1_w);

  // Compute the Jacobians.
  Matrix<double, 4, 3> dz_dpc0 = Matrix<double, 4, 3>::Zero();
  dz_dpc0(0, 0) = 1 / p_c0(2);
  dz_dpc0(1, 1) = 1 / p_c0(2);
  dz_dpc0(0, 2) = -p_c0(0) / (p_c0(2)*p_c0(2));
  dz_dpc0(1, 2) = -p_c0(1) / (p_c0(2)*p_c0(2));

  Matrix<double, 4, 3> dz_dpc1 = Matrix<double, 4, 3>::Zero();
  dz_dpc1(2, 0) = 1 / p_c1(2);
  dz_dpc1(3, 1) = 1 / p_c1(2);
  dz_dpc1(2, 2) = -p_c1(0) / (p_c1(2)*p_c1(2));
  dz_dpc1(3, 2) = -p_c1(1) / (p_c1(2)*p_c1(2));

  // Matrix<double, 3, 6> dpc0_dxc = Matrix<double, 3, 6>::Zero();
  // dpc0_dxc.leftCols(3) = skewSymmetric(p_c0);
  // dpc0_dxc.rightCols(3) = -R_w_c0;

  //  changed as per inekf
  Matrix<double, 3, 6> dpc0_dtwc0 = Matrix<double, 3, 6>::Zero();
  Matrix<double, 3, 6> tempd0 = Matrix<double,  3, 6>::Zero();
  tempd0.block<3, 3>(0, 0)  = skewSymmetric(p_w-t_c0_w);
  tempd0.block<3, 3>(0, 3)  = -Matrix3d::Identity();
  dpc0_dtwc0 = R_w_c0*tempd0;

  MatrixXd dpc1_dtwc0;
  dpc1_dtwc0 = R_c0_c1*dpc0_dtwc0;
  

  // Matrix<double, 3, 6> dpc1_dxc = Matrix<double, 3, 6>::Zero();
  // dpc1_dxc.leftCols(3) = R_c0_c1 * skewSymmetric(p_c0);
  // dpc1_dxc.rightCols(3) = -R_w_c1;

  Matrix3d dpc0_dpg = R_w_c0;
  Matrix3d dpc1_dpg = R_w_c1;

  H_x = dz_dpc0*dpc0_dtwc0 + dz_dpc1*dpc1_dtwc0;
  H_f = dz_dpc0*dpc0_dpg + dz_dpc1*dpc1_dpg;

  //  // Modifty the measurement Jacobian to ensure
  // // observability constrain.
  // Matrix<double, 4, 6> A = H_x;
  // Matrix<double, 6, 1> u = Matrix<double, 6, 1>::Zero();
  // u.block<3, 1>(0, 0) = quaternionToRotation(
  //     cam_state.orientation_null) * IMUState::gravity;
  // u.block<3, 1>(3, 0) = skewSymmetric(
  //     p_w-cam_state.position_null) * IMUState::gravity;
  // H_x = A - A*u*(u.transpose()*u).inverse()*u.transpose();
  // H_f = -H_x.block<4, 3>(0, 3);

  // Compute the residual.
  r = z - Vector4d(p_c0(0)/p_c0(2), p_c0(1)/p_c0(2),
      p_c1(0)/p_c1(2), p_c1(1)/p_c1(2));

  return;
}

void MsckfVio::featureJacobian(
    const FeatureIDType& feature_id,
    const std::vector<StateIDType>& cam_state_ids,
    MatrixXd& H_x, VectorXd& r) {

  const auto& feature = map_server[feature_id];

  // Check how many camera states in the provided camera
  // id camera has actually seen this feature.
  vector<StateIDType> valid_cam_state_ids(0);
  for (const auto& cam_id : cam_state_ids) {
    if (feature.observations.find(cam_id) ==
        feature.observations.end()) continue;

    valid_cam_state_ids.push_back(cam_id);
  }

  int jacobian_row_size = 0;
  jacobian_row_size = 4 * valid_cam_state_ids.size();

  MatrixXd H_xj = MatrixXd::Zero(jacobian_row_size,
      21+state_server.cam_states.size()*6);
  MatrixXd H_fj = MatrixXd::Zero(jacobian_row_size, 3);
  VectorXd r_j = VectorXd::Zero(jacobian_row_size);
  int stack_cntr = 0;

  for (const auto& cam_id : valid_cam_state_ids) {

    Matrix<double, 4, 6> H_xi = Matrix<double, 4, 6>::Zero();
    Matrix<double, 4, 3> H_fi = Matrix<double, 4, 3>::Zero();
    Vector4d r_i = Vector4d::Zero();
    measurementJacobian(cam_id, feature.id, H_xi, H_fi, r_i);

    auto cam_state_iter = state_server.cam_states.find(cam_id);
    int cam_state_cntr = std::distance(
        state_server.cam_states.begin(), cam_state_iter);

    // Stack the Jacobians.
    H_xj.block<4, 6>(stack_cntr, 21+6*cam_state_cntr) = H_xi;
    H_fj.block<4, 3>(stack_cntr, 0) = H_fi;
    r_j.segment<4>(stack_cntr) = r_i;
    stack_cntr += 4;
  }

  // Project the residual and Jacobians onto the nullspace
  // of H_fj.
  JacobiSVD<MatrixXd> svd_helper(H_fj, ComputeFullU | ComputeThinV);
  MatrixXd A = svd_helper.matrixU().rightCols(
      jacobian_row_size - 3);

  H_x = A.transpose() * H_xj;
  r = A.transpose() * r_j;

  return;
}

void MsckfVio::measurementUpdate(
    const MatrixXd& H, const VectorXd& r) {

  if (H.rows() == 0 || r.rows() == 0) return;

  // Decompose the final Jacobian matrix to reduce computational
  // complexity as in Equation (28), (29).
  MatrixXd H_thin;
  VectorXd r_thin;

  if (H.rows() > H.cols()) {
    // Convert H to a sparse matrix.
    SparseMatrix<double> H_sparse = H.sparseView();

    // Perform QR decompostion on H_sparse.
    SPQR<SparseMatrix<double> > spqr_helper;
    spqr_helper.setSPQROrdering(SPQR_ORDERING_NATURAL);
    spqr_helper.compute(H_sparse);

    MatrixXd H_temp;
    VectorXd r_temp;
    (spqr_helper.matrixQ().transpose() * H).evalTo(H_temp);
    (spqr_helper.matrixQ().transpose() * r).evalTo(r_temp);

    H_thin = H_temp.topRows(21+state_server.cam_states.size()*6);
    r_thin = r_temp.head(21+state_server.cam_states.size()*6);

    //HouseholderQR<MatrixXd> qr_helper(H);
    //MatrixXd Q = qr_helper.householderQ();
    //MatrixXd Q1 = Q.leftCols(21+state_server.cam_states.size()*6);

    //H_thin = Q1.transpose() * H;
    //r_thin = Q1.transpose() * r;
  } else {
    H_thin = H;
    r_thin = r;
  }





  // Compute the Kalman gain.
  const MatrixXd& P = state_server.state_cov;
  MatrixXd S = H_thin*P*H_thin.transpose() +
      Feature::observation_noise*MatrixXd::Identity(
        H_thin.rows(), H_thin.rows());
  //MatrixXd K_transpose = S.fullPivHouseholderQr().solve(H_thin*P);
  MatrixXd K_transpose = S.ldlt().solve(H_thin*P);
  MatrixXd K = K_transpose.transpose();

// ROS_INFO_STREAM("State_pred"<<
//         H_thin.rows());
  // Compute the error of the state.
  VectorXd delta_x = K * r_thin;

  // Update state covariance.
  MatrixXd I_KH = MatrixXd::Identity(K.rows(), H_thin.cols()) - K*H_thin;
  //state_server.state_cov = I_KH*state_server.state_cov*I_KH.transpose() +
  //  K*K.transpose()*Feature::observation_noise;
  // state_server.state_cov = I_KH*state_server.state_cov*I_KH.transpose() + K*Feature::observation_noise*MatrixXd::Identity(
  //       H_thin.rows(), H_thin.rows())*K.transpose() ;
  state_server.state_cov = I_KH*state_server.state_cov;

  // Fix the covariance to be symmetric
  MatrixXd state_cov_fixed = (state_server.state_cov + 
      state_server.state_cov.transpose()) / 2.0;
  state_server.state_cov = state_cov_fixed;



  // // Update the IMU state.
  // const VectorXd& delta_x_imu = delta_x.head<21>();

  // if (//delta_x_imu.segment<3>(0).norm() > 0.15 ||
  //     //delta_x_imu.segment<3>(3).norm() > 0.15 ||
  //     delta_x_imu.segment<3>(3).norm() > 0.5 ||
  //     //delta_x_imu.segment<3>(9).norm() > 0.5 ||
  //     delta_x_imu.segment<3>(6).norm() > 1.0) {
  //   printf("delta velocity: %f\n", delta_x_imu.segment<3>(6).norm());
  //   printf("delta position: %f\n", delta_x_imu.segment<3>(12).norm());
  //   ROS_WARN("Update change is too large.");
  //   //return;
  // }

  // const Vector4d dq_imu =
  //   smallAngleQuaternion(delta_x_imu.head<3>());
  // state_server.imu_state.orientation = quaternionMultiplication(
  //     dq_imu, state_server.imu_state.orientation);
  // state_server.imu_state.gyro_bias += delta_x_imu.segment<3>(9);
  // state_server.imu_state.velocity += delta_x_imu.segment<3>(3);
  // state_server.imu_state.acc_bias += delta_x_imu.segment<3>(12);
  // state_server.imu_state.position += delta_x_imu.segment<3>(6);

  // const Vector4d dq_extrinsic =
  //   smallAngleQuaternion(delta_x_imu.segment<3>(15));
  // state_server.imu_state.R_imu_cam0 = quaternionToRotation(
  //     dq_extrinsic) * state_server.imu_state.R_imu_cam0;
  // state_server.imu_state.t_cam0_imu += delta_x_imu.segment<3>(18);

  // // Update the camera states.
  // auto cam_state_iter = state_server.cam_states.begin();
  // for (int i = 0; i < state_server.cam_states.size();
  //     ++i, ++cam_state_iter) {
  //   const VectorXd& delta_x_cam = delta_x.segment<6>(21+i*6);
  //   const Vector4d dq_cam = smallAngleQuaternion(delta_x_cam.head<3>());
  //   cam_state_iter->second.orientation = quaternionMultiplication(
  //       dq_cam, cam_state_iter->second.orientation);
  //   cam_state_iter->second.position += delta_x_cam.tail<3>();
  // }

  // Update all

  // Update the IMU state.
  const VectorXd& delta_x_imu = delta_x.head<9>();

  if (
      delta_x_imu.segment<3>(3).norm() > 0.5 ||
      delta_x_imu.segment<3>(6).norm() > 1.0) {
    printf("delta velocity: %f\n", delta_x_imu.segment<3>(3).norm());
    printf("delta position: %f\n", delta_x_imu.segment<3>(6).norm());
    ROS_WARN("Update change is too large.");
    //return;
  }

  Matrix<double, 5, 5> X_pred = Matrix<double, 5, 5>::Zero();
  X_pred.block<3, 3>(0,0) = quaternionToRotation(state_server.imu_state.orientation).transpose();
  X_pred.block<3, 1>(0,3) = state_server.imu_state.velocity;
  X_pred.block<3, 1>(0,4) = state_server.imu_state.position;
  X_pred(3,3)= 1;
  X_pred(4,4)= 1;

//    ROS_INFO_STREAM("State_pred"<<
//       X_pred);
// // ROS_INFO_STREAM("State_rungk:"<<
//       // q_P<<v_P<<p_P);

  Matrix<double, 5, 5> Error_imu = Matrix<double, 5, 5>::Zero();
  Error_imu.block<3, 3>(0,0) = skewSymmetric(delta_x_imu.segment(0,3)) ;
  Error_imu.block<3, 1>(0,3) = delta_x_imu.segment(3,3);
  Error_imu.block<3, 1>(0,4) = delta_x_imu.segment(6,3);

  Matrix<double, 5, 5> X_ = Matrix<double, 5, 5>::Zero();
  X_ = Error_imu.exp()*X_pred;

  // ROS_INFO_STREAM("State_"<<
  //     X_);

  // update stateserver
  state_server.imu_state.orientation = rotationToQuaternion(X_.block<3, 3>(0,0).transpose());
  state_server.imu_state.velocity = X_.block<3, 1>(0,3);
  state_server.imu_state.position  = X_pred.block<3, 1>(0,4);

  // Update the IMU state.
  const VectorXd& delta_x_bias = delta_x.segment(9,6);

  Matrix<double, 4, 4> Bg_pred = Matrix<double, 4, 4>::Zero();
  Bg_pred.block<3, 3>(0,0) = Matrix3d::Identity();
  Bg_pred.block<3, 1>(0,3) = state_server.imu_state.gyro_bias;
  Bg_pred(3,3)= 1;

  // ROS_INFO_STREAM("bias_pre"<<
  //     Bg_pred);

  Matrix<double, 4, 4> Errg_bias = Matrix<double, 4, 4>::Zero();
  // Errg_bias.block<3, 3>(0,0) = Matrix3d::Identity();
  Errg_bias.block<3, 1>(0,3) = delta_x_bias.segment(0,3);

  Matrix<double, 4, 4> Bg_ = Matrix<double, 4, 4>::Zero();
  Bg_ = Errg_bias.exp()*Bg_pred;
  // Bg_ = Bg_pred*Errg_bias.exp();

// ROS_INFO_STREAM("bias_"<<
//       Bg_);
  // update stateserver
  state_server.imu_state.gyro_bias = Bg_.block<3, 1>(0,3);

// acc bias

  Matrix<double, 4, 4> Ba_pred = Matrix<double, 4, 4>::Zero();
  Ba_pred.block<3, 3>(0,0) = Matrix3d::Identity();
  Ba_pred.block<3, 1>(0,3) = state_server.imu_state.acc_bias;
  Ba_pred(3,3)= 1;


ROS_INFO_STREAM("bias_a_pre"<<
      state_server.imu_state.acc_bias);

  Matrix<double, 4, 4> Erra_bias = Matrix<double, 4, 4>::Zero();
  // Erra_bias.block<3, 3>(0,0) = Matrix3d::Identity();
  Erra_bias.block<3, 1>(0,3) = delta_x_bias.segment(3,3);

  Matrix<double, 4, 4> Ba_ = Matrix<double, 4, 4>::Zero();
  Ba_ = Erra_bias.exp()*Ba_pred;
  // Ba_ = Ba_pred*Erra_bias.exp();

  // update stateserver
  state_server.imu_state.acc_bias = Ba_.block<3, 1>(0,3);

  ROS_INFO_STREAM("bias_a_pred"<<
      state_server.imu_state.acc_bias);

//  extrensic
  const VectorXd& delta_x_ext = delta_x.segment(15,6);

  Matrix<double, 4, 4> Ext_pred = Matrix<double, 4, 4>::Zero();
  Ext_pred.block<3, 3>(0,0) = state_server.imu_state.R_imu_cam0.transpose();
  Ext_pred.block<3, 1>(0,3) = state_server.imu_state.t_cam0_imu ;
  Ext_pred(3,3)= 1;

  Matrix<double, 4, 4> Errext = Matrix<double, 4, 4>::Zero();
  Errext.block<3, 3>(0,0) = skewSymmetric(delta_x_ext.segment(0,3));
  Errext.block<3, 1>(0,3) = delta_x_ext.segment(3,3);

  Matrix<double, 4, 4> Ext_ = Matrix<double, 4, 4>::Zero();
  Ext_ = Errext.exp()*Ext_pred;

  // update stateserver
  state_server.imu_state.R_imu_cam0 = (Ext_.block<3, 3>(0,0)).transpose();
  state_server.imu_state.t_cam0_imu = Ext_.block<3, 1>(0,3);


  // Update the camera states.
  auto cam_state_iter = state_server.cam_states.begin();
  for (int i = 0; i < state_server.cam_states.size();
      ++i, ++cam_state_iter) {
    const VectorXd& delta_x_cam = delta_x.segment<6>(21+i*6);

    Matrix<double, 4, 4> cam_pred = Matrix<double, 4, 4>::Zero();
    cam_pred.block<3, 3>(0,0) = quaternionToRotation(cam_state_iter->second.orientation).transpose();
    cam_pred.block<3, 1>(0,3) = cam_state_iter->second.position;
    cam_pred(3,3)= 1;


    Matrix<double, 4, 4> Errcam = Matrix<double, 4, 4>::Zero();
    Errcam.block<3, 3>(0,0) = skewSymmetric(delta_x_cam.segment(0,3));
    Errcam.block<3, 1>(0,3) = delta_x_cam.segment(3,3);

    Matrix<double, 4, 4> cam_ = Matrix<double, 4, 4>::Zero();
    cam_ = Errcam.exp()*cam_pred;

    cam_state_iter->second.orientation = rotationToQuaternion(cam_.block<3, 3>(0,0).transpose());
    cam_state_iter->second.position = cam_.block<3, 1>(0,3);
  }




    // const Vector4d dq_cam = smallAngleQuaternion(delta_x_cam.head<3>());
    // cam_state_iter->second.orientation = quaternionMultiplication(
    //     dq_cam, cam_state_iter->second.orientation);
    // cam_state_iter->second.position += delta_x_cam.tail<3>();
  

  

  return;
}

bool MsckfVio::gatingTest(
    const MatrixXd& H, const VectorXd& r, const int& dof) {

  MatrixXd P1 = H * state_server.state_cov * H.transpose();
  MatrixXd P2 = Feature::observation_noise *
    MatrixXd::Identity(H.rows(), H.rows());
  double gamma = r.transpose() * (P1+P2).ldlt().solve(r);

  //cout << dof << " " << gamma << " " <<
  //  chi_squared_test_table[dof] << " ";

  if (gamma < chi_squared_test_table[dof]) {
    //cout << "passed" << endl;
    return true;
  } else {
    //cout << "failed" << endl;
    return false;
  }
}

void MsckfVio::removeLostFeatures() {

  // Remove the features that lost track.
  // BTW, find the size the final Jacobian matrix and residual vector.
  int jacobian_row_size = 0;
  vector<FeatureIDType> invalid_feature_ids(0);
  vector<FeatureIDType> processed_feature_ids(0);

  for (auto iter = map_server.begin();
      iter != map_server.end(); ++iter) {
    // Rename the feature to be checked.
    auto& feature = iter->second;

    // Pass the features that are still being tracked.
    if (feature.observations.find(state_server.imu_state.id) !=
        feature.observations.end()) continue;
    if (feature.observations.size() < 3) {
      invalid_feature_ids.push_back(feature.id);
      continue;
    }

    // Check if the feature can be initialized if it
    // has not been.
    if (!feature.is_initialized) {
      if (!feature.checkMotion(state_server.cam_states)) {
        invalid_feature_ids.push_back(feature.id);
        continue;
      } else {
        if(!feature.initializePosition(state_server.cam_states)) {
          invalid_feature_ids.push_back(feature.id);
          continue;
        }
      }
    }

    jacobian_row_size += 4*feature.observations.size() - 3;
    processed_feature_ids.push_back(feature.id);
  }

  //cout << "invalid/processed feature #: " <<
  //  invalid_feature_ids.size() << "/" <<
  //  processed_feature_ids.size() << endl;
  //cout << "jacobian row #: " << jacobian_row_size << endl;

  // Remove the features that do not have enough measurements.
  for (const auto& feature_id : invalid_feature_ids)
    map_server.erase(feature_id);

  // Return if there is no lost feature to be processed.
  if (processed_feature_ids.size() == 0) return;

  MatrixXd H_x = MatrixXd::Zero(jacobian_row_size,
      21+6*state_server.cam_states.size());
  VectorXd r = VectorXd::Zero(jacobian_row_size);
  int stack_cntr = 0;

  // Process the features which lose track.
  for (const auto& feature_id : processed_feature_ids) {
    auto& feature = map_server[feature_id];

    vector<StateIDType> cam_state_ids(0);
    for (const auto& measurement : feature.observations)
      cam_state_ids.push_back(measurement.first);

    MatrixXd H_xj;
    VectorXd r_j;
    featureJacobian(feature.id, cam_state_ids, H_xj, r_j);

    if (gatingTest(H_xj, r_j, cam_state_ids.size()-1)) {
      H_x.block(stack_cntr, 0, H_xj.rows(), H_xj.cols()) = H_xj;
      r.segment(stack_cntr, r_j.rows()) = r_j;
      stack_cntr += H_xj.rows();
    }

    // Put an upper bound on the row size of measurement Jacobian,
    // which helps guarantee the executation time.
    if (stack_cntr > 1500) break;
  }

  H_x.conservativeResize(stack_cntr, H_x.cols());
  r.conservativeResize(stack_cntr);

  // Perform the measurement update step.
  measurementUpdate(H_x, r);

  // Remove all processed features from the map.
  for (const auto& feature_id : processed_feature_ids)
    map_server.erase(feature_id);

  return;
}

void MsckfVio::findRedundantCamStates(
    vector<StateIDType>& rm_cam_state_ids) {

  // Move the iterator to the key position.
  auto key_cam_state_iter = state_server.cam_states.end();
  for (int i = 0; i < 4; ++i)
    --key_cam_state_iter;
  auto cam_state_iter = key_cam_state_iter;
  ++cam_state_iter;
  auto first_cam_state_iter = state_server.cam_states.begin();

  // Pose of the key camera state.
  const Vector3d key_position =
    key_cam_state_iter->second.position;
  const Matrix3d key_rotation = quaternionToRotation(
      key_cam_state_iter->second.orientation);

  // Mark the camera states to be removed based on the
  // motion between states.
  for (int i = 0; i < 2; ++i) {
    const Vector3d position =
      cam_state_iter->second.position;
    const Matrix3d rotation = quaternionToRotation(
        cam_state_iter->second.orientation);

    double distance = (position-key_position).norm();
    double angle = AngleAxisd(
        rotation*key_rotation.transpose()).angle();

    if (angle < rotation_threshold &&
        distance < translation_threshold &&
        tracking_rate > tracking_rate_threshold) {
      rm_cam_state_ids.push_back(cam_state_iter->first);
      ++cam_state_iter;
      // ROS_INFO_STREAM("Error?");
    } else {
      rm_cam_state_ids.push_back(first_cam_state_iter->first);
      ++first_cam_state_iter;
    }
  }

  // Sort the elements in the output vector.
  sort(rm_cam_state_ids.begin(), rm_cam_state_ids.end());

  return;
}

void MsckfVio::pruneCamStateBuffer() {

  if (state_server.cam_states.size() < max_cam_state_size)
    return;

  // Find two camera states to be removed.
  vector<StateIDType> rm_cam_state_ids(0);
  findRedundantCamStates(rm_cam_state_ids);

  // Find the size of the Jacobian matrix.
  int jacobian_row_size = 0;
  for (auto& item : map_server) {
    auto& feature = item.second;
    // Check how many camera states to be removed are associated
    // with this feature.
    vector<StateIDType> involved_cam_state_ids(0);
    for (const auto& cam_id : rm_cam_state_ids) {
      if (feature.observations.find(cam_id) !=
          feature.observations.end())
        involved_cam_state_ids.push_back(cam_id);
    }

    if (involved_cam_state_ids.size() == 0) continue;
    if (involved_cam_state_ids.size() == 1) {
      feature.observations.erase(involved_cam_state_ids[0]);
      continue;
    }

    if (!feature.is_initialized) {
      // Check if the feature can be initialize.
      if (!feature.checkMotion(state_server.cam_states)) {
        // If the feature cannot be initialized, just remove
        // the observations associated with the camera states
        // to be removed.
        for (const auto& cam_id : involved_cam_state_ids)
          feature.observations.erase(cam_id);
        continue;
      } else {
        if(!feature.initializePosition(state_server.cam_states)) {
          for (const auto& cam_id : involved_cam_state_ids)
            feature.observations.erase(cam_id);
          continue;
        }
      }
    }

    jacobian_row_size += 4*involved_cam_state_ids.size() - 3;
  }

  //cout << "jacobian row #: " << jacobian_row_size << endl;

  // Compute the Jacobian and residual.
  MatrixXd H_x = MatrixXd::Zero(jacobian_row_size,
      21+6*state_server.cam_states.size());
  VectorXd r = VectorXd::Zero(jacobian_row_size);
  int stack_cntr = 0;

  for (auto& item : map_server) {
    auto& feature = item.second;
    // Check how many camera states to be removed are associated
    // with this feature.
    vector<StateIDType> involved_cam_state_ids(0);
    for (const auto& cam_id : rm_cam_state_ids) {
      if (feature.observations.find(cam_id) !=
          feature.observations.end())
        involved_cam_state_ids.push_back(cam_id);
    }

    if (involved_cam_state_ids.size() == 0) continue;

    MatrixXd H_xj;
    VectorXd r_j;
    featureJacobian(feature.id, involved_cam_state_ids, H_xj, r_j);

    if (gatingTest(H_xj, r_j, involved_cam_state_ids.size())) {
      H_x.block(stack_cntr, 0, H_xj.rows(), H_xj.cols()) = H_xj;
      r.segment(stack_cntr, r_j.rows()) = r_j;
      stack_cntr += H_xj.rows();
    }

    for (const auto& cam_id : involved_cam_state_ids)
      feature.observations.erase(cam_id);
  }

  H_x.conservativeResize(stack_cntr, H_x.cols());
  r.conservativeResize(stack_cntr);

  // Perform measurement update.
  measurementUpdate(H_x, r);

  for (const auto& cam_id : rm_cam_state_ids) {
    int cam_sequence = std::distance(state_server.cam_states.begin(),
        state_server.cam_states.find(cam_id));
    int cam_state_start = 21 + 6*cam_sequence;
    int cam_state_end = cam_state_start + 6;

    // Remove the corresponding rows and columns in the state
    // covariance matrix.
    if (cam_state_end < state_server.state_cov.rows()) {
      state_server.state_cov.block(cam_state_start, 0,
          state_server.state_cov.rows()-cam_state_end,
          state_server.state_cov.cols()) =
        state_server.state_cov.block(cam_state_end, 0,
            state_server.state_cov.rows()-cam_state_end,
            state_server.state_cov.cols());

      state_server.state_cov.block(0, cam_state_start,
          state_server.state_cov.rows(),
          state_server.state_cov.cols()-cam_state_end) =
        state_server.state_cov.block(0, cam_state_end,
            state_server.state_cov.rows(),
            state_server.state_cov.cols()-cam_state_end);

      state_server.state_cov.conservativeResize(
          state_server.state_cov.rows()-6, state_server.state_cov.cols()-6);
    } else {
      state_server.state_cov.conservativeResize(
          state_server.state_cov.rows()-6, state_server.state_cov.cols()-6);
    }

    // Remove this camera state in the state vector.
    state_server.cam_states.erase(cam_id);
  }

  return;
}

void MsckfVio::onlineReset() {

  // Never perform online reset if position std threshold
  // is non-positive.
  if (position_std_threshold <= 0) return;
  static long long int online_reset_counter = 0;

  // Check the uncertainty of positions to determine if
  // the system can be reset.

  //  Cov changed so need to chnage thissss!!!
  double position_x_std = std::sqrt(state_server.state_cov(6, 6));
  double position_y_std = std::sqrt(state_server.state_cov(7, 7));
  double position_z_std = std::sqrt(state_server.state_cov(8, 8));

  if (position_x_std < position_std_threshold &&
      position_y_std < position_std_threshold &&
      position_z_std < position_std_threshold) return;

  ROS_WARN("Start %lld online reset procedure...",
      ++online_reset_counter);
  ROS_INFO("Stardard deviation in xyz: %f, %f, %f",
      position_x_std, position_y_std, position_z_std);

  // Remove all existing camera states.
  state_server.cam_states.clear();

  // Clear all exsiting features in the map.
  map_server.clear();

  // Reset the state covariance.
  double gyro_bias_cov, acc_bias_cov, velocity_cov;
  nh.param<double>("initial_covariance/velocity",
      velocity_cov, 0.25);
  nh.param<double>("initial_covariance/gyro_bias",
      gyro_bias_cov, 1e-4);
  nh.param<double>("initial_covariance/acc_bias",
      acc_bias_cov, 1e-2);

  double extrinsic_rotation_cov, extrinsic_translation_cov;
  nh.param<double>("initial_covariance/extrinsic_rotation_cov",
      extrinsic_rotation_cov, 3.0462e-4);
  nh.param<double>("initial_covariance/extrinsic_translation_cov",
      extrinsic_translation_cov, 1e-4);


//  new cov
    state_server.state_cov = MatrixXd::Zero(21, 21);
  for (int i = 3; i < 6; ++i)
    state_server.state_cov(i, i) = velocity_cov; 
  for (int i = 9; i < 12; ++i)
    state_server.state_cov(i, i) = gyro_bias_cov;
  for (int i = 12; i < 15; ++i)
    state_server.state_cov(i, i) = acc_bias_cov;
  for (int i = 15; i < 18; ++i)
    state_server.state_cov(i, i) = extrinsic_rotation_cov;
  for (int i = 18; i < 21; ++i)
    state_server.state_cov(i, i) = extrinsic_translation_cov;


  // state_server.state_cov = MatrixXd::Zero(21, 21);
  // for (int i = 3; i < 6; ++i)
  //   state_server.state_cov(i, i) = gyro_bias_cov;
  // for (int i = 6; i < 9; ++i)
  //   state_server.state_cov(i, i) = velocity_cov;
  // for (int i = 9; i < 12; ++i)
  //   state_server.state_cov(i, i) = acc_bias_cov;
  // for (int i = 15; i < 18; ++i)
  //   state_server.state_cov(i, i) = extrinsic_rotation_cov;
  // for (int i = 18; i < 21; ++i)
  //   state_server.state_cov(i, i) = extrinsic_translation_cov;

  ROS_WARN("%lld online reset complete...", online_reset_counter);
  return;
}

void MsckfVio::publish(const ros::Time& time) {

  // Convert the IMU frame to the body frame.
  const IMUState& imu_state = state_server.imu_state;
  Eigen::Isometry3d T_i_w = Eigen::Isometry3d::Identity();
  T_i_w.linear() = quaternionToRotation(
      imu_state.orientation).transpose();
  T_i_w.translation() = imu_state.position;

  Eigen::Isometry3d T_b_w = IMUState::T_imu_body * T_i_w *
    IMUState::T_imu_body.inverse();
  Eigen::Vector3d body_velocity =
    IMUState::T_imu_body.linear() * imu_state.velocity;

  // Publish tf
  if (publish_tf) {
    tf::Transform T_b_w_tf;
    tf::transformEigenToTF(T_b_w, T_b_w_tf);
    tf_pub.sendTransform(tf::StampedTransform(
          T_b_w_tf, time, fixed_frame_id, child_frame_id));
  }

  // Publish the odometry
  nav_msgs::Odometry odom_msg;
  odom_msg.header.stamp = time;
  odom_msg.header.frame_id = fixed_frame_id;
  odom_msg.child_frame_id = child_frame_id;

  tf::poseEigenToMsg(T_b_w, odom_msg.pose.pose);
  tf::vectorEigenToMsg(body_velocity, odom_msg.twist.twist.linear);

//  Cov changed so need to chnage thissss!!!
  // Convert the covariance.
  Matrix3d P_oo = state_server.state_cov.block<3, 3>(0, 0);
  Matrix3d P_op = state_server.state_cov.block<3, 3>(0, 6);
  Matrix3d P_po = state_server.state_cov.block<3, 3>(6, 0);
  Matrix3d P_pp = state_server.state_cov.block<3, 3>(6, 6);
  Matrix<double, 6, 6> P_imu_pose = Matrix<double, 6, 6>::Zero();
  P_imu_pose << P_pp, P_po, P_op, P_oo;

  Matrix<double, 6, 6> H_pose = Matrix<double, 6, 6>::Zero();
  H_pose.block<3, 3>(0, 0) = IMUState::T_imu_body.linear();
  H_pose.block<3, 3>(3, 3) = IMUState::T_imu_body.linear();
  Matrix<double, 6, 6> P_body_pose = H_pose *
    P_imu_pose * H_pose.transpose();

  for (int i = 0; i < 6; ++i)
    for (int j = 0; j < 6; ++j)
      odom_msg.pose.covariance[6*i+j] = P_body_pose(i, j);

//  Cov changed so need to chnage thissss!!!
  // Construct the covariance for the velocity.
  Matrix3d P_imu_vel = state_server.state_cov.block<3, 3>(3, 3);
  Matrix3d H_vel = IMUState::T_imu_body.linear();
  Matrix3d P_body_vel = H_vel * P_imu_vel * H_vel.transpose();
  for (int i = 0; i < 3; ++i)
    for (int j = 0; j < 3; ++j)
      odom_msg.twist.covariance[i*6+j] = P_body_vel(i, j);

  odom_pub.publish(odom_msg);

  // Publish the 3D positions of the features that
  // has been initialized.
  boost::shared_ptr<pcl::PointCloud<pcl::PointXYZ> > feature_msg_ptr(
      new pcl::PointCloud<pcl::PointXYZ>());
  feature_msg_ptr->header.frame_id = fixed_frame_id;
  feature_msg_ptr->height = 1;
  for (const auto& item : map_server) {
    const auto& feature = item.second;
    if (feature.is_initialized) {
      Vector3d feature_position =
        IMUState::T_imu_body.linear() * feature.position;
      feature_msg_ptr->points.push_back(pcl::PointXYZ(
            feature_position(0), feature_position(1), feature_position(2)));
    }
  }
  feature_msg_ptr->width = feature_msg_ptr->points.size();

  feature_pub.publish(feature_msg_ptr);

  return;
}

} // namespace msckf_vio

