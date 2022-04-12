#include <opencv2/cudalegacy.hpp>
#include <Eigen/Dense>
#include <opencv2/core/mat.hpp>
#include <opencv2/core/cuda_types.hpp>
#include <opencv2/core/types.hpp>
#include <opencv2/core/matx.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/core/eigen.hpp>
#include <opencv2/core.hpp>
#include <opencv2/freetype.hpp>
#include <opencv2/opencv.hpp>
#include <iostream>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>
#include <pybind11/eigen.h>

namespace py = pybind11;

/* Bind MatrixXd (or some other Eigen type) to Python */
typedef Eigen::MatrixXf Matrix;

//typedef Matrix::Scalar Scalar;
constexpr bool rowMajor = Matrix::Flags & Eigen::RowMajorBit;


//py::array_t<double> solvepnp_gpu(int n,const py::array_t<float> pos_temp_world_py,const py::array_t<float> pos_target_py,const py::array_t<float> K_py,const py::array_t<float> dist_coeffs_py) {
py::array_t<float> solvepnp_gpu(int it_amount, float reproj_error, int min_inlier_count,int n,const Matrix pos_temp_world,const Matrix pos_target,const Matrix K,const Matrix dist_coeffs) {
  cv::Mat rvec(3,1, CV_32FC1);
  cv::Mat tvec(3,1, CV_32FC1);
  Matrix dist_coeffs_tr = dist_coeffs.transpose();
  const cv::Mat dist_coef_mat(1,4, CV_32FC1);
  const cv::Mat temp_camera_mat(3,3, CV_32FC1);
  cv::Mat pos_temp_world_mat(n,3,CV_32FC1);
  cv::Mat pos_target_mat(n,2,CV_32FC1);
  eigen2cv(pos_temp_world, pos_temp_world_mat);
  eigen2cv(pos_target, pos_target_mat);
  eigen2cv(K, temp_camera_mat);
  eigen2cv(dist_coeffs_tr, dist_coef_mat);
  const cv::Mat pos_temp_world_mat_chan =  pos_temp_world_mat.reshape(3,1);
  const cv::Mat pos_target_mat_chan =  pos_target_mat.reshape(2,1);
  std::vector< int > inliers;

  cv::cuda::solvePnPRansac (pos_temp_world_mat_chan,pos_target_mat_chan,temp_camera_mat,dist_coef_mat,rvec,tvec,false,it_amount,reproj_error,min_inlier_count,&inliers);
  float inlier_len = inliers.size();
  std::array<float,4> trans_inl = {tvec.at<float>(0),tvec.at<float>(1),tvec.at<float>(2),inlier_len};
  py::array_t<float> trans_inl_py = py::cast(trans_inl);
  return trans_inl_py;
}

PYBIND11_MODULE(solvepnp_gpu, m) {
    m.doc() = "pybind11 solvepnp_gpu plugin"; // optional module docstring

    m.def("solvepnp_gpu", &solvepnp_gpu, "A function that performs Perspective-n-point pose estimation on GPU");
}
