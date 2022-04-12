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
typedef Eigen::MatrixXd Matrix;

//typedef Matrix::Scalar Scalar;
constexpr bool rowMajor = Matrix::Flags & Eigen::RowMajorBit;


//py::array_t<double> solvepnp_gpu(int n,const py::array_t<float> pos_temp_world_py,const py::array_t<float> pos_target_py,const py::array_t<float> K_py,const py::array_t<float> dist_coeffs_py) {
py::array_t<double> solvepnp_gpu(int n,const Matrix pos_temp_world,const Matrix pos_target,const Matrix K,const Matrix dist_coeffs) {
  //cv::cuda::GpuMat rvec, tvec, object, image, camera_mat, dist_coef;
  cv::Mat rvec, tvec;
//   const Matrix pos_temp_world = pos_temp_world_py.cast<Matrix>();
//   const Matrix pos_target = pos_target_py.cast<Matrix>();
//   const Matrix K = K_py.cast<Matrix>();
//   const Matrix dist_coeffs = dist_coeffs_py.cast<Matrix>();
  const cv::Mat dist_coef_mat(1,4, CV_64FC1);
  const cv::Mat temp_camera_mat(3,3, CV_64FC1);
  const cv::Mat pos_temp_world_mat(n,3,CV_64FC1);
  const cv::Mat pos_target_mat(n,2,CV_64FC1);
//   const cv::Mat dist_coef_mat = cv::Mat(dist_coeffs,false);
//   const cv::Mat temp_camera_mat = cv::Mat(K,false);
//   const cv::Mat pos_temp_world_mat = cv::Mat(pos_temp_world,false);
//   const cv::Mat pos_target_mat = cv::Mat(pos_target,false);
  eigen2cv(pos_temp_world, pos_temp_world_mat);
  cout << "pos_temp_world coef ok";
  eigen2cv(pos_target, pos_target_mat);
  cout << "pos_target ok";
  eigen2cv(K, temp_camera_mat);
  cout << "K  ok";
  eigen2cv(dist_coeffs, dist_coef_mat);
  cout << "dist coef ok";
  //object.upload(cv::Mat(pos_temp_world));
  //image.upload(cv::Mat(pos_target));
  //camera_mat.upload(temp_camera_mat);
  //dist_coef.upload(dist_coef_mat);
  std::vector< int > inliers;
  cv::cuda::solvePnPRansac (pos_temp_world_mat,pos_target_mat,temp_camera_mat,dist_coef_mat,rvec,tvec,false,2000,2.0,0.99,&inliers);
  //cv::Mat trans_mat;
  //tvec.download(trans_mat);
  float inlier_len = inliers.size();
  std::array<float,4> trans_inl = {tvec.at<float>(0),tvec.at<float>(1),tvec.at<float>(2),inlier_len};
  py::array_t<float> trans_inl_py = py::cast(trans_inl);
//   auto result = py::array_t<double>(dist_coeffs_py.size);
//   py::buffer_info buf5 = result.request();
//   double *ptr5 = static_cast<double *>(buf5.ptr);
//   for (size_t idx = 0; idx < buf5.shape[0]; idx++)
//       ptr5[idx] = trans_inl[idx];
  return trans_inl_py;
}

PYBIND11_MODULE(solvepnp_gpu, m) {
    m.doc() = "pybind11 solvepnp_gpu plugin"; // optional module docstring

    m.def("solvepnp_gpu", &solvepnp_gpu, "A function that performs Perspective-n-point pose estimation on GPU");
}
