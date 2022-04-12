#include <opencv2/cudalegacy.hpp>
#include <opencv2/core/mat.hpp>
#include <opencv2/core/cuda_types.hpp>
#include <opencv2/core/types.hpp>
#include <opencv2/core/matx.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/core.hpp>
#include <opencv2/freetype.hpp>
#include <opencv2/opencv.hpp>
#include <iostream>
#include <eigen3/Eigen/Dense>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>


namespace py = pybind11;

/* Bind MatrixXd (or some other Eigen type) to Python */
typedef Eigen::MatrixXd Matrix;

typedef Matrix::Scalar Scalar;
constexpr bool rowMajor = Matrix::Flags & Eigen::RowMajorBit;


py::array_t<double> solvepnp_gpu(int n,const cv::Mat pos_temp_world_mat,const cv::Mat pos_target_mat,const cv::Mat K_mat,const cv::Mat dist_coeffs_mat) {
  //cv::cuda::GpuMat rvec, tvec, object, image, camera_mat, dist_coef;
  cv::Mat rvec, tvec;
  std::vector< int > inliers;
  cv::cuda::solvePnPRansac (pos_temp_world_mat,pos_target_mat,K_mat,dist_coeffs_mat,rvec,tvec,false,2000,2.0,0.99,&inliers);
  //cv::Mat trans_mat;
  //tvec.download(trans_mat);
  double inlier_len = inliers.size();
  std::array<double,4> trans_inl = {tvec.at<double>(0),tvec.at<double>(1),tvec.at<double>(2),inlier_len};
  py::array trans_inl_py = py::cast(trans_inl);
  return trans_inl_py;
}

PYBIND11_MODULE(solvepnp_gpu, m) {
    m.doc() = "pybind11 solvepnp_gpu plugin"; // optional module docstring

    m.def("solvepnp_gpu", &solvepnp_gpu, "A function that performs Perspective-n-point pose estimation on GPU");
}
