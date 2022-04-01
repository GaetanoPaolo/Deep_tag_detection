#include <opencv2/cudalegacy.hpp>
#include <opencv2/core/mat.hpp>
#include <iostream>
#include <pybind11/pybind11.h>
// Beware of the type of the inputs: int or float or double may not correspond with python 
float solvepnp_gpu( const std::vector<Point3f> &pos_temp_world, const std::vector<Point2f> &pos_target, const float &K [3][3],const float &dist_coeffs [4],   ) {
  cv::cuda::GpuMat rvec, tvec, object, image, camera_mat, dist_coef;
  float inliers;
  cv::Mat dist_coef_mat(4,1, CV_64F,dist_coeffs);
  cv::Mat temp_camera_mat(3,3, CV_64F,K);
  std::vector< int > *inliers;
  object.upload(pos_temp_world);
  image.upload(pos_target);
  camera_mat.upload(temp_camera_mat);
  dist_coef.upload(dist_coef_mat);
  cv::cuda::solvePnPRansac (object,image,camera_mat,dist_coef,rvec,tvec,2000,2.0,inliers);
  cv::Mat trans_mat
  tvec.download(trans_mat);
  float inlier_len = inliers.size();
  float trans_inl [4] = {trans_mat.at<float>(0),trans_mat.at<float>(1),trans_mat.at<float>(2),inlier_len}
  return trans_inl;
}

PYBIND11_MODULE(solvepnp_gpu, m) {
    m.doc() = "pybind11 solvepnp_gpu plugin"; // optional module docstring

    m.def("solvepnp_gpu", &solvepnp_gpu, "A function that performs Perspective-n-point pose estimation on GPU");
}
