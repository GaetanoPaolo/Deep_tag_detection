#include <opencv2/cudalegacy.hpp>
#include <opencv2/core/mat.hpp>
#include <iostream>
#include <pybind11/pybind11.h>
// Beware of the type of the inputs: int or float or double may not correspond with python 
float* solvepnp_gpu( int n,void* pos_temp_world, void* pos_target, void* K,void* dist_coeffs) {
  //cv::cuda::GpuMat rvec, tvec, object, image, camera_mat, dist_coef;
  cv::Mat rvec, tvec;
  const cv::Mat dist_coef_mat(4,1, CV_64FC1,dist_coeffs);
  const cv::Mat temp_camera_mat(3,3, CV_64FC1,K);
  const cv::Mat pos_temp_world_mat(n,3,CV_64FC1,pos_temp_world);
  const cv::Mat pos_target_mat (n,2,CV_64FC1,pos_target);
  //object.upload(cv::Mat(pos_temp_world));
  //image.upload(cv::Mat(pos_target));
  //camera_mat.upload(temp_camera_mat);
  //dist_coef.upload(dist_coef_mat);
  std::vector< int > inliers;
  cv::cuda::solvePnPRansac (pos_temp_world_mat,pos_target_mat,temp_camera_mat,dist_coef_mat,rvec,tvec,false,2000,2.0,0.99,&inliers);
  //cv::Mat trans_mat;
  //tvec.download(trans_mat);
  float inlier_len = inliers.size();
  static float trans_inl [4] = {tvec.at<float>(0),tvec.at<float>(1),tvec.at<float>(2),inlier_len};
  return trans_inl;
}

PYBIND11_MODULE(solvepnp_gpu, m) {
    m.doc() = "pybind11 solvepnp_gpu plugin"; // optional module docstring

    m.def("solvepnp_gpu", &solvepnp_gpu, "A function that performs Perspective-n-point pose estimation on GPU");
}
