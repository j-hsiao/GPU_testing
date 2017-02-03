#include <stdio.h>
#include <iostream>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

// __global__ void convolve(float* image, float* kernel) {
//   __shared__
// }

void filter2D(cv::Mat &inframe, cv::Mat &outframe, int depth, cv::Mat &kernel) {
  int W = inframe.size[0];
  int H = inframe.size[1];
  int KW = kernel.size[0];
  int KH = kernel.size[1];
  int KSW = KW / 2;
  int KSH = KH / 2;
  int chans = inframe.channels();
  //inframe.copyTo(outframe);
  if (outframe.size[0] == 0) {
    outframe = cv::Mat(W, H, CV_32FC3);
  }
  outframe *= 0;
  //naive implementation
  
  //loop over each element in array
  for (int i = 0; i < W; i++) {
    for (int j = 0; j < H; j++) {
      //add the multiplications for each elem
      for (int ki = 0; ki < KW; ki++) {
        for (int kj = 0; kj < KH; kj++) {
          int ni = (i - KSW) + ki;
          int nj = (j - KSH) + kj;
          if ((ni >= 0) && (nj >= 0)
              && (ni < W) && (nj < H)) {
            for (int c = 0; c < chans; c++) {
              outframe.at<cv::Vec3f>(i, j)[c] += ((float) inframe.at<cv::Vec3b>(((i - KSW) + ki), ((j - KSH) + kj))[c]) * kernel.at<float>(ki, kj);
            }
          }
        }
      }
      cv::Vec3f check = outframe.at<cv::Vec3f>(i, j);
      for(int checkin = 0; checkin < 3; checkin++) {
        if (check[checkin] < 0) {
          outframe.at<cv::Vec3f>(i, j)[checkin] = 0;
        } else if (check[checkin] > 255) {
          outframe.at<cv::Vec3f>(i, j)[checkin] = 255;
        }
      }
      
    }
  }
  outframe /= 255;
  return;
}

void filter2DBordered(cv::Mat &inframe, cv::Mat &outframe, int depth, cv::Mat &kernel) {
  int KW = kernel.size[0];
  int KH = kernel.size[1];
  int KSW = KW / 2;
  int KSH = KH / 2;
  int W = inframe.size[0] - (KSW * 2);
  int H = inframe.size[1] - (KSH * 2);
  int chans = inframe.channels();


  //inframe.copyTo(outframe);
  if (outframe.size[0] == 0) {
    outframe = cv::Mat(W, H, CV_32FC3);
  }
  outframe *= 0;
  //naive implementation
  
  //loop over each element in array
  for (int i = 0; i < W; i++) {
    for (int j = 0; j < H; j++) {
      //add the multiplications for each elem
      for (int ki = 0; ki < KW; ki++) {
        for (int kj = 0; kj < KH; kj++) {
          int ni = i + ki;
          int nj = j + kj;
          for (int c = 0; c < chans; c++) {
            outframe.at<cv::Vec3f>(i, j)[c] += ((float) inframe.at<cv::Vec3b>(ni, nj)[c]) * kernel.at<float>(ki, kj);
          }
        }
      }
      cv::Vec3f check = outframe.at<cv::Vec3f>(i, j);
      for(int checkin = 0; checkin < 3; checkin++) {
        if (check[checkin] < 0) {
          outframe.at<cv::Vec3f>(i, j)[checkin] = 0;
        } else if (check[checkin] > 255) {
          outframe.at<cv::Vec3f>(i, j)[checkin] = 255;
        }
      }
      
    }
  }
  outframe /= 255;
  return;
}




int main(int argn, char** args){
  cv::VideoCapture cap = cv::VideoCapture(0);
  cv::Mat imframe;
  cv::Mat cvProcessed;
  cv::Mat filtdisp;
  cv::Mat myProcessed;
  cv::Mat bordered;
  int cvswitch;
  int fh = 5;

  cv::Mat blurfilt = cv::Mat(fh,fh,CV_32FC1);
  cv::Size fdispsize = cv::Size(500,500);
  for(int i = 0; i < fh; i++) {
    for (int j = 0; j < fh; j++) {
      blurfilt.at<float>(i,j) = 1.0 / (fh*fh);
      // std::cout << ((float*) blurfilt.data)[blurfilt.step[0]*i + blurfilt.step[1] * j] << std::endl;
      // blurfilt.data<float>[(blurfilt.step[0]*i) + blurfilt.step[1]*j] = 1.0;
      // std::cout << blurfilt.data<float>[(blurfilt.step[0]*i) + blurfilt.step[1]*j];
    }
  }
  cv::resize(blurfilt, filtdisp, fdispsize);


  std::cout << blurfilt.size[0] << "," << blurfilt.size[1];
  std::cout.flush();
  while(true) {
    if(cap.read(imframe)) {
      std::cout << imframe.isContinuous() << std::endl;
      cv::flip(imframe, imframe, 2);
      cv::filter2D(imframe, cvProcessed, -1, blurfilt);
      cv::copyMakeBorder(imframe, bordered, fh / 2, fh / 2, fh / 2, fh / 2, cv::BORDER_DEFAULT);
      filter2DBordered(bordered, myProcessed, -1, blurfilt);
      // filter2D(imframe, myProcessed, -1, blurfilt);



      cv::imshow("hello", imframe);
      cv::imshow("cvProcessed", cvProcessed);
      cv::imshow("myProcessed", myProcessed);


      cv::imshow("filt", filtdisp);
    }
    cvswitch = cv::waitKey(1);
    if (cvswitch >= 0) {
      break;
    }
  }
  imframe.release();
  cvProcessed.release();
  cap.release();
  myProcessed.release();
  bordered.release();
}
