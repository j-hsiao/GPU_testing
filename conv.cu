#include <stdio.h>
#include <iostream>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

__global__ void convolveKernelBordered(uchar* indata, uchar* outdata, float* kernel, int IW,
                                       int KH, int KW, int BW) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  int idy = blockIdx.y * blockDim.y + threadIdx.y;
  int channel = threadIdx.z;
  float val = 0.0f;
  for (int i = 0; i < KH; i++) {
    for (int j = 0; j < KW; j++) {
      int imx = idx + j;
      int imy = idy + i;
      val += indata[(((imy * BW) + imx) * 3) + channel] * kernel[(i * KW) + j];
    }
  }
  val = val < 0 ? 0:val;
  outdata[(((idy * IW) + idx) * 3) + channel] = val > 255 ? 255: (uchar) val;

  //outdata[(((idy * IW) + idx) * 3) + channel] = indata[((((idy + (KH / 2)) * BW) + (idx + (KW / 2))) * 3) + channel];
  //outdata[(((idy * IW) + idx) * 3) + channel] = 0;
}



//this should not be called with a block with z dimension
__global__ void convolveKernelBordered_pixel(uchar* indata, uchar* outdata, float* kernel, int IW,
                                       int KH, int KW, int BW) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  int idy = blockIdx.y * blockDim.y + threadIdx.y;
  float val[3] = {0.0f, 0.0f, 0.0f};
  for (int i = 0; i < KH; i++) {
    for (int j = 0; j < KW; j++) {
      int imx = idx + j;
      int imy = idy + i;
      for (int channel = 0; channel < 3; channel++) {
        val[channel] += indata[(((imy * BW) + imx) * 3) + channel] * kernel[(i * KW) + j];
      }
    }
  }
  for (int channel = 0; channel < 3; channel++) {
    val[channel] = val[channel] < 0 ? 0:val[channel];
    outdata[(((idy * IW) + idx) * 3) + channel] = val[channel] > 255 ? 255: (uchar) val[channel];
  }

  //outdata[(((idy * IW) + idx) * 3) + channel] = indata[((((idy + (KH / 2)) * BW) + (idx + (KW / 2))) * 3) + channel];
  //outdata[(((idy * IW) + idx) * 3) + channel] = 0;
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

  cudaEvent_t start, stop;



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
  int KW = blurfilt.size().width;
  int KH = blurfilt.size().height;

  int TOTW = 640;
  int TOTH = 480;
  int BLOCKW = 128;
  int BLOCKH = 1;

  //640x480x3 total
  dim3 grid(TOTW/BLOCKW, TOTH/BLOCKH);
  dim3 block(BLOCKW, BLOCKH, 1);
  /*
  indiv conv:
    grid:      block:     runtime:
    160 120    4  4  3    1.4-1.7 msec per image
    40  480    16 1  3    1.22    msec per image
    ~15% increase in speed when matching half-warp

    80  120    8  4  3    1.21    msec per image
    20  480    32 1  3    0.98    msec per image
    ~23% increase in speed using full warp

    40  30     16 16 3    1.04    msec per image
    20  60     32 8  3    0.98    msec per image
                          occasional 1.2
    6% increase in speed from halfwarps to full warp

    10  60     64 4  3    0.98, occasional 1.2

    5   240   128 2  3    0.98, occasional 1.2

    20  480    32 1  3    0.98, occasional 1.3

  pixel conv:
    20  30     32 16 1    0.45-0.46
    20  15     32 32 1    0.46

    X   X      8  4  1    0.62/0.63
    X   X      16 2  1    0.51
    X   X      32 1  1    0.51

    X   X      8  8  1    0.62/0.63
    X   X      16 4  1    0.50
    X   X      32 2  1    0.458
    X   X      64 1  1    0.461
    X   X      4  16 1    0.82
    X   X      2  32 1    1.33
    X   X      1  64 1    2.20

    
    X   X      1  1  1    10.35
    X   X      2  1  1    5.53
    X   X      4  1  1    2.79
    X   X      5  1  1    2.26
    X   X      8  1  1    1.48
    X   X      10 1  1    1.16
    X   X      16 1  1    0.77
    X   X      20 1  1    0.64
    X   X      32 1  1    0.46
    X   X      40 1  1    0.5
    X   X      64 1  1    0.46
    X   X      80 1  1    0.52
    X   X     128 1  1    0.46


  */




  uchar* kernelInput = 0;
  uchar* kernelOutput = 0;
  float* kernelFilter = 0;

  cudaMalloc((void**) &kernelInput, sizeof(uchar) * (640 + (2 * (KH / 2))) * (480 + (2 * (KW / 2))) * 3);
  // printf("%s\n", cudaGetErrorString(cudaGetLastError()));

  cudaMalloc((void**) &kernelOutput, sizeof(uchar) * 640 * 480 * 3);
  // printf("%s\n", cudaGetErrorString(cudaGetLastError()));

  cudaMalloc((void**) &kernelFilter, sizeof(float) * fh * fh);
  // printf("%s\n", cudaGetErrorString(cudaGetLastError()));

  if (!blurfilt.isContinuous()) {
    std::cout << "blurfilt is not continuous" << std::endl;
    return 1;
  }




  cudaMemcpy((void*) kernelFilter, (void*) blurfilt.data, sizeof(float)*fh*fh, cudaMemcpyHostToDevice);
  // printf("%s\n", cudaGetErrorString(cudaGetLastError()));




  if ((kernelInput == 0)
      || (kernelOutput == 0)
      || (kernelFilter == 0)) {
    std::cout << "failed to allocate" << std::endl;
    return 0;
  }



  float totalTime = 0.0f;
  int totalFrames = 0;

  while(true) {
    if(cap.read(imframe)) {
      cv::flip(imframe, imframe, 2);
      cv::filter2D(imframe, cvProcessed, -1, blurfilt);
      myProcessed = imframe.clone() * 0;

      // std::cout << myProcessed.size().width << "," << myProcessed.size().height << std::endl;


      if (!myProcessed.isContinuous()) {
        std::cout << "myProcessed not continuous" << std::endl;
        return 1;
      }
      cv::copyMakeBorder(imframe, bordered, fh / 2, fh / 2, fh / 2, fh / 2, cv::BORDER_DEFAULT);
      if (!bordered.isContinuous()) {
        std::cout << "bordered not continuous" << std::endl;
        return 1;
      }

      int IW = imframe.size().width;
      int IH = imframe.size().height;

      int BW = bordered.size().width;


      cudaMemcpy((void*) kernelInput, (void*) bordered.data, sizeof(uchar)*(bordered.size[0]*bordered.size[1]*3), cudaMemcpyHostToDevice);
      // printf("copy input: %s\n", cudaGetErrorString(cudaGetLastError()));

      // std::cout << "size of input: " << sizeof(uchar)*(bordered.size[0]*bordered.size[1]*3) << std::endl;
      // std::cout.flush();

      cudaEventCreate(&start);
      cudaEventCreate(&stop);
      cudaEventRecord(start, 0);

      convolveKernelBordered_pixel<<<grid,block>>>(kernelInput, kernelOutput, kernelFilter, IW, KW, KH, BW);
      
      cudaEventRecord(stop, 0);
      cudaEventSynchronize(stop);
      float et;
      cudaEventElapsedTime(&et, start, stop);
      totalTime += et;
      totalFrames++;
      cudaEventDestroy(start);
      cudaEventDestroy(stop);

      // printf("kernel runtime: %8.7f\n", et);

      cudaMemcpy((void*) myProcessed.data, (void*) kernelOutput, (sizeof(uchar) * IW * IH * 3), cudaMemcpyDeviceToHost);
      // printf("copy output: %s\n", cudaGetErrorString(cudaGetLastError()));

      // std::cout << "size of output: " << sizeof(uchar) * IW * IH * 3 << std::endl;
      // std::cout.flush();

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
  printf("average kernel time: %2.7f\n", totalTime / totalFrames);
  imframe.release();
  cvProcessed.release();
  cap.release();
  myProcessed.release();
  bordered.release();
}
