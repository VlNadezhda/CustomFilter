#include <iostream>
#include <opencv2/opencv.hpp>
#include <vector>

using namespace cv;
using std::vector;

uchar calculatePixel(const Mat& image, const vector<vector<double>>& kernel,
                     int ii, int jj){
    int newPixel = 0;

    for (unsigned i = 0; i < kernel.size(); ++i) {
        for (unsigned j = 0; j < kernel.size(); ++j) {
            newPixel += image.at<uchar>(ii + static_cast<int>(i),
                                        jj + static_cast<int>(j)) * kernel[i][j];
        }
    }
    //защита от переполнения
    return saturate_cast<uchar>(newPixel);
}

uchar Convolution(const Mat& image, Mat& result,
                  const vector<vector<double> >& kernel){
    int border = static_cast<int>(kernel.size() / 2);

    //если каналов больше одного 2д массив становится 3д
    if (image.type() == 0) {
        for (uint i = 0; i < image.cols - border + 1; ++i) {
            for (uint j = 0; j < image.rows - border + 1; ++j) {
                result.at<uchar>(i, j) = calculatePixel(image, kernel, i, j);
            }
        }
    }else {
        vector<Mat> channels;
        //разделяем каналы
        split(image, channels);
        vector<Mat> results = channels;

        //применяем фильтр для всех каналов
        for (int i = border; i < image.cols - border + 1; ++i) {
            for (int j = border; j < image.rows - border + 1; ++j) {
                results[0].at<uchar>(i, j) = calculatePixel(channels[0], kernel, i, j);
                results[1].at<uchar>(i, j) = calculatePixel(channels[1], kernel, i, j);
                results[2].at<uchar>(i, j) = calculatePixel(channels[2], kernel, i, j);
            }
        }
       //слияние
        merge(results, result);
    }

}

void customSmoothing(const Mat& image, Mat& result, unsigned size) {
    image.copyTo(result);

    //усредняющий фильтр
    vector<vector<double> > kernel(size,vector<double>(size, 1. / (size * size)));
    Convolution(image, result, kernel);
}

void customGradient(const Mat& image, Mat& result) {
    image.copyTo(result);
    vector<vector<double> > kernel = {{-1, 0, 1}, {-2, 0, 2}, {-1, 0, 1}};
    Convolution(image, result, kernel);
}

int main() {
    cv::Mat image= cv::imread("ll.jpg");
    cv::Mat res;
    if (image.empty()) {
        std::cout << "Error\n";
        return -1;
    }

    customSmoothing(image, res, 6);
    namedWindow("Smoothing", WINDOW_AUTOSIZE);
    imshow("Smoothing", res);

    customGradient(image, res);
    namedWindow("Gradient", WINDOW_AUTOSIZE);
    imshow("Gradient", res);

    waitKey(0);

    return 0;
}
