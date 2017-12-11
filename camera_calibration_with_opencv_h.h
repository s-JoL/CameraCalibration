//
//  camera_calibration_with_opencv.h
//  opencv_test
//
//  Created by Bayes on 2017/10/28.
//  Copyright © 2017年 Bayes. All rights reserved.
//

#ifndef camera_calibration_with_opencv_h
#define camera_calibration_with_opencv_h

#include <iostream>
#include <fstream>

#include "opencv2/core/core.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/calib3d/calib3d.hpp"
#include "opencv2/imgproc/imgproc.hpp"

//using namespace std;
using namespace cv;

int Calibration(std::string FILE_PATH, std::string FILE_NAME, std::string FILE_TYPE, int FILES_NUMBER, Size board_size, Size square_size, std::vector<std::vector<Point2f>> &chess_points, Mat &camera_matrix, std::vector<Mat> &rotation_matrix, std::vector<Mat> &translate_matrix) {
    /*
     定标函数
     输入：定标图片路径 文件名 文件类型 图片数量 棋盘格点数 棋盘大小
     棋盘在图片中像素坐标  相机参数矩阵  每张图片的旋转向量  每张图片的平移向量
     例：
     定标图片 路径 文件名 文件类型 图片数量 棋盘格点数 棋盘大小分别为：
     somefolder/  chess  .jpg  10  Size(6, 9)  Size(25, 25)
     则定标图片为  somefolder/chess1.jpg - chess10.jpg 每张图片棋盘为 9x6 棋盘大小为 25mm x 25mm
     棋盘在图片中像素位置  相机参数矩阵  每张图片的旋转向量  每张图片的平移向量 通过引用输出
     并生成txt保存相机相关参数
     */
    
    Size image_size;
    
    for (int i = 0; i < FILES_NUMBER; ++i) {
        std::string file_name = FILE_PATH + FILE_NAME + std::to_string(i + 1) + FILE_TYPE;
        Mat image = imread(file_name);
        
        
        std::vector<Point2f> chess_points_tmp;
        findChessboardCorners(image, board_size, chess_points_tmp);//寻找棋盘格点
        
        if (i == 0) {
            image_size.height = image.rows;
            image_size.width = image.cols;
        }//读取图片大小
        
        Mat view_gray;
        //cvtColor(image, view_gray, CV_RGB2GRAY);
        //find4QuadCornerSubpix(view_gray, chess_points_tmp, Size(5,5)); //对粗提取的角点进行精确化
        chess_points.push_back(chess_points_tmp);
        
//        std::cout << chess_points_tmp.size() << std::endl;
//        Mat show_chess = image;
//        drawChessboardCorners(show_chess, board_size, chess_points_tmp, false);
//        imshow("Camera Calibration", show_chess);//显示图片
//        waitKey(3000);//暂停5S
    }
    
    std::vector<std::vector<Point3f>> real_pos;//假设棋盘处于z = 0平面，以棋盘右上角格点为原点，横为x， 纵为y
    for (int k = 0; k < FILES_NUMBER; ++k) {
        
        std::vector<Point3f> tmp_points;
        for (int i = 0; i < board_size.height; ++i) {
            for (int j = 0; j < board_size.width; ++j) {
                Point3f real_point;
                real_point.z = 0;
                real_point.x = i * square_size.width;
                real_point.y = j * square_size.height;
                tmp_points.push_back(real_point);
            }
        }
        real_pos.push_back(tmp_points);
    }
    Mat cameraMatrix = Mat(3, 3, CV_64FC1, Scalar::all(0)); /* 摄像机内参数矩阵 */
    Mat distCoeffs = Mat(1, 5, CV_64FC1, Scalar::all(0)); /* 摄像机的5个畸变系数：k1,k2,p1,p2,k3 */
    std::vector<Mat> tvecsMat;  /* 每幅图像的平移向量 */
    std::vector<Mat> rvecsMat; /* 每幅图像的旋转向量 */
    
    calibrateCamera(real_pos, chess_points, image_size, cameraMatrix, distCoeffs, rvecsMat, tvecsMat, 0);
    
//    std::ofstream fout("cameraMatrix.dat");
//    //fout << "相机参数矩阵" << std::endl;
//    fout << cameraMatrix;// << std::endl << std::endl;
//    fout = std::ofstream("rotation.dat");
//    fout << rvecsMat[0];
//    fout = std::ofstream("translation.dat");
//    fout << tvecsMat[0];
    
    FILE *fp;
    fp = fopen("camera_matrix.dat", "w");
    for (int i = 0; i < 3; ++i) {
        fprintf(fp, "%lf %lf %lf\n", cameraMatrix.at<double>(i, 0), cameraMatrix.at<double>(i, 1), cameraMatrix.at<double>(i, 2));
    }
    fclose(fp);
    
    fp = fopen("rotation.dat", "w");
    fprintf(fp, "%lf %lf %lf\n", rvecsMat[0].at<double>(0, 0), rvecsMat[0].at<double>(1, 0), rvecsMat[0].at<double>(2, 0));
    fclose(fp);
    
    fp = fopen("translation.dat", "w");
    fprintf(fp, "%lf %lf %lf\n", tvecsMat[0].at<double>(0, 0), tvecsMat[0].at<double>(1, 0), tvecsMat[0].at<double>(2, 0));
    fclose(fp);
    
    //fout << "畸变系数" << std::endl;
    //fout << distCoeffs << std::endl << std::endl;//将相机参数写入文件
    
    
    //    fout << "平移向量" << std::endl;
    //    fout << tvecsMat[0] << std::endl << std::endl;
    //    CvMat *rotation = cvCreateMat(3, 3, CV_64FC1), tmp = rvecsMat[0];
    //    cvRodrigues2(&tmp, rotation);
    //    fout << "旋转矩阵" << std::endl;
    //    fout << cvarrToMat(rotation) << std::endl << std::endl;
    
    camera_matrix = cameraMatrix;
    rotation_matrix = rvecsMat;
    translate_matrix = tvecsMat;//输出变换向量
    
    return 1;
    
}

int Calibration(std::string FILE_PATH, std::string FILE_NAME, std::string FILE_TYPE, int FILES_NUMBER, int board_size_width, int board_size_height, int square_size_width, int square_size_height) {
    /*
     定标函数
     输入：定标图片路径 文件名 文件类型 图片数量 棋盘格点数 棋盘大小
     棋盘在图片中像素坐标  相机参数矩阵  每张图片的旋转向量  每张图片的平移向量
     例：
     定标图片 路径 文件名 文件类型 图片数量 棋盘格点数 棋盘大小分别为：
     somefolder/  chess  .jpg  10  Size(6, 9)  Size(25, 25)
     则定标图片为  somefolder/chess1.jpg - chess10.jpg 每张图片棋盘为 9x6 棋盘大小为 25mm x 25mm
     棋盘在图片中像素位置  相机参数矩阵  每张图片的旋转向量  每张图片的平移向量 通过引用输出
     并生成txt保存相机相关参数
     */
    
    Size image_size;
    Size board_size = Size(board_size_height, board_size_width);
    Size square_size = Size(square_size_height, square_size_width);
    std::vector<std::vector<Point2f>> chess_points;
    
    for (int i = 0; i < FILES_NUMBER; ++i) {
        std::string file_name = FILE_PATH + FILE_NAME + std::to_string(i + 1) + FILE_TYPE;
        Mat image = imread(file_name);
        
        
        std::vector<Point2f> chess_points_tmp;
        findChessboardCorners(image, board_size, chess_points_tmp);//寻找棋盘格点
        
        if (i == 0) {
            image_size.height = image.rows;
            image_size.width = image.cols;
        }//读取图片大小
        
        Mat view_gray;
        
        
        cvtColor(image, view_gray, CV_RGB2GRAY);
        imshow("test", view_gray);
        waitKey(3000);//暂停5S
        std::cout << chess_points_tmp.size() << std::endl;
        find4QuadCornerSubpix(view_gray, chess_points_tmp, Size(5,5)); //对粗提取的角点进行精确化
        chess_points.push_back(chess_points_tmp);
        
        std::cout << chess_points_tmp.size() << std::endl;
        
        for (auto i: chess_points_tmp)
            std::cout << i << std::endl;
        Mat show_chess = image;
        drawChessboardCorners(show_chess, board_size, chess_points_tmp, false);
        imshow("Camera Calibration", show_chess);//显示图片
        waitKey(3000);//暂停5S
    }
    
    std::vector<std::vector<Point3f>> real_pos;//假设棋盘处于z = 0平面，以棋盘右上角格点为原点，横为x， 纵为y
    for (int k = 0; k < FILES_NUMBER; ++k) {
        
        std::vector<Point3f> tmp_points;
        for (int i = 0; i < board_size.height; ++i) {
            for (int j = 0; j < board_size.width; ++j) {
                Point3f real_point;
                real_point.z = 0;
                real_point.x = i * square_size.width;
                real_point.y = j * square_size.height;
                tmp_points.push_back(real_point);
            }
        }
        real_pos.push_back(tmp_points);
    }
    Mat cameraMatrix = Mat(3, 3, CV_64FC1, Scalar::all(0)); /* 摄像机内参数矩阵 */
    Mat distCoeffs = Mat(1, 5, CV_64FC1, Scalar::all(0)); /* 摄像机的5个畸变系数：k1,k2,p1,p2,k3 */
    std::vector<Mat> tvecsMat;  /* 每幅图像的平移向量 */
    std::vector<Mat> rvecsMat; /* 每幅图像的旋转向量 */
    
    calibrateCamera(real_pos, chess_points, image_size, cameraMatrix, distCoeffs, rvecsMat, tvecsMat, 0);
    
    //    std::ofstream fout("cameraMatrix.dat");
    //    //fout << "相机参数矩阵" << std::endl;
    //    fout << cameraMatrix;// << std::endl << std::endl;
    //    fout = std::ofstream("rotation.dat");
    //    fout << rvecsMat[0];
    //    fout = std::ofstream("translation.dat");
    //    fout << tvecsMat[0];
    
    FILE *fp;
    fp = fopen("camera_matrix.dat", "w");
    for (int i = 0; i < 3; ++i) {
        fprintf(fp, "%lf %lf %lf\n", cameraMatrix.at<double>(i, 0), cameraMatrix.at<double>(i, 1), cameraMatrix.at<double>(i, 2));
    }
    fclose(fp);
    
    fp = fopen("rotation.dat", "w");
    fprintf(fp, "%lf %lf %lf\n", rvecsMat[0].at<double>(0, 0), rvecsMat[0].at<double>(1, 0), rvecsMat[0].at<double>(2, 0));
    fclose(fp);
    
    fp = fopen("translation.dat", "w");
    fprintf(fp, "%lf %lf %lf\n", tvecsMat[0].at<double>(0, 0), tvecsMat[0].at<double>(1, 0), tvecsMat[0].at<double>(2, 0));
    fclose(fp);
    
    //fout << "畸变系数" << std::endl;
    //fout << distCoeffs << std::endl << std::endl;//将相机参数写入文件
    
    return 1;
    
}

Point2f MatrixTransform(Mat camera_matrix, Mat rotation_matrix, Mat translate_matrix, Point2f point) {
    /*
     计算世界坐标系中坐标
     输入：相机参数矩阵 旋转向量 平移向量 像素坐标
     输出：世界坐标系坐标
     世界坐标系为以棋盘右上角点为原点，棋盘平面为 z=0，向右为 x正方向， 向下为 y正方向坐标系
     */
    CvMat *rotation = cvCreateMat(3, 3, CV_64FC1), tmp = rotation_matrix;
    cvRodrigues2(&tmp, rotation);//将旋转向量转换为旋转矩阵
    cv::Mat H(cvarrToMat(rotation));
    cv::Mat translation_ve;//平移向量
    translate_matrix.copyTo(translation_ve);
    H.at<double>(0, 2) = translation_ve.at<double>(0, 0);
    H.at<double>(1, 2) = translation_ve.at<double>(1, 0);
    H.at<double>(2, 2) = translation_ve.at<double>(2, 0);
    cv::Mat hu;
    hu = camera_matrix * H;
    cv::Mat hu2 = hu.inv();
    double a1, a2, a3, a4, a5, a6, a7, a8, a9;
    a1 = hu2.at<double>(0, 0);
    a2 = hu2.at<double>(0, 1);
    a3 = hu2.at<double>(0, 2);
    a4 = hu2.at<double>(1, 0);
    a5 = hu2.at<double>(1, 1);
    a6 = hu2.at<double>(1, 2);
    a7 = hu2.at<double>(2, 0);
    a8 = hu2.at<double>(2, 1);
    a9 = hu2.at<double>(2, 2);
    Point2f tmp_point;
    double xe = point.x;//图像中点坐标x
    double ye = point.y;//图像中点坐标y
    tmp_point.x = (a1*xe + a2*ye + a3) / (a7*xe + a8*ye + a9);//世界坐标中x值
    tmp_point.y = (a4*xe + a5*ye + a6) / (a7*xe + a8*ye + a9);//世界坐标中Y值
    
    return tmp_point;//返回该点世界坐标，以棋盘右上角为原点向左为x正，向下为y正方向
}

int MatrixTransform(double x, double y, double &x_out, double &y_out) {
    /*
     计算世界坐标系中坐标
     输入：相机参数矩阵 旋转向量 平移向量 像素坐标
     输出：世界坐标系坐标
     世界坐标系为以棋盘右上角点为原点，棋盘平面为 z=0，向右为 x正方向， 向下为 y正方向坐标系
     */
    Mat camera_matrix, rotation_matrix, translate_matrix;
    
    double tmp1[9], tmp2[3], tmp3[3];
    FILE *fp = fopen("camera_matrix.dat", "r");
    for (int i = 0; i < 3; ++i) {
        fscanf(fp, "%lf %lf %lf\n", &tmp1[3 * i], &tmp1[3 * i + 1], &tmp1[3 * i + 2]);
    }
    camera_matrix = Mat(3, 3, CV_64FC1, tmp1);
    fclose(fp);
    
    fp = fopen("rotation.dat", "r");
    fscanf(fp, "%lf %lf %lf\n", &tmp2[0], &tmp2[1], &tmp2[2]);
    rotation_matrix = Mat(3, 1, CV_64FC1, tmp2);
    fclose(fp);
    
    fp = fopen("translation.dat", "r");
    fscanf(fp, "%lf %lf %lf\n", &tmp3[0], &tmp3[1], &tmp3[2]);
    translate_matrix = Mat(3, 1, CV_64FC1, tmp3);
    fclose(fp);
    

    CvMat *rotation = cvCreateMat(3, 3, CV_64FC1), tmp = rotation_matrix;
    cvRodrigues2(&tmp, rotation);//将旋转向量转换为旋转矩阵
    cv::Mat H(cvarrToMat(rotation));
    cv::Mat translation_ve;//平移向量
    translate_matrix.copyTo(translation_ve);
    H.at<double>(0, 2) = translation_ve.at<double>(0, 0);
    H.at<double>(1, 2) = translation_ve.at<double>(1, 0);
    H.at<double>(2, 2) = translation_ve.at<double>(2, 0);
    cv::Mat hu;
    hu = camera_matrix * H;
    cv::Mat hu2 = hu.inv();
    double a1, a2, a3, a4, a5, a6, a7, a8, a9;
    a1 = hu2.at<double>(0, 0);
    a2 = hu2.at<double>(0, 1);
    a3 = hu2.at<double>(0, 2);
    a4 = hu2.at<double>(1, 0);
    a5 = hu2.at<double>(1, 1);
    a6 = hu2.at<double>(1, 2);
    a7 = hu2.at<double>(2, 0);
    a8 = hu2.at<double>(2, 1);
    a9 = hu2.at<double>(2, 2);
    //Point2f tmp_point;
    double xe = x;//图像中点坐标x
    double ye = y;//图像中点坐标y
    x_out = (a1*xe + a2*ye + a3) / (a7*xe + a8*ye + a9);//世界坐标中x值
    y_out = (a4*xe + a5*ye + a6) / (a7*xe + a8*ye + a9);//世界坐标中Y值
    
    return 1;//返回该点世界坐标，以棋盘右上角为原点向左为x正，向下为y正方向
}

double CalcAngle(double x, double y, double z) {
    // x y 为在平面坐标，z为平面与摄像头距离
    return atan(sqrt(x * x + y * y) / z);
}

int MatrixTransform(char *file_path, double x, double y, float *x_out, float *y_out) {
    /*
     计算世界坐标系中坐标
     输入：相机参数矩阵 旋转向量 平移向量 像素坐标
     输出：世界坐标系坐标
     世界坐标系为以棋盘右上角点为原点，棋盘平面为 z=0，向右为 x正方向， 向下为 y正方向坐标系
     */
    Mat camera_matrix, rotation_matrix, translate_matrix;
    std::string file_path_str = file_path, input_path;
    
    double tmp1[9], tmp2[3], tmp3[3];
    input_path = file_path_str + "camera_matrix.dat";
    FILE *fp = fopen(input_path.c_str(), "r");
    for (int i = 0; i < 3; ++i) {
        fscanf(fp, "%lf %lf %lf\n", &tmp1[3 * i], &tmp1[3 * i + 1], &tmp1[3 * i + 2]);
    }
    camera_matrix = Mat(3, 3, CV_64FC1, tmp1);
    fclose(fp);
    
    input_path = file_path_str + "rotation.dat";
    fp = fopen(input_path.c_str(), "r");
    fscanf(fp, "%lf %lf %lf\n", &tmp2[0], &tmp2[1], &tmp2[2]);
    rotation_matrix = Mat(3, 1, CV_64FC1, tmp2);
    fclose(fp);
    
    input_path = file_path_str + "translation.dat";
    fp = fopen(input_path.c_str(), "r");
    fscanf(fp, "%lf %lf %lf\n", &tmp3[0], &tmp3[1], &tmp3[2]);
    translate_matrix = Mat(3, 1, CV_64FC1, tmp3);
    fclose(fp);
    
    
    CvMat *rotation = cvCreateMat(3, 3, CV_64FC1), tmp = rotation_matrix;
    cvRodrigues2(&tmp, rotation);//将旋转向量转换为旋转矩阵
    cv::Mat H(cvarrToMat(rotation));
    cv::Mat translation_ve;//平移向量
    translate_matrix.copyTo(translation_ve);
    H.at<double>(0, 2) = translation_ve.at<double>(0, 0);
    H.at<double>(1, 2) = translation_ve.at<double>(1, 0);
    H.at<double>(2, 2) = translation_ve.at<double>(2, 0);
    cv::Mat hu;
    hu = camera_matrix * H;
    cv::Mat hu2 = hu.inv();
    double a1, a2, a3, a4, a5, a6, a7, a8, a9;
    a1 = hu2.at<double>(0, 0);
    a2 = hu2.at<double>(0, 1);
    a3 = hu2.at<double>(0, 2);
    a4 = hu2.at<double>(1, 0);
    a5 = hu2.at<double>(1, 1);
    a6 = hu2.at<double>(1, 2);
    a7 = hu2.at<double>(2, 0);
    a8 = hu2.at<double>(2, 1);
    a9 = hu2.at<double>(2, 2);
    //Point2f tmp_point;
    double xe = x;//图像中点坐标x
    double ye = y;//图像中点坐标y
    *x_out = (a1*xe + a2*ye + a3) / (a7*xe + a8*ye + a9);//世界坐标中x值
    *y_out = (a4*xe + a5*ye + a6) / (a7*xe + a8*ye + a9);//世界坐标中Y值
    
    return 1;//返回该点世界坐标，以棋盘右上角为原点向左为x正，向下为y正方向
}

int TakePhoto(char *path, int choice)
{
    char keyCode;
    VideoCapture capture(choice);
    int count = 1;
    if (!capture.isOpened())
        return -1;
    Mat frame;
    while (keyCode = cvWaitKey(30))
    {
        if (keyCode == 27)
        {
            break;
        }

        capture >> frame;
        imshow("读取视频", frame);
        if (keyCode == 13)
        {
            std::string name = std::string(path) + "chess" + std::to_string(count) + ".jpg";
            imwrite(name, frame);
            ++count;
        }
    }

    return 1;
}

#endif /* camera_calibration_with_opencv_h */

