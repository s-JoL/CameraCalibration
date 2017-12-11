#include <time.h>

#include <opencv2/video.hpp>
#include "camera_calibration_with_opencv.h"

using namespace std;

int main(int argc, char *argv[])
{
    time_t start = clock();
    
    
//    vector<vector<Point2f>> chess_points;
//    Mat camera_matrix;
//    vector<Mat> rotation_matrix, translate_matrix;
//    Size board_size(6, 9);//棋盘为纵向6格横向9格
//    //Size square_size(25, 25);//棋盘大小每格为 (x, y) = (25mm, 25mm)
//
//    //Calibration("CameraCalibration/", "chess", ".jpg", 4, board_size, square_size, chess_points, camera_matrix, rotation_matrix, translate_matrix);//6张标定图片
//    //Calibration("CameraCalibration/", "chess", ".jpg", 4, 9, 6, 36, 36);//6张标定图片
//
//    float x, y;
//    //cout << tmp << endl;
//    //cout << chess_points[3][53] << endl;
//    //cout << chess_points[0][0] << endl;
//    //cout << MatrixTransform(camera_matrix, rotation_matrix[0], translate_matrix[0], chess_points[0][0]);
//    //cout << MatrixTransform(camera_matrix, rotation_matrix[0], translate_matrix[0], tmp) << endl;
//    //MatrixTransform(500000, 0, x, y);
//    MatrixTransform("/Users/souryou/Downloads/picture/calibration/", 0, 0, &x, &y);
//    cout << x << " " << y << endl;
    
    
    cout << "Used Time: " << double(clock() - start) / CLOCKS_PER_SEC << endl;
    return 1;
}

