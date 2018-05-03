
#include <iostream>
#include <string>
#include <stdexcept>

#include "opencv2/core.hpp"
#include <opencv2/highgui.hpp>
#include "opencv2/calib3d.hpp"
#include "opencv2/aruco.hpp"

using namespace cv;
using namespace std;

/* Degree to radians and viceversa conversion multiplication consants*/
constexpr double DegToRad = CV_PI / 180.0;
constexpr double RadToDeg = 180.0 / CV_PI;

void printRigidBodyTransformInfo(const Vec3d& rvec, const Vec3d& tvec)
{
    cout << "Rotation vector:"  << rvec << " ,translation vector" << tvec  <<endl;
    
    cout << "distance to center (in meters): " << cv::norm(tvec) <<endl;
    cout << "axis: " << cv::normalize(rvec) << " angle(deg): " << cv::norm(rvec)*RadToDeg <<endl;
}

/**
  * @brief Return object points for the system centered in a single marker, given the marker length
  */
void getSingleMarkerObjectPointsDouble(float markerLength, OutputArray _objPoints) {

    CV_Assert(markerLength > 0);

    _objPoints.create(4, 1, CV_64FC3);
    Mat objPoints = _objPoints.getMat();
    // set coordinate system in the middle of the marker, with Z pointing out
    objPoints.ptr< Vec3d >(0)[0] = Vec3d(-markerLength / 2.0,  markerLength / 2.0, 0);
    objPoints.ptr< Vec3d >(0)[1] = Vec3d( markerLength / 2.0,  markerLength / 2.0, 0);
    objPoints.ptr< Vec3d >(0)[2] = Vec3d( markerLength / 2.0, -markerLength / 2.0, 0);
    objPoints.ptr< Vec3d >(0)[3] = Vec3d(-markerLength / 2.0, -markerLength / 2.0, 0);
}

/**
  * @brief Return object points for the system centered in a single marker, given the marker length
  */
void getSingleMarkerObjectPoints(float markerLength, OutputArray _objPoints) {

    CV_Assert(markerLength > 0);

    _objPoints.create(4, 1, CV_32FC3);
    Mat objPoints = _objPoints.getMat();
    // set coordinate system in the middle of the marker, with Z pointing out
    objPoints.ptr< Vec3f >(0)[0] = Vec3f(-markerLength / 2.f,  markerLength / 2.f, 0);
    objPoints.ptr< Vec3f >(0)[1] = Vec3f( markerLength / 2.f,  markerLength / 2.f, 0);
    objPoints.ptr< Vec3f >(0)[2] = Vec3f( markerLength / 2.f, -markerLength / 2.f, 0);
    objPoints.ptr< Vec3f >(0)[3] = Vec3f(-markerLength / 2.f, -markerLength / 2.f, 0);
}

int main()
{
    try
    {
        //--  Read the input image wiht the ARUCO target
        //Mat inputImage = imread("data/test_pose_Gazebo.jpg",  IMREAD_COLOR);
        Mat inputImage = imread("data/Gazebo_testInitialPose.jpg",  IMREAD_COLOR);

        vector<int> markerIds;
        vector<vector<Point2f>> markerCorners, rejectedCandidates;
        cv::Ptr<aruco::DetectorParameters> parameters = aruco::DetectorParameters::create(); //default params
        /* We set the same parameters that we have in python*/
        parameters->minMarkerPerimeterRate = 0.1 ;
        parameters->cornerRefinementMethod = aruco::CORNER_REFINE_SUBPIX;
        cv::Ptr<cv::aruco::Dictionary> dictionary = cv::aruco::getPredefinedDictionary(cv::aruco::DICT_ARUCO_ORIGINAL);

        cv::aruco::detectMarkers(inputImage, dictionary, markerCorners, markerIds, parameters, rejectedCandidates);
        if(markerIds.size() > 0)
        {
            cv::aruco::drawDetectedMarkers(inputImage, markerCorners, markerIds);
        }


        /* From ROS with the current camera location (obtained by "rostopic echo -n1 /camera1/camera_info") */
        //K: [866.2229887853814, 0.0, 500.5, 0.0, 866.2229887853814, 500.5, 0.0, 0.0, 1.0]
        //D: [0.0, 0.0, 0.0, 0.0, 0.0]
        //Mat cameraMatrix = (Mat1d(3, 3) << fx, 0, cx, 0, fx, cx, 0, 0, 1);
        Mat cameraMatrix = (Mat1d(3, 3) << 866.2229887853814, 0.0, 500.5, 0.0, 866.2229887853814, 500.5, 0.0, 0.0, 1.0);
        Mat distortionCoefficients = (Mat1d(1, 4) << 0, 0, 0, 0);



        vector< Vec3d > rvecs, tvecs;
        float markerLength = 0.07; // This is 0.07m that is 7cm of side length
        cv::aruco::estimatePoseSingleMarkers(markerCorners, markerLength, cameraMatrix, distortionCoefficients, rvecs, tvecs);

        if( rvecs.size() != 1)
        {
            throw std::runtime_error("We have either no pose or more than 1 pose");
        }
        Vec3d rvec=rvecs[0];
        Vec3d tvec=tvecs[0];

        cout << "ARUCO pose estimation:" <<  endl;
        printRigidBodyTransformInfo(rvec,tvec);
        float axisLength=0.1; // 10cm
        aruco::drawAxis(inputImage, cameraMatrix, distortionCoefficients, rvec, tvec , axisLength);

        /* Let's compare it to direcly doing solvePnP()*/
        float s=markerLength/2.0;
        vector<Point3f> markerObjPoints={ Point3f(-s, s,0), Point3f(s,s,0), Point3f(s,-s,0), Point3f(-s,-s,0) };
        //Mat markerObjPoints;
        //getSingleMarkerObjectPointsDouble(markerLength,markerObjPoints);
        cv::solvePnP(markerObjPoints, markerCorners[0], cameraMatrix, distortionCoefficients,rvec,tvec);

        cout << "solvePnP() pose estimation:" <<  endl;
        printRigidBodyTransformInfo(rvec,tvec);



        //-- Show output image
        imshow("detection", inputImage );
        waitKey(0);
    }
    catch (std::exception& ex)

    {
        cout << "Exception :" << ex.what() << endl;
    }
    std::cin.ignore();
}
