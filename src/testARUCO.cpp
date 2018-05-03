
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


int main()
{
    try
    {
        //--  Read the input image wiht the ARUCO target
        //Mat inputImage = imread("data/test_pose_Gazebo.jpg",  IMREAD_COLOR);
        Mat inputImage = imread("data/Gazebo_testInitialPose100.jpg",  IMREAD_COLOR);
        int wantedId=100; // That's the ID we are looking for
        float markerLength = 0.07; // This is 0.07m that is 7cm of side length


        vector<int> markerIds;
        vector<vector<Point2f>> markerCorners, rejectedCandidates;
        cv::Ptr<aruco::DetectorParameters> parameters = aruco::DetectorParameters::create(); //default params
        /* We set the same parameters that we have in python*/
        parameters->minMarkerPerimeterRate = 0.1 ;
        parameters->cornerRefinementMethod = aruco::CORNER_REFINE_SUBPIX;
        cv::Ptr<cv::aruco::Dictionary> dictionary = cv::aruco::getPredefinedDictionary(cv::aruco::DICT_ARUCO_ORIGINAL);


        cv::aruco::detectMarkers(inputImage, dictionary, markerCorners, markerIds, parameters, rejectedCandidates);
        for(auto id: markerIds)
        {
             cout << "Detect id " << id << endl; 
        }

        // Draw accepted markers
        if(markerIds.size() > 0)
        {
            cv::aruco::drawDetectedMarkers(inputImage, markerCorners, markerIds);
        }

        // Draw rejected candidates
        if(rejectedCandidates.size() > 0)
        {
            Scalar pink(157, 35,228);
            cv::aruco::drawDetectedMarkers(inputImage,rejectedCandidates,noArray(), pink);
        }

        /* Get the corners of the wanted ID (the only one we are looking for)*/
        vector<vector<Point2f>> wantedMarkerCorners;
        for(size_t i=0; i<markerIds.size(); i++)
        {
            if(markerIds[i]==wantedId)
            {
                wantedMarkerCorners.emplace_back(markerCorners[i]);
            }
        }

        if(wantedMarkerCorners.size() == 0)
        {
            cout << "We didn't find the ID we were looking for" << endl; 
        }
        else
        {
            /* From ROS with the current camera location (obtained by "rostopic echo -n1 /camera1/camera_info") */
            //K: [866.2229887853814, 0.0, 500.5, 0.0, 866.2229887853814, 500.5, 0.0, 0.0, 1.0]
            //D: [0.0, 0.0, 0.0, 0.0, 0.0]
            //Mat cameraMatrix = (Mat1d(3, 3) << fx, 0, cx, 0, fx, cx, 0, 0, 1);
            Mat cameraMatrix = (Mat1d(3, 3) << 866.2229887853814, 0.0, 500.5, 0.0, 866.2229887853814, 500.5, 0.0, 0.0, 1.0);
            Mat distortionCoefficients = (Mat1d(1, 4) << 0, 0, 0, 0);

            vector< Vec3d > rvecs, tvecs;
            cv::aruco::estimatePoseSingleMarkers(wantedMarkerCorners, markerLength, cameraMatrix, distortionCoefficients, rvecs, tvecs);

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
            cv::solvePnP(markerObjPoints, wantedMarkerCorners[0], cameraMatrix, distortionCoefficients,rvec,tvec);

            cout << "solvePnP() pose estimation:" <<  endl;
            printRigidBodyTransformInfo(rvec,tvec);
        }


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
