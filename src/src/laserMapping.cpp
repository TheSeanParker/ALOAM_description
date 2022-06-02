// This is an advanced implementation of the algorithm described in the following paper:
//   J. Zhang and S. Singh. LOAM: Lidar Odometry and Mapping in Real-time.
//     Robotics: Science and Systems Conference (RSS). Berkeley, CA, July 2014. 

// Modifier: Tong Qin               qintonguav@gmail.com
// 	         Shaozu Cao 		    saozu.cao@connect.ust.hk


// Copyright 2013, Ji Zhang, Carnegie Mellon University
// Further contributions copyright (c) 2016, Southwest Research Institute
// All rights reserved.
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions are met:
//
// 1. Redistributions of source code must retain the above copyright notice,
//    this list of conditions and the following disclaimer.
// 2. Redistributions in binary form must reproduce the above copyright notice,
//    this list of conditions and the following disclaimer in the documentation
//    and/or other materials provided with the distribution.
// 3. Neither the name of the copyright holder nor the names of its
//    contributors may be used to endorse or promote products derived from this
//    software without specific prior written permission.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
// AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
// IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
// ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
// LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
// CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
// SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
// INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
// CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
// ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
// POSSIBILITY OF SUCH DAMAGE.

#include <math.h>
#include <vector>
#include <aloam_velodyne/common.h>
#include <nav_msgs/Odometry.h>
#include <nav_msgs/Path.h>
#include <geometry_msgs/PoseStamped.h>
#include <pcl_conversions/pcl_conversions.h>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/kdtree/kdtree_flann.h>
#include <ros/ros.h>
#include <sensor_msgs/Imu.h>
#include <sensor_msgs/PointCloud2.h>
#include <tf/transform_datatypes.h>
#include <tf/transform_broadcaster.h>
#include <eigen3/Eigen/Dense>
#include <ceres/ceres.h>
#include <mutex>
#include <queue>
#include <thread>
#include <iostream>
#include <string>

#include "lidarFactor.hpp"
#include "aloam_velodyne/common.h"
#include "aloam_velodyne/tic_toc.h"


int frameCount = 0;

double timeLaserCloudCornerLast = 0;
double timeLaserCloudSurfLast = 0;
double timeLaserCloudFullRes = 0;
double timeLaserOdometry = 0;


int laserCloudCenWidth = 10;
int laserCloudCenHeight = 10;
int laserCloudCenDepth = 5;
const int laserCloudWidth = 21;
const int laserCloudHeight = 21;
const int laserCloudDepth = 11;

const int laserCloudNum = laserCloudWidth * laserCloudHeight * laserCloudDepth; //21*21*11=4851

int laserCloudValidInd[125];
int laserCloudSurroundInd[125];

// input: from odom
pcl::PointCloud<PointType>::Ptr laserCloudCornerLast(new pcl::PointCloud<PointType>());
pcl::PointCloud<PointType>::Ptr laserCloudSurfLast(new pcl::PointCloud<PointType>());

// ouput: all visualble cube points
pcl::PointCloud<PointType>::Ptr laserCloudSurround(new pcl::PointCloud<PointType>());

// surround points in map to build tree
pcl::PointCloud<PointType>::Ptr laserCloudCornerFromMap(new pcl::PointCloud<PointType>());
pcl::PointCloud<PointType>::Ptr laserCloudSurfFromMap(new pcl::PointCloud<PointType>());

//input & output: points in one frame. local --> global
pcl::PointCloud<PointType>::Ptr laserCloudFullRes(new pcl::PointCloud<PointType>());

// points in every cube
pcl::PointCloud<PointType>::Ptr laserCloudCornerArray[laserCloudNum];
pcl::PointCloud<PointType>::Ptr laserCloudSurfArray[laserCloudNum];

//kd-tree
pcl::KdTreeFLANN<PointType>::Ptr kdtreeCornerFromMap(new pcl::KdTreeFLANN<PointType>());
pcl::KdTreeFLANN<PointType>::Ptr kdtreeSurfFromMap(new pcl::KdTreeFLANN<PointType>());

double parameters[7] = {0, 0, 0, 1, 0, 0, 0};
Eigen::Map<Eigen::Quaterniond> q_w_curr(parameters);
Eigen::Map<Eigen::Vector3d> t_w_curr(parameters + 4);//C++将数组名的地址解释为首元素的地址

// wmap_T_odom * odom_T_curr = wmap_T_curr;
// transformation between odom's world and map's world frame，地图坐标系和odom坐标系的位姿变换
Eigen::Quaterniond q_wmap_wodom(1, 0, 0, 0);
Eigen::Vector3d t_wmap_wodom(0, 0, 0);

Eigen::Quaterniond q_wodom_curr(1, 0, 0, 0);
Eigen::Vector3d t_wodom_curr(0, 0, 0);


std::queue<sensor_msgs::PointCloud2ConstPtr> cornerLastBuf;//queue队列是先进先出，deque是两端都可以进出，queue 来自deque
std::queue<sensor_msgs::PointCloud2ConstPtr> surfLastBuf;
std::queue<sensor_msgs::PointCloud2ConstPtr> fullResBuf;
std::queue<nav_msgs::Odometry::ConstPtr> odometryBuf;
std::mutex mBuf;

pcl::VoxelGrid<PointType> downSizeFilterCorner;//创建体素滤波对象
pcl::VoxelGrid<PointType> downSizeFilterSurf;  //创建体素滤波对象

std::vector<int> pointSearchInd;
std::vector<float> pointSearchSqDis;

PointType pointOri, pointSel;

ros::Publisher pubLaserCloudSurround, pubLaserCloudMap, pubLaserCloudFullRes, pubOdomAftMapped, pubOdomAftMappedHighFrec, pubLaserAfterMappedPath;

nav_msgs::Path laserAfterMappedPath;

   // set initial guess（依靠着上一祯的位姿，设定初始的位姿猜测）
   // 将odom坐标系变换到map坐标系中，odom和map一开始的坐标是相近的，但是随着时间的推移，二者将会产生极大的偏差
   /*(1)针对这块的坐标系逻辑，统一梳理一下；首先是各个激光点的坐标系用curr表示
     (2)当进行了odom节点处理以后的激光点以后，将会得到odom坐标系中的坐标
	 (3)进行mapping算法匹配时，首先需要将odom里的坐标先转换到map坐标系，然后再统一的map坐标系中才能进行匹配算法计算
	 (4)当进行完第3步以后，我们就可以知道一个激光点在world坐标系curr转换到map坐标系的数值了，如此方便针对原始点云的匹配算法求解位姿*/
void transformAssociateToMap()
   // 将odom里面的curr点旋转到map的坐标系，当得到这个位姿转换关系以后，新来的雷达原始激光点就可直接旋转到map坐标系
{
	q_w_curr = q_wmap_wodom * q_wodom_curr;
	t_w_curr = q_wmap_wodom * t_wodom_curr + t_wmap_wodom;
}
   //更新odom到map之间的位姿变换
void transformUpdate()//地图位姿更新，已知Tmap_curr和Todom_curr求出（更新）Tmap_odom
{
	q_wmap_wodom = q_w_curr * q_wodom_curr.inverse();
	t_wmap_wodom = t_w_curr - q_wmap_wodom * t_wodom_curr;
}
   //将当前坐标投影到地图坐标中去
void pointAssociateToMap(PointType const *const pi, PointType *const po)//临时一个点转过去，进行寻找
{
	Eigen::Vector3d point_curr(pi->x, pi->y, pi->z);
	Eigen::Vector3d point_w = q_w_curr * point_curr + t_w_curr;//将初始值投影到地图坐标系
	po->x = point_w.x();
	po->y = point_w.y();
	po->z = point_w.z();
	po->intensity = pi->intensity;
	//po->intensity = 1.0;
}

void pointAssociateTobeMapped(PointType const *const pi, PointType *const po)
{
	Eigen::Vector3d point_w(pi->x, pi->y, pi->z);
	Eigen::Vector3d point_curr = q_w_curr.inverse() * (point_w - t_w_curr);
	po->x = point_curr.x();
	po->y = point_curr.y();
	po->z = point_curr.z();
	po->intensity = pi->intensity;
}

void laserCloudCornerLastHandler(const sensor_msgs::PointCloud2ConstPtr &laserCloudCornerLast2)
{
	mBuf.lock();
	cornerLastBuf.push(laserCloudCornerLast2);//一个handler就是一个线程
	mBuf.unlock();
}

void laserCloudSurfLastHandler(const sensor_msgs::PointCloud2ConstPtr &laserCloudSurfLast2)
{
	mBuf.lock();
	surfLastBuf.push(laserCloudSurfLast2);
	mBuf.unlock();
}

void laserCloudFullResHandler(const sensor_msgs::PointCloud2ConstPtr &laserCloudFullRes2)
{
	mBuf.lock();
	fullResBuf.push(laserCloudFullRes2);
	mBuf.unlock();
}
//receive odomtry，然后将消息转换以下以后直接pub出去
void laserOdometryHandler(const nav_msgs::Odometry::ConstPtr &laserOdometry)
{
	mBuf.lock();
	odometryBuf.push(laserOdometry);//push()：inserts element at the end 
	mBuf.unlock();
	// high frequence publish
	Eigen::Quaterniond q_wodom_curr;//隐藏了全局同名变量
	Eigen::Vector3d t_wodom_curr;
	//下面是将odom获取到的消息转换到eigen里面的消息格式,然后借助eigen进行处理
	q_wodom_curr.x() = laserOdometry->pose.pose.orientation.x;
	q_wodom_curr.y() = laserOdometry->pose.pose.orientation.y;
	q_wodom_curr.z() = laserOdometry->pose.pose.orientation.z;
	q_wodom_curr.w() = laserOdometry->pose.pose.orientation.w;
	t_wodom_curr.x() = laserOdometry->pose.pose.position.x;
	t_wodom_curr.y() = laserOdometry->pose.pose.position.y;
	t_wodom_curr.z() = laserOdometry->pose.pose.position.z;
    //下面函数和transformAssociateToMap_()函数一模一样
	Eigen::Quaterniond q_w_curr = q_wmap_wodom * q_wodom_curr;//第1祯，map和odom的坐标变换是无差别的，随着时间累积odom的效果逐渐不如map
	Eigen::Vector3d t_w_curr = q_wmap_wodom * t_wodom_curr + t_wmap_wodom;//里程计位姿到达后，先将它转换到map坐标系下的位姿
    //在执行上面函数前，需进行transformUpdate()
	nav_msgs::Odometry odomAftMapped;//odomAftMapped就是经过坐标转换将odom坐标系转换到map坐标系
	odomAftMapped.header.frame_id = "/camera_init";
	odomAftMapped.child_frame_id = "/aft_mapped";
	odomAftMapped.header.stamp = laserOdometry->header.stamp;
	odomAftMapped.pose.pose.orientation.x = q_w_curr.x();
	odomAftMapped.pose.pose.orientation.y = q_w_curr.y();
	odomAftMapped.pose.pose.orientation.z = q_w_curr.z();
	odomAftMapped.pose.pose.orientation.w = q_w_curr.w();
	odomAftMapped.pose.pose.position.x = t_w_curr.x();
	odomAftMapped.pose.pose.position.y = t_w_curr.y();
	odomAftMapped.pose.pose.position.z = t_w_curr.z();
	pubOdomAftMappedHighFrec.publish(odomAftMapped);//想发布出去就首先转换到ros的消息格式，
}
//主处理线程
void process()
{
	while(1)
	{
		//之前把消息(通过subscribe进行的缓存)都存储在了buff里面,现在从buff里面进行取值.
		//下面要进行判断,确保buff里面数据是非空的?这块还是有疑问的
		while (!cornerLastBuf.empty() && !surfLastBuf.empty() && !fullResBuf.empty() && !odometryBuf.empty())
		{
			mBuf.lock();
			//下面以cornerLastBuf为基准,把时间戳小于其数值的,全部pop出去(扔出去).为什么以cornerLastBuf作为基准？
			//下面的3个循环都是采取这样的模式,好处1 避免内存存储的数据过多 2使得后端地图取得的数据都是最新的不是延时的
			while (!odometryBuf.empty() && odometryBuf.front()->header.stamp.toSec() < cornerLastBuf.front()->header.stamp.toSec())
				    odometryBuf.pop();//pop()  :removes the first element 
					                  //front():access  the first element 
									  //empty():checks whether the underlying container is empty 
			if (odometryBuf.empty())
			{
				mBuf.unlock();
				break;
			}

			while (!surfLastBuf.empty() && surfLastBuf.front()->header.stamp.toSec() < cornerLastBuf.front()->header.stamp.toSec())
				    surfLastBuf.pop();
			if (surfLastBuf.empty())
			{
				mBuf.unlock();
				break;
			}

			while (!fullResBuf.empty() && fullResBuf.front()->header.stamp.toSec() < cornerLastBuf.front()->header.stamp.toSec())
				    fullResBuf.pop();
			if (fullResBuf.empty())
			{
				mBuf.unlock();
				break;
			}
            
			timeLaserCloudCornerLast = cornerLastBuf.front()->header.stamp.toSec();
			timeLaserCloudSurfLast   =   surfLastBuf.front()->header.stamp.toSec();
			timeLaserCloudFullRes    =    fullResBuf.front()->header.stamp.toSec();
			timeLaserOdometry        =   odometryBuf.front()->header.stamp.toSec();
            //以上原则上就是进行了时间同步处理，使得这4个数据的时间戳是相同的,但是它的时间戳据为什么是完全一样的？
			if (timeLaserCloudCornerLast != timeLaserOdometry ||
				timeLaserCloudSurfLast   != timeLaserOdometry ||
				timeLaserCloudFullRes    != timeLaserOdometry)
			{
				printf("time corner %f surf %f full %f odom %f \n", timeLaserCloudCornerLast, timeLaserCloudSurfLast, timeLaserCloudFullRes, timeLaserOdometry);
				printf("unsync messeage!");
				mBuf.unlock();
				break;
			}
            
			laserCloudCornerLast->clear();
			pcl::fromROSMsg(*cornerLastBuf.front(), *laserCloudCornerLast);
			cornerLastBuf.pop();

			laserCloudSurfLast->clear();
			pcl::fromROSMsg(*surfLastBuf.front(), *laserCloudSurfLast);//点云全部转换为pcl的数据格式
			surfLastBuf.pop();

			laserCloudFullRes->clear();
			pcl::fromROSMsg(*fullResBuf.front(), *laserCloudFullRes);
			fullResBuf.pop();
            //lidar odom的结果转换为eigen的数据格式
			//将laserOdometryHandler()函数里，odom传来的数据再提取出来一次，将q_wodom_curr、t_wodom_curr更新
			q_wodom_curr.x() = odometryBuf.front()->pose.pose.orientation.x;
			q_wodom_curr.y() = odometryBuf.front()->pose.pose.orientation.y;
			q_wodom_curr.z() = odometryBuf.front()->pose.pose.orientation.z;
			q_wodom_curr.w() = odometryBuf.front()->pose.pose.orientation.w;
			t_wodom_curr.x() = odometryBuf.front()->pose.pose.position.x;
			t_wodom_curr.y() = odometryBuf.front()->pose.pose.position.y;
			t_wodom_curr.z() = odometryBuf.front()->pose.pose.position.z;
			odometryBuf.pop();
            //考虑实时性，把队列里其他的都pop出去，不然可能会出现处理延时问题
			while(!cornerLastBuf.empty())
			{
				cornerLastBuf.pop();//将buf中的数据一个个的pop()出去
				printf("drop lidar frame in mapping for real time performance \n");//在mapping时候，丢掉雷达祯来达到实时性目的
			}
			mBuf.unlock();

			TicToc t_whole;
            // 先根据函数transformUpdate()更新后的q_wmap_wodom、t_wmap_wodom，再将q_wodom_curr、t_wodom_curr更新后，最后调用下面函数得到当前地图下的位姿
			transformAssociateToMap();//此函数最关键一点是，没有对点进行移动，只更新了q_w_curr、t_w_curr量值而已
			// 根据前端结果，得到后端的一个相对准确的初值,与laserOdometryHandler()句柄函数中的转换方式一模一样

			TicToc t_shift;
			// 下面展示的是动态调整栅格地图的过程
			int centerCubeI = int((t_w_curr.x() + 25.0) / 50.0) + laserCloudCenWidth;
			int centerCubeJ = int((t_w_curr.y() + 25.0) / 50.0) + laserCloudCenHeight;
			int centerCubeK = int((t_w_curr.z() + 25.0) / 50.0) + laserCloudCenDepth;

			if (t_w_curr.x() + 25.0 < 0)
				centerCubeI--;
			if (t_w_curr.y() + 25.0 < 0)
				centerCubeJ--;
			if (t_w_curr.z() + 25.0 < 0)
				centerCubeK--;
            //如果当前格栅索引小于3，说明当前点快接近地图边界了，需要进行调整，相当于地图整体往x轴正方向移动
			while (centerCubeI < 3)
			{
				for (int j = 0; j < laserCloudHeight; j++)
				{
					for (int k = 0; k < laserCloudDepth; k++)
					{ 
						int i = laserCloudWidth - 1;
						pcl::PointCloud<PointType>::Ptr laserCloudCubeCornerPointer =
						    laserCloudCornerArray[i + laserCloudWidth * j + laserCloudWidth * laserCloudHeight * k]; 
							//用一维连续的数组来表示三维的空间，相当于知道了i、j、k之后，我们研究如果用一维坐标去表示3维坐标
							//在C++算法里设计的数组是非常具有启发意义的设计方式
						pcl::PointCloud<PointType>::Ptr laserCloudCubeSurfPointer =
							                            laserCloudSurfArray[i + laserCloudWidth * j + laserCloudWidth * laserCloudHeight * k];
						//整体右移
						for (; i >= 1; i--)//initial value在前面已经定义好了的
						{
							laserCloudCornerArray[i + laserCloudWidth * j + laserCloudWidth * laserCloudHeight * k] =
								laserCloudCornerArray[i - 1 + laserCloudWidth * j + laserCloudWidth * laserCloudHeight * k];
							laserCloudSurfArray[i + laserCloudWidth * j + laserCloudWidth * laserCloudHeight * k] =
								laserCloudSurfArray[i - 1 + laserCloudWidth * j + laserCloudWidth * laserCloudHeight * k];
						}
						//此时i=0，
						laserCloudCornerArray[i + laserCloudWidth * j + laserCloudWidth * laserCloudHeight * k] =laserCloudCubeCornerPointer;
						laserCloudSurfArray[  i + laserCloudWidth * j + laserCloudWidth * laserCloudHeight * k] =laserCloudCubeSurfPointer;
						//该点云清零，由于是指针操作，相当于
						laserCloudCubeCornerPointer->clear();
						laserCloudCubeSurfPointer->clear();
					}
				}
                //索引右移
				centerCubeI++;
				laserCloudCenWidth++;
			}

			while (centerCubeI >= laserCloudWidth - 3)
			{ 
				for (int j = 0; j < laserCloudHeight; j++)
				{
					for (int k = 0; k < laserCloudDepth; k++)
					{
						int i = 0;
						pcl::PointCloud<PointType>::Ptr laserCloudCubeCornerPointer =
							laserCloudCornerArray[i + laserCloudWidth * j + laserCloudWidth * laserCloudHeight * k];
						pcl::PointCloud<PointType>::Ptr laserCloudCubeSurfPointer =
							laserCloudSurfArray[i + laserCloudWidth * j + laserCloudWidth * laserCloudHeight * k];
						for (; i < laserCloudWidth - 1; i++)
						{
							laserCloudCornerArray[i + laserCloudWidth * j + laserCloudWidth * laserCloudHeight * k] =
								laserCloudCornerArray[i + 1 + laserCloudWidth * j + laserCloudWidth * laserCloudHeight * k];
							laserCloudSurfArray[i + laserCloudWidth * j + laserCloudWidth * laserCloudHeight * k] =
								laserCloudSurfArray[i + 1 + laserCloudWidth * j + laserCloudWidth * laserCloudHeight * k];
						}
						laserCloudCornerArray[i + laserCloudWidth * j + laserCloudWidth * laserCloudHeight * k] =
							laserCloudCubeCornerPointer;
						laserCloudSurfArray[i + laserCloudWidth * j + laserCloudWidth * laserCloudHeight * k] =
							laserCloudCubeSurfPointer;
						laserCloudCubeCornerPointer->clear();
						laserCloudCubeSurfPointer->clear();
					}
				}

				centerCubeI--;
				laserCloudCenWidth--;
			}

			while (centerCubeJ < 3)
			{
				for (int i = 0; i < laserCloudWidth; i++)
				{
					for (int k = 0; k < laserCloudDepth; k++)
					{
						int j = laserCloudHeight - 1;
						pcl::PointCloud<PointType>::Ptr laserCloudCubeCornerPointer =
							laserCloudCornerArray[i + laserCloudWidth * j + laserCloudWidth * laserCloudHeight * k];
						pcl::PointCloud<PointType>::Ptr laserCloudCubeSurfPointer =
							laserCloudSurfArray[i + laserCloudWidth * j + laserCloudWidth * laserCloudHeight * k];
						for (; j >= 1; j--)
						{
							laserCloudCornerArray[i + laserCloudWidth * j + laserCloudWidth * laserCloudHeight * k] =
								laserCloudCornerArray[i + laserCloudWidth * (j - 1) + laserCloudWidth * laserCloudHeight * k];
							laserCloudSurfArray[i + laserCloudWidth * j + laserCloudWidth * laserCloudHeight * k] =
								laserCloudSurfArray[i + laserCloudWidth * (j - 1) + laserCloudWidth * laserCloudHeight * k];
						}
						laserCloudCornerArray[i + laserCloudWidth * j + laserCloudWidth * laserCloudHeight * k] =
							laserCloudCubeCornerPointer;
						laserCloudSurfArray[i + laserCloudWidth * j + laserCloudWidth * laserCloudHeight * k] =
							laserCloudCubeSurfPointer;
						laserCloudCubeCornerPointer->clear();
						laserCloudCubeSurfPointer->clear();
					}
				}

				centerCubeJ++;
				laserCloudCenHeight++;
			}

			while (centerCubeJ >= laserCloudHeight - 3)
			{
				for (int i = 0; i < laserCloudWidth; i++)
				{
					for (int k = 0; k < laserCloudDepth; k++)
					{
						int j = 0;
						pcl::PointCloud<PointType>::Ptr laserCloudCubeCornerPointer =
							laserCloudCornerArray[i + laserCloudWidth * j + laserCloudWidth * laserCloudHeight * k];
						pcl::PointCloud<PointType>::Ptr laserCloudCubeSurfPointer =
							laserCloudSurfArray[i + laserCloudWidth * j + laserCloudWidth * laserCloudHeight * k];
						for (; j < laserCloudHeight - 1; j++)
						{
							laserCloudCornerArray[i + laserCloudWidth * j + laserCloudWidth * laserCloudHeight * k] =
								laserCloudCornerArray[i + laserCloudWidth * (j + 1) + laserCloudWidth * laserCloudHeight * k];
							laserCloudSurfArray[i + laserCloudWidth * j + laserCloudWidth * laserCloudHeight * k] =
								laserCloudSurfArray[i + laserCloudWidth * (j + 1) + laserCloudWidth * laserCloudHeight * k];
						}
						laserCloudCornerArray[i + laserCloudWidth * j + laserCloudWidth * laserCloudHeight * k] =
							laserCloudCubeCornerPointer;
						laserCloudSurfArray[i + laserCloudWidth * j + laserCloudWidth * laserCloudHeight * k] =
							laserCloudCubeSurfPointer;
						laserCloudCubeCornerPointer->clear();
						laserCloudCubeSurfPointer->clear();
					}
				}

				centerCubeJ--;
				laserCloudCenHeight--;
			}

			while (centerCubeK < 3)
			{
				for (int i = 0; i < laserCloudWidth; i++)
				{
					for (int j = 0; j < laserCloudHeight; j++)
					{
						int k = laserCloudDepth - 1;
						pcl::PointCloud<PointType>::Ptr laserCloudCubeCornerPointer =
							laserCloudCornerArray[i + laserCloudWidth * j + laserCloudWidth * laserCloudHeight * k];
						pcl::PointCloud<PointType>::Ptr laserCloudCubeSurfPointer =
							laserCloudSurfArray[i + laserCloudWidth * j + laserCloudWidth * laserCloudHeight * k];
						for (; k >= 1; k--)
						{
							laserCloudCornerArray[i + laserCloudWidth * j + laserCloudWidth * laserCloudHeight * k] =
								laserCloudCornerArray[i + laserCloudWidth * j + laserCloudWidth * laserCloudHeight * (k - 1)];
							laserCloudSurfArray[i + laserCloudWidth * j + laserCloudWidth * laserCloudHeight * k] =
								laserCloudSurfArray[i + laserCloudWidth * j + laserCloudWidth * laserCloudHeight * (k - 1)];
						}
						laserCloudCornerArray[i + laserCloudWidth * j + laserCloudWidth * laserCloudHeight * k] =
							laserCloudCubeCornerPointer;
						laserCloudSurfArray[i + laserCloudWidth * j + laserCloudWidth * laserCloudHeight * k] =
							laserCloudCubeSurfPointer;
						laserCloudCubeCornerPointer->clear();
						laserCloudCubeSurfPointer->clear();
					}
				}

				centerCubeK++;
				laserCloudCenDepth++;
			}

			while (centerCubeK >= laserCloudDepth - 3)
			{
				for (int i = 0; i < laserCloudWidth; i++)
				{
					for (int j = 0; j < laserCloudHeight; j++)
					{
						int k = 0;
						pcl::PointCloud<PointType>::Ptr laserCloudCubeCornerPointer =
							laserCloudCornerArray[i + laserCloudWidth * j + laserCloudWidth * laserCloudHeight * k];
						pcl::PointCloud<PointType>::Ptr laserCloudCubeSurfPointer =
							laserCloudSurfArray[i + laserCloudWidth * j + laserCloudWidth * laserCloudHeight * k];
						for (; k < laserCloudDepth - 1; k++)
						{
							laserCloudCornerArray[i + laserCloudWidth * j + laserCloudWidth * laserCloudHeight * k] =
								laserCloudCornerArray[i + laserCloudWidth * j + laserCloudWidth * laserCloudHeight * (k + 1)];
							laserCloudSurfArray[i + laserCloudWidth * j + laserCloudWidth * laserCloudHeight * k] =
								laserCloudSurfArray[i + laserCloudWidth * j + laserCloudWidth * laserCloudHeight * (k + 1)];
						}
						laserCloudCornerArray[i + laserCloudWidth * j + laserCloudWidth * laserCloudHeight * k] =
							laserCloudCubeCornerPointer;
						laserCloudSurfArray[i + laserCloudWidth * j + laserCloudWidth * laserCloudHeight * k] =
							laserCloudCubeSurfPointer;
						laserCloudCubeCornerPointer->clear();
						laserCloudCubeSurfPointer->clear();
					}
				}

				centerCubeK--;
				laserCloudCenDepth--;
			}
            //以上操作相当于维护了一个局部地图，保证当前祯不在这个局部地图的边缘，这样才可以从地图中获取足够的约束。才能够保证约束的效果是良好的
			//**往下才是真正核心的算法匹配地方**
			int laserCloudValidNum = 0;
			int laserCloudSurroundNum = 0;
            //从当前格子为中心，选出地图中一定范围的点云
			for (int i = centerCubeI - 2; i <= centerCubeI + 2; i++)
			{
				for (int j = centerCubeJ - 2; j <= centerCubeJ + 2; j++)
				{
					for (int k = centerCubeK - 1; k <= centerCubeK + 1; k++)
					{
						if (i >= 0 && i < laserCloudWidth &&
							j >= 0 && j < laserCloudHeight &&
							k >= 0 && k < laserCloudDepth)
					        //上面是检测索引不会越界
						{   //把各自有效的点云地图索引记录下来
							laserCloudValidInd[laserCloudValidNum] = i + laserCloudWidth * j + laserCloudWidth * laserCloudHeight * k;
							laserCloudValidNum++;
							laserCloudSurroundInd[laserCloudSurroundNum] = i + laserCloudWidth * j + laserCloudWidth * laserCloudHeight * k;
							laserCloudSurroundNum++;
						}
					}
				}
			}

			laserCloudCornerFromMap->clear();
			laserCloudSurfFromMap->clear();
			//开始构建用来这一祯优化局部地图的子图
			for (int i = 0; i < laserCloudValidNum; i++)
			{
				*laserCloudCornerFromMap += *laserCloudCornerArray[laserCloudValidInd[i]];//通过指针的形式，在内存上完成点云地图上的“累加”
				std::cout<<"laserCloudValidNum="<<laserCloudValidNum<<"输出数据="<<*laserCloudSurfFromMap<<std::endl;
				*laserCloudSurfFromMap += *laserCloudSurfArray[laserCloudValidInd[i]];
			}
			int laserCloudCornerFromMapNum = laserCloudCornerFromMap->points.size();
			int laserCloudSurfFromMapNum = laserCloudSurfFromMap->points.size();

            //为了减少运算量,对上一祯odom节点传来的点云进行下采样
			pcl::PointCloud<PointType>::Ptr laserCloudCornerStack(new pcl::PointCloud<PointType>());
			downSizeFilterCorner.setInputCloud(laserCloudCornerLast);
			downSizeFilterCorner.filter(*laserCloudCornerStack);
			int laserCloudCornerStackNum = laserCloudCornerStack->points.size();//下采样后点的个数

			pcl::PointCloud<PointType>::Ptr laserCloudSurfStack(new pcl::PointCloud<PointType>());
			downSizeFilterSurf.setInputCloud(laserCloudSurfLast);//第3步，给滤波对象设置需要过滤的点云
			downSizeFilterSurf.filter(*laserCloudSurfStack);     //第4步，执行滤波处理，输出滤波后的点云laserCloudSurfStack
			int laserCloudSurfStackNum = laserCloudSurfStack->points.size();
            //到此整个子图构建结束，接下来计算特征和子图的匹配
			printf("map prepare time %f ms\n", t_shift.toc());
			printf("map corner num %d  surf num %d \n", laserCloudCornerFromMapNum, laserCloudSurfFromMapNum);
			//对地图点数进行判断,不能太少 
			if (laserCloudCornerFromMapNum > 10 && laserCloudSurfFromMapNum > 50)//中心cube附近的激光特征点大于一定数量
			{
				TicToc t_opt;
				TicToc t_tree;
				kdtreeCornerFromMap->setInputCloud(laserCloudCornerFromMap);//设置kdtree 将局部地图子图加入到kdtree中
				kdtreeSurfFromMap->setInputCloud(laserCloudSurfFromMap);
				//建立kdtree需要很多的时间,所以需要关注一下它具体花费了多少的时间
				printf("build tree time %f ms \n", t_tree.toc());
                //建立对应关系的迭代次数不超2次
				for (int iterCount = 0; iterCount < 2; iterCount++)//根odom前端线程类似，进行2次大循环优化迭代
				{
					//ceres::LossFunction *loss_function = NULL;
					//建立核函数目的是为了避免一些outlier的出现,这里用的是Huber核函数
					ceres::LossFunction *loss_function = new ceres::HuberLoss(0.1);
                    //下行，4元数并不满足广义上的加法,所以需要进行借助ceres已经定义好的一个类EigenQuaternionParameterization()进行运算
					ceres::LocalParameterization *q_parameterization =new ceres::EigenQuaternionParameterization();
					ceres::Problem::Options problem_options;

					ceres::Problem problem(problem_options);
					//下面是构建ceres问题,添加参数块的一个过程,下面两行将旋转和平移分割开了.旋转是前4位,平移是后3位
					problem.AddParameterBlock(parameters, 4, q_parameterization);
					//上面的旋转第一个变量的存在是因为q_parameterization不满足广义意义上的加法,所以要定义一个localparameterization
					problem.AddParameterBlock(parameters + 4, 3);

					TicToc t_data;
					int corner_num = 0;
                    //下面是在地图中寻找构建相似的corner点约束,与前端的构建方法是相似的
					for (int i = 0; i < laserCloudCornerStackNum; i++)
					{
						pointOri = laserCloudCornerStack->points[i];//针对上一祯点云数据进行下采样后的每一个激光点为搜索原点进行kdtree搜索
						//double sqrtDis = pointOri.x * pointOri.x + pointOri.y * pointOri.y + pointOri.z * pointOri.z;
		                //将当前点投影到地图坐标系中,便于Kdtree检索,做地图匹配时,将两帧点云投影到同一个坐标系下的习惯;
						pointAssociateToMap(&pointOri, &pointSel);//把当前点根据初值投影到地图坐标系中去，sel：select
						kdtreeCornerFromMap->nearestKSearch(pointSel, 5, pointSearchInd, pointSearchSqDis); 
						//kdtree搜索的范围是局部地图的子图
						//由于在地图里面寻找线特征点和面特征点的约束对，没有线号信息，我们需要额外的操作来判断激光点是否复合线特征和面特征给定的约束
                        //判断最远的点距离不能超过1m,否则就是无效约束。kdtree搜索时候，最近的点下标是[0],最远点的下标是[4]
						if (pointSearchSqDis[4] < 1.0)
						{ 
							std::vector<Eigen::Vector3d> nearCorners;
							Eigen::Vector3d center(0, 0, 0);
							for (int j = 0; j < 5; j++)
							{
								Eigen::Vector3d tmp(laserCloudCornerFromMap->points[pointSearchInd[j]].x,
													laserCloudCornerFromMap->points[pointSearchInd[j]].y,
													laserCloudCornerFromMap->points[pointSearchInd[j]].z);
								center = center + tmp;
								nearCorners.push_back(tmp);
							}
							center = center / 5.0;//求出5个点的均值

							Eigen::Matrix3d covMat = Eigen::Matrix3d::Zero();
							//构建协方差矩阵
							for (int j = 0; j < 5; j++)
							{
								Eigen::Matrix<double, 3, 1> tmpZeroMean = nearCorners[j] - center;//坐标祛质心
								covMat = covMat + tmpZeroMean * tmpZeroMean.transpose();//// 这5个点的协方差,?这不就是平均值么?怎么能叫做协方差呢?
							}
                            //进行特征值分解
							Eigen::SelfAdjointEigenSolver<Eigen::Matrix3d> saes(covMat);

							// if is indeed line feature
							// note Eigen library sort eigenvalues in increasing order
							//根据特征值分解情况看看是不是真正的线特征
							//特征向量就是线特征的方向
							Eigen::Vector3d unit_direction = saes.eigenvectors().col(2);
							Eigen::Vector3d curr_point(pointOri.x, pointOri.y, pointOri.z);
							//判断最大特征值大于次大特征值的3倍,则认为是线特征
							
							if (saes.eigenvalues()[2] > 3 * saes.eigenvalues()[1])//这里相当于判断这些点的分布是否足够成线状分布
							//上面的if满足,就说明这5个点,的确在同一个线束上面去
							//然后需要如何把这5个点告诉优化器
							{ 
								Eigen::Vector3d point_on_line = center;
								Eigen::Vector3d point_a, point_b;
					            //根据拟合出来的线特征方向,以平均点为中心构建2个虚拟点(能够比较好的模拟线特征),构建约束,和lidar odom约束一致
								point_a =  0.1 * unit_direction + point_on_line;//相当于拟合出了这条线的两个端点
								point_b = -0.1 * unit_direction + point_on_line;
                                //计算点到直线距离作为误差函数，这里没有运动畸变，不需要插值，所以s=1
								ceres::CostFunction *cost_function = LidarEdgeFactor::Create(curr_point, point_a, point_b, 1.0);
								problem.AddResidualBlock(cost_function, loss_function, parameters, parameters + 4);
								corner_num++;	
							}							
						}
						/*
						else if(pointSearchSqDis[4] < 0.01 * sqrtDis)
						{
							Eigen::Vector3d center(0, 0, 0);
							for (int j = 0; j < 5; j++)
							{
								Eigen::Vector3d tmp(laserCloudCornerFromMap->points[pointSearchInd[j]].x,
													laserCloudCornerFromMap->points[pointSearchInd[j]].y,
													laserCloudCornerFromMap->points[pointSearchInd[j]].z);
								center = center + tmp;
							}
							center = center / 5.0;	
							Eigen::Vector3d curr_point(pointOri.x, pointOri.y, pointOri.z);
							ceres::CostFunction *cost_function = LidarDistanceFactor::Create(curr_point, center);
							problem.AddResidualBlock(cost_function, loss_function, parameters, parameters + 4);
						}
						*/
					}

					int surf_num = 0;
					for (int i = 0; i < laserCloudSurfStackNum; i++)
					{
						pointOri = laserCloudSurfStack->points[i];
						//double sqrtDis = pointOri.x * pointOri.x + pointOri.y * pointOri.y + pointOri.z * pointOri.z;
						//同样先将特征点转换到世界坐标系下，然后找5个最近点（区别于odom只找3个线特征点）
						pointAssociateToMap(&pointOri, &pointSel);
						kdtreeSurfFromMap->nearestKSearch(pointSel, 5, pointSearchInd, pointSearchSqDis);

						Eigen::Matrix<double, 5, 3> matA0;
						Eigen::Matrix<double, 5, 1> matB0 = -1 * Eigen::Matrix<double, 5, 1>::Ones();
						if (pointSearchSqDis[4] < 1.0)
						{
							
							for (int j = 0; j < 5; j++)
							{
								matA0(j, 0) = laserCloudSurfFromMap->points[pointSearchInd[j]].x;
								matA0(j, 1) = laserCloudSurfFromMap->points[pointSearchInd[j]].y;
								matA0(j, 2) = laserCloudSurfFromMap->points[pointSearchInd[j]].z;
								//printf(" pts %f %f %f ", matA0(j, 0), matA0(j, 1), matA0(j, 2));
							}
							// find the norm of plane
							Eigen::Vector3d norm = matA0.colPivHouseholderQr().solve(matB0);
							double negative_OA_dot_norm = 1 / norm.norm();
							norm.normalize();//法向量归一化

							// Here n(pa, pb, pc) is unit norm of plane
							bool planeValid = true;
							for (int j = 0; j < 5; j++)
							{
								// if OX * n > 0.2, then plane is not fit well
								if (fabs(norm(0) * laserCloudSurfFromMap->points[pointSearchInd[j]].x +
										 norm(1) * laserCloudSurfFromMap->points[pointSearchInd[j]].y +
										 norm(2) * laserCloudSurfFromMap->points[pointSearchInd[j]].z + negative_OA_dot_norm) > 0.2)
								{
									planeValid = false;
									break;
								}
							}
							Eigen::Vector3d curr_point(pointOri.x, pointOri.y, pointOri.z);
							if (planeValid)
							{
								ceres::CostFunction *cost_function = LidarPlaneNormFactor::Create(curr_point, norm, negative_OA_dot_norm);
								problem.AddResidualBlock(cost_function, loss_function, parameters, parameters + 4);
								surf_num++;
							}
						}
						/*
						else if(pointSearchSqDis[4] < 0.01 * sqrtDis)
						{
							Eigen::Vector3d center(0, 0, 0);
							for (int j = 0; j < 5; j++)
							{
								Eigen::Vector3d tmp(laserCloudSurfFromMap->points[pointSearchInd[j]].x,
													laserCloudSurfFromMap->points[pointSearchInd[j]].y,
													laserCloudSurfFromMap->points[pointSearchInd[j]].z);
								center = center + tmp;
							}
							center = center / 5.0;	
							Eigen::Vector3d curr_point(pointOri.x, pointOri.y, pointOri.z);
							ceres::CostFunction *cost_function = LidarDistanceFactor::Create(curr_point, center);
							problem.AddResidualBlock(cost_function, loss_function, parameters, parameters + 4);
						}
						*/
					}

					//printf("corner num %d used corner num %d \n", laserCloudCornerStackNum, corner_num);
					//printf("surf num %d used surf num %d \n", laserCloudSurfStackNum, surf_num);

					printf("mapping data assosiation time %f ms \n", t_data.toc());

					TicToc t_solver;
					ceres::Solver::Options options;
					options.linear_solver_type = ceres::DENSE_QR;
					options.max_num_iterations = 4;
					options.minimizer_progress_to_stdout = false;
					options.check_gradients = false;
					options.gradient_check_relative_precision = 1e-4;
					ceres::Solver::Summary summary;
					ceres::Solve(options, &problem, &summary);
					printf("mapping solver time %f ms \n", t_solver.toc());

					//printf("time %f \n", timeLaserOdometry);
					//printf("corner factor num %d surf factor num %d\n", corner_num, surf_num);
					//printf("result q %f %f %f %f result t %f %f %f\n", parameters[3], parameters[0], parameters[1], parameters[2],
					//	   parameters[4], parameters[5], parameters[6]);
				}
				printf("mapping optimization time %f \n", t_opt.toc());
			}
			else
			{
				ROS_WARN("time Map corner and surf num are not enough");//这个时常会报出来
			}
			transformUpdate();//求出Tmap_curr以后，反求odom和map之间的转换关系，作为下一祯的预测

			TicToc t_add;
			for (int i = 0; i < laserCloudCornerStackNum; i++)//将这一祯的corner特征点加入到地图中
			{
				//该点根据位姿投到地图坐标系
				pointAssociateToMap(&laserCloudCornerStack->points[i], &pointSel);
                //算出这个点所在格子的索引
				int cubeI = int((pointSel.x + 25.0) / 50.0) + laserCloudCenWidth;
				int cubeJ = int((pointSel.y + 25.0) / 50.0) + laserCloudCenHeight;
				int cubeK = int((pointSel.z + 25.0) / 50.0) + laserCloudCenDepth;

				if (pointSel.x + 25.0 < 0)
					cubeI--;
				if (pointSel.y + 25.0 < 0)
					cubeJ--;
				if (pointSel.z + 25.0 < 0)
					cubeK--;

				if (cubeI >= 0 && cubeI < laserCloudWidth &&
					cubeJ >= 0 && cubeJ < laserCloudHeight &&
					cubeK >= 0 && cubeK < laserCloudDepth)
				{
					int cubeInd = cubeI + laserCloudWidth * cubeJ + laserCloudWidth * laserCloudHeight * cubeK;
					laserCloudCornerArray[cubeInd]->push_back(pointSel);
				}
			}

			for (int i = 0; i < laserCloudSurfStackNum; i++)//将这一祯的surf特征点加入到地图中去
			{
				pointAssociateToMap(&laserCloudSurfStack->points[i], &pointSel);

				int cubeI = int((pointSel.x + 25.0) / 50.0) + laserCloudCenWidth;
				int cubeJ = int((pointSel.y + 25.0) / 50.0) + laserCloudCenHeight;
				int cubeK = int((pointSel.z + 25.0) / 50.0) + laserCloudCenDepth;

				if (pointSel.x + 25.0 < 0)
					cubeI--;
				if (pointSel.y + 25.0 < 0)
					cubeJ--;
				if (pointSel.z + 25.0 < 0)
					cubeK--;

				if (cubeI >= 0 && cubeI < laserCloudWidth &&
					cubeJ >= 0 && cubeJ < laserCloudHeight &&
					cubeK >= 0 && cubeK < laserCloudDepth)
				{
					int cubeInd = cubeI + laserCloudWidth * cubeJ + laserCloudWidth * laserCloudHeight * cubeK;
					laserCloudSurfArray[cubeInd]->push_back(pointSel);
				}
			}
			printf("add points time %f ms\n", t_add.toc());

			
			TicToc t_filter;
			for (int i = 0; i < laserCloudValidNum; i++)
			{
				int ind = laserCloudValidInd[i];

				pcl::PointCloud<PointType>::Ptr tmpCorner(new pcl::PointCloud<PointType>());
				downSizeFilterCorner.setInputCloud(laserCloudCornerArray[ind]);
				downSizeFilterCorner.filter(*tmpCorner);
				laserCloudCornerArray[ind] = tmpCorner;

				pcl::PointCloud<PointType>::Ptr tmpSurf(new pcl::PointCloud<PointType>());
				downSizeFilterSurf.setInputCloud(laserCloudSurfArray[ind]);
				downSizeFilterSurf.filter(*tmpSurf);
				laserCloudSurfArray[ind] = tmpSurf;
			}
			printf("filter time %f ms \n", t_filter.toc());
			
			TicToc t_pub;
			//publish surround map for every 5 frame
			if (frameCount % 5 == 0)
			{
				laserCloudSurround->clear();
				for (int i = 0; i < laserCloudSurroundNum; i++)
				{
					int ind = laserCloudSurroundInd[i];
					*laserCloudSurround += *laserCloudCornerArray[ind];
					*laserCloudSurround += *laserCloudSurfArray[ind];
				}

				sensor_msgs::PointCloud2 laserCloudSurround3;
				pcl::toROSMsg(*laserCloudSurround, laserCloudSurround3);
				laserCloudSurround3.header.stamp = ros::Time().fromSec(timeLaserOdometry);
				laserCloudSurround3.header.frame_id = "/camera_init";
				pubLaserCloudSurround.publish(laserCloudSurround3);
			}

			if (frameCount % 20 == 0)
			{
				pcl::PointCloud<PointType> laserCloudMap;
				for (int i = 0; i < 4851; i++)
				{
					laserCloudMap += *laserCloudCornerArray[i];
					laserCloudMap += *laserCloudSurfArray[i];
				}
				sensor_msgs::PointCloud2 laserCloudMsg;
				pcl::toROSMsg(laserCloudMap, laserCloudMsg);
				laserCloudMsg.header.stamp = ros::Time().fromSec(timeLaserOdometry);
				laserCloudMsg.header.frame_id = "/camera_init";
				pubLaserCloudMap.publish(laserCloudMsg);
			}

			int laserCloudFullResNum = laserCloudFullRes->points.size();
			for (int i = 0; i < laserCloudFullResNum; i++)
			{
				pointAssociateToMap(&laserCloudFullRes->points[i], &laserCloudFullRes->points[i]);
			}

			sensor_msgs::PointCloud2 laserCloudFullRes3;
			pcl::toROSMsg(*laserCloudFullRes, laserCloudFullRes3);
			laserCloudFullRes3.header.stamp = ros::Time().fromSec(timeLaserOdometry);
			laserCloudFullRes3.header.frame_id = "/camera_init";
			pubLaserCloudFullRes.publish(laserCloudFullRes3);

			printf("mapping pub time %f ms \n", t_pub.toc());

			printf("whole mapping time %f ms +++++\n", t_whole.toc());

			nav_msgs::Odometry odomAftMapped;
			odomAftMapped.header.frame_id = "/camera_init";
			odomAftMapped.child_frame_id = "/aft_mapped";
			odomAftMapped.header.stamp = ros::Time().fromSec(timeLaserOdometry);
			odomAftMapped.pose.pose.orientation.x = q_w_curr.x();
			odomAftMapped.pose.pose.orientation.y = q_w_curr.y();
			odomAftMapped.pose.pose.orientation.z = q_w_curr.z();
			odomAftMapped.pose.pose.orientation.w = q_w_curr.w();
			odomAftMapped.pose.pose.position.x = t_w_curr.x();
			odomAftMapped.pose.pose.position.y = t_w_curr.y();
			odomAftMapped.pose.pose.position.z = t_w_curr.z();
			pubOdomAftMapped.publish(odomAftMapped);

			geometry_msgs::PoseStamped laserAfterMappedPose;
			laserAfterMappedPose.header = odomAftMapped.header;
			laserAfterMappedPose.pose = odomAftMapped.pose.pose;
			laserAfterMappedPath.header.stamp = odomAftMapped.header.stamp;
			laserAfterMappedPath.header.frame_id = "/camera_init";
			laserAfterMappedPath.poses.push_back(laserAfterMappedPose);
			pubLaserAfterMappedPath.publish(laserAfterMappedPath);

			static tf::TransformBroadcaster br;
			tf::Transform transform;
			tf::Quaternion q;
			transform.setOrigin(tf::Vector3(t_w_curr(0),
											t_w_curr(1),
											t_w_curr(2)));
			q.setW(q_w_curr.w());
			q.setX(q_w_curr.x());
			q.setY(q_w_curr.y());
			q.setZ(q_w_curr.z());
			transform.setRotation(q);
			br.sendTransform(tf::StampedTransform(transform, odomAftMapped.header.stamp, "/camera_init", "/aft_mapped"));

			frameCount++;
		}
		std::chrono::milliseconds dura(2);
        std::this_thread::sleep_for(dura);
	}
}

int main(int argc, char **argv)
{
	ros::init(argc, argv, "laserMapping");
	ros::NodeHandle nh;

	float lineRes = 0;
	float planeRes = 0;
	nh.param<float>("mapping_line_resolution", lineRes, 0.4);
	nh.param<float>("mapping_plane_resolution", planeRes, 0.8);
	printf("line resolution %f plane resolution %f \n", lineRes, planeRes);
	downSizeFilterCorner.setLeafSize(lineRes, lineRes,lineRes);//第2步，设置滤波时，创建的体素体积大小
	downSizeFilterSurf.setLeafSize(planeRes, planeRes, planeRes);

	ros::Subscriber subLaserCloudCornerLast = nh.subscribe<sensor_msgs::PointCloud2>("/laser_cloud_corner_last", 100, laserCloudCornerLastHandler);

	ros::Subscriber subLaserCloudSurfLast = nh.subscribe<sensor_msgs::PointCloud2>("/laser_cloud_surf_last", 100, laserCloudSurfLastHandler);

	ros::Subscriber subLaserOdometry = nh.subscribe<nav_msgs::Odometry>("/laser_odom_to_init", 100, laserOdometryHandler);

	ros::Subscriber subLaserCloudFullRes = nh.subscribe<sensor_msgs::PointCloud2>("/velodyne_cloud_3", 100, laserCloudFullResHandler);

	pubLaserCloudSurround = nh.advertise<sensor_msgs::PointCloud2>("/laser_cloud_surround", 100);

	pubLaserCloudMap = nh.advertise<sensor_msgs::PointCloud2>("/laser_cloud_map", 100);

	pubLaserCloudFullRes = nh.advertise<sensor_msgs::PointCloud2>("/velodyne_cloud_registered", 100);

	pubOdomAftMapped = nh.advertise<nav_msgs::Odometry>("/aft_mapped_to_init", 100);

	pubOdomAftMappedHighFrec = nh.advertise<nav_msgs::Odometry>("/aft_mapped_to_init_high_frec", 100);

	pubLaserAfterMappedPath = nh.advertise<nav_msgs::Path>("/aft_mapped_path", 100);

	for (int i = 0; i < laserCloudNum; i++)
	{
		//利用智能指针的reset接口在"堆"上(new)开辟了一个空间;不这样就会生成nullptr或者野指针
		laserCloudCornerArray[i].reset(new pcl::PointCloud<PointType>());
		laserCloudSurfArray[i].reset(new pcl::PointCloud<PointType>());
	}

	std::thread mapping_process{process};

	ros::spin();

	return 0;
}