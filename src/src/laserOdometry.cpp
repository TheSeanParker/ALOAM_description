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

#include <cmath>
#include <nav_msgs/Odometry.h>
#include <nav_msgs/Path.h>
#include <geometry_msgs/PoseStamped.h>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/kdtree/kdtree_flann.h>
#include <pcl_conversions/pcl_conversions.h>
#include <ros/ros.h>
#include <sensor_msgs/Imu.h>
#include <sensor_msgs/PointCloud2.h>
#include <tf/transform_datatypes.h>
#include <tf/transform_broadcaster.h>
#include <eigen3/Eigen/Dense>
#include <mutex>
#include <queue>
#include <iostream>

#include "aloam_velodyne/common.h"
#include "aloam_velodyne/tic_toc.h"
#include "lidarFactor.hpp"

#define DISTORTION 0
//上面属于在编译之前,预处理器执行的预处理变量,将DISTORTION翻译成对应的数值0
//上面失真为0，表示一直没有失真，就是不需要进行去除运动畸变

int corner_correspondence = 0, plane_correspondence = 0;

constexpr double SCAN_PERIOD = 0.1;//常量表达式,此类型不必利用if进行判断是否为0.1,10Hz的雷达
constexpr double DISTANCE_SQ_THRESHOLD = 25;
constexpr double NEARBY_SCAN = 2.5;

int skipFrameNum = 5;
bool systemInited = false;

double timeCornerPointsSharp = 0;
double timeCornerPointsLessSharp = 0;
double timeSurfPointsFlat = 0;
double timeSurfPointsLessFlat = 0;
double timeLaserCloudFullRes = 0;

pcl::KdTreeFLANN<pcl::PointXYZI>::Ptr kdtreeCornerLast(new pcl::KdTreeFLANN<pcl::PointXYZI>());
pcl::KdTreeFLANN<pcl::PointXYZI>::Ptr kdtreeSurfLast(new pcl::KdTreeFLANN<pcl::PointXYZI>());

pcl::PointCloud<PointType>::Ptr cornerPointsSharp(new pcl::PointCloud<PointType>());
pcl::PointCloud<PointType>::Ptr cornerPointsLessSharp(new pcl::PointCloud<PointType>());
pcl::PointCloud<PointType>::Ptr surfPointsFlat(new pcl::PointCloud<PointType>());
pcl::PointCloud<PointType>::Ptr surfPointsLessFlat(new pcl::PointCloud<PointType>());

pcl::PointCloud<PointType>::Ptr laserCloudCornerLast(new pcl::PointCloud<PointType>());//等价于赋值初始化，但统一初始化{}效率更高
pcl::PointCloud<PointType>::Ptr laserCloudSurfLast(new pcl::PointCloud<PointType>());
pcl::PointCloud<PointType>::Ptr laserCloudFullRes(new pcl::PointCloud<PointType>());

int laserCloudCornerLastNum = 0;
int laserCloudSurfLastNum = 0;

//将当前帧的位姿转到世界坐标系下面去// Transformation from current frame to world frame
Eigen::Quaterniond q_w_curr(1, 0, 0, 0);//实部分是序号1,这是一个没有任何旋转的四元数;ros里表达四元数的方法是(0,0,0,1),w放在最后
Eigen::Vector3d t_w_curr(0, 0, 0);//动态长度double型列向量

// q_curr_last(x, y, z, w), t_curr_last
//定义好变换矩阵R(4元数的四个参数)及平移的三个参数
double para_q[4] = {0, 0, 0, 1};//实部为序号4,这是一个没有任何旋转的四元数
double para_t[3] = {0, 0, 0};

//给R和t赋予初值,同时二者还是全局变量
Eigen::Map<Eigen::Quaterniond> q_last_curr(para_q);
Eigen::Map<Eigen::Vector3d> t_last_curr(para_t);

std::queue<sensor_msgs::PointCloud2ConstPtr> cornerSharpBuf;
//typedef boost::shared_ptr< ::sensor_msgs::PointCloud2 const> sensor_msgs::PointCloud2ConstPtr
std::queue<sensor_msgs::PointCloud2ConstPtr> cornerLessSharpBuf;
std::queue<sensor_msgs::PointCloud2ConstPtr> surfFlatBuf;
std::queue<sensor_msgs::PointCloud2ConstPtr> surfLessFlatBuf;
std::queue<sensor_msgs::PointCloud2ConstPtr> fullPointsBuf;
std::mutex mBuf;

// undistort lidar point,运动畸变的补偿,实现过程?
void TransformToStart(PointType const *const pi, PointType *const po)
{
    //interpolation ratio
    double s;
    //因为DISTORTION一直是0，所以s一直是1；之所以一直是0，是因为kitti已经将畸变去除掉了
    if (DISTORTION)
        s = (pi->intensity - int(pi->intensity)) / SCAN_PERIOD; //每一帧点云占据整个扫描周期的时间比例
    else
        s = 1.0;
    // 下面可以理解成，将当前祯的点云投影到这一祯的起始时刻
    // 这里相当于一个匀速模型假设，将位姿分解成为旋转和平移
    //std::cout<<"(pi->intensity - int(pi->intensity)) / SCAN_PERIOD="<<(pi->intensity - int(pi->intensity)) / SCAN_PERIOD<<std::endl;//demo，s是变化的
                  
    //std::cout<<"s的数值="<<s<<std::endl;//demo 恒为1
    
    Eigen::Quaterniond q_point_last = Eigen::Quaterniond::Identity().slerp(s, q_last_curr);
    //将上一祯的结果用在本祯来去畸变，这是一个匀速模型的假设，但是在实际车上，这个模型较为实用。但是手持有设备。就会显得不是很实用
    //s为0时，结果为Identity(),为1时;
    Eigen::Vector3d t_point_last = s * t_last_curr;

    Eigen::Vector3d point(pi->x, pi->y, pi->z);
    Eigen::Vector3d un_point = q_point_last * point + t_point_last;

    po->x = un_point.x();//undistortion。所以un表示不失真的
    po->y = un_point.y();
    po->z = un_point.z();
    po->intensity = pi->intensity;
}
 
// transform all lidar points to the start of the next frame
// 此函数是将当前祯的点云转移到，上一祯结束的位置，也就是下一祯开始的位置
void TransformToEnd(PointType const *const pi, PointType *const po)
{
    // undistort point first,首先做运动畸变的补偿
    pcl::PointXYZI un_point_tmp;//定义一个点云点
    TransformToStart(pi, &un_point_tmp);//转到祯起始时刻坐标系下的点

    Eigen::Vector3d un_point(un_point_tmp.x, un_point_tmp.y, un_point_tmp.z);
    Eigen::Vector3d point_end = q_last_curr.inverse() * (un_point - t_last_curr);
    //**重要q_last_curr，t_last_curr 上一祯起始时刻转到上一祯结束时刻的旋转和平移，un_point相当于是旋转的起始点位置
    //通过代码想反推公式是本末倒置的

    po->x = point_end.x();
    po->y = point_end.y();
    po->z = point_end.z();

    //Remove distortion time info这里是移除匀速带来的运动畸变;前面IMU去除的是变加速带来的运动畸变
    po->intensity = int(pi->intensity);
}

void laserCloudSharpHandler(const sensor_msgs::PointCloud2ConstPtr &cornerPointsSharp2)
{
    mBuf.lock();
    cornerSharpBuf.push(cornerPointsSharp2);
    mBuf.unlock();
}

void laserCloudLessSharpHandler(const sensor_msgs::PointCloud2ConstPtr &cornerPointsLessSharp2)
{
    mBuf.lock();
    cornerLessSharpBuf.push(cornerPointsLessSharp2);
    mBuf.unlock();
}

void laserCloudFlatHandler(const sensor_msgs::PointCloud2ConstPtr &surfPointsFlat2)
{
    mBuf.lock();
    surfFlatBuf.push(surfPointsFlat2);
    mBuf.unlock();
}

void laserCloudLessFlatHandler(const sensor_msgs::PointCloud2ConstPtr &surfPointsLessFlat2)
{
    mBuf.lock();
    surfLessFlatBuf.push(surfPointsLessFlat2);
    mBuf.unlock();
}

//receive all point cloud
void laserCloudFullResHandler(const sensor_msgs::PointCloud2ConstPtr &laserCloudFullRes2)
{
    mBuf.lock();
    fullPointsBuf.push(laserCloudFullRes2);
    mBuf.unlock();
}

int main(int argc, char **argv)
{
    ros::init(argc, argv, "laserOdometry");
    ros::NodeHandle nh;

    nh.param<int>("mapping_skip_frame", skipFrameNum, 2);

    printf("Mapping %d Hz \n", 10 / skipFrameNum);

    ros::Subscriber subCornerPointsSharp = nh.subscribe<sensor_msgs::PointCloud2>("/laser_cloud_sharp", 100, laserCloudSharpHandler);

    ros::Subscriber subCornerPointsLessSharp = nh.subscribe<sensor_msgs::PointCloud2>("/laser_cloud_less_sharp", 100, laserCloudLessSharpHandler);

    ros::Subscriber subSurfPointsFlat = nh.subscribe<sensor_msgs::PointCloud2>("/laser_cloud_flat", 100, laserCloudFlatHandler);

    ros::Subscriber subSurfPointsLessFlat = nh.subscribe<sensor_msgs::PointCloud2>("/laser_cloud_less_flat", 100, laserCloudLessFlatHandler);

    ros::Subscriber subLaserCloudFullRes = nh.subscribe<sensor_msgs::PointCloud2>("/velodyne_cloud_2", 100, laserCloudFullResHandler);

    ros::Publisher pubLaserCloudCornerLast = nh.advertise<sensor_msgs::PointCloud2>("/laser_cloud_corner_last", 100);

    ros::Publisher pubLaserCloudSurfLast = nh.advertise<sensor_msgs::PointCloud2>("/laser_cloud_surf_last", 100);

    ros::Publisher pubLaserCloudFullRes = nh.advertise<sensor_msgs::PointCloud2>("/velodyne_cloud_3", 100);

    ros::Publisher pubLaserOdometry = nh.advertise<nav_msgs::Odometry>("/laser_odom_to_init", 100);

    ros::Publisher pubLaserPath = nh.advertise<nav_msgs::Path>("/laser_odom_path", 100);

    nav_msgs::Path laserPath;

    int frameCount = 0;
    ros::Rate rate(100);

    while (ros::ok())
    {
        ros::spinOnce();
            //确保订阅的5个存放地址的buf全不为空
        if (!cornerSharpBuf.empty() && !cornerLessSharpBuf.empty() &&
            !surfFlatBuf.empty() && !surfLessFlatBuf.empty() &&!fullPointsBuf.empty())
        {   //分别求出队列的第一个指针对应的地址时间
            timeCornerPointsSharp = cornerSharpBuf.front()->header.stamp.toSec();
            timeCornerPointsLessSharp = cornerLessSharpBuf.front()->header.stamp.toSec();
            timeSurfPointsFlat = surfFlatBuf.front()->header.stamp.toSec();
            timeSurfPointsLessFlat = surfLessFlatBuf.front()->header.stamp.toSec();
            timeLaserCloudFullRes = fullPointsBuf.front()->header.stamp.toSec();//先是成员函数运行,然后才是箭头运算符
            //如果时间戳有一个不等,就会报错中止
            if (timeCornerPointsSharp != timeLaserCloudFullRes ||timeCornerPointsLessSharp != timeLaserCloudFullRes ||
                timeSurfPointsFlat    != timeLaserCloudFullRes ||timeSurfPointsLessFlat    != timeLaserCloudFullRes)
            {
                printf("unsync messeage!");//未时间同步消息
                ROS_BREAK();//用于中断程序并输出本句所在文件/行数
            }

            mBuf.lock();
            cornerPointsSharp->clear();
            pcl::fromROSMsg(*cornerSharpBuf.front(), *cornerPointsSharp);//from ROSMsg to PCL data type
            cornerSharpBuf.pop();//上面是一帧一帧的传,传完一个pop出去一个,但是为什么要用适配器呢?是因为想要使用他的FIFO原则么?

            cornerPointsLessSharp->clear();
            //找一个空杯子,但用之前还是再清洗一下,然后再去装载东西
            pcl::fromROSMsg(*cornerLessSharpBuf.front(), *cornerPointsLessSharp);
            //左侧queue类型的buf是从laserCloudLessSharpHandler函数里导入来的;注意,只是传了ROSMsg的第一个数据到雷达PCL点云格式的
            cornerLessSharpBuf.pop();//将先进入的第一个元素,先第一个删除

            surfPointsFlat->clear();
            pcl::fromROSMsg(*surfFlatBuf.front(), *surfPointsFlat);//左侧的*不是指针符号，而是解引用
            surfFlatBuf.pop();

            surfPointsLessFlat->clear();
            pcl::fromROSMsg(*surfLessFlatBuf.front(), *surfPointsLessFlat);
            surfLessFlatBuf.pop();

            laserCloudFullRes->clear();//地址清零
            pcl::fromROSMsg(*fullPointsBuf.front(), *laserCloudFullRes);
            fullPointsBuf.pop();//弹出队顶元素,但不返回任何值
            mBuf.unlock();

            TicToc t_whole;
            // initializing一个什么也不干的初始化,在这里也意味着第一帧跳过,因为第一帧也没法建立kdtree,需要第二帧后才能建立
            if (!systemInited)
            {
                systemInited = true;
                std::cout << "Initialization finished \n";
            }
            //在这里切割分析代码后，会显得比较清晰合理一些，思考一个很重要的问题，就是前一祯的点云到底有没有投影到开始的位置？
            else
            {
                //从第2帧开始取出特征点
                int cornerPointsSharpNum = cornerPointsSharp->points.size();
                std::cout<<"输出cornerPointsSharp点的数量="<<cornerPointsSharpNum<<std::endl;//192
                int surfPointsFlatNum = surfPointsFlat->points.size();
                std::cout<<"输出surfPointsFlatNum点的数量="<<surfPointsFlatNum<<std::endl;//370

                TicToc t_opt;
                //避免代码的细节，可以先明白他的含义，搁置。避免一开始就陷入到其中的泥潭之中，很难自拔出来

                //进行两次迭代求解,匹配对建立的过程进行两次,可能前后两次不一致的情形,但是以第一次为准
                //但是实际代码里面只是出现了一次，没有进行两次，还是理解的不对呢，答：进行了两次
                //但是上面执行的两次，怎么会知道到底是选择了哪一次呢？
                for (size_t opti_counter = 0; opti_counter < 2; ++opti_counter)
                {
                    corner_correspondence = 0;
                    plane_correspondence = 0;
                    std::cout<<"opti_counter数值="<<opti_counter<<std::endl;
                    //ceres::LossFunction *loss_function = NULL;

                    //下面定义了ceres的核函数,当残差大于0.1时就降低它的权重,否则就按正常权重处理
                    ceres::LossFunction* loss_function = new ceres::HuberLoss(0.1);
                    
                    ceres::LocalParameterization *q_parameterization = new ceres::EigenQuaternionParameterization();
                    ceres::Problem::Options problem_options;

                    ceres::Problem problem(problem_options);

                    //下面是告诉函数的优化器,里面的参数含义.待优化的变量是帧间位姿,平移和旋转.这里旋转使用四元数表示
                    //第一个参数表示数组的首指针,第三个是DIY的加法
                    /********************************************************/
                    /*************参数块就是自己想要求得的参数是什么**************/
                    /********************************************************/
                    /*因为是四元数旋转,不满足一般意义上的加法关系,所以必须要加下行代码*/
                    problem.AddParameterBlock(para_q, 4, q_parameterization);
                    problem.AddParameterBlock(para_t, 3);
                    //第二个参数表示数组的位数,旋转不需要额外定义的,所以第三个没有对应参数
                    pcl::PointXYZI pointSel;//将这个点的地址构建出来以后，就可以对变量的地址进行处理，提高处理效率
                    std::vector<int> pointSearchInd;
                    std::vector<float> pointSearchSqDis;

                    TicToc t_data;
                 //1. 寻找角点的约束对 find correspondence for corner features
                     int inum=0;
                     //std::cout<<"输出cornerPointsSharpNum="<<cornerPointsSharpNum<<std::endl;//demo value=const 192
                    for (int i = 0; i < cornerPointsSharpNum; ++i)                    
                    {
                        //**重要；随着当前祯的点云的增加，逐渐投影到前一祯时间的结尾，投影方式按照上一祯的R(k)和t(k)进行投影，
                        //**当得出当前祯整体旋转位姿的R(k+1),t(k+1)后，再和前一祯相乘就能得出当前祯(k+1时刻)的旋转位姿。
                        TransformToStart(&(cornerPointsSharp->points[i]), &pointSel);//调用时取地址符，那么原函数一定是指针
                        //上面不仅仅是运动畸变的去除，还是将【当前祯】的所有点投影到该祯起始start的位置
                        //答：kitti数据集里已经是去除畸变的
                        //对当前祯运动畸变的补偿,pointSel是起始时刻的角点,是任意的么？
                        //答：不是。而是遍历cornerPointsSharpNum中的每一个点去畸变
                        //并且将该周期中的每一个点投影到该周期的初始时刻
                        kdtreeCornerLast->nearestKSearch(pointSel, 1, pointSearchInd, pointSearchSqDis);
                        //在当前祯去除畸变poinSel点，在上一祯的Corner特征点中寻找最近点
                        //++inum;//demo
                        //std::cout<<"输出的数据pointSearchInd[0]="<<pointSearchInd[0]<<";inum数值="<<inum<<std::endl;//demo
                                  //输出的数据pointSearchInd[0]=1716;inum数值=192
                        //搜索结果: pointSearchInd是索引; pointSearchSqDis是近邻对应距离的平方(以25作为阈值);
                        //搜索范围:kdtreeCornerLast->setInputCloud(laserCloudCornerLast);
                        //因为去掉了第一帧,所以搜索在前,更新input在后
                        //pointSel：查询点；1：邻近个数；pointSearchInd：储存搜索到的近邻点的索引；
                        //pointSearchSqDis：储存查询点与对应近邻点中心距离平方
                        int closestPointInd = -1, minPointInd2 = -1;
                        //遍历当前帧所有的corner点,然后分别找出他们各个点在上一帧所有corner点里的最近邻,
                        //最终会得到1个最近邻的值(分成两部分的各自最小值),这样的逻辑合理么?为啥这个最小值就是最优的结果呢?
                        //是否和点云的计算效率有关,基本筛选特征的原因都是一些相对"简易"的加减乘除,用不得其他复杂算法有关呢?                        
                        if (pointSearchSqDis[0] < DISTANCE_SQ_THRESHOLD)//若点在距离5m*5m范围内则进行计算,否则不计算
                        {
                            closestPointInd = pointSearchInd[0]; //最近邻只检索周围一个点，所以只有pointSearchInd[0]，否则还会有pointSearchInd[1]                           
                            int closestPointScanID = int(laserCloudCornerLast->points[closestPointInd].intensity);
                            // std::cout<<"pointSearchInd[0]"<<pointSearchInd[0]<<";closestPointScanID"<<closestPointScanID<<std::endl;
                            double minPointSqDis2 = DISTANCE_SQ_THRESHOLD;//DISTANCE_SQ_THRESHOLD=25
                            // search in the direction of increasing scan line
                            // ***重要***在当前祯的corner点中找到上一祯的最近点，同时解析出这一激光点的线号、点序号，然后在激光点所在线的上下2根范围内进行寻找距离最近匹配点
                            // 下面只在最近邻下标增长方向进行搜索，而后还会进行下标减小方向的搜索
                            // ***重要***下面的代码含义是，将在j所在线号的two consecutive scans中找到与pointSel点最近的点，作为线点的第二个点（第一个点是k近邻找到的）
                            //这里面还要求，j、l线号上寻找的两个激光点，不应该在同一根线上
                            for (int j = closestPointInd + 1; j < (int)laserCloudCornerLast->points.size(); ++j)
                            {
                                //std::cout<<"输出laserCloudCornerLast->points.size()的数量="<<(laserCloudCornerLast->points.size())<<std::endl;
                                // if in the same scan line, continue
                                if (int(laserCloudCornerLast->points[j].intensity) <= closestPointScanID)
                                //上面这段也是和前一个文件scanRegistration.cpp相呼应的;因在scanRegistration.cpp文件中
                                //提取特征点时,是先按照线号从小到大进行提取的,所以在laserOdometry.cpp里面提取线号的时候,也是这样进行的
                                //当激光点序号从大到小进行提取时,点序号增加时，激光线号是增加才对，若激光线号不增反减时,是不合理现象,要进行continue跳过这个不合理的点
                                    continue;
                                // if not in nearby scans, end the loop
                                //超过3根线以外的时候,就break掉这个程序,但是为啥是3,不是4,2等等是很值得探讨的问题?

                                if (int(laserCloudCornerLast->points[j].intensity) > (closestPointScanID + NEARBY_SCAN))
                                    break;
                                double pointSqDis = (laserCloudCornerLast->points[j].x - pointSel.x) *
                                                    (laserCloudCornerLast->points[j].x - pointSel.x) +
                                                        (laserCloudCornerLast->points[j].y - pointSel.y) *
                                                        (laserCloudCornerLast->points[j].y - pointSel.y) +
                                                    (laserCloudCornerLast->points[j].z - pointSel.z) *
                                                    (laserCloudCornerLast->points[j].z - pointSel.z);

                                if (pointSqDis < minPointSqDis2)
                                {
                                    //find nearer point,afer this process ,only one resultant will be left
                                    minPointSqDis2 = pointSqDis;//不断地更新最近距离，将最近距离进行缩窄，确保最后得出的最近点只有一个
                                    minPointInd2 = j;
                                }
                            }
                            //下面只是在最近邻下标减小方向进行搜索，而后还会进行下标增长方向的搜索
                            // search in the direction of decreasing scan line

                            //std::cout<<"minPointSqDis2数值="<<minPointSqDis2<<std::endl;//demo
                            //如果关于上半段向着高纬度下标搜索成功的话，那么就更新了minPointSqDis2数值，否则它的数值还是25！！！
                            //所以虽然是分成了上下两块，但是最终获取的最值将会是一个数值！！！
                            for (int j = closestPointInd - 1; j >= 0; --j)
                            {
                                // if in the same scan line, continue
                                if (int(laserCloudCornerLast->points[j].intensity) >= closestPointScanID)
                                    continue;//continue跳过此次循环的其余部分，scanID号码不能处于同一根线上，
                                //只是在scanID下面2个下标范围内，是可以进行选取对应雷达上面数据的
                                // if not in nearby scans, end the loop
                                if (int(laserCloudCornerLast->points[j].intensity) < (closestPointScanID - NEARBY_SCAN))
                                    break;//break将会跳过整个for循环，并执行for循环以后的内容

                                    double pointSqDis = (laserCloudCornerLast->points[j].x - pointSel.x) *
                                                        (laserCloudCornerLast->points[j].x - pointSel.x) +
                                                            (laserCloudCornerLast->points[j].y - pointSel.y) *
                                                            (laserCloudCornerLast->points[j].y - pointSel.y) +
                                                        (laserCloudCornerLast->points[j].z - pointSel.z) *
                                                        (laserCloudCornerLast->points[j].z - pointSel.z);

                                if (pointSqDis < minPointSqDis2)
                                {
                                    // find nearer point
                                    minPointSqDis2 = pointSqDis;
                                    minPointInd2 = j;
                                }
                            }
                        }
                            // both closestPointInd and minPointInd2 is valid.如果不是,minPointInd2将会是-1
                        if (minPointInd2 >= 0) 
                        {   
                            //除非有其他序号给他赋值，否则默认minPointInd2是-1
                            //minPointInd2>=0意味着,当每一个当前帧的corner点,经过两组针对与上一帧的corner点进行匹配筛选
                            //具体经过两个for循环以后,当前的corner点云集里面的激光点,符合要求的都会进到这个if()判断里面来
                            //但是每当有当前帧新来的corner点时,minPointInd2都会继续再恢复到原始的值,minPointInd2 = -1;
                            //取出当前点和上一帧的两个点(形成的线约束)
                            Eigen::Vector3d curr_point(cornerPointsSharp->points[i].x,
                                                       cornerPointsSharp->points[i].y,
                                                       cornerPointsSharp->points[i].z);
                            Eigen::Vector3d last_point_a(laserCloudCornerLast->points[closestPointInd].x,
                                                         laserCloudCornerLast->points[closestPointInd].y,
                                                         laserCloudCornerLast->points[closestPointInd].z);
                            Eigen::Vector3d last_point_b(laserCloudCornerLast->points[minPointInd2].x,
                                                         laserCloudCornerLast->points[minPointInd2].y,
                                                         laserCloudCornerLast->points[minPointInd2].z);
                            double s;
                            if (DISTORTION)
                                s = (cornerPointsSharp->points[i].intensity - int(cornerPointsSharp->points[i].intensity)) / SCAN_PERIOD;
                                //s表达每个点占据整个扫描周期里面的时间比例
                            else
                                s = 1.0;
                            // 先去理解上面的curr_point last_point_a last_point_b 三者的来历至关重要，首先要还原问题本身是什么
                            // 答：上面的3个量，分别是当前祯的当前点和上一祯的两个线点  
                            ceres::CostFunction* cost_function = LidarEdgeFactor::Create(curr_point, last_point_a, last_point_b, s);
                            //上面是通用模板库？，构建求解非线性最小二乘的方程，上面函数是为了构建线点的约束和优化
                            std::cout<<"Before Output para_q="<<para_q<<std::endl;
                            problem.AddResidualBlock(cost_function, loss_function, para_q, para_t);
                            std::cout<<"After  Output para_q="<<para_q<<std::endl;
                            //输入cost_function loss_function 得出para_q-和para_t-的计算结果
                            corner_correspondence++;//用来起到计数的作用
                        }
                    }

                 //2.寻找面点的约束对 find correspondence for plane features，在已有面点数据里寻找匹配对的面点约束
                    for (int i = 0; i < surfPointsFlatNum; ++i)
                    {
                        //下面的函数是进行运动补偿，同时将当前祯的点投影到这一祯起始祯的位置
                        TransformToStart(&(surfPointsFlat->points[i]), &pointSel);//调用函数，原函数一定是指针
                        //void TransformToStart(PointType const *const pi, PointType *const po)//形参
                        //在上一帧所有角点构成的kdtree里面寻找当前帧最近的一个点,nearestKSearch是kdtreeSurfLast下面的子函数
                        //在pointSel点集里面找到1个点
                        kdtreeSurfLast->nearestKSearch(pointSel, 1, pointSearchInd, pointSearchSqDis);

                        int closestPointInd = -1, minPointInd2 = -1, minPointInd3 = -1;
                        //只有小于给定的门限值才被认为是有效的
                        if (pointSearchSqDis[0] < DISTANCE_SQ_THRESHOLD)
                        {
                            closestPointInd = pointSearchInd[0];

                            // get closest point's scan ID
                            int closestPointScanID = int(laserCloudSurfLast->points[closestPointInd].intensity);
                            double minPointSqDis2 = DISTANCE_SQ_THRESHOLD, minPointSqDis3 = DISTANCE_SQ_THRESHOLD;

                            // search in the direction of increasing scan line
                            for (int j = closestPointInd + 1; j < (int)laserCloudSurfLast->points.size(); ++j)
                            {
                                // if not in nearby scans, end the loop
                                if (int(laserCloudSurfLast->points[j].intensity) > (closestPointScanID + NEARBY_SCAN))
                                    break;

                                double pointSqDis = (laserCloudSurfLast->points[j].x - pointSel.x) *
                                                        (laserCloudSurfLast->points[j].x - pointSel.x) +
                                                    (laserCloudSurfLast->points[j].y - pointSel.y) *
                                                        (laserCloudSurfLast->points[j].y - pointSel.y) +
                                                    (laserCloudSurfLast->points[j].z - pointSel.z) *
                                                        (laserCloudSurfLast->points[j].z - pointSel.z);

                                // if in the same or lower scan line
                                if (int(laserCloudSurfLast->points[j].intensity) <= closestPointScanID && pointSqDis < minPointSqDis2)
                                {
                                    minPointSqDis2 = pointSqDis;
                                    minPointInd2 = j;
                                }
                                // if in the higher scan line
                                else if (int(laserCloudSurfLast->points[j].intensity) > closestPointScanID && pointSqDis < minPointSqDis3)
                                {
                                    minPointSqDis3 = pointSqDis;
                                    minPointInd3 = j;
                                }
                            }

                            // search in the direction of decreasing scan line
                            for (int j = closestPointInd - 1; j >= 0; --j)
                            {
                                // if not in nearby scans, end the loop
                                if (int(laserCloudSurfLast->points[j].intensity) < (closestPointScanID - NEARBY_SCAN))
                                    break;

                                double pointSqDis = (laserCloudSurfLast->points[j].x - pointSel.x) *
                                                        (laserCloudSurfLast->points[j].x - pointSel.x) +
                                                    (laserCloudSurfLast->points[j].y - pointSel.y) *
                                                        (laserCloudSurfLast->points[j].y - pointSel.y) +
                                                    (laserCloudSurfLast->points[j].z - pointSel.z) *
                                                        (laserCloudSurfLast->points[j].z - pointSel.z);

                                // if in the same or higher scan line
                                if (int(laserCloudSurfLast->points[j].intensity) >= closestPointScanID && pointSqDis < minPointSqDis2)
                                {
                                    minPointSqDis2 = pointSqDis;
                                    minPointInd2 = j;
                                }
                                else if (int(laserCloudSurfLast->points[j].intensity) < closestPointScanID && pointSqDis < minPointSqDis3)
                                {
                                    // find nearer point
                                    minPointSqDis3 = pointSqDis;
                                    minPointInd3 = j;
                                }
                            }

                            if (minPointInd2 >= 0 && minPointInd3 >= 0)
                            {

                                Eigen::Vector3d curr_point( surfPointsFlat->points[i].x,
                                                            surfPointsFlat->points[i].y,
                                                            surfPointsFlat->points[i].z);
                                Eigen::Vector3d last_point_a(laserCloudSurfLast->points[closestPointInd].x,
                                                                laserCloudSurfLast->points[closestPointInd].y,
                                                                laserCloudSurfLast->points[closestPointInd].z);
                                Eigen::Vector3d last_point_b(laserCloudSurfLast->points[minPointInd2].x,
                                                                laserCloudSurfLast->points[minPointInd2].y,
                                                                laserCloudSurfLast->points[minPointInd2].z);
                                Eigen::Vector3d last_point_c(laserCloudSurfLast->points[minPointInd3].x,
                                                                laserCloudSurfLast->points[minPointInd3].y,
                                                                laserCloudSurfLast->points[minPointInd3].z);

                                double s;
                                if (DISTORTION)//失真
                                    s = (surfPointsFlat->points[i].intensity - int(surfPointsFlat->points[i].intensity)) / SCAN_PERIOD;
                                else
                                    s = 1.0;
                                ceres::CostFunction *cost_function = LidarPlaneFactor::Create(curr_point, last_point_a, last_point_b, last_point_c, s);
                                problem.AddResidualBlock(cost_function, loss_function, para_q, para_t);
                                plane_correspondence++;
                            }
                        }
                    }

                    printf("corner_correspondance %d, plane_correspondence %d \n", corner_correspondence, plane_correspondence);
                    printf("data association time %f ms \n", t_data.toc());//数据关联时间3ms

                    if ((corner_correspondence + plane_correspondence) < 10)//一帧点云提取明显的匹配对小于10是有问题的
                    {
                        printf("less correspondence! *************************************************\n");
                    }
                 //调用ceres求解器
                    TicToc t_solver;
                    ceres::Solver::Options options;
                    options.linear_solver_type = ceres::DENSE_QR;//
                    options.max_num_iterations = 4;//因为lidar里程计前端要求求解的速率,所以这里的最大单次求解次数设置为4
                    std::cout<<"options.minimizer_progress_to_stdout = true_begin;"<<std::endl;
                    options.minimizer_progress_to_stdout = true;
                    std::cout<<"options.minimizer_progress_to_stdout = true_end"<<std::endl;
                    ceres::Solver::Summary summary;//求解器的一些选项
                    ceres::Solve(options, &problem, &summary);
                    std::cout<<"summary"<<&summary<<std::endl;
                    printf("solver time %f ms \n", t_solver.toc());//求解器求解时间3ms
                }
                printf("optimization twice time %f \n", t_opt.toc());//前端里程计优化求解的时间
                //**重要，更新R和t，旋转和平移.得出的是，当前祯相对于上一祯的位姿变化
                t_w_curr = t_w_curr + q_w_curr * t_last_curr;//t_w_curr为原矩阵t,t_last_curr为平移转换矩阵
                q_w_curr = q_w_curr * q_last_curr;           //q_w_curr为原矩阵R,q_last_curr为转移矩阵
            }   //最顶层的if else开始的

            TicToc t_pub;

            // publish odometry
            nav_msgs::Odometry laserOdometry;
            laserOdometry.header.frame_id = "/camera_init";
            laserOdometry.child_frame_id = "/laser_odom";
            laserOdometry.header.stamp = ros::Time().fromSec(timeSurfPointsLessFlat);
            //4元数的旋转位姿态,和t表示的平移.首先将这些位姿转换到ROS的格式,然后发布出去
            laserOdometry.pose.pose.orientation.x = q_w_curr.x();
            laserOdometry.pose.pose.orientation.y = q_w_curr.y();
            laserOdometry.pose.pose.orientation.z = q_w_curr.z();
            laserOdometry.pose.pose.orientation.w = q_w_curr.w();
            laserOdometry.pose.pose.position.x = t_w_curr.x();
            laserOdometry.pose.pose.position.y = t_w_curr.y();
            laserOdometry.pose.pose.position.z = t_w_curr.z();
            pubLaserOdometry.publish(laserOdometry);
//下面的7行代码块,主要是为了给rviz输入的,因为其他节点是不会接受轨迹的输入
            geometry_msgs::PoseStamped laserPose;
            laserPose.header = laserOdometry.header;
            laserPose.pose = laserOdometry.pose.pose;
            laserPath.header.stamp = laserOdometry.header.stamp;
            laserPath.poses.push_back(laserPose);
            laserPath.header.frame_id = "/camera_init";
            pubLaserPath.publish(laserPath);

    //transform corner features and plane features to the scan end point
    //下面这段无效的本质是，kitti已经处理过了运动畸变，因此没有此功能，如果是其他未去除畸变的数据，可以改成if(1)来启用，
    //**将当前祯点云转到当前祯结尾的时刻（即下一祯开始时刻）**//
            if(0)
            {
                int cornerPointsLessSharpNum = cornerPointsLessSharp->points.size();
                for (int i = 0; i < cornerPointsLessSharpNum; i++)
                {
                    TransformToEnd(&cornerPointsLessSharp->points[i], &cornerPointsLessSharp->points[i]);
                }

                int surfPointsLessFlatNum = surfPointsLessFlat->points.size();
                for (int i = 0; i < surfPointsLessFlatNum; i++)
                {
                    TransformToEnd(&surfPointsLessFlat->points[i], &surfPointsLessFlat->points[i]);
                }

                int laserCloudFullResNum = laserCloudFullRes->points.size();
                for (int i = 0; i < laserCloudFullResNum; i++)
                {
                    TransformToEnd(&laserCloudFullRes->points[i], &laserCloudFullRes->points[i]);
                }
            }
            //下面3行,就是基本地角点交换二者内容,将点云内容交给Last，方便下一轮和下一祯进行匹配，为什么要交换而不是直接赋值？
            pcl::PointCloud<PointType>::Ptr laserCloudTemp = cornerPointsLessSharp;
                                        cornerPointsLessSharp = laserCloudCornerLast;
                                        laserCloudCornerLast = laserCloudTemp;
            
            //下面3行,就是基础的面点交换二者内容。surfPointsLessFlat里面包含着surfPointsFlat
                                        laserCloudTemp = surfPointsLessFlat;
                                        surfPointsLessFlat = laserCloudSurfLast;
                                        laserCloudSurfLast = laserCloudTemp;

            laserCloudCornerLastNum = laserCloudCornerLast->points.size();
            laserCloudSurfLastNum = laserCloudSurfLast->points.size();

            // std::cout << "the size of corner last is " << laserCloudCornerLastNum << ", and the size of surf last is " << laserCloudSurfLastNum << '\n';
            //经过了一次处理以后,然后将当前帧送到kdtree用于下一帧的匹配对的寻找
            //if(!systemInited) 语句非常巧妙地将第一祯数据，放弃此if(!systemInited)对应else语句处理，直接给到了下面的setInputCloud语句
            //所以带Last关键词的标识符，永远会超前一祯周期
            kdtreeCornerLast->setInputCloud(laserCloudCornerLast);
            kdtreeSurfLast->setInputCloud(laserCloudSurfLast);

            if (frameCount % skipFrameNum == 0)//求余数,就是经降频发送出去
            {
                frameCount = 0;
                //给后端的点云要尽可能的多,这样才能够保持足够的精度
                sensor_msgs::PointCloud2 laserCloudCornerLast2;
                pcl::toROSMsg(*laserCloudCornerLast, laserCloudCornerLast2);
                laserCloudCornerLast2.header.stamp = ros::Time().fromSec(timeSurfPointsLessFlat);
                laserCloudCornerLast2.header.frame_id = "/camera";
                pubLaserCloudCornerLast.publish(laserCloudCornerLast2);

                sensor_msgs::PointCloud2 laserCloudSurfLast2;
                pcl::toROSMsg(*laserCloudSurfLast, laserCloudSurfLast2);
                laserCloudSurfLast2.header.stamp = ros::Time().fromSec(timeSurfPointsLessFlat);
                laserCloudSurfLast2.header.frame_id = "/camera";
                pubLaserCloudSurfLast.publish(laserCloudSurfLast2);

                sensor_msgs::PointCloud2 laserCloudFullRes3;
                pcl::toROSMsg(*laserCloudFullRes, laserCloudFullRes3);//所有的点:laserCloudFullRes
                laserCloudFullRes3.header.stamp = ros::Time().fromSec(timeSurfPointsLessFlat);
                laserCloudFullRes3.header.frame_id = "/camera";
                pubLaserCloudFullRes.publish(laserCloudFullRes3);
            }
            printf("publication time %f ms \n", t_pub.toc());
            printf("whole laserOdometry time %f ms \n \n", t_whole.toc());//16.45ms
            if(t_whole.toc() > 100)
                ROS_WARN("odometry process over 100ms");

            frameCount++;
        }
        rate.sleep();
    }
    return 0;
}