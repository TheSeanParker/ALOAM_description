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
// 整个算法都是建立在，void fun(),只是在传参时候比较复杂

#include <cmath>
#include <vector>
#include <string>
#include "aloam_velodyne/common.h"
#include "aloam_velodyne/tic_toc.h"
#include <nav_msgs/Odometry.h>
#include <opencv/cv.h>
#include <pcl_conversions/pcl_conversions.h>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/kdtree/kdtree_flann.h>
#include <ros/ros.h>
//一个实用的头文件，它引用了 ROS 系统中大部分常用的头文件
#include <sensor_msgs/Imu.h>
#include <sensor_msgs/PointCloud2.h>
#include <tf/transform_datatypes.h>
#include <tf/transform_broadcaster.h>
#include <iostream>
using namespace std;
using std::atan2;
//用于计算y / x的反正切的主值
using std::cos;
using std::sin;

const double scanPeriod = 0.1;

const int systemDelay = 0; 
int systemInitCount = 0;
bool systemInited = false;
int N_SCANS = 0;

float cloudCurvature[400000];//曲率
int cloudOriginalSortInd[400000];
int cloudNeighborPicked[400000];
int cloudLabel[400000];

bool comp (int i,int j) 
{return (cloudCurvature[i]<cloudCurvature[j]); }//从小到大排序
//{ return (i<j);}上下应该是同一个意思

ros::Publisher pubLaserCloud;
ros::Publisher pubCornerPointsSharp;
ros::Publisher pubCornerPointsLessSharp;
ros::Publisher pubSurfPointsFlat;
ros::Publisher pubSurfPointsLessFlat;
ros::Publisher pubRemovePoints;
std::vector<ros::Publisher> pubEachScan;

bool PUB_EACH_LINE = false;

double MINIMUM_RANGE = 0.1; 

template <typename PointT>
void removeClosedPointCloud (const pcl::PointCloud<PointT> &cloud_in,pcl::PointCloud<PointT> &cloud_out, float thres)
{
    if (&cloud_in != &cloud_out)
    {
        cloud_out.header = cloud_in.header;
        cloud_out.points.resize(cloud_in.points.size());
        //入参和出参进行一致性判断,这样来达到节约资源开销的问题?
    }

    size_t j = 0;//size_t类型是为了增加可移植性定义size_t可能不一样。32位系统中size_t是4字节的.64位系统中，size_t是8字节的

    for (size_t i = 0; i < cloud_in.points.size(); ++i)
    {
        if (cloud_in.points[i].x * cloud_in.points[i].x + cloud_in.points[i].y * cloud_in.points[i].y + cloud_in.points[i].z * cloud_in.points[i].z < thres * thres)
            continue;
        cloud_out.points[j] = cloud_in.points[i];
        j++;//j的大小是用在存储非异常点的数量
    }
    if (j != cloud_in.points.size())
    {
        cloud_out.points.resize(j);
    }

    cloud_out.height = 1;
    cloud_out.width = static_cast<uint32_t>(j);//宽度是它的所有数目std::uint32_t
    cloud_out.is_dense = true;
    //经过上面的处理后,就将原来有序的点变成了无序的点
}
//下面的回调函数,是点云每100ms来一次,就会处理一次,所以这个函数的运行时间尽量控制在100ms以内是合理的,可能会丢帧,里程计和mapping的精度就会下降
void laserCloudHandler(const sensor_msgs::PointCloud2ConstPtr &laserCloudMsg)//ConstPtr常量指针，入参是一个指针
{
    if (!systemInited)//下面是系统进行初始化对应的时间,如果没有达到对应要求便需要反复进行初始化
    { 
        systemInitCount++;
        if (systemInitCount >= systemDelay)
        {
            systemInited = true;
        }
        else
            return;
    }

    TicToc t_whole;
    TicToc t_prepare;
    std::vector<int> scanStartInd(N_SCANS, 0);//N_SCANS个数，初始值全为0
    std::vector<int>   scanEndInd(N_SCANS, 0);

    pcl::PointCloud<pcl::PointXYZ> laserCloudIn;//类似std::vector<int> scanStartInd(N_SCANS, 0);
    pcl::fromROSMsg(*laserCloudMsg, laserCloudIn);//完成一次函数的转移，从ROS类型转移到PCL库的数据类型
    std::vector<int> indices;

    pcl::removeNaNFromPointCloud(laserCloudIn, laserCloudIn, indices);//Removes points with x, y, or z equal to NaN;接收空间中某点时距离太近或者太远
    removeClosedPointCloud(laserCloudIn, laserCloudIn, MINIMUM_RANGE);//移除不满足条件的点,然后生成1*n的无序点云矩阵

    int cloudSize = laserCloudIn.points.size();

    // typedef pcl::PointXYZ = PointT;
    // std::vector<PointT, Eigen::aligned_allocator<PointT> > pcl::PointCloud< PointT >::pointss =laserCloudIn.points;

    float startOri = -atan2(laserCloudIn.points[0].y, laserCloudIn.points[0].x);
    //取负数是为了将旋转方向改为逆时针,为了和习惯一致
    //计算起始点和结束点的角度，由于激光雷达是顺时针旋转，这里取反就相当于转成了逆时针
    float endOri = -atan2(laserCloudIn.points[cloudSize - 1].y,
                          laserCloudIn.points[cloudSize - 1].x) + 2 * M_PI;//为了表达起始点和结束点正好差2π
    //就是保证雷达的起始点和结束点,因为以上的起始点是雷达什么时候开始转决定的,所以比较的随机,在什么样的位置角度都是有可能的
    if (endOri - startOri > 3 * M_PI)
    {
        endOri -= 2 * M_PI;
    }
    else if (endOri - startOri < M_PI)
    {
        endOri += 2 * M_PI;
    }
    //printf("end Ori %f\n", endOri);

    bool halfPassed = false;
    int count = cloudSize;
    PointType Point_Object;//类声明一个对象
    std::vector<pcl::PointCloud<PointType>> laserCloudScans(N_SCANS);//一个容器里面含类实例化的6个对象,并且都执行构造函数PointCloud()的初始化
    //下面的函数遍历所有的点,逐个算出每个点垂直方向的倾角,然后找到属于哪条激光线,最后再去将这个点归类到这根线的数组里面
    //**下面的atan2()函数就是为了求得时间，
    for (int i = 0; i < cloudSize; i++)
    {
        Point_Object.x = laserCloudIn.points[i].x;
        Point_Object.y = laserCloudIn.points[i].y;
        Point_Object.z = laserCloudIn.points[i].z;
        //计算俯仰角
        float angle = atan(Point_Object.z / sqrt(Point_Object.x * Point_Object.x + Point_Object.y * Point_Object.y)) * 180 / M_PI;
        //atan()函数不能区分180度之内和之外的角度，atan2()函数是可以的
        int scanID = 0;//scanID就是第几根线的意思
        //将这里面的信息是第几根线读取出来，当下雷达驱动里面直接就会得到
        if (N_SCANS == 16)
        {
            scanID = int((angle + 15) / 2 + 0.5);
            //加上0.5是为了去尾部，实现四舍五入，就是求第几根线，大于16肯定不合理的
            if (scanID > (N_SCANS - 1) || scanID < 0)
            {
                count--;
                continue;
            }
        }
        else if (N_SCANS == 32)
        {
            scanID = int((angle + 92.0/3.0) * 3.0 / 4.0);
            if (scanID > (N_SCANS - 1) || scanID < 0)
            {
                count--;
                continue;
            }
        }
        else if (N_SCANS == 64)
        {   
            if (angle >= -8.83)
                scanID = int((2 - angle) * 3.0 + 0.5);
            else
                scanID = N_SCANS / 2 + int((-8.83 - angle) * 2.0 + 0.5);

            // use [0 50]  > 50 remove outlies 
            if (angle > 2 || angle < -24.33 || scanID > 50 || scanID < 0)
            {
                count--;
                continue;
            }
        }
        else
        {
            printf("wrong scan number\n");
            ROS_BREAK();
        }
        //printf("angle %f scanID %d \n", angle, scanID);
        //计算水平角
        float ori = -atan2(Point_Object.y, Point_Object.x);
        if (!halfPassed)//不超过半圈
        { 
            if (ori < startOri - M_PI / 2)
            {
                ori += 2 * M_PI;
            }
            else if (ori > startOri + M_PI * 3 / 2)
            {
                ori -= 2 * M_PI;
            }

            if (ori - startOri > M_PI)
            {
                halfPassed = true;
            }
        }
        else
        {
            ori += 2 * M_PI;
            if (ori < endOri - M_PI * 3 / 2)
            {
                ori += 2 * M_PI;
            }
            else if (ori > endOri + M_PI / 2)
            {
                ori -= 2 * M_PI;
            }
        }
        //角度的计算是为了计算相对起始时刻的时间，占一个旋转周期的百分比，为了计算经历的时间戳
        float relTime = (ori - startOri) / (endOri - startOri);
        //下面的函数是集成，scan的索引，小数部分是相对起始时刻的时间长度
        Point_Object.intensity = scanID + scanPeriod * relTime;//intensity里面是空的内容,放我们自己的东西.小数的部分是放时间
        //现在的雷达会把每一个点云的点都附上时间戳,所以现在可以在雷达的驱动的层面进行完成了
        laserCloudScans[scanID].push_back(Point_Object);//针对实例化后的每一个对象laserCloudScans[scanID]进行调用方法填充数据
        //pcl::PointCloud<PointT>.points
    }  
    pcl::PointCloud<PointType>::Ptr laserCloud(new pcl::PointCloud<PointType>());
    //智能指针laserPoint指向一个动态分配的,未初始化的无名PointType类型的对象
    for (int i = 0; i < N_SCANS; i++)
    { 
        scanStartInd[i] = laserCloud->size() + 5; 
        *laserCloud += laserCloudScans[i];   //这行代码写的很漂亮,是累计迭代的一个集成点的箩筐!                            
        scanEndInd[i]   = laserCloud->size() -6;                     
        //cout<<"第"<<i<<"根线的结束点坐标为："<<scanEndInd[i]<<endl; 
        //因为laserCloudScans[]是定义在laserCloudHandler()函数里面的变量,首先在这个函数内部一直有生命周期
        //首先第一行+5那行,取完的size()是等于0的,关键在于第二行,经过laserCloudScans[]的导入,结尾的点下标自然会递增了
    }
    printf("prepare time %f \n", t_prepare.toc());
    //这里就是进行了前后五个点的数据处理,计算曲率,角点曲率较大,面点曲率较小
    for (int i = 5; i < cloudSize - 5; i++)
    { 
        float diffX = laserCloud->points[i - 5].x + laserCloud->points[i - 4].x + laserCloud->points[i - 3].x + laserCloud->points[i - 2].x + laserCloud->points[i - 1].x - 10 * laserCloud->points[i].x + laserCloud->points[i + 1].x + laserCloud->points[i + 2].x + laserCloud->points[i + 3].x + laserCloud->points[i + 4].x + laserCloud->points[i + 5].x;
        float diffY = laserCloud->points[i - 5].y + laserCloud->points[i - 4].y + laserCloud->points[i - 3].y + laserCloud->points[i - 2].y + laserCloud->points[i - 1].y - 10 * laserCloud->points[i].y + laserCloud->points[i + 1].y + laserCloud->points[i + 2].y + laserCloud->points[i + 3].y + laserCloud->points[i + 4].y + laserCloud->points[i + 5].y;
        float diffZ = laserCloud->points[i - 5].z + laserCloud->points[i - 4].z + laserCloud->points[i - 3].z + laserCloud->points[i - 2].z + laserCloud->points[i - 1].z - 10 * laserCloud->points[i].z + laserCloud->points[i + 1].z + laserCloud->points[i + 2].z + laserCloud->points[i + 3].z + laserCloud->points[i + 4].z + laserCloud->points[i + 5].z;

        cloudCurvature[i] = diffX * diffX + diffY * diffY + diffZ * diffZ;
        cloudOriginalSortInd[i] = i;//编辑一个数组,将激光雷达里面的点按照顺序进行排列
        cloudNeighborPicked[i] = 0;//进到这里判断曲率的都是正常点
        cloudLabel[i] = 0;//后面会对它进行置1或者2,未分类时候是置0
    }

    TicToc t_pts;

    pcl::PointCloud<PointType> cornerPointsSharp;
    pcl::PointCloud<PointType> cornerPointsLessSharp;
    pcl::PointCloud<PointType> surfPointsFlat;//pcl::PointCloud<PointType> 是类名,后接实例化对象名
    pcl::PointCloud<PointType> surfPointsLessFlat;

    float t_q_sort = 0;
    for (int i = 0; i < N_SCANS; i++)
    {//如果某根线束激光雷达的点去掉开头和结尾的10个点,小于6个点,将无法进行下一步的处理
        if( scanEndInd[i] - scanStartInd[i] < 6)
            continue;
        pcl::PointCloud<PointType>::Ptr surfPointsLessFlatScan(new pcl::PointCloud<PointType>);
        //explicit修饰的构造函数只能直接初始化,不能拷贝初始化
        //6次循环，每次都要找2个corner点和4个planar点
        for (int j = 0; j < 6; j++)
        {
            int sp = scanStartInd[i] + (scanEndInd[i] - scanStartInd[i]) * j / 6; 
            //cout<<"scanStartInd["<<i<<"]数值="<<scanStartInd[i]<<endl;
            int ep = scanStartInd[i] + (scanEndInd[i] - scanStartInd[i]) * (j + 1) / 6 - 1;//减去1才能保证这一小段是6个数
            //cout<<"scanEndInd["<<i<<"]数值="<<scanEndInd[i]<<endl;

            TicToc t_tmp;
            //按照升序排列,但只是改变了点在容器中排列的顺序,并不改变每个点的下标值
            std::sort (cloudOriginalSortInd + sp, cloudOriginalSortInd + ep + 1, comp);//左闭右开区间,所以右侧要+1
            //对于16线激光雷达而言,需要进行比较排列96次;cloudSortInd一直是0,sp是变化的

            t_q_sort += t_tmp.toc();
            //printf("sort time %f \n", t_tmp.toc());

            int largestPickedNum = 0;
            //降序找最大的值来当作边缘点
            for (int k = ep; k >= sp; k--)
            {
                int ind = cloudOriginalSortInd[k]; //按照排序之后的数组,形成局部调整后的作用域
                if (cloudNeighborPicked[ind] == 0 &&cloudCurvature[ind] > 0.1)
                //首先需要判断是否为一个有效的点,其次需要进行判断曲率是否大于0.1,分类到面点还是角点
                //至于为啥是0.1,个人觉得他应该是把所有点的曲率都打印了出来,然后选择几个代表性的曲率值
                {
                    //分成6段以后,每段选择2个曲率比较大的点
                    largestPickedNum++;
                    if (largestPickedNum <= 2)
                    {                        
                        cloudLabel[ind] = 2;//这里是2个边缘点
                        cornerPointsSharp.push_back(laserCloud->points[ind]);
                        //pionts本来就是一种vector的数据类型;并将原始点云数组里面的点云插入到对应的容器里面
                        cornerPointsLessSharp.push_back(laserCloud->points[ind]);
                    }
                    //下面这种类型的角点,包含上面那种类型的角点
                    else if (largestPickedNum <= 20)
                    {                        
                        cloudLabel[ind] = 1; 
                        cornerPointsLessSharp.push_back(laserCloud->points[ind]);
                    }
                    else
                    {
                        break;
                    }

                    cloudNeighborPicked[ind] = 1; //被选中过的特征点,标志位置1

                    for (int l = 1; l <= 5; l++)
                    {
                        float diffX = laserCloud->points[ind + l].x - laserCloud->points[ind + l - 1].x;
                        float diffY = laserCloud->points[ind + l].y - laserCloud->points[ind + l - 1].y;
                        float diffZ = laserCloud->points[ind + l].z - laserCloud->points[ind + l - 1].z;
                        if (diffX * diffX + diffY * diffY + diffZ * diffZ > 0.05)
                        {
                            break;//break负责终止离它最近的for循环
                        }
                        //此for循环是为了避免特征点周围的一定距离(0.05m)里面的点,下次被选取到,显得在300个点中选取的几个点分布不均匀
                        //而且这里面,corner点是相对稀缺的,所以提取特征点时候,先提取角点,可以因为角点,忽略面点.但是不能够相反
                        cloudNeighborPicked[ind + l] = 1;
                    }
                    for (int l = -1; l >= -5; l--)//下面同理,再往右面找5个
                    {
                        float diffX = laserCloud->points[ind + l].x - laserCloud->points[ind + l + 1].x;
                        float diffY = laserCloud->points[ind + l].y - laserCloud->points[ind + l + 1].y;
                        float diffZ = laserCloud->points[ind + l].z - laserCloud->points[ind + l + 1].z;
                        if (diffX * diffX + diffY * diffY + diffZ * diffZ > 0.05)
                        {
                            break;
                        }
                        cloudNeighborPicked[ind + l] = 1;
                    }
                }
            }
               //注意面点和角点寻找曲率的最大值和最小值的方向正好是相反的,优化的效率很重要!
            int smallestPickedNum = 0;
            for (int k = sp; k <= ep; k++)
            {
                int ind = cloudOriginalSortInd[k];

                if (cloudNeighborPicked[ind] == 0 &&cloudCurvature[ind] < 0.1)
                {
                    cloudLabel[ind] = -1; //这里认为是比较平坦的点,不具体区分比较平坦的点和非常平坦的点
                    surfPointsFlat.push_back(laserCloud->points[ind]);

                    smallestPickedNum++;
                    if (smallestPickedNum >= 4)
                    { 
                        break;
                    }

                    cloudNeighborPicked[ind] = 1;
                    for (int l = 1; l <= 5; l++)
                    { 
                        float diffX = laserCloud->points[ind + l].x - laserCloud->points[ind + l - 1].x;
                        float diffY = laserCloud->points[ind + l].y - laserCloud->points[ind + l - 1].y;
                        float diffZ = laserCloud->points[ind + l].z - laserCloud->points[ind + l - 1].z;
                        if (diffX * diffX + diffY * diffY + diffZ * diffZ > 0.05)
                        {
                            break;
                        }

                        cloudNeighborPicked[ind + l] = 1;
                    }
                    for (int l = -1; l >= -5; l--)
                    {
                        float diffX = laserCloud->points[ind + l].x - laserCloud->points[ind + l + 1].x;
                        float diffY = laserCloud->points[ind + l].y - laserCloud->points[ind + l + 1].y;
                        float diffZ = laserCloud->points[ind + l].z - laserCloud->points[ind + l + 1].z;
                        if (diffX * diffX + diffY * diffY + diffZ * diffZ > 0.05)
                        {
                            break;
                        }

                        cloudNeighborPicked[ind + l] = 1;
                    }
                }
            }

            for (int k = sp; k <= ep; k++)
            {
                //这里可以看到,除了两种角点,剩下来的点都认为是面点
                //为什么可以的,可以查看雷达里程计了解详情,会对角点和面点进行校验
                if (cloudLabel[k] <= 0)
                {
                    surfPointsLessFlatScan->push_back(laserCloud->points[k]);
                    //pcl::PointCloud<PointType> 类型既有points这个属性,还有push_back这个方法
                }
            }
        }

        pcl::PointCloud<PointType> surfPointsLessFlatScanDS;
        pcl::VoxelGrid<PointType> downSizeFilter;
        //一般平坦的点较多,这里做了体素滤波(下采样的一种),然后针对于这样的方式在空间建立一定体积的格栅,保证分散和精度;节约计算资源
        downSizeFilter.setInputCloud(surfPointsLessFlatScan);
        downSizeFilter.setLeafSize(0.2, 0.2, 0.2);
        downSizeFilter.filter(surfPointsLessFlatScanDS);

        surfPointsLessFlat += surfPointsLessFlatScanDS;
    }
    printf("sort q time %f \n", t_q_sort);
    printf("seperate points time %f \n", t_pts.toc());

    //分别将当前点云,4种特征的点云发布出去
    sensor_msgs::PointCloud2 laserCloudOutMsg;
    pcl::toROSMsg(*laserCloud, laserCloudOutMsg);//转换到ROSMsg名字为laserCloudOutMsg的数据类型
    laserCloudOutMsg.header.stamp = laserCloudMsg->header.stamp;
    laserCloudOutMsg.header.frame_id = "/camera_init";
    pubLaserCloud.publish(laserCloudOutMsg);
    // std::cout<<"scanRegistration输出cornerPointsSharp.size()="<<cornerPointsSharp.size()<<std::endl;
    // for(int i=0;i<=(int)cornerPointsSharp.size();i++)
    // {
    //   std::cout<<"scanRegistration输出cornerPointsSharp["<<i<<"]数据="<<cornerPointsSharp[i]<<std::endl;
    // }
    //上面这段儿,只是为了验证,将曲率符合标准的激光点,扔入到对应的容器里面,得到的顺序将会是从0开始重新进行排序的结果了
    sensor_msgs::PointCloud2 cornerPointsSharpMsg;
    pcl::toROSMsg(cornerPointsSharp, cornerPointsSharpMsg);
    cornerPointsSharpMsg.header.stamp = laserCloudMsg->header.stamp;
    cornerPointsSharpMsg.header.frame_id = "/camera_init";
    pubCornerPointsSharp.publish(cornerPointsSharpMsg);

    sensor_msgs::PointCloud2 cornerPointsLessSharpMsg;
    pcl::toROSMsg(cornerPointsLessSharp, cornerPointsLessSharpMsg);
    cornerPointsLessSharpMsg.header.stamp = laserCloudMsg->header.stamp;
    cornerPointsLessSharpMsg.header.frame_id = "/camera_init";
    pubCornerPointsLessSharp.publish(cornerPointsLessSharpMsg);//发送的频率完全是自己掌握的。但是subscrib不可以

    sensor_msgs::PointCloud2 surfPointsFlat2;
    pcl::toROSMsg(surfPointsFlat, surfPointsFlat2);
    surfPointsFlat2.header.stamp = laserCloudMsg->header.stamp;
    surfPointsFlat2.header.frame_id = "/camera_init";
    pubSurfPointsFlat.publish(surfPointsFlat2);

    sensor_msgs::PointCloud2 surfPointsLessFlat2;
    pcl::toROSMsg(surfPointsLessFlat, surfPointsLessFlat2);
    //surfPointsLessFlatScan经过下采样体素滤波变成surfPointsLessFlat
    surfPointsLessFlat2.header.stamp = laserCloudMsg->header.stamp;
    surfPointsLessFlat2.header.frame_id = "/camera_init";
    pubSurfPointsLessFlat.publish(surfPointsLessFlat2);

    // pub each scan
    //按照每一个scan pub出去,但是这里没有执行
    if(PUB_EACH_LINE)
    {
        for(int i = 0; i< N_SCANS; i++)
        {
            sensor_msgs::PointCloud2 scanMsg;
            pcl::toROSMsg(laserCloudScans[i], scanMsg);
            scanMsg.header.stamp = laserCloudMsg->header.stamp;
            scanMsg.header.frame_id = "/camera_init";
            pubEachScan[i].publish(scanMsg);
        }
    }
    printf("scan registration time %f ms *************\n", t_whole.toc());
    if(t_whole.toc() > 100)//计算每一次配准的时间是否超过100毫秒
        ROS_WARN("scan registration process over 100ms");
        //等级由低到高：debug<info<warn<Error<Fatal;后三个，警告、错误、严重错误
}
int main(int argc, char **argv)
{
    ros::init(argc, argv, "scanRegistration");//是ROS程序调用的第一个函数，用于对ROS程序的初始化
    ros::NodeHandle nh;//是全局命名空间
    //ros::NodeHandle nh("~");是局部命名空间
    nh.param<int>("scan_line", N_SCANS, 16);//初始化lidar线束数值是16线的

    nh.param<double>("minimum_range", MINIMUM_RANGE, 0.1);//把近距离的点去除掉,使得车不会出现在地图里
    //nh.param("param3",parameter3,123);//去param3上找value,如果没有找到就会把123赋值给parameter3,123相当于默认值
    printf("scan line number %d \n", N_SCANS);

    if(N_SCANS != 16 && N_SCANS != 32 && N_SCANS != 64)
    {
        printf("only support velodyne with 16, 32 or 64 scan line!");
        return 0;
    }

    ros::Subscriber subLaserCloud = nh.subscribe<sensor_msgs::PointCloud2>("/velodyne_points", 100, laserCloudHandler);
//告诉 master 我们要订阅 chatter 话题上的消息。当有消息发布到这个话题一次时，ROS 就会调用一次 chatterCallback() 函数
//第二个参数是队列大小，以防我们处理消息的速度不够快，当缓存达到 1000 条消息后，再有新的消息到来就将开始丢弃先前接收的消息
//前面代码提到的laserCloudHandler函数，再次作为参数被新的函数所引用的时候，就是回调函数
    pubLaserCloud = nh.advertise<sensor_msgs::PointCloud2>("/velodyne_cloud_2", 100);

    pubCornerPointsSharp = nh.advertise<sensor_msgs::PointCloud2>("/laser_cloud_sharp", 100);

    pubCornerPointsLessSharp = nh.advertise<sensor_msgs::PointCloud2>("/laser_cloud_less_sharp", 100);

    pubSurfPointsFlat = nh.advertise<sensor_msgs::PointCloud2>("/laser_cloud_flat", 100);

    pubSurfPointsLessFlat = nh.advertise<sensor_msgs::PointCloud2>("/laser_cloud_less_flat", 100);

    pubRemovePoints = nh.advertise<sensor_msgs::PointCloud2>("/laser_remove_points", 100);

    if(PUB_EACH_LINE)
    {
        for(int i = 0; i < N_SCANS; i++)
        {
            ros::Publisher tmp = nh.advertise<sensor_msgs::PointCloud2>("/laser_scanid_" + std::to_string(i), 100);
            pubEachScan.push_back(tmp);
        }
    }
    ros::spin();
//因此千万不要认为，只要指定了回调函数，系统就回去自动触发，你必须ros::spin()或者ros::spinOnce()才能真正使回调函数生效
//ros::spin()用于调用所有可触发的回调函数，将进入循环，不会返回，类似于在循环里反复调用spinOnce() 
    return 0;
}
