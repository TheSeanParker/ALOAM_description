// Author:   Tong Qin               qintonguav@gmail.com
// 	         Shaozu Cao 		    saozu.cao@connect.ust.hk

#include <ceres/ceres.h>
#include <ceres/rotation.h>
#include <eigen3/Eigen/Dense>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl/kdtree/kdtree_flann.h>
#include <pcl_conversions/pcl_conversions.h>
//代价函数的计算模型
struct LidarEdgeFactor
{  
	Eigen::Vector3d     curr_point,
						last_point_a,
						last_point_b;
						double s; 
	//struct和class的差别是在访问权限上面,在一个访问符出现之前,struct定义的成员默认都是public的,而class默认都是private的
	//Eigen::Vector3d v_3d;声明一个3*1(3行1列向量,类型是double)
	/*******************************************************************/
	/************下面两行最主要的作用是传入点一个点和两个线点******************/
	/******************************************************************/
	//计算点到直线的距离作为误差函数
	LidarEdgeFactor(Eigen::Vector3d curr_point_, Eigen::Vector3d last_point_a_,Eigen::Vector3d last_point_b_, double s_)
		            : curr_point(curr_point_)
					, last_point_a(last_point_a_)
					, last_point_b(last_point_b_), 
					  s(s_) {}
	//初始化列表的顺序一定要和private中变量声明的顺序是一致的
    //上面的写法是初始化列表,为了区分初始化和赋值,提升代码效率;上面这段就是起到了赋初始值的作用了,构造函数一般就是初始化变量
	//相当于将输入变量curr_point_赋值给curr_point()函数
	template <typename T>
	//定义一个类模板,写一个模板函数,避免写类型的复杂定义;而后在重载()运算符的时候,无论输入什么类型的变量,都可以进行运算
	//下面叫做重载函数调用运算符,它需要类构造对象以后,才能调用对应的运算符,实际操作时,类也没有构造对象,怎样才能使用上对应的重载运算符呢?
	bool operator()(const T* q, const T* t, T* residual) const
	//左面3个量分别是旋转,平移,然后计算残差
	{   //Matrix<typename Scalar, int RowsAtCompileTime, int ColsAtCompileTime>
		//表达的是Scalar是表示元素的类型，RowsAtCompileTime为矩阵的行，ColsAtCompileTime为矩阵的列
		Eigen::Matrix<T, 3, 1> cp{T(curr_point.x()), T(curr_point.y()), T(curr_point.z())};//cp表达当前点
		//声明一个3*1的T类型的矩阵.T待定.T()是对里面的元素进行了强制转换
		Eigen::Matrix<T, 3, 1> lpa{T(last_point_a.x()), T(last_point_a.y()), T(last_point_a.z())};
		//左侧如果把T定义成为double就是vector3d
		Eigen::Matrix<T, 3, 1> lpb{T(last_point_b.x()), T(last_point_b.y()), T(last_point_b.z())};
        
		//std::cout<<"标识符重载里面lidar_odom t_last_curr数值="<<std::endl;

		//Eigen::Quaternion<T> q_last_curr{q[3], T(s) * q[0], T(s) * q[1], T(s) * q[2]};
		Eigen::Quaternion<T> q_last_curr{q[3], q[0], q[1], q[2]};//将从rosMsg()里面传来数据以后,进行一下数据顺序的调整
		Eigen::Quaternion<T> q_identity{T(1), T(0), T(0), T(0)};
		//定义了一个单位四元数,identity是单位四元数的含义,雷达帧间的运动位姿是什么样子的(主要靠插值)

        // if(s!=9)
		// {
		// std::cout<<"Pre_q_last_curr_X="<<q_last_curr.x()<<std::endl;
        // std::cout<<"Pre_q_last_curr_Y="<<q_last_curr.y()<<std::endl;
        // std::cout<<"Pre_q_last_curr_Z="<<q_last_curr.z()<<std::endl;
        // std::cout<<"Pre_q_last_curr_W="<<q_last_curr.w()<<std::endl;
		// }
			//验证s的数值
			// if(s!=1)
			// {std::cout<<"s数值="<<s<<std::endl;}//s就是一直等于1
			// else
			// {std::cout<<"s数值="<<s<<std::endl;}
		//以下已经是kitti数据集去掉运动畸变之后的数据,所以可以忽略下面两行的作用
		Eigen::Matrix<T, 3, 1> t_last_curr{T(s) * t[0], T(s) * t[1], T(s) * t[2]};

		// std::cout<<"t_last_curr t[0]数值="<<t[0]<<std::endl;
		// std::cout<<"T(s) * t[0]数值="<<T(s)*t[0]<<std::endl;
		// std::cout<<"t_last_curr t[1]="<<t[1]<<std::endl;
		// std::cout<<"T(s) * t[1]数值="<<T(s)*t[1]<<std::endl;
		// std::cout<<"t_last_curr t[2]="<<t[2]<<std::endl;
		// std::cout<<"T(s) * t[2]数值="<<T(s)*t[2]<<std::endl;
             //上下数值是一样的
		q_last_curr = q_identity.slerp(T(s), q_last_curr);//s恒等于1
		//可能是激光雷达没有完全转过一圈,所以将实际转过了多少打印出来  
		
		Eigen::Matrix<T, 3, 1> lp;
		//将当前帧的点云根据当前计算帧间位姿变换到上一帧坐标系中,只有在同一个雷达坐标系才能计算残差(计算机life)
		//Odometry线程时,下面是将当前帧Lidar坐标系下的cp点变换到上一帧的Lidar坐标系下,然后在上一帧Lidar坐标系下计算点到线的残差距离(任乾)
		//Mapping线程时,下面是将当前帧Lidar坐标系下的cp点变换到world坐标系下,然后在world坐标系下计算点到线的残差距离
		/******************************************************************/
	    /********转换点云并计算残差,但不再代码里输入雅克比(自动求导方式)************/
		/********F_LOAM使用的方法是解析求导,计算效率更高************/
	    /******************************************************************/
		lp = q_last_curr * cp + t_last_curr;//cp：current point
		//这里面的q_last_curr就是表示转移矩阵R，t_last_curr 表示转移矩阵的t，这二者都是待求的数据！！！
		//std::cout<<"t_last_curr数值=\n"<<t_last_curr<<std::endl;//demo
		//std::cout<<"q_last_curr数值="<<q_last_curr<<std::endl;//demo
		//点到线的计算如下所示
		//下面定义的模式是矢量形式的,没有取模.但是实际论文里面是有取模的.但就结果而言是一致的
		Eigen::Matrix<T, 3, 1> nu = (lp - lpa).cross(lp - lpb);//叉乘后是3点所在平面的法向量
		//上面标识符各自含义是lpa：last point a; lp:last point
		Eigen::Matrix<T, 3, 1> de = lpa - lpb;//模是底边的长度
        //Eigen是通过重载C++操作运算符通过dot()、cross()等来实现矩阵/向量的操作运算符
		//向量的点乘也叫向量的内积、数量积，对两个向量执行点乘运算，就是将两个向量对应一一相乘之后求和的操作，点乘结果是标量
		//叉乘又叫向量积、外积、叉集、叉乘的运算结果是一个向量而不是一个标量。两个向量的叉积与这两个向量组成的坐标平面垂直
		residual[0] = nu.x() / de.norm();//残差的x坐标
		residual[1] = nu.y() / de.norm();//残差的y坐标
		residual[2] = nu.z() / de.norm();//残差的z坐标
		//最终的残差本应该是residual[0]=nu.norm()/de.norm();为啥分成3个,任乾也不清楚.但他从试验反馈,下面的残差形式,效果会好一些的
        //上面不定义三个量,直接求出模的值也是可以
		//需要注意的是,所有的redisual都不用加fabs,因为ceres内部会对其求平方,作为最终的残差项
		return true;
	}
    //类内静态成员函数;
	//下面是函数指针,返回结果是ceres::CostFunction *类对象类型的指针
	static ceres::CostFunction* Create(const Eigen::Vector3d curr_point_,   const Eigen::Vector3d last_point_a_, 
									   const Eigen::Vector3d last_point_b_, const double s_)
	{
		return (new ceres::AutoDiffCostFunction<LidarEdgeFactor, 3, 4, 3>(new LidarEdgeFactor(curr_point_, last_point_a_, last_point_b_, s_)));
		//&取函数地址,new指针,堆只能用指针去存储地址间接存储数据//第一个LidarEdgeFactor是类,要求后面传入的指针类型是LidarEdgeFactor
	}
};

struct LidarPlaneFactor
{
	/*******************************************************************/
	/************下面两行最主要的作用是传入1个独立点和3个面点*****************/
	/******************************************************************/
	//(1)类内的构造函数
	LidarPlaneFactor(Eigen::Vector3d curr_point_, Eigen::Vector3d last_point_j_,
					 Eigen::Vector3d last_point_l_, Eigen::Vector3d last_point_m_, double s_)
		           : curr_point(curr_point_), last_point_j(last_point_j_),
				     last_point_l(last_point_l_),   last_point_m(last_point_m_),    s(s_)
	{
		ljm_norm = (last_point_j - last_point_l).cross(last_point_j - last_point_m);//得出在此平面的法向量
		ljm_norm.normalize();//相当于处于abs(last_point_j - last_point_l)*abs(last_point_j - last_point_m)；
		//上面得出的是单位法向量
	}
    //(2)类内的操作符重载函数
	template <typename T>//函数模板
	bool operator()(const T *q, const T *t, T *residual) const
	{
		Eigen::Matrix<T, 3, 1> cp{T(curr_point.x()), T(curr_point.y()), T(curr_point.z())};
		Eigen::Matrix<T, 3, 1> lpj{T(last_point_j.x()), T(last_point_j.y()), T(last_point_j.z())};
		//Eigen::Matrix<T, 3, 1> lpl{T(last_point_l.x()), T(last_point_l.y()), T(last_point_l.z())};
		//Eigen::Matrix<T, 3, 1> lpm{T(last_point_m.x()), T(last_point_m.y()), T(last_point_m.z())};
		Eigen::Matrix<T, 3, 1> ljm{T(ljm_norm.x()), T(ljm_norm.y()), T(ljm_norm.z())};

		//Eigen::Quaternion<T> q_last_curr{q[3], T(s) * q[0], T(s) * q[1], T(s) * q[2]};
		Eigen::Quaternion<T> q_last_curr{q[3], q[0], q[1], q[2]};
		Eigen::Quaternion<T> q_identity{T(1), T(0), T(0), T(0)};
		q_last_curr = q_identity.slerp(T(s), q_last_curr);
		Eigen::Matrix<T, 3, 1> t_last_curr{T(s) * t[0], T(s) * t[1], T(s) * t[2]};

		Eigen::Matrix<T, 3, 1> lp;
		lp = q_last_curr * cp + t_last_curr;

		residual[0] = (lp - lpj).dot(ljm);//残差[0]

		return true;
	}
    //(3)类内其他函数
	static ceres::CostFunction *Create(const Eigen::Vector3d curr_point_, const Eigen::Vector3d last_point_j_,
									   const Eigen::Vector3d last_point_l_, const Eigen::Vector3d last_point_m_,
									   const double s_)
	{
		return (new ceres::AutoDiffCostFunction<LidarPlaneFactor, 1, 4, 3>(//模板重载函数
			new LidarPlaneFactor(curr_point_, last_point_j_, last_point_l_, last_point_m_, s_)));
	}
	//类内的成员变量
    private:
	Eigen::Vector3d curr_point, last_point_j, last_point_l, last_point_m;
	Eigen::Vector3d ljm_norm;
	double s;
};

struct LidarPlaneNormFactor
{

	LidarPlaneNormFactor(Eigen::Vector3d curr_point_, Eigen::Vector3d plane_unit_norm_,
						 double negative_OA_dot_norm_) : curr_point(curr_point_), plane_unit_norm(plane_unit_norm_),
														 negative_OA_dot_norm(negative_OA_dot_norm_) {}

	template <typename T>
	bool operator()(const T *q, const T *t, T *residual) const
	{
		Eigen::Quaternion<T> q_w_curr{q[3], q[0], q[1], q[2]};
		Eigen::Matrix<T, 3, 1> t_w_curr{t[0], t[1], t[2]};
		Eigen::Matrix<T, 3, 1> cp{T(curr_point.x()), T(curr_point.y()), T(curr_point.z())};
		Eigen::Matrix<T, 3, 1> point_w;
		point_w = q_w_curr * cp + t_w_curr;

		Eigen::Matrix<T, 3, 1> norm(T(plane_unit_norm.x()), T(plane_unit_norm.y()), T(plane_unit_norm.z()));
		residual[0] = norm.dot(point_w) + T(negative_OA_dot_norm);
		//dot就是内积的含义
		return true;
	}

	static ceres::CostFunction* Create(const Eigen::Vector3d curr_point_, const Eigen::Vector3d plane_unit_norm_,
									   const double negative_OA_dot_norm_)
	{
		return (new ceres::AutoDiffCostFunction<LidarPlaneNormFactor, 1, 4, 3>(
			new LidarPlaneNormFactor(curr_point_, plane_unit_norm_, negative_OA_dot_norm_)));
	}

	Eigen::Vector3d curr_point;
	Eigen::Vector3d plane_unit_norm;
	double negative_OA_dot_norm;
};


struct LidarDistanceFactor
{

	LidarDistanceFactor(Eigen::Vector3d curr_point_, Eigen::Vector3d closed_point_) 
						: curr_point(curr_point_), closed_point(closed_point_){}

	template <typename T>
	bool operator()(const T *q, const T *t, T *residual) const
	{
		Eigen::Quaternion<T> q_w_curr{q[3], q[0], q[1], q[2]};
		Eigen::Matrix<T, 3, 1> t_w_curr{t[0], t[1], t[2]};
		Eigen::Matrix<T, 3, 1> cp{T(curr_point.x()), T(curr_point.y()), T(curr_point.z())};
		Eigen::Matrix<T, 3, 1> point_w;
		point_w = q_w_curr * cp + t_w_curr;


		residual[0] = point_w.x() - T(closed_point.x());
		residual[1] = point_w.y() - T(closed_point.y());
		residual[2] = point_w.z() - T(closed_point.z());
		return true;
	}

	static ceres::CostFunction *Create(const Eigen::Vector3d curr_point_, const Eigen::Vector3d closed_point_)
	{
		return (new ceres::AutoDiffCostFunction<LidarDistanceFactor, 3, 4, 3>(
			new LidarDistanceFactor(curr_point_, closed_point_)));
	}

	Eigen::Vector3d curr_point;
	Eigen::Vector3d closed_point;
};