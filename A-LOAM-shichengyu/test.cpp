#include<iostream>
int main(int argc, char const *argv[])\
{
    int x=3;
    int y=4;
    int* p =&x;
    int* p1=&y;
    *p +=*p1;
    std::cout<<"输出数值="<<*p<<std::endl;
    return 0;
}
