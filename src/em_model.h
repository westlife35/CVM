/*************************************************************************
	> File Name: em_model.h
	> Author: CeQi
	> Mail: qi_ce_0@163.com
	> Created Time: Nov 11 Oct 2015 09:52:49 PM CST
 ************************************************************************/

#ifndef _EM_MODEL_H
#define _EM_MODEL_H

#include <vector>
#include <iostream>
#include <string>
#include <random>
//using std::vector;

#include "cv.h"
#include "highgui.h"


namespace CVM{

class EmModel{

public:
    //init to get the training data by hand;just for the show case, I just generate the normal data of dimension (n)(n=2 in this implementation) and indepent with each other
    bool Init(const std::vector<std::vector<double> > &vec_orginal_mu,const std::vector<std::vector<double> > &vec_original_sig,std::vector<int> vec_normal_data);
    //ques 7.1
	void FitMoG();
	//ques 7.2
	void FitT();

private:
	double GetNormalValue(const double &x,const double &mu,const double &sig_square);

private:
	int num_center_;	
	std::vector<std::vector<double> > vec_original_mu_;
	std::vector<std::vector<double> > vec_original_sig_;
	//save all the original data(n=2,so it is enoughs)
	std::vector<double> vec_original_lambda_;
	std::vector<double> vec_original_x_;
	std::vector<double> vec_original_y_;


    
};


}//namespace CVM
#endif
