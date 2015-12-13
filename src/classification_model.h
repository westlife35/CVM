/*************************************************************************
	> File Name: classification_model.h
	> Author: CeQi
	> Mail: qi_ce_0@163.com
	> Created Time: Sat 24 Dec 2015 09:52:49 PM CST
 ************************************************************************/

#ifndef _CLASSIFICATION_MODEL_H
#define _CLASSIFICATION_MODEL_H

#include <vector>
#include <iostream>
#include <string>
#include <random>
using std::vector;
using std::cout;
using std::endl;

#include "cv.h"
#include "highgui.h"
using cv::Mat;
using cv::Mat_;



namespace CVM{

class ClassificationModel{

public:
    //init to get the training data by hand;just for the show case, I just generate the normal data of dimension (n)(n=2 in this implementation) and indepent with each other
    bool Init(const std::vector<std::vector<double> > &vec_orginal_mu,const std::vector<std::vector<double> > &vec_original_sig,std::vector<int> vec_normal_data);
    //ques 9.2
	void BayesianLogisticRegression(const double var_prior,const vector<double> vec_initial_phi);

private:
	void NewtonMin(vector<double> vec_phi,const double var_prior);
	double Sigmoid(const double x);
	
private:
	std::vector<std::vector<double> > vec_original_mu_;
	std::vector<std::vector<double> > vec_original_sig_;
	std::vector<double> vec_original_x_;
	//vec_original_x_ with addesd one at starting of every vector
	std::vector<std::vector<double> > vec_original_x_train_;
	std::vector<std::vector<double> > vec_original_x_test_;
	//label w
	std::vector<double> vec_original_y_;
	std::vector<double> vec_original_w_;


    
};


}//namespace CVM
#endif
