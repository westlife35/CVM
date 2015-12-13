/*************************************************************************
	> File Name: reg_model.h
	> Author: CeQi
	> Mail: qi_ce_0@163.com
	> Created Time: Tue 24 Nov 2015 09:52:49 PM CST
 ************************************************************************/

#ifndef _REG_MODEL_H
#define _REG_MODEL_H

#include <vector>
#include <iostream>
#include <string>
#include <random>
//using std::vector;

#include "cv.h"
#include "highgui.h"


namespace CVM{

class RegModel{

public:
    //init to get the training data by hand;just for the show case, I just generate the normal data of dimension (n)(n=2 in this implementation) and indepent with each other
    bool Init(const std::vector<std::vector<double> > &vec_orginal_mu,const std::vector<std::vector<double> > &vec_original_sig,std::vector<int> vec_normal_data);
    //ques 8.3
	void GaussianProcessRegression(const double var_prior,const double lambda);

private:
	double GetNormalValue(const double &x,const double &mu,const double &sig_square);
	bool KernelGauss(const std::vector<double> &vec_x_i,const std::vector<double> &vec_x_j,const double &lambda,double &result);
	//not input the w, because it is the member of class, and can be used directly in the function
	double GoldenDivSearch(const double &under_boundary,const double &upper_boundary,const std::vector<std::vector<double> > &vec_K,const double &var_prior);
	//not input the w, because it is the member of class, and can be used directly in the function
	double var_log_fun(const double &var,const std::vector<std::vector<double> > &vec_K,const double &var_prior);
	double MvnPdf(const std::vector<double> &vec_w,const std::vector<double> &vec_mu,const std::vector<std::vector<double> > &vec_covariance);
	//matrix calculation
	double CalculateMatrixModule(const std::vector<std::vector<double>> &vec_matrix,const int &nDim);
	bool CalculateMatrixInverse(const std::vector<std::vector<double>> &vec_matrix_in,std::vector<std::vector<double>> &vec_matrix_out,const int &nDim);
	void CalculateMatrixAdjoint(const std::vector<std::vector<double>> &vec_matrix_in,std::vector<std::vector<double>> &vec_matrix_out,const int &nDim);

private:
	std::vector<std::vector<double> > vec_original_mu_;
	std::vector<std::vector<double> > vec_original_sig_;
	std::vector<double> vec_original_x_;
	//vec_original_x_ with added one at starting of every vector
	std::vector<std::vector<double> > vec_original_x_train;
	std::vector<std::vector<double> > vec_original_x_test;
	//label w
	std::vector<double> vec_original_w_;


    
};


}//namespace CVM
#endif
