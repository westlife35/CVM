/*************************************************************************
	> File Name: prob_model.h
	> Author: CeQi
	> Mail: qi_ce_0@163.com
	> Created Time: Sun 11 Oct 2015 08:52:48 PM CST
 ************************************************************************/

#ifndef _PROB_MODEL_H
#define _PROB_MODEL_H

#include <vector>
#include <iostream>
#include <string>
#include <random>

namespace CVM{

class ProbModel{

public:
    //init to get the training data by hand
    bool Init(const double &orginal_mu,const double &original_sig,const int &num_normal_data,const std::vector<double> &vec_original_probabilities,const int &num_categorical_data);
    //ques 4.1
    void MleNorm();
	void MapNorm();
	void ByNorm();
	void MleCat();
	void MapCat();
	void ByCat();

private:
	void NormPdf(const std::vector<double> &vec_x,const double &mu,const double &sig,std::vector<double> &y);
	void ShowAccuracy(const std::vector<double> &vec_y_original,const std::vector<double> &vec_y_prediction);
	double Gamma(double x);

private:
	std::vector<double> vec_normal_data_;
	std::vector<int> vec_categorical_data_;
	std::vector<double> vec_original_probabilities_;
	int num_category_;
	double original_mu_;
	double original_sig_;
	std::vector<double> vec_x_;
	std::vector<double> vec_original_y_;


    
};


}//namespace CVM
#endif
