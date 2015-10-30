/*************************************************************************
	> File Name: class_gen_model.h
	> Author: CeQi
	> Mail: qi_ce_0@163.com
	> Created Time: Tu 27 Oct 2015 03:17:48 PM CST
 ************************************************************************/

#ifndef _CLASS_GEN_MODEL_H
#define _CLASS_GEN_MODEL_H

#include <vector>
#include <iostream>
#include <string>
#include<windows.h>
#include "cv.h"
#include "highgui.h"

//typedef unsigned char uchar;

struct TrainFilePathAndLabel{
	std::string strFilePath;
	char label;
};

struct ZerooneFeatureAndLabel{
	std::vector<uchar> vec_zeroone_feature;
	//std::vector<double> vec_zeroone_feature;
	char label;
};

namespace CVM{

class ClassGenModel{

public:
	//init to get the training data by 
	//bool Init(const std::vector<double> &vec_orginal_mu,const std::vector<double> &vec_original_sig,const std::vector<int> &vec_num_normal_data);
	bool Init(const std::string &strTrainImagePath,const int &nPattern);
	bool GetFeature();
	//ques 6.1
	void BasicGenNorm();
//private:
	void CountTimeStart();
	void CountTimeStop();
	void ShowAccuracy(const std::vector<ZerooneFeatureAndLabel> &vec_zeroone_feature_and_label);
	double MvnPdf(const std::vector<uchar> &vec_feature,const std::vector<double> &vec_mu,const std::vector< std::vector<double> > &vec_sig);
	//double MvnPdf(const std::vector<double> &vec_feature,const std::vector<double> &vec_mu,const std::vector< std::vector<double> > &vec_sig);
	//use nDim will make the calculation easy
	double CalculateMatrixModule(const std::vector<std::vector<double>> &vec_matrix,const int &nDim);
	bool CalculateMatrixInverse(const std::vector<std::vector<double>> &vec_matrix_in,std::vector<std::vector<double>> &vec_matrix_out,const int &nDim);
	void CalculateMatrixAdjoint(const std::vector<std::vector<double>> &vec_matrix_in,std::vector<std::vector<double>> &vec_matrix_out,const int &nDim);

	double SumMatValue(const cv::Mat& image); 
	void CalcGradientFeat(const cv::Mat& imgSrc, std::vector<double>& feat);
private:
	std::vector<TrainFilePathAndLabel> vec_train_file_path_and_label_; 
	std::vector<ZerooneFeatureAndLabel> vec_zeroone_feature_and_label_;
	std::vector<int> vec_pattern_count_;
	int nPattern_;
	std::vector< std::vector<double> > vec_mu;
	std::vector< std::vector< std::vector<double> > > vec_sig;
	//count time
	 LARGE_INTEGER tStart_,tStop_,tc_;
};




}//namespace CVM
#endif
