/*************************************************************************
	> File Name: main.cpp
	> Author: CeQi
	> Mail: qi_ce_0@163.com
	> Created Time: Sun 11 Oct 2015 08:49:46 PM CST
 ************************************************************************/

#include "prob_model.h"//chapter 4
#include "class_gen_model.h"//chapter 6
#include "em_model.h"//chapter 7
#include "reg_model.h"//chapter 8
#include "classification_model.h"//chapter 9


int main(){
	//chapter 4
	/*CVM::ProbModel probModel;
	double orginal_mu=0.0;
	double original_sig=8.0;
	int num_normal_data=1000000;
	std::vector<double> vec_original_probabilities;//{0.25,0.15,0.1,0.1,0.15,0.25};
	vec_original_probabilities.push_back(0.25);
	vec_original_probabilities.push_back(0.15);
	vec_original_probabilities.push_back(0.1);
	vec_original_probabilities.push_back(0.1);
	vec_original_probabilities.push_back(0.15);
	vec_original_probabilities.push_back(0.25);
	int num_categorical_data=1000000;
	if (!probModel.Init(orginal_mu,original_sig,num_normal_data,vec_original_probabilities,num_categorical_data))
	{
		return 1;
	}
	probModel.MleNorm();
	probModel.MapNorm();
	probModel.ByNorm();
	probModel.MleCat();
	probModel.MapCat();
	probModel.ByCat();*/

	//chapter 6
	/*CVM::ClassGenModel classGenModel;
	classGenModel.Init("D:\\课程\\研一\\计算机视觉模型学习与推理\\数字字符图像\\char",10);
	classGenModel.GetFeature();
	classGenModel.BasicGenNorm();*/

	//chapter 7
	/*CVM::EmModel emModel;
	std::vector<std::vector<double> > vec_orginal_mu;
	vec_orginal_mu.resize(2);
	vec_orginal_mu[0].push_back(-1);
	vec_orginal_mu[0].push_back(2);
	vec_orginal_mu[1].push_back(1);
	vec_orginal_mu[1].push_back(7);
	std::vector<std::vector<double> > vec_original_sig;
	vec_original_sig.resize(2);
	vec_original_sig[0].push_back(2);
	vec_original_sig[0].push_back(0.5);
	vec_original_sig[1].push_back(1);
	vec_original_sig[1].push_back(1);
	std::vector<int> vec_normal_data;
	vec_normal_data.push_back(5000);
	vec_normal_data.push_back(5000);
	emModel.Init(vec_orginal_mu,vec_original_sig,vec_normal_data);
	emModel.FitMoG();*/

	//chapter 8
	/*CVM::RegModel regModel;
	std::vector<std::vector<double> > vec_orginal_mu;
	vec_orginal_mu.resize(3);
	vec_orginal_mu[0].push_back(-3);
	vec_orginal_mu[0].push_back(2);
	vec_orginal_mu[1].push_back(0);
	vec_orginal_mu[1].push_back(-3);
	vec_orginal_mu[2].push_back(4);
	vec_orginal_mu[2].push_back(3);
	std::vector<std::vector<double> > vec_original_sig;
	vec_original_sig.resize(3);
	vec_original_sig[0].push_back(0.5);
	vec_original_sig[0].push_back(0.5);
	vec_original_sig[1].push_back(0.5);
	vec_original_sig[1].push_back(0.5);
	vec_original_sig[2].push_back(0.5);
	vec_original_sig[2].push_back(0.5);
	std::vector<int> vec_normal_data;
	vec_normal_data.push_back(5);
	vec_normal_data.push_back(5);
	vec_normal_data.push_back(5);
	regModel.Init(vec_orginal_mu,vec_original_sig,vec_normal_data);
	regModel.GaussianProcessRegression(6,0.5);*/

	//chapter 9
	CVM::ClassificationModel classificationModel;
	std::vector<std::vector<double> > vec_orginal_mu;
	vec_orginal_mu.resize(2);
	vec_orginal_mu[0].push_back(-1);
	vec_orginal_mu[0].push_back(2.5);
	vec_orginal_mu[1].push_back(1);
	vec_orginal_mu[1].push_back(-2.5);
	std::vector<std::vector<double> > vec_original_sig;
	vec_original_sig.resize(2);
	vec_original_sig[0].push_back(0.5);
	vec_original_sig[0].push_back(0.5);
	vec_original_sig[1].push_back(0.5);
	vec_original_sig[1].push_back(0.5);
	std::vector<int> vec_normal_data;
	vec_normal_data.push_back(20);
	vec_normal_data.push_back(20);
	classificationModel.Init(vec_orginal_mu,vec_original_sig,vec_normal_data);
	vector<double> vec_initial_phi;
	vec_initial_phi.push_back(1);
	vec_initial_phi.push_back(1);
	vec_initial_phi.push_back(1);
	classificationModel.BayesianLogisticRegression(6,vec_initial_phi);

    return 0;
}
