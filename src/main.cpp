/*************************************************************************
	> File Name: main.cpp
	> Author: CeQi
	> Mail: qi_ce_0@163.com
	> Created Time: Sun 11 Oct 2015 08:49:46 PM CST
 ************************************************************************/

#include "prob_model.h"

int main(){
	CVM::ProbModel probModel;
	double orginal_mu=5.0;
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
	probModel.ByCat();

    return 0;
}
