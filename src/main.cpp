/*************************************************************************
	> File Name: main.cpp
	> Author: CeQi
	> Mail: qi_ce_0@163.com
	> Created Time: Sun 11 Oct 2015 08:49:46 PM CST
 ************************************************************************/

#include "prob_model.h"
#include "class_gen_model.h"

int main(){
	//CVM::ProbModel probModel;
	//double orginal_mu=0.0;
	//double original_sig=8.0;
	//int num_normal_data=1000000;
	//std::vector<double> vec_original_probabilities;//{0.25,0.15,0.1,0.1,0.15,0.25};
	//vec_original_probabilities.push_back(0.25);
	//vec_original_probabilities.push_back(0.15);
	//vec_original_probabilities.push_back(0.1);
	//vec_original_probabilities.push_back(0.1);
	//vec_original_probabilities.push_back(0.15);
	//vec_original_probabilities.push_back(0.25);
	//int num_categorical_data=1000000;
	//if (!probModel.Init(orginal_mu,original_sig,num_normal_data,vec_original_probabilities,num_categorical_data))
	//{
	//	return 1;
	//}
	//probModel.MleNorm();
	//probModel.MapNorm();
	//probModel.ByNorm();
	//probModel.MleCat();
	//probModel.MapCat();
	//probModel.ByCat();

	CVM::ClassGenModel classGenModel;
	classGenModel.Init("D:\\课程\\研一\\计算机视觉模型学习与推理\\数字字符图像\\char",10);
	classGenModel.GetFeature();
	classGenModel.BasicGenNorm();

	/*std::vector<std::vector<double>> vec_matrix;
	vec_matrix.resize(3);*/
	/*vec_matrix[0].push_back(60.105378704720090);
	vec_matrix[0].push_back(63.444566410537874);
	vec_matrix[0].push_back(55.481888035126230);
	vec_matrix[1].push_back(63.444566410537874);
	vec_matrix[1].push_back(66.969264544456640);
	vec_matrix[1].push_back(58.564215148188800);
	vec_matrix[2].push_back(55.481888035126230);
	vec_matrix[2].push_back(58.564215148188800);
	vec_matrix[2].push_back(51.214050493962680);*/
	/*vec_matrix[0].push_back(3);
	vec_matrix[0].push_back(1);
	vec_matrix[0].push_back(8);
	vec_matrix[1].push_back(1);
	vec_matrix[1].push_back(1);
	vec_matrix[1].push_back(6);
	vec_matrix[2].push_back(1);
	vec_matrix[2].push_back(2);
	vec_matrix[2].push_back(8);*/
	/*double a=classGenModel.CalculateMatrixModule(vec_matrix,3);
	std::cout<<"ss"<<a<<std::endl;*/


    return 0;
}
