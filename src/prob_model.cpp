/*************************************************************************
	> File Name: prob_model.cc
	> Author: CeQi
	> Mail: qi_ce_0@163.com
	> Created Time: Sun 11 Oct 2015 08:53:18 PM CST
 ************************************************************************/

#include"prob_model.h"
#include <cmath>
//#include <math.h>
#define NUM_SHOW_STARS_SCOPE 60 //should be even
#define M_PI 3.14159265358979323846 // so strange

namespace CVM{
    
    bool ProbModel::Init(const double &orginal_mu,const double &original_sig,const int &num_normal_data,const std::vector<double> &vec_original_probabilities,const int &num_categorical_data){
		//check parameters
		if (original_sig<0 || num_normal_data<=0 || vec_original_probabilities.size()==0 || num_categorical_data<=0){
			std::cout<<"Bad init parameters"<<std::endl;
			return false;
		}
		double dSum=0;
		int size=vec_original_probabilities.size();
		for (int x=0;x<size;++x){
			dSum+=vec_original_probabilities[x];
		}
		if (dSum!=1){
			std::cout<<"Bad init parameters:vec_original_probabilities"<<std::endl;
			return false;
		}
		//Init
		original_mu_=orginal_mu;
		original_sig_=original_sig;
		double nForX=-20.0;
		while (nForX<=30.0){
			vec_x_.push_back(nForX);
			nForX+=0.01;
		}
		NormPdf(vec_x_, original_mu_, original_sig_,vec_original_y_);
		const int nstars_normal=1000;    // maximum number of stars to distribute
		const int nstars_categorical=200;    // maximum number of stars to distribute
		std::default_random_engine generator;
		std::normal_distribution<double> distribution_normal(orginal_mu,original_sig);
		std::uniform_real_distribution<double> distribution_uniform_real(0.0,1.0);
		//normal distribution
		std::cout << "Generate normal_distribution ("<<orginal_mu<<","<<original_sig<<"):" << std::endl;
		int p[NUM_SHOW_STARS_SCOPE]={};
		for (int i=0; i<num_normal_data; ++i) {
			double number = distribution_normal(generator);
			vec_normal_data_.push_back(number);
			if ((number>=int(orginal_mu)-NUM_SHOW_STARS_SCOPE/2)&&(number<int(orginal_mu)+NUM_SHOW_STARS_SCOPE/2))
				++p[int(number+NUM_SHOW_STARS_SCOPE/2-int(orginal_mu))];
		}		
		for (int i=0; i<NUM_SHOW_STARS_SCOPE; ++i) {
			std::cout << int(orginal_mu)-NUM_SHOW_STARS_SCOPE/2+i << "бл" << int(orginal_mu)-NUM_SHOW_STARS_SCOPE/2+(i+1) << ": ";
			std::cout << std::string(p[i]*nstars_normal/num_normal_data,'.') << std::endl;
		}
		std::cout << "Generate normal_distribution finish"<< std::endl<< std::endl;
		//categorical distribution
		std::cout << "Generate categorical_distribution (";
		for (int i=0;i<size-1;++i){
			std::cout <<vec_original_probabilities[i]<<",";
		}
		std::cout <<vec_original_probabilities[size-1];
		std::cout <<")"<<std::endl;
		std::vector<int> vec_q;
		vec_q.resize(size);
		num_category_=size;
		vec_original_probabilities_=vec_original_probabilities;
		int flag;
		std::vector<double> vec_original_accumulative_probabilities=vec_original_probabilities;
		for (int m=1;m<size;++m){
			vec_original_accumulative_probabilities[m]=vec_original_accumulative_probabilities[m-1]+vec_original_probabilities[m];
		}
		for (int i=0; i<num_categorical_data; ++i) {
			double number = distribution_uniform_real(generator);
			flag=0;
			while(number>vec_original_accumulative_probabilities[flag]){
				flag++;
			}
			vec_categorical_data_.push_back(flag+1);//save 1,2,3,4..... rather than 0,1,2,3
			vec_q[flag]++;
		}		
		for (int i=0; i<size; ++i) {
			std::cout << i+1 << ": ";
			std::cout << std::string(vec_q[i]*nstars_categorical/num_categorical_data,'.') << std::endl;
		}
		std::cout << "Generate normal_distribution finish"<< std::endl<< std::endl;
		return true;
    }

	void ProbModel::MleNorm(){
		//Estimate the mean and the variance for the normal data .
		double estimated_mu=0.0;
		double estimated_sig=0.0;
		int size=vec_normal_data_.size();
		for (int i=0;i<size;++i)
		{
			estimated_mu+=vec_normal_data_[i];
		}
		estimated_mu/=size;
		for (int i=0;i<size;++i)
		{
			estimated_sig+=pow(vec_normal_data_[i]-estimated_mu,2.0);
		}
		estimated_sig/=size;	
		estimated_sig=sqrt(estimated_sig);
		//Estimate and print the error for the mean and the standard deviation.
		double muError = abs(original_mu_ - estimated_mu);
		double sigError = abs(original_sig_ - estimated_sig);
		std::cout<<"------MleNorm------"<<std::endl;
		std::cout<<"muError:"<<muError<<std::endl;
		std::cout<<"sigError:"<<sigError<<std::endl;
		std::vector<double> vec_estimated_y;
		NormPdf(vec_x_, estimated_mu, estimated_sig,vec_estimated_y);
		ShowAccuracy(vec_original_y_,vec_estimated_y);
		std::cout<<"------MleNorm------"<<std::endl<<std::endl;
	}

	void ProbModel::MapNorm(){
		//set parameters
		double alpha=1;
		double beta=1;
		double gamma=1;
		double delta=0;
		//Estimate the mean and the variance for the normal data .
		double estimated_mu=0.0;
		double estimated_sig=0.0;
		int size=vec_normal_data_.size();
		for (int i=0;i<size;++i)
		{
			estimated_mu+=vec_normal_data_[i];
		}
		estimated_mu+=gamma*delta;
		estimated_mu/=(size+gamma);
		for (int i=0;i<size;++i)
		{
			estimated_sig+=pow(vec_normal_data_[i]-estimated_mu,2.0);
		}
		estimated_sig+2*beta+gamma*pow( (delta-estimated_mu),2.0 );
		estimated_sig/=(size+3+2*alpha);	
		estimated_sig=sqrt(estimated_sig);
		//Estimate and print the error for the mean and the standard deviation.
		double muError = abs(original_mu_ - estimated_mu);
		double sigError = abs(original_sig_ - estimated_sig);
		std::cout<<"------MapNorm------"<<std::endl;
		std::cout<<"muError:"<<muError<<std::endl;
		std::cout<<"sigError:"<<sigError<<std::endl;
		std::vector<double> vec_estimated_y;
		NormPdf(vec_x_, estimated_mu, estimated_sig,vec_estimated_y);
		ShowAccuracy(vec_original_y_,vec_estimated_y);
		std::cout<<"------MapNorm------"<<std::endl<<std::endl;
	}

	void ProbModel::ByNorm(){
		//set parameters
		//prior parameters
		double alpha_prior=1;
		double beta_prior=1;
		double gamma_prior=1;
		double delta_prior=0;
		//post parameters
		double alpha_post;
		double beta_post;
		double gamma_post;
		double delta_post;
		//intermediate parameters
		double alpha_int;
		double beta_int;
		double gamma_int;
		double delta_int;
		std::vector<double> vec_estimated_y;
		//Compute posterior parameters.
		double sum=0.0;
		double sum2=0.0;
		//int size=vec_normal_data.size();
		int size=100;
		for (int i=0;i<size;++i)
		{
			sum+=vec_normal_data_[i];
			sum2+=pow(vec_normal_data_[i],2.0);
		}
		alpha_post = alpha_prior + size/2;
		beta_post = sum2/2 + beta_prior + (gamma_prior*pow(delta_prior,2.0))/2 - pow((gamma_prior*delta_prior + sum),2.0) / (2*(gamma_prior + size));
		gamma_post = gamma_prior + size;
		delta_post = (gamma_prior*delta_prior + sum) / (gamma_prior + size);
		//Compute intermediate parameters.
		alpha_int = alpha_post + 0.5;
		int nNumOfX=vec_x_.size();
		gamma_int = gamma_post + 1;
		double	x_prediction_up = sqrt(gamma_post) * pow(beta_post,alpha_post) * Gamma(alpha_int);
		for (int i=0;i<nNumOfX;++i)
		{
			beta_int = pow(vec_x_[i],2.0)/2 + beta_post + (gamma_post*pow(delta_post,2.0))/2 -	pow((gamma_post*delta_post + vec_x_[i]),2.0) / (2*gamma_post + 2);
			//Predict values for x_test.			
			double x_prediction_down = sqrt(2*M_PI) * sqrt(gamma_int) * Gamma(alpha_post)* pow(beta_int,alpha_int);
			vec_estimated_y.push_back( x_prediction_up / x_prediction_down);
		}
		
		//Estimate and print the error for the mean and the standard deviation.
		std::cout<<"------ByNorm------"<<std::endl;
		ShowAccuracy(vec_original_y_,vec_estimated_y);
		std::cout<<"------ByNorm------"<<std::endl<<std::endl;
	}

	void ProbModel::MleCat(){
		const int nstars_categorical=200;    // maximum number of stars to distribute
		std::vector<int> vec_q;
		vec_q.resize(num_category_);
		int num_categorical_data=vec_categorical_data_.size();
		for (int i=0;i<num_categorical_data;++i){
			vec_q[vec_categorical_data_[i]-1]++;
		}		
		//Print
		std::cout<<"------MleCat------"<<std::endl;
		for (int i=0; i<num_category_; ++i) {
			std::cout << i+1 << ": ";
			std::cout << std::string(vec_q[i]*nstars_categorical/num_categorical_data,'.') << std::endl;
		}
		//transfer from std::vector<int> -> std::vector<double>
		std::vector<double> vec_estimated_probabilities;
		vec_estimated_probabilities.resize(num_category_);
		for (int i=0;i<num_category_;++i){
			vec_estimated_probabilities[i]=double(vec_q[i])/num_categorical_data;
		}
		ShowAccuracy(vec_original_probabilities_,vec_estimated_probabilities);
		std::cout<<"------MleCat------"<<std::endl<<std::endl;
	}

	void ProbModel::MapCat(){
		const int nstars_categorical=200;    // maximum number of stars to distribute
		std::vector<double> vec_alpha;
		/*vec_alpha.push_back(1.0);
		vec_alpha.push_back(1.0);
		vec_alpha.push_back(1.0);
		vec_alpha.push_back(1.0);
		vec_alpha.push_back(1.0);
		vec_alpha.push_back(1.0);*/
		vec_alpha.push_back(10.0);
		vec_alpha.push_back(100.0);
		vec_alpha.push_back(1000.0);
		vec_alpha.push_back(1000.0);
		vec_alpha.push_back(100.0);
		vec_alpha.push_back(10.0);
		std::vector<int> vec_q;
		vec_q.resize(num_category_);
		int num_categorical_data=vec_categorical_data_.size();
		int num_length_of_alpha=vec_alpha.size();
		for (int i=0;i<num_categorical_data;++i){
			vec_q[vec_categorical_data_[i]-1]++;
		}		
		//Print
		std::cout<<"------MapCat------"<<std::endl;
		for (int i=0; i<num_category_; ++i) {
			std::cout << i+1 << ": ";
			std::cout << std::string(vec_q[i]*nstars_categorical/num_categorical_data,'.') << std::endl;
		}
		//transfer from std::vector<int> -> std::vector<double>
		std::vector<double> vec_estimated_probabilities;
		vec_estimated_probabilities.resize(num_category_);
		double sum_of_vec_alpha=0.0;
		for (int i=0;i>num_length_of_alpha;++i){
			sum_of_vec_alpha+=vec_alpha[i];
		}
		
		for (int i=0;i<num_category_;++i){
			vec_estimated_probabilities[i]=double(vec_q[i]-1+vec_alpha[i])/(num_categorical_data-num_length_of_alpha+sum_of_vec_alpha);
		}
		ShowAccuracy(vec_original_probabilities_,vec_estimated_probabilities);
		std::cout<<"------MapCat------"<<std::endl<<std::endl;
	}

	void ProbModel::ByCat(){
		const int nstars_categorical=200;    // maximum number of stars to distribute
		std::vector<double> vec_alpha_prior;
		vec_alpha_prior.push_back(1.0);
		vec_alpha_prior.push_back(1.0);
		vec_alpha_prior.push_back(1.0);
		vec_alpha_prior.push_back(1.0);
		vec_alpha_prior.push_back(1.0);
		vec_alpha_prior.push_back(1.0);
		//vec_alpha.push_back(10.0);
		//vec_alpha.push_back(100.0);
		//vec_alpha.push_back(1000.0);
		//vec_alpha.push_back(1000.0);
		//vec_alpha.push_back(100.0);
		//vec_alpha.push_back(10.0);
		std::vector<int> vec_q;
		vec_q.resize(num_category_);
		int num_categorical_data=vec_categorical_data_.size();
		int num_length_of_alpha=vec_alpha_prior.size();
		for (int i=0;i<num_categorical_data;++i){
			vec_q[vec_categorical_data_[i]-1]++;
		}		
		//Print
		std::cout<<"------ByCat------"<<std::endl;
		for (int i=0; i<num_category_; ++i) {
			std::cout << i+1 << ": ";
			std::cout << std::string(vec_q[i]*nstars_categorical/num_categorical_data,'.') << std::endl;
		}
		//transfer from std::vector<int> -> std::vector<double>
		std::vector<double> vec_estimated_probabilities;
		vec_estimated_probabilities.resize(num_category_);
		double sum_of_vec_alpha_post=0.0;
		std::vector<double> vec_alpha_post;
		vec_alpha_post.resize(vec_alpha_prior.size());
		int num_vec_alpha_post_size=vec_alpha_prior.size();
		for (int i=0;i<num_vec_alpha_post_size;++i){
			vec_alpha_post[i]=vec_q[i]+vec_alpha_prior[i];
		}
		
		for (int i=0;i<num_length_of_alpha;++i){
			sum_of_vec_alpha_post+=vec_alpha_post[i];
		}
		
		for (int i=0;i<num_category_;++i){
			vec_estimated_probabilities[i]=double(vec_alpha_post[i])/sum_of_vec_alpha_post;
		}
		ShowAccuracy(vec_original_probabilities_,vec_estimated_probabilities);
		std::cout<<"------ByCat------"<<std::endl<<std::endl;
	}

	void ProbModel::NormPdf(const std::vector<double> &vec_x,const double &mu,const double &sig,std::vector<double> &y){
		int size=vec_x.size();
		for (int i=0;i<size;++i)
		{
			y.push_back(  ( exp( -1*pow(vec_x[i]-mu,2.0)/(2*pow(sig,2.0))) ) / (sig*sqrt(2*M_PI))   );
		}		
	}

	void ProbModel::ShowAccuracy(const std::vector<double> &vec_y_original,const std::vector<double> &vec_y_prediction){
		double sum=0.0;
		if (vec_y_original.size()!=vec_y_prediction.size())
		{
			std::cout<<"the inputs of ShowAccuracy are not int the same size"<<std::endl;
			return;
		}
		int size=vec_y_original.size();
		for (int i=0;i<size;++i)
		{
			sum+=abs(vec_y_original[i] - vec_y_prediction[i])/vec_y_original[i];
		}
		sum/=size;
		std::cout<<"Error rate:"<<sum*100<<"%"<<std::endl;
	}

	double ProbModel::Gamma(double x)
	{ 
		int i;
		double y,t,s,u;
		static double a[11]={ 0.0000677106,-0.0003442342,
			0.0015397681,-0.0024467480,0.0109736958,
			-0.0002109075,0.0742379071,0.0815782188,
			0.4118402518,0.4227843370,1.0};
		if (x<=0.0){ 
			printf("err**x<=0!\n"); 
			return(-1.0);
		}
		y=x;
		if (y<=1.0){ 
			t=1.0/(y*(y+1.0)); 
			y=y+2.0;
		}
		else if (y<=2.0){ 
			t=1.0/y; 
			y=y+1.0;
		}
		else if (y<=3.0) 
			t=1.0;
		else{ 
			t=1.0;
			while (y>3.0){ 
				y=y-1.0; 
				t=t*y;
			}
		}
		s=a[0]; 
		u=y-2.0;
		for (i=1; i<=10; i++)
			s=s*u+a[i];
		s=s*t;
		return(s);
	}
    
}//namespace CVM
