/*************************************************************************
	> File Name: em_model.cpp
	> Author: CeQi
	> Mail: qi_ce_0@163.com
	> Created Time: Nov 11 Oct 2015 09:52:49 PM CST
 ************************************************************************/

#include"em_model.h"
#include <cmath>
#include <numeric>
//#include <math.h>
#define NUM_SHOW_STARS_SCOPE 60 //should be even
#define M_PI 3.14159265358979323846 // so strange
#define MIN_DISTANCE_OF_OBJ_OF_TWO_ITERATION 0.01
#define NUM_OF_DIMENSION 2

namespace CVM{
    
    bool EmModel::Init(const std::vector<std::vector<double> > &vec_orginal_mu,const std::vector<std::vector<double> > &vec_original_sig,std::vector<int> vec_normal_data){
		//check parameters->just for the show case, I just generate the normal data of dimension (n)(n=2 in this implementation) and indepent with each other
		if (vec_orginal_mu.size()<0 || vec_original_sig.size()<=0 || vec_orginal_mu.size()!=vec_original_sig.size() || vec_orginal_mu.size()!=vec_normal_data.size() || vec_normal_data.size()!=vec_original_sig.size() ){
			std::cout<<"Bad init parameters"<<std::endl;
			return false;
		}
		//Init
		num_center_=vec_orginal_mu.size();
		vec_original_mu_=vec_orginal_mu;
		vec_original_sig_=vec_original_sig;
		int num_of_total_data=accumulate(vec_normal_data.begin(),vec_normal_data.end(),0);
		for (int i=0;i<num_center_;++i){
			vec_original_lambda_.push_back((double)vec_normal_data[i]/num_of_total_data);
		}		
		//generate data
		const int nstars_normal=1000;    // maximum number of stars to distribute
		const int nstars_categorical=200;    // maximum number of stars to distribute
		for (int n=0;n<num_center_;++n){
			for (int x=0;x<vec_orginal_mu[n].size();++x){
				std::default_random_engine generator;
				std::normal_distribution<double> distribution_normal(vec_orginal_mu[n][x],vec_original_sig[n][x]);
				//normal distribution
				std::cout << "Generate normal_distribution ("<<vec_orginal_mu[n][x]<<","<<vec_original_sig[n][x]<<"):" << std::endl;
				int p[NUM_SHOW_STARS_SCOPE]={};
				for (int i=0; i<vec_normal_data[n]; ++i) {
					double number = distribution_normal(generator);
					if (x==0)//x data:x==0
					{
						vec_original_x_.push_back(number);
					}
					else{ //y data:x==1
						vec_original_y_.push_back(number);
					}
					
					if ((number>=int(vec_orginal_mu[n][x])-NUM_SHOW_STARS_SCOPE/2)&&(number<int(vec_orginal_mu[n][x])+NUM_SHOW_STARS_SCOPE/2))
						++p[int(number+NUM_SHOW_STARS_SCOPE/2-int(vec_orginal_mu[n][x]))];
				}		
				for (int i=0; i<NUM_SHOW_STARS_SCOPE; ++i) {
					std::cout << int(vec_orginal_mu[n][x])-NUM_SHOW_STARS_SCOPE/2+i << "～" << int(vec_orginal_mu[n][x])-NUM_SHOW_STARS_SCOPE/2+(i+1) << ": ";
					std::cout << std::string(p[i]*nstars_normal/vec_normal_data[n],'.') << std::endl;
				}
				std::cout << "Generate normal_distribution ("<<vec_orginal_mu[n][x]<<","<<vec_original_sig[n][x]<<"):finish"<< std::endl<< std::endl;
			}			
		}	

		cv::Mat mCoordinate=cv::Mat::zeros(16*20,16*20,CV_8UC3);
		int num_of_point=vec_original_x_.size();
		////测试均值和方差是否正确
		//double sum=0.0;
		//for (int i=0;i<num_of_point;++i)
		//{
		//	sum+=vec_original_y_[i];
		//}
		//double avg=sum/num_of_point;
		//sum=0.0;
		//for (int i=0;i<num_of_point;++i)
		//{
		//	sum+=pow(vec_original_y_[i]-avg,2.0);
		//}
		//double siug=sum/num_of_point;
		//


		for (int i=0;i<num_of_point;++i){
			cv::circle(mCoordinate,cv::Point((vec_original_x_[i]+8)*20,(vec_original_y_[i]+8)*20),1,cv::Scalar(255,255,255),-1,8,0);
		}
		cv::imshow("original data",mCoordinate);
		cv::waitKey();

		return true;
    }

	void EmModel::FitMoG(){
		//check data;
		if (vec_original_x_.size()!=vec_original_y_.size()){
			std::cout<<"Bad training data"<<std::endl;
			return;
		}
		//alloc memory to save result
		std::vector<std::vector<double> > vec_estimate_mu;
		vec_estimate_mu.resize(num_center_);
		std::vector<std::vector<double> > vec_estimate_sig_square;
		vec_estimate_sig_square.resize(num_center_);
		//Initialize all values in lambda to 1/K. K equals to the number of the center
		//Initialize the values in mu to K randomly chosen unique datapoints.
		std::vector<double> vec_lambda;
		vec_lambda.resize(num_center_);
		int num_of_point=vec_original_x_.size();
		std::default_random_engine generator;
		std::uniform_int_distribution<int> distribution(0,num_of_point);
		for (int i=0;i<num_center_;++i){
			vec_lambda[i]= 1.0/num_center_; // using  1/double(num_center_); is right,too. but using 1.0/num_center_ is not work.all are set 0.
			int num_of_no=distribution(generator);
			num_of_no=299+i;
			vec_estimate_mu[i].push_back(  vec_original_x_[num_of_no]  );
			vec_estimate_mu[i].push_back(  vec_original_y_[num_of_no]  );
		}
		//For a diagonal covariance retain only the diagonal of the covariance update.
		double mu_x=0.0;
		double mu_y=0.0;
		double sig_square_x=0.0;
		double sig_square_y=0.0;
		for(int i=0;i<num_of_point;++i){
			mu_x+=vec_original_x_[i];
			mu_y+=vec_original_y_[i];
		}
		mu_x/=num_of_point;
		mu_y/=num_of_point;
		for (int i=0;i<num_of_point;++i){
			sig_square_x+=pow(vec_original_x_[i]-mu_x,2);
			sig_square_y+=pow(vec_original_y_[i]-mu_y,2);
		}
		sig_square_x/=num_of_point;
		sig_square_y/=num_of_point;
		for (int i=0;i<num_center_;++i){
			vec_estimate_sig_square[i].push_back(sig_square_x);
			vec_estimate_sig_square[i].push_back(sig_square_y);
		}
		//The main loop to get the estimate result
		double num_of_L=0.0;
		int no_of_iterations = 0;    
		double num_of_previous_L = 1000000;
		std::vector<std::vector<double> > vec_l;
		vec_l.resize(num_of_point);
		std::vector<std::vector<double> > vec_r;
		vec_r.resize(num_of_point);
		std::vector<double> vec_s;
		vec_s.resize(num_of_point);
		for (int i=0;i<num_of_point;++i){
			vec_l[i].resize(num_center_);
			vec_r[i].resize(num_center_);
		}	
		std::vector<double> vec_r_summed_rows;
		vec_r_summed_rows.resize(num_center_);
		double r_summed_all=0.0;
		while(true){
			//Expectation step.
			//Compute the numerator of Bayes' rule.
			for (int i=0;i<num_center_;++i){
				for (int j=0;j<num_of_point;++j){
					//the next line of code miss something
					//vec_l[j][i]=vec_lambda[i]*   ( exp( -1*pow(vec_original_x_[j]-vec_estimate_mu[i][0],2.0)/(2*pow(vec_estimate_sig[i][0],2.0))) ) / ( exp( -1*pow(vec_original_y_[j]-vec_estimate_mu[i][1],2.0)/(2*pow(vec_estimate_sig[i][1],2.0))) )  ;
					vec_l[j][i]=vec_lambda[i]*   GetNormalValue(vec_original_x_[j],vec_estimate_mu[i][0],vec_estimate_sig_square[i][0]) * GetNormalValue(vec_original_y_[j],vec_estimate_mu[i][1],vec_estimate_sig_square[i][1])    ;
				}				
			}
			//Compute the responsibilities by normalizing.
			/*vec_s.swap(std::vector<double>());
			vec_s.resize(num_of_point);*/
			memset(&*vec_s.begin(),0.0,vec_s.size()*sizeof(double));
			for (int i=0;i<num_of_point;++i){
				for (int j=0;j<num_center_;++j){
					vec_s[i]+=vec_l[i][j];
				}
			}
			for (int i=0;i<num_of_point;++i){
				for (int j=0;j<num_center_;++j){
					vec_r[i][j]=vec_l[i][j]/vec_s[i];
				}
			}
			//Maximization step.
			/*vec_r_summed_rows.swap(std::vector<double>());
			vec_r_summed_rows.resize(num_center_);*/
			memset(&*vec_r_summed_rows.begin(),0.0,vec_r_summed_rows.size()*sizeof(double));
			for (int i=0;i<num_center_;++i){
				for (int j=0;j<num_of_point;++j){
					vec_r_summed_rows[i]+=vec_r[j][i];
				}				
			}
			r_summed_all=0.0;
			for (int i=0;i<num_center_;++i){
				r_summed_all+=vec_r_summed_rows[i];
			}
			for(int i=0;i<num_center_;++i){
				//Update lambda.
				vec_lambda[i]=vec_r_summed_rows[i]/r_summed_all;
				//Update mu.
				std::vector<double> vec_new_mu;
				vec_new_mu.resize(NUM_OF_DIMENSION);
				for (int j=0;j<num_of_point;++j){
					for(int k=0;k<NUM_OF_DIMENSION;++k){
						if(k==0){//x
							vec_new_mu[k]+=vec_r[j][i]*vec_original_x_[j];
						}
						else{//y
							vec_new_mu[k]+=vec_r[j][i]*vec_original_y_[j];
						}						
					}					
				}				
				for(int k=0;k<NUM_OF_DIMENSION;++k){
					vec_estimate_mu[i][k]=vec_new_mu[k]/vec_r_summed_rows[i];
				}				
				//Update sigma.
				std::vector<double> vec_new_sig_square;
				vec_new_sig_square.resize(NUM_OF_DIMENSION);
				double sig_square_x=0.0;
				double sig_square_y=0.0;
				for(int j=0;j<num_of_point;++j){
					for(int k=0;k<NUM_OF_DIMENSION;++k){
						if (k==0){//x
							sig_square_x+=vec_r[j][i]*pow(vec_original_x_[j]-vec_estimate_mu[i][k],2);
						}
						else{
							sig_square_y+=vec_r[j][i]*pow(vec_original_y_[j]-vec_estimate_mu[i][k],2);
						}
					}					
				}
				sig_square_x/=vec_r_summed_rows[i];
				sig_square_y/=vec_r_summed_rows[i];
				vec_estimate_sig_square[i][0]=sig_square_x;
				vec_estimate_sig_square[i][1]=sig_square_y;				
			}
			//Compute the log likelihood L.
			//std::vector<std::vector<double> > vec_temp;
			std::vector<double> vec_temp;
			vec_temp.resize(num_of_point);
			/*for (int i=0;i<num_of_point;++i){
			vec_temp[i].resize(num_center_);
			}*/
			for (int i=0;i<num_center_;++i){
				for (int j=0;j<num_of_point;++j){
					//the next line of code miss something
					//vec_temp[j][i]=vec_lambda[i]*   ( exp( -1*pow(vec_original_x_[j]-vec_estimate_mu[i][0],2.0)/(2*pow(vec_estimate_sig[i][0],2.0))) ) / ( exp( -1*pow(vec_original_y_[j]-vec_estimate_mu[i][1],2.0)/(2*pow(vec_estimate_sig[i][1],2.0))) )  ;
					vec_temp[j]+=vec_lambda[i] * GetNormalValue(vec_original_x_[j],vec_estimate_mu[i][0],vec_estimate_sig_square[i][0]) * GetNormalValue(vec_original_y_[j],vec_estimate_mu[i][1],vec_estimate_sig_square[i][1])   ;					
				}
			}
			//log
			for (int j=0;j<num_of_point;++j){
				vec_temp[j]=log(vec_temp[j]);
			}
			num_of_L=accumulate(vec_temp.begin(),vec_temp.end(),0.0);
			no_of_iterations++;
			std::cout<<"-------------------------"<<std::endl;
			std::cout<<"iterations: "<<no_of_iterations<<"  L:"<<num_of_L<<std::endl;
			std::cout<<"-->lambda:"<<" ";
			for (int i=0;i<num_center_;++i){					
				std::cout<<vec_lambda[i]<<" ";				
			}
			std::cout<<std::endl;
			std::cout<<"-->mu:"<<std::endl;
			for (int i=0;i<num_center_;++i){
				for (int j=0;j<NUM_OF_DIMENSION;++j){
					std::cout<<vec_estimate_mu[i][j]<<" ";
				}
				std::cout<<std::endl;
			}
			std::cout<<"-->sig:"<<std::endl;
			for (int i=0;i<num_center_;++i){
				for (int j=0;j<NUM_OF_DIMENSION;++j){
					std::cout<<sqrt(vec_estimate_sig_square[i][j])<<" ";
				}
				std::cout<<std::endl;
			}
			std::cout<<"-------------------------"<<std::endl;

			//judge
			if(  abs(num_of_L-num_of_previous_L)<MIN_DISTANCE_OF_OBJ_OF_TWO_ITERATION  ){
				std::cout<<"iterations finished"<<std::endl;
				std::cout<<"------Result------"<<std::endl;
				//original mu and sig
				std::cout<<"---original data---"<<std::endl;
				std::cout<<"lambda:"<<" ";
				for (int i=0;i<num_center_;++i){					
					std::cout<<vec_original_lambda_[i]<<" ";				
				}
				std::cout<<std::endl;
				std::cout<<"mu:"<<std::endl;
				for (int i=0;i<num_center_;++i){
					for (int j=0;j<NUM_OF_DIMENSION;++j){
						std::cout<<vec_original_mu_[i][j]<<" ";
					}
					std::cout<<std::endl;
				}
				std::cout<<"sig:"<<std::endl;
				for (int i=0;i<num_center_;++i){
					for (int j=0;j<NUM_OF_DIMENSION;++j){
						std::cout<<vec_original_sig_[i][j]<<" ";
					}
					std::cout<<std::endl;
				}
				//estimated mu and sig
				std::cout<<"---estimated data---"<<std::endl;
				std::cout<<"lambda:"<<" ";
				for (int i=0;i<num_center_;++i){					
					std::cout<<vec_lambda[i]<<" ";				
				}
				std::cout<<std::endl;
				std::cout<<"mu:"<<std::endl;
				for (int i=0;i<num_center_;++i){
					for (int j=0;j<NUM_OF_DIMENSION;++j){
						std::cout<<vec_estimate_mu[i][j]<<" ";
					}
					std::cout<<std::endl;
				}
				std::cout<<"sig:"<<std::endl;
				for (int i=0;i<num_center_;++i){
					for (int j=0;j<NUM_OF_DIMENSION;++j){
						std::cout<<sqrt(vec_estimate_sig_square[i][j])<<" ";
					}
					std::cout<<std::endl;
				}
				break;
			}
			num_of_previous_L =num_of_L;
		}
	}

	void EmModel::FitT(){




	}

	double EmModel::GetNormalValue(const double &x,const double &mu,const double &sig_square){
		return ( exp( -1*pow(x-mu,2.0)/(2*sig_square)) ) / (sqrt(sig_square)*sqrt(2*M_PI));
	}

}//namespace CVM
