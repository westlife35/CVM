/*************************************************************************
	> File Name: classification_model.cpp
	> Author: CeQi
	> Mail: qi_ce_0@163.com
	> Created Time: Sats 24 Dec 2015 09:52:49 PM CST
 ************************************************************************/

#include"classification_model.h"
#include <cmath>
#include <numeric>
//#include <math.h>

#include <windows.h>

#define NUM_SHOW_STARS_SCOPE 60 //should be even
#define M_PI 3.14159265358979323846 // so strange
#define MIN_DISTANCE_OF_OBJ_OF_TWO_ITERATION 0.01
#define NUM_OF_DIMENSION 2

namespace CVM{
    
    bool ClassificationModel::Init(const std::vector<std::vector<double> > &vec_orginal_mu,const std::vector<std::vector<double> > &vec_original_sig,std::vector<int> vec_normal_data){
		//check parameters->just for the show case, I just generate the normal data of dimension (n)(n=2 in this implementation) and indepent with each other
		if (vec_orginal_mu.size()<0 || vec_original_sig.size()<=0 || vec_orginal_mu.size()!=vec_original_sig.size() || vec_orginal_mu.size()!=vec_normal_data.size() || vec_normal_data.size()!=vec_original_sig.size() ){
			std::cout<<"Bad init parameters"<<std::endl;
			return false;
		}
		//Init
		vec_original_mu_=vec_orginal_mu;
		vec_original_sig_=vec_original_sig;

		//generate data
		int num_center=vec_normal_data.size();
		for (int n=0;n<num_center;++n){
			for (int x=0;x<vec_normal_data[n];++x){
				if(n==0)
				{
					vec_original_w_.push_back(0);
				}
				else
				{
					vec_original_w_.push_back(1);
				}
			}
		}		
		const int nstars_normal=1000;    // maximum number of stars to distribute
		for (int n=0;n<num_center;++n){
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
					std::cout << int(vec_orginal_mu[n][x])-NUM_SHOW_STARS_SCOPE/2+i << "бл" << int(vec_orginal_mu[n][x])-NUM_SHOW_STARS_SCOPE/2+(i+1) << ": ";
					std::cout << std::string(p[i]*nstars_normal/vec_normal_data[n],'.') << std::endl;
				}
				std::cout << "Generate normal_distribution ("<<vec_orginal_mu[n][x]<<","<<vec_original_sig[n][x]<<"):finish"<< std::endl<< std::endl;
			}			
		}	

		/*cv::Mat mCoordinate=cv::Mat::zeros(16*20,16*20,CV_8UC3);
		int num_of_point=vec_original_x_.size();
		for (int i=0;i<num_of_point;++i){
			cv::circle(mCoordinate,cv::Point((vec_original_x_[i]+8)*20,(vec_original_w_[i]+8)*20),1,cv::Scalar(255,255,255),-1,8,0);
		}
		cv::imshow("original data",mCoordinate);
		cv::waitKey();*/
		
		//generate the train data
		int num_of_train_data=vec_original_x_.size();
		vec_original_x_train_.resize(num_of_train_data);
		for (int i=0;i<num_of_train_data;++i){
			vec_original_x_train_[i].push_back(1.0);
			vec_original_x_train_[i].push_back(vec_original_x_[i]);
			vec_original_x_train_[i].push_back(vec_original_y_[i]);
		}
		//generate the test data  -5 -> +5   size:100
		/*vec_original_x_test.resize(100);
		for (int i=0;i<100;++i){
			vec_original_x_test[i].push_back(1);
			vec_original_x_test[i].push_back(-5+0.1*i);
		}*/
		//just the classification is right or not
		vec_original_x_test_=vec_original_x_train_;

		return true;
    }

	//ques 9.2
	void ClassificationModel::BayesianLogisticRegression(const double var_prior,const vector<double> vec_initial_phi){
		//time the time
		DWORD t1,t2;
		t1 = GetTickCount();
		//regression start
		int num_of_train_data=vec_original_x_train_.size();
		int num_of_test_data=vec_original_x_test_.size();
		int num_of_dimension=vec_original_x_train_[0].size()-1;
		// Find the MAP estimate of the parameters phi.
		vector<double> vec_phi=vec_initial_phi;
		NewtonMin(vec_phi,var_prior);
		//Compute the Hessian at phi.
		int size_of_vec_phi=vec_phi.size();
		vector<vector<double> > vec_H;
		vec_H.resize(size_of_vec_phi);
		for (int i=0;i<size_of_vec_phi;++i){
			vec_H[i].resize(size_of_vec_phi);
		}
		for (int i=0;i<size_of_vec_phi;++i){
			for(int j=0;j<size_of_vec_phi;++j){
				if (i==j){
					vec_H[i][j]+=1/var_prior*num_of_train_data;
				}
			}
		}
		for (int i=0;i<num_of_train_data;++i){
			//the w_prediction represents the probability and the label in some significances
			double phi_innerproduct_x_train=0.0;
			for (int j=0;j<size_of_vec_phi;++j)	{
				phi_innerproduct_x_train+=vec_phi[j]*vec_original_x_train_[i][j];
			}
			double w_prediction=Sigmoid(phi_innerproduct_x_train);
			for (int j=0;j<size_of_vec_phi;++j)	{
				for (int k=0;k<size_of_vec_phi;++k){
					vec_H[j][k]-=w_prediction*(1-w_prediction)*vec_original_x_train_[i][j]*vec_original_x_train_[i][k];
				}					
			}
		}
		//Set the mean and variance of the Laplace approximation.
		vector<double> vec_mu=vec_phi;
		vector<vector<double> > vec_var=vec_H;
		Mat_<double> mH=Mat::ones(size_of_vec_phi,size_of_vec_phi,CV_64F);
		for (int i=0;i<size_of_vec_phi;++i){
			for (int j=0;j<size_of_vec_phi;++j)	{
				mH.at<double>(i,j)=vec_H[i][j];
			}				
		}
		mH=mH.inv();
		for (int i=0;i<size_of_vec_phi;++i){
			for (int j=0;j<size_of_vec_phi;++j)	{
				vec_var[i][j]=-1*mH.at<double>(i,j);
			}				
		}
		//Compute mean and variance of the activation.
		vector<double>  vec_mu_a;
		vec_mu_a.resize(num_of_train_data);
		for (int i=0;i<num_of_train_data;++i){
			for (int j=0;j<size_of_vec_phi;++j){
				vec_mu_a[i]+=vec_mu[j]*vec_original_x_train_[i][j];
			}			
		}
		vector<vector<double> >  vec_var_a_temp;
		vec_var_a_temp.resize(num_of_train_data);
		for (int i=0;i<num_of_train_data;++i){
			vec_var_a_temp[i].resize(size_of_vec_phi);
		}
		Mat_<double> m_vec_var_a_temp=Mat::ones(num_of_train_data,size_of_vec_phi,CV_64F);
		Mat_<double> m_vec_original_x_train=Mat::ones(num_of_train_data,size_of_vec_phi,CV_64F);
		for (int i=0;i<num_of_train_data;++i){
			for (int j=0;j<size_of_vec_phi;++j)	{
				m_vec_original_x_train.at<double>(i,j)=vec_original_x_train_[i][j];
			}			
		}		
		Mat_<double> m_vec_var=Mat::ones(size_of_vec_phi,size_of_vec_phi,CV_64F);
		for (int i=0;i<size_of_vec_phi;++i){
			for (int j=0;j<size_of_vec_phi;++j)	{
				m_vec_var.at<double>(i,j)=vec_var[i][j];
			}			
		}
		m_vec_var_a_temp=m_vec_original_x_train*m_vec_var;
		for (int i=0;i<num_of_train_data;++i){
			for (int j=0;j<size_of_vec_phi;++j)	{
				vec_var_a_temp[i][j]=m_vec_var_a_temp.at<double>(i,j);
			}			
		}
		vector<double> vec_var_a;
		vec_var_a.resize(num_of_train_data);
		for(int i=0;i<num_of_train_data;++i){
			for (int j=0;j<size_of_vec_phi;++j){
				vec_var_a[i]+=vec_var_a_temp[i][j]*vec_original_x_train_[i][j];
			}		
		}
		// Approximate the integral to get the Bernoulli parameter
		vector<double> vec_lambda;
		vec_lambda.resize(num_of_train_data);
		for (int i=0;i<num_of_train_data;++i){
			vec_lambda[i]=sqrt(1 + M_PI * vec_var_a[i] / 8);
		}
		for (int i=0;i<num_of_train_data;++i){
			vec_lambda[i]=Sigmoid(vec_mu_a[i]/vec_lambda[i]);
		}
		cout<<"---- Result ----"<<endl;
		for (int i=0;i<num_of_train_data;++i){
			cout<<vec_lambda[i]<<endl;
		}
		

		t2 = GetTickCount();
		printf("Use Time:%f\n",(t2-t1)*1.0/1000);

		return;
	}

	void ClassificationModel::NewtonMin(vector<double> vec_phi,const double var_prior){
		int nIterCount=0;
		vector<double> vec_old_phi=vec_phi;
		int size_of_vec_phi=vec_phi.size();
		int num_of_train_data=vec_original_x_train_.size();
		for (int i=0;i<size_of_vec_phi;++i){
			vec_old_phi[i]-=1;
		}		
		double num_of_distance_of_phi=0.0;
		for (int i=0;i<size_of_vec_phi;++i){
			num_of_distance_of_phi+=pow(vec_phi[i]-vec_old_phi[i],2.0);
		}		
		vector<double> vec_g;
		vec_g.resize(size_of_vec_phi);// num_of_demension+1  , size_of_vec_phi
		vector<vector<double> > vec_H;
		vec_H.resize(size_of_vec_phi);
		for (int i=0;i<size_of_vec_phi;++i){
			vec_H[i].resize(size_of_vec_phi);
		}		
		while(num_of_distance_of_phi>0.001){
			//show the vec_phi
			cout<<"Iter    ->    "<<++nIterCount<<endl;
			cout<<"-- The phi --"<<endl;
			for (int i=0;i<size_of_vec_phi;++i)	{
				cout<<vec_phi[i]<<"  ";
			}
			cout<<endl;		

			if (nIterCount==50)
			{
				break;
			}
			

			//reinit the data
			memset(&*vec_g.begin(),0.0,vec_g.size()*sizeof(double));
			for (int i=0;i<size_of_vec_phi;++i){
				memset(&*vec_H[i].begin(),0.0,vec_H[i].size()*sizeof(double));
			}					
			//prepare for calculate the derivative and second derivative
			for (int i=0;i<size_of_vec_phi;++i){
				vec_g[i]+=vec_phi[i]/var_prior*num_of_train_data;
				for(int j=0;j<size_of_vec_phi;++j){
					if (i==j){
						vec_H[i][j]+=1/var_prior*num_of_train_data;
					}
				}
			}
			//calculate the derivative and second derivative over all the train data
			for (int i=0;i<num_of_train_data;++i){
				//the w_prediction represents the probability and the label in some significances
				double phi_innerproduct_x_train=0.0;
				for (int j=0;j<size_of_vec_phi;++j)	{
					phi_innerproduct_x_train+=vec_phi[j]*vec_original_x_train_[i][j];
				}
				double w_prediction=Sigmoid(phi_innerproduct_x_train);
				//in there, do not calculate the loss L because I do not use it
				// update the vec_g and vec_H
				for (int j=0;j<size_of_vec_phi;++j)	{
					vec_g[j]+=(w_prediction-vec_original_w_[i])*vec_original_x_train_[i][j];
					for (int k=0;k<size_of_vec_phi;++k){
						vec_H[j][k]+=w_prediction*(1-w_prediction)*vec_original_x_train_[i][j]*vec_original_x_train_[i][k];
					}					
				}
			}
			//calculate the inverse of vec_H
			Mat_<double> mH=Mat::ones(size_of_vec_phi,size_of_vec_phi,CV_64F);
			for (int i=0;i<size_of_vec_phi;++i){
				for (int j=0;j<size_of_vec_phi;++j)	{
					mH.at<double>(i,j)=vec_H[i][j];
				}				
			}
			//cout<<mH<<endl;
			mH=mH.inv();
			//cout<<mH<<endl;
			//Mat
			Mat_<double> mg=Mat::ones(size_of_vec_phi,1,CV_64F);
			for (int i=0;i<size_of_vec_phi;++i){
					mg.at<double>(i)=vec_g[i];			
			}
			cout<<mg<<endl;
			mH=mH*mg;			
			//update the vec_phi
			vec_old_phi=vec_phi;
			for (int i=0;i<size_of_vec_phi;++i){
				vec_phi[i]+=-0.1* mH.at<double>(i);
			}
			//recalculate the distance of vec_phi and vec_old_phi
			num_of_distance_of_phi=0.0;
			for (int i=0;i<size_of_vec_phi;++i){
				num_of_distance_of_phi+=pow(vec_phi[i]-vec_old_phi[i],2.0);
			}	
		}
	}

	double ClassificationModel::Sigmoid(const double x){
		return 1/(1+exp(-x));
	}








}//namespace CVM
