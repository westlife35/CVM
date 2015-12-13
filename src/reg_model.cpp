/*************************************************************************
	> File Name: reg_model.cpp
	> Author: CeQi
	> Mail: qi_ce_0@163.com
	> Created Time: Tus 24 Oct 2015 09:52:49 PM CST
 ************************************************************************/

#include"reg_model.h"
#include <cmath>
#include <numeric>
//#include <math.h>

#include <windows.h>

#define NUM_SHOW_STARS_SCOPE 60 //should be even
#define M_PI 3.14159265358979323846 // so strange
#define MIN_DISTANCE_OF_OBJ_OF_TWO_ITERATION 0.01
#define NUM_OF_DIMENSION 2

namespace CVM{
    
    bool RegModel::Init(const std::vector<std::vector<double> > &vec_orginal_mu,const std::vector<std::vector<double> > &vec_original_sig,std::vector<int> vec_normal_data){
		//test 
		/*std::vector<std::vector<double> > vec_t;
		int nDim=9;
		vec_t.resize(nDim);
		for (int i=0;i<nDim;++i){
			vec_t[i].resize(nDim);
		}
		std::cout<<"-------------"<<std::endl;
		double de=CalculateMatrixModule(vec_t,nDim);
		std::cout<<"-------------"<<std::endl;*/



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
		const int nstars_normal=1000;    // maximum number of stars to distribute
		const int nstars_categorical=200;    // maximum number of stars to distribute
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
						vec_original_w_.push_back(number);
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

		/*cv::Mat mCoordinate=cv::Mat::zeros(16*20,16*20,CV_8UC3);
		int num_of_point=vec_original_x_.size();
		for (int i=0;i<num_of_point;++i){
			cv::circle(mCoordinate,cv::Point((vec_original_x_[i]+8)*20,(vec_original_w_[i]+8)*20),1,cv::Scalar(255,255,255),-1,8,0);
		}
		cv::imshow("original data",mCoordinate);
		cv::waitKey();*/
		
		//generate the train data
		int num_of_train_data=vec_original_x_.size();
		vec_original_x_train.resize(num_of_train_data);
		for (int i=0;i<num_of_train_data;++i){
			vec_original_x_train[i].push_back(1.0);
			vec_original_x_train[i].push_back(vec_original_x_[i]);
		}
		//generate the test data  -5 -> +5   size:100
		/*vec_original_x_test.resize(100);
		for (int i=0;i<100;++i){
			vec_original_x_test[i].push_back(1);
			vec_original_x_test[i].push_back(-5+0.1*i);
		}*/
		vec_original_x_test=vec_original_x_train;

		return true;
    }

	//ques 8.3
	void RegModel::GaussianProcessRegression(const double var_prior,const double lambda){
		//time the time
		DWORD t1,t2;
		t1 = GetTickCount();
		//regression start
		int num_of_train_data=vec_original_x_train.size();
		int num_of_test_data=vec_original_x_test.size();
		//compute  K[X,X]
		std::vector<std::vector<double> > vec_K;
		vec_K.resize(num_of_train_data);
		for (int i=0;i<num_of_train_data;++i){
			vec_K[i].resize(num_of_train_data);
		}		
		for (int i=0;i<num_of_train_data;++i){
			for (int j=0;j<num_of_train_data;++j){
				//just do not do the judgment
				KernelGauss(vec_original_x_train[i],vec_original_x_train[j],lambda,vec_K[i][j]);
			}
		}
		//Compute K[X_test,X].
		std::vector<std::vector<double> > vec_K_test;
		vec_K_test.resize(num_of_test_data);
		for (int i=0;i<num_of_test_data;++i){
			vec_K_test[i].resize(num_of_train_data);
		}		
		for (int i=0;i<num_of_test_data;++i){
			for (int j=0;j<num_of_train_data;++j){
				//just do not do the judgment
				KernelGauss(vec_original_x_test[i],vec_original_x_train[j],lambda,vec_K_test[i][j]);
			}
		}
		//Compute the variance. Use the range [0,variance of world values].
		double mu_world=accumulate(vec_original_w_.begin(),vec_original_w_.end(),0.0)/num_of_train_data;
		std::vector<double> vec_temp=vec_original_w_;
		for (int i=0;i<num_of_train_data;++i){
			vec_temp[i]=pow(vec_temp[i]-mu_world,2.0);
		}
		double var_world=accumulate(vec_temp.begin(),vec_temp.end(),0.0)/num_of_train_data;
		//Golden Div Search
		double var=GoldenDivSearch(0,var_world,vec_K,var_prior);
		//Compute A_inv.
		/*std::vector<std::vector<double> > vec_A_inv;
		std::vector<std::vector<double> > vec_temp_K=vec_K;
		for (int i=0;i<num_of_train_data;++i){
			for (int j=0;j<num_of_train_data;++j){
				vec_temp_K[i][j]+=(i==j?1:0)*var/var_prior;
			}			
		}
		CalculateMatrixInverse(vec_temp_K,vec_A_inv,num_of_train_data);*/
		std::vector<std::vector<double> > vec_A_inv=vec_K;
		for (int i=0;i<num_of_train_data;++i){
			for (int j=0;j<num_of_train_data;++j){
				vec_A_inv[i][j]+=(i==j?1:0)*var/var_prior;
			}			
		}
		//下边的这句话因为第一个变量被设置成了常引用，第二个是引用，不知道会不会出现内存问题
		CalculateMatrixInverse(vec_A_inv,vec_A_inv,num_of_train_data);
		//Compute the mean for each test example
		//mu_temp_1
		std::vector<double> vec_mu_temp_1;
		vec_mu_temp_1.resize(num_of_test_data);
		for (int i=0;i<num_of_test_data;++i){
			for (int j=0;j<num_of_train_data;++j){
				vec_mu_temp_1[i]+=vec_K_test[i][j]*vec_original_w_[j];
			}			
		}
		//K_test_A_inv
		std::vector<std::vector<double> > vec_K_test_A_inv;
		vec_K_test_A_inv.resize(num_of_test_data);
		for (int i=0;i<num_of_test_data;++i){
			vec_K_test_A_inv[i].resize(num_of_train_data);
		}
		for (int i=0;i<num_of_test_data;++i){
			for (int j=0;j<num_of_train_data;++j){
				for (int k=0;k<num_of_train_data;++k){
					vec_K_test_A_inv[i][j]+=vec_K_test[i][k]*vec_A_inv[k][j];
				}				
			}			
		}
		//mu_temp_2
		std::vector<double> vec_mu_temp_2;
		vec_mu_temp_2.resize(num_of_test_data);
		std::vector<std::vector<double> > vec_temp_K_test_A_inv_same_size;
		vec_temp_K_test_A_inv_same_size.resize(num_of_test_data);
		for (int i=0;i<num_of_test_data;++i){
			vec_temp_K_test_A_inv_same_size[i].resize(num_of_train_data);
		}
		for (int i=0;i<num_of_test_data;++i){
			for (int j=0;j<num_of_train_data;++j){
				for (int k=0;k<num_of_train_data;++k){
					vec_temp_K_test_A_inv_same_size[i][j]+=vec_K_test_A_inv[i][k]*vec_K[k][j];
				}				
			}			
		}
		for (int i=0;i<num_of_test_data;++i){
			for (int j=0;j<num_of_train_data;++j){
				vec_mu_temp_2[i]+=vec_temp_K_test_A_inv_same_size[i][j]*vec_original_w_[j];
			}			
		}
		// c
		double c=var_prior/var;
		//mu_test
		std::vector<double> vec_mu_test=vec_mu_temp_1;
		for (int i=0;i<num_of_test_data;++i){
			vec_mu_test[i]=c*(vec_mu_test[i]-vec_mu_temp_2[i]);
		}
		//Compute the variance for each test example
		std::vector<double> vec_var_test(num_of_test_data,var);
		for (int i=0;i<num_of_test_data;++i){
			double part1;
			KernelGauss(vec_original_x_test[i],vec_original_x_test[i],lambda,part1);
			double part2=0.0;
			for (int j=0;j<num_of_train_data;++j){
				part2+=vec_K_test_A_inv[i][j]*vec_K_test[i][j];
			}
			vec_var_test[i]+=var_prior * (part1 - part2);			
		}

		//show the result
		/*cv::Mat mCoordinate=cv::Mat::zeros(16*20,16*20,CV_8UC3);
		for (int i=0;i<num_of_train_data;++i){
			cv::circle(mCoordinate,cv::Point((vec_original_x_[i]+8)*20,(vec_original_w_[i]+8)*20),1,cv::Scalar(255,255,255),-1,8,0);
		}
		cv::imwrite("D:\\program\\CVM\\CVM\\Results\\originalData.jpg",mCoordinate);
		std::vector<std::vector<double> > vec_w_test;
		vec_w_test.resize(num_of_test_data);
		for (int i=0;i<num_of_test_data;++i){
			vec_w_test[i].resize(num_of_test_data);
		}
		double maxPixelValue=0.0;
		double minPixelValue=100.0;
		for (int j=0;j<num_of_test_data;++j){
			double mu=vec_mu_test[j];
			for (int i=0;i<num_of_test_data;++i){
				double ww=vec_original_x_test[i][1];
				vec_w_test[i][j]=GetNormalValue(ww,mu,vec_var_test[j]);
				if (vec_w_test[i][j]>maxPixelValue){
					maxPixelValue=vec_w_test[i][j];
					continue;
				}
				if (vec_w_test[i][j]<minPixelValue){
					minPixelValue=vec_w_test[i][j];
				}
			}
		}
		for (int j=0;j<num_of_test_data;++j){
			for (int i=0;i<num_of_test_data;++i){
				if ((vec_w_test[i][j]-minPixelValue)/(maxPixelValue-minPixelValue)*255<0 || (vec_w_test[i][j]-minPixelValue)/(maxPixelValue-minPixelValue)*255>255)
				{
					int a=0;
				}
				
				cv::circle(mCoordinate,cv::Point((vec_original_x_test[i][1]+8)*20,(vec_original_x_test[j][1]+8)*20),1,cv::Scalar(0,(vec_w_test[i][j]-minPixelValue)/(maxPixelValue-minPixelValue)*255,0),-1,8,0);
			}
		}
		


		cv::imwrite("D:\\program\\CVM\\CVM\\Results\\ResultData.jpg",mCoordinate);*/



		t2 = GetTickCount();
		printf("Use Time:%f\n",(t2-t1)*1.0/1000);

		return;
	}

	double RegModel::GetNormalValue(const double &x,const double &mu,const double &sig_square){
		return ( exp( -1*pow(x-mu,2.0)/(2*sig_square)) ) / (sqrt(sig_square)*sqrt(2*M_PI));
	}

	bool RegModel::KernelGauss(const std::vector<double> &vec_x_i,const std::vector<double> &vec_x_j,const double &lambda,double &result){
		if (vec_x_i.size()!=vec_x_j.size()){
			std::cout<<"the input data of function KernelGauss is not in the same size"<<std::endl;
			return false;
		}		
		//calculation
		std::vector<double> vec_x_diff;
		int size=vec_x_i.size();
		vec_x_diff.resize(size);
		for (int i=0;i<size;++i){
			vec_x_diff[i]=vec_x_i[i]-vec_x_j[i];
		}
		double resultTemp=0.0;
		for (int i=0;i<size;++i){
			resultTemp+=vec_x_diff[i]*vec_x_diff[i];
		}
		resultTemp=(-0.5)*resultTemp/pow(lambda,2.0);
		resultTemp=exp(resultTemp);
		result=resultTemp;

		return true;
	}

	double RegModel::GoldenDivSearch(const double &under_boundary,const double &upper_boundary,const std::vector<std::vector<double> > &vec_K,const double &var_prior){
		if (under_boundary>=upper_boundary){
			std::cout<<"wrong search scope in function : GoldenDivSearch()"<<std::endl;
			return 0.0;
		}		
		double b=upper_boundary;
		double a=under_boundary;
		double epsilon=0.0000001;
		double x2=under_boundary+0.618*(b-a);
		double f2=var_log_fun(x2,vec_K,var_prior);
		double x1=under_boundary+0.382*(b-a);
		double f1=var_log_fun(x1,vec_K,var_prior);
		int nFlag=1;
		while( abs(b-a)>epsilon ){
			if (f1<f2){
				b=x2;x2=x1;f2=f1;
				x1=a+0.382*(b-a);
				f1=var_log_fun(x1,vec_K,var_prior);
			} 
			else if (f1==f2){
				a=x1;b=x2;
				x2=a+0.618*(b-a);
				f2=var_log_fun(x2,vec_K,var_prior);
				x1=a+0.382*(b-a);
				f1=var_log_fun(x1,vec_K,var_prior);
			} 
			else{
				a=x1;x1=x2;f1=f2;
				x2=a+0.618*(b-a);
				f2=var_log_fun(x2,vec_K,var_prior);
			}			
			std::cout<<"GoldenDivSearch: "<<nFlag<<std::endl;
			std::cout<<"a: "<<a<<std::endl;
			std::cout<<"b: "<<b<<std::endl;
			nFlag++;
		}
		double x_best=(a+b)/2;

		return x_best;
	}

	double RegModel::var_log_fun(const double &var,const std::vector<std::vector<double> > &vec_K,const double &var_prior){
		int num_of_label=vec_original_w_.size();
		std::vector<std::vector<double> > vec_covariance;
		vec_covariance.resize(num_of_label);
		for (int i=0;i<num_of_label;++i){
			vec_covariance[i].resize(num_of_label);
		}
		for (int i=0;i<num_of_label;++i){
			for (int j=0;j<num_of_label;++j){
				vec_covariance[i][j]=var_prior*vec_K[i][j]+(i==j?1:0)*abs(var);
			}
		}
		//prepare the zero average value vector for next line code
		std::vector<double> vec_mu(num_of_label,0);		
		double f=MvnPdf(vec_original_w_,vec_mu,vec_covariance);
		f=-log(f);

		return f;
	}

	double RegModel::MvnPdf(const std::vector<double> &vec_w,const std::vector<double> &vec_mu,const std::vector<std::vector<double> > &vec_covariance){
		int nDim=vec_w.size();
		if (nDim<=0){
			std::cout<<"Error: the dimension of the input of MvnPdf is less than 0."<<std::endl;
			return 0.0;
		}
		//calculate numerator
		std::vector<double> vec_feature_subtract_mu=vec_w;
		for (int i=0;i<nDim;++i){
			vec_feature_subtract_mu[i]-=vec_mu[i];
		}
		std::vector< std::vector<double> > vec_covariance_inverse=vec_covariance;
		CalculateMatrixInverse(vec_covariance,vec_covariance_inverse,nDim);
		std::vector<double> vec_intermidiate;
		vec_intermidiate.resize(nDim);
		for (int i=0;i<nDim;++i){
			for (int j=0;j<nDim;++j){
				vec_intermidiate[i]+=vec_feature_subtract_mu[j]*vec_covariance_inverse[j][i];
			}			
		}
		double numerator=0.0;
		for (int i=0;i<nDim;++i){
			numerator+=vec_intermidiate[i]*vec_feature_subtract_mu[i];
		}
		double dResult=exp(-0.5*numerator);
		//calculate denominator
		std::vector< std::vector<double> > vec_sig_double;
		vec_sig_double.resize(nDim);
		for (int i=0;i<nDim;++i){
			vec_sig_double[i].resize(nDim);
			for (int j=0;j<nDim;++j){
				vec_sig_double[i][j]=double(vec_covariance[i][j]);
			}			
		}
		double debug=CalculateMatrixModule(vec_sig_double,nDim);
		double dDenominator=pow(2*M_PI,nDim/2)  *  sqrt( CalculateMatrixModule(vec_sig_double,nDim) ) ;
		//calculate result
		dResult/=dDenominator;

		return dResult;
	}

	double RegModel::CalculateMatrixModule(const std::vector<std::vector<double>> &vec_matrix,const int &nDim){		
		std::vector<std::vector<double>> vec_matrix_temp=vec_matrix;
		int x,y;  
		double t;  
		double result = 0.0;  
		//std::cout<<nDim<<std::endl;
		/*if(nDim == 1){  
		return vec_matrix[0][0];  
		}*/
		if(nDim == 2){  
			return vec_matrix[1][1]*vec_matrix[0][0]-vec_matrix[0][1]*vec_matrix[1][0];  			
		}
		//spread the first row
		for(int i=0;i <nDim;++i){  
			//add a zero judgement to speed up
			if (vec_matrix[0][i]>1e-6){
				for(int j =0;j < nDim-1;++j){  
					for(int k=0;k<nDim-1;++k){  
						x = j + 1;  
						y = k >= i ? k + 1 : k;  
						vec_matrix_temp[j][k] = vec_matrix[x][y];  
					}  
				} 
				t = CalculateMatrixModule(vec_matrix_temp,nDim-1);
			}
			else{
				t=0;
			}			
			if(i%2==0){  
				result += vec_matrix[0][i] * t;  
				//result += vec_matrix[i][0] * t;  
			}  
			else {  
				result -= vec_matrix[0][i] * t;  
				//result -= vec_matrix[i][0] * t;  
			}  
		}
		return result;  
	}

	bool RegModel::CalculateMatrixInverse(const std::vector<std::vector<double>> &vec_matrix_in,std::vector<std::vector<double>> &vec_matrix_out,const int &nDim){  
		double A = CalculateMatrixModule(vec_matrix_in,nDim);
		if(fabs(A-0)<=0.000000001){  
			printf("Error: there is no inverse matrix.\n");  
			return false;  
		}
		std::vector<std::vector<double>> vec_matrix_adjoint=vec_matrix_in;
		CalculateMatrixAdjoint(vec_matrix_in, vec_matrix_adjoint,nDim);    
		for(int i=0;i<nDim;++i){    
			for(int j=0;j<nDim;++j){    
				vec_matrix_out[i][j] = (double)(vec_matrix_adjoint[i][j]/A);  
			}    
		}  
		return true;  
	}

	void RegModel::CalculateMatrixAdjoint(const std::vector<std::vector<double>> &vec_matrix_in,std::vector<std::vector<double>> &vec_matrix_out,const int &nDim){
		int x, y;  
		std::vector<std::vector<double>> vec_matrix_temp=vec_matrix_in;
		if(nDim==1){  
			vec_matrix_out[0][0] = 1;  
			return;
		}
		for(int i=0;i<nDim;++i){  
			for(int j=0;j<nDim;++j){  
				for(int k=0;k<nDim-1;++k){  
					for(int t = 0;t<nDim-1;++t){  
						x = k >= i ? k + 1 : k;  
						y = t >= j ? t + 1 : t;
						vec_matrix_temp[k][t] = vec_matrix_in[x][y];  
					}
				} 
				vec_matrix_out[j][i]  =  CalculateMatrixModule(vec_matrix_temp,nDim-1);
				if((i+j)%2==1){  
					vec_matrix_out[j][i] = -1*vec_matrix_out[j][i];
				}  
			}  
		} 
	}




}//namespace CVM
