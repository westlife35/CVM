/*************************************************************************
	> File Name: prob_model.cc
	> Author: CeQi
	> Mail: qi_ce_0@163.com
	> Created Time: Tu 27 Oct 2015 03:17:48 PM CST
 ************************************************************************/

#include <io.h>
#include "class_gen_model.h"

#define NUM_SHOW_STARS_SCOPE 60 //should be even
#define M_PI 3.14159265358979323846
//5 5 - 54.2453% 6 3 48.0703 6 4 51.8228 6 5 53.8178 6 6 55.8247 6 7 56.6441 6 8 57.1547 6 9 58.6985 2 3 39.0571 6 10 58.5441 6 11 59.8622 6 12 57.9504 6 13 58.8172 6 14 59.8147 7 13 62.6529 7 14 63.0091 7 15 64.4223 9 15 64.6092 15 7 61.507 7 16 64.1017 7 17 63.5554 8 15 61.6316 8 16 10.8182 8 17 10.8182 //最后这两个就算调整了大小也是64点几的准确率
#define  IMAGE_RESIZE_HEIGHT 9//9
#define  IMAGE_RESIZE_WIDTH 15//15
#define  IMAGE_PATCH_HEIGHT 4//20
#define  IMAGE_PATCH_WIDTH 2//10

namespace CVM{

	bool ClassGenModel::Init(const std::string &strTrainImagePath,const int &nPattern){
		//check parameters
		if (nPattern<=1){
			std::cout<<"Bad init parameters"<<std::endl;
		}
		//exclude the \ in the strTrainImagePath
		std::string strTrainImagePathChecked;
		if (strTrainImagePath[strTrainImagePath.length()-1]=='\\'){
			strTrainImagePathChecked=strTrainImagePath.substr(0,strTrainImagePath.length()-1);
		}		
		else{
			strTrainImagePathChecked=strTrainImagePath;
		}
		//Init
		nPattern_=nPattern;
		vec_pattern_count_.resize(nPattern_);
		TrainFilePathAndLabel tempTrainFilePathAndLabel;
		long Handle;
		_finddata_t FileInfo;
		if((Handle=_findfirst( (strTrainImagePathChecked+"\\*.*").data() ,&FileInfo))==-1L)
			printf("The input train data path is not valid\n");
		else{
			do{
				if (FileInfo.attrib & _A_SUBDIR){  
					if( (strcmp(FileInfo.name,".") != 0 ) &&(strcmp(FileInfo.name,"..") != 0)){  
						std::string strSubPath = strTrainImagePathChecked + "\\" + FileInfo.name;  
						tempTrainFilePathAndLabel.label=std::string(FileInfo.name)[0];
						long Handle_1;
						_finddata_t FileInfo_1;
						//if((Handle_1=_findfirst( (strSubPath+"\\*.*").data() ,&FileInfo_1))==-1L)
						if((Handle_1=_findfirst( (strSubPath+"\\*.bmp").data() ,&FileInfo_1))==-1L)// thumbs.bd!!
							printf("The input train data path is not valid\n");
						else{
							do{
								if ( !(FileInfo_1.attrib & _A_SUBDIR) ){    //should be a file
									if( (strcmp(FileInfo_1.name,".") == 0 ) || (strcmp(FileInfo_1.name,"..") == 0)){  
										continue;
									}  
									//vec_train_file_path.push_back(FileInfo_1.name);
									tempTrainFilePathAndLabel.strFilePath=strTrainImagePathChecked + "\\" + FileInfo.name+"\\"+FileInfo_1.name;
									vec_train_file_path_and_label_.push_back(tempTrainFilePathAndLabel);
								}
							} while (_findnext(Handle_1,&FileInfo_1)==0);
							_findclose(Handle_1);
						}
					}  
				}
			} while (_findnext(Handle,&FileInfo)==0);
			_findclose(Handle);
		}
		return true;
	}

	bool ClassGenModel::GetFeature(){
		//check parameter
		if (vec_train_file_path_and_label_.size()==0){
			std::cout<<"The vec_train_file_path_label.size() equals to 0"<<std::endl;
			return false;
		}
		//read all the images and resize
		//std::vector<cv::Mat> vec_train_image;//do not make the train images as the member of class to save memory
		cv::Mat mTempImage;
		int nSize=vec_train_file_path_and_label_.size();
		std::cout<<"------Read images and extract feature : start------"<<std::endl;
		CountTimeStart();
		ZerooneFeatureAndLabel tempZerooneFeatureAndLabel;
		for (int i=0;i<nSize;++i){
			mTempImage=cv::imread(vec_train_file_path_and_label_[i].strFilePath,CV_LOAD_IMAGE_GRAYSCALE);
			if (!mTempImage.data){
				std::cout<<"Read image:["<<vec_train_file_path_and_label_[i].strFilePath<<"] failed"<<std::endl;
			}

			resize(mTempImage,mTempImage,cv::Size(IMAGE_RESIZE_WIDTH,IMAGE_RESIZE_HEIGHT));//the Size of 20*10 is decided by myself
			equalizeHist(mTempImage,mTempImage);
			//vec_train_image.push_back(mTempImage);	
			//directly get the feature of black(0) or white(1)
			tempZerooneFeatureAndLabel.vec_zeroone_feature.swap(std::vector<uchar>());
			for (int y=0;y<IMAGE_RESIZE_HEIGHT;++y){
				const uchar* ptr = mTempImage.ptr(y);
				for (int x=0;x<IMAGE_RESIZE_WIDTH;++x){
					//tempZerooneFeatureAndLabel.vec_zeroone_feature.push_back(*(ptr+x)/32);
					tempZerooneFeatureAndLabel.vec_zeroone_feature.push_back(*(ptr+x));
					//if(*(ptr+x)>128){
					//	tempZerooneFeatureAndLabel.vec_zeroone_feature.push_back(1);//'1'
					//}
					//else{
					//	tempZerooneFeatureAndLabel.vec_zeroone_feature.push_back(0);//'0'
					//}
				}				
			}

			//get feature
			/*tempZerooneFeatureAndLabel.vec_zeroone_feature.swap(std::vector<double>());
			CalcGradientFeat(mTempImage,tempZerooneFeatureAndLabel.vec_zeroone_feature);*/


			tempZerooneFeatureAndLabel.label=vec_train_file_path_and_label_[i].label;
			vec_pattern_count_[tempZerooneFeatureAndLabel.label-'0']++;
			vec_zeroone_feature_and_label_.push_back(tempZerooneFeatureAndLabel);
		}
		std::cout<<"------Read images and extract feature :  finish------"<<std::endl;
		CountTimeStop();
		return false;
	}

	void ClassGenModel::CalcGradientFeat(const cv::Mat& imgSrc, std::vector<double>& feat) 
	{ 
		cv::Mat image=imgSrc; 
		//cvtColor(imgSrc,image,CV_BGR2GRAY); 
		resize(image,image,cv::Size(IMAGE_RESIZE_WIDTH,IMAGE_RESIZE_HEIGHT)); 
		// 计算x方向和y方向上的滤波 
		//float mask[3][3] = { { 1, 2, 1 }, { 0, 0, 0 }, { -1, -2, -1 } };
		float mask[3][3] = { { -1, 0, 1 }, { -2, 0, 2 }, { -1, 0, 1 } };
		cv::Mat y_mask = cv::Mat(3, 3, CV_32F, mask) / 8; 
		cv::Mat x_mask = y_mask.t(); // 转置 
		cv::Mat sobelX, sobelY;
		filter2D(image, sobelX, CV_32F, x_mask); 
		filter2D(image, sobelY, CV_32F, y_mask);
		sobelX = abs(sobelX); 
		sobelY = abs(sobelY);
		float totleValueX = SumMatValue(sobelX); 
		float totleValueY = SumMatValue(sobelY);
		// 将图像划分为4*2共8个格子，计算每个格子里灰度值总和的百分比 
		for (int i = 0; i < image.rows; i = i + IMAGE_PATCH_HEIGHT) 
		{ 
			for (int j = 0; j < image.cols; j = j + IMAGE_PATCH_WIDTH) 
			{ 
				/*cv::Mat subImageX = sobelX(cv::Rect(j, i, 4, 4)); 
				feat.push_back(SumMatValue(subImageX) / totleValueX); 
				cv::Mat subImageY= sobelY(cv::Rect(j, i, 4, 4)); 
				feat.push_back(SumMatValue(subImageY) / totleValueY); */

				cv::Mat subImageX = sobelX(cv::Rect(j, i, IMAGE_PATCH_WIDTH, IMAGE_PATCH_HEIGHT)); 
				feat.push_back(SumMatValue(subImageX) / totleValueX); 
				cv::Mat subImageY= sobelY(cv::Rect(j, i, IMAGE_PATCH_WIDTH, IMAGE_PATCH_HEIGHT)); 
				feat.push_back(SumMatValue(subImageY) / totleValueY); 

				//cv::Mat subImageX = sobelX(cv::Rect(j, i, 4, 4)); 
				////feat.push_back(SumMatValue(subImageX) / totleValueX); 
				//cv::Mat subImageY= sobelY(cv::Rect(j, i, 4, 4)); 
				//feat.push_back(SumMatValue(subImageY)+SumMatValue(subImageX) / totleValueY); 
			} 
		} 
	}

	double ClassGenModel::SumMatValue(const cv::Mat& image) 
	{ 
		double sumValue = 0; 
		int r = image.rows; 
		int c = image.cols; 
		if (image.isContinuous()) 
		{ 
			c = r*c; 
			r = 1;    
		} 
		for (int i = 0; i < r; i++) 
		{ 
			const uchar* linePtr = image.ptr<uchar>(i); 
			for (int j = 0; j < c; j++) 
			{ 
				sumValue += linePtr[j]; 
			} 
		} 
		return sumValue; 
	}

	void ClassGenModel::BasicGenNorm(){
		std::cout<<"------Training :  start------"<<std::endl;
		CountTimeStart();
		//reserve the memory
		int nNumOfTrainData = vec_zeroone_feature_and_label_.size();
		int nDim=vec_zeroone_feature_and_label_[0].vec_zeroone_feature.size();
		vec_mu.resize(nPattern_);
		vec_sig.resize(nPattern_);
		for (int i=0;i<nPattern_;++i){
			vec_mu[i].resize(nDim);
			vec_sig[i].resize(nDim);
			for(int j=0;j<nDim;++j){
				vec_sig[i][j].resize(nDim);
			}
		}
		//computer the vec_mu
		for (int i=0;i<nNumOfTrainData;++i){
			for (int j=0;j<nDim;++j){
				vec_mu[vec_zeroone_feature_and_label_[i].label-'0'][j]+=vec_zeroone_feature_and_label_[i].vec_zeroone_feature[j];  
			}			
		}
		//divide nCount
		for (int i=0;i<nPattern_;++i){
			int nCount=vec_pattern_count_[i];
			for (int j=0;j<nDim;++j){
				vec_mu[i][j]/=nCount;
	        }			
		}
		//computer the vec_sig
		std::vector<ZerooneFeatureAndLabel> vec_zeroone_feature_and_label=vec_zeroone_feature_and_label_;
		for (int i=0;i<nNumOfTrainData;++i){
			for (int j=0;j<nDim;++j){
				//subtract the mu(average)
				vec_zeroone_feature_and_label[i].vec_zeroone_feature[j]-=vec_mu[vec_zeroone_feature_and_label[i].label-'0'][j];				
			}
			//add the result to the covariance matrix
			for (int x=0;x<nDim;++x){
				for (int y=0;y<nDim;++y){
					vec_sig[vec_zeroone_feature_and_label[i].label-'0'][x][y]+=vec_zeroone_feature_and_label[i].vec_zeroone_feature[x]*vec_zeroone_feature_and_label[i].vec_zeroone_feature[y];
				}				
			}			
		}
		//divide nCount
		for (int i=0;i<nPattern_;++i){
			int nCount=vec_pattern_count_[i];
			for (int x=0;x<nDim;++x){
				for (int y=0;y<nDim;++y){
					vec_sig[i][x][y]/=nCount;
				}				
			}			
		}
		std::cout<<"------Training :  finish------"<<std::endl;
		CountTimeStop();
		std::cout<<std::endl;
		std::cout<<"----<-The training accuracy->------"<<std::endl;
		//use original feature
		ShowAccuracy(vec_zeroone_feature_and_label_);
		std::cout<<"----<-The training accuracy->------"<<std::endl;
	}

	void ClassGenModel::CountTimeStart(){
		QueryPerformanceFrequency(&tc_);
		QueryPerformanceCounter(&tStart_);
	}

	void ClassGenModel::CountTimeStop(){
		QueryPerformanceCounter(&tStop_);
		printf("Use Time:%f\n",(tStop_.QuadPart - tStart_.QuadPart)*1.0/tc_.QuadPart);
	}

	void ClassGenModel::ShowAccuracy(const std::vector<ZerooneFeatureAndLabel> &vec_zeroone_feature_and_label){
		//reserve memory 
		std::vector<std::vector<double> > vec_likelihoods;
		vec_likelihoods.resize(nPattern_);
		int nNumOfTestData;
		nNumOfTestData=vec_zeroone_feature_and_label.size();
		for (int i=0;i<nPattern_;++i){
			vec_likelihoods[i].resize(nNumOfTestData);
		}
		//Compute likelihoods for each class for the test data.
		for (int i=0;i<nPattern_;++i){
			for (int j=0;j<nNumOfTestData;++j){
				vec_likelihoods[i][j]=MvnPdf(vec_zeroone_feature_and_label[j].vec_zeroone_feature,vec_mu[i],vec_sig[i]);
			}			
		}
		//Classify the data with Bayes' rule.
		std::vector<std::vector<double> > vec_posterior;
		vec_posterior.resize(nPattern_);
		for (int i=0;i<nPattern_;++i){
			vec_posterior[i].resize(nNumOfTestData);
		}		
		std::vector<char> vec_posterior_pattern_result;
		vec_posterior_pattern_result.resize(nNumOfTestData);
		int nNumOfTrainData=vec_zeroone_feature_and_label_.size();
		for (int i=0;i<nNumOfTestData;++i){
			double dSumBayes=0.0;
			int nMaxNo=0;
			double dMaxProb=0.0;
			for (int j=0;j<nPattern_;++j){
				dSumBayes+=vec_likelihoods[j][i]*vec_pattern_count_[j]/nNumOfTrainData;
			}	
			for (int j=0;j<nPattern_;++j){
				vec_posterior[j][i]=(vec_likelihoods[j][i]*vec_pattern_count_[j]/nNumOfTrainData) /dSumBayes;
				if (dMaxProb<vec_posterior[j][i]){
					dMaxProb=vec_posterior[j][i];
					nMaxNo=j;
				}				
			}
			//judge the data's result(pattern)
			vec_posterior_pattern_result[i]=nMaxNo;
		}
		//calculate the accuracy
		int nRightCount=0;
		for (int i=0;i<nNumOfTestData;++i){
			if ((int)vec_posterior_pattern_result[i]==int(vec_zeroone_feature_and_label[i].label-'0') ){
				nRightCount++;
			}			
		}
		std::cout<<double(nRightCount)/nNumOfTestData*100<<"%"<<std::endl;
	}    

	double ClassGenModel::MvnPdf(const std::vector<uchar> &vec_feature,const std::vector<double> &vec_mu,const std::vector< std::vector<double> > &vec_sig){
	//double ClassGenModel::MvnPdf(const std::vector<double> &vec_feature,const std::vector<double> &vec_mu,const std::vector< std::vector<double> > &vec_sig){
		int nDim=vec_mu.size();
		if (nDim<=0){
			std::cout<<"Error: the dimension of the input of MvnPdf is less than 0."<<std::endl;
			return 0.0;
		}
		//calculate the result
		double dResult=1.0;
		for (int i=0;i<nDim;++i){
			//dResult*=exp(-0.5*(vec_feature[i]-vec_mu[i])*(vec_feature[i]-vec_mu[i])/vec_sig[i][i])/sqrt(2*M_PI*vec_sig[i][i]);
			dResult*=exp(-0.5*(vec_feature[i]-vec_mu[i])*(vec_feature[i]-vec_mu[i])/vec_sig[i][i])/sqrt(vec_sig[i][i]);
			//dResult*=exp( -0.5*(vec_feature[i]-vec_mu[i])*(vec_feature[i]-vec_mu[i])/(vec_sig[i][i]+0.1) )/sqrt(vec_sig[i][i]+0.1);
		}
		dResult*=pow(2*M_PI,nDim/2);
		
		return dResult;




		//int nDim=vec_mu.size();
		//if (nDim<=0){
		//	std::cout<<"Error: the dimension of the input of MvnPdf is less than 0."<<std::endl;
		//	return 0.0;
		//}
		////calculate numerator
		//std::vector<uchar> vec_feature_subtract_mu=vec_feature;
		//for (int i=0;i<nDim;++i){
		//	vec_feature_subtract_mu[i]-=vec_mu[i];
		//}
		//std::vector< std::vector<double> > vec_sig_inverse=vec_sig;
		//CalculateMatrixInverse(vec_sig,vec_sig_inverse,nDim);
		//std::vector<double> vec_intermidiate;
		//vec_intermidiate.resize(nDim);
		//for (int i=0;i<nDim;++i){
		//	for (int j=0;j<nDim;++j){
		//		vec_intermidiate[i]+=vec_feature_subtract_mu[j]*vec_sig_inverse[j][i];
		//	}			
		//}
		//double numerator=0.0;
		//for (int i=0;i<nDim;++i){
		//	numerator+=vec_intermidiate[i]*vec_feature_subtract_mu[i];
		//}
		//double dResult=exp(-0.5*numerator);
		////calculate denominator
		//std::vector< std::vector<double> > vec_sig_double;
		//vec_sig_double.resize(nDim);
		//for (int i=0;i<nDim;++i){
		//	vec_sig_double[i].resize(nDim);
		//	for (int j=0;j<nDim;++j){
		//		vec_sig_double[i][j]=double(vec_sig[i][j]);
		//	}			
		//}
		//double debug=CalculateMatrixModule(vec_sig_double,nDim);
		//double dDenominator=pow(2*M_PI,nDim/2)  *  sqrt( CalculateMatrixModule(vec_sig_double,nDim) ) ;
		////calculate result
		//dResult/=dDenominator;
		//return dResult;
	}

	double ClassGenModel::CalculateMatrixModule(const std::vector<std::vector<double>> &vec_matrix,const int &nDim){
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
			for(int j =0;j < nDim-1;++j){  
				for(int k=0;k<nDim-1;++k){  
					x = j + 1;  
					y = k >= i ? k + 1 : k;  
					vec_matrix_temp[j][k] = vec_matrix[x][y];  
				}  
			} 
			t = CalculateMatrixModule(vec_matrix_temp,nDim-1);
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

	bool ClassGenModel::CalculateMatrixInverse(const std::vector<std::vector<double>> &vec_matrix_in,std::vector<std::vector<double>> &vec_matrix_out,const int &nDim){  
		double A = CalculateMatrixModule(vec_matrix_in,nDim);
		if(fabs(A-0)<=0.000000001){  
			printf("Error: there is no inverse matrix.\n");  
			return false;  
		}
		std::vector<std::vector<double>> vec_matrix_adjoint;
		CalculateMatrixAdjoint(vec_matrix_in, vec_matrix_adjoint,nDim);    
		for(int i=0;i<nDim;++i){    
			for(int j=0;j<nDim;++j){    
				vec_matrix_out[i][j] = (double)(vec_matrix_adjoint[i][j]/A);  
			}    
		}  
		return true;  
	}

	void ClassGenModel::CalculateMatrixAdjoint(const std::vector<std::vector<double>> &vec_matrix_in,std::vector<std::vector<double>> &vec_matrix_out,const int &nDim){
		int x, y;  
		std::vector<std::vector<double>> vec_matrix_temp;
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
