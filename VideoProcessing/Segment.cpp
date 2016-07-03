#include<iostream>
#include<fstream>
#include<sstream>
#include<time.h>
#include<math.h>

#include "opencv2/highgui/highgui.hpp"
#include <io.h>
#include <cv.h>
#include <ml.h>
using namespace std;
using namespace cv;

void svm_train(vector<vector<double>>& vfeatures, vector<int>& vlabels, string model_file);
void svm_test(vector<vector<double>>& vfeatures, vector<int>& vlabels, string model_file);
string decide(int classNo);

void saveFeature(vector<vector<double>>& features, vector<int>& labels, string featureFile);
void loadFeature(vector<vector<double>>& features, vector<int>& labels, string featureFile);
void split(string line, string separator, vector<double>& data);
void combineData(vector<vector<double>>& feature, vector<vector<double>>& feature1, vector<vector<double>>& feature2, vector<vector<double>>& feature3);
void dataToLibSVM(vector<vector<double>>& vfeatures, vector<int>& vlabels, string outputFile);

void extractFeature(string srcPath, vector<vector<double>>& features, vector<int>& labels, bool save_immediate_result, string mhiDir, string adiDir, string ddiDir);

void extractFeatureFromHBB(IplImage* ddiImg, vector<double>& feature, int start);
void box(IplImage* img, int horPatches, int verPatches, vector<int>& win);
int max(IplImage* img, int x, int y, int horSize, int verSize);
int min(IplImage* img, int x, int y, int horSize, int verSize);

void extractFeatureFromHu(IplImage* mhiImg, IplImage* adiImg, vector<double>& feature, int start);
void HuMoment(IplImage* img, vector<double>& feature);
long long getMpq(IplImage* img,int p, int q);
long long getUpq(IplImage* img, double xbar, double ybar, int p, int q);

void initialize(IplImage *img, int value);
void binaryProcessing(IplImage *img, int value);
void getMHI(string srcPath, string srcFile, IplImage *mhiImg, bool isSave, string destFile);
void getADI(string srcPath, string srcFile, IplImage *adiImg, bool isSave, string destFile);
void getDDI(string srcPath, string srcFile, IplImage *ddiImg, bool isSave, string destFile);

void getAllFrame(string srcPath, string destPath);
void cutAllVideos(string srcPath, string destPath);
void videoCutFrame(string filepath, string filename, string destDir);
vector<string> getAllFiles(string path);

#define SAVE_IMMEDIATE_RESULTS false

int main( int argc, char** argv ){
	//All dataset
	string path="..\\Datasets\\Video_All";

	//Dataset has been splited Training set and Testing set
	string trainPath="..\\Datasets\\Videos_Split\\Train";
	string testPath="..\\Datasets\\Videos_Split\\Test";

	//immediate file output directory
	bool save_immediate_result=false;               //whether save immediate result files
	string mhiDir="..\\Datasets\\ImmediateResults\\MHI";
	string adiDir="..\\Datasets\\ImmediateResults\\ADI";
	string ddiDir="..\\Datasets\\ImmediateResults\\DDI";
/*****************************cut all videos into frames**********************************************/
	string destPath="..\\Datasets\\Frames";
	getAllFrame(path, destPath);
/*****************************************************************************************************/
	
	vector<vector<double>> train_features, test_features;
	vector<int> train_labels, test_labels;
/*******************************Using 122 features****************************************************/
	
	string train_122="..\\Datasets\\Features\\Split\\train_122.data";
	string test_122="..\\Datasets\\Features\\Split\\test_122.data";
	string model_122="..\\Datasets\\Features\\Split\\model_122.xml";

	//Extract trainging set features and labels
	//extractFeature(trainPath, train_features, train_labels, save_immediate_result, mhiDir, adiDir, ddiDir);
	//saveFeature(train_features, train_labels, train_122);

	//Extract testing set features and labels
	//extractFeature(testPath, test_features, test_labels, save_immediate_result, mhiDir, adiDir, ddiDir);
	//saveFeature(test_features, test_labels, test_122);

	//Training
	//loadFeature(train_features, train_labels, train_122);
	//svm_train(train_features, train_labels, model_122);

	//Testing 
	//loadFeature(test_features, test_labels, test_122);
	//svm_test(test_features, test_labels, model_122);

	//dataToLibSVM(train_features, train_labels, "..\\Datasets\\Features\\Split\\train_122");
	//dataToLibSVM(test_features, test_labels, "..\\Datasets\\Features\\Split\\test_122");
/*******************************Using 108 features****************************************************/
	
	string train_108="..\\Datasets\\Features\\Split\\train_108.data";
	string test_108="..\\Datasets\\Features\\Split\\test_108.data";
	string model_108="..\\Datasets\\Features\\Split\\model_108.xml";
	//Extract trainging set features and labels
	//extractFeature(trainPath, train_features, train_labels, save_immediate_result, mhiDir, adiDir, ddiDir);
	//saveFeature(train_features, train_labels, train_108);

	//Extract testing set features and labels
	//extractFeature(testPath, test_features, test_labels, save_immediate_result, mhiDir, adiDir, ddiDir);
	//saveFeature(test_features, test_labels, test_108);

	//Training
	//loadFeature(train_features, train_labels, train_108);
	//svm_train(train_features, train_labels, model_108);

	//Testing 
	//loadFeature(test_features, test_labels, test_108);
	//svm_test(test_features, test_labels, model_108);
	
	//dataToLibSVM(train_features, train_labels, "..\\Datasets\\Features\\Split\\train_108");
/*******************************Using 14 features****************************************************/
	
	string train_14="..\\Datasets\\Features\\Split\\train_14.data";
	string test_14="..\\Datasets\\Features\\Split\\test_14.data";
	string model_14="..\\Datasets\\Features\\Split\\model_14.xml";
	//Extract trainging set features and labels
	//extractFeature(trainPath, train_features, train_labels, save_immediate_result, mhiDir, adiDir, ddiDir);
	//saveFeature(train_features, train_labels, train_14);

	//Extract testing set features and labels
	//extractFeature(testPath, test_features, test_labels, save_immediate_result, mhiDir, adiDir, ddiDir);
	//saveFeature(test_features, test_labels, test_14);

	//Training
	//loadFeature(train_features, train_labels, train_14);
	//svm_train(train_features, train_labels, model_14);

	//Testing 
	//loadFeature(test_features, test_labels, test_14);
	//svm_test(test_features, test_labels, model_14);
	
	//dataToLibSVM(train_features, train_labels, "..\\Datasets\\Features\\Split\\train_14");
/***********************************Combine Features*************************************************/
	vector<vector<double>> feature1, feature2, feature3, cfeature;
	vector<int> label1, label2, label3;

	//loadFeature(feature1, label1, "..\\Datasets\\Features\\Split\\MHI\\test_108.data");
	//loadFeature(feature2, label2, "..\\Datasets\\Features\\Split\\DDI\\test_108.data");
	//loadFeature(feature3, label3, "..\\Datasets\\Features\\Split\\ADI\\test_108.data");
	//combineData(cfeature, feature1, feature2, feature3);
	//saveFeature(cfeature, label1, "..\\Datasets\\Features\\Split\\All_Data_Test.txt");
	
	//loadFeature(feature1, label1, "..\\Datasets\\Features\\Split\\All_Data_Train.txt");
	//dataToLibSVM(feature1, label1, "..\\Datasets\\Features\\Split\\All_Data_Train");

	train_features.clear(), test_features.clear(), train_labels.clear(), test_labels.clear();
	feature1.clear(), feature2.clear(), feature3.clear(), cfeature.clear();
	label1.clear(), label2.clear(), label3.clear();
	return 0;
}

void svm_train(vector<vector<double>>& vfeatures, vector<int>& vlabels, string model_file){
	cout<<"Training Process Begin..."<<endl;
	//step1: prepare trainning examples	
	int num_example=vfeatures.size();
	int num_feature=vfeatures[0].size();
	CvMat *features=cvCreateMat(num_example, num_feature, CV_32FC1);
	CvMat *labels=cvCreateMat(num_example, 1, CV_32FC1);
	for(int i=0; i<num_example; ++i){
		for(int j=0 ;j<num_feature ;++j)
			cvmSet(features, i, j, vfeatures[i][j]);
		cvmSet(labels, i, 0, vlabels[i]) ;
	}
	cvNormalize(features, features, -1, 1, CV_MINMAX, NULL);

	//step2: set training parameters
	CvSVMParams params;
	params.svm_type=CvSVM::C_SVC;//RBF
	params.kernel_type=CvSVM::LINEAR;
	params.term_crit=cvTermCriteria(CV_TERMCRIT_ITER, 100, FLT_EPSILON);

	//step3: start training process
	CvSVM svm;
	svm.train(features, labels, NULL, NULL, params);

	//set all unused value to 0
	CvParamGrid nuGrid = CvParamGrid(1,1,0.0);
	CvParamGrid coeffGrid = CvParamGrid(1,1,0.0);
	CvParamGrid degreeGrid = CvParamGrid(1,1,0.0);
	CvSVM regressor;
	//svm.train_auto(features, labels, NULL, NULL, params, 10, regressor.get_default_grid(CvSVM::C), regressor.get_default_grid(CvSVM::GAMMA), regressor.get_default_grid(CvSVM::P), nuGrid, coeffGrid, degreeGrid);
	
	svm.save(model_file.c_str());
	cout<<"Training Process End..."<<endl;
}

void svm_test(vector<vector<double>>& vfeatures, vector<int>& vlabels, string model_file){
	cout<<"Testing Process Begin..."<<endl;

	//step1: prepare testing examples	
	int num_example=vfeatures.size();
	int num_feature=vfeatures[0].size();
	CvMat *features=cvCreateMat(1, num_feature, CV_32FC1);

	int correct=0;
	for(int i=0; i<num_example; ++i){
		for(int j=0 ;j<num_feature ;++j)
			cvmSet(features, 0, j, vfeatures[i][j]);
		cvNormalize(features, features, -1, 1, CV_MINMAX, NULL);

		//step2: load svm from model file
		CvSVM svm;
		svm.load(model_file.c_str());

		//step3: start test process
		int classNo;
		
		classNo=svm.predict(features);
		if(classNo==vlabels[i])
			correct++;
		cout<<"Actual: "<<decide(vlabels[i])<<"\tClassfier: "<<decide(classNo)<<endl;
	}
	cout<<"Accuracy = "<<((double)(1.00*correct/num_example))<<endl;
	cout<<"Testing Process End..."<<endl;
}

string decide(int classNo){
	switch(classNo){
		case 1:return "bend";
		case 2:return "bowling";
		case 3:return "box";
		case 4:return "jump";
		case 5:return "kick";
		case 6:return "squat";
		case 7:return "strenth";
		case 8:return "swim";
		case 9:return "tumble";
		case 10:return "wave";
	}
}

void extractFeature(string srcPath, vector<vector<double>>& features, vector<int>& labels, bool save_immediate_result, string mhiDir, string adiDir, string ddiDir){
	int width=640, height=480, channels=3, step=1920;
	CvSize size = cvSize(width, height);
	IplImage *mhiImg=cvCreateImage(size, IPL_DEPTH_8U, channels);
	IplImage *adiImg=cvCreateImage(size, IPL_DEPTH_8U, channels);
	IplImage *ddiImg=cvCreateImage(size, IPL_DEPTH_8U, channels);

	int classNo=1;
	string tpath;
	vector<string> dirs=getAllFiles(srcPath);
	for(int i=0; i<dirs.size(); i++){
		string dir=tpath.assign(srcPath).append("\\").append(dirs[i]);
		vector<string> files=getAllFiles(dir);
		for(int j=0; j<files.size(); j++){
			string fileName=files[j].substr(0, files[j].size()-4);
			stringstream mhiFile, adiFile, ddiFile;
			mhiFile<<mhiDir<<"\\"<<dirs[i]<<"\\"<<fileName<<".jpg";
			adiFile<<adiDir<<"\\"<<dirs[i]<<"\\"<<fileName<<".jpg";
			ddiFile<<ddiDir<<"\\"<<dirs[i]<<"\\"<<fileName<<".jpg";
			
			getMHI(dir, files[j], mhiImg, save_immediate_result, mhiFile.str());
			getADI(dir, files[j], adiImg, save_immediate_result, adiFile.str());
			getDDI(dir, files[j], ddiImg, save_immediate_result, ddiFile.str());

			vector<double> feature=vector<double>(108, 0);
			//extractFeatureFromHu(mhiImg, adiImg, feature, 0);
			extractFeatureFromHBB(ddiImg, feature, 0);
			features.push_back(feature);
			labels.push_back(classNo);
			feature.clear();
		}
		classNo++;
	}
	dirs.clear();
}

void saveFeature(vector<vector<double>>& features, vector<int>& labels, string featureFile){
	ofstream out(featureFile);
	if(!out){
		cout<<"Unable to open output file"<<endl;
		exit(0);
	}

	vector<double> data;
	for(int i=0; i<features.size(); i++){
		data=features[i];
		out<<labels[i];
		for(int i=0; i<data.size(); i++)
			out<<" "<<data[i];
		if(i!=(features.size()-1))
			out<<endl;
	}
}

void loadFeature(vector<vector<double>>& features, vector<int>& labels, string featureFile){
	ifstream in(featureFile);
	if(!in){
		cout<<"Unable to open output file"<<endl;
		exit(0);
	}

	string line;
	while(getline(in, line)){
		vector<double> data;
		vector<double> feature;
		split(line, " ", data);
		for(int i=1; i<data.size(); i++)
			feature.push_back(data[i]);
		features.push_back(feature);
		labels.push_back((int)data[0]);
		feature.clear();
		data.clear();
	}
}

void split(string line, string separator, vector<double>& data){
	string str;
	int slen=separator.size(), lastPosition=0, index=-1;
	while((index=line.find(separator, lastPosition))!=-1){
		str=line.substr(lastPosition, index-lastPosition);
		data.push_back(atof(str.c_str()));
		lastPosition=index+slen;
	}
	str=line.substr(lastPosition, line.size()-lastPosition);
	data.push_back(atof(str.c_str()));
}

void combineData(vector<vector<double>>& feature, vector<vector<double>>& feature1, vector<vector<double>>& feature2, vector<vector<double>>& feature3) {
	for(int i=0; i<feature1.size(); i++){
		vector<double> f;
		for(int j=0; j<feature1[i].size(); j++)
			f.push_back(feature1[i][j]);
		for(int j=0; j<feature2[i].size(); j++)
			f.push_back(feature2[i][j]);
		for(int j=0; j<feature3[i].size(); j++)
			f.push_back(feature3[i][j]);
		feature.push_back(f);
		f.clear();
	}
}

void dataToLibSVM(vector<vector<double>>& vfeatures, vector<int>& vlabels, string outputFile){
	ofstream out(outputFile);
	if(!out){
		cout<<"Unable to open output file"<<endl;
		exit(0);
	}

	for(int i=0; i<vfeatures.size(); i++){
		out<<vlabels[i]<<" ";
		for(int j=0; j<vfeatures[0].size(); j++){
			double v=vfeatures[i][j];
			//if(v!=0.0)
				out<<(j+1)<<":"<<vfeatures[i][j]<<" ";
		}
		out<<endl;
	}
}

void extractFeatureFromHBB(IplImage* ddiImg, vector<double>& feature, int start){
	vector<int> winxy=vector<int>(6*6*2, 0);   //36 windows each of size x and y
	vector<int> win2x2y=vector<int>(3*3*2, 0); //9 windows each of size 2x and 2y
	vector<int> win3x3y=vector<int>(2*2*2, 0); //4 windows each of size 3x and 3y
	vector<int> win6x3y=vector<int>(1*2*2, 0); //2 windows each of size 6x and 3y
	vector<int> win3x6y=vector<int>(2*1*2, 0); //2 windows each of size 3x and 6y
	vector<int> win6x6y=vector<int>(1*1*2, 0); //1 windows each of size 6x and 6y
	box(ddiImg, 6, 6, winxy);
	box(ddiImg, 3, 3, win2x2y);
	box(ddiImg, 2, 2, win3x3y);
	box(ddiImg, 1, 2, win6x3y);
	box(ddiImg, 2, 1, win3x6y);
	box(ddiImg, 1, 1, win6x6y);

	//Combine all the 54 patches data as the feature vector of length 54*2 which is 108
	for(int i=0; i<72; i++)
		feature[i+start]=winxy[i];
	for(int i=72; i<90; i++)
		feature[i+start]=win2x2y[i-72];
	for(int i=90; i<98; i++)
		feature[i+start]=win3x3y[i-90];
	for(int i=98; i<102; i++)
		feature[i+start]=win6x3y[i-98];
	for(int i=102; i<106; i++)
		feature[i+start]=win3x6y[i-102];
	for(int i=106; i<108; i++)
		feature[i+start]=win6x6y[i-106];
	winxy.clear(), win2x2y.clear(), win3x3y.clear(), win6x3y.clear(), win3x6y.clear(), win6x6y.clear();
}

void box(IplImage* img, int horPatches, int verPatches, vector<int>& win){
	int width=img->width, height=img->height;
	int horSize=width/horPatches, verSize=height/verPatches;
	vector<int> maxData=vector<int>(horPatches*verPatches, 0);
	vector<int> minData=vector<int>(horPatches*verPatches, 0);
	int n=0;
	for(int i=0; i<verPatches; i++){
		for(int j=0; j<horPatches; j++){
			maxData[n]=max(img, j*horSize, i*verSize, horSize, verSize);
			minData[n]=min(img, j*horSize, i*verSize, horSize, verSize);
			n++;
		}
	}
	for(int i=0; i<horPatches*verPatches; i++)
		win[i]=maxData[i];
	for(int i=0; i<horPatches*verPatches; i++)
		win[i+horPatches*verPatches]=minData[i];
	maxData.clear(), minData.clear();
}

int max(IplImage* img, int x, int y, int horSize, int verSize){
	int val=0;
	int step=img->widthStep, chanels=img->nChannels;
	for(int i=y; i<y+verSize; i++){
		for(int j=x; j<x+horSize; j++){
			int k=(uchar)img->imageData[i*step+j*chanels+0] ;
			if(val<k)
				val=k ; 
		}
	}
	return val;
}

int min(IplImage* img, int x, int y, int horSize, int verSize){
	int val=255;
	int step=img->widthStep, chanels=img->nChannels;
	for(int i=y; i<y+verSize; i++){
		for(int j=x; j<x+horSize; j++){
			int k=(uchar)img->imageData[i*step+j*chanels+0] ;
			if(val>k)
				val=k ; 
		}
	}
	return val;
}

void extractFeatureFromHu(IplImage* mhiImg, IplImage* adiImg, vector<double>& feature, int start){
	vector<double> feature1=vector<double>(7, 0);
	vector<double> feature2=vector<double>(7, 0);
	HuMoment(mhiImg, feature1);
	HuMoment(adiImg, feature2);

	for(int i=0; i<7; i++)
		feature[i+start]=feature1[i];
	for(int i=7; i<14; i++)
		feature[i+start]=feature1[i-7];
	feature1.clear(), feature2.clear();
}

void HuMoment(IplImage* img, vector<double>& feature){
	long long M00=getMpq(img, 0, 0);
	long long M01=getMpq(img, 0, 1);
	long long M10=getMpq(img, 1, 0);

	double xbar=M10/M00, ybar=M01/M00;

	double N00=getUpq(img, xbar, ybar, 0, 0);
	double N01=getUpq(img, xbar, ybar, 0, 1)/pow(N00, 1.5);
	double N02=getUpq(img, xbar, ybar, 0, 2)/pow(N00, 2);
	double N03=getUpq(img, xbar, ybar, 0, 3)/pow(N00, 2.5);
	double N10=getUpq(img, xbar, ybar, 1, 0)/pow(N00, 1.5);
	double N11=getUpq(img, xbar, ybar, 1, 1)/pow(N00, 2);
	double N12=getUpq(img, xbar, ybar, 1, 2)/pow(N00, 2.5);
	double N20=getUpq(img, xbar, ybar, 2, 0)/pow(N00, 2);
	double N21=getUpq(img, xbar, ybar, 2, 1)/pow(N00, 2.5);
	double N30=getUpq(img, xbar, ybar, 3, 0)/pow(N00, 2.5);

	double M1=N20+N02;
	double M2=pow(N20-N02, 2)+4*pow(N11, 2);
	double M3=pow(N30-3*N12, 2)+pow(3*N21-N03, 2);
	double M4=pow(N30+N12, 2)+pow(N21+N03, 2);
	double M5=(N30-3*N12)*(N30+N12)*(pow(N30+N12, 2)-3*pow(N21+N03, 2))+(3*N21-N03)*(N21+N03)*(3*pow(N30+N12, 2)-pow(N21+N03, 2));
	double M6=(N20-N02)*(pow(N30+N12, 2)-pow(N21+N03, 2))+4*N11*(N30+N12)*(N21+N03);
	double M7=(3*N21-N03)*(N30+N12)*(pow(N30+N12, 2)-3*pow(N21+N03, 2))-(N30-3*N12)*(N21+N03)*(3*pow(N30+N12, 2)-pow(N21+N03, 2));

	feature[0]=M1;
	feature[1]=M2;
	feature[2]=M3;
	feature[3]=M4;
	feature[4]=M5;
	feature[5]=M6;
	feature[6]=M7;
}

//p+q阶几何矩
long long getMpq(IplImage* img,int p, int q){
	int width=img->width, height=img->height;
	int step=img->widthStep, channels=img->nChannels;
	long long sum =0 ;
	uchar *data=(uchar*)img->imageData;
	for(int h=0; h<height; h++)
		for(int w=0; w<width; w++)
			sum+=pow(1.0*w, p)*pow(1.0*h ,q)*data[h*step+w*channels];
	return sum ;
}

//p+q阶中心距
long long getUpq(IplImage* img, double xbar, double ybar, int p, int q){
	int width=img->width, height=img->height;
	int step=img->widthStep, chanels=img->nChannels;
	
	uchar* data=(uchar*)img->imageData ;
	long long sum=0 ;
	for(int h=0; h<height; h++)
		for(int w=0; w<width; w++)
			sum+=pow(w-xbar, p)*pow(h-ybar, q)*data[h*step+w*chanels];
	return  sum ;	
}

void getMHI(string srcPath, string srcFile, IplImage *mhiImg, bool isSave, string destFile){
	int width=640, height=480, channels=3, step=1920;

	CvSize size=cvSize(width, height);
	IplImage *tmhi=cvCreateImage(size, IPL_DEPTH_8U, channels);

	initialize(mhiImg, 0);
	initialize(tmhi, 32);

	uchar *tmhiData=(uchar*)tmhi->imageData;
	uchar *mhiData=(uchar*)mhiImg->imageData;	

	cout<<"MHI: "<<srcFile<<endl;
	string file=srcPath.append("\\").append(srcFile);
	CvCapture* preCapture=cvCaptureFromAVI(file.c_str());
	CvCapture* curCapture=cvCaptureFromAVI(file.c_str());
	if(preCapture==NULL || curCapture==NULL){
		printf("Could not initialize capturing...\n");
		return;
	}

	IplImage *preFrame=NULL;
	IplImage *curFrame=NULL;
	uchar *preData, *curData;

	preFrame=cvQueryFrame(preCapture);
	curFrame=cvQueryFrame(curCapture);
	curFrame=cvQueryFrame(curCapture);
	do{
		binaryProcessing(preFrame, 1);
		binaryProcessing(curFrame, 1);
		preData=(uchar*)preFrame->imageData;
		curData=(uchar*)curFrame->imageData;
		for(int w=0; w<width; w++){
			for(int h=0; h<height; h++){
				if((curData[h*step+w*channels]-preData[h*step+w*channels])==1){
					tmhiData[h*step+w*channels+0]=15;
					tmhiData[h*step+w*channels+1]=15;
					tmhiData[h*step+w*channels+2]=15;
				}else{
					int pixel0=tmhiData[h*step+w*channels+0];
					int pixel1=tmhiData[h*step+w*channels+1];
					int pixel2=tmhiData[h*step+w*channels+2];

					tmhiData[h*step+w*channels+0]=(pixel0-15)>0 ? pixel0:0;
					tmhiData[h*step+w*channels+1]=(pixel1-15)>0 ? pixel1:0;
					tmhiData[h*step+w*channels+2]=(pixel2-15)>0 ? pixel2:0;
				}
				int p=tmhiData[h*step+w*channels+0];
				mhiData[h*step+w*channels+0]+=p;
			    mhiData[h*step+w*channels+1]+=p;
			    mhiData[h*step+w*channels+2]+=p;
			}
		}
		preFrame=cvQueryFrame(preCapture);
		curFrame=cvQueryFrame(curCapture);
	}while(curFrame!=NULL);
	cvReleaseCapture(&preCapture);
	cvReleaseCapture(&curCapture);

	//Save Motion History Image to Dest Path
	if(isSave)
		cvSaveImage(destFile.c_str(), mhiImg);
}

void initialize(IplImage *img, int value){
	int width=img->width, height=img->height, channels=img->nChannels, step=img->widthStep;
	uchar* data=(uchar*)img->imageData  ;

	for(int w=0; w<width; w++){
		for(int h=0; h<height; h++){
			data[h*step+w*channels+0]=value ;
			data[h*step+w*channels+1]=value ;
			data[h*step+w*channels+2]=value  ;
		}
	}
}

void binaryProcessing(IplImage *img, int value){
	int width=img->width, height=img->height, channels=img->nChannels, step=img->widthStep;

	uchar *data=(uchar*)img->imageData;
	for(int w=0; w<width; w++){
		for(int h=0; h<height; h++){
			if(data[h*step+w*channels] != 0)
				data[h*step+w*channels+0]=value;
			if(data[h*step+w*channels+1] != 0)
				data[h*step+w*channels+1]=value;
			if(data[h*step+w*channels+2] != 0)
				data[h*step+w*channels+2]=value;
		}
	}
}

void getADI(string srcPath, string srcFile, IplImage *adiImg, bool isSave, string destFile){
	int width=640, height=480, channels=3, step=1920;

	CvSize size = cvSize(width, height);
	IplImage *adi=cvCreateImage(size, IPL_DEPTH_8U, channels);
	uchar *adiData=(uchar*)adi->imageData;

	CvMat* DMat=cvCreateMat(480 ,640, CV_32FC1); 
	CvMat* BMat=cvCreateMat(480, 640, CV_32FC1);

	//Calculate Matrix D and B
	cout<<"ADI: "<<srcFile<<endl;
	string file=srcPath.append("\\").append(srcFile);
	CvCapture* capture=cvCaptureFromAVI(file.c_str());
	if(capture==NULL){
		printf("Could not initialize capturing...\n");
		return;
	}
	IplImage *frame=NULL;
	uchar *frameData;
	int frameNo=0;
	while(1){
		frame=cvQueryFrame(capture);
		if(frame!=NULL){
			frameData=(uchar*)frame->imageData ;
			frameNo++;
			for(int w=0; w<width; w++){
				for(int h=0; h<height; h++){
					int td, tb;
					if(frameNo==1){
						td=0;
						tb=0;
					}else{
						td=cvmGet(DMat, h, w); 
						tb=cvmGet(BMat, h, w);
					}
					int d=frameData[h*step+w*channels];
					td+=d;
					if(d != 0)
					    tb+=1;
					else
						tb+=0;
					cvmSet(DMat, h, w, td);
				    cvmSet(BMat, h, w, tb);
				}
			}
		}else{
			cvReleaseCapture(&capture);
			break;
		}
	}

	//Calculate Average Depth Image
	for(int w=0; w<width; w++){
		for(int h=0; h<height; h++){
			int dd=cvmGet(DMat, h, w);
			int bb=cvmGet(BMat, h, w);
			if(bb!=0){
				float pixel=dd/bb ;
				adiData[h*step+w*channels+0]=pixel;
				adiData[h*step+w*channels+1]=pixel ;
				adiData[h*step+w*channels+2]=pixel ;
			}else{
				adiData[h*step+w*channels+0]=0 ;
				adiData[h*step+w*channels+1]=0 ;
				adiData[h*step+w*channels+2]=0 ;
			}
		}
	}
	//Save Average Depth Image to Dest Path
	if(isSave)
		cvSaveImage(destFile.c_str(), adi);
}

void getDDI(string srcPath, string srcFile, IplImage *ddiImg, bool isSave, string destFile){
	int width=640, height=480, channels=3, step=1920;

	CvSize size = cvSize(width, height);
	uchar *dmiData=(uchar*)ddiImg->imageData;

	//Initialization Matrix Max and Min
	IplImage *maxImg = cvCreateImage(size, IPL_DEPTH_8U, channels);
	IplImage *minImg = cvCreateImage(size, IPL_DEPTH_8U, channels);
	uchar *maxData=(uchar*)maxImg->imageData;
	uchar *minData=(uchar*)minImg->imageData;
	for(int w=0; w<width; w++){
		for(int h=0; h<height; h++){
			maxData[h*step+w*channels+0]=0;
			maxData[h*step+w*channels+1]=0;
			maxData[h*step+w*channels+2]=0;

			minData[h*step+w*channels+0]=255;
			minData[h*step+w*channels+1]=255;
			minData[h*step+w*channels+2]=255;
		}
	}

	//Calculate Matrix Max and Min
	cout<<"DDI: "<<srcFile<<endl;
	string file=srcPath.append("\\").append(srcFile);
	CvCapture* capture=cvCaptureFromAVI(file.c_str());
	if(capture==NULL){
		printf("Could not initialize capturing...\n");
		return;
	}
	IplImage *frame=NULL;
	uchar *frameData;
	while(1){
		frame=cvQueryFrame(capture);
		if(frame!=NULL){
			frameData=(uchar*)frame->imageData;
			for(int w=0; w<width; w++){
				for(int h=0; h<height; h++){
					(uchar)maxData[h*step+w*channels+0]=(uchar)frameData[h*step+w*channels+0]>(uchar)maxData[h*step+w*channels+0]?(uchar)frameData[h*step+w*channels+0]:(uchar)maxData[h*step+w*channels+0];
					(uchar)maxData[h*step+w*channels+1]=(uchar)frameData[h*step+w*channels+1]>(uchar)maxData[h*step+w*channels+1]?(uchar)frameData[h*step+w*channels+1]:(uchar)maxData[h*step+w*channels+1];
					(uchar)maxData[h*step+w*channels+2]=(uchar)frameData[h*step+w*channels+2]>(uchar)maxData[h*step+w*channels+2]?(uchar)frameData[h*step+w*channels+2]:(uchar)maxData[h*step+w*channels+2];

					(uchar)minData[h*step+w*channels+0]=(uchar)frameData[h*step+w*channels+0]<(uchar)minData[h*step+w*channels+0]?(uchar)frameData[h*step+w*channels+0]:(uchar)minData[h*step+w*channels+0];
					(uchar)minData[h*step+w*channels+1]=(uchar)frameData[h*step+w*channels+1]<(uchar)minData[h*step+w*channels+1]?(uchar)frameData[h*step+w*channels+1]:(uchar)minData[h*step+w*channels+1];
					(uchar)minData[h*step+w*channels+2]=(uchar)frameData[h*step+w*channels+2]<(uchar)minData[h*step+w*channels+2]?(uchar)frameData[h*step+w*channels+2]:(uchar)minData[h*step+w*channels+2];
				}
			}
		}else{
			cvReleaseCapture(&capture);
			break;
		}
	}

	//Calculate Depth Difference Image
	for(int w=0; w<width; w++){
		for(int h=0; h<height; h++){
			(uchar)dmiData[h*step+w*channels+0]=(uchar)maxData[h*step+w*channels+0]-(uchar)minData[h*step+w*channels+0];
			(uchar)dmiData[h*step+w*channels+1]=(uchar)maxData[h*step+w*channels+1]-(uchar)minData[h*step+w*channels+1];
			(uchar)dmiData[h*step+w*channels+2]=(uchar)maxData[h*step+w*channels+2]-(uchar)minData[h*step+w*channels+2];
		}
	}

	//Save Depth Difference Image to Dest Path
	if(isSave)
		cvSaveImage(destFile.c_str(), ddiImg);
}

//Get frames of the videos in the src directory
void getAllFrame(string srcPath, string destPath){
	cutAllVideos(srcPath, destPath);
}

void cutAllVideos(string srcPath, string destPath){
	string tpath;
	vector<string> dirs=getAllFiles(srcPath);
	for(int i=0; i<dirs.size(); i++){
		string dir=tpath.assign(srcPath).append("\\").append(dirs[i]);
		vector<string> files=getAllFiles(dir);
		for(int j=0; j<files.size(); j++)
			videoCutFrame(dir, files[j], destPath);
	}
	dirs.clear();
}

void videoCutFrame(string filepath, string filename, string destDir){
	cout<<filename<<endl;
	string file=filepath.append("\\").append(filename);
	CvCapture* capture=cvCaptureFromAVI(file.c_str());
	if(capture==NULL){
		printf("Could not initialize capturing...\n");
		return;
	}
	int frameNo=0;
	IplImage *image=NULL;
	while(1){
		image=cvQueryFrame(capture);
		if(image!=NULL){
			frameNo++;
			stringstream path;
			path<<destDir<<"\\"<<filename.substr(0, filename.length()-6)<<"\\"<<filename.substr(0, filename.length()-4)<<"_"<<frameNo<<".png";
			cvSaveImage(path.str().c_str(), image);
		}else{
			return;
		}
	}
}

vector<string> getAllFiles(string filepath){
	vector<string> files;
    _finddata_t file;
    long hFile;
	string tpath;
	if((hFile = _findfirst(tpath.assign(filepath).append("\\*.*").c_str(), &file))!=-1){
        while(_findnext(hFile, &file)==0){
			if(strcmp(file.name, ".")!=0 && strcmp(file.name, "..")!=0)
				files.push_back(file.name);
        }
    }
    _findclose(hFile);
    return files;
}