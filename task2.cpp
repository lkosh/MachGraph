#include <string>
#include <vector>
#include <fstream>
#include <cassert>
#include <iostream>
#include <cmath>
#include <math.h>
#include "classifier.h"
#include "EasyBMP.h"
#include "linear.h"
#include "argvparser.h"
#include "matrix.h"
#include "io.h"
using std::string;
using std::vector;
using std::ifstream;
using std::ofstream;
using std::pair;
using std::make_pair;
using std::cout;
using std::cerr;
using std::endl;
using std::get;
using CommandLineProcessing::ArgvParser;

typedef vector<pair<BMP*, int> > TDataSet;
typedef vector<pair<string, int> > TFileList;
typedef vector<pair<vector<float>, int> > TFeatures;

// Load list of files and its labels from 'data_file' and
// stores it in 'file_list'
void LoadFileList(const string& data_file, TFileList* file_list) {
    ifstream stream(data_file.c_str());

    string filename;
    int label;
    
    int char_idx = data_file.size() - 1;
    for (; char_idx >= 0; --char_idx)
        if (data_file[char_idx] == '/' || data_file[char_idx] == '\\')
            break;
    string data_path = data_file.substr(0,char_idx+1);
    
    while(!stream.eof() && !stream.fail()) {
        stream >> filename >> label;
        if (filename.size())
            file_list->push_back(make_pair(data_path + filename, label));
    }

    stream.close();
}

// Load images by list of files 'file_list' and store them in 'data_set'
void LoadImages(const TFileList& file_list, TDataSet* data_set) {
    for (size_t img_idx = 0; img_idx < file_list.size(); ++img_idx) {
            // Create image
        BMP* image = new BMP();
            // Read image from file
        image->ReadFromFile(file_list[img_idx].first.c_str());
            // Add image and it's label to dataset
        data_set->push_back(make_pair(image, file_list[img_idx].second));
    }
}

// Save result of prediction to file
void SavePredictions(const TFileList& file_list,
                     const TLabels& labels, 
                     const string& prediction_file) {
        // Check that list of files and list of labels has equal size 
    assert(file_list.size() == labels.size());
        // Open 'prediction_file' for writing
    ofstream stream(prediction_file.c_str());

        // Write file names and labels to stream
    for (size_t image_idx = 0; image_idx < file_list.size(); ++image_idx)
        stream << file_list[image_idx].first << " " << labels[image_idx] << endl;
    stream.close();
}
typedef Matrix<float> BinImage;

uint GetHeight(BMP *bmp) {
    return static_cast<uint>(bmp->TellHeight());
}

uint GetWidth(BMP *bmp) {
    return static_cast<uint>(bmp->TellWidth());
}

template<typename T, typename V, typename W>
inline T clamp(T value, V bottom, W top) {
    return std::max(bottom, std::min(top,value));
}
BinImage ExtendImage(BinImage &bmp, uint len) {
    BinImage result(bmp.n_rows + len * 2, bmp.n_cols + 2 * len);
    for (uint i = 0; i < result.n_rows; ++i) {
        for (uint j = 0; j < result.n_cols; ++j) {
            result(i, j) = bmp(clamp(static_cast<int>(i) - static_cast<int>(len), 0, static_cast<int>(bmp.n_rows - 1)), clamp(static_cast<int>(j) - static_cast<int>(len), 0, static_cast<int>(bmp.n_cols - 1)));
        }
    }
    return result.deep_copy();
}

BinImage Grayscale(Image rgb){
	BinImage result (rgb.n_rows, rgb.n_cols);
	double s = 0;
	for (uint i = 0; i < result.n_rows; i ++)
		for (uint j = 0; j < result.n_cols; j ++){
			
			s = 0.299 * get<0>(rgb(i,j)) + 0.587*get<1>(rgb(i,j)) + 0.114 * get<2>(rgb(i,j));
			result(i,j) = s;
			//cout<<"i: "<<i<<"j: "<<j<<endl;
		}
	return result.deep_copy();
}


BinImage custom(BinImage &src_image, Matrix<double> kernel ) {//let's assume that kernel is 3x3
	double pix;
	
	int a = kernel.n_rows;
	long double sum_pix = 0;
	int m,n,l,p;
	
	int img_rows = src_image.n_rows, img_cols = src_image.n_cols; 
		BinImage tmp(img_rows, img_cols);
	for (int i = a/2; i < img_rows - a/2; i ++){
		for (int j = a/2; j < img_cols - a/2; j ++){
			l = p  = -a/2;
			m = n = 0;
			
			while ( m < a){
				//tie(r, g, b) = src_image(i+l, j+p);
				pix = src_image(i+l, j+p);
				
				sum_pix += kernel(m,n) * pix;
				
				if (p < a/2) p ++;
				else { p =-a/2; l ++;}
				 
				if (n < a-1) n++;
				else { n =0; m ++;}
			}
	
		
			tmp(i,j) = sum_pix;
		
		sum_pix = 0;
		}
	} 
 
   
    return tmp.deep_copy();
}

BinImage sobel_x(BinImage &src_image) {
    Matrix<double> kernel = {{-1, 0, 1},
                             {-2, 0, 2},
                             {-1, 0, 1}};
    return custom(src_image, kernel );
}

BinImage sobel_y(BinImage &src_image) {
    Matrix<double> kernel = {{ 1,  2,  1},
                             { 0,  0,  0},
                             {-1, -2, -1}};
    return custom(src_image, kernel);
}

vector<float> ComputeHistogram(const BinImage & direct, const BinImage &module) {
	// 8 bins
	vector <float> hist;
	float bins[8] = {0,0,0,0,0,0,0,0};
	for (uint i = 0; i < direct.n_rows; i ++)
		for (uint j = 0; j < direct.n_cols; j ++){
			double gradient_dir = direct(i, j) / M_PI;
			
			if (7.0/8 > gradient_dir && gradient_dir >= 5.0/8)
			{
				bins[0] += module(i,j);
			}
			else if (5.0/8 > gradient_dir && gradient_dir >= 3.0/8)
			{
				bins[1] += module(i,j);
			}
			else if (3.0/8 > gradient_dir && gradient_dir >= 1.0/8)
			{
				bins[2] += module(i,j);
			}
			else if (1.0/8 > gradient_dir && gradient_dir >= -1.0/8)
			{
				bins[3] += module(i,j);
			}
			else if (-1.0/8 > gradient_dir && gradient_dir >= -3.0/8)
			{
				bins[4] += module(i,j);
			}
			else if (-3.0/8 > gradient_dir && gradient_dir >= -5.0/8)
			{
				bins[5] += module(i,j);
			}
			else if (-5.0/8 > gradient_dir && gradient_dir >= -7.0/8)
			{
				bins[6] += module(i,j);
			}
			else
			{
				bins[7] += module(i,j);
			}
		}
		double sum = 0;
		for (int i = 0; i < 8; i ++)
			sum += bins[i]*bins[i];
		if (sum <= 0){
			for (int i = 0; i <8 ; i ++)
				hist.push_back(0);
			return hist;
		}
		
		sum = sqrt(sum);
		for (int i = 0; i < 8; i ++)
			hist.push_back(bins[i]/sum);
	return hist;
}


vector <float> LBP (BinImage image){
	int dec, mass[256];
	for(int i = 0; i < 256; i ++)
		mass[i] = 0;
	int bins[8] = {0, 0, 0, 0, 0, 0, 0, 0};
	BinImage img = ExtendImage(image,1);
	for (uint i = 1; i < img.n_rows-1; i ++){
		for (uint j = 1; j < img.n_cols-1; j ++){
			bins[0] = img(i-1,j-1) >= img(i,j);
			bins[1] = img(i-1, j)  >= img(i,j);
			bins[2] = img(i-1,j+1) >= img(i,j);
			bins[3] = img(i, j+1)  >= img(i,j);
			bins[4] = img(i+1, j+1)  >= img(i,j);
			bins[5] = img(i+1,j) >= img(i,j);
			bins[6] = img(i+1, j-1)  >= img(i,j);
			bins[7] = img(i,j-1) >= img(i,j);
			
			dec = bins[7] + bins[6] * 2 + bins[5]*4 + bins[4]*8 + bins[3]*16+
					bins[2]*32 + bins[1]*64 + bins[0]*128;
					//cout<<"dec"<<dec<<endl;
			mass[dec] ++;
		}
	}
	int sum = 0;
	for (int i = 0; i < 256; i ++)
		sum += mass[i] * mass[i];
	
	vector <float> hist;
	if (sum <= 0){
		for (int i = 0; i <256 ; i ++)
			hist.push_back(0);
		return hist;
	}
	sum = sqrt(sum);
	for (int i = 0; i < 256; i ++)
		hist.push_back(mass[i]/sum);
		
	//for (std::vector<float>::const_iterator i = hist.begin(); i != hist.end(); ++i)
		//		cout << *i << ' '; 
	return hist;
}
Image BitmapToRgb (BMP *bitmap){
	Image result (GetHeight(bitmap), GetWidth(bitmap));
	
	for (uint i = 0; i < result.n_rows; i ++)
		for (uint j = 0; j < result.n_cols; j ++){
			RGBApixel pixel = bitmap->GetPixel(j,i);
			get<0>(result(i,j)) = pixel.Red;
			get<1>(result(i,j)) = pixel.Green;
			get<2>(result(i,j)) = pixel.Blue;
			
			//cout<<"i: "<<i<<"j: "<<j<<endl;
		}
	return result.deep_copy();
}
Image Resize(Image img, float size_x, float size_y){
	Image rescaled(size_y, size_x);
	float scale_row = float(img.n_rows) / float(size_y);
	float scale_col = float(img.n_cols) / float(size_x);
	//cout<<scale_col<<" "<<scale_row<<endl;
	//cout<<scale_row<<endl;
	uint k=1, l=1;
	float kk = 1, ll = 1;
	for (uint i = 1; i< rescaled.n_rows ; i ++){
		for (uint j =1;  j < rescaled.n_cols; j ++){
			get<0>(rescaled(i,j)) = (get<0>(img(k,l)) + get<0>(img(k+1,l)) + get<0>(img(k,l+1)) + get<0>(img(k+1,l+1))) / 4;
			get<1>(rescaled(i,j)) = (get<1>(img(k,l)) + get<1>(img(k+1,l)) + get<1>(img(k,l+1)) + get<1>(img(k+1,l+1))) / 4;
			get<2>(rescaled(i,j)) = (get<2>(img(k,l)) + get<2>(img(k+1,l)) + get<2>(img(k,l+1)) + get<2>(img(k+1,l+1))) / 4;
			ll += scale_col;
			l = static_cast<int>(ll);
			if ((l+1 )>=img.n_cols) break;
		}
	l = 0;
	ll =0;
	kk += scale_row;
	k = static_cast<int>(kk);
	if (k+1 >=img.n_rows) break;
	}

	return rescaled.deep_copy();
}
void CalculateMean(Image img, float *m){
	float m_r = 0, m_g = 0, m_b = 0;
	for (uint i = 0; i < img.n_rows; i ++)
		for (uint j = 0; j < img.n_cols; j ++){
			m_r += get<0>(img(i,j));
			m_g += get<1>(img(i,j));
			m_b += get<2>(img(i,j));
		}
	m_r /= (img.n_rows * img.n_cols);
	m_g /= (img.n_rows * img.n_cols);
	m_b /= (img.n_rows * img.n_cols);
	m[0] = m_r;
	m[1] = m_g;
	m[2] = m_b;
//	cout<<m[0]<<" "<<m[1]<<" "<<m[2]<<endl;
}
vector <float> ColorFeatures (Image img){
	int sq_size_r = img.n_rows / 8;
	int sq_size_c = img.n_cols / 8;
	vector <float> map;
	float mean_colors[3] = {0,0,0};
	for (uint i = 0; i < img.n_rows - sq_size_r; i += sq_size_r){
		for (uint j = 0; j < img.n_cols - sq_size_c; j += sq_size_c){
			CalculateMean (img.submatrix(i,j, sq_size_r, sq_size_c), mean_colors);
			mean_colors[0] /= 255;
			mean_colors[1] /= 255;
			mean_colors[2] /= 255;
			//cout<<mean_colors[0]<<" "<<mean_colors[1]<<" "<<mean_colors[2]<<endl;
			map.push_back(mean_colors[0]);
			map.push_back(mean_colors[1]);
			map.push_back(mean_colors[2]);
		}
	}
	//for (std::vector<float>::const_iterator i = map.begin(); i != map.end(); ++i)
		//		cout << *i << ' '; 
	//cout<<endl;
	return map;
}
// Exatract features from dataset.
// You should implement this function by yourself =)
void ExtractFeatures(const TDataSet& data_set, TFeatures* features) {
    for (size_t image_idx = 0; image_idx < data_set.size(); ++image_idx) {
	//	vector <float> color_map = ColorFeatures ( data_set[image_idx].first);
		Image rgb = BitmapToRgb(data_set[image_idx].first);
		save_image(rgb, "rgb.bmp");
		Image resized = Resize(rgb, 32, 64);
		vector <float> color_map = ColorFeatures(resized);
        BinImage gray = Grayscale(resized);
        
        //gray = ExtendImage(gray,1);
        BinImage Ext = ExtendImage(gray,1);
        BinImage sobel_horiz = sobel_x(gray);
        
        BinImage sobel_vert = sobel_y(gray);
        
        BinImage module_gradient (gray.n_rows, gray.n_cols);
		BinImage gradient_direction (gray.n_rows, gray.n_cols);
		double color1,color2;
		for (uint i = 0; i< gray.n_rows; i ++)
			for (uint j = 0; j< gray.n_cols; j ++){
				color1 = sobel_horiz(i, j);
				color2 = sobel_vert(i, j);
				module_gradient(i, j) = sqrt (color1*color1 + color2*color2);
				gradient_direction(i, j) = atan2(color2, color1); 
			}
			
		//32 - size of a square for computing a histogram 
		int sq_size = 4;
		
        vector<float> one_image_features;
        //vector <float> hist;
        for (uint i = 0; i < gray.n_rows - sq_size ; i += sq_size){
			for (uint j = 0; j < gray.n_cols - sq_size; j += sq_size){
				vector <float> hist = ComputeHistogram(gradient_direction.submatrix(i,j, sq_size, sq_size), module_gradient.submatrix(i,j,sq_size, sq_size));
				vector <float> lbp_hist = LBP(gray.submatrix(i, j, sq_size, sq_size));
				for (int k = 0; k < 8 ; k ++){
					one_image_features.push_back(hist.back());
					
					hist.pop_back();
				}
				one_image_features.insert(one_image_features.end(), color_map.begin(), color_map.end());
				//one_image_features.insert(one_image_features.end(), lbp_hist.begin(), lbp_hist.end());
				
				}
			//cout<<i<<j<<endl;
			}
			//for (std::vector<float>::const_iterator i = one_image_features.begin(); i != one_image_features.end(); ++i)
				//cout << *i << ' '; 
			//cout<<endl<<"new image"<<endl;
        features->push_back(make_pair(one_image_features, data_set[image_idx].second));
        // End of sample code

    }
}

// Clear dataset structure
void ClearDataset(TDataSet* data_set) {
        // Delete all images from dataset
    for (size_t image_idx = 0; image_idx < data_set->size(); ++image_idx)
        delete (*data_set)[image_idx].first;
        // Clear dataset
    data_set->clear();
}

// Train SVM classifier using data from 'data_file' and save trained model
// to 'model_file'
void TrainClassifier(const string& data_file, const string& model_file) {
        // List of image file names and its labels
    TFileList file_list;
        // Structure of images and its labels
    TDataSet data_set;
        // Structure of features of images and its labels
    TFeatures features;
        // Model which would be trained
    TModel model;
        // Parameters of classifier
    TClassifierParams params;
    
        // Load list of image file names and its labels
    LoadFileList(data_file, &file_list);
        // Load images
    LoadImages(file_list, &data_set);
        // Extract features from images
    ExtractFeatures(data_set, &features);

        // PLACE YOUR CODE HERE
        // You can change parameters of classifier here
    params.C = 0.01;
    TClassifier classifier(params);
        // Train classifier
    classifier.Train(features, &model);
        // Save model to file
    model.Save(model_file);
        // Clear dataset structure
    ClearDataset(&data_set);
}

// Predict data from 'data_file' using model from 'model_file' and
// save predictions to 'prediction_file'
void PredictData(const string& data_file,
                 const string& model_file,
                 const string& prediction_file) {
        // List of image file names and its labels
    TFileList file_list;
        // Structure of images and its labels
    TDataSet data_set;
        // Structure of features of images and its labels
    TFeatures features;
        // List of image labels
    TLabels labels;

        // Load list of image file names and its labels
    LoadFileList(data_file, &file_list);
        // Load images
    LoadImages(file_list, &data_set);
        // Extract features from images
    ExtractFeatures(data_set, &features);

        // Classifier 
    TClassifier classifier = TClassifier(TClassifierParams());
        // Trained model
    TModel model;
        // Load model from file
    model.Load(model_file);
        // Predict images by its features using 'model' and store predictions
        // to 'labels'
    classifier.Predict(features, model, &labels);

        // Save predictions
    SavePredictions(file_list, labels, prediction_file);
        // Clear dataset structure
    ClearDataset(&data_set);
}

int main(int argc, char** argv) {
    // Command line options parser
    ArgvParser cmd;
        // Description of program
    cmd.setIntroductoryDescription("Machine graphics course, task 2. CMC MSU, 2014.");
        // Add help option
    cmd.setHelpOption("h", "help", "Print this help message");
        // Add other options
    cmd.defineOption("data_set", "File with dataset",
        ArgvParser::OptionRequiresValue | ArgvParser::OptionRequired);
    cmd.defineOption("model", "Path to file to save or load model",
        ArgvParser::OptionRequiresValue | ArgvParser::OptionRequired);
    cmd.defineOption("predicted_labels", "Path to file to save prediction results",
        ArgvParser::OptionRequiresValue);
    cmd.defineOption("train", "Train classifier");
    cmd.defineOption("predict", "Predict dataset");
        
        // Add options aliases
    cmd.defineOptionAlternative("data_set", "d");
    cmd.defineOptionAlternative("model", "m");
    cmd.defineOptionAlternative("predicted_labels", "l");
    cmd.defineOptionAlternative("train", "t");
    cmd.defineOptionAlternative("predict", "p");

        // Parse options
    int result = cmd.parse(argc, argv);

        // Check for errors or help option
    if (result) {
        cout << cmd.parseErrorDescription(result) << endl;
        return result;
    }

        // Get values 
    string data_file = cmd.optionValue("data_set");
    string model_file = cmd.optionValue("model");
    bool train = cmd.foundOption("train");
    bool predict = cmd.foundOption("predict");

        // If we need to train classifier
    if (train)
        TrainClassifier(data_file, model_file);
        // If we need to predict data
    if (predict) {
            // You must declare file to save images
        if (!cmd.foundOption("predicted_labels")) {
            cerr << "Error! Option --predicted_labels not found!" << endl;
            return 1;
        }
            // File to save predictions
        string prediction_file = cmd.optionValue("predicted_labels");
            // Predict data
        PredictData(data_file, model_file, prediction_file);
    }
}
