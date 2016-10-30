/**
 * @file   fortask2.h
 * @mainpage Functions for task2
 * @author Kosheleva Elena
 * @date   November 4, 2016
 * @brief  Functions for the second task for computer graphics.
 *
 * Detailed description of file.
 */
#include <string>
#include <vector>
#include <fstream>
#include <cassert>
#include <iostream>
#include <cmath>

#include "classifier.h"
#include "EasyBMP.h"
#include "linear.h"
#include "argvparser.h"
#include "matrix.h"
#include "Timer.h"
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

/** Type for working with grayscale images.
 Each pixel is represented by a single floating-point value 
  */


typedef Matrix<float> BinImage;
/**
@function height of BMP.

This function gets height of BMP and returns it in a uint value.
Standart method TellHeight() returns int-type value,
but uint is a more useful type for this program.
@param bmp - bitmap for extracting height
@returns height of bmp
*/
uint GetHeight(BMP *bmp) {
    return static_cast<uint>(bmp->TellHeight());
}
/**
@function width of BMP.

This function gets height of BMP and returns it in a uint value.
Standart method TellHeight() returns int-type value,
but uint is a more useful type for this program.
@param bmp - bitmap for extracting height
@returns height of bmp
*/
uint GetWidth(BMP *bmp) {
    return static_cast<uint>(bmp->TellWidth());
}

/** 
@function Compute length of gradient for image.

@param hor - horisontal part of gradient
@param vert - vertical part of gradient

@returns matrix with lengthes of gradient for each pixel
*/
BinImage ModGrad(BinImage vert, BinImage horiz){
	BinImage module_gradient(vert.n_rows , vert.n_cols );
	double color1, color2;
		for (uint i = 0; i< vert.n_rows; i ++)
			for (uint j = 0; j< vert.n_cols; j ++){
				color1 = horiz(i, j);
				color2 = vert(i, j);
				module_gradient(i, j) = sqrt (color1*color1 + color2*color2);
				//gradient_direction(i, j) = atan2(color2, color1); 
			}
	
	return module_gradient;
}
/** 
@function Compute length of gradient for image using SSE.
* 
@param hor - horisontal part of gradient
@param vert - vertical part of gradient

@returns matrix with lengthes of gradient for each pixel
*/

BinImage ModGradSSE(BinImage vert, BinImage horiz){
	BinImage result(vert.n_rows, vert.n_cols);
	__m128 buf_vert, buf_horiz;
	for (uint i = 0; i < vert.n_rows; i ++)
		for (uint j = 0; j < vert.n_cols ; j+=4){
			buf_vert = _mm_set_ps(vert(i, j+3), vert(i,j+2), vert(i,j+1), vert(i,j));
			buf_horiz = _mm_set_ps(horiz(i,j+3), horiz(i, j+2), horiz(i, j+1), horiz(i,j));
			float tmp[4];
			_mm_store_ps(tmp, _mm_sqrt_ps(_mm_add_ps( _mm_mul_ps(buf_vert, buf_vert), _mm_mul_ps(buf_horiz, buf_horiz) ) ) );
			result(i,j) = tmp[0];
			result(i,j+1) = tmp[1];
			result(i,j+2) = tmp[2];
			result(i,j+3) = tmp[3];
		}
	return result;
}

/**
@function Convert rgb image to grayscaled image.
@param  rgb image (type Image) to create grayscale
@returns image in grayscale
*/
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
/**
@function Convert BMP to Image. 
@param bitmap - *BMP to create grayscale
@returns rgb Image 
*/
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

/**
@function Apply vertical Sobel kernel to image.

This function use SSE.

@param bmp - original image
@returns image after Sobel kernel
*/
BinImage UseSobelFilterVerticalSSE(BinImage bmp) {
    // Result image
    BinImage result(bmp.n_rows , bmp.n_cols );

    //Variables for lines in Sobel kernel matrix
    __m128 topLine, botLine;

    //Initialization of Sobel kernel lines
    topLine = _mm_set_ps(0.0f, -1.0f, -2.0f, -1.0f);
    botLine = _mm_set_ps(0.0f,  1.0f,  2.0f,  1.0f);

    //Loop for rows of result image
    for (uint i = 0; i < result.n_rows - 2; ++i) {
        //Loop for rows of result image
        for (uint j = 0; j < result.n_cols - 2; ++j) {
            //Initialize lines of submatrix of original image
            __m128 bmpTop = _mm_set_ps(0.0f, bmp(i, j), bmp(i, j + 1), bmp(i, j + 2));
            __m128 bmpBot = _mm_set_ps(0.0f, bmp(i + 2, j), bmp(i + 2, j + 1), bmp(i + 2, j + 2));

            //Array with result of SSE operations
            float tmp[4];
            //Compute result for current pixel
            _mm_store_ps(tmp, _mm_add_ps(_mm_mul_ps(bmpTop, topLine), _mm_mul_ps(bmpBot, botLine)));
            result(i, j) = tmp[1] + tmp[2] + tmp[0];
        }
    }
    return result.deep_copy();
}

/**
@function Apply vertical Sobel kernel to image.

@param bmp - original image
@returns image after Sobel kernel is applied
*/
BinImage UseSobelFilterVertical(BinImage bmp) {
    // Result image
    BinImage result(bmp.n_rows , bmp.n_cols);

    //Loop for rows of result image
    for (uint i = 0; i < result.n_rows - 2; ++i) {
        //Loop for rows of result image
        for (uint j = 0; j < result.n_cols - 2; ++j) {
            //Compute result for current pixel
            result(i, j) = -(bmp(i, j) + 2 * bmp(i, j + 1) + bmp(i, j + 2));
            result(i, j) += bmp(i + 2, j) + 2 * bmp(i + 2, j + 1) + bmp(i + 2, j + 2);
        }
    }
    return result.deep_copy();
}

/**
@function Apply horisontal Sobel kernel to image.

This function use SSE.

@param bmp - original image
@returns  image after Sobel kernel
*/

BinImage UseSobelFilterHorisontalSSE(BinImage bmp) {
    // Result image
    BinImage result(bmp.n_rows , bmp.n_cols );

    //Variables for columns in Sobel kernel matrix
    __m128 leftLine, rightLine;
    //Initialization of Sobel kernel columns
    leftLine = _mm_set_ps(0.0f, -1.0f, -2.0f, -1.0f);
    rightLine = _mm_set_ps(0.0f,  1.0f,  2.0f,  1.0f);
    //Loop for rows of result image
    for (uint i = 0; i < result.n_rows-2; ++i) {
        //Loop for rows of result image
        for (uint j = 0; j < result.n_cols - 2; ++j) {
            //Initialize columns of submatrix of original image
            __m128 bmpLeft = _mm_set_ps(0.0f, bmp(i, j), bmp(i + 1, j), bmp(i + 2, j));
            __m128 bmpRight = _mm_set_ps(0.0f, bmp(i, j + 2), bmp(i + 1, j + 2), bmp(i + 2, j + 2));

            //Array with result of SSE operations
            float tmp[4];
            //Compute result for current pixel
            _mm_store_ps(tmp, _mm_add_ps(_mm_mul_ps(bmpLeft, leftLine), _mm_mul_ps(bmpRight, rightLine)));
            result(i, j) = tmp[1] + tmp[2] + tmp[0];
        }
    }
    return result.deep_copy();
}

/**
@function Apply horisontal Sobel kernel to image.

This function use SSE.

@param bmp - original image
@returns image after Sobel kernel
*/
BinImage UseSobelFilterHorisontal(BinImage bmp) {
    // Result image
    BinImage result(bmp.n_rows , bmp.n_cols );

    //Loop for rows of result image
    for (uint i = 0; i < result.n_rows - 2; ++i) {
        //Loop for rows of result image
        for (uint j = 0; j < result.n_cols - 2; ++j) {
            //Compute result for current pixel
            result(i, j) = -(bmp(i, j) + 2 * bmp(i + 1, j) + bmp(i + 2, j));
            result(i, j) += bmp(i, j + 2) + 2 * bmp(i + 1, j + 2) + bmp(i + 2, j + 2);
        }
    }
    return result.deep_copy();
}
/**
@function Extend grayscale image by 1 pizel in each direction
by zero-padding.

@param grayscale image to perform zero-padding on.
@returns zero-padded extended image
*/
BinImage ExtendImage(BinImage &bmp, uint len = 1){
	BinImage result(bmp.n_rows + len*2, bmp.n_cols + len*2);
	for (uint i = 0 ; i < result.n_rows ; i ++)
		for (uint j = 0; j < result.n_cols; j ++)
			result(i,j) = 0;
	for (uint i = 0 ; i < bmp.n_rows ; i ++)
		for (uint j = 0; j < bmp.n_cols; j ++)
			result(i+1,j+1) = bmp(i,j);
	//for (uint i = 1; i < result.n_rows; i ++)
		//result(i,0) = img(i, 1);
//	result(0,0) = img(1,1);
	return result;
}



/**
@function Compute HOG.

@param direct - matrix with directions of gradient
@param module - matrix with modules of gradient

@returns HOG
*/
vector<float> ComputeHistogram(const BinImage & direct, const BinImage &module) {
	// 8 bins
	vector <float> hist;
	float bins[8] = {0,0,0,0,0,0,0,0};
	for (uint i = 0; i < direct.n_rows; i ++)
		for (uint j = 0; j < direct.n_cols; j ++){
			double gradient_dir = direct(i, j) / 3.14;
			
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
/**
@function Extract Local Binary Plates features

@param image - from which to extract features
@returns vector with LBP features
*/

vector <float> LBP (BinImage &image){
	int dec;
	float  mass[256];
	for(int i = 0; i < 256; i ++)
		mass[i] = 0;
	int bins[8] = {0, 0, 0, 0, 0, 0, 0, 0};
	BinImage img = ExtendImage(image,1);
	
	/*for (uint i = 0; i < img.n_rows; i ++){
		for (uint j = 0; j < img.n_cols; j ++)
				cout<< img(i,j)<<" ";
			cout<<endl;
		}
	*/
	
	//cout<<img.n_rows<<" "<<img.n_cols<<endl;
	for (uint i = 1; i < img.n_rows-1; i ++){
		for (uint j = 1; j < img.n_cols-1; j ++){
			bins[0] = img(i-1,j-1) >= img(i,j) ? 1: 0;
			bins[1] = img(i-1, j)  >= img(i,j)? 1:0;
			bins[2] = img(i-1,j+1) >= img(i,j)? 1:0;
			bins[3] = img(i, j+1)  >= img(i,j)? 1:0;
			bins[4] = img(i+1, j+1)  >= img(i,j)?1:0;
			bins[5] = img(i+1,j) >= img(i,j)? 1:0;
			bins[6] = img(i+1, j-1)  >= img(i,j)? 1:0;
			bins[7] = img(i,j-1) >= img(i,j)? 1:0;
			dec = bins[7] + bins[6] * 2 + bins[5]*4 + bins[4]*8 + bins[3]*16+
					bins[2]*32 + bins[1]*64 + bins[0]*128;
			//for (int k = 0; k < 8; k ++)
				//cout<<bins[k];
			//cout<<" "<<i<<" "<<j;
			//cout<<endl;
			mass[dec] ++;
		}
	}
	//float sum = 0;
	//for (int i = 0; i < 256; i ++)
		//sum += mass[i] * mass[i];
	//cout<<sum<<endl;
	//sum = sqrt(sum);
	float sum =  1;
	vector <float> hist;
	if (sum <= 0){
		for (int i = 0; i <256 ; i ++)
			hist.push_back(0);
		return hist;
	}
	
	for (int i = 0; i < 256; i ++){
	//	cout<<mass[i]/sum<<" ";
		hist.push_back(mass[i]/sum);
	}
	/*
	for (int i =0 ; i < 256; i ++)
		if (mass[i] >= 1)
			cout<<i<<" ";
	cout<<endl;*/
	//for (std::vector<float>::const_iterator i = hist.begin(); i != hist.end(); ++i)
		//		cout << *i << ' '; 
	//cout<<endl<<endl;
	return hist;
}
/** 
@function Resize image
 
@param img- an image to resize
@param size_x - output width
@param size_y - output height

@returns resized image
*/
Image Resize(Image &img, float size_x, float size_y){
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
/**
@function Calculate mean colors in rgb image
@param img - rgb Image
@param m - float* to store the results
*/

void CalculateMean(Image &img, float *m){
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

/**
@function Extract color features
@param rgb Image from which to extract features
@returns vector with color features
*/

vector <float> ColorFeatures (Image &img){
	int sq_size_r = img.n_rows / 8;
	int sq_size_c = img.n_cols / 8;
	vector <float> map;
	float mean_colors[3] = {0,0,0};
	for (uint i = 0; i < img.n_rows - sq_size_r; i += sq_size_r){
		for (uint j = 0; j < img.n_cols - sq_size_c; j += sq_size_c){
			Image patch (img.submatrix(i,j, sq_size_r, sq_size_c));
			CalculateMean (patch, mean_colors);
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




