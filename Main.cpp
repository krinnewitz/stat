#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <opencv/cv.h>
#include <opencv/highgui.h>
#include <iostream>
#include <map>

/**
 * \file	Main.cpp
 * \brief 	This is an implementation of texture comparison using 
 *		statistical methods. Its' goal is mainly to determine
 *		how good this method is in the context of applying
 *		textures in polygonal models obtained from 3d point 
 *		clouds by 3d reconstruction.
 *
 * \author	Kim Oliver Rinnewitz, krinnewitz@uos.de
 */

using namespace std;


/**
 * \brief Reduces the number of colors in the given image
 * 
 * \param input		The input image to reduce the colors in.
			This must be a 3 channel image with 8 bit
			per channel.
 * \param output 	The destination to store the result in.
			This will be an 8 bit one channel image.
 * \param numColors	The maximum number of colors in the 
 *			output image. Note, that this value must
 *			be less than or equal to 256 since the 
 *			output image has only one 8 bit channel.
 */
void reduceColors(cv::Mat input, cv::Mat &output, int numColors)
{
	//allocate output
	output = cv::Mat(input.size(), CV_8U);
	//3 channel pointer to input image
	cv::Mat_<cv::Vec3b>& ptrInput = (cv::Mat_<cv::Vec3b>&)input; 
	//1 channel pointer to output image
	cv::Mat_<uchar>& ptrOutput = (cv::Mat_<uchar>&)output;

	for (int y = 0; y < input.size().height; y++)
	{
		for(int x = 0; x < input.size().width; x++)
		{
			unsigned long int currCol = 0;
			currCol |= (ptrInput(y, x)[0]) << 16;
			currCol |= (ptrInput(y, x)[1]) <<  8;
			currCol |= (ptrInput(y, x)[2]) <<  0;
			ptrOutput(y,x) = currCol / (pow(2, 24) / numColors);
		}
	}
}

/**
 * \brief	Calculates the energy of the given image
 *
 * \param	input	The image to calculate the energy of
 *
 * \retrun	The energy of the given image
 */
float calcEnergy(const cv::Mat &input)
{
	cv::Mat sqr;
	cv::pow(input, 2, sqr);
	return cv::sum(sqr)[0];
}

/**
 * \brief	Calculates the inertia of the given image
 *
 * \param	input	The image to calculate the inertia of
 *
 * \retrun	The inertia of the given image
 */
float calcInertia(const cv::Mat &input)
{
	//Convert input from unsigned char to float matrix
	cv::Mat A;
	input.convertTo(A, CV_32FC1);

	cv::Mat x_minus_y_sqr = cv::Mat(input.size(), CV_32FC1);
	for (int y = 0; y < x_minus_y_sqr.rows; y++)
	{
		for(int x = 0; x < x_minus_y_sqr.cols; x++)
		{
			x_minus_y_sqr.at<float>(y, x) = (x - y) * (x - y);
		}
	}
	
	return cv::sum(x_minus_y_sqr.mul(A))[0];
}

/**
 * \brief	Calculates the homogeneity of the given image
 *
 * \param	input	The image to calculate the homogeneity of
 *
 * \retrun	The homogeneity of the given image
 */
float calcHomogeneity(const cv::Mat &input)
{
	//Convert input from unsigned char to float matrix
	cv::Mat A;
	input.convertTo(A, CV_32FC1);

	for (int y = 0; y < A.rows; y++)
	{
		for(int x = 0; x < A.cols; x++)
		{
			A.at<float>(y, x) /= (1 + fabs(x - y));
		}
	}
	
		
	return cv::sum(A)[0];
}

/**
 * \brief	Calculates the entropy of the given image
 *
 * \param	input	The image to calculate the entropy of
 *
 * \retrun	The entropy of the given image
 */
float calcEntropy(const cv::Mat &input)
{
	//Convert input from unsigned char to float matrix
	cv::Mat A;
	input.convertTo(A, CV_32FC1);
	cv::Mat tmp;
	cv::log(A, tmp);
	return cv::sum(A.mul(tmp))[0];
}

/**
 * \brief	Calculates the x mean i.e. sum x*f(x,y)
 *
 * \param	Input	The input image
 *
 * \return	The x mean
 */
float calcXmean(const cv::Mat & input)
{
	cv::Mat A;
	input.convertTo(A, CV_32FC1);
	cv::Mat meanX = cv::Mat(A.size(), CV_32FC1);
	for (int y = 0; y < meanX.rows; y++)
	{
		for(int x = 0; x < meanX.cols; x++)
		{
			meanX.at<float>(y, x) = x;
		}
	}
	return cv::sum(meanX.mul(A))[0];
}


/**
 * \brief	Calculates the y mean i.e. sum y*f(x,y)
 *
 * \param	Input	The input image
 *
 * \return	The y mean
 */
float calcYmean(const cv::Mat & input)
{
	cv::Mat A;
	input.convertTo(A, CV_32FC1);

	cv::Mat meanY = cv::Mat(A.size(), CV_32FC1);
	for (int y = 0; y < meanY.rows; y++)
	{
		for(int x = 0; x < meanY.cols; x++)
		{
			meanY.at<float>(y, x) = y;
		}
	}
	return cv::sum(meanY.mul(A))[0];
}

/**
 * \brief	Calculates the covariance of the rows and cols 
 *		of the given image
 *
 * \param	input	The image to calculate the covariance of
 *
 * \retrun	The covariance of the given image
 */
float calcCovariance(const cv::Mat &input)
{
	//Convert input from unsigned char to float matrix
	cv::Mat A;
	input.convertTo(A, CV_32FC1);

	//calculate x mean
	float mx = calcXmean(input);

	//calculate y mean
	float my = calcYmean(input);

	//calculate (x - mx) * (y - my)
	cv::Mat tmp(A.size(), CV_32FC1);
	for (int y = 0; y < tmp.rows; y++)
	{
		for(int x = 0; x < tmp.cols; x++)
		{
			tmp.at<float>(y, x) = (x - mx) * (y - my);
		}
	}

	//calculate covariance
	return cv::sum(A.mul(tmp))[0];
}

/**
 * \brief	Calculates the correlation of the cols and rows
 *		of the given image
 *
 * \param	input		The image
 * \param	covariance	The precomputed covarianace
 *
 * \return	The correlation
 */
float calcCorrelation(const cv::Mat &input, float covariance)
{
	//Convert input from unsigned char to float matrix
	cv::Mat A;
	input.convertTo(A, CV_32FC1);

	//calculate x mean
	float mx = calcXmean(input);

	//calculate y mean
	float my = calcYmean(input);

	//calculate (x - mx)^2
	cv::Mat tmpX(A.size(), CV_32FC1);
	for (int y = 0; y < tmpX.rows; y++)
	{
		for(int x = 0; x < tmpX.cols; x++)
		{
			tmpX.at<float>(y, x) = (x - mx) * (x - my);
		}
	}
	//calculate (y - my)^2
	cv::Mat tmpY(A.size(), CV_32FC1);
	for (int y = 0; y < tmpY.rows; y++)
	{
		for(int x = 0; x < tmpY.cols; x++)
		{
			tmpY.at<float>(y, x) = (y - my) * (y - my);
		}
	}

	//calculate varianceX
	float varX = cv::sum(A.mul(tmpX))[0];
	//calculate varianceY
	float varY = cv::sum(A.mul(tmpY))[0];
	//calculate correlation
	return covariance/sqrt(varX*varY); 
}

/**
 * \brief	Calculates several statistical values for the given
 *		input image
 *
 * \param	input		The input image
 * \param	numValues	Variable to take the number of values
 *
 * \return	A vector of statistical values describing the image
 */
float* statisticalAnalysis(const cv::Mat &input, int &numValues)
{
	//We currently have 8 statistical values
	numValues = 8;

	//Allocate result vector
	float* result = new float[numValues];

	//calculate mean and standard deviation
	cv::Scalar mean, stddev;
	cv::meanStdDev(input, mean, stddev);

	//calculate mean
	result[0] = mean[0];

	//calculate variance
	result[1] = stddev[0] * stddev[0];

	//calculate energy
	result[2] = calcEnergy(input);

	//calculate inertia 
	result[3] = calcInertia(input);

	//calculate entropy
	result[4] = calcEntropy(input);
	
	//calculate homogeneity 
	result[5] = calcHomogeneity(input);

	//calculate covariance 
	result[6] = calcCovariance(input);
	
	//calculate correlation 
	result[7] = calcCorrelation(input, result[6]);

	//return the result vector
	return result;
}

/**
 * \brief	Calculates the distance of two texture vectors
 *
 * \param	v1	The first texture vector
 * \param	v2	The second texture vector
 * \param	nComps	The number of components of the texture
 *			vectors
 *
 * \return	The distance of the given texture vectors
 */
float textureVectorDistance(float* v1, float* v2, int nComps)
{
	float result = 0;
	for (int i = 0; i < nComps; i++)
	{
		result += fabs(v1[i] - v2[i]);	
	}
	return result;
}

int main (int argc, char** argv)
{
	if (argc == 4)
	{
		//number of colors to reduce the color space to.
		//This value has to be smaller than or equal to 256.
		int numColors = atoi(argv[3]);
		
		//input images
		cv::Mat img1 = cv::imread(argv[1], 0);
		cv::Mat img2 = cv::imread(argv[2], 0);

		int n = 0;	
		float* stats1 = statisticalAnalysis(img1, n);	
		float* stats2 = statisticalAnalysis(img2, n);	
		cout<<textureVectorDistance(stats1, stats2, n)<<endl;
		for(int i = 0; i < n; i++)
		{
			cout<<stats1[i]<<" "<<stats2[i]<<endl;
		}

		return EXIT_SUCCESS;			
	}
	else
	{
		cout<<"Usage: "<<argv[0]<<" <first image> <second image> <number of colors>"<<endl;
		return EXIT_FAILURE;
	}
}
