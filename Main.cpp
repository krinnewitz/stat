#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <opencv/cv.h>
#include <opencv/highgui.h>
#include <iostream>
#include <map>
#include "Statistics.hpp"

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
 * \brief	Calculates the energy of the given image
 *
 * \param	input	The image to calculate the energy of
 *
 * \retrun	The energy of the given image
 */
float calcEnergy(const cv::Mat &input)
{
	cv::Mat A;
	input.convertTo(A, CV_32FC1);
	cv::Mat sqr;
	cv::pow(A, 2, sqr);
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
	return covariance / (sqrt(varX) * sqrt(varY)); 
}



/**
 * \brief	Calculates several statistical values for the given
 *		input image
 *
 * \param	input		The input image
 * \param	numValues	Variable to take the number of values
 * \param	numColors	The number of gray levels to use 
 *
 * \return	A vector of statistical values describing the image
 */
float* statisticalAnalysis(const cv::Mat &input, int &numValues, int numColors)
{
	//We currently have 22 statistical values
	numValues = 22;

	//Allocate result vector
	float* result = new float[numValues];

//*****************statistical features of first order -> computed directly on the image***********************
	//calculate mean and standard deviation
	cv::Scalar mean, stddev;
	cv::meanStdDev(input, mean, stddev);

	//calculate mean
	result[0] = mean[0];

	//calculate variance
	result[1] = stddev[0] * stddev[0];

	//calculate energy
	result[2] = calcEnergy(input) / input.rows / input.cols;

	//calculate inertia 
	result[3] = calcInertia(input) / input.rows / input.cols;

	//calculate entropy
	result[4] = calcEntropy(input) / input.rows / input.cols;
	
	//calculate homogeneity 
	result[5] = calcHomogeneity(input) / input.rows / input.cols;

	//calculate covariance 
	result[6] = calcCovariance(input) / input.rows / input.cols;
	
	//calculate correlation 
	result[7] = calcCorrelation(input, result[6] * input.rows * input.cols);

//*****************statistical features of second order -> computed  on the image's cooccurrence matrix*****************

	lssr::Statistics* stats = new lssr::Statistics(input, numColors);

	//Angular second moment
	result[8]  = stats->calcASM();

	//constrast
	result[9] = stats->calcContrast();
	
	//correlation
	result[10] = stats->calcCorrelation();

	//Sum of squares: Variance
	result[11] = stats->calcSumOfSquares();

	//Inverse difference moment
	result[12] = stats->calcInverseDifference();
	
	//sum average
	result[13] = stats->calcSumAvg();

	//sum entropy
	result[14] = stats->calcSumEntropy();

	//sum variance
	result[15] = stats->calcSumVariance();

	//entropy
	result[16] = stats->calcEntropy();

	//difference variance
	result[17] = stats->calcDifferenceVariance();

	//difference entropy
	result[18] = stats->calcDifferenceEntropy();

	//information meeasures 1 of correlation
	result[19] = stats->calcInformationMeasures1();
	
	//information meeasures 2 of correlation
	result[20] = stats->calcInformationMeasures2();

	//Maximal correlation coefficient
	result[21] = stats->calcMaxCorrelationCoefficient();

	delete stats;

	//return the result vector
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
		float* stats1 = statisticalAnalysis(img1, n, numColors);	
		float* stats2 = statisticalAnalysis(img2, n, numColors);	
		cout<<lssr::Statistics::textureVectorDistance(stats1, stats2, n)<<endl;
		for(int i = 0; i < n; i++)
		{
			cout<<stats1[i]<<" "<<stats2[i]<<endl;
		}

		/*float** com = calcCooccurrenceMatrix(img1, numColors, 2);
		for(int i = 0; i < numColors; i++)
		{
			for (int j = 0; j < numColors; j++)
			{
				cout<<setw(14)<<com[i][j];
			}
			cout<<endl;
		}
*/
		return EXIT_SUCCESS;			
	}
	else
	{
		cout<<"Usage: "<<argv[0]<<" <first image> <second image> <number of colors>"<<endl;
		return EXIT_FAILURE;
	}
}
