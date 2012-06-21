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
 * \brief Calculates the cooccurrence matrix for the given texture
 *
 * \param	input		The texture
 * \param	numColors	The number of gray levels to use
 * \param	direction	The direction to calculate the
 *				cooccurrence matrix for
 *
 * \return	The normalized cooccurrence matrix of size
 *		numColors x numColors
 */
float**  calcCooccurrenceMatrix(const cv::Mat &input, int numColors, unsigned char direction)
{
	
	//reduce the number of colors
	cv::Mat img = input;
	if(input.channels() == 1)
	{
		cv::cvtColor(input, img, CV_GRAY2RGB);
	}
	reduceColors(img, img, numColors);
	
	
	int dx, dy;
	
	//0 degrees -> horizontal
	if (direction == 0)
	{
		dx = 1;
		dy = 0;
	}
	//45 degrees -> diagonal
	if (direction == 1)
	{
		dx = 1;
		dy = 1;
	}
	//90 degrees -> vertical
	if(direction == 2)
	{
		dx = 0;
		dy = 1;
	}
	//135 degrees -> diagonal
	if (direction >=3)
	{
		dx = -1,
		dy =  1;
	}

	//allocate output matrix
	float** result = new float*[numColors];
	for(int j = 0; j < numColors; j++)
	{
		result[j] = new float[numColors];
		memset(result[j], 0, numColors * sizeof(float));
	}

	//calculate cooccurrence matrix
	for (int y = 0; y < img.rows; y++)
	{
		for(int x = 0; x < img.cols; x++)
		{
			if (x + dx >= 0 && x + dx < img.cols && y + dy >= 0 && y + dy < img.rows)
			{
				result[img.at<unsigned char>(y,x)][img.at<unsigned char>(y+dy,x+dx)]++;
			}
			if (x - dx >= 0 && x - dx < img.cols && y - dy >= 0 && y - dy < img.rows)
			{
				result[img.at<unsigned char>(y,x)][img.at<unsigned char>(y-dy,x-dx)]++;
			}
		}
	}

	
	//normalize cooccurrence matrix
	float denom = 1;
	if (direction == 0 || direction == 2)
	{
		denom = 2 * img.rows * (img.cols -1);
	}
	else
	{
		denom = 2 * (img.rows - 1) * (img.cols - 1);
	}
	for (int i = 0; i < numColors; i++) 
	{
		for(int j = 0; j < numColors; j++)
		{
			result[i][j] /= denom;
		}
	}
		
	return result;
}

/**
 * \brief	Returns the i-th entry of the magrginal probability matrix
 *		of the given cooccurrence matrix
 *
 * \param	com		The cooccurrence matrix
 * \param	numColors	The number of rows and cols of the com
 * \param	i		The entry to get
 *
 *
 */
float px(float** com, int numColors, int i)
{
	float result = 0;
	for (int j = 0; j < numColors; j++)
	{
		result += com[i][j];
	}
}


/**
 * \brief	Returns the j-th entry of the magrginal probability matrix
 *		of the given cooccurrence matrix
 *
 * \param	com		The cooccurrence matrix
 * \param	numColors	The number of rows and cols of the com
 * \param	j		The entry to get
 *
 *
 */
float py(float** com, int numColors, int j)
{
	float result = 0;
	for (int i = 0; i < numColors; i++)
	{
		result += com[i][j];
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
 * \brief	Calculates p_{x+y}(k)	
 *		
 *
 * \param	com		The cooccurrence matrix
 * \param	numColors	The number of colors
 * \param	k		k
 *
 * \return 	p_{x+y}(k)
 */
float pxplusy(float** com, int numColors, int k)
{
	float result = 0;
	for (int i = 0; i < numColors; i++)
	{
		for (int j = 0; j < numColors; j++)
		{
			if (i + j == k)
			{
				result += com[i][j];
			}
		}
	}
}

/**
 * \brief	Calculates p_{x-y}(k)	
 *		
 *
 * \param	com		The cooccurrence matrix
 * \param	numColors	The number of colors
 * \param	k		k
 *
 * \return 	p_{x-y}(k)
 */
float pxminusy(float** com, int numColors, int k)
{
	float result = 0;
	for (int i = 0; i < numColors; i++)
	{
		for (int j = 0; j < numColors; j++)
		{
			if (abs(i - j) == k)
			{
				result += com[i][j];
			}
		}
	}
}

/**
 * \brief	Calculates the angular second moment of the texture
 *		represented by the given cooccurrence matrix
 *
 * \param	com		The cooccurrence matrix
 * \param	numColors	The number of colors
 *
 * \return 	The angular second moment
 */
float calcASM(float** com, int numColors)
{
	float result = 0;
	for (int i = 0; i < numColors; i++)
	{
		for (int j = 0; j < numColors; j++)
		{
			result += com[i][j] * com[i][j];
		}
	}
}

/**
 * \brief	Calculates the contrast of the texture
 *		represented by the given cooccurrence matrix
 *
 * \param	com		The cooccurrence matrix
 * \param	numColors	The number of colors
 *
 * \return 	The contrast of the texture
 */
float calcContrast(float** com, int numColors)
{
	float result = 0;
	for (int n = 0; n < numColors; n++)
	{
		for (int i = 0; i < numColors; i++)
		{
			for (int j = 0; j < numColors; j++)
			{
				if (abs(i-j) == n)
				{
					result += n * n * com[i][j];
				}
			}
		}
	}
}


/**
 * \brief	Calculates the correlation of the texture
 *		represented by the given cooccurrence matrix
 *
 * \param	com		The cooccurrence matrix
 * \param	numColors	The number of colors
 *
 * \return 	The correlation of the texture
 */
float calcCorrelation(float** com, int numColors)
{
	float ux = 0, uy = 0, sx = 0, sy = 0;

	//calculate means of px and py
	for (int i = 0; i < numColors; i++)
	{
		ux += px(com, numColors, i) / numColors;
		uy += py(com, numColors, i) / numColors;
	}
	//calculate standard deviations of px and py
	for (int i = 0; i < numColors; i++)
	{
		sx += (px(com, numColors, i) - ux) * (px(com, numColors, i) - ux);
		sy += (py(com, numColors, i) - uy) * (py(com, numColors, i) - uy);
	}
	sx = sqrt(sx);
	sy = sqrt(sy);
		
	//calculate correlatione
	float result = 0;
	for (int i = 0; i < numColors; i++)
	{
		for (int j = 0; j < numColors; j++)
		{
			result += (i * j * com[i][j] - ux * uy) / (sx * sy);
		}
	}
	return result;
}

/**
 * \brief	Calculates the sum of squares of the texture
 *		represented by the given cooccurrence matrix
 *
 * \param	com		The cooccurrence matrix
 * \param	numColors	The number of colors
 *
 * \return 	The sum of squares of the texture
 */
float calcSumOfSquares(float** com, int numColors)
{
	float u = 0;

	//calculate mean of the com
	for (int i = 0; i < numColors; i++)
	{
		for (int j = 0; j < numColors; j++)
		{
			u += com[i][j] / (numColors * numColors);
		}
	}
		
	//calculate sum of squares : variance
	float result = 0;
	for (int i = 0; i < numColors; i++)
	{
		for (int j = 0; j < numColors; j++)
		{
			result += (i - u) * (i - u) * com[i][j];
		}
	}
	return result;
}

/**
 * \brief	Calculates the inverse difference moment of the texture
 *		represented by the given cooccurrence matrix
 *
 * \param	com		The cooccurrence matrix
 * \param	numColors	The number of colors
 *
 * \return 	The inverse difference moment of the texture
 */
float calcInverseDifference(float** com, int numColors)
{
	//calculate inverse difference moment
	float result = 0;
	for (int i = 0; i < numColors; i++)
	{
		for (int j = 0; j < numColors; j++)
		{
			result += 1 / ( 1 + (i - j) * (i - j)) * com[i][j];
		}
	}
	return result;
}


/**
 * \brief	Calculates the sum average of the texture
 *		represented by the given cooccurrence matrix
 *
 * \param	com		The cooccurrence matrix
 * \param	numColors	The number of colors
 *
 * \return 	The sum average of the texture
 */
float calcSumAvg(float** com, int numColors)
{
	//calculate sum average
	float result = 0;
	for (int i = 0; i < 2 * numColors - 1; i++)
	{
		result += i * pxplusy(com, numColors, i);
	}
	return result;
}


/**
 * \brief	Calculates the sum entropy of the texture
 *		represented by the given cooccurrence matrix
 *
 * \param	com		The cooccurrence matrix
 * \param	numColors	The number of colors
 *
 * \return 	The sum entropy of the texture
 */
float calcSumEntropy(float** com, int numColors)
{
	const float epsilon = 0.000001;

	//calculate sum entropy
	float result = 0;
	for (int i = 0; i < 2 * numColors - 1; i++)
	{
		float p = pxplusy(com, numColors, i);
		result +=  p * log(p + epsilon);
	}
	result *= -1;
	return result;
}

/**
 * \brief	Calculates the sum variance of the texture
 *		represented by the given cooccurrence matrix
 *
 * \param	com		The cooccurrence matrix
 * \param	numColors	The number of colors
 *
 * \return 	The sum variance of the texture
 */
float calcSumVariance(float** com, int numColors)
{
	//calculate sum variance
	float result = 0;
	float sumEntropy = calcSumEntropy(com, numColors);
	for (int i = 0; i < 2 * numColors - 1; i++)
	{
		result += (i - sumEntropy) * (i - sumEntropy) * pxplusy(com, numColors, i);
	}
	return result;
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
	//We currently have 40 statistical values
	numValues = 40;

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
	//calculate the cooccurrence matrix in horizontal direction
	float** com0 = calcCooccurrenceMatrix(input, numColors, 0);
	//calculate the cooccurrence matrix in diagonal direction
	float** com1 = calcCooccurrenceMatrix(input, numColors, 1);
	//calculate the cooccurrence matrix in vertical direction
	float** com2 = calcCooccurrenceMatrix(input, numColors, 2);
	//calculate the cooccurrence matrix in diagonal direction
	float** com3 = calcCooccurrenceMatrix(input, numColors, 3);


//TODO: we should use average and range values later on for rotation invariance in case of 45, 90 and 135 degree rotations
	//Angular second moment
	result[8]  = calcASM(com0, numColors);
	result[9]  = calcASM(com1, numColors);
	result[10] = calcASM(com2, numColors);
	result[11] = calcASM(com3, numColors);

	//constrast
	result[12] = calcContrast(com0, numColors);
	result[13] = calcContrast(com1, numColors);
	result[14] = calcContrast(com2, numColors);
	result[15] = calcContrast(com3, numColors);
	
	//correlation
	result[16] = calcCorrelation(com0, numColors);
	result[17] = calcCorrelation(com1, numColors);
	result[18] = calcCorrelation(com2, numColors);
	result[19] = calcCorrelation(com3, numColors);

	//Sum of squares: Variance
	result[20] = calcSumOfSquares(com0, numColors);
	result[21] = calcSumOfSquares(com1, numColors);
	result[22] = calcSumOfSquares(com2, numColors);
	result[23] = calcSumOfSquares(com3, numColors);

	//Inverse difference moment
	result[24] = calcInverseDifference(com0, numColors);
	result[25] = calcInverseDifference(com1, numColors);
	result[26] = calcInverseDifference(com2, numColors);
	result[27] = calcInverseDifference(com3, numColors);
	
	//sum average
	result[28] = calcSumAvg(com0, numColors);
	result[29] = calcSumAvg(com1, numColors);
	result[30] = calcSumAvg(com2, numColors);
	result[31] = calcSumAvg(com3, numColors);

	//sum entropy
	result[32] = calcSumEntropy(com0, numColors);
	result[33] = calcSumEntropy(com3, numColors);
	result[34] = calcSumEntropy(com2, numColors);
	result[35] = calcSumEntropy(com3, numColors);

	//sum variance
	result[36] = calcSumVariance(com0, numColors);
	result[37] = calcSumVariance(com3, numColors);
	result[38] = calcSumVariance(com2, numColors);
	result[39] = calcSumVariance(com3, numColors);

	//entropy
	//TODO

	//difference variance
	//TODO

	//difference entropy
	//TODO

	//information meeasures of correlation
	//TODO
	
	//information meeasures of correlation
	//TODO

	//Maximal correlation coefficient
	//TODO

	//free the cooccurrence matrix
	for (int i = 0; i < numColors; i++)
	{
		delete[] com0[i];
		delete[] com1[i];
		delete[] com2[i];
		delete[] com3[i];
	}
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
		result += fabs(v1[i] - v2[i]) / max(1.0f, max(v1[i], v2[i]));	
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
		float* stats1 = statisticalAnalysis(img1, n, numColors);	
		float* stats2 = statisticalAnalysis(img2, n, numColors);	
		cout<<textureVectorDistance(stats1, stats2, n)<<endl;
		for(int i = 0; i < n; i++)
		{
			cout<<stats1[i]<<" "<<stats2[i]<<endl;
		}

		float** com = calcCooccurrenceMatrix(img1, numColors, 2);
		for(int i = 0; i < numColors; i++)
		{
			for (int j = 0; j < numColors; j++)
			{
				cout<<setw(14)<<com[i][j];
			}
			cout<<endl;
		}

		return EXIT_SUCCESS;			
	}
	else
	{
		cout<<"Usage: "<<argv[0]<<" <first image> <second image> <number of colors>"<<endl;
		return EXIT_FAILURE;
	}
}
