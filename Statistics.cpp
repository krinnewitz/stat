/* Copyright (C) 2011 Uni Osnabrück
 * This file is part of the LAS VEGAS Reconstruction Toolkit,
 *
 * LAS VEGAS is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 2 of the License, or
 * (at your option) any later version.
 *
 * LAS VEGAS is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with this program; if not, write to the Free Software
 * Foundation, Inc., 59 Temple Place - Suite 330, Boston, MA  02111-1307, USA
 */


/*
 * Statistics.cpp
 *
 *  @date 24.06.2012
 *  @author Kim Rinnewitz (krinnewitz@uos.de)
 */

#include "Statistics.hpp"
#include "ImageProcessor.hpp"

namespace lssr {

float Statistics::epsilon = 0.0000001;

Statistics::Statistics(Texture* t, int numColors)
{
	m_numColors = numColors;

	//convert texture to cv::Mat
	cv::Mat img(cv::Size(t->m_width, t->m_height), CV_MAKETYPE(t->m_numBytesPerChan * 8, t->m_numChannels), t->m_data);
	
	calcCooc(img);
}

Statistics::Statistics(const cv::Mat &t, int numColors)
{
	m_numColors = numColors;
	calcCooc(t);
}

float Statistics::textureVectorDistance(float* v1, float* v2, int nComps)
{
	float result = 0;
	for (int i = 0; i < nComps; i++)
	{
		result += fabs(v1[i] - v2[i]) / max(1.0f, max(v1[i], v2[i]));	
	}
	return result;
}

Statistics::~Statistics() {
	//free the cooccurrence matrix
	for (int i = 0; i < m_numColors; i++)
	{
		delete[] cooc0[i];
		delete[] cooc1[i];
		delete[] cooc2[i];
		delete[] cooc3[i];
	}
}

float calcASM()
{
	float result = 0;
	for (int direction = 0; direction < 4; direction++)
	{
		float** com = 0;
		switch(direction)
		{
			case 0:
				com = m_cooc0;
				break;
			case 1:
				com = m_cooc1;
				break;
			case 2:
				com = m_cooc2;
				break;
			case 3:
				com = m_cooc3;
				break;
		}
			
		for (int i = 0; i < m_numColors; i++)
		{
			for (int j = 0; j < m_numColors; j++)
			{
				result += com[i][j] * com[i][j];
			}
		}
	}
	return result / 4;
}

float calcContrast()
{
	float result = 0;
	for (int direction = 0; direction < 4; direction++)
	{
		float** com = 0;
		switch(direction)
		{
			case 0:
				com = m_cooc0;
				break;
			case 1:
				com = m_cooc1;
				break;
			case 2:
				com = m_cooc2;
				break;
			case 3:
				com = m_cooc3;
				break;
		}
			
		for (int n = 0; n < m_numColors; n++)
		{
			for (int i = 0; i < m_numColors; i++)
			{
				for (int j = 0; j < m_numColors; j++)
				{
					if (abs(i-j) == n)
					{
						result += n * n * com[i][j];
					}
				}
			}
		}
	}
	return result / 4;
}


float calcCorrelation()
{
	float result = 0;
	for (int direction = 0; direction < 4; direction++)
	{
		float** com = 0;
		switch(direction)
		{
			case 0:
				com = m_cooc0;
				break;
			case 1:
				com = m_cooc1;
				break;
			case 2:
				com = m_cooc2;
				break;
			case 3:
				com = m_cooc3;
				break;
		}
			
		float ux = 0, uy = 0, sx = 0, sy = 0;

		//calculate means of px and py
		for (int i = 0; i < m_numColors; i++)
		{
			ux += px(com, m_numColors, i) / m_numColors;
			uy += py(com, m_numColors, i) / m_numColors;
		}
		//calculate standard deviations of px and py
		for (int i = 0; i < m_numColors; i++)
		{
			sx += (px(com, m_numColors, i) - ux) * (px(com, m_numColors, i) - ux);
			sy += (py(com, m_numColors, i) - uy) * (py(com, m_numColors, i) - uy);
		}
		sx = sqrt(sx);
		sy = sqrt(sy);
			
		//calculate correlation
		for (int i = 0; i < m_numColors; i++)
		{
			for (int j = 0; j < m_numColors; j++)
			{
				result += (i * j * com[i][j] - ux * uy) / (sx * sy);
			}
		}
	}
	return result / 4;
}

float calcSumOfSquares()
{
	float result = 0;
	for (int direction = 0; direction < 4; direction++)
	{
		float** com = 0;
		switch(direction)
		{
			case 0:
				com = m_cooc0;
				break;
			case 1:
				com = m_cooc1;
				break;
			case 2:
				com = m_cooc2;
				break;
			case 3:
				com = m_cooc3;
				break;
		}
		float u = 0;

		//calculate mean of the com
		for (int i = 0; i < m_numColors; i++)
		{
			for (int j = 0; j < m_numColors; j++)
			{
				u += com[i][j] / (m_numColors * m_numColors);
			}
		}
			
		//calculate sum of squares : variance
		for (int i = 0; i < m_numColors; i++)
		{
			for (int j = 0; j < m_numColors; j++)
			{
				result += (i - u) * (i - u) * com[i][j];
			}
		}
	}
	return result / 4;
}

float calcInverseDifference()
{
	//calculate inverse difference moment
	float result = 0;
	for (int direction = 0; direction < 4; direction++)
	{
		float** com = 0;
		switch(direction)
		{
			case 0:
				com = m_cooc0;
				break;
			case 1:
				com = m_cooc1;
				break;
			case 2:
				com = m_cooc2;
				break;
			case 3:
				com = m_cooc3;
				break;
		}
		for (int i = 0; i < m_numColors; i++)
		{
			for (int j = 0; j < m_numColors; j++)
			{
				result += 1 / ( 1 + (i - j) * (i - j)) * com[i][j];
			}
		}
	}
	return result / 4;
}

float calcSumAvg()
{
	//calculate sum average
	float result = 0;
	for (int direction = 0; direction < 4; direction++)
	{
		float** com = 0;
		switch(direction)
		{
			case 0:
				com = m_cooc0;
				break;
			case 1:
				com = m_cooc1;
				break;
			case 2:
				com = m_cooc2;
				break;
			case 3:
				com = m_cooc3;
				break;
		}
		for (int i = 0; i < 2 * m_numColors - 1; i++)
		{
			result += i * pxplusy(com, m_numColors, i);
		}
	}
	return result / 4;
}

float calcSumEntropy()
{
	//calculate sum entropy
	float result = 0;
	for (int direction = 0; direction < 4; direction++)
	{
		float** com = 0;
		switch(direction)
		{
			case 0:
				com = m_cooc0;
				break;
			case 1:
				com = m_cooc1;
				break;
			case 2:
				com = m_cooc2;
				break;
			case 3:
				com = m_cooc3;
				break;
		}
		for (int i = 0; i < 2 * m_numColors - 1; i++)
		{
			float p = pxplusy(com, m_numColors, i);
			result +=  p * log(p + Statistics::epsilon);
		}
	}
	return result / -4;
}

float calcSumVariance()
{
	//calculate sum variance
	float result = 0;
	for (int direction = 0; direction < 4; direction++)
	{
		float** com = 0;
		switch(direction)
		{
			case 0:
				com = m_cooc0;
				break;
			case 1:
				com = m_cooc1;
				break;
			case 2:
				com = m_cooc2;
				break;
			case 3:
				com = m_cooc3;
				break;
		}
		float sumEntropy = calcSumEntropy(com, m_numColors);
		for (int i = 0; i < 2 * m_numColors - 1; i++)
		{
			result += (i - sumEntropy) * (i - sumEntropy) * pxplusy(com, m_numColors, i);
		}
	}
	return result / 4;
}

float calcEntropy()
{
	//calculate entropy
	float result = 0;
	for (int direction = 0; direction < 4; direction++)
	{
		float** com = 0;
		switch(direction)
		{
			case 0:
				com = m_cooc0;
				break;
			case 1:
				com = m_cooc1;
				break;
			case 2:
				com = m_cooc2;
				break;
			case 3:
				com = m_cooc3;
				break;
		}
		for (int i = 0; i < m_numColors; i++)
		{
			for (int j = 0; j < m_numColors; j++)
			{
				result += com[i][j] * log(com[i][j] + Statistics::epsilon);
			}
		}
	}
	return result / -4;
}

float calcDifferenceVariance()
{
	float result = 0;
	for (int direction = 0; direction < 4; direction++)
	{
		float** com = 0;
		switch(direction)
		{
			case 0:
				com = m_cooc0;
				break;
			case 1:
				com = m_cooc1;
				break;
			case 2:
				com = m_cooc2;
				break;
			case 3:
				com = m_cooc3;
				break;
		}
		float u = 0;

		//calculate mean of pxminusy
		for (int i = 0; i < m_numColors; i++)
		{
			u += pxminusy(com, m_numColors, i) / m_numColors;
		}

		//calculate difference variance
		for (int i = 0; i < m_numColors; i++)
		{
			result += (pxminusy(com, m_numColors, i) - u) * (pxminusy(com, m_numColors, i) - u);
		}
	}
	return result / 4;
}

float calcDifferenceEntropy()
{
	//calculate difference entropy
	float result = 0;
	for (int direction = 0; direction < 4; direction++)
	{
		float** com = 0;
		switch(direction)
		{
			case 0:
				com = m_cooc0;
				break;
			case 1:
				com = m_cooc1;
				break;
			case 2:
				com = m_cooc2;
				break;
			case 3:
				com = m_cooc3;
				break;
		}
		for (int i = 0; i < m_numColors; i++)
		{
			float p = pxminusy(com, m_numColors, i);
			result +=  p * log(p + Statistics::epsilon);
		}
	}
	return result / -4;
}

float calcInformationMeasures1()
{
	float result = 0;
	for (int direction = 0; direction < 4; direction++)
	{
		float** com = 0;
		switch(direction)
		{
			case 0:
				com = m_cooc0;
				break;
			case 1:
				com = m_cooc1;
				break;
			case 2:
				com = m_cooc2;
				break;
			case 3:
				com = m_cooc3;
				break;
		}
		//calculate HXY1
		float HXY1 = 0;
		for (int i = 0; i < m_numColors; i++)
		{
			for (int j = 0; j < m_numColors; j++)
			{
				HXY1 += com[i][j] * log(px(com, m_numColors, i) * py(com, m_numColors, j) + Statistics::epsilon);
			}
		}
		HXY1 *= -1;

		//calculate HX and HY
		float HX = 0, HY = 0;
		for (int i = 0; i < m_numColors; i++)
		{
			HX += px(com, m_numColors, i) * log(px(com, m_numColors, i) + Statistics::epsilon);
			HY += py(com, m_numColors, i) * log(py(com, m_numColors, i) + Statistics::epsilon);
		}
		HX *= -1;
		HY *= -1;

		//calculate HXY
		float HXY = calcEntropy(com, m_numColors);

		//calculate  information measures 1
		result += (HXY - HXY1) / max(HX, HY);
	}

	return result / 4;
}

float calcInformationMeasures2()
{
	float result = 0;
	for (int direction = 0; direction < 4; direction++)
	{
		float** com = 0;
		switch(direction)
		{
			case 0:
				com = m_cooc0;
				break;
			case 1:
				com = m_cooc1;
				break;
			case 2:
				com = m_cooc2;
				break;
			case 3:
				com = m_cooc3;
				break;
		}
		//calculate HXY2
		float HXY2 = 0;
		for (int i = 0; i < m_numColors; i++)
		{
			for (int j = 0; j < m_numColors; j++)
			{
				HXY2 += px(com, m_numColors, i) * py(com, m_numColors, j) * log(px(com, m_numColors, i) * py(com, m_numColors, j) + Statistics::epsilon);
			}
		}
		HXY2 *= -1;

		//calculate HXY
		float HXY = calcEntropy(com, m_numColors);

		//calculate  information measures 1
		result += sqrt(1 - exp(-2.0 * (HXY2 - HXY)));
	}
	return result / 4;
}

float calcMaxCorrelationCoefficient()
{
	double result = 0;
	for (int direction = 0; direction < 4; direction++)
	{
		float** com = 0;
		switch(direction)
		{
			case 0:
				com = m_cooc0;
				break;
			case 1:
				com = m_cooc1;
				break;
			case 2:
				com = m_cooc2;
				break;
			case 3:
				com = m_cooc3;
				break;
		}
		//calculate Q
		cv::Mat Q(m_numColors, m_numColors, CV_64FC1);
		for (int i = 0; i < m_numColors; i++)
		{
			for(int j = 0; j < m_numColors; j++)
			{
				Q.at<double>(i,j) = 0;
				for (int k = 0; k < m_numColors; k++)
				{
					Q.at<double>(i,j) += (com[i][k] * com[j][k]) / (px(com, m_numColors, i) * py(com, m_numColors, k) + Statistics::epsilon);
				}
			}
		}
		
		//get the second largest eigenvalue of Q
		cv::Mat E, V;
		cv::eigen(Q, E, V);
		result += sqrt(E.at<double>(1,0));
	}
	return result / 4;
}

void Statistics::calcCooc(const cv::Mat &t)
{
	cv::Mat img(t);
	if(img.channels() == 1)
	{
		cv::cvtColor(img, img, CV_GRAY2RGB);
	}
	//reduce the number of colors
	Imageprocessor::reduceColors(img, img, m_numColors);

	for(int direction = 0; direction < 4; direction++)
	{	
		//allocate output matrix
		float** cooc = new float*[m_numColors];
		for(int j = 0; j < m_numColors; j++)
		{
			cooc[j] = new float[m_numColors];
			memset(cooc[j], 0, m_numColors * sizeof(float));
		}

		int dx, dy;
		switch(direction)
		{
			case 0://0 degrees -> horizontal
				dx = 1;
				dy = 0;
				m_cooc0 = cooc;
				break;
			case 1://45 degrees -> diagonal
				dx = 1;
				dy = 1;
				m_cooc1 = cooc;
				break;
			case 2://90 degrees -> vertical
				dx = 0;
				dy = 1;
				m_cooc2 = cooc;
				break;
			case 3://135 degrees -> diagonal
				dx = -1;
				dy = 1;
				m_cooc3 = cooc;
				break
		}	


		//calculate cooccurrence matrix
		for (int y = 0; y < img.rows; y++)
		{
			for(int x = 0; x < img.cols; x++)
			{
				if (x + dx >= 0 && x + dx < img.cols && y + dy >= 0 && y + dy < img.rows)
				{
					cooc[img.at<unsigned char>(y,x)][img.at<unsigned char>(y+dy,x+dx)]++;
				}
				if (x - dx >= 0 && x - dx < img.cols && y - dy >= 0 && y - dy < img.rows)
				{
					cooc[img.at<unsigned char>(y,x)][img.at<unsigned char>(y-dy,x-dx)]++;
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
		for (int i = 0; i < m_numColors; i++) 
		{
			for(int j = 0; j < m_numColors; j++)
			{
				cooc[i][j] /= denom;
			}
		}
	}
}

float Statistics::px(float** com, int i)
{
	float result = 0;
	for (int j = 0; j < m_numColors; j++)
	{
		result += com[i][j];
	}
}

float Statistics::py(float** com, int j)
{
	float result = 0;
	for (int i = 0; i < m_numColors; i++)
	{
		result += com[i][j];
	}
}

float Statistics::pxplusy(float** com, int k)
{
	float result = 0;
	for (int i = 0; i < m_numColors; i++)
	{
		for (int j = 0; j < m_numColors; j++)
		{
			if (i + j == k)
			{
				result += com[i][j];
			}
		}
	}
}

float Statistics::pxminusy(float** com, int k)
{
	float result = 0;
	for (int i = 0; i < m_numColors; i++)
	{
		for (int j = 0; j < m_numColors; j++)
		{
			if (abs(i - j) == k)
			{
				result += com[i][j];
			}
		}
	}
}

}
