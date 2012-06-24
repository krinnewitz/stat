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
	if(img.channels() == 1)
	{
		cv::cvtColor(img, img, CV_GRAY2RGB);
	}
	//reduce the number of colors
	Imageprocessor::reduceColors(img, img, m_numColors);

	for(int direction = 0; direction < 4; direction++)
	{	
		//allocate output matrix
		float** result = new float*[m_numColors];
		for(int j = 0; j < m_numColors; j++)
		{
			result[j] = new float[m_numColors];
			memset(result[j], 0, m_numColors * sizeof(float));
		}

		int dx, dy;
		switch(direction)
		{
			case 0://0 degrees -> horizontal
				dx = 1;
				dy = 0;
				m_cooc0 = result;
				break;
			case 1://45 degrees -> diagonal
				dx = 1;
				dy = 1;
				m_cooc1 = result;
				break;
			case 2://90 degrees -> vertical
				dx = 0;
				dy = 1;
				m_cooc2 = result;
				break;
			case 3://135 degrees -> diagonal
				dx = -1;
				dy = 1;
				m_cooc3 = result;
				break
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
		for (int i = 0; i < m_numColors; i++) 
		{
			for(int j = 0; j < m_numColors; j++)
			{
				result[i][j] /= denom;
			}
		}
	}
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
