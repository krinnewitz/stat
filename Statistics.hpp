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
 * Statistics.hpp
 *
 *  @date 24.06.2012
 *  @author Kim Rinnewitz (krinnewitz@uos.de)
 */

#ifndef STATISTICS_HPP_
#define STATISTICS_HPP_

#include <cstring>
#include <opencv/highgui.h>
#include <opencv/cv.h>
#include <cstdio>

namespace lssr {


/**
 * @brief	This class provides statistical methods for texture analysis..
 */
class Statistics {
public:


	/**
	* \brief Constructor. Calculates the cooccurrence matrix for the given Texture.
	*
	* \param	t		The texture
	* \param	numColors	The number of gray levels to use
	*
	*/
	Statistics(Texture* t, int numColors)


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
	static float textureVectorDistance(float* v1, float* v2, int nComps);

	/**
	 * Destructor.
	 */
	virtual ~Statistics();

private:
	/**
	 * \brief	Returns the i-th entry of the magrginal probability matrix
	 *		of the given cooccurrence matrix
	 *
	 * \param	com		The cooccurrence matrix
	 * \param	i		The entry to get
	 *
	 *
	 */
	float px(float** com, int i);

	/**
	 * \brief	Returns the j-th entry of the magrginal probability matrix
	 *		of the given cooccurrence matrix
	 *
	 * \param	com		The cooccurrence matrix
	 * \param	j		The entry to get
	 *
	 *
	 */
	float py(float** com, int j);

	/**
	 * \brief	Calculates p_{x+y}(k)	
	 *		
	 *
	 * \param	com		The cooccurrence matrix
	 * \param	k		k
	 *
	 * \return 	p_{x+y}(k)
	 */
	float pxplusy(float** com,  int k);

	/**
	 * \brief	Calculates p_{x-y}(k)	
	 *		
	 *
	 * \param	com		The cooccurrence matrix
	 * \param	k		k
	 *
	 * \return 	p_{x-y}(k)
	 */
	float pxminusy(float** com, int k);

	//The number of rows and cols of the cooccurrence matrix
	int m_numColors;

	//cooccurrence matrix for 0 degrees direction
	float** m_cooc0;
	
	//cooccurrence matrix for 45 degrees direction
	float** m_cooc1;

	//cooccurrence matrix for 90 degrees direction
	float** m_cooc2;

	//cooccurrence matrix for 135 degrees direction
	float** m_cooc3;
};

}

#endif /* STATISTICS_HPP_ */