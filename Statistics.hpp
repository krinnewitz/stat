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
#include <cstdio>

namespace lssr {


/**
 * @brief	This class provides statistical methods for texture analysis..
 */
class Statistics {
public:


	/**
	 * @brief 	Constructor.
	 *
	 */
	Statistics( );

	/**
	 * Destructor.
	 */
	virtual ~Statistics();
};

}

#endif /* STATISTICS_HPP_ */
