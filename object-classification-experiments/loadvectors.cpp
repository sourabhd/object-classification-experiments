#include "AdjMatrix.h"
#include <octave/oct.h>
#include <octave/dRowVector.h>
#include <octave/Cell.h>

/**
 * @file
 * @author Sourabh Daptardar <saurabh.daptardar@gmail.com>
 * @version 1.0
 *
 * @section LICENSE
 * This file is part of SerrePoggioClassifier.
 * 
 * SerrePoggioClassifier is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 * 
 * SerrePoggioClassifier is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 * 
 * You should have received a copy of the GNU General Public License
 * along with SerrePoggioClassifier.  If not, see <http://www.gnu.org/licenses/>.
 *
 * @section DESCRIPTION
 * This code has been developed in partial fulfillment of the M.Tech thesis
 * "Explorations on a neurologically plausible model of image object classification"
 * by Sourabh Daptardar, Y7111009, CSE, IIT Kanpur.
 *
 * This code implements normalized graph cut clustering technique.
 */

////////////////////////////////////////////////////////////////////////////////

DEFUN_DLD(loadvectors, args, nargout, "Function to read C2 vectors")
{

	const int64_t NDIM = args(1).int_value();
	std::ifstream ifs(args(0).string_value().c_str());

	if(!ifs)
	{
		CERR << "Could not open file : " << args(0).string_value().c_str() << std::endl;
		exit(FILE_OPEN_FAILED);
	}

	int64_t id = -1;
	std::string category = "";
	vec_t v(NDIM);
	mvvec_t vvmap;
	while(true)
	{
		id = -1 ; category = "" ; v.clear();
		ifs >> id >> category >> v;
		if (ifs.eof())
		{
			break;
		}	
		vvmap[category].push_back(v);
	}	

	mvvecitr_t mend = vvmap.end();
	octave_value_list ovl; 
	for ( mvvecitr_t itr = vvmap.begin() ; itr != mend ; ++itr )
	{
		const vvec_t &X = vvmap[itr->first];
		int64_t sz = (itr->second).size();
		for ( int32_t i = 0 ; i < sz ; ++i )
		{
			RowVector R(NDIM);
			for ( int32_t j = 0 ; j < NDIM ; ++j )
			{
				R(j) = X[i][j];
			}
			ovl(i) = octave_value(R);
		}
	} 

	ifs.close();

	Cell V(ovl); 
	return octave_value(V);
}

