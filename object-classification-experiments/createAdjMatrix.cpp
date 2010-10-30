#include "AdjMatrix.h"
#include <octave/oct.h>

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

DEFUN_DLD(createAdjMatrix, args, nargout, "Function to create adjacency matrix")
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

	int64_t s = (vvmap.begin()->second).size();
	Matrix t(s,s);
	for ( mvvecitr_t itr = vvmap.begin() ; itr != mend ; ++itr )
	{
		const vvec_t &X = vvmap[itr->first];
		int64_t sz = (itr->second).size();

		//tmat_t t(sz,sz);
		//Matrix t(sz,sz);

		//t = b::identity_matrix<double>(sz,sz);
		for ( int32_t i = 0 ; i < sz ; ++i )
		{
			for ( int32_t j = 0 ; j <= i ; ++j )
			{
				vec_t d = X[i] - X[j];
				t(i,j) = t(j,i) = exp(-1.0 * norm_2(d));		
			}
		}

		//std:: cout << itr->first << " :: " << score << endl;
		//cerr << t << endl;

	} 

	ifs.close();

	return octave_value(t);
}

////////////////////////////////////////////////////////////////////////////////




////////////////////////////////////////////////////////////////////////////////
#if 0
int main(int argc,char *argv[])
{

	if ( argc !=  3 )
	{
		std::cerr << "Usage: ./anyl <inputfile> <outputfile>" << endl;
		exit(INCORRECT_ARGS);
	}

	std::string infile(argv[1]);
	std::string outfile(argv[2]);

	createAdjMatrix(infile);

	return 0;
}

#endif
