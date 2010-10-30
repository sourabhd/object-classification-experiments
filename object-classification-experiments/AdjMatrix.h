#ifndef _ADJMATRIX_H
#define _ADJMATRIX_H

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

#include <boost/numeric/ublas/vector.hpp>
#include <boost/numeric/ublas/triangular.hpp>
#include <boost/numeric/ublas/io.hpp>
#include <boost/numeric/ublas/matrix.hpp>
#include <boost/numeric/ublas/matrix_proxy.hpp>
#include <boost/numeric/ublas/io.hpp>
//#include "invert_matrix.hpp"

#include <fstream>
#include <iostream>
#include <string>
#include <cstdlib>
#include <vector>
#include <map>

#include <octave/oct.h>
#include <octave/Cell.h>
#include <octave/dRowVector.h>

//#include <armadillo>

using namespace std;
//using namespace arma;
namespace b = boost::numeric::ublas;

#define CERR std::cerr << __FILE__ << " : " << __LINE__ << " : " << __func__ <<  " :: "   
typedef enum ErrorCodes { INCORRECT_ARGS=1, IMG_LOAD_FAILED, FILE_OPEN_FAILED} err_t;
//const int64_t NDIM = 944;
typedef b::vector<double> vec_t;
typedef std::vector<vec_t> vvec_t;
typedef std::map<std::string,vec_t> mvec_t;
typedef std::map<std::string,vvec_t> mvvec_t;
typedef std::map<std::string,vvec_t>::iterator mvvecitr_t;
typedef b::matrix<double> tmat_t;
typedef b::matrix<double> mat_t;
//typedef std::vector<Matrix> vamat_t;
//typedef std::vector<RowVector> vrv_t;

//void createAdjMatrix(string infile);
//void mergeClusters(string infile);


#endif
