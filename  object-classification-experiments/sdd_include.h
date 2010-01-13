#ifndef __SDD_INCLUDE
#define __SDD_INCLUDE

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
 * This code implements utility functions. 
 *
 */


#include <iostream>
#include <fstream>
#include <ctime>
#include <cstdlib>
#include <cerrno>
#include <vector>
#include <valarray>
#include <complex>
#include <map>
#include <string>
#include <sstream>
#include <list>
#include <queue>
#include <set>
#include <algorithm>
#include <iomanip>
#include <cassert>
#include <typeinfo>


extern "C" {
#include <sys/time.h>
#include <stdint.h>
}

using namespace std;

#define ERR std::cerr << __FILE__ << " : " << __LINE__ << " : " << __func__ << " :: "  

///////////////////// DYNAMIC 2D ARRAYS ////////////////////////////////////////
template<class T>
static inline T** malloc2D(int64_t rows,int64_t cols)
{
	T **t = new T*[rows];
	for(int64_t i = 0 ; i < rows ; ++i)
	{
		t[i] = new T[cols];
	}
	return t;
}	

////////////////////////////////////////////////////////////////////////////////
template<class T>
static inline void free2D(T **t,int64_t rows)
{
	for(int64_t i = 0 ; i < rows ; ++i)
	{
		delete [] t[i];
	}
	delete [] t;
}	


////////////////////////////////////////////////////////////////////////////////
////////////////////// 2D POINTS ///////////////////////////////////////////////

typedef std::complex<double> point2D;

static inline double euclid(point2D a, point2D b)
{
	return sqrt(norm(a-b));
}	

////////////////////////////////////////////////////////////////////////////////
////////////////////////////// FACTORIAL ///////////////////////////////////////

static inline int64_t fact(int64_t n)
{
	assert( n < 26) ; // 26! onwards will cause overflow
	int64_t product = 1;
	for(int64_t i = 2 ; i <= n ; i++)
	{
		product *= i;
	}

	return product;
}

////////////////////////////////////////////////////////////////////////////////

static inline double dfact(int64_t n)
{
	double product = 1;
	for(int64_t i = 2 ; i <= n ; i++)
	{
		product *= double(i);
	}

	return product;
}

////////////////////////////////////////////////////////////////////////////////
//////////////////////////// TIME MESUREMENT ///////////////////////////////////


#define TIME(t1) timeval t1 ; gettimeofday(&t1,0); 

static inline double difftime(timeval &t1, timeval &t2 ) // t2 - t1 
{
	return (
			( ( double(t2.tv_sec) * 1000.0 ) + ( double(t2.tv_usec) / 1000.0 ) )
			-
			( ( double(t1.tv_sec) * 1000.0 ) + ( double(t1.tv_usec) / 1000.0 ) )
		   );

}

////////////////////////////////////////////////////////////////////////////////
/////////////////////// RANDOM NUMBERS /////////////////////////////////////////

const int64_t tworaisedto32 = 0x1uL<< 32;

template <class T>
class Random
{
	drand48_data buffer;	
	T ulimit;
	T llimit;
	long int seedval;
	T result;
	double tresult;
	std::string type;
	public:

	T nextRandom(void)
	{
		if ( type == "d" || type == "l" )
		{
			drand48_r(&buffer,&tresult);
			result = T( double(llimit) + ( double(ulimit) - double(llimit) ) * double(tresult));
		}
		else
		{
			ERR << "Typeid received: " << type.c_str() << " .It is not supported" << endl;
		}
		return result;
	}

	Random(T ulimit_=1,T llimit_=0,long int seedval_=0):ulimit(ulimit_),llimit(llimit_),seedval(seedval_)
	{
		type = typeid(ulimit).name();
		assert( type == "d" || type == "l");
		if ( seedval_ == 0 )
		{
			TIME(tv);
			seedval = tv.tv_sec * tv.tv_usec;
		}
		srand48_r(seedval,&buffer);
	}



};


////////////////////////////////////////////////////////////////////////////////
/////////////////////// HOSTNAME ///////////////////////////////////////////////
static const int32_t __HostNameLength = 200;
static std::string __Hostname__ = "";

static inline const std::string& hostname()
{
	if ( __Hostname__ == "" )
	{
		char HostName[__HostNameLength];
		gethostname(HostName,__HostNameLength);
		__Hostname__ = std::string(HostName);
	}
	return __Hostname__;
}

////////////////////////////////////////////////////////////////////////////////
//////////////////////// STRING MANIPULATION ///////////////////////////////////
template <class T>
static inline string itoa(T num)
{
	stringstream  s;
	s << num;
	return s.str();
}

////////////////////////////////////////////////////////////////////////////////
/////////////////////// PRINTING STL CONTAINERS ////////////////////////////////
template <class T>
static inline ostream& operator<<(ostream &out,const  std::vector<T> &v)
{
	int64_t sz = v.size();
	if ( sz == 0 )
	{
		out <<  "[]";
		return out;
	}

	out << "[";
	for ( int64_t i = 0 ; i < sz-1 ; ++i )
	{
		out << v[i] << ",";
	}
	out << v[sz-1] << "]";
	return out;
}

template <class T>
static inline ostream& operator<<(ostream &out,const  std::set<T> &s)
{
	int64_t sz = s.size();
	if ( sz == 0 )
	{
		out << "{}" << endl;
		return out;
	}
	typename std::set<T>::iterator itr,end;
	end = s.end();
	out << "{";
	int64_t count = 1;
	for ( count = 1,  itr = s.begin(); count < sz  ; ++itr, ++count  )
	{
		out << *itr << ",";
	}
	out << *itr << "}";
	return out;
}


////////////////////////////////////////////////////////////////////////////////
////////////////////////// SET OPERATIONS //////////////////////////////////////

template <class T>
static inline bool belongsto(const std::set<T> &s,const T& el)
{
	return ((s.find(el) == s.end()) ? false : true );
}

template <class T>
static inline bool belongsto(const std::vector<T> &s,const T& el)
{
	return ((s.find(el) == s.end()) ? false : true );
}


#endif
