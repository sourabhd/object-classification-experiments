#ifndef _PARAMETER_EXTRACTOR
#define _PARAMETER_EXTRACTOR

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
 * This code implements utility functions for reading of configuration files.
 *
 */


#include <iostream>
#include <cstdlib>
#include <string>
#include <vector>
#include <map>
#include <fstream>

using namespace std;
//const int MAX_PARAMETER_LENGTH = 100;


class ParameterExtractor
{
    static fstream paramFile;
    static map <string,string> parameters;

    public:
    ParameterExtractor();
    ParameterExtractor(string pfilename);
    static void loadParameters();
    static string getParameter(string key);


};

#endif
