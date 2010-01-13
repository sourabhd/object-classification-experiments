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


#include <ParameterExtractor.h>



    fstream ParameterExtractor::paramFile;
    map <string,string> ParameterExtractor::parameters;
    string paramFileName ;

ParameterExtractor::ParameterExtractor()
{
    paramFileName = "../settings/param.conf";
    paramFile.open(paramFileName.c_str(),ios::in);
    loadParameters();
}

ParameterExtractor::ParameterExtractor(string pfilename)
{
    paramFileName = pfilename;
    paramFile.open(paramFileName.c_str(),ios::in);
    loadParameters();
}

void ParameterExtractor::loadParameters()
{

    //char token[MAX_PARAMETER_LENGTH];
	std::string token = "";

    string key = "";
    string value = "";
    string line = "";

    //while ( ( paramFile.getline(token,MAX_PARAMETER_LENGTH) )  )
    while ( ( getline(paramFile,token) )  )
    {

	// cout << token << "****" << endl;
	line.assign(token);     
	string::size_type loc = line.find("=", 0 );
	if( loc != string::npos ) 
	{
	} 
	else 
	{
	    std::cerr << "Error in parsing parameters ..." << endl;         
	} 

	key   = line.substr(0,loc);
	value = line.substr(loc+1);

	// cout << "Token is : " << key.c_str() << "???" << endl;
	// cout << "Value is : " << value.c_str() << "####" << endl;

	parameters.insert(make_pair(key,value));

	
    }



}

string ParameterExtractor::getParameter(string key)
{

    map<string,string>::iterator iter = parameters.find(key);
    if( iter != parameters.end() ) 
    {
	// cout << "Key : " << (iter->first).c_str() << " Value : " << (iter->second).c_str()  << endl;
	return ( iter -> second );
    }
    else
    {
	std::cerr << "The parameter : " << key.c_str() << " not found in the settings file " << paramFileName.c_str() << " !!!" << endl;
	return "";
    }


}

/*
int main()
{

    ParameterExtractor *pr = new ParameterExtractor();
    string st = pr->getParameter("ABC");
    std::cout << st.c_str() << endl;



}

*/
