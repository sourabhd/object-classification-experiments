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
 * This code implements a directory traversal routine.
 */


#include "dir.h"

void dfsvisit(const fs::path &dir,int64_t &id,std::vector<fs::path> &list,std::ofstream &fout)
{
	fs::directory_iterator end_itr;
	for ( fs::directory_iterator itr(dir) ; itr != end_itr ; ++itr )
	{
		if( fs::is_directory(*itr) )
		{
			dfsvisit(*itr,id,list,fout);
		}
		else
		{
			id++;
//#ifdef _DEBUG
			stringstream s;
			s << "I" << id;
			fout << s.str() << "=" << *itr << endl;
//#endif
			list.push_back(*itr);
		}
	}

}

#if 0
//int main(int argc,char *argv[])
//{
//	std::ofstream ofs("list.txt");
//	int64_t *id = new int64_t;
//	*id = -1; // Important !!
//	std::vector<fs::path> list;
//	list.clear();
//	dfsvisit("/mnt/input/iitk/sourabhd/Caltech101AndCars/",*id,list);
//	cout << "Size :: " << list.size() << endl;
//	delete id;
//	return 0;
//}

#endif
