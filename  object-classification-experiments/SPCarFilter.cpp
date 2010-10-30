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
 * This code implements biologically inspired model of Serre-Wolf-Poggio 
 * http://cbcl.mit.edu/projects/cbcl/publications/ps/serre-PID73457-05.pdf
 * and a number of variations with the model.
 *
 */


#include <GaborFilterModel.h>

////////////////////////////////////////////////////////////////////////////////
/** Version 2 training phase */
////////////////////////////////////////////////////////////////////////////////

void GaborFilterModel::trainCarFilter()
{
		
	int32_t rank = MPI::COMM_WORLD.Get_rank();
	string s2file_un = "";
	string c2file_un = "";
	keyfile      = keydir    + OSPATHSEP + "train.key" ;
	patchposfile = pposdir   + OSPATHSEP + "train.ploc";
	s2file       = s2dir     + OSPATHSEP + "train.s2";
	s2file_un    = s2dir     + OSPATHSEP + "train.s2.unsorted";
	c2file       = c2dir     + OSPATHSEP + "train.c2";
	c2file_un    = c2dir     + OSPATHSEP + "train.c2.unsorted";
	svmtrfile    = svmtrdir  + OSPATHSEP + "train.svmtr"; 	
	svmmofile    = svmmodir  + OSPATHSEP + "train.svmmo"; 

	if ( rank == 0 ) // Single Host 
	{
		createKeyFile(trdir,keyfile);
		readKeyFile(keyfile);
		createRandomPatches(patchposfile); 
		calcS2(s2file_un);
		s2file = s2dir + OSPATHSEP + "train.s2"; // reset s2 file name as it is overwritten
		sortS2(s2file_un,s2file);
		readKeyFile(keyfile);
		calcFVStage1(s2file,patchposfile);
		calcFVStage2(c2file_un);
		c2file = c2dir + OSPATHSEP + "train.c2";
		sortS2(c2file_un,c2file);
		convC2ToSVM(c2file,svmtrfile,this->FVDim);
		svmtrain(svmtrfile,svmmofile);
	}

}


////////////////////////////////////////////////////////////////////////////////
/** Version 2 test phase */
////////////////////////////////////////////////////////////////////////////////

void GaborFilterModel::testCarFilter()
{
	int32_t rank = MPI::COMM_WORLD.Get_rank();
	string trkeyfile = keydir    + OSPATHSEP + "train.key" ;
	string tekeyfile = keydir    + OSPATHSEP + "test.key" ;
	patchposfile     = pposdir   + OSPATHSEP + "train.ploc";
	string trs2file  = s2dir     + OSPATHSEP + "train.s2";
	string tes2file  = s2dir     + OSPATHSEP + "test.s2";
	string tes2file_un  = s2dir     + OSPATHSEP + "test.s2.unsorted";
	c2file           = c2dir     + OSPATHSEP + "test.c2";
	string c2file_un = c2dir     + OSPATHSEP + "test.c2.unsorted";
	svmmofile        = svmmodir  + OSPATHSEP + "train.svmmo"; 
	svmtefile        = svmtedir  + OSPATHSEP + "train.svmte"; 
	svmoufile        = svmoudir  + OSPATHSEP + "train.svmou"; 
	
	if ( rank == 0 ) // Single host
	{
		keyfile = trkeyfile;
		readKeyFile(trkeyfile);
		s2file = trs2file;
		calcFVStage1(s2file,patchposfile);

		// At this moment vpatch has S2 patches of training images .. OK
		// .. but vmmat also has S2 patches of training images ..
		// .. so we need to flush them and bring in S2 features of test images
		s2file = tes2file_un;  // Change S2 file name
		keyfile = tekeyfile; // Change the key file name
		createKeyFile(tedir,tekeyfile); // Create key file of test images
		readKeyFile(tekeyfile); // Change Key file name and read key file for test images
		calcS2(s2file); // Calculate S2 features for test images ( note S2 features do not require patches)
		s2file = tes2file;
		sortS2(tes2file_un,s2file);
		readS2File(s2file); // S2 file written is read in memory for calling calcFVStage2

		// Now, perform C2 computations 
		calcFVStage2(c2file_un); // Calculate C2 features
		c2file = c2dir + OSPATHSEP + "test.c2";
		sortS2(c2file_un,c2file);
		convC2ToSVM(c2file,svmtefile,this->FVDim);
		// Run SVM predict
		stringstream svmtecmdstr;
		svmtecmdstr << svmpredexe <<  " " <<  svmtefile << " " << svmmofile << " " << svmoufile;
		string svmtecmd = svmtecmdstr.str();
		CERR << svmtecmd.c_str() << endl;
		system(svmtecmd.c_str()); // Execute SVM predict 
		perror(0); // Report error if any in the system call

	}


}

#if 0
int main(int argc,char *argv[])
{

	if ( argc !=  2)
	{
		CERR << "Usage: gfmodel <settingsfile>" << endl;
		exit(INCORRECT_ARGS);
	}

#ifdef _MPI
	MPI::Init(argc,argv);
#endif

	std::string conffl(argv[1]);
	GaborFilterModel gfm(conffl.c_str(),"version2");
	gfm.trainCarFilter();
	//gfm.testCarFilter();

#ifdef _MPI
	MPI::Finalize();
#endif


	return 0;
}
#endif
