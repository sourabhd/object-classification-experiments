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
/** Building classifer for query */
////////////////////////////////////////////////////////////////////////////////

void GaborFilterModel::buildClassifierForQuery(const string &query)
{
	// Get directory locations
	string qTrDir = pe->getParameter("QTRDIR");
	string keyfileDir = pe->getParameter("QKEYFILEDIR");
	string s2dir = pe->getParameter("QS2DIR");
	string pposdirnm = pe->getParameter("QPATCHPOSDIR");
	string c2dir = pe->getParameter("QC2DIR");
	string svmtrdir = pe->getParameter("QSVMTRDIR");
	string svmmodeldir = pe->getParameter("QMODELDIR");
	string svmtrainexe = pe->getParameter("QSVMTRAINEXE");
	// Form filenames
//	string trainImgLoc = oneClsTrDir + OSPATHSEP + cls;
//	string kfile = keyfileDir + OSPATHSEP + query + ".key"; 
//	string s2filenm = s2dir + OSPATHSEP + query + ".s2";
//	string pposfilenm = pposdirnm + OSPATHSEP + cls + ".ploc"; 
//	string c2filenm = c2dir + OSPATHSEP + cls + ".c2";
//	string svmtrfilenm = svmtrdir + OSPATHSEP + cls + ".svmtr";
//	string svmmodelfilenm = svmmodeldir + OSPATHSEP + cls + MODEL_EXT;
	// Processing
//	s2file = s2filenm; // Change S2 file name
//	createKeyFile(trainImgLoc,kfile); // Create key file
//	readKeyFile(kfile); // Change Key file name and read key file
//	calcS2(s2filenm); // Calculate S2 features 
//	createRandomPatches(pposfilenm); // Change patch location file name and generate patches	
//	calcFVStage1(s2filenm,pposfilenm); // Prepare for C2 (SVM) feature calculation
//	calcFVStage2(c2filenm); // Calculate C2 features 
//	convC2ToSVM(c2filenm,svmtrfilenm,NumPatchesPerClass,classid);
//	stringstream svmtrcmdstr;
//	svmtrcmdstr << svmtrainexe << LIBSVMTYPESWITCH << ONECLASS << " " 
//		<<  svmtrfilenm << " " << svmmodelfilenm;  // One class SVM 
//	string svmtrcmd = svmtrcmdstr.str();
//	CERR << svmtrcmd.c_str() << endl;
//	system(svmtrcmd.c_str()); // Execute SVM train 
//	perror(0); // Report error if any in the system call

}

#if 0
int main(int argc, char *argv[])
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
	GaborFilterModel gfm(conffl.c_str());
	gfm.buildClassifierForQuery("lotus");
#ifdef _MPI
	MPI::Finalize();
#endif


	return 0;
}

#endif
