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
/** create once class SVM */
////////////////////////////////////////////////////////////////////////////////

void GaborFilterModel::buildOneClassSVM(const std::string &cls,const int32_t classid)
{
	// Get directory locations
	string oneClsTrDir = pe->getParameter("ONECLASSTRDIR");
	string keyfileDir = pe->getParameter("KEYFILEDIR");
	string s2dir = pe->getParameter("S2DIR");
	string pposdirnm = pe->getParameter("PATCHPOSDIR");
	string c2dir = pe->getParameter("C2DIR");
	string svmtrdir = pe->getParameter("ONECLASSSVMTRDIR");
	string svmmodeldir = pe->getParameter("ONECLASSMODELDIR");
	string svmtrainexe = pe->getParameter("SVMTRAINEXE");
	// Form filenames
	string trainImgLoc = oneClsTrDir + OSPATHSEP + cls;
	string kfile = keyfileDir + OSPATHSEP + cls + ".key"; 
	string s2filenm = s2dir + OSPATHSEP + cls + ".s2";
	string pposfilenm = pposdirnm + OSPATHSEP + cls + ".ploc"; 
	string c2filenm = c2dir + OSPATHSEP + cls + ".c2";
	string svmtrfilenm = svmtrdir + OSPATHSEP + cls + ".svmtr";
	string svmmodelfilenm = svmmodeldir + OSPATHSEP + cls + MODEL_EXT;
	// Processing
	s2file = s2filenm; // Change S2 file name
	createKeyFile(trainImgLoc,kfile); // Create key file
	readKeyFile(kfile); // Change Key file name and read key file
	calcS2(s2filenm); // Calculate S2 features 
	createRandomPatches(pposfilenm); // Change patch location file name and generate patches	
	calcFVStage1(s2filenm,pposfilenm); // Prepare for C2 (SVM) feature calculation
	calcFVStage2(c2filenm); // Calculate C2 features 
	convC2ToSVM(c2filenm,svmtrfilenm,NumPatchesPerClass,classid);
	stringstream svmtrcmdstr;
	svmtrcmdstr << svmtrainexe << LIBSVMTYPESWITCH << ONECLASS << " " 
		<<  svmtrfilenm << " " << svmmodelfilenm;  // One class SVM 
	string svmtrcmd = svmtrcmdstr.str();
	CERR << svmtrcmd.c_str() << endl;
	system(svmtrcmd.c_str()); // Execute SVM train 
	perror(0); // Report error if any in the system call

}

////////////////////////////////////////////////////////////////////////////////
/** One class SVM test phase */
////////////////////////////////////////////////////////////////////////////////

void GaborFilterModel::testOneClassSVM(const std::string &cls,const std::string &testdir
		,const int32_t testclassid) 
{
	// Get directory locations
	string oneClsTrDir = pe->getParameter("ONECLASSTRDIR");
	string oneClsTeDir = pe->getParameter("ONECLASSTEDIR");
	string keyfileDir = pe->getParameter("KEYFILEDIR");
	string s2dir = pe->getParameter("S2DIR");
	string pposdirnm = pe->getParameter("PATCHPOSDIR");
	string c2dir = pe->getParameter("C2DIR");
	string svmtedir = pe->getParameter("ONECLASSSVMTEDIR");
	string svmmodeldir = pe->getParameter("ONECLASSMODELDIR");
	string svmoutdir = pe->getParameter("ONECLASSSVMOUTDIR");
	string svmpredexe = pe->getParameter("SVMPREDEXE");
	// Form filenames
	string trainImgLoc = oneClsTrDir + OSPATHSEP + cls;
	string testImgLoc = oneClsTeDir + OSPATHSEP + testdir;
	string kfile = keyfileDir + OSPATHSEP + cls + ".key"; 
	string kfiletest = keyfileDir + OSPATHSEP + cls + "__" + testdir + "__test" + ".key"; 
	string s2filenm = s2dir + OSPATHSEP + cls + ".s2";
	string s2filenmtest = s2dir + OSPATHSEP + cls + "__" + testdir + "__test" + ".s2";
	string pposfilenm = pposdirnm + OSPATHSEP + cls + ".ploc"; 
	string c2filenm = c2dir + OSPATHSEP + cls + "__" + testdir + "__test" + ".c2";
	string svmtefilenm = svmtedir + OSPATHSEP + cls + "__" + testdir +  ".svmte";
	string svmmodelfilenm = svmmodeldir + OSPATHSEP + cls + MODEL_EXT;
	string svmoutfilenm = svmoutdir + OSPATHSEP + cls + "__" + testdir +  ".svmout";
	// Processing
	s2file = s2filenm; // Change S2 file name
	createKeyFile(trainImgLoc,kfile); // Create key file
	readKeyFile(kfile); // Change Key file name and read key file
	createRandomPatches(pposfilenm); // Change patch location file name and generate patches	
	calcFVStage1(s2filenm,pposfilenm); // Prepare for C2 (SVM) feature calculation
	CERR << "Training patches brought in memory ... Now, S2 features of test images will be tried" << endl;
	
	// At this moment vpatch has S2 patches of training images .. OK
	// .. but vmmat also has S2 patches of training images ..
	// .. so we need to flush them and bring in S2 features of test images
	s2file = s2filenmtest;  // Change S2 file name
	createKeyFile(testImgLoc,kfiletest); // Create key file of test images
	readKeyFile(kfiletest); // Change Key file name and read key file for test images
	calcS2(s2file); // Calculate S2 features for test images ( note S2 features do not require patches)
	readS2File(s2file); // S2 file written is read in memory for calling calcFVStage2

	// Now, perform C2 computations 
	calcFVStage2(c2filenm); // Calculate C2 features 
	convC2ToSVM(c2filenm,svmtefilenm,NumPatchesPerClass,testclassid);
	stringstream svmtecmdstr;
	svmtecmdstr << svmpredexe <<  " " <<  svmtefilenm << " " << svmmodelfilenm << " " << svmoutfilenm;  // One class SVM 
	string svmtecmd = svmtecmdstr.str();
	CERR << svmtecmd.c_str() << endl;
	system(svmtecmd.c_str()); // Execute SVM predict 
	perror(0); // Report error if any in the system call

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
	GaborFilterModel gfm(conffl.c_str());
	gfm.buildOneClassSVM("car_side",0);
	gfm.testOneClassSVM("car_side","car_side",0);
	gfm.testOneClassSVM("car_side","BACKGROUND_Google",1);
	gfm.testOneClassSVM("car_side","elephant",2);
#ifdef _MPI
	MPI::Finalize();
#endif

	return 0;
}

#endif
