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


#include "GaborFilterModel.h"

////////////////////////////////////////////////////////////////////////////////
/** creating multiple classifiers */
////////////////////////////////////////////////////////////////////////////////

void GaborFilterModel::multipleSVM(const int32_t groupsize,const int32_t repeatFactor)
{
	//		//calcS2();
	//	createRandomPatches(); // Serre-Poggio Model
	calcFVStage1Multi();
	vivec_t v;
	createPatternsMulti(v);  // Create classifier patterns
	// Create the SVM models
	int32_t vsz = v.size();
	for ( int32_t i = 0 ; i < vsz ; i++ )
	{
		CERR << "Classifier # " << i << "\t: " << v[i] << endl;
		//calcFVStage2Multi(v[i],i);
	}


	//	int32_t classid = catnum["minaret"];
	ivec_t execlist(NumCat);
	for ( int32_t i = 0 ; i < NumCat ; i++ )
	{
		execlist[i] = i;
	}

#if 0
	// Pre-computing training set C2 vectors
	//	std::vector<vec_t> allc2train;
	//	for ( int32_t i = 0 ; i < NumCat ; i++ )
	//	{
	//		for ( int32_t j = 0 ; j  < NumTrainingImages ; j++ )
	//		{
	//			vec_t c2(NumCat); 
	//			//TIME(t1);
	//			calcC2Multi(i,j,execlist,c2) ;  // calculate entire C2 feature
	//			allc2train.push_back(c2);
	//			//TIME(t2);
	//			//CERR << "Time taken "  << difftime(t1,t2) << endl;
	//		}
	//	}
#endif

	string outfile = pe->getParameter("OUTFILE");
	ofstream ofile(outfile.c_str());
	if (!ofile)
	{
		CERR << "Could not open " << outfile.c_str() << endl;
		ofile.close();
		exit(FILE_OPEN_FAILED);
	}

	for ( int cls = 2 ; cls < 10 ; cls++ )
	{
		int32_t classid = cls;
		int32_t classtrsz = v_vmmat[classid].size() - NumTrainingImages ;
		CERR << "Category Id : " << classid << " :: " << categorylist[classid] << " :: " << classtrsz << " test images"<< endl << flush;
		std::vector<vec_t> allc2;
		allc2.clear();
		for ( int32_t i = NumTrainingImages ; i < NumTrainingImages+classtrsz ; i++ )
		{
			vec_t c2(NumCat); 
			//TIME(t1);
			calcC2Multi(classid,i,execlist,c2) ;  // calculate entire C2 feature
			allc2.push_back(c2);
			//TIME(t2);
			//CERR << "Time taken "  << difftime(t1,t2) << endl;
			cerr << i << " ... ";
		}

		int allc2sz = allc2.size();
		CERR << allc2sz << " C2 vectors calculated  " << endl << flush; 
		CERR << "Creating test files" << endl;
		TIME(t3);
		ofstream ofs[NumClassifiers];


		// Creating test files
		for ( int32_t j = 0 ; j < NumClassifiers ; j++ )
		{
			stringstream fname;
			fname << "test_" << classid << "_" << j << ".txt";
			ofs[j].open(fname.str().c_str());
		}

		for ( int32_t i = 0 ; i < allc2sz ; i++ )
		{
			for ( int32_t j = 0 ; j < NumClassifiers ; j++ )
			{
				ofs[j] << classid << " ";
				int32_t vsize = v[j].size();
				for ( int32_t k = 0 ; k < vsize ; k++ )
				{
					for ( int32_t l = 0 ; l < NumPatchesPerClass ; l++ )
					{
						ofs[j] << ((k)*NumPatchesPerClass+l) << ":" << allc2[i]((v[j](k))*NumPatchesPerClass+l) << " ";
					} // end of l loop
				} // end of k loop
				ofs[j] << endl;
			} // end of j loop
		} // end of i loop

		for ( int32_t j = 0 ; j < NumClassifiers ; j++ )
		{
			ofs[j].close();
		}

		TIME(t4);
		CERR << " It took " << difftime(t3,t4) << " ms for creating test files" << endl;

		// Run Prediction
		string svmpred = pe->getParameter("SVMPREDEXE");
		int32_t rank = 0;  
		for ( int32_t i = 0 ; i < NumClassifiers ; i++ )
		{
			stringstream cmdstr;
			cmdstr << svmpred  << " test_" << i << ".txt svm__" << rank << "___" << i << ".txt.model svmout_" << classid << "_" << i << ".txt > " 
				<< "out_te_" << classid << "_" << i << ".txt";
			CERR << cmdstr.str().c_str() << endl;
			system(cmdstr.str().c_str());
			perror(0);
		}

		// Combine all results related to this class and get frequently occuring classes
		stringstream cdstr;
		cdstr << "paste " << "svmout_" << classid << "_* > " <<  "allres_" << classid <<".txt";
		system(cdstr.str().c_str());
		perror(0);

		stringstream arsltfl;
		arsltfl << "allres_" << classid << ".txt";
		string allclsoutfile = arsltfl.str();

		ifstream ifs(allclsoutfile.c_str());
		if(!ifs)
		{
			CERR << "Could not open " << allclsoutfile.c_str() << endl;
			ifs.close();
			exit(1);
		}


		int64_t tid = -1;
		string line = "";
		while(getline(ifs,line))
		{

			TIME(t1);
			tid++;
			stringstream l(line,ios::in);
			int count[NClasses] = {0};
			for ( int i = 0 ; i < NumClassifiers ; i++ )
			{
				int cl = -1;
				l >> cl;
				count[cl]++;
			} // end of i loop

			//cout << tid << " ::=> " << endl;
			std::vector<int32_t> v;
			v.clear();
			for ( int i = 0 ; i < NClasses ; i++ )
			{
				if((( count[i] == MaxOcc )) || ( count[i] == MaxOcc -1 ))
				{
					//cout << i << "::";
					v.push_back(i);
				}

			} // end of i loop

			//cout << endl;
			int32_t vsz = v.size();
			ivec_t clist(vsz);
			for ( int32_t i = 0 ; i < vsz ; i++ )
			{
				clist[i] = v[i];
			}

			int32_t Ntrain = 4;
			vec_t c2test(vsz*NumPatchesPerClass);
			calcC2Multi(classid,tid+NumTrainingImages,clist,c2test) ;  // calculate reduced C2 feature
			double mindist = INF; 
			int32_t mindistcl = -1;
			for ( int32_t i = 0 ; i < vsz ; i++ ) // for all candidate classes
			{
				double avg = 0.0;
				for ( int32_t j = 0 ; j  < Ntrain ; j++ ) // for all training samples
				{
					vec_t c2(vsz*NumPatchesPerClass); 
					calcC2Multi(clist[i],j,clist,c2) ;  // calculate reduced C2 feature
					double dist = b::norm_2(c2-c2test);
					avg += dist;
					//CERR << dist << endl;
				}
				avg /= double(Ntrain);
				if ( avg < mindist )
				{
					mindist = avg;
					mindistcl = clist[i];
				}

			}

			ofile << categorylist[classid] << " " << classid << " " << mindistcl << endl;

			TIME(t2);
			CERR << "Time taken "  << difftime(t1,t2) << endl;

		} // end of while

	} // end of cls loop

	ofile.close();

}

////////////////////////////////////////////////////////////////////////////////
/** Create Patterns for selecting classes for multiple SP classifiers */
////////////////////////////////////////////////////////////////////////////////

void GaborFilterModel::createPatternsMulti(vivec_t &v)
{
	int32_t N = 64; // number of classes 
	int32_t M = 8;  // number of classes in a group
	int32_t K = N / M;

	for ( int32_t i = 0 ; i <  K ; i++ )
	{
		ivec_t v1(M);
		for ( int32_t j = 0 ; j < M ; j++ )
		{
			v1(j) = (i)*M+j;
		}
		v.push_back(v1);
	}

	for ( int32_t i = 0 ; i <  M ; i++ )
	{
		ivec_t v1(K);
		for ( int32_t j = 0 ; j < K ; j++ )
		{
			v1(j) = (j)*K+i;
		}
		v.push_back(v1);
	}

	assert( K == M );
	for ( int32_t l = 0 ; l < M ; l++ )
	{
		ivec_t v1(M);
		for ( int32_t k = 0 ; k < M ; k++ )
		{
			int32_t i = (k+l)%M;
			int32_t j = k;
			v1(k) = (i)*M+j;

		}
		v.push_back(v1);
	}
}


///////////////////////////////////////////////////////////////////////////////
/** Stage 1 of the feature vector computation */
///////////////////////////////////////////////////////////////////////////////

void GaborFilterModel::calcFVStage1Multi()
{
	s2file =  pe->getParameter("S2FILE");
	ifstream ifs(s2file.c_str());
	if ( !ifs )
	{
		CERR << "Could not open file :" <<  s2file.c_str() << std::endl;
		ifs.close();
		exit(FILE_OPEN_FAILED);
	}

	patchposfile = pe->getParameter("PATCHPOSFILE");
	ifstream ifs2(patchposfile.c_str());
	if ( !ifs2 )
	{
		CERR << "Could not open file :" <<  patchposfile.c_str() << std::endl;
		ifs2.close();
		exit(FILE_OPEN_FAILED);
	}



	vmmat_t v;
	std::string prevcat = "__prevcat__";
	int32_t intid = -1, sum = 0;
	for ( int64_t i = 0 ; i < NumFiles ; i++ ) // for each file
	{
		int64_t imgID;
		ifs >> imgID;
		std::string cat;
		ifs >> cat;
#ifdef _DEBUG
		CERR << imgID << " " << cat << endl;
#endif

		mmat_t mmat(NumOrientations,NumBands);
		ifs >> mmat;
		intid++;
		if ( ( i > 0 ) && ( prevcat != cat ) )
		{
			v_vmmat.push_back(v);
			int64_t ctcnt = v.size();
			sum += ctcnt;
			CERR << "Category : " << prevcat << ": " << ctcnt << " S2 images" << endl;
			v.clear();
			intid = 0;
		}
		v.push_back(mmat);
		intcatid.push_back(intid);

		prevcat = cat;

	}

	v_vmmat.push_back(v);
	intcatid.push_back(intid);
	int64_t ctcnt = v.size();
	sum += ctcnt;
	CERR << "Category : " << prevcat << ": " << ctcnt << " S2 images" << endl;
	CERR << "Total " << v_vmmat.size() << " Categories ; " << sum << " images !! " << endl;


	v_vpatch.clear();
	vbvmat_t vp;
	std::string prevcat2 = "__prevcat__";
	int32_t sum2 = 0;

	//for ( int64_t i = 0 ; i < FVDim ; i++ )
	for ( int64_t i = 0 ; ; i++ )
	{
		b::vector<int32_t>	pl(5);
		ifs2 >> pl ;
		if ( ifs2.eof() )
		{
			break;
		}
#ifdef _DEBUG
		CERR << pl << endl;
#endif

		string cat2 = category[pl[0]];
		bvmat_t opatch(NumOrientations); 	
		for ( int32_t j = 0 ; j < NumOrientations ; j++ )
		{
			//opatch[j] = b::project(vmmat[pl[0]](j,pl[1]),b::range(pl[2],pl[2]+pl[4]),b::range(pl[3],pl[3]+pl[4]));
			int32_t cnum = catnum[category[pl[0]]];
			int32_t inum = intcatid[pl[0]];
			opatch[j] = b::project(v_vmmat[cnum][inum](j,pl[1]),b::range(pl[2],pl[2]+pl[4]),b::range(pl[3],pl[3]+pl[4]));
		}

		if ( ( i > 0 ) && ( prevcat2 != cat2 ) )
		{
			v_vpatch.push_back(vp);
			int64_t ctcnt2 = vp.size();
			sum2 += ctcnt2;
			CERR << "Category : " << prevcat2 << ": " << ctcnt2 << " patches" << endl;
			vp.clear();
		}
		vp.push_back(opatch);
		prevcat2 = cat2;
	}

	//this->FVDim = vpatch.size();
	v_vpatch.push_back(vp);
	int64_t ctcnt2 = vp.size();
	sum2 += ctcnt2;
	CERR << "Category : " << prevcat2 << ": " << ctcnt2 << " patches" << endl;
	CERR << v_vpatch.size() << " Categories ; " << sum2 << " patches pushed !!" << endl;

	ifs2.close();
	ifs.close();

}

///////////////////////////////////////////////////////////////////////////////
/** Stage 2 of the feature vector computation for multiple classifiers method */
///////////////////////////////////////////////////////////////////////////////

void GaborFilterModel::calcFVStage2Multi(const ivec_t &execlist, const int32_t id)
{

	double cval = 0.0 , gval = 0.0;
	const int32_t esize = execlist.size();
	assert ( (esize > 0) && (esize <= ((int32_t)(v_vmmat.size()))) );

	// Create SVM input input file
	std::string svmfile = pe->getParameter("SVMFILE");
	int32_t rank = 0;
	int32_t numprocs = 1;
	std::stringstream s;
	s << svmfile << "_" << rank << "___" << id << ".txt";

#ifdef _MPI
	int argc = 0;
	char **argv = 0;	
	//MPI::Init(argc,argv);
	//numprocs = MPI::COMM_WORLD.Get_size();
	//rank = MPI::COMM_WORLD.Get_rank();
	numprocs = 1; 
	rank = 0;
	int hostnameLength;
	char processorName[MPI_MAX_PROCESSOR_NAME];
	strcpy(processorName,"test");
	//MPI::Get_processor_name(processorName,hostnameLength);
#endif

	int64_t lsize = esize ;
	int64_t chunksize = lsize / numprocs;
	int64_t start = rank * chunksize;
	int64_t end = start + chunksize;
	//int32_t counter = start - 1 ;

#ifdef _MPI 
	CERR  << " : PROCESS: " << rank << "  :: " << start << " " << end << endl;
#endif

	int64_t fdim = NumTrainingImages * NumPatchesPerImage;
	this->FVDim = fdim * esize;
	this->NumFiles = NumTrainingImages * esize;
	b::vector<vec_t> output(NumFiles);

	for ( int64_t i = 0 ; i < NumFiles ; i++ )
	{
		output[i] = b::zero_vector<double>(FVDim);
	}

	ofstream ofs(s.str().c_str());
#ifdef _OPENMP
#pragma omp parallel for 
#endif
	for ( int64_t i1 = start ; i1 < end ; i1++ ) // for every input category
	{
		for ( int64_t i2 = 0 ; i2 < NumTrainingImages  ; i2++ ) // for every training image
		{ // i1,i2: for each file
			int64_t i = (i1)*NumTrainingImages+i2;

			for ( int64_t l1 = 0 ; l1 < esize ; l1++ ) // for every input category
			{
				for ( int64_t l2 = 0 ; l2 < fdim ; l2++ )
				{
					int64_t l = (l1)*fdim+l2;
					double max = -INF;
					for ( int64_t k = 0 ; k < NumBands ; k++ ) // for each band
					{

						bvmat_t inp(NumOrientations);
						for ( int64_t j = 0 ; j < NumOrientations ; j++ ) // for each orientation
						{
							inp[j] = v_vmmat[execlist[i1]][i2](j,k);

						} // end of j loop

						double resp =  maxresp(inp,v_vpatch[execlist[l1]][l2]) ;
						if ( resp > max )
						{
							max = resp;
						}

					} // end of k loop

					//cerr << max << " " ;
					output[i][l] = max;

				} // end of l2 loop
			} // end of l1 loop

			//#ifdef _OPENMP
			//#pragma omp critical
			//#endif
			//			{
			//				ofs << i << " " << category[execlist[i1]] << " " << output[i] << std::endl ;
			//			}
			//			//cerr << endl;
		} // end of i2 loop
	} // end of i1 loop

	for ( int64_t i = 0 ; i < NumFiles ; i++ )
	{
		ofs << execlist[i/NumTrainingImages] << " ";
		for ( int64_t j = 0 ; j < this->FVDim ; j++ )
		{
			ofs << j << ":" << output[i][j] << " "; 
		}
		ofs << endl;
	}
	ofs.close();

	// Do grid search with 3-fold validation
	string gridexe = pe->getParameter("GRIDEXE");
	string svminpfile = s.str();
	string gridoutput = s.str() + ".out";
	stringstream svmcmdstr;
	svmcmdstr << "python " << gridexe << " -v " << NumFolds << " " << svminpfile << " -out " << gridoutput ;
	string svmcmd = svmcmdstr.str();
	CERR << svmcmd.c_str() << end;
	system(svmcmd.c_str());
	perror(0);

	ifstream f1(gridoutput.c_str());
	if(!f1)
	{
		CERR << "Could not open file " << s.str().c_str() << endl;
	}

	string line = "";
	double maxc = 0.0, maxg= 0.0, maxacc = 0.0; // Accuracy can not be negative
	double curc = 0.0, curg = 0.0, curacc = 0.0; 
	while(getline(f1,line))
	{
		stringstream l(line,ios::in);
		l >> curc >> curg >> curacc;
		if ( curacc >= maxacc )
		{
			maxacc = curacc;
			maxc = curc;
			maxg = curg;
		}

	}

	cval = pow(2.0,maxc);
	gval = pow(2.0,maxg);
	CERR << "Best Values :" << maxacc << " " << cval <<" " << gval << endl;

	f1.close();

	// Now train the SVM
	string svmexe = pe->getParameter("SVMTRAINEXE");
	stringstream svmcmdstr2;
	svmcmdstr2 << svmexe << " -c " << cval << " -g " << gval << " " << svminpfile ;
	string svmcmd2 = svmcmdstr2.str();
	CERR << svmcmd2.c_str() << end;
	system(svmcmd2.c_str());
	perror(0);

#ifdef _MPI
	//MPI::Finalize();
#endif

}

///////////////////////////////////////////////////////////////////////////////
/** Calculating C2feature vectors for multple classifier method */
///////////////////////////////////////////////////////////////////////////////

void GaborFilterModel::calcC2Multi(const int32_t catid, 
		const int32_t intid, const ivec_t &execlist,
		vec_t &output)
{

	const int32_t esize = execlist.size();
	assert ( (esize > 0) && (esize <= ((int32_t)(v_vmmat.size()))) );

	int64_t fdim = NumTrainingImages * NumPatchesPerImage;
	this->FVDim = fdim * esize;
	this->NumFiles = NumTrainingImages * esize;

	output = b::zero_vector<double>(FVDim);

	int64_t i1 = catid;
	int64_t i2 = intid;

	for ( int64_t l1 = 0 ; l1 < esize ; l1++ ) // for every input category
	{
		for ( int64_t l2 = 0 ; l2 < fdim ; l2++ )
		{
			int64_t l = (l1)*fdim+l2;
			double max = -INF;
			for ( int64_t k = 0 ; k < NumBands ; k++ ) // for each band
			{

				bvmat_t inp(NumOrientations);
				for ( int64_t j = 0 ; j < NumOrientations ; j++ ) // for each orientation
				{
					inp[j] = v_vmmat[i1][i2](j,k);

				} // end of j loop

				double resp =  maxresp(inp,v_vpatch[execlist[l1]][l2]) ;
				if ( resp > max )
				{
					max = resp;
				}

			} // end of k loop

			//cerr << max << " " ;
			output[l] = max;

		} // end of l2 loop
	} // end of l1 loop
}

///////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
#if 0
int main(int argc,char *argv[])
{

	if ( argc !=  2)
	{
		CERR << "Usage: gfmodel <settingsfile>" << endl;
		exit(INCORRECT_ARGS);
	}

	std::string conffl(argv[1]);
	GaborFilterModel gfm(conffl.c_str());
	gfm.multipleSVM(10,3);

	return 0;
}

#endif
