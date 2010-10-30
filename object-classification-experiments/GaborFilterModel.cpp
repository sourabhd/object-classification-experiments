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
extern "C" {
#ifndef _EMD2_H
#define _EMD2_H
#include "emd.h" // Rubner's implementation of Earth Mover's Distance
//inline float dist(feature_t *F1, feature_t *F2) { return cm[(*F1)*len+(*F2)]; } // precomputed cost matrix
float dist2(feature_t *F1, feature_t *F2); // Euclidean distance
#endif
}



extern "C" float emd(signature_t *Signature1, signature_t *Signature2,
		float (*func)(feature_t *, feature_t *),
		flow_t *Flow, int *FlowSize);

////////////////////////////////////////////////////////////////////////////////
float *cm = 0;
int32_t len = 0;
typedef float farray_t[NumPatchVecPerImage][PLength];
typedef float weight_t[NumPatchVecPerImage];
weight_t wgt = { 1.0/NumPatchVecPerImage };
////////////////////////////////////////////////////////////////////////////////


float dist2(feature_t *F1, feature_t *F2)
{
	float sum = 0; int i; float diff;
	for ( i = 0 ; i < NumPatchVecPerImage ; i++ )
	{
		diff = ((*F1)[i]- (*F2)[i]);
		sum += (diff*diff);
	}

	return sqrt(sum);
}

////////////////////////////////////////////////////////////////////////////////
/**
 * Generate parameters of the Cortex Model as described in
 *                  
 *      @article{serre:rms,
 *       title={{Realistic Modeling of Simple and Complex Cell Tuning
 *		         in the HMAX Model, and Implications for Invariant 
 *				 Object Recognition in Cortex}},
 *       author={Serre, T. and Riesenhuber, M.}
 *       }
 * URL: http://cbcl.mit.edu/cbcl/publications/ai-publications/2004/AIM-2004-017.pdf
 * 
 * The model used is the "Gabor filters" HMAX model
 */
////////////////////////////////////////////////////////////////////////////////

void GaborFilterModel::genModelParam()
{
	const int32_t RFSizeStart = 7;
	const int32_t RFSizeStep  = 2;
	const double orientationStart = 0.0;
	const double orientationStep = 45.0;
	const double a = 0.0036;
	const double b = 0.35;
	const double c = 0.18;
	const double AspectRatio = 0.3;
	const double WaveLengthDivisor = 0.8;
	double o = orientationStart; 
	const int32_t gridSizeStartMinus2 = 6;
	const int32_t gridSizeStep = 2;

	for ( int32_t i = 0 ; i < NumOrientations ; i++ , o += orientationStep )
	{
		double s = RFSizeStart;
		int32_t gsize = gridSizeStartMinus2;

		for ( int32_t j = 0 ; j < NumRFSizes ; j++, s += RFSizeStep )
		{
			S1LayerParam[i][j].theta = o;
			S1LayerParam[i][j].s = s;
			S1LayerParam[i][j].sigma = a * s * s + b * s + c;
			S1LayerParam[i][j].gamma = AspectRatio;
			S1LayerParam[i][j].lambda = S1LayerParam[i][j].sigma / WaveLengthDivisor;
			if ( j % 2 == 0 )
			{
				gsize += gridSizeStep;
			}
			S1LayerParam[i][j].gridSize = gsize;
			S1LayerParam[i][j].s2cellsize = ImageSize / S1LayerParam[i][j].gridSize;

#ifdef _DEBUG
			CERR << i << "," << j << " :: " << S1LayerParam[i][j].theta << " " << S1LayerParam[i][j].s  \
				<< " " << S1LayerParam[i][j].sigma << " " << S1LayerParam[i][j].gamma \
				<< " " << S1LayerParam[i][j].lambda << " " << S1LayerParam[i][j].gridSize << endl;
#endif
		}
	}

}

////////////////////////////////////////////////////////////////////////////////
/** Generate Gabor filter bank */
////////////////////////////////////////////////////////////////////////////////

void GaborFilterModel::genFilterBank()
{
	for ( int32_t i = 0 ; i < NumOrientations ; i++ )
	{
		for ( int32_t j = 0 ; j < NumRFSizes ; j++ )
		{
			int32_t size = S1LayerParam[i][j].s * S1LayerParam[i][j].s ;
			filter_t f(size);
			getGaborCoeff(S1LayerParam[i][j].s,S1LayerParam[i][j].sigma,
					S1LayerParam[i][j].lambda,S1LayerParam[i][j].theta,S1LayerParam[i][j].gamma,f);
			GaborFilterBank[i][j] = new filter_t(size);
			*GaborFilterBank[i][j] = f;
#ifdef _DEBUG
			//			CERR << *GaborFilterBank[i][j] << endl << endl;
#endif

		}
	}
}


////////////////////////////////////////////////////////////////////////////////
/** getGaborCoeff : Calculate Gabor Filter co-efficients */
////////////////////////////////////////////////////////////////////////////////

void GaborFilterModel::getGaborCoeff(const int16_t filterSize,
		const double effectiveWidth,const double wavelength,
		const double theta,const double aspectRatio,filter_t &f) 
{
	const double TwoPiByWavelength = 2 * M_PI /  wavelength ;
	const double AspectRatioSquared = aspectRatio * aspectRatio;
	const double TwiceEffectiveWidthSquared = 2.0 * effectiveWidth * effectiveWidth;
	const int32_t size = filterSize * filterSize;

	f.clear();
	int i = 0;
	int16_t fsize = filterSize / 2; // floor
	for ( int x = -fsize ; x <= fsize ; x++ )
	{
		for ( int y = -fsize ; y <= fsize ; y++ )
		{
			double Theta = M_PI * theta / 180.0;
			double X = x * cos(Theta) - y * sin(Theta);
			double Y = x * sin(Theta) + y * cos(Theta);
			f[i] =  ( exp( -1.0 * ( X * X + AspectRatioSquared * Y * Y   )  / TwiceEffectiveWidthSquared ) * cos(TwoPiByWavelength * X )  ) ;
			i++;
		} // end of y loop
	} // end of x loop

	long double mean = 0.0;
	for( int32_t j = 0 ; j < size ; j++ )
	{
		mean += f[j];
	} // end of j loop
	mean /= size;

	for (  int32_t k = 0 ; k < size ; k++ )
	{
		f[k] = ( f[k] - mean ) ;
	}	

	double norm = boost::numeric::ublas::norm_2(f);
	f /= norm;

#ifdef _DEBUG
	//	CERR << "{ ";
	//	for (  int32_t k = 0 ; k < size ; k++ )
	//	{
	//		if ( k%filterSize == 0 ) { CERR << endl; }
	//		CERR << f[k] << "," ;
	//	}	
	//	CERR << " } ," << endl << endl;
	//CERR << "norm: " <<  norm << " :: mean: " << mean << endl;

	///////////////////////////////////////////////////////////
	//		double norm2 = boost::numeric::ublas::norm_2(f);
	//		mean = 0;
	//		for( int32_t j = 0 ; j < size ; j++ )
	//		{
	//			mean += f[j];
	//		} // end of j loop
	//		mean /= size;
	//
	//		CERR << "L2 norm squared  and mean of the normalized filter: " << norm2 * norm2 << "   " << mean  ;
	//

#endif

}	


////////////////////////////////////////////////////////////////////////////////
/** calcS2 : Calculate S2 features */
////////////////////////////////////////////////////////////////////////////////

void GaborFilterModel::calcS2(const string &s2file_)
{

	int32_t rank = 0;
	int32_t numprocs = 1;
	std::stringstream s;

#ifdef _MPI
	numprocs = MPI::COMM_WORLD.Get_size();
	rank = MPI::COMM_WORLD.Get_rank();
	int hostnameLength;
	char processorName[MPI_MAX_PROCESSOR_NAME];
	MPI::Get_processor_name(processorName,hostnameLength);
#endif

	int64_t lsize_pl = NumFiles ;
	int64_t chunksize_pl = lsize_pl / numprocs;
	int64_t start_pl = rank * chunksize_pl;
	int64_t end_pl = start_pl + chunksize_pl;

#ifdef _MPI 
	CERR <<  processorName << " : PROCESS: " << rank << "  :: " << start_pl << " " << end_pl << endl;
#endif

	if ( s2file_ == "" )
	{
		s2file =  pe->getParameter("S2FILE");
		s << s2file << "_" << rank << ".txt";
		s2file = s.str();
	}
	else
	{
		s2file = s2file_;
	}

	ofstream ofs(s2file.c_str());
	if ( !ofs )
	{
		CERR << "Could not open file :" <<  s2file.c_str() << std::endl;
		ofs.close();
		exit(FILE_OPEN_FAILED);
	}

	TIME(t3);
	CERR << "Start of S2 file " << s2file.c_str() << endl;
#ifdef _OPENMP
#pragma omp parallel for 
#endif
	for ( int64_t i = start_pl ; i < end_pl ; i++ ) 
	{
		TIME(t1);
		// CERR << filename[i] << endl;
		// Convert images to gray scale and resize them
		IplImage *img = 0, *imgt = 0, *gray = 0 , *grays = 0;
		img = cvLoadImage(filename[i].string().c_str());
		imgt = cvCreateImage(cvGetSize(img),IPL_DEPTH_8U,img->nChannels);
		cvConvertScale(img,imgt);
		cvReleaseImage(&img);
		gray = cvCreateImage(cvGetSize(imgt),IPL_DEPTH_8U,1);
		cvCvtColor(imgt,gray,CV_BGR2GRAY);
		cvReleaseImage(&imgt);
		grays = cvCreateImage(isz,IPL_DEPTH_8U,1);
		cvResize(gray,grays);
		cvReleaseImage(&gray);

		mmat_t DD(NumOrientations,NumRFSizes);
		mmat_t DD2(NumOrientations,NumRFSizes/2);
		for ( int32_t j = 0 ; j < NumOrientations ; j++ ) // for every orientation
		{
			for ( int32_t k = 0 ; k < NumRFSizes ; k++ ) // for each RF size 
			{

				mat_t R  = b::zero_matrix<double>(ImageSize,ImageSize);

				int32_t start = 0;
				int32_t end = ImageSize - S1LayerParam[j][k].s;

				for ( int32_t l = start ; l < end ; l++ )
				{
					for ( int32_t m = start ; m < end ; m++ )
					{
						R(l,m) =  getGaborRespAtPt(grays,l,m,k,j);
						assert(!isnan(R(l,m)));
					} // end of m loop 

				} // end of l loop

				DD(j,k) = R;
				//cout << R << endl;
			} // end of k loop 

		} // end of j loop

		doMaxOp(DD,DD2);

		cvReleaseImage(&grays);
		TIME(t2);
		//cout << i << " " << category[i] << " " << DD2 << endl;
#ifdef _OPENMP
#pragma omp critical
#endif
		{
			ofs << i << " " << category[i] << " " << DD2 << endl;
			CERR << "File  " << i << ": " << difftime(t1,t2) << " ms  :" << filename[i] << endl;
		}
	}  // end of i loop

	TIME(t4);
	CERR << "S2 file " << s2file.c_str() << " creation completed in " << difftime(t3,t4) << " ms" << endl;

	ofs.close();
}


////////////////////////////////////////////////////////////////////////////////
/** Do Max pooling operations : ( postion and scale ) */
////////////////////////////////////////////////////////////////////////////////

void GaborFilterModel::doMaxOp(const mmat_t &in,mmat_t &out)
{
	// MAX over position
	mmat_t maxp(NumOrientations,NumRFSizes);
	for ( int32_t i = 0 ; i < NumOrientations ; i++ ) // for each orientation
	{
		for ( int32_t j = 0 ; j < NumRFSizes ; j++ ) // for each scale
		{
			mat_t A = in(i,j);
			int32_t sqsize = S1LayerParam[i][j].gridSize;
			const int32_t osize = ImageSize / sqsize;
			mat_t B(osize,osize);
			int32_t gsize = sqsize * osize;
			for ( int32_t k = 0 ; k < gsize ; k += sqsize )
			{
				for ( int32_t l = 0 ; l < gsize ; l += sqsize )
				{
					// k,l => move over image for selecting a square 
					double max = -INF;
					int32_t mlim = k + sqsize;
					int32_t nlim = l + sqsize;
					for( int32_t m = k ; m < mlim ; m++ )
					{
						for ( int32_t n = l ; n < nlim ; n++ )
						{
							// m,n => loop over a patch
							if ( A(m,n) > max )
							{
								max = A(m,n);
							}
						} // end of n loop
					} // end of m loop
					B(k/sqsize,l/sqsize) = max;

				} // end of l loop
			} // end of k loop
			maxp(i,j) = B;

		} // end of j loop
	} // end of i loop


	// MAX over scale
	for ( int32_t i = 0 ; i < NumOrientations ; i++ ) // for each orientation
	{
		for ( int32_t j = 0 ; j < NumRFSizes ; j++ ) // for each scale
		{
			if ( j % 2 == 0 )
			{
				out(i,j/2) = maxp(i,j);
			}
			else
			{
				int32_t sqsize = S1LayerParam[i][j].gridSize;
				const int32_t osize = ImageSize / sqsize;
				mat_t A(osize,osize);
				mat_t B = out(i,j/2);
				mat_t C = maxp(i,j);

				for ( int32_t k = 0 ; k < osize ; k++ )
				{
					for ( int32_t l = 0 ; l < osize ; l++ )
					{
						A(k,l) = std::max(B(k,l),C(k,l));
					} // end of l loop
				} // end of k loop

				out(i,j/2) = A;

			}

		} // end of j loop
	} // end of i loop

}


////////////////////////////////////////////////////////////////////////////////
/** Calculating Gabor responses using the filter bank */
////////////////////////////////////////////////////////////////////////////////

double GaborFilterModel::getGaborRespAtPt(IplImage *gray,
		const int32_t xpos,
		const int32_t ypos, 
		int32_t sIndex,
		int32_t thetaIndex)
{
	assert(gray);
	assert(gray->nChannels == 1);

	double dRetVal = -INF;
	filter_t I( S1LayerParam[thetaIndex][sIndex].s * S1LayerParam[thetaIndex][sIndex].s); // Don't get confused by the data type 
	// Its a boost vector of double
	I.clear();

	int32_t cnt = 0;
	int32_t xstart = xpos;
	int32_t ystart = ypos;
	int32_t xend = xstart + S1LayerParam[thetaIndex][sIndex].s;
	int32_t yend = ystart + S1LayerParam[thetaIndex][sIndex].s;
	for ( int32_t l = xstart ; l < xend ; l++ )
	{
		for ( int32_t m = ystart ; m < yend ; m++ )
		{
			double val = double(cvGetReal2D(gray,l,m)) / double(NumGrayValues) ;
			I[cnt] = val;
			cnt++;	
		} // end of m loop

	} // end of l loop

	// Calculate S1 Layer (Gabor reponse)
	double num = b::inner_prod(*GaborFilterBank[thetaIndex][sIndex],I);
	double den = sqrt(b::inner_prod(I,I));

	double result = 0.0;
	if ( den != 0 )
	{
		result = num / den;
	}

	assert(!isnan(result));

	dRetVal = result;
	return dRetVal;

}


////////////////////////////////////////////////////////////////////////////////
/** comparision function for patchloc_t */
////////////////////////////////////////////////////////////////////////////////

bool patchlocLessthan(const patchloc_t &o1,const patchloc_t &o2)
{
	return (o1.imgID < o2.imgID); 
}


///////////////////////////////////////////////////////////////////////////////
/** readKeyFile : Read the Key File */
///////////////////////////////////////////////////////////////////////////////

void GaborFilterModel::readKeyFile(const string &kfile_) 
{
	if ( kfile_ == "" )
		{
			keyfile = pe->getParameter("KEYFILE");
		}
		else
		{
			keyfile = kfile_;
		}
		//imagedir = pe->getParameter("IMAGEDIR");
		std::ifstream kfile(keyfile.c_str()); 
		if(!kfile)
		{
			CERR << "Could not open file " << keyfile.c_str() << endl;
			exit(FILE_OPEN_FAILED);
		}

		filename.clear();
		category.clear();
		categorylist.clear();
		std::string line = "";
		std::string prevcat = "_sentinel_";
		while (true)
		{
			line = "";
			std::getline(kfile,line);
			if ( kfile.eof() )
			{
				break;
			}
			size_t pos = line.find(FDELIM);
			assert(pos != string::npos);
			fs::path imagepath(line.substr(pos+1));
			filename.push_back(fs::path(imagepath));
			std::string cat = "";
			cat = imagepath.branch_path().leaf();
			category.push_back(cat);
			if( prevcat != cat )
			{
				categorylist.push_back(cat);
				catnum[cat] = categorylist.size() - 1;
			}
			prevcat = cat;
		}	
		kfile.close();
		NumFiles = filename.size();
}


///////////////////////////////////////////////////////////////////////////////
/** Create the Key file for image locations */
///////////////////////////////////////////////////////////////////////////////

void GaborFilterModel::createKeyFile(const string &dir, const string &kfile)
{

	std::ofstream ofs(kfile.c_str());
	if (!ofs)
	{
		CERR << "Could not open file " << kfile.c_str() << endl;
		ofs.close();
		exit(FILE_OPEN_FAILED);
	}
	int64_t *id = new int64_t;
	*id = -1;  // Important;
	std::vector<fs::path> list;
	list.clear();
	dfsvisit(fs::path(dir),*id,list,ofs);
	CERR << "Key File :" << kfile.c_str() << " Size :: " << list.size() << endl << flush;
	delete id;
}


///////////////////////////////////////////////////////////////////////////////
/** Create patches at random positions (Serre-Poggio model) */
///////////////////////////////////////////////////////////////////////////////

void GaborFilterModel::createRandomPatches(const string& pposfile)
{

	if ( pposfile == "" )
	{
		patchposfile = pe->getParameter("PATCHPOSFILE");
	}
	else
	{
		patchposfile = pposfile;
	}
	ofstream ofs(patchposfile.c_str());
	if ( !ofs )
	{
		CERR << "Could not open file :" <<  patchposfile.c_str() << std::endl;
		ofs.close();
		exit(FILE_OPEN_FAILED);
	}

	Random<double> r1;
	std::string prevcat = category[0] ;
	int64_t start = 0;
	vpatchloc_t rp;
	rp.clear();
	for ( int64_t i = 0 ;  i < NumFiles ; i++ )
	{
		if ( category[i] != prevcat )
		{
#ifdef _DEBUG
			//cerr << start << " " << i-1  << " :: " ;
#endif

			createRandomPatches(i-1,start,rp);
			start = i; 
		}

		prevcat = category[i];

	}

#ifdef _DEBUG
	// cerr << start << " " << NumFiles-1 << endl;
#endif
	createRandomPatches(NumFiles-1,start,rp);
	CERR << rp.size() << " Patch locations created "  << std::endl;

	std::sort(rp.begin(),rp.end(),patchlocLessthan);

	int64_t rpsize = rp.size();
	for ( int64_t i = 0 ; i < rpsize ; i++ )
	{
		ofs << "[5](" << rp[i].imgID << "," << rp[i].bandNum   
			<< "," <<  rp[i].xpos << "," << rp[i].ypos << ","  
			<< rp[i].psize << ")" <<  std::endl;
	}

	CERR << rp.size() << " Patch locations pushed into " << patchposfile.c_str()  << std::endl << flush;
	ofs.close();

}


///////////////////////////////////////////////////////////////////////////////
/** Auxilary function for generating random patches */
///////////////////////////////////////////////////////////////////////////////

void GaborFilterModel::getRandomPos(const long bandNum, long &ipos, long &jpos, long &psizeNum)
{
	long  msize = ImageSize / S1LayerParam[0][2*bandNum].gridSize;
	long possiblePos = 0;
	for ( int64_t k = 0 ; k < NumPatchSizes ; k++ )
	{
		long diff = msize - PatchSizes[k] + 1;
		if ( diff > 0 )
		{
			possiblePos += (diff)*(diff);
		}
		else
		{
			break;
		}
	}

	Random<long> r3(possiblePos,0);
	long pnum1 = r3.nextRandom();
	long diff = 1;
	long pnum = pnum1;

	for ( int64_t k = 0 ; k < NumPatchSizes ; k++ )
	{
		diff = msize - PatchSizes[k] + 1;
		long diffsq = diff * diff;
		if ( pnum >= diffsq)
		{
			pnum -= diff * diff;
		}
		else
		{
			psizeNum = k;
			break;
		}
	}

	ipos = pnum / diff;
	jpos = pnum % diff;

	assert(bandNum >= 0 && bandNum <= NumBands-1);
	assert( 
			( (ipos + PatchSizes[psizeNum]) <= msize + 1 ) &&
			( (jpos + PatchSizes[psizeNum]) <= msize + 1 )
		  );


}


///////////////////////////////////////////////////////////////////////////////
/** createRandomPatches : Auxilary function for creating random patches */
///////////////////////////////////////////////////////////////////////////////

void GaborFilterModel::createRandomPatches(const long lb,const long ub, vpatchloc_t &rp)
{
	Random<long> r1(lb,ub);
	//Random<long> r1(i-1,start);
	assert ( lb + 1  - ub >= NumTrainingImages);
	Random<long> r2(NumBands,0);

	for ( int64_t j = 0 ; j < NumTrainingImages ; j++ )
	{
		long  imgNum =  r1.nextRandom(); // select images at random
		if ( version == "version2" )
		{
			imgNum = ub + j; // We have training set in a separate directory
						     // Pick from each image
		}
		assert(imgNum >= ub && imgNum <= lb);
		for ( int32_t k = 0 ; k < NumPatchesPerImage ; k++ )
		{
			long  bandNum = r2.nextRandom();
			long ipos = 0, jpos = 0, psizeNum = 0;
			getRandomPos(bandNum,ipos,jpos,psizeNum);
			patchloc_t pl;
			pl.imgID = imgNum;
			pl.bandNum = bandNum;
			pl.xpos = ipos;
			pl.ypos = jpos;
			pl.psize = PatchSizes[psizeNum];
			rp.push_back(pl);
			//	#ifdef _DEBUG
			/* cerr << imgNum   << ":" << bandNum     << ":" 
			   << PatchSizes[psizeNum] 
			   << ":" << ipos << ":" << jpos << "\t" ;
			   */
			//#endif
		}
	}

#ifdef _DEBUG
	//cerr << endl;
#endif
}


///////////////////////////////////////////////////////////////////////////////
/** calcFVStage1 : Stage 1 of the C2 computation */
///////////////////////////////////////////////////////////////////////////////

void GaborFilterModel::calcFVStage1(const string &s2filenm, const string &pposfilenm)
{
	if ( s2filenm == "" )
	{
		s2file =  pe->getParameter("S2FILE");
	}
	else
	{
		s2file = s2filenm;
	}
	ifstream ifs(s2file.c_str());
	if ( !ifs )
	{
		CERR << "Could not open file :" <<  s2file.c_str() << std::endl;
		ifs.close();
		exit(FILE_OPEN_FAILED);
	}

	if ( pposfilenm == "" )
	{
		patchposfile = pe->getParameter("PATCHPOSFILE");
	}
	else
	{
		patchposfile = pposfilenm;
	}
	ifstream ifs2(patchposfile.c_str());
	if ( !ifs2 )
	{
		CERR << "Could not open file :" <<  patchposfile.c_str() << std::endl;
		ifs2.close();
		exit(FILE_OPEN_FAILED);
	}


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
		vmmat.push_back(mmat);

	}

	CERR << vmmat.size() << " Records read from " << s2file.c_str() << endl << flush;

	vpatch.clear();
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
		bvmat_t opatch(NumOrientations); 	
		for ( int32_t j = 0 ; j < NumOrientations ; j++ )
		{
			opatch[j] = b::project(vmmat[pl[0]](j,pl[1]),b::range(pl[2],pl[2]+pl[4]),b::range(pl[3],pl[3]+pl[4]));
		}

		vpatch.push_back(opatch);
	}

	this->FVDim = vpatch.size();
	CERR << this->FVDim << " patches pushed from " << patchposfile.c_str() << endl << flush;

	ifs2.close();
	ifs.close();

}


///////////////////////////////////////////////////////////////////////////////
/** Reads a S2 file to bring S2 features in memory */
///////////////////////////////////////////////////////////////////////////////

void GaborFilterModel::readS2File(const string &s2filenm)
{
	vmmat.clear(); // Extremely important !!! Clear all training vectors

	if ( s2filenm == "" )
	{
		s2file =  pe->getParameter("S2FILE");
	}
	else
	{
		s2file = s2filenm;
	}

	ifstream ifs(s2file.c_str());
	if ( !ifs )
	{
		CERR << "Could not open file :" <<  s2file.c_str() << std::endl;
		ifs.close();
		exit(FILE_OPEN_FAILED);
	}

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
		vmmat.push_back(mmat);

	}

	CERR << vmmat.size() << " Records read from " << s2file.c_str() << endl;

}


///////////////////////////////////////////////////////////////////////////////
/** Stage 2 of the C2 computation */
///////////////////////////////////////////////////////////////////////////////

void GaborFilterModel::calcFVStage2(const string &c2file)
{
	int32_t rank = 0;
	int32_t numprocs = 1;
	std::stringstream s;
	string svmfile = "";

#ifdef _MPI
	numprocs = MPI::COMM_WORLD.Get_size();
	rank = MPI::COMM_WORLD.Get_rank();
	int hostnameLength;
	char processorName[MPI_MAX_PROCESSOR_NAME];
	MPI::Get_processor_name(processorName,hostnameLength);
#endif
	
	if ( c2file == "" )
	{
		svmfile = pe->getParameter("SVMFILE");
		s << svmfile << "_" << rank << ".c2";
	}
	else
	{
		svmfile = c2file;
		s << svmfile; 
	}


	int64_t lsize = NumFiles ;
	int64_t chunksize = lsize / numprocs;
	int64_t start = rank * chunksize;
	int64_t end = start + chunksize;
	//int32_t counter = start - 1 ;

#ifdef _MPI 
	CERR <<  processorName << " : PROCESS: " << rank << "  :: " << start << " " << end << endl;
#endif

	b::vector<vec_t> output(NumFiles);

	for ( int64_t i = 0 ; i < NumFiles ; i++ )
	{
		output[i] = b::zero_vector<double>(FVDim);
	}

	ofstream ofs(s.str().c_str());
#ifdef _OPENMP
#pragma omp parallel for 
#endif
	for ( int64_t i = start ; i < end ; i++ ) // for each file
	{
		for ( int64_t l = 0 ; l < FVDim ; l++ ) // for each patch
		{
			double max = -INF;
			for ( int64_t k = 0 ; k < NumBands ; k++ ) // for each band
			{

				bvmat_t inp(NumOrientations);
				for ( int64_t j = 0 ; j < NumOrientations ; j++ ) // for each orientation
				{
					inp[j] = vmmat[i](j,k);

				} // end of j loop

				double resp =  maxresp(inp,vpatch[l]) ;
				if ( resp > max )
				{
					max = resp;
				}

			} // end of k loop

			//cerr << max << " " ;
			output[i][l] = max;

		} // end of m loop

#ifdef _OPENMP
#pragma omp critical
#endif
		{
			ofs << i << " " << category[i] << " " << output[i] << std::endl ;
		}
		//cerr << endl;
	} // end of i loop

	//	 for ( int64_t i = 0 ; i < NumFiles ; i++ )
	//	 {
	//		 ofs << output[i] << std::endl ;
	//	 }
	ofs.close();


}


///////////////////////////////////////////////////////////////////////////////
/** Convert C2 features to libsvm format */
///////////////////////////////////////////////////////////////////////////////

void GaborFilterModel::convC2ToSVM(const string &c2file,const string &svmfile
		,const int32_t dim,const int32_t classid)
{
	ifstream infile(c2file.c_str());
	if(!infile)
	{
		CERR << "Could not open " << c2file.c_str() << endl;
		infile.close();
		exit(FILE_OPEN_FAILED);
	}

	ofstream outfile(svmfile.c_str());
	if(!outfile)
	{
		CERR << "Could not open " << svmfile.c_str() << endl;
		outfile.close();
		exit(FILE_OPEN_FAILED);
	}

	while(true)
	{
		int32_t imgID = -1; string cat = ""; vec_t v(dim);
		infile >> imgID >> cat >> v;
		if ( infile.eof() )
		{
			break;
		}
		if ( classid == -1 )
		{
			outfile << catnum[cat] << " ";
		}
		else
		{
			outfile << classid << " ";
		}
		for ( int32_t i = 0 ; i < dim ; i++ )
		{
			outfile << i << ":" << v[i] << " ";
		}
		outfile << endl;
	}

	infile.close();
	outfile.close();
}


///////////////////////////////////////////////////////////////////////////////
/** Auxilary function for Gabor filter response */
///////////////////////////////////////////////////////////////////////////////

double GaborFilterModel::maxresp(const bvmat_t &A,const bvmat_t &B)
{
	int32_t bsize1 = B[0].size1();
	int32_t bsize2 = B[0].size2();
	int32_t xlim = A[0].size1() - B[0].size1();
	int32_t ylim = A[0].size2() - B[0].size2();

	double min = INF;
	for ( int32_t i = 0 ; i <= xlim ; i++ )
	{
		for ( int32_t j = 0 ; j <= ylim ; j++ )
		{

			double sum = 0.0;

			for ( int32_t m = 0 ; m < NumOrientations ; m++ )
			{
				mat_t C = b::project(A[m],b::range(i,i+bsize1),b::range(j,j+bsize2)); 
				for ( int32_t k = 0 ; k < bsize1 ; k++ )
				{
					for ( int32_t l = 0 ; l < bsize2 ; l++ )
					{
						double t = ( C(k,l) - B[m](k,l) );
						sum += (t * t);
					} // end of l loop
				} // end of k loop


			} // end of m loop

			if ( sum < min)
			{
				min = sum;
			}

		} // end of j loop


	} // end of i loop

	return exp(-1.0 *min);

}


///////////////////////////////////////////////////////////////////////////////
/** Create patches at corners (shi-Tomasi) positions */
///////////////////////////////////////////////////////////////////////////////

void GaborFilterModel::createCornerPatches()
{
	CERR << " Numfiles: " << endl;
	Random<double> r1;
	std::string prevcat = category[0] ;
	int64_t start = 0;
	vpatchloc_t rp;
	rp.clear();
	for ( int64_t i = 0 ;  i < NumFiles ; i++ )
	{
		if ( category[i] != prevcat )
		{
#ifdef _DEBUG
			//cerr << start << " " << i-1  << " :: " ;
#endif

			createCornerPatches(i-1,start,rp);

			start = i; 
		}

		prevcat = category[i];

	}

#ifdef _DEBUG
	// cerr << start << " " << NumFiles-1 << endl;
#endif
	createCornerPatches(NumFiles-1,start,rp);
	CERR << rp.size() << " Patch locations pushed !!" << std::endl;

	std::sort(rp.begin(),rp.end(),patchlocLessthan);

	int64_t rpsize = rp.size();
	for ( int64_t i = 0 ; i < rpsize ; i++ )
	{
		std::cout << "[5](" << rp[i].imgID << "," << rp[i].bandNum   
			<< "," <<  rp[i].xpos << "," << rp[i].ypos << ","  
			<< rp[i].psize << ")" <<  std::endl;
	}


}


///////////////////////////////////////////////////////////////////////////////
/** Auxilary function for createCornerPatches */
///////////////////////////////////////////////////////////////////////////////

void GaborFilterModel::createCornerPatches(const long lb,const long ub, vpatchloc_t &rp)
{
	Random<long> r1(lb,ub);
	//Random<long> r1(i-1,start);
	assert ( lb + 1  - ub > NumTrainingImages);

	for ( int64_t j = 0 ; j < NumTrainingImages ; j++ )
	{
		long  imgNum =  r1.nextRandom();
		assert(imgNum >= ub && imgNum <= lb);
		getCornerPos(imgNum,rp);

	} // end of j loop 


}


///////////////////////////////////////////////////////////////////////////////
/** Calculates corner positions */
///////////////////////////////////////////////////////////////////////////////

void GaborFilterModel::getCornerPos( const long imgNum, vpatchloc_t &rp)
{
	Random<long> r2(NumBands,0);
	IplImage *img = 0;
	img = cvLoadImage(filename[imgNum].string().c_str());
	assert(img);
	// Get corners from an image
	IplImage *gray = 0,*grayt = 0, *grayt2 = 0;
	grayt = cvCreateImage(cvGetSize(img),IPL_DEPTH_8U,img->nChannels);
	assert(grayt);
	cvConvertScale(img,grayt);
	grayt2 = cvCreateImage(cvSize(ImageSize,ImageSize),IPL_DEPTH_8U,img->nChannels);
	assert(grayt2);
	cvResize(grayt,grayt2);
	cvReleaseImage(&grayt);
	gray = cvCreateImage(cvSize(ImageSize,ImageSize),IPL_DEPTH_8U,1);
	assert(gray);
	cvCvtColor(grayt2,gray,CV_BGR2GRAY);
	cvReleaseImage(&grayt2);
	CvMat *img2 = cvCreateMat(ImageSize,ImageSize,CV_32FC1);
	CvMat *img3 = cvCreateMat(ImageSize,ImageSize,CV_32FC1);
	int32_t cornerCount = MaxCorners;
	CvPoint2D32f *corners = new CvPoint2D32f[MaxCorners];
	cvGoodFeaturesToTrack(gray,img2,img3,corners,&cornerCount,GFTTQualityLevel,GFTTMinDistance);
	int32_t count = 0;
	for ( int32_t j = 0 ; j < cornerCount ; j++ ) // for each corner
	{

		patchloc_t pl;
		pl.imgID = imgNum;
		long  bandNum = r2.nextRandom();
		long msize1 = S1LayerParam[0][2*bandNum].gridSize;
		long  msize = ImageSize / msize1;
		pl.bandNum = bandNum;
		pl.xpos = long(corners[j].x / msize1);
		pl.ypos = long(corners[j].y / msize1);
		// Now that bandNum , xpos and ypos are fixed
		// can we extract a patch ?
		int32_t possible = -1;
		for ( int32_t k = 0 ; k < NumPatchSizes ; k++ )
		{
			int32_t half = PatchSizes[k] / 2;
			if ( \
					( pl.xpos < half ) || ( pl.ypos < half ) || \
					( msize - pl.xpos < half ) || ( msize - pl.ypos < half ) \
			   )
			{
				break;
			}
			else
			{
				possible++;
			}
		}

		if ( possible == -1 )
		{
			continue;
		}
		else
		{

			Random<long> rpatch(possible,0);
			pl.psize = PatchSizes[rpatch.nextRandom()];
			int32_t phalf = pl.psize / 2;
			pl.xpos -= phalf;
			pl.ypos -= phalf;

			// Check for duplicates
			vpatchlocitr_t vitr = std::find(rp.begin(),rp.end(),pl);
			if ( vitr != rp.end() )
			{
				continue;
			}
			else
			{
				assert ( ( pl.xpos >= 0 ) && \
						( pl.ypos >= 0 ) && \
						( msize - pl.xpos >= pl.psize ) && \
						( msize - pl.ypos >= pl.psize ) \
					   );
				rp.push_back(pl);
				count++;
				if ( count >= NumPatchesPerImage )
				{
					break;
				}
			}
		}
	}

	cvReleaseMat(&img2);
	cvReleaseMat(&img3);
	cvReleaseImage(&gray);
	cvReleaseImage(&img);
	delete [] corners;

}


///////////////////////////////////////////////////////////////////////////////
/** Creating patches at edges */
////////////////////////////////////////////////////////////////////////////////

void GaborFilterModel::createEdgePatches()
{
	for ( int64_t i = 0 ; i <  NumFiles ; i++ )
	{
		CERR << filename[i].string().c_str() << endl;
		IplImage *img = 0, *out = 0;
		img = cvLoadImage(filename[i].string().c_str());
		out = cvCreateImage(isz,IPL_DEPTH_8U,1);
		getEdgePoints(img, out);
		fs::path catdir = fs::path(cannydir) / category[i];
		if ( ! fs::exists(catdir) )
		{
			fs::create_directory(catdir);
		}
		fs::path outfile = catdir / filename[i].leaf();
		cvSaveImage(outfile.string().c_str(), out);
		cvReleaseImage(&img);
		cvReleaseImage(&out);
	}
}


///////////////////////////////////////////////////////////////////////////////
/** Auxilary function for createEdgePatches */
///////////////////////////////////////////////////////////////////////////////

void GaborFilterModel::getEdgePoints(IplImage *img,IplImage *out)
{
	assert(img);
	IplImage *img2 = 0, *img3 = 0, *img4 = 0;
	img2 = cvCreateImage(isz, img->depth, img->nChannels);
	cvResize(img,img2);	
	img3 = cvCreateImage(isz, IPL_DEPTH_8U,img2->nChannels);
	cvConvertScale(img2,img3);
	cvReleaseImage(&img2);
	if ( img3->nChannels == 3 )
	{
		img4 = cvCreateImage(isz,IPL_DEPTH_8U,1);
		cvCvtColor(img3,img4,CV_BGR2GRAY);
	}
	else if ( img3->nChannels == 1 )
	{
		img4  = cvCloneImage(img3);
	}
	else
	{
		CERR << "Incorrect nChannels value " << img3->nChannels << endl;
		exit(INCORRECT_VALUE);
	}
	cvReleaseImage(&img3);
	cvCanny(img4,out,CannyUpperThreshold,CannyLowerThreshold);


}


///////////////////////////////////////////////////////////////////////////////
/** Create patches over an overlapping grid */
///////////////////////////////////////////////////////////////////////////////

void GaborFilterModel::createOverlappingGridPatches()
{
	int64_t halfsz = CodebookPatchSize / 2;
	//int64_t start =  CodebookPatchSize;
	int64_t start =  0;
	int64_t start2 = start + halfsz;

	for ( int64_t i = 0 ; i < NumFiles ; i++ ) // for each band
	{

		//for ( int64_t j = 0 ; j < NumOrientations ; j++ ) // for each orientation
		//{
		for ( int64_t k = 0 ; k < NumBands ; k++ ) // for each band
		{
			int64_t ulimit = S1LayerParam[0][2*k].s2cellsize - CodebookPatchSize;
			int64_t end = ulimit - CodebookPatchSize;
			int64_t end2 = end - halfsz;

			for ( int64_t l = start ; l <= end ; l+= CodebookPatchSize )
			{
				for ( int64_t m = start ; m <= end ; m+= CodebookPatchSize )
				{
					cerr << "[5](" << i << "," << k << "," << l 
						<< "," << m << "," << CodebookPatchSize << ")" << endl; 
				} // end of m loop
			} // end of l loop


			for ( int64_t l = start2 ; l <= end2 ; l+= CodebookPatchSize )
			{
				for ( int64_t m = start2 ; m <= end2 ; m+= CodebookPatchSize )
				{
					cerr << "[5](" << i << "," << k << "," << l 
						<< "," << m << "," << CodebookPatchSize << ")" << endl; 
				} // end of m loop
			} // end of l loop


		} // end of k loop

		//} // end of j loop
	} // end of i loop

}


///////////////////////////////////////////////////////////////////////////////
/** Create patches at all possible positions */
////////////////////////////////////////////////////////////////////////////////

void GaborFilterModel::createAllPatches()
{
	//int64_t start =  CodebookPatchSize;
	int64_t start =  0;

	for ( int64_t i = 0 ; i < NumFiles ; i++ ) // for each file
	{

		for ( int64_t k = 0 ; k < NumBands ; k++ ) // for each band
		{
			int64_t ulimit = S1LayerParam[0][2*k].s2cellsize - CodebookPatchSize;
			int64_t end = ulimit ;

			for ( int64_t l = start ; l <= end ; l+= 1 )
			{
				for ( int64_t m = start ; m <= end ; m+= 1 )
				{
					cerr << "[5](" << i << "," << k << "," << l 
						<< "," << m << "," << CodebookPatchSize << ")" << endl; 
				} // end of m loop
			} // end of l loop

		} // end of k loop

	} // end of i loop

}


///////////////////////////////////////////////////////////////////////////////
/** Create all possible patches from training images */
////////////////////////////////////////////////////////////////////////////////

void GaborFilterModel::createAllPatchesTrain()
{
	//int64_t start =  CodebookPatchSize;
	int64_t start =  0;
	std::map<std::string,int64_t> categorywiseCount;
	categorywiseCount.clear();
	int32_t clistsz = categorylist.size();
	for ( int32_t i = 0 ; i < clistsz ; i++ )
	{
		categorywiseCount[categorylist[i]] = 0;
	}

	for ( int64_t i = 0 ; i < NumFiles ; i++ ) // for each file
	{
		categorywiseCount[category[i]] += 1;
		if ( categorywiseCount[category[i]] <= NumFilesPerCat )
		{
			for ( int64_t k = 0 ; k < NumBands ; k++ ) // for each band
			{
				int64_t ulimit = S1LayerParam[0][2*k].s2cellsize - CodebookPatchSize;
				int64_t end = ulimit ;

				for ( int64_t l = start ; l <= end ; l+= 1 )
				{
					for ( int64_t m = start ; m <= end ; m+= 1 )
					{
						cerr << "[5](" << i << "," << k << "," << l 
							<< "," << m << "," << CodebookPatchSize << ")" << endl; 
					} // end of m loop

				} // end of l loop

			} // end of k loop

		}

	} // end of i loop

}


///////////////////////////////////////////////////////////////////////////////
/** Create a bag-of-words kind of model */
///////////////////////////////////////////////////////////////////////////////

void GaborFilterModel::formCodeBook()
{

	// Open S2 features file
	s2file =  pe->getParameter("S2FILE");
	ifstream ifs(s2file.c_str());
	if ( !ifs )
	{
		CERR << "Could not open file :" <<  s2file.c_str() << std::endl;
		ifs.close();
		exit(FILE_OPEN_FAILED);
	}

	// Open Patch position file
	patchposfile = pe->getParameter("PATCHPOSFILE");
	ifstream ifs2(patchposfile.c_str());
	if ( !ifs2 )
	{
		CERR << "Could not open file :" <<  patchposfile.c_str() << std::endl;
		ifs2.close();
		exit(FILE_OPEN_FAILED);
	}


	// Read all the s2 patches
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
		vmmat.push_back(mmat);

	}

	CERR << vmmat.size() << " Records read" << endl;

	// Crude line count implementation
	// Required of memory allocation of CvMat
	// for the cvKMeans2 function
	int64_t cnt = -1;
	for (std::string s = ""; !ifs2.eof() ; getline(ifs2,s),cnt++);
	ifs2.close();
	ifs2.open(patchposfile.c_str());
	CERR << "File has " << cnt << " patches"<< endl;
	this->FVDim = cnt;

	int64_t ptsize = NumOrientations * CodebookPatchSize * CodebookPatchSize;
	int64_t datasize = ptsize * (this->FVDim);
	float *pts = new float[datasize];
	memset(pts,0,datasize*sizeof(float));

	vivec_t vpl;
	vpl.clear();

	// Read patch position from patchposition file
	// Fetch the patch from s2 features space
	// Prepare data for K-means
	for ( int64_t i = 0 ; i < this->FVDim ; i++ )
	{
		ivec_t	pl(5);
		ifs2 >> pl ;
		vpl.push_back(pl);

#ifdef _DEBUG
		CERR << pl << endl;
#endif

		for ( int32_t j = 0 ; j < NumOrientations ; j++ )
		{
			mat_t opatch = b::project(vmmat[pl[0]](j,pl[1]),b::range(pl[2],pl[2]+pl[4]),b::range(pl[3],pl[3]+pl[4]));
			for ( int32_t k = 0 ; k < CodebookPatchSize ; k++ )
			{
				for ( int32_t l = 0 ; l < CodebookPatchSize ; l++ )
				{
					// int64_t offset =  (((i)*NumOrientations+j)*CodebookPatchSize+k)*CodebookPatchSize+l; 
					int64_t offset = l + CodebookPatchSize * 
						( k + CodebookPatchSize * ( j + NumOrientations * ( i ) )   );
					pts[offset] = float(opatch(k,l));
				} // end of l loop
			} // end of k loop
		}

	} // end of i loop

	CvMat points, clusters;
	int32_t *clstr = new int32_t[(this->FVDim) * sizeof(int32_t)];
	memset(clstr,0,(this->FVDim) * sizeof(int32_t));
	cvInitMatHeader(&clusters,this->FVDim,1,CV_32SC1,clstr);
	cvInitMatHeader(&points,this->FVDim,ptsize,CV_32FC1,pts);
	// The definition of CvKMeans2 has changed in the newer version
#if CV_MAJOR_VERSION == 0 || ( CV_MAJOR_VERSION == 1  && CV_MINOR_VERSION == 0  )
	cvKMeans2(&points,NumWords,&clusters,cvTermCriteria(CV_TERMCRIT_EPS,0,1.0));
#else
	cvKMeans2(&points,NumWords,&clusters,cvTermCriteria(CV_TERMCRIT_EPS,0,1.0),
			5,0,0,0,0);
#endif

	b::vector<int64_t> hst(NumWords);
	for ( int32_t i = 0 ; i < NumWords ; i++ )
	{
		hst[i] = 0;
	}

	// This is the category-patch-cluster(word) mapping
	std::vector<int32_t> ptword;
	ptword.clear();
	for ( int64_t i = 0 ; i < this->FVDim ; i++ )
	{
		int32_t clusterID =  cvGetReal2D(&clusters,i,0);
		ptword.push_back(clusterID);
		// cout << catnum[category[vpl[i][0]]] << " " << category[vpl[i][0]] 
		//	 << " " << vpl[i] << " " << clusterID << endl;
		hst[clusterID] = hst[clusterID] + 1;
	} // end of i loop



	cout << "Histogram :" << hst << endl;

	// Top 3 clusters have to be eliminated
	std::vector<int32_t> ElemClusterIdx;
	std::vector<int64_t> RemovedWordCnt;
	ElemClusterIdx.clear();
	RemovedWordCnt.clear();

	for ( int32_t i = 0 ; i < NumClustersEliminated ; i++ )
	{
		int32_t elidx = 0;
		int64_t maxel = hst[0];
		for ( int32_t j = 0 ; j < NumWords ; j++ )
		{
			if ( hst[j] > maxel )
			{
				maxel = hst[j];
				elidx = j;
			}
		} // end of j loop 
		// std::swap(hst[elidx],hst[NumWords-i-1]); // incorrect
		ElemClusterIdx.push_back(elidx);
		RemovedWordCnt.push_back(hst[elidx]);
		hst[elidx] = -INF;
	} // end of i loop


	std::sort(ElemClusterIdx.begin(),ElemClusterIdx.end());
	CERR << "Elements to be removed (indices) in sorted order: " ;
	for ( int32_t i = 0 ; i < NumClustersEliminated ; i++ )
	{
		cerr << ElemClusterIdx[i] << " ";
	} // end of i loop 
	cerr << endl;

	CERR << "Frequencies removed in descending order: " ;
	for ( int32_t i = 0 ; i < NumClustersEliminated ; i++ )
	{
		cerr << RemovedWordCnt[i] << " ";
	} // end of i loop 
	cerr << endl;



	// Remove the patches that were classified to top-4 cluster centroids
	vivec_t vpl2;
	vpl2.clear();
	int64_t vplsize = vpl.size();
	std::vector<int32_t>::iterator eciitr;
	for ( int64_t i = 0 ; i < vplsize ; i++ ) // for every patch
	{
		eciitr = std::find(ElemClusterIdx.begin(),ElemClusterIdx.end(),ptword[i]);
		if ( eciitr == ElemClusterIdx.end())  // patch has not been assigned to top-4 clusters
		{
			vpl2.push_back(vpl[i]);
		}
	} // end of i loop
	vpl.clear(); // should not be used beyond this point
	delete [] pts;
	delete [] clstr;

	int64_t vpl2size = vpl2.size();

	CERR << vpl2size << " patches remaining after the pruning " <<  endl;

	// Now we apply K-Means again on the pruned dataset but with lower threshold value

	// Get the patches again ... indices have changed

	int64_t datasize2 = ptsize * vpl2size;
	float *pts2 = new float[datasize2];
	memset(pts2,0,datasize2*sizeof(float));

	for ( int64_t i = 0 ; i < vpl2size ; i++ )
	{
		ivec_t	pl(5);
		pl = vpl2[i];

#ifdef _DEBUG
		CERR << pl << endl;
#endif
		for ( int32_t j = 0 ; j < NumOrientations ; j++ )
		{
			mat_t opatch = b::project(vmmat[pl[0]](j,pl[1]),b::range(pl[2],pl[2]+pl[4]),b::range(pl[3],pl[3]+pl[4]));
			for ( int32_t k = 0 ; k < CodebookPatchSize ; k++ )
			{
				for ( int32_t l = 0 ; l < CodebookPatchSize ; l++ )
				{
					// int64_t offset =  (((i)*NumOrientations+j)*CodebookPatchSize+k)*CodebookPatchSize+l; 
					int64_t offset = l + CodebookPatchSize * 
						( k + CodebookPatchSize * ( j + NumOrientations * ( i ) )   );
					pts2[offset] = float(opatch(k,l));
				} // end of l loop
			} // end of k loop
		}
	} // end of i loop

	// And now the K-means .... (second time)
	CvMat points2, clusters2;
	int32_t *clstr2 = new int32_t[vpl2size * sizeof(int32_t)];
	memset(clstr2,0,vpl2size * sizeof(int32_t));
	cvInitMatHeader(&clusters2,vpl2size,1,CV_32SC1,clstr2);
	cvInitMatHeader(&points2,vpl2size,ptsize,CV_32FC1,pts2);

	TIME(t5);
	// The definition of KMeans2 has changed in the newer version of OpenCV
#if CV_MAJOR_VERSION == 0 || ( CV_MAJOR_VERSION == 1  && CV_MINOR_VERSION == 0  )
	cvKMeans2(&points,NumWords,&clusters,cvTermCriteria(CV_TERMCRIT_EPS,0,1.0));
#else
	cvKMeans2(&points2,NumWords2,&clusters2,cvTermCriteria(CV_TERMCRIT_EPS,0,0.01),
			5,0,0,0,0);
#endif
	TIME(t6);
	CERR << "KMeans2 took " <<  difftime(t5,t6) << " ms" << endl;

	// This is the new (and hopefully improved) category-patch-cluster(word) mapping
	std::vector<int32_t> ptword2;
	ptword2.clear();
	for ( int64_t i = 0 ; i < vpl2size ; i++ )
	{
		int32_t clusterID =  cvGetReal2D(&clusters2,i,0);
		ptword.push_back(clusterID);
		cout << catnum[category[vpl2[i][0]]] << " " << category[vpl2[i][0]] 
			<< " " << vpl2[i] << " " << clusterID << endl;
	} // end of i loop


	// Cluster centroids or the words 
	//	Matrix centers2(NumWords2,ptsize);
	//	for ( int64_t i = 0 ; i < NumWords2 ; i++ )
	//	{
	//		for ( int64_t j = 0 ; j < ptsize ; j++)
	//		{
	//			//cout << cvGetReal2D(&points,i,j) << " ";
	//			centers2(i,j) = cvGetReal2D(&points2,i,j);
	//		} // end of j loop
	//		//cout << endl;
	//	} // end of i loop
	//	//cout << centers2 << endl;
	//

	ofstream ofs(wordsfile.c_str()); 
	for ( int64_t i = 0 ; i < NumWords2 ; i++ )
	{
		bvmat_t wctr(NumOrientations);
		for ( int32_t  j = 0 ; j < NumOrientations ; j++ )
		{
			mat_t mt(CodebookPatchSize,CodebookPatchSize);
			for ( int32_t k = 0 ; k < CodebookPatchSize ; k++ )
			{
				for ( int32_t l = 0 ; l < CodebookPatchSize ; l++ )
				{
					int32_t offset =  ((j)*CodebookPatchSize+k)*CodebookPatchSize+l;
					mt(k,l) = cvGetReal2D(&points2,i,offset); 
				} // end of l loop
			} // end of k loop
			wctr[j] = mt;
		} // end of j loop
		//vpatch.push_back(wctr);
		ofs << wctr << endl;

	} // end of i loop
	ofs.close();

#ifdef _DEBUG
	//	cout << endl << "WORDS:" << endl;
	//	Matrix centers2(NumWords2,ptsize);
	//	for ( int64_t i = 0 ; i < NumWords2 ; i++ )
	//	{
	//		for ( int64_t j = 0 ; j < ptsize ; j++)
	//		{
	//			//cout << cvGetReal2D(&points,i,j) << " ";
	//			centers2(i,j) = cvGetReal2D(&points2,i,j);
	//		} // end of j loop
	//		cout << centers2.row(i);
	//		cout << endl;
	//		cout << vpatch[i] << endl << endl << endl;
	//	} // end of i loop
	//	//cout << centers2 << endl;
#endif


	delete [] pts2;
	delete [] clstr2;
	ifs2.close();
	ifs.close();

#if 0
	//  Uncomment to see distance between words
	//	Matrix dotpr = 2.0 * centers2 * centers2.transpose();
	//	Matrix C = centers2.sumsq(1);
	//	Matrix D(NumWords2,NumWords2);
	//	double max = 0.0; // distance is always non-negative
	//	for ( int64_t i = 0 ; i < NumWords2 ; i++ )
	//	{
	//		for ( int64_t j = 0 ; j < NumWords2 ; j++ )
	//		{
	//			D(i,j) = C(i) + C(j) - dotpr(i,j);
	//			if ( D(i,j) > max )
	//			{
	//				max = D(i,j);
	//			}
	//		}
	//	}
	//
	//	cout << " Distance between words :" << endl
	//		 <<  D  / max << endl;
	//
#endif

}


///////////////////////////////////////////////////////////////////////////////
/** Stage 2 of the words model */
////////////////////////////////////////////////////////////////////////////////

void GaborFilterModel::calcFVStage2Words()
{
	// Now create vectors for SVM ...
	// almost replicating cvCalcFVStage2()
	std::string svmfile = pe->getParameter("SVMFILE");
	int32_t rank = 0;
	int32_t numprocs = 1;
	std::stringstream s;
#ifdef _MPI
	int argc = 0;
	char **argv = 0;	
	MPI::Init(argc,argv);
	numprocs = MPI::COMM_WORLD.Get_size();
	rank = MPI::COMM_WORLD.Get_rank();
	int hostnameLength;
	char processorName[MPI_MAX_PROCESSOR_NAME];
	MPI::Get_processor_name(processorName,hostnameLength);
#endif

	s << svmfile << "__" << rank << "__" << "serrepoggio"<< "__" << NumWords2 << ".txt";

	int64_t lsize = NumFiles ;
	int64_t chunksize = lsize / numprocs;
	int64_t start = rank * chunksize;
	int64_t end = start + chunksize;
	//int32_t counter = start - 1 ;

#ifdef _MPI 
	CERR <<  processorName << " : PROCESS: " << rank << "  :: " << start << " " << end << endl;
#endif


	ifstream ifs(wordsfile.c_str());
	if (!ifs)
	{
		CERR << "Could not open file " << wordsfile.c_str() << endl;
		exit(FILE_OPEN_FAILED);
	}

	vpatch.clear();	
	while(true)
	{
		bvmat_t wctr(NumOrientations);
		ifs >> wctr;

		if ( ifs.eof() )
		{
			break;
		}
		vpatch.push_back(wctr);
	}
	ifs.close();

	CERR << vpatch.size() << " words read from the file" << endl << flush;

	// Open S2 features file
	s2file =  pe->getParameter("S2FILE");
	ifstream ifs2(s2file.c_str());
	if ( !ifs2 )
	{
		CERR << "Could not open file :" <<  s2file.c_str() << std::endl;
		ifs2.close();
		exit(FILE_OPEN_FAILED);
	}

	// Read all the s2 patches
	for ( int64_t i = 0 ; i < NumFiles ; i++ ) // for each file
	{
		int64_t imgID;
		ifs2 >> imgID;
		std::string cat;
		ifs2 >> cat;
#ifdef _DEBUG
		CERR << imgID << " " << cat << endl;
#endif
		mmat_t mmat(NumOrientations,NumBands);
		ifs2 >> mmat;
		vmmat.push_back(mmat);

	}

	ifs2.close();
	CERR << vmmat.size() << " S2 Records read" << endl;


	// Now form the vectors for SVM
	// Distribute the job across the processors using OpenMPI and OpenMP

	b::vector<vec_t> output(NumFiles);
	for ( int64_t i = 0 ; i < NumFiles ; i++ )
	{
		output[i] = b::zero_vector<double>(NumWords2);
	}

	ofstream ofs(s.str().c_str());
	CERR << "Started at  " << __DATE__ << " " << __TIME__ << endl;
	TIME(t1);
#ifdef _OPENMP
#pragma omp parallel for 
#endif
	for ( int64_t i = start ; i < end ; i++ ) // for each file
	{
		for ( int64_t l = 0 ; l < NumWords2 ; l++ ) // for each patch ... each word in this case
		{
			double max = -INF;
			for ( int64_t k = 0 ; k < NumBands ; k++ ) // for each band
			{

				bvmat_t inp(NumOrientations);
				for ( int64_t j = 0 ; j < NumOrientations ; j++ ) // for each orientation
				{
					inp[j] = vmmat[i](j,k);

				} // end of j loop

				double resp =  maxresp(inp,vpatch[l]) ;
				if ( resp > max )
				{
					max = resp;
				}

			} // end of k loop

			//cerr << max << " " ;
			output[i][l] = max;

		} // end of m loop

#ifdef _OPENMP
#pragma omp critical
#endif
		{
			ofs << i << " " << category[i] << " " << output[i] << std::endl ;
		}
		//cerr << endl;
	} // end of i loop

	//	 for ( int64_t i = 0 ; i < NumFiles ; i++ )
	//	 {
	//		 ofs << output[i] << std::endl ;
	//	 }

	TIME(t2);
	CERR << "Ended at  " << __DATE__ << " " << __TIME__ << endl;
	CERR << "It took : " << difftime(t1,t2) << " msec "  << endl;
	ofs.close();

#ifdef _MPI
	MPI::Finalize();
#endif

}

///////////////////////////////////////////////////////////////////////////////
/** Run Earth Mover's distance  */
///////////////////////////////////////////////////////////////////////////////

void GaborFilterModel::runKernelEMD()
{
	// Open S2 features file
	s2file =  pe->getParameter("S2FILE");
	ifstream ifs(s2file.c_str());
	if ( !ifs )
	{
		CERR << "Could not open file :" <<  s2file.c_str() << std::endl;
		ifs.close();
		exit(FILE_OPEN_FAILED);
	}

	// Open Patch position file
	patchposfile = pe->getParameter("PATCHPOSFILE");
	ifstream ifs2(patchposfile.c_str());
	if ( !ifs2 )
	{
		CERR << "Could not open file :" <<  patchposfile.c_str() << std::endl;
		ifs2.close();
		exit(FILE_OPEN_FAILED);
	}


	// Read all the s2 patches
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
		vmmat.push_back(mmat);

	}

	CERR << vmmat.size() << " Records read" << endl;

	// Crude line count implementation
	// Required of memory allocation of CvMat
	// for the cvKMeans2 function
	int64_t cnt = -1;
	for (std::string s = ""; !ifs2.eof() ; getline(ifs2,s),cnt++);
	ifs2.close();
	ifs2.open(patchposfile.c_str());
	CERR << "File has " << cnt << " patches"<< endl;
	this->FVDim = cnt;

	int64_t ptsize = NumOrientations * CodebookPatchSize * CodebookPatchSize;
	int64_t datasize = ptsize * (this->FVDim);
	float *pts = new float[datasize];
	memset(pts,0,datasize*sizeof(float));

	vivec_t vpl;
	vpl.clear();

	// Read patch position from patchposition file
	// Fetch the patch from s2 features space
	// Prepare data for K-means
	for ( int64_t i = 0 ; i < this->FVDim ; i++ )
	{
		ivec_t	pl(5);
		ifs2 >> pl ;
		vpl.push_back(pl);

#ifdef _DEBUG
		CERR << pl << endl;
#endif

		for ( int32_t j = 0 ; j < NumOrientations ; j++ )
		{
			mat_t opatch = b::project(vmmat[pl[0]](j,pl[1]),b::range(pl[2],pl[2]+pl[4]),b::range(pl[3],pl[3]+pl[4]));
			for ( int32_t k = 0 ; k < CodebookPatchSize ; k++ )
			{
				for ( int32_t l = 0 ; l < CodebookPatchSize ; l++ )
				{
					// int64_t offset =  (((i)*NumOrientations+j)*CodebookPatchSize+k)*CodebookPatchSize+l; 
					int64_t offset = l + CodebookPatchSize * 
						( k + CodebookPatchSize * ( j + NumOrientations * ( i ) )   );
					pts[offset] = float(opatch(k,l));
				} // end of l loop
			} // end of k loop
		}

	} // end of i loop


	CvMat points, clusters;
	int32_t *clstr = new int32_t[(this->FVDim) * sizeof(int32_t)];
	memset(clstr,0,(this->FVDim) * sizeof(int32_t));
	cvInitMatHeader(&clusters,this->FVDim,1,CV_32SC1,clstr);
	cvInitMatHeader(&points,this->FVDim,ptsize,CV_32FC1,pts);
	// The definition of CvKMeans2 has changed in the newer version
#if CV_MAJOR_VERSION == 0 || ( CV_MAJOR_VERSION == 1  && CV_MINOR_VERSION == 0  )
	cvKMeans2(&points,NumWords,&clusters,cvTermCriteria(CV_TERMCRIT_EPS,0,KMeansThreshold1));
#else
	cvKMeans2(&points,NumWords,&clusters,cvTermCriteria(CV_TERMCRIT_EPS,0,KMeansThreshold1),
			5,0,0,0,0);
#endif

	b::vector<int64_t> hst(NumWords);
	for ( int32_t i = 0 ; i < NumWords ; i++ )
	{
		hst[i] = 0;
	}

	// This is the category-patch-cluster(word) mapping
	std::vector<int32_t> ptword;
	ptword.clear();
	for ( int64_t i = 0 ; i < this->FVDim ; i++ )
	{
		int32_t clusterID =  cvGetReal2D(&clusters,i,0);
		ptword.push_back(clusterID);
		// cout << catnum[category[vpl[i][0]]] << " " << category[vpl[i][0]] 
		//	 << " " << vpl[i] << " " << clusterID << endl;
		hst[clusterID] = hst[clusterID] + 1;
	} // end of i loop



	cout << "Histogram :" << hst << endl;

	// Top 4 clusters have to be eliminated
	std::vector<int32_t> ElemClusterIdx;
	std::vector<int64_t> RemovedWordCnt;
	ElemClusterIdx.clear();
	RemovedWordCnt.clear();

	for ( int32_t i = 0 ; i < NumClustersEliminated ; i++ )
	{
		int32_t elidx = 0;
		int64_t maxel = hst[0];
		for ( int32_t j = 0 ; j < NumWords ; j++ )
		{
			if ( hst[j] > maxel )
			{
				maxel = hst[j];
				elidx = j;
			}
		} // end of j loop 
		// std::swap(hst[elidx],hst[NumWords-i-1]); // incorrect
		ElemClusterIdx.push_back(elidx);
		RemovedWordCnt.push_back(hst[elidx]);
		hst[elidx] = -INF;
	} // end of i loop


	std::sort(ElemClusterIdx.begin(),ElemClusterIdx.end());
	CERR << "Elements to be removed (indices) in sorted order: " ;
	for ( int32_t i = 0 ; i < NumClustersEliminated ; i++ )
	{
		cerr << ElemClusterIdx[i] << " ";
	} // end of i loop 
	cerr << endl;

	CERR << "Frequencies removed in descending order: " ;
	for ( int32_t i = 0 ; i < NumClustersEliminated ; i++ )
	{
		cerr << RemovedWordCnt[i] << " ";
	} // end of i loop 
	cerr << endl;

	// Remove the patches that were classified to top-4 cluster centroids
	vivec_t vpl2;
	vpl2.clear();
	int64_t vplsize = vpl.size();
	std::vector<int32_t>::iterator eciitr;
	for ( int64_t i = 0 ; i < vplsize ; i++ ) // for every patch
	{
		eciitr = std::find(ElemClusterIdx.begin(),ElemClusterIdx.end(),ptword[i]);
		if ( eciitr == ElemClusterIdx.end())  // patch has not been assigned to top-4 clusters
		{
			vpl2.push_back(vpl[i]);
		}
	} // end of i loop
	vpl.clear(); // should not be used beyond this point
	vpl.~vector();
	ElemClusterIdx.clear();
	RemovedWordCnt.clear();
	ElemClusterIdx.~vector();
	RemovedWordCnt.~vector();
	hst.clear();
	hst.~vector();
	delete [] pts;
	delete [] clstr;


	int64_t vpl2size = vpl2.size();

	CERR << vpl2size << " patches remaining after the pruning " <<  endl;

	// Now we apply K-Means again on the pruned dataset but with lower threshold value

	// Get the patches again ... indices have changed

	int64_t datasize2 = ptsize * vpl2size;
	float *pts2 = new float[datasize2];
	memset(pts2,0,datasize2*sizeof(float));

	for ( int64_t i = 0 ; i < vpl2size ; i++ )
	{
		ivec_t	pl(5);
		pl = vpl2[i];

#ifdef _DEBUG
		//		CERR << pl << endl;
#endif
		for ( int32_t j = 0 ; j < NumOrientations ; j++ )
		{
			mat_t opatch = b::project(vmmat[pl[0]](j,pl[1]),b::range(pl[2],pl[2]+pl[4]),b::range(pl[3],pl[3]+pl[4]));
			for ( int32_t k = 0 ; k < CodebookPatchSize ; k++ )
			{
				for ( int32_t l = 0 ; l < CodebookPatchSize ; l++ )
				{
					// int64_t offset =  (((i)*NumOrientations+j)*CodebookPatchSize+k)*CodebookPatchSize+l; 
					int64_t offset = l + CodebookPatchSize * 
						( k + CodebookPatchSize * ( j + NumOrientations * ( i ) )   );
					pts2[offset] = float(opatch(k,l));
				} // end of l loop
			} // end of k loop
		}
	} // end of i loop


	vmmat.clear(); // We are short of memory ; S2 files you need to go out !
	vmmat.~vector();

	// And now the K-means .... (second time)
	CvMat points2, clusters2;
	int32_t *clstr2 = new int32_t[vpl2size * sizeof(int32_t)];
	memset(clstr2,0,vpl2size * sizeof(int32_t));
	cvInitMatHeader(&clusters2,vpl2size,1,CV_32SC1,clstr2);
	cvInitMatHeader(&points2,vpl2size,ptsize,CV_32FC1,pts2);
#if 0
	float *test = new float[3*1000000*64+5000*64+5000*64*64+10000]; 
	CERR << "New returned: " << test << endl << flush;
	test[0] = 1;
	float tmp = test[12];
	CERR << "Memory Test: " << tmp << endl << flush;
#endif

	TIME(t7);
	// The definition of KMeans2 has changed in the newer version of OpenCV
#if CV_MAJOR_VERSION == 0 || ( CV_MAJOR_VERSION == 1  && CV_MINOR_VERSION == 0  )
	cvKMeans2(&points,NumWords,&clusters,cvTermCriteria(CV_TERMCRIT_EPS,0,KMeansThreshold2));
#else
	cvKMeans2(&points2,NumWords2,&clusters2,cvTermCriteria(CV_TERMCRIT_EPS,0,KMeansThreshold2),
			5,0,0,0,0);
#endif
	TIME(t8);
	CERR << "KMeans2 took " << difftime(t7,t8) << " ms" << endl;

#if 0
	TIME(t11);
	CvEM em_model;
	CvEMParams params;
	// initialize model's parameters
	params.covs      = NULL; 
	params.means     = NULL; // Not providing it as START_AUTO_STEP is used ; hope it will do the job 
	params.weights   = NULL;
	params.probs     = NULL;
	params.nclusters = NumWords2;
	params.cov_mat_type       = CvEM::COV_MAT_SPHERICAL;
	params.start_step         = CvEM::START_AUTO_STEP; // K-Means algorithm is used to determine initial parameters
	params.term_crit.max_iter = 10;
	params.term_crit.epsilon  = KMeansThreshold2;
	//params.term_crit.type     = CV_TERMCRIT_ITER|CV_TERMCRIT_EPS;
	params.term_crit.type     = CV_TERMCRIT_EPS;


	// cluster the data
	em_model.train( &points2, 0, params, &clusters2 );
	TIME(t12);
	CERR << "EM took " << difftime(t11,t12) << " ms" << endl;
#endif

	b::mapped_matrix<int32_t> words(NumFiles,NumWords2);
	std::vector<int32_t> ptword2;
	ptword2.clear();
	CERR << "Number of patches: " << vpl2size << endl;
	for ( int64_t i = 0 ; i < vpl2size ; i++ )
	{
		int32_t clusterID =  cvGetReal2D(&clusters2,i,0);
		ptword2.push_back(clusterID);
		cout << catnum[category[vpl2[i][0]]] << " " << category[vpl2[i][0]] 
			<< " " << vpl2[i] << " " << clusterID << endl << flush;
		//int64_t offset = (vpl2[i][0])*NumWords2 + clusterID;
		//words[offset] += 1;
		words((vpl2[i][0]),clusterID) += 1;

	} // end of i loop

	int ct = 0;
	vsivec_t words2;
	vsivec_t wordcount2;
	b::mapped_matrix<int32_t>::iterator1 itr1, start1 = words.begin1(), end1 = words.end1();
	b::mapped_matrix<int32_t>::iterator2 itr2;
	for ( itr1 = start1 ; itr1 != end1 ; ++itr1 ) // iterate over rows
	{
		sivec_t wd;
		sivec_t cnt;
		for ( itr2 = itr1.begin() ; itr2 != itr1.end() ; ++itr2 ) // iterate over columns
		{ 
			int32_t count = (*itr2);
			int64_t idx2 = itr2.index2();
			wd.push_back(idx2);
			cnt.push_back(count);
			cout << itr2.index1() << ":" << itr2.index2() << "::" <<  *itr2 << " "; 
			ct += count;
		}
		words2.push_back(wd);
		wordcount2.push_back(cnt);
		cout << endl;
	}

	CERR << "Total words counted from the matrix:" << ct << endl;
	CERR << "Pushed " << words2.size() << " vectors of indices" << endl;



	// Cluster centroids or the words 
	Matrix centers2(NumWords2,ptsize);
	for ( int64_t i = 0 ; i < NumWords2 ; i++ )
	{
		for ( int64_t j = 0 ; j < ptsize ; j++)
		{
			//cout << cvGetReal2D(&points,i,j) << " ";
			centers2(i,j) = cvGetReal2D(&points2,i,j);
		} // end of j loop
		//cout << endl;
	} // end of i loop
	//cout << centers2 << endl;

	delete [] pts2;
	delete [] clstr2;
	ifs2.close();
	ifs.close();

	Matrix dotpr = 2.0 * centers2 * centers2.transpose();
	Matrix D(NumWords2,NumWords2);
	Matrix C = centers2.sumsq(1);
	double max = 0.0; // distance is always non-negative
	for ( int64_t i = 0 ; i < NumWords2 ; i++ )
	{
		for ( int64_t j = 0 ; j < NumWords2 ; j++ )
		{
			D(i,j) = C(i) + C(j) - dotpr(i,j);
			if ( D(i,j) > max )
			{
				max = D(i,j);
			}
		}
	}

	D = D / max;

#ifdef _DEBUG
	// cout << " Distance between words :" << endl
	//	<<  D   << endl;
#endif



	TIME(t9);
	CERR << "Testing EMD: " << calcEMDKernelEntry(D,words2,wordcount2,9133,9136,NumWords2,NumFiles) << endl;
	TIME(t10);
	CERR << "It took " << difftime(t9,t10) << " ms for calculating EMD" << endl << flush;

	///////////////////////////////////////////////////////////////////////////////
	CERR << "Starting the creation of libsvm inputs "<< endl;
	// Create training and test sets
	TrainSetImgId.clear();
	TestSetImgId.clear();
	for ( int64_t i = 0 ; i < NumFiles ; i++ )
	{
		catcount[category[i]] += 1;
		if ( catcount[category[i]] <= NumTrainingImages )
		{
			TrainSetImgId.push_back(i);
		}
		else
		{
			TestSetImgId.push_back(i);
		}
	}

	int64_t trsz = TrainSetImgId.size();
	int64_t tesz = TestSetImgId.size();
	CERR << trsz << " images selected for training "  << endl;
	CERR << tesz << " images selected for testing " << endl;

	// Create inputs for libsvm
	std::string svmtrainfile = pe->getParameter("SVMTRAINFILE");
	std::string svmtestfile = pe->getParameter("SVMTESTFILE");
	std::string testmapfile = pe->getParameter("TESTMAPFILE");

	// Create the libsvm map file
	std::ofstream fo3(testmapfile.c_str());
	if(!fo3)
	{
		CERR << "Could not open " << svmtestfile.c_str() << endl;
		exit(FILE_OPEN_FAILED);
	}

	fo3 << "TRAINING_SET" << endl;
	for ( int64_t i = 0 ; i < trsz ; i++ )
	{
		fo3 << i << " " << TrainSetImgId[i] 
			<< " " << category[TrainSetImgId[i]] 
			<< " " << catnum[category[TrainSetImgId[i]]] 
			<< endl;
	}

	fo3 << "TEST_SET" << endl;
	for ( int64_t i = 0 ; i < tesz ; i++ )
	{
		fo3 << i << " " << TestSetImgId[i] 
			<< " " << category[TestSetImgId[i]] 
			<< " " << catnum[category[TestSetImgId[i]]] 
			<< endl;
	}

	fo3.close();

	CERR << "Map file created "<< endl;
	mat_t TRM(trsz,trsz); // should be symmetric
	TRM = b::identity_matrix<double>(trsz,trsz);
	CERR << "Starting training file creation "<< endl;
	// Create the libsvm training set file
	std::ofstream fo1(svmtrainfile.c_str());
	if(!fo1)
	{
		CERR << "Could not open " << svmtrainfile.c_str() << endl;
		exit(FILE_OPEN_FAILED);
	}

	for ( int64_t i = 0 ; i < trsz ; i++ )
	{
		for( int64_t j = i+1 ; j < trsz ; j++ )
		{
			//TIME(t3);
			TRM(i,j) = TRM(j,i) = calcEMDKernelEntry(D,words2,wordcount2,TrainSetImgId[i],TrainSetImgId[j],NumWords2,NumFiles);
			//TIME(t4);
			//CERR << i << " " << j << " :: " << difftime(t3,t4) << endl;
		}

		printRow(TRM,i,TrainSetImgId,trsz,fo1);
	}

	fo1.close();

	CERR << "Maximum EMD value (after training set) : " << maxemd << endl;

	CERR << "Starting test file creation " << endl;
	// Create the libsvm testing set file
	std::ofstream fo2(svmtestfile.c_str());
	if(!fo2)
	{
		CERR << " Could not open " << svmtestfile.c_str() << endl;
		exit(FILE_OPEN_FAILED);
	}

	for ( int64_t i = 0 ; i < tesz ; i++ )
	{
		vec_t TEM(trsz);
		for ( int64_t j = 0 ; j < trsz ; j++ )
		{
			TEM[j] = calcEMDKernelEntry(D,words2,wordcount2,TestSetImgId[i],TrainSetImgId[j],NumWords2,NumFiles);
		}
		printRow2(TEM,i,TestSetImgId,trsz,fo2);
	}

	fo2.close();

	CERR << "Maximum EMD value (after test set): " << maxemd << endl;

}


////////////////////////////////////////////////////////////////////////////////
/** Printing a row of output */ 
////////////////////////////////////////////////////////////////////////////////

void GaborFilterModel::printRow(const mat_t &S, const int64_t i,
		const vector<int64_t> &ImgId, const int64_t trsz, 
		std::ofstream &fo1)
{
	fo1 << catnum[category[ImgId[i]]] << " 0:" << i+1 << " ";
	for( int64_t j = 0 ; j < trsz ; j++ )
	{
		fo1 << j+1 << ":" << S(i,j) << " " ;
	}
	fo1 << endl << flush;
}


////////////////////////////////////////////////////////////////////////////////
/** Printing a row of output */
////////////////////////////////////////////////////////////////////////////////

void GaborFilterModel::printRow2(const vec_t &S, const int64_t i,
		const vector<int64_t> &ImgId, const int64_t trsz, 
		std::ofstream &fo2)
{
	fo2 << catnum[category[ImgId[i]]] << " 0:" << i+1 << " ";
	for( int64_t j = 0 ; j < trsz ; j++ )
	{
		fo2 << j+1 << ":" << S[j] << " " ;
	}
	fo2 << endl << flush;
}


////////////////////////////////////////////////////////////////////////////////
/** Calcuation of EMD */
////////////////////////////////////////////////////////////////////////////////

double GaborFilterModel::calcEMDKernelEntry(const Matrix &D, const vsivec_t &wd,
		const vsivec_t &wdcnt, const int64_t i, const int64_t j,
		const int64_t d1, const int64_t d2)
{
	assert ( (i < d2) && (j < d2));
#ifdef _DEBUG
	//	CERR << "i= " << i << " " << "j= " << j << endl;
#endif
	if ( i == j )
	{
		return 1;
	}
	int64_t s1 = wd[i].size();
	int64_t s2 = wd[j].size();
	if ( (s1 == 0) || (s2 == 0) )
	{
		return 0;
	}
	int64_t maxlen = s1 + s2;
	float *w1 =  new float[maxlen];
	float *w2 = new float[maxlen];
	int32_t *w3 = new int32_t[maxlen];
	memset(w1,0,maxlen*sizeof(float));
	memset(w2,0,maxlen*sizeof(float));
	memset(w3,0,maxlen*sizeof(int32_t));

	int32_t c1 = 0,c2 = 0,c3 = 0;
	int32_t flag = 0;

	while(true)
	{
		if ( wd[i][c1] == wd[j][c2] )
		{
			w3[c3] = wd[i][c1]; // or the other one.. they are equal
			w1[c3] = wdcnt[i][c1];
			w2[c3] = wdcnt[j][c2];
			c1++;
			c2++;
		}
		else if ( wd[i][c1] < wd[j][c2] )
		{
			w3[c3] = wd[i][c1];
			w1[c3] = wdcnt[i][c1];
			c1++;
		}
		else 
		{
			w3[c3] = wd[j][c2];
			w2[c3] = wdcnt[j][c2];
			c2++;
		}
		c3++;

		if ( c1 >= s1 )
		{
			flag |= 0x01;
		}
		if ( c2 >= s2 )
		{
			flag |= 0x02;
		}

		if ( flag > 0 )
		{
			break;
		}

	}

#ifdef _DEBUG
	//	CERR << "Flag for :" << i << "," << j << " :: " <<  hex << flag << dec << endl;
#endif
	if ( flag == 1 ) // only w1 is exhausted
	{
		// Copy w2
		while ( c2 < s2 )
		{
			w3[c3] =  wd[j][c2];
			w2[c3] = wdcnt[j][c2];
			c3++;
			c2++;
		}
	}
	else if ( flag == 2 ) // only w2 is exhausted
	{
		// Copy w1
		while ( c1 < s1 )
		{
			w3[c3] =  wd[i][c1];
			w1[c3] = wdcnt[i][c1];
			c3++;
			c1++;
		}
	}
	else if ( flag == 3 ) // both are exhausted
	{
		// Enjoy !!
	}
	else
	{
		// We are in serious trouble !! 
		CERR << "Else part reached !! Check the logic" << endl;
	}

	len = c3; 

#ifdef _DEBUG
	//	for ( int32_t i = 0 ; i < len ; i++ )
	//	{
	//		cerr << w1[i] << " ";
	//	}
	//	cerr << endl;
	//
	//	for ( int32_t i = 0 ; i < len ; i++ )
	//	{
	//		cerr << w2[i] << " ";
	//	}
	//	cerr << endl;
	//
	//	for ( int32_t i = 0 ; i < len ; i++ )
	//	{
	//		cerr << w3[i] << " ";
	//	}
	//	cerr << endl << endl << flush;
#endif

	int64_t lensq = len*len;
	cm = new float[lensq];
	memset(cm,0,lensq*sizeof(float));
	for ( int32_t ii = 0 ; ii < len ; ii++ )
	{
		for ( int32_t jj = 0 ; jj < ii ; jj++ )
		{
			int32_t offset = (ii)*len+jj;
			int32_t offsettr = (jj)*len+ii;
			cm[offset] = cm[offsettr] = D(w3[ii],w3[jj]);


		}
	}


#ifdef _DEBUG
	//	for ( int32_t ii = 0 ; ii < len ; ii++ )
	//	{
	//		for ( int32_t jj = 0 ; jj < len ; jj++ )
	//		{
	//			int32_t offset = (ii)*len+jj;
	//			cerr << cm[offset] << " "; 	
	//		}
	//		cerr << endl;
	//	}
#endif

	// Now we are ready for the EMD computation
	// OpenCV calcEMD2 function is giving a segmentation fault
	// Hope that Rubner's code works !
	//CvMat W1,W2,CM;
	//cvInitMatHeader(&W1,len,1,CV_32FC1,w1);
	//cvInitMatHeader(&W2,len,1,CV_32FC1,w2);
	//cvInitMatHeader(&CM,len,len,CV_32FC1,cm);

	// Commented to resolve clashes of feature_t typedefs
	
	float e = 0;
	/*
	feature_t f1[len], f2[len] ;
	for ( int32_t  ii = 0 ; ii < len ; ii++ )
	{
		f1[ii] = f2[ii] = ii;
	}
	signature_t s11 = { len, f1, w1}, s22 = { len, f2, w2};
	int32_t flowSize;
	float e = emd(&s11, &s22, dist, 0, &flowSize);
	if ( e  > maxemd )
	{
		maxemd = e;
	}
	*/

	//double emdval =  cvCalcEMD2(&W1,&W2,CV_DIST_USER,0,&CM,0, 0, 0);
	//cout << "TWO EMD values for " << i << "," << j << " : " << e << " " << emdval << endl << flush; 
	//cout << "EMD" << i << " , " << j << " :: " << e << endl;
	//return (exp(-1.0 * Alpha * e));


	delete [] cm;
	delete [] w1;
	delete [] w2;
	delete [] w3;

	return (1.0 - e);
}


#if 0
double GaborFilterModel::calcEMDKernelEntry(const CvMat &D, const float * const points,
		const int64_t i, const int64_t j, const int64_t d1, const int64_t d2)  // i and j are rows int points array 
// be sure to map the training/set IDs to the real ones 
{
	assert ( (i < d2) && (j < d2));
	const float *w1 = points + i * NumWords2; 
	const float *w2 = points + j * NumWords2; 

	CvMat W1,W2;
	cvInitMatHeader(&W1,NumWords2,1,CV_32FC1,(float*)w1);
	cvInitMatHeader(&W2,NumWords2,1,CV_32FC1,(float*)w2);

	return (cvCalcEMD2(&W1,&W2,CV_DIST_USER,0,&D,0, 0, 0));

}

#endif

///////////////////////////////////////////////////////////////////////////////
/** Run Hausdorff kernel experiment */
///////////////////////////////////////////////////////////////////////////////

void GaborFilterModel::runLSA()
{
	// Code replicated from formCodeBook()
	// Open S2 features file
	s2file =  pe->getParameter("S2FILE");
	ifstream ifs(s2file.c_str());
	if ( !ifs )
	{
		CERR << "Could not open file :" <<  s2file.c_str() << std::endl;
		ifs.close();
		exit(FILE_OPEN_FAILED);
	}

	// Open Patch position file
	patchposfile = pe->getParameter("PATCHPOSFILE");
	ifstream ifs2(patchposfile.c_str());
	if ( !ifs2 )
	{
		CERR << "Could not open file :" <<  patchposfile.c_str() << std::endl;
		ifs2.close();
		exit(FILE_OPEN_FAILED);
	}


	// Read all the s2 patches
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
		vmmat.push_back(mmat);

	}

	CERR << vmmat.size() << " Records read" << endl;

	// Crude line count implementation
	// Required of memory allocation of CvMat
	// for the cvKMeans2 function
	int64_t cnt = -1;
	for (std::string s = ""; !ifs2.eof() ; getline(ifs2,s),cnt++);
	ifs2.close();
	ifs2.open(patchposfile.c_str());
	CERR << "File has " << cnt << " patches"<< endl;
	this->FVDim = cnt;

	int64_t ptsize = NumOrientations * CodebookPatchSize * CodebookPatchSize;
	int64_t datasize = ptsize * (this->FVDim);
	float *pts = new float[datasize];
	memset(pts,0,datasize*sizeof(float));

	vivec_t vpl;
	vpl.clear();

	// Read patch position from patchposition file
	// Fetch the patch from s2 features space
	// Prepare data for K-means
	for ( int64_t i = 0 ; i < this->FVDim ; i++ )
	{
		ivec_t	pl(5);
		ifs2 >> pl ;
		vpl.push_back(pl);

#ifdef _DEBUG
		CERR << pl << endl;
#endif

		for ( int32_t j = 0 ; j < NumOrientations ; j++ )
		{
			mat_t opatch = b::project(vmmat[pl[0]](j,pl[1]),b::range(pl[2],pl[2]+pl[4]),b::range(pl[3],pl[3]+pl[4]));
			for ( int32_t k = 0 ; k < CodebookPatchSize ; k++ )
			{
				for ( int32_t l = 0 ; l < CodebookPatchSize ; l++ )
				{
					// int64_t offset =  (((i)*NumOrientations+j)*CodebookPatchSize+k)*CodebookPatchSize+l; 
					int64_t offset = l + CodebookPatchSize * 
						( k + CodebookPatchSize * ( j + NumOrientations * ( i ) )   );
					pts[offset] = float(opatch(k,l));
				} // end of l loop
			} // end of k loop
		}

	} // end of i loop

	CvMat points, clusters;
	int32_t *clstr = new int32_t[(this->FVDim) * sizeof(int32_t)];
	memset(clstr,0,(this->FVDim) * sizeof(int32_t));
	cvInitMatHeader(&clusters,this->FVDim,1,CV_32SC1,clstr);
	cvInitMatHeader(&points,this->FVDim,ptsize,CV_32FC1,pts);
	// The definition of CvKMeans2 has changed in the newer version
#if CV_MAJOR_VERSION == 0 || ( CV_MAJOR_VERSION == 1  && CV_MINOR_VERSION == 0  )
	cvKMeans2(&points,NumWords,&clusters,cvTermCriteria(CV_TERMCRIT_EPS,0,1.0));
#else
	cvKMeans2(&points,NumWords,&clusters,cvTermCriteria(CV_TERMCRIT_EPS,0,1.0),
			5,0,0,0,0);
#endif

	b::vector<int64_t> hst(NumWords);
	for ( int32_t i = 0 ; i < NumWords ; i++ )
	{
		hst[i] = 0;
	}

	// This is the category-patch-cluster(word) mapping
	std::vector<int32_t> ptword;
	ptword.clear();
	for ( int64_t i = 0 ; i < this->FVDim ; i++ )
	{
		int32_t clusterID =  cvGetReal2D(&clusters,i,0);
		ptword.push_back(clusterID);
		// cout << catnum[category[vpl[i][0]]] << " " << category[vpl[i][0]] 
		//	 << " " << vpl[i] << " " << clusterID << endl;
		hst[clusterID] = hst[clusterID] + 1;
	} // end of i loop



	cout << "Histogram :" << hst << endl;

	// Top 4 clusters have to be eliminated
	std::vector<int32_t> ElemClusterIdx;
	std::vector<int64_t> RemovedWordCnt;
	ElemClusterIdx.clear();
	RemovedWordCnt.clear();

	for ( int32_t i = 0 ; i < NumClustersEliminated ; i++ )
	{
		int32_t elidx = 0;
		int64_t maxel = hst[0];
		for ( int32_t j = 0 ; j < NumWords ; j++ )
		{
			if ( hst[j] > maxel )
			{
				maxel = hst[j];
				elidx = j;
			}
		} // end of j loop 
		// std::swap(hst[elidx],hst[NumWords-i-1]); // incorrect
		ElemClusterIdx.push_back(elidx);
		RemovedWordCnt.push_back(hst[elidx]);
		hst[elidx] = -INF;
	} // end of i loop


	std::sort(ElemClusterIdx.begin(),ElemClusterIdx.end());
	CERR << "Elements to be removed (indices) in sorted order: " ;
	for ( int32_t i = 0 ; i < NumClustersEliminated ; i++ )
	{
		cerr << ElemClusterIdx[i] << " ";
	} // end of i loop 
	cerr << endl;

	CERR << "Frequencies removed in descending order: " ;
	for ( int32_t i = 0 ; i < NumClustersEliminated ; i++ )
	{
		cerr << RemovedWordCnt[i] << " ";
	} // end of i loop 
	cerr << endl;



	// Remove the patches that were classified to top-4 cluster centroids
	vivec_t vpl2;
	vpl2.clear();
	int64_t vplsize = vpl.size();
	std::vector<int32_t>::iterator eciitr;
	for ( int64_t i = 0 ; i < vplsize ; i++ ) // for every patch
	{
		eciitr = std::find(ElemClusterIdx.begin(),ElemClusterIdx.end(),ptword[i]);
		if ( eciitr == ElemClusterIdx.end())  // patch has not been assigned to top-4 clusters
		{
			vpl2.push_back(vpl[i]);
		}
	} // end of i loop
	vpl.clear(); // should not be used beyond this point
	delete [] pts;
	delete [] clstr;

	int64_t vpl2size = vpl2.size();

	CERR << vpl2size << " patches remaining after the pruning " <<  endl;

	// Now we apply K-Means again on the pruned dataset but with lower threshold value

	// Get the patches again ... indices have changed

	int64_t datasize2 = ptsize * vpl2size;
	float *pts2 = new float[datasize2];
	memset(pts2,0,datasize2*sizeof(float));

	for ( int64_t i = 0 ; i < vpl2size ; i++ )
	{
		ivec_t	pl(5);
		pl = vpl2[i];

#ifdef _DEBUG
		CERR << pl << endl;
#endif
		for ( int32_t j = 0 ; j < NumOrientations ; j++ )
		{
			mat_t opatch = b::project(vmmat[pl[0]](j,pl[1]),b::range(pl[2],pl[2]+pl[4]),b::range(pl[3],pl[3]+pl[4]));
			for ( int32_t k = 0 ; k < CodebookPatchSize ; k++ )
			{
				for ( int32_t l = 0 ; l < CodebookPatchSize ; l++ )
				{
					// int64_t offset =  (((i)*NumOrientations+j)*CodebookPatchSize+k)*CodebookPatchSize+l; 
					int64_t offset = l + CodebookPatchSize * 
						( k + CodebookPatchSize * ( j + NumOrientations * ( i ) )   );
					pts2[offset] = float(opatch(k,l));
				} // end of l loop
			} // end of k loop
		}
	} // end of i loop

	// And now the K-means .... (second time)
	CvMat points2, clusters2;
	int32_t *clstr2 = new int32_t[vpl2size * sizeof(int32_t)];
	memset(clstr2,0,vpl2size * sizeof(int32_t));
	cvInitMatHeader(&clusters2,vpl2size,1,CV_32SC1,clstr2);
	cvInitMatHeader(&points2,vpl2size,ptsize,CV_32FC1,pts2);

	TIME(t7);
	// The definition of KMeans2 has changed in the newer version of OpenCV
#if CV_MAJOR_VERSION == 0 || ( CV_MAJOR_VERSION == 1  && CV_MINOR_VERSION == 0  )
	cvKMeans2(&points,NumWords,&clusters,cvTermCriteria(CV_TERMCRIT_EPS,0,1.0));
#else
	cvKMeans2(&points2,NumWords2,&clusters2,cvTermCriteria(CV_TERMCRIT_EPS,0,0.1),
			5,0,0,0,0);
#endif
	TIME(t8);
	CERR << "KMeans2 took " << difftime(t7,t8) << " ms" << endl;

	// This is the new (and hopefully improved) category-patch-cluster(word) mapping
	Matrix tf(NumWords2,NumFiles);
	//Initialize to zero matrix
	for ( int64_t i = 0 ; i <  NumWords2 ; i++ )
	{
		for ( int64_t j = 0 ; j < NumFiles ; j++ )
		{
			tf(i,j) = 0;
		}
	}


	b::mapped_matrix<int32_t> words(NumFiles,NumWords2);
	//SparseMatrix words(NumFiles,NumWords2);

	std::vector<int32_t> ptword2;
	ptword2.clear();
	CERR << "Number of patches: " << vpl2size << endl;
	for ( int64_t i = 0 ; i < vpl2size ; i++ )
	{
		int32_t clusterID =  cvGetReal2D(&clusters2,i,0);
		ptword2.push_back(clusterID);
		cout << catnum[category[vpl2[i][0]]] << " " << category[vpl2[i][0]] 
			<< " " << vpl2[i] << " " << clusterID << endl;
		words((vpl2[i][0]),clusterID) += 1;
		//tf(clusterID,vpl2[i](0)) = tf(clusterID,vpl2[i](0)) + 1;


	} // end of i loop

	cout << endl << endl << "WORDS:: " << endl;
	cout << words << endl << endl;

	int ct = 0;
	vsivec_t words2;
	b::mapped_matrix<int32_t>::iterator1 itr1, start1 = words.begin1(), end1 = words.end1();
	b::mapped_matrix<int32_t>::iterator2 itr2;
	for ( itr1 = start1 ; itr1 != end1 ; ++itr1 ) // iterate over rows
	{
		sivec_t wd;
		for ( itr2 = itr1.begin() ; itr2 != itr1.end() ; ++itr2 ) // iterate over columns
		{ 
			int32_t count = (*itr2);
			//int64_t idx1 = itr2.index1();
			int64_t idx2 = itr2.index2();

			for ( int32_t i = 0 ; i < count ; ++i )
			{
				wd.push_back(idx2);
			}

			cout << itr2.index1() << ":" << itr2.index2() << "::" <<  *itr2 << " "; 
			ct += (*itr2);

		}
		words2.push_back(wd);
		cout << endl;
	}

	CERR << "Total words counted from the matrix:" << ct << endl;
	CERR << "Pushed " << words2.size() << " vectors of indices" << endl;

#if 0
	cout << "Frequency Matrix :" << endl;
	cout << tf << endl;
	SVD svdtf = SVD(tf);
	cout << endl << "SVD ::" << endl;
	cout << svdtf << endl;
#endif


	// Cluster centroids or the words 
	Matrix centers2(NumWords2,ptsize);
	for ( int64_t i = 0 ; i < NumWords2 ; i++ )
	{
		for ( int64_t j = 0 ; j < ptsize ; j++)
		{
			//cout << cvGetReal2D(&points,i,j) << " ";
			centers2(i,j) = cvGetReal2D(&points2,i,j);
		} // end of j loop
		//cout << endl;
	} // end of i loop
	//cout << centers2 << endl;




#if 0
	//	ofstream ofs(wordsfile.c_str()); 
	//	for ( int64_t i = 0 ; i < NumWords2 ; i++ )
	//	{
	//		bvmat_t wctr(NumOrientations);
	//		for ( int32_t  j = 0 ; j < NumOrientations ; j++ )
	//		{
	//			mat_t mt(CodebookPatchSize,CodebookPatchSize);
	//			for ( int32_t k = 0 ; k < CodebookPatchSize ; k++ )
	//			{
	//				for ( int32_t l = 0 ; l < CodebookPatchSize ; l++ )
	//				{
	//					int32_t offset =  ((j)*CodebookPatchSize+k)*CodebookPatchSize+l;
	//					mt(k,l) = cvGetReal2D(&points2,i,offset); 
	//				} // end of l loop
	//			} // end of k loop
	//			wctr[j] = mt;
	//		} // end of j loop
	//		//vpatch.push_back(wctr);
	//		ofs << wctr << endl;
	//
	//	} // end of i loop
	//	ofs.close();
	//
	//	cout << endl << "WORDS:" << endl;
	//	Matrix centers2(NumWords2,ptsize);
	//	for ( int64_t i = 0 ; i < NumWords2 ; i++ )
	//	{
	//		for ( int64_t j = 0 ; j < ptsize ; j++)
	//		{
	//			//cout << cvGetReal2D(&points,i,j) << " ";
	//			centers2(i,j) = cvGetReal2D(&points2,i,j);
	//		} // end of j loop
	//		cout << centers2.row(i);
	//		cout << endl;
	//		//cout << vpatch[i] << endl << endl << endl;
	//	} // end of i loop
	//	//cout << centers2 << endl;
	//
#endif

	delete [] pts2;
	delete [] clstr2;
	ifs2.close();
	ifs.close();

	Matrix dotpr = 2.0 * centers2 * centers2.transpose();
	Matrix D(NumWords2,NumWords2);
	Matrix C = centers2.sumsq(1);
	double max = 0.0; // distance is always non-negative
	for ( int64_t i = 0 ; i < NumWords2 ; i++ )
	{
		for ( int64_t j = 0 ; j < NumWords2 ; j++ )
		{
			D(i,j) = C(i) + C(j) - dotpr(i,j);
			if ( D(i,j) > max )
			{
				max = D(i,j);
			}
		}
	}

	D = D / max;
#if 0
	D = 10.0*D;
	for ( int64_t i = 0 ; i < NumWords2 ; i++ )
	{
		for ( int64_t j = 0 ; j < NumWords2 ; j++ )
		{
			D(i,j) = exp(D(i,j));
		}
	}

#endif
	cout << " Distance between words :" << endl
		<<  D   << endl;

#if 0
	// Pre-compute combinations 
	CERR << "Precomputing combinations" << endl;
	mvsivec_t Combination(MaxPatchesPerImage+1,MaxPatchesPerImage+1);
	for ( int64_t i = 1 ; i <= MaxPatchesPerImage ; i++ ) // for every n
	{
		for ( int64_t j = 1 ; j <= i ; j++ ) // for every r
		{
			genPartialMatchPoss(i,j,Combination(i,j));
		}
	}
#endif

	CERR << "Starting the creation of libsvm inputs "<< endl;
	// Create training and test sets
	TrainSetImgId.clear();
	TestSetImgId.clear();
	for ( int64_t i = 0 ; i < NumFiles ; i++ )
	{
		catcount[category[i]] += 1;
		if ( catcount[category[i]] <= NumTrainingImages )
		{
			TrainSetImgId.push_back(i);
		}
		else
		{
			TestSetImgId.push_back(i);
		}
	}

	int64_t trsz = TrainSetImgId.size();
	int64_t tesz = TestSetImgId.size();
	CERR << trsz << " images selected for training "  << endl;
	CERR << tesz << " images selected for testing " << endl;

	// Create inputs for libsvm
	std::string svmtrainfile = pe->getParameter("SVMTRAINFILE");
	std::string svmtestfile = pe->getParameter("SVMTESTFILE");
	std::string testmapfile = pe->getParameter("TESTMAPFILE");

	// Create the libsvm map file
	std::ofstream fo3(testmapfile.c_str());
	if(!fo3)
	{
		CERR << "Could not open " << svmtestfile.c_str() << endl;
		exit(FILE_OPEN_FAILED);
	}

	fo3 << "TRAINING_SET" << endl;
	for ( int64_t i = 0 ; i < trsz ; i++ )
	{
		fo3 << i << " " << TrainSetImgId[i] 
			<< " " << category[TrainSetImgId[i]] 
			<< " " << catnum[category[TrainSetImgId[i]]] 
			<< endl;
	}

	fo3 << "TEST_SET" << endl;
	for ( int64_t i = 0 ; i < tesz ; i++ )
	{
		fo3 << i << " " << TestSetImgId[i] 
			<< " " << category[TestSetImgId[i]] 
			<< " " << catnum[category[TestSetImgId[i]]] 
			<< endl;
	}

	fo3.close();

	CERR << "Map file created "<< endl;
	CERR << "Starting training file creation "<< endl;
	// Create the libsvm training set file
	std::ofstream fo1(svmtrainfile.c_str());
	if(!fo1)
	{
		CERR << "Could not open " << svmtrainfile.c_str() << endl;
		exit(FILE_OPEN_FAILED);
	}

	for ( int64_t i = 0 ; i < trsz ; i++ )
	{
		fo1 << catnum[category[TrainSetImgId[i]]] << " 0:" << i+1 << " ";
		for( int64_t j = 0 ; j < trsz ; j++ )
		{

			//TIME(t3);
			fo1 << j+1 << ":" << getKernelEntry(D,words2,i,j) << " " << flush;
			//TIME(t4);
			//CERR << i << " " << j << " :: " << difftime(t3,t4) << endl;
		}

		fo1 << endl;
	}

	fo1.close();

	CERR << "Starting test file creation "<< endl;
	// Create the libsvm testing set file
	std::ofstream fo2(svmtestfile.c_str());
	if(!fo2)
	{
		CERR << " Could not open " << svmtestfile.c_str() << endl;
		exit(FILE_OPEN_FAILED);
	}

	for ( int64_t i = 0 ; i < tesz ; i++ )
	{
		fo2 << catnum[category[TestSetImgId[i]]] << " 0:" << i+1 << " ";
		for ( int64_t j = 0 ; j < tesz ; j++ )
		{
			fo2 << j+1 << ":" << getKernelEntry(D,words2,i,j) << " ";
		}
		fo2 << endl;
	}

	fo2.close();
}


///////////////////////////////////////////////////////////////////////////////
/** calculate Hausdorff kind of distance measures */
///////////////////////////////////////////////////////////////////////////////

double GaborFilterModel::getKernelEntry(const Matrix& D, const vsivec_t &W, const int64_t x1, const int64_t x2)
{
	if ( x1 == x2 )
	{
		return 1;
	}

	int64_t s1 = W[x1].size();
	int64_t s2 = W[x2].size();
	if ( s1 == 0 || s2 == 0 )
	{
		return 0;
	}
	int64_t mn = std::min(s1,s2);
	int64_t small = -1,large = -1,mx = -1;
	if ( s1 == mn )
	{
		small = x1;
		large = x2;
		mx = s2;
	}
	else
	{
		small = x2;
		large = x1;
		mx = s1;
	}

	double avgdist = 0.0;
	//double maxdist = 0.0; // note: distance cannot be negative

	//double mindist = D(W[small][0],W[large][0]);
	for ( int32_t i = 0 ; i < mn ; i++ )
	{
		double mindist = D(W[small][i],W[large][0]);
		for ( int32_t j = 0 ; j < mx ; j++ )
		{
			if (  D(W[small][i],W[large][j]) < mindist )
			{
				mindist =  D(W[small][i],W[large][j]);
			}

			//avgdist += D(W[small][i],W[large][j]);
		}

		avgdist += mindist;
		//		if ( mindist > maxdist )
		//		{
		//			maxdist = mindist;
		//		}
	}

	avgdist /= mn;

	//return (exp(-1.0*avgdist)); 
	//return (exp(-1.0*maxdist)); 
	return (exp(-1.0*avgdist)); 
	//return (1 - avgdist); 

}


////////////////////////////////////////////////////////////////////////////////
/** Calculate kernel entry */
////////////////////////////////////////////////////////////////////////////////

double GaborFilterModel::getKernelEntry(const Matrix& D, const vsivec_t &W, const mvsivec_t& Combination, const int64_t x1, const int64_t x2)
{
	int64_t s1 = W[x1].size();
	int64_t s2 = W[x2].size();
	if ( s1 == 0 || s2 == 0 )
	{
		return 0;
	}
	int64_t mn = std::min(s1,s2);
	int64_t small = -1,large = -1,mx = -1;
	if ( s1 == mn )
	{
		small = x1;
		large = x2;
		mx = s2;
	}
	else
	{
		small = x2;
		large = x1;
		mx = s1;
	}

	vec_t apd(s1*s2);
	getAllPairsDist(D,W[small],W[large],apd);

	assert ( mx <= MaxPatchesPerImage );
	//genPartialMatchPoss(mx,mn,ind);
	const vsivec_t &ind = Combination(mx,mn);
	int64_t indsz = ind.size();
	double mindist = INF;
	for ( int64_t i = 0 ; i < indsz ; i++ ) // for each combination
	{
		double dist = 0.0;
		for ( int32_t j = 0 ; j < mn ; j++ ) 
		{
			//CERR << i << " " << j << " " << mx << " " << mn << " " << ind[i][j] << endl << flush;
			// cout <<  apd[ind[i][j] * mn + j]   << " ";
			dist += apd[ind[i][j] * mn + j];
		}
		dist /= mn;
		if ( dist < mindist )
		{
			mindist = dist;
		}
	}


	return (1.0 - mindist);
}


///////////////////////////////////////////////////////////////////////////////
/** Generating distance of all pairs */
///////////////////////////////////////////////////////////////////////////////

void GaborFilterModel::getAllPairsDist(const Matrix& D, const sivec_t &v1, const sivec_t &v2 , vec_t &A)
{
	int64_t s1 = v1.size();
	int64_t s2 = v2.size();

	A.clear();
	int64_t offset = 0;
	for ( int64_t i = 0 ; i < s1 ; i++ )
	{
		for ( int64_t j = 0 ; j < s2 ; j++ )
		{
			offset = (i)*s2+j; 
			A[offset] = D(v1[i],v2[j]);

		}
	}

}


///////////////////////////////////////////////////////////////////////////////
/** Generate all possible partial match possibilities */
///////////////////////////////////////////////////////////////////////////////

void GaborFilterModel::genPartialMatchPoss(const int32_t n,const int32_t r, vsivec_t &ans)
{

	ans.clear();
	std::queue<sivec_t> q;
	for ( int32_t i = 0 ; i < n ; i++ )
	{
		// Push singleton elements in the queue
		sivec_t v;
		v.clear();
		v.push_back(i);
		q.push(v);
	}

	for ( int32_t i = 0 ; i < r ; i++ )
	{

		while(!q.empty())
		{
			// Pop from queue
			sivec_t fr = q.front();
			q.pop();

			int32_t frs = fr.size();
			if ( frs  == r )
			{
				// Output the ans
				//for ( int32_t k = 0 ; k < r ; k++ )
				//{
				//cout << fr[k] << " ";
				//}
				//cout << endl;
				ans.push_back(fr);
				continue; // Look out for next solution
			}

			int32_t top = fr[frs-1];
			for( int32_t j = top+1 ; j < n ; j++ )
			{
				sivec_t ne;
				ne.clear();
				ne = fr;
				ne.push_back(j);
				q.push(ne); // Enqueue the concatenated string
			}

		}

	}


}


///////////////////////////////////////////////////////////////////////////////
/** Generate patches filled with random numbers */
///////////////////////////////////////////////////////////////////////////////

void GaborFilterModel::fillS2PatchesWithRandomNumbers()
{
	int32_t dim = CodebookPatchSize*CodebookPatchSize*NumOrientations;
	Random<double> *r[dim];
	int32_t sz = CodebookPatchSize;
	int32_t szsq = CodebookPatchSize*CodebookPatchSize;

	for ( int32_t i = 0 ; i < dim ; i++ )
	{
		r[i] = new Random<double>();
	}

	for ( int32_t i = 0 ; i < NDIM ; i++ )
	{
		cout << "[" << sz << "]([" << sz << "," << sz << "](("; 
		for ( int32_t j = 0 ; j < dim ; j++ )
		{

			if ( (j > 0) && (j % szsq == 0) )
			{
				cout << ")),[" << sz <<"," << sz << "]((";
			}
			else if ( ( j > 0 ) && ( ( j% sz == 0 ) ) )
			{
				cout << "),(";
			}
			cout << r[j]->nextRandom() ;
			if ( ( j == 0 ) || ( (j > 0) && ( (j+1)% sz != 0 ) ) )
			{
				cout	<< ",";
			}
		}
		cout << ")))" << endl;
	}

	for ( int32_t i = 0 ; i < dim ; i++ )
	{
		delete  r[i];
	}

}


///////////////////////////////////////////////////////////////////////////////
/** calculating EMD directly on the C1 features */
///////////////////////////////////////////////////////////////////////////////

void GaborFilterModel::calcEMDDirectly()
{
	s2file =  pe->getParameter("S2FILE");
	ifstream ifs(s2file.c_str());
	if ( !ifs )
	{
		CERR << "Could not open file :" <<  s2file.c_str() << std::endl;
		ifs.close();
		exit(FILE_OPEN_FAILED);
	}

	// Open Patch position file
	patchposfile = pe->getParameter("PATCHPOSFILE");
	ifstream ifs2(patchposfile.c_str());
	if ( !ifs2 )
	{
		CERR << "Could not open file :" <<  patchposfile.c_str() << std::endl;
		ifs2.close();
		exit(FILE_OPEN_FAILED);
	}

	// Read all the c1 patches in the s2 files
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
		vmmat.push_back(mmat);

	}

	CERR << vmmat.size() << " Records read" << endl;

	// Crude line count implementation
	// Required of memory allocation of CvMat
	int64_t cnt = -1;
	for (std::string s = ""; !ifs2.eof() ; getline(ifs2,s),cnt++);
	ifs2.close();
	ifs2.open(patchposfile.c_str());
	CERR << "File has " << cnt << " patches"<< endl;
	this->FVDim = cnt;

	//float *pts = new float[datasize];
	//float points[NumFiles][NumPatchVecPerImage][PLength];
	float (*points)[NumPatchVecPerImage][PLength] = new float[NumFiles][NumPatchVecPerImage][PLength];

	vivec_t vpl;
	vpl.clear();

	// Read patch position from patchposition file
	// Fetch the patch from s2 features space
	for ( int64_t i = 0 ; i < this->FVDim ; i++ )
	{
		ivec_t	pl(5);
		ifs2 >> pl ;
		vpl.push_back(pl);

#ifdef _DEBUG
		CERR << pl << endl;
#endif
		int64_t fileid = pl[0];
		int64_t patchid = i%NumPatchVecPerImage;
		for ( int32_t j = 0 ; j < NumOrientations ; j++ )
		{
			mat_t opatch = b::project(vmmat[pl[0]](j,pl[1]),b::range(pl[2],pl[2]+pl[4]),b::range(pl[3],pl[3]+pl[4]));
			for ( int32_t k = 0 ; k < CodebookPatchSize ; k++ )
			{
				for ( int32_t l = 0 ; l < CodebookPatchSize ; l++ )
				{
					int64_t offset = ((j)*CodebookPatchSize+k)*CodebookPatchSize+l;
					points[fileid][patchid][offset] = float(opatch(k,l));
					//cout << fileid << " " << patchid << " " << offset << endl << flush;
				} // end of l loop
			} // end of k loop
		}

	} // end of i loop


	CERR << "Before deallocation " << endl << flush;
	vmmat.clear(); // We are short of memory ; S2 files you need to go out !
	vmmat.~vector();
	vpl.clear();
	vpl.~vector();

	CERR << "After deallocation " << endl << flush;
	TIME(t1);
	CERR << "Testing EMD " << getEMDVal(points,100,250) << endl << flush;
	TIME(t2);
	CERR<< "It took " << difftime(t1,t2) << " ms" << endl;
	CERR << "Testing EMD " << getEMDVal(points,100,100) << endl << flush;
	CERR << "Testing EMD " << getEMDVal(points,250,250) << endl << flush;

	///////////////////////////////////////////////////////////////////////////////
	CERR << "Starting the creation of libsvm inputs "<< endl;
	// Create training and test sets
	TrainSetImgId.clear();
	TestSetImgId.clear();
	for ( int64_t i = 0 ; i < NumFiles ; i++ )
	{
		catcount[category[i]] += 1;
		if ( catcount[category[i]] <= NumTrainingImages )
		{
			TrainSetImgId.push_back(i);
		}
		else
		{
			TestSetImgId.push_back(i);
		}
	}

	int64_t trsz = TrainSetImgId.size();
	int64_t tesz = TestSetImgId.size();
	CERR << trsz << " images selected for training "  << endl;
	CERR << tesz << " images selected for testing " << endl;

	// Create inputs for libsvm
	std::string svmtrainfile = pe->getParameter("SVMTRAINFILE");
	std::string svmtestfile = pe->getParameter("SVMTESTFILE");
	std::string testmapfile = pe->getParameter("TESTMAPFILE");

	// Create the libsvm map file
	std::ofstream fo3(testmapfile.c_str());
	if(!fo3)
	{
		CERR << "Could not open " << svmtestfile.c_str() << endl;
		exit(FILE_OPEN_FAILED);
	}

	fo3 << "TRAINING_SET" << endl;
	for ( int64_t i = 0 ; i < trsz ; i++ )
	{
		fo3 << i+1 << " " << TrainSetImgId[i]+1 
			<< " " << category[TrainSetImgId[i]] 
			<< " " << catnum[category[TrainSetImgId[i]]] 
			<< endl;
	}

	fo3 << "TEST_SET" << endl;
	for ( int64_t i = 0 ; i < tesz ; i++ )
	{
		fo3 << i+1 << " " << TestSetImgId[i]+1 
			<< " " << category[TestSetImgId[i]] 
			<< " " << catnum[category[TestSetImgId[i]]] 
			<< endl;
	}

	fo3.close();

	CERR << "Map file created "<< endl;
	mat_t TRM(trsz,trsz); // should be symmetric
	TRM = b::identity_matrix<double>(trsz,trsz);
	CERR << "Starting training file creation "<< endl;
	// Create the libsvm training set file
	std::ofstream fo1(svmtrainfile.c_str());
	if(!fo1)
	{
		CERR << "Could not open " << svmtrainfile.c_str() << endl;
		exit(FILE_OPEN_FAILED);
	}

	for ( int64_t i = 0 ; i < trsz ; i++ )
	{
		for( int64_t j = i+1 ; j < trsz ; j++ )
		{
			TRM(i,j) = TRM(j,i) = getEMDVal(points,TrainSetImgId[i],TrainSetImgId[j]);
		}

		printRow(TRM,i,TrainSetImgId,trsz,fo1);
	}

	fo1.close();

	CERR << "Maximum EMD value (after training set) : " << maxemd << endl;

	CERR << "Starting test file creation " << endl;
	// Create the libsvm testing set file
	std::ofstream fo2(svmtestfile.c_str());
	if(!fo2)
	{
		CERR << " Could not open " << svmtestfile.c_str() << endl;
		exit(FILE_OPEN_FAILED);
	}

	for ( int64_t i = 0 ; i < tesz ; i++ )
	{
		vec_t TEM(trsz);
		for ( int64_t j = 0 ; j < trsz ; j++ )
		{
			TEM[j] = getEMDVal(points,TrainSetImgId[i],TrainSetImgId[j]);
		}
		printRow2(TEM,i,TestSetImgId,trsz,fo2);
	}

	fo2.close();

	CERR << "Maximum EMD value (after test set): " << maxemd << endl;


	delete [] points;

}


///////////////////////////////////////////////////////////////////////////////
/** calculate EMD using Rubner's code */
///////////////////////////////////////////////////////////////////////////////

float GaborFilterModel::getEMDVal(float points[NFiles][NumPatchVecPerImage][PLength], int64_t i, int64_t j)
{
	if ( i == j )
	{
		return 1.0;
	}
	farray_t &f1 = points[i];	
	farray_t &f2 = points[j];
	signature_t s1 = { NumPatchVecPerImage, f1, wgt };
	signature_t s2 = { NumPatchVecPerImage, f2, wgt };
	float e = emd(&s1, &s2, dist2, 0, 0);
	return (exp(-1.0*e));
}


///////////////////////////////////////////////////////////////////////////////
/** Run libsvm svm-train command */
///////////////////////////////////////////////////////////////////////////////

void GaborFilterModel::svmtrain(const string & svmtrfilenm, const string &svmmodelfilenm)
{
	stringstream svmtrcmdstr;
	svmtrcmdstr << svmtrainexe <<  " " 
		<<  svmtrfilenm << " " << svmmodelfilenm;  // One class SVM 
	string svmtrcmd = svmtrcmdstr.str();
	CERR << svmtrcmd.c_str() << endl;
	system(svmtrcmd.c_str()); // Execute SVM train 
	perror(0); // Report error if any in the system call
}

void GaborFilterModel::sortS2(const string &s2fileun, const string &s2filesorted)
{
	assert(s2fileun != s2filesorted);
	stringstream sortcmdstr;
	sortcmdstr <<  "sort -n " 
		<<  s2fileun << " > " << s2filesorted;
	string sortcmd = sortcmdstr.str();
	CERR << sortcmd.c_str() << endl;
	system(sortcmd.c_str()); // Execute sort command
	perror(0); // Report error if any in the system call
}

///////////////////////////////////////////////////////////////////////////////
/** Basic implementation of a Serre Poggio Algorithm (Driver program) */
///////////////////////////////////////////////////////////////////////////////

void GaborFilterModel::basicSerrePoggio()
{
	// Comment and uncomment as per your needs
	//calcS2();
	//createRandomPatches(); 
	calcFVStage1();
	calcFVStage2();
}


void GaborFilterModel::SP()
{
	trainCarFilter();
	testCarFilter();
}


///////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////

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

	TIME(tstart);
	std::string conffl(argv[1]);
	GaborFilterModel gfm(conffl.c_str(),"version2");
	gfm.SP();
	//gfm.createAllPatches();
	//gfm.createAllPatchesTrain();
	//gfm.createOverlappingGridPatches();
	//gfm.createEdgePatches();
	//gfm.createCornerPatches();
	//gfm.calcS2();
	//gfm.createRandomPatches(); // Serre-Poggio Model
	//gfm.calcFVStage1();
	//gfm.calcFVStage2();
	//gfm.formCodeBook();
	//gfm.calcFVStage2Words();
	//gfm.runLSA();
	//gfm.fillS2PatchesWithRandomNumbers();
	//gfm.runKernelEMD();
	//gfm.calcEMDDirectly();
	//gfm.convC2ToSVM("/mnt/output/Caltech101AndCarsOutput/ccars.c2","/mnt/output/Caltech101AndCarsOutput/ccars.svmin",6480);
	TIME(tend);
	CERR << "The program ran for " << difftime(tstart,tend) << " ms" << endl; 

#ifdef _MPI
	MPI::Finalize();
#endif



	return 0;
}

