#ifndef _GABOR_FILTER_MODEL
#define _GABOR_FILTER_MODEL

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

#include <dir.h>
#include <ParameterExtractor.h>

#include <iostream>
#include <iomanip>
#include <cstdlib>
#include <ctime>
#include <cassert>
#include <vector>
#include <map>
#include <algorithm>
#include <string>

extern "C" {
#include <sys/time.h>
}

#include <boost/numeric/ublas/exception.hpp>
#include <boost/numeric/ublas/vector.hpp>
#include <boost/numeric/ublas/vector_proxy.hpp>
#include <boost/numeric/ublas/matrix.hpp>
#include <boost/numeric/ublas/matrix_proxy.hpp>
#include <boost/numeric/ublas/io.hpp>
#include <boost/numeric/ublas/detail/iterator.hpp>
#include <boost/numeric/ublas/detail/iterator.hpp>
#include <boost/numeric/ublas/vector_sparse.hpp>
#include <boost/numeric/ublas/matrix_sparse.hpp>
#include <boost/serialization/serialization.hpp>
#include <boost/tokenizer.hpp>
#include <boost/foreach.hpp>
#include <boost/filesystem/operations.hpp>
#include <boost/filesystem/fstream.hpp>
#include <boost/filesystem/path.hpp>
#include <boost/timer.hpp>
#include <boost/progress.hpp>
#include <octave/oct.h>

#include <omp.h>
#ifdef _MPI
#include <mpi.h>
#endif

#include <cv.h>
#include <highgui.h>
#include <ml.h>
#include "sdd_include.h"

////////////////////////////////////////////////////////////////////////////////

using namespace std;
namespace b = boost::numeric::ublas;
namespace fs = boost::filesystem;

////////////////////////////////////////////////////////////////////////////////

#define foreach BOOST_FOREACH
#define MODEL_EXT ".model"
#define FDELIM "="
#define CERR std::cerr << hostname() << " : " << __FILE__ << " : " << __LINE__ << " : " << __func__ <<  " :: "   
#define OSPATHSEP "/"
#define LIBSVMTYPESWITCH " -s "
#define LIBSVMKERNELSWITCH " -t "

////////////////////////////////////////////////////////////////////////////////

typedef enum ErrorCodes { INCORRECT_ARGS=1, IMG_LOAD_FAILED, FILE_OPEN_FAILED, INCORRECT_VALUE} err_t;
typedef enum LibSVM_SVMTypes { CSVC=0, NUSVC, ONECLASS, ESVR, NUSVR } libsvm_svm_t;
typedef enum LibSVM_KernelTypes { LINEAR=0, POLY, RBF, SIGMOID, PRECOMPUTED} libsvm_kernel_t; 
typedef b::matrix<double> mat_t;
typedef	boost::numeric::ublas::vector<double> filter_t; 
typedef boost::numeric::ublas::vector_range<filter_t> range_t;
typedef std::vector<filter_t> s2proto_t;
typedef std::vector<mat_t> vmat_t;
typedef b::matrix<mat_t> mmat_t;
typedef std::vector<mmat_t> vmmat_t;
typedef std::vector<vmmat_t> vvmmat_t;
typedef b::matrix_vector_range<mat_t> mvr_t;
typedef b::vector<mat_t> bvmat_t;
typedef std::vector<bvmat_t> vbvmat_t;
typedef std::vector<vbvmat_t> vvbvmat_t;
typedef b::vector<double> vec_t;
typedef b::vector<int32_t> ivec_t;
typedef std::vector<ivec_t> vivec_t;
typedef std::vector<int32_t> sivec_t;
typedef std::vector<sivec_t> vsivec_t;
typedef b::matrix<vsivec_t> mvsivec_t;

////////////////////////////////////////////////////////////////////////////////

const int32_t ImageSize = 128;
const double INF = 1e6;
const int32_t NumCat = 102;
const int32_t NumOrientations = 4;
const int32_t NumRFSizes = 16;
//const int32_t NumRFSizes = 4;
const int32_t FilterBankSize = NumOrientations * NumRFSizes;
const int32_t NumGrayValues = 1 << IPL_DEPTH_8U;
const int32_t C1Size = (NumOrientations * NumRFSizes) / 2;
const int32_t NumTrainingImages = 60;
const int32_t NumPatchSizes = 4;
const int32_t NumBands = NumRFSizes / 2;
const int32_t PatchSizes[NumPatchSizes] = {4,8,12,16};
const int32_t MinPatchSize = PatchSizes[0];
const int32_t NumPatchesPerImage = 16;
//const int32_t FVDim = NumPatchesPerImage * NumTrainingImages * NumCat;
const int32_t FVDim = 5388;
const int32_t MaxCorners = 10;
const double GFTTMinDistance = 5.0;
const double GFTTQualityLevel = 0.05; 
const double CannyUpperThreshold = 150;
const double CannyLowerThreshold = 50;
const double CodebookPatchSize = 4;
const int64_t NumWords = 50;
const int64_t NumWords2 = 50;
const int64_t NumClustersEliminated = 0;
const int64_t MaxPatchesPerImage = 21;
const int64_t NDIM = 5000;
const double Alpha = 1.0;
const double KMeansThreshold1 = 1.0;
const double KMeansThreshold2 = 0.001;
const int64_t NumFilesPerCat = 30;
const int64_t NumPatchVecPerImage = 21;
const int64_t PatchVectorLength = NumOrientations * CodebookPatchSize * CodebookPatchSize; 
const int64_t PLength = 64; 
const int64_t NFiles = 9145;
const int32_t NumFolds = 3;
const int32_t NumClassifiers = 24;
const int32_t NumPatchesPerClass = NumPatchesPerImage * NumTrainingImages;
const int32_t NClasses = 64;
const int32_t MaxOcc = 3;

////////////////////////////////////////////////////////////////////////////////

typedef struct GaborFilterParams
{
	int32_t s; /** RF (Receptive Field) Size */
	double theta; /** Orientation in degrees */
	double sigma; /** Effective Width */
	double gamma; /** Aspect Ratio */
	double lambda; /** Wavelength */
	int32_t gridSize; /** Grid Size over which MAX is to be taken */
	int32_t s2cellsize; /** Size of S2 cell ( = ImageSize / gridSize ) */
} gfp_t ;

////////////////////////////////////////////////////////////////////////////////
typedef struct PatchLoc
{
	int64_t imgID;
	int64_t bandNum;
	int64_t xpos;
	int64_t ypos;
	int64_t psize;
} patchloc_t;
typedef std::vector<patchloc_t> vpatchloc_t;
bool patchlocLessthan(const patchloc_t &o1,const patchloc_t &o2);
typedef vpatchloc_t::iterator vpatchlocitr_t;

static inline bool operator==(const patchloc_t &o1,const patchloc_t &o2)
{
	return ( \
			( o1.imgID == o2.imgID ) && \
			( o1.bandNum == o2.bandNum ) && \
			( o1.xpos == o2.xpos ) && \
			( o1.ypos == o2.ypos ) && \
			( o1.psize == o2.psize ) \
		   );

}

////////////////////////////////////////////////////////////////////////////////

class GaborFilterModel
{	protected:
	int64_t NumFiles;     /** Stores the number offiles */
	std::string version;  /** Version */
	std::string conffile; /** Configuration file */
	std::string keyfile;  /** Key file (defines the order in which files are read) */
	std::string imagedir; /** Top level directory containing images */
	std::string s2file;   /** Pathname of the S2 fetures */
	std::string patchposfile; /** Pathname of patch positions file */
	std::string inputdir; /** Input directory ; Version 2 */
	std::string outputdir; /** Output directory ; Version 2 */
	std::string c2file;  /** C2 features file */
	std::string svmtrfile; /** SVM training file */
	std::string svmtefile; /** SVM test file */
	std::string svmmofile; /** SVM model file */
	std::string svmoufile; /** SVM output file */
	std::string cannydir;          
	std::string wordsfile;
	ParameterExtractor *pe; /** Object for parsing parameter values from file */
	std::vector<std::string> category; /** imageID - category mapping */
	std::vector<fs::path> filename; /** imageID - filename mapping */
	std::vector<std::string> categorylist; /** list of categories */
	std::map<std::string,int64_t> catnum; /** category_name - category_id mapping */
	std::map<std::string,int64_t> catcount; /** number of images per category */
	sivec_t intcatid; 
	CvSize isz; /** Image size */
	gfp_t S1LayerParam[NumOrientations][NumRFSizes]; /** Parameters for the S1 layer */
	filter_t* GaborFilterBank[NumOrientations][NumRFSizes]; 
	vpatchloc_t randomPatches;
	vmmat_t vmmat; /** Contains all S2 feature */
	vbvmat_t vpatch; /** Contains all the patches */
	vvmmat_t v_vmmat; /** Contaiins all S2 features ( Multiple classifier version ) */
	vvbvmat_t v_vpatch; /** All the patches ( Multiple classifier version ) */
	int64_t FVDim; /** Dimesion of the C2 feature vector */
	std::vector<int64_t> TrainSetImgId;  
	std::vector<int64_t> TestSetImgId;
	double maxemd;
	std::string trdir;      /** Training images directory (version 2) */ 
	std::string tedir;      /** Test images directory (version 2) */
	std::string urldir;     /** URL mapping directory (version 2) */
	std::string svmtrdir;   /** libsvm training files directory (version 2) */
	std::string svmtedir;   /** libsvm test files directory (version 2) */
	std::string svmmodir;   /** libsvm model files directory (version 2) */
	std::string svmoudir;   /** libsvm output files directory (version 2) */
	std::string pposdir;    /** Patch location files directory (version 2) */
	std::string s2dir;       /** S2 features directory (version 2) */
	std::string c2dir;       /** C2 features directory (version 2) */
	std::string keydir;      /** key files directory (version 2) */
	std::string svmtrainexe; /** Path of libsvm svm-train exe */
	std::string svmpredexe;  /** Path of libsvm svm-predict exe */
	

	public:
	/** Constructor function */
	GaborFilterModel( std::string _conffile="caltech101.conf",const string &kfile_="")
		: conffile(_conffile), keyfile(kfile_),pe(new ParameterExtractor(conffile))
	{
		isz = cvSize(ImageSize,ImageSize); // Set image size
		// Generate 
		genModelParam(); /** Initilize SP model parameters */
		genFilterBank(); // Create all Gabor filters
		this->FVDim = FVDim; // Set dimensions of feature vectors to the default 
		maxemd = -INF;

		if ( kfile_ ==  "version2" )
		{
			version     = "version2"; // Set code version number
			// Read top level directory names
			inputdir    = pe->getParameter("INPDIR");
			outputdir   = pe->getParameter("OUTDIR");
			svmtrainexe = pe->getParameter("SVMTRAINEXE");
			svmpredexe  = pe->getParameter("SVMPREDEXE");
			// Form directory names
			trdir       = inputdir  + OSPATHSEP + "train";
			tedir       = inputdir  + OSPATHSEP + "test";
			urldir      = inputdir  + OSPATHSEP + "url";
			svmtrdir    = outputdir + OSPATHSEP + "svmtr";
			svmtedir    = outputdir + OSPATHSEP + "svmte";
			svmmodir    = outputdir + OSPATHSEP + "svmmo";
			svmoudir    = outputdir + OSPATHSEP + "svmou";
			pposdir     = outputdir + OSPATHSEP + "ppos";
			s2dir       = outputdir + OSPATHSEP + "s2";
			c2dir       = outputdir + OSPATHSEP + "c2";
			keydir      = outputdir + OSPATHSEP + "key";

		}
		else
		{
			//isz = cvSize(ImageSize,ImageSize);
			if ( kfile_ == "" )
			{
				keyfile = pe->getParameter("KEYFILE");
			}
			else
			{
				keyfile = kfile_;
			}
			imagedir = pe->getParameter("IMAGEDIR");
			readKeyFile(keyfile);


			//	cannydir = pe->getParameter("CANNYDIR"); // uncomment if required
			s2file = pe->getParameter("S2FILE");
			//	wordsfile = pe->getParameter("WORDSFILE"); // uncomment if required
		}
	}

	void genModelParam();  /** Create Serre-Poggio model parameters */
	void calcS2(const string &s2file_="");  /** Calculate S2 features */
	void readKeyFile(const string &kfile);  /** Loads key file into memory */
	void createKeyFile(const string &dir, const string &kfile="filelist.key"); /** Create a key file */
	void getC1(IplImage *gray,const int32_t xpos,const int32_t ypos,range_t &R); /** Create C1 layer */
	void genFilterBank(); /** Generate Gabor Filter co-efficients */
	void getGaborCoeff(const int16_t filterSize, 
			const double effectiveWidth,
			const double wavelength,
			const double theta,const 
			double aspectRatio,
			filter_t &f);  /** Calculate response with a Gabor filter */ 
	double getGaborRespAtPt(IplImage *gray,
			const int32_t xpos,
			const int32_t ypos, 
			int32_t sIndex,
			int32_t thetaIndex); /** Gabor response at a point */
	void doMaxOp(const mmat_t &in,mmat_t &out); /** Perform MAX over loaction and scale */
	void createRandomPatches(const string& pposfile="");  /** Generate patches at random positions */
	void createAllPatches(); /** Generate all possible patches */
	void createAllPatchesTrain(); 
	void getRandomPos(const long bandNum, long &ipos, long &jpos, long &psizeNum);
	void createRandomPatches(const long lb,const long ub, vpatchloc_t& rp);
	void calcFVStage1(const string &s2filenm="", const string &pposfilenm="");
	void calcFVStage2(const string &c2file="");
	double maxresp(const bvmat_t &A,const bvmat_t &B);
	void createCornerPatches();
	void createCornerPatches(const long lb,const long ub, vpatchloc_t& rp);
	void getCornerPos( const long imgNum, vpatchloc_t &rp);
	void getEdgePoints( IplImage *img, IplImage *out);
	void createEdgePatches();
	void createOverlappingGridPatches();
	void formCodeBook();
	void calcFVStage2Words();
	void runLSA();
	void runKernelEMD();
	void getAllPairsDist( const Matrix& D, const sivec_t &v1,
			const sivec_t &v2 , vec_t &A);
	double getKernelEntry(const Matrix& D, const vsivec_t &W,
			const mvsivec_t &Combination,  const int64_t x1, const int64_t x2);
	double getKernelEntry(const Matrix& D, const vsivec_t &W,
			const int64_t x1, const int64_t x2);
	void genPartialMatchPoss(const int32_t n,const int32_t r, vsivec_t &ans);
	void fillS2PatchesWithRandomNumbers();
	//double calcEMDKernelEntry(const CvMat &D, const float * const points,
	//	const int64_t i, const int64_t j, const int64_t d1, const int64_t d2); 
	double calcEMDKernelEntry( const Matrix &D, const vsivec_t &wd,
		const vsivec_t &wdcnt, const int64_t i, const int64_t j,
		const int64_t d1, const int64_t d2);
	void printRow(const mat_t &S, const int64_t i,
		const vector<int64_t> &ImgId, const int64_t trsz, 
		std::ofstream &fo1);
	void printRow2(const vec_t &S, const int64_t i,
		const vector<int64_t> &ImgId, const int64_t trsz, 
		std::ofstream &fo2);
	void calcEMDDirectly();
	float getEMDVal(float points[NFiles][NumPatchVecPerImage][PLength], int64_t i, int64_t j);
	void multipleSVM(const int32_t groupsize,const int32_t repeatFactor);
	void calcFVStage1Multi();
	void calcFVStage2Multi(const ivec_t &execlist, const int32_t id);
	void calcC2Multi(const int32_t catid, 
			const int32_t intid, const ivec_t &execlist,
			vec_t &output);
	void createPatternsMulti(vivec_t &v);
	void buildOneClassSVM(const std::string &cls,const int32_t testclassid=-1);
	void testOneClassSVM(const std::string &cls,const std::string &testdir, const int32_t testclassid=-1) ;
	void buildModelForQuery(const std::string &cls);
	void convC2ToSVM(const string &c2file,const string &svmfile,const int32_t dim,const int32_t classid=-1);
	void readS2File(const string &s2filenm) ;
	void buildClassifierForQuery(const string &query);
	void basicSerrePoggio();
	void trainCarFilter();
	void testCarFilter();
	void svmtrain(const string & svmtrfilenm, const string &svmmodelfilenm);


};

//inline float dist(feature_t *F1, feature_t *F2) { return cm[(*F1)*len+(*F2)]; } // precomputed cost matrix
//float dist2(feature_t *F1, feature_t *F2); // Euclidean distance

#endif
