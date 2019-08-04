// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

/// \file TrackResiduals.h
/// \brief Definition of the TrackResiduals class
///
/// \author Ruben Shahoyan, ruben.shahoyan@cern.ch (original code in AliRoot)
///         Ole Schmidt, ole.schmidt@cern.ch (porting to O2)

#ifndef ALICEO2_TPC_TRACKRESIDUALS_H_
#define ALICEO2_TPC_TRACKRESIDUALS_H_

#define TPC_RUN2

#include <memory>
#include <vector>
#include <array>
#include <bitset>
#include <string>
#include <Rtypes.h>

#include "DataFormatsTPC/Defs.h"
#include "SpacePoints/SpacePointsCalibParam.h"
#include "SpacePoints/TrackInterpolation.h"

#include "TTree.h"
#include "TFile.h"
#include "TChain.h"
#include "TVectorT.h"

namespace o2
{
namespace tpc
{

/// \class TrackResiduals
/// This class is steering the space point calibration of the TPC from track residuals.
/// Residual maps are created using track interpolation from ITS/TRD/TOF tracks and comparing
/// them to the cluster positions in the TPC.
/// It has been ported from the AliTPCDcalibRes clas from AliRoot.
class TrackResiduals
{
 public:
  /// Default constructor
  TrackResiduals() = default;

  /// Enumeration for different voxel dimensions
  enum { VoxZ,          ///< Z/X index
         VoxF,          ///< Y/X index
         VoxX,          ///< X index
         VoxV,          ///< voxel dispersions
         VoxDim = 3,    ///< dimensionality of the voxels
         VoxHDim = 4 }; ///< dimensionality of the voxel + 1 for kernel weights

  /// Enumeration for the result indices
  enum { ResX,     ///< X index
         ResY,     ///< Y index
         ResZ,     ///< Z index
         ResD,     ///< index for dispersions
         ResDim }; ///< dimensionality for results structure (X, Y, Z and dispersions)

  /// Enumeration for voxel status flags
  enum { DistDone = 1,   ///< voxel residuals have been processed
         DispDone = 2,   ///< voxel dispersions have been processed
         SmoothDone = 4, ///< voxel has been smoothed
         Masked = 8 };   ///< voxel is masked

  enum class KernelType { Epanechnikov,
                          Gaussian };

  /// Structure which gets filled with the results for each voxel
  struct VoxRes {
    std::array<float, ResDim> D{};            ///< values of extracted distortions
    std::array<float, ResDim> E{};            ///< their errors
    std::array<float, ResDim> DS{};           ///< smoothed residual
    std::array<float, ResDim> DC{};           ///< Cheb parameterized residual
    float EXYCorr{0.f};                       ///< correlation between extracted X and Y
    float dYSigMAD{0.f};                      ///< MAD estimator of dY sigma (dispersion after slope removal)
    float dZSigLTM{0.f};                      ///< Z sigma from unbinned LTM estimator
    std::array<float, VoxHDim> stat{};        ///< statistics: averages of each voxel dimension + entries
    std::array<unsigned char, VoxDim> bvox{}; ///< voxel identifier: VoxZ, VoxF, VoxX
    unsigned char bsec{0};                    ///< sector ID (0-35)
    unsigned char flags{0};                   ///< status flag
  };

  /// Structure for local residuals (y/z position, dip angle, voxel identifier)
  struct LocalResid {
    short dy;                                 ///< residual in y, ranges from -param::sMaxResid to +param::sMaxResid
    short dz;                                 ///< residual in z, ranges from -param::sMaxResid to +param::sMaxResid
    short tgSlp;                              ///< track dip angle, ranges from -param::sMaxAngle to +param::sMaxAngle
    std::array<unsigned char, VoxDim> bvox{}; ///< voxel identifier: VoxZ, VoxF, VoxX
  };

  /// Helper structure to organize acess to delta trees from Run2 (legacy method)
  /// All parameters are on a per-track basis
  struct DeltaStruct {
    TVectorF* vecR{nullptr};     ///< cluster radius
    TVectorF* vecSec{nullptr};   ///< cluster sector (0..71) A/C side, IROC/OROC
    TVectorF* vecPhi{nullptr};   ///< azimuthal angle of cluster frame (-pi..pi)
    TVectorF* vecZ{nullptr};     ///< cluster z position
    TVectorF* vecDYits{nullptr}; ///< cluster y residual wrt ITS track
    TVectorF* vecDZits{nullptr}; ///< cluster z residual wrt ITS track
    TVectorF* vecDYtrd{nullptr}; ///< cluster y residual wrt ITS-TRD track
    TVectorF* vecDZtrd{nullptr}; ///< cluster z residual wrt ITS-TRD track
    Double32_t param[5] = {0.f}; ///< track parameters at inner wall of TPC
    Char_t trdOK{0};             ///< track had matched points in TRD
    Char_t itsOK{0};             ///< track had matched points in ITS
    UShort_t npValid{0};         ///< number of valid TPC clusters
  };

  // -------------------------------------- initialization --------------------------------------------------
  /// Steers the initialization (binning, default settings for smoothing, container for the results).
  void init();
  /// Initializes the binning in X, Y/X and Z.
  void initBinning();
  /// Initializes the results structure for given sector.
  /// For each voxel the bin indices are set and the COG is set to the center of the voxel.
  /// \param iSec TPC sector number
  void initResultsContainer(int iSec);
  /// Resets all (also intermediate) results
  void reset();

  // -------------------------------------- settings --------------------------------------------------
  /// Sets a flag to print the memory usage at certain points in the program for performance studies.
  void setPrintMemoryUsage() { mPrintMem = true; }
  /// Sets the kernel type used for smoothing.
  /// \param kernel Kernel type (Epanechnikov / Gaussian)
  /// \param bwX Bin width in X
  /// \param bwP Bin width in Y/X
  /// \param bwZ Bin width in Z
  /// \param scX Scale factor to increase smoothing bandwidth at sector edges in X
  /// \param scP Scale factor to increase smoothing bandwidth at sector edges in Y/X
  /// \param scZ Scale factor to increase smoothing bandwidth at sector edges in Z
  void setKernelType(KernelType kernel = KernelType::Epanechnikov, float bwX = 2.1f, float bwP = 2.1f, float bwZ = 1.7f, float scX = 1.f, float scP = 1.f, float scZ = 1.f);

  // -------------------------------------- steering functions --------------------------------------------------

  /// Build local residual trees from Run 2 legacy data in the same way as in AliRoot
  void buildLocalResidualTreesFromRun2Data();

  /// Fill the tree with local residuals with input from the buffer arrays
  void fillLocalResidualsTrees();

  /// Activate necessary branches from Run 2 delta trees
  void prepareDeltaTreeBranches();

  /// Create output files for each sector with trees for local residuals
  void prepareLocalResidualTrees();

  /// Write trees with local residuals to file
  void writeLocalResidualTreesToFile();

  /// Loads residual data from track interpolation and fills voxel data structures local residuals
  void convertToLocalResiduals();

  /// Steers the processing of the residuals for all sectors.
  void processResiduals();

  /// Processes residuals for given sector.
  /// \param iSec Sector to process
  void processSectorResiduals(Int_t iSec);

  /// Performs the robust linear fit for one voxel to estimate the distortions in X, Y and Z and their errors.
  /// \param dy Vector with residuals in y
  /// \param dz Vector with residuals in z
  /// \param tg Vector with tan(phi) of the tracks
  /// \param resVox Voxel results structure
  void processVoxelResiduals(std::vector<float>& dy, std::vector<float>& dz, std::vector<float>& tg, VoxRes& resVox);

  /// Estimates dispersion for given voxel
  /// \param tg Vector with tan(phi) of the tracks
  /// \param dy Vector with residuals in y
  /// \param resVox Voxel results structure
  void processVoxelDispersions(std::vector<float>& tg, std::vector<float>& dy, VoxRes& resVox);

  /// Applies voxel validation cuts.
  /// Bad X bins are stored in mXBinsIgnore bitset
  /// \param iSec Sector to process
  /// \return Number of good rows in X
  int validateVoxels(int iSec);

  /// Smooths the residuals for given sector
  /// \param iSec Sector to process
  void smooth(int iSec);

  // -------------------------------------- statistics --------------------------------------------------

  /// Performs a robust linear fit y(x) = a + b * x for given x and y.
  /// The input data is trimmed to reject outliers.
  /// \param x First vector with input data
  /// \param y Second vector with input data
  /// \param res Array storing the fit results a and b
  /// \param err Array storing the uncertainties
  /// \param cutLTM Fraction of the input data to keep
  /// \return Median of the absolute deviations of the median of the data points to the fit
  float fitPoly1Robust(std::vector<float>& x, std::vector<float>& y, std::array<float, 2>& res, std::array<float, 3>& err, float cutLTM) const;

  /// Calculates the median of the absolute deviations to the median of the data.
  /// The input vector is copied such that the original vector is not modified.
  /// \param data Input data vector
  /// \return Median of absolute deviations to the median
  float getMAD2Sigma(std::vector<float> data) const;

  /// Fits a straight line to given x and y minimizing the absolute deviations y(x|a, b) = a + b * x.
  /// Not all data points need to be considered, but only a fraction of the input is used to perform the fit.
  /// \param nPoints Number of points to consider
  /// \param offset Starting index for the input vectors
  /// \param x First vector with input data
  /// \param y Second vector with input data
  /// \param a Stores the result for a
  /// \param b Stores the result for b
  /// \param err Stores the uncertainties
  void medFit(int nPoints, int offset, const std::vector<float>& x, const std::vector<float>& y, float& a, float& b, std::array<float, 3>& err) const;

  /// Helper function for medFit.
  /// Calculates sum(x_i * sgn(y_i - a - b * x_i)) for a given b
  /// \param nPoints Number of points to consider
  /// \param offset Starting index for the input vectors
  /// \param x First vector with input data
  /// \param y Second vector with input data
  /// \param b Given b
  /// \param aa Parameter a for linear fit (will be set by roFunc)
  /// \return The calculated sum
  float roFunc(int nPoints, int offset, const std::vector<float>& x, const std::vector<float>& y, float b, float& aa) const;

  /// Returns the k-th smallest value in the vector.
  /// The input vector is rearranged such that the k-th smallest value is at the k-th position.
  /// \todo Can probably be replaced by std::nth_element(), need to check which one is faster
  /// All smaller values will be placed in before it in arbitrary order, all large values behind it in arbitrary order.
  /// \param k Which value to get
  /// \param data Vector with input data
  /// \return k-th smallest value in the input vector
  float selectKthMin(const int k, std::vector<float>& data);

  /// Calculates a smooth estimate for the distortions in specified dimensions around the COG for a given voxel.
  /// \param iSec Sector in which the voxel is located
  /// \param x COG position in X
  /// \param p COG position in Y/X
  /// \param z COG position in Z
  /// \param res Array to store the results
  /// \param whichDim Integer value with bits set for the dimensions which need to be smoothed
  /// \return Flag if the estimate was successfull
  bool getSmoothEstimate(int iSec, float x, float p, float z, std::array<float, ResDim>& res, int whichDim = 0);

  /// Calculates the weight of the given point used for the kernel smoothing.
  /// Takes into account the defined kernel in mKernelType.
  /// \param u2vec Weighted distance in X, Y/X and Z
  /// \return Kernel weight
  double getKernelWeight(std::array<double, 3> u2vec) const;

  /// Calculates the differences in Y and Z for a given set of clusters to a fitted helix.
  /// First a circular fit in the azimuthal plane is performed and subsequently a linear fit in the transversal plane
  bool compareToHelix(std::array<float, param::NPadRows>& residHelixY, std::array<float, param::NPadRows>& residHelixZ);

  /// Fits a circle to a given set of points in x and y. Kasa algorithm is used.
  /// \param nCl number of used points
  /// \param x array with values for x
  /// \param y array with values for y
  /// \param xc fit result for circle center position in x is stored here
  /// \param yc fit result for circle center position in y is stored here
  /// \param r fit result for circle radius is stored here
  /// \param residHelixY residuals in y from fitted circle to given points is stored here
  void fitCircle(int nCl, std::array<float, param::NPadRows>& x, std::array<float, param::NPadRows>& y, float& xc, float& yc, float& r, std::array<float, param::NPadRows>& residHelixY);

  /// Fits a straight line to a given set of points, w/o taking into account measurement errors or different weights for the points
  /// Straight line is given by y = a * x + b
  /// \param res[0] contains the slope (a)
  /// \param res[1] contains the offset (b)
  bool fitPoly1(int nCl, std::array<float, param::NPadRows>& x, std::array<float, param::NPadRows>& y, std::array<float, 2>& res);

  /// For a given set of points, calculate the differences from each point to the fitted lines from all other points in their neighbourhoods (+- mNMALong points)
  void diffToLocLine(int np, int idxOffset, const std::array<float, param::NPadRows>& x, const std::array<float, param::NPadRows>& y, std::array<float, param::NPadRows>& diffY);

  /// For a given set of points, calculate their deviation from the moving average (build from the neighbourhood +- mNMALong points)
  void diffToMA(int np, const std::array<float, param::NPadRows>& y, std::array<float, param::NPadRows>& diffMA);

  // -------------------------------------- binning / geometry --------------------------------------------------

  /// Calculates the global bin number
  /// \param ix Bin index in X
  /// \param ip Bin index in Y/X
  /// \param iz Bin index in Z/X
  /// \return global bin number
  unsigned short getGlbVoxBin(int ix, int ip, int iz) const;

  /// Calculates the global bin number
  /// \param bvox Array with the voxels bin indices in X, Y/X and Z/X
  /// \return global bin number
  unsigned short getGlbVoxBin(const std::array<unsigned char, VoxDim>& bvox) const;

  /// Calculates the coordinates of the center for a given voxel.
  /// These are not global TPC coordinates, but the coordinates for the given global binning system.
  /// E.g. z ranges from -1 to 1.
  /// \param isec The sector in which we are
  /// \param ix Bin index in X
  /// \param ip Bin index in Y/X
  /// \param iz Bin index in Z
  /// \param x Coordinate in X
  /// \param p Coordinate in Y/X
  /// \param z Coordinate in Z
  void getVoxelCoordinates(int isec, int ix, int ip, int iz, float& x, float& p, float& z) const;

  /// Calculates the x-coordinate for given x bin.
  /// \param i Bin index
  /// \return Coordinate in X
  float getX(int i) const;

  /// Calculates the y/x-coordinate.
  /// \param ix Bin index in X
  /// \param ip Bin index in Y/X
  /// \return Coordinate in Y/X
  float getY2X(int ix, int ip) const;

  /// Calculates the z-coordinate for given z bin
  /// \param i Bin index
  /// \return Coordinate in Z
  float getZ(int i) const;

  /// Tests whether a bin in X is set to be ignored.
  /// \param iSec Sector number
  /// \param bin Bin index in X
  /// \return Ignore flag
  bool getXBinIgnored(int iSec, int bin) const { return mXBinsIgnore[iSec].test(bin); }

  /// Calculates the bin indices of the closest voxel.
  /// \param x Coordinate in X
  /// \param y2x Coordinate in Y/X
  /// \param z2x Coordinate in Z
  /// \param ix Resulting bin index in X
  /// \param ip Resulting bin index in Y/X
  /// \param iz Resulting bin index in Z/X
  void findVoxel(float x, float y2x, float z2x, int& ix, int& ip, int& iz) const;

  /// Calculates the bin indices for given x, y, z in sector coordinates
  bool findVoxelBin(float x, float y, float z, std::array<unsigned char, VoxDim>& bvox) const;

  /// Transforms X coordinate to bin index
  /// \param x Coordinate in X
  /// \return Bin index in X
  int getXBin(float x) const;

  /// Transforms Y/X coordinate to bin index at given X
  /// \param y2x Coordinate in Y/X
  /// \param ix Bin index in X
  /// \return Bin index in Y/X
  int getY2XBin(float y2x, int ix) const;

  /// Transforms Z coordinate to bin index
  /// \param z2x Coordinate in Z
  /// \return Bin index in Z/X
  int getZ2XBin(float z2x) const;

  /// Returns the inverse of the distance between two bins in X
  /// \param ix Bin index in X
  /// \return Inverse of the distance between bins
  float getDXI(int ix) const;

  /// Returns the inverse of the distance between two bins in Y/X
  /// \param ix Bin index in X
  /// \return Inverse of the distance between bins
  float getDY2XI(int ix) const { return mDY2XI[ix]; }

  /// Returns the inverse of the distance between two bins in Z/X
  /// \return Inverse of the distance between bins
  float getDZ2XI() const { return mDZI; }

  // -------------------------------------- settings --------------------------------------------------

  void setLocalResFileName(std::string fName) { mLocalResFileName = fName; }
  void setLocalResTreeName(std::string tName) { mLocalResTreeName = tName; }
  void setLocalResBranchName(std::string bName) { mLocalResBranchName = bName; }
  void setMaxPointsPerSector(int nPoints) { mMaxPointsPerSector = nPoints; }
  void setMinEntriesPerVoxel(int nEntries) { mMinEntriesPerVoxel = nEntries; }
  void setLTMCut(float ltmCut) { mLTMCut = ltmCut; }
  void setMinFracLTM(float ltmCut) { mMinFracLTM = ltmCut; }
  void setMinValidVoxFracDrift(float frac) { mMinValidVoxFracDrift = frac; }
  void setMinGoodXBinsToCover(int n) { mMinGoodXBinsToCover = n; }
  void setMaxBadXBinsToCover(int n) { mMaxBadXBinsToCover = n; }
  void setMaxFracBadRowsPerSector(float frac) { mMaxFracBadRowsPerSector = frac; }
  void setMaxFitErrY2(float err) { mMaxFitErrY2 = err; }
  void setMaxFitErrX2(float err) { mMaxFitErrX2 = err; }
  void setMaxFitCorrXY(float corr) { mMaxFitCorrXY = corr; }
  void setMaxSigY(float sigY) { mMaxSigY = sigY; }
  void setMaxSigZ(float sigZ) { mMaxSigZ = sigZ; }
  void setMaxGaussStdDev(float sigmas) { mMaxGaussStdDev = sigmas; }

  std::string getLocalResFileName() const { return mLocalResFileName; }
  std::string getLocalResTreeName() const { return mLocalResTreeName; }
  std::string getLocalResBranchName() const { return mLocalResBranchName; }
  int getMaxPointsPerSector() const { return mMaxPointsPerSector; }
  int getMinEntriesPerVoxel() const { return mMinEntriesPerVoxel; }
  float getLTMCut() const { return mLTMCut; }
  float getMinFracLTM() const { return mMinFracLTM; }
  float getMinValidVoxFracDrift() const { return mMinValidVoxFracDrift; }
  int getMinGoodXBinsToCover() const { return mMinGoodXBinsToCover; }
  int getMaxBadXBinsToCover() const { return mMaxBadXBinsToCover; }
  float getMaxFracBadRowsPerSector() const { return mMaxFracBadRowsPerSector; }
  float getMaxFitErrY2() const { return mMaxFitErrY2; }
  float getMaxFitErrX2() const { return mMaxFitErrX2; }
  float getMaxFitCorrXY() const { return mMaxFitCorrXY; }
  float getMaxSigY() const { return mMaxSigY; }
  float getMaxSigZ() const { return mMaxSigZ; }
  float getMaxGaussStdDev() const { return mMaxGaussStdDev; }

  // ------------------------- conversion of delta trees -> compact trees ------------------------------
  /// For use with Run 2 data, outlier filtering
  bool validateTrack(std::array<int, 3>& counterTrkValidation);

  /// For use with Run 2 data, outlier filtering
  int checkResiduals(std::bitset<param::NPadRows>& rejCl, float& rmsLong);

  // -------------------------------------- debugging --------------------------------------------------

  /// Prints the current memory usage
  void printMem() const;

  /// Dumps the content of a vector to the specified file
  /// \param vec Data vector
  /// \param fName Filename
  void dumpToFile(const std::vector<float>& vec, const std::string fName) const;

  /// Dumps the full results for a given sector to the debug tree (only if an output file has been created before).
  /// \param iSec Sector to dump
  void dumpResults(int iSec);

  /// Creates a file for the debug output.
  void createOutputFile();

  /// Closes the file with the debug output.
  void closeOutputFile();

 private:
  // names of input files / trees
  std::string mInputFileNameResiduals{"residuals_tpc.root"}; ///< name of file with track residuals
  // some constants
  static constexpr float sFloatEps{1.e-7f}; ///< float epsilon for robust linear fitting
  static constexpr float sDeadZone{1.5f};   ///< dead zone for TPC in between sectors
  static constexpr float sMaxZ2X{1.f};      ///< max value for Z2X
  static constexpr int sSmtLinDim{4};       ///< max matrix size for smoothing (pol1)
  static constexpr int sMaxSmtDim{7};       ///< max matrix size for smoothing (pol2)

  // input data
  std::unique_ptr<TFile> mFileIn{};                     ///< input file with residuals data
  TTree* mTreeInTracks{nullptr};                        ///< tree with input track information
  std::vector<TrackData> mTrackData{};                  ///< vector with input track information
  std::vector<TrackData>* mTrackDataPtr{&mTrackData};   ///< pointer to mTrackData
  TTree* mTreeInClRes{nullptr};                         ///< tree with TPC cluster residuals
  std::vector<TPCClusterResiduals> mClRes{};            ///< vector with TPC cluster residuals
  std::vector<TPCClusterResiduals>* mClResPtr{&mClRes}; ///< pointer to mClRes
  // output data
  std::unique_ptr<TFile> mFileOut{}; ///< output debug file
  std::unique_ptr<TTree> mTreeOut{}; ///< tree holding debug output
  // status flags
  bool mIsInitialized{}; ///< initialize only once
  bool mPrintMem{};      ///< turn on to print memory usage at certain points
  // binning
  int mNXBins{param::NPadRows};            ///< number of bins in radial direction
  int mNY2XBins{param::NY2XBins};          ///< number of y/x bins per sector
  int mNZ2XBins{param::NZ2XBins};          ///< number of z/x bins per sector
  int mNVoxPerSector{};                    ///< number of voxels per sector
  float mDX{};                             ///< x bin size
  float mDXI{};                            ///< inverse of x bin size
  std::vector<float> mMaxY2X{};            ///< max y/x at each x bin, accounting dead zones
  std::vector<float> mDY2X{};              ///< y/x bin size at given x bin
  std::vector<float> mDY2XI{};             ///< inverse y/x bin size at given x bin
  float mDZ{};                             ///< bin size in z
  float mDZI{};                            ///< inverse of bin size in z
  std::array<bool, VoxDim> mUniformBins{}; ///< if binning is uniform for each dimension
  // local residual data, extracted from track interpolation
  std::array<std::unique_ptr<TFile>, SECTORSPERSIDE * SIDES> mTmpFile{}; ///< I/O file
  std::array<std::unique_ptr<TTree>, SECTORSPERSIDE * SIDES> mTmpTree{}; ///< I/O tree per sector
  LocalResid mLocalResid{};                                              ///< data exchange structure for filling mTmpTree
  LocalResid* mLocalResidPtr{&mLocalResid};                              ///< pointer to mLocalResid
  // settings
  std::string mLocalResFileName{"deltasSect"};   ///< filename for local residuals input
  std::string mLocalResTreeName{"treeSec"};      ///< name for tree with local residuals
  std::string mLocalResBranchName{"localResid"}; ///< branch with LocalResid objects
  int mMaxPointsPerSector{30'000'000};           ///< maximum number of accepted points per sector
  int mMinEntriesPerVoxel{15};                   ///< minimum number of points in voxel for processing
  float mLTMCut{.75f};                           ///< fraction op points to keep when trimming input data
  float mMinFracLTM{.5f};                        ///< minimum fraction of points to keep when trimming data to fit expected sigma
  float mMinValidVoxFracDrift{.5f};              ///< if more than this fraction of bins are bad for one pad row the bad row is declared bad
  int mMinGoodXBinsToCover{3};                   ///< minimum number of consecutive good bins, otherwise bins are declared bad
  int mMaxBadXBinsToCover{4};                    ///< a lower number of consecutive bad X bins will not be declared bad
  float mMaxFracBadRowsPerSector{.4f};           ///< maximum fraction of bad rows before whole sector is masked
  float mMaxFitErrY2{1.f};                       ///< maximum fit error for Y2
  float mMaxFitErrX2{9.f};                       ///< maximum fit error for X2
  float mMaxFitCorrXY{.95f};                     ///< maximum fit correlation for x and y
  float mMaxSigY{1.1f};                          ///< maximum sigma for y of the voxel
  float mMaxSigZ{.7f};                           ///< maximum sigma for z of the voxel
  float mMaxGaussStdDev{5.f};                    ///< maximum number of sigmas to be considered for gaussian kernel smoothing
  // smoothing
  KernelType mKernelType{KernelType::Epanechnikov};                ///< kernel type (Epanechnikov / Gaussian)
  bool mUseErrInSmoothing{true};                                   ///< weight kernel by point error
  std::array<bool, VoxDim> mSmoothPol2{};                          ///< option to use pol1 or pol2 in each direction
  std::array<int, SECTORSPERSIDE * SIDES> mNSmoothingFailedBins{}; ///< number of failed bins / sector
  std::array<int, VoxDim> mStepKern{};                             ///< N bins to consider with given kernel settings
  std::array<float, VoxDim> mKernelScaleEdge{};                    ///< optional scaling factors for kernel width on the edge
  std::array<float, VoxDim> mKernelWInv{};                         ///< inverse kernel width in bins
  std::array<double, ResDim * sMaxSmtDim> mLastSmoothingRes{};     ///< results of last smoothing operation
  // (intermediate) results
  std::array<std::bitset<param::NPadRows>, SECTORSPERSIDE * SIDES> mXBinsIgnore{};          ///< flags which X bins to ignore
  std::array<std::array<float, param::NPadRows>, SECTORSPERSIDE * SIDES> mValidFracXBins{}; ///< for each sector for each X-bin the fraction of validated voxels
  std::array<std::vector<VoxRes>, SECTORSPERSIDE * SIDES> mVoxelResults{};                  ///< results per sector and per voxel for 3-D distortions
  // conversion of Run 2 data to local residuals
  std::string mPathToResidualFiles{"~/tmp/"};
  std::string mResidualDataFileName{"ResidualTrees.root"};
  std::string mResidualDataTreeName{"delta"};
  DeltaStruct mDeltaStruct;
  std::unique_ptr<TChain> mRun2DeltaTree{};
  bool mFilterOutliers = true;
  int mNMALong{15}; ///< number of points to be used for moving average (long range)
  float mMaxRejFrac{.15f};
  float mMaxRMSLong{.8f};
  // buffer arrays as in AliTPCDcalibRes
  std::array<float, param::NPadRows> mArrX;
  std::array<float, param::NPadRows> mArrR;
  std::array<float, param::NPadRows> mArrYTr;
  std::array<float, param::NPadRows> mArrZTr;
  std::array<float, param::NPadRows> mArrYCl;
  std::array<float, param::NPadRows> mArrZCl;
  std::array<float, param::NPadRows> mArrDZ;
  std::array<float, param::NPadRows> mArrDY;
  std::array<float, param::NPadRows> mArrPhi;
  std::array<float, param::NPadRows> mArrTgSlp;
  std::array<int, param::NPadRows> mArrSecId;
  float mQpt{0.f};
  float mTgl{0.f};
  int mNCl{0};
};

//_____________________________________________________
inline unsigned short TrackResiduals::getGlbVoxBin(const std::array<unsigned char, VoxDim>& bvox) const
{
  return bvox[VoxX] + (bvox[VoxF] + bvox[VoxZ] * mNY2XBins) * mNXBins;
}

//_____________________________________________________
inline unsigned short TrackResiduals::getGlbVoxBin(int ix, int ip, int iz) const
{
  return ix + (ip + iz * mNY2XBins) * mNXBins;
}

//_____________________________________________________
inline void TrackResiduals::getVoxelCoordinates(int isec, int ix, int ip, int iz, float& x, float& p, float& z) const
{
  x = getX(ix);
  p = getY2X(ix, ip);
  z = getZ(iz);
  if (isec >= SECTORSPERSIDE) {
    z = -z;
  }
}

//_____________________________________________________
inline float TrackResiduals::getDXI(int ix) const
{
  if (mUniformBins[VoxX]) {
    return mDXI;
  } else {
    if (ix < param::NRowsPerROC[0]) {
      // we are in the IROC
      return 1.f / param::RowDX[0];
    } else if (ix > param::NRowsAccumulated[param::NROCTypes - 1]) {
      // we are in the last OROC
      return 1.f / param::RowDX[param::NROCTypes - 1];
    }
#ifdef TPC_RUN2
    else {
      // we are in OROC1
      return 1.f / param::RowDX[1];
    }
#else
    else if (ix < param::NRowsAccumulated[2]) {
      // OROC1
      return 1.f / param::RowDX[1];
    } else {
      // OROC2
      return 1.f / param::RowDX[2];
    }
#endif
  }
}

//_____________________________________________________
inline float TrackResiduals::getX(int i) const
{
  return mUniformBins[VoxX] ? param::MinX[0] + (i + 0.5) * mDX : param::RowX[i];
}

//_____________________________________________________
inline float TrackResiduals::getY2X(int ix, int ip) const
{
  return (0.5f + ip) * mDY2X[ix] - mMaxY2X[ix];
}

//_____________________________________________________
inline float TrackResiduals::getZ(int i) const
{
  // always positive
  return (0.5f + i) * mDZ;
}

//_____________________________________________________
inline void TrackResiduals::findVoxel(float x, float y2x, float z2x, int& ix, int& ip, int& iz) const
{
  ix = getXBin(x);
  ip = getY2XBin(y2x, ix);
  iz = getZ2XBin(z2x);
}

//_____________________________________________________
inline int TrackResiduals::getY2XBin(float y2x, int ix) const
{
  int bp = (y2x + mMaxY2X[ix]) * mDY2XI[ix];
  if (bp < 0) {
    bp = 0;
  }
  return (bp < mNY2XBins) ? bp : mNY2XBins - 1;
}

//_____________________________________________________
inline int TrackResiduals::getZ2XBin(float z2x) const
{
  int bz = z2x * mDZI;
  if (bz < 0) {
    // accounting for clusters which were moved to the wrong side
    // TODO: how can this happen?
    bz = 0;
  }
  return (bz < mNZ2XBins) ? bz : mNZ2XBins - 1;
}

} // namespace tpc

} // namespace o2

// This is a hack to load the local residual trees created with AliRoot into O2
namespace AliTPCDcalibRes
{
struct dts_t {                                   // struct for basic local residual
  Double32_t dy;                                 //[-20.,20.,15] // [-kMaxResid,kMaxResid,14]
  Double32_t dz;                                 //[-20.,20.,15] // [-kMaxResid,kMaxResid,14]
  Double32_t tgSlp;                              //[-2,2,14]  //[kMaxTgSlp,kMaxTgSlp,14]
  UChar_t bvox[o2::tpc::TrackResiduals::VoxDim]; // voxel bin info: VoxF,VoxX,VoxZ
  //
  dts_t() { memset(this, 0, sizeof(dts_t)); }
};
} // namespace AliTPCDcalibRes

#endif
