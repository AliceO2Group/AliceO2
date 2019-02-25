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
/// \author Ole Schmidt, ole.schmidt@cern.ch

#ifndef ALICEO2_CALIB_TRACKRESIDUALS_H_
#define ALICEO2_CALIB_TRACKRESIDUALS_H_

#include <memory>
#include <vector>
#include <bitset>
#include <Rtypes.h>

#include "SpacePoints/Param.h"

#include "TTree.h"
#include "TFile.h"

namespace AliTPCDcalibRes
{
struct dts_t {                            // struct for basic local residual
  Double32_t dy;                          //[-20.,20.,15] // [-kMaxResid,kMaxResid,14]
  Double32_t dz;                          //[-20.,20.,15] // [-kMaxResid,kMaxResid,14]
  Double32_t tgSlp;                       //[-2,2,14]  //[kMaxTgSlp,kMaxTgSlp,14]
  UChar_t bvox[o2::calib::param::VoxDim]; // voxel bin info: VoxF,kVoxX,kVoxZ
  //
  dts_t() { memset(this, 0, sizeof(dts_t)); }
};
} // namespace AliTPCDcalibRes

namespace o2
{
namespace calib
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

  /// Structure which gets filled with the results
  struct bres_t {
    std::array<float, param::ResDim> D{};            // values of extracted distortions
    std::array<float, param::ResDim> E{};            // their errors
    std::array<float, param::ResDim> DS{};           // smoothed residual
    std::array<float, param::ResDim> DC{};           // Cheb parameterized residual
    float EXYCorr{ 0.f };                            // correlation between extracted X and Y
    float dYSigMAD{ 0.f };                           // MAD estimator of dY sigma (dispersion after slope removal)
    float dZSigLTM{ 0.f };                           // Z sigma from unbinned LTM estimator
    std::array<float, param::VoxHDim> stat{};        // statistics: averages of each voxel dimension + entries
    std::array<unsigned char, param::VoxDim> bvox{}; // voxel identifier, here the bvox[0] shows number of Q bins used for Y
    unsigned char bsec{ 0 };                         // sector ID (0-35)
    unsigned char flags{ 0 };                        // status flag
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
  /// \param type Kernel type (Epanechnikov / Gaussian)
  /// \param bwX Bin width in X
  /// \param bwP Bin width in Y/X
  /// \param bwZ Bin width in Z
  /// \param scX Scale factor to increase smoothing bandwidth at sector edges in X
  /// \param scP Scale factor to increase smoothing bandwidth at sector edges in Y/X
  /// \param scZ Scale factor to increase smoothing bandwidth at sector edges in Z
  void setKernelType(int type = param::EpanechnikovKernel, float bwX = 2.1f, float bwP = 2.1f, float bwZ = 1.7f, float scX = 1.f, float scP = 1.f, float scZ = 1.f);

  // -------------------------------------- steering functions --------------------------------------------------

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
  void processVoxelResiduals(std::vector<float>& dy, std::vector<float>& dz, std::vector<float>& tg, bres_t& resVox);

  /// Estimates dispersion for given voxel
  /// \param tg Vector with tan(phi) of the tracks
  /// \param dy Vector with residuals in y
  /// \param resVox Voxel results structure
  void processVoxelDispersions(std::vector<float>& tg, std::vector<float>& dy, bres_t& resVox);

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
  bool getSmoothEstimate(int iSec, float x, float p, float z, std::array<float, param::ResDim>& res, int whichDim = 0);

  /// Calculates the weight of the given point used for the kernel smoothing.
  /// \param u2vec Weighted distance in X, Y/X and Z
  /// \param kernelType Wich kernel is being used
  /// \return Kernel weight
  double getKernelWeight(std::array<double, 3> u2vec, int kernelType) const;

  // -------------------------------------- binning / geometry --------------------------------------------------

  /// Calculates the global bin number
  /// \param ix Bin index in X
  /// \param ip Bin index in Y/X
  /// \param iz Bin index in Z
  /// \return global bin number
  unsigned short getGlbVoxBin(int ix, int ip, int iz) const;

  /// Calculates the global bin number
  /// \param bvox Array with the voxels bin indices in X, Y/X and Z
  /// \return global bin number
  unsigned short getGlbVoxBin(std::array<unsigned char, param::VoxDim> bvox) const;

  /// Calculates the coordinates of the center for a given voxel.
  /// These are not global TPC coordinates, but the coordinates for the given global binning system.
  /// E.g. z ranges from -1 to 1.
  /// \param iSec The sector in which we are
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
  /// \param iz Resulting bin index in Z
  void findVoxel(float x, float y2x, float z2x, int& ix, int& ip, int& iz) const;

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
  /// \return Bin index in Z
  int getZ2XBin(float z2x) const;

  /// Returns the inverse of the distance between two bins in X
  /// \parma ix Bin index in X
  /// \return Inverse of the distance between bins
  float getDXI(int ix) const { return mUniformBins[param::VoxX] ? mDXI : 1.f / param::RowDX[ix]; }

  /// Returns the inverse of the distance between two bins in Y/X
  /// \parma ix Bin index in X
  /// \return Inverse of the distance between bins
  float getDY2XI(int ix) const { return mDY2XI[ix]; }

  /// Returns the inverse of the distance between two bins in Z
  /// \return Inverse of the distance between bins
  float getDZ2XI() const { return mDZI; }

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
  // input data
  std::unique_ptr<TFile> mFileOut{}; ///< output debug file
  std::unique_ptr<TTree> mTreeOut{}; ///< tree holding debug output
  // status flags
  bool mIsInitialized{}; ///< initialize only once
  bool mPrintMem{};      ///< turn on to print memory usage at certain points
  // binning
  int mNXBins{};                                  ///< number of bins in radial direction
  int mNY2XBins{};                                ///< number of y/x bins per sector
  int mNZBins{};                                  ///< number of z/x bins per sector
  int mNVoxPerSector{};                           ///< number of voxels per sector
  float mDX{};                                    ///< x bin size
  float mDXI{};                                   ///< inverse of x bin size
  std::vector<float> mMaxY2X{};                   ///< max y/x at each x bin, accounting dead zones
  std::vector<float> mDY2X{};                     ///< y/x bin size at given x bin
  std::vector<float> mDY2XI{};                    ///< inverse y/x bin size at given x bin
  float mDZ{};                                    ///< bin size in z
  float mDZI{};                                   ///< inverse of bin size in z
  std::array<bool, param::VoxDim> mUniformBins{}; ///< if binning is uniform for each dimension
  // smoothing
  int mKernelType{};                                                        ///< kernel type (Epanechnikov / Gaussian)
  bool mUseErrInSmoothing{ true };                                          ///< weight kernel by point error
  std::array<bool, param::VoxDim> mSmoothPol2{};                            ///< option to use pol1 or pol2 in each direction
  std::array<int, param::NSectors2> mNSmoothingFailedBins{};                ///< number of failed bins / sector
  std::array<int, param::VoxDim> mStepKern{};                               ///< N bins to consider with given kernel settings
  std::array<float, param::VoxDim> mKernelScaleEdge{};                      ///< optional scaling factors for kernel width on the edge
  std::array<float, param::VoxDim> mKernelWInv{};                           ///< inverse kernel width in bins
  std::array<double, param::ResDim * param::MaxSmtDim> mLastSmoothingRes{}; ///< results of last smoothing operation
  // (intermidiate) results
  std::array<std::bitset<param::NPadRows>, param::NSectors2> mXBinsIgnore{};          ///< flags which X bins to ignore
  std::array<std::array<float, param::NPadRows>, param::NSectors2> mValidFracXBins{}; ///< for each sector for each X-bin the fraction of validated voxels
  std::vector<std::vector<bres_t>> mVoxelResults{};                                   ///< results per sector and per voxel for 3-D distortions
};

//_____________________________________________________
inline unsigned short TrackResiduals::getGlbVoxBin(std::array<unsigned char, param::VoxDim> bvox) const
{
  return bvox[param::VoxX] + bvox[param::VoxF] * mNXBins + bvox[param::VoxZ] * mNXBins * mNY2XBins;
}

//_____________________________________________________
inline unsigned short TrackResiduals::getGlbVoxBin(int ix, int ip, int iz) const
{
  std::array<unsigned char, param::VoxDim> bvox;
  bvox[param::VoxX] = ix;
  bvox[param::VoxF] = ip;
  bvox[param::VoxZ] = iz;
  return getGlbVoxBin(bvox);
}

//_____________________________________________________
inline void TrackResiduals::getVoxelCoordinates(int isec, int ix, int ip, int iz, float& x, float& p, float& z) const
{
  x = getX(ix);
  p = getY2X(ix, ip);
  z = getZ(iz);
  if (isec >= param::NSectors) {
    z = -z;
  }
}

//_____________________________________________________
inline float TrackResiduals::getX(int i) const
{
  return mUniformBins[param::VoxX] ? param::MinX + (i + 0.5) * mDX : param::RowX[i];
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
    bz = 0;
  }
  return (bz < mNZBins) ? bz : mNZBins - 1;
}

} // namespace calib

} // namespace o2
#endif
