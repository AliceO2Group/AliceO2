// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

/// \file  dEdxCalibrationSplines.h
/// \brief Definition of dEdxCalibrationSplines class
///
/// \author  Matthias Kleiner <matthias.kleiner@cern.ch>

#ifndef TPCdEdxCalibrationSplines_H
#define TPCdEdxCalibrationSplines_H

#include "FlatObject.h"
#include "Spline2D.h"

#if !defined(GPUCA_GPUCODE) && !defined(GPUCA_STANDALONE) // code invisible on GPU and in the standalone compilation
#include "Rtypes.h"                                       // for ClassDefNV
#include <fmt/format.h>
#endif

namespace GPUCA_NAMESPACE
{
namespace gpu
{

///
/// The dEdxCalibrationSplines class represents the calibration of the dEdx of mostly geometrical effects
///
/// The splines which are used to correct the dE/dx for detector effects like different pad sizes and
/// track topologies (local dip angle: z angle - dz/dx and diffusion from electron drift) are created by
/// the Random Forest ML algorithm. The training data is created by the O2 simulation with artificial tracks
/// (no Landau distribution, constant dE/dx, fixed GEM amplification of 2000).
/// For each region and for qMax/qTot an individual spline is created which leads to a total of 20 splines.
/// After applying the correction with the splines the dE/dx for qMax and qTot should be normalized such that
/// the MIP position is roughly equal for both.
/// The absolute return values of the splines is arbitrary (only the relative differences between the values matter)
///
/// It is planned to make further topological corrections also with the splines: relative pad and time position
/// of a track, local inclination angle: y angle = dy/dx (improves dE/dx at low momenta) and threshold effects.
///
///
/// Definitions:
///
/// angleZ
/// snp2 = mP[2] * mP[2]
/// tgl2 = mP[3] * mP[3]
/// sec2 = 1.f / (1.f - snp2)
/// angleZ = sqrt(tgl2 * sec2)
///
/// angleY
/// angleY = sqrt( snp2 * sec2 )
///
/// relative pad position
/// relPadPos = COG_Pad - int(COG_Pad + 0.5);
/// -0.5<relPadPos<0.5
///
/// relative time position
/// relTimePos = COG_Time - int(COG_Time + 0.5);
/// -0.5<relTimePos<0.5
///

class TPCdEdxCalibrationSplines : public FlatObject
{
 public:
  typedef Spline2D<float, 1> SplineType;

  /// _____________  Constructors / destructors __________________________

#if !defined(GPUCA_GPUCODE)
  /// Default constructor
  TPCdEdxCalibrationSplines();

  /// constructor with initialization of the splines from file
  /// \param dEdxSplinesFile path to root file containing the splines
  TPCdEdxCalibrationSplines(const char* dEdxSplinesFile);

  /// Copy constructor
  TPCdEdxCalibrationSplines(const TPCdEdxCalibrationSplines&);

  /// Assignment operator
  TPCdEdxCalibrationSplines& operator=(const TPCdEdxCalibrationSplines&);

  void recreate(int nKnotsU1[], int nKnotsU2[]);
#else
  /// Disable constructors for the GPU implementation

  TPCdEdxCalibrationSplines() CON_DELETE;
  TPCdEdxCalibrationSplines(const TPCdEdxCalibrationSplines&) CON_DELETE;
  TPCdEdxCalibrationSplines& operator=(const TPCdEdxCalibrationSplines&) CON_DELETE;
#endif

  /// Destructor
  ~TPCdEdxCalibrationSplines() CON_DEFAULT;

  /// _____________  FlatObject functionality, see FlatObject class for description  ____________

  using FlatObject::getBufferAlignmentBytes;
  using FlatObject::getClassAlignmentBytes;

#if !defined(GPUCA_GPUCODE)
  void cloneFromObject(const TPCdEdxCalibrationSplines& obj, char* newFlatBufferPtr);
  void moveBufferTo(char* newBufferPtr);
#endif

  using FlatObject::releaseInternalBuffer;

  void destroy();
  void setActualBufferAddress(char* actualFlatBufferPtr);
  void setFutureBufferAddress(char* futureFlatBufferPtr);

  /// ______________

  /// Gives pointer to a spline
  GPUd() const SplineType& getSpline(int chargeType, int region) const;

#if !defined(GPUCA_GPUCODE) && !defined(GPUCA_STANDALONE)
  /// sets the splines from an input file
  void setSplinesFromFile(TFile& inpf);

  /// set default splines: the return value of the splines will be 1 (no correction will be applied)
  void setDefaultSplines();
#endif

  /// returns the number of splines stored in the calibration object
  GPUd() unsigned int getFSplines() const
  {
    return FSplines;
  };

  /// \param splineInd index of the spline (region)
  /// \param angleZ local dip angle: z angle - dz/dx
  /// \param z drift length
  /// \return returns the spline (correction) value for qMax
  GPUd() float interpolateqMax(const int splineInd, const float angleZ, const float z) const
  {
    return mCalibSplinesqMax[splineInd].interpolate(angleZ, z);
  };

  /// \param splineInd index of the spline (region)
  /// \param angleZ local dip angle: z angle - dz/dx
  /// \param z drift length
  /// \return returns the spline (correction) value for qMax
  GPUd() float interpolateqTot(const int splineInd, const float angleZ, const float z) const
  {
    return mCalibSplinesqTot[splineInd].interpolate(angleZ, z);
  };

  /// \param splineInd index of the spline (region)
  /// \return returns the spline for qMax
  GPUd() SplineType& getSplineqMax(const int splineInd)
  {
    return mCalibSplinesqMax[splineInd];
  };

  /// \param splineInd index of the spline (region)
  /// \return returns the spline for qTot
  GPUd() SplineType& getSplineqTot(const int splineInd)
  {
    return mCalibSplinesqTot[splineInd];
  };

    /// _______________  IO   ________________________
#if !defined(GPUCA_GPUCODE) && !defined(GPUCA_STANDALONE)
  /// write a class object to the file
  int writeToFile(TFile& outf, const char* name);

  /// read a class object from the file
  static TPCdEdxCalibrationSplines* readFromFile(TFile& inpf, const char* name);
#endif

 private:
  constexpr static unsigned int FSplines = 10; ///< number of splines stored for each type
  SplineType mCalibSplinesqMax[FSplines];      ///< spline objects storage for the splines for qMax
  SplineType mCalibSplinesqTot[FSplines];      ///< spline objects storage for the splines for qTot

  ClassDefNV(TPCdEdxCalibrationSplines, 1);
};

#if !defined(GPUCA_GPUCODE) && !defined(GPUCA_STANDALONE)

inline void TPCdEdxCalibrationSplines::setSplinesFromFile(TFile& inpf)
{
  FlatObject::startConstruction();

  int buffSize = 0;
  int offsets1[FSplines];
  int offsets2[FSplines];

  for (unsigned int ireg = 0; ireg < FSplines; ++ireg) {
    std::string splinename = fmt::format("spline_qMax_region{}", ireg);
    o2::gpu::Spline2D<float, 1>* splineTmpqMax = o2::gpu::Spline2D<float, 1>::readFromFile(inpf, splinename.data());
    mCalibSplinesqMax[ireg] = *splineTmpqMax;
    buffSize = alignSize(buffSize, mCalibSplinesqMax[ireg].getBufferAlignmentBytes());
    offsets1[ireg] = buffSize;
    buffSize += mCalibSplinesqMax[ireg].getFlatBufferSize();
  }

  for (unsigned int ireg = 0; ireg < FSplines; ++ireg) {
    std::string splinename = fmt::format("spline_qTot_region{}", ireg);
    o2::gpu::Spline2D<float, 1>* splineTmpqTot = o2::gpu::Spline2D<float, 1>::readFromFile(inpf, splinename.data());
    mCalibSplinesqTot[ireg] = *splineTmpqTot;
    buffSize = alignSize(buffSize, mCalibSplinesqTot[ireg].getBufferAlignmentBytes());
    offsets2[ireg] = buffSize;
    buffSize += mCalibSplinesqTot[ireg].getFlatBufferSize();
  }

  FlatObject::finishConstruction(buffSize);

  for (unsigned int i = 0; i < FSplines; i++) {
    mCalibSplinesqMax[i].moveBufferTo(mFlatBufferPtr + offsets1[i]);
  }
  for (unsigned int i = 0; i < FSplines; i++) {
    mCalibSplinesqTot[i].moveBufferTo(mFlatBufferPtr + offsets2[i]);
  }
}

#endif

} // namespace gpu
} // namespace GPUCA_NAMESPACE

#endif
