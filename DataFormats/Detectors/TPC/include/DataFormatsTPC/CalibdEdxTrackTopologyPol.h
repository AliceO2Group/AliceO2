// Copyright 2019-2020 CERN and copyright holders of ALICE O2.
// See https://alice-o2.web.cern.ch/copyright for details of the copyright holders.
// All rights not expressly granted are reserved.
//
// This software is distributed under the terms of the GNU General Public
// License v3 (GPL Version 3), copied verbatim in the file "COPYING".
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

/// \file CalibdEdxTrackTopologyPol.h
/// \author Matthias Kleiner <mkleiner@ikf.uni-frankfurt.de>

#ifndef ALICEO2_TPC_CALIBDEDXTRACKTOPOLOGYPOL_H_
#define ALICEO2_TPC_CALIBDEDXTRACKTOPOLOGYPOL_H_

#include "GPUCommonRtypes.h"
#include "MultivariatePolynomial.h"
#include "GPUCommonDef.h"
#include "FlatObject.h"
#include "DataFormatsTPC/Defs.h"
#ifndef GPUCA_ALIGPUCODE
#include <string_view>
#endif

namespace o2::tpc
{

#if !defined(GPUCA_GPUCODE)
/// simple struct to enable writing the MultivariatePolynomialCT to file
struct CalibdEdxTrackTopologyPolContainer {
  /// constructor
  /// \param maxTheta maximum tanTheta for which the polynomials are valid
  /// \param maxSinPhi maximum sinPhi for which the polynomials are valid
  /// \param thresholdMin minimum zero supression threshold for which the polynomials are valid
  /// \param thresholdMax maximum zero supression threshold for which the polynomials are valid
  CalibdEdxTrackTopologyPolContainer(const float maxTheta, const float maxSinPhi, const float thresholdMin, const float thresholdMax) : mMaxTanTheta{maxTheta}, mMaxSinPhi{maxSinPhi}, mThresholdMin{thresholdMin}, mThresholdMax{thresholdMax} {};

  /// for ROOT I/O
  CalibdEdxTrackTopologyPolContainer() = default;

  std::vector<gpu::MultivariatePolynomialContainer> mCalibPols{}; ///< parameters of the polynomial
  float mMaxTanTheta{2.f};                                        ///< max tanTheta for which the correction is stored
  float mMaxSinPhi{0.99f};                                        ///< max snp for which the correction is stored
  float mThresholdMin{2.5f};                                      ///< min zero supression for which the correction is stored
  float mThresholdMax{5};                                         ///< max zero supression for which the correction is stored
};
#endif

/// calibration class for the track topology correction of the dE/dx using multvariate polynomials
class CalibdEdxTrackTopologyPol : public o2::gpu::FlatObject
{
 public:
#if !defined(GPUCA_GPUCODE) && !defined(GPUCA_STANDALONE)
  /// constructor constructs an object Initialized from file
  /// \param fileName name of the input file containing the object
  /// \parma name name of the object
  CalibdEdxTrackTopologyPol(std::string_view fileName, std::string_view name = "CalibdEdxTrackTopologyPol") { loadFromFile(fileName.data(), name.data()); };
#endif

  /// Default constructor: creates an empty uninitialized object
  CalibdEdxTrackTopologyPol() CON_DEFAULT;

  /// destructor
  ~CalibdEdxTrackTopologyPol() CON_DEFAULT;

  /// \return returns the track topology correction
  /// \param region region of the TPC
  /// \param charge correction for maximum or total charge
  /// \param tanTheta tan of local inclination angle theta
  /// \param sinPhi track parameter sinphi
  /// \param z z position of the cluster
  /// \param relPad absolute relative pad position of the track
  /// \param relTime relative time position of the track
  GPUd() float getCorrection(const int region, const ChargeType charge, const float tanTheta, const float sinPhi, const float z, const float relPad, const float relTime, const float threshold = 0) const
  {
    const float x[]{tanTheta, sinPhi, z, relPad, relTime, threshold};
    const float corr = (charge == ChargeType::Tot) ? mCalibPolsqTot[region].eval(x) : mCalibPolsqMax[region].eval(x);
    return corr;
  }

  /// \return returns the track topology correction
  /// \param region region of the TPC
  /// \param charge correction for maximum or total charge
  /// \param x coordinates where the correction is evaluated
  GPUd() float getCorrection(const int region, const ChargeType charge, const float x[/*inpXdim*/]) const { return (charge == ChargeType::Tot) ? mCalibPolsqTot[region].eval(x) : mCalibPolsqMax[region].eval(x); }

  /// returns the maximum tanTheta for which the polynomials are valid
  GPUd() float getMaxTanTheta() const { return mMaxTanTheta; };

  /// returns the maximum sinPhi for which the polynomials are valid
  GPUd() float getMaxSinPhi() const { return mMaxSinPhi; };

  /// returns the minimum zero supression threshold for which the polynomials are valid
  GPUd() float getMinThreshold() const { return mThresholdMin; };

  /// returns the maximum zero supression threshold for which the polynomials are valid
  GPUd() float getMaxThreshold() const { return mThresholdMax; };

#if !defined(GPUCA_GPUCODE)
  /// \return returns polynomial for qTot
  /// \param region region of the TPC
  const auto& getPolyqTot(const int region) const { return mCalibPolsqTot[region]; }

  /// \return returns polynomial for qMax
  /// \param region region of the TPC
  const auto& getPolyqMax(const int region) const { return mCalibPolsqMax[region]; }

  /// set the maximum tanTheta for which the polynomials are valid
  /// \param maxTanTheta maximum tanTheta
  void setMaxTanTheta(const float maxTanTheta) { mMaxTanTheta = maxTanTheta; };

  /// set the maximum sinPhi for which the polynomials are valid
  /// \param maxSinPhi maximum sinPhi
  void setMaxSinPhi(const float maxSinPhi) { mMaxSinPhi = maxSinPhi; };

  /// set the the minimum zero supression threshold for which the polynomials are valid
  /// \param thresholdMin minimum threshold
  void setMinThreshold(const float thresholdMin) { mThresholdMin = thresholdMin; };

  /// set the the maximum zero supression threshold for which the polynomials are valid
  /// \param thresholdMax maximum threshold
  void setMaxThreshold(const float thresholdMax) { mThresholdMax = thresholdMax; };

#ifndef GPUCA_STANDALONE
  /// write a class object to the file
  /// \param outf file where the object will be written to
  /// \param name name of the object in the output file
  void writeToFile(TFile& outf, const char* name) const;

  /// init parameters from CalibdEdxTrackTopologyPolContainer
  /// \param container container for the members
  void setFromContainer(const CalibdEdxTrackTopologyPolContainer& container);

  /// load members from a file
  /// \param fileName file where the object will be read from
  /// \param name name of the object in the output file
  void loadFromFile(const char* fileName, const char* name);

  /// sets the polynomials from an input file. The names of the objects have to be the same as in the getPolyName() function
  /// \param inpf file where the polynomials are stored
  void setPolynomialsFromFile(TFile& inpf);
#endif

  /// \return returns the name of the polynomial object which can be read in with the setPolynomialsFromFile() function
  /// \param region region of the TPC
  /// \param charge correction for maximum or total charge
  static std::string getPolyName(const int region, const ChargeType charge);
#endif

/// ========== FlatObject functionality, see FlatObject class for description  =================
#if !defined(GPUCA_GPUCODE)
  /// cloning a container object (use newFlatBufferPtr=nullptr for simple copy)
  void cloneFromObject(const CalibdEdxTrackTopologyPol& obj, char* newFlatBufferPtr);

  /// move flat buffer to new location
  /// \param newBufferPtr new buffer location
  void moveBufferTo(char* newBufferPtr);
#endif

  /// destroy the object (release internal flat buffer)
  void destroy();

  /// set location of external flat buffer
  void setActualBufferAddress(char* actualFlatBufferPtr);

  /// set future location of the flat buffer
  void setFutureBufferAddress(char* futureFlatBufferPtr);
  /// ================================================================================================

 private:
  constexpr static int FFits{10};                              ///< total number of fits: 10 regions * 2 charge types
  o2::gpu::MultivariatePolynomial<5, 4> mCalibPolsqTot[FFits]; ///< polynomial objects storage for the polynomials for qTot
  o2::gpu::MultivariatePolynomial<5, 4> mCalibPolsqMax[FFits]; ///< polynomial objects storage for the polynomials for qMax
  float mMaxTanTheta{2.f};                                     ///< max tanTheta for which the correction is stored
  float mMaxSinPhi{0.99f};                                     ///< max snp for which the correction is stored
  float mThresholdMin{2.5f};                                   ///< min zero supression for which the correction is stored
  float mThresholdMax{5};                                      ///< max zero supression for which the correction is stored

#if !defined(GPUCA_GPUCODE)
  void construct();
#endif

  ClassDefNV(CalibdEdxTrackTopologyPol, 1);
};

} // namespace o2::tpc

#endif
