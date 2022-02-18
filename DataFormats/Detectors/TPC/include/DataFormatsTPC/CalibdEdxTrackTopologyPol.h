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
  CalibdEdxTrackTopologyPolContainer(const float maxTheta, const float maxSinPhi, const float thresholdMin, const float thresholdMax, const float qTotMin, const float qTotMax) : mMaxTanTheta{maxTheta}, mMaxSinPhi{maxSinPhi}, mThresholdMin{thresholdMin}, mThresholdMax{thresholdMax}, mQTotMin{qTotMin}, mQTotMax{qTotMax} {};

  /// for ROOT I/O
  CalibdEdxTrackTopologyPolContainer() = default;

  std::vector<gpu::MultivariatePolynomialContainer> mCalibPols{}; ///< parameters of the polynomial
  float mMaxTanTheta{2.f};                                        ///< max tanTheta for which the correction is stored
  float mMaxSinPhi{0.99f};                                        ///< max snp for which the correction is stored
  float mThresholdMin{2.5f};                                      ///< min zero supression for which the correction is stored
  float mThresholdMax{5};                                         ///< max zero supression for which the correction is stored
  float mQTotMin{30};                                             ///< minimum qTot for which the correction is valid
  float mQTotMax{200};                                            ///< maximum qTot for which the correction is valid
  std::vector<float> mScalingFactorsqTot{};                       ///< value which is used to scale the result of the polynomial for qTot (can be used for normalization)
  std::vector<float> mScalingFactorsqMax{};                       ///< value which is used to scale the result of the polynomial for qMax (can be used for normalization)
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
  /// \param chargeT correction for maximum or total charge
  /// \param tanTheta tan of local inclination angle theta
  /// \param sinPhi track parameter sinphi
  /// \param z z position of the cluster
  /// \param relPad absolute relative pad position of the track
  /// \param relTime relative time position of the track
  /// \param threshold zero supression threshold
  /// \param charge charge of the cluster
  GPUd() float getCorrection(const int region, const ChargeType chargeT, const float tanTheta, const float sinPhi, const float z, const float relPad, const float relTime, const float threshold, const float charge) const
  {
    const float corr = chargeT == ChargeType::Tot ? getCorrectionqTot(region, tanTheta, sinPhi, z, threshold, charge) : getCorrectionqMax(region, tanTheta, sinPhi, z, relPad, relTime);
    return corr;
  }

  /// \return returns the track topology correction for qTot
  /// \param region region of the TPC
  /// \param tanTheta tan of local inclination angle theta
  /// \param sinPhi track parameter sinphi
  /// \param z z position of the cluster
  /// \param threshold zero supression threshold
  /// \param charge charge of the cluster
  GPUd() float getCorrectionqTot(const int region, const float tanTheta, const float sinPhi, const float z, const float threshold, const float charge) const
  {
    const float x[]{z, tanTheta, sinPhi, threshold, charge};
    const float corr = mScalingFactorsqTot[region] * mCalibPolsqTot[region].eval(x);
    return corr;
  }

  /// \return returns the track topology correction for qMax
  /// \param region region of the TPC
  /// \param tanTheta tan of local inclination angle theta
  /// \param sinPhi track parameter sinphi
  /// \param z z position of the cluster
  /// \param relPad absolute relative pad position of the track
  /// \param relTime relative time position of the track
  GPUd() float getCorrectionqMax(const int region, const float tanTheta, const float sinPhi, const float z, const float relPad, const float relTime) const
  {
    const float x[]{z, tanTheta, sinPhi, relPad, relTime};
    const float corr = mScalingFactorsqMax[region] * mCalibPolsqMax[region].eval(x);
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

  /// \return returns the the minimum qTot for which the polynomials are valid
  GPUd() float getMinqTot() const { return mQTotMin; };

  /// \return returns the the maximum qTot for which the polynomials are valid
  GPUd() float getMaxqTot() const { return mQTotMax; };

  /// \return returns the the scaling factors for the polynomials for qTot
  /// \param region region of the scaling factor
  GPUd() float getScalingFactorqTot(const int region) const { return mScalingFactorsqTot[region]; };

  /// \return returns the the scaling factors for the polynomials for qMax
  /// \param region region of the scaling factor
  GPUd() float getScalingFactorqMax(const int region) const { return mScalingFactorsqMax[region]; };

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
  /// set the the minimum qTot for which the polynomials are valid
  /// \param qTotMin minimum qTot
  void setMinqTot(const float qTotMin) { mQTotMin = qTotMin; };

  /// set the the maximum qTot for which the polynomials are valid
  /// \param qTotMax maximum qTot
  void setMaxTot(const float qTotMax) { mQTotMax = qTotMax; };

  /// set the the scaling factors for the polynomials for qTot
  /// \param factor scaling factor
  /// \param region region of the scaling factor
  void setScalingFactorqTot(const float factor, const int region) { mScalingFactorsqTot[region] = factor; };

  /// set the the scaling factors for the polynomials for qMax
  /// \param factor scaling factor
  /// \param region region of the scaling factor
  void setScalingFactorqMax(const float factor, const int region) { mScalingFactorsqMax[region] = factor; };

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
  constexpr static int FFits{10};                                 ///< total number of fits: 10 regions * 2 charge types
  o2::gpu::MultivariatePolynomial<5, 4> mCalibPolsqTot[FFits];    ///< polynomial objects storage for the polynomials for qTot
  o2::gpu::MultivariatePolynomial<5, 4> mCalibPolsqMax[FFits];    ///< polynomial objects storage for the polynomials for qMax
  float mMaxTanTheta{2.f};                                        ///< max tanTheta for which the correction is stored
  float mMaxSinPhi{0.99f};                                        ///< max snp for which the correction is stored
  float mThresholdMin{2.5f};                                      ///< min zero supression for which the correction is stored
  float mThresholdMax{5};                                         ///< max zero supression for which the correction is stored
  float mQTotMin{30};                                             ///< minimum qTot for which the correction is valid
  float mQTotMax{200};                                            ///< maximum qTot for which the correction is valid
  float mScalingFactorsqTot[FFits]{1, 1, 1, 1, 1, 1, 1, 1, 1, 1}; ///< value which is used to scale the result of the polynomial for qTot (can be used for normalization)
  float mScalingFactorsqMax[FFits]{1, 1, 1, 1, 1, 1, 1, 1, 1, 1}; ///< value which is used to scale the result of the polynomial for qMax (can be used for normalization)

#if !defined(GPUCA_GPUCODE)
  void construct();
#endif

  ClassDefNV(CalibdEdxTrackTopologyPol, 1);
};

} // namespace o2::tpc

#endif
