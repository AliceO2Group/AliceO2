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

/// \file CalibdEdxContainer.h
/// \brief Definition of container class for dE/dx corrections
/// \author Matthias Kleiner <mkleiner@ikf.uni-frankfurt.de>

#ifndef ALICEO2_TPC_CALIBDEDXCONTAINER_H_
#define ALICEO2_TPC_CALIBDEDXCONTAINER_H_

#include "GPUCommonRtypes.h"
#include "GPUCommonDef.h"
#include "GPUCommonMath.h"
#include "DataFormatsTPC/CalibdEdxCorrection.h"
#include "DataFormatsTPC/CalibdEdxTrackTopologyPol.h"
#include "DataFormatsTPC/CalibdEdxTrackTopologySpline.h"
#include "FlatObject.h"
#include "TPCPadGainCalib.h"

#ifndef GPUCA_ALIGPUCODE
#include <string_view>
#endif

namespace o2::tpc
{

///
/// This container class contains all necessary corrections for the dE/dx
/// Currently it holds the residual dE/dx correction and the track topology correction, which can be either provided by 2D-splines or by 5D-polynomials.
/// To be able to correctly use the flat buffer functionality only one of the topology corrections should be set: 2D-splines or by 5D-polynomials!
///   The track topology corrections are set as a nullptr for default as a member.
///   By loading only one of them memory overhead for the classes are avoided and appropriate flat buffer sizes are used.
///
class CalibdEdxContainer : public o2::gpu::FlatObject
{
 public:
  /// Default constructor: creates an empty uninitialized object
  CalibdEdxContainer() CON_DEFAULT;

  /// Copy constructor: disabled to avoid ambiguity. Use cloneFromObject() instead
  CalibdEdxContainer(const CalibdEdxContainer&) CON_DELETE;

  /// Assignment operator: disabled to avoid ambiguity. Use cloneFromObject() instead
  CalibdEdxContainer& operator=(const CalibdEdxContainer&) CON_DELETE;

  /// Destructor
  ~CalibdEdxContainer() CON_DEFAULT;

  /// \return returns the topology correction for the cluster charge
  /// \param region region of the TPC
  /// \param charge type of the charge (qMax or qTot)
  /// \param tanTheta local inclination angle tanTheta
  /// \param sinPhi snp track parameter
  /// \param z z position
  /// \param relPad relative pad position of the cluster
  /// \param relTime relative time position of the cluster
  GPUd() float getTopologyCorrection(const int region, const ChargeType charge, const float tanTheta, const float sinPhi, const float z, const float relPad, const float relTime, const float threshold = 0) const
  {
    return mCalibTrackTopologyPol ? mCalibTrackTopologyPol->getCorrection(region, charge, tanTheta, sinPhi, z, relPad, relTime, threshold) : (mCalibTrackTopologySpline ? mCalibTrackTopologySpline->getCorrection(region, charge, tanTheta, sinPhi, z) : getDefaultTopologyCorrection(tanTheta, sinPhi));
  }

  /// \return returns the topology correction for the cluster charge
  /// \param region region of the TPC
  /// \param charge type of the charge (qMax or qTot)
  /// \param x coordinates where the correction is evaluated
  GPUd() float getTopologyCorrection(const int region, const ChargeType charge, const float x[]) const
  {
    return mCalibTrackTopologyPol ? mCalibTrackTopologyPol->getCorrection(region, charge, x) : (mCalibTrackTopologySpline ? mCalibTrackTopologySpline->getCorrection(region, charge, x) : getDefaultTopologyCorrection(x[0], x[1]));
  }

  /// \return returns analytical default correction
  /// Correction corresponds to: "sqrt((dz/dx)^2 + (dy/dx)^2 + (dx/dx)^2) / padlength" simple track length correction (ToDo add division by pad length)
  GPUd() float getDefaultTopologyCorrection(const float tanTheta, const float sinPhi) const { return gpu::CAMath::Sqrt(tanTheta * tanTheta + 1 / (1 - sinPhi * sinPhi)); }

  /// \return returns maximum tanTheta for which the topology correction is valid
  GPUd() float getMaxTanThetaTopologyCorrection() const { return mCalibTrackTopologyPol ? mCalibTrackTopologyPol->getMaxTanTheta() : (mCalibTrackTopologySpline ? mCalibTrackTopologySpline->getMaxTanTheta() : 2); }

  /// \return returns maximum sinPhi for which the topology correction is valid
  GPUd() float getMaxSinPhiTopologyCorrection() const { return mCalibTrackTopologyPol ? mCalibTrackTopologyPol->getMaxSinPhi() : (mCalibTrackTopologySpline ? mCalibTrackTopologySpline->getMaxSinPhi() : 1); }

#if !defined(GPUCA_GPUCODE)
  /// \returns the minimum zero supression threshold for which the track topology correction is valid
  float getMinZeroSupresssionThreshold() const;

  /// \returns the maximum zero supression threshold for which the track topology correction is valid
  float getMaxZeroSupresssionThreshold() const;
#endif

  /// \return returns zero supression threshold
  /// \param sector tpc sector
  /// \param row global pad row
  GPUd() float getZeroSupressionThreshold(const int sector, const gpu::tpccf::Row row, const gpu::tpccf::Pad pad) const { return mThresholdMap.getGainCorrection(sector, row, pad); }

  /// \return returns the residual dE/dx correction for the cluster charge
  /// \param stack ID of the GEM stack
  /// \param charge type of the charge (qMax or qTot)
  /// \param z z position
  /// \param tgl tracking parameter tgl
  GPUd() float getResidualCorrection(const StackID& stack, const ChargeType charge, const float z = 0, const float tgl = 0) const { return mCalibResidualdEdx.getCorrection(stack, charge, z, tgl); }

  /// ========== FlatObject functionality, see FlatObject class for description  =================
#if !defined(GPUCA_GPUCODE)
  /// cloning a container object (use newFlatBufferPtr=nullptr for simple copy)
  void cloneFromObject(const CalibdEdxContainer& obj, char* newFlatBufferPtr);

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

#if !defined(GPUCA_GPUCODE) && !defined(GPUCA_STANDALONE)
  // loading the polynomial track topology correction from a file
  /// \param fileName input file containing the correction
  void loadPolTopologyCorrectionFromFile(std::string_view fileName);

  // loading the spline track topology correction from a file
  /// \param fileName input file containing the correction
  void loadSplineTopologyCorrectionFromFile(std::string_view fileName);

  // loading the residual dE/dx correction from a file
  /// \param fileName input file containing the correction
  void loadResidualCorrectionFromFile(std::string_view fileName) { mCalibResidualdEdx.loadFromFile(fileName); }

  // loading the zero supression threshold map from a file
  /// \param fileName input file containing the CalDet map
  void loadZeroSupresssionThresholdFromFile(std::string_view fileName, std::string_view objName, const float minCorrectionFactor, const float maxCorrectionFactor);

  // loading the zero supression threshold map from a file
  /// \param fileName input file containing the CalDet map
  void setZeroSupresssionThreshold(const CalDet<float>& thresholdMap) { setZeroSupresssionThreshold(thresholdMap, getMinZeroSupresssionThreshold(), getMaxZeroSupresssionThreshold()); }

  // loading the zero supression threshold map from a file
  /// \param fileName input file containing the CalDet map
  void setZeroSupresssionThreshold(const CalDet<float>& thresholdMap, const float minCorrectionFactor, const float maxCorrectionFactor);

  /// setting default zero supression threshold map (all values are set to getMinZeroSupresssionThreshold())
  /// \param fileName input file containing the CalDet map
  void setDefaultZeroSupresssionThreshold();
#endif // !GPUCA_GPUCODE

 private:
  CalibdEdxTrackTopologySpline* mCalibTrackTopologySpline{nullptr}; ///< calibration for the track topology correction (splines)
  CalibdEdxTrackTopologyPol* mCalibTrackTopologyPol{nullptr};       ///< calibration for the track topology correction (polynomial)
  o2::gpu::TPCPadGainCalib mThresholdMap{};                         ///< calibration object containing the zero supression threshold map
  CalibdEdxCorrection mCalibResidualdEdx{};                         ///< calibration for the residual dE/dx correction

#if !defined(GPUCA_GPUCODE)
  template <class Type>
  std::size_t sizeOfCalibdEdxTrackTopologyObj() const
  {
    return alignSize(sizeof(Type), FlatObject::getClassAlignmentBytes());
  }

  template <class Type>
  void loadTopologyCorrectionFromFile(std::string_view fileName, Type*& obj);
#endif

  template <class Type>
  void setActualBufferAddress(Type*& obj);

  template <class Type>
  void setFutureBufferAddress(Type*& obj, char* futureFlatBufferPtr);

#if !defined(GPUCA_GPUCODE)
  template <class Type>
  void cloneFromObject(Type*& obj, const Type* objOld, char* newFlatBufferPtr, const char* oldFlatBufferPtr);
#endif

#ifndef GPUCA_ALIROOT_LIB
  ClassDefNV(CalibdEdxContainer, 1);
#endif
};

} // namespace o2::tpc

#endif
