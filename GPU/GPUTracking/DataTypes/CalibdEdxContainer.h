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
#include "CalibdEdxTrackTopologyPol.h"
#include "CalibdEdxTrackTopologySpline.h"
#include "FlatObject.h"
#include "TPCPadGainCalib.h"

#ifndef GPUCA_ALIGPUCODE
#include <string_view>
#endif

namespace o2::tpc
{

/// flags to set which corrections will be loaded from the CCDB
enum class CalibsdEdx : unsigned short {
  CalTopologySpline = 1 << 0,  ///< flag for a topology correction using splines
  CalTopologyPol = 1 << 1,     ///< flag for a topology correction using polynomials
  CalThresholdMap = 1 << 2,    ///< flag for using threshold map
  CalGainMap = 1 << 3,         ///< flag for using the gain map to get the correct cluster charge
  CalResidualGainMap = 1 << 4, ///< flag for applying residual gain map
  CalTimeGain = 1 << 5,        ///< flag for residual dE/dx time dependent gain correction
};

inline CalibsdEdx operator|(CalibsdEdx a, CalibsdEdx b) { return static_cast<CalibsdEdx>(static_cast<int>(a) | static_cast<int>(b)); }

inline CalibsdEdx operator&(CalibsdEdx a, CalibsdEdx b) { return static_cast<CalibsdEdx>(static_cast<int>(a) & static_cast<int>(b)); }

inline CalibsdEdx operator~(CalibsdEdx a) { return static_cast<CalibsdEdx>(~static_cast<int>(a)); }

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
  /// \param chargeT type of the charge (qMax or qTot)
  /// \param tanTheta local inclination angle tanTheta
  /// \param sinPhi snp track parameter
  /// \param z z position
  /// \param relPad relative pad position of the cluster
  /// \param relTime relative time position of the cluster
  /// \param threshold zero supression threshold
  /// \param charge charge of the cluster
  GPUd() float getTopologyCorrection(const int region, const ChargeType chargeT, const float tanTheta, const float sinPhi, const float z, const float relPad, const float relTime, const float threshold, const float charge) const
  {
    return mCalibTrackTopologyPol ? mCalibTrackTopologyPol->getCorrection(region, chargeT, tanTheta, sinPhi, z, relPad, relTime, threshold, charge) : (mCalibTrackTopologySpline ? mCalibTrackTopologySpline->getCorrection(region, chargeT, tanTheta, sinPhi, z) : getDefaultTopologyCorrection(tanTheta, sinPhi));
  }

  /// \return returns the topology correction for the cluster charge
  /// \param region region of the TPC
  /// \param chargeT type of the charge (qMax or qTot)
  /// \param x coordinates where the correction is evaluated
  GPUd() float getTopologyCorrection(const int region, const ChargeType chargeT, const float x[]) const
  {
    return mCalibTrackTopologyPol ? mCalibTrackTopologyPol->getCorrection(region, chargeT, x) : (mCalibTrackTopologySpline ? mCalibTrackTopologySpline->getCorrection(region, chargeT, x) : getDefaultTopologyCorrection(x[0], x[1]));
  }

  /// \return returns analytical default correction
  /// Correction corresponds to: "sqrt((dz/dx)^2 + (dy/dx)^2 + (dx/dx)^2) / padlength" simple track length correction (ToDo add division by pad length)
  GPUd() float getDefaultTopologyCorrection(const float tanTheta, const float sinPhi) const { return gpu::CAMath::Sqrt(tanTheta * tanTheta + 1 / (1 - sinPhi * sinPhi)); }

  /// \return returns maximum tanTheta for which the topology correction is valid
  GPUd() float getMaxTanThetaTopologyCorrection() const { return mCalibTrackTopologyPol ? mCalibTrackTopologyPol->getMaxTanTheta() : (mCalibTrackTopologySpline ? mCalibTrackTopologySpline->getMaxTanTheta() : 2); }

  /// \return returns maximum sinPhi for which the topology correction is valid
  GPUd() float getMaxSinPhiTopologyCorrection() const { return mCalibTrackTopologyPol ? mCalibTrackTopologyPol->getMaxSinPhi() : (mCalibTrackTopologySpline ? mCalibTrackTopologySpline->getMaxSinPhi() : 1); }

  /// \return returns the the minimum qTot for which the polynomials are valid
  GPUd() float getMinqTot() const { return mCalibTrackTopologyPol ? mCalibTrackTopologyPol->getMinqTot() : 0; };

  /// \return returns the the maximum qTot for which the polynomials are valid
  GPUd() float getMaxqTot() const { return mCalibTrackTopologyPol ? mCalibTrackTopologyPol->getMaxqTot() : 10000; };

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

  /// \return returns gain from the full gain map
  /// \param sector tpc sector
  /// \param row global pad row
  GPUd() float getGain(const int sector, const gpu::tpccf::Row row, const gpu::tpccf::Pad pad) const { return mGainMap.getGainCorrection(sector, row, pad); }

  /// \return returns gain from residual gain map
  /// \param sector tpc sector
  /// \param row global pad row
  GPUd() float getResidualGain(const int sector, const gpu::tpccf::Row row, const gpu::tpccf::Pad pad) const { return mGainMapResidual.getGainCorrection(sector, row, pad); }

  /// \return returns the residual dE/dx correction for the cluster charge
  /// \param stack ID of the GEM stack
  /// \param charge type of the charge (qMax or qTot)
  /// \param z z position
  /// \param tgl tracking parameter tgl
  GPUd() float getResidualCorrection(const StackID& stack, const ChargeType charge, const float tgl = 0, const float snp = 0) const { return mCalibResidualdEdx.getCorrection(stack, charge, tgl, snp); }

  /// \return returns if the full gain map will be used during the calculation of the dE/dx to correct the cluster charge
  GPUd() bool isUsageOfFullGainMap() const { return mApplyFullGainMap; }

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

  /// set the polynomial track topology
  /// \param calibTrackTopology polynomial track topology correction
  void setPolTopologyCorrection(const CalibdEdxTrackTopologyPol& calibTrackTopology);

  /// setting a default topology correction which just returns 1
  void setDefaultPolTopologyCorrection();

  /// set the spline track topology
  /// \param calibTrackTopology spline track topology correction
  void setSplineTopologyCorrection(const CalibdEdxTrackTopologySpline& calibTrackTopology);

  // loading the residual dE/dx correction from a file
  /// \param fileName input file containing the correction
  void loadResidualCorrectionFromFile(std::string_view fileName) { mCalibResidualdEdx.loadFromFile(fileName); }

  /// setting the residual dEdx correction
  /// \param residualCorr residual gain calibration object
  void setResidualCorrection(const CalibdEdxCorrection& residualCorr) { mCalibResidualdEdx = residualCorr; }

  // loading the zero supression threshold map from a file
  /// \param fileName input file containing the CalDet map
  void loadZeroSupresssionThresholdFromFile(std::string_view fileName, std::string_view objName, const float minCorrectionFactor, const float maxCorrectionFactor);

  /// setting the zero supression threshold map from a CalDet
  /// \param thresholdMap CalDet containing the zero supression threshold
  void setZeroSupresssionThreshold(const CalDet<float>& thresholdMap) { setZeroSupresssionThreshold(thresholdMap, getMinZeroSupresssionThreshold(), getMaxZeroSupresssionThreshold()); }

  /// setting the zero supression threshold map from a CalDet
  /// \param thresholdMap CalDet containing the zero supression threshold
  void setZeroSupresssionThreshold(const CalDet<float>& thresholdMap, const float minCorrectionFactor, const float maxCorrectionFactor);

  /// setting the gain map map from a CalDet
  void setGainMap(const CalDet<float>& gainMap, const float minGain, const float maxGain);

  /// setting the gain map map from a CalDet
  void setGainMapResidual(const CalDet<float>& gainMapResidual, const float minResidualGain = 0.7f, const float maxResidualGain = 1.3f);

  /// setting default zero supression threshold map (all values are set to getMinZeroSupresssionThreshold())
  /// \param fileName input file containing the CalDet map
  void setDefaultZeroSupresssionThreshold();

  /// returns status if the spline correction is set
  bool isTopologyCorrectionSplinesSet() const { return mCalibTrackTopologySpline ? true : false; }

  /// returns status if the polynomials correction is set
  bool isTopologyCorrectionPolynomialsSet() const { return mCalibTrackTopologyPol ? true : false; }

  /// set loading of a correction from CCDB
  /// \param calib calibration which will be loaded from CCDB
  void setCorrectionCCDB(const CalibsdEdx calib) { mCalibsLoad = calib | mCalibsLoad; }

  /// disable loading of a correction from CCDB
  /// \param calib calibration which will not be loaded from CCDB
  void disableCorrectionCCDB(const CalibsdEdx calib) { mCalibsLoad = ~calib & mCalibsLoad; }

  /// check if a correction will be loaded from CCDB
  /// \param calib calibration which will be loaded from CCDB
  bool isCorrectionCCDB(const CalibsdEdx calib) const { return ((mCalibsLoad & calib) == calib) ? true : false; }

  /// \param applyFullGainMap if set to true the cluster charge will be corrected with the full gain map
  void setUsageOfFullGainMap(const bool applyFullGainMap) { mApplyFullGainMap = applyFullGainMap; }
#endif // !GPUCA_GPUCODE

 private:
  CalibdEdxTrackTopologySpline* mCalibTrackTopologySpline{nullptr};                                                                                                     ///< calibration for the track topology correction (splines)
  CalibdEdxTrackTopologyPol* mCalibTrackTopologyPol{nullptr};                                                                                                           ///< calibration for the track topology correction (polynomial)
  o2::gpu::TPCPadGainCalib mThresholdMap{};                                                                                                                             ///< calibration object containing the zero supression threshold map
  o2::gpu::TPCPadGainCalib mGainMap{};                                                                                                                                  ///< calibration object containing the gain map
  o2::gpu::TPCPadGainCalib mGainMapResidual{};                                                                                                                          ///< calibration object containing the residual gain map
  CalibdEdxCorrection mCalibResidualdEdx{};                                                                                                                             ///< calibration for the residual dE/dx correction
  bool mApplyFullGainMap{false};                                                                                                                                        ///< if set to true the cluster charge will be corrected with the full gain map (when the gain map was not applied during the clusterizer)
  CalibsdEdx mCalibsLoad{CalibsdEdx::CalTopologyPol | CalibsdEdx::CalThresholdMap | CalibsdEdx::CalGainMap | CalibsdEdx::CalResidualGainMap | CalibsdEdx::CalTimeGain}; ///< flags to set which corrections will be loaded from the CCDB and used during calculation of the dE/dx

#if !defined(GPUCA_GPUCODE)
  template <class Type>
  std::size_t sizeOfCalibdEdxTrackTopologyObj() const
  {
    return alignSize(sizeof(Type), FlatObject::getClassAlignmentBytes());
  }

  template <class Type>
  void loadTopologyCorrectionFromFile(std::string_view fileName, Type*& obj);

  template <class Type>
  void setTopologyCorrection(const Type& calibTrackTopologyTmp, Type*& obj);
#endif

  template <class Type>
  void setActualBufferAddress(Type*& obj);

  template <class Type>
  void setFutureBufferAddress(Type*& obj, char* futureFlatBufferPtr);

#if !defined(GPUCA_GPUCODE)
  template <class Type>
  void subobjectCloneFromObject(Type*& obj, const Type* objOld);

  /// this functions 'smoothes' a CalDet by calculating for each value in a pad the average value using the neighbouring pads, but do not take into account the current pad
  /// \return returns 'smoothed' CalDet object
  /// \param thresholdMap zero supression threshold map which will be 'smoothed'
  /// \param maxThreshold max threshold value which will be considered for averaging
  /// \param nPadsInRowCl number of pads in row direction which will be taken into account (+- nPadsInRowCl)
  /// \param nPadsInPadCl number of pads in pad direction which will be taken into account (+- nPadsInPadCl)
  CalDet<float> processThresholdMap(const CalDet<float>& thresholdMap, const float maxThreshold, const int nPadsInRowCl = 2, const int nPadsInPadCl = 2) const;
#endif

#ifndef GPUCA_ALIROOT_LIB
  ClassDefNV(CalibdEdxContainer, 1);
#endif
};

} // namespace o2::tpc

#endif
