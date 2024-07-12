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

/// \file  TPCFastTransformPOD.cxx
/// \brief Implementation of POD correction map
///
/// \author  ruben.shahoayn@cern.ch

#include "TPCFastTransformPOD.h"
#include "GPUDebugStreamer.h"
#if !defined(GPUCA_GPUCODE)
#include <TRandom.h>
#endif

namespace GPUCA_NAMESPACE
{
namespace gpu
{

#if !defined(GPUCA_GPUCODE)

size_t TPCFastTransformPOD::estimateSize(const TPCFastSpaceChargeCorrection& origCorr)
{
  // estimate size of own buffer
  const size_t selfSizeFix = sizeof(TPCFastTransformPOD);
  size_t nextDynOffs = alignOffset(selfSizeFix);
  nextDynOffs = alignOffset(nextDynOffs + origCorr.mNumberOfScenarios * sizeof(size_t)); // spline scenarios start here
  // space for splines
  for (int isc = 0; isc < origCorr.mNumberOfScenarios; isc++) {
    const auto& spline = origCorr.mScenarioPtr[isc];
    nextDynOffs = alignOffset(nextDynOffs + sizeof(spline));
  }
  // space for splines data
  for (int is = 0; is < 3; is++) {
    for (int slice = 0; slice < origCorr.mGeo.getNumberOfSlices(); slice++) {
      for (int row = 0; row < NROWS; row++) {
        const auto& spline = origCorr.getSpline(slice, row);
        int nPar = spline.getNumberOfParameters();
        if (is == 1) {
          nPar = nPar / 3;
        }
        if (is == 2) {
          nPar = nPar * 2 / 3;
        }
        nextDynOffs += nPar * sizeof(float);
      }
    }
  }
  nextDynOffs = alignOffset(nextDynOffs);
  return nextDynOffs;
}

TPCFastTransformPOD& TPCFastTransformPOD::create(char* buff, size_t buffSize, const TPCFastSpaceChargeCorrection& origCorr)
{
  // instantiate object to already created buffer of the right size
  assert(buffSize > sizeof(TPCFastTransformPOD));
  auto& podMap = getNonConst(buff);
  podMap.mApplyCorrections = true; // by default always apply corrections

  // copy fixed size data --- start
  podMap.mNumberOfScenarios = origCorr.mNumberOfScenarios;
  std::memcpy(&podMap.mGeo, &origCorr.mGeo, sizeof(TPCFastTransformGeo)); // copy geometry (fixed size)
  for (int row = 0; row < NROWS; row++) {
    podMap.mRowInfo[row] = origCorr.getRowInfo(row); // dataOffsetBytes will be modified later
  }
  for (int slice = 0; slice < TPCFastTransformGeo::getNumberOfSlices(); slice++) {
    podMap.mSliceInfo[slice] = origCorr.getSliceInfo(slice);
    for (int row = 0; row < NROWS; row++) {
      podMap.mSliceRowInfo[NROWS * slice + row] = origCorr.getSliceRowInfo(slice, row);
    }
  }
  podMap.mInterpolationSafetyMargin = origCorr.fInterpolationSafetyMargin;
  podMap.mTimeStamp = origCorr.mTimeStamp;
  //
  // init data members coming from the TPCFastTrasform
  podMap.mVdrift = 0.;
  podMap.mT0 = 0.;
  podMap.mVdriftCorrY = 0.;
  podMap.mLdriftCorr = 0.;
  podMap.mTOFcorr = 0.;
  podMap.mPrimVtxZ = 0.;
  // copy fixed size data --- end

  size_t nextDynOffs = alignOffset(sizeof(TPCFastTransformPOD));
  // copy slice scenarios
  podMap.mOffsScenariosOffsets = nextDynOffs; // spline scenarios offsets start here
  LOGP(debug, "Set mOffsScenariosOffsets = {}", podMap.mOffsScenariosOffsets);
  nextDynOffs = alignOffset(nextDynOffs + podMap.mNumberOfScenarios * sizeof(size_t)); // spline scenarios start here

  // copy spline objects
  size_t* scenOffs = reinterpret_cast<size_t*>(buff + podMap.mOffsScenariosOffsets);
  for (int isc = 0; isc < origCorr.mNumberOfScenarios; isc++) {
    scenOffs[isc] = nextDynOffs;
    const auto& spline = origCorr.mScenarioPtr[isc];
    if (buffSize < nextDynOffs + sizeof(spline)) {
      throw std::runtime_error(fmt::format("attempt to copy {} bytes for spline for scenario {} to {}, overflowing the buffer of size {}", sizeof(spline), isc, nextDynOffs + sizeof(spline), buffSize));
    }
    std::memcpy(buff + scenOffs[isc], &spline, sizeof(spline));
    nextDynOffs = alignOffset(nextDynOffs + sizeof(spline));
    LOGP(debug, "Copy {} bytes for spline scenario {} (ptr:{}) to offsset {}", sizeof(spline), isc, (void*)&spline, scenOffs[isc]);
  }

  // copy splines data
  for (int is = 0; is < 3; is++) {
    float* data = reinterpret_cast<float*>(buff + nextDynOffs);
    LOGP(debug, "splinID={} start offset {} -> {}", is, nextDynOffs, (void*)data);
    for (int slice = 0; slice < origCorr.mGeo.getNumberOfSlices(); slice++) {
      podMap.mSplineDataOffsets[slice][is] = nextDynOffs;
      size_t rowDataOffs = 0;
      for (int row = 0; row < NROWS; row++) {
        const auto& spline = origCorr.getSpline(slice, row);
        const float* dataOr = origCorr.getSplineData(slice, row, is);
        int nPar = spline.getNumberOfParameters();
        if (is == 1) {
          nPar = nPar / 3;
        }
        if (is == 2) {
          nPar = nPar * 2 / 3;
        }
        LOGP(debug, "Copying {} floats for spline{} of slice:{} row:{} to offset {}", nPar, is, slice, row, nextDynOffs);
        size_t nbcopy = nPar * sizeof(float);
        if (buffSize < nextDynOffs + nbcopy) {
          throw std::runtime_error(fmt::format("attempt to copy {} bytes of data for spline{} of slice{}/row{} to {}, overflowing the buffer of size {}", nbcopy, is, slice, row, nextDynOffs, buffSize));
        }
        std::memcpy(data, dataOr, nbcopy);
        podMap.getRowInfo(row).dataOffsetBytes[is] = rowDataOffs;
        rowDataOffs += nbcopy;
        data += nPar;
        nextDynOffs += nbcopy;
      }
    }
  }
  podMap.mTotalSize = alignOffset(nextDynOffs);
  if (buffSize != podMap.mTotalSize) {
    throw std::runtime_error(fmt::format("Estimated buffer size {} differs from filled one {}", buffSize, podMap.mTotalSize));
  }
  return podMap;
}

TPCFastTransformPOD& TPCFastTransformPOD::create(char* buff, size_t buffSize, const TPCFastTransform& src)
{
  // instantiate objec to already created buffer of the right size
  auto& podMap = create(buff, buffSize, src.getCorrection());
  // set data members of TPCFastTransform
  podMap.mVdrift = src.getVDrift();
  podMap.mT0 = src.getT0();
  podMap.mVdriftCorrY = src.getVdriftCorrY();
  podMap.mLdriftCorr = src.getLdriftCorr();
  podMap.mTOFcorr = src.getTOFCorr();
  podMap.mPrimVtxZ = src.getPrimVtxZ();
  // copy fixed size data --- end
  return podMap;
}

bool TPCFastTransformPOD::test(const TPCFastSpaceChargeCorrection& origCorr, int npoints) const
{
  if (npoints < 1) {
    return false;
  }
  std::vector<unsigned char> slice, row;
  std::vector<float> u, v, dxO, duO, dvO, dxP, duP, dvP, corrXO, corrXP, nomUO, nomVO, nomUP, nomVP;
  slice.reserve(npoints);
  row.reserve(npoints);
  u.reserve(npoints);
  v.reserve(npoints);
  dxO.resize(npoints);
  duO.resize(npoints);
  dvO.resize(npoints);
  corrXO.resize(npoints);
  nomUO.resize(npoints);
  nomVO.resize(npoints);
  dxP.resize(npoints);
  duP.resize(npoints);
  dvP.resize(npoints);
  corrXP.resize(npoints);
  nomUP.resize(npoints);
  nomVP.resize(npoints);

  for (int i = 0; i < npoints; i++) {
    slice.push_back(gRandom->Integer(NSLICES));
    row.push_back(gRandom->Integer(NROWS));
    u.push_back(gRandom->Rndm() * 15);
    v.push_back(gRandom->Rndm() * 200);
  }
  long origStart[3], origEnd[3], thisStart[3], thisEnd[3];
  origStart[0] = std::chrono::time_point_cast<std::chrono::microseconds>(std::chrono::system_clock::now()).time_since_epoch().count();
  for (int i = 0; i < npoints; i++) {
    origCorr.getCorrection(slice[i], row[i], u[i], v[i], dxO[i], duO[i], dvO[i]);
  }
  origEnd[0] = origStart[1] = std::chrono::time_point_cast<std::chrono::microseconds>(std::chrono::system_clock::now()).time_since_epoch().count();
  for (int i = 0; i < npoints; i++) {
    origCorr.getCorrectionInvCorrectedX(slice[i], row[i], u[i], v[i], corrXO[i]);
  }
  origEnd[1] = origStart[2] = std::chrono::time_point_cast<std::chrono::microseconds>(std::chrono::system_clock::now()).time_since_epoch().count();
  for (int i = 0; i < npoints; i++) {
    origCorr.getCorrectionInvUV(slice[i], row[i], u[i], v[i], nomUO[i], nomVO[i]);
  }
  origEnd[2] = thisStart[0] = std::chrono::time_point_cast<std::chrono::microseconds>(std::chrono::system_clock::now()).time_since_epoch().count();
  for (int i = 0; i < npoints; i++) {
    this->getCorrection(slice[i], row[i], u[i], v[i], dxP[i], duP[i], dvP[i]);
  }
  thisEnd[0] = thisStart[1] = std::chrono::time_point_cast<std::chrono::microseconds>(std::chrono::system_clock::now()).time_since_epoch().count();
  for (int i = 0; i < npoints; i++) {
    this->getCorrectionInvCorrectedX(slice[i], row[i], u[i], v[i], corrXP[i]);
  }
  thisEnd[1] = thisStart[2] = std::chrono::time_point_cast<std::chrono::microseconds>(std::chrono::system_clock::now()).time_since_epoch().count();
  for (int i = 0; i < npoints; i++) {
    this->getCorrectionInvUV(slice[i], row[i], u[i], v[i], nomUP[i], nomVP[i]);
  }
  thisEnd[2] = std::chrono::time_point_cast<std::chrono::microseconds>(std::chrono::system_clock::now()).time_since_epoch().count();
  //
  size_t ndiff[3] = {};
  for (int i = 0; i < npoints; i++) {
    if (dxO[i] != dxP[i] || duO[i] != duP[i] || dvO[i] != dvP[i]) {
      ndiff[0]++;
    }
    if (corrXO[i] != corrXP[i]) {
      ndiff[1]++;
    }
    if (nomUO[i] != nomUP[i] || nomVO[i] != nomVP[i]) {
      ndiff[2]++;
    }
  }
  //
  LOGP(info, " (ns per call)              original        this     Nmissmatch");
  LOGP(info, "getCorrection               {:.3e}    {:.3e}   {}", double(origEnd[0] - origStart[0]) / npoints * 1000., double(thisEnd[0] - thisStart[0]) / npoints * 1000., ndiff[0]);
  LOGP(info, "getCorrectionInvCorrectedX  {:.3e}    {:.3e}   {}", double(origEnd[1] - origStart[1]) / npoints * 1000., double(thisEnd[1] - thisStart[1]) / npoints * 1000., ndiff[1]);
  LOGP(info, "getCorrectionInvUV          {:.3e}    {:.3e}   {}", double(origEnd[2] - origStart[2]) / npoints * 1000., double(thisEnd[2] - thisStart[2]) / npoints * 1000., ndiff[2]);
  return ndiff[0] == 0 && ndiff[1] == 0 && ndiff[2] == 0;
}

#endif

} // namespace gpu
} // namespace GPUCA_NAMESPACE
