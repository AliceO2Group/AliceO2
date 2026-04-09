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
//
/// \author  ruben.shahoayn@cern.ch
/// \file  TPCFastTransformPOD.cxx
/// \brief Implementation of POD correction map
///

#if !defined(GPUCA_NO_ROOT) && !defined(GPUCA_NO_FMT) && !defined(GPUCA_STANDALONE)
#include <TRandom.h>
#endif
#include "TPCFastTransformPOD.h"
#include "GPUDebugStreamer.h"
#include "GPUCommonLogger.h"

namespace o2
{
namespace gpu
{

/// Create POD transform from old flat-buffer one. Provided vector will serve as a buffer
TPCFastTransformPOD* TPCFastTransformPOD::create(aligned_unique_buffer_ptr<TPCFastTransformPOD>& destVector, const TPCFastTransform& src)
{
  size_t size = estimateSize(src);
  destVector.alloc(size); // allocate exact size
  LOGP(debug, "OrigCorrSize:{} SelfSize: {} Estimated POS size: {}", src.getCorrection().getFlatBufferSize(), sizeof(TPCFastTransformPOD), size);
  auto res = create(destVector.getraw(), size, src);
  res->setTimeStamp(src.getTimeStamp());
  res->setVDrift(src.getVDrift());
  res->setT0(src.getT0());
  res->setLumi(src.getLumi());
  if (src.isIDCSet()) {
    res->setIDC(src.getIDC());
  }
  return res;
}

TPCFastTransformPOD* TPCFastTransformPOD::create(aligned_unique_buffer_ptr<TPCFastTransformPOD>& destVector, const TPCFastSpaceChargeCorrection& origCorr)
{
  // create filling only part corresponding to TPCFastSpaceChargeCorrection. Data members coming from TPCFastTransform (e.g. VDrift, T0..) are not set
  size_t size = estimateSize(origCorr);
  destVector.alloc(size);
  LOGP(debug, "OrigCorrSize:{} SelfSize: {} Estimated POS size: {}", origCorr.getFlatBufferSize(), sizeof(TPCFastTransformPOD), size);
  return create(destVector.getraw(), size, origCorr);
}

size_t TPCFastTransformPOD::estimateSize(const TPCFastSpaceChargeCorrection& origCorr)
{
  // estimate size of own buffer
  const size_t selfSizeFix = sizeof(TPCFastTransformPOD);
  size_t nextDynOffs = alignOffset(selfSizeFix);
  nextDynOffs = alignOffset(nextDynOffs + origCorr.mNumberOfScenarios * sizeof(size_t)); // spline scenarios start here
  nextDynOffs = alignOffset(nextDynOffs + origCorr.mNumberOfScenarios * sizeof(size_t)); // flatBufOffs array
  // space for splines (use sizeof(SplineType) = slim size, not the origCorr spline size)
  for (int isc = 0; isc < origCorr.mNumberOfScenarios; isc++) {
    const auto& spline = origCorr.mScenarioPtr[isc];
    nextDynOffs = alignOffset(nextDynOffs + sizeof(SplineType));
    nextDynOffs = alignOffset(nextDynOffs + spline.getFlatBufferSize());
  }
  // space for splines data
  for (int is = 0; is < 3; is++) {
    for (int sector = 0; sector < origCorr.mGeo.getNumberOfSectors(); sector++) {
      for (int row = 0; row < NROWS; row++) {
        const auto& spline = origCorr.getSpline(sector, row);
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

void TPCFastTransformPOD::print() const
{
  LOGP(info, "TPCFastTransformPOD: this={:p} sizeof={} mApplyCorrection={} mNumberOfScenarios={} mTotalSize={} mOffsScenariosOffsets={} mT0={} mVdrift={} mLumi={} mIDC={}",
       (void*)this, sizeof(*this), mApplyCorrection, mNumberOfScenarios, mTotalSize, mOffsScenariosOffsets, mT0, mVdrift, mLumi, mIDC);

  for (int s = 0; s < TPCFastTransformGeo::getNumberOfSectors(); s++) {
    for (int i = 0; i < NSplineIDs; i++) {
      LOGP(info, "mSplineDataOffsets[{}][{}]={}", s, i, mSplineDataOffsets[s][i]);
    }
  }
  const size_t scenOffset = getScenarioOffset(0);
  const auto& spline = getSpline(0, 0);
  LOGP(info, "scenOffset={} spline_addr={:p} expected={:p}", scenOffset, (void*)&spline, (void*)(getThis() + scenOffset));

  const float* splineData = getCorrectionData(0, 0);
  LOGP(info, "spline internal check: &spline={:p} splineData={:p} buf_start={:p} buf_end={:p}",
       (void*)&spline, (void*)splineData,
       (void*)getThis(), (void*)(getThis() + mTotalSize));

  // check if splineData is within buffer
  bool dataInBuf = (splineData >= (float*)getThis()) && (splineData < (float*)(getThis() + mTotalSize));
  LOGP(info, "splineData in buffer: {}", dataInBuf);

  LOGP(info, "splineData offset from buf_start = {}", (size_t)((const char*)splineData - getThis()));
}

TPCFastTransformPOD* TPCFastTransformPOD::create(char* buff, size_t buffSize, const TPCFastSpaceChargeCorrection& origCorr)
{
  // instantiate object to already created buffer of the right size
  assert(buffSize > sizeof(TPCFastTransformPOD));
  auto& podMap = getNonConst(buff);
  podMap.mApplyCorrection = true; // by default always apply corrections

  // copy fixed size data --- start
  podMap.mNumberOfScenarios = origCorr.mNumberOfScenarios;
  std::memcpy((void*)&podMap.mGeo, (const void*)&origCorr.mGeo, sizeof(TPCFastTransformGeo)); // copy geometry (fixed size)
  static_assert(sizeof(podMap.mGeo) == sizeof(origCorr.mGeo));
  for (int sector = 0; sector < TPCFastTransformGeo::getNumberOfSectors(); sector++) {
    for (int row = 0; row < NROWS; row++) {
      podMap.mSectorRowInfos[NROWS * sector + row] = origCorr.getSectorRowInfo(sector, row);
    }
  }
  podMap.mTimeStamp = origCorr.mTimeStamp;
  //
  // init data members coming from the TPCFastTrasform
  podMap.mVdrift = 0.;
  podMap.mT0 = 0.;
  // copy fixed size data --- end

  size_t nextDynOffs = alignOffset(sizeof(TPCFastTransformPOD));

  // copy sector scenarios
  podMap.mOffsScenariosOffsets = nextDynOffs; // spline scenarios offsets start here
  LOGP(debug, "Set mOffsScenariosOffsets = {}", podMap.mOffsScenariosOffsets);
  nextDynOffs = alignOffset(nextDynOffs + podMap.mNumberOfScenarios * sizeof(size_t)); // spline scenarios start here

  podMap.mOffsFlatBufferOffsets = nextDynOffs; // <-- add this
  nextDynOffs = alignOffset(nextDynOffs + podMap.mNumberOfScenarios * sizeof(size_t));

  // copy spline objects
  size_t* scenOffs = reinterpret_cast<size_t*>(buff + podMap.mOffsScenariosOffsets);
  size_t* flatBufOffs = reinterpret_cast<size_t*>(buff + podMap.mOffsFlatBufferOffsets);

  for (int isc = 0; isc < origCorr.mNumberOfScenarios; isc++) {
    scenOffs[isc] = nextDynOffs;
    const auto& spline = origCorr.mScenarioPtr[isc];
    if (buffSize < nextDynOffs + sizeof(SplineType)) {
      throw std::runtime_error(fmt::format("attempt to write {} bytes for slim spline for scenario {} to {}, overflowing the buffer of size {}", sizeof(SplineType), isc, nextDynOffs + sizeof(SplineType), buffSize));
    }

    // Placement-new a slim (NoFlatObject) spline and populate its schema from the source
    auto* slimSpline = new (buff + scenOffs[isc]) SplineType();
    slimSpline->importFrom(spline);
    nextDynOffs = alignOffset(nextDynOffs + sizeof(SplineType));
    LOGP(debug, "Write {} bytes for slim spline scenario {} to offset {}", sizeof(SplineType), isc, scenOffs[isc]);

    // copy spline flat buffer (layout identical regardless of FlatBase)
    flatBufOffs[isc] = nextDynOffs;
    std::memcpy(buff + nextDynOffs, spline.getFlatBufferPtr(), spline.getFlatBufferSize());

    // fix up internal pointers (mParameters, mGridX1.mKnots, mGridX2.mKnots)
    slimSpline->setActualBufferAddress(buff + nextDynOffs);

    nextDynOffs = alignOffset(nextDynOffs + spline.getFlatBufferSize());
  }

  // copy splines data
  for (int is = 0; is < 3; is++) {
    float* data = reinterpret_cast<float*>(buff + nextDynOffs);
    LOGP(debug, "splinID={} start offset {} -> {}", is, nextDynOffs, (void*)data);
    for (int sector = 0; sector < origCorr.mGeo.getNumberOfSectors(); sector++) {
      podMap.mSplineDataOffsets[sector][is] = nextDynOffs;
      size_t rowDataOffs = 0;
      for (int row = 0; row < NROWS; row++) {
        const auto& spline = origCorr.getSpline(sector, row);
        const float* dataOr = origCorr.getCorrectionData(sector, row, is);
        int nPar = spline.getNumberOfParameters();
        if (is == 1) {
          nPar = nPar / 3;
        }
        if (is == 2) {
          nPar = nPar * 2 / 3;
        }
        LOGP(debug, "Copying {} floats for spline{} of sector:{} row:{} to offset {}", nPar, is, sector, row, nextDynOffs);
        size_t nbcopy = nPar * sizeof(float);
        if (buffSize < nextDynOffs + nbcopy) {
          throw std::runtime_error(fmt::format("attempt to copy {} bytes of data for spline{} of sector{}/row{} to {}, overflowing the buffer of size {}", nbcopy, is, sector, row, nextDynOffs, buffSize));
        }
        std::memcpy(data, dataOr, nbcopy);
        podMap.getSectorRowInfo(sector, row).dataOffsetBytes[is] = rowDataOffs;
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
  return &getNonConst(buff);
}

TPCFastTransformPOD* TPCFastTransformPOD::create(char* buff, size_t buffSize, const TPCFastTransform& src)
{
  // instantiate objec to already created buffer of the right size
  auto podMap = create(buff, buffSize, src.getCorrection());
  // set data members of TPCFastTransform
  podMap->mVdrift = src.getVDrift();
  podMap->mT0 = src.getT0();
  podMap->mLumi = src.getLumi();
  if (src.isIDCSet()) {
    podMap->mIDC = src.getIDC();
  }
  podMap->mTimeStamp = src.getTimeStamp();
  // copy fixed size data --- end
  return podMap;
}

#ifndef GPUCA_STANDALONE

bool TPCFastTransformPOD::test(const TPCFastSpaceChargeCorrection& origCorr, int npoints) const
{
  if (npoints < 1) {
    return false;
  }
  std::vector<unsigned char> sector, row;
  std::vector<float> y, z;
  std::vector<std::array<float, 3>> corr0, corr1;
  std::vector<std::array<float, 2>> corrInv0, corrInv1;
  std::vector<float> corrInvX0, corrInvX1;

  sector.reserve(npoints);
  row.reserve(npoints);
  y.reserve(npoints);
  z.reserve(npoints);
  corr0.reserve(npoints);
  corr1.reserve(npoints);
  corrInv0.reserve(npoints);
  corrInv1.reserve(npoints);
  corrInvX0.reserve(npoints);
  corrInvX1.reserve(npoints);

  for (int i = 0; i < npoints; i++) {
    sector.push_back(gRandom->Integer(NSECTORS));
    row.push_back(gRandom->Integer(NROWS));
    y.push_back((gRandom->Rndm() - 0.5) * mGeo.getRowInfoMaxPad(row.back()) * mGeo.getRowInfoPadWidth(row.back()));
    z.push_back((sector.back() < NSECTORS / 2 ? 1.f : -1.f) * gRandom->Rndm() * 240);
  }
  long origStart[3], origEnd[3], thisStart[3], thisEnd[3];
  origStart[0] = std::chrono::time_point_cast<std::chrono::microseconds>(std::chrono::system_clock::now()).time_since_epoch().count();
  for (int i = 0; i < npoints; i++) {
    std::array<float, 3> val;
    origCorr.getCorrectionLocal(sector[i], row[i], y[i], z[i], val[0], val[1], val[2]);
    corr0.push_back(val);
  }

  origEnd[0] = origStart[1] = std::chrono::time_point_cast<std::chrono::microseconds>(std::chrono::system_clock::now()).time_since_epoch().count();
  for (int i = 0; i < npoints; i++) {
    std::array<float, 2> val;
    origCorr.getCorrectionYZatRealYZ(sector[i], row[i], y[i], z[i], val[0], val[1]);
    corrInv0.push_back(val);
  }

  origEnd[1] = origStart[2] = std::chrono::time_point_cast<std::chrono::microseconds>(std::chrono::system_clock::now()).time_since_epoch().count();
  for (int i = 0; i < npoints; i++) {
    corrInvX0.push_back(origCorr.getCorrectionXatRealYZ(sector[i], row[i], y[i], z[i]));
  }
  //
  origEnd[2] = thisStart[0] = std::chrono::time_point_cast<std::chrono::microseconds>(std::chrono::system_clock::now()).time_since_epoch().count();
  for (int i = 0; i < npoints; i++) {
    std::array<float, 3> val;
    this->getCorrectionLocal(sector[i], row[i], y[i], z[i], val[0], val[1], val[2]);
    corr1.push_back(val);
  }
  thisEnd[0] = thisStart[1] = std::chrono::time_point_cast<std::chrono::microseconds>(std::chrono::system_clock::now()).time_since_epoch().count();
  for (int i = 0; i < npoints; i++) {
    std::array<float, 2> val;
    this->getCorrectionYZatRealYZ(sector[i], row[i], y[i], z[i], val[0], val[1]);
    corrInv1.push_back(val);
  }

  thisEnd[1] = thisStart[2] = std::chrono::time_point_cast<std::chrono::microseconds>(std::chrono::system_clock::now()).time_since_epoch().count();
  for (int i = 0; i < npoints; i++) {
    corrInvX1.push_back(this->getCorrectionXatRealYZ(sector[i], row[i], y[i], z[i]));
  }
  thisEnd[2] = std::chrono::time_point_cast<std::chrono::microseconds>(std::chrono::system_clock::now()).time_since_epoch().count();
  //
  size_t ndiff[3] = {};
  for (int i = 0; i < npoints; i++) {
    if (corr0[i][0] != corr1[i][0] || corr0[i][1] != corr1[i][1] || corr0[i][2] != corr1[i][2]) {
      ndiff[0]++;
    }
    if (corrInv0[i][0] != corrInv1[i][0] || corrInv0[i][1] != corrInv1[i][1]) {
      ndiff[1]++;
    }
    if (corrInvX0[i] != corrInvX1[i]) {
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
} // namespace o2
