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

///
/// @file   CalibPadGainTracks.cxx
/// @author Matthias Kleiner, matthias.kleiner@cern.ch
///

#include "TPCCalibration/CalibPadGainTracks.h"
#include "TPCBase/PadPos.h"
#include "TPCBase/ROC.h"
#include "TPCBase/Mapper.h"
#include "TPCCalibration/IDCDrawHelper.h"
#include "CorrectionMapsHelper.h"
#include "TPCReconstruction/TPCFastTransformHelperO2.h"
#include "GPUO2InterfaceRefit.h"
#include "GPUO2Interface.h"
#include "DataFormatsTPC/ClusterNative.h"
#include "DataFormatsTPC/VDriftCorrFact.h"

// root includes
#include "TFile.h"
#include <random>

using namespace o2::tpc;

void CalibPadGainTracks::processTracks(const int nMaxTracks)
{
  std::unique_ptr<o2::gpu::GPUO2InterfaceRefit> refit;
  if (!mPropagateTrack) {
    mBufVec.resize(mClusterIndex->nClustersTotal);
    o2::gpu::GPUO2InterfaceRefit::fillSharedClustersMap(mClusterIndex, *mTracks, mTPCTrackClIdxVecInput->data(), mBufVec.data());
    mClusterShMapTPC = mBufVec.data();
    refit = std::make_unique<o2::gpu::GPUO2InterfaceRefit>(mClusterIndex, mTPCCorrMapsHelper, mField, mTPCTrackClIdxVecInput->data(), mClusterShMapTPC);
  }

  const size_t loopEnd = (nMaxTracks < 0) ? mTracks->size() : ((nMaxTracks > mTracks->size()) ? mTracks->size() : size_t(nMaxTracks));

  if (loopEnd < mTracks->size()) {
    // draw random tracks
    std::vector<size_t> ind(mTracks->size());
    std::iota(ind.begin(), ind.end(), 0);
    std::minstd_rand rng(std::time(nullptr));
    std::shuffle(ind.begin(), ind.end(), rng);
    for (size_t i = 0; i < loopEnd; ++i) {
      processTrack((*mTracks)[ind[i]], refit.get());
    }
  } else {
    for (const auto& trk : *mTracks) {
      processTrack(trk, refit.get());
    }
  }
}

void CalibPadGainTracks::processTrack(o2::tpc::TrackTPC track, o2::gpu::GPUO2InterfaceRefit* refit)
{
  // make momentum cut
  const float mom = track.getP();
  const int nClusters = track.getNClusterReferences();
  if (mom < mMomMin || mom > mMomMax || std::abs(track.getEta()) > mEtaMax || nClusters < mMinClusters) {
    return;
  }

  // clearing memory
  for (auto& buffer : mDEdxBuffer) {
    buffer.clear();
  }
  mClTrk.clear();

  for (int iCl = 0; iCl < nClusters; iCl++) { // loop over cluster
    const o2::tpc::ClusterNative& cl = track.getCluster(*mTPCTrackClIdxVecInput, iCl, *mClusterIndex);

    const auto flagsCl = cl.getFlags();
    if ((flagsCl & ClusterNative::flagSingle) == ClusterNative::flagSingle) {
      continue;
    }

    unsigned char sectorIndex = 0;
    unsigned char rowIndex = 0;
    unsigned int clusterIndexNumb = 0;

    // this function sets sectorIndex, rowIndex, clusterIndexNumb
    track.getClusterReference(*mTPCTrackClIdxVecInput, iCl, sectorIndex, rowIndex, clusterIndexNumb);
    const float xPosition = Mapper::instance().getPadCentre(PadPos(rowIndex, 0)).X();
    if (!mPropagateTrack) {
      refit->setTrackReferenceX(xPosition);
    }
    const bool check = mPropagateTrack ? track.propagateTo(xPosition, mField) : ((refit->RefitTrackAsGPU(track, false, true) < 0) ? false : true); // propagate this track to the plane X=xk (cm) in the field "b" (kG)

    if (!check || std::isnan(track.getParam(1))) {
      continue;
    }

    const int region = Mapper::REGION[rowIndex];
    const float charge = (mChargeType == ChargeType::Max) ? cl.qMax : cl.qTot;
    const float effectiveLength = mCalibTrackTopologyPol ? getTrackTopologyCorrectionPol(track, cl, region, charge) : getTrackTopologyCorrection(track, region);

    const unsigned char pad = std::clamp(static_cast<unsigned int>(cl.getPad() + 0.5f), static_cast<unsigned int>(0), Mapper::PADSPERROW[region][Mapper::getLocalRowFromGlobalRow(rowIndex)] - 1); // the left side of the pad is defined at e.g. 3.5 and the right side at 4.5
    const float gain = mGainMapRef ? mGainMapRef->getValue(sectorIndex, rowIndex, pad) : 1;
    const float chargeNorm = charge / (effectiveLength * gain);

    if (mMode == dedxTracking) {
      const auto& dEdx = track.getdEdx();
      float dedx = -1;
      const CRU cru(Sector(sectorIndex), region);

      if (mDedxRegion == stack) {
        const auto stack = cru.gemStack();
        if (stack == GEMstack::IROCgem && dEdx.NHitsIROC > mMinClusters) {
          dedx = getdEdxIROC(dEdx);
        } else if (stack == GEMstack::OROC1gem && dEdx.NHitsOROC1 > mMinClusters) {
          dedx = getdEdxOROC1(dEdx);
        } else if (stack == GEMstack::OROC2gem && dEdx.NHitsOROC2 > mMinClusters) {
          dedx = getdEdxOROC2(dEdx);
        } else if (stack == GEMstack::OROC3gem && dEdx.NHitsOROC3 > mMinClusters) {
          dedx = getdEdxOROC3(dEdx);
        }
      } else if (mDedxRegion == chamber) {
        if (cru.isIROC() && dEdx.NHitsIROC > mMinClusters) {
          dedx = getdEdxIROC(dEdx);
        } else {
          int count = 0;
          if (dEdx.NHitsOROC1 > mMinClusters) {
            dedx += getdEdxOROC1(dEdx);
            ++count;
          }
          if (dEdx.NHitsOROC2 > mMinClusters) {
            dedx += getdEdxOROC2(dEdx);
            ++count;
          }
          if (dEdx.NHitsOROC3 > mMinClusters) {
            dedx += getdEdxOROC3(dEdx);
            ++count;
          }
          if (count > 0) {
            dedx /= count;
          }
        }
      } else if (mDedxRegion == sector) {
        dedx = getdEdxTPC(dEdx);
      }

      if (dedx <= 0) {
        continue;
      }

      if (dedx < mDedxMin || (mDedxMax > 0 && dedx > mDedxMax)) {
        continue;
      }

      const float fillVal = mDoNotNormCharge ? chargeNorm : chargeNorm / dedx;
      int index = Mapper::GLOBALPADOFFSET[region] + Mapper::OFFSETCRUGLOBAL[rowIndex] + pad;
      if (cru.isOROC()) {
        index -= Mapper::getPadsInIROC();
      }

      fillPadByPadHistogram(cru.roc().getRoc(), index, fillVal);
    }

    if (mMode == dedxTrack) {
      const int indexBuffer = getdEdxBufferIndex(region);

      const bool isEdge = (flagsCl & ClusterNative::flagEdge) == ClusterNative::flagEdge;
      const int isSectorCentre = std::abs(static_cast<int>(pad) - static_cast<int>(Mapper::PADSPERROW[region][rowIndex] / 2)); // do not use clusters at the centre of a sector for dE/dx calculation
      const int nPadsSector = 1;

      if (!isEdge && (isSectorCentre > nPadsSector)) {
        mDEdxBuffer[indexBuffer].emplace_back(chargeNorm);
      }

      mClTrk.emplace_back(std::make_tuple(sectorIndex, rowIndex, pad, chargeNorm)); // fill with dummy dedx value
    }
  }

  if (mMode == dedxTrack) {
    getTruncMean();

    // set the dEdx
    for (auto& x : mClTrk) {
      const unsigned char globRow = std::get<1>(x);
      const int region = Mapper::REGION[globRow];
      const int indexBuffer = getdEdxBufferIndex(region);

      const float dedxTmp = mDedxTmp[indexBuffer];
      if (dedxTmp <= 0 || dedxTmp < mDedxMin || (mDedxMax > 0 && dedxTmp > mDedxMax)) {
        continue;
      }

      const unsigned char pad = std::get<2>(x);
      int index = Mapper::GLOBALPADOFFSET[region] + Mapper::OFFSETCRUGLOBAL[globRow] + pad;
      const ROC roc = CRU(Sector(std::get<0>(x)), region).roc();
      if (roc.isOROC()) {
        index -= Mapper::getPadsInIROC();
      }

      // fill the normalizes charge in pad histogram
      const float fillVal = mDoNotNormCharge ? std::get<3>(x) : std::get<3>(x) / dedxTmp;
      fillPadByPadHistogram(roc.getRoc(), index, fillVal);
    }
  } else {
  }
}

void CalibPadGainTracks::getTruncMean(float low, float high)
{
  mDedxTmp.clear();
  mDedxTmp.reserve(mDEdxBuffer.size());
  // returns the truncated mean for input vector
  for (auto& charge : mDEdxBuffer) {
    const int nClustersUsed = static_cast<int>(charge.size());
    if (nClustersUsed < mMinClusters) {
      mDedxTmp.emplace_back(-1);
      continue;
    }

    std::sort(charge.begin(), charge.end()); // sort the vector for performing truncated mean

    const int startInd = static_cast<int>(low * nClustersUsed);
    const int endInd = static_cast<int>(high * nClustersUsed);

    if (endInd <= startInd) {
      mDedxTmp.emplace_back(-1);
      continue;
    }

    const float dEdx = std::accumulate(charge.begin() + startInd, charge.begin() + endInd, 0.f);
    const int nClustersTrunc = endInd - startInd; // count number of clusters
    mDedxTmp.emplace_back(dEdx / nClustersTrunc);
  }
}

float CalibPadGainTracks::getTrackTopologyCorrection(const o2::tpc::TrackTPC& track, const unsigned int region) const
{
  const float padLength = Mapper::instance().getPadRegionInfo(region).getPadHeight();
  const float sinPhi = track.getSnp();
  const float tgl = track.getTgl();
  const float snp2 = sinPhi * sinPhi;
  const float effectiveLength = padLength * std::sqrt((1 + tgl * tgl) / (1 - snp2)); // calculate the trace length of the track over the pad
  return effectiveLength;
}

float CalibPadGainTracks::getTrackTopologyCorrectionPol(const o2::tpc::TrackTPC& track, const o2::tpc::ClusterNative& cl, const unsigned int region, const float charge) const
{
  const float trackSnp = std::abs(track.getSnp());
  float snp2 = trackSnp * trackSnp;
  const float sec2 = 1.f / (1.f - snp2);
  const float trackTgl = track.getTgl();
  const float tgl2 = trackTgl * trackTgl;
  const float tanTheta = std::sqrt(tgl2 * sec2);

  const float z = std::abs(track.getParam(1));
  const float padTmp = cl.getPad();
  const float absRelPad = std::abs(padTmp - int(padTmp + 0.5f));
  const float relTime = cl.getTime() - int(cl.getTime() + 0.5f);

  const float effectiveLength = (mChargeType == ChargeType::Max) ? mCalibTrackTopologyPol->getCorrectionqMax(region, tanTheta, trackSnp, z, absRelPad, relTime) : mCalibTrackTopologyPol->getCorrectionqTot(region, tanTheta, trackSnp, z, 3.5f /*dummy threshold for now*/, charge);
  return effectiveLength;
}

void CalibPadGainTracks::reserveMemory()
{
  mClTrk.reserve(Mapper::PADROWS);
  resizedEdxBuffer();
}

void CalibPadGainTracks::resizedEdxBuffer()
{
  if (mDedxRegion == stack) {
    mDEdxBuffer.resize(4);
    mDEdxBuffer[0].reserve(Mapper::getNumberOfRowsInIROC());
    mDEdxBuffer[1].reserve(Mapper::getNumberOfRowsInOROC());
    mDEdxBuffer[2].reserve(Mapper::getNumberOfRowsInOROC());
    mDEdxBuffer[3].reserve(Mapper::getNumberOfRowsInOROC());
  } else if (mDedxRegion == chamber) {
    mDEdxBuffer.resize(2);
    mDEdxBuffer[0].reserve(Mapper::getNumberOfRowsInIROC());
    mDEdxBuffer[1].reserve(Mapper::getNumberOfRowsInOROC());
  } else if (mDedxRegion == sector) {
    mDEdxBuffer.resize(1);
    mDEdxBuffer[0].reserve(Mapper::instance().getNumberOfRows());
  } else {
    LOGP(warning, "wrong dE/dx type");
  }
}

void CalibPadGainTracks::setdEdxRegion(const DEdxRegion dedx)
{
  mDedxRegion = dedx;
  resizedEdxBuffer();
}

void CalibPadGainTracks::dumpToFile(const char* outFileName, const char* outName) const
{
  TFile fOut(outFileName, "RECREATE");
  fOut.WriteObject(this, outName);
  fOut.Close();
}

void CalibPadGainTracks::setMembers(gsl::span<const o2::tpc::TrackTPC>* vTPCTracksArrayInp, gsl::span<const o2::tpc::TPCClRefElem>* tpcTrackClIdxVecInput, const o2::tpc::ClusterNativeAccess& clIndex)
{
  mTracks = vTPCTracksArrayInp;
  mTPCTrackClIdxVecInput = tpcTrackClIdxVecInput;
  mClusterIndex = &clIndex;
}

void CalibPadGainTracks::setMomentumRange(const float momMin, const float momMax)
{
  mMomMin = momMin;
  mMomMax = momMax;
}

void CalibPadGainTracks::setRefGainMap(const char* inpFile, const char* mapName)
{
  TFile f(inpFile, "READ");
  o2::tpc::CalPad* gainMap = nullptr;
  f.GetObject(mapName, gainMap);

  if (!gainMap) {
    LOGP(info, "GainMap {} not found returning", mapName);
    return;
  }
  setRefGainMap(*gainMap);
  delete gainMap;
}

int CalibPadGainTracks::getdEdxBufferIndex(const int region) const
{
  if (mDedxRegion == stack) {
    return static_cast<int>(CRU(region).gemStack());
  } else if (mDedxRegion == chamber) {
    return static_cast<int>(CRU(region).rocType());
  } else if (mDedxRegion == sector) {
    return 0;
  } else {
    LOGP(warning, "wrong dE/dx type");
    return -1;
  }
}

void CalibPadGainTracks::loadPolTopologyCorrectionFromFile(std::string_view fileName)
{
  mCalibTrackTopologyPol = std::make_unique<CalibdEdxTrackTopologyPol>();
  mCalibTrackTopologyPol->loadFromFile(fileName.data(), "CalibdEdxTrackTopologyPol");
}

void CalibPadGainTracks::setPolTopologyCorrectionFromContainer(const CalibdEdxTrackTopologyPolContainer& polynomials)
{
  mCalibTrackTopologyPol = std::make_unique<CalibdEdxTrackTopologyPol>();
  mCalibTrackTopologyPol->setFromContainer(polynomials);
}

void CalibPadGainTracks::drawRefGainMapHelper(const bool type, const Sector sector, const std::string filename, const float minZ, const float maxZ) const
{
  if (!mGainMapRef) {
    LOGP(error, "Map not set");
    return;
  }

  std::function<float(const unsigned int, const unsigned int, const unsigned int, const unsigned int)> idcFunc = [mapTmp = mGainMapRef.get()](const unsigned int sector, const unsigned int region, const unsigned int lrow, const unsigned int pad) {
    return mapTmp->getValue(sector, Mapper::getGlobalPadNumber(lrow, pad, region));
  };

  IDCDrawHelper::IDCDraw drawFun;
  drawFun.mIDCFunc = idcFunc;
  const std::string zAxisTitle = "rel. gain";
  type ? IDCDrawHelper::drawSide(drawFun, sector.side(), zAxisTitle, filename, minZ, maxZ) : IDCDrawHelper::drawSector(drawFun, 0, Mapper::NREGIONS, sector, zAxisTitle, filename, minZ, maxZ);
}

void CalibPadGainTracks::dumpReferenceExtractedGainMap(const char* outFileName, const char* outName) const
{
  if (!mGainMapRef) {
    LOGP(error, "Map not set");
    return;
  }
  CalDet gainMapRef(*mGainMapRef.get());
  gainMapRef *= getPadGainMap();

  TFile f(outFileName, "RECREATE");
  f.WriteObject(&gainMapRef, outName);
}

int CalibPadGainTracks::getIndex(o2::tpc::PadSubset padSub, int padSubsetNumber, const int row, const int pad)
{
  return Mapper::instance().getPadNumber(padSub, padSubsetNumber, row, pad);
}

//______________________________________________
void CalibPadGainTracks::setTPCVDrift(const o2::tpc::VDriftCorrFact& v)
{
  mTPCVDrift = v.refVDrift * v.corrFact;
  mTPCVDriftCorrFact = v.corrFact;
  mTPCVDriftRef = v.refVDrift;
}

//______________________________________________
void CalibPadGainTracks::setTPCCorrMaps(o2::gpu::CorrectionMapsHelper* maph)
{
  mTPCCorrMapsHelper = maph;
}
