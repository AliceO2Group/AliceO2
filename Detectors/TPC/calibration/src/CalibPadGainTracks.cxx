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
/// @file   CalibPadGainTracks.h
/// @author Matthias Kleiner, matthias.kleiner@cern.ch
///

#include "TPCCalibration/CalibPadGainTracks.h"
#include "TPCBase/PadPos.h"
#include "TPCBase/ROC.h"

// root includes
#include "TFile.h"

using namespace o2::tpc;

void CalibPadGainTracks::processTracks()
{
  for (const auto& trk : *mTracks) {
    processTrack(trk);
  }
}

void CalibPadGainTracks::processTrack(o2::tpc::TrackTPC track)
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

    unsigned char sectorIndex = 0;
    unsigned char rowIndex = 0;
    unsigned int clusterIndexNumb = 0;

    // this function sets sectorIndex, rowIndex, clusterIndexNumb
    track.getClusterReference(*mTPCTrackClIdxVecInput, iCl, sectorIndex, rowIndex, clusterIndexNumb);
    const float xPosition = mapper.getPadCentre(PadPos(rowIndex, 0)).X();
    const bool check = track.propagateTo(xPosition, mField); // propagate this track to the plane X=xk (cm) in the field "b" (kG)
    if (!check) {
      continue;
    }

    const int region = Mapper::REGION[rowIndex];
    const float effectiveLength = mCalibTrackTopologyPol ? getTrackTopologyCorrectionPol(track, cl, region) : getTrackTopologyCorrection(track, region);

    const unsigned char pad = static_cast<unsigned char>(cl.getPad() + 0.5f); // the left side of the pad ist defined at e.g. 3.5 and the right side at 4.5
    const float gain = mGainMapRef ? mGainMapRef->getValue(sectorIndex, rowIndex, pad) : 1;
    const float chargeNorm = cl.qMax / (effectiveLength * gain);

    if (mMode == dedxTracking) {
      const auto& dEdx = track.getdEdx();
      float dedx = -1;
      const CRU cru(Sector(sectorIndex), region);

      if (mDedxRegion == stack) {
        const auto stack = cru.gemStack();
        if (stack == GEMstack::IROCgem && dEdx.NHitsIROC > mMinClusters) {
          dedx = dEdx.dEdxMaxIROC;
        } else if (stack == GEMstack::OROC1gem && dEdx.dEdxMaxOROC1 > mMinClusters) {
          dedx = dEdx.dEdxMaxOROC1;
        } else if (stack == GEMstack::OROC2gem && dEdx.dEdxMaxOROC2 > mMinClusters) {
          dedx = dEdx.dEdxMaxOROC2;
        } else if (stack == GEMstack::OROC3gem && dEdx.dEdxMaxOROC3 > mMinClusters) {
          dedx = dEdx.dEdxMaxOROC3;
        }
      } else if (mDedxRegion == chamber) {
        if (cru.isIROC() && dEdx.NHitsIROC > mMinClusters) {
          dedx = dEdx.dEdxMaxIROC;
        } else {
          int count = 0;
          if (dEdx.NHitsOROC1 > mMinClusters) {
            dedx += dEdx.dEdxMaxOROC1;
            ++count;
          }
          if (dEdx.NHitsOROC2 > mMinClusters) {
            dedx += dEdx.dEdxMaxOROC2;
            ++count;
          }
          if (dEdx.NHitsOROC3 > mMinClusters) {
            dedx += dEdx.dEdxMaxOROC3;
            ++count;
          }
          if (count > 0) {
            dedx /= count;
          }
        }
      } else if (mDedxRegion == sector) {
        dedx = dEdx.dEdxMaxTPC;
      }

      if (dedx <= 0) {
        continue;
      }

      const float fillVal = chargeNorm / dedx;
      int index = Mapper::GLOBALPADOFFSET[region] + Mapper::OFFSETCRUGLOBAL[rowIndex] + pad;
      if (cru.isOROC()) {
        index -= Mapper::getPadsInIROC();
      }

      fillPadByPadHistogram(cru.roc().getRoc(), index, fillVal);
    }

    if (mMode == dedxTrack) {
      const int indexBuffer = getdEdxBufferIndex(region);
      mDEdxBuffer[indexBuffer].emplace_back(chargeNorm);
      mClTrk.emplace_back(std::make_tuple(sectorIndex, rowIndex, pad, chargeNorm)); // fill with dummy dedx value
    }
  }

  if (mMode == dedxTrack) {
    const auto dedx = getTruncMean(mDEdxBuffer);

    // set the dEdx
    for (auto& x : mClTrk) {
      const unsigned char globRow = std::get<1>(x);
      const int region = Mapper::REGION[globRow];
      const int indexBuffer = getdEdxBufferIndex(region);

      const float dedxTmp = dedx[indexBuffer];
      if (dedxTmp <= 0) {
        continue;
      }

      const unsigned char pad = std::get<2>(x);
      int index = Mapper::GLOBALPADOFFSET[region] + Mapper::OFFSETCRUGLOBAL[globRow] + pad;
      const ROC roc = CRU(Sector(std::get<0>(x)), region).roc();
      if (roc.isOROC()) {
        index -= Mapper::getPadsInIROC();
      }

      // fill the normalizes charge in pad histogram
      const float fillVal = std::get<3>(x) / dedxTmp;
      fillPadByPadHistogram(roc.getRoc(), index, fillVal);
    }
  } else {
  }
}

std::vector<float> CalibPadGainTracks::getTruncMean(std::vector<std::vector<float>>& vCharge, float low, float high) const
{
  std::vector<float> dedx;
  dedx.reserve(vCharge.size());
  // returns the truncated mean for input vector
  for (auto& charge : vCharge) {
    const int nClustersUsed = static_cast<int>(charge.size());
    if (nClustersUsed < mMinClusters) {
      dedx.emplace_back(-1);
      continue;
    }

    std::sort(charge.begin(), charge.end()); // sort the vector for performing truncated mean

    const int startInd = static_cast<int>(low * nClustersUsed);
    const int endInd = static_cast<int>(high * nClustersUsed);

    if (endInd <= startInd) {
      dedx.emplace_back(-1);
      continue;
    }

    const float dEdx = std::accumulate(charge.begin() + startInd, charge.begin() + endInd, 0.f);
    const int nClustersTrunc = endInd - startInd; // count number of clusters
    dedx.emplace_back(dEdx / nClustersTrunc);
  }
  return dedx;
}

float CalibPadGainTracks::getTrackTopologyCorrection(const o2::tpc::TrackTPC& track, const unsigned int region) const
{
  const float padLength = mapper.getPadRegionInfo(region).getPadHeight();
  const float sinPhi = track.getSnp();
  const float tgl = track.getTgl();
  const float snp2 = sinPhi * sinPhi;
  const float effectiveLength = padLength * std::sqrt((1 + tgl * tgl) / (1 - snp2)); // calculate the trace length of the track over the pad
  return effectiveLength;
}

float CalibPadGainTracks::getTrackTopologyCorrectionPol(const o2::tpc::TrackTPC& track, const o2::tpc::ClusterNative& cl, const unsigned int region) const
{
  const float trackSnp = track.getSnp();
  const float maxSnp = mCalibTrackTopologyPol->getMaxSinPhi();
  float snp = std::abs(trackSnp);
  if (snp > maxSnp) {
    snp = maxSnp;
  }

  float snp2 = trackSnp * trackSnp;
  const float sec2 = 1.f / (1.f - snp2);
  const float trackTgl = track.getTgl();
  const float tgl2 = trackTgl * trackTgl;
  float tanTheta = std::sqrt(tgl2 * sec2);
  const float maxTanTheta = mCalibTrackTopologyPol->getMaxTanTheta();
  if (tanTheta > maxTanTheta) {
    tanTheta = maxTanTheta;
  }

  const float z = std::abs(track.getParam(1));
  const float padTmp = cl.getPad();
  const float absRelPad = std::abs(padTmp - int(padTmp + 0.5f));
  const float relTime = cl.getTime() - int(cl.getTime() + 0.5f);
  const float effectiveLength = mCalibTrackTopologyPol->getCorrectionqMax(region, tanTheta, snp, z, absRelPad, relTime);
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
    mDEdxBuffer[0].reserve(mapper.getNumberOfRows());
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
