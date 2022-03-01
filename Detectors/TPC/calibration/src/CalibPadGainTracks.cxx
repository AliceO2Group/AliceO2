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
  mDEdxIROC.clear();
  mDEdxOROC.clear();
  mCLNat.clear();
  mClTrk.clear();

  for (int iCl = 0; iCl < nClusters; iCl++) { // loop over cluster
    mCLNat.emplace_back(track.getCluster(*mTPCTrackClIdxVecInput, iCl, *mClusterIndex));
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

    const float effectiveLength = getTrackTopologyCorrection(track, rowIndex);
    const unsigned char pad = static_cast<unsigned char>(mCLNat[static_cast<unsigned int>(iCl)].getPad() + 0.5f); // the left side of the pad ist defined at e.g. 3.5 and the right side at 4.5

    const float gain = mGainMapRef ? mGainMapRef->getValue(sectorIndex, rowIndex, pad) : 1;
    const float charge = mCLNat[static_cast<unsigned int>(iCl)].qMax / gain;

    // fill IROC dedx
    const float chargeNorm = charge / effectiveLength;
    if (rowIndex < mapper.getNumberOfRowsROC(0)) {
      mDEdxIROC.emplace_back(chargeNorm);
    } else {
      mDEdxOROC.emplace_back(chargeNorm);
    }
    mClTrk.emplace_back(std::make_tuple(sectorIndex, rowIndex, pad, chargeNorm)); // fill with dummy dedx value
  }

  // use dedx from track as reference
  if (mMode == DedxTrack) {
    const float dedxIROC = getTruncMean(mDEdxIROC);
    const float dedxOROC = getTruncMean(mDEdxOROC);

    // set the dEdx
    for (auto& x : mClTrk) {
      const unsigned char globRow = std::get<1>(x);
      const unsigned char pad = std::get<2>(x);

      // get globalPadNumber (index)
      const auto rowsIROC = mapper.getNumberOfRowsROC(0);
      const auto roctype = (globRow < rowsIROC) ? RocType::IROC : RocType::OROC;
      const float fillVal = (roctype == RocType::IROC) ? (std::get<3>(x) / dedxIROC) : (std::get<3>(x) / dedxOROC);
      const ROC roc(std::get<0>(x), roctype);
      const o2::tpc::PadSubset sub = getHistos()->getPadSubset();
      const int num = getHistos()->getCalArray(roc.getRoc()).getPadSubsetNumber();
      const int rowinROC = roctype == RocType::IROC ? std::get<1>(x) : std::get<1>(x) - rowsIROC;
      const auto index = static_cast<unsigned int>(getIndex(sub, num, rowinROC, pad)); // index=globalpadnumber
      // fill the normalizes charge in pad histogram
      fillPadByPadHistogram(roc.getRoc(), index, fillVal);
    }
  } else {
  }
}

float CalibPadGainTracks::getTruncMean(std::vector<float>& vCharge, float low, float high) const
{
  // returns the truncated mean for input vector
  std::sort(vCharge.begin(), vCharge.end()); // sort the vector for performing truncated mean

  const int nClustersUsed = static_cast<int>(vCharge.size());
  const int startInd = static_cast<int>(low * nClustersUsed);
  const int endInd = static_cast<int>(high * nClustersUsed);

  if (endInd <= startInd) {
    return 0;
  }

  const float dEdx = std::accumulate(vCharge.begin() + startInd, vCharge.begin() + endInd, 0.f);
  const int nClustersTrunc = endInd - startInd; // count number of clusters
  return dEdx / nClustersTrunc;
}

float CalibPadGainTracks::getTrackTopologyCorrection(const o2::tpc::TrackTPC& track, const unsigned char rowIndex) const
{
  const PadRegionInfo& region = mapper.getPadRegionInfo(Mapper::REGION[rowIndex]);
  const float padLength = region.getPadHeight();
  const float sinPhi = track.getSnp();
  const float tgl = track.getTgl();
  const float snp2 = sinPhi * sinPhi;
  const float effectiveLength = padLength * std::sqrt((1 + tgl * tgl) / (1 - snp2)); // calculate the trace length of the track over the pad
  return effectiveLength;
}

void CalibPadGainTracks::reserveMemory()
{
  mDEdxIROC.reserve(Mapper::getNumberOfRowsInIROC());
  mDEdxOROC.reserve(Mapper::getNumberOfRowsInOROC());
  mCLNat.reserve(Mapper::PADROWS);
  mClTrk.reserve(Mapper::PADROWS);
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
