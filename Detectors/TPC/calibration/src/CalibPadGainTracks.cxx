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
#include "TPCCalibration/IDCDrawHelper.h"
#include "TPCBase/PadPos.h"
#include "TPCBase/ROC.h"
#include "TPCBase/Painter.h"
#include "CommonUtils/TreeStreamRedirector.h"

//root includes
#include "TFile.h"
#include "TTree.h"
#include "TCanvas.h"

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

  float effectiveLengthLastCluster = 0; // set the effective length of the track over pad for the last cluster

  for (int iCl = 0; iCl < nClusters; iCl++) { //loop over cluster
    const float effectiveLength = getTrackTopologyCorrection(track, iCl);
    mCLNat.emplace_back(track.getCluster(*mTPCTrackClIdxVecInput, iCl, *mClusterIndex));
    unsigned char sectorIndex = 0;
    unsigned char rowIndex = 0;
    unsigned int clusterIndexNumb = 0;

    // this function sets sectorIndex, rowIndex, clusterIndexNumb
    track.getClusterReference(*mTPCTrackClIdxVecInput, iCl, sectorIndex, rowIndex, clusterIndexNumb);
    const float charge = mCLNat[static_cast<unsigned int>(iCl)].qMax;
    const unsigned char pad = static_cast<unsigned char>(mCLNat[static_cast<unsigned int>(iCl)].getPad() + 0.5f); // the left side of the pad ist defined at e.g. 3.5 and the right side at 4.5

    // propagateTo delivers sometimes wrong values! break if the effLength doesnt change
    // TODO check this
    if (effectiveLength - effectiveLengthLastCluster == 0) {
      break;
    }

    // fill IROC dedx
    if (rowIndex < mapper.getNumberOfRowsROC(0)) {
      mDEdxIROC.emplace_back(charge / effectiveLength);
    } else {
      mDEdxOROC.emplace_back(charge / effectiveLength);
    }
    mClTrk.emplace_back(std::make_tuple(sectorIndex, rowIndex, pad, charge / effectiveLength)); // fill with dummy dedx value
    effectiveLengthLastCluster = effectiveLength;
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
      ROC roc(std::get<0>(x), roctype);
      const o2::tpc::PadSubset sub = mPadHistosDet->getPadSubset();
      const int num = mPadHistosDet->getCalArray(roc.getRoc()).getPadSubsetNumber();
      const int rowinROC = roctype == RocType::IROC ? std::get<1>(x) : std::get<1>(x) - rowsIROC;
      auto index = static_cast<unsigned int>(getIndex(sub, num, rowinROC, pad)); // index=globalpadnumber
      //fill the normalizes charge in pad histogram
      mPadHistosDet->getCalArray(roc.getRoc()).getData()[index].fill(fillVal);
    }
  } else {
  }
}

void CalibPadGainTracks::dumpToTree(const char* outFileName) const
{
  o2::utils::TreeStreamRedirector pcstream(outFileName, "RECREATE");
  pcstream.GetFile()->cd();

  unsigned char iROC = 0; // roc: 0...71
  for (auto& calArray : mPadHistosDet->getData()) {
    o2::tpc::ROC roc(iROC);
    GlobalPadNumber globPadNumber = roc.rocType() == RocType::OROC ? Mapper::getPadsInIROC() : 0;

    for (auto& val : calArray.getData()) {
      const PadPos pos = mapper.padPos(globPadNumber);
      int padRow = pos.getRow();
      int pad = pos.getPad();
      int sector = roc.getSector();
      pcstream << "tree"
               << "hist=" << val
               << "padRow=" << padRow
               << "pad=" << pad
               << "sector=" << sector
               << "\n";

      ++globPadNumber;
    }
    ++iROC;
  }
  pcstream.Close();
}

float CalibPadGainTracks::getTruncMean(std::vector<float>& vCharge, float low, float high) const
{
  // returns the truncated mean for input vector
  std::sort(vCharge.begin(), vCharge.end()); //sort the vector for performing truncated mean

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

void CalibPadGainTracks::fillgainMap()
{
  //fill the gain values in CalPad object
  unsigned long int iROC = 0; // roc: 0...71
  for (auto& calArray : mGainMap.getData()) {
    unsigned int pad = 0; // pad in roc
    for (auto& val : calArray.getData()) {
      const auto entries = mPadHistosDet->getCalArray(iROC).getData()[pad].getEntries();
      val = entries < mMinEntries ? 0 : mPadHistosDet->getCalArray(iROC).getData()[pad].getStatisticsData(mLowTruncation, mUpTruncation).mCOG;
      pad++;
    }
    iROC++;
  }
  LOG(info) << "GAINMAP SUCCESFULLY FILLED";
}

void CalibPadGainTracks::dumpGainMap(const char* fileName) const
{
  TFile f(fileName, "RECREATE");
  f.WriteObject(&mGainMap, "GainMap");
}

float CalibPadGainTracks::getTrackTopologyCorrection(o2::tpc::TrackTPC& track, int iCl)
{
  unsigned char sectorIndex = 0;
  unsigned char rowIndex = 0;
  unsigned int clusterIndexNumb = 0;
  // this function sets sectorIndex, rowIndex, clusterIndexNumb
  track.getClusterReference(*mTPCTrackClIdxVecInput, iCl, sectorIndex, rowIndex, clusterIndexNumb);

  const PadRegionInfo& region = mapper.getPadRegionInfo(Mapper::REGION[rowIndex]);
  const float padLength = region.getPadHeight();

  // to correct the cluster charge for the track topology, the track parameters have to be propagated to the x position if the cluster
  const float xPosition = mapper.getPadCentre(PadPos(rowIndex, 0)).X();
  track.propagateTo(xPosition, mField); // propagate this track to the plane X=xk (cm) in the field "b" (kG)
  const float sinPhi = track.getSnp();
  const float tgl = track.getTgl();
  const float snp2 = sinPhi * sinPhi;
  const float effectiveLength = padLength * std::sqrt((1 + tgl * tgl) / (1 - snp2)); // calculate the trace length of the track over the pad
  return effectiveLength;
}

void CalibPadGainTracks::init(const unsigned int nBins, const float xmin, const float xmax, const bool useUnderflow, const bool useOverflow)
{
  o2::tpc::FastHisto<float> hist(nBins, xmin, xmax, useUnderflow, useOverflow);
  initDefault();
  for (auto& calArray : mPadHistosDet->getData()) {
    for (auto& tHist : calArray.getData()) {
      tHist = hist;
    }
  }
}

void CalibPadGainTracks::initDefault()
{
  mDEdxIROC.reserve(Mapper::getNumberOfRowsInIROC());
  mDEdxOROC.reserve(Mapper::getNumberOfRowsInOROC());
  mCLNat.reserve(Mapper::PADROWS);
  mClTrk.reserve(Mapper::PADROWS);
  mPadHistosDet = std::make_unique<o2::tpc::CalDet<o2::tpc::FastHisto<float>>>("Histo");
}

void CalibPadGainTracks::drawExtractedGainMapHelper(const bool type, const Sector sector, const std::string filename, const float minZ, const float maxZ) const
{
  std::function<float(const unsigned int, const unsigned int, const unsigned int, const unsigned int)> idcFunc = [this](const unsigned int sector, const unsigned int region, const unsigned int lrow, const unsigned int pad) {
    return this->mGainMap.getValue(sector, Mapper::getGlobalPadNumber(lrow, pad, region));
  };

  IDCDrawHelper::IDCDraw drawFun;
  drawFun.mIDCFunc = idcFunc;
  const std::string zAxisTitle = "rel. gain";
  type ? IDCDrawHelper::drawSide(drawFun, sector.side(), zAxisTitle, filename, minZ, maxZ) : IDCDrawHelper::drawSector(drawFun, 0, Mapper::NREGIONS, sector, zAxisTitle, filename, minZ, maxZ);
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

void CalibPadGainTracks::setTruncationRange(const float low, const float high)
{
  mLowTruncation = low;
  mUpTruncation = high;
}

void CalibPadGainTracks::divideGainMap(const char* inpFile, const char* mapName)
{
  TFile f(inpFile, "READ");
  o2::tpc::CalPad* gainMap = nullptr;
  f.GetObject(mapName, gainMap);

  if (!gainMap) {
    LOGP(info, "GainMap {} not found returning", mapName);
    return;
  }

  mGainMap /= *gainMap;
  delete gainMap;
}

void CalibPadGainTracks::setGainMap(const char* inpFile, const char* mapName)
{
  TFile f(inpFile, "READ");
  o2::tpc::CalPad* gainMap = nullptr;
  f.GetObject(mapName, gainMap);

  if (!gainMap) {
    LOGP(info, "GainMap {} not found returning", mapName);
    return;
  }

  mGainMap = *gainMap;
  delete gainMap;
}

TCanvas* CalibPadGainTracks::drawExtractedGainMapPainter() const
{
  return painter::draw(mGainMap);
}

void CalibPadGainTracks::resetHistos()
{
  for (auto& calArray : mPadHistosDet->getData()) {
    for (auto& tHist : calArray.getData()) {
      tHist.reset();
    }
  }
}
