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

using namespace o2::tpc;

void CalibPadGainTracks::processTracks(bool bwriteTree, float momMin, float momMax)
{
  for (const auto& trk : *mTracks) {
    processTrack(trk, momMin, momMax);
  }
  if (bwriteTree) {
    writeTree();
  }
}

void CalibPadGainTracks::processTrack(o2::tpc::TrackTPC track, float momMin, float momMax)
{
  // make momentum cut
  const float mom = track.getP();
  if (mom < momMin || mom > momMax) {
    return;
  }

  std::array<std::vector<float>, 2> vTempdEdx{}; // this vector is filled with the cluster charge and used for dedx calculation
  vTempdEdx[0].reserve(NROWSIROC);
  vTempdEdx[1].reserve(NROWSOROC);
  std::vector<o2::tpc::ClusterNative> clNat{};
  clNat.reserve(NROWS);
  std::vector<std::tuple<unsigned char, unsigned char, unsigned char, float>> tClTrk{}; //temp tuple which will be filled
  tClTrk.reserve(NROWS);
  const int nClusters = track.getNClusterReferences();
  float effectiveLengthLastCluster = 0; // set the effective length of the track over pad for the last cluster

  for (int iCl = 0; iCl < nClusters; iCl++) { //loop over cluster
    const float effectiveLength = getTrackTopologyCorrection(track, iCl);
    clNat.emplace_back(track.getCluster(*mTPCTrackClIdxVecInput, iCl, *mClusterIndex));
    unsigned char sectorIndex = 0;
    unsigned char rowIndex = 0;
    unsigned int clusterIndexNumb = 0;

    // this function sets sectorIndex, rowIndex, clusterIndexNumb
    track.getClusterReference(*mTPCTrackClIdxVecInput, iCl, sectorIndex, rowIndex, clusterIndexNumb);
    const float charge = clNat[static_cast<unsigned int>(iCl)].qMax;
    const unsigned char pad = static_cast<unsigned char>(clNat[static_cast<unsigned int>(iCl)].getPad() + 0.5f); // the left side of the pad ist defined at e.g. 3.5 and the right side at 4.5

    // propagateTo delivers sometimes wrong values! break if the effLength doesnt change
    // TODO check this
    if (effectiveLength - effectiveLengthLastCluster == 0) {
      break;
    }

    // fill IROC dedx
    if (rowIndex < mapper.getNumberOfRowsROC(0)) {
      vTempdEdx[0].emplace_back(charge / effectiveLength);
    } else {
      vTempdEdx[1].emplace_back(charge / effectiveLength);
    }
    tClTrk.emplace_back(std::make_tuple(sectorIndex, rowIndex, pad, charge / effectiveLength)); // fill with dummy dedx value
    effectiveLengthLastCluster = effectiveLength;
  }

  // use dedx from track as reference
  if (mMode == DedxTrack) {
    const float dedxIROC = getTruncMean(vTempdEdx[0]);
    const float dedxOROC = getTruncMean(vTempdEdx[1]);

    // set the dEdx
    for (auto& x : tClTrk) {
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

void CalibPadGainTracks::writeTree() const
{
  // loop over the tuple and fill the entries in a tree
  TFile file("debug.root", "RECREATE");
  TTree tCluster("debug", "debug");
  int sector{};
  int padRow{};
  unsigned int pad{};
  o2::tpc::FastHisto<float> fastHisto{};

  tCluster.Branch("sector", &sector);
  tCluster.Branch("padRow", &padRow);
  tCluster.Branch("pad", &pad);
  tCluster.Branch("hist", &fastHisto);

  unsigned char iROC = 0; // roc: 0...71
  for (auto& calArray : mPadHistosDet->getData()) {
    o2::tpc::ROC roc(iROC);
    GlobalPadNumber globPadNumber = roc.rocType() == RocType::OROC ? mapper.getPadsInIROC() : 0;

    for (auto& val : calArray.getData()) {
      fastHisto = val;
      const PadPos pos = mapper.padPos(globPadNumber);
      padRow = pos.getRow();
      pad = pos.getPad();
      sector = roc.getSector();
      tCluster.Fill();
      ++globPadNumber;
    }
    ++iROC;
  }
  file.cd();
  tCluster.Write();
  file.Close();
}

float CalibPadGainTracks::getTruncMean(std::vector<float> vCharge, float low, float high) const
{
  // returns the truncated mean for input vector
  std::sort(vCharge.begin(), vCharge.end()); //sort the vector for performing truncated mean
  float dEdx = 0.f;                          // count total dedx
  int nClustersTrunc = 0;                    // count number of clusters
  const int nClustersUsed = static_cast<int>(vCharge.size());

  for (int icl = static_cast<int>(low * nClustersUsed); icl < static_cast<int>(high * nClustersUsed); ++icl) {
    dEdx += vCharge[static_cast<unsigned int>(icl)];
    ++nClustersTrunc;
  }

  if (nClustersTrunc > 0) {
    dEdx /= nClustersTrunc;
  }
  return dEdx;
}

void CalibPadGainTracks::fillgainMap()
{
  //fill the gain values in CalPad object
  unsigned long int iROC = 0; // roc: 0...71
  for (auto& calArray : mGainMap.getData()) {
    unsigned int pad = 0; // pad in roc
    for (auto& val : calArray.getData()) {
      val = mPadHistosDet->getCalArray(iROC).getData()[pad].getStatisticsData().mCOG;
      pad++;
    }
    iROC++;
  }
  LOG(INFO) << "GAINMAP SUCCESFULLY FILLED";
}

void CalibPadGainTracks::dumpGainMap()
{
  TFile f("GainMap.root", "RECREATE");
  f.WriteObject(&mGainMap, "GainMap");
}

float CalibPadGainTracks::getTrackTopologyCorrection(o2::tpc::TrackTPC& track, int iCl)
{
  unsigned char sectorIndex = 0;
  unsigned char rowIndex = 0;
  unsigned int clusterIndexNumb = 0;
  // this function sets sectorIndex, rowIndex, clusterIndexNumb
  track.getClusterReference(*mTPCTrackClIdxVecInput, iCl, sectorIndex, rowIndex, clusterIndexNumb);

  int nPadRows = 0; // used for getting the current region (e.g. set the padLength and padHeight)
  float padLength = 0;

  // TODO optimize for loop
  for (unsigned char iRegion = 0; iRegion < 10; iRegion++) {
    const PadRegionInfo& region = mapper.getPadRegionInfo(iRegion);
    padLength = region.getPadHeight();
    nPadRows += static_cast<int>(region.getNumberOfPadRows());
    if (static_cast<int>(rowIndex) < nPadRows) {
      break;
    }
  }

  // to correct the cluster charge for the track topology, the track parameters have to be propagated to the x position if the cluster
  const float xPosition = mapper.getPadCentre(PadPos(rowIndex, 0)).X();
  const float bField = 5.00668f;        // magnetic field in "kilo Gaus" // TODO Get b field from o2.
  track.propagateTo(xPosition, bField); // propagate this track to the plane X=xk (cm) in the field "b" (kG)
  const float sinPhi = track.getSnp();
  const float tgl = track.getTgl();
  const float snp2 = sinPhi * sinPhi;
  const float effectiveLength = padLength * std::sqrt((1 + tgl * tgl) / (1 - snp2)); // calculate the trace length of the track over the pad
  return effectiveLength;
}
