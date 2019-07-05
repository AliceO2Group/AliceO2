// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

/// \file GPUCATracking.cxx
/// \author David Rohr

#include "TPCReconstruction/GPUCATracking.h"

#include "FairLogger.h"
#include "ReconstructionDataFormats/Track.h"
#include "SimulationDataFormat/MCCompLabel.h"
#include "SimulationDataFormat/MCTruthContainer.h"
#include "TChain.h"
#include "TClonesArray.h"
#include "TPCBase/Mapper.h"
#include "TPCBase/PadRegionInfo.h"
#include "TPCBase/ParameterDetector.h"
#include "TPCBase/ParameterElectronics.h"
#include "TPCBase/ParameterGas.h"
#include "TPCBase/Sector.h"

// This class is only a wrapper for the actual tracking contained in the HLT O2 CA Tracking library.
#include "GPUO2Interface.h"
#include "GPUTPCGMMergedTrack.h"
#include "GPUTPCGMMergedTrackHit.h"
#include "GPUO2InterfaceConfiguration.h"

using namespace o2::gpu;
using namespace o2::tpc;
using namespace o2;
using namespace o2::dataformats;

using MCLabelContainer = MCTruthContainer<MCCompLabel>;

GPUCATracking::GPUCATracking() : mTrackingCAO2Interface() {}
GPUCATracking::~GPUCATracking() { deinitialize(); }

int GPUCATracking::initialize(const GPUO2InterfaceConfiguration& config)
{
  mTrackingCAO2Interface.reset(new GPUTPCO2Interface);
  int retVal = mTrackingCAO2Interface->Initialize(config);
  if (retVal) {
    mTrackingCAO2Interface.reset();
  }
  return (retVal);
}

void GPUCATracking::deinitialize()
{
  mTrackingCAO2Interface.reset();
}

int GPUCATracking::runTracking(GPUO2InterfaceIOPtrs* data)
{
  if (data->clusters == nullptr) {
    return 0;
  }

  const ClusterNativeAccessFullTPC& clusters = *data->clusters;
  std::vector<TrackTPC>* outputTracks = data->outputTracks;
  MCLabelContainer* outputTracksMCTruth = data->outputTracksMCTruth;

  if (outputTracks == nullptr) {
    return 0;
  }
  auto& detParam = ParameterDetector::Instance();
  auto& gasParam = ParameterGas::Instance();
  auto& elParam = ParameterElectronics::Instance();
  float vzbin = (elParam.ZbinWidth * gasParam.DriftV);
  float vzbinInv = 1.f / vzbin;
  Mapper& mapper = Mapper::instance();

  GPUTrackingInOutPointers ptrs;
  ptrs.clustersNative = &clusters;
  int retVal = mTrackingCAO2Interface->RunTracking(&ptrs);
  const GPUTPCGMMergedTrack* tracks = ptrs.mergedTracks;
  int nTracks = ptrs.nMergedTracks;
  const GPUTPCGMMergedTrackHit* trackClusters = ptrs.mergedTrackHits;

  if (retVal) {
    return retVal;
  }

  std::vector<std::pair<int, float>> trackSort(nTracks);
  int tmp = 0, tmp2 = 0;
  for (char cside = 0; cside < 2; cside++) {
    for (int i = 0; i < nTracks; i++) {
      if (tracks[i].OK() && tracks[i].CSide() == cside)
        trackSort[tmp++] = { i, (cside == 0 ? 1.f : -1.f) * tracks[i].GetParam().GetZOffset() };
    }
    std::sort(trackSort.data() + tmp2, trackSort.data() + tmp,
              [](const auto& a, const auto& b) { return (a.second > b.second); });
    tmp2 = tmp;
    if (cside == 0)
      mNTracksASide = tmp;
  }
  nTracks = tmp;

  outputTracks->resize(nTracks);

  for (int iTmp = 0; iTmp < nTracks; iTmp++) {
    auto& oTrack = (*outputTracks)[iTmp];
    const int i = trackSort[iTmp].first;
    float time0 = 0.f, tFwd = 0.f, tBwd = 0.f;

    if (mTrackingCAO2Interface->GetParamContinuous()) {
      float zoffset = tracks[i].CSide() ? -tracks[i].GetParam().GetZOffset() : tracks[i].GetParam().GetZOffset();
      time0 = sContinuousTFReferenceLength - zoffset * vzbinInv;

      if (tracks[i].CCE()) {
        bool lastSide = trackClusters[tracks[i].FirstClusterRef()].slice < Sector::MAXSECTOR / 2;
        float delta = 0.f;
        for (int iCl = 1; iCl < tracks[i].NClusters(); iCl++) {
          if (lastSide ^ (trackClusters[tracks[i].FirstClusterRef() + iCl].slice < Sector::MAXSECTOR / 2)) {
            auto& hltcl1 = trackClusters[tracks[i].FirstClusterRef() + iCl];
            auto& hltcl2 = trackClusters[tracks[i].FirstClusterRef() + iCl - 1];
            auto& cl1 = clusters.clusters[hltcl1.slice][hltcl1.row][hltcl1.num];
            auto& cl2 = clusters.clusters[hltcl2.slice][hltcl2.row][hltcl2.num];
            delta = fabs(cl1.getTime() - cl2.getTime()) * 0.5f;
            break;
          }
        }
        tFwd = tBwd = delta;
      } else {
        // estimate max/min time increments which still keep track in the physical limits of the TPC
        float zHigh = trackClusters[tracks[i].FirstClusterRef()].z - tracks[i].GetParam().GetZOffset(); // high R cluster
        float zLow = trackClusters[tracks[i].FirstClusterRef() + tracks[i].NClusters() - 1].z -
                     tracks[i].GetParam().GetZOffset(); // low R cluster

        bool sideHighA = trackClusters[tracks[i].FirstClusterRef()].slice < Sector::MAXSECTOR / 2;
        bool sideLowA =
          trackClusters[tracks[i].FirstClusterRef() + tracks[i].NClusters() - 1].slice < Sector::MAXSECTOR / 2;

        // calculate time bracket
        float zLowAbs = zLow < 0.f ? -zLow : zLow;
        float zHighAbs = zHigh < 0.f ? -zHigh : zHigh;
        //
        // tFwd = (Lmax - max(|zLow|,|zAbs|))/vzbin  = drift time from cluster current Z till endcap
        // tBwd = min(|zLow|,|zAbs|))/vzbin          = drift time from CE till cluster current Z
        //
        if (zLowAbs < zHighAbs) {
          tFwd = (detParam.TPClength - zHighAbs) * vzbinInv;
          tBwd = zLowAbs * vzbinInv;
        } else {
          tFwd = (detParam.TPClength - zLowAbs) * vzbinInv;
          tBwd = zHighAbs * vzbinInv;
        }
      }
    }

    oTrack =
      TrackTPC(tracks[i].GetParam().GetX(), tracks[i].GetAlpha(),
               { tracks[i].GetParam().GetY(), tracks[i].GetParam().GetZ(), tracks[i].GetParam().GetSinPhi(),
                 tracks[i].GetParam().GetDzDs(), tracks[i].GetParam().GetQPt() },
               { tracks[i].GetParam().GetCov(0), tracks[i].GetParam().GetCov(1), tracks[i].GetParam().GetCov(2),
                 tracks[i].GetParam().GetCov(3), tracks[i].GetParam().GetCov(4), tracks[i].GetParam().GetCov(5),
                 tracks[i].GetParam().GetCov(6), tracks[i].GetParam().GetCov(7), tracks[i].GetParam().GetCov(8),
                 tracks[i].GetParam().GetCov(9), tracks[i].GetParam().GetCov(10), tracks[i].GetParam().GetCov(11),
                 tracks[i].GetParam().GetCov(12), tracks[i].GetParam().GetCov(13), tracks[i].GetParam().GetCov(14) });
    oTrack.setTime0(time0);
    oTrack.setDeltaTBwd(tBwd);
    oTrack.setDeltaTFwd(tFwd);
    if (tracks[i].CCE()) {
      oTrack.setHasCSideClusters();
      oTrack.setHasASideClusters();
    } else if (tracks[i].CSide()) {
      oTrack.setHasCSideClusters();
    } else {
      oTrack.setHasASideClusters();
    }

    oTrack.setChi2(tracks[i].GetParam().GetChi2());
    auto& outerPar = tracks[i].OuterParam();
    oTrack.setdEdx(tracks[i].dEdxInfo());
    oTrack.setOuterParam(o2::track::TrackParCov(
      outerPar.X, outerPar.alpha,
      { outerPar.P[0], outerPar.P[1], outerPar.P[2], outerPar.P[3], outerPar.P[4] },
      { outerPar.C[0], outerPar.C[1], outerPar.C[2], outerPar.C[3], outerPar.C[4], outerPar.C[5],
        outerPar.C[6], outerPar.C[7], outerPar.C[8], outerPar.C[9], outerPar.C[10], outerPar.C[11],
        outerPar.C[12], outerPar.C[13], outerPar.C[14] }));
    int nOutCl = 0;
    for (int j = 0; j < tracks[i].NClusters(); j++) {
      if (!(trackClusters[tracks[i].FirstClusterRef() + j].state & GPUTPCGMMergedTrackHit::flagReject)) {
        nOutCl++;
      }
    }
    oTrack.resetClusterReferences(nOutCl);
    std::vector<std::pair<MCCompLabel, unsigned int>> labels;
    nOutCl = 0;
    for (int j = 0; j < tracks[i].NClusters(); j++) {
      if (trackClusters[tracks[i].FirstClusterRef() + j].state & GPUTPCGMMergedTrackHit::flagReject) {
        continue;
      }
      int clusterId = trackClusters[tracks[i].FirstClusterRef() + j].num;
      Sector sector = trackClusters[tracks[i].FirstClusterRef() + j].slice;
      int globalRow = trackClusters[tracks[i].FirstClusterRef() + j].row;
      const ClusterNative& cl = clusters.clusters[sector][globalRow][clusterId];
      int regionNumber = 0;
      while (globalRow > mapper.getGlobalRowOffsetRegion(regionNumber) + mapper.getNumberOfRowsRegion(regionNumber))
        regionNumber++;
      CRU cru(sector, regionNumber);
      oTrack.setClusterReference(nOutCl++, sector, globalRow, clusterId);
      if (outputTracksMCTruth) {
        for (const auto& element : clusters.clustersMCTruth[sector][globalRow]->getLabels(clusterId)) {
          bool found = false;
          for (int l = 0; l < labels.size(); l++) {
            if (labels[l].first == element) {
              labels[l].second++;
              found = true;
              break;
            }
          }
          if (!found)
            labels.emplace_back(element, 1);
        }
      }
    }
    if (outputTracksMCTruth) {
      if (labels.size() == 0) {
        outputTracksMCTruth->addElement(iTmp, MCCompLabel()); //default constructor creates NotSet label
      } else {
        int bestLabelNum = 0, bestLabelCount = 0;
        for (int j = 0; j < labels.size(); j++) {
          if (labels[j].second > bestLabelCount) {
            bestLabelNum = j;
            bestLabelCount = labels[j].second;
          }
        }
        MCCompLabel& bestLabel = labels[bestLabelNum].first;
        if (bestLabelCount < (1.f - sTrackMCMaxFake) * nOutCl) {
          bestLabel.set(-bestLabel.getTrackID(), bestLabel.getEventID(), bestLabel.getSourceID());
        }
        outputTracksMCTruth->addElement(iTmp, bestLabel);
      }
    }
    int lastSector = trackClusters[tracks[i].FirstClusterRef() + tracks[i].NClusters() - 1].slice;
  }

  data->compressedClusters = ptrs.tpcCompressedClusters;
  mTrackingCAO2Interface->Clear(false);

  return (retVal);
}

float GPUCATracking::getPseudoVDrift()
{
  auto& gasParam = ParameterGas::Instance();
  auto& elParam = ParameterElectronics::Instance();
  return (elParam.ZbinWidth * gasParam.DriftV);
}

void GPUCATracking::GetClusterErrors2(int row, float z, float sinPhi, float DzDs, float& ErrY2, float& ErrZ2) const
{
  if (mTrackingCAO2Interface == nullptr) {
    return;
  }
  mTrackingCAO2Interface->GetClusterErrors2(row, z, sinPhi, DzDs, ErrY2, ErrZ2);
}
