// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

/// \file TPCCATracking.cxx
/// \author David Rohr

#include "TPCReconstruction/TPCCATracking.h"

#include "DetectorsBase/Track.h"
#include "FairLogger.h"
#include "TClonesArray.h"
#include "TChain.h"
#include "TPCBase/Mapper.h"
#include "TPCBase/PadRegionInfo.h"
#include "TPCBase/Sector.h"
#include "TPCReconstruction/TrackTPC.h"
#include "TPCReconstruction/Cluster.h"
#include "TPCBase/ParameterDetector.h"
#include "TPCBase/ParameterGas.h"
#include "TPCBase/ParameterElectronics.h"
#include "TPCBase/Sector.h"

//The AliHLTTPCCAO2Interface.h needs certain macro definitions.
//The AliHLTTPCCAO2Interface will only be included once here, all O2 TPC tracking will run through this TPCCATracking class.
//Therefore, the macros are defined here and not globally, in order not to pollute the global namespace
#define HLTCA_STANDALONE 
#define HLTCA_TPC_GEOMETRY_O2

//This class is only a wrapper for the actual tracking contained in the HLT O2 CA Tracking library.
#include "AliHLTTPCCAO2Interface.h"

using namespace o2::TPC;

TPCCATracking::TPCCATracking() : mTrackingCAO2Interface(), mClusterData_UPTR(), mClusterData(nullptr) {
}

TPCCATracking::~TPCCATracking() {
  deinitialize();
}

int TPCCATracking::initialize(const char* options) {
  mTrackingCAO2Interface.reset(new AliHLTTPCCAO2Interface);
  int retVal = mTrackingCAO2Interface->Initialize(options);
  if (retVal) {
    mTrackingCAO2Interface.reset();
  } else {
    mClusterData_UPTR.reset(new AliHLTTPCCAClusterData[Sector::MAXSECTOR]);
    mClusterData = mClusterData_UPTR.get();
  }
  return (retVal);
}

void TPCCATracking::deinitialize() {
  mTrackingCAO2Interface.reset();
  mClusterData_UPTR.reset();
  mClusterData = nullptr;
}

int TPCCATracking::runTracking(TChain* inputClustersChain, const std::vector<o2::TPC::Cluster>* inputClustersArray, std::vector<TrackTPC>* outputTracks) {
  if (mTrackingCAO2Interface == nullptr) return (1);
  if (inputClustersChain && inputClustersArray) {
    LOG(FATAL) << "Internal error, must not pass in both TChain and TClonesArray of clusters\n";
  }

  int retVal = 0;
  const static ParameterDetector &detParam = ParameterDetector::defaultInstance();
  const static ParameterGas &gasParam = ParameterGas::defaultInstance();
  const static ParameterElectronics &elParam = ParameterElectronics::defaultInstance();
  
  std::vector<Cluster> clusterCache;

  int nClusters[Sector::MAXSECTOR] = {};
  int nClustersConverted = 0;
  int nClustersTotal = 0;
  int numChunks;
  if (inputClustersChain) {
    inputClustersChain->SetBranchAddress("TPCClusterHW", &inputClustersArray);
    numChunks = inputClustersChain->GetEntries();
  } else {
    numChunks = 1;
  }
  
  for (int iChunk = 0;iChunk < numChunks;iChunk++) {
    if (inputClustersChain) {
      inputClustersChain->GetEntry(iChunk);
    }
    for (const auto& cluster : *inputClustersArray) {
      const Sector sector = CRU(cluster.getCRU()).sector();
      nClusters[sector.getSector()]++;
    }
    nClustersTotal += inputClustersArray->size();
    if (inputClustersChain) clusterCache.resize(nClustersTotal);
    for (int i = 0; i < Sector::MAXSECTOR; i++) {
      if (iChunk == 0) {
        mClusterData[i].StartReading(i, nClusters[i]);
      } else {
        mClusterData[i].Allocate(mClusterData[i].NumberOfClusters() + nClusters[i]);
      }
    }
    for (Int_t icluster = 0; icluster < inputClustersArray->size(); ++icluster) {
      const auto& cluster = (*inputClustersArray)[icluster];
      const CRU cru(cluster.getCRU());
      const Sector sector = cru.sector();
      AliHLTTPCCAClusterData& cd = mClusterData[sector.getSector()];
      AliHLTTPCCAClusterData::Data& hltCluster = cd.Clusters()[cd.NumberOfClusters()];

      // ===| mapper |==============================================================
      Mapper& mapper = Mapper::instance();
      const PadRegionInfo& region = mapper.getPadRegionInfo(cru.region());
      const int rowInSector = cluster.getRow() + region.getGlobalRowOffset();
      const float padY = cluster.getPadMean();
      const int padNumber = int(padY);
      const GlobalPadNumber pad = mapper.globalPadNumber(PadPos(rowInSector, padNumber));
      const PadCentre& padCentre = mapper.padCentre(pad);
      const float localY = padCentre.Y() - (padY - padNumber - 0.5) * region.getPadWidth();
      const float localYfactor = (cru.side() == Side::A) ? -1.f : 1.f;
      float zPositionAbs = cluster.getTimeMean()*elParam.getZBinWidth()*gasParam.getVdrift();
      if (!mTrackingCAO2Interface->GetParamContinuous()) zPositionAbs = detParam.getTPClength() - zPositionAbs;

      Point2D<float> clusterPos(padCentre.X(), localY);

      // sanity checks
      if (zPositionAbs < 0 || (!mTrackingCAO2Interface->GetParamContinuous() && zPositionAbs > detParam.getTPClength())) {
        LOG(INFO) << "Removing cluster " << icluster << "/" << inputClustersArray->size() << " time: " << cluster.getTimeMean() << ", abs. z: " << zPositionAbs << "\n";
        continue;
      }

      hltCluster.fX = clusterPos.X();
      hltCluster.fY = clusterPos.Y() * (localYfactor);
      hltCluster.fZ = zPositionAbs * (-localYfactor);
      hltCluster.fRow = rowInSector;
      hltCluster.fAmp = cluster.getQmax();
      if (inputClustersChain) {
        clusterCache[nClustersConverted] = cluster;
        hltCluster.fId = nClustersConverted;
      } else {
        hltCluster.fId = icluster;
      }

      cd.SetNumberOfClusters(cd.NumberOfClusters() + 1);
      nClustersConverted++;
    }
  }
  
  if (nClustersTotal != nClustersConverted) {
    LOG(INFO) << "Passed " << nClustersConverted << " (out of " << nClustersTotal << ") clusters to CA tracker\n";
  }

  const AliHLTTPCGMMergedTrack* tracks;
  int nTracks;
  const AliHLTTPCGMMergedTrackHit* trackClusters;
  retVal = mTrackingCAO2Interface->RunTracking(mClusterData, tracks, nTracks, trackClusters);
  if (retVal == 0)
  {
    std::vector<std::pair<int, float>> trackSort(nTracks);
    int tmp = 0;
    for (int i = 0;i < nTracks;i++)
    {
      if (tracks[i].OK()) trackSort[tmp++] = {i, tracks[i].GetParam().GetZOffset()};
    }
    nTracks = tmp;
    std::sort(trackSort.data(), trackSort.data() + nTracks, [](const auto& a, const auto& b) {
      return(a.second > b.second);
    });
    
    outputTracks->resize(nTracks);
    for (int iTmp = 0; iTmp < nTracks; iTmp++) {
      auto& oTrack = (*outputTracks)[iTmp];
      const int i = trackSort[iTmp].first;
      oTrack = TrackTPC(
        tracks[i].GetParam().GetX(), tracks[i].GetAlpha(),
        { tracks[i].GetParam().GetY(), tracks[i].GetParam().GetZ(), tracks[i].GetParam().GetSinPhi(),
          tracks[i].GetParam().GetDzDs(), tracks[i].GetParam().GetQPt() },
        { tracks[i].GetParam().GetCov(0), tracks[i].GetParam().GetCov(1), tracks[i].GetParam().GetCov(2),
          tracks[i].GetParam().GetCov(3), tracks[i].GetParam().GetCov(4), tracks[i].GetParam().GetCov(5),
          tracks[i].GetParam().GetCov(6), tracks[i].GetParam().GetCov(7), tracks[i].GetParam().GetCov(8),
          tracks[i].GetParam().GetCov(9), tracks[i].GetParam().GetCov(10), tracks[i].GetParam().GetCov(11),
          tracks[i].GetParam().GetCov(12), tracks[i].GetParam().GetCov(13), tracks[i].GetParam().GetCov(14) });
      oTrack.setTime0(tracks[i].GetParam().GetZOffset() / (elParam.getZBinWidth()*gasParam.getVdrift()));
      oTrack.setLastClusterZ(trackClusters[tracks[i].FirstClusterRef() + tracks[i].NClusters() - 1].fZ - tracks[i].GetParam().GetZOffset());
      for (int j = 0; j < tracks[i].NClusters(); j++) {
        if (inputClustersChain) {
          oTrack.addCluster(clusterCache[trackClusters[tracks[i].FirstClusterRef() + j].fId]);
        } else {
          oTrack.addCluster((*inputClustersArray)[trackClusters[tracks[i].FirstClusterRef() + j].fId]);
        }
      }
    }
  }
  mTrackingCAO2Interface->Cleanup();
  return (retVal);
}
