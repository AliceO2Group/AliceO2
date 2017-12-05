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
using namespace o2::DataFormat::TPC;

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

int TPCCATracking::convertClusters(TChain* inputClustersChain, const std::vector<o2::TPC::Cluster>* inputClustersArray, ClusterNativeAccessFullTPC& outputClusters, std::unique_ptr<ClusterNative[]>& clusterMemory) {
  if (inputClustersChain && inputClustersArray) {
    LOG(FATAL) << "Internal error, must not pass in both TChain and TClonesArray of clusters\n";
  }
  int numChunks;
  if (inputClustersChain) {
    inputClustersChain->SetBranchAddress("TPCClusterHW", &inputClustersArray);
    numChunks = inputClustersChain->GetEntries();
  } else {
    numChunks = 1;
  }
  
  Mapper& mapper = Mapper::instance();
  for (int iter = 0;iter < 2;iter++)
  {
    for (int i = 0;i < Sector::MAXSECTOR;i++) {
      for (int j = 0;j < Constants::MAXGLOBALPADROW;j++) {
        outputClusters.mNClusters[i][j] = 0;
      }
    }

    unsigned int nClusters = 0;
    for (int iChunk = 0;iChunk < numChunks;iChunk++) {
      if (inputClustersChain) {
        inputClustersChain->GetEntry(iChunk);
      }
      for (int icluster = 0; icluster < inputClustersArray->size(); ++icluster) {
        const auto& cluster = (*inputClustersArray)[icluster];
        const CRU cru(cluster.getCRU());
        const Sector sector = cru.sector();
        const PadRegionInfo& region = mapper.getPadRegionInfo(cru.region());
        const int rowInSector = cluster.getRow() + region.getGlobalRowOffset();

        if (iter == 1) {
          ClusterNative& oCluster = outputClusters.mClusters[sector][rowInSector][outputClusters.mNClusters[sector][rowInSector]];
          oCluster.setTimeFlags(cluster.getTimeMean(), 0);
          oCluster.setPad(cluster.getPadMean());
          oCluster.setSigmaTime(cluster.getTimeSigma());
          oCluster.setSigmaPad(cluster.getPadSigma());
          oCluster.mQTot = cluster.getQmax();
          oCluster.mQMax = cluster.getQ();
        }
        outputClusters.mNClusters[sector][rowInSector]++;
      }
      nClusters += inputClustersArray->size();
    }
    if (iter == 0)
    {
      clusterMemory.reset(new ClusterNative[nClusters]);
      unsigned int pos = 0;
      for (int i = 0;i < Sector::MAXSECTOR;i++) {
        for (int j = 0;j < Constants::MAXGLOBALPADROW;j++) {
          outputClusters.mClusters[i][j] = &clusterMemory[pos];
          pos += outputClusters.mNClusters[i][j];
        }
      }
    }
  }
  for (int i = 0;i < Sector::MAXSECTOR;i++) {
    for (int j = 0;j < Constants::MAXGLOBALPADROW;j++) {
      std::sort(outputClusters.mClusters[i][j], outputClusters.mClusters[i][j] + outputClusters.mNClusters[i][j], ClusterNativeContainer::sortComparison);
    }
  }

  return(0);
}

int TPCCATracking::runTracking(const ClusterNativeAccessFullTPC& clusters, std::vector<TrackTPC>* outputTracks) {
  const static ParameterDetector &detParam = ParameterDetector::defaultInstance();
  const static ParameterGas &gasParam = ParameterGas::defaultInstance();
  const static ParameterElectronics &elParam = ParameterElectronics::defaultInstance();
  
  static const float continuousTFReferenceLength = 0.023 * 5e6;
  
  int nClusters[Sector::MAXSECTOR] = {};
  int nClustersConverted = 0;
  int nClustersTotal = 0;
  
  
  Mapper& mapper = Mapper::instance();
  for (int i = 0; i< Sector::MAXSECTOR;i++) {
    for (int j = 0;j < Constants::MAXGLOBALPADROW;j++) {
      if (clusters.mNClusters[i][j] > 0xFFFF) {
        LOG(ERROR) << "Number of clusters in sector " << i << " row " << j << " exceeds 0xFFFF, which is currently a hard limit of the wrapper for the tracking!\n";
        return(1);
      }
      nClusters[i] += clusters.mNClusters[i][j];
    }
    nClustersTotal += nClusters[i];

    mClusterData[i].StartReading(i, nClusters[i]);
  
    for (int j = 0;j < Constants::MAXGLOBALPADROW;j++) {
      Sector sector = i;
      int regionNumber = 0;
      while (j > mapper.getGlobalRowOffsetRegion(regionNumber) + mapper.getNumberOfRowsRegion(regionNumber)) regionNumber++;
      CRU cru(sector, regionNumber);
      const PadRegionInfo& region = mapper.getPadRegionInfo(cru.region());
      for (int k = 0;k < clusters.mNClusters[i][j];k++) {
        const ClusterNative& cluster = clusters.mClusters[i][j][k];
        AliHLTTPCCAClusterData& cd = mClusterData[i];
        AliHLTTPCCAClusterData::Data& hltCluster = cd.Clusters()[cd.NumberOfClusters()];

        const float padY = cluster.getPad();
        const int padNumber = int(padY);
        const GlobalPadNumber pad = mapper.globalPadNumber(PadPos(j, padNumber));
        const PadCentre& padCentre = mapper.padCentre(pad);
        const float localY = padCentre.Y() - (padY - padNumber - 0.5) * region.getPadWidth();
        const float localYfactor = (cru.side() == Side::A) ? -1.f : 1.f;
        float zPositionAbs = cluster.getTime()*elParam.getZBinWidth()*gasParam.getVdrift();
        if (!mTrackingCAO2Interface->GetParamContinuous())
          zPositionAbs = detParam.getTPClength() - zPositionAbs;
        else
          zPositionAbs = continuousTFReferenceLength * elParam.getZBinWidth()*gasParam.getVdrift() - zPositionAbs;

        // sanity checks
        if (zPositionAbs < 0 || (!mTrackingCAO2Interface->GetParamContinuous() && zPositionAbs > detParam.getTPClength())) {
          LOG(INFO) << "Removing cluster " << i << " time: " << cluster.getTime() << ", abs. z: " << zPositionAbs << "\n";
          continue;
        }

        hltCluster.fX = padCentre.X();
        hltCluster.fY = localY * (localYfactor);
        hltCluster.fZ = zPositionAbs * (-localYfactor);
        hltCluster.fRow = j;
        hltCluster.fAmp = cluster.mQMax;
        hltCluster.fId = (i << 24) | (j << 16) | (k);

        cd.SetNumberOfClusters(cd.NumberOfClusters() + 1);
        nClustersConverted++;
      }
    }
  }
  
  if (nClustersTotal != nClustersConverted) {
    LOG(INFO) << "Passed " << nClustersConverted << " (out of " << nClustersTotal << ") clusters to CA tracker\n";
  }

  const AliHLTTPCGMMergedTrack* tracks;
  int nTracks;
  const AliHLTTPCGMMergedTrackHit* trackClusters;
  int retVal = mTrackingCAO2Interface->RunTracking(mClusterData, tracks, nTracks, trackClusters);
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
          
      if (!mTrackingCAO2Interface->GetParamContinuous())
        oTrack.setTime0(0);
      else
        oTrack.setTime0(continuousTFReferenceLength - tracks[i].GetParam().GetZOffset() / (elParam.getZBinWidth()*gasParam.getVdrift()));
      oTrack.setLastClusterZ(trackClusters[tracks[i].FirstClusterRef() + tracks[i].NClusters() - 1].fZ - tracks[i].GetParam().GetZOffset());
      oTrack.resetClusterReferences(tracks[i].NClusters());
      for (int j = 0; j < tracks[i].NClusters(); j++) {
        int clusterId = trackClusters[tracks[i].FirstClusterRef() + j].fId;
        Sector sector = clusterId >> 24;
        int globalRow = (clusterId >> 16) & 0xFF;
        const ClusterNative& cl = clusters.mClusters[sector][globalRow][clusterId & 0xFFFF];
        int regionNumber = 0;
        while (globalRow > mapper.getGlobalRowOffsetRegion(regionNumber) + mapper.getNumberOfRowsRegion(regionNumber)) regionNumber++;
        CRU cru(sector, regionNumber);
        oTrack.addCluster(Cluster(cru, globalRow - mapper.getGlobalRowOffsetRegion(regionNumber), cl.mQTot, cl.mQMax, cl.getPad(), cl.getSigmaPad(), cl.getTime(), cl.getSigmaTime()));
        oTrack.setClusterReference(j, sector, globalRow, clusterId);
      }
    }
  }
  mTrackingCAO2Interface->Cleanup();
  return (retVal);
}

int TPCCATracking::runTracking(TChain* inputClustersChain, const std::vector<o2::TPC::Cluster>* inputClustersArray, std::vector<TrackTPC>* outputTracks) {
  if (mTrackingCAO2Interface == nullptr) return (1);
  int retVal = 0;
  
  std::unique_ptr<ClusterNative[]> clusterMemory;
  ClusterNativeAccessFullTPC clusters;
  retVal = convertClusters(inputClustersChain, inputClustersArray, clusters, clusterMemory);
  if (retVal) return(retVal);
  return(runTracking(clusters, outputTracks));
}

float TPCCATracking::getPseudoVDrift()
{
    const static ParameterGas &gasParam = ParameterGas::defaultInstance();
    const static ParameterElectronics &elParam = ParameterElectronics::defaultInstance();
    return(elParam.getZBinWidth()*gasParam.getVdrift());
}
