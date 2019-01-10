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
#include "TPCReconstruction/TPCFastTransformHelperO2.h"
#include "TPCFastTransform.h"

// This class is only a wrapper for the actual tracking contained in the HLT O2 CA Tracking library.
#include "AliHLTTPCCAO2Interface.h"

using namespace o2::TPC;
using namespace o2;
using namespace o2::dataformats;

using MCLabelContainer = MCTruthContainer<MCCompLabel>;

TPCCATracking::TPCCATracking() : mTrackingCAO2Interface() {}
TPCCATracking::~TPCCATracking() { deinitialize(); }

int TPCCATracking::initialize(const AliGPUCAConfiguration& config)
{
  std::unique_ptr<TPCFastTransform> fastTransform(TPCFastTransformHelperO2::instance()->create(0));
  mTrackingCAO2Interface.reset(new AliHLTTPCCAO2Interface);
  int retVal = mTrackingCAO2Interface->Initialize(config, std::move(fastTransform));
  if (retVal) {
    mTrackingCAO2Interface.reset();
  }
  return (retVal);
}

int TPCCATracking::initialize(const char* options)
{
  std::unique_ptr<TPCFastTransform> fastTransform(TPCFastTransformHelperO2::instance()->create(0));
  mTrackingCAO2Interface.reset(new AliHLTTPCCAO2Interface);
  int retVal = mTrackingCAO2Interface->Initialize(options, std::move(fastTransform));
  if (retVal) {
    mTrackingCAO2Interface.reset();
  }
  return (retVal);
}

void TPCCATracking::deinitialize()
{
  mTrackingCAO2Interface.reset();
}

int TPCCATracking::runTracking(const ClusterNativeAccessFullTPC& clusters, std::vector<TrackTPC>* outputTracks,
                               MCLabelContainer* outputTracksMCTruth)
{
  const static ParameterDetector& detParam = ParameterDetector::defaultInstance();
  const static ParameterGas& gasParam = ParameterGas::defaultInstance();
  const static ParameterElectronics& elParam = ParameterElectronics::defaultInstance();
  float vzbin = (elParam.getZBinWidth() * gasParam.getVdrift());
  float vzbinInv = 1.f / vzbin;
  Mapper& mapper = Mapper::instance();

  const AliHLTTPCGMMergedTrack* tracks;
  int nTracks;
  const AliHLTTPCGMMergedTrackHit* trackClusters;
  int retVal = mTrackingCAO2Interface->RunTracking(&clusters, tracks, nTracks, trackClusters);
  if (retVal == 0) {
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

      float zHigh = 0, zLow = 0;
      if (mTrackingCAO2Interface->GetParamContinuous()) {
        float zoffset = tracks[i].CSide() ? -tracks[i].GetParam().GetZOffset() : tracks[i].GetParam().GetZOffset();
        time0 = sContinuousTFReferenceLength - zoffset * vzbinInv;

        if (tracks[i].CCE()) {
          bool lastSide = trackClusters[tracks[i].FirstClusterRef()].fSlice < Sector::MAXSECTOR / 2;
          float delta = 0.f;
          for (int iCl = 1; iCl < tracks[i].NClusters(); iCl++) {
            if (lastSide ^ (trackClusters[tracks[i].FirstClusterRef() + iCl].fSlice < Sector::MAXSECTOR / 2)) {
              auto& hltcl1 = trackClusters[tracks[i].FirstClusterRef() + iCl];
              auto& hltcl2 = trackClusters[tracks[i].FirstClusterRef() + iCl - 1];
              auto& cl1 = clusters.clusters[hltcl1.fSlice][hltcl1.fRow][hltcl1.fNum];
              auto& cl2 = clusters.clusters[hltcl2.fSlice][hltcl2.fRow][hltcl2.fNum];
              delta = fabs(cl1.getTime() - cl2.getTime()) * 0.5f;
              break;
            }
          }
          tFwd = tBwd = delta;
        } else {
          // estimate max/min time increments which still keep track in the physical limits of the TPC
          zHigh = trackClusters[tracks[i].FirstClusterRef()].fZ - tracks[i].GetParam().GetZOffset(); // high R cluster
          zLow = trackClusters[tracks[i].FirstClusterRef() + tracks[i].NClusters() - 1].fZ -
                 tracks[i].GetParam().GetZOffset(); // low R cluster

          bool sideHighA = trackClusters[tracks[i].FirstClusterRef()].fSlice < Sector::MAXSECTOR / 2;
          bool sideLowA =
            trackClusters[tracks[i].FirstClusterRef() + tracks[i].NClusters() - 1].fSlice < Sector::MAXSECTOR / 2;

          // calculate time bracket
          float zLowAbs = zLow < 0.f ? -zLow : zLow;
          float zHighAbs = zHigh < 0.f ? -zHigh : zHigh;
          //
          // tFwd = (Lmax - max(|zLow|,|zAbs|))/vzbin  = drift time from cluster current Z till endcap
          // tBwd = min(|zLow|,|zAbs|))/vzbin          = drift time from CE till cluster current Z
          //
          if (zLowAbs < zHighAbs) {
            tFwd = (detParam.getTPClength() - zHighAbs) * vzbinInv;
            tBwd = zLowAbs * vzbinInv;
          } else {
            tFwd = (detParam.getTPClength() - zLowAbs) * vzbinInv;
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
      oTrack.setOuterParam(o2::track::TrackParCov(
        outerPar.fX, outerPar.fAlpha,
        { outerPar.fP[0], outerPar.fP[1], outerPar.fP[2], outerPar.fP[3], outerPar.fP[4] },
        { outerPar.fC[0], outerPar.fC[1], outerPar.fC[2], outerPar.fC[3], outerPar.fC[4], outerPar.fC[5],
          outerPar.fC[6], outerPar.fC[7], outerPar.fC[8], outerPar.fC[9], outerPar.fC[10], outerPar.fC[11],
          outerPar.fC[12], outerPar.fC[13], outerPar.fC[14] }));
      oTrack.resetClusterReferences(tracks[i].NClusters());
      std::vector<std::pair<MCCompLabel, unsigned int>> labels;
      for (int j = 0; j < tracks[i].NClusters(); j++) {
        int clusterId = trackClusters[tracks[i].FirstClusterRef() + j].fNum;
        Sector sector = trackClusters[tracks[i].FirstClusterRef() + j].fSlice;
        int globalRow = trackClusters[tracks[i].FirstClusterRef() + j].fRow;
        const ClusterNative& cl = clusters.clusters[sector][globalRow][clusterId];
        int regionNumber = 0;
        while (globalRow > mapper.getGlobalRowOffsetRegion(regionNumber) + mapper.getNumberOfRowsRegion(regionNumber))
          regionNumber++;
        CRU cru(sector, regionNumber);
        oTrack.addCluster(Cluster(cru, globalRow - mapper.getGlobalRowOffsetRegion(regionNumber), cl.qTot, cl.qMax,
                                  cl.getPad(), cl.getSigmaPad(), cl.getTime(), cl.getSigmaTime()));
        oTrack.setClusterReference(j, sector, globalRow, clusterId);
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
          if (bestLabelCount < (1.f - sTrackMCMaxFake) * tracks[i].NClusters())
            bestLabel.set(-bestLabel.getTrackID(), bestLabel.getEventID(), bestLabel.getSourceID());
          outputTracksMCTruth->addElement(iTmp, bestLabel);
        }
      }
      int lastSector = trackClusters[tracks[i].FirstClusterRef() + tracks[i].NClusters() - 1].fNum >> 24;
    }
  }
  mTrackingCAO2Interface->Cleanup();
  return (retVal);
}

float TPCCATracking::getPseudoVDrift()
{
  const static ParameterGas& gasParam = ParameterGas::defaultInstance();
  const static ParameterElectronics& elParam = ParameterElectronics::defaultInstance();
  return (elParam.getZBinWidth() * gasParam.getVdrift());
}

void TPCCATracking::GetClusterErrors2(int row, float z, float sinPhi, float DzDs, float& ErrY2, float& ErrZ2) const
{
  if (mTrackingCAO2Interface == nullptr) {
    return;
  }
  mTrackingCAO2Interface->GetClusterErrors2(row, z, sinPhi, DzDs, ErrY2, ErrZ2);
}
