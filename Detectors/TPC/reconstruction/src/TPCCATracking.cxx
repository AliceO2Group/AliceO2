// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See https://alice-o2.web.cern.ch/ for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

/// \file TPCCATracking.cxx
/// \author David Rohr

#include "TPCReconstruction/TPCCATracking.h"
#ifdef HAVE_O2_TPCCA_TRACKING_LIB
#include "TObject.h"
#include "AliHLTTPCCAO2Interface.h"
#else
class AliHLTTPCCAO2Interface {}; // Dummy class such that the compiler can create a destructor for the unique_ptr
class AliHLTTPCCAClusterData {}; // same
#endif

#include "DetectorsBase/Track.h"
#include "FairLogger.h"
#include "TClonesArray.h"
#include "TPCBase/Mapper.h"
#include "TPCBase/PadRegionInfo.h"
#include "TPCBase/Sector.h"
#include "TPCReconstruction/TrackTPC.h"
#include "TPCSimulation/Cluster.h"
#include "TPCSimulation/Constants.h"

using namespace o2::TPC;

TPCCATracking::TPCCATracking() : mTrackingCAO2Interface(), mClusterData_UPTR(), mClusterData(nullptr) {
}

TPCCATracking::~TPCCATracking() {
  deinitialize();
}

int TPCCATracking::initialize(const char* options) {
#ifndef HAVE_O2_TPCCA_TRACKING_LIB
  LOG(FATAL) << "O2 compiled without TPC CA Tracking library, cannot run TPC CA Tracker\n";
  return (1);
#else
  mTrackingCAO2Interface.reset(new AliHLTTPCCAO2Interface);
  int retVal = mTrackingCAO2Interface->Initialize(options);
  if (retVal) {
    mTrackingCAO2Interface.reset();
  } else {
    mClusterData_UPTR.reset(new AliHLTTPCCAClusterData[36]);
    mClusterData = mClusterData_UPTR.get();
  }
  return (retVal);
#endif
}

void TPCCATracking::deinitialize() {
  mTrackingCAO2Interface.reset();
  mClusterData_UPTR.reset();
  mClusterData = NULL;
}

int TPCCATracking::runTracking(const TClonesArray* inputClusters, std::vector<TrackTPC>* outputTracks) {
  if (mTrackingCAO2Interface == nullptr) return (1);
  int retVal = 0;

#ifdef HAVE_O2_TPCCA_TRACKING_LIB
  const AliHLTTPCGMMergedTrack* tracks;
  int nTracks;
  const unsigned int* trackClusterIDs;

  int nClusters[36] = {};
  for (Int_t icluster = 0; icluster < inputClusters->GetEntries(); ++icluster) {
    Cluster& cluster = *static_cast<Cluster*>(inputClusters->At(icluster));
    const Sector sector = CRU(cluster.getCRU()).sector();
    nClusters[sector.getSector()]++;
  }
  for (int i = 0; i < 36; i++) {
    mClusterData[i].StartReading(i, nClusters[i]);
  }
  for (Int_t icluster = 0; icluster < inputClusters->GetEntries(); ++icluster) {
    Cluster& cluster = *static_cast<Cluster*>(inputClusters->At(icluster));
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
    const float localY = padCentre.getY() - (padY - padNumber - 0.5) * region.getPadWidth();
    const float localYfactor = (cru.side() == Side::A) ? -1.f : 1.f;
    float zPosition = TPCLENGTH - cluster.getTimeMean()*ZBINWIDTH*DRIFTV;

    Point2D<float> clusterPos(padCentre.getX(), localY);

    // sanity checks
    if (zPosition < 0)
      continue;
    if (zPosition > TPCLENGTH)
      continue;

    hltCluster.fId = icluster;
    hltCluster.fX = clusterPos.getX();
    hltCluster.fY = clusterPos.getY() * (localYfactor);
    hltCluster.fZ = zPosition * (-localYfactor);
    hltCluster.fRow = rowInSector;
    hltCluster.fAmp = cluster.getQmax();

    cd.SetNumberOfClusters(cd.NumberOfClusters() + 1);
  }
  retVal = mTrackingCAO2Interface->RunTracking(mClusterData, tracks, nTracks, trackClusterIDs);
  if (retVal == 0)
  {
    for (int i = 0; i < nTracks; i++) {
      if (!tracks[i].OK()) continue;
      TrackTPC trackTPC(
        tracks[i].GetParam().GetX(), tracks[i].GetAlpha(),
        { tracks[i].GetParam().GetY(), tracks[i].GetParam().GetZ(), tracks[i].GetParam().GetSinPhi(),
          tracks[i].GetParam().GetDzDs(), tracks[i].GetParam().GetQPt() },
        { tracks[i].GetParam().GetCov(0), tracks[i].GetParam().GetCov(1), tracks[i].GetParam().GetCov(2),
          tracks[i].GetParam().GetCov(3), tracks[i].GetParam().GetCov(4), tracks[i].GetParam().GetCov(5),
          tracks[i].GetParam().GetCov(6), tracks[i].GetParam().GetCov(7), tracks[i].GetParam().GetCov(8),
          tracks[i].GetParam().GetCov(9), tracks[i].GetParam().GetCov(10), tracks[i].GetParam().GetCov(11),
          tracks[i].GetParam().GetCov(12), tracks[i].GetParam().GetCov(13), tracks[i].GetParam().GetCov(14) });
      for (int j = 0; j < tracks[i].NClusters(); j++) {
        trackTPC.addCluster(*(static_cast<Cluster*>(inputClusters->At(trackClusterIDs[tracks[i].FirstClusterRef() + j]))));
      }
      outputTracks->push_back(trackTPC);
    }
  }
  mTrackingCAO2Interface->Cleanup();
#endif

  return (retVal);
}
