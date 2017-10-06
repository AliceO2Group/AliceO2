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

//This class is only a wrapper for the actual tracking contained in the HLT O2 CA Tracking library.
//Currently, this library is only linked to O2 conditionally, because it needs a special AliRoot branch.
//Therefore, for the time being, we build the O2 tracking class only if the standalone library is available.
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
#include "TPCBase/ParameterDetector.h"
#include "TPCBase/ParameterGas.h"
#include "TPCBase/ParameterElectronics.h"
#include "TPCBase/Sector.h"

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
    mClusterData_UPTR.reset(new AliHLTTPCCAClusterData[Sector::MAXSECTOR]);
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
#ifdef HAVE_O2_TPCCA_TRACKING_LIB
  if (mTrackingCAO2Interface == nullptr) return (1);

  int retVal = 0;
  const static ParameterDetector &detParam = ParameterDetector::defaultInstance();
  const static ParameterGas &gasParam = ParameterGas::defaultInstance();
  const static ParameterElectronics &elParam = ParameterElectronics::defaultInstance();
  
  const AliHLTTPCGMMergedTrack* tracks;
  int nTracks;
  const unsigned int* trackClusterIDs;

  int nClusters[Sector::MAXSECTOR] = {};
  for (Int_t icluster = 0; icluster < inputClusters->GetEntries(); ++icluster) {
    Cluster& cluster = *static_cast<Cluster*>(inputClusters->At(icluster));
    const Sector sector = CRU(cluster.getCRU()).sector();
    nClusters[sector.getSector()]++;
  }
  for (int i = 0; i < Sector::MAXSECTOR; i++) {
    mClusterData[i].StartReading(i, nClusters[i]);
  }
  int nClustersConverted = 0;
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
    const float localY = padCentre.Y() - (padY - padNumber - 0.5) * region.getPadWidth();
    const float localYfactor = (cru.side() == Side::A) ? -1.f : 1.f;
    float zPositionAbs = cluster.getTimeMean()*elParam.getZBinWidth()*gasParam.getVdrift();
    if (!mTrackingCAO2Interface->GetParamContinuous()) zPositionAbs = detParam.getTPClength() - zPositionAbs;

    Point2D<float> clusterPos(padCentre.X(), localY);

    // sanity checks
    if (zPositionAbs < 0 || (!mTrackingCAO2Interface->GetParamContinuous() && zPositionAbs > detParam.getTPClength())) {
      LOG(INFO) << "Removing cluster " << icluster << "/" << inputClusters->GetEntries() << " time: " << cluster.getTimeMean() << ", abs. z: " << zPositionAbs << "\n";
      continue;
    }

    hltCluster.fId = icluster;
    hltCluster.fX = clusterPos.X();
    hltCluster.fY = clusterPos.Y() * (localYfactor);
    hltCluster.fZ = zPositionAbs * (-localYfactor);
    hltCluster.fRow = rowInSector;
    hltCluster.fAmp = cluster.getQmax();

    cd.SetNumberOfClusters(cd.NumberOfClusters() + 1);
    nClustersConverted++;
  }
  if (inputClusters->GetEntries() != nClustersConverted) {
    LOG(INFO) << "Passed " << nClustersConverted << " (out of " << inputClusters->GetEntries() << ") clusters to CA tracker\n";
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
  return (retVal);
#else
  return(1); //Return error code when the HLT O2 CA tracking library is not available.
#endif
}
