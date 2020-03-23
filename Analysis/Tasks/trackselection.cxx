// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

//
// Task performing basic track selection
//

#include "Framework/AnalysisDataModel.h"
#include "Framework/AnalysisTask.h"
#include <TH1F.h>

#include <cmath>

#include "Analysis/TrackSelection.h"
#include "Analysis/TrackSelectionTables.h"

using namespace o2;
using namespace o2::framework;
using namespace o2::framework::expressions;

void customize(std::vector<o2::framework::ConfigParamSpec>& workflowOptions)
{
  std::vector<ConfigParamSpec> options{
    {"qa-histos", VariantType::Bool, false, {"Generate QA histograms"}}};
  std::swap(workflowOptions, options);
}

#include "Framework/runDataProcessing.h"

// Default track selection requiring one hit in the SPD
TrackSelection getGlobalTrackSelection()
{
  TrackSelection selectedTracks;
  selectedTracks.SetMinPt(0.1f);
  selectedTracks.SetMaxPt(1e10f);
  selectedTracks.SetMinEta(-0.8f);
  selectedTracks.SetMaxEta(0.8f);
  selectedTracks.SetRequireITSRefit(true);
  selectedTracks.SetRequireTPCRefit(true);
  selectedTracks.SetMinNCrossedRowsTPC(70);
  selectedTracks.SetMinNCrossedRowsOverFindableClustersTPC(0.8f);
  selectedTracks.SetMaxChi2PerClusterTPC(4.f);
  selectedTracks.SetMaxChi2PerClusterITS(36.f);
  selectedTracks.SetRequireHitsInITSLayers(1,
                                           {0, 1}); // one hit in any SPD layer
  selectedTracks.SetMaxDcaXY(2.4f);
  selectedTracks.SetMaxDcaZ(2.f);
  return selectedTracks;
}

// Default track selection requiring no hit in the SPD and one in the innermost
// SDD
TrackSelection getGlobalTrackSelectionSDD()
{
  TrackSelection selectedTracks = getGlobalTrackSelection();
  selectedTracks.ResetITSRequirements();
  selectedTracks.SetRequireNoHitsInITSLayers({0, 1}); // no hit in SPD layers
  selectedTracks.SetRequireHitsInITSLayers(1,
                                           {2}); // one hit in first SDD layer
  return selectedTracks;
}

//--------------------------------------------------------------------
// This task generates the filter table
//--------------------------------------------------------------------
struct TrackFilterTask {
  Produces<aod::TrackSelection> filterTable;

  TrackSelection globalTracks;
  TrackSelection globalTracksSDD;

  void init(InitContext&)
  {
    globalTracks = getGlobalTrackSelection();
    globalTracksSDD = getGlobalTrackSelectionSDD();
  }

  void process(soa::Join<aod::Tracks, aod::TracksCov, aod::TracksExtra> const& tracks)
  {
    for (auto& track : tracks) {
      filterTable(globalTracks.IsSelected(track),
                  globalTracksSDD.IsSelected(track));
    }
  }
};

//--------------------------------------------------------------------
// This task generates QA histograms for track selection
//--------------------------------------------------------------------
struct TrackQATask {
  //Configurable<std::string>selectedTracks{"select", std::string("globalTracks"), "Choice of track selection."};
  //-> Mismatch between declared option type and default value type for select

  // Filter selectedTracks = (aod::track::isGlobalTrack == true);
  // -> error while setting up workflow: Invalid combination of argument types

  Configurable<int> selectedTracks{"select", 1, "Choice of track selection. 0 = no selection, 1 = globalTracks, 2 = globalTracksSDD"};

  OutputObj<TH2F> xy{
    TH2F("trackpar-global-xy",
         "Track XY at dca global coordinate system;x [cm];y [cm];# tracks",
         100, -0.02, 0.14, 100, -0.05, 0.4)};
  OutputObj<TH1F> alpha{
    TH1F("trackpar-alpha",
         "Rotation angle of local wrt. global coordinate system;#alpha [rad]",
         1000, -(M_PI + 0.01), (M_PI + 0.01))};
  OutputObj<TH1F> dcaxy{
    TH1F("track-dcaXY",
         "Distance of closest approach in xy plane;dca-xy [cm];# tracks", 500,
         -1.5, 1.5)};
  OutputObj<TH1F> dcaz{TH1F(
    "track-dcaZ", "Distance of closest approach in z;dca-z [cm];# tracks",
    500, -15, 15)};
  OutputObj<TH1F> flag{TH1F("track-flags", "Track flag bits;", 70, -0.5, 70.5)};

  // ITS related quantities
  OutputObj<TH1F> itsFoundClusters{
    TH1F("its-foundClusters", "nClustersITS;# clusters ITS", 8, -0.5, 7.5)};
  OutputObj<TH1F> chi2PerClusterITS{TH1F(
    "chi2PerClusterITS", "chi2PerClusterITS;chi2/cluster ITS", 100, 0, 50)};
  OutputObj<TH1F> itsHits{TH1F("its-hits", "hitmap its", 7, -0.5, 6.5)};

  // TPC related quantities
  OutputObj<TH1F> tpcFindableClusters{TH1F(
    "tpc-findableClusters", "Findable Clusters;;# tracks", 165, -0.5, 164.5)};
  OutputObj<TH1F> tpcFoundClusters{
    TH1F("tpc-foundClusters", "Found Clusters;;# tracks", 165, -0.5, 164.5)};
  OutputObj<TH1F> tpcSharedClusters{TH1F(
    "tpc-sharedClusters", "Shared Clusters;;# tracks", 165, -0.5, 164.5)};
  OutputObj<TH1F> tpcFractionSharedClusters{
    TH1F("tpc-fractionSharedClusters",
         "Fraction of Shared Clusters;;# tracks", 500, -1.5, 1.5)};
  OutputObj<TH1F> tpcCrossedRows{
    TH1F("tpc-crossedRows", "Crossed Rows;;# tracks", 165, -0.5, 164.5)};
  OutputObj<TH1F> tpcCrossedRowsOverFindableClusters{
    TH1F("tpc-crossedRowsOverFindableClusters",
         "Crossed rows over findable clusters;;# tracks", 110, 0.0, 1.1)};
  OutputObj<TH1F> tpcChi2PerCluster{
    TH1F("tpc-chi2PerCluster", "chi2 per cluster in TPC;chi2/cluster TPC",
         100, 0, 10)};

  // physics quantities
  OutputObj<TH1F> pt{TH1F("track-pt", "pt", 100, 0., 50.)};
  OutputObj<TH1F> eta{TH1F("track-eta", "eta", 102, -2.01, 2.01)};
  OutputObj<TH1F> phi{TH1F("track-phi", "phi", 100, 0., 2 * M_PI)};

  // track parameters
  OutputObj<TH1F> x{TH1F(
    "trackpar-x", "Track X at dca in local coordinate system;x [cm];# tracks",
    500, -0.41, 0.41)};
  OutputObj<TH1F> y{TH1F(
    "trackpar-y", "Track Y at dca in local coordinate system;y [cm];# tracks",
    100, -4., 4.)};
  OutputObj<TH1F> z{TH1F(
    "trackpar-z", "Track Z at dca in local coordinate system;z [cm];# tracks",
    100, -20., 20.)};
  OutputObj<TH1F> signed1Pt{TH1F("trackpar-signed1Pt",
                                 "Track signed 1/p_{T};q/p_{T};# tracks", 1000,
                                 -10, 10)};
  OutputObj<TH1F> snp{TH1F(
    "trackpar-snp", "Sinus of track momentum azimuthal angle;snp;# tracks",
    1000, -0.0001, 0.0001)};
  OutputObj<TH1F> tgl{TH1F(
    "trackpar-tgl", "Tangent of the track momentum dip angle;tgl;# tracks",
    1000, -2, 2)};

  OutputObj<TH2F> collisionPos{
    TH2F("collision-xy", "Position of collision;x [cm];y [cm];# tracks", 100,
         -0.5, 0.5, 100, -0.5, 0.5)};

  void init(o2::framework::InitContext&) {}

  void process(aod::Collision const& collision,
               soa::Join<aod::Tracks, aod::TracksExtra, aod::TracksCov,
                         aod::TrackSelection> const& tracks)
  {

    collisionPos->Fill(collision.posX(), collision.posY());

    for (auto& track : tracks) {

      if (selectedTracks == 1 && !track.isGlobalTrack())
        continue; // FIXME: this should be replaced by framework filter
      else if (selectedTracks == 2 && !track.isGlobalTrackSDD())
        continue;

      // TPC
      tpcFindableClusters->Fill(track.tpcNClsFindable());
      tpcFoundClusters->Fill(track.tpcNClsFound());
      tpcSharedClusters->Fill(track.tpcNClsShared());
      tpcCrossedRows->Fill(track.tpcNClsCrossedRows());

      tpcCrossedRowsOverFindableClusters->Fill(
        track.tpcCrossedRowsOverFindableCls());
      tpcFractionSharedClusters->Fill(track.tpcFractionSharedCls());
      tpcChi2PerCluster->Fill(
        track.tpcChi2Ncl()); // todo: fix typo in AnalysisDatamodel.h

      // ITS
      itsFoundClusters->Fill(track.itsNCls());

      // Tracks
      alpha->Fill(track.alpha());
      x->Fill(track.x());
      y->Fill(track.y());
      z->Fill(track.z());

      float sinAlpha = sin(track.alpha());
      float cosAlpha = cos(track.alpha());
      float globalX = track.x() * cosAlpha - track.y() * sinAlpha;
      float globalY = track.x() * sinAlpha + track.y() * cosAlpha;
      xy->Fill(globalX, globalY);

      dcaxy->Fill(track.charge() * sqrt(pow((globalX - collision.posX()), 2) +
                                        pow((globalY - collision.posY()), 2)));
      dcaz->Fill(track.charge() * sqrt(pow(track.z() - collision.posZ(), 2)));

      signed1Pt->Fill(track.signed1Pt());
      snp->Fill(track.snp());
      tgl->Fill(track.tgl());

      chi2PerClusterITS->Fill(track.itsChi2NCl());

      pt->Fill(track.pt());
      eta->Fill(track.eta());
      phi->Fill(track.phi());

      for (unsigned int i = 0; i < 64; i++) {
        if (track.flags() & (1 << i))
          flag->Fill(i);
      }

      for (unsigned int i = 0; i < 7; i++) {
        if (track.itsClusterMap() & (1 << i))
          itsHits->Fill(i);
      }
    }
  }
};

//--------------------------------------------------------------------
// Workflow definition
//--------------------------------------------------------------------
WorkflowSpec defineDataProcessing(ConfigContext const& cfgc)
{
  bool createQAplots = cfgc.options().get<bool>("qa-histos");
  WorkflowSpec workflow{
    adaptAnalysisTask<TrackFilterTask>("track-filter-task")};
  if (createQAplots)
    workflow.push_back(adaptAnalysisTask<TrackQATask>("track-qa-histograms"));
  return workflow;
}
