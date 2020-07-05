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

//****************************************************************************************
/**
 * Produce the derived track quantities needed for track selection.
 */
//****************************************************************************************
struct TrackExtensionTask {

  Produces<aod::TracksExtended> extendedTrackQuantities;

  void process(aod::Collision const& collision,
               soa::Join<aod::Tracks, aod::TracksExtra, aod::TracksCov> const& tracks)
  {
    float sinAlpha = 0.f;
    float cosAlpha = 0.f;
    float globalX = 0.f;
    float globalY = 0.f;
    float dcaXY = 0.f;
    float dcaZ = 0.f;

    for (auto& track : tracks) {

      sinAlpha = sin(track.alpha());
      cosAlpha = cos(track.alpha());
      globalX = track.x() * cosAlpha - track.y() * sinAlpha;
      globalY = track.x() * sinAlpha + track.y() * cosAlpha;

      dcaXY = track.charge() * sqrt(pow((globalX - collision.posX()), 2) +
                                    pow((globalY - collision.posY()), 2));
      dcaZ = track.charge() * sqrt(pow(track.z() - collision.posZ(), 2));

      extendedTrackQuantities(dcaXY, dcaZ);
    }
  }
};

//****************************************************************************************
/**
 * Produce track filter table.
 */
//****************************************************************************************
struct TrackSelectionTask {
  Produces<aod::TrackSelection> filterTable;

  TrackSelection globalTracks;
  TrackSelection globalTracksSDD;

  void init(InitContext&)
  {
    globalTracks = getGlobalTrackSelection();
    globalTracksSDD = getGlobalTrackSelectionSDD();
  }

  void process(soa::Join<aod::Tracks, aod::TracksCov, aod::TracksExtra, aod::TracksExtended> const& tracks)
  {
    for (auto& track : tracks) {
      filterTable((uint8_t)globalTracks.IsSelected(track),
                  (uint8_t)globalTracksSDD.IsSelected(track));
    }
  }
};

//****************************************************************************************
/**
 * Generate QA histograms for track selection.
 */
//****************************************************************************************
struct TrackQATask {

  Configurable<int> selectedTracks{"select", 1, "Choice of track selection. 0 = no selection, 1 = globalTracks, 2 = globalTracksSDD"};
  //Filter trackFilter = ((selectedTracks == 1) && (aod::track::isGlobalTrack == true)) || ((selectedTracks == 2) && (aod::track::isGlobalTrackSDD == true));

  // track parameters
  OutputObj<TH1F> x{TH1F(
                      "trackpar-x", "track x position at dca in local coordinate system;x [cm]",
                      200, -0.4, 0.4),
                    OutputObjHandlingPolicy::QAObject};

  OutputObj<TH1F> y{TH1F(
                      "trackpar-y", "track y position at dca in local coordinate system;y [cm]",
                      100, -4., 4.),
                    OutputObjHandlingPolicy::QAObject};

  OutputObj<TH1F> z{TH1F(
                      "trackpar-z", "track z position at dca in local coordinate system;z [cm]",
                      100, -20., 20.),
                    OutputObjHandlingPolicy::QAObject};

  OutputObj<TH1F> alpha{
    TH1F("trackpar-alpha",
         "rotation angle of local wrt. global coordinate system;#alpha [rad]",
         100, -(M_PI + 0.01), (M_PI + 0.01)),
    OutputObjHandlingPolicy::QAObject};

  OutputObj<TH1F> signed1Pt{TH1F("trackpar-signed1Pt",
                                 "track signed 1/p_{T};q/p_{T}", 200,
                                 -8, 8),
                            OutputObjHandlingPolicy::QAObject};

  OutputObj<TH1F> snp{TH1F(
                        "trackpar-snp", "sinus of track momentum azimuthal angle;snp",
                        100, -1., 1.),
                      OutputObjHandlingPolicy::QAObject};

  OutputObj<TH1F> tgl{TH1F(
                        "trackpar-tgl", "tangent of the track momentum dip angle;tgl;",
                        1000, -2, 2),
                      OutputObjHandlingPolicy::QAObject};

  OutputObj<TH1F> dcaxy{
    TH1F("track-dcaXY",
         "distance of closest approach in xy plane;dca-xy [cm];", 200,
         -3., 3.),
    OutputObjHandlingPolicy::QAObject};

  OutputObj<TH1F> dcaz{TH1F(
                         "track-dcaZ", "distance of closest approach in z;dca-z [cm];",
                         200, -3., 3.),
                       OutputObjHandlingPolicy::QAObject};

  OutputObj<TH1F> flag{TH1F("track-flags", "track flag;flag bit", 64, -0.5, 63.5), OutputObjHandlingPolicy::QAObject};

  OutputObj<TH1F> pt{TH1F("track-pt", "p_{T};p_{T} [GeV/c]", 100, 0., 50.), OutputObjHandlingPolicy::QAObject};

  OutputObj<TH1F> eta{TH1F("track-eta", "#eta;#eta", 101, -1.0, 1.0), OutputObjHandlingPolicy::QAObject};

  OutputObj<TH1F> phi{TH1F("track-phi", "#phi;#phi [rad]", 100, 0., 2 * M_PI), OutputObjHandlingPolicy::QAObject};

  // ITS related quantities
  OutputObj<TH1F> itsFoundClusters{
    TH1F("its-foundClusters", "number of found ITS clusters;# clusters ITS", 8, -0.5, 7.5), OutputObjHandlingPolicy::QAObject};

  OutputObj<TH1F> itsChi2PerCluster{TH1F(
                                      "its-chi2PerCluster", "chi2 per ITS cluster;chi2 / cluster ITS", 100, 0, 40),
                                    OutputObjHandlingPolicy::QAObject};

  OutputObj<TH1F> itsHits{TH1F("its-hits", "hitmap ITS;layer ITS", 7, -0.5, 6.5), OutputObjHandlingPolicy::QAObject};

  // TPC related quantities
  OutputObj<TH1F> tpcFindableClusters{TH1F(
                                        "tpc-findableClusters", "number of findable TPC clusters;# clusters TPC", 165, -0.5, 164.5),
                                      OutputObjHandlingPolicy::QAObject};

  OutputObj<TH1F> tpcFoundClusters{
    TH1F("tpc-foundClusters", "number of found TPC clusters;# clusters TPC", 165, -0.5, 164.5), OutputObjHandlingPolicy::QAObject};

  OutputObj<TH1F> tpcSharedClusters{TH1F(
                                      "tpc-sharedClusters", "number of shared TPC clusters;;# shared clusters TPC", 165, -0.5, 164.5),
                                    OutputObjHandlingPolicy::QAObject};

  OutputObj<TH1F> tpcFractionSharedClusters{
    TH1F("tpc-fractionSharedClusters",
         "fraction of shared TPC clusters;fraction shared clusters TPC", 100, 0., 1.),
    OutputObjHandlingPolicy::QAObject};

  OutputObj<TH1F> tpcCrossedRows{
    TH1F("tpc-crossedRows", "number of crossed TPC rows;# crossed rows TPC", 165, -0.5, 164.5), OutputObjHandlingPolicy::QAObject};

  OutputObj<TH1F> tpcCrossedRowsOverFindableClusters{
    TH1F("tpc-crossedRowsOverFindableClusters",
         "crossed TPC rows over findable clusters;crossed rows / findable clusters TPC", 110, 0.0, 1.1),
    OutputObjHandlingPolicy::QAObject};

  OutputObj<TH1F> tpcChi2PerCluster{
    TH1F("tpc-chi2PerCluster", "chi2 per cluster in TPC;chi2 / cluster TPC",
         100, 0, 10),
    OutputObjHandlingPolicy::QAObject};

  // collision related quantities
  OutputObj<TH2F> collisionPos{
    TH2F("collision-xy", "Position of collision;x [cm];y [cm]", 100,
         -0.5, 0.5, 100, -0.5, 0.5),
    OutputObjHandlingPolicy::QAObject};

  void init(o2::framework::InitContext&) {}

  void process(aod::Collision const& collision,
               soa::Join<aod::Tracks, aod::TracksExtra, aod::TracksCov, aod::TracksExtended,
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
        track.tpcChi2NCl());

      // ITS
      itsFoundClusters->Fill(track.itsNCls());
      itsChi2PerCluster->Fill(track.itsChi2NCl());
      for (unsigned int i = 0; i < 7; i++) {
        if (track.itsClusterMap() & (1 << i))
          itsHits->Fill(i);
      }

      // Tracks
      alpha->Fill(track.alpha());
      x->Fill(track.x());
      y->Fill(track.y());
      z->Fill(track.z());

      dcaxy->Fill(track.dcaXY());
      dcaz->Fill(track.dcaZ());

      signed1Pt->Fill(track.signed1Pt());
      snp->Fill(track.snp());
      tgl->Fill(track.tgl());

      for (unsigned int i = 0; i < 64; i++) {
        if (track.flags() & (1 << i))
          flag->Fill(i);
      }

      pt->Fill(track.pt());
      eta->Fill(track.eta());
      phi->Fill(track.phi());
    }
  }
};

//****************************************************************************************
/**
 * Workflow definition.
 */
//****************************************************************************************
WorkflowSpec defineDataProcessing(ConfigContext const& cfgc)
{
  bool createQAplots = cfgc.options().get<bool>("qa-histos");
  WorkflowSpec workflow{
    adaptAnalysisTask<TrackExtensionTask>("track-extension"),
    adaptAnalysisTask<TrackSelectionTask>("track-selection")};
  if (createQAplots)
    workflow.push_back(adaptAnalysisTask<TrackQATask>("track-qa-histograms"));
  return workflow;
}
