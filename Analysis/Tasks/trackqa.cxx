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
// Task producing basic tracking qa histograms
//

#include "Framework/AnalysisDataModel.h"
#include "Framework/AnalysisTask.h"
#include "Framework/HistogramRegistry.h"
#include "Analysis/MC.h"
#include "Analysis/TrackSelectionTables.h"
#include "Analysis/TrackSelection.h"
#include "Analysis/TrackSelectionDefaults.h"

using namespace o2;
using namespace o2::framework;
using namespace o2::framework::expressions;

void customize(std::vector<o2::framework::ConfigParamSpec>& workflowOptions)
{
  std::vector<ConfigParamSpec> options{
    {"mc", VariantType::Bool, false, {"Add MC QA histograms."}},
    {"add-cut-qa", VariantType::Int, 0, {"Add track cut QA histograms."}}};
  std::swap(workflowOptions, options);
}
#include "Framework/runDataProcessing.h"

//****************************************************************************************
/**
 * QA histograms for track quantities.
 */
//****************************************************************************************
struct TrackQATask {

  HistogramRegistry kine{"Kine", true, {}, OutputObjHandlingPolicy::QAObject};
  HistogramRegistry trackpar{"TrackPar", true, {}, OutputObjHandlingPolicy::QAObject};
  HistogramRegistry its{"ITS", true, {}, OutputObjHandlingPolicy::QAObject};
  HistogramRegistry tpc{"TPC", true, {}, OutputObjHandlingPolicy::QAObject};

  //HistogramRegistry trd{"TRD", true, {}, OutputObjHandlingPolicy::QAObject};
  //HistogramRegistry tof{"TOF", true, {}, OutputObjHandlingPolicy::QAObject};
  //HistogramRegistry emcal{"EMCAL", true, {}, OutputObjHandlingPolicy::QAObject};

  Configurable<int> selectedTracks{"select", 1, "Choice of track selection. 0 = no selection, 1 = globalTracks, 2 = globalTracksSDD"};

  Filter trackFilter = aod::track::isGlobalTrack == true;
  /*
  //Function float castFLOAT4(uint8) not supported yet
  Filter trackFilter = ((0 * aod::track::isGlobalTrack == (float)selectedTracks) ||
                        (1 * aod::track::isGlobalTrack == (float)selectedTracks) ||
                        (2 * aod::track::isGlobalTrackSDD == (float)selectedTracks) ||
                        (3 * aod::track::isGlobalTrackwTOF == (float)selectedTracks));
  */

  void init(o2::framework::InitContext&)
  {
    std::vector<double> ptBinning = {0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0,
                                     1.1, 1.2, 1.3, 1.4, 1.5, 2.0, 5.0, 10.0, 20.0, 50.0};

    // kine histograms
    kine.add("pt", "#it{p}_{T};#it{p}_{T} [GeV/c]", kTH1D, {{ptBinning}});
    kine.add("eta", "#eta;#eta", kTH1D, {{180, -0.9, 0.9}});
    kine.add("phi", "#phi;#phi [rad]", kTH1D, {{180, 0., 2 * M_PI}});

    // track histograms
    trackpar.add("x", "track #it{x} position at dca in local coordinate system;#it{x} [cm]", kTH1D, {{200, -0.36, 0.36}});
    trackpar.add("y", "track #it{y} position at dca in local coordinate system;#it{y} [cm]", kTH1D, {{200, -0.5, 0.5}});
    trackpar.add("z", "track #it{z} position at dca in local coordinate system;#it{z} [cm]", kTH1D, {{200, -11., 11.}});
    trackpar.add("alpha", "rotation angle of local wrt. global coordinate system;#alpha [rad]", kTH1D, {{36, -M_PI, M_PI}});
    trackpar.add("signed1Pt", "track signed 1/#it{p}_{T};#it{q}/#it{p}_{T}", kTH1D, {{200, -8, 8}});
    trackpar.add("snp", "sinus of track momentum azimuthal angle;snp", kTH1D, {{11, -0.1, 0.1}});
    trackpar.add("tgl", "tangent of the track momentum dip angle;tgl;", kTH1D, {{200, -1., 1.}});
    trackpar.add("flags", "track flag;flag bit", kTH1D, {{64, -0.5, 63.5}});
    trackpar.add("dcaXY", "distance of closest approach in #it{xy} plane;#it{dcaXY} [cm];", kTH1D, {{200, -0.15, 0.15}});
    trackpar.add("dcaZ", "distance of closest approach in #it{z};#it{dcaZ} [cm];", kTH1D, {{200, -0.15, 0.15}});

    // its histograms
    its.add("itsNCls", "number of found ITS clusters;# clusters ITS", kTH1D, {{8, -0.5, 7.5}});
    its.add("itsChi2NCl", "chi2 per ITS cluster;chi2 / cluster ITS", kTH1D, {{100, 0, 40}});
    its.add("itsHits", "hitmap ITS;layer ITS", kTH1D, {{7, -0.5, 6.5}});

    // tpc histograms
    tpc.add("tpcNClsFindable", "number of findable TPC clusters;# findable clusters TPC", kTH1D, {{165, -0.5, 164.5}});
    tpc.add("tpcNClsFound", "number of found TPC clusters;# clusters TPC", kTH1D, {{165, -0.5, 164.5}});
    tpc.add("tpcNClsShared", "number of shared TPC clusters;# shared clusters TPC", kTH1D, {{165, -0.5, 164.5}});
    tpc.add("tpcNClsCrossedRows", "number of crossed TPC rows;# crossed rows TPC", kTH1D, {{165, -0.5, 164.5}});
    tpc.add("tpcFractionSharedCls", "fraction of shared TPC clusters;fraction shared clusters TPC", kTH1D, {{100, 0., 1.}});
    tpc.add("tpcCrossedRowsOverFindableCls", "crossed TPC rows over findable clusters;crossed rows / findable clusters TPC", kTH1D, {{120, 0.0, 1.2}});
    tpc.add("tpcChi2NCl", "chi2 per cluster in TPC;chi2 / cluster TPC", kTH1D, {{100, 0, 10}});
  }

  void process(soa::Filtered<soa::Join<aod::FullTracks, aod::TracksExtended, aod::TrackSelection>>::iterator const& track)
  {
    // fill kinematic variables
    kine.fill("pt", track.pt());
    kine.fill("eta", track.eta());
    kine.fill("phi", track.phi());

    // fill track parameters
    trackpar.fill("alpha", track.alpha());
    trackpar.fill("x", track.x());
    trackpar.fill("y", track.y());
    trackpar.fill("z", track.z());
    trackpar.fill("signed1Pt", track.signed1Pt());
    trackpar.fill("snp", track.snp());
    trackpar.fill("tgl", track.tgl());
    for (unsigned int i = 0; i < 64; i++) {
      if (track.flags() & (1 << i)) {
        trackpar.fill("flags", i);
      }
    }
    trackpar.fill("dcaXY", track.dcaXY());
    trackpar.fill("dcaZ", track.dcaZ());

    // fill ITS variables
    its.fill("itsNCls", track.itsNCls());
    its.fill("itsChi2NCl", track.itsChi2NCl());
    for (unsigned int i = 0; i < 7; i++) {
      if (track.itsClusterMap() & (1 << i)) {
        its.fill("itsHits", i);
      }
    }

    // fill TPC variables
    tpc.fill("tpcNClsFindable", track.tpcNClsFindable());
    tpc.fill("tpcNClsFound", track.tpcNClsFound());
    tpc.fill("tpcNClsShared", track.tpcNClsShared());
    tpc.fill("tpcNClsCrossedRows", track.tpcNClsCrossedRows());
    tpc.fill("tpcCrossedRowsOverFindableCls", track.tpcCrossedRowsOverFindableCls());
    tpc.fill("tpcFractionSharedCls", track.tpcFractionSharedCls());
    tpc.fill("tpcChi2NCl", track.tpcChi2NCl());

    // fill TRD variables

    // fill TOF variables

    // fill EMCAL variables
  }
};

struct TrackCutQATask {
  HistogramRegistry cuts{"Cuts", true, {}, OutputObjHandlingPolicy::QAObject};
  TrackSelection selectedTracks = getGlobalTrackSelection();
  static constexpr int ncuts = static_cast<int>(TrackSelection::TrackCuts::kNCuts);
  void init(InitContext&)
  {
    cuts.add("single_cut", ";Cut;Tracks", kTH1D, {{ncuts, 0, ncuts}});
    for (int i = 0; i < ncuts; i++) {
      cuts.get<TH1>("single_cut")->GetXaxis()->SetBinLabel(1 + i, TrackSelection::mCutNames[i].data());
    }
  }

  void process(soa::Join<aod::FullTracks, aod::TracksExtended>::iterator const& track)
  {
    for (int i = 0; i < ncuts; i++) {
      if (selectedTracks.IsSelected(track, static_cast<TrackSelection::TrackCuts>(i))) {
        cuts.fill("single_cut", i);
      }
    }
  }
};

//****************************************************************************************
/**
 * QA task including MC truth info.
 */
//****************************************************************************************
struct TrackQATaskMC {

  HistogramRegistry resolution{"Resolution", true, {}, OutputObjHandlingPolicy::QAObject};

  void init(o2::framework::InitContext&){

  };

  void process(soa::Join<aod::FullTracks, aod::McTrackLabels>::iterator const& track){

  };
};

//****************************************************************************************
/**
 * Workflow definition.
 */
//****************************************************************************************
WorkflowSpec defineDataProcessing(ConfigContext const& cfgc)
{
  const bool isMC = cfgc.options().get<bool>("mc");
  const int add_cut_qa = cfgc.options().get<int>("add-cut-qa");

  WorkflowSpec workflow;
  workflow.push_back(adaptAnalysisTask<TrackQATask>("track-qa-histograms"));
  if (add_cut_qa) {
    workflow.push_back(adaptAnalysisTask<TrackCutQATask>("track-cut-qa-histograms"));
  }
  if (isMC) {
    workflow.push_back(adaptAnalysisTask<TrackQATaskMC>("track-qa-histograms-mc"));
  }
  return workflow;
}
