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
#include "AnalysisCore/MC.h"
#include "AnalysisDataModel/TrackSelectionTables.h"
#include "AnalysisCore/TrackSelection.h"
#include "AnalysisCore/TrackSelectionDefaults.h"

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

  HistogramRegistry histos{"Histos", {}, OutputObjHandlingPolicy::QAObject};

  Configurable<int> selectedTracks{"select", 1, "Choice of track selection. 0 = no selection, 1 = globalTracks, 2 = globalTracksSDD"};

  Filter trackFilter = aod::track::isGlobalTrack == true;

  void init(o2::framework::InitContext&)
  {
    std::vector<double> ptBinning = {0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0,
                                     1.1, 1.2, 1.3, 1.4, 1.5, 2.0, 5.0, 10.0, 20.0, 50.0};

    // kine histograms
    histos.add("Kine/pt", "#it{p}_{T};#it{p}_{T} [GeV/c]", kTH1D, {{ptBinning}});
    histos.add("Kine/eta", "#eta;#eta", kTH1D, {{180, -0.9, 0.9}});
    histos.add("Kine/phi", "#phi;#phi [rad]", kTH1D, {{180, 0., 2 * M_PI}});

    // track histograms
    histos.add("TrackPar/x", "track #it{x} position at dca in local coordinate system;#it{x} [cm]", kTH1D, {{200, -0.36, 0.36}});
    histos.add("TrackPar/y", "track #it{y} position at dca in local coordinate system;#it{y} [cm]", kTH1D, {{200, -0.5, 0.5}});
    histos.add("TrackPar/z", "track #it{z} position at dca in local coordinate system;#it{z} [cm]", kTH1D, {{200, -11., 11.}});
    histos.add("TrackPar/alpha", "rotation angle of local wrt. global coordinate system;#alpha [rad]", kTH1D, {{36, -M_PI, M_PI}});
    histos.add("TrackPar/signed1Pt", "track signed 1/#it{p}_{T};#it{q}/#it{p}_{T}", kTH1D, {{200, -8, 8}});
    histos.add("TrackPar/snp", "sinus of track momentum azimuthal angle;snp", kTH1D, {{11, -0.1, 0.1}});
    histos.add("TrackPar/tgl", "tangent of the track momentum dip angle;tgl;", kTH1D, {{200, -1., 1.}});
    histos.add("TrackPar/flags", "track flag;flag bit", kTH1D, {{64, -0.5, 63.5}});
    histos.add("TrackPar/dcaXY", "distance of closest approach in #it{xy} plane;#it{dcaXY} [cm];", kTH1D, {{200, -0.15, 0.15}});
    histos.add("TrackPar/dcaZ", "distance of closest approach in #it{z};#it{dcaZ} [cm];", kTH1D, {{200, -0.15, 0.15}});
    histos.add("TrackPar/length", "track length in cm;#it{Length} [cm];", kTH1D, {{400, 0, 1000}});

    // its histograms
    histos.add("ITS/itsNCls", "number of found ITS clusters;# clusters ITS", kTH1D, {{8, -0.5, 7.5}});
    histos.add("ITS/itsChi2NCl", "chi2 per ITS cluster;chi2 / cluster ITS", kTH1D, {{100, 0, 40}});
    histos.add("ITS/itsHits", "hitmap ITS;layer ITS", kTH1D, {{7, -0.5, 6.5}});

    // tpc histograms
    histos.add("TPC/tpcNClsFindable", "number of findable TPC clusters;# findable clusters TPC", kTH1D, {{165, -0.5, 164.5}});
    histos.add("TPC/tpcNClsFound", "number of found TPC clusters;# clusters TPC", kTH1D, {{165, -0.5, 164.5}});
    histos.add("TPC/tpcNClsShared", "number of shared TPC clusters;# shared clusters TPC", kTH1D, {{165, -0.5, 164.5}});
    histos.add("TPC/tpcNClsCrossedRows", "number of crossed TPC rows;# crossed rows TPC", kTH1D, {{165, -0.5, 164.5}});
    histos.add("TPC/tpcFractionSharedCls", "fraction of shared TPC clusters;fraction shared clusters TPC", kTH1D, {{100, 0., 1.}});
    histos.add("TPC/tpcCrossedRowsOverFindableCls", "crossed TPC rows over findable clusters;crossed rows / findable clusters TPC", kTH1D, {{120, 0.0, 1.2}});
    histos.add("TPC/tpcChi2NCl", "chi2 per cluster in TPC;chi2 / cluster TPC", kTH1D, {{100, 0, 10}});

    histos.print();
  }

  void process(soa::Filtered<soa::Join<aod::FullTracks, aod::TracksExtended, aod::TrackSelection>>::iterator const& track)
  {
    // fill kinematic variables
    histos.fill("Kine/pt", track.pt());
    histos.fill("Kine/eta", track.eta());
    histos.fill("Kine/phi", track.phi());

    // fill track parameters
    histos.fill("TrackPar/alpha", track.alpha());
    histos.fill("TrackPar/x", track.x());
    histos.fill("TrackPar/y", track.y());
    histos.fill("TrackPar/z", track.z());
    histos.fill("TrackPar/signed1Pt", track.signed1Pt());
    histos.fill("TrackPar/snp", track.snp());
    histos.fill("TrackPar/tgl", track.tgl());
    for (unsigned int i = 0; i < 64; i++) {
      if (track.flags() & (1 << i)) {
        histos.fill("TrackPar/flags", i);
      }
    }
    histos.fill("TrackPar/dcaXY", track.dcaXY());
    histos.fill("TrackPar/dcaZ", track.dcaZ());
    histos.fill("TrackPar/length", track.length());

    // fill ITS variables
    histos.fill("ITS/itsNCls", track.itsNCls());
    histos.fill("ITS/itsChi2NCl", track.itsChi2NCl());
    for (unsigned int i = 0; i < 7; i++) {
      if (track.itsClusterMap() & (1 << i)) {
        histos.fill("ITS/itsHits", i);
      }
    }

    // fill TPC variables
    histos.fill("TPC/tpcNClsFindable", track.tpcNClsFindable());
    histos.fill("TPC/tpcNClsFound", track.tpcNClsFound());
    histos.fill("TPC/tpcNClsShared", track.tpcNClsShared());
    histos.fill("TPC/tpcNClsCrossedRows", track.tpcNClsCrossedRows());
    histos.fill("TPC/tpcCrossedRowsOverFindableCls", track.tpcCrossedRowsOverFindableCls());
    histos.fill("TPC/tpcFractionSharedCls", track.tpcFractionSharedCls());
    histos.fill("TPC/tpcChi2NCl", track.tpcChi2NCl());
  }
};

struct TrackCutQATask {
  HistogramRegistry cuts{"Cuts", {}, OutputObjHandlingPolicy::QAObject};
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

  HistogramRegistry resolution{"Resolution", {}, OutputObjHandlingPolicy::QAObject};

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
