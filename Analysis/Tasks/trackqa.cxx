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
#include "Analysis/MC.h"
#include "Analysis/HistHelpers.h"

#include <cmath>

#include "Analysis/TrackSelectionTables.h"

using namespace o2;
using namespace o2::framework;
using namespace o2::framework::expressions;
using namespace o2::experimental::histhelpers;

void customize(std::vector<o2::framework::ConfigParamSpec>& workflowOptions)
{
  std::vector<ConfigParamSpec> options{
    {"mc", VariantType::Bool, false, {"Add MC QA histograms."}}};
  std::swap(workflowOptions, options);
}
#include "Framework/runDataProcessing.h"

//****************************************************************************************
/**
 * QA histograms for track quantities.
 */
//****************************************************************************************
struct TrackQATask {

  enum HistosKine : uint8_t {
    pt,
    eta,
    phi,
  };
  enum HistosTrackPar : uint8_t {
    x,
    y,
    z,
    alpha,
    signed1Pt,
    snp,
    tgl,
    dcaXY,
    dcaZ,
    flags,
  };
  enum HistosITS : uint8_t {
    itsNCls,
    itsChi2NCl,
    itsHits,
  };
  enum HistosTPC : uint8_t {
    tpcNClsFindable,
    tpcNClsFound,
    tpcNClsShared,
    tpcNClsCrossedRows,
    tpcCrossedRowsOverFindableCls,
    tpcFractionSharedCls,
    tpcChi2NCl,
  };

  OutputObj<HistFolder> kine{HistFolder("Kine"), OutputObjHandlingPolicy::QAObject};
  OutputObj<HistFolder> trackpar{HistFolder("TrackPar"), OutputObjHandlingPolicy::QAObject};
  OutputObj<HistFolder> its{HistFolder("ITS"), OutputObjHandlingPolicy::QAObject};
  OutputObj<HistFolder> tpc{HistFolder("TPC"), OutputObjHandlingPolicy::QAObject};
  //OutputObj<HistFolder> trd{HistFolder("TRD"), OutputObjHandlingPolicy::QAObject};
  //OutputObj<HistFolder> tof{HistFolder("TOF"), OutputObjHandlingPolicy::QAObject};
  //OutputObj<HistFolder> emcal{HistFolder("EMCAL"), OutputObjHandlingPolicy::QAObject};

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
    // kine histograms
    kine->Add(pt, new TH1D("pt", "#it{p}_{T};#it{p}_{T} [GeV/c]", 200, 0., 5.));
    kine->Add(eta, new TH1D("eta", "#eta;#eta", 180, -0.9, 0.9));
    kine->Add(phi, new TH1D("phi", "#phi;#phi [rad]", 180, 0., 2 * M_PI));

    // track histograms
    trackpar->Add(x, new TH1D("x", "track #it{x} position at dca in local coordinate system;#it{x} [cm]", 200, -0.36, 0.36));
    trackpar->Add(y, new TH1D("y", "track #it{y} position at dca in local coordinate system;#it{y} [cm]", 200, -0.5, 0.5));
    trackpar->Add(z, new TH1D("z", "track #it{z} position at dca in local coordinate system;#it{z} [cm]", 200, -11., 11.));
    trackpar->Add(alpha, new TH1D("alpha", "rotation angle of local wrt. global coordinate system;#alpha [rad]", 36, -M_PI, M_PI));
    trackpar->Add(signed1Pt, new TH1D("signed1Pt", "track signed 1/#it{p}_{T};#it{q}/#it{p}_{T}", 200, -8, 8));
    trackpar->Add(snp, new TH1D("snp", "sinus of track momentum azimuthal angle;snp", 11, -0.1, 0.1));
    trackpar->Add(tgl, new TH1D("tgl", "tangent of the track momentum dip angle;tgl;", 200, -1., 1.));
    trackpar->Add(flags, new TH1D("flags", "track flag;flag bit", 64, -0.5, 63.5));
    trackpar->Add(dcaXY, new TH1D("dcaXY", "distance of closest approach in #it{xy} plane;#it{dcaXY} [cm];", 200, -0.15, 0.15));
    trackpar->Add(dcaZ, new TH1D("dcaZ", "distance of closest approach in #it{z};#it{dcaZ} [cm];", 200, -0.15, 0.15));

    // its histograms
    its->Add(itsNCls, new TH1D("itsNCls", "number of found ITS clusters;# clusters ITS", 8, -0.5, 7.5));
    its->Add(itsChi2NCl, new TH1D("itsChi2NCl", "chi2 per ITS cluster;chi2 / cluster ITS", 100, 0, 40));
    its->Add(itsHits, new TH1D("itsHits", "hitmap ITS;layer ITS", 7, -0.5, 6.5));

    // tpc histograms
    tpc->Add(tpcNClsFindable, new TH1D("tpcNClsFindable", "number of findable TPC clusters;# findable clusters TPC", 165, -0.5, 164.5));
    tpc->Add(tpcNClsFound, new TH1D("tpcNClsFound", "number of found TPC clusters;# clusters TPC", 165, -0.5, 164.5));
    tpc->Add(tpcNClsShared, new TH1D("tpcNClsShared", "number of shared TPC clusters;# shared clusters TPC", 165, -0.5, 164.5));
    tpc->Add(tpcNClsCrossedRows, new TH1D("tpcNClsCrossedRows", "number of crossed TPC rows;# crossed rows TPC", 165, -0.5, 164.5));
    tpc->Add(tpcFractionSharedCls, new TH1D("tpcFractionSharedCls", "fraction of shared TPC clusters;fraction shared clusters TPC", 100, 0., 1.));
    tpc->Add(tpcCrossedRowsOverFindableCls, new TH1D("tpcCrossedRowsOverFindableCls", "crossed TPC rows over findable clusters;crossed rows / findable clusters TPC", 120, 0.0, 1.2));
    tpc->Add(tpcChi2NCl, new TH1D("tpcChi2NCl", "chi2 per cluster in TPC;chi2 / cluster TPC", 100, 0, 10));
  }

  void process(soa::Filtered<soa::Join<aod::FullTracks, aod::TracksExtended, aod::TrackSelection>>::iterator const& track)
  {
    // fill kinematic variables
    kine->Fill(pt, track.pt());
    kine->Fill(eta, track.eta());
    kine->Fill(phi, track.phi());

    // fill track parameters
    trackpar->Fill(alpha, track.alpha());
    trackpar->Fill(x, track.x());
    trackpar->Fill(y, track.y());
    trackpar->Fill(z, track.z());
    trackpar->Fill(signed1Pt, track.signed1Pt());
    trackpar->Fill(snp, track.snp());
    trackpar->Fill(tgl, track.tgl());
    for (unsigned int i = 0; i < 64; i++) {
      if (track.flags() & (1 << i))
        trackpar->Fill(flags, i);
    }
    trackpar->Fill(dcaXY, track.dcaXY());
    trackpar->Fill(dcaZ, track.dcaZ());

    // fill ITS variables
    its->Fill(itsNCls, track.itsNCls());
    its->Fill(itsChi2NCl, track.itsChi2NCl());
    for (unsigned int i = 0; i < 7; i++) {
      if (track.itsClusterMap() & (1 << i))
        its->Fill(itsHits, i);
    }

    // fill TPC variables
    tpc->Fill(tpcNClsFindable, track.tpcNClsFindable());
    tpc->Fill(tpcNClsFound, track.tpcNClsFound());
    tpc->Fill(tpcNClsShared, track.tpcNClsShared());
    tpc->Fill(tpcNClsCrossedRows, track.tpcNClsCrossedRows());
    tpc->Fill(tpcCrossedRowsOverFindableCls, track.tpcCrossedRowsOverFindableCls());
    tpc->Fill(tpcFractionSharedCls, track.tpcFractionSharedCls());
    tpc->Fill(tpcChi2NCl, track.tpcChi2NCl());

    // fill TRD variables

    // fill TOF variables

    // fill EMCAL variables
  }
};

//****************************************************************************************
/**
 * QA task including MC truth info.
 */
//****************************************************************************************
struct TrackQATaskMC {

  OutputObj<HistFolder> resolution{HistFolder("Resolution"), OutputObjHandlingPolicy::QAObject};

  // unique identifiers for each variable
  enum QuantitiesMC : uint8_t {

  };

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
  bool isMC = cfgc.options().get<bool>("mc");

  WorkflowSpec workflow;
  workflow.push_back(adaptAnalysisTask<TrackQATask>("track-qa-histograms"));
  if (isMC) {
    workflow.push_back(adaptAnalysisTask<TrackQATaskMC>("track-qa-histograms-mc"));
  }
  return workflow;
}
