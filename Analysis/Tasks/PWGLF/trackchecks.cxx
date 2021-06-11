// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.
#include "Framework/runDataProcessing.h"
#include "Framework/AnalysisTask.h"
#include "Framework/AnalysisDataModel.h"
#include "Framework/HistogramRegistry.h"
#include "AnalysisDataModel/EventSelection.h"
#include "AnalysisCore/MC.h"
#include "AnalysisCore/TrackSelection.h"
#include "AnalysisDataModel/TrackSelectionTables.h"

#include <cmath>
#include <array>
#include <utility>

#include <TF1.h>

using namespace o2;
using namespace o2::framework;
using namespace o2::framework::expressions;
using namespace std;

#define NPARTICLES 5

const std::vector<double> ptBins = {0.0, 0.05, 0.08, 0.1, 0.12, 0.14, 0.16, 0.18, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45,
                                    0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95, 1.0, 1.1, 1.2, 1.3, 1.4,
                                    1.5, 1.6, 1.7, 1.8, 1.9, 2.0, 2.2, 2.4, 2.6, 2.8, 3.0, 3.2, 3.4, 3.6, 3.8, 4.0,
                                    4.5, 5.0, 5.5, 6.0, 6.5, 7.0, 8.0, 10.0, 20.0};
const double mass[NPARTICLES] = {0.000510999, 0.139570, 0.493677, 0.938272, 1.87561};
enum PType : uint8_t {
  kEl,
  kPi,
  kKa,
  kPr,
  kDe,
  kNull
};

//function to convert eta to y
double eta2y(double pt, double m, double eta)
{
  double mt = sqrt(m * m + pt * pt);
  return asinh(pt / mt * sinh(eta));
}

int getparticleint(int pdgcode)
{
  if (pdgcode == 11) {
    return kEl;
  } else if (pdgcode == 211) {
    return kPi;
  } else if (pdgcode == 321) {
    return kKa;
  } else if (pdgcode == 2212) {
    return kPr;
  } else if (pdgcode == 1000010020) {
    return kDe;
  } else {
    return kNull;
  }
}

//No track selection --> only event selection here
struct TrackCheckTaskEvSel {

  Configurable<bool> isMC{"isMC", false, "option to flag mc"};
  Configurable<double> cfgCutY{"cfgCutY", 0.5, "option to configure rapidity cut"};
  Configurable<float> cfgCutVZ{"cfgCutVZ", 10.f, "option to configure z-vertex cut"};

  HistogramRegistry histograms{"histograms"};

  void init(InitContext&)
  {
    histograms.add("hTrkPrimAftEvSel", "Reco Prim tracks AftEvSel (charged); #it{p}_{T} (GeV/#it{c}); Counts", {kTH1F, {{ptBins}}});
    histograms.add("hTrkPrimAftEvSel_truepid_el", "Gen tracks aft. ev. sel. (true El); #it{p}_{T} (GeV/#it{c}); Counts", {kTH1F, {{ptBins}}});
    histograms.add("hTrkPrimAftEvSel_truepid_pi", "Gen tracks aft. ev. sel. (true Pi); #it{p}_{T} (GeV/#it{c}); Counts", {kTH1F, {{ptBins}}});
    histograms.add("hTrkPrimAftEvSel_truepid_ka", "Gen. tracks aft. ev. sel. (true Ka); #it{p}_{T} (GeV/#it{c}); Counts", {kTH1F, {{ptBins}}});
    histograms.add("hTrkPrimAftEvSel_truepid_pr", "Gen tracks aft. ev. sel. (true Pr); #it{p}_{T} (GeV/#it{c}); Counts", {kTH1F, {{ptBins}}});
    histograms.add("hTrkPrimAftEvSel_truepid_de", "Gen tracks aft. ev. sel. (true De); #it{p}_{T} (GeV/#it{c}); Counts", {kTH1F, {{ptBins}}});
  } //init

  //Filters
  Filter collfilter = nabs(aod::collision::posZ) < cfgCutVZ;
  void process(soa::Filtered<soa::Join<aod::Collisions, aod::EvSels>>::iterator const& col,
               soa::Join<aod::Tracks, aod::TracksExtra, aod::McTrackLabels>& tracks, aod::McParticles& mcParticles)
  {

    //event selection
    if (!isMC && !col.alias()[kINT7]) { // trigger (should be skipped in MC)
      return;
    }
    if (!col.sel7()) { //additional cuts
      return;
    }

    //Loop on tracks
    for (auto& track : tracks) {
      double y = -999.;
      bool isPrimary = false;

      if (isMC) { //determine particle species base on MC truth and if it is primary or not
        int pdgcode = track.mcParticle().pdgCode();

        if (MC::isPhysicalPrimary(mcParticles, track.mcParticle())) { //is primary?
          isPrimary = true;
        }

        //calculate rapidity
        int pint = getparticleint(pdgcode);
        if (pint == kNull) {
          y = (double)kNull;
        } else {
          y = eta2y(track.pt(), mass[pint], track.eta());
        }

        if (isPrimary && abs(y) < cfgCutY) {
          //histograms with generated distribution (after event selection)
          if (pdgcode == 11) {
            histograms.fill(HIST("hTrkPrimAftEvSel_truepid_el"), track.pt());
          } else if (pdgcode == 211) {
            histograms.fill(HIST("hTrkPrimAftEvSel_truepid_pi"), track.pt());
          } else if (pdgcode == 321) {
            histograms.fill(HIST("hTrkPrimAftEvSel_truepid_ka"), track.pt());
          } else if (pdgcode == 2212) {
            histograms.fill(HIST("hTrkPrimAftEvSel_truepid_pr"), track.pt());
          } else if (pdgcode == 1000010020) {
            histograms.fill(HIST("hTrkPrimAftEvSel_truepid_de"), track.pt());
          }

          histograms.fill(HIST("hTrkPrimAftEvSel"), track.pt()); // charged particles
        }
      }
    } // end loop on tracks
  }   // end of process
};    // end struct TrackCheckTaskEvSel

//event selection + track selection here
struct TrackCheckTaskEvSelTrackSel {

  Configurable<bool> isMC{"isMC", false, "option to flag mc"};
  Configurable<double> cfgCutY{"cfgCutY", 0.5, "option to configure rapidity cut"};
  Configurable<float> cfgCutVZ{"cfgCutVZ", 10.f, "option to configure z-vertex cut"};

  HistogramRegistry histograms{"histograms"};

  void init(InitContext&)
  {
    AxisSpec dcaAxis = {800, -4., 4.};

    histograms.add("hTrkPrimReco", "Reco Prim tracks (charged); #it{p}_{T} (GeV/#it{c}); Counts", {kTH1F, {{ptBins}}});
    histograms.add("hTrkPrimReco_truepid_el", "Primary Reco tracks (true El); #it{p}_{T} (GeV/#it{c}); Counts", {kTH1F, {{ptBins}}});
    histograms.add("hTrkPrimReco_truepid_pi", "Primary Reco tracks (true Pi); #it{p}_{T} (GeV/#it{c}); Counts", {kTH1F, {{ptBins}}});
    histograms.add("hTrkPrimReco_truepid_ka", "Primary Reco tracks (true Ka); #it{p}_{T} (GeV/#it{c}); Counts", {kTH1F, {{ptBins}}});
    histograms.add("hTrkPrimReco_truepid_pr", "Primary Reco tracks (true Pr); #it{p}_{T} (GeV/#it{c}); Counts", {kTH1F, {{ptBins}}});
    histograms.add("hTrkPrimReco_truepid_de", "Primary Reco tracks (true De); #it{p}_{T} (GeV/#it{c}); Counts", {kTH1F, {{ptBins}}});

    histograms.add("hDCAxyReco_truepid_el", "DCAxy reco (true El); #it{p}_{T} (GeV/#it{c}); DCAxy (cm)", {kTH2F, {{ptBins}, dcaAxis}});
    histograms.add("hDCAxyReco_truepid_pi", "DCAxy reco (true Pi); #it{p}_{T} (GeV/#it{c}); DCAxy (cm)", {kTH2F, {{ptBins}, dcaAxis}});
    histograms.add("hDCAxyReco_truepid_ka", "DCAxy reco (true Ka); #it{p}_{T} (GeV/#it{c}); DCAxy (cm)", {kTH2F, {{ptBins}, dcaAxis}});
    histograms.add("hDCAxyReco_truepid_pr", "DCAxy reco (true Pr); #it{p}_{T} (GeV/#it{c}); DCAxy (cm)", {kTH2F, {{ptBins}, dcaAxis}});
    histograms.add("hDCAxyReco_truepid_de", "DCAxy reco (true De); #it{p}_{T} (GeV/#it{c}); DCAxy (cm)", {kTH2F, {{ptBins}, dcaAxis}});

    histograms.add("hDCAxyPrim_truepid_el", "DCAxy primaries (true El); #it{p}_{T} (GeV/#it{c}); DCAxy (cm)", {kTH2F, {{ptBins}, dcaAxis}});
    histograms.add("hDCAxyPrim_truepid_pi", "DCAxy primaries (true Pi); #it{p}_{T} (GeV/#it{c}); DCAxy (cm)", {kTH2F, {{ptBins}, dcaAxis}});
    histograms.add("hDCAxyPrim_truepid_ka", "DCAxy primaries (true Ka); #it{p}_{T} (GeV/#it{c}); DCAxy (cm)", {kTH2F, {{ptBins}, dcaAxis}});
    histograms.add("hDCAxyPrim_truepid_pr", "DCAxy primaries (true Pr); #it{p}_{T} (GeV/#it{c}); DCAxy (cm)", {kTH2F, {{ptBins}, dcaAxis}});
    histograms.add("hDCAxyPrim_truepid_de", "DCAxy primaries (true De); #it{p}_{T} (GeV/#it{c}); DCAxy (cm)", {kTH2F, {{ptBins}, dcaAxis}});

    histograms.add("hDCAxySeco_truepid_el", "DCAxy secondaries (true El); #it{p}_{T} (GeV/#it{c}); DCAxy (cm)", {kTH2F, {{ptBins}, dcaAxis}});
    histograms.add("hDCAxySeco_truepid_pi", "DCAxy secondaries (true Pi); #it{p}_{T} (GeV/#it{c}); DCAxy (cm)", {kTH2F, {{ptBins}, dcaAxis}});
    histograms.add("hDCAxySeco_truepid_ka", "DCAxy secondaries (true Ka); #it{p}_{T} (GeV/#it{c}); DCAxy (cm)", {kTH2F, {{ptBins}, dcaAxis}});
    histograms.add("hDCAxySeco_truepid_pr", "DCAxy secondaries (true Pr); #it{p}_{T} (GeV/#it{c}); DCAxy (cm)", {kTH2F, {{ptBins}, dcaAxis}});
    histograms.add("hDCAxySeco_truepid_de", "DCAxy secondaries (true De); #it{p}_{T} (GeV/#it{c}); DCAxy (cm)", {kTH2F, {{ptBins}, dcaAxis}});
  } //init

  //Filters
  Filter collfilter = nabs(aod::collision::posZ) < cfgCutVZ;
  Filter trackfilter = aod::track::isGlobalTrack == (uint8_t) true;
  void process(soa::Filtered<soa::Join<aod::Collisions, aod::EvSels>>::iterator const& col,
               soa::Filtered<soa::Join<aod::Tracks, aod::TracksExtra, aod::TracksExtended,
                                       aod::TrackSelection, aod::McTrackLabels>>& tracks,
               aod::McParticles& mcParticles)
  {

    //event selection
    if (!isMC && !col.alias()[kINT7]) { // trigger (should be skipped in MC)
      return;
    }
    if (!col.sel7()) { //additional cuts
      return;
    }

    //Loop on tracks
    for (auto& track : tracks) {
      double y = -999.;
      bool isPrimary = false;

      if (isMC) { //determine particle species base on MC truth and if it is primary or not
        int pdgcode = track.mcParticle().pdgCode();

        if (MC::isPhysicalPrimary(mcParticles, track.mcParticle())) { //is primary?
          isPrimary = true;
        }
        //Calculate y
        int pint = getparticleint(pdgcode);
        if (pint == kNull) {
          y = (double)kNull;
        } else {
          y = eta2y(track.pt(), mass[pint], track.eta());
        }

        //DCAxy distributions for reco, primary, secondaries (= not primary)
        // + distributions for reco primaries (MC truth)
        if (abs(y) < cfgCutY) {
          if (pdgcode == 11) {
            histograms.fill(HIST("hDCAxyReco_truepid_el"), track.pt(), track.dcaXY());
            if (isPrimary) {
              histograms.fill(HIST("hDCAxyPrim_truepid_el"), track.pt(), track.dcaXY());
              histograms.fill(HIST("hTrkPrimReco_truepid_el"), track.pt());
            } else {
              histograms.fill(HIST("hDCAxySeco_truepid_el"), track.pt(), track.dcaXY());
            }
          } else if (pdgcode == 211) {
            histograms.fill(HIST("hDCAxyReco_truepid_pi"), track.pt(), track.dcaXY());
            if (isPrimary) {
              histograms.fill(HIST("hDCAxyPrim_truepid_pi"), track.pt(), track.dcaXY());
              histograms.fill(HIST("hTrkPrimReco_truepid_pi"), track.pt());
            } else {
              histograms.fill(HIST("hDCAxySeco_truepid_pi"), track.pt(), track.dcaXY());
            }
          } else if (pdgcode == 321) {
            histograms.fill(HIST("hDCAxyReco_truepid_ka"), track.pt(), track.dcaXY());
            if (isPrimary) {
              histograms.fill(HIST("hDCAxyPrim_truepid_ka"), track.pt(), track.dcaXY());
              histograms.fill(HIST("hTrkPrimReco_truepid_ka"), track.pt());
            } else {
              histograms.fill(HIST("hDCAxySeco_truepid_ka"), track.pt(), track.dcaXY());
            }
          } else if (pdgcode == 2212) {
            histograms.fill(HIST("hDCAxyReco_truepid_pr"), track.pt(), track.dcaXY());
            if (isPrimary) {
              histograms.fill(HIST("hDCAxyPrim_truepid_pr"), track.pt(), track.dcaXY());
              histograms.fill(HIST("hTrkPrimReco_truepid_pr"), track.pt());
            } else {
              histograms.fill(HIST("hDCAxySeco_truepid_pr"), track.pt(), track.dcaXY());
            }
          } else if (pdgcode == 1000010020) {
            histograms.fill(HIST("hDCAxyReco_truepid_de"), track.pt(), track.dcaXY());
            if (isPrimary) {
              histograms.fill(HIST("hDCAxyPrim_truepid_de"), track.pt(), track.dcaXY());
              histograms.fill(HIST("hTrkPrimReco_truepid_de"), track.pt());
            } else {
              histograms.fill(HIST("hDCAxySeco_truepid_de"), track.pt(), track.dcaXY());
            }
          }
        }

        //reco histos (charged particles)
        if (isPrimary && abs(y) < cfgCutY) {
          histograms.fill(HIST("hTrkPrimReco"), track.pt()); // charged particles
        }
      }
    } // end loop on tracks
  }
}; // struct TrackCheckTask1

WorkflowSpec defineDataProcessing(ConfigContext const& cfgc)
{
  return WorkflowSpec{
    adaptAnalysisTask<TrackCheckTaskEvSel>(cfgc, TaskName{"track-histos-evsel"}),
    adaptAnalysisTask<TrackCheckTaskEvSelTrackSel>(cfgc, TaskName{"track-histos-evsel-trksel"})};
}
