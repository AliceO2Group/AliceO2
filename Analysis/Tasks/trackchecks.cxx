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
#include "Analysis/EventSelection.h"
#include "Analysis/MC.h"
#include "Analysis/HistHelpers.h"
#include "Analysis/TrackSelection.h"
#include "Analysis/TrackSelectionTables.h"

#include <cmath>
#include <array>
#include <utility>

#include <TH1.h>
#include <TH2.h>
#include <TF1.h>

using namespace o2;
using namespace o2::framework;
using namespace o2::framework::expressions;
using namespace o2::experimental::histhelpers;
using namespace std;

#define NPTBINS 53
#define NPARTICLES 5
const double pt_bins[NPTBINS + 1] = {0.0, 0.05, 0.08, 0.1, 0.12, 0.14, 0.16, 0.18, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45,
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
  if (pdgcode == 11)
    return kEl;
  else if (pdgcode == 211)
    return kPi;
  else if (pdgcode == 321)
    return kKa;
  else if (pdgcode == 2212)
    return kPr;
  else if (pdgcode == 1000010020)
    return kDe;
  else
    return kNull;
}

//No track selection --> only event selection here
struct TrackCheckTaskEvSel {

  Configurable<bool> isMC{"isMC", false, "option to flag mc"};
  Configurable<double> cfgCutY{"cfgCutY", 0.5, "option to configure rapidity cut"};
  Configurable<float> cfgCutVZ{"cfgCutVZ", 10.f, "option to configure z-vertex cut"};
  OutputObj<TH1F> hTrkPrimAftEvSel{TH1F("hTrkPrimAftEvSel",
                                        "Reco Prim tracks AftEvSel (charged); #it{p}_{T} (GeV/#it{c}); Counts", NPTBINS, pt_bins)};
  OutputObj<HistArray> hTrkPrimAftEvSel_truepid{HistArray("hTrkPrimAftEvSel_truepid"),
                                                OutputObjHandlingPolicy::AnalysisObject};

  void init(InitContext&)
  {
    hTrkPrimAftEvSel_truepid->Add<kEl>(TH1F("hTrkPrimAftEvSel_truepid_el",
                                            "Generated tracks after event selection (true El); #it{p}_{T} (GeV/#it{c}); Counts", NPTBINS, pt_bins));
    hTrkPrimAftEvSel_truepid->Add<kPi>(TH1F("hTrkPrimAftEvSel_truepid_pi",
                                            "Generated tracks after event selection (true Pi); #it{p}_{T} (GeV/#it{c}); Counts", NPTBINS, pt_bins));
    hTrkPrimAftEvSel_truepid->Add<kKa>(TH1F("hTrkPrimAftEvSel_truepid_ka",
                                            "Generated tracks after event selection (true Ka); #it{p}_{T} (GeV/#it{c}); Counts", NPTBINS, pt_bins));
    hTrkPrimAftEvSel_truepid->Add<kPr>(TH1F("hTrkPrimAftEvSel_truepid_pr",
                                            "Generated tracks after event selection (true Pr); #it{p}_{T} (GeV/#it{c}); Counts", NPTBINS, pt_bins));
    hTrkPrimAftEvSel_truepid->Add<kDe>(TH1F("hTrkPrimAftEvSel_truepid_de",
                                            "Generated tracks after event selection (true De); #it{p}_{T} (GeV/#it{c}); Counts", NPTBINS, pt_bins));
  } //init

  //Filters
  Filter collfilter = nabs(aod::collision::posZ) < cfgCutVZ;
  void process(soa::Filtered<soa::Join<aod::Collisions, aod::EvSels>>::iterator const& col,
               soa::Join<aod::Tracks, aod::TracksExtra, aod::McTrackLabels>& tracks, aod::McParticles& mcParticles)
  {

    //event selection
    if (!isMC && !col.alias()[kINT7]) // trigger (should be skipped in MC)
      return;
    if (!col.sel7()) //additional cuts
      return;

    //Loop on tracks
    for (auto& track : tracks) {
      double y = -999.;
      bool isPrimary = false;

      if (isMC) { //determine particle species base on MC truth and if it is primary or not
        int pdgcode = track.label().pdgCode();

        if (MC::isPhysicalPrimary(mcParticles, track.label())) { //is primary?
          isPrimary = true;
        }

        //calculate rapidity
        int pint = getparticleint(pdgcode);
        if (pint == kNull)
          y = (double)kNull;
        else
          y = eta2y(track.pt(), mass[pint], track.eta());

        if (isPrimary && abs(y) < cfgCutY) {
          //histograms with generated distribution (after event selection)
          if (pdgcode == 11)
            hTrkPrimAftEvSel_truepid->Fill<kEl>(track.pt());
          else if (pdgcode == 211)
            hTrkPrimAftEvSel_truepid->Fill<kPi>(track.pt());
          else if (pdgcode == 321)
            hTrkPrimAftEvSel_truepid->Fill<kKa>(track.pt());
          else if (pdgcode == 2212)
            hTrkPrimAftEvSel_truepid->Fill<kPr>(track.pt());
          else if (pdgcode == 1000010020)
            hTrkPrimAftEvSel_truepid->Fill<kDe>(track.pt());

          hTrkPrimAftEvSel->Fill(track.pt()); //charged particles
        }
      }
    } // end loop on tracks
  }   //end of process
};    //end struct TrackCheckTaskEvSel

//event selection + track selection here
struct TrackCheckTaskEvSelTrackSel {

  Configurable<bool> isMC{"isMC", false, "option to flag mc"};
  Configurable<double> cfgCutY{"cfgCutY", 0.5, "option to configure rapidity cut"};
  Configurable<float> cfgCutVZ{"cfgCutVZ", 10.f, "option to configure z-vertex cut"};
  OutputObj<TH1F> hTrkPrimReco{TH1F("hTrkPrimReco", "Reco Prim tracks (charged); #it{p}_{T} (GeV/#it{c}); Counts",
                                    NPTBINS, pt_bins)};
  OutputObj<HistArray> hTrkPrimReco_truepid{HistArray("hTrkPrimReco_truepid"),
                                            OutputObjHandlingPolicy::AnalysisObject};
  OutputObj<HistArray> hDCAxyReco_truepid{HistArray("hDCAxyReco_truepid"), OutputObjHandlingPolicy::AnalysisObject};
  OutputObj<HistArray> hDCAxyPrim_truepid{HistArray("hDCAxyPrim_truepid"), OutputObjHandlingPolicy::AnalysisObject};
  OutputObj<HistArray> hDCAxySeco_truepid{HistArray("hDCAxySeco_truepid"), OutputObjHandlingPolicy::AnalysisObject};

  void init(InitContext&)
  {
    hTrkPrimReco_truepid->Add<kEl>(TH1F("hTrkPrimReco_truepid_el",
                                        "Primary Reco tracks (true El); #it{p}_{T} (GeV/#it{c}); Counts", NPTBINS, pt_bins));
    hTrkPrimReco_truepid->Add<kPi>(TH1F("hTrkPrimReco_truepid_pi",
                                        "Primary Reco tracks (true Pi); #it{p}_{T} (GeV/#it{c}); Counts", NPTBINS, pt_bins));
    hTrkPrimReco_truepid->Add<kKa>(TH1F("hTrkPrimReco_truepid_ka",
                                        "Primary Reco tracks (true Ka); #it{p}_{T} (GeV/#it{c}); Counts", NPTBINS, pt_bins));
    hTrkPrimReco_truepid->Add<kPr>(TH1F("hTrkPrimReco_truepid_pr",
                                        "Primary Reco tracks (true Pr); #it{p}_{T} (GeV/#it{c}); Counts", NPTBINS, pt_bins));
    hTrkPrimReco_truepid->Add<kDe>(TH1F("hTrkPrimReco_truepid_de",
                                        "Primary Reco tracks (true De); #it{p}_{T} (GeV/#it{c}); Counts", NPTBINS, pt_bins));

    hDCAxyReco_truepid->Add<kEl>(TH2F("hDCAxyReco_truepid_el",
                                      "DCAxy reco (true El); #it{p}_{T} (GeV/#it{c}); DCAxy (cm)", NPTBINS, pt_bins, 800, -4., 4.));
    hDCAxyReco_truepid->Add<kPi>(TH2F("hDCAxyReco_truepid_pi",
                                      "DCAxy reco (true Pi); #it{p}_{T} (GeV/#it{c}); DCAxy (cm)", NPTBINS, pt_bins, 800, -4., 4.));
    hDCAxyReco_truepid->Add<kKa>(TH2F("hDCAxyReco_truepid_ka",
                                      "DCAxy reco (true Ka); #it{p}_{T} (GeV/#it{c}); DCAxy (cm)", NPTBINS, pt_bins, 800, -4., 4.));
    hDCAxyReco_truepid->Add<kPr>(TH2F("hDCAxyReco_truepid_pr",
                                      "DCAxy reco (true Pr); #it{p}_{T} (GeV/#it{c}); DCAxy (cm)", NPTBINS, pt_bins, 800, -4., 4.));
    hDCAxyReco_truepid->Add<kDe>(TH2F("hDCAxyReco_truepid_de",
                                      "DCAxy reco (true De); #it{p}_{T} (GeV/#it{c}); DCAxy (cm)", NPTBINS, pt_bins, 800, -4., 4.));

    hDCAxyPrim_truepid->Add<kEl>(TH2F("hDCAxyPrim_truepid_el",
                                      "DCAxy primaries (true El); #it{p}_{T} (GeV/#it{c}); DCAxy (cm)", NPTBINS, pt_bins, 800, -4., 4.));
    hDCAxyPrim_truepid->Add<kPi>(TH2F("hDCAxyPrim_truepid_pi",
                                      "DCAxy primaries (true Pi); #it{p}_{T} (GeV/#it{c}); DCAxy (cm)", NPTBINS, pt_bins, 800, -4., 4.));
    hDCAxyPrim_truepid->Add<kKa>(TH2F("hDCAxyPrim_truepid_ka",
                                      "DCAxy primaries (true Ka); #it{p}_{T} (GeV/#it{c}); DCAxy (cm)", NPTBINS, pt_bins, 800, -4., 4.));
    hDCAxyPrim_truepid->Add<kPr>(TH2F("hDCAxyPrim_truepid_pr",
                                      "DCAxy primaries (true Pr); #it{p}_{T} (GeV/#it{c}); DCAxy (cm)", NPTBINS, pt_bins, 800, -4., 4.));
    hDCAxyPrim_truepid->Add<kDe>(TH2F("hDCAxyPrim_truepid_de",
                                      "DCAxy primaries (true De); #it{p}_{T} (GeV/#it{c}); DCAxy (cm)", NPTBINS, pt_bins, 800, -4., 4.));

    hDCAxySeco_truepid->Add<kEl>(TH2F("hDCAxySeco_truepid_el",
                                      "DCAxy secondaries (true El); #it{p}_{T} (GeV/#it{c}); DCAxy (cm)", NPTBINS, pt_bins, 800, -4., 4.));
    hDCAxySeco_truepid->Add<kPi>(TH2F("hDCAxySeco_truepid_pi",
                                      "DCAxy secondaries (true Pi); #it{p}_{T} (GeV/#it{c}); DCAxy (cm)", NPTBINS, pt_bins, 800, -4., 4.));
    hDCAxySeco_truepid->Add<kKa>(TH2F("hDCAxySeco_truepid_ka",
                                      "DCAxy secondaries (true Ka); #it{p}_{T} (GeV/#it{c}); DCAxy (cm)", NPTBINS, pt_bins, 800, -4., 4.));
    hDCAxySeco_truepid->Add<kPr>(TH2F("hDCAxySeco_truepid_pr",
                                      "DCAxy secondaries (true Pr); #it{p}_{T} (GeV/#it{c}); DCAxy (cm)", NPTBINS, pt_bins, 800, -4., 4.));
    hDCAxySeco_truepid->Add<kDe>(TH2F("hDCAxySeco_truepid_de",
                                      "DCAxy secondaries (true De); #it{p}_{T} (GeV/#it{c}); DCAxy (cm)", NPTBINS, pt_bins, 800, -4., 4.));

  } //init

  //Filters
  Filter collfilter = nabs(aod::collision::posZ) < cfgCutVZ;
  Filter trackfilter = aod::track::isGlobalTrack == true;
  void process(soa::Filtered<soa::Join<aod::Collisions, aod::EvSels>>::iterator const& col,
               soa::Filtered<soa::Join<aod::Tracks, aod::TracksExtra, aod::TracksExtended,
                                       aod::TrackSelection, aod::McTrackLabels>>& tracks,
               aod::McParticles& mcParticles)
  {

    //event selection
    if (!isMC && !col.alias()[kINT7]) // trigger (should be skipped in MC)
      return;
    if (!col.sel7()) //additional cuts
      return;

    //Loop on tracks
    for (auto& track : tracks) {
      double y = -999.;
      bool isPrimary = false;

      if (isMC) { //determine particle species base on MC truth and if it is primary or not
        int pdgcode = track.label().pdgCode();

        if (MC::isPhysicalPrimary(mcParticles, track.label())) { //is primary?
          isPrimary = true;
        }
        //Calculate y
        int pint = getparticleint(pdgcode);
        if (pint == kNull)
          y = (double)kNull;
        else
          y = eta2y(track.pt(), mass[pint], track.eta());

        //DCAxy distributions for reco, primary, secondaries (= not primary)
        // + distributions for reco primaries (MC truth)
        if (abs(y) < cfgCutY) {
          if (pdgcode == 11) {
            hDCAxyReco_truepid->Fill<kEl>(track.pt(), track.dcaXY());
            if (isPrimary) {
              hDCAxyPrim_truepid->Fill<kEl>(track.pt(), track.dcaXY());
              hTrkPrimReco_truepid->Fill<kEl>(track.pt());
            } else
              hDCAxySeco_truepid->Fill<kEl>(track.pt(), track.dcaXY());
          } else if (pdgcode == 211) {
            hDCAxyReco_truepid->Fill<kPi>(track.pt(), track.dcaXY());
            if (isPrimary) {
              hDCAxyPrim_truepid->Fill<kPi>(track.pt(), track.dcaXY());
              hTrkPrimReco_truepid->Fill<kPi>(track.pt());
            } else
              hDCAxySeco_truepid->Fill<kPi>(track.pt(), track.dcaXY());
          } else if (pdgcode == 321) {
            hDCAxyReco_truepid->Fill<kKa>(track.pt(), track.dcaXY());
            if (isPrimary) {
              hDCAxyPrim_truepid->Fill<kKa>(track.pt(), track.dcaXY());
              hTrkPrimReco_truepid->Fill<kKa>(track.pt());
            } else
              hDCAxySeco_truepid->Fill<kKa>(track.pt(), track.dcaXY());
          } else if (pdgcode == 2212) {
            hDCAxyReco_truepid->Fill<kPr>(track.pt(), track.dcaXY());
            if (isPrimary) {
              hDCAxyPrim_truepid->Fill<kPr>(track.pt(), track.dcaXY());
              hTrkPrimReco_truepid->Fill<kPr>(track.pt());
            } else
              hDCAxySeco_truepid->Fill<kPr>(track.pt(), track.dcaXY());
          } else if (pdgcode == 1000010020) {
            hDCAxyReco_truepid->Fill<kDe>(track.pt(), track.dcaXY());
            if (isPrimary) {
              hDCAxyPrim_truepid->Fill<kDe>(track.pt(), track.dcaXY());
              hTrkPrimReco_truepid->Fill<kDe>(track.pt());
            } else
              hDCAxySeco_truepid->Fill<kDe>(track.pt(), track.dcaXY());
          }
        }

        //reco histos (charged particles)
        if (isPrimary && abs(y) < cfgCutY) {
          hTrkPrimReco->Fill(track.pt()); //charged particles
        }
      }
    } // end loop on tracks
  }
}; // struct TrackCheckTask1

WorkflowSpec defineDataProcessing(ConfigContext const&)
{
  return WorkflowSpec{
    adaptAnalysisTask<TrackCheckTaskEvSel>("track-histos-evsel"),
    adaptAnalysisTask<TrackCheckTaskEvSelTrackSel>("track-histos-evsel-trksel")};
}
