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
const double pt_bins[NPTBINS + 1] = {0.0, 0.05, 0.08, 0.1, 0.12, 0.14, 0.16, 0.18, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65,
                                     0.7, 0.75, 0.8, 0.85, 0.9, 0.95, 1.0, 1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7, 1.8, 1.9, 2.0, 2.2,
                                     2.4, 2.6, 2.8, 3.0, 3.2, 3.4, 3.6, 3.8, 4.0, 4.5, 5.0, 5.5, 6.0, 6.5, 7.0, 8.0, 10.0, 20.0};
const double mass[NPARTICLES] = {0.000510999, 0.139570, 0.493677, 0.938272, 1.87561};
enum PType { kEl,
             kPi,
             kKa,
             kPr,
             kDe };

#define TRACKSELECTION                                                                                                                                                                                                                                               \
  UChar_t clustermap = track.itsClusterMap();                                                                                                                                                                                                                        \
  bool issel = (abs(track.eta()) < 0.8) && (track.tpcNClsFindable() > 70) && (track.tpcChi2NCl() < 4.) && (track.tpcCrossedRowsOverFindableCls() > 0.8) && (track.flags() & 0x4) && (track.itsChi2NCl() < 36) && (TESTBIT(clustermap, 0) || TESTBIT(clustermap, 1)); \
  issel = issel && (track.flags() & (uint64_t(1) << 2)); /*ITS refit*/                                                                                                                                                                                               \
  issel = issel && (track.flags() & (uint64_t(1) << 6)); /*TPC refit*/                                                                                                                                                                                               \
  if (!issel)                                                                                                                                                                                                                                                        \
    continue;

struct TrackCheckTask1 {

  Configurable<bool> isMC{"isMC", false, "option to flag mc"};
  OutputObj<TH1F> hTrkPrimReco{TH1F("hTrkPrimReco", "Reco Prim tracks (charged); #it{p}_{T} (GeV/#it{c}); Counts", NPTBINS, pt_bins)};
  OutputObj<TH1F> hTrkPrimAftEvSel{TH1F("hTrkPrimAftEvSel", "Reco Prim tracks AftEvSel (charged); #it{p}_{T} (GeV/#it{c}); Counts", NPTBINS, pt_bins)};
  OutputObj<HistArray> hTrkPrimReco_truepid{HistArray("hTrkPrimReco_truepid"), OutputObjHandlingPolicy::AnalysisObject};
  OutputObj<HistArray> hTrkPrimAftEvSel_truepid{HistArray("hTrkPrimAftEvSel_truepid"), OutputObjHandlingPolicy::AnalysisObject};
  OutputObj<HistArray> hDCAxyReco_truepid{HistArray("hDCAxyReco_truepid"), OutputObjHandlingPolicy::AnalysisObject};
  OutputObj<HistArray> hDCAxyPrim_truepid{HistArray("hDCAxyPrim_truepid"), OutputObjHandlingPolicy::AnalysisObject};
  OutputObj<HistArray> hDCAxySeco_truepid{HistArray("hDCAxySeco_truepid"), OutputObjHandlingPolicy::AnalysisObject};
  TF1* fCutDCAxy;

  void init(InitContext&)
  {
    hTrkPrimReco_truepid->Add(0, TH1F("hTrkPrimReco_truepid_el", "Primary Reco tracks (true El); #it{p}_{T} (GeV/#it{c}); Counts", NPTBINS, pt_bins));
    hTrkPrimReco_truepid->Add(1, TH1F("hTrkPrimReco_truepid_pi", "Primary Reco tracks (true Pi); #it{p}_{T} (GeV/#it{c}); Counts", NPTBINS, pt_bins));
    hTrkPrimReco_truepid->Add(2, TH1F("hTrkPrimReco_truepid_ka", "Primary Reco tracks (true Ka); #it{p}_{T} (GeV/#it{c}); Counts", NPTBINS, pt_bins));
    hTrkPrimReco_truepid->Add(3, TH1F("hTrkPrimReco_truepid_pr", "Primary Reco tracks (true Pr); #it{p}_{T} (GeV/#it{c}); Counts", NPTBINS, pt_bins));
    hTrkPrimReco_truepid->Add(4, TH1F("hTrkPrimReco_truepid_de", "Primary Reco tracks (true De); #it{p}_{T} (GeV/#it{c}); Counts", NPTBINS, pt_bins));

    hTrkPrimAftEvSel_truepid->Add(0, TH1F("hTrkPrimAftEvSel_truepid_el", "Generated tracks after event selection (true El); #it{p}_{T} (GeV/#it{c}); Counts", NPTBINS, pt_bins));
    hTrkPrimAftEvSel_truepid->Add(1, TH1F("hTrkPrimAftEvSel_truepid_pi", "Generated tracks after event selection (true Pi); #it{p}_{T} (GeV/#it{c}); Counts", NPTBINS, pt_bins));
    hTrkPrimAftEvSel_truepid->Add(2, TH1F("hTrkPrimAftEvSel_truepid_ka", "Generated tracks after event selection (true Ka); #it{p}_{T} (GeV/#it{c}); Counts", NPTBINS, pt_bins));
    hTrkPrimAftEvSel_truepid->Add(3, TH1F("hTrkPrimAftEvSel_truepid_pr", "Generated tracks after event selection (true Pr); #it{p}_{T} (GeV/#it{c}); Counts", NPTBINS, pt_bins));
    hTrkPrimAftEvSel_truepid->Add(4, TH1F("hTrkPrimAftEvSel_truepid_de", "Generated tracks after event selection (true De); #it{p}_{T} (GeV/#it{c}); Counts", NPTBINS, pt_bins));

    hDCAxyReco_truepid->Add(0, TH2F("hDCAxyReco_truepid_el", "DCAxy reco (true El); #it{p}_{T} (GeV/#it{c}); DCAxy (cm)", NPTBINS, pt_bins, 800, -4., 4.));
    hDCAxyReco_truepid->Add(1, TH2F("hDCAxyReco_truepid_pi", "DCAxy reco (true Pi); #it{p}_{T} (GeV/#it{c}); DCAxy (cm)", NPTBINS, pt_bins, 800, -4., 4.));
    hDCAxyReco_truepid->Add(2, TH2F("hDCAxyReco_truepid_ka", "DCAxy reco (true Ka); #it{p}_{T} (GeV/#it{c}); DCAxy (cm)", NPTBINS, pt_bins, 800, -4., 4.));
    hDCAxyReco_truepid->Add(3, TH2F("hDCAxyReco_truepid_pr", "DCAxy reco (true Pr); #it{p}_{T} (GeV/#it{c}); DCAxy (cm)", NPTBINS, pt_bins, 800, -4., 4.));
    hDCAxyReco_truepid->Add(4, TH2F("hDCAxyReco_truepid_de", "DCAxy reco (true De); #it{p}_{T} (GeV/#it{c}); DCAxy (cm)", NPTBINS, pt_bins, 800, -4., 4.));

    hDCAxyPrim_truepid->Add(0, TH2F("hDCAxyPrim_truepid_el", "DCAxy primaries (true El); #it{p}_{T} (GeV/#it{c}); DCAxy (cm)", NPTBINS, pt_bins, 800, -4., 4.));
    hDCAxyPrim_truepid->Add(1, TH2F("hDCAxyPrim_truepid_pi", "DCAxy primaries (true Pi); #it{p}_{T} (GeV/#it{c}); DCAxy (cm)", NPTBINS, pt_bins, 800, -4., 4.));
    hDCAxyPrim_truepid->Add(2, TH2F("hDCAxyPrim_truepid_ka", "DCAxy primaries (true Ka); #it{p}_{T} (GeV/#it{c}); DCAxy (cm)", NPTBINS, pt_bins, 800, -4., 4.));
    hDCAxyPrim_truepid->Add(3, TH2F("hDCAxyPrim_truepid_pr", "DCAxy primaries (true Pr); #it{p}_{T} (GeV/#it{c}); DCAxy (cm)", NPTBINS, pt_bins, 800, -4., 4.));
    hDCAxyPrim_truepid->Add(4, TH2F("hDCAxyPrim_truepid_de", "DCAxy primaries (true De); #it{p}_{T} (GeV/#it{c}); DCAxy (cm)", NPTBINS, pt_bins, 800, -4., 4.));

    hDCAxySeco_truepid->Add(0, TH2F("hDCAxySeco_truepid_el", "DCAxy secondaries (true El); #it{p}_{T} (GeV/#it{c}); DCAxy (cm)", NPTBINS, pt_bins, 800, -4., 4.));
    hDCAxySeco_truepid->Add(1, TH2F("hDCAxySeco_truepid_pi", "DCAxy secondaries (true Pi); #it{p}_{T} (GeV/#it{c}); DCAxy (cm)", NPTBINS, pt_bins, 800, -4., 4.));
    hDCAxySeco_truepid->Add(2, TH2F("hDCAxySeco_truepid_ka", "DCAxy secondaries (true Ka); #it{p}_{T} (GeV/#it{c}); DCAxy (cm)", NPTBINS, pt_bins, 800, -4., 4.));
    hDCAxySeco_truepid->Add(3, TH2F("hDCAxySeco_truepid_pr", "DCAxy secondaries (true Pr); #it{p}_{T} (GeV/#it{c}); DCAxy (cm)", NPTBINS, pt_bins, 800, -4., 4.));
    hDCAxySeco_truepid->Add(4, TH2F("hDCAxySeco_truepid_de", "DCAxy secondaries (true De); #it{p}_{T} (GeV/#it{c}); DCAxy (cm)", NPTBINS, pt_bins, 800, -4., 4.));

    fCutDCAxy = new TF1("fMaxDCAxy", "[0]+[1]/(x^[2])", 0, 1e10); //from TPC analysis task (Run2)
    fCutDCAxy->SetParameter(0, 0.0105);
    fCutDCAxy->SetParameter(1, 0.0350);
    fCutDCAxy->SetParameter(2, 1.1);
  } //init

  //Filters
  Filter collfilter = nabs(aod::collision::posZ) < (float)10.;
  void process(soa::Filtered<soa::Join<aod::Collisions, aod::EvSels>>::iterator const& col, soa::Join<aod::Tracks, aod::TracksExtra, aod::McTrackLabels>& tracks, aod::McParticles& mcParticles)
  {

    if (isMC) {
      LOGF(info, "isMC=%d - Running on MC", isMC.value);
    } else {
      LOGF(info, "isMC=%d - Running on DATA", isMC.value);
    }

    //event selection
    if (!isMC && !col.alias()[kINT7]) // trigger (should be skipped in MC)
      return;
    if (!col.sel7()) //additional cuts
      return;

    //Loop on tracks
    for (auto& track : tracks) {
      double y = -999.;
      int pdgcode = -999;
      int ptype = -999;
      bool isPrimary = false;

      if (isMC) { //determine particle species base on MC truth and if it is primary or not
        pdgcode = track.label().pdgCode();
        if (pdgcode == 11)
          ptype = kEl;
        else if (pdgcode == 211)
          ptype = kPi;
        else if (pdgcode == 321)
          ptype = kKa;
        else if (pdgcode == 2212)
          ptype = kPr;
        else if (pdgcode == 1000010020)
          ptype = kDe;

        if (MC::isPhysicalPrimary(mcParticles, track.label())) { //is primary?
          isPrimary = true;
        }
      }

      if (isMC) {
        //calculate rapidity
        y = eta2y(track.pt(), mass[ptype], track.eta());

        if (isPrimary && abs(y) < 0.5) {
          //histograms with generated distribution (after event selection)
          hTrkPrimAftEvSel_truepid->Fill(ptype, track.pt());
          hTrkPrimAftEvSel->Fill(track.pt()); //charged particles
        }
      }

      //track selection - DCAxy and DCAz cut done later - other cuts??
      TRACKSELECTION;

      //DCAxy distributions for reco, primary, secondaries (= not primary)
      //calculate dcaxy and dcaz
      std::pair<float, float> dcas = getdcaxyz(track, col);
      if (isMC) {
        if (abs(y) < 0.5) {
          hDCAxyReco_truepid->Fill(ptype, track.pt(), dcas.first);
          if (isPrimary)
            hDCAxyPrim_truepid->Fill(ptype, track.pt(), dcas.first);
          else
            hDCAxySeco_truepid->Fill(ptype, track.pt(), dcas.first);
        }
      }

      //Additional CUTS
      //apply cut on DCAz
      if (abs(dcas.second) > 2.0)
        continue;
      //apply pt-dependent cut on dcaxy
      if (abs(dcas.first) > fCutDCAxy->Eval(track.pt()))
        continue;

      //fill reco histos
      if (isMC) {
        if (isPrimary && abs(y) < 0.5) {
          hTrkPrimReco_truepid->Fill(ptype, track.pt());
          hTrkPrimReco->Fill(track.pt()); //charged particles
        }
      }
    } // end loop on tracks
  }

  //convert eta to y
  double eta2y(double pt, double m, double eta) const
  {
    // convert eta to y
    double mt = sqrt(m * m + pt * pt);
    return asinh(pt / mt * sinh(eta));
  }

  //calculated DCAxy and DCAz
  template <typename T, typename C>
  std::pair<float, float> getdcaxyz(T& track, C const& collision)
  {
    float sinAlpha = 0.f;
    float cosAlpha = 0.f;
    float globalX = 0.f;
    float globalY = 0.f;
    float dcaXY = 0.f;
    float dcaZ = 0.f;

    sinAlpha = sin(track.alpha());
    cosAlpha = cos(track.alpha());
    globalX = track.x() * cosAlpha - track.y() * sinAlpha;
    globalY = track.x() * sinAlpha + track.y() * cosAlpha;

    dcaXY = track.charge() * sqrt(pow((globalX - collision.posX()), 2) +
                                  pow((globalY - collision.posY()), 2));
    dcaZ = track.charge() * sqrt(pow(track.z() - collision.posZ(), 2));

    return std::make_pair(dcaXY, dcaZ);
  }

}; // struct TrackCheckTask1

WorkflowSpec defineDataProcessing(ConfigContext const&)
{
  return WorkflowSpec{
    adaptAnalysisTask<TrackCheckTask1>("track-histos")};
}
