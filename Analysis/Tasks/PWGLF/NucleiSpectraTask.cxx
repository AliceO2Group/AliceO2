// Copyright 2019-2020 CERN and copyright holders of ALICE O2.
// See https://alice-o2.web.cern.ch/copyright for details of the copyright holders.
// All rights not expressly granted are reserved.
//
// This software is distributed under the terms of the GNU General Public
// License v3 (GPL Version 3), copied verbatim in the file "COPYING".
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.
// O2 includes

#include "ReconstructionDataFormats/Track.h"
#include "Framework/runDataProcessing.h"
#include "Framework/AnalysisTask.h"
#include "Framework/AnalysisDataModel.h"
#include "Framework/ASoAHelpers.h"
#include "AnalysisDataModel/PID/PIDResponse.h"
#include "AnalysisDataModel/TrackSelectionTables.h"

#include "AnalysisDataModel/EventSelection.h"
#include "AnalysisDataModel/TrackSelectionTables.h"
#include "AnalysisDataModel/Centrality.h"

#include "Framework/HistogramRegistry.h"

#include <TLorentzVector.h>
#include <TMath.h>
#include <TObjArray.h>

#include <cmath>

using namespace o2;
using namespace o2::framework;
using namespace o2::framework::expressions;

struct NucleiSpectraTask {

  HistogramRegistry spectra{"spectra", {}, OutputObjHandlingPolicy::AnalysisObject, true, true};

  void init(o2::framework::InitContext&)
  {
    std::vector<double> ptBinning = {0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3, 1.4, 1.5, 1.6,
                                     1.8, 2.0, 2.2, 2.4, 2.8, 3.2, 3.6, 4., 5., 6., 8., 10., 12., 14.};
    std::vector<double> centBinning = {0., 1., 5., 10., 20., 30., 40., 50., 70., 100.};

    AxisSpec ptAxis = {ptBinning, "#it{p}_{T} (GeV/#it{c})"};
    AxisSpec centAxis = {centBinning, "V0M (%)"};

    spectra.add("histRecVtxZData", "collision z position", HistType::kTH1F, {{200, -20., +20., "z position (cm)"}});
    spectra.add("histKeepEventData", "skimming histogram", HistType::kTH1F, {{2, -0.5, +1.5, "true: keep event, false: reject event"}});
    spectra.add("histTpcSignalData", "Specific energy loss", HistType::kTH2F, {{600, -6., 6., "#it{p} (GeV/#it{c})"}, {1400, 0, 1400, "d#it{E} / d#it{X} (a. u.)"}});
    spectra.add("histTofSignalData", "TOF signal", HistType::kTH2F, {{600, -6., 6., "#it{p} (GeV/#it{c})"}, {500, 0.0, 1.0, "#beta (TOF)"}});
    spectra.add("histTpcNsigmaData", "n-sigma TPC", HistType::kTH2F, {ptAxis, {200, -100., +100., "n#sigma_{He} (a. u.)"}});
    spectra.add("histDcaVsPtData", "dca vs Pt", HistType::kTH2F, {ptAxis, {400, -0.2, 0.2, "dca"}});
    spectra.add("histInvMassData", "Invariant mass", HistType::kTH1F, {{600, 5.0, +15., "inv. mass GeV/c^{2}"}});
  }

  Configurable<float> yMin{"yMin", -0.5, "Maximum rapidity"};
  Configurable<float> yMax{"yMax", 0.5, "Minimum rapidity"};

  Configurable<float> cfgCutVertex{"cfgCutVertex", 10.0f, "Accepted z-vertex range"};
  Configurable<float> cfgCutEta{"cfgCutEta", 0.8f, "Eta range for tracks"};
  Configurable<float> nsigmacutLow{"nsigmacutLow", -10.0, "Value of the Nsigma cut"};
  Configurable<float> nsigmacutHigh{"nsigmacutHigh", +10.0, "Value of the Nsigma cut"};

  Filter collisionFilter = nabs(aod::collision::posZ) < cfgCutVertex;
  Filter trackFilter = (nabs(aod::track::eta) < cfgCutEta) && (aod::track::isGlobalTrack == (uint8_t) true);

  using TrackCandidates = soa::Filtered<soa::Join<aod::Tracks, aod::TracksExtra, aod::TracksExtended, aod::pidTPCFullHe, aod::pidTOFFullHe, aod::TrackSelection>>;

  void process(soa::Filtered<soa::Join<aod::Collisions, aod::EvSels>>::iterator const& collision, TrackCandidates const& tracks)
  {
    //
    // collision process loop
    //
    bool keepEvent = kFALSE;
    //
    spectra.fill(HIST("histRecVtxZData"), collision.posZ());
    //
    std::vector<TLorentzVector> posTracks;
    std::vector<TLorentzVector> negTracks;
    //
    for (auto track : tracks) { // start loop over tracks

      TLorentzVector lorentzVector{};
      lorentzVector.SetPtEtaPhiM(track.pt() * 2.0, track.eta(), track.phi(), constants::physics::MassHelium3);
      if (lorentzVector.Rapidity() < yMin || lorentzVector.Rapidity() > yMax) {
        continue;
      }
      //
      // fill QA histograms
      //
      float nSigmaHe3 = track.tpcNSigmaHe();
      nSigmaHe3 += 94.222101 * TMath::Exp(-0.905203 * track.tpcInnerParam());
      //
      spectra.fill(HIST("histTpcSignalData"), track.tpcInnerParam() * track.sign(), track.tpcSignal());
      if (track.sign() < 0) {
        spectra.fill(HIST("histTpcNsigmaData"), track.pt() * 2.0, nSigmaHe3);
      }
      //
      // check offline-trigger (skimming) condidition
      //
      if (nSigmaHe3 > nsigmacutLow && nSigmaHe3 < nsigmacutHigh) {
        keepEvent = kTRUE;
        if (track.sign() < 0) {
          spectra.fill(HIST("histDcaVsPtData"), track.pt() * 2.0, track.dcaXY());
        }
        //
        // store tracks for invariant mass calculation
        //
        if (track.sign() < 0) {
          negTracks.push_back(lorentzVector);
        }
        if (track.sign() > 0) {
          posTracks.push_back(lorentzVector);
        }
        //
        // calculate beta
        //
        if (!track.hasTOF()) {
          continue;
        }
        Float_t tofTime = track.tofSignal();
        Float_t tofLength = track.length();
        Float_t beta = tofLength / (TMath::C() * 1e-10 * tofTime);
        spectra.fill(HIST("histTofSignalData"), track.tpcInnerParam() * track.sign(), beta);
      }

    } // end loop over tracks
    //
    // fill trigger (skimming) results
    //
    spectra.fill(HIST("histKeepEventData"), keepEvent);
    //
    // calculate invariant mass
    //
    for (Int_t iPos = 0; iPos < posTracks.size(); iPos++) {
      TLorentzVector& vecPos = posTracks[iPos];
      for (Int_t jNeg = 0; jNeg < negTracks.size(); jNeg++) {
        TLorentzVector& vecNeg = negTracks[jNeg];
        TLorentzVector vecMother = vecPos + vecNeg;
        spectra.fill(HIST("histInvMassData"), vecMother.M());
      }
    }
  }
};

WorkflowSpec defineDataProcessing(ConfigContext const& cfgc)
{
  return WorkflowSpec{
    adaptAnalysisTask<NucleiSpectraTask>(cfgc, TaskName{"nuclei-spectra"})};
}
