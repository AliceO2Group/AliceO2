// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.
/// \author Peter Hristov <Peter.Hristov@cern.ch>, CERN
/// \author Gian Michele Innocenti <gian.michele.innocenti@cern.ch>, CERN
/// \author Henrique J C Zanoli <henrique.zanoli@cern.ch>, Utrecht University

#include "Framework/runDataProcessing.h"
#include "Framework/AnalysisTask.h"
#include "Framework/AnalysisDataModel.h"
#include "Analysis/trackUtilities.h"
#include "ReconstructionDataFormats/DCA.h"
#include "Analysis/MC.h"

#include <TH1F.h>
#include <TH2F.h>
#include <cmath>

namespace o2fw = o2::framework;
namespace o2exp = o2::framework::expressions;
namespace o2df = o2::dataformats;

/// Task to QA global observables of the event
struct QAGlobalObservables {
  std::array<float, 2> collisionPositionRange = {-20., 20.};
  int nBinsPosition{100};

  o2fw::OutputObj<TH1F> hCollisionX{TH1F("collisionX", "; X [cm];Counts", nBinsPosition,
                                         collisionPositionRange[0], collisionPositionRange[1])};

  o2fw::OutputObj<TH1F> hCollisionY{TH1F("collisionY", "; X [cm];Counts", nBinsPosition,
                                         collisionPositionRange[0], collisionPositionRange[1])};

  o2fw::OutputObj<TH1F> hCollisionZ{TH1F("collisionZ", "; X [cm];Counts", nBinsPosition,
                                         collisionPositionRange[0], collisionPositionRange[1])};

  o2fw::OutputObj<TH1F> hNumberOfTracks{TH1F("NumberOfTracks", "; Number of Tracks;Counts",
                                             nBinsPosition, collisionPositionRange[0],
                                             collisionPositionRange[1])};

  void process(const o2::aod::Collisions& collisions)
  {
    for (auto collision : collisions) {
      hCollisionX->Fill(collision.posX());
      hCollisionY->Fill(collision.posY());
      hCollisionZ->Fill(collision.posZ());
    }
  }
};

/// Task to QA the kinematic properties of the tracks
struct QATrackingKine {
  o2fw::Configurable<bool> isMC{"isMC", false, "Option to flag simulations (MC)"};

  o2fw::Configurable<int> nBinsPt{"nBinsPt", 100, "Number of bins for Pt"};
  std::array<double, 2> ptRange = {0, 10.};
  o2fw::Configurable<int> nBinsPhi{"nBinsPhi", 100, "Number of bins for Phi"};
  o2fw::Configurable<int> nBinsEta{"nBinsEta", 100,
                                   "Number of bins for the pseudorapidity histogram."};
  std::array<double, 2> etaRange = {-6, 6};

  o2fw::OutputObj<TH1F> hPt{TH1F("pt", ";p_{T} [GeV];Counts", nBinsPt, ptRange[0], ptRange[1])};
  o2fw::OutputObj<TH1F> hEta{TH1F("eta", ";#eta;Counts", nBinsEta, etaRange[0], etaRange[1])};
  o2fw::OutputObj<TH1F> hPhi{TH1F("phi", ";#phi;Counts", nBinsPhi, 0, 2 * M_PI)};

  void process(const o2::aod::Tracks& tracks)
  {
    for (const auto& track : tracks) {
      hEta->Fill(track.eta());
      hPhi->Fill(track.phi());
      hPt->Fill(track.pt());
    }
  }
};

/// Converts a angle phi to the -pi to pi range.
double ConvertPhiRange(double phi)
{
  if (phi > M_PI) {
    phi -= 2 * M_PI;
  } else if (phi < -M_PI) {
    phi += 2 * M_PI;
  }

  return phi;
}

/// Task to evaluate the tracking resolution (Pt, Eta, Phi and impact parameter)
struct QATrackingResolution {
  o2fw::Configurable<int> nBinsPtTrack{"nBinsPtTrack", 100,
                                       "Number of bins for the transverse momentum"};
  std::array<double, 2> ptRange = {0, 10.};

  o2fw::Configurable<int> nBinsEtaTrack{"nBinsEtaTrack", 400,
                                        "Number of bins for the pseudorapidity"};
  std::array<double, 2> etaRange = {-6, 6};

  o2fw::Configurable<int> nBinsDeltaPt{"nBinsDeltaPt", 400,
                                       "Number of bins for the transverse momentum differences"};

  o2fw::Configurable<int> nBinsDeltaPhi{"nBinsPhi", 100,
                                        "Number of bins for the azimuthal angle differences"};

  o2fw::Configurable<int> nBinsDeltaEta{"nBinsEta", 100,
                                        "Number of bins for the pseudorapidity differences"};

  o2fw::Configurable<int> nBinsImpactParameter{"nBinsImpactParameter", 1000,
                                               "Number of bins for the Impact parameter"};

  std::array<double, 2> impactParameterRange = {-1500, 1500};       // micrometer
  std::array<double, 2> impactParameterResolutionRange = {0, 1000}; // micrometer

  o2fw::OutputObj<TH1F> etaDiffMCRec{
    TH1F("etaDiffMCReco", ";eta_{MC} - eta_{Rec}", nBinsDeltaEta, -2, 2)};

  o2fw::OutputObj<TH1F> phiDiffMCRec{
    TH1F("phiDiffMCRec", ";phi_{MC} - phi_{Rec}", nBinsDeltaPhi, -M_PI, M_PI)};

  o2fw::OutputObj<TH1F> ptDiffMCRec{
    TH1F("ptDiffMCRec", ";p_{T}_{MC} - p_{T}_{Rec}", nBinsDeltaPt, -2., 2.)};

  o2fw::OutputObj<TH1F> ptResolution{
    TH1F("ptResolution", ";#frac{p_{T}_{MC} - p_{T}_{Rec}}{p_{T}_{Rec}} ", nBinsDeltaPt, -2., 2.)};

  o2fw::OutputObj<TH2F> ptResolutionVsPt{
    TH2F("ptResolutionVsPt", ";p_{T};Res p_{T}", nBinsPtTrack, 0., 10., nBinsDeltaPt, -2., 2.)};

  o2fw::OutputObj<TH2F> ptResolutionVsEta{
    TH2F("ptResolutionVsEta", ";#eta;Res p_{T}", nBinsEtaTrack, -4., 4., nBinsDeltaPt, -2., 2.)};

  o2fw::OutputObj<TH2F> impactParameterVsPt{
    TH2F("impactParameterVsPt", ";p_{T};Impact Parameter[#mu m]", nBinsPtTrack, ptRange[0],
         ptRange[1], nBinsImpactParameter, impactParameterRange[0], impactParameterRange[1])};

  o2fw::OutputObj<TH2F> impactParameterVsEta{
    TH2F("impactParameterVsPt", "#eta;Impact Parameter[#mu m]", nBinsEtaTrack, etaRange[0],
         etaRange[1], nBinsImpactParameter, impactParameterRange[0], impactParameterRange[1])};

  o2fw::OutputObj<TH2F> impactParameterErrorVsPt{
    TH2F("impactParameterErrorVsPt", ";p_{T};Impact Parameter Error", nBinsPtTrack,
         ptRange[0], ptRange[1], nBinsImpactParameter, impactParameterRange[0],
         impactParameterRange[1])};

  o2fw::OutputObj<TH2F> impactParameterErrorVsEta{
    TH2F("impactParameterErrorVsEta", ";#eta;Impact Parameter Error", nBinsEtaTrack, etaRange[0],
         etaRange[1], nBinsImpactParameter, impactParameterRange[0], impactParameterRange[1])};

  void process(
    o2::soa::Join<o2::aod::Collisions, o2::aod::McCollisionLabels>::iterator const& collision,
    o2::soa::Join<o2::aod::Tracks, o2::aod::TracksCov, o2::aod::McTrackLabels> const& tracks,
    o2::aod::McParticles const& mcParticles, o2::aod::McCollisions const& mcCollisions)
  {
    const o2::dataformats::VertexBase primaryVertex = getPrimaryVertex(collision);

    for (const auto& track : tracks) {
      double deltaPt = track.label().pt() - track.pt();
      ptDiffMCRec->Fill(deltaPt);

      double deltaPtOverPt = deltaPt / track.pt();
      ptResolution->Fill((deltaPtOverPt));
      ptResolutionVsPt->Fill(track.pt(), abs(deltaPtOverPt));
      ptResolutionVsEta->Fill(track.eta(), abs(deltaPtOverPt));

      double deltaEta = track.label().eta() - track.eta();
      etaDiffMCRec->Fill(deltaEta);

      auto deltaPhi = ConvertPhiRange(track.label().phi() - track.phi());
      phiDiffMCRec->Fill(deltaPhi);

      o2df::DCA dca;
      constexpr float magneticField{5.0}; // in kG (FIXME: get this from CCDB)
      auto trackParameter = getTrackParCov(track);
      trackParameter.propagateToDCA(primaryVertex, magneticField, &dca);

      double impact_parameter = 1000*dca.getY();
      double impact_parameter_error = 1000*std::sqrt(dca.getSigmaY2());

      impactParameterVsPt->Fill(track.pt(), impact_parameter);
      impactParameterVsEta->Fill(track.eta(), impact_parameter);
      impactParameterErrorVsPt->Fill(track.pt(), impact_parameter_error);
      impactParameterErrorVsEta->Fill(track.eta(), impact_parameter_error);
    }
  }
};

o2fw::WorkflowSpec defineDataProcessing(o2fw::ConfigContext const&)
{
  return o2fw::WorkflowSpec{
    o2fw::adaptAnalysisTask<QAGlobalObservables>("qa-global-observables"),
    o2fw::adaptAnalysisTask<QATrackingKine>("qa-tracking-kine"),
    o2fw::adaptAnalysisTask<QATrackingResolution>("qa-tracking-resolution")};
}
