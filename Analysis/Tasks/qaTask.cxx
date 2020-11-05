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
/// \author Nicolo' Jacazio <nicolo.jacazio@cern.ch>, CERN

#include "Framework/runDataProcessing.h"
#include "Framework/AnalysisTask.h"
#include "Framework/AnalysisDataModel.h"
#include "Framework/HistogramRegistry.h"
#include "Analysis/trackUtilities.h"
#include "ReconstructionDataFormats/DCA.h"
#include "Analysis/MC.h"

#include <TH1F.h>
#include <TH2F.h>
#include <cmath>

namespace o2fw = o2::framework;
namespace o2exp = o2::framework::expressions;
namespace o2df = o2::dataformats;

namespace track_utils
{
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

/// Determines the impact parameter and its error for a given track.
/// \param track the track to get the impact parameter from.
/// \param primaryVertex the primary vertex of th collision.
/// \param impactParameter variable to save the impact parameter in micrometers.
/// \param impactParameterError variable to save the impact parameter error in micrometers.
template <typename Track>
bool GetImpactParameterAndError(const Track& track, const o2df::VertexBase& primaryVertex, double& impactParameter,
                                double& impactParameterError)
{
  impactParameter = -999.;
  impactParameterError = -999.;

  o2df::DCA dca;
  // FIXME: get this from CCDB
  constexpr float magneticField{5.0}; // in kG
  auto trackParameter = getTrackParCov(track);
  bool propagate = trackParameter.propagateToDCA(primaryVertex, magneticField, &dca);

  if (propagate) {
    impactParameter = 1000 * dca.getY();
    impactParameterError = 1000 * std::sqrt(dca.getSigmaY2());
  }
  return propagate;
}
} // namespace track_utils

/// Task to QA global observables of the event
struct QAGlobalObservables {
  std::array<float, 2> collisionPositionRange = {-20., 20.};
  std::array<float, 2> numberOfTracksRange = {0, 2000};

  o2fw::Configurable<int> nBinsNumberOfTracks{"nBinsNumberOfTracks", 2000, "Number of bins fot the Number of Tracks"};

  o2fw::Configurable<int> nBinsVertexPosition{"nBinsPt", 100, "Number of bins for the Vertex Position"};

  o2fw::OutputObj<TH1F> hCollisionX{
    TH1F("collisionX", "; X [cm];Counts", nBinsVertexPosition, collisionPositionRange[0], collisionPositionRange[1])};

  o2fw::OutputObj<TH1F> hCollisionY{
    TH1F("collisionY", "; Y [cm];Counts", nBinsVertexPosition, collisionPositionRange[0], collisionPositionRange[1])};

  o2fw::OutputObj<TH1F> hCollisionZ{
    TH1F("collisionZ", "; Z [cm];Counts", nBinsVertexPosition, collisionPositionRange[0], collisionPositionRange[1])};

  o2fw::OutputObj<TH1F> hNumberOfTracks{TH1F("NumberOfTracks", "; Number of Tracks;Counts", nBinsNumberOfTracks,
                                             numberOfTracksRange[0], numberOfTracksRange[1])};

  void process(const o2::aod::Collision& collision, const o2::aod::Tracks& tracks)
  {
    hCollisionX->Fill(collision.posX());
    hCollisionY->Fill(collision.posY());
    hCollisionZ->Fill(collision.posZ());

    int nTracks(0);
    for (const auto& track : tracks) {
      nTracks++;
    }
    hNumberOfTracks->Fill(nTracks);
  }
};

/// Task to QA the kinematic properties of the tracks
struct QATrackingKine {
  o2fw::Configurable<int> nBinsPt{"nBinsPt", 100, "Number of bins for Pt"};
  std::array<double, 2> ptRange = {0, 10.};
  o2fw::Configurable<int> nBinsPhi{"nBinsPhi", 100, "Number of bins for Phi"};
  o2fw::Configurable<int> nBinsEta{"nBinsEta", 100, "Number of bins for the eta histogram."};
  std::array<double, 2> etaRange = {-6, 6};

  o2fw::OutputObj<TH1F> hPt{TH1F("pt", ";p_{T} [GeV];Counts", nBinsPt, ptRange[0], ptRange[1])};
  o2fw::OutputObj<TH1F> hEta{TH1F("eta", ";#eta;Counts", nBinsEta, etaRange[0], etaRange[1])};
  o2fw::OutputObj<TH1F> hPhi{TH1F("phi", ";#varphi [rad];Counts", nBinsPhi, 0, 2 * M_PI)};

  void process(const o2::aod::Track& track)
  {
    hEta->Fill(track.eta());
    hPhi->Fill(track.phi());
    hPt->Fill(track.pt());
  }
};

/// Task to evaluate the tracking resolution (Pt, Eta, Phi and impact parameter)
struct QATrackingResolution {
  o2fw::Configurable<int> nBinsPtTrack{"nBinsPtTrack", 100, "Number of bins for the transverse momentum"};
  std::array<double, 2> ptRange = {0, 10.};

  o2fw::Configurable<int> nBinsEta{"nBinsEta", 400, "Number of bins for the pseudorapidity"};
  std::array<double, 2> etaRange = {-6, 6};

  o2fw::Configurable<int> nBinsDeltaPt{"nBinsDeltaPt", 400, "Number of bins for the transverse momentum differences"};

  o2fw::Configurable<int> nBinsDeltaPhi{"nBinsPhi", 100, "Number of bins for the azimuthal angle differences"};

  o2fw::Configurable<int> nBinsDeltaEta{"nBinsDeltaEta", 100, "Number of bins for the pseudorapidity differences"};

  o2fw::Configurable<int> nBinsImpactParameter{"nBinsImpactParameter", 1000, "Number of bins for the Impact parameter"};

  std::array<double, 2> impactParameterRange = {-1500, 1500};       // micrometer
  std::array<double, 2> impactParameterResolutionRange = {0, 1000}; // micrometer

  // Registry of histograms
  o2fw::HistogramRegistry histos{"Histos", {}, o2fw::OutputObjHandlingPolicy::AnalysisObject};

  void init(o2fw::InitContext&)
  {
    // Eta
    histos.add("Eta/etaDiffMCReco", ";#eta_{MC} - #eta_{Rec}", o2fw::kTH1F,
               {{nBinsDeltaEta, -2, 2}});
    histos.add("Eta/etaDiffMCRecoVsEtaMC", ";#eta_{MC} - #eta_{Rec};#eta_{MC}", o2fw::kTH2F,
               {{nBinsDeltaEta, -2, 2}, {nBinsEta, etaRange[0], etaRange[1]}});
    histos.add("Eta/etaDiffMCRecoVsEtaReco", ";#eta_{MC} - #eta_{Rec};#eta_{Rec}", o2fw::kTH2F,
               {{nBinsDeltaEta, -2, 2}, {nBinsEta, etaRange[0], etaRange[1]}});
    // Phi
    histos.add("Phi/phiDiffMCRec", ";#varphi_{MC} - #varphi_{Rec} [rad]", o2fw::kTH1F,
               {{nBinsDeltaPhi, -M_PI, M_PI}});
    // Pt
    histos.add("Pt/ptDiffMCRec", ";p_{T}_{MC} - p_{T}_{Rec} [GeV/c]", o2fw::kTH1F,
               {{nBinsDeltaPt, -2., 2.}});
    histos.add("Pt/ptResolution", ";(p_{T}_{MC} - p_{T}_{Rec})/(p_{T}_{Rec})", o2fw::kTH1F,
               {{nBinsDeltaPt, -2., 2.}});
    histos.add("Pt/ptResolutionVsPt", ";p_{T} [GeV/c];(p_{T}_{MC} - p_{T}_{Rec})/(p_{T}_{Rec})", o2fw::kTH2F,
               {{nBinsPtTrack, ptRange[0], ptRange[1]}, {nBinsDeltaPt, 0, 2.}});
    histos.add("Pt/ptResolutionVsEta", ";#eta;(p_{T}_{MC} - p_{T}_{Rec})/(p_{T}_{Rec})", o2fw::kTH2F,
               {{nBinsEta, -4., 4.}, {nBinsDeltaPt, -2., 2.}});
    // Impact parameters
    histos.add("ImpactParameter/impactParameterVsPt", ";p_{T} [GeV/c];Impact Parameter [{#mu}m]", o2fw::kTH2F,
               {{nBinsPtTrack, ptRange[0], ptRange[1]}, {nBinsImpactParameter, impactParameterRange[0], impactParameterRange[1]}});
    histos.add("ImpactParameter/impactParameterVsEta", "#eta;Impact Parameter [{#mu}m]", o2fw::kTH2F,
               {{nBinsEta, etaRange[0], etaRange[1]}, {nBinsImpactParameter, impactParameterRange[0], impactParameterRange[1]}});
    histos.add("ImpactParameter/impactParameterErrorVsPt", ";p_{T} [GeV/c];Impact Parameter Error [#mum]", o2fw::kTH2F,
               {{nBinsPtTrack, ptRange[0], ptRange[1]}, {nBinsImpactParameter, impactParameterRange[0], impactParameterRange[1]}});
    histos.add("ImpactParameter/impactParameterErrorVsEta", ";#eta;Impact Parameter Error [#mum]", o2fw::kTH2F,
               {{nBinsEta, etaRange[0], etaRange[1]}, {nBinsImpactParameter, 0, impactParameterRange[1]}});
  }

  void process(const o2::soa::Join<o2::aod::Collisions, o2::aod::McCollisionLabels>::iterator& collision,
               const o2::soa::Join<o2::aod::Tracks, o2::aod::TracksCov, o2::aod::McTrackLabels>& tracks,
               const o2::aod::McParticles& mcParticles, const o2::aod::McCollisions& mcCollisions)
  {
    const o2df::VertexBase primaryVertex = getPrimaryVertex(collision);

    for (const auto& track : tracks) {
      const double deltaPt = track.label().pt() - track.pt();
      histos.fill("Pt/ptDiffMCRec", deltaPt);

      const double deltaPtOverPt = deltaPt / track.pt();
      histos.fill("Pt/ptResolution", deltaPtOverPt);
      histos.fill("Pt/ptResolutionVsPt", track.pt(), abs(deltaPtOverPt));
      histos.fill("Pt/ptResolutionVsEta", track.eta(), abs(deltaPtOverPt));

      const double deltaEta = track.label().eta() - track.eta();
      histos.fill("Eta/etaDiffMCReco", deltaEta);
      histos.fill("Eta/etaDiffMCRecoVsEtaMC", deltaEta, track.label().eta());
      histos.fill("Eta/etaDiffMCRecoVsEtaReco", deltaEta, track.eta());

      const auto deltaPhi = track_utils::ConvertPhiRange(track.label().phi() - track.phi());
      histos.fill("Phi/phiDiffMCRec", deltaPhi);

      double impactParameter{-999.};
      double impactParameterError{-999.};

      const bool propagate =
        track_utils::GetImpactParameterAndError(track, primaryVertex, impactParameter, impactParameterError);

      if (propagate) {
        histos.fill("ImpactParameter/impactParameterVsPt", track.pt(), impactParameter);
        histos.fill("ImpactParameter/impactParameterVsEta", track.eta(), impactParameter);
        histos.fill("ImpactParameter/impactParameterErrorVsPt", track.pt(), impactParameterError);
        histos.fill("ImpactParameter/impactParameterErrorVsEta", track.eta(), impactParameterError);
      }
    }
  }
};

o2fw::WorkflowSpec defineDataProcessing(o2fw::ConfigContext const&)
{
  return o2fw::WorkflowSpec{o2fw::adaptAnalysisTask<QAGlobalObservables>("qa-global-observables"),
                            o2fw::adaptAnalysisTask<QATrackingKine>("qa-tracking-kine"),
                            o2fw::adaptAnalysisTask<QATrackingResolution>("qa-tracking-resolution")};
}
