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

// O2 inlcudes
#include "Framework/runDataProcessing.h"
#include "Framework/AnalysisTask.h"
#include "Framework/AnalysisDataModel.h"
#include "Framework/HistogramRegistry.h"
#include "ReconstructionDataFormats/DCA.h"
#include "AnalysisCore/trackUtilities.h"
#include "AnalysisCore/MC.h"
#include "AnalysisDataModel/TrackSelectionTables.h"

namespace o2fw = o2::framework;
namespace o2exp = o2::framework::expressions;
namespace o2df = o2::dataformats;

/// Determines the impact parameter and its error for a given track.
/// \param track the track to get the impact parameter from.
/// \param primaryVertex the primary vertex of th collision.
/// \param impactParameterRPhi variable to save the impact parameter (in r phi) in micrometers.
/// \param impactParameterRPhiError variable to save the impact parameter (in r phi) error in micrometers.
/// \param impactParameterZ variable to save the impact parameter (in Z) in micrometers.
/// \param impactParameterZError variable to save the impact parameter (in Z) error in micrometers.
template <typename Track>
bool GetImpactParameterAndError(const Track& track, const o2df::VertexBase& primaryVertex, double& impactParameterRPhi,
                                double& impactParameterRPhiError, double& impactParameterZ,
                                double& impactParameterErrorZ)
{
  impactParameterRPhi = -999.;
  impactParameterRPhiError = -999.;
  impactParameterZ = -999;
  impactParameterErrorZ = -999;

  o2df::DCA dca;
  // FIXME: get this from CCDB
  constexpr float magneticField{5.0}; // in kG
  auto trackParameter = getTrackParCov(track);
  bool propagate = trackParameter.propagateToDCA(primaryVertex, magneticField, &dca);

  constexpr float conversion_to_micrometer = 1000;
  if (propagate) {
    impactParameterRPhi = conversion_to_micrometer * dca.getY();
    impactParameterRPhiError = conversion_to_micrometer * std::sqrt(dca.getSigmaY2());
    impactParameterZ = conversion_to_micrometer * dca.getZ();
    impactParameterErrorZ = conversion_to_micrometer * std::sqrt(dca.getSigmaZ2());
  }
  return propagate;
}

/// Task to QA global observables of the event
struct QAGlobalObservables {

  o2fw::Configurable<int> nBinsNumberOfTracks{"nBinsNumberOfTracks", 2000, "Number of bins for the Number of Tracks"};
  std::array<float, 2> numberOfTracksRange = {0, 2000};

  o2fw::Configurable<int> nBinsVertexPosition{"nBinsVertexPosition", 100, "Number of bins for the Vertex Position"};
  std::array<float, 2> collisionZRange = {-20., 20.};
  std::array<float, 2> collisionXYRange = {-0.01, 0.01};

  o2fw::Configurable<int> nBinsNumberOfContributorsVertex{"nBinsNumberOfContributorsVertex",
                                                          200, "Number bins for the number of contributors to the primary vertex"};
  o2fw::Configurable<int> numberOfContributorsVertexMax{"numberOfContributorsVertexMax",
                                                        200, "Maximum value for the Number of contributors to the primary vertex"};

  o2fw::Configurable<int> nBinsVertexCovarianceMatrix{"nBinsVertexCovarianceMatrix", 100,
                                                      "Number bins for the vertex covariance matrix"};
  std::array<float, 2> vertexCovarianceMatrixRange = {-0.1, 0.1};

  o2fw::HistogramRegistry histograms{"HistogramsGlobalQA"};
  void init(o2fw::InitContext&)
  {

    o2fw::AxisSpec collisionXYAxis{nBinsVertexPosition, collisionXYRange[0], collisionXYRange[1]};
    o2fw::AxisSpec collisionZAxis{nBinsVertexPosition, collisionZRange[0], collisionZRange[1]};
    o2fw::AxisSpec numberOfContributorsAxis{nBinsNumberOfContributorsVertex, 0, float(numberOfContributorsVertexMax)};
    o2fw::AxisSpec vertexCovarianceMatrixAxis{nBinsVertexCovarianceMatrix,
                                              vertexCovarianceMatrixRange[0], vertexCovarianceMatrixRange[1]};
    o2fw::AxisSpec numberOfTrackAxis{nBinsNumberOfTracks, numberOfTracksRange[0], numberOfTracksRange[1]};

    // Global
    histograms.add("eventCount", ";Selected Events", o2fw::kTH1D, {{2, 0, 2}});

    // Collision
    histograms.add("collision/X", ";X [cm]", o2fw::kTH1D, {collisionXYAxis});
    histograms.add("collision/Y", ";Y [cm]", o2fw::kTH1D, {collisionXYAxis});
    histograms.add("collision/Z", ";Z [cm]", o2fw::kTH1D, {collisionZAxis});
    histograms.add("collision/numberOfContributors", ";Number Of contributors to the PV.", o2fw::kTH1D, {numberOfContributorsAxis});
    histograms.add("collision/vertexChi2", ";#Chi^{2}", o2fw::kTH1D, {{100, 0, 10}});

    // Covariance
    histograms.add("covariance/xx", ";Cov_{xx} [cm^{2}]", o2fw::kTH1D, {vertexCovarianceMatrixAxis});
    histograms.add("covariance/xy", ";Cov_{xy} [cm^{2}]", o2fw::kTH1D, {vertexCovarianceMatrixAxis});
    histograms.add("covariance/xz", ";Cov_{xz} [cm^{2}]", o2fw::kTH1D, {vertexCovarianceMatrixAxis});
    histograms.add("covariance/yy", ";Cov_{yy} [cm^{2}]", o2fw::kTH1D, {vertexCovarianceMatrixAxis});
    histograms.add("covariance/yz", ";Cov_{yz} [cm^{2}]", o2fw::kTH1D, {vertexCovarianceMatrixAxis});
    histograms.add("covariance/zz", ";Cov_{zz} [cm^{2}]", o2fw::kTH1D, {vertexCovarianceMatrixAxis});
    // Multiplicity
    histograms.add("multiplicity/numberOfTracks", ";Track Multiplicity", o2fw::kTH1D, {numberOfTrackAxis});
  }

  void process(const o2::aod::Collision& collision, const o2::aod::Tracks& tracks)
  {
    histograms.fill(HIST("eventCount"), 0);

    histograms.fill(HIST("collision/X"), collision.posX());
    histograms.fill(HIST("collision/Y"), collision.posY());
    histograms.fill(HIST("collision/Z"), collision.posZ());

    histograms.fill(HIST("collision/numberOfContributors"), collision.numContrib());
    histograms.fill(HIST("collision/vertexChi2"), collision.chi2());

    histograms.fill(HIST("covariance/xx"), collision.covXX());
    histograms.fill(HIST("covariance/xy"), collision.covXY());
    histograms.fill(HIST("covariance/xz"), collision.covXZ());
    histograms.fill(HIST("covariance/yy"), collision.covYY());
    histograms.fill(HIST("covariance/yz"), collision.covYZ());
    histograms.fill(HIST("covariance/zz"), collision.covZZ());

    int nTracks(0);
    for (const auto& track : tracks) {
      nTracks++;
    }

    histograms.fill(HIST("multiplicity/numberOfTracks"), nTracks);
  }
};

/// Task to QA the kinematic properties of the tracks
struct QATrackingKine {
  o2fw::Configurable<int> nBinsPt{"nBinsPt", 100, "Number of bins for Pt"};
  std::array<double, 2> ptRange = {0, 10.};
  o2fw::Configurable<int> nBinsEta{"nBinsEta", 100, "Number of bins for the eta histogram."};
  std::array<double, 2> etaRange = {-6, 6};
  o2fw::Configurable<int> nBinsPhi{"nBinsPhi", 100, "Number of bins for Phi"};

  o2fw::HistogramRegistry histos{"HistogramsKineQA"};
  void init(o2fw::InitContext&)
  {
    histos.add("tracking/pt", ";#it{p}_{T} [GeV]", o2fw::kTH1D, {{nBinsPt, ptRange[0], ptRange[1]}});
    histos.add("tracking/eta", ";#eta", o2fw::kTH1D, {{nBinsEta, etaRange[0], etaRange[1]}});
    histos.add("tracking/phi", ";#varphi [rad]", o2fw::kTH1D, {{nBinsPhi, 0, 2 * M_PI}});
  }

  void process(const o2::aod::Track& track)
  {
    histos.fill(HIST("tracking/eta"), track.eta());
    histos.fill(HIST("tracking/pt"), track.pt());
    histos.fill(HIST("tracking/phi"), track.phi());
  }
};

/// Task to evaluate the tracking resolution (Pt, Eta, Phi and impact parameter)
struct QATrackingResolution {

  o2fw::Configurable<bool> useOnlyPhysicsPrimary{"useOnlyPhysicsPrimary", true,
                                                 "Whether to use only physical primary particles for the resolution."};

  o2fw::Configurable<int> nBinsPt{"nBinsPt", 100, "Number of bins for the transverse momentum"};
  std::array<double, 2> ptRange = {0, 10.};

  o2fw::Configurable<int> nBinsEta{"nBinsEta", 60, "Number of bins for the pseudorapidity"};
  std::array<double, 2> etaRange = {-3, 3};

  o2fw::Configurable<int> nBinsPhi{"nBinsPhi", 50, "Number of bins for Phi"};
  std::array<double, 2> phiRange = {0, 2 * M_PI};

  o2fw::Configurable<int> nBinsDeltaPt{"nBinsDeltaPt", 100, "Number of bins for the transverse momentum differences"};
  std::array<double, 2> deltaPtRange = {-0.5, 0.5};

  o2fw::Configurable<int> nBinsDeltaPhi{"nBinsDeltaPhi", 100, "Number of bins for the azimuthal angle differences"};
  std::array<double, 2> deltaPhiRange = {-0.1, 0.1};

  o2fw::Configurable<int> nBinsDeltaEta{"nBinsDeltaEta", 100, "Number of bins for the pseudorapidity differences"};
  std::array<double, 2> deltaEtaRange = {-0.1, 0.1};

  o2fw::Configurable<int> nBinsImpactParameter{"nBinsImpactParameter", 2000, "Number of bins for the Impact parameter"};

  std::array<double, 2> impactParameterRange = {-500, 500};         // micrometer
  std::array<double, 2> impactParameterResolutionRange = {0, 1000}; // micrometer

  o2fw::HistogramRegistry histos{"HistogramsTrackingResolutionQA"};
  void init(o2fw::InitContext&)
  {
    // Histogram axis definitions

    o2fw::AxisSpec ptAxis{nBinsPt, ptRange[0], ptRange[1]};
    o2fw::AxisSpec deltaPtAxis{nBinsDeltaPt, deltaPtRange[0], deltaPtRange[1]};
    o2fw::AxisSpec deltaPtRelativeAxis{nBinsDeltaPt, deltaPtRange[0], deltaPtRange[1]};
    o2fw::AxisSpec deltaPtAbsoluteRelativeAxis{nBinsDeltaPt, 0., deltaPtRange[1]};

    o2fw::AxisSpec etaAxis{nBinsEta, etaRange[0], etaRange[1]};
    o2fw::AxisSpec deltaEtaAxis{nBinsDeltaEta, deltaEtaRange[0], deltaEtaRange[1]};

    o2fw::AxisSpec phiAxis{nBinsPhi, phiRange[0], phiRange[1]};
    o2fw::AxisSpec deltaPhiAxis{nBinsDeltaPhi, deltaPhiRange[0], deltaPhiRange[1]};

    o2fw::AxisSpec impactParRPhiAxis{nBinsImpactParameter, impactParameterRange[0], impactParameterRange[1]};
    o2fw::AxisSpec impactParRPhiErrorAxis{nBinsImpactParameter, impactParameterResolutionRange[0],
                                          impactParameterResolutionRange[1]};

    o2fw::AxisSpec impactParZAxis{nBinsImpactParameter, impactParameterRange[0], impactParameterRange[1]};
    o2fw::AxisSpec impactParZErrorAxis{nBinsImpactParameter, impactParameterResolutionRange[0],
                                       impactParameterResolutionRange[1]};

    // Eta
    histos.add("eta/etaDiffMCReco", ";#eta_{MC} - #eta_{Rec}", o2fw::kTH1D, {deltaEtaAxis});
    histos.add("eta/etaDiffMCRecoVsEtaMC", ";#eta_{MC} - #eta_{Rec};#eta_{MC}", o2fw::kTH2D, {deltaEtaAxis, etaAxis});
    histos.add("eta/etaDiffMCRecoVsEtaReco", ";#eta_{MC} - #eta_{Rec};#eta_{Rec}", o2fw::kTH2D, {deltaEtaAxis, etaAxis});

    // Phi
    histos.add("phi/phiDiffMCRec", ";#varphi_{MC} - #varphi_{Rec} [rad]", o2fw::kTH1D, {deltaPhiAxis});

    // Pt
    histos.add("pt/ptDiffMCRec", ";p_{T}_{MC} - p_{T}_{Rec} [GeV/c]", o2fw::kTH1D, {deltaPtAxis});
    histos.add("pt/ptResolution", ";(p_{T}_{MC} - p_{T}_{Rec})/(p_{T}_{Rec})", o2fw::kTH1D, {deltaPtRelativeAxis});
    histos.add("pt/ptResolutionVsPt", ";p_{T} [GeV/c];(p_{T}_{MC} - p_{T}_{Rec})/(p_{T}_{Rec})", o2fw::kTH2D, {ptAxis, deltaPtAbsoluteRelativeAxis});
    histos.add("pt/ptResolutionVsEta", ";#eta;(p_{T}_{MC} - p_{T}_{Rec})/(p_{T}_{Rec})", o2fw::kTH2D, {etaAxis, deltaPtAbsoluteRelativeAxis});
    histos.add("pt/ptResolutionVsPhi", ";#varphi;(p_{T}_{MC} - p_{T}_{Rec})/(p_{T}_{Rec})", o2fw::kTH2D, {phiAxis, deltaPtAbsoluteRelativeAxis});

    // Impact parameters
    const TString imp_rphi = "Impact Parameter r#varphi [{#mu}m]";
    const TString imp_rphi_err = "Impact Parameter Error r#varphi [{#mu}m]";
    const TString pt = "#it{p}_{T} [GeV/c]";
    const TString pt_rec = "#it{p}_{T}_{Rec} [GeV/c]";
    const TString eta_rec = "#eta_{Rec}";
    const TString phi_rec = "#varphi_{Rec} [rad]";

    histos.add("impactParameter/impactParameterRPhiVsPt", ";" + pt_rec + ";" + imp_rphi, o2fw::kTH2D, {ptAxis, impactParRPhiAxis});
    histos.add("impactParameter/impactParameterRPhiVsEta", ";" + eta_rec + ";" + imp_rphi, o2fw::kTH2D, {etaAxis, impactParRPhiAxis});
    histos.add("impactParameter/impactParameterRPhiVsPhi", ";" + phi_rec + ";" + imp_rphi, o2fw::kTH2D, {phiAxis, impactParRPhiAxis});

    histos.add("impactParameter/impactParameterErrorRPhiVsPt", ";" + pt_rec + ";" + imp_rphi_err, o2fw::kTH2D, {ptAxis, impactParRPhiErrorAxis});
    histos.add("impactParameter/impactParameterErrorRPhiVsEta", ";" + eta_rec + ";" + imp_rphi_err, o2fw::kTH2D, {etaAxis, impactParRPhiErrorAxis});
    histos.add("impactParameter/impactParameterErrorRPhiVsPhi", ";" + phi_rec + ";" + imp_rphi_err, o2fw::kTH2D, {phiAxis, impactParRPhiErrorAxis});

    const TString imp_z = "Impact Parameter Z [#mum]";
    const TString imp_z_err = "Impact Parameter Error Z [#mum]";

    histos.add("impactParameter/impactParameterZVsPt", ";" + pt_rec + ";" + imp_z, o2fw::kTH2D, {ptAxis, impactParZAxis});
    histos.add("impactParameter/impactParameterZVsEta", ";" + eta_rec + ";" + imp_z, o2fw::kTH2D, {etaAxis, impactParZAxis});
    histos.add("impactParameter/impactParameterZVsPhi", ";" + phi_rec + ";" + imp_z, o2fw::kTH2D, {phiAxis, impactParZAxis});

    histos.add("impactParameter/impactParameterErrorZVsPt", ";" + pt_rec + ";" + imp_z_err, o2fw::kTH2D, {ptAxis, impactParZErrorAxis});
    histos.add("impactParameter/impactParameterErrorZVsEta", ";" + eta_rec + ";" + imp_z_err, o2fw::kTH2D, {etaAxis, impactParZErrorAxis});
    histos.add("impactParameter/impactParameterErrorZVsPhi", ";" + phi_rec + ";" + imp_z_err, o2fw::kTH2D, {phiAxis, impactParZErrorAxis});
  }

  void process(const o2::soa::Join<o2::aod::Collisions, o2::aod::McCollisionLabels>::iterator& collision,
               const o2::soa::Join<o2::aod::Tracks, o2::aod::TracksCov, o2::aod::McTrackLabels>& tracks,
               const o2::aod::McParticles& mcParticles, const o2::aod::McCollisions& mcCollisions)
  {
    const o2df::VertexBase primaryVertex = getPrimaryVertex(collision);

    for (const auto& track : tracks) {

      if (useOnlyPhysicsPrimary) {
        const auto mcParticle = track.label();
        if (!MC::isPhysicalPrimary(mcParticles, mcParticle)) {
          continue;
        }
      }
      const double deltaPt = track.label().pt() - track.pt();
      histos.fill(HIST("pt/ptDiffMCRec"), deltaPt);

      const double deltaPtOverPt = deltaPt / track.pt();

      histos.fill(HIST("pt/ptResolution"), deltaPtOverPt);
      histos.fill(HIST("pt/ptResolutionVsPt"), track.pt(), std::abs(deltaPtOverPt));
      histos.fill(HIST("pt/ptResolutionVsEta"), track.eta(), std::abs(deltaPtOverPt));
      histos.fill(HIST("pt/ptResolutionVsPhi"), track.phi(), std::abs(deltaPtOverPt));

      const double deltaEta = track.label().eta() - track.eta();
      histos.fill(HIST("eta/etaDiffMCReco"), deltaEta);
      histos.fill(HIST("eta/etaDiffMCRecoVsEtaMC"), deltaEta, track.label().eta());
      histos.fill(HIST("eta/etaDiffMCRecoVsEtaReco"), deltaEta, track.eta());

      histos.fill(HIST("phi/phiDiffMCRec"), track.label().phi() - track.phi());

      double impactParameterRPhi{-999.}, impactParameterRPhiError{-999.};
      double impactParameterZ{-999.}, impactParameterErrorZ{-999.};

      const bool propagate = GetImpactParameterAndError(
        track, primaryVertex, impactParameterRPhi, impactParameterRPhiError, impactParameterZ, impactParameterErrorZ);

      if (propagate) {
        histos.fill(HIST("impactParameter/impactParameterRPhiVsPt"), track.pt(), impactParameterRPhi);
        histos.fill(HIST("impactParameter/impactParameterRPhiVsEta"), track.eta(), impactParameterRPhi);
        histos.fill(HIST("impactParameter/impactParameterRPhiVsPhi"), track.phi(), impactParameterRPhi);

        histos.fill(HIST("impactParameter/impactParameterZVsPt"), track.pt(), impactParameterZ);
        histos.fill(HIST("impactParameter/impactParameterZVsEta"), track.eta(), impactParameterZ);
        histos.fill(HIST("impactParameter/impactParameterZVsPhi"), track.phi(), impactParameterZ);

        histos.fill(HIST("impactParameter/impactParameterErrorRPhiVsPt"), track.pt(), impactParameterRPhiError);
        histos.fill(HIST("impactParameter/impactParameterErrorRPhiVsEta"), track.eta(), impactParameterRPhiError);
        histos.fill(HIST("impactParameter/impactParameterErrorRPhiVsPhi"), track.phi(), impactParameterRPhiError);

        histos.fill(HIST("impactParameter/impactParameterErrorZVsPt"), track.pt(), impactParameterErrorZ);
        histos.fill(HIST("impactParameter/impactParameterErrorZVsEta"), track.eta(), impactParameterErrorZ);
        histos.fill(HIST("impactParameter/impactParameterErrorZVsPhi"), track.phi(), impactParameterErrorZ);
      }
    }
  }
};

o2fw::WorkflowSpec defineDataProcessing(o2fw::ConfigContext const&)
{
  o2fw::WorkflowSpec w;
  w.push_back(o2fw::adaptAnalysisTask<QAGlobalObservables>("qa-global-observables"));
  w.push_back(o2fw::adaptAnalysisTask<QATrackingKine>("qa-tracking-kine"));
  w.push_back(o2fw::adaptAnalysisTask<QATrackingResolution>("qa-tracking-resolution"));
  return w;
}
