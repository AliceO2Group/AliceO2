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
#include "AnalysisCore/trackUtilities.h"
#include "ReconstructionDataFormats/DCA.h"
#include "AnalysisDataModel/TrackSelectionTables.h"
#include "AnalysisCore/MC.h"
#include "TPDGCode.h"

#include "TH1D.h"

#include <cmath>
#include <string>
#include <algorithm>
#include "boost/algorithm/string.hpp"

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

} // namespace track_utils

namespace o2::qa::features
{

/// Class to abstract the naming of a particular feature. It can help you to build the histogram
/// labels in a consistent way and generate the titles.
class Feature
{
 public:
  Feature() = default;
  Feature(std::string name, std::string unit = "") : mName(std::move(name)), mUnit(std::move(unit)){};

  /// Returns the feature tagged as MC with no unit. Example: p_{T}^{MC}.
  std::string MCRaw() const { return mName + "^{MC}"; };

  /// Returns the feature tagged as Reconstructed with no unit. Example: p_{T}^{Rec}.
  std::string RecRaw() const { return mName + "^{Rec}"; };

  /// Returns the unit with no brackets. Example: GeV/c.
  std::string UnitRaw() const { return mUnit; };

  /// Returns the name with no units. Example: p_{T}.
  std::string NameRaw() const { return mName; };

  /// Returns the name with the unit. Example: p_{T} [Gev/c].
  std::string Name() const { return mName + Unit(); };

  /// Returns the name tagged as MC with the unit. Example: p_{T}^{MC} [Gev/c].
  std::string MC() const { return MCRaw() + Unit(); };

  /// Returns the name tagged as Reconstructed with the unit. Example: p_{T}^{Rec} [Gev/c]
  std::string Rec() const { return RecRaw() + Unit(); };

  /// Returns the name difference between the MC and reconstructed with the unit.
  /// Example: p_{T}^{MC} - p_{T}^{Rec} [Gev/c]
  std::string MCRecDiff() const { return MCRaw() + " - " + RecRaw() + Unit(); };

  /// Returns the name difference between the MC and reconstructed divided by the reconstructed.
  /// For example: (p_{T}^{MC} - p_{T}^{Rec})/p_{T}^{Rec}
  std::string RelativeMCRecDiff() const { return "(" + MCRaw() + " - " + RecRaw() + ")/" + RecRaw(); };

  /// Returns the unit formatted to be used in the axis. If no unit is present, returns an empty value.
  /// Example: [GeV/c]
  std::string Unit() const
  {
    if (UnitRaw().empty()) {
      return "";
    }
    return " [" + UnitRaw() + "]";
  };

  operator std::string() { return Name(); }

 private:
  const std::string mName;
  const std::string mUnit;
};

/// Makes a title for an histogram
std::string MakeTitle(std::vector<std::string> axisTitles, const std::string& counts = "Counts")
{
  axisTitles.push_back(counts);
  return "; " + boost::algorithm::join(axisTitles, "; ");
}

Feature Eta("#eta");
Feature TrackMultiplicity("Track Multiplicity");
Feature Phi{"#varphi", "rad"};
Feature Pt("p_{T}", "GeV/c");
Feature VertexX("X", "cm");
Feature VertexY("Y", "cm");
Feature VertexZ("Z", "cm");
Feature ImpactParameterRPhi("Impact Parameter r#varphi", "#mum");
Feature ImpactParameterRPhiError("Impact Parameter Error r#varphi", "#mum");
Feature ImpactParameterZ("Impact Parameter Z", "#mum");
Feature ImpactParameterZError("Impact Parameter Z Error", "#mum");
Feature NumberOfContributors("Number Of contributors to the PV.");
Feature CovarianceXX("Cov_{xx}", "cm^{2}");
Feature CovarianceXY("Cov_{xy}", "cm^{2}");
Feature CovarianceXZ("Cov_{xz}", "cm^{2}");
Feature CovarianceYY("Cov_{yy}", "cm^{2}");
Feature CovarianceYZ("Cov_{yz}", "cm^{2}");
Feature CovarianceZZ("Cov_{zz}", "cm^{2}");
Feature VertexChi2("#Chi^{2}");

std::vector<double> PtBins = {
  0.01, 0.0101, 0.0102, 0.0103, 0.0104, 0.0105, 0.0106, 0.0107, 0.0108, 0.0109, 0.011,
  0.0111, 0.0112, 0.0113, 0.0114, 0.0115, 0.0116, 0.0117, 0.0118, 0.0119, 0.012,
  0.0121, 0.0122, 0.0123, 0.0124, 0.0125, 0.0126, 0.0127, 0.0128, 0.0129, 0.013,
  0.0131, 0.0132, 0.0133, 0.0134, 0.0135, 0.0136, 0.0137, 0.0138, 0.0139, 0.014,
  0.0141, 0.0142, 0.0143, 0.0144, 0.0145, 0.0146, 0.0147, 0.0148, 0.0149, 0.015,
  0.0151, 0.0152, 0.0153, 0.0154, 0.0155, 0.0156, 0.0157, 0.0158, 0.0159, 0.016,
  0.0161, 0.0162, 0.0163, 0.0164, 0.0165, 0.0166, 0.0167, 0.0168, 0.0169, 0.017,
  0.0171, 0.0172, 0.0173, 0.0174, 0.0175, 0.0176, 0.0177, 0.0178, 0.0179, 0.018,
  0.0181, 0.0182, 0.0183, 0.0184, 0.0185, 0.0186, 0.0187, 0.0188, 0.0189, 0.019,
  0.0191, 0.0192, 0.0193, 0.0194, 0.0195, 0.0196, 0.0197, 0.0198, 0.0199, 0.02,
  0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09, 0.1, 0.12, 0.14, 0.155, 0.16, 0.165,
  0.175, 0.18, 0.185, 0.2, 0.225, 0.25, 0.275, 0.3, 0.35, 0.4, 0.45, 0.5, 0.6,
  0.7, 0.8, 0.9, 1.0, 1.2, 1.4, 1.6, 1.8, 2.0, 2.5, 3.0, 3.5, 4.0, 5.0, 6.0,
  8.0, 10.0, 15.0, 20.0, 30.0, 50.0, 100.0};

} // namespace o2::qa::features

namespace qafeat = o2::qa::features;

/// Task to QA global observables of the event
struct QAGlobalObservables {
  o2fw::Configurable<int> nBinsNumberOfTracks{"nBinsNumberOfTracks", 2000, "Number of bins for the Number of Tracks"};

  o2fw::Configurable<int> nBinsVertexPosition{"nBinsVertexPosition", 100, "Number of bins for the Vertex Position"};

  o2fw::Configurable<int> nBinsNumberOfContributorsVertex{
    "nBinsNumberOfContributorsVertex", 200, "Number bins for the number of contributors to the primary vertex"};

  o2fw::Configurable<int> numberOfContributorsVertexMax{
    "numberOfContributorsVertexMax", 200, "Maximum value for the Number of contributors to the primary vertex"};

  o2fw::Configurable<int> nBinsVertexCovarianceMatrix{"nBinsVertexCovarianceMatrix", 100,
                                                      "Number bins for the vertex covariance matrix"};

  std::array<float, 2> collisionZRange = {-20., 20.};
  std::array<float, 2> collisionXYRange = {-0.01, 0.01};
  std::array<float, 2> numberOfTracksRange = {0, 400};
  std::array<float, 2> vertexCovarianceMatrixRange = {-0.1, 0.1};

  o2fw::OutputObj<TH1D> eventCount{TH1D("eventCount", qafeat::MakeTitle({"Selected Events"}).c_str(), 2, 0, 2)};
  o2fw::HistogramRegistry histograms{"HistogramsGlobalQA"};

  void init(o2fw::InitContext&)
  {
    o2fw::AxisSpec collisionXAxis{nBinsVertexPosition, collisionXYRange[0], collisionXYRange[1]};
    o2fw::AxisSpec collisionYAxis{nBinsVertexPosition, collisionXYRange[0], collisionXYRange[1]};
    o2fw::AxisSpec collisionZAxis{nBinsVertexPosition, collisionZRange[0], collisionZRange[1]};

    o2fw::AxisSpec numberOfContributorsAxis{nBinsNumberOfContributorsVertex, 0, float(numberOfContributorsVertexMax)};
    o2fw::AxisSpec numberOfTrackAxis{nBinsNumberOfTracks, numberOfTracksRange[0], numberOfTracksRange[1]};

    o2fw::AxisSpec vertexCovarianceMatrixAxis{nBinsVertexCovarianceMatrix, vertexCovarianceMatrixRange[0],
                                              vertexCovarianceMatrixRange[1]};

    histograms.add("collision/collisionX", qafeat::MakeTitle({qafeat::VertexX}).c_str(), o2fw::kTH1D, {collisionXAxis});

    histograms.add("collision/collisionY", qafeat::MakeTitle({qafeat::VertexY}).c_str(), o2fw::kTH1D, {collisionYAxis});

    histograms.add("collision/collisionZ", qafeat::MakeTitle({qafeat::VertexZ}).c_str(), o2fw::kTH1D, {collisionZAxis});

    histograms.add("collision/numberOfContributors", qafeat::MakeTitle({qafeat::NumberOfContributors}).c_str(),
                   o2fw::kTH1D, {numberOfContributorsAxis});

    histograms.add("collision/vertexChi2", qafeat::MakeTitle({qafeat::VertexChi2}).c_str(),
                   o2fw::kTH1D, {{100, 0, 10}});

    histograms.add("multiplicity/numberOfTracks", qafeat::MakeTitle({qafeat::TrackMultiplicity}).c_str(), o2fw::kTH1D,
                   {numberOfTrackAxis});

    histograms.add("covariance/xx", qafeat::MakeTitle({qafeat::CovarianceXX}).c_str(), o2fw::kTH1D,
                   {vertexCovarianceMatrixAxis});
    histograms.add("covariance/xy", qafeat::MakeTitle({qafeat::CovarianceXY}).c_str(), o2fw::kTH1D,
                   {vertexCovarianceMatrixAxis});
    histograms.add("covariance/xz", qafeat::MakeTitle({qafeat::CovarianceXZ}).c_str(), o2fw::kTH1D,
                   {vertexCovarianceMatrixAxis});
    histograms.add("covariance/yy", qafeat::MakeTitle({qafeat::CovarianceYY}).c_str(), o2fw::kTH1D,
                   {vertexCovarianceMatrixAxis});
    histograms.add("covariance/yz", qafeat::MakeTitle({qafeat::CovarianceYZ}).c_str(), o2fw::kTH1D,
                   {vertexCovarianceMatrixAxis});
    histograms.add("covariance/zz", qafeat::MakeTitle({qafeat::CovarianceZZ}).c_str(), o2fw::kTH1D,
                   {vertexCovarianceMatrixAxis});
  }

  void process(const o2::aod::Collision& collision, const o2::aod::Tracks& tracks)
  {
    eventCount->Fill(0);
    histograms.fill(HIST("collision/collisionX"), collision.posX());
    histograms.fill(HIST("collision/collisionY"), collision.posY());
    histograms.fill(HIST("collision/collisionZ"), collision.posZ());

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
  o2fw::Configurable<int> nBinsPhi{"nBinsPhi", 100, "Number of bins for Phi"};

  o2fw::Configurable<int> nBinsEta{"nBinsEta", 100, "Number of bins for the eta histogram."};
  std::array<double, 2> etaRange = {-6, 6};

  o2fw::HistogramRegistry histos{"HistogramsKineQA"};

  void init(o2fw::InitContext&)
  {
    histos.add("tracking/pt", qafeat::MakeTitle({qafeat::Pt}).c_str(), o2fw::kTH1D,
               {qafeat::PtBins});
    histos.add("tracking/eta", qafeat::MakeTitle({qafeat::Eta.NameRaw()}).c_str(), o2fw::kTH1D,
               {{nBinsEta, etaRange[0], etaRange[1]}});
    histos.add("tracking/phi", qafeat::MakeTitle({qafeat::Phi}).c_str(), o2fw::kTH1D, {{nBinsPhi, 0, 2 * M_PI}});
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

  // Registry of histograms
  o2fw::HistogramRegistry histos{"HistogramsTrackingResolutionQA"};

  void init(o2fw::InitContext&)
  {
    // Histogram axis definitions

    o2fw::AxisSpec ptAxis{qafeat::PtBins};
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
    histos.add("eta/etaDiffMCReco", qafeat::MakeTitle({qafeat::Eta.MCRecDiff()}).c_str(), o2fw::kTH1D, {deltaEtaAxis});

    histos.add("eta/etaDiffMCRecoVsEtaMC", qafeat::MakeTitle({qafeat::Eta.MCRecDiff(), qafeat::Eta.MC()}).c_str(),
               o2fw::kTH2D, {deltaEtaAxis, etaAxis});

    histos.add("eta/etaDiffMCRecoVsEtaReco", qafeat::MakeTitle({qafeat::Eta.MCRecDiff(), qafeat::Eta.Rec()}).c_str(),
               o2fw::kTH2D, {deltaEtaAxis, etaAxis});

    // Phi
    histos.add("phi/phiDiffMCRec", qafeat::MakeTitle({qafeat::Phi.MCRecDiff()}).c_str(), o2fw::kTH1D, {deltaPhiAxis});

    // Pt
    histos.add("pt/ptDiffMCRec", qafeat::MakeTitle({qafeat::Pt.MCRecDiff()}).c_str(), o2fw::kTH1D, {deltaPtAxis});

    histos.add("pt/ptResolution", qafeat::MakeTitle({qafeat::Pt.RelativeMCRecDiff()}).c_str(), o2fw::kTH1D,
               {deltaPtRelativeAxis});

    histos.add("pt/ptResolutionVsPt", qafeat::MakeTitle({qafeat::Pt.Rec(), qafeat::Pt.RelativeMCRecDiff()}).c_str(),
               o2fw::kTH2D, {ptAxis, deltaPtAbsoluteRelativeAxis});

    histos.add("pt/ptResolutionVsEta", qafeat::MakeTitle({qafeat::Eta.Rec(), qafeat::Pt.RelativeMCRecDiff()}).c_str(),
               o2fw::kTH2D, {etaAxis, deltaPtAbsoluteRelativeAxis});

    histos.add("pt/ptResolutionVsPhi", qafeat::MakeTitle({qafeat::Phi.Rec(), qafeat::Pt.RelativeMCRecDiff()}).c_str(),
               o2fw::kTH2D, {phiAxis, deltaPtAbsoluteRelativeAxis});

    // Impact parameters
    histos.add("impactParameter/impactParameterRPhiVsPt",
               qafeat::MakeTitle({qafeat::Pt.Rec(), qafeat::ImpactParameterRPhi}).c_str(), o2fw::kTH2D,
               {ptAxis, impactParRPhiAxis});

    histos.add("impactParameter/impactParameterRPhiVsEta",
               qafeat::MakeTitle({qafeat::Eta.Rec(), qafeat::ImpactParameterRPhi}).c_str(), o2fw::kTH2D,
               {etaAxis, impactParRPhiAxis});

    histos.add("impactParameter/impactParameterRPhiVsPhi",
               qafeat::MakeTitle({qafeat::Phi.Rec(), qafeat::ImpactParameterRPhi}).c_str(), o2fw::kTH2D,
               {phiAxis, impactParRPhiAxis});

    histos.add("impactParameter/impactParameterErrorRPhiVsPt",
               qafeat::MakeTitle({qafeat::Pt.Rec(), qafeat::ImpactParameterRPhiError}).c_str(), o2fw::kTH2D,
               {ptAxis, impactParRPhiErrorAxis});

    histos.add("impactParameter/impactParameterErrorRPhiVsEta",
               qafeat::MakeTitle({qafeat::Eta.Rec(), qafeat::ImpactParameterRPhiError}).c_str(), o2fw::kTH2D,
               {etaAxis, impactParRPhiErrorAxis});

    histos.add("impactParameter/impactParameterErrorRPhiVsPhi",
               qafeat::MakeTitle({qafeat::Phi.Rec(), qafeat::ImpactParameterRPhiError}).c_str(), o2fw::kTH2D,
               {phiAxis, impactParRPhiErrorAxis});

    histos.add("impactParameter/impactParameterZVsPt",
               qafeat::MakeTitle({qafeat::Pt.Rec(), qafeat::ImpactParameterZ}).c_str(), o2fw::kTH2D,
               {ptAxis, impactParZAxis});

    histos.add("impactParameter/impactParameterZVsEta",
               qafeat::MakeTitle({qafeat::Eta.Rec(), qafeat::ImpactParameterZ}).c_str(), o2fw::kTH2D,
               {etaAxis, impactParZAxis});

    histos.add("impactParameter/impactParameterZVsPhi",
               qafeat::MakeTitle({qafeat::Phi.Rec(), qafeat::ImpactParameterZ}).c_str(), o2fw::kTH2D,
               {phiAxis, impactParZAxis});

    histos.add("impactParameter/impactParameterErrorZVsPt",
               qafeat::MakeTitle({qafeat::Pt.Rec(), qafeat::ImpactParameterZError}).c_str(), o2fw::kTH2D,
               {ptAxis, impactParZErrorAxis});

    histos.add("impactParameter/impactParameterErrorZVsEta",
               qafeat::MakeTitle({qafeat::Eta.Rec(), qafeat::ImpactParameterZError}).c_str(), o2fw::kTH2D,
               {etaAxis, impactParZErrorAxis});

    histos.add("impactParameter/impactParameterErrorZVsPhi",
               qafeat::MakeTitle({qafeat::Phi.Rec(), qafeat::ImpactParameterZError}).c_str(), o2fw::kTH2D,
               {phiAxis, impactParZErrorAxis});
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

      const double deltaPhi = track_utils::ConvertPhiRange(track.label().phi() - track.phi());
      histos.fill(HIST("phi/phiDiffMCRec"), deltaPhi);

      double impactParameterRPhi{-999.}, impactParameterRPhiError{-999.};
      double impactParameterZ{-999.}, impactParameterErrorZ{-999.};

      const bool propagate = track_utils::GetImpactParameterAndError(
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

/// Task to QA the efficiency of a particular particle defined by particlePDG
template <PDG_t particlePDG>
struct QATrackingEfficiency {
  o2fw::Configurable<int> nBinsEta{"nBinsEta", 30, "Number of bins for the pseudorapidity"};
  std::array<double, 2> etaRange = {-3, 3};

  o2fw::Configurable<int> nBinsPhi{"nBinsPhi", 20, "Number of bins for Phi"};
  std::array<double, 2> phiRange = {0, 2 * M_PI};

  o2fw::HistogramRegistry histos{"histogramsTrackingEfficiencyQA"};

  void init(o2fw::InitContext&)
  {
    o2fw::AxisSpec ptAxis{qafeat::PtBins};
    o2fw::AxisSpec phiAxis{nBinsPhi, phiRange[0], phiRange[1]};
    o2fw::AxisSpec etaAxis{nBinsEta, etaRange[0], etaRange[1]};

    histos.add("reconstructedKinematics",
               qafeat::MakeTitle({qafeat::Pt.MC(), qafeat::Eta.MC(), qafeat::Phi.MC()}).c_str(),
               o2fw::kTH3D, {ptAxis, etaAxis, phiAxis});

    histos.add("generatedKinematics",
               qafeat::MakeTitle({qafeat::Pt.MC(), qafeat::Eta.MC(), qafeat::Phi.MC()}).c_str(),
               o2fw::kTH3D, {ptAxis, etaAxis, phiAxis});
  }

  void process(const o2::soa::Join<o2::aod::Tracks, o2::aod::McTrackLabels>& tracks,
               const o2::aod::McParticles& mcParticles)
  {
    for (const auto& track : tracks) {
      const auto mcParticle = track.label();
      if (MC::isPhysicalPrimary(mcParticles, mcParticle) &&
          abs(mcParticle.pdgCode()) == particlePDG) {
        histos.fill(HIST("reconstructedKinematics"), mcParticle.pt(), mcParticle.eta(), mcParticle.phi());
      }
    }

    for (const auto& mcParticle : mcParticles) {
      if (MC::isPhysicalPrimary(mcParticles, mcParticle) &&
          abs(mcParticle.pdgCode()) == particlePDG) {
        histos.fill(HIST("generatedKinematics"), mcParticle.pt(), mcParticle.eta(), mcParticle.phi());
      }
    }
  }
};

o2fw::WorkflowSpec defineDataProcessing(o2fw::ConfigContext const&)
{
  return o2fw::WorkflowSpec{o2fw::adaptAnalysisTask<QAGlobalObservables>("qa-global-observables"),
                            o2fw::adaptAnalysisTask<QATrackingKine>("qa-tracking-kine"),
                            o2fw::adaptAnalysisTask<QATrackingResolution>("qa-tracking-resolution"),
                            o2fw::adaptAnalysisTask<QATrackingEfficiency<PDG_t::kPiPlus>>("qa-tracking-efficiency-pion"),
                            o2fw::adaptAnalysisTask<QATrackingEfficiency<PDG_t::kProton>>("qa-tracking-efficiency-proton"),
                            o2fw::adaptAnalysisTask<QATrackingEfficiency<PDG_t::kElectron>>("qa-tracking-efficiency-electron"),
                            o2fw::adaptAnalysisTask<QATrackingEfficiency<PDG_t::kMuonMinus>>("qa-tracking-efficiency-muon"),
                            o2fw::adaptAnalysisTask<QATrackingEfficiency<PDG_t::kKPlus>>("qa-tracking-efficiency-kaon")};
};
