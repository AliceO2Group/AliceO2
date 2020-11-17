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

#include "TH1D.h"

#include <cmath>
#include <string>
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

namespace o2::qa
{

/// Class to abstract the naming of a particular feature. It can help you to build the histogram
/// labels in a consistent way, and can generate the titles. You can also pass the
/// plotWithMatplotlib when constructing it to make the labels friendly to matplotlib.
class Feature
{
 public:
  Feature() = default;
  Feature(std::string name, std::string unit) : mName(std::move(name)), mUnit(std::move(unit)){};

  std::string MCRaw() const { return mName + "^{MC}"; };
  std::string RecRaw() const { return mName + "^{Rec}"; };
  std::string UnitRaw() const { return mUnit; };
  std::string NameRaw() const { return mName; };

  std::string Name() const { return mName + UnitFormatted(); };
  std::string MC() const { return MCRaw() + UnitFormatted(); };
  std::string Rec() const { return RecRaw() + UnitFormatted(); };

  std::string MCRecDiff() const { return MCRaw() + " - " + RecRaw() + UnitFormatted(); };
  std::string RelativeMCRecDiff() const { return "(" + MCRaw() + " - " + RecRaw() + ")/(" + RecRaw() + ")"; };

  std::string UnitFormatted() const
  {
    if (UnitRaw().empty()) {
      return "";
    }
    return " [" + UnitRaw() + "]";
  };

 private:
  const bool mPlotWithMatplotlib{false};
  const std::string mName;
  const std::string mUnit;
};

/// Handle consistent naming of the histograms in the task and adds the possibility to make the
/// histogram titles friendly to matplotlib.
class Features
{
 public:
  Features(bool plotWithMatplotlib = false, std::string countsLabel = "Counts")
    : mPlotWithMatplotlib(plotWithMatplotlib), mCountLabel(countsLabel){};

  const std::string& CountsLabel() const { return mCountLabel; }
  bool PlotWithMatplotlib() const { return mPlotWithMatplotlib; }

  /// Given a string with the content of the X title, builds the title field for the histogram using
  /// an empty title and the string that represents the number of counts.
  std::string Title1D(const std::string& xAxis) const
  {
    return RemoveExtraMathSymbols(";" + xAxis + ";" + CountsLabel());
  }

  /// Given a string with the content of the X and Y titles, builds the title field for the
  /// histogram using an empty title and the string that represents the number of counts.
  std::string Title2D(const std::string& xAxis, const std::string& yAxis) const
  {
    return RemoveExtraMathSymbols(";" + xAxis + ";" + yAxis + ";" + CountsLabel());
  }

  /// Returns the symbol used to give math commands: \ for matplotlib and # for ROOT.
  std::string MathSymbol() const
  {
    if (PlotWithMatplotlib()) {
      return "\\";
    }
    return "#";
  }

  /// Returns the symbol used to start/end math text: $ for matplotlib and nothing for ROOT.
  std::string MathDelimiter() const
  {
    if (PlotWithMatplotlib()) {
      return "$";
    }
    return "";
  }
  /// Tags that this text should be displayed in math mode using the correct MathDelimiter.
  std::string MathText(const std::string& text) const { return MathDelimiter() + text + MathDelimiter(); }

  std::string RemoveExtraMathSymbols(std::string text) const
  {
    if (!PlotWithMatplotlib())
      return text;

    boost::replace_all(text, MathDelimiter() + MathDelimiter(), "");
    return text;
  }

 private:
  const bool mPlotWithMatplotlib;
  const std::string mCountLabel;
};

class QAFeatures : public Features
{
 public:
  QAFeatures(bool plotWithMatplotlib = false, std::string countsLabel = "Counts")
    : Features(plotWithMatplotlib, countsLabel),
      mTrackMultiplicity("Track Multiplicity", ""),
      mEta(MathText(MathSymbol() + "eta"), ""),
      mPhi{MathText(MathSymbol() + "varphi"), "rad"},
      mPt(MathText("p_{T}"), "GeV/c"),
      mVertexX("X", "cm"),
      mVertexY("Y", "cm"),
      mVertexZ("Z", "cm"),
      mImpactParameterRPhi("Impact Parameter r" + MathText(MathSymbol() + "varphi"),
                           MathText(MathSymbol() + "mu") + "m"),
      mImpactParameterRPhiError("Impact Parameter Error r" + MathText(MathSymbol() + "varphi"),
                                MathText(MathSymbol() + "mu") + "m"),
      mImpactParameterZ("Impact Parameter Z", MathText(MathSymbol() + "mu") + "m"),
      mImpactParameterZError("Impact Parameter Z Error ", MathText("mu") + "m"){};

  const Feature& Eta() const { return mEta; }
  const Feature& Phi() const { return mPhi; }
  const Feature& Pt() const { return mPt; }
  const Feature& VertexX() const { return mVertexX; }
  const Feature& VertexY() const { return mVertexY; }
  const Feature& VertexZ() const { return mVertexZ; }
  const Feature& TrackMultiplicity() const { return mTrackMultiplicity; }
  const Feature& ImpactParameterRPhi() const { return mImpactParameterRPhi; }
  const Feature& ImpactParameterRPhiError() const { return mImpactParameterRPhiError; }
  const Feature& ImpactParameterZ() const { return mImpactParameterZ; }
  const Feature& ImpactParameterZError() const { return mImpactParameterZError; }

 private:
  const Feature mTrackMultiplicity;
  const Feature mEta;
  const Feature mPhi;
  const Feature mPt;
  const Feature mVertexX;
  const Feature mVertexY;
  const Feature mVertexZ;
  const Feature mImpactParameterRPhi;
  const Feature mImpactParameterRPhiError;
  const Feature mImpactParameterZ;
  const Feature mImpactParameterZError;
};
} // namespace o2::qa

/// Task to QA global observables of the event
struct QAGlobalObservables {
  o2fw::Configurable<bool> histogramAxisForMatplotlib{
    "histogramAxisForMatplotlib", false, "Sets the histograms title to be friendly to matplotlib instead of ROOT"};
  o2::qa::QAFeatures qa = o2::qa::QAFeatures(histogramAxisForMatplotlib);

  o2fw::Configurable<int> nBinsNumberOfTracks{"nBinsNumberOfTracks", 2000, "Number of bins fot the Number of Tracks"};
  o2fw::Configurable<int> nBinsVertexPosition{"nBinsPt", 100, "Number of bins for the Vertex Position"};

  std::array<float, 2> collisionZRange = {-20., 20.};
  std::array<float, 2> collisionXYRange = {-0.01, 0.01};

  std::array<float, 2> numberOfTracksRange = {0, 400};

  o2fw::OutputObj<TH1D> eventCount{TH1D("eventCount", qa.Title1D("").c_str(), 2, 0, 2)};
  o2fw::HistogramRegistry histograms{"HistogramsGlobalQA"};

  void init(o2fw::InitContext&)
  {
    histograms.add("collision/collisionX", qa.Title1D(qa.VertexX().Name()).c_str(), o2fw::kTH1D,
                   {{nBinsVertexPosition, collisionXYRange[0], collisionXYRange[1]}});

    histograms.add("collision/collisionY", qa.Title1D(qa.VertexY().Name()).c_str(), o2fw::kTH1D,
                   {{nBinsVertexPosition, collisionXYRange[0], collisionXYRange[1]}});

    histograms.add("collision/collisionZ", qa.Title1D(qa.VertexZ().Name()).c_str(), o2fw::kTH1D,
                   {{nBinsVertexPosition, collisionZRange[0], collisionZRange[1]}});

    histograms.add("multiplicity/numberOfTracks", qa.Title1D(qa.TrackMultiplicity().Name()).c_str(), o2fw::kTH1D,
                   {{nBinsNumberOfTracks, numberOfTracksRange[0], numberOfTracksRange[1]}});
  }

  void process(const o2::aod::Collision& collision, const o2::aod::Tracks& tracks)
  {
    eventCount->Fill(0);
    histograms.fill("collision/collisionX", collision.posX());
    histograms.fill("collision/collisionY", collision.posY());
    histograms.fill("collision/collisionZ", collision.posZ());

    int nTracks(0);
    for (const auto& track : tracks) {
      nTracks++;
    }

    histograms.fill("multiplicity/numberOfTracks", nTracks);
  }
};

/// Task to QA the kinematic properties of the tracks
struct QATrackingKine {
  o2fw::Configurable<bool> histogramAxisForMatplotlib{
    "histogramAxisForMatplotlib", false, "Sets the histograms title to be friendly to matplotlib instead of ROOT"};
  o2::qa::QAFeatures qa = o2::qa::QAFeatures(histogramAxisForMatplotlib);
  o2fw::Configurable<int> nBinsPt{"nBinsPt", 100, "Number of bins for Pt"};
  std::array<double, 2> ptRange = {0, 10.};

  o2fw::Configurable<int> nBinsPhi{"nBinsPhi", 100, "Number of bins for Phi"};

  o2fw::Configurable<int> nBinsEta{"nBinsEta", 100, "Number of bins for the eta histogram."};
  std::array<double, 2> etaRange = {-6, 6};

  o2fw::HistogramRegistry histos{"HistogramsKineQA"};

  void init(o2fw::InitContext&)
  {
    histos.add("tracking/pt", qa.Title1D(qa.Pt().Name()).c_str(), o2fw::kTH1D, {{nBinsPt, ptRange[0], ptRange[1]}});
    histos.add("tracking/eta", qa.Title1D(qa.Eta().NameRaw()).c_str(), o2fw::kTH1D,
               {{nBinsEta, etaRange[0], etaRange[1]}});
    histos.add("tracking/phi", qa.Title1D(qa.Phi().Name()).c_str(), o2fw::kTH1D, {{nBinsPhi, 0, 2 * M_PI}});
  }

  void process(const o2::aod::Track& track)
  {
    histos.fill("tracking/eta", track.eta());
    histos.fill("tracking/pt", track.pt());
    histos.fill("tracking/phi", track.phi());
  }
};

/// Task to evaluate the tracking resolution (Pt, Eta, Phi and impact parameter)
struct QATrackingResolution {
  o2fw::Configurable<bool> setupHistogramAxisForMatplotlib{
    "setupHistogramAxisForMatplotlib", false, "Sets the histograms title to be friendly to matplotlib instead of ROOT"};
  o2::qa::QAFeatures qa = o2::qa::QAFeatures(setupHistogramAxisForMatplotlib);

  o2fw::Configurable<int> nBinsPt{"nBinsPt", 100, "Number of bins for the transverse momentum"};
  std::array<double, 2> ptRange = {0, 10.};

  o2fw::Configurable<int> nBinsEta{"nBinsEta", 400, "Number of bins for the pseudorapidity"};
  std::array<double, 2> etaRange = {-3, 3};

  o2fw::Configurable<int> nBinsPhi{"nBinsPhi", 100, "Number of bins for Phi"};
  std::array<double, 2> phiRange = {0, 2 * M_PI};

  o2fw::Configurable<int> nBinsDeltaPt{"nBinsDeltaPt", 400, "Number of bins for the transverse momentum differences"};
  std::array<double, 2> deltaPtRange = {-0.5, 0.5};

  o2fw::Configurable<int> nBinsDeltaPhi{"nBinsDeltaPhi", 100, "Number of bins for the azimuthal angle differences"};
  std::array<double, 2> deltaPhiRange = {-0.1, 0.1};

  o2fw::Configurable<int> nBinsDeltaEta{"nBinsDeltaEta", 100, "Number of bins for the pseudorapidity differences"};
  std::array<double, 2> deltaEtaRange = {-0.1, 0.1};

  o2fw::Configurable<int> nBinsImpactParameter{"nBinsImpactParameter", 1000, "Number of bins for the Impact parameter"};

  std::array<double, 2> impactParameterRange = {-1500, 1500};       // micrometer
  std::array<double, 2> impactParameterResolutionRange = {0, 1000}; // micrometer

  // Registry of histograms
  o2fw::HistogramRegistry histos{"HistogramsTrackingResolutionQA"};

  void init(o2fw::InitContext&)
  {
    // Eta
    histos.add("eta/etaDiffMCReco", qa.Title1D(qa.Eta().MCRecDiff()).c_str(), o2fw::kTH1D,
               {{nBinsDeltaEta, deltaEtaRange[0], deltaEtaRange[1]}});

    histos.add("eta/etaDiffMCRecoVsEtaMC", qa.Title2D(qa.Eta().MCRecDiff(), qa.Eta().MC()).c_str(), o2fw::kTH2D,
               {{nBinsDeltaEta, deltaEtaRange[0], deltaEtaRange[1]}, {nBinsEta, etaRange[0], etaRange[1]}});

    histos.add("eta/etaDiffMCRecoVsEtaReco", qa.Title2D(qa.Eta().MCRecDiff(), qa.Eta().Rec()).c_str(), o2fw::kTH2D,
               {{nBinsDeltaEta, deltaEtaRange[0], deltaEtaRange[1]}, {nBinsEta, etaRange[0], etaRange[1]}});

    // Phi
    histos.add("phi/phiDiffMCRec", qa.Title1D(qa.Phi().MCRecDiff()).c_str(), o2fw::kTH1D,
               {{nBinsDeltaPhi, deltaPhiRange[0], deltaPhiRange[1]}});

    // Pt
    histos.add("pt/ptDiffMCRec", qa.Title1D(qa.Pt().MCRecDiff()).c_str(), o2fw::kTH1D,
               {{nBinsDeltaPt, deltaPtRange[0], deltaPtRange[1]}});

    histos.add("pt/ptResolution", qa.Title1D(qa.Pt().RelativeMCRecDiff()).c_str(), o2fw::kTH1D,
               {{nBinsDeltaPt, -1., 1.}});

    histos.add("pt/ptResolutionVsPt", qa.Title2D(qa.Pt().Rec(), qa.Pt().RelativeMCRecDiff()).c_str(), o2fw::kTH2D,
               {{nBinsPt, ptRange[0], ptRange[1]}, {nBinsDeltaPt, 0., deltaPtRange[1]}});

    histos.add("pt/ptResolutionVsEta", qa.Title2D(qa.Eta().Rec(), qa.Pt().RelativeMCRecDiff()).c_str(), o2fw::kTH2D,
               {{nBinsEta, etaRange[0], etaRange[1]}, {nBinsDeltaPt, 0., deltaPtRange[1]}});

    histos.add("pt/ptResolutionVsPhi", qa.Title2D(qa.Phi().Rec(), qa.Pt().RelativeMCRecDiff()).c_str(), o2fw::kTH2D,
               {{nBinsPhi, phiRange[0], phiRange[1]}, {nBinsDeltaPt, 0., deltaPtRange[1]}});

    // Impact parameters
    histos.add(
      "impactParameter/impactParameterRPhiVsPt", qa.Title2D(qa.Pt().Rec(), qa.ImpactParameterRPhi().Name()).c_str(),
      o2fw::kTH2D,
      {{nBinsPt, ptRange[0], ptRange[1]}, {nBinsImpactParameter, impactParameterRange[0], impactParameterRange[1]}});

    histos.add(
      "impactParameter/impactParameterRPhiVsEta", qa.Title2D(qa.Eta().Rec(), qa.ImpactParameterRPhi().Name()).c_str(),
      o2fw::kTH2D,
      {{nBinsEta, etaRange[0], etaRange[1]}, {nBinsImpactParameter, impactParameterRange[0], impactParameterRange[1]}});

    histos.add(
      "impactParameter/impactParameterRPhiVsPhi", qa.Title2D(qa.Phi().Rec(), qa.ImpactParameterRPhi().Name()).c_str(),
      o2fw::kTH2D,
      {{nBinsPhi, phiRange[0], phiRange[1]}, {nBinsImpactParameter, impactParameterRange[0], impactParameterRange[1]}});

    histos.add("impactParameter/impactParameterErrorRPhiVsPt",
               qa.Title2D(qa.Pt().Rec(), qa.ImpactParameterRPhiError().Name()).c_str(), o2fw::kTH2D,
               {{nBinsPt, ptRange[0], ptRange[1]},
                {nBinsImpactParameter, impactParameterResolutionRange[0], impactParameterResolutionRange[1]}});

    histos.add("impactParameter/impactParameterErrorRPhiVsEta",
               qa.Title2D(qa.Eta().Rec(), qa.ImpactParameterRPhiError().Name()).c_str(), o2fw::kTH2D,
               {{nBinsEta, etaRange[0], etaRange[1]},
                {nBinsImpactParameter, impactParameterResolutionRange[0], impactParameterResolutionRange[1]}});

    histos.add("impactParameter/impactParameterErrorRPhiVsPhi",
               qa.Title2D(qa.Phi().Rec(), qa.ImpactParameterRPhiError().Name()).c_str(), o2fw::kTH2D,
               {{nBinsPhi, phiRange[0], phiRange[1]},
                {nBinsImpactParameter, impactParameterResolutionRange[0], impactParameterResolutionRange[1]}});

    histos.add(
      "impactParameter/impactParameterZVsPt", qa.Title2D(qa.Pt().Rec(), qa.ImpactParameterZ().Name()).c_str(),
      o2fw::kTH2D,
      {{nBinsPt, ptRange[0], ptRange[1]}, {nBinsImpactParameter, impactParameterRange[0], impactParameterRange[1]}});

    histos.add(
      "impactParameter/impactParameterZVsEta", qa.Title2D(qa.Eta().Rec(), qa.ImpactParameterZ().Name()).c_str(),
      o2fw::kTH2D,
      {{nBinsEta, etaRange[0], etaRange[1]}, {nBinsImpactParameter, impactParameterRange[0], impactParameterRange[1]}});

    histos.add(
      "impactParameter/impactParameterZVsPhi", qa.Title2D(qa.Phi().Rec(), qa.ImpactParameterZ().Name()).c_str(),
      o2fw::kTH2D,
      {{nBinsPhi, phiRange[0], phiRange[1]}, {nBinsImpactParameter, impactParameterRange[0], impactParameterRange[1]}});

    histos.add("impactParameter/impactParameterErrorZVsPt",
               qa.Title2D(qa.Pt().Rec(), qa.ImpactParameterZError().Name()).c_str(), o2fw::kTH2D,
               {{nBinsPt, ptRange[0], ptRange[1]},
                {nBinsImpactParameter, impactParameterResolutionRange[0], impactParameterResolutionRange[1]}});

    histos.add(
      "impactParameter/impactParameterErrorZVsEta",
      qa.Title2D(qa.Eta().Rec(), qa.ImpactParameterZError().Name()).c_str(), o2fw::kTH2D,
      {{nBinsEta, etaRange[0], etaRange[1]}, {nBinsImpactParameter, impactParameterRange[0], impactParameterRange[1]}});

    histos.add(
      "impactParameter/impactParameterErrorZVsPhi",
      qa.Title2D(qa.Phi().Rec(), qa.ImpactParameterZError().Name()).c_str(), o2fw::kTH2D,
      {{nBinsPhi, phiRange[0], phiRange[1]}, {nBinsImpactParameter, impactParameterRange[0], impactParameterRange[1]}});
  }

  void process(const o2::soa::Join<o2::aod::Collisions, o2::aod::McCollisionLabels>::iterator& collision,
               const o2::soa::Join<o2::aod::Tracks, o2::aod::TracksCov, o2::aod::McTrackLabels>& tracks,
               const o2::aod::McParticles& mcParticles, const o2::aod::McCollisions& mcCollisions)
  {
    const o2df::VertexBase primaryVertex = getPrimaryVertex(collision);

    for (const auto& track : tracks) {
      const double deltaPt = track.label().pt() - track.pt();
      histos.fill("pt/ptDiffMCRec", deltaPt);

      const double deltaPtOverPt = deltaPt / track.pt();

      histos.fill("pt/ptResolution", deltaPtOverPt);
      histos.fill("pt/ptResolutionVsPt", track.pt(), std::abs(deltaPtOverPt));
      histos.fill("pt/ptResolutionVsEta", track.eta(), std::abs(deltaPtOverPt));
      histos.fill("pt/ptResolutionVsPhi", track.phi(), std::abs(deltaPtOverPt));

      const double deltaEta = track.label().eta() - track.eta();
      histos.fill("eta/etaDiffMCReco", deltaEta);
      histos.fill("eta/etaDiffMCRecoVsEtaMC", deltaEta, track.label().eta());
      histos.fill("eta/etaDiffMCRecoVsEtaReco", deltaEta, track.eta());

      const double deltaPhi = track_utils::ConvertPhiRange(track.label().phi() - track.phi());
      histos.fill("phi/phiDiffMCRec", deltaPhi);

      double impactParameterRPhi{-999.}, impactParameterRPhiError{-999.};
      double impactParameterZ{-999.}, impactParameterErrorZ{-999.};

      const bool propagate = track_utils::GetImpactParameterAndError(
        track, primaryVertex, impactParameterRPhi, impactParameterRPhiError, impactParameterZ, impactParameterErrorZ);

      if (propagate) {
        histos.fill("impactParameter/impactParameterRPhiVsPt", track.pt(), impactParameterRPhi);
        histos.fill("impactParameter/impactParameterRPhiVsEta", track.eta(), impactParameterRPhi);
        histos.fill("impactParameter/impactParameterRPhiVsPhi", track.phi(), impactParameterRPhi);

        histos.fill("impactParameter/impactParameterZVsPt", track.pt(), impactParameterZ);
        histos.fill("impactParameter/impactParameterZVsEta", track.eta(), impactParameterZ);
        histos.fill("impactParameter/impactParameterZVsPhi", track.phi(), impactParameterZ);

        histos.fill("impactParameter/impactParameterErrorRPhiVsPt", track.pt(), impactParameterRPhiError);
        histos.fill("impactParameter/impactParameterErrorRPhiVsEta", track.eta(), impactParameterRPhiError);
        histos.fill("impactParameter/impactParameterErrorRPhiVsPhi", track.phi(), impactParameterRPhiError);

        histos.fill("impactParameter/impactParameterErrorZVsPt", track.pt(), impactParameterErrorZ);
        histos.fill("impactParameter/impactParameterErrorZVsEta", track.eta(), impactParameterErrorZ);
        histos.fill("impactParameter/impactParameterErrorZVsPhi", track.phi(), impactParameterErrorZ);
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
