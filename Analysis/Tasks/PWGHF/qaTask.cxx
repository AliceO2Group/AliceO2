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

#include "Framework/AnalysisTask.h"
#include "Framework/runDataProcessing.h"
#include "Framework/HistogramRegistry.h"
#include "ReconstructionDataFormats/DCA.h"
#include "AnalysisCore/trackUtilities.h"
#include "AnalysisCore/MC.h"
#include "AnalysisDataModel/TrackSelectionTables.h"

using namespace o2::framework;
using namespace o2::dataformats;

/// Task to QA global observables of the event
struct QaGlobalObservables {
  // Cuts
  Configurable<int> numberOfContributorsMin{"numberOfContributorsMin", 0, "Minimum required number of contributors to the vertex"};
  Configurable<float> etaMin{"etaMin", -0.8f, "Minimum eta in range to count the track multiplicity"};
  Configurable<float> etaMax{"etaMax", 0.8f, "Maximum eta in range to count the track multiplicity"};

  // Binning
  Configurable<int> numberOfTracksBins{"numberOfTracksBins", 2000, "Number of bins for the Number of Tracks"};
  Configurable<float> numberOfTracksMin{"numberOfTracksMin", 0, "Lower limit in the Number of Tracks plot"};
  Configurable<float> numberOfTracksMax{"numberOfTracksMax", 2000, "Upper limit in the Number of Tracks plot"};

  Configurable<int> vertexPositionBins{"vertexPositionBins", 100, "Number of bins for the Vertex Position"};
  Configurable<float> vetexPositionZMin{"vetexPositionZMin", -20.f, "Lower limit in the Vertex Position Z"};
  Configurable<float> vetexPositionZMax{"vetexPositionZMax", 20.f, "Upper limit in the Vertex Position Z"};
  Configurable<float> vetexPositionXYMin{"vetexPositionXYMin", -0.01f, "Lower limit in the Vertex Position XY"};
  Configurable<float> vetexPositionXYMax{"vetexPositionXYMax", 0.01f, "Upper limit in the Vertex Position XY"};

  Configurable<int> vertexPositionDeltaBins{"vertexPositionDeltaBins", 100, "Number of bins for the histograms of the difference between reconstructed and generated vertex positions"};
  Configurable<float> vetexPositionXYDeltaPtRange{"vetexPositionXYDeltaPtRange", 0.5, "Range of the resolution of the vertex position plot in X and Y"};
  Configurable<float> vetexPositionZDeltaPtRange{"vetexPositionZDeltaPtRange", 0.5, "Range of the resolution of the vertex position plot in Z"};

  Configurable<int> numbersOfContributorsToPVBins{"numbersOfContributorsToPVBins", 200, "Number bins for the number of contributors to the primary vertex"};
  Configurable<float> numbersOfContributorsToPVMax{"numbersOfContributorsToPVMax", 200, "Maximum value for the Number of contributors to the primary vertex"};

  Configurable<int> vertexCovarianceMatrixBins{"vertexCovarianceMatrixBins", 100, "Number bins for the vertex covariance matrix"};
  Configurable<float> vertexCovarianceMatrixMin{"vertexCovarianceMatrixMin", -0.01f, "Lower limit in the Vertex Covariance matrix XY"};
  Configurable<float> vertexCovarianceMatrixMax{"vertexCovarianceMatrixMax", 0.01f, "Upper limit in the Vertex Covariance matrix XY"};

  HistogramRegistry histograms{"HistogramsGlobalQA"};
  void init(InitContext&)
  {

    AxisSpec numberOfTrackAxis{numberOfTracksBins, numberOfTracksMin, numberOfTracksMax};
    AxisSpec collisionXYAxis{vertexPositionBins, vetexPositionXYMin, vetexPositionXYMax};
    AxisSpec collisionZAxis{vertexPositionBins, vetexPositionZMin, vetexPositionZMax};
    AxisSpec numberOfContributorsAxis{numbersOfContributorsToPVBins, 0, numbersOfContributorsToPVMax};
    AxisSpec vertexCovarianceMatrixAxis{vertexCovarianceMatrixBins, vertexCovarianceMatrixMin, vertexCovarianceMatrixMax};
    AxisSpec collisionXYDeltaAxis{vertexPositionDeltaBins, -vetexPositionXYDeltaPtRange, vetexPositionXYDeltaPtRange};
    AxisSpec collisionZDeltaAxis{vertexPositionDeltaBins, -vetexPositionZDeltaPtRange, vetexPositionZDeltaPtRange};

    // Global
    histograms.add("eventCount", ";Selected Events", kTH1D, {{2, 0, 2}});

    // Collision
    histograms.add("collision/X", ";X [cm]", kTH1D, {collisionXYAxis});
    histograms.add("collision/Y", ";Y [cm]", kTH1D, {collisionXYAxis});
    histograms.add("collision/Z", ";Z [cm]", kTH1D, {collisionZAxis});
    histograms.add("collision/XvsNContrib", ";X [cm];Number Of contributors to the PV", kTH2D, {collisionXYAxis, numberOfContributorsAxis});
    histograms.add("collision/YvsNContrib", ";Y [cm];Number Of contributors to the PV", kTH2D, {collisionXYAxis, numberOfContributorsAxis});
    histograms.add("collision/ZvsNContrib", ";Z [cm];Number Of contributors to the PV", kTH2D, {collisionZAxis, numberOfContributorsAxis});
    histograms.add("collision/numberOfContributors", ";Number Of contributors to the PV", kTH1D, {numberOfContributorsAxis});
    histograms.add("collision/numberOfContributorsVsMult", ";Number Of contributors to the PV;Track Multiplicity", kTH2D, {numberOfContributorsAxis, numberOfTrackAxis});
    histograms.add("collision/vertexChi2", ";#chi^{2}", kTH1D, {{100, 0, 10}});

    // Covariance
    histograms.add("covariance/xx", ";Cov_{xx} [cm^{2}]", kTH1D, {vertexCovarianceMatrixAxis});
    histograms.add("covariance/xy", ";Cov_{xy} [cm^{2}]", kTH1D, {vertexCovarianceMatrixAxis});
    histograms.add("covariance/xz", ";Cov_{xz} [cm^{2}]", kTH1D, {vertexCovarianceMatrixAxis});
    histograms.add("covariance/yy", ";Cov_{yy} [cm^{2}]", kTH1D, {vertexCovarianceMatrixAxis});
    histograms.add("covariance/yz", ";Cov_{yz} [cm^{2}]", kTH1D, {vertexCovarianceMatrixAxis});
    histograms.add("covariance/zz", ";Cov_{zz} [cm^{2}]", kTH1D, {vertexCovarianceMatrixAxis});
    // Multiplicity
    histograms.add("multiplicity/numberOfTracks", ";Track Multiplicity", kTH1D, {numberOfTrackAxis});
    // Resolution
    histograms.add("resolution/X", ";X_{Rec} - X_{Gen} [cm];Track Multiplicity", kTH2D, {collisionXYDeltaAxis, numberOfContributorsAxis});
    histograms.add("resolution/Y", ";Y_{Rec} - Y_{Gen} [cm];Track Multiplicity", kTH2D, {collisionXYDeltaAxis, numberOfContributorsAxis});
    histograms.add("resolution/Z", ";Z_{Rec} - Z_{Gen} [cm];Track Multiplicity", kTH2D, {collisionZDeltaAxis, numberOfContributorsAxis});
  }

  void process(const o2::soa::Join<o2::aod::Collisions, o2::aod::McCollisionLabels>::iterator& collision,
               const o2::aod::McCollisions&,
               const o2::aod::Tracks& tracks)
  {
    if (collision.numContrib() < numberOfContributorsMin) {
      return;
    }
    int nTracks = 0;
    for (const auto& track : tracks) {
      if (track.eta() < etaMin || track.eta() > etaMax) {
        continue;
      }
      nTracks++;
    }
    histograms.fill(HIST("eventCount"), 0);

    histograms.fill(HIST("collision/X"), collision.posX());
    histograms.fill(HIST("collision/Y"), collision.posY());
    histograms.fill(HIST("collision/Z"), collision.posZ());

    histograms.fill(HIST("collision/XvsNContrib"), collision.posX(), collision.numContrib());
    histograms.fill(HIST("collision/YvsNContrib"), collision.posY(), collision.numContrib());
    histograms.fill(HIST("collision/ZvsNContrib"), collision.posZ(), collision.numContrib());

    histograms.fill(HIST("collision/numberOfContributors"), collision.numContrib());
    histograms.fill(HIST("collision/numberOfContributorsVsMult"), collision.numContrib(), nTracks);
    histograms.fill(HIST("collision/vertexChi2"), collision.chi2());

    histograms.fill(HIST("covariance/xx"), collision.covXX());
    histograms.fill(HIST("covariance/xy"), collision.covXY());
    histograms.fill(HIST("covariance/xz"), collision.covXZ());
    histograms.fill(HIST("covariance/yy"), collision.covYY());
    histograms.fill(HIST("covariance/yz"), collision.covYZ());
    histograms.fill(HIST("covariance/zz"), collision.covZZ());

    histograms.fill(HIST("multiplicity/numberOfTracks"), nTracks);

    const auto mcColl = collision.mcCollision();
    histograms.fill(HIST("resolution/X"), collision.posX() - mcColl.posX(), collision.numContrib());
    histograms.fill(HIST("resolution/Y"), collision.posY() - mcColl.posY(), collision.numContrib());
    histograms.fill(HIST("resolution/Z"), collision.posZ() - mcColl.posZ(), collision.numContrib());
  }
};

/// Task to QA the kinematic properties of the tracks
struct QaTrackingKine {
  Configurable<int> pdgCodeSel{"pdgCodeSel", 0, "PDG code of the particle to select in absolute value, 0 selects every particle"};

  Configurable<int> ptBins{"ptBins", 100, "Number of pT bins"};
  Configurable<float> ptMin{"ptMin", 0.f, "Lower limit in pT"};
  Configurable<float> ptMax{"ptMax", 5.f, "Upper limit in pT"};

  Configurable<int> etaBins{"etaBins", 100, "Number of eta bins"};
  Configurable<float> etaMin{"etaMin", -3.f, "Lower limit in eta"};
  Configurable<float> etaMax{"etaMax", 3.f, "Upper limit in eta"};

  Configurable<int> phiBins{"phiBins", 100, "Number of phi bins"};
  Configurable<float> phiMin{"phiMin", 0.f, "Lower limit in phi"};
  Configurable<float> phiMax{"phiMax", TMath::TwoPi(), "Upper limit in phi"};

  HistogramRegistry histos{"HistogramsKineQA"};
  void init(InitContext&)
  {
    AxisSpec ptAxis{ptBins, ptMin, ptMax};
    AxisSpec etaAxis{etaBins, etaMin, etaMax};
    AxisSpec phiAxis{phiBins, phiMin, phiMax};

    TString commonTitle = "";
    if (pdgCodeSel != 0) {
      commonTitle += Form("PDG %i", pdgCodeSel.value);
    }

    const TString pt = "#it{p}_{T} [GeV/#it{c}]";
    const TString eta = "#it{#eta}";
    const TString phi = "#it{#varphi} [rad]";

    histos.add("tracking/pt", commonTitle + ";" + pt, kTH1D, {ptAxis});
    histos.add("tracking/eta", commonTitle + ";" + eta, kTH1D, {etaAxis});
    histos.add("tracking/phi", commonTitle + ";" + phi, kTH1D, {phiAxis});

    histos.add("trackingPrm/pt", commonTitle + " Primary;" + pt, kTH1D, {ptAxis});
    histos.add("trackingPrm/eta", commonTitle + " Primary;" + eta, kTH1D, {etaAxis});
    histos.add("trackingPrm/phi", commonTitle + " Primary;" + phi, kTH1D, {phiAxis});

    histos.add("trackingSec/pt", commonTitle + " Secondary;" + pt, kTH1D, {ptAxis});
    histos.add("trackingSec/eta", commonTitle + " Secondary;" + eta, kTH1D, {etaAxis});
    histos.add("trackingSec/phi", commonTitle + " Secondary;" + phi, kTH1D, {phiAxis});

    histos.add("particle/pt", commonTitle + ";" + pt, kTH1D, {ptAxis});
    histos.add("particle/eta", commonTitle + ";" + eta, kTH1D, {etaAxis});
    histos.add("particle/phi", commonTitle + ";" + phi, kTH1D, {phiAxis});

    histos.add("particlePrm/pt", commonTitle + " Primary;" + pt, kTH1D, {ptAxis});
    histos.add("particlePrm/eta", commonTitle + " Primary;" + eta, kTH1D, {etaAxis});
    histos.add("particlePrm/phi", commonTitle + " Primary;" + phi, kTH1D, {phiAxis});

    histos.add("particleSec/pt", commonTitle + " Secondary;" + pt, kTH1D, {ptAxis});
    histos.add("particleSec/eta", commonTitle + " Secondary;" + eta, kTH1D, {etaAxis});
    histos.add("particleSec/phi", commonTitle + " Secondary;" + phi, kTH1D, {phiAxis});
  }

  void process(const o2::soa::Join<o2::aod::Tracks, o2::aod::TracksCov, o2::aod::McTrackLabels>& tracks,
               const o2::aod::McParticles& mcParticles)
  {
    for (const auto& t : tracks) {
      const auto particle = t.mcParticle();
      if (pdgCodeSel != 0 && particle.pdgCode() != pdgCodeSel) { // Checking PDG code
        continue;
      }
      if (MC::isPhysicalPrimary(mcParticles, particle)) {
        histos.fill(HIST("trackingPrm/pt"), t.pt());
        histos.fill(HIST("trackingPrm/eta"), t.eta());
        histos.fill(HIST("trackingPrm/phi"), t.phi());
      } else {
        histos.fill(HIST("trackingSec/pt"), t.pt());
        histos.fill(HIST("trackingSec/eta"), t.eta());
        histos.fill(HIST("trackingSec/phi"), t.phi());
      }
      histos.fill(HIST("tracking/pt"), t.pt());
      histos.fill(HIST("tracking/eta"), t.eta());
      histos.fill(HIST("tracking/phi"), t.phi());
    }
    for (const auto& particle : mcParticles) {
      if (pdgCodeSel != 0 && particle.pdgCode() != pdgCodeSel) { // Checking PDG code
        continue;
      }
      histos.fill(HIST("particle/pt"), particle.pt());
      histos.fill(HIST("particle/eta"), particle.eta());
      histos.fill(HIST("particle/phi"), particle.phi());
      if (MC::isPhysicalPrimary(mcParticles, particle)) {
        histos.fill(HIST("particlePrm/pt"), particle.pt());
        histos.fill(HIST("particlePrm/eta"), particle.eta());
        histos.fill(HIST("particlePrm/phi"), particle.phi());
      } else {
        histos.fill(HIST("particleSec/pt"), particle.pt());
        histos.fill(HIST("particleSec/eta"), particle.eta());
        histos.fill(HIST("particleSec/phi"), particle.phi());
      }
    }
  }
};

/// Task to evaluate the tracking resolution (Pt, Eta, Phi and impact parameter)
struct QaTrackingResolution {

  Configurable<int> useOnlyPhysicsPrimary{"useOnlyPhysicsPrimary", 1,
                                          "Whether to use only physical primary particles for the resolution."};

  Configurable<int> pdgCodeSel{"pdgCodeSel", 0, "PDG code of the particle to select in absolute value, 0 selects every particle"};

  Configurable<int> ptBins{"ptBins", 100, "Number of bins for the transverse momentum"};
  Configurable<float> ptMin{"ptMin", 0.f, "Lower limit in pT"};
  Configurable<float> ptMax{"ptMax", 5.f, "Upper limit in pT"};

  Configurable<int> etaBins{"etaBins", 100, "Number of eta bins"};
  Configurable<float> etaMin{"etaMin", -3.f, "Lower limit in eta"};
  Configurable<float> etaMax{"etaMax", 3.f, "Upper limit in eta"};

  Configurable<int> phiBins{"phiBins", 100, "Number of phi bins"};
  Configurable<float> phiMin{"phiMin", 0.f, "Lower limit in phi"};
  Configurable<float> phiMax{"phiMax", TMath::TwoPi(), "Upper limit in phi"};

  Configurable<int> deltaPtBins{"deltaPtBins", 100, "Number of bins for the transverse momentum differences"};
  Configurable<float> deltaPtMin{"deltaPtMin", -0.5, "Lower limit in delta pT"};
  Configurable<float> deltaPtMax{"deltaPtMax", 0.5, "Upper limit in delta pT"};

  Configurable<int> deltaEtaBins{"deltaEtaBins", 100, "Number of bins for the pseudorapidity differences"};
  Configurable<float> deltaEtaMin{"deltaEtaMin", -0.1, "Lower limit in delta eta"};
  Configurable<float> deltaEtaMax{"deltaEtaMax", 0.1, "Upper limit in delta eta"};

  Configurable<int> deltaPhiBins{"deltaPhiBins", 100, "Number of bins for the azimuthal angle differences"};
  Configurable<float> deltaPhiMin{"deltaPhiMin", -0.1, "Lower limit in delta phi"};
  Configurable<float> deltaPhiMax{"deltaPhiMax", 0.1, "Upper limit in delta phi"};

  Configurable<int> impactParameterBins{"impactParameterBins", 2000, "Number of bins for the Impact parameter"};
  Configurable<float> impactParameterMin{"impactParameterMin", -500, "Lower limit in impact parameter (micrometers)"};
  Configurable<float> impactParameterMax{"impactParameterMax", 500, "Upper limit in impact parameter (micrometers)"};
  Configurable<float> impactParameterResoMin{"impactParameterResoMin", 0, "Lower limit in impact parameter resolution (micrometers)"};
  Configurable<float> impactParameterResoMax{"impactParameterResoMax", 1000, "Upper limit in impact parameter resolution (micrometers)"};

  HistogramRegistry histos{"HistogramsTrackingResolutionQA"};
  void init(InitContext&)
  {
    // Histogram axis definitions

    AxisSpec ptAxis{ptBins, ptMin, ptMax};
    AxisSpec deltaPtAxis{deltaPtBins, deltaPtMin, deltaPtMax};
    AxisSpec deltaPtRelativeAxis{deltaPtBins, deltaPtMin, deltaPtMax};

    AxisSpec etaAxis{etaBins, etaMin, etaMax};
    AxisSpec deltaEtaAxis{deltaEtaBins, deltaEtaMin, deltaEtaMax};

    AxisSpec phiAxis{phiBins, phiMin, phiMax};
    AxisSpec deltaPhiAxis{deltaPhiBins, deltaPhiMin, deltaPhiMax};

    AxisSpec impactParRPhiAxis{impactParameterBins, impactParameterMin, impactParameterMax};
    AxisSpec impactParRPhiErrorAxis{impactParameterBins, impactParameterResoMin, impactParameterResoMax};

    AxisSpec impactParZAxis{impactParameterBins, impactParameterMin, impactParameterMax};
    AxisSpec impactParZErrorAxis{impactParameterBins, impactParameterResoMin, impactParameterResoMax};

    TString commonTitle = "";
    if (pdgCodeSel != 0) {
      commonTitle += Form("PDG %i", pdgCodeSel.value);
    }
    if (useOnlyPhysicsPrimary == 1) {
      commonTitle += " Primary";
    }
    const TString pt = "#it{p}_{T} [GeV/#it{c}]";
    const TString eta = "#it{#eta}";
    const TString phi = "#it{#varphi} [rad]";

    const TString ptRec = "#it{p}_{T}_{Rec} [GeV/#it{c}]";
    const TString etaRec = "#it{#eta}_{Rec}";
    const TString phiRec = "#it{#varphi}_{Rec} [rad]";

    const TString ptGen = "#it{p}_{T}_{Gen} [GeV/#it{c}]";
    const TString etaGen = "#it{#eta}_{Gen}";

    const TString ptDelta = "#it{p}_{T}_{Rec} - #it{p}_{T}_{Gen} [GeV/#it{c}]";
    const TString ptReso = "(#it{p}_{T}_{Rec} - #it{p}_{T}_{Gen})/(#it{p}_{T}_{Gen})";
    const TString etaDelta = "#it{#eta}_{Rec} - #it{#eta}_{Gen}";

    // Eta
    histos.add("eta/etaDiffRecGen", commonTitle + ";" + etaDelta, kTH1D, {deltaEtaAxis});
    histos.add("eta/etaDiffRecGenVsEtaGen", commonTitle + ";" + etaDelta + ";" + etaGen, kTH2D, {deltaEtaAxis, etaAxis});
    histos.add("eta/etaDiffRecGenVsEtaRec", commonTitle + ";" + etaDelta + ";" + etaRec, kTH2D, {deltaEtaAxis, etaAxis});

    // Phi
    histos.add("phi/phiDiffRecGen", commonTitle + ";#it{#varphi}_{Gen} - #it{#varphi}_{Rec} [rad]", kTH1D, {deltaPhiAxis});

    // Pt
    histos.add("pt/ptDiffRecGen", commonTitle + ";" + ptDelta, kTH1D, {deltaPtAxis});
    histos.add("pt/ptResolution", commonTitle + ";" + ptReso, kTH1D, {deltaPtRelativeAxis});
    histos.add("pt/ptResolutionVsPt", commonTitle + ";" + ptRec + ";" + ptReso, kTH2D, {ptAxis, deltaPtRelativeAxis});
    histos.add("pt/ptResolutionVsEta", commonTitle + ";" + eta + ";" + ptReso, kTH2D, {etaAxis, deltaPtRelativeAxis});
    histos.add("pt/ptResolutionVsPhi", commonTitle + ";" + phi + ";" + ptReso, kTH2D, {phiAxis, deltaPtRelativeAxis});

    // Impact parameters
    const TString impRPhi = "Impact Parameter r#it{#varphi} [#mum]";
    const TString impRPhiErr = "Impact Parameter Error r#it{#varphi} [#mum]";

    histos.add("impactParameter/impactParameterRPhiVsPt", commonTitle + ";" + ptRec + ";" + impRPhi, kTH2D, {ptAxis, impactParRPhiAxis});
    histos.add("impactParameter/impactParameterRPhiVsEta", commonTitle + ";" + etaRec + ";" + impRPhi, kTH2D, {etaAxis, impactParRPhiAxis});
    histos.add("impactParameter/impactParameterRPhiVsPhi", commonTitle + ";" + phiRec + ";" + impRPhi, kTH2D, {phiAxis, impactParRPhiAxis});

    histos.add("impactParameter/impactParameterErrorRPhiVsPt", commonTitle + ";" + ptRec + ";" + impRPhiErr, kTH2D, {ptAxis, impactParRPhiErrorAxis});
    histos.add("impactParameter/impactParameterErrorRPhiVsEta", commonTitle + ";" + etaRec + ";" + impRPhiErr, kTH2D, {etaAxis, impactParRPhiErrorAxis});
    histos.add("impactParameter/impactParameterErrorRPhiVsPhi", commonTitle + ";" + phiRec + ";" + impRPhiErr, kTH2D, {phiAxis, impactParRPhiErrorAxis});

    const TString impZ = "Impact Parameter Z [#mum]";
    const TString impZErr = "Impact Parameter Error Z [#mum]";

    histos.add("impactParameter/impactParameterZVsPt", commonTitle + ";" + ptRec + ";" + impZ, kTH2D, {ptAxis, impactParZAxis});
    histos.add("impactParameter/impactParameterZVsEta", commonTitle + ";" + etaRec + ";" + impZ, kTH2D, {etaAxis, impactParZAxis});
    histos.add("impactParameter/impactParameterZVsPhi", commonTitle + ";" + phiRec + ";" + impZ, kTH2D, {phiAxis, impactParZAxis});

    histos.add("impactParameter/impactParameterErrorZVsPt", commonTitle + ";" + ptRec + ";" + impZErr, kTH2D, {ptAxis, impactParZErrorAxis});
    histos.add("impactParameter/impactParameterErrorZVsEta", commonTitle + ";" + etaRec + ";" + impZErr, kTH2D, {etaAxis, impactParZErrorAxis});
    histos.add("impactParameter/impactParameterErrorZVsPhi", commonTitle + ";" + phiRec + ";" + impZErr, kTH2D, {phiAxis, impactParZErrorAxis});
  }

  void process(const o2::soa::Join<o2::aod::Collisions, o2::aod::McCollisionLabels>::iterator& collision,
               const o2::soa::Join<o2::aod::Tracks, o2::aod::TracksCov, o2::aod::McTrackLabels>& tracks,
               const o2::aod::McParticles& mcParticles, const o2::aod::McCollisions& mcCollisions)
  {
    const VertexBase primaryVertex = getPrimaryVertex(collision);
    DCA dca;
    // FIXME: get this from CCDB
    constexpr float magneticField{5.0};      // in kG
    constexpr float toMicrometers = 10000.f; // Conversion from [cm] to [mum]
    float impactParameterRPhi = -999.f;
    float impactParameterRPhiError = -999.f;
    float impactParameterZ = -999.f;
    float impactParameterErrorZ = -999.f;

    for (const auto& track : tracks) {

      const auto particle = track.mcParticle();
      if (pdgCodeSel != 0 && particle.pdgCode() != pdgCodeSel) {
        continue;
      }
      if (useOnlyPhysicsPrimary && !MC::isPhysicalPrimary(mcParticles, particle)) {
        continue;
      }
      const double deltaPt = track.pt() - particle.pt();
      histos.fill(HIST("pt/ptDiffRecGen"), deltaPt);

      const double deltaPtOverPt = deltaPt / particle.pt();

      histos.fill(HIST("pt/ptResolution"), deltaPtOverPt);
      histos.fill(HIST("pt/ptResolutionVsPt"), track.pt(), deltaPtOverPt);
      histos.fill(HIST("pt/ptResolutionVsEta"), track.eta(), deltaPtOverPt);
      histos.fill(HIST("pt/ptResolutionVsPhi"), track.phi(), deltaPtOverPt);

      const double deltaEta = track.eta() - particle.eta();
      histos.fill(HIST("eta/etaDiffRecGen"), deltaEta);
      histos.fill(HIST("eta/etaDiffRecGenVsEtaGen"), deltaEta, particle.eta());
      histos.fill(HIST("eta/etaDiffRecGenVsEtaRec"), deltaEta, track.eta());

      histos.fill(HIST("phi/phiDiffRecGen"), track.phi() - particle.phi());

      if (getTrackParCov(track).propagateToDCA(primaryVertex, magneticField, &dca, 100.)) { // Check that the propagation is successfull
        impactParameterRPhi = toMicrometers * dca.getY();
        impactParameterRPhiError = toMicrometers * sqrt(dca.getSigmaY2());
        impactParameterZ = toMicrometers * dca.getZ();
        impactParameterErrorZ = toMicrometers * sqrt(dca.getSigmaZ2());

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

WorkflowSpec defineDataProcessing(ConfigContext const& cfgc)
{
  return WorkflowSpec{adaptAnalysisTask<QaGlobalObservables>(cfgc),
                      adaptAnalysisTask<QaTrackingKine>(cfgc),
                      adaptAnalysisTask<QaTrackingResolution>(cfgc)};
}
