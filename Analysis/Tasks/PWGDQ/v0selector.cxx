// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.
//
// Example analysis task to select clean V0 sample
// ========================
//
// This code loops over a V0Data table and produces some
// standard analysis output. It requires either
// the lambdakzerofinder or the lambdakzeroproducer tasks
// to have been executed in the workflow (before).
//
//    Comments, questions, complaints, suggestions?
//    Please write to:
//    daiki.sekihata@cern.ch
//
#include "Framework/runDataProcessing.h"
#include "Framework/AnalysisTask.h"
#include "Framework/AnalysisDataModel.h"
#include "Framework/ASoAHelpers.h"
#include "ReconstructionDataFormats/Track.h"
#include "AnalysisCore/trackUtilities.h"
#include "AnalysisDataModel/StrangenessTables.h"
#include "AnalysisCore/TrackSelection.h"
#include "AnalysisDataModel/TrackSelectionTables.h"
#include "AnalysisDataModel/EventSelection.h"
#include "AnalysisDataModel/Centrality.h"
#include "AnalysisCore/RecoDecay.h"
#include "DetectorsVertexing/DCAFitterN.h"

#include <Math/Vector4D.h>
#include <array>
#include "Framework/ASoAHelpers.h"
#include "AnalysisDataModel/PID/PIDResponse.h"

using namespace o2;
using namespace o2::framework;
using namespace o2::framework::expressions;
using std::array;

using FullTracksExt = soa::Join<aod::FullTracks, aod::TracksExtra, aod::TracksExtended,
                                aod::pidTPCFullEl, aod::pidTPCFullPi,
                                aod::pidTPCFullKa, aod::pidTPCFullPr,
                                aod::pidTOFFullEl, aod::pidTOFFullPi,
                                aod::pidTOFFullKa, aod::pidTOFFullPr, aod::pidTOFbeta>;

namespace o2::aod
{

namespace reducedv0
{
DECLARE_SOA_INDEX_COLUMN(Collision, collision); //!
DECLARE_SOA_COLUMN(v0Pt, pt, float);            //!
DECLARE_SOA_COLUMN(v0Eta, eta, float);          //!
DECLARE_SOA_COLUMN(v0Phi, phi, float);          //!
DECLARE_SOA_COLUMN(v0Mass, mass, float);        //!
} //namespace reducedv0

// basic track information
DECLARE_SOA_TABLE(ReducedV0s, "AOD", "REDUCEDV0", //!
                  o2::soa::Index<>, reducedv0::CollisionId, reducedv0::v0Pt, reducedv0::v0Eta, reducedv0::v0Phi, reducedv0::v0Mass);

// iterators
using ReducedV0 = ReducedV0s::iterator;

//namespace v0bits
//{
//DECLARE_SOA_COLUMN(bit, bit, unsigned int);            //!
//} //namespace v0bit
//
//// basic track information
//DECLARE_SOA_TABLE(V0Bits, "AOD", "V0BITS", //!
//                  o2::soa::Index<>, v0bits::bit);
//
//// iterators
//using V0Bit = V0Bits::iterator;

} // namespace o2::aod

struct v0selector {

  Produces<aod::ReducedV0s> v0Gamma;
  //Produces<aod::V0Bits> v0bits;

  enum { // Reconstructed V0
    kUndef = -1,
    kGamma = 0,
    kK0S = 1,
    kLambda = 2,
    kAntiLambda = 3
  };

  float alphav0(const array<float, 3>& ppos, const array<float, 3>& pneg)
  {
    std::array<float, 3> pv0 = {ppos[0] + pneg[0], ppos[1] + pneg[1], ppos[2] + pneg[2]};
    float momTot = RecoDecay::P(pv0);
    float lQlNeg = RecoDecay::dotProd(pneg, pv0) / momTot;
    float lQlPos = RecoDecay::dotProd(ppos, pv0) / momTot;
    return (lQlPos - lQlNeg) / (lQlPos + lQlNeg); //longitudinal momentum asymmetry
  }

  float qtarmv0(const array<float, 3>& ppos, const array<float, 3>& pneg)
  {
    std::array<float, 3> pv0 = {ppos[0] + pneg[0], ppos[1] + pneg[1], ppos[2] + pneg[2]};
    float momTot2 = RecoDecay::P2(pv0);
    float dp = RecoDecay::dotProd(pneg, pv0);
    return std::sqrt(RecoDecay::P2(pneg) - dp * dp / momTot2); //qtarm
  }

  float phivv0(const array<float, 3>& ppos, const array<float, 3>& pneg, const int cpos, const int cneg, const float bz)
  {
    //momentum of e+ and e- in (ax,ay,az) axis. Note that az=0 by definition.
    //vector product of pep X pem
    float vpx = 0, vpy = 0, vpz = 0;
    if (cpos * cneg > 0.) { // Like Sign
      if (bz * cpos < 0) {
        vpx = ppos[1] * pneg[2] - ppos[2] * pneg[1];
        vpy = ppos[2] * pneg[0] - ppos[0] * pneg[2];
        vpz = ppos[0] * pneg[1] - ppos[2] * pneg[0];
      } else {
        vpx = pneg[1] * ppos[2] - pneg[2] * ppos[1];
        vpy = pneg[2] * ppos[0] - pneg[0] * ppos[2];
        vpz = pneg[0] * ppos[1] - pneg[2] * ppos[0];
      }

    } else { // Unlike Sign
      if (bz * cpos > 0) {
        vpx = ppos[1] * pneg[2] - ppos[2] * pneg[1];
        vpy = ppos[2] * pneg[0] - ppos[0] * pneg[2];
        vpz = ppos[0] * pneg[1] - ppos[2] * pneg[0];
      } else {
        vpx = pneg[1] * ppos[2] - pneg[2] * ppos[1];
        vpy = pneg[2] * ppos[0] - pneg[0] * ppos[2];
        vpz = pneg[0] * ppos[1] - pneg[2] * ppos[0];
      }
    }
    
    float vp = RecoDecay::P(array{vpx, vpy, vpz});
    //unit vector of pep X pem
    float vx = vpx / vp;
    float vy = vpy / vp;
    float vz = vpz / vp;
    
    float px = ppos[0] + pneg[0];
    float py = ppos[1] + pneg[1];
    float pz = ppos[2] + pneg[2];

    //unit vector of (pep+pem)
    float pl = RecoDecay::P(array{px, py, pz});
    float ux = px / pl;
    float uy = py / pl;
    float uz = pz / pl;
    float ax = uy / RecoDecay::sqrtSumOfSquares(ux, uy);
    float ay = -ux / RecoDecay::sqrtSumOfSquares(ux, uy);
   
    //The third axis defined by vector product (ux,uy,uz)X(vx,vy,vz)
    float wx = uy * vz - uz * vy;
    float wy = uz * vx - ux * vz;
    // by construction, (wx,wy,wz) must be a unit vector. Measure angle between (wx,wy,wz) and (ax,ay,0).
    // The angle between them should be small if the pair is conversion. This function then returns values close to pi!
    float cosPhiV = wx * ax + wy * ay;
    return TMath::ACos(cosPhiV); //phiv in [0,pi]
  }

  float psipairv0(const array<float, 3>& ppos, const array<float, 3>& pneg, const float bz)
  {
    //Following idea to use opening of colinear pairs in magnetic field from e.g. PHENIX to ID conversions.
    float deltat = TMath::ATan(pneg[2] / (TMath::Sqrt(pneg[0] * pneg[0] + pneg[1] * pneg[1]))) - TMath::ATan(ppos[2] / (TMath::Sqrt(ppos[0] * ppos[0] + ppos[1] * ppos[1]))); //difference of angles of the two daughter tracks with z-axis
    float pEle = RecoDecay::P(pneg);                                                                                                                                          //absolute momentum val
    float pPos = RecoDecay::P(ppos);                                                                                                                                          //absolute momentum val
    float chipair = TMath::ACos(RecoDecay::dotProd(ppos, pneg) / (pEle * pPos));                                                                                              //Angle between daughter tracks
    return TMath::Abs(TMath::ASin(deltat / chipair));                                                                                                                         //psipair in [0,pi/2]
  }

  int processV0(const array<float, 3>& ppos, const array<float, 3>& pneg)
  {
    float alpha = alphav0(ppos, pneg);
    float qt = qtarmv0(ppos, pneg);

    // Gamma cuts
    const float cutAlphaG = 0.4;
    const float cutQTG = 0.03;
    const float cutAlphaG2[2] = {0.4, 0.8};
    const float cutQTG2 = 0.02;

    // K0S cuts
    const float cutQTK0S[2] = {0.1075, 0.215};
    const float cutAPK0S[2] = {0.199, 0.8}; // parameters for curved QT cut

    // Lambda & A-Lambda cuts
    const float cutQTL = 0.03;
    const float cutAlphaL[2] = {0.35, 0.7};
    const float cutAlphaAL[2] = {-0.7, -0.35};
    const float cutAPL[3] = {0.107, -0.69, 0.5}; // parameters fir curved QT cut

    // Check for Gamma candidates
    if (qt < cutQTG) {
      if ((TMath::Abs(alpha) < cutAlphaG))
        return kGamma;
    }
    if (qt < cutQTG2) {
      // additional region - should help high pT gammas
      if ((TMath::Abs(alpha) > cutAlphaG2[0]) && (TMath::Abs(alpha) < cutAlphaG2[1]))
        return kGamma;
    }

    // Check for K0S candidates
    float q = cutAPK0S[0] * TMath::Sqrt(TMath::Abs(1 - alpha * alpha / (cutAPK0S[1] * cutAPK0S[1])));
    if ((qt > cutQTK0S[0]) && (qt < cutQTK0S[1]) && (qt > q)) {
      return kK0S;
    }

    // Check for Lambda candidates
    q = cutAPL[0] * TMath::Sqrt(TMath::Abs(1 - ((alpha + cutAPL[1]) * (alpha + cutAPL[1])) / (cutAPL[2] * cutAPL[2])));
    if ((alpha > cutAlphaL[0]) && (alpha < cutAlphaL[1]) && (qt > cutQTL) && (qt < q)) {
      return kLambda;
    }

    // Check for A-Lambda candidates
    q = cutAPL[0] * TMath::Sqrt(TMath::Abs(1 - ((alpha - cutAPL[1]) * (alpha - cutAPL[1])) / (cutAPL[2] * cutAPL[2])));
    if ((alpha > cutAlphaAL[0]) && (alpha < cutAlphaAL[1]) && (qt > cutQTL) && (qt < q)) {
      return kAntiLambda;
    }

    return kUndef;
  }

  //Basic checks
  HistogramRegistry registry{
    "registry",
    {
      {"hEventCounter", "hEventCounter", {HistType::kTH1F, {{1, 0.0f, 1.0f}}}},
      {"hV0Candidate", "hV0Candidate", {HistType::kTH1F, {{2, 0.0f, 2.0f}}}},
      {"hMassGamma", "hMassGamma", {HistType::kTH1F, {{100, 0.0f, 0.1f}}}},
      {"hMassK0S", "hMassK0S", {HistType::kTH1F, {{100, 0.45, 0.55}}}},
      {"hMassLambda", "hMasLambda", {HistType::kTH1F, {{100, 1.05, 1.15f}}}},
      {"hMassAntiLambda", "hAntiMasLambda", {HistType::kTH1F, {{100, 1.05, 1.15f}}}},

      {"hMassGamma_AP", "hMassGamma_AP", {HistType::kTH1F, {{100, 0.0f, 0.1f}}}},
      {"hMassK0S_AP", "hMassK0S_AP", {HistType::kTH1F, {{100, 0.45, 0.55}}}},
      {"hMassLambda_AP", "hMasLambda_AP", {HistType::kTH1F, {{100, 1.05, 1.15}}}},
      {"hMassAntiLambda_AP", "hAntiMasLambda_AP", {HistType::kTH1F, {{100, 1.05, 1.15}}}},

      {"h2MassGammaR", "h2MassGammaR", {HistType::kTH2F, {{1000, 0.0, 100}, {100, 0.0f, 0.1f}}}},

      {"hV0Pt", "pT", {HistType::kTH1F, {{100, 0.0f, 10}}}},
      {"hV0EtaPhi", "#eta vs. #varphi", {HistType::kTH2F, {{63, 0, 6.3}, {20, -1.0f, 1.0f}}}},

      {"hV0Radius", "hV0Radius", {HistType::kTH1F, {{1000, 0.0f, 100.0f}}}},
      {"hV0CosPA", "hV0CosPA", {HistType::kTH1F, {{1000, 0.95f, 1.0f}}}},
      {"hV0Chi2", "hV0Chi2", {HistType::kTH1F, {{300, 0.0f, 30.0f}}}},
      {"hDCAPosToPV", "hDCAPosToPV", {HistType::kTH1F, {{1000, 0.0f, 10.0f}}}},
      {"hDCANegToPV", "hDCANegToPV", {HistType::kTH1F, {{1000, 0.0f, 10.0f}}}},
      {"hDCAV0Dau", "hDCAV0Dau", {HistType::kTH1F, {{1000, 0.0f, 10.0f}}}},

      {"hGammaCandidate", "hV0Candidate", {HistType::kTH1F, {{101, -0.5, 100.5}}}},

      {"hV0APplot", "hV0APplot", {HistType::kTH2F, {{200, -1.0f, +1.0f}, {250, 0.0f, 0.25f}}}},
      {"hV0APplot_Gamma", "hV0APplot Gamma", {HistType::kTH2F, {{200, -1.0f, +1.0f}, {250, 0.0f, 0.25f}}}},
      {"hV0APplot_K0S", "hV0APplot K0S", {HistType::kTH2F, {{200, -1.0f, +1.0f}, {250, 0.0f, 0.25f}}}},
      {"hV0APplot_Lambda", "hV0APplot Lambda", {HistType::kTH2F, {{200, -1.0f, +1.0f}, {250, 0.0f, 0.25f}}}},
      {"hV0APplot_AntiLambda", "hV0APplot AntiLambda", {HistType::kTH2F, {{200, -1.0f, +1.0f}, {250, 0.0f, 0.25f}}}},

      {"hV0PhiV_Gamma_2D", "hV0PhiV Gamma", {HistType::kTH2F, {{100, 0, TMath::Pi()}, {100, 0.0f, 0.1f}}}},
      {"hV0PhiV", "hV0PhiV", {HistType::kTH1F, {{100, 0, TMath::Pi()}}}},
      {"hV0PhiV_Gamma", "hV0PhiV Gamma", {HistType::kTH1F, {{100, 0, TMath::Pi()}}}},
      {"hV0PhiV_K0S", "hV0PhiV K0S", {HistType::kTH1F, {{100, 0, TMath::Pi()}}}},
      {"hV0PhiV_Lambda", "hV0PhiV Lambda", {HistType::kTH1F, {{100, 0, TMath::Pi()}}}},
      {"hV0PhiV_AntiLambda", "hV0PhiV AntiLambda", {HistType::kTH1F, {{100, 0, TMath::Pi()}}}},

      {"hV0Psi_Gamma_2D", "hV0Psi Gamma", {HistType::kTH2F, {{100, 0, TMath::PiOver2()}, {100, 0.0f, 0.1f}}}},
      {"hV0Psi", "hV0Psi", {HistType::kTH1F, {{100, 0, TMath::PiOver2()}}}},
      {"hV0Psi_Gamma", "hV0Psi Gamma", {HistType::kTH1F, {{100, 0, TMath::PiOver2()}}}},
      {"hV0Psi_K0S", "hV0Psi K0S", {HistType::kTH1F, {{100, 0, TMath::PiOver2()}}}},
      {"hV0Psi_Lambda", "hV0Psi Lambda", {HistType::kTH1F, {{100, 0, TMath::PiOver2()}}}},
      {"hV0Psi_AntiLambda", "hV0Psi AntiLambda", {HistType::kTH1F, {{100, 0, TMath::PiOver2()}}}},
      {"hV0PsiPhiV_Gamma", "hV0Psi PhiV Gamma", {HistType::kTH2F, {{100, 0, TMath::PiOver2()}, {100, 0, TMath::Pi()}}}},

      {"h2TPCdEdx_Pin_Pos", "TPC dEdx vs. p_{in}", {HistType::kTH2F, {{1000, 0.0, 10}, {200, 0.0, 200.}}}},
      {"h2TPCdEdx_Pin_Neg", "TPC dEdx vs. p_{in}", {HistType::kTH2F, {{1000, 0.0, 10}, {200, 0.0, 200.}}}},
      {"h2TPCdEdx_Pin_El_plus", "TPC dEdx vs. p_{in}", {HistType::kTH2F, {{1000, 0.0, 10}, {200, 0.0, 200.}}}},
      {"h2TPCdEdx_Pin_El_minus", "TPC dEdx vs. p_{in}", {HistType::kTH2F, {{1000, 0.0, 10}, {200, 0.0, 200.}}}},
      {"h2TPCdEdx_Pin_Pi_plus", "TPC dEdx vs. p_{in}", {HistType::kTH2F, {{1000, 0.0, 10}, {200, 0.0, 200.}}}},
      {"h2TPCdEdx_Pin_Pi_minus", "TPC dEdx vs. p_{in}", {HistType::kTH2F, {{1000, 0.0, 10}, {200, 0.0, 200.}}}},
      {"h2TPCdEdx_Pin_Ka_plus", "TPC dEdx vs. p_{in}", {HistType::kTH2F, {{1000, 0.0, 10}, {200, 0.0, 200.}}}},
      {"h2TPCdEdx_Pin_Ka_minus", "TPC dEdx vs. p_{in}", {HistType::kTH2F, {{1000, 0.0, 10}, {200, 0.0, 200.}}}},
      {"h2TPCdEdx_Pin_Pr_plus", "TPC dEdx vs. p_{in}", {HistType::kTH2F, {{1000, 0.0, 10}, {200, 0.0, 200.}}}},
      {"h2TPCdEdx_Pin_Pr_minus", "TPC dEdx vs. p_{in}", {HistType::kTH2F, {{1000, 0.0, 10}, {200, 0.0, 200.}}}},

      {"h2TPCnSigma_Pin_El_plus", "TPC n#sigma_{e} vs. p_{in}", {HistType::kTH2F, {{1000, 0.0, 10}, {100, -5, +5}}}},
      {"h2TPCnSigma_Pin_El_minus", "TPC n#sigma_{e} vs. p_{in}", {HistType::kTH2F, {{1000, 0.0, 10}, {100, -5, +5}}}},
      {"h2TPCnSigma_Pin_Pi_plus", "TPC n#sigma_{#pi} vs. p_{in}", {HistType::kTH2F, {{1000, 0.0, 10}, {100, -5, +5}}}},
      {"h2TPCnSigma_Pin_Pi_minus", "TPC n#sigma_{#pi} vs. p_{in}", {HistType::kTH2F, {{1000, 0.0, 10}, {100, -5, +5}}}},
      {"h2TPCnSigma_Pin_Ka_plus", "TPC n#sigma_{K} vs. p_{in}", {HistType::kTH2F, {{1000, 0.0, 10}, {100, -5, +5}}}},
      {"h2TPCnSigma_Pin_Ka_minus", "TPC n#sigma_{K} vs. p_{in}", {HistType::kTH2F, {{1000, 0.0, 10}, {100, -5, +5}}}},
      {"h2TPCnSigma_Pin_Pr_plus", "TPC n#sigma_{p} vs. p_{in}", {HistType::kTH2F, {{1000, 0.0, 10}, {100, -5, +5}}}},
      {"h2TPCnSigma_Pin_Pr_minus", "TPC n#sigma_{p} vs. p_{in}", {HistType::kTH2F, {{1000, 0.0, 10}, {100, -5, +5}}}},

      {"h2TOFbeta_Pin_Pos", "TOF #beta vs. p_{in}", {HistType::kTH2F, {{1000, 0.0, 10}, {120, 0.0, 1.2}}}},
      {"h2TOFbeta_Pin_Neg", "TOF #beta vs. p_{in}", {HistType::kTH2F, {{1000, 0.0, 10}, {120, 0.0, 1.2}}}},
      {"h2TOFbeta_Pin_El_plus", "TOF #beta vs. p_{in}", {HistType::kTH2F, {{1000, 0.0, 10}, {120, 0.0, 1.2}}}},
      {"h2TOFbeta_Pin_El_minus", "TOF #beta vs. p_{in}", {HistType::kTH2F, {{1000, 0.0, 10}, {120, 0.0, 1.2}}}},
      {"h2TOFbeta_Pin_Pi_plus", "TOF #beta vs. p_{in}", {HistType::kTH2F, {{1000, 0.0, 10}, {120, 0.0, 1.2}}}},
      {"h2TOFbeta_Pin_Pi_minus", "TOF #beta vs. p_{in}", {HistType::kTH2F, {{1000, 0.0, 10}, {120, 0.0, 1.2}}}},
      {"h2TOFbeta_Pin_Ka_plus", "TOF #beta vs. p_{in}", {HistType::kTH2F, {{1000, 0.0, 10}, {120, 0.0, 1.2}}}},
      {"h2TOFbeta_Pin_Ka_minus", "TOF #beta vs. p_{in}", {HistType::kTH2F, {{1000, 0.0, 10}, {120, 0.0, 1.2}}}},
      {"h2TOFbeta_Pin_Pr_plus", "TOF #beta vs. p_{in}", {HistType::kTH2F, {{1000, 0.0, 10}, {120, 0.0, 1.2}}}},
      {"h2TOFbeta_Pin_Pr_minus", "TOF #beta vs. p_{in}", {HistType::kTH2F, {{1000, 0.0, 10}, {120, 0.0, 1.2}}}},

      {"h2TOFnSigma_Pin_El_plus", "TOF n#sigma_{e} vs. p_{in}", {HistType::kTH2F, {{1000, 0.0, 10}, {100, -5, +5}}}},
      {"h2TOFnSigma_Pin_El_minus", "TOF n#sigma_{e} vs. p_{in}", {HistType::kTH2F, {{1000, 0.0, 10}, {100, -5, +5}}}},
      {"h2TOFnSigma_Pin_Pi_plus", "TOF n#sigma_{#pi} vs. p_{in}", {HistType::kTH2F, {{1000, 0.0, 10}, {100, -5, +5}}}},
      {"h2TOFnSigma_Pin_Pi_minus", "TOF n#sigma_{#pi} vs. p_{in}", {HistType::kTH2F, {{1000, 0.0, 10}, {100, -5, +5}}}},
      {"h2TOFnSigma_Pin_Ka_plus", "TOF n#sigma_{K} vs. p_{in}", {HistType::kTH2F, {{1000, 0.0, 10}, {100, -5, +5}}}},
      {"h2TOFnSigma_Pin_Ka_minus", "TOF n#sigma_{K} vs. p_{in}", {HistType::kTH2F, {{1000, 0.0, 10}, {100, -5, +5}}}},
      {"h2TOFnSigma_Pin_Pr_plus", "TOF n#sigma_{p} vs. p_{in}", {HistType::kTH2F, {{1000, 0.0, 10}, {100, -5, +5}}}},
      {"h2TOFnSigma_Pin_Pr_minus", "TOF n#sigma_{p} vs. p_{in}", {HistType::kTH2F, {{1000, 0.0, 10}, {100, -5, +5}}}},

      {"h2MggPt", "M_{#gamma#gamma} vs. p_{T}", {HistType::kTH2F, {{400, 0.0, 0.8}, {100, 0.0, 10.}}}},
    },
  };

  //Configurables
  Configurable<double> d_bz{"d_bz", -5.0, "bz field"};
  Configurable<double> v0cospa{"v0cospa", 0.998, "V0 CosPA"}; //double -> N.B. dcos(x)/dx = 0 at x=0)
  Configurable<float> dcav0dau{"dcav0dau", 0.3, "DCA V0 Daughters"};
  Configurable<float> v0Rmin{"v0Rmin", 3.0, "v0Rmin"};
  Configurable<float> v0Rmax{"v0Rmax", 60.0, "v0Rmax"};
  Configurable<float> dcamin{"dcamin", 0.0, "dcamin"};
  Configurable<int> mincrossedrows{"mincrossedrows", 70, "min crossed rows"};
  Configurable<float> maxchi2tpc{"maxchi2tpc", 4.0, "max chi2/NclsTPC"};

  void process(aod::Collision const& collision, aod::V0s const& V0s, FullTracksExt const& tracks)
  {

    //printf("begining of process\n");
    registry.fill(HIST("hEventCounter"), 0.5);
    std::array<float, 3> pVtx = {collision.posX(), collision.posY(), collision.posZ()};

    //Define o2 fitter, 2-prong
    o2::vertexing::DCAFitterN<2> fitter;
    fitter.setBz(d_bz);
    fitter.setPropagateToPCA(true);
    fitter.setMaxR(200.);
    fitter.setMinParamChange(1e-3);
    fitter.setMinRelChi2Change(0.9);
    fitter.setMaxDZIni(1e9);
    fitter.setMaxChi2(1e9);
    fitter.setUseAbsDCA(true); // use d_UseAbsDCA once we want to use the weighted DCA

    //printf("before entering V0 loop\n");

    int Ngamma = 0;
    for (auto& V0 : V0s) {
      if (!(V0.posTrack_as<FullTracksExt>().trackType() & o2::aod::track::TPCrefit)) {
        continue;
      }
      if (!(V0.negTrack_as<FullTracksExt>().trackType() & o2::aod::track::TPCrefit)) {
        continue;
      }

      if (V0.posTrack_as<FullTracksExt>().tpcNClsCrossedRows() < mincrossedrows) {
        continue;
      }
      if (V0.negTrack_as<FullTracksExt>().tpcNClsCrossedRows() < mincrossedrows) {
        continue;
      }

      if (V0.posTrack_as<FullTracksExt>().tpcChi2NCl() > maxchi2tpc) {
        continue;
      }
      if (V0.negTrack_as<FullTracksExt>().tpcChi2NCl() > maxchi2tpc) {
        continue;
      }

      if (fabs(V0.posTrack_as<FullTracksExt>().dcaXY()) < dcamin) {
        continue;
      }
      if (fabs(V0.negTrack_as<FullTracksExt>().dcaXY()) < dcamin) {
        continue;
      }

      if (V0.posTrack_as<FullTracksExt>().sign() * V0.negTrack_as<FullTracksExt>().sign() > 0) { //reject same sign pair
        continue;
      }

      if (V0.posTrack_as<FullTracksExt>().collisionId() != V0.negTrack_as<FullTracksExt>().collisionId()) {
        continue;
      }

      //printf("bcId = %d , collisionId = %d\n",collision.bcId(), V0.negTrack_as<FullTracksExt>().collisionId());

      registry.fill(HIST("hV0Candidate"), 0.5);

      std::array<float, 3> pos = {0.};
      std::array<float, 3> pvec0 = {0.};
      std::array<float, 3> pvec1 = {0.};

      int cpos = V0.posTrack_as<FullTracksExt>().sign();
      int cneg = V0.negTrack_as<FullTracksExt>().sign();

      auto pTrack = getTrackParCov(V0.posTrack_as<FullTracksExt>());
      auto nTrack = getTrackParCov(V0.negTrack_as<FullTracksExt>());

      int nCand = fitter.process(pTrack, nTrack);
      if (nCand != 0) {
        fitter.propagateTracksToVertex();
        const auto& vtx = fitter.getPCACandidate();
        for (int i = 0; i < 3; i++) {
          pos[i] = vtx[i];
        }
        fitter.getTrack(0).getPxPyPzGlo(pvec0); //positive
        fitter.getTrack(1).getPxPyPzGlo(pvec1); //negative
      } else {
        continue;
      }

      auto px = pvec0[0] + pvec1[0];
      auto py = pvec0[1] + pvec1[1];
      auto pz = pvec0[2] + pvec1[2];
      auto pt = RecoDecay::sqrtSumOfSquares(pvec0[0] + pvec1[0], pvec0[1] + pvec1[1]);
      auto eta = RecoDecay::Eta(array{px, py, pz});
      auto phi = RecoDecay::Phi(px, py);

      //Apply selections so a skimmed table is created only
      auto V0dca = fitter.getChi2AtPCACandidate(); //distance between 2 legs.
      //auto V0CosinePA = RecoDecay::CPA(array{collision.posX(), collision.posY(), collision.posZ()}, array{pos[0], pos[1], pos[2]}, array{pvec0[0] + pvec1[0], pvec0[1] + pvec1[1], pvec0[2] + pvec1[2]});
      auto V0CosinePA = RecoDecay::CPA(pVtx, array{pos[0], pos[1], pos[2]}, array{px, py, pz});
      auto V0radius = RecoDecay::sqrtSumOfSquares(pos[0], pos[1]);

      registry.fill(HIST("hV0Pt"), pt);
      registry.fill(HIST("hV0EtaPhi"), phi, eta);

      registry.fill(HIST("hV0Radius"), V0radius);
      registry.fill(HIST("hV0CosPA"), V0CosinePA);
      registry.fill(HIST("hV0Chi2"), V0dca);

      if (V0dca > dcav0dau) {
        continue;
      }

      if (V0CosinePA < v0cospa) {
        continue;
      }

      if (V0radius < v0Rmin || v0Rmax < V0radius) {
        continue;
      }

      registry.fill(HIST("h2TPCdEdx_Pin_Neg"), V0.negTrack_as<FullTracksExt>().tpcInnerParam(), V0.negTrack_as<FullTracksExt>().tpcSignal());
      registry.fill(HIST("h2TPCdEdx_Pin_Pos"), V0.posTrack_as<FullTracksExt>().tpcInnerParam(), V0.posTrack_as<FullTracksExt>().tpcSignal());
      registry.fill(HIST("h2TOFbeta_Pin_Neg"), V0.negTrack_as<FullTracksExt>().tpcInnerParam(), V0.negTrack_as<FullTracksExt>().beta());
      registry.fill(HIST("h2TOFbeta_Pin_Pos"), V0.posTrack_as<FullTracksExt>().tpcInnerParam(), V0.posTrack_as<FullTracksExt>().beta());

      float alpha = alphav0(pvec0, pvec1);
      float qtarm = qtarmv0(pvec0, pvec1);
      float phiv = phivv0(pvec0, pvec1, cpos, cneg, d_bz);
      float psipair = psipairv0(pvec0, pvec1, d_bz);

      registry.fill(HIST("hV0APplot"), alpha, qtarm);
      registry.fill(HIST("hV0PhiV"), phiv);
      registry.fill(HIST("hV0Psi"), psipair);

      float mGamma = RecoDecay::M(array{pvec0, pvec1}, array{RecoDecay::getMassPDG(kElectron), RecoDecay::getMassPDG(kElectron)});
      float mK0S = RecoDecay::M(array{pvec0, pvec1}, array{RecoDecay::getMassPDG(kPiPlus), RecoDecay::getMassPDG(kPiPlus)});
      float mLambda = RecoDecay::M(array{pvec0, pvec1}, array{RecoDecay::getMassPDG(kProton), RecoDecay::getMassPDG(kPiPlus)});
      float mAntiLambda = RecoDecay::M(array{pvec0, pvec1}, array{RecoDecay::getMassPDG(kPiPlus), RecoDecay::getMassPDG(kProton)});

      registry.fill(HIST("hMassGamma"), mGamma);
      registry.fill(HIST("hMassK0S"), mK0S);
      registry.fill(HIST("hMassLambda"), mLambda);
      registry.fill(HIST("hMassAntiLambda"), mAntiLambda);

      int v0id = processV0(pvec0, pvec1);
      if (v0id < 0) {
        continue;
      }

      if (v0id == kGamma) { //photon conversion
        registry.fill(HIST("hV0APplot_Gamma"), alpha, qtarm);
        registry.fill(HIST("hV0PhiV_Gamma"), phiv);
        registry.fill(HIST("hV0PhiV_Gamma_2D"), phiv, mGamma);
        registry.fill(HIST("hV0Psi_Gamma"), psipair);
        registry.fill(HIST("hV0Psi_Gamma_2D"), psipair, mGamma);
        registry.fill(HIST("hV0PsiPhiV_Gamma"), psipair, phiv);
        registry.fill(HIST("h2MassGammaR"), V0radius, mGamma);

        if ((70 < V0.posTrack_as<FullTracksExt>().tpcSignal() && V0.posTrack_as<FullTracksExt>().tpcSignal() < 90) && (70 < V0.negTrack_as<FullTracksExt>().tpcSignal() && V0.negTrack_as<FullTracksExt>().tpcSignal() < 90)
            //&& mGamma < 0.01
        ) {
          v0Gamma(V0.negTrack_as<FullTracksExt>().collisionId(), pt, eta, phi, mGamma);
          Ngamma++;
        }

      } else if (v0id == kK0S) { //K0S-> pi pi
        registry.fill(HIST("hV0APplot_K0S"), alpha, qtarm);
        registry.fill(HIST("hV0PhiV_K0S"), phiv);
        registry.fill(HIST("hV0Psi_K0S"), psipair);

      } else if (v0id == kLambda) { //L->p + pi-
        registry.fill(HIST("hV0APplot_Lambda"), alpha, qtarm);
        registry.fill(HIST("hV0PhiV_Lambda"), phiv);
        registry.fill(HIST("hV0Psi_Lambda"), psipair);

      } else if (v0id == kAntiLambda) { //Lbar -> pbar + pi+
        registry.fill(HIST("hV0APplot_AntiLambda"), alpha, qtarm);
        registry.fill(HIST("hV0PhiV_AntiLambda"), phiv);
        registry.fill(HIST("hV0Psi_AntiLambda"), psipair);
      }

      if (v0id == kGamma && mGamma < 0.01 && TMath::Abs(V0.posTrack_as<FullTracksExt>().tpcNSigmaEl()) < 5 && TMath::Abs(V0.negTrack_as<FullTracksExt>().tpcNSigmaEl()) < 5) { //photon conversion
        registry.fill(HIST("h2TPCdEdx_Pin_El_minus"), V0.negTrack_as<FullTracksExt>().tpcInnerParam(), V0.negTrack_as<FullTracksExt>().tpcSignal());
        registry.fill(HIST("h2TPCdEdx_Pin_El_plus"), V0.posTrack_as<FullTracksExt>().tpcInnerParam(), V0.posTrack_as<FullTracksExt>().tpcSignal());
        registry.fill(HIST("h2TOFbeta_Pin_El_minus"), V0.negTrack_as<FullTracksExt>().tpcInnerParam(), V0.negTrack_as<FullTracksExt>().beta());
        registry.fill(HIST("h2TOFbeta_Pin_El_plus"), V0.posTrack_as<FullTracksExt>().tpcInnerParam(), V0.posTrack_as<FullTracksExt>().beta());

        registry.fill(HIST("h2TPCnSigma_Pin_El_minus"), V0.negTrack_as<FullTracksExt>().tpcInnerParam(), V0.negTrack_as<FullTracksExt>().tpcNSigmaEl());
        registry.fill(HIST("h2TPCnSigma_Pin_El_plus"), V0.posTrack_as<FullTracksExt>().tpcInnerParam(), V0.posTrack_as<FullTracksExt>().tpcNSigmaEl());
        registry.fill(HIST("h2TOFnSigma_Pin_El_minus"), V0.negTrack_as<FullTracksExt>().tpcInnerParam(), V0.negTrack_as<FullTracksExt>().tofNSigmaEl());
        registry.fill(HIST("h2TOFnSigma_Pin_El_plus"), V0.posTrack_as<FullTracksExt>().tpcInnerParam(), V0.posTrack_as<FullTracksExt>().tofNSigmaEl());

      } else if (v0id == kK0S && (0.49 < mK0S && mK0S < 0.51) && TMath::Abs(V0.posTrack_as<FullTracksExt>().tpcNSigmaPi()) < 5 && TMath::Abs(V0.negTrack_as<FullTracksExt>().tpcNSigmaPi()) < 5) { //K0S-> pi pi
        registry.fill(HIST("h2TPCdEdx_Pin_Pi_minus"), V0.negTrack_as<FullTracksExt>().tpcInnerParam(), V0.negTrack_as<FullTracksExt>().tpcSignal());
        registry.fill(HIST("h2TPCdEdx_Pin_Pi_plus"), V0.posTrack_as<FullTracksExt>().tpcInnerParam(), V0.posTrack_as<FullTracksExt>().tpcSignal());
        registry.fill(HIST("h2TOFbeta_Pin_Pi_minus"), V0.negTrack_as<FullTracksExt>().tpcInnerParam(), V0.negTrack_as<FullTracksExt>().beta());
        registry.fill(HIST("h2TOFbeta_Pin_Pi_plus"), V0.posTrack_as<FullTracksExt>().tpcInnerParam(), V0.posTrack_as<FullTracksExt>().beta());

        registry.fill(HIST("h2TPCnSigma_Pin_Pi_minus"), V0.negTrack_as<FullTracksExt>().tpcInnerParam(), V0.negTrack_as<FullTracksExt>().tpcNSigmaPi());
        registry.fill(HIST("h2TPCnSigma_Pin_Pi_plus"), V0.posTrack_as<FullTracksExt>().tpcInnerParam(), V0.posTrack_as<FullTracksExt>().tpcNSigmaPi());
        registry.fill(HIST("h2TOFnSigma_Pin_Pi_minus"), V0.negTrack_as<FullTracksExt>().tpcInnerParam(), V0.negTrack_as<FullTracksExt>().tofNSigmaPi());
        registry.fill(HIST("h2TOFnSigma_Pin_Pi_plus"), V0.posTrack_as<FullTracksExt>().tpcInnerParam(), V0.posTrack_as<FullTracksExt>().tofNSigmaPi());

      } else if (v0id == kLambda && (1.112 < mLambda && mLambda < 1.120)) { //L->p + pi-

        if (cpos > 0 && cneg < 0 && TMath::Abs(V0.posTrack_as<FullTracksExt>().tpcNSigmaPr()) < 5 && TMath::Abs(V0.negTrack_as<FullTracksExt>().tpcNSigmaPi()) < 5) { //Lambda
          registry.fill(HIST("h2TPCdEdx_Pin_Pi_minus"), V0.negTrack_as<FullTracksExt>().tpcInnerParam(), V0.negTrack_as<FullTracksExt>().tpcSignal());
          registry.fill(HIST("h2TPCdEdx_Pin_Pr_plus"), V0.posTrack_as<FullTracksExt>().tpcInnerParam(), V0.posTrack_as<FullTracksExt>().tpcSignal());
          registry.fill(HIST("h2TOFbeta_Pin_Pi_minus"), V0.negTrack_as<FullTracksExt>().tpcInnerParam(), V0.negTrack_as<FullTracksExt>().beta());
          registry.fill(HIST("h2TOFbeta_Pin_Pr_plus"), V0.posTrack_as<FullTracksExt>().tpcInnerParam(), V0.posTrack_as<FullTracksExt>().beta());

          registry.fill(HIST("h2TPCnSigma_Pin_Pi_minus"), V0.negTrack_as<FullTracksExt>().tpcInnerParam(), V0.negTrack_as<FullTracksExt>().tpcNSigmaPi());
          registry.fill(HIST("h2TPCnSigma_Pin_Pr_plus"), V0.posTrack_as<FullTracksExt>().tpcInnerParam(), V0.posTrack_as<FullTracksExt>().tpcNSigmaPr());
          registry.fill(HIST("h2TOFnSigma_Pin_Pi_minus"), V0.negTrack_as<FullTracksExt>().tpcInnerParam(), V0.negTrack_as<FullTracksExt>().tofNSigmaPi());
          registry.fill(HIST("h2TOFnSigma_Pin_Pr_plus"), V0.posTrack_as<FullTracksExt>().tpcInnerParam(), V0.posTrack_as<FullTracksExt>().tofNSigmaPr());

        } else if (cpos < 0 && cneg > 0 && TMath::Abs(V0.posTrack_as<FullTracksExt>().tpcNSigmaPi()) < 5 && TMath::Abs(V0.negTrack_as<FullTracksExt>().tpcNSigmaPr()) < 5) { //AntiLambda
          registry.fill(HIST("h2TPCdEdx_Pin_Pr_minus"), V0.negTrack_as<FullTracksExt>().tpcInnerParam(), V0.negTrack_as<FullTracksExt>().tpcSignal());
          registry.fill(HIST("h2TPCdEdx_Pin_Pi_plus"), V0.posTrack_as<FullTracksExt>().tpcInnerParam(), V0.posTrack_as<FullTracksExt>().tpcSignal());
          registry.fill(HIST("h2TOFbeta_Pin_Pr_minus"), V0.negTrack_as<FullTracksExt>().tpcInnerParam(), V0.negTrack_as<FullTracksExt>().beta());
          registry.fill(HIST("h2TOFbeta_Pin_Pi_plus"), V0.posTrack_as<FullTracksExt>().tpcInnerParam(), V0.posTrack_as<FullTracksExt>().beta());

          registry.fill(HIST("h2TPCnSigma_Pin_Pr_minus"), V0.negTrack_as<FullTracksExt>().tpcInnerParam(), V0.negTrack_as<FullTracksExt>().tpcNSigmaPr());
          registry.fill(HIST("h2TPCnSigma_Pin_Pi_plus"), V0.posTrack_as<FullTracksExt>().tpcInnerParam(), V0.posTrack_as<FullTracksExt>().tpcNSigmaPi());
          registry.fill(HIST("h2TOFnSigma_Pin_Pr_minus"), V0.negTrack_as<FullTracksExt>().tpcInnerParam(), V0.negTrack_as<FullTracksExt>().tofNSigmaPr());
          registry.fill(HIST("h2TOFnSigma_Pin_Pi_plus"), V0.posTrack_as<FullTracksExt>().tpcInnerParam(), V0.posTrack_as<FullTracksExt>().tofNSigmaPi());
        }

      } else if (v0id == kAntiLambda && (1.112 < mAntiLambda && mAntiLambda < 1.120)) {                                                                               //Lbar -> pbar + pi+
        if (cpos < 0 && cneg > 0 && TMath::Abs(V0.posTrack_as<FullTracksExt>().tpcNSigmaPr()) < 5 && TMath::Abs(V0.negTrack_as<FullTracksExt>().tpcNSigmaPi()) < 5) { //AntiLambda
          registry.fill(HIST("h2TPCdEdx_Pin_Pr_minus"), V0.negTrack_as<FullTracksExt>().tpcInnerParam(), V0.negTrack_as<FullTracksExt>().tpcSignal());
          registry.fill(HIST("h2TPCdEdx_Pin_Pi_plus"), V0.posTrack_as<FullTracksExt>().tpcInnerParam(), V0.posTrack_as<FullTracksExt>().tpcSignal());
          registry.fill(HIST("h2TOFbeta_Pin_Pr_minus"), V0.negTrack_as<FullTracksExt>().tpcInnerParam(), V0.negTrack_as<FullTracksExt>().beta());
          registry.fill(HIST("h2TOFbeta_Pin_Pi_plus"), V0.posTrack_as<FullTracksExt>().tpcInnerParam(), V0.posTrack_as<FullTracksExt>().beta());

          registry.fill(HIST("h2TPCnSigma_Pin_Pr_minus"), V0.negTrack_as<FullTracksExt>().tpcInnerParam(), V0.negTrack_as<FullTracksExt>().tpcNSigmaPr());
          registry.fill(HIST("h2TPCnSigma_Pin_Pi_plus"), V0.posTrack_as<FullTracksExt>().tpcInnerParam(), V0.posTrack_as<FullTracksExt>().tpcNSigmaPi());
          registry.fill(HIST("h2TOFnSigma_Pin_Pr_minus"), V0.negTrack_as<FullTracksExt>().tpcInnerParam(), V0.negTrack_as<FullTracksExt>().tofNSigmaPr());
          registry.fill(HIST("h2TOFnSigma_Pin_Pi_plus"), V0.posTrack_as<FullTracksExt>().tpcInnerParam(), V0.posTrack_as<FullTracksExt>().tofNSigmaPi());
        } else if (cpos > 0 && cneg < 0 && TMath::Abs(V0.posTrack_as<FullTracksExt>().tpcNSigmaPi()) < 5 && TMath::Abs(V0.negTrack_as<FullTracksExt>().tpcNSigmaPr()) < 5) { //Lambda
          registry.fill(HIST("h2TPCdEdx_Pin_Pr_minus"), V0.negTrack_as<FullTracksExt>().tpcInnerParam(), V0.negTrack_as<FullTracksExt>().tpcSignal());
          registry.fill(HIST("h2TPCdEdx_Pin_Pi_plus"), V0.posTrack_as<FullTracksExt>().tpcInnerParam(), V0.posTrack_as<FullTracksExt>().tpcSignal());
          registry.fill(HIST("h2TOFbeta_Pin_Pr_minus"), V0.negTrack_as<FullTracksExt>().tpcInnerParam(), V0.negTrack_as<FullTracksExt>().beta());
          registry.fill(HIST("h2TOFbeta_Pin_Pi_plus"), V0.posTrack_as<FullTracksExt>().tpcInnerParam(), V0.posTrack_as<FullTracksExt>().beta());

          registry.fill(HIST("h2TPCnSigma_Pin_Pr_minus"), V0.negTrack_as<FullTracksExt>().tpcInnerParam(), V0.negTrack_as<FullTracksExt>().tpcNSigmaPr());
          registry.fill(HIST("h2TPCnSigma_Pin_Pi_plus"), V0.posTrack_as<FullTracksExt>().tpcInnerParam(), V0.posTrack_as<FullTracksExt>().tpcNSigmaPi());
          registry.fill(HIST("h2TOFnSigma_Pin_Pr_minus"), V0.negTrack_as<FullTracksExt>().tpcInnerParam(), V0.negTrack_as<FullTracksExt>().tofNSigmaPr());
          registry.fill(HIST("h2TOFnSigma_Pin_Pi_plus"), V0.posTrack_as<FullTracksExt>().tpcInnerParam(), V0.posTrack_as<FullTracksExt>().tofNSigmaPi());
        }
      }

    } //end of V0 loop
    registry.fill(HIST("hGammaCandidate"), Ngamma);

  } //end of process
};

struct v0gammaQA {

  //Basic checks
  HistogramRegistry registry{
    "registry",
    {
      {"hEventCounter", "hEventCounter", {HistType::kTH1F, {{5, 0.5f, 5.5f}}}},
      {"hV0Pt", "pT", {HistType::kTH1F, {{100, 0.0, 10}}}},
      {"hV0EtaPhi", "#eta vs. #varphi", {HistType::kTH2F, {{63, 0, 6.3}, {20, -1.0f, 1.0f}}}},
      {"h2MggPt", "M_{#gamma#gamma} vs. p_{T}", {HistType::kTH2F, {{400, 0.0, 0.8}, {100, 0.0, 10.}}}},
    },
  };

  void process(soa::Join<aod::Collisions, aod::EvSels>::iterator const& collision, aod::ReducedV0s const& v0Gamma)
  {

    //printf("begining of process\n");
    registry.fill(HIST("hEventCounter"), 1.0); //all

    if (!collision.alias()[kINT7]) {
      return;
    }
    //if (!collision.sel7()) {
    //  return;
    //}

    registry.fill(HIST("hEventCounter"), 2.0); //INT7

    for (auto& g : v0Gamma) {
      registry.fill(HIST("hV0Pt"), g.pt());
      registry.fill(HIST("hV0EtaPhi"), g.phi(), g.eta());
    }

    for (auto& [g1, g2] : combinations(v0Gamma, v0Gamma)) {
      //printf("fill 2 gammas\n");
      ROOT::Math::PtEtaPhiMVector v1(g1.pt(), g1.eta(), g1.phi(), g1.mass());
      ROOT::Math::PtEtaPhiMVector v2(g2.pt(), g2.eta(), g2.phi(), g2.mass());
      ROOT::Math::PtEtaPhiMVector v12 = v1 + v2;
      registry.fill(HIST("h2MggPt"), v12.M(), v12.Pt());
    } //end of combination

  } //end of process
};

WorkflowSpec defineDataProcessing(ConfigContext const& cfgc)
{
  return WorkflowSpec{
    adaptAnalysisTask<v0selector>(cfgc, TaskName{"v0-selector"}),
    adaptAnalysisTask<v0gammaQA>(cfgc, TaskName{"v0-gamma-qa"})};
}
