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
#include <map>
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

namespace v0bits
{
DECLARE_SOA_COLUMN(PIDBit, pidbit, uint8_t); //!
} // namespace v0bits

// basic track information
DECLARE_SOA_TABLE(V0Bits, "AOD", "V0BITS", //!
                  o2::soa::Index<>, v0bits::PIDBit);

// iterators
using V0Bit = V0Bits::iterator;

} // namespace o2::aod

struct v0selector {

  enum { // Reconstructed V0
    kUndef = -1,
    kGamma = 0,
    kK0S = 1,
    kLambda = 2,
    kAntiLambda = 3,
    kOmega = 4
  };

  Produces<aod::ReducedV0s> v0Gamma;
  Produces<aod::V0Bits> v0bits;

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

    //unit vector of pep X pem
    float vx = vpx / RecoDecay::P(array{vpx, vpy, vpz});
    float vy = vpy / RecoDecay::P(array{vpx, vpy, vpz});
    float vz = vpz / RecoDecay::P(array{vpx, vpy, vpz});

    float px = ppos[0] + pneg[0];
    float py = ppos[1] + pneg[1];
    float pz = ppos[2] + pneg[2];

    //unit vector of (pep+pem)
    float ux = px / RecoDecay::P(array{px, py, pz});
    float uy = py / RecoDecay::P(array{px, py, pz});
    float uz = pz / RecoDecay::P(array{px, py, pz});
    float ax = uy / RecoDecay::sqrtSumOfSquares(ux, uy);
    float ay = -ux / RecoDecay::sqrtSumOfSquares(ux, uy);

    //The third axis defined by vector product (ux,uy,uz)X(vx,vy,vz)
    float wx = uy * vz - uz * vy;
    float wy = uz * vx - ux * vz;
    // by construction, (wx,wy,wz) must be a unit vector. Measure angle between (wx,wy,wz) and (ax,ay,0).
    // The angle between them should be small if the pair is conversion. This function then returns values close to pi!
    return TMath::ACos(wx * ax + wy * ay); //phiv in [0,pi] //cosPhiV = wx * ax + wy * ay;
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
      if ((TMath::Abs(alpha) < cutAlphaG)) {
        return kGamma;
      }
    }
    if (qt < cutQTG2) {
      // additional region - should help high pT gammas
      if ((TMath::Abs(alpha) > cutAlphaG2[0]) && (TMath::Abs(alpha) < cutAlphaG2[1])) {
        return kGamma;
      }
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
      {"hMassLambda", "hMassLambda", {HistType::kTH1F, {{100, 1.05, 1.15f}}}},
      {"hMassAntiLambda", "hAntiMassLambda", {HistType::kTH1F, {{100, 1.05, 1.15f}}}},
      {"hV0Pt", "pT", {HistType::kTH1F, {{100, 0.0f, 10}}}},
      {"hV0EtaPhi", "#eta vs. #varphi", {HistType::kTH2F, {{63, 0, 6.3}, {20, -1.0f, 1.0f}}}},
      {"hV0Radius", "hV0Radius", {HistType::kTH1F, {{1000, 0.0f, 100.0f}}}},
      {"hV0CosPA", "hV0CosPA", {HistType::kTH1F, {{1000, 0.95f, 1.0f}}}},
      {"hV0CosPA_Casc", "hV0CosPA_Casc", {HistType::kTH1F, {{1000, 0.95f, 1.0f}}}},
      {"hDCAxyPosToPV", "hDCAxyPosToPV", {HistType::kTH1F, {{200, -10.0f, 10.0f}}}},
      {"hDCAxyNegToPV", "hDCAxyNegToPV", {HistType::kTH1F, {{200, -10.0f, 10.0f}}}},
      {"hDCAV0Dau", "hDCAV0Dau", {HistType::kTH1F, {{1000, 0.0f, 10.0f}}}},
      {"hDCAV0Dau_Casc", "hDCAV0Dau_Casc", {HistType::kTH1F, {{1000, 0.0f, 10.0f}}}},
      {"hGammaCandidate", "hV0Candidate", {HistType::kTH1F, {{101, -0.5, 100.5}}}},
      {"hV0APplot", "hV0APplot", {HistType::kTH2F, {{200, -1.0f, +1.0f}, {250, 0.0f, 0.25f}}}},
      {"hV0PhiV", "hV0PhiV", {HistType::kTH1F, {{100, 0, TMath::Pi()}}}},
      {"hV0Psi", "hV0Psi", {HistType::kTH1F, {{100, 0, TMath::PiOver2()}}}},
      {"hCascCandidate", "hCascCandidate", {HistType::kTH1F, {{10, 0.0, 10.0}}}},
      {"hCascCosPA", "hCascCosPA", {HistType::kTH1F, {{1000, 0.95f, 1.0f}}}},
      {"hDCACascDau", "hDCACascDau", {HistType::kTH1F, {{1000, 0.0f, 10.0f}}}},
      {"hMassLambda_Casc", "hMassLambda", {HistType::kTH1F, {{100, 1.05, 1.15f}}}},
      {"hMassAntiLambda_Casc", "hAntiMassLambda", {HistType::kTH1F, {{100, 1.05, 1.15f}}}},
      {"hMassXiMinus", "hMassXiMinus", {HistType::kTH1F, {{200, 1.25, 1.45}}}},
      {"hMassXiPlus", "hMassXiPlus", {HistType::kTH1F, {{200, 1.25, 1.45}}}},
      {"hMassOmegaMinus", "hMassOmegaMinus", {HistType::kTH1F, {{200, 1.6, 1.8}}}},
      {"hMassOmegaPlus", "hMassOmegaPlus", {HistType::kTH1F, {{200, 1.6, 1.8}}}},
    },
  };

  //Configurables
  Configurable<double> d_bz{"d_bz", -5.0, "bz field"};
  Configurable<double> v0cospa{"v0cospa", 0.998, "V0 CosPA"}; //double -> N.B. dcos(x)/dx = 0 at x=0)
  Configurable<float> dcav0dau{"dcav0dau", 0.3, "DCA V0 Daughters"};
  Configurable<float> v0Rmin{"v0Rmin", 3.0, "v0Rmin"};
  Configurable<float> v0Rmax{"v0Rmax", 60.0, "v0Rmax"};
  Configurable<float> dcamin{"dcamin", 0.0, "dcamin"};
  Configurable<float> dcamax{"dcamax", 1e+10, "dcamax"};
  Configurable<int> mincrossedrows{"mincrossedrows", 70, "min crossed rows"};
  Configurable<float> maxchi2tpc{"maxchi2tpc", 4.0, "max chi2/NclsTPC"};

  void process(aod::Collision const& collision, aod::V0s const& V0s, aod::Cascades const& Cascades, FullTracksExt const& tracks)
  {
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

    std::map<int, uint8_t> pidmap;

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

      if (fabs(V0.posTrack_as<FullTracksExt>().dcaXY()) > dcamax) {
        continue;
      }
      if (fabs(V0.negTrack_as<FullTracksExt>().dcaXY()) > dcamax) {
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

      if (cpos < 0) { //swap charge
        nTrack = getTrackParCov(V0.posTrack_as<FullTracksExt>());
        pTrack = getTrackParCov(V0.negTrack_as<FullTracksExt>());
      }

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

      auto V0dca = fitter.getChi2AtPCACandidate(); //distance between 2 legs.
      auto V0CosinePA = RecoDecay::CPA(pVtx, array{pos[0], pos[1], pos[2]}, array{px, py, pz});
      auto V0radius = RecoDecay::sqrtSumOfSquares(pos[0], pos[1]);

      registry.fill(HIST("hV0Pt"), pt);
      registry.fill(HIST("hV0EtaPhi"), phi, eta);
      registry.fill(HIST("hDCAxyPosToPV"), V0.posTrack_as<FullTracksExt>().dcaXY());
      registry.fill(HIST("hDCAxyNegToPV"), V0.negTrack_as<FullTracksExt>().dcaXY());

      registry.fill(HIST("hV0Radius"), V0radius);
      registry.fill(HIST("hV0CosPA"), V0CosinePA);
      registry.fill(HIST("hDCAV0Dau"), V0dca);

      if (V0dca > dcav0dau) {
        continue;
      }

      if (V0CosinePA < v0cospa) {
        continue;
      }

      if (V0radius < v0Rmin || v0Rmax < V0radius) {
        continue;
      }

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
        if ((70 < V0.posTrack_as<FullTracksExt>().tpcSignal() && V0.posTrack_as<FullTracksExt>().tpcSignal() < 90) && (70 < V0.negTrack_as<FullTracksExt>().tpcSignal() && V0.negTrack_as<FullTracksExt>().tpcSignal() < 90)) {
          v0Gamma(V0.negTrack_as<FullTracksExt>().collisionId(), pt, eta, phi, mGamma);
          Ngamma++;
        }
      }

      if (v0id == kGamma && mGamma < 0.01 && TMath::Abs(V0.posTrack_as<FullTracksExt>().tpcNSigmaEl()) < 5 && TMath::Abs(V0.negTrack_as<FullTracksExt>().tpcNSigmaEl()) < 5) { //photon conversion
        pidmap[V0.posTrackId()] |= (uint8_t(1) << kGamma);
        pidmap[V0.negTrackId()] |= (uint8_t(1) << kGamma);
      } else if (v0id == kK0S && (0.49 < mK0S && mK0S < 0.51) && TMath::Abs(V0.posTrack_as<FullTracksExt>().tpcNSigmaPi()) < 5 && TMath::Abs(V0.negTrack_as<FullTracksExt>().tpcNSigmaPi()) < 5) { //K0S-> pi pi
        pidmap[V0.posTrackId()] |= (uint8_t(1) << kK0S);
        pidmap[V0.negTrackId()] |= (uint8_t(1) << kK0S);
      } else if (v0id == kLambda && (1.112 < mLambda && mLambda < 1.120) && TMath::Abs(V0.posTrack_as<FullTracksExt>().tpcNSigmaPr()) < 5 && TMath::Abs(V0.negTrack_as<FullTracksExt>().tpcNSigmaPi()) < 5) { //L->p + pi-
        pidmap[V0.posTrackId()] |= (uint8_t(1) << kLambda);
        pidmap[V0.negTrackId()] |= (uint8_t(1) << kLambda);
      } else if (v0id == kAntiLambda && (1.112 < mAntiLambda && mAntiLambda < 1.120) && TMath::Abs(V0.posTrack_as<FullTracksExt>().tpcNSigmaPi()) < 5 && TMath::Abs(V0.negTrack_as<FullTracksExt>().tpcNSigmaPr()) < 5) { //Lbar -> pbar + pi+
        pidmap[V0.posTrackId()] |= (uint8_t(1) << kAntiLambda);
        pidmap[V0.negTrackId()] |= (uint8_t(1) << kAntiLambda);
      }

      //printf("posTrackId = %d\n",V0.posTrackId());
      //printf("negTrackId = %d\n",V0.negTrackId());

      o2::vertexing::DCAFitterN<2> fitterCasc;
      fitterCasc.setBz(d_bz);
      fitterCasc.setPropagateToPCA(true);
      fitterCasc.setMaxR(200.);
      fitterCasc.setMinParamChange(1e-3);
      fitterCasc.setMinRelChi2Change(0.9);
      fitterCasc.setMaxDZIni(1e9);
      fitterCasc.setMaxChi2(1e9);
      fitterCasc.setUseAbsDCA(true);

      //cascade loop
      for (auto& casc : Cascades) {
        auto v0 = casc.v0_as<aod::V0s>();
        //hCascCandidate->Fill(0.5);

        if (!(v0.posTrack_as<FullTracksExt>().trackType() & o2::aod::track::TPCrefit)) {
          continue; //TPC refit
        }
        if (!(v0.negTrack_as<FullTracksExt>().trackType() & o2::aod::track::TPCrefit)) {
          continue; //TPC refit
        }
        if (!(casc.bachelor_as<FullTracksExt>().trackType() & o2::aod::track::TPCrefit)) {
          continue; //TPC refit
        }

        if (V0.posTrack_as<FullTracksExt>().sign() * V0.negTrack_as<FullTracksExt>().sign() > 0) { //reject same sign pair
          continue;
        }

        if (v0.posTrack_as<FullTracksExt>().tpcNClsCrossedRows() < mincrossedrows) {
          continue;
        }
        if (v0.negTrack_as<FullTracksExt>().tpcNClsCrossedRows() < mincrossedrows) {
          continue;
        }
        if (casc.bachelor_as<FullTracksExt>().tpcNClsCrossedRows() < mincrossedrows) {
          continue;
        }

        std::array<float, 3> pos = {0.};
        std::array<float, 3> pvecpos = {0.};
        std::array<float, 3> pvecneg = {0.};
        std::array<float, 3> pvecbach = {0.};

        int cpos = casc.v0_as<aod::V0s>().posTrack_as<FullTracksExt>().sign();
        int cneg = casc.v0_as<aod::V0s>().negTrack_as<FullTracksExt>().sign();
        //auto pTrack = getTrackParCov(V0.posTrack_as<FullTracksExt>());
        //auto nTrack = getTrackParCov(V0.negTrack_as<FullTracksExt>());

        auto pTrack = getTrackParCov(casc.v0_as<aod::V0s>().posTrack_as<FullTracksExt>());
        auto nTrack = getTrackParCov(casc.v0_as<aod::V0s>().negTrack_as<FullTracksExt>());
        auto bTrack = getTrackParCov(casc.bachelor_as<FullTracksExt>());

        if (cpos < 0) { //swap charge
          auto pTrack = getTrackParCov(casc.v0_as<aod::V0s>().negTrack_as<FullTracksExt>());
          auto nTrack = getTrackParCov(casc.v0_as<aod::V0s>().posTrack_as<FullTracksExt>());
        }

        int nCand = fitter.process(pTrack, nTrack);
        if (nCand != 0) {
          fitter.propagateTracksToVertex();
        } else {
          continue;
        }
        const auto& v0vtx = fitter.getPCACandidate();
        for (int i = 0; i < 3; i++) {
          pos[i] = v0vtx[i];
        }

        auto V0dca = fitter.getChi2AtPCACandidate(); //distance between 2 legs.
        registry.fill(HIST("hDCAV0Dau_Casc"), V0dca);
        if (V0dca > 0.8) {
          continue;
        }

        //hCascCandidate->Fill(1.5);
        std::array<float, 21> cov0 = {0};
        std::array<float, 21> cov1 = {0};
        std::array<float, 21> covV0 = {0};

        //Covariance matrix calculation
        const int momInd[6] = {9, 13, 14, 18, 19, 20}; // cov matrix elements for momentum component
        fitter.getTrack(0).getPxPyPzGlo(pvecpos);
        fitter.getTrack(1).getPxPyPzGlo(pvecneg);
        fitter.getTrack(0).getCovXYZPxPyPzGlo(cov0);
        fitter.getTrack(1).getCovXYZPxPyPzGlo(cov1);

        for (int i = 0; i < 6; i++) {
          int j = momInd[i];
          covV0[j] = cov0[j] + cov1[j];
        }
        auto covVtxV0 = fitter.calcPCACovMatrix();
        covV0[0] = covVtxV0(0, 0);
        covV0[1] = covVtxV0(1, 0);
        covV0[2] = covVtxV0(1, 1);
        covV0[3] = covVtxV0(2, 0);
        covV0[4] = covVtxV0(2, 1);
        covV0[5] = covVtxV0(2, 2);

        const std::array<float, 3> vertex = {(float)v0vtx[0], (float)v0vtx[1], (float)v0vtx[2]};
        const std::array<float, 3> pvecv0 = {pvecpos[0] + pvecneg[0], pvecpos[1] + pvecneg[1], pvecpos[2] + pvecneg[2]};
        auto V0CosinePA = RecoDecay::CPA(pVtx, array{pos[0], pos[1], pos[2]}, pvecv0);
        registry.fill(HIST("hV0CosPA_Casc"), V0CosinePA);
        if (V0CosinePA < 0.97) {
          continue;
        }

        auto tV0 = o2::track::TrackParCov(vertex, pvecv0, covV0, 0);
        tV0.setQ2Pt(0); //No bending, please
        int nCand2 = fitterCasc.process(tV0, bTrack);
        if (nCand2 != 0) {
          fitterCasc.propagateTracksToVertex();
          //hCascCandidate->Fill(2.5);
          fitterCasc.getTrack(1).getPxPyPzGlo(pvecbach);
        } else {
          continue;
        }

        auto Cascdca = fitterCasc.getChi2AtPCACandidate(); //distance between V0 and bachelor
        registry.fill(HIST("hDCACascDau"), Cascdca);
        if (Cascdca > 0.8) {
          continue;
        }

        const auto& cascvtx = fitterCasc.getPCACandidate();
        const std::array<float, 3> pvecCasc = {pvecpos[0] + pvecneg[0] + pvecbach[0], pvecpos[1] + pvecneg[1] + pvecbach[1], pvecpos[2] + pvecneg[2] + pvecbach[2]};
        auto CascCosinePA = RecoDecay::CPA(pVtx, array{cascvtx[0], cascvtx[1], cascvtx[2]}, pvecbach);
        registry.fill(HIST("hCascCosPA"), CascCosinePA);

        //if(CascCosinePA < 0.995){
        //  continue;
        //}

        float mLambda = RecoDecay::M(array{pvecpos, pvecneg}, array{RecoDecay::getMassPDG(kProton), RecoDecay::getMassPDG(kPiPlus)});
        float mAntiLambda = RecoDecay::M(array{pvecpos, pvecneg}, array{RecoDecay::getMassPDG(kPiPlus), RecoDecay::getMassPDG(kProton)});
        float mXi = RecoDecay::M(array{pvecv0, pvecbach}, array{RecoDecay::getMassPDG(kLambda0), RecoDecay::getMassPDG(kPiPlus)});
        float mOmega = RecoDecay::M(array{pvecv0, pvecbach}, array{RecoDecay::getMassPDG(kLambda0), RecoDecay::getMassPDG(kKPlus)});

        //for Lambda->p + pi-
        if (cpos > 0 && cneg < 0 && (1.112 < mLambda && mLambda < 1.120) && TMath::Abs(casc.v0_as<aod::V0s>().posTrack_as<FullTracksExt>().tpcNSigmaPr()) < 3 && TMath::Abs(casc.v0_as<aod::V0s>().negTrack_as<FullTracksExt>().tpcNSigmaPi()) < 3) {
          registry.fill(HIST("hMassLambda_Casc"), mLambda);

          if (casc.bachelor_as<FullTracksExt>().sign() < 0) {

            if (TMath::Abs(casc.bachelor_as<FullTracksExt>().tpcNSigmaPi()) < 5) {
              registry.fill(HIST("hMassXiMinus"), mXi);
            }

            if (TMath::Abs(mXi - 1.321) > 0.006) {
              if (TMath::Abs(casc.bachelor_as<FullTracksExt>().tpcNSigmaKa()) < 5) {
                registry.fill(HIST("hMassOmegaMinus"), mOmega);
                if (TMath::Abs(mOmega - 1.672) < 0.006) {
                  pidmap[casc.bachelorId()] |= (uint8_t(1) << kOmega);
                }
              }
            }
          }
        }

        //for AntiLambda->pbar + pi+
        if (cpos > 0 && cneg < 0 && (1.112 < mAntiLambda && mAntiLambda < 1.120) && TMath::Abs(casc.v0_as<aod::V0s>().posTrack_as<FullTracksExt>().tpcNSigmaPi()) < 3 && TMath::Abs(casc.v0_as<aod::V0s>().negTrack_as<FullTracksExt>().tpcNSigmaPr()) < 3) {
          registry.fill(HIST("hMassAntiLambda_Casc"), mAntiLambda);
          if (casc.bachelor_as<FullTracksExt>().sign() > 0) {
            if (TMath::Abs(casc.bachelor_as<FullTracksExt>().tpcNSigmaPi()) < 5) {
              registry.fill(HIST("hMassXiPlus"), mXi);
            }
            if (TMath::Abs(mXi - 1.321) > 0.006) {
              if (TMath::Abs(casc.bachelor_as<FullTracksExt>().tpcNSigmaKa()) < 5) {
                registry.fill(HIST("hMassOmegaPlus"), mOmega);
                if (TMath::Abs(mOmega - 1.672) < 0.006) {
                  pidmap[casc.bachelorId()] |= (uint8_t(1) << kOmega);
                }
              }
            }
          }
        }
      }

    } //end of V0 loop
    registry.fill(HIST("hGammaCandidate"), Ngamma);

    for (auto& track : tracks) {
      //printf("setting pidmap[%d] = %d\n",track.globalIndex(),pidmap[track.globalIndex()]);
      v0bits(pidmap[track.globalIndex()]);
    } //end of track loop

  } //end of process
};

struct trackPIDQA {

  //Basic checks
  HistogramRegistry registry{
    "registry",
    {
      {"hEventCounter", "hEventCounter", {HistType::kTH1F, {{5, 0.5f, 5.5f}}}},
      {"hTrackPt_all", "pT", {HistType::kTH1F, {{100, 0.0, 10}}}},
      {"hTrackEtaPhi_all", "#eta vs. #varphi", {HistType::kTH2F, {{63, 0, 6.3}, {20, -1.0f, 1.0f}}}},

      {"hTrackPt", "pT", {HistType::kTH1F, {{100, 0.0, 10}}}},
      {"hTrackEtaPhi", "#eta vs. #varphi", {HistType::kTH2F, {{63, 0, 6.3}, {20, -1.0f, 1.0f}}}},

      {"h2TPCdEdx_Pin", "TPC dEdx vs. p_{in}", {HistType::kTH2F, {{1000, 0.0, 10}, {200, 0.0, 200.}}}},
      {"h2TPCdEdx_Pin_El", "TPC dEdx vs. p_{in}", {HistType::kTH2F, {{1000, 0.0, 10}, {200, 0.0, 200.}}}},
      {"h2TPCdEdx_Pin_Pi", "TPC dEdx vs. p_{in}", {HistType::kTH2F, {{1000, 0.0, 10}, {200, 0.0, 200.}}}},
      {"h2TPCdEdx_Pin_Ka", "TPC dEdx vs. p_{in}", {HistType::kTH2F, {{1000, 0.0, 10}, {200, 0.0, 200.}}}},
      {"h2TPCdEdx_Pin_Pr", "TPC dEdx vs. p_{in}", {HistType::kTH2F, {{1000, 0.0, 10}, {200, 0.0, 200.}}}},

      {"h2TPCnSigma_Pin_El", "TPC n#sigma_{e} vs. p_{in}", {HistType::kTH2F, {{1000, 0.0, 10}, {100, -5, +5}}}},
      {"h2TPCnSigma_Pin_Pi", "TPC n#sigma_{#pi} vs. p_{in}", {HistType::kTH2F, {{1000, 0.0, 10}, {100, -5, +5}}}},
      {"h2TPCnSigma_Pin_Ka", "TPC n#sigma_{K} vs. p_{in}", {HistType::kTH2F, {{1000, 0.0, 10}, {100, -5, +5}}}},
      {"h2TPCnSigma_Pin_Pr", "TPC n#sigma_{p} vs. p_{in}", {HistType::kTH2F, {{1000, 0.0, 10}, {100, -5, +5}}}},

      {"h2TOFbeta_Pin", "TOF #beta vs. p_{in}", {HistType::kTH2F, {{1000, 0.0, 10}, {120, 0.0, 1.2}}}},
      {"h2TOFbeta_Pin_El", "TOF #beta vs. p_{in}", {HistType::kTH2F, {{1000, 0.0, 10}, {120, 0.0, 1.2}}}},
      {"h2TOFbeta_Pin_Pi", "TOF #beta vs. p_{in}", {HistType::kTH2F, {{1000, 0.0, 10}, {120, 0.0, 1.2}}}},
      {"h2TOFbeta_Pin_Ka", "TOF #beta vs. p_{in}", {HistType::kTH2F, {{1000, 0.0, 10}, {120, 0.0, 1.2}}}},
      {"h2TOFbeta_Pin_Pr", "TOF #beta vs. p_{in}", {HistType::kTH2F, {{1000, 0.0, 10}, {120, 0.0, 1.2}}}},

      {"h2TOFnSigma_Pin_El", "TOF n#sigma_{e} vs. p_{in}", {HistType::kTH2F, {{1000, 0.0, 10}, {100, -5, +5}}}},
      {"h2TOFnSigma_Pin_Pi", "TOF n#sigma_{#pi} vs. p_{in}", {HistType::kTH2F, {{1000, 0.0, 10}, {100, -5, +5}}}},
      {"h2TOFnSigma_Pin_Ka", "TOF n#sigma_{K} vs. p_{in}", {HistType::kTH2F, {{1000, 0.0, 10}, {100, -5, +5}}}},
      {"h2TOFnSigma_Pin_Pr", "TOF n#sigma_{p} vs. p_{in}", {HistType::kTH2F, {{1000, 0.0, 10}, {100, -5, +5}}}},

    },
  };

  void process(soa::Join<aod::Collisions, aod::EvSels>::iterator const& collision, soa::Join<aod::V0Bits, FullTracksExt> const& tracks)
  {
    //printf("begining of process\n");
    registry.fill(HIST("hEventCounter"), 1.0); //all

    if (!collision.alias()[kINT7]) {
      return;
    }

    registry.fill(HIST("hEventCounter"), 2.0); //INT7

    for (auto& track : tracks) {
      registry.fill(HIST("hTrackPt_all"), track.pt());
      registry.fill(HIST("hTrackEtaPhi_all"), track.phi(), track.eta());
      if (track.pidbit() > 0) {
        registry.fill(HIST("hTrackPt"), track.pt());
        registry.fill(HIST("hTrackEtaPhi"), track.phi(), track.eta());
        registry.fill(HIST("h2TPCdEdx_Pin"), track.tpcInnerParam(), track.tpcSignal());
        registry.fill(HIST("h2TOFbeta_Pin"), track.tpcInnerParam(), track.beta());
      }

      if (bool(track.pidbit() & (1 << v0selector::kGamma))) {
        registry.fill(HIST("h2TPCdEdx_Pin_El"), track.tpcInnerParam(), track.tpcSignal());
        registry.fill(HIST("h2TOFbeta_Pin_El"), track.tpcInnerParam(), track.beta());
        registry.fill(HIST("h2TPCnSigma_Pin_El"), track.tpcInnerParam(), track.tpcNSigmaEl());
        registry.fill(HIST("h2TOFnSigma_Pin_El"), track.tpcInnerParam(), track.tofNSigmaEl());
      }
      if (bool(track.pidbit() & (1 << v0selector::kK0S))) {
        registry.fill(HIST("h2TPCdEdx_Pin_Pi"), track.tpcInnerParam(), track.tpcSignal());
        registry.fill(HIST("h2TOFbeta_Pin_Pi"), track.tpcInnerParam(), track.beta());
        registry.fill(HIST("h2TPCnSigma_Pin_Pi"), track.tpcInnerParam(), track.tpcNSigmaPi());
        registry.fill(HIST("h2TOFnSigma_Pin_Pi"), track.tpcInnerParam(), track.tofNSigmaPi());
      }
      if (bool(track.pidbit() & (1 << v0selector::kLambda))) {
        if (track.sign() > 0) {
          registry.fill(HIST("h2TPCdEdx_Pin_Pr"), track.tpcInnerParam(), track.tpcSignal());
          registry.fill(HIST("h2TOFbeta_Pin_Pr"), track.tpcInnerParam(), track.beta());
          registry.fill(HIST("h2TPCnSigma_Pin_Pr"), track.tpcInnerParam(), track.tpcNSigmaPr());
          registry.fill(HIST("h2TOFnSigma_Pin_Pr"), track.tpcInnerParam(), track.tofNSigmaPr());
        } else {
          registry.fill(HIST("h2TPCdEdx_Pin_Pi"), track.tpcInnerParam(), track.tpcSignal());
          registry.fill(HIST("h2TOFbeta_Pin_Pi"), track.tpcInnerParam(), track.beta());
          registry.fill(HIST("h2TPCnSigma_Pin_Pi"), track.tpcInnerParam(), track.tpcNSigmaPi());
          registry.fill(HIST("h2TOFnSigma_Pin_Pi"), track.tpcInnerParam(), track.tofNSigmaPi());
        }
      }
      if (bool(track.pidbit() & (1 << v0selector::kAntiLambda))) {
        if (track.sign() > 0) {
          registry.fill(HIST("h2TPCdEdx_Pin_Pi"), track.tpcInnerParam(), track.tpcSignal());
          registry.fill(HIST("h2TOFbeta_Pin_Pi"), track.tpcInnerParam(), track.beta());
          registry.fill(HIST("h2TPCnSigma_Pin_Pi"), track.tpcInnerParam(), track.tpcNSigmaPi());
          registry.fill(HIST("h2TOFnSigma_Pin_Pi"), track.tpcInnerParam(), track.tofNSigmaPi());
        } else {
          registry.fill(HIST("h2TPCdEdx_Pin_Pr"), track.tpcInnerParam(), track.tpcSignal());
          registry.fill(HIST("h2TOFbeta_Pin_Pr"), track.tpcInnerParam(), track.beta());
          registry.fill(HIST("h2TPCnSigma_Pin_Pr"), track.tpcInnerParam(), track.tpcNSigmaPr());
          registry.fill(HIST("h2TOFnSigma_Pin_Pr"), track.tpcInnerParam(), track.tofNSigmaPr());
        }
      }
      if (bool(track.pidbit() & (1 << v0selector::kOmega))) {
        registry.fill(HIST("h2TPCdEdx_Pin_Ka"), track.tpcInnerParam(), track.tpcSignal());
        registry.fill(HIST("h2TOFbeta_Pin_Ka"), track.tpcInnerParam(), track.beta());
        registry.fill(HIST("h2TPCnSigma_Pin_Ka"), track.tpcInnerParam(), track.tpcNSigmaKa());
        registry.fill(HIST("h2TOFnSigma_Pin_Ka"), track.tpcInnerParam(), track.tofNSigmaKa());
      }
    } //end of track loop
  }   //end of process
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

  void process(soa::Join<aod::Collisions, aod::EvSels>::iterator const& collision, aod::ReducedV0s const& v0Gammas)
  {
    //printf("begining of process\n");
    registry.fill(HIST("hEventCounter"), 1.0); //all

    if (!collision.alias()[kINT7]) {
      return;
    }

    registry.fill(HIST("hEventCounter"), 2.0); //INT7

    for (auto& g : v0Gammas) {
      registry.fill(HIST("hV0Pt"), g.pt());
      registry.fill(HIST("hV0EtaPhi"), g.phi(), g.eta());
    }

    for (auto& [g1, g2] : combinations(v0Gammas, v0Gammas)) {
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
    adaptAnalysisTask<v0selector>(cfgc, TaskName{"v0-selector"}), adaptAnalysisTask<trackPIDQA>(cfgc, TaskName{"track-pid-qa"}), adaptAnalysisTask<v0gammaQA>(cfgc, TaskName{"v0-gamma-qa"})};
}
