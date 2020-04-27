// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

#define BOOST_TEST_MODULE Test DCAFitterN class
#define BOOST_TEST_MAIN
#define BOOST_TEST_DYN_LINK
#include <boost/test/unit_test.hpp>

#include "DetectorsVertexing/DCAFitterN.h"
#include "CommonUtils/TreeStreamRedirector.h"
#include <TRandom.h>
#include <TGenPhaseSpace.h>
#include <TLorentzVector.h>
#include <TStopwatch.h>
#include <Math/SVector.h>
#include <array>

namespace o2
{
namespace vertexing
{

using Vec3D = ROOT::Math::SVector<double, 3>;

template <class FITTER>
float checkResults(o2::utils::TreeStreamRedirector& outs, std::string& treeName, const FITTER& fitter,
                   Vec3D& vgen, TLorentzVector& genPar, const std::vector<double>& dtMass)
{
  int nCand = fitter.getNCandidates();
  std::array<float, 3> p;
  float distMin = 1e9;
  for (int ic = 0; ic < nCand; ic++) {
    const auto& vtx = fitter.getPCACandidate(ic);
    auto df = vgen;
    df -= vtx;

    TLorentzVector moth, prong;
    for (int i = 0; i < fitter.getNProngs(); i++) {
      const auto& trc = fitter.getTrack(i, ic);
      trc.getPxPyPzGlo(p);
      prong.SetVectM({p[0], p[1], p[2]}, dtMass[i]);
      moth += prong;
    }
    auto nIter = fitter.getNIterations(ic);
    auto chi2 = fitter.getChi2AtPCACandidate(ic);
    double dst = TMath::Sqrt(df[0] * df[0] + df[1] * df[1] + df[2] * df[2]);
    distMin = dst < distMin ? dst : distMin;
    //    float genX
    outs << treeName.c_str() << "cand=" << ic << "ncand=" << nCand << "nIter=" << nIter << "chi2=" << chi2
         << "genPart=" << genPar << "recPart=" << moth
         << "genX=" << vgen[0] << "genY=" << vgen[1] << "genZ=" << vgen[2]
         << "dx=" << df[0] << "dy=" << df[1] << "dz=" << df[2] << "dst=" << dst << "\n";
  }
  return distMin;
}

TLorentzVector generate(Vec3D& vtx, std::vector<o2::track::TrackParCov>& vctr, float bz,
                        TGenPhaseSpace& genPHS, double parMass, const std::vector<double>& dtMass, std::vector<int> forceQ)
{
  const float errYZ = 1e-2, errSlp = 1e-3, errQPT = 2e-2;
  std::array<float, 15> covm = {
    errYZ * errYZ,
    0., errYZ * errYZ,
    0, 0., errSlp * errSlp,
    0., 0., 0., errSlp * errSlp,
    0., 0., 0., 0., errQPT * errQPT};
  bool accept = true;
  TLorentzVector parent, d0, d1, d2;
  do {
    accept = true;
    double y = gRandom->Rndm() - 0.5;
    double pt = 0.1 + gRandom->Rndm() * 3;
    double mt = TMath::Sqrt(parMass * parMass + pt * pt);
    double pz = mt * TMath::SinH(y);
    double phi = gRandom->Rndm() * TMath::Pi() * 2;
    double en = mt * TMath::CosH(y);
    double rdec = 10.; // radius of the decay
    vtx[0] = rdec * TMath::Cos(phi);
    vtx[1] = rdec * TMath::Sin(phi);
    vtx[2] = rdec * pz / pt;
    parent.SetPxPyPzE(pt * TMath::Cos(phi), pt * TMath::Sin(phi), pz, en);
    int nd = dtMass.size();
    genPHS.SetDecay(parent, nd, dtMass.data());
    genPHS.Generate();
    vctr.clear();
    float p[4];
    for (int i = 0; i < nd; i++) {
      auto* dt = genPHS.GetDecay(i);
      if (dt->Pt() < 0.05) {
        accept = false;
        break;
      }
      dt->GetXYZT(p);
      float s, c, x;
      std::array<float, 5> params;
      o2::utils::sincosf(dt->Phi(), s, c);
      o2::utils::rotateZInv(vtx[0], vtx[1], x, params[0], s, c);

      params[1] = vtx[2];
      params[2] = 0.; // since alpha = phi
      params[3] = 1. / TMath::Tan(dt->Theta());
      params[4] = (i % 2 ? -1. : 1.) / dt->Pt();
      covm[14] = errQPT * errQPT * params[4] * params[4];
      //
      // randomize
      float r1, r2;
      gRandom->Rannor(r1, r2);
      params[0] += r1 * errYZ;
      params[1] += r2 * errYZ;
      gRandom->Rannor(r1, r2);
      params[2] += r1 * errSlp;
      params[3] += r2 * errSlp;
      params[4] *= gRandom->Gaus(1., errQPT);
      if (forceQ[i] == 0) {
        params[4] = 0.; // impose straight track
      }
      auto& trc = vctr.emplace_back(x, dt->Phi(), params, covm);
      float rad = forceQ[i] == 0 ? 600. : TMath::Abs(1. / trc.getCurvature(bz));
      if (!trc.propagateTo(trc.getX() + (gRandom->Rndm() - 0.5) * rad * 0.05, bz) ||
          !trc.rotate(trc.getAlpha() + (gRandom->Rndm() - 0.5) * 0.2)) {
        printf("Failed to randomize ");
        trc.print();
      }
    }
  } while (!accept);

  return parent;
}

BOOST_AUTO_TEST_CASE(DCAFitterNProngs)
{
  constexpr int NTest = 10000;
  o2::utils::TreeStreamRedirector outStream("dcafitterNTest.root");

  TGenPhaseSpace genPHS;
  constexpr double pion = 0.13957;
  constexpr double k0 = 0.49761;
  constexpr double kch = 0.49368;
  constexpr double dch = 1.86965;
  std::vector<double> k0dec = {pion, pion};
  std::vector<double> dchdec = {pion, kch, pion};
  std::vector<o2::track::TrackParCov> vctracks;
  Vec3D vtxGen;

  double bz = 5.0;
  // 2 prongs vertices
  {
    LOG(INFO) << "Processing 2-prong Helix - Helix case";
    std::vector<int> forceQ{1, 1};

    o2::vertexing::DCAFitterN<2> ft; // 2 prong fitter
    ft.setBz(bz);
    ft.setPropagateToPCA(true);  // After finding the vertex, propagate tracks to the DCA. This is default anyway
    ft.setMaxR(200);             // do not consider V0 seeds with 2D circles crossing above this R. This is default anyway
    ft.setMaxDZIni(4);           // do not consider V0 seeds with tracks Z-distance exceeding this. This is default anyway
    ft.setMinParamChange(1e-3);  // stop iterations if max correction is below this value. This is default anyway
    ft.setMinRelChi2Change(0.9); // stop iterations if chi2 improves by less that this factor

    std::string treeName2A = "pr2a", treeName2W = "pr2w";
    TStopwatch swA, swW;
    int nfoundA = 0, nfoundW = 0;
    double meanDA = 0, meanDW = 0;
    swA.Stop();
    swW.Stop();
    for (int iev = 0; iev < NTest; iev++) {
      auto genParent = generate(vtxGen, vctracks, bz, genPHS, k0, k0dec, forceQ);

      ft.setUseAbsDCA(true);
      swA.Start(false);
      int ncA = ft.process(vctracks[0], vctracks[1]); // HERE WE FIT THE VERTICES
      swA.Stop();
      LOG(DEBUG) << "fit abs.dist " << iev << " NC: " << ncA << " Chi2: " << (ncA ? ft.getChi2AtPCACandidate(0) : -1);
      if (ncA) {
        auto minD = checkResults(outStream, treeName2A, ft, vtxGen, genParent, k0dec);
        meanDA += minD;
        nfoundA++;
      }

      ft.setUseAbsDCA(false);
      swW.Start(false);
      int ncW = ft.process(vctracks[0], vctracks[1]); // HERE WE FIT THE VERTICES
      swW.Stop();
      LOG(DEBUG) << "fit wgh.dist " << iev << " NC: " << ncW << " Chi2: " << (ncW ? ft.getChi2AtPCACandidate(0) : -1);
      if (ncW) {
        auto minD = checkResults(outStream, treeName2W, ft, vtxGen, genParent, k0dec);
        meanDW += minD;
        nfoundW++;
      }
    }
    ft.print();
    meanDA /= nfoundA ? nfoundA : 1;
    meanDW /= nfoundW ? nfoundW : 1;
    LOG(INFO) << "Processed " << NTest << " 2-prong vertices Helix : Helix";
    LOG(INFO) << "2-prongs with abs.dist minization: eff= " << float(nfoundA) / NTest
              << " mean.dist to truth: " << meanDA << " CPU time: " << swA.CpuTime();
    LOG(INFO) << "2-prongs with wgh.dist minization: eff= " << float(nfoundW) / NTest
              << " mean.dist to truth: " << meanDW << " CPU time: " << swW.CpuTime();
    BOOST_CHECK(nfoundA > 0.99 * NTest);
    BOOST_CHECK(nfoundW > 0.99 * NTest);
    BOOST_CHECK(meanDA < 0.1);
    BOOST_CHECK(meanDW < 0.1);
  }

  // 2 prongs vertices with one of charges set to 0: Helix : Line
  {
    std::vector<int> forceQ{1, 1};
    LOG(INFO) << "Processing 2-prong Helix - Line case";
    o2::vertexing::DCAFitterN<2> ft; // 2 prong fitter
    ft.setBz(bz);
    ft.setPropagateToPCA(true);  // After finding the vertex, propagate tracks to the DCA. This is default anyway
    ft.setMaxR(200);             // do not consider V0 seeds with 2D circles crossing above this R. This is default anyway
    ft.setMaxDZIni(4);           // do not consider V0 seeds with tracks Z-distance exceeding this. This is default anyway
    ft.setMinParamChange(1e-3);  // stop iterations if max correction is below this value. This is default anyway
    ft.setMinRelChi2Change(0.9); // stop iterations if chi2 improves by less that this factor

    std::string treeName2A = "pr2aHL", treeName2W = "pr2wHL";
    TStopwatch swA, swW;
    int nfoundA = 0, nfoundW = 0;
    double meanDA = 0, meanDW = 0;
    swA.Stop();
    swW.Stop();
    for (int iev = 0; iev < NTest; iev++) {
      forceQ[iev % 2] = 1;
      forceQ[1 - iev % 2] = 0;
      auto genParent = generate(vtxGen, vctracks, bz, genPHS, k0, k0dec, forceQ);

      ft.setUseAbsDCA(true);
      swA.Start(false);
      int ncA = ft.process(vctracks[0], vctracks[1]); // HERE WE FIT THE VERTICES
      swA.Stop();
      LOG(DEBUG) << "fit abs.dist " << iev << " NC: " << ncA << " Chi2: " << (ncA ? ft.getChi2AtPCACandidate(0) : -1);
      if (ncA) {
        auto minD = checkResults(outStream, treeName2A, ft, vtxGen, genParent, k0dec);
        meanDA += minD;
        nfoundA++;
      }

      ft.setUseAbsDCA(false);
      swW.Start(false);
      int ncW = ft.process(vctracks[0], vctracks[1]); // HERE WE FIT THE VERTICES
      swW.Stop();
      LOG(DEBUG) << "fit wgh.dist " << iev << " NC: " << ncW << " Chi2: " << (ncW ? ft.getChi2AtPCACandidate(0) : -1);
      if (ncW) {
        auto minD = checkResults(outStream, treeName2W, ft, vtxGen, genParent, k0dec);
        meanDW += minD;
        nfoundW++;
      }
    }
    ft.print();
    meanDA /= nfoundA ? nfoundA : 1;
    meanDW /= nfoundW ? nfoundW : 1;
    LOG(INFO) << "Processed " << NTest << " 2-prong vertices: Helix : Line";
    LOG(INFO) << "2-prongs with abs.dist minization: eff= " << float(nfoundA) / NTest
              << " mean.dist to truth: " << meanDA << " CPU time: " << swA.CpuTime();
    LOG(INFO) << "2-prongs with wgh.dist minization: eff= " << float(nfoundW) / NTest
              << " mean.dist to truth: " << meanDW << " CPU time: " << swW.CpuTime();
    BOOST_CHECK(nfoundA > 0.99 * NTest);
    BOOST_CHECK(nfoundW > 0.99 * NTest);
    BOOST_CHECK(meanDA < 0.1);
    BOOST_CHECK(meanDW < 0.1);
  }

  // 2 prongs vertices with both of charges set to 0: Line : Line
  {
    std::vector<int> forceQ{0, 0};
    LOG(INFO) << "Processing 2-prong Line - Line case";
    o2::vertexing::DCAFitterN<2> ft; // 2 prong fitter
    ft.setBz(bz);
    ft.setPropagateToPCA(true);  // After finding the vertex, propagate tracks to the DCA. This is default anyway
    ft.setMaxR(200);             // do not consider V0 seeds with 2D circles crossing above this R. This is default anyway
    ft.setMaxDZIni(4);           // do not consider V0 seeds with tracks Z-distance exceeding this. This is default anyway
    ft.setMinParamChange(1e-3);  // stop iterations if max correction is below this value. This is default anyway
    ft.setMinRelChi2Change(0.9); // stop iterations if chi2 improves by less that this factor

    std::string treeName2A = "pr2aLL", treeName2W = "pr2wLL";
    TStopwatch swA, swW;
    int nfoundA = 0, nfoundW = 0;
    double meanDA = 0, meanDW = 0;
    swA.Stop();
    swW.Stop();
    for (int iev = 0; iev < NTest; iev++) {
      forceQ[0] = forceQ[1] = 0;
      auto genParent = generate(vtxGen, vctracks, bz, genPHS, k0, k0dec, forceQ);

      ft.setUseAbsDCA(true);
      swA.Start(false);
      int ncA = ft.process(vctracks[0], vctracks[1]); // HERE WE FIT THE VERTICES
      swA.Stop();
      LOG(DEBUG) << "fit abs.dist " << iev << " NC: " << ncA << " Chi2: " << (ncA ? ft.getChi2AtPCACandidate(0) : -1);
      if (ncA) {
        auto minD = checkResults(outStream, treeName2A, ft, vtxGen, genParent, k0dec);
        meanDA += minD;
        nfoundA++;
      }

      ft.setUseAbsDCA(false);
      swW.Start(false);
      int ncW = ft.process(vctracks[0], vctracks[1]); // HERE WE FIT THE VERTICES
      swW.Stop();
      LOG(DEBUG) << "fit wgh.dist " << iev << " NC: " << ncW << " Chi2: " << (ncW ? ft.getChi2AtPCACandidate(0) : -1);
      if (ncW) {
        auto minD = checkResults(outStream, treeName2W, ft, vtxGen, genParent, k0dec);
        meanDW += minD;
        nfoundW++;
      }
    }
    ft.print();
    meanDA /= nfoundA ? nfoundA : 1;
    meanDW /= nfoundW ? nfoundW : 1;
    LOG(INFO) << "Processed " << NTest << " 2-prong vertices: Line : Line";
    LOG(INFO) << "2-prongs with abs.dist minization: eff= " << float(nfoundA) / NTest
              << " mean.dist to truth: " << meanDA << " CPU time: " << swA.CpuTime();
    LOG(INFO) << "2-prongs with wgh.dist minization: eff= " << float(nfoundW) / NTest
              << " mean.dist to truth: " << meanDW << " CPU time: " << swW.CpuTime();
    BOOST_CHECK(nfoundA > 0.99 * NTest);
    BOOST_CHECK(nfoundW > 0.99 * NTest);
    BOOST_CHECK(meanDA < 0.1);
    BOOST_CHECK(meanDW < 0.1);
  }

  // 3 prongs vertices
  {
    std::vector<int> forceQ{1, 1, 1};

    o2::vertexing::DCAFitterN<3> ft; // 3 prong fitter
    ft.setBz(bz);
    ft.setPropagateToPCA(true);  // After finding the vertex, propagate tracks to the DCA. This is default anyway
    ft.setMaxR(200);             // do not consider V0 seeds with 2D circles crossing above this R. This is default anyway
    ft.setMaxDZIni(4);           // do not consider V0 seeds with tracks Z-distance exceeding this. This is default anyway
    ft.setMinParamChange(1e-3);  // stop iterations if max correction is below this value. This is default anyway
    ft.setMinRelChi2Change(0.9); // stop iterations if chi2 improves by less that this factor

    std::string treeName3A = "pr3a", treeName3W = "pr3w";
    TStopwatch swA, swW;
    int nfoundA = 0, nfoundW = 0;
    double meanDA = 0, meanDW = 0;
    swA.Stop();
    swW.Stop();
    for (int iev = 0; iev < NTest; iev++) {
      auto genParent = generate(vtxGen, vctracks, bz, genPHS, dch, dchdec, forceQ);

      ft.setUseAbsDCA(true);
      swA.Start(false);
      int ncA = ft.process(vctracks[0], vctracks[1], vctracks[2]); // HERE WE FIT THE VERTICES
      swA.Stop();
      LOG(DEBUG) << "fit abs.dist " << iev << " NC: " << ncA << " Chi2: " << (ncA ? ft.getChi2AtPCACandidate(0) : -1);
      if (ncA) {
        auto minD = checkResults(outStream, treeName3A, ft, vtxGen, genParent, dchdec);
        meanDA += minD;
        nfoundA++;
      }

      ft.setUseAbsDCA(false);
      swW.Start(false);
      int ncW = ft.process(vctracks[0], vctracks[1], vctracks[2]); // HERE WE FIT THE VERTICES
      swW.Stop();
      LOG(DEBUG) << "fit wgh.dist " << iev << " NC: " << ncW << " Chi2: " << (ncW ? ft.getChi2AtPCACandidate(0) : -1);
      if (ncW) {
        auto minD = checkResults(outStream, treeName3W, ft, vtxGen, genParent, dchdec);
        meanDW += minD;
        nfoundW++;
      }
    }
    ft.print();
    meanDA /= nfoundA ? nfoundA : 1;
    meanDW /= nfoundW ? nfoundW : 1;
    LOG(INFO) << "Processed " << NTest << " 3-prong vertices";
    LOG(INFO) << "3-prongs with abs.dist minization: eff= " << float(nfoundA) / NTest
              << " mean.dist to truth: " << meanDA << " CPU time: " << swA.CpuTime();
    LOG(INFO) << "3-prongs with wgh.dist minization: eff= " << float(nfoundW) / NTest
              << " mean.dist to truth: " << meanDW << " CPU time: " << swW.CpuTime();
    BOOST_CHECK(nfoundA > 0.9 * NTest);
    BOOST_CHECK(nfoundW > 0.9 * NTest);
    BOOST_CHECK(meanDA < 0.1);
    BOOST_CHECK(meanDW < 0.1);
  }

  outStream.Close();
}

} // namespace vertexing
} // namespace o2
