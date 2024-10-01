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

#define BOOST_TEST_MODULE Test DCAFitterN class
#define BOOST_TEST_MAIN
#define BOOST_TEST_DYN_LINK
#include <boost/test/unit_test.hpp>

#include "DCAFitter/DCAFitterN.h"
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
float checkResults(o2::utils::TreeStreamRedirector& outs, std::string& treeName, FITTER& fitter,
                   Vec3D& vgen, TLorentzVector& genPar, const std::vector<double>& dtMass)
{
  int nCand = fitter.getNCandidates();
  std::array<float, 3> p;
  float distMin = 1e9;
  bool absDCA = fitter.getUseAbsDCA();
  bool useWghDCA = fitter.getWeightedFinalPCA();
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
    auto parentTrack = fitter.createParentTrackParCov(ic);
    outs << treeName.c_str() << "cand=" << ic << "ncand=" << nCand << "nIter=" << nIter << "chi2=" << chi2
         << "genPart=" << genPar << "recPart=" << moth
         << "genX=" << vgen[0] << "genY=" << vgen[1] << "genZ=" << vgen[2]
         << "dx=" << df[0] << "dy=" << df[1] << "dz=" << df[2] << "dst=" << dst
         << "useAbsDCA=" << absDCA << "useWghDCA=" << useWghDCA << "parent=" << parentTrack;
    for (int i = 0; i < fitter.getNProngs(); i++) {
      outs << treeName.c_str() << fmt::format("prong{}=", i).c_str() << fitter.getTrack(i, ic);
    }
    outs << treeName.c_str() << "\n";
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
      o2::math_utils::sincos(dt->Phi(), s, c);
      o2::math_utils::rotateZInv(vtx[0], vtx[1], x, params[0], s, c);

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

#ifdef DO_SINGLE_THREAD_TEST
BOOST_AUTO_TEST_CASE(DCAFitterNProngs)
{
  // gRandom->Delete();
  // gRandom = new TRandom(42);
  o2::utils::TreeStreamRedirector outStream("dcafitterNTest.root");

  TGenPhaseSpace genPHS;
  constexpr double ele = 0.00051;
  constexpr double gamma = 2 * ele + 1e-6;
  constexpr double pion = 0.13957;
  constexpr double k0 = 0.49761;
  constexpr double kch = 0.49368;
  constexpr double dch = 1.86965;
  std::vector<double> gammadec = {ele, ele};
  std::vector<double> k0dec = {pion, pion};
  std::vector<double> dchdec = {pion, kch, pion};
  std::vector<o2::track::TrackParCov> vctracks;
  Vec3D vtxGen;

  double bz = 5.0;
  // 2 prongs vertices
  {
    LOG(info) << "Processing 2-prong Helix - Helix case";
    std::vector<int> forceQ{1, 1};

    o2::vertexing::DCAFitterN<2> ft; // 2 prong fitter
    ft.setBz(bz);
    ft.setPropagateToPCA(true);  // After finding the vertex, propagate tracks to the DCA. This is default anyway
    ft.setMaxR(200);             // do not consider V0 seeds with 2D circles crossing above this R. This is default anyway
    ft.setMaxDZIni(4);           // do not consider V0 seeds with tracks Z-distance exceeding this. This is default anyway
    ft.setMaxDXYIni(4);          // do not consider V0 seeds with tracks XY-distance exceeding this. This is default anyway
    ft.setMinParamChange(1e-3);  // stop iterations if max correction is below this value. This is default anyway
    ft.setMinRelChi2Change(0.9); // stop iterations if chi2 improves by less that this factor

    std::string treeName2A = "pr2a", treeName2AW = "pr2aw", treeName2W = "pr2w";
    TStopwatch swA, swAW, swW;
    int nfoundA = 0, nfoundAW = 0, nfoundW = 0;
    double meanDA = 0, meanDAW = 0, meanDW = 0;
    swA.Stop();
    swAW.Stop();
    swW.Stop();
    for (int iev = 0; iev < NTest; iev++) {
      auto genParent = generate(vtxGen, vctracks, bz, genPHS, k0, k0dec, forceQ);

      ft.setUseAbsDCA(true);
      swA.Start(false);
      int ncA = device::process(1, 1, ft, vctracks[0], vctracks[1]); // HERE WE FIT THE VERTICES
      swA.Stop();
      LOG(debug) << "fit abs.dist " << iev << " NC: " << ncA << " Chi2: " << (ncA ? ft.getChi2AtPCACandidate(0) : -1);
      if (ncA) {
        auto minD = checkResults(outStream, treeName2A, ft, vtxGen, genParent, k0dec);
        meanDA += minD;
        nfoundA++;
      }

      ft.setUseAbsDCA(true);
      ft.setWeightedFinalPCA(true);
      swAW.Start(false);
      int ncAW = device::process(1, 1, ft, vctracks[0], vctracks[1]); // HERE WE FIT THE VERTICES
      swAW.Stop();
      LOG(debug) << "fit abs.dist with final weighted DCA " << iev << " NC: " << ncAW << " Chi2: " << (ncAW ? ft.getChi2AtPCACandidate(0) : -1);
      if (ncAW) {
        auto minD = checkResults(outStream, treeName2AW, ft, vtxGen, genParent, k0dec);
        meanDAW += minD;
        nfoundAW++;
      }

      ft.setUseAbsDCA(false);
      ft.setWeightedFinalPCA(false);
      swW.Start(false);
      int ncW = device::process(1, 1, ft, vctracks[0], vctracks[1]); // HERE WE FIT THE VERTICES
      swW.Stop();
      LOG(debug) << "fit wgh.dist " << iev << " NC: " << ncW << " Chi2: " << (ncW ? ft.getChi2AtPCACandidate(0) : -1);
      if (ncW) {
        auto minD = checkResults(outStream, treeName2W, ft, vtxGen, genParent, k0dec);
        meanDW += minD;
        nfoundW++;
      }
    }

    meanDA /= nfoundA ? nfoundA : 1;
    meanDAW /= nfoundAW ? nfoundA : 1;
    meanDW /= nfoundW ? nfoundW : 1;
    LOG(info) << "Processed " << NTest << " 2-prong vertices Helix : Helix";
    LOG(info) << "2-prongs with abs.dist minimization: eff = " << float(nfoundA) / NTest
              << " mean.dist to truth: " << meanDA << " Total time: " << swA.CpuTime() * 1000 << " ms";
    LOG(info) << "2-prongs with abs.dist but wghPCA: eff = " << float(nfoundAW) / NTest
              << " mean.dist to truth: " << meanDAW << " Total time: " << swAW.CpuTime() * 1000 << " ms";
    LOG(info) << "2-prongs with wgh.dist minimization: eff = " << float(nfoundW) / NTest
              << " mean.dist to truth: " << meanDW << " Total time: " << swW.CpuTime() * 1000 << " ms";
    BOOST_CHECK(nfoundA > 0.99 * NTest);
    BOOST_CHECK(nfoundAW > 0.99 * NTest);
    BOOST_CHECK(nfoundW > 0.99 * NTest);
    BOOST_CHECK(meanDA < 0.1);
    BOOST_CHECK(meanDAW < 0.1);
    BOOST_CHECK(meanDW < 0.1);
  }

  // 2 prongs vertices with collinear tracks (gamma conversion)
  {
    LOG(info) << "Processing 2-prong Helix - Helix case gamma conversion";
    std::vector<int> forceQ{1, 1};

    o2::vertexing::DCAFitterN<2> ft; // 2 prong fitter
    ft.setBz(bz);
    ft.setPropagateToPCA(true);  // After finding the vertex, propagate tracks to the DCA. This is default anyway
    ft.setMaxR(200);             // do not consider V0 seeds with 2D circles crossing above this R. This is default anyway
    ft.setMaxDZIni(4);           // do not consider V0 seeds with tracks Z-distance exceeding this. This is default anyway
    ft.setMaxDXYIni(4);          // do not consider V0 seeds with tracks XY-distance exceeding this. This is default anyway
    ft.setMinParamChange(1e-3);  // stop iterations if max correction is below this value. This is default anyway
    ft.setMinRelChi2Change(0.9); // stop iterations if chi2 improves by less that this factor

    std::string treeName2A = "gpr2a", treeName2AW = "gpr2aw", treeName2W = "gpr2w";
    TStopwatch swA, swAW, swW;
    int nfoundA = 0, nfoundAW = 0, nfoundW = 0;
    double meanDA = 0, meanDAW = 0, meanDW = 0;
    swA.Stop();
    swAW.Stop();
    swW.Stop();
    for (int iev = 0; iev < NTest; iev++) {
      auto genParent = generate(vtxGen, vctracks, bz, genPHS, gamma, gammadec, forceQ);

      ft.setUseAbsDCA(true);
      swA.Start(false);
      int ncA = device::process(1, 1, ft, vctracks[0], vctracks[1]);
      swA.Stop();
      LOG(debug) << "fit abs.dist " << iev << " NC: " << ncA << " Chi2: " << (ncA ? ft.getChi2AtPCACandidate(0) : -1);
      if (ncA) {
        auto minD = checkResults(outStream, treeName2A, ft, vtxGen, genParent, gammadec);
        meanDA += minD;
        nfoundA++;
      }

      ft.setUseAbsDCA(true);
      ft.setWeightedFinalPCA(true);
      swAW.Start(false);
      int ncAW = device::process(1, 1, ft, vctracks[0], vctracks[1]); // HERE WE FIT THE VERTICES
      swAW.Stop();
      LOG(debug) << "fit abs.dist with final weighted DCA " << iev << " NC: " << ncAW << " Chi2: " << (ncAW ? ft.getChi2AtPCACandidate(0) : -1);
      if (ncAW) {
        auto minD = checkResults(outStream, treeName2AW, ft, vtxGen, genParent, gammadec);
        meanDAW += minD;
        nfoundAW++;
      }

      ft.setUseAbsDCA(false);
      ft.setWeightedFinalPCA(false);
      swW.Start(false);
      int ncW = device::process(1, 1, ft, vctracks[0], vctracks[1]); // HERE WE FIT THE VERTICES
      swW.Stop();
      LOG(debug) << "fit wgh.dist " << iev << " NC: " << ncW << " Chi2: " << (ncW ? ft.getChi2AtPCACandidate(0) : -1);
      if (ncW) {
        auto minD = checkResults(outStream, treeName2W, ft, vtxGen, genParent, gammadec);
        meanDW += minD;
        nfoundW++;
      }
    }

    meanDA /= nfoundA ? nfoundA : 1;
    meanDAW /= nfoundA ? nfoundA : 1;
    meanDW /= nfoundW ? nfoundW : 1;
    LOG(info) << "Processed " << NTest << " 2-prong vertices Helix : Helix from gamma conversion";
    LOG(info) << "2-prongs with abs.dist minimization: eff = " << float(nfoundA) / NTest
              << " mean.dist to truth: " << meanDA << " Total time: " << swA.CpuTime();
    LOG(info) << "2-prongs with abs.dist but wghPCA: eff = " << float(nfoundAW) / NTest
              << " mean.dist to truth: " << meanDAW << " Total time: " << swAW.CpuTime();
    LOG(info) << "2-prongs with wgh.dist minimization: eff = " << float(nfoundW) / NTest
              << " mean.dist to truth: " << meanDW << " Total time: " << swW.CpuTime();
    BOOST_CHECK(nfoundA > 0.99 * NTest);
    BOOST_CHECK(nfoundAW > 0.99 * NTest);
    BOOST_CHECK(nfoundW > 0.99 * NTest);
    BOOST_CHECK(meanDA < 2.1);
    BOOST_CHECK(meanDAW < 2.1);
    BOOST_CHECK(meanDW < 2.1);
  }

  // 2 prongs vertices with one of charges set to 0: Helix : Line
  {
    std::vector<int> forceQ{1, 1};
    LOG(info) << "Processing 2-prong Helix - Line case";
    o2::vertexing::DCAFitterN<2> ft; // 2 prong fitter
    ft.setBz(bz);
    ft.setPropagateToPCA(true);  // After finding the vertex, propagate tracks to the DCA. This is default anyway
    ft.setMaxR(200);             // do not consider V0 seeds with 2D circles crossing above this R. This is default anyway
    ft.setMaxDZIni(4);           // do not consider V0 seeds with tracks Z-distance exceeding this. This is default anyway
    ft.setMinParamChange(1e-3);  // stop iterations if max correction is below this value. This is default anyway
    ft.setMinRelChi2Change(0.9); // stop iterations if chi2 improves by less that this factor

    std::string treeName2A = "pr2aHL", treeName2AW = "pr2awHL", treeName2W = "pr2wHL";
    TStopwatch swA, swAW, swW;
    int nfoundA = 0, nfoundAW = 0, nfoundW = 0;
    double meanDA = 0, meanDAW = 0, meanDW = 0;
    swA.Stop();
    swAW.Stop();
    swW.Stop();
    for (int iev = 0; iev < NTest; iev++) {
      forceQ[iev % 2] = 1;
      forceQ[1 - iev % 2] = 0;
      auto genParent = generate(vtxGen, vctracks, bz, genPHS, k0, k0dec, forceQ);

      ft.setUseAbsDCA(true);
      swA.Start(false);
      int ncA = device::process(1, 1, ft, vctracks[0], vctracks[1]);
      swA.Stop();
      LOG(debug) << "fit abs.dist with final weighted DCA " << iev << " NC: " << ncA << " Chi2: " << (ncA ? ft.getChi2AtPCACandidate(0) : -1);
      if (ncA) {
        auto minD = checkResults(outStream, treeName2A, ft, vtxGen, genParent, k0dec);
        meanDA += minD;
        nfoundA++;
      }

      ft.setUseAbsDCA(true);
      ft.setWeightedFinalPCA(true);
      swAW.Start(false);
      int ncAW = device::process(1, 1, ft, vctracks[0], vctracks[1]); // HERE WE FIT THE VERTICES
      swAW.Stop();
      LOG(debug) << "fit abs.dist  " << iev << " NC: " << ncAW << " Chi2: " << (ncAW ? ft.getChi2AtPCACandidate(0) : -1);
      if (ncAW) {
        auto minD = checkResults(outStream, treeName2AW, ft, vtxGen, genParent, k0dec);
        meanDAW += minD;
        nfoundAW++;
      }

      ft.setUseAbsDCA(false);
      ft.setWeightedFinalPCA(false);
      swW.Start(false);
      int ncW = device::process(1, 1, ft, vctracks[0], vctracks[1]); // HERE WE FIT THE VERTICES
      swW.Stop();
      LOG(debug) << "fit wgh.dist " << iev << " NC: " << ncW << " Chi2: " << (ncW ? ft.getChi2AtPCACandidate(0) : -1);
      if (ncW) {
        auto minD = checkResults(outStream, treeName2W, ft, vtxGen, genParent, k0dec);
        meanDW += minD;
        nfoundW++;
      }
    }

    meanDA /= nfoundA ? nfoundA : 1;
    meanDAW /= nfoundAW ? nfoundAW : 1;
    meanDW /= nfoundW ? nfoundW : 1;
    LOG(info) << "Processed " << NTest << " 2-prong vertices: Helix : Line";
    LOG(info) << "2-prongs with abs.dist minimization: eff = " << float(nfoundA) / NTest
              << " mean.dist to truth: " << meanDA << " Total time: " << swA.CpuTime();
    LOG(info) << "2-prongs with abs.dist but wghPCA: eff = " << float(nfoundAW) / NTest
              << " mean.dist to truth: " << meanDAW << " Total time: " << swAW.CpuTime();
    LOG(info) << "2-prongs with wgh.dist minimization: eff = " << float(nfoundW) / NTest
              << " mean.dist to truth: " << meanDW << " Total time: " << swW.CpuTime();
    BOOST_CHECK(nfoundA > 0.99 * NTest);
    BOOST_CHECK(nfoundAW > 0.99 * NTest);
    BOOST_CHECK(nfoundW > 0.99 * NTest);
    BOOST_CHECK(meanDA < 0.1);
    BOOST_CHECK(meanDAW < 0.1);
    BOOST_CHECK(meanDW < 0.1);
  }

  // 2 prongs vertices with both of charges set to 0: Line : Line
  {
    std::vector<int> forceQ{0, 0};
    LOG(info) << "Processing 2-prong Line - Line case";
    o2::vertexing::DCAFitterN<2> ft; // 2 prong fitter
    ft.setBz(bz);
    ft.setPropagateToPCA(true);  // After finding the vertex, propagate tracks to the DCA. This is default anyway
    ft.setMaxR(200);             // do not consider V0 seeds with 2D circles crossing above this R. This is default anyway
    ft.setMaxDZIni(4);           // do not consider V0 seeds with tracks Z-distance exceeding this. This is default anyway
    ft.setMinParamChange(1e-3);  // stop iterations if max correction is below this value. This is default anyway
    ft.setMinRelChi2Change(0.9); // stop iterations if chi2 improves by less that this factor

    std::string treeName2A = "pr2aLL", treeName2AW = "pr2awLL", treeName2W = "pr2wLL";
    TStopwatch swA, swAW, swW;
    int nfoundA = 0, nfoundAW = 0, nfoundW = 0;
    double meanDA = 0, meanDAW = 0, meanDW = 0;
    swA.Stop();
    swAW.Stop();
    swW.Stop();
    for (int iev = 0; iev < NTest; iev++) {
      forceQ[0] = forceQ[1] = 0;
      auto genParent = generate(vtxGen, vctracks, bz, genPHS, k0, k0dec, forceQ);

      ft.setUseAbsDCA(true);
      swA.Start(false);
      int ncA = device::process(1, 1, ft, vctracks[0], vctracks[1]); // HERE WE FIT THE VERTICES
      swA.Stop();
      LOG(debug) << "fit abs.dist " << iev << " NC: " << ncA << " Chi2: " << (ncA ? ft.getChi2AtPCACandidate(0) : -1);
      if (ncA) {
        auto minD = checkResults(outStream, treeName2A, ft, vtxGen, genParent, k0dec);
        meanDA += minD;
        nfoundA++;
      }

      ft.setUseAbsDCA(true);
      ft.setWeightedFinalPCA(true);
      swAW.Start(false);
      int ncAW = device::process(1, 1, ft, vctracks[0], vctracks[1]); // HERE WE FIT THE VERTICES
      swAW.Stop();
      LOG(debug) << "fit abs.dist " << iev << " NC: " << ncAW << " Chi2: " << (ncAW ? ft.getChi2AtPCACandidate(0) : -1);
      if (ncAW) {
        auto minD = checkResults(outStream, treeName2AW, ft, vtxGen, genParent, k0dec);
        meanDAW += minD;
        nfoundAW++;
      }

      ft.setUseAbsDCA(false);
      ft.setWeightedFinalPCA(false);
      swW.Start(false);
      int ncW = device::process(1, 1, ft, vctracks[0], vctracks[1]); // HERE WE FIT THE VERTICES
      swW.Stop();
      LOG(debug) << "fit wgh.dist " << iev << " NC: " << ncW << " Chi2: " << (ncW ? ft.getChi2AtPCACandidate(0) : -1);
      if (ncW) {
        auto minD = checkResults(outStream, treeName2W, ft, vtxGen, genParent, k0dec);
        meanDW += minD;
        nfoundW++;
      }
    }

    meanDA /= nfoundA ? nfoundA : 1;
    meanDAW /= nfoundAW ? nfoundAW : 1;
    meanDW /= nfoundW ? nfoundW : 1;
    LOG(info) << "Processed " << NTest << " 2-prong vertices: Line : Line";
    LOG(info) << "2-prongs with abs.dist minimization: eff = " << float(nfoundA) / NTest
              << " mean.dist to truth: " << meanDA << " Total time: " << swA.CpuTime();
    LOG(info) << "2-prongs with abs.dist but wghPCA: eff = " << float(nfoundAW) / NTest
              << " mean.dist to truth: " << meanDAW << " Total time: " << swAW.CpuTime();
    LOG(info) << "2-prongs with wgh.dist minimization: eff = " << float(nfoundW) / NTest
              << " mean.dist to truth: " << meanDW << " Total time: " << swW.CpuTime();
    BOOST_CHECK(nfoundA > 0.99 * NTest);
    BOOST_CHECK(nfoundAW > 0.99 * NTest);
    BOOST_CHECK(nfoundW > 0.99 * NTest);
    BOOST_CHECK(meanDA < 0.1);
    BOOST_CHECK(meanDAW < 0.1);
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

    std::string treeName3A = "pr3a", treeName3AW = "pr3aw", treeName3W = "pr3w";
    TStopwatch swA, swAW, swW;
    int nfoundA = 0, nfoundAW = 0, nfoundW = 0;
    double meanDA = 0, meanDAW = 0, meanDW = 0;
    swA.Stop();
    swAW.Stop();
    swW.Stop();
    for (int iev = 0; iev < NTest; iev++) {
      auto genParent = generate(vtxGen, vctracks, bz, genPHS, dch, dchdec, forceQ);

      ft.setUseAbsDCA(true);
      swA.Start(false);
      int ncA = device::process(1, 1, ft, vctracks[0], vctracks[1], vctracks[2]);
      swA.Stop();
      LOG(debug) << "fit abs.dist " << iev << " NC: " << ncA << " Chi2: " << (ncA ? ft.getChi2AtPCACandidate(0) : -1);
      if (ncA) {
        auto minD = checkResults(outStream, treeName3A, ft, vtxGen, genParent, dchdec);
        meanDA += minD;
        nfoundA++;
      }

      ft.setUseAbsDCA(true);
      ft.setWeightedFinalPCA(true);
      swAW.Start(false);
      int ncAW = device::process(1, 1, ft, vctracks[0], vctracks[1], vctracks[2]); // HERE WE FIT THE VERTICES
      swAW.Stop();
      LOG(debug) << "fit abs.dist " << iev << " NC: " << ncAW << " Chi2: " << (ncAW ? ft.getChi2AtPCACandidate(0) : -1);
      if (ncAW) {
        auto minD = checkResults(outStream, treeName3AW, ft, vtxGen, genParent, dchdec);
        meanDAW += minD;
        nfoundAW++;
      }

      ft.setUseAbsDCA(false);
      ft.setWeightedFinalPCA(false);
      swW.Start(false);
      int ncW = device::process(1, 1, ft, vctracks[0], vctracks[1], vctracks[2]); // HERE WE FIT THE VERTICES
      swW.Stop();
      LOG(debug) << "fit wgh.dist " << iev << " NC: " << ncW << " Chi2: " << (ncW ? ft.getChi2AtPCACandidate(0) : -1);
      if (ncW) {
        auto minD = checkResults(outStream, treeName3W, ft, vtxGen, genParent, dchdec);
        meanDW += minD;
        nfoundW++;
      }
    }

    meanDA /= nfoundA ? nfoundA : 1;
    meanDAW /= nfoundAW ? nfoundAW : 1;
    meanDW /= nfoundW ? nfoundW : 1;
    LOG(info) << "Processed " << NTest << " 3-prong vertices";
    LOG(info) << "3-prongs with abs.dist minimization: eff = " << float(nfoundA) / NTest
              << " mean.dist to truth: " << meanDA << " Total time: " << swA.CpuTime();
    LOG(info) << "3-prongs with abs.dist but wghPCA: eff = " << float(nfoundAW) / NTest
              << " mean.dist to truth: " << meanDAW << " Total time: " << swAW.CpuTime();
    LOG(info) << "3-prongs with wgh.dist minimization: eff = " << float(nfoundW) / NTest
              << " mean.dist to truth: " << meanDW << " Total time: " << swW.CpuTime();
    BOOST_CHECK(nfoundA > 0.99 * NTest);
    BOOST_CHECK(nfoundAW > 0.99 * NTest);
    BOOST_CHECK(nfoundW > 0.99 * NTest);
    BOOST_CHECK(meanDA < 0.1);
    BOOST_CHECK(meanDAW < 0.1);
    BOOST_CHECK(meanDW < 0.1);
  }

  outStream.Close();
}
#endif

BOOST_AUTO_TEST_CASE(DCAFitterNProngsBulk)
{
  const char* nThreadsEnvVarName = "DCAFITTERGPU_TEST_NTHREADS";
  const char* nBlocksEnvVarName = "DCAFITTERGPU_TEST_NBLOCKS";
  const char* nBatchesEnvVarName = "DCAFITTERGPU_TEST_NBATCHES";
  const char* nTestsEnvVarName = "DCAFITTERGPU_TEST_NTESTS";
  int nBlocks = std::getenv(nThreadsEnvVarName) == nullptr ? 30 : std::stoi(std::getenv(nThreadsEnvVarName));
  int nThreads = std::getenv(nBlocksEnvVarName) == nullptr ? 256 : std::stoi(std::getenv(nBlocksEnvVarName));
  int nBatches = std::getenv(nBatchesEnvVarName) == nullptr ? 8 : std::stoi(std::getenv(nBatchesEnvVarName));
  int NTest = std::getenv(nTestsEnvVarName) == nullptr ? 100001 : std::stoi(std::getenv(nTestsEnvVarName));

  o2::utils::TreeStreamRedirector outStreamB("dcafitterNTestBulk.root");

  TGenPhaseSpace genPHS;
  constexpr double ele = 0.00051;
  constexpr double gamma = 2 * ele + 1e-6;
  constexpr double pion = 0.13957;
  constexpr double k0 = 0.49761;
  constexpr double kch = 0.49368;
  constexpr double dch = 1.86965;
  std::vector<double> gammadec = {ele, ele};
  std::vector<double> k0dec = {pion, pion};
  std::vector<double> dchdec = {pion, kch, pion};
  std::vector<std::vector<o2::track::TrackParCov>> vctracks(3, std::vector<o2::track::TrackParCov>(NTest));
  std::vector<Vec3D> vtxGen(NTest);

  double bz = 5.0;
  { // 2 prongs vertices bulk processing
    LOG(info) << "\n\nBulk-processing 2-prong Helix - Helix case";
    std::vector<int> forceQ{1, 1};

    o2::vertexing::DCAFitterN<2> ft; // 2 prong fitter
    ft.setBz(bz);
    ft.setPropagateToPCA(true);  // After finding the vertex, propagate tracks to the DCA. This is default anyway
    ft.setMaxR(200);             // do not consider V0 seeds with 2D circles crossing above this R. This is default anyway
    ft.setMaxDZIni(4);           // do not consider V0 seeds with tracks Z-distance exceeding this. This is default anyway
    ft.setMaxDXYIni(4);          // do not consider V0 seeds with tracks XY-distance exceeding this. This is default anyway
    ft.setMinParamChange(1e-3);  // stop iterations if max correction is below this value. This is default anyway
    ft.setMinRelChi2Change(0.9); // stop iterations if chi2 improves by less that this factor

    std::vector<o2::vertexing::DCAFitterN<2>> fitters_host(NTest);
    std::vector<TLorentzVector> genParents(NTest);

    std::string treeName2Abulk = "pr2aBulk", treeName2AWbulk = "pr2awBulk", treeName2Wbulk = "pr2wBulk";
    TStopwatch swAb, swAWb, swWb;
    int nfoundAb = 0, nfoundAWb = 0, nfoundWb = 0;
    double meanDAb = 0, meanDAWb = 0, meanDWb = 0;
    swAb.Stop();
    swAWb.Stop();
    swWb.Stop();

    ft.setUseAbsDCA(true);
    std::fill(fitters_host.begin(), fitters_host.end(), ft);
    for (int iev = 0; iev < NTest; iev++) {
      std::vector<o2::track::TrackParCov> vc(2);
      genParents[iev] = generate(vtxGen[iev], vc, bz, genPHS, k0, k0dec, forceQ);
      vctracks[0][iev] = vc[0];
      vctracks[1][iev] = vc[1];
    }

    swAb.Start(false);
    std::vector<int> ncAb(NTest, 0);
    device::processBulk(nBlocks, nThreads, nBatches, fitters_host, ncAb, vctracks[0], vctracks[1]); // HERE WE FIT THE VERTICES
    swAb.Stop();

    for (int iev = 0; iev < NTest; iev++) {
      LOG(debug) << "fit abs.dist " << iev << " NC: " << ncAb[iev] << " Chi2: " << (ncAb[iev] ? fitters_host[iev].getChi2AtPCACandidate(0) : -1);
      if (ncAb[iev]) {
        auto minDb = checkResults(outStreamB, treeName2Abulk, fitters_host[iev], vtxGen[iev], genParents[iev], k0dec);
        meanDAb += minDb;
        nfoundAb++;
      }
    }

    ft.setUseAbsDCA(true);
    ft.setWeightedFinalPCA(true);
    std::fill(fitters_host.begin(), fitters_host.end(), ft);
    swAWb.Start(false);
    std::vector<int> ncAWb(NTest, 0);
    device::processBulk(nBlocks, nThreads, nBatches, fitters_host, ncAWb, vctracks[0], vctracks[1]); // HERE WE FIT THE VERTICES
    swAWb.Stop();

    for (int iev = 0; iev < NTest; iev++) {
      LOG(debug) << "fit abs.dist with final weighted DCA " << iev << " NC: " << ncAWb[iev] << " Chi2: " << (ncAWb[iev] ? fitters_host[iev].getChi2AtPCACandidate(0) : -1);
      if (ncAWb[iev]) {
        auto minDb = checkResults(outStreamB, treeName2AWbulk, fitters_host[iev], vtxGen[iev], genParents[iev], k0dec);
        meanDAWb += minDb;
        nfoundAWb++;
      }
    }

    ft.setUseAbsDCA(false);
    ft.setWeightedFinalPCA(false);
    std::fill(fitters_host.begin(), fitters_host.end(), ft);
    swWb.Start(false);
    std::vector<int> ncWb(NTest, 0);
    device::processBulk(nBlocks, nThreads, nBatches, fitters_host, ncWb, vctracks[0], vctracks[1]); // HERE WE FIT THE VERTICES
    swWb.Stop();

    for (int iev = 0; iev < NTest; iev++) {
      LOG(debug) << "fit wgh.dist " << iev << " NC: " << ncWb[iev] << " Chi2: " << (ncWb[iev] ? fitters_host[iev].getChi2AtPCACandidate(0) : -1);
      if (ncWb[iev]) {
        auto minDb = checkResults(outStreamB, treeName2Wbulk, fitters_host[iev], vtxGen[iev], genParents[iev], k0dec);
        meanDWb += minDb;
        nfoundWb++;
      }
    }
    //
    meanDAb /= nfoundAb ? nfoundAb : 1;
    meanDAWb /= nfoundAWb ? nfoundAWb : 1;
    meanDWb /= nfoundWb ? nfoundWb : 1;
    LOGP(info, "Bulk-processed {} 2-prong vertices Helix : Helix", NTest);
    LOG(info) << "2-prongs with abs.dist minimization: eff = " << float(nfoundAb) / NTest
              << " mean.dist to truth: " << meanDAb << " Total time: " << swAb.CpuTime() * 1000 << " ms";
    LOG(info) << "2-prongs with abs.dist but wghPCA: eff = " << float(nfoundAWb) / NTest
              << " mean.dist to truth: " << meanDAWb << " Total time: " << swAWb.CpuTime() * 1000 << " ms";
    LOG(info) << "2-prongs with wgh.dist minimization: eff = " << float(nfoundWb) / NTest
              << " mean.dist to truth: " << meanDWb << " Total time: " << swWb.CpuTime() * 1000 << " ms";
    BOOST_CHECK(nfoundAb > 0.99 * NTest);
    BOOST_CHECK(nfoundAWb > 0.99 * NTest);
    BOOST_CHECK(nfoundWb > 0.99 * NTest);
    BOOST_CHECK(meanDAb < 0.1);
    BOOST_CHECK(meanDAWb < 0.1);
    BOOST_CHECK(meanDWb < 0.1);
  }

  { // 2 prongs vertices bulk processing for gamma conversion
    LOG(info) << "\n\nBulk-processing 2-prong Helix - Helix case gamma conversion";
    std::vector<int> forceQ{1, 1};

    o2::vertexing::DCAFitterN<2> ft; // 2 prong fitter
    ft.setBz(bz);
    ft.setPropagateToPCA(true);  // After finding the vertex, propagate tracks to the DCA. This is default anyway
    ft.setMaxR(200);             // do not consider V0 seeds with 2D circles crossing above this R. This is default anyway
    ft.setMaxDZIni(4);           // do not consider V0 seeds with tracks Z-distance exceeding this. This is default anyway
    ft.setMaxDXYIni(4);          // do not consider V0 seeds with tracks XY-distance exceeding this. This is default anyway
    ft.setMinParamChange(1e-3);  // stop iterations if max correction is below this value. This is default anyway
    ft.setMinRelChi2Change(0.9); // stop iterations if chi2 improves by less that this factor

    std::vector<o2::vertexing::DCAFitterN<2>> fitters_host(NTest);
    std::vector<TLorentzVector> genParents(NTest);

    std::string treeName2Abulk = "gpr2aBulk", treeName2AWbulk = "gpr2awBulk", treeName2Wbulk = "gpr2wBulk";
    TStopwatch swAb, swAWb, swWb;
    int nfoundAb = 0, nfoundAWb = 0, nfoundWb = 0;
    double meanDAb = 0, meanDAWb = 0, meanDWb = 0;
    swAb.Stop();
    swAWb.Stop();
    swWb.Stop();

    ft.setUseAbsDCA(true);
    std::fill(fitters_host.begin(), fitters_host.end(), ft);
    for (int iev = 0; iev < NTest; iev++) {
      std::vector<o2::track::TrackParCov> vc(2);
      genParents[iev] = generate(vtxGen[iev], vc, bz, genPHS, gamma, gammadec, forceQ);
      vctracks[0][iev] = vc[0];
      vctracks[1][iev] = vc[1];
    }

    swAb.Start(false);
    std::vector<int> ncAb(NTest, 0);
    device::processBulk(nBlocks, nThreads, nBatches, fitters_host, ncAb, vctracks[0], vctracks[1]); // HERE WE FIT THE VERTICES
    swAb.Stop();

    for (int iev = 0; iev < NTest; iev++) {
      LOG(debug) << "fit abs.dist " << iev << " NC: " << ncAb[iev] << " Chi2: " << (ncAb[iev] ? fitters_host[iev].getChi2AtPCACandidate(0) : -1);
      if (ncAb[iev]) {
        auto minDb = checkResults(outStreamB, treeName2Abulk, fitters_host[iev], vtxGen[iev], genParents[iev], gammadec);
        meanDAb += minDb;
        nfoundAb++;
      }
    }
    //
    ft.setUseAbsDCA(true);
    ft.setWeightedFinalPCA(true);
    std::fill(fitters_host.begin(), fitters_host.end(), ft);
    swAWb.Start(false);
    std::vector<int> ncAWb(NTest, 0);
    device::processBulk(nBlocks, nThreads, nBatches, fitters_host, ncAWb, vctracks[0], vctracks[1]); // HERE WE FIT THE VERTICES
    swAWb.Stop();

    for (int iev = 0; iev < NTest; iev++) {
      LOG(debug) << "fit abs.dist with final weighted DCA " << iev << " NC: " << ncAWb[iev] << " Chi2: " << (ncAWb[iev] ? fitters_host[iev].getChi2AtPCACandidate(0) : -1);
      if (ncAWb[iev]) {
        auto minDb = checkResults(outStreamB, treeName2AWbulk, fitters_host[iev], vtxGen[iev], genParents[iev], gammadec);
        meanDAWb += minDb;
        nfoundAWb++;
      }
    }

    ft.setUseAbsDCA(false);
    ft.setWeightedFinalPCA(false);
    std::fill(fitters_host.begin(), fitters_host.end(), ft);
    swWb.Start(false);
    std::vector<int> ncWb(NTest, 0);
    device::processBulk(nBlocks, nThreads, nBatches, fitters_host, ncWb, vctracks[0], vctracks[1]); // HERE WE FIT THE VERTICES
    swWb.Stop();

    for (int iev = 0; iev < NTest; iev++) {
      LOG(debug) << "fit wgh.dist " << iev << " NC: " << ncWb[iev] << " Chi2: " << (ncWb[iev] ? fitters_host[iev].getChi2AtPCACandidate(0) : -1);
      if (ncWb[iev]) {
        auto minDb = checkResults(outStreamB, treeName2Wbulk, fitters_host[iev], vtxGen[iev], genParents[iev], gammadec);
        meanDWb += minDb;
        nfoundWb++;
      }
    }
    //

    meanDAb /= nfoundAb ? nfoundAb : 1;
    meanDAWb /= nfoundAWb ? nfoundAWb : 1;
    meanDWb /= nfoundWb ? nfoundWb : 1;
    LOGP(info, "Bulk-processed {} 2-prong vertices Helix : Helix from gamma conversion", NTest);
    LOG(info) << "2-prongs with abs.dist minimization: eff = " << float(nfoundAb) / NTest
              << " mean.dist to truth: " << meanDAb << " Total time: " << swAb.CpuTime() * 1000 << " ms";
    LOG(info) << "2-prongs with abs.dist but wghPCA: eff = " << float(nfoundAWb) / NTest
              << " mean.dist to truth: " << meanDAWb << " Total time: " << swAWb.CpuTime() * 1000 << " ms";
    LOG(info) << "2-prongs with wgh.dist minimization: eff = " << float(nfoundWb) / NTest
              << " mean.dist to truth: " << meanDWb << " Total time: " << swWb.CpuTime() * 1000 << " ms";
    BOOST_CHECK(nfoundAb > 0.99 * NTest);
    BOOST_CHECK(nfoundAWb > 0.99 * NTest);
    BOOST_CHECK(nfoundWb > 0.99 * NTest);
    BOOST_CHECK(meanDAb < 2.1);
    BOOST_CHECK(meanDAWb < 2.1);
    BOOST_CHECK(meanDWb < 2.1);
  }

  // 2 prongs vertices bulk processing with one of charges set to 0: Helix : Line
  {
    std::vector<int> forceQ{1, 1};
    LOG(info) << "\n\nBulk-processing 2-prong Helix - Line case";
    o2::vertexing::DCAFitterN<2> ft; // 2 prong fitter
    ft.setBz(bz);
    ft.setPropagateToPCA(true);  // After finding the vertex, propagate tracks to the DCA. This is default anyway
    ft.setMaxR(200);             // do not consider V0 seeds with 2D circles crossing above this R. This is default anyway
    ft.setMaxDZIni(4);           // do not consider V0 seeds with tracks Z-distance exceeding this. This is default anyway
    ft.setMinParamChange(1e-3);  // stop iterations if max correction is below this value. This is default anyway
    ft.setMinRelChi2Change(0.9); // stop iterations if chi2 improves by less that this factor

    std::vector<o2::vertexing::DCAFitterN<2>> fitters_host(NTest);
    std::vector<TLorentzVector> genParents(NTest);

    std::string treeName2Abulk = "pr2aHLb", treeName2AWbulk = "pr2awHLb", treeName2Wbulk = "pr2wHLb";
    TStopwatch swAb, swAWb, swWb;
    int nfoundAb = 0, nfoundAWb = 0, nfoundWb = 0;
    double meanDAb = 0, meanDAWb = 0, meanDWb = 0;
    swAb.Stop();
    swAWb.Stop();
    swWb.Stop();

    for (int iev = 0; iev < NTest; iev++) {
      forceQ[iev % 2] = 1;
      forceQ[1 - iev % 2] = 0;
      std::vector<o2::track::TrackParCov> vc(2);
      genParents[iev] = generate(vtxGen[iev], vc, bz, genPHS, k0, k0dec, forceQ);
      vctracks[0][iev] = vc[0];
      vctracks[1][iev] = vc[1];
    }
    ft.setUseAbsDCA(true);
    std::fill(fitters_host.begin(), fitters_host.end(), ft);

    swAb.Start(false);
    std::vector<int> ncAb(NTest, 0);
    device::processBulk(nBlocks, nThreads, nBatches, fitters_host, ncAb, vctracks[0], vctracks[1]); // HERE WE FIT THE VERTICES
    swAb.Stop();

    for (int iev = 0; iev < NTest; iev++) {
      LOG(debug) << "fit abs.dist with final weighted DCA " << iev << " NC: " << ncAb[iev] << " Chi2: " << (ncAb[iev] ? fitters_host[iev].getChi2AtPCACandidate(0) : -1);
      if (ncAb[iev]) {
        auto minDb = checkResults(outStreamB, treeName2Abulk, fitters_host[iev], vtxGen[iev], genParents[iev], k0dec);
        meanDAb += minDb;
        nfoundAb++;
      }
    }

    ft.setUseAbsDCA(true);
    ft.setWeightedFinalPCA(true);
    std::fill(fitters_host.begin(), fitters_host.end(), ft);
    swAWb.Start(false);
    std::vector<int> ncAWb(NTest, 0);
    device::processBulk(nBlocks, nThreads, nBatches, fitters_host, ncAWb, vctracks[0], vctracks[1]); // HERE WE FIT THE VERTICES
    swAWb.Stop();

    for (int iev = 0; iev < NTest; iev++) {
      LOG(debug) << "fit abs.dist  " << iev << " NC: " << ncAWb[iev] << " Chi2: " << (ncAWb[iev] ? fitters_host[iev].getChi2AtPCACandidate(0) : -1);
      if (ncAWb[iev]) {
        auto minDb = checkResults(outStreamB, treeName2AWbulk, fitters_host[iev], vtxGen[iev], genParents[iev], k0dec);
        meanDAWb += minDb;
        nfoundAWb++;
      }
    }

    ft.setUseAbsDCA(false);
    ft.setWeightedFinalPCA(false);
    std::fill(fitters_host.begin(), fitters_host.end(), ft);
    swWb.Start(false);
    std::vector<int> ncWb(NTest, 0);
    device::processBulk(nBlocks, nThreads, nBatches, fitters_host, ncWb, vctracks[0], vctracks[1]); // HERE WE FIT THE VERTICES
    swWb.Stop();

    for (int iev = 0; iev < NTest; iev++) {
      LOG(debug) << "fit wgh.dist " << iev << " NC: " << ncWb[iev] << " Chi2: " << (ncWb[iev] ? fitters_host[iev].getChi2AtPCACandidate(0) : -1);
      if (ncWb[iev]) {
        auto minDb = checkResults(outStreamB, treeName2Wbulk, fitters_host[iev], vtxGen[iev], genParents[iev], k0dec);
        meanDWb += minDb;
        nfoundWb++;
      }
    }

    //
    meanDAb /= nfoundAb ? nfoundAb : 1;
    meanDAWb /= nfoundAWb ? nfoundAWb : 1;
    meanDWb /= nfoundWb ? nfoundWb : 1;
    LOG(info) << "Bulk-processed " << NTest << " 2-prong vertices: Helix : Line";
    LOG(info) << "2-prongs with abs.dist minimization: eff = " << float(nfoundAb) / NTest
              << " mean.dist to truth: " << meanDAb << " Total time: " << swAb.CpuTime() * 1000 << " ms";
    LOG(info) << "2-prongs with abs.dist but wghPCA: eff = " << float(nfoundAWb) / NTest
              << " mean.dist to truth: " << meanDAWb << " Total time: " << swAWb.CpuTime() * 1000 << " ms";
    LOG(info) << "2-prongs with wgh.dist minimization: eff = " << float(nfoundWb) / NTest
              << " mean.dist to truth: " << meanDWb << " Total time: " << swWb.CpuTime() * 1000 << " ms";
    BOOST_CHECK(nfoundAb > 0.99 * NTest);
    BOOST_CHECK(nfoundAWb > 0.99 * NTest);
    BOOST_CHECK(nfoundWb > 0.99 * NTest);
    BOOST_CHECK(meanDAb < 0.1);
    BOOST_CHECK(meanDAWb < 0.1);
    BOOST_CHECK(meanDWb < 0.1);
  }

  // 2 prongs vertices with both of charges set to 0: Line : Line
  {
    std::vector<int> forceQ{0, 0};
    LOG(info) << "\n\nBulk-processing 2-prong Line - Line case";
    o2::vertexing::DCAFitterN<2> ft; // 2 prong fitter
    ft.setBz(bz);
    ft.setPropagateToPCA(true);  // After finding the vertex, propagate tracks to the DCA. This is default anyway
    ft.setMaxR(200);             // do not consider V0 seeds with 2D circles crossing above this R. This is default anyway
    ft.setMaxDZIni(4);           // do not consider V0 seeds with tracks Z-distance exceeding this. This is default anyway
    ft.setMinParamChange(1e-3);  // stop iterations if max correction is below this value. This is default anyway
    ft.setMinRelChi2Change(0.9); // stop iterations if chi2 improves by less that this factor

    std::vector<o2::vertexing::DCAFitterN<2>> fitters_host(NTest);
    std::vector<TLorentzVector> genParents(NTest);

    std::string treeName2Abulk = "pr2aLL", treeName2AWbulk = "pr2awLL", treeName2Wbulk = "pr2wLL";
    TStopwatch swAb, swAWb, swWb;
    int nfoundAb = 0, nfoundAWb = 0, nfoundWb = 0;
    double meanDAb = 0, meanDAWb = 0, meanDWb = 0;
    swAb.Stop();
    swAWb.Stop();
    swWb.Stop();
    for (int iev = 0; iev < NTest; iev++) {
      forceQ[0] = forceQ[1] = 0;
      std::vector<o2::track::TrackParCov> vc(2);
      genParents[iev] = generate(vtxGen[iev], vc, bz, genPHS, k0, k0dec, forceQ);
      vctracks[0][iev] = vc[0];
      vctracks[1][iev] = vc[1];
    }

    ft.setUseAbsDCA(true);
    std::fill(fitters_host.begin(), fitters_host.end(), ft);

    swAb.Start(false);
    std::vector<int> ncAb(NTest, 0);
    device::processBulk(nBlocks, nThreads, nBatches, fitters_host, ncAb, vctracks[0], vctracks[1]); // HERE WE FIT THE VERTICES
    swAb.Stop();

    for (int iev = 0; iev < NTest; iev++) {
      LOG(debug) << "fit abs.dist " << iev << " NC: " << ncAb[iev] << " Chi2: " << (ncAb[iev] ? fitters_host[iev].getChi2AtPCACandidate(0) : -1);
      if (ncAb[iev]) {
        auto minDb = checkResults(outStreamB, treeName2Abulk, fitters_host[iev], vtxGen[iev], genParents[iev], k0dec);
        meanDAb += minDb;
        nfoundAb++;
      }
    }

    ft.setUseAbsDCA(true);
    ft.setWeightedFinalPCA(true);
    std::fill(fitters_host.begin(), fitters_host.end(), ft);
    swAWb.Start(false);
    std::vector<int> ncAWb(NTest, 0);
    device::processBulk(nBlocks, nThreads, nBatches, fitters_host, ncAWb, vctracks[0], vctracks[1]); // HERE WE FIT THE VERTICES
    swAWb.Stop();
    for (int iev = 0; iev < NTest; iev++) {
      LOG(debug) << "fit abs.dist " << iev << " NC: " << ncAWb[iev] << " Chi2: " << (ncAWb[iev] ? fitters_host[iev].getChi2AtPCACandidate(0) : -1);
      if (ncAWb[iev]) {
        auto minDb = checkResults(outStreamB, treeName2AWbulk, fitters_host[iev], vtxGen[iev], genParents[iev], k0dec);
        meanDAWb += minDb;
        nfoundAWb++;
      }
    }

    ft.setUseAbsDCA(false);
    ft.setWeightedFinalPCA(false);
    std::fill(fitters_host.begin(), fitters_host.end(), ft);

    swWb.Start(false);
    std::vector<int> ncWb(NTest, 0);
    device::processBulk(nBlocks, nThreads, nBatches, fitters_host, ncWb, vctracks[0], vctracks[1]); // HERE WE FIT THE VERTICES
    swWb.Stop();

    for (int iev = 0; iev < NTest; iev++) {
      LOG(debug) << "fit wgh.dist " << iev << " NC: " << ncWb[iev] << " Chi2: " << (ncWb[iev] ? fitters_host[iev].getChi2AtPCACandidate(0) : -1);
      if (ncWb[iev]) {
        auto minDb = checkResults(outStreamB, treeName2Wbulk, fitters_host[iev], vtxGen[iev], genParents[iev], k0dec);
        meanDWb += minDb;
        nfoundWb++;
      }
    }
    // ft.print();
    meanDAb /= nfoundAb ? nfoundAb : 1;
    meanDAWb /= nfoundAWb ? nfoundAWb : 1;
    meanDWb /= nfoundWb ? nfoundWb : 1;
    LOG(info) << "Bulk-processed " << NTest << " 2-prong vertices: Line : Line";
    LOG(info) << "2-prongs with abs.dist minimization: eff = " << float(nfoundAb) / NTest
              << " mean.dist to truth: " << meanDAb << " Total time: " << swAb.CpuTime() * 1000 << " ms";
    LOG(info) << "2-prongs with abs.dist but wghPCA: eff = " << float(nfoundAWb) / NTest
              << " mean.dist to truth: " << meanDAWb << " Total time: " << swAWb.CpuTime() * 1000 << " ms";
    LOG(info) << "2-prongs with wgh.dist minimization: eff = " << float(nfoundWb) / NTest
              << " mean.dist to truth: " << meanDWb << " Total time: " << swWb.CpuTime() * 1000 << " ms";
    BOOST_CHECK(nfoundAb > 0.99 * NTest);
    BOOST_CHECK(nfoundAWb > 0.99 * NTest);
    BOOST_CHECK(nfoundWb > 0.99 * NTest);
    BOOST_CHECK(meanDAb < 0.1);
    BOOST_CHECK(meanDAWb < 0.1);
    BOOST_CHECK(meanDWb < 0.1);
  }

  // Bulk-process 3 prongs vertices
  {
    LOG(info) << "\n\nBulk-processing 3-prongs";
    std::vector<int> forceQ{1, 1, 1};

    o2::vertexing::DCAFitterN<3> ft; // 3 prong fitter
    ft.setBz(bz);
    ft.setPropagateToPCA(true);  // After finding the vertex, propagate tracks to the DCA. This is default anyway
    ft.setMaxR(200);             // do not consider V0 seeds with 2D circles crossing above this R. This is default anyway
    ft.setMaxDZIni(4);           // do not consider V0 seeds with tracks Z-distance exceeding this. This is default anyway
    ft.setMinParamChange(1e-3);  // stop iterations if max correction is below this value. This is default anyway
    ft.setMinRelChi2Change(0.9); // stop iterations if chi2 improves by less that this factor

    std::vector<o2::vertexing::DCAFitterN<3>> fitters_host(NTest);
    std::vector<TLorentzVector> genParents(NTest);

    std::string treeName3Abulk = "pr3a", treeName3AWbulk = "pr3aw", treeName3Wbulk = "pr3w";
    TStopwatch swAb, swAWb, swWb;
    int nfoundAb = 0, nfoundAWb = 0, nfoundWb = 0;
    double meanDAb = 0, meanDAWb = 0, meanDWb = 0;
    swAb.Stop();
    swAWb.Stop();
    swWb.Stop();
    for (int iev = 0; iev < NTest; iev++) {
      std::vector<o2::track::TrackParCov> vc(3);
      genParents[iev] = generate(vtxGen[iev], vc, bz, genPHS, dch, dchdec, forceQ);

      vctracks[0][iev] = vc[0];
      vctracks[1][iev] = vc[1];
      vctracks[2][iev] = vc[2];
    }

    ft.setUseAbsDCA(true);
    std::fill(fitters_host.begin(), fitters_host.end(), ft);
    swAb.Start(false);
    std::vector<int> ncAb(NTest, 0);
    device::processBulk(nBlocks, nThreads, nBatches, fitters_host, ncAb, vctracks[0], vctracks[1], vctracks[2]); // HERE WE FIT THE VERTICES
    swAb.Stop();
    for (int iev = 0; iev < NTest; iev++) {
      LOG(debug) << "fit abs.dist " << iev << " NC: " << ncAb[iev] << " Chi2: " << (ncAb[iev] ? fitters_host[iev].getChi2AtPCACandidate(0) : -1);
      if (ncAb[iev]) {
        auto minDb = checkResults(outStreamB, treeName3Abulk, fitters_host[iev], vtxGen[iev], genParents[iev], dchdec);
        meanDAb += minDb;
        nfoundAb++;
      }
    }

    ft.setUseAbsDCA(true);
    ft.setWeightedFinalPCA(true);
    std::fill(fitters_host.begin(), fitters_host.end(), ft);

    swAWb.Start(false);
    std::vector<int> ncAWb(NTest, 0);
    device::processBulk(nBlocks, nThreads, nBatches, fitters_host, ncAWb, vctracks[0], vctracks[1], vctracks[2]); // HERE WE FIT THE VERTICES
    swAWb.Stop();
    for (int iev = 0; iev < NTest; iev++) {
      LOG(debug) << "fit abs.dist " << iev << " NC: " << ncAWb[iev] << " Chi2: " << (ncAWb[iev] ? fitters_host[iev].getChi2AtPCACandidate(0) : -1);
      if (ncAWb[iev]) {
        auto minDb = checkResults(outStreamB, treeName3AWbulk, fitters_host[iev], vtxGen[iev], genParents[iev], dchdec);
        meanDAWb += minDb;
        nfoundAWb++;
      }
    }

    ft.setUseAbsDCA(false);
    ft.setWeightedFinalPCA(false);
    std::fill(fitters_host.begin(), fitters_host.end(), ft);

    swWb.Start(false);
    std::vector<int> ncWb(NTest, 0);
    device::processBulk(nBlocks, nThreads, nBatches, fitters_host, ncWb, vctracks[0], vctracks[1], vctracks[2]); // HERE WE FIT THE VERTICES
    swWb.Stop();
    for (int iev = 0; iev < NTest; iev++) {
      LOG(debug) << "fit wgh.dist " << iev << " NC: " << ncWb[iev] << " Chi2: " << (ncWb[iev] ? fitters_host[iev].getChi2AtPCACandidate(0) : -1);
      if (ncWb[iev]) {
        auto minDb = checkResults(outStreamB, treeName3Wbulk, fitters_host[iev], vtxGen[iev], genParents[iev], dchdec);
        meanDWb += minDb;
        nfoundWb++;
      }
    }

    // ft.print();
    meanDAb /= nfoundAb ? nfoundAb : 1;
    meanDAWb /= nfoundAWb ? nfoundAWb : 1;
    meanDWb /= nfoundWb ? nfoundWb : 1;
    LOG(info) << "Bulk-processed " << NTest << " 3-prong vertices";
    LOG(info) << "3-prongs with abs.dist minimization: eff = " << float(nfoundAb) / NTest
              << " mean.dist to truth: " << meanDAb << " Total time: " << swAb.CpuTime() * 1000 << " ms";
    LOG(info) << "3-prongs with abs.dist but wghPCA: eff = " << float(nfoundAWb) / NTest
              << " mean.dist to truth: " << meanDAWb << " Total time: " << swAWb.CpuTime() * 1000 << " ms";
    LOG(info) << "3-prongs with wgh.dist minimization: eff = " << float(nfoundWb) / NTest
              << " mean.dist to truth: " << meanDWb << " Total time: " << swWb.CpuTime() * 1000 << " ms";
    BOOST_CHECK(nfoundAb > 0.99 * NTest);
    BOOST_CHECK(nfoundAWb > 0.99 * NTest);
    BOOST_CHECK(nfoundWb > 0.99 * NTest);
    BOOST_CHECK(meanDAb < 0.1);
    BOOST_CHECK(meanDAWb < 0.1);
    BOOST_CHECK(meanDWb < 0.1);
  }
  outStreamB.Close();
}

} // namespace vertexing
} // namespace o2