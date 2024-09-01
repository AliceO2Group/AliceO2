#include "DCAFitter/DCAFitterN.h"
#include "CommonUtils/TreeStreamRedirector.h"
#include <TRandom.h>
#include <TGenPhaseSpace.h>
#include <TLorentzVector.h>
#include <TStopwatch.h>
#include <Math/SVector.h>
#include <array>

#include "../DCAFitterGPUAPI.h"

namespace o2::vertexing
{
using Vec3D = ROOT::Math::SVector<double, 3>;
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

int run()
{
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
  std::vector<int> forceQ{1, 1};

  o2::vertexing::DCAFitterN<2> ft; // 2 prong fitter
  ft.setBz(bz);
  ft.setPropagateToPCA(true);  // After finding the vertex, propagate tracks to the DCA. This is default anyway
  ft.setMaxR(200);             // do not consider V0 seeds with 2D circles crossing above this R. This is default anyway
  ft.setMaxDZIni(4);           // do not consider V0 seeds with tracks Z-distance exceeding this. This is default anyway
  ft.setMaxDXYIni(4);          // do not consider V0 seeds with tracks XY-distance exceeding this. This is default anyway
  ft.setMinParamChange(1e-3);  // stop iterations if max correction is below this value. This is default anyway
  ft.setMinRelChi2Change(0.9); // stop iterations if chi2 improves by less that this factor

  auto genParent = generate(vtxGen, vctracks, bz, genPHS, k0, k0dec, forceQ);
  ft.setUseAbsDCA(true);
  auto res = ft.process(vctracks[0], vctracks[1]);
  ft.print();
  std::cout << "returned value: " << res << std::endl;

  doPrintOnGPU(&ft);
  return 0;
}
} // namespace o2::vertexing

int main() { return o2::vertexing::run(); }