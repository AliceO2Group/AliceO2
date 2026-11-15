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

// Writes the TRD gas collision tables (C++) from Geant4's PAI cross-section: number of collisions per cm with an
// energy transfer below the delta-ray cut, restricted dE/dx, and N(>E)/N on a log energy grid, per beta*gamma.
// Standalone program, not built with O2. Build: g++ -O2 $(geant4-config --cflags) makeIonisationTables.cxx -o makeIonisationTables $(geant4-config --libs)
#include "G4Element.hh"
#include "G4Material.hh"
#include "G4PAIxSection.hh"
#include "G4SandiaTable.hh"
#include "G4SystemOfUnits.hh"
#include "G4Version.hh"
#include <cmath>
#include <cstdio>
#include <string>
#include <vector>

constexpr int kNBetaGamma = 31;
constexpr double kLogBetaGammaMin = -1.4;
constexpr double kLogBetaGammaStep = 0.2;

struct Gas {
  std::string name;
  double z, a, nNoble, nC, nO, fNoble, dNoble;
};

static void table(const Gas& gas, double cut)
{
  const double dco = 0.00186; // CO2 at 20 C, as in Detector::createMaterials
  const double dens = (gas.fNoble * gas.dNoble + (1. - gas.fNoble) * dco) * g / cm3;
  auto* el = new G4Element(gas.name, gas.name, gas.z, gas.a * g / mole);
  auto* c = new G4Element("C_" + gas.name, "C", 6., 12.0107 * g / mole);
  auto* o = new G4Element("O_" + gas.name, "O", 8., 15.9994 * g / mole);
  const double w[3] = {gas.nNoble * gas.a, gas.nC * 12.0107, gas.nO * 15.9994};
  const double ws = w[0] + w[1] + w[2];
  auto* mat = new G4Material("TRD_" + gas.name + "CO2", dens, 3, kStateGas, 293.15 * kelvin);
  mat->AddElement(el, w[0] / ws);
  mat->AddElement(c, w[1] / ws);
  mat->AddElement(o, w[2] / ws);

  std::vector<double> lbgs;
  for (int k = 0; k < kNBetaGamma; ++k) {
    lbgs.push_back(kLogBetaGammaMin + kLogBetaGammaStep * k);
  }
  std::vector<double> egrid;
  std::vector<std::vector<double>> ccdf;
  std::vector<double> nsub, dedx;
  for (double lbg : lbgs) {
    const double bg = std::pow(10., lbg);
    G4SandiaTable sandia;
    sandia.Initialize(mat);
    G4PAIxSection xs;
    xs.Initialize(mat, 1. * MeV, bg * bg, &sandia);
    const int n = xs.GetSplineSize();
    std::vector<double> en(n), in(n), de(n);
    for (int i = 1; i <= n; ++i) {
      en[i - 1] = xs.GetSplineEnergy(i);
      in[i - 1] = xs.GetIntegralPAIxSection(i);
      de[i - 1] = xs.GetIntegralPAIdEdx(i);
    }
    auto interp = [&](const std::vector<double>& v, double e) {
      if (e <= en.front()) {
        return v.front();
      }
      for (int i = 1; i < n; ++i) {
        if (en[i] >= e) {
          const double t = std::log(e / en[i - 1]) / std::log(en[i] / en[i - 1]);
          return v[i - 1] + t * (v[i] - v[i - 1]);
        }
      }
      return v.back();
    };
    if (egrid.empty()) {
      egrid.push_back(en[0] / eV);
      for (int j = 0; j <= 30; ++j) {
        const double e = std::pow(10., 1. + 0.1 * j);
        if (e > en[0] / eV * 1.001) {
          egrid.push_back(e);
        }
      }
    }
    const double ns = (in[0] - interp(in, cut)) * 10.; // per cm
    nsub.push_back(ns);
    dedx.push_back((de[0] - interp(de, cut)) * 10. / eV);
    std::vector<double> row;
    for (double e : egrid) {
      row.push_back(std::max((interp(in, e * eV) - interp(in, cut)) * 10. / ns, 0.));
    }
    row.front() = 1.;
    row.back() = 0.;
    ccdf.push_back(row);
  }
  std::printf("// %s/CO2: density %.6g g/cm3, mean excitation energy %.4g eV\n", gas.name.c_str(), dens / (g / cm3),
              mat->GetIonisation()->GetMeanExcitationEnergy() / eV);
  std::printf("constexpr double k%sEnergy[%zu] = {", gas.name.c_str(), egrid.size());
  for (size_t j = 0; j < egrid.size(); ++j) {
    std::printf("%s%.5g", j ? ", " : "", egrid[j]);
  }
  std::printf("};\nconstexpr double k%sCollisionsPerCm[%zu] = {", gas.name.c_str(), nsub.size());
  for (size_t k = 0; k < nsub.size(); ++k) {
    std::printf("%s%.6g", k ? ", " : "", nsub[k]);
  }
  std::printf("};\nconstexpr double k%sRestrictedDEdx[%zu] = {", gas.name.c_str(), dedx.size());
  for (size_t k = 0; k < dedx.size(); ++k) {
    std::printf("%s%.6g", k ? ", " : "", dedx[k]);
  }
  std::printf("};\nconstexpr float k%sFractionAbove[%zu][%zu] = {\n", gas.name.c_str(), ccdf.size(), egrid.size());
  for (auto& row : ccdf) {
    std::printf("  {");
    for (size_t j = 0; j < row.size(); ++j) {
      std::printf("%s%.5g", j ? ", " : "", row[j]);
    }
    std::printf("},\n");
  }
  std::printf("};\n");
}

int main()
{
  const double cut = 10. * keV;
  std::printf("// Generated with Geant4 %d G4PAIxSection, delta-ray cut %.4g keV\n", G4VERSION_NUMBER, cut / keV);
  std::printf("constexpr double kDeltaRayCut = %.6g; // eV\n", cut / eV);
  std::printf("constexpr int kNBetaGamma = %d;\n", kNBetaGamma);
  std::printf("constexpr double kLogBetaGammaMin = %.6g; // beta*gamma of row k is 10^(min + k * step)\n", kLogBetaGammaMin);
  std::printf("constexpr double kLogBetaGammaStep = %.6g;\n", kLogBetaGammaStep);
  table({"Xe", 54., 131.29, 8.5, 1.5, 3.0, 0.85, 0.00549}, cut);
  table({"Ar", 18., 39.948, 8.2, 1.8, 3.6, 0.82, 0.00166}, cut);
  return 0;
}
