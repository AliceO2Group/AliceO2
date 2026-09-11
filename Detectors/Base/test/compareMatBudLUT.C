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

/// \file compareMatBudLUT.C
/// \brief Compare two material budget LUTs cell by cell
///
/// Used to check that filling the LUT in parallel gives the same result as filling it serially,
/// or that the VecGeom and ROOT geometry backends agree within a given tolerance:
///
///   root -b -q 'compareMatBudLUT.C("matbud_serial.root","matbud_parallel.root")'
///   root -b -q 'compareMatBudLUT.C("matbud_ROOT.root","matbud_VECGEOM.root", 0.01, 20, "sweep.csv")'

#if !defined(__CLING__) || defined(__ROOTCLING__)
#include "DetectorsBase/MatLayerCylSet.h"
#include "GPUCommonLogger.h"
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <fstream>
#include <vector>
#endif

namespace
{
struct CellDiff {
  int layer, iz, ip;
  float rhoA, rhoB, x2x0A, x2x0B;
  double rRho, rX;
  double score() const { return std::max(rRho, rX); }
};

struct LayerStat {
  float rMin = 0.f, rMax = 0.f;
  size_t nBad = 0;
  double maxRelRho = 0., maxRelX2X0 = 0.;
};
} // namespace

/// Returns true if the two LUTs agree everywhere within tol (relative).
/// nWorst: number of worst-offending cells to print, ranked by max(relRho, relX2X0) over the
///         whole comparison, not just the first ones found in scan order.
/// csvSummary: if non-empty, append one summary row to this CSV file (header written once).
bool compareMatBudLUT(const std::string& fileA = "matbud_serial.root",
                      const std::string& fileB = "matbud_parallel.root",
                      float tol = 0.f,
                      int nWorst = 10,
                      const std::string& csvSummary = "")
{
  auto* lutA = o2::base::MatLayerCylSet::loadFromFile(fileA);
  auto* lutB = o2::base::MatLayerCylSet::loadFromFile(fileB);
  if (!lutA) {
    LOG(error) << "Failed to load LUT from " << fileA;
    return false;
  }
  if (!lutB) {
    LOG(error) << "Failed to load LUT from " << fileB;
    return false;
  }

  if (lutA->getNLayers() != lutB->getNLayers()) {
    LOG(error) << "Layer count differs: " << lutA->getNLayers() << " vs " << lutB->getNLayers();
    return false;
  }

  size_t nCells = 0, nBad = 0;
  double maxRelRho = 0., maxRelX2X0 = 0.;
  std::vector<LayerStat> layerStats(lutA->getNLayers());
  std::vector<CellDiff> allCells;

  for (int il = 0; il < lutA->getNLayers(); il++) {
    const auto& la = lutA->getLayer(il);
    const auto& lb = lutB->getLayer(il);
    if (la.getNZBins() != lb.getNZBins() || la.getNPhiBins() != lb.getNPhiBins()) {
      LOG(error) << "Layer " << il << " segmentation differs: "
                 << la.getNZBins() << "x" << la.getNPhiBins() << " vs "
                 << lb.getNZBins() << "x" << lb.getNPhiBins();
      return false;
    }
    auto& ls = layerStats[il];
    ls.rMin = la.getRMin();
    ls.rMax = la.getRMax();

    for (int iz = 0; iz < la.getNZBins(); iz++) {
      for (int ip = 0; ip < la.getNPhiBins(); ip++) {
        const auto& ca = la.getCellPhiBin(ip, iz);
        const auto& cb = lb.getCellPhiBin(ip, iz);
        nCells++;

        auto rel = [](float a, float b) {
          const float den = std::max(std::abs(a), std::abs(b));
          return den > 0.f ? std::abs(a - b) / den : 0.f;
        };
        const double rRho = rel(ca.meanRho, cb.meanRho);
        const double rX = rel(ca.meanX2X0, cb.meanX2X0);
        maxRelRho = std::max(maxRelRho, rRho);
        maxRelX2X0 = std::max(maxRelX2X0, rX);
        ls.maxRelRho = std::max(ls.maxRelRho, rRho);
        ls.maxRelX2X0 = std::max(ls.maxRelX2X0, rX);

        if (rRho > tol || rX > tol) {
          ls.nBad++;
          nBad++;
        }
        allCells.push_back({il, iz, ip, ca.meanRho, cb.meanRho, ca.meanX2X0, cb.meanX2X0, rRho, rX});
      }
    }
  }

  const int nw = std::min<int>(nWorst, (int)allCells.size());
  std::partial_sort(allCells.begin(), allCells.begin() + nw, allCells.end(),
                    [](const CellDiff& a, const CellDiff& b) { return a.score() > b.score(); });
  printf("--- %d worst cells (by max relative deviation) ---\n", nw);
  for (int i = 0; i < nw; i++) {
    const auto& c = allCells[i];
    printf("Lr %3d iz %4d ip %4d : rho %.9g vs %.9g (rel %.3g) | x2x0 %.9g vs %.9g (rel %.3g)\n",
           c.layer, c.iz, c.ip, c.rhoA, c.rhoB, c.rRho, c.x2x0A, c.x2x0B, c.rX);
  }

  printf("--- per-layer summary (%d layers) ---\n", lutA->getNLayers());
  for (int il = 0; il < lutA->getNLayers(); il++) {
    const auto& ls = layerStats[il];
    printf("Lr %3d %8.3f<R<%8.3f : nBad %6zu  maxRelRho %.3g  maxRelX2X0 %.3g\n",
           il, ls.rMin, ls.rMax, ls.nBad, ls.maxRelRho, ls.maxRelX2X0);
  }

  printf("Compared %zu cells over %d layers\n", nCells, lutA->getNLayers());
  printf("Max relative difference: meanRho %.3g, meanX2X0 %.3g (tolerance %.3g)\n", maxRelRho, maxRelX2X0, tol);

  if (!csvSummary.empty()) {
    const bool writeHeader = gSystem->AccessPathName(csvSummary.c_str()); // true if it does NOT exist
    std::ofstream csv(csvSummary, std::ios::app);
    if (writeHeader) {
      csv << "fileA,fileB,nLayers,nCells,nBad,maxRelRho,maxRelX2X0,tol\n";
    }
    csv << fileA << "," << fileB << "," << lutA->getNLayers() << "," << nCells << "," << nBad << ","
        << maxRelRho << "," << maxRelX2X0 << "," << tol << "\n";
  }

  if (nBad) {
    LOG(error) << nBad << " cells differ beyond tolerance";
    return false;
  }
  LOG(info) << "LUTs agree";
  return true;
}
