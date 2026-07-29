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
/// Used to check that filling the LUT in parallel gives the same result as filling it serially:
///
///   root -b -q 'compareMatBudLUT.C("matbud_serial.root","matbud_parallel.root")'

#if !defined(__CLING__) || defined(__ROOTCLING__)
#include "DetectorsBase/MatLayerCylSet.h"
#include "GPUCommonLogger.h"
#include <cmath>
#include <cstdio>
#endif

/// Returns true if the two LUTs agree everywhere within tol (relative).
bool compareMatBudLUT(const std::string& fileA = "matbud_serial.root",
                      const std::string& fileB = "matbud_parallel.root",
                      float tol = 0.f)
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

  for (int il = 0; il < lutA->getNLayers(); il++) {
    const auto& la = lutA->getLayer(il);
    const auto& lb = lutB->getLayer(il);
    if (la.getNZBins() != lb.getNZBins() || la.getNPhiBins() != lb.getNPhiBins()) {
      LOG(error) << "Layer " << il << " segmentation differs: "
                 << la.getNZBins() << "x" << la.getNPhiBins() << " vs "
                 << lb.getNZBins() << "x" << lb.getNPhiBins();
      return false;
    }
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

        if (rRho > tol || rX > tol) {
          if (nBad < 10) {
            printf("Lr %3d iz %4d ip %4d : rho %.9g vs %.9g (rel %.3g) | x2x0 %.9g vs %.9g (rel %.3g)\n",
                   il, iz, ip, ca.meanRho, cb.meanRho, rRho, ca.meanX2X0, cb.meanX2X0, rX);
          }
          nBad++;
        }
      }
    }
  }

  printf("Compared %zu cells over %d layers\n", nCells, lutA->getNLayers());
  printf("Max relative difference: meanRho %.3g, meanX2X0 %.3g (tolerance %.3g)\n", maxRelRho, maxRelX2X0, tol);
  if (nBad) {
    LOG(error) << nBad << " cells differ beyond tolerance";
    return false;
  }
  LOG(info) << "LUTs agree";
  return true;
}
