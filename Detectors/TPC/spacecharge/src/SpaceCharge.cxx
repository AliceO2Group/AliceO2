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

/// \file  SpaceCharge.cxx
/// \brief Definition of SpaceCharge class
///
/// \author  Matthias Kleiner <mkleiner@ikf.uni-frankfurt.de>
/// \date Aug 21, 2020

#include "TPCSpaceCharge/SpaceCharge.h"
#include "fmt/core.h"
#include "Framework/Logger.h"
#include "TPCSpaceCharge/PoissonSolver.h"
#include "Framework/Logger.h"
#include "TGeoGlobalMagField.h"
#include "TPCBase/ParameterGas.h"
#include "TPCBase/ParameterElectronics.h"
#include "Field/MagneticField.h"
#include "CommonUtils/TreeStreamRedirector.h"
#include "TPCBase/CalDet.h"
#include "TPCBase/Painter.h"
#include "MathUtils/Utils.h"

#include <numeric>
#include <chrono>
#include "TF1.h"
#include "TH3.h"
#include "TH2F.h"
#include "TGraph.h"
#include "TCanvas.h"
#include "TROOT.h"

#if defined(WITH_OPENMP) || defined(_OPENMP)
#include <omp.h>
#else
static inline int omp_get_thread_num() { return 0; }
static inline int omp_get_max_threads() { return 1; }
#endif

templateClassImp(o2::tpc::SpaceCharge);

using namespace o2::tpc;

template <typename DataT>
SpaceCharge<DataT>::SpaceCharge(const DataT omegaTau, const DataT t1, const DataT t2)
{
  ROOT::EnableThreadSafety();
  setOmegaTauT1T2(omegaTau, t1, t2);
};

template <typename DataT>
void SpaceCharge<DataT>::setGrid(const unsigned short nZVertices, const unsigned short nRVertices, const unsigned short nPhiVertices)
{
  o2::conf::ConfigurableParam::setValue<unsigned short>("TPCSpaceChargeParam", "NZVertices", nZVertices);
  o2::conf::ConfigurableParam::setValue<unsigned short>("TPCSpaceChargeParam", "NRVertices", nRVertices);
  o2::conf::ConfigurableParam::setValue<unsigned short>("TPCSpaceChargeParam", "NPhiVertices", nPhiVertices);
}

template <typename DataT>
int SpaceCharge<DataT>::getOMPMaxThreads()
{
  return omp_get_max_threads();
}

template <typename DataT>
void SpaceCharge<DataT>::calculateDistortionsCorrections(const o2::tpc::Side side, const bool calcVectors)
{
  using timer = std::chrono::high_resolution_clock;
  using SC = o2::tpc::SpaceCharge<DataT>;
  if (!mIsChargeSet[side]) {
    LOGP(error, "the charge is not set!");
  }

  const std::array<std::string, 2> sglobalType{"local distortion/correction interpolator", "Electric fields"};
  const std::array<std::string, 2> sglobalDistType{"Standard method", "interpolation of global corrections"};
  const std::array<std::string, 2> sideName{"A", "C"};

  LOGP(info, "====== starting calculation of distortions and corrections for Side {} ======", sideName[side]);
  LOGP(info, "Using {} threads", getNThreads());

  if (getGlobalDistCorrMethod() == SC::GlobalDistCorrMethod::LocalDistCorr) {
    LOGP(info, "calculation of global distortions and corrections are performed by using: {}", sglobalType[0]);
  } else {
    LOGP(info, "calculation of global distortions and corrections are performed by using: {}", sglobalType[1]);
  }

  if (getGlobalDistType() == SC::GlobalDistType::Fast) {
    LOGP(info, "calculation of global distortions performed by following method: {}", sglobalDistType[1]);
  } else if (getGlobalDistType() == SC::GlobalDistType::Standard) {
    LOGP(info, "calculation of global distortions performed by following method: {}", sglobalDistType[0]);
  } else {
    LOGP(info, "skipping calculation of global distortions");
  }

  auto startTotal = timer::now();

  auto start = timer::now();
  poissonSolver(side);
  auto stop = timer::now();
  std::chrono::duration<float> time = stop - start;
  LOGP(info, "Poisson Solver time: {}", time.count());

  start = timer::now();
  calcEField(side);
  stop = timer::now();
  time = stop - start;
  LOGP(info, "electric field calculation time: {}", time.count());

  const auto numEFields = getElectricFieldsInterpolator(side);
  if (getGlobalDistType() == SC::GlobalDistType::Standard) {
    start = timer::now();
    const auto dist = o2::tpc::SpaceCharge<DataT>::Type::Distortions;
    calcLocalDistortionsCorrections(dist, numEFields); // local distortion calculation
    stop = timer::now();
    time = stop - start;
    LOGP(info, "local distortions time: {}", time.count());
  } else {
    LOGP(info, "skipping local distortions (not needed)");
  }

  start = timer::now();
  const auto corr = o2::tpc::SpaceCharge<DataT>::Type::Corrections;
  calcLocalDistortionsCorrections(corr, numEFields); // local correction calculation
  stop = timer::now();
  time = stop - start;
  LOGP(info, "local corrections time: {}", time.count());

  if (calcVectors) {
    start = timer::now();
    calcLocalDistortionCorrectionVector(numEFields);
    stop = timer::now();
    time = stop - start;
    LOGP(info, "local correction/distortion vector time: {}", time.count());
  }

  start = timer::now();
  const auto lCorrInterpolator = getLocalCorrInterpolator(side);
  (getGlobalDistCorrMethod() == SC::GlobalDistCorrMethod::LocalDistCorr) ? calcGlobalCorrections(lCorrInterpolator) : calcGlobalCorrections(numEFields);
  stop = timer::now();
  time = stop - start;
  LOGP(info, "global corrections time: {}", time.count());
  start = timer::now();
  if (getGlobalDistType() == SC::GlobalDistType::Fast) {
    const auto globalCorrInterpolator = getGlobalCorrInterpolator(side);
    calcGlobalDistWithGlobalCorrIterative(globalCorrInterpolator);
  } else if (getGlobalDistType() == SC::GlobalDistType::Standard) {
    const auto lDistInterpolator = getLocalDistInterpolator(side);
    (getGlobalDistCorrMethod() == SC::GlobalDistCorrMethod::LocalDistCorr) ? calcGlobalDistortions(lDistInterpolator) : calcGlobalDistortions(numEFields);
  } else {
  }

  stop = timer::now();
  time = stop - start;
  LOGP(info, "global distortions time: {}", time.count());

  stop = timer::now();
  time = stop - startTotal;
  LOGP(info, "everything is done. Total Time: {}", time.count());
}

template <typename DataT>
DataT SpaceCharge<DataT>::regulateR(const DataT posR, const Side side) const
{
  const DataT minR = getRMinSim(side);
  if (posR < minR) {
    return minR;
  }
  const DataT maxR = getRMaxSim(side);
  if (posR > maxR) {
    return maxR;
  }
  return posR;
}

template <typename DataT>
void SpaceCharge<DataT>::setChargeDensityFromFormula(const AnalyticalFields<DataT>& formulaStruct)
{
  const Side side = formulaStruct.getSide();
  for (size_t iPhi = 0; iPhi < mParamGrid.NPhiVertices; ++iPhi) {
    const DataT phi = getPhiVertex(iPhi, side);
    for (size_t iR = 0; iR < mParamGrid.NRVertices; ++iR) {
      const DataT radius = getRVertex(iR, side);
      for (size_t iZ = 0; iZ < mParamGrid.NZVertices; ++iZ) {
        const DataT z = getZVertex(iZ, side);
        mDensity[side](iZ, iR, iPhi) = formulaStruct.evalDensity(z, radius, phi);
      }
    }
  }
  mIsChargeSet[side] = true;
}

template <typename DataT>
void SpaceCharge<DataT>::setDefaultStaticDistortionsGEMFrameChargeUp(const Side side, const DataT deltaPotential)
{
  std::function<DataT(DataT)> deltaPotFormula = [deltaPotential](const DataT) {
    return deltaPotential;
  };

  setPotentialBoundaryGEMFrameAlongR(deltaPotFormula, side);
  setPotentialBoundaryGEMFrameIROCBottomAlongPhi(deltaPotFormula, side);
  setPotentialBoundaryGEMFrameIROCTopAlongPhi(deltaPotFormula, side);
  setPotentialBoundaryGEMFrameOROC1BottomAlongPhi(deltaPotFormula, side);
  setPotentialBoundaryGEMFrameOROC1TopAlongPhi(deltaPotFormula, side);
  setPotentialBoundaryGEMFrameOROC2BottomAlongPhi(deltaPotFormula, side);
  setPotentialBoundaryGEMFrameOROC2TopAlongPhi(deltaPotFormula, side);
  setPotentialBoundaryGEMFrameOROC3BottomAlongPhi(deltaPotFormula, side);
  setPotentialBoundaryGEMFrameOROC3TopAlongPhi(deltaPotFormula, side);
}

template <typename DataT>
size_t SpaceCharge<DataT>::getPhiBinsGapFrame(const Side side) const
{
  const auto& regInf = Mapper::instance().getPadRegionInfo(0);
  const float localYEdgeIROC = regInf.getPadsInRowRegion(0) / 2 * regInf.getPadWidth();
  const auto globalPosGap = Mapper::LocalToGlobal(LocalPosition2D(regInf.getRadiusFirstRow(), -(localYEdgeIROC + GEMFrameParameters<DataT>::WIDTHFRAME)), Sector(0));
  const auto phiGap = std::atan(globalPosGap.Y() / globalPosGap.X());

  auto nBinsPhiGap = getNearestPhiVertex(phiGap, side);
  if (nBinsPhiGap == 0) {
    nBinsPhiGap = 1;
  }

  return nBinsPhiGap;
}

template <typename DataT>
void SpaceCharge<DataT>::setPotentialBoundaryGEMFrameAlongR(const std::function<DataT(DataT)>& potentialFunc, const Side side)
{
  const auto radiusStart = std::sqrt(std::pow(GEMFrameParameters<DataT>::LENGTHFRAMEIROCBOTTOM / 2, 2) + std::pow(GEMFrameParameters<DataT>::POSBOTTOM[0], 2));
  const auto rStart = getNearestRVertex(radiusStart, side);
  const int verticesPerSector = mParamGrid.NPhiVertices / SECTORSPERSIDE;

  const auto& regInf = Mapper::instance().getPadRegionInfo(0);
  const float localYEdgeIROC = regInf.getPadsInRowRegion(0) / 2 * regInf.getPadWidth();
  const auto globalPosEdgeIROC = Mapper::LocalToGlobal(LocalPosition2D(regInf.getRadiusFirstRow(), -localYEdgeIROC), Sector(0));

  const int stacks = sizeof(GEMFrameParameters<DataT>::POSTOP) / sizeof(GEMFrameParameters<DataT>::POSTOP[0]);

  std::vector<DataT> radii;
  for (int stack = 0; stack < stacks; ++stack) {
    int region = 3;
    if (stack == 1) {
      region = 5;
    } else if (stack == 2) {
      region = 7;
    } else if (stack == 3) {
      region = 9;
    }
    const auto& regInf = Mapper::instance().getPadRegionInfo(region);
    const float localYEdge = regInf.getPadsInRowRegion(region) / 2 * regInf.getPadWidth();
    radii.emplace_back(std::sqrt(std::pow(GEMFrameParameters<DataT>::POSTOP[stack], 2) + std::pow(localYEdge, 2)));
  }

  for (size_t iR = rStart; iR < mParamGrid.NRVertices - 1; ++iR) {
    const DataT radius = getRVertex(iR, side);
    auto const it = std::lower_bound(radii.begin(), radii.end(), radius);
    const int stack = (it == radii.end()) ? (stacks - 1) : (it - radii.begin());

    // for stack 4 use the the number of phi bins at the edge
    const auto radiusCompare = (stack == 4) ? GEMFrameParameters<DataT>::POSTOP[stack] : GEMFrameParameters<DataT>::POSTOP[stack] + (GEMFrameParameters<DataT>::POSTOP[stack] - GEMFrameParameters<DataT>::POSTOP[stack]) / 2;
    for (size_t iPhiTmp = 0; iPhiTmp < getNPhiVertices(); ++iPhiTmp) {
      const DataT offsetGlobalY = radiusCompare * iPhiTmp * getGridSpacingPhi(side);
      if (iPhiTmp > 0 && offsetGlobalY > globalPosEdgeIROC.Y()) {
        break;
      }

      for (int sector = 0; sector < SECTORSPERSIDE; ++sector) {
        const size_t iPhiLeft = sector * verticesPerSector + iPhiTmp;
        const size_t iZ = mParamGrid.NZVertices - 1;
        mPotential[side](iZ, iR, iPhiLeft) = potentialFunc(radius);
        if (iPhiTmp > 0) {
          const size_t iPhiRight = (sector + 1) * verticesPerSector - iPhiTmp;
          mPotential[side](iZ, iR, iPhiRight) = potentialFunc(radius);
        }
      }
    }
  }
}

template <typename DataT>
void SpaceCharge<DataT>::setPotentialBoundaryGEMFrameAlongPhi(const std::function<DataT(DataT)>& potentialFunc, const GEMstack stack, const bool bottom, const Side side, const bool outerFrame)
{
  int region = 0;
  if (bottom) {
    if (stack == GEMstack::IROCgem) {
      region = 0;
    } else if (stack == GEMstack::OROC1gem) {
      region = 4;
    } else if (stack == GEMstack::OROC2gem) {
      region = 6;
    } else if (stack == GEMstack::OROC3gem) {
      region = 8;
    }
  } else {
    if (stack == GEMstack::IROCgem) {
      region = 3;
    } else if (stack == GEMstack::OROC1gem) {
      region = 5;
    } else if (stack == GEMstack::OROC2gem) {
      region = 7;
    } else if (stack == GEMstack::OROC3gem) {
      region = 9;
    }
  }

  const auto& regInf = Mapper::instance().getPadRegionInfo(region);
  const auto radiusFirstRow = regInf.getRadiusFirstRow();
  DataT radiusStart = bottom ? GEMFrameParameters<DataT>::POSBOTTOM[stack] : radiusFirstRow + regInf.getPadHeight() * regInf.getNumberOfPadRows();
  if (bottom && stack != GEMstack::IROCgem) {
    radiusStart -= GEMFrameParameters<DataT>::WIDTHFRAME;
  }
  const auto radiusMax = bottom ? radiusFirstRow : GEMFrameParameters<DataT>::POSTOP[stack];

  if (outerFrame) {
    radiusStart = radiusMax - 0.5;
  }

  auto nVerticesR = std::round((radiusMax - radiusStart) / getGridSpacingR(side));
  if (nVerticesR == 0) {
    nVerticesR = 1;
  }

  const int verticesPerSector = mParamGrid.NPhiVertices / SECTORSPERSIDE;
  const auto nBinsPhi = getPhiBinsGapFrame(side);
  for (int sector = 0; sector < SECTORSPERSIDE; ++sector) {
    const auto offsetPhi = sector * verticesPerSector + verticesPerSector / 2;
    for (size_t iPhiLocal = 0; iPhiLocal <= verticesPerSector / 2 - nBinsPhi; ++iPhiLocal) {
      const auto iPhiLeft = offsetPhi + iPhiLocal;
      const auto iPhiRight = offsetPhi - iPhiLocal;
      const DataT phiLeft = getPhiVertex(iPhiLeft, side);
      const DataT phiRight = getPhiVertex(iPhiRight, side);
      const DataT localphi = getPhiVertex(iPhiLocal, side);
      const DataT radiusBottom = radiusStart / std::cos(localphi);
      auto rStart = getNearestRVertex(radiusBottom, side);
      const auto nREnd = outerFrame ? mParamGrid.NRVertices - 1 : rStart + nVerticesR;

      if (rStart == 0) {
        rStart = 1;
      }

      for (size_t iR = rStart; iR < nREnd; ++iR) {
        const size_t iZ = mParamGrid.NZVertices - 1;
        if (!outerFrame) {
          mPotential[side](iZ, iR, iPhiLeft) = potentialFunc(phiLeft);
          mPotential[side](iZ, iR, iPhiRight) = potentialFunc(phiRight);
        } else {
          const DataT r = getRVertex(iR, side);
          mPotential[side](iZ, iR, iPhiLeft) = potentialFunc(r);
          mPotential[side](iZ, iR, iPhiRight) = potentialFunc(r);
        }
      }
    }
  }
}

template <typename DataT>
void SpaceCharge<DataT>::setPotentialBoundaryInnerRadius(const std::function<DataT(DataT)>& potentialFunc, const Side side)
{
  for (size_t iZ = 0; iZ < mParamGrid.NZVertices; ++iZ) {
    const DataT z = getZVertex(iZ, side);
    const auto pot = potentialFunc(z);
    for (size_t iPhi = 0; iPhi < mParamGrid.NPhiVertices; ++iPhi) {
      const size_t iR = 0;
      mPotential[side](iZ, iR, iPhi) = pot;
    }
  }
}

template <typename DataT>
void SpaceCharge<DataT>::setPotentialBoundaryOuterRadius(const std::function<DataT(DataT)>& potentialFunc, const Side side)
{
  for (size_t iZ = 0; iZ < mParamGrid.NZVertices; ++iZ) {
    const DataT z = getZVertex(iZ, side);
    const auto pot = potentialFunc(z);
    for (size_t iPhi = 0; iPhi < mParamGrid.NPhiVertices; ++iPhi) {
      const size_t iR = mParamGrid.NRVertices - 1;
      mPotential[side](iZ, iR, iPhi) = pot;
    }
  }
}

template <typename DataT>
void SpaceCharge<DataT>::setPotentialFromFormula(const AnalyticalFields<DataT>& formulaStruct)
{
  const Side side = formulaStruct.getSide();
  for (size_t iPhi = 0; iPhi < mParamGrid.NPhiVertices; ++iPhi) {
    const DataT phi = getPhiVertex(iPhi, side);
    for (size_t iR = 0; iR < mParamGrid.NRVertices; ++iR) {
      const DataT radius = getRVertex(iR, side);
      for (size_t iZ = 0; iZ < mParamGrid.NZVertices; ++iZ) {
        const DataT z = getZVertex(iZ, side);
        mPotential[side](iZ, iR, iPhi) = formulaStruct.evalPotential(z, radius, phi);
      }
    }
  }
}

template <typename DataT>
void SpaceCharge<DataT>::setPotentialBoundaryFromFormula(const AnalyticalFields<DataT>& formulaStruct)
{
  const Side side = formulaStruct.getSide();
  for (size_t iPhi = 0; iPhi < mParamGrid.NPhiVertices; ++iPhi) {
    const DataT phi = getPhiVertex(iPhi, side);
    for (size_t iZ = 0; iZ < mParamGrid.NZVertices; ++iZ) {
      const DataT z = getZVertex(iZ, side);
      const size_t iR = 0;
      const DataT radius = getRVertex(iR, side);
      mPotential[side](iZ, iR, iPhi) = formulaStruct.evalPotential(z, radius, phi);
    }
  }

  for (size_t iPhi = 0; iPhi < mParamGrid.NPhiVertices; ++iPhi) {
    const DataT phi = getPhiVertex(iPhi, side);
    for (size_t iZ = 0; iZ < mParamGrid.NZVertices; ++iZ) {
      const DataT z = getZVertex(iZ, side);
      const size_t iR = mParamGrid.NRVertices - 1;
      const DataT radius = getRVertex(iR, side);
      mPotential[side](iZ, iR, iPhi) = formulaStruct.evalPotential(z, radius, phi);
    }
  }

  for (size_t iPhi = 0; iPhi < mParamGrid.NPhiVertices; ++iPhi) {
    const DataT phi = getPhiVertex(iPhi, side);
    for (size_t iR = 0; iR < mParamGrid.NRVertices; ++iR) {
      const DataT radius = getRVertex(iR, side);
      const size_t iZ = 0;
      const DataT z = getZVertex(iZ, side);
      mPotential[side](iZ, iR, iPhi) = formulaStruct.evalPotential(z, radius, phi);
    }
  }

  for (size_t iPhi = 0; iPhi < mParamGrid.NPhiVertices; ++iPhi) {
    const DataT phi = getPhiVertex(iPhi, side);
    for (size_t iR = 0; iR < mParamGrid.NRVertices; ++iR) {
      const DataT radius = getRVertex(iR, side);
      const size_t iZ = mParamGrid.NZVertices - 1;
      const DataT z = getZVertex(iZ, side);
      mPotential[side](iZ, iR, iPhi) = formulaStruct.evalPotential(z, radius, phi);
    }
  }
}

template <typename DataT>
void SpaceCharge<DataT>::poissonSolver(const Side side, const DataT stoppingConvergence, const int symmetry)
{
  PoissonSolver<DataT>::setConvergenceError(stoppingConvergence);
  PoissonSolver<DataT> poissonSolver(mGrid3D[0]);
  poissonSolver.poissonSolver3D(mPotential[side], mDensity[side], symmetry);
}

template <typename DataT>
void SpaceCharge<DataT>::setEFieldFromFormula(const AnalyticalFields<DataT>& formulaStruct)
{
  const Side side = formulaStruct.getSide();
  for (size_t iPhi = 0; iPhi < mParamGrid.NPhiVertices; ++iPhi) {
    for (size_t iR = 0; iR < mParamGrid.NRVertices; ++iR) {
      for (size_t iZ = 0; iZ < mParamGrid.NZVertices; ++iZ) {
        const DataT radius = getRVertex(iR, side);
        const DataT z = getZVertex(iZ, side);
        const DataT phi = getPhiVertex(iPhi, side);
        mElectricFieldEr[side](iZ, iR, iPhi) = formulaStruct.evalEr(z, radius, phi);
        mElectricFieldEz[side](iZ, iR, iPhi) = formulaStruct.evalEz(z, radius, phi);
        mElectricFieldEphi[side](iZ, iR, iPhi) = formulaStruct.evalEphi(z, radius, phi);
      }
    }
  }
  mIsEfieldSet[side] = true;
}

template <typename DataT>
void SpaceCharge<DataT>::calcEField(const Side side)
{
#pragma omp parallel for num_threads(sNThreads)
  for (size_t iPhi = 0; iPhi < mParamGrid.NPhiVertices; ++iPhi) {
    const int symmetry = 0;
    size_t tmpPlus = iPhi + 1;
    int signPlus = 1;
    int tmpMinus = static_cast<int>(iPhi - 1);
    int signMinus = 1;
    if (symmetry == 1 || symmetry == -1) { // Reflection symmetry in phi (e.g. symmetry at sector boundaries, or half sectors, etc.)
      if (tmpPlus > mParamGrid.NPhiVertices - 1) {
        if (symmetry == -1) {
          signPlus = -1;
        }
        tmpPlus = mParamGrid.NPhiVertices - 2;
      }
      if (tmpMinus < 0) {
        tmpMinus = 1; // SHOULD IT BE =0?
        if (symmetry == -1) {
          signMinus = -1;
        }
      }
    } else { // No Symmetries in phi, no boundaries, the calculations is continuous across all phi
      if (tmpPlus > mParamGrid.NPhiVertices - 1) {
        tmpPlus = iPhi + 1 - mParamGrid.NPhiVertices;
      }
      if (tmpMinus < 0) {
        tmpMinus = static_cast<int>(iPhi - 1 + mParamGrid.NPhiVertices);
      }
    }

    // for non-boundary V
    for (size_t iR = 1; iR < mParamGrid.NRVertices - 1; iR++) {
      const DataT radius = getRVertex(iR, side);
      for (size_t iZ = 1; iZ < mParamGrid.NZVertices - 1; iZ++) {
        mElectricFieldEr[side](iZ, iR, iPhi) = -1 * (mPotential[side](iZ, iR + 1, iPhi) - mPotential[side](iZ, iR - 1, iPhi)) * static_cast<DataT>(0.5) * getInvSpacingR(side);                                    // r direction
        mElectricFieldEz[side](iZ, iR, iPhi) = -1 * (mPotential[side](iZ + 1, iR, iPhi) - mPotential[side](iZ - 1, iR, iPhi)) * static_cast<DataT>(0.5) * getInvSpacingZ(side);                                    // z direction
        mElectricFieldEphi[side](iZ, iR, iPhi) = -1 * (signPlus * mPotential[side](iZ, iR, tmpPlus) - signMinus * mPotential[side](iZ, iR, tmpMinus)) * static_cast<DataT>(0.5) * getInvSpacingPhi(side) / radius; // phi direction
      }
    }

    // for boundary-r
    for (size_t iZ = 0; iZ < mParamGrid.NZVertices; iZ++) {
      mElectricFieldEr[side](iZ, 0, iPhi) = -1 * (-static_cast<DataT>(0.5) * mPotential[side](iZ, 2, iPhi) + 2 * mPotential[side](iZ, 1, iPhi) - static_cast<DataT>(1.5) * mPotential[side](iZ, 0, iPhi)) * getInvSpacingR(side);                                                                                                // forward difference
      mElectricFieldEr[side](iZ, mParamGrid.NRVertices - 1, iPhi) = -1 * (static_cast<DataT>(1.5) * mPotential[side](iZ, mParamGrid.NRVertices - 1, iPhi) - 2 * mPotential[side](iZ, mParamGrid.NRVertices - 2, iPhi) + static_cast<DataT>(0.5) * mPotential[side](iZ, mParamGrid.NRVertices - 3, iPhi)) * getInvSpacingR(side); // backward difference
    }

    for (size_t iR = 0; iR < mParamGrid.NRVertices; iR += mParamGrid.NRVertices - 1) {
      const DataT radius = getRVertex(iR, side);
      for (size_t iZ = 1; iZ < mParamGrid.NZVertices - 1; iZ++) {
        mElectricFieldEz[side](iZ, iR, iPhi) = -1 * (mPotential[side](iZ + 1, iR, iPhi) - mPotential[side](iZ - 1, iR, iPhi)) * static_cast<DataT>(0.5) * getInvSpacingZ(side);                                    // z direction
        mElectricFieldEphi[side](iZ, iR, iPhi) = -1 * (signPlus * mPotential[side](iZ, iR, tmpPlus) - signMinus * mPotential[side](iZ, iR, tmpMinus)) * static_cast<DataT>(0.5) * getInvSpacingPhi(side) / radius; // phi direction
      }
    }

    // for boundary-z
    for (size_t iR = 0; iR < mParamGrid.NRVertices; ++iR) {
      mElectricFieldEz[side](0, iR, iPhi) = -1 * (-static_cast<DataT>(0.5) * mPotential[side](2, iR, iPhi) + 2 * mPotential[side](1, iR, iPhi) - static_cast<DataT>(1.5) * mPotential[side](0, iR, iPhi)) * getInvSpacingZ(side);
      mElectricFieldEz[side](mParamGrid.NZVertices - 1, iR, iPhi) = -1 * (static_cast<DataT>(1.5) * mPotential[side](mParamGrid.NZVertices - 1, iR, iPhi) - 2 * mPotential[side](mParamGrid.NZVertices - 2, iR, iPhi) + static_cast<DataT>(0.5) * mPotential[side](mParamGrid.NZVertices - 3, iR, iPhi)) * getInvSpacingZ(side);
    }

    for (size_t iR = 1; iR < mParamGrid.NRVertices - 1; ++iR) {
      const DataT radius = getRVertex(iR, side);
      for (size_t iZ = 0; iZ < mParamGrid.NZVertices; iZ += mParamGrid.NZVertices - 1) {
        mElectricFieldEr[side](iZ, iR, iPhi) = -1 * (mPotential[side](iZ, iR + 1, iPhi) - mPotential[side](iZ, iR - 1, iPhi)) * static_cast<DataT>(0.5) * getInvSpacingR(side);                                    // r direction
        mElectricFieldEphi[side](iZ, iR, iPhi) = -1 * (signPlus * mPotential[side](iZ, iR, tmpPlus) - signMinus * mPotential[side](iZ, iR, tmpMinus)) * static_cast<DataT>(0.5) * getInvSpacingPhi(side) / radius; // phi direction
      }
    }

    // corner points for EPhi
    for (size_t iR = 0; iR < mParamGrid.NRVertices; iR += mParamGrid.NRVertices - 1) {
      const DataT radius = getRVertex(iR, side);
      for (size_t iZ = 0; iZ < mParamGrid.NZVertices; iZ += mParamGrid.NZVertices - 1) {
        mElectricFieldEphi[side](iZ, iR, iPhi) = -1 * (signPlus * mPotential[side](iZ, iR, tmpPlus) - signMinus * mPotential[side](iZ, iR, tmpMinus)) * static_cast<DataT>(0.5) * getInvSpacingPhi(side) / radius; // phi direction
      }
    }
  }
  mIsEfieldSet[side] = true;
}

template <typename DataT>
void SpaceCharge<DataT>::calcGlobalDistWithGlobalCorrIterative(const DistCorrInterpolator<DataT>& globCorr, const int maxIter, const DataT approachZ, const DataT approachR, const DataT approachPhi, const DataT diffCorr)
{
  const Side side = globCorr.getSide();

#pragma omp parallel for num_threads(sNThreads)
  for (unsigned int iPhi = 0; iPhi < mParamGrid.NPhiVertices; ++iPhi) {
    const DataT phi = getPhiVertex(iPhi, side);
    for (unsigned int iR = 0; iR < mParamGrid.NRVertices; ++iR) {
      const DataT radius = getRVertex(iR, side);
      for (unsigned int iZ = 0; iZ < mParamGrid.NZVertices; ++iZ) {
        const DataT z = getZVertex(iZ, side);

        unsigned int nearestiZ = iZ;
        unsigned int nearestiR = iR;
        unsigned int nearestiPhi = iPhi;

        DataT nearestZ = getZVertex(nearestiZ, side);
        DataT nearestR = getRVertex(nearestiR, side);
        DataT nearestPhi = getPhiVertex(nearestiPhi, side);

        //
        //==========================================================================================
        //==== start algorithm: use tricubic upsampling to numerically approach the query point ====
        //==========================================================================================
        //
        // 1. calculate difference from nearest point to query point with stepwidth factor x
        // and approach the new point
        //
        DataT stepR = (radius - nearestR) * approachR;
        DataT stepZ = (z - nearestZ) * approachZ;
        DataT stepPhi = (phi - nearestPhi) * approachPhi;

        // needed to check for convergence
        DataT lastCorrdR = std::numeric_limits<DataT>::max();
        DataT lastCorrdZ = std::numeric_limits<DataT>::max();
        DataT lastCorrdRPhi = std::numeric_limits<DataT>::max();

        // interpolated global correction
        DataT corrdR = 0;
        DataT corrdRPhi = 0;
        DataT corrdZ = 0;

        for (int iter = 0; iter < maxIter; ++iter) {
          // 2. get new point coordinates
          const DataT rCurrPos = getRVertex(nearestiR, side) + stepR;
          const DataT zCurrPos = getZVertex(nearestiZ, side) + stepZ;
          const DataT phiCurrPos = getPhiVertex(nearestiPhi, side) + stepPhi;

          // abort calculation of drift path if electron reached inner/outer field cage or central electrode
          if (rCurrPos <= getRMinSim(side) || rCurrPos >= getRMaxSim(side) || getSide(zCurrPos) != side) {
            break;
          }

          // interpolate global correction at new point and calculate position of global correction
          corrdR = globCorr.evaldR(zCurrPos, rCurrPos, phiCurrPos);
          const DataT rNewPos = rCurrPos + corrdR;

          const DataT corrPhi = globCorr.evaldRPhi(zCurrPos, rCurrPos, phiCurrPos) / rCurrPos;
          corrdRPhi = corrPhi * rNewPos; // normalize to new r coordinate
          const DataT phiNewPos = phiCurrPos + corrPhi;

          corrdZ = globCorr.evaldZ(zCurrPos, rCurrPos, phiCurrPos);
          const DataT zNewPos = zCurrPos + corrdZ;

          // approach desired coordinate
          stepR += (radius - rNewPos) * approachR;
          stepZ += (z - zNewPos) * approachZ;
          stepPhi += (phi - phiNewPos) * approachPhi;

          // check for convergence
          const DataT diffCorrdR = std::abs(corrdR - lastCorrdR);
          const DataT diffCorrdRZ = std::abs(corrdZ - lastCorrdZ);
          const DataT diffCorrdRPhi = std::abs(corrdRPhi - lastCorrdRPhi);

          // stop algorithm if converged
          if (diffCorrdR < diffCorr && diffCorrdRZ < diffCorr && diffCorrdRPhi < diffCorr) {
            break;
          }

          lastCorrdR = corrdR;
          lastCorrdZ = corrdZ;
          lastCorrdRPhi = corrdRPhi;
        }
        // set global distortions if algorithm converged or iterations exceed max numbers of iterations
        mGlobalDistdR[side](iZ, iR, iPhi) = -corrdR;
        mGlobalDistdRPhi[side](iZ, iR, iPhi) = -corrdRPhi;
        mGlobalDistdZ[side](iZ, iR, iPhi) = -corrdZ;
      }
    }
  }
  // set flag that global distortions are set to true
  mIsGlobalDistSet[side] = true;
}

template <typename DataT>
NumericalFields<DataT> SpaceCharge<DataT>::getElectricFieldsInterpolator(const Side side) const
{
  if (!mIsEfieldSet[side]) {
    LOGP(warning, "============== E-Fields are not set! ==============\n");
  }
  NumericalFields<DataT> numFields(mElectricFieldEr[side], mElectricFieldEz[side], mElectricFieldEphi[side], mGrid3D[side], side);
  return numFields;
}

template <typename DataT>
DistCorrInterpolator<DataT> SpaceCharge<DataT>::getLocalDistInterpolator(const Side side) const
{
  if (!mIsLocalDistSet[side]) {
    LOGP(warning, "============== local distortions not set! ==============\n");
  }
  DistCorrInterpolator<DataT> numFields(mLocalDistdR[side], mLocalDistdZ[side], mLocalDistdRPhi[side], mGrid3D[side], side);
  return numFields;
}

template <typename DataT>
DistCorrInterpolator<DataT> SpaceCharge<DataT>::getLocalCorrInterpolator(const Side side) const
{
  if (!mIsLocalCorrSet[side]) {
    LOGP(warning, "============== local corrections not set!  ==============\n");
  }
  DistCorrInterpolator<DataT> numFields(mLocalCorrdR[side], mLocalCorrdZ[side], mLocalCorrdRPhi[side], mGrid3D[side], side);
  return numFields;
}

template <typename DataT>
DistCorrInterpolator<DataT> SpaceCharge<DataT>::getGlobalDistInterpolator(const Side side) const
{
  if (!mIsGlobalDistSet[side]) {
    LOGP(warning, "============== global distortions not set ==============\n");
  }
  DistCorrInterpolator<DataT> numFields(mGlobalDistdR[side], mGlobalDistdZ[side], mGlobalDistdRPhi[side], mGrid3D[side], side);
  return numFields;
}

template <typename DataT>
DistCorrInterpolator<DataT> SpaceCharge<DataT>::getGlobalCorrInterpolator(const Side side) const
{
  if (!mIsGlobalCorrSet[side]) {
    LOGP(warning, "============== global corrections not set ==============\n");
  }
  DistCorrInterpolator<DataT> numFields(mGlobalCorrdR[side], mGlobalCorrdZ[side], mGlobalCorrdRPhi[side], mGrid3D[side], side);
  return numFields;
}

template <typename DataT>
void SpaceCharge<DataT>::fillChargeDensityFromFile(TFile& fInp, const char* name)
{
  TH3* hisSCDensity3D = (TH3*)fInp.Get(name);
  fillChargeDensityFromHisto(*hisSCDensity3D);
  delete hisSCDensity3D;
}

template <typename DataT>
void SpaceCharge<DataT>::fillChargeDensityFromHisto(const TH3& hisSCDensity3D)
{
  TH3DataT hRebin = rebinDensityHisto(hisSCDensity3D, mParamGrid.NZVertices, mParamGrid.NRVertices, mParamGrid.NPhiVertices);
  for (int side = Side::A; side < SIDES; ++side) {
    for (size_t iPhi = 0; iPhi < mParamGrid.NPhiVertices; ++iPhi) {
      for (size_t iR = 0; iR < mParamGrid.NRVertices; ++iR) {
        for (size_t iZ = 0; iZ < mParamGrid.NZVertices; ++iZ) {
          const size_t zBin = side == Side::A ? mParamGrid.NZVertices + iZ + 1 : mParamGrid.NZVertices - iZ;
          mDensity[side](iZ, iR, iPhi) = hRebin.GetBinContent(iPhi + 1, iR + 1, zBin);
        }
      }
    }
    mIsChargeSet[side] = true;
  }
}

template <typename DataT>
void SpaceCharge<DataT>::fillChargeDensityFromCalDet(const std::vector<CalDet<float>>& calSCDensity3D)
{
  const auto hConverted = o2::tpc::painter::convertCalDetToTH3(calSCDensity3D, true, mParamGrid.NRVertices, getRMin(Side::A), getRMax(Side::A), mParamGrid.NPhiVertices, getZMax(Side::A));
  fillChargeDensityFromHisto(hConverted);
}

template <typename DataT>
void SpaceCharge<DataT>::fillChargeFromCalDet(const std::vector<CalDet<float>>& calCharge3D)
{
  auto hConverted = o2::tpc::painter::convertCalDetToTH3(calCharge3D, false, mParamGrid.NRVertices, getRMin(Side::A), getRMax(Side::A), mParamGrid.NPhiVertices, getZMax(Side::A));
  normalizeHistoQVEps0(hConverted);
  fillChargeDensityFromHisto(hConverted);
}

template <typename DataT>
typename SpaceCharge<DataT>::TH3DataT SpaceCharge<DataT>::rebinDensityHisto(const TH3& hOrig, const unsigned short nBinsZNew, const unsigned short nBinsRNew, const unsigned short nBinsPhiNew)
{
  TH3DataT hRebin{};
  const int nBinsZNewTwo = 2 * nBinsZNew;

  const auto phiLow = hOrig.GetXaxis()->GetBinLowEdge(1);
  const auto phiUp = hOrig.GetXaxis()->GetBinUpEdge(hOrig.GetNbinsX());
  const auto rLow = hOrig.GetYaxis()->GetBinLowEdge(1);
  const auto rUp = hOrig.GetYaxis()->GetBinUpEdge(hOrig.GetNbinsY());
  const auto zLow = hOrig.GetZaxis()->GetBinLowEdge(1);
  const auto zUp = hOrig.GetZaxis()->GetBinUpEdge(hOrig.GetNbinsZ());
  hRebin.SetBins(nBinsPhiNew, phiLow, phiUp, nBinsRNew, rLow, rUp, nBinsZNewTwo, zLow, zUp);

  for (int iBinPhi = 1; iBinPhi <= nBinsPhiNew; ++iBinPhi) {
    const auto phiLowEdge = hRebin.GetXaxis()->GetBinLowEdge(iBinPhi);
    const auto phiUpEdge = hRebin.GetXaxis()->GetBinUpEdge(iBinPhi);

    const int phiLowBinOrig = hOrig.GetXaxis()->FindBin(phiLowEdge);
    const int phiUpBinOrig = hOrig.GetXaxis()->FindBin(phiUpEdge);

    // calculate the weights (area of original bin lies in the new bin / binwidthOrig) of the first and last bins
    const auto binWidthPhiOrig = hOrig.GetXaxis()->GetBinWidth(phiLowBinOrig);
    const auto lowerBinWeightPhi = std::abs(phiLowEdge - hOrig.GetXaxis()->GetBinUpEdge(phiLowBinOrig)) / binWidthPhiOrig;
    const auto upperBinWeightPhi = std::abs(phiUpEdge - hOrig.GetXaxis()->GetBinLowEdge(phiUpBinOrig)) / binWidthPhiOrig;

    for (int iBinR = 1; iBinR <= nBinsRNew; ++iBinR) {
      const auto rLowEdge = hRebin.GetYaxis()->GetBinLowEdge(iBinR);
      const auto rUpEdge = hRebin.GetYaxis()->GetBinUpEdge(iBinR);

      const int rLowBinOrig = hOrig.GetYaxis()->FindBin(rLowEdge);
      const int rUpBinOrig = hOrig.GetYaxis()->FindBin(rUpEdge);

      // calculate the weights (area of original bin lies in the new bin / binwidthOrig) of the first and last bins
      const auto binWidthROrig = hOrig.GetYaxis()->GetBinWidth(rLowBinOrig);
      const auto lowerBinWeightR = std::abs(rLowEdge - hOrig.GetYaxis()->GetBinUpEdge(rLowBinOrig)) / binWidthROrig;
      const auto upperBinWeightR = std::abs(rUpEdge - hOrig.GetYaxis()->GetBinLowEdge(rUpBinOrig)) / binWidthROrig;

      for (int iBinZ = 1; iBinZ <= nBinsZNewTwo; ++iBinZ) {
        const auto zLowEdge = hRebin.GetZaxis()->GetBinLowEdge(iBinZ);
        const auto zUpEdge = hRebin.GetZaxis()->GetBinUpEdge(iBinZ);
        const auto zCenter = hRebin.GetZaxis()->GetBinCenter(iBinZ);

        int zLowBinOrig = hOrig.GetZaxis()->FindBin(zLowEdge);
        int zUpBinOrig = hOrig.GetZaxis()->FindBin(zUpEdge);
        const int currside = getSide(zCenter); // set the side of the current z-bin
        // get the side of the lowest and uppest bin from the orig histo
        const int sideLowOrig = getSide(hOrig.GetZaxis()->GetBinCenter(zLowBinOrig));
        const int sideUpOrig = getSide(hOrig.GetZaxis()->GetBinCenter(zUpBinOrig));

        // make bounds/side check of the zLowBinOrig and zUpBinOrig bins. They must be on the same side as the currside!!!
        if (currside != sideLowOrig && zLowBinOrig != zUpBinOrig) {
          // if the lower bins from the orig histo are not on the same side as the rebinned increase the binnumber until they are on the same side
          bool notequal = true;
          do {
            zLowBinOrig += 1;
            if (zLowBinOrig > zUpBinOrig) {
              LOGP(warning, "SOMETHING WENT WRONG: SETTING BINS TO: {}", zUpBinOrig);
              zLowBinOrig = zUpBinOrig;
              notequal = false;
            }
            const int sideTmp = getSide(hOrig.GetZaxis()->GetBinCenter(zLowBinOrig));
            if (sideTmp == currside) {
              notequal = false;
            }
          } while (notequal);
        }

        if (currside != sideUpOrig && zLowBinOrig != zUpBinOrig) {
          // if the upper bins from the orig histo are not on the same side as the rebinned increase the binnumber until they are on the same side
          bool notequal = true;
          do {
            zUpBinOrig -= 1;
            if (zUpBinOrig < zLowBinOrig) {
              LOGP(warning, "SOMETHING WENT WRONG: SETTING BINS TO: {}", zLowBinOrig);
              zUpBinOrig = zLowBinOrig;
              notequal = false;
            }
            const int sideTmp = getSide(hOrig.GetZaxis()->GetBinCenter(zUpBinOrig));
            if (sideTmp == currside) {
              notequal = false;
            }
          } while (notequal);
        }

        const auto binWidthZOrig = hOrig.GetZaxis()->GetBinWidth(zLowBinOrig);
        const auto lowerBinWeightZ = std::abs(zLowEdge - hOrig.GetZaxis()->GetBinUpEdge(zLowBinOrig)) / binWidthZOrig;
        const auto upperBinWeightZ = std::abs(zUpEdge - hOrig.GetZaxis()->GetBinLowEdge(zUpBinOrig)) / binWidthZOrig;

        // get the mean value of the original histogram of the found bin range
        DataT sum = 0;
        DataT sumW = 0;
        for (int iPhi = phiLowBinOrig; iPhi <= phiUpBinOrig; ++iPhi) {
          DataT weightPhi = 1;
          if (iPhi == phiLowBinOrig) {
            weightPhi = lowerBinWeightPhi;
          } else if (iPhi == phiUpBinOrig) {
            weightPhi = upperBinWeightPhi;
          }

          for (int iR = rLowBinOrig; iR <= rUpBinOrig; ++iR) {
            DataT weightR = 1;
            if (iR == rLowBinOrig) {
              weightR = lowerBinWeightR;
            } else if (iR == rUpBinOrig) {
              weightR = upperBinWeightR;
            }

            for (int iZ = zLowBinOrig; iZ <= zUpBinOrig; ++iZ) {
              DataT weightZ = 1;
              if (iZ == zLowBinOrig) {
                weightZ = lowerBinWeightZ;
              } else if (iZ == zUpBinOrig) {
                weightZ = upperBinWeightZ;
              }
              const auto val = hOrig.GetBinContent(iPhi, iR, iZ);
              // if(val==0){
              // what to do now???
              // }
              const auto totalWeight = weightPhi * weightR * weightZ;
              sum += val * totalWeight;
              sumW += totalWeight;
            }
          }
        }
        sum /= sumW;
        hRebin.SetBinContent(iBinPhi, iBinR, iBinZ, sum);
      }
    }
  }
  return hRebin;
}

template <typename DataT>
template <typename ElectricFields>
void SpaceCharge<DataT>::calcLocalDistortionsCorrections(const SpaceCharge<DataT>::Type type, const ElectricFields& formulaStruct)
{
  const Side side = formulaStruct.getSide();
  // calculate local distortions/corrections for each vertex in the tpc
#pragma omp parallel for num_threads(sNThreads)
  for (size_t iPhi = 0; iPhi < mParamGrid.NPhiVertices; ++iPhi) {
    const DataT phi = getPhiVertex(iPhi, side);
    for (size_t iR = 0; iR < mParamGrid.NRVertices; ++iR) {
      const DataT radius = getRVertex(iR, side);
      for (size_t iZ = 0; iZ < mParamGrid.NZVertices - 1; ++iZ) {
        // set z coordinate depending on distortions or correction calculation
        const DataT z0 = type == Type::Corrections ? getZVertex(iZ + 1, side) : getZVertex(iZ, side);
        const DataT z1 = type == Type::Corrections ? getZVertex(iZ, side) : getZVertex(iZ + 1, side);

        DataT drTmp = 0;   // local distortion dR
        DataT dPhiTmp = 0; // local distortion dPhi (multiplication with R has to be done at the end)
        DataT dzTmp = 0;   // local distortion dZ

        const DataT stepSize = (z1 - z0) / sSteps; // the distortions are calculated by leting the elctron drift this distance in z direction
        for (int iter = 0; iter < sSteps; ++iter) {
          const DataT z0Tmp = (z0 + iter * stepSize + dzTmp); // starting z position
          const DataT z1Tmp = (z0Tmp + stepSize);             // electron drifts from z0Tmp to z1Tmp

          DataT ddR = 0;   // distortion dR for drift from z0Tmp to z1Tmp
          DataT ddPhi = 0; // distortion dPhi for drift from z0Tmp to z1Tmp
          DataT ddZ = 0;   // distortion dZ for drift from z0Tmp to z1Tmp

          const DataT radiusTmp = regulateR(radius + drTmp, side); // current radial position
          const DataT phiTmp = regulatePhi(phi + dPhiTmp, side);   // current phi position

          // calculate distortions/corrections
          calcDistCorr(radiusTmp, phiTmp, z0Tmp, z1Tmp, ddR, ddPhi, ddZ, formulaStruct, true);

          // add temp distortions to local distortions
          drTmp += ddR;
          dPhiTmp += ddPhi;
          dzTmp += ddZ;
        }

        // store local distortions/corrections
        switch (type) {
          case Type::Corrections:
            mLocalCorrdR[side](iZ + 1, iR, iPhi) = drTmp;
            mLocalCorrdRPhi[side](iZ + 1, iR, iPhi) = dPhiTmp * radius;
            mLocalCorrdZ[side](iZ + 1, iR, iPhi) = dzTmp;
            break;

          case Type::Distortions:
            mLocalDistdR[side](iZ, iR, iPhi) = drTmp;
            mLocalDistdRPhi[side](iZ, iR, iPhi) = dPhiTmp * radius;
            mLocalDistdZ[side](iZ, iR, iPhi) = dzTmp;
            break;
        }
      }
      // extrapolate local distortion/correction to last/first bin using legendre polynoms with x0=0, x1=1, x2=2 and x=-1. This has to be done to ensure correct interpolation in the last,second last/first,second bin!
      switch (type) {
        case Type::Corrections:
          mLocalCorrdR[side](0, iR, iPhi) = 3 * (mLocalCorrdR[side](1, iR, iPhi) - mLocalCorrdR[side](2, iR, iPhi)) + mLocalCorrdR[side](3, iR, iPhi);
          mLocalCorrdRPhi[side](0, iR, iPhi) = 3 * (mLocalCorrdRPhi[side](1, iR, iPhi) - mLocalCorrdRPhi[side](2, iR, iPhi)) + mLocalCorrdRPhi[side](3, iR, iPhi);
          mLocalCorrdZ[side](0, iR, iPhi) = 3 * (mLocalCorrdZ[side](1, iR, iPhi) - mLocalCorrdZ[side](2, iR, iPhi)) + mLocalCorrdZ[side](3, iR, iPhi);
          break;

        case Type::Distortions:
          mLocalDistdR[side](mParamGrid.NZVertices - 1, iR, iPhi) = 3 * (mLocalDistdR[side](mParamGrid.NZVertices - 2, iR, iPhi) - mLocalDistdR[side](mParamGrid.NZVertices - 3, iR, iPhi)) + mLocalDistdR[side](mParamGrid.NZVertices - 4, iR, iPhi);
          mLocalDistdRPhi[side](mParamGrid.NZVertices - 1, iR, iPhi) = 3 * (mLocalDistdRPhi[side](mParamGrid.NZVertices - 2, iR, iPhi) - mLocalDistdRPhi[side](mParamGrid.NZVertices - 3, iR, iPhi)) + mLocalDistdRPhi[side](mParamGrid.NZVertices - 4, iR, iPhi);
          mLocalDistdZ[side](mParamGrid.NZVertices - 1, iR, iPhi) = 3 * (mLocalDistdZ[side](mParamGrid.NZVertices - 2, iR, iPhi) - mLocalDistdZ[side](mParamGrid.NZVertices - 3, iR, iPhi)) + mLocalDistdZ[side](mParamGrid.NZVertices - 4, iR, iPhi);
          break;
      }
    }
  }
  switch (type) {
    case Type::Corrections:
      mIsLocalCorrSet[side] = true;
      break;
    case Type::Distortions:
      mIsLocalDistSet[side] = true;
      break;
  }
}

template <typename DataT>
template <typename ElectricFields>
void SpaceCharge<DataT>::calcLocalDistortionCorrectionVector(const ElectricFields& formulaStruct)
{
  const Side side = formulaStruct.getSide();
  // calculate local distortion/correction vector for each vertex in the tpc
#pragma omp parallel for num_threads(sNThreads)
  for (size_t iPhi = 0; iPhi < mParamGrid.NPhiVertices; ++iPhi) {
    for (size_t iR = 0; iR < mParamGrid.NRVertices; ++iR) {
      for (size_t iZ = 0; iZ < mParamGrid.NZVertices; ++iZ) {
        const DataT ezField = getEzField(formulaStruct.getSide());
        const DataT er = mElectricFieldEr[side](iZ, iR, iPhi);
        const DataT ez0 = mElectricFieldEz[side](iZ, iR, iPhi);
        const DataT ephi = mElectricFieldEphi[side](iZ, iR, iPhi);
        const DataT ez = getSign(formulaStruct.getSide()) * 1. / (ezField + ez0);
        const DataT erez = er * ez;
        const DataT ephiez = ephi * ez;

        const DataT vecdR = mC0 * erez + mC1 * ephiez;
        const DataT vecdRPhi = mC0 * ephiez - mC1 * erez;
        const DataT vecdZ = -ez0 * TPCParameters<DataT>::DVDE;

        mLocalVecDistdR[side](iZ, iR, iPhi) = vecdR;
        mLocalVecDistdRPhi[side](iZ, iR, iPhi) = vecdRPhi;
        mLocalVecDistdZ[side](iZ, iR, iPhi) = vecdZ;
      }
    }
  }
  mIsLocalVecDistSet[side] = true;
}

template <typename DataT>
template <typename ElectricFields>
void SpaceCharge<DataT>::calcLocalDistortionsCorrectionsRK4(const SpaceCharge<DataT>::Type type, const Side side)
{
  // see: https://en.wikipedia.org/wiki/Runge%E2%80%93Kutta_methods
  // calculate local distortions/corrections for each vertex in the tpc using Runge Kutta 4 method
#pragma omp parallel for num_threads(sNThreads)
  for (size_t iPhi = 0; iPhi < mParamGrid.NPhiVertices; ++iPhi) {
    const DataT phi = getPhiVertex(iPhi, side);
    for (size_t iR = 0; iR < mParamGrid.NRVertices; ++iR) {
      const DataT radius = getRVertex(iR, side);
      for (size_t iZ = 0; iZ < mParamGrid.NZVertices - 1; ++iZ) {
        // set z coordinate depending on distortions or correction calculation
        const size_t iZ0 = type == Type::Corrections ? iZ + 1 : iZ;
        const size_t iZ1 = type == Type::Corrections ? iZ : iZ + 1;

        const DataT z0 = getZVertex(iZ0, side);
        const DataT z1 = getZVertex(iZ1, side);

        const DataT stepSize = z1 - z0; // h in the RK4 method
        const DataT absstepSize = std::abs(stepSize);

        const DataT stepSizeHalf = 0.5 * stepSize; // half z bin for RK4
        const DataT absstepSizeHalf = std::abs(stepSizeHalf);

        // starting position for RK4
        const DataT zk1 = z0;
        const DataT rk1 = radius;
        const DataT phik1 = phi;

        DataT k1dR = 0;    // derivative in r direction
        DataT k1dZ = 0;    // derivative in z direction
        DataT k1dRPhi = 0; // derivative in rphi direction

        // get derivative on current vertex
        switch (type) {
          case Type::Corrections:
            k1dR = getLocalVecCorrR(iZ0, iR, iPhi, side);
            k1dZ = getLocalVecCorrZ(iZ0, iR, iPhi, side);
            k1dRPhi = getLocalVecCorrRPhi(iZ0, iR, iPhi, side);
            break;

          case Type::Distortions:
            k1dR = getLocalVecDistR(iZ0, iR, iPhi, side);
            k1dZ = getLocalVecDistZ(iZ0, iR, iPhi, side);
            k1dRPhi = getLocalVecDistRPhi(iZ0, iR, iPhi, side);
            break;
        }

        // approximate position after half stepSize
        const DataT zk2 = zk1 + stepSizeHalf + absstepSizeHalf * k1dZ;
        const DataT rk2 = rk1 + absstepSizeHalf * k1dR;
        const DataT k1dPhi = k1dRPhi / rk1;
        const DataT phik2 = phik1 + absstepSizeHalf * k1dPhi;

        // get derivative for new position
        DataT k2dR = 0;
        DataT k2dZ = 0;
        DataT k2dRPhi = 0;
        type == Type::Corrections ? getLocalCorrectionVectorCyl(zk2, rk2, phik2, side, k2dZ, k2dR, k2dRPhi) : getLocalDistortionVectorCyl(zk2, rk2, phik2, side, k2dZ, k2dR, k2dRPhi);

        // approximate new position
        const DataT zk3 = zk1 + stepSizeHalf + absstepSizeHalf * k2dZ;
        const DataT rk3 = rk1 + absstepSizeHalf * k2dR;
        const DataT k2dPhi = k2dRPhi / rk2;
        const DataT phik3 = phik1 + absstepSizeHalf * k2dPhi;

        DataT k3dR = 0;
        DataT k3dZ = 0;
        DataT k3dRPhi = 0;
        type == Type::Corrections ? getLocalCorrectionVectorCyl(zk3, rk3, phik3, side, k3dZ, k3dR, k3dRPhi) : getLocalDistortionVectorCyl(zk3, rk3, phik3, side, k3dZ, k3dR, k3dRPhi);

        const DataT zk4 = zk1 + stepSize + absstepSize * k3dZ;
        const DataT rk4 = rk1 + absstepSize * k3dR;
        const DataT k3dPhi = k3dRPhi / rk3;
        const DataT phik4 = phik1 + absstepSize * k3dPhi;

        DataT k4dR = 0;
        DataT k4dZ = 0;
        DataT k4dRPhi = 0;
        type == Type::Corrections ? getLocalCorrectionVectorCyl(zk4, rk4, phik4, side, k4dZ, k4dR, k4dRPhi) : getLocalDistortionVectorCyl(zk4, rk4, phik4, side, k4dZ, k4dR, k4dRPhi);
        const DataT k4dPhi = k4dRPhi / rk4;

        // RK4 formula. See wikipedia: u = h * 1/6 * (k1 + 2*k2 + 2*k3 + k4)
        const DataT stepsizeSixth = absstepSize / 6;
        const DataT drRK = stepsizeSixth * (k1dR + 2 * k2dR + 2 * k3dR + k4dR);
        const DataT dzRK = stepsizeSixth * (k1dZ + 2 * k2dZ + 2 * k3dZ + k4dZ);
        const DataT dphiRK = stepsizeSixth * (k1dPhi + 2 * k2dPhi + 2 * k3dPhi + k4dPhi);

        // store local distortions/corrections
        switch (type) {
          case Type::Corrections:
            mLocalCorrdR[side](iZ + 1, iR, iPhi) = drRK;
            mLocalCorrdRPhi[side](iZ + 1, iR, iPhi) = dphiRK * radius;
            mLocalCorrdZ[side](iZ + 1, iR, iPhi) = dzRK;
            break;

          case Type::Distortions:
            mLocalDistdR[side](iZ, iR, iPhi) = drRK;
            mLocalDistdRPhi[side](iZ, iR, iPhi) = dphiRK * radius;
            mLocalDistdZ[side](iZ, iR, iPhi) = dzRK;
            break;
        }
      }
      // extrapolate local distortion/correction to last/first bin using legendre polynoms with x0=0, x1=1, x2=2 and x=-1. This has to be done to ensure correct interpolation in the last,second last/first,second bin!
      switch (type) {
        case Type::Corrections:
          mLocalCorrdR[side](0, iR, iPhi) = 3 * (mLocalCorrdR[side](1, iR, iPhi) - mLocalCorrdR[side](2, iR, iPhi)) + mLocalCorrdR[side](3, iR, iPhi);
          mLocalCorrdRPhi[side](0, iR, iPhi) = 3 * (mLocalCorrdRPhi[side](1, iR, iPhi) - mLocalCorrdRPhi[side](2, iR, iPhi)) + mLocalCorrdRPhi[side](3, iR, iPhi);
          mLocalCorrdZ[side](0, iR, iPhi) = 3 * (mLocalCorrdZ[side](1, iR, iPhi) - mLocalCorrdZ[side](2, iR, iPhi)) + mLocalCorrdZ[side](3, iR, iPhi);
          break;

        case Type::Distortions:
          mLocalDistdR[side](mParamGrid.NZVertices - 1, iR, iPhi) = 3 * (mLocalDistdR[side](mParamGrid.NZVertices - 2, iR, iPhi) - mLocalDistdR[side](mParamGrid.NZVertices - 3, iR, iPhi)) + mLocalDistdR[side](mParamGrid.NZVertices - 4, iR, iPhi);
          mLocalDistdRPhi[side](mParamGrid.NZVertices - 1, iR, iPhi) = 3 * (mLocalDistdRPhi[side](mParamGrid.NZVertices - 2, iR, iPhi) - mLocalDistdRPhi[side](mParamGrid.NZVertices - 3, iR, iPhi)) + mLocalDistdRPhi[side](mParamGrid.NZVertices - 4, iR, iPhi);
          mLocalDistdZ[side](mParamGrid.NZVertices - 1, iR, iPhi) = 3 * (mLocalDistdZ[side](mParamGrid.NZVertices - 2, iR, iPhi) - mLocalDistdZ[side](mParamGrid.NZVertices - 3, iR, iPhi)) + mLocalDistdZ[side](mParamGrid.NZVertices - 4, iR, iPhi);
          break;
      }
    }
  }
  switch (type) {
    case Type::Corrections:
      mIsLocalCorrSet[side] = true;
      break;
    case Type::Distortions:
      mIsLocalDistSet[side] = true;
      break;
  }
}

template <typename DataT>
template <typename Fields>
void SpaceCharge<DataT>::calcGlobalDistortions(const Fields& formulaStruct, const int maxIterations)
{
  const Side side = formulaStruct.getSide();
  const DataT stepSize = formulaStruct.getID() == 2 ? getGridSpacingZ(side) : getGridSpacingZ(side) / sSteps; // if one used local distortions then no smaller stepsize is needed. if electric fields are used then smaller stepsize can be used
  // loop over tpc volume and let the electron drift from each vertex to the readout of the tpc
#pragma omp parallel for num_threads(sNThreads)
  for (size_t iPhi = 0; iPhi < mParamGrid.NPhiVertices; ++iPhi) {
    const DataT phi0 = getPhiVertex(iPhi, side);
    for (size_t iR = 0; iR < mParamGrid.NRVertices; ++iR) {
      const DataT r0 = getRVertex(iR, side);
      for (size_t iZ = 0; iZ < mParamGrid.NZVertices - 1; ++iZ) {
        const DataT z0 = getZVertex(iZ, side); // the electron starts at z0, r0, phi0
        DataT drDist = 0.0;                    // global distortion dR
        DataT dPhiDist = 0.0;                  // global distortion dPhi (multiplication with R has to be done at the end)
        DataT dzDist = 0.0;                    // global distortion dZ
        int iter = 0;

        for (;;) {
          if (iter > maxIterations) {
            LOGP(error, "Aborting calculation of distortions for iZ: {}, iR: {}, iPhi: {} due to iteration '{}' > maxIterations '{}'!", iZ, iR, iPhi, iter, maxIterations);
            break;
          }
          const DataT z0Tmp = z0 + dzDist + iter * stepSize; // starting z position

          if (getSide(z0Tmp) != side) {
            LOGP(error, "Aborting calculation of distortions for iZ: {}, iR: {}, iPhi: {} due to change in the sides!", iZ, iR, iPhi);
            break;
          }

          const DataT z1Tmp = z0Tmp + stepSize;                 // electron drifts from z0Tmp to z1Tmp
          const DataT radius = regulateR(r0 + drDist, side);    // current radial position of the electron
          const DataT phi = regulatePhi(phi0 + dPhiDist, side); // current phi position of the electron

          DataT ddR = 0;   // distortion dR for drift from z0Tmp to z1Tmp
          DataT ddPhi = 0; // distortion dPhi for drift from z0Tmp to z1Tmp
          DataT ddZ = 0;   // distortion dZ for drift from z0Tmp to z1Tmp

          // get the distortion from interpolation of local distortions or calculate distortions with the electric field
          processGlobalDistCorr(radius, phi, z0Tmp, z1Tmp, ddR, ddPhi, ddZ, formulaStruct);

          // if one uses local distortions the interpolated value for the last bin has to be scaled.
          // This has to be done because of the interpolated value is defined for a drift length of one z bin, but in the last bin the distance to the readout can be smaller than one z bin.
          const bool checkReached = side == Side::A ? z1Tmp >= getZMax(side) : z1Tmp <= getZMax(side);
          if (formulaStruct.getID() == 2 && checkReached) {
            const DataT fac = std::abs((getZMax(side) - z0Tmp) * getInvSpacingZ(side));
            ddR *= fac;
            ddZ *= fac;
            ddPhi *= fac;
          }

          // add local distortions to global distortions
          drDist += ddR;
          dPhiDist += ddPhi;
          dzDist += ddZ;

          // set loop to exit if the readout is reached and approximate distortion of 'missing' (one never ends exactly on the readout: z1Tmp + ddZ != ZMAX) drift distance.
          // approximation is done by the current calculated values of the distortions and scaled linear to the 'missing' distance.
          if (checkReached) {
            const DataT endPoint = z1Tmp + ddZ;
            const DataT deltaZ = getZMax(side) - endPoint; // distance from last point to read out
            const DataT diff = endPoint - z0Tmp;
            const DataT fac = diff != 0 ? std::abs(deltaZ / diff) : 0; // approximate the distortions for the 'missing' distance deltaZ
            drDist += ddR * fac;
            dPhiDist += ddPhi * fac;
            dzDist += ddZ * fac;
            break;
          }
          ++iter;
        }
        // store global distortions
        mGlobalDistdR[side](iZ, iR, iPhi) = drDist;
        mGlobalDistdRPhi[side](iZ, iR, iPhi) = dPhiDist * r0;
        mGlobalDistdZ[side](iZ, iR, iPhi) = dzDist;
      }
    }
  }
  // set flag that global distortions are set to true
  mIsGlobalDistSet[side] = true;
}

template <typename DataT>
template <typename Formulas>
void SpaceCharge<DataT>::calcGlobalCorrections(const Formulas& formulaStruct)
{
  const Side side = formulaStruct.getSide();
  const int iSteps = formulaStruct.getID() == 2 ? 1 : sSteps; // if one used local corrections no step width is needed. since it is already used for calculation of the local corrections
  const DataT stepSize = -getGridSpacingZ(side) / iSteps;
// loop over tpc volume and let the electron drift from each vertex to the readout of the tpc
#pragma omp parallel for num_threads(sNThreads)
  for (size_t iPhi = 0; iPhi < mParamGrid.NPhiVertices; ++iPhi) {
    const DataT phi0 = getPhiVertex(iPhi, side);
    for (size_t iR = 0; iR < mParamGrid.NRVertices; ++iR) {

      const DataT r0 = getRVertex(iR, side);
      DataT drCorr = 0;
      DataT dPhiCorr = 0;
      DataT dzCorr = 0;
      bool isOutOfVolume = false;

      // start at the readout and follow electron towards central electrode
      for (size_t iZ = mParamGrid.NZVertices - 1; iZ >= 1; --iZ) {
        const DataT z0 = getZVertex(iZ, side); // the electron starts at z0, r0, phi0
        // flag which is set when the central electrode is reached. if the central electrode is reached the calculation of the global corrections is aborted and the value set is the last calculated value.
        bool centralElectrodeReached = false;
        for (int iter = 0; iter < iSteps; ++iter) {
          if (centralElectrodeReached || isOutOfVolume) {
            break;
          }
          const DataT radius = regulateR(r0 + drCorr, side);     // current radial position of the electron
          const DataT phi = regulatePhi(phi0 + dPhiCorr, side);  // current phi position of the electron
          const DataT z0Tmp = z0 + dzCorr + iter * stepSize;     // starting z position
          const DataT z1Tmp = regulateZ(z0Tmp + stepSize, side); // follow electron from z0Tmp to z1Tmp
          DataT ddR = 0;                                         // distortion dR for z0Tmp to z1Tmp
          DataT ddPhi = 0;                                       // distortion dPhi for z0Tmp to z1Tmp
          DataT ddZ = 0;                                         // distortion dZ for z0Tmp to z1Tmp

          // get the distortion from interpolation of local distortions or calculate distortions with the electric field
          processGlobalDistCorr(radius, phi, z0Tmp, z1Tmp, ddR, ddPhi, ddZ, formulaStruct);

          // if one uses local corrections the interpolated value for the first bin has to be scaled.
          // This has to be done because of the interpolated value is defined for a drift length of one z bin, but in the first bin the distance to the readout can be smaller than one z bin.
          centralElectrodeReached = getSign(side) * z1Tmp <= getZMin(side);
          if (formulaStruct.getID() == 2 && centralElectrodeReached) {
            const DataT fac = (z0Tmp - getZMin(side)) * getInvSpacingZ(side);
            ddR *= fac;
            ddZ *= fac;
            ddPhi *= fac;
          }

          // calculate current r and z position after correction
          const DataT rCurr = r0 + drCorr + ddR;
          const DataT zCurr = z0Tmp + dzCorr + ddZ + stepSize;

          // check if current position lies in the TPC volume if not the electron gets corrected outside of the TPC volume and the calculation of the following corrections can be skipped
          if (rCurr <= getRMinSim(side) || rCurr >= getRMaxSim(side) || (std::abs(zCurr) > 1.2 * std::abs(getZMax(side)))) {
            isOutOfVolume = true;
            break;
          }

          // add local corrections to global corrections
          drCorr += ddR;
          dPhiCorr += ddPhi;
          dzCorr += ddZ;

          // set loop to exit if the central electrode is reached and approximate correction of 'missing' (one never ends exactly on the central electrode: z1Tmp + ddZ != ZMIN) distance.
          // approximation is done by the current calculated values of the corrections and scaled linear to the 'missing' distance deltaZ. (NOT TESTED)
          if (centralElectrodeReached) {
            const DataT endPoint = z1Tmp + ddZ;
            const DataT deltaZ = endPoint - getZMin(side);
            const DataT diff = z0Tmp - endPoint;
            const DataT fac = diff != 0 ? deltaZ / diff : 0; // approximate the distortions for the 'missing' distance deltaZ
            drCorr += ddR * fac;
            dPhiCorr += ddPhi * fac;
            dzCorr += ddZ * fac;
            break;
          }
        }
        // store global corrections
        mGlobalCorrdR[side](iZ - 1, iR, iPhi) = drCorr;
        mGlobalCorrdRPhi[side](iZ - 1, iR, iPhi) = dPhiCorr * r0;
        mGlobalCorrdZ[side](iZ - 1, iR, iPhi) = dzCorr;
      }
    }
  }
  // set flag that global corrections are set to true
  mIsGlobalCorrSet[side] = true;
}

template <typename DataT>
void SpaceCharge<DataT>::correctElectron(GlobalPosition3D& point)
{
  DataT corrX{};
  DataT corrY{};
  DataT corrZ{};
  const Side side = getSide(point.Z());

  // get the distortions for input coordinate
  getCorrections(point.X(), point.Y(), point.Z(), side, corrX, corrY, corrZ);

  // set distorted coordinates
  point.SetXYZ(point.X() + corrX, point.Y() + corrY, point.Y() + corrY);
}

template <typename DataT>
void SpaceCharge<DataT>::distortElectron(GlobalPosition3D& point) const
{
  DataT distX{};
  DataT distY{};
  DataT distZ{};
  const Side side = getSide(point.Z());
  // get the distortions for input coordinate
  getDistortions(point.X(), point.Y(), point.Z(), side, distX, distY, distZ);

  using Streamer = o2::utils::DebugStreamer;
  if (Streamer::checkStream(o2::utils::StreamFlags::streamDistortionsSC)) {
    auto& streamer = (const_cast<SpaceCharge<DataT>*>(this))->mStreamer;
    streamer.setStreamer("debug_distortions", "UPDATE");

    GlobalPosition3D pos(point);
    float phi = std::atan2(pos.Y(), pos.X());
    if (phi < 0.) {
      phi += TWOPI;
    }
    unsigned char secNum = std::floor(phi / SECPHIWIDTH);
    const Sector sector(secNum + (pos.Z() < 0) * SECTORSPERSIDE);
    LocalPosition3D lPos = Mapper::GlobalToLocal(pos, sector);

    streamer.getStreamer() << streamer.getUniqueTreeName("tree").data()
                           << "pos=" << pos
                           << "lPos=" << lPos
                           << "phi=" << phi
                           << "secNum=" << secNum
                           << "distX=" << distX
                           << "distY=" << distY
                           << "distZ=" << distZ
                           << "\n";
  }

  // set distorted coordinates
  point.SetXYZ(point.X() + distX, point.Y() + distY, point.Z() + distZ);
}

template <typename DataT>
DataT SpaceCharge<DataT>::getDensityCyl(const DataT z, const DataT r, const DataT phi, const Side side) const
{
  return mInterpolatorDensity[side](z, r, phi);
}

template <typename DataT>
DataT SpaceCharge<DataT>::getPotentialCyl(const DataT z, const DataT r, const DataT phi, const Side side) const
{
  return mInterpolatorPotential[side](z, r, phi);
}

template <typename DataT>
void SpaceCharge<DataT>::getElectricFieldsCyl(const DataT z, const DataT r, const DataT phi, const Side side, DataT& eZ, DataT& eR, DataT& ePhi) const
{
  eZ = mInterpolatorEField[side].evalEz(z, r, phi);
  eR = mInterpolatorEField[side].evalEr(z, r, phi);
  ePhi = mInterpolatorEField[side].evalEphi(z, r, phi);
}

template <typename DataT>
void SpaceCharge<DataT>::getLocalCorrectionsCyl(const DataT z, const DataT r, const DataT phi, const Side side, DataT& lcorrZ, DataT& lcorrR, DataT& lcorrRPhi) const
{
  lcorrZ = mInterpolatorLocalCorr[side].evaldZ(z, r, phi);
  lcorrR = mInterpolatorLocalCorr[side].evaldR(z, r, phi);
  lcorrRPhi = mInterpolatorLocalCorr[side].evaldRPhi(z, r, phi);
}

template <typename DataT>
void SpaceCharge<DataT>::getLocalCorrectionsCyl(const std::vector<DataT>& z, const std::vector<DataT>& r, const std::vector<DataT>& phi, const Side side, std::vector<DataT>& lcorrZ, std::vector<DataT>& lcorrR, std::vector<DataT>& lcorrRPhi) const
{
  const auto nPoints = z.size();
  lcorrZ.resize(nPoints);
  lcorrR.resize(nPoints);
  lcorrRPhi.resize(nPoints);
#pragma omp parallel for num_threads(sNThreads)
  for (size_t i = 0; i < nPoints; ++i) {
    lcorrZ[i] = mInterpolatorLocalCorr[side].evaldZ(z[i], r[i], phi[i]);
    lcorrR[i] = mInterpolatorLocalCorr[side].evaldR(z[i], r[i], phi[i]);
    lcorrRPhi[i] = mInterpolatorLocalCorr[side].evaldRPhi(z[i], r[i], phi[i]);
  }
}

template <typename DataT>
void SpaceCharge<DataT>::getCorrectionsCyl(const DataT z, const DataT r, const DataT phi, const Side side, DataT& corrZ, DataT& corrR, DataT& corrRPhi) const
{
  corrZ = mInterpolatorGlobalCorr[side].evaldZ(z, r, phi);
  corrR = mInterpolatorGlobalCorr[side].evaldR(z, r, phi);
  corrRPhi = mInterpolatorGlobalCorr[side].evaldRPhi(z, r, phi);
}

template <typename DataT>
void SpaceCharge<DataT>::getCorrectionsCyl(const std::vector<DataT>& z, const std::vector<DataT>& r, const std::vector<DataT>& phi, const Side side, std::vector<DataT>& corrZ, std::vector<DataT>& corrR, std::vector<DataT>& corrRPhi) const
{
  const auto nPoints = z.size();
  corrZ.resize(nPoints);
  corrR.resize(nPoints);
  corrRPhi.resize(nPoints);
#pragma omp parallel for num_threads(sNThreads)
  for (size_t i = 0; i < nPoints; ++i) {
    corrZ[i] = mInterpolatorGlobalCorr[side].evaldZ(z[i], r[i], phi[i]);
    corrR[i] = mInterpolatorGlobalCorr[side].evaldR(z[i], r[i], phi[i]);
    corrRPhi[i] = mInterpolatorGlobalCorr[side].evaldRPhi(z[i], r[i], phi[i]);
  }
}

template <typename DataT>
void SpaceCharge<DataT>::getCorrections(const DataT x, const DataT y, const DataT z, const Side side, DataT& corrX, DataT& corrY, DataT& corrZ) const
{
  if (mUseAnaDistCorr) {
    getCorrectionsAnalytical(x, y, z, side, corrX, corrY, corrZ);
  } else {
    // convert cartesian to polar
    const DataT radius = getRadiusFromCartesian(x, y);
    const DataT phi = getPhiFromCartesian(x, y);

    DataT corrR{};
    DataT corrRPhi{};
    getCorrectionsCyl(z, radius, phi, side, corrZ, corrR, corrRPhi);

    // Calculate corrected position
    const DataT radiusCorr = radius + corrR;
    const DataT phiCorr = phi + corrRPhi / radius;

    corrX = getXFromPolar(radiusCorr, phiCorr) - x; // difference between corrected and original x coordinate
    corrY = getYFromPolar(radiusCorr, phiCorr) - y; // difference between corrected and original y coordinate
  }
}

template <typename DataT>
void SpaceCharge<DataT>::getLocalDistortionsCyl(const DataT z, const DataT r, const DataT phi, const Side side, DataT& ldistZ, DataT& ldistR, DataT& ldistRPhi) const
{
  ldistZ = mInterpolatorLocalDist[side].evaldZ(z, r, phi);
  ldistR = mInterpolatorLocalDist[side].evaldR(z, r, phi);
  ldistRPhi = mInterpolatorLocalDist[side].evaldRPhi(z, r, phi);
}

template <typename DataT>
void SpaceCharge<DataT>::getLocalDistortionsCyl(const std::vector<DataT>& z, const std::vector<DataT>& r, const std::vector<DataT>& phi, const Side side, std::vector<DataT>& ldistZ, std::vector<DataT>& ldistR, std::vector<DataT>& ldistRPhi) const
{
  const auto nPoints = z.size();
  ldistZ.resize(nPoints);
  ldistR.resize(nPoints);
  ldistRPhi.resize(nPoints);
#pragma omp parallel for num_threads(sNThreads)
  for (size_t i = 0; i < nPoints; ++i) {
    ldistZ[i] = mInterpolatorLocalDist[side].evaldZ(z[i], r[i], phi[i]);
    ldistR[i] = mInterpolatorLocalDist[side].evaldR(z[i], r[i], phi[i]);
    ldistRPhi[i] = mInterpolatorLocalDist[side].evaldRPhi(z[i], r[i], phi[i]);
  }
}

template <typename DataT>
void SpaceCharge<DataT>::getLocalDistortionVectorCyl(const DataT z, const DataT r, const DataT phi, const Side side, DataT& lvecdistZ, DataT& lvecdistR, DataT& lvecdistRPhi) const
{
  lvecdistZ = mInterpolatorLocalVecDist[side].evaldZ(z, r, phi);
  lvecdistR = mInterpolatorLocalVecDist[side].evaldR(z, r, phi);
  lvecdistRPhi = mInterpolatorLocalVecDist[side].evaldRPhi(z, r, phi);
}

template <typename DataT>
void SpaceCharge<DataT>::getLocalDistortionVectorCyl(const std::vector<DataT>& z, const std::vector<DataT>& r, const std::vector<DataT>& phi, const Side side, std::vector<DataT>& lvecdistZ, std::vector<DataT>& lvecdistR, std::vector<DataT>& lvecdistRPhi) const
{
  const auto nPoints = z.size();
  lvecdistZ.resize(nPoints);
  lvecdistR.resize(nPoints);
  lvecdistRPhi.resize(nPoints);
#pragma omp parallel for num_threads(sNThreads)
  for (size_t i = 0; i < nPoints; ++i) {
    lvecdistZ[i] = mInterpolatorLocalVecDist[side].evaldZ(z[i], r[i], phi[i]);
    lvecdistR[i] = mInterpolatorLocalVecDist[side].evaldR(z[i], r[i], phi[i]);
    lvecdistRPhi[i] = mInterpolatorLocalVecDist[side].evaldRPhi(z[i], r[i], phi[i]);
  }
}

template <typename DataT>
void SpaceCharge<DataT>::getLocalCorrectionVectorCyl(const DataT z, const DataT r, const DataT phi, const Side side, DataT& lveccorrZ, DataT& lveccorrR, DataT& lveccorrRPhi) const
{
  lveccorrZ = -mInterpolatorLocalVecDist[side].evaldZ(z, r, phi);
  lveccorrR = -mInterpolatorLocalVecDist[side].evaldR(z, r, phi);
  lveccorrRPhi = -mInterpolatorLocalVecDist[side].evaldRPhi(z, r, phi);
}

template <typename DataT>
void SpaceCharge<DataT>::getLocalCorrectionVectorCyl(const std::vector<DataT>& z, const std::vector<DataT>& r, const std::vector<DataT>& phi, const Side side, std::vector<DataT>& lveccorrZ, std::vector<DataT>& lveccorrR, std::vector<DataT>& lveccorrRPhi) const
{
  const auto nPoints = z.size();
  lveccorrZ.resize(nPoints);
  lveccorrR.resize(nPoints);
  lveccorrRPhi.resize(nPoints);
#pragma omp parallel for num_threads(sNThreads)
  for (size_t i = 0; i < nPoints; ++i) {
    lveccorrZ[i] = -mInterpolatorLocalVecDist[side].evaldZ(z[i], r[i], phi[i]);
    lveccorrR[i] = -mInterpolatorLocalVecDist[side].evaldR(z[i], r[i], phi[i]);
    lveccorrRPhi[i] = -mInterpolatorLocalVecDist[side].evaldRPhi(z[i], r[i], phi[i]);
  }
}

template <typename DataT>
void SpaceCharge<DataT>::getDistortionsCyl(const DataT z, const DataT r, const DataT phi, const Side side, DataT& distZ, DataT& distR, DataT& distRPhi) const
{
  distZ = mInterpolatorGlobalDist[side].evaldZ(z, r, phi);
  distR = mInterpolatorGlobalDist[side].evaldR(z, r, phi);
  distRPhi = mInterpolatorGlobalDist[side].evaldRPhi(z, r, phi);
}

template <typename DataT>
void SpaceCharge<DataT>::getDistortionsCyl(const std::vector<DataT>& z, const std::vector<DataT>& r, const std::vector<DataT>& phi, const Side side, std::vector<DataT>& distZ, std::vector<DataT>& distR, std::vector<DataT>& distRPhi) const
{
  const auto nPoints = z.size();
  distZ.resize(nPoints);
  distR.resize(nPoints);
  distRPhi.resize(nPoints);
#pragma omp parallel for num_threads(sNThreads)
  for (size_t i = 0; i < nPoints; ++i) {
    distZ[i] = mInterpolatorGlobalDist[side].evaldZ(z[i], r[i], phi[i]);
    distR[i] = mInterpolatorGlobalDist[side].evaldR(z[i], r[i], phi[i]);
    distRPhi[i] = mInterpolatorGlobalDist[side].evaldRPhi(z[i], r[i], phi[i]);
  }
}

template <typename DataT>
void SpaceCharge<DataT>::getDistortions(const DataT x, const DataT y, const DataT z, const Side side, DataT& distX, DataT& distY, DataT& distZ) const
{
  if (mUseAnaDistCorr) {
    getDistortionsAnalytical(x, y, z, side, distX, distY, distZ);
  } else {
    // convert cartesian to polar
    const DataT radius = getRadiusFromCartesian(x, y);
    const DataT phi = getPhiFromCartesian(x, y);

    DataT distR{};
    DataT distRPhi{};
    getDistortionsCyl(z, radius, phi, side, distZ, distR, distRPhi);

    // Calculate distorted position
    const DataT radiusDist = radius + distR;
    const DataT phiDist = phi + distRPhi / radius;

    distX = getXFromPolar(radiusDist, phiDist) - x; // difference between distorted and original x coordinate
    distY = getYFromPolar(radiusDist, phiDist) - y; // difference between distorted and original y coordinate
  }
}

template <typename DataT>
void SpaceCharge<DataT>::getDistortionsCorrectionsAnalytical(const DataT x, const DataT y, const DataT z, const Side side, DataT& distX, DataT& distY, DataT& distZ, const bool dist) const
{
  const GlobalPosition3D pos(x, y, z);
  float phi = std::atan2(pos.Y(), pos.X());
  if (phi < 0.) {
    phi += TWOPI;
  }
  const unsigned char secNum = std::floor(phi / SECPHIWIDTH);
  const Sector sector(secNum + (pos.Z() < 0) * SECTORSPERSIDE);
  const LocalPosition3D lPos = Mapper::GlobalToLocal(pos, sector);

  // convert dlx and dlY to r, rPhi
  const DataT dlX = dist ? mAnaDistCorr.getDistortionsLX(lPos.X(), lPos.Y(), lPos.Z(), side) : mAnaDistCorr.getCorrectionsLX(lPos.X(), lPos.Y(), lPos.Z(), side);
  const DataT dlY = dist ? mAnaDistCorr.getDistortionsLY(lPos.X(), lPos.Y(), lPos.Z(), side) : mAnaDistCorr.getCorrectionsLY(lPos.X(), lPos.Y(), lPos.Z(), side);
  const DataT dlZ = dist ? mAnaDistCorr.getDistortionsLZ(lPos.X(), lPos.Y(), lPos.Z(), side) : mAnaDistCorr.getCorrectionsLZ(lPos.X(), lPos.Y(), lPos.Z(), side);

  // convert distortios in local coordinates to global coordinates
  // distorted local position
  const LocalPosition3D lPosDist(lPos.X() + dlX, lPos.Y() + dlY, lPos.Z() + dlZ);

  // calc global position
  const auto globalPosDist = Mapper::LocalToGlobal(lPosDist, sector);
  distX = globalPosDist.X() - x;
  distY = globalPosDist.Y() - y;
  distZ = globalPosDist.Z() - z;

  using Streamer = o2::utils::DebugStreamer;
  if (Streamer::checkStream(o2::utils::StreamFlags::streamDistortionsSC)) {
    auto& streamer = (const_cast<SpaceCharge<DataT>*>(this))->mStreamer;
    streamer.setStreamer("debug_distortions_analytical", "UPDATE");
    float dlXTmp = dlX;
    float dlYTmp = dlY;
    float dlZTmp = dlZ;
    auto posTmp = pos;
    auto lPosTmp = lPos;
    streamer.getStreamer() << streamer.getUniqueTreeName("tree_ana").data()
                           << "pos=" << posTmp
                           << "lPos=" << lPosTmp
                           << "dlX=" << dlXTmp
                           << "dlY=" << dlYTmp
                           << "dlZ=" << dlZTmp
                           << "distX=" << distX
                           << "distY=" << distY
                           << "distZ=" << distZ
                           << "\n";
  }
}

template <typename DataT>
void SpaceCharge<DataT>::init()
{
  using timer = std::chrono::high_resolution_clock;
  if (!mInitLookUpTables) {
    auto start = timer::now();
    auto o2field = static_cast<o2::field::MagneticField*>(TGeoGlobalMagField::Instance()->GetField());
    const float bzField = o2field->solenoidField(); // magnetic field in kGauss
    /// TODO is there a faster way to get the drift velocity
    auto& gasParam = ParameterGas::Instance();
    float vDrift = gasParam.DriftV; // drift velocity in cm/us
    /// TODO fix hard coded values (ezField, t1, t2): export to Constants.h or get from somewhere?
    const float t1 = 1.;
    const float t2 = 1.;
    /// TODO use this parameterization or fixed value(s) from Magboltz calculations?
    const float omegaTau = -10. * bzField * vDrift / std::abs(getEzField(Side::A));
    setOmegaTauT1T2(omegaTau, t1, t2);
    if (mUseInitialSCDensity) {
      LOG(warning) << "mUseInitialSCDensity" << mUseInitialSCDensity;
      calculateDistortionsCorrections(Side::A);
      calculateDistortionsCorrections(Side::C);
      mInitLookUpTables = true;
    }
    auto stop = timer::now();
    std::chrono::duration<float> time = stop - start;
    LOGP(info, "Total Time Distortions and Corrections for A and C Side: {}", time.count());
  }
}

template <typename DataT>
void SpaceCharge<DataT>::setDistortionLookupTables(const DataContainer& distdZ, const DataContainer& distdR, const DataContainer& distdRPhi, const Side side)
{
  mGlobalDistdR[side] = distdR;
  mGlobalDistdZ[side] = distdZ;
  mGlobalDistdRPhi[side] = distdRPhi;
  mIsGlobalDistSet[side] = true;
}

template <typename DataT>
template <typename Fields>
void SpaceCharge<DataT>::integrateEFieldsRoot(const DataT p1r, const DataT p1phi, const DataT p1z, const DataT p2z, DataT& localIntErOverEz, DataT& localIntEPhiOverEz, DataT& localIntDeltaEz, const Fields& formulaStruct) const
{
  const DataT ezField = getEzField(formulaStruct.getSide());
  TF1 fErOverEz(
    "fErOverEz", [&](double* x, double* p) { (void)p; return static_cast<double>(formulaStruct.evalEr(static_cast<DataT>(x[0]), p1r, p1phi) / (formulaStruct.evalEz(static_cast<DataT>(x[0]), p1r, p1phi) + ezField)); }, p1z, p2z, 1);
  localIntErOverEz = static_cast<DataT>(fErOverEz.Integral(p1z, p2z));

  TF1 fEphiOverEz(
    "fEPhiOverEz", [&](double* x, double* p) { (void)p; return static_cast<double>(formulaStruct.evalEphi(static_cast<DataT>(x[0]), p1r, p1phi) / (formulaStruct.evalEz(static_cast<DataT>(x[0]), p1r, p1phi) + ezField)); }, p1z, p2z, 1);
  localIntEPhiOverEz = static_cast<DataT>(fEphiOverEz.Integral(p1z, p2z));

  TF1 fEz(
    "fEZOverEz", [&](double* x, double* p) { (void)p; return static_cast<double>(formulaStruct.evalEz(static_cast<DataT>(x[0]), p1r, p1phi) - ezField); }, p1z, p2z, 1);
  localIntDeltaEz = getSign(formulaStruct.getSide()) * static_cast<DataT>(fEz.Integral(p1z, p2z));
}

template <typename DataT>
std::vector<std::pair<std::vector<o2::math_utils::Point3D<float>>, std::array<DataT, 3>>> SpaceCharge<DataT>::calculateElectronDriftPath(const std::vector<GlobalPosition3D>& elePos, const int nSamplingPoints, const std::string_view outFile) const
{
  const unsigned int nElectrons = elePos.size();
  std::vector<std::pair<std::vector<o2::math_utils::Point3D<float>>, std::array<DataT, 3>>> electronTracks(nElectrons);

  for (unsigned int i = 0; i < nElectrons; ++i) {
    electronTracks[i].first.reserve(nSamplingPoints + 1);
  }

  for (unsigned int i = 0; i < nElectrons; ++i) {
    const DataT z0 = elePos[i].Z();
    const DataT r0 = elePos[i].Rho();
    const DataT phi0 = elePos[i].Phi();
    const Side side = getSide(z0);
    if (!mIsEfieldSet[side]) {
      LOGP(warning, "E-Fields are not set! Calculation of drift path is not possible\n");
      continue;
    }
    const NumericalFields<DataT> numEFields{getElectricFieldsInterpolator(side)};
    const DataT stepSize = getZMax(side) / nSamplingPoints;

    DataT drDist = 0.0;   // global distortion dR
    DataT dPhiDist = 0.0; // global distortion dPhi (multiplication with R has to be done at the end)
    DataT dzDist = 0.0;   // global distortion dZ
    int iter = 0;
    for (;;) {
      const DataT z0Tmp = z0 + dzDist + iter * stepSize;     // starting z position
      const DataT z1Tmp = regulateZ(z0Tmp + stepSize, side); // electron drifts from z0Tmp to z1Tmp
      const DataT radius = r0 + drDist;                      // current radial position of the electron

      // abort calculation of drift path if electron reached inner/outer field cage or central electrode
      if (radius <= getRMin(side) || radius >= getRMax(side) || getSide(z0Tmp) != side) {
        break;
      }

      const DataT phi = regulatePhi(phi0 + dPhiDist, side); // current phi position of the electron
      electronTracks[i].first.emplace_back(GlobalPosition3D(radius * std::cos(phi), radius * std::sin(phi), z0Tmp));

      DataT ddR = 0;   // distortion dR for drift from z0Tmp to z1Tmp
      DataT ddPhi = 0; // distortion dPhi for drift from z0Tmp to z1Tmp
      DataT ddZ = 0;   // distortion dZ for drift from z0Tmp to z1Tmp

      // get the distortion from interpolation of local distortions or calculate distortions with the electric field
      processGlobalDistCorr(radius, phi, z0Tmp, z1Tmp, ddR, ddPhi, ddZ, numEFields);

      // add local distortions to global distortions
      drDist += ddR;
      dPhiDist += ddPhi;
      dzDist += ddZ;

      // if one uses local distortions the interpolated value for the last bin has to be scaled.
      // This has to be done because of the interpolated value is defined for a drift length of one z bin, but in the last bin the distance to the readout can be smaller than one z bin.
      const bool checkReached = side == Side::A ? z1Tmp >= getZMax(side) : z1Tmp <= getZMax(side);

      // set loop to exit if the readout is reached and approximate distortion of 'missing' (one never ends exactly on the readout: z1Tmp + ddZ != ZMAX) drift distance.
      // approximation is done by the current calculated values of the distortions and scaled linear to the 'missing' distance.
      if (checkReached) {
        const DataT endPoint = z1Tmp + ddZ;
        const DataT deltaZ = getZMax(side) - endPoint; // distance from last point to read out
        const DataT diff = endPoint - z0Tmp;
        const DataT fac = diff != 0 ? std::abs(deltaZ / diff) : 0; // approximate the distortions for the 'missing' distance deltaZ
        drDist += ddR * fac;
        dPhiDist += ddPhi * fac;
        dzDist += ddZ * fac;
        const DataT z1TmpEnd = regulateZ(z0Tmp + stepSize, side); // electron drifts from z0Tmp to z1Tmp
        const DataT radiusEnd = regulateR(r0 + drDist, side);     // current radial position of the electron
        const DataT phiEnd = regulatePhi(phi0 + dPhiDist, side);  // current phi position of the electron
        electronTracks[i].first.emplace_back(GlobalPosition3D(radiusEnd * std::cos(phiEnd), radiusEnd * std::sin(phiEnd), z1TmpEnd));
        break;
      }
      ++iter;
    }
    electronTracks[i].second = std::array<DataT, 3>{drDist, dPhiDist * r0, dzDist};
  }
  if (!outFile.empty()) {
    dumpElectronTracksToTree(electronTracks, nSamplingPoints, outFile.data());
  }
  return electronTracks;
}

template <typename DataT>
void SpaceCharge<DataT>::dumpElectronTracksToTree(const std::vector<std::pair<std::vector<o2::math_utils::Point3D<float>>, std::array<DataT, 3>>>& electronTracks, const int nSamplingPoints, const char* outFile) const
{
  o2::utils::TreeStreamRedirector pcstream(outFile, "RECREATE");
  pcstream.GetFile()->cd();

  auto& gasParam = ParameterGas::Instance();
  auto& eleParam = ParameterElectronics::Instance();

  for (int i = 0; i < electronTracks.size(); ++i) {
    auto electronPath = electronTracks[i].first;
    const int nPoints = electronPath.size();
    if (electronPath.empty()) {
      LOGP(warning, "Track is empty. Continue to next track.");
      continue;
    }
    std::vector<float> relDriftVel;
    relDriftVel.reserve(nPoints);

    for (int iPoint = 0; iPoint < (nPoints - 2); ++iPoint) {
      const DataT relDriftVelTmp = (electronPath[iPoint + 1].Z() - electronPath[iPoint].Z()) / getZMax(getSide(electronPath[iPoint].Z())) * nSamplingPoints; // comparison of drift distance without distortions and with distortions (rel. drift velocity)
      relDriftVel.emplace_back(std::abs(relDriftVelTmp));
    }

    // just copy the last value to avoid wrong values
    relDriftVel.emplace_back(relDriftVel.back());
    relDriftVel.emplace_back(relDriftVel.back());

    DataT distR = electronTracks[i].second[0];
    DataT distRPhi = electronTracks[i].second[1];
    DataT distZ = electronTracks[i].second[2];

    DataT driftTime = std::abs(getZMax(getSide(electronPath.front().Z())) - (distZ + electronPath.front().Z())) / gasParam.DriftV;
    DataT timeBin = driftTime / eleParam.ZbinWidth;

    pcstream << "drift"
             << "electronPath=" << electronPath
             << "relDriftVel.=" << relDriftVel // relative drift velocity in z direction
             << "distR=" << distR
             << "distRPhi=" << distRPhi
             << "distZ=" << distZ
             << "driftTime=" << driftTime
             << "timeBin=" << timeBin
             << "\n";
  }
  pcstream.Close();
}

template <typename DataT>
void SpaceCharge<DataT>::makeElectronDriftPathGif(const char* inpFile, TH2F& hDummy, const int type, const int gifSpeed, const int maxsamplingpoints, const char* outName)
{
  // read in the tree and convert to vector of std::vector<o2::tpc::GlobalPosition3D>
  TFile fInp(inpFile, "READ");
  TTree* tree = (TTree*)fInp.Get("drift");
  std::vector<o2::tpc::GlobalPosition3D>* electronPathTree = new std::vector<o2::tpc::GlobalPosition3D>;
  tree->SetBranchAddress("electronPath", &electronPathTree);

  std::vector<std::vector<o2::tpc::GlobalPosition3D>> electronPaths;
  std::vector<o2::tpc::GlobalPosition3D> elePosTmp;
  const int entries = tree->GetEntriesFast();
  for (int i = 0; i < entries; ++i) {
    tree->GetEntry(i);
    electronPaths.emplace_back(*electronPathTree);
  }
  delete electronPathTree;
  fInp.Close();

  TCanvas can("canvas", "canvas", 1000, 600);
  can.SetTopMargin(0.04f);
  can.SetRightMargin(0.04f);
  can.SetBottomMargin(0.12f);
  can.SetLeftMargin(0.11f);

  const int nElectrons = electronPaths.size();
  std::vector<int> indexStartEle(nElectrons);
  std::vector<int> countReadoutReached(nElectrons);

  // define colors of electrons
  const std::vector<int> colorsPalette{kViolet + 2, kViolet + 1, kViolet, kViolet - 1, kGreen + 3, kGreen + 2, kGreen + 1, kOrange - 1, kOrange, kOrange + 1, kOrange + 2, kRed - 1, kRed, kRed + 1, kRed + 2, kBlue - 1, kBlue, kBlue + 1, kBlue + 2};

  // create for each electron an individual graph
  unsigned int maxPoints = 0;
  std::vector<TGraph> gr(nElectrons);
  for (int i = 0; i < nElectrons; ++i) {
    gr[i].SetMarkerColor(colorsPalette[i % colorsPalette.size()]);

    if (electronPaths[i].size() > maxPoints) {
      maxPoints = electronPaths[i].size();
    }
  }

  const DataT pointsPerIteration = maxPoints / static_cast<DataT>(maxsamplingpoints);
  std::vector<DataT> zRemainder(nElectrons);

  for (;;) {
    for (auto& graph : gr) {
      graph.Set(0);
    }

    for (int iEle = 0; iEle < nElectrons; ++iEle) {
      const int nSamplingPoints = electronPaths[iEle].size();
      const int nPoints = std::round(pointsPerIteration + zRemainder[iEle]);
      zRemainder[iEle] = pointsPerIteration - nPoints;
      const auto& electronPath = electronPaths[iEle];

      if (nPoints == 0 && countReadoutReached[iEle] == 0) {
        const int indexPoint = indexStartEle[iEle];
        const DataT radius = electronPath[indexPoint].Rho();
        const DataT z = electronPath[indexPoint].Z();
        const DataT phi = electronPath[indexPoint].Phi();
        type == 0 ? gr[iEle].AddPoint(z, radius) : gr[iEle].AddPoint(phi, radius);
      }

      for (int iPoint = 0; iPoint < nPoints; ++iPoint) {
        const int indexPoint = indexStartEle[iEle];
        if (indexPoint >= nSamplingPoints) {
          countReadoutReached[iEle] = 1;
          break;
        }

        const DataT radius = electronPath[indexPoint].Rho();
        const DataT z = electronPath[indexPoint].Z();
        const DataT phi = electronPath[indexPoint].Phi();
        if (iPoint == nPoints / 2) {
          type == 0 ? gr[iEle].AddPoint(z, radius) : gr[iEle].AddPoint(phi, radius);
        }
        ++indexStartEle[iEle];
      }
    }
    hDummy.Draw();
    for (auto& graph : gr) {
      if (graph.GetN() > 0) {
        graph.Draw("P SAME");
      }
    }
    can.Print(Form("%s.gif+%i", outName, gifSpeed));

    const int sumReadoutReached = std::accumulate(countReadoutReached.begin(), countReadoutReached.end(), 0);
    if (sumReadoutReached == nElectrons) {
      break;
    }
  }

  can.Print(Form("%s.gif++", outName));
}

template <typename DataT>
void SpaceCharge<DataT>::dumpToTree(const char* outFileName, const Side side, const int nZPoints, const int nRPoints, const int nPhiPoints) const
{
  const DataT phiSpacing = GridProp::getGridSpacingPhi(nPhiPoints);
  const DataT rSpacing = GridProp::getGridSpacingR(nRPoints);
  const DataT zSpacing = side == Side::A ? GridProp::getGridSpacingZ(nZPoints) : -GridProp::getGridSpacingZ(nZPoints);

  o2::utils::TreeStreamRedirector pcstream(outFileName, "RECREATE");
  pcstream.GetFile()->cd();
  for (int iPhi = 0; iPhi < nPhiPoints; ++iPhi) {
    DataT phiPos = iPhi * phiSpacing;
    for (int iR = 0; iR < nRPoints; ++iR) {
      DataT rPos = getRMin(side) + iR * rSpacing;
      for (int iZ = 0; iZ < nZPoints; ++iZ) {
        DataT zPos = getZMin(side) + iZ * zSpacing;
        DataT density = getDensityCyl(zPos, rPos, phiPos, side);
        DataT potential = getPotentialCyl(zPos, rPos, phiPos, side);

        DataT distZ{};
        DataT distR{};
        DataT distRPhi{};
        getDistortionsCyl(zPos, rPos, phiPos, side, distZ, distR, distRPhi);

        // get average distortions
        DataT corrZ{};
        DataT corrR{};
        DataT corrRPhi{};
        getCorrectionsCyl(zPos, rPos, phiPos, side, corrZ, corrR, corrRPhi);

        DataT lcorrZ{};
        DataT lcorrR{};
        DataT lcorrRPhi{};
        getLocalCorrectionsCyl(zPos, rPos, phiPos, side, lcorrZ, lcorrR, lcorrRPhi);

        DataT ldistZ{};
        DataT ldistR{};
        DataT ldistRPhi{};
        getLocalDistortionsCyl(zPos, rPos, phiPos, side, ldistZ, ldistR, ldistRPhi);

        // get average distortions
        DataT eZ{};
        DataT eR{};
        DataT ePhi{};
        getElectricFieldsCyl(zPos, rPos, phiPos, side, eZ, eR, ePhi);

        pcstream << "sc"
                 << "phi=" << phiPos
                 << "r=" << rPos
                 << "z=" << zPos
                 << "scdensity=" << density
                 << "potential=" << potential
                 << "eZ=" << eZ
                 << "eR=" << eR
                 << "ePhi=" << ePhi
                 << "distZ=" << distZ
                 << "distR=" << distR
                 << "distRPhi=" << distRPhi
                 << "corrZ=" << corrZ
                 << "corrR=" << corrR
                 << "corrRPhi=" << corrRPhi
                 << "lcorrZ=" << lcorrZ
                 << "lcorrR=" << lcorrR
                 << "lcorrRPhi=" << lcorrRPhi
                 << "ldistZ=" << ldistZ
                 << "ldistR=" << ldistR
                 << "ldistRPhi=" << ldistRPhi
                 << "\n";
      }
    }
  }
  pcstream.Close();
}

template <typename DataT>
void SpaceCharge<DataT>::normalizeHistoQVEps0(TH3& histoIonsPhiRZ)
{
  const auto deltaPhi = histoIonsPhiRZ.GetXaxis()->GetBinWidth(1);
  const auto deltaZ = histoIonsPhiRZ.GetZaxis()->GetBinWidth(1);
  const auto fac = deltaPhi * deltaZ * o2::tpc::TPCParameters<DataT>::E0 / (2 * 100 * TMath::Qe()); // 100 to normalize to cm: vacuum permittivity [A·s/(V·cm)]
  for (int ir = 1; ir <= histoIonsPhiRZ.GetNbinsY(); ++ir) {
    const auto r0 = histoIonsPhiRZ.GetYaxis()->GetBinLowEdge(ir);
    const auto r1 = histoIonsPhiRZ.GetYaxis()->GetBinUpEdge(ir);
    const auto norm = fac * (r1 * r1 - r0 * r0);
    for (int iphi = 1; iphi <= histoIonsPhiRZ.GetNbinsX(); ++iphi) {
      for (int iz = 1; iz <= histoIonsPhiRZ.GetNbinsZ(); ++iz) {
        const auto charge = histoIonsPhiRZ.GetBinContent(iphi, ir, iz);
        histoIonsPhiRZ.SetBinContent(iphi, ir, iz, charge / norm);
      }
    }
  }
}

template <typename DataT>
int SpaceCharge<DataT>::dumpAnalyticalCorrectionsDistortions(TFile& outf) const
{
  if (!mAnaDistCorr.isValid()) {
    LOGP(info, "============== analytical functions are not set! returning ==============\n");
    return 0;
  }
  bool isOK = outf.WriteObject(&mAnaDistCorr, "analyticalDistCorr");
  return isOK;
}

template <typename DataT>
void SpaceCharge<DataT>::setAnalyticalCorrectionsDistortionsFromFile(TFile& inpf)
{
  const bool containsFormulas = inpf.GetListOfKeys()->Contains("analyticalDistCorr");
  if (!containsFormulas) {
    LOGP(info, "============== analytical functions are not stored! returning ==============\n");
    return;
  }
  LOGP(info, "Using analytical corrections and distortions");
  setUseAnalyticalDistCorr(true);
  AnalyticalDistCorr<DataT>* form = (AnalyticalDistCorr<DataT>*)inpf.Get("analyticalDistCorr");
  mAnaDistCorr = *form;
  delete form;
}

using DataTD = double;
template class o2::tpc::SpaceCharge<DataTD>;

using NumFieldsD = NumericalFields<DataTD>;
using AnaFieldsD = AnalyticalFields<DataTD>;
using DistCorrInterpD = DistCorrInterpolator<DataTD>;
using O2TPCSpaceCharge3DCalcD = SpaceCharge<DataTD>;

template void O2TPCSpaceCharge3DCalcD::integrateEFieldsRoot(const DataTD, const DataTD, const DataTD, const DataTD, DataTD&, DataTD&, DataTD&, const NumFieldsD&) const;
template void O2TPCSpaceCharge3DCalcD::integrateEFieldsRoot(const DataTD, const DataTD, const DataTD, const DataTD, DataTD&, DataTD&, DataTD&, const AnaFieldsD&) const;
template void O2TPCSpaceCharge3DCalcD::calcLocalDistortionsCorrections(const O2TPCSpaceCharge3DCalcD::Type, const NumFieldsD&);
template void O2TPCSpaceCharge3DCalcD::calcLocalDistortionsCorrections(const O2TPCSpaceCharge3DCalcD::Type, const AnaFieldsD&);
template void O2TPCSpaceCharge3DCalcD::calcLocalDistortionCorrectionVector(const NumFieldsD&);
template void O2TPCSpaceCharge3DCalcD::calcLocalDistortionCorrectionVector(const AnaFieldsD&);
template void O2TPCSpaceCharge3DCalcD::calcGlobalCorrections(const NumFieldsD&);
template void O2TPCSpaceCharge3DCalcD::calcGlobalCorrections(const AnaFieldsD&);
template void O2TPCSpaceCharge3DCalcD::calcGlobalCorrections(const DistCorrInterpD&);
template void O2TPCSpaceCharge3DCalcD::calcGlobalDistortions(const NumFieldsD&, const int maxIterations);
template void O2TPCSpaceCharge3DCalcD::calcGlobalDistortions(const AnaFieldsD&, const int maxIterations);
template void O2TPCSpaceCharge3DCalcD::calcGlobalDistortions(const DistCorrInterpD&, const int maxIterations);

using DataTF = float;
template class o2::tpc::SpaceCharge<DataTF>;

using NumFieldsF = NumericalFields<DataTF>;
using AnaFieldsF = AnalyticalFields<DataTF>;
using DistCorrInterpF = DistCorrInterpolator<DataTF>;
using O2TPCSpaceCharge3DCalcF = SpaceCharge<DataTF>;

template void O2TPCSpaceCharge3DCalcF::integrateEFieldsRoot(const DataTF, const DataTF, const DataTF, const DataTF, DataTF&, DataTF&, DataTF&, const NumFieldsF&) const;
template void O2TPCSpaceCharge3DCalcF::integrateEFieldsRoot(const DataTF, const DataTF, const DataTF, const DataTF, DataTF&, DataTF&, DataTF&, const AnaFieldsF&) const;
template void O2TPCSpaceCharge3DCalcF::calcLocalDistortionsCorrections(const O2TPCSpaceCharge3DCalcF::Type, const NumFieldsF&);
template void O2TPCSpaceCharge3DCalcF::calcLocalDistortionsCorrections(const O2TPCSpaceCharge3DCalcF::Type, const AnaFieldsF&);
template void O2TPCSpaceCharge3DCalcF::calcLocalDistortionCorrectionVector(const NumFieldsF&);
template void O2TPCSpaceCharge3DCalcF::calcLocalDistortionCorrectionVector(const AnaFieldsF&);
template void O2TPCSpaceCharge3DCalcF::calcGlobalCorrections(const NumFieldsF&);
template void O2TPCSpaceCharge3DCalcF::calcGlobalCorrections(const AnaFieldsF&);
template void O2TPCSpaceCharge3DCalcF::calcGlobalCorrections(const DistCorrInterpF&);
template void O2TPCSpaceCharge3DCalcF::calcGlobalDistortions(const NumFieldsF&, const int maxIterations);
template void O2TPCSpaceCharge3DCalcF::calcGlobalDistortions(const AnaFieldsF&, const int maxIterations);
template void O2TPCSpaceCharge3DCalcF::calcGlobalDistortions(const DistCorrInterpF&, const int maxIterations);
