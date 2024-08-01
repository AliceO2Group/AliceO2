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
#include "TGeoGlobalMagField.h"
#include "TPCBase/ParameterGas.h"
#include "TPCBase/ParameterElectronics.h"
#include "Field/MagneticField.h"
#include "CommonUtils/TreeStreamRedirector.h"
#include "TPCBase/CalDet.h"
#include "TPCBase/Painter.h"
#include "MathUtils/Utils.h"
#include "DataFormatsParameters/GRPMagField.h"
#include "GPUDebugStreamer.h"
#include "TPCBase/ParameterElectronics.h"
#include "CommonConstants/LHCConstants.h"

#include <numeric>
#include <chrono>
#include "TF1.h"
#include "TH3.h"
#include "TH2F.h"
#include "TGraph.h"
#include "TCanvas.h"
#include "TROOT.h"
#include "TStopwatch.h"
#include "ROOT/RDataFrame.hxx"
#include "THnSparse.h"

#include <random>

#if defined(WITH_OPENMP) || defined(_OPENMP)
#include <omp.h>
#else
static inline int omp_get_thread_num() { return 0; }
static inline int omp_get_max_threads() { return 1; }
#endif

templateClassImp(o2::tpc::SpaceCharge);

using namespace o2::tpc;

template <typename DataT>
SpaceCharge<DataT>::SpaceCharge(const int bfield, const unsigned short nZVertices, const unsigned short nRVertices, const unsigned short nPhiVertices, const bool initBuffers) : mParamGrid{nRVertices, nZVertices, nPhiVertices}
{
  ROOT::EnableThreadSafety();
  initBField(bfield);
  if (initBuffers) {
    initAllBuffers();
  }
};

template <typename DataT>
SpaceCharge<DataT>::SpaceCharge()
{
  ROOT::EnableThreadSafety();
}

template <typename DataT>
void SpaceCharge<DataT>::initAllBuffers()
{
  for (int iside = 0; iside < SIDES; ++iside) {
    const Side side = (iside == 0) ? Side::A : Side::C;
    initContainer(mLocalDistdR[side], true);
    initContainer(mLocalDistdZ[side], true);
    initContainer(mLocalDistdRPhi[side], true);
    initContainer(mLocalVecDistdR[side], true);
    initContainer(mLocalVecDistdZ[side], true);
    initContainer(mLocalVecDistdRPhi[side], true);
    initContainer(mLocalCorrdR[side], true);
    initContainer(mLocalCorrdZ[side], true);
    initContainer(mLocalCorrdRPhi[side], true);
    initContainer(mGlobalDistdR[side], true);
    initContainer(mGlobalDistdZ[side], true);
    initContainer(mGlobalDistdRPhi[side], true);
    initContainer(mGlobalCorrdR[side], true);
    initContainer(mGlobalCorrdZ[side], true);
    initContainer(mGlobalCorrdRPhi[side], true);
    initContainer(mDensity[side], true);
    initContainer(mPotential[side], true);
    initContainer(mElectricFieldEr[side], true);
    initContainer(mElectricFieldEz[side], true);
    initContainer(mElectricFieldEphi[side], true);
  }
}

template <typename DataT>
int SpaceCharge<DataT>::getOMPMaxThreads()
{
  return std::clamp(omp_get_max_threads(), 1, 16);
}

template <typename DataT>
void SpaceCharge<DataT>::calculateDistortionsCorrections(const o2::tpc::Side side, const bool calcVectors)
{
  using timer = std::chrono::high_resolution_clock;
  using SC = o2::tpc::SpaceCharge<DataT>;
  if (!mDensity[side].getNDataPoints()) {
    LOGP(info, "the charge is not set!");
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

  poissonSolver(side);
  calcEField(side);

  const auto numEFields = getElectricFieldsInterpolator(side);
  if (getGlobalDistType() == SC::GlobalDistType::Standard) {
    auto start = timer::now();
    const auto dist = o2::tpc::SpaceCharge<DataT>::Type::Distortions;
    calcLocalDistortionsCorrections(dist, numEFields); // local distortion calculation
    auto stop = timer::now();
    std::chrono::duration<float> time = stop - start;
    LOGP(info, "local distortions time: {}", time.count());
  } else {
    LOGP(info, "skipping local distortions (not needed)");
  }

  auto start = timer::now();
  const auto corr = o2::tpc::SpaceCharge<DataT>::Type::Corrections;
  calcLocalDistortionsCorrections(corr, numEFields); // local correction calculation
  auto stop = timer::now();
  std::chrono::duration<float> time = stop - start;
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
    (getGlobalDistCorrMethod() == SC::GlobalDistCorrMethod::LocalDistCorr) ? calcGlobalDistortions(lDistInterpolator, 3 * sSteps * getNZVertices()) : calcGlobalDistortions(numEFields, 3 * sSteps * getNZVertices());
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
  initContainer(mDensity[side], true);
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
  initContainer(mPotential[side], true);
  const auto indices = getPotentialBoundaryGEMFrameAlongRIndices(side);
  setBoundaryFromIndices(potentialFunc, indices, side);
}

template <typename DataT>
std::vector<size_t> SpaceCharge<DataT>::getPotentialBoundaryGEMFrameAlongRIndices(const Side side) const
{
  const bool simOneSectorOnly = MGParameters::normalizeGridToOneSector;
  const auto radiusStart = std::sqrt(std::pow(GEMFrameParameters<DataT>::LENGTHFRAMEIROCBOTTOM / 2, 2) + std::pow(GEMFrameParameters<DataT>::POSBOTTOM[0], 2));
  const auto rStart = getNearestRVertex(radiusStart, side);

  const auto radiusEnd = std::sqrt(std::pow(GEMFrameParameters<DataT>::LENGTHFRAMEOROC3TOP / 2, 2) + std::pow(GEMFrameParameters<DataT>::POSTOP[3], 2));
  const auto rEnd = getNearestRVertex(radiusEnd, side); // mParamGrid.NRVertices - 1

  const int verticesPerSector = simOneSectorOnly ? mParamGrid.NPhiVertices : mParamGrid.NPhiVertices / SECTORSPERSIDE;

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

  std::vector<size_t> potentialInd;
  for (size_t iR = rStart; iR < rEnd; ++iR) {
    const DataT radius = getRVertex(iR, side);
    auto const it = std::lower_bound(radii.begin(), radii.end(), radius);
    const int stack = (it == radii.end()) ? (stacks - 1) : (it - radii.begin());

    // for stack 4 use the the number of phi bins at the edge
    const auto radiusCompare = (stack == 4) ? GEMFrameParameters<DataT>::POSTOP[stack] : GEMFrameParameters<DataT>::POSTOP[stack] + (GEMFrameParameters<DataT>::POSTOP[stack] - GEMFrameParameters<DataT>::POSTOP[stack]) / 2;
    for (size_t iPhiTmp = 0; iPhiTmp < getNPhiVertices(); ++iPhiTmp) {
      const float margin = 0.5;
      const DataT offsetGlobalY = radiusCompare * iPhiTmp * getGridSpacingPhi(side) + margin;
      if (iPhiTmp > 0 && offsetGlobalY > globalPosEdgeIROC.Y()) {
        break;
      }

      for (int sector = 0; sector < (simOneSectorOnly ? 1 : SECTORSPERSIDE); ++sector) {
        const size_t iPhiLeft = sector * verticesPerSector + iPhiTmp;
        const size_t iZ = mParamGrid.NZVertices - 1;
        potentialInd.emplace_back(mPotential[side].getDataIndex(iZ, iR, iPhiLeft));
        if (iPhiTmp > 0) {
          const size_t iPhiRight = (sector + 1) * verticesPerSector - iPhiTmp;
          potentialInd.emplace_back(mPotential[side].getDataIndex(iZ, iR, iPhiRight));
        }
      }
    }
  }
  std::sort(potentialInd.begin(), potentialInd.end());
  return potentialInd;
}

template <typename DataT>
void SpaceCharge<DataT>::setPotentialBoundaryGEMFrameAlongPhi(const std::function<DataT(DataT)>& potentialFunc, const GEMstack stack, const bool bottom, const Side side, const bool outerFrame)
{
  initContainer(mPotential[side], true);
  const auto indices = getPotentialBoundaryGEMFrameAlongPhiIndices(stack, bottom, side, outerFrame);
  setBoundaryFromIndices(potentialFunc, indices, side);
}

template <typename DataT>
void SpaceCharge<DataT>::setBoundaryFromIndices(const std::function<DataT(DataT)>& potentialFunc, const std::vector<size_t>& indices, const Side side)
{
  for (const auto& index : indices) {
    const int iZ = mPotential[side].getIndexZ(index);
    const int iR = mPotential[side].getIndexR(index);
    const int iPhi = mPotential[side].getIndexPhi(index);
    const DataT radius = getRVertex(iR, side);
    mPotential[side](iZ, iR, iPhi) = potentialFunc(radius);
  }
}

template <typename DataT>
std::vector<size_t> SpaceCharge<DataT>::getPotentialBoundaryGEMFrameAlongPhiIndices(const GEMstack stack, const bool bottom, const Side side, const bool outerFrame, const bool noGap) const
{
  const bool simOneSectorOnly = MGParameters::normalizeGridToOneSector;

  // to avoid double counting
  auto indices = getPotentialBoundaryGEMFrameAlongRIndices(side);

  if (!bottom && outerFrame) {
    // if OROC3 to OFC check outer GEM frame from OROC3!
    const auto indicesOROC3 = getPotentialBoundaryGEMFrameAlongPhiIndices(GEMstack::OROC3gem, false, side, false);
    indices.insert(indices.end(), indicesOROC3.begin(), indicesOROC3.end());
    std::sort(indices.begin(), indices.end());
  } else if (bottom && outerFrame) {
    // if IROC to IFC check inner GEM frame from IROC
    const auto indicesIROC = getPotentialBoundaryGEMFrameAlongPhiIndices(GEMstack::IROCgem, true, side, false);
    indices.insert(indices.end(), indicesIROC.begin(), indicesIROC.end());
    std::sort(indices.begin(), indices.end());
  }

  int region = 0;
  float offsStart = 0;
  float offsEnd = 0;
  if (bottom) {
    offsEnd = -0.5;
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
    offsStart = 0.5;
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
  if (bottom && (stack != GEMstack::IROCgem)) {
    radiusStart -= 0.5;
  }

  auto radiusMax = bottom ? radiusFirstRow : GEMFrameParameters<DataT>::POSTOP[stack];

  // add marging for first last padrows close to gap to improve numerical stabillity there
  radiusStart += offsStart;
  radiusMax += offsEnd;

  int nVerticesR = (radiusMax - radiusStart) / getGridSpacingR(side);
  if (nVerticesR == 0) {
    nVerticesR = 1;
  }

  std::vector<size_t> potentialInd;
  const int verticesPerSector = simOneSectorOnly ? mParamGrid.NPhiVertices : mParamGrid.NPhiVertices / SECTORSPERSIDE;
  const auto nBinsPhi = (outerFrame || noGap) ? 0 : (simOneSectorOnly ? 0 : getPhiBinsGapFrame(side));
  for (int sector = 0; sector < (simOneSectorOnly ? 1 : SECTORSPERSIDE); ++sector) {
    const auto offsetPhi = sector * verticesPerSector + verticesPerSector / 2;
    for (size_t iPhiLocal = 0; iPhiLocal <= (verticesPerSector / 2 - nBinsPhi); ++iPhiLocal) {
      const auto iPhiLeft = offsetPhi + iPhiLocal;
      const auto iPhiRight = offsetPhi - iPhiLocal;
      const DataT phiLeft = getPhiVertex(iPhiLeft, side);
      const DataT phiRight = getPhiVertex(iPhiRight, side);
      const DataT localphi = getPhiVertex(iPhiLocal, side);
      const DataT radiusBottom = radiusStart / std::cos(localphi);
      auto rStart = (outerFrame && (stack == GEMstack::IROCgem)) ? 1 : std::round((radiusBottom - getRMin(side)) / getGridSpacingR(side) + 0.5); // round up to use only bins whihc are fully covered
      auto nREnd = (outerFrame && (stack == GEMstack::OROC3gem)) ? mParamGrid.NRVertices - 1 : rStart + nVerticesR;

      // end at gem frame
      if ((outerFrame && (stack == GEMstack::IROCgem))) {
        nREnd = (radiusBottom - getRVertex(1, side)) / getGridSpacingR(side) + 2; // 2 safety margin
      }

      if (rStart == 0) {
        rStart = 1;
      }

      for (size_t iR = rStart; iR < nREnd; ++iR) {
        const size_t iZ = mParamGrid.NZVertices - 1;
        if (iPhiLeft < getNPhiVertices()) {
          if (noGap || !std::binary_search(indices.begin(), indices.end(), mPotential[side].getDataIndex(iZ, iR, iPhiLeft))) {
            potentialInd.emplace_back(mPotential[side].getDataIndex(iZ, iR, iPhiLeft));
          }
        }

        if (iPhiLocal && (noGap || !std::binary_search(indices.begin(), indices.end(), mPotential[side].getDataIndex(iZ, iR, iPhiRight)))) {
          potentialInd.emplace_back(mPotential[side].getDataIndex(iZ, iR, iPhiRight));
        }
      }
    }
  }
  // remove duplicate entries
  std::unordered_set<size_t> set(potentialInd.begin(), potentialInd.end());
  potentialInd.assign(set.begin(), set.end());
  std::sort(potentialInd.begin(), potentialInd.end());
  return potentialInd;
}

template <typename DataT>
void SpaceCharge<DataT>::setPotentialBoundaryInnerRadius(const std::function<DataT(DataT)>& potentialFunc, const Side side)
{
  initContainer(mPotential[side], true);
  for (size_t iZ = 0; iZ < mParamGrid.NZVertices; ++iZ) {
    const DataT z = getZVertex(iZ, side);
    const auto pot = potentialFunc(z);
    for (size_t iPhi = 0; iPhi < mParamGrid.NPhiVertices; ++iPhi) {
      const size_t iR = 0;
      mPotential[side](iZ, iR, iPhi) += pot;
    }
  }
}

template <typename DataT>
void SpaceCharge<DataT>::setPotentialBoundaryOuterRadius(const std::function<DataT(DataT)>& potentialFunc, const Side side)
{
  initContainer(mPotential[side], true);
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
  initContainer(mPotential[side], true);
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
void SpaceCharge<DataT>::mirrorPotential(const Side sideRef, const Side sideMirrored)
{
  initContainer(mPotential[sideRef], true);
  initContainer(mPotential[sideMirrored], true);
  for (size_t iPhi = 0; iPhi < mParamGrid.NPhiVertices; ++iPhi) {
    for (size_t iR = 0; iR < mParamGrid.NRVertices; ++iR) {
      for (size_t iZ = 0; iZ < mParamGrid.NZVertices; ++iZ) {
        mPotential[sideMirrored](iZ, iR, iPhi) = mPotential[sideRef](iZ, iR, iPhi);
      }
    }
  }
}

template <typename DataT>
void SpaceCharge<DataT>::setPotentialBoundaryFromFormula(const AnalyticalFields<DataT>& formulaStruct)
{
  const Side side = formulaStruct.getSide();
  initContainer(mPotential[side], true);
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
  initContainer(mDensity[side], true);
  initContainer(mPotential[side], true);
  PoissonSolver<DataT>::setConvergenceError(stoppingConvergence);
  PoissonSolver<DataT> poissonSolver(mGrid3D[0]);
  poissonSolver.poissonSolver3D(mPotential[side], mDensity[side], symmetry);
}

template <typename DataT>
void SpaceCharge<DataT>::poissonSolver(const DataT stoppingConvergence, const int symmetry)
{
#pragma omp parallel for num_threads(2)
  for (int iside = 0; iside < FNSIDES; ++iside) {
    const o2::tpc::Side side = (iside == 0) ? Side::A : Side::C;
    poissonSolver(side, stoppingConvergence, symmetry);
  }
}

template <typename DataT>
void SpaceCharge<DataT>::setEFieldFromFormula(const AnalyticalFields<DataT>& formulaStruct)
{
  const Side side = formulaStruct.getSide();
  initContainer(mElectricFieldEr[side], true);
  initContainer(mElectricFieldEz[side], true);
  initContainer(mElectricFieldEphi[side], true);
  for (size_t iPhi = 0; iPhi < mParamGrid.NPhiVertices; ++iPhi) {
    for (size_t iR = 0; iR < mParamGrid.NRVertices; ++iR) {
      for (size_t iZ = 0; iZ < mParamGrid.NZVertices; ++iZ) {
        const DataT radius = getRVertex(iR, side);
        const DataT z = getZVertex(iZ, side);
        const DataT phi = getPhiVertex(iPhi, side);
        mElectricFieldEr[side](iZ, iR, iPhi) = formulaStruct.evalFieldR(z, radius, phi);
        mElectricFieldEz[side](iZ, iR, iPhi) = formulaStruct.evalFieldZ(z, radius, phi);
        mElectricFieldEphi[side](iZ, iR, iPhi) = formulaStruct.evalFieldPhi(z, radius, phi);
      }
    }
  }
}

template <typename DataT>
void SpaceCharge<DataT>::calcEField(const Side side)
{
  using timer = std::chrono::high_resolution_clock;
  auto start = timer::now();
  initContainer(mPotential[side], true);
  initContainer(mElectricFieldEr[side], true);
  initContainer(mElectricFieldEz[side], true);
  initContainer(mElectricFieldEphi[side], true);
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
  auto stop = timer::now();
  std::chrono::duration<float> time = stop - start;
  const float totalTime = time.count();
  LOGP(detail, "electric field calculation took {}s", totalTime);
}

template <typename DataT>
void SpaceCharge<DataT>::calcGlobalDistWithGlobalCorrIterative(const DistCorrInterpolator<DataT>& globCorr, const int maxIter, const DataT approachZ, const DataT approachR, const DataT approachPhi, const DataT diffCorr, const SpaceCharge<DataT>* scSCale, float scale)
{
  calcGlobalDistCorrIterative(globCorr, maxIter, approachZ, approachR, approachPhi, diffCorr, scSCale, scale, Type::Distortions);
}

template <typename DataT>
void SpaceCharge<DataT>::calcGlobalDistWithGlobalCorrIterative(const Side side, const SpaceCharge<DataT>* scSCale, float scale, const int maxIter, const DataT approachZ, const DataT approachR, const DataT approachPhi, const DataT diffCorr)
{
  calcGlobalDistCorrIterative(getGlobalCorrInterpolator(side), maxIter, approachZ, approachR, approachPhi, diffCorr, scSCale, scale, Type::Distortions);
}

template <typename DataT>
void SpaceCharge<DataT>::calcGlobalDistWithGlobalCorrIterative(const SpaceCharge<DataT>* scSCale, float scale, const int maxIter, const DataT approachZ, const DataT approachR, const DataT approachPhi, const DataT diffCorr)
{
#pragma omp parallel for num_threads(sNThreads)
  for (int iside = 0; iside < FNSIDES; ++iside) {
    const o2::tpc::Side side = (iside == 0) ? Side::A : Side::C;
    calcGlobalDistWithGlobalCorrIterative(side, scSCale, scale, maxIter, approachZ, approachR, approachPhi, diffCorr);
  }
}

template <typename DataT>
void SpaceCharge<DataT>::calcGlobalCorrWithGlobalDistIterative(const Side side, const SpaceCharge<DataT>* scSCale, float scale, const int maxIter, const DataT approachZ, const DataT approachR, const DataT approachPhi, const DataT diffCorr)
{
  calcGlobalDistCorrIterative(getGlobalDistInterpolator(side), maxIter, approachZ, approachR, approachPhi, diffCorr, scSCale, scale, Type::Corrections);
}

template <typename DataT>
void SpaceCharge<DataT>::calcGlobalCorrWithGlobalDistIterative(const SpaceCharge<DataT>* scSCale, float scale, const int maxIter, const DataT approachZ, const DataT approachR, const DataT approachPhi, const DataT diffCorr)
{
#pragma omp parallel for num_threads(sNThreads)
  for (int iside = 0; iside < FNSIDES; ++iside) {
    const o2::tpc::Side side = (iside == 0) ? Side::A : Side::C;
    calcGlobalCorrWithGlobalDistIterative(side, scSCale, scale, maxIter, approachZ, approachR, approachPhi, diffCorr);
  }
}

template <typename DataT>
void SpaceCharge<DataT>::calcGlobalDistCorrIterative(const DistCorrInterpolator<DataT>& globCorr, const int maxIter, const DataT approachZ, const DataT approachR, const DataT approachPhi, const DataT diffCorr, const SpaceCharge<DataT>* scSCale, float scale, const Type type)
{
  const Side side = globCorr.getSide();
  if (type == Type::Distortions) {
    initContainer(mGlobalDistdR[side], true);
    initContainer(mGlobalDistdZ[side], true);
    initContainer(mGlobalDistdRPhi[side], true);
  } else {
    initContainer(mGlobalCorrdR[side], true);
    initContainer(mGlobalCorrdZ[side], true);
    initContainer(mGlobalCorrdRPhi[side], true);
  }

  const auto& scScale = (type == Type::Distortions) ? scSCale->mInterpolatorGlobalCorr[side] : scSCale->mInterpolatorGlobalDist[side];

#pragma omp parallel for num_threads(sNThreads)
  for (unsigned int iPhi = 0; iPhi < mParamGrid.NPhiVertices; ++iPhi) {
    const DataT phi = getPhiVertex(iPhi, side);
    for (unsigned int iR = 0; iR < mParamGrid.NRVertices; ++iR) {
      const DataT radius = getRVertex(iR, side);
      for (unsigned int iZ = 1; iZ < mParamGrid.NZVertices; ++iZ) {
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
          if (scSCale && scale != 0) {
            corrdR += scale * scScale.evaldR(zCurrPos, rCurrPos, phiCurrPos);
          }
          const DataT rNewPos = rCurrPos + corrdR;

          DataT corrPhi = 0;
          if (scSCale && scale != 0) {
            corrPhi = scale * scScale.evaldRPhi(zCurrPos, rCurrPos, phiCurrPos);
          }
          corrPhi += globCorr.evaldRPhi(zCurrPos, rCurrPos, phiCurrPos);
          corrPhi /= rCurrPos;

          corrdRPhi = corrPhi * rNewPos; // normalize to new r coordinate
          const DataT phiNewPos = phiCurrPos + corrPhi;

          corrdZ = globCorr.evaldZ(zCurrPos, rCurrPos, phiCurrPos);
          if (scSCale && scale != 0) {
            corrdZ += scale * scScale.evaldZ(zCurrPos, rCurrPos, phiCurrPos);
          }
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
        if (type == Type::Distortions) {
          mGlobalDistdR[side](iZ, iR, iPhi) = -corrdR;
          mGlobalDistdRPhi[side](iZ, iR, iPhi) = -corrdRPhi;
          mGlobalDistdZ[side](iZ, iR, iPhi) = -corrdZ;
        } else {
          mGlobalCorrdR[side](iZ, iR, iPhi) = -corrdR;
          mGlobalCorrdRPhi[side](iZ, iR, iPhi) = -corrdRPhi;
          mGlobalCorrdZ[side](iZ, iR, iPhi) = -corrdZ;
        }
      }
    }
    for (unsigned int iR = 0; iR < mParamGrid.NRVertices; ++iR) {
      if (type == Type::Distortions) {
        mGlobalDistdR[side](0, iR, iPhi) = 3 * (mGlobalDistdR[side](1, iR, iPhi) - mGlobalDistdR[side](2, iR, iPhi)) + mGlobalDistdR[side](3, iR, iPhi);
        mGlobalDistdRPhi[side](0, iR, iPhi) = 3 * (mGlobalDistdRPhi[side](1, iR, iPhi) - mGlobalDistdRPhi[side](2, iR, iPhi)) + mGlobalDistdRPhi[side](3, iR, iPhi);
        mGlobalDistdZ[side](0, iR, iPhi) = 3 * (mGlobalDistdZ[side](1, iR, iPhi) - mGlobalDistdZ[side](2, iR, iPhi)) + mGlobalDistdZ[side](3, iR, iPhi);
      } else {
        mGlobalCorrdR[side](0, iR, iPhi) = 3 * (mGlobalCorrdR[side](1, iR, iPhi) - mGlobalCorrdR[side](2, iR, iPhi)) + mGlobalCorrdR[side](3, iR, iPhi);
        mGlobalCorrdRPhi[side](0, iR, iPhi) = 3 * (mGlobalCorrdRPhi[side](1, iR, iPhi) - mGlobalCorrdRPhi[side](2, iR, iPhi)) + mGlobalCorrdRPhi[side](3, iR, iPhi);
        mGlobalCorrdZ[side](0, iR, iPhi) = 3 * (mGlobalCorrdZ[side](1, iR, iPhi) - mGlobalCorrdZ[side](2, iR, iPhi)) + mGlobalCorrdZ[side](3, iR, iPhi);
      }
    }
  }
}

template <typename DataT>
NumericalFields<DataT> SpaceCharge<DataT>::getElectricFieldsInterpolator(const Side side) const
{
  if (!mElectricFieldEr[side].getNDataPoints()) {
    LOGP(warning, "============== E-Fields are not set! ==============");
  }
  NumericalFields<DataT> numFields(mElectricFieldEr[side], mElectricFieldEz[side], mElectricFieldEphi[side], mGrid3D[side], side);
  return numFields;
}

template <typename DataT>
DistCorrInterpolator<DataT> SpaceCharge<DataT>::getLocalDistInterpolator(const Side side) const
{
  if (!mLocalDistdR[side].getNDataPoints()) {
    LOGP(warning, "============== local distortions not set! ==============");
  }
  DistCorrInterpolator<DataT> numFields(mLocalDistdR[side], mLocalDistdZ[side], mLocalDistdRPhi[side], mGrid3D[side], side);
  return numFields;
}

template <typename DataT>
DistCorrInterpolator<DataT> SpaceCharge<DataT>::getLocalCorrInterpolator(const Side side) const
{
  if (!mLocalCorrdR[side].getNDataPoints()) {
    LOGP(warning, "============== local corrections not set!  ==============");
  }
  DistCorrInterpolator<DataT> numFields(mLocalCorrdR[side], mLocalCorrdZ[side], mLocalCorrdRPhi[side], mGrid3D[side], side);
  return numFields;
}

template <typename DataT>
DistCorrInterpolator<DataT> SpaceCharge<DataT>::getGlobalDistInterpolator(const Side side) const
{
  if (!mGlobalDistdR[side].getNDataPoints()) {
    LOGP(warning, "============== global distortions not set ==============");
  }
  DistCorrInterpolator<DataT> numFields(mGlobalDistdR[side], mGlobalDistdZ[side], mGlobalDistdRPhi[side], mGrid3D[side], side);
  return numFields;
}

template <typename DataT>
DistCorrInterpolator<DataT> SpaceCharge<DataT>::getGlobalCorrInterpolator(const Side side) const
{
  if (!mGlobalCorrdR[side].getNDataPoints()) {
    LOGP(warning, "============== global corrections not set ==============");
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
void SpaceCharge<DataT>::fillChargeDensityFromHisto(const TH3& hOrig)
{
  const unsigned short nBinsZNew = mParamGrid.NZVertices;
  const unsigned short nBinsRNew = mParamGrid.NRVertices;
  const unsigned short nBinsPhiNew = mParamGrid.NPhiVertices;

  initContainer(mDensity[Side::A], true);
  initContainer(mDensity[Side::C], true);

  const int nBinsZNewTwo = 2 * nBinsZNew;
  const auto phiLow = hOrig.GetXaxis()->GetBinLowEdge(1);
  const auto phiUp = hOrig.GetXaxis()->GetBinUpEdge(hOrig.GetNbinsX());
  const auto rLow = hOrig.GetYaxis()->GetBinLowEdge(1);
  const auto rUp = hOrig.GetYaxis()->GetBinUpEdge(hOrig.GetNbinsY());
  const auto zLow = hOrig.GetZaxis()->GetBinLowEdge(1);
  const auto zUp = hOrig.GetZaxis()->GetBinUpEdge(hOrig.GetNbinsZ());

  const int dim = 3;
  int bins[dim]{nBinsPhiNew, nBinsRNew, nBinsZNewTwo};
  double xmin[dim]{phiLow, rLow, zLow};
  double xmax[dim]{phiUp, rUp, zUp};
  const THnSparseF hRebin("hTmp", "hTmp", dim, bins, xmin, xmax);

#pragma omp parallel for num_threads(sNThreads)
  for (int iBinPhi = 1; iBinPhi <= nBinsPhiNew; ++iBinPhi) {
    const auto phiLowEdge = hRebin.GetAxis(0)->GetBinLowEdge(iBinPhi);
    const auto phiUpEdge = hRebin.GetAxis(0)->GetBinUpEdge(iBinPhi);

    const int phiLowBinOrig = hOrig.GetXaxis()->FindBin(phiLowEdge);
    const int phiUpBinOrig = hOrig.GetXaxis()->FindBin(phiUpEdge);

    // calculate the weights (area of original bin lies in the new bin / binwidthOrig) of the first and last bins
    const auto binWidthPhiOrig = hOrig.GetXaxis()->GetBinWidth(phiLowBinOrig);
    const auto lowerBinWeightPhi = std::abs(phiLowEdge - hOrig.GetXaxis()->GetBinUpEdge(phiLowBinOrig)) / binWidthPhiOrig;
    const auto upperBinWeightPhi = std::abs(phiUpEdge - hOrig.GetXaxis()->GetBinLowEdge(phiUpBinOrig)) / binWidthPhiOrig;

    for (int iBinR = 1; iBinR <= nBinsRNew; ++iBinR) {
      const auto rLowEdge = hRebin.GetAxis(1)->GetBinLowEdge(iBinR);
      const auto rUpEdge = hRebin.GetAxis(1)->GetBinUpEdge(iBinR);

      const int rLowBinOrig = hOrig.GetYaxis()->FindBin(rLowEdge);
      const int rUpBinOrig = hOrig.GetYaxis()->FindBin(rUpEdge);

      // calculate the weights (area of original bin lies in the new bin / binwidthOrig) of the first and last bins
      const auto binWidthROrig = hOrig.GetYaxis()->GetBinWidth(rLowBinOrig);
      const auto lowerBinWeightR = std::abs(rLowEdge - hOrig.GetYaxis()->GetBinUpEdge(rLowBinOrig)) / binWidthROrig;
      const auto upperBinWeightR = std::abs(rUpEdge - hOrig.GetYaxis()->GetBinLowEdge(rUpBinOrig)) / binWidthROrig;

      for (int iBinZ = 1; iBinZ <= nBinsZNewTwo; ++iBinZ) {
        const auto zLowEdge = hRebin.GetAxis(2)->GetBinLowEdge(iBinZ);
        const auto zUpEdge = hRebin.GetAxis(2)->GetBinUpEdge(iBinZ);
        const auto zCenter = hRebin.GetAxis(2)->GetBinCenter(iBinZ);

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
        const Side side = (iBinZ > mParamGrid.NZVertices) ? Side::A : Side::C;
        const int iZ = (side == Side::A) ? (iBinZ - mParamGrid.NZVertices - 1) : (mParamGrid.NZVertices - iBinZ);
        mDensity[side](iZ, iBinR - 1, iBinPhi - 1) = sum;
      }
    }
  }
}

template <typename DataT>
template <typename ElectricFields>
void SpaceCharge<DataT>::calcLocalDistortionsCorrections(const SpaceCharge<DataT>::Type type, const ElectricFields& formulaStruct)
{
  const Side side = formulaStruct.getSide();
  if (type == Type::Distortions) {
    initContainer(mLocalDistdR[side], true);
    initContainer(mLocalDistdZ[side], true);
    initContainer(mLocalDistdRPhi[side], true);
  } else {
    initContainer(mLocalCorrdR[side], true);
    initContainer(mLocalCorrdZ[side], true);
    initContainer(mLocalCorrdRPhi[side], true);
  }

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
          calcDistCorr(radiusTmp, phiTmp, z0Tmp, z1Tmp, ddR, ddPhi, ddZ, formulaStruct, true, side);

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
}

template <typename DataT>
template <typename ElectricFields>
void SpaceCharge<DataT>::calcLocalDistortionCorrectionVector(const ElectricFields& formulaStruct)
{
  const Side side = formulaStruct.getSide();
  initContainer(mLocalVecDistdR[side], true);
  initContainer(mLocalVecDistdZ[side], true);
  initContainer(mLocalVecDistdRPhi[side], true);
  initContainer(mElectricFieldEr[side], true);
  initContainer(mElectricFieldEz[side], true);
  initContainer(mElectricFieldEphi[side], true);
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
}

template <typename DataT>
template <typename ElectricFields>
void SpaceCharge<DataT>::calcLocalDistortionsCorrectionsRK4(const SpaceCharge<DataT>::Type type, const Side side)
{
  if (type == Type::Distortions) {
    initContainer(mLocalDistdR[side], true);
    initContainer(mLocalDistdZ[side], true);
    initContainer(mLocalDistdRPhi[side], true);
  } else {
    initContainer(mLocalCorrdR[side], true);
    initContainer(mLocalCorrdZ[side], true);
    initContainer(mLocalCorrdRPhi[side], true);
  }
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
}

template <typename DataT>
template <typename Fields>
void SpaceCharge<DataT>::calcGlobalDistortions(const Fields& formulaStruct, const int maxIterations)
{
  const Side side = formulaStruct.getSide();
  initContainer(mGlobalDistdR[side], true);
  initContainer(mGlobalDistdZ[side], true);
  initContainer(mGlobalDistdRPhi[side], true);
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

          // do not do check for first iteration
          if ((getSide(z0Tmp) != side) && iter) {
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
}

template <typename DataT>
template <typename Formulas>
void SpaceCharge<DataT>::calcGlobalCorrections(const Formulas& formulaStruct, const int type)
{
  using timer = std::chrono::high_resolution_clock;
  auto start = timer::now();
  const Side side = formulaStruct.getSide();
  initContainer(mGlobalCorrdR[side], true);
  initContainer(mGlobalCorrdZ[side], true);
  initContainer(mGlobalCorrdRPhi[side], true);

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
          if ((type != 3) && (centralElectrodeReached || isOutOfVolume)) {
            break;
          }
          DataT radius = r0 + drCorr;                        // current radial position of the electron
          DataT phi = phi0 + dPhiCorr;                       // current phi position of the electron
          const DataT z0Tmp = z0 + dzCorr + iter * stepSize; // starting z position
          DataT z1Tmp = z0Tmp + stepSize;                    // follow electron from z0Tmp to z1Tmp

          // restrict to inner TPC volume
          if (type != 3) {
            radius = regulateR(radius, side);
            phi = regulatePhi(phi, side);
            z1Tmp = regulateZ(z1Tmp, side);
          }

          DataT ddR = 0;   // distortion dR for z0Tmp to z1Tmp
          DataT ddPhi = 0; // distortion dPhi for z0Tmp to z1Tmp
          DataT ddZ = 0;   // distortion dZ for z0Tmp to z1Tmp

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
          if ((type != 3) && (rCurr <= getRMinSim(side) || rCurr >= getRMaxSim(side) || (std::abs(zCurr) > 1.2 * std::abs(getZMax(side))))) {
            isOutOfVolume = true;
            break;
          }

          // add local corrections to global corrections
          drCorr += ddR;
          dPhiCorr += ddPhi;
          dzCorr += ddZ;

          // set loop to exit if the central electrode is reached and approximate correction of 'missing' (one never ends exactly on the central electrode: z1Tmp + ddZ != ZMIN) distance.
          // approximation is done by the current calculated values of the corrections and scaled linear to the 'missing' distance deltaZ. (NOT TESTED)
          if ((type != 3) && centralElectrodeReached) {
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
        if ((type == 1 || type == 2) && (centralElectrodeReached || isOutOfVolume)) {
          mGlobalCorrdR[side](iZ - 1, iR, iPhi) = -1;
          mGlobalCorrdRPhi[side](iZ - 1, iR, iPhi) = -1;
          mGlobalCorrdZ[side](iZ - 1, iR, iPhi) = -1;
        } else {
          mGlobalCorrdR[side](iZ - 1, iR, iPhi) = drCorr;
          mGlobalCorrdRPhi[side](iZ - 1, iR, iPhi) = dPhiCorr * r0;
          mGlobalCorrdZ[side](iZ - 1, iR, iPhi) = dzCorr;
        }
      }
    }

    if (type != 0) {
      // fill / extrapolate out of volume values along r
      for (int iZ = mParamGrid.NZVertices - 1; iZ >= 0; --iZ) {
        // from middle of the radius to IFC
        for (int iR = (mParamGrid.NRVertices / 2); iR >= 0; --iR) {
          if ((mGlobalCorrdR[side](iZ, iR, iPhi) == -1) && (mGlobalCorrdRPhi[side](iZ, iR, iPhi) == -1) && (mGlobalCorrdZ[side](iZ, iR, iPhi) == -1)) {
            const size_t iRUp = iR + 1;
            if (type == 1) {
              // just replace with last valid number (assumption: values at iR==mParamGrid.NRVertices / 2 are valid)
              mGlobalCorrdR[side](iZ, iR, iPhi) = mGlobalCorrdR[side](iZ, iRUp, iPhi);
              mGlobalCorrdRPhi[side](iZ, iR, iPhi) = mGlobalCorrdR[side](iZ, iRUp, iPhi);
              mGlobalCorrdZ[side](iZ, iR, iPhi) = mGlobalCorrdR[side](iZ, iRUp, iPhi);
            } else if (type == 2) {
              // extrapolate values
              const size_t iRUpTwo = iR + 2;
              const size_t iRUpThree = iR + 3;
              mGlobalCorrdR[side](iZ, iR, iPhi) = 3 * (mGlobalCorrdR[side](iZ, iRUp, iPhi) - mGlobalCorrdR[side](iZ, iRUpTwo, iPhi)) + mGlobalCorrdR[side](iZ, iRUpThree, iPhi);
              mGlobalCorrdRPhi[side](iZ, iR, iPhi) = 3 * (mGlobalCorrdRPhi[side](iZ, iRUp, iPhi) - mGlobalCorrdRPhi[side](iZ, iRUpTwo, iPhi)) + mGlobalCorrdRPhi[side](iZ, iRUpThree, iPhi);
              mGlobalCorrdZ[side](iZ, iR, iPhi) = 3 * (mGlobalCorrdZ[side](iZ, iRUp, iPhi) - mGlobalCorrdZ[side](iZ, iRUpTwo, iPhi)) + mGlobalCorrdZ[side](iZ, iRUpThree, iPhi);
            }
          }
        }
        // from middle of the radius to OFC
        for (int iR = (mParamGrid.NRVertices / 2); iR < mParamGrid.NRVertices; ++iR) {
          if ((mGlobalCorrdR[side](iZ, iR, iPhi) == -1) && (mGlobalCorrdRPhi[side](iZ, iR, iPhi) == -1) && (mGlobalCorrdZ[side](iZ, iR, iPhi) == -1)) {
            const size_t iRUp = iR - 1;
            if (type == 1) {
              // just replace with last valid number (assumption: values at iR==mParamGrid.NRVertices / 2 are valid)
              mGlobalCorrdR[side](iZ, iR, iPhi) = mGlobalCorrdR[side](iZ, iRUp, iPhi);
              mGlobalCorrdRPhi[side](iZ, iR, iPhi) = mGlobalCorrdR[side](iZ, iRUp, iPhi);
              mGlobalCorrdZ[side](iZ, iR, iPhi) = mGlobalCorrdR[side](iZ, iRUp, iPhi);
            } else if (type == 2) {
              // extrapolate values
              const size_t iRUpTwo = iR - 2;
              const size_t iRUpThree = iR - 3;
              mGlobalCorrdR[side](iZ, iR, iPhi) = 3 * (mGlobalCorrdR[side](iZ, iRUp, iPhi) - mGlobalCorrdR[side](iZ, iRUpTwo, iPhi)) + mGlobalCorrdR[side](iZ, iRUpThree, iPhi);
              mGlobalCorrdRPhi[side](iZ, iR, iPhi) = 3 * (mGlobalCorrdRPhi[side](iZ, iRUp, iPhi) - mGlobalCorrdRPhi[side](iZ, iRUpTwo, iPhi)) + mGlobalCorrdRPhi[side](iZ, iRUpThree, iPhi);
              mGlobalCorrdZ[side](iZ, iR, iPhi) = 3 * (mGlobalCorrdZ[side](iZ, iRUp, iPhi) - mGlobalCorrdZ[side](iZ, iRUpTwo, iPhi)) + mGlobalCorrdZ[side](iZ, iRUpThree, iPhi);
            }
          }
        }
      }
    }
  }
  // set flag that global corrections are set to true
  auto stop = timer::now();
  std::chrono::duration<float> time = stop - start;
  const float totalTime = time.count();
  LOGP(detail, "calcGlobalCorrections took {}s", totalTime);
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
void SpaceCharge<DataT>::distortElectron(GlobalPosition3D& point, const SpaceCharge<DataT>* scSCale, float scale) const
{
  DataT distX{};
  DataT distY{};
  DataT distZ{};
  const Side side = getSide(point.Z());
  // get the distortions for input coordinate
  getDistortions(point.X(), point.Y(), point.Z(), side, distX, distY, distZ);

  DataT distXTmp{};
  DataT distYTmp{};
  DataT distZTmp{};

  // scale distortions if requested
  if (scSCale && scale != 0) {
    scSCale->getDistortions(point.X() + distX, point.Y() + distY, point.Z() + distZ, side, distXTmp, distYTmp, distZTmp);
    distX += distXTmp * scale;
    distY += distYTmp * scale;
    distZ += distZTmp * scale;
  }

  GPUCA_DEBUG_STREAMER_CHECK(if (o2::utils::DebugStreamer::checkStream(o2::utils::StreamFlags::streamDistortionsSC)) {
    GlobalPosition3D pos(point);
    float phi = std::atan2(pos.Y(), pos.X());
    if (phi < 0.) {
      phi += TWOPI;
    }
    unsigned char secNum = std::floor(phi / SECPHIWIDTH);
    const Sector sector(secNum + (pos.Z() < 0) * SECTORSPERSIDE);
    LocalPosition3D lPos = Mapper::GlobalToLocal(pos, sector);

    o2::utils::DebugStreamer::instance()->getStreamer("debug_distortElectron", "UPDATE") << o2::utils::DebugStreamer::instance()->getUniqueTreeName("debug_distortElectron").data()
                                                                                         << "pos=" << pos
                                                                                         << "lPos=" << lPos
                                                                                         << "phi=" << phi
                                                                                         << "secNum=" << secNum
                                                                                         << "distX=" << distX
                                                                                         << "distY=" << distY
                                                                                         << "distZ=" << distZ
                                                                                         << "distXDer=" << distXTmp
                                                                                         << "distYDer=" << distYTmp
                                                                                         << "distZDer=" << distZTmp
                                                                                         << "scale=" << scale
                                                                                         << "\n";
  })

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
std::vector<float> SpaceCharge<DataT>::getPotentialCyl(const std::vector<DataT>& z, const std::vector<DataT>& r, const std::vector<DataT>& phi, const Side side) const
{
  const auto nPoints = z.size();
  std::vector<float> potential(nPoints);
#pragma omp parallel for num_threads(sNThreads)
  for (size_t i = 0; i < nPoints; ++i) {
    potential[i] = getPotentialCyl(z[i], r[i], phi[i], side);
  }
  return potential;
}

template <typename DataT>
void SpaceCharge<DataT>::getElectricFieldsCyl(const DataT z, const DataT r, const DataT phi, const Side side, DataT& eZ, DataT& eR, DataT& ePhi) const
{
  eZ = mInterpolatorEField[side].evalFieldZ(z, r, phi);
  eR = mInterpolatorEField[side].evalFieldR(z, r, phi);
  ePhi = mInterpolatorEField[side].evalFieldPhi(z, r, phi);
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
    getLocalCorrectionsCyl(z[i], r[i], phi[i], side, lcorrZ[i], lcorrR[i], lcorrRPhi[i]);
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
    getCorrectionsCyl(z[i], r[i], phi[i], side, corrZ[i], corrR[i], corrRPhi[i]);
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
    getLocalDistortionsCyl(z[i], r[i], phi[i], side, ldistZ[i], ldistR[i], ldistRPhi[i]);
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
    getLocalDistortionVectorCyl(z[i], r[i], phi[i], side, lvecdistZ[i], lvecdistR[i], lvecdistRPhi[i]);
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
    getLocalCorrectionVectorCyl(z[i], r[i], phi[i], side, lveccorrZ[i], lveccorrR[i], lveccorrRPhi[i]);
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
    getDistortionsCyl(z[i], r[i], phi[i], side, distZ[i], distR[i], distRPhi[i]);
  }
}

template <typename DataT>
void SpaceCharge<DataT>::getDistortions(const DataT x, const DataT y, const DataT z, const Side side, DataT& distX, DataT& distY, DataT& distZ) const
{
  DataT zClamped = regulateZ(z, side);

  if (mUseAnaDistCorr) {
    getDistortionsAnalytical(x, y, zClamped, side, distX, distY, distZ);
  } else {
    // convert cartesian to polar
    const DataT radius = getRadiusFromCartesian(x, y);
    const DataT phi = getPhiFromCartesian(x, y);

    DataT distR{};
    DataT distRPhi{};
    DataT rClamped = regulateR(radius, side);
    getDistortionsCyl(zClamped, rClamped, phi, side, distZ, distR, distRPhi);

    // Calculate distorted position
    const DataT radiusDist = rClamped + distR;
    const DataT phiDist = phi + distRPhi / rClamped;

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

  GPUCA_DEBUG_STREAMER_CHECK(if (o2::utils::DebugStreamer::checkStream(o2::utils::StreamFlags::streamDistortionsSC)) {
    o2::utils::DebugStreamer::instance()->getStreamer("debug_distortions_analytical", "UPDATE") << o2::utils::DebugStreamer::instance()->getUniqueTreeName("debug_distortions_analytical").data()
                                                                                                << "pos=" << (*const_cast<GlobalPosition3D*>(&pos))
                                                                                                << "lPos=" << (*const_cast<LocalPosition3D*>(&lPos))
                                                                                                << "dlX=" << (*const_cast<DataT*>(&dlX))
                                                                                                << "dlY=" << (*const_cast<DataT*>(&dlY))
                                                                                                << "dlZ=" << (*const_cast<DataT*>(&dlZ))
                                                                                                << "distX=" << distX
                                                                                                << "distY=" << distY
                                                                                                << "distZ=" << distZ
                                                                                                << "\n";
  })
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
}

template <typename DataT>
template <typename Fields>
void SpaceCharge<DataT>::integrateEFieldsRoot(const DataT p1r, const DataT p1phi, const DataT p1z, const DataT p2z, DataT& localIntErOverEz, DataT& localIntEPhiOverEz, DataT& localIntDeltaEz, const Fields& formulaStruct, const DataT ezField, const Side side) const
{
  TF1 fErOverEz(
    "fErOverEz", [&](double* x, double* p) { (void)p; return static_cast<double>(formulaStruct.evalFieldR(static_cast<DataT>(x[0]), p1r, p1phi) / (formulaStruct.evalFieldZ(static_cast<DataT>(x[0]), p1r, p1phi) + ezField)); }, p1z, p2z, 1);
  localIntErOverEz = static_cast<DataT>(fErOverEz.Integral(p1z, p2z));

  TF1 fEphiOverEz(
    "fEPhiOverEz", [&](double* x, double* p) { (void)p; return static_cast<double>(formulaStruct.evalFieldPhi(static_cast<DataT>(x[0]), p1r, p1phi) / (formulaStruct.evalFieldZ(static_cast<DataT>(x[0]), p1r, p1phi) + ezField)); }, p1z, p2z, 1);
  localIntEPhiOverEz = static_cast<DataT>(fEphiOverEz.Integral(p1z, p2z));

  TF1 fEz(
    "fEZOverEz", [&](double* x, double* p) { (void)p; return static_cast<double>(formulaStruct.evalFieldZ(static_cast<DataT>(x[0]), p1r, p1phi) - ezField); }, p1z, p2z, 1);
  localIntDeltaEz = getSign(side) * static_cast<DataT>(fEz.Integral(p1z, p2z));
}

template <typename DataT>
template <typename Fields>
void SpaceCharge<DataT>::integrateEFieldsTrapezoidal(const DataT p1r, const DataT p1phi, const DataT p1z, const DataT p2z, DataT& localIntErOverEz, DataT& localIntEPhiOverEz, DataT& localIntDeltaEz, const Fields& formulaStruct, const DataT ezField, const Side side) const
{
  //========trapezoidal rule see: https://en.wikipedia.org/wiki/Trapezoidal_rule ==============
  const DataT fielder0 = formulaStruct.evalFieldR(p1z, p1r, p1phi);
  const DataT fieldez0 = formulaStruct.evalFieldZ(p1z, p1r, p1phi);
  const DataT fieldephi0 = formulaStruct.evalFieldPhi(p1z, p1r, p1phi);

  const DataT fielder1 = formulaStruct.evalFieldR(p2z, p1r, p1phi);
  const DataT fieldez1 = formulaStruct.evalFieldZ(p2z, p1r, p1phi);
  const DataT fieldephi1 = formulaStruct.evalFieldPhi(p2z, p1r, p1phi);

  const DataT eZ0 = isCloseToZero(ezField, fieldez0) ? 0 : 1. / (ezField + fieldez0);
  const DataT eZ1 = isCloseToZero(ezField, fieldez1) ? 0 : 1. / (ezField + fieldez1);

  const DataT deltaX = 0.5 * (p2z - p1z);
  localIntErOverEz = deltaX * (fielder0 * eZ0 + fielder1 * eZ1);
  localIntEPhiOverEz = deltaX * (fieldephi0 * eZ0 + fieldephi1 * eZ1);
  localIntDeltaEz = getSign(side) * deltaX * (fieldez0 + fieldez1);
}

template <typename DataT>
template <typename Fields>
void SpaceCharge<DataT>::integrateEFieldsSimpson(const DataT p1r, const DataT p1phi, const DataT p1z, const DataT p2z, DataT& localIntErOverEz, DataT& localIntEPhiOverEz, DataT& localIntDeltaEz, const Fields& formulaStruct, const DataT ezField, const Side side) const
{
  //==========simpsons rule see: https://en.wikipedia.org/wiki/Simpson%27s_rule =============================
  const DataT fielder0 = formulaStruct.evalFieldR(p1z, p1r, p1phi);
  const DataT fieldez0 = formulaStruct.evalFieldZ(p1z, p1r, p1phi);
  const DataT fieldephi0 = formulaStruct.evalFieldPhi(p1z, p1r, p1phi);

  const DataT fielder1 = formulaStruct.evalFieldR(p2z, p1r, p1phi);
  const DataT fieldez1 = formulaStruct.evalFieldZ(p2z, p1r, p1phi);
  const DataT fieldephi1 = formulaStruct.evalFieldPhi(p2z, p1r, p1phi);

  const DataT deltaX = p2z - p1z;
  const DataT xk2N = (p2z - static_cast<DataT>(0.5) * deltaX);
  const DataT ezField2 = formulaStruct.evalFieldZ(xk2N, p1r, p1phi);
  const DataT ezField2Denominator = isCloseToZero(ezField, ezField2) ? 0 : 1. / (ezField + ezField2);
  const DataT fieldSum2ErOverEz = formulaStruct.evalFieldR(xk2N, p1r, p1phi) * ezField2Denominator;
  const DataT fieldSum2EphiOverEz = formulaStruct.evalFieldPhi(xk2N, p1r, p1phi) * ezField2Denominator;

  const DataT eZ0 = isCloseToZero(ezField, fieldez0) ? 0 : 1. / (ezField + fieldez0);
  const DataT eZ1 = isCloseToZero(ezField, fieldez1) ? 0 : 1. / (ezField + fieldez1);

  const DataT deltaXSimpsonSixth = deltaX / 6.;
  localIntErOverEz = deltaXSimpsonSixth * (4. * fieldSum2ErOverEz + fielder0 * eZ0 + fielder1 * eZ1);
  localIntEPhiOverEz = deltaXSimpsonSixth * (4. * fieldSum2EphiOverEz + fieldephi0 * eZ0 + fieldephi1 * eZ1);
  localIntDeltaEz = getSign(side) * deltaXSimpsonSixth * (4. * ezField2 + fieldez0 + fieldez1);
}

template <typename DataT>
template <typename Fields>
void SpaceCharge<DataT>::integrateEFieldsSimpsonIterative(const DataT p1r, const DataT p2r, const DataT p1phi, const DataT p2phi, const DataT p1z, const DataT p2z, DataT& localIntErOverEz, DataT& localIntEPhiOverEz, DataT& localIntDeltaEz, const Fields& formulaStruct, const DataT ezField, const Side side) const
{
  //==========simpsons rule see: https://en.wikipedia.org/wiki/Simpson%27s_rule =============================
  // const Side side = formulaStruct.getSide();
  // const DataT ezField = getEzField(side);
  const DataT p2phiSave = regulatePhi(p2phi, side);

  const DataT fielder0 = formulaStruct.evalFieldR(p1z, p1r, p1phi);
  const DataT fieldez0 = formulaStruct.evalFieldZ(p1z, p1r, p1phi);
  const DataT fieldephi0 = formulaStruct.evalFieldPhi(p1z, p1r, p1phi);

  const DataT fielder1 = formulaStruct.evalFieldR(p2z, p2r, p2phiSave);
  const DataT fieldez1 = formulaStruct.evalFieldZ(p2z, p2r, p2phiSave);
  const DataT fieldephi1 = formulaStruct.evalFieldPhi(p2z, p2r, p2phiSave);

  const DataT eZ0Inv = isCloseToZero(ezField, fieldez0) ? 0 : 1. / (ezField + fieldez0);
  const DataT eZ1Inv = isCloseToZero(ezField, fieldez1) ? 0 : 1. / (ezField + fieldez1);

  const DataT pHalfZ = 0.5 * (p1z + p2z);                              // dont needs to be regulated since p1z and p2z are already regulated
  const DataT pHalfPhiSave = regulatePhi(0.5 * (p1phi + p2phi), side); // needs to be regulated since p2phi is not regulated
  const DataT pHalfR = 0.5 * (p1r + p2r);

  const DataT ezField2 = formulaStruct.evalFieldZ(pHalfZ, pHalfR, pHalfPhiSave);
  const DataT eZHalfInv = (isCloseToZero(ezField, ezField2) | isCloseToZero(ezField, fieldez0) | isCloseToZero(ezField, fieldez1)) ? 0 : 1. / (ezField + ezField2);
  const DataT fieldSum2ErOverEz = formulaStruct.evalFieldR(pHalfZ, pHalfR, pHalfPhiSave);
  const DataT fieldSum2EphiOverEz = formulaStruct.evalFieldPhi(pHalfZ, pHalfR, pHalfPhiSave);

  const DataT deltaXSimpsonSixth = (p2z - p1z) / 6;
  localIntErOverEz = deltaXSimpsonSixth * (4 * fieldSum2ErOverEz * eZHalfInv + fielder0 * eZ0Inv + fielder1 * eZ1Inv);
  localIntEPhiOverEz = deltaXSimpsonSixth * (4 * fieldSum2EphiOverEz * eZHalfInv + fieldephi0 * eZ0Inv + fieldephi1 * eZ1Inv);
  localIntDeltaEz = getSign(side) * deltaXSimpsonSixth * (4 * ezField2 + fieldez0 + fieldez1);
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
    if (!mElectricFieldEr[side].getNDataPoints()) {
      LOGP(warning, "E-Fields are not set! Calculation of drift path is not possible");
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
void SpaceCharge<DataT>::dumpToTree(const char* outFileName, const Side side, const int nZPoints, const int nRPoints, const int nPhiPoints, const bool randomize) const
{
  const DataT phiSpacing = GridProp::getGridSpacingPhi(nPhiPoints) / (MGParameters::normalizeGridToOneSector ? SECTORSPERSIDE : 1);
  const DataT rSpacing = GridProp::getGridSpacingR(nRPoints);
  const DataT zSpacing = side == Side::A ? GridProp::getGridSpacingZ(nZPoints) : -GridProp::getGridSpacingZ(nZPoints);

  std::uniform_real_distribution<DataT> uniR(-rSpacing / 2, rSpacing / 2);
  std::uniform_real_distribution<DataT> uniPhi(-phiSpacing / 2, phiSpacing / 2);

  std::vector<std::vector<float>> phiPosOut(nPhiPoints);
  std::vector<std::vector<float>> rPosOut(nPhiPoints);
  std::vector<std::vector<float>> zPosOut(nPhiPoints);
  std::vector<std::vector<int>> iPhiOut(nPhiPoints);
  std::vector<std::vector<int>> iROut(nPhiPoints);
  std::vector<std::vector<int>> iZOut(nPhiPoints);
  std::vector<std::vector<float>> densityOut(nPhiPoints);
  std::vector<std::vector<float>> potentialOut(nPhiPoints);
  std::vector<std::vector<float>> eZOut(nPhiPoints);
  std::vector<std::vector<float>> eROut(nPhiPoints);
  std::vector<std::vector<float>> ePhiOut(nPhiPoints);
  std::vector<std::vector<float>> distZOut(nPhiPoints);
  std::vector<std::vector<float>> distROut(nPhiPoints);
  std::vector<std::vector<float>> distRPhiOut(nPhiPoints);
  std::vector<std::vector<float>> corrZOut(nPhiPoints);
  std::vector<std::vector<float>> corrROut(nPhiPoints);
  std::vector<std::vector<float>> corrRPhiOut(nPhiPoints);
  std::vector<std::vector<float>> lcorrZOut(nPhiPoints);
  std::vector<std::vector<float>> lcorrROut(nPhiPoints);
  std::vector<std::vector<float>> lcorrRPhiOut(nPhiPoints);
  std::vector<std::vector<float>> ldistZOut(nPhiPoints);
  std::vector<std::vector<float>> ldistROut(nPhiPoints);
  std::vector<std::vector<float>> ldistRPhiOut(nPhiPoints);
  std::vector<std::vector<float>> xOut(nPhiPoints);
  std::vector<std::vector<float>> yOut(nPhiPoints);
  std::vector<std::vector<float>> bROut(nPhiPoints);
  std::vector<std::vector<float>> bZOut(nPhiPoints);
  std::vector<std::vector<float>> bPhiOut(nPhiPoints);
  std::vector<std::vector<LocalPosition3D>> lPosOut(nPhiPoints);
  std::vector<std::vector<int>> sectorOut(nPhiPoints);
  std::vector<std::vector<size_t>> globalIdxOut(nPhiPoints);

#pragma omp parallel for num_threads(sNThreads)
  for (int iPhi = 0; iPhi < nPhiPoints; ++iPhi) {
    const int nPoints = nZPoints * nRPoints;
    phiPosOut[iPhi].reserve(nPoints);
    rPosOut[iPhi].reserve(nPoints);
    zPosOut[iPhi].reserve(nPoints);
    iPhiOut[iPhi].reserve(nPoints);
    iROut[iPhi].reserve(nPoints);
    iZOut[iPhi].reserve(nPoints);
    densityOut[iPhi].reserve(nPoints);
    potentialOut[iPhi].reserve(nPoints);
    eZOut[iPhi].reserve(nPoints);
    eROut[iPhi].reserve(nPoints);
    ePhiOut[iPhi].reserve(nPoints);
    distZOut[iPhi].reserve(nPoints);
    distROut[iPhi].reserve(nPoints);
    distRPhiOut[iPhi].reserve(nPoints);
    corrZOut[iPhi].reserve(nPoints);
    corrROut[iPhi].reserve(nPoints);
    corrRPhiOut[iPhi].reserve(nPoints);
    lcorrZOut[iPhi].reserve(nPoints);
    lcorrROut[iPhi].reserve(nPoints);
    lcorrRPhiOut[iPhi].reserve(nPoints);
    ldistZOut[iPhi].reserve(nPoints);
    ldistROut[iPhi].reserve(nPoints);
    ldistRPhiOut[iPhi].reserve(nPoints);
    xOut[iPhi].reserve(nPoints);
    yOut[iPhi].reserve(nPoints);
    bROut[iPhi].reserve(nPoints);
    bZOut[iPhi].reserve(nPoints);
    bPhiOut[iPhi].reserve(nPoints);
    lPosOut[iPhi].reserve(nPoints);
    sectorOut[iPhi].reserve(nPoints);
    globalIdxOut[iPhi].reserve(nPoints);

    std::mt19937 rng(std::random_device{}());
    DataT phiPos = iPhi * phiSpacing;
    for (int iR = 0; iR < nRPoints; ++iR) {
      DataT rPos = getRMin(side) + iR * rSpacing;
      for (int iZ = 0; iZ < nZPoints; ++iZ) {
        DataT zPos = getZMin(side) + iZ * zSpacing;
        if (randomize) {
          phiPos += uniPhi(rng);
          o2::math_utils::detail::bringTo02PiGen<DataT>(phiPos);
          rPos += uniR(rng);
        }

        DataT density = getDensityCyl(zPos, rPos, phiPos, side);
        DataT potential = getPotentialCyl(zPos, rPos, phiPos, side);

        DataT distZ{};
        DataT distR{};
        DataT distRPhi{};
        getDistortionsCyl(zPos, rPos, phiPos, side, distZ, distR, distRPhi);

        DataT ldistZ{};
        DataT ldistR{};
        DataT ldistRPhi{};
        getLocalDistortionsCyl(zPos, rPos, phiPos, side, ldistZ, ldistR, ldistRPhi);

        // get average distortions
        DataT corrZ{};
        DataT corrR{};
        DataT corrRPhi{};
        // getCorrectionsCyl(zPos, rPos, phiPos, side, corrZ, corrR, corrRPhi);

        const DataT zDistorted = zPos + distZ;
        const DataT radiusDistorted = rPos + distR;
        const DataT phiDistorted = regulatePhi(phiPos + distRPhi / rPos, side);
        getCorrectionsCyl(zDistorted, radiusDistorted, phiDistorted, side, corrZ, corrR, corrRPhi);
        corrRPhi *= rPos / radiusDistorted;

        DataT lcorrZ{};
        DataT lcorrR{};
        DataT lcorrRPhi{};
        getLocalCorrectionsCyl(zPos, rPos, phiPos, side, lcorrZ, lcorrR, lcorrRPhi);

        // get average distortions
        DataT eZ{};
        DataT eR{};
        DataT ePhi{};
        getElectricFieldsCyl(zPos, rPos, phiPos, side, eZ, eR, ePhi);

        // global coordinates
        const float x = getXFromPolar(rPos, phiPos);
        const float y = getYFromPolar(rPos, phiPos);

        // b field
        const float bR = mBField.evalFieldR(zPos, rPos, phiPos);
        const float bZ = mBField.evalFieldZ(zPos, rPos, phiPos);
        const float bPhi = mBField.evalFieldPhi(zPos, rPos, phiPos);

        const LocalPosition3D pos(x, y, zPos);
        unsigned char secNum = std::floor(phiPos / SECPHIWIDTH);
        Sector sector(secNum + (pos.Z() < 0) * SECTORSPERSIDE);
        LocalPosition3D lPos = Mapper::GlobalToLocal(pos, sector);

        phiPosOut[iPhi].emplace_back(phiPos);
        rPosOut[iPhi].emplace_back(rPos);
        zPosOut[iPhi].emplace_back(zPos);
        iPhiOut[iPhi].emplace_back(iPhi);
        iROut[iPhi].emplace_back(iR);
        iZOut[iPhi].emplace_back(iZ);
        if (mDensity[side].getNDataPoints()) {
          densityOut[iPhi].emplace_back(density);
        }
        if (mPotential[side].getNDataPoints()) {
          potentialOut[iPhi].emplace_back(potential);
        }
        if (mElectricFieldEr[side].getNDataPoints()) {
          eZOut[iPhi].emplace_back(eZ);
          eROut[iPhi].emplace_back(eR);
          ePhiOut[iPhi].emplace_back(ePhi);
        }
        if (mGlobalDistdR[side].getNDataPoints()) {
          distZOut[iPhi].emplace_back(distZ);
          distROut[iPhi].emplace_back(distR);
          distRPhiOut[iPhi].emplace_back(distRPhi);
        }
        if (mGlobalCorrdR[side].getNDataPoints()) {
          corrZOut[iPhi].emplace_back(corrZ);
          corrROut[iPhi].emplace_back(corrR);
          corrRPhiOut[iPhi].emplace_back(corrRPhi);
        }
        if (mLocalCorrdR[side].getNDataPoints()) {
          lcorrZOut[iPhi].emplace_back(lcorrZ);
          lcorrROut[iPhi].emplace_back(lcorrR);
          lcorrRPhiOut[iPhi].emplace_back(lcorrRPhi);
        }
        if (mLocalDistdR[side].getNDataPoints()) {
          ldistZOut[iPhi].emplace_back(ldistZ);
          ldistROut[iPhi].emplace_back(ldistR);
          ldistRPhiOut[iPhi].emplace_back(ldistRPhi);
        }
        xOut[iPhi].emplace_back(x);
        yOut[iPhi].emplace_back(y);
        bROut[iPhi].emplace_back(bR);
        bZOut[iPhi].emplace_back(bZ);
        bPhiOut[iPhi].emplace_back(bPhi);
        lPosOut[iPhi].emplace_back(lPos);
        sectorOut[iPhi].emplace_back(sector);
        const size_t idx = (iZ + nZPoints * (iR + iPhi * nRPoints));
        globalIdxOut[iPhi].emplace_back(idx);
      }
    }
  }

  if (ROOT::IsImplicitMTEnabled() && (ROOT::GetThreadPoolSize() != sNThreads)) {
    ROOT::DisableImplicitMT();
  }
  ROOT::EnableImplicitMT(sNThreads);
  ROOT::RDataFrame dFrame(nPhiPoints);

  TStopwatch timer;
  auto dfStore = dFrame.DefineSlotEntry("x", [&xOut = xOut](unsigned int, ULong64_t entry) { return xOut[entry]; });
  dfStore = dfStore.DefineSlotEntry("y", [&yOut = yOut](unsigned int, ULong64_t entry) { return yOut[entry]; });
  dfStore = dfStore.DefineSlotEntry("phi", [&phiPosOut = phiPosOut](unsigned int, ULong64_t entry) { return phiPosOut[entry]; });
  dfStore = dfStore.DefineSlotEntry("r", [&rPosOut = rPosOut](unsigned int, ULong64_t entry) { return rPosOut[entry]; });
  dfStore = dfStore.DefineSlotEntry("z", [&zPosOut = zPosOut](unsigned int, ULong64_t entry) { return zPosOut[entry]; });
  dfStore = dfStore.DefineSlotEntry("iPhi", [&iPhiOut = iPhiOut](unsigned int, ULong64_t entry) { return iPhiOut[entry]; });
  dfStore = dfStore.DefineSlotEntry("iR", [&iROut = iROut](unsigned int, ULong64_t entry) { return iROut[entry]; });
  dfStore = dfStore.DefineSlotEntry("iZ", [&iZOut = iZOut](unsigned int, ULong64_t entry) { return iZOut[entry]; });
  dfStore = dfStore.DefineSlotEntry("lPos", [&lPosOut = lPosOut](unsigned int, ULong64_t entry) { return lPosOut[entry]; });
  dfStore = dfStore.DefineSlotEntry("sector", [&sectorOut = sectorOut](unsigned int, ULong64_t entry) { return sectorOut[entry]; });
  dfStore = dfStore.DefineSlotEntry("scdensity", [&densityOut = densityOut](unsigned int, ULong64_t entry) { return densityOut[entry]; });
  dfStore = dfStore.DefineSlotEntry("potential", [&potentialOut = potentialOut](unsigned int, ULong64_t entry) { return potentialOut[entry]; });
  dfStore = dfStore.DefineSlotEntry("eZ", [&eZOut = eZOut](unsigned int, ULong64_t entry) { return eZOut[entry]; });
  dfStore = dfStore.DefineSlotEntry("eR", [&eROut = eROut](unsigned int, ULong64_t entry) { return eROut[entry]; });
  dfStore = dfStore.DefineSlotEntry("ePhi", [&ePhiOut = ePhiOut](unsigned int, ULong64_t entry) { return ePhiOut[entry]; });
  dfStore = dfStore.DefineSlotEntry("distZ", [&distZOut = distZOut](unsigned int, ULong64_t entry) { return distZOut[entry]; });
  dfStore = dfStore.DefineSlotEntry("distR", [&distROut = distROut](unsigned int, ULong64_t entry) { return distROut[entry]; });
  dfStore = dfStore.DefineSlotEntry("distRPhi", [&distRPhiOut = distRPhiOut](unsigned int, ULong64_t entry) { return distRPhiOut[entry]; });
  dfStore = dfStore.DefineSlotEntry("corrZ", [&corrZOut = corrZOut](unsigned int, ULong64_t entry) { return corrZOut[entry]; });
  dfStore = dfStore.DefineSlotEntry("corrR", [&corrROut = corrROut](unsigned int, ULong64_t entry) { return corrROut[entry]; });
  dfStore = dfStore.DefineSlotEntry("corrRPhi", [&corrRPhiOut = corrRPhiOut](unsigned int, ULong64_t entry) { return corrRPhiOut[entry]; });
  dfStore = dfStore.DefineSlotEntry("lcorrZ", [&lcorrZOut = lcorrZOut](unsigned int, ULong64_t entry) { return lcorrZOut[entry]; });
  dfStore = dfStore.DefineSlotEntry("lcorrR", [&lcorrROut = lcorrROut](unsigned int, ULong64_t entry) { return lcorrROut[entry]; });
  dfStore = dfStore.DefineSlotEntry("lcorrRPhi", [&lcorrRPhiOut = lcorrRPhiOut](unsigned int, ULong64_t entry) { return lcorrRPhiOut[entry]; });
  dfStore = dfStore.DefineSlotEntry("ldistZ", [&ldistZOut = ldistZOut](unsigned int, ULong64_t entry) { return ldistZOut[entry]; });
  dfStore = dfStore.DefineSlotEntry("ldistR", [&ldistROut = ldistROut](unsigned int, ULong64_t entry) { return ldistROut[entry]; });
  dfStore = dfStore.DefineSlotEntry("ldistRPhi", [&ldistRPhiOut = ldistRPhiOut](unsigned int, ULong64_t entry) { return ldistRPhiOut[entry]; });
  dfStore = dfStore.DefineSlotEntry("bR", [&bROut = bROut](unsigned int, ULong64_t entry) { return bROut[entry]; });
  dfStore = dfStore.DefineSlotEntry("bZ", [&bZOut = bZOut](unsigned int, ULong64_t entry) { return bZOut[entry]; });
  dfStore = dfStore.DefineSlotEntry("bPhi", [&bPhiOut = bPhiOut](unsigned int, ULong64_t entry) { return bPhiOut[entry]; });
  dfStore = dfStore.DefineSlotEntry("globalIndex", [&globalIdxOut = globalIdxOut](unsigned int, ULong64_t entry) { return globalIdxOut[entry]; });
  dfStore.Snapshot("tree", outFileName);
  timer.Print("u");
}

template <typename DataT>
void SpaceCharge<DataT>::dumpToTree(const char* outFileName, const Sector& sector, const int nZPoints) const
{
  const Side side = sector.side();
  const DataT zSpacing = (side == Side::A) ? GridProp::getGridSpacingZ(nZPoints) : -GridProp::getGridSpacingZ(nZPoints);
  const Mapper& mapper = Mapper::instance();

  const int nPads = Mapper::getPadsInSector();
  std::vector<std::vector<float>> phiPosOut(nZPoints);
  std::vector<std::vector<float>> rPosOut(nZPoints);
  std::vector<std::vector<float>> zPosOut(nZPoints);
  std::vector<std::vector<int>> rowOut(nZPoints);
  std::vector<std::vector<float>> lxOut(nZPoints);
  std::vector<std::vector<float>> lyOut(nZPoints);
  std::vector<std::vector<float>> xOut(nZPoints);
  std::vector<std::vector<float>> yOut(nZPoints);
  std::vector<std::vector<float>> corrZOut(nZPoints);
  std::vector<std::vector<float>> corrROut(nZPoints);
  std::vector<std::vector<float>> corrRPhiOut(nZPoints);
  std::vector<std::vector<float>> erOut(nZPoints);
  std::vector<std::vector<float>> ezOut(nZPoints);
  std::vector<std::vector<float>> ephiOut(nZPoints);
  std::vector<std::vector<float>> potentialOut(nZPoints);
  std::vector<std::vector<int>> izOut(nZPoints);
  std::vector<std::vector<size_t>> globalIdxOut(nZPoints);

#pragma omp parallel for num_threads(sNThreads)
  for (int iZ = 0; iZ < nZPoints; ++iZ) {
    phiPosOut[iZ].reserve(nPads);
    rPosOut[iZ].reserve(nPads);
    zPosOut[iZ].reserve(nPads);
    corrZOut[iZ].reserve(nPads);
    corrROut[iZ].reserve(nPads);
    corrRPhiOut[iZ].reserve(nPads);
    rowOut[iZ].reserve(nPads);
    lxOut[iZ].reserve(nPads);
    lyOut[iZ].reserve(nPads);
    xOut[iZ].reserve(nPads);
    yOut[iZ].reserve(nPads);
    erOut[iZ].reserve(nPads);
    ezOut[iZ].reserve(nPads);
    ephiOut[iZ].reserve(nPads);
    izOut[iZ].reserve(nPads);
    potentialOut[iZ].reserve(nPads);
    globalIdxOut[iZ].reserve(nPads);

    DataT zPos = getZMin(side) + iZ * zSpacing;
    for (unsigned int region = 0; region < Mapper::NREGIONS; ++region) {
      for (unsigned int irow = 0; irow < Mapper::ROWSPERREGION[region]; ++irow) {
        for (unsigned int ipad = 0; ipad < Mapper::PADSPERROW[region][irow]; ++ipad) {
          GlobalPadNumber globalpad = Mapper::getGlobalPadNumber(irow, ipad, region);
          const PadCentre& padcentre = mapper.padCentre(globalpad);
          auto lx = padcentre.X();
          auto ly = padcentre.Y();
          // local to global
          auto globalPos = Mapper::LocalToGlobal(padcentre, sector);
          auto x = globalPos.X();
          auto y = globalPos.Y();

          auto r = getRadiusFromCartesian(x, y);
          auto phi = getPhiFromCartesian(x, y);
          DataT corrZ{};
          DataT corrR{};
          DataT corrRPhi{};
          getCorrectionsCyl(zPos, r, phi, side, corrZ, corrR, corrRPhi);

          DataT eZ{};
          DataT eR{};
          DataT ePhi{};
          getElectricFieldsCyl(zPos, r, phi, side, eZ, eR, ePhi);

          potentialOut[iZ].emplace_back(getPotentialCyl(zPos, r, phi, side));
          erOut[iZ].emplace_back(eR);
          ezOut[iZ].emplace_back(eZ);
          ephiOut[iZ].emplace_back(ePhi);
          phiPosOut[iZ].emplace_back(phi);
          rPosOut[iZ].emplace_back(r);
          zPosOut[iZ].emplace_back(zPos);
          corrZOut[iZ].emplace_back(corrZ);
          corrROut[iZ].emplace_back(corrR);
          corrRPhiOut[iZ].emplace_back(corrRPhi);
          rowOut[iZ].emplace_back(irow + Mapper::ROWOFFSET[region]);
          lxOut[iZ].emplace_back(lx);
          lyOut[iZ].emplace_back(ly);
          xOut[iZ].emplace_back(x);
          yOut[iZ].emplace_back(y);
          izOut[iZ].emplace_back(iZ);
          const size_t idx = globalpad + Mapper::getPadsInSector() * iZ;
          globalIdxOut[iZ].emplace_back(idx);
        }
      }
    }
  }

  if (ROOT::IsImplicitMTEnabled() && (ROOT::GetThreadPoolSize() != sNThreads)) {
    ROOT::DisableImplicitMT();
  }
  ROOT::EnableImplicitMT(sNThreads);
  ROOT::RDataFrame dFrame(nZPoints);

  TStopwatch timer;
  auto dfStore = dFrame.DefineSlotEntry("phi", [&phiPosOut = phiPosOut](unsigned int, ULong64_t entry) { return phiPosOut[entry]; });
  dfStore = dfStore.DefineSlotEntry("r", [&rPosOut = rPosOut](unsigned int, ULong64_t entry) { return rPosOut[entry]; });
  dfStore = dfStore.DefineSlotEntry("z", [&zPosOut = zPosOut](unsigned int, ULong64_t entry) { return zPosOut[entry]; });
  dfStore = dfStore.DefineSlotEntry("iz", [&izOut = izOut](unsigned int, ULong64_t entry) { return izOut[entry]; });
  dfStore = dfStore.DefineSlotEntry("corrZ", [&corrZOut = corrZOut](unsigned int, ULong64_t entry) { return corrZOut[entry]; });
  dfStore = dfStore.DefineSlotEntry("corrR", [&corrROut = corrROut](unsigned int, ULong64_t entry) { return corrROut[entry]; });
  dfStore = dfStore.DefineSlotEntry("corrRPhi", [&corrRPhiOut = corrRPhiOut](unsigned int, ULong64_t entry) { return corrRPhiOut[entry]; });
  dfStore = dfStore.DefineSlotEntry("row", [&rowOut = rowOut](unsigned int, ULong64_t entry) { return rowOut[entry]; });
  dfStore = dfStore.DefineSlotEntry("lx", [&lxOut = lxOut](unsigned int, ULong64_t entry) { return lxOut[entry]; });
  dfStore = dfStore.DefineSlotEntry("ly", [&lyOut = lyOut](unsigned int, ULong64_t entry) { return lyOut[entry]; });
  dfStore = dfStore.DefineSlotEntry("x", [&xOut = xOut](unsigned int, ULong64_t entry) { return xOut[entry]; });
  dfStore = dfStore.DefineSlotEntry("y", [&yOut = yOut](unsigned int, ULong64_t entry) { return yOut[entry]; });
  dfStore = dfStore.DefineSlotEntry("er", [&erOut = erOut](unsigned int, ULong64_t entry) { return erOut[entry]; });
  dfStore = dfStore.DefineSlotEntry("ez", [&ezOut = ezOut](unsigned int, ULong64_t entry) { return ezOut[entry]; });
  dfStore = dfStore.DefineSlotEntry("ephi", [&ephiOut = ephiOut](unsigned int, ULong64_t entry) { return ephiOut[entry]; });
  dfStore = dfStore.DefineSlotEntry("potential", [&potentialOut = potentialOut](unsigned int, ULong64_t entry) { return potentialOut[entry]; });
  dfStore = dfStore.DefineSlotEntry("globalIndex", [&globalIdxOut = globalIdxOut](unsigned int, ULong64_t entry) { return globalIdxOut[entry]; });
  dfStore.Snapshot("tree", outFileName);
  timer.Print("u");
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
    LOGP(info, "============== analytical functions are not set! returning ==============");
    return 0;
  }
  bool isOK = outf.WriteObject(&mAnaDistCorr, "analyticalDistCorr");
  return isOK;
}

template <typename DataT>
void SpaceCharge<DataT>::setAnalyticalCorrectionsDistortionsFromFile(std::string_view inpf)
{
  TFile fIn(inpf.data(), "READ");
  const bool containsFormulas = fIn.GetListOfKeys()->Contains("analyticalDistCorr");
  if (!containsFormulas) {
    LOGP(info, "============== analytical functions are not stored! returning ==============");
    return;
  }
  LOGP(info, "Using analytical corrections and distortions");
  setUseAnalyticalDistCorr(true);
  AnalyticalDistCorr<DataT>* form = (AnalyticalDistCorr<DataT>*)fIn.Get("analyticalDistCorr");
  mAnaDistCorr = *form;
  delete form;
}

template <typename DataT>
void SpaceCharge<DataT>::langevinCylindricalE(DataT& ddR, DataT& ddPhi, DataT& ddZ, const DataT radius, const DataT localIntErOverEz, const DataT localIntEPhiOverEz, const DataT localIntDeltaEz) const
{
  // calculated distortions/correction with the formula described in https://edms.cern.ch/ui/file/1108138/1/ALICE-INT-2010-016.pdf page 7.
  ddR = mC0 * localIntErOverEz + mC1 * localIntEPhiOverEz;
  ddPhi = (mC0 * localIntEPhiOverEz - mC1 * localIntErOverEz) / radius;
  ddZ = -localIntDeltaEz * TPCParameters<DataT>::DVDE;
}

template <typename DataT>
void SpaceCharge<DataT>::langevinCylindricalB(DataT& ddR, DataT& ddPhi, const DataT radius, const DataT localIntBrOverBz, const DataT localIntBPhiOverBz) const
{
  // calculated distortions/correction with the formula described in https://edms.cern.ch/ui/file/1108138/1/ALICE-INT-2010-016.pdf page 7.
  ddR = mC2 * localIntBrOverBz - mC1 * localIntBPhiOverBz;
  ddPhi = (mC2 * localIntBPhiOverBz + mC1 * localIntBrOverBz) / radius;
}

template <typename DataT>
void SpaceCharge<DataT>::initBField(const int field)
{
  const std::unordered_map<int, std::pair<int, int>> field_to_current = {{2, {12000, 6000}},
                                                                         {5, {30000, 6000}},
                                                                         {-2, {-12000, -6000}},
                                                                         {-5, {-30000, -6000}},
                                                                         {0, {0, 0}}};
  auto currents_iter = field_to_current.find(field);
  if (currents_iter == field_to_current.end()) {
    LOG(error) << " Could not lookup currents for fieldvalue " << field;
    return;
  }
  mBField.setBField(field);
  o2::parameters::GRPMagField magField;
  magField.setL3Current((*currents_iter).second.first);
  magField.setDipoleCurrent((*currents_iter).second.second);
  setBFields(magField);
}

template <typename DataT>
void SpaceCharge<DataT>::setBFields(o2::parameters::GRPMagField& magField)
{
  const float bzField = int(magField.getNominalL3Field());
  auto& gasParam = ParameterGas::Instance();
  float vDrift = gasParam.DriftV; // drift velocity in cm/us
  const float omegaTau = -10. * bzField * vDrift / std::abs(getEzField(Side::A));
  LOGP(detail, "Setting omegaTau to {} for {}kG", omegaTau, bzField);
  const float t1 = 1.;
  const float t2 = 1.;
  setOmegaTauT1T2(omegaTau, t1, t2);
}

template <typename DataT>
template <typename Fields>
void SpaceCharge<DataT>::calcDistCorr(const DataT p1r, const DataT p1phi, const DataT p1z, const DataT p2z, DataT& ddR, DataT& ddPhi, DataT& ddZ, const Fields& formulaStruct, const bool localDistCorr, const Side side) const
{
  // see: https://edms.cern.ch/ui/file/1108138/1/ALICE-INT-2010-016.pdf
  // needed for calculation of distortions/corrections
  DataT localIntErOverEz = 0;   // integral_p1z^p2z Er/Ez dz
  DataT localIntEPhiOverEz = 0; // integral_p1z^p2z Ephi/Ez dz
  DataT localIntDeltaEz = 0;    // integral_p1z^p2z Ez dz

  DataT localIntBrOverBz = 0;   // integral_p1z^p2z Br/Bz dz
  DataT localIntBPhiOverBz = 0; // integral_p1z^p2z Bphi/Bz dz
  DataT localIntDeltaBz = 0;    // integral_p1z^p2z Bz dz
  DataT ddRExB = 0;
  DataT ddPhiExB = 0;

  // there are differentnumerical integration strategys implements. for details see each function.
  switch (sNumericalIntegrationStrategy) {
    case IntegrationStrategy::SimpsonIterative:                                                         // iterative simpson integration (should be more precise at least for the analytical E-Field case but takes alot more time than normal simpson integration)
      for (int i = 0; i < sSimpsonNIteratives; ++i) {                                                   // TODO define a convergence criterion to abort the algorithm earlier for speed up.
        const DataT tmpZ = localDistCorr ? (p2z + ddZ) : regulateZ(p2z + ddZ, formulaStruct.getSide()); // dont regulate for local distortions/corrections! (to get same result as using electric field at last/first bin)
        if (mSimEDistortions) {
          integrateEFieldsSimpsonIterative(p1r, p1r + ddR + ddRExB, p1phi, p1phi + ddPhi + ddPhiExB, p1z, tmpZ, localIntErOverEz, localIntEPhiOverEz, localIntDeltaEz, formulaStruct, getEzField(side), side);
          langevinCylindricalE(ddR, ddPhi, ddZ, (p1r + 0.5 * (ddR + ddRExB)), localIntErOverEz, localIntEPhiOverEz, localIntDeltaEz); // using the mean radius '(p1r + 0.5 * ddR)' for calculation of distortions/corections
        }
        if (mSimExBMisalignment) {
          integrateEFieldsSimpsonIterative(p1r, p1r + ddR + ddRExB, p1phi, p1phi + ddPhi + ddPhiExB, p1z, tmpZ, localIntBrOverBz, localIntBPhiOverBz, localIntDeltaBz, mBField, 0, side);
          langevinCylindricalB(ddRExB, ddPhiExB, (p1r + 0.5 * (ddR + ddRExB)), localIntBrOverBz, localIntBPhiOverBz);
        }
      }
      break;
    case IntegrationStrategy::Simpson: // simpson integration
      if (mSimEDistortions) {
        integrateEFieldsSimpson(p1r, p1phi, p1z, p2z, localIntErOverEz, localIntEPhiOverEz, localIntDeltaEz, formulaStruct, getEzField(side), side);
        langevinCylindricalE(ddR, ddPhi, ddZ, p1r, localIntErOverEz, localIntEPhiOverEz, localIntDeltaEz);
      }
      if (mSimExBMisalignment) {
        integrateEFieldsSimpson(p1r, p1phi, p1z, p2z, localIntBrOverBz, localIntBPhiOverBz, localIntDeltaBz, mBField, 0, side);
        langevinCylindricalB(ddRExB, ddPhiExB, p1r, localIntBrOverBz, localIntBPhiOverBz);
      }
      break;
    case IntegrationStrategy::Trapezoidal: // trapezoidal integration (fastest)
      if (mSimEDistortions) {
        integrateEFieldsTrapezoidal(p1r, p1phi, p1z, p2z, localIntErOverEz, localIntEPhiOverEz, localIntDeltaEz, formulaStruct, getEzField(side), side);
        langevinCylindricalE(ddR, ddPhi, ddZ, p1r, localIntErOverEz, localIntEPhiOverEz, localIntDeltaEz);
      }
      if (mSimExBMisalignment) {
        integrateEFieldsTrapezoidal(p1r, p1phi, p1z, p2z, localIntBrOverBz, localIntBPhiOverBz, localIntDeltaBz, mBField, 0, side);
        langevinCylindricalB(ddRExB, ddPhiExB, p1r, localIntBrOverBz, localIntBPhiOverBz);
      }
      break;
    case IntegrationStrategy::Root: // using integration implemented in ROOT (slow)
      if (mSimEDistortions) {
        integrateEFieldsRoot(p1r, p1phi, p1z, p2z, localIntErOverEz, localIntEPhiOverEz, localIntDeltaEz, formulaStruct, getEzField(side), side);
        langevinCylindricalE(ddR, ddPhi, ddZ, p1r, localIntErOverEz, localIntEPhiOverEz, localIntDeltaEz);
      }
      if (mSimExBMisalignment) {
        integrateEFieldsRoot(p1r, p1phi, p1z, p2z, localIntBrOverBz, localIntBPhiOverBz, localIntDeltaBz, mBField, 0, side);
        langevinCylindricalB(ddRExB, ddPhiExB, p1r, localIntBrOverBz, localIntBPhiOverBz);
      }
      break;
    default:
      if (mSimEDistortions) {
        integrateEFieldsSimpson(p1r, p1phi, p1z, p2z, localIntErOverEz, localIntEPhiOverEz, localIntDeltaEz, formulaStruct, getEzField(side), side);
        langevinCylindricalE(ddR, ddPhi, ddZ, p1r, localIntErOverEz, localIntEPhiOverEz, localIntDeltaEz);
      }
      if (mSimExBMisalignment) {
        integrateEFieldsSimpson(p1r, p1phi, p1z, p2z, localIntBrOverBz, localIntBPhiOverBz, localIntDeltaBz, mBField, 0, side);
        langevinCylindricalB(ddRExB, ddPhiExB, p1r, localIntBrOverBz, localIntBPhiOverBz);
      }
  }

  GPUCA_DEBUG_STREAMER_CHECK(if (o2::utils::DebugStreamer::checkStream(o2::utils::StreamFlags::streamDistortionsSC)) {
    o2::utils::DebugStreamer::instance()->getStreamer("debug_calcDistCorr", "UPDATE") << o2::utils::DebugStreamer::instance()->getUniqueTreeName("debug_calcDistCorr").data()
                                                                                      << "p1r=" << (*const_cast<DataT*>(&p1r))
                                                                                      << "p1phi=" << (*const_cast<DataT*>(&p1phi))
                                                                                      << "p1z=" << (*const_cast<DataT*>(&p1z))
                                                                                      << "p2z=" << (*const_cast<DataT*>(&p2z))
                                                                                      << "localIntErOverEz=" << localIntErOverEz
                                                                                      << "localIntEPhiOverEz=" << localIntEPhiOverEz
                                                                                      << "localIntDeltaEz=" << localIntDeltaEz
                                                                                      << "ddR=" << ddR
                                                                                      << "ddPhi=" << ddPhi
                                                                                      << "ddZ=" << ddZ
                                                                                      << "localIntBrOverBz=" << localIntBrOverBz
                                                                                      << "localIntBPhiOverBz=" << localIntBPhiOverBz
                                                                                      << "localIntDeltaBz=" << localIntDeltaBz
                                                                                      << "ddRExB=" << ddRExB
                                                                                      << "ddPhiExB=" << ddPhiExB
                                                                                      << "\n";
  })

  ddR += ddRExB;
  ddPhi += ddPhiExB;
}

template <typename DataT>
int SpaceCharge<DataT>::dumpElectricFields(std::string_view file, const Side side, std::string_view option) const
{
  if (!mElectricFieldEr[side].getNDataPoints()) {
    LOGP(info, "============== E-Fields are not set! returning ==============");
    return 0;
  }
  const std::string sideName = getSideName(side);
  const int er = mElectricFieldEr[side].writeToFile(file, option, fmt::format("fieldEr_side{}", sideName), sNThreads);
  const int ez = mElectricFieldEz[side].writeToFile(file, "UPDATE", fmt::format("fieldEz_side{}", sideName), sNThreads);
  const int ephi = mElectricFieldEphi[side].writeToFile(file, "UPDATE", fmt::format("fieldEphi_side{}", sideName), sNThreads);
  dumpMetaData(file, "UPDATE", false);
  return er + ez + ephi;
}

template <typename DataT>
void SpaceCharge<DataT>::setElectricFieldsFromFile(std::string_view file, const Side side)
{
  const std::string sideName = getSideName(side);
  std::string_view treeEr{fmt::format("fieldEr_side{}", sideName)};
  if (!checkGridFromFile(file, treeEr)) {
    return;
  }
  initContainer(mElectricFieldEr[side], true);
  initContainer(mElectricFieldEz[side], true);
  initContainer(mElectricFieldEphi[side], true);
  mElectricFieldEr[side].initFromFile(file, treeEr, sNThreads);
  mElectricFieldEz[side].initFromFile(file, fmt::format("fieldEz_side{}", sideName), sNThreads);
  mElectricFieldEphi[side].initFromFile(file, fmt::format("fieldEphi_side{}", sideName), sNThreads);
  readMetaData(file);
}

template <typename DataT>
int SpaceCharge<DataT>::dumpGlobalDistortions(std::string_view file, const Side side, std::string_view option) const
{
  if (!mGlobalDistdR[side].getNDataPoints()) {
    LOGP(info, "============== global distortions are not set! returning ==============");
    return 0;
  }
  const std::string sideName = getSideName(side);
  const int er = mGlobalDistdR[side].writeToFile(file, option, fmt::format("distR_side{}", sideName), sNThreads);
  const int ez = mGlobalDistdZ[side].writeToFile(file, "UPDATE", fmt::format("distZ_side{}", sideName), sNThreads);
  const int ephi = mGlobalDistdRPhi[side].writeToFile(file, "UPDATE", fmt::format("distRphi_side{}", sideName), sNThreads);
  dumpMetaData(file, "UPDATE", false);
  return er + ez + ephi;
}

template <typename DataT>
void SpaceCharge<DataT>::setGlobalDistortionsFromFile(std::string_view file, const Side side)
{
  const std::string sideName = getSideName(side);
  std::string_view tree{fmt::format("distR_side{}", sideName)};
  if (!checkGridFromFile(file, tree)) {
    return;
  }
  initContainer(mGlobalDistdR[side], true);
  initContainer(mGlobalDistdZ[side], true);
  initContainer(mGlobalDistdRPhi[side], true);
  mGlobalDistdR[side].initFromFile(file, tree, sNThreads);
  mGlobalDistdZ[side].initFromFile(file, fmt::format("distZ_side{}", sideName), sNThreads);
  mGlobalDistdRPhi[side].initFromFile(file, fmt::format("distRphi_side{}", sideName), sNThreads);
  readMetaData(file);
}

template <typename DataT>
template <typename DataTIn>
void SpaceCharge<DataT>::setGlobalDistortionsFromFile(TFile& inpf, const Side side)
{
  initContainer(mGlobalDistdR[side], false);
  initContainer(mGlobalDistdZ[side], false);
  initContainer(mGlobalDistdRPhi[side], false);
  const std::string sideName = getSideName(side);
  mGlobalDistdR[side].template initFromFile<DataTIn>(inpf, fmt::format("distR_side{}", sideName).data());
  mGlobalDistdZ[side].template initFromFile<DataTIn>(inpf, fmt::format("distZ_side{}", sideName).data());
  mGlobalDistdRPhi[side].template initFromFile<DataTIn>(inpf, fmt::format("distRphi_side{}", sideName).data());
}

template <typename DataT>
int SpaceCharge<DataT>::dumpGlobalCorrections(std::string_view file, const Side side, std::string_view option) const
{
  if (!mGlobalCorrdR[side].getNDataPoints()) {
    LOGP(info, "============== global corrections are not set! returning ==============");
    return 0;
  }
  const std::string sideName = getSideName(side);
  const int er = mGlobalCorrdR[side].writeToFile(file, option, fmt::format("corrR_side{}", sideName), sNThreads);
  const int ez = mGlobalCorrdZ[side].writeToFile(file, "UPDATE", fmt::format("corrZ_side{}", sideName), sNThreads);
  const int ephi = mGlobalCorrdRPhi[side].writeToFile(file, "UPDATE", fmt::format("corrRPhi_side{}", sideName), sNThreads);
  dumpMetaData(file, "UPDATE", false);
  return er + ez + ephi;
}

template <typename DataT>
void SpaceCharge<DataT>::setGlobalCorrectionsFromFile(std::string_view file, const Side side)
{
  const std::string sideName = getSideName(side);
  const std::string_view treename{fmt::format("corrR_side{}", getSideName(side))};
  if (!checkGridFromFile(file, treename)) {
    return;
  }

  initContainer(mGlobalCorrdR[side], true);
  initContainer(mGlobalCorrdZ[side], true);
  initContainer(mGlobalCorrdRPhi[side], true);
  mGlobalCorrdR[side].initFromFile(file, treename, sNThreads);
  mGlobalCorrdZ[side].initFromFile(file, fmt::format("corrZ_side{}", sideName), sNThreads);
  mGlobalCorrdRPhi[side].initFromFile(file, fmt::format("corrRPhi_side{}", sideName), sNThreads);
  readMetaData(file);
}

template <typename DataT>
template <typename DataTIn>
void SpaceCharge<DataT>::setGlobalCorrectionsFromFile(TFile& inpf, const Side side)
{
  initContainer(mGlobalCorrdR[side], false);
  initContainer(mGlobalCorrdZ[side], false);
  initContainer(mGlobalCorrdRPhi[side], false);
  const std::string sideName = getSideName(side);
  mGlobalCorrdR[side].template initFromFile<DataTIn>(inpf, fmt::format("corrR_side{}", sideName).data());
  mGlobalCorrdZ[side].template initFromFile<DataTIn>(inpf, fmt::format("corrZ_side{}", sideName).data());
  mGlobalCorrdRPhi[side].template initFromFile<DataTIn>(inpf, fmt::format("corrRPhi_side{}", sideName).data());
}

template <typename DataT>
int SpaceCharge<DataT>::dumpLocalCorrections(std::string_view file, const Side side, std::string_view option) const
{
  if (!mLocalCorrdR[side].getNDataPoints()) {
    LOGP(info, "============== local corrections are not set! returning ==============");
    return 0;
  }
  const std::string sideName = getSideName(side);
  const int lCorrdR = mLocalCorrdR[side].writeToFile(file, option, fmt::format("lcorrR_side{}", sideName), sNThreads);
  const int lCorrdZ = mLocalCorrdZ[side].writeToFile(file, "UPDATE", fmt::format("lcorrZ_side{}", sideName), sNThreads);
  const int lCorrdRPhi = mLocalCorrdRPhi[side].writeToFile(file, "UPDATE", fmt::format("lcorrRPhi_side{}", sideName), sNThreads);
  dumpMetaData(file, "UPDATE", false);
  return lCorrdR + lCorrdZ + lCorrdRPhi;
}

template <typename DataT>
void SpaceCharge<DataT>::setLocalCorrectionsFromFile(std::string_view file, const Side side)
{
  const std::string sideName = getSideName(side);
  const std::string_view treename{fmt::format("lcorrR_side{}", getSideName(side))};
  if (!checkGridFromFile(file, treename)) {
    return;
  }
  initContainer(mLocalCorrdR[side], true);
  initContainer(mLocalCorrdZ[side], true);
  initContainer(mLocalCorrdRPhi[side], true);
  const bool lCorrdR = mLocalCorrdR[side].initFromFile(file, treename, sNThreads);
  const bool lCorrdZ = mLocalCorrdZ[side].initFromFile(file, fmt::format("lcorrZ_side{}", sideName), sNThreads);
  const bool lCorrdRPhi = mLocalCorrdRPhi[side].initFromFile(file, fmt::format("lcorrRPhi_side{}", sideName), sNThreads);
  readMetaData(file);
}

template <typename DataT>
int SpaceCharge<DataT>::dumpLocalDistortions(std::string_view file, const Side side, std::string_view option) const
{
  if (!mLocalDistdR[side].getNDataPoints()) {
    LOGP(info, "============== local distortions are not set! returning ==============");
    return 0;
  }
  const std::string sideName = getSideName(side);
  const int lDistdR = mLocalDistdR[side].writeToFile(file, option, fmt::format("ldistR_side{}", sideName), sNThreads);
  const int lDistdZ = mLocalDistdZ[side].writeToFile(file, "UPDATE", fmt::format("ldistZ_side{}", sideName), sNThreads);
  const int lDistdRPhi = mLocalDistdRPhi[side].writeToFile(file, "UPDATE", fmt::format("ldistRPhi_side{}", sideName), sNThreads);
  dumpMetaData(file, "UPDATE", false);
  return lDistdR + lDistdZ + lDistdRPhi;
}

template <typename DataT>
int SpaceCharge<DataT>::dumpLocalDistCorrVectors(std::string_view file, const Side side, std::string_view option) const
{
  if (!mLocalVecDistdR[side].getNDataPoints()) {
    LOGP(info, "============== local distortion vectors are not set! returning ==============");
    return 0;
  }
  const std::string sideName = getSideName(side);
  const int lVecDistdR = mLocalVecDistdR[side].writeToFile(file, option, fmt::format("lvecdistR_side{}", sideName), sNThreads);
  const int lVecDistdZ = mLocalVecDistdZ[side].writeToFile(file, "UPDATE", fmt::format("lvecdistZ_side{}", sideName), sNThreads);
  const int lVecDistdRPhi = mLocalVecDistdRPhi[side].writeToFile(file, "UPDATE", fmt::format("lvecdistRPhi_side{}", sideName), sNThreads);
  dumpMetaData(file, "UPDATE", false);
  return lVecDistdR + lVecDistdZ + lVecDistdRPhi;
}

template <typename DataT>
void SpaceCharge<DataT>::setLocalDistortionsFromFile(std::string_view file, const Side side)
{
  const std::string sideName = getSideName(side);
  const std::string_view treename{fmt::format("ldistR_side{}", getSideName(side))};
  if (!checkGridFromFile(file, treename)) {
    return;
  }
  initContainer(mLocalDistdR[side], true);
  initContainer(mLocalDistdZ[side], true);
  initContainer(mLocalDistdRPhi[side], true);
  const bool lDistdR = mLocalDistdR[side].initFromFile(file, treename, sNThreads);
  const bool lDistdZ = mLocalDistdZ[side].initFromFile(file, fmt::format("ldistZ_side{}", sideName), sNThreads);
  const bool lDistdRPhi = mLocalDistdRPhi[side].initFromFile(file, fmt::format("ldistRPhi_side{}", sideName), sNThreads);
  readMetaData(file);
}

template <typename DataT>
void SpaceCharge<DataT>::setLocalDistCorrVectorsFromFile(std::string_view file, const Side side)
{
  const std::string sideName = getSideName(side);
  const std::string_view treename{fmt::format("lvecdistR_side{}", getSideName(side))};
  if (!checkGridFromFile(file, treename)) {
    return;
  }
  initContainer(mLocalVecDistdR[side], true);
  initContainer(mLocalVecDistdZ[side], true);
  initContainer(mLocalVecDistdRPhi[side], true);
  const bool lVecDistdR = mLocalVecDistdR[side].initFromFile(file, treename, sNThreads);
  const bool lVecDistdZ = mLocalVecDistdZ[side].initFromFile(file, fmt::format("lvecdistZ_side{}", sideName), sNThreads);
  const bool lVecDistdRPhi = mLocalVecDistdRPhi[side].initFromFile(file, fmt::format("lvecdistRPhi_side{}", sideName), sNThreads);
  readMetaData(file);
}

template <typename DataT>
int SpaceCharge<DataT>::dumpPotential(std::string_view file, const Side side, std::string_view option) const
{
  if (!mPotential[side].getNDataPoints()) {
    LOGP(info, "============== potential not set! returning ==============");
    return 0;
  }
  int status = mPotential[side].writeToFile(file, option, fmt::format("potential_side{}", getSideName(side)), sNThreads);
  dumpMetaData(file, "UPDATE", false);
  return status;
}

template <typename DataT>
void SpaceCharge<DataT>::setPotentialFromFile(std::string_view file, const Side side)
{
  const std::string_view treename{fmt::format("potential_side{}", getSideName(side))};
  if (!checkGridFromFile(file, treename)) {
    return;
  }
  initContainer(mPotential[side], true);
  mPotential[side].initFromFile(file, treename, sNThreads);
  readMetaData(file);
}

template <typename DataT>
int SpaceCharge<DataT>::dumpDensity(std::string_view file, const Side side, std::string_view option) const
{
  if (!mDensity[side].getNDataPoints()) {
    LOGP(info, "============== space charge density are not set! returning ==============");
    return 0;
  }
  int status = mDensity[side].writeToFile(file, option, fmt::format("density_side{}", getSideName(side)), sNThreads);
  dumpMetaData(file, "UPDATE", false);
  return status;
}

template <typename DataT>
bool SpaceCharge<DataT>::checkGridFromFile(std::string_view file, std::string_view tree)
{
  unsigned short nr, nz, nphi;
  if (!DataContainer::getVertices(tree, file, nr, nz, nphi)) {
    return false;
  }

  // check if stored grid definition and current grid definition is the same
  if ((mParamGrid.NRVertices != nr) || (mParamGrid.NZVertices != nz) || (mParamGrid.NPhiVertices != nphi)) {
    LOGP(info, "Different number of vertices found in input file. Initializing new space charge object with nR {} nZ {} nPhi {} vertices", nr, nz, nphi);
    SpaceCharge<DataT> scTmp(mBField.getBField(), nz, nr, nphi, false);
    scTmp.mC0 = mC0;
    scTmp.mC1 = mC1;
    scTmp.mC2 = mC2;
    *this = std::move(scTmp);
  }
  return true;
}

template <typename DataT>
void SpaceCharge<DataT>::setDensityFromFile(std::string_view file, const Side side)
{
  const std::string_view treename{fmt::format("density_side{}", getSideName(side))};
  if (!checkGridFromFile(file, treename)) {
    return;
  }
  initContainer(mDensity[side], true);
  mDensity[side].initFromFile(file, treename, sNThreads);
  readMetaData(file);
}

template <typename DataT>
int SpaceCharge<DataT>::dumpGlobalCorrections(TFile& outf, const Side side) const
{
  if (!mGlobalCorrdR[side].getNDataPoints()) {
    LOGP(info, "============== global corrections are not set! returning ==============");
    return 0;
  }
  const std::string sideName = getSideName(side);
  const int er = mGlobalCorrdR[side].template writeToFile<float>(outf, fmt::format("corrR_side{}", sideName).data());
  const int ez = mGlobalCorrdZ[side].template writeToFile<float>(outf, fmt::format("corrZ_side{}", sideName).data());
  const int ephi = mGlobalCorrdRPhi[side].template writeToFile<float>(outf, fmt::format("corrRPhi_side{}", sideName).data());
  return er + ez + ephi;
}

template <typename DataT>
void SpaceCharge<DataT>::dumpToFile(std::string_view file, const Side side, std::string_view option) const
{
  if (option == "RECREATE") {
    // delete the file
    gSystem->Unlink(file.data());
  }
  dumpElectricFields(file, side, "UPDATE");
  dumpPotential(file, side, "UPDATE");
  dumpDensity(file, side, "UPDATE");
  dumpGlobalDistortions(file, side, "UPDATE");
  dumpGlobalCorrections(file, side, "UPDATE");
  dumpLocalCorrections(file, side, "UPDATE");
  dumpLocalDistortions(file, side, "UPDATE");
  dumpLocalDistCorrVectors(file, side, "UPDATE");
}

template <typename DataT>
void SpaceCharge<DataT>::dumpToFile(std::string_view file) const
{
  dumpToFile(file, Side::A, "RECREATE");
  dumpToFile(file, Side::C, "UPDATE");
}

template <typename DataT>
void SpaceCharge<DataT>::dumpMetaData(std::string_view file, std::string_view option, const bool overwriteExisting) const
{
  TFile f(file.data(), option.data());
  if (!overwriteExisting && f.GetListOfKeys()->Contains("meta")) {
    return;
  }
  f.Close();

  // create meta objects
  std::vector<float> params{static_cast<float>(mC0), static_cast<float>(mC1), static_cast<float>(mC2)};
  auto helperA = mGrid3D[Side::A].getHelper();
  auto helperC = mGrid3D[Side::C].getHelper();

  // define dataframe
  ROOT::RDataFrame dFrame(1);
  auto dfStore = dFrame.DefineSlotEntry("paramsC", [&params = params](unsigned int, ULong64_t entry) { return params; });
  dfStore = dfStore.DefineSlotEntry("grid_A", [&helperA = helperA](unsigned int, ULong64_t entry) { return helperA; });
  dfStore = dfStore.DefineSlotEntry("grid_C", [&helperC = helperC](unsigned int, ULong64_t entry) { return helperC; });
  dfStore = dfStore.DefineSlotEntry("BField", [field = mBField.getBField()](unsigned int, ULong64_t entry) { return field; });
  dfStore = dfStore.DefineSlotEntry("metaInf", [meta = mMeta](unsigned int, ULong64_t entry) { return meta; });

  // write to TTree
  ROOT::RDF::RSnapshotOptions opt;
  opt.fMode = option;
  opt.fOverwriteIfExists = true; // overwrite if already exists
  dfStore.Snapshot("meta", file, {"paramsC", "grid_A", "grid_C", "BField", "metaInf"}, opt);
}

template <typename DataT>
void SpaceCharge<DataT>::readMetaData(std::string_view file)
{
  if (mReadMetaData) {
    return;
  }

  // check if TTree exists
  TFile f(file.data(), "READ");
  if (!f.GetListOfKeys()->Contains("meta")) {
    return;
  }
  f.Close();

  auto readMeta = [&mC0 = mC0, &mC1 = mC1, &mC2 = mC2, &mGrid3D = mGrid3D, &mBField = mBField](const std::vector<float>& paramsC, const RegularGridHelper<double>& gridA, const RegularGridHelper<double>& gridC, int field) {
    mC0 = paramsC[0];
    mC1 = paramsC[1];
    mC2 = paramsC[2];
    mGrid3D[Side::A] = RegularGrid3D<DataT>(gridA.zmin, gridA.rmin, gridA.phimin, gridA.spacingZ, gridA.spacingR, gridA.spacingPhi, gridA.params);
    mGrid3D[Side::C] = RegularGrid3D<DataT>(gridC.zmin, gridC.rmin, gridC.phimin, gridC.spacingZ, gridC.spacingR, gridC.spacingPhi, gridC.params);
    mBField.setBField(field);
  };

  ROOT::RDataFrame dFrame("meta", file);
  dFrame.Foreach(readMeta, {"paramsC", "grid_A", "grid_C", "BField"});

  const auto& cols = dFrame.GetColumnNames();
  if (std::find(cols.begin(), cols.end(), "metaInf") != cols.end()) {
    auto readMetaInf = [&mMeta = mMeta](const SCMetaData& meta) {
      mMeta = meta;
    };
    dFrame.Foreach(readMetaInf, {"metaInf"});
  }

  LOGP(info, "Setting meta data: mC0={}  mC1={}  mC2={}", mC0, mC1, mC2);
  mReadMetaData = true;
}

template <typename DataT>
void SpaceCharge<DataT>::setSimOneSector()
{
  LOGP(warning, "Use this feature only if you know what you are doing!");
  o2::tpc::MGParameters::normalizeGridToOneSector = true;
  RegularGrid gridTmp[FNSIDES]{{GridProp::ZMIN, GridProp::RMIN, GridProp::PHIMIN, getSign(Side::A) * GridProp::getGridSpacingZ(mParamGrid.NZVertices), GridProp::getGridSpacingR(mParamGrid.NRVertices), GridProp::getGridSpacingPhi(mParamGrid.NPhiVertices) / SECTORSPERSIDE, mParamGrid},
                               {GridProp::ZMIN, GridProp::RMIN, GridProp::PHIMIN, getSign(Side::C) * GridProp::getGridSpacingZ(mParamGrid.NZVertices), GridProp::getGridSpacingR(mParamGrid.NRVertices), GridProp::getGridSpacingPhi(mParamGrid.NPhiVertices) / SECTORSPERSIDE, mParamGrid}};
  mGrid3D[0] = gridTmp[0];
  mGrid3D[1] = gridTmp[1];
}

template <typename DataT>
void SpaceCharge<DataT>::unsetSimOneSector()
{
  o2::tpc::MGParameters::normalizeGridToOneSector = false;
}

template <typename DataT>
void SpaceCharge<DataT>::setFromFile(std::string_view file, const Side side)
{
  setDensityFromFile(file, side);
  setPotentialFromFile(file, side);
  setElectricFieldsFromFile(file, side);
  setLocalDistortionsFromFile(file, side);
  setLocalCorrectionsFromFile(file, side);
  setGlobalDistortionsFromFile(file, side);
  setGlobalCorrectionsFromFile(file, side);
  setLocalDistCorrVectorsFromFile(file, side);
}

template <typename DataT>
void SpaceCharge<DataT>::setFromFile(std::string_view file)
{
  setFromFile(file, Side::A);
  setFromFile(file, Side::C);
}

template <typename DataT>
void SpaceCharge<DataT>::initContainer(DataContainer& data, const bool initMem)
{
  if (!data.getNDataPoints()) {
    data.setGrid(getNZVertices(), getNRVertices(), getNPhiVertices(), initMem);
  }
}

template <typename DataT>
void SpaceCharge<DataT>::setOmegaTauT1T2(const DataT omegaTau, const DataT t1, const DataT t2)
{
  const DataT wt0 = t2 * omegaTau;
  const DataT wt02 = wt0 * wt0;
  mC0 = 1 / (1 + wt02);
  const DataT wt1 = t1 * omegaTau;
  mC1 = wt1 / (1 + wt1 * wt1);
  mC2 = wt02 / (1 + wt02);
}

template <typename DataT>
void SpaceCharge<DataT>::addChargeDensity(const SpaceCharge<DataT>& otherSC)
{
  const bool sameGrid = (getNPhiVertices() == otherSC.getNPhiVertices()) && (getNRVertices() == otherSC.getNRVertices()) && (getNZVertices() == otherSC.getNZVertices());
  if (!sameGrid) {
    LOGP(warning, "Space charge objects have different grid definition");
    return;
  }

  mDensity[Side::A] += otherSC.mDensity[Side::A];
  mDensity[Side::C] += otherSC.mDensity[Side::C];
}

template <typename DataT>
void SpaceCharge<DataT>::fillChargeDensityFromHisto(const char* file, const char* nameA, const char* nameC)
{
  TFile fInp(file, "READ");
  TH3F* hSCA = (TH3F*)fInp.Get(nameA);
  TH3F* hSCC = (TH3F*)fInp.Get(nameC);
  if (!hSCA) {
    LOGP(error, "Histogram {} not found", nameA);
  }
  if (!hSCC) {
    LOGP(error, "Histogram {} not found", nameC);
  }
  fillChargeDensityFromHisto(*hSCA, *hSCC);
}

template <typename DataT>
void SpaceCharge<DataT>::fillChargeDensityFromHisto(const TH3& hisSCDensity3D_A, const TH3& hisSCDensity3D_C)
{
  const int nPhiBinsTmp = hisSCDensity3D_A.GetXaxis()->GetNbins();
  const int nRBinsTmp = hisSCDensity3D_A.GetYaxis()->GetNbins();
  const int nZBins = hisSCDensity3D_A.GetZaxis()->GetNbins();
  const auto phiLow = hisSCDensity3D_A.GetXaxis()->GetBinLowEdge(1);
  const auto phiUp = hisSCDensity3D_A.GetXaxis()->GetBinUpEdge(nPhiBinsTmp);
  const auto rLow = hisSCDensity3D_A.GetYaxis()->GetBinLowEdge(1);
  const auto rUp = hisSCDensity3D_A.GetYaxis()->GetBinUpEdge(nRBinsTmp);
  const auto zUp = hisSCDensity3D_A.GetZaxis()->GetBinUpEdge(nZBins);

  TH3F hisSCMerged("hisMerged", "hisMerged", nPhiBinsTmp, phiLow, phiUp, nRBinsTmp, rLow, rUp, 2 * nZBins, -zUp, zUp);

  for (int iside = 0; iside < FNSIDES; ++iside) {
    const auto& hSC = (iside == 0) ? hisSCDensity3D_A : hisSCDensity3D_C;
#pragma omp parallel for num_threads(sNThreads)
    for (int iz = 1; iz <= nZBins; ++iz) {
      const int izTmp = (iside == 0) ? (nZBins + iz) : iz;
      for (int ir = 1; ir <= nRBinsTmp; ++ir) {
        for (int iphi = 1; iphi <= nPhiBinsTmp; ++iphi) {
          hisSCMerged.SetBinContent(iphi, ir, izTmp, hSC.GetBinContent(iphi, ir, iz));
        }
      }
    }
  }
  fillChargeDensityFromHisto(hisSCMerged);
}

template <typename DataT>
void SpaceCharge<DataT>::convertIDCsToCharge(std::vector<CalDet<float>>& idcZero, const CalDet<float>& mapIBF, const float ionDriftTimeMS, const bool normToPadArea)
{
  // 1. integration time per IDC interval in ms
  const int nOrbits = 12;
  const float idcIntegrationTimeMS = (nOrbits * o2::constants::lhc::LHCOrbitMUS) / 1e3;

  // number of time stamps for each integration interval (5346) (see: IDCSim.h getNTimeStampsPerIntegrationInterval())
  const unsigned int nTimeStampsPerIDCInterval{(o2::constants::lhc::LHCMaxBunches * nOrbits) / o2::tpc::constants::LHCBCPERTIMEBIN};

  const int nIDCSlices = idcZero.size();
  // IDCs are normalized for each interval to 5346 time bins
  // IDC0 = <IDC> per ms = 5346 / ~1ms
  const float idcsPerMS = nTimeStampsPerIDCInterval / idcIntegrationTimeMS;

  // length of one z slice on ms
  const float lengthZSliceMS = ionDriftTimeMS / nIDCSlices;

  // IDCs for one z slice
  const float scaleToIonDrift = lengthZSliceMS * idcsPerMS;

  // get conversion factor from ADC to electrons (see: SAMPAProcessing::getADCvalue())
  const static ParameterElectronics& parameterElectronics = ParameterElectronics::Instance();
  const float conversionADCToEle = parameterElectronics.ElectronCharge * 1.e15 * parameterElectronics.ChipGain * parameterElectronics.ADCsaturation / parameterElectronics.ADCdynamicRange;

  const float conversionFactor = scaleToIonDrift / conversionADCToEle;
  LOGP(info, "Converting IDCs to space-charge density with conversion factor of {}", conversionFactor);

  for (auto& calIDC : idcZero) {
    if (normToPadArea) {
      for (unsigned int sector = 0; sector < Mapper::NSECTORS; ++sector) {
        for (unsigned int region = 0; region < Mapper::NREGIONS; ++region) {
          for (int lrow = 0; lrow < Mapper::ROWSPERREGION[region]; ++lrow) {
            for (unsigned int pad = 0; pad < Mapper::PADSPERROW[region][lrow]; ++pad) {
              const int globalPad = Mapper::getGlobalPadNumber(lrow, pad, region);
              float idcTmp = calIDC.getValue(sector, globalPad);
              calIDC.setValue(sector, globalPad, conversionFactor * idcTmp / Mapper::INVPADAREA[region]);
            }
          }
        }
      }
    } else {
      calIDC *= conversionFactor;
    }
    // take IBF into account
    calIDC *= mapIBF;
    calIDC *= 0.01f; // ibf values are in %
  }
}

template <typename DataT>
void SpaceCharge<DataT>::fillChargeFromIDCs(std::vector<CalDet<float>>& idcZero, const CalDet<float>& mapIBF, const float ionDriftTimeMS, const bool normToPadArea)
{
  convertIDCsToCharge(idcZero, mapIBF, ionDriftTimeMS, normToPadArea);
  fillChargeFromCalDet(idcZero);
}

template <typename DataT>
void SpaceCharge<DataT>::initRodAlignmentVoltages(const MisalignmentType misalignmentType, const FCType fcType, const int sector, const Side side, const float deltaPot)
{
  // see also original implementation in AliTPCFCVoltError3D::InitFCVoltError3D (https://github.com/alisw/AliRoot/blob/master/TPC/TPCbase/AliTPCFCVoltError3D.h)

  const int iside = static_cast<int>(side);
  initContainer(mPotential[iside], true);
  const int phiVerticesPerSector = mParamGrid.NPhiVertices / SECTORSPERSIDE;
  int phiVerticesEnd = phiVerticesPerSector;

  if (misalignmentType == MisalignmentType::RodShift) {
    const float rodDiameter = 1; // 1 cm SET SOMEWEHERE ELSE
    const float rodRadius = rodDiameter / 2;
    const float radiusTmp = (fcType == FCType::IFC) ? TPCParameters<DataT>::IFCRADIUS : TPCParameters<DataT>::OFCRADIUS;
    int nPhiVerticesPerRod = static_cast<int>(rodRadius * mParamGrid.NPhiVertices / (TWOPI * radiusTmp) + 0.5);
    if (nPhiVerticesPerRod == 0) {
      nPhiVerticesPerRod = 1;
    }
    phiVerticesEnd = nPhiVerticesPerRod;
  }

  const int phiStart = sector * phiVerticesPerSector;
  const int phiEnd = phiStart + phiVerticesEnd;
  const int nRVertex = (fcType == FCType::IFC) ? 0 : (mParamGrid.NRVertices - 1);
  for (size_t iPhi = phiStart; iPhi < phiEnd; ++iPhi) {
    const int iPhiSector = iPhi % phiVerticesPerSector;

    float potentialSector = 0;
    float potentialLastSector = 0;
    if ((misalignmentType == MisalignmentType::ShiftedClip) || (misalignmentType == MisalignmentType::RodShift)) {
      const float potentialShiftedClips = deltaPot - iPhiSector * deltaPot / phiVerticesEnd;
      potentialSector = potentialShiftedClips;
      potentialLastSector = potentialShiftedClips;
    } else if (misalignmentType == MisalignmentType::RotatedClip) {
      // set to zero for first vertex
      if (iPhiSector == 0) {
        potentialSector = 0;
        potentialLastSector = 0;
      } else {
        const float potentialRotatedClip = -deltaPot + (iPhiSector - 1) * deltaPot / (phiVerticesPerSector - 2);
        potentialSector = potentialRotatedClip;
        potentialLastSector = -potentialRotatedClip;
      }
    }

    for (size_t iZ = 0; iZ < mParamGrid.NZVertices; ++iZ) {
      mPotential[iside](iZ, nRVertex, iPhi) += potentialSector;
      if (iPhiSector > 0) {
        const int iPhiMirror = ((phiStart - iPhiSector) + mParamGrid.NPhiVertices) % mParamGrid.NPhiVertices;
        mPotential[iside](iZ, nRVertex, iPhiMirror) += potentialLastSector;
      }
    }
  }
}

template <typename DataT>
void SpaceCharge<DataT>::addBoundaryPotential(const SpaceCharge<DataT>& other, const Side side, const float scaling)
{
  if (other.mPotential[side].getData().empty()) {
    LOGP(info, "Other space-charge object is empty!");
    return;
  }

  if ((mParamGrid.NRVertices != other.mParamGrid.NRVertices) || (mParamGrid.NZVertices != other.mParamGrid.NZVertices) || (mParamGrid.NPhiVertices != other.mParamGrid.NPhiVertices)) {
    LOGP(info, "Different number of vertices found in input file. Initializing new space charge object with nR {} nZ {} nPhi {} vertices", other.mParamGrid.NRVertices, other.mParamGrid.NZVertices, other.mParamGrid.NPhiVertices);
    SpaceCharge<DataT> scTmp(mBField.getBField(), other.mParamGrid.NZVertices, other.mParamGrid.NRVertices, other.mParamGrid.NPhiVertices, false);
    scTmp.mC0 = mC0;
    scTmp.mC1 = mC1;
    scTmp.mC2 = mC2;
    *this = std::move(scTmp);
  }

  initContainer(mPotential[side], true);

  for (size_t iPhi = 0; iPhi < mParamGrid.NPhiVertices; ++iPhi) {
    for (size_t iZ = 1; iZ < mParamGrid.NZVertices; ++iZ) {
      const size_t iRFirst = 0;
      mPotential[side](iZ, iRFirst, iPhi) += scaling * other.mPotential[side](iZ, iRFirst, iPhi);

      const size_t iRLast = mParamGrid.NRVertices - 1;
      mPotential[side](iZ, iRLast, iPhi) += scaling * other.mPotential[side](iZ, iRLast, iPhi);
    }
  }

  for (size_t iPhi = 0; iPhi < mParamGrid.NPhiVertices; ++iPhi) {
    for (size_t iR = 0; iR < mParamGrid.NRVertices; ++iR) {
      const size_t iZFirst = 0;
      mPotential[side](iZFirst, iR, iPhi) += scaling * other.mPotential[side](iZFirst, iR, iPhi);

      const size_t iZLast = mParamGrid.NZVertices - 1;
      mPotential[side](iZLast, iR, iPhi) += scaling * other.mPotential[side](iZLast, iR, iPhi);
    }
  }
}

template <typename DataT>
void SpaceCharge<DataT>::resetBoundaryPotentialToZeroInRangeZ(float zMin, float zMax, const Side side)
{
  const float zMaxAbs = std::abs(zMax);
  for (size_t iPhi = 0; iPhi < mParamGrid.NPhiVertices; ++iPhi) {
    for (size_t iZ = 1; iZ < mParamGrid.NZVertices; ++iZ) {
      const DataT z = std::abs(getZVertex(iZ, side));
      if ((z < zMin) || (z > zMax)) {
        const size_t iRFirst = 0;
        mPotential[side](iZ, iRFirst, iPhi) = 0;

        const size_t iRLast = mParamGrid.NRVertices - 1;
        mPotential[side](iZ, iRLast, iPhi) = 0;
      }
    }
  }

  for (size_t iPhi = 0; iPhi < mParamGrid.NPhiVertices; ++iPhi) {
    for (size_t iR = 0; iR < mParamGrid.NRVertices; ++iR) {
      const size_t iZFirst = 0;
      const float zFirst = std::abs(getZVertex(iZFirst, side));
      if ((zFirst < zMin) || (zFirst > zMax)) {
        mPotential[side](iZFirst, iR, iPhi) = 0;
      }

      const size_t iZLast = mParamGrid.NZVertices - 1;
      const float zLast = std::abs(getZVertex(iZLast, side));
      if ((zLast < zMin) || (zLast > zMax)) {
        mPotential[side](iZLast, iR, iPhi) = 0;
      }
    }
  }
}

template <typename DataT>
void SpaceCharge<DataT>::setIFCChargeUpRisingPot(const float deltaPot, const float zMaxDeltaPot, const int type, const float zStart, const float offs, const Side side)
{
  std::function<DataT(DataT)> chargeUpIFCLinear = [zStart, type, offs, deltaPot, zMaxDeltaPot](const DataT z) {
    const float absZ = std::abs(z);
    const float absZMaxDeltaPot = std::abs(zMaxDeltaPot);
    if ((absZ <= absZMaxDeltaPot) && (absZ >= zStart)) {
      // 1/x
      if (type == 1) {
        const float offsZ = 1;
        const float zMaxDeltaPotTmp = zMaxDeltaPot - zStart + offsZ;
        const float p1 = deltaPot / (1 / offsZ - 1 / zMaxDeltaPotTmp);
        const float p2 = -p1 / zMaxDeltaPotTmp;
        const float absZShifted = zMaxDeltaPotTmp - (absZ - zStart);
        DataT pot = p2 + p1 / absZShifted;
        return pot;
      } else if (type == 0 || type == 4) {
        // linearly rising potential
        return static_cast<DataT>(deltaPot / (absZMaxDeltaPot - zStart) * (absZ - zStart) + offs);
      } else if (type == 2) {
        // flat
        return DataT(deltaPot);
      } else if (type == 3) {
        // linear falling
        return static_cast<DataT>(-deltaPot / (absZMaxDeltaPot - zStart) * (absZ - zStart) + deltaPot);
      } else {
        return DataT(0);
      }
    } else if (type == 4) {
      // flat no z dependence
      return DataT(offs);
    } else {
      return DataT(0);
    }
  };
  setPotentialBoundaryInnerRadius(chargeUpIFCLinear, side);
}

template <typename DataT>
void SpaceCharge<DataT>::setIFCChargeUpFallingPot(const float deltaPot, const float zMaxDeltaPot, const int type, const float zEnd, const float offs, const Side side)
{
  std::function<DataT(DataT)> chargeUpIFCLinear = [zEnd, type, offs, zMax = getZMax(Side::A), deltaPot, zMaxDeltaPot](const DataT z) {
    const float absZ = std::abs(z);
    const float absZMaxDeltaPot = std::abs(zMaxDeltaPot);

    bool check = (absZ >= absZMaxDeltaPot);
    if (type == 0 || type == 3) {
      check = (absZ >= absZMaxDeltaPot);
    }

    if (check && (absZ <= zEnd)) {
      // 1/x dependency
      if (type == 1) {
        const float p1 = (deltaPot - offs) / (1 / zMaxDeltaPot - 1 / zEnd);
        const float p2 = offs - p1 / zEnd;
        DataT pot = p2 + p1 / absZ;
        return pot;
      } else if (type == 2) {
        // 1/x dependency steep fall off!
        const float offsZ = 1 + offs;
        const float zEndTmp = zEnd - zMaxDeltaPot + offsZ;
        const float p1 = deltaPot / (1 / offsZ - 1 / zEndTmp);
        const float p2 = -p1 / zEndTmp;
        const float absZShifted = absZ - zMaxDeltaPot + offsZ;
        DataT pot = p2 + p1 / absZShifted;
        return pot;
      } else if (type == 0 || type == 3) {
        // linearly falling potential
        const float zPos = absZ - zEnd;
        return static_cast<DataT>(deltaPot / (absZMaxDeltaPot - zEnd) * zPos + offs);
      } else {
        return DataT(0);
      }
    } else if (type == 3) {
      return DataT(offs);
    } else {
      return DataT(0);
    }
  };
  setPotentialBoundaryInnerRadius(chargeUpIFCLinear, side);
}

template <typename DataT>
void SpaceCharge<DataT>::setGlobalCorrections(const std::function<void(int sector, DataT gx, DataT gy, DataT gz, DataT& gCx, DataT& gCy, DataT& gCz)>& gCorr, const Side side)
{
  initContainer(mGlobalCorrdR[side], true);
  initContainer(mGlobalCorrdZ[side], true);
  initContainer(mGlobalCorrdRPhi[side], true);

#pragma omp parallel for num_threads(sNThreads)
  for (unsigned int iPhi = 0; iPhi < mParamGrid.NPhiVertices; ++iPhi) {
    const int sector = iPhi / (mParamGrid.NPhiVertices / SECTORSPERSIDE) + (side == Side::A ? 0 : SECTORSPERSIDE);
    DataT phi = getPhiVertex(iPhi, side);
    phi = o2::math_utils::detail::toPMPi(phi);

    for (unsigned int iR = 0; iR < mParamGrid.NRVertices; ++iR) {
      const DataT radius = getRVertex(iR, side);
      const DataT x = getXFromPolar(radius, phi);
      const DataT y = getYFromPolar(radius, phi);

      for (unsigned int iZ = 0; iZ < mParamGrid.NZVertices; ++iZ) {
        const DataT z = getZVertex(iZ, side);

        // get corrected points
        DataT gCx = 0;
        DataT gCy = 0;
        DataT gCz = 0;
        gCorr(sector, x, y, z, gCx, gCy, gCz);
        const DataT gCxCorr = x + gCx;
        const DataT gCyCorr = y + gCy;

        // get corrections
        const DataT corrR = getRadiusFromCartesian(gCxCorr, gCyCorr) - radius;
        DataT phiDiff = getPhiFromCartesian(gCxCorr, gCyCorr) - phi;
        phiDiff = o2::math_utils::detail::toPMPi(phiDiff);
        const DataT corrRPhi = phiDiff * radius;

        // store corrections
        mGlobalCorrdR[side](iZ, iR, iPhi) = corrR;
        mGlobalCorrdZ[side](iZ, iR, iPhi) = gCz;
        mGlobalCorrdRPhi[side](iZ, iR, iPhi) = corrRPhi;
      }
    }
  }
}

template <typename DataT>
void SpaceCharge<DataT>::setROCMisalignmentShiftZ(const int sector, const int type, const float potential)
{
  setROCMisalignment(type, 2, sector, potential, potential);
}

template <typename DataT>
void SpaceCharge<DataT>::setROCMisalignmentRotationAlongX(const int sector, const int type, const float potentialMin, const float potentialMax)
{
  setROCMisalignment(type, 0, sector, potentialMin, potentialMax);
}

template <typename DataT>
void SpaceCharge<DataT>::setROCMisalignmentRotationAlongY(const int sector, const int type, const float potentialMin, const float potentialMax)
{
  setROCMisalignment(type, 1, sector, potentialMin, potentialMax);
}

template <typename DataT>
void SpaceCharge<DataT>::setROCMisalignment(int stackType, int misalignmentType, int sector, const float potMin, const float potMax)
{
  initContainer(mPotential[Sector(sector).side()], true);
  const auto indPhiTopIROC = getPotentialBoundaryGEMFrameAlongPhiIndices(GEMstack::IROCgem, false, Side::A, false, true);

  const auto rotationPoints = [](const int regStart, const int regEnd, int misalignmentType, const float potMax, const float potMin) {
    if (misalignmentType == 0) {
      const auto& mapper = o2::tpc::Mapper::instance();
      const float radStart = mapper.getPadRegionInfo(regStart).getRadiusFirstRow();
      const auto& padReg = mapper.getPadRegionInfo(regEnd);
      const float radEnd = padReg.getRadiusFirstRow() + padReg.getPadHeight() * padReg.getNumberOfPadRows() - radStart;
      const float rotationPoint = radStart + radEnd / 2;
      const float slope = (potMax - potMin) / radEnd;
      return std::pair<float, float>{rotationPoint, slope};
    } else if (misalignmentType == 1) {
      return std::pair<float, float>{0, (potMax - potMin) / 100};
    } else {
      return std::pair<float, float>{0, (potMax + potMin) / 2};
    }
  };

  if (stackType == 0) {
    const auto deltaPotPar = rotationPoints(0, 3, misalignmentType, potMax, potMin);
    const auto indPhiBottomIROC = getPotentialBoundaryGEMFrameAlongPhiIndices(GEMstack::IROCgem, true, Side::A, false, true);
    fillROCMisalignment(indPhiTopIROC, indPhiBottomIROC, sector, misalignmentType, deltaPotPar);
  } else if (stackType == 1) {
    const auto deltaPotPar = rotationPoints(4, 9, misalignmentType, potMax, potMin);
    const auto indPhiTopOROC3 = getPotentialBoundaryGEMFrameAlongPhiIndices(GEMstack::OROC3gem, false, Side::A, false, true);
    fillROCMisalignment(indPhiTopOROC3, indPhiTopIROC, sector, misalignmentType, deltaPotPar);
  } else if (stackType == 2) {
    const auto deltaPotPar = rotationPoints(0, 9, misalignmentType, potMax, potMin);
    const auto indPhiBottomIROC = getPotentialBoundaryGEMFrameAlongPhiIndices(GEMstack::IROCgem, true, Side::A, false, true);
    const auto indPhiTopOROC3 = getPotentialBoundaryGEMFrameAlongPhiIndices(GEMstack::OROC3gem, false, Side::A, false, true);
    fillROCMisalignment(indPhiTopIROC, indPhiBottomIROC, sector, misalignmentType, deltaPotPar);
    fillROCMisalignment(indPhiTopOROC3, indPhiTopIROC, sector, misalignmentType, deltaPotPar);
  }
}

template <typename DataT>
void SpaceCharge<DataT>::fillROCMisalignment(const std::vector<size_t>& indicesTop, const std::vector<size_t>& indicesBottom, int sector, int misalignmentType, const std::pair<float, float>& deltaPotPar)
{
  for (const auto& index : indicesTop) {
    const int iZ = DataContainer3D<float>::getIndexZ(index, getNZVertices(), getNRVertices(), getNPhiVertices());
    const int iRStart = DataContainer3D<float>::getIndexR(index, getNZVertices(), getNRVertices(), getNPhiVertices());
    const int iPhi = DataContainer3D<float>::getIndexPhi(index, getNZVertices(), getNRVertices(), getNPhiVertices());

    const int sectorTmp = iPhi / (getNPhiVertices() / SECTORSPERSIDE) + ((sector >= SECTORSPERSIDE) ? SECTORSPERSIDE : 0);
    if ((sector != -1) && (sectorTmp != sector)) {
      continue;
    }
    const Sector sec(sectorTmp);

    for (size_t iR = iRStart; iR > 0; --iR) {
      const size_t currInd = (iZ + getNZVertices() * (iR + iPhi * getNRVertices()));
      const bool foundVertexBottom = std::binary_search(indicesBottom.begin(), indicesBottom.end(), currInd);
      if (foundVertexBottom) {
        break;
      }

      // get local coordinates
      const float rPos = getRVertex(iR, sec.side());
      const float phiPos = getPhiVertex(iPhi, sec.side());
      const float zPos = getZVertex(iZ, sec.side());
      const float x = getXFromPolar(rPos, phiPos);
      const float y = getYFromPolar(rPos, phiPos);
      const LocalPosition3D pos(x, y, zPos);
      const LocalPosition3D lPos = Mapper::GlobalToLocal(pos, sec);
      float deltaPot = 0;
      if (misalignmentType == 0) {
        deltaPot = (lPos.X() - deltaPotPar.first) * deltaPotPar.second;
      } else if (misalignmentType == 1) {
        deltaPot = lPos.Y() * deltaPotPar.second;
      } else {
        deltaPot = deltaPotPar.second;
      }
      mPotential[sec.side()](iZ, iR, iPhi) += deltaPot;
    }
  }
}

template <typename DataT>
void SpaceCharge<DataT>::subtractGlobalCorrections(const SpaceCharge<DataT>& otherSC, const Side side)
{
  mGlobalCorrdR[side] -= otherSC.mGlobalCorrdR[side];
  mGlobalCorrdZ[side] -= otherSC.mGlobalCorrdZ[side];
  mGlobalCorrdRPhi[side] -= otherSC.mGlobalCorrdRPhi[side];
}

template <typename DataT>
void SpaceCharge<DataT>::subtractGlobalDistortions(const SpaceCharge<DataT>& otherSC, const Side side)
{
  mGlobalDistdR[side] -= otherSC.mGlobalDistdR[side];
  mGlobalDistdZ[side] -= otherSC.mGlobalDistdZ[side];
  mGlobalDistdRPhi[side] -= otherSC.mGlobalDistdRPhi[side];
}

template <typename DataT>
void SpaceCharge<DataT>::scaleCorrections(const float val, const Side side)
{
  mGlobalCorrdR[side] *= val;
  mGlobalCorrdZ[side] *= val;
  mGlobalCorrdRPhi[side] *= val;
}

template <typename DataT>
void SpaceCharge<DataT>::averageDensityPerSector(const Side side)
{
  initContainer(mDensity[side], true);
  const int verticesPerSector = mParamGrid.NPhiVertices / SECTORSPERSIDE;
  for (unsigned int iR = 0; iR < mParamGrid.NRVertices; ++iR) {
    for (unsigned int iZ = 0; iZ < mParamGrid.NZVertices; ++iZ) {
      for (unsigned int iPhi = 0; iPhi <= (verticesPerSector / 2); ++iPhi) {
        float meanDensity = 0;
        for (int iter = 0; iter < 2; ++iter) {
          for (unsigned int sec = 0; sec < SECTORSPERSIDE; ++sec) {
            const int iPhiTmpA = iPhi + sec * verticesPerSector;
            const int iPhiTmpB = ((sec + 1) * verticesPerSector - iPhi) % mParamGrid.NPhiVertices;
            if (iter == 0) {
              meanDensity += mDensity[side](iZ, iR, iPhiTmpA);
              meanDensity += mDensity[side](iZ, iR, iPhiTmpB);
            } else {
              const float densMean = meanDensity / (2 * SECTORSPERSIDE);
              mDensity[side](iZ, iR, iPhiTmpA) = densMean;
              mDensity[side](iZ, iR, iPhiTmpB) = densMean;
            }
          }
        }
      }
    }
  }
}

template <typename DataT>
void SpaceCharge<DataT>::scaleChargeDensitySector(const float scalingFactor, const Sector sector)
{
  const Side side = sector.side();
  initContainer(mDensity[side], true);
  const int verticesPerSector = mParamGrid.NPhiVertices / SECTORSPERSIDE;
  const int sectorInSide = sector % SECTORSPERSIDE;
  const int iPhiFirst = sectorInSide * verticesPerSector;
  const int iPhiLast = iPhiFirst + verticesPerSector;
  for (unsigned int iR = 0; iR < mParamGrid.NRVertices; ++iR) {
    for (unsigned int iZ = 0; iZ < mParamGrid.NZVertices; ++iZ) {
      for (unsigned int iPhi = iPhiFirst; iPhi < iPhiLast; ++iPhi) {
        mDensity[side](iZ, iR, iPhi) *= scalingFactor;
      }
    }
  }
}

template <typename DataT>
void SpaceCharge<DataT>::scaleChargeDensityStack(const float scalingFactor, const Sector sector, const GEMstack stack)
{
  const Side side = sector.side();
  initContainer(mDensity[side], true);
  const int verticesPerSector = mParamGrid.NPhiVertices / SECTORSPERSIDE;
  const int sectorInSide = sector % SECTORSPERSIDE;
  const int iPhiFirst = sectorInSide * verticesPerSector;
  const int iPhiLast = iPhiFirst + verticesPerSector;
  for (unsigned int iR = 0; iR < mParamGrid.NRVertices; ++iR) {
    const DataT radius = getRVertex(iR, side);
    for (unsigned int iPhi = iPhiFirst; iPhi < iPhiLast; ++iPhi) {
      const DataT phi = getPhiVertex(iR, side);
      const GlobalPosition3D pos(getXFromPolar(radius, phi), getYFromPolar(radius, phi), ((side == Side::A) ? 10 : -10));
      const auto& mapper = o2::tpc::Mapper::instance();
      const o2::tpc::DigitPos digiPadPos = mapper.findDigitPosFromGlobalPosition(pos);
      if (digiPadPos.isValid() && digiPadPos.getCRU().gemStack() == stack) {
        for (unsigned int iZ = 0; iZ < mParamGrid.NZVertices; ++iZ) {
          mDensity[side](iZ, iR, iPhi) *= scalingFactor;
        }
      }
    }
  }
}

template <typename DataT>
void SpaceCharge<DataT>::initAfterReadingFromFile()
{
  mGrid3D[Side::A] = RegularGrid(GridProp::ZMIN, GridProp::RMIN, GridProp::PHIMIN, getSign(Side::A) * GridProp::getGridSpacingZ(mParamGrid.NZVertices), GridProp::getGridSpacingR(mParamGrid.NRVertices), GridProp::getGridSpacingPhi(mParamGrid.NPhiVertices), mParamGrid);
  mGrid3D[Side::C] = RegularGrid(GridProp::ZMIN, GridProp::RMIN, GridProp::PHIMIN, getSign(Side::C) * GridProp::getGridSpacingZ(mParamGrid.NZVertices), GridProp::getGridSpacingR(mParamGrid.NRVertices), GridProp::getGridSpacingPhi(mParamGrid.NPhiVertices), mParamGrid);
}

template <typename DataT>
float SpaceCharge<DataT>::getDCAr(float tgl, const int nPoints, const float phi, o2::utils::TreeStreamRedirector* pcstream) const
{
  const float rmin = getRMin(o2::tpc::Side::A);
  std::vector<float> dRphi;
  std::vector<float> r;
  dRphi.reserve(nPoints);
  r.reserve(nPoints);
  for (int i = 0; i < nPoints; ++i) {
    float radius = rmin + i;
    float z = tgl * radius;
    DataT distZ = 0;
    DataT distR = 0;
    DataT distRPhi = 0;
    getDistortionsCyl(z, radius, phi, o2::tpc::Side::A, distZ, distR, distRPhi);
    dRphi.emplace_back(distRPhi);
    r.emplace_back(radius);
  }

  TF1 fPol("pol2", "pol2", rmin, r.back());
  fPol.SetParameter(0, 0);
  fPol.SetParameter(1, 0);
  fPol.SetParameter(2, 0);
  TGraph gr(r.size(), r.data(), dRphi.data());
  gr.Fit(&fPol, "QNRC");
  float dca = fPol.Eval(0);
  if (pcstream) {
    std::vector<double> params{fPol.GetParameter(0), fPol.GetParameter(1), fPol.GetParameter(2)};
    std::vector<float> rInterpol;
    std::vector<float> dRPhiInterpol;
    std::vector<float> distanceInterpol;

    for (int i = 0; i < 500; ++i) {
      float radius = rmin + float(i) / 10;
      rInterpol.emplace_back(radius);
      dRPhiInterpol.emplace_back(fPol.Eval(radius));
      distanceInterpol.emplace_back(std::sqrt(rInterpol.back() * rInterpol.back() + dRPhiInterpol.back() * dRPhiInterpol.back()));
    }

    for (int i = -200; i < 200; ++i) {
      float radius = float(i) / 10;
      rInterpol.emplace_back(radius);
      dRPhiInterpol.emplace_back(fPol.Eval(radius));
      distanceInterpol.emplace_back(std::sqrt(rInterpol.back() * rInterpol.back() + dRPhiInterpol.back() * dRPhiInterpol.back()));
    }
    (*pcstream) << "tree"
                << "r=" << r
                << "dRphi=" << dRphi
                << "tgl=" << tgl
                << "dca=" << dca
                << "rInterpol=" << rInterpol
                << "dRPhiInterpol=" << dRPhiInterpol
                << "distanceInterpol=" << distanceInterpol
                << "param=" << params
                << "\n";
  }
  return dca;
}

template <typename DataT>
void SpaceCharge<DataT>::setPotential(int iz, int ir, int iphi, Side side, float val)
{
  initContainer(mPotential[side], true);
  mPotential[side](iz, ir, iphi) = val;
}

using DataTD = double;
template class o2::tpc::SpaceCharge<DataTD>;

using NumFieldsD = NumericalFields<DataTD>;
using AnaFieldsD = AnalyticalFields<DataTD>;
using DistCorrInterpD = DistCorrInterpolator<DataTD>;
using O2TPCSpaceCharge3DCalcD = SpaceCharge<DataTD>;

template void O2TPCSpaceCharge3DCalcD::calcLocalDistortionsCorrections(const O2TPCSpaceCharge3DCalcD::Type, const NumFieldsD&);
template void O2TPCSpaceCharge3DCalcD::calcLocalDistortionsCorrections(const O2TPCSpaceCharge3DCalcD::Type, const AnaFieldsD&);
template void O2TPCSpaceCharge3DCalcD::calcLocalDistortionCorrectionVector(const NumFieldsD&);
template void O2TPCSpaceCharge3DCalcD::calcLocalDistortionCorrectionVector(const AnaFieldsD&);
template void O2TPCSpaceCharge3DCalcD::calcGlobalCorrections(const NumFieldsD&, const int);
template void O2TPCSpaceCharge3DCalcD::calcGlobalCorrections(const AnaFieldsD&, const int);
template void O2TPCSpaceCharge3DCalcD::calcGlobalCorrections(const DistCorrInterpD&, const int);
template void O2TPCSpaceCharge3DCalcD::calcGlobalDistortions(const NumFieldsD&, const int maxIterations);
template void O2TPCSpaceCharge3DCalcD::calcGlobalDistortions(const AnaFieldsD&, const int maxIterations);
template void O2TPCSpaceCharge3DCalcD::calcGlobalDistortions(const DistCorrInterpD&, const int maxIterations);
template void O2TPCSpaceCharge3DCalcD::setGlobalCorrectionsFromFile<double>(TFile&, const Side);
template void O2TPCSpaceCharge3DCalcD::setGlobalCorrectionsFromFile<float>(TFile&, const Side);
template void O2TPCSpaceCharge3DCalcD::setGlobalDistortionsFromFile<double>(TFile&, const Side);
template void O2TPCSpaceCharge3DCalcD::setGlobalDistortionsFromFile<float>(TFile&, const Side);

using DataTF = float;
template class o2::tpc::SpaceCharge<DataTF>;

using NumFieldsF = NumericalFields<DataTF>;
using AnaFieldsF = AnalyticalFields<DataTF>;
using DistCorrInterpF = DistCorrInterpolator<DataTF>;
using O2TPCSpaceCharge3DCalcF = SpaceCharge<DataTF>;

template void O2TPCSpaceCharge3DCalcF::calcLocalDistortionsCorrections(const O2TPCSpaceCharge3DCalcF::Type, const NumFieldsF&);
template void O2TPCSpaceCharge3DCalcF::calcLocalDistortionsCorrections(const O2TPCSpaceCharge3DCalcF::Type, const AnaFieldsF&);
template void O2TPCSpaceCharge3DCalcF::calcLocalDistortionCorrectionVector(const NumFieldsF&);
template void O2TPCSpaceCharge3DCalcF::calcLocalDistortionCorrectionVector(const AnaFieldsF&);
template void O2TPCSpaceCharge3DCalcF::calcGlobalCorrections(const NumFieldsF&, const int);
template void O2TPCSpaceCharge3DCalcF::calcGlobalCorrections(const AnaFieldsF&, const int);
template void O2TPCSpaceCharge3DCalcF::calcGlobalCorrections(const DistCorrInterpF&, const int);
template void O2TPCSpaceCharge3DCalcF::calcGlobalDistortions(const NumFieldsF&, const int maxIterations);
template void O2TPCSpaceCharge3DCalcF::calcGlobalDistortions(const AnaFieldsF&, const int maxIterations);
template void O2TPCSpaceCharge3DCalcF::calcGlobalDistortions(const DistCorrInterpF&, const int maxIterations);
template void O2TPCSpaceCharge3DCalcF::setGlobalCorrectionsFromFile<double>(TFile&, const Side);
template void O2TPCSpaceCharge3DCalcF::setGlobalCorrectionsFromFile<float>(TFile&, const Side);
template void O2TPCSpaceCharge3DCalcF::setGlobalDistortionsFromFile<double>(TFile&, const Side);
template void O2TPCSpaceCharge3DCalcF::setGlobalDistortionsFromFile<float>(TFile&, const Side);
