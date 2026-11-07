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

/// \file PoissonSolver.cxx
/// \brief This class provides implementation of Poisson Eq
/// solver by MultiGrid Method
///
///
///
/// \author  Matthias Kleiner <mkleiner@ikf.uni-frankfurt.de>
/// \date Aug 21, 2020

#include "TPCSpaceCharge/PoissonSolver.h"
#include "TPCSpaceCharge/PoissonSolverHelpers.h"
#include "Framework/Logger.h"
#include <numeric>
#include <cmath>
#include <algorithm>
#include <chrono>
#include <fmt/core.h>
#include <fmt/format.h>
#include <fmt/ranges.h>
#include "TPCSpaceCharge/Vector3D.h"
#include "TPCSpaceCharge/DataContainer3D.h"
#include "DataFormatsTPC/Defs.h"

#ifdef WITH_OPENMP
#include <omp.h>
#endif

using namespace o2::tpc;

template <typename DataT>
void PoissonSolver<DataT>::poissonSolver3D(DataContainer& matricesV, const DataContainer& matricesCharge)
{
  using timer = std::chrono::high_resolution_clock;
  auto start = timer::now();
  poissonMultiGrid3D(matricesV, matricesCharge);
  auto stop = timer::now();
  std::chrono::duration<float> time = stop - start;
  const float totalTime = time.count();
  LOGP(detail, "poissonSolver3D took {}s", totalTime);
}

namespace
{
/// set (Add = false) or add (Add = true) the value to the target
template <bool Add, typename DataT>
inline void setOrAdd(DataT& target, const DataT value)
{
  if constexpr (Add) {
    target += value;
  } else {
    target = value;
  }
}

/// copy src to dst (same size) in parallel
template <typename DataT>
void copyParallel(o2::tpc::Vector3D<DataT>& dst, const o2::tpc::Vector3D<DataT>& src, const int nThreads, const long minPointsParallel)
{
  if ((dst.size() != src.size()) || (dst.getNr() != src.getNr()) || (dst.getNz() != src.getNz())) {
    dst = src;
    return;
  }
  const size_t n = src.size();
  const int nChunks = (nThreads > 1 && static_cast<long>(n) > minPointsParallel) ? nThreads : 1;
#pragma omp parallel for num_threads(nThreads) if (nChunks > 1)
  for (int chunk = 0; chunk < nChunks; ++chunk) {
    const size_t begin = n * chunk / nChunks;
    const size_t end = n * (chunk + 1) / nChunks;
    std::copy(src.begin() + begin, src.begin() + end, dst.begin() + begin);
  }
}
} // namespace

template <typename DataT>
void PoissonSolver<DataT>::poissonMultiGrid3D(DataContainer& matricesV, const DataContainer& matricesCharge)
{
  const DataT gridSpacingR = getSpacingR();
  const DataT gridSpacingZ = getSpacingZ();
  const DataT ratioZ = gridSpacingR * gridSpacingR / (gridSpacingZ * gridSpacingZ); // ratio_{Z} = gridSize_{r} / gridSize_{z}

  LOGP(detail, "{}", fmt::format("PoissonMultiGrid3D: in Poisson Solver 3D multi grid full coarsening  mParamGrid.NRVertices={}, cols={}, mParamGrid.NPhiVertices={}", mParamGrid.NRVertices, mParamGrid.NZVertices, mParamGrid.NPhiVertices));

  // Check that the number of mParamGrid.NRVertices and mParamGrid.NZVertices is suitable for a binary expansion
  if (!isPowerOfTwo((mParamGrid.NRVertices - 1))) {
    LOGP(error, "PoissonMultiGrid3D: Poisson3DMultiGrid - Error in the number of mParamGrid.NRVertices. Must be 2**M + 1");
    return;
  }
  if (!isPowerOfTwo((mParamGrid.NZVertices - 1))) {
    LOGP(error, "PoissonMultiGrid3D: Poisson3DMultiGrid - Error in the number of mParamGrid.NZVertices. Must be 2**N + 1");
    return;
  }
  if (mParamGrid.NPhiVertices <= 3) {
    LOGP(error, "PoissonMultiGrid3D: Poisson3DMultiGrid - Error in the number of mParamGrid.NPhiVertices. Must be larger than 3");
    return;
  }

  // number of levels for the coarsening in r and z
  int nGridRow = 0;
  int nnRow = mParamGrid.NRVertices;
  while (nnRow >>= 1) {
    ++nGridRow;
  }
  int nGridCol = 0;
  int nnCol = mParamGrid.NZVertices;
  while (nnCol >>= 1) {
    ++nGridCol;
  }
  LOGP(detail, "{}", fmt::format("PoissonMultiGrid3D: nGridRow={}, nGridCol={}", nGridRow, nGridCol));

  // grids of the multi grid levels
  const std::vector<MGLevel> levels = getMultiGridLevels(std::max(nGridRow, nGridCol));
  const int nLoop = levels.size();

  // 1) Memory allocation for multi grid
  std::vector<Vector> tvArrayV(nLoop);     // potential <--> error
  std::vector<Vector> tvChargeFMG(nLoop);  // charge is restricted in full multiGrid
  std::vector<Vector> tvCharge(nLoop);     // charge <--> residue
  std::vector<Vector> tvPrevArrayV(nLoop); // error calculation
  std::vector<Vector> tvResidue(nLoop);    // residue calculation

  std::vector<DataT> coefficient1(mParamGrid.NRVertices);        // coefficient1(mParamGrid.NRVertices) for storing (1 + h_{r}/2r_{i}) from central differences in r direction
  std::vector<DataT> coefficient2(mParamGrid.NRVertices);        // coefficient2(mParamGrid.NRVertices) for storing (1 + h_{r}/2r_{i}) from central differences in r direction
  std::vector<DataT> coefficient3(mParamGrid.NRVertices);        // coefficient3(mParamGrid.NRVertices) for storing (1/r_{i}^2) from central differences in phi direction
  std::vector<DataT> coefficient4(mParamGrid.NRVertices);        // coefficient4(mParamGrid.NRVertices) for storing  1/2
  std::vector<DataT> inverseCoefficient4(mParamGrid.NRVertices); // inverse of coefficient4(mParamGrid.NRVertices)

  // the buffers are allocated concurrently: for large grids the first access to the memory (page faults) by the single thread zeroing a buffer dominates
  std::vector<std::pair<Vector*, const MGLevel*>> buffers;
  for (int index = 0; index < nLoop; ++index) {
    for (Vector* buffer : {&tvResidue[index], &tvPrevArrayV[index], &tvChargeFMG[index], &tvArrayV[index], &tvCharge[index]}) {
      buffers.emplace_back(buffer, &levels[index]);
    }
  }
#pragma omp parallel for num_threads(sNThreads) schedule(dynamic, 1)
  for (size_t i = 0; i < buffers.size(); ++i) {
    const MGLevel& level = *buffers[i].second;
    buffers[i].first->resize(level.nr, level.nz, level.nphi);
  }

  // memory for the finest grid is from parameters
#pragma omp parallel for num_threads(sNThreads)
  for (int iphi = 0; iphi < mParamGrid.NPhiVertices; ++iphi) {
    for (int ir = 0; ir < mParamGrid.NRVertices; ++ir) {
      for (int iz = 0; iz < mParamGrid.NZVertices; ++iz) {
        tvChargeFMG[0](ir, iz, iphi) = matricesCharge(iz, ir, iphi);
        tvArrayV[0](ir, iz, iphi) = matricesV(iz, ir, iphi);
      }
    }
  }
  copyParallel(tvCharge[0], tvChargeFMG[0], sNThreads, MINPOINTSPARALLEL);

  // full multi grid (FMG)
  // 1) Restrict the charge and the boundary of the potential to the coarser grids
  for (int index = 1; index < nLoop; ++index) {
    LOGP(detail, "{}", fmt::format("PoissonMultiGrid3D: restriction to tnRRow={}, tnZColumn={}, newPhiSlice={}, oldPhiSlice={}", levels[index].nr, levels[index].nz, levels[index].nphi, levels[index - 1].nphi));
    restrictLevel(tvChargeFMG[index], tvChargeFMG[index - 1], levels[index], levels[index - 1]);
    restrictBoundaryLevel(tvArrayV[index], tvArrayV[index - 1], levels[index], levels[index - 1]);
  }

  // 2) Relax on the coarsest grid
  {
    const MGLevel& level = levels[nLoop - 1];
    const DataT h = gridSpacingR * level.rzFactor;
    const DataT h2 = h * h;
    const DataT gridSizePhiInv = level.nphi * getGridSizePhiInv();   // h_{phi}
    const DataT tempRatioPhi = h2 * gridSizePhiInv * gridSizePhiInv; // ratio_{phi} = gridSize_{r} / gridSize_{phi}
    calcCoefficients(1, level.nr - 1, h, ratioZ, tempRatioPhi, coefficient1, coefficient2, coefficient3, coefficient4);
    relaxLevel(level, tvArrayV[nLoop - 1], tvChargeFMG[nLoop - 1], h2, ratioZ, coefficient1, coefficient2, coefficient3, coefficient4);
  }

  // 3) V Cycle from coarsest to finest
  for (int count = nLoop - 2; count >= 0; --count) {
    // move to finer grid
    std::fill(std::begin(coefficient1), std::end(coefficient1), 0);
    std::fill(std::begin(coefficient2), std::end(coefficient2), 0);
    std::fill(std::begin(coefficient3), std::end(coefficient3), 0);
    std::fill(std::begin(coefficient4), std::end(coefficient4), 0);
    std::fill(std::begin(inverseCoefficient4), std::end(inverseCoefficient4), 0);

    // a) interpolate from 2h --> h grid
    interpolateLevel<false>(tvArrayV[count], tvArrayV[count + 1], levels[count], levels[count + 1]);

    // Copy the relax charge to the tvCharge
    if (count > 0) {
      copyParallel(tvCharge[count], tvChargeFMG[count], sNThreads, MINPOINTSPARALLEL);
    }

    using timer = std::chrono::high_resolution_clock;
    auto start = timer::now();
    for (int mgCycle = 0; mgCycle < MGParameters::nMGCycle; ++mgCycle) {
      // copy to store previous potential
      copyParallel(tvPrevArrayV[count], tvArrayV[count], sNThreads, MINPOINTSPARALLEL);

      vCycle3D(count + 1, nLoop, MGParameters::nPre, MGParameters::nPost, ratioZ, levels, tvArrayV, tvCharge, tvResidue, coefficient1, coefficient2, coefficient3, coefficient4, inverseCoefficient4);

      // converge error
      const DataT convergenceError = getConvergenceError(tvArrayV[count], tvPrevArrayV[count]);
      // if already converge just break move to finer grid
      if (convergenceError <= sConvergenceError) {
        LOGP(detail, "Cycle converged after {} V cycles on level {}. Continue to next cycle...", mgCycle + 1, count);
        break;
      }
      if (count <= 1 && !(mgCycle % 10)) {
        auto stop = timer::now();
        std::chrono::duration<float> time = stop - start;
        const float totalTime = time.count();
        const float timePerCycle = totalTime / (mgCycle + 1);
        const float remaining = timePerCycle * (MGParameters::nMGCycle - mgCycle);
        LOGP(detail, "Cycle {} out of {} for current V cycle {}. Processed time {}s with {}s per cycle. Max remaining time for current cycle {}s. Convergence {} > {}", mgCycle, MGParameters::nMGCycle, count, time.count(), timePerCycle, remaining, convergenceError, sConvergenceError);
      }
      if (mgCycle == (MGParameters::nMGCycle - 1)) {
        LOGP(warning, "Cycle {} on level {} did not convergence! Current convergence error is larger than expected convergence error: {} > {}", mgCycle, count, convergenceError, sConvergenceError);
      }
    }
  }

  // fill output
#pragma omp parallel for num_threads(sNThreads)
  for (int iphi = 0; iphi < mParamGrid.NPhiVertices; ++iphi) {
    for (int ir = 0; ir < mParamGrid.NRVertices; ++ir) {
      for (int iz = 0; iz < mParamGrid.NZVertices; ++iz) {
        matricesV(iz, ir, iphi) = tvArrayV[0](ir, iz, iphi);
      }
    }
  }
}

template <typename DataT>
void PoissonSolver<DataT>::vCycle3D(const int gridFrom, const int gridTo, const int nPre, const int nPost, const DataT ratioZ, const std::vector<MGLevel>& levels, std::vector<Vector>& tvArrayV,
                                    std::vector<Vector>& tvCharge, std::vector<Vector>& tvResidue, std::vector<DataT>& coefficient1, std::vector<DataT>& coefficient2, std::vector<DataT>& coefficient3,
                                    std::vector<DataT>& coefficient4, std::vector<DataT>& inverseCoefficient4) const
{
  const DataT gridSpacingR = getSpacingR();

  // set the coefficients for given level
  const auto setCoefficients = [&](const MGLevel& level, DataT& h2) {
    const DataT h = gridSpacingR * level.rzFactor;
    h2 = h * h;
    const DataT tempGridSizePhiInv = level.nphi * getGridSizePhiInv();
    const DataT tempRatioPhi = h2 * tempGridSizePhiInv * tempGridSizePhiInv; // ratio_{phi} = gridSize_{r} / gridSize_{phi}
    calcCoefficients(1, level.nr - 1, h, ratioZ, tempRatioPhi, coefficient1, coefficient2, coefficient3, coefficient4);
  };

  for (int count = gridFrom; count <= gridTo - 1; ++count) {
    const int index = count - 1;
    const MGLevel& level = levels[index];
    DataT h2 = 0;
    setCoefficients(level, h2);
    const DataT ih2 = 1 / h2;
    for (int i = 1; i < level.nr - 1; ++i) {
      inverseCoefficient4[i] = 1 / coefficient4[i];
    }

    // 1) Pre-Smoothing: Gauss-Seidel Relaxation
    for (int jPre = 1; jPre <= nPre; ++jPre) {
      relaxLevel(level, tvArrayV[index], tvCharge[index], h2, ratioZ, coefficient1, coefficient2, coefficient3, coefficient4);
    } // end pre smoothing

    // 2) Residue calculation
    residue3D(tvResidue[index], tvArrayV[index], tvCharge[index], level.nr, level.nz, level.nphi, ih2, ratioZ, coefficient1, coefficient2, coefficient3, inverseCoefficient4);

    // 3) Restriction
    restrictLevel(tvCharge[count], tvResidue[index], levels[count], level);

    // 4) Zeroing coarser V
    std::fill(tvArrayV[count].begin(), tvArrayV[count].end(), 0);
  }

  // coarsest grid
  {
    const MGLevel& level = levels[gridTo - 1];
    DataT h2 = 0;
    setCoefficients(level, h2);
    relaxLevel(level, tvArrayV[gridTo - 1], tvCharge[gridTo - 1], h2, ratioZ, coefficient1, coefficient2, coefficient3, coefficient4);
  }

  // back to fine
  for (int count = gridTo - 1; count >= gridFrom; --count) {
    const int index = count - 1;
    const MGLevel& level = levels[index];

    // 4) Interpolation/Prolongation
    interpolateLevel<true>(tvArrayV[index], tvArrayV[count], level, levels[count]);

    DataT h2 = 0;
    setCoefficients(level, h2);

    // 5) Post-Smoothing: Gauss-Seidel Relaxation
    for (int jPost = 1; jPost <= nPost; ++jPost) {
      relaxLevel(level, tvArrayV[index], tvCharge[index], h2, ratioZ, coefficient1, coefficient2, coefficient3, coefficient4);
    }
  }
}

template <typename DataT>
void PoissonSolver<DataT>::restrictLevel(Vector& coarse, const Vector& fine, const MGLevel& levelCoarse, const MGLevel& levelFine) const
{
  restrict3D(coarse, fine, levelCoarse.nr, levelCoarse.nz, levelCoarse.nphi, levelFine.nphi);
}

template <typename DataT>
void PoissonSolver<DataT>::restrictBoundaryLevel(Vector& coarse, const Vector& fine, const MGLevel& levelCoarse, const MGLevel& levelFine) const
{
  restrictBoundary3D(coarse, fine, levelCoarse.nr, levelCoarse.nz, levelCoarse.nphi, levelFine.nphi);
}

template <typename DataT>
template <bool Add>
void PoissonSolver<DataT>::interpolateLevel(Vector& fine, const Vector& coarse, const MGLevel& levelFine, const MGLevel& levelCoarse) const
{
  interpolate3D<Add>(fine, coarse, levelFine.nr, levelFine.nz, levelFine.nphi, levelCoarse.nphi);
}

template <typename DataT>
void PoissonSolver<DataT>::residue3D(Vector& residue, const Vector& matricesCurrentV, const Vector& matricesCurrentCharge, const int tnRRow, const int tnZColumn, const int tnPhi,
                                     const DataT ih2, const DataT tempRatioZ, const std::vector<DataT>& coefficient1, const std::vector<DataT>& coefficient2, const std::vector<DataT>& coefficient3, const std::vector<DataT>& inverseCoefficient4) const
{
  // strides of the storage: index = i + nr * (j + nz * m)
  const int strideZ = matricesCurrentV.getNr();
  const int stridePhi = strideZ * matricesCurrentV.getNz();
  const DataT* pot = matricesCurrentV.data().data();
  const DataT* charge = matricesCurrentCharge.data().data();
  DataT* res = residue.data().data();
  const bool runParallel = (sNThreads > 1) && (static_cast<long>(tnRRow) * tnZColumn * tnPhi > MINPOINTSPARALLEL);

#pragma omp parallel for num_threads(sNThreads) if (runParallel)
  for (int m = 0; m < tnPhi; ++m) {
    int mp1, mm1;
    getPhiNeighbours(m, tnPhi, mp1, mm1);
    const DataT* potM = pot + m * stridePhi;
    const DataT* potP = pot + mp1 * stridePhi;
    const DataT* potMm = pot + mm1 * stridePhi;
    const DataT* chargeM = charge + m * stridePhi;
    DataT* resM = res + m * stridePhi;

    for (int j = 1; j < tnZColumn - 1; ++j) {
      const int row = j * strideZ;
      for (int i = 1; i < tnRRow - 1; ++i) {
        const int idx = row + i;
        resM[idx] = ih2 * (coefficient2[i] * potM[idx - 1] + tempRatioZ * (potM[idx - strideZ] + potM[idx + strideZ]) + coefficient1[i] * potM[idx + 1] +
                           coefficient3[i] * (potP[idx] + potMm[idx]) - inverseCoefficient4[i] * potM[idx]) +
                    chargeM[idx];
      } // end cols
    } // end mParamGrid.NRVertices
  }
}

template <typename DataT>
void PoissonSolver<DataT>::getPhiNeighbours(const int m, const int nPhi, int& mp1, int& mm1)
{
  // periodic boundary in phi
  mp1 = (m + 1 > nPhi - 1) ? (m + 1 - nPhi) : (m + 1);
  mm1 = (m - 1 < 0) ? (m - 1 + nPhi) : (m - 1);
}

template <typename DataT>
template <bool Add>
void PoissonSolver<DataT>::interpolate3D(Vector& matricesCurrentV, const Vector& matricesCurrentVC, const int tnRRow, const int tnZColumn, const int newPhiSlice, const int oldPhiSlice) const
{
  if (newPhiSlice == 2 * oldPhiSlice) {
    // interpolation also in phi: each iteration writes only to the fine slices m and m + 1
#pragma omp parallel for num_threads(sNThreads) if ((sNThreads > 1) && (static_cast<long>(tnRRow) * tnZColumn * newPhiSlice > MINPOINTSPARALLEL))
    for (int mm = 0; mm < oldPhiSlice; ++mm) {
      const int m = 2 * mm;
      const int mp1 = m + 1;                                                             // newPhiSlice is even: m + 1 < newPhiSlice
      const int mmPlus = (mm + 1 > oldPhiSlice - 1) ? (mm + 1 - oldPhiSlice) : (mm + 1); // periodic boundary in phi

      for (int j = 2; j < tnZColumn - 1; j += 2) {
        for (int i = 2; i < tnRRow - 1; i += 2) {
          const int iHalf = i / 2;
          const int jHalf = j / 2;
          setOrAdd<Add>(matricesCurrentV(i, j, m), matricesCurrentVC(iHalf, jHalf, mm));
          // point on corner lines at phi direction
          setOrAdd<Add>(matricesCurrentV(i, j, mp1), (matricesCurrentVC(iHalf, jHalf, mm) + matricesCurrentVC(iHalf, jHalf, mmPlus)) / 2);
        }
      }

      for (int j = 1; j < tnZColumn - 1; j += 2) {
        for (int i = 2; i < tnRRow - 1; i += 2) {
          const int iHalf = i / 2;
          const int jHalf = j / 2;
          setOrAdd<Add>(matricesCurrentV(i, j, m), (matricesCurrentVC(iHalf, jHalf, mm) + matricesCurrentVC(iHalf, jHalf + 1, mm)) / 2);
          // point on corner lines at phi direction
          setOrAdd<Add>(matricesCurrentV(i, j, mp1), (matricesCurrentVC(iHalf, jHalf, mm) + matricesCurrentVC(iHalf, jHalf + 1, mm) + matricesCurrentVC(iHalf, jHalf, mmPlus) + matricesCurrentVC(iHalf, jHalf + 1, mmPlus)) / 4);
        }
      }

      for (int j = 2; j < tnZColumn - 1; j += 2) {
        for (int i = 1; i < tnRRow - 1; i += 2) {
          const int iHalf = i / 2;
          const int jHalf = j / 2;
          setOrAdd<Add>(matricesCurrentV(i, j, m), (matricesCurrentVC(iHalf, jHalf, mm) + matricesCurrentVC(iHalf + 1, jHalf, mm)) / 2);
          // point on line at phi direction
          setOrAdd<Add>(matricesCurrentV(i, j, mp1), ((matricesCurrentVC(iHalf, jHalf, mm) + matricesCurrentVC(iHalf, jHalf, mmPlus)) + (matricesCurrentVC(iHalf + 1, jHalf, mmPlus) + matricesCurrentVC(iHalf + 1, jHalf, mm))) / 4);
        }
      }

      for (int j = 1; j < tnZColumn - 1; j += 2) {
        for (int i = 1; i < tnRRow - 1; i += 2) {
          const int iHalf = i / 2;
          const int jHalf = j / 2;
          setOrAdd<Add>(matricesCurrentV(i, j, m), ((matricesCurrentVC(iHalf, jHalf, mm) + matricesCurrentVC(iHalf, jHalf + 1, mm)) + (matricesCurrentVC(iHalf + 1, jHalf, mm) + matricesCurrentVC(iHalf + 1, jHalf + 1, mm))) / 4);
          // point at the center at phi direction
          setOrAdd<Add>(matricesCurrentV(i, j, mp1), ((matricesCurrentVC(iHalf, jHalf, mm) + matricesCurrentVC(iHalf, jHalf + 1, mm) + matricesCurrentVC(iHalf, jHalf, mmPlus) + matricesCurrentVC(iHalf, jHalf + 1, mmPlus)) +
                                                      (matricesCurrentVC(iHalf + 1, jHalf, mm) + matricesCurrentVC(iHalf + 1, jHalf + 1, mm) + matricesCurrentVC(iHalf + 1, jHalf, mmPlus) + matricesCurrentVC(iHalf + 1, jHalf + 1, mmPlus))) /
                                                       8);
        }
      }
    }
  } else {
    // same number of phi slices: interpolation in r and z for each phi slice
#pragma omp parallel for num_threads(sNThreads) if ((sNThreads > 1) && (static_cast<long>(tnRRow) * tnZColumn * newPhiSlice > MINPOINTSPARALLEL))
    for (int m = 0; m < newPhiSlice; ++m) {
      interpolate2D<Add>(matricesCurrentV, matricesCurrentVC, tnRRow, tnZColumn, m);
    }
  }
}

template <typename DataT>
template <bool Add>
void PoissonSolver<DataT>::interpolate2D(Vector& matricesCurrentV, const Vector& matricesCurrentVC, const int tnRRow, const int tnZColumn, const int iphi) const
{
  for (int j = 2; j < tnZColumn - 1; j += 2) {
    for (int i = 2; i < tnRRow - 1; i += 2) {
      setOrAdd<Add>(matricesCurrentV(i, j, iphi), matricesCurrentVC(i / 2, j / 2, iphi));
    }
  }

  for (int j = 1; j < tnZColumn - 1; j += 2) {
    for (int i = 2; i < tnRRow - 1; i += 2) {
      const int iHalf = i / 2;
      const int jHalf = j / 2;
      setOrAdd<Add>(matricesCurrentV(i, j, iphi), (matricesCurrentVC(iHalf, jHalf, iphi) + matricesCurrentVC(iHalf, jHalf + 1, iphi)) / 2);
    }
  }

  for (int j = 2; j < tnZColumn - 1; j += 2) {
    for (int i = 1; i < tnRRow - 1; i += 2) {
      const int iHalf = i / 2;
      const int jHalf = j / 2;
      setOrAdd<Add>(matricesCurrentV(i, j, iphi), (matricesCurrentVC(iHalf, jHalf, iphi) + matricesCurrentVC(iHalf + 1, jHalf, iphi)) / 2);
    }
  }

  for (int j = 1; j < tnZColumn - 1; j += 2) {
    for (int i = 1; i < tnRRow - 1; i += 2) {
      const int iHalf = i / 2;
      const int jHalf = j / 2;
      setOrAdd<Add>(matricesCurrentV(i, j, iphi), (matricesCurrentVC(iHalf, jHalf, iphi) + matricesCurrentVC(iHalf, jHalf + 1, iphi) + matricesCurrentVC(iHalf + 1, jHalf, iphi) + matricesCurrentVC(iHalf + 1, jHalf + 1, iphi)) / 4);
    }
  }
}

template <typename DataT>
void PoissonSolver<DataT>::relax3D(Vector& matricesCurrentV, const Vector& matricesCurrentCharge, const int tnRRow, const int tnZColumn, const int iPhi, const DataT h2,
                                   const DataT tempRatioZ, const std::vector<DataT>& coefficient1, const std::vector<DataT>& coefficient2, const std::vector<DataT>& coefficient3, const std::vector<DataT>& coefficient4) const
{
  // Gauss-Seidel (Red Black)
  // strides of the storage: index = i + nr * (j + nz * m)
  const int nr = matricesCurrentV.getNr();
  const int strideZ = nr;
  const int stridePhi = nr * matricesCurrentV.getNz();
  const DataT* charge = matricesCurrentCharge.data().data();
  DataT* pot = matricesCurrentV.data().data();

  // relax all points of one colour in phi slice m
  auto relaxSlice = [&](const int msw, const int m) {
    const int jsw = ((msw + m) % 2) ? 1 : 2;
    int mp1, mm1;
    getPhiNeighbours(m, iPhi, mp1, mm1);
    DataT* potM = pot + m * stridePhi;
    const DataT* potP = pot + mp1 * stridePhi;
    const DataT* potMm = pot + mm1 * stridePhi;
    const DataT* chargeM = charge + m * stridePhi;
    int isw = jsw;
    for (int j = 1; j < tnZColumn - 1; ++j, isw = 3 - isw) {
      const int row = j * strideZ;
      for (int i = isw; i < tnRRow - 1; i += 2) {
        const int idx = row + i;
        potM[idx] = (coefficient2[i] * potM[idx - 1] + tempRatioZ * (potM[idx - strideZ] + potM[idx + strideZ]) + coefficient1[i] * potM[idx + 1] + coefficient3[i] * (potP[idx] + potMm[idx]) + (h2 * chargeM[idx])) * coefficient4[i];
      } // end cols
    } // end mParamGrid.NRVertices
  };

  // In one pass only the points with one parity of (i + j + m) are updated. All direct neighbours of these points have the other parity,
  // which makes the updates within one pass independent of each other and the phi slices can be processed in parallel.
  // The only exception is the periodic case with an odd number of phi slices: the first and the last slice are neighbours with the same parity.
  // To get exactly the same result as with the sequential loop (m = 0, 1, ..., iPhi - 1), the last slice is processed after all other slices.
  const bool runParallel = (sNThreads > 1) && (static_cast<long>(tnRRow) * tnZColumn * iPhi > MINPOINTSPARALLEL);
  for (int iPass = 1; iPass <= 2; ++iPass) {
    const int msw = (iPass % 2) ? 1 : 2;
#pragma omp parallel for num_threads(sNThreads) if (runParallel)
    for (int m = 0; m < iPhi - 1; ++m) {
      relaxSlice(msw, m);
    }
    relaxSlice(msw, iPhi - 1);
  } // end sweep
}

template <typename DataT>
void PoissonSolver<DataT>::relaxLevel(const MGLevel& level, Vector& matricesCurrentV, const Vector& matricesCurrentCharge, const DataT h2, const DataT tempRatioZ,
                                      const std::vector<DataT>& coefficient1, const std::vector<DataT>& coefficient2, const std::vector<DataT>& coefficient3, const std::vector<DataT>& coefficient4) const
{
  if (level.lineRelaxPhi) {
    relaxLinePhi(matricesCurrentV, matricesCurrentCharge, level.nr, level.nz, level.nphi, h2, tempRatioZ, coefficient1, coefficient2, coefficient3, coefficient4);
  } else {
    relax3D(matricesCurrentV, matricesCurrentCharge, level.nr, level.nz, level.nphi, h2, tempRatioZ, coefficient1, coefficient2, coefficient3, coefficient4);
  }
}

template <typename DataT>
void PoissonSolver<DataT>::relaxLinePhi(Vector& matricesCurrentV, const Vector& matricesCurrentCharge, const int tnRRow, const int tnZColumn, const int iPhi, const DataT h2, const DataT tempRatioZ,
                                        const std::vector<DataT>& coefficient1, const std::vector<DataT>& coefficient2, const std::vector<DataT>& coefficient3, const std::vector<DataT>& coefficient4) const
{
  // For each line (i, j) the equations a_i * V_m - b_i * (V_{m-1} + V_{m+1}) = rhs_m with periodic boundary in phi are solved, with
  // a_i = 1 / coefficient4[i] = 2 * (1 + tempRatioZ + coefficient3[i]), b_i = coefficient3[i] and rhs_m containing the neighbours in r and z and the charge.
  // The periodic tridiagonal system is solved with the Thomas algorithm and the Sherman-Morrison formula (see Numerical Recipes, cyclic tridiagonal systems).
  // The matrix depends only on i: the factorisation is calculated once for all lines. Storage of the factorisation: index = i + nr * m
  const int nr = matricesCurrentV.getNr();
  const int strideZ = nr;
  const int stridePhi = nr * matricesCurrentV.getNz();
  const int n = iPhi;
  std::vector<DataT> cPrime(static_cast<size_t>(nr) * n);    // modified super diagonal of the Thomas algorithm
  std::vector<DataT> invDen(static_cast<size_t>(nr) * n);    // inverse of the modified diagonal of the Thomas algorithm
  std::vector<DataT> zSM(static_cast<size_t>(nr) * n);       // solution of T z = u for the Sherman-Morrison correction
  std::vector<DataT> factDenInv(nr), bOverA(nr), bCoeff(nr); // 1 / (1 + z_0 + b/a * z_{n-1}), b/a, b
  const bool runParallel = (sNThreads > 1) && (static_cast<long>(tnRRow) * tnZColumn * iPhi > MINPOINTSPARALLEL);

  // the factorisations for different i are independent
#pragma omp parallel for num_threads(sNThreads) if (runParallel)
  for (int i = 1; i < tnRRow - 1; ++i) {
    const DataT a = 1 / coefficient4[i];
    const DataT b = coefficient3[i];
    const DataT gamma = -a;
    bCoeff[i] = b;
    bOverA[i] = b / a;
    // modified diagonal: corners -b of the periodic matrix are moved to the rank one correction u * v^T with u = (gamma, 0, ..., 0, -b), v = (1, 0, ..., 0, -b / gamma)
    const auto diag = [&](const int m) { return (m == 0) ? (a - gamma) : ((m == n - 1) ? (a - b * b / gamma) : a); };
    DataT den = diag(0);
    invDen[i] = 1 / den;
    cPrime[i] = -b / den;
    for (int m = 1; m < n; ++m) {
      den = diag(m) + b * cPrime[i + nr * (m - 1)];
      invDen[i + nr * m] = 1 / den;
      cPrime[i + nr * m] = -b / den;
    }
    // solve T z = u
    DataT d = gamma * invDen[i];
    zSM[i] = d;
    for (int m = 1; m < n; ++m) {
      const DataT u = (m == n - 1) ? -b : 0;
      d = (u + b * d) * invDen[i + nr * m];
      zSM[i + nr * m] = d;
    }
    for (int m = n - 2; m >= 0; --m) {
      zSM[i + nr * m] -= cPrime[i + nr * m] * zSM[i + nr * (m + 1)];
    }
    factDenInv[i] = 1 / (1 + zSM[i] + bOverA[i] * zSM[i + nr * (n - 1)]);
  }

  const DataT* charge = matricesCurrentCharge.data().data();
  DataT* pot = matricesCurrentV.data().data();

  // the lines are distributed to the threads in blocks of consecutive i of one row j. Several blocks per row are used to get enough tasks for a good load balance
  const int nRows = tnZColumn - 2;
  const int nInner = tnRRow - 2; // number of inner vertices in r
  int nBlocks = 1;
  if (runParallel) {
    const int tasksPerThread = 16;
    const int minLinesPerBlock = 2;
    nBlocks = std::clamp((tasksPerThread * sNThreads + nRows - 1) / nRows, 1, std::max(1, nInner / (2 * minLinesPerBlock)));
  }
  const int blockSize = 2 * ((nInner + 2 * nBlocks - 1) / (2 * nBlocks)); // even number of i per block
  const int nTasks = nRows * nBlocks;

#pragma omp parallel num_threads(sNThreads) if (runParallel)
  {
    std::vector<DataT> work(static_cast<size_t>(blockSize) * n); // forward elimination and solution for the lines of one block: index = (i - iLow) + blockSize * m

    // the lines with even and odd (i + j) are processed one after the other: all neighbours in r and z of a line have the other parity, the lines of one parity are independent
    for (int parity = 0; parity < 2; ++parity) {
#pragma omp for schedule(static)
      for (int task = 0; task < nTasks; ++task) {
        const int j = 1 + task / nBlocks;
        const int iLow = 1 + (task % nBlocks) * blockSize;
        const int iEnd = std::min(iLow + blockSize, tnRRow - 1);
        const int iStart = ((iLow + j) % 2 == parity) ? iLow : iLow + 1; // lines with (i + j) % 2 == parity
        const int row = j * strideZ;
        DataT* workBlock = work.data();
        // forward elimination
        for (int m = 0; m < n; ++m) {
          const DataT* potM = pot + m * stridePhi;
          const DataT* chargeM = charge + m * stridePhi;
          DataT* workM = workBlock + blockSize * m;
          const DataT* workPrev = workBlock + blockSize * std::max(m - 1, 0); // not used for m = 0
          for (int i = iStart; i < iEnd; i += 2) {
            const int idx = row + i;
            const int iw = i - iLow;
            const DataT rhs = coefficient2[i] * potM[idx - 1] + coefficient1[i] * potM[idx + 1] + tempRatioZ * (potM[idx - strideZ] + potM[idx + strideZ]) + h2 * chargeM[idx];
            workM[iw] = (m == 0) ? rhs * invDen[i] : (rhs + bCoeff[i] * workPrev[iw]) * invDen[i + nr * m];
          }
        }
        // back substitution
        for (int m = n - 2; m >= 0; --m) {
          DataT* workM = workBlock + blockSize * m;
          const DataT* workNext = workBlock + blockSize * (m + 1);
          for (int i = iStart; i < iEnd; i += 2) {
            workM[i - iLow] -= cPrime[i + nr * m] * workNext[i - iLow];
          }
        }
        // Sherman-Morrison correction and store the solution
        const DataT* work0 = workBlock;
        const DataT* workLast = workBlock + blockSize * (n - 1);
        for (int m = 0; m < n; ++m) {
          DataT* potM = pot + m * stridePhi;
          const DataT* workM = workBlock + blockSize * m;
          for (int i = iStart; i < iEnd; i += 2) {
            const DataT fact = (work0[i - iLow] + bOverA[i] * workLast[i - iLow]) * factDenInv[i];
            potM[row + i] = workM[i - iLow] - fact * zSM[i + nr * m];
          }
        }
      }
    }
  }
}

template <typename DataT>
void PoissonSolver<DataT>::restrictBoundary3D(Vector& matricesCurrentCharge, const Vector& residue, const int tnRRow, const int tnZColumn, const int newPhiSlice, const int oldPhiSlice) const
{
  // the phi direction is coarsened if the number of phi slices is halved
  const int phiFactor = (2 * newPhiSlice == oldPhiSlice) ? 2 : 1;
  for (int m = 0; m < newPhiSlice; ++m) {
    restrictBoundary2D(matricesCurrentCharge, residue, tnRRow, tnZColumn, m, phiFactor * m);
  }
}

template <typename DataT>
void PoissonSolver<DataT>::restrictBoundary2D(Vector& matricesCurrentCharge, const Vector& residue, const int tnRRow, const int tnZColumn, const int phiCoarse, const int phiFine) const
{
  // boundary in r
  for (int j = 0, jj = 0; j < tnZColumn; ++j, jj += 2) {
    matricesCurrentCharge(0, j, phiCoarse) = residue(0, jj, phiFine);
    matricesCurrentCharge(tnRRow - 1, j, phiCoarse) = residue((tnRRow - 1) * 2, jj, phiFine);
  }

  // boundary in z
  for (int i = 0, ii = 0; i < tnRRow; ++i, ii += 2) {
    matricesCurrentCharge(i, 0, phiCoarse) = residue(ii, 0, phiFine);
    matricesCurrentCharge(i, tnZColumn - 1, phiCoarse) = residue(ii, (tnZColumn - 1) * 2, phiFine);
  }
}

template <typename DataT>
void PoissonSolver<DataT>::restrict3D(Vector& matricesCurrentCharge, const Vector& residue, const int tnRRow, const int tnZColumn, const int newPhiSlice, const int oldPhiSlice) const
{
  const bool runParallel = (sNThreads > 1) && (static_cast<long>(tnRRow) * tnZColumn * newPhiSlice > MINPOINTSPARALLEL / 4);
  if (2 * newPhiSlice == oldPhiSlice) {
    // each iteration writes only to the coarse slice m
#pragma omp parallel for num_threads(sNThreads) if (runParallel)
    for (int m = 0; m < newPhiSlice; ++m) {
      const int mm = 2 * m;
      int mp1, mm1;
      getPhiNeighbours(mm, oldPhiSlice, mp1, mm1);

      for (int i = 1, ii = 2; i < tnRRow - 1; ++i, ii += 2) {
        for (int j = 1, jj = 2; j < tnZColumn - 1; ++j, jj += 2) {

          // at the same plane
          const int iip1 = ii + 1;
          const int iim1 = ii - 1;
          const int jjp1 = jj + 1;
          const int jjm1 = jj - 1;
          const DataT s1 = residue(iip1, jj, mm) + residue(iim1, jj, mm) + residue(ii, jjp1, mm) + residue(ii, jjm1, mm) + residue(ii, jj, mp1) + residue(ii, jj, mm1);

          const DataT s2 = (residue(iip1, jjp1, mm) + residue(iip1, jjm1, mm) + residue(iip1, jj, mp1) + residue(iip1, jj, mm1)) +
                           (residue(iim1, jjm1, mm) + residue(iim1, jjp1, mm) + residue(iim1, jj, mp1) + residue(iim1, jj, mm1)) +
                           residue(ii, jjm1, mp1) + residue(ii, jjp1, mm1) + residue(ii, jjm1, mm1) + residue(ii, jjp1, mp1);

          const DataT s3 = (residue(iip1, jjp1, mp1) + residue(iip1, jjm1, mp1) + residue(iip1, jjp1, mm1) + residue(iip1, jjm1, mm1)) +
                           (residue(iim1, jjm1, mm1) + residue(iim1, jjp1, mm1) + residue(iim1, jjm1, mp1) + residue(iim1, jjp1, mp1));

          matricesCurrentCharge(i, j, m) = residue(ii, jj, mm) / 8 + s1 / 16 + s2 / 32 + s3 / 64;
        } // end cols
      } // end mParamGrid.NRVertices
      restrictBoundary2D(matricesCurrentCharge, residue, tnRRow, tnZColumn, m, mm);
    } // end phis

  } else {
#pragma omp parallel for num_threads(sNThreads) if (runParallel)
    for (int m = 0; m < newPhiSlice; ++m) {
      restrict2D(matricesCurrentCharge, residue, tnRRow, tnZColumn, m);
    }
  }
}

template <typename DataT>
void PoissonSolver<DataT>::restrict2D(Vector& matricesCurrentCharge, const Vector& residue, const int tnRRow, const int tnZColumn, const int iphi) const
{
  for (int i = 1, ii = 2; i < tnRRow - 1; ++i, ii += 2) {
    for (int j = 1, jj = 2; j < tnZColumn - 1; ++j, jj += 2) {
      const int iip1 = ii + 1;
      const int iim1 = ii - 1;
      const int jjp1 = jj + 1;
      const int jjm1 = jj - 1;
      matricesCurrentCharge(i, j, iphi) = residue(ii, jj, iphi) / 4 + (residue(iip1, jj, iphi) + residue(iim1, jj, iphi) + residue(ii, jjp1, iphi) + residue(ii, jjm1, iphi)) / 8 +
                                          (residue(iip1, jjp1, iphi) + residue(iim1, jjp1, iphi) + residue(iip1, jjm1, iphi) + residue(iim1, jjm1, iphi)) / 16;
    } // end cols
  } // end mParamGrid.NRVertices
  restrictBoundary2D(matricesCurrentCharge, residue, tnRRow, tnZColumn, iphi, iphi);
}

template <typename DataT>
DataT PoissonSolver<DataT>::getConvergenceError(const Vector& matricesCurrentV, const Vector& prevArrayV) const
{
  const int nPhi = prevArrayV.getNphi();
  const size_t phiStep = static_cast<size_t>(prevArrayV.getNr()) * prevArrayV.getNz(); // number of points in one phi slice
  std::vector<DataT> errorArr(nPhi);
  std::vector<DataT> normArr(nPhi);
  const DataT* prev = prevArrayV.data().data();
  const DataT* curr = matricesCurrentV.data().data();

#pragma omp parallel for num_threads(sNThreads) if ((sNThreads > 1) && (phiStep * nPhi > MINPOINTSPARALLEL))
  for (int m = 0; m < nPhi; ++m) {
    // sum of the squared differences and of the squared potential in the phi slice
    const DataT* prevM = prev + m * phiStep;
    const DataT* currM = curr + m * phiStep;
    DataT sum = 0;
    DataT sumNorm = 0;
    for (size_t k = 0; k < phiStep; ++k) {
      const DataT diff = prevM[k] - currM[k];
      sum = sum + diff * diff;
      sumNorm = sumNorm + currM[k] * currM[k];
    }
    errorArr[m] = sum;
    normArr[m] = sumNorm;
  }

  if (MGParameters::relativeConvergence) {
    // relative change of the potential: sqrt(sum dV^2 / sum V^2)
    const DataT sumDiff = std::accumulate(errorArr.begin(), errorArr.end(), DataT(0));
    const DataT sumNorm = std::accumulate(normArr.begin(), normArr.end(), DataT(0));
    return (sumNorm > 0) ? std::sqrt(sumDiff / sumNorm) : std::sqrt(sumDiff);
  }

  // largest sum of dV^2 of the phi slices
  return *std::max_element(std::begin(errorArr), std::end(errorArr));
}

template <typename DataT>
void PoissonSolver<DataT>::calcCoefficients(unsigned int from, unsigned int to, const DataT h, const DataT tempRatioZ, const DataT tempRatioPhi, std::vector<DataT>& coefficient1, std::vector<DataT>& coefficient2, std::vector<DataT>& coefficient3, std::vector<DataT>& coefficient4) const
{
  for (unsigned int i = from; i < to; ++i) {
    const DataT radiusInv = 1 / (mGrid3D.getGridMinR() + i * h);
    const DataT hRadiusTmp = h * radiusInv / 2;
    coefficient1[i] = 1 + hRadiusTmp;
    coefficient2[i] = 1 - hRadiusTmp;
    coefficient3[i] = tempRatioPhi * radiusInv * radiusInv;
    coefficient4[i] = 1 / (2 * (1 + tempRatioZ + coefficient3[i]));
  }
}

template <typename DataT>
DataT PoissonSolver<DataT>::getRelCouplingPhi(const MGLevel& level, const DataT radius) const
{
  const DataT spacingRZ = std::min(getSpacingR(), getSpacingZ()) * level.rzFactor;
  const DataT spacingPhiInv = level.nphi * getGridSizePhiInv();
  const DataT rel = spacingRZ * spacingPhiInv / radius;
  return rel * rel;
}

template <typename DataT>
std::vector<typename PoissonSolver<DataT>::MGLevel> PoissonSolver<DataT>::getMultiGridLevels(const int nLevelsRZ) const
{
  std::vector<MGLevel> levels{MGLevel{mParamGrid.NRVertices, mParamGrid.NZVertices, mParamGrid.NPhiVertices, 1}};
  const DataT radiusMin = mGrid3D.getGridMinR();
  const int maxRZFactor = 1 << (nLevelsRZ - 1);
  while (levels.back().rzFactor < maxRZFactor) {
    const MGLevel fine = levels.back();

    // The point-wise relaxation does not smooth the error well in a direction which is only weakly coupled compared to the other directions
    // and coarsening in such a direction leads to a slow convergence of the multi grid (number of V cycles increases).
    // Coarsen r and z. Coarsen phi only if the coupling in phi (largest at the inner radius) is not weak compared to the coupling in r/z: (h_{r,z} / (r_{min} * h_{phi}))^2
    bool coarsenPhi = (fine.nphi % 2 == 0);
    if (coarsenPhi && (MGParameters::phiCoarseningThreshold > 0)) {
      coarsenPhi = getRelCouplingPhi(fine, radiusMin) >= MGParameters::phiCoarseningThreshold;
    }
    const int rzFactor = 2 * fine.rzFactor;
    levels.push_back(MGLevel{mParamGrid.NRVertices / rzFactor + 1, mParamGrid.NZVertices / rzFactor + 1, coarsenPhi ? fine.nphi / 2 : fine.nphi, rzFactor});
  }

  // line relaxation in phi on the levels where phi is strongly coupled
  std::string levelInfo;
  for (auto& level : levels) {
    // the periodic tridiagonal solver needs at least 3 phi slices (for 1 or 2 slices the neighbours in phi coincide, the point-wise relaxation is used)
    level.lineRelaxPhi = (MGParameters::phiLineRelaxThreshold > 0) && (level.nphi >= 3) && (getRelCouplingPhi(level, radiusMin) >= MGParameters::phiLineRelaxThreshold);
    levelInfo += fmt::format(" {}x{}x{}{}", level.nr, level.nz, level.nphi, level.lineRelaxPhi ? "(line)" : "");
  }
  LOGP(detail, "Multi grid levels (r x z x phi):{}", levelInfo);
  return levels;
}

template <typename DataT>
DataT PoissonSolver<DataT>::getGridSizePhiInv()
{
  return INVTWOPI * SECTORSPERSIDE / MGParameters::normalizeGridToNSector;
}

template class o2::tpc::PoissonSolver<double>;
template class o2::tpc::PoissonSolver<float>;
