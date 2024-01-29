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
#include <fmt/core.h>
#include "TPCSpaceCharge/Vector3D.h"
#include "TPCSpaceCharge/DataContainer3D.h"
#include "DataFormatsTPC/Defs.h"

#ifdef WITH_OPENMP
#include <omp.h>
#endif

using namespace o2::tpc;

template <typename DataT>
void PoissonSolver<DataT>::poissonSolver3D(DataContainer& matricesV, const DataContainer& matricesCharge, const int symmetry)
{
  using timer = std::chrono::high_resolution_clock;
  auto start = timer::now();
  if (MGParameters::isFull3D) {
    poissonMultiGrid3D(matricesV, matricesCharge, symmetry);
  } else {
    poissonMultiGrid3D2D(matricesV, matricesCharge, symmetry);
  }
  auto stop = timer::now();
  std::chrono::duration<float> time = stop - start;
  const float totalTime = time.count();
  LOGP(detail, "poissonSolver3D took {}s", totalTime);
}

template <typename DataT>
void PoissonSolver<DataT>::poissonSolver2D(DataContainer& matricesV, const DataContainer& matricesCharge)
{
  poissonMultiGrid2D(matricesV, matricesCharge);
}

template <typename DataT>
void PoissonSolver<DataT>::poissonMultiGrid2D(DataContainer& matricesV, const DataContainer& matricesCharge, const int iPhi)
{
  /// Geometry of TPC -- should be use AliTPCParams instead
  const DataT gridSpacingR = getSpacingR();
  const DataT gridSpacingZ = getSpacingZ();
  const DataT ratioZ = gridSpacingR * gridSpacingR / (gridSpacingZ * gridSpacingZ); // ratio_{Z} = gridSize_{r} / gridSize_{z}

  int nGridRow = 0; // number grid
  int nGridCol = 0; // number grid

  int nnRow = mParamGrid.NRVertices;
  while (nnRow >>= 1) {
    ++nGridRow;
  }

  int nnCol = mParamGrid.NZVertices;
  while (nnCol >>= 1) {
    ++nGridCol;
  }

  // Check that number of mParamGrid.NRVertices and mParamGrid.NZVertices is suitable for multi grid
  if (!isPowerOfTwo(mParamGrid.NRVertices - 1)) {
    LOGP(error, "PoissonMultiGrid2D: PoissonMultiGrid - Error in the number of mParamGrid.NRVertices. Must be 2**M + 1");
    return;
  }
  if (!isPowerOfTwo(mParamGrid.NZVertices - 1)) {
    LOGP(error, "PoissonMultiGrid2D: PoissonMultiGrid - Error in the number of mParamGrid.NZVertices. Must be 2**N + 1");
    return;
  }

  const int nLoop = std::max(nGridRow, nGridCol); // Calculate the number of nLoop for the binary expansion

  LOGP(detail, "{}", fmt::format("PoissonMultiGrid2D: nGridRow={}, nGridCol={}, nLoop={}, nMGCycle={}", nGridRow, nGridCol, nLoop, MGParameters::nMGCycle));

  unsigned int iOne = 1; // index
  unsigned int jOne = 1; // index
  int tnRRow = mParamGrid.NRVertices;
  int tnZColumn = mParamGrid.NZVertices;

  // Vector for storing multi grid array
  std::vector<Vector> tvArrayV(nLoop);    // potential <--> error
  std::vector<Vector> tvChargeFMG(nLoop); // charge is restricted in full multiGrid
  std::vector<Vector> tvCharge(nLoop);    // charge <--> residue
  std::vector<Vector> tvResidue(nLoop);   // residue calculation

  // Allocate memory for temporary grid
  for (int count = 1; count <= nLoop; ++count) {
    tnRRow = iOne == 1 ? mParamGrid.NRVertices : mParamGrid.NRVertices / iOne + 1;
    tnZColumn = jOne == 1 ? mParamGrid.NZVertices : mParamGrid.NZVertices / jOne + 1;
    // if one just address to matrixV
    const int index = count - 1;
    tvResidue[index].resize(tnRRow, tnZColumn, 1);
    tvChargeFMG[index].resize(tnRRow, tnZColumn, 1);
    tvArrayV[index].resize(tnRRow, tnZColumn, 1);
    tvCharge[index].resize(tnRRow, tnZColumn, 1);

    if (count == 1) {
      for (int iphi = iPhi; iphi <= iPhi; ++iphi) {
        for (int ir = 0; ir < mParamGrid.NRVertices; ++ir) {
          for (int iz = 0; iz < mParamGrid.NZVertices; ++iz) {
            tvChargeFMG[index](ir, iz, iphi) = matricesCharge(iz, ir, iphi);
            tvCharge[index](ir, iz, iphi) = matricesCharge(iz, ir, iphi);
            tvArrayV[index](ir, iz, iphi) = matricesV(iz, ir, iphi);
          }
        }
      }
    } else {
      restrict2D(tvChargeFMG[index], tvChargeFMG[count - 2], tnRRow, tnZColumn, 0);
    }
    iOne *= 2;
    jOne *= 2;
  }

  /// full multi grid
  if (MGParameters::cycleType == CycleType::FCycle) {

    LOGP(detail, "PoissonMultiGrid2D: Do full cycle");
    // FMG
    // 1) Relax on the coarsest grid
    iOne /= 2;
    jOne /= 2;
    tnRRow = iOne == 1 ? mParamGrid.NRVertices : mParamGrid.NRVertices / iOne + 1;
    tnZColumn = jOne == 1 ? mParamGrid.NZVertices : mParamGrid.NZVertices / jOne + 1;
    const DataT h = gridSpacingR * nLoop;
    const DataT h2 = h * h;
    const DataT tempRatio = ratioZ * iOne * iOne / (jOne * jOne);
    const DataT tempFourth = 1 / (2 + 2 * tempRatio);

    std::vector<DataT> coefficient1(tnRRow);
    std::vector<DataT> coefficient2(tnRRow);
    calcCoefficients2D(1, tnRRow - 1, h, coefficient1, coefficient2);

    relax2D(tvArrayV[nLoop - 1], tvChargeFMG[nLoop - 1], tnRRow, tnZColumn, h2, tempFourth, tempRatio, coefficient1, coefficient2);

    // Do VCycle from nLoop H to h
    for (int count = nLoop - 2; count >= 0; --count) {
      iOne /= 2;
      jOne /= 2;

      tnRRow = iOne == 1 ? mParamGrid.NRVertices : mParamGrid.NRVertices / iOne + 1;
      tnZColumn = jOne == 1 ? mParamGrid.NZVertices : mParamGrid.NZVertices / jOne + 1;

      interp2D(tvArrayV[count], tvArrayV[count + 1], tnRRow, tnZColumn, iPhi);

      // Copy the relax charge to the tvCharge
      tvCharge[count] = tvChargeFMG[count]; // copy

      // Do V cycle
      for (int mgCycle = 0; mgCycle < MGParameters::nMGCycle; ++mgCycle) {
        vCycle2D(count + 1, nLoop, MGParameters::nPre, MGParameters::nPost, gridSpacingR, ratioZ, tvArrayV, tvCharge, tvResidue);
      }
    }
  } else if (MGParameters::cycleType == CycleType::VCycle) {
    // 2. VCycle
    LOGP(detail, "PoissonMultiGrid2D: Do V cycle");

    int gridFrom = 1;
    int gridTo = nLoop;

    // Do MGCycle
    for (int mgCycle = 0; mgCycle < MGParameters::nMGCycle; ++mgCycle) {
      vCycle2D(gridFrom, gridTo, MGParameters::nPre, MGParameters::nPost, gridSpacingR, ratioZ, tvArrayV, tvCharge, tvResidue);
    }
  } else if (MGParameters::cycleType == CycleType::WCycle) {
    // 3. W Cycle (TODO:)
    int gridFrom = 1;
    int gridTo = nLoop;
    // Do MGCycle
    for (Int_t mgCycle = 0; mgCycle < MGParameters::nMGCycle; mgCycle++) {
      wCycle2D(gridFrom, gridTo, MGParameters::gamma, MGParameters::nPre, MGParameters::nPost, gridSpacingR, ratioZ, tvArrayV, tvCharge, tvResidue);
    }
  }

  // fill output
  for (int iphi = iPhi; iphi <= iPhi; ++iphi) {
    for (int ir = 0; ir < mParamGrid.NRVertices; ++ir) {
      for (int iz = 0; iz < mParamGrid.NZVertices; ++iz) {
        matricesV(iz, ir, iphi) = tvArrayV[0](ir, iz, iphi);
      }
    }
  }
}

template <typename DataT>
void PoissonSolver<DataT>::poissonMultiGrid3D2D(DataContainer& matricesV, const DataContainer& matricesCharge, const int symmetry)
{
  LOGP(detail, "{}", fmt::format("PoissonMultiGrid3D2D: in Poisson Solver 3D multiGrid semi coarsening mParamGrid.NRVertices={}, cols={}, mParamGrid.NPhiVertices={}", mParamGrid.NZVertices, mParamGrid.NRVertices, mParamGrid.NPhiVertices));

  // Check that the number of mParamGrid.NRVertices and mParamGrid.NZVertices is suitable for a binary expansion
  if (!isPowerOfTwo((mParamGrid.NRVertices - 1))) {
    LOGP(error, "PoissonMultiGrid3D2D: Poisson3DMultiGrid - Error in the number of mParamGrid.NRVertices. Must be 2**M + 1");
    return;
  }
  if (!isPowerOfTwo((mParamGrid.NZVertices - 1))) {
    LOGP(error, "PoissonMultiGrid3D2D: Poisson3DMultiGrid - Error in the number of mParamGrid.NZVertices. Must be 2**N + 1");
    return;
  }
  if (mParamGrid.NPhiVertices <= 3) {
    LOGP(error, "PoissonMultiGrid3D2D: Poisson3DMultiGrid - Error in the number of mParamGrid.NPhiVertices. Must be larger than 3");
    return;
  }

  const DataT gridSpacingR = getSpacingR();
  const DataT gridSpacingZ = getSpacingZ();
  const DataT gridSpacingPhi = getSpacingPhi();
  const DataT ratioPhi = gridSpacingR * gridSpacingR / (gridSpacingPhi * gridSpacingPhi); // ratio_{phi} = gridSize_{r} / gridSize_{phi}
  const DataT ratioZ = gridSpacingR * gridSpacingR / (gridSpacingZ * gridSpacingZ);       // ratio_{Z} = gridSize_{r} / gridSize_{z}

  // Solve Poisson's equation in cylindrical coordinates by multiGrid technique
  // Allow for different size grid spacing in R and Z directions
  int nGridRow = 0; // number grid
  int nGridCol = 0; // number grid
  int nnRow = mParamGrid.NRVertices;
  int nnCol = mParamGrid.NZVertices;

  while (nnRow >>= 1) {
    ++nGridRow;
  }
  while (nnCol >>= 1) {
    ++nGridCol;
  }

  const int maxVal = std::max(nGridRow, nGridCol); // Calculate the number of nLoop for the binary expansion
  const size_t nLoop = (maxVal > MGParameters::maxLoop) ? MGParameters::maxLoop : maxVal;
  unsigned int iOne = 1; // index i in gridSize r (original)
  unsigned int jOne = 1; // index j in gridSize z (original)

  std::vector<Vector> tvArrayV(nLoop);     // potential <--> error
  std::vector<Vector> tvChargeFMG(nLoop);  // charge is restricted in full multiGrid
  std::vector<Vector> tvCharge(nLoop);     // charge <--> residue
  std::vector<Vector> tvPrevArrayV(nLoop); // error calculation
  std::vector<Vector> tvResidue(nLoop);    // residue calculation

  for (unsigned int count = 1; count <= nLoop; count++) {
    const unsigned int tnRRow = iOne == 1 ? mParamGrid.NRVertices : mParamGrid.NRVertices / iOne + 1;
    const unsigned int tnZColumn = jOne == 1 ? mParamGrid.NZVertices : mParamGrid.NZVertices / jOne + 1;
    const unsigned int index = count - 1;
    tvResidue[index].resize(tnRRow, tnZColumn, mParamGrid.NPhiVertices);
    tvPrevArrayV[index].resize(tnRRow, tnZColumn, mParamGrid.NPhiVertices);

    // memory for the finest grid is from parameters
    tvChargeFMG[index].resize(tnRRow, tnZColumn, mParamGrid.NPhiVertices);
    tvArrayV[index].resize(tnRRow, tnZColumn, mParamGrid.NPhiVertices);
    tvCharge[index].resize(tnRRow, tnZColumn, mParamGrid.NPhiVertices);

    if (count == 1) {
      for (int iphi = 0; iphi < mParamGrid.NPhiVertices; ++iphi) {
        for (int ir = 0; ir < mParamGrid.NRVertices; ++ir) {
          for (int iz = 0; iz < mParamGrid.NZVertices; ++iz) {
            tvChargeFMG[index](ir, iz, iphi) = matricesCharge(iz, ir, iphi);
            tvArrayV[index](ir, iz, iphi) = matricesV(iz, ir, iphi);
          }
        }
      }
    } else {
      restrict3D(tvChargeFMG[index], tvChargeFMG[count - 2], tnRRow, tnZColumn, mParamGrid.NPhiVertices, mParamGrid.NPhiVertices);
      restrictBoundary3D(tvArrayV[index], tvArrayV[count - 2], tnRRow, tnZColumn, mParamGrid.NPhiVertices, mParamGrid.NPhiVertices);
    }
    iOne *= 2; // doubling
    jOne *= 2; // doubling
  }

  std::vector<DataT> coefficient1(mParamGrid.NRVertices);        // coefficient1(mParamGrid.NRVertices) for storing (1 + h_{r}/2r_{i}) from central differences in r direction
  std::vector<DataT> coefficient2(mParamGrid.NRVertices);        // coefficient2(mParamGrid.NRVertices) for storing (1 + h_{r}/2r_{i}) from central differences in r direction
  std::vector<DataT> coefficient3(mParamGrid.NRVertices);        // coefficient3(mParamGrid.NRVertices) for storing (1/r_{i}^2) from central differences in phi direction
  std::vector<DataT> coefficient4(mParamGrid.NRVertices);        // coefficient4(mParamGrid.NRVertices) for storing  1/2
  std::vector<DataT> inverseCoefficient4(mParamGrid.NRVertices); // inverse of coefficient4(mParamGrid.NRVertices)

  // Case full multi grid (FMG)
  if (MGParameters::cycleType == CycleType::FCycle) {
    // 1) Relax on the coarsest grid
    iOne /= 2;
    jOne /= 2;
    int tnRRow = iOne == 1 ? mParamGrid.NRVertices : mParamGrid.NRVertices / iOne + 1;
    int tnZColumn = jOne == 1 ? mParamGrid.NZVertices : mParamGrid.NZVertices / jOne + 1;

    const DataT h = getSpacingR() * iOne;
    const DataT h2 = h * h;
    const DataT iOne2 = iOne * iOne;
    const DataT tempRatioPhi = ratioPhi * iOne2; // Used tobe divided by ( m_one * m_one ) when m_one was != 1
    const DataT tempRatioZ = ratioZ * iOne2 / (jOne * jOne);

    calcCoefficients(1, tnRRow - 1, h, tempRatioZ, tempRatioPhi, coefficient1, coefficient2, coefficient3, coefficient4);

    // relax on the coarsest level
    relax3D(tvArrayV[nLoop - 1], tvChargeFMG[nLoop - 1], tnRRow, tnZColumn, mParamGrid.NPhiVertices, symmetry, h2, tempRatioZ, coefficient1, coefficient2, coefficient3, coefficient4);

    // 2) Do multiGrid v-cycle from coarsest to finest
    for (int count = nLoop - 2; count >= 0; --count) {
      // move to finer grid
      iOne /= 2;
      jOne /= 2;
      tnRRow = iOne == 1 ? mParamGrid.NRVertices : mParamGrid.NRVertices / iOne + 1;
      tnZColumn = jOne == 1 ? mParamGrid.NZVertices : mParamGrid.NZVertices / jOne + 1;

      // 2) a) Interpolate potential for h -> 2h (coarse -> fine)
      interp3D(tvArrayV[count], tvArrayV[count + 1], tnRRow, tnZColumn, mParamGrid.NPhiVertices, mParamGrid.NPhiVertices);

      // 2) c) Copy the restricted charge to charge for calculation
      tvCharge[count] = tvChargeFMG[count]; // copy

      // 2) c) Do V cycle MGParameters::nMGCycle times at most
      for (int mgCycle = 0; mgCycle < MGParameters::nMGCycle; ++mgCycle) {
        // Copy the potential to temp array for convergence calculation
        tvPrevArrayV[count] = tvArrayV[count];

        // 2) c) i) Call V cycle from grid count+1 (current fine level) to nLoop (coarsest)
        vCycle3D2D(symmetry, count + 1, nLoop, MGParameters::nPre, MGParameters::nPost, ratioZ, ratioPhi, tvArrayV, tvCharge, tvResidue, coefficient1, coefficient2, coefficient3, coefficient4, inverseCoefficient4);

        const DataT convergenceError = getConvergenceError(tvArrayV[count], tvPrevArrayV[count]);

        /// if already converge just break move to finer grid
        if (convergenceError <= sConvergenceError) {
          break;
        }
      }
    }
  }

  // fill output
  for (int iphi = 0; iphi < mParamGrid.NPhiVertices; ++iphi) {
    for (int ir = 0; ir < mParamGrid.NRVertices; ++ir) {
      for (int iz = 0; iz < mParamGrid.NZVertices; ++iz) {
        matricesV(iz, ir, iphi) = tvArrayV[0](ir, iz, iphi);
      }
    }
  }
}

template <typename DataT>
void PoissonSolver<DataT>::poissonMultiGrid3D(DataContainer& matricesV, const DataContainer& matricesCharge, const int symmetry)
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

  // Solve Poisson's equation in cylindrical coordinates by multi grid technique
  // Allow for different size grid spacing in R and Z directions
  int nGridRow = 0; // number grid
  int nGridCol = 0; // number grid
  int nGridPhi = 0;

  int nnRow = mParamGrid.NRVertices;
  while (nnRow >>= 1) {
    ++nGridRow;
  }

  int nnCol = mParamGrid.NZVertices;
  while (nnCol >>= 1) {
    ++nGridCol;
  }

  int nnPhi = mParamGrid.NPhiVertices;
  while (nnPhi % 2 == 0) {
    ++nGridPhi;
    nnPhi /= 2;
  }

  LOGP(detail, "{}", fmt::format("PoissonMultiGrid3D: nGridRow={}, nGridCol={}, nGridPhi={}", nGridRow, nGridCol, nGridPhi));
  const int nLoop = std::max({nGridRow, nGridCol, nGridPhi}); // Calculate the number of nLoop for the binary expansion

  // Vector for storing multi grid array
  unsigned int iOne = 1; // index i in gridSize r (original)
  unsigned int jOne = 1; // index j in gridSize z (original)
  unsigned int kOne = 1; // index k in gridSize phi
  int tnRRow = mParamGrid.NRVertices;
  int tnZColumn = mParamGrid.NZVertices;
  int tPhiSlice = mParamGrid.NPhiVertices;

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

  for (int count = 1; count <= nLoop; ++count) {
    // tnRRow,tnZColumn in new grid
    tnRRow = iOne == 1 ? mParamGrid.NRVertices : mParamGrid.NRVertices / iOne + 1;
    tnZColumn = jOne == 1 ? mParamGrid.NZVertices : mParamGrid.NZVertices / jOne + 1;
    tPhiSlice = kOne == 1 ? mParamGrid.NPhiVertices : mParamGrid.NPhiVertices / kOne;
    tPhiSlice = tPhiSlice < nnPhi ? nnPhi : tPhiSlice;

    // allocate memory for residue
    const int index = count - 1;
    tvResidue[index].resize(tnRRow, tnZColumn, tPhiSlice);
    tvPrevArrayV[index].resize(tnRRow, tnZColumn, tPhiSlice);
    tvChargeFMG[index].resize(tnRRow, tnZColumn, tPhiSlice);
    tvArrayV[index].resize(tnRRow, tnZColumn, tPhiSlice);
    tvCharge[index].resize(tnRRow, tnZColumn, tPhiSlice);

    // memory for the finest grid is from parameters
    if (count == 1) {
      for (int iphi = 0; iphi < mParamGrid.NPhiVertices; ++iphi) {
        for (int ir = 0; ir < mParamGrid.NRVertices; ++ir) {
          for (int iz = 0; iz < mParamGrid.NZVertices; ++iz) {
            tvChargeFMG[index](ir, iz, iphi) = matricesCharge(iz, ir, iphi);
            tvArrayV[index](ir, iz, iphi) = matricesV(iz, ir, iphi);
          }
        }
      }
      tvCharge[index] = tvChargeFMG[index];
    }
    iOne *= 2; // doubling
    jOne *= 2; // doubling
    kOne *= 2;
  }

  // Case full multi grid (FMG)
  if (MGParameters::cycleType == CycleType::FCycle) {
    // Restrict the charge to coarser grid
    iOne = 2;
    jOne = 2;
    kOne = 2;
    int otPhiSlice = mParamGrid.NPhiVertices;

    // 1) Restrict Charge and Boundary to coarser grid
    for (int count = 2; count <= nLoop; ++count) {
      // tnRRow,tnZColumn in new grid
      tnRRow = iOne == 1 ? mParamGrid.NRVertices : mParamGrid.NRVertices / iOne + 1;
      tnZColumn = jOne == 1 ? mParamGrid.NZVertices : mParamGrid.NZVertices / jOne + 1;
      tPhiSlice = kOne == 1 ? mParamGrid.NPhiVertices : mParamGrid.NPhiVertices / kOne;
      tPhiSlice = tPhiSlice < nnPhi ? nnPhi : tPhiSlice;

      LOGP(detail, "{}", fmt::format("PoissonMultiGrid3D: Restrict3D, tnRRow={}, tnZColumn={}, newPhiSlice={}, oldPhiSlice={}", tnRRow, tnZColumn, tPhiSlice, otPhiSlice));
      restrict3D(tvChargeFMG[count - 1], tvChargeFMG[count - 2], tnRRow, tnZColumn, tPhiSlice, otPhiSlice);
      // copy boundary values of V
      restrictBoundary3D(tvArrayV[count - 1], tvArrayV[count - 2], tnRRow, tnZColumn, tPhiSlice, otPhiSlice);
      otPhiSlice = tPhiSlice;

      iOne *= 2; // doubling
      jOne *= 2; // doubling
      kOne *= 2;
    }

    // Relax on the coarsest grid
    // FMG
    // 2) Relax on the coarsest grid
    // move to the coarsest + 1
    iOne /= 2;
    jOne /= 2;
    kOne /= 2;
    tnRRow = iOne == 1 ? mParamGrid.NRVertices : mParamGrid.NRVertices / iOne + 1;
    tnZColumn = jOne == 1 ? mParamGrid.NZVertices : mParamGrid.NZVertices / jOne + 1;
    tPhiSlice = kOne == 1 ? mParamGrid.NPhiVertices : mParamGrid.NPhiVertices / kOne;
    tPhiSlice = tPhiSlice < nnPhi ? nnPhi : tPhiSlice;
    otPhiSlice = tPhiSlice;

    const DataT h = gridSpacingR * iOne;
    const DataT h2 = h * h;
    const DataT gridSizePhiInv = tPhiSlice * getGridSizePhiInv();    // h_{phi}
    const DataT tempRatioPhi = h2 * gridSizePhiInv * gridSizePhiInv; // ratio_{phi} = gridSize_{r} / gridSize_{phi}
    const DataT tempRatioZ = ratioZ * iOne * iOne / (jOne * jOne);

    calcCoefficients(1, tnRRow - 1, h, tempRatioZ, tempRatioPhi, coefficient1, coefficient2, coefficient3, coefficient4);

    // 3) Relax on the coarsest grid
    relax3D(tvArrayV[nLoop - 1], tvChargeFMG[nLoop - 1], tnRRow, tnZColumn, tPhiSlice, symmetry, h2, tempRatioZ, coefficient1, coefficient2, coefficient3, coefficient4);

    // 4) V Cycle from coarsest to finest
    for (int count = nLoop - 2; count >= 0; --count) {
      // move to finer grid
      std::fill(std::begin(coefficient1), std::end(coefficient1), 0);
      std::fill(std::begin(coefficient2), std::end(coefficient2), 0);
      std::fill(std::begin(coefficient3), std::end(coefficient3), 0);
      std::fill(std::begin(coefficient4), std::end(coefficient4), 0);
      std::fill(std::begin(inverseCoefficient4), std::end(inverseCoefficient4), 0);

      iOne /= 2;
      jOne /= 2;
      kOne /= 2;

      tnRRow = iOne == 1 ? mParamGrid.NRVertices : mParamGrid.NRVertices / iOne + 1;
      tnZColumn = jOne == 1 ? mParamGrid.NZVertices : mParamGrid.NZVertices / jOne + 1;
      tPhiSlice = kOne == 1 ? mParamGrid.NPhiVertices : mParamGrid.NPhiVertices / kOne;
      tPhiSlice = tPhiSlice < nnPhi ? nnPhi : tPhiSlice;

      // 4) a) interpolate from 2h --> h grid
      interp3D(tvArrayV[count], tvArrayV[count + 1], tnRRow, tnZColumn, tPhiSlice, otPhiSlice);

      // Copy the relax charge to the tvCharge
      if (count > 0) {
        tvCharge[count] = tvChargeFMG[count];
      }

      using timer = std::chrono::high_resolution_clock;
      auto start = timer::now();
      for (int mgCycle = 0; mgCycle < MGParameters::nMGCycle; ++mgCycle) {
        // copy to store previous potential
        tvPrevArrayV[count] = tvArrayV[count];

        vCycle3D(symmetry, count + 1, nLoop, MGParameters::nPre, MGParameters::nPost, ratioZ, tvArrayV, tvCharge, tvResidue, coefficient1, coefficient2, coefficient3, coefficient4, inverseCoefficient4);

        // converge error
        const DataT convergenceError = getConvergenceError(tvArrayV[count], tvPrevArrayV[count]);
        // if already converge just break move to finer grid
        if (convergenceError <= sConvergenceError) {
          LOGP(detail, "Cycle converged. Continue to next cycle...");
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
          LOGP(warning, "Cycle {} did not convergence! Current convergence error is larger than expected convergence error: {} > {}", mgCycle, convergenceError, sConvergenceError);
        }
      }
      // keep old slice information
      otPhiSlice = tPhiSlice;
    }
  } else if (MGParameters::cycleType == CycleType::VCycle) {
    // V-cycle
    int gridFrom = 1;
    int gridTo = nLoop;

    for (int mgCycle = 0; mgCycle < MGParameters::nMGCycle; ++mgCycle) {
      // copy to store previous potential
      tvPrevArrayV[0] = tvArrayV[0];

      // Do V Cycle from the coarsest to finest grid
      vCycle3D(symmetry, gridFrom, gridTo, MGParameters::nPre, MGParameters::nPost, ratioZ, tvArrayV, tvCharge, tvResidue, coefficient1, coefficient2, coefficient3, coefficient4, inverseCoefficient4);

      // convergence error
      const DataT convergenceError = getConvergenceError(tvArrayV[0], tvPrevArrayV[0]);

      // if error already achieved then stop mg iteration
      if (convergenceError <= sConvergenceError) {
        break;
      }
    }
  }

  // fill output
  for (int iphi = 0; iphi < mParamGrid.NPhiVertices; ++iphi) {
    for (int ir = 0; ir < mParamGrid.NRVertices; ++ir) {
      for (int iz = 0; iz < mParamGrid.NZVertices; ++iz) {
        matricesV(iz, ir, iphi) = tvArrayV[0](ir, iz, iphi);
      }
    }
  }
}

template <typename DataT>
void PoissonSolver<DataT>::wCycle2D(const int gridFrom, const int gridTo, const int gamma, const int nPre, const int nPost, const DataT gridSizeR, const DataT ratio,
                                    std::vector<Vector>& tvArrayV, std::vector<Vector>& tvCharge, std::vector<Vector>& tvResidue)
{
  unsigned int iOne = 1 << (gridFrom - 1);
  unsigned int jOne = 1 << (gridFrom - 1);

  int tnRRow = iOne == 1 ? mParamGrid.NRVertices : mParamGrid.NRVertices / iOne + 1;
  int tnZColumn = jOne == 1 ? mParamGrid.NZVertices : mParamGrid.NZVertices / jOne + 1;

  std::vector<DataT> coefficient1(mParamGrid.NRVertices);
  std::vector<DataT> coefficient2(mParamGrid.NZVertices);

  // 1) Go to coarsest level
  for (int count = gridFrom; count <= gridTo - 2; ++count) {
    const int index = count - 1;
    const DataT h = gridSizeR * iOne;
    const DataT h2 = h * h;
    const DataT ih2 = 1 / h2;
    const DataT tempRatio = ratio * iOne * iOne / (jOne * jOne);
    const DataT tempFourth = 1 / (2 + 2 * tempRatio);
    const DataT inverseTempFourth = 1 / tempFourth;
    calcCoefficients2D(1, tnRRow - 1, h, coefficient1, coefficient2);
    Vector matricesCurrentV = tvArrayV[index];
    Vector matricesCurrentCharge = tvCharge[index];
    Vector residue = tvResidue[index];

    // 1) Pre-Smoothing: Gauss-Seidel Relaxation or Jacobi
    for (int jPre = 1; jPre <= nPre; ++jPre) {
      relax2D(matricesCurrentV, matricesCurrentCharge, tnRRow, tnZColumn, h2, tempFourth, tempRatio, coefficient1, coefficient2);
    }

    // 2) Residue calculation
    residue2D(residue, matricesCurrentV, matricesCurrentCharge, tnRRow, tnZColumn, ih2, inverseTempFourth, tempRatio, coefficient1, coefficient2);

    iOne *= 2;
    jOne *= 2;
    tnRRow = iOne == 1 ? mParamGrid.NRVertices : mParamGrid.NRVertices / iOne + 1;
    tnZColumn = jOne == 1 ? mParamGrid.NZVertices : mParamGrid.NZVertices / jOne + 1;

    matricesCurrentCharge = tvCharge[count];
    matricesCurrentV = tvArrayV[count];

    // 3) Restriction
    restrict2D(matricesCurrentCharge, residue, tnRRow, tnZColumn, 0);
  }

  // Do V cycle from: gridTo-1 to gridTo gamma times
  for (int iGamma = 0; iGamma < gamma; ++iGamma) {
    vCycle2D(gridTo - 1, gridTo, nPre, nPost, gridSizeR, ratio, tvArrayV, tvCharge, tvResidue);
  }

  // Go to finest grid
  for (int count = gridTo - 2; count >= gridFrom; --count) {
    iOne /= 2;
    jOne /= 2;

    const DataT h = gridSizeR * iOne;
    const DataT h2 = h * h;
    const DataT tempRatio = ratio * iOne * iOne / (jOne * jOne);
    const DataT tempFourth = 1 / (2 + 2 * tempRatio);

    tnRRow = iOne == 1 ? mParamGrid.NRVertices : mParamGrid.NRVertices / iOne + 1;
    tnZColumn = jOne == 1 ? mParamGrid.NZVertices : mParamGrid.NZVertices / jOne + 1;
    const Vector matricesCurrentCharge = tvCharge[count - 1];
    Vector matricesCurrentV = tvArrayV[count - 1];
    const Vector matricesCurrentVC = tvArrayV[count];

    // 6) Interpolation/Prolongation
    addInterp2D(matricesCurrentV, matricesCurrentVC, tnRRow, tnZColumn, 0);

    calcCoefficients2D(1, tnRRow - 1, h, coefficient1, coefficient2);

    // 7) Post-Smoothing: Gauss-Seidel Relaxation
    for (Int_t jPost = 1; jPost <= nPost; ++jPost) {
      relax2D(matricesCurrentV, matricesCurrentCharge, tnRRow, tnZColumn, h2, tempFourth, tempRatio, coefficient1, coefficient2);
    } // end post smoothing
  }
}

template <typename DataT>
void PoissonSolver<DataT>::vCycle2D(const int gridFrom, const int gridTo, const int nPre, const int nPost, const DataT gridSizeR, const DataT ratio, std::vector<Vector>& tvArrayV,
                                    std::vector<Vector>& tvCharge, std::vector<Vector>& tvResidue)
{
  unsigned int iOne = 1 << (gridFrom - 1);
  unsigned int jOne = 1 << (gridFrom - 1);

  int tnRRow = iOne == 1 ? mParamGrid.NRVertices : mParamGrid.NRVertices / iOne + 1;
  int tnZColumn = jOne == 1 ? mParamGrid.NZVertices : mParamGrid.NZVertices / jOne + 1;

  std::vector<DataT> coefficient1(mParamGrid.NRVertices);
  std::vector<DataT> coefficient2(mParamGrid.NZVertices);

  // 1) Go to coarsest level
  for (int count = gridFrom; count <= gridTo - 1; ++count) {
    const int index = count - 1;
    const DataT h = gridSizeR * iOne;
    const DataT h2 = h * h;
    const DataT ih2 = 1 / h2;
    const DataT tempRatio = ratio * iOne * iOne / (jOne * jOne);
    const DataT tempFourth = 1 / (2 + 2 * tempRatio);
    const DataT inverseTempFourth = 1 / tempFourth;
    calcCoefficients2D(1, tnRRow - 1, h, coefficient1, coefficient2);

    for (int jPre = 1; jPre <= nPre; ++jPre) {
      relax2D(tvArrayV[index], tvCharge[index], tnRRow, tnZColumn, h2, tempFourth, tempRatio, coefficient1, coefficient2);
    }

    // 2) Residue calculation
    residue2D(tvResidue[index], tvArrayV[index], tvCharge[index], tnRRow, tnZColumn, ih2, inverseTempFourth, tempRatio, coefficient1, coefficient2);

    iOne *= 2;
    jOne *= 2;
    tnRRow = iOne == 1 ? mParamGrid.NRVertices : mParamGrid.NRVertices / iOne + 1;
    tnZColumn = jOne == 1 ? mParamGrid.NZVertices : mParamGrid.NZVertices / jOne + 1;

    // 3) Restriction
    restrict2D(tvCharge[count], tvResidue[index], tnRRow, tnZColumn, 0);

    // 4) Zeroing coarser V
    std::fill(tvArrayV[count].begin(), tvArrayV[count].end(), 0); // is this necessary???
  }

  // 5) coarsest grid
  const DataT h = gridSizeR * iOne;
  const DataT h2 = h * h;
  const DataT tempRatio = ratio * iOne * iOne / (jOne * jOne);
  const DataT tempFourth = 1 / (2 + 2 * tempRatio);
  calcCoefficients2D(1, tnRRow - 1, h, coefficient1, coefficient2);

  relax2D(tvArrayV[gridTo - 1], tvCharge[gridTo - 1], tnRRow, tnZColumn, h2, tempFourth, tempRatio, coefficient1, coefficient2);

  // Go to finest grid
  for (int count = gridTo - 1; count >= gridFrom; count--) {
    const int index = count - 1;
    iOne /= 2;
    jOne /= 2;

    const DataT hTmp = gridSizeR * iOne;
    const DataT h2Tmp = hTmp * hTmp;
    const DataT tempRatioTmp = ratio * iOne * iOne / (jOne * jOne);
    const DataT tempFourthTmp = 1 / (2 + 2 * tempRatioTmp);

    tnRRow = iOne == 1 ? mParamGrid.NRVertices : mParamGrid.NRVertices / iOne + 1;
    tnZColumn = jOne == 1 ? mParamGrid.NZVertices : mParamGrid.NZVertices / jOne + 1;

    // 6) Interpolation/Prolongation
    addInterp2D(tvArrayV[index], tvArrayV[count], tnRRow, tnZColumn, 0);

    calcCoefficients2D(1, tnRRow - 1, hTmp, coefficient1, coefficient2);

    // 7) Post-Smoothing: Gauss-Seidel Relaxation
    for (int jPost = 1; jPost <= nPost; ++jPost) {
      relax2D(tvArrayV[index], tvCharge[index], tnRRow, tnZColumn, h2Tmp, tempFourthTmp, tempRatioTmp, coefficient1, coefficient2);
    } // end post smoothing
  }
}

template <typename DataT>
void PoissonSolver<DataT>::vCycle3D2D(const int symmetry, const int gridFrom, const int gridTo, const int nPre, const int nPost, const DataT ratioZ, const DataT ratioPhi,
                                      std::vector<Vector>& tvArrayV, std::vector<Vector>& tvCharge, std::vector<Vector>& tvResidue, std::vector<DataT>& coefficient1,
                                      std::vector<DataT>& coefficient2, std::vector<DataT>& coefficient3, std::vector<DataT>& coefficient4, std::vector<DataT>& inverseCoefficient4) const
{
  unsigned int iOne = 1 << (gridFrom - 1);
  unsigned int jOne = 1 << (gridFrom - 1);
  int tnRRow = iOne == 1 ? mParamGrid.NRVertices : mParamGrid.NRVertices / iOne + 1;
  int tnZColumn = jOne == 1 ? mParamGrid.NZVertices : mParamGrid.NZVertices / jOne + 1;

  for (int count = gridFrom; count <= gridTo - 1; ++count) {
    const int index = count - 1;
    const DataT h = getSpacingR() * iOne;
    const DataT h2 = h * h;
    const DataT ih2 = 1 / h2;
    const int iOne2 = iOne * iOne;
    const DataT tempRatioPhi = ratioPhi * iOne2; // Used tobe divided by ( m_one * m_one ) when m_one was != 1
    const DataT tempRatioZ = ratioZ * iOne2 / (jOne * jOne);

    calcCoefficients(1, tnRRow - 1, h, tempRatioZ, tempRatioPhi, coefficient1, coefficient2, coefficient3, coefficient4);
    for (unsigned int i = 1; i < tnRRow - 1; ++i) {
      inverseCoefficient4[i] = 1 / coefficient4[i];
    }

    // Info("VCycle3D2D","Before Pre-smoothing");
    //  1) Pre-Smoothing: Gauss-Seidel Relaxation or Jacobi
    for (int jPre = 1; jPre <= nPre; ++jPre) {
      relax3D(tvArrayV[index], tvCharge[index], tnRRow, tnZColumn, mParamGrid.NPhiVertices, symmetry, h2, tempRatioZ, coefficient1, coefficient2, coefficient3, coefficient4);
    } // end pre smoothing

    // 2) Residue calculation
    residue3D(tvResidue[index], tvArrayV[index], tvCharge[index], tnRRow, tnZColumn, mParamGrid.NPhiVertices, symmetry, ih2, tempRatioZ, coefficient1, coefficient2, coefficient3, inverseCoefficient4);

    iOne *= 2;
    jOne *= 2;
    tnRRow = iOne == 1 ? mParamGrid.NRVertices : mParamGrid.NRVertices / iOne + 1;
    tnZColumn = jOne == 1 ? mParamGrid.NZVertices : mParamGrid.NZVertices / jOne + 1;

    // 3) Restriction
    restrict3D(tvCharge[count], tvResidue[index], tnRRow, tnZColumn, mParamGrid.NPhiVertices, mParamGrid.NPhiVertices);

    // 4) Zeroing coarser V
    std::fill(tvArrayV[count].begin(), tvArrayV[count].end(), 0);
  }

  // coarsest grid
  const DataT hTmp = getSpacingR() * iOne;
  const DataT h2Tmp = hTmp * hTmp;

  const int iOne2Tmp = iOne * iOne;
  const DataT tempRatioPhiTmp = ratioPhi * iOne2Tmp;
  const DataT tempRatioZTmp = ratioZ * iOne2Tmp / (jOne * jOne);

  calcCoefficients(1, tnRRow - 1, hTmp, tempRatioZTmp, tempRatioPhiTmp, coefficient1, coefficient2, coefficient3, coefficient4);

  // 3) Relax on the coarsest grid
  relax3D(tvArrayV[gridTo - 1], tvCharge[gridTo - 1], tnRRow, tnZColumn, mParamGrid.NPhiVertices, symmetry, h2Tmp, tempRatioZTmp, coefficient1, coefficient2, coefficient3, coefficient4);

  // back to fine
  for (int count = gridTo - 1; count >= gridFrom; --count) {
    const int index = count - 1;
    iOne /= 2;
    jOne /= 2;

    tnRRow = iOne == 1 ? mParamGrid.NRVertices : mParamGrid.NRVertices / iOne + 1;
    tnZColumn = jOne == 1 ? mParamGrid.NZVertices : mParamGrid.NZVertices / jOne + 1;

    const DataT h = getSpacingR() * iOne;
    const DataT h2 = h * h;
    const int iOne2 = iOne * iOne;
    const DataT tempRatioPhi = ratioPhi * iOne2; // Used tobe divided by ( m_one * m_one ) when m_one was != 1
    const DataT tempRatioZ = ratioZ * iOne2 / (jOne * jOne);

    // 4) Interpolation/Prolongation
    addInterp3D(tvArrayV[index], tvArrayV[count], tnRRow, tnZColumn, mParamGrid.NPhiVertices, mParamGrid.NPhiVertices);

    calcCoefficients(1, tnRRow - 1, h, tempRatioZ, tempRatioPhi, coefficient1, coefficient2, coefficient3, coefficient4);

    // 5) Post-Smoothing: Gauss-Seidel Relaxation
    for (int jPost = 1; jPost <= nPost; ++jPost) {
      relax3D(tvArrayV[index], tvCharge[index], tnRRow, tnZColumn, mParamGrid.NPhiVertices, symmetry, h2, tempRatioZ, coefficient1, coefficient2, coefficient3, coefficient4);
    } // end post smoothing
  }
}

template <typename DataT>
void PoissonSolver<DataT>::vCycle3D(const int symmetry, const int gridFrom, const int gridTo, const int nPre, const int nPost, const DataT ratioZ, std::vector<Vector>& tvArrayV,
                                    std::vector<Vector>& tvCharge, std::vector<Vector>& tvResidue, std::vector<DataT>& coefficient1, std::vector<DataT>& coefficient2, std::vector<DataT>& coefficient3,
                                    std::vector<DataT>& coefficient4, std::vector<DataT>& inverseCoefficient4) const
{
  const DataT gridSpacingR = getSpacingR();

  unsigned int iOne = 1 << (gridFrom - 1);
  unsigned int jOne = 1 << (gridFrom - 1);
  unsigned int kOne = 1 << (gridFrom - 1);

  int nnPhi = mParamGrid.NPhiVertices;
  while (nnPhi % 2 == 0) {
    nnPhi /= 2;
  }

  int tnRRow = iOne == 1 ? mParamGrid.NRVertices : mParamGrid.NRVertices / iOne + 1;
  int tnZColumn = jOne == 1 ? mParamGrid.NZVertices : mParamGrid.NZVertices / jOne + 1;
  int tPhiSlice = kOne == 1 ? mParamGrid.NPhiVertices : mParamGrid.NPhiVertices / kOne;
  tPhiSlice = tPhiSlice < nnPhi ? nnPhi : tPhiSlice;

  for (int count = gridFrom; count <= gridTo - 1; ++count) {
    const int index = count - 1;
    const int otPhiSlice = tPhiSlice;
    const DataT h = gridSpacingR * iOne;
    const DataT h2 = h * h;
    const DataT ih2 = 1 / h2;
    const DataT tempGridSizePhiInv = tPhiSlice * getGridSizePhiInv();        // phi now is multiGrid
    const DataT tempRatioPhi = h2 * tempGridSizePhiInv * tempGridSizePhiInv; // ratio_{phi} = gridSize_{r} / gridSize_{phi}
    const DataT tempRatioZ = ratioZ * iOne * iOne / (jOne * jOne);

    calcCoefficients(1, tnRRow - 1, h, tempRatioZ, tempRatioPhi, coefficient1, coefficient2, coefficient3, coefficient4);

    for (int i = 1; i < tnRRow - 1; ++i) {
      inverseCoefficient4[i] = 1 / coefficient4[i];
    }

    // 1) Pre-Smoothing: Gauss-Seidel Relaxation or Jacobi
    for (int jPre = 1; jPre <= nPre; ++jPre) {
      relax3D(tvArrayV[index], tvCharge[index], tnRRow, tnZColumn, tPhiSlice, symmetry, h2, tempRatioZ, coefficient1, coefficient2, coefficient3, coefficient4);
    } // end pre smoothing

    // 2) Residue calculation
    residue3D(tvResidue[index], tvArrayV[index], tvCharge[index], tnRRow, tnZColumn, tPhiSlice, symmetry, ih2, tempRatioZ, coefficient1, coefficient2, coefficient3, inverseCoefficient4);

    iOne *= 2;
    jOne *= 2;
    kOne *= 2;
    tnRRow = iOne == 1 ? mParamGrid.NRVertices : mParamGrid.NRVertices / iOne + 1;
    tnZColumn = jOne == 1 ? mParamGrid.NZVertices : mParamGrid.NZVertices / jOne + 1;
    tPhiSlice = mParamGrid.NPhiVertices / kOne;
    tPhiSlice = tPhiSlice < nnPhi ? nnPhi : tPhiSlice;

    // 3) Restriction
    restrict3D(tvCharge[count], tvResidue[index], tnRRow, tnZColumn, tPhiSlice, otPhiSlice);

    // 4) Zeroing coarser V
    std::fill(tvArrayV[count].begin(), tvArrayV[count].end(), 0);
  }

  // coarsest grid
  const DataT hTmp = gridSpacingR * iOne;
  const DataT h2Tmp = hTmp * hTmp;
  const DataT tempGridSizePhiInvTmp = tPhiSlice * getGridSizePhiInv();                 // phi now is multiGrid
  const DataT tempRatioPhiTmp = h2Tmp * tempGridSizePhiInvTmp * tempGridSizePhiInvTmp; // ratio_{phi} = gridSize_{r} / gridSize_{phi}
  const DataT tempRatioZTmp = ratioZ * iOne * iOne / (jOne * jOne);

  calcCoefficients(1, tnRRow - 1, hTmp, tempRatioZTmp, tempRatioPhiTmp, coefficient1, coefficient2, coefficient3, coefficient4);

  // 3) Relax on the coarsest grid
  relax3D(tvArrayV[gridTo - 1], tvCharge[gridTo - 1], tnRRow, tnZColumn, tPhiSlice, symmetry, h2Tmp, tempRatioZTmp, coefficient1, coefficient2, coefficient3, coefficient4);
  // back to fine
  for (int count = gridTo - 1; count >= gridFrom; --count) {
    const int index = count - 1;
    const int otPhiSlice = tPhiSlice;
    iOne /= 2;
    jOne /= 2;
    kOne /= 2;

    tnRRow = iOne == 1 ? mParamGrid.NRVertices : mParamGrid.NRVertices / iOne + 1;
    tnZColumn = jOne == 1 ? mParamGrid.NZVertices : mParamGrid.NZVertices / jOne + 1;
    tPhiSlice = kOne == 1 ? mParamGrid.NPhiVertices : mParamGrid.NPhiVertices / kOne;
    tPhiSlice = tPhiSlice < nnPhi ? nnPhi : tPhiSlice;

    const DataT h = gridSpacingR * iOne;
    const DataT h2 = h * h;
    const DataT tempGridSizePhiInv = tPhiSlice * getGridSizePhiInv();
    const DataT tempRatioPhi = h2 * tempGridSizePhiInv * tempGridSizePhiInv; // ratio_{phi} = gridSize_{r} / gridSize_{phi}
    const DataT tempRatioZ = ratioZ * iOne * iOne / (jOne * jOne);

    // 4) Interpolation/Prolongation
    addInterp3D(tvArrayV[index], tvArrayV[count], tnRRow, tnZColumn, tPhiSlice, otPhiSlice);

    calcCoefficients(1, tnRRow - 1, h, tempRatioZ, tempRatioPhi, coefficient1, coefficient2, coefficient3, coefficient4);

    // 5) Post-Smoothing: Gauss-Seidel Relaxation
    for (int jPost = 1; jPost <= nPost; ++jPost) {
      relax3D(tvArrayV[index], tvCharge[index], tnRRow, tnZColumn, tPhiSlice, symmetry, h2, tempRatioZ, coefficient1, coefficient2, coefficient3, coefficient4);
    }
  }
}

template <typename DataT>
void PoissonSolver<DataT>::residue2D(Vector& residue, const Vector& matricesCurrentV, const Vector& matricesCurrentCharge, const int tnRRow, const int tnZColumn, const DataT ih2, const DataT inverseTempFourth,
                                     const DataT tempRatio, std::vector<DataT>& coefficient1, std::vector<DataT>& coefficient2)
{
  const int iPhi = 0;
#pragma omp parallel for num_threads(sNThreads)
  for (int i = 1; i < tnRRow - 1; ++i) {
    for (int j = 1; j < tnZColumn - 1; ++j) {
      residue(i, j, iPhi) = ih2 * (coefficient1[i] * matricesCurrentV(i + 1, j, iPhi) + coefficient2[i] * matricesCurrentV(i - 1, j, iPhi) + tempRatio * (matricesCurrentV(i, j + 1, iPhi) + matricesCurrentV(i, j - 1, iPhi)) - inverseTempFourth * matricesCurrentV(i, j, iPhi)) + matricesCurrentCharge(i, j, iPhi);
    } // end cols
  }   // end nRRow

  // Boundary points.
  for (int i = 0; i < tnRRow; ++i) {
    residue(i, 0, iPhi) = residue(i, tnZColumn - 1, iPhi) = 0.0;
  }

  for (int j = 0; j < tnZColumn; ++j) {
    residue(0, j, iPhi) = residue(tnRRow - 1, j, iPhi) = 0.0;
  }
}

template <typename DataT>
void PoissonSolver<DataT>::residue3D(Vector& residue, const Vector& matricesCurrentV, const Vector& matricesCurrentCharge, const int tnRRow, const int tnZColumn, const int tnPhi, const int symmetry,
                                     const DataT ih2, const DataT tempRatioZ, const std::vector<DataT>& coefficient1, const std::vector<DataT>& coefficient2, const std::vector<DataT>& coefficient3, const std::vector<DataT>& inverseCoefficient4) const
{
#pragma omp parallel for num_threads(sNThreads) // parallising this loop is possible - but using more than 2 cores makes it slower -
  for (int m = 0; m < tnPhi; ++m) {
    int mp1 = m + 1;
    int signPlus = 1;
    int mm1 = m - 1;
    int signMinus = 1;

    // Reflection symmetry in phi (e.g. symmetry at sector boundaries, or half sectors, etc.)
    if (symmetry == 1) {
      if (mp1 > tnPhi - 1) {
        mp1 = tnPhi - 2;
      }
      if (mm1 < 0) {
        mm1 = 1;
      }
    }
    // Anti-symmetry in phi
    else if (symmetry == -1) {
      if (mp1 > tnPhi - 1) {
        mp1 = tnPhi - 2;
        signPlus = -1;
      }
      if (mm1 < 0) {
        mm1 = 1;
        signMinus = -1;
      }
    } else { // No Symmetries in phi, no boundaries, the calculation is continuous across all phi
      if (mp1 > tnPhi - 1) {
        mp1 = m + 1 - tnPhi;
      }
      if (mm1 < 0) {
        mm1 = m - 1 + tnPhi;
      }
    }

    for (int j = 1; j < tnZColumn - 1; ++j) {
      for (int i = 1; i < tnRRow - 1; ++i) {
        residue(i, j, m) = ih2 * (coefficient2[i] * matricesCurrentV(i - 1, j, m) + tempRatioZ * (matricesCurrentV(i, j - 1, m) + matricesCurrentV(i, j + 1, m)) + coefficient1[i] * matricesCurrentV(i + 1, j, m) +
                                  coefficient3[i] * (signPlus * matricesCurrentV(i, j, mp1) + signMinus * matricesCurrentV(i, j, mm1)) - inverseCoefficient4[i] * matricesCurrentV(i, j, m)) +
                           matricesCurrentCharge(i, j, m);
      } // end cols
    }   // end mParamGrid.NRVertices
  }
}

template <typename DataT>
void PoissonSolver<DataT>::interp3D(Vector& matricesCurrentV, const Vector& matricesCurrentVC, const int tnRRow, const int tnZColumn, const int newPhiSlice, const int oldPhiSlice) const
{
  // Do restrict 2 D for each slice
  if (newPhiSlice == 2 * oldPhiSlice) {
    for (int m = 0; m < newPhiSlice; m += 2) {
      // assuming no symmetry
      int mm = m / 2;
      int mmPlus = mm + 1;
      int mp1 = m + 1;

      // round
      if (mmPlus > oldPhiSlice - 1) {
        mmPlus = mm + 1 - oldPhiSlice;
      }
      if (mp1 > newPhiSlice - 1) {
        mp1 = m + 1 - newPhiSlice;
      }

      for (int j = 2; j < tnZColumn - 1; j += 2) {
        for (int i = 2; i < tnRRow - 1; i += 2) {
          const int iHalf = i / 2;
          const int jHalf = j / 2;
          matricesCurrentV(i, j, m) = matricesCurrentVC(iHalf, jHalf, mm);
          // point on corner lines at phi direction
          matricesCurrentV(i, j, mp1) = (matricesCurrentVC(iHalf, jHalf, mm) + matricesCurrentVC(iHalf, jHalf, mmPlus)) / 2;
        }
      }

      for (int j = 1; j < tnZColumn - 1; j += 2) {
        for (int i = 2; i < tnRRow - 1; i += 2) {
          const int iHalf = i / 2;
          const int jHalf = j / 2;
          matricesCurrentV(i, j, m) = (matricesCurrentVC(iHalf, jHalf, mm) + matricesCurrentVC(iHalf, jHalf + 1, mm)) / 2;
          // point on corner lines at phi direction
          matricesCurrentV(i, j, mp1) = (matricesCurrentVC(iHalf, jHalf, mm) + matricesCurrentVC(iHalf, jHalf + 1, mm) + matricesCurrentVC(iHalf, jHalf, mmPlus) + matricesCurrentVC(iHalf, jHalf + 1, mmPlus)) / 4;
        }
      }

      for (int j = 2; j < tnZColumn - 1; j += 2) {
        for (int i = 1; i < tnRRow - 1; i += 2) {
          const int iHalf = i / 2;
          const int jHalf = j / 2;
          matricesCurrentV(i, j, m) = (matricesCurrentVC(iHalf, jHalf, mm) + matricesCurrentVC(iHalf + 1, jHalf, mm)) / 2;
          // point on line at phi direction
          matricesCurrentV(i, j, mp1) = ((matricesCurrentVC(iHalf, jHalf, mm) + matricesCurrentVC(iHalf, jHalf, mmPlus)) + (matricesCurrentVC(iHalf + 1, jHalf, mmPlus) + matricesCurrentVC(iHalf + 1, jHalf, mm))) / 4;
        }
      }

      for (int j = 1; j < tnZColumn - 1; j += 2) {
        for (int i = 1; i < tnRRow - 1; i += 2) {
          const int iHalf = i / 2;
          const int jHalf = j / 2;
          matricesCurrentV(i, j, m) = ((matricesCurrentVC(iHalf, jHalf, mm) + matricesCurrentVC(iHalf, jHalf + 1, mm)) + (matricesCurrentVC(iHalf + 1, jHalf, mm) + matricesCurrentVC(iHalf + 1, jHalf + 1, mm))) / 4;
          // point at the center at phi direction
          matricesCurrentV(i, j, mp1) = ((matricesCurrentVC(iHalf, jHalf, mm) + matricesCurrentVC(iHalf, jHalf + 1, mm) + matricesCurrentVC(iHalf, jHalf, mmPlus) + matricesCurrentVC(iHalf, jHalf + 1, mmPlus)) +
                                         (matricesCurrentVC(iHalf + 1, jHalf, mm) + matricesCurrentVC(iHalf + 1, jHalf + 1, mm) + matricesCurrentVC(iHalf + 1, jHalf, mmPlus) + matricesCurrentVC(iHalf + 1, jHalf + 1, mmPlus))) /
                                        8;
        }
      }
    }

  } else {
#pragma omp parallel for num_threads(sNThreads) // no change
    for (int m = 0; m < newPhiSlice; ++m) {
      interp2D(matricesCurrentV, matricesCurrentVC, tnRRow, tnZColumn, m);
    }
  }
}

template <typename DataT>
void PoissonSolver<DataT>::interp2D(Vector& matricesCurrentV, const Vector& matricesCurrentVC, const int tnRRow, const int tnZColumn, const int iphi) const
{
  for (int j = 2; j < tnZColumn - 1; j += 2) {
    for (int i = 2; i < tnRRow - 1; i += 2) {
      const int jHalf = j / 2;
      matricesCurrentV(i, j, iphi) = matricesCurrentVC(i / 2, jHalf, iphi);
    }
  }

  for (int j = 1; j < tnZColumn - 1; j += 2) {
    for (int i = 2; i < tnRRow - 1; i += 2) {
      const int iHalf = i / 2;
      const int jHalf = j / 2;
      matricesCurrentV(i, j, iphi) = (matricesCurrentVC(iHalf, jHalf, iphi) + matricesCurrentVC(iHalf, jHalf + 1, iphi)) / 2;
    }
  }

  for (int j = 2; j < tnZColumn - 1; j += 2) {
    for (int i = 1; i < tnRRow - 1; i += 2) {
      const int iHalf = i / 2;
      const int jHalf = j / 2;
      matricesCurrentV(i, j, iphi) = (matricesCurrentVC(iHalf, jHalf, iphi) + matricesCurrentVC(iHalf + 1, jHalf, iphi)) / 2;
    }
  }

  // only if full
  if (MGParameters::gtType == GridTransferType::Full) {
    for (int j = 1; j < tnZColumn - 1; j += 2) {
      for (int i = 1; i < tnRRow - 1; i += 2) {
        const int iHalf = i / 2;
        const int jHalf = j / 2;
        matricesCurrentV(i, j, iphi) = (matricesCurrentVC(iHalf, jHalf, iphi) + matricesCurrentVC(iHalf, jHalf + 1, iphi) + matricesCurrentVC(iHalf + 1, jHalf, iphi) + matricesCurrentVC(iHalf + 1, jHalf + 1, iphi)) / 4;
      }
    }
  }
}

template <typename DataT>
void PoissonSolver<DataT>::addInterp3D(Vector& matricesCurrentV, const Vector& matricesCurrentVC, const int tnRRow, const int tnZColumn, const int newPhiSlice, const int oldPhiSlice) const
{
  // Do restrict 2 D for each slice
  if (newPhiSlice == 2 * oldPhiSlice) {
    for (int m = 0; m < newPhiSlice; m += 2) {
      // assuming no symmetry
      int mm = m / 2;
      int mmPlus = mm + 1;
      int mp1 = m + 1;

      // round
      if (mmPlus > (oldPhiSlice)-1) {
        mmPlus = mm + 1 - (oldPhiSlice);
      }
      if (mp1 > (newPhiSlice)-1) {
        mp1 = m + 1 - (newPhiSlice);
      }

      for (int j = 2; j < tnZColumn - 1; j += 2) {
        for (int i = 2; i < tnRRow - 1; i += 2) {
          const int iHalf = i / 2;
          const int jHalf = j / 2;
          matricesCurrentV(i, j, m) += matricesCurrentVC(iHalf, jHalf, mm);
          // point on corner lines at phi direction
          matricesCurrentV(i, j, mp1) += (matricesCurrentVC(iHalf, jHalf, mm) + matricesCurrentVC(iHalf, jHalf, mmPlus)) / 2;
        }
      }

      for (int j = 1; j < tnZColumn - 1; j += 2) {
        for (int i = 2; i < tnRRow - 1; i += 2) {
          const int iHalf = i / 2;
          const int jHalf = j / 2;
          matricesCurrentV(i, j, m) += (matricesCurrentVC(iHalf, jHalf, mm) + matricesCurrentVC(iHalf, jHalf + 1, mm)) / 2;
          // point on corner lines at phi direction
          matricesCurrentV(i, j, mp1) += (matricesCurrentVC(iHalf, jHalf, mm) + matricesCurrentVC(iHalf, jHalf + 1, mm) + matricesCurrentVC(iHalf, jHalf, mmPlus) + matricesCurrentVC(iHalf, jHalf + 1, mmPlus)) / 4;
        }
      }

      for (int j = 2; j < tnZColumn - 1; j += 2) {
        for (int i = 1; i < tnRRow - 1; i += 2) {
          const int iHalf = i / 2;
          const int jHalf = j / 2;
          matricesCurrentV(i, j, m) += (matricesCurrentVC(iHalf, jHalf, mm) + matricesCurrentVC(iHalf + 1, jHalf, mm)) / 2;
          // point on line at phi direction
          matricesCurrentV(i, j, mp1) += ((matricesCurrentVC(iHalf, jHalf, mm) + matricesCurrentVC(iHalf, jHalf, mmPlus)) + (matricesCurrentVC(iHalf + 1, jHalf, mmPlus) + matricesCurrentVC(iHalf + 1, jHalf, mm))) / 4;
        }
      }

      for (int j = 1; j < tnZColumn - 1; j += 2) {
        for (int i = 1; i < tnRRow - 1; i += 2) {
          const int iHalf = i / 2;
          const int jHalf = j / 2;
          matricesCurrentV(i, j, m) += ((matricesCurrentVC(iHalf, jHalf, mm) + matricesCurrentVC(iHalf, jHalf + 1, mm)) + (matricesCurrentVC(iHalf + 1, jHalf, mm) + matricesCurrentVC(iHalf + 1, jHalf + 1, mm))) / 4;
          // point at the center at phi direction
          matricesCurrentV(i, j, mp1) += ((matricesCurrentVC(iHalf, jHalf, mm) + matricesCurrentVC(iHalf, jHalf + 1, mm) + matricesCurrentVC(iHalf, jHalf, mmPlus) + matricesCurrentVC(iHalf, jHalf + 1, mmPlus)) +
                                          (matricesCurrentVC(iHalf + 1, jHalf, mm) + matricesCurrentVC(iHalf + 1, jHalf + 1, mm) + matricesCurrentVC(iHalf + 1, jHalf, mmPlus) + matricesCurrentVC(iHalf + 1, jHalf + 1, mmPlus))) /
                                         8;
        }
      }
    }

  } else {
#pragma omp parallel for num_threads(sNThreads) // no change
    for (int m = 0; m < newPhiSlice; m++) {
      addInterp2D(matricesCurrentV, matricesCurrentVC, tnRRow, tnZColumn, m);
    }
  }
}

template <typename DataT>
void PoissonSolver<DataT>::addInterp2D(Vector& matricesCurrentV, const Vector& matricesCurrentVC, const int tnRRow, const int tnZColumn, const int tnPhi) const
{
  for (int j = 2; j < tnZColumn - 1; j += 2) {
    for (int i = 2; i < tnRRow - 1; i += 2) {
      matricesCurrentV(i, j, tnPhi) = matricesCurrentV(i, j, tnPhi) + matricesCurrentVC(i / 2, j / 2, tnPhi);
    }
  }

  for (int j = 1; j < tnZColumn - 1; j += 2) {
    for (int i = 2; i < tnRRow - 1; i += 2) {
      const int iHalf = i / 2;
      const int jHalf = j / 2;
      matricesCurrentV(i, j, tnPhi) = matricesCurrentV(i, j, tnPhi) + (matricesCurrentVC(iHalf, jHalf, tnPhi) + matricesCurrentVC(iHalf, jHalf + 1, tnPhi)) / 2;
    }
  }

  for (int j = 2; j < tnZColumn - 1; j += 2) {
    for (int i = 1; i < tnRRow - 1; i += 2) {
      const int iHalf = i / 2;
      const int jHalf = j / 2;
      matricesCurrentV(i, j, tnPhi) = matricesCurrentV(i, j, tnPhi) + (matricesCurrentVC(iHalf, jHalf, tnPhi) + matricesCurrentVC(iHalf + 1, jHalf, tnPhi)) / 2;
    }
  }

  // only if full
  if (MGParameters::gtType == GridTransferType::Full) {
    for (int j = 1; j < tnZColumn - 1; j += 2) {
      for (int i = 1; i < tnRRow - 1; i += 2) {
        const int iHalf = i / 2;
        const int jHalf = j / 2;
        matricesCurrentV(i, j, tnPhi) = matricesCurrentV(i, j, tnPhi) + (matricesCurrentVC(iHalf, jHalf, tnPhi) + matricesCurrentVC(iHalf, jHalf + 1, tnPhi) + matricesCurrentVC(iHalf + 1, jHalf, tnPhi) + matricesCurrentVC(iHalf + 1, jHalf + 1, tnPhi)) / 4;
      }
    }
  }
}

template <typename DataT>
void PoissonSolver<DataT>::relax3D(Vector& matricesCurrentV, const Vector& matricesCurrentCharge, const int tnRRow, const int tnZColumn, const int iPhi, const int symmetry, const DataT h2,
                                   const DataT tempRatioZ, const std::vector<DataT>& coefficient1, const std::vector<DataT>& coefficient2, const std::vector<DataT>& coefficient3, const std::vector<DataT>& coefficient4) const
{
  // Gauss-Seidel (Read Black}
  if (MGParameters::relaxType == RelaxType::GaussSeidel) {
    // for each slice
    for (int iPass = 1; iPass <= 2; ++iPass) {
      const int msw = (iPass % 2) ? 1 : 2;
      for (int m = 0; m < iPhi; ++m) {
        const int jsw = ((msw + m) % 2) ? 1 : 2;
        int mp1 = m + 1;
        int signPlus = 1;
        int mm1 = m - 1;
        int signMinus = 1;
        // Reflection symmetry in phi (e.g. symmetry at sector boundaries, or half sectors, etc.)
        if (symmetry == 1) {
          if (mp1 > iPhi - 1) {
            mp1 = iPhi - 2;
          }
          if (mm1 < 0) {
            mm1 = 1;
          }
        }
        // Anti-symmetry in phi
        else if (symmetry == -1) {
          if (mp1 > iPhi - 1) {
            mp1 = iPhi - 2;
            signPlus = -1;
          }
          if (mm1 < 0) {
            mm1 = 1;
            signMinus = -1;
          }
        } else { // No Symmetries in phi, no boundaries, the calculation is continuous across all phi
          if (mp1 > iPhi - 1) {
            mp1 = m + 1 - iPhi;
          }
          if (mm1 < 0) {
            mm1 = m - 1 + iPhi;
          }
        }
        int isw = jsw;
        for (int j = 1; j < tnZColumn - 1; ++j, isw = 3 - isw) {
          for (int i = isw; i < tnRRow - 1; i += 2) {
            (matricesCurrentV)(i, j, m) = (coefficient2[i] * (matricesCurrentV)(i - 1, j, m) + tempRatioZ * ((matricesCurrentV)(i, j - 1, m) + (matricesCurrentV)(i, j + 1, m)) + coefficient1[i] * (matricesCurrentV)(i + 1, j, m) + coefficient3[i] * (signPlus * (matricesCurrentV)(i, j, mp1) + signMinus * (matricesCurrentV)(i, j, mm1)) + (h2 * (matricesCurrentCharge)(i, j, m))) * coefficient4[i];
          } // end cols
        }   // end mParamGrid.NRVertices
      }     // end phi
    }       // end sweep
  } else if (MGParameters::relaxType == RelaxType::Jacobi) {
    // for each slice
    for (int m = 0; m < iPhi; ++m) {
      int mp1 = m + 1;
      int signPlus = 1;
      int mm1 = m - 1;
      int signMinus = 1;

      // Reflection symmetry in phi (e.g. symmetry at sector boundaries, or half sectors, etc.)
      if (symmetry == 1) {
        if (mp1 > iPhi - 1) {
          mp1 = iPhi - 2;
        }
        if (mm1 < 0) {
          mm1 = 1;
        }
      }
      // Anti-symmetry in phi
      else if (symmetry == -1) {
        if (mp1 > iPhi - 1) {
          mp1 = iPhi - 2;
          signPlus = -1;
        }
        if (mm1 < 0) {
          mm1 = 1;
          signMinus = -1;
        }
      } else { // No Symmetries in phi, no boundaries, the calculation is continuous across all phi
        if (mp1 > iPhi - 1) {
          mp1 = m + 1 - iPhi;
        }
        if (mm1 < 0) {
          mm1 = m - 1 + iPhi;
        }
      }
      // Jacobian
      for (int j = 1; j < tnZColumn - 1; ++j) {
        for (int i = 1; i < tnRRow - 1; ++i) {
          (matricesCurrentV)(i, j, m) = (coefficient2[i] * (matricesCurrentV)(i - 1, j, m) + tempRatioZ * ((matricesCurrentV)(i, j - 1, m) + (matricesCurrentV)(i, j + 1, m)) + coefficient1[i] * (matricesCurrentV)(i + 1, j, m) + coefficient3[i] * (signPlus * (matricesCurrentV)(i, j, mp1) + signMinus * (matricesCurrentV)(i, j, mm1)) + (h2 * (matricesCurrentCharge)(i, j, m))) * coefficient4[i];
        } // end cols
      }   // end mParamGrid.NRVertices
    }     // end phi
  } else {
    // Case weighted Jacobi
    // TODO
  }
}

template <typename DataT>
void PoissonSolver<DataT>::relax2D(Vector& matricesCurrentV, const Vector& matricesCurrentCharge, const int tnRRow, const int tnZColumn, const DataT h2, const DataT tempFourth, const DataT tempRatio,
                                   std::vector<DataT>& coefficient1, std::vector<DataT>& coefficient2)
{
  // Gauss-Seidel
  const int iPhi = 0;
  if (MGParameters::relaxType == RelaxType::GaussSeidel) {
    int jsw = 1;
    for (int iPass = 1; iPass <= 2; ++iPass, jsw = 3 - jsw) {
      int isw = jsw;
      for (int j = 1; j < tnZColumn - 1; ++j, isw = 3 - isw) {
        for (int i = isw; i < tnRRow - 1; i += 2) {
          matricesCurrentV(i, j, iPhi) = tempFourth * (coefficient1[i] * matricesCurrentV(i + 1, j, iPhi) + coefficient2[i] * matricesCurrentV(i - 1, j, iPhi) +
                                                       tempRatio * (matricesCurrentV(i, j + 1, iPhi) + matricesCurrentV(i, j - 1, iPhi)) + (h2 * matricesCurrentCharge(i, j, iPhi)));
        } // end cols
      }   // end mParamGrid.NRVertices
    }     // end pass red-black
  } else if (MGParameters::relaxType == RelaxType::Jacobi) {
    for (int j = 1; j < tnZColumn - 1; ++j) {
      for (int i = 1; i < tnRRow - 1; ++i) {
        matricesCurrentV(i, j, iPhi) = tempFourth * (coefficient1[i] * matricesCurrentV(i + 1, j, iPhi) + coefficient2[i] * matricesCurrentV(i - 1, j, iPhi) +
                                                     tempRatio * (matricesCurrentV(i, j + 1, iPhi) + matricesCurrentV(i, j - 1, iPhi)) + (h2 * matricesCurrentCharge(i, j, iPhi)));
      } // end cols
    }   // end mParamGrid.NRVertices
  } else if (MGParameters::relaxType == RelaxType::WeightedJacobi) {
    // Weighted Jacobi
    // TODO
  }
}

template <typename DataT>
void PoissonSolver<DataT>::restrictBoundary3D(Vector& matricesCurrentCharge, const Vector& residue, const int tnRRow, const int tnZColumn, const int newPhiSlice, const int oldPhiSlice) const
{
  // in case of full 3d and the mParamGrid.NPhiVertices is also coarsening
  if (2 * newPhiSlice == oldPhiSlice) {
    for (int m = 0, mm = 0; m < newPhiSlice; ++m, mm += 2) {
      // for boundary
      for (int j = 0, jj = 0; j < tnZColumn; ++j, jj += 2) {
        matricesCurrentCharge(0, j, m) = residue(0, jj, mm);
        matricesCurrentCharge(tnRRow - 1, j, m) = residue((tnRRow - 1) * 2, jj, mm);
      }

      // for boundary
      for (int i = 0, ii = 0; i < tnRRow; ++i, ii += 2) {
        matricesCurrentCharge(i, 0, m) = residue(ii, 0, mm);
        matricesCurrentCharge(i, tnZColumn - 1, m) = residue(ii, (tnZColumn - 1) * 2, mm);
      }
    } // end phis
  } else {
    for (int m = 0; m < newPhiSlice; ++m) {
      restrictBoundary2D(matricesCurrentCharge, residue, tnRRow, tnZColumn, m);
    }
  }
}

template <typename DataT>
void PoissonSolver<DataT>::restrictBoundary2D(Vector& matricesCurrentCharge, const Vector& residue, const int tnRRow, const int tnZColumn, const int tnPhi) const
{
  // for boundary
  for (int j = 0, jj = 0; j < tnZColumn; ++j, jj += 2) {
    matricesCurrentCharge(0, j, tnPhi) = residue(0, jj, tnPhi);
    matricesCurrentCharge(tnRRow - 1, j, tnPhi) = residue((tnRRow - 1) * 2, jj, tnPhi);
  }

  // for boundary
  for (int i = 0, ii = 0; i < tnRRow; ++i, ii += 2) {
    matricesCurrentCharge(i, 0, tnPhi) = residue(ii, 0, tnPhi);
    matricesCurrentCharge(i, tnZColumn - 1, tnPhi) = residue(ii, (tnZColumn - 1) * 2, tnPhi);
  }
}

template <typename DataT>
void PoissonSolver<DataT>::restrict3D(Vector& matricesCurrentCharge, const Vector& residue, const int tnRRow, const int tnZColumn, const int newPhiSlice, const int oldPhiSlice) const
{
  if (2 * newPhiSlice == oldPhiSlice) {
    int mm = 0;
    for (int m = 0; m < newPhiSlice; m++, mm += 2) {
      // assuming no symmetry
      int mp1 = mm + 1;
      int mm1 = mm - 1;

      if (mp1 > (oldPhiSlice)-1) {
        mp1 = mm + 1 - (oldPhiSlice);
      }
      if (mm1 < 0) {
        mm1 = mm - 1 + (oldPhiSlice);
      }

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
      }   // end mParamGrid.NRVertices

      // for boundary
      for (int j = 0, jj = 0; j < tnZColumn; ++j, jj += 2) {
        matricesCurrentCharge(0, j, m) = residue(0, jj, mm);
        matricesCurrentCharge(tnRRow - 1, j, m) = residue((tnRRow - 1) * 2, jj, mm);
      }

      // for boundary
      for (int i = 0, ii = 0; i < tnRRow; ++i, ii += 2) {
        matricesCurrentCharge(i, 0, m) = residue(ii, 0, mm);
        matricesCurrentCharge(i, tnZColumn - 1, m) = residue(ii, (tnZColumn - 1) * 2, mm);
      }
    } // end phis

  } else {
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
      if (MGParameters::gtType == GridTransferType::Half) {
        // half
        matricesCurrentCharge(i, j, iphi) = residue(ii, jj, iphi) / 2 + (residue(iip1, jj, iphi) + residue(iim1, jj, iphi) + residue(ii, jjp1, iphi) + residue(ii, jjm1, iphi)) / 8;
      } else if (MGParameters::gtType == GridTransferType::Full) {
        matricesCurrentCharge(i, j, iphi) = residue(ii, jj, iphi) / 4 + (residue(iip1, jj, iphi) + residue(iim1, jj, iphi) + residue(ii, jjp1, iphi) + residue(ii, jjm1, iphi)) / 8 +
                                            (residue(iip1, jjp1, iphi) + residue(iim1, jjp1, iphi) + residue(iip1, jjm1, iphi) + residue(iim1, jjm1, iphi)) / 16;
      }
    } // end cols
  }   // end mParamGrid.NRVertices
  // boundary
  // for boundary
  for (int j = 0, jj = 0; j < tnZColumn; ++j, jj += 2) {
    matricesCurrentCharge(0, j, iphi) = residue(0, jj, iphi);
    matricesCurrentCharge(tnRRow - 1, j, iphi) = residue((tnRRow - 1) * 2, jj, iphi);
  }
  // for boundary
  for (int i = 0, ii = 0; i < tnRRow; ++i, ii += 2) {
    matricesCurrentCharge(i, 0, iphi) = residue(ii, 0, iphi);
    matricesCurrentCharge(i, tnZColumn - 1, iphi) = residue(ii, (tnZColumn - 1) * 2, iphi);
  }
}

template <typename DataT>
DataT PoissonSolver<DataT>::getConvergenceError(const Vector& matricesCurrentV, Vector& prevArrayV) const
{
  std::vector<DataT> errorArr(prevArrayV.getNphi());

  // subtract the two matrices
  std::transform(prevArrayV.begin(), prevArrayV.end(), matricesCurrentV.begin(), prevArrayV.begin(), std::minus<DataT>());

#pragma omp parallel for num_threads(sNThreads) // parallising this loop is possible - but using more than 2 cores makes it slower -
  for (unsigned int m = 0; m < prevArrayV.getNphi(); ++m) {
    // square each entry in the vector and sum them up
    const auto phiStep = prevArrayV.getNr() * prevArrayV.getNz(); // number of points in one phi slice
    const auto start = prevArrayV.begin() + m * phiStep;
    const auto end = start + phiStep;
    errorArr[m] = std::inner_product(start, end, start, DataT(0)); // inner product "Sum (matrix[a]*matrix[a])"
  }
  // return largest error
  return *std::max_element(std::begin(errorArr), std::end(errorArr));
}

template <typename DataT>
void PoissonSolver<DataT>::calcCoefficients(unsigned int from, unsigned int to, const DataT h, const DataT tempRatioZ, const DataT tempRatioPhi, std::vector<DataT>& coefficient1, std::vector<DataT>& coefficient2, std::vector<DataT>& coefficient3, std::vector<DataT>& coefficient4) const
{
  for (unsigned int i = from; i < to; ++i) {
    const DataT radiusInv = 1 / (TPCParameters<DataT>::IFCRADIUS + i * h);
    const DataT hRadiusTmp = h * radiusInv / 2;
    coefficient1[i] = 1 + hRadiusTmp;
    coefficient2[i] = 1 - hRadiusTmp;
    coefficient3[i] = tempRatioPhi * radiusInv * radiusInv;
    coefficient4[i] = 1 / (2 * (1 + tempRatioZ + coefficient3[i]));
  }
}

template <typename DataT>
void PoissonSolver<DataT>::calcCoefficients2D(unsigned int from, unsigned int to, const DataT h, std::vector<DataT>& coefficient1, std::vector<DataT>& coefficient2) const
{
  for (int i = from; i < to; ++i) {
    DataT radiusInvHalf = h / (2 * (TPCParameters<DataT>::IFCRADIUS + i * h));
    coefficient1[i] = 1 + radiusInvHalf;
    coefficient2[i] = 1 - radiusInvHalf;
  }
}

template <typename DataT>
DataT PoissonSolver<DataT>::getGridSizePhiInv()
{
  return MGParameters::normalizeGridToOneSector ? (INVTWOPI * SECTORSPERSIDE) : INVTWOPI;
}

template class o2::tpc::PoissonSolver<double>;
template class o2::tpc::PoissonSolver<float>;
