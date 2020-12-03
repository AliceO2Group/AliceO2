// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
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
#include "Framework/Logger.h"
#include <numeric>
#include <fmt/core.h>

#ifdef WITH_OPENMP
#include <omp.h>
#endif

using namespace o2::tpc;

template <typename DataT, size_t Nz, size_t Nr, size_t Nphi>
void PoissonSolver<DataT, Nz, Nr, Nphi>::poissonSolver3D(DataContainer& matricesV, const DataContainer& matricesCharge, const int symmetry)
{
  if (MGParameters::isFull3D) {
    poissonMultiGrid3D(matricesV, matricesCharge, symmetry);
  } else {
    poissonMultiGrid3D2D(matricesV, matricesCharge, symmetry);
  }
}

template <typename DataT, size_t Nz, size_t Nr, size_t Nphi>
void PoissonSolver<DataT, Nz, Nr, Nphi>::poissonSolver2D(DataContainer& matricesV, const DataContainer& matricesCharge)
{
  poissonMultiGrid2D(matricesV, matricesCharge);
}

template <typename DataT, size_t Nz, size_t Nr, size_t Nphi>
void PoissonSolver<DataT, Nz, Nr, Nphi>::poissonMultiGrid2D(DataContainer& matricesV, const DataContainer& matricesCharge, const int iPhi)
{
  /// Geometry of TPC -- should be use AliTPCParams instead
  const DataT gridSpacingR = getSpacingR();
  const DataT gridSpacingZ = getSpacingZ();
  const DataT ratioZ = gridSpacingR * gridSpacingR / (gridSpacingZ * gridSpacingZ); // ratio_{Z} = gridSize_{r} / gridSize_{z}

  int nGridRow = 0; // number grid
  int nGridCol = 0; // number grid

  int nnRow = Nr;
  while (nnRow >>= 1) {
    ++nGridRow;
  }

  int nnCol = Nz;
  while (nnCol >>= 1) {
    ++nGridCol;
  }

  //Check that number of Nr and Nz is suitable for multi grid
  if (!isPowerOfTwo(Nr - 1)) {
    LOGP(ERROR, "PoissonMultiGrid2D: PoissonMultiGrid - Error in the number of Nr. Must be 2**M + 1");
    return;
  }
  if (!isPowerOfTwo(Nz - 1)) {
    LOGP(ERROR, "PoissonMultiGrid2D: PoissonMultiGrid - Error in the number of Nz. Must be 2**N - 1");
    return;
  }

  const int nLoop = std::max(nGridRow, nGridCol); // Calculate the number of nLoop for the binary expansion

  LOGP(info, "{}", fmt::format("PoissonMultiGrid2D: nGridRow={}, nGridCol={}, nLoop={}, nMGCycle={}", nGridRow, nGridCol, nLoop, MGParameters::nMGCycle));

  int iOne = 1; // in/dex
  int jOne = 1; // index
  int tnRRow = Nr;
  int tnZColumn = Nz;

  // Vector for storing multi grid array
  std::vector<Vector> tvArrayV(nLoop);    // potential <--> error
  std::vector<Vector> tvChargeFMG(nLoop); // charge is restricted in full multiGrid
  std::vector<Vector> tvCharge(nLoop);    // charge <--> residue
  std::vector<Vector> tvResidue(nLoop);   // residue calculation

  // Allocate memory for temporary grid
  for (int count = 1; count <= nLoop; ++count) {
    tnRRow = iOne == 1 ? Nr : Nr / iOne + 1;
    tnZColumn = jOne == 1 ? Nz : Nz / jOne + 1;
    // if one just address to matrixV
    tvResidue[count - 1].resize(tnRRow, tnZColumn, 1);
    tvChargeFMG[count - 1].resize(tnRRow, tnZColumn, 1);
    tvArrayV[count - 1].resize(tnRRow, tnZColumn, 1);
    tvCharge[count - 1].resize(tnRRow, tnZColumn, 1);

    if (count == 1) {
      for (int iphi = iPhi; iphi <= iPhi; ++iphi) {
        for (int ir = 0; ir < Nr; ++ir) {
          for (int iz = 0; iz < Nz; ++iz) {
            tvChargeFMG[count - 1](ir, iz, iphi) = matricesCharge(iz, ir, iphi);
            tvCharge[count - 1](ir, iz, iphi) = matricesCharge(iz, ir, iphi);
            tvArrayV[count - 1](ir, iz, iphi) = matricesV(iz, ir, iphi);
          }
        }
      }
    } else {
      restrict2D(tvChargeFMG[count - 1], tvChargeFMG[count - 2], tnRRow, tnZColumn, 0);
    }
    iOne = 2 * iOne;
    jOne = 2 * jOne;
  }

  /// full multi grid
  if (MGParameters::cycleType == CycleType::FCycle) {

    LOGP(info, "PoissonMultiGrid2D: Do full cycle");
    // FMG
    // 1) Relax on the coarsest grid
    iOne = iOne * 0.5;
    jOne = jOne * 0.5;
    tnRRow = iOne == 1 ? Nr : Nr / iOne + 1;
    tnZColumn = jOne == 1 ? Nz : Nz / jOne + 1;
    const DataT h = gridSpacingR * nLoop;
    const DataT h2 = h * h;
    const DataT tempRatio = ratioZ * iOne * iOne / (jOne * jOne);
    const DataT tempFourth = 1.0 / (2.0 + 2.0 * tempRatio);

    std::vector<DataT> coefficient1(tnRRow);
    std::vector<DataT> coefficient2(tnRRow);
    calcCoefficients2D(1, tnRRow - 1, h, coefficient1, coefficient2);

    relax2D(tvArrayV[nLoop - 1], tvChargeFMG[nLoop - 1], tnRRow, tnZColumn, h2, tempFourth, tempRatio, coefficient1, coefficient2);

    // Do VCycle from nLoop H to h
    for (int count = nLoop - 2; count >= 0; --count) {

      iOne = iOne * 0.5;
      jOne = jOne * 0.5;

      tnRRow = iOne == 1 ? Nr : Nr / iOne + 1;
      tnZColumn = jOne == 1 ? Nz : Nz / jOne + 1;

      interp2D(tvArrayV[count], tvArrayV[count + 1], tnRRow, tnZColumn, iPhi);

      // Copy the relax charge to the tvCharge
      tvCharge[count] = tvChargeFMG[count]; //copy

      // Do V cycle
      for (int mgCycle = 0; mgCycle < MGParameters::nMGCycle; ++mgCycle) {
        vCycle2D(count + 1, nLoop, MGParameters::nPre, MGParameters::nPost, gridSpacingR, ratioZ, tvArrayV, tvCharge, tvResidue);
      }
    }
  } else if (MGParameters::cycleType == CycleType::VCycle) {
    // 2. VCycle
    LOGP(info, "PoissonMultiGrid2D: Do V cycle");

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
    for (int ir = 0; ir < Nr; ++ir) {
      for (int iz = 0; iz < Nz; ++iz) {
        matricesV(iz, ir, iphi) = tvArrayV[0](ir, iz, iphi);
      }
    }
  }
}

template <typename DataT, size_t Nz, size_t Nr, size_t Nphi>
void PoissonSolver<DataT, Nz, Nr, Nphi>::poissonMultiGrid3D2D(DataContainer& matricesV, const DataContainer& matricesCharge, const int symmetry)
{
  LOGP(info, "{}", fmt::format("PoissonMultiGrid3D2D: in Poisson Solver 3D multiGrid semi coarsening Nr={}, cols={}, Nphi={}", Nz, Nr, Nphi));

  // Check that the number of Nr and Nz is suitable for a binary expansion
  if (!isPowerOfTwo((Nr - 1))) {
    LOGP(ERROR, "PoissonMultiGrid3D2D: Poisson3DMultiGrid - Error in the number of Nr. Must be 2**M + 1");
    return;
  }
  if (!isPowerOfTwo((Nz - 1))) {
    LOGP(ERROR, "PoissonMultiGrid3D2D: Poisson3DMultiGrid - Error in the number of Nz. Must be 2**N - 1");
    return;
  }
  if (Nphi <= 3) {
    LOGP(ERROR, "PoissonMultiGrid3D2D: Poisson3DMultiGrid - Error in the number of Nphi. Must be larger than 3");
    return;
  }
  if (Nphi > 1000) {
    LOGP(ERROR, "PoissonMultiGrid3D2D: Poisson3D  Nphi > 1000 is not allowed (nor wise)");
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
  int nnRow = Nr;
  int nnCol = Nz;

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
    const unsigned int tnRRow = iOne == 1 ? Nr : Nr / iOne + 1;
    const unsigned int tnZColumn = jOne == 1 ? Nz : Nz / jOne + 1;
    tvResidue[count - 1].resize(tnRRow, tnZColumn, Nphi);
    tvPrevArrayV[count - 1].resize(tnRRow, tnZColumn, Nphi);

    // memory for the finest grid is from parameters
    tvChargeFMG[count - 1].resize(tnRRow, tnZColumn, Nphi);
    tvArrayV[count - 1].resize(tnRRow, tnZColumn, Nphi);
    tvCharge[count - 1].resize(tnRRow, tnZColumn, Nphi);

    if (count == 1) {
      for (int iphi = 0; iphi < Nphi; ++iphi) {
        for (int ir = 0; ir < Nr; ++ir) {
          for (int iz = 0; iz < Nz; ++iz) {
            tvChargeFMG[count - 1](ir, iz, iphi) = matricesCharge(iz, ir, iphi);
            tvArrayV[count - 1](ir, iz, iphi) = matricesV(iz, ir, iphi);
          }
        }
      }
    } else {
      restrict3D(tvChargeFMG[count - 1], tvChargeFMG[count - 2], tnRRow, tnZColumn, Nphi, Nphi);
      restrictBoundary3D(tvArrayV[count - 1], tvArrayV[count - 2], tnRRow, tnZColumn, Nphi, Nphi);
    }
    iOne = 2 * iOne; // doubling
    jOne = 2 * jOne; // doubling
  }

  std::array<DataT, Nr> coefficient1{};        // coefficient1(Nr) for storing (1 + h_{r}/2r_{i}) from central differences in r direction
  std::array<DataT, Nr> coefficient2{};        // coefficient2(Nr) for storing (1 + h_{r}/2r_{i}) from central differences in r direction
  std::array<DataT, Nr> coefficient3{};        // coefficient3(Nr) for storing (1/r_{i}^2) from central differences in phi direction
  std::array<DataT, Nr> coefficient4{};        // coefficient4(Nr) for storing  1/2
  std::array<DataT, Nr> inverseCoefficient4{}; // inverse of coefficient4(Nr)

  // Case full multi grid (FMG)
  if (MGParameters::cycleType == CycleType::FCycle) {
    // 1) Relax on the coarsest grid
    iOne = iOne * 0.5;
    jOne = jOne * 0.5;
    int tnRRow = iOne == 1 ? Nr : Nr / iOne + 1;
    int tnZColumn = jOne == 1 ? Nz : Nz / jOne + 1;

    const DataT h = getSpacingR() * iOne;
    const DataT h2 = h * h;
    const DataT iOne2 = iOne * iOne;
    const DataT tempRatioPhi = ratioPhi * iOne2; // Used tobe divided by ( m_one * m_one ) when m_one was != 1
    const DataT tempRatioZ = ratioZ * iOne2 / (jOne * jOne);

    calcCoefficients(1, tnRRow - 1, h, tempRatioZ, tempRatioPhi, coefficient1, coefficient2, coefficient3, coefficient4);

    // relax on the coarsest level
    relax3D(tvArrayV[nLoop - 1], tvChargeFMG[nLoop - 1], tnRRow, tnZColumn, Nphi, symmetry, h2, tempRatioZ, coefficient1, coefficient2, coefficient3, coefficient4);

    // 2) Do multiGrid v-cycle from coarsest to finest
    for (int count = nLoop - 2; count >= 0; --count) {
      // move to finer grid
      iOne = iOne * 0.5;
      jOne = jOne * 0.5;
      tnRRow = iOne == 1 ? Nr : Nr / iOne + 1;
      tnZColumn = jOne == 1 ? Nz : Nz / jOne + 1;

      // 2) a) Interpolate potential for h -> 2h (coarse -> fine)
      interp3D(tvArrayV[count], tvArrayV[count + 1], tnRRow, tnZColumn, Nphi, Nphi);

      // 2) c) Copy the restricted charge to charge for calculation
      tvCharge[count] = tvChargeFMG[count]; //copy

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
  for (int iphi = 0; iphi < Nphi; ++iphi) {
    for (int ir = 0; ir < Nr; ++ir) {
      for (int iz = 0; iz < Nz; ++iz) {
        matricesV(iz, ir, iphi) = tvArrayV[0](ir, iz, iphi);
      }
    }
  }
}

template <typename DataT, size_t Nz, size_t Nr, size_t Nphi>
void PoissonSolver<DataT, Nz, Nr, Nphi>::poissonMultiGrid3D(DataContainer& matricesV, const DataContainer& matricesCharge, const int symmetry)
{
  const DataT gridSpacingR = getSpacingR();
  const DataT gridSpacingZ = getSpacingZ();
  const DataT ratioZ = gridSpacingR * gridSpacingR / (gridSpacingZ * gridSpacingZ); // ratio_{Z} = gridSize_{r} / gridSize_{z}

  LOGP(info, "{}", fmt::format("PoissonMultiGrid3D: in Poisson Solver 3D multi grid full coarsening  Nr={}, cols={}, Nphi={}", Nr, Nz, Nphi));

  // Check that the number of Nr and Nz is suitable for a binary expansion
  if (!isPowerOfTwo((Nr - 1))) {
    LOGP(ERROR, "PoissonMultiGrid3D: Poisson3DMultiGrid - Error in the number of Nr. Must be 2**M + 1");
    return;
  }
  if (!isPowerOfTwo((Nz - 1))) {
    LOGP(ERROR, "PoissonMultiGrid3D: Poisson3DMultiGrid - Error in the number of Nz. Must be 2**N - 1");
    return;
  }
  if (Nphi <= 3) {
    LOGP(ERROR, "PoissonMultiGrid3D: Poisson3DMultiGrid - Error in the number of Nphi. Must be larger than 3");
    return;
  }
  if (Nphi > 1000) {
    LOGP(ERROR, "PoissonMultiGrid3D: Poisson3D  Nphi > 1000 is not allowed (nor wise)");
    return;
  }

  // Solve Poisson's equation in cylindrical coordinates by multi grid technique
  // Allow for different size grid spacing in R and Z directions
  int nGridRow = 0; // number grid
  int nGridCol = 0; // number grid
  int nGridPhi = 0;

  int nnRow = Nr;
  while (nnRow >>= 1) {
    ++nGridRow;
  }

  int nnCol = Nz;
  while (nnCol >>= 1) {
    ++nGridCol;
  }

  int nnPhi = Nphi;
  while (nnPhi % 2 == 0) {
    ++nGridPhi;
    nnPhi *= 0.5;
  }

  LOGP(info, "{}", fmt::format("PoissonMultiGrid3D: nGridRow={}, nGridCol={}, nGridPhi={}", nGridRow, nGridCol, nGridPhi));
  const int nLoop = std::max({nGridRow, nGridCol, nGridPhi}); // Calculate the number of nLoop for the binary expansion

  // Vector for storing multi grid array
  int iOne = 1; // index i in gridSize r (original)
  int jOne = 1; // index j in gridSize z (original)
  int kOne = 1; // index k in gridSize phi
  int tnRRow = Nr;
  int tnZColumn = Nz;
  int tPhiSlice = Nphi;

  // 1)	Memory allocation for multi grid
  std::vector<Vector> tvArrayV(nLoop);     // potential <--> error
  std::vector<Vector> tvChargeFMG(nLoop);  // charge is restricted in full multiGrid
  std::vector<Vector> tvCharge(nLoop);     // charge <--> residue
  std::vector<Vector> tvPrevArrayV(nLoop); // error calculation
  std::vector<Vector> tvResidue(nLoop);    // residue calculation

  std::array<DataT, Nr> coefficient1{};        // coefficient1(Nr) for storing (1 + h_{r}/2r_{i}) from central differences in r direction
  std::array<DataT, Nr> coefficient2{};        // coefficient2(Nr) for storing (1 + h_{r}/2r_{i}) from central differences in r direction
  std::array<DataT, Nr> coefficient3{};        // coefficient3(Nr) for storing (1/r_{i}^2) from central differences in phi direction
  std::array<DataT, Nr> coefficient4{};        // coefficient4(Nr) for storing  1/2
  std::array<DataT, Nr> inverseCoefficient4{}; // inverse of coefficient4(Nr)

  for (int count = 1; count <= nLoop; ++count) {
    // tnRRow,tnZColumn in new grid
    tnRRow = iOne == 1 ? Nr : Nr / iOne + 1;
    tnZColumn = jOne == 1 ? Nz : Nz / jOne + 1;
    tPhiSlice = kOne == 1 ? Nphi : Nphi / kOne;
    tPhiSlice = tPhiSlice < nnPhi ? nnPhi : tPhiSlice;

    // allocate memory for residue
    tvResidue[count - 1].resize(tnRRow, tnZColumn, tPhiSlice);
    tvPrevArrayV[count - 1].resize(tnRRow, tnZColumn, tPhiSlice);
    tvChargeFMG[count - 1].resize(tnRRow, tnZColumn, tPhiSlice);
    tvArrayV[count - 1].resize(tnRRow, tnZColumn, tPhiSlice);
    tvCharge[count - 1].resize(tnRRow, tnZColumn, tPhiSlice);

    // memory for the finest grid is from parameters
    if (count == 1) {
      for (int iphi = 0; iphi < Nphi; ++iphi) {
        for (int ir = 0; ir < Nr; ++ir) {
          for (int iz = 0; iz < Nz; ++iz) {
            tvChargeFMG[count - 1](ir, iz, iphi) = matricesCharge(iz, ir, iphi);
            tvArrayV[count - 1](ir, iz, iphi) = matricesV(iz, ir, iphi);
          }
        }
      }
      tvCharge[count - 1] = tvChargeFMG[count - 1];
    }
    iOne = 2 * iOne; // doubling
    jOne = 2 * jOne; // doubling
    kOne = 2 * kOne;
  }

  // Case full multi grid (FMG)
  if (MGParameters::cycleType == CycleType::FCycle) {
    // Restrict the charge to coarser grid
    iOne = 2;
    jOne = 2;
    kOne = 2;
    int otPhiSlice = Nphi;

    // 1) Restrict Charge and Boundary to coarser grid
    for (int count = 2; count <= nLoop; ++count) {
      // tnRRow,tnZColumn in new grid
      tnRRow = iOne == 1 ? Nr : Nr / iOne + 1;
      tnZColumn = jOne == 1 ? Nz : Nz / jOne + 1;
      tPhiSlice = kOne == 1 ? Nphi : Nphi / kOne;
      tPhiSlice = tPhiSlice < nnPhi ? nnPhi : tPhiSlice;

      LOGP(info, "{}", fmt::format("PoissonMultiGrid3D: Restrict3D, tnRRow={}, tnZColumn={}, newPhiSlice={}, oldPhiSlice={}", tnRRow, tnZColumn, tPhiSlice, otPhiSlice));
      restrict3D(tvChargeFMG[count - 1], tvChargeFMG[count - 2], tnRRow, tnZColumn, tPhiSlice, otPhiSlice);
      // copy boundary values of V
      restrictBoundary3D(tvArrayV[count - 1], tvArrayV[count - 2], tnRRow, tnZColumn, tPhiSlice, otPhiSlice);
      otPhiSlice = tPhiSlice;

      iOne = 2 * iOne; // doubling
      jOne = 2 * jOne; // doubling
      kOne = 2 * kOne;
    }

    // Relax on the coarsest grid
    // FMG
    // 2) Relax on the coarsest grid
    // move to the coarsest + 1
    iOne = iOne * 0.5;
    jOne = jOne * 0.5;
    kOne = kOne * 0.5;
    tnRRow = iOne == 1 ? Nr : Nr / iOne + 1;
    tnZColumn = jOne == 1 ? Nz : Nz / jOne + 1;
    tPhiSlice = kOne == 1 ? Nphi : Nphi / kOne;
    tPhiSlice = tPhiSlice < nnPhi ? nnPhi : tPhiSlice;
    otPhiSlice = tPhiSlice;

    const DataT h = gridSpacingR * iOne;
    const DataT h2 = h * h;
    const DataT gridSizePhiInv = tPhiSlice * INVTWOPI;               // h_{phi}
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

      iOne = iOne * 0.5;
      jOne = jOne * 0.5;
      kOne = kOne * 0.5;

      tnRRow = iOne == 1 ? Nr : Nr / iOne + 1;
      tnZColumn = jOne == 1 ? Nz : Nz / jOne + 1;
      tPhiSlice = kOne == 1 ? Nphi : Nphi / kOne;
      tPhiSlice = tPhiSlice < nnPhi ? nnPhi : tPhiSlice;

      // 4) a) interpolate from 2h --> h grid
      interp3D(tvArrayV[count], tvArrayV[count + 1], tnRRow, tnZColumn, tPhiSlice, otPhiSlice);

      // Copy the relax charge to the tvCharge
      if (count > 0) {
        tvCharge[count] = tvChargeFMG[count];
      }
      for (int mgCycle = 0; mgCycle < MGParameters::nMGCycle; ++mgCycle) {
        // copy to store previous potential
        tvPrevArrayV[count] = tvArrayV[count];

        vCycle3D(symmetry, count + 1, nLoop, MGParameters::nPre, MGParameters::nPost, ratioZ, tvArrayV, tvCharge, tvResidue, coefficient1, coefficient2, coefficient3, coefficient4, inverseCoefficient4);

        // converge error
        const DataT convergenceError = getConvergenceError(tvArrayV[count], tvPrevArrayV[count]);
        // if already converge just break move to finer grid
        if (convergenceError <= sConvergenceError) {
          break;
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
  for (int iphi = 0; iphi < Nphi; ++iphi) {
    for (int ir = 0; ir < Nr; ++ir) {
      for (int iz = 0; iz < Nz; ++iz) {
        matricesV(iz, ir, iphi) = tvArrayV[0](ir, iz, iphi);
      }
    }
  }
}

template <typename DataT, size_t Nz, size_t Nr, size_t Nphi>
void PoissonSolver<DataT, Nz, Nr, Nphi>::wCycle2D(const int gridFrom, const int gridTo, const int gamma, const int nPre, const int nPost, const DataT gridSizeR, const DataT ratio,
                                                  std::vector<Vector>& tvArrayV, std::vector<Vector>& tvCharge, std::vector<Vector>& tvResidue)
{
  int iOne = 1 << (gridFrom - 1);
  int jOne = 1 << (gridFrom - 1);

  int tnRRow = iOne == 1 ? Nr : Nr / iOne + 1;
  int tnZColumn = jOne == 1 ? Nz : Nz / jOne + 1;

  std::vector<DataT> coefficient1(Nr);
  std::vector<DataT> coefficient2(Nz);

  // 1) Go to coarsest level
  for (int count = gridFrom; count <= gridTo - 2; ++count) {
    const DataT h = gridSizeR * iOne;
    const DataT h2 = h * h;
    const DataT ih2 = 1.0 / h2;
    const DataT tempRatio = ratio * iOne * iOne / (jOne * jOne);
    const DataT tempFourth = 1.0 / (2.0 + 2.0 * tempRatio);
    const DataT inverseTempFourth = 1.0 / tempFourth;
    calcCoefficients2D(1, tnRRow - 1, h, coefficient1, coefficient2);
    Vector matricesCurrentV = tvArrayV[count - 1];
    Vector matricesCurrentCharge = tvCharge[count - 1];
    Vector residue = tvResidue[count - 1];

    // 1) Pre-Smoothing: Gauss-Seidel Relaxation or Jacobi
    for (int jPre = 1; jPre <= nPre; ++jPre) {
      relax2D(matricesCurrentV, matricesCurrentCharge, tnRRow, tnZColumn, h2, tempFourth, tempRatio, coefficient1, coefficient2);
    }

    // 2) Residue calculation
    residue2D(residue, matricesCurrentV, matricesCurrentCharge, tnRRow, tnZColumn, ih2, inverseTempFourth, tempRatio, coefficient1, coefficient2);

    iOne = 2 * iOne;
    jOne = 2 * jOne;
    tnRRow = iOne == 1 ? Nr : Nr / iOne + 1;
    tnZColumn = jOne == 1 ? Nz : Nz / jOne + 1;

    matricesCurrentCharge = tvCharge[count];
    matricesCurrentV = tvArrayV[count];

    //3) Restriction
    restrict2D(matricesCurrentCharge, residue, tnRRow, tnZColumn, 0);
  }

  // Do V cycle from: gridTo-1 to gridTo gamma times
  for (int iGamma = 0; iGamma < gamma; ++iGamma) {
    vCycle2D(gridTo - 1, gridTo, nPre, nPost, gridSizeR, ratio, tvArrayV, tvCharge, tvResidue);
  }

  // Go to finest grid
  for (int count = gridTo - 2; count >= gridFrom; --count) {
    iOne = iOne * 0.5;
    jOne = jOne * 0.5;

    const DataT h = gridSizeR * iOne;
    const DataT h2 = h * h;
    const DataT tempRatio = ratio * iOne * iOne / (jOne * jOne);
    const DataT tempFourth = 1.0 / (2.0 + 2.0 * tempRatio);

    tnRRow = iOne == 1 ? Nr : Nr / iOne + 1;
    tnZColumn = jOne == 1 ? Nz : Nz / jOne + 1;
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

template <typename DataT, size_t Nz, size_t Nr, size_t Nphi>
void PoissonSolver<DataT, Nz, Nr, Nphi>::vCycle2D(const int gridFrom, const int gridTo, const int nPre, const int nPost, const DataT gridSizeR, const DataT ratio, std::vector<Vector>& tvArrayV,
                                                  std::vector<Vector>& tvCharge, std::vector<Vector>& tvResidue)
{
  int iOne = 1 << (gridFrom - 1);
  int jOne = 1 << (gridFrom - 1);

  int tnRRow = iOne == 1 ? Nr : Nr / iOne + 1;
  int tnZColumn = jOne == 1 ? Nz : Nz / jOne + 1;

  std::vector<DataT> coefficient1(Nr);
  std::vector<DataT> coefficient2(Nz);

  // 1) Go to coarsest level
  for (int count = gridFrom; count <= gridTo - 1; ++count) {
    const DataT h = gridSizeR * iOne;
    const DataT h2 = h * h;
    const DataT ih2 = 1.0 / h2;
    const DataT tempRatio = ratio * iOne * iOne / (jOne * jOne);
    const DataT tempFourth = 1.0 / (2.0 + 2.0 * tempRatio);
    const DataT inverseTempFourth = 1.0 / tempFourth;
    calcCoefficients2D(1, tnRRow - 1, h, coefficient1, coefficient2);

    for (int jPre = 1; jPre <= nPre; ++jPre) {
      relax2D(tvArrayV[count - 1], tvCharge[count - 1], tnRRow, tnZColumn, h2, tempFourth, tempRatio, coefficient1, coefficient2);
    }

    // 2) Residue calculation
    residue2D(tvResidue[count - 1], tvArrayV[count - 1], tvCharge[count - 1], tnRRow, tnZColumn, ih2, inverseTempFourth, tempRatio, coefficient1, coefficient2);

    iOne = 2 * iOne;
    jOne = 2 * jOne;
    tnRRow = iOne == 1 ? Nr : Nr / iOne + 1;
    tnZColumn = jOne == 1 ? Nz : Nz / jOne + 1;

    //3) Restriction
    restrict2D(tvCharge[count], tvResidue[count - 1], tnRRow, tnZColumn, 0);

    //4) Zeroing coarser V
    std::fill(tvArrayV[count].begin(), tvArrayV[count].end(), 0); // is this necessary???
  }

  // 5) coarsest grid
  const DataT h = gridSizeR * iOne;
  const DataT h2 = h * h;
  const DataT tempRatio = ratio * iOne * iOne / (jOne * jOne);
  const DataT tempFourth = 1.0 / (2.0 + 2.0 * tempRatio);
  calcCoefficients2D(1, tnRRow - 1, h, coefficient1, coefficient2);

  relax2D(tvArrayV[gridTo - 1], tvCharge[gridTo - 1], tnRRow, tnZColumn, h2, tempFourth, tempRatio, coefficient1, coefficient2);

  // Go to finest grid
  for (int count = gridTo - 1; count >= gridFrom; count--) {

    iOne = iOne * 0.5;
    jOne = jOne * 0.5;

    const DataT h = gridSizeR * iOne;
    const DataT h2 = h * h;
    const DataT tempRatio = ratio * iOne * iOne / (jOne * jOne);
    const DataT tempFourth = 1.0 / (2.0 + 2.0 * tempRatio);

    tnRRow = iOne == 1 ? Nr : Nr / iOne + 1;
    tnZColumn = jOne == 1 ? Nz : Nz / jOne + 1;

    // 6) Interpolation/Prolongation
    addInterp2D(tvArrayV[count - 1], tvArrayV[count], tnRRow, tnZColumn, 0);

    calcCoefficients2D(1, tnRRow - 1, h, coefficient1, coefficient2);

    // 7) Post-Smoothing: Gauss-Seidel Relaxation
    for (int jPost = 1; jPost <= nPost; ++jPost) {
      relax2D(tvArrayV[count - 1], tvCharge[count - 1], tnRRow, tnZColumn, h2, tempFourth, tempRatio, coefficient1, coefficient2);
    } // end post smoothing
  }
}

template <typename DataT, size_t Nz, size_t Nr, size_t Nphi>
void PoissonSolver<DataT, Nz, Nr, Nphi>::vCycle3D2D(const int symmetry, const int gridFrom, const int gridTo, const int nPre, const int nPost, const DataT ratioZ, const DataT ratioPhi,
                                                    std::vector<Vector>& tvArrayV, std::vector<Vector>& tvCharge, std::vector<Vector>& tvResidue, std::array<DataT, Nr>& coefficient1,
                                                    std::array<DataT, Nr>& coefficient2, std::array<DataT, Nr>& coefficient3, std::array<DataT, Nr>& coefficient4, std::array<DataT, Nr>& inverseCoefficient4) const
{
  int iOne = 1 << (gridFrom - 1);
  int jOne = 1 << (gridFrom - 1);
  int tnRRow = iOne == 1 ? Nr : Nr / iOne + 1;
  int tnZColumn = jOne == 1 ? Nz : Nz / jOne + 1;

  for (int count = gridFrom; count <= gridTo - 1; ++count) {
    const DataT h = getSpacingR() * iOne;
    const DataT h2 = h * h;
    const DataT ih2 = 1.0 / h2;
    const int iOne2 = iOne * iOne;
    const DataT tempRatioPhi = ratioPhi * iOne2; // Used tobe divided by ( m_one * m_one ) when m_one was != 1
    const DataT tempRatioZ = ratioZ * iOne2 / (jOne * jOne);

    calcCoefficients(1, tnRRow - 1, h, tempRatioZ, tempRatioPhi, coefficient1, coefficient2, coefficient3, coefficient4);
    for (unsigned int i = 1; i < tnRRow - 1; ++i) {
      inverseCoefficient4[i] = 1.0 / coefficient4[i];
    }

    //Info("VCycle3D2D","Before Pre-smoothing");
    // 1) Pre-Smoothing: Gauss-Seidel Relaxation or Jacobi
    for (int jPre = 1; jPre <= nPre; ++jPre) {
      relax3D(tvArrayV[count - 1], tvCharge[count - 1], tnRRow, tnZColumn, Nphi, symmetry, h2, tempRatioZ, coefficient1, coefficient2, coefficient3, coefficient4);
    } // end pre smoothing

    // 2) Residue calculation
    residue3D(tvResidue[count - 1], tvArrayV[count - 1], tvCharge[count - 1], tnRRow, tnZColumn, Nphi, symmetry, ih2, tempRatioZ, coefficient1, coefficient2, coefficient3, inverseCoefficient4);

    iOne = 2 * iOne;
    jOne = 2 * jOne;
    tnRRow = iOne == 1 ? Nr : Nr / iOne + 1;
    tnZColumn = jOne == 1 ? Nz : Nz / jOne + 1;

    //3) Restriction
    restrict3D(tvCharge[count], tvResidue[count - 1], tnRRow, tnZColumn, Nphi, Nphi);

    //4) Zeroing coarser V
    std::fill(tvArrayV[count].begin(), tvArrayV[count].end(), 0);
  }

  // coarsest grid
  const DataT h = getSpacingR() * iOne;
  const DataT h2 = h * h;

  const int iOne2 = iOne * iOne;
  const DataT tempRatioPhi = ratioPhi * iOne2;
  const DataT tempRatioZ = ratioZ * iOne2 / (jOne * jOne);

  calcCoefficients(1, tnRRow - 1, h, tempRatioZ, tempRatioPhi, coefficient1, coefficient2, coefficient3, coefficient4);

  // 3) Relax on the coarsest grid
  relax3D(tvArrayV[gridTo - 1], tvCharge[gridTo - 1], tnRRow, tnZColumn, Nphi, symmetry, h2, tempRatioZ, coefficient1, coefficient2, coefficient3, coefficient4);

  // back to fine
  for (int count = gridTo - 1; count >= gridFrom; --count) {
    iOne = iOne * 0.5;
    jOne = jOne * 0.5;

    tnRRow = iOne == 1 ? Nr : Nr / iOne + 1;
    tnZColumn = jOne == 1 ? Nz : Nz / jOne + 1;

    const DataT h = getSpacingR() * iOne;
    const DataT h2 = h * h;
    const int iOne2 = iOne * iOne;
    const DataT tempRatioPhi = ratioPhi * iOne2; // Used tobe divided by ( m_one * m_one ) when m_one was != 1
    const DataT tempRatioZ = ratioZ * iOne2 / (jOne * jOne);

    // 4) Interpolation/Prolongation
    addInterp3D(tvArrayV[count - 1], tvArrayV[count], tnRRow, tnZColumn, Nphi, Nphi);

    calcCoefficients(1, tnRRow - 1, h, tempRatioZ, tempRatioPhi, coefficient1, coefficient2, coefficient3, coefficient4);

    // 5) Post-Smoothing: Gauss-Seidel Relaxation
    for (int jPost = 1; jPost <= nPost; ++jPost) {
      relax3D(tvArrayV[count - 1], tvCharge[count - 1], tnRRow, tnZColumn, Nphi, symmetry, h2, tempRatioZ, coefficient1, coefficient2, coefficient3, coefficient4);
    } // end post smoothing
  }
}

template <typename DataT, size_t Nz, size_t Nr, size_t Nphi>
void PoissonSolver<DataT, Nz, Nr, Nphi>::vCycle3D(const int symmetry, const int gridFrom, const int gridTo, const int nPre, const int nPost, const DataT ratioZ, std::vector<Vector>& tvArrayV,
                                                  std::vector<Vector>& tvCharge, std::vector<Vector>& tvResidue, std::array<DataT, Nr>& coefficient1, std::array<DataT, Nr>& coefficient2, std::array<DataT, Nr>& coefficient3,
                                                  std::array<DataT, Nr>& coefficient4, std::array<DataT, Nr>& inverseCoefficient4) const
{
  const DataT gridSpacingR = getSpacingR();

  int iOne = 1 << (gridFrom - 1);
  int jOne = 1 << (gridFrom - 1);
  int kOne = 1 << (gridFrom - 1);

  int nnPhi = Nphi;
  while (nnPhi % 2 == 0) {
    nnPhi *= 0.5;
  }

  int tnRRow = iOne == 1 ? Nr : Nr / iOne + 1;
  int tnZColumn = jOne == 1 ? Nz : Nz / jOne + 1;
  int tPhiSlice = kOne == 1 ? Nphi : Nphi / kOne;
  tPhiSlice = tPhiSlice < nnPhi ? nnPhi : tPhiSlice;

  for (int count = gridFrom; count <= gridTo - 1; ++count) {
    const int otPhiSlice = tPhiSlice;
    const DataT h = gridSpacingR * iOne;
    const DataT h2 = h * h;
    const DataT ih2 = 1.0 / h2;
    const DataT tempGridSizePhiInv = tPhiSlice * INVTWOPI;                   // phi now is multiGrid
    const DataT tempRatioPhi = h2 * tempGridSizePhiInv * tempGridSizePhiInv; // ratio_{phi} = gridSize_{r} / gridSize_{phi}
    const DataT tempRatioZ = ratioZ * iOne * iOne / (jOne * jOne);

    calcCoefficients(1, tnRRow - 1, h, tempRatioZ, tempRatioPhi, coefficient1, coefficient2, coefficient3, coefficient4);

    for (int i = 1; i < tnRRow - 1; ++i) {
      inverseCoefficient4[i] = 1.0 / coefficient4[i];
    }

    // 1) Pre-Smoothing: Gauss-Seidel Relaxation or Jacobi
    for (int jPre = 1; jPre <= nPre; ++jPre) {
      relax3D(tvArrayV[count - 1], tvCharge[count - 1], tnRRow, tnZColumn, tPhiSlice, symmetry, h2, tempRatioZ, coefficient1, coefficient2, coefficient3, coefficient4);
    } // end pre smoothing

    // 2) Residue calculation
    residue3D(tvResidue[count - 1], tvArrayV[count - 1], tvCharge[count - 1], tnRRow, tnZColumn, tPhiSlice, symmetry, ih2, tempRatioZ, coefficient1, coefficient2, coefficient3, inverseCoefficient4);

    iOne = 2 * iOne;
    jOne = 2 * jOne;
    kOne = 2 * kOne;
    tnRRow = iOne == 1 ? Nr : Nr / iOne + 1;
    tnZColumn = jOne == 1 ? Nz : Nz / jOne + 1;
    tPhiSlice = Nphi / kOne;
    tPhiSlice = tPhiSlice < nnPhi ? nnPhi : tPhiSlice;

    //3) Restriction
    restrict3D(tvCharge[count], tvResidue[count - 1], tnRRow, tnZColumn, tPhiSlice, otPhiSlice);

    //4) Zeroing coarser V
    std::fill(tvArrayV[count].begin(), tvArrayV[count].end(), 0);
  }

  // coarsest grid
  const DataT h = gridSpacingR * iOne;
  const DataT h2 = h * h;
  const DataT tempGridSizePhiInv = tPhiSlice * INVTWOPI;                   // phi now is multiGrid
  const DataT tempRatioPhi = h2 * tempGridSizePhiInv * tempGridSizePhiInv; // ratio_{phi} = gridSize_{r} / gridSize_{phi}
  const DataT tempRatioZ = ratioZ * iOne * iOne / (jOne * jOne);

  calcCoefficients(1, tnRRow - 1, h, tempRatioZ, tempRatioPhi, coefficient1, coefficient2, coefficient3, coefficient4);

  // 3) Relax on the coarsest grid
  relax3D(tvArrayV[gridTo - 1], tvCharge[gridTo - 1], tnRRow, tnZColumn, tPhiSlice, symmetry, h2, tempRatioZ, coefficient1, coefficient2, coefficient3, coefficient4);
  // back to fine
  for (int count = gridTo - 1; count >= gridFrom; --count) {
    const int otPhiSlice = tPhiSlice;
    iOne = iOne * 0.5;
    jOne = jOne * 0.5;
    kOne = kOne * 0.5;

    tnRRow = iOne == 1 ? Nr : Nr / iOne + 1;
    tnZColumn = jOne == 1 ? Nz : Nz / jOne + 1;
    tPhiSlice = kOne == 1 ? Nphi : Nphi / kOne;
    tPhiSlice = tPhiSlice < nnPhi ? nnPhi : tPhiSlice;

    const DataT h = gridSpacingR * iOne;
    const DataT h2 = h * h;
    const DataT tempGridSizePhiInv = tPhiSlice * INVTWOPI;
    const DataT tempRatioPhi = h2 * tempGridSizePhiInv * tempGridSizePhiInv; // ratio_{phi} = gridSize_{r} / gridSize_{phi}
    const DataT tempRatioZ = ratioZ * iOne * iOne / (jOne * jOne);

    // 4) Interpolation/Prolongation
    addInterp3D(tvArrayV[count - 1], tvArrayV[count], tnRRow, tnZColumn, tPhiSlice, otPhiSlice);

    calcCoefficients(1, tnRRow - 1, h, tempRatioZ, tempRatioPhi, coefficient1, coefficient2, coefficient3, coefficient4);

    // 5) Post-Smoothing: Gauss-Seidel Relaxation
    for (int jPost = 1; jPost <= nPost; ++jPost) {
      relax3D(tvArrayV[count - 1], tvCharge[count - 1], tnRRow, tnZColumn, tPhiSlice, symmetry, h2, tempRatioZ, coefficient1, coefficient2, coefficient3, coefficient4);
    }
  }
}

template <typename DataT, size_t Nz, size_t Nr, size_t Nphi>
void PoissonSolver<DataT, Nz, Nr, Nphi>::residue2D(Vector& residue, const Vector& matricesCurrentV, const Vector& matricesCurrentCharge, const int tnRRow, const int tnZColumn, const DataT ih2, const DataT inverseTempFourth,
                                                   const DataT tempRatio, std::vector<DataT>& coefficient1, std::vector<DataT>& coefficient2)
{
  const int iPhi = 0;
#pragma omp parallel for num_threads(sNThreads)
  for (int i = 1; i < tnRRow - 1; ++i) {
    for (int j = 1; j < tnZColumn - 1; ++j) {
      residue(i, j, iPhi) = ih2 * (coefficient1[i] * matricesCurrentV(i + 1, j, iPhi) + coefficient2[i] * matricesCurrentV(i - 1, j, iPhi) + tempRatio * (matricesCurrentV(i, j + 1, iPhi) + matricesCurrentV(i, j - 1, iPhi)) - inverseTempFourth * matricesCurrentV(i, j, iPhi)) + matricesCurrentCharge(i, j, iPhi);
    } // end cols
  }   // end nRRow

  //Boundary points.
  for (int i = 0; i < tnRRow; ++i) {
    residue(i, 0, iPhi) = residue(i, tnZColumn - 1, iPhi) = 0.0;
  }

  for (int j = 0; j < tnZColumn; ++j) {
    residue(0, j, iPhi) = residue(tnRRow - 1, j, iPhi) = 0.0;
  }
}

template <typename DataT, size_t Nz, size_t Nr, size_t Nphi>
void PoissonSolver<DataT, Nz, Nr, Nphi>::residue3D(Vector& residue, const Vector& matricesCurrentV, const Vector& matricesCurrentCharge, const int tnRRow, const int tnZColumn, const int tnPhi, const int symmetry,
                                                   const DataT ih2, const DataT tempRatioZ, const std::array<DataT, Nr>& coefficient1, const std::array<DataT, Nr>& coefficient2, const std::array<DataT, Nr>& coefficient3, const std::array<DataT, Nr>& inverseCoefficient4) const
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
    }   // end Nr
  }
}

template <typename DataT, size_t Nz, size_t Nr, size_t Nphi>
void PoissonSolver<DataT, Nz, Nr, Nphi>::interp3D(Vector& matricesCurrentV, const Vector& matricesCurrentVC, const int tnRRow, const int tnZColumn, const int newPhiSlice, const int oldPhiSlice) const
{
  // Do restrict 2 D for each slice
  if (newPhiSlice == 2 * oldPhiSlice) {
    for (int m = 0; m < newPhiSlice; m += 2) {
      // assuming no symmetry
      int mm = m * 0.5;
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
          const int iHalf = i * 0.5;
          const int jHalf = j * 0.5;
          matricesCurrentV(i, j, m) = matricesCurrentVC(iHalf, jHalf, mm);
          // point on corner lines at phi direction
          matricesCurrentV(i, j, mp1) = 0.5 * (matricesCurrentVC(iHalf, jHalf, mm) + matricesCurrentVC(iHalf, jHalf, mmPlus));
        }
      }

      for (int j = 1; j < tnZColumn - 1; j += 2) {
        for (int i = 2; i < tnRRow - 1; i += 2) {
          const int iHalf = i * 0.5;
          const int jHalf = j * 0.5;
          matricesCurrentV(i, j, m) = 0.5 * (matricesCurrentVC(iHalf, jHalf, mm) + matricesCurrentVC(iHalf, jHalf + 1, mm));
          // point on corner lines at phi direction
          matricesCurrentV(i, j, mp1) = 0.25 * (matricesCurrentVC(iHalf, jHalf, mm) + matricesCurrentVC(iHalf, jHalf + 1, mm) + matricesCurrentVC(iHalf, jHalf, mmPlus) + matricesCurrentVC(iHalf, jHalf + 1, mmPlus));
        }
      }

      for (int j = 2; j < tnZColumn - 1; j += 2) {
        for (int i = 1; i < tnRRow - 1; i += 2) {
          const int iHalf = i * 0.5;
          const int jHalf = j * 0.5;
          matricesCurrentV(i, j, m) = 0.5 * (matricesCurrentVC(iHalf, jHalf, mm) + matricesCurrentVC(iHalf + 1, jHalf, mm));
          // point on line at phi direction
          matricesCurrentV(i, j, mp1) = 0.25 * ((matricesCurrentVC(iHalf, jHalf, mm) + matricesCurrentVC(iHalf, jHalf, mmPlus)) + (matricesCurrentVC(iHalf + 1, jHalf, mmPlus) + matricesCurrentVC(iHalf + 1, jHalf, mm)));
        }
      }

      for (int j = 1; j < tnZColumn - 1; j += 2) {
        for (int i = 1; i < tnRRow - 1; i += 2) {
          const int iHalf = i * 0.5;
          const int jHalf = j * 0.5;
          matricesCurrentV(i, j, m) = 0.25 * ((matricesCurrentVC(iHalf, jHalf, mm) + matricesCurrentVC(iHalf, jHalf + 1, mm)) + (matricesCurrentVC(iHalf + 1, jHalf, mm) + matricesCurrentVC(iHalf + 1, jHalf + 1, mm)));
          // point at the center at phi direction
          matricesCurrentV(i, j, mp1) = 0.125 * ((matricesCurrentVC(iHalf, jHalf, mm) + matricesCurrentVC(iHalf, jHalf + 1, mm) + matricesCurrentVC(iHalf, jHalf, mmPlus) + matricesCurrentVC(iHalf, jHalf + 1, mmPlus)) +
                                                 (matricesCurrentVC(iHalf + 1, jHalf, mm) + matricesCurrentVC(iHalf + 1, jHalf + 1, mm) + matricesCurrentVC(iHalf + 1, jHalf, mmPlus) + matricesCurrentVC(iHalf + 1, jHalf + 1, mmPlus)));
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

template <typename DataT, size_t Nz, size_t Nr, size_t Nphi>
void PoissonSolver<DataT, Nz, Nr, Nphi>::interp2D(Vector& matricesCurrentV, const Vector& matricesCurrentVC, const int tnRRow, const int tnZColumn, const int iphi) const
{
  for (int j = 2; j < tnZColumn - 1; j += 2) {
    for (int i = 2; i < tnRRow - 1; i += 2) {
      matricesCurrentV(i, j, iphi) = matricesCurrentVC(i * 0.5, j * 0.5, iphi);
    }
  }

  for (int j = 1; j < tnZColumn - 1; j += 2) {
    for (int i = 2; i < tnRRow - 1; i += 2) {
      const int iHalf = i * 0.5;
      const int jHalf = j * 0.5;
      matricesCurrentV(i, j, iphi) = 0.5 * (matricesCurrentVC(iHalf, jHalf, iphi) + matricesCurrentVC(iHalf, jHalf + 1, iphi));
    }
  }

  for (int j = 2; j < tnZColumn - 1; j += 2) {
    for (int i = 1; i < tnRRow - 1; i += 2) {
      const int iHalf = i * 0.5;
      const int jHalf = j * 0.5;
      matricesCurrentV(i, j, iphi) = 0.5 * (matricesCurrentVC(iHalf, jHalf, iphi) + matricesCurrentVC(iHalf + 1, jHalf, iphi));
    }
  }

  // only if full
  if (MGParameters::gtType == GridTransferType::Full) {
    for (int j = 1; j < tnZColumn - 1; j += 2) {
      for (int i = 1; i < tnRRow - 1; i += 2) {
        const int iHalf = i * 0.5;
        const int jHalf = j * 0.5;
        matricesCurrentV(i, j, iphi) = 0.25 * (matricesCurrentVC(iHalf, jHalf, iphi) + matricesCurrentVC(iHalf, jHalf + 1, iphi) + matricesCurrentVC(iHalf + 1, jHalf, iphi) + matricesCurrentVC(iHalf + 1, jHalf + 1, iphi));
      }
    }
  }
}

template <typename DataT, size_t Nz, size_t Nr, size_t Nphi>
void PoissonSolver<DataT, Nz, Nr, Nphi>::addInterp3D(Vector& matricesCurrentV, const Vector& matricesCurrentVC, const int tnRRow, const int tnZColumn, const int newPhiSlice, const int oldPhiSlice) const
{
  // Do restrict 2 D for each slice
  if (newPhiSlice == 2 * oldPhiSlice) {
    for (int m = 0; m < newPhiSlice; m += 2) {
      // assuming no symmetry
      int mm = m * 0.5;
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
          const int iHalf = i * 0.5;
          const int jHalf = j * 0.5;
          matricesCurrentV(i, j, m) += matricesCurrentVC(iHalf, jHalf, mm);
          // point on corner lines at phi direction
          matricesCurrentV(i, j, mp1) += 0.5 * (matricesCurrentVC(iHalf, jHalf, mm) + matricesCurrentVC(iHalf, jHalf, mmPlus));
        }
      }

      for (int j = 1; j < tnZColumn - 1; j += 2) {
        for (int i = 2; i < tnRRow - 1; i += 2) {
          const int iHalf = i * 0.5;
          const int jHalf = j * 0.5;
          matricesCurrentV(i, j, m) += 0.5 * (matricesCurrentVC(iHalf, jHalf, mm) + matricesCurrentVC(iHalf, jHalf + 1, mm));
          // point on corner lines at phi direction
          matricesCurrentV(i, j, mp1) += 0.25 * (matricesCurrentVC(iHalf, jHalf, mm) + matricesCurrentVC(iHalf, jHalf + 1, mm) + matricesCurrentVC(iHalf, jHalf, mmPlus) + matricesCurrentVC(iHalf, jHalf + 1, mmPlus));
        }
      }

      for (int j = 2; j < tnZColumn - 1; j += 2) {
        for (int i = 1; i < tnRRow - 1; i += 2) {
          const int iHalf = i * 0.5;
          const int jHalf = j * 0.5;
          matricesCurrentV(i, j, m) += 0.5 * (matricesCurrentVC(iHalf, jHalf, mm) + matricesCurrentVC(iHalf + 1, jHalf, mm));
          // point on line at phi direction
          matricesCurrentV(i, j, mp1) += 0.25 * ((matricesCurrentVC(iHalf, jHalf, mm) + matricesCurrentVC(iHalf, jHalf, mmPlus)) + (matricesCurrentVC(iHalf + 1, jHalf, mmPlus) + matricesCurrentVC(iHalf + 1, jHalf, mm)));
        }
      }

      for (int j = 1; j < tnZColumn - 1; j += 2) {
        for (int i = 1; i < tnRRow - 1; i += 2) {
          const int iHalf = i * 0.5;
          const int jHalf = j * 0.5;
          matricesCurrentV(i, j, m) += 0.25 * ((matricesCurrentVC(iHalf, jHalf, mm) + matricesCurrentVC(iHalf, jHalf + 1, mm)) + (matricesCurrentVC(iHalf + 1, jHalf, mm) + matricesCurrentVC(iHalf + 1, jHalf + 1, mm)));
          // point at the center at phi direction
          matricesCurrentV(i, j, mp1) += 0.125 * ((matricesCurrentVC(iHalf, jHalf, mm) + matricesCurrentVC(iHalf, jHalf + 1, mm) + matricesCurrentVC(iHalf, jHalf, mmPlus) + matricesCurrentVC(iHalf, jHalf + 1, mmPlus)) +
                                                  (matricesCurrentVC(iHalf + 1, jHalf, mm) + matricesCurrentVC(iHalf + 1, jHalf + 1, mm) + matricesCurrentVC(iHalf + 1, jHalf, mmPlus) + matricesCurrentVC(iHalf + 1, jHalf + 1, mmPlus)));
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

template <typename DataT, size_t Nz, size_t Nr, size_t Nphi>
void PoissonSolver<DataT, Nz, Nr, Nphi>::addInterp2D(Vector& matricesCurrentV, const Vector& matricesCurrentVC, const int tnRRow, const int tnZColumn, const int tnPhi) const
{
  for (int j = 2; j < tnZColumn - 1; j += 2) {
    for (int i = 2; i < tnRRow - 1; i += 2) {
      matricesCurrentV(i, j, tnPhi) = matricesCurrentV(i, j, tnPhi) + matricesCurrentVC(i * 0.5, j * 0.5, tnPhi);
    }
  }

  for (int j = 1; j < tnZColumn - 1; j += 2) {
    for (int i = 2; i < tnRRow - 1; i += 2) {
      const int iHalf = 0.5 * i;
      const int jHalf = 0.5 * j;
      matricesCurrentV(i, j, tnPhi) = matricesCurrentV(i, j, tnPhi) + 0.5 * (matricesCurrentVC(iHalf, jHalf, tnPhi) + matricesCurrentVC(iHalf, jHalf + 1, tnPhi));
    }
  }

  for (int j = 2; j < tnZColumn - 1; j += 2) {
    for (int i = 1; i < tnRRow - 1; i += 2) {
      const int iHalf = 0.5 * i;
      const int jHalf = 0.5 * j;
      matricesCurrentV(i, j, tnPhi) = matricesCurrentV(i, j, tnPhi) + 0.5 * (matricesCurrentVC(iHalf, jHalf, tnPhi) + matricesCurrentVC(iHalf + 1, jHalf, tnPhi));
    }
  }

  // only if full
  if (MGParameters::gtType == GridTransferType::Full) {
    for (int j = 1; j < tnZColumn - 1; j += 2) {
      for (int i = 1; i < tnRRow - 1; i += 2) {
        const int iHalf = 0.5 * i;
        const int jHalf = 0.5 * j;
        matricesCurrentV(i, j, tnPhi) = matricesCurrentV(i, j, tnPhi) + 0.25 * (matricesCurrentVC(iHalf, jHalf, tnPhi) + matricesCurrentVC(iHalf, jHalf + 1, tnPhi) + matricesCurrentVC(iHalf + 1, jHalf, tnPhi) + matricesCurrentVC(iHalf + 1, jHalf + 1, tnPhi));
      }
    }
  }
}

template <typename DataT, size_t Nz, size_t Nr, size_t Nphi>
void PoissonSolver<DataT, Nz, Nr, Nphi>::relax3D(Vector& matricesCurrentV, const Vector& matricesCurrentCharge, const int tnRRow, const int tnZColumn, const int iPhi, const int symmetry, const DataT h2,
                                                 const DataT tempRatioZ, const std::array<DataT, Nr>& coefficient1, const std::array<DataT, Nr>& coefficient2, const std::array<DataT, Nr>& coefficient3, const std::array<DataT, Nr>& coefficient4) const
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
        }   // end Nr
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
      }   // end Nr
    }     // end phi
  } else {
    // Case weighted Jacobi
    // TODO
  }
}

template <typename DataT, size_t Nz, size_t Nr, size_t Nphi>
void PoissonSolver<DataT, Nz, Nr, Nphi>::relax2D(Vector& matricesCurrentV, const Vector& matricesCurrentCharge, const int tnRRow, const int tnZColumn, const DataT h2, const DataT tempFourth, const DataT tempRatio,
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
      }   // end Nr
    }     // end pass red-black
  } else if (MGParameters::relaxType == RelaxType::Jacobi) {
    for (int j = 1; j < tnZColumn - 1; ++j) {
      for (int i = 1; i < tnRRow - 1; ++i) {
        matricesCurrentV(i, j, iPhi) = tempFourth * (coefficient1[i] * matricesCurrentV(i + 1, j, iPhi) + coefficient2[i] * matricesCurrentV(i - 1, j, iPhi) +
                                                     tempRatio * (matricesCurrentV(i, j + 1, iPhi) + matricesCurrentV(i, j - 1, iPhi)) + (h2 * matricesCurrentCharge(i, j, iPhi)));
      } // end cols
    }   // end Nr
  } else if (MGParameters::relaxType == RelaxType::WeightedJacobi) {
    // Weighted Jacobi
    // TODO
  }
}

template <typename DataT, size_t Nz, size_t Nr, size_t Nphi>
void PoissonSolver<DataT, Nz, Nr, Nphi>::restrictBoundary3D(Vector& matricesCurrentCharge, const Vector& residue, const int tnRRow, const int tnZColumn, const int newPhiSlice, const int oldPhiSlice) const
{
  // in case of full 3d and the Nphi is also coarsening
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

template <typename DataT, size_t Nz, size_t Nr, size_t Nphi>
void PoissonSolver<DataT, Nz, Nr, Nphi>::restrictBoundary2D(Vector& matricesCurrentCharge, const Vector& residue, const int tnRRow, const int tnZColumn, const int tnPhi) const
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

template <typename DataT, size_t Nz, size_t Nr, size_t Nphi>
void PoissonSolver<DataT, Nz, Nr, Nphi>::restrict3D(Vector& matricesCurrentCharge, const Vector& residue, const int tnRRow, const int tnZColumn, const int newPhiSlice, const int oldPhiSlice) const
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

          matricesCurrentCharge(i, j, m) = 0.125 * residue(ii, jj, mm) + 0.0625 * s1 + 0.03125 * s2 + 0.015625 * s3;
        } // end cols
      }   // end Nr

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

template <typename DataT, size_t Nz, size_t Nr, size_t Nphi>
void PoissonSolver<DataT, Nz, Nr, Nphi>::restrict2D(Vector& matricesCurrentCharge, const Vector& residue, const int tnRRow, const int tnZColumn, const int iphi) const
{
  for (int i = 1, ii = 2; i < tnRRow - 1; ++i, ii += 2) {
    for (int j = 1, jj = 2; j < tnZColumn - 1; ++j, jj += 2) {
      const int iip1 = ii + 1;
      const int iim1 = ii - 1;
      const int jjp1 = jj + 1;
      const int jjm1 = jj - 1;
      if (MGParameters::gtType == GridTransferType::Half) {
        // half
        matricesCurrentCharge(i, j, iphi) = 0.5 * residue(ii, jj, iphi) + 0.125 * (residue(iip1, jj, iphi) + residue(iim1, jj, iphi) + residue(ii, jjp1, iphi) + residue(ii, jjm1, iphi));
      } else if (MGParameters::gtType == GridTransferType::Full) {
        matricesCurrentCharge(i, j, iphi) = 0.25 * residue(ii, jj, iphi) + 0.125 * (residue(iip1, jj, iphi) + residue(iim1, jj, iphi) + residue(ii, jjp1, iphi) + residue(ii, jjm1, iphi)) +
                                            0.0625 * (residue(iip1, jjp1, iphi) + residue(iim1, jjp1, iphi) + residue(iip1, jjm1, iphi) + residue(iim1, jjm1, iphi));
      }
    } // end cols
  }   // end Nr
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

template <typename DataT, size_t Nz, size_t Nr, size_t Nphi>
DataT PoissonSolver<DataT, Nz, Nr, Nphi>::getConvergenceError(const Vector& matricesCurrentV, Vector& prevArrayV) const
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
    errorArr[m] = std::inner_product(start, end, start, 0.); // inner product "Sum (matrix[a]*matrix[a])"
  }
  // return largest error
  return *std::max_element(std::begin(errorArr), std::end(errorArr));
}

template <typename DataT, size_t Nz, size_t Nr, size_t Nphi>
void PoissonSolver<DataT, Nz, Nr, Nphi>::calcCoefficients(unsigned int from, unsigned int to, const DataT h, const DataT tempRatioZ, const DataT tempRatioPhi, std::array<DataT, Nr>& coefficient1, std::array<DataT, Nr>& coefficient2, std::array<DataT, Nr>& coefficient3, std::array<DataT, Nr>& coefficient4) const
{
  for (unsigned int i = from; i < to; ++i) {
    const DataT radiusInv = 1. / (TPCParameters<DataT>::IFCRADIUS + i * h);
    const DataT hRadiusTmp = h * 0.5 * radiusInv;
    coefficient1[i] = 1.0 + hRadiusTmp;
    coefficient2[i] = 1.0 - hRadiusTmp;
    coefficient3[i] = tempRatioPhi * radiusInv * radiusInv;
    coefficient4[i] = 0.5 / (1.0 + tempRatioZ + coefficient3[i]);
  }
}

template <typename DataT, size_t Nz, size_t Nr, size_t Nphi>
void PoissonSolver<DataT, Nz, Nr, Nphi>::calcCoefficients2D(unsigned int from, unsigned int to, const DataT h, std::vector<DataT>& coefficient1, std::vector<DataT>& coefficient2) const
{
  for (int i = from; i < to; ++i) {
    DataT radiusInvHalf = h * 0.5 / (TPCParameters<DataT>::IFCRADIUS + i * h);
    coefficient1[i] = 1.0 + radiusInvHalf;
    coefficient2[i] = 1.0 - radiusInvHalf;
  }
}

template class o2::tpc::PoissonSolver<double, 17, 17, 90>;
template class o2::tpc::PoissonSolver<double, 33, 33, 180>;
template class o2::tpc::PoissonSolver<double, 65, 65, 180>;
template class o2::tpc::PoissonSolver<double, 129, 129, 180>;
template class o2::tpc::PoissonSolver<double, 257, 257, 180>;
template class o2::tpc::PoissonSolver<double, 257, 257, 360>;
template class o2::tpc::PoissonSolver<double, 129, 129, 1>; // for 2D Poisson Solver
