// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

/// \file AliTPCPoissonSolver.cxx
/// \brief This class provides implementation of Poisson Eq
/// solver by MultiGrid Method
///
///
///
/// \author Rifki Sadikin <rifki.sadikin@cern.ch>, Indonesian Institute of Sciences
/// \date Nov 20, 2017

#include <TMath.h>
#include "AliTPCPoissonSolver.h"

/// \cond CLASSIMP
ClassImp(AliTPCPoissonSolver);
/// \endcond

const Double_t AliTPCPoissonSolver::fgkTPCZ0 = 249.7;                          ///< nominal gating grid position
const Double_t AliTPCPoissonSolver::fgkIFCRadius = 83.5;                       ///< radius which renders the "18 rod manifold" best -> compare calc. of Jim Thomas
const Double_t AliTPCPoissonSolver::fgkOFCRadius = 254.5;                      ///< Mean Radius of the Outer Field Cage (252.55 min, 256.45 max) (cm)
const Double_t AliTPCPoissonSolver::fgkZOffSet = 0.2;                          ///< Offset from CE: calculate all distortions closer to CE as if at this point
const Double_t AliTPCPoissonSolver::fgkCathodeV = -100000.0;                   ///< Cathode Voltage (volts)
const Double_t AliTPCPoissonSolver::fgkGG = -70.0;                             ///< Gating Grid voltage (volts)
const Double_t AliTPCPoissonSolver::fgkdvdE = 0.0024;                          ///< [cm/V] drift velocity dependency on the E field (from Magboltz for NeCO2N2 at standard environment)
const Double_t AliTPCPoissonSolver::fgkEM = -1.602176487e-19 / 9.10938215e-31; ///< charge/mass in [C/kg]
const Double_t AliTPCPoissonSolver::fgke0 = 8.854187817e-12;                   ///< vacuum permittivity [A·s/(V·m)]

Double_t AliTPCPoissonSolver::fgExactErr = 1e-4;
Double_t AliTPCPoissonSolver::fgConvergenceError = 1e-3;

/// constructor
///
AliTPCPoissonSolver::AliTPCPoissonSolver()
  : TNamed("poisson solver", "solver"),
    fErrorConvergenceNorm2{new TVectorD(fMgParameters.nMGCycle)},
    fErrorConvergenceNormInf{new TVectorD(fMgParameters.nMGCycle)},
    fError{new TVectorD(fMgParameters.nMGCycle)}

{
  // default strategy
}

/// Constructor
/// \param name name of the object
/// \param title title of the object
AliTPCPoissonSolver::AliTPCPoissonSolver(const char* name, const char* title)
  : TNamed(name, title),
    fErrorConvergenceNorm2{new TVectorD(fMgParameters.nMGCycle)},
    fErrorConvergenceNormInf{new TVectorD(fMgParameters.nMGCycle)},
    fError{new TVectorD(fMgParameters.nMGCycle)}
{
  /// constructor
}

/// destructor
AliTPCPoissonSolver::~AliTPCPoissonSolver()
{
  /// virtual destructor
  delete[] fExactSolution;
  delete fErrorConvergenceNorm2;
  delete fErrorConvergenceNormInf;
  delete fError;
}

/// Provides poisson solver in 2D
///
/// Based on the strategy (relaxation, multi grid or FFT)
///
/// \param matrixV TMatrixD& potential in matrix
/// \param matrixCharge TMatrixD& charge density in matrix (side effect
/// \param nRRow Int_t number of nRRow in the grid
/// \param nZColumn Int_t number of nZColumn in the grid
/// \param maxIteration Int_t maximum iteration for relaxation method
///
/// \return A fixed number that has nothing to do with what the function does
void AliTPCPoissonSolver::PoissonSolver2D(TMatrixD& matrixV, TMatrixD& matrixCharge, Int_t nRRow, Int_t nZColumn,
                                          Int_t maxIteration)
{
  switch (fStrategy) {
    case kMultiGrid:
      PoissonMultiGrid2D(matrixV, matrixCharge, nRRow, nZColumn);
      break;
    default:
      PoissonRelaxation2D(matrixV, matrixCharge, nRRow, nZColumn, maxIteration);
  }
}

/// Provides poisson solver in Cylindrical 3D (TPC geometry)
///
/// Strategy based on parameter settings (fStrategy and fMgParameters)provided
/// * Cascaded multi grid with S.O.R
/// * Geometric MultiGrid
///		* Cycles: V, W, Full
///		* Relaxation: Jacobi, Weighted-Jacobi, Gauss-Seidel
///		* Grid transfer operators: Full, Half
/// * Spectral Methods (TODO)
///
/// \param matricesV TMatrixD** potential in 3D matrix
/// \param matricesCharge TMatrixD** charge density in 3D matrix (side effect)
/// \param nRRow Int_t number of nRRow in the r direction of TPC
/// \param nZColumn Int_t number of nZColumn in z direction of TPC
/// \param phiSlice Int_t number of phiSlice in phi direction of T{C
/// \param maxIteration Int_t maximum iteration for relaxation method
/// \param symmetry Int_t symmetry or not
///
/// \pre Charge density distribution in **matricesCharge** is known and boundary values for **matricesV** are set
/// \post Numerical solution for potential distribution is calculated and stored in each rod at **matricesV**
void AliTPCPoissonSolver::PoissonSolver3D(TMatrixD** matricesV, TMatrixD** matricesCharge,
                                          Int_t nRRow, Int_t nZColumn, Int_t phiSlice, Int_t maxIteration,
                                          Int_t symmetry)
{
  switch (fStrategy) {
    case kMultiGrid:
      if (fMgParameters.isFull3D) {
        PoissonMultiGrid3D(matricesV, matricesCharge, nRRow, nZColumn, phiSlice, symmetry);
      } else {
        PoissonMultiGrid3D2D(matricesV, matricesCharge, nRRow, nZColumn, phiSlice, symmetry);
      }
      break;
    default:
      PoissonRelaxation3D(matricesV, matricesCharge, nRRow, nZColumn, phiSlice, maxIteration, symmetry);
  }
}

/// Solve Poisson's Equation by Relaxation Technique in 2D (assuming cylindrical symmetry)
///
/// Solve Poisson's equation in a cylindrical coordinate system. The matrixV matrix must be filled with the
/// boundary conditions on the first and last nRRow, and the first and last nZColumn.  The remainder of the
/// array can be blank or contain a preliminary guess at the solution.  The Charge density matrix contains
/// the enclosed spacecharge density at each point. The charge density matrix can be full of zero's if
/// you wish to solve Laplace equation however it should not contain random numbers or you will get
/// random numbers back as a solution.
/// Poisson's equation is solved by iteratively relaxing the matrix to the final solution.  In order to
/// speed up the convergence to the best solution, this algorithm does a binary expansion of the solution
/// space.  First it solves the problem on a very sparse grid by skipping nRRow and nZColumn in the original
/// matrix.  Then it doubles the number of points and solves the problem again.  Then it doubles the
/// number of points and solves the problem again.  This happens several times until the maximum number
/// of points has been included in the array.
///
/// NOTE: In order for this algorithm to work, the number of nRRow and nZColumn must be a power of 2 plus one.
/// So nRRow == 2**M + 1 and nZColumn == 2**N + 1.  The number of nRRow and nZColumn can be different.
///
/// Method for relaxation: S.O.R Weighted Jacobi
///
/// \param matrixV TMatrixD& potential in matrix
/// \param matrixCharge TMatrixD& charge density in matrix (side effect
/// \param nRRow Int_t number of nRRow in the grid
/// \param nZColumn Int_t number of nZColumn in the grid
/// \param maxIteration Int_t maximum iteration for relaxation method
///
/// \return A fixed number that has nothing to do with what the function does
///
///
/// Original code by Jim Thomas (STAR TPC Collaboration)
void AliTPCPoissonSolver::PoissonRelaxation2D(TMatrixD& matrixV, TMatrixD& matrixCharge, Int_t nRRow, Int_t nZColumn,
                                              Int_t maxIteration)
{
  const Float_t gridSizeR = (AliTPCPoissonSolver::fgkOFCRadius - AliTPCPoissonSolver::fgkIFCRadius) / (nRRow - 1);
  const Float_t gridSizeZ = AliTPCPoissonSolver::fgkTPCZ0 / (nZColumn - 1);
  const Float_t ratio = gridSizeR * gridSizeR / (gridSizeZ * gridSizeZ);

  TMatrixD arrayEr(nRRow, nZColumn);
  TMatrixD arrayEz(nRRow, nZColumn);

  //Check that number of nRRow and nZColumn is suitable for a binary expansion

  if (!IsPowerOfTwo(nRRow - 1)) {
    Error("PoissonRelaxation2D", "PoissonRelaxation - Error in the number of nRRow. Must be 2**M - 1");
    return;
  }

  if (!IsPowerOfTwo(nZColumn - 1)) {
    Error("PoissonRelaxation2D", "PoissonRelaxation - Error in the number of nZColumn. Must be 2**N - 1");
    return;
  }

  // Solve Poisson's equation in cylindrical coordinates by relaxation technique
  // Allow for different size grid spacing in R and Z directions
  // Use a binary expansion of the size of the matrix to speed up the solution of the problem

  Int_t iOne = (nRRow - 1) / 4;
  Int_t jOne = (nZColumn - 1) / 4;

  // Coarse until nLoop
  Int_t nLoop = 1 + (int)(0.5 + TMath::Log2((double)TMath::Max(iOne, jOne)));

  // Loop while the matrix expands & the resolution increases.
  for (Int_t count = 0; count < nLoop; count++) {

    Float_t tempGridSizeR = gridSizeR * iOne;
    Float_t tempRatio = ratio * iOne * iOne / (jOne * jOne);
    Float_t tempFourth = 1.0 / (2.0 + 2.0 * tempRatio);

    // Do this the standard C++ way to avoid gcc extensions for Float_t coefficient1[nRRow]
    std::vector<float> coefficient1(nRRow);
    std::vector<float> coefficient2(nRRow);

    for (Int_t i = iOne; i < nRRow - 1; i += iOne) {
      Float_t radius = AliTPCPoissonSolver::fgkIFCRadius + i * gridSizeR;
      coefficient1[i] = 1.0 + tempGridSizeR / (2 * radius);
      coefficient2[i] = 1.0 - tempGridSizeR / (2 * radius);
    }

    TMatrixD sumChargeDensity(nRRow, nZColumn);

    // average charge at the coarse point
    for (Int_t i = iOne; i < nRRow - 1; i += iOne) {
      Float_t radius = AliTPCPoissonSolver::fgkIFCRadius + iOne * gridSizeR;
      for (Int_t j = jOne; j < nZColumn - 1; j += jOne) {
        if (iOne == 1 && jOne == 1) {
          sumChargeDensity(i, j) = matrixCharge(i, j);
        } else {
          // Add up all enclosed charge density contributions within 1/2 unit in all directions
          Float_t weight = 0.0;
          Float_t sum = 0.0;
          sumChargeDensity(i, j) = 0.0;
          for (Int_t ii = i - iOne / 2; ii <= i + iOne / 2; ii++) {
            for (Int_t jj = j - jOne / 2; jj <= j + jOne / 2; jj++) {
              if (ii == i - iOne / 2 || ii == i + iOne / 2 || jj == j - jOne / 2 || jj == j + jOne / 2) {
                weight = 0.5;
              } else {
                weight = 1.0;
              }
              sumChargeDensity(i, j) += matrixCharge(ii, jj) * weight * radius;
              sum += weight * radius;
            }
          }
          sumChargeDensity(i, j) /= sum;
        }
        sumChargeDensity(i, j) *= tempGridSizeR * tempGridSizeR; // just saving a step later on
      }
    }

    // Iterate on the current level
    for (Int_t k = 1; k <= maxIteration; k++) {
      // Solve Poisson's Equation
      // Over-relaxation index, must be >= 1 but < 2.  Arrange for it to evolve from 2 => 1
      // as iteration increase.
      Float_t overRelax = 1.0 + TMath::Sqrt(TMath::Cos((k * TMath::PiOver2()) / maxIteration));
      Float_t overRelaxM1 = overRelax - 1.0;
      Float_t overRelaxTemp4, overRelaxCoefficient5;
      overRelaxTemp4 = overRelax * tempFourth;
      overRelaxCoefficient5 = overRelaxM1 / overRelaxTemp4;

      for (Int_t i = iOne; i < nRRow - 1; i += iOne) {
        for (Int_t j = jOne; j < nZColumn - 1; j += jOne) {
          // S.O.R
          //
          matrixV(i, j) = (coefficient2[i] * matrixV(i - iOne, j) + tempRatio * (matrixV(i, j - jOne) + matrixV(i, j + jOne)) - overRelaxCoefficient5 * matrixV(i, j) + coefficient1[i] * matrixV(i + iOne, j) + sumChargeDensity(i, j)) * overRelaxTemp4;
        }
      }

      // if already at maxIteration
      // TODO: stop when it converged
      if (k == maxIteration) {

        // After full solution is achieved, copy low resolution solution into higher res array
        // Interpolate solution
        for (Int_t i = iOne; i < nRRow - 1; i += iOne) {
          for (Int_t j = jOne; j < nZColumn - 1; j += jOne) {
            if (iOne > 1) {
              matrixV(i + iOne / 2, j) = (matrixV(i + iOne, j) + matrixV(i, j)) / 2;
              if (i == iOne) {
                matrixV(i - iOne / 2, j) = (matrixV(0, j) + matrixV(iOne, j)) / 2;
              }
            }
            if (jOne > 1) {
              matrixV(i, j + jOne / 2) = (matrixV(i, j + jOne) + matrixV(i, j)) / 2;
              if (j == jOne) {
                matrixV(i, j - jOne / 2) = (matrixV(i, 0) + matrixV(i, jOne)) / 2;
              }
            }
            if (iOne > 1 && jOne > 1) {
              matrixV(i + iOne / 2, j + jOne / 2) = (matrixV(i + iOne, j + jOne) + matrixV(i, j)) / 2;
              if (i == iOne) {
                matrixV(i - iOne / 2, j - jOne / 2) = (matrixV(0, j - jOne) + matrixV(iOne, j)) / 2;
              }
              if (j == jOne) {
                matrixV(i - iOne / 2, j - jOne / 2) = (matrixV(i - iOne, 0) + matrixV(i, jOne)) / 2;
              }
              // Note that this leaves a point at the upper left and lower right corners uninitialized.
              // -> Not a big deal.
            }
          }
        }
      }
    }

    iOne = iOne / 2;
    if (iOne < 1) {
      iOne = 1;
    }
    jOne = jOne / 2;
    if (jOne < 1) {
      jOne = 1;
    }
    sumChargeDensity.Clear();
  }
}

/// Solve Poisson's Equation by MultiGrid Technique in 2D (assuming cylindrical symmetry)
///
/// NOTE: In order for this algorithm to work, the number of nRRow and nZColumn must be a power of 2 plus one.
/// So nRRow == 2**M + 1 and nZColumn == 2**N + 1.  The number of nRRow and nZColumn can be different.
///
/// \param matrixV TMatrixD& potential in matrix
/// \param matrixCharge TMatrixD& charge density in matrix (side effect
/// \param nRRow Int_t number of nRRow
/// \param nZColumn Int_t number of nZColumn
/// \param maxIteration Int_t maximum iteration for relaxation method
///
/// \return A fixed number that has nothing to do with what the function does
void AliTPCPoissonSolver::PoissonMultiGrid2D(TMatrixD& matrixV, TMatrixD& matrixCharge, Int_t nRRow, Int_t nZColumn)
{
  /// Geometry of TPC -- should be use AliTPCParams instead
  const Float_t gridSizeR = (AliTPCPoissonSolver::fgkOFCRadius - AliTPCPoissonSolver::fgkIFCRadius) / (nRRow - 1);
  const Float_t gridSizeZ = AliTPCPoissonSolver::fgkTPCZ0 / (nZColumn - 1);
  const Float_t ratio = gridSizeR * gridSizeR / (gridSizeZ * gridSizeZ);

  Int_t nGridRow = 0; // number grid
  Int_t nGridCol = 0; // number grid
  Int_t nnRow;
  Int_t nnCol;

  nnRow = nRRow;
  while (nnRow >>= 1) {
    nGridRow++;
  }

  nnCol = nZColumn;
  while (nnCol >>= 1) {
    nGridCol++;
  }

  //Check that number of nRRow and nZColumn is suitable for multi grid
  if (!IsPowerOfTwo(nRRow - 1)) {
    Error("PoissonMultiGrid2D", "PoissonMultiGrid - Error in the number of nRRow. Must be 2**M - 1");
    return;
  }
  if (!IsPowerOfTwo(nZColumn - 1)) {
    Error("PoissonMultiGrid2D", "PoissonMultiGrid - Error in the number of nZColumn. Must be 2**N - 1");
    return;
  }

  Int_t nLoop = TMath::Max(nGridRow, nGridCol); // Calculate the number of nLoop for the binary expansion

  Info("PoissonMultiGrid2D", "%s", Form("nGridRow=%d, nGridCol=%d, nLoop=%d, nMGCycle=%d", nGridRow, nGridCol, nLoop, fMgParameters.nMGCycle));

  Float_t h, h2, radius;
  Int_t iOne = 1; // in/dex
  Int_t jOne = 1; // index
  Int_t tnRRow = nRRow, tnZColumn = nZColumn;
  Int_t count;
  Float_t tempRatio, tempFourth;

  // Vector for storing multi grid array
  std::vector<TMatrixD*> tvChargeFMG(nLoop);
  std::vector<TMatrixD*> tvArrayV(nLoop);
  std::vector<TMatrixD*> tvCharge(nLoop);
  std::vector<TMatrixD*> tvResidue(nLoop);

  // Allocate memory for temporary grid
  for (count = 1; count <= nLoop; count++) {
    tnRRow = iOne == 1 ? nRRow : nRRow / iOne + 1;
    tnZColumn = jOne == 1 ? nZColumn : nZColumn / jOne + 1;
    // if one just address to matrixV
    tvResidue[count - 1] = new TMatrixD(tnRRow, tnZColumn);
    if (count == 1) {
      tvChargeFMG[count - 1] = &matrixCharge;
      tvArrayV[count - 1] = &matrixV;
      tvCharge[count - 1] = &matrixCharge;
    } else {
      tvArrayV[count - 1] = new TMatrixD(tnRRow, tnZColumn);
      tvCharge[count - 1] = new TMatrixD(tnRRow, tnZColumn);
      tvChargeFMG[count - 1] = new TMatrixD(tnRRow, tnZColumn);
      Restrict2D(*tvChargeFMG[count - 1], *tvChargeFMG[count - 2], tnRRow, tnZColumn);
    }
    iOne = 2 * iOne;
    jOne = 2 * jOne;
  }

  /// full multi grid
  if (fMgParameters.cycleType == kFCycle) {

    Info("PoissonMultiGrid2D", "Do full cycle");
    // FMG
    // 1) Relax on the coarsest grid
    iOne = iOne / 2;
    jOne = jOne / 2;
    tnRRow = iOne == 1 ? nRRow : nRRow / iOne + 1;
    tnZColumn = jOne == 1 ? nZColumn : nZColumn / jOne + 1;
    h = gridSizeR * count;
    h2 = h * h;
    tempRatio = ratio * iOne * iOne / (jOne * jOne);
    tempFourth = 1.0 / (2.0 + 2.0 * tempRatio);

    std::vector<float> coefficient1(tnRRow);
    std::vector<float> coefficient2(tnRRow);

    for (Int_t i = 1; i < tnRRow - 1; i++) {
      radius = AliTPCPoissonSolver::fgkIFCRadius + i * h;
      coefficient1[i] = 1.0 + h / (2 * radius);
      coefficient2[i] = 1.0 - h / (2 * radius);
    }

    Relax2D(*tvArrayV[nLoop - 1], *tvChargeFMG[nLoop - 1], tnRRow, tnZColumn, h2, tempFourth, tempRatio, coefficient1,
            coefficient2);

    // Do VCycle from nLoop H to h
    for (count = nLoop - 2; count >= 0; count--) {

      iOne = iOne / 2;
      jOne = jOne / 2;

      tnRRow = iOne == 1 ? nRRow : nRRow / iOne + 1;
      tnZColumn = jOne == 1 ? nZColumn : nZColumn / jOne + 1;

      Interp2D(*tvArrayV[count], *tvArrayV[count + 1], tnRRow, tnZColumn);
      // Copy the relax charge to the tvCharge
      *tvCharge[count] = *tvChargeFMG[count]; //copy
      //tvCharge[count]->Print();
      // Do V cycle

      for (Int_t mgCycle = 0; mgCycle < fMgParameters.nMGCycle; mgCycle++) {

        VCycle2D(nRRow, nZColumn, count + 1, nLoop, fMgParameters.nPre, fMgParameters.nPost, gridSizeR, ratio, tvArrayV,
                 tvCharge, tvResidue);
      }
    }
  } else if (fMgParameters.cycleType == kVCycle) {
    // 2. VCycle
    Info("PoissonMultiGrid2D", "Do V cycle");

    Int_t gridFrom = 1;
    Int_t gridTo = nLoop;

    // Do MGCycle
    for (Int_t mgCycle = 0; mgCycle < fMgParameters.nMGCycle; mgCycle++) {
      VCycle2D(nRRow, nZColumn, gridFrom, gridTo, fMgParameters.nPre, fMgParameters.nPost, gridSizeR, ratio, tvArrayV,
               tvCharge, tvResidue);
    }
  } else if (fMgParameters.cycleType == kWCycle) {

    // 3. W Cycle (TODO:)

    Int_t gridFrom = 1;

    //nLoop = nLoop >= 4 ? 4 : nLoop;

    Int_t gridTo = nLoop;
    //Int_t gamma = 1;

    // Do MGCycle
    for (Int_t mgCycle = 0; mgCycle < fMgParameters.nMGCycle; mgCycle++) {
      WCycle2D(nRRow, nZColumn, gridFrom, gridTo, fMgParameters.gamma, fMgParameters.nPre, fMgParameters.nPost,
               gridSizeR, ratio, tvArrayV, tvCharge, tvResidue);
    }
  }

  // Deallocate memory
  for (count = nLoop; count >= 1; count--) {
    // if one just address to matrixV
    if (count > 1) {
      delete tvArrayV[count - 1];
      delete tvCharge[count - 1];
      delete tvChargeFMG[count - 1];
    }
    delete tvResidue[count - 1];
  }
}

/// 3D - Solve Poisson's Equation in 3D by Relaxation Technique
///
///    NOTE: In order for this algorithm to work, the number of nRRow and nZColumn must be a power of 2 plus one.
///    The number of nRRow and Z Column can be different.
///
///    R Row       ==  2**M + 1
///    Z Column  ==  2**N + 1
///    Phi Slice  ==  Arbitrary but greater than 3
///
///    DeltaPhi in Radians
///
///    SYMMETRY = 0 if no phi symmetries, and no phi boundary conditions
/// = 1 if we have reflection symmetry at the boundaries (eg. sector symmetry or half sector symmetries).
///
/// \param matricesV TMatrixD** potential in 3D matrix
/// \param matricesCharge TMatrixD** charge density in 3D matrix (side effect)
/// \param nRRow Int_t number of nRRow in the r direction of TPC
/// \param nZColumn Int_t number of nZColumn in z direction of TPC
/// \param phiSlice Int_t number of phiSlice in phi direction of T{C
/// \param maxIteration Int_t maximum iteration for relaxation method
/// \param symmetry Int_t symmetry or not
///
void AliTPCPoissonSolver::PoissonRelaxation3D(TMatrixD** matricesV, TMatrixD** matricesCharge,
                                              Int_t nRRow, Int_t nZColumn, Int_t phiSlice, Int_t maxIteration,
                                              Int_t symmetry)
{

  const Float_t gridSizeR = (AliTPCPoissonSolver::fgkOFCRadius - AliTPCPoissonSolver::fgkIFCRadius) / (nRRow - 1);
  const Float_t gridSizePhi = TMath::TwoPi() / phiSlice;
  const Float_t gridSizeZ = AliTPCPoissonSolver::fgkTPCZ0 / (nZColumn - 1);
  const Float_t ratioPhi = gridSizeR * gridSizeR / (gridSizePhi * gridSizePhi);
  const Float_t ratioZ = gridSizeR * gridSizeR / (gridSizeZ * gridSizeZ);

  Info("PoissonRelaxation3D", "%s", Form("in Poisson Solver 3D relaxation nRRow=%d, cols=%d, phiSlice=%d \n", nRRow, nZColumn, phiSlice));
  // Check that the number of nRRow and nZColumn is suitable for a binary expansion
  if (!IsPowerOfTwo((nRRow - 1))) {
    Error("PoissonRelaxation3D", "Poisson3DRelaxation - Error in the number of nRRow. Must be 2**M - 1");
    return;
  }
  if (!IsPowerOfTwo((nZColumn - 1))) {
    Error("PoissonRelaxation3D", "Poisson3DRelaxation - Error in the number of nZColumn. Must be 2**N - 1");
    return;
  }
  if (phiSlice <= 3) {
    Error("PoissonRelaxation3D", "Poisson3DRelaxation - Error in the number of phiSlice. Must be larger than 3");
    return;
  }
  if (phiSlice > 1000) {
    Error("PoissonRelaxation3D", "Poisson3D  phiSlice > 1000 is not allowed (nor wise) ");
    return;
  }

  // Solve Poisson's equation in cylindrical coordinates by relaxation technique
  // Allow for different size grid spacing in R and Z directions
  // Use a binary expansion of the matrix to speed up the solution of the problem

  Int_t nLoop, mPlus, mMinus, signPlus, signMinus;
  Int_t iOne = (nRRow - 1) / 4;
  Int_t jOne = (nZColumn - 1) / 4;
  nLoop = TMath::Max(iOne, jOne);                      // Calculate the number of nLoop for the binary expansion
  nLoop = 1 + (int)(0.5 + TMath::Log2((double)nLoop)); // Solve for N in 2**N

  TMatrixD* matricesSumChargeDensity[1000]; // Create temporary arrays to store low resolution charge arrays

  std::vector<float> coefficient1(
    nRRow); // Do this the standard C++ way to avoid gcc extensions for Float_t coefficient1[nRRow]
  std::vector<float> coefficient2(
    nRRow); // Do this the standard C++ way to avoid gcc extensions for Float_t coefficient1[nRRow]
  std::vector<float> coefficient3(
    nRRow); // Do this the standard C++ way to avoid gcc extensions for Float_t coefficient1[nRRow]
  std::vector<float> coefficient4(
    nRRow);                                        // Do this the standard C++ way to avoid gcc extensions for Float_t coefficient1[nRRow]
  std::vector<float> overRelaxCoefficient4(nRRow); // Do this the standard C++ way to avoid gcc extensions
  std::vector<float> overRelaxCoefficient5(nRRow); // Do this the standard C++ way to avoid gcc extensions
  for (Int_t i = 0; i < phiSlice; i++) {
    matricesSumChargeDensity[i] = new TMatrixD(nRRow, nZColumn);
  }

  ///// Test of Convergence
  TMatrixD* prevArrayV[phiSlice];

  for (Int_t m = 0; m < phiSlice; m++) {
    prevArrayV[m] = new TMatrixD(nRRow, nZColumn);
  }
  /////

  // START the master loop and do the binary expansion
  for (Int_t count = 0; count < nLoop; count++) {
    Float_t tempGridSizeR = gridSizeR * iOne;
    Float_t tempRatioPhi = ratioPhi * iOne * iOne;
    Float_t tempRatioZ = ratioZ * iOne * iOne / (jOne * jOne);

    for (Int_t i = iOne; i < nRRow - 1; i += iOne) {
      Float_t radius = AliTPCPoissonSolver::fgkIFCRadius + i * gridSizeR;
      coefficient1[i] = 1.0 + tempGridSizeR / (2 * radius);
      coefficient2[i] = 1.0 - tempGridSizeR / (2 * radius);
      coefficient3[i] = tempRatioPhi / (radius * radius);
      coefficient4[i] = 0.5 / (1.0 + tempRatioZ + coefficient3[i]);
    }

    for (Int_t m = 0; m < phiSlice; m++) {
      TMatrixD& matrixCharge = *matricesCharge[m];
      TMatrixD& sumChargeDensity = *matricesSumChargeDensity[m];
      for (Int_t i = iOne; i < nRRow - 1; i += iOne) {
        Float_t radius = AliTPCPoissonSolver::fgkIFCRadius + i * gridSizeR;
        for (Int_t j = jOne; j < nZColumn - 1; j += jOne) {
          if (iOne == 1 && jOne == 1) {
            sumChargeDensity(i, j) = matrixCharge(i, j);
          } else { // Add up all enclosed charge density contributions within 1/2 unit in all directions
            Float_t weight = 0.0;
            Float_t sum = 0.0;
            sumChargeDensity(i, j) = 0.0;
            for (Int_t ii = i - iOne / 2; ii <= i + iOne / 2; ii++) {
              for (Int_t jj = j - jOne / 2; jj <= j + jOne / 2; jj++) {
                if (ii == i - iOne / 2 || ii == i + iOne / 2 || jj == j - jOne / 2 || jj == j + jOne / 2) {
                  weight = 0.5;
                } else {
                  weight = 1.0;
                }
                sumChargeDensity(i, j) += matrixCharge(ii, jj) * weight * radius;
                sum += weight * radius;
              }
            }
            sumChargeDensity(i, j) /= sum;
          }
          sumChargeDensity(i, j) *= tempGridSizeR * tempGridSizeR; // just saving a step later on
        }
      }
    }

    for (Int_t k = 1; k <= maxIteration; k++) {
      if (count == nLoop - 1) {
        //// Test of Convergence
        for (Int_t m = 0; m < phiSlice; m++) {
          (*prevArrayV[m]) = (*matricesV[m]);
        }
        ////
      }

      Float_t overRelax = 1.0 + TMath::Sqrt(TMath::Cos((k * TMath::PiOver2()) / maxIteration));
      Float_t overRelaxM1 = overRelax - 1.0;

      for (Int_t i = iOne; i < nRRow - 1; i += iOne) {
        overRelaxCoefficient4[i] = overRelax * coefficient4[i];
        overRelaxCoefficient5[i] = overRelaxM1 / overRelaxCoefficient4[i];
      }

      for (Int_t m = 0; m < phiSlice; m++) {
        mPlus = m + 1;
        signPlus = 1;
        mMinus = m - 1;
        signMinus = 1;
        if (symmetry == 1) { // Reflection symmetry in phi (e.g. symmetry at sector boundaries, or half sectors, etc.)
          if (mPlus > phiSlice - 1) {
            mPlus = phiSlice - 2;
          }
          if (mMinus < 0) {
            mMinus = 1;
          }
        } else if (symmetry == -1) { // Anti-symmetry in phi
          if (mPlus > phiSlice - 1) {
            mPlus = phiSlice - 2;
            signPlus = -1;
          }
          if (mMinus < 0) {
            mMinus = 1;
            signMinus = -1;
          }
        } else { // No Symmetries in phi, no boundaries, the calculation is continuous across all phi
          if (mPlus > phiSlice - 1) {
            mPlus = m + 1 - phiSlice;
          }
          if (mMinus < 0) {
            mMinus = m - 1 + phiSlice;
          }
        }

        TMatrixD& matrixV = *matricesV[m];
        TMatrixD& matrixVP = *matricesV[mPlus];
        TMatrixD& matrixVM = *matricesV[mMinus];
        TMatrixD& sumChargeDensity = *matricesSumChargeDensity[m];
        Double_t* matrixVFast = matrixV.GetMatrixArray();
        Double_t* matrixVPFast = matrixVP.GetMatrixArray();
        Double_t* matrixVMFast = matrixVM.GetMatrixArray();
        Double_t* sumChargeDensityFast = sumChargeDensity.GetMatrixArray();

        if (fStrategy == kRelaxation) {
          // slow implementation
          for (Int_t i = iOne; i < nRRow - 1; i += iOne) {
            for (Int_t j = jOne; j < nZColumn - 1; j += jOne) {

              matrixV(i, j) = (coefficient2[i] * matrixV(i - iOne, j) + tempRatioZ * (matrixV(i, j - jOne) + matrixV(i, j + jOne)) - overRelaxCoefficient5[i] * matrixV(i, j) + coefficient1[i] * matrixV(i + iOne, j) + coefficient3[i] * (signPlus * matrixVP(i, j) + signMinus * matrixVM(i, j)) + sumChargeDensity(i, j)) * overRelaxCoefficient4[i];
              // Note: over-relax the solution at each step.  This speeds up the convergence.
            }
          }
        } else {
          for (Int_t i = iOne; i < nRRow - 1; i += iOne) {
            Double_t* matrixVFastI = &(matrixVFast[i * nZColumn]);
            Double_t* matrixVPFastI = &(matrixVPFast[i * nZColumn]);
            Double_t* matrixVMFastI = &(matrixVMFast[i * nZColumn]);
            Double_t* sumChargeDensityFastI = &(sumChargeDensityFast[i * nZColumn]);

            for (Int_t j = jOne; j < nZColumn - 1; j += jOne) {
              Double_t /*resSlow*/ resFast;

              resFast = (coefficient2[i] * matrixVFastI[j - nZColumn * iOne] + tempRatioZ * (matrixVFastI[j - jOne] + matrixVFastI[j + jOne]) - overRelaxCoefficient5[i] * matrixVFastI[j] + coefficient1[i] * matrixVFastI[j + nZColumn * iOne] + coefficient3[i] * (signPlus * matrixVPFastI[j] + signMinus * matrixVMFastI[j]) + sumChargeDensityFastI[j]) * overRelaxCoefficient4[i];
              matrixVFastI[j] = resFast;
              // Note: over-relax the solution at each step.  This speeds up the convergence.
            } // end j
          }   //end i
        }     // end phi

        // After full solution is achieved, copy low resolution solution into higher res array
        if (k == maxIteration) {
          for (Int_t i = iOne; i < nRRow - 1; i += iOne) {
            for (Int_t j = jOne; j < nZColumn - 1; j += jOne) {

              if (iOne > 1) {
                matrixV(i + iOne / 2, j) = (matrixV(i + iOne, j) + matrixV(i, j)) / 2;
                if (i == iOne) {
                  matrixV(i - iOne / 2, j) = (matrixV(0, j) + matrixV(iOne, j)) / 2;
                }
              }
              if (jOne > 1) {
                matrixV(i, j + jOne / 2) = (matrixV(i, j + jOne) + matrixV(i, j)) / 2;
                if (j == jOne) {
                  matrixV(i, j - jOne / 2) = (matrixV(i, 0) + matrixV(i, jOne)) / 2;
                }
              }
              if (iOne > 1 && jOne > 1) {
                matrixV(i + iOne / 2, j + jOne / 2) = (matrixV(i + iOne, j + jOne) + matrixV(i, j)) / 2;
                if (i == iOne) {
                  matrixV(i - iOne / 2, j - jOne / 2) = (matrixV(0, j - jOne) + matrixV(iOne, j)) / 2;
                }
                if (j == jOne) {
                  matrixV(i - iOne / 2, j - jOne / 2) = (matrixV(i - iOne, 0) + matrixV(i, jOne)) / 2;
                }
                // Note that this leaves a point at the upper left and lower right corners uninitialized. Not a big deal.
              }
            }
          }
        }
      }

      if (count == nLoop - 1) {

        (*fErrorConvergenceNormInf)(k - 1) = GetConvergenceError(matricesV, prevArrayV, phiSlice);
        (*fError)(k - 1) = GetExactError(matricesV, prevArrayV, phiSlice);

        // if error already achieved then stop mg iteration
        fIterations = k - 1;
        if ((*fErrorConvergenceNormInf)(k - 1) <= fgConvergenceError) {
          Info("PoissonRelaxation3D", "%s", Form("Exact Err: %f, Iteration : %d", (*fError)(k - 1), k - 1));
          break;
        }
        if (k == maxIteration) {
          Info("PoissonRelaxation3D", "%s", Form("Exact Err: %f, Iteration : %d", (*fError)(k - 1), k - 1));
        }
      }
    }

    iOne = iOne / 2;
    if (iOne < 1) {
      iOne = 1;
    }
    jOne = jOne / 2;
    if (jOne < 1) {
      jOne = 1;
    }
  }

  for (Int_t k = 0; k < phiSlice; k++) {
    matricesSumChargeDensity[k]->Delete();
  }

  for (Int_t m = 0; m < phiSlice; m++) {
    delete prevArrayV[m];
  }
}

/// 3D - Solve Poisson's Equation in 3D by MultiGrid with constant phi slices
///
///    NOTE: In order for this algorithm to work, the number of nRRow and nZColumn must be a power of 2 plus one.
///    The number of nRRow and Z Column can be different.
///
///    R Row       ==  2**M + 1
///    Z Column  ==  2**N + 1
///    Phi Slice  ==  Arbitrary but greater than 3
///
///		 Solving: \f$  \nabla^{2}V(r,\phi,z) = - f(r,\phi,z) \f$
///
/// Algorithm for MultiGrid Full Cycle (FMG)
/// - Relax on the coarsest grid
/// - Do from coarsest to finest
///     - Interpolate potential from coarse -> fine
///   - Do V-Cycle to the current coarse level to the coarsest
///   - Stop if converged
///
/// DeltaPhi in Radians
/// \param matricesV TMatrixD** potential in 3D matrix \f$ V(r,\phi,z) \f$
/// \param matricesCharge TMatrixD** charge density in 3D matrix (side effect) \f$ - f(r,\phi,z) \f$
/// \param nRRow Int_t number of nRRow in the r direction of TPC
/// \param nZColumn Int_t number of nZColumn in z direction of TPC
/// \param phiSlice Int_t number of phiSlice in phi direction of T{C
/// \param maxIteration Int_t maximum iteration for relaxation method (NOT USED)
/// \param symmetry Int_t symmetry (TODO for symmetry = 1)
//
///    SYMMETRY = 0 if no phi symmetries, and no phi boundary condition
///    = 1 if we have reflection symmetry at the boundaries (eg. sector symmetry or half sector symmetries).
///
void AliTPCPoissonSolver::PoissonMultiGrid3D2D(TMatrixD** matricesV, TMatrixD** matricesCharge, Int_t nRRow,
                                               Int_t nZColumn, Int_t phiSlice, Int_t symmetry)
{

  const Float_t gridSizeR =
    (AliTPCPoissonSolver::fgkOFCRadius - AliTPCPoissonSolver::fgkIFCRadius) / (nRRow - 1); // h_{r}
  const Float_t gridSizePhi = TMath::TwoPi() / phiSlice;                                   // h_{phi}
  const Float_t gridSizeZ = AliTPCPoissonSolver::fgkTPCZ0 / (nZColumn - 1);                // h_{z}
  const Float_t ratioPhi =
    gridSizeR * gridSizeR / (gridSizePhi * gridSizePhi);                  // ratio_{phi} = gridSize_{r} / gridSize_{phi}
  const Float_t ratioZ = gridSizeR * gridSizeR / (gridSizeZ * gridSizeZ); // ratio_{Z} = gridSize_{r} / gridSize_{z}

  // error tolerate
  //const Float_t  ERR = 1e-8;
  Double_t convergenceError;

  Info("PoissonMultiGrid3D2D", "%s", Form("in Poisson Solver 3D multiGrid semi coarsening nRRow=%d, cols=%d, phiSlice=%d \n", nRRow, nZColumn, phiSlice));

  // Check that the number of nRRow and nZColumn is suitable for a binary expansion
  if (!IsPowerOfTwo((nRRow - 1))) {
    Error("PoissonMultiGrid3D2D", "Poisson3DMultiGrid - Error in the number of nRRow. Must be 2**M + 1");
    return;
  }
  if (!IsPowerOfTwo((nZColumn - 1))) {
    Error("PoissonMultiGrid3D2D", "Poisson3DMultiGrid - Error in the number of nZColumn. Must be 2**N - 1");
    return;
  }
  if (phiSlice <= 3) {
    Error("PoissonMultiGrid3D2D", "Poisson3DMultiGrid - Error in the number of phiSlice. Must be larger than 3");
    return;
  }
  if (phiSlice > 1000) {
    Error("PoissonMultiGrid3D2D", "Poisson3D  phiSlice > 1000 is not allowed (nor wise) ");
    return;
  }

  // Solve Poisson's equation in cylindrical coordinates by multiGrid technique
  // Allow for different size grid spacing in R and Z directions

  Int_t nGridRow = 0; // number grid
  Int_t nGridCol = 0; // number grid
  Int_t nnRow;
  Int_t nnCol;

  nnRow = nRRow;
  while (nnRow >>= 1) {
    nGridRow++;
  }
  nnCol = nZColumn;
  while (nnCol >>= 1) {
    nGridCol++;
  }

  Int_t nLoop = TMath::Max(nGridRow, nGridCol); // Calculate the number of nLoop for the binary expansion
  nLoop = (nLoop > fMgParameters.maxLoop) ? fMgParameters.maxLoop : nLoop;
  Int_t count;
  Int_t iOne = 1; // index i in gridSize r (original)
  Int_t jOne = 1; // index j in gridSize z (original)
  Int_t tnRRow = nRRow, tnZColumn = nZColumn;
  std::vector<TMatrixD**> tvChargeFMG(nLoop);  // charge is restricted in full multiGrid
  std::vector<TMatrixD**> tvArrayV(nLoop);     // potential <--> error
  std::vector<TMatrixD**> tvCharge(nLoop);     // charge <--> residue
  std::vector<TMatrixD**> tvResidue(nLoop);    // residue calculation
  std::vector<TMatrixD**> tvPrevArrayV(nLoop); // error calculation

  for (count = 1; count <= nLoop; count++) {
    tnRRow = iOne == 1 ? nRRow : nRRow / iOne + 1;
    tnZColumn = jOne == 1 ? nZColumn : nZColumn / jOne + 1;
    tvResidue[count - 1] = new TMatrixD*[phiSlice];
    tvPrevArrayV[count - 1] = new TMatrixD*[phiSlice];
    for (Int_t k = 0; k < phiSlice; k++) {
      tvResidue[count - 1][k] = new TMatrixD(tnRRow, tnZColumn);
      tvPrevArrayV[count - 1][k] = new TMatrixD(tnRRow, tnZColumn);
    }

    // memory for the finest grid is from parameters
    if (count == 1) {
      tvChargeFMG[count - 1] = matricesCharge;
      tvArrayV[count - 1] = matricesV;
      tvCharge[count - 1] = matricesCharge;
    } else {
      // allocate for coarser grid
      tvChargeFMG[count - 1] = new TMatrixD*[phiSlice];
      tvArrayV[count - 1] = new TMatrixD*[phiSlice];
      tvCharge[count - 1] = new TMatrixD*[phiSlice];
      for (Int_t k = 0; k < phiSlice; k++) {
        tvArrayV[count - 1][k] = new TMatrixD(tnRRow, tnZColumn);
        tvCharge[count - 1][k] = new TMatrixD(tnRRow, tnZColumn);
        tvChargeFMG[count - 1][k] = new TMatrixD(tnRRow, tnZColumn);
      }
      Restrict3D(tvChargeFMG[count - 1], tvChargeFMG[count - 2], tnRRow, tnZColumn, phiSlice, phiSlice);
      RestrictBoundary3D(tvArrayV[count - 1], tvArrayV[count - 2], tnRRow, tnZColumn, phiSlice, phiSlice);
    }
    iOne = 2 * iOne; // doubling
    jOne = 2 * jOne; // doubling
  }
  Float_t h, h2, radius;
  Float_t tempRatioPhi, tempRatioZ;
  std::vector<float> coefficient1(
    nRRow); // coefficient1(nRRow) for storing (1 + h_{r}/2r_{i}) from central differences in r direction
  std::vector<float> coefficient2(
    nRRow); // coefficient2(nRRow) for storing (1 + h_{r}/2r_{i}) from central differences in r direction
  std::vector<float> coefficient3(
    nRRow);                                      // coefficient3(nRRow) for storing (1/r_{i}^2) from central differences in phi direction
  std::vector<float> coefficient4(nRRow);        // coefficient4(nRRow) for storing  1/2
  std::vector<float> inverseCoefficient4(nRRow); // inverse of coefficient4(nRRow)

  // Case full multi grid (FMG)
  if (fMgParameters.cycleType == kFCycle) {

    // 1) Relax on the coarsest grid
    iOne = iOne / 2;
    jOne = jOne / 2;
    tnRRow = iOne == 1 ? nRRow : nRRow / iOne + 1;
    tnZColumn = jOne == 1 ? nZColumn : nZColumn / jOne + 1;

    h = gridSizeR * iOne;
    h2 = h * h;

    tempRatioPhi = ratioPhi * iOne * iOne; // Used tobe divided by ( m_one * m_one ) when m_one was != 1
    tempRatioZ = ratioZ * iOne * iOne / (jOne * jOne);

    for (Int_t i = 1; i < tnRRow - 1; i++) {
      radius = AliTPCPoissonSolver::fgkIFCRadius + i * h;
      coefficient1[i] = 1.0 + h / (2 * radius);
      coefficient2[i] = 1.0 - h / (2 * radius);
      coefficient3[i] = tempRatioPhi / (radius * radius);
      coefficient4[i] = 0.5 / (1.0 + tempRatioZ + coefficient3[i]);
    }
    // relax on the coarsest level
    Relax3D(tvArrayV[nLoop - 1], tvChargeFMG[nLoop - 1], tnRRow, tnZColumn, phiSlice, symmetry, h2, tempRatioZ,
            coefficient1,
            coefficient2, coefficient3, coefficient4);
    // 2) Do multiGrid v-cycle from coarsest to finest
    for (count = nLoop - 2; count >= 0; count--) {
      // move to finer grid
      iOne = iOne / 2;
      jOne = jOne / 2;
      tnRRow = iOne == 1 ? nRRow : nRRow / iOne + 1;
      tnZColumn = jOne == 1 ? nZColumn : nZColumn / jOne + 1;
      // 2) a) Interpolate potential for h -> 2h (coarse -> fine)
      Interp3D(tvArrayV[count], tvArrayV[count + 1], tnRRow, tnZColumn, phiSlice, phiSlice);
      // 2) c) Copy the restricted charge to charge for calculation
      for (Int_t m = 0; m < phiSlice; m++) {
        *tvCharge[count][m] = *tvChargeFMG[count][m]; //copy
      }
      // 2) c) Do V cycle fMgParameters.nMGCycle times at most
      for (Int_t mgCycle = 0; mgCycle < fMgParameters.nMGCycle; mgCycle++) {
        // Copy the potential to temp array for convergence calculation
        for (Int_t m = 0; m < phiSlice; m++) {
          *tvPrevArrayV[count][m] = *tvArrayV[count][m]; //copy
        }
        // 2) c) i) Call V cycle from grid count+1 (current fine level) to nLoop (coarsest)
        VCycle3D2D(nRRow, nZColumn, phiSlice, symmetry, count + 1, nLoop, fMgParameters.nPre, fMgParameters.nPost,
                   gridSizeR, ratioZ, ratioPhi, tvArrayV, tvCharge, tvResidue, coefficient1, coefficient2, coefficient3,
                   coefficient4, inverseCoefficient4);

        convergenceError = GetConvergenceError(tvArrayV[count], tvPrevArrayV[count], phiSlice);

        if (count == 0) {
          (*fErrorConvergenceNormInf)(mgCycle) = convergenceError;
          (*fError)(mgCycle) = GetExactError(matricesV, tvPrevArrayV[count], phiSlice);
        }
        /// if already converge just break move to finer grid
        if (convergenceError <= fgConvergenceError) {
          fIterations = mgCycle + 1;
          break;
        }
      }
    }
  } // Case V multi grid (VMG)
  else if (fMgParameters.cycleType == kVCycle) {
    Int_t gridFrom = 1;
    Int_t gridTo = nLoop;
    // do v cycle fMgParameters.nMGCycle from the coarsest to finest
    for (Int_t mgCycle = 0; mgCycle < fMgParameters.nMGCycle; mgCycle++) {
      // copy to store previous potential
      for (Int_t m = 0; m < phiSlice; m++) {
        *tvPrevArrayV[0][m] = *tvArrayV[0][m]; //copy
      }
      // Do V Cycle for constant phiSlice
      VCycle3D2D(nRRow, nZColumn, phiSlice, symmetry, gridFrom, gridTo, fMgParameters.nPre, fMgParameters.nPost,
                 gridSizeR, ratioZ, ratioPhi, tvArrayV, tvCharge, tvResidue, coefficient1, coefficient2, coefficient3,
                 coefficient4, inverseCoefficient4);

      // convergence error
      convergenceError = GetConvergenceError(tvArrayV[0], tvPrevArrayV[0], phiSlice);
      (*fErrorConvergenceNormInf)(mgCycle) = convergenceError;
      (*fError)(mgCycle) = GetExactError(matricesV, tvPrevArrayV[0], phiSlice);

      // if error already achieved then stop mg iteration
      if (convergenceError <= fgConvergenceError) {
        fIterations = mgCycle + 1;
        break;
      }
    }
  }
  // Deallocate memory
  for (count = 1; count <= nLoop; count++) {
    delete[] tvResidue[count - 1];
    delete[] tvPrevArrayV[count - 1];

    if (count > 1) {
      delete[] tvChargeFMG[count - 1];
      delete[] tvArrayV[count - 1];
      delete[] tvCharge[count - 1];
    }
  }
}

/// 3D - Solve Poisson's Equation in 3D in all direction by MultiGrid
///
///    NOTE: In order for this algorithm to work, the number of nRRow and nZColumn must be a power of 2 plus one.
///    The number of nRRow and Z Column can be different.
///
///    R Row       ==  2**M + 1
///    Z Column   ==  2**N + 1
///    Phi Slices  ==  Arbitrary but greater than 3
///
///		 Solving: \f$  \nabla^{2}V(r,\phi,z) = - f(r,\phi,z) \f$
///
///  Algorithm for MultiGrid Full Cycle (FMG)
/// - Relax on the coarsest grid
/// - Do from coarsest to finest
///     - Interpolate potential from coarse -> fine
///   - Do V-Cycle to the current coarse level to the coarsest
///   - Stop if converged
///
///    DeltaPhi in Radians
/// \param matricesV TMatrixD** potential in 3D matrix
/// \param matricesCharge TMatrixD** charge density in 3D matrix (side effect)
/// \param nRRow Int_t number of nRRow in the r direction of TPC
/// \param nZColumn Int_t number of nZColumn in z direction of TPC
/// \param phiSlice Int_t number of phiSlice in phi direction of T{C
/// \param maxIteration Int_t maximum iteration for relaxation method
/// \param symmetry Int_t symmetry or not
//
///    SYMMETRY = 0 if no phi symmetries, and no phi boundary condition
/// = 1 if we have reflection symmetry at the boundaries (eg. sector symmetry or half sector symmetries).
///
void AliTPCPoissonSolver::PoissonMultiGrid3D(TMatrixD** matricesV, TMatrixD** matricesCharge,
                                             Int_t nRRow, Int_t nZColumn, Int_t phiSlice, Int_t symmetry)
{

  const Float_t gridSizeR =
    (AliTPCPoissonSolver::fgkOFCRadius - AliTPCPoissonSolver::fgkIFCRadius) / (nRRow - 1); // h_{r}
  const Float_t gridSizeZ = AliTPCPoissonSolver::fgkTPCZ0 / (nZColumn - 1);                // h_{z}
  const Float_t ratioZ = gridSizeR * gridSizeR / (gridSizeZ * gridSizeZ);                  // ratio_{Z} = gridSize_{r} / gridSize_{z}

  Float_t gridSizePhi = TMath::TwoPi() / phiSlice; // h_{phi}
  Float_t h, h2, radius;
  Float_t tempRatioPhi, tempRatioZ;

  Float_t convergenceError; // Convergence error

  Info("PoissonMultiGrid3D", "%s", Form("in Poisson Solver 3D multi grid full coarsening  nRRow=%d, cols=%d, phiSlice=%d \n", nRRow, nZColumn, phiSlice));

  // Check that the number of nRRow and nZColumn is suitable for a binary expansion
  if (!IsPowerOfTwo((nRRow - 1))) {
    Error("PoissonMultiGrid3D", "Poisson3DMultiGrid - Error in the number of nRRow. Must be 2**M + 1");
    return;
  }
  if (!IsPowerOfTwo((nZColumn - 1))) {
    Error("PoissonMultiGrid3D", "Poisson3DMultiGrid - Error in the number of nZColumn. Must be 2**N - 1");
    return;
  }
  if (phiSlice <= 3) {
    Error("PoissonMultiGrid3D", "Poisson3DMultiGrid - Error in the number of phiSlice. Must be larger than 3");
    return;
  }
  if (phiSlice > 1000) {
    Error("PoissonMultiGrid3D", "Poisson3D  phiSlice > 1000 is not allowed (nor wise) ");
    return;
  }

  // Solve Poisson's equation in cylindrical coordinates by multi grid technique
  // Allow for different size grid spacing in R and Z directions

  Int_t nGridRow = 0; // number grid
  Int_t nGridCol = 0; // number grid
  Int_t nGridPhi = 0;

  Int_t nnRow;
  Int_t nnCol;
  Int_t nnPhi;

  nnRow = nRRow;
  while (nnRow >>= 1) {
    nGridRow++;
  }

  nnCol = nZColumn;
  while (nnCol >>= 1) {
    nGridCol++;
  }

  nnPhi = phiSlice;

  while (nnPhi % 2 == 0) {
    nGridPhi++;
    nnPhi /= 2;
  }

  Info("PoissonMultiGrid3D", "%s", Form("nGridRow=%d, nGridCol=%d, nGridPhi=%d", nGridRow, nGridCol, nGridPhi));
  Int_t nLoop = TMath::Max(nGridRow, nGridCol); // Calculate the number of nLoop for the binary expansion
  nLoop = TMath::Max(nLoop, nGridPhi);

  // Vector for storing multi grid array
  Int_t iOne = 1; // index i in gridSize r (original)
  Int_t jOne = 1; // index j in gridSize z (original)
  Int_t kOne = 1; // index k in gridSize phi
  Int_t tnRRow = nRRow, tnZColumn = nZColumn, tPhiSlice = phiSlice, otPhiSlice;

  // 1)	Memory allocation for multi grid
  std::vector<TMatrixD**> tvChargeFMG(nLoop);  // charge is restricted in full multiGrid
  std::vector<TMatrixD**> tvArrayV(nLoop);     // potential <--> error
  std::vector<TMatrixD**> tvCharge(nLoop);     // charge <--> residue
  std::vector<TMatrixD**> tvResidue(nLoop);    // residue calculation
  std::vector<TMatrixD**> tvPrevArrayV(nLoop); // error calculation

  // these vectors for storing the coefficients in smoother
  std::vector<float> coefficient1(
    nRRow); // coefficient1(nRRow) for storing (1 + h_{r}/2r_{i}) from central differences in r direction
  std::vector<float> coefficient2(
    nRRow); // coefficient2(nRRow) for storing (1 + h_{r}/2r_{i}) from central differences in r direction
  std::vector<float> coefficient3(
    nRRow);                                      // coefficient3(nRRow) for storing (1/r_{i}^2) from central differences in phi direction
  std::vector<float> coefficient4(nRRow);        // coefficient4(nRRow) for storing  1/2
  std::vector<float> inverseCoefficient4(nRRow); // inverse of coefficient4(nRRow)

  for (Int_t count = 1; count <= nLoop; count++) {

    // tnRRow,tnZColumn in new grid
    tnRRow = iOne == 1 ? nRRow : nRRow / iOne + 1;
    tnZColumn = jOne == 1 ? nZColumn : nZColumn / jOne + 1;
    tPhiSlice = kOne == 1 ? phiSlice : phiSlice / kOne;
    tPhiSlice = tPhiSlice < nnPhi ? nnPhi : tPhiSlice;

    // allocate memory for residue
    tvResidue[count - 1] = new TMatrixD*[tPhiSlice];
    tvPrevArrayV[count - 1] = new TMatrixD*[tPhiSlice];
    for (Int_t k = 0; k < tPhiSlice; k++) {
      tvResidue[count - 1][k] = new TMatrixD(tnRRow, tnZColumn);
      tvPrevArrayV[count - 1][k] = new TMatrixD(tnRRow, tnZColumn);
    }

    // memory for the finest grid is from parameters
    if (count == 1) {
      tvChargeFMG[count - 1] = matricesCharge;
      tvArrayV[count - 1] = matricesV;
      tvCharge[count - 1] = matricesCharge;
    } else {
      // allocate for coarser grid
      tvChargeFMG[count - 1] = new TMatrixD*[tPhiSlice];
      tvArrayV[count - 1] = new TMatrixD*[tPhiSlice];
      tvCharge[count - 1] = new TMatrixD*[tPhiSlice];
      for (Int_t k = 0; k < tPhiSlice; k++) {
        tvArrayV[count - 1][k] = new TMatrixD(tnRRow, tnZColumn);
        tvCharge[count - 1][k] = new TMatrixD(tnRRow, tnZColumn);
        tvChargeFMG[count - 1][k] = new TMatrixD(tnRRow, tnZColumn);
      }
    }
    iOne = 2 * iOne; // doubling
    jOne = 2 * jOne; // doubling
    kOne = 2 * kOne;
  }

  // Case full multi grid (FMG)

  if (fMgParameters.cycleType == kFCycle) {
    // Restrict the charge to coarser grid
    iOne = 2;
    jOne = 2;
    kOne = 2;
    otPhiSlice = phiSlice;

    // 1) Restrict Charge and Boundary to coarser grid
    for (Int_t count = 2; count <= nLoop; count++) {
      // tnRRow,tnZColumn in new grid
      tnRRow = iOne == 1 ? nRRow : nRRow / iOne + 1;
      tnZColumn = jOne == 1 ? nZColumn : nZColumn / jOne + 1;
      tPhiSlice = kOne == 1 ? phiSlice : phiSlice / kOne;
      tPhiSlice = tPhiSlice < nnPhi ? nnPhi : tPhiSlice;

      Info("PoissonMultiGrid3D", "%s", Form("Restrict3D, tnRRow=%d, tnZColumn=%d, newPhiSlice=%d, oldPhiSlice=%d\n", tnRRow, tnZColumn, tPhiSlice, otPhiSlice));
      Restrict3D(tvChargeFMG[count - 1], tvChargeFMG[count - 2], tnRRow, tnZColumn, tPhiSlice, otPhiSlice);
      // copy boundary values of V
      RestrictBoundary3D(tvArrayV[count - 1], tvArrayV[count - 2], tnRRow, tnZColumn, tPhiSlice, otPhiSlice);
      otPhiSlice = tPhiSlice;

      iOne = 2 * iOne; // doubling
      jOne = 2 * jOne; // doubling
      kOne = 2 * kOne;
    }

    // Relax on the coarsest grid
    // FMG
    // 2) Relax on the coarsest grid

    // move to the coarsest + 1
    iOne = iOne / 2;
    jOne = jOne / 2;
    kOne = kOne / 2;
    tnRRow = iOne == 1 ? nRRow : nRRow / iOne + 1;
    tnZColumn = jOne == 1 ? nZColumn : nZColumn / jOne + 1;
    tPhiSlice = kOne == 1 ? phiSlice : phiSlice / kOne;
    tPhiSlice = tPhiSlice < nnPhi ? nnPhi : tPhiSlice;
    otPhiSlice = tPhiSlice;

    h = gridSizeR * iOne;
    h2 = h * h;
    gridSizePhi = TMath::TwoPi() / tPhiSlice;           // h_{phi}
    tempRatioPhi = h * h / (gridSizePhi * gridSizePhi); // ratio_{phi} = gridSize_{r} / gridSize_{phi}
    tempRatioZ = ratioZ * iOne * iOne / (jOne * jOne);
    for (Int_t i = 1; i < tnRRow - 1; i++) {
      radius = AliTPCPoissonSolver::fgkIFCRadius + i * h;
      coefficient1[i] = 1.0 + h / (2 * radius);
      coefficient2[i] = 1.0 - h / (2 * radius);
      coefficient3[i] = tempRatioPhi / (radius * radius);
      coefficient4[i] = 0.5 / (1.0 + tempRatioZ + coefficient3[i]);
    }
    // 3) Relax on the coarsest grid
    Relax3D(tvArrayV[nLoop - 1], tvChargeFMG[nLoop - 1], tnRRow, tnZColumn, tPhiSlice, symmetry, h2, tempRatioZ,
            coefficient1,
            coefficient2, coefficient3, coefficient4);

    // 4) V Cycle from coarsest to finest
    for (Int_t count = nLoop - 2; count >= 0; count--) {
      // move to finer grid
      coefficient1.clear();
      coefficient2.clear();
      coefficient3.clear();
      coefficient4.clear();
      inverseCoefficient4.clear();

      iOne = iOne / 2;
      jOne = jOne / 2;
      kOne = kOne / 2;

      tnRRow = iOne == 1 ? nRRow : nRRow / iOne + 1;
      tnZColumn = jOne == 1 ? nZColumn : nZColumn / jOne + 1;
      tPhiSlice = kOne == 1 ? phiSlice : phiSlice / kOne;
      tPhiSlice = tPhiSlice < nnPhi ? nnPhi : tPhiSlice;
      // 4) a) interpolate from 2h --> h grid
      Interp3D(tvArrayV[count], tvArrayV[count + 1], tnRRow, tnZColumn, tPhiSlice, otPhiSlice);

      // Copy the relax charge to the tvCharge
      if (count > 0) {
        for (Int_t m = 0; m < tPhiSlice; m++) {
          *tvCharge[count][m] = *tvChargeFMG[count][m]; //copy
        }
      }
      for (Int_t mgCycle = 0; mgCycle < fMgParameters.nMGCycle; mgCycle++) {
        // copy to store previous potential
        for (Int_t m = 0; m < tPhiSlice; m++) {
          *tvPrevArrayV[count][m] = *tvArrayV[count][m]; //copy
        }

        VCycle3D(nRRow, nZColumn, phiSlice, symmetry, count + 1, nLoop, fMgParameters.nPre, fMgParameters.nPost,
                 gridSizeR, ratioZ, tvArrayV,
                 tvCharge, tvResidue, coefficient1, coefficient2, coefficient3, coefficient4, inverseCoefficient4);

        /// converge error
        convergenceError = GetConvergenceError(tvArrayV[count], tvPrevArrayV[count], tPhiSlice);
        //// error counting /////
        if (count == 0) {
          (*fErrorConvergenceNormInf)(mgCycle) = convergenceError;
          (*fError)(mgCycle) = GetExactError(matricesV, tvPrevArrayV[count], phiSlice);
        }
        /// if already converge just break move to finer grid
        if (convergenceError <= fgConvergenceError) {
          fIterations = mgCycle + 1;
          break;
        }
      }
      // keep old slice information
      otPhiSlice = tPhiSlice;
    }

  } else if (fMgParameters.cycleType == kVCycle) {
    // V-cycle
    Int_t gridFrom = 1;
    Int_t gridTo = nLoop;

    for (Int_t mgCycle = 0; mgCycle < fMgParameters.nMGCycle; mgCycle++) {
      // copy to store previous potential
      for (Int_t m = 0; m < phiSlice; m++) {
        *tvPrevArrayV[0][m] = *tvArrayV[0][m]; //copy
      }
      // Do V Cycle from the coarsest to finest grid
      VCycle3D(nRRow, nZColumn, phiSlice, symmetry, gridFrom, gridTo, fMgParameters.nPre, fMgParameters.nPost,
               gridSizeR, ratioZ, tvArrayV, tvCharge, tvResidue,
               coefficient1, coefficient2, coefficient3, coefficient4, inverseCoefficient4);
      // convergence error
      convergenceError = GetConvergenceError(tvArrayV[0], tvPrevArrayV[0], phiSlice);
      (*fErrorConvergenceNormInf)(mgCycle) = convergenceError;
      (*fError)(mgCycle) = GetExactError(matricesV, tvPrevArrayV[0], phiSlice);
      // if error already achieved then stop mg iteration
      if (convergenceError <= fgConvergenceError) {
        //Info("PoissonMultiGrid3D",Form("Exact Err: %f, MG Iteration : %d", (*fError)(mgCycle), mgCycle));
        fIterations = mgCycle + 1;
        break;
      }
    }
  }
  // deallocate memory for multiGrid
  for (Int_t count = 1; count <= nLoop; count++) {
    delete[] tvResidue[count - 1];
    delete[] tvPrevArrayV[count - 1];
    if (count > 1) {
      delete[] tvChargeFMG[count - 1];
      delete[] tvArrayV[count - 1];
      delete[] tvCharge[count - 1];
    }
  }
}

/// Helper function to check if the integer is equal to a power of two
/// \param i Int_t the number
/// \return 1 if it is a power of two, else 0
Int_t AliTPCPoissonSolver::IsPowerOfTwo(Int_t i) const
{
  Int_t j = 0;
  while (i > 0) {
    j += (i & 1);
    i = (i >> 1);
  }
  if (j == 1) {
    return (1); // True
  }
  return (0); // False
}

/// Relax3D
///
///    Relaxation operation for multiGrid
///		 relaxation used 7 stencil in cylindrical coordinate
///
/// Using the following equations
/// \f$ U_{i,j,k} = (1 + \frac{1}{r_{i}h_{r}}) U_{i+1,j,k}  + (1 - \frac{1}{r_{i}h_{r}}) U_{i+1,j,k}  \f$
///
/// \param matricesCurrentV TMatrixD** potential in 3D (matrices of matrix)
/// \param matricesCurrentCharge TMatrixD** charge in 3D
/// \param nRRow const Int_t number of nRRow in the r direction of TPC
/// \param nZColumn const Int_t number of nZColumn in z direction of TPC
/// \param phiSlice const Int_t number of phiSlice in phi direction of TPC
/// \param symmetry const Int_t is the cylinder has symmetry
/// \param h2 const Float_t \f$  h_{r}^{2} \f$
/// \param tempRatioZ const Float_t ration between grid size in z-direction and r-direction
/// \param coefficient1 std::vector<float> coefficient for \f$  V_{x+1,y,z} \f$
/// \param coefficient2 std::vector<float> coefficient for \f$  V_{x-1,y,z} \f$
/// \param coefficient3 std::vector<float> coefficient for z
/// \param coefficient4 std::vector<float> coefficient for f(r,\phi,z)
///
void AliTPCPoissonSolver::Relax3D(TMatrixD** matricesCurrentV, TMatrixD** matricesCurrentCharge, const Int_t tnRRow,
                                  const Int_t tnZColumn,
                                  const Int_t phiSlice, const Int_t symmetry, const Float_t h2,
                                  const Float_t tempRatioZ, std::vector<float>& coefficient1,
                                  std::vector<float>& coefficient2,
                                  std::vector<float>& coefficient3, std::vector<float>& coefficient4)
{

  Int_t mPlus, mMinus, signPlus, signMinus;
  TMatrixD* matrixV;
  TMatrixD* matrixVP;
  TMatrixD* matrixVM;
  TMatrixD* arrayCharge;

  // Gauss-Seidel (Read Black}
  if (fMgParameters.relaxType == kGaussSeidel) {
    // for each slice
    Int_t isw, jsw, msw;
    msw = 1;
    for (Int_t iPass = 1; iPass <= 2; iPass++, msw = 3 - msw) {
      jsw = msw;
      for (Int_t m = 0; m < phiSlice; m++, jsw = 3 - jsw) {
        mPlus = m + 1;
        signPlus = 1;
        mMinus = m - 1;
        signMinus = 1;
        // Reflection symmetry in phi (e.g. symmetry at sector boundaries, or half sectors, etc.)
        if (symmetry == 1) {
          if (mPlus > phiSlice - 1) {
            mPlus = phiSlice - 2;
          }
          if (mMinus < 0) {
            mMinus = 1;
          }
        }
        // Anti-symmetry in phi
        else if (symmetry == -1) {
          if (mPlus > phiSlice - 1) {
            mPlus = phiSlice - 2;
            signPlus = -1;
          }
          if (mMinus < 0) {
            mMinus = 1;
            signMinus = -1;
          }
        } else { // No Symmetries in phi, no boundaries, the calculation is continuous across all phi
          if (mPlus > phiSlice - 1) {
            mPlus = m + 1 - phiSlice;
          }
          if (mMinus < 0) {
            mMinus = m - 1 + phiSlice;
          }
        }
        matrixV = matricesCurrentV[m];
        matrixVP = matricesCurrentV[mPlus];  // slice
        matrixVM = matricesCurrentV[mMinus]; // slice
        arrayCharge = matricesCurrentCharge[m];

        isw = jsw;
        for (Int_t j = 1; j < tnZColumn - 1; j++, isw = 3 - isw) {
          for (Int_t i = isw; i < tnRRow - 1; i += 2) {
            //Info("Relax3D",Form("Doing slice %d, z=%d, r=%d", m,j,i));
            (*matrixV)(i, j) = (coefficient2[i] * (*matrixV)(i - 1, j) + tempRatioZ * ((*matrixV)(i, j - 1) + (*matrixV)(i, j + 1)) + coefficient1[i] * (*matrixV)(i + 1, j) + coefficient3[i] * (signPlus * (*matrixVP)(i, j) + signMinus * (*matrixVM)(i, j)) + (h2 * (*arrayCharge)(i, j))) * coefficient4[i];
          } // end cols
        }   // end nRRow
      }     // end phi
    }       // end sweep
  } else if (fMgParameters.relaxType == kJacobi) {
    // for each slice
    for (Int_t m = 0; m < phiSlice; m++) {

      mPlus = m + 1;
      signPlus = 1;
      mMinus = m - 1;
      signMinus = 1;

      // Reflection symmetry in phi (e.g. symmetry at sector boundaries, or half sectors, etc.)
      if (symmetry == 1) {
        if (mPlus > phiSlice - 1) {
          mPlus = phiSlice - 2;
        }
        if (mMinus < 0) {
          mMinus = 1;
        }
      }
      // Anti-symmetry in phi
      else if (symmetry == -1) {
        if (mPlus > phiSlice - 1) {
          mPlus = phiSlice - 2;
          signPlus = -1;
        }
        if (mMinus < 0) {
          mMinus = 1;
          signMinus = -1;
        }
      } else { // No Symmetries in phi, no boundaries, the calculation is continuous across all phi
        if (mPlus > phiSlice - 1) {
          mPlus = m + 1 - phiSlice;
        }
        if (mMinus < 0) {
          mMinus = m - 1 + phiSlice;
        }
      }

      matrixV = matricesCurrentV[m];
      matrixVP = matricesCurrentV[mPlus];  // slice
      matrixVM = matricesCurrentV[mMinus]; // slice
      arrayCharge = matricesCurrentCharge[m];

      // Jacobian
      for (Int_t j = 1; j < tnZColumn - 1; j++) {
        for (Int_t i = 1; i < tnRRow - 1; i++) {
          (*matrixV)(i, j) = (coefficient2[i] * (*matrixV)(i - 1, j) + tempRatioZ * ((*matrixV)(i, j - 1) + (*matrixV)(i, j + 1)) + coefficient1[i] * (*matrixV)(i + 1, j) + coefficient3[i] * (signPlus * (*matrixVP)(i, j) + signMinus * (*matrixVM)(i, j)) + (h2 * (*arrayCharge)(i, j))) * coefficient4[i];

        } // end cols
      }   // end nRRow

    } // end phi

  } else {
    // Case weighted Jacobi
    // TODO
  }
}

/// Relax2D
///
///    Relaxation operation for multiGrid
///		 relaxation used 5 stencil in cylindrical coordinate
///
/// Using the following equations
/// \f$ U_{i,j,k} = (1 + \frac{1}{r_{i}h_{r}}) U_{i+1,j,k}  + (1 - \frac{1}{r_{i}h_{r}}) U_{i+1,j,k}  \f$
///
/// \param matricesCurrentV TMatrixD& potential in 3D (matrices of matrix)
/// \param matricesCurrentCharge TMatrixD& charge in 3D
/// \param nRRow const Int_t number of nRRow in the r direction of TPC
/// \param nZColumn const Int_t number of nZColumn in z direction of TPC
/// \param phiSlice const Int_t number of phiSlice in phi direction of TPC
/// \param symmetry const Int_t is the cylinder has symmetry
/// \param h2 const Float_t \f$  h_{r}^{2} \f$
/// \param tempFourth const Float_t coefficient for h
/// \param tempRatio const Float_t ratio between grid size in z-direction and r-direction
/// \param coefficient1 std::vector<float> coefficient for \f$  V_{x+1,y,z} \f$
/// \param coefficient2 std::vector<float> coefficient for \f$  V_{x-1,y,z} \f$
///
void AliTPCPoissonSolver::Relax2D(TMatrixD& matricesCurrentV, TMatrixD& matricesCurrentCharge, const Int_t tnRRow,
                                  const Int_t tnZColumn,
                                  const Float_t h2, const Float_t tempFourth, const Float_t tempRatio,
                                  std::vector<float>& coefficient1, std::vector<float>& coefficient2)
{

  // Gauss-Seidel
  if (fMgParameters.relaxType == kGaussSeidel) {

    Int_t isw, jsw = 1;
    for (Int_t iPass = 1; iPass <= 2; iPass++, jsw = 3 - jsw) {
      isw = jsw;
      for (Int_t j = 1; j < tnZColumn - 1; j++, isw = 3 - isw) {
        for (Int_t i = isw; i < tnRRow - 1; i += 2) {
          matricesCurrentV(i, j) = tempFourth * (coefficient1[i] * matricesCurrentV(i + 1, j) +
                                                 coefficient2[i] * matricesCurrentV(i - 1, j) +
                                                 tempRatio * (matricesCurrentV(i, j + 1) + matricesCurrentV(i, j - 1)) +
                                                 (h2 * matricesCurrentCharge(i, j)));
        } // end cols
      }   // end nRRow
    }     // end pass red-black
  } else if (fMgParameters.relaxType == kJacobi) {
    for (Int_t j = 1; j < tnZColumn - 1; j++) {
      for (Int_t i = 1; i < tnRRow - 1; i++) {
        matricesCurrentV(i, j) = tempFourth * (coefficient1[i] * matricesCurrentV(i + 1, j) +
                                               coefficient2[i] * matricesCurrentV(i - 1, j) + tempRatio * (matricesCurrentV(i, j + 1) + matricesCurrentV(i, j - 1)) +
                                               (h2 * matricesCurrentCharge(i, j)));
      } // end cols
    }   // end nRRow
  } else if (fMgParameters.relaxType == kWeightedJacobi) {
    // Weighted Jacobi
    // TODO
  }
}

/// Residue3D
///
///    Compute residue from V(.) where V(.) is numerical potential and f(.).
///		 residue used 7 stencil in cylindrical coordinate
///
/// Using the following equations
/// \f$ U_{i,j,k} = (1 + \frac{1}{r_{i}h_{r}}) U_{i+1,j,k}  + (1 - \frac{1}{r_{i}h_{r}}) U_{i+1,j,k}  \f$
///
/// \param residue TMatrixD** residue in 3D (matrices of matrix)
/// \param matricesCurrentV TMatrixD** potential in 3D (matrices of matrix)
/// \param matricesCurrentCharge TMatrixD** charge in 3D
/// \param nRRow const Int_t number of nRRow in the r direction of TPC
/// \param nZColumn const Int_t number of nZColumn in z direction of TPC
/// \param phiSlice const Int_t number of phiSlice in phi direction of TPC
/// \param symmetry const Int_t is the cylinder has symmetry
/// \param ih2 const Float_t \f$ 1/ h_{r}^{2} \f$
/// \param tempRatioZ const Float_t ration between grid size in z-direction and r-direction
/// \param coefficient1 std::vector<float> coefficient for \f$  V_{x+1,y,z} \f$
/// \param coefficient2 std::vector<float> coefficient for \f$  V_{x-1,y,z} \f$
/// \param coefficient3 std::vector<float> coefficient for z
/// \param inverseCoefficient4 std::vector<float> inverse coefficient for f(r,\phi,z)
///
void AliTPCPoissonSolver::Residue3D(TMatrixD** residue, TMatrixD** matricesCurrentV, TMatrixD** matricesCurrentCharge,
                                    const Int_t tnRRow,
                                    const Int_t tnZColumn, const Int_t phiSlice, const Int_t symmetry,
                                    const Float_t ih2,
                                    const Float_t tempRatioZ, std::vector<float>& coefficient1,
                                    std::vector<float>& coefficient2,
                                    std::vector<float>& coefficient3, std::vector<float>& inverseCoefficient4)
{
  Int_t mPlus, mMinus, signPlus, signMinus;
  for (Int_t m = 0; m < phiSlice; m++) {

    mPlus = m + 1;
    signPlus = 1;
    mMinus = m - 1;
    signMinus = 1;

    // Reflection symmetry in phi (e.g. symmetry at sector boundaries, or half sectors, etc.)
    if (symmetry == 1) {
      if (mPlus > phiSlice - 1) {
        mPlus = phiSlice - 2;
      }
      if (mMinus < 0) {
        mMinus = 1;
      }
    }
    // Anti-symmetry in phi
    else if (symmetry == -1) {
      if (mPlus > phiSlice - 1) {
        mPlus = phiSlice - 2;
        signPlus = -1;
      }
      if (mMinus < 0) {
        mMinus = 1;
        signMinus = -1;
      }
    } else { // No Symmetries in phi, no boundaries, the calculation is continuous across all phi
      if (mPlus > phiSlice - 1) {
        mPlus = m + 1 - phiSlice;
      }
      if (mMinus < 0) {
        mMinus = m - 1 + phiSlice;
      }
    }

    TMatrixD& arrayResidue = *residue[m];
    TMatrixD& matrixV = *matricesCurrentV[m];
    TMatrixD& matrixVP = *matricesCurrentV[mPlus];  // slice
    TMatrixD& matrixVM = *matricesCurrentV[mMinus]; // slice
    TMatrixD& arrayCharge = *matricesCurrentCharge[m];

    for (Int_t j = 1; j < tnZColumn - 1; j++) {
      for (Int_t i = 1; i < tnRRow - 1; i++) {

        arrayResidue(i, j) =
          ih2 * (coefficient2[i] * matrixV(i - 1, j) + tempRatioZ * (matrixV(i, j - 1) + matrixV(i, j + 1)) + coefficient1[i] * matrixV(i + 1, j) +
                 coefficient3[i] * (signPlus * matrixVP(i, j) + signMinus * matrixVM(i, j)) -
                 inverseCoefficient4[i] * matrixV(i, j)) +
          arrayCharge(i, j);

      } // end cols
    }   // end nRRow

    //arrayResidue.Print();
  }
}

/// Residue2D
///
///    Compute residue from V(.) where V(.) is numerical potential and f(.).
///		 residue used 5 stencil in cylindrical coordinate
///
/// Using the following equations
/// \f$ U_{i,j,k} = (1 + \frac{1}{r_{i}h_{r}}) U_{i+1,j,k}  + (1 - \frac{1}{r_{i}h_{r}}) U_{i+1,j,k}  \f$
///
/// \param residue TMatrixD& potential in 2D
/// \param matricesCurrentV TMatrixD& potential in 2D
/// \param matricesCurrentCharge TMatrixD& charge in 2D
/// \param nRRow const Int_t number of nRRow in the r direction of TPC
/// \param nZColumn const Int_t number of nZColumn in z direction of TPC
/// \param phiSlice const Int_t number of phiSlice in phi direction of TPC
/// \param symmetry const Int_t is the cylinder has symmetry
/// \param h2 const Float_t \f$  h_{r}^{2} \f$
/// \param tempFourth const Float_t coefficient for h
/// \param tempRatio const Float_t ratio between grid size in z-direction and r-direction
/// \param coefficient1 std::vector<float> coefficient for \f$  V_{x+1,y,z} \f$
/// \param coefficient2 std::vector<float> coefficient for \f$  V_{x-1,y,z} \f$
///
void AliTPCPoissonSolver::Residue2D(TMatrixD& residue, TMatrixD& matricesCurrentV, TMatrixD& matricesCurrentCharge,
                                    const Int_t tnRRow,
                                    const Int_t tnZColumn, const Float_t ih2, const Float_t inverseTempFourth,
                                    const Float_t tempRatio, std::vector<float>& coefficient1,
                                    std::vector<float>& coefficient2)
{
  for (Int_t i = 1; i < tnRRow - 1; i++) {
    for (Int_t j = 1; j < tnZColumn - 1; j++) {
      residue(i, j) = ih2 * (coefficient1[i] * matricesCurrentV(i + 1, j) + coefficient2[i] * matricesCurrentV(i - 1, j) + tempRatio * (matricesCurrentV(i, j + 1) + matricesCurrentV(i, j - 1)) -
                             inverseTempFourth * matricesCurrentV(i, j)) +
                      matricesCurrentCharge(i, j);

    } // end cols
  }   // end nRRow

  //Boundary points.
  for (Int_t i = 0; i < tnRRow; i++) {
    residue(i, 0) = residue(i, tnZColumn - 1) = 0.0;
  }

  for (Int_t j = 0; j < tnZColumn; j++) {
    residue(0, j) = residue(tnRRow - 1, j) = 0.0;
  }
}

/// Restrict2D
///
///    Grid transfer operator, restrict from fine -> coarse grid
///		 provide full-half weighting
///
///		 \[ \frac{1}{16}\left( \begin{array}{ccc}
///      1 & 2 & 1 \\
///      2 & 4 & 2 \\
///      1 & 2 & 1 \end{array} \right) \]
///
/// \param matricesCurrentCharge TMatrixD& coarse grid (2h)
/// \param residue TMatrixD& fine grid  (h)
/// \param nRRow const Int_t number of nRRow in the r direction of TPC
/// \param nZColumn const Int_t number of nZColumn in z direction of TPC
///
void AliTPCPoissonSolver::Restrict2D(TMatrixD& matricesCurrentCharge, TMatrixD& residue, const Int_t tnRRow,
                                     const Int_t tnZColumn)
{

  for (Int_t i = 1, ii = 2; i < tnRRow - 1; i++, ii += 2) {
    for (Int_t j = 1, jj = 2; j < tnZColumn - 1; j++, jj += 2) {
      if (fMgParameters.gtType == kHalf) {
        // half
        matricesCurrentCharge(i, j) = 0.5 * residue(ii, jj) +
                                      0.125 *
                                        (residue(ii + 1, jj) + residue(ii - 1, jj) + residue(ii, jj + 1) +
                                         residue(ii, jj - 1));

      } else
        // full
        if (fMgParameters.gtType == kFull) {
        matricesCurrentCharge(i, j) = 0.25 * residue(ii, jj) +
                                      0.125 *
                                        (residue(ii + 1, jj) + residue(ii - 1, jj) + residue(ii, jj + 1) +
                                         residue(ii, jj - 1)) +
                                      0.0625 *
                                        (residue(ii + 1, jj + 1) + residue(ii - 1, jj + 1) + residue(ii + 1, jj - 1) +
                                         residue(ii - 1, jj - 1));
      }

    } // end cols
  }   // end nRRow

  // boundary
  // for boundary
  for (Int_t j = 0, jj = 0; j < tnZColumn; j++, jj += 2) {
    matricesCurrentCharge(0, j) = residue(0, jj);
    matricesCurrentCharge(tnRRow - 1, j) = residue((tnRRow - 1) * 2, jj);
  }

  // for boundary
  for (Int_t i = 0, ii = 0; i < tnZColumn; i++, ii += 2) {
    matricesCurrentCharge(i, 0) = residue(ii, 0);
    matricesCurrentCharge(i, tnZColumn - 1) = residue(ii, (tnZColumn - 1) * 2);
  }
}

/// RestrictBoundary2D
///
///    Boundary transfer  restrict from fine -> coarse grid
///
/// \param matricesCurrentCharge TMatrixD& coarse grid (2h)
/// \param residue TMatrixD& fine grid  (h)
/// \param nRRow const Int_t number of nRRow in the r direction of TPC
/// \param nZColumn const Int_t number of nZColumn in z direction of TPC
///
void AliTPCPoissonSolver::RestrictBoundary2D(TMatrixD& matricesCurrentCharge, TMatrixD& residue, const Int_t tnRRow,
                                             const Int_t tnZColumn)
{
  // for boundary
  for (Int_t j = 0, jj = 0; j < tnZColumn; j++, jj += 2) {
    matricesCurrentCharge(0, j) = residue(0, jj);
    matricesCurrentCharge(tnRRow - 1, j) = residue((tnRRow - 1) * 2, jj);
  }

  // for boundary
  for (Int_t i = 0, ii = 0; i < tnZColumn; i++, ii += 2) {
    matricesCurrentCharge(i, 0) = residue(ii, 0);
    matricesCurrentCharge(i, tnZColumn - 1) = residue(ii, (tnZColumn - 1) * 2);
  }
}

/// Restriction in 3D
///
/// Restriction is a map from fine grid (h) to coarse grid (2h)
///
/// In case of 3D
/// Full weighting:
/// \f[ (R u)_{i,j,k} = \frac{1}{2} u_{2i,2j,2k} + \frac{1}{4} S_{1} + \frac{1}{8} S_{2} + \frac{1}{16} S_{3}\f]
///
///
/// Restriction in all direction r-phi-z
/// restriction in phi only if oldPhi == 2*newPhi
/// \param matricesCurrentCharge TMatrixD** coarser grid 2h
/// \param residue TMatrixD ** fine grid h
/// \param tnRRow Int_t number of grid in nRRow (in r-direction) for coarser grid should be 2^N + 1, finer grid in 2^{N+1} + 1
/// \param tnZColumn Int_t number of grid in nZColumn (in z-direction) for coarser grid should be  2^M + 1, finer grid in 2^{M+1} + 1
/// \param newPhiSlice Int_t number of phiSlice (in phi-direction) for coarser grid
/// \param oldPhiSlice Int_t number of phiSlice (in phi-direction) for finer grid
///
void AliTPCPoissonSolver::Restrict3D(TMatrixD** matricesCurrentCharge, TMatrixD** residue, const Int_t tnRRow,
                                     const Int_t tnZColumn,
                                     const Int_t newPhiSlice, const Int_t oldPhiSlice)
{

  Double_t s1, s2, s3;

  if (2 * newPhiSlice == oldPhiSlice) {

    Int_t mPlus, mMinus;
    Int_t mm = 0;

    for (Int_t m = 0; m < newPhiSlice; m++, mm += 2) {

      // assuming no symmetry
      mPlus = mm + 1;
      mMinus = mm - 1;

      if (mPlus > (oldPhiSlice)-1) {
        mPlus = mm + 1 - (oldPhiSlice);
      }
      if (mMinus < 0) {
        mMinus = mm - 1 + (oldPhiSlice);
      }

      TMatrixD& arrayResidue = *residue[mm];
      TMatrixD& arrayResidueP = *residue[mPlus];
      TMatrixD& arrayResidueM = *residue[mMinus]; // slice
      TMatrixD& arrayCharge = *matricesCurrentCharge[m];

      for (Int_t i = 1, ii = 2; i < tnRRow - 1; i++, ii += 2) {
        for (Int_t j = 1, jj = 2; j < tnZColumn - 1; j++, jj += 2) {

          // at the same plane
          s1 = arrayResidue(ii + 1, jj) + arrayResidue(ii - 1, jj) + arrayResidue(ii, jj + 1) +
               arrayResidue(ii, jj - 1) + arrayResidueP(ii, jj) + arrayResidueM(ii, jj);
          s2 = (arrayResidue(ii + 1, jj + 1) + arrayResidue(ii + 1, jj - 1) + arrayResidueP(ii + 1, jj) +
                arrayResidueM(ii + 1, jj)) +
               (arrayResidue(ii - 1, jj - 1) + arrayResidue(ii - 1, jj + 1) + arrayResidueP(ii - 1, jj) +
                arrayResidueM(ii - 1, jj)) +
               arrayResidueP(ii, jj - 1) + arrayResidueM(ii, jj + 1) + arrayResidueM(ii, jj - 1) +
               arrayResidueP(ii, jj + 1);

          s3 = (arrayResidueP(ii + 1, jj + 1) + arrayResidueP(ii + 1, jj - 1) + arrayResidueM(ii + 1, jj + 1) +
                arrayResidueM(ii + 1, jj - 1)) +
               (arrayResidueM(ii - 1, jj - 1) + arrayResidueM(ii - 1, jj + 1) + arrayResidueP(ii - 1, jj - 1) +
                arrayResidueP(ii - 1, jj + 1));

          arrayCharge(i, j) = 0.125 * arrayResidue(ii, jj) + 0.0625 * s1 + 0.03125 * s2 + 0.015625 * s3;
        } // end cols
      }   // end nRRow

      // for boundary
      for (Int_t j = 0, jj = 0; j < tnZColumn; j++, jj += 2) {
        arrayCharge(0, j) = arrayResidue(0, jj);
        arrayCharge(tnRRow - 1, j) = arrayResidue((tnRRow - 1) * 2, jj);
      }

      // for boundary
      for (Int_t i = 0, ii = 0; i < tnZColumn; i++, ii += 2) {
        arrayCharge(i, 0) = arrayResidue(ii, 0);
        arrayCharge(i, tnZColumn - 1) = arrayResidue(ii, (tnZColumn - 1) * 2);
      }
    } // end phis

  } else {
    for (int m = 0; m < newPhiSlice; m++) {
      Restrict2D(*matricesCurrentCharge[m], *residue[m], tnRRow, tnZColumn);
    }
  }
}

/// Restrict Boundary in 3D
///
/// Pass boundary information to coarse grid
///
/// \param matricesCurrentCharge TMatrixD** coarser grid 2h
/// \param residue TMatrixD ** fine grid h
/// \param tnRRow Int_t number of grid in nRRow (in r-direction) for coarser grid should be 2^N + 1, finer grid in 2^{N+1} + 1
/// \param tnZColumn Int_t number of grid in nZColumn (in z-direction) for coarser grid should be  2^M + 1, finer grid in 2^{M+1} + 1
/// \param newPhiSlice Int_t number of phiSlice (in phi-direction) for coarser grid
/// \param oldPhiSlice Int_t number of phiSlice (in phi-direction) for finer grid
///
void AliTPCPoissonSolver::RestrictBoundary3D(TMatrixD** matricesCurrentCharge, TMatrixD** residue, const Int_t tnRRow,
                                             const Int_t tnZColumn, const Int_t newPhiSlice, const Int_t oldPhiSlice)
{

  // in case of full 3d and the phiSlice is also coarsening

  if (2 * newPhiSlice == oldPhiSlice) {

    for (Int_t m = 0, mm = 0; m < newPhiSlice; m++, mm += 2) {

      TMatrixD& arrayResidue = *residue[mm];
      TMatrixD& arrayCharge = *matricesCurrentCharge[m];
      // for boundary
      for (Int_t j = 0, jj = 0; j < tnZColumn; j++, jj += 2) {
        arrayCharge(0, j) = arrayResidue(0, jj);
        arrayCharge(tnRRow - 1, j) = arrayResidue((tnRRow - 1) * 2, jj);
      }

      // for boundary
      for (Int_t i = 0, ii = 0; i < tnZColumn; i++, ii += 2) {
        arrayCharge(i, 0) = arrayResidue(ii, 0);
        arrayCharge(i, tnZColumn - 1) = arrayResidue(ii, (tnZColumn - 1) * 2);
      }
    } // end phis
  } else {
    for (int m = 0; m < newPhiSlice; m++) {
      RestrictBoundary2D(*matricesCurrentCharge[m], *residue[m], tnRRow, tnZColumn);
    }
  }
}

/// Prolongation with Addition for 2D
///
/// Interpolation with addition from coarse level (2h) -->  fine level (h)
///
/// Interpolation in all direction r-phi-z
/// \param matricesCurrentV TMatrixD& fine grid h
/// \param matricesCurrentVC TMatrixD& coarse grid 2h
/// \param tnRRow Int_t number of grid in nRRow (in r-direction) for coarser grid should be 2^N + 1, finer grid in 2^{N+1} + 1
/// \param tnZColumn Int_t number of grid in nZColumn (in z-direction) for coarser grid should be  2^M + 1, finer grid in 2^{M+1} + 1a
///
void AliTPCPoissonSolver::AddInterp2D(TMatrixD& matricesCurrentV, TMatrixD& matricesCurrentVC, const Int_t tnRRow,
                                      const Int_t tnZColumn)
{
  for (Int_t j = 2; j < tnZColumn - 1; j += 2) {
    for (Int_t i = 2; i < tnRRow - 1; i += 2) {
      matricesCurrentV(i, j) = matricesCurrentV(i, j) + matricesCurrentVC(i / 2, j / 2);
    }
  }

  for (Int_t j = 1; j < tnZColumn - 1; j += 2) {
    for (Int_t i = 2; i < tnRRow - 1; i += 2) {
      matricesCurrentV(i, j) =
        matricesCurrentV(i, j) + 0.5 * (matricesCurrentVC(i / 2, j / 2) + matricesCurrentVC(i / 2, j / 2 + 1));
    }
  }

  for (Int_t j = 2; j < tnZColumn - 1; j += 2) {
    for (Int_t i = 1; i < tnRRow - 1; i += 2) {
      matricesCurrentV(i, j) =
        matricesCurrentV(i, j) + 0.5 * (matricesCurrentVC(i / 2, j / 2) + matricesCurrentVC(i / 2 + 1, j / 2));
    }
  }

  // only if full
  if (fMgParameters.gtType == kFull) {
    for (Int_t j = 1; j < tnZColumn - 1; j += 2) {
      for (Int_t i = 1; i < tnRRow - 1; i += 2) {
        matricesCurrentV(i, j) =
          matricesCurrentV(i, j) + 0.25 * (matricesCurrentVC(i / 2, j / 2) + matricesCurrentVC(i / 2, j / 2 + 1) + matricesCurrentVC(i / 2 + 1, j / 2) + matricesCurrentVC(i / 2 + 1, j / 2 + 1));
      }
    }
  }
}

/// Prolongation with Addition for 3D
///
/// Interpolation with addition from coarse level (2h) -->  fine level (h)
///
/// Interpolation in all direction r-phi-z
/// Interpolation in phi only if oldPhi == 2*newPhi
/// \param matricesCurrentV TMatrixD& fine grid h
/// \param matricesCurrentVC TMatrixD& coarse grid 2h
/// \param tnRRow Int_t number of grid in nRRow (in r-direction) for coarser grid should be 2^N + 1, finer grid in 2^{N+1} + 1
/// \param tnZColumn Int_t number of grid in nZColumn (in z-direction) for coarser grid should be  2^M + 1, finer grid in 2^{M+1} + 1a
/// \param newPhiSlice Int_t number of phiSlice (in phi-direction) for coarser grid
/// \param oldPhiSlice Int_t number of phiSlice (in phi-direction) for finer grid
///
void AliTPCPoissonSolver::AddInterp3D(TMatrixD** matricesCurrentV, TMatrixD** matricesCurrentVC, const Int_t tnRRow,
                                      const Int_t tnZColumn,
                                      const Int_t newPhiSlice, const Int_t oldPhiSlice)
{
  // Do restrict 2 D for each slice

  //const Float_t  h   =  (AliTPCPoissonSolver::fgkOFCRadius-AliTPCPoissonSolver::fgkIFCRadius) / ((tnRRow-1)/2); // h_{r}
  //Float_t radius,ratio;
  //std::vector<float> coefficient1((tnRRow-1) / 2 );  // coefficient1(nRRow) for storing (1 + h_{r}/2r_{i}) from central differences in r direction
  //std::vector<float> coefficient2((tnRRow-1) / 2);  // coefficient2(nRRow) for storing (1 + h_{r}/2r_{i}) from central differences in r direction

  if (newPhiSlice == 2 * oldPhiSlice) {
    Int_t mPlus, mmPlus;
    Int_t mm = 0;

    for (Int_t m = 0; m < newPhiSlice; m += 2) {

      // assuming no symmetry
      mm = m / 2;
      mmPlus = mm + 1;
      mPlus = m + 1;

      // round
      if (mmPlus > (oldPhiSlice)-1) {
        mmPlus = mm + 1 - (oldPhiSlice);
      }
      if (mPlus > (newPhiSlice)-1) {
        mPlus = m + 1 - (newPhiSlice);
      }

      TMatrixD& fineV = *matricesCurrentV[m];
      TMatrixD& fineVP = *matricesCurrentV[mPlus];
      TMatrixD& coarseV = *matricesCurrentVC[mm];
      TMatrixD& coarseVP = *matricesCurrentVC[mmPlus];

      for (Int_t j = 2; j < tnZColumn - 1; j += 2) {
        for (Int_t i = 2; i < tnRRow - 1; i += 2) {
          fineV(i, j) += coarseV(i / 2, j / 2);
          // point on corner lines at phi direction
          fineVP(i, j) += 0.5 * (coarseV(i / 2, j / 2) + coarseVP(i / 2, j / 2));
        }
      }

      for (Int_t j = 1; j < tnZColumn - 1; j += 2) {
        for (Int_t i = 2; i < tnRRow - 1; i += 2) {
          fineV(i, j) += 0.5 * (coarseV(i / 2, j / 2) + coarseV(i / 2, j / 2 + 1));
          // point on corner lines at phi direction
          fineVP(i, j) += 0.25 * (coarseV(i / 2, j / 2) + coarseV(i / 2, j / 2 + 1) + coarseVP(i / 2, j / 2) +
                                  coarseVP(i / 2, j / 2 + 1));
        }
      }

      for (Int_t j = 2; j < tnZColumn - 1; j += 2) {
        for (Int_t i = 1; i < tnRRow - 1; i += 2) {
          fineV(i, j) += 0.5 * (coarseV(i / 2, j / 2) + coarseV(i / 2 + 1, j / 2));

          // point on line at phi direction
          fineVP(i, j) += 0.25 * ((coarseV(i / 2, j / 2) + coarseVP(i / 2, j / 2)) +
                                  (coarseVP(i / 2 + 1, j / 2) + coarseV(i / 2 + 1, j / 2)));
        }
      }

      for (Int_t j = 1; j < tnZColumn - 1; j += 2) {
        for (Int_t i = 1; i < tnRRow - 1; i += 2) {
          fineV(i, j) += 0.25 * ((coarseV(i / 2, j / 2) + coarseV(i / 2, j / 2 + 1)) +
                                 (coarseV(i / 2 + 1, j / 2) + coarseV(i / 2 + 1, j / 2 + 1)));

          // point at the center at phi direction
          fineVP(i, j) += 0.125 * ((coarseV(i / 2, j / 2) + coarseV(i / 2, j / 2 + 1) + coarseVP(i / 2, j / 2) +
                                    coarseVP(i / 2, j / 2 + 1)) +
                                   (coarseV(i / 2 + 1, j / 2) + coarseV(i / 2 + 1, j / 2 + 1) +
                                    coarseVP(i / 2 + 1, j / 2) + coarseVP(i / 2 + 1, j / 2 + 1)));
        }
      }
    }

  } else {
    for (int m = 0; m < newPhiSlice; m++) {
      AddInterp2D(*matricesCurrentV[m], *matricesCurrentVC[m], tnRRow, tnZColumn);
    }
  }
}

/// Interpolation/Prolongation in 3D
///
/// Interpolation is a map from coarse grid (h) to fine grid (2h)
///
/// In case of 3D
/// Full weighting:
/// \f[ (R u)_{i,j,k} = \frac{1}{2} u_{2i,2j,2k} + \frac{1}{4} S_{1} + \frac{1}{8} S_{2} + \frac{1}{16} S_{3}\f]
///
///
/// Restriction in all direction r-phi-z
/// restriction in phi only if oldPhi == 2*newPhi
/// \param matricesCurrentV TMatrixD** finer grid h
/// \param curArrayCV TMatrixD ** coarse grid 2h
/// \param tnRRow Int_t number of grid in nRRow (in r-direction) for coarser grid should be 2^N + 1, finer grid in 2^{N+1} + 1
/// \param tnZColumn Int_t number of grid in nZColumn (in z-direction) for coarser grid should be  2^M + 1, finer grid in 2^{M+1} + 1
/// \param newPhiSlice Int_t number of phiSlice (in phi-direction) for coarser grid
/// \param oldPhiSlice Int_t number of phiSlice (in phi-direction) for finer grid
///
void AliTPCPoissonSolver::Interp3D(TMatrixD** matricesCurrentV, TMatrixD** matricesCurrentVC, const Int_t tnRRow,
                                   const Int_t tnZColumn,
                                   const Int_t newPhiSlice, const Int_t oldPhiSlice)
{

  // Do restrict 2 D for each slice
  if (newPhiSlice == 2 * oldPhiSlice) {
    Int_t mPlus, mmPlus;
    Int_t mm = 0;

    for (Int_t m = 0; m < newPhiSlice; m += 2) {

      // assuming no symmetry
      mm = m / 2;
      mmPlus = mm + 1;
      mPlus = m + 1;

      // round
      if (mmPlus > (oldPhiSlice)-1) {
        mmPlus = mm + 1 - (oldPhiSlice);
      }
      if (mPlus > (newPhiSlice)-1) {
        mPlus = m + 1 - (newPhiSlice);
      }

      TMatrixD& fineV = *matricesCurrentV[m];
      TMatrixD& fineVP = *matricesCurrentV[mPlus];
      TMatrixD& coarseV = *matricesCurrentVC[mm];
      TMatrixD& coarseVP = *matricesCurrentVC[mmPlus];

      for (Int_t j = 2; j < tnZColumn - 1; j += 2) {
        for (Int_t i = 2; i < tnRRow - 1; i += 2) {
          fineV(i, j) = coarseV(i / 2, j / 2);

          // point on corner lines at phi direction
          fineVP(i, j) = 0.5 * (coarseV(i / 2, j / 2) + coarseVP(i / 2, j / 2));
        }
      }

      for (Int_t j = 1; j < tnZColumn - 1; j += 2) {
        for (Int_t i = 2; i < tnRRow - 1; i += 2) {
          fineV(i, j) = 0.5 * (coarseV(i / 2, j / 2) + coarseV(i / 2, j / 2 + 1));

          // point on corner lines at phi direction
          fineVP(i, j) = 0.25 * (coarseV(i / 2, j / 2) + coarseV(i / 2, j / 2 + 1) + coarseVP(i / 2, j / 2) +
                                 coarseVP(i / 2, j / 2 + 1));
        }
      }

      for (Int_t j = 2; j < tnZColumn - 1; j += 2) {
        for (Int_t i = 1; i < tnRRow - 1; i += 2) {
          fineV(i, j) = 0.5 * (coarseV(i / 2, j / 2) + coarseV(i / 2 + 1, j / 2));

          // point on line at phi direction
          fineVP(i, j) = 0.25 * ((coarseV(i / 2, j / 2) + coarseVP(i / 2, j / 2)) +
                                 (coarseVP(i / 2 + 1, j / 2) + coarseV(i / 2 + 1, j / 2)));
        }
      }

      for (Int_t j = 1; j < tnZColumn - 1; j += 2) {
        for (Int_t i = 1; i < tnRRow - 1; i += 2) {
          fineV(i, j) = 0.25 * ((coarseV(i / 2, j / 2) + coarseV(i / 2, j / 2 + 1)) +
                                (coarseV(i / 2 + 1, j / 2) + coarseV(i / 2 + 1, j / 2 + 1)));

          // point at the center at phi direction
          fineVP(i, j) = 0.125 * ((coarseV(i / 2, j / 2) + coarseV(i / 2, j / 2 + 1) + coarseVP(i / 2, j / 2) +
                                   coarseVP(i / 2, j / 2 + 1)) +
                                  (coarseV(i / 2 + 1, j / 2) + coarseV(i / 2 + 1, j / 2 + 1) +
                                   coarseVP(i / 2 + 1, j / 2) + coarseVP(i / 2 + 1, j / 2 + 1)));
        }
      }
    }

  } else {
    for (int m = 0; m < newPhiSlice; m++) {
      Interp2D(*matricesCurrentV[m], *matricesCurrentVC[m], tnRRow, tnZColumn);
    }
  }
}

/// Interpolation/Prolongation in 2D
///
/// Interpolation is a map from coarse grid (h) to fine grid (2h)
///
/// In case of 2D
/// Full weighting:
/// \f[ (R u)_{i,j,k} = \frac{1}{2} u_{2i,2j,2k} + \frac{1}{4} S_{1} + \frac{1}{8} S_{2} + \frac{1}{16} S_{3}\f]
///
///
/// Restriction in all direction r-phi-z
/// \param matricesCurrentV TMatrixD** finer grid h
/// \param curArrayCV TMatrixD ** coarse grid 2h
/// \param tnRRow Int_t number of grid in nRRow (in r-direction) for coarser grid should be 2^N + 1, finer grid in 2^{N+1} + 1
/// \param tnZColumn Int_t number of grid in nZColumn (in z-direction) for coarser grid should be  2^M + 1, finer grid in 2^{M+1} + 1
///
void AliTPCPoissonSolver::Interp2D(TMatrixD& matricesCurrentV, TMatrixD& matricesCurrentVC, const Int_t tnRRow,
                                   const Int_t tnZColumn)
{
  for (Int_t j = 2; j < tnZColumn - 1; j += 2) {
    for (Int_t i = 2; i < tnRRow - 1; i += 2) {
      matricesCurrentV(i, j) = matricesCurrentVC(i / 2, j / 2);
    }
  }

  for (Int_t j = 1; j < tnZColumn - 1; j += 2) {
    for (Int_t i = 2; i < tnRRow - 1; i += 2) {
      matricesCurrentV(i, j) = 0.5 * (matricesCurrentVC(i / 2, j / 2) + matricesCurrentVC(i / 2, j / 2 + 1));
    }
  }

  for (Int_t j = 2; j < tnZColumn - 1; j += 2) {
    for (Int_t i = 1; i < tnRRow - 1; i += 2) {
      matricesCurrentV(i, j) = 0.5 * (matricesCurrentVC(i / 2, j / 2) + matricesCurrentVC(i / 2 + 1, j / 2));
    }
  }

  // only if full
  if (fMgParameters.gtType == kFull) {
    for (Int_t j = 1; j < tnZColumn - 1; j += 2) {
      for (Int_t i = 1; i < tnRRow - 1; i += 2) {
        matricesCurrentV(i, j) = 0.25 *
                                 (matricesCurrentVC(i / 2, j / 2) + matricesCurrentVC(i / 2, j / 2 + 1) +
                                  matricesCurrentVC(i / 2 + 1, j / 2) +
                                  matricesCurrentVC(i / 2 + 1, j / 2 + 1));
      }
    }
  }
}

/// V-Cycle 2D
///
/// Implementation non-recursive V-cycle for 2D
///
///	Algorithms:
///
/// \param nRRow Int_t number of grid in nRRow (in r-direction) for coarser grid should be 2^N + 1, finer grid in 2^{N+1} + 1
/// \param nZColumn Int_t number of grid in nZColumn (in z-direction) for coarser grid should be  2^M + 1, finer grid in 2^{M+1} + 1
/// \param gridFrom const Int_t finest level of grid
/// \param gridTo const Int_t coarsest level of grid
/// \param nPre const Int_t number of smoothing before coarsening
/// \param nPost const Int_t number of smoothing after coarsening
/// \param gridSizeR const Float_t grid size in r direction (OPTION,  recalculate)
/// \param ratio const Float_t ratio between square of grid r and grid z (OPTION,  recalculate)
/// \param tvArrayV vector<TMatrixD *> vector of V potential in different grids
/// \param tvCharge vector<TMatrixD *> vector of charge distribution in different grids
/// \param tvResidue vector<TMatrixD *> vector of residue calculation in different grids
///
void AliTPCPoissonSolver::VCycle2D(const Int_t nRRow, const Int_t nZColumn, const Int_t gridFrom, const Int_t gridTo,
                                   const Int_t nPre, const Int_t nPost, const Float_t gridSizeR, const Float_t ratio,
                                   std::vector<TMatrixD*>& tvArrayV,
                                   std::vector<TMatrixD*>& tvCharge, std::vector<TMatrixD*>& tvResidue)
{

  Float_t h, h2, ih2, tempRatio, tempFourth, inverseTempFourth, radius;
  TMatrixD *matricesCurrentV, *matricesCurrentVC;
  TMatrixD* matricesCurrentCharge;
  TMatrixD* residue;
  Int_t iOne, jOne, tnRRow, tnZColumn, count;
  iOne = 1 << (gridFrom - 1);
  jOne = 1 << (gridFrom - 1);

  matricesCurrentV = nullptr;
  matricesCurrentVC = nullptr;
  matricesCurrentCharge = nullptr;
  residue = nullptr;

  tnRRow = iOne == 1 ? nRRow : nRRow / iOne + 1;
  tnZColumn = jOne == 1 ? nZColumn : nZColumn / jOne + 1;

  std::vector<float> coefficient1(nRRow);
  std::vector<float> coefficient2(nZColumn);

  // 1) Go to coarsest level
  for (count = gridFrom; count <= gridTo - 1; count++) {
    h = gridSizeR * iOne;
    h2 = h * h;
    ih2 = 1.0 / h2;
    tempRatio = ratio * iOne * iOne / (jOne * jOne);
    tempFourth = 1.0 / (2.0 + 2.0 * tempRatio);
    inverseTempFourth = 1.0 / tempFourth;
    for (Int_t i = 1; i < tnRRow - 1; i++) {
      radius = AliTPCPoissonSolver::fgkIFCRadius + i * h;
      coefficient1[i] = 1.0 + h / (2 * radius);
      coefficient2[i] = 1.0 - h / (2 * radius);
    }
    matricesCurrentV = tvArrayV[count - 1];
    matricesCurrentCharge = tvCharge[count - 1];
    residue = tvResidue[count - 1];

    // 1) Pre-Smoothing: Gauss-Seidel Relaxation or Jacobi
    for (Int_t jPre = 1; jPre <= nPre; jPre++) {
      Relax2D(*matricesCurrentV, *matricesCurrentCharge, tnRRow, tnZColumn, h2, tempFourth, tempRatio, coefficient1,
              coefficient2);
    }

    // 2) Residue calculation
    Residue2D(*residue, *matricesCurrentV, *matricesCurrentCharge, tnRRow, tnZColumn, ih2, inverseTempFourth, tempRatio,
              coefficient1,
              coefficient2);

    iOne = 2 * iOne;
    jOne = 2 * jOne;
    tnRRow = iOne == 1 ? nRRow : nRRow / iOne + 1;
    tnZColumn = jOne == 1 ? nZColumn : nZColumn / jOne + 1;

    matricesCurrentCharge = tvCharge[count];
    matricesCurrentV = tvArrayV[count];

    //3) Restriction
    Restrict2D(*matricesCurrentCharge, *residue, tnRRow, tnZColumn);

    //4) Zeroing coarser V
    matricesCurrentV->Zero();
  }

  // 5) coarsest grid
  h = gridSizeR * iOne;
  h2 = h * h;
  tempRatio = ratio * iOne * iOne / (jOne * jOne);
  tempFourth = 1.0 / (2.0 + 2.0 * tempRatio);

  for (Int_t i = 1; i < tnRRow - 1; i++) {
    radius = AliTPCPoissonSolver::fgkIFCRadius + i * h;
    coefficient1[i] = 1.0 + h / (2 * radius);
    coefficient2[i] = 1.0 - h / (2 * radius);
  }

  Relax2D(*matricesCurrentV, *matricesCurrentCharge, tnRRow, tnZColumn, h2, tempFourth, tempRatio, coefficient1,
          coefficient2);

  // Go to finest grid
  for (count = gridTo - 1; count >= gridFrom; count--) {

    iOne = iOne / 2;
    jOne = jOne / 2;

    h = gridSizeR * iOne;
    h2 = h * h;
    ih2 = 1.0 / h2;
    tempRatio = ratio * iOne * iOne / (jOne * jOne);
    tempFourth = 1.0 / (2.0 + 2.0 * tempRatio);
    inverseTempFourth = 1.0 / tempFourth;

    tnRRow = iOne == 1 ? nRRow : nRRow / iOne + 1;
    tnZColumn = jOne == 1 ? nZColumn : nZColumn / jOne + 1;
    matricesCurrentCharge = tvCharge[count - 1];
    matricesCurrentV = tvArrayV[count - 1];
    matricesCurrentVC = tvArrayV[count];

    // 6) Interpolation/Prolongation
    AddInterp2D(*matricesCurrentV, *matricesCurrentVC, tnRRow, tnZColumn);

    for (Int_t i = 1; i < tnRRow - 1; i++) {
      radius = AliTPCPoissonSolver::fgkIFCRadius + i * h;
      coefficient1[i] = 1.0 + h / (2 * radius);
      coefficient2[i] = 1.0 - h / (2 * radius);
    }

    // 7) Post-Smoothing: Gauss-Seidel Relaxation
    for (Int_t jPost = 1; jPost <= nPost; jPost++) {
      Relax2D(*matricesCurrentV, *matricesCurrentCharge, tnRRow, tnZColumn, h2, tempFourth, tempRatio, coefficient1,
              coefficient2);
    } // end post smoothing

    //// DEBUG ////
    //Info("VCycle2D",Form("Count %d", count));
    //Info("VCycle2D",Form("Exact Err: %f, MG Iteration : %d", (*fError)(mgCycle), mgCycle));
    //matricesCurrentV->Print();
    //matricesCurrentCharge->Print();
  }
}

/// W-Cycle 2D
///
/// Implementation non-recursive W-cycle for 2D
///
///	Algorithms:
///
/// \param nRRow Int_t number of grid in nRRow (in r-direction) for coarser grid should be 2^N + 1, finer grid in 2^{N+1} + 1
/// \param nZColumn Int_t number of grid in nZColumn (in z-direction) for coarser grid should be  2^M + 1, finer grid in 2^{M+1} + 1
/// \param gridFrom const Int_t finest level of grid
/// \param gridTo const Int_t coarsest level of grid
/// \param gamma const Int_t number of iterations at coarsest level
/// \param nPre const Int_t number of smoothing before coarsening
/// \param nPost const Int_t number of smoothing after coarsening
/// \param gridSizeR const Float_t grid size in r direction (OPTION,  recalculate)
/// \param ratio const Float_t ratio between square of grid r and grid z (OPTION,  recalculate)
/// \param tvArrayV vector<TMatrixD *> vector of V potential in different grids
/// \param tvCharge vector<TMatrixD *> vector of charge distribution in different grids
/// \param tvResidue vector<TMatrixD *> vector of residue calculation in different grids
///
void AliTPCPoissonSolver::WCycle2D(const Int_t nRRow, const Int_t nZColumn, const Int_t gridFrom, const Int_t gridTo,
                                   const int gamma,
                                   const Int_t nPre, const Int_t nPost, const Float_t gridSizeR, const Float_t ratio,
                                   std::vector<TMatrixD*>& tvArrayV,
                                   std::vector<TMatrixD*>& tvCharge, std::vector<TMatrixD*>& tvResidue)
{

  Float_t h, h2, ih2, tempRatio, tempFourth, inverseTempFourth, radius;
  TMatrixD *matricesCurrentV, *matricesCurrentVC;
  TMatrixD* matricesCurrentCharge;
  TMatrixD* residue;
  Int_t iOne, jOne, tnRRow, tnZColumn, count;
  iOne = 1 << (gridFrom - 1);
  jOne = 1 << (gridFrom - 1);

  tnRRow = iOne == 1 ? nRRow : nRRow / iOne + 1;
  tnZColumn = jOne == 1 ? nZColumn : nZColumn / jOne + 1;

  std::vector<float> coefficient1(nRRow);
  std::vector<float> coefficient2(nZColumn);

  // 1) Go to coarsest level
  for (count = gridFrom; count <= gridTo - 2; count++) {
    h = gridSizeR * iOne;
    h2 = h * h;
    ih2 = 1.0 / h2;
    tempRatio = ratio * iOne * iOne / (jOne * jOne);
    tempFourth = 1.0 / (2.0 + 2.0 * tempRatio);
    inverseTempFourth = 1.0 / tempFourth;
    for (Int_t i = 1; i < tnRRow - 1; i++) {
      radius = AliTPCPoissonSolver::fgkIFCRadius + i * h;
      coefficient1[i] = 1.0 + h / (2 * radius);
      coefficient2[i] = 1.0 - h / (2 * radius);
    }
    matricesCurrentV = tvArrayV[count - 1];
    matricesCurrentCharge = tvCharge[count - 1];
    residue = tvResidue[count - 1];

    // 1) Pre-Smoothing: Gauss-Seidel Relaxation or Jacobi
    for (Int_t jPre = 1; jPre <= nPre; jPre++) {
      Relax2D(*matricesCurrentV, *matricesCurrentCharge, tnRRow, tnZColumn, h2, tempFourth, tempRatio, coefficient1,
              coefficient2);
    }

    // 2) Residue calculation
    Residue2D(*residue, *matricesCurrentV, *matricesCurrentCharge, tnRRow, tnZColumn, ih2, inverseTempFourth, tempRatio,
              coefficient1,
              coefficient2);

    iOne = 2 * iOne;
    jOne = 2 * jOne;
    tnRRow = iOne == 1 ? nRRow : nRRow / iOne + 1;
    tnZColumn = jOne == 1 ? nZColumn : nZColumn / jOne + 1;

    matricesCurrentCharge = tvCharge[count];
    matricesCurrentV = tvArrayV[count];

    //3) Restriction
    Restrict2D(*matricesCurrentCharge, *residue, tnRRow, tnZColumn);

    //4) Zeroing coarser V
    matricesCurrentV->Zero();
  }

  // Do V cycle from: gridTo-1 to gridTo gamma times
  for (Int_t iGamma = 0; iGamma < gamma; iGamma++) {
    VCycle2D(nRRow, nZColumn, gridTo - 1, gridTo,
             nPre, nPost, gridSizeR, ratio, tvArrayV,
             tvCharge, tvResidue);
  }

  // Go to finest grid
  for (count = gridTo - 2; count >= gridFrom; count--) {

    iOne = iOne / 2;
    jOne = jOne / 2;

    h = gridSizeR * iOne;
    h2 = h * h;
    ih2 = 1.0 / h2;
    tempRatio = ratio * iOne * iOne / (jOne * jOne);
    tempFourth = 1.0 / (2.0 + 2.0 * tempRatio);
    inverseTempFourth = 1.0 / tempFourth;

    tnRRow = iOne == 1 ? nRRow : nRRow / iOne + 1;
    tnZColumn = jOne == 1 ? nZColumn : nZColumn / jOne + 1;
    matricesCurrentCharge = tvCharge[count - 1];
    matricesCurrentV = tvArrayV[count - 1];
    matricesCurrentVC = tvArrayV[count];

    // 6) Interpolation/Prolongation
    AddInterp2D(*matricesCurrentV, *matricesCurrentVC, tnRRow, tnZColumn);

    for (Int_t i = 1; i < tnRRow - 1; i++) {
      radius = AliTPCPoissonSolver::fgkIFCRadius + i * h;
      coefficient1[i] = 1.0 + h / (2 * radius);
      coefficient2[i] = 1.0 - h / (2 * radius);
    }

    // 7) Post-Smoothing: Gauss-Seidel Relaxation
    for (Int_t jPost = 1; jPost <= nPost; jPost++) {
      Relax2D(*matricesCurrentV, *matricesCurrentCharge, tnRRow, tnZColumn, h2, tempFourth, tempRatio, coefficient1,
              coefficient2);
    } // end post smoothing
  }
}

/// VCycle 3D2D, V Cycle 3D in multiGrid with constant phiSlice
/// fine-->coarsest-->fine, propagating the residue to correct initial guess of V
///
/// Algorithm:
///
///    NOTE: In order for this algorithm to work, the number of nRRow and nZColumn must be a power of 2 plus one.
///    The number of nRRow and Z Column can be different.
///
///    R Row       ==  2**M + 1
///    Z Column    ==  2**N + 1
///    Phi Slice  ==  Arbitrary but greater than 3
///
///    DeltaPhi in Radians
/// \param nRRow Int_t number of grid in nRRow (in r-direction) for coarser grid should be 2^N + 1, finer grid in 2^{N+1} + 1
/// \param nZColumn Int_t number of grid in nZColumn (in z-direction) for coarser grid should be  2^M + 1, finer grid in 2^{M+1} + 1
/// \param gridFrom const Int_t finest level of grid
/// \param gridTo const Int_t coarsest level of grid
/// \param nPre const Int_t number of smoothing before coarsening
/// \param nPost const Int_t number of smoothing after coarsening
/// \param gridSizeR const Float_t grid size in r direction (OPTION,  recalculate)
/// \param ratio const Float_t ratio between square of grid r and grid z (OPTION,  recalculate)
/// \param tvArrayV vector<TMatrixD *> vector of V potential in different grids
/// \param tvCharge vector<TMatrixD *> vector of charge distribution in different grids
/// \param tvResidue vector<TMatrixD *> vector of residue calculation in different grids
/// \param coefficient1 std::vector<float>& coefficient for relaxation (r direction)
/// \param coefficient2 std::vector<float>& coefficient for relaxation (r direction)
/// \param coefficient3 std::vector<float>& coefficient for relaxation (ratio r/z)
/// \param coefficient4 std::vector<float>& coefficient for relaxation (ratio for grid_r)
/// \param inverseCoefficient4 std::vector<float>& coefficient for relaxation (inverse coefficient4)
///
void AliTPCPoissonSolver::VCycle3D2D(const Int_t nRRow, const Int_t nZColumn, const Int_t phiSlice, const Int_t symmetry,
                                     const Int_t gridFrom, const Int_t gridTo, const Int_t nPre, const Int_t nPost,
                                     const Float_t gridSizeR, const Float_t ratioZ, const Float_t ratioPhi,
                                     std::vector<TMatrixD**>& tvArrayV, std::vector<TMatrixD**>& tvCharge,
                                     std::vector<TMatrixD**>& tvResidue, std::vector<float>& coefficient1,
                                     std::vector<float>& coefficient2, std::vector<float>& coefficient3,
                                     std::vector<float>& coefficient4,
                                     std::vector<float>& inverseCoefficient4)
{

  Float_t h, h2, ih2, tempRatioZ, tempRatioPhi, radius;
  TMatrixD **matricesCurrentV, **matricesCurrentVC;
  TMatrixD** matricesCurrentCharge;
  TMatrixD** residue;
  Int_t iOne, jOne, tnRRow, tnZColumn, count;

  matricesCurrentV = nullptr;
  matricesCurrentVC = nullptr;
  matricesCurrentCharge = nullptr;
  residue = nullptr;

  iOne = 1 << (gridFrom - 1);
  jOne = 1 << (gridFrom - 1);

  tnRRow = iOne == 1 ? nRRow : nRRow / iOne + 1;
  tnZColumn = jOne == 1 ? nZColumn : nZColumn / jOne + 1;

  for (count = gridFrom; count <= gridTo - 1; count++) {
    h = gridSizeR * iOne;
    h2 = h * h;
    ih2 = 1.0 / h2;

    tempRatioPhi = ratioPhi * iOne * iOne; // Used tobe divided by ( m_one * m_one ) when m_one was != 1
    tempRatioZ = ratioZ * iOne * iOne / (jOne * jOne);

    for (Int_t i = 1; i < tnRRow - 1; i++) {
      radius = AliTPCPoissonSolver::fgkIFCRadius + i * h;
      coefficient1[i] = 1.0 + h / (2 * radius);
      coefficient2[i] = 1.0 - h / (2 * radius);
      coefficient3[i] = tempRatioPhi / (radius * radius);
      coefficient4[i] = 0.5 / (1.0 + tempRatioZ + coefficient3[i]);
      inverseCoefficient4[i] = 1.0 / coefficient4[i];
    }

    matricesCurrentV = tvArrayV[count - 1];
    matricesCurrentCharge = tvCharge[count - 1];
    residue = tvResidue[count - 1];

    //Info("VCycle3D2D","Before Pre-smoothing");
    //matricesCurrentV->Print();

    // 1) Pre-Smoothing: Gauss-Seidel Relaxation or Jacobi
    for (Int_t jPre = 1; jPre <= nPre; jPre++) {
      Relax3D(matricesCurrentV, matricesCurrentCharge, tnRRow, tnZColumn, phiSlice, symmetry, h2, tempRatioZ,
              coefficient1, coefficient2,
              coefficient3, coefficient4);
    } // end pre smoothing

    // 2) Residue calculation
    Residue3D(residue, matricesCurrentV, matricesCurrentCharge, tnRRow, tnZColumn, phiSlice, symmetry, ih2, tempRatioZ,
              coefficient1,
              coefficient2,
              coefficient3, inverseCoefficient4);

    iOne = 2 * iOne;
    jOne = 2 * jOne;
    tnRRow = iOne == 1 ? nRRow : nRRow / iOne + 1;
    tnZColumn = jOne == 1 ? nZColumn : nZColumn / jOne + 1;

    matricesCurrentCharge = tvCharge[count];
    matricesCurrentV = tvArrayV[count];

    //3) Restriction
    //Restrict2D(*matricesCurrentCharge,*residue,tnRRow,tnZColumn);
    Restrict3D(matricesCurrentCharge, residue, tnRRow, tnZColumn, phiSlice, phiSlice);

    //4) Zeroing coarser V
    for (Int_t m = 0; m < phiSlice; m++) {
      matricesCurrentV[m]->Zero();
    }
  }

  // coarsest grid
  h = gridSizeR * iOne;
  h2 = h * h;

  tempRatioPhi = ratioPhi * iOne * iOne;
  tempRatioZ = ratioZ * iOne * iOne / (jOne * jOne);

  for (Int_t i = 1; i < tnRRow - 1; i++) {
    radius = AliTPCPoissonSolver::fgkIFCRadius + i * h;
    coefficient1[i] = 1.0 + h / (2 * radius);
    coefficient2[i] = 1.0 - h / (2 * radius);
    coefficient3[i] = tempRatioPhi / (radius * radius);
    coefficient4[i] = 0.5 / (1.0 + tempRatioZ + coefficient3[i]);
  }

  // 3) Relax on the coarsest grid
  Relax3D(matricesCurrentV, matricesCurrentCharge, tnRRow, tnZColumn, phiSlice, symmetry, h2, tempRatioZ, coefficient1,
          coefficient2,
          coefficient3, coefficient4);

  // back to fine
  for (count = gridTo - 1; count >= gridFrom; count--) {
    iOne = iOne / 2;
    jOne = jOne / 2;

    tnRRow = iOne == 1 ? nRRow : nRRow / iOne + 1;
    tnZColumn = jOne == 1 ? nZColumn : nZColumn / jOne + 1;

    h = gridSizeR * iOne;
    h2 = h * h;

    tempRatioPhi = ratioPhi * iOne * iOne; // Used tobe divided by ( m_one * m_one ) when m_one was != 1
    tempRatioZ = ratioZ * iOne * iOne / (jOne * jOne);

    matricesCurrentCharge = tvCharge[count - 1];
    matricesCurrentV = tvArrayV[count - 1];
    matricesCurrentVC = tvArrayV[count];

    // 4) Interpolation/Prolongation
    AddInterp3D(matricesCurrentV, matricesCurrentVC, tnRRow, tnZColumn, phiSlice, phiSlice);

    for (Int_t i = 1; i < tnRRow - 1; i++) {
      radius = AliTPCPoissonSolver::fgkIFCRadius + i * h;
      coefficient1[i] = 1.0 + h / (2 * radius);
      coefficient2[i] = 1.0 - h / (2 * radius);
      coefficient3[i] = tempRatioPhi / (radius * radius);
      coefficient4[i] = 0.5 / (1.0 + tempRatioZ + coefficient3[i]);
    }

    // 5) Post-Smoothing: Gauss-Seidel Relaxation
    for (Int_t jPost = 1; jPost <= nPost; jPost++) {
      Relax3D(matricesCurrentV, matricesCurrentCharge, tnRRow, tnZColumn, phiSlice, symmetry, h2, tempRatioZ,
              coefficient1, coefficient2,
              coefficient3, coefficient4);
    } // end post smoothing
  }
}

/// VCycle 3D, V Cycle in multiGrid, fine-->coarsest-->fine, propagating the residue to correct initial guess of V
///
///    NOTE: In order for this algorithm to work, the number of nRRow and nZColumn must be a power of 2 plus one.
///    The number of nRRow and Z Column can be different.
///
///    R Row       ==  2**M + 1
///    Z Column    ==  2**N + 1
///    Phi Slice  ==  Arbitrary but greater than 3
///
///    DeltaPhi in Radians
///
/// \param nRRow Int_t number of nRRow in the r direction of TPC
/// \param nZColumn Int_t number of nZColumn in z direction of TPC
/// \param phiSlice Int_t number of phiSlice in phi direction of T{C
/// \param symmetry Int_t symmetry or not
/// \param gridFrom const Int_t finest level of grid
/// \param gridTo const Int_t coarsest level of grid
/// \param nPre const Int_t number of smoothing before coarsening
/// \param nPost const Int_t number of smoothing after coarsening
/// \param gridSizeR const Float_t grid size in r direction (OPTION,  recalculate)
/// \param ratioz const Float_t ratio between square of grid r and grid z (OPTION,  recalculate)
/// \param tvArrayV vector<TMatrixD *> vector of V potential in different grids
/// \param tvCharge vector<TMatrixD *> vector of charge distribution in different grids
/// \param tvResidue vector<TMatrixD *> vector of residue calculation in different grids
/// \param coefficient1 std::vector<float>& coefficient for relaxation (r direction)
/// \param coefficient2 std::vector<float>& coefficient for relaxation (r direction)
/// \param coefficient3 std::vector<float>& coefficient for relaxation (ratio r/z)
/// \param coefficient4 std::vector<float>& coefficient for relaxation (ratio for grid_r)
/// \param inverseCoefficient4 std::vector<float>& coefficient for relaxation (inverse coefficient4)
///
void AliTPCPoissonSolver::VCycle3D(const Int_t nRRow, const Int_t nZColumn, const Int_t phiSlice, const Int_t symmetry,
                                   const Int_t gridFrom, const Int_t gridTo,
                                   const Int_t nPre, const Int_t nPost, const Float_t gridSizeR, const Float_t ratioZ,
                                   std::vector<TMatrixD**>& tvArrayV, std::vector<TMatrixD**>& tvCharge,
                                   std::vector<TMatrixD**>& tvResidue,
                                   std::vector<float>& coefficient1, std::vector<float>& coefficient2,
                                   std::vector<float>& coefficient3,
                                   std::vector<float>& coefficient4, std::vector<float>& inverseCoefficient4)
{

  Float_t h, h2, ih2, tempRatioZ, tempRatioPhi, radius, tempGridSizePhi;
  TMatrixD **matricesCurrentV, **matricesCurrentVC;
  TMatrixD** matricesCurrentCharge;
  TMatrixD** residue;
  Int_t iOne, jOne, kOne, tnRRow, tnZColumn, tPhiSlice, otPhiSlice, count, nnPhi;

  matricesCurrentV = nullptr;
  matricesCurrentVC = nullptr;
  matricesCurrentCharge = nullptr;
  residue = nullptr;

  iOne = 1 << (gridFrom - 1);
  jOne = 1 << (gridFrom - 1);
  kOne = 1 << (gridFrom - 1);

  nnPhi = phiSlice;

  while (nnPhi % 2 == 0) {
    nnPhi /= 2;
  }

  tnRRow = iOne == 1 ? nRRow : nRRow / iOne + 1;
  tnZColumn = jOne == 1 ? nZColumn : nZColumn / jOne + 1;
  tPhiSlice = kOne == 1 ? phiSlice : phiSlice / kOne;
  tPhiSlice = tPhiSlice < nnPhi ? nnPhi : tPhiSlice;

  //Info("VCycle3D",Form("Grid information: tnRRow=%d, tcols=%d, tPhiSlice=%d\n", tnRRow,tnZColumn,tPhiSlice));

  for (count = gridFrom; count <= gridTo - 1; count++) {
    otPhiSlice = tPhiSlice;

    h = gridSizeR * iOne;
    h2 = h * h;
    ih2 = 1.0 / h2;
    tempGridSizePhi = TMath::TwoPi() / tPhiSlice; // phi now is multiGrid

    tempRatioPhi = h * h / (tempGridSizePhi * tempGridSizePhi); // ratio_{phi} = gridSize_{r} / gridSize_{phi}

    tempRatioZ = ratioZ * iOne * iOne / (jOne * jOne);

    for (Int_t i = 1; i < tnRRow - 1; i++) {
      radius = AliTPCPoissonSolver::fgkIFCRadius + i * h;
      coefficient1[i] = 1.0 + h / (2 * radius);
      coefficient2[i] = 1.0 - h / (2 * radius);
      coefficient3[i] = tempRatioPhi / (radius * radius);
      coefficient4[i] = 0.5 / (1.0 + tempRatioZ + coefficient3[i]);
      inverseCoefficient4[i] = 1.0 / coefficient4[i];
    }

    matricesCurrentV = tvArrayV[count - 1];
    matricesCurrentCharge = tvCharge[count - 1];
    residue = tvResidue[count - 1];

    //Info("VCycle3D","Before Pre-smoothing");
    //matricesCurrentV->Print();

    // 1) Pre-Smoothing: Gauss-Seidel Relaxation or Jacobi
    for (Int_t jPre = 1; jPre <= nPre; jPre++) {
      Relax3D(matricesCurrentV, matricesCurrentCharge, tnRRow, tnZColumn, tPhiSlice, symmetry, h2, tempRatioZ,
              coefficient1, coefficient2, coefficient3, coefficient4);
    } // end pre smoothing

    // 2) Residue calculation

    Residue3D(residue, matricesCurrentV, matricesCurrentCharge, tnRRow, tnZColumn, tPhiSlice, symmetry, ih2, tempRatioZ,
              coefficient1, coefficient2, coefficient3, inverseCoefficient4);

    iOne = 2 * iOne;
    jOne = 2 * jOne;
    kOne = 2 * kOne;
    tnRRow = iOne == 1 ? nRRow : nRRow / iOne + 1;
    tnZColumn = jOne == 1 ? nZColumn : nZColumn / jOne + 1;
    tPhiSlice = phiSlice / kOne;
    tPhiSlice = tPhiSlice < nnPhi ? nnPhi : tPhiSlice;

    matricesCurrentCharge = tvCharge[count];
    matricesCurrentV = tvArrayV[count];
    //3) Restriction
    Restrict3D(matricesCurrentCharge, residue, tnRRow, tnZColumn, tPhiSlice, otPhiSlice);

    //4) Zeroing coarser V
    for (Int_t m = 0; m < tPhiSlice; m++) {
      matricesCurrentV[m]->Zero();
    }
  }

  // coarsest grid
  h = gridSizeR * iOne;
  h2 = h * h;
  tempGridSizePhi = TMath::TwoPi() / tPhiSlice; // phi now is multiGrid

  tempRatioPhi = h * h / (tempGridSizePhi * tempGridSizePhi); // ratio_{phi} = gridSize_{r} / gridSize_{phi}
  tempRatioZ = ratioZ * iOne * iOne / (jOne * jOne);

  for (Int_t i = 1; i < tnRRow - 1; i++) {
    radius = AliTPCPoissonSolver::fgkIFCRadius + i * h;
    coefficient1[i] = 1.0 + h / (2 * radius);
    coefficient2[i] = 1.0 - h / (2 * radius);
    coefficient3[i] = tempRatioPhi / (radius * radius);
    coefficient4[i] = 0.5 / (1.0 + tempRatioZ + coefficient3[i]);
  }

  // 3) Relax on the coarsest grid
  Relax3D(matricesCurrentV, matricesCurrentCharge, tnRRow, tnZColumn, tPhiSlice, symmetry, h2, tempRatioZ, coefficient1,
          coefficient2, coefficient3, coefficient4);

  // back to fine
  for (count = gridTo - 1; count >= gridFrom; count--) {
    otPhiSlice = tPhiSlice;

    iOne = iOne / 2;
    jOne = jOne / 2;
    kOne = kOne / 2;

    tnRRow = iOne == 1 ? nRRow : nRRow / iOne + 1;
    tnZColumn = jOne == 1 ? nZColumn : nZColumn / jOne + 1;
    tPhiSlice = kOne == 1 ? phiSlice : phiSlice / kOne;
    tPhiSlice = tPhiSlice < nnPhi ? nnPhi : tPhiSlice;

    h = gridSizeR * iOne;
    h2 = h * h;
    tempGridSizePhi = TMath::TwoPi() / tPhiSlice; // phi now is multiGrid

    tempRatioPhi = h * h / (tempGridSizePhi * tempGridSizePhi); // ratio_{phi} = gridSize_{r} / gridSize_{phi}

    tempRatioZ = ratioZ * iOne * iOne / (jOne * jOne);

    matricesCurrentCharge = tvCharge[count - 1];
    matricesCurrentV = tvArrayV[count - 1];
    matricesCurrentVC = tvArrayV[count];

    // 4) Interpolation/Prolongation

    AddInterp3D(matricesCurrentV, matricesCurrentVC, tnRRow, tnZColumn, tPhiSlice, otPhiSlice);

    for (Int_t i = 1; i < tnRRow - 1; i++) {
      radius = AliTPCPoissonSolver::fgkIFCRadius + i * h;
      coefficient1[i] = 1.0 + h / (2 * radius);
      coefficient2[i] = 1.0 - h / (2 * radius);
      coefficient3[i] = tempRatioPhi / (radius * radius);
      coefficient4[i] = 0.5 / (1.0 + tempRatioZ + coefficient3[i]);
    }

    // 5) Post-Smoothing: Gauss-Seidel Relaxation
    for (Int_t jPost = 1; jPost <= nPost; jPost++) {
      Relax3D(matricesCurrentV, matricesCurrentCharge, tnRRow, tnZColumn, tPhiSlice, symmetry, h2, tempRatioZ,
              coefficient1, coefficient2,
              coefficient3, coefficient4);
    }
  }
}

///
/// Set matrix exact solution for relative error calculation
///
/// \param exactSolution TMatrixD** pointer to exact solution (potential) in 3D
/// \param fPhiSlices const Int_t number of phi slices
///
void AliTPCPoissonSolver::SetExactSolution(TMatrixD** exactSolution, const Int_t fPhiSlices)
{
  Double_t maxAbs;
  fExactSolution = exactSolution;
  fExactPresent = kTRUE;
  fMaxExact = 0.0;
  for (Int_t m = 0; m < fPhiSlices; m++) {
    maxAbs = TMath::Max(TMath::Abs((*fExactSolution[m]).Max()), TMath::Abs((*fExactSolution[m]).Min()));
    if (maxAbs > fMaxExact) {
      fMaxExact = maxAbs;
    }
  }
}

///
/// Relative error calculation: comparison with exact solution
///
/// \param matricesCurrentV TMatrixD** current potential (numerical solution)
/// \param tempArrayV TMatrixD** temporary matrix for calculating error
/// \param nRRow Int_t number of grid in nRRow (in r-direction) for coarser grid should be 2^N + 1, finer grid in 2^{N+1} + 1
/// \param nZColumn Int_t number of grid in nZColumn (in z-direction) for coarser grid should be  2^M + 1, finer grid in 2^{M+1} + 1
/// \param phiSlice const Int_t phi slices
///
Double_t AliTPCPoissonSolver::GetExactError(TMatrixD** matricesCurrentV, TMatrixD** tempArrayV, const Int_t phiSlice)
{
  Double_t error = 0.0;

  if (fExactPresent == kTRUE) {
    for (Int_t m = 0; m < phiSlice; m++) {
      (*tempArrayV[m]) = (*fExactSolution[m]) - (*matricesCurrentV[m]);
      (*tempArrayV[m]) *= 1.0 / GetMaxExact();
      if (tempArrayV[m]->E2Norm() > error) {
        error = tempArrayV[m]->E2Norm();
      }
      //printf("%f\n",tempArrayV[m]->E2Norm();
    }
  }
  return error;
}

///
/// Relative error calculation: comparison with exact solution
///
/// \param matricesCurrentV TMatrixD** current potential (numerical solution)
/// \param tempArrayV TMatrixD** temporary matrix for calculating error
/// \param nRRow Int_t number of grid in nRRow (in r-direction) for coarser grid should be 2^N + 1, finer grid in 2^{N+1} + 1
/// \param nZColumn Int_t number of grid in nZColumn (in z-direction) for coarser grid should be  2^M + 1, finer grid in 2^{M+1} + 1
/// \param phiSlice const Int_t phi slices
///
Double_t
  AliTPCPoissonSolver::GetConvergenceError(TMatrixD** matricesCurrentV, TMatrixD** prevArrayV, const Int_t phiSlice)
{
  Double_t error = 0.0;

  for (Int_t m = 0; m < phiSlice; m++) {

    // absolute
    (*prevArrayV[m]) = (*prevArrayV[m]) - (*matricesCurrentV[m]);

    if (prevArrayV[m]->E2Norm() > error) {
      error = prevArrayV[m]->E2Norm();
    }
  }
  return error;
}

///////////////////// interface for GPU ///////////////////

/// VCycle 3D2D, V Cycle 3D in multiGrid with constant phiSlice
/// fine-->coarsest-->fine, propagating the residue to correct initial guess of V
///
/// Algorithm:
///
///    NOTE: In order for this algorithm to work, the number of nRRow and nZColumn must be a power of 2 plus one.
///    The number of nRRow and Z Column can be different.
///
///    R Row       ==  2**M + 1
///    Z Column    ==  2**N + 1
///    Phi Slice  ==  Arbitrary but greater than 3
///
///    DeltaPhi in Radians
/// \param nRRow Int_t number of grid in nRRow (in r-direction) for coarser grid should be 2^N + 1, finer grid in 2^{N+1} + 1
/// \param nZColumn Int_t number of grid in nZColumn (in z-direction) for coarser grid should be  2^M + 1, finer grid in 2^{M+1} + 1
/// \param gridFrom const Int_t finest level of grid
/// \param gridTo const Int_t coarsest level of grid
/// \param nPre const Int_t number of smoothing before coarsening
/// \param nPost const Int_t number of smoothing after coarsening
/// \param gridSizeR const Float_t grid size in r direction (OPTION,  recalculate)
/// \param ratio const Float_t ratio between square of grid r and grid z (OPTION,  recalculate)
/// \param tvArrayV vector<TMatrixD *> vector of V potential in different grids
/// \param tvCharge vector<TMatrixD *> vector of charge distribution in different grids
/// \param tvResidue vector<TMatrixD *> vector of residue calculation in different grids
/// \param coefficient1 std::vector<float>& coefficient for relaxation (r direction)
/// \param coefficient2 std::vector<float>& coefficient for relaxation (r direction)
/// \param coefficient3 std::vector<float>& coefficient for relaxation (ratio r/z)
/// \param coefficient4 std::vector<float>& coefficient for relaxation (ratio for grid_r)
/// \param inverseCoefficient4 std::vector<float>& coefficient for relaxation (inverse coefficient4)
///
void AliTPCPoissonSolver::VCycle3D2DGPU(
  const Int_t nRRow, const Int_t nZColumn, const Int_t phiSlice, const Int_t symmetry,
  const Int_t gridFrom, const Int_t gridTo, const Int_t nPre, const Int_t nPost, const Float_t gridSizeR,
  const Float_t ratioZ, const Float_t ratioPhi, std::vector<TMatrixD**>& tvArrayV,
  std::vector<TMatrixD**>& tvCharge, std::vector<TMatrixD**>& tvResidue, std::vector<float>& coefficient1,
  std::vector<float>& coefficient2, std::vector<float>& coefficient3, std::vector<float>& coefficient4,
  std::vector<float>& inverseCoefficient4)
{
  Float_t h, h2, ih2, tempRatioZ, tempRatioPhi, radius;
  TMatrixD **matricesCurrentV, **matricesCurrentVC;
  TMatrixD** matricesCurrentCharge;
  TMatrixD** residue;
  Int_t iOne, jOne, tnRRow, tnZColumn, count;

  matricesCurrentV = nullptr;
  matricesCurrentVC = nullptr;
  matricesCurrentCharge = nullptr;
  residue = nullptr;

  iOne = 1 << (gridFrom - 1);
  jOne = 1 << (gridFrom - 1);

  tnRRow = iOne == 1 ? nRRow : nRRow / iOne + 1;
  tnZColumn = jOne == 1 ? nZColumn : nZColumn / jOne + 1;

  for (count = gridFrom; count <= gridTo - 1; count++) {
    h = gridSizeR * iOne;
    h2 = h * h;
    ih2 = 1.0 / h2;

    tempRatioPhi = ratioPhi * iOne * iOne; // Used tobe divided by ( m_one * m_one ) when m_one was != 1
    tempRatioZ = ratioZ * iOne * iOne / (jOne * jOne);

    for (Int_t i = 1; i < tnRRow - 1; i++) {
      radius = AliTPCPoissonSolver::fgkIFCRadius + i * h;
      coefficient1[i] = 1.0 + h / (2 * radius);
      coefficient2[i] = 1.0 - h / (2 * radius);
      coefficient3[i] = tempRatioPhi / (radius * radius);
      coefficient4[i] = 0.5 / (1.0 + tempRatioZ + coefficient3[i]);
      inverseCoefficient4[i] = 1.0 / coefficient4[i];
    }

    matricesCurrentV = tvArrayV[count - 1];
    matricesCurrentCharge = tvCharge[count - 1];
    residue = tvResidue[count - 1];

    //Info("VCycle3D2DGPU","Before Pre-smoothing");
    //matricesCurrentV->Print();

    // 1) Pre-Smoothing: Gauss-Seidel Relaxation or Jacobi
    for (Int_t jPre = 1; jPre <= nPre; jPre++) {
      Relax3D(matricesCurrentV, matricesCurrentCharge, tnRRow, tnZColumn, phiSlice, symmetry, h2, tempRatioZ,
              coefficient1, coefficient2,
              coefficient3, coefficient4);
    } // end pre smoothing

    // 2) Residue calculation
    Residue3D(residue, matricesCurrentV, matricesCurrentCharge, tnRRow, tnZColumn, phiSlice, symmetry, ih2, tempRatioZ,
              coefficient1,
              coefficient2,
              coefficient3, inverseCoefficient4);

    iOne = 2 * iOne;
    jOne = 2 * jOne;
    tnRRow = iOne == 1 ? nRRow : nRRow / iOne + 1;
    tnZColumn = jOne == 1 ? nZColumn : nZColumn / jOne + 1;

    matricesCurrentCharge = tvCharge[count];
    matricesCurrentV = tvArrayV[count];

    //3) Restriction
    //Restrict2D(*matricesCurrentCharge,*residue,tnRRow,tnZColumn);
    Restrict3D(matricesCurrentCharge, residue, tnRRow, tnZColumn, phiSlice, phiSlice);

    //4) Zeroing coarser V
    for (Int_t m = 0; m < phiSlice; m++) {
      matricesCurrentV[m]->Zero();
    }
  }

  // coarsest grid
  h = gridSizeR * iOne;
  h2 = h * h;

  tempRatioPhi = ratioPhi * iOne * iOne;
  tempRatioZ = ratioZ * iOne * iOne / (jOne * jOne);

  for (Int_t i = 1; i < tnRRow - 1; i++) {
    radius = AliTPCPoissonSolver::fgkIFCRadius + i * h;
    coefficient1[i] = 1.0 + h / (2 * radius);
    coefficient2[i] = 1.0 - h / (2 * radius);
    coefficient3[i] = tempRatioPhi / (radius * radius);
    coefficient4[i] = 0.5 / (1.0 + tempRatioZ + coefficient3[i]);
  }

  // 3) Relax on the coarsest grid
  Relax3D(matricesCurrentV, matricesCurrentCharge, tnRRow, tnZColumn, phiSlice, symmetry, h2, tempRatioZ, coefficient1,
          coefficient2,
          coefficient3, coefficient4);

  // back to fine
  for (count = gridTo - 1; count >= gridFrom; count--) {
    iOne = iOne / 2;
    jOne = jOne / 2;

    tnRRow = iOne == 1 ? nRRow : nRRow / iOne + 1;
    tnZColumn = jOne == 1 ? nZColumn : nZColumn / jOne + 1;

    h = gridSizeR * iOne;
    h2 = h * h;

    tempRatioPhi = ratioPhi * iOne * iOne; // Used tobe divided by ( m_one * m_one ) when m_one was != 1
    tempRatioZ = ratioZ * iOne * iOne / (jOne * jOne);

    matricesCurrentCharge = tvCharge[count - 1];
    matricesCurrentV = tvArrayV[count - 1];
    matricesCurrentVC = tvArrayV[count];

    // 4) Interpolation/Prolongation
    AddInterp3D(matricesCurrentV, matricesCurrentVC, tnRRow, tnZColumn, phiSlice, phiSlice);

    for (Int_t i = 1; i < tnRRow - 1; i++) {
      radius = AliTPCPoissonSolver::fgkIFCRadius + i * h;
      coefficient1[i] = 1.0 + h / (2 * radius);
      coefficient2[i] = 1.0 - h / (2 * radius);
      coefficient3[i] = tempRatioPhi / (radius * radius);
      coefficient4[i] = 0.5 / (1.0 + tempRatioZ + coefficient3[i]);
    }

    // 5) Post-Smoothing: Gauss-Seidel Relaxation
    for (Int_t jPost = 1; jPost <= nPost; jPost++) {
      Relax3D(matricesCurrentV, matricesCurrentCharge, tnRRow, tnZColumn, phiSlice, symmetry, h2, tempRatioZ,
              coefficient1, coefficient2,
              coefficient3, coefficient4);
    } // end post smoothing
  }
}
