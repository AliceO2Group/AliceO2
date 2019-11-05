// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

/// \file AliTPCPoissonSolver.h
/// \brief This class provides implementation of Poisson Eq
/// solver by MultiGrid Method
///
///
///
/// \author Rifki Sadikin <rifki.sadikin@cern.ch>, Indonesian Institute of Sciences
/// \date Nov 20, 2017

#ifndef ALITPCPOISSONSOLVER_H
#define ALITPCPOISSONSOLVER_H

#include <TNamed.h>
#include "TMatrixD.h"
#include "TVectorD.h"

class AliTPCPoissonSolver : public TNamed
{
 public:
  ///< Enumeration of Poisson Solver Strategy Type
  enum StrategyType {
    kRelaxation = 0,    ///< S.O.R Cascaded MultiGrid
    kMultiGrid = 1,     ///< Geometric MG
    kFastRelaxation = 2 ///< Spectral (TODO)
  };

  ///< Enumeration of Cycles Type
  enum CycleType {
    kVCycle = 0, ///< V Cycle
    kWCycle = 1, ///< W Cycle (TODO)
    kFCycle = 2  ///< Full Cycle
  };

  ///< Fine -> Coarse Grid transfer operator types
  enum GridTransferType {
    kHalf = 0, ///< Half weighting
    kFull = 1, ///< Full weighting
  };

  ///< Smoothing (Relax) operator types
  enum RelaxType {
    kJacobi = 0,         ///< Jacobi (5 Stencil 2D, 7 Stencil 3D_
    kWeightedJacobi = 1, ///< (TODO)
    kGaussSeidel = 2     ///< Gauss Seidel 2D (2 Color, 5 Stencil), 3D (7 Stencil)
  };

  ///< Coarse -> fine  operator types (TODO: Interp and Restrict in one packet, just one enumeration)
  enum InterpType {
    kHalfInterp = 0, ///< Half bi linear interpolation
    kFullInterp = 1  ///< Full bi linear interpolation
  };

  ///< Parameters choice for MultiGrid    algorithm
  struct MGParameters {
    Bool_t isFull3D;         ///<  TRUE: full coarsening, FALSE: semi coarsening
    CycleType cycleType;     ///< cycleType follow  CycleType
    GridTransferType gtType; ///< gtType grid transfer type follow GridTransferType
    RelaxType relaxType;     ///< relaxType follow RelaxType
    Int_t gamma;             ///< number of iteration at coarsest level
    Int_t nPre;              ///< number of iteration for pre smoothing
    Int_t nPost;             ///< number of iteration for post smoothing
    Int_t nMGCycle;          ///< number of multi grid cycle (V type)
    Int_t maxLoop;           ///< the number of tree-deep of multi grid

    // default values
    MGParameters()
    {
      isFull3D = kFALSE;
      cycleType = kFCycle;
      gtType = kFull;           // default full
      relaxType = kGaussSeidel; // default relaxation method
      nPre = 2;
      nPost = 2;
      nMGCycle = 200;
      maxLoop = 6;
    }
  };

  AliTPCPoissonSolver();
  AliTPCPoissonSolver(const char* name, const char* title);
#if (defined(__CINT__) || defined(__ROOTCINT__)) && !defined(__CLING__)
  ~AliTPCPoissonSolver();
#else
  ~AliTPCPoissonSolver() override;
#endif
  void PoissonSolver2D(TMatrixD& matrixV, TMatrixD& chargeDensity, Int_t nRRow, Int_t nZColumn, Int_t maxIterations);
  void PoissonSolver3D(TMatrixD** matricesV, TMatrixD** matricesChargeDensities, Int_t nRRow, Int_t nZColumn,
                       Int_t phiSlice, Int_t maxIterations, Int_t symmetry);

  void SetStrategy(StrategyType strategy) { fStrategy = strategy; }
  StrategyType GetStrategy() { return fStrategy; }

  static const Double_t fgkTPCZ0;     ///< nominal gating grid position
  static const Double_t fgkIFCRadius; ///< Mean Radius of the Inner Field Cage ( 82.43 min,  83.70 max) (cm)
  static const Double_t fgkOFCRadius; ///< Mean Radius of the Outer Field Cage (252.55 min, 256.45 max) (cm)
  static const Double_t fgkZOffSet;   ///< Offset from CE: calculate all distortions closer to CE as if at this point
  static const Double_t fgkCathodeV;  ///< Cathode Voltage (volts)
  static const Double_t fgkGG;        ///< Gating Grid voltage (volts)
  static const Double_t fgkdvdE;      ///< [cm/V] drift velocity dependency on the E field (from Magboltz for NeCO2N2 at standard environment)
  static const Double_t fgkEM;        ///< charge/mass in [C/kg]
  static const Double_t fgke0;        ///< vacuum permittivity [A·s/(V·m)]

  static Double_t fgExactErr;         ///< Error tolerated
  static Double_t fgConvergenceError; ///< Error tolerated
  Int_t fIterations;                  ///< number of maximum iteration
  MGParameters fMgParameters;         ///< parameters multi grid

  void SetExactSolution(TMatrixD** exactSolution, const Int_t fPhiSlices);
  void SetCycleType(AliTPCPoissonSolver::CycleType cycleType) { fMgParameters.cycleType = cycleType; }

 private:
  AliTPCPoissonSolver(const AliTPCPoissonSolver&);            // not implemented
  AliTPCPoissonSolver& operator=(const AliTPCPoissonSolver&); // not implemented
  StrategyType fStrategy = kMultiGrid;                        ///< strategy used default multiGrid
  TMatrixD** fExactSolution = nullptr;                        //!<! Pointer to exact solution
  /// TODO: remove pointers?
  TVectorD* fErrorConvergenceNorm2;   ///< for storing convergence error  norm2
  TVectorD* fErrorConvergenceNormInf; ///< for storing convergence error normInf
  TVectorD* fError;                   ///< for storing error
  Double_t GetMaxExact() { return fMaxExact; };

  void PoissonRelaxation2D(TMatrixD& matrixV, TMatrixD& chargeDensity, Int_t nRRow, Int_t nZColumn,
                           Int_t maxIterations);
  void PoissonRelaxation3D(TMatrixD** matricesV, TMatrixD** matricesChargeDensities, Int_t nRRow,
                           Int_t nZColumn, Int_t phiSlice, Int_t maxIterations, Int_t symmetry);
  void PoissonMultiGrid2D(TMatrixD& matrixV, TMatrixD& chargeDensity, Int_t nRRow, Int_t nZColumn);
  void PoissonMultiGrid3D2D(TMatrixD** matricesV, TMatrixD** matricesChargeDensities, Int_t nRRow,
                            Int_t nZColumn, Int_t phiSlice, Int_t symmetry);
  void PoissonMultiGrid3D(TMatrixD** matricesV, TMatrixD** matricesChargeDensities, Int_t nRRow,
                          Int_t nZColumn, Int_t phiSlice, Int_t symmetry);
  Int_t IsPowerOfTwo(Int_t i) const;
  void Relax2D(TMatrixD& matrixV, TMatrixD& matrixCharge, const Int_t tnRRow, const Int_t tnZColumn,
               const Float_t h2, const Float_t tempFourth, const Float_t tempRatio,
               std::vector<float>& vectorCoefficient1,
               std::vector<float>& vectorCoefficient2);
  void Relax3D(TMatrixD** currentMatricesV, TMatrixD** matricesCharge, const Int_t tnRRow, const Int_t tnZColumn,
               const Int_t phiSlice, const Int_t symmetry, const Float_t h2, const Float_t tempRatioZ,
               std::vector<float>& vectorCoefficient1, std::vector<float>& vectorCoefficient2,
               std::vector<float>& vectorCoefficient3,
               std::vector<float>& vectorCoefficient4);
  void Residue2D(TMatrixD& residue, TMatrixD& matrixV, TMatrixD& matrixCharge,
                 const Int_t tnRRow, const Int_t tnZColumn, const Float_t ih2, const Float_t iTempFourth,
                 const Float_t tempRatio, std::vector<float>& vectorCoefficient1,
                 std::vector<float>& vectorCoefficient2);
  void Residue3D(TMatrixD** residue, TMatrixD** currentMatricesV, TMatrixD** matricesCharge, const Int_t tnRRow,
                 const Int_t tnZColumn, const Int_t phiSlice, const Int_t symmetry, const Float_t ih2,
                 const Float_t tempRatio, std::vector<float>& vectorCoefficient1,
                 std::vector<float>& vectorCoefficient2,
                 std::vector<float>& vectorCoefficient3, std::vector<float>& vectorInverseCoefficient4);
  void Restrict2D(TMatrixD& matrixCharge, TMatrixD& residue, const Int_t tnRRow, const Int_t tnZColumn);
  void Restrict3D(TMatrixD** matricesCharge, TMatrixD** residue, const Int_t tnRRow, const Int_t tnZColumn,
                  const Int_t newPhiSlice, const Int_t oldPhiSlice);
  void RestrictBoundary2D(TMatrixD& matrixCharge, TMatrixD& residue, const Int_t tnRRow, const Int_t tnZColumn);
  void RestrictBoundary3D(TMatrixD** matricesCharge, TMatrixD** residue, const Int_t tnRRow, const Int_t tnZColumn,
                          const Int_t newPhiSlice, const Int_t oldPhiSlice);

  void AddInterp2D(TMatrixD& matrixV, TMatrixD& matrixVC, const Int_t tnRRow, const Int_t tnZColumn);
  void AddInterp3D(TMatrixD** currentMatricesV, TMatrixD** currentMatricesVC, const Int_t tnRRow, const Int_t tnZColumn,
                   const Int_t newPhiSlice, const Int_t oldPhiSlice);
  void Interp2D(TMatrixD& matrixV, TMatrixD& matrixVC, const Int_t tnRRow, const Int_t tnZColumn);

  void Interp3D(TMatrixD** currentMatricesV, TMatrixD** currentMatricesVC, const Int_t tnRRow, const Int_t tnZColumn,
                const Int_t newPhiSlice, const Int_t oldPhiSlice);
  void VCycle2D(const Int_t nRRow, const Int_t nZColumn, const Int_t gridFrom, const Int_t gridTo, const Int_t nPre,
                const Int_t nPost, const Float_t gridSizeR, const Float_t ratio, std::vector<TMatrixD*>& tvArrayV,
                std::vector<TMatrixD*>& tvCharge, std::vector<TMatrixD*>& tvResidue);
  void WCycle2D(const Int_t nRRow, const Int_t nZColumn, const Int_t gridFrom, const Int_t gridTo, const Int_t gamma,
                const Int_t nPre, const Int_t nPost, const Float_t gridSizeR, const Float_t ratio,
                std::vector<TMatrixD*>& tvArrayV,
                std::vector<TMatrixD*>& tvCharge, std::vector<TMatrixD*>& tvResidue);
  void
    VCycle3D(const Int_t nRRow, const Int_t nZColumn, const Int_t phiSlice, const Int_t symmetry, const Int_t gridFrom,
             const Int_t gridTo, const Int_t nPre, const Int_t nPost, const Float_t gridSizeR, const Float_t ratioZ,
             std::vector<TMatrixD**>& tvArrayV, std::vector<TMatrixD**>& tvCharge,
             std::vector<TMatrixD**>& tvResidue, std::vector<float>& vectorCoefficient1,
             std::vector<float>& vectorCoefficient2,
             std::vector<float>& vectorCoefficient3, std::vector<float>& vectorCoefficient4,
             std::vector<float>& vectorInverseCoefficient4);
  void VCycle3D2D(const Int_t nRRow, const Int_t nZColumn, const Int_t phiSlice, const Int_t symmetry,
                  const Int_t gridFrom, const Int_t gridTo, const Int_t nPre, const Int_t nPost,
                  const Float_t gridSizeR,
                  const Float_t ratioZ, const Float_t ratioPhi, std::vector<TMatrixD**>& tvArrayV,
                  std::vector<TMatrixD**>& tvCharge, std::vector<TMatrixD**>& tvResidue,
                  std::vector<float>& vectorCoefficient1,
                  std::vector<float>& vectorCoefficient2, std::vector<float>& vectorCoefficient3,
                  std::vector<float>& vectorCoefficient4,
                  std::vector<float>& vectorInverseCoefficient4);
  void VCycle3D2DGPU(const Int_t nRRow, const Int_t nZColumn, const Int_t phiSlice, const Int_t symmetry,
                     const Int_t gridFrom, const Int_t gridTo, const Int_t nPre,
                     const Int_t nPost, const Float_t gridSizeR, const Float_t ratioZ, const Float_t ratioPhi,
                     std::vector<TMatrixD**>& tvArrayV, std::vector<TMatrixD**>& tvCharge,
                     std::vector<TMatrixD**>& tvResidue, std::vector<float>& vectorCoefficient1,
                     std::vector<float>& vectorCoefficient2,
                     std::vector<float>& vectorCoefficient3, std::vector<float>& vectorCoefficient4,
                     std::vector<float>& vectorInverseCoefficient4);
  Double_t GetExactError(TMatrixD** currentMatricesV, TMatrixD** tempArrayV, const Int_t phiSlice);
  Double_t GetConvergenceError(TMatrixD** currentMatricesV, TMatrixD** prevArrayV, const Int_t phiSlice);
  Double_t fMaxExact;
  Bool_t fExactPresent = kFALSE;
  /// \cond CLASSIMP
#if defined(ROOT_VERSION_CODE) && ROOT_VERSION_CODE >= ROOT_VERSION(6, 0, 0)
  ClassDefOverride(AliTPCPoissonSolver, 5);
#else
  ClassDefNV(AliTPCPoissonSolver, 5);
#endif
  /// \endcond
};

#endif
