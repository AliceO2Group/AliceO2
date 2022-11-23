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

/// \file MinResSolve.h
/// \brief General class (from AliROOT) for solving large system of linear equations
/// \author ruben.shahoyan@cern.ch

#ifndef ALICEO2_MFT_MINRESSOLVE_H
#define ALICEO2_MFT_MINRESSOLVE_H

#include <TObject.h>
#include <TVectorD.h>
#include <TString.h>

namespace o2
{
namespace mft
{

class MatrixSq;
class MatrixSparse;
class SymBDMatrix;

/// \class MinResSolve
/// \brief for solving large system of linear equations
///
/// Includes MINRES, FGMRES methods as well as a few precondiotiong methods
class MinResSolve : public TObject
{

 public:
  enum { kPreconBD = 1,
         kPreconILU0 = 100,
         kPreconILU10 = kPreconILU0 + 10,
         kPreconsTot };
  enum { kSolMinRes,
         kSolFGMRes,
         kNSolvers };

 public:
  /// \brief default constructor
  MinResSolve();

  /// \brief copy accepting equation
  MinResSolve(const MatrixSq* mat, const TVectorD* rhs);

  /// \brief copy accepting equation
  MinResSolve(const MatrixSq* mat, const double* rhs);

  /// \brief copy constructor
  MinResSolve(const MinResSolve& src);

  /// \brief destructor
  ~MinResSolve() override;

  /// \brief assignment op.
  MinResSolve& operator=(const MinResSolve& rhs);

  /// \brief MINRES method (for symmetric matrices)
  Bool_t SolveMinRes(Double_t* VecSol, Int_t precon = 0, int itnlim = 2000, double rtol = 1e-12);

  /// \brief MINRES method (for symmetric matrices)
  Bool_t SolveMinRes(TVectorD& VecSol, Int_t precon = 0, int itnlim = 2000, double rtol = 1e-12);

  /// \brief FGMRES method (for general symmetric matrices)
  Bool_t SolveFGMRES(Double_t* VecSol, Int_t precon = 0, int itnlim = 2000, double rtol = 1e-12, int nkrylov = 60);

  /// \brief FGMRES method (for general symmetric matrices)
  Bool_t SolveFGMRES(TVectorD& VecSol, Int_t precon = 0, int itnlim = 2000, double rtol = 1e-12, int nkrylov = 60);

  /// \brief init auxiliary space for MinRes
  Bool_t InitAuxMinRes();

  /// \brief init auxiliary space for fgmres
  Bool_t InitAuxFGMRES(int nkrylov);

  /// \brief apply precond.
  void ApplyPrecon(const TVectorD& vecRHS, TVectorD& vecOut) const;

  /// \brief Application of preconditioner matrix: implicitly defines the matrix solving the M*VecOut = VecRHS
  void ApplyPrecon(const double* vecRHS, double* vecOut) const;

  /// \brief preconditioner building
  Int_t BuildPrecon(Int_t val = 0);

  Int_t GetPrecon() const { return fPrecon; }

  /// \brief clear aux. space
  void ClearAux();

  /// \brief build Band-Diagonal preconditioner
  Int_t BuildPreconBD(Int_t hwidth);

  /// \brief ILUK preconditioner
  Int_t BuildPreconILUK(Int_t lofM);

  /// \brief ILUK preconditioner
  Int_t BuildPreconILUKDense(Int_t lofM);

  /// \brief ILUK preconditioner
  Int_t PreconILUKsymb(Int_t lofM);

  /// \brief ILUK preconditioner
  Int_t PreconILUKsymbDense(Int_t lofM);

 protected:
  Int_t fSize;       ///< dimension of the input matrix
  Int_t fPrecon;     ///< preconditioner type
  MatrixSq* fMatrix; ///< matrix defining the equations
  Double_t* fRHS;    ///< right hand side

  Double_t* fPVecY;    ///< aux. space
  Double_t* fPVecR1;   // aux. space
  Double_t* fPVecR2;   // aux. space
  Double_t* fPVecV;    // aux. space
  Double_t* fPVecW;    // aux. space
  Double_t* fPVecW1;   // aux. space
  Double_t* fPVecW2;   // aux. space
  Double_t** fPvv;     // aux. space
  Double_t** fPvz;     // aux. space
  Double_t** fPhh;     // aux. space
  Double_t* fDiagLU;   // aux space
  MatrixSparse* fMatL; // aux. space
  MatrixSparse* fMatU; // aux. space
  SymBDMatrix* fMatBD; // aux. space

  ClassDefOverride(MinResSolve, 0);
};

} // namespace mft
} // namespace o2

#endif
