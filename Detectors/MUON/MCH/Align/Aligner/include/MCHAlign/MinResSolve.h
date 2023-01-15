#ifndef ALICEO2_MCH_MINRESSOLVE_H
#define ALICEO2_MCH_MINRESSOLVE_H

#include <TObject.h>
#include <TVectorD.h>
#include <TString.h>

namespace o2
{
namespace mch
{

class MatrixSq;
class MatrixSparse;
class SymBDMatrix;

class MinResSolve : public TObject
{
  //
 public:
  enum { kPreconBD = 1,
         kPreconILU0 = 100,
         kPreconILU10 = kPreconILU0 + 10,
         kPreconsTot };
  enum { kSolMinRes,
         kSolFGMRes,
         kNSolvers };

 public:
  MinResSolve();
  MinResSolve(const MatrixSq* mat, const TVectorD* rhs);
  MinResSolve(const MatrixSq* mat, const double* rhs);
  MinResSolve(const MinResSolve& src);
  ~MinResSolve() override;
  MinResSolve& operator=(const MinResSolve& rhs);
  //
  // ---------  MINRES method (for symmetric matrices)
  bool SolveMinRes(double* VecSol, int precon = 0, int itnlim = 2000, double rtol = 1e-12);
  bool SolveMinRes(TVectorD& VecSol, int precon = 0, int itnlim = 2000, double rtol = 1e-12);
  //
  // ---------  FGMRES method (for general symmetric matrices)
  bool SolveFGMRES(double* VecSol, int precon = 0, int itnlim = 2000, double rtol = 1e-12, int nkrylov = 60);
  bool SolveFGMRES(TVectorD& VecSol, int precon = 0, int itnlim = 2000, double rtol = 1e-12, int nkrylov = 60);
  //
  bool InitAuxMinRes();
  bool InitAuxFGMRES(int nkrylov);
  void ApplyPrecon(const TVectorD& vecRHS, TVectorD& vecOut) const;
  void ApplyPrecon(const double* vecRHS, double* vecOut) const;
  //
  int BuildPrecon(int val = 0);
  int GetPrecon() const { return fPrecon; }
  void ClearAux();
  //
  int BuildPreconBD(int hwidth);
  int BuildPreconILUK(int lofM);
  int BuildPreconILUKDense(int lofM);
  int PreconILUKsymb(int lofM);
  int PreconILUKsymbDense(int lofM);
  //
 protected:
  //
  int fSize;         // dimension of the input matrix
  int fPrecon;       // preconditioner type
  MatrixSq* fMatrix; // matrix defining the equations
  double* fRHS;      // right hand side
  //
  double* fPVecY;      // aux. space
  double* fPVecR1;     // aux. space
  double* fPVecR2;     // aux. space
  double* fPVecV;      // aux. space
  double* fPVecW;      // aux. space
  double* fPVecW1;     // aux. space
  double* fPVecW2;     // aux. space
  double** fPvv;       // aux. space
  double** fPvz;       // aux. space
  double** fPhh;       // aux. space
  double* fDiagLU;     // aux space
  MatrixSparse* fMatL; // aux. space
  MatrixSparse* fMatU; // aux. space
  SymBDMatrix* fMatBD; // aux. space
  //
  ClassDefOverride(MinResSolve, 0)
};
} // namespace mch
} // namespace o2

#endif
