#include "MCHAlign/MinResSolve.h"
#include "Framework/Logger.h"
#include "MCHAlign/MatrixSq.h"
#include "MCHAlign/MatrixSparse.h"
#include "MCHAlign/SymBDMatrix.h"
#include <TStopwatch.h>
#include <float.h>
#include <TMath.h>

using namespace o2::mch;

ClassImp(MinResSolve);

//______________________________________________________
MinResSolve::MinResSolve()
  : fSize(0),
    fPrecon(0),
    fMatrix(nullptr),
    fRHS(nullptr),
    fPVecY(nullptr),
    fPVecR1(nullptr),
    fPVecR2(nullptr),
    fPVecV(nullptr),
    fPVecW(nullptr),
    fPVecW1(nullptr),
    fPVecW2(nullptr),
    fPvv(nullptr),
    fPvz(nullptr),
    fPhh(nullptr),
    fDiagLU(nullptr),
    fMatL(nullptr),
    fMatU(nullptr),
    fMatBD(nullptr)
{
  // default constructor
}

//______________________________________________________
MinResSolve::MinResSolve(const MinResSolve& src)
  : TObject(src),
    fSize(src.fSize),
    fPrecon(src.fPrecon),
    fMatrix(src.fMatrix),
    fRHS(src.fRHS),
    fPVecY(nullptr),
    fPVecR1(nullptr),
    fPVecR2(nullptr),
    fPVecV(nullptr),
    fPVecW(nullptr),
    fPVecW1(nullptr),
    fPVecW2(nullptr),
    fPvv(nullptr),
    fPvz(nullptr),
    fPhh(nullptr),
    fDiagLU(nullptr),
    fMatL(nullptr),
    fMatU(nullptr),
    fMatBD(nullptr)
{
  // copy constructor
}

//______________________________________________________
MinResSolve::MinResSolve(const MatrixSq* mat, const TVectorD* rhs)
  : fSize(mat->GetSize()),
    fPrecon(0),
    fMatrix((MatrixSq*)mat),
    fRHS((double*)rhs->GetMatrixArray()),
    fPVecY(nullptr),
    fPVecR1(nullptr),
    fPVecR2(nullptr),
    fPVecV(nullptr),
    fPVecW(nullptr),
    fPVecW1(nullptr),
    fPVecW2(nullptr),
    fPvv(nullptr),
    fPvz(nullptr),
    fPhh(nullptr),
    fDiagLU(nullptr),
    fMatL(nullptr),
    fMatU(nullptr),
    fMatBD(nullptr)
{
  // copy accepting equation
}

//______________________________________________________
MinResSolve::MinResSolve(const MatrixSq* mat, const double* rhs)
  : fSize(mat->GetSize()),
    fPrecon(0),
    fMatrix((MatrixSq*)mat),
    fRHS((double*)rhs),
    fPVecY(nullptr),
    fPVecR1(nullptr),
    fPVecR2(nullptr),
    fPVecV(nullptr),
    fPVecW(nullptr),
    fPVecW1(nullptr),
    fPVecW2(nullptr),
    fPvv(nullptr),
    fPvz(nullptr),
    fPhh(nullptr),
    fDiagLU(nullptr),
    fMatL(nullptr),
    fMatU(nullptr),
    fMatBD(nullptr)
{
  // copy accepting equation
}

//______________________________________________________
MinResSolve::~MinResSolve()
{
  // destructor
  ClearAux();
}

//______________________________________________________
MinResSolve& MinResSolve::operator=(const MinResSolve& src)
{
  // assignment op.
  if (this != &src) {
    fSize = src.fSize;
    fPrecon = src.fPrecon;
    fMatrix = src.fMatrix;
    fRHS = src.fRHS;
  }
  return *this;
}

//_______________________________________________________________
int MinResSolve::BuildPrecon(int prec)
{
  // preconditioner building
  //  const double kTiny = 1E-12;
  fPrecon = prec;
  //
  if (fPrecon >= kPreconBD && fPrecon < kPreconILU0) { // band diagonal decomposition
    return BuildPreconBD(fPrecon - kPreconBD);         // with halfbandwidth + diagonal = fPrecon
  }
  //
  if (fPrecon >= kPreconILU0 && fPrecon <= kPreconILU10) {
    if (fMatrix->InheritsFrom("MatrixSparse")) {
      return BuildPreconILUK(fPrecon - kPreconILU0);
    } else {
      return BuildPreconILUKDense(fPrecon - kPreconILU0);
    }
  }
  //
  return -1;
}

//________________________________ FGMRES METHODS ________________________________
bool MinResSolve::SolveFGMRES(TVectorD& VecSol, int precon, int itnlim, double rtol, int nkrylov)
{
  // solve by fgmres
  return SolveFGMRES(VecSol.GetMatrixArray(), precon, itnlim, rtol, nkrylov);
}

//________________________________________________________________________________
bool MinResSolve::SolveFGMRES(double* VecSol, int precon, int itnlim, double rtol, int nkrylov)
{
  // Adapted from Y.Saad fgmrs.c of ITSOL_1 package by Y.Saad: http://www-users.cs.umn.edu/~saad/software/
  /*----------------------------------------------------------------------
    |                 *** Preconditioned FGMRES ***
    +-----------------------------------------------------------------------
    | This is a simple version of the ARMS preconditioned FGMRES algorithm.
    +-----------------------------------------------------------------------
    | Y. S. Dec. 2000. -- Apr. 2008
    +-----------------------------------------------------------------------
    | VecSol  = real vector of length n containing an initial guess to the
    | precon  = precondtioner id (0 = no precon)
    | itnlim  = max n of iterations
    | rtol     = tolerance for stopping iteration
    | nkrylov = N of Krylov vectors to store
    +---------------------------------------------------------------------*/
  int l;
  double status = true;
  double t, beta, eps1 = 0;
  const double epsmac = 2.22E-16;
  //
  LOG(info) << Form("Solution by FGMRes: Preconditioner#%d Max.iter.: %d Tol.: %e NKrylov: %d",
                    precon, itnlim, rtol, nkrylov);
  //
  int its = 0;
  if (nkrylov < 10) {
    LOG(info) << Form("Changing N Krylov vectors from %d to %d", nkrylov, 10);
    nkrylov = 10;
  }
  //
  if (precon > 0) {
    if (precon >= kPreconsTot) {
      LOG(info) << Form("Unknown preconditioner identifier %d, ignore", precon);
    } else {
      if (BuildPrecon(precon) < 0) {
        ClearAux();
        LOG(info) << "FGMRES failed to build the preconditioner";
        return false;
      }
    }
  }
  //
  if (!InitAuxFGMRES(nkrylov)) {
    return false;
  }
  //
  for (l = fSize; l--;) {
    VecSol[l] = 0;
  }
  //
  //-------------------- outer loop starts here
  TStopwatch timer;
  timer.Start();
  while (1) {
    //
    //-------------------- compute initial residual vector
    fMatrix->MultiplyByVec(VecSol, fPvv[0]);
    for (l = fSize; l--;) {
      fPvv[0][l] = fRHS[l] - fPvv[0][l]; //  fPvv[0]= initial residual
    }
    beta = 0;
    for (l = fSize; l--;) {
      beta += fPvv[0][l] * fPvv[0][l];
    }
    beta = TMath::Sqrt(beta);
    //
    if (beta < epsmac) {
      break; // success?
    }
    t = 1.0 / beta;
    //--------------------   normalize:  fPvv[0] = fPvv[0] / beta
    for (l = fSize; l--;) {
      fPvv[0][l] *= t;
    }
    if (its == 0) {
      eps1 = rtol * beta;
    }
    //
    //    ** initialize 1-st term  of rhs of hessenberg system
    fPVecV[0] = beta;
    int i = -1;
    do {
      i++;
      its++;
      int i1 = i + 1;
      //
      //  (Right) Preconditioning Operation   z_{j} = M^{-1} v_{j}
      //
      if (precon > 0) {
        ApplyPrecon(fPvv[i], fPvz[i]);
      } else {
        for (l = fSize; l--;) {
          fPvz[i][l] = fPvv[i][l];
        }
      }
      //
      //-------------------- matvec operation w = A z_{j} = A M^{-1} v_{j}
      fMatrix->MultiplyByVec(fPvz[i], fPvv[i1]);
      //
      // modified gram - schmidt...
      // h_{i,j} = (w,v_{i})
      // w  = w - h_{i,j} v_{i}
      //
      for (int j = 0; j <= i; j++) {
        for (t = 0, l = fSize; l--;) {
          t += fPvv[j][l] * fPvv[i1][l];
        }
        fPhh[i][j] = t;
        for (l = fSize; l--;) {
          fPvv[i1][l] -= t * fPvv[j][l];
        }
      }
      // -------------------- h_{j+1,j} = ||w||_{2}
      for (t = 0, l = fSize; l--;) {
        t += fPvv[i1][l] * fPvv[i1][l];
      }
      t = TMath::Sqrt(t);
      fPhh[i][i1] = t;
      if (t > 0)
        for (t = 1. / t, l = 0; l < fSize; l++) {
          fPvv[i1][l] *= t; //  v_{j+1} = w / h_{j+1,j}
        }
      //
      // done with modified gram schimdt and arnoldi step
      // now  update factorization of fPhh
      //
      // perform previous transformations  on i-th column of h
      //
      for (l = 1; l <= i; l++) {
        int l1 = l - 1;
        t = fPhh[i][l1];
        fPhh[i][l1] = fPVecR1[l1] * t + fPVecR2[l1] * fPhh[i][l];
        fPhh[i][l] = -fPVecR2[l1] * t + fPVecR1[l1] * fPhh[i][l];
      }
      double gam = TMath::Sqrt(fPhh[i][i] * fPhh[i][i] + fPhh[i][i1] * fPhh[i][i1]);
      //
      // if gamma is zero then any small value will do...
      // will affect only residual estimate
      if (gam < epsmac) {
        gam = epsmac;
      }
      //  get  next plane rotation
      fPVecR1[i] = fPhh[i][i] / gam;
      fPVecR2[i] = fPhh[i][i1] / gam;
      fPVecV[i1] = -fPVecR2[i] * fPVecV[i];
      fPVecV[i] *= fPVecR1[i];
      //
      //  determine residual norm and test for convergence
      fPhh[i][i] = fPVecR1[i] * fPhh[i][i] + fPVecR2[i] * fPhh[i][i1];
      beta = TMath::Abs(fPVecV[i1]);
      //
    } while ((i < nkrylov - 1) && (beta > eps1) && (its < itnlim));
    //
    // now compute solution. 1st, solve upper triangular system
    fPVecV[i] = fPVecV[i] / fPhh[i][i];
    for (int j = 1; j <= i; j++) {
      int k = i - j;
      for (t = fPVecV[k], l = k + 1; l <= i; l++) {
        t -= fPhh[l][k] * fPVecV[l];
      }
      fPVecV[k] = t / fPhh[k][k];
    }
    // --------------------  linear combination of v[i]'s to get sol.
    for (int j = 0; j <= i; j++) {
      for (t = fPVecV[j], l = 0; l < fSize; l++) {
        VecSol[l] += t * fPvz[j][l];
      }
    }
    //
    // --------------------  restart outer loop if needed
    //
    if (beta <= eps1) {
      timer.Stop();
      LOG(info) << Form("FGMRES converged in %d iterations, CPU time: %.1f sec", its, timer.CpuTime());
      break; // success
    }
    //
    if (its >= itnlim) {
      timer.Stop();
      LOG(info) << Form("%d iterations limit exceeded, CPU time: %.1f sec", itnlim, timer.CpuTime());
      status = false;
      break;
    }
  }
  //
  ClearAux();
  return status;
  //
}

//________________________________ MINRES METHODS ________________________________
bool MinResSolve::SolveMinRes(TVectorD& VecSol, int precon, int itnlim, double rtol)
{
  // solve by minres
  return SolveMinRes(VecSol.GetMatrixArray(), precon, itnlim, rtol);
}

//________________________________________________________________________________
bool MinResSolve::SolveMinRes(double* VecSol, int precon, int itnlim, double rtol)
{
  /*
    Adapted from author's Fortran code:
    Michael A. Saunders           na.msaunders@na-net.ornl.gov

    MINRES is an implementation of the algorithm described in the following reference:
    C. C. Paige and M. A. Saunders (1975),
    Solution of sparse indefinite systems of linear equations,
    SIAM J. Numer. Anal. 12(4), pp. 617-629.

  */
  if (!fMatrix->IsSymmetric()) {
    LOG(info) << "MinRes cannot solve asymmetric matrices, use FGMRes instead";
    return false;
  }
  //
  ClearAux();
  const double eps = 2.22E-16;
  double beta1;
  //
  if (precon > 0) {
    if (precon >= kPreconsTot) {
      LOG(info) << Form("Unknown preconditioner identifier %d, ignore", precon);
    } else {
      if (BuildPrecon(precon) < 0) {
        ClearAux();
        LOG(info) << "MinRes failed to build the preconditioner";
        return false;
      }
    }
  }
  LOG(info) << Form("Solution by MinRes: Preconditioner#%d Max.iter.: %d Tol.: %e", precon, itnlim, rtol);
  //
  // ------------------------ initialization  ---------------------->>>>
  memset(VecSol, 0, fSize * sizeof(double));
  int status = 0, itn = 0;
  double normA = 0;
  double condA = 0;
  double ynorm = 0;
  double rnorm = 0;
  double gam, gmax = 1, gmin = 1, gbar, oldeps, epsa, epsx, epsr, diag, delta, phi, denom, z;
  //
  if (!InitAuxMinRes()) {
    return false;
  }
  //
  memset(VecSol, 0, fSize * sizeof(double));
  //
  // ------------ init aux -------------------------<<<<
  //   Set up y and v for the first Lanczos vector v1.
  //   y  =  beta1 P' v1,  where  P = C**(-1). v is really P' v1.
  //
  for (int i = fSize; i--;) {
    fPVecY[i] = fPVecR1[i] = fRHS[i];
  }
  //
  if (precon > 0) {
    ApplyPrecon(fRHS, fPVecY);
  }
  beta1 = 0;
  for (int i = fSize; i--;)
    beta1 += fRHS[i] * fPVecY[i]; //
  //
  if (beta1 < 0) {
    LOG(info) << Form("Preconditioner is indefinite (init) (%e).", beta1);
    ClearAux();
    status = 7;
    return false;
  }
  //
  if (beta1 < eps) {
    LOG(info) << Form("RHS is zero or is the nullspace of the Preconditioner: Solution is {0}");
    ClearAux();
    return true;
  }
  //
  beta1 = TMath::Sqrt(beta1); // Normalize y to get v1 later.
  //
  //      See if Msolve is symmetric. //RS: Skept
  //      See if Aprod  is symmetric. //RS: Skept
  //
  double oldb = 0;
  double beta = beta1;
  double dbar = 0;
  double epsln = 0;
  double qrnorm = beta1;
  double phibar = beta1;
  double rhs1 = beta1;
  double rhs2 = 0;
  double tnorm2 = 0;
  double ynorm2 = 0;
  double cs = -1;
  double sn = 0;
  for (int i = fSize; i--;) {
    fPVecR2[i] = fPVecR1[i];
  }
  //
  TStopwatch timer;
  timer.Start();
  while (status == 0) { //-----------------  Main iteration loop ---------------------->>>>
    //
    itn++;
    /*-----------------------------------------------------------------
      Obtain quantities for the next Lanczos vector vk+1, k = 1, 2,...
      The general iteration is similar to the case k = 1 with v0 = 0:
      p1      = Operator * v1  -  beta1 * v0,
      alpha1  = v1'p1,
      q2      = p2  -  alpha1 * v1,
      beta2^2 = q2'q2,
      v2      = (1/beta2) q2.
      Again, y = betak P vk,  where  P = C**(-1).
      .... more description needed.
      -----------------------------------------------------------------*/
    //
    double s = 1. / beta; // Normalize previous vector (in y).
    for (int i = fSize; i--;) {
      fPVecV[i] = s * fPVecY[i]; // v = vk if P = I
    }
    //
    fMatrix->MultiplyByVec(fPVecV, fPVecY); //      APROD (VecV, VecY);
    //
    if (itn >= 2) {
      double btrat = beta / oldb;
      for (int i = fSize; i--;) {
        fPVecY[i] -= btrat * fPVecR1[i];
      }
    }
    double alfa = 0;
    for (int i = fSize; i--;) {
      alfa += fPVecV[i] * fPVecY[i]; //      alphak
    }
    //
    double alf2bt = alfa / beta;
    for (int i = fSize; i--;) {
      fPVecY[i] -= alf2bt * fPVecR2[i];
      fPVecR1[i] = fPVecR2[i];
      fPVecR2[i] = fPVecY[i];
    }
    //
    if (precon > 0) {
      ApplyPrecon(fPVecR2, fPVecY);
    }
    //
    oldb = beta; //      oldb = betak
    beta = 0;
    for (int i = fSize; i--;)
      beta += fPVecR2[i] * fPVecY[i]; // beta = betak+1^2
    //
    if (beta < 0) {
      LOG(info) << Form("Preconditioner is indefinite (%e)", beta);
      status = 7;
      break;
    }
    //
    beta = TMath::Sqrt(beta); //            beta = betak+1
    tnorm2 += alfa * alfa + oldb * oldb + beta * beta;
    //
    if (itn == 1) { //     Initialize a few things.
      if (beta / beta1 <= 10.0 * eps) {
        status = 0; //-1   //?????  beta2 = 0 or ~ 0,  terminate later.
        LOG(info) << "RHS is eigenvector";
      }
      //        !tnorm2 = alfa**2
      gmax = TMath::Abs(alfa); //              alpha1
      gmin = gmax;             //              alpha1
    }
    //
    /*
      Apply previous rotation Qk-1 to get
      [deltak epslnk+1] = [cs  sn][dbark    0   ]
      [gbar k dbar k+1]   [sn -cs][alfak betak+1].
    */
    //
    oldeps = epsln;
    delta = cs * dbar + sn * alfa; //  delta1 = 0         deltak
    gbar = sn * dbar - cs * alfa;  //  gbar 1 = alfa1     gbar k
    epsln = sn * beta;             //  epsln2 = 0         epslnk+1
    dbar = -cs * beta;             //  dbar 2 = beta2     dbar k+1
    //
    // Compute the next plane rotation Qk
    //
    gam = TMath::Sqrt(gbar * gbar + beta * beta); // gammak
    cs = gbar / gam;                              // ck
    sn = beta / gam;                              // sk
    phi = cs * phibar;                            // phik
    phibar = sn * phibar;                         // phibark+1
    //
    // Update  x.
    denom = 1. / gam;
    //
    for (int i = fSize; i--;) {
      fPVecW1[i] = fPVecW2[i];
      fPVecW2[i] = fPVecW[i];
      fPVecW[i] = denom * (fPVecV[i] - oldeps * fPVecW1[i] - delta * fPVecW2[i]);
      VecSol[i] += phi * fPVecW[i];
    }
    //
    //  Go round again.
    //
    gmax = TMath::Max(gmax, gam);
    gmin = TMath::Min(gmin, gam);
    z = rhs1 / gam;
    ynorm2 += z * z;
    rhs1 = rhs2 - delta * z;
    rhs2 = -epsln * z;
    //
    //   Estimate various norms and test for convergence.
    normA = TMath::Sqrt(tnorm2);
    ynorm = TMath::Sqrt(ynorm2);
    epsa = normA * eps;
    epsx = normA * ynorm * eps;
    epsr = normA * ynorm * rtol;
    diag = gbar;
    if (diag == 0) {
      diag = epsa;
    }
    //
    qrnorm = phibar;
    rnorm = qrnorm;
    /*
      Estimate  cond(A).
      In this version we look at the diagonals of  R  in the
      factorization of the lower Hessenberg matrix,  Q * H = R,
      where H is the tridiagonal matrix from Lanczos with one
      extra row, beta(k+1) e_k^T.
    */
    condA = gmax / gmin;
    //
    // See if any of the stopping criteria are satisfied.
    // In rare cases, istop is already -1 from above (Abar = const*I).
    //
    LOG(debug) << Form("#%5d |qnrm: %+.2e Anrm:%+.2e Cnd:%+.2e Rnrm:%+.2e Ynrm:%+.2e EpsR:%+.2e EpsX:%+.2e Beta1:%+.2e",
                       itn, qrnorm, normA, condA, rnorm, ynorm, epsr, epsx, beta1);

    if (status == 0) {
      if (itn >= itnlim) {
        status = 5;
        LOG(info) << Form("%d iterations limit exceeded", itnlim);
      }
      if (condA >= 0.1 / eps) {
        status = 4;
        LOG(info) << Form("Matrix condition nymber %e exceeds limit %e", condA, 0.1 / eps);
      }
      if (epsx >= beta1) {
        status = 3;
        LOG(info) << Form("Approximate convergence");
      }
      if (qrnorm <= epsx) {
        status = 2;
        LOG(info) << Form("Converged within machine precision");
      }
      if (qrnorm <= epsr) {
        status = 1;
        LOG(info) << Form("Converged");
      }
    }
    //
  } //-----------------  Main iteration loop ----------------------<<<
  //
  ClearAux();
  //
  timer.Stop();
  LOG(info) << Form(
    "Exit from MinRes: CPU time: %.2f sec\n"
    "Status    :  %2d\n"
    "Iterations:  %4d\n"
    "Norm      :  %+e\n"
    "Condition :  %+e\n"
    "Res.Norm  :  %+e\n"
    "Sol.Norm  :  %+e",
    timer.CpuTime(), status, itn, normA, condA, rnorm, ynorm);
  //
  return status >= 0 && status <= 3;
  //
}

//______________________________________________________________
void MinResSolve::ApplyPrecon(const TVectorD& vecRHS, TVectorD& vecOut) const
{
  // apply precond.
  ApplyPrecon(vecRHS.GetMatrixArray(), vecOut.GetMatrixArray());
}

//______________________________________________________________
void MinResSolve::ApplyPrecon(const double* vecRHS, double* vecOut) const
{
  // Application of the preconditioner matrix:
  // implicitly defines the matrix solving the M*VecOut = VecRHS
  //  const double kTiny = 1E-12;
  if (fPrecon >= kPreconBD && fPrecon < kPreconILU0) { // band diagonal decomposition
    fMatBD->Solve(vecRHS, vecOut);
    //    return;
  } else if (fPrecon >= kPreconILU0 && fPrecon <= kPreconILU10) {
    //
    for (int i = 0; i < fSize; i++) { // Block L solve
      vecOut[i] = vecRHS[i];
      VectorSparse& rowLi = *fMatL->GetRow(i);
      int n = rowLi.GetNElems();
      for (int j = 0; j < n; j++) {
        vecOut[i] -= vecOut[rowLi.GetIndex(j)] * rowLi.GetElem(j);
      }
    }
    //
    for (int i = fSize; i--;) { // Block -- U solve
      VectorSparse& rowUi = *fMatU->GetRow(i);
      int n = rowUi.GetNElems();
      for (int j = 0; j < n; j++) {
        vecOut[i] -= vecOut[rowUi.GetIndex(j)] * rowUi.GetElem(j);
      }
      vecOut[i] *= fDiagLU[i];
    }
    //
  }
  //
}

//___________________________________________________________
bool MinResSolve::InitAuxMinRes()
{
  // init auxiliary space for minres
  fPVecY = new double[fSize];
  fPVecR1 = new double[fSize];
  fPVecR2 = new double[fSize];
  fPVecV = new double[fSize];
  fPVecW = new double[fSize];
  fPVecW1 = new double[fSize];
  fPVecW2 = new double[fSize];

  for (int i = fSize; i--;) {
    fPVecY[i] = fPVecR1[i] = fPVecR2[i] = fPVecV[i] = fPVecW[i] = fPVecW1[i] = fPVecW2[i] = 0.0;
  }

  return true;
}

//___________________________________________________________
bool MinResSolve::InitAuxFGMRES(int nkrylov)
{
  // init auxiliary space for fgmres
  fPvv = new double*[nkrylov + 1];
  fPvz = new double*[nkrylov];
  for (int i = 0; i <= nkrylov; i++) {
    fPvv[i] = new double[fSize];
  }
  fPhh = new double*[nkrylov];
  for (int i = 0; i < nkrylov; i++) {
    fPhh[i] = new double[i + 2];
    fPvz[i] = new double[fSize];
  }
  //
  fPVecR1 = new double[nkrylov];
  fPVecR2 = new double[nkrylov];
  fPVecV = new double[nkrylov + 1];
  //
  return true;
}

//___________________________________________________________
void MinResSolve::ClearAux()
{
  // clear aux. space
  if (fPVecY) {
    delete[] fPVecY;
  }
  fPVecY = nullptr;
  if (fPVecR1) {
    delete[] fPVecR1;
  }
  fPVecR1 = nullptr;
  if (fPVecR2) {
    delete[] fPVecR2;
  }
  fPVecR2 = nullptr;
  if (fPVecV) {
    delete[] fPVecV;
  }
  fPVecV = nullptr;
  if (fPVecW) {
    delete[] fPVecW;
  }
  fPVecW = nullptr;
  if (fPVecW1) {
    delete[] fPVecW1;
  }
  fPVecW1 = nullptr;
  if (fPVecW2) {
    delete[] fPVecW2;
  }
  fPVecW2 = nullptr;
  if (fDiagLU) {
    delete[] fDiagLU;
  }
  fDiagLU = nullptr;
  if (fMatL) {
    delete fMatL;
  }
  fMatL = nullptr;
  if (fMatU) {
    delete fMatU;
  }
  fMatU = nullptr;
  if (fMatBD) {
    delete fMatBD;
  }
  fMatBD = nullptr;
}

//___________________________________________________________
int MinResSolve::BuildPreconBD(int hwidth)
{
  // build preconditioner
  LOG(info) << Form("Building Band-Diagonal preconditioner of half-width = %d", hwidth);
  fMatBD = new SymBDMatrix(fMatrix->GetSize(), hwidth);
  //
  // fill the band-diagonal part of the matrix
  if (fMatrix->InheritsFrom("MatrixSparse")) {
    for (int ir = fMatrix->GetSize(); ir--;) {
      int jmin = TMath::Max(0, ir - hwidth);
      VectorSparse& irow = *((MatrixSparse*)fMatrix)->GetRow(ir);
      for (int j = irow.GetNElems(); j--;) {
        int jind = irow.GetIndex(j);
        if (jind < jmin) {
          break;
        }
        (*fMatBD)(ir, jind) = irow.GetElem(j);
      }
    }
  } else {
    for (int ir = fMatrix->GetSize(); ir--;) {
      int jmin = TMath::Max(0, ir - hwidth);
      for (int jr = jmin; jr <= ir; jr++) {
        (*fMatBD)(ir, jr) = fMatrix->Query(ir, jr);
      }
    }
  }
  //
  fMatBD->DecomposeLDLT();
  //
  return 0;
}

//___________________________________________________________
int MinResSolve::BuildPreconILUK(int lofM)
{
  /*----------------------------------------------------------------------------
   * ILUK preconditioner
   * incomplete LU factorization with level of fill dropping
   * Adapted from iluk.c of ITSOL_1 package by Y.Saad: http://www-users.cs.umn.edu/~saad/software/
   *----------------------------------------------------------------------------*/
  //
  LOG(info) << Form("Building ILU%d preconditioner", lofM);
  //
  TStopwatch sw;
  sw.Start();
  fMatL = new MatrixSparse(fSize);
  fMatU = new MatrixSparse(fSize);
  fMatL->SetSymmetric(false);
  fMatU->SetSymmetric(false);
  fDiagLU = new double[fSize];
  MatrixSparse* matrix = (MatrixSparse*)fMatrix;
  // symbolic factorization to calculate level of fill index arrays
  if (PreconILUKsymb(lofM) < 0) {
    ClearAux();
    return -1;
  }
  //
  int* jw = new int[fSize];
  for (int j = fSize; j--;) {
    jw[j] = -1; // set indicator array jw to -1
  }
  //
  for (int i = 0; i < fSize; i++) { // beginning of main loop
    if ((i % int(0.1 * fSize)) == 0) {
      LOG(info) << Form("BuildPrecon: row %d of %d", i, fSize);
      sw.Stop();
      sw.Print();
      sw.Start(false);
    }
    /* setup array jw[], and initial i-th row */
    VectorSparse& rowLi = *fMatL->GetRow(i);
    VectorSparse& rowUi = *fMatU->GetRow(i);
    VectorSparse& rowM = *matrix->GetRow(i);
    //
    for (int j = rowLi.GetNElems(); j--;) { // initialize L part
      int col = rowLi.GetIndex(j);
      jw[col] = j;
      rowLi.GetElem(j) = 0.; // do we need this ?
    }
    jw[i] = i;
    fDiagLU[i] = 0; // initialize diagonal
    //
    for (int j = rowUi.GetNElems(); j--;) { // initialize U part
      int col = rowUi.GetIndex(j);
      jw[col] = j;
      rowUi.GetElem(j) = 0;
    }
    // copy row from csmat into L,U D
    for (int j = rowM.GetNElems(); j--;) { // L and D part
      if (MatrixSq::IsZero(rowM.GetElem(j))) {
        continue;
      }
      int col = rowM.GetIndex(j); // (the original matrix stores only lower triangle)
      if (col < i) {
        rowLi.GetElem(jw[col]) = rowM.GetElem(j);
      } else if (col == i) {
        fDiagLU[i] = rowM.GetElem(j);
      } else {
        rowUi.GetElem(jw[col]) = rowM.GetElem(j);
      }
    }
    if (matrix->IsSymmetric()) {
      for (int col = i + 1; col < fSize; col++) { // part of the row I on the right of diagonal is stored as
        double vl = matrix->Query(col, i);        // the lower part of the column I
        if (MatrixSq::IsZero(vl))
          continue;
        rowUi.GetElem(jw[col]) = vl;
      }
    }
    //
    // eliminate previous rows
    for (int j = 0; j < rowLi.GetNElems(); j++) {
      int jrow = rowLi.GetIndex(j);
      // get the multiplier for row to be eliminated (jrow)
      rowLi.GetElem(j) *= fDiagLU[jrow];
      //
      // combine current row and row jrow
      VectorSparse& rowUj = *fMatU->GetRow(jrow);
      for (int k = 0; k < rowUj.GetNElems(); k++) {
        int col = rowUj.GetIndex(k);
        int jpos = jw[col];
        if (jpos == -1) {
          continue;
        }
        if (col < i) {
          rowLi.GetElem(jpos) -= rowLi.GetElem(j) * rowUj.GetElem(k);
        } else if (col == i) {
          fDiagLU[i] -= rowLi.GetElem(j) * rowUj.GetElem(k);
        } else {
          rowUi.GetElem(jpos) -= rowLi.GetElem(j) * rowUj.GetElem(k);
        }
      }
    }
    // reset double-pointer to -1 ( U-part)
    for (int j = rowLi.GetNElems(); j--;) {
      jw[rowLi.GetIndex(j)] = -1;
    }
    jw[i] = -1;
    for (int j = rowUi.GetNElems(); j--;) {
      jw[rowUi.GetIndex(j)] = -1;
    }
    //
    if (MatrixSq::IsZero(fDiagLU[i])) {
      LOG(info) << Form("Fatal error in ILIk: Zero diagonal found...");
      delete[] jw;
      return -1;
    }
    fDiagLU[i] = 1.0 / fDiagLU[i];
  }
  //
  delete[] jw;
  //
  sw.Stop();
  LOG(info) << Form("ILU%d preconditioner OK, CPU time: %.1f sec", lofM, sw.CpuTime());
  LOG(info) << Form("Densities: M %f L %f U %f", matrix->GetDensity(), fMatL->GetDensity(), fMatU->GetDensity());
  //
  return 0;
}

//___________________________________________________________
int MinResSolve::BuildPreconILUKDense(int lofM)
{
  /*----------------------------------------------------------------------------
   * ILUK preconditioner
   * incomplete LU factorization with level of fill dropping
   * Adapted from iluk.c of ITSOL_1 package by Y.Saad: http://www-users.cs.umn.edu/~saad/software/
   *----------------------------------------------------------------------------*/
  //
  TStopwatch sw;
  sw.Start();
  LOG(info) << Form("Building ILU%d preconditioner for dense matrix", lofM);
  //
  fMatL = new MatrixSparse(fSize);
  fMatU = new MatrixSparse(fSize);
  fMatL->SetSymmetric(false);
  fMatU->SetSymmetric(false);
  fDiagLU = new double[fSize];
  //
  // symbolic factorization to calculate level of fill index arrays
  if (PreconILUKsymbDense(lofM) < 0) {
    ClearAux();
    return -1;
  }
  //
  int* jw = new int[fSize];
  for (int j = fSize; j--;) {
    jw[j] = -1; // set indicator array jw to -1
  }
  //
  for (int i = 0; i < fSize; i++) { // beginning of main loop
    /* setup array jw[], and initial i-th row */
    VectorSparse& rowLi = *fMatL->GetRow(i);
    VectorSparse& rowUi = *fMatU->GetRow(i);
    //
    for (int j = rowLi.GetNElems(); j--;) { // initialize L part
      int col = rowLi.GetIndex(j);
      jw[col] = j;
      rowLi.GetElem(j) = 0.; // do we need this ?
    }
    jw[i] = i;
    fDiagLU[i] = 0; // initialize diagonal
    //
    for (int j = rowUi.GetNElems(); j--;) { // initialize U part
      int col = rowUi.GetIndex(j);
      jw[col] = j;
      rowUi.GetElem(j) = 0;
    }
    // copy row from csmat into L,U D
    for (int j = fSize; j--;) { // L and D part
      double vl = fMatrix->Query(i, j);
      if (MatrixSq::IsZero(vl)) {
        continue;
      }
      if (j < i) {
        rowLi.GetElem(jw[j]) = vl;
      } else if (j == i) {
        fDiagLU[i] = vl;
      } else {
        rowUi.GetElem(jw[j]) = vl;
      }
    }
    // eliminate previous rows
    for (int j = 0; j < rowLi.GetNElems(); j++) {
      int jrow = rowLi.GetIndex(j);
      // get the multiplier for row to be eliminated (jrow)
      rowLi.GetElem(j) *= fDiagLU[jrow];
      //
      // combine current row and row jrow
      VectorSparse& rowUj = *fMatU->GetRow(jrow);
      for (int k = 0; k < rowUj.GetNElems(); k++) {
        int col = rowUj.GetIndex(k);
        int jpos = jw[col];
        if (jpos == -1) {
          continue;
        }
        if (col < i) {
          rowLi.GetElem(jpos) -= rowLi.GetElem(j) * rowUj.GetElem(k);
        } else if (col == i) {
          fDiagLU[i] -= rowLi.GetElem(j) * rowUj.GetElem(k);
        } else {
          rowUi.GetElem(jpos) -= rowLi.GetElem(j) * rowUj.GetElem(k);
        }
      }
    }
    // reset double-pointer to -1 ( U-part)
    for (int j = rowLi.GetNElems(); j--;) {
      jw[rowLi.GetIndex(j)] = -1;
    }
    jw[i] = -1;
    for (int j = rowUi.GetNElems(); j--;) {
      jw[rowUi.GetIndex(j)] = -1;
    }
    //
    if (MatrixSq::IsZero(fDiagLU[i])) {
      LOG(info) << Form("Fatal error in ILIk: Zero diagonal found...");
      delete[] jw;
      return -1;
    }
    fDiagLU[i] = 1.0 / fDiagLU[i];
  }
  //
  delete[] jw;
  //
  sw.Stop();
  LOG(info) << Form("ILU%d dense preconditioner OK, CPU time: %.1f sec", lofM, sw.CpuTime());
  return 0;
}

//___________________________________________________________
int MinResSolve::PreconILUKsymb(int lofM)
{
  /*----------------------------------------------------------------------------
   * ILUK preconditioner
   * incomplete LU factorization with level of fill dropping
   * Adapted from iluk.c: lofC of ITSOL_1 package by Y.Saad: http://www-users.cs.umn.edu/~saad/software/
   *----------------------------------------------------------------------------*/
  //
  TStopwatch sw;
  LOG(info) << "PreconILUKsymb>>";
  MatrixSparse* matrix = (MatrixSparse*)fMatrix;
  sw.Start();
  //
  UChar_t **ulvl = 0, *levls = 0;
  unsigned short int* jbuf = 0;
  int* iw = 0;
  ulvl = new UChar_t*[fSize]; // stores lev-fils for U part of ILU factorization
  levls = new UChar_t[fSize];
  jbuf = new unsigned short int[fSize];
  iw = new int[fSize];
  //
  for (int j = fSize; j--;) {
    iw[j] = -1; // initialize iw
  }
  for (int i = 0; i < fSize; i++) {
    int incl = 0;
    int incu = i;
    VectorSparse& row = *matrix->GetRow(i);
    //
    // assign lof = 0 for matrix elements
    for (int j = 0; j < row.GetNElems(); j++) {
      int col = row.GetIndex(j);
      if (MatrixSq::IsZero(row.GetElem(j))) {
        continue; // !!!! matrix is sparse but sometimes 0 appears
      }
      if (col < i) { // L-part
        jbuf[incl] = col;
        levls[incl] = 0;
        iw[col] = incl++;
      } else if (col > i) { // This works only for general matrix
        jbuf[incu] = col;
        levls[incu] = 0;
        iw[col] = incu++;
      }
    }
    if (matrix->IsSymmetric()) {
      for (int col = i + 1; col < fSize; col++) { // U-part of symmetric matrix
        if (MatrixSq::IsZero(matrix->Query(col, i)))
          continue; // Due to the symmetry  == matrix(i,col)
        jbuf[incu] = col;
        levls[incu] = 0;
        iw[col] = incu++;
      }
    }
    //
    // symbolic k,i,j Gaussian elimination
    int jpiv = -1;
    while (++jpiv < incl) {
      int k = jbuf[jpiv]; // select leftmost pivot
      int kmin = k;
      int jmin = jpiv;
      for (int j = jpiv + 1; j < incl; j++)
        if (jbuf[j] < kmin) {
          kmin = jbuf[j];
          jmin = j;
        }
      //
      // ------------------------------------  swap
      if (jmin != jpiv) {
        jbuf[jpiv] = kmin;
        jbuf[jmin] = k;
        iw[kmin] = jpiv;
        iw[k] = jmin;
        int tj = levls[jpiv];
        levls[jpiv] = levls[jmin];
        levls[jmin] = tj;
        k = kmin;
      }
      // ------------------------------------ symbolic linear combinaiton of rows
      VectorSparse& rowU = *fMatU->GetRow(k);
      for (int j = 0; j < rowU.GetNElems(); j++) {
        int col = rowU.GetIndex(j);
        int it = ulvl[k][j] + levls[jpiv] + 1;
        if (it > lofM) {
          continue;
        }
        int ip = iw[col];
        if (ip == -1) {
          if (col < i) {
            jbuf[incl] = col;
            levls[incl] = it;
            iw[col] = incl++;
          } else if (col > i) {
            jbuf[incu] = col;
            levls[incu] = it;
            iw[col] = incu++;
          }
        } else {
          levls[ip] = TMath::Min(levls[ip], it);
        }
      }
      //
    } // end - while loop
    //
    // reset iw
    for (int j = 0; j < incl; j++) {
      iw[jbuf[j]] = -1;
    }
    for (int j = i; j < incu; j++) {
      iw[jbuf[j]] = -1;
    }
    //
    // copy L-part
    VectorSparse& rowLi = *fMatL->GetRow(i);
    rowLi.ReSize(incl);
    if (incl > 0) {
      memcpy(rowLi.GetIndices(), jbuf, sizeof(unsigned short int) * incl);
    }
    // copy U-part
    int k = incu - i;
    VectorSparse& rowUi = *fMatU->GetRow(i);
    rowUi.ReSize(k);
    if (k > 0) {
      memcpy(rowUi.GetIndices(), jbuf + i, sizeof(unsigned short int) * k);
      ulvl[i] = new UChar_t[k]; // update matrix of levels
      memcpy(ulvl[i], levls + i, k * sizeof(UChar_t));
    }
  }
  //
  // free temp space and leave
  delete[] levls;
  delete[] jbuf;
  for (int i = fSize; i--;) {
    if (fMatU->GetRow(i)->GetNElems()) {
      delete[] ulvl[i];
    }
  }
  delete[] ulvl;
  delete[] iw;
  //
  fMatL->SortIndices();
  fMatU->SortIndices();
  sw.Stop();
  sw.Print();
  LOG(info) << "PreconILUKsymb<<";
  return 0;
}

//___________________________________________________________
int MinResSolve::PreconILUKsymbDense(int lofM)
{
  /*----------------------------------------------------------------------------
   * ILUK preconditioner
   * incomplete LU factorization with level of fill dropping
   * Adapted from iluk.c: lofC of ITSOL_1 package by Y.Saad: http://www-users.cs.umn.edu/~saad/software/
   *----------------------------------------------------------------------------*/
  //
  UChar_t **ulvl = 0, *levls = 0;
  unsigned short int* jbuf = 0;
  int* iw = 0;
  ulvl = new UChar_t*[fSize]; // stores lev-fils for U part of ILU factorization
  levls = new UChar_t[fSize];
  jbuf = new unsigned short int[fSize];
  iw = new int[fSize];
  //
  for (int j = fSize; j--;) {
    iw[j] = -1; // initialize iw
  }
  for (int i = 0; i < fSize; i++) {
    int incl = 0;
    int incu = i;
    //
    // assign lof = 0 for matrix elements
    for (int j = 0; j < fSize; j++) {
      if (MatrixSq::IsZero(fMatrix->Query(i, j))) {
        continue;
      }
      if (j < i) { // L-part
        jbuf[incl] = j;
        levls[incl] = 0;
        iw[j] = incl++;
      } else if (j > i) { // This works only for general matrix
        jbuf[incu] = j;
        levls[incu] = 0;
        iw[j] = incu++;
      }
    }
    //
    // symbolic k,i,j Gaussian elimination
    int jpiv = -1;
    while (++jpiv < incl) {
      int k = jbuf[jpiv]; // select leftmost pivot
      int kmin = k;
      int jmin = jpiv;
      for (int j = jpiv + 1; j < incl; j++) {
        if (jbuf[j] < kmin) {
          kmin = jbuf[j];
          jmin = j;
        }
      }
      //
      // ------------------------------------  swap
      if (jmin != jpiv) {
        jbuf[jpiv] = kmin;
        jbuf[jmin] = k;
        iw[kmin] = jpiv;
        iw[k] = jmin;
        int tj = levls[jpiv];
        levls[jpiv] = levls[jmin];
        levls[jmin] = tj;
        k = kmin;
      }
      // ------------------------------------ symbolic linear combinaiton of rows
      VectorSparse& rowU = *fMatU->GetRow(k);
      for (int j = 0; j < rowU.GetNElems(); j++) {
        int col = rowU.GetIndex(j);
        int it = ulvl[k][j] + levls[jpiv] + 1;
        if (it > lofM) {
          continue;
        }
        int ip = iw[col];
        if (ip == -1) {
          if (col < i) {
            jbuf[incl] = col;
            levls[incl] = it;
            iw[col] = incl++;
          } else if (col > i) {
            jbuf[incu] = col;
            levls[incu] = it;
            iw[col] = incu++;
          }
        } else {
          levls[ip] = TMath::Min(levls[ip], it);
        }
      }
      //
    } // end - while loop
    //
    // reset iw
    for (int j = 0; j < incl; j++) {
      iw[jbuf[j]] = -1;
    }
    for (int j = i; j < incu; j++) {
      iw[jbuf[j]] = -1;
    }
    //
    // copy L-part
    VectorSparse& rowLi = *fMatL->GetRow(i);
    rowLi.ReSize(incl);
    if (incl > 0) {
      memcpy(rowLi.GetIndices(), jbuf, sizeof(unsigned short int) * incl);
    }
    // copy U-part
    int k = incu - i;
    VectorSparse& rowUi = *fMatU->GetRow(i);
    rowUi.ReSize(k);
    if (k > 0) {
      memcpy(rowUi.GetIndices(), jbuf + i, sizeof(unsigned short int) * k);
      ulvl[i] = new UChar_t[k]; // update matrix of levels
      memcpy(ulvl[i], levls + i, k * sizeof(UChar_t));
    }
  }
  //
  // free temp space and leave
  delete[] levls;
  delete[] jbuf;
  for (int i = fSize; i--;)
    if (fMatU->GetRow(i)->GetNElems())
      delete[] ulvl[i];
  delete[] ulvl;
  delete[] iw;
  //
  fMatL->SortIndices();
  fMatU->SortIndices();
  return 0;
}
