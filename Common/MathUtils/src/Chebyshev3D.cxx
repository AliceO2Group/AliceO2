/// \file Cheb3D.cxx
/// \brief Implementation of the Cheb3D class
/// \author ruben.shahoyan@cern.ch 09/09/2006

#include "MathUtils/Chebyshev3D.h"
#include <TH1.h>              // for TH1D, TH1
#include <TMath.h>            // for Cos, Pi
#include <TMethodCall.h>      // for TMethodCall
#include <TROOT.h>            // for TROOT, gROOT
#include <TRandom.h>          // for TRandom, gRandom
#include <TString.h>          // for TString
#include <TSystem.h>          // for TSystem, gSystem
#include <cstdio>            // for printf, fprintf, FILE, fclose, fflush, etc
#include "MathUtils/Chebyshev3DCalc.h"  // for Chebyshev3DCalc, etc
#include "FairLogger.h"       // for FairLogger, MESSAGE_ORIGIN
#include "TMathBase.h"        // for Max, Abs
#include "TNamed.h"           // for TNamed
#include "TObjArray.h"        // for TObjArray

using namespace o2::mathUtils;

ClassImp(Chebyshev3D)

  const
Float_t Chebyshev3D::sMinimumPrecision = 1.e-12f;

Chebyshev3D::Chebyshev3D()
  : mOutputArrayDimension(0),
    mPrecision(sMinimumPrecision),
    mChebyshevParameter(1),
    mMaxCoefficients(0),
    mTemporaryUserResults(nullptr),
    mTemporaryChebyshevGrid(nullptr),
    mUserFunctionName(""),
    mUserMacro(nullptr),
    mLogger(FairLogger::GetLogger())
{
  // Default constructor
  for (int i = 3; i--;) {
    mMinBoundaries[i] = mMaxBoundaries[i] = mBoundaryMappingScale[i] = mBoundaryMappingOffset[i] =
    mTemporaryCoefficient[i] = 0;
    mNumberOfPoints[i] = 0;
    mTemporaryChebyshevGridOffs[i] = 0;
  }
}

Chebyshev3D::Chebyshev3D(const Chebyshev3D &src)
  : TNamed(src),
    mOutputArrayDimension(src.mOutputArrayDimension),
    mPrecision(src.mPrecision),
    mChebyshevParameter(1),
    mMaxCoefficients(src.mMaxCoefficients),
    mTemporaryUserResults(nullptr),
    mTemporaryChebyshevGrid(nullptr),
    mUserFunctionName(src.mUserFunctionName),
    mUserMacro(nullptr),
    mLogger(FairLogger::GetLogger())
{
  // read coefs from text file
  for (int i = 3; i--;) {
    mMinBoundaries[i] = src.mMinBoundaries[i];
    mMaxBoundaries[i] = src.mMaxBoundaries[i];
    mBoundaryMappingScale[i] = src.mBoundaryMappingScale[i];
    mBoundaryMappingOffset[i] = src.mBoundaryMappingOffset[i];
    mNumberOfPoints[i] = src.mNumberOfPoints[i];
    mTemporaryChebyshevGridOffs[i] = src.mTemporaryChebyshevGridOffs[i];
    mTemporaryCoefficient[i] = 0;
  }
  for (int i = 0; i < mOutputArrayDimension; i++) {
    Chebyshev3DCalc *cbc = src.getChebyshevCalc(i);
    if (cbc) {
      mChebyshevParameter.AddAtAndExpand(new Chebyshev3DCalc(*cbc), i);
    }
  }
}

Chebyshev3D::Chebyshev3D(const char *inpFile)
  : mOutputArrayDimension(0),
    mPrecision(0),
    mChebyshevParameter(1),
    mMaxCoefficients(0),
    mTemporaryUserResults(nullptr),
    mTemporaryChebyshevGrid(nullptr),
    mUserFunctionName(""),
    mUserMacro(nullptr),
    mLogger(FairLogger::GetLogger())
{
  // read coefs from text file
  for (int i = 3; i--;) {
    mMinBoundaries[i] = mMaxBoundaries[i] = mBoundaryMappingScale[i] = mBoundaryMappingOffset[i] = 0;
    mNumberOfPoints[i] = 0;
    mTemporaryChebyshevGridOffs[i] = 0;
    mTemporaryCoefficient[i] = 0;
  }
  loadData(inpFile);
}

Chebyshev3D::Chebyshev3D(FILE *stream)
  : mOutputArrayDimension(0),
    mPrecision(0),
    mChebyshevParameter(1),
    mMaxCoefficients(0),
    mTemporaryUserResults(nullptr),
    mTemporaryChebyshevGrid(nullptr),
    mUserFunctionName(""),
    mUserMacro(nullptr),
    mLogger(FairLogger::GetLogger())
{
  // read coefs from stream
  for (int i = 3; i--;) {
    mMinBoundaries[i] = mMaxBoundaries[i] = mBoundaryMappingScale[i] = mBoundaryMappingOffset[i] = 0;
    mNumberOfPoints[i] = 0;
    mTemporaryChebyshevGridOffs[i] = 0;
    mTemporaryCoefficient[i] = 0;
  }
  loadData(stream);
}

#ifdef _INC_CREATION_Chebyshev3D_
Chebyshev3D::Chebyshev3D(const char* funName, int dimOut, const Float_t* bmin, const Float_t* bmax,
       const Int_t* npoints, Float_t prec, const Float_t* precD)
  : TNamed(funName, funName),
    mOutputArrayDimension(0),
    mPrecision(TMath::Max(sMinimumPrecision, prec)),
    mChebyshevParameter(1),
    mMaxCoefficients(0),
    mTemporaryUserResults(nullptr),
    mTemporaryChebyshevGrid(nullptr),
    mUserFunctionName(""),
    mUserMacro(nullptr),
    mLogger(FairLogger::GetLogger())
{
  if (dimOut < 1) {
    Error("Chebyshev3D", "Requested output dimension is %d\nStop\n", mOutputArrayDimension);
    exit(1);
  }
  for (int i = 3; i--;) {
    mMinBoundaries[i] = mMaxBoundaries[i] = mBoundaryMappingScale[i] = mBoundaryMappingOffset[i] = 0;
    mNumberOfPoints[i] = 0;
    mTemporaryChebyshevGridOffs[i] = 0.;
    mTemporaryCoefficient[i] = 0;
  }
  setDimOut(dimOut,precD);
  prepareBoundaries(bmin, bmax);
  defineGrid(npoints);
  setuserFunction(funName);
  chebyshevFit();
}
#endif

#ifdef _INC_CREATION_Chebyshev3D_
Chebyshev3D::Chebyshev3D(void (*ptr)(float*, float*), int dimOut, const Float_t* bmin, const Float_t* bmax,
       const Int_t* npoints, Float_t prec, const Float_t* precD)
  : mOutputArrayDimension(0),
    mPrecision(TMath::Max(sMinimumPrecision, prec)),
    mChebyshevParameter(1),
    mMaxCoefficients(0),
    mTemporaryUserResults(nullptr),
    mTemporaryChebyshevGrid(nullptr),
    mUserFunctionName(""),
    mUserMacro(nullptr),
    mLogger(FairLogger::GetLogger())
{
  if (dimOut < 1) {
    Error("Chebyshev3D", "Requested output dimension is %d\nStop\n", mOutputArrayDimension);
    exit(1);
  }
  for (int i = 3; i--;) {
    mMinBoundaries[i] = mMaxBoundaries[i] = mBoundaryMappingScale[i] = mBoundaryMappingOffset[i] = 0;
    mNumberOfPoints[i] = 0;
    mTemporaryChebyshevGridOffs[i] = 0.;
    mTemporaryCoefficient[i] = 0;
  }
  setDimOut(dimOut,precD);
  prepareBoundaries(bmin, bmax);
  defineGrid(npoints);
  setuserFunction(ptr);
  chebyshevFit();
}
#endif

#ifdef _INC_CREATION_Chebyshev3D_
Chebyshev3D::Chebyshev3D(void (*ptr)(float*, float*), int dimOut, const Float_t* bmin, const Float_t* bmax,
       const Int_t* npX, const Int_t* npY, const Int_t* npZ, Float_t prec, const Float_t* precD)
  : mOutputArrayDimension(0),
    mPrecision(TMath::Max(sMinimumPrecision, prec)),
    mChebyshevParameter(1),
    mMaxCoefficients(0),
    mTemporaryUserResults(nullptr),
    mTemporaryChebyshevGrid(nullptr),
    mUserFunctionName(""),
    mUserMacro(nullptr),
    mLogger(FairLogger::GetLogger())
{
  if (dimOut < 1) {
    Error("Chebyshev3D", "Requested output dimension is %d\nStop\n", mOutputArrayDimension);
    exit(1);
  }
  for (int i = 3; i--;) {
    mMinBoundaries[i] = mMaxBoundaries[i] = mBoundaryMappingScale[i] = mBoundaryMappingOffset[i] = 0;
    mNumberOfPoints[i] = 0;
    mTemporaryChebyshevGridOffs[i] = 0.;
    mTemporaryCoefficient[i] = 0;
  }
  setDimOut(dimOut,precD);
  prepareBoundaries(bmin, bmax);
  setuserFunction(ptr);

  defineGrid(npX);
  chebyshevFit(0);
  defineGrid(npY);
  chebyshevFit(1);
  defineGrid(npZ);
  chebyshevFit(2);
}
#endif

#ifdef _INC_CREATION_Chebyshev3D_
Chebyshev3D::Chebyshev3D(void (*ptr)(float*, float*), int dimOut, const Float_t* bmin, const Float_t* bmax,
       Float_t prec, Bool_t run, const Float_t* precD)
  : mOutputArrayDimension(0),
    mPrecision(TMath::Max(sMinimumPrecision, prec)),
    mChebyshevParameter(1),
    mMaxCoefficients(0),
    mTemporaryUserResults(nullptr),
    mTemporaryChebyshevGrid(nullptr),
    mUserFunctionName(""),
    mUserMacro(nullptr),
    mLogger(FairLogger::GetLogger())
{
  if (dimOut != 3) {
    Error("Chebyshev3D", "This constructor works only for 3D fits, %dD fit was requested\n", mOutputArrayDimension);
    exit(1);
  }
  if (dimOut < 1) {
    Error("Chebyshev3D", "Requested output dimension is %d\nStop\n", mOutputArrayDimension);
    exit(1);
  }
  for (int i = 3; i--;) {
    mMinBoundaries[i] = mMaxBoundaries[i] = mBoundaryMappingScale[i] = mBoundaryMappingOffset[i] = 0;
    mNumberOfPoints[i] = 0;
    mTemporaryChebyshevGridOffs[i] = 0.;
    mTemporaryCoefficient[i] = 0;
  }
  setDimOut(dimOut, precD);
  prepareBoundaries(bmin, bmax);
  setuserFunction(ptr);

  if (run) {
    int gridNC[3][3];
    estimateNumberOfPoints(prec, gridNC);
    defineGrid(gridNC[0]);
    chebyshevFit(0);
    defineGrid(gridNC[1]);
    chebyshevFit(1);
    defineGrid(gridNC[2]);
    chebyshevFit(2);
  }
}
#endif

Chebyshev3D &Chebyshev3D::operator=(const Chebyshev3D &rhs)
{
  // assignment operator
  if (this != &rhs) {
    Clear();
    mOutputArrayDimension = rhs.mOutputArrayDimension;
    mPrecision = rhs.mPrecision;
    mMaxCoefficients = rhs.mMaxCoefficients;
    mUserFunctionName = rhs.mUserFunctionName;
    mUserMacro = nullptr;
    for (int i = 3; i--;) {
      mMinBoundaries[i] = rhs.mMinBoundaries[i];
      mMaxBoundaries[i] = rhs.mMaxBoundaries[i];
      mBoundaryMappingScale[i] = rhs.mBoundaryMappingScale[i];
      mBoundaryMappingOffset[i] = rhs.mBoundaryMappingOffset[i];
      mNumberOfPoints[i] = rhs.mNumberOfPoints[i];
    }
    for (int i = 0; i < mOutputArrayDimension; i++) {
      Chebyshev3DCalc *cbc = rhs.getChebyshevCalc(i);
      if (cbc) {
        mChebyshevParameter.AddAtAndExpand(new Chebyshev3DCalc(*cbc), i);
      }
    }
  }
  return *this;
}

void Chebyshev3D::Clear(const Option_t *)
{
  // clear all dynamic structures
  if (mTemporaryUserResults) {
    delete[] mTemporaryUserResults;
    mTemporaryUserResults = nullptr;
  }
  if (mTemporaryChebyshevGrid) {
    delete[] mTemporaryChebyshevGrid;
    mTemporaryChebyshevGrid = nullptr;
  }
  if (mUserMacro) {
    delete mUserMacro;
    mUserMacro = nullptr;
  }
  mChebyshevParameter.SetOwner(kTRUE);
  mChebyshevParameter.Delete();
}

void Chebyshev3D::Print(const Option_t *opt) const
{
  // print info
  printf("%s: Chebyshev parameterization for 3D->%dD function. Precision: %e\n", GetName(), mOutputArrayDimension,
         mPrecision);
  printf("Region of validity: [%+.5e:%+.5e] [%+.5e:%+.5e] [%+.5e:%+.5e]\n", mMinBoundaries[0], mMaxBoundaries[0],
         mMinBoundaries[1], mMaxBoundaries[1], mMinBoundaries[2], mMaxBoundaries[2]);
  TString opts = opt;
  opts.ToLower();
  if (opts.Contains("l")) {
    for (int i = 0; i < mOutputArrayDimension; i++) {
      printf("Output dimension %d:\n", i + 1);
      getChebyshevCalc(i)->Print();
    }
  }
}

void Chebyshev3D::prepareBoundaries(const Float_t *bmin, const Float_t *bmax)
{
  // Set and check boundaries defined by user, prepare coefficients for their conversion to [-1:1] interval
  for (int i = 3; i--;) {
    mMinBoundaries[i] = bmin[i];
    mMaxBoundaries[i] = bmax[i];
    mBoundaryMappingScale[i] = bmax[i] - bmin[i];
    if (mBoundaryMappingScale[i] <= 0) {
      mLogger->Fatal(MESSAGE_ORIGIN, "Boundaries for %d-th dimension are not increasing: %+.4e %+.4e\nStop\n", i,
                     mMinBoundaries[i], mMaxBoundaries[i]);
    }
    mBoundaryMappingOffset[i] = bmin[i] + mBoundaryMappingScale[i] / 2.0;
    mBoundaryMappingScale[i] = 2. / mBoundaryMappingScale[i];
  }
}

#ifdef _INC_CREATION_Chebyshev3D_

// Pointer on user function (faster altrnative to TMethodCall)
void (*gUsrFunChebyshev3D)(float*, float*);

void Chebyshev3D::evaluateUserFunction()
{
  // call user supplied function
  if (gUsrFunChebyshev3D) {
    gUsrFunChebyshev3D(mTemporaryCoefficient, mTemporaryUserResults);
  }
  else {
    mUserMacro->Execute();
  }
}

void Chebyshev3D::setuserFunction(const char* name)
{
  // load user macro with function definition and compile it
  gUsrFunChebyshev3D = nullptr;
  mUserFunctionName = name;
  gSystem->ExpandPathName(mUserFunctionName);

  if (mUserMacro) {
    delete mUserMacro;
  }

  TString tmpst = mUserFunctionName;
  tmpst += "+"; // prepare filename to compile

  if (gROOT->LoadMacro(tmpst.Data())) {
    Error("SetUsrFunction", "Failed to load user function from %s\nStop\n", name);
    exit(1);
  }

  mUserMacro = new TMethodCall();
  tmpst = tmpst.Data() + tmpst.Last('/') + 1; // Strip away any path preceding the macro file name
  int dot = tmpst.Last('.');

  if (dot > 0) {
    tmpst.Resize(dot);
  }
  mUserMacro->InitWithPrototype(tmpst.Data(), "Float_t *,Float_t *");
  long args[2];
  args[0] = (long)mTemporaryCoefficient;
  args[1] = (long)mTemporaryUserResults;
  mUserMacro->SetParamPtrs(args);
}
#endif

#ifdef _INC_CREATION_Chebyshev3D_
void Chebyshev3D::setuserFunction(void (*ptr)(float*, float*))
{
  // assign user training function
  if (mUserMacro) {
    delete mUserMacro;
  }
  mUserMacro = nullptr;
  mUserFunctionName = "";
  gUsrFunChebyshev3D = ptr;
}
#endif

#ifdef _INC_CREATION_Chebyshev3D_
void Chebyshev3D::evaluateUserFunction(const Float_t* x, Float_t* res)
{
  // evaluate user function value
  for (int i = 3; i--;) {
    mTemporaryCoefficient[i] = x[i];
  }

  if (gUsrFunChebyshev3D) {
    gUsrFunChebyshev3D(mTemporaryCoefficient, mTemporaryUserResults);
  } else {
    mUserMacro->Execute();
  }

  for (int i = mOutputArrayDimension; i--;) {
    res[i] = mTemporaryUserResults[i];
  }
}
#endif

#ifdef _INC_CREATION_Chebyshev3D_
Int_t Chebyshev3D::calculateChebyshevCoefficients(const Float_t* funval, int np, Float_t* outCoefs, Float_t prec)
{
  // Calculate Chebyshev coeffs using precomputed function values at np roots.
  // If prec>0, estimate the highest coeff number providing the needed precision
  double sm;                        // do summations in double to minimize the roundoff error
  for (int ic = 0; ic < np; ic++) { // compute coeffs
    sm = 0;
    for (int ir = 0; ir < np; ir++) {
      float rt = TMath::Cos(ic * (ir + 0.5) * TMath::Pi() / np);
      sm += funval[ir] * rt;
    }
    outCoefs[ic] = Float_t(sm * ((ic == 0) ? 1. / np : 2. / np));
  }

  if (prec <= 0) {
    return np;
  }

  sm = 0;
  int cfMax = 0;
  for (cfMax = np; cfMax--;) {
    sm += TMath::Abs(outCoefs[cfMax]);
    if (sm >= prec) {
      break;
    }
  }
  if (++cfMax == 0) {
    cfMax = 1;
  }
  return cfMax;
}
#endif

#ifdef _INC_CREATION_Chebyshev3D_
void Chebyshev3D::defineGrid(const Int_t* npoints)
{
  // prepare the grid of Chebyshev roots in each dimension
  const int kMinPoints = 1;
  int ntot = 0;
  mMaxCoefficients = 1;
  for (int id = 3; id--;) {
    mNumberOfPoints[id] = npoints[id];
    if (mNumberOfPoints[id] < kMinPoints) {
      Error("DefineGrid", "at %d-th dimension %d point is requested, at least %d is needed\nStop\n", id,
            mNumberOfPoints[id], kMinPoints);
      exit(1);
    }
    ntot += mNumberOfPoints[id];
    mMaxCoefficients *= mNumberOfPoints[id];
  }
  printf("Computing Chebyshev nodes on [%2d/%2d/%2d] grid\n", npoints[0], npoints[1], npoints[2]);
  if (mTemporaryChebyshevGrid) {
    delete[] mTemporaryChebyshevGrid;
  }
  mTemporaryChebyshevGrid = new Float_t[ntot];

  int curp = 0;
  for (int id = 3; id--;) {
    int np = mNumberOfPoints[id];
    mTemporaryChebyshevGridOffs[id] = curp;
    for (int ip = 0; ip < np; ip++) {
      Float_t x = TMath::Cos(TMath::Pi() * (ip + 0.5) / np);
      mTemporaryChebyshevGrid[curp++] = mapToExternal(x, id);
    }
  }
}
#endif

#ifdef _INC_CREATION_Chebyshev3D_
Int_t Chebyshev3D::chebyshevFit()
{
  // prepare parameterization for all output dimensions
  int ir = 0;
  for (int i = mOutputArrayDimension; i--;) {
    ir += chebyshevFit(i);
  }
  return ir;
}
#endif

#ifdef _INC_CREATION_Chebyshev3D_
Int_t Chebyshev3D::chebyshevFit(int dmOut)
{
  // prepare paramaterization of 3D function for dmOut-th dimension
  int maxDim = 0;
  for (int i = 0; i < 3; i++) {
    if (maxDim < mNumberOfPoints[i]) {
      maxDim = mNumberOfPoints[i];
    }
  }
  auto* fvals = new Float_t[mNumberOfPoints[0]];
  auto* tmpCoef3D = new Float_t[mNumberOfPoints[0] * mNumberOfPoints[1] * mNumberOfPoints[2]];
  auto* tmpCoef2D = new Float_t[mNumberOfPoints[0] * mNumberOfPoints[1]];
  auto* tmpCoef1D = new Float_t[maxDim];

  // 1D Cheb.fit for 0-th dimension at current steps of remaining dimensions
  int ncmax = 0;

  printf("Dim%d : 00.00%% Done", dmOut);
  fflush(stdout);
  Chebyshev3DCalc* cheb = getChebyshevCalc(dmOut);

  Float_t prec = cheb->getPrecision(); 
  if (prec<sMinimumPrecision) prec = mPrecision;         // no specific precision for this dim.
  Float_t rTiny = 0.1 * prec / Float_t(maxDim); // neglect coefficient below this threshold

  float ncals2count = mNumberOfPoints[2] * mNumberOfPoints[1] * mNumberOfPoints[0];
  float ncals = 0;
  float frac = 0;
  float fracStep = 0.001;

  for (int id2 = mNumberOfPoints[2]; id2--;) {
    mTemporaryCoefficient[2] = mTemporaryChebyshevGrid[mTemporaryChebyshevGridOffs[2] + id2];

    for (int id1 = mNumberOfPoints[1]; id1--;) {
      mTemporaryCoefficient[1] = mTemporaryChebyshevGrid[mTemporaryChebyshevGridOffs[1] + id1];

      for (int id0 = mNumberOfPoints[0]; id0--;) {
        mTemporaryCoefficient[0] = mTemporaryChebyshevGrid[mTemporaryChebyshevGridOffs[0] + id0];
        evaluateUserFunction(); // compute function values at Chebyshev roots of 0-th dimension
        fvals[id0] = mTemporaryUserResults[dmOut];
        float fr = (++ncals) / ncals2count;
        if (fr - frac >= fracStep) {
          frac = fr;
          printf("\b\b\b\b\b\b\b\b\b\b\b");
          printf("%05.2f%% Done", fr * 100);
          fflush(stdout);
        }
      }
      int nc = calculateChebyshevCoefficients(fvals, mNumberOfPoints[0], tmpCoef1D, prec);
      for (int id0 = mNumberOfPoints[0]; id0--;) {
        tmpCoef2D[id1 + id0 * mNumberOfPoints[1]] = tmpCoef1D[id0];
      }
      if (ncmax < nc) {
        ncmax = nc; // max coefs to be kept in dim0 to guarantee needed precision
      }
    }
    // once each 1d slice of given 2d slice is parametrized, parametrize the Cheb.coeffs
    for (int id0 = mNumberOfPoints[0]; id0--;) {
      calculateChebyshevCoefficients(tmpCoef2D + id0 * mNumberOfPoints[1], mNumberOfPoints[1], tmpCoef1D, -1);
      for (int id1 = mNumberOfPoints[1]; id1--;) {
        tmpCoef3D[id2 + mNumberOfPoints[2] * (id1 + id0 * mNumberOfPoints[1])] = tmpCoef1D[id1];
      }
    }
  }
  // now fit the last dimensions Cheb.coefs
  for (int id0 = mNumberOfPoints[0]; id0--;) {
    for (int id1 = mNumberOfPoints[1]; id1--;) {
      calculateChebyshevCoefficients(tmpCoef3D + mNumberOfPoints[2] * (id1 + id0 * mNumberOfPoints[1]),
                                     mNumberOfPoints[2], tmpCoef1D, -1);
      for (int id2 = mNumberOfPoints[2]; id2--;) {
        tmpCoef3D[id2 + mNumberOfPoints[2] * (id1 + id0 * mNumberOfPoints[1])] = tmpCoef1D[id2]; // store on place
      }
    }
  }

  // now find 2D surface which separates significant coefficients of 3D matrix from nonsignificant ones (up to
  // prec)
  auto* tmpCoefSurf = new UShort_t[mNumberOfPoints[0] * mNumberOfPoints[1]];
  for (int id0 = mNumberOfPoints[0]; id0--;) {
    for (int id1 = mNumberOfPoints[1]; id1--;) {
      tmpCoefSurf[id1 + id0 * mNumberOfPoints[1]] = 0;
    }
  }
  Double_t resid = 0;
  for (int id0 = mNumberOfPoints[0]; id0--;) {
    for (int id1 = mNumberOfPoints[1]; id1--;) {
      for (int id2 = mNumberOfPoints[2]; id2--;) {
        int id = id2 + mNumberOfPoints[2] * (id1 + id0 * mNumberOfPoints[1]);
        Float_t cfa = TMath::Abs(tmpCoef3D[id]);
        if (cfa < rTiny) {
          tmpCoef3D[id] = 0;
          continue;
        } // neglect coefs below the threshold
        resid += cfa;
        if (resid < prec) {
          continue; // this coeff is negligible
        }
        // otherwise go back 1 step
        resid -= cfa;
        tmpCoefSurf[id1 + id0 * mNumberOfPoints[1]] = id2 + 1; // how many coefs to keep
        break;
      }
    }
  }

  // printf("\n\nCoeffs\n");
  // int cnt = 0;
  // for (int id0=0;id0<mNumberOfPoints[0];id0++) {
  //  for (int id1=0;id1<mNumberOfPoints[1];id1++) {
  //    for (int id2=0;id2<mNumberOfPoints[2];id2++) {
  // printf("%2d%2d%2d %+.4e |",id0,id1,id2,tmpCoef3D[cnt++]);
  //    }
  //    printf("\n");
  //  }
  //  printf("\n");
  //}

  // see if there are rows to reject, find max.significant column at each row
  int nRows = mNumberOfPoints[0];
  auto* tmpCols = new UShort_t[nRows];
  for (int id0 = mNumberOfPoints[0]; id0--;) {
    int id1 = mNumberOfPoints[1];
    while (id1 > 0 && tmpCoefSurf[(id1 - 1) + id0 * mNumberOfPoints[1]] == 0) {
      id1--;
    }
    tmpCols[id0] = id1;
  }
  // find max significant row
  for (int id0 = nRows; id0--;) {
    if (tmpCols[id0] > 0) {
      break;
    }
    nRows--;
  }
  // find max significant column and fill the permanent storage for the max sigificant column of each row
  cheb->initializeRows(nRows); // create needed arrays;
  UShort_t* nColsAtRow = cheb->getNumberOfColumnsAtRow();
  UShort_t* colAtRowBg = cheb->getColAtRowBg();
  int nCols = 0;
  int nElemBound2D = 0;
  for (int id0 = 0; id0 < nRows; id0++) {
    nColsAtRow[id0] = tmpCols[id0]; // number of columns to store for this row
    colAtRowBg[id0] = nElemBound2D; // begining of this row in 2D boundary surface
    nElemBound2D += tmpCols[id0];
    if (nCols < nColsAtRow[id0]) {
      nCols = nColsAtRow[id0];
    }
  }
  cheb->initializeColumns(nCols);
  delete[] tmpCols;

  // create the 2D matrix defining the boundary of significance for 3D coeffs.matrix
  // and count the number of siginifacnt coefficients
  cheb->initializeElementBound2D(nElemBound2D);
  UShort_t* coefBound2D0 = cheb->getCoefficientBound2D0();
  UShort_t* coefBound2D1 = cheb->getCoefficientBound2D1();
  mMaxCoefficients = 0; // redefine number of coeffs
  for (int id0 = 0; id0 < nRows; id0++) {
    int nCLoc = nColsAtRow[id0];
    int col0 = colAtRowBg[id0];
    for (int id1 = 0; id1 < nCLoc; id1++) {
      coefBound2D0[col0 + id1] =
        tmpCoefSurf[id1 + id0 * mNumberOfPoints[1]]; // number of coefs to store for 3-d dimension
      coefBound2D1[col0 + id1] = mMaxCoefficients;
      mMaxCoefficients += coefBound2D0[col0 + id1];
    }
  }

  // create final compressed 3D matrix for significant coeffs
  cheb->initializeCoefficients(mMaxCoefficients);
  Float_t* coefs = cheb->getCoefficients();
  int count = 0;
  for (int id0 = 0; id0 < nRows; id0++) {
    int ncLoc = nColsAtRow[id0];
    int col0 = colAtRowBg[id0];
    for (int id1 = 0; id1 < ncLoc; id1++) {
      int ncf2 = coefBound2D0[col0 + id1];
      for (int id2 = 0; id2 < ncf2; id2++) {
        coefs[count++] = tmpCoef3D[id2 + mNumberOfPoints[2] * (id1 + id0 * mNumberOfPoints[1])];
      }
    }
  }

  // printf("\n\nNewSurf\n");
  // for (int id0=0;id0<mNumberOfPoints[0];id0++) {
  //  for (int id1=0;id1<mNumberOfPoints[1];id1++) {
  //    printf("(%2d %2d) %2d |",id0,id1,tmpCoefSurf[id1+id0*mNumberOfPoints[1]]);
  //  }
  //  printf("\n");
  //}

  delete[] tmpCoefSurf;
  delete[] tmpCoef1D;
  delete[] tmpCoef2D;
  delete[] tmpCoef3D;
  delete[] fvals;

  printf("\b\b\b\b\b\b\b\b\b\b\b\b");
  printf("100.00%% Done\n");
  return 1;
}
#endif

#ifdef _INC_CREATION_Chebyshev3D_
void Chebyshev3D::saveData(const char* outfile, Bool_t append) const
{
  // writes coefficients data to output text file, optionallt appending on the end of existing file
  TString strf = outfile;
  gSystem->ExpandPathName(strf);
  FILE* stream = fopen(strf, append ? "a" : "w");
  saveData(stream);
  fclose(stream);
}
#endif

#ifdef _INC_CREATION_Chebyshev3D_
void Chebyshev3D::saveData(FILE* stream) const
{
  // writes coefficients data to existing output stream
  fprintf(stream, "\n# These are automatically generated data for the Chebyshev interpolation of 3D->%dD function\n",
          mOutputArrayDimension);
  fprintf(stream, "#\nSTART %s\n", GetName());
  fprintf(stream, "# Dimensionality of the output\n%d\n", mOutputArrayDimension);
  fprintf(stream, "# Interpolation abs. precision\n%+.8e\n", mPrecision);

  fprintf(stream, "# Lower boundaries of interpolation region\n");
  for (int i = 0; i < 3; i++) {
    fprintf(stream, "%+.8e\n", mMinBoundaries[i]);
  }
  fprintf(stream, "# Upper boundaries of interpolation region\n");
  for (int i = 0; i < 3; i++) {
    fprintf(stream, "%+.8e\n", mMaxBoundaries[i]);
  }
  fprintf(stream, "# Parameterization for each output dimension follows:\n");

  for (int i = 0; i < mOutputArrayDimension; i++) {
    getChebyshevCalc(i)->saveData(stream);
  }
  fprintf(stream, "#\nEND %s\n#\n", GetName());
}
#endif

#ifdef _INC_CREATION_Chebyshev3D_
void Chebyshev3D::invertSign()
{
  // invert the sign of all parameterizations
  for (int i = mOutputArrayDimension; i--;) {
    Chebyshev3DCalc* par = getChebyshevCalc(i);
    int ncf = par->getNumberOfCoefficients();
    float* coefs = par->getCoefficients();
    for (int j = ncf; j--;) {
      coefs[j] = -coefs[j];
    }
  }
}
#endif

void Chebyshev3D::loadData(const char *inpFile)
{
  // load coefficients data from txt file
  TString strf = inpFile;
  gSystem->ExpandPathName(strf);
  FILE *stream = fopen(strf.Data(), "r");
  loadData(stream);
  fclose(stream);
}

void Chebyshev3D::loadData(FILE *stream)
{
  // load coefficients data from stream
  if (!stream) {
    mLogger->Fatal(MESSAGE_ORIGIN, "No stream provided.\nStop");
  }
  TString buffs;
  Clear();
  Chebyshev3DCalc::readLine(buffs, stream);
  if (!buffs.BeginsWith("START")) {
    mLogger->Fatal(MESSAGE_ORIGIN, "Expected: \"START <fit_name>\", found \"%s\"\nStop\n", buffs.Data());
  }
  SetName(buffs.Data() + buffs.First(' ') + 1);

  Chebyshev3DCalc::readLine(buffs, stream); // N output dimensions
  mOutputArrayDimension = buffs.Atoi();
  if (mOutputArrayDimension < 1) {
    mLogger->Fatal(MESSAGE_ORIGIN, "Expected: '<number_of_output_dimensions>', found \"%s\"\nStop\n", buffs.Data());
  }

  setDimOut(mOutputArrayDimension);

  Chebyshev3DCalc::readLine(buffs, stream); // Interpolation abs. precision
  mPrecision = buffs.Atof();
  if (mPrecision <= 0) {
    mLogger->Fatal(MESSAGE_ORIGIN, "Expected: '<abs.precision>', found \"%s\"\nStop\n", buffs.Data());
  }

  for (int i = 0; i < 3; i++) { // Lower boundaries of interpolation region
    Chebyshev3DCalc::readLine(buffs, stream);
    mMinBoundaries[i] = buffs.Atof();
  }

  for (int i = 0; i < 3; i++) { // Upper boundaries of interpolation region
    Chebyshev3DCalc::readLine(buffs, stream);
    mMaxBoundaries[i] = buffs.Atof();
  }
  prepareBoundaries(mMinBoundaries, mMaxBoundaries);

  // data for each output dimension
  for (int i = 0; i < mOutputArrayDimension; i++) {
    getChebyshevCalc(i)->loadData(stream);
  }

  // check end_of_data record
  Chebyshev3DCalc::readLine(buffs, stream);
  if (!buffs.BeginsWith("END") || !buffs.Contains(GetName())) {
    mLogger->Fatal(MESSAGE_ORIGIN, "Expected \"END %s\", found \"%s\".\nStop\n", GetName(), buffs.Data());
  }
}

void Chebyshev3D::setDimOut(const int d, const float *prec)
{
  // init output dimensions
  mOutputArrayDimension = d;
  if (mTemporaryUserResults) {
    delete mTemporaryUserResults;
  }
  mTemporaryUserResults = new Float_t[mOutputArrayDimension];
  mChebyshevParameter.Delete();
  for (int i = 0; i < d; i++) {
    auto *clc = new Chebyshev3DCalc();
    clc->setPrecision(prec && prec[i] > sMinimumPrecision ? prec[i] : mPrecision);
    mChebyshevParameter.AddAtAndExpand(new Chebyshev3DCalc(), i);
  }
}

void Chebyshev3D::shiftBound(int id, float dif)
{
  // modify the bounds of the grid
  if (id < 0 || id > 2) {
    printf("Maximum 3 dimensions are supported\n");
    return;
  }
  mMinBoundaries[id] += dif;
  mMaxBoundaries[id] += dif;
  mBoundaryMappingOffset[id] += dif;
}

#ifdef _INC_CREATION_Chebyshev3D_
TH1* Chebyshev3D::TestRMS(int idim, int npoints, TH1* histo)
{
  // fills the difference between the original function and parameterization (for idim-th component of the output)
  // to supplied histogram. Calculations are done in npoints random points.
  // If the hostgram was not supplied, it will be created. It is up to the user to delete it!
  if (!mUserMacro) {
    printf("No user function is set\n");
    return nullptr;
  }
  if (!histo) {
    histo = new TH1D(GetName(), "Control: Function - Parametrization", 100, -2 * mPrecision, 2 * mPrecision);
  }

  float prc = getChebyshevCalc(idim)->getPrecision();
  if (prc<sMinimumPrecision) prc = mPrecision;   // no dimension specific precision
 
  for (int ip = npoints; ip--;) {
    gRandom->RndmArray(3, (Float_t*)mTemporaryCoefficient);
    for (int i = 3; i--;) {
      mTemporaryCoefficient[i] = mMinBoundaries[i] + mTemporaryCoefficient[i] * (mMaxBoundaries[i] - mMinBoundaries[i]);
    }
    evaluateUserFunction();
    Float_t valFun = mTemporaryUserResults[idim];
    Eval(mTemporaryCoefficient, mTemporaryUserResults);
    Float_t valPar = mTemporaryUserResults[idim];
    histo->Fill(valFun - valPar);
  }
  return histo;
}
#endif

#ifdef _INC_CREATION_Chebyshev3D_

void Chebyshev3D::estimateNumberOfPoints(float prec, int gridBC[3][3], Int_t npd1, Int_t npd2, Int_t npd3)
{
  // Estimate number of points to generate a training data
  const int kScp = 9;
  const float kScl[9] = { 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9 };

  const float sclDim[2] = { 0.001, 0.999 };
  const int compDim[3][2] = { { 1, 2 }, { 2, 0 }, { 0, 1 } };
  static float xyz[3];
  Int_t npdTst[3] = { npd1, npd2, npd3 };

  for (int i = 3; i--;) {
    for (int j = 3; j--;) {
      gridBC[i][j] = -1;
    }
  }

  for (int idim = 0; idim < 3; idim++) {
    float dimMN = mMinBoundaries[idim] + sclDim[0] * (mMaxBoundaries[idim] - mMinBoundaries[idim]);
    float dimMX = mMinBoundaries[idim] + sclDim[1] * (mMaxBoundaries[idim] - mMinBoundaries[idim]);

    int id1 = compDim[idim][0]; // 1st fixed dim
    int id2 = compDim[idim][1]; // 2nd fixed dim
    for (int i1 = 0; i1 < kScp; i1++) {
      xyz[id1] = mMinBoundaries[id1] + kScl[i1] * (mMaxBoundaries[id1] - mMinBoundaries[id1]);
      for (int i2 = 0; i2 < kScp; i2++) {
        xyz[id2] = mMinBoundaries[id2] + kScl[i2] * (mMaxBoundaries[id2] - mMinBoundaries[id2]);
        int* npt = getNcNeeded(xyz, idim, dimMN, dimMX, prec, npdTst[idim]); // npoints for Bx,By,Bz
        for (int ib = 0; ib < 3; ib++) {
          if (npt[ib] > gridBC[ib][idim]) {
            gridBC[ib][idim] = npt[ib];
          }
        }
      }
    }
  }
}

// void Chebyshev3D::estimateNumberOfPoints(float Prec, int gridBC[3][3])
// {
//   // Estimate number of points to generate a training data
//
//   const float sclA[9] = {0.1, 0.5, 0.9, 0.1, 0.5, 0.9, 0.1, 0.5, 0.9} ;
//   const float sclB[9] = {0.1, 0.1, 0.1, 0.5, 0.5, 0.5, 0.9, 0.9, 0.9} ;
//   const float sclDim[2] = {0.01,0.99};
//   const int   compDim[3][2] = { {1,2}, {2,0}, {0,1} };
//   static float xyz[3];
//
//   for (int i=3;i--;)for (int j=3;j--;) gridBC[i][j] = -1;
//
//   for (int idim=0;idim<3;idim++) {
//     float dimMN = mMinBoundaries[idim] + sclDim[0]*(mMaxBoundaries[idim]-mMinBoundaries[idim]);
//     float dimMX = mMinBoundaries[idim] + sclDim[1]*(mMaxBoundaries[idim]-mMinBoundaries[idim]);
//
//     for (int it=0;it<9;it++) { // test in 9 points
//       int id1 = compDim[idim][0]; // 1st fixed dim
//       int id2 = compDim[idim][1]; // 2nd fixed dim
//       xyz[ id1 ] = mMinBoundaries[id1] + sclA[it]*( mMaxBoundaries[id1]-mMinBoundaries[id1] );
//       xyz[ id2 ] = mMinBoundaries[id2] + sclB[it]*( mMaxBoundaries[id2]-mMinBoundaries[id2] );
//
//       int* npt = getNcNeeded(xyz,idim, dimMN,dimMX, Prec); // npoints for Bx,By,Bz
//       for (int ib=0;ib<3;ib++) if (npt[ib]>gridBC[ib][idim]) gridBC[ib][idim] = npt[ib];//+2;
//
//     }
//   }
// }
//
//
// int* Chebyshev3D::getNcNeeded(float xyz[3],int DimVar, float mn,float mx, float prec)
// {
//   // estimate needed number of chebyshev coefs for given function description in DimVar dimension
//   // The values for two other dimensions must be set beforehand
//
//   static int curNC[3];
//   static int retNC[3];
//   const int kMaxPoint = 400;
//   float* gridVal = new float[3*kMaxPoint];
//   float* coefs   = new float[3*kMaxPoint];
//
//   float scale = mx-mn;
//   float offs  = mn + scale/2.0;
//   scale = 2./scale;
//
//   int curNP;
//   int maxNC=-1;
//   int maxNCPrev=-1;
//   for (int i=0;i<3;i++) retNC[i] = -1;
//   for (int i=0;i<3;i++) mTemporaryCoefficient[i] = xyz[i];
//
//   for (curNP=3; curNP<kMaxPoint; curNP+=3) {
//     maxNCPrev = maxNC;
//
//     for (int i=0;i<curNP;i++) { // get function values on Cheb. nodes
//       float x = TMath::Cos( TMath::Pi()*(i+0.5)/curNP );
//       mTemporaryCoefficient[DimVar] =  x/scale+offs; // map to requested interval
//       evaluateUserFunction();
//       for (int ib=3;ib--;) gridVal[ib*kMaxPoint + i] = mTemporaryUserResults[ib];
//     }
//
//     for (int ib=0;ib<3;ib++) {
//       curNC[ib] = Chebyshev3D::calculateChebyshevCoefficients(&gridVal[ib*kMaxPoint], curNP,
// &coefs[ib*kMaxPoint],prec);
//       if (maxNC < curNC[ib]) maxNC = curNC[ib];
//       if (retNC[ib] < curNC[ib]) retNC[ib] = curNC[ib];
//     }
//     if ( (curNP-maxNC)>3 &&  (maxNC-maxNCPrev)<1 ) break;
//     maxNCPrev = maxNC;
//
//   }
//   delete[] gridVal;
//   delete[] coefs;
//   return retNC;
// }

int* Chebyshev3D::getNcNeeded(float xyz[3], int DimVar, float mn, float mx, float prec, Int_t npCheck)
{
  // estimate needed number of chebyshev coefs for given function description in DimVar dimension
  // The values for two other dimensions must be set beforehand
  static int retNC[3];
  static int npChLast = 0;
  static float* gridVal = nullptr, *coefs = nullptr;
  if (npCheck < 3) {
    npCheck = 3;
  }
  if (npChLast < npCheck) {
    if (gridVal) {
      delete[] gridVal;
    }
    if (coefs) {
      delete[] coefs;
    }
    gridVal = new float[3 * npCheck];
    coefs = new float[3 * npCheck];
    npChLast = npCheck;
  }
  float scale = mx - mn;
  float offs = mn + scale / 2.0;
  scale = 2. / scale;

  for (int i = 0; i < 3; i++) {
    mTemporaryCoefficient[i] = xyz[i];
  }
  for (int i = 0; i < npCheck; i++) {
    mTemporaryCoefficient[DimVar] =
      TMath::Cos(TMath::Pi() * (i + 0.5) / npCheck) / scale + offs; // map to requested interval
    evaluateUserFunction();
    for (int ib = 3; ib--;) {
      gridVal[ib * npCheck + i] = mTemporaryUserResults[ib];
    }
  }
  for (int ib = 0; ib < 3; ib++) {
    retNC[ib] =
      Chebyshev3D::calculateChebyshevCoefficients(&gridVal[ib * npCheck], npCheck, &coefs[ib * npCheck], prec);
  }
  return retNC;
}

#endif
