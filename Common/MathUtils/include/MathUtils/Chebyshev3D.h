// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

/// \file Cheb3D.h
/// \brief Definition of the Cheb3D class
/// \author ruben.shahoyan@cern.ch 09/09/2006

#ifndef ALICEO2_MATHUTILS_CHEBYSHEV3D_H_
#define ALICEO2_MATHUTILS_CHEBYSHEV3D_H_

#include <TNamed.h>                    // for TNamed
#include <TObjArray.h>                 // for TObjArray
#include <cstdio>                      // for FILE, stdout
#include "MathUtils/Chebyshev3DCalc.h" // for Chebyshev3DCalc, etc
#include "Rtypes.h"                    // for Float_t, Int_t, Double_t, Bool_t, etc
#include "TString.h"                   // for TString

class TH1;         // lines 15-15
class TMethodCall; // lines 16-16

namespace o2
{
namespace math_utils
{

/// Chebyshev3D produces the interpolation of the user 3D->NDimOut arbitrary function supplied in
/// "void (*fcn)(float* inp,float* out)" format either in a separate macro file or as a function pointer. Only
/// coefficients needed to guarantee the requested precision are kept. The user-callable methods are:
/// To create the interpolation use:
/// Cheb3D(const char* funName, // name of the file with user function or
/// Cheb3D(void (*ptr)(float*,float*) // pointer on the  user function
/// \param Int_t DimOut dimensionality of the function's output
/// \param Float_t *bmin lower 3D bounds of interpolation domain
/// \param Float_t *bmax upper 3D bounds of interpolation domain
/// \param Int_t *npoints number of points in each of 3 input dimension, defining the interpolation grid
/// \param Float_t prec=1E-6); requested max.absolute difference between the interpolation and any point on grid
/// To test obtained parameterization use the method TH1* TestRMS(int idim,int npoints = 1000,TH1* histo=0);
/// it will compare the user output of the user function and interpolation for idim-th output dimension and
/// fill the difference in the supplied histogram. If no histogram is supplied, it will be created.
/// To save the interpolation data: saveData(const char* filename, Bool_t append ) write text file with data.
/// If append is kTRUE and the output file already exists, data will be added in the end of the file.
/// Alternatively, saveData(FILE* stream) will write the data to already existing stream.
/// To read back already stored interpolation use either the constructor Chebyshev3D(const char* inpFile);
/// or the default constructor Chebyshev3D() followed by Chebyshev3D::loadData(const char* inpFile);
/// To compute the interpolation use Eval(float* par,float *res) method, with par being 3D vector of arguments
/// (inside the validity region) and res is the array of DimOut elements for the output.
/// If only one component (say, idim-th) of the output is needed, use faster Float_t Eval(Float_t *par,int idim) method
/// void Print(option="") will print the name, the ranges of validity and the absolute precision of the
/// parameterization. Option "l" will also print the information about the number of coefficients for each output
/// dimension.
/// NOTE: during the evaluation no check is done for parameter vector being outside the interpolation region.
/// If there is such a risk, use Bool_t isInside(float *par) method. Chebyshev parameterization is not
/// good for extrapolation!
/// For the properties of Chebyshev parameterization see:
/// H.Wind, CERN EP Internal Report, 81-12/Rev.
class Chebyshev3D : public TNamed
{
 public:
  Chebyshev3D();

  Chebyshev3D(const Chebyshev3D& src);

  Chebyshev3D(const char* inpFile);

  Chebyshev3D(FILE* stream);

#ifdef _INC_CREATION_Chebyshev3D_
  /// Construct the parameterization for the function
  /// \param funName : name of the file containing the function: void funName(Float_t * inp,Float_t * out)
  /// \param DimOut  : dimension of the vector computed by the user function
  /// \param bmin    : array of 3 elements with the lower boundaries of the region where the function is defined
  /// \param bmax    : array of 3 elements with the upper boundaries of the region where the function is defined
  /// \param npoints : array of 3 elements with the number of points to compute in each of 3 dimension
  /// \param prec    : max allowed absolute difference between the user function and computed parameterization on the
  /// requested grid, common for all 1D components
  /// \param precD   : optional precison per component
  Chebyshev3D(const char* funName, Int_t dimOut, const Float_t* bmin, const Float_t* bmax, const Int_t* npoints,
              Float_t prec = 1E-6, const Float_t* precD = nullptr);
  /// Construct the parameterization for the function
  /// \param ptr     : pointer on the function: void fun(Float_t * inp,Float_t * out)
  /// \param DimOut  : dimension of the vector computed by the user function
  /// \param bmin    : array of 3 elements with the lower boundaries of the region where the function is defined
  /// \param bmax    : array of 3 elements with the upper boundaries of the region where the function is defined
  /// \param npoints : array of 3 elements with the number of points to compute in each of 3 dimension
  /// \param prec    : max allowed absolute difference between the user function and computed parameterization on the
  /// requested grid, common for all 1D components
  /// \param precD   : optional precison per component
  Chebyshev3D(void (*ptr)(float*, float*), Int_t dimOut, const Float_t* bmin, const Float_t* bmax, const Int_t* npoints,
              Float_t prec = 1E-6, const Float_t* precD = nullptr);
  /// Construct very economic  parameterization for the function
  /// \param ptr     : pointer on the function: void fun(Float_t * inp,Float_t * out)
  /// \param DimOut  : dimension of the vector computed by the user function
  /// \param bmin    : array of 3 elements with the lower boundaries of the region where the function is defined
  /// \param bmax    : array of 3 elements with the upper boundaries of the region where the function is defined
  /// \param npX     : array of 3 elements with the number of points to compute in each dimension for 1st component
  /// \param npY     : array of 3 elements with the number of points to compute in each dimension for 2nd component
  /// \param npZ     : array of 3 elements with the number of points to compute in each dimension for 3d  component
  /// \param prec    : max allowed absolute difference between the user function and computed parameterization on the
  /// requested grid, common for all 1D components
  /// \param precD   : optional precison per component
  Chebyshev3D(void (*ptr)(float*, float*), int dimOut, const Float_t* bmin, const Float_t* bmax,
              const Int_t* npX, const Int_t* npY, const Int_t* npZ,
              Float_t prec = 1E-6, const Float_t* precD = nullptr);
  /// Construct very economic  parameterization for the function with automatic calculation of the root's grid
  /// \param ptr     : pointer on the function: void fun(Float_t * inp,Float_t * out)
  /// \param DimOut  : dimension of the vector computed by the user function
  /// \param bmin    : array of 3 elements with the lower boundaries of the region where the function is defined
  /// \param bmax    : array of 3 elements with the upper boundaries of the region where the function is defined
  /// \param prec    : max allowed absolute difference between the user function and computed parameterization on the
  /// \param requested grid, common for all 1D components
  /// \param precD   : optional precison per component
  Chebyshev3D(void (*ptr)(float*, float*), int DimOut, const Float_t* bmin, const Float_t* bmax, Float_t prec = 1E-6,
              Bool_t run = kTRUE, const Float_t* precD = nullptr);
#endif

  ~Chebyshev3D() override
  {
    Clear();
  }

  Chebyshev3D& operator=(const Chebyshev3D& rhs);

  void Eval(const Float_t* par, Float_t* res);

  Float_t Eval(const Float_t* par, int idim);

  void Eval(const Double_t* par, Double_t* res);

  Double_t Eval(const Double_t* par, int idim);

  void evaluateDerivative(int dimd, const Float_t* par, Float_t* res);

  void evaluateDerivative2(int dimd1, int dimd2, const Float_t* par, Float_t* res);

  Float_t evaluateDerivative(int dimd, const Float_t* par, int idim);

  Float_t evaluateDerivative2(int dimd1, int dimd2, const Float_t* par, int idim);

  void evaluateDerivative3D(const Float_t* par, Float_t dbdr[3][3]);

  void evaluateDerivative3D2(const Float_t* par, Float_t dbdrdr[3][3][3]);

  void Print(const Option_t* opt = "") const override;

  Bool_t isInside(const Float_t* par) const;

  Bool_t isInside(const Double_t* par) const;

  Chebyshev3DCalc* getChebyshevCalc(int i) const
  {
    return (Chebyshev3DCalc*)mChebyshevParameter.UncheckedAt(i);
  }

  Float_t getBoundMin(int i) const
  {
    return mMinBoundaries[i];
  }

  Float_t getBoundMax(int i) const
  {
    return mMaxBoundaries[i];
  }

  Float_t* getBoundMin() const
  {
    return (float*)mMinBoundaries;
  }

  Float_t* getBoundMax() const
  {
    return (float*)mMaxBoundaries;
  }

  Float_t getPrecision() const
  {
    return mPrecision;
  }

  void shiftBound(int id, float dif);

  void loadData(const char* inpFile);

  void loadData(FILE* stream);

#ifdef _INC_CREATION_Chebyshev3D_
  void invertSign();
  int* getNcNeeded(float xyz[3], int dimVar, float mn, float mx, float prec, Int_t npCheck = 30);
  void estimateNumberOfPoints(float prec, int gridBC[3][3], Int_t npd1 = 30, Int_t npd2 = 30, Int_t npd3 = 30);
  void saveData(const char* outfile, Bool_t append = kFALSE) const;
  void saveData(FILE* stream = stdout) const;

  void setuserFunction(const char* name);
  void setuserFunction(void (*ptr)(float*, float*));
  void evaluateUserFunction(const Float_t* x, Float_t* res);
  TH1* TestRMS(int idim, int npoints = 1000, TH1* histo = nullptr);
  static Int_t calculateChebyshevCoefficients(const Float_t* funval, int np, Float_t* outCoefs, Float_t prec = -1);
#endif

 protected:
  void Clear(const Option_t* option = "") override;

  void setDimOut(const int d, const float* prec = nullptr);

  void prepareBoundaries(const Float_t* bmin, const Float_t* bmax);

#ifdef _INC_CREATION_Chebyshev3D_
  void evaluateUserFunction();
  void defineGrid(const Int_t* npoints);
  Int_t chebyshevFit(); // fit all output dimensions
  Int_t chebyshevFit(int dmOut);
  void setPrecision(float prec)
  {
    mPrecision = prec;
  }
#endif

  Float_t mapToInternal(Float_t x, Int_t d) const; // map x to [-1:1]
  Float_t mapToExternal(Float_t x, Int_t d) const
  {
    return x / mBoundaryMappingScale[d] + mBoundaryMappingOffset[d];
  }                                                  // map from [-1:1] to x
  Double_t mapToInternal(Double_t x, Int_t d) const; // map x to [-1:1]
  Double_t mapToExternal(Double_t x, Int_t d) const
  {
    return x / mBoundaryMappingScale[d] + mBoundaryMappingOffset[d];
  } // map from [-1:1] to x

 private:
  Int_t mOutputArrayDimension;       ///< dimension of the ouput array
  Float_t mPrecision;                ///< requested precision
  Float_t mMinBoundaries[3];         ///< min boundaries in each dimension
  Float_t mMaxBoundaries[3];         ///< max boundaries in each dimension
  Float_t mBoundaryMappingScale[3];  ///< scale for boundary mapping to [-1:1] interval
  Float_t mBoundaryMappingOffset[3]; ///< offset for boundary mapping to [-1:1] interval
  TObjArray mChebyshevParameter;     ///< Chebyshev parameterization for each output dimension

  Int_t mMaxCoefficients;               //! max possible number of coefs per parameterization
  Int_t mNumberOfPoints[3];             //! number of used points in each dimension
  Float_t mTemporaryCoefficient[3];     //! temporary vector for coefs calculation
  Float_t* mTemporaryUserResults;       //! temporary vector for results of user function calculation
  Float_t* mTemporaryChebyshevGrid;     //! temporary buffer for Chebyshef roots grid
  Int_t mTemporaryChebyshevGridOffs[3]; //! start of grid for each dimension
  TString mUserFunctionName;            //! name of user macro containing the function of  "void (*fcn)(float*,float*)" format
  TMethodCall* mUserMacro;              //! Pointer to MethodCall for function from user macro

  static const Float_t sMinimumPrecision; ///< minimum precision allowed

  ClassDefOverride(o2::math_utils::Chebyshev3D,
                   2) // Chebyshev parametrization for 3D->N function
};

/// Checks if the point is inside of the fitted box
inline Bool_t Chebyshev3D::isInside(const Float_t* par) const
{
  for (int i = 3; i--;) {
    if (mMinBoundaries[i] > par[i] || par[i] > mMaxBoundaries[i]) {
      return kFALSE;
    }
  }
  return kTRUE;
}

/// Checks if the point is inside of the fitted box
inline Bool_t Chebyshev3D::isInside(const Double_t* par) const
{
  for (int i = 3; i--;) {
    if (mMinBoundaries[i] > par[i] || par[i] > mMaxBoundaries[i]) {
      return kFALSE;
    }
  }
  return kTRUE;
}

/// Evaluates Chebyshev parameterization for 3d->DimOut function
inline void Chebyshev3D::Eval(const Float_t* par, Float_t* res)
{
  for (int i = 3; i--;) {
    mTemporaryCoefficient[i] = mapToInternal(par[i], i);
  }
  for (int i = mOutputArrayDimension; i--;) {
    res[i] = getChebyshevCalc(i)->Eval(mTemporaryCoefficient);
  }
}

/// Evaluates Chebyshev parameterization for 3d->DimOut function
inline void Chebyshev3D::Eval(const Double_t* par, Double_t* res)
{
  for (int i = 3; i--;) {
    mTemporaryCoefficient[i] = mapToInternal(par[i], i);
  }
  for (int i = mOutputArrayDimension; i--;) {
    res[i] = getChebyshevCalc(i)->Eval(mTemporaryCoefficient);
  }
}

/// Evaluates Chebyshev parameterization for idim-th output dimension of 3d->DimOut function
inline Double_t Chebyshev3D::Eval(const Double_t* par, int idim)
{
  for (int i = 3; i--;) {
    mTemporaryCoefficient[i] = mapToInternal(par[i], i);
  }
  return getChebyshevCalc(idim)->Eval(mTemporaryCoefficient);
}

/// Evaluates Chebyshev parameterization for idim-th output dimension of 3d->DimOut function
inline Float_t Chebyshev3D::Eval(const Float_t* par, int idim)
{
  for (int i = 3; i--;) {
    mTemporaryCoefficient[i] = mapToInternal(par[i], i);
  }
  return getChebyshevCalc(idim)->Eval(mTemporaryCoefficient);
}

/// Returns the gradient matrix
inline void Chebyshev3D::evaluateDerivative3D(const Float_t* par, Float_t dbdr[3][3])
{
  for (int i = 3; i--;) {
    mTemporaryCoefficient[i] = mapToInternal(par[i], i);
  }
  for (int ib = 3; ib--;) {
    for (int id = 3; id--;) {
      dbdr[ib][id] = getChebyshevCalc(ib)->evaluateDerivative(id, mTemporaryCoefficient) * mBoundaryMappingScale[id];
    }
  }
}

/// Returns the gradient matrix
inline void Chebyshev3D::evaluateDerivative3D2(const Float_t* par, Float_t dbdrdr[3][3][3])
{
  for (int i = 3; i--;) {
    mTemporaryCoefficient[i] = mapToInternal(par[i], i);
  }
  for (int ib = 3; ib--;) {
    for (int id = 3; id--;) {
      for (int id1 = 3; id1--;) {
        dbdrdr[ib][id][id1] = getChebyshevCalc(ib)->evaluateDerivative2(id, id1, mTemporaryCoefficient) *
                              mBoundaryMappingScale[id] * mBoundaryMappingScale[id1];
      }
    }
  }
}

// Evaluates Chebyshev parameterization derivative for 3d->DimOut function
inline void Chebyshev3D::evaluateDerivative(int dimd, const Float_t* par, Float_t* res)
{
  for (int i = 3; i--;) {
    mTemporaryCoefficient[i] = mapToInternal(par[i], i);
  }
  for (int i = mOutputArrayDimension; i--;) {
    res[i] = getChebyshevCalc(i)->evaluateDerivative(dimd, mTemporaryCoefficient) * mBoundaryMappingScale[dimd];
  };
}

// Evaluates Chebyshev parameterization 2nd derivative over dimd1 and dimd2 dimensions for 3d->DimOut function
inline void Chebyshev3D::evaluateDerivative2(int dimd1, int dimd2, const Float_t* par, Float_t* res)
{
  for (int i = 3; i--;) {
    mTemporaryCoefficient[i] = mapToInternal(par[i], i);
  }
  for (int i = mOutputArrayDimension; i--;) {
    res[i] = getChebyshevCalc(i)->evaluateDerivative2(dimd1, dimd2, mTemporaryCoefficient) *
             mBoundaryMappingScale[dimd1] * mBoundaryMappingScale[dimd2];
  }
}

/// Evaluates Chebyshev parameterization derivative over dimd dimention for idim-th output dimension of 3d->DimOut
/// function
inline Float_t Chebyshev3D::evaluateDerivative(int dimd, const Float_t* par, int idim)
{
  for (int i = 3; i--;) {
    mTemporaryCoefficient[i] = mapToInternal(par[i], i);
  }
  return getChebyshevCalc(idim)->evaluateDerivative(dimd, mTemporaryCoefficient) * mBoundaryMappingScale[dimd];
}

/// Evaluates Chebyshev parameterization 2ns derivative over dimd1 and dimd2 dimensions for idim-th output dimension of
/// 3d->DimOut function
inline Float_t Chebyshev3D::evaluateDerivative2(int dimd1, int dimd2, const Float_t* par, int idim)
{
  for (int i = 3; i--;) {
    mTemporaryCoefficient[i] = mapToInternal(par[i], i);
  }
  return getChebyshevCalc(idim)->evaluateDerivative2(dimd1, dimd2, mTemporaryCoefficient) *
         mBoundaryMappingScale[dimd1] * mBoundaryMappingScale[dimd2];
}

/// Μaps x to [-1:1]
inline Float_t Chebyshev3D::mapToInternal(Float_t x, Int_t d) const
{
#ifdef _BRING_TO_BOUNDARY_
  T res = (x - mBoundaryMappingOffset[d]) * mBoundaryMappingScale[d];
  if (res < -1) {
    return -1;
  }
  if (res > 1) {
    return 1;
  }
  return res;
#else
  return (x - mBoundaryMappingOffset[d]) * mBoundaryMappingScale[d];
#endif
}

/// Μaps x to [-1:1]
inline Double_t Chebyshev3D::mapToInternal(Double_t x, Int_t d) const
{
#ifdef _BRING_TO_BOUNDARY_
  T res = (x - mBoundaryMappingOffset[d]) * mBoundaryMappingScale[d];
  if (res < -1) {
    return -1;
  }
  if (res > 1) {
    return 1;
  }
  return res;
#else
  return (x - mBoundaryMappingOffset[d]) * mBoundaryMappingScale[d];
#endif
}
} // namespace math_utils
} // namespace o2

#endif
