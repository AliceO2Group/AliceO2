// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

/// \file Cheb3DCalc.h
/// \brief Definition of the Cheb3DCalc class
/// \author ruben.shahoyan@cern.ch 09/09/2006

#ifndef ALICEO2_MATHUTILS_CHEBYSHEV3DCALC_H_
#define ALICEO2_MATHUTILS_CHEBYSHEV3DCALC_H_

#include <TNamed.h>  // for TNamed
#include <cstdio>   // for FILE, stdout
#include "Rtypes.h"  // for Float_t, UShort_t, Int_t, Double_t, etc

class TString;


// To decrease the compilable code size comment this define. This will exclude the routines
// used for the calculation and saving of the coefficients.
#define _INC_CREATION_Chebyshev3D_

// When _BRING_TO_BOUNDARY_ is defined, the point outside of the fitted folume is assumed to be on the surface
// #define _BRING_TO_BOUNDARY_

namespace o2 {
namespace mathUtils {
class Chebyshev3DCalc : public TNamed
{

  public:

    /// Default constructor
    Chebyshev3DCalc();

    /// Copy constructor
    Chebyshev3DCalc(const Chebyshev3DCalc &src);

    /// Constructor from coefficients stream
    Chebyshev3DCalc(FILE *stream);

    /// Default destructor
    ~Chebyshev3DCalc() override
    {
      Clear();
    }

    /// Assignment operator
    Chebyshev3DCalc &operator=(const Chebyshev3DCalc &rhs);

    /// Prints info
    void Print(const Option_t *opt = "") const override;

    /// Loads coefficients from the stream
    void loadData(FILE *stream);

    /// Evaluates Chebyshev parameterization derivative in given dimension for 3D function.
    /// VERY IMPORTANT: par must contain the function arguments ALREADY MAPPED to [-1:1] interval
    Float_t evaluateDerivative(int dim, const Float_t *par) const;

    /// Evaluates Chebyshev parameterization 2n derivative in given dimensions  for 3D function.
    /// VERY IMPORTANT: par must contain the function arguments ALREADY MAPPED to [-1:1] interval
    Float_t evaluateDerivative2(int dim1, int dim2, const Float_t *par) const;

#ifdef _INC_CREATION_Chebyshev3D_

    /// Writes coefficients data to output text file, optionally appending on the end of existing file
    void saveData(const char *outfile, Bool_t append = kFALSE) const;

    // Writes coefficients data to existing output stream
    // Note: mNumberOfColumns, mNumberOfElementsBound2D and mColumnAtRowBeginning are not stored, will be computed on fly
    // during the loading of this file
    void saveData(FILE *stream = stdout) const;

#endif

    /// Sets maximum number of significant rows in the coefficients matrix
    void initializeRows(int nr);

    /// Sets maximum number of significant columns in the coefficients matrix
    void initializeColumns(int nc);

    Int_t getNumberOfCoefficients() const
    {
      return mNumberOfCoefficients;
    }

    Int_t getNumberOfColumns() const
    {
      return (Int_t) mNumberOfColumns;
    }

    Int_t getNumberOfRows() const
    {
      return (Int_t) mNumberOfRows;
    }

    Int_t getNumberOfElementsBound2D() const
    {
      return (Int_t) mNumberOfElementsBound2D;
    }

    Int_t getMaxColumnsAtRow() const;

    UShort_t *getNumberOfColumnsAtRow() const
    {
      return mNumberOfColumnsAtRow;
    }

    UShort_t *getColAtRowBg() const
    {
      return mColumnAtRowBeginning;
    }

    Float_t getPrecision() const
    {
      return mPrecision;
    }

    /// Sets requested precision
    void setPrecision(Float_t prc = 1e-6)
    {
      mPrecision = prc;
    }

    /// Sets maximum number of significant coefficients for given row/column of coefficients 3D matrix
    void initializeElementBound2D(int ne);

    UShort_t *getCoefficientBound2D0() const
    {
      return mCoefficientBound2D0;
    }

    UShort_t *getCoefficientBound2D1() const
    {
      return mCoefficientBound2D1;
    }

    /// Deletes all dynamically allocated structures
    void Clear(const Option_t *option = "") override;

    static Float_t chebyshevEvaluation1D(Float_t x, const Float_t *array, int ncf);

    /// Evaluates 1D Chebyshev parameterization's derivative. x is the argument mapped to [-1:1] interval
    static Float_t chebyshevEvaluation1Derivative(Float_t x, const Float_t *array, int ncf);

    /// Evaluates 1D Chebyshev parameterization's 2nd derivative. x is the argument mapped to [-1:1] interval
    static Float_t chebyshevEvaluation1Derivative2(Float_t x, const Float_t *array, int ncf);

    /// Sets total number of significant coefficients
    void initializeCoefficients(int nc);

    Float_t *getCoefficients() const
    {
      return mCoefficients;
    }

    /// Reads single line from the stream, skipping empty and commented lines. EOF is not expected
    static void readLine(TString &str, FILE *stream);

    Float_t Eval(const Float_t *par) const;

    Double_t Eval(const Double_t *par) const;

  private:
    Int_t mNumberOfCoefficients;    ///< total number of coeeficients
    Int_t mNumberOfRows;            ///< number of significant rows in the 3D coeffs matrix
    Int_t mNumberOfColumns;         ///< max number of significant cols in the 3D coeffs matrix
    Int_t mNumberOfElementsBound2D; ///< number of elements (mNumberOfRows*mNumberOfColumns) to store for the 2D boundary
    Float_t mPrecision;             ///< requested precision
    /// of significant coeffs
    UShort_t *
      mNumberOfColumnsAtRow; //[mNumberOfRows] number of sighificant columns (2nd dim) at each row of 3D coefs matrix
    UShort_t *mColumnAtRowBeginning; //[mNumberOfRows] beginning of significant columns (2nd dim) for row in the 2D
    // boundary matrix
    UShort_t *mCoefficientBound2D0; //[mNumberOfElementsBound2D] 2D matrix defining the boundary of significance for 3D
    // coeffs.matrix
    //(Ncoefs for col/row)
    UShort_t *mCoefficientBound2D1; //[mNumberOfElementsBound2D] 2D matrix defining the start beginning of significant
    // coeffs for col/row
    Float_t *mCoefficients; //[mNumberOfCoefficients] array of Chebyshev coefficients

    Float_t *mTemporaryCoefficients2D; //[mNumberOfColumns] temp. coeffs for 2d summation
    Float_t *mTemporaryCoefficients1D; //[mNumberOfRows] temp. coeffs for 1d summation

    ClassDefOverride(o2::mathUtils::Chebyshev3DCalc,
    2) // Class for interpolation of 3D->1 function by Chebyshev parametrization
};

/// Evaluates 1D Chebyshev parameterization. x is the argument mapped to [-1:1] interval
inline Float_t Chebyshev3DCalc::chebyshevEvaluation1D(Float_t x, const Float_t *array, int ncf)
{
  if (ncf <= 0) {
    return 0;
  }

  Float_t b0, b1, b2, x2 = x + x;
  b0 = array[--ncf];
  b1 = b2 = 0;

  for (int i = ncf; i--;) {
    b2 = b1;
    b1 = b0;
    b0 = array[i] + x2 * b1 - b2;
  }
  return b0 - x * b1;
}

/// Evaluates Chebyshev parameterization for 3D function.
/// VERY IMPORTANT: par must contain the function arguments ALREADY MAPPED to [-1:1] interval
inline Float_t Chebyshev3DCalc::Eval(const Float_t *par) const
{
  if (!mNumberOfRows) {
    return 0.;
  }
  int ncfRC;
  for (int id0 = mNumberOfRows; id0--;) {
    int nCLoc = mNumberOfColumnsAtRow[id0]; // number of significant coefs on this row
    int col0 = mColumnAtRowBeginning[id0];  // beginning of local column in the 2D boundary matrix
    for (int id1 = nCLoc; id1--;) {
      int id = id1 + col0;
      mTemporaryCoefficients2D[id1] = (ncfRC = mCoefficientBound2D0[id])
                                      ? chebyshevEvaluation1D(par[2], mCoefficients + mCoefficientBound2D1[id], ncfRC)
                                      : 0.0;
    }
    mTemporaryCoefficients1D[id0] = nCLoc > 0 ? chebyshevEvaluation1D(par[1], mTemporaryCoefficients2D, nCLoc) : 0.0;
  }
  return chebyshevEvaluation1D(par[0], mTemporaryCoefficients1D, mNumberOfRows);
}

/// Evaluates Chebyshev parameterization for 3D function.
/// VERY IMPORTANT: par must contain the function arguments ALREADY MAPPED to [-1:1] interval
inline Double_t Chebyshev3DCalc::Eval(const Double_t *par) const
{
  if (!mNumberOfRows) {
    return 0.;
  }
  int ncfRC;
  for (int id0 = mNumberOfRows; id0--;) {
    int nCLoc = mNumberOfColumnsAtRow[id0]; // number of significant coefs on this row
    int col0 = mColumnAtRowBeginning[id0];  // beginning of local column in the 2D boundary matrix
    for (int id1 = nCLoc; id1--;) {
      int id = id1 + col0;
      mTemporaryCoefficients2D[id1] = (ncfRC = mCoefficientBound2D0[id])
                                      ? chebyshevEvaluation1D(par[2], mCoefficients + mCoefficientBound2D1[id], ncfRC)
                                      : 0.0;
    }
    mTemporaryCoefficients1D[id0] = nCLoc > 0 ? chebyshevEvaluation1D(par[1], mTemporaryCoefficients2D, nCLoc) : 0.0;
  }
  return chebyshevEvaluation1D(par[0], mTemporaryCoefficients1D, mNumberOfRows);
}
}
}

#endif
