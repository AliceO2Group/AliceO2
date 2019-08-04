// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

/// \file Cheb3DCalc.cxx
/// \brief Implementation of the Cheb3DCalc class
/// \author ruben.shahoyan@cern.ch 09/09/2006

#include "MathUtils/Chebyshev3DCalc.h"
#include <TSystem.h> // for TSystem, gSystem
#include "TNamed.h"  // for TNamed
#include "TString.h" // for TString, TString::EStripType::kBoth

using namespace o2::math_utils;

ClassImp(Chebyshev3DCalc);

Chebyshev3DCalc::Chebyshev3DCalc()
  : mNumberOfCoefficients(0),
    mNumberOfRows(0),
    mNumberOfColumns(0),
    mNumberOfElementsBound2D(0),
    mPrecision(0),
    mNumberOfColumnsAtRow(nullptr),
    mColumnAtRowBeginning(nullptr),
    mCoefficientBound2D0(nullptr),
    mCoefficientBound2D1(nullptr),
    mCoefficients(nullptr),
    mTemporaryCoefficients2D(nullptr),
    mTemporaryCoefficients1D(nullptr)
{
}

Chebyshev3DCalc::Chebyshev3DCalc(const Chebyshev3DCalc& src)
  : TNamed(src),
    mNumberOfCoefficients(src.mNumberOfCoefficients),
    mNumberOfRows(src.mNumberOfRows),
    mNumberOfColumns(src.mNumberOfColumns),
    mNumberOfElementsBound2D(src.mNumberOfElementsBound2D),
    mPrecision(src.mPrecision),
    mNumberOfColumnsAtRow(nullptr),
    mColumnAtRowBeginning(nullptr),
    mCoefficientBound2D0(nullptr),
    mCoefficientBound2D1(nullptr),
    mCoefficients(nullptr),
    mTemporaryCoefficients2D(nullptr),
    mTemporaryCoefficients1D(nullptr)
{
  if (src.mNumberOfColumnsAtRow) {
    mNumberOfColumnsAtRow = new UShort_t[mNumberOfRows];
    for (int i = mNumberOfRows; i--;) {
      mNumberOfColumnsAtRow[i] = src.mNumberOfColumnsAtRow[i];
    }
  }
  if (src.mColumnAtRowBeginning) {
    mColumnAtRowBeginning = new UShort_t[mNumberOfRows];
    for (int i = mNumberOfRows; i--;) {
      mColumnAtRowBeginning[i] = src.mColumnAtRowBeginning[i];
    }
  }
  if (src.mCoefficientBound2D0) {
    mCoefficientBound2D0 = new UShort_t[mNumberOfElementsBound2D];
    for (int i = mNumberOfElementsBound2D; i--;) {
      mCoefficientBound2D0[i] = src.mCoefficientBound2D0[i];
    }
  }
  if (src.mCoefficientBound2D1) {
    mCoefficientBound2D1 = new UShort_t[mNumberOfElementsBound2D];
    for (int i = mNumberOfElementsBound2D; i--;) {
      mCoefficientBound2D1[i] = src.mCoefficientBound2D1[i];
    }
  }
  if (src.mCoefficients) {
    mCoefficients = new Float_t[mNumberOfCoefficients];
    for (int i = mNumberOfCoefficients; i--;) {
      mCoefficients[i] = src.mCoefficients[i];
    }
  }
  if (src.mTemporaryCoefficients2D) {
    mTemporaryCoefficients2D = new Float_t[mNumberOfColumns];
  }
  if (src.mTemporaryCoefficients1D) {
    mTemporaryCoefficients1D = new Float_t[mNumberOfRows];
  }
}

Chebyshev3DCalc::Chebyshev3DCalc(FILE* stream)
  : mNumberOfCoefficients(0),
    mNumberOfRows(0),
    mNumberOfColumns(0),
    mNumberOfElementsBound2D(0),
    mPrecision(0),
    mNumberOfColumnsAtRow(nullptr),
    mColumnAtRowBeginning(nullptr),
    mCoefficientBound2D0(nullptr),
    mCoefficientBound2D1(nullptr),
    mCoefficients(nullptr),
    mTemporaryCoefficients2D(nullptr),
    mTemporaryCoefficients1D(nullptr)
{
  loadData(stream);
}

Chebyshev3DCalc& Chebyshev3DCalc::operator=(const Chebyshev3DCalc& rhs)
{
  if (this != &rhs) {
    Clear();
    SetName(rhs.GetName());
    SetTitle(rhs.GetTitle());
    mNumberOfCoefficients = rhs.mNumberOfCoefficients;
    mNumberOfRows = rhs.mNumberOfRows;
    mNumberOfColumns = rhs.mNumberOfColumns;
    mPrecision = rhs.mPrecision;
    if (rhs.mNumberOfColumnsAtRow) {
      mNumberOfColumnsAtRow = new UShort_t[mNumberOfRows];
      for (int i = mNumberOfRows; i--;) {
        mNumberOfColumnsAtRow[i] = rhs.mNumberOfColumnsAtRow[i];
      }
    }
    if (rhs.mColumnAtRowBeginning) {
      mColumnAtRowBeginning = new UShort_t[mNumberOfRows];
      for (int i = mNumberOfRows; i--;) {
        mColumnAtRowBeginning[i] = rhs.mColumnAtRowBeginning[i];
      }
    }
    if (rhs.mCoefficientBound2D0) {
      mCoefficientBound2D0 = new UShort_t[mNumberOfElementsBound2D];
      for (int i = mNumberOfElementsBound2D; i--;) {
        mCoefficientBound2D0[i] = rhs.mCoefficientBound2D0[i];
      }
    }
    if (rhs.mCoefficientBound2D1) {
      mCoefficientBound2D1 = new UShort_t[mNumberOfElementsBound2D];
      for (int i = mNumberOfElementsBound2D; i--;) {
        mCoefficientBound2D1[i] = rhs.mCoefficientBound2D1[i];
      }
    }
    if (rhs.mCoefficients) {
      mCoefficients = new Float_t[mNumberOfCoefficients];
      for (int i = mNumberOfCoefficients; i--;) {
        mCoefficients[i] = rhs.mCoefficients[i];
      }
    }
    if (rhs.mTemporaryCoefficients2D) {
      mTemporaryCoefficients2D = new Float_t[mNumberOfColumns];
    }
    if (rhs.mTemporaryCoefficients1D) {
      mTemporaryCoefficients1D = new Float_t[mNumberOfRows];
    }
  }
  return *this;
}

void Chebyshev3DCalc::Clear(const Option_t*)
{
  if (mTemporaryCoefficients2D) {
    delete[] mTemporaryCoefficients2D;
    mTemporaryCoefficients2D = nullptr;
  }
  if (mTemporaryCoefficients1D) {
    delete[] mTemporaryCoefficients1D;
    mTemporaryCoefficients1D = nullptr;
  }
  if (mCoefficients) {
    delete[] mCoefficients;
    mCoefficients = nullptr;
  }
  if (mCoefficientBound2D0) {
    delete[] mCoefficientBound2D0;
    mCoefficientBound2D0 = nullptr;
  }
  if (mCoefficientBound2D1) {
    delete[] mCoefficientBound2D1;
    mCoefficientBound2D1 = nullptr;
  }
  if (mNumberOfColumnsAtRow) {
    delete[] mNumberOfColumnsAtRow;
    mNumberOfColumnsAtRow = nullptr;
  }
  if (mColumnAtRowBeginning) {
    delete[] mColumnAtRowBeginning;
    mColumnAtRowBeginning = nullptr;
  }
}

void Chebyshev3DCalc::Print(const Option_t*) const
{
  printf("Chebyshev parameterization data %s for 3D->1 function, precision: %e\n",
         GetName(), mPrecision);
  int nmax3d = 0;
  for (int i = mNumberOfElementsBound2D; i--;) {
    if (mCoefficientBound2D0[i] > nmax3d) {
      nmax3d = mCoefficientBound2D0[i];
    }
  }
  printf("%d coefficients in %dx%dx%d matrix\n", mNumberOfCoefficients, mNumberOfRows, mNumberOfColumns, nmax3d);
}

Float_t Chebyshev3DCalc::evaluateDerivative(int dim, const Float_t* par) const
{
  int ncfRC;
  for (int id0 = mNumberOfRows; id0--;) {
    int nCLoc = mNumberOfColumnsAtRow[id0]; // number of significant coefs on this row
    if (!nCLoc) {
      mTemporaryCoefficients1D[id0] = 0;
      continue;
    }
    //
    int col0 = mColumnAtRowBeginning[id0]; // beginning of local column in the 2D boundary matrix
    for (int id1 = nCLoc; id1--;) {
      int id = id1 + col0;
      if (!(ncfRC = mCoefficientBound2D0[id])) {
        mTemporaryCoefficients2D[id1] = 0;
        continue;
      }
      if (dim == 2) {
        mTemporaryCoefficients2D[id1] =
          chebyshevEvaluation1Derivative(par[2], mCoefficients + mCoefficientBound2D1[id], ncfRC);
      } else {
        mTemporaryCoefficients2D[id1] = chebyshevEvaluation1D(par[2], mCoefficients + mCoefficientBound2D1[id], ncfRC);
      }
    }
    if (dim == 1) {
      mTemporaryCoefficients1D[id0] = chebyshevEvaluation1Derivative(par[1], mTemporaryCoefficients2D, nCLoc);
    } else {
      mTemporaryCoefficients1D[id0] = chebyshevEvaluation1D(par[1], mTemporaryCoefficients2D, nCLoc);
    }
  }
  return (dim == 0) ? chebyshevEvaluation1Derivative(par[0], mTemporaryCoefficients1D, mNumberOfRows)
                    : chebyshevEvaluation1D(par[0], mTemporaryCoefficients1D, mNumberOfRows);
}

Float_t Chebyshev3DCalc::evaluateDerivative2(int dim1, int dim2, const Float_t* par) const
{
  Bool_t same = dim1 == dim2;
  int ncfRC;
  for (int id0 = mNumberOfRows; id0--;) {
    int nCLoc = mNumberOfColumnsAtRow[id0]; // number of significant coefs on this row
    if (!nCLoc) {
      mTemporaryCoefficients1D[id0] = 0;
      continue;
    }
    int col0 = mColumnAtRowBeginning[id0]; // beginning of local column in the 2D boundary matrix
    for (int id1 = nCLoc; id1--;) {
      int id = id1 + col0;
      if (!(ncfRC = mCoefficientBound2D0[id])) {
        mTemporaryCoefficients2D[id1] = 0;
        continue;
      }
      if (dim1 == 2 || dim2 == 2) {
        mTemporaryCoefficients2D[id1] =
          same ? chebyshevEvaluation1Derivative2(par[2], mCoefficients + mCoefficientBound2D1[id], ncfRC)
               : chebyshevEvaluation1Derivative(par[2], mCoefficients + mCoefficientBound2D1[id], ncfRC);
      } else {
        mTemporaryCoefficients2D[id1] = chebyshevEvaluation1D(par[2], mCoefficients + mCoefficientBound2D1[id], ncfRC);
      }
    }
    if (dim1 == 1 || dim2 == 1) {
      mTemporaryCoefficients1D[id0] = same ? chebyshevEvaluation1Derivative2(par[1], mTemporaryCoefficients2D, nCLoc)
                                           : chebyshevEvaluation1Derivative(par[1], mTemporaryCoefficients2D, nCLoc);
    } else {
      mTemporaryCoefficients1D[id0] = chebyshevEvaluation1D(par[1], mTemporaryCoefficients2D, nCLoc);
    }
  }
  return (dim1 == 0 || dim2 == 0)
           ? (same ? chebyshevEvaluation1Derivative2(par[0], mTemporaryCoefficients1D, mNumberOfRows)
                   : chebyshevEvaluation1Derivative(par[0], mTemporaryCoefficients1D, mNumberOfRows))
           : chebyshevEvaluation1D(par[0], mTemporaryCoefficients1D, mNumberOfRows);
}

#ifdef _INC_CREATION_Chebyshev3D_
void Chebyshev3DCalc::saveData(const char* outfile, Bool_t append) const
{
  TString strf = outfile;
  gSystem->ExpandPathName(strf);
  FILE* stream = fopen(strf, append ? "a" : "w");
  saveData(stream);
  fclose(stream);
}
#endif

#ifdef _INC_CREATION_Chebyshev3D_
void Chebyshev3DCalc::saveData(FILE* stream) const
{
  fprintf(stream, "#\nSTART %s\n", GetName());
  fprintf(stream, "# Number of rows\n%d\n", mNumberOfRows);

  fprintf(stream, "# Number of columns per row\n");
  for (int i = 0; i < mNumberOfRows; i++) {
    fprintf(stream, "%d\n", mNumberOfColumnsAtRow[i]);
  }

  fprintf(stream, "# Number of Coefs in each significant block of third dimension\n");
  for (int i = 0; i < mNumberOfElementsBound2D; i++) {
    fprintf(stream, "%d\n", mCoefficientBound2D0[i]);
  }

  fprintf(stream, "# Coefficients\n");
  for (int i = 0; i < mNumberOfCoefficients; i++) {
    fprintf(stream, "%+.8e\n", mCoefficients[i]);
  }
  fprintf(stream, "# Precision\n");
  fprintf(stream, "%+.8e\n", mPrecision);
  //
  fprintf(stream, "END %s\n", GetName());
}
#endif

void Chebyshev3DCalc::loadData(FILE* stream)
{
  if (!stream) {
    Error("LoadData", "No stream provided.\nStop");
    exit(1);
  }
  TString buffs;
  Clear();
  readLine(buffs, stream);

  if (!buffs.BeginsWith("START")) {
    Error("LoadData", "Expected: \"START <fit_name>\", found \"%s\"\nStop\n", buffs.Data());
    exit(1);
  }

  if (buffs.First(' ') > 0) {
    SetName(buffs.Data() + buffs.First(' ') + 1);
  }

  readLine(buffs, stream); // NRows
  mNumberOfRows = buffs.Atoi();

  if (mNumberOfRows < 0 || !buffs.IsDigit()) {
    Error("LoadData", "Expected: '<number_of_rows>', found \"%s\"\nStop\n", buffs.Data());
    exit(1);
  }

  mNumberOfColumns = 0;
  mNumberOfElementsBound2D = 0;
  initializeRows(mNumberOfRows);

  for (int id0 = 0; id0 < mNumberOfRows; id0++) {
    readLine(buffs, stream); // n.cols at this row
    mNumberOfColumnsAtRow[id0] = buffs.Atoi();
    mColumnAtRowBeginning[id0] = mNumberOfElementsBound2D; // begining of this row in 2D boundary surface
    mNumberOfElementsBound2D += mNumberOfColumnsAtRow[id0];
    if (mNumberOfColumns < mNumberOfColumnsAtRow[id0]) {
      mNumberOfColumns = mNumberOfColumnsAtRow[id0];
    }
  }
  initializeColumns(mNumberOfColumns);

  mNumberOfCoefficients = 0;
  initializeElementBound2D(mNumberOfElementsBound2D);

  for (int i = 0; i < mNumberOfElementsBound2D; i++) {
    readLine(buffs, stream); // n.coeffs at 3-d dimension for the given column/row
    mCoefficientBound2D0[i] = buffs.Atoi();
    mCoefficientBound2D1[i] = mNumberOfCoefficients;
    mNumberOfCoefficients += mCoefficientBound2D0[i];
  }

  initializeCoefficients(mNumberOfCoefficients);
  for (int i = 0; i < mNumberOfCoefficients; i++) {
    readLine(buffs, stream);
    mCoefficients[i] = buffs.Atof();
  }
  // read precision
  readLine(buffs, stream);
  mPrecision = buffs.Atof();

  // check end_of_data record
  readLine(buffs, stream);
  if (!buffs.BeginsWith("END") || !buffs.Contains(GetName())) {
    Error("LoadData", "Expected \"END %s\", found \"%s\".\nStop\n", GetName(), buffs.Data());
    exit(1);
  }
}

void Chebyshev3DCalc::readLine(TString& str, FILE* stream)
{
  while (str.Gets(stream)) {
    str = str.Strip(TString::kBoth, ' ');
    if (str.IsNull() || str.BeginsWith("#")) {
      continue;
    }
    return;
  }
  fprintf(stderr, "Chebyshev3D::readLine: Failed to read from stream.\nStop");
  exit(1); // normally, should not reach here
}

void Chebyshev3DCalc::initializeRows(int nr)
{
  if (mNumberOfColumnsAtRow) {
    delete[] mNumberOfColumnsAtRow;
    mNumberOfColumnsAtRow = nullptr;
  }
  if (mColumnAtRowBeginning) {
    delete[] mColumnAtRowBeginning;
    mColumnAtRowBeginning = nullptr;
  }
  if (mTemporaryCoefficients1D) {
    delete[] mTemporaryCoefficients1D;
    mTemporaryCoefficients1D = nullptr;
  }
  mNumberOfRows = nr;
  if (mNumberOfRows) {
    mNumberOfColumnsAtRow = new UShort_t[mNumberOfRows];
    mTemporaryCoefficients1D = new Float_t[mNumberOfRows];
    mColumnAtRowBeginning = new UShort_t[mNumberOfRows];
    for (int i = mNumberOfRows; i--;) {
      mNumberOfColumnsAtRow[i] = mColumnAtRowBeginning[i] = 0;
    }
  }
}

void Chebyshev3DCalc::initializeColumns(int nc)
{
  mNumberOfColumns = nc;
  if (mTemporaryCoefficients2D) {
    delete[] mTemporaryCoefficients2D;
    mTemporaryCoefficients2D = nullptr;
  }
  if (mNumberOfColumns) {
    mTemporaryCoefficients2D = new Float_t[mNumberOfColumns];
  }
}

void Chebyshev3DCalc::initializeElementBound2D(int ne)
{
  if (mCoefficientBound2D0) {
    delete[] mCoefficientBound2D0;
    mCoefficientBound2D0 = nullptr;
  }
  if (mCoefficientBound2D1) {
    delete[] mCoefficientBound2D1;
    mCoefficientBound2D1 = nullptr;
  }
  mNumberOfElementsBound2D = ne;
  if (mNumberOfElementsBound2D) {
    mCoefficientBound2D0 = new UShort_t[mNumberOfElementsBound2D];
    mCoefficientBound2D1 = new UShort_t[mNumberOfElementsBound2D];
    for (int i = mNumberOfElementsBound2D; i--;) {
      mCoefficientBound2D0[i] = mCoefficientBound2D1[i] = 0;
    }
  }
}

void Chebyshev3DCalc::initializeCoefficients(int nc)
{
  if (mCoefficients) {
    delete[] mCoefficients;
    mCoefficients = nullptr;
  }
  mNumberOfCoefficients = nc;
  if (mNumberOfCoefficients) {
    mCoefficients = new Float_t[mNumberOfCoefficients];
    for (int i = mNumberOfCoefficients; i--;) {
      mCoefficients[i] = 0.0;
    }
  }
}

Float_t Chebyshev3DCalc::chebyshevEvaluation1Derivative(Float_t x, const Float_t* array, int ncf)
{
  if (--ncf < 1) {
    return 0;
  }
  Float_t b0, b1, b2;
  Float_t x2 = x + x;
  b1 = b2 = 0;
  float dcf0 = 0, dcf1, dcf2 = 0;
  b0 = dcf1 = 2 * ncf * array[ncf];
  if (!(--ncf)) {
    return b0 / 2;
  }

  for (int i = ncf; i--;) {
    b2 = b1;
    b1 = b0;
    dcf0 = dcf2 + 2 * (i + 1) * array[i + 1];
    b0 = dcf0 + x2 * b1 - b2;
    dcf2 = dcf1;
    dcf1 = dcf0;
  }

  return b0 - x * b1 - dcf0 / 2;
}

Float_t Chebyshev3DCalc::chebyshevEvaluation1Derivative2(Float_t x, const Float_t* array, int ncf)
{
  if (--ncf < 2) {
    return 0;
  }
  Float_t b0, b1, b2;
  Float_t x2 = x + x;
  b1 = b2 = 0;
  float dcf0 = 0, dcf1 = 0, dcf2 = 0;
  float ddcf0 = 0, ddcf1, ddcf2 = 0;

  dcf2 = 2 * ncf * array[ncf];
  --ncf;

  dcf1 = 2 * ncf * array[ncf];
  b0 = ddcf1 = 2 * ncf * dcf2;

  if (!(--ncf)) {
    return b0 / 2;
  }

  for (int i = ncf; i--;) {
    b2 = b1;
    b1 = b0;
    dcf0 = dcf2 + 2 * (i + 1) * array[i + 1];
    ddcf0 = ddcf2 + 2 * (i + 1) * dcf1;
    b0 = ddcf0 + x2 * b1 - b2;

    ddcf2 = ddcf1;
    ddcf1 = ddcf0;

    dcf2 = dcf1;
    dcf1 = dcf0;
  }
  return b0 - x * b1 - ddcf0 / 2;
}

Int_t Chebyshev3DCalc::getMaxColumnsAtRow() const
{
  int nmax3d = 0;
  for (int i = mNumberOfElementsBound2D; i--;) {
    if (mCoefficientBound2D0[i] > nmax3d) {
      nmax3d = mCoefficientBound2D0[i];
    }
  }
  return nmax3d;
}
