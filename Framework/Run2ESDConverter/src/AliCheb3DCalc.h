#ifndef ALICHEB3DCALC_H
#define ALICHEB3DCALC_H
/* Copyright(c) 1998-1999, ALICE Experiment at CERN, All rights reserved. *
 * See cxx source for full Copyright notice                               */
#include <TNamed.h>
class TSystem;
//
// Author: Ruben Shahoyan
// ruben.shahoyan@cern.ch   09/09/2006
// See Comments in AliCheb3D.h
//


// to decrease the compilable code size comment this define. This will exclude the routines 
// used for the calculation and saving of the coefficients. 
#define _INC_CREATION_ALICHEB3D_

// when _BRING_TO_BOUNDARY_ is defined, the point outside of the fitted folume is assumed
// to be on the surface 
// #define _BRING_TO_BOUNDARY_
//


class AliCheb3DCalc: public TNamed
{
 public:
  AliCheb3DCalc();
  AliCheb3DCalc(const AliCheb3DCalc& src);
  AliCheb3DCalc(FILE* stream);
  ~AliCheb3DCalc()                                                           {Clear();}
  //
  AliCheb3DCalc& operator=(const AliCheb3DCalc& rhs);
  void       Print(const Option_t* opt="")                              const;
  void       LoadData(FILE* stream);
  Float_t    EvalDeriv(int dim, const Float_t  *par)                    const;
  Float_t    EvalDeriv2(int dim1,int dim2, const Float_t  *par)         const;
  //
#ifdef _INC_CREATION_ALICHEB3D_
  void       SaveData(const char* outfile,Bool_t append=kFALSE)         const;
  void       SaveData(FILE* stream=stdout)                              const;
#endif
  //
  void       InitRows(int nr);
  void       InitCols(int nc);
  void       SetPrecision(Float_t prc=1e-6)                                   {fPrec = prc;}
  Float_t    GetPrecision()                                             const {return fPrec;}
  Int_t      GetNCoefs()                                                const {return fNCoefs;}
  Int_t      GetNCols()                                                 const {return (Int_t)fNCols;}
  Int_t      GetNRows()                                                 const {return (Int_t)fNRows;}
  Int_t      GetNElemBound2D()                                          const {return (Int_t)fNElemBound2D;}  
  Int_t      GetMaxColsAtRow()                                          const;
  UShort_t*  GetNColsAtRow()                                            const {return fNColsAtRow;}
  UShort_t*  GetColAtRowBg()                                            const {return fColAtRowBg;}
  void       InitElemBound2D(int ne);
  UShort_t*  GetCoefBound2D0()                                          const {return fCoefBound2D0;}
  UShort_t*  GetCoefBound2D1()                                          const {return fCoefBound2D1;}
  void       Clear(const Option_t* option = "");
  static Float_t    ChebEval1D(Float_t  x, const Float_t * array, int ncf);
  static Float_t    ChebEval1Deriv(Float_t  x, const Float_t * array, int ncf);
  static Float_t    ChebEval1Deriv2(Float_t  x, const Float_t * array, int ncf);
  void       InitCoefs(int nc);
  Float_t *  GetCoefs()                                                 const {return fCoefs;}
  //
  static void ReadLine(TString& str,FILE* stream);
  //
  Float_t    Eval(const Float_t  *par)                                  const;
  Double_t   Eval(const Double_t *par)                                  const;
  //
 protected:
  Int_t      fNCoefs;            // total number of coeeficients
  Int_t      fNRows;             // number of significant rows in the 3D coeffs matrix
  Int_t      fNCols;             // max number of significant cols in the 3D coeffs matrix
  Int_t      fNElemBound2D;      // number of elements (fNRows*fNCols) to store for the 2D boundary of significant coeffs
  UShort_t*  fNColsAtRow;        //[fNRows] number of sighificant columns (2nd dim) at each row of 3D coefs matrix
  UShort_t*  fColAtRowBg;        //[fNRows] beginnig of significant columns (2nd dim) for row in the 2D boundary matrix
  UShort_t*  fCoefBound2D0;      //[fNElemBound2D] 2D matrix defining the boundary of significance for 3D coeffs.matrix (Ncoefs for col/row)
  UShort_t*  fCoefBound2D1;      //[fNElemBound2D] 2D matrix defining the start beginnig of significant coeffs for col/row
  Float_t *  fCoefs;             //[fNCoefs] array of Chebyshev coefficients
  //
  Float_t *  fTmpCf1;            //[fNCols] temp. coeffs for 2d summation
  Float_t *  fTmpCf0;            //[fNRows] temp. coeffs for 1d summation
  //
  Float_t    fPrec;              // Requested precision
  ClassDef(AliCheb3DCalc,4)      // Class for interpolation of 3D->1 function by Chebyshev parametrization 
};

//__________________________________________________________________________________________
inline Float_t AliCheb3DCalc::ChebEval1D(Float_t  x, const Float_t * array, int ncf ) 
{
  // evaluate 1D Chebyshev parameterization. x is the argument mapped to [-1:1] interval
  if (ncf<=0) return 0;
  Float_t b0, b1, b2, x2 = x+x;
  b0 = array[--ncf]; 
  b1 = b2 = 0;
  for (int i=ncf;i--;) {
    b2 = b1;
    b1 = b0;
    b0 = array[i] + x2*b1 -b2;
  }
  return b0 - x*b1;
  //
}

//__________________________________________________________________________________________
inline Float_t AliCheb3DCalc::Eval(const Float_t  *par) const 
{
  // evaluate Chebyshev parameterization for 3D function.
  // VERY IMPORTANT: par must contain the function arguments ALREADY MAPPED to [-1:1] interval
  if (!fNRows) return 0.;
  int ncfRC;
  for (int id0=fNRows;id0--;) {
    int nCLoc = fNColsAtRow[id0];                   // number of significant coefs on this row
    int col0  = fColAtRowBg[id0];                   // beginning of local column in the 2D boundary matrix
    for (int id1=nCLoc;id1--;) {
      int id = id1+col0;
      fTmpCf1[id1] = (ncfRC=fCoefBound2D0[id]) ? ChebEval1D(par[2],fCoefs + fCoefBound2D1[id], ncfRC) : 0.0;
    }
    fTmpCf0[id0] = nCLoc>0 ? ChebEval1D(par[1],fTmpCf1,nCLoc):0.0;
  }
  return ChebEval1D(par[0],fTmpCf0,fNRows);
}

//__________________________________________________________________________________________
inline Double_t AliCheb3DCalc::Eval(const Double_t  *par) const 
{
  // evaluate Chebyshev parameterization for 3D function.
  // VERY IMPORTANT: par must contain the function arguments ALREADY MAPPED to [-1:1] interval
  if (!fNRows) return 0.;
  int ncfRC;
  for (int id0=fNRows;id0--;) {
    int nCLoc = fNColsAtRow[id0];                   // number of significant coefs on this row
    int col0  = fColAtRowBg[id0];                   // beginning of local column in the 2D boundary matrix
    for (int id1=nCLoc;id1--;) {
      int id = id1+col0;
      fTmpCf1[id1] = (ncfRC=fCoefBound2D0[id]) ? ChebEval1D(par[2],fCoefs + fCoefBound2D1[id], ncfRC) : 0.0;
    }
    fTmpCf0[id0] = nCLoc>0 ? ChebEval1D(par[1],fTmpCf1,nCLoc):0.0;
  }
  return ChebEval1D(par[0],fTmpCf0,fNRows);
}

#endif
