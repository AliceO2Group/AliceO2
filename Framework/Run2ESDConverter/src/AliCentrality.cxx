/**************************************************************************
 * Copyright(c) 1998-2008, ALICE Experiment at CERN, All rights reserved. *
 *                                                                        *
 * Author: The ALICE Off-line Project.                                    *
 * Contributors are mentioned in the code where appropriate.              *
 *                                                                        *
 * Permission to use, copy, modify and distribute this software and its   *
 * documentation strictly for non-commercial purposes is hereby granted   *
 * without fee, provided that the above copyright notice appears in all   *
 * copies and that both the copyright notice and this permission notice   *
 * appear in the supporting documentation. The authors make no claims     *
 * about the suitability of this software for any purpose. It is          *
 * provided "as is" without express or implied warranty.                  *
 **************************************************************************/

//*****************************************************
//   Class AliCentrality
//   author: Alberica Toia
//*****************************************************
/// A container for the centrality stored in AOD in ESD
 
#include "AliCentrality.h"

ClassImp(AliCentrality)

AliCentrality::AliCentrality() : TNamed("Centrality", "Centrality"),
  fQuality(999),
  fCentralityV0M(0),
  fCentralityV0A(0),
  fCentralityV0A0(0),
  fCentralityV0A123(0),
  fCentralityV0C(0),
  fCentralityV0A23(0),
  fCentralityV0C01(0),
  fCentralityV0S(0),
  fCentralityV0MEq(0),
  fCentralityV0AEq(0),
  fCentralityV0CEq(0),
  fCentralityFMD(0),
  fCentralityTRK(0),
  fCentralityTKL(0),
  fCentralityCL0(0),
  fCentralityCL1(0),
  fCentralityCND(0),
  fCentralityZNA(0),
  fCentralityZNC(0),
  fCentralityZPA(0),
  fCentralityZPC(0),
  fCentralityNPA(0),
  fCentralityV0MvsFMD(0),
  fCentralityTKLvsV0M(0),
  fCentralityZEMvsZDC(0),
  fCentralityV0Mtrue(0),
  fCentralityV0Atrue(0),
  fCentralityV0Ctrue(0),
  fCentralityV0MEqtrue(0),
  fCentralityV0AEqtrue(0),
  fCentralityV0CEqtrue(0),
  fCentralityFMDtrue(0),
  fCentralityTRKtrue(0),
  fCentralityTKLtrue(0),
  fCentralityCL0true(0),
  fCentralityCL1true(0),
  fCentralityCNDtrue(0),
  fCentralityZNAtrue(0),
  fCentralityZNCtrue(0),
  fCentralityZPAtrue(0),
  fCentralityZPCtrue(0)
{
  /// constructor
}

AliCentrality::AliCentrality(const AliCentrality& cnt) : 
  TNamed(cnt),
  fQuality(cnt.fQuality), 
  fCentralityV0M(cnt.fCentralityV0M),
  fCentralityV0A(cnt.fCentralityV0A),
  fCentralityV0A0(cnt.fCentralityV0A0),
  fCentralityV0A123(cnt.fCentralityV0A123),
  fCentralityV0C(cnt.fCentralityV0C),
  fCentralityV0A23(cnt.fCentralityV0A23),
  fCentralityV0C01(cnt.fCentralityV0C01),
  fCentralityV0S(cnt.fCentralityV0S),
  fCentralityV0MEq(cnt.fCentralityV0MEq),
  fCentralityV0AEq(cnt.fCentralityV0AEq),
  fCentralityV0CEq(cnt.fCentralityV0CEq),
  fCentralityFMD(cnt.fCentralityFMD),
  fCentralityTRK(cnt.fCentralityTRK),
  fCentralityTKL(cnt.fCentralityTKL),
  fCentralityCL0(cnt.fCentralityCL0),
  fCentralityCL1(cnt.fCentralityCL1),
  fCentralityCND(cnt.fCentralityCND),
  fCentralityZNA(cnt.fCentralityZNA),
  fCentralityZNC(cnt.fCentralityZNC),
  fCentralityZPA(cnt.fCentralityZPA),
  fCentralityZPC(cnt.fCentralityZPC),
  fCentralityNPA(cnt.fCentralityNPA),
  fCentralityV0MvsFMD(cnt.fCentralityV0MvsFMD),
  fCentralityTKLvsV0M(cnt.fCentralityTKLvsV0M),
  fCentralityZEMvsZDC(cnt.fCentralityZEMvsZDC),
  fCentralityV0Mtrue(cnt.fCentralityV0Mtrue),
  fCentralityV0Atrue(cnt.fCentralityV0Atrue),
  fCentralityV0Ctrue(cnt.fCentralityV0Ctrue),
  fCentralityV0MEqtrue(cnt.fCentralityV0MEqtrue),
  fCentralityV0AEqtrue(cnt.fCentralityV0AEqtrue),
  fCentralityV0CEqtrue(cnt.fCentralityV0CEqtrue),
  fCentralityFMDtrue(cnt.fCentralityFMDtrue),
  fCentralityTRKtrue(cnt.fCentralityTRKtrue),
  fCentralityTKLtrue(cnt.fCentralityTKLtrue),
  fCentralityCL0true(cnt.fCentralityCL0true),
  fCentralityCL1true(cnt.fCentralityCL1true),
  fCentralityCNDtrue(cnt.fCentralityCNDtrue),
  fCentralityZNAtrue(cnt.fCentralityZNAtrue),
  fCentralityZNCtrue(cnt.fCentralityZNCtrue),
  fCentralityZPAtrue(cnt.fCentralityZPAtrue),
  fCentralityZPCtrue(cnt.fCentralityZPCtrue)
{
  /// Copy constructor
}

AliCentrality& AliCentrality::operator=(const AliCentrality& c)
{
  /// Assignment operator
  if (this!=&c) {
    TNamed::operator=(c);
    fQuality = c.fQuality;
    fCentralityV0M = c.fCentralityV0M;
    fCentralityV0A = c.fCentralityV0A;
    fCentralityV0A0 = c.fCentralityV0A0;
    fCentralityV0A123 = c.fCentralityV0A123;
    fCentralityV0C = c.fCentralityV0C;
    fCentralityV0A23 = c.fCentralityV0A23;
    fCentralityV0C01 = c.fCentralityV0C01;
    fCentralityV0S = c.fCentralityV0S;
    fCentralityV0MEq = c.fCentralityV0MEq;
    fCentralityV0AEq = c.fCentralityV0AEq;
    fCentralityV0CEq = c.fCentralityV0CEq;
    fCentralityFMD = c.fCentralityFMD;
    fCentralityTRK = c.fCentralityTRK;
    fCentralityTKL = c.fCentralityTKL;
    fCentralityCL0 = c.fCentralityCL0;
    fCentralityCL1 = c.fCentralityCL1;
    fCentralityCND = c.fCentralityCND;
    fCentralityZNA = c.fCentralityZNA;
    fCentralityZNC = c.fCentralityZNC;
    fCentralityZPA = c.fCentralityZPA;
    fCentralityZPC = c.fCentralityZPC;
    fCentralityNPA = c.fCentralityNPA;
    fCentralityV0MvsFMD = c.fCentralityV0MvsFMD;
    fCentralityTKLvsV0M = c.fCentralityTKLvsV0M;
    fCentralityZEMvsZDC = c.fCentralityZEMvsZDC;
    fCentralityV0Mtrue = c.fCentralityV0Mtrue;
    fCentralityV0Atrue = c.fCentralityV0Atrue;
    fCentralityV0Ctrue = c.fCentralityV0Ctrue;
    fCentralityV0MEqtrue = c.fCentralityV0MEqtrue;
    fCentralityV0AEqtrue = c.fCentralityV0AEqtrue;
    fCentralityV0CEqtrue = c.fCentralityV0CEqtrue;
    fCentralityFMDtrue = c.fCentralityFMDtrue;
    fCentralityTRKtrue = c.fCentralityTRKtrue;
    fCentralityTKLtrue = c.fCentralityTKLtrue;
    fCentralityCL0true = c.fCentralityCL0true;
    fCentralityCL1true = c.fCentralityCL1true;
    fCentralityCNDtrue = c.fCentralityCNDtrue;
    fCentralityZNAtrue = c.fCentralityZNAtrue;
    fCentralityZNCtrue = c.fCentralityZNCtrue;
    fCentralityZPAtrue = c.fCentralityZPAtrue;
    fCentralityZPCtrue = c.fCentralityZPCtrue;
  }

  return *this;
}

AliCentrality::~AliCentrality()
{
  /// destructor
}

Int_t AliCentrality::GetQuality() const
{
  return fQuality;
}

Float_t AliCentrality::GetCentralityPercentile(const char *x) const
{
// Return the centrality percentile
  if (fQuality == 0) {
    TString method = x;
    if(method.CompareTo("V0M")==0)      return fCentralityV0M;
    if(method.CompareTo("V0A")==0)      return fCentralityV0A;
    if(method.CompareTo("V0A0")==0)   return fCentralityV0A0;
    if(method.CompareTo("V0A123")==0)   return fCentralityV0A123;
    if(method.CompareTo("V0C")==0)      return fCentralityV0C;
    if(method.CompareTo("V0A23")==0)    return fCentralityV0A23;
    if(method.CompareTo("V0C01")==0)    return fCentralityV0C01;
    if(method.CompareTo("V0S")==0)      return fCentralityV0S;
    if(method.CompareTo("V0MEq")==0)    return fCentralityV0MEq;
    if(method.CompareTo("V0AEq")==0)    return fCentralityV0AEq;
    if(method.CompareTo("V0CEq")==0)    return fCentralityV0CEq;
    if(method.CompareTo("FMD")==0)      return fCentralityFMD;
    if(method.CompareTo("TRK")==0)      return fCentralityTRK;
    if(method.CompareTo("TKL")==0)      return fCentralityTKL;
    if(method.CompareTo("CL0")==0)      return fCentralityCL0;
    if(method.CompareTo("CL1")==0)      return fCentralityCL1;
    if(method.CompareTo("CND")==0)      return fCentralityCND;
    if(method.CompareTo("ZNA")==0)      return fCentralityZNA;
    if(method.CompareTo("ZNC")==0)      return fCentralityZNC;
    if(method.CompareTo("ZPA")==0)      return fCentralityZPA;
    if(method.CompareTo("ZPC")==0)      return fCentralityZPC;
    if(method.CompareTo("NPA")==0)      return fCentralityNPA;
    if(method.CompareTo("V0MvsFMD")==0) return fCentralityV0MvsFMD;
    if(method.CompareTo("TKLvsV0M")==0) return fCentralityTKLvsV0M;
    if(method.CompareTo("ZEMvsZDC")==0) return fCentralityZEMvsZDC;
    if(method.CompareTo("V0Mtrue")==0)      return fCentralityV0Mtrue;
    if(method.CompareTo("V0Atrue")==0)      return fCentralityV0Atrue;
    if(method.CompareTo("V0Ctrue")==0)      return fCentralityV0Ctrue;
    if(method.CompareTo("V0MEqtrue")==0)    return fCentralityV0MEqtrue;
    if(method.CompareTo("V0AEqtrue")==0)    return fCentralityV0AEqtrue;
    if(method.CompareTo("V0CEqtrue")==0)    return fCentralityV0CEqtrue;
    if(method.CompareTo("FMDtrue")==0)      return fCentralityFMDtrue;
    if(method.CompareTo("TRKtrue")==0)      return fCentralityTRKtrue;
    if(method.CompareTo("TKLtrue")==0)      return fCentralityTKLtrue;
    if(method.CompareTo("CL0true")==0)      return fCentralityCL0true;
    if(method.CompareTo("CL1true")==0)      return fCentralityCL1true;
    if(method.CompareTo("CNDtrue")==0)      return fCentralityCNDtrue;
    if(method.CompareTo("ZNAtrue")==0)      return fCentralityZNAtrue;
    if(method.CompareTo("ZNCtrue")==0)      return fCentralityZNCtrue;
    if(method.CompareTo("ZPAtrue")==0)      return fCentralityZPAtrue;
    if(method.CompareTo("ZPCtrue")==0)      return fCentralityZPCtrue;
    return -1;
  } else {
    return -1;
  }
}

Int_t AliCentrality::GetCentralityClass10(const char *x) const
{
// Return the centrality class
  if (fQuality == 0) {
    TString method = x;
    if(method.CompareTo("V0M")==0)      return (Int_t) (fCentralityV0M / 10.0);
    if(method.CompareTo("V0A")==0)      return (Int_t) (fCentralityV0A / 10.0);
    if(method.CompareTo("V0A0")==0)     return (Int_t) (fCentralityV0A0 / 10.0);
    if(method.CompareTo("V0A123")==0)   return (Int_t) (fCentralityV0A123 / 10.0);
    if(method.CompareTo("V0C")==0)      return (Int_t) (fCentralityV0C / 10.0);
    if(method.CompareTo("V0A23")==0)    return (Int_t) (fCentralityV0A23 / 10.0);
    if(method.CompareTo("V0C01")==0)    return (Int_t) (fCentralityV0C01 / 10.0);
    if(method.CompareTo("V0S")==0)      return (Int_t) (fCentralityV0S / 10.0);
    if(method.CompareTo("V0MEq")==0)    return (Int_t) (fCentralityV0MEq / 10.0);
    if(method.CompareTo("V0AEq")==0)    return (Int_t) (fCentralityV0AEq / 10.0);
    if(method.CompareTo("V0CEq")==0)    return (Int_t) (fCentralityV0CEq / 10.0);
    if(method.CompareTo("FMD")==0)      return (Int_t) (fCentralityFMD / 10.0);
    if(method.CompareTo("TRK")==0)      return (Int_t) (fCentralityTRK / 10.0);
    if(method.CompareTo("TKL")==0)      return (Int_t) (fCentralityTKL / 10.0);
    if(method.CompareTo("CL0")==0)      return (Int_t) (fCentralityCL0 / 10.0);
    if(method.CompareTo("CL1")==0)      return (Int_t) (fCentralityCL1 / 10.0);
    if(method.CompareTo("CND")==0)      return (Int_t) (fCentralityCND / 10.0);
    if(method.CompareTo("ZNA")==0)      return (Int_t) (fCentralityZNA / 10.0);
    if(method.CompareTo("ZNC")==0)      return (Int_t) (fCentralityZNC / 10.0);
    if(method.CompareTo("ZPA")==0)      return (Int_t) (fCentralityZPA / 10.0);
    if(method.CompareTo("ZPC")==0)      return (Int_t) (fCentralityZPC / 10.0);
    if(method.CompareTo("NPA")==0)      return (Int_t) (fCentralityNPA / 10.0);
    if(method.CompareTo("V0MvsFMD")==0) return (Int_t) (fCentralityV0MvsFMD / 10.0);
    if(method.CompareTo("TKLvsV0M")==0) return (Int_t) (fCentralityTKLvsV0M / 10.0);
    if(method.CompareTo("ZEMvsZDC")==0) return (Int_t) (fCentralityZEMvsZDC / 10.0);
    if(method.CompareTo("V0Mtrue")==0)  return (Int_t) (fCentralityV0Mtrue / 10.0);
    if(method.CompareTo("V0Atrue")==0)  return (Int_t) (fCentralityV0Atrue / 10.0);
    if(method.CompareTo("V0Ctrue")==0)  return (Int_t) (fCentralityV0Ctrue / 10.0);
    if(method.CompareTo("V0MEqtrue")==0)return (Int_t) (fCentralityV0MEqtrue / 10.0);
    if(method.CompareTo("V0AEqtrue")==0)return (Int_t) (fCentralityV0AEqtrue / 10.0);
    if(method.CompareTo("V0CEqtrue")==0)return (Int_t) (fCentralityV0CEqtrue / 10.0);
    if(method.CompareTo("FMDtrue")==0)  return (Int_t) (fCentralityFMDtrue / 10.0);
    if(method.CompareTo("TRKtrue")==0)  return (Int_t) (fCentralityTRKtrue / 10.0);
    if(method.CompareTo("TKLtrue")==0)  return (Int_t) (fCentralityTKLtrue / 10.0);
    if(method.CompareTo("CL0true")==0)  return (Int_t) (fCentralityCL0true / 10.0);
    if(method.CompareTo("CL1true")==0)  return (Int_t) (fCentralityCL1true / 10.0);
    if(method.CompareTo("CNDtrue")==0)  return (Int_t) (fCentralityCNDtrue / 10.0);
    if(method.CompareTo("ZNAtrue")==0)  return (Int_t) (fCentralityZNAtrue / 10.0);
    if(method.CompareTo("ZNCtrue")==0)  return (Int_t) (fCentralityZNCtrue / 10.0);
    if(method.CompareTo("ZPAtrue")==0)  return (Int_t) (fCentralityZPAtrue / 10.0);
    if(method.CompareTo("ZPCtrue")==0)  return (Int_t) (fCentralityZPCtrue / 10.0);
    return -1;
  } else {
    return -1;
  }
}

Int_t AliCentrality::GetCentralityClass5(const char *x) const
{
// Return the centrality class
  if (fQuality == 0) {
    TString method = x;
    if(method.CompareTo("V0M")==0)      return (Int_t) (fCentralityV0M / 5.0);
    if(method.CompareTo("V0A")==0)      return (Int_t) (fCentralityV0A / 5.0);
    if(method.CompareTo("V0A0")==0)     return (Int_t) (fCentralityV0A0 / 5.0);
    if(method.CompareTo("V0A123")==0)   return (Int_t) (fCentralityV0A123 / 5.0);
    if(method.CompareTo("V0C")==0)      return (Int_t) (fCentralityV0C / 5.0);
    if(method.CompareTo("V0A23")==0)    return (Int_t) (fCentralityV0A23 / 5.0);
    if(method.CompareTo("V0C01")==0)    return (Int_t) (fCentralityV0C01 / 5.0);
    if(method.CompareTo("V0S")==0)      return (Int_t) (fCentralityV0S / 5.0);
    if(method.CompareTo("V0MEq")==0)    return (Int_t) (fCentralityV0MEq / 5.0);
    if(method.CompareTo("V0AEq")==0)    return (Int_t) (fCentralityV0AEq / 5.0);
    if(method.CompareTo("V0CEq")==0)    return (Int_t) (fCentralityV0CEq / 5.0);
    if(method.CompareTo("FMD")==0)      return (Int_t) (fCentralityFMD / 5.0);
    if(method.CompareTo("TRK")==0)      return (Int_t) (fCentralityTRK / 5.0);
    if(method.CompareTo("TKL")==0)      return (Int_t) (fCentralityTKL / 5.0);
    if(method.CompareTo("CL0")==0)      return (Int_t) (fCentralityCL0 / 5.0);
    if(method.CompareTo("CL1")==0)      return (Int_t) (fCentralityCL1 / 5.0);
    if(method.CompareTo("CND")==0)      return (Int_t) (fCentralityCND / 5.0);
    if(method.CompareTo("ZNA")==0)      return (Int_t) (fCentralityZNA / 5.0);
    if(method.CompareTo("ZNC")==0)      return (Int_t) (fCentralityZNC / 5.0);
    if(method.CompareTo("ZPA")==0)      return (Int_t) (fCentralityZPA / 5.0);
    if(method.CompareTo("ZPC")==0)      return (Int_t) (fCentralityZPC / 5.0);
    if(method.CompareTo("NPA")==0)      return (Int_t) (fCentralityNPA / 5.0);
    if(method.CompareTo("V0MvsFMD")==0) return (Int_t) (fCentralityV0MvsFMD / 5.0);
    if(method.CompareTo("TKLvsV0M")==0) return (Int_t) (fCentralityTKLvsV0M / 5.0);
    if(method.CompareTo("ZEMvsZDC")==0) return (Int_t) (fCentralityZEMvsZDC / 5.0);
    if(method.CompareTo("V0Mtrue")==0)  return (Int_t) (fCentralityV0Mtrue / 5.0);
    if(method.CompareTo("V0Atrue")==0)  return (Int_t) (fCentralityV0Atrue / 5.0);
    if(method.CompareTo("V0Ctrue")==0)  return (Int_t) (fCentralityV0Ctrue / 5.0);
    if(method.CompareTo("V0MEqtrue")==0)return (Int_t) (fCentralityV0MEqtrue / 5.0);
    if(method.CompareTo("V0AEqtrue")==0)return (Int_t) (fCentralityV0AEqtrue / 5.0);
    if(method.CompareTo("V0CEqtrue")==0)return (Int_t) (fCentralityV0CEqtrue / 5.0);
    if(method.CompareTo("FMDtrue")==0)  return (Int_t) (fCentralityFMDtrue / 5.0);
    if(method.CompareTo("TRKtrue")==0)  return (Int_t) (fCentralityTRKtrue / 5.0);
    if(method.CompareTo("TKLtrue")==0)  return (Int_t) (fCentralityTKLtrue / 5.0);
    if(method.CompareTo("CL0true")==0)  return (Int_t) (fCentralityCL0true / 5.0);
    if(method.CompareTo("CL1true")==0)  return (Int_t) (fCentralityCL1true / 5.0);
    if(method.CompareTo("CNDtrue")==0)  return (Int_t) (fCentralityCNDtrue / 5.0);
    if(method.CompareTo("ZNAtrue")==0)  return (Int_t) (fCentralityZNAtrue / 5.0);
    if(method.CompareTo("ZNCtrue")==0)  return (Int_t) (fCentralityZNCtrue / 5.0);
    if(method.CompareTo("ZPAtrue")==0)  return (Int_t) (fCentralityZPAtrue / 5.0);
    if(method.CompareTo("ZPCtrue")==0)  return (Int_t) (fCentralityZPCtrue / 5.0);
    return -1;
  } else {
    return -1;
  }
}


Bool_t AliCentrality::IsEventInCentralityClass(Float_t a, Float_t b, const char *x) const
{
// True if event is inside a given class
  if (fQuality == 0) {
    TString method = x;
    if ((method.CompareTo("V0M")==0) && (fCentralityV0M >=a && fCentralityV0M < b)) return kTRUE;
    if ((method.CompareTo("V0A")==0) && (fCentralityV0A >=a && fCentralityV0A < b)) return kTRUE;
    if ((method.CompareTo("V0A0")==0) && (fCentralityV0A0 >=a && fCentralityV0A0 < b)) return kTRUE;
    if ((method.CompareTo("V0A123")==0) && (fCentralityV0A123 >=a && fCentralityV0A123 < b)) return kTRUE;
    if ((method.CompareTo("V0C")==0) && (fCentralityV0C >=a && fCentralityV0C < b)) return kTRUE;
    if ((method.CompareTo("V0A23")==0) && (fCentralityV0A23 >=a && fCentralityV0A23 < b)) return kTRUE;
    if ((method.CompareTo("V0C01")==0) && (fCentralityV0C01 >=a && fCentralityV0C01 < b)) return kTRUE;
    if ((method.CompareTo("V0S")==0) && (fCentralityV0S >=a && fCentralityV0S < b)) return kTRUE;
    if ((method.CompareTo("V0MEq")==0) && (fCentralityV0MEq >=a && fCentralityV0MEq < b)) return kTRUE;
    if ((method.CompareTo("V0AEq")==0) && (fCentralityV0AEq >=a && fCentralityV0AEq < b)) return kTRUE;
    if ((method.CompareTo("V0CEq")==0) && (fCentralityV0CEq >=a && fCentralityV0CEq < b)) return kTRUE;
    if ((method.CompareTo("FMD")==0) && (fCentralityFMD >=a && fCentralityFMD < b)) return kTRUE;
    if ((method.CompareTo("TRK")==0) && (fCentralityTRK >=a && fCentralityTRK < b)) return kTRUE;
    if ((method.CompareTo("TKL")==0) && (fCentralityTKL >=a && fCentralityTKL < b)) return kTRUE;
    if ((method.CompareTo("CL0")==0) && (fCentralityCL0 >=a && fCentralityCL0 < b)) return kTRUE;
    if ((method.CompareTo("CL1")==0) && (fCentralityCL1 >=a && fCentralityCL1 < b)) return kTRUE;
    if ((method.CompareTo("CND")==0) && (fCentralityCND >=a && fCentralityCND < b)) return kTRUE;
    if ((method.CompareTo("ZNA")==0) && (fCentralityZNA >=a && fCentralityZNA < b)) return kTRUE;
    if ((method.CompareTo("ZNC")==0) && (fCentralityZNC >=a && fCentralityZNC < b)) return kTRUE;
    if ((method.CompareTo("ZPA")==0) && (fCentralityZPA >=a && fCentralityZPA < b)) return kTRUE;
    if ((method.CompareTo("ZPC")==0) && (fCentralityZPC >=a && fCentralityZPC < b)) return kTRUE;
    if ((method.CompareTo("NPA")==0) && (fCentralityNPA >=a && fCentralityNPA < b)) return kTRUE;
    if ((method.CompareTo("V0MvsFMD")==0) && (fCentralityV0MvsFMD >=a && fCentralityV0MvsFMD < b)) return kTRUE;
    if ((method.CompareTo("TKLvsV0M")==0) && (fCentralityTKLvsV0M >=a && fCentralityTKLvsV0M < b)) return kTRUE;
    if ((method.CompareTo("ZEMvsZDC")==0) && (fCentralityZEMvsZDC >=a && fCentralityZEMvsZDC < b)) return kTRUE;
    if ((method.CompareTo("V0Mtrue")==0) && (fCentralityV0Mtrue >=a && fCentralityV0Mtrue < b)) return kTRUE;
    if ((method.CompareTo("V0Atrue")==0) && (fCentralityV0Atrue >=a && fCentralityV0Atrue < b)) return kTRUE;
    if ((method.CompareTo("V0Ctrue")==0) && (fCentralityV0Ctrue >=a && fCentralityV0Ctrue < b)) return kTRUE;
    if ((method.CompareTo("V0MEqtrue")==0) && (fCentralityV0MEqtrue >=a && fCentralityV0MEqtrue < b)) return kTRUE;
    if ((method.CompareTo("V0AEqtrue")==0) && (fCentralityV0AEqtrue >=a && fCentralityV0AEqtrue < b)) return kTRUE;
    if ((method.CompareTo("V0CEqtrue")==0) && (fCentralityV0CEqtrue >=a && fCentralityV0CEqtrue < b)) return kTRUE;
    if ((method.CompareTo("FMDtrue")==0) && (fCentralityFMDtrue >=a && fCentralityFMDtrue < b)) return kTRUE;
    if ((method.CompareTo("TRKtrue")==0) && (fCentralityTRKtrue >=a && fCentralityTRKtrue < b)) return kTRUE;
    if ((method.CompareTo("TKLtrue")==0) && (fCentralityTKLtrue >=a && fCentralityTKLtrue < b)) return kTRUE;
    if ((method.CompareTo("CL0true")==0) && (fCentralityCL0true >=a && fCentralityCL0true < b)) return kTRUE;
    if ((method.CompareTo("CL1true")==0) && (fCentralityCL1true >=a && fCentralityCL1true < b)) return kTRUE;
    if ((method.CompareTo("CNDtrue")==0) && (fCentralityCNDtrue >=a && fCentralityCNDtrue < b)) return kTRUE;
    if ((method.CompareTo("ZNAtrue")==0) && (fCentralityZNAtrue >=a && fCentralityZNAtrue < b)) return kTRUE;
    if ((method.CompareTo("ZNCtrue")==0) && (fCentralityZNCtrue >=a && fCentralityZNCtrue < b)) return kTRUE;
    if ((method.CompareTo("ZPAtrue")==0) && (fCentralityZPAtrue >=a && fCentralityZPAtrue < b)) return kTRUE;
    if ((method.CompareTo("ZPCtrue")==0) && (fCentralityZPCtrue >=a && fCentralityZPCtrue < b)) return kTRUE;
    else return kFALSE;
  } else {
    return kFALSE;
  }
}

Float_t AliCentrality::GetCentralityPercentileUnchecked(const char *x) const
{
// Return the centrality percentile
  TString method = x;
  if(method.CompareTo("V0M")==0)      return fCentralityV0M;
  if(method.CompareTo("V0A")==0)      return fCentralityV0A;
  if(method.CompareTo("V0A0")==0)     return fCentralityV0A0;
  if(method.CompareTo("V0A123")==0)   return fCentralityV0A123;
  if(method.CompareTo("V0C")==0)      return fCentralityV0C;
  if(method.CompareTo("V0A23")==0)    return fCentralityV0A23;
  if(method.CompareTo("V0C01")==0)    return fCentralityV0C01;
  if(method.CompareTo("V0S")==0)      return fCentralityV0S;
  if(method.CompareTo("V0MEq")==0)    return fCentralityV0MEq;
  if(method.CompareTo("V0AEq")==0)    return fCentralityV0AEq;
  if(method.CompareTo("V0CEq")==0)    return fCentralityV0CEq;
  if(method.CompareTo("FMD")==0)      return fCentralityFMD;
  if(method.CompareTo("TRK")==0)      return fCentralityTRK;
  if(method.CompareTo("TKL")==0)      return fCentralityTKL;
  if(method.CompareTo("CL0")==0)      return fCentralityCL0;
  if(method.CompareTo("CL1")==0)      return fCentralityCL1;
  if(method.CompareTo("CND")==0)      return fCentralityCND;
  if(method.CompareTo("ZNA")==0)      return fCentralityZNA;
  if(method.CompareTo("ZNC")==0)      return fCentralityZNC;
  if(method.CompareTo("ZPA")==0)      return fCentralityZPA;
  if(method.CompareTo("ZPC")==0)      return fCentralityZPC;
  if(method.CompareTo("NPA")==0)      return fCentralityNPA;
  if(method.CompareTo("V0MvsFMD")==0) return fCentralityV0MvsFMD;
  if(method.CompareTo("TKLvsV0M")==0) return fCentralityTKLvsV0M;
  if(method.CompareTo("ZEMvsZDC")==0) return fCentralityZEMvsZDC;
  if(method.CompareTo("V0Mtrue")==0)  return fCentralityV0Mtrue;
  if(method.CompareTo("V0Atrue")==0)  return fCentralityV0Atrue;
  if(method.CompareTo("V0Ctrue")==0)  return fCentralityV0Ctrue;
  if(method.CompareTo("V0MEqtrue")==0)    return fCentralityV0MEqtrue;
  if(method.CompareTo("V0AEqtrue")==0)    return fCentralityV0AEqtrue;
  if(method.CompareTo("V0CEqtrue")==0)    return fCentralityV0CEqtrue;
  if(method.CompareTo("FMDtrue")==0)  return fCentralityFMDtrue;
  if(method.CompareTo("TRKtrue")==0)  return fCentralityTRKtrue;
  if(method.CompareTo("TKLtrue")==0)  return fCentralityTKLtrue;
  if(method.CompareTo("CL0true")==0)  return fCentralityCL0true;
  if(method.CompareTo("CL1true")==0)  return fCentralityCL1true;
  if(method.CompareTo("CNDtrue")==0)  return fCentralityCNDtrue;
  if(method.CompareTo("ZNAtrue")==0)  return fCentralityZNAtrue;
  if(method.CompareTo("ZNCtrue")==0)  return fCentralityZNCtrue;
  if(method.CompareTo("ZPAtrue")==0)  return fCentralityZPAtrue;
  if(method.CompareTo("ZPCtrue")==0)  return fCentralityZPCtrue;
  return -1;
}

Int_t AliCentrality::GetCentralityClass10Unchecked(const char *x) const
{
// Return the centrality class
  TString method = x;
    if(method.CompareTo("V0M")==0)      return (Int_t) (fCentralityV0M / 10.0);
    if(method.CompareTo("V0A")==0)      return (Int_t) (fCentralityV0A / 10.0);
    if(method.CompareTo("V0A0")==0)     return (Int_t) (fCentralityV0A0 / 10.0);
    if(method.CompareTo("V0C")==0)      return (Int_t) (fCentralityV0C / 10.0);
    if(method.CompareTo("V0A23")==0)    return (Int_t) (fCentralityV0A23 / 10.0);
    if(method.CompareTo("V0C01")==0)    return (Int_t) (fCentralityV0C01 / 10.0);
    if(method.CompareTo("V0S")==0)      return (Int_t) (fCentralityV0S / 10.0);
    if(method.CompareTo("V0MEq")==0)    return (Int_t) (fCentralityV0MEq / 10.0);
    if(method.CompareTo("V0AEq")==0)    return (Int_t) (fCentralityV0AEq / 10.0);
    if(method.CompareTo("V0CEq")==0)    return (Int_t) (fCentralityV0CEq / 10.0);
    if(method.CompareTo("FMD")==0)      return (Int_t) (fCentralityFMD / 10.0);
    if(method.CompareTo("TRK")==0)      return (Int_t) (fCentralityTRK / 10.0);
    if(method.CompareTo("TKL")==0)      return (Int_t) (fCentralityTKL / 10.0);
    if(method.CompareTo("CL0")==0)      return (Int_t) (fCentralityCL0 / 10.0);
    if(method.CompareTo("CL1")==0)      return (Int_t) (fCentralityCL1 / 10.0);
    if(method.CompareTo("CND")==0)      return (Int_t) (fCentralityCND / 10.0);
    if(method.CompareTo("ZNA")==0)      return (Int_t) (fCentralityZNA / 10.0);
    if(method.CompareTo("ZNC")==0)      return (Int_t) (fCentralityZNC / 10.0);
    if(method.CompareTo("ZPA")==0)      return (Int_t) (fCentralityZPA / 10.0);
    if(method.CompareTo("ZPC")==0)      return (Int_t) (fCentralityZPC / 10.0);
    if(method.CompareTo("NPA")==0)      return (Int_t) (fCentralityNPA / 10.0);
    if(method.CompareTo("V0MvsFMD")==0) return (Int_t) (fCentralityV0MvsFMD / 10.0);
    if(method.CompareTo("TKLvsV0M")==0) return (Int_t) (fCentralityTKLvsV0M / 10.0);
    if(method.CompareTo("ZEMvsZDC")==0) return (Int_t) (fCentralityZEMvsZDC / 10.0);
    if(method.CompareTo("V0Mtrue")==0)  return (Int_t) (fCentralityV0Mtrue / 10.0);
    if(method.CompareTo("V0Atrue")==0)  return (Int_t) (fCentralityV0Atrue / 10.0);
    if(method.CompareTo("V0Ctrue")==0)  return (Int_t) (fCentralityV0Ctrue / 10.0);
    if(method.CompareTo("V0MEqtrue")==0)return (Int_t) (fCentralityV0MEqtrue / 10.0);
    if(method.CompareTo("V0AEqtrue")==0)return (Int_t) (fCentralityV0AEqtrue / 10.0);
    if(method.CompareTo("V0CEqtrue")==0)return (Int_t) (fCentralityV0CEqtrue / 10.0);
    if(method.CompareTo("FMDtrue")==0)  return (Int_t) (fCentralityFMDtrue / 10.0);
    if(method.CompareTo("TRKtrue")==0)  return (Int_t) (fCentralityTRKtrue / 10.0);
    if(method.CompareTo("TKLtrue")==0)  return (Int_t) (fCentralityTKLtrue / 10.0);
    if(method.CompareTo("CL0true")==0)  return (Int_t) (fCentralityCL0true / 10.0);
    if(method.CompareTo("CL1true")==0)  return (Int_t) (fCentralityCL1true / 10.0);
    if(method.CompareTo("CNDtrue")==0)  return (Int_t) (fCentralityCNDtrue / 10.0);
    if(method.CompareTo("ZNAtrue")==0)  return (Int_t) (fCentralityZNAtrue / 10.0);
    if(method.CompareTo("ZNCtrue")==0)  return (Int_t) (fCentralityZNCtrue / 10.0);
    if(method.CompareTo("ZPAtrue")==0)  return (Int_t) (fCentralityZPAtrue / 10.0);
    if(method.CompareTo("ZPCtrue")==0)  return (Int_t) (fCentralityZPCtrue / 10.0);
  return -1;
}

Int_t AliCentrality::GetCentralityClass5Unchecked(const char *x) const
{
// Return the centrality class
  TString method = x;
    if(method.CompareTo("V0M")==0)      return (Int_t) (fCentralityV0M / 5.0);
    if(method.CompareTo("V0A")==0)      return (Int_t) (fCentralityV0A / 5.0);
    if(method.CompareTo("V0A0")==0)     return (Int_t) (fCentralityV0A0 / 5.0);
    if(method.CompareTo("V0A123")==0)   return (Int_t) (fCentralityV0A123 / 5.0);
    if(method.CompareTo("V0C")==0)      return (Int_t) (fCentralityV0C / 5.0);
    if(method.CompareTo("V0A23")==0)    return (Int_t) (fCentralityV0A23 / 5.0);
    if(method.CompareTo("V0C01")==0)    return (Int_t) (fCentralityV0C01 / 5.0);
    if(method.CompareTo("V0S")==0)      return (Int_t) (fCentralityV0S / 5.0);
    if(method.CompareTo("V0MEq")==0)    return (Int_t) (fCentralityV0MEq / 5.0);
    if(method.CompareTo("V0AEq")==0)    return (Int_t) (fCentralityV0AEq / 5.0);
    if(method.CompareTo("V0CEq")==0)    return (Int_t) (fCentralityV0CEq / 5.0);
    if(method.CompareTo("FMD")==0)      return (Int_t) (fCentralityFMD / 5.0);
    if(method.CompareTo("TRK")==0)      return (Int_t) (fCentralityTRK / 5.0);
    if(method.CompareTo("TKL")==0)      return (Int_t) (fCentralityTKL / 5.0);
    if(method.CompareTo("CL0")==0)      return (Int_t) (fCentralityCL0 / 5.0);
    if(method.CompareTo("CL1")==0)      return (Int_t) (fCentralityCL1 / 5.0);
    if(method.CompareTo("CND")==0)      return (Int_t) (fCentralityCND / 5.0);
    if(method.CompareTo("ZNA")==0)      return (Int_t) (fCentralityZNA / 5.0);
    if(method.CompareTo("ZNC")==0)      return (Int_t) (fCentralityZNC / 5.0);
    if(method.CompareTo("ZPA")==0)      return (Int_t) (fCentralityZPA / 5.0);
    if(method.CompareTo("ZPC")==0)      return (Int_t) (fCentralityZPC / 5.0);
    if(method.CompareTo("NPA")==0)      return (Int_t) (fCentralityNPA / 5.0);
    if(method.CompareTo("V0MvsFMD")==0) return (Int_t) (fCentralityV0MvsFMD / 5.0);
    if(method.CompareTo("TKLvsV0M")==0) return (Int_t) (fCentralityTKLvsV0M / 5.0);
    if(method.CompareTo("ZEMvsZDC")==0) return (Int_t) (fCentralityZEMvsZDC / 5.0);
    if(method.CompareTo("V0Mtrue")==0)  return (Int_t) (fCentralityV0Mtrue / 5.0);
    if(method.CompareTo("V0Atrue")==0)  return (Int_t) (fCentralityV0Atrue / 5.0);
    if(method.CompareTo("V0Ctrue")==0)  return (Int_t) (fCentralityV0Ctrue / 5.0);
    if(method.CompareTo("V0MEqtrue")==0)return (Int_t) (fCentralityV0MEqtrue / 5.0);
    if(method.CompareTo("V0AEqtrue")==0)return (Int_t) (fCentralityV0AEqtrue / 5.0);
    if(method.CompareTo("V0CEqtrue")==0)return (Int_t) (fCentralityV0CEqtrue / 5.0);
    if(method.CompareTo("FMDtrue")==0)  return (Int_t) (fCentralityFMDtrue / 5.0);
    if(method.CompareTo("TRKtrue")==0)  return (Int_t) (fCentralityTRKtrue / 5.0);
    if(method.CompareTo("TKLtrue")==0)  return (Int_t) (fCentralityTKLtrue / 5.0);
    if(method.CompareTo("CL0true")==0)  return (Int_t) (fCentralityCL0true / 5.0);
    if(method.CompareTo("CL1true")==0)  return (Int_t) (fCentralityCL1true / 5.0);
    if(method.CompareTo("CNDtrue")==0)  return (Int_t) (fCentralityCNDtrue / 5.0);
    if(method.CompareTo("ZNAtrue")==0)  return (Int_t) (fCentralityZNAtrue / 5.0);
    if(method.CompareTo("ZNCtrue")==0)  return (Int_t) (fCentralityZNCtrue / 5.0);
    if(method.CompareTo("ZPAtrue")==0)  return (Int_t) (fCentralityZPAtrue / 5.0);
    if(method.CompareTo("ZPCtrue")==0)  return (Int_t) (fCentralityZPCtrue / 5.0);
  return -1;
} 

Bool_t AliCentrality::IsEventInCentralityClassUnchecked(Float_t a, Float_t b, const char *x) const
{
// True if event inside given centrality class
  TString method = x;
    if ((method.CompareTo("V0M")==0) && (fCentralityV0M >=a && fCentralityV0M < b)) return kTRUE;
    if ((method.CompareTo("V0A")==0) && (fCentralityV0A >=a && fCentralityV0A < b)) return kTRUE;
    if ((method.CompareTo("V0A0")==0) && (fCentralityV0A0 >=a && fCentralityV0A0 < b)) return kTRUE;
    if ((method.CompareTo("V0A123")==0) && (fCentralityV0A123 >=a && fCentralityV0A123 < b)) return kTRUE;
    if ((method.CompareTo("V0C")==0) && (fCentralityV0C >=a && fCentralityV0C < b)) return kTRUE;
    if ((method.CompareTo("V0A23")==0) && (fCentralityV0A23 >=a && fCentralityV0A23 < b)) return kTRUE;
    if ((method.CompareTo("V0C01")==0) && (fCentralityV0C01 >=a && fCentralityV0C01 < b)) return kTRUE;
    if ((method.CompareTo("V0S")==0) && (fCentralityV0S >=a && fCentralityV0S < b)) return kTRUE;
    if ((method.CompareTo("V0MEq")==0) && (fCentralityV0MEq >=a && fCentralityV0MEq < b)) return kTRUE;
    if ((method.CompareTo("V0AEq")==0) && (fCentralityV0AEq >=a && fCentralityV0AEq < b)) return kTRUE;
    if ((method.CompareTo("V0CEq")==0) && (fCentralityV0CEq >=a && fCentralityV0CEq < b)) return kTRUE;
    if ((method.CompareTo("FMD")==0) && (fCentralityFMD >=a && fCentralityFMD < b)) return kTRUE;
    if ((method.CompareTo("TRK")==0) && (fCentralityTRK >=a && fCentralityTRK < b)) return kTRUE;
    if ((method.CompareTo("TKL")==0) && (fCentralityTKL >=a && fCentralityTKL < b)) return kTRUE;
    if ((method.CompareTo("CL0")==0) && (fCentralityCL0 >=a && fCentralityCL0 < b)) return kTRUE;
    if ((method.CompareTo("CL1")==0) && (fCentralityCL1 >=a && fCentralityCL1 < b)) return kTRUE;
    if ((method.CompareTo("CND")==0) && (fCentralityCND >=a && fCentralityCND < b)) return kTRUE;
    if ((method.CompareTo("ZNA")==0) && (fCentralityZNA >=a && fCentralityZNA < b)) return kTRUE;
    if ((method.CompareTo("ZNC")==0) && (fCentralityZNC >=a && fCentralityZNC < b)) return kTRUE;
    if ((method.CompareTo("ZPA")==0) && (fCentralityZPA >=a && fCentralityZPA < b)) return kTRUE;
    if ((method.CompareTo("ZPC")==0) && (fCentralityZPC >=a && fCentralityZPC < b)) return kTRUE;
    if ((method.CompareTo("NPA")==0) && (fCentralityNPA >=a && fCentralityNPA < b)) return kTRUE;
    if ((method.CompareTo("V0MvsFMD")==0) && (fCentralityV0MvsFMD >=a && fCentralityV0MvsFMD < b)) return kTRUE;
    if ((method.CompareTo("TKLvsV0M")==0) && (fCentralityTKLvsV0M >=a && fCentralityTKLvsV0M < b)) return kTRUE;
    if ((method.CompareTo("ZEMvsZDC")==0) && (fCentralityZEMvsZDC >=a && fCentralityZEMvsZDC < b)) return kTRUE;
    if ((method.CompareTo("V0Mtrue")==0) && (fCentralityV0Mtrue >=a && fCentralityV0Mtrue < b)) return kTRUE;
    if ((method.CompareTo("V0Atrue")==0) && (fCentralityV0Atrue >=a && fCentralityV0Atrue < b)) return kTRUE;
    if ((method.CompareTo("V0Ctrue")==0) && (fCentralityV0Ctrue >=a && fCentralityV0Ctrue < b)) return kTRUE;
    if ((method.CompareTo("V0MEqtrue")==0) && (fCentralityV0MEqtrue >=a && fCentralityV0MEqtrue < b)) return kTRUE;
    if ((method.CompareTo("V0AEqtrue")==0) && (fCentralityV0AEqtrue >=a && fCentralityV0AEqtrue < b)) return kTRUE;
    if ((method.CompareTo("V0CEqtrue")==0) && (fCentralityV0CEqtrue >=a && fCentralityV0CEqtrue < b)) return kTRUE;
    if ((method.CompareTo("FMDtrue")==0) && (fCentralityFMDtrue >=a && fCentralityFMDtrue < b)) return kTRUE;
    if ((method.CompareTo("TRKtrue")==0) && (fCentralityTRKtrue >=a && fCentralityTRKtrue < b)) return kTRUE;
    if ((method.CompareTo("TKLtrue")==0) && (fCentralityTKLtrue >=a && fCentralityTKLtrue < b)) return kTRUE;
    if ((method.CompareTo("CL0true")==0) && (fCentralityCL0true >=a && fCentralityCL0true < b)) return kTRUE;
    if ((method.CompareTo("CL1true")==0) && (fCentralityCL1true >=a && fCentralityCL1true < b)) return kTRUE;
    if ((method.CompareTo("CNDtrue")==0) && (fCentralityCNDtrue >=a && fCentralityCNDtrue < b)) return kTRUE;
    if ((method.CompareTo("ZNAtrue")==0) && (fCentralityZNAtrue >=a && fCentralityZNAtrue < b)) return kTRUE;
    if ((method.CompareTo("ZNCtrue")==0) && (fCentralityZNCtrue >=a && fCentralityZNCtrue < b)) return kTRUE;
    if ((method.CompareTo("ZPAtrue")==0) && (fCentralityZPAtrue >=a && fCentralityZPAtrue < b)) return kTRUE;
    if ((method.CompareTo("ZPCtrue")==0) && (fCentralityZPCtrue >=a && fCentralityZPCtrue < b)) return kTRUE;
  else return kFALSE;
} 

void AliCentrality::Reset()
{
// Reset.

  fQuality            =  999;
  fCentralityV0M      =  0;
  fCentralityV0A      =  0;
  fCentralityV0A0     =  0;
  fCentralityV0A123   =  0;
  fCentralityV0C      =  0;
  fCentralityV0A23    =  0;
  fCentralityV0C01    =  0;
  fCentralityV0S      =  0;
  fCentralityV0MEq    =  0;
  fCentralityV0AEq    =  0;
  fCentralityV0CEq    =  0;
  fCentralityFMD      =  0;
  fCentralityTRK      =  0;
  fCentralityTKL      =  0;
  fCentralityCL0      =  0;
  fCentralityCL1      =  0;
  fCentralityCND      =  0;
  fCentralityZNA      =  0;
  fCentralityZNC      =  0;
  fCentralityZPA      =  0;
  fCentralityZPC      =  0;
  fCentralityNPA      =  0;
  fCentralityV0MvsFMD =  0;
  fCentralityTKLvsV0M =  0;
  fCentralityZEMvsZDC =  0;
  fCentralityV0Mtrue  =  0;
  fCentralityV0Atrue  =  0;
  fCentralityV0Ctrue  =  0;
  fCentralityV0MEqtrue  =  0;
  fCentralityV0AEqtrue  =  0;
  fCentralityV0CEqtrue  =  0;
  fCentralityFMDtrue  =  0;
  fCentralityTRKtrue  =  0;
  fCentralityTKLtrue  =  0;
  fCentralityCL0true  =  0;
  fCentralityCL1true  =  0;
  fCentralityCNDtrue  =  0;
  fCentralityZNAtrue  =  0;
  fCentralityZNCtrue  =  0;
  fCentralityZPAtrue  =  0;
  fCentralityZPCtrue  =  0;
}
