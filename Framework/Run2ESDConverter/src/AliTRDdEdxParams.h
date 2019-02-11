/**************************************************************************
 * Copyright(c) 1998-1999, ALICE Experiment at CERN, All rights reserved. *
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
//
//  Xianguo Lu <lu@physi.uni-heidelberg.de>
//

//
// modified 10/08/15 by Lucas Altenkaemper <altenkaemper@physi.uni-heidelberg.de>
//

#ifndef ALITRDDEDXPARAMS_H
#define ALITRDDEDXPARAMS_H

#include "TNamed.h"
#include "TVectorT.h"
#include "TString.h"

//maximum number of array size
#define MAXSIZE 160

class AliTRDdEdxParams: public TNamed
{
 public:
  AliTRDdEdxParams(const TString name="name", const TString title="title");
  void Print(Option_t* option = "") const;

  const TVectorF& GetMeanParameter( const Int_t itype, const Int_t nch, const Int_t ncls, const Bool_t etaCorrection) const { return GetParameter(fMeanPar,  itype, nch, ncls, etaCorrection);}
  const TVectorF& GetSigmaParameter(const Int_t itype, const Int_t nch, const Int_t ncls, const Bool_t etaCorrection) const { return GetParameter(fSigmaPar, itype, nch, ncls, etaCorrection);}

  void SetMeanParameter( const Int_t itype, const Int_t nch, const Int_t ncls, const Int_t npar, const Float_t vals[], const Bool_t etaCorrection){ SetParameter(fMeanPar,  itype, nch, ncls, npar, vals, etaCorrection); }
  void SetSigmaParameter(const Int_t itype, const Int_t nch, const Int_t ncls, const Int_t npar, const Float_t vals[], const Bool_t etaCorrection){ SetParameter(fSigmaPar, itype, nch, ncls, npar, vals, etaCorrection); }

 private:
  const TVectorF& GetParameter(const TVectorF par[], const Int_t itype, const Int_t nch, const Int_t ncls, const Bool_t etaCorrection) const;
  void SetParameter(TVectorF par[], const Int_t itype, const Int_t nch, const Int_t ncls, const Int_t npar, const Float_t vals[], const Bool_t etaCorrection);

  TVectorF fMeanPar[MAXSIZE];
  TVectorF fSigmaPar[MAXSIZE];

  Int_t GetIter(const Int_t itype, const Int_t nch, const Int_t ncls, const Bool_t etaCorrection) const;

  ClassDef(AliTRDdEdxParams,3);
};

#endif
