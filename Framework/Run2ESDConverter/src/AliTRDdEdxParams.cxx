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


#include "AliLog.h"
#include "AliTRDdEdxParams.h"

ClassImp(AliTRDdEdxParams);

AliTRDdEdxParams::AliTRDdEdxParams(const TString name, const TString title): TNamed(name,title)
{
  //
  //constructor
  //
}

Int_t AliTRDdEdxParams::GetIter(const Int_t itype, const Int_t nch, const Int_t ncls, const Bool_t etaCorrection) const
{
  //
  //return array iterator
  //

    Int_t itNch = -999, itNcls = -999, itEtaCorr = -999;

    //hard coded cuts       // <4, 4, 5 or 6 layer
    if (nch == 6) {
        itNch = 0;
    } else if (nch == 5) {
        itNch = 1;
    } else if (nch == 4) {
        itNch = 2;
    } else if (nch < 4){
        itNch = 3;
    }

    if (nch != 0 && ncls/nch >= 17) {       // QA cut minimum ncls
        itNcls = 0;
    }
    else {
        itNcls = 1;
    }
    
    if (etaCorrection) {                // eta correction
        itEtaCorr = 0;
    } else {
        itEtaCorr = 1;
    }
        
    const Int_t finaliter = itEtaCorr*80 + itNcls*40 + itNch*10 + itype;
    
    if (finaliter < 0 || finaliter >= MAXSIZE) {
        AliError(Form("out of range itype %d, nch %d, ncls %d\n", itype, nch, ncls));
    }

    return finaliter;
}

const TVectorF& AliTRDdEdxParams::GetParameter(const TVectorF par[], const Int_t itype, const Int_t nch, const Int_t ncls, const Bool_t etaCorrection)const
{
  //
  //return parameter for particle itype from par[]
  //

    const Int_t iter = GetIter(itype, nch, ncls, etaCorrection);

    return par[iter];
}

void AliTRDdEdxParams::SetParameter(TVectorF par[], const Int_t itype, const Int_t nch, const Int_t ncls, const Int_t npar, const Float_t vals[], const Bool_t etaCorrection)
{
  //
  //set parameter, vals of dimension npar, for particle itype
  //

    const Int_t iter = GetIter(itype, nch, ncls, etaCorrection);

  TVectorF p2(npar, vals);

  par[iter].ResizeTo(p2);
  par[iter] = p2;
}

void AliTRDdEdxParams::Print(Option_t* option) const
{
  //
  //print all members
  //

  TObject::Print(option);

    printf("\n======================= Mean ========================\n");
    for(Int_t ii = 0; ii < MAXSIZE; ii++){
        printf("%d: Nrows() %d\n",ii, fMeanPar[ii].GetNrows());
        if(fMeanPar[ii].GetNrows()) fMeanPar[ii].Print();
    }

    printf("\n======================= Sigma ========================\n");
    for(Int_t ii = 0; ii < MAXSIZE; ii++){
        printf("%d: Nrows() %d\n",ii, fSigmaPar[ii].GetNrows());
        if(fSigmaPar[ii].GetNrows()) fSigmaPar[ii].Print();
    }
    
    printf("AliTRDdEdxParams::Print done.\n\n");
}
