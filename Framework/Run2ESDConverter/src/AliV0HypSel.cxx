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

//-------------------------------------------------------------------------
//    Implementation of the V0 selection hypothesis.
//    V0 is validated if it passes at least 1 hypotheses selection
//-------------------------------------------------------------------------


#include "AliV0HypSel.h"
#include "AliLog.h"

ClassImp(AliV0HypSel)

float AliV0HypSel::fgBFieldCoef = 1.0;

//________________________________________________________________________
AliV0HypSel::AliV0HypSel()
:  fM0(0.)
  ,fM1(0.)
  ,fMass(0.)
  ,fSigmaM(0.)
  ,fCoef0Pt(0.)
  ,fCoef1Pt(0.)
  ,fNSigma(0.)
  ,fMarginAdd(0.)
{}

//________________________________________________________________________
AliV0HypSel::AliV0HypSel(const AliV0HypSel& src)
  : TNamed(src)
  ,fM0(src.fM0)
  ,fM1(src.fM1)
  ,fMass(src.fMass)
  ,fSigmaM(src.fSigmaM)
  ,fCoef0Pt(src.fCoef0Pt)
  ,fCoef1Pt(src.fCoef1Pt)
  ,fNSigma(src.fNSigma)
  ,fMarginAdd(src.fMarginAdd)
{}

//________________________________________________________________________
AliV0HypSel::AliV0HypSel(const char *name, float m0,float m1, float mass, float sigma,
			 float nsig, float margin, float cf0, float cf1) :
  TNamed(name,name)
  ,fM0(m0)
  ,fM1(m1)
  ,fMass(mass)
  ,fSigmaM(sigma)
  ,fCoef0Pt(cf0)
  ,fCoef1Pt(cf1)
  ,fNSigma(nsig)
  ,fMarginAdd(margin)
{
  Validate();
}

//________________________________________________________________________
void AliV0HypSel::Validate()
{
  // consistency check
  if (fM0<0.0005 || fM1<0.0005) {
    AliFatal("V0 decay product mass cannot be lighter than electron");
  }
  if (fMass<fM0+fM1) {
    AliFatal("V0 mass is less than sum of product masses");
  }
  if ( (fSigmaM<=0 || fNSigma<=0) && fMarginAdd<=1e-3) {
    AliFatal("No safety margin is provided");
  }
}

//________________________________________________________________________
void AliV0HypSel::Print(const Option_t *) const
{
  // print itself
  printf("%-15s | m0: %.4e m1: %.4e -> M: %.4e\nCut margin: %.1f*%.3e*(%.f+%.f*pT)+%.3e\n",GetName(),fM0,fM1,fMass,
	 fNSigma,fSigmaM,fCoef0Pt,fCoef1Pt,fMarginAdd);
}

//________________________________________________________________________
void AliV0HypSel::AccountBField(float b)
{
  // account effect of B-field on pT resolution, ignoring the fact that the V0 mass resolution
  // is only partially determined by the prongs pT resolution
  const float kNomField = 5.00668e+00;
  float babs = TMath::Abs(b);
  if (babs>1e-3) SetBFieldCoef(kNomField/babs);
}
