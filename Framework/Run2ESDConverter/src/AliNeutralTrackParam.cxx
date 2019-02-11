/**************************************************************************
 * Copyright(c) 1998-2009, ALICE Experiment at CERN, All rights reserved. *
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

///////////////////////////////////////////////////////////////////////////////
//                                                                           //
// Implementation of the "neutral" track parameterisation class.             //
//                                                                           //
// At the moment we use a standard AliExternalTrackParam with 0 curvature.   //
//                                                                           //
//        Origin: A.Dainese, I.Belikov                                       //
///////////////////////////////////////////////////////////////////////////////

#include "AliExternalTrackParam.h"
#include "AliNeutralTrackParam.h"

ClassImp(AliNeutralTrackParam)
 
//_____________________________________________________________________________
AliNeutralTrackParam::AliNeutralTrackParam() :
AliExternalTrackParam()
{
  //
  // default constructor
  //
}

//_____________________________________________________________________________
AliNeutralTrackParam::AliNeutralTrackParam(const AliNeutralTrackParam &track):
AliExternalTrackParam(track)
{
  //
  // copy constructor
  //
}

//_____________________________________________________________________________
AliNeutralTrackParam& AliNeutralTrackParam::operator=(const AliNeutralTrackParam &trkPar)
{
  //
  // assignment operator
  //
  
  if (this!=&trkPar) {
    AliExternalTrackParam::operator=(trkPar);
  }

  return *this;
}

//_____________________________________________________________________________
AliNeutralTrackParam::AliNeutralTrackParam(Double_t x, Double_t alpha, 
					     const Double_t param[5], 
					     const Double_t covar[15]) :
AliExternalTrackParam(x,alpha,param,covar)
{
  //
  // create external track parameters from given arguments
  //
}

//_____________________________________________________________________________
AliNeutralTrackParam::AliNeutralTrackParam(const AliVTrack *vTrack) :
AliExternalTrackParam(vTrack)
{
  //
  // Constructor from virtual track,
  // This is not a copy contructor !
  //

}

//_____________________________________________________________________________
AliNeutralTrackParam::AliNeutralTrackParam(Double_t xyz[3],Double_t pxpypz[3],
					     Double_t cv[21],Short_t sign) :
AliExternalTrackParam(xyz,pxpypz,cv,sign)
{
  //
  // constructor from the global parameters
  //
}

