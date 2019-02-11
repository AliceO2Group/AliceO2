/**************************************************************************
 * Copyright(c) 1998-2010, ALICE Experiment at CERN, All rights reserved. *
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

//***********************************************************
// Class AliHMPIDPIDParams
//
// class to store PID parameters for HMPID in OADB
//
// Author: G. Volpe, giacomo.volpe@cern.ch
//***********************************************************

#include <TNamed.h>
#include "AliHMPIDPIDParams.h"

ClassImp(AliHMPIDPIDParams)

//_____________________________________________________________________________
AliHMPIDPIDParams::AliHMPIDPIDParams():
 TNamed("default",""),
 fHMPIDRefIndexArray(0x0)
{
  
}
//_____________________________________________________________________________
AliHMPIDPIDParams::AliHMPIDPIDParams(Char_t *name):
  TNamed(name,""),
  fHMPIDRefIndexArray(0x0)
{
  
}
//___________________________________________________________________________
AliHMPIDPIDParams& AliHMPIDPIDParams::operator=(const AliHMPIDPIDParams& c)
{
  //
  // Assignment operator
  //
  if (this!=&c) {
    fHMPIDRefIndexArray=c.fHMPIDRefIndexArray;
  }
  return *this;
}

//___________________________________________________________________________
AliHMPIDPIDParams::AliHMPIDPIDParams(const AliHMPIDPIDParams& c) :
 TNamed(c),
 fHMPIDRefIndexArray(c.fHMPIDRefIndexArray)   
{
  //
  // Copy Constructor
  //
  
}
 
//_____________________________________________________________________________
AliHMPIDPIDParams::~AliHMPIDPIDParams(){
}


