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


//-------------------------------------------------------------------------
//     base class for ESD and AOD vertices
//     Author: A. Dainese
//-------------------------------------------------------------------------

#include "AliVVertex.h"
#include "AliVTrack.h"

ClassImp(AliVVertex)

AliVVertex::AliVVertex(const AliVVertex& vVert) :
  TNamed(vVert) { } // Copy constructor

AliVVertex& AliVVertex::operator=(const AliVVertex& vVert)
{ 
    // Copy constructor
    if (this!=&vVert) { 
	TNamed::operator=(vVert); 
    }
  
  return *this; 
}

Int_t AliVVertex::GetBC() const 
{
  // get BCID
  return AliVTrack::kTOFBCNA;
}
