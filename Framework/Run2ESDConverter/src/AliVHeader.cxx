/**************************************************************************
 * Copyright(c) 1998-2007, ALICE Experiment at CERN, All rights reserved. *
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

/* $Id$ */

//-------------------------------------------------------------------------
//     Event header base class
//     Author: Markus Oldenburg, CERN
//-------------------------------------------------------------------------

#include "AliVHeader.h"

ClassImp(AliVHeader)

//______________________________________________________________________________
AliVHeader::AliVHeader() : 
  TNamed("header","") { } // default constructor 

//______________________________________________________________________________
AliVHeader::AliVHeader(const AliVHeader& hdr) :
  TNamed(hdr) { } // Copy constructor

//______________________________________________________________________________
AliVHeader& AliVHeader::operator=(const AliVHeader& hdr)
{
  // Assignment operator
  if(this!=&hdr) { 
    TNamed::operator=(hdr);
  }
  return *this;
}

//____________________________________________
ULong64_t AliVHeader::GetEventIdAsLong() const 
{
  // get global bunch corssing ID - as in  AliRawReader::GetEventIdAsLong 
  return (((ULong64_t)GetPeriodNumber()<<36)|((ULong64_t)GetOrbitNumber()<<12)|((ULong64_t)GetBunchCrossNumber()));
}  
