// $Id$
// **************************************************************************
// This file is property of and copyright by the ALICE HLT Project          *
// ALICE Experiment at CERN, All rights reserved.                           *
//                                                                          *
// Primary Authors: Sergey Gorbunov <sergey.gorbunov@kip.uni-heidelberg.de> *
//                  Ivan Kisel <kisel@kip.uni-heidelberg.de>                *
//                  for The ALICE HLT Project.                              *
//                                                                          *
// Permission to use, copy, modify and distribute this software and its     *
// documentation strictly for non-commercial purposes is hereby granted     *
// without fee, provided that the above copyright notice appears in all     *
// copies and that both the copyright notice and this permission notice     *
// appear in the supporting documentation. The authors make no claims       *
// about the suitability of this software for any purpose. It is            *
// provided "as is" without express or implied warranty.                    *
//                                                                          *
//***************************************************************************

#include "AliHLTTPCCAGBHit.h"

//ClassImp(AliHLTTPCCAGBHit)

bool AliHLTTPCCAGBHit::Compare( const AliHLTTPCCAGBHit &a, const AliHLTTPCCAGBHit &b )
{
  //* Comparison function for sorting hits
  if ( a.fISlice < b.fISlice ) return 1;
  if ( a.fISlice > b.fISlice ) return 0;
  if ( a.fIRow < b.fIRow ) return 1;
  if ( a.fIRow > b.fIRow ) return 0;
  return ( a.fZ < b.fZ );
}
