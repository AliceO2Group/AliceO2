// @(#) $Id: AliHLTTPCCARow.cxx 31983 2009-04-17 15:46:49Z sgorbuno $
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

#include "AliHLTTPCCARow.h"


#if !defined(HLTCA_GPUCODE)
AliHLTTPCCARow::AliHLTTPCCARow()
    :
    fNHits( 0 ), fX( 0 ), fMaxY( 0 ), fGrid(),
    fHy0( 0 ), fHz0( 0 ), fHstepY( 0 ), fHstepZ( 0 ), fHstepYi( 0 ), fHstepZi( 0 ),
    fFullSize( 0 ), fHitNumberOffset( 0 ), fFirstHitInBinOffset( 0 )
{
  // dummy constructor
}

#endif

