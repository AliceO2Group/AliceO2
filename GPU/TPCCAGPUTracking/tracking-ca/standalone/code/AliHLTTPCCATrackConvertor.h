//-*- Mode: C++ -*-
// ************************************************************************
// This file is property of and copyright by the ALICE HLT Project        *
// ALICE Experiment at CERN, All rights reserved.                         *
// See cxx source for full Copyright notice                               *
//                                                                        *
//*************************************************************************

#ifndef ALIHLTTPCCATRACKCONVERTOR_H
#define ALIHLTTPCCATRACKCONVERTOR_H

#include "AliHLTTPCCADef.h"

class AliExternalTrackParam;
class AliHLTTPCCATrackParam;

/**
 * @class AliHLTTPCCATrackConvertor
 *
 * AliHLTTPCCATrackConvertor class converts tracks to different parameterisations
 * it is used by the AliHLTTPCCATrackerComponent
 *
 */
class AliHLTTPCCATrackConvertor
{
  public:

    AliHLTTPCCATrackConvertor() {}

    static bool GetExtParam( const AliHLTTPCCATrackParam &T1, AliExternalTrackParam &T2, double alpha );
    static void SetExtParam( AliHLTTPCCATrackParam &T1, const AliExternalTrackParam &T2 );

};


#endif //ALIHLTTPCCATRACKCONVERTOR_H
