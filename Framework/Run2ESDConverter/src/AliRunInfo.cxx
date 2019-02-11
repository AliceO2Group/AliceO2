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

//////////////////////////////////////////////////////////////////////////////
//                          Class AliRunInfo                                //
//   Container class for all the information related to LHCstate, run type, //
//   active detectors, beam energy etc.                                     //
//   It is used together with the AliEventInfo in order to provide          //
//   the AliRecoParam object with                                           //
//   the necessary information so that it can decide which instance of      //
//   AliDetectorRecoParam objects to use in reconstruction one particular   //
//   event.                                                                 //
//                                                                          //
//   cvetan.cheshkov@cern.ch 12/06/2008                                     //
//////////////////////////////////////////////////////////////////////////////

#include "AliRunInfo.h"

ClassImp(AliRunInfo)

//______________________________________________________________________________
AliRunInfo::AliRunInfo():
  TObject(),
  fLHCState("UNKNOWN"),
  fBeamType("UNKNOWN"),
  fBeamEnergy(0),
  fRunType("UNKNOWN"),
  fActiveDetectors(0)
{
  // default constructor
  // ...
}

//______________________________________________________________________________
AliRunInfo::AliRunInfo(const char *lhcState,
		       const char *beamType,
		       Float_t beamEnergy,
		       const char *runType,
		       UInt_t activeDetectors):
  TObject(),
  fLHCState(lhcState),
  fBeamType(beamType),
  fBeamEnergy(beamEnergy),
  fRunType(runType),
  fActiveDetectors(activeDetectors)
{
  // constructor
  // ...
}

//______________________________________________________________________________
AliRunInfo::AliRunInfo(const AliRunInfo &evInfo):
  TObject(evInfo),
  fLHCState(evInfo.fLHCState),
  fBeamType(evInfo.fBeamType),
  fBeamEnergy(evInfo.fBeamEnergy),
  fRunType(evInfo.fRunType),
  fActiveDetectors(evInfo.fActiveDetectors)
{
  // Copy constructor
  // ...
}

//_____________________________________________________________________________
AliRunInfo &AliRunInfo::operator =(const AliRunInfo& evInfo)
{
  // assignment operator
  // ...
  if(this==&evInfo) return *this;
  ((TObject *)this)->operator=(evInfo);

  fLHCState = evInfo.fLHCState;
  fBeamType = evInfo.fBeamType;
  fBeamEnergy = evInfo.fBeamEnergy;
  fRunType = evInfo.fRunType;
  fActiveDetectors = evInfo.fActiveDetectors;

  return *this;
}
