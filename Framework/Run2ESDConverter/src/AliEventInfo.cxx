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
//                          Class AliEventInfo                              //
//   Container class for all the information related to                     //
//   event type, trigger mask and trigger clusters.                         //
//   It is used together with AliRunInfo in order to provide the detector's //
//   AliRecoParam object with                                               //
//   the necessary information so that it can decide which instance of      //
//   AliDetectorRecoParam objects to use in reconstruction one particular   //
//   event.                                                                 //
//                                                                          //
//   cvetan.cheshkov@cern.ch 12/06/2008                                     //
//////////////////////////////////////////////////////////////////////////////

#include "AliEventInfo.h"

ClassImp(AliEventInfo)

//______________________________________________________________________________
AliEventInfo::AliEventInfo():
  TObject(),
  fEventType(0),
  fTriggerClasses(""),
  fTriggerMask(0),
  fTriggerCluster(""),
  fHLTDecision("")
{
  // default constructor
  // ...
}

//______________________________________________________________________________
AliEventInfo::AliEventInfo(UInt_t evType,
			   const char *classes,
			   ULong64_t mask,
			   const char *cluster,
			   const char *decision):
  TObject(),
  fEventType(evType),
  fTriggerClasses(classes),
  fTriggerMask(mask),
  fTriggerCluster(cluster),
  fHLTDecision(decision)
{
  // constructor
  // ...
}

//______________________________________________________________________________
AliEventInfo::AliEventInfo(const AliEventInfo &evInfo):
  TObject(evInfo),
  fEventType(evInfo.fEventType),
  fTriggerClasses(evInfo.fTriggerClasses),
  fTriggerMask(evInfo.fTriggerMask),
  fTriggerCluster(evInfo.fTriggerCluster),
  fHLTDecision(evInfo.fHLTDecision)
{
  // Copy constructor
  // ...
}

//_____________________________________________________________________________
AliEventInfo &AliEventInfo::operator =(const AliEventInfo& evInfo)
{
  // assignment operator
  // ...
  if(this==&evInfo) return *this;
  ((TObject *)this)->operator=(evInfo);

  fEventType = evInfo.fEventType;
  fTriggerClasses = evInfo.fTriggerClasses;
  fTriggerMask = evInfo.fTriggerMask; 
  fTriggerCluster = evInfo.fTriggerCluster;
  fHLTDecision = evInfo.fHLTDecision;

  return *this;
}

//______________________________________________________________________________
void AliEventInfo::Reset()
{
  // Reset the contents
  // ...
  fEventType = 0;
  fTriggerClasses = "";
  fTriggerMask = 0;
  fTriggerCluster = "";
  fHLTDecision = "";
  ResetBit(0xffffffff);
}

//______________________________________________________________________________
void AliEventInfo::Print(Option_t* ) const
{
  // print itself
  printf("EventInfo for EventType:\t%d Cosmic:%s Laser:%s Other:%s\n",
	 fEventType,
	 HasCosmicTrigger() ? "ON":"OFF",
	 HasCalibLaserTrigger() ? "ON":"OFF",
	 HasBeamTrigger() ? "ON":"OFF");
  //
  printf("fTriggerMask/fTriggerMaskNext50:\t%#llx/%#llx\n",fTriggerMask,fTriggerMaskNext50);
  printf("TriggerCluster:\t%s\n",fTriggerCluster.Data());
  printf("TriggerClasses:\t%s\n",fTriggerClasses.Data());
  printf("HLT desicion  :\t%s\n",fHLTDecision.Data());  
}
