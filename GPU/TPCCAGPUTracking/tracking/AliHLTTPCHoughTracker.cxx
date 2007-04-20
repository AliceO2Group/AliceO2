// $Id$
// origin: src/AliL3TPCtracker.cxx 1.2 Fri Jun 10 04:25:00 2005 UTC by cvetan

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

/** @file   AliHLTTPCHoughTracker.cxx
    @author Cvetan Cheshkov
    @date   
    @brief  Implementation of the HLT TPC hough transform tracker. */

//-------------------------------------------------------------------------
//       Implementation of the HLT TPC hough transform tracker class
//
//    It reads directly TPC digits using runloader and runs the HT
//    algorithm over them.
//    It stores the reconstructed hough tracks in the HLT ESD using the
//    the off-line AliESDtrack format.
//
//       Origin: Cvetan Cheshkov, CERN, Cvetan.Cheshkov@cern.ch
//-------------------------------------------------------------------------

#include "AliESD.h"
#include "AliRunLoader.h"
#include "AliHLTTPCHoughTracker.h"
#include "AliHLTTPCHough.h"
#include "AliLog.h"
#include "AliHLTTPCTransform.h"
#include "AliHLTTPCLog.h"

/** ROOT macro for the implementation of ROOT specific class methods */
ClassImp(AliHLTTPCHoughTracker);

/** 
 * creator
 * used for instantiation when the library is loaded dynamically
 */
extern "C" AliTracker* CreateHLTTPCHoughTrackerInstance(AliRunLoader* runLoader)
{
  return new AliHLTTPCHoughTracker(runLoader);
}

AliHLTTPCHoughTracker::AliHLTTPCHoughTracker(AliRunLoader *runLoader):AliTracker()
{
  //--------------------------------------------------------------
  // Constructor
  //--------------------------------------------------------------

  if(AliHLTTPCTransform::GetVersion() == AliHLTTPCTransform::kVdefault) {
    Bool_t isinit=AliHLTTPCTransform::Init(runLoader);
    if(!isinit) AliWarning("Could not init AliHLTTPCTransform settings, using defaults!");
  }

  fRunLoader = runLoader;
}

Int_t AliHLTTPCHoughTracker::Clusters2Tracks(AliESD *event)
{
  //--------------------------------------------------------------------
  // This method reconstructs HLT TPC Hough tracks
  //--------------------------------------------------------------------
  
  if (!fRunLoader) {
    AliError("Missing runloader!");
    return kTRUE;
  }
  Int_t iEvent = fRunLoader->GetEventNumber();
  
  Float_t ptmin = 0.1*AliHLTTPCTransform::GetSolenoidField();

  Float_t zvertex = GetZ();

  AliInfo(Form("Hough Transform will run with ptmin=%f and zvertex=%f",ptmin,zvertex));

  // increase logging level temporarily to avoid bunches of info messages
  AliHLTTPCLog::TLogLevel loglevelbk=AliHLTTPCLog::fgLevel;
  AliHLTTPCLog::fgLevel=AliHLTTPCLog::kWarning;

  AliHLTTPCHough *hough = new AliHLTTPCHough();
    
  hough->SetThreshold(4);
  hough->CalcTransformerParams(ptmin);
  hough->SetPeakThreshold(70,-1);
  hough->SetRunLoader(fRunLoader);
  hough->Init("./", kFALSE, 100, kFALSE,4,0,0,zvertex);
  hough->SetAddHistograms();

  for(Int_t slice=0; slice<=35; slice++)
    {
      hough->ReadData(slice,iEvent);
      hough->Transform();
      hough->AddAllHistogramsRows();
      hough->FindTrackCandidatesRow();
      hough->AddTracks();
    }

  Int_t ntrk = hough->FillESD(event);

  Info("Clusters2Tracks","Number of found tracks: %d\n",ntrk);
  
  delete hough;
  AliHLTTPCLog::fgLevel=loglevelbk;

  return 0;
}
