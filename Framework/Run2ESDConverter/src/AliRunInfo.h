#ifndef ALIRUNINFO_H
#define ALIRUNINFO_H
/* Copyright(c) 1998-1999, ALICE Experiment at CERN, All rights reserved. *
 * See cxx source for full Copyright notice                               */

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

#include <TObject.h>
#include <TObjString.h>

#include "AliDAQ.h"

class AliRunInfo : public TObject {

 public:
  AliRunInfo();
  AliRunInfo(const char *lhcState,
	     const char *beamType,
	     Float_t beamEnergy,
	     const char *runType,
	     UInt_t activeDetectors);
  virtual ~AliRunInfo() {}

  virtual void Print(Option_t */*option=""*/) const { Dump(); }

  const char *GetLHCState() const { return fLHCState.Data(); }
  const char *GetBeamType() const { return fBeamType.Data(); }
  Float_t     GetBeamEnergy() const { return fBeamEnergy; }
  const char *GetRunType() const { return fRunType.Data(); }
  UInt_t      GetDetectorMask() const { return fActiveDetectors; }
  const char *GetActiveDetectors() const { return AliDAQ::ListOfTriggeredDetectors(fActiveDetectors); }

  AliRunInfo(const AliRunInfo &evInfo);
  AliRunInfo& operator= (const AliRunInfo& evInfo);

 private:

  TString  fLHCState;       // state of the machine as provided by DCS and DAQ log-book (per run)
  TString  fBeamType;       // beam type for Alice
  Float_t  fBeamEnergy;     // beam energy (in GeV)
  TString  fRunType;        // run type accoring to ECS (per run)
  UInt_t   fActiveDetectors;// list of active detectors (per run)

  ClassDef(AliRunInfo,1)     // Run info class
};

#endif
