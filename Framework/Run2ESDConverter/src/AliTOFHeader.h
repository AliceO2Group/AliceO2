#ifndef ALITOFHEADER_H
#define ALITOFHEADER_H
/* Copyright(c) 1998-2003, ALICE Experiment at CERN, All rights reserved. *
 * See cxx source for full Copyright notice                               */

/* $Id$ */

//-------------------------------------------------------
//  Class for TOF event times, time resolution and T0spread
//          for the Event Data Summary Class
//   Origin: A.De Caro, Salerno, decaro@sa.infn.it
//-------------------------------------------------------

/**************************************************************
 *                                                            *
 * This class deals with:                                     *
 *       event time estimated by TOF combinatorial algorithm; *
 *       event time resolution;                               *
 *       TOF time resolution as written in TOF OCDB;          *
 *       T0spread as written in GRP.                          *
 *                                                            *
 **************************************************************/

#include "TObject.h"
//#include "TArrayF.h"
//#include "TArrayI.h"
#include "AliTOFTriggerMask.h"

class TArrayI;
class TArrayF;

class AliTOFHeader : public TObject {
 
 public:
 
  AliTOFHeader();
  AliTOFHeader(Float_t defEvTime, Float_t defResEvTime,
	       Int_t nDifPbins, Float_t *times, Float_t *res,
	       Int_t *nPbin, Float_t tofTimeRes, Float_t t0spread);
  AliTOFHeader(const AliTOFHeader &source);
  AliTOFHeader &operator=(const AliTOFHeader &source);
  virtual void Copy(TObject &obj) const;

  virtual ~AliTOFHeader();

  void SetTOFResolution(Float_t res) {fTOFtimeResolution=res;}
  Float_t GetTOFResolution()       const {return fTOFtimeResolution;}

  void SetT0spread(Float_t res)      {fT0spread=res;}
  Float_t GetT0spread()            const {return fT0spread;}

  Float_t GetDefaultEventTimeVal() const {return fDefaultEventTimeValue;}
  Float_t GetDefaultEventTimeRes() const {return fDefaultEventTimeRes;}
  TArrayF *GetEventTimeValues()    const {return fEventTimeValues;}
  TArrayF *GetEventTimeRes()       const {return fEventTimeRes;}
  TArrayI *GetNvalues()            const {return fNvalues;}
  Int_t GetNbins()                 const {return fNvalues ? fNvalues->GetSize() : 0;}
  Int_t GetNumberOfTOFclusters()   const {return fNumberOfTOFclusters;}
  Int_t GetNumberOfTOFtrgPads()    const {return fNumberOfTOFtrgPads;}
  Int_t GetNumberOfTOFmaxipad()    const {if(fTrigMask) return fTrigMask->GetNumberMaxiPadOn(); else return 0;}
  AliTOFTriggerMask *GetTriggerMask() const {return fTrigMask;}

  void SetDefaultEventTimeVal(Float_t val) {fDefaultEventTimeValue=val;}
  void SetDefaultEventTimeRes(Float_t res) {fDefaultEventTimeRes=res;}
  void SetEventTimeValues(TArrayF *arr);
  void SetEventTimeRes(TArrayF *arr);
  void SetNvalues(TArrayI *arr);
  void SetNbins(Int_t nbins);
  void SetNumberOfTOFclusters(Int_t a) {fNumberOfTOFclusters=a;}
  void SetNumberOfTOFtrgPads(Int_t a) {fNumberOfTOFtrgPads=a;}
  void SetTriggerMask(AliTOFTriggerMask *trigmask) {if(fTrigMask) *fTrigMask=*trigmask; else fTrigMask=new AliTOFTriggerMask(*trigmask);}

 protected:

  Float_t  fDefaultEventTimeValue; // TOF event time value more frequent
  Float_t  fDefaultEventTimeRes;   // TOF event time res more frequent
  Int_t    fNbins;                 // number of bins with TOF event
				   // time values different from
				   // default one
  TArrayF *fEventTimeValues;       // array for TOF event time values
				   // different from default one
  TArrayF *fEventTimeRes;          // array for TOF event time resolutions
  TArrayI *fNvalues;               // array for order numbers of momentum bin
  Float_t fTOFtimeResolution;      // TOF time resolution as written in TOF OCDB
  Float_t fT0spread;               // t0spread as written in TOF OCDB
  Int_t fNumberOfTOFclusters;      //[0,170000,18] number of reconstructed TOF clusters
  Int_t fNumberOfTOFtrgPads;       //[0,170000,18] number of reconstructed TOF trigger pads
  AliTOFTriggerMask *fTrigMask;    // Trigger mask

 private:

  ClassDef(AliTOFHeader,3)  // Class for TOF event times and so on
};

#endif
