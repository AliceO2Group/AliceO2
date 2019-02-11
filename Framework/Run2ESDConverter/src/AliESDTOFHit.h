#ifndef ALIESDTOFHIT_H
#define ALIESDTOFHIT_H

/* Copyright(c) 1998-1999, ALICE Experiment at CERN, All rights reserved. *
 * See cxx source for full Copyright notice                               */

/* $Id: $ */

//----------------------------------------------------------------------//
//                                                                      //
// AliESDTOFHit Class                                                   //
//                                                                      //
//----------------------------------------------------------------------//

#include "AliVTOFHit.h"

class AliESDTOFHit : public AliVTOFHit
{
   public:
     AliESDTOFHit();
     AliESDTOFHit(Double_t time,Double_t timeraw, Double_t tot, Int_t channel, Int_t label[3],Int_t latency,Int_t deltabc,Int_t cluster,Float_t z,Float_t r,Float_t phi);
     AliESDTOFHit(AliESDTOFHit &source);
     virtual ~AliESDTOFHit() {}
     AliESDTOFHit & operator=(const AliESDTOFHit & source);
     //
     virtual Int_t   GetESDTOFClusterIndex()             const {return GetUniqueID();};
     virtual void    SetESDTOFClusterIndex(Int_t ind)          {SetUniqueID(ind);};
     //
     virtual void    SetTime(Double_t time) {fTime = time;}
     virtual void    SetTimeRaw(Double_t timeraw) {fTimeRaw=timeraw;};
     virtual void    SetTOT(Double_t tot) {fTOT = tot;};
     virtual void    SetL0L1Latency(Int_t latency) {fL0L1Latency = latency;};
     virtual void    SetDeltaBC(Int_t deltabc) {fDeltaBC = deltabc;};
     virtual void    SetTOFchannel(Int_t tofch) {fTOFchannel = tofch;};
     virtual Double_t GetTimeRaw() const {return fTimeRaw;};
     virtual Double_t GetTOT() const {return fTOT;};
     virtual Int_t   GetL0L1Latency() const {return fL0L1Latency;};
     virtual Int_t   GetDeltaBC() const {return fDeltaBC;};
     virtual Int_t   GetTOFchannel() const {return fTOFchannel;};
     virtual Double_t GetTime() const {return fTime;}
     virtual Int_t   GetTOFLabel(Int_t i) const {return (i >=0 && i < 3) ? fTOFLabel[i] : -1;}
     virtual void    SetTOFLabel(const Int_t label[3])  {for(Int_t i=3;i--;) fTOFLabel[i] = label[i];}
     Float_t GetR() const {return fR;};
     Float_t GetZ() const {return fZ;};
     Float_t GetPhi() const {return fPhi;};
     void SetR(Float_t val) {fR=val;};
     void SetZ(Float_t val) {fZ=val;};
     void SetPhi(Float_t val) {fPhi=val;};
     //
     void Print(const Option_t *opt=0) const;
     //
   protected:
     // additional info for ESD
     Double32_t fTimeRaw;         // Time Raw
     Double32_t fTime;            // TOF calibrated time
     Double32_t fTOT;             // Time Over Threshold
     Int_t fTOFLabel[3];          // TOF MC labels
     Int_t fL0L1Latency;          // L0L1 latency
     Int_t fDeltaBC;              // DeltaBC can it be Char_t of Short_t ?
     Int_t fTOFchannel;           // TOF channel

     Float_t fZ;                  //! coordinate for reco
     Float_t fR;                  //! coordinate for reco
     Float_t fPhi;                //! coordinate for reco

  ClassDef(AliESDTOFHit, 1) // TOF matchable hit

};
#endif
