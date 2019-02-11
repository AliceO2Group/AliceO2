#ifndef ALIESDTRDTRACKLET_H
#define ALIESDTRDTRACKLET_H

/* Copyright(c) 1998-1999, ALICE Experiment at CERN, All rights reserved. *
 * See cxx source for full Copyright notice                               */

// ESD format for TRD tracklet from FEE used for triggering

#include "AliVTrdTracklet.h"

class AliESDTrdTracklet : public AliVTrdTracklet
{
 public:
  AliESDTrdTracklet();
  AliESDTrdTracklet(UInt_t trackletWord, Short_t hcid, const Int_t *label=0);
  AliESDTrdTracklet(const AliESDTrdTracklet &trkl);
  AliESDTrdTracklet& operator=(const AliESDTrdTracklet &trkl);
  ~AliESDTrdTracklet();

  void SetTrackletWord(UInt_t trklWord) { fTrackletWord = trklWord; }
  void SetHCId(Short_t hcid) { fHCId = hcid; }
  void SetLabel(Int_t label[]) { for (int i=3;i--;) fLabel[i] = label[i]; }
  void SetLabel(int ilb, Int_t label) { fLabel[ilb] = label; }

  // ----- tracklet information -----
  virtual UInt_t GetTrackletWord() const { return fTrackletWord; }
  virtual Int_t  GetBinY()  const;
  virtual Int_t  GetBinDy() const;
  virtual Int_t  GetBinZ()  const { return ((fTrackletWord >> 20) & 0xf);  }
  virtual Int_t  GetPID()   const { return ((fTrackletWord >> 24) & 0xff); }

  // ----- geometrical information -----
  Int_t GetHCId() const { return fHCId; }
  Int_t GetDetector() const { return fHCId / 2; }
  Int_t GetROB() const { return -1; }
  Int_t GetMCM() const { return -1; }

  // ----- MC information -----
  Int_t GetLabel()         const { return fLabel[0]; }
  Int_t GetLabel(int i)    const { return fLabel[i]; }
  const Int_t* GetLabels() const { return fLabel; }
  
 protected:
  Short_t fHCId;		// half-chamber ID

  UInt_t fTrackletWord;		// tracklet word (as from FEE)
				// pppp : pppp : zzzz : dddd : dddy : yyyy : yyyy : yyyy
  Int_t  fLabel[3];		// MC labels

  ClassDef(AliESDTrdTracklet, 3);

};

#endif
