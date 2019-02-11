#ifndef ALIVTRDTRACKLET_H
#define ALIVTRDTRACKLET_H

// format for the TRD tracklets calculated in the FEE,
// used for the TRD L1 trigger
// Author: Jochen Klein <jochen.klein@cern.ch>

#include "TObject.h"

class AliVTrdTracklet : public TObject {
 public:

  AliVTrdTracklet();
  virtual ~AliVTrdTracklet() {};
  AliVTrdTracklet(const AliVTrdTracklet& track);
  AliVTrdTracklet& operator=(const AliVTrdTracklet& track);
  virtual void Copy(TObject &obj) const;

  // tracklet information
  virtual UInt_t GetTrackletWord() const = 0;
  virtual Int_t  GetBinY() const = 0;
  virtual Int_t  GetBinDy() const = 0;
  virtual Int_t  GetBinZ() const = 0;
  virtual Int_t  GetPID() const = 0;

  // position and deflection information (chamber-local)
  Float_t GetLocalY() const { return GetBinY() * fgkBinWidthY; }
  Float_t GetDyDx() const { return GetBinDy() * fgkBinWidthDy/fgkDriftLength; }

  virtual Int_t GetHCId() const = 0;
  virtual Int_t GetDetector() const = 0;
  virtual Int_t GetLabel() const = 0;

 protected:

  static const Float_t fgkBinWidthY;   // bin width y-position
  static const Float_t fgkBinWidthDy;  // bin width deflection length
  static const Float_t fgkDriftLength; // drift length to which the deflection length is scaled

  ClassDef(AliVTrdTracklet,0)
};

#endif
