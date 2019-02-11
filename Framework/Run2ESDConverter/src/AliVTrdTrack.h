#ifndef ALIVTRDTRACK_H
#define ALIVTRDTRACK_H

//
// format for the TRD tracks calculated in the
// Global Tracking Unit, used for the TRD L1 trigger
// Author: Jochen Klein <jochen.klein@cern.ch>

#include "TObject.h"

class AliVTrack;
class AliVTrdTracklet;

class AliVTrdTrack : public TObject {
 public:

  AliVTrdTrack();
  virtual ~AliVTrdTrack() {};
  AliVTrdTrack(const AliVTrdTrack& track);
  AliVTrdTrack& operator=(const AliVTrdTrack& track);
  virtual void Copy(TObject &obj) const;

  virtual Int_t GetA()         const = 0;
  virtual Int_t GetLayerMask() const = 0;
  virtual Int_t GetPID()       const = 0;
  virtual Int_t GetPt()        const = 0;
  virtual Int_t GetStack()     const = 0;
  virtual Int_t GetSector()    const = 0;

  virtual Bool_t GetTrackInTime() const = 0;
  virtual UChar_t GetFlagsTiming() const = 0;

  virtual Int_t GetLabel()     const = 0;

  virtual Double_t Pt()        const = 0;

  virtual Int_t GetNTracklets() const = 0;
  virtual AliVTrdTracklet* GetTracklet(Int_t idx) const = 0;

  virtual AliVTrack* GetTrackMatch() const = 0;

  virtual void SetA(Int_t a) = 0;
  virtual void SetLayerMask(Int_t mask) = 0;
  virtual void SetPID(Int_t pid) = 0;
  virtual void SetLabel(Int_t label) = 0;
  virtual void SetSector(Int_t sector) = 0;
  virtual void SetStack(Int_t stack) = 0;

  virtual Bool_t IsSortable() const = 0;
  virtual Int_t Compare(const TObject* obj) const = 0;

 protected:

  static const Int_t fgkNlayers = 6;      // number of TRD layers

  ClassDef(AliVTrdTrack,0)
};

#endif
