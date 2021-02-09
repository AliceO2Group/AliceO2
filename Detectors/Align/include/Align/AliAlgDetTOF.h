#ifndef ALIALGDETTOF_H
#define ALIALGDETTOF_H

#include "AliAlgDet.h"

/*--------------------------------------------------------
  Wrapper for TOF detector
  -------------------------------------------------------*/

// Author: ruben.shahoyan@cern.ch

class AliAlgDetTOF : public AliAlgDet
{
 public:
  AliAlgDetTOF(const char* title="");
  virtual ~AliAlgDetTOF();
  //
  virtual void DefineVolumes();  
  //
  Bool_t AcceptTrack(const AliESDtrack* trc,Int_t trtype) const;
  //
 protected:
  //
  // -------- dummies --------
  AliAlgDetTOF(const AliAlgDetTOF&);
  AliAlgDetTOF& operator=(const AliAlgDetTOF&);
  //
 protected:

  ClassDef(AliAlgDetTOF,1);
};

#endif
