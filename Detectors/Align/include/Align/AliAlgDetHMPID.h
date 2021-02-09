#ifndef ALIALGDETHMPID_H
#define ALIALGDETHMPID_H

#include "AliAlgDet.h"

/*--------------------------------------------------------
  HMPID detector wrapper
  -------------------------------------------------------*/

// Author: ruben.shahoyan@cern.ch

class AliAlgDetHMPID : public AliAlgDet
{
 public:
  AliAlgDetHMPID(const char* title="");
  virtual ~AliAlgDetHMPID();
  //
  virtual void DefineVolumes();  
  //
  Bool_t AcceptTrack(const AliESDtrack* trc, Int_t trtype) const;
  //
 protected:
  //
  // -------- dummies --------
  AliAlgDetHMPID(const AliAlgDetHMPID&);
  AliAlgDetHMPID& operator=(const AliAlgDetHMPID&);
  //
 protected:

  ClassDef(AliAlgDetHMPID,1);
};

#endif
