#ifndef ALIALGDETTPC_H
#define ALIALGDETTPC_H

#include "AliAlgDet.h"

/*--------------------------------------------------------
  TPC detector wrapper
  -------------------------------------------------------*/

// Author: ruben.shahoyan@cern.ch

class AliAlgDetTPC : public AliAlgDet
{
 public:
  AliAlgDetTPC(const char* title="");
  virtual ~AliAlgDetTPC();
  //
  virtual void DefineVolumes();  
  //
  Bool_t AcceptTrack(const AliESDtrack* trc, Int_t trtype) const;
  //
 protected:
  //
  // -------- dummies --------
  AliAlgDetTPC(const AliAlgDetTPC&);
  AliAlgDetTPC& operator=(const AliAlgDetTPC&);
  //
 protected:

  ClassDef(AliAlgDetTPC,1);
};

#endif
