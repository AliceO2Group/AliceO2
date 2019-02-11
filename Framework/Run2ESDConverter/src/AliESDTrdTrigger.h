#ifndef ALIESDTRDTRIGGER_H
#define ALIESDTRDTRIGGER_H

#include "TObject.h"

class AliESDTrdTrigger : public TObject
{
 public:
  AliESDTrdTrigger();
  AliESDTrdTrigger(const AliESDTrdTrigger &rhs);
  AliESDTrdTrigger& operator=(const AliESDTrdTrigger &rhs);
  ~AliESDTrdTrigger();

  UInt_t GetFlags(const Int_t sector) const { return fFlags[sector]; }

  void SetFlags(const Int_t sector, const UInt_t flags) { fFlags[sector] = flags; }

 protected:
  static const Int_t fgkNsectors = 18;	  // number of sectors

  UInt_t fFlags[fgkNsectors];	          // trigger flags for every sector

  ClassDef(AliESDTrdTrigger, 1);
};

#endif
