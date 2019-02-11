#ifndef AliESDACORDE_H
#define AliESDACORDE_H

#include <TObject.h>
// Send comments to:
// Mario Rodriguez <mrodrigu@mail.cern.ch>
// Luciano Diaz <luciano.diaz@nucleares.unam.mx>
// Last update: October 2th 2009 

class AliESDACORDE : public TObject
{

 public:
  AliESDACORDE();
  AliESDACORDE(const AliESDACORDE&);
  AliESDACORDE(Bool_t *ACORDEBitPattern); 
  virtual ~AliESDACORDE() {};
  virtual void Copy(TObject &) const;

 // We define the "setters" for ACORDE

  void SetACORDEBitPattern(Bool_t ACORDEBitPattern[60]){for (Int_t i=0;i<60;i++){fACORDEBitPattern[i]=ACORDEBitPattern[i];}}

  
  
 // Getters  	
  Bool_t GetHitChannel(Int_t i) const;
  AliESDACORDE &operator=(const AliESDACORDE& source);
  
 protected:

  Bool_t	fACORDEBitPattern[60];  // Array with the ACORDE's bitpattern

  ClassDef(AliESDACORDE, 3)

};

#endif
