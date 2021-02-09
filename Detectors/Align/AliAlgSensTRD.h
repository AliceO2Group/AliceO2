#ifndef ALIALGSENSTRD_H
#define ALIALGSENSTRD_H

#include "AliAlgSens.h"
class AliTrackPointArray;
class AliESDtrack;
class AliAlgPoint;
class TObjArray;


/*--------------------------------------------------------
  TRD sensor
  -------------------------------------------------------*/

// Author: ruben.shahoyan@cern.ch


class AliAlgSensTRD : public AliAlgSens
{
 public:
  AliAlgSensTRD(const char* name=0, Int_t vid=0, Int_t iid=0, Int_t isec=0);
  virtual ~AliAlgSensTRD();
  //
  Int_t GetSector()                      const {return fSector;}
  void  SetSector(UInt_t sc)                   {fSector = (UChar_t)sc;}
  //
  virtual AliAlgPoint* TrackPoint2AlgPoint(int pntId, const AliTrackPointArray* trpArr, const AliESDtrack* t);
  //
  virtual void DPosTraDParCalib(const AliAlgPoint* pnt,double* deriv,int calibID,const AliAlgVol* parent=0) const;
  //
  //  virtual void   SetTrackingFrame();
  virtual void PrepareMatrixT2L();
  //
 protected:
  //
  UChar_t fSector;                      // sector ID

  ClassDef(AliAlgSensTRD,1)
};


#endif
