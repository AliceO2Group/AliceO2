#ifndef ALIALGSENSHMPID_H
#define ALIALGSENSHMPID_H

#include "AliAlgSens.h"


class TObjArray;
class AliTrackPointArray;
class AliESDtrack;
class AliAlgPoint;


/*--------------------------------------------------------
  HMPID sensor (chamber)
  -------------------------------------------------------*/

// Author: ruben.shahoyan@cern.ch


class AliAlgSensHMPID : public AliAlgSens
{
 public:
  AliAlgSensHMPID(const char* name=0, Int_t vid=0, Int_t iid=0, Int_t isec=0);
  virtual ~AliAlgSensHMPID();
  //
  virtual AliAlgPoint* TrackPoint2AlgPoint(int pntId, const AliTrackPointArray* trpArr, const AliESDtrack* t);
  //  virtual void   SetTrackingFrame();
  virtual void PrepareMatrixT2L();
  //
 protected:
  //
  ClassDef(AliAlgSensHMPID,1)
};


#endif
