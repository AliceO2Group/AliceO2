#ifndef ALIALGSENSITS_H
#define ALIALGSENSITS_H

#include "AliAlgSens.h"


class TObjArray;
class AliTrackPointArray;
class AliESDtrack;
class AliAlgPoint;


/*--------------------------------------------------------
  ITS sensor
  -------------------------------------------------------*/

// Author: ruben.shahoyan@cern.ch


class AliAlgSensITS : public AliAlgSens
{
 public:
  AliAlgSensITS(const char* name=0, Int_t vid=0, Int_t iid=0);
  virtual ~AliAlgSensITS();
  //
  virtual AliAlgPoint* TrackPoint2AlgPoint(int pntId, const AliTrackPointArray* trpArr, const AliESDtrack* t);

  //  virtual void   SetTrackingFrame();
  //
 protected:
  //
  ClassDef(AliAlgSensITS,1)
};


#endif
