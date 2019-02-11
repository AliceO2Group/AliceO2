#ifndef ALIVFRIENDTRACK_H
#define ALIVFRIENDTRACK_H

//_____________________________________________________________________________
#include "AliVMisc.h"

#include "TObject.h"
#include "Rtypes.h"
#include "TObject.h"
class AliTPCseed;
class AliVVtrackPointArray;
class AliExternalTrackParam;
class AliTrackPointArray;

//_____________________________________________________________________________
class AliVfriendTrack :public TObject {
public:

  AliVfriendTrack(){}
  AliVfriendTrack( const AliVfriendTrack &t): TObject(t) {}
  // constructor for reinitialisation of vtable
  AliVfriendTrack( AliVConstructorReinitialisationFlag ): TObject(){}
  virtual ~AliVfriendTrack(){}

  //used in calibration
  
  virtual Int_t GetTPCseed( AliTPCseed &) const = 0;
  virtual TObject* GetCalibObject(Int_t /*index*/) const = 0;

  virtual const AliExternalTrackParam* GetTPCOut() const = 0;
  virtual const AliExternalTrackParam * GetITSOut() const = 0;  

  virtual Int_t GetTrackParamTPCOut( AliExternalTrackParam & ) const { return 0; }
  virtual Int_t GetTrackParamITSOut( AliExternalTrackParam & ) const { return 0; }
  //Int_t GetTrackParamTRDIn( AliExternalTrackParam &p ) const { return GetExternalTrackParam( p, 0x0  ); }

  virtual void ResetTrackParamTPCOut( const AliExternalTrackParam* ){ ; }
  virtual void ResetTPCseed( const AliTPCseed* /*s*/){}

  virtual const AliTrackPointArray *GetTrackPointArray() const {return NULL ;}

  // bit manipulation for filtering
  virtual void SetSkipBit(Bool_t skip) = 0;
  virtual Bool_t TestSkipBit() const = 0;

private: 
  AliVfriendTrack& operator=(const AliVfriendTrack& esd);  

  ClassDef(AliVfriendTrack, 0)  // 
};

#endif

