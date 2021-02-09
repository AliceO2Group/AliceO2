#ifndef ALIALGDETITS_H
#define ALIALGDETITS_H

#include "AliAlgDet.h"
#include "AliAlgAux.h"

/*--------------------------------------------------------
  ITS detector wrapper
  -------------------------------------------------------*/

// Author: ruben.shahoyan@cern.ch


class AliAlgDetITS : public AliAlgDet
{
 public:
  //
  enum ITSSel_t {kSPDNoSel,kSPDBoth,kSPDAny,kSPD0,kSPD1,kNSPDSelTypes};
  //
  AliAlgDetITS(const char* title="");
  virtual ~AliAlgDetITS();
  //
  virtual void DefineVolumes();  
  //
  Bool_t AcceptTrack(const AliESDtrack* trc,Int_t trtype) const;

  void   SetAddErrorLr(int ilr, double sigY, double sigZ);
  void   SetSkipLr(int ilr);
  //
  virtual void  UpdatePointByTrackInfo(AliAlgPoint* pnt, const AliExternalTrackParam* t) const;
  virtual void  SetUseErrorParam(Int_t v=1);
  void          SetITSSelPattern(Int_t trtype, ITSSel_t sel) {fITSPatt[trtype]=sel;}
  void          SetITSSelPatternColl(ITSSel_t sel=kSPDAny)   {SetITSSelPattern(AliAlgAux::kColl,sel);}
  void          SetITSSelPatternCosm(ITSSel_t sel=kSPDNoSel) {SetITSSelPattern(AliAlgAux::kCosm,sel);}

  Int_t         GetITSSelPattern(int tp) const {return fITSPatt[tp];}
  Int_t         GetITSSelPatternColl()   const {return fITSPatt[AliAlgAux::kColl];}
  Int_t         GetITSSelPatternCosm()   const {return fITSPatt[AliAlgAux::kCosm];}
  //
  virtual void  Print(const Option_t *opt="")    const;
  //
  static Bool_t CheckHitPattern(const AliESDtrack* trc, Int_t sel);
  static const char* GetITSPattName(Int_t sel) {return sel<kNSPDSelTypes ? fgkHitsSel[sel]:0;}
  //
 protected:
  //
  void GetErrorParamAngle(int layer,double tgl,double tgphitr,double &erry,double &errz) const;
  //
  // -------- dummies --------
  AliAlgDetITS(const AliAlgDetITS&);
  AliAlgDetITS& operator=(const AliAlgDetITS&);
  //
 protected:
  //
  Int_t fITSPatt[AliAlgAux::kNTrackTypes];                   // ITS hits selection pattern for coll/cosm tracks
  //
  static const Char_t* fgkHitsSel[kNSPDSelTypes];            // ITS selection names
  //
  ClassDef(AliAlgDetITS,1);
};

#endif
