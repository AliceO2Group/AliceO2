#ifndef ALIALGDET_H
#define ALIALGDET_H

#include <TNamed.h>
#include <TObjArray.h>
#include <stdio.h>
#include "AliAlgAux.h"
#include "AliESDtrack.h"
class AliAlgTrack;
class AliAlgDOFStat;
class AliAlgPoint;
class AliAlgSens;
class AliAlgVol;
class AliAlgSteer;
class AliTrackPointArray;
class AliExternalTrackParam;
class TH1;

/*--------------------------------------------------------
  Base class for detector: wrapper for set of volumes
  -------------------------------------------------------*/

// Author: ruben.shahoyan@cern.ch

class AliAlgDet : public TNamed
{
 public:
  enum {kInitGeomDone=BIT(14),kInitDOFsDone=BIT(15)};
  enum {kNMaxKalibDOF=64};
  //
  AliAlgDet();
  AliAlgDet(const char* name, const char* title="");
  virtual ~AliAlgDet();
  Int_t   GetDetID()                            const {return GetUniqueID();}
  void    SetDetID(UInt_t tp);
  //
  virtual void  CacheReferenceOCDB();
  virtual void  AcknowledgeNewRun(Int_t run);
  virtual void  UpdateL2GRecoMatrices();
  virtual void  ApplyAlignmentFromMPSol();
  //
  Int_t   VolID2SID(Int_t vid)                  const;
  Int_t   SID2VolID(Int_t sid)                  const {return sid<GetNSensors() ? fSID2VolID[sid] : -1;} //todo
  Int_t   GetNSensors()                         const {return fSensors.GetEntriesFast();}
  Int_t   GetNVolumes()                         const {return fVolumes.GetEntriesFast();}
  Int_t   GetVolIDMin()                         const {return fVolIDMin;}
  Int_t   GetVolIDMax()                         const {return fVolIDMax;}
  Bool_t  SensorOfDetector(Int_t vid)           const {return vid>=fVolIDMin && vid<=fVolIDMax;}
  void    SetAddError(double y, double z);
  const   Double_t* GetAddError()               const {return fAddError;} 
  //
  Int_t   GetNPoints()                          const {return fNPoints;}
  //
  void        SetAlgSteer(AliAlgSteer* s)             {fAlgSteer = s;}
  AliAlgSens* GetSensor(Int_t id)               const {return (AliAlgSens*)fSensors.UncheckedAt(id);}
  AliAlgSens* GetSensorByVolId(Int_t vid)       const {int sid=VolID2SID(vid); return sid<0 ? 0:GetSensor(sid);}
  AliAlgSens* GetSensor(const char* symname)    const {return (AliAlgSens*)fSensors.FindObject(symname);}
  AliAlgVol*  GetVolume(Int_t id)               const {return (AliAlgVol*)fVolumes.UncheckedAt(id);}
  AliAlgVol*  GetVolume(const char* symname)    const {return (AliAlgVol*)fVolumes.FindObject(symname);}
  //
  Bool_t      OwnsDOFID(Int_t id)               const;
  AliAlgVol*  GetVolOfDOFID(Int_t id)           const;
  //
  Int_t       GetDetLabel()                     const {return (GetDetID()+1)*1000000;}
  void        SetFreeDOF(Int_t dof);
  void        FixDOF(Int_t dof);
  void        SetFreeDOFPattern(ULong64_t pat)        {fCalibDOF = pat; CalcFree();}
  Bool_t      IsFreeDOF(Int_t dof)              const {return (fCalibDOF&(0x1<<dof))!=0;}
  Bool_t      IsCondDOF(Int_t dof)              const;
  ULong64_t   GetFreeDOFPattern()               const {return fCalibDOF;}
  Int_t       GetNProcessedPoints()             const {return fNProcPoints;}
  virtual const char* GetCalibDOFName(int)      const {return 0;}
  virtual     Double_t GetCalibDOFVal(int)      const {return 0;}
  virtual     Double_t GetCalibDOFValWithCal(int) const {return 0;}
  //
  virtual Int_t InitGeom();
  virtual Int_t AssignDOFs();
  virtual void  InitDOFs();
  virtual void  Terminate();
  void          FillDOFStat(AliAlgDOFStat* dofst=0) const;
  virtual void  AddVolume(AliAlgVol* vol);
  virtual void  DefineVolumes();
  virtual void  DefineMatrices();
  virtual void  Print(const Option_t *opt="")    const;
  virtual Int_t ProcessPoints(const AliESDtrack* esdTr, AliAlgTrack* algTrack,Bool_t inv=kFALSE);
  virtual void  UpdatePointByTrackInfo(AliAlgPoint* pnt, const AliExternalTrackParam* t) const;
  virtual void  SetUseErrorParam(Int_t v=0);
  Int_t         GetUseErrorParam()                   const {return fUseErrorParam;}
  //
  virtual Bool_t AcceptTrack(const AliESDtrack* trc,Int_t trtype) const = 0;
  Bool_t         CheckFlags(const AliESDtrack* trc,Int_t trtype) const;
  //
  virtual AliAlgPoint* GetPointFromPool();
  virtual void ResetPool();
  virtual void WriteSensorPositions(const char* outFName);
  //
  void      SetInitGeomDone()                             {SetBit(kInitGeomDone);}
  Bool_t    GetInitGeomDone()                       const {return TestBit(kInitGeomDone);}
  //
  void      SetInitDOFsDone()                             {SetBit(kInitDOFsDone);}
  Bool_t    GetInitDOFsDone()                       const {return TestBit(kInitDOFsDone);}
  void      FixNonSensors();
  void      SetFreeDOFPattern(UInt_t pat=0xffffffff, int lev=-1,const char* match=0);
  void      SetDOFCondition(int dof, float condErr, int lev=-1,const char* match=0);
  int       SelectVolumes(TObjArray* arr, int lev=-1,const char* match=0);
  //
  Int_t     GetNDOFs()                              const {return fNDOFs;}
  Int_t     GetNCalibDOFs()                         const {return fNCalibDOF;}
  Int_t     GetNCalibDOFsFree()                     const {return fNCalibDOFFree;}
  //
  void      SetDisabled(Int_t tp,Bool_t v)                {fDisabled[tp]=v;SetObligatory(tp,!v);}
  void      SetDisabled()                                 {SetDisabledColl();SetDisabledCosm();}
  void      SetDisabledColl(Bool_t v=kTRUE)               {SetDisabled(AliAlgAux::kColl,v);}
  void      SetDisabledCosm(Bool_t v=kTRUE)               {SetDisabled(AliAlgAux::kCosm,v);}
  Bool_t    IsDisabled(Int_t tp)                    const {return fDisabled[tp];}
  Bool_t    IsDisabled()                            const {return IsDisabledColl()&&IsDisabledCosm();}
  Bool_t    IsDisabledColl()                        const {return IsDisabled(AliAlgAux::kColl);}
  Bool_t    IsDisabledCosm()                        const {return IsDisabled(AliAlgAux::kCosm);}
  //
  void      SetTrackFlagSel(Int_t tp,ULong_t f)           {fTrackFlagSel[tp] = f;}
  void      SetTrackFlagSelColl(ULong_t f)                {SetTrackFlagSel(AliAlgAux::kColl,f);}
  void      SetTrackFlagSelCosm(ULong_t f)                {SetTrackFlagSel(AliAlgAux::kCosm,f);}
  ULong_t   GetTrackFlagSel(Int_t tp)               const {return fTrackFlagSel[tp];}
  ULong_t   GetTrackFlagSelColl()                   const {return GetTrackFlagSel(AliAlgAux::kColl);}
  ULong_t   GetTrackFlagSelCosm()                   const {return GetTrackFlagSel(AliAlgAux::kCosm);}
  //
  void      SetNPointsSel(Int_t tp,Int_t n)               {fNPointsSel[tp] = n;}
  void      SetNPointsSelColl(Int_t n)                    {SetNPointsSel(AliAlgAux::kColl,n);}
  void      SetNPointsSelCosm(Int_t n)                    {SetNPointsSel(AliAlgAux::kCosm,n);}
  Int_t     GetNPointsSel(Int_t tp)                 const {return fNPointsSel[tp];}
  Int_t     GetNPointsSelColl()                     const {return GetNPointsSel(AliAlgAux::kColl);}
  Int_t     GetNPointsSelCosm()                     const {return GetNPointsSel(AliAlgAux::kCosm);}
  //
  //
  Bool_t    IsObligatory(Int_t tp)                  const {return fObligatory[tp];}
  Bool_t    IsObligatoryColl()                      const {return IsObligatory(AliAlgAux::kColl);}
  Bool_t    IsObligatoryCosm()                      const {return IsObligatory(AliAlgAux::kCosm);}
  void      SetObligatory(Int_t tp,Bool_t v=kTRUE);
  void      SetObligatoryColl(Bool_t v=kTRUE)             {SetObligatory(AliAlgAux::kColl,v);}
  void      SetObligatoryCosm(Bool_t v=kTRUE)             {SetObligatory(AliAlgAux::kCosm,v);}
  //
  void      AddAutoConstraints()                    const;
  void      ConstrainOrphans(const double* sigma, const char* match=0);

  virtual void      WritePedeInfo(FILE* parOut,const Option_t *opt="") const;
  virtual void      WriteCalibrationResults()       const;
  virtual void      WriteAlignmentResults()         const;
  //
  Float_t*   GetParVals()                           const {return fParVals;}
  Double_t   GetParVal(int par)                     const {return fParVals ? fParVals[par] : 0;}
  Double_t   GetParErr(int par)                     const {return fParErrs ? fParErrs[par] : 0;}
  Int_t      GetParLab(int par)                     const {return fParLabs ? fParLabs[par] : 0;}
  //
  void       SetParVals(Int_t npar,Double_t *vl,Double_t *er);
  void       SetParVal(Int_t par,Double_t v=0)            {fParVals[par] = v;}
  void       SetParErr(Int_t par,Double_t e=0)            {fParErrs[par] = e;}
  //
  Int_t      GetFirstParGloID()                     const {return fFirstParGloID;}
  Int_t      GetParGloID(Int_t par)                 const {return fFirstParGloID+par;}
  void       SetFirstParGloID(Int_t id)                   {fFirstParGloID=id;}
  //
 protected:
  void     SortSensors();
  void     CalcFree(Bool_t condFree=kFALSE);
  //
  // ------- dummies ---------
  AliAlgDet(const AliAlgDet&);
  AliAlgDet& operator=(const AliAlgDet&);
  //
 protected:
  //
  Int_t     fNDOFs;                      // number of DOFs free
  Int_t     fVolIDMin;                   // min volID for this detector (for sensors only)
  Int_t     fVolIDMax;                   // max volID for this detector (for sensors only)
  Int_t     fNSensors;                   // number of sensors (i.e. volID's)
  Int_t*    fSID2VolID;                  //[fNSensors] table of conversion from VolID to sid
  Int_t     fNProcPoints;                // total number of points processed
  //
  // Detector specific calibration degrees of freedom
  Int_t     fNCalibDOF;                 // number of calibDOFs for detector (preset)
  Int_t     fNCalibDOFFree;             // number of calibDOFs for detector (preset)
  ULong64_t fCalibDOF;                  // status of calib dof
  Int_t     fFirstParGloID;             // ID of the 1st parameter in the global results array
  Float_t*  fParVals;                   //! values of the fitted params
  Float_t*  fParErrs;                   //! errors of the fitted params
  Int_t*    fParLabs;                   //! labels for parameters
  //
  // Track selection
  Bool_t    fDisabled[AliAlgAux::kNTrackTypes];      // detector disabled/enabled in the track
  Bool_t    fObligatory[AliAlgAux::kNTrackTypes];    // detector must be present in the track
  ULong_t   fTrackFlagSel[AliAlgAux::kNTrackTypes];  // flag for track selection
  Int_t     fNPointsSel[AliAlgAux::kNTrackTypes];    // min number of points to require                 
  //
  Int_t     fUseErrorParam;          // signal that points need to be updated using track info, 0 - no
  Double_t  fAddError[2];            // additional error increment for measurement
  TObjArray fSensors;                // all sensors of the detector
  TObjArray fVolumes;                // all volumes of the detector  
  //
  // this is transient info
  Int_t     fNPoints;                //! number of points from this detector
  Int_t     fPoolNPoints;            //! number of points in the pool
  Int_t     fPoolFreePointID;        //! id of the last free point in the pool
  TObjArray fPointsPool;             //! pool of aligment points
  //
  AliAlgSteer* fAlgSteer;            // pointer to alignment steering object
  //
  ClassDef(AliAlgDet,1);             // base class for detector global alignment
};

//_____________________________________________________
inline Bool_t AliAlgDet::CheckFlags(const AliESDtrack* trc,Int_t trtype) const 
{
  // check if flags are ok
  return (trc->GetStatus()&fTrackFlagSel[trtype]) == fTrackFlagSel[trtype];
}

#endif
