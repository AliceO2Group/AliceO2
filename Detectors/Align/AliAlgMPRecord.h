#ifndef ALIALGMPRECORD_H
#define ALIALGMPRECORD_H

#include <TObject.h>
class AliAlgTrack;

/*--------------------------------------------------------
  Millepede record in root format (can be converted to proper 
  pede binary format.
  -------------------------------------------------------*/

// Author: ruben.shahoyan@cern.ch


class AliAlgMPRecord : public TObject
{
 public:
  enum {kCosmicBit=BIT(14)};
  //
  AliAlgMPRecord();
  virtual ~AliAlgMPRecord();
  //
  Int_t        GetRun()                const {return GetUniqueID();}
  void         SetRun(Int_t r)               {SetUniqueID(r);}
  UInt_t       GetTimeStamp()          const {return fTimeStamp;}
  void         SetTimeStamp(UInt_t t)        {fTimeStamp = t;}
  UInt_t       GetTrackID()            const {return fTrackID;}
  void         SetTrackID(UInt_t t)          {fTrackID = t;}
  Bool_t       IsCosmic()              const {return TestBit(kCosmicBit);}
  void         SetCosmic(Bool_t v=kTRUE)     {SetBit(kCosmicBit,v);}
  //
  Int_t        GetNVarGlo()            const {return fNVarGlo;}
  void         SetNVarGlo(int n)             {fNVarGlo = n;}
  //
  Int_t        GetNResid()             const {return fNResid;}
  Int_t        GetNVarLoc()            const {return fNVarLoc;}
  //
  Int_t        GetNDLoc(int id)        const {return fNDLoc[id];}
  Int_t        GetNDGlo(int id)        const {return fNDGlo[id];}
  Int_t        GetVolID(int id)        const {return fVolID ? fVolID[id]-1:-1;}
  Float_t      GetResid(int id)        const {return fResid[id];}
  Float_t      GetResErr(int id)       const {return fResErr[id];}
  //
  Float_t      GetChi2Ini()            const {return fChi2Ini;}
  Float_t      GetQ2Pt()               const {return fQ2Pt;}
  Float_t      GetTgl()                const {return fTgl;}
  Int_t        GetNDLocTot()           const {return fNDLocTot;}
  Int_t        GetNDGloTot()           const {return fNDGloTot;}
  const Float_t* GetArrGlo()           const {return fDGlo;}
  const Float_t* GetArrLoc()           const {return fDLoc;}
  const Short_t* GetArrLabLoc()        const {return fIDLoc;}
  const Int_t*   GetArrLabGlo()        const {return fIDGlo;}
  //
  Bool_t       FillTrack(const AliAlgTrack* trc, const Int_t *id2Lab=0);
  void         DummyRecord(Float_t res, Float_t err, Float_t dGlo, Int_t labGlo);
  //
  void         Resize(Int_t nresid, Int_t nloc, Int_t nglo);
  //
  virtual void Clear(const Option_t *opt="");
  virtual void Print(const Option_t *opt="") const;
  //
 protected:
  //
  // ------- dummies --------
  AliAlgMPRecord(const AliAlgMPRecord&);
  AliAlgMPRecord& operator=(const AliAlgMPRecord&);
    //
 protected:
  //
  UInt_t       fTrackID;         // track in the event
  UInt_t       fTimeStamp;       // event time stamp
  Int_t        fNResid;          // number of residuals for the track (=2 npoints)
  Int_t        fNVarLoc;         // number of local variables for the track
  Int_t        fNVarGlo;         // number of global variables defined
  Int_t        fNDLocTot;        // total number of non-zero local derivatives
  Int_t        fNDGloTot;        // total number of non-zero global derivatives
  Int_t        fNMeas;           // number of measured points
  Float_t      fChi2Ini;         // chi2 of initial kalman fit
  Float_t      fQ2Pt;            // q/pt at ref point
  Float_t      fTgl;             // dip angle at ref point
  //
  Short_t*     fNDLoc;           //[fNResid] number of non-0 local derivatives per residual
  Int_t*       fNDGlo;           //[fNResid] number of non-0 global derivatives per residual
  Int_t*       fVolID;           //[fNResid] volume id + 1 (0 - not a volume)
  Float_t*     fResid;           //[fNResid] residuals
  Float_t*     fResErr;          //[fNResid] error associated to residual
  //
  Short_t*     fIDLoc;           //[fNDLocTot] ID of local variables for non-0 local derivatives
  Int_t*       fIDGlo;           //[fNDGloTot] ID of global variables for non-0 global derivatives  
  Float_t*     fDLoc;            //[fNDLocTot] non-0 local derivatives
  Float_t*     fDGlo;            //[fNDGloTot] non-0 global derivatives
  //
  // aux info
  Int_t        fNResidBook;      //! number of slots booked for residuals
  Int_t        fNDLocTotBook;    //! number of slots booked for local derivatives
  Int_t        fNDGloTotBook;    //! number of slots booked for global derivatives
  //
  ClassDef(AliAlgMPRecord,4);
};


#endif
