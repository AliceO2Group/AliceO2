#ifndef ALIALGRESFAST_H
#define ALIALGRESFAST_H

#include <TObject.h>

/*--------------------------------------------------------
  Container for control fast residuals evaluated via derivatives
  -------------------------------------------------------*/

// Author: ruben.shahoyan@cern.ch



class AliAlgResFast: public TObject
{
 public:
  enum {kCosmicBit=BIT(14),kVertexBit=BIT(15)};
  //
  AliAlgResFast();
  virtual ~AliAlgResFast();
  //
  void     SetNPoints(Int_t n)                        {fNPoints=n; Resize(n);}
  void     SetNMatSol(Int_t n)                        {fNMatSol = n;}
  //
  void     SetChi2(float v)                           {fChi2 = v;}
  Float_t  GetChi2()                            const {return fChi2;}
  //
  void     SetChi2Ini(float v)                        {fChi2Ini = v;}
  Float_t  GetChi2Ini()                         const {return fChi2Ini;}
  //
  Bool_t   IsCosmic()                           const {return TestBit(kCosmicBit);}
  Bool_t   HasVertex()                          const {return TestBit(kVertexBit);}
  void     SetCosmic(Bool_t v=kTRUE)                  {SetBit(kCosmicBit,v);}
  void     SetHasVertex(Bool_t v=kTRUE)               {SetBit(kVertexBit,v);}
  //
  Int_t    GetNPoints()                         const {return fNPoints;}
  Int_t    GetNMatSol()                         const {return fNMatSol;}
  Int_t    GetNBook()                           const {return fNBook;}      
  Float_t  GetD0(int i)                         const {return fD0[i];}      
  Float_t  GetD1(int i)                         const {return fD1[i];}      
  Float_t  GetSig0(int i)                       const {return fSig0[i];}      
  Float_t  GetSig1(int i)                       const {return fSig1[i];}      
  Int_t    GetVolID(int i)                      const {return fVolID[i];}
  Int_t    GetLabel(int i)                      const {return fLabel[i];}
  //
  Float_t* GetTrCor()                           const {return (Float_t*)fTrCorr;}
  Float_t* GetD0()                              const {return (Float_t*)fD0;}
  Float_t* GetD1()                              const {return (Float_t*)fD1;}
  Float_t* GetSig0()                            const {return (Float_t*)fSig0;}
  Float_t* GetSig1()                            const {return (Float_t*)fSig1;}
  Int_t*   GetVolID()                           const {return (Int_t*)fVolID;}
  Int_t*   GetLaber()                           const {return (Int_t*)fLabel;}
  Float_t* GetSolMat()                          const {return (Float_t*)fSolMat;}
  Float_t* GetMatErr()                          const {return (Float_t*)fMatErr;}
  //
  void     SetResSigMeas(int ip, int ord, float res, float sig);
  void     SetMatCorr(int id, float res, float sig);
  void     SetLabel(int ip, Int_t lab, Int_t vol);
  //
  void         Resize(Int_t n);
  virtual void Clear(const Option_t *opt="");
  virtual void Print(const Option_t *opt="") const;
  //
 protected:
  //
  // -------- dummies --------
  AliAlgResFast(const AliAlgResFast&);
  AliAlgResFast& operator=(const AliAlgResFast&);
  //
 protected:
  //
  Int_t    fNPoints;                // n meas points
  Int_t    fNMatSol;                // n local params - ExtTrPar corrections
  Int_t    fNBook;                  //! booked lenfth
  Float_t  fChi2;                   // chi2
  Float_t  fChi2Ini;                // chi2 before local fit
  //  
  Float_t  fTrCorr[5];              //  correction to ExternalTrackParam
  Float_t* fD0;                     //[fNPoints] 1st residual (track - meas)
  Float_t* fD1;                     //[fNPoints] 2ns residual (track - meas)
  Float_t* fSig0;                   //[fNPoints] ort. error 0
  Float_t* fSig1;                   //[fNPoints] ort. errir 1
  Int_t*   fVolID;                  //[fNPoints] volume id (0 for vertex constraint)
  Int_t*   fLabel;                  //[fNPoints] label of the volume
  //
  Float_t* fSolMat;                 //[fNMatSol] // material corrections
  Float_t* fMatErr;                 //[fNMatSol] // material corrections errors
  //
  ClassDef(AliAlgResFast,1);
};

#endif
