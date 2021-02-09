#ifndef ALIALGRES_H
#define ALIALGRES_H

#include <TObject.h>
#include <TMath.h>
class AliAlgTrack;

/*--------------------------------------------------------
  Container for control residuals
  -------------------------------------------------------*/

// Author: ruben.shahoyan@cern.ch



class AliAlgRes: public TObject
{
 public:
  enum {kCosmicBit=BIT(14),kVertexBit=BIT(15),kKalmanDoneBit=BIT(16)};
  //
  AliAlgRes();
  virtual ~AliAlgRes();
  //
  void     SetRun(int r)                              {fRun = r;}
  void     SetBz(float v)                             {fBz = v;}
  void     SetTimeStamp(UInt_t v)                     {fTimeStamp = v;}
  void     SetTrackID(UInt_t v)                       {fTrackID = v;}
  void     SetNPoints(Int_t n)                        {fNPoints=n; Resize(n);}
  //
  Bool_t   IsCosmic()                           const {return TestBit(kCosmicBit);}
  Bool_t   HasVertex()                          const {return TestBit(kVertexBit);}
  void     SetCosmic(Bool_t v=kTRUE)                  {SetBit(kCosmicBit,v);}
  void     SetHasVertex(Bool_t v=kTRUE)               {SetBit(kVertexBit,v);}
  //
  Bool_t   GetKalmanDone()                      const {return TestBit(kKalmanDoneBit);}
  void     SetKalmanDone(Bool_t v=kTRUE)              {SetBit(kKalmanDoneBit,v);}
  //
  Int_t    GetRun()                             const {return fRun;}
  Float_t  GetBz()                              const {return fBz;}
  UInt_t   GetTimeStamp()                       const {return fTimeStamp;}
  UInt_t   GetTrackID()                         const {return fTrackID;}    
  Int_t    GetNPoints()                         const {return fNPoints;}    
  Int_t    GetNBook()                           const {return fNBook;}      
  Float_t  GetChi2()                            const {return fChi2;}       
  Float_t  GetChi2Ini()                         const {return fChi2Ini;}    
  Float_t  GetChi2K()                           const {return fChi2K;}
  Float_t  GetQ2Pt()                            const {return fQ2Pt;}       
  Float_t  GetX(int i)                          const {return fX[i];}      
  Float_t  GetY(int i)                          const {return fY[i];}      
  Float_t  GetZ(int i)                          const {return fZ[i];}      
  Float_t  GetSnp(int i)                        const {return fSnp[i];}        
  Float_t  GetTgl(int i)                        const {return fTgl[i];}        
  Float_t  GetAlpha(int i)                      const {return fAlpha[i];}      
  Float_t  GetDY(int i)                         const {return fDY[i];}      
  Float_t  GetDZ(int i)                         const {return fDZ[i];}
  Float_t  GetDYK(int i)                        const {return fDYK[i];}
  Float_t  GetDZK(int i)                        const {return fDZK[i];}
  //
  Float_t  GetSigY2K(int i)                     const {return fSigY2K[i];}
  Float_t  GetSigYZK(int i)                     const {return fSigYZK[i];}
  Float_t  GetSigZ2K(int i)                     const {return fSigZ2K[i];}
  Float_t  GetSigmaYK(int i)                    const {return TMath::Sqrt(fSigY2K[i]);}      
  Float_t  GetSigmaZK(int i)                    const {return TMath::Sqrt(fSigZ2K[i]);}      
  //
  Float_t  GetSigY2(int i)                      const {return fSigY2[i];}
  Float_t  GetSigYZ(int i)                      const {return fSigYZ[i];}
  Float_t  GetSigZ2(int i)                      const {return fSigZ2[i];}
  Float_t  GetSigmaY(int i)                     const {return TMath::Sqrt(fSigY2[i]);}      
  Float_t  GetSigmaZ(int i)                     const {return TMath::Sqrt(fSigZ2[i]);}      
  //
  Float_t  GetSigY2Tot(int i)                   const {return fSigY2K[i]+fSigY2[i];}
  Float_t  GetSigYZTot(int i)                   const {return fSigYZK[i]+fSigYZ[i];}
  Float_t  GetSigZ2Tot(int i)                   const {return fSigZ2K[i]+fSigZ2[i];}
  Float_t  GetSigmaYTot(int i)                  const {return TMath::Sqrt(GetSigY2Tot(i));}      
  Float_t  GetSigmaZTot(int i)                  const {return TMath::Sqrt(GetSigZ2Tot(i));}      
  //
  Int_t    GetVolID(int i)                      const {return fVolID[i];}
  //
  Float_t  GetXLab(int i)                       const;
  Float_t  GetYLab(int i)                       const;
  Float_t  GetZLab(int i)                       const;
  //
  Bool_t       FillTrack(AliAlgTrack* trc, Bool_t doKalman=kTRUE);
  void         Resize(Int_t n);
  virtual void Clear(const Option_t *opt="");
  virtual void Print(const Option_t *opt="re") const;
  //
 protected:
  //
  // -------- dummies --------
  AliAlgRes(const AliAlgRes&);
  AliAlgRes& operator=(const AliAlgRes&);
  //
 protected:
  //
  Int_t    fRun;                    // run
  Float_t  fBz;                     // field
  UInt_t   fTimeStamp;              // event time
  UInt_t   fTrackID;                // track ID
  Int_t    fNPoints;                // n meas points
  Int_t    fNBook;                  //! booked lenfth
  Float_t  fChi2;                   //  chi2 after solution
  Float_t  fChi2Ini;                //  chi2 before solution
  Float_t  fChi2K;                  //  chi2 from kalman
  Float_t  fQ2Pt;                   //  Q/Pt at reference point
  Float_t* fX;                      //[fNPoints] tracking X of cluster
  Float_t* fY;                      //[fNPoints] tracking Y of cluster
  Float_t* fZ;                      //[fNPoints] tracking Z of cluster
  Float_t* fSnp;                    //[fNPoints] track Snp
  Float_t* fTgl;                    //[fNPoints] track Tgl
  Float_t* fAlpha;                  //[fNPoints] track alpha
  Float_t* fDY;                     //[fNPoints] Y residual (track - meas)
  Float_t* fDZ;                     //[fNPoints] Z residual (track - meas)
  Float_t* fDYK;                    //[fNPoints] Y residual (track - meas) Kalman
  Float_t* fDZK;                    //[fNPoints] Z residual (track - meas) Kalman
  Float_t* fSigY2;                  //[fNPoints] Y err^2
  Float_t* fSigYZ;                  //[fNPoints] YZ err
  Float_t* fSigZ2;                  //[fNPoints] Z err^2
  Float_t* fSigY2K;                 //[fNPoints] Y err^2 of Kalman track smoothing
  Float_t* fSigYZK;                 //[fNPoints] YZ err  of Kalman track smoothing
  Float_t* fSigZ2K;                 //[fNPoints] Z err^2 of Kalman track smoothing
  Int_t*   fVolID;                  //[fNPoints] volume id (0 for vertex constraint)
  Int_t*   fLabel;                  //[fNPoints] label of the volume
  //
  ClassDef(AliAlgRes,2);
};

#endif
