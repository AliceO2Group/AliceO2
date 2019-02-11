#ifndef ALIHMPIDPIDRESPONSE_H
#define ALIHMPIDPIDRESPONSE_H

/* Copyright(c) 1998-1999, ALICE Experiment at CERN, All rights reserved. *
 * See cxx source for full Copyright notice                               */

//***********************************************************
// Class AliHMPIDPIDResponse
//
// HMPID class to perfom particle identification
// 
// Author: G. Volpe, giacomo.volpe@cern.ch
//***********************************************************


#include <TNamed.h>          //base class
#include <TVector3.h>
#include <TVector2.h>

#include "AliPID.h"
        
class AliVTrack;
class TObjArray;
class TGeoHMatrix;

class AliHMPIDPIDResponse : public TNamed 
{
public : 
             AliHMPIDPIDResponse();    //ctor
             AliHMPIDPIDResponse(const AliHMPIDPIDResponse& c);                //copy constructor
             AliHMPIDPIDResponse &operator=(const AliHMPIDPIDResponse& c);     //dummy assignment operator
	     virtual ~AliHMPIDPIDResponse();
    
    enum EChamberData{kMinCh=0,kMaxCh=6,kMinPc=0,kMaxPc=5};      //Segmenation
    enum EPadxData{kPadPcX=80,kMinPx=0,kMaxPx=79,kMaxPcx=159};   //Segmentation structure along x
    enum EPadyData{kPadPcY=48,kMinPy=0,kMaxPy=47,kMaxPcy=143};   //Segmentation structure along y 
    
    Double_t GetExpectedSignal  (const AliVTrack *vTrk, AliPID::EParticleType specie                 ) const;
    Double_t GetExpectedSigma   (const AliVTrack *vTrk, AliPID::EParticleType specie                 ) const;                                                                     //Find the sigma for a given ThetaCerTh
    Double_t GetNumberOfSigmas  (const AliVTrack *vTrk, AliPID::EParticleType specie                 ) const;                                                                     //Find the expected Cherenkov angle for a given track
    void     GetProbability     (const AliVTrack *vTrk, Int_t nSpecies,Double_t *prob                ) const;                                                                     //Find the PID probability array
    Double_t GetSignalDelta     (const AliVTrack *vTrk, AliPID::EParticleType specie, Bool_t ratio=kFALSE) const;    
    void     Propagate          (const TVector3  dir,   TVector3 &pos,  Double_t z                   ) const;                                                                     //propagate photon alogn the line  
    void     Refract            (TVector3 &dir,         Double_t n1,    Double_t n2                  ) const;                                                                     //refract photon on the boundary
    TVector2 TracePhot          (Double_t xRa, Double_t yRa,  Double_t thRa, Double_t phRa, Double_t ckovThe,Double_t ckovPhi) const;                                             //trace photon created by track to PC 
    void     Trs2Lors           (Double_t thRa, Double_t phRa, TVector3 dirCkov,      Double_t &thetaCer,Double_t &phiCer) const;                                                 //TRS to LORS
    TVector2 TraceForward       (Double_t xRa, Double_t yRa, TVector3 dirCkov                        ) const;                                                                     //tracing forward a photon from (x,y) to PC
    void     SetTrack           (Double_t xRad,         Double_t yRad,  Double_t theta,Double_t phi  )       {fTrkDir.SetMagThetaPhi(1,theta,phi);  fTrkPos.Set(xRad,yRad);}      //set track parameter at RAD
    Double_t RadThick           (                                                                    ) const {return 1.5;}                                                        //Radiator thickness
    Double_t WinThick           (                                                                    ) const {return 0.5;}                                                        //Window thickness
    Double_t GapThick           (                                                                    ) const {return 8.0;}                                                        //Proximity gap thicknes
    Double_t GetRefIdx          (                                                                    ) const {return fRefIdx;}                                                    //running refractive index
    Double_t WinIdx             (                                                                    ) const {return 1.5787;}                                                     //Mean refractive index of WIN material (SiO2) 
    Double_t GapIdx             (                                                                    ) const {return 1.0005;}                                                     //Mean refractive index of GAP material (CH4)
    static Bool_t  IsInside     (Float_t x,Float_t y,Float_t d=0                                     )       {return  x>-d&&y>-d&&x<fgkMaxPcX[kMaxPc]+d&&y<fgkMaxPcY[kMaxPc]+d; } //is point inside chamber boundaries?
    static Bool_t  IsInDead     (Float_t x,Float_t y                                                 );                                                                           //is the point in a dead area?
    static Float_t SizeAllX     (                                                                    )       {return fgAllX;}                                                     //all PCs size x, [cm]        
    static Float_t SizeAllY     (                                                                    )       {return fgAllY;}                                                     //all PCs size y, [cm]    
    static void    IdealPosition(Int_t iCh,TGeoHMatrix *m                                            );                                                                           //ideal position of given chamber 
        
    Double_t SigLoc             (Double_t trkTheta,Double_t trkPhi,Double_t ckovTh,Double_t ckovPh,Double_t beta) const;                                                          //error due to cathode segmetation
    Double_t SigGeom            (Double_t trkTheta,Double_t trkPhi,Double_t ckovTh,Double_t ckovPh,Double_t beta) const;                                                          //error due to unknown photon origin
    Double_t SigCrom            (Double_t trkTheta,Double_t ckovTh,Double_t ckovPh,Double_t beta                ) const;                                                          //error due to unknonw photon energy
    Double_t Sigma2             (Double_t trkTheta,Double_t trkPhi,Double_t ckovTh,Double_t ckovPh              ) const;                                                          //photon candidate sigma^2  
    Double_t GetNMean           (const AliVTrack *vTrk                                                          ) const;
    static Double_t SigmaCorrFact(Int_t iPart, Double_t occupancy                                               )      ;                                                          //correction factor for theoretical resolution
    
    void SetRefIndexArray       (TObjArray *array                                                     )       {fRefIndexArray = array;}
    TObjArray* GetRefIndexArray (                                                                     ) const {return fRefIndexArray;}

//
private:
        
    Double_t ExpectedSignal     (const AliVTrack *vTrk, Double_t nmean, AliPID::EParticleType specie ) const;
    Double_t ExpectedSigma      (const AliVTrack *vTrk, Double_t nmean, AliPID::EParticleType specie ) const;                                                                     //Find the sigma for a given ThetaCerTh
    
protected:

  static /*const*/ Float_t fgkMinPcX[6];                               //limits PC
  static /*const*/ Float_t fgkMinPcY[6];                               //limits PC
  static /*const*/ Float_t fgkMaxPcX[6];                               //limits PC
  static /*const*/ Float_t fgkMaxPcY[6]; 
          
  static Float_t fgCellX, fgCellY, fgPcX, fgPcY, fgAllX, fgAllY;       //definition of HMPID geometric parameters 
        
  TGeoHMatrix *fM[7];                                                  //pointers to matrices defining HMPID chambers rotations-translations
  
  Double_t  fRefIdx;                                                   //running refractive index of C6F14
  TVector3  fTrkDir;                                                   //track direction in LORS at RAD
  TVector2  fTrkPos;                                                   //track positon in LORS at RAD
  TObjArray *fRefIndexArray;                                           //array of refracive index funxtion;
  
  ClassDef(AliHMPIDPIDResponse,1)
};
#endif // #ifdef AliHMPIDPIDResponse_cxx

