
#ifndef REDUCEDTRACK_H
#define REDUCEDTRACK_H

#include <TMath.h>
#include <TObject.h>

//_____________________________________________________________________
class ReducedTrack : public TObject {
  
  public:
    ReducedTrack();
    ReducedTrack(const ReducedTrack &c);
    virtual ~ReducedTrack();
  
    // getters
    unsigned short TrackId() const {return fTrackId;}
    float          Px()      const {return (fIsCartesian ? fP[0] : TMath::Abs(fP[0])*TMath::Cos(fP[1]));}
    float          Py()      const {return (fIsCartesian ? fP[1] : TMath::Abs(fP[0])*TMath::Sin(fP[1]));}
    float          Pz()      const {return (fIsCartesian ? fP[2] : TMath::Abs(fP[0])*TMath::SinH(fP[2]));}
    float          P()       const {return (fIsCartesian ? TMath::Sqrt(fP[0]*fP[0]+fP[1]*fP[1]+fP[2]*fP[2]) : TMath::Abs(fP[0])*TMath::CosH(fP[2]));}
    float          Phi()     const;
    float          Pt()      const {return (fIsCartesian ? TMath::Sqrt(fP[0]*fP[0]+fP[1]*fP[1]) : fP[0]);}
    float          Eta()     const;
    float          Theta()   const;
    
    float          Rapidity(float massAssumption) const;
    float          Energy(float massAssumption)   const {return TMath::Sqrt(massAssumption*massAssumption+P()*P());}
    
    int            Charge()              const {return fCharge;} 
    bool           IsCartesian()         const {return fIsCartesian;}
    bool           TestFlag(short iflag) const {return ((iflag<(8*sizeof(uint64_t))) ? fFlags&(uint64_t(1)<<iflag) : kFALSE);} 
    uint64_t       Flags()               const {return fFlags;}
    
    // setters
    void TrackId(unsigned short value)            {fTrackId=value;}
    void PxPyPz(float px, float py, float pz)     {fP[0]=px;fP[1]=py;fP[2]=pz;fIsCartesian=kTRUE;}
    void PtPhiEta(float pt, float phi, float eta) {fP[0]=pt;fP[1]=phi;fP[2]=eta;fIsCartesian=kFALSE;}
    void Charge(int ch)                           {fCharge=ch;}
    void ResetFlags()                             {fFlags=0;}
    void SetFlags(uint64_t flags)                 {fFlags=flags;}
    void SetFlag(short iflag)                     {if(iflag>=8*sizeof(uint64_t)) return; fFlags|=(uint64_t(1)<<iflag);}
    void UnsetFlag(short iflag)                   {if(iflag>=8*sizeof(uint64_t)) return; if(TestFlag(iflag)) fFlags^=(uint64_t(1)<<iflag);}
   
  protected:
    short    fTrackId;     // track id 
    float    fP[3];           // 3-momentum vector
    bool     fIsCartesian;    // if false then the 3-momentum vector is in spherical coordinates (pt,phi,eta)
    char     fCharge;         // electrical charge
    uint64_t fFlags;        // flags reserved for various operations during analysis
        
    ReducedTrack& operator= (const ReducedTrack &c);
    
    ClassDef(ReducedTrack, 1)
};

//_______________________________________________________________________________
inline float ReducedTrack::Phi() const {
  //
  // Return the azimuthal angle of this particle
  //
  if(!fIsCartesian) return fP[1];
  float phi=TMath::ATan2(fP[1],fP[0]); 
  if(phi>=0.0) 
    return phi;
  else 
    return (TMath::TwoPi()+phi);
}

//_______________________________________________________________________________
inline float ReducedTrack::Theta() const {
  //
  // Return the polar angle for this particle
  //
  float p=P(); 
  if(p>=1.0e-6) 
    return TMath::ACos(Pz()/p);
  else 
    return 0.0;
}

//_______________________________________________________________________________
inline float ReducedTrack::Eta() const {
  //
  // Return the pseudorapidity of this particle
  //
  if(!fIsCartesian) return fP[2];
  float eta = TMath::Tan(0.5*Theta());
  if(eta>1.0e-6) 
    return -1.0*TMath::Log(eta);
  else 
    return 0.0;
}

//_______________________________________________________________________________
inline float ReducedTrack::Rapidity(float massAssumption) const {
  //
  // Return the rapidity of this particle using a massAssumption
  //
  float e = Energy(massAssumption);
  float factor = e-Pz();
  if(TMath::Abs(factor)<1.0e-6) return -999.0;
  factor = (e+Pz())/factor;
  if(factor<1.0e-6) return -999.0;
  return 0.5*TMath::Log(factor);
}

#endif
