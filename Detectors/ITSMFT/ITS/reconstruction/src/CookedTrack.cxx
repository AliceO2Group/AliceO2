//-------------------------------------------------------------------------
//               Implementation of the ITSU track class
//             based on the "cooked covariance" approach
//-------------------------------------------------------------------------
#include "AliITSUTrackCooked.h"
#include "AliCluster.h"
#include "AliESDtrack.h"

ClassImp(AliITSUTrackCooked)

AliITSUTrackCooked::AliITSUTrackCooked(): 
AliKalmanTrack()
{
  //--------------------------------------------------------------------
  // This default constructor needs to be provided
  //--------------------------------------------------------------------
    for (Int_t i=0; i<2*AliITSUTrackerCooked::kNLayers; i++) {
        fIndex[i]=0;
    }
}

AliITSUTrackCooked::AliITSUTrackCooked(const AliITSUTrackCooked &t):
AliKalmanTrack(t)
{
    //--------------------------------------------------------------------
    // Copy constructor
    //--------------------------------------------------------------------
    for (Int_t i=0; i<2*AliITSUTrackerCooked::kNLayers; i++) {
        fIndex[i]=t.fIndex[i];
    }
}

AliITSUTrackCooked::AliITSUTrackCooked(const AliESDtrack &t):
AliKalmanTrack()
{
    //--------------------------------------------------------------------
    // Constructior from an ESD track
    //--------------------------------------------------------------------
    Set(t.GetX(), t.GetAlpha(), t.GetParameter(), t.GetCovariance());
    SetLabel(t.GetITSLabel());
    SetChi2(t.GetITSchi2());
    SetNumberOfClusters(t.GetITSclusters(fIndex));
}

AliITSUTrackCooked &AliITSUTrackCooked::operator=(const AliITSUTrackCooked &o){
    if (this != &o) {
       AliKalmanTrack::operator=(o);
       for (Int_t i=0; i<2*AliITSUTrackerCooked::kNLayers; i++)
           fIndex[i]=o.fIndex[i];
    }
    return *this;
}

AliITSUTrackCooked::~AliITSUTrackCooked()
{
  //--------------------------------------------------------------------
  // Virtual destructor
  //--------------------------------------------------------------------
}

Int_t AliITSUTrackCooked::Compare(const TObject *o) const {
  //-----------------------------------------------------------------
  // This function compares tracks according to the their curvature
  //-----------------------------------------------------------------
  const AliITSUTrackCooked *t=(const AliITSUTrackCooked*)o;
  Double_t co=TMath::Abs(t->OneOverPt());
  Double_t c =TMath::Abs(OneOverPt());
  //Double_t co=t->GetSigmaY2()*t->GetSigmaZ2();
  //Double_t c =GetSigmaY2()*GetSigmaZ2();
  if (c>co) return 1;
  else if (c<co) return -1;
  return 0;
}

void AliITSUTrackCooked::ResetClusters() {
  //------------------------------------------------------------------
  // Reset the array of attached clusters.
  //------------------------------------------------------------------
  for (Int_t i=0; i<2*AliITSUTrackerCooked::kNLayers; i++) fIndex[i]=-1;
  SetChi2(0.); 
  SetNumberOfClusters(0);
} 

void AliITSUTrackCooked::SetClusterIndex(Int_t l, Int_t i)
{
    //--------------------------------------------------------------------
    // Set the cluster index
    //--------------------------------------------------------------------
    Int_t idx = (l<<28) + i;
    Int_t n=GetNumberOfClusters();
    fIndex[n]=idx;
    SetNumberOfClusters(n+1);
}

Double_t AliITSUTrackCooked::GetPredictedChi2(const AliCluster *c) const {
  //-----------------------------------------------------------------
  // This function calculates a predicted chi2 increment.
  //-----------------------------------------------------------------
  Double_t p[2]={c->GetY(), c->GetZ()};
  Double_t cov[3]={c->GetSigmaY2(), 0., c->GetSigmaZ2()};
  return AliExternalTrackParam::GetPredictedChi2(p,cov);
}

Bool_t AliITSUTrackCooked::PropagateTo(Double_t xk, Double_t t,Double_t x0rho) {
  //------------------------------------------------------------------
  // This function propagates a track
  // t is the material thicknes in units X/X0
  // x0rho is the material X0*density
  //------------------------------------------------------------------
  Double_t xOverX0,xTimesRho; 
  xOverX0 = t; xTimesRho = t*x0rho;
  if (!CorrectForMeanMaterial(xOverX0,xTimesRho,GetMass(),kTRUE)) return kFALSE;

  Double_t bz=GetBz();
  if (!AliExternalTrackParam::PropagateTo(xk,bz)) return kFALSE;
  //Double_t b[3]; GetBxByBz(b);
  //if (!AliExternalTrackParam::PropagateToBxByBz(xk,b)) return kFALSE;

  return kTRUE;
}

Bool_t AliITSUTrackCooked::Update(const AliCluster *c, Double_t chi2, Int_t idx)
{
  //--------------------------------------------------------------------
  // Update track params
  //--------------------------------------------------------------------
  Double_t p[2]={c->GetY(), c->GetZ()};
  Double_t cov[3]={c->GetSigmaY2(), c->GetSigmaYZ(), c->GetSigmaZ2()};

  if (!AliExternalTrackParam::Update(p,cov)) return kFALSE;

  Int_t n=GetNumberOfClusters();
  fIndex[n]=idx;
  SetNumberOfClusters(n+1);
  SetChi2(GetChi2()+chi2);

  return kTRUE;
}

Bool_t AliITSUTrackCooked::
GetPhiZat(Double_t r, Double_t &phi, Double_t &z) const {
  //------------------------------------------------------------------
  // This function returns the global cylindrical (phi,z) of the track 
  // position estimated at the radius r. 
  // The track curvature is neglected.
  //------------------------------------------------------------------
  Double_t d=GetD(0., 0., GetBz());
  if (TMath::Abs(d) > r) {
    if (r>1e-1) return kFALSE;
    r = TMath::Abs(d);
  }

  Double_t rcurr=TMath::Sqrt(GetX()*GetX() + GetY()*GetY());
  if (TMath::Abs(d) > rcurr) return kFALSE;
  Double_t globXYZcurr[3]; GetXYZ(globXYZcurr); 
  Double_t phicurr=TMath::ATan2(globXYZcurr[1],globXYZcurr[0]);

  if (GetX()>=0.) {
    phi=phicurr+TMath::ASin(d/r)-TMath::ASin(d/rcurr);
  } else {
    phi=phicurr+TMath::ASin(d/r)+TMath::ASin(d/rcurr)-TMath::Pi();
  }

  // return a phi in [0,2pi[ 
  if (phi<0.) phi+=2.*TMath::Pi();
  else if (phi>=2.*TMath::Pi()) phi-=2.*TMath::Pi();
  z=GetZ()+GetTgl()*(TMath::Sqrt((r-d)*(r+d))-TMath::Sqrt((rcurr-d)*(rcurr+d)));

  return kTRUE;
}

Bool_t
AliITSUTrackCooked::IsBetter(const AliKalmanTrack *best, Double_t maxChi2)
const {
  Int_t ncl=GetNumberOfClusters();
  Int_t nclb=best->GetNumberOfClusters();

  if (ncl >= nclb) {
     Double_t chi2=GetChi2();
     if (chi2 < maxChi2) {
        if (ncl > nclb || chi2 < best->GetChi2()) {
	   return kTRUE;
        }
     }
  }
  return kFALSE;
}
