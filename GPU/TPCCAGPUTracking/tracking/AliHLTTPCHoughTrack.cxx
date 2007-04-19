// @(#) $Id$
// origin: hough/AliL3HoughTrack.cxx,v 1.23 ue Mar 28 18:05:12 2006 UTC by alibrary

// Author: Anders Vestbo <mailto:vestbo@fi.uib.no>
//*-- Copyright &copy ALICE HLT Group

#include "AliHLTStdIncludes.h"

#include "AliHLTTPCLogging.h"
#include "AliHLTTPCHoughTrack.h"
#include "AliHLTTPCTransform.h"
#include "AliHLTTPCHoughTransformerRow.h"

#if __GNUC__ >= 3
using namespace std;
#endif

/** \class AliHLTTPCHoughTrack
<pre>
//_____________________________________________________________
// AliHLTTPCHoughTrack
//
// Track class for Hough tracklets
//
</pre>
*/

ClassImp(AliHLTTPCHoughTrack);


AliHLTTPCHoughTrack::AliHLTTPCHoughTrack() : AliHLTTPCTrack()
{
  //Constructor
  
  fWeight = 0;
  fMinDist=0;
  fDLine = 0;
  fPsiLine = 0;
  fIsHelix = true;
  fEtaIndex = -1;
  fEta = 0;
  ComesFromMainVertex(kTRUE);
}

AliHLTTPCHoughTrack::~AliHLTTPCHoughTrack()
{
  //dtor
}

void AliHLTTPCHoughTrack::Set(AliHLTTPCTrack *track)
{
  //Basically copy constructor
  AliHLTTPCHoughTrack *tpt = (AliHLTTPCHoughTrack*)track;
  SetTrackParameters(tpt->GetKappa(),tpt->GetPsi(),tpt->GetWeight());
  SetEtaIndex(tpt->GetEtaIndex());
  SetEta(tpt->GetEta());
  SetTgl(tpt->GetTgl());
  SetPsi(tpt->GetPsi());
  SetPterr(tpt->GetPterr());
  SetTglerr(tpt->GetTglerr());
  SetPsierr(tpt->GetPsierr());
  SetCenterX(tpt->GetCenterX());
  SetCenterY(tpt->GetCenterY());
  SetFirstPoint(tpt->GetFirstPointX(),tpt->GetFirstPointY(),tpt->GetFirstPointZ());
  SetLastPoint(tpt->GetLastPointX(),tpt->GetLastPointY(),tpt->GetLastPointZ());
  SetCharge(tpt->GetCharge());
  SetRowRange(tpt->GetFirstRow(),tpt->GetLastRow());
  SetSlice(tpt->GetSlice());
  SetHits(tpt->GetNHits(),(UInt_t *)tpt->GetHitNumbers());
  SetMCid(tpt->GetMCid());
  SetBinXY(tpt->GetBinX(),tpt->GetBinY(),tpt->GetSizeX(),tpt->GetSizeY());
  SetSector(tpt->GetSector());
  return;

//    fWeight = tpt->GetWeight();
//    fDLine = tpt->GetDLine();
//    fPsiLine = tpt->GetPsiLine();
//    SetNHits(tpt->GetWeight());
//    SetRowRange(tpt->GetFirstRow(),tpt->GetLastRow());
//    fIsHelix = false;
}

Int_t AliHLTTPCHoughTrack::Compare(const AliHLTTPCTrack *tpt) const
{
  //Compare 2 hough tracks according to their weight
  AliHLTTPCHoughTrack *track = (AliHLTTPCHoughTrack*)tpt;
  if(track->GetWeight() < GetWeight()) return 1;
  if(track->GetWeight() > GetWeight()) return -1;
  return 0;
}

void AliHLTTPCHoughTrack::SetEta(Double_t f)
{
  //Set eta, and calculate fTanl, which is the tan of dipangle

  fEta = f;
  Double_t theta = 2*atan(exp(-1.*fEta));
  Double_t dipangle = AliHLTTPCTransform::PiHalf() - theta;
  Double_t tgl = tan(dipangle);
  SetTgl(tgl);
}

void AliHLTTPCHoughTrack::UpdateToFirstRow()
{
  //Update the track parameters to the point where track cross
  //its first padrow.`
  
  //Get the crossing point with the first padrow:
  Float_t xyz[3];
  if(!GetCrossingPoint(GetFirstRow(),xyz))
    LOG(AliHLTTPCLog::kWarning,"AliHLTTPCHoughTrack::UpdateToFirstRow()","Track parameters")
      <<AliHLTTPCLog::kDec<<"Track does not cross padrow "<<GetFirstRow()<<" centerx "
      <<GetCenterX()<<" centery "<<GetCenterY()<<" Radius "<<GetRadius()<<" tgl "<<GetTgl()<<ENDLOG;
  
  //printf("Track with eta %f tgl %f crosses at x %f y %f z %f on padrow %d\n",GetEta(),GetTgl(),xyz[0],xyz[1],xyz[2],GetFirstRow());
  //printf("Before: first %f %f %f tgl %f center %f %f charge %d\n",GetFirstPointX(),GetFirstPointY(),GetFirstPointZ(),GetTgl(),GetCenterX(),GetCenterY(),GetCharge());
  
  Double_t radius = sqrt(xyz[0]*xyz[0] + xyz[1]*xyz[1]);

  //Get the track parameters
  
  /*
    Double_t x0    = GetR0() * cos(GetPhi0()) ;
    Double_t y0    = GetR0() * sin(GetPhi0()) ;
  */
  Double_t rc    = GetRadius();//fabs(GetPt()) / AliHLTTPCTransform::GetBFieldValue();
  Double_t tPhi0 = GetPsi() + GetCharge() * AliHLTTPCTransform::PiHalf() / abs(GetCharge()) ;
  Double_t xc    = GetCenterX();//x0 - rc * cos(tPhi0) ;
  Double_t yc    = GetCenterY();//y0 - rc * sin(tPhi0) ;
  
  //Check helix and cylinder intersect
  Double_t fac1 = xc*xc + yc*yc ;
  Double_t sfac = sqrt( fac1 ) ;
  
  if ( fabs(sfac-rc) > radius || fabs(sfac+rc) < radius ) {
    LOG(AliHLTTPCLog::kError,"AliHLTTPCHoughTrack::UpdateToFirstRow","Tracks")<<AliHLTTPCLog::kDec<<
      "Track does not intersect"<<ENDLOG;
    return;
  }
  
  //Find intersection
  Double_t fac2   = (radius*radius + fac1 - rc*rc) / (2.00 * radius * sfac ) ;
  Double_t phi    = atan2(yc,xc) + GetCharge()*acos(fac2) ;
  Double_t td     = atan2(radius*sin(phi) - yc,radius*cos(phi) - xc) ;
  
  //Intersection in z
  if ( td < 0 ) td = td + AliHLTTPCTransform::TwoPi();
  Double_t deltat = fmod((-GetCharge()*td + GetCharge()*tPhi0),AliHLTTPCTransform::TwoPi());
  if ( deltat < 0. ) deltat += AliHLTTPCTransform::TwoPi();
  else if ( deltat > AliHLTTPCTransform::TwoPi() ) deltat -= AliHLTTPCTransform::TwoPi();
  Double_t z = GetZ0() + rc * GetTgl() * deltat ;
  
  Double_t xExtra = radius * cos(phi) ;
  Double_t yExtra = radius * sin(phi) ;
  
  Double_t tPhi = atan2(yExtra-yc,xExtra-xc);
  
  //if ( tPhi < 0 ) tPhi += 2. * M_PI ;
  Double_t tPsi = tPhi - GetCharge() * AliHLTTPCTransform::PiHalf() / abs(GetCharge()) ;
  if ( tPsi > AliHLTTPCTransform::TwoPi() ) tPsi -= AliHLTTPCTransform::TwoPi() ;
  else if ( tPsi < 0. ) tPsi += AliHLTTPCTransform::TwoPi();
  
  //And finally, update the track parameters
  SetR0(radius);
  SetPhi0(phi);
  SetZ0(z);
  SetPsi(tPsi);
  SetFirstPoint(xyz[0],xyz[1],z);
  //printf("After: first %f %f %f tgl %f center %f %f charge %d\n",GetFirstPointX(),GetFirstPointY(),GetFirstPointZ(),GetTgl(),GetCenterX(),GetCenterY(),GetCharge());
  
  //printf("First point set %f %f %f\n",xyz[0],xyz[1],z);
  
  //Also, set the coordinates of the point where track crosses last padrow:
  GetCrossingPoint(GetLastRow(),xyz);
  SetLastPoint(xyz[0],xyz[1],xyz[2]);
  //printf("last point %f %f %f\n",xyz[0],xyz[1],xyz[2]);
}

void AliHLTTPCHoughTrack::SetTrackParameters(Double_t kappa,Double_t eangle,Int_t weight)
{
  //Set track parameters - sort of ctor
  fWeight = weight;
  fMinDist = 100000;
  SetKappa(kappa);
  Double_t pt = fabs(AliHLTTPCTransform::GetBFieldValue()/kappa);
  SetPt(pt);
  Double_t radius = 1/fabs(kappa);
  SetRadius(radius);
  SetFirstPoint(0,0,0);
  SetPsi(eangle); //Psi = emission angle when first point is vertex
  SetPhi0(0);     //not defined for vertex reference point
  SetR0(0);
  Double_t charge = -1.*kappa;
  SetCharge((Int_t)copysign(1.,charge));
  Double_t trackPhi0 = GetPsi() + charge*0.5*AliHLTTPCTransform::Pi()/fabs(charge);
  Double_t xc = GetFirstPointX() - GetRadius() * cos(trackPhi0) ;
  Double_t yc = GetFirstPointY() - GetRadius() * sin(trackPhi0) ;
  SetCenterX(xc);
  SetCenterY(yc);
  SetNHits(1); //just for the trackarray IO
  fIsHelix = true;
}

void AliHLTTPCHoughTrack::SetTrackParametersRow(Double_t alpha1,Double_t alpha2,Double_t eta,Int_t weight)
{
  //Set track parameters for HoughTransformerRow
  //This includes curvature,emission angle and eta
  Double_t psi = atan((alpha1-alpha2)/(AliHLTTPCHoughTransformerRow::GetBeta1()-AliHLTTPCHoughTransformerRow::GetBeta2()));
  Double_t kappa = 2.0*(alpha1*cos(psi)-AliHLTTPCHoughTransformerRow::GetBeta1()*sin(psi));
  SetTrackParameters(kappa,psi,weight);

  Double_t zovr;
  Double_t etaparam1 = AliHLTTPCHoughTransformerRow::GetEtaCalcParam1();
  Double_t etaparam2 = AliHLTTPCHoughTransformerRow::GetEtaCalcParam2();
  if(eta>0)
    zovr = (etaparam1 - sqrt(etaparam1*etaparam1 - 4.*etaparam2*eta))/(2.*etaparam2);
  else
    zovr = -1.*(etaparam1 - sqrt(etaparam1*etaparam1 + 4.*etaparam2*eta))/(2.*etaparam2);
  Double_t r = sqrt(1.+zovr*zovr);
  Double_t exacteta = 0.5*log((1+zovr/r)/(1-zovr/r));
  SetEta(exacteta);
}

void AliHLTTPCHoughTrack::SetLineParameters(Double_t psi,Double_t D,Int_t weight,Int_t *rowrange,Int_t /*ref_row*/)
{
  //Initialize a track piece, not yet a track
  //Used in case of straight line transformation

  //Transform line parameters to coordinate system of slice:
  
  //  D = D + fTransform->Row2X(ref_row)*cos(psi);

  fDLine = D;
  fPsiLine = psi;
  fWeight = weight;
  SetNHits(1);
  SetRowRange(rowrange[0],rowrange[1]);
  fIsHelix = false;
}

void AliHLTTPCHoughTrack::SetBestMCid(Int_t mcid,Double_t mindist)
{
  //Finds and set the closest mc label
  if(mindist < fMinDist)
    {
      fMinDist = mindist;
      SetMCid(mcid);
    }
}

void AliHLTTPCHoughTrack::GetLineCrossingPoint(Int_t padrow,Float_t *xy)
{
  //Returns the crossing point of the track with a given padrow
  if(fIsHelix)
    {
      printf("AliHLTTPCHoughTrack::GetLineCrossingPoint : Track is not a line\n");
      return;
    }

  Float_t xhit = AliHLTTPCTransform::Row2X(padrow) - AliHLTTPCTransform::Row2X(GetFirstRow());
  Float_t a = -1/tan(fPsiLine);
  Float_t b = fDLine/sin(fPsiLine);
  Float_t yhit = a*xhit + b;
  xy[0] = xhit;
  xy[1] = yhit;
}
