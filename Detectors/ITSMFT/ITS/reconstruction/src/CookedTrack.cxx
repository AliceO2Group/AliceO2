/// \file CookedTrack.cxx
/// \brief Implementation of the ITS cooked track
/// \author iouri.belikov@cern.ch

#include "ITSReconstruction/CookedTrack.h"
#include "ITSReconstruction/Cluster.h"

ClassImp(AliceO2::ITS::CookedTrack)

  using namespace AliceO2::ITS;

CookedTrack::CookedTrack() : TrackParCov(), mMass(0.14), mChi2(-1.)
{
  //--------------------------------------------------------------------
  // This default constructor needs to be provided
  //--------------------------------------------------------------------
}

CookedTrack::CookedTrack(float x,float alpha, const float *par, const float *cov) : TrackParCov(x,alpha,par,cov), mMass(0.14), mChi2(-1.)
{
  //--------------------------------------------------------------------
  // Main constructor
  //--------------------------------------------------------------------
}

CookedTrack::CookedTrack(const CookedTrack& t) : TrackParCov(t), mMass(t.mMass), mChi2(t.mChi2), mIndex(t.mIndex)
{
  //--------------------------------------------------------------------
  // Copy constructor
  //--------------------------------------------------------------------
}

CookedTrack::~CookedTrack()
{
  //--------------------------------------------------------------------
  // Virtual destructor
  //--------------------------------------------------------------------
}

Int_t CookedTrack::compare(const CookedTrack* o) const
{
  //-----------------------------------------------------------------
  // This function compares tracks according to the their curvature
  //-----------------------------------------------------------------
  Double_t co = TMath::Abs(o->GetPt());
  Double_t c = TMath::Abs(GetPt());
  // Double_t co=t->GetSigmaY2()*t->GetSigmaZ2();
  // Double_t c =GetSigmaY2()*GetSigmaZ2();
  if (c > co)
    return -1;
  else if (c < co)
    return 1;
  return 0;
}

void CookedTrack::resetClusters()
{
  //------------------------------------------------------------------
  // Reset the array of attached clusters.
  //------------------------------------------------------------------
  mChi2 = -1.;
  mIndex.clear();
}

void CookedTrack::setClusterIndex(Int_t l, Int_t i)
{
  //--------------------------------------------------------------------
  // Set the cluster index
  //--------------------------------------------------------------------
  Int_t idx = (l << 28) + i;
  mIndex.push_back(idx);
}

Double_t CookedTrack::getPredictedChi2(const Cluster* c) const
{
  //-----------------------------------------------------------------
  // This function calculates a predicted chi2 increment.
  //-----------------------------------------------------------------
  float p[2] = { c->getY(), c->getZ() };
  float cov[3] = { c->getSigmaY2(), c->getSigmaYZ(), c->getSigmaZ2() };
  return TrackParCov::GetPredictedChi2(p, cov);
}

Bool_t CookedTrack::propagateTo(Double_t xk, Double_t bz, Double_t t, Double_t x0rho)
{
  //------------------------------------------------------------------
  // This function propagates a track
  // t is the material thicknes in units X/X0
  // x0rho is the material X0*density
  //------------------------------------------------------------------
  float xOverX0 = float(t), xTimesRho = float(t * x0rho), mass = float(mMass);
  if (!CorrectForMaterial(xOverX0, xTimesRho, mass, kTRUE))
    return kFALSE;

  if (!TrackParCov::PropagateTo(float(xk), float(bz)))
    return kFALSE;

  return kTRUE;
}

Bool_t CookedTrack::update(const Cluster* c, Double_t chi2, Int_t idx)
{
  //--------------------------------------------------------------------
  // Update track params
  //--------------------------------------------------------------------
  float p[2] = { c->getY(), c->getZ() };
  float cov[3] = { c->getSigmaY2(), c->getSigmaYZ(), c->getSigmaZ2() };

  if (!TrackParCov::Update(p, cov))
    return kFALSE;

  mChi2 += chi2;
  mIndex.push_back(idx);

  return kTRUE;
}

/*
Bool_t CookedTrack::getPhiZat(Double_t r, Double_t &phi, Double_t &z) const
{
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
*/

Bool_t CookedTrack::isBetter(const CookedTrack* best, Double_t maxChi2) const
{
  Int_t ncl = getNumberOfClusters();
  Int_t nclb = best->getNumberOfClusters();

  if (ncl >= nclb) {
    Double_t chi2 = getChi2();
    if (chi2 < maxChi2) {
      if (ncl > nclb || chi2 < best->getChi2()) {
        return kTRUE;
      }
    }
  }
  return kFALSE;
}
