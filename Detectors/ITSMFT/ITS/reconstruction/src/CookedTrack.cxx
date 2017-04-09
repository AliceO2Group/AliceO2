/// \file CookedTrack.cxx
/// \brief Implementation of the ITS cooked track
/// \author iouri.belikov@cern.ch

#include <TMath.h>

#include "DetectorsBase/Constants.h"
#include "ITSReconstruction/Cluster.h"
#include "ITSReconstruction/CookedTrack.h"

ClassImp(o2::ITS::CookedTrack)

using namespace o2::ITS;
using namespace o2::Base::Constants;
using namespace o2::Base::Track;

CookedTrack::CookedTrack() : TObject(), mTrack(), mLabel(-1), mMass(0.14), mChi2(0.)
{
  //--------------------------------------------------------------------
  // This default constructor needs to be provided
  //--------------------------------------------------------------------
  mIndex.reserve(7);
}

CookedTrack::CookedTrack(float x, float alpha, const std::array<float,kNParams> &par, const std::array<float,kCovMatSize> &cov)
  : TObject(), mTrack(x, alpha, par, cov), mLabel(-1), mMass(0.14), mChi2(0.)
{
  //--------------------------------------------------------------------
  // Main constructor
  //--------------------------------------------------------------------
}

CookedTrack::CookedTrack(const CookedTrack& t)
  : TObject(t), mTrack(t.mTrack), mLabel(t.mLabel), mMass(t.mMass), mChi2(t.mChi2), mIndex(t.mIndex)
{
  //--------------------------------------------------------------------
  // Copy constructor
  //--------------------------------------------------------------------
}

CookedTrack& CookedTrack::operator=(const CookedTrack& t)
{
  if (&t != this) {
    TObject::operator=(t);
    mTrack=t.mTrack;
    mLabel = t.mLabel;
    mMass = t.mMass;
    mChi2 = t.mChi2;
    mIndex = t.mIndex;
  }
  return *this;
}

CookedTrack::~CookedTrack()
{
  //--------------------------------------------------------------------
  // Virtual destructor
  //--------------------------------------------------------------------
}

bool CookedTrack::operator<(const CookedTrack& o) const
{
  //-----------------------------------------------------------------
  // This function compares tracks according to the their curvature
  //-----------------------------------------------------------------
  Double_t co = TMath::Abs(o.getPt());
  Double_t c = TMath::Abs(getPt());
  // Double_t co=o.GetSigmaY2()*o.GetSigmaZ2();
  // Double_t c =GetSigmaY2()*GetSigmaZ2();

  return (c > co);
}

void CookedTrack::getImpactParams(Double_t x, Double_t y, Double_t z, Double_t bz, Double_t ip[2]) const
{
  //------------------------------------------------------------------
  // This function calculates the transverse and longitudinal impact parameters
  // with respect to a point with global coordinates (x,y,0)
  // in the magnetic field "bz" (kG)
  //------------------------------------------------------------------
  Double_t f1 = getSnp(), r1 = TMath::Sqrt((1. - f1) * (1. + f1));
  Double_t xt = getX(), yt = getY();
  Double_t sn = TMath::Sin(getAlpha()), cs = TMath::Cos(getAlpha());
  Double_t a = x * cs + y * sn;
  y = -x * sn + y * cs;
  x = a;
  xt -= x;
  yt -= y;

  Double_t rp4 = getCurvature(bz);
  if ((TMath::Abs(bz) < kAlmost0) || (TMath::Abs(rp4) < kAlmost0)) {
    ip[0] = -(xt * f1 - yt * r1);
    ip[1] = getZ() + (ip[0] * f1 - xt) / r1 * getTgl() - z;
    return;
  }

  sn = rp4 * xt - f1;
  cs = rp4 * yt + r1;
  a = 2 * (xt * f1 - yt * r1) - rp4 * (xt * xt + yt * yt);
  Double_t rr = TMath::Sqrt(sn * sn + cs * cs);
  ip[0] = -a / (1 + rr);
  Double_t f2 = -sn / rr, r2 = TMath::Sqrt((1. - f2) * (1. + f2));
  ip[1] = getZ() + getTgl() / rp4 * TMath::ASin(f2 * r1 - f1 * r2) - z;
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

void CookedTrack::setExternalClusterIndex(Int_t layer, Int_t idx)
{
  //--------------------------------------------------------------------
  // Set the cluster index within an external cluster array 
  //--------------------------------------------------------------------
  mIndex.at(layer)=idx;
}

Double_t CookedTrack::getPredictedChi2(const Cluster* c) const
{
  //-----------------------------------------------------------------
  // This function calculates a predicted chi2 increment.
  //-----------------------------------------------------------------
  std::array<float,2> p{ c->getY(), c->getZ() };
  std::array<float,3> cov{ c->getSigmaY2(), c->getSigmaYZ(), c->getSigmaZ2() };
  return mTrack.GetPredictedChi2(p, cov);
}

Bool_t CookedTrack::propagate(Double_t alpha, Double_t x, Double_t bz)
{
  if (mTrack.Rotate(float(alpha)))
    if (mTrack.PropagateTo(float(x), float(bz)))
      return kTRUE;

  return kFALSE;
}

Bool_t CookedTrack::correctForMeanMaterial(Double_t x2x0, Double_t xrho, Bool_t anglecorr)
{
  return mTrack.CorrectForMaterial(float(x2x0), float(xrho), float(mMass), anglecorr);
}

Bool_t CookedTrack::update(const Cluster* c, Double_t chi2, Int_t idx)
{
  //--------------------------------------------------------------------
  // Update track params
  //--------------------------------------------------------------------
  std::array<float,2> p{ c->getY(), c->getZ() };
  std::array<float,3> cov{ c->getSigmaY2(), c->getSigmaYZ(), c->getSigmaZ2() };

  if (!mTrack.Update(p, cov))
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

Bool_t CookedTrack::isBetter(const CookedTrack& best, Double_t maxChi2) const
{
  Int_t ncl = getNumberOfClusters();
  Int_t nclb = best.getNumberOfClusters();

  if (ncl >= nclb) {
    Double_t chi2 = getChi2();
    if (chi2 < maxChi2) {
      if (ncl > nclb || chi2 < best.getChi2()) {
        return kTRUE;
      }
    }
  }
  return kFALSE;
}
