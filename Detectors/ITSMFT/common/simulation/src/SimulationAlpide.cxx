// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

/// \file SimulationAlpide.cxx
/// \brief Simulation of the ALIPIDE chip response

#include <TRandom.h>
#include <TLorentzVector.h>
#include <TClonesArray.h>
#include <TSeqCollection.h>
#include <climits>

#include "FairLogger.h"

#include "ITSMFTBase/Digit.h"
#include "ITSMFTBase/SegmentationPixel.h"
#include "ITSMFTSimulation/SimulationAlpide.h"
#include "ITSMFTSimulation/SimuClusterShaper.h"
#include "ITSMFTSimulation/Hit.h"
#include "ITSMFTSimulation/DigiParams.h"



using namespace o2::ITSMFT;

constexpr float sec2ns = 1e9;

//______________________________________________________________________
Double_t SimulationAlpide::betaGammaFunction(Double_t Par0, Double_t Par1, Double_t Par2, Double_t x) const {
  auto xsqr = x*x;
  Double_t y = Par0*((1+xsqr)/xsqr)*(0.5*TMath::Log(Par1*xsqr) - (xsqr/(1+xsqr)) - Par2*TMath::Log(x));
  return std::max(0.35, y);
}


//______________________________________________________________________
Double_t SimulationAlpide::getACSFromBetaGamma(Double_t x) const
{
  Double_t evalX = betaGammaFunction(mParams->getACSFromBGPar0(), mParams->getACSFromBGPar1(),
				     mParams->getACSFromBGPar2(), x);
  return evalX;
}


//______________________________________________________________________
Double_t SimulationAlpide::gaussian2D(Double_t sigma, Double_t offc, Double_t x, Double_t y) const {
  return (offc-1)*(1-TMath::Gaus(x,0,sigma)*TMath::Gaus(y,0,sigma))+1;
}


//______________________________________________________________________
Int_t SimulationAlpide::getPixelPositionResponse(const SegmentationPixel *seg, Int_t idPadX,
						 Int_t idPadZ, Float_t locx, Float_t locz, Double_t acs) const {
  Float_t centerX, centerZ;
  seg->detectorToLocal(idPadX, idPadZ, centerX, centerZ);

  Double_t Dx = locx-centerX;
  Double_t Dy = locz-centerZ;

  Double_t sigma = 0.001; // = 10 um
  Double_t offc  = acs; // WARNING: this is just temporary! (a function for this is ready but need further testing)
  Int_t cs = (Int_t) round(gaussian2D(sigma, offc, Dx, Dy));
  return cs;
}


//______________________________________________________________________
Int_t SimulationAlpide::sampleCSFromLandau(Double_t mpv, Double_t w) const {
  Double_t x = std::max(1., gRandom->Landau(mpv, w));
  Int_t cs = (Int_t) round(x);
  return cs;
}


//______________________________________________________________________
Double_t SimulationAlpide::computeIncidenceAngle(TLorentzVector dir) const
{
  Vector3D<float> glob(dir.Px()/dir.P(), dir.Py()/dir.P(), dir.Pz()/dir.P());
  auto loc = (*mMat)^(glob);

  /* 
     proper way to calculate real impact angle
     Vector3D<float> normal(0.f,-1.f,0.f);
     return TMath::Abs(loc.Dot(normal)/TMath::Sqrt(loc.Mag2()));
  */
  
  // but the method defines its own angle
  TVector3 pdirection(loc.X(), loc.Y(), loc.Z());
  TVector3 normal(0., -1., 0.);
  Double_t theta = pdirection.Theta() - normal.Theta();
  Double_t phi   = pdirection.Phi() - normal.Phi();
  Double_t angle = TMath::Sqrt(theta*theta + phi*phi);
  return TMath::Sqrt(theta*theta + phi*phi);

}


//______________________________________________________________________
void SimulationAlpide::updateACSWithAngle(Double_t& acs, Double_t angle) const {
  acs = acs/TMath::Power(TMath::Cos(angle*TMath::Pi()/180.), 0.7873);
}

//______________________________________________________________________
void SimulationAlpide::Hits2Digits(const SegmentationPixel *seg, Double_t eventTime, UInt_t &minFr, UInt_t &maxFr)
{
  Int_t nhits = GetNumberOfHits();

  // convert hits to digits, returning the min and max RO frames processed
  //
  // addNoise(mParam[Noise], seg);
  // RS: attention: the noise will be added just before transferring the digits of the given frame to the output,
  // because at this stage we don't know to which RO frame the noise should be added

  for (Int_t h = 0; h < nhits; ++h) {

    const Hit *hit = GetHitAt(h);
    double hTime0 = hit->GetTime()*sec2ns + eventTime - mParams->getTimeOffset(); // time from the RO start, in ns
    if (hTime0 > UINT_MAX) {
      LOG(WARNING) << "Hit RO Frame undefined: time: " << hTime0 << " is in far future: hitTime: "
		   << hit->GetTime() << " EventTime: " << eventTime << " ChipOffset: "
		   << mParams->getTimeOffset() << FairLogger::endl;
      return;
    }
    
    // calculate RO Frame for this hit
    if (hTime0<0) hTime0 = 0.;
    UInt_t roframe = static_cast<UInt_t>(hTime0/mParams->getROFrameLenght());
    if (roframe<minFr) minFr = roframe;
    if (roframe>maxFr) maxFr = roframe;
    
    // check if the hit time is not in the dead time of the chip
    // RS: info from L.Musa: no inefficiency on the frame boundary
    //if ( mParams->getROFrameLenght() - roframe%mParams->getROFrameLenght() < mParams->getROFrameDeadTime()) continue;

    switch (mParams->getHit2DigitsMethod()) {
    case DigiParams::p2dCShape :
      Hit2DigitsCShape(hit, roframe, eventTime, seg);
      break;
    case DigiParams::p2dSimple :
      Hit2DigitsSimple(hit, roframe, eventTime, seg);
      break;
    default:
      LOG(ERROR) << "Unknown point to digit mode " <<  mParams->getHit2DigitsMethod() << FairLogger::endl;
      break;
    }
  }
}

//________________________________________________________________________
void SimulationAlpide::Hit2DigitsCShape(const Hit *hit, UInt_t roFrame, double eventTime, const SegmentationPixel* seg)
{
  // convert single hit to digits with CShape generation method
  Double_t x0, x1, y0, y1, z0, z1, tof, de;
  if (!LineSegmentLocal(hit, x0, x1, y0, y1, z0, z1, tof, de)) return;
  double hTime  = hit->GetTime()*sec2ns + eventTime; // time in ns
  
  // To local coordinates
  Float_t x = x0 + 0.5*x1;
  Float_t y = y0 + 0.5*y1;
  Float_t z = z0 + 0.5*z1;
  
  TLorentzVector trackP4;
  trackP4.SetPxPyPzE(hit->GetPx(), hit->GetPy(), hit->GetPz(), hit->GetTotalEnergy());
  Double_t beta = std::min(0.99999, trackP4.Beta());
  Double_t bgamma = beta / std::sqrt(1. - beta*beta);
  if (bgamma < 0.001) return;
  Double_t effangle = computeIncidenceAngle(trackP4);
  
  // Get the pixel ID
  Int_t ix, iz;
  if (!seg->localToDetector(x, z, ix, iz)) return;
  
  Double_t acs = getACSFromBetaGamma(bgamma);
  updateACSWithAngle(acs, effangle);
  UInt_t cs = getPixelPositionResponse(seg, ix, iz, x, z, acs);
  // cs = 6; // uncomment to set the cluster size manually
  
  // Create the shape
  std::vector<UInt_t> cshape;
  auto csManager = std::make_unique<SimuClusterShaper>(cs);
  csManager->SetFireCenter(true);
  
  csManager->SetHit(ix, iz, x, z, seg);
  csManager->FillClusterSorted();
  
  cs = csManager->GetCS();
  csManager->GetShape(cshape);
  UInt_t nrows = csManager->GetNRows();
  UInt_t ncols = csManager->GetNCols();
  Int_t cx = csManager->GetCenterC();
  Int_t cz = csManager->GetCenterR();
  
  LOG(DEBUG) << "_/_/_/_/_/_/_/_/_/_/_/_/_/_/" << FairLogger::endl;
  LOG(DEBUG) << "_/_/_/ pALPIDE debug  _/_/_/" << FairLogger::endl;
  LOG(DEBUG) << "_/_/_/_/_/_/_/_/_/_/_/_/_/_/" << FairLogger::endl;
  LOG(DEBUG) << " Beta*Gamma: " << bgamma << FairLogger::endl;
  LOG(DEBUG) << "        ACS: " << acs << FairLogger::endl;
  LOG(DEBUG) << "         CS: " <<  cs << FairLogger::endl;
  LOG(DEBUG) << "      Shape: " << ClusterShape::ShapeSting(cshape).c_str() << FairLogger::endl;
  LOG(DEBUG) << "     Center: " << cx << ' ' << cz << FairLogger::endl;
  LOG(DEBUG) << "_/_/_/_/_/_/_/_/_/_/_/_/_/_/" << FairLogger::endl;
  
  for (Int_t ipix = 0; ipix < cs; ++ipix) {
    Int_t r = (Int_t) cshape[ipix] / nrows;
    Int_t c = (Int_t) cshape[ipix] % nrows;
    Int_t nx = ix - cx + c;
    if (nx<0) continue;
    if (nx>=seg->getNumberOfRows()) continue;
    Int_t nz = iz - cz + r;
    if (nz<0) continue;
    if (nz>=seg->getNumberOfColumns()) continue;
    Double_t charge = hit->GetEnergyLoss();
    
    addDigit(roFrame,nx,nz,charge,hit->getCombLabel(),hTime);
  }

}

//________________________________________________________________________
void SimulationAlpide::Hit2DigitsSimple(const Hit *hit, UInt_t roFrame, double eventTime, const SegmentationPixel* seg)
{
  // convert single hit to digits with 1 to 1 mapping
  Point3D<float> glo( 0.5*(hit->GetX() + hit->GetStartX()),
		      0.5*(hit->GetY() + hit->GetStartY()),
		      0.5*(hit->GetZ() + hit->GetStartZ()) );
  auto loc = (*mMat)^( glo );
  
  int ix, iz;
  seg->localToDetector(loc.X(), loc.Z(), ix, iz);
  if ((ix < 0) || (iz < 0)) {
    LOG(DEBUG) << "Out of the chip" << FairLogger::endl;
    return;
  }
  addDigit(roFrame, ix, iz, hit->GetEnergyLoss(), hit->getCombLabel(), hit->GetTime()*sec2ns + eventTime);
}


//______________________________________________________________________
void SimulationAlpide::addNoise(const SegmentationPixel* seg, UInt_t rofMin, UInt_t rofMax)
{
  UInt_t row = 0;
  UInt_t col = 0;
  Int_t nhits = 0;
  constexpr float ns2sec = 1e-9;

  float mean = mParams->getNoisePerPixel()*seg->getNumberOfPads();
  float nel = mParams->getThreshold()*1.1;  // RS: TODO: need realistic spectrum of noise abovee threshold
  
  for (UInt_t rof = rofMin; rof<=rofMax; rof++) {
    nhits = gRandom->Poisson(mean);
    double tstamp = mParams->getTimeOffset()+rof*mParams->getROFrameLenght(); // time in ns
    for (Int_t i = 0; i < nhits; ++i) {
      row = gRandom->Integer(seg->getNumberOfRows());
      col = gRandom->Integer(seg->getNumberOfColumns());
      // RS TODO: why the noise was added with 0 charge? It should be above the threshold!
      addDigit(rof, row, col, nel, Label(-1,0,0), tstamp);
    }
  }
}
