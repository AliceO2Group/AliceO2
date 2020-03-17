// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

/** @file Response.cxx
 * C++ MCH charge induction and signal generation incl. Mathieson.
 * constants and functions taken from Aliroot.
 * @author Michael Winn, Laurent Aphecetche
 */

#include "MCHSimulation/Response.h"

#include "TMath.h"
#include "TRandom.h"

using namespace o2::mch;

Response::Response(Station station) : mStation(station)
{
  if (mStation == Station::Type1) {
    mK2x = 1.021026;
    mSqrtK3x = 0.7000;
    mK4x = 0.40934890;
    mK2y = 0.9778207;
    mSqrtK3y = 0.7550;
    mK4y = 0.38658194;
    mInversePitch = 1. / 0.21; // ^cm-1
  } else {
    mK2x = 1.010729;
    mSqrtK3x = 0.7131;
    mK4x = 0.40357476;
    mK2y = 0.970595;
    mSqrtK3y = 0.7642;
    mK4y = 0.38312571;
    mInversePitch = 1. / 0.25; // cm^-1
  }
}

//_____________________________________________________________________
float Response::etocharge(float edepos)
{
  int nel = int(edepos * 1.e9 / 27.4);
  float charge = 0;
  if (nel == 0)
    nel = 1;
  for (int i = 1; i <= nel; i++) {
    float arg = 0.;
    while (!arg)
      arg = gRandom->Rndm();
    charge -= mChargeSlope * TMath::Log(arg);
  }
  //translate to fC roughly,
  //equivalent to AliMUONConstants::DefaultADC2MV()*AliMUONConstants::DefaultA0()*AliMUONConstants::DefaultCapa() multiplication in aliroot
  charge *= 0.61 * 1.25 * 0.2; //TODO: to be verified precisely!
  return charge;
}
//_____________________________________________________________________
double Response::chargePadfraction(float xmin, float xmax, float ymin, float ymax)
{
  //see AliMUONResponseV0.cxx (inside DisIntegrate)
  // and AliMUONMathieson.cxx (IntXY)
  //see: https://edms.cern.ch/ui/file/1054937/1/ALICE-INT-2009-044.pdf
  // normalise w.r.t. Pitch
  xmin *= mInversePitch;
  xmax *= mInversePitch;
  ymin *= mInversePitch;
  ymax *= mInversePitch;

  return chargefrac1d(xmin, xmax, mK2x, mSqrtK3x, mK4x) * chargefrac1d(ymin, ymax, mK2y, mSqrtK3y, mK4y);
}
//______________________________________________________________________
double Response::chargefrac1d(float min, float max, double k2, double sqrtk3, double k4)
{
  // The Mathieson function integral (1D)
  double u1 = sqrtk3 * TMath::TanH(k2 * min);
  double u2 = sqrtk3 * TMath::TanH(k2 * max);
  return 2. * k4 * (TMath::ATan(u2) - TMath::ATan(u1));
}
//______________________________________________________________________
unsigned long Response::response(unsigned long adc)
{
  //TODO
  //not well done
  //FEE effects
  //equivalent: implementation of AliMUONDigitizerV3.cxx
  //TODO: for SAMPA modify
  //only called at merging step
  /*
  static const Int_t kMaxADC = (1<<12)-1; // We code the charge on a 12 bits ADC.                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                          

  Int_t thres(4095);
  Int_t qual(0xF);
  Float_t capa(AliMUONConstants::DefaultCapa()); // capa = 0.2 and a0 = 1.25                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                               
  Float_t a0(AliMUONConstants::DefaultA0());  // is equivalent to gain = 4 mV/fC                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                           
  Float_t adc2mv(AliMUONConstants::DefaultADC2MV()); // 1 ADC channel = 0.61 mV                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                            

  Float_t pedestalMean = pedestals.ValueAsFloat(channel,0);
  Float_t pedestalSigma = pedestals.ValueAsFloat(channel,1);

  AliDebugClass(2,Form("DE %04d MANU %04d CH %02d PEDMEAN %7.2f PEDSIGMA %7.2f",
                       pedestals.ID0(),pedestals.ID1(),channel,pedestalMean,pedestalSigma));

  if ( qual <= 0 ) return 0;

  Float_t chargeThres = a0*thres;

  Float_t padc = charge/a0; // (adc - ped) value                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                           

  padc /= capa*adc2mv;

  Int_t adc(0);

  Float_t adcNoise = 0.0;

  if ( addNoise )
  {
    if ( noiseOnly )
    {
      adcNoise = NoiseFunction()->GetRandom()*pedestalSigma;
    }
    else
    {
      adcNoise = gRandom->Gaus(0.0,pedestalSigma);
    }
  }

  adc = TMath::Nint(padc + pedestalMean + adcNoise + 0.5);

  if ( adc < TMath::Nint(pedestalMean + fgNSigmas*pedestalSigma + 0.5) )
  {
    // this is an error only in specific cases                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                             
    if ( !addNoise || (addNoise && noiseOnly) )
    {
      AliDebugClass(1,Form(" DE %04d Manu %04d Channel %02d "
                                                                                                         " a0 %7.2f thres %04d ped %7.2f pedsig %7.2f adcNoise %7.2f "
                                                                                                         " charge=%7.2f padc=%7.2f adc=%04d ZS=%04d fgNSigmas=%e addNoise %d noiseOnly %d ",
                                                                                                         pedestals.ID0(),pedestals.ID1(),channel,
                                                                                                         a0, thres, pedestalMean, pedestalSigma, adcNoise,
                                                                                                         charge, padc, adc,
                                                                                                         TMath::Nint(pedestalMean + fgNSigmas*pedestalSigma + 0.5),
                                                                                                         fgNSigmas,addNoise,noiseOnly));
    }

    adc = 0;
  }
   */
  return (unsigned long)adc;
}
//______________________________________________________________________
float Response::getAnod(float x)
{
  int n = Int_t(x / mInversePitch);
  float wire = (x > 0) ? n + 0.5 : n - 0.5;
  return mInversePitch * wire;
}
//______________________________________________________________________
float Response::chargeCorr()
{
  //taken from AliMUONResponseV0
  //conceptually not at all understood why this should make sense
  return TMath::Exp(gRandom->Gaus(0.0, mChargeCorr / 2.0));
}
