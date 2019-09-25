// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

/**************************************************************************
 * Copyright(c) 1998-1999, ALICE Experiment at CERN, All rights reserved. *
 *                                                                        *
 * Author: The ALICE Off-line Project.                                    *
 * Contributors are mentioned in the code where appropriate.              *
 *                                                                        *
 * Permission to use, copy, modify and distribute this software and its   *
 * documentation strictly for non-commercial purposes is hereby granted   *
 * without fee, provided that the above copyright notice appears in all   *
 * copies and that both the copyright notice and this permission notice   *
 * appear in the supporting documentation. The authors make no claims     *
 * about the suitability of this software for any purpose. It is          *
 * provided "as is" without express or implied warranty.                  *
 **************************************************************************/

/* $Id$ */

////////////////////////////////////////////////////////////////////////////
//                                                                        //
//  TRD simulation - multimodule (regular rad.)                           //
//  after: M. CASTELLANO et al., COMP. PHYS. COMM. 51 (1988) 431          //
//                             + COMP. PHYS. COMM. 61 (1990) 395          //
//                                                                        //
//   17.07.1998 - A.Andronic                                              //
//   08.12.1998 - simplified version                                      //
//   11.07.2000 - Adapted code to aliroot environment (C.Blume)           //
//   04.06.2004 - Momentum dependent parameters implemented (CBL)         //
//   28.01.2010 - Adapted code to O2 environment (J. Lopez)               //
//                                                                        //
////////////////////////////////////////////////////////////////////////////

#include <TH1.h>
#include <TRandom.h>
#include <TMath.h>
#include <TVirtualMC.h>
#include <TVirtualMCStack.h>

#include "TRDSimulation/TRsim.h"

#include "FairModule.h"

using namespace o2::trd;

//_____________________________________________________________________________
TRsim::TRsim()
{
  //
  // TRsim default constructor
  //

  init();
}

//_____________________________________________________________________________
TRsim::~TRsim() = default;

//_____________________________________________________________________________
void TRsim::init()
{
  //
  // Initialization
  // The default radiator are prolypropilene foils of 10 mu thickness
  // with gaps of 80 mu filled with N2.
  //

  mNFoils[0] = 170;
  mNFoils[1] = 225;
  mNFoils[2] = 275;
  mNFoils[3] = 305;
  mNFoils[4] = 325;
  mNFoils[5] = 340;
  mNFoils[6] = 350;

  mNFoilsUp[0] = 1.25;
  mNFoilsUp[1] = 1.75;
  mNFoilsUp[2] = 2.50;
  mNFoilsUp[3] = 3.50;
  mNFoilsUp[4] = 4.50;
  mNFoilsUp[5] = 5.50;
  mNFoilsUp[6] = 10000.0;

  mFoilThick = 0.0013;
  mGapThick = 0.0060;
  mFoilDens = 0.92;
  mGapDens = 0.00125;

  mFoilZ = 5.28571;
  mGapZ = 7.0;
  mFoilA = 10.4286;
  mGapA = 14.00674;
  mTemp = 293.16;

  mFoilOmega = getOmega(mFoilDens, mFoilZ, mFoilA);
  mGapOmega = getOmega(mGapDens, mGapZ, mGapA);

  mSpBinWidth = mSpRange / mSpNBins;
  mSpLower = 1.0 - 0.5 * mSpBinWidth;
  mSpUpper = mSpLower + mSpRange;

  if (mSpectrum) {
    delete mSpectrum;
  }
  mSpectrum = new TH1D("TRspectrum", "TR spectrum", mSpNBins, mSpLower, mSpUpper);
  mSpectrum->SetDirectory(nullptr);

  // Set the sigma values
  setSigma();
}

//_____________________________________________________________________________
int TRsim::createPhotons(int pdg, float p, std::vector<float>& ePhoton)
{
  //
  // Create TRD photons for a charged particle of type <pdg> with the total
  // momentum <p>.
  // Energies of the produced TR photons: <ePhoton>
  //

  // PDG codes
  const int kPdgEle = 11;
  const int kPdgMuon = 13;
  const int kPdgPion = 211;
  const int kPdgKaon = 321;

  float mass = 0;
  switch (TMath::Abs(pdg)) {
    case kPdgEle:
      mass = 5.11e-4;
      break;
    case kPdgMuon:
      mass = 0.10566;
      break;
    case kPdgPion:
      mass = 0.13957;
      break;
    case kPdgKaon:
      mass = 0.4937;
      break;
    default:
      return 0;
      break;
  };

  // Calculate the TR photons
  return calculatePhotons(p, mass, ePhoton);
}

//_____________________________________________________________________________
int TRsim::calculatePhotons(float p, float mass, std::vector<float>& ePhoton)
{
  //
  // Produces TR photons using a parametric model for regular radiator. Photons
  // with energy larger than 15 keV are included in the MC stack and tracked by VMC
  // machinary.
  //
  // Input parameters:
  // p    - parent momentum (GeV/c)
  // mass - parent mass
  //
  // Output :
  // ePhoton - energy container of this photons in keV.
  //

  const double kAlpha = 0.0072973;
  const int kSumMax = 30;
  double tau = mGapThick / mFoilThick;
  // Calculate gamma
  double gamma = TMath::Sqrt(p * p + mass * mass) / mass;
  // Select the number of foils corresponding to momentum
  int foils = selectNFoils(p);

  // The TR spectrum
  double csi1;
  double csi2;
  double rho1;
  double rho2;
  double sigma;
  double sum;
  double nEqu;
  double thetaN;
  double aux;
  double energyeV;
  double energykeV;

  mSpectrum->Reset();
  for (int iBin = 1; iBin <= mSpNBins; iBin++) {

    energykeV = mSpectrum->GetBinCenter(iBin);
    energyeV = energykeV * 1.0e3;

    sigma = getSigma(energykeV);

    csi1 = mFoilOmega / energyeV;
    csi2 = mGapOmega / energyeV;

    rho1 = 2.5 * energyeV * mFoilThick * 1.0e4 * (1.0 / (gamma * gamma) + csi1 * csi1);
    rho2 = 2.5 * energyeV * mFoilThick * 1.0e4 * (1.0 / (gamma * gamma) + csi2 * csi2);

    // Calculate the sum
    sum = 0.0;
    for (int n = 1; n <= kSumMax; n++) {
      thetaN = (TMath::Pi() * 2.0 * n - (rho1 + tau * rho2)) / (1.0 + tau);
      if (thetaN < 0.0) {
        thetaN = 0.0;
      }
      aux = 1.0 / (rho1 + thetaN) - 1.0 / (rho2 + thetaN);
      sum += thetaN * (aux * aux) * (1.0 - TMath::Cos(rho1 + thetaN));
    }

    // Equivalent number of foils
    nEqu = (1.0 - TMath::Exp(-foils * sigma)) / (1.0 - TMath::Exp(-sigma));

    // dN / domega
    mSpectrum->SetBinContent(iBin, 4.0 * kAlpha * nEqu * sum / (energykeV * (1.0 + tau)));
  }

  // <nTR> (binsize corr.)
  float nTr = mSpBinWidth * mSpectrum->Integral();
  // Number of TR photons from Poisson distribution with mean <nTr>
  int nPhCand = gRandom->Poisson(nTr);

  // Link the MC stack and get info about parent electron
  TVirtualMCStack* stack = TVirtualMC::GetMC()->GetStack();
  int track = stack->GetCurrentTrackNumber();
  double px, py, pz, ptot;
  TVirtualMC::GetMC()->TrackMomentum(px, py, pz, ptot);
  ptot = TMath::Sqrt(px * px + py * py + pz * pz);
  px /= ptot;
  py /= ptot;
  pz /= ptot;
  // Current position of electron
  double x, y, z;
  TVirtualMC::GetMC()->TrackPosition(x, y, z);
  double t = TVirtualMC::GetMC()->TrackTime();
  for (int iPhoton = 0; iPhoton < nPhCand; ++iPhoton) {
    double e = mSpectrum->GetRandom(); // Energy of the TR photon
    // Put TR photon on particle stack
    if (e > 15) {
      e *= 1e-6; // Convert it to GeV
      int phtrack;
      stack->PushTrack(1,                // Must be 1
                       track,            // Identifier of the parent track, -1 for a primary
                       22,               // Particle code.
                       px * e,           // 4 momentum (The photon is generated on the same
                       py * e,           // direction as the parent. For irregular radiator one
                       pz * e,           // can calculate also the angle but this is a second
                       e,                // order effect)
                       x, y, z, t,       // 4 vertex
                       0.0, 0.0, 0.0,    // Polarisation
                       kPFeedBackPhoton, // Production mechanism (there is no TR in G3 so one has to make some convention)
                       phtrack,          // On output the number of the track stored
                       1.0,
                       1);
    }
    // Custom treatment of TR photons
    else {
      ePhoton.push_back(e);
    }
  }
  return 1;
}

//_____________________________________________________________________________
void TRsim::setSigma()
{
  //
  // Sets the absorbtion crosssection for the energies of the TR spectrum
  //

  for (int iBin = 0; iBin < mSpNBins; iBin++) {
    double energykeV = iBin * mSpBinWidth + 1.0;
    mSigma[iBin] = getSigma(energykeV);
  }
}

//_____________________________________________________________________________
double TRsim::getSigma(double energykeV)
{
  //
  // Calculates the absorbtion crosssection for a one-foil-one-gap-radiator
  //

  // keV -> MeV
  double energyMeV = energykeV * 0.001;
  if (energyMeV >= 0.001) {
    return (getMuPo(energyMeV) * mFoilDens * mFoilThick +
            getMuAi(energyMeV) * mGapDens * mGapThick * getTemp());
  } else {
    return 1.0e6;
  }
}

//_____________________________________________________________________________
double TRsim::getMuPo(double energyMeV)
{
  //
  // Returns the photon absorbtion cross section for polypropylene
  //

  constexpr int kN = 36;
  double mu[kN] = {1.894E+03, 5.999E+02, 2.593E+02,
                   7.743E+01, 3.242E+01, 1.643E+01,
                   9.432E+00, 3.975E+00, 2.088E+00,
                   7.452E-01, 4.315E-01, 2.706E-01,
                   2.275E-01, 2.084E-01, 1.970E-01,
                   1.823E-01, 1.719E-01, 1.534E-01,
                   1.402E-01, 1.217E-01, 1.089E-01,
                   9.947E-02, 9.198E-02, 8.078E-02,
                   7.262E-02, 6.495E-02, 5.910E-02,
                   5.064E-02, 4.045E-02, 3.444E-02,
                   3.045E-02, 2.760E-02, 2.383E-02,
                   2.145E-02, 1.819E-02, 1.658E-02};
  double en[kN] = {1.000E-03, 1.500E-03, 2.000E-03,
                   3.000E-03, 4.000E-03, 5.000E-03,
                   6.000E-03, 8.000E-03, 1.000E-02,
                   1.500E-02, 2.000E-02, 3.000E-02,
                   4.000E-02, 5.000E-02, 6.000E-02,
                   8.000E-02, 1.000E-01, 1.500E-01,
                   2.000E-01, 3.000E-01, 4.000E-01,
                   5.000E-01, 6.000E-01, 8.000E-01,
                   1.000E+00, 1.250E+00, 1.500E+00,
                   2.000E+00, 3.000E+00, 4.000E+00,
                   5.000E+00, 6.000E+00, 8.000E+00,
                   1.000E+01, 1.500E+01, 2.000E+01};
  return interpolate(energyMeV, en, mu, kN);
}

//_____________________________________________________________________________
double TRsim::getMuCO(double energyMeV)
{
  //
  // Returns the photon absorbtion cross section for CO2
  //

  constexpr int kN = 36;
  double mu[kN] = {0.39383E+04, 0.13166E+04, 0.58750E+03,
                   0.18240E+03, 0.77996E+02, 0.40024E+02,
                   0.23116E+02, 0.96997E+01, 0.49726E+01,
                   0.15543E+01, 0.74915E+00, 0.34442E+00,
                   0.24440E+00, 0.20589E+00, 0.18632E+00,
                   0.16578E+00, 0.15394E+00, 0.13558E+00,
                   0.12336E+00, 0.10678E+00, 0.95510E-01,
                   0.87165E-01, 0.80587E-01, 0.70769E-01,
                   0.63626E-01, 0.56894E-01, 0.51782E-01,
                   0.44499E-01, 0.35839E-01, 0.30825E-01,
                   0.27555E-01, 0.25269E-01, 0.22311E-01,
                   0.20516E-01, 0.18184E-01, 0.17152E-01};
  double en[kN] = {0.10000E-02, 0.15000E-02, 0.20000E-02,
                   0.30000E-02, 0.40000E-02, 0.50000E-02,
                   0.60000E-02, 0.80000E-02, 0.10000E-01,
                   0.15000E-01, 0.20000E-01, 0.30000E-01,
                   0.40000E-01, 0.50000E-01, 0.60000E-01,
                   0.80000E-01, 0.10000E+00, 0.15000E+00,
                   0.20000E+00, 0.30000E+00, 0.40000E+00,
                   0.50000E+00, 0.60000E+00, 0.80000E+00,
                   0.10000E+01, 0.12500E+01, 0.15000E+01,
                   0.20000E+01, 0.30000E+01, 0.40000E+01,
                   0.50000E+01, 0.60000E+01, 0.80000E+01,
                   0.10000E+02, 0.15000E+02, 0.20000E+02};
  return interpolate(energyMeV, en, mu, kN);
}

//_____________________________________________________________________________
double TRsim::getMuXe(double energyMeV)
{
  //
  // Returns the photon absorbtion cross section for xenon
  //

  constexpr int kN = 48;
  double mu[kN] = {9.413E+03, 8.151E+03, 7.035E+03,
                   7.338E+03, 4.085E+03, 2.088E+03,
                   7.780E+02, 3.787E+02, 2.408E+02,
                   6.941E+02, 6.392E+02, 6.044E+02,
                   8.181E+02, 7.579E+02, 6.991E+02,
                   8.064E+02, 6.376E+02, 3.032E+02,
                   1.690E+02, 5.743E+01, 2.652E+01,
                   8.930E+00, 6.129E+00, 3.316E+01,
                   2.270E+01, 1.272E+01, 7.825E+00,
                   3.633E+00, 2.011E+00, 7.202E-01,
                   3.760E-01, 1.797E-01, 1.223E-01,
                   9.699E-02, 8.281E-02, 6.696E-02,
                   5.785E-02, 5.054E-02, 4.594E-02,
                   4.078E-02, 3.681E-02, 3.577E-02,
                   3.583E-02, 3.634E-02, 3.797E-02,
                   3.987E-02, 4.445E-02, 4.815E-02};
  double en[kN] = {1.00000E-03, 1.07191E-03, 1.14900E-03,
                   1.14900E-03, 1.50000E-03, 2.00000E-03,
                   3.00000E-03, 4.00000E-03, 4.78220E-03,
                   4.78220E-03, 5.00000E-03, 5.10370E-03,
                   5.10370E-03, 5.27536E-03, 5.45280E-03,
                   5.45280E-03, 6.00000E-03, 8.00000E-03,
                   1.00000E-02, 1.50000E-02, 2.00000E-02,
                   3.00000E-02, 3.45614E-02, 3.45614E-02,
                   4.00000E-02, 5.00000E-02, 6.00000E-02,
                   8.00000E-02, 1.00000E-01, 1.50000E-01,
                   2.00000E-01, 3.00000E-01, 4.00000E-01,
                   5.00000E-01, 6.00000E-01, 8.00000E-01,
                   1.00000E+00, 1.25000E+00, 1.50000E+00,
                   2.00000E+00, 3.00000E+00, 4.00000E+00,
                   5.00000E+00, 6.00000E+00, 8.00000E+00,
                   1.00000E+01, 1.50000E+01, 2.00000E+01};
  return interpolate(energyMeV, en, mu, kN);
}

//_____________________________________________________________________________
double TRsim::getMuAr(double energyMeV)
{
  //
  // Returns the photon absorbtion cross section for argon
  //

  constexpr int kN = 38;
  double mu[kN] = {3.184E+03, 1.105E+03, 5.120E+02,
                   1.703E+02, 1.424E+02, 1.275E+03,
                   7.572E+02, 4.225E+02, 2.593E+02,
                   1.180E+02, 6.316E+01, 1.983E+01,
                   8.629E+00, 2.697E+00, 1.228E+00,
                   7.012E-01, 4.664E-01, 2.760E-01,
                   2.043E-01, 1.427E-01, 1.205E-01,
                   9.953E-02, 8.776E-02, 7.958E-02,
                   7.335E-02, 6.419E-02, 5.762E-02,
                   5.150E-02, 4.695E-02, 4.074E-02,
                   3.384E-02, 3.019E-02, 2.802E-02,
                   2.667E-02, 2.517E-02, 2.451E-02,
                   2.418E-02, 2.453E-02};
  double en[kN] = {1.00000E-03, 1.50000E-03, 2.00000E-03,
                   3.00000E-03, 3.20290E-03, 3.20290E-03,
                   4.00000E-03, 5.00000E-03, 6.00000E-03,
                   8.00000E-03, 1.00000E-02, 1.50000E-02,
                   2.00000E-02, 3.00000E-02, 4.00000E-02,
                   5.00000E-02, 6.00000E-02, 8.00000E-02,
                   1.00000E-01, 1.50000E-01, 2.00000E-01,
                   3.00000E-01, 4.00000E-01, 5.00000E-01,
                   6.00000E-01, 8.00000E-01, 1.00000E+00,
                   1.25000E+00, 1.50000E+00, 2.00000E+00,
                   3.00000E+00, 4.00000E+00, 5.00000E+00,
                   6.00000E+00, 8.00000E+00, 1.00000E+01,
                   1.50000E+01, 2.00000E+01};
  return interpolate(energyMeV, en, mu, kN);
}

//_____________________________________________________________________________
double TRsim::getMuMy(double energyMeV)
{
  //
  // Returns the photon absorbtion cross section for mylar
  //

  constexpr int kN = 36;
  double mu[kN] = {2.911E+03, 9.536E+02, 4.206E+02,
                   1.288E+02, 5.466E+01, 2.792E+01,
                   1.608E+01, 6.750E+00, 3.481E+00,
                   1.132E+00, 5.798E-01, 3.009E-01,
                   2.304E-01, 2.020E-01, 1.868E-01,
                   1.695E-01, 1.586E-01, 1.406E-01,
                   1.282E-01, 1.111E-01, 9.947E-02,
                   9.079E-02, 8.395E-02, 7.372E-02,
                   6.628E-02, 5.927E-02, 5.395E-02,
                   4.630E-02, 3.715E-02, 3.181E-02,
                   2.829E-02, 2.582E-02, 2.257E-02,
                   2.057E-02, 1.789E-02, 1.664E-02};
  double en[kN] = {1.00000E-03, 1.50000E-03, 2.00000E-03,
                   3.00000E-03, 4.00000E-03, 5.00000E-03,
                   6.00000E-03, 8.00000E-03, 1.00000E-02,
                   1.50000E-02, 2.00000E-02, 3.00000E-02,
                   4.00000E-02, 5.00000E-02, 6.00000E-02,
                   8.00000E-02, 1.00000E-01, 1.50000E-01,
                   2.00000E-01, 3.00000E-01, 4.00000E-01,
                   5.00000E-01, 6.00000E-01, 8.00000E-01,
                   1.00000E+00, 1.25000E+00, 1.50000E+00,
                   2.00000E+00, 3.00000E+00, 4.00000E+00,
                   5.00000E+00, 6.00000E+00, 8.00000E+00,
                   1.00000E+01, 1.50000E+01, 2.00000E+01};
  return interpolate(energyMeV, en, mu, kN);
}

//_____________________________________________________________________________
double TRsim::getMuN2(double energyMeV)
{
  //
  // Returns the photon absorbtion cross section for nitrogen
  //

  constexpr int kN = 36;
  double mu[kN] = {3.311E+03, 1.083E+03, 4.769E+02,
                   1.456E+02, 6.166E+01, 3.144E+01,
                   1.809E+01, 7.562E+00, 3.879E+00,
                   1.236E+00, 6.178E-01, 3.066E-01,
                   2.288E-01, 1.980E-01, 1.817E-01,
                   1.639E-01, 1.529E-01, 1.353E-01,
                   1.233E-01, 1.068E-01, 9.557E-02,
                   8.719E-02, 8.063E-02, 7.081E-02,
                   6.364E-02, 5.693E-02, 5.180E-02,
                   4.450E-02, 3.579E-02, 3.073E-02,
                   2.742E-02, 2.511E-02, 2.209E-02,
                   2.024E-02, 1.782E-02, 1.673E-02};
  double en[kN] = {1.00000E-03, 1.50000E-03, 2.00000E-03,
                   3.00000E-03, 4.00000E-03, 5.00000E-03,
                   6.00000E-03, 8.00000E-03, 1.00000E-02,
                   1.50000E-02, 2.00000E-02, 3.00000E-02,
                   4.00000E-02, 5.00000E-02, 6.00000E-02,
                   8.00000E-02, 1.00000E-01, 1.50000E-01,
                   2.00000E-01, 3.00000E-01, 4.00000E-01,
                   5.00000E-01, 6.00000E-01, 8.00000E-01,
                   1.00000E+00, 1.25000E+00, 1.50000E+00,
                   2.00000E+00, 3.00000E+00, 4.00000E+00,
                   5.00000E+00, 6.00000E+00, 8.00000E+00,
                   1.00000E+01, 1.50000E+01, 2.00000E+01};
  return interpolate(energyMeV, en, mu, kN);
}

//_____________________________________________________________________________
double TRsim::getMuO2(double energyMeV)
{
  //
  // Returns the photon absorbtion cross section for oxygen
  //

  constexpr int kN = 36;
  double mu[kN] = {4.590E+03, 1.549E+03, 6.949E+02,
                   2.171E+02, 9.315E+01, 4.790E+01,
                   2.770E+01, 1.163E+01, 5.952E+00,
                   1.836E+00, 8.651E-01, 3.779E-01,
                   2.585E-01, 2.132E-01, 1.907E-01,
                   1.678E-01, 1.551E-01, 1.361E-01,
                   1.237E-01, 1.070E-01, 9.566E-02,
                   8.729E-02, 8.070E-02, 7.087E-02,
                   6.372E-02, 5.697E-02, 5.185E-02,
                   4.459E-02, 3.597E-02, 3.100E-02,
                   2.777E-02, 2.552E-02, 2.263E-02,
                   2.089E-02, 1.866E-02, 1.770E-02};
  double en[kN] = {1.00000E-03, 1.50000E-03, 2.00000E-03,
                   3.00000E-03, 4.00000E-03, 5.00000E-03,
                   6.00000E-03, 8.00000E-03, 1.00000E-02,
                   1.50000E-02, 2.00000E-02, 3.00000E-02,
                   4.00000E-02, 5.00000E-02, 6.00000E-02,
                   8.00000E-02, 1.00000E-01, 1.50000E-01,
                   2.00000E-01, 3.00000E-01, 4.00000E-01,
                   5.00000E-01, 6.00000E-01, 8.00000E-01,
                   1.00000E+00, 1.25000E+00, 1.50000E+00,
                   2.00000E+00, 3.00000E+00, 4.00000E+00,
                   5.00000E+00, 6.00000E+00, 8.00000E+00,
                   1.00000E+01, 1.50000E+01, 2.00000E+01};
  return interpolate(energyMeV, en, mu, kN);
}

//_____________________________________________________________________________
double TRsim::getMuHe(double energyMeV)
{
  //
  // Returns the photon absorbtion cross section for helium
  //

  constexpr int kN = 36;
  double mu[kN] = {6.084E+01, 1.676E+01, 6.863E+00,
                   2.007E+00, 9.329E-01, 5.766E-01,
                   4.195E-01, 2.933E-01, 2.476E-01,
                   2.092E-01, 1.960E-01, 1.838E-01,
                   1.763E-01, 1.703E-01, 1.651E-01,
                   1.562E-01, 1.486E-01, 1.336E-01,
                   1.224E-01, 1.064E-01, 9.535E-02,
                   8.707E-02, 8.054E-02, 7.076E-02,
                   6.362E-02, 5.688E-02, 5.173E-02,
                   4.422E-02, 3.503E-02, 2.949E-02,
                   2.577E-02, 2.307E-02, 1.940E-02,
                   1.703E-02, 1.363E-02, 1.183E-02};
  double en[kN] = {1.00000E-03, 1.50000E-03, 2.00000E-03,
                   3.00000E-03, 4.00000E-03, 5.00000E-03,
                   6.00000E-03, 8.00000E-03, 1.00000E-02,
                   1.50000E-02, 2.00000E-02, 3.00000E-02,
                   4.00000E-02, 5.00000E-02, 6.00000E-02,
                   8.00000E-02, 1.00000E-01, 1.50000E-01,
                   2.00000E-01, 3.00000E-01, 4.00000E-01,
                   5.00000E-01, 6.00000E-01, 8.00000E-01,
                   1.00000E+00, 1.25000E+00, 1.50000E+00,
                   2.00000E+00, 3.00000E+00, 4.00000E+00,
                   5.00000E+00, 6.00000E+00, 8.00000E+00,
                   1.00000E+01, 1.50000E+01, 2.00000E+01};
  return interpolate(energyMeV, en, mu, kN);
}

//_____________________________________________________________________________
double TRsim::getMuAi(double energyMeV)
{
  //
  // Returns the photon absorbtion cross section for air
  // Implemented by Oliver Busch
  //

  constexpr int kN = 38;
  double mu[kN] = {0.35854E+04, 0.11841E+04, 0.52458E+03,
                   0.16143E+03, 0.14250E+03, 0.15722E+03,
                   0.77538E+02, 0.40099E+02, 0.23313E+02,
                   0.98816E+01, 0.51000E+01, 0.16079E+01,
                   0.77536E+00, 0.35282E+00, 0.24790E+00,
                   0.20750E+00, 0.18703E+00, 0.16589E+00,
                   0.15375E+00, 0.13530E+00, 0.12311E+00,
                   0.10654E+00, 0.95297E-01, 0.86939E-01,
                   0.80390E-01, 0.70596E-01, 0.63452E-01,
                   0.56754E-01, 0.51644E-01, 0.44382E-01,
                   0.35733E-01, 0.30721E-01, 0.27450E-01,
                   0.25171E-01, 0.22205E-01, 0.20399E-01,
                   0.18053E-01, 0.18057E-01};
  double en[kN] = {0.10000E-02, 0.15000E-02, 0.20000E-02,
                   0.30000E-02, 0.32029E-02, 0.32029E-02,
                   0.40000E-02, 0.50000E-02, 0.60000E-02,
                   0.80000E-02, 0.10000E-01, 0.15000E-01,
                   0.20000E-01, 0.30000E-01, 0.40000E-01,
                   0.50000E-01, 0.60000E-01, 0.80000E-01,
                   0.10000E+00, 0.15000E+00, 0.20000E+00,
                   0.30000E+00, 0.40000E+00, 0.50000E+00,
                   0.60000E+00, 0.80000E+00, 0.10000E+01,
                   0.12500E+01, 0.15000E+01, 0.20000E+01,
                   0.30000E+01, 0.40000E+01, 0.50000E+01,
                   0.60000E+01, 0.80000E+01, 0.10000E+02,
                   0.15000E+02, 0.20000E+02};
  return interpolate(energyMeV, en, mu, kN);
}

//_____________________________________________________________________________
double TRsim::interpolate(double energyMeV, double* en, const double* const mu, int n)
{
  //
  // interpolates the photon absorbtion cross section
  // for a given energy <energyMeV>.
  //

  double de = 0;
  int index = 0;
  int istat = locate(en, n, energyMeV, index, de);
  if (istat == 0) {
    return (mu[index] - de * (mu[index] - mu[index + 1]) / (en[index + 1] - en[index]));
  } else {
    return 0.0;
  }
}

//_____________________________________________________________________________
int TRsim::locate(double* xv, int n, double xval, int& kl, double& dx)
{
  //
  // locates a point (xval) in a 1-dim grid (xv(n))
  //

  if (xval >= xv[n - 1]) {
    return 1;
  }
  if (xval < xv[0]) {
    return -1;
  }
  int km;
  int kh = n - 1;
  kl = 0;
  while (kh - kl > 1) {
    if (xval < xv[km = (kl + kh) / 2]) {
      kh = km;
    } else {
      kl = km;
    }
  }
  if ((xval < xv[kl]) ||
      (xval > xv[kl + 1]) ||
      (kl >= n - 1)) {
    LOG(FATAL) << Form("locate failed xv[%d] %f xval %f xv[%d] %f!!!\n", kl, xv[kl], xval, kl + 1, xv[kl + 1]);
    exit(1);
  }
  dx = xval - xv[kl];
  return 0;
}

//_____________________________________________________________________________
int TRsim::selectNFoils(float p) const
{
  //
  // Selects the number of foils corresponding to the momentum
  //

  int foils = mNFoils[mNFoilsDim - 1];

  for (int iFoil = 0; iFoil < mNFoilsDim; iFoil++) {
    if (p < mNFoilsUp[iFoil]) {
      foils = mNFoils[iFoil];
      break;
    }
  }

  return foils;
}
