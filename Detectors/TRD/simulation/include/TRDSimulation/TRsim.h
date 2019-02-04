// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

/* Copyright(c) 1998-1999, ALICE Experiment at CERN, All rights reserved. *
 * See cxx source for full Copyright notice                               */

/* $Id$ */

////////////////////////////////////////////////////////////////////////////
//                                                                        //
//  TRD simulation - multimodule (regular rad.)                           //
//                                                                        //
////////////////////////////////////////////////////////////////////////////
#ifndef ALICEO2_TRD_TRSIM_H_
#define ALICEO2_TRD_TRSIM_H_

#include <TMath.h>

class TH1D;

namespace o2
{
namespace trd
{
class TRsim
{
 public:
  TRsim();
  ~TRsim();
  void init();
  int createPhotons(int pdg, float p, std::vector<float>& ePhoton);
  int calculatePhotons(float p, float mass, std::vector<float>& ePhoton);
  double getSigma(double energykeV);
  double interpolate(double energyMeV, double* en, const double* const mu, int n);
  int locate(double* xv, int n, double xval, int& kl, double& dx);
  double getOmega(float rho, float z, float a) { return (28.8 * TMath::Sqrt(rho * z / a)); };
  int selectNFoils(float p) const;

  void setFoilThick(float t)
  {
    mFoilThick = t;
    setSigma();
  };
  void setGapThick(float t)
  {
    mGapThick = t;
    setSigma();
  };
  void setFoilDens(float d)
  {
    mFoilDens = d;
    mFoilOmega = getOmega(mFoilDens, mFoilZ, mFoilA);
    setSigma();
  };
  void setFoilZ(float z)
  {
    mFoilZ = z;
    mFoilOmega = getOmega(mFoilDens, mFoilZ, mFoilA);
  };
  void setFoilA(float a)
  {
    mFoilA = a;
    mFoilOmega = getOmega(mFoilDens, mFoilZ, mFoilA);
  };
  void setGapDens(float d)
  {
    mGapDens = d;
    mGapOmega = getOmega(mGapDens, mGapZ, mGapA);
    setSigma();
  };
  void setGapZ(float z)
  {
    mGapZ = z;
    mGapOmega = getOmega(mGapDens, mGapZ, mGapA);
  };
  void setGapA(float a)
  {
    mGapA = a;
    mGapOmega = getOmega(mGapDens, mGapZ, mGapA);
  };
  void setTemp(float t)
  {
    mTemp = t;
    setSigma();
  };
  void setSigma();

  double getMuPo(double energyMeV);
  double getMuCO(double energyMeV);
  double getMuXe(double energyMeV);
  double getMuAr(double energyMeV);
  double getMuMy(double energyMeV);
  double getMuN2(double energyMeV);
  double getMuO2(double energyMeV);
  double getMuHe(double energyMeV);
  double getMuAi(double energyMeV);

  float getFoilThick() const { return mFoilThick; };
  float getGapThick() const { return mGapThick; };
  float getFoilDens() const { return mFoilDens; };
  float getGapDens() const { return mGapDens; };
  double getFoilgetOmega() const { return mFoilOmega; };
  double getGapgetOmega() const { return mGapOmega; };
  float getTemp() const { return mTemp / 273.16; };
  TH1D* getSpectrum() const { return mSpectrum; };

 protected:
  static constexpr int mNFoilsDim = 7;      // Dimension of the NFoils array
  std::array<int, mNFoilsDim> mNFoils;      // [mNFoilsDim] Number of foils in the radiator stack
  std::array<double, mNFoilsDim> mNFoilsUp; // [mNFoilsDim] Upper momenta for a given number of foils
  float mFoilThick;                         // Thickness of the foils (cm)
  float mGapThick;                          // Thickness of the gaps between the foils (cm)
  float mFoilDens;                          // Density of the radiator foils (g/cm^3)
  float mGapDens;                           // Density of the gas in the radiator gaps (g/cm^3)
  double mFoilOmega;                        // Plasma frequency of the radiator foils
  double mGapOmega;                         // Plasma frequency of the gas in the radiator gaps
  float mFoilZ;                             // Z of the foil material
  float mGapZ;                              // Z of the gas in the gaps
  float mFoilA;                             // A of the foil material
  float mGapA;                              // A of the gas in the gaps
  float mTemp;                              // Temperature of the radiator gas (Kelvin)
  static constexpr int mSpNBins = 200;      // Number of bins of the TR spectrum
  static constexpr float mSpRange = 100;    // Range of the TR spectrum
  float mSpBinWidth;                        // Bin width of the TR spectrum
  float mSpLower;                           // Lower border of the TR spectrum
  float mSpUpper;                           // Upper border of the TR spectrum
  std::array<double, mSpNBins> mSigma;      // [mSpNBins] Array of sigma values
  TH1D* mSpectrum = nullptr;                // TR photon energy spectrum
};
} // namespace trd
} // namespace o2
#endif
