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

#include <TObject.h>
#include <TMath.h>

class TH1D;

// class FairModule;

class TRsim : public TObject
{
 public:
  TRsim();
  // TRsim(const TRsim& s);
  // TRsim(FairModule* mod, int foil, int gap);
  virtual ~TRsim();
  TRsim& operator=(const TRsim& s);
  virtual void Copy(TObject& s) const;
  virtual void Init();
  virtual int CreatePhotons(int pdg, float p, int& nPhoton, float* ePhoton);
  virtual int TrPhotons(float p, float mass, int& nPhoton, float* ePhoton);
  virtual double Sigma(double energykeV);
  virtual double Interpolate(double energyMeV, double* en, const double* const mu, int n);
  virtual int Locate(double* xv, int n, double xval, int& kl, double& dx);
  virtual double Omega(float rho, float z, float a) { return (28.8 * TMath::Sqrt(rho * z / a)); };
  virtual int SelectNFoils(float p) const;

  void SetFoilThick(float t)
  {
    mFoilThick = t;
    SetSigma();
  };
  void SetGapThick(float t)
  {
    mGapThick = t;
    SetSigma();
  };
  void SetFoilDens(float d)
  {
    mFoilDens = d;
    mFoilOmega = Omega(mFoilDens, mFoilZ, mFoilA);
    SetSigma();
  };
  void SetFoilZ(float z)
  {
    mFoilZ = z;
    mFoilOmega = Omega(mFoilDens, mFoilZ, mFoilA);
  };
  void SetFoilA(float a)
  {
    mFoilA = a;
    mFoilOmega = Omega(mFoilDens, mFoilZ, mFoilA);
  };
  void SetGapDens(float d)
  {
    mGapDens = d;
    mGapOmega = Omega(mGapDens, mGapZ, mGapA);
    SetSigma();
  };
  void SetGapZ(float z)
  {
    mGapZ = z;
    mGapOmega = Omega(mGapDens, mGapZ, mGapA);
  };
  void SetGapA(float a)
  {
    mGapA = a;
    mGapOmega = Omega(mGapDens, mGapZ, mGapA);
  };
  void SetTemp(float t)
  {
    mTemp = t;
    SetSigma();
  };
  void SetSigma();

  virtual double GetMuPo(double energyMeV);
  virtual double GetMuCO(double energyMeV);
  virtual double GetMuXe(double energyMeV);
  virtual double GetMuAr(double energyMeV);
  virtual double GetMuMy(double energyMeV);
  virtual double GetMuN2(double energyMeV);
  virtual double GetMuO2(double energyMeV);
  virtual double GetMuHe(double energyMeV);
  virtual double GetMuAi(double energyMeV);

  float GetFoilThick() const { return mFoilThick; };
  float GetGapThick() const { return mGapThick; };
  float GetFoilDens() const { return mFoilDens; };
  float GetGapDens() const { return mGapDens; };
  double GetFoilOmega() const { return mFoilOmega; };
  double GetGapOmega() const { return mGapOmega; };
  float GetTemp() const { return mTemp / 273.16; };
  TH1D* GetSpectrum() const { return mSpectrum; };

 protected:
  int mNFoilsDim;    //  Dimension of the NFoils array
  int* mNFoils;      //[mNFoilsDim] Number of foils in the radiator stack
  double* mNFoilsUp; //[mNFoilsDim] Upper momenta for a given number of foils
  float mFoilThick;  //  Thickness of the foils (cm)
  float mGapThick;   //  Thickness of the gaps between the foils (cm)
  float mFoilDens;   //  Density of the radiator foils (g/cm^3)
  float mGapDens;    //  Density of the gas in the radiator gaps (g/cm^3)
  double mFoilOmega; //  Plasma frequency of the radiator foils
  double mGapOmega;  //  Plasma frequency of the gas in the radiator gaps
  float mFoilZ;      //  Z of the foil material
  float mGapZ;       //  Z of the gas in the gaps
  float mFoilA;      //  A of the foil material
  float mGapA;       //  A of the gas in the gaps
  float mTemp;       //  Temperature of the radiator gas (Kelvin)
  int mSpNBins;      //  Number of bins of the TR spectrum
  float mSpRange;    //  Range of the TR spectrum
  float mSpBinWidth; //  Bin width of the TR spectrum
  float mSpLower;    //  Lower border of the TR spectrum
  float mSpUpper;    //  Upper border of the TR spectrum
  double* mSigma;    //[mSpNBins] Array of sigma values
  TH1D* mSpectrum;   //! TR photon energy spectrum
};
#endif