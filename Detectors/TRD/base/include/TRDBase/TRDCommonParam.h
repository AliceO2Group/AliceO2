// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See https://alice-o2.web.cern.ch/ for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

#ifndef O2_TRDCOMMONPARAM_H
#define O2_TRDCOMMONPARAM_H

#include "TRDBase/TRDSimParam.h"

class TRDPadPlane;

namespace o2
{
namespace trd
{
class TRDCommonParam
{
 public:
  enum { kNlayer = 6, kNstack = 5, kNsector = 18, kNdet = 540 };

  enum { kXenon = 0, kArgon = 1 };

  TRDCommonParam(const TRDCommonParam& p);
  TRDCommonParam& operator=(const TRDCommonParam& p);
  ~TRDCommonParam();

  static TRDCommonParam* Instance();
  static void Terminate();

  void SetExB(int exbOn = 1) { mExBOn = exbOn; }
  void SetSamplingFrequency(float freq) { mSamplingFrequency = freq; }
  void SetXenon()
  {
    mGasMixture = kXenon;
    TRDSimParam::Instance()->ReInit();
  }
  void SetArgon()
  {
    mGasMixture = kArgon;
    TRDSimParam::Instance()->ReInit();
  }

  bool ExBOn() const { return mExBOn; }
  bool IsXenon() const { return (mGasMixture == kXenon); }
  bool IsArgon() const { return (mGasMixture == kArgon); }
  int GetGasMixture() const { return mGasMixture; }
  float GetSamplingFrequency() const { return mSamplingFrequency; }
  float GetOmegaTau(float vdrift);
  bool GetDiffCoeff(float& dl, float& dt, float vdrift);

  double TimeStruct(float vdrift, double xd, double z);

 protected:
  void SampleTimeStruct(float vdrift);

  static TRDCommonParam* fgInstance; //  Instance of this class (singleton implementation)
  static bool fgTerminated;          //  Defines if this class has already been terminated

  int mExBOn; //  Switch for the ExB effects

  float mDiffusionT;     //  Transverse drift coefficient
  float mDiffusionL;     //  Longitudinal drift coefficient
  float mDiffLastVdrift; //  The structures are valid for fLastVdrift (caching)

  float* mTimeStruct1;   //! Time Structure of Drift Cells
  float* mTimeStruct2;   //! Time Structure of Drift Cells
  float mVDlo;           //  Lower drift velocity, for interpolation
  float mVDhi;           //  Higher drift velocity, for interpolation
  float mTimeLastVdrift; //  The structures are valid for fLastVdrift (caching)

  float mSamplingFrequency; //  Sampling Frequency in MHz

  int mGasMixture; //  Gas mixture: 0-Xe/C02 1-Ar/CO2.

 private:
  // This is a singleton, constructor is private!
  TRDCommonParam();

  ClassDef(TRDCommonParam, 1) // The constant parameters common to simulation and reconstruction
};
}
}
#endif
