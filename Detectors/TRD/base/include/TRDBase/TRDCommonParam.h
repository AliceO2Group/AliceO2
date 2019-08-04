// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

#ifndef O2_TRDCOMMONPARAM_H
#define O2_TRDCOMMONPARAM_H

#include "GPUCommonRtypes.h"

namespace o2
{
namespace trd
{

class TRDPadPlane;
constexpr int kNlayer = 6, kNstack = 5, kNsector = 18, kNdet = 540;
constexpr int kTimeBins = 30;

class TRDCommonParam
{
 public:
  enum { kXenon = 0,
         kArgon = 1 };

  TRDCommonParam(const TRDCommonParam& p);
  TRDCommonParam& operator=(const TRDCommonParam& p);
  ~TRDCommonParam();

  static TRDCommonParam* Instance();
  static void Terminate();

  void SetExB(int exbOn = 1) { mExBOn = exbOn; }
  void SetSamplingFrequency(float freq) { mSamplingFrequency = freq; }
  void SetXenon();
  void SetArgon();

  bool ExBOn() const { return mExBOn; }
  bool IsXenon() const { return (mGasMixture == kXenon); }
  bool IsArgon() const { return (mGasMixture == kArgon); }
  int GetGasMixture() const { return mGasMixture; }
  float GetSamplingFrequency() const { return mSamplingFrequency; }

  // Cached magnetic field, to be called by the user before using GetDiffCoeff or GetOmegaTau
  bool cacheMagField();
  float GetOmegaTau(float vdrift);
  bool GetDiffCoeff(float& dl, float& dt, float vdrift);

  double TimeStruct(float vdrift, double xd, double z);

 protected:
  void SampleTimeStruct(float vdrift);

#ifndef GPUCA_GPUCODE_DEVICE
  static TRDCommonParam* fgInstance; //  Instance of this class (singleton implementation)
  static bool fgTerminated;          //  Defines if this class has already been terminated
#endif
  int mExBOn;            // Switch for the ExB effects
  double mField;         // cached magnetic field
  float mDiffusionT;     // Transverse drift coefficient
  float mDiffusionL;     // Longitudinal drift coefficient
  float mDiffLastVdrift; // The structures are valid for fLastVdrift (caching)

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

  ClassDef(TRDCommonParam, 1); // The constant parameters common to simulation and reconstruction
};
} // namespace trd
} // namespace o2
#endif
