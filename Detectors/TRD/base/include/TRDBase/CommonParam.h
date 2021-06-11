// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

#ifndef O2_TRD_COMMONPARAM_H
#define O2_TRD_COMMONPARAM_H

#include <array>
#include "Rtypes.h" // for ClassDef

namespace o2
{
namespace trd
{

class PadPlane;

class CommonParam
{
 public:
  enum { kXenon = 0,
         kArgon = 1 };

  CommonParam(const CommonParam&) = delete;
  CommonParam& operator=(const CommonParam&) = delete;
  ~CommonParam() = default;

  static CommonParam* instance();

  void setExB(bool flag = true) { mExBOn = flag; }
  void setSamplingFrequency(float freq) { mSamplingFrequency = freq; }
  void setXenon();
  void setArgon();

  bool isExBOn() const { return mExBOn; }
  bool isXenon() const { return (mGasMixture == kXenon); }
  bool isArgon() const { return (mGasMixture == kArgon); }
  int getGasMixture() const { return mGasMixture; }
  float getSamplingFrequency() const { return mSamplingFrequency; }
  float getCachedField() const { return mField; }

  // Cached magnetic field, to be called by the user before using GetDiffCoeff or GetOmegaTau
  bool cacheMagField();
  float getOmegaTau(float vdrift);
  bool getDiffCoeff(float& dl, float& dt, float vdrift);

  double timeStruct(float vdrift, double xd, double z);

 protected:
  void sampleTimeStruct(float vdrift);

  static CommonParam* mgInstance;    ///<  Instance of this class (singleton implementation)
  static constexpr int TIMEBIN = 38; ///< Number of bins in time direction used for garfield simulation
  static constexpr int ZBIN = 11;    ///< Number of bins in z direction used for garfield simulation
  bool mExBOn{true};                 ///< Switch for the ExB effects
  double mField{-0.5};               ///< Cached magnetic field
  float mDiffusionT{0.};             ///< Transverse drift coefficient
  float mDiffusionL{0.};             ///< Longitudinal drift coefficient
  float mDiffLastVdrift{-1.};        ///< The structures are valid for fLastVdrift (caching)

  std::array<float, TIMEBIN * ZBIN> mTimeStruct1{}; ///< Time Structure of Drift Cells
  std::array<float, TIMEBIN * ZBIN> mTimeStruct2{}; ///< Time Structure of Drift Cells
  float mVDlo{0.};                                  ///< Lower drift velocity, for interpolation
  float mVDhi{0.};                                  ///< Higher drift velocity, for interpolation
  float mTimeLastVdrift{-1.};                       ///< The structures are valid for fLastVdrift (caching)

  float mSamplingFrequency{10.}; ///< Sampling Frequency in MHz

  int mGasMixture{kXenon}; ///< Gas mixture: 0-Xe/C02 1-Ar/CO2.

 private:
  /// This is a singleton, constructor is private!
  CommonParam() = default;

  ClassDef(CommonParam, 1); // The constant parameters common to simulation and reconstruction
};
} // namespace trd
} // namespace o2
#endif
