// Copyright 2019-2020 CERN and copyright holders of ALICE O2.
// See https://alice-o2.web.cern.ch/copyright for details of the copyright holders.
// All rights not expressly granted are reserved.
//
// This software is distributed under the terms of the GNU General Public
// License v3 (GPL Version 3), copied verbatim in the file "COPYING".
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

/// \file MathiesonOriginal.h
/// \brief Original definition of the Mathieson function
///
/// \author Philippe Pillot, Subatech

#ifndef O2_MCH_MATHIESONORIGINAL_H_
#define O2_MCH_MATHIESONORIGINAL_H_

namespace o2
{
namespace mch
{

/// Original Mathieson function
class MathiesonOriginal
{
 public:
  MathiesonOriginal() = default;
  ~MathiesonOriginal() = default;

  MathiesonOriginal(const MathiesonOriginal& cl) = default;
  MathiesonOriginal& operator=(const MathiesonOriginal& cl) = default;
  MathiesonOriginal(MathiesonOriginal&&) = default;
  MathiesonOriginal& operator=(MathiesonOriginal&&) = default;

  /// set the inverse of the anode-cathode pitch
  void setPitch(float pitch) { mInversePitch = (pitch > 0.) ? 1. / pitch : 0.; }

  void setSqrtKx3AndDeriveKx2Kx4(float sqrtKx3);
  void setSqrtKy3AndDeriveKy2Ky4(float sqrtKy3);

  void setFastIntegral(bool fastIntegral) { mFastIntegral = fastIntegral; }

  void init();

  float integrate(float xMin, float yMin, float xMax, float yMax) const;

 private:
  struct LUT
  {
    LUT() = default;
    LUT(int size, double min, double max);
    ~LUT();
    void init(int size, double min, double max);
    double getX(int i) const { return ((mStep * i) + mMin); }
    double getY(int j) const { return ((mStep * j) + mMin); }
    void set(int i, int j, double val)
    {
      if(mSize > 0) {
        mTable[j][i] = val;
      }
    }
    bool get(double x, double y, double& val) const;
    bool isIncluded(double x, double y) const
    {
      if (x <= mMin || x >= mMax) {
        return false;
      }
      if (y <= mMin || y >= mMax) {
        return false;
      }
      return true;
    }


    //std::vector<std::vector<double>> mTable{};
    double** mTable{ nullptr };
    int mSize{ 0 };
    double mMin{ 0 };
    double mMax{ 0 };
    double mStep{ 0 };
    double mInverseStep{ 0 };
    double mInverseWidth{ 0 };
  } mLUT;

  float integrateAnalytic(float xMin, float yMin, float xMax, float yMax) const;
  float integrateLUT(float xMin, float yMin, float xMax, float yMax) const;

  float mSqrtKx3 = 0.;      ///< Mathieson Sqrt(Kx3)
  float mKx2 = 0.;          ///< Mathieson Kx2
  float mKx4 = 0.;          ///< Mathieson Kx4 = Kx1/Kx2/Sqrt(Kx3)
  float mSqrtKy3 = 0.;      ///< Mathieson Sqrt(Ky3)
  float mKy2 = 0.;          ///< Mathieson Ky2
  float mKy4 = 0.;          ///< Mathieson Ky4 = Ky1/Ky2/Sqrt(Ky3)
  float mInversePitch = 0.; ///< 1 / anode-cathode pitch

  bool mFastIntegral = true; ///< use a fast approximation to compute the charge integral
};

} // namespace mch
} // namespace o2

#endif // O2_MCH_MATHIESONORIGINAL_H_
