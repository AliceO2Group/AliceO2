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

#ifndef ALICEO2_ZDC_FRAGMENTPARAM_H_
#define ALICEO2_ZDC_FRAGMENTPARAM_H_

#include <Rtypes.h>
#include <array>
#include <TF1.h>

namespace o2
{
namespace zdc
{

class FragmentParam
{

 public:
  static constexpr int POLDEG = 4;           // degree of polyomial for spectator parametrization function
  static constexpr int NCOEFFS = POLDEG + 1; // number of coefficients

  FragmentParam();
  FragmentParam(std::array<double, NCOEFFS> const& fn, std::array<double, NCOEFFS> const& fp,
                std::array<double, NCOEFFS> const& sigman, std::array<double, NCOEFFS> const& sigmap);

  void print() const;

  std::array<double, NCOEFFS> const& getParamsfn() const { return mParamfn; }
  std::array<double, NCOEFFS> const& getParamsfp() const { return mParamfp; }
  std::array<double, NCOEFFS> const& getParamssigman() const { return mParamsigman; }
  std::array<double, NCOEFFS> const& getParamssigmap() const { return mParamsigmap; }
  //
  TF1 const& getfNeutrons() const { return *mFNeutrons.get(); }
  TF1 const& getsigmaNeutrons() const { return *mFSigmaNeutrons.get(); }
  TF1 const& getfProtons() const { return *mFProtons.get(); }
  TF1 const& getsigmaProtons() const { return *mFSigmaProtons.get(); }
  //
  void setParamsfn(std::array<double, NCOEFFS> const& arrvalues);
  void setParamsfp(std::array<double, NCOEFFS> const& arrvalues);
  void setParamssigman(std::array<double, NCOEFFS> const& arrvalues);
  void setParamssigmap(std::array<double, NCOEFFS> const& arrvalues);

 private:
  std::array<double, NCOEFFS> mParamfn{8.536764, -0.841422, 1.403253, -0.117200, 0.002091};
  std::array<double, NCOEFFS> mParamsigman{0.689335, -0.238903, 0.044252, -0.003913, 0.000133};
  std::array<double, NCOEFFS> mParamfp{1.933857, -1.285600, 0.670891, -0.064681, 0.001718};
  std::array<double, NCOEFFS> mParamsigmap{1.456359, -0.505661, 0.088046, -0.007115, 0.000225};

  // functions
  void initFunctions();

  std::unique_ptr<TF1> mFNeutrons;
  std::unique_ptr<TF1> mFSigmaNeutrons;
  std::unique_ptr<TF1> mFProtons;
  std::unique_ptr<TF1> mFSigmaProtons;

  ClassDefNV(FragmentParam, 1);
};

} // namespace zdc
} // namespace o2

#endif /* ALICEO2_ZDC_FRAGMENTPARAM_H_ */
