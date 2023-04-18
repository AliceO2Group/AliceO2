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

#include "ZDCBase/FragmentParam.h"

using namespace o2::zdc;

//______________________________________________________________________________
FragmentParam::FragmentParam()
{
  // default constructor
  initFunctions();
}

//______________________________________________________________________________
FragmentParam::FragmentParam(std::array<double, NCOEFFS> const& fn, std::array<double, NCOEFFS> const& fp,
                             std::array<double, NCOEFFS> const& sigman, std::array<double, NCOEFFS> const& sigmap)
{
  setParamsfn(fn);
  setParamsfp(fp);
  setParamssigman(sigman);
  setParamssigmap(sigmap);
  initFunctions();
}

void FragmentParam::initFunctions()
{
  // lambda helper
  auto makeFct = [](auto name, auto params, double limit) {
    const char* polynomstr = "[0]+[1]*x+[2]*x*x+[3]*x*x*x+[4]*x*x*x*x";
    auto f = new TF1(name, polynomstr, 0., limit);
    for (int j = 0; j < NCOEFFS; ++j) {
      f->SetParameter(j, params[j]);
    }
    return f;
  };

  // the proton function
  mFNeutrons.reset(makeFct("fneutrons", mParamfn, 126));

  // the proton function
  mFProtons.reset(makeFct("fprotons", mParamfp, 82));

  // the neutron sigma function
  mFSigmaNeutrons.reset(makeFct("fsigman", mParamsigman, 126));

  // the proton sigma function
  mFSigmaProtons.reset(makeFct("fsigmap", mParamsigmap, 82));
}

//______________________________________________________________________________
void FragmentParam::print() const
{
  printf(" Parameters fn: %1.6f %1.6f %1.6f %1.6f %1.6f \n", mParamfn[0], mParamfn[1], mParamfn[2], mParamfn[3], mParamfn[4]);
  printf(" Parameters fp: %1.6f %1.6f %1.6f %1.6f %1.6f \n", mParamfp[0], mParamfp[1], mParamfp[2], mParamfp[3], mParamfp[4]);
  printf(" Parameters sigman: %1.6f %1.6f %1.6f %1.6f %1.6f \n", mParamsigman[0], mParamsigman[1], mParamsigman[2], mParamsigman[3], mParamsigman[4]);
  printf(" Parameters sigmap: %1.6f %1.6f %1.6f %1.6f %1.6f \n", mParamsigmap[0], mParamsigmap[1], mParamsigmap[2], mParamsigmap[3], mParamsigmap[4]);
}

//______________________________________________________________________________
void FragmentParam::setParamsfn(std::array<double, NCOEFFS> const& arrvalues)
{
  mParamfn = arrvalues;
}

//______________________________________________________________________________
void FragmentParam::setParamsfp(std::array<double, NCOEFFS> const& arrvalues)
{
  mParamfp = arrvalues;
}

//______________________________________________________________________________
void FragmentParam::setParamssigman(std::array<double, NCOEFFS> const& arrvalues)
{
  mParamsigman = arrvalues;
}

//______________________________________________________________________________
void FragmentParam::setParamssigmap(std::array<double, NCOEFFS> const& arrvalues)
{
  mParamsigmap = arrvalues;
}
