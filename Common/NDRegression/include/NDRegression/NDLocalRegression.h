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

#ifndef ALICEO2_NDLOCALREGRESSION_H_
#define ALICEO2_NDLOCALREGRESSION_H_

/// \file   NDLocalRegression.h
/// \author Gabor Biro, biro.gabor@wigner.hu

namespace o2
{
namespace nd_local_regression
{

class NDLocalRegression
{

 public:
  NDLocalRegression() = default;
  // NDLocalRegression();
  ~NDLocalRegression() = default;

  bool init();
};

} // namespace nd_local_regression
} // namespace o2

#endif
