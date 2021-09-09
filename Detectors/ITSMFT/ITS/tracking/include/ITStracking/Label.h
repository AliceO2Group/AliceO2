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
///
/// \file Label.h
/// \brief
///

#ifndef TRACKINGITSU_INCLUDE_LABEL_H_
#define TRACKINGITSU_INCLUDE_LABEL_H_

#include <ostream>

namespace o2
{
namespace its
{

struct Label final {
  Label(const int, const float, const float, const float, const int, const int);

  int monteCarloId;
  float transverseMomentum;
  float phiCoordinate;
  float pseudorapidity;
  int pdgCode;
  int numberOfClusters;

  friend std::ostream& operator<<(std::ostream&, const Label&);
};
} // namespace its
} // namespace o2

#endif /* TRACKINGITSU_INCLUDE_LABEL_H_ */
