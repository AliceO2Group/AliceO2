// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

/*
 * O2MCApplicationBase.h
 *
 *  Created on: Feb 13, 2019
 *      Author: swenzel
 */

#ifndef STEER_INCLUDE_STEER_O2MCAPPLICATIONBASE_H_
#define STEER_INCLUDE_STEER_O2MCAPPLICATIONBASE_H_

#include "FairMCApplication.h"
#include "Rtypes.h" // for Int_t, Bool_t, Double_t, etc

namespace o2
{
namespace steer
{

// O2 specific changes/overrides to FairMCApplication
// Here in particular for custom adjustments to stepping logic
// and tracking limits
class O2MCApplicationBase : public FairMCApplication
{
 public:
  using FairMCApplication::FairMCApplication;
  ~O2MCApplicationBase() override = default;

  // specific implementation of our hard geometry limits
  double TrackingRmax() const override { return 1E20; }
  double TrackingZmax() const override { return 1E20; }

  ClassDefOverride(O2MCApplicationBase, 1) //Interface to MonteCarlo application
};

} // end namespace steer
} // end namespace o2

#endif /* STEER_INCLUDE_STEER_O2MCAPPLICATIONBASE_H_ */
