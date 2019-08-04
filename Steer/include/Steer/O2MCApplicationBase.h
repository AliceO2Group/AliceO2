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

#include <FairMCApplication.h>
#include "Rtypes.h" // for Int_t, Bool_t, Double_t, etc
#include <TVirtualMC.h>
#include "SimConfig/SimCutParams.h"

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
  O2MCApplicationBase() : FairMCApplication(), mCutParams(o2::conf::SimCutParams::Instance()) {}
  O2MCApplicationBase(const char* name, const char* title, TObjArray* ModList, const char* MatName) : FairMCApplication(name, title, ModList, MatName), mCutParams(o2::conf::SimCutParams::Instance())
  {
  }

  ~O2MCApplicationBase() override = default;

  void Stepping() override;
  void PreTrack() override;
  void BeginEvent() override;
  void FinishEvent() override;
  void ConstructGeometry() override;
  void InitGeometry() override;

  // specific implementation of our hard geometry limits
  double TrackingRmax() const override { return mCutParams.maxRTracking; }
  double TrackingZmax() const override { return mCutParams.maxAbsZTracking; }

 protected:
  o2::conf::SimCutParams const& mCutParams; // reference to parameter system
  unsigned long long mStepCounter{0};
  std::map<int, std::string> mModIdToName{};      // mapping of module id to name
  std::map<int, std::string> mSensitiveVolumes{}; // collection of all sensitive volumes with
                                                  // keeping track of volumeIds and volume names

  /// some common parts of finishEvent
  void finishEventCommon();

  ClassDefOverride(O2MCApplicationBase, 1);
};

} // end namespace steer
} // end namespace o2

#endif /* STEER_INCLUDE_STEER_O2MCAPPLICATIONBASE_H_ */
