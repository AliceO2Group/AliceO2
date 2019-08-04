// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

#ifndef O2_MC_APPLICATION_H
#define O2_MC_APPLICATION_H

#include "FairMCApplication.h"
#include "Steer/O2MCApplicationBase.h"
#include "Rtypes.h" // for Int_t, Bool_t, Double_t, etc
#include <iostream>
#include <TParticle.h>
#include <vector>
#include <SimulationDataFormat/Stack.h>
#include <SimulationDataFormat/PrimaryChunk.h>
#include <FairRootManager.h>
#include <FairDetector.h>

class FairMQParts;
class FairMQChannel;

namespace o2
{
namespace steer
{

// O2 specific changes/overrides to FairMCApplication
// (like for device based processing in which we
//  forward the data instead of using FairRootManager::Fill())
class O2MCApplication : public O2MCApplicationBase
{
 public:
  using O2MCApplicationBase::O2MCApplicationBase;
  ~O2MCApplication() override = default;

  // triggers data sending/io
  void SendData();

  void initLate();

  /** Define actions at the end of event */
  void FinishEvent() override
  {
    // update the stack
    fStack->FillTrackArray();
    fStack->UpdateTrackIndex(fActiveDetectors);

    finishEventCommon();

    // This special finish event version does not fill the output tree of FairRootManager
    // but forwards the data to the HitMerger
    SendData();

    // call end of event on active detectors
    for (auto det : listActiveDetectors) {
      det->FinishEvent();
      det->EndOfEvent();
    }
    fStack->Reset();
    LOG(INFO) << "This event/chunk did " << mStepCounter << " steps";
  }

  /** Define actions at the end of run */
  void FinishRun();

  void attachSubEventInfo(FairMQParts&, o2::data::SubEventInfo const& info) const;

  /** Generate primary particles */
  void GeneratePrimaries() override
  {
    // ordinarily we would call the event generator ...

    // correct status code
    int i = 0;
    for (auto& p : mPrimaries) {
      p.SetStatusCode(i);
      i++;
    }

    LOG(INFO) << "Generate primaries " << mPrimaries.size() << "\n";
    GetStack()->Reset();

    // but here we init the stack from
    // a vector of particles that someone sets externally
    static_cast<o2::data::Stack*>(GetStack())->initFromPrimaries(mPrimaries);
  }

  void setPrimaries(std::vector<TParticle> const& p)
  {
    mPrimaries = p;
  }

  void setSimDataChannel(FairMQChannel* channel) { mSimDataChannel = channel; }
  void setSubEventInfo(o2::data::SubEventInfo* i);

  std::vector<TParticle> mPrimaries; //!

  FairMQChannel* mSimDataChannel;                      //! generic channel on which to send sim data
  o2::data::SubEventInfo* mSubEventInfo = nullptr;     //! what are we currently processing?
  std::vector<o2::base::Detector*> mActiveO2Detectors; //! active (data taking) o2 detectors

  ClassDefOverride(O2MCApplication, 1); //Interface to MonteCarlo application
};

} // end namespace steer
} // end namespace o2

#endif
