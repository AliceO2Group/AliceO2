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

#include "ZDCBase/Constants.h"
#include "ZDCBase/ModuleConfig.h"
#include "DataFormatsZDC/RecEventAux.h"
#include "ZDCSimulation/DigitizerTest.h"
#include "ZDCReconstruction/DigiReco.h"

#ifndef ALICEO2_ZDC_DIGI_RECO_TEST_H
#define ALICEO2_ZDC_DIGI_RECO_TEST_H
namespace o2
{
namespace zdc
{
class DigiRecoTest
{
 public:
  DigiRecoTest() = default;
  ~DigiRecoTest() = default;
  void setCCDBServer(const std::string& s) { mCCDBServer = s; }
  void setVerbosity(int v) { mVerbosity = v; }
  int getVerbosity() const { return mVerbosity; }
  void setDebugOutput(bool state = true) { mDR.setDebugOutput(state); }
  void setAlpha(double v) { mDR.setAlpha(v); }
  double getAlpha() { return mDR.getAlpha(); }
  uint8_t getTriggerCondition() { return mDR.getTriggerCondition(); }
  void setTripleTrigger() { mDR.setTripleTrigger(); }
  void setDoubleTrigger() { mDR.setDoubleTrigger(); }
  void setSingleTrigger() { mDR.setSingleTrigger(); }
  DigiReco *getDigiReco() { return &mDR; }
  void init();
  o2::zdc::Digitizer::BCCache &getCreateBCCache(const o2::InteractionRecord &ir){
    return mDigi.getCreateBCCache(ir);
  }
  double add(int ic, float myAmp, const o2::InteractionRecord irpk, float myShift, bool hasJitter = true)
  {
    return mDigi.add(ic, myAmp, irpk, myShift, hasJitter);
  }
  void process()
  {
    mDigi.digitize();
    mDR.process(mDigi.getZDCOrbitData(), mDigi.getZDCBCData(), mDigi.getZDCChannelData());
  }
  void clear()
  {
    mDigi.clear();
  }
  const std::vector<o2::zdc::RecEventAux>& getReco() { return mDR.getReco(); }
  const DigitizerTest& getDigi() { return mDigi; }
  const uint32_t* getTDCMask() const { return mDR.getTDCMask(); }
  const uint32_t* getChMask() const { return mDR.getChMask(); }

 private:
  std::string mCCDBServer = "";
  int32_t mVerbosity = DbgMinimal;
  DigitizerTest mDigi;
  DigiReco mDR;
};
} // namespace zdc
} // namespace o2
#endif
