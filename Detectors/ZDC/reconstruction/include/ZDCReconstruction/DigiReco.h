#include <deque>
#include <TFile.h>
#include <TTree.h>
#include "ZDCBase/Constants.h"
#include "ZDCSimulation/ZDCSimParam.h"
#include "ZDCReconstruction/RecoParamZDC.h"
#include "ZDCReconstruction/ZDCTDCParam.h"
#include "ZDCReconstruction/ZDCIntegrationParam.h"
#include "ZDCBase/ModuleConfig.h"
#include "DataFormatsZDC/BCData.h"
#include "DataFormatsZDC/ChannelData.h"
#include "DataFormatsZDC/OrbitData.h"
#include "DataFormatsZDC/RecEvent.h"

#ifndef ALICEO2_ZDC_DIGI_RECO_H
#define ALICEO2_ZDC_DIGI_RECO_H
namespace o2
{
namespace zdc
{
class DigiReco
{
 public:
  DigiReco() = default;
  ~DigiReco() {
    mTDbg->Write();
    mDbg->Close();
  }
  void init();
  int process(const std::vector<o2::zdc::OrbitData> *orbitdata, const std::vector<o2::zdc::BCData> *bcdata, const std::vector<o2::zdc::ChannelData> *chdata);
  int write();
  void setVerbosity(int v)
  {
    mVerbosity = v;
  }
  int getVerbosity() const { return mVerbosity; }

  void setModuleConfig(const ModuleConfig* moduleConfig) { mModuleConfig = moduleConfig; };
  const ModuleConfig* getModuleConfig() { return mModuleConfig; };
  void setTDCParam(const ZDCTDCParam* param) { mTDCParam = param; };
  const ZDCTDCParam* getTDCParam() { return mTDCParam; };
  void setIntegrationParam(const ZDCIntegrationParam* param) { mIntParam = param; };
  const ZDCIntegrationParam* getIntegrationParam() { return mIntParam; };
  const ModuleConfig* mModuleConfig = nullptr;    /// Trigger/readout configuration object

 private:
  int reconstruct(int seq_beg, int seq_end); /// Main method for data reconstruction
  void processTrigger(int itdc, int ibeg, int iend); /// Replay of trigger algorithm on acquired data
  void interpolate(int itdc, int ibeg, int iend); /// Interpolation of samples to evaluate signal amplitude and arrival time
  void assignTDC(int ibun, int ibeg, int iend, int itdc, int tdc, float amp); // Set reconstructed TDC values
  bool mIsContinuous = true;                      /// continuous (self-triggered) or externally-triggered readout
  int mNBCAHead = 0;                              /// when storing triggered BC, store also mNBCAHead BCs
  const ZDCTDCParam* mTDCParam = nullptr;         /// TDC calibration object
  uint32_t mTDCMask[NTDCChannels]={0}; /// Identify TDC channels in trigger mask
  const ZDCIntegrationParam* mIntParam = nullptr; /// Configuration of integration
  bool mVerbosity=DbgFull;
  Double_t mTS[NTS];                          /// Tapered sinc function
  TFile* mDbg = nullptr;                      /// Debug output
  TTree* mTDbg = nullptr;                     /// Debug tree
  const std::vector<o2::zdc::OrbitData> *mOrbitData;   /// Reconstructed data
  const std::vector<o2::zdc::BCData> *mBCData;      /// BC info
  const std::vector<o2::zdc::ChannelData> *mChData; /// Payload
  std::vector<o2::zdc::RecEvent> mReco;      /// Reconstructed data
  std::map<uint32_t,int> mOrbit;   /// Information about orbit
  static constexpr int mNSB = TSN * NTimeBinsPerBC;    /// Total number of interpolated points per bunch crossing
  RecEvent mRec;                              /// Debug reconstruction event
  int mNBC = 0;
  int16_t tdc_shift[NTDCChannels] = {0};       /// TDC correction (units of 1/96 ns)
  constexpr static uint16_t mMask[NTimeBinsPerBC]={0x0001,0x002,0x004,0x008, 0x0010,0x0020,0x0040,0x0080, 0x0100,0x0200,0x0400,0x0800};
};
} // namespace zdc
} // namespace o2
#endif
