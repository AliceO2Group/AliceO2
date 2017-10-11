// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

#ifndef ALICEO2_READOUT_EMULATOR_DEVICE_H_
#define ALICEO2_READOUT_EMULATOR_DEVICE_H_

#include "ReadoutEmulator/CruEmulator.h"

#include "Common/ReadoutDataModel.h"
#include "Common/Utilities.h"

#include <O2Device/O2Device.h>

#include <memory>
#include <deque>
#include <condition_variable>

#include <TApplication.h>
#include <TCanvas.h>
#include <TH1.h>

namespace o2 {
namespace DataDistribution {

class ReadoutDevice : public Base::O2Device {
public:
  static constexpr const char* OptionKeyOutputChannelName = "output-channel-name";
  static constexpr const char* OptionKeyFreeShmChannelName = "free-shm-channel-name";

  static constexpr const char* OptionKeyReadoutDataRegionSize = "data-shm-region-size";
  static constexpr const char* OptionKeyReadoutDescRegionSize = "desc-shm-region-size";

  static constexpr const char* OptionKeyCruId = "cru-id";

  static constexpr const char* OptionKeyCruSuperpageSize = "cru-superpage-size";
  static constexpr const char* OptionKeyCruDmaChunkSize = "cru-dma-chunk-size";

  static constexpr const char* OptionKeyCruLinkCount = "cru-link-count";
  static constexpr const char* OptionKeyCruLinkBitsPerS = "cru-link-bits-per-s";

  /// Default constructor
  ReadoutDevice();

  /// Default destructor
  ~ReadoutDevice() override;

  void InitTask() final;

protected:
  bool ConditionalRun() final;
  void PreRun() final;
  void PostRun() final;

  void FreeShmThread();
  void GuiThread();

  // data and Descriptor regions
  // must be here because NewUnmanagedRegionFor() is a method of FairMQDevice...
  FairMQUnmanagedRegionPtr mDataRegion;
  FairMQUnmanagedRegionPtr mDescRegion;

  std::string mOutChannelName;
  std::string mFreeShmChannelName;
  std::size_t mDataRegionSize;
  std::size_t mDescRegionSize;

  std::size_t mCruId;

  std::size_t mSuperpageSize;
  std::size_t mDmaChunkSize;
  unsigned mCruLinkCount;
  std::uint64_t mCruLinkBitsPerS;

  std::shared_ptr<CruMemoryHandler> mCruMemoryHandler;

  std::vector<std::unique_ptr<CruLinkEmulator>> mCruLinks;
  std::thread mFreeShmThread;

  /// Observables
  bool mBuildHistograms = true;
  TApplication mRootApp; // !?
  TCanvas mReadoutCanvas;
  std::thread mGuiThread;

  RunningSamples<uint64_t> mFreeSuperpagesSamples;
};
}
} /* namespace o2::DataDistribution */

#endif /* ALICEO2_READOUT_EMULATOR_DEVICE_H_ */
