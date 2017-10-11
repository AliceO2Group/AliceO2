// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

#ifndef ALICEO2_SUBTIMEFRAME_FILE_SINK_H_
#define ALICEO2_SUBTIMEFRAME_FILE_SINK_H_

#include "Common/SubTimeFrameDataModel.h"
#include "Common/SubTimeFrameFileWriter.h"
#include "Common/ConcurrentQueue.h"

#include <Headers/DataHeader.h>

#include <boost/program_options/options_description.hpp>
#include <boost/filesystem.hpp>
#include <fstream>
#include <vector>

class O2Device;

namespace o2
{
namespace DataDistribution
{

namespace bpo = boost::program_options;

////////////////////////////////////////////////////////////////////////////////
/// SubTimeFrameFileSink
////////////////////////////////////////////////////////////////////////////////

class SubTimeFrameFileSink
{
  using stf_pipeline = IFifoPipeline<std::unique_ptr<SubTimeFrame>>;

 public:
  static constexpr const char* OptionKeyStfSinkEnable = "data-sink-enable";
  static constexpr const char* OptionKeyStfSinkDir = "data-sink-dir";
  static constexpr const char* OptionKeyStfSinkFileName = "data-sink-file-name";
  static constexpr const char* OptionKeyStfSinkStfsPerFile = "data-sink-max-stfs-per-file";
  static constexpr const char* OptionKeyStfSinkFileSize = "data-sink-max-file-size";
  static constexpr const char* OptionKeyStfSinkSidecar = "data-sink-sidecar";

  static bpo::options_description getProgramOptions();

  SubTimeFrameFileSink() = delete;

  SubTimeFrameFileSink(O2Device& pDevice, stf_pipeline& pPipeline, unsigned pPipelineStageIn, unsigned pPipelineStageOut)
    : mDeviceI(pDevice),
      mPipelineI(pPipeline),
      mPipelineStageIn(pPipelineStageIn),
      mPipelineStageOut(pPipelineStageOut)
  {
  }

  ~SubTimeFrameFileSink()
  {
    if (mSinkThread.joinable()) {
      mSinkThread.join();
    }
    LOG(INFO) << "(Sub)TimeFrame Sink terminated...";
  }

  bool loadVerifyConfig(const FairMQProgOptions& pFMQProgOpt);

  bool enabled() const { return mEnabled; }

  void start();
  void stop();

  void DataHandlerThread(const unsigned pIdx);

  std::string newStfFileName();

 private:
  const O2Device& mDeviceI;
  stf_pipeline& mPipelineI;

  std::unique_ptr<SubTimeFrameFileWriter> mStfWriter = nullptr;

  /// Configuration
  bool mEnabled = false;
  std::string mRootDir;
  std::string mCurrentDir;
  std::string mFileNamePattern;
  std::uint64_t mStfsPerFile;
  std::uint64_t mFileSize;
  bool mSidecar = false;

  /// Thread for file writing
  std::thread mSinkThread;
  unsigned mPipelineStageIn;
  unsigned mPipelineStageOut;

  /// variables
  unsigned mCurrentFileIdx = 0;
};
}
} /* o2::DataDistribution */

#endif /* ALICEO2_SUBTIMEFRAME_FILE_SINK_H_ */
