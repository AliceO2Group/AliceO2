// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

/// \file   STFDecoderSpec.h
/// \brief  Device to decode EC0 raw data from STF
/// \author ruben.shahoyan@cern.ch

#ifndef O2_ENDCAPS_STFDECODER_
#define O2_ENDCAPS_STFDECODER_

#include <TStopwatch.h>
#include "Framework/DataProcessorSpec.h"
#include "Framework/Task.h"
#include <memory>
#include <string>
#include <string_view>
#include "EndCapsReconstruction/ChipMappingEC0.h"

#include "EndCapsReconstruction/RawPixelDecoder.h"

using namespace o2::framework;

namespace o2
{
namespace endcaps
{
class Clusterer;

template <class Mapping>
class STFDecoder : public Task
{
 public:
  STFDecoder(bool clusters = true, bool pattern = true, bool digits = false, std::string_view dict = "");
  ~STFDecoder() override = default;
  void init(InitContext& ic) final;
  void run(ProcessingContext& pc) final;
  void endOfStream(EndOfStreamContext& ec) final;

 private:
  std::unique_ptr<o2::endcaps::Clusterer> setupClusterer(const std::string& dictName);
  TStopwatch mTimer;
  bool mDoClusters = false;
  bool mDoPatterns = false;
  bool mDoDigits = false;
  int mNThreads = 1;
  size_t mTFCounter = 0;
  std::string mSelfName;
  std::string mDictName;
  std::unique_ptr<RawPixelDecoder<Mapping>> mDecoder;
  std::unique_ptr<Clusterer> mClusterer;
};

using STFDecoderEC0 = STFDecoder<ChipMappingEC0>;

/// create a processor spec
//o2::framework::DataProcessorSpec getSTFDecoderITSSpec(bool doClusters, bool doPatterns, bool doDigits, const std::string& dict);
//o2::framework::DataProcessorSpec getSTFDecoderMFTSpec(bool doClusters, bool doPatterns, bool doDigits, const std::string& dict);
o2::framework::DataProcessorSpec getSTFDecoderEC0Spec(bool doClusters, bool doPatterns, bool doDigits, const std::string& dict);


} // namespace endcaps
} // namespace o2

#endif /*  O2_EC0_DIGITREADER */
