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
/// \brief  Device to decode ITS/MFT raw data from STF
/// \author ruben.shahoyan@cern.ch

#ifndef O2_ITSMFT_STFDECODER_
#define O2_ITSMFT_STFDECODER_

#include <TStopwatch.h>
#include "Framework/DataProcessorSpec.h"
#include "Framework/Task.h"
#include <memory>
#include <string>
#include <string_view>
#include "ITSMFTReconstruction/ChipMappingITS.h"
#include "ITSMFTReconstruction/ChipMappingMFT.h"
#include "ITSMFTReconstruction/RawPixelDecoder.h"

using namespace o2::framework;

namespace o2
{
namespace itsmft
{
class Clusterer;

template <class Mapping>
class STFDecoder : public Task
{
 public:
  STFDecoder(bool clusters = true, bool pattern = true, bool digits = false, std::string_view dict = "");
  ~STFDecoder() override;
  void init(InitContext& ic) final;
  void run(ProcessingContext& pc) final;

 private:
  std::unique_ptr<o2::itsmft::Clusterer> setupClusterer(const std::string& dictName);
  TStopwatch mTimer;
  bool mDoClusters = false;
  bool mDoPatterns = false;
  bool mDoDigits = false;
  int mNThreads = 1;
  std::string mSelfName;
  std::string mDictName;
  std::unique_ptr<RawPixelDecoder<Mapping>> mDecoder;
  std::unique_ptr<Clusterer> mClusterer;
};

using STFDecoderITS = STFDecoder<ChipMappingITS>;
using STFDecoderMFT = STFDecoder<ChipMappingMFT>;

/// create a processor spec
o2::framework::DataProcessorSpec getSTFDecoderITSSpec(bool doClusters, bool doPatterns, bool doDigits, const std::string& dict);
o2::framework::DataProcessorSpec getSTFDecoderMFTSpec(bool doClusters, bool doPatterns, bool doDigits, const std::string& dict);

} // namespace itsmft
} // namespace o2

#endif /* O2_ITS_DIGITREADER */
