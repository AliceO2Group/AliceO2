// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

/// @file   CompressedDecodingTask.h
/// @author Francesco Noferini
/// @since  2020-02-25
/// @brief  TOF compressed data decoding task

#ifndef O2_TOF_COMPRESSEDDECODINGTASK
#define O2_TOF_COMPRESSEDDECODINGTASK

#include "Framework/Task.h"
#include "Framework/DataProcessorSpec.h"
#include <fstream>
#include "TOFReconstruction/DecoderBase.h"
#include "TOFReconstruction/Decoder.h"
#include "TOFBase/Digit.h"

using namespace o2::framework;

namespace o2
{
namespace tof
{

using namespace compressed;

class CompressedDecodingTask : public DecoderBase, public Task
{
 public:
  CompressedDecodingTask() = default;
  ~CompressedDecodingTask() override = default;
  void init(InitContext& ic) final;
  void run(ProcessingContext& pc) final;

  void postData(ProcessingContext& pc);

 private:
  /** decoding handlers **/
  void rdhHandler(const o2::header::RAWDataHeader* rdh);
  void frameHandler(const CrateHeader_t* crateHeader, const CrateOrbit_t* crateOrbit,
                    const FrameHeader_t* frameHeader, const PackedHit_t* packedHits);

  bool mStatus = false;
  o2::tof::compressed::Decoder mDecoder;
  std::vector<std::vector<o2::tof::Digit>> mDigits;
  int mNTF = 0;
  int mNCrateOpenTF = 0;
  int mNCrateCloseTF = 0;
  bool mHasToBePosted = false;
  int mInitOrbit = 0;
};

framework::DataProcessorSpec getCompressedDecodingSpec(std::string inputDesc);

} // namespace tof
} // namespace o2

#endif /* O2_TOF_COMPRESSEDDECODINGTASK */
