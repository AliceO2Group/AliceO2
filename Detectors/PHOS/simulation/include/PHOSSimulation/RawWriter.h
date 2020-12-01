// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

#ifndef ALICEO2_PHOS_RAWWRITER_H
#define ALICEO2_PHOS_RAWWRITER_H

#include <gsl/span>

#include <array>
#include <fstream>
#include <memory>
#include <string>
#include <map>
#include <vector>

#include "Rtypes.h"

#include "DetectorsRaw/RawFileWriter.h"
#include "PHOSBase/Mapping.h"
#include "DataFormatsPHOS/Digit.h"
#include "DataFormatsPHOS/TriggerRecord.h"
#include "PHOSCalib/CalibParams.h"

namespace o2
{

namespace phos
{

static constexpr short kNPHOSSAMPLES = 15;      ///< Maximal number of samples in altro
static constexpr short kOVERFLOW = 970;         ///< Overflow level: 1023-pedestal~50
static constexpr float kPHOSTIMETICK = 100.e-9; ///< PHOS sampling time step

struct AltroBunch {
  int mStarttime;
  std::vector<int> mADCs;
};

struct SRUDigitContainer {
  int mSRUid;
  std::map<short, std::vector<o2::phos::Digit*>> mChannels;
};

union ChannelHeader {
  uint32_t mDataWord;
  struct {
    uint32_t mHardwareAddress : 16; ///< Bits  0 - 15: Hardware address
    uint32_t mPayloadSize : 10;     ///< Bits 16 - 25: Payload size
    uint32_t mZero1 : 3;            ///< Bits 26 - 28: zeroed
    uint32_t mBadChannel : 1;       ///< Bit  29: Bad channel status
    uint32_t mZero2 : 2;            ///< Bits 30 - 31: zeroed
  };
};

union CaloBunchWord {
  uint32_t mDataWord;
  struct {
    uint32_t mWord2 : 10; ///< Bits  0 - 9  : Word 2
    uint32_t mWord1 : 10; ///< Bits 10 - 19 : Word 1
    uint32_t mWord0 : 10; ///< Bits 20 - 29 : Word 0
    uint32_t mZero : 2;   ///< Bits 30 - 31 : zeroed
  };
};

class RawWriter
{
 public:
  enum class FileFor_t {
    kFullDet,
    kLink
  };
  RawWriter() = default;
  RawWriter(const char* outputdir) { setOutputLocation(outputdir); }
  ~RawWriter() = default;

  o2::raw::RawFileWriter& getWriter() const { return *mRawWriter; }

  void setOutputLocation(const char* outputdir) { mOutputLocation = outputdir; }
  void setFileFor(FileFor_t filefor) { mFileFor = filefor; }

  void init();
  void digitsToRaw(gsl::span<o2::phos::Digit> digits, gsl::span<o2::phos::TriggerRecord> triggers);
  bool processTrigger(const gsl::span<o2::phos::Digit> digitsbranch, const o2::phos::TriggerRecord& trg);

  int carryOverMethod(const header::RDHAny* rdh, const gsl::span<char> data,
                      const char* ptr, int maxSize, int splitID,
                      std::vector<char>& trailer, std::vector<char>& header) const;

 protected:
  void createRawBunches(const std::vector<o2::phos::Digit*>& digits, std::vector<o2::phos::AltroBunch>& bunchHG,
                        std::vector<o2::phos::AltroBunch>& bunchLG, bool& isLGFilled);

  std::vector<int> encodeBunchData(const std::vector<int>& data);
  void fillGamma2(float amp, float time, short* samples);
  // std::tuple<int, int, int> getOnlineID(int towerID);
  // std::tuple<int, int> getLinkAssignment(int ddlID);

  ChannelHeader createChannelHeader(int hardwareAddress, int payloadSize);
  std::vector<char> createRCUTrailer(int payloadsize, int feca, int fecb, double timesample, double l1phase);

 private:
  FileFor_t mFileFor = FileFor_t::kFullDet;           ///< Granularity of the output files
  std::string mOutputLocation = "./";                 ///< Rawfile name
  std::unique_ptr<o2::phos::Mapping> mMapping;        ///< Mapping handler
  const CalibParams* mCalibParams = nullptr;          ///< PHOS calibration
  gsl::span<o2::phos::Digit> mDigits;                 ///< Digits input vector - must be in digitized format including the time response
  std::vector<SRUDigitContainer> mSRUdata;            ///< Internal helper of digits assigned to SRUs
  std::unique_ptr<o2::raw::RawFileWriter> mRawWriter; ///< Raw writer

  ClassDefNV(RawWriter, 1);
};

} // namespace phos

} // namespace o2

#endif
