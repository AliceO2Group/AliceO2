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
#include "DataFormatsPHOS/Digit.h"
#include "DataFormatsPHOS/TriggerRecord.h"
#include "DataFormatsPHOS/CalibParams.h"
#include "PHOSBase/RCUTrailer.h"

namespace o2
{

namespace phos
{

static constexpr short kNPHOSSAMPLES = 30;   ///< Maximal number of samples in altro
static constexpr short kNPRESAMPLES = 2;     ///< Number of pre-samples in altro
static constexpr short kOVERFLOW = 970;      ///< Overflow level: 1023-pedestal~50
static constexpr float kPHOSTIMETICK = 100.; ///< PHOS sampling time step in ns (hits/digits keep time in ns)

struct AltroBunch {
  int mStarttime;
  std::vector<int> mADCs;
};

struct SRUDigitContainer {
  int mSRUid;
  std::map<short, std::vector<o2::phos::Digit*>> mChannels;
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
  void createTRUBunches(short truId, const std::vector<o2::phos::Digit*>& channelDigits, std::vector<o2::phos::AltroBunch>& bunchs);
  void createRawBunches(short absId, const std::vector<o2::phos::Digit*>& digits, std::vector<o2::phos::AltroBunch>& bunchHG,
                        std::vector<o2::phos::AltroBunch>& bunchLG, bool& isLGFilled);

  std::vector<uint32_t> encodeBunchData(const std::vector<uint32_t>& data);
  void fillGamma2(float amp, float time, short* samples);
  // std::tuple<int, int, int> getOnlineID(int towerID);
  // std::tuple<int, int> getLinkAssignment(int ddlID);

  std::vector<char> createRCUTrailer(int payloadsize, int feca, int fecb, double timesample, double l1phase);

 private:
  FileFor_t mFileFor = FileFor_t::kFullDet;           ///< Granularity of the output files
  std::string mOutputLocation = "./";                 ///< Rawfile name
  std::unique_ptr<const CalibParams> mCalibParams;    ///< PHOS calibration
  gsl::span<o2::phos::Digit> mDigits;                 ///< Digits input vector - must be in digitized format including the time response
  std::vector<SRUDigitContainer> mSRUdata;            ///< Internal helper of digits assigned to SRUs
  std::vector<SRUDigitContainer> mTRUdata;            ///< Internal helper of digits assigned to TRUs
  std::unique_ptr<o2::raw::RawFileWriter> mRawWriter; ///< Raw writer

  ClassDefNV(RawWriter, 1);
};

} // namespace phos

} // namespace o2

#endif
