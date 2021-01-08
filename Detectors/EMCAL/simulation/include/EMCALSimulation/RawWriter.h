// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

#ifndef ALICEO2_EMCAL_RAWWRITER_H
#define ALICEO2_EMCAL_RAWWRITER_H

#include <gsl/span>

#include <array>
#include <fstream>
#include <memory>
#include <string>
#include <map>
#include <vector>

#include "Rtypes.h"

#include "DetectorsRaw/RawFileWriter.h"
#include "EMCALBase/Mapper.h"
#include "DataFormatsEMCAL/Digit.h"
#include "DataFormatsEMCAL/TriggerRecord.h"

namespace o2
{

namespace emcal
{

class Geometry;

struct AltroBunch {
  int mStarttime;
  std::vector<int> mADCs;
};

struct ChannelData {
  int mRow;
  int mCol;
  std::vector<o2::emcal::Digit*> mDigits;
};

struct SRUDigitContainer {
  int mSRUid;
  std::map<int, ChannelData> mChannels;
};

union ChannelHeader {
  uint32_t mDataWord;
  struct {
    uint32_t mHardwareAddress : 16; ///< Bits  0 - 15: Hardware address
    uint32_t mPayloadSize : 10;     ///< Bits 16 - 25: Payload size
    uint32_t mZero1 : 3;            ///< Bits 26 - 28: zeroed
    uint32_t mBadChannel : 1;       ///< Bit  29: Bad channel status
    uint32_t mHeaderBits : 2;       ///< Bits 30 - 31: channel header bits (1)
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
    kSubDet,
    kLink
  };
  RawWriter() = default;
  RawWriter(const char* outputdir) { setOutputLocation(outputdir); }
  ~RawWriter() = default;

  o2::raw::RawFileWriter& getWriter() const { return *mRawWriter; }

  void setOutputLocation(const char* outputdir) { mOutputLocation = outputdir; }
  void setDigits(gsl::span<o2::emcal::Digit> digits) { mDigits = digits; }
  void setFileFor(FileFor_t filefor) { mFileFor = filefor; }
  void setNumberOfADCSamples(int nsamples) { mNADCSamples = nsamples; }
  void setPedestal(int pedestal) { mPedestal = pedestal; }
  void setGeometry(o2::emcal::Geometry* geo) { mGeometry = geo; }

  void init();
  void digitsToRaw(gsl::span<o2::emcal::Digit> digits, gsl::span<o2::emcal::TriggerRecord> triggers);
  bool processTrigger(const o2::emcal::TriggerRecord& trg);

  int carryOverMethod(const header::RDHAny* rdh, const gsl::span<char> data,
                      const char* ptr, int maxSize, int splitID,
                      std::vector<char>& trailer, std::vector<char>& header) const;

 protected:
  std::vector<AltroBunch> findBunches(const std::vector<o2::emcal::Digit*>& channelDigits);
  std::tuple<int, int, int> getOnlineID(int towerID);
  std::tuple<int, int> getLinkAssignment(int ddlID);

  ChannelHeader createChannelHeader(int hardwareAddress, int payloadSize, bool isBadChannel);
  std::vector<char> createRCUTrailer(int payloadsize, int feca, int fecb, double timesample, double l1phase);
  std::vector<int> encodeBunchData(const std::vector<int>& data);

 private:
  int mNADCSamples = 15;                                      ///< Number of time samples
  int mPedestal = 0;                                          ///< Pedestal
  FileFor_t mFileFor = FileFor_t::kFullDet;                   ///< Granularity of the output files
  o2::emcal::Geometry* mGeometry = nullptr;                   ///< EMCAL geometry
  std::string mOutputLocation;                                ///< Rawfile name
  std::unique_ptr<o2::emcal::MappingHandler> mMappingHandler; ///< Mapping handler
  gsl::span<o2::emcal::Digit> mDigits;                        ///< Digits input vector - must be in digitized format including the time response
  std::vector<SRUDigitContainer> mSRUdata;                    ///< Internal helper of digits assigned to SRUs
  std::unique_ptr<o2::raw::RawFileWriter> mRawWriter;         ///< Raw writer

  ClassDefNV(RawWriter, 1);
};

} // namespace emcal

} // namespace o2

#endif