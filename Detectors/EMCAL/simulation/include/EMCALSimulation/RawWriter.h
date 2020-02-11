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

#include <array>
#include <fstream>
#include <memory>
#include <string>
#include <map>
#include <vector>

#include "Rtypes.h"

#include "EMCALBase/Mapper.h"
#include "EMCALSimulation/DMAOutputStream.h"
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
  RawWriter() = default;
  RawWriter(const char* rawfilename) { setRawFileName(rawfilename); }
  ~RawWriter() = default;

  void setRawFileName(const char* filename) { mOutputStream.setOutputFilename(filename); }
  void setDigits(std::vector<o2::emcal::Digit>* digits) { mDigits = digits; }
  void setTriggerRecords(std::vector<o2::emcal::TriggerRecord>* triggers);
  void setNumberOfADCSamples(int nsamples) { mNADCSamples = nsamples; }
  void setPedestal(int pedestal) { mPedestal = pedestal; }
  void setGeometry(o2::emcal::Geometry* geo) { mGeometry = geo; }

  bool hasNextTrigger() const { return mCurrentTrigger != mTriggers->end(); }

  void init();
  void process();
  void processNextTrigger();

 protected:
  std::vector<AltroBunch> findBunches(const std::vector<o2::emcal::Digit*>& channelDigits);
  std::tuple<int, int, int> getOnlineID(int towerID);

  ChannelHeader createChannelHeader(int hardwareAddress, int payloadSize, bool isBadChannel);
  std::vector<char> createRCUTrailer();
  std::vector<int> encodeBunchData(const std::vector<int>& data);

 private:
  DMAOutputStream mOutputStream;                                   ///< DMA output stream
  int mNADCSamples = 15;                                           ///< Number of time samples
  int mPedestal = 0;                                               ///< Pedestal
  o2::emcal::Geometry* mGeometry = nullptr;                        ///< EMCAL geometry
  std::array<o2::emcal::Mapper, 4> mMappers;                       ///< EMCAL mappers
  std::vector<o2::emcal::Digit>* mDigits;                          ///< Digits input vector - must be in digitized format including the time response
  std::vector<o2::emcal::TriggerRecord>* mTriggers;                ///< Trigger records, separating the data from different triggers
  std::vector<SRUDigitContainer> mSRUdata;                         ///< Internal helper of digits assigned to SRUs
  std::vector<o2::emcal::TriggerRecord>::iterator mCurrentTrigger; ///< Current trigger in the trigger records

  ClassDefNV(RawWriter, 1);
};

} // namespace emcal

} // namespace o2

#endif