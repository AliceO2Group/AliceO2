// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

#ifndef O2_TPC_RAWREADER_H_
#define O2_TPC_RAWREADER_H_

/// \file RawReader.h
/// \author Sebastian Klewin (Sebastian.Klewin@cern.ch)

#include <string>
#include <vector>
#include <map>
#include <memory>
#include <utility>
#include <tuple>

#include "TPCBase/PadPos.h"
#include "TPCBase/CalDet.h"
#include "TPCReconstruction/RawReaderEventSync.h"

namespace o2
{
namespace tpc
{

/// \class RawReader
/// \brief Reader for RAW TPC data
/// \author Sebastian Klewin (Sebastian.Klewin@cern.ch)
class RawReader
{
 public:
  /// Data header struct
  struct Header {
    uint16_t dataType;     ///< readout mode, 1: GBT frames, 2: decoded data, 3: both
    uint8_t reserved_01;   ///< reserved part
    uint8_t headerVersion; ///< Header version
    uint32_t nWords;       ///< number of 32 bit words of header + payload
    uint64_t timeStamp_w;  ///< time stamp of header, high and low fields are reversed
    uint64_t eventCount_w; ///< Event counter, high and low fields are reversed
    uint64_t reserved_2_w; ///< Reserved part, high and low fields are reversed

    /// Get the time stamp
    /// @return corrected header time stamp
    uint64_t timeStamp() { return (timeStamp_w << 32) | (timeStamp_w >> 32); }

    /// Get event counter
    /// @return corrected event counter
    uint64_t eventCount() { return (eventCount_w << 32) | (eventCount_w >> 32); }

    /// Get reserved data field
    /// @return corrected data field
    uint64_t reserved_2() { return (reserved_2_w << 32) | (reserved_2_w >> 32); }

    /// Default constructor
    Header() = default;

    /// Copy constructor
    Header(const Header& other) = default; // : dataType(h.dataType), reserved_01(h.reserved_01),
                                           //headerVersion(h.headerVersion), nWords(h.nWords), timeStamp_w(h.timeStamp_w),
                                           //eventCount_w(h.eventCount_w), reserved_2_w(h.reserved_2_w) {};
  };

  /// Data struct
  struct EventInfo {
    std::string path; ///< Path to data file
    int posInFile;    ///< Position in data file
    Header header;    ///< Header of this evend

    /// Default constructor
    EventInfo() : path(""), posInFile(-1), header(){};

    /// Copy constructor
    EventInfo(const EventInfo& other) = default;
    //: path(e.path), posInFile(e.posInFile), region(e.region),
    //  link(e.link), header(e.header) {};
  };

  /// Default constructor
  /// @param region Region of the data
  /// @param link FEC of the data
  /// @param run RUN number of the data
  /// @param sampaVersion Version of SAMPA chips, -1 is latest
  RawReader(int region = -1, int link = -1, int run = -1, int sampaVersion = -1);

  /// Copy constructor
  RawReader(const RawReader& other) = default;

  /// Destructor
  ~RawReader() = default;

  /// Reads (and decodes) the next event, starts again with event 0 after last one
  /// @return loaded event number
  int loadNextEvent();

  /// Reads (and decodes) the next event
  /// @return loaded event number, -1 after last event
  int loadNextEventNoWrap();

  /// Reads (and decodes) the previous event, starts again with last one after event 0
  /// @return loaded event number
  int loadPreviousEvent();

  /// Reads (and decodes) the previous event
  /// @return loaded event number, -1 after first event
  int loadPreviousEventNoWrap();

  /// Reads (and decodes) given event
  /// @param event Event number to read
  /// @return Indicator of success
  bool loadEvent(int64_t event);

  /// Add input file for decoding
  /// @param infile Input file string in the format "path_to_file:#region:#fec[:sampaVersion]", where
  //                #region/#fec/sampaVersion is a number and sampaVersion optional
  /// @return True if string has correct format and file can be opened
  bool addInputFile(std::string infile);

  /// Add several input files for decoding
  /// @param infiles Vector of input file strings, each formatted as "path_to_file:#region:#fec[:sampaVersion]", where
  //                 #region/#fec/sampaVersion is a number and sampaVersion optional
  /// @return True if at least one string has correct format and file can be opened
  bool addInputFile(const std::vector<std::string>* infiles);

  /// Add input file for decoding
  /// @param region Region of the data
  /// @param link FEC of the data
  /// @param sampaVersion Version of SAMPA chip
  /// @param path Path to data
  /// @param run Run number
  /// @return True file can be opened
  bool addInputFile(int region, int link, int sampaVersion, std::string path, int run = -1);

  /// Get the first event
  /// @return Event number of first event in data
  int64_t getFirstEvent() const { return mEvents.begin()->first; };

  /// Get the last event
  /// @return Event number of last event in data
  int64_t getLastEvent() const { return (mEvents.size() == 0) ? mEvents.begin()->first : mEvents.rbegin()->first; };

  /// Get number of events
  /// @return If events are continuous, it's the number of stored events
  int64_t getNumberOfEvents() const { return mEvents.size(); };

  /// Get time stamp of first data
  /// @param hf half SAMPA
  /// @return Time stamp of first decoded ADC value
  uint64_t getTimeStamp(short hf) const { return mTimestampOfFirstData[hf]; };

  /// Get data
  /// @param padPos local pad position (row starts with 0 in each region)
  /// @return shared pointer to data vector, each element is one time bin
  std::shared_ptr<std::vector<uint16_t>> getData(const PadPos& padPos);

  /// Get data of next pad position
  /// @param padPos local pad position (row starts with 0 in each region)
  /// @return shared pointer to data vector, each element is one time bin
  std::shared_ptr<std::vector<uint16_t>> getNextData(PadPos& padPos);

  int getRegion() const { return mRegion; };
  int getLink() const { return mLink; };
  int getEventNumber() const { return mLastEvent; };
  int getRunNumber() const { return mRun; };
  short getSyncPos(short hf) const { return mSyncPos[hf]; };

  /// Set type of decoding for data in readout mode 3
  /// @param useRaw If true, raw GBT frames are decoded, if false, pre-decoded data is used
  void setUseRawInMode3(bool useRaw) { mUseRawInMode3 = useRaw; };

  /// Apply a channel mask
  /// @param applyMask The channel mask set via setChannelMask() is applied
  void setApplyChannelMask(bool applyMask) { mApplyChannelMask = applyMask; };

  /// Set a channel mask
  /// @param channelMask Pointer to a existing mask, is applied only if enable flag is set
  void setChannelMask(std::shared_ptr<CalDet<bool>> channelMask) { mChannelMask = channelMask; };

  /// Does the ADC clock check
  /// @param checkAdc Checks for a valid ADC clock in the data stream
  void setCheckAdcClock(bool checkAdc) { mCheckAdcClock = checkAdc; };
  /// Set the SAMPA version
  /// @param sampaVersion Version to be set
  void setSampaVersion(int sampaVerson) { mSampaVersion = sampaVerson; };
  /// Returns some information about the event, e.g. the header
  /// @param event Event number
  /// @return shared pointer to vector with event informations
  std::shared_ptr<std::vector<EventInfo>> getEventInfo(uint64_t event) const;

  /// Add Event Synchronizer
  /// @param eventSync Synchronizer instance
  void addEventSynchronizer(std::shared_ptr<RawReaderEventSync> eventSync) { mEventSynchronizer = eventSync; };

  /// Returns the result of the ADC clock check for the last loaded event
  /// @return shared pointer to the clock check results. The tuples consist of the SAMPA ID, index in data stream and sync pattern position
  std::shared_ptr<std::vector<std::tuple<short, short, short>>> getAdcError() { return mAdcError; };

 private:
  bool decodeRawGBTFrames(EventInfo eventInfo);
  bool decodePreprocessedData(EventInfo eventInfo);

  bool mUseRawInMode3;                                                              ///< in readout mode 3 decode GBT frames
  bool mApplyChannelMask;                                                           ///< apply channel mask
  bool mCheckAdcClock;                                                              ///< check the ADC clock
  int mRegion;                                                                      ///< Region of the data
  int mLink;                                                                        ///< FEC of the data
  int mRun;                                                                         ///< Run number
  int mSampaVersion;                                                                ///< Version of SAMPA chip
  int64_t mLastEvent;                                                               ///< Number of last loaded event
  std::array<uint64_t, 5> mTimestampOfFirstData;                                    ///< Time stamp of first decoded ADC value, individually for each half SAMPA
  std::map<uint64_t, std::shared_ptr<std::vector<EventInfo>>> mEvents;              ///< all "event data" - headers, file path, etc. NOT actual data
  std::map<PadPos, std::shared_ptr<std::vector<uint16_t>>> mData;                   ///< ADC values of last loaded Event
  std::map<PadPos, std::shared_ptr<std::vector<uint16_t>>>::iterator mDataIterator; ///< Iterator to last requested data
  std::array<short, 5> mSyncPos;                                                    ///< positions of the sync pattern (for readout mode 3)

  std::shared_ptr<CalDet<bool>> mChannelMask;                              ///< Channel mask
  std::shared_ptr<std::vector<std::tuple<short, short, short>>> mAdcError; ///< Storage for found ADC errors, tuple consists of SAMPA ID, index in data stream and sync pattern position

  std::shared_ptr<RawReaderEventSync> mEventSynchronizer; ///< Event synchronizer for triggered readout
};

inline std::shared_ptr<std::vector<RawReader::EventInfo>> RawReader::getEventInfo(uint64_t event) const
{
  auto evIterator = mEvents.find(event);
  if (evIterator == mEvents.end()) {
    std::shared_ptr<std::vector<EventInfo>> emptyVecPtr(new std::vector<EventInfo>);
    return emptyVecPtr;
  }
  return evIterator->second;
};

inline std::shared_ptr<std::vector<uint16_t>> RawReader::getData(const PadPos& padPos)
{
  mDataIterator = mData.find(padPos);
  if (mDataIterator == mData.end()) {
    std::shared_ptr<std::vector<uint16_t>> emptyVecPtr(new std::vector<uint16_t>);
    return emptyVecPtr;
  }
  return mDataIterator->second;
};

inline std::shared_ptr<std::vector<uint16_t>> RawReader::getNextData(PadPos& padPos)
{
  if (mDataIterator == mData.end())
    return nullptr;
  std::map<PadPos, std::shared_ptr<std::vector<uint16_t>>>::iterator last = mDataIterator;
  mDataIterator++;
  padPos = last->first;
  return last->second;
};

inline int RawReader::loadNextEvent()
{
  if (mLastEvent == -1) {

    mSyncPos.fill(-1);
    mTimestampOfFirstData.fill(0);

    loadEvent(getFirstEvent());
    return getFirstEvent();

  } else if (mLastEvent == getLastEvent()) {

    mSyncPos.fill(-1);
    mTimestampOfFirstData.fill(0);

    loadEvent(getFirstEvent());
    return getFirstEvent();

  } else {

    int event = mLastEvent + 1;
    loadEvent(event);
    return event;
  }
};

inline int RawReader::loadPreviousEvent()
{
  if (mLastEvent <= getFirstEvent()) {

    mSyncPos.fill(-1);
    mTimestampOfFirstData.fill(0);

    loadEvent(getFirstEvent());
    loadEvent(getLastEvent());
    return getLastEvent();

  } else if (mLastEvent == getFirstEvent() + 1) {

    mSyncPos.fill(-1);
    mTimestampOfFirstData.fill(0);

    loadEvent(getFirstEvent());
    return getFirstEvent();

  } else {

    int event = mLastEvent - 1;
    loadEvent(event);
    return event;
  }
};

inline int RawReader::loadNextEventNoWrap()
{
  if (mLastEvent == -1) {

    mSyncPos.fill(-1);
    mTimestampOfFirstData.fill(0);

    loadEvent(getFirstEvent());
    return getFirstEvent();

  } else if (mLastEvent == getLastEvent()) {

    return -1;

  } else {

    int event = mLastEvent + 1;
    loadEvent(event);
    return event;
  }
};

inline int RawReader::loadPreviousEventNoWrap()
{
  if (mLastEvent <= getFirstEvent()) {
    return -1;

  } else if (mLastEvent == getFirstEvent() + 1) {

    mSyncPos.fill(-1);
    mTimestampOfFirstData.fill(0);

    loadEvent(getFirstEvent());
    return getFirstEvent();

  } else {

    int event = mLastEvent - 1;
    loadEvent(event);
    return event;
  }
};

} // namespace tpc
} // namespace o2
#endif
