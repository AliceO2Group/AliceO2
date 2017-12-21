// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

//-*- Mode: C++ -*-

#ifndef ALICEO2_HEADER_HEARTBEATFRAME_H
#define ALICEO2_HEADER_HEARTBEATFRAME_H

// @file   Heartbeatframe.h
// @author Matthias Richter
// @since  2017-02-02
// @brief  Definition of the heartbeat frame layout

#include "Headers/DataHeader.h"
#include <functional>
#include <map>
#include <vector>

namespace o2 {
namespace header {

// The Heartbeat frame layout is specified in
// http://svnweb.cern.ch/world/wsvn/alicetdrrun3/Notes/Run34SystemNote/detector-read-alice/ALICErun34_readout.pdf
// TODO: replace citation with correct ALICE note reference when published

// general remark:
// at the moment its not clear how the heartbeat frame is transmitted in  AliceO2.
// Current understanding is that the heartbeat header and trailer embed some detector
// data and form the heartbeat frame. The detector data is the payload which is going
// to be sent as the payload of the O2 data packet. The HBH and HBT must be stripped
// from the payload and probably be added to the header stack and later on be stripped
// from there as well during the accumulation of subtimeframes and timeframes as the
// HB information should be identical in the (sub)timeframes

// define the data type id for the heartbeat frame
// check if this is the correct term, we probably won't send what is referred to be
// the heartbeat frame (composition of HBH - detector payload - HBT); instead, the
// HBH and HBT can be added to the header stack
extern const o2::header::DataDescription gDataDescriptionHeartbeatFrame;

struct HeartbeatHeader
{
  union {
    // the complete 64 bit header word, initialize with blockType 1 and size 1
    uint64_t headerWord = 0x1100000000000000;
    struct {
      // bit 0 to 31: orbit number
      uint32_t orbit;
      // bit 32 to 43: bunch crossing id
      uint16_t bcid:12;
      // bit 44 to 47: reserved
      uint16_t reserved:4;
      // bit 48 to 51: trigger type
      uint8_t triggerType:4;
      // bit 52 to 55: reserved
      uint8_t reservedTriggerType:4;
      // bit 56 to 59: header length
      uint8_t headerLength:4;
      // bit 60 to 63: block type (=1 HBF/trigger Header)
      uint8_t blockType:4;
    };
  }; // end union

  operator bool() const {return headerWord != 0 && blockType == 0x1;}

  bool operator<(const HeartbeatHeader& other) const {
    return this->orbit < other.orbit;
  }

  operator uint64_t() const {return headerWord;}
};
static_assert(sizeof(HeartbeatHeader) == 8, "Heartbeat header must be 64bit");

struct HeartbeatTrailer
{
  union {
    // the complete 64 bit trailer word, initialize with blockType 5 and size 1
    uint64_t trailerWord = 0x5100000000000000;
    struct {
      // bit 0 to 31: data length in words
      uint32_t dataLength;
      // bit 32 to 52: detector specific status words
      uint32_t status:21;
      // bit 53: =1 in case a new physics trigger arrived within read-out period
      uint16_t hbfTruncated:1;
      // bit 54: =0 HBF correctly transmitted
      uint16_t hbfStatus:1;
      // bit 55: =1 HBa/0 HBr received
      uint16_t hbAccept:1;
      // bit 56 to 59: trailer length
      uint16_t trailerLength:4;
      // bit 60 to 63: block type (=5 HBF Trailer)
      uint8_t blockType:4;
    };
  }; // end union

  operator bool() const {return trailerWord != 0 && blockType == 0x5;}

  operator uint64_t() const {return trailerWord;}
};
static_assert(sizeof(HeartbeatTrailer) == 8, "Heartbeat trailer must be 64bit");

// composite struct for the HBH and HBT which are the envelope for the payload
// in the heartbeat frame
// TODO: check if the copying of header and trailer can be avoided if references
// are used in a temporary object inserted to the header stack
struct HeartbeatFrameEnvelope : public BaseHeader
{
  //static data for this header type/version
  static const uint32_t sVersion;
  static const o2::header::HeaderType sHeaderType;
  static const o2::header::SerializationMethod sSerializationMethod;

  HeartbeatHeader header;
  HeartbeatTrailer trailer;

  HeartbeatFrameEnvelope()
    : BaseHeader(sizeof(HeartbeatFrameEnvelope), sHeaderType, sSerializationMethod, sVersion)
    , header(), trailer() {}

  HeartbeatFrameEnvelope(const HeartbeatHeader& h, const HeartbeatTrailer& t)
    : BaseHeader(sizeof(HeartbeatFrameEnvelope), sHeaderType, sSerializationMethod, sVersion)
    , header(h), trailer(t) {}
};

// a statistics data block for heartbeat frames
// it transmits real time as the payload of the HB frame in AliceO2
// eventually to be dropped later, its intended for the first experimental work
struct HeartbeatStatistics
{
  // time tick when this statistics was created
  uint64_t timeTickNanoSeconds;
  // difference to the previous time tick
  uint64_t durationNanoSeconds;

  HeartbeatStatistics() : timeTickNanoSeconds(0), durationNanoSeconds(0) {}
};

/**
 * @class ReverseParser
 * Parser for a sequence of frames with header, trailer and variable payload.
 * The size is expected to be part of the trailer, the parsing is thus in
 * reverse direction.
 */
template<typename HeaderT, typename TrailerT>
class ReverseParser {
public:
  using HeaderType = HeaderT;
  using TrailerType = TrailerT;

  struct FrameEntry {
    const HeaderType* header = nullptr;
    const TrailerType* trailer = nullptr;
    const byte* payload = nullptr;
    size_t length = 0;
  };
  static const size_t envelopeLength = sizeof(HeaderType) + sizeof(TrailerType);

  using CheckHeaderFct = std::function<bool(const HeaderType*)>;
  using CheckTrailerFct = std::function<bool(const TrailerType*)>;
  using GetFrameSizeFct = std::function<size_t(const TrailerType& )>;
  using InsertFct = std::function<bool(FrameEntry )>;

  int parse(const byte* buffer, size_t bufferSize,
            CheckHeaderFct checkHeader,
            CheckTrailerFct checkTrailer,
            GetFrameSizeFct getFrameSize,
            InsertFct insert) {
    if (buffer == nullptr || bufferSize == 0) return 0;
    unsigned nFrames = 0;
    auto position = bufferSize;
    do {
      FrameEntry entry;
      if (sizeof(TrailerType) > position) break;
      entry.trailer = reinterpret_cast<const TrailerType*>(buffer + position - sizeof(TrailerType));
      if (!checkTrailer(entry.trailer)) break;
      auto frameSize = getFrameSize(*entry.trailer);
      if (frameSize > position) break;
      entry.header = reinterpret_cast<const HeaderType*>(buffer + position - frameSize);
      if (!checkHeader(entry.header)) break;
      entry.payload = reinterpret_cast<const byte*>(entry.header + 1);
      entry.length = frameSize - sizeof(HeaderType) - sizeof(TrailerType);
      if (!insert(entry)) break;
      ++nFrames;
      position -= frameSize;
    } while (position > 0);

    if (position == 0) {
      // frames found and format consistent
      return nFrames;
    } else if (nFrames == 0) {
      // no frames found at all, th buffer does not contain any
      return 0;
    }

    // format error detected
    // TODO: decide about error policy
    return -1;
  }
};

/**
 * @class HeartbeatFrameSequence
 * Container class for multiple sequences of heartbeat frames
 *
 * Multiple sequences can be added as "slot", a descriptive data
 * structure per slot of type specified as template parameter can
 * be provided. Each sequence of heartbeatframes is passed recursively
 * to extract information about the individual heartbeatframes in the
 * sequence. An index is created with the slots as rows and the frames
 * as columns.
 */
template<typename SlotDataT>
class HeartbeatFrameSequence {
public:
  HeartbeatFrameSequence() {}
  ~HeartbeatFrameSequence() = default;

  using SlotDataType = SlotDataT;
  using ColumnIndexType = HeartbeatHeader;

  /// FrameIndex is composed from HB header and slot number
  struct FrameIndex {
    ColumnIndexType columnIndex;
    unsigned slot;

    bool operator<(const FrameIndex& other) const {
      // std::map.find uses the logic !(a < b) && !(b < a)
      // have to combine the two fields in one variable for
      // comparison
      uint64_t us = columnIndex;
      us &= 0x00000fffffffffff;
      us <<= 20;
      us |= slot & 0xfffff;
      uint64_t them = other.columnIndex;
      them &= 0x00000fffffffffff;
      them <<= 20;
      them |= other.slot & 0xfffff;
      return us < them;
    }
  };

  /// descriptor pointing to one frame
  struct FrameData {
    const byte* buffer = nullptr;
    size_t size = 0;
  };

  /**
   * Add a new data sequence, the set is parsed recursively
   *
   * @param slotData   Descriptive data struct for the sequence
   * @param seqData    Pointer to sequence
   * @param seqSize    Length of sequence
   * @return number of inserted elements
   */
  size_t addSlot(SlotDataType slotData, byte* seqData, size_t seqSize) {
    unsigned nFrames = mFrames.size();
    unsigned currentSlot = mSlotData.size();
    mSlotData.emplace_back(slotData);
    using ParserT = o2::header::ReverseParser<HeartbeatHeader, HeartbeatTrailer>;
    ParserT p;
    p.parse(seqData, seqSize,
            [](const typename ParserT::HeaderType* h) {return (*h);},
            [](const typename ParserT::TrailerType* t) {return (*t);},
            [](const typename ParserT::TrailerType& t) {
              return t.dataLength + ParserT::envelopeLength;
            },
            [this, currentSlot](typename ParserT::FrameEntry entry) {
              // insert the heartbeat header as column index in ascending
              // order
              auto position = mColumns.begin();
              while (position != mColumns.end() && *position < *entry.header) {
                position++;
              }
              if (position == mColumns.end() || *entry.header < *position) {
                mColumns.emplace(position, *entry.header);
              }

              // insert frame descriptor under key composed from header and slot
              auto result = mFrames.emplace(FrameIndex{*entry.header, currentSlot},
                                            FrameData{entry.payload, entry.length});
              return result.second;
            }
            );
    return mFrames.size() - nFrames;
  }

  /// clear the index, i.e. all internal lists
  void clear() {
    mFrames.clear();
    mColumns.clear();
    mSlotData.clear();
  }

  /// get number of columns in the created index
  size_t getNColumns() const {return mColumns.size();}

  /// get number of slots, i.e. number rows in the created index
  size_t getNSlots() const {return mSlotData.size();}

  /// get slot data for a data set
  const SlotDataType& getSlotData(size_t slot) const {
    if (slot < mSlotData.size()) return mSlotData[slot];
    // TODO: better to throw exception?
    static SlotDataType dummy;
    return dummy;
  }

  // TODO:
  // instead of a member with this pointer of parent class, the access
  // function was supposed to be specified as a lambda. This definition
  // was supposed to be the type of the function member.
  // passing the access function to the iterator did not work because
  // the typedef for the access function is without the capture, so there
  // is no matching conversion.
  // Solution would be to use std::function but that's probably slow and
  // the function is called often. Can be checked later.
  typedef FrameData (*AccessFct)(unsigned, unsigned);

  /// Iterator class for configurable direction, i.e. either row or column
  class iterator { // TODO: derive from forward_iterator
  public:
    using self_type = iterator;
    using value_type = FrameData;

    enum IteratorDirections {
      kAlongRow,
      kAlongColumn
    };

    iterator() = delete;
    ~iterator() = default;
    iterator(IteratorDirections direction, HeartbeatFrameSequence* parent, unsigned row = 0, unsigned column = 0)
      : mDirection(direction)
      , mRow(row)
      , mColumn(column)
      , mEnd(direction==kAlongRow?parent->getNColumns():parent->getNSlots())
      , mParent(parent)
      , mCache()
      , mIsCached(false)
    {
      while (!isValid() && !isEnd()) operator++();
    }

    self_type& operator++() {
      mIsCached = false;
      if (mDirection==kAlongRow) {
        if (mColumn<mEnd) mColumn++;
      } else {
        if (mRow<mEnd) mRow++;
      }
      while (!isEnd() && !isValid()) operator++();
      return *this;
    }

    value_type operator*() const {
      if (!mIsCached) {
        self_type* ncthis = const_cast<self_type*>(this);
        mParent->get(mRow, mColumn, ncthis->mCache);
        ncthis->mIsCached = true;
      }
      return mCache;
    }

    bool operator==(const self_type& other) const {
      return mDirection==kAlongRow?(mColumn == other.mColumn):(mRow == other.mRow);
    }

    bool operator!=(const self_type& other) const {
      return mDirection==kAlongRow?(mColumn != other.mColumn):(mRow != other.mRow);
    }

    bool isEnd() const {
      return (mDirection==kAlongRow)?(mColumn>=mEnd):(mRow>=mEnd);
    }

    bool isValid() const {
      if (!mIsCached) {
        self_type* ncthis = const_cast<self_type*>(this);
        ncthis->mIsCached = mParent->get(mRow, mColumn, ncthis->mCache);
      }
      return mIsCached;
    }

    const SlotDataType& getSlotData() const {
      static SlotDataType invalid;
      return invalid;
    }

  protected:
    IteratorDirections mDirection;
    unsigned mRow;
    unsigned mColumn;
    unsigned mEnd;
    HeartbeatFrameSequence* mParent;
    value_type mCache;
    bool mIsCached;
  };

  /// iterator for the outer access of the index, either row or column direction
  template<unsigned Direction>
  class outerIterator : public iterator {
  public:
    using base = iterator;
    using value_type = typename base::value_type;
    using self_type = outerIterator;
    static const unsigned direction = Direction;

    outerIterator() = delete;
    ~outerIterator() = default;
    outerIterator(HeartbeatFrameSequence* parent, unsigned index)
      : iterator(typename iterator::IteratorDirections(direction), parent, direction==iterator::kAlongColumn?index:0, direction==iterator::kAlongRow?index:0) {
    }

    self_type& operator++() {
      if (base::mDirection==iterator::kAlongRow) {
        if (base::mColumn<base::mEnd) base::mColumn++;
      } else {
        if (base::mRow<base::mEnd) base::mRow++;
      }
      return *this;
    }

    /// begin the inner iteration
    iterator begin() {
      return iterator((base::mDirection==iterator::kAlongColumn)?iterator::kAlongRow:iterator::kAlongColumn,
                      base::mParent,
                      (base::mDirection==iterator::kAlongColumn)?base::mRow:0,
                      (base::mDirection==iterator::kAlongRow)?base::mColumn:0);
    }

    /// end of the inner iteration
    iterator end() {
      return iterator((base::mDirection==iterator::kAlongColumn)?iterator::kAlongRow:iterator::kAlongColumn,
                      base::mParent,
                      (base::mDirection==iterator::kAlongRow)?base::mParent->getNSlots():0,
                      (base::mDirection==iterator::kAlongColumn)?base::mParent->getNColumns():0);
    }
  };

  /// definition of the outer iterator over column
  using ColumnIterator = outerIterator<iterator::kAlongRow>;
  /// definition of the outer iterator over row
  using RowIterator = outerIterator<iterator::kAlongColumn>;

  /// begin of the outer iteration
  ColumnIterator begin() {
    return ColumnIterator(this, 0);
  }

  /// end of outer iteration
  ColumnIterator end() {
    return ColumnIterator(this, mColumns.size());
  }

private:
  /// private access function for the iterators
  bool get(unsigned row, unsigned column, FrameData& data) {
    if (this->mColumns.size() == 0) return false;
    auto element = this->mFrames.find(FrameIndex{this->mColumns[column], row});
    if (element != this->mFrames.end()) {
      data = element->second;
      return true;
    }
    return false;
  }

  /// map of frame descriptors with key composed from header and slot number
  std::map<FrameIndex, FrameData> mFrames;
  /// list of indices in row direction
  std::vector<ColumnIndexType> mColumns;
  /// data descriptor of each slot forming the columns
  std::vector<SlotDataType> mSlotData;
};

};
};
#endif
