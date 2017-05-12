#ifndef O2_TPC_RAWREADER_H_
#define O2_TPC_RAWREADER_H_

/// \file RawReader.h
/// \author Sebastian Klewin (Sebastian.Klewin@cern.ch)

#include <string>
#include <vector>
#include <map>
#include <memory>

#include "TPCBase/PadPos.h"
namespace o2 {
namespace TPC {

/// \class RawReader
/// \brief Reader for RAW TPC data
/// \author Sebastian Klewin (Sebastian.Klewin@cern.ch)
class RawReader {
  public:

    /// Default constructor
    RawReader();

    /// Destructor
    ~RawReader() = default;

    bool loadNextEvent() { return loadEvent(mLastEvent+1); };
    bool loadEvent(int64_t event);
    bool addInputFile(const std::vector<std::string>* infiles);
    bool addInputFile(std::string infile);
    bool addInputFile(int region, int link, std::string infile);
    uint64_t getFirstEvent() const { return mEvents.begin()->first; };
    uint64_t getLastEvent() const {  return (mEvents.size() == 0) ? mEvents.begin()->first : mEvents.rbegin()->first; };
    int getNumberOfEvents() const { return  (mEvents.size() == 0) ? 0 : mEvents.rbegin()->first - mEvents.begin()->first; };

    std::shared_ptr<std::vector<uint16_t>> getData(const PadPos& padPos);
    std::shared_ptr<std::vector<uint16_t>> getNextData(PadPos& padPos);

  private:

    struct header {
      uint16_t dataType;
      uint8_t channelID;
      uint8_t headerVersion;
      uint32_t nWords;
      uint32_t timeStamp_h;
      uint32_t timeStamp_l;
      uint32_t eventCount_h;
      uint32_t eventCount_l;
      uint32_t reserved_h;
      uint32_t reserved_l;

      uint64_t timeStamp() { return (((uint64_t)timeStamp_h)<<32) | timeStamp_l;}
      uint64_t eventCount() { return (((uint64_t)eventCount_h)<<32) | eventCount_l;}
      uint64_t reserved() { return (((uint64_t)reserved_h)<<32) | reserved_l;}

      header() {};
      header(const header& h) : dataType(h.dataType), channelID(h.channelID), 
        headerVersion(h.headerVersion), nWords(h.nWords), timeStamp_h(h.timeStamp_h),
        timeStamp_l(h.timeStamp_l), eventCount_h(h.eventCount_h), eventCount_l(h.eventCount_l),
        reserved_h(h.reserved_h), reserved_l(h.reserved_l) {};
    };

    struct eventData {
      std::string path;
      int posInFile;
      int region;
      int link;
      bool isProcessed;
      header headerInfo;

      eventData() : path(""), posInFile(-1), region(-1), link(-1), headerInfo() {};
      eventData(const eventData& e) : path(e.path), posInFile(e.posInFile), region(e.region),
        link(e.link), headerInfo(e.headerInfo) {};
    };

    int64_t mLastEvent;
    std::map<uint64_t, std::unique_ptr<std::vector<eventData>>> mEvents;
    std::map<PadPos,std::shared_ptr<std::vector<uint16_t>>> mData;
    std::map<PadPos,std::shared_ptr<std::vector<uint16_t>>>::iterator mDataIterator;
};

inline
std::shared_ptr<std::vector<uint16_t>> RawReader::getData(const PadPos& padPos) {
  mDataIterator = mData.find(padPos);
  if (mDataIterator == mData.end()) {
    std::shared_ptr<std::vector<uint16_t>> emptyVecPtr(new std::vector<uint16_t>{});
    return emptyVecPtr;
  }
  return mDataIterator->second; 
};

inline
std::shared_ptr<std::vector<uint16_t>> RawReader::getNextData(PadPos& padPos) { 
  if (mDataIterator == mData.end()) return nullptr;
  std::map<PadPos,std::shared_ptr<std::vector<uint16_t>>>::iterator last = mDataIterator;
  mDataIterator++;
  padPos = last->first;
  return last->second; 
};

}
}
#endif
