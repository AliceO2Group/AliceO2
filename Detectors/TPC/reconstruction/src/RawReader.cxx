// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

/// \file RawReader.cxx
/// \author Sebastian Klewin (Sebastian.Klewin@cern.ch)

#include <boost/tokenizer.hpp>
#include <boost/lexical_cast.hpp>
#include <list>

#include "TPCReconstruction/RawReader.h"
#include "TPCReconstruction/GBTFrame.h"
#include "TPCReconstruction/SyncPatternMonitor.h"
#include "TPCReconstruction/AdcClockMonitor.h"
#include "TPCBase/Mapper.h"
#include "TPCBase/PartitionInfo.h"

#include "FairLogger.h"

using namespace o2::TPC;

RawReader::RawReader(int region, int link, int run)
  : mUseRawInMode3(true)
  , mApplyChannelMask(false)
  , mCheckAdcClock(false)
  , mRegion(region)
  , mLink(link)
  , mRun(run)
  , mLastEvent(-1)
  , mTimestampOfFirstData({0,0,0,0,0})
  , mEvents()
  , mData()
  , mDataIterator(mData.end())
  , mSyncPos()
  , mChannelMask(nullptr)
  , mAdcError(std::make_shared<std::vector<std::tuple<short,short,short>>>())
  , mEventSynchronizer(std::make_shared<RawReaderEventSync>())
{
  mSyncPos.fill(-1);
}

bool RawReader::addInputFile(const std::vector<std::string>* infiles) {
  bool ret = false;
  for (const auto &f : *infiles) {
    ret |= addInputFile(f);
  }

  return ret;
}

bool RawReader::addInputFile(std::string infile) {
  typedef boost::tokenizer<boost::char_separator<char> >  tokenizer;
  boost::char_separator<char> sep{":"};
  tokenizer tok{infile,sep};
  if (std::distance(tok.begin(), tok.end()) < 3) {
    LOG(ERROR) << "Not enough arguments" << FairLogger::endl;
    return false;
  }

  tokenizer::iterator it1= tok.begin();
  std::string path = boost::lexical_cast<std::string>(*it1);
  int run = std::stoi(path.substr(path.find("run")+3,6));

  int region, link;
  try {
//    std::advance(it1,1);
    ++it1;
    region = boost::lexical_cast<int>(*it1);
  } catch(boost::bad_lexical_cast) {
    LOG(ERROR) << "Please enter a region for " << path << FairLogger::endl;
    return false;
  }

  try {
//    std::advance(it1,1);
    ++it1;
    link = boost::lexical_cast<int>(*it1);
  } catch(boost::bad_lexical_cast) {
    LOG(ERROR) << "Please enter a link number for " << FairLogger::endl;
    return false;
  }

  return addInputFile(region, link, path, run);
}

bool RawReader::addInputFile(int region, int link, std::string path, int run){

  if (mRun == -1) mRun = run;
  if (mRegion == -1 ) mRegion = region;
  if (mLink == -1) mLink = link;

  if (run != mRun) {
    LOG(DEBUG) << "Run of RawReader is " << mRun << " and not " << run << FairLogger::endl;
    return false;
  }

  if (region != mRegion) {
    LOG(DEBUG) << "Region of RawReader is " << mRegion << " and not " << region << FairLogger::endl;
    return false;
  }

  if (link != mLink) {
    LOG(DEBUG) << "Link of RawReader is " << mLink << " and not " << link << FairLogger::endl;
    return false;
  }

  std::ifstream file(path);
  if (!file.is_open()) {
    LOG(ERROR) << "Can't read file " << path << FairLogger::endl;
    return false;
  }

  Header h;
  int pos = 0;
  file.seekg(0,file.end);
  int length = file.tellg();
  file.seekg(0,file.beg);

  while (pos < length) {
    file.read((char*)&h, sizeof(h));
    if (h.reserved_01 != 0x0F || h.reserved_2() != 0x3fec2fec1fec0fec) {
      LOG(ERROR) << "Header does not look consistent" << FairLogger::endl;
    }
    EventInfo eD;
    eD.path = path;
    eD.posInFile =  file.tellg();
    eD.header = h;
    if (h.headerVersion == 1) {
//      auto ins = mEvents.insert(std::make_pair(h.eventCount(),std::shared_ptr<std::vector<EventInfo>>(new std::vector<EventInfo>)));
      auto ins = mEvents.insert(std::make_pair(h.eventCount(),std::make_shared<std::vector<EventInfo>>()));
      ins.first->second->push_back(eD);

      if (h.eventCount() == 0) {
        LOG(DEBUG) << "Processing event 0 to find sync pattern for event synchronization" << FairLogger::endl;
        loadEvent(0);   // processing first event to find the SYNC pattern (mode 1) or start of data (mode 2)
        mLastEvent=-1;
      }

      mEventSynchronizer->addEventOffset(
          h.eventCount(),
          (h.timeStamp() - mTimestampOfFirstData[0]));

      file.seekg(pos+(h.nWords*4));
      pos = file.tellg();
    } else {
      LOG(ERROR) << "Header version " << (int)h.headerVersion << " not implemented." << FairLogger::endl;
      return false;
    }
  }

  return true;
}

bool RawReader::loadEvent(int64_t event) {
  const Mapper& mapper = Mapper::instance();
  const PartitionInfo& partInfo = mapper.getPartitionInfo(mRegion/2);
  LOG(DEBUG) << "Loading event " << event << " for Link " << (mRegion*partInfo.getNumberOfFECs())+mLink << FairLogger::endl;

  if (mLastEvent == -1 && event != getFirstEvent() ) {
    LOG(DEBUG) << "First event was not loaded (maybe important for decoding RAW GBT frames) loading first event." << FairLogger::endl;
    mSyncPos.fill(-1);
    mTimestampOfFirstData.fill(0);
    loadEvent(getFirstEvent());
    LOG(DEBUG) << "Continue with event " << event << FairLogger::endl;
  }
  mData.clear();

  auto ev = mEvents.find(event);

  if (ev == mEvents.end()) return false;
  mLastEvent = event;

  for (auto &eventInfo : *(ev->second)) {
    switch (eventInfo.header.dataType) {
      case 1: // RAW GBT frames
        {
          LOG(DEBUG) << "Data of readout mode 1 (RAW GBT frames)" << FairLogger::endl;
          if (!decodeRawGBTFrames(eventInfo)) return false;
          break;
        }
      case 2: // Decoded data
        {
          LOG(DEBUG) << "Data of readout mode 2 (decoded data)" << FairLogger::endl;
          if (!decodePreprocessedData(eventInfo)) return false;
          break;
        }
      case 3: // both, RAW GBT frames and decoded data
        {
          if (mUseRawInMode3) {
            LOG(DEBUG) << "Data of readout mode 3 (decoding RAW GBT frames)" << FairLogger::endl;
            if (!decodeRawGBTFrames(eventInfo)) return false;
          } else {
            LOG(DEBUG) << "Data of readout mode 3 (using decoded data)" << FairLogger::endl;
            if (!decodePreprocessedData(eventInfo)) return false;
          }
          break;
        }
      default:
        {
          LOG(DEBUG) << "Readout mode not known" << FairLogger::endl;
          break;
        }
    }
  }

  mDataIterator = mData.begin();
  LOG(DEBUG) << FairLogger::endl;
  LOG(DEBUG) << FairLogger::endl;
  return true;
}

bool RawReader::decodePreprocessedData(EventInfo eventInfo) {
  std::ifstream file(eventInfo.path);
  if (!file.is_open()) {
    LOG(ERROR) << "Can't read file " << eventInfo.path << FairLogger::endl;
    return false;
  }

  int nWords = eventInfo.header.nWords-8;
  uint32_t words[nWords];
  LOG(DEBUG) << "reading " << nWords << " words from position " << eventInfo.posInFile << " in file " << eventInfo.path << FairLogger::endl;
  file.seekg(eventInfo.posInFile);
  file.read((char*)&words, nWords*sizeof(words[0]));

  const Mapper& mapper = Mapper::instance();

  std::array<char,5> sampas;
  std::array<char,5> sampaChannelStart;
  for (char i=0; i<5; ++i) {
    sampas[i] = (i == 4) ? 2 : (mRegion%2) ? i/2+3 : i/2;
    sampaChannelStart[i] = (i == 4) ?   // 5th half SAMPA corresponds to  SAMPA2
      ((mRegion%2) ? 16 : 0) :          // every even CRU receives channel 0-15 from SAMPA 2, the odd ones channel 16-31
      ((i%2) ? 16 : 0);                 // every even half SAMPA containes channel 0-15, the odd ones channel 16-31
  }


  std::array<uint32_t,5> ids;
  std::array<bool,5> writeValue;
  writeValue.fill(false);
  std::array<std::array<uint16_t,16>,5> adcValues;


  int indexStep = (eventInfo.header.dataType == 3) ? 8 : 4;
  int offset = (eventInfo.header.dataType == 3) ? 4: 0;

  long i_start = 0;
  if (eventInfo.header.eventCount() > 0) {
    LOG(DEBUG) << "Using offset [" << mTimestampOfFirstData[0] + mEventSynchronizer->getEventOffset(eventInfo.header.eventCount()) << "] "
               << "instead of [" << eventInfo.header.timeStamp() << "] "
               << " (diff = " << mTimestampOfFirstData[0] + mEventSynchronizer->getEventOffset(eventInfo.header.eventCount()) - eventInfo.header.timeStamp() << ") "
               << FairLogger::endl;

    i_start = mTimestampOfFirstData[0] + mEventSynchronizer->getEventOffset(eventInfo.header.eventCount()) - eventInfo.header.timeStamp() -1;
    i_start *= indexStep;
  }

  LOG(DEBUG) << "Start index for Event " << eventInfo.header.eventCount() << " is " << i_start << FairLogger::endl;


  for (long i=i_start; i<nWords; i=i+indexStep) {
    ids[4] = (words[i+offset] >> 4) & 0xF;
    ids[3] = (words[i+offset] >> 8) & 0xF;
    ids[2] = (words[i+offset] >> 12) & 0xF;
    ids[1] = (words[i+offset] >> 16) & 0xF;
    ids[0] = (words[i+offset] >> 20) & 0xF;

    adcValues[4][((ids[4] & 0x7)*2)+1] = (((ids[4]>>3)&0x1) == 0) ? 0 :   words[i+offset+3] & 0x3FF;
    adcValues[4][((ids[4] & 0x7)*2)  ] = (((ids[4]>>3)&0x1) == 0) ? 0 :  (words[i+offset+3] >> 10) & 0x3FF;
    adcValues[3][((ids[3] & 0x7)*2)+1] = (((ids[3]>>3)&0x1) == 0) ? 0 :  (words[i+offset+3] >> 20) & 0x3FF;
    adcValues[3][((ids[3] & 0x7)*2)  ] = (((ids[3]>>3)&0x1) == 0) ? 0 : ((words[i+offset+2] & 0xFF) << 2) | ((words[i+offset+3] >> 30) & 0x3);
    adcValues[2][((ids[2] & 0x7)*2)+1] = (((ids[2]>>3)&0x1) == 0) ? 0 :  (words[i+offset+2] >> 8) & 0x3FF;
    adcValues[2][((ids[2] & 0x7)*2)  ] = (((ids[2]>>3)&0x1) == 0) ? 0 :  (words[i+offset+2] >> 18) & 0x3FF;
    adcValues[1][((ids[1] & 0x7)*2)+1] = (((ids[1]>>3)&0x1) == 0) ? 0 : ((words[i+offset+1] & 0x3F) << 4) | ((words[i+offset+2] >> 28) & 0xF);
    adcValues[1][((ids[1] & 0x7)*2)  ] = (((ids[1]>>3)&0x1) == 0) ? 0 :  (words[i+offset+1] >> 6) & 0x3FF;
    adcValues[0][((ids[0] & 0x7)*2)+1] = (((ids[0]>>3)&0x1) == 0) ? 0 :  (words[i+offset+1] >> 16) & 0x3FF;
    adcValues[0][((ids[0] & 0x7)*2)  ] = (((ids[0]>>3)&0x1) == 0) ? 0 : ((words[i+offset+0] & 0xF) << 6) | ((words[i+offset+1] >> 26) & 0x3F);

    for (char j=0; j<5; ++j) {
      if (ids[j] == 0x8) {
        writeValue[j] = true;
        // header TS is one before first word -> +1
        if (mTimestampOfFirstData[j] == 0) mTimestampOfFirstData[j] = eventInfo.header.timeStamp() + 1 + i/indexStep;
      }
    }

    for (char j=0; j<5; ++j) {
      if (writeValue[j] & (ids[j] == 0xF)) {
        for (int k=0; k<16; ++k) {
          const PadPos padPos = mapper.padPosRegion(
              mRegion, mLink,sampas[j],k+sampaChannelStart[j]);

          if (mApplyChannelMask &&          // channel mask should be applied
              (mChannelMask != nullptr) &&  // channel mask is available
              !mChannelMask->getValue(CRU(mRegion),padPos.getPad(),padPos.getRow())) { // channel mask
              continue;
          }

//          auto ins = mData.insert(std::make_pair(padPos, std::shared_ptr<std::vector<uint16_t>>(new std::vector<uint16_t>)));
          auto ins = mData.insert(std::make_pair(padPos, std::make_shared<std::vector<uint16_t>>()));
          ins.first->second->push_back(adcValues[j][k]);
        }
      }
    }

  }
  return true;
}

bool RawReader::decodeRawGBTFrames(EventInfo eventInfo) {
  std::ifstream file(eventInfo.path);
  if (!file.is_open()) {
    LOG(ERROR) << "Can't read file " << eventInfo.path << FairLogger::endl;
    return false;
  }

  int nWords = eventInfo.header.nWords-8;
  uint32_t words[nWords];
  LOG(DEBUG) << "reading " << nWords << " words from position " << eventInfo.posInFile << " in file " << eventInfo.path << FairLogger::endl;
  LOG(DEBUG) << "Header time stamp is " << eventInfo.header.timeStamp() << FairLogger::endl;
  file.seekg(eventInfo.posInFile);
  file.read((char*)&words, nWords*sizeof(words[0]));

  const Mapper& mapper = Mapper::instance();

  std::array<char,5> sampas;
  std::array<char,5> sampaChannelStart;
  for (char i=0; i<5; ++i) {
    sampas[i] = (i == 4) ? 2 : (mRegion%2) ? i/2+3 : i/2;
    sampaChannelStart[i] = (i == 4) ?   // 5th half SAMPA corresponds to  SAMPA2
      ((mRegion%2) ? 16 : 0) :          // every even CRU receives channel 0-15 from SAMPA 2, the odd ones channel 16-31
      ((i%2) ? 16 : 0);                 // every even half SAMPA containes channel 0-15, the odd ones channel 16-31
  }

  std::array<SyncPatternMonitor,5> syncMon{
    SyncPatternMonitor((mRegion%2) ? 3 : 0,0),
    SyncPatternMonitor((mRegion%2) ? 3 : 0,1),
    SyncPatternMonitor((mRegion%2) ? 4 : 1,0),
    SyncPatternMonitor((mRegion%2) ? 4 : 1,1),
    SyncPatternMonitor(2,(mRegion%2) ? 1 : 0)};
  std::array<AdcClockMonitor,3> adcClockMon{
    AdcClockMonitor((mRegion%2) ? 3 : 0),
    AdcClockMonitor((mRegion%2) ? 4 : 1),
    AdcClockMonitor(2)};
  std::array<bool,3> adcClockFound{false, false, false};

  std::array<short,5> lastSyncPos;
  std::array<std::list<uint16_t>,5> adcValues;
  mAdcError->clear();
  uint64_t timebin = 0;

  int indexStep = (eventInfo.header.dataType == 3) ? 8 : 4;

  long i_start = 0;
  if (eventInfo.header.eventCount() > 0) {
    LOG(DEBUG) << "Using offset [" << mTimestampOfFirstData[0] + mEventSynchronizer->getEventOffset(eventInfo.header.eventCount()) << "] "
               << "instead of [" << eventInfo.header.timeStamp() << "] "
               << " (diff = " << mTimestampOfFirstData[0] + mEventSynchronizer->getEventOffset(eventInfo.header.eventCount()) - eventInfo.header.timeStamp() << ") "
               << FairLogger::endl;

    i_start = mTimestampOfFirstData[0] + mEventSynchronizer->getEventOffset(eventInfo.header.eventCount()) - eventInfo.header.timeStamp() -1;
    i_start *= indexStep;
  }

  LOG(DEBUG) << "Start index for Event " << eventInfo.header.eventCount() << " is " << i_start << FairLogger::endl;

  GBTFrame frame;
  if (eventInfo.header.eventCount() > 0) frame.setData(words[i_start-indexStep],words[i_start-indexStep+1],words[i_start-indexStep+2],words[i_start-indexStep+3]);
  GBTFrame lastFrame;

  for (long i=i_start; i<nWords; i=i+indexStep) {

    lastSyncPos = mSyncPos;
    lastFrame = frame;
    frame.setData(words[i],words[i+1],words[i+2],words[i+3]);

    if (syncMon[0].addSequence(frame.getHalfWord(0,0,0), frame.getHalfWord(0,1,0), frame.getHalfWord(0,2,0), frame.getHalfWord(0,3,0))) mSyncPos[0] = syncMon[0].getPosition();
    if (syncMon[1].addSequence(frame.getHalfWord(0,0,1), frame.getHalfWord(0,1,1), frame.getHalfWord(0,2,1), frame.getHalfWord(0,3,1))) mSyncPos[1] = syncMon[1].getPosition();
    if (syncMon[2].addSequence(frame.getHalfWord(1,0,0), frame.getHalfWord(1,1,0), frame.getHalfWord(1,2,0), frame.getHalfWord(1,3,0))) mSyncPos[2] = syncMon[2].getPosition();
    if (syncMon[3].addSequence(frame.getHalfWord(1,0,1), frame.getHalfWord(1,1,1), frame.getHalfWord(1,2,1), frame.getHalfWord(1,3,1))) mSyncPos[3] = syncMon[3].getPosition();
    if (syncMon[4].addSequence(frame.getHalfWord(2,0),   frame.getHalfWord(2,1),   frame.getHalfWord(2,2),   frame.getHalfWord(2,3)))   mSyncPos[4] = syncMon[4].getPosition();

    if (mCheckAdcClock) {

      bool adcCheckErr0 = adcClockMon[0].addSequence(frame.getAdcClock(0));
      bool adcCheckErr1 = adcClockMon[1].addSequence(frame.getAdcClock(1));
      bool adcCheckErr2 = adcClockMon[2].addSequence(frame.getAdcClock(2));

      if (mSyncPos[0] >= 0) adcClockFound[0] = adcClockFound[0] | !adcCheckErr0;
      if (mSyncPos[2] >= 0) adcClockFound[1] = adcClockFound[1] | !adcCheckErr1;
      if (mSyncPos[4] >= 0) adcClockFound[2] = adcClockFound[2] | !adcCheckErr2;
      if(adcClockFound[0] & adcCheckErr0) {
        adcClockFound[0] = false;
        LOG(DEBUG) << "ADC clock error of SAMPA " << ((mRegion%2) ? 3 : 0) << " in frame [" << i/indexStep << "]"  << FairLogger::endl;
        mAdcError->emplace_back(std::make_tuple((mRegion%2) ? 3 : 0,i,mSyncPos[0]));
      };
      if(adcClockFound[1] & adcCheckErr1) {
        adcClockFound[1] = false;
        LOG(DEBUG) << "ADC clock error of SAMPA " << ((mRegion%2) ? 4 : 1) << " in frame [" << i/indexStep << "]"  << FairLogger::endl;
        mAdcError->emplace_back(std::make_tuple((mRegion%2) ? 4 : 1,i,mSyncPos[2]));
      };
      if(adcClockFound[2] & adcCheckErr2) {
        adcClockFound[2] = false;
        LOG(DEBUG) << "ADC clock error of SAMPA " << 2 << " in frame [" << i/indexStep << "]"  << FairLogger::endl;
        mAdcError->emplace_back(std::make_tuple(2,i,mSyncPos[4]));
      };
    }

    short value1;
    short value2;
    for (short iHalfSampa = 0; iHalfSampa < 5; ++iHalfSampa) {
      if (mSyncPos[iHalfSampa] < 0 ) {
        LOG(DEBUG1) << "Sync pattern for half SAMPA " << iHalfSampa << " not yet found" << FairLogger::endl;
        continue;
      }
      if (lastSyncPos[iHalfSampa] < 0 ) {
        LOG(DEBUG1) << "Sync pattern for half SAMPA " << iHalfSampa << " not yet found" << FairLogger::endl;
        LOG(DEBUG1) << lastFrame << FairLogger::endl;
        continue;
      }
      if (mTimestampOfFirstData[iHalfSampa] == 0) {
        if(iHalfSampa == 0) LOG(DEBUG) << "Setting timestamp of first data to " << eventInfo.header.timeStamp() + 1 + i/indexStep << " for half SAMPA " << iHalfSampa << " (" << 1 + i/indexStep << ")" << FairLogger::endl;
        mTimestampOfFirstData[iHalfSampa] = eventInfo.header.timeStamp() + 1 + i/indexStep;
      }

      switch(mSyncPos[iHalfSampa]) {
        case 0:
          value1 = (frame.getHalfWord(iHalfSampa/2,1,iHalfSampa%2) << 5) | frame.getHalfWord(iHalfSampa/2,0,iHalfSampa%2);
          value2 = (frame.getHalfWord(iHalfSampa/2,3,iHalfSampa%2) << 5) | frame.getHalfWord(iHalfSampa/2,2,iHalfSampa%2);
          break;

        case 1:
          value1 = (lastFrame.getHalfWord(iHalfSampa/2,2,iHalfSampa%2) << 5) | lastFrame.getHalfWord(iHalfSampa/2,1,iHalfSampa%2);
          value2 =     (frame.getHalfWord(iHalfSampa/2,0,iHalfSampa%2) << 5) | lastFrame.getHalfWord(iHalfSampa/2,3,iHalfSampa%2);
          break;

        case 2:
          value1 = (lastFrame.getHalfWord(iHalfSampa/2,3,iHalfSampa%2) << 5) | lastFrame.getHalfWord(iHalfSampa/2,2,iHalfSampa%2);
          value2 =     (frame.getHalfWord(iHalfSampa/2,1,iHalfSampa%2) << 5) |     frame.getHalfWord(iHalfSampa/2,0,iHalfSampa%2);
          break;

        case 3:
          value1 = (frame.getHalfWord(iHalfSampa/2,0,iHalfSampa%2) << 5) | lastFrame.getHalfWord(iHalfSampa/2,3,iHalfSampa%2);
          value2 = (frame.getHalfWord(iHalfSampa/2,2,iHalfSampa%2) << 5) |     frame.getHalfWord(iHalfSampa/2,1,iHalfSampa%2);
          break;

        default:
          return false;
      }

      adcValues[iHalfSampa].emplace_back( value1 ^ (1 << 9));
      adcValues[iHalfSampa].emplace_back( value2 ^ (1 << 9));
    }

    for (char j=0; j<5; ++j) {
      if (adcValues[j].size() == 16) {
        for (int k=0; k<16; ++k) {
          const PadPos padPos = mapper.padPosRegion(
              mRegion, mLink,sampas[j],k+sampaChannelStart[j]);

          if (mApplyChannelMask &&          // channel mask should be applied
              (mChannelMask != nullptr) &&  // channel mask is available
              !mChannelMask->getValue(CRU(mRegion),padPos.getPad(),padPos.getRow())) {     // channel mask
            adcValues[j].pop_front();           // discard value
            continue;
          }

          //          auto ins = mData.insert(std::make_pair(padPos, std::shared_ptr<std::vector<uint16_t>>(new std::vector<uint16_t>)));
          auto ins = mData.insert(std::make_pair(padPos, std::make_shared<std::vector<uint16_t>>()));
          ins.first->second->push_back(adcValues[j].front());
          adcValues[j].pop_front();
        }
      }
    }
  }
  return true;
}

