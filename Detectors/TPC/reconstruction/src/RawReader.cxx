/// \file RawReader.cxx
/// \author Sebastian Klewin (Sebastian.Klewin@cern.ch)

#include <boost/tokenizer.hpp>
#include <boost/lexical_cast.hpp>
#include <iostream>
#include <queue>

#include "TPCReconstruction/RawReader.h"
#include "TPCReconstruction/GBTFrame.h"
#include "TPCReconstruction/SyncPatternMonitor.h" 
#include "TPCBase/Mapper.h"

#include "FairLogger.h" 

using namespace o2::TPC;

RawReader::RawReader(int region, int link)
  : mRegion(region)
  , mLink(link)
  , mLastEvent(-1)
  , mUseRawInMode3(false)
  , mApplyChannelMask(false)
  , mTimestampOfFirstData({0,0,0,0,0})
  , mEvents()
  , mData()
  , mDataIterator(mData.end())
  , mSyncPos()
  , mChannelMask(nullptr)
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

  return addInputFile(region,link,path);
}

bool RawReader::addInputFile(int region, int link, std::string path){

  if (mRegion == -1 ) mRegion = region;
  if (mLink == -1) mLink = link;

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
    eventData eD;
    eD.path = path;
    eD.posInFile =  file.tellg();
    eD.region = region;
    eD.link = link;
    eD.headerInfo = h;
    if (h.headerVersion == 1) {
      auto it = mEvents.find(h.eventCount());
      if (it != mEvents.end()) it->second->push_back(eD);
      //else mEvents.insert(std::pair< uint64_t, std::shared_ptr<std::vector<eventData>>> (h.eventCount(),new std::vector<eventData>{eD}));
      else mEvents.insert(std::make_pair(h.eventCount(),std::shared_ptr<std::vector<eventData>>(new std::vector<eventData>{eD})));
//      std::cout << mEvents.size() << " " << mEvents[h.eventCount()]->size() << std::endl
//        << mEvents[h.eventCount()]->at(0).headerInfo.eventCount() << std::endl;
//      std::cout 
//        << CHAR_BIT << " "
//        << "Size: " << sizeof(h) 
//        << " V: " << (int)h.headerVersion
//        << " ID: " << (int)(h.channelID >> 4)
//        << " T: " << (int)h.dataType
//        << " W: " << h.nWords
//        << " TS: " << h.timeStamp()
//        << " EC: " << h.eventCount()
//        << " R: " << h.reserved()
//        << std::endl;
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
  LOG(DEBUG) << "Loading new event " << event << FairLogger::endl;
  mData.clear();

  auto ev = mEvents.find(event);
  mLastEvent = event;

  if (ev == mEvents.end()) return false;
  
  const Mapper& mapper = Mapper::instance();
  for (auto &data : *(ev->second)) {
    switch (data.headerInfo.dataType) {
      case 1: // RAW GBT frames
        { 
          LOG(DEBUG) << "Data of readout mode 1 (RAW GBT frames)" << FairLogger::endl;
          if (!decodeRawGBTFrames(data)) return false;
          break;
        }
      case 2: // Decoded data
        {
          LOG(DEBUG) << "Data of readout mode 2 (decoded data)" << FairLogger::endl;
          if (!decodePreprocessedData(data)) return false;
          break;
        }
      case 3: // both, RAW GBT frames and decoded data
        {
          if (mUseRawInMode3) {
            LOG(DEBUG) << "Data of readout mode 3 (decoding RAW GBT frames)" << FairLogger::endl;
            if (!decodeRawGBTFrames(data)) return false;
          } else {
            LOG(DEBUG) << "Data of readout mode 3 (using decoded data)" << FairLogger::endl;
            if (!decodePreprocessedData(data)) return false;
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
  return true;
}

bool RawReader::decodePreprocessedData(eventData dataHeader) {
  std::ifstream file(dataHeader.path);
  if (!file.is_open()) {
    LOG(ERROR) << "Can't read file " << dataHeader.path << FairLogger::endl;
    return false;
  }

  int nWords = dataHeader.headerInfo.nWords-8;
  uint32_t words[nWords];
  LOG(DEBUG) << "reading " << nWords << " words from position " << dataHeader.posInFile << " in file " << dataHeader.path << FairLogger::endl;
  file.seekg(dataHeader.posInFile);
  file.read((char*)&words, nWords*sizeof(words[0]));

  const Mapper& mapper = Mapper::instance();

  std::array<char,5> sampas;
  std::array<char,5> sampaChannelStart;
  for (char i=0; i<5; ++i) {
    sampas[i] = (i == 4) ? 2 : (dataHeader.region%2) ? i/2+3 : i/2;
    sampaChannelStart[i] = (i == 4) ?   // 5th half SAMPA corresponds to  SAMPA2
      ((dataHeader.region%2) ? 16 : 0) :      // every even CRU receives channel 0-15 from SAMPA 2, the odd ones channel 16-31
      ((i%2) ? 16 : 0);                 // every even half SAMPA containes channel 0-15, the odd ones channel 16-31
  }


  std::array<uint32_t,5> ids;
  std::array<bool,5> writeValue;
  writeValue.fill(false);
  std::array<std::array<uint16_t,16>,5> adcValues;


  int indexStep = (dataHeader.headerInfo.dataType == 3) ? 8 : 4;
  int offset = (dataHeader.headerInfo.dataType == 3) ? 4: 0;
  for (int i=0; i<nWords; i=i+indexStep) {
    ids[4] = (words[i+offset] >> 4) & 0xF;
    ids[3] = (words[i+offset] >> 8) & 0xF;
    ids[2] = (words[i+offset] >> 12) & 0xF;
    ids[1] = (words[i+offset] >> 16) & 0xF;
    ids[0] = (words[i+offset] >> 20) & 0xF;

    adcValues[4][((ids[4] & 0x7)*2)+1] = ((ids[4]>>3)&0x1 == 0) ? 0 :   words[i+offset+3] & 0x3FF;
    adcValues[4][((ids[4] & 0x7)*2)  ] = ((ids[4]>>3)&0x1 == 0) ? 0 :  (words[i+offset+3] >> 10) & 0x3FF;
    adcValues[3][((ids[3] & 0x7)*2)+1] = ((ids[3]>>3)&0x1 == 0) ? 0 :  (words[i+offset+3] >> 20) & 0x3FF;
    adcValues[3][((ids[3] & 0x7)*2)  ] = ((ids[3]>>3)&0x1 == 0) ? 0 : ((words[i+offset+2] & 0xFF) << 2) | ((words[i+3] >> 30) & 0x3);
    adcValues[2][((ids[2] & 0x7)*2)+1] = ((ids[2]>>3)&0x1 == 0) ? 0 :  (words[i+offset+2] >> 8) & 0x3FF;
    adcValues[2][((ids[2] & 0x7)*2)  ] = ((ids[2]>>3)&0x1 == 0) ? 0 :  (words[i+offset+2] >> 18) & 0x3FF;
    adcValues[1][((ids[1] & 0x7)*2)+1] = ((ids[1]>>3)&0x1 == 0) ? 0 : ((words[i+offset+1] & 0x3F) << 4) | ((words[i+2] >> 28) & 0xF);
    adcValues[1][((ids[1] & 0x7)*2)  ] = ((ids[1]>>3)&0x1 == 0) ? 0 :  (words[i+offset+1] >> 6) & 0x3FF;
    adcValues[0][((ids[0] & 0x7)*2)+1] = ((ids[0]>>3)&0x1 == 0) ? 0 :  (words[i+offset+1] >> 16) & 0x3FF;
    adcValues[0][((ids[0] & 0x7)*2)  ] = ((ids[0]>>3)&0x1 == 0) ? 0 : ((words[i+offset+0] & 0xF) << 6) | ((words[i+1] >> 26) & 0x3F);

    for (char j=0; j<5; ++j) {
      if (ids[j] == 0x8) {
        writeValue[j] = true;
        // header TS is one before first word -> +1
        if (mTimestampOfFirstData[j] == 0) mTimestampOfFirstData[j] = dataHeader.headerInfo.timeStamp() + 1 + i/8;
      }
    }

    for (char j=0; j<5; ++j) {
      if (writeValue[j] & (ids[j] == 0xF)) {
        for (int k=0; k<16; ++k) {
          const PadPos padPos = mapper.padPosRegion(
              dataHeader.region, dataHeader.link,sampas[j],k+sampaChannelStart[j]);

          if (mApplyChannelMask &&          // channel mask should be applied
              (mChannelMask != nullptr) &&  // channel mask is available
              mChannelMask->getValue(CRU(dataHeader.region),padPos.getPad(),padPos.getRow())) { // channel mask
              continue;
          }

          auto ins = mData.insert(std::make_pair(padPos, std::shared_ptr<std::vector<uint16_t>>(new std::vector<uint16_t>)));
          ins.first->second->push_back(adcValues[j][k]);
        }
      }
    }

  }
  return true;
}

bool RawReader::decodeRawGBTFrames(eventData dataHeader) {
  std::ifstream file(dataHeader.path);
  if (!file.is_open()) {
    LOG(ERROR) << "Can't read file " << dataHeader.path << FairLogger::endl;
    return false;
  }

  int nWords = dataHeader.headerInfo.nWords-8;
  uint32_t words[nWords];
  LOG(DEBUG) << "reading " << nWords << " words from position " << dataHeader.posInFile << " in file " << dataHeader.path << FairLogger::endl;
  file.seekg(dataHeader.posInFile);
  file.read((char*)&words, nWords*sizeof(words[0]));

  const Mapper& mapper = Mapper::instance();

  std::array<char,5> sampas;
  std::array<char,5> sampaChannelStart;
  for (char i=0; i<5; ++i) {
    sampas[i] = (i == 4) ? 2 : (dataHeader.region%2) ? i/2+3 : i/2;
    sampaChannelStart[i] = (i == 4) ?   // 5th half SAMPA corresponds to  SAMPA2
      ((dataHeader.region%2) ? 16 : 0) :      // every even CRU receives channel 0-15 from SAMPA 2, the odd ones channel 16-31
      ((i%2) ? 16 : 0);                 // every even half SAMPA containes channel 0-15, the odd ones channel 16-31
  }

  std::array<SyncPatternMonitor,5> syncMon{
    SyncPatternMonitor(0,0),
      SyncPatternMonitor(0,1),
      SyncPatternMonitor(1,0),
      SyncPatternMonitor(1,1),
      SyncPatternMonitor(2,0)};
  std::array<short,5> lastSyncPos;
  std::array<std::queue<uint16_t>,5> adcValues;
  GBTFrame frame; 
  GBTFrame lastFrame; 

  int indexStep = (dataHeader.headerInfo.dataType == 3) ? 8 : 4;
  for (int i=0; i<nWords; i=i+indexStep) {

    for (char j=0; j<5; ++j) {
      if ((mTimestampOfFirstData[j] != 0) && 
          ((mTimestampOfFirstData[j] & 0x7) == ((dataHeader.headerInfo.timeStamp() + 1 + i/indexStep) & 0x7))) {
        if (adcValues[j].size() < 16) {
          std::queue<uint16_t> empty;
          std::swap(adcValues[j], empty);
          continue;
        }
        for (int k=0; k<16; ++k) {
          const PadPos padPos = mapper.padPosRegion(
              dataHeader.region, dataHeader.link,sampas[j],k+sampaChannelStart[j]);

          if (mApplyChannelMask &&          // channel mask should be applied
              (mChannelMask != nullptr) &&  // channel mask is available
              mChannelMask->getValue(CRU(dataHeader.region),padPos.getPad(),padPos.getRow())) {     // channel mask
              adcValues[j].pop();           // discard value
              continue;
          }

          auto ins = mData.insert(std::make_pair(padPos, std::shared_ptr<std::vector<uint16_t>>(new std::vector<uint16_t>)));
          ins.first->second->push_back(adcValues[j].front());
          adcValues[j].pop();
        }
      }
    }

    lastSyncPos = mSyncPos;
    lastFrame = frame;
    frame.setData(words[i],words[i+1],words[i+2],words[i+3]);

    if (syncMon[0].addSequence(
          frame.getHalfWord(0,0,0),
          frame.getHalfWord(0,1,0),
          frame.getHalfWord(0,2,0),
          frame.getHalfWord(0,3,0))) mSyncPos[0] = syncMon[0].getPosition();

    if (syncMon[1].addSequence(
          frame.getHalfWord(0,0,1),
          frame.getHalfWord(0,1,1),
          frame.getHalfWord(0,2,1),
          frame.getHalfWord(0,3,1))) mSyncPos[1] = syncMon[1].getPosition();

    if (syncMon[2].addSequence(
          frame.getHalfWord(1,0,0),
          frame.getHalfWord(1,1,0),
          frame.getHalfWord(1,2,0),
          frame.getHalfWord(1,3,0))) mSyncPos[2] = syncMon[2].getPosition();

    if (syncMon[3].addSequence(
          frame.getHalfWord(1,0,1),
          frame.getHalfWord(1,1,1),
          frame.getHalfWord(1,2,1),
          frame.getHalfWord(1,3,1))) mSyncPos[3] = syncMon[3].getPosition();

    if (syncMon[4].addSequence(
          frame.getHalfWord(2,0),
          frame.getHalfWord(2,1),
          frame.getHalfWord(2,2),
          frame.getHalfWord(2,3))) mSyncPos[4] = syncMon[4].getPosition();

    short value1;
    short value2;
    for (short iHalfSampa = 0; iHalfSampa < 5; ++iHalfSampa) {
      if (i==0) continue;
      if (mSyncPos[iHalfSampa] < 0) continue;
      if (lastSyncPos[iHalfSampa] < 0) continue;
      if (mTimestampOfFirstData[iHalfSampa] == 0) mTimestampOfFirstData[iHalfSampa] = dataHeader.headerInfo.timeStamp() + 1 + i/4;

      switch(mSyncPos[iHalfSampa]) {
        case 0:
          value1 = (frame.getHalfWord(iHalfSampa/2,1,iHalfSampa%2) << 5) |
            frame.getHalfWord(iHalfSampa/2,0,iHalfSampa%2);
          value2 = (frame.getHalfWord(iHalfSampa/2,3,iHalfSampa%2) << 5) | 
            frame.getHalfWord(iHalfSampa/2,2,iHalfSampa%2);
          break;

        case 1:
          value1 = (lastFrame.getHalfWord(iHalfSampa/2,2,iHalfSampa%2) << 5) |
            lastFrame.getHalfWord(iHalfSampa/2,1,iHalfSampa%2);
          value2 = (frame.getHalfWord(iHalfSampa/2,0,iHalfSampa%2) << 5) |
            lastFrame.getHalfWord(iHalfSampa/2,3,iHalfSampa%2);
          break;

        case 2:
          value1 = (lastFrame.getHalfWord(iHalfSampa/2,3,iHalfSampa%2) << 5) |
            lastFrame.getHalfWord(iHalfSampa/2,2,iHalfSampa%2);
          value2 = (frame.getHalfWord(iHalfSampa/2,1,iHalfSampa%2) << 5) | 
            frame.getHalfWord(iHalfSampa/2,0,iHalfSampa%2);
          break;

        case 3:
          value1 = (frame.getHalfWord(iHalfSampa/2,0,iHalfSampa%2) << 5) |
            lastFrame.getHalfWord(iHalfSampa/2,3,iHalfSampa%2);
          value2 = (frame.getHalfWord(iHalfSampa/2,2,iHalfSampa%2) << 5) | 
            frame.getHalfWord(iHalfSampa/2,1,iHalfSampa%2);
          break;

        default:
          return false;
      }

      adcValues[iHalfSampa].emplace(value1 ^ (1 << 9));
      adcValues[iHalfSampa].emplace(value2 ^ (1 << 9));
    }
  }
  return true;
}

