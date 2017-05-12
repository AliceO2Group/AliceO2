/// \file RawReader.cxx
/// \author Sebastian Klewin (Sebastian.Klewin@cern.ch)

#include <boost/tokenizer.hpp>
#include <boost/lexical_cast.hpp>
#include <iostream>
#include <bitset>

#include "TPCReconstruction/RawReader.h"
#include "TPCBase/Mapper.h"

#include "FairLogger.h" 

using namespace o2::TPC;

RawReader::RawReader()
  : mLastEvent(-1)
  , mEvents()
  , mData()
  , mDataIterator(mData.end())
{}

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

bool RawReader::addInputFile(int region, int link, std::string infile){

  std::ifstream file(infile);
  if (!file.is_open()) {
    LOG(ERROR) << "Can't read file " << infile << FairLogger::endl;
    return false;
  }

  header h;
  int pos = 0;
  file.seekg(0,file.end);
  int length = file.tellg();
  file.seekg(0,file.beg);

  while (pos < length) {
    file.read((char*)&h, sizeof(h));
    eventData eD;
    eD.path = infile;
    eD.posInFile =  file.tellg();
    eD.region = region;
    eD.link = link;
    eD.isProcessed = false;
    eD.headerInfo = h;
    if (h.headerVersion == 0) {
      auto it = mEvents.find(h.eventCount());
      if (it != mEvents.end()) it->second->push_back(eD);
      else mEvents.insert(std::pair< uint64_t, std::unique_ptr<std::vector<eventData>>> (h.eventCount(),new std::vector<eventData>{eD}));
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
      LOG(ERROR) << "Header version " << h.headerVersion << " not implemented." << FairLogger::endl;
      return false;
    }
  }

  return true;
}

bool RawReader::loadEvent(int64_t event) {
  mData.clear();
  auto ev = mEvents.find(event);
  mLastEvent = event;

  if (ev == mEvents.end()) return false;
  
  const Mapper& mapper = Mapper::instance();
  for (auto &data : *(ev->second)) {
    std::ifstream file(data.path);
    if (!file.is_open()) {
      LOG(ERROR) << "Can't read file " << data.path << FairLogger::endl;
      return false;
    }
    int nWords = data.headerInfo.nWords-8;
    uint32_t words[nWords];
    file.seekg(data.posInFile);
    file.read((char*)&words, nWords*sizeof(words[0]));
    LOG(DEBUG) << "reading " << nWords << " from file " << data.path << FairLogger::endl;

    switch (data.headerInfo.dataType) {
      case 1: // RAW GBT frames
        { 
          LOG(DEBUG) << "Readout Mode 1 not yet implemented" << FairLogger::endl;
          break;
        }
      case 2: // Decoded data
        {
          LOG(DEBUG) << "Data of readout mode 2 (decoded data)" << FairLogger::endl;
          std::array<uint32_t,5> ids;
          std::array<bool,5> writeValue;
          writeValue.fill(false);
          std::array<std::array<uint16_t,16>,5> adcValues;
          std::array<char,5> sampas;
          std::array<char,5> sampaChannelStart;
          for (char i=0; i<5; ++i) {
            sampas[i] = (i == 4) ? 2 : (data.region%2) ? i/2+3 : i/2;
            sampaChannelStart[i] = (i == 4) ?   // 5th half SAMPA corresponds to  SAMPA2
              ((data.region%2) ? 16 : 0) :      // every even CRU receives channel 0-15 from SAMPA 2, the odd ones channel 16-31
              ((i%2) ? 16 : 0);                 // every even half SAMPA containes channel 0-15, the odd ones channel 16-31
          }

          for (int i=0; i<nWords; i=i+4) {
            ids[4] = (words[i] >> 4) & 0xF;
            ids[3] = (words[i] >> 8) & 0xF;
            ids[2] = (words[i] >> 12) & 0xF;
            ids[1] = (words[i] >> 16) & 0xF;
            ids[0] = (words[i] >> 20) & 0xF;

            adcValues[4][((ids[4] & 0x7)*2)+1] = ((ids[4]>>3)&0x1 == 0) ? 0 : words[i+3] & 0x3FF;
            adcValues[4][((ids[4] & 0x7)*2)  ] = ((ids[4]>>3)&0x1 == 0) ? 0 : (words[i+3] >> 10) & 0x3FF;
            adcValues[3][((ids[3] & 0x7)*2)+1] = ((ids[3]>>3)&0x1 == 0) ? 0 : (words[i+3] >> 20) & 0x3FF;
            adcValues[3][((ids[3] & 0x7)*2)  ] = ((ids[3]>>3)&0x1 == 0) ? 0 : ((words[i+2] & 0xFF) << 2) | ((words[i+3] >> 30) & 0x3);
            adcValues[2][((ids[2] & 0x7)*2)+1] = ((ids[2]>>3)&0x1 == 0) ? 0 : (words[i+2] >> 8) & 0x3FF;
            adcValues[2][((ids[2] & 0x7)*2)  ] = ((ids[2]>>3)&0x1 == 0) ? 0 : (words[i+2] >> 18) & 0x3FF;
            adcValues[1][((ids[1] & 0x7)*2)+1] = ((ids[1]>>3)&0x1 == 0) ? 0 : ((words[i+1] & 0x3F) << 4) | ((words[i+2] >> 28) & 0xF);
            adcValues[1][((ids[1] & 0x7)*2)  ] = ((ids[1]>>3)&0x1 == 0) ? 0 : (words[i+1] >> 6) & 0x3FF;
            adcValues[0][((ids[0] & 0x7)*2)+1] = ((ids[0]>>3)&0x1 == 0) ? 0 : (words[i+1] >> 16) & 0x3FF;
            adcValues[0][((ids[0] & 0x7)*2)  ] = ((ids[0]>>3)&0x1 == 0) ? 0 : ((words[i+0] & 0xF) << 6) | ((words[i+1] >> 26) & 0x3F);

            for (char j=0; j<5; ++j) {
              if (ids[j] == 0x8) writeValue[j] = true;
            }

            for (char j=0; j<5; ++j) {
              if (writeValue[j] & ids[j] == 0xF) {
                for (int k=0; k<16; ++k) {
                  const PadPos padPos = mapper.padPosRegion(
                      data.region, data.link,sampas[j],k+sampaChannelStart[j]);
//                  std::cout << "Row: " << (int)padPos.getRow() << " Pad: " << (int)padPos.getPad() << std::endl;

                  auto it = mData.find(padPos);
                  if (it != mData.end()) {
//                    std::cout << "padpos (Row: " << (int)padPos.getRow() << " Pad: " << (int)padPos.getPad() << ") already there, appending value." << std::endl;
                    it->second->push_back(adcValues[j][k]);
                  } else {
//                    std::cout << "padpos (Row: " << (int)padPos.getRow() << " Pad: " << (int)padPos.getPad() << ") not yet found." << std::endl;
                    mData.insert(std::pair<PadPos, std::shared_ptr<std::vector<uint16_t>>> (padPos,new std::vector<uint16_t>{adcValues[j][k]}));
                  }
//                  std::cout << adcValues[j][k] << " ";
                }
//                std::cout << std::endl;
              }
            }
//            std::cout << std::endl;
          }
          break;
        }
      case 3: // both, RAW GBT frames and decoded data
        {
          LOG(DEBUG) << "Data of readout mode 3 (using decoded data)" << FairLogger::endl;
          std::array<uint32_t,5> ids;
          std::array<bool,5> writeValue;
          writeValue.fill(false);
          std::array<std::array<uint16_t,16>,5> adcValues;
          std::array<char,5> sampas;
          std::array<char,5> sampaChannelStart;
          for (char i=0; i<5; ++i) {
            sampas[i] = (i == 4) ? 2 : (data.region%2) ? i/2+3 : i/2;
            sampaChannelStart[i] = (i == 4) ?   // 5th half SAMPA corresponds to  SAMPA2
              ((data.region%2) ? 16 : 0) :      // every even CRU receives channel 0-15 from SAMPA 2, the odd ones channel 16-31
              ((i%2) ? 16 : 0);                 // every even half SAMPA containes channel 0-15, the odd ones channel 16-31
          }

          for (int i=0; i<nWords; i=i+8) {
            ids[4] = (words[i+4] >> 4) & 0xF;
            ids[3] = (words[i+4] >> 8) & 0xF;
            ids[2] = (words[i+4] >> 12) & 0xF;
            ids[1] = (words[i+4] >> 16) & 0xF;
            ids[0] = (words[i+4] >> 20) & 0xF;

            adcValues[4][((ids[4] & 0x7)*2)+1] = ((ids[4]>>3)&0x1 == 0) ? 0 : words[i+7] & 0x3FF;
            adcValues[4][((ids[4] & 0x7)*2)  ] = ((ids[4]>>3)&0x1 == 0) ? 0 : (words[i+7] >> 10) & 0x3FF;
            adcValues[3][((ids[3] & 0x7)*2)+1] = ((ids[3]>>3)&0x1 == 0) ? 0 : (words[i+7] >> 20) & 0x3FF;
            adcValues[3][((ids[3] & 0x7)*2)  ] = ((ids[3]>>3)&0x1 == 0) ? 0 : ((words[i+6] & 0xFF) << 2) | ((words[i+7] >> 30) & 0x3);
            adcValues[2][((ids[2] & 0x7)*2)+1] = ((ids[2]>>3)&0x1 == 0) ? 0 : (words[i+6] >> 8) & 0x3FF;
            adcValues[2][((ids[2] & 0x7)*2)  ] = ((ids[2]>>3)&0x1 == 0) ? 0 : (words[i+6] >> 18) & 0x3FF;
            adcValues[1][((ids[1] & 0x7)*2)+1] = ((ids[1]>>3)&0x1 == 0) ? 0 : ((words[i+5] & 0x3F) << 4) | ((words[i+6] >> 28) & 0xF);
            adcValues[1][((ids[1] & 0x7)*2)  ] = ((ids[1]>>3)&0x1 == 0) ? 0 : (words[i+5] >> 6) & 0x3FF;
            adcValues[0][((ids[0] & 0x7)*2)+1] = ((ids[0]>>3)&0x1 == 0) ? 0 : (words[i+5] >> 16) & 0x3FF;
            adcValues[0][((ids[0] & 0x7)*2)  ] = ((ids[0]>>3)&0x1 == 0) ? 0 : ((words[i+4] & 0xF) << 6) | ((words[i+5] >> 26) & 0x3F);

            for (char j=0; j<5; ++j) {
              if (ids[j] == 0x8) writeValue[j] = true;
            }

            for (char j=0; j<5; ++j) {
              if (writeValue[j] & ids[j] == 0xF) {
                for (int k=0; k<16; ++k) {
                  const PadPos padPos = mapper.padPosRegion(
                      data.region, data.link,sampas[j],k+sampaChannelStart[j]);
//                  std::cout << "Row: " << (int)padPos.getRow() << " Pad: " << (int)padPos.getPad() << std::endl;

                  auto it = mData.find(padPos);
                  if (it != mData.end()) {
//                    std::cout << "padpos (Row: " << (int)padPos.getRow() << " Pad: " << (int)padPos.getPad() << ") already there, appending value." << std::endl;
                    it->second->push_back(adcValues[j][k]);
                  } else {
//                    std::cout << "padpos (Row: " << (int)padPos.getRow() << " Pad: " << (int)padPos.getPad() << ") not yet found." << std::endl;
                    mData.insert(std::pair<PadPos, std::shared_ptr<std::vector<uint16_t>>> (padPos,new std::vector<uint16_t>{adcValues[j][k]}));
                  }
//                  std::cout << adcValues[j][k] << " ";
                }
//                std::cout << std::endl;
              }
            }
//            std::cout << std::endl;
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
