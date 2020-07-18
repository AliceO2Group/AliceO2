// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.
//
//file DataBlockRaw.h class  for RAW data format data blocks
//
// Artur.Furs
// afurs@cern.ch
//DataBlockWrapper - wrapper for raw data structures
//There should be three static fields in raw data structs, which defines its "signature":
//  payloadSize - actual payload size per one raw data struct element (can be larger than GBTword size!)
//  payloadPerGBTword - maximum payload per one GBT word
//  MaxNelements - maximum number of elements per data block(for header it should be equal to 1)
//
//Also it requares several methods:
//  print() - for printing raw data structs
//  getIntRec() - for InteractionRecord extraction, should be in Header struct

//DataBlockBase - base class for making composition of raw data structures, uses CRTP(static polyporphism)
//usage:
//  class DataBlockOfYourModule: public DataBlockBase< DataBlockOfYourModule, RawHeaderStruct, RawDataStruct ...>
//  define "deserialization" method with deserialization logic for current DataBlock
//  define "sanityCheck" method for checking if the DataBlock is correct
//Warning! Classes should be simple, without refs and pointers!
//TODO:
//  need to use references on the DataBlock fileds, with fast access
//  traites for classes and structs
//

#ifndef ALICEO2_FIT_DATABLOCKRAW_H_
#define ALICEO2_FIT_DATABLOCKRAW_H_
#include <iostream>
#include <vector>
#include <Rtypes.h>
#include "CommonDataFormat/InteractionRecord.h"
#include <DataFormatsFT0/RawEventData.h>
#include <gsl/span>
#include <boost/mpl/inherit.hpp>
#include <boost/mpl/vector.hpp>

#include <vector>
#include <tuple>
#include <array>
#include <iostream>
#include <cassert>

namespace o2
{
namespace ft0
{
using RawHeaderPM = o2::ft0::EventHeader;
using RawDataPM = o2::ft0::EventData;
using RawHeaderTCM = o2::ft0::EventHeader;
using RawDataTCM = o2::ft0::TCMdata;
using RawHeaderTCMext = o2::ft0::EventHeader;
using RawDataTCMext = o2::ft0::TCMdataExtended;

using namespace std;

template <typename T>
struct DataBlockWrapper {
  DataBlockWrapper() = default;
  DataBlockWrapper(const DataBlockWrapper& other) = default;
  DataBlockWrapper(DataBlockWrapper&& other) = default;
  // virtual ~DataBlockWrapper() = default;

  std::vector<uint8_t> serialize(int nWords)
  {
    std::vector<uint8_t> vecBytes(sizeWord * nWords);
    uint8_t* srcAddress = (uint8_t*)mData;
    if (nWords == 0)
      return std::move(vecBytes);
    gsl::span<uint8_t> serializedBytes(vecBytes);
    size_t countBytes = 0;
    int nSteps = std::get<kNSTEPS>(sReadingLookupTable[nWords - 1]);
    for (int iStep = 0; iStep < nSteps; iStep++) {
      memcpy(serializedBytes.data() + std::get<kSRCBYTEPOS>(sByteLookupTable[iStep]), srcAddress + std::get<kDESTBYTEPOS>(sByteLookupTable[iStep]), std::get<kNBYTES>(sByteLookupTable[iStep]));
      countBytes += std::get<kSRCBYTEPOS>(sByteLookupTable[iStep]);
    }
    return std::move(vecBytes);
  }

  void deserialize(const gsl::span<const uint8_t> inputBytes, int nWords, size_t& srcPos)
  {
    mNelements = 0;
    mNwords = 0;
    if (nWords == 0 || nWords > MaxNwords || inputBytes.size() - srcPos < nWords * sizeWord) {
      //in case of bad fields responsible for deserialization logic, byte position will be pushed to the end of binary sequence
      srcPos = inputBytes.size();
      return;
    }
    uint8_t* destAddress = (uint8_t*)mData;
    size_t countBytes = 0;
    int nSteps = std::get<kNSTEPS>(sReadingLookupTable[nWords - 1]);
    mNwords = nWords;
    mNelements = std::get<kNELEMENTS>(sReadingLookupTable[nWords - 1]);
    for (int iStep = 0; iStep < nSteps; iStep++) {
      memcpy(destAddress + std::get<kDESTBYTEPOS>(sByteLookupTable[iStep]), inputBytes.data() + std::get<kSRCBYTEPOS>(sByteLookupTable[iStep]) + srcPos, std::get<kNBYTES>(sByteLookupTable[iStep]));
      countBytes += std::get<kSRCBYTEPOS>(sByteLookupTable[iStep]);
    }
    srcPos += mNwords * sizeWord;
  }

  static constexpr int MaxNwords = T::PayloadSize * T::MaxNelements / T::PayloadPerGBTword + (T::PayloadSize * T::MaxNelements % T::PayloadPerGBTword > 0); //calculating max GBT words per block
  static constexpr int MaxNbytes = sizeWord * MaxNwords;

  //get number of byte reading steps
  static constexpr size_t getNsteps()
  {
    int count = 0;
    size_t payloadFull = T::MaxNelements * T::PayloadSize;
    size_t payloadInWord = T::PayloadPerGBTword;
    size_t payloadPerElem = T::PayloadSize;
    while (payloadFull > 0) {
      if (payloadPerElem < payloadInWord) {
        count++;
        payloadFull -= payloadPerElem;
        payloadInWord -= payloadPerElem;
        payloadPerElem = 0;
      } else {
        count++;
        payloadFull -= payloadInWord;
        payloadPerElem -= payloadInWord;
        payloadInWord = 0;
      }
      if (payloadInWord == 0)
        payloadInWord = T::PayloadPerGBTword;
      if (payloadPerElem == 0)
        payloadPerElem = T::PayloadSize;
    }
    return count;
  }
  //enumerator for tuple access:
  //[Index] is index of step to read bytes
  //kNBYTES - number of bytes to read from source and to write into destination
  //kSRCBYTEPOS - Byte position in the source(binary raw data sequence)
  //kDESTBYTEPOS - Byte position to write at destionation(memory allocation of T-element array)
  //kELEMENTINDEX - element index at current step
  //kWORDINDEX - word index at current step
  //
  enum AccessByteLUT { kNBYTES,
                       kSRCBYTEPOS,
                       kDESTBYTEPOS,
                       kELEMENTINDEX,
                       kWORDINDEX };
  static constexpr std::array<std::tuple<size_t, size_t, size_t, int, int>, getNsteps()> GetByteLookupTable()
  {
    std::array<std::tuple<size_t, size_t, size_t, int, int>, getNsteps()> seqBytes{};
    int count = 0;
    int countElement = 0;
    int countWord = 0;
    size_t destBytePosPerElem = 0;
    size_t srcBytePos = 0;
    size_t payloadFull = T::MaxNelements * T::PayloadSize;

    size_t bytesInWord = sizeWord;
    size_t payloadInWord = T::PayloadPerGBTword;

    size_t payloadPerElem = T::PayloadSize;

    uint64_t indexElem = 0;
    uint64_t indexLastElem = T::MaxNelements - 1;

    while (payloadFull > 0) {
      if (payloadPerElem < payloadInWord) { //new element
        std::get<kNBYTES>(seqBytes[count]) = payloadPerElem;
        std::get<kSRCBYTEPOS>(seqBytes[count]) = srcBytePos;
        std::get<kDESTBYTEPOS>(seqBytes[count]) = destBytePosPerElem;
        std::get<kELEMENTINDEX>(seqBytes[count]) = countElement;
        std::get<kWORDINDEX>(seqBytes[count]) = countWord;
        srcBytePos += payloadPerElem;
        count++;
        payloadFull -= payloadPerElem;
        payloadInWord -= payloadPerElem;
        bytesInWord -= payloadPerElem;
        payloadPerElem = 0;

      } else {
        std::get<kNBYTES>(seqBytes[count]) = payloadInWord;
        std::get<kSRCBYTEPOS>(seqBytes[count]) = srcBytePos;
        std::get<kDESTBYTEPOS>(seqBytes[count]) = destBytePosPerElem;
        std::get<kELEMENTINDEX>(seqBytes[count]) = countElement;
        std::get<kWORDINDEX>(seqBytes[count]) = countWord;
        srcBytePos += bytesInWord;
        count++;
        destBytePosPerElem += payloadInWord;

        payloadFull -= payloadInWord;
        payloadPerElem -= payloadInWord;
        payloadInWord = 0;
        bytesInWord = 0;
      }

      if (payloadInWord == 0)
        payloadInWord = T::PayloadPerGBTword;
      if (payloadPerElem == 0) {
        payloadPerElem = T::PayloadSize;
        countElement++;
        destBytePosPerElem = countElement * sizeof(T);
      }
      if (bytesInWord == 0) {
        bytesInWord = sizeWord;
        countWord++;
      }
    }
    return seqBytes;
  }
  static constexpr std::array<std::tuple<size_t, size_t, size_t, int, int>, getNsteps()> sByteLookupTable = GetByteLookupTable();

  //enumerator for tuple access:
  //[Index] is word index position, i.e. "Index+1" number of words will be deserialized
  //kNELEMENTS - number of T elements will be fully deserialized in "Index+1" words
  //kNSTEPS - number of steps for reading "Index+1" words
  //kISPARTED - if one T-element is parted at current word,i.e. current word contains partially deserialized T element at the end of the word
  enum AccessReadingLUT { kNELEMENTS,
                          kNSTEPS,
                          kISPARTED };
  static constexpr std::array<std::tuple<unsigned int, unsigned int, bool>, MaxNwords> GetReadingLookupTable()
  {
    std::array<std::tuple<unsigned int, unsigned int, bool>, MaxNwords> readingScheme{};
    size_t payloadPerElem = T::PayloadSize;
    int countWord = 0;
    for (int iStep = 0; iStep < getNsteps(); iStep++) {
      if (countWord < std::get<kWORDINDEX>((GetByteLookupTable())[iStep])) { //new word
        std::get<kNSTEPS>(readingScheme[countWord]) = iStep;
        std::get<kNELEMENTS>(readingScheme[countWord]) = std::get<kELEMENTINDEX>((GetByteLookupTable())[iStep]);
        if (payloadPerElem > 0) {
          std::get<kISPARTED>(readingScheme[countWord]) = true;
        } else {
          std::get<kISPARTED>(readingScheme[countWord]) = false;
        }
        countWord++;
      }
      if (payloadPerElem == 0)
        payloadPerElem = T::PayloadSize;
      payloadPerElem -= std::get<kNBYTES>((GetByteLookupTable())[iStep]);
    }
    //Last step checking
    std::get<kNSTEPS>(readingScheme[countWord]) = getNsteps();
    if (payloadPerElem > 0) {
      std::get<kISPARTED>(readingScheme[countWord]) = true;
      std::get<kNELEMENTS>(readingScheme[countWord]) = std::get<kELEMENTINDEX>((GetByteLookupTable())[getNsteps() - 1]);
    } else {
      std::get<kISPARTED>(readingScheme[countWord]) = false;
      std::get<kNELEMENTS>(readingScheme[countWord]) = std::get<kELEMENTINDEX>((GetByteLookupTable())[getNsteps() - 1]) + 1;
    }
    return readingScheme;
  }
  static constexpr std::array<std::tuple<unsigned int, unsigned int, bool>, MaxNwords> sReadingLookupTable = GetReadingLookupTable();
  //
  //Printing LookupTables
  static void printLUT()
  {
    cout << endl
         << "-------------------------------------------" << endl;
    std::cout << "kNELEMENTS|kNSTEPS|kISPARTED" << std::endl;
    for (int i = 0; i < MaxNwords; i++) {
      std::cout << std::endl
                << std::get<kNELEMENTS>(sReadingLookupTable[i]) << "|"
                << std::get<kNSTEPS>(sReadingLookupTable[i]) << "|"
                << std::get<kISPARTED>(sReadingLookupTable[i]) << endl;
    }
    cout << endl
         << "-------------------------------------------" << endl;
    std::cout << "kELEMENTINDEX|kWORDINDEX|kNBYTES|kSRCBYTEPOS|kDESTBYTEPOS" << std::endl;
    for (int i = 0; i < getNsteps(); i++) {
      cout << endl
           << std::get<kELEMENTINDEX>(sByteLookupTable[i]) << "|"
           << std::get<kWORDINDEX>(sByteLookupTable[i]) << "|"
           << std::get<kNBYTES>(sByteLookupTable[i]) << "|"
           << std::get<kSRCBYTEPOS>(sByteLookupTable[i]) << "|"
           << std::get<kDESTBYTEPOS>(sByteLookupTable[i]) << endl;
    }
  }
  void print()
  {
    assert(mNelements <= T::MaxNelements);
    for (int i = 0; i < mNelements; i++) {
      std::cout << "\nPringting element number: " << i << std::endl;
      mData[i].print();
    }
  }
  T mData[T::MaxNelements];
  unsigned int mNelements; //number of deserialized elements;
  unsigned int mNwords;    //number of deserialized GBT words;
};

//CRTP(static polymorphism) + Composition over multiple inheritance(Header + multiple data structures)
template <class DataBlock, class Header, class... DataStructures>
class DataBlockBase : public boost::mpl::inherit<DataBlockWrapper<Header>, DataBlockWrapper<DataStructures>...>::type
{
  typedef boost::mpl::vector<DataStructures...> DataBlockTypes;
  typedef DataBlockBase<DataBlock, Header, DataStructures...> TemplateHeader;
  typedef typename boost::mpl::inherit<DataBlockWrapper<Header>, DataBlockWrapper<DataStructures>...>::type DataBlockDerivedBase;

 public:
  DataBlockBase() = default;
  DataBlockBase(const DataBlockBase& other) = default;

  virtual ~DataBlockBase() = default;

  static void printLUT()
  {
    DataBlockWrapper<Header>::printLUT();
    (static_cast<void>(DataBlockWrapper<DataStructures>::printLUT()), ...);
  }

  void print()
  {
    LOG(INFO) << "HEADER";
    DataBlockWrapper<Header>::print();
    LOG(INFO) << "DATA";
    (static_cast<void>(DataBlockWrapper<DataStructures>::print()), ...);
  }

  InteractionRecord getInteractionRecord()
  {
    return DataBlockWrapper<Header>::mData[0].getIntRec();
  }
  //
  //use this for block decoding
  void decodeBlock(gsl::span<const uint8_t> payload, size_t srcPos)
  {
    mSize = 0;
    size_t bytePos = srcPos;
    static_cast<DataBlock*>(this)->deserialize(payload, bytePos);
    mSize = bytePos - srcPos;
    //checking sanity and updating
    update();
  }

  bool isCorrect() { return mIsCorrect; }

  void update()
  {
    mIsCorrect = true;
    isNonZeroBlockSizes(mIsCorrect, DataBlockWrapper<Header>::mNelements);                // checking sub-block(header) size
    (isNonZeroBlockSizes(mIsCorrect, DataBlockWrapper<DataStructures>::mNelements), ...); // checking sub-block sizes
    static_cast<DataBlock*>(this)->sanityCheck(mIsCorrect);
  }

  size_t mSize; //deserialized size
  bool mIsCorrect;

 protected:
  //check if there are sub blocks with zero number of elements
  void isNonZeroBlockSizes(bool& flag, unsigned int nElements) { flag &= (bool)nElements; }
  //void updateSize(size_t blockSize) { mSize += blockSize; }
};

//FT0 DATA BLOCK DEFINITIONS

//standard data block from PM
class DataBlockPM : public DataBlockBase<DataBlockPM, RawHeaderPM, RawDataPM>
{
 public:
  DataBlockPM() = default;
  DataBlockPM(const DataBlockPM& other) = default;
  virtual ~DataBlockPM() = default;
  void deserialize(gsl::span<const uint8_t> srcBytes, size_t& srcByteShift)
  {
    DataBlockWrapper<RawHeaderPM>::deserialize(srcBytes, DataBlockWrapper<RawHeaderPM>::MaxNwords, srcByteShift);
    DataBlockWrapper<RawDataPM>::deserialize(srcBytes, DataBlockWrapper<RawHeaderPM>::mData[0].nGBTWords, srcByteShift);
  }
  //Custom sanity checking for current deserialized block
  // put here code for raw data checking
  void sanityCheck(bool& flag)
  {
    if (DataBlockWrapper<RawDataPM>::mNelements % 2 == 0 && DataBlockWrapper<RawDataPM>::mData[DataBlockWrapper<RawDataPM>::mNelements - 1].channelID == 0)
      DataBlockWrapper<RawDataPM>::mNelements--; //in case of half GBT-word filling
    //TODO
  }
  //
  ClassDefNV(DataBlockPM, 1);
};
//standard data block from TCM
class DataBlockTCM : public DataBlockBase<DataBlockTCM, RawHeaderTCM, RawDataTCM>
{
 public:
  DataBlockTCM() = default;
  DataBlockTCM(const DataBlockTCM& other) = default;
  virtual ~DataBlockTCM() = default;
  void deserialize(gsl::span<const uint8_t> srcBytes, size_t& srcByteShift)
  {
    DataBlockWrapper<RawHeaderTCM>::deserialize(srcBytes, DataBlockWrapper<RawHeaderTCM>::MaxNwords, srcByteShift);
    DataBlockWrapper<RawDataTCM>::deserialize(srcBytes, DataBlockWrapper<RawHeaderTCM>::mData[0].nGBTWords, srcByteShift);
  }
  //Custom sanity checking for current deserialized block
  // put here code for raw data checking
  void sanityCheck(bool& flag)
  {

  } //TODO
  //
  ClassDefNV(DataBlockTCM, 1);
};

//extended TCM mode, 1 TCMdata + 8 TCMdataExtendedstructs
class DataBlockTCMext : public DataBlockBase<DataBlockTCMext, RawHeaderTCMext, RawDataTCM, RawDataTCMext>
{
 public:
  DataBlockTCMext() = default;
  DataBlockTCMext(const DataBlockTCMext& other) = default;
  virtual ~DataBlockTCMext() = default;
  void deserialize(gsl::span<const uint8_t> srcBytes, size_t& srcByteShift)
  {
    DataBlockWrapper<RawHeaderTCMext>::deserialize(srcBytes, DataBlockWrapper<RawHeaderTCMext>::MaxNwords, srcByteShift);
    DataBlockWrapper<RawDataTCM>::deserialize(srcBytes, DataBlockWrapper<RawDataTCM>::MaxNwords, srcByteShift);
    DataBlockWrapper<RawDataTCMext>::deserialize(srcBytes, DataBlockWrapper<RawHeaderTCMext>::mData[0].nGBTWords - DataBlockWrapper<RawDataTCM>::MaxNwords, srcByteShift);
  }

  //Custom sanity checking for current deserialized block
  // put here code for raw data checking
  void sanityCheck(bool& flag)
  {

  } //TODO
  //
  ClassDefNV(DataBlockTCMext, 1);
};

} // namespace ft0
} // namespace o2
#endif
