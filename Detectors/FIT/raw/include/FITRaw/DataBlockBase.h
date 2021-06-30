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
//
//file DataBlockBase.h base class for RAW data format data blocks
//
// Artur.Furs
// afurs@cern.ch
//DataBlockWrapper - wrapper for raw data structures
//There should be three static fields in raw data structs, which defines its "signature":
//  payloadSize - actual payload size per one raw data struct element (can be larger than GBTword size!)
//  payloadPerGBTword - maximum payload per one GBT word
//  MaxNelements - maximum number of elements per data block(for header it should be equal to 1)
//  MinNelements - minimum number of elements per data block(for header it should be equal to 1)

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

#ifndef ALICEO2_FIT_DATABLOCKBASE_H_
#define ALICEO2_FIT_DATABLOCKBASE_H_
#include <iostream>
#include <vector>
#include <Rtypes.h>
#include "CommonDataFormat/InteractionRecord.h"
#include "Headers/RAWDataHeader.h"
#include <gsl/span>
#include <boost/mpl/inherit.hpp>
#include <boost/mpl/vector.hpp>
#include <Framework/Logger.h>
#include <vector>
#include <tuple>
#include <array>
#include <iostream>
#include <cassert>
#include <type_traits>
namespace o2
{
namespace fit
{

using namespace std;

static constexpr size_t SIZE_WORD = 16;                                                       // should be changed to gloabal variable
static constexpr size_t SIZE_MAX_PAGE = 8192;                                                 // should be changed to gloabal variable
static constexpr size_t SIZE_MAX_PAYLOAD = SIZE_MAX_PAGE - sizeof(o2::header::RAWDataHeader); // should be changed to gloabal variable

template <typename T>
struct DataBlockHelper {
  template <typename, typename = void>
  struct CheckTypeMaxNelem : std::false_type {
  };
  template <typename U>
  struct CheckTypeMaxNelem<U, std::enable_if_t<std::is_same<decltype(U::MaxNelements), const std::size_t>::value>> : std::true_type {
  };

  template <typename, typename = void>
  struct CheckTypeMinNelem : std::false_type {
  };
  template <typename U>
  struct CheckTypeMinNelem<U, std::enable_if_t<std::is_same<decltype(U::MinNelements), const std::size_t>::value>> : std::true_type {
  };

  template <typename, typename = void>
  struct CheckTypePayloadSize : std::false_type {
  };
  template <typename U>
  struct CheckTypePayloadSize<U, std::enable_if_t<std::is_same<decltype(U::PayloadSize), const std::size_t>::value>> : std::true_type {
  };

  template <typename, typename = void>
  struct CheckTypePayloadPerGBTword : std::false_type {
  };
  template <typename U>
  struct CheckTypePayloadPerGBTword<U, std::enable_if_t<std::is_same<decltype(U::PayloadPerGBTword), const std::size_t>::value>> : std::true_type {
  };

  template <typename, typename = void>
  struct CheckMaxElemSize : std::false_type {
  };
  template <typename U>
  struct CheckMaxElemSize<U, std::enable_if_t<((T::MaxNelements * T::PayloadSize <= SIZE_MAX_PAYLOAD) && (T::MaxNelements * T::PayloadPerGBTword <= SIZE_MAX_PAYLOAD))>> : std::true_type {
  };

  template <typename, typename = void>
  struct CheckNelemRange : std::false_type {
  };
  template <typename U>
  struct CheckNelemRange<U, std::enable_if_t<(T::MaxNelements >= T::MinNelements)>> : std::true_type {
  };

  static constexpr bool check()
  {
    static_assert(CheckTypeMaxNelem<T>::value, "Error! MaxNelements type should be \"static constexpr std::size_t\"!");
    static_assert(CheckTypeMinNelem<T>::value, "Error! MinNelements type should be \"static constexpr std::size_t\"!");
    static_assert(CheckTypePayloadSize<T>::value, "Error! PayloadSize type should be \"static constexpr std::size_t\"!");
    static_assert(CheckTypePayloadPerGBTword<T>::value, "Error! PayloadPerGBTword type should be \"static constexpr std::size_t\"!");
    static_assert(CheckMaxElemSize<T>::value, "Error! Check maximum number of elements, they are larger than payload size!");
    static_assert(CheckNelemRange<T>::value, "Error! Check range for number of elements, max should be bigger or equal to min!");
    return CheckTypeMaxNelem<T>::value && CheckTypeMinNelem<T>::value && CheckTypePayloadSize<T>::value && CheckTypePayloadPerGBTword<T>::value && CheckMaxElemSize<T>::value && CheckNelemRange<T>::value;
  }
};

template <typename T, typename = typename std::enable_if_t<DataBlockHelper<T>::check()>>
struct DataBlockWrapper {
  DataBlockWrapper() = default;
  DataBlockWrapper(const DataBlockWrapper&) = default;
  typedef T Data_t;
  void serialize(std::vector<char>& vecBytes, size_t nWords, size_t& destPos) const
  {
    const uint8_t* srcAddress = (uint8_t*)mData;
    if (nWords == 0 || nWords > MaxNwords || vecBytes.size() - destPos < nWords * SIZE_WORD) {
      LOG(INFO) << "Warning! Incorrect serialisation procedure!";
      return;
    }
    gsl::span<char> serializedBytes(vecBytes);
    size_t countBytes = 0;
    int nSteps = std::get<kNSTEPS>(sReadingLookupTable[nWords]);
    for (int iStep = 0; iStep < nSteps; iStep++) {
      memcpy(serializedBytes.data() + std::get<kSRCBYTEPOS>(sByteLookupTable[iStep]) + destPos, srcAddress + std::get<kDESTBYTEPOS>(sByteLookupTable[iStep]), std::get<kNBYTES>(sByteLookupTable[iStep]));
      countBytes += std::get<kSRCBYTEPOS>(sByteLookupTable[iStep]);
    }
    destPos += nWords * SIZE_WORD;
  }

  void deserialize(const gsl::span<const uint8_t> inputBytes, size_t nWords, size_t& srcPos)
  {
    mNelements = 0;
    mNwords = 0;
    if (nWords < MinNwords || nWords > MaxNwords || inputBytes.size() - srcPos < nWords * SIZE_WORD) {
      //in case of bad fields responsible for deserialization logic, byte position will be pushed to the end of binary sequence
      srcPos = inputBytes.size();
      mIsIncorrect = true;
      return;
    }
    uint8_t* destAddress = (uint8_t*)mData;
    size_t countBytes = 0;
    int nSteps = std::get<kNSTEPS>(sReadingLookupTable[nWords]);
    mNwords = nWords;
    mNelements = std::get<kNELEMENTS>(sReadingLookupTable[nWords]);
    for (int iStep = 0; iStep < nSteps; iStep++) {
      memcpy(destAddress + std::get<kDESTBYTEPOS>(sByteLookupTable[iStep]), inputBytes.data() + std::get<kSRCBYTEPOS>(sByteLookupTable[iStep]) + srcPos, std::get<kNBYTES>(sByteLookupTable[iStep]));
      countBytes += std::get<kSRCBYTEPOS>(sByteLookupTable[iStep]);
    }
    srcPos += mNwords * SIZE_WORD;
  }

  static constexpr int MaxNwords = Data_t::PayloadSize * Data_t::MaxNelements / Data_t::PayloadPerGBTword + (Data_t::PayloadSize * Data_t::MaxNelements % Data_t::PayloadPerGBTword > 0); //calculating max GBT words per block
  static constexpr int MaxNbytes = SIZE_WORD * MaxNwords;

  static constexpr int MinNwords = Data_t::PayloadSize * Data_t::MinNelements / Data_t::PayloadPerGBTword + (Data_t::PayloadSize * Data_t::MinNelements % Data_t::PayloadPerGBTword > 0); //calculating min GBT words per block
  static constexpr int MinNbytes = SIZE_WORD * MinNwords;

  //get number of byte reading steps
  static constexpr size_t getNsteps()
  {
    int count = 0;
    size_t payloadFull = Data_t::MaxNelements * Data_t::PayloadSize;
    size_t payloadInWord = Data_t::PayloadPerGBTword;
    size_t payloadPerElem = Data_t::PayloadSize;
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
      if (payloadInWord == 0) {
        payloadInWord = Data_t::PayloadPerGBTword;
      }
      if (payloadPerElem == 0) {
        payloadPerElem = Data_t::PayloadSize;
      }
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
    size_t payloadFull = Data_t::MaxNelements * Data_t::PayloadSize;

    size_t bytesInWord = SIZE_WORD;
    size_t payloadInWord = Data_t::PayloadPerGBTword;

    size_t payloadPerElem = Data_t::PayloadSize;

    uint64_t indexElem = 0;
    uint64_t indexLastElem = Data_t::MaxNelements - 1;

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

      if (payloadInWord == 0) {
        payloadInWord = Data_t::PayloadPerGBTword;
      }
      if (payloadPerElem == 0) {
        payloadPerElem = Data_t::PayloadSize;
        countElement++;
        destBytePosPerElem = countElement * sizeof(T);
      }
      if (bytesInWord == 0) {
        bytesInWord = SIZE_WORD;
        countWord++;
      }
    }
    return seqBytes;
  }
  static constexpr std::array<std::tuple<size_t, size_t, size_t, int, int>, getNsteps()> sByteLookupTable = GetByteLookupTable();

  //enumerator for tuple access:
  //[Index] is word index position, i.e. "Index" number of words will be deserialized
  //kNELEMENTS - number of T elements will be fully deserialized in "Index+1" words
  //kNSTEPS - number of steps for reading "Index" words
  //kISPARTED - if one T-element is parted at current word,i.e. current word contains partially deserialized T element at the end of the word
  enum AccessReadingLUT { kNELEMENTS,
                          kNSTEPS,
                          kISPARTED };
  static constexpr std::array<std::tuple<unsigned int, unsigned int, bool>, MaxNwords + 1> GetReadingLookupTable()
  {
    std::array<std::tuple<unsigned int, unsigned int, bool>, MaxNwords + 1> readingScheme{};
    size_t payloadPerElem = Data_t::PayloadSize;
    std::get<kNSTEPS>(readingScheme[0]) = 0;
    std::get<kNELEMENTS>(readingScheme[0]) = 0;
    std::get<kISPARTED>(readingScheme[0]) = false;
    int countWord = 1;
    for (int iStep = 0; iStep < getNsteps(); iStep++) {
      if (countWord - 1 < std::get<kWORDINDEX>((GetByteLookupTable())[iStep])) { //new word
        std::get<kNSTEPS>(readingScheme[countWord]) = iStep;
        std::get<kNELEMENTS>(readingScheme[countWord]) = std::get<kELEMENTINDEX>((GetByteLookupTable())[iStep]);
        if (payloadPerElem > 0) {
          std::get<kISPARTED>(readingScheme[countWord]) = true;
        } else {
          std::get<kISPARTED>(readingScheme[countWord]) = false;
        }
        countWord++;
      }
      if (payloadPerElem == 0) {
        payloadPerElem = Data_t::PayloadSize;
      }
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
  static constexpr std::array<std::tuple<unsigned int, unsigned int, bool>, MaxNwords + 1> sReadingLookupTable = GetReadingLookupTable();
  //
  //Printing LookupTables
  static void printLUT()
  {
    LOG(INFO) << "-------------------------------------------";
    LOG(INFO) << "kNELEMENTS|kNSTEPS|kISPARTED";
    for (int i = 0; i < MaxNwords + 1; i++) {
      LOG(INFO) << std::get<kNELEMENTS>(sReadingLookupTable[i]) << "|"
                << std::get<kNSTEPS>(sReadingLookupTable[i]) << "|"
                << std::get<kISPARTED>(sReadingLookupTable[i]);
    }
    LOG(INFO) << "-------------------------------------------";
    LOG(INFO) << "kELEMENTINDEX|kWORDINDEX|kNBYTES|kSRCBYTEPOS|kDESTBYTEPOS";
    for (int i = 0; i < getNsteps(); i++) {
      LOG(INFO) << std::get<kELEMENTINDEX>(sByteLookupTable[i]) << "|"
                << std::get<kWORDINDEX>(sByteLookupTable[i]) << "|"
                << std::get<kNBYTES>(sByteLookupTable[i]) << "|"
                << std::get<kSRCBYTEPOS>(sByteLookupTable[i]) << "|"
                << std::get<kDESTBYTEPOS>(sByteLookupTable[i]);
    }
  }
  void print() const
  {
    assert(mNelements <= Data_t::MaxNelements);
    for (int i = 0; i < mNelements; i++) {
      LOG(INFO) << "Printing element number: " << i;
      mData[i].print();
    }
  }
  Data_t mData[Data_t::MaxNelements];
  unsigned int mNelements; //number of deserialized elements;
  unsigned int mNwords;    //number of deserialized GBT words; //can be excluded
  bool mIsIncorrect;
};

//CRTP(static polymorphism) + Composition over multiple inheritance(Header + multiple data structures)
template <template <typename...> class DataBlock, class Header, class... DataStructures>
class DataBlockBase : public boost::mpl::inherit<DataBlockWrapper<Header>, DataBlockWrapper<DataStructures>...>::type
{
  typedef boost::mpl::vector<DataStructures...> DataBlockTypes;
  typedef DataBlock<Header, DataStructures...> DataBlock_t;
  typedef DataBlockBase<DataBlock, Header, DataStructures...> TemplateHeader;
  typedef typename boost::mpl::inherit<DataBlockWrapper<Header>, DataBlockWrapper<DataStructures>...>::type DataBlockDerivedBase;

 public:
  DataBlockBase() = default;
  DataBlockBase(const DataBlockBase&) = default;

  static void printLUT()
  {
    DataBlockWrapper<Header>::printLUT();
    (static_cast<void>(DataBlockWrapper<DataStructures>::printLUT()), ...);
  }

  void print() const
  {
    LOG(INFO) << "HEADER";
    DataBlockWrapper<Header>::print();
    LOG(INFO) << "DATA";
    (static_cast<void>(DataBlockWrapper<DataStructures>::print()), ...);
  }

  InteractionRecord getInteractionRecord() const
  {
    return DataBlockWrapper<Header>::mData[0].getIntRec();
  }

  void setInteractionRecord(const InteractionRecord& intRec)
  {
    DataBlockWrapper<Header>::mData[0].setIntRec(intRec);
  }
  //
  //use this for block decoding
  void decodeBlock(gsl::span<const uint8_t> payload, size_t srcPos)
  {
    mSize = 0;
    size_t bytePos = srcPos;
    static_cast<DataBlock_t*>(this)->deserialize(payload, bytePos);
    mSize = bytePos - srcPos;
    //checking sanity and updating
    update();
  }

  bool isCorrect() const { return mIsCorrect; }

  void update()
  {
    mIsCorrect = true;
    checkDeserialization(mIsCorrect, DataBlockWrapper<Header>::mIsIncorrect);                // checking deserialization status for header
    (checkDeserialization(mIsCorrect, DataBlockWrapper<DataStructures>::mIsIncorrect), ...); // checking deserialization status for sub-block
    static_cast<DataBlock_t*>(this)->sanityCheck(mIsCorrect);
  }

  size_t mSize; //deserialized size
  bool mIsCorrect;

 protected:
  //check if there are sub blocks with zero number of elements
  void isNonZeroBlockSizes(bool& flag, unsigned int nElements) { flag &= (bool)nElements; }
  void checkDeserialization(bool& flag, bool isIncorrect) { flag &= !(isIncorrect); }
};

} // namespace fit
} // namespace o2
#endif
