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

/// \file SACDecoder.h
/// \brief Decoding of integrated analogue currents
/// @author Jens Wiechula, Jens.Wiechula@ikf.uni-frankfurt.de

#ifndef ALICEO2_SACDECODER_H
#define ALICEO2_SACDECODER_H

#include <algorithm>
#include <array>
#include <deque>
#include <unordered_map>
#include <vector>
#include <bitset>
#include <gsl/span>

#include "CommonUtils/TreeStreamRedirector.h"
#include "CommonConstants/LHCConstants.h"

#include "DataFormatsTPC/Defs.h"
#include "DataFormatsTPC/SAC.h"

using o2::constants::lhc::LHCBunchSpacingMUS;

namespace o2::tpc::sac
{

constexpr float ADCtoNanoAmp = 125000.f / 8388608.f;     ///< 125000 nA / std::pow(2,23) bits
constexpr uint32_t ChannelsPerFE = 8;                    ///< Channels per front-end card. One channel is one stack
constexpr size_t Instances = 2;                          ///< Number of instances to process
constexpr size_t NumberFEs = FEsPerInstance * Instances; ///< Total number of frontends to process

/// Decoded data of one FE
struct DecodedDataFE {
  uint32_t timeStamp{};
  std::array<int32_t, 8> currents{};

  void resetCurrent()
  {
    std::fill(currents.begin(), currents.end(), 0);
  }

  void reset()
  {
    timeStamp = 0;
    resetCurrent();
  }
};

struct DataPoint {
  uint32_t time{};               ///< Reference time since epoch in ms
  int32_t currents[GEMSTACKS]{}; ///< Current in signed ADC values, use SACDecoder::ADCtoNanoAmp, to convert to nA

  void reset()
  {
    time = 0;
    std::fill(&currents[0], &currents[GEMSTACKS], 0);
  }

  ClassDefNV(DataPoint, 2);
};

struct DecodedData {
  double referenceTime{-1.};               ///< reference time when sampling clock was started
  std::vector<DataPoint> data;             ///< decoded data
  std::vector<std::bitset<NumberFEs>> fes; ///< bitmask of decoded FEs

  void resize(size_t newSize)
  {
    data.resize(newSize);
    fes.resize(newSize);
  }

  /// Number of good entries, for which all FEs were decoded
  size_t getNGoodEntries() const
  {
    size_t posGood = 0;
    while (posGood < fes.size()) {
      if (!fes[posGood].all()) {
        break;
      }
      ++posGood;
    }
    return posGood;
  }

  /// span of good entries for which all FEs were already decoded
  gsl::span<const DataPoint> getGoodData() const
  {
    return gsl::span<const DataPoint>(data.data(), data.data() + getNGoodEntries());
  }

  /// clear entries where all FEs were already decoded
  void clearGoodData()
  {
    const auto pos = getNGoodEntries();
    data.erase(data.begin(), data.begin() + pos);
    fes.erase(fes.begin(), fes.begin() + pos);
  }

  /// clear all data
  void clear()
  {
    data.clear();
    fes.clear();
  }

  /// insert entries at the front
  void insertFront(const size_t entries)
  {
    data.insert(data.begin(), entries, DataPoint());
    fes.insert(fes.begin(), entries, std::bitset<NumberFEs>());
  }

  /// copy decoded data from single FE and mark FE as received
  void setData(const size_t pos, uint32_t time, const DecodedDataFE& decdata, const int feid);

  ClassDefNV(DecodedData, 1);
};

class Decoder
{
 public:
  static constexpr std::string_view AllowedAdditionalStreams{"MRLIX"};          ///< Allowed additional data streams that can be decoded with debug stream enabled
  static constexpr uint32_t SampleDistance = 16;                                ///< Number of samples between time data stamps
  static constexpr double SampleTimeMS = LHCBunchSpacingMUS * 2.5;              ///< Internal timer sampling time in milli seconds (downsampled from LHC clock x 2500)
  static constexpr double SampleDistanceTimeMS = SampleTimeMS * SampleDistance; ///< Time distance in MS between samples

  enum class DebugFlags {
    PacketInfo = 0x01,       ///< Print packe information
    TimingInfo = 0x02,       ///< Print timing information
    ProcessingInfo = 0x04,   ///< Print some processing info
    DumpFullStream = 0x10,   ///< Dump the data character streams
    StreamSingleFE = 0x100,  ///< Stream debug output for each single FE
    StreamFinalData = 0x200, ///< Stream debug output for each single FE
  };

  enum class ReAlignType {
    None = 0,                      ///< Print packe information
    AlignOnly = 1,                 ///< Try re-alignment
    AlignAndFillMissing = 2,       ///< Try re-alignment and fill missing packets with 0s
    MaxType = AlignAndFillMissing, ///< Largest type number
  };

  bool process(const char* data, size_t size);

  void runDecoding();

  void finalize();

  void setReferenceTime(double time)
  {
    mDecodedData.referenceTime = time;
  }

  double getReferenceTime() const
  {
    return mDecodedData.referenceTime;
  }

  /// Set additional data to decode
  ///
  /// These data streams will only be decoded for debug purposes and written in the debug tree, if debug tree output is enabled
  /// \param additional additional data streams (case sensitive)
  /// Parameters are 'M': mean values over some integration time
  ///                'R': RMS values
  ///                'L': low voltage data
  ///                'I': min values
  ///                'X': max values
  void setDecodeAdditional(std::string_view additional)
  {
    mDecodeAdditional.clear();
    for (const auto c : additional) {
      if (AllowedAdditionalStreams.find(c) != std::string_view::npos) {
        mDecodeAdditional += c;
      }
    }
  }

  void enableDebugTree()
  {
    if (!mDebugStream) {
      mDebugStream = std::make_unique<o2::utils::TreeStreamRedirector>(mDebugOutputName.data(), "recreate");
    }
  }

  /// set a debug level, see `DebugFlags`
  void setDebugLevel(uint32_t level = (uint32_t)DebugFlags::PacketInfo)
  {
    mDebugLevel = level;
    if ((level & (uint32_t)DebugFlags::StreamSingleFE) || (level & (uint32_t)DebugFlags::StreamFinalData)) {
      enableDebugTree();
    }
  }

  void clearDecodedData();
  void streamDecodedData(bool streamAll = false);

  const DecodedData& getDecodedData() const { return mDecodedData; }

  void setReAlignType(ReAlignType type = ReAlignType::AlignOnly) { mReAlignType = type; }
  ReAlignType getReAlignType() const { return mReAlignType; }

  /// set the number of threads used for decoding
  /// \param nThreads number of threads
  static void setNThreads(const int nThreads) { sNThreads = nThreads; }

  /// \return returns the number of threads used for decoding
  static int getNThreads() { return sNThreads; }

 private:
  inline static int sNThreads{1};                                   ///< number of threads for decoding FEs
  size_t mCollectedDataPackets{};                                   ///< Number of collected data packets
  std::array<uint32_t, Instances> mPktCountInstance{};              ///< Packet counter for the instance
  std::array<uint32_t, NumberFEs> mPktCountFEs{};                   ///< Packet counter for the single FEs
  std::array<std::pair<uint32_t, uint32_t>, NumberFEs> mTSCountFEs; ///< Counter how often the time stamp was seen for the single FEs, all / valid
  std::array<std::string, NumberFEs> mDataStrings;                  ///< ASCI data sent by FE
  DecodedData mDecodedData;                                         ///< decoded data
  std::unique_ptr<o2::utils::TreeStreamRedirector> mDebugStream;    ///< Debug output streamer
  std::string mDecodeAdditional;                                    ///< Decode these additional data for debugging purposes
  std::string mDebugOutputName{"SAC_debug.root"};                   ///< name of the debug output tree
  ReAlignType mReAlignType{ReAlignType::None};                      ///< if data cannot be dedoced, try to re-align the stream

  uint32_t mDebugLevel{0}; ///< Amount of debug information to print

  uint32_t decodeTimeStamp(const char* data);

  /// \return status message: 1 = good, 0 = data length too short, -1 = decoding error
  int decodeChannels(DecodedDataFE& sacs, size_t& carry, int feid);
  void decode(int feid);

  void printPacketInfo(const sac::packet& sac);

  void dumpStreams();

  ClassDefNV(Decoder, 0);
};

} // namespace o2::tpc::sac
#endif
