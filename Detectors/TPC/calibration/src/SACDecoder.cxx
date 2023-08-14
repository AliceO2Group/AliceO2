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

#include <chrono>
#include <cassert>
#include <fstream>
#include <string_view>
#if (defined(WITH_OPENMP) || defined(_OPENMP))
#include <omp.h>
#endif

#include "Framework/Logger.h"

#include "TPCCalibration/SACDecoder.h"

using HighResClock = std::chrono::high_resolution_clock;
using namespace o2::tpc::sac;

void DecodedData::setData(const size_t pos, uint32_t time, const DecodedDataFE& decdata, const int feid)
{
  data[pos].time = time;
  auto& currents = data[pos].currents;
  std::copy(decdata.currents.begin(), decdata.currents.end(), &currents[0] + feid * ChannelsPerFE);
  if (fes[pos].test(feid)) {
    LOGP(warning, "FE {} already set at time {}, position {}", feid, time, pos);
  }
  fes[pos].set(feid);
}

//______________________________________________________________________________
bool Decoder::process(const char* data, size_t size)
{
  const auto startTime = HighResClock::now();
  assert(size == sizeof(sac::packet));
  auto& sac = *(sac::packet*)data;
  const auto instance = sac.getInstance();
  if (instance >= Instances) {
    return true;
  }

  if (mDebugLevel & (uint32_t)DebugFlags::PacketInfo) {
    printPacketInfo(sac);
  }

  const auto packetInstance = sac.header.pktCount;
  const auto packetFE = sac.data.pktNumber;
  const bool isOK = sac.check();
  const auto dataWords = sac.getDataWords();
  const auto feIndex = sac.getFEIndex();
  auto& lastPacketInstance = mPktCountInstance[instance];
  auto& lastPacketFE = mPktCountFEs[feIndex];

  // check packet counters are increasing by one
  //
  if (lastPacketInstance && (packetInstance != (lastPacketInstance + 1))) {
    LOGP(error, "Packet for instance {} missing, last packet {}, this packet {}", instance, lastPacketInstance, packetInstance);
  }

  if (lastPacketFE && (packetFE != (lastPacketFE + 1))) {
    LOGP(error, "Packet for frontend {} missing, last packet {}, this packet {}", feIndex, lastPacketFE, packetFE);
  }
  lastPacketInstance = packetInstance;
  lastPacketFE = packetFE;

  if (isOK) {
    auto& dataStrings = mDataStrings[feIndex];
    dataStrings.insert(dataStrings.end(), dataWords.begin(), dataWords.end());
  } else {
    LOGP(error, "Problem in SAC data found, header check: {}, data check: {}", sac.header.check(), sac.data.check());
  }

  if (mDebugLevel & (uint32_t)DebugFlags::TimingInfo) {
    auto endTime = HighResClock::now();
    auto elapsed_seconds = std::chrono::duration_cast<std::chrono::duration<double>>(endTime - startTime);
    LOGP(detail, "Time to process data of size {}: {} s", size, elapsed_seconds.count());
  }

  ++mCollectedDataPackets;
  return isOK;
}

int Decoder::decodeChannels(DecodedDataFE& sacs, size_t& carry, int feid)
{
  const auto& data = mDataStrings[feid];
  const size_t dataSize = data.size();
  const size_t next = std::min(size_t(8 * 8), dataSize - carry);
  const size_t start = carry;
  while (carry < dataSize) {
    if (carry + 8 >= dataSize) {
      return 0;
    }
    if (data[carry] >= '0' && data[carry] <= '7') {
      const uint32_t channel = data[carry] - '0';
      ++carry;
      uint32_t value = 0;
      for (int i = 0; i < 6; ++i) {
        const auto c = data[carry];
        uint32_t nibble = 0;
        if ((c >= '0') && (c <= '9')) {
          nibble = c - '0';
        } else if ((c >= 'A') && (c <= 'F')) {
          nibble = c - 'A' + 10;
        } else {
          LOGP(warning, "Problem decoding data value for FE {}, channel {} at position {} / {}, no valid hex charakter, dump: {}\n",
               feid, channel, carry, dataSize, std::string_view(&data[start], next));
          return -1;
        }
        value <<= 4;
        value |= (nibble & 0xF);
        ++carry;
      }
      int32_t valueSigned = value & 0x00FFFFFF;
      // negative value?
      if ((valueSigned >> 23) & 1) {
        valueSigned |= 0xff000000;
      }
      sacs.currents[channel] = valueSigned;

      if (data[carry] != '\n') {
        LOGP(warning, "Problem decoding data value for FE {}, channel {} at position {} / {}, CR expected, dump: {}\n",
             feid, channel, carry, dataSize, std::string_view(&data[start], next));
        return -1;
      }
      ++carry;
    } else {
      return 1;
    }
  }
  return 1;
}

void Decoder::decode(int feid)
{
  const auto startTime = HighResClock::now();
  auto& data = mDataStrings[feid];
  DecodedDataFE decdata;
  DecodedDataFE decAdditional;
  bool aligned{false};
  bool syncLost{false};

  size_t carry = 0;
  size_t deletePosition = 0;
  const size_t dataSize = data.size();

  // fmt::print("================ Processing feid {:2} with size {}  =================\n", feid, data.size());
  while (carry < dataSize) {
    if (!aligned) {
      // check for re-aligning sequence
      if ((carry == 0) && (data[0] == '\n') && (data[1] == 's')) {
        carry += 2;
      } else {
        while (data[carry] != '\n') {
          if (carry >= dataSize) {
            break;
          }
          ++carry;
        }
        ++carry;
      }
      aligned = true;
    }
    // fmt::print("Checking position {} / {}, {}\n", carry, dataSize, std::string_view(&data[carry], std::min(size_t(20), dataSize - carry)));

    if (data[carry] >= '0' && data[carry] <= '7') {
      const auto status = decodeChannels(decdata, carry, feid);
      if (status == 0) {
        break;
      } else if (status == -1) {
        if (mReAlignType != ReAlignType::None) {
          LOGP(warn, "trying to re-align data stream\n");
          aligned = false;
          syncLost = true;
        } else {
          LOGP(error, "stopping decoding\n");
          break;
        }
      }
    } else if (data[carry] == 'S') {
      if (carry + 11 >= dataSize) {
        break;
      }
      const auto streamStart = carry;
      // time stamp comes after channel data
      ++carry;
      std::string_view vd(&data[carry], 8);
      std::stringstream str(vd.data());
      str.flags(std::ios_base::hex);
      uint32_t timeStamp;
      str >> timeStamp;
      decdata.timeStamp = timeStamp;
      decAdditional.timeStamp = timeStamp;

      carry += 8;
      if (data[carry] != '\n' || data[carry + 1] != 's') {
        LOGP(warning, "Problem decoding time stamp for FE ({}) at position {} / {}, dump: {}\n",
             feid, carry - 8, dataSize, std::string_view(&data[carry - 8], std::min(size_t(20), dataSize - 8 - carry)));
        break; // TODO: makes sense?
      } else {
        deletePosition = carry; // keep \ns to align in next iteration
        carry += 2;

#pragma omp critical
        {
          if (mDebugStream && (mDebugLevel & (uint32_t)DebugFlags::StreamSingleFE)) {
            (*mDebugStream) << "d"
                            << "data=" << decdata
                            << "feid=" << feid
                            << "tscount=" << mTSCountFEs[feid]
                            << "\n";
          }
          ++mTSCountFEs[feid].first;

          // copy decoded data to output
          const auto refTime = timeStamp / SampleDistance;
          auto& currentsTime = mDecodedData.data;
          const auto nSamples = currentsTime.size();
          auto firstRefTime = (nSamples > 0) ? currentsTime[0].time : refTime;
          auto& refCount = mTSCountFEs[feid].second;
          // NOTE: use (refTime > 1) instead of (refTime > 0), since in some cases the packet with TS 0 is missing
          if ((refCount == 0) && (refTime > 1)) {
            LOGP(info, "Skipping initial data packet {} with time stamp {} for FE {}", mTSCountFEs[feid].first, timeStamp, feid);
          } else {
            if (refTime < firstRefTime) {
              // LOGP(info, "FE {}: {} < {}, adding {} DataPoint(s)", feid, refTime, firstRefTime, firstRefTime - refTime);
              mDecodedData.insertFront(firstRefTime - refTime);
              firstRefTime = refTime;
            } else if (nSamples < refTime - firstRefTime + 1) {
              // LOGP(info, "FE {}: refTime {}, firstRefTime {}, resize from {} to {}", feid, refTime, firstRefTime, currentsTime.size(), refTime - firstRefTime + 1);
              mDecodedData.resize(refTime - firstRefTime + 1);
            }
            // LOGP(info, "FE {}: insert refTime {} at pos {}, with firstRefTime {}", feid, refTime, refTime - firstRefTime, firstRefTime);
            mDecodedData.setData(refTime - firstRefTime, refTime, decdata, feid);

            if (refCount != refTime) {
              LOGP(warning, "Unexpected time stamp in FE {}. Count {} != TS {} ({}), dump: {}", feid, refCount, refTime, timeStamp, std::string_view(&data[streamStart], std::min(size_t(20), dataSize - streamStart - carry)));
              // NOTE: be graceful in case TS 0 is missing and avoid furhter warnings
              if (((refCount == 0) && (refTime == 1)) || ((mReAlignType == ReAlignType::AlignAndFillMissing) && syncLost)) {
                while (refCount < refTime) {
                  mDecodedData.setData(refCount - firstRefTime, refCount, DecodedDataFE(), feid);
                  LOGP(warning, "Adding dummy data for FE {}, TS {}", feid, refCount);
                  ++refCount;
                }
                syncLost = false;
              }
            }
            ++refCount;
          }
        }
      }

      decdata.reset();
    } else if (const auto pos = mDecodeAdditional.find(data[carry]); (pos != std::string::npos) && mDebugStream) {
      // in case of debug stream output, decode additionally configured data streams
      const auto streamStart = carry;
      const char streamType = data[carry];
      const char endMarker = streamType + 32;
      ++carry;

      if (!decodeChannels(decAdditional, carry, feid)) {
        break;
      }

      if (data[carry] != endMarker) {
        LOGP(warning, "Problem decoding additional stream '{}' values for FE ({}) at position {} / {}, dump: {}",
             streamType, feid, carry, dataSize, std::string_view(&data[streamStart], std::min(size_t(20), dataSize - streamStart - carry)));
      } else {
        const char treeName[2] = {streamType, '\0'};
        (*mDebugStream) << treeName
                        << "data=" << decAdditional
                        << "feid=" << feid
                        << "\n";
      }

      decAdditional.reset();
      ++carry;
    } else if (AllowedAdditionalStreams.find(data[carry]) != std::string_view::npos) {
      // skip stream if not configured or no debug stream
      const char streamType = data[carry];
      const char endMarker = streamType + 32;
      while (data[carry] != endMarker) {
        if (carry >= dataSize) {
          break;
        }
        ++carry;
      }
      ++carry;
    } else if (data[carry] >= 'a' && data[carry] <= 'z') {
      if (mReAlignType != ReAlignType::None) {
        LOGP(error, "Skipping {} for FE {}, trying to re-align data stream", data[carry], feid);
        aligned = false;
        syncLost = true;
      } else {
        LOGP(error, "Skipping {} for FE {}, might lead to decoding problems", data[carry], feid);
        ++carry;
      }
      decdata.reset();
    } else {
      if (mReAlignType != ReAlignType::None) {
        LOGP(warn, "Can't interpret position for FE {}, {} / {}, {}, trying to re-align data stream\n", feid, carry, dataSize, std::string_view(&data[carry - 8], std::min(size_t(20), dataSize - 8 - carry)));
        aligned = false;
        syncLost = true;
      } else {
        LOGP(error, "Can't interpret position for FE {}, {} / {}, {}, stopping decoding\n", feid, carry, dataSize, std::string_view(&data[carry - 8], std::min(size_t(20), dataSize - 8 - carry)));
        break;
      }
    }
  }

  // Remove already decoded data
  data.erase(data.begin(), data.begin() + deletePosition);
  // LOGP(info, "removing {} characters from stream. Old size {}, new size {}", deletePosition, dataSize, data.size());

  if (mDebugLevel & (uint32_t)DebugFlags::TimingInfo) {
    auto endTime = HighResClock::now();
    auto elapsed_seconds = std::chrono::duration_cast<std::chrono::duration<double>>(endTime - startTime);
    LOGP(detail, "Time to decode feid {}: {} s", feid, elapsed_seconds.count());
  }
}

void Decoder::runDecoding()
{
  const auto startTime = HighResClock::now();

#pragma omp parallel for num_threads(sNThreads)
  for (size_t feid = 0; feid < NumberFEs; ++feid) {
    decode(feid);
  }

  if (mDebugLevel & (uint32_t)DebugFlags::TimingInfo) {
    auto endTime = HighResClock::now();
    auto elapsed_seconds = std::chrono::duration_cast<std::chrono::duration<double>>(endTime - startTime);
    LOGP(detail, "Time to decode all feids {} s, {} s per packet ({})", elapsed_seconds.count(), elapsed_seconds.count() / mCollectedDataPackets, mCollectedDataPackets);
  }
}

void Decoder::streamDecodedData(bool streamAll)
{
  if (mDebugStream && (mDebugLevel & (uint32_t)DebugFlags::StreamFinalData)) {
    const size_t nDecodedData = (streamAll) ? mDecodedData.data.size() : mDecodedData.getNGoodEntries();
    LOGP(info, "streamDecodedData (streamAll {}): {} / {}", streamAll, nDecodedData, mDecodedData.data.size());
    auto refTime = mDecodedData.referenceTime;
    for (size_t ientry = 0; ientry < nDecodedData; ++ientry) {
      auto& currentData = mDecodedData.data[ientry];
      auto fes = mDecodedData.fes[ientry].to_ulong();
      auto nfes = mDecodedData.fes[ientry].count();
      (*mDebugStream) << "c"
                      << "refTime=" << refTime
                      << "values=" << currentData
                      << "fes=" << fes
                      << "nfes=" << nfes
                      << "\n";
    }
  }
}

void Decoder::finalize()
{
  LOGP(info, "Finalize sac::Decoder with {} good / {} remaining entries",
       mDecodedData.getNGoodEntries(), mDecodedData.data.size());

  if (mDebugLevel & (uint32_t)DebugFlags::DumpFullStream) {
    dumpStreams();
  }

  runDecoding();
  streamDecodedData(true);

  if (mDebugStream) {
    mDebugStream->Close();
    mDebugStream.reset();
  }
}

void Decoder::clearDecodedData()
{
  streamDecodedData();
  if (mDebugLevel & (uint32_t)DebugFlags::ProcessingInfo) {
    auto& data = mDecodedData.data;
    const auto posGood = mDecodedData.getNGoodEntries();
    LOGP(info, "Clearing data of size {}, firstTS {}, lastTS {}",
         posGood, (data.size() > 0) ? data.front().time : -1, (posGood > 0) ? data[posGood - 1].time : 0);
  }
  mDecodedData.clearGoodData();
}

void Decoder::printPacketInfo(const sac::packet& sac)
{
  const auto& header = sac.header;
  const auto& sacc = sac.data;

  LOGP(info, "{:>4} {:>4} {:>8} {:>8} -- {:>4} {:>4} {:>8} {:>8} {:>10} -- {:>4}\n", //
       "vers",                                                                       //
       "inst",                                                                       //
       "bc",                                                                         //
       "pktCnt",                                                                     //
       "feid",                                                                       //
       "size",                                                                       //
       "pktNum",                                                                     //
       "time",                                                                       //
       "crc32",                                                                      //
       "ok"                                                                          //
  );

  LOGP(info, "{:>4} {:>4} {:>8} {:>8} -- {:>4} {:>4} {:>8} {:>8} {:>#10x} -- {:>4b}\n", //
       header.version,                                                                  //
       header.instance,                                                                 //
       header.bunchCrossing,                                                            //
       header.pktCount,                                                                 //
       sacc.feid,                                                                       //
       sacc.pktSize,                                                                    //
       sacc.pktNumber,                                                                  //
       sacc.timeStamp,                                                                  //
       sacc.crc32,                                                                      //
       sacc.check()                                                                     //
  );
}

void Decoder::dumpStreams()
{
  const std::string outNameBase(mDebugOutputName.substr(0, mDebugOutputName.size() - 5));
  for (size_t feid = 0; feid < NumberFEs; ++feid) {
    std::string outName = outNameBase;
    outName += fmt::format(".feid_{}.stream.txt", feid);
    std::ofstream fout(outName.data());
    const auto& data = mDataStrings[feid];
    fout << std::string_view(&data[0], data.size());
  }
}
