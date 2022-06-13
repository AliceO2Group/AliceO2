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

/// @file   FITDataDecoderDPLSpec.cxx

#include "FT0Workflow/FT0DataDecoderDPLSpec.h"
#include <numeric>
#include <emmintrin.h>
#include <immintrin.h>
#include <algorithm>
#include <cstdlib>
#include <cstring>
#include <string>
namespace o2
{
namespace ft0
{

void FT0DataDecoderDPLSpec::run(ProcessingContext& pc)
{
  auto t1 = std::chrono::high_resolution_clock::now();

  // if we see requested data type input with 0xDEADBEEF subspec and 0 payload this means that the "delayed message"
  // mechanism created it in absence of real data from upstream. Processor should send empty output to not block the workflow
  {
    static size_t contDeadBeef = 0; // number of times 0xDEADBEEF was seen continuously
    std::vector<InputSpec> dummy{InputSpec{"dummy", ConcreteDataMatcher{"FT0", o2::header::gDataDescriptionRawData, 0xDEADBEEF}}};
    for (const auto& ref : InputRecordWalker(pc.inputs(), dummy)) {
      const auto dh = o2::framework::DataRefUtils::getHeader<o2::header::DataHeader*>(ref);
      auto payloadSize = DataRefUtils::getPayloadSize(ref);
      if (payloadSize == 0) {
        auto maxWarn = o2::conf::VerbosityConfig::Instance().maxWarnDeadBeef;
        if (++contDeadBeef <= maxWarn) {
          LOGP(alarm, "Found input [{}/{}/{:#x}] TF#{} 1st_orbit:{} Payload {} : assuming no payload for all links in this TF{}",
               dh->dataOrigin.str, dh->dataDescription.str, dh->subSpecification, dh->tfCounter, dh->firstTForbit, payloadSize,
               contDeadBeef == maxWarn ? fmt::format(". {} such inputs in row received, stopping reporting", contDeadBeef) : "");
        }
        mVecDigits.resize(0);
        pc.outputs().snapshot(o2::framework::Output{o2::header::gDataOriginFT0, "DIGITSBC", 0, o2::framework::Lifetime::Timeframe}, mVecDigits);
        mVecChannelData.resize(0);
        pc.outputs().snapshot(o2::framework::Output{o2::header::gDataOriginFT0, "DIGITSCH", 0, o2::framework::Lifetime::Timeframe}, mVecChannelData);
        return;
      }
    }
    contDeadBeef = 0; // if good data, reset the counter
  }
  std::vector<InputSpec> filter{InputSpec{"filter", ConcreteDataTypeMatcher{"FT0", o2::header::gDataDescriptionRawData}, Lifetime::Timeframe}};
  DPLRawParser parser(pc.inputs(), filter);

  using ArrRdhPtrPerLink = std::array<std::vector<const o2::header::RAWDataHeader*>, sNlinksMax>;
  using ArrDataPerLink = std::array<std::vector<gsl::span<const uint8_t>>, sNlinksMax>;
  std::array<ArrRdhPtrPerLink, sNorbits> arrRdhPtrPerOrbit{};
  std::array<ArrDataPerLink, sNorbits> arrDataPerOrbit{};
  std::array<std::vector<const o2::header::RAWDataHeader*>, sNorbits> arrRdhTCMperOrbit{};
  std::array<std::vector<gsl::span<const uint8_t>>, sNorbits> arrDataTCMperOrbit{};
  std::array<std::size_t, sNorbits> arrOrbitSizePages{};
  std::array<std::size_t, sNorbits> arrOrbitSizePagesTCM{};
  std::array<uint64_t, sNorbits> arrOrbit{};

  for (auto it = parser.begin(), end = parser.end(); it != end; ++it) {
    // Aggregating pages by orbit and FeeID
    if (!it.size()) {
      continue; // excluding pages without payload
    }
    auto rdhPtr = it.get_if<o2::header::RAWDataHeader>();
    const uint16_t orbitTF = (rdhPtr->orbit) % 256;
    // const uint16_t feeID=rdhPtr->feeId;
    arrOrbit[orbitTF] = rdhPtr->orbit;
    const auto& linkID = rdhPtr->linkID;
    const auto& endPoint = rdhPtr->endPointID;
    const uint16_t feeID = linkID + 12 * endPoint;
    if (feeID == mFEEID_TCM) {
      // Iterator is noncopyable, preparing RDH pointers and span objects
      arrOrbitSizePagesTCM[orbitTF] += it.size();
      arrRdhTCMperOrbit[orbitTF].push_back(it.get_if<o2::header::RAWDataHeader>());
      arrDataTCMperOrbit[orbitTF].emplace_back(it.data(), it.size());
    } else {
      arrOrbitSizePages[orbitTF] += it.size();
      arrRdhPtrPerOrbit[orbitTF][feeID].push_back(it.get_if<o2::header::RAWDataHeader>());
      arrDataPerOrbit[orbitTF][feeID].emplace_back(it.data(), it.size());
    }
  }
  uint64_t chPosOrbit{0};
  uint64_t eventPosPerOrbit{0};

  uint64_t posChDataPerOrbit[sNorbits]{}; // position per orbit
  uint64_t nChDataPerOrbit[sNorbits]{};   // number of events per orbit
  NChDataBC_t posChDataPerBC[sNorbits]{};
  for (int iOrbit = 0; iOrbit < sNorbits; iOrbit++) {
    const auto& orbit = arrOrbit[iOrbit];
    NChDataOrbitBC_t bufBC{};
    NChDataBC_t buf_nChPerBC{};
    if (arrOrbitSizePages[iOrbit] > 0) {
      for (int iFeeID = 0; iFeeID < sNlinksMax; iFeeID++) {
        if (iFeeID == mFEEID_TCM) {
          continue;
        }
        const auto& nPages = arrRdhPtrPerOrbit[iOrbit][iFeeID].size();

        for (int iPage = 0; iPage < nPages; iPage++) {
          const auto& rdhPtr = arrRdhPtrPerOrbit[iOrbit][iFeeID][iPage];
          const auto& payload = arrDataPerOrbit[iOrbit][iFeeID][iPage].data();
          const auto& payloadSize = arrDataPerOrbit[iOrbit][iFeeID][iPage].size();
          const uint8_t* src = (uint8_t*)payload;
          const auto nNGBTwords = payloadSize / 16;
          const int nNGBTwordsDiff = nNGBTwords % 16;
          const int nChunks = nNGBTwords / 16 + static_cast<int>(nNGBTwordsDiff > 0);
          const auto lastChunk = nChunks - 1;
          const uint16_t mask = (0xffff << (16 - nNGBTwordsDiff)) | (0xffff * (nNGBTwordsDiff == 0));
          __m512i zmm_pos1 = _mm512_set_epi32(0, 16, 32, 48, 64, 80, 96, 112, 128, 144, 160, 176, 192, 208, 224, 240);
          __m512i zmm_pos2 = _mm512_set1_epi32(6);
          zmm_pos2 = _mm512_add_epi32(zmm_pos1, zmm_pos2);
          for (int iChunk = 0; iChunk < nChunks; iChunk++) {
            __mmask16 mask16_MaxGBTwords = _mm512_int2mask((0xffff * (iChunk != lastChunk)) | mask);
            __m512i zmm_mask_zero = _mm512_setzero_epi32();

            __m512i zmm_src_column0_part0 = _mm512_mask_i32gather_epi32(zmm_mask_zero, mask16_MaxGBTwords, zmm_pos1, src, 1);
            __m512i zmm_src_column1_part1 = _mm512_mask_i32gather_epi32(zmm_mask_zero, mask16_MaxGBTwords, zmm_pos2, src, 1);
            // ChID column(contains ChID in data and descriptor in header)
            __m512i zmm_mask_localChID = _mm512_set1_epi32(0xf);
            __m512i zmm_buf = _mm512_srai_epi32(zmm_src_column1_part1, 28);
            __m512i zmm_ChID_column1 = _mm512_and_epi32(zmm_buf, zmm_mask_localChID);
            // Header
            __mmask16 mask16_header = _mm512_cmpeq_epi32_mask(zmm_ChID_column1, zmm_mask_localChID);
            // NGBTwords column
            zmm_buf = _mm512_srai_epi32(zmm_src_column1_part1, 24);
            __m512i zmm_NGBTwords = _mm512_maskz_and_epi32(mask16_header, zmm_buf, zmm_mask_localChID);

            // Checking for empty events which contains only header(NGBTwords=0), and getting last header position within chunk
            __mmask16 mask16_header_final = _mm512_mask_cmpgt_epu32_mask(mask16_header, zmm_NGBTwords, zmm_mask_zero);

            // BC
            __m512i zmm_mask_time = _mm512_set1_epi32(0xfff);
            __m512i zmm_bc = _mm512_mask_and_epi32(zmm_mask_zero, mask16_header_final, zmm_src_column0_part0, zmm_mask_time);

            // Estimation for number of channels
            __m512i zmm_Nchannels = _mm512_slli_epi32(zmm_NGBTwords, 1); // multiply by 2

            __m512i zmm_last_word_pos = zmm_NGBTwords;
            zmm_last_word_pos = _mm512_slli_epi32(zmm_last_word_pos, 4); // multiply by 16, byte position for last word
            zmm_last_word_pos = _mm512_add_epi32(zmm_last_word_pos, zmm_pos1);
            zmm_buf = _mm512_i32gather_epi32(zmm_last_word_pos, src + 8, 1);
            zmm_buf = _mm512_srai_epi32(zmm_buf, 12);
            __m512i zmm_ChID_last_column1 = _mm512_and_epi32(zmm_buf, zmm_mask_localChID);

            __mmask16 mask16_half_word = _mm512_cmpeq_epi32_mask(zmm_ChID_last_column1, zmm_mask_zero);
            __m512i zmm_mask_one = _mm512_set1_epi32(1);
            __m512i zmm_Nch = _mm512_mask_sub_epi32(zmm_Nchannels, mask16_half_word, zmm_Nchannels, zmm_mask_one);
            __m512i zmm_nEventPerBC = _mm512_mask_i32gather_epi32(zmm_mask_zero, mask16_header_final, zmm_bc, buf_nChPerBC.data(), 4);
            zmm_buf = _mm512_add_epi32(zmm_nEventPerBC, zmm_Nch);
            _mm512_mask_i32scatter_epi32(buf_nChPerBC.data(), mask16_header_final, zmm_bc, zmm_buf, 4);

            zmm_buf = _mm512_set1_epi32(256);
            zmm_pos1 = _mm512_add_epi32(zmm_buf, zmm_pos1);
            zmm_pos2 = _mm512_add_epi32(zmm_buf, zmm_pos2);

          } // chunk
        }   // Page
        if (iFeeID != sNlinksMax - 1) {
          memcpy(bufBC[iFeeID + 1].data(), buf_nChPerBC.data(), (sNBC + 4) * 4);
        }
      } // linkID
    }
    // Channel data position within BC per LinkID
    memcpy(mPosChDataPerLinkOrbit[iOrbit].data(), bufBC.data(), (sNBC + 4) * 4 * sNlinksMax);
    // TCM proccessing

    uint8_t* ptrDstTCM = (uint8_t*)mVecTriggers.data();
    if (arrOrbitSizePagesTCM[iOrbit] > 0) {
      memset(mVecTriggers.data(), 0, 16 * 3564);
      const auto& nPagesTCM = arrRdhTCMperOrbit[iOrbit].size();
      for (int iPage = 0; iPage < nPagesTCM; iPage++) {
        const auto& rdhPtr = arrRdhTCMperOrbit[iOrbit][iPage];
        const auto& payload = arrDataTCMperOrbit[iOrbit][iPage].data();
        const auto& payloadSize = arrDataTCMperOrbit[iOrbit][iPage].size();
        const uint8_t* src = (uint8_t*)payload;
        const auto nNGBTwords = payloadSize / 16;
        const int nNGBTwordsDiff = nNGBTwords % 16;
        const int nChunks = nNGBTwords / 16 + static_cast<int>(nNGBTwordsDiff > 0);
        const auto lastChunk = nChunks - 1;
        const uint16_t mask = (0xffff << (16 - nNGBTwordsDiff)) | (0xffff * (nNGBTwordsDiff == 0));
        __m512i zmm_pos1 = _mm512_set_epi32(0, 16, 32, 48, 64, 80, 96, 112, 128, 144, 160, 176, 192, 208, 224, 240);
        for (int iChunk = 0; iChunk < nChunks; iChunk++) {
          __mmask16 mask16_MaxGBTwords = _mm512_int2mask((0xffff * (iChunk != lastChunk)) | mask);
          __m512i zmm_mask_zero = _mm512_setzero_epi32();

          __m512i zmm_src_header = _mm512_mask_i32gather_epi32(zmm_mask_zero, mask16_MaxGBTwords, zmm_pos1, src + 9, 1);

          __m512i zmm_mask_header = _mm512_set1_epi32(0xf1); // one GBT word + descriptor
          // Header
          __mmask16 mask16_header = _mm512_cmpeq_epi32_mask(zmm_src_header, zmm_mask_header);
          // BC
          __m512i zmm_bc = _mm512_mask_i32gather_epi32(zmm_mask_zero, mask16_MaxGBTwords, zmm_pos1, src, 1);
          __m512i zmm_mask_12bit = _mm512_set1_epi32(0xfff);

          zmm_bc = _mm512_maskz_and_epi32(mask16_header, zmm_mask_12bit, zmm_bc);
          // Position of first GBT word with data
          __m512i zmm_pos2 = _mm512_set1_epi32(16);
          zmm_pos2 = _mm512_maskz_add_epi32(mask16_header, zmm_pos1, zmm_pos2);

          __m512i zmm_src_part0 = _mm512_mask_i32gather_epi32(zmm_mask_zero, mask16_header, zmm_pos2, src, 1);
          __m512i zmm_src_part1 = _mm512_mask_i32gather_epi32(zmm_mask_zero, mask16_header, zmm_pos2, src + 3, 1);
          __m512i zmm_src_part2 = _mm512_mask_i32gather_epi32(zmm_mask_zero, mask16_header, zmm_pos2, src + 7, 1);
          // Trigger bits + NchanA + NchanC
          __m512i zmm_mask_3byte = _mm512_set1_epi32(0xffffff);
          __m512i zmm_dst_part0 = _mm512_and_epi32(zmm_src_part0, zmm_mask_3byte);
          // Sum AmpA
          __m512i zmm_mask_17bit = _mm512_set1_epi32(0x1ffff);
          __m512i zmm_dst_part1 = _mm512_and_epi32(zmm_src_part1, zmm_mask_17bit);
          // Sum AmpC
          __m512i zmm_dst_part2 = _mm512_srai_epi32(zmm_src_part1, 18);
          __m512i zmm_mask_14bit = _mm512_set1_epi32(0b11111111111111);
          zmm_dst_part2 = _mm512_and_epi32(zmm_mask_14bit, zmm_dst_part2);

          __m512i zmm_buf = _mm512_slli_epi32(zmm_src_part2, 14);
          __m512i zmm_mask_3bit = _mm512_set1_epi32(0b11100000000000);
          zmm_buf = _mm512_and_epi32(zmm_mask_3bit, zmm_buf);
          zmm_dst_part2 = _mm512_or_epi32(zmm_dst_part2, zmm_buf);
          // Average time A + C
          __m512i zmm_dst_part3 = _mm512_srai_epi32(zmm_src_part2, 4);
          __m512i zmm_mask_9bit = _mm512_set1_epi32(0x1ff);
          zmm_dst_part3 = _mm512_and_epi32(zmm_dst_part3, zmm_mask_9bit);

          zmm_buf = _mm512_slli_epi32(zmm_src_part2, 2);
          __m512i zmm_mask_9bit_2 = _mm512_set1_epi32(0x1ff0000);
          zmm_buf = _mm512_and_epi32(zmm_buf, zmm_mask_9bit_2);

          zmm_dst_part3 = _mm512_or_epi32(zmm_buf, zmm_dst_part3);
          // Position
          __m512i zmm_dst_pos = _mm512_slli_epi32(zmm_bc, 4);
          // Pushing data to buffer
          _mm512_mask_i32scatter_epi32(ptrDstTCM, mask16_header, zmm_dst_pos, zmm_dst_part0, 1);
          _mm512_mask_i32scatter_epi32(ptrDstTCM + 4, mask16_header, zmm_dst_pos, zmm_dst_part1, 1);
          _mm512_mask_i32scatter_epi32(ptrDstTCM + 8, mask16_header, zmm_dst_pos, zmm_dst_part2, 1);
          _mm512_mask_i32scatter_epi32(ptrDstTCM + 12, mask16_header, zmm_dst_pos, zmm_dst_part3, 1);

          zmm_buf = _mm512_set1_epi32(256);
          zmm_pos1 = _mm512_add_epi32(zmm_buf, zmm_pos1);
        } // chunk
      }   // page
    }
    NChDataBC_t buf_nPosPerBC{};
    uint64_t nChPerBC{0};
    uint64_t nEventOrbit{0};
    posChDataPerOrbit[iOrbit] = chPosOrbit; //

    __m512i zmm_mask_seq2 = _mm512_set_epi32(15, 14, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1, 0);
    __m512i zmm_pos2 = _mm512_set_epi32(75, 70, 65, 60, 55, 50, 45, 40, 35, 30, 25, 20, 15, 10, 5, 0);
    for (int iBC = 0; iBC < sNBC; iBC += 16) {
      uint32_t buf0[16], buf1[16], buf2[16], buf3[16];
      uint8_t* dst = (uint8_t*)&mVecDigits[eventPosPerOrbit + nEventOrbit];
      __m512i zmm_nChPerBC = _mm512_loadu_si512(&buf_nChPerBC[iBC]);

#define SUM(N)                       \
  buf_nPosPerBC[iBC + N] = nChPerBC; \
  nChPerBC += buf_nChPerBC[iBC + N];

      SUM(0);
      SUM(1);
      SUM(2);
      SUM(3);

      SUM(4);
      SUM(5);
      SUM(6);
      SUM(7);

      SUM(8);
      SUM(9);
      SUM(10);
      SUM(11);

      SUM(12);
      SUM(13);
      SUM(14);
      SUM(15);
      __m512i zmm_mask_zero = _mm512_setzero_epi32();
      __mmask16 mask16_bc = _mm512_cmpneq_epi32_mask(zmm_nChPerBC, zmm_mask_zero);
      __m512i zmm_pos3 = _mm512_maskz_expand_epi32(mask16_bc, zmm_pos2);
      const auto nEvents = _mm_popcnt_u32(_cvtmask16_u32(mask16_bc));
      nEventOrbit += nEvents;
      __m512i zmm_pos = _mm512_loadu_si512(&buf_nPosPerBC[iBC]);
      __m512i zmm_pos_per_orbit = _mm512_set1_epi32(chPosOrbit);
      zmm_pos = _mm512_add_epi32(zmm_pos, zmm_pos_per_orbit);
      __m512i zmm_bc = _mm512_set1_epi32(iBC);
      zmm_bc = _mm512_add_epi32(zmm_bc, zmm_mask_seq2);
      __m512i zmm_orbit = _mm512_set1_epi32(orbit);

      _mm512_mask_i32scatter_epi32(dst, mask16_bc, zmm_pos3, zmm_pos, 8);
      _mm512_mask_i32scatter_epi32(dst + 4, mask16_bc, zmm_pos3, zmm_nChPerBC, 8);
      _mm512_mask_i32scatter_epi32(dst + 28, mask16_bc, zmm_pos3, zmm_bc, 8);
      _mm512_mask_i32scatter_epi32(dst + 32, mask16_bc, zmm_pos3, zmm_orbit, 8);
      // TCM
      __m512i zmm_src_pos = _mm512_slli_epi32(zmm_bc, 4);

      __m512i zmm_dst_part0 = _mm512_mask_i32gather_epi32(zmm_mask_zero, mask16_bc, zmm_src_pos, ptrDstTCM, 1);
      __m512i zmm_dst_part1 = _mm512_mask_i32gather_epi32(zmm_mask_zero, mask16_bc, zmm_src_pos, ptrDstTCM + 4, 1);
      __m512i zmm_dst_part2 = _mm512_mask_i32gather_epi32(zmm_mask_zero, mask16_bc, zmm_src_pos, ptrDstTCM + 8, 1);
      __m512i zmm_dst_part3 = _mm512_mask_i32gather_epi32(zmm_mask_zero, mask16_bc, zmm_src_pos, ptrDstTCM + 12, 1);

      _mm512_mask_i32scatter_epi32(dst + 8, mask16_bc, zmm_pos3, zmm_dst_part0, 8);
      _mm512_mask_i32scatter_epi32(dst + 12, mask16_bc, zmm_pos3, zmm_dst_part1, 8);
      _mm512_mask_i32scatter_epi32(dst + 16, mask16_bc, zmm_pos3, zmm_dst_part2, 8);
      _mm512_mask_i32scatter_epi32(dst + 20, mask16_bc, zmm_pos3, zmm_dst_part3, 8);

    } // BC

    const uint64_t buf_nChDataPerOrbit = nChPerBC;
    chPosOrbit += nChPerBC;

    nChDataPerOrbit[iOrbit] = nChPerBC;
    eventPosPerOrbit += nEventOrbit;
    memcpy(posChDataPerBC[iOrbit].data(), buf_nPosPerBC.data(), (sNBC + 4) * 4);

  } // Orbit

  mVecDigits.resize(eventPosPerOrbit);
  mVecChannelData.resize(chPosOrbit);

  for (int iOrbit = 0; iOrbit < sNorbits; iOrbit++) {
    if (!arrOrbitSizePages[iOrbit]) {
      continue;
    }
    const auto& orbit = arrOrbit[iOrbit];
    const auto& posChDataOrbit = posChDataPerOrbit[iOrbit];
    const auto& ptrPosChDataPerBC = posChDataPerBC[iOrbit].data();
    void* ptrDst = (void*)mVecChannelDataBuf.data();
    for (int iLink = 0; iLink < sNlinksMax; iLink++) {
      if (iLink == mFEEID_TCM) {
        continue;
      }
      const auto& nPages = arrRdhPtrPerOrbit[iOrbit][iLink].size();
      const auto& ptrChDataPosPerLinks = mPosChDataPerLinkOrbit[iOrbit][iLink].data();
      void const* lutPerLink = &mLUT[iLink][0];
      __m512i zmm_lut = _mm512_loadu_si512((void const*)lutPerLink);
      for (int iPage = 0; iPage < nPages; iPage++) {
        const auto& rdhPtr = arrRdhPtrPerOrbit[iOrbit][iLink][iPage];
        const auto& payload = arrDataPerOrbit[iOrbit][iLink][iPage].data();
        const auto& payloadSize = arrDataPerOrbit[iOrbit][iLink][iPage].size();
        const uint8_t* src = (uint8_t*)payload;
        const auto nNGBTwords = payloadSize / 16;
        const int nNGBTwordsDiff = nNGBTwords % 16;
        const int nChunks = nNGBTwords / 16 + static_cast<int>(nNGBTwordsDiff > 0);
        const auto lastChunk = nChunks - 1;
        const uint16_t mask = (0xffff << (16 - nNGBTwordsDiff)) | (0xffff * (nNGBTwordsDiff == 0));

        uint16_t firstBC{0};
        uint8_t nGBTwordPrevChunk{0};
        uint8_t nGBTwordPrevChunkDiff{0};
        bool firstWordIsNotHeader{false};

        __m512i zmm_pos1 = _mm512_set_epi32(0, 16, 32, 48, 64, 80, 96, 112, 128, 144, 160, 176, 192, 208, 224, 240);
        for (int iChunk = 0; iChunk < nChunks; iChunk++) {
          __m512i zmm_mask_charge = _mm512_set1_epi32(0x1fff0000);
          __m512i zmm_mask_PMbits = _mm512_set1_epi32(0xff00);
          __m512i zmm_mask_localChID = _mm512_set1_epi32(0xf);
          __m512i zmm_mask_time = _mm512_set1_epi32(0xfff);

          __m512i zmm_buf, zmm_buf2, zmm_buf3;

          __mmask16 mask16_MaxGBTwords = _mm512_int2mask((0xffff * (iChunk != lastChunk)) | mask);
          __m512i zmm_mask_zero = _mm512_setzero_epi32();
          // Gathering data from page
          __m512i zmm_src_part0 = _mm512_mask_i32gather_epi32(zmm_mask_zero, mask16_MaxGBTwords, zmm_pos1, src, 1);
          __m512i zmm_src_part1 = _mm512_mask_i32gather_epi32(zmm_mask_zero, mask16_MaxGBTwords, zmm_pos1, src + 3, 1);
          __m512i zmm_src_part2 = _mm512_mask_i32gather_epi32(zmm_mask_zero, mask16_MaxGBTwords, zmm_pos1, src + 6, 1);
          // Column 0
          // Time
          __m512i zmm_src_column0_time = _mm512_and_epi32(zmm_src_part0, zmm_mask_time);
          // Charge
          zmm_buf = _mm512_slli_epi32(zmm_src_part0, 4);
          __m512i zmm_src_column0_charge = _mm512_and_epi32(zmm_buf, zmm_mask_charge);
          __m512i zmm_dst_column0_part1 = _mm512_or_epi32(zmm_src_column0_time, zmm_src_column0_charge);
          // PM bits
          zmm_buf = _mm512_slli_epi32(zmm_src_part1, 7);
          __m512i zmm_src_column0_PMbits = _mm512_and_epi32(zmm_buf, zmm_mask_PMbits);
          // ChannelID
          zmm_buf = _mm512_srai_epi32(zmm_src_part1, 12);
          __m512i zmm_dst_column0_chID = _mm512_and_epi32(zmm_buf, zmm_mask_localChID);
          __m512i zmm_dst_column0_globalChID = _mm512_permutexvar_epi32(zmm_dst_column0_chID, zmm_lut);
          __m512i zmm_dst_column0_part0 = _mm512_or_epi32(zmm_dst_column0_globalChID, zmm_src_column0_PMbits);
          // Column 1
          // Time
          zmm_buf = _mm512_srai_epi32(zmm_src_part1, 16);
          __m512i zmm_src_column1_time = _mm512_and_epi32(zmm_buf, zmm_mask_time);
          // Charge
          zmm_buf = _mm512_slli_epi32(zmm_src_part2, 12);
          __m512i zmm_src_column1_charge = _mm512_and_epi32(zmm_buf, zmm_mask_charge);
          __m512i zmm_dst_column1_part1 = _mm512_or_epi32(zmm_src_column1_time, zmm_src_column1_charge);

          // PM bits
          zmm_buf = _mm512_srai_epi32(zmm_src_part2, 9);
          __m512i zmm_src_column1_PMbits = _mm512_and_epi32(zmm_buf, zmm_mask_PMbits);
          // ChannelID
          zmm_buf = _mm512_srai_epi32(zmm_src_part2, 28);
          __m512i zmm_dst_column1_chID = _mm512_and_epi32(zmm_buf, zmm_mask_localChID);
          __m512i zmm_dst_column1_globalChID = _mm512_permutexvar_epi32(zmm_dst_column1_chID, zmm_lut);
          __m512i zmm_dst_column1_part0 = _mm512_or_epi32(zmm_dst_column1_globalChID, zmm_src_column1_PMbits);
          // Preparing masks for data
          //  getting header and nGBTword masks
          __mmask16 mask16_header = _mm512_cmpeq_epi32_mask(zmm_dst_column1_chID, zmm_mask_localChID); // check for header
          zmm_buf = _mm512_srai_epi32(zmm_src_part2, 24);
          __m512i zmm_NGBTwords = _mm512_maskz_and_epi32(mask16_header, zmm_buf, zmm_mask_localChID);
          __mmask16 mask16_header_final = _mm512_mask_cmpgt_epu32_mask(mask16_header, zmm_NGBTwords, zmm_mask_zero);
          __mmask16 mask16_data = _mm512_knot(mask16_header_final);

          // main column wich contains also header descriptor 0xf
          __m512i zmm_mask_maxLocalChID = _mm512_set1_epi32(12);
          __mmask16 mask16_nonzeroChID = _mm512_mask_cmpneq_epi32_mask(mask16_data, zmm_dst_column1_chID, zmm_mask_zero);                  // check for non-zero channelIDs
          __mmask16 mask16_column1_isData = _mm512_mask_cmple_epi32_mask(mask16_nonzeroChID, zmm_dst_column1_chID, zmm_mask_maxLocalChID); // check for max local channel ID - 12
          // first column
          mask16_nonzeroChID = _mm512_mask_cmpneq_epi32_mask(mask16_data, zmm_dst_column0_chID, zmm_mask_zero); // check for non-zero channelIDs
          __mmask16 mask16_column0_isData = _mm512_mask_cmple_epi32_mask(mask16_nonzeroChID, zmm_dst_column0_chID, zmm_mask_maxLocalChID);

          // BC
          __m512i zmm_bc = _mm512_mask_and_epi32(zmm_mask_zero, mask16_header_final, zmm_src_part0, zmm_mask_time);
          // Calculation for GBT word position
          __m512i zmm_mask_seq2 = _mm512_set_epi32(15, 14, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1, 0);

          __m512i zmm_column0_data_seq = _mm512_maskz_expand_epi32(mask16_column0_isData, zmm_mask_seq2);
          __mmask16 mask16_buf = _mm512_kor(mask16_header_final, mask16_column0_isData);
          __m512i zmm_column0_data2header = _mm512_maskz_expand_epi32(mask16_buf, zmm_mask_seq2);

          __m512i zmm_column1_data_seq = _mm512_maskz_expand_epi32(mask16_column1_isData, zmm_mask_seq2);
          mask16_buf = _mm512_kor(mask16_header_final, mask16_column1_isData);
          __m512i zmm_column1_data2header = _mm512_maskz_expand_epi32(mask16_buf, zmm_mask_seq2);

          zmm_column0_data2header = _mm512_maskz_sub_epi32(mask16_column0_isData, zmm_column0_data2header, zmm_column0_data_seq);
          zmm_column1_data2header = _mm512_maskz_sub_epi32(mask16_column1_isData, zmm_column1_data2header, zmm_column1_data_seq);

          __m512i zmm_column0_NGBTwords = _mm512_setzero_epi32();
          __m512i zmm_column1_NGBTwords = zmm_NGBTwords;
          __mmask16 mask16_header_first = _mm512_int2mask(firstWordIsNotHeader * 0b1000000000000000); // fake header, to put metadata(BC and nGBTwords) from previous chunk
          zmm_bc = _mm512_mask_set1_epi32(zmm_bc, mask16_header_first, firstBC);

          zmm_column0_NGBTwords = _mm512_mask_set1_epi32(zmm_column0_NGBTwords, mask16_header_first, nGBTwordPrevChunkDiff);
          zmm_column1_NGBTwords = _mm512_mask_set1_epi32(zmm_column1_NGBTwords, mask16_header_first, nGBTwordPrevChunk + nGBTwordPrevChunkDiff);
          mask16_header_final = _mm512_kor(mask16_header_first, mask16_header_final);
          uint32_t bufBC[16]{}, bufNGBTwords[16]{};

          zmm_buf = _mm512_maskz_compress_epi32(mask16_header_final, zmm_bc);
          zmm_buf2 = _mm512_i32gather_epi32(zmm_buf, ptrChDataPosPerLinks, 4);
          zmm_buf3 = _mm512_i32gather_epi32(zmm_buf, ptrPosChDataPerBC, 4);
          zmm_buf2 = _mm512_add_epi32(zmm_buf2, zmm_buf3);

          __m512i zmm_column0_pos = _mm512_permutexvar_epi32(zmm_column0_data2header, zmm_buf2);
          __m512i zmm_column1_pos = _mm512_permutexvar_epi32(zmm_column1_data2header, zmm_buf2);

          // Column0
          __m512i zmm_mask_seq3 = _mm512_set_epi32(16, 15, 14, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1);
          __m512i zmm_6 = _mm512_set1_epi32(6);
          __m512i zmm_2 = _mm512_set1_epi32(2);

          zmm_buf2 = _mm512_add_epi32(zmm_mask_seq2, zmm_column0_NGBTwords);
          zmm_buf = _mm512_maskz_compress_epi32(mask16_header_final, zmm_buf2);

          zmm_buf2 = _mm512_permutexvar_epi32(zmm_column0_data2header, zmm_buf);
          zmm_buf = _mm512_maskz_sub_epi32(mask16_column0_isData, zmm_buf2, zmm_mask_seq3);

          zmm_column0_pos = _mm512_maskz_add_epi32(mask16_column0_isData, zmm_column0_pos, zmm_buf);

          __m512i zmm_column0_part0_pos = _mm512_maskz_mullo_epi32(mask16_column0_isData, zmm_column0_pos, zmm_6);
          __m512i zmm_column0_part1_pos = _mm512_maskz_add_epi32(mask16_column0_isData, zmm_column0_part0_pos, zmm_2);

          // Column1
          zmm_buf2 = _mm512_add_epi32(zmm_mask_seq2, zmm_column1_NGBTwords);
          zmm_buf = _mm512_maskz_compress_epi32(mask16_header_final, zmm_buf2);

          zmm_buf2 = _mm512_permutexvar_epi32(zmm_column1_data2header, zmm_buf);
          zmm_buf = _mm512_maskz_sub_epi32(mask16_column1_isData, zmm_buf2, zmm_mask_seq3);

          zmm_column1_pos = _mm512_maskz_add_epi32(mask16_column1_isData, zmm_column1_pos, zmm_buf);
          __m512i zmm_column1_part0_pos = _mm512_maskz_mullo_epi32(mask16_column1_isData, zmm_column1_pos, zmm_6); // Todo: exclude multilication, on fly calculation for byte position?
          __m512i zmm_column1_part1_pos = _mm512_maskz_add_epi32(mask16_column1_isData, zmm_column1_part0_pos, zmm_2);
          // Pushing data
          _mm512_mask_i32scatter_epi32(ptrDst, mask16_column0_isData, zmm_column0_part0_pos, zmm_dst_column0_part0, 1);
          _mm512_mask_i32scatter_epi32(ptrDst, mask16_column1_isData, zmm_column1_part0_pos, zmm_dst_column1_part0, 1);

          _mm512_mask_i32scatter_epi32(ptrDst, mask16_column0_isData, zmm_column0_part1_pos, zmm_dst_column0_part1, 1);
          _mm512_mask_i32scatter_epi32(ptrDst, mask16_column1_isData, zmm_column1_part1_pos, zmm_dst_column1_part1, 1);

          // Getting last header position
          _mm512_storeu_si512(bufBC, zmm_bc);
          _mm512_storeu_si512(bufNGBTwords, zmm_NGBTwords);

          const uint32_t header32 = _cvtmask16_u32(mask16_header_final);
          const uint16_t lastHeaderPos = (__builtin_ctz(header32)) * (header32 > 0);
          firstBC = bufBC[lastHeaderPos];
          nGBTwordPrevChunk = bufNGBTwords[lastHeaderPos];
          nGBTwordPrevChunkDiff = lastHeaderPos;
          firstWordIsNotHeader = nGBTwordPrevChunk != lastHeaderPos;
          nGBTwordPrevChunkDiff++;

          zmm_buf = _mm512_set1_epi32(256);
          zmm_pos1 = _mm512_add_epi32(zmm_buf, zmm_pos1);
        } // chunk
      }   // page
    }     // link
    memcpy(&mVecChannelData[posChDataOrbit], mVecChannelDataBuf.data(), 6 * nChDataPerOrbit[iOrbit]);
  } // orbit
  if (mEnableEmptyTFprotection && mVecDigits.size() == 0) {
    // In case of empty payload within TF, there will be inly single dummy object in ChannelData container.
    // Due to empty Digit container this dummy object will never participate in any further tasks.
    mVecChannelData.emplace_back();
  }
  pc.outputs().snapshot(o2::framework::Output{o2::header::gDataOriginFT0, "DIGITSBC", 0, o2::framework::Lifetime::Timeframe}, mVecDigits);
  pc.outputs().snapshot(o2::framework::Output{o2::header::gDataOriginFT0, "DIGITSCH", 0, o2::framework::Lifetime::Timeframe}, mVecChannelData);
  auto t2 = std::chrono::high_resolution_clock::now();
  auto delay = std::chrono::duration_cast<std::chrono::microseconds>(t2 - t1);
  LOG(debug) << "Decoder delay: " << delay.count();
}
} // namespace ft0
} // namespace o2
