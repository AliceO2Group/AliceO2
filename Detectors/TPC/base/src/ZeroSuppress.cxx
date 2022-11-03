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

/// \file ZeroSuppress.cxx
/// \brief Class for the TPC zero suppressed data format

#include <cmath>
#include <iostream>
#include <array>

#include "TPCBase/ZeroSuppress.h"
#include "CommonConstants/LHCConstants.h"
#include "DetectorsRaw/RDHUtils.h"

#include "DataFormatsTPC/Digit.h"
#include "TPCBase/Mapper.h"
#include "TPCBase/CRU.h"

#include "DataFormatsTPC/Defs.h"
#include "DataFormatsTPC/Helpers.h"

using namespace o2::tpc;
using namespace o2::tpc::constants;

void ZeroSuppress::process()
{
  std::cout << "Zero Suppress process call" << std::endl;
}

void ZeroSuppress::DecodeZSPages(gsl::span<const ZeroSuppressedContainer8kb>* z0in, std::vector<Digit>* outDigits, int firstHBF)
{
  std::vector<Digit>& digits = *outDigits;
  std::vector<unsigned int> _ADC = {};
  _ADC.resize(TPCZSHDR::MAX_DIGITS_IN_PAGE);
  gsl::span<const ZeroSuppressedContainer8kb>& z0input = *z0in;

  for (auto inputPage : z0input) {
    unsigned char* pageStart = reinterpret_cast<unsigned char*>(&inputPage);
    unsigned char* startPtr = pageStart;
    unsigned char* nextPage = pageStart + TPCZSHDR::TPC_ZS_PAGE_SIZE;

    ZeroSuppressedContainer* z0Container = reinterpret_cast<ZeroSuppressedContainer*>(startPtr);
    unsigned int _adcBits = (z0Container->hdr.version == 1) ? 10 : ((z0Container->hdr.version == 2) ? 12 : 0);
    if (!_adcBits) {
      if ((int)z0Container->hdr.nADCsamples) {
        std::cout << "unsupported zero suppressed version" << std::endl;
        break;
      } else {
        continue;
      }
    }
    uint16_t _CRUID = z0Container->hdr.cruID;
    uint16_t _timeOffset = z0Container->hdr.timeOffset;
    unsigned int mask = static_cast<unsigned int>(pow(2, _adcBits)) - 1;
    float decodeFactor = 1.0 / (1 << (_adcBits - 10));

    const o2::header::RAWDataHeader* rdh = (const o2::header::RAWDataHeader*)&inputPage;
    auto orbit = o2::raw::RDHUtils::getHeartBeatOrbit(rdh);
    int timeBin = (_timeOffset + (o2::raw::RDHUtils::getHeartBeatOrbit(rdh) - firstHBF) * o2::constants::lhc::LHCMaxBunches) / LHCBCPERTIMEBIN;

    unsigned int ct = 0;
    startPtr += (sizeof(z0Container->rdh) + sizeof(z0Container->hdr)); // move to first time bin
    // iterate through all time bins indicated in the z0 header
    for (int tb = 0; tb < z0Container->hdr.nTimeBinSpan; tb++) {
      startPtr += (startPtr - pageStart) % 2;
      TPCZSTBHDR* tbHdr = reinterpret_cast<TPCZSTBHDR*>(startPtr);
      unsigned int numberRows = __builtin_popcount((tbHdr->rowMask & 0x7FFF)); // number of ones, excluding upper/lower bit
      if (startPtr > nextPage) {
        throw std::runtime_error("pointer for time bin outside current zs page");
      }
      unsigned int rowUpperOffset = (tbHdr->rowMask & (1 << 15)) ? (mapper.getNumberOfRowsRegion(_CRUID % 10) / 2) : 0;
      unsigned int _timeBin = timeBin + tb;

      //if rows in timebin
      if ((numberRows != 0)) {
        startPtr += 2 * numberRows;
        for (unsigned int pos = 0; pos < 15; pos++) { // TODO: end iterations at max number of rows for that endpoint
          if (tbHdr->rowMask & (1 << pos)) {          //row bit set decode row
            unsigned int _row = pos + rowUpperOffset;
            uint8_t numberSequences = *startPtr;
            uint8_t numberADCs = *(startPtr + (2 * numberSequences));

            unsigned char* _adcstart = startPtr + 2 * numberSequences + 1;
            unsigned int length = (numberADCs * _adcBits + 7) / 8;
            unsigned int temp = 0;
            unsigned int t = 0;
            unsigned int shift = 0;
            unsigned int a1 = 0;
            unsigned int count = 0;
            if (_adcstart + length > nextPage) {
              throw std::runtime_error("pointer for current sequence outside current zs page");
            }
            for (unsigned int a = 0; a < length; a++) {
              temp = (*_adcstart) << shift;
              _adcstart++;
              t |= temp;
              shift += 8;
              while (shift >= _adcBits) {
                a1 = t & mask;
                _ADC[count] = a1;
                count++;
                shift -= _adcBits;
                t >>= _adcBits;
              }
            }
            for (unsigned int nADC = 0; nADC < numberADCs; nADC++) {
              for (unsigned int seq = 0; seq < numberSequences; seq++) {
                unsigned int seqLength = seq ? startPtr[2 * (seq + 1)] - startPtr[2 * seq] : startPtr[2];
                for (unsigned int s = 0; s < seqLength; s++) {
                  digits.emplace_back(Digit((int)_CRUID, (float)(_ADC[nADC] * decodeFactor), (int)(_row + mapper.getGlobalRowOffsetRegion(_CRUID % 10)), (int)(*(startPtr + 2 * seq + 1) + s), _timeBin));
                  nADC++;
                }
              }
            }
            startPtr += (numberSequences * 2) + length + 1;
          }
        }

      } else { // pointer + empty tb hdr
        startPtr += 2;
        continue;
      }
    }
  }
}
