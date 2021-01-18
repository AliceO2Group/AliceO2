// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

///
/// \file   HmpidDecodeRawMem.h
/// \author Antonio Franco - INFN Bari
/// \brief Derived Class for decoding Raw Data Memory stream
/// \version 1.0
/// \date 24 set 2020

#ifndef COMMON_HMPIDDECODERAWMEM_H_
#define COMMON_HMPIDDECODERAWMEM_H_

#include <ctype.h>
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <string.h>
#include <unistd.h>
#include <vector.h>

#include "HMPIDBase/Digit.h"
#include "HMPIDBase/Geo.h"
#include "HmpidDecoder.h"

namespace o2{
  namespace hmpid {


class HmpidDecodeRawMem: public HmpidDecoder
{
  public:
    HmpidDecodeRawMem(int *EqIds, int *CruIds, int *LinkIds, int numOfEquipments);
    HmpidDecodeRawMem(int numOfEquipments);
    ~HmpidDecodeRawMem();

    bool setUpStream(void *Buffer, long BufferLen);

  private:
    bool getBlockFromStream(uint32_t **streamPtr, uint32_t Size);
    bool getHeaderFromStream(uint32_t **streamPtr);
    bool getWordFromStream(uint32_t *word);
    void setPad(HmpidEquipment *eq, int col, int dil, int ch, int charge);

  private:

};

class HmpidDecodeRawDigit: public HmpidDecodeRawMem
{
  public:
    HmpidDecodeRawDigit(int *EqIds, int *CruIds, int *LinkIds, int numOfEquipments);
    HmpidDecodeRawDigit(int numOfEquipments);
    ~HmpidDecodeRawDigit();

    vector<o2::hmpid::Digit> mDigits;

  private:
    void setPad(HmpidEquipment *eq, int col, int dil, int ch, int charge);


};

}
}
#endif /* COMMON_HMPIDDECODERAWFILE_H_ */
