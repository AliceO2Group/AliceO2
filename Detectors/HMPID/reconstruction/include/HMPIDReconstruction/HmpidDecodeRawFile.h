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
/// \file   HmpidDecodeRawFile.h
/// \author Antonio Franco - INFN Bari
/// \brief Derived Class for decoding Raw Data File stream
/// \version 1.0
/// \date 24 set 2020

#ifndef COMMON_HMPIDDECODERAWFILE_H_
#define COMMON_HMPIDDECODERAWFILE_H_

#include <ctype.h>
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <string.h>
#include <unistd.h>

#include "HMPIDReconstruction/HmpidDecoder.h"

#define MAXFILENAMEBUFFER 512
#define MAXRAWFILEBUFFER RAWBLOCKDIMENSION_W * 4 + 8

namespace o2
{
namespace hmpid
{

class HmpidDecodeRawFile : public HmpidDecoder
{
 public:
  HmpidDecodeRawFile(int* EqIds, int* CruIds, int* LinkIds, int numOfEquipments);
  HmpidDecodeRawFile(int numOfEquipments);
  ~HmpidDecodeRawFile();

  bool setUpStream(void* InpuFileName, long Size);

 private:
  bool getBlockFromStream(uint32_t** streamPtr, uint32_t Size);
  bool getHeaderFromStream(uint32_t** streamPtr);
  bool getWordFromStream(uint32_t* word);
  int fileExists(char* filewithpath);
  void setPad(HmpidEquipment* eq, int col, int dil, int ch, uint16_t charge);

 private:
  FILE* fh;
  char mInputFile[MAXFILENAMEBUFFER];
  uint32_t mFileBuffer[MAXRAWFILEBUFFER];
};

} // namespace hmpid
} // namespace o2
#endif /* COMMON_HMPIDDECODERAWFILE_H_ */
