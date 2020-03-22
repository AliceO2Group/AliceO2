// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

#include "EMCALSimulation/DMAOutputStream.h"

using namespace o2::emcal;

DMAOutputStream::DMAOutputStream(const char* filename) : mFilename(filename) {}

DMAOutputStream::~DMAOutputStream()
{
  if (mOutputFile.is_open())
    mOutputFile.close();
}

void DMAOutputStream::open()
{
  if (!mOutputFile.is_open()) {
    if (!mFilename.length())
      throw OutputFileException(mFilename);
    mOutputFile.open(mFilename, std::ios::out | std::ios::binary);
    mInitialized = true;
  }
}

void DMAOutputStream::writeSingleHeader(const RawHeader& header)
{
  if (!mInitialized)
    open();
  std::vector<char> outputpage(sizeof(RawHeader));
  RawHeader* outputheader = reinterpret_cast<RawHeader*>(outputpage.data());
  *outputheader = header;
  outputheader->memorySize = sizeof(RawHeader);
  outputheader->offsetToNext = sizeof(RawHeader);
  mOutputFile.write(outputpage.data(), outputpage.size());
}

int DMAOutputStream::writeData(RawHeader header, gsl::span<const char> buffer)
{
  if (!mInitialized)
    open();

  constexpr int PAGESIZE = 8192;
  // Handling of the termination word (0x001d3082): The termination word is added to the payload
  // but not included in the payload size (as done in the hardware). Therefore it has to be subtracted
  // from the maximum possible payload size
  constexpr int MAXNWORDS = PAGESIZE - sizeof(header) - sizeof(uint32_t);
  bool writeNext = true;
  int pagecounter = header.pageCnt, currentindex = 0;
  while (writeNext) {
    int sizeRemain = buffer.size() - currentindex;
    int nwordspage = MAXNWORDS;
    if (sizeRemain < MAXNWORDS) {
      // Last page
      nwordspage = sizeRemain;
      writeNext = false;
    }
    header.pageCnt = pagecounter;
    header.memorySize = nwordspage + sizeof(RawHeader);
    header.offsetToNext = 8192;

    writeDMAPage(header, gsl::span(buffer.data() + currentindex, nwordspage), PAGESIZE);

    if (writeNext) {
      currentindex += nwordspage;
    }
    pagecounter++;
  }
  return pagecounter;
}

void DMAOutputStream::writeDMAPage(const RawHeader& header, gsl::span<const char> payload, int pagesize)
{
  std::vector<char> dmapage(pagesize);
  RawHeader* outheader = reinterpret_cast<RawHeader*>(dmapage.data());
  *outheader = header;
  char* outpayload = dmapage.data() + sizeof(header);
  memcpy(outpayload, payload.data(), payload.size());
  // Write termination character
  uint32_t terminationCharacter = 0x001d3082;
  char* terminationString = reinterpret_cast<char*>(&terminationCharacter);
  memcpy(outpayload + payload.size(), terminationString, sizeof(uint32_t));
  mOutputFile.write(dmapage.data(), dmapage.size());
}