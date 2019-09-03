// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.
#include <fstream>
#include <iostream>
#include <cstring>
#include "EMCALReconstruction/RawBuffer.h"

using namespace o2::emcal;

void RawBuffer::flush()
{
  mCurrentDataWord = 0;
  mNDataWords = 0;
  memset(mDataWords.data(), 0, sizeof(uint32_t) * mDataWords.size());
}

void RawBuffer::readFromStream(std::istream& in, uint32_t payloadsize)
{
  flush();
  uint32_t word(0);
  auto address = reinterpret_cast<char*>(&word);
  int nbyte = 0;
  while (nbyte < payloadsize) {
    in.read(address, sizeof(word));
    nbyte += sizeof(word);
    if ((word & 0xFFF) == 0x082) {
      // Termination word
      // should normally not be decoded in case the payload size
      // is determined correctly
      break;
    }
    mDataWords[mNDataWords++] = word;
  }
}

uint32_t RawBuffer::getWord(int index) const
{
  if (index >= mNDataWords)
    throw std::runtime_error("Index out of range");
  return mDataWords[index];
}

uint32_t RawBuffer::getNextDataWord()
{
  if (!hasNext())
    throw std::runtime_error("No more data words in buffer");
  return mDataWords[mCurrentDataWord++];
}
