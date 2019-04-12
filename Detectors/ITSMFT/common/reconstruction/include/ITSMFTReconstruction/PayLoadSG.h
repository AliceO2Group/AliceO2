// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

#ifndef ALICEO2_ITSMFT_PAYLOADSG_H
#define ALICEO2_ITSMFT_PAYLOADSG_H

#include <cstdint>
#include <vector>
#include <Rtypes.h>

/// \file PayLoadSG.h
/// \brief Declaration of class for scatter-gather buffer of ALPIDE data
/// \author ruben.shahoyan@cern.ch

namespace o2
{
namespace itsmft
{

class PayLoadSG
{
  // scatter-gather buffer for the payload: base pointer + vector of references for pieces to collect

 public:
  PayLoadSG() = default;
  ~PayLoadSG() = default;

  ///< add n bytes to the buffer
  void add(const uint8_t* ptr, size_t n)
  {
    if (n) {
      mBuffer.emplace_back(ptr, n);
    }
  }

  ///< read current character value from buffer w/o stepping forward
  bool current(uint8_t& v)
  {
    if (mCurrPiece < mBuffer.size()) {
      const auto& piece = mBuffer[mCurrPiece];
      if (mCurrEntryInPiece < piece.sz) {
        v = piece.start[mCurrEntryInPiece];
        return true;
      } else {
        nextPiece();
        return current(v);
      }
    }
    return false;
  }

  ///< read character value from buffer
  bool next(uint8_t& v)
  {
    if (mCurrPiece < mBuffer.size()) {
      const auto& piece = mBuffer[mCurrPiece];
      if (mCurrEntryInPiece < piece.sz) {
        v = piece.start[mCurrEntryInPiece++];
        return true;
      } else {
        nextPiece();
        return next(v);
      }
    }
    return false;
  }

  ///< read short value from buffer
  bool next(uint16_t& v)
  {
    uint8_t b0, b1;
    if (next(b0) && next(b1)) {
      v = (b0 << 8) | b1;
      return true;
    }
    return false;
  }

  ///< move current pointer to the head
  void rewind()
  {
    mCurrPiece = mCurrEntryInPiece = 0;
  }

  ///< make buffer empty
  void clear()
  {
    mBuffer.clear();
    mCurrPiece = mCurrEntryInPiece = 0;
  }

  struct SGPiece {
    const uint8_t* start = nullptr; // start of the piece
    uint32_t sz = 0;                // size of the piece
    SGPiece() = default;
    SGPiece(const uint8_t* st, int n) : start(st), sz(n) {}
  };

 private:
  void nextPiece()
  {
    // move to the next piece
    mCurrPiece++;
    mCurrEntryInPiece = 0;
  }

  std::vector<SGPiece> mBuffer;   // list of pieces to fetch
  uint32_t mCurrPiece = 0;        // current piece
  uint32_t mCurrEntryInPiece = 0; // offset within current piece

  ClassDefNV(PayLoadSG, 1);
};
} // namespace itsmft
} // namespace o2

#endif
