// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

/// \file SimpleRawReader.h
/// \brief Definition of the simple reader for non-DPL tests
#ifndef ALICEO2_ITSMFT_SIMPLERAWREADER_H_
#define ALICEO2_ITSMFT_SIMPLERAWREADER_H_

#include "DetectorsRaw/RawFileReader.h"

namespace o2
{
namespace framework
{
class InputRecord;
}
namespace raw
{
struct SimpleSTF;

/// Class to read raw data from the file and to create a STF-like structure which can be navigated using e.g. DPLRawParser
/// but which can be used in the standalone code not using exchange over the DPL.
/// Simple reader for non-DPL tests
class SimpleRawReader
{

 public:
  SimpleRawReader();
  SimpleRawReader(const std::string& cfg, bool tfPerMessage = false, int loop = 1);
  ~SimpleRawReader();
  void init();
  bool loadNextTF();
  int getNLinks() const { return mReader ? mReader->getNLinks() : 0; }
  bool isDone() const { return mDone; }
  void setTFPerMessage(bool v = true) { mHBFPerMessage = !v; }
  void setConfigFileName(const std::string& fn) { mCFGName = fn; }
  void printStat() const;

  SimpleSTF* getSimpleSTF();
  o2::framework::InputRecord* getInputRecord();

 private:
  int mLoop = 0;              // once last TF reached, loop while mLoop>=0
  bool mHBFPerMessage = true; // true: send TF as multipart of HBFs, false: single message per TF
  bool mDone = false;
  std::string mCFGName{};
  std::unique_ptr<o2::raw::RawFileReader> mReader;
  size_t mLoopsDone = 0, mSentSize = 0, mSentMessages = 0, mTFIDaccum = 0; // statistics

  std::unique_ptr<SimpleSTF> mSTF;

  ClassDefNV(SimpleRawReader, 1);
};

} // namespace itsmft
} // namespace o2

#endif
