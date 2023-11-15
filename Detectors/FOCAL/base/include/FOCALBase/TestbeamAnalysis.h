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
#ifndef ALICEO2_FOCAL_TESTBEAMANALYSIS_H
#define ALICEO2_FOCAL_TESTBEAMANALYSIS_H

#include <memory>
#include <string>
#include <string_view>
#include <TFile.h>
#include <DataFormatsFOCAL/Event.h>
#include <FOCALBase/EventReader.h>

namespace o2::focal
{

class TestbeamAnalysis
{
 public:
  TestbeamAnalysis() = default;
  virtual ~TestbeamAnalysis() = default;

  void run();
  void setInputFile(const std::string_view inputfile) { mInputFilename = inputfile.data(); }
  void setVerbose(bool doVerbose) { mVerbose = doVerbose; }

 protected:
  virtual void init() {}
  virtual void process(const Event& event) {}
  virtual void terminate() {}

  int getCurrentEventNumber() const { return mCurrentEventNumber; }

 private:
  std::unique_ptr<EventReader> mEventReader;
  std::unique_ptr<TFile> mCurrentFile;
  std::string mInputFilename;
  int mCurrentEventNumber = 0;
  bool mVerbose = false;

  ClassDefNV(TestbeamAnalysis, 1);
};

} // namespace o2::focal
#endif // ALICEO2_FOCAL_TESTBEAMANALYSIS_H