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
#ifndef ALICEO2_FOCAL_EVENTREADER_H
#define ALICEO2_FOCAL_EVENTREADER_H

#include <memory>
#include <vector>
#include <TTreeReader.h>
#include <DataFormatsFOCAL/Event.h>
#include <DataFormatsFOCAL/PixelHit.h>
#include <DataFormatsFOCAL/PixelChipRecord.h>
#include <DataFormatsFOCAL/TriggerRecord.h>

namespace o2::focal
{

class EventReader
{
 public:
  EventReader() = default;
  EventReader(TTree* eventTree);
  ~EventReader() = default;

  bool hasNext() const;
  Event readNextEvent();

 private:
  void init(TTree* eventTree);
  void nextTimeframe();

  bool mHasEntryLoaded = false;
  bool mTreeReaderHasNext = true;
  int mEntryInTF = 0;
  std::unique_ptr<TTreeReader> mTreeReader;
  std::unique_ptr<TTreeReaderValue<std::vector<PadLayerEvent>>> mPadBranch;
  std::unique_ptr<TTreeReaderValue<std::vector<PixelHit>>> mPixelHitBranch;
  std::unique_ptr<TTreeReaderValue<std::vector<PixelChipRecord>>> mPixelChipBranch;
  std::unique_ptr<TTreeReaderValue<std::vector<TriggerRecord>>> mTriggerBranch;

  ClassDefNV(EventReader, 1);
};

} // namespace o2::focal

#endif // ALICEO2_FOCAL_EVENTREADER_H