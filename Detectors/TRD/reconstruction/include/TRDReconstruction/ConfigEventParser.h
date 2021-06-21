// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

/// @file   ConfigEventReader.h
/// @author Sean Murray
/// @brief  TRD cru output data to tracklet task

#ifndef O2_TRD_CONFIGEVENTREADER
#define O2_TRD_CONFIGEVENTREADER

#include "Framework/Task.h"
#include "Framework/DataProcessorSpec.h"
#include "TRDReconstruction/CruRawReader.h"
#include "TRDReconstruction/CompressedRawReader.h"
#include "DataFormatsTRD/RawData.h"
#include "DataFormatsTRD/TriggerRecord.h"
#include "DataFormatsTRD/Constants.h"
#include "DataFormatsTRD/CompressedDigit.h"
#include <fstream>

namespace o2::trd
{

// class to Parse a single link of digits data.
// calling class splits data by link and this gets called per link.

class ConfigEventParser
{

 public:
  ConfigEventParser() = default;
  ~ConfigEventParser() = default;
  bool getVerbose() { return mVerbose; }
  void setVerbose(bool value, bool header, bool data)
  {
    mVerbose = value;
    mHeaderVerbose = header;
    mDataVerbose = data;
  }

 private:
  int mState;
  int mDataWordsParsed; // count of data wordsin data that have been parsed in current call to parse.
  int mBufferLocation;
  bool mDataVerbose{false};
  bool mHeaderVerbose{false};
  bool mVerbose{false};
};

} // namespace o2::trd

#endif // O2_TRD_CONFIGEVENTREADER
