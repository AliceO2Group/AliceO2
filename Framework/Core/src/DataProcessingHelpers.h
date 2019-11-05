// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.
#ifndef O2_FRAMEWORK_DATAPROCESSINGHELPERS_H_
#define O2_FRAMEWORK_DATAPROCESSINGHELPERS_H_

class FairMQDevice;

namespace o2::framework
{

struct OutputChannelSpec;

/// Generic helpers for DataProcessing releated functions.
struct DataProcessingHelpers {
  /// Send EndOfStream message to a given channel
  /// @param device the FairMQDevice which needs to send the EndOfStream message
  /// @param channel the OutputChannelSpec of the channel which needs to be signaled
  ///        for EndOfStream
  static void sendEndOfStream(FairMQDevice& device, OutputChannelSpec const& channel);
};

} // namespace o2::framework
#endif // O2_FRAMEWORK_DATAPROCESSINGHELPERS_H_
