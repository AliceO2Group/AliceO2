// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

/// @since 2018-02-19
/// @author P. Pillot
/// @brief Device reading the digits as input, running the preclusterizer and sending the preclusters as output

#ifndef ALICEO2_MCH_PRECLUSTERFINDERDEVICE_H_
#define ALICEO2_MCH_PRECLUSTERFINDERDEVICE_H_

#include <FairMQDevice.h>

#include <cstdint>

#include "MCHBase/PreClusterBlock.h"
#include "PreClusterFinder.h"
#include "aliceHLTwrapper/AliHLTDataTypes.h"
#include "aliceHLTwrapper/MessageFormat.h"

namespace o2
{
namespace mch
{

class PreClusterFinderDevice : public FairMQDevice
{
 public:
  PreClusterFinderDevice();
  ~PreClusterFinderDevice() override = default;

  PreClusterFinderDevice(const PreClusterFinderDevice&) = delete;
  PreClusterFinderDevice& operator=(const PreClusterFinderDevice&) = delete;
  PreClusterFinderDevice(PreClusterFinderDevice&&) = delete;
  PreClusterFinderDevice& operator=(PreClusterFinderDevice&&) = delete;

 protected:
  void InitTask() override;
  void ResetTask() override;

 private:
  bool processData(FairMQMessagePtr& msg, int /*index*/);

  void fillBlockData(AliHLTComponentBlockData& blockData);
  void storePreClusters(uint8_t* buffer, uint32_t size);

  o2::alice_hlt::MessageFormat mFormatHandler{}; /// HLT format handler

  PreClusterFinder mPreClusterFinder{}; /// preclusterizer

  PreClusterBlock mPreClusterBlock{}; ///< to fill preclusters data blocks
};

} // namespace mch
} // namespace o2

#endif // ALICEO2_MCH_PRECLUSTERFINDERDEVICE_H_
