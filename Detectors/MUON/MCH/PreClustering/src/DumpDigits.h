// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

/// @since 2016-10-18
/// @author P. Pillot
/// @brief Simple device to print the digits

#ifndef ALICEO2_MCH_DUMPDIGITS_H_
#define ALICEO2_MCH_DUMPDIGITS_H_

#include <FairMQDevice.h>

#include "aliceHLTwrapper/MessageFormat.h"

namespace o2
{
namespace mch
{

class DumpDigits : public FairMQDevice
{
 public:
  DumpDigits();
  ~DumpDigits() override = default;

  DumpDigits(const DumpDigits&) = delete;
  DumpDigits& operator=(const DumpDigits&) = delete;
  DumpDigits(DumpDigits&&) = delete;
  DumpDigits& operator=(DumpDigits&&) = delete;

 protected:
  bool handleData(FairMQMessagePtr& msg, int /*index*/);

 private:
  o2::alice_hlt::MessageFormat mFormatHandler{};
};

} // namespace mch
} // namespace o2

#endif // ALICEO2_MCH_DUMPDIGITS_H_
