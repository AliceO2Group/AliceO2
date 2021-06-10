// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

#ifndef O2_MCH_RAW_PAYLOAD_PAGINATOR_H
#define O2_MCH_RAW_PAYLOAD_PAGINATOR_H

#include <string>
#include <gsl/span>
#include <functional>
#include <optional>
#include <set>
#include <iostream>

namespace o2::raw
{
class RawFileWriter;
}

namespace o2::mch::raw
{

// helper struct with the smallest information to uniquely identify
// one data link
struct LinkInfo {
  uint16_t feeId;
  uint16_t cruId;
  uint8_t linkId;
  uint8_t endPoint;
};

bool operator<(const LinkInfo&, const LinkInfo&);
std::ostream& operator<<(std::ostream& os, const LinkInfo& li);

using Solar2LinkInfo = std::function<std::optional<LinkInfo>(uint16_t)>;

/** Creates a function that is able to convert a solarId into a LinkInfo */
template <typename ELECMAP, typename FORMAT, typename CHARGESUM, int VERSION>
Solar2LinkInfo createSolar2LinkInfo();

void registerLinks(o2::raw::RawFileWriter& rawFileWriter,
                   std::string outputBase,
                   const std::set<LinkInfo>& links,
                   bool filePerLink);

void paginate(o2::raw::RawFileWriter& rawFileWriter,
              gsl::span<const std::byte> buffer,
              const std::set<LinkInfo>& links,
              Solar2LinkInfo solar2LinkInfo);

} // namespace o2::mch::raw
#endif
