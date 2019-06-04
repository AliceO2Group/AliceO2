// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

#include "Framework/InputSpec.h"
#include "Framework/OutputSpec.h"
#include "Framework/DataSpecUtils.h"

#include <ostream>
#include <string>

namespace o2
{
namespace framework
{

std::ostream& operator<<(std::ostream& stream, o2::framework::InputSpec const& arg)
{
  // FIXME: should have stream operators for the header fields
  stream << arg.binding << " {" << DataSpecUtils::describe(arg) << "}";
  return stream;
}

std::ostream& operator<<(std::ostream& stream, o2::framework::OutputSpec const& arg)
{
  stream << arg.binding.value << " { " << DataSpecUtils::describe(arg) << "}";
  return stream;
}

} // namespace framework
} // namespace o2
