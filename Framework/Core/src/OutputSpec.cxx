// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

#include "Framework/OutputSpec.h"
#include "Framework/Lifetime.h"

namespace o2::framework
{
OutputSpec::OutputSpec(OutputLabel const& inBinding, header::DataOrigin inOrigin, header::DataDescription inDescription,
                       header::DataHeader::SubSpecificationType inSubSpec, enum Lifetime inLifetime)
  : binding{ inBinding },
    origin{ inOrigin },
    description{ inDescription },
    subSpec{ inSubSpec },
    lifetime{ inLifetime }
{
}

OutputSpec::OutputSpec(header::DataOrigin inOrigin, header::DataDescription inDescription,
                       header::DataHeader::SubSpecificationType inSubSpec, enum Lifetime inLifetime)
  : binding{ OutputLabel{ "" } },
    origin{ inOrigin },
    description{ inDescription },
    subSpec{ inSubSpec },
    lifetime{ inLifetime }
{
}

OutputSpec::OutputSpec(OutputLabel const& inBinding, header::DataOrigin inOrigin, header::DataDescription inDescription,
                       enum Lifetime inLifetime)
  : binding{ inBinding }, origin{ inOrigin }, description{ inDescription }, subSpec{ 0 }, lifetime{ inLifetime }
{
}

OutputSpec::OutputSpec(header::DataOrigin inOrigin, header::DataDescription inDescription,
                       enum Lifetime inLifetime)
  : binding{ OutputLabel{ "" } },
    origin{ inOrigin },
    description{ inDescription },
    subSpec{ 0 },
    lifetime{ inLifetime }
{
}

bool OutputSpec::operator==(OutputSpec const& that) const
{
  return origin == that.origin && description == that.description && subSpec == that.subSpec &&
         lifetime == that.lifetime;
};

} // namespace o2::framework
