// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.
#ifndef FRAMEWORK_DATAREF_H
#define FRAMEWORK_DATAREF_H

namespace o2
{
namespace framework
{

struct InputSpec;

struct DataRef {
  // FIXME: had to remove the second 'const' in const T* const
  // to allow assignment
  const InputSpec* spec;
  const char* header;
  const char* payload;
};

} // namespace framework
} // namespace o2

#endif
