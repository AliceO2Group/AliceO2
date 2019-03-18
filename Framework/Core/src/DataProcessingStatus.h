// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.
#ifndef o2_framework_DataProcessingStatus_H_INCLUDED
#define o2_framework_DataProcessingStatus_H_INCLUDED

namespace o2
{
namespace framework
{

enum DataProcessingStatus {
  ID = 0,
  IN_DPL_OVERHEAD = 0,
  IN_DPL_USER_CALLBACK = 1,
  IN_DPL_ERROR_CALLBACK = 2
};

} // namespace framework
} // namespace o2

#endif // o2_framework_DataProcessingStatus_H_INCLUDED
