// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

#ifndef FILE_PATH_UTILS_H_
#define FILE_PATH_UTILS_H_

namespace o2
{
namespace DataDistribution
{

////////////////////////////////////////////////////////////////////////////////
/// FilePathUtils class
////////////////////////////////////////////////////////////////////////////////

class FilePathUtils
{
 public:
  FilePathUtils() = delete;

  static std::string getNextSeqName(const std::string& pRoot);
};
}
} /* o2::DataDistribution */

#endif /* FILE_PATH_UTILS_H_ */
