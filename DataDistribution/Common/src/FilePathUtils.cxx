// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

#include <boost/filesystem.hpp>
#include <boost/range/iterator_range.hpp>
#include <regex>

#include <iostream>
#include <sstream>
#include <iomanip>

#include "Common/FilePathUtils.h"

namespace o2
{
namespace DataDistribution
{

std::string FilePathUtils::getNextSeqName(const std::string& pRootDir)
{
  static const std::regex seq_regex("(\\d+)(?=\\D*$)", std::regex::icase);

  namespace fsb = boost::filesystem;

  fsb::path lRootPath(pRootDir);

  // check if root directory exists
  if (!fsb::is_directory(lRootPath)) {
    using namespace std::string_literals;
    throw std::invalid_argument("'"s + pRootDir + "' is not a directory"s);
  }

  // try to match the elements
  std::string lNameMatch;
  std::uint64_t lMaxSeq = 0;
  std::size_t lLen = 1;
  std::string lPrefix, lSuffix;

  for (auto& entry : boost::make_iterator_range(fsb::directory_iterator(lRootPath), {})) {
    std::smatch result;
    const std::string lBaseName = entry.path().filename().string();
    if (std::regex_search(lBaseName, result, seq_regex)) {

      const std::uint64_t lCurrSeq = std::stoull(result[1]) + 1;
      if (lCurrSeq >= lMaxSeq) {
        lMaxSeq = lCurrSeq;
        lLen = std::max(lLen, std::size_t(result[1].length()));
        lNameMatch = lBaseName;
        lPrefix = result.prefix().str();
        lSuffix = result.suffix().str();
      }
    }
  }

  // make sure length is big enough
  lLen = std::max(lLen, std::to_string(lMaxSeq).length());

  // replace the string sequence
  if (lNameMatch.length() > 0) {
    std::stringstream lRet;
    lRet << std::dec << std::setw(lLen) << std::setfill('0') << lMaxSeq;
    lNameMatch = lPrefix + lRet.str() + lSuffix;
  } else {
    lNameMatch = std::to_string(lMaxSeq);
  }

  return lNameMatch;
}
}
} /* o2::DataDistribution */
