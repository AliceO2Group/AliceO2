// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

// A C++ way to list / iterate files with glob searches

#ifndef O2_FILEITERATOR_H
#define O2_FILEITERATOR_H

#include <vector>
#include <string>

namespace o2::utils
{

// Return a vector of file names (not directories), with full path information, in a given directory "dir".
// If searchpattern is empty, all files will be returned. Otherwise
// searchpattern will be treated/parsed as a proper regular expression.
std::vector<std::string> listFiles(std::string const& dir, std::string const& searchpattern);

// same in the current dir
std::vector<std::string> listFiles(std::string const& searchpattern);

} // namespace o2::utils

#endif //O2_FILEITERATOR_H
