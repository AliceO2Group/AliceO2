// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

/// @author Ruben Shahoyan, ruben.shahoyan@cern.ch

#ifndef ALICEO2_ROOTCHAIN_H
#define ALICEO2_ROOTCHAIN_H

#include <memory>
#include <TChain.h>
#include <string>

namespace o2
{
namespace utils
{
/// Simple utility to create a chain from single root file or text list with root file names
/// Any input file not ending by ".root" will be considered as a list of input root files or
/// other lists (may be mixed with root files).
/// The lines starting by "//" or "#" will be ignored
class RootChain
{
 public:
  static std::unique_ptr<TChain> load(const std::string trName, const std::string inpFile);
  static void addFile(TChain* ch, const std::string inp);
};
} // namespace utils
} // namespace o2

#endif // ALICEO2_ROOTCHAIN_H
