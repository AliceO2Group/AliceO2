// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

// a couple of static functions that serve as name generators to standardize
// file names per contract

#ifndef O2_DETECTORSCOMMON_FILENAMEGENERATOR
#define O2_DETECTORSCOMMON_FILENAMEGENERATOR

#include "DetectorsCommonDataFormats/DetID.h"

namespace o2
{
namespace filenames
{

class SimFileNameGenerator
{
 public:
  constexpr static const char* STANDARDSIMPREFIX = "o2sim";

  // GRP
  constexpr static const char* GRPSTR = "grp";
  inline static std::string getGRPFileName(const char* prefix = STANDARDSIMPREFIX)
  {
    return std::string(prefix) + std::string("_") + std::string(GRPSTR) + std::string(".root");
  }

  // Hit file names
  constexpr static const char* HITSTR = "hits";
  inline static std::string getHitFileName(o2::detectors::DetID id, const char* prefix = STANDARDSIMPREFIX)
  {
    return std::string(prefix) + std::string("_") + std::string(id.getName()) + std::string(HITSTR) + std::string(".root");
  }

  // Kinematics file names
  constexpr static const char* KINEMATICSSTR = "kine";
  inline static std::string getKinematicsFileName(const char* prefix = STANDARDSIMPREFIX)
  {
    return std::string(prefix) + std::string("_") + std::string(KINEMATICSSTR) + std::string(".root");
  }
};

//
} // namespace filenames
} // end namespace o2

#endif
