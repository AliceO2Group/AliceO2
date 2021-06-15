// Copyright 2019-2020 CERN and copyright holders of ALICE O2.
// See http://alice-o2.web.cern.ch/copyright for details of the copyright holders.
// All rights not expressly granted are reserved.
//
// This software is distributed under the terms of the GNU General Public
// License v3 (GPL Version 3), copied verbatim in the file "COPYING".
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

#ifndef ALICEO2_NAME_GENERATOR_H_
#define ALICEO2_NAME_GENERATOR_H_

#include "DetectorsCommonDataFormats/DetID.h"
#include "CommonUtils/StringUtils.h"
#include "CommonUtils/ConfigurableParamHelper.h"
#include <string_view>

/// \file NameConf.h
/// \brief Definition of the Names Generator class
namespace o2
{
namespace base
{

// Class for standardization of the names for output files and trees
class NameConf : public o2::conf::ConfigurableParamHelper<NameConf>
{
  using DId = o2::detectors::DetID;

 public:

  // The Hits file name are generated by hardcoded schema, only prefix is mutable to allow the embedding
  static std::string getHitsFileName(o2::detectors::DetID d, const std::string_view prefix = STANDARDSIMPREFIX)
  {
    return o2::utils::Str::concat_string(prefix, "_", HITS_STRING, d.getName(), ".root");
  }

  // The Digits file name are generated by hardcoded schema, only prefix is mutable to allow the embedding
  static std::string getDigitsFileName(o2::detectors::DetID d, const std::string_view prefix = STANDARDSIMPREFIX)
  {
    return o2::utils::Str::concat_string(prefix, "_", DIGITS_STRING, d.getName(), ".root");
  }

  // Filename of collision context
  static std::string getCollisionContextFileName(const std::string_view prefix = "");

  // Filename of general run parameters (GRP)
  static std::string getGRPFileName(const std::string_view prefix = STANDARDSIMPREFIX);

  // Filename to store kinematics + TrackRefs
  static std::string getMCKinematicsFileName(const std::string_view prefix = STANDARDSIMPREFIX)
  {
    return o2::utils::Str::concat_string(prefix, "_", KINE_STRING, ".root");
  }

  // Filename to store kinematics + TrackRefs
  static std::string getMCHeadersFileName(const std::string_view prefix = STANDARDSIMPREFIX)
  {
    return o2::utils::Str::concat_string(prefix, "_", MCHEADER_STRING, ".root");
  }

  // Filename to store final MC configuration file
  static std::string getMCConfigFileName(const std::string_view prefix = STANDARDSIMPREFIX)
  {
    return o2::utils::Str::concat_string(prefix, "_", CONFIG_STRING, ".ini");
  }

  // Filename to store geometry file
  static std::string getGeomFileName(const std::string_view prefix = "");

  // Filename to for decoding dictionaries
  static std::string getAlpideClusterDictionaryFileName(DId det, const std::string_view prefix = "", const std::string_view ext = "");

  // Temporary alias to not crash the QC: Filename to for decoding dictionaries
  static std::string getDictionaryFileName(DId det, const std::string_view prefix = "", const std::string_view ext = "")
  {
    return getAlpideClusterDictionaryFileName(det, prefix, ext);
  }

  // Filename to store material LUT file
  static std::string getMatLUTFileName(const std::string_view prefix = "");

  // Filename to store summary about simulation processes and cut values
  static std::string getCutProcFileName(const std::string_view prefix = "");

  // TGeometry object name
  static constexpr std::string_view GEOMOBJECTNAME = "FAIRGeom"; // hardcoded

  // public standard TTree key (for MC ) -- not a function
  static constexpr std::string_view MCTTREENAME = "o2sim"; // hardcoded

  // standard name for digitization configuration output
  static constexpr std::string_view DIGITIZATIONCONFIGFILE = "o2simdigitizerworkflow_configuration.ini";

  // public standard CTF dictionary
  static constexpr std::string_view CTFDICT = "ctf_dictionary"; // hardcoded

  // CTF tree name
  static constexpr std::string_view CTFTREENAME = "ctf"; // hardcoded

  // CTF Filename
  static std::string getCTFFileName(uint32_t run, uint32_t orb, uint32_t id, const std::string_view prefix = "o2_ctf");

  // CTF Dictionary
  static std::string getCTFDictFileName();

  // The alignment object path in CCDB
  static std::string getAlignmentPath(o2::detectors::DetID d)
  {
    return o2::utils::Str::concat_string(d.getName(), "/", ALIGNPATH);
  }

 private:
  // helper method to build filenames
  static std::string buildFileName(const std::string_view prefix, const std::string_view delimiter, const std::string_view defPrefix, const std::string_view defName,
                                   const std::string_view extension, const std::string_view optDir = "");

  // unmodifiable constants used to construct filenames etc
  static constexpr std::string_view STANDARDSIMPREFIX = "o2sim";
  static constexpr std::string_view HITS_STRING = "Hits";     // hardcoded
  static constexpr std::string_view DIGITS_STRING = "Digits"; // hardcoded
  static constexpr std::string_view GRP_STRING = "grp";       // hardcoded
  static constexpr std::string_view KINE_STRING = "Kine";     // hardcoded
  static constexpr std::string_view MCHEADER_STRING = "MCHeader"; // hardcoded
  static constexpr std::string_view GEOM_FILE_STRING = "geometry";
  static constexpr std::string_view CUT_FILE_STRING = "proc-cut";
  static constexpr std::string_view CONFIG_STRING = "configuration";
  static constexpr std::string_view ROOT_EXT_STRING = "root";
  static constexpr std::string_view DAT_EXT_STRING = "dat";
  static constexpr std::string_view ALPIDECLUSDICTFILENAME = "dictionary";
  static constexpr std::string_view MATBUDLUT = "matbud";
  static constexpr std::string_view COLLISIONCONTEXT = "collisioncontext";
  static constexpr std::string_view ALIGNPATH = "Align";

  // these are configurable paths for some commonly used files
  std::string mDirGRP = "none";    // directory for GRP file ("none" == "")
  std::string mDirGeom = "none";   // directory for geometry file
  std::string mDirMatLUT = "none"; // directory for material LUT
  std::string mDirCollContext = "none"; // directory for collision context

  O2ParamDef(NameConf, "NameConf");
};

} // namespace base
} // namespace o2

#endif
