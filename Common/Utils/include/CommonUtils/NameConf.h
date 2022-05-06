// Copyright 2019-2020 CERN and copyright holders of ALICE O2.
// See https://alice-o2.web.cern.ch/copyright for details of the copyright holders.
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
 public:
  // Filename of collision context
  static std::string getCollisionContextFileName(const std::string_view prefix = "");

  // Filename of general run parameters (GRP)
  static std::string getGRPFileName(const std::string_view prefix = STANDARDSIMPREFIX);

  // Filename of general run parameters from ECS (GRPECS)
  static std::string getGRPECSFileName(const std::string_view prefix = STANDARDSIMPREFIX);

  // Filename of general run parameters from LHCIF (GRPLHCIF)
  static std::string getGRPLHCIFFileName(const std::string_view prefix = STANDARDSIMPREFIX);

  // Filename of general run parameters fof B field (GRPMagField)
  static std::string getGRPMagFieldFileName(const std::string_view prefix = STANDARDSIMPREFIX);

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

  // Filename for TFIDInfo vector
  static std::string getTFIDInfoFileName(const std::string_view prefix = "o2");

  static constexpr std::string_view CCDBOBJECT = "ccdb_object"; // hardcoded
  static constexpr std::string_view CCDBMETA = "ccdb_meta";     // hardcoded
  static constexpr std::string_view CCDBQUERY = "ccdb_query";   // hardcoded
  static constexpr std::string_view CCDBRunTag = "runNumber";
  static constexpr std::string_view CCDBDefaultEntry = "default";

  // Filename to store geometry file
  static std::string getGeomFileName(const std::string_view prefix = "");
  static std::string getAlignedGeomFileName(const std::string_view prefix = "");

  // Filename to store material LUT file
  static std::string getMatLUTFileName(const std::string_view prefix = "");

  // Filename to store summary about simulation processes and cut values
  static std::string getCutProcFileName(const std::string_view prefix = "");

  // TGeometry object name
  static constexpr std::string_view GEOMOBJECTNAME_FAIR = "FAIRGeom"; // hardcoded

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

  // Default CCDB server
  static std::string getCCDBServer();

 protected:
  // helper method to build filenames
  static std::string buildFileName(const std::string_view prefix, const std::string_view delimiter, const std::string_view defPrefix, const std::string_view defName,
                                   const std::string_view extension, const std::string_view optDir = "");

  // unmodifiable constants used to construct filenames etc
  static constexpr std::string_view STANDARDSIMPREFIX = "o2sim";
  static constexpr std::string_view HITS_STRING = "Hits";               // hardcoded
  static constexpr std::string_view DIGITS_STRING = "Digits";           // hardcoded
  static constexpr std::string_view GRP_STRING = "grp";                 // hardcoded
  static constexpr std::string_view GRPECS_STRING = "grpecs";           // hardcoded
  static constexpr std::string_view GRPLHCIF_STRING = "grplhcif";       // hardcoded
  static constexpr std::string_view GRPMAGFIELD_STRING = "grpMagField"; // hardcoded
  static constexpr std::string_view KINE_STRING = "Kine";               // hardcoded
  static constexpr std::string_view MCHEADER_STRING = "MCHeader";       // hardcoded
  static constexpr std::string_view GEOM_FILE_STRING = "geometry";
  static constexpr std::string_view ALIGNEDGEOM_FILE_STRING = "geometry-aligned";
  static constexpr std::string_view CUT_FILE_STRING = "proc-cut";
  static constexpr std::string_view CONFIG_STRING = "configuration";
  static constexpr std::string_view ROOT_EXT_STRING = "root";
  static constexpr std::string_view DAT_EXT_STRING = "dat";
  static constexpr std::string_view ALPIDECLUSDICTFILENAME = "dictionary";
  static constexpr std::string_view NOISEFILENAME = "NoiseMap";
  static constexpr std::string_view MATBUDLUT = "matbud";
  static constexpr std::string_view COLLISIONCONTEXT = "collisioncontext";
  static constexpr std::string_view ALIGNPATH = "Calib/Align";
  static constexpr std::string_view TFIDINFO = "tfidinfo";

  // these are configurable paths for some commonly used files
  std::string mDirGRP = "none";                           // directory for GRP file ("none" == "")
  std::string mDirGRPECS = "none";                        // directory for GRPECS file ("none" == "")
  std::string mDirGRPLHCIF = "none";                      // directory for GRPLHCIF file ("none" == "")
  std::string mDirGRPMagField = "none";                   // directory for GRPMagField file ("none" == "")
  std::string mDirGeom = "none";                          // directory for geometry file
  std::string mDirMatLUT = "none";                        // directory for material LUT
  std::string mDirCollContext = "none";                   // directory for collision context
  std::string mDirTFIDINFO = "none";                      // directory for TFIDInfo vector
  std::string mCCDBServer = "http://alice-ccdb.cern.ch/"; // default CCDB server

  O2ParamDef(NameConf, "NameConf");
};

} // namespace base
} // namespace o2

#endif
