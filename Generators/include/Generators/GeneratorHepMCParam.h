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

/// \author R+Preghenella - January 2020

#ifndef ALICEO2_EVENTGEN_GENERATORHEPMCPARAM_H_
#define ALICEO2_EVENTGEN_GENERATORHEPMCPARAM_H_

#include "CommonUtils/ConfigurableParam.h"
#include "CommonUtils/ConfigurableParamHelper.h"
#include <string>

namespace o2
{
namespace eventgen
{

/**
 ** a parameter class/struct to keep the settings of
 ** the HepMC event generator and
 ** allow the user to modify them
 **/

struct GeneratorHepMCParam : public o2::conf::ConfigurableParamHelper<GeneratorHepMCParam> {
  /** Version number of event structure to decode.  Note, when reading
   *  from a file, this key is ignored.  The interface will figure out
   *  the version automatically.  When reading from a pipe, and the
   *  command line writes HepMC version 2 format, then this parameter
   *  should be set to 2 */
  int version = 0;
  /** Number of events to skip at the start of the input */
  uint64_t eventsToSkip = 0;
  /** Deprecated.  Set the name of the file to read from.  Use
   * GeneratorFileOrCmd.fileNames instead. */
  std::string fileName = "";
  /** Wether to prune the event tree.  If true, then only particles that are
   *
   * - beam particles (status == 4)
   * - decayed (status == 2), or
   * - final state (status == 1)
   *
   * are kept.  This reduces the event size.  How much depends on the
   * event generator producing the event.  Use with caution, as it may
   * corrupt the event record. */
  bool prune = false;
  O2ParamDef(GeneratorHepMCParam, "HepMC");
};

} // end namespace eventgen
} // end namespace o2

#endif // ALICEO2_EVENTGEN_GENERATORHEPMCPARAM_H_
