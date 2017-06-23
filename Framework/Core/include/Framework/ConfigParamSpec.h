// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See https://alice-o2.web.cern.ch/ for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.
#ifndef FRAMEWORK_CONFIGPARAMSPEC_H
#define FRAMEWORK_CONFIGPARAMSPEC_H

#include <string>
#include "Framework/Variant.h"

namespace o2 {
namespace framework {
struct ConfigParamSpec {
  enum ParamType {
    Int,
    Float,
    Double,
    String,
    Bool
  };

  std::string name;
  enum ParamType type;
  Variant defaultValue;
};

}
}

#endif // FRAMEWORK_CONFIGPARAMSPEC_H
