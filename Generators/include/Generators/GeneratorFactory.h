// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

/// \author S. Wenzel - Mai 2018

#ifndef ALICEO2_GENERATORFACTORY_H_
#define ALICEO2_GENERATORFACTORY_H_

class FairPrimaryGenerator;
namespace o2
{
namespace conf
{
class SimConfig;
}
} // namespace o2

namespace o2
{
namespace eventgen
{

// reusable helper class
// main purpose is to init a FairPrimGen given some (Sim)Config
struct GeneratorFactory {
  static void setPrimaryGenerator(o2::conf::SimConfig const&, FairPrimaryGenerator*);
};

} // end namespace eventgen
} // end namespace o2

#endif // ALICEO2_GENERATORFACTORY_H_
