// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

/// \author R+Preghenella - August 2017

#include "SimulationDataFormat/GeneratorInfo.h"

namespace o2
{
namespace dataformats
{

/*****************************************************************/
/*****************************************************************/

GeneratorInfo::GeneratorInfo() : TObject() { /** default constructor **/}

/*****************************************************************/

GeneratorInfo::GeneratorInfo(const GeneratorInfo& rhs) : TObject(rhs) { /** copy constructor **/}

/*****************************************************************/

GeneratorInfo& GeneratorInfo::operator=(const GeneratorInfo& rhs)
{
  /** operator= **/

  if (this == &rhs)
    return *this;
  TObject::operator=(rhs);
  return *this;
}

/*****************************************************************/

GeneratorInfo::~GeneratorInfo() { /** default destructor **/}

/*****************************************************************/
/*****************************************************************/

} /* namespace dataformats */
} /* namespace o2 */

ClassImp(o2::dataformats::GeneratorInfo)
