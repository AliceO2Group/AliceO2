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

#ifndef ALICEO2_DATAFORMATS_GENERATORINFO_H_
#define ALICEO2_DATAFORMATS_GENERATORINFO_H_

#include "TObject.h"

namespace o2
{
namespace dataformats
{

/*****************************************************************/
/*****************************************************************/

class GeneratorInfo : public TObject
{

 public:
  /** default constructor **/
  GeneratorInfo();
  /** copy constructor **/
  GeneratorInfo(const GeneratorInfo& rhs);
  /** operator= **/
  GeneratorInfo& operator=(const GeneratorInfo& rhs);
  /** destructor **/
  ~GeneratorInfo() override;

  /** methods to override **/
  virtual void Reset() = 0;

 protected:
  ClassDefOverride(GeneratorInfo, 1);

}; /** class GeneratorInfo **/

/*****************************************************************/
/*****************************************************************/

} /* namespace dataformats */
} /* namespace o2 */

#endif /* ALICEO2_DATAFORMATS_GENERATORINFO_H_ */
