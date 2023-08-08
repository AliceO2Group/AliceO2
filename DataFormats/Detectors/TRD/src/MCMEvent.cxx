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

#include "DataFormatsTRD/MCMEvent.h"

namespace o2
{
namespace trd
{

const uint32_t MCMEvent::getRegister(const uint32_t regidx, const TrapRegInfo& trapreg) const
{
  // get the register value based on the address of the register;
  // find register in mTrapRegisters.
  // calculate the offset from the base for the register and get the mask.
  /*if(regidx == TrapRegisters::kADCMSK){
    LOGP(info,"mcmevent : reading back :ADCMSK  adcmsk : 0x0 mask {:08x} base {:08x} datawordnumber: {} shift: {} name : {}",  trapreg.getMask(),trapreg.getBase(), trapreg.getDataWordNumber(), trapreg.getShift(), trapreg.getName());
  }*/
  int regoffset = trapreg.getBase() + trapreg.getDataWordNumber(); // get the offset to the register in question
  uint32_t data = mRegisterData[regoffset];
  data = data >> trapreg.getShift();
  data &= trapreg.getMask(); // mask the data off as need be.
  /*if(regidx == TrapRegisters::kADCMSK){
    LOGP(info,"mcmevent  reading back :ADCMSK  adcmsk : {:08x} mask {:08x} base {:08x} shift {:08x} name : {}",data, trapreg.getMask(),trapreg.getBase(),trapreg.getShift(), trapreg.getName());
  }*/
  return data;
}

bool MCMEvent::setRegister(const uint32_t data, const uint32_t regidx, const TrapRegInfo& trapreg)
{
  uint32_t regvalue = data;
  int regoffset = trapreg.getBase() + trapreg.getDataWordNumber(); // wordnumber; // get the offset to the register in question
  regvalue &= trapreg.getMask();                                   // mask the data off as need be.
  uint32_t notdatamask = ~(trapreg.getMask() << trapreg.getShift());
  regvalue = regvalue << trapreg.getShift();
  auto trapregvalue = mRegisterData[regoffset];
  mRegisterData[regoffset] = mRegisterData[regoffset] & notdatamask;
  mRegisterData[regoffset] = mRegisterData[regoffset] | regvalue;
  return true;
}

} // namespace trd
} // namespace o2
