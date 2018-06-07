// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

/// \file ElectronTransport.h
/// \brief Definition of the electron transport
/// \author Andi Mathis, TU München, andreas.mathis@ph.tum.de

#ifndef ALICEO2_TPC_ElectronTransport_H_
#define ALICEO2_TPC_ElectronTransport_H_

#include "TPCBase/ParameterDetector.h"
#include "TPCBase/ParameterGas.h"

#include "TPCBase/Mapper.h"
#include "TPCBase/RandomRing.h"

namespace o2
{
namespace TPC
{

/// \class ElectronTransport
/// This class handles the electron transport in the active volume of the TPC.
/// In particular, in deals with the diffusion of the charge cloud while drifting towards the readout chambers and the
/// loss of electrons during that drift due to attachement.

class ElectronTransport
{
 public:
  static ElectronTransport& instance()
  {
    static ElectronTransport electronTransport;
    return electronTransport;
  }

  /// Destructor
  ~ElectronTransport();

  /// Update the OCDB parameters cached in the class. To be called once per event
  void updateParameters();

  /// Drift of electrons in electric field taking into account diffusion
  /// \param posEle GlobalPosition3D with start position of the electrons
  /// \return driftTime Drift time taking into account diffusion in z direction
  /// \return GlobalPosition3D with position of the electrons after the drift taking into account diffusion
  GlobalPosition3D getElectronDrift(GlobalPosition3D posEle, float& driftTime);

  /// Drift of electrons in electric field taking into account diffusion with 3 sigma of the width
  /// \param posEle GlobalPosition3D with start position of the electrons
  /// \return GlobalPosition3D with position of the electrons after the drift taking into account diffusion with
  /// 3 sigma of the width
  bool isCompletelyOutOfSectorCoarseElectronDrift(GlobalPosition3D posEle, const Sector& sector) const;

  /// Attachment probability for a given drift time
  /// \param driftTime Drift time of the electron
  /// \return Boolean whether the electron is attached (and lost) or not
  bool isElectronAttachment(float driftTime);

  /// Compute electron drift time from z position
  /// \param zPos z position of the charge
  /// \param signChange If the zPosition of the charge is shifted to the other TPC side, the drift length needs to be
  /// accordingly longer. In such cases, this parameter is set to -1
  /// \return Time of the charge
  float getDriftTime(float zPos, float signChange = 1.f) const;

 private:
  ElectronTransport();

  /// Circular random buffer containing random values of the Gauss distribution to take into account diffusion of the
  /// electrons
  RandomRing mRandomGaus;
  /// Circular random buffer containing flat random values to take into account electron attachment during drift
  RandomRing mRandomFlat;

  std::array<double, 18> mSinsPerSector{
    { 0, 0.3420201433256687129080830800376133993268, 0.6427876096865392518964199553010985255241,
      0.8660254037844385965883020617184229195118, 0.9848077530122080203156542665965389460325,
      0.9848077530122080203156542665965389460325, 0.866025403784438707610604524234076961875,
      0.6427876096865394739410248803324066102505, 0.3420201433256688794415367738110944628716, 0.,
      -0.3420201433256686573969318487797863781452, -0.6427876096865392518964199553010985255241,
      -0.8660254037844383745436971366871148347855, -0.9848077530122080203156542665965389460325,
      -0.9848077530122081313379567291121929883957, -0.8660254037844385965883020617184229195118,
      -0.6427876096865395849633273428480606526136, -0.3420201433256686018857806175219593569636 }
  }; ///< Array of sin for all sectors

  std::array<double, 18> mCosinsPerSector{
    { 1, 0.9396926207859084279050421173451468348503, 0.7660444431189780134516809084743726998568,
      0.5000000000000001110223024625156540423632, 0.1736481776669304144533612088707741349936,
      -0.1736481776669303034310587463551200926304, -0.4999999999999997779553950749686919152737,
      -0.7660444431189779024293784459587186574936, -0.9396926207859083168827396548294927924871, -1,
      -0.9396926207859084279050421173451468348503, -0.7660444431189780134516809084743726998568,
      -0.5000000000000004440892098500626161694527, -0.1736481776669303311866343619840336032212,
      0.1736481776669299703641513588081579655409, 0.5000000000000001110223024625156540423632,
      0.7660444431189777914070759834430646151304, 0.9396926207859084279050421173451468348503 }
  }; ///< Array of cos for all sectors

  const ParameterDetector* mDetParam; ///< Caching of the parameter class to avoid multiple CDB calls
  const ParameterGas* mGasParam;      ///< Caching of the parameter class to avoid multiple CDB calls
};

inline bool ElectronTransport::isElectronAttachment(float driftTime)
{
  if (mRandomFlat.getNextValue() < mGasParam->getAttachmentCoefficient() * mGasParam->getOxygenContent() * driftTime) {
    return true; /// electron is attached and lost
  } else
    return false; /// not attached
}

inline float ElectronTransport::getDriftTime(float zPos, float signChange) const
{
  float time = (mDetParam->getTPClength() - signChange * std::abs(zPos)) / mGasParam->getVdrift();
  return time;
}
}
}

#endif // ALICEO2_TPC_ElectronTransport_H_
