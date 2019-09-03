// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

#ifndef ALICEO2_EMCAL_SHISHKEBABTRD1MODULE_H
#define ALICEO2_EMCAL_SHISHKEBABTRD1MODULE_H

#include <iomanip>
#include <memory>

#include <TMath.h>
#include <TNamed.h>
#include <TVector2.h>

#include <FairLogger.h>

namespace o2
{
namespace emcal
{

//_________________________________________________________________________
/// \class ShishKebabTrd1Module
/// \ingroup EMCALUtils
/// \brief Main class for TRD1 geometry of Shish-Kebab case.
///
/// Sep 20004 - Nov 2006; Apr 2010
/// See web page with description of Shish-Kebab geometries:
/// http://pdsfweb01.nersc.gov/~pavlinov/ALICE/SHISHKEBAB/RES/shishkebabALICE.html
/// Nov 9,2006 - added case of 3X3
///
/// \author: Alexei Pavlinov (WSU).
///
class ShishKebabTrd1Module
{
 public:
  ///
  /// Constructor.
  ///
  ShishKebabTrd1Module(Double_t theta = 0.0, Geometry* g = nullptr);

  ///
  /// Constructor.
  ///
  ShishKebabTrd1Module(ShishKebabTrd1Module& leftNeighbor);

  ///
  /// Init (add explanation)
  ///
  void Init(Double_t A, Double_t B);

  ///
  /// Define more things (add explanation)
  ///
  void DefineAllStuff();

  ///
  /// Copy Constructor.
  ///
  ShishKebabTrd1Module(const ShishKebabTrd1Module& mod);

  ShishKebabTrd1Module& operator=(const ShishKebabTrd1Module& /*rvalue*/)
  {
    LOG(FATAL) << "operator = not implemented";
    return *this;
  }

  ~ShishKebabTrd1Module() = default;

  ///
  /// Recover module parameters stored in geometry
  ///
  Bool_t SetParameters();

  ///
  /// This is what we have in produced SM. (add explanation)
  ///    Oct 23-25, 2010
  ///  key=0 - zero tilt of first module;
  ///  key=1 - angle=fgangle/2 = 0.75 degree.
  ///
  void DefineFirstModule(const Int_t key = 0); // key=0-zero tilt of first module

  Double_t GetTheta() const { return mTheta; }
  const TVector2& GetCenterOfModule() const { return mOK; }
  Double_t GetPosX() const { return mOK.Y(); }
  Double_t GetPosZ() const { return mOK.X(); }
  Double_t GetPosXfromR() const { return mOK.Y() - sr; }
  Double_t GetA() const { return mA; }
  Double_t GetB() const { return mB; }
  Double_t GetRadius() const { return sr; }
  TVector2 GetORB() const { return mORB; }
  TVector2 GetORT() const { return mORT; }

  //  Additional offline stuff
  //  ieta=0 or 1 - Jun 02, 2006
  const TVector2& GetCenterOfCellInLocalCoordinateofSM(Int_t ieta) const
  {
    if (ieta <= 0)
      return mOK2;
    else
      return mOK1;
  }

  void GetCenterOfCellInLocalCoordinateofSM(Int_t ieta, Double_t& xr, Double_t& zr) const
  {
    if (ieta <= 0) {
      xr = mOK2.Y();
      zr = mOK2.X();
    } else {
      xr = mOK1.Y();
      zr = mOK1.X();
    }
    LOG(DEBUG2) << " ieta " << std::setw(2) << std::setprecision(2) << ieta << " xr " << std::setw(8)
                << std::setprecision(4) << xr << " zr " << std::setw(8) << std::setprecision(4) << zr;
  }

  void GetCenterOfCellInLocalCoordinateofSM3X3(Int_t ieta, Double_t& xr, Double_t& zr) const
  { // 3X3 case - Nov 9,2006
    if (ieta < 0)
      ieta = 0; // ieta = ieta<0? ieta=0 : ieta; // check index
    if (ieta > 2)
      ieta = 2; // ieta = ieta>2? ieta=2 : ieta;
    xr = mOK3X3[2 - ieta].Y();
    zr = mOK3X3[2 - ieta].X();
  }

  void GetCenterOfCellInLocalCoordinateofSM1X1(Double_t& xr, Double_t& zr) const
  { // 1X1 case - Nov 27,2006 // Center of cell is center of module
    xr = mOK.Y() - sr;
    zr = mOK.X();
  }

  // 15-may-06
  const TVector2& GetCenterOfModuleFace() const { return mOB; }
  const TVector2& GetCenterOfModuleFace(Int_t ieta) const
  {
    if (ieta <= 0)
      return mOB2;
    else
      return mOB1;
  }

  // Jul 30, 2007
  void GetPositionAtCenterCellLine(Int_t ieta, Double_t dist, TVector2& v) const;

  //
  Double_t GetTanBetta() const { return stanBetta; }
  Double_t Getb() const { return sb; }

  // service methods
  void PrintShish(Int_t pri = 1) const; // *MENU*
  Double_t GetThetaInDegree() const;
  Double_t GetEtaOfCenterOfModule() const;
  Double_t GetMaxEtaOfModule() const;
  static Double_t ThetaToEta(Double_t theta) { return -TMath::Log(TMath::Tan(theta / 2.)); }

 protected:
  // geometry info
  Geometry* mGeometry;       //!<! pointer to geometry info
  static Double_t sa;        ///<  2*dx1=2*dy1
  static Double_t sa2;       ///<  2*dx2
  static Double_t sb;        ///<  2*dz1
  static Double_t sangle;    ///<  in rad (1.5 degree)
  static Double_t stanBetta; ///<  tan(fgangle/2.)
  static Double_t sr;        ///<  radius to IP

  TVector2 mOK;         ///< position the module center in ALICE system; x->y; z->x;
  Double_t mA{0.};      ///< parameters of right line : y = A*z + B
  Double_t mB{0.};      ///< system where zero point is IP.
  Double_t mThetaA{0.}; ///< angle coresponding fA - for convinience
  Double_t mTheta;      ///< theta angle of perpendicular to SK module

  // position of towers(cells) with differents ieta (1 or 2) in local coordinate of SM
  // Nov 04,2004; Feb 19,2006
  TVector2 mOK1; ///< ieta=1
  TVector2 mOK2; ///< ieta=0

  // May 13, 2006; local position of module (cells) center face
  TVector2 mOB;  ///< module
  TVector2 mOB1; ///< ieta=1
  TVector2 mOB2; ///< ieta=0

  // Jul 30, 2007
  Double_t mThetaOB1{0.}; ///< theta of cell center line (go through OB1)
  Double_t mThetaOB2{0.}; ///< theta of cell center line (go through OB2)

  // 3X3 case - Nov 9,2006
  TVector2 mOK3X3[3];

  // Apr 14, 2010 - checking of geometry
  TVector2 mORB; ///< position of right/bottom point of module
  TVector2 mORT; ///< position of right/top    point of module
};
} // namespace emcal
} // namespace o2
#endif
