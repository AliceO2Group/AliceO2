/// \file MagWrapCheb.h
/// \brief Definition of the MagWrapCheb class
/// \author ruben.shahoyan@cern.ch 20/03/2007

#ifndef ALICEO2_FIELD_MAGNETICWRAPPERCHEBYSHEV_H_
#define ALICEO2_FIELD_MAGNETICWRAPPERCHEBYSHEV_H_

#include <TMath.h>                      // for ATan2, Cos, Sin, Sqrt
#include <TNamed.h>                     // for TNamed
#include <TObjArray.h>                  // for TObjArray
#include "MathUtils/Chebyshev3D.h"      // for Chebyshev3D
#include "MathUtils/Chebyshev3DCalc.h"  // for _INC_CREATION_Chebyshev3D_
#include "Rtypes.h"                     // for Double_t, Int_t, Float_t, etc

class FairLogger;  // lines 16-16

namespace o2 {
namespace field {

///  Wrapper for the set of mag.field parameterizations by Chebyshev polinomials
///  To obtain the field in cartesian coordinates/components use
///    Field(double* xyz, double* bxyz);
///  For cylindrical coordinates/components:
///    fieldCylindrical(double* rphiz, double* brphiz)
///  The solenoid part is parameterized in the volume  R<500, -550<Z<550 cm
///  The region R<423 cm,  -343.3<Z<481.3 for 30kA and -343.3<Z<481.3 for 12kA
///  is parameterized using measured data while outside the Tosca calculation
///  is used (matched to data on the boundary of the measurements)
///  Two options are possible:
///  1) _BRING_TO_BOUNDARY_ is defined in the Chebyshev3D:
///     If the querried point is outside of the validity region then the field
///     at the closest point on the fitted surface is returned.
///  2) _BRING_TO_BOUNDARY_ is not defined in the Chebyshev3D:
///     If the querried point is outside of the validity region the return
///     value for the field components are set to 0.
///  To obtain the field integral in the TPC region from given point to nearest
///  cathod plane (+- 250 cm) use:
///  getTPCIntegral(double* xyz, double* bxyz);  for cartesian frame
///  or getTPCIntegralCylindrical(Double_t *rphiz, Double_t *b); for cylindrical frame
///  The units are kiloGauss and cm.
class MagneticWrapperChebyshev : public TNamed
{

  public:
    /// Default constructor
    MagneticWrapperChebyshev();

    /// Copy constructor
    MagneticWrapperChebyshev(const MagneticWrapperChebyshev &src);

    ~MagneticWrapperChebyshev() override
    {
      Clear();
    }

    /// Copy method
    void copyFrom(const MagneticWrapperChebyshev &src);

    /// Assignment operator
    MagneticWrapperChebyshev &operator=(const MagneticWrapperChebyshev &rhs);

    /// Clears all dynamic parts
    void Clear(const Option_t * = "") override;

    Int_t getNumberOfParametersSol() const
    {
      return mNumberOfParameterizationSolenoid;
    }

    Int_t getNumberOmCoordinatesSegmentsZSolenoid() const
    {
      return mNumberOfDistinctZSegmentsSolenoid;
    }

    Float_t *getSegZSol() const
    {
      return mCoordinatesSegmentsZSolenoid;
    }

    Int_t getNumberOfParametersTPCIntegral() const
    {
      return mNumberOfParameterizationTPC;
    }

    Int_t getNumberOmCoordinatesSegmentsZTPCInt() const
    {
      return mNumberOfDistinctZSegmentsTPC;
    }

    Int_t getNumberOfParametersTPCRatIntegral() const
    {
      return mNumberOfParameterizationTPCRat;
    }

    Int_t getNumberOmCoordinatesSegmentsZTPCRatIntegral() const
    {
      return mNumberOfDistinctZSegmentsTPCRat;
    }

    Int_t getNumberOfParametersDip() const
    {
      return mNumberOfParameterizationDipole;
    }

    Int_t getNumberOmCoordinatesSegmentsZDipole() const
    {
      return mNumberOfDistinctZSegmentsDipole;
    }

    Float_t getMaxZ() const
    {
      return getMaxZSol();
    }

    Float_t getMinZ() const
    {
      return mParameterizationDipole ? getMinZDip() : getMinZSol();
    }

    Float_t getMinZSol() const
    {
      return mMinZSolenoid;
    }

    Float_t getMaxZSol() const
    {
      return mMaxZSolenoid;
    }

    Float_t getMaxRSol() const
    {
      return mMaxRadiusSolenoid;
    }

    Float_t getMinZDip() const
    {
      return mMinDipoleZ;
    }

    Float_t getMaxZDip() const
    {
      return mMaxDipoleZ;
    }

    Float_t getMinZTPCIntegral() const
    {
      return mMinZTPC;
    }

    Float_t getMaxZTPCIntegral() const
    {
      return mMaxZTPC;
    }

    Float_t getMaxRTPCIntegral() const
    {
      return mMaxRadiusTPC;
    }

    Float_t getMinZTPCRatIntegral() const
    {
      return mMinZTPCRat;
    }

    Float_t getMaxZTPCRatIntegral() const
    {
      return mMaxZTPCRat;
    }

    Float_t getMaxRTPCRatIntegral() const
    {
      return mMaxRadiusTPCRat;
    }

    o2::mathUtils::Chebyshev3D *getParameterSolenoid(Int_t ipar) const
    {
      return (o2::mathUtils::Chebyshev3D *) mParameterizationSolenoid->UncheckedAt(ipar);
    }

    o2::mathUtils::Chebyshev3D *getParameterTPCRatIntegral(Int_t ipar) const
    {
      return (o2::mathUtils::Chebyshev3D *) mParameterizationTPCRat->UncheckedAt(ipar);
    }

    o2::mathUtils::Chebyshev3D *getParameterTPCIntegral(Int_t ipar) const
    {
      return (o2::mathUtils::Chebyshev3D *) mParameterizationTPC->UncheckedAt(ipar);
    }

    o2::mathUtils::Chebyshev3D *getParameterDipole(Int_t ipar) const
    {
      return (o2::mathUtils::Chebyshev3D *) mParameterizationDipole->UncheckedAt(ipar);
    }

    /// Prints info
    void Print(Option_t * = "") const override;

    /// Computes field in cartesian coordinates. If point is outside of the parameterized region
    /// it gets it at closest valid point
    virtual void Field(const Double_t *xyz, Double_t *b) const;

    /// Computes Bz for the point in cartesian coordinates. If point is outside of the parameterized region
    /// it gets it at closest valid point
    Double_t getBz(const Double_t *xyz) const;

    void fieldCylindrical(const Double_t *rphiz, Double_t *b) const;

    /// Computes TPC region field integral in cartesian coordinates.
    /// If point is outside of the parameterized region it gets it at closeset valid point
    void getTPCIntegral(const Double_t *xyz, Double_t *b) const;

    // Computes field integral in TPC region in Cylindircal coordinates
    // note: the check for the point being inside the parameterized region is done outside
    void getTPCIntegralCylindrical(const Double_t *rphiz, Double_t *b) const;

    /// Computes TPCRat region field integral in cartesian coordinates.
    /// If point is outside of the parameterized region it gets it at closeset valid point
    void getTPCRatIntegral(const Double_t *xyz, Double_t *b) const;

    // Computes field integral in TPCRat region in Cylindircal coordinates
    // note: the check for the point being inside the parameterized region is done outside
    void getTPCRatIntegralCylindrical(const Double_t *rphiz, Double_t *b) const;

    /// Finds the segment containing point xyz. If it is outside it finds the closest segment
    Int_t findSolenoidSegment(const Double_t *xyz) const;

    /// Finds the segment containing point xyz. If it is outside it finds the closest segment
    Int_t findTPCSegment(const Double_t *xyz) const;

    /// Finds the segment containing point xyz. If it is outside it finds the closest segment
    Int_t findTPCRatSegment(const Double_t *xyz) const;

    /// Finds the segment containing point xyz. If it is outside it finds the closest segment
    Int_t findDipoleSegment(const Double_t *xyz) const;

    static void cylindricalToCartesianCylB(const Double_t *rphiz, const Double_t *brphiz, Double_t *bxyz);

    static void cylindricalToCartesianCartB(const Double_t *xyz, const Double_t *brphiz, Double_t *bxyz);

    static void cartesianToCylindricalCartB(const Double_t *xyz, const Double_t *bxyz, Double_t *brphiz);

    static void cartesianToCylindricalCylB(const Double_t *rphiz, const Double_t *bxyz, Double_t *brphiz);

    static void cartesianToCylindrical(const Double_t *xyz, Double_t *rphiz);

    static void cylindricalToCartesian(const Double_t *rphiz, Double_t *xyz);

#ifdef _INC_CREATION_Chebyshev3D_ // see Cheb3D.h for explanation
    /// Reads coefficients data from the text file
    void loadData(const char* inpfile);

    /// Construct from coefficients from the text file
    MagneticWrapperChebyshev(const char* inputFile);

    /// Writes coefficients data to output text file
    void saveData(const char* outfile) const;

    /// Finds all boundaries in dimension dim for boxes in given region.
    /// if mn > mx for given projection the check is not done for it.
    Int_t segmentDimension(Float_t** seg, const TObjArray* par, int npar, int dim, Float_t xmn, Float_t xmx, Float_t ymn,
                           Float_t ymx, Float_t zmn, Float_t zmx);

    /// Adds new parameterization piece for Solenoid
    /// NOTE: pieces must be added strictly in increasing R then increasing Z order
    void addParameterSolenoid(const o2::mathUtils::Chebyshev3D* param);

    // Adds new parameterization piece for TPCIntegral
    // NOTE: pieces must be added strictly in increasing R then increasing Z order

    void addParameterTPCIntegral(const o2::mathUtils::Chebyshev3D* param);
    /// Adds new parameterization piece for TPCRatInt
    // NOTE: pieces must be added strictly in increasing R then increasing Z order
    void addParameterTPCRatIntegral(const o2::mathUtils::Chebyshev3D* param);

    /// Adds new parameterization piece for Dipole
    void addParameterDipole(const o2::mathUtils::Chebyshev3D* param);

    /// Builds lookup table for dipole
    void buildTable(Int_t npar, TObjArray* parArr, Int_t& nZSeg, Int_t& nYSeg, Int_t& nXSeg, Float_t& minZ, Float_t& maxZ,
                    Float_t** segZ, Float_t** segY, Float_t** segX, Int_t** begSegY, Int_t** nSegY, Int_t** begSegX,
                    Int_t** nSegX, Int_t** segID);

    /// Builds lookup table
    void buildTableSolenoid();

    /// Builds lookup table
    void buildTableDipole();

    /// Builds lookup table
    void buildTableTPCIntegral();

    /// Builds lookup table
    void buildTableTPCRatIntegral();

    /// Cleans TPC field integral (used for update)
    void resetTPCIntegral();

    /// Cleans TPCRat field integral (used for update)
    void resetTPCRatIntegral();

    /// Cleans Solenoid field (used for update)
    void resetSolenoid();

    /// Cleans Dipole field (used for update)
    void resetDipole();
#endif

  protected:
    /// Compute Solenoid field in Cylindircal coordinates
    /// note: if the point is outside the volume it gets the field in closest parameterized point
    void fieldCylindricalSolenoid(const Double_t *rphiz, Double_t *b) const;

    /// Compute Solenoid field in Cylindircal coordinates
    /// note: if the point is outside the volume it gets the field in closest parameterized point
    Double_t fieldCylindricalSolenoidBz(const Double_t *rphiz) const;

  protected:
    Int_t mNumberOfParameterizationSolenoid;  ///< Total number of parameterization pieces for solenoid
    Int_t mNumberOfDistinctZSegmentsSolenoid; ///< number of distinct Z segments in Solenoid
    Int_t mNumberOfDistinctPSegmentsSolenoid; ///< number of distinct P segments in Solenoid
    Int_t mNumberOfDistinctRSegmentsSolenoid; ///< number of distinct R segments in Solenoid
    Float_t *
      mCoordinatesSegmentsZSolenoid; //[mNumberOfDistinctZSegmentsSolenoid] coordinates of distinct Z segments in Solenoid
    Float_t *mCoordinatesSegmentsPSolenoid; //[mNumberOfDistinctPSegmentsSolenoid] coordinates of P segments for each
    // Zsegment in Solenoid
    Float_t *mCoordinatesSegmentsRSolenoid; //[mNumberOfDistinctRSegmentsSolenoid] coordinates of R segments for each
    // Psegment in Solenoid
    Int_t *mBeginningOfSegmentsPSolenoid; //[mNumberOfDistinctPSegmentsSolenoid] beginning of P segments array for each Z
    // segment
    Int_t *mNumberOfSegmentsPSolenoid;    //[mNumberOfDistinctZSegmentsSolenoid] number of P segments for each Z segment
    Int_t *mBeginningOfSegmentsRSolenoid; //[mNumberOfDistinctPSegmentsSolenoid] beginning of R segments array for each P
    // segment
    Int_t *mNumberOfRSegmentsSolenoid; //[mNumberOfDistinctPSegmentsSolenoid] number of R segments for each P segment
    Int_t *
      mSegmentIdSolenoid; //[mNumberOfDistinctRSegmentsSolenoid] ID of the solenoid parameterization for given RPZ segment
    Float_t mMinZSolenoid;                ///< Min Z of Solenoid parameterization
    Float_t mMaxZSolenoid;                ///< Max Z of Solenoid parameterization
    TObjArray *mParameterizationSolenoid; ///< Parameterization pieces for Solenoid field
    Float_t mMaxRadiusSolenoid;           ///< max radius for Solenoid field

    Int_t mNumberOfParameterizationTPC;  ///< Total number of parameterization pieces for TPCint
    Int_t mNumberOfDistinctZSegmentsTPC; ///< number of distinct Z segments in TPCint
    Int_t mNumberOfDistinctPSegmentsTPC; ///< number of distinct P segments in TPCint
    Int_t mNumberOfDistinctRSegmentsTPC; ///< number of distinct R segments in TPCint
    Float_t *mCoordinatesSegmentsZTPC;   //[mNumberOfDistinctZSegmentsTPC] coordinates of distinct Z segments in TPCint
    Float_t *
      mCoordinatesSegmentsPTPC; //[mNumberOfDistinctPSegmentsTPC] coordinates of P segments for each Zsegment in TPCint
    Float_t *
      mCoordinatesSegmentsRTPC; //[mNumberOfDistinctRSegmentsTPC] coordinates of R segments for each Psegment in TPCint
    Int_t *mBeginningOfSegmentsPTPC; //[mNumberOfDistinctPSegmentsTPC] beginning of P segments array for each Z segment
    Int_t *mNumberOfSegmentsPTPC;    //[mNumberOfDistinctZSegmentsTPC] number of P segments for each Z segment
    Int_t *mBeginningOfSegmentsRTPC; //[mNumberOfDistinctPSegmentsTPC] beginning of R segments array for each P segment
    Int_t *mNumberOfRSegmentsTPC;    //[mNumberOfDistinctPSegmentsTPC] number of R segments for each P segment
    Int_t *mSegmentIdTPC; //[mNumberOfDistinctRSegmentsTPC] ID of the TPCint parameterization for given RPZ segment
    Float_t mMinZTPC;     ///< Min Z of TPCint parameterization
    Float_t mMaxZTPC;     ///< Max Z of TPCint parameterization
    TObjArray *mParameterizationTPC; ///< Parameterization pieces for TPCint field
    Float_t mMaxRadiusTPC;           ///< max radius for Solenoid field integral in TPC

    Int_t
      mNumberOfParameterizationTPCRat; ///< Total number of parameterization pieces for tr.field to Bz integrals in TPC
    ///< region
    Int_t mNumberOfDistinctZSegmentsTPCRat; ///< number of distinct Z segments in TpcRatInt
    Int_t mNumberOfDistinctPSegmentsTPCRat; ///< number of distinct P segments in TpcRatInt
    Int_t mNumberOfDistinctRSegmentsTPCRat; ///< number of distinct R segments in TpcRatInt
    Float_t *
      mCoordinatesSegmentsZTPCRat; //[mNumberOfDistinctZSegmentsTPCRat] coordinates of distinct Z segments in TpcRatInt
    Float_t *mCoordinatesSegmentsPTPCRat; //[mNumberOfDistinctPSegmentsTPCRat] coordinates of P segments for each Zsegment
    // in TpcRatInt
    Float_t *mCoordinatesSegmentsRTPCRat; //[mNumberOfDistinctRSegmentsTPCRat] coordinates of R segments for each Psegment
    // in TpcRatInt
    Int_t *
      mBeginningOfSegmentsPTPCRat;   //[mNumberOfDistinctPSegmentsTPCRat] beginning of P segments array for each Z segment
    Int_t *mNumberOfSegmentsPTPCRat; //[mNumberOfDistinctZSegmentsTPCRat] number of P segments for each Z segment
    Int_t *
      mBeginningOfSegmentsRTPCRat;   //[mNumberOfDistinctPSegmentsTPCRat] beginning of R segments array for each P segment
    Int_t *mNumberOfRSegmentsTPCRat; //[mNumberOfDistinctPSegmentsTPCRat] number of R segments for each P segment
    Int_t *
      mSegmentIdTPCRat;  //[mNumberOfDistinctRSegmentsTPCRat] ID of the TpcRatInt parameterization for given RPZ segment
    Float_t mMinZTPCRat; ///< Min Z of TpcRatInt parameterization
    Float_t mMaxZTPCRat; ///< Max Z of TpcRatInt parameterization
    TObjArray *mParameterizationTPCRat; ///< Parameterization pieces for TpcRatInt field
    Float_t mMaxRadiusTPCRat;           ///< max radius for Solenoid field ratios integral in TPC

    Int_t mNumberOfParameterizationDipole;  ///< Total number of parameterization pieces for dipole
    Int_t mNumberOfDistinctZSegmentsDipole; ///< number of distinct Z segments in Dipole
    Int_t mNumberOfDistinctYSegmentsDipole; ///< number of distinct Y segments in Dipole
    Int_t mNumberOfDistinctXSegmentsDipole; ///< number of distinct X segments in Dipole
    Float_t *
      mCoordinatesSegmentsZDipole; //[mNumberOfDistinctZSegmentsDipole] coordinates of distinct Z segments in Dipole
    Float_t *mCoordinatesSegmentsYDipole; //[mNumberOfDistinctYSegmentsDipole] coordinates of Y segments for each Zsegment
    // in Dipole
    Float_t *mCoordinatesSegmentsXDipole; //[mNumberOfDistinctXSegmentsDipole] coordinates of X segments for each Ysegment
    // in Dipole
    Int_t *
      mBeginningOfSegmentsYDipole;   //[mNumberOfDistinctZSegmentsDipole] beginning of Y segments array for each Z segment
    Int_t *mNumberOfSegmentsYDipole; //[mNumberOfDistinctZSegmentsDipole] number of Y segments for each Z segment
    Int_t *
      mBeginningOfSegmentsXDipole;   //[mNumberOfDistinctYSegmentsDipole] beginning of X segments array for each Y segment
    Int_t *mNumberOfSegmentsXDipole; //[mNumberOfDistinctYSegmentsDipole] number of X segments for each Y segment
    Int_t *mSegmentIdDipole; //[mNumberOfDistinctXSegmentsDipole] ID of the dipole parameterization for given XYZ segment
    Float_t mMinDipoleZ;     ///< Min Z of Dipole parameterization
    Float_t mMaxDipoleZ;     ///< Max Z of Dipole parameterization
    TObjArray *mParameterizationDipole; ///< Parameterization pieces for Dipole field

    FairLogger *mLogger; //!
    ClassDefOverride(o2::field::MagneticWrapperChebyshev,
    2) // Wrapper class for the set of Chebishev parameterizations of Alice mag.field
};

/// Computes field in Cylindircal coordinates
inline void MagneticWrapperChebyshev::fieldCylindrical(const Double_t *rphiz, Double_t *b) const
{
  //  if (rphiz[2]<GetMinZSol() || rphiz[2]>GetMaxZSol() || rphiz[0]>GetMaxRSol()) {for (int i=3;i--;) b[i]=0; return;}
  b[0] = b[1] = b[2] = 0;
  fieldCylindricalSolenoid(rphiz, b);
}

/// Converts field in cylindrical coordinates to cartesian system, point is in cyl.system
inline void MagneticWrapperChebyshev::cylindricalToCartesianCylB(const Double_t *rphiz, const Double_t *brphiz,
                                                                 Double_t *bxyz)
{
  Double_t btr = TMath::Sqrt(brphiz[0] * brphiz[0] + brphiz[1] * brphiz[1]);
  Double_t psiPLUSphi = TMath::ATan2(brphiz[1], brphiz[0]) + rphiz[1];
  bxyz[0] = btr * TMath::Cos(psiPLUSphi);
  bxyz[1] = btr * TMath::Sin(psiPLUSphi);
  bxyz[2] = brphiz[2];
}

/// Converts field in cylindrical coordinates to cartesian system, point is in cart.system
inline void MagneticWrapperChebyshev::cylindricalToCartesianCartB(const Double_t *xyz, const Double_t *brphiz,
                                                                  Double_t *bxyz)
{
  Double_t btr = TMath::Sqrt(brphiz[0] * brphiz[0] + brphiz[1] * brphiz[1]);
  Double_t phiPLUSpsi = TMath::ATan2(xyz[1], xyz[0]) + TMath::ATan2(brphiz[1], brphiz[0]);
  bxyz[0] = btr * TMath::Cos(phiPLUSpsi);
  bxyz[1] = btr * TMath::Sin(phiPLUSpsi);
  bxyz[2] = brphiz[2];
}

/// Converts field in cylindrical coordinates to cartesian system, poin is in cart.system
inline void MagneticWrapperChebyshev::cartesianToCylindricalCartB(const Double_t *xyz, const Double_t *bxyz,
                                                                  Double_t *brphiz)
{
  Double_t btr = TMath::Sqrt(bxyz[0] * bxyz[0] + bxyz[1] * bxyz[1]);
  Double_t psiMINphi = TMath::ATan2(bxyz[1], bxyz[0]) - TMath::ATan2(xyz[1], xyz[0]);

  brphiz[0] = btr * TMath::Cos(psiMINphi);
  brphiz[1] = btr * TMath::Sin(psiMINphi);
  brphiz[2] = bxyz[2];
}

/// Converts field in cylindrical coordinates to cartesian system, point is in cyl.system
inline void MagneticWrapperChebyshev::cartesianToCylindricalCylB(const Double_t *rphiz, const Double_t *bxyz,
                                                                 Double_t *brphiz)
{
  Double_t btr = TMath::Sqrt(bxyz[0] * bxyz[0] + bxyz[1] * bxyz[1]);
  Double_t psiMINphi = TMath::ATan2(bxyz[1], bxyz[0]) - rphiz[1];
  brphiz[0] = btr * TMath::Cos(psiMINphi);
  brphiz[1] = btr * TMath::Sin(psiMINphi);
  brphiz[2] = bxyz[2];
}

inline void MagneticWrapperChebyshev::cartesianToCylindrical(const Double_t *xyz, Double_t *rphiz)
{
  rphiz[0] = TMath::Sqrt(xyz[0] * xyz[0] + xyz[1] * xyz[1]);
  rphiz[1] = TMath::ATan2(xyz[1], xyz[0]);
  rphiz[2] = xyz[2];
}

inline void MagneticWrapperChebyshev::cylindricalToCartesian(const Double_t *rphiz, Double_t *xyz)
{
  xyz[0] = rphiz[0] * TMath::Cos(rphiz[1]);
  xyz[1] = rphiz[0] * TMath::Sin(rphiz[1]);
  xyz[2] = rphiz[2];
}
}
}

#endif
