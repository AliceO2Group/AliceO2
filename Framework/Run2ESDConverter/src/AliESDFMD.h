#ifndef ALIESDFMD_H
#define ALIESDFMD_H
/* Copyright(c) 1998-1999, ALICE Experiment at CERN, All rights
 * reserved. 
 *
 * See cxx source for full Copyright notice                               
 */
//___________________________________________________________________
//
// AliESDFMD is the Event Summary Data entry for the FMD.  It contains
// a rough estimate of the charged particle multiplicity in each strip
// of the FMD.    It also contains the psuedo-rapidity of each strip.
// This is important, as it varies from event to event, due to a
// finite interaction point probability distribution. 
//
#ifndef ROOT_TObject
# include <TObject.h>
#endif
#ifndef ALIFMDFLOATMAP_H
# include "AliFMDFloatMap.h"
#endif

//___________________________________________________________________
/** @class AliESDFMD 
    @brief Event Summary Data for the Forward Multiplicity Detector. 
    @ingroup FMD_data
    This stores the psuedo-multiplicity and -rapidiy for each strip of
    the FMD. 
 */
class AliESDFMD : public TObject
{
public:
  /** 
   * Base class of looping over the FMD ESD object 
   *
   * A simple example could be 
   * 
   * @code 
   * struct ESDFMDPrinter : AliESDFMD::ForOne
   * { 
   *   Bool_t operator()(UShort_t d, Char_t r, UShort_t s, UShort_t t,
   *                     Float_t m, Float_t e) 
   *   { 
   *     Printf("FMD%d%c[%2d,%3d]=%7.4f @ %7.4f", d, r, s, t, m, e);
   *     return kTRUE;
   *   }
   * };
   * @endcode
   */
  class ForOne 
  {
  public:
    /** 
     * Destructor
     */    
    virtual ~ForOne() {}
    /** 
     * Functional operator called for each entry 
     * 
     * @param d Detector number
     * @param r Ring identifier 
     * @param s Sector number
     * @param t Strip number
     * @param m 'Bare' multiplicity of this strip
     * @param e Pseudo-rapidity of this strip
     * 
     * @return @c kTRUE in case of success, @c kFALSE in case of failure.
     *         If the method returns @c kFALSE, the loop stops. 
     */    
    virtual bool operator()(UShort_t d, Char_t r, UShort_t s, UShort_t t, 
			      Float_t m, Float_t e) = 0;
  };
  enum { 
    kNeedNoiseFix = (1 << 14)
  };
  /** 
   * Default constructor 
   */
  AliESDFMD();
  /** 
   * Copy constructor 
   * 
   * @param other Object to construct from 
   */
  AliESDFMD(const AliESDFMD& other);
  /** 
   * Assignment operator 
   * 
   * @param other Object to assign from
   *
   * @return  reference to this object 
   */
  AliESDFMD& operator=(const AliESDFMD& other);
  /** 
   * Destructor - does nothing 
   */
  virtual ~AliESDFMD() {}
  /** 
   * Copy the content of this object to @a obj which must have been 
   * preallocated
   *
   * @param obj Object to copy to
   */
  virtual void Copy(TObject &obj) const;

  /**
   * Reset the object 
   */
  void Clear(Option_t *option="");
  /** 
   * Get the pseudo-multiplicity of 
   * @f$ \text{FMD}\langle detector\rangle\lange ring\rangle_{\langle
   * sector\rangle\langle strip\rangle}@f$ 
   * 
   * @param detector Detector number (1-3)
   * @param ring     Ring identifier ('I' or 'O')
   * @param sector   Sector number (0-511, or 0-255)
   * @param strip    Strip number (0-19, or 0-39)
   *
   * @return Psuedo multiplicity 
   */
  Float_t Multiplicity(UShort_t detector, Char_t ring, 
		       UShort_t sector, UShort_t strip) const;
  /** 
   * Get the pseudo-rapidity of 
   * @f$ \text{FMD}\langle detector\rangle\lange ring\rangle_{\langle
   * sector\rangle\langle strip\rangle}@f$ 
   *
   * @param detector Detector number (1-3)
   * @param ring     Ring identifier ('I' or 'O')
   * @param sector   Sector number (0-511, or 0-255)
   * @param strip    Strip number (0-19, or 0-39)
   *
   * @return Psuedo rapidity 
   */
  Float_t Eta(UShort_t detector, Char_t ring, 
	      UShort_t sector, UShort_t strip) const;
  /** 
   * Get the azimuthal angle of 
   * @f$ \text{FMD}\langle detector\rangle\lange ring\rangle_{\langle
   * sector\rangle\langle strip\rangle}@f$ 
   *
   * @param detector Detector number (1-3)
   * @param ring     Ring identifier ('I' or 'O')
   * @param sector   Sector number (0-511, or 0-255)
   * @param strip    Strip number (0-19, or 0-39)
   *
   * @return Azimuthal angle 
   */
  Float_t Phi(UShort_t detector, Char_t ring, 
	      UShort_t sector, UShort_t strip) const;
  /** 
   * Get the polar angle (in degrees) from beam line of 
   * @f$ \text{FMD}\langle detector\rangle\lange ring\rangle_{\langle
   * sector\rangle\langle strip\rangle}@f$ 
   *
   * @param detector Detector number (1-3)
   * @param ring     Ring identifier ('I' or 'O')
   * @param sector   Sector number (0-511, or 0-255)
   * @param strip    Strip number (0-19, or 0-39)
   *
   * @return Polar angle
   */
  Float_t Theta(UShort_t detector, Char_t ring, 
		UShort_t sector, UShort_t strip) const;
  /** 
   * Get the radial distance (in cm) from beam line of 
   * @f$ \text{FMD}\langle detector\rangle\lange ring\rangle_{\langle
   * sector\rangle\langle strip\rangle}@f$ 
   *
   * @param detector Detector number (1-3)
   * @param ring     Ring identifier ('I' or 'O')
   * @param sector   Sector number (0-511, or 0-255)
   * @param strip    Strip number (0-19, or 0-39)
   *
   * @return Radial distance
   */
  Float_t R(UShort_t detector, Char_t ring, 
	    UShort_t sector, UShort_t strip) const;
  /** 
   * Set the pseudo-multiplicity of 
   * @f$ \text{FMD}\langle detector\rangle\lange ring\rangle_{\langle
   * sector\rangle\langle strip\rangle}@f$ 
   * 
   * @param detector Detector number (1-3)
   * @param ring     Ring identifier ('I' or 'O')
   * @param sector   Sector number (0-511, or 0-255)
   * @param strip    Strip number (0-19, or 0-39)
   * @param mult     Psuedo multiplicity 
   */
  void SetMultiplicity(UShort_t detector, Char_t ring, 
		       UShort_t sector, UShort_t strip, 
		       Float_t mult);
  /** 
   * Set the pseudo-rapidity of 
   * @f$ \text{FMD}\langle detector\rangle\lange ring\rangle_{\langle
   * sector\rangle\langle strip\rangle}@f$ 
   * 
   * @param detector Detector number (1-3)
   * @param ring     Ring identifier ('I' or 'O')
   * @param sector   Sector number (0-511, or 0-255)
   * @param strip    Strip number (0-19, or 0-39)
   * @param eta      Psuedo rapidity 
   */
  void SetEta(UShort_t detector, Char_t ring, 
	      UShort_t sector, UShort_t strip, 
	      Float_t eta);
  /** 
   * @param f the factor for noise suppression 
   */
  void SetNoiseFactor(Float_t f) { fNoiseFactor = f; }
  /** 
   * @param done Whether we've done angle correction or not 
   */
  void SetAngleCorrected(Bool_t done) { fAngleCorrected = done; }
  /** 
   * @return Whether we've done angle correction or not 
   */
  Bool_t IsAngleCorrected() const { return fAngleCorrected; }
  /** 
   * @return the  factor for noise suppression 
   */
  Float_t GetNoiseFactor() const { return fNoiseFactor; }
  /** 
   * @return maximum number of detectors 
   */
  UShort_t MaxDetectors() const { return fMultiplicity.MaxDetectors(); }
  /** 
   * @return maximum number of rings 
   */
  UShort_t MaxRings()     const { return fMultiplicity.MaxRings(); }
  /** 
   * @return maximum number of sectors 
   */
  UShort_t MaxSectors()   const { return fMultiplicity.MaxSectors(); }
  /** 
   * @return Maximum number of strips 
   */
  UShort_t MaxStrips()    const { return fMultiplicity.MaxStrips(); }
  /** 
   * Print this object to standard out. 
   * 
   * @param option Options 
   */
  void Print(Option_t* option="") const;
  /** 
   * Check if we need the @c UShort_t fix 
   * 
   * @param file File this object was read from 
   */
  void CheckNeedUShort(TFile* file);
  /** 
   * Check if we need the noise fix 
   * 
   * @return true if class version of read object is 3 or smaller 
   */
  Bool_t NeedNoiseFix() const { return TestBit(kNeedNoiseFix); }
  /** 
   * Call the function operator of the passed object @a algo for each
   * entry in this object
   * 
   * @param algo Algorithm
   * 
   * @return @c kTRUE on success, @c kFALSE if the passed object
   * failed at any entry.  It will return immediately on error. 
   */
  Bool_t ForEach(ForOne& algo) const;
  enum {
    /** Value used for undefined multiplicity */
    kInvalidMult = 1024
  };
  enum {
    /** Value used for undefined pseudo rapidity */
    kInvalidEta = 1024
  };
  /** 
   * @return constant reference to multiplicity map 
   */
  const AliFMDFloatMap& MultiplicityMap() const { return fMultiplicity; }
  /** 
   * @return constant reference to pseudo-rapidity map 
   */
  const AliFMDFloatMap& EtaMap() const { return fEta; }
protected:
  AliFMDFloatMap fMultiplicity;   // Psuedo multplicity per strip
  AliFMDFloatMap fEta;            // Psuedo-rapidity per strip
  Float_t        fNoiseFactor;    // Factor used for noise suppresion
  Bool_t         fAngleCorrected; // Whether we've done angle correction
  ClassDef(AliESDFMD,4)           // ESD info from FMD
};
#endif
//____________________________________________________________________
//
// Local Variables:
//   mode: C++
// End:
//
// EOF
//
