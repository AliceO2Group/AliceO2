#ifndef ALIFMDMAP_H
#define ALIFMDMAP_H
/* Copyright(c) 1998-2000, ALICE Experiment at CERN, All rights
 * reserved. 
 *
 * See cxx source for full Copyright notice                               
 */
#ifndef ROOT_TObject
# include <TObject.h>
#endif 
class TFile;

//____________________________________________________________________
/** @class AliFMDMap
    @brief Base class for caches of per-strip information.
    @ingroup FMD_data
    This is used to index a strip. Data stored depends on derived
    class.  */
class AliFMDMap : public TObject 
{
public:
  enum { 
    /** Default maximum detector number */
    kMaxDetectors = 3, 
    /** Default maximum number of rings */
    kMaxRings     = 2, 
    /** Default maximum number of sectors */
    kMaxSectors   = 40, 
    /** Default maximum number of strips */
    kMaxStrips    = 512
  };
  enum { 
    /** Number used for inner rings */
    kInner = 0, 
    /** Number used for outer rings */
    kOuter
  };
  enum { 
    /** Number of strips in outer rings */ 
    kNStripOuter = 256, 
    /** Number of strips in the inner rings */ 
    kNStripInner = 512
  };
  enum { 
    /** Number of sectors in the inner rings */ 
    kNSectorInner = 20, 
    /** Number of sectorts in the outer rings */ 
    kNSectorOuter = 40
 };
  enum { 
    /** Base index for inner rings */ 
    kBaseInner = 0, 
    /** Base index for outer rings */
    kBaseOuter = kNSectorInner * kNStripInner
  };
  enum { 
    /** Base for FMD1 index */ 
    kFMD1Base = 0, 
    /** Base for FMD2 index */ 
    kFMD2Base = kNSectorInner * kNStripInner, 
    /** Base for FMD3 index */ 
    kFMD3Base = (kBaseOuter + kNSectorOuter * kNStripOuter + kFMD2Base)
  };
  /** 
   * Class to do stuff on each element of a map in an efficient
   * way. 
   */ 
  class ForOne 
  {
  public:
  /** Destructor */
    virtual ~ForOne() { }
    /** 
     * Called for each element considered a floating point number 
     *
     * @param d   Detector number
     * @param r   Ring identifier 
     * @param s   Sector number
     * @param t   Strip number
     * @param v   Value (as a floating point number)
     * 
     * @return Should return @c true on success, @c false otherwise
     */
    virtual Bool_t operator()(UShort_t d, Char_t r, UShort_t s, UShort_t t, 
			      Float_t v);
    /** 
     * Called for each element considered an integer
     *
     * @param d   Detector number
     * @param r   Ring identifier 
     * @param s   Sector number
     * @param t   Strip number
     * @param v   Value (as an integer)
     * 
     * @return Should return @c true on success, @c false otherwise
     */
    virtual Bool_t operator()(UShort_t d, Char_t r, UShort_t s, UShort_t t, 
			      Int_t v);
    /** 
     * Called for each element considered an integer
     *
     * @param d   Detector number
     * @param r   Ring identifier 
     * @param s   Sector number
     * @param t   Strip number
     * @param v   Value (as an unsigned short integer)
     * 
     * @return Should return @c true on success, @c false otherwise
     */
    virtual Bool_t operator()(UShort_t d, Char_t r, UShort_t s, UShort_t t, 
			      UShort_t v);
    /** 
     * Called for each element considered an integer
     *
     * @param d   Detector number
     * @param r   Ring identifier 
     * @param s   Sector number
     * @param t   Strip number
     * @param v   Value (as a boolean)
     * 
     * @return Should return @c true on success, @c false otherwise
     */
    virtual Bool_t operator()(UShort_t d, Char_t r, UShort_t s, UShort_t t, 
			      Bool_t v);
  };
  /**
   * Class to print content of map 
   * 
   */
  class Printer : public ForOne
  {
  public:
    /** 
     * Constructor 
     * 
     * @param format Output format (argument to printf)
     */
    Printer(const char* format);
    /** 
     * Destructor 
     */
    virtual ~Printer() {}
    /** 
     * Print a floating point entry
     * 
     * @return true
     */
    Bool_t operator()(UShort_t d, Char_t r, UShort_t s, UShort_t t, Float_t m);
    /** 
     * Print a integer entry
     * 
     * @return true
     */
    Bool_t operator()(UShort_t d, Char_t r, UShort_t s, UShort_t t, Int_t m);
    /** 
     * Print a integer entry
     * 
     * @return true
     */
    Bool_t operator()(UShort_t d, Char_t r, UShort_t s, UShort_t t, UShort_t m);
    /** 
     * Print a boolean entry
     * 
     * @return true
     */
    Bool_t operator()(UShort_t d, Char_t r, UShort_t s, UShort_t t, Bool_t m);
  private:
    /** 
     * Copy constructor
     * 
     * @param p Object to copy from 
     */
    Printer(const Printer& p);
    /** 
     * Assignment operator
     * 
     * @return Reference to this 
     */
    Printer& operator=(const Printer&) { return *this; }
    /** 
     * Print headings 
     * 
     * @param d Current detector
     * @param r Current ring 
     * @param s Current sector
     * @param t Current strip
     */
    virtual void PrintHeadings(UShort_t d, Char_t r, UShort_t s, UShort_t t);
    /** Printf like format */
    const char* fFormat;
    /** Last detector */
    UShort_t    fOldD;
    /** Last ring */
    Char_t      fOldR;
    /** Last sector */
    UShort_t    fOldS;
  };

  /** 
   * Constructor 
   * 
   * @param maxDet  Maximum allowed detector number
   * @param maxRing Maximum number of rings
   * @param maxSec  Maximum number of sectors
   * @param maxStr  Maximum number of strips
   */
  AliFMDMap(UShort_t maxDet = 0, 
	    UShort_t maxRing= 0, 
	    UShort_t maxSec = 0, 
	    UShort_t maxStr = 0);
  /** 
   * Copy constructor
   * 
   * @param other Object to construct from
   */  
  AliFMDMap(const AliFMDMap& other);
  /** Destructor */
  virtual ~AliFMDMap() {}
  /** @return  Maximum detector number */
  UShort_t MaxDetectors() const { return fMaxDetectors==0 ?   3 :fMaxDetectors;}
  /** @return  Maximum number of rings */
  UShort_t MaxRings()     const { return fMaxRings    ==0 ?   2 :fMaxRings; }
  /** @return  Maximum number of sectors */
  UShort_t MaxSectors()   const { return fMaxSectors  ==0 ?  40 :fMaxSectors; }
  /** @return  Maximum number of strip */
  UShort_t MaxStrips()    const { return fMaxStrips   ==0 ? 512 :fMaxStrips; }
  /** 
   * Calculate the detector coordinates for a given index number
   * 
   * @param idx  Index to find cooresponding detector coordinates for
   * @param det  On return, contain the detector number 
   * @param ring On return, contain the ring identifier
   * @param sec  On return, contain the sector number
   * @param str  On return, contain the strip number 
   */
  void CalcCoords(Int_t     idx, 
		  UShort_t& det, 
		  Char_t&   ring, 
		  UShort_t& sec, 
		  UShort_t& str) const;
  /** 
   * Calculate index and return 
   *
   * @param det  Detector number
   * @param ring Ring identifier 
   * @param sec  Sector number 
   * @param str  Strip number 
   * 
   * @return  Index (not checked) 
   */
  Int_t CalcIndex(UShort_t det, Char_t ring, 
		  UShort_t sec, UShort_t str) const;
  /** 
   * Calculate index and return 
   *
   * @param det  Detector number
   * @param ring Ring identifier 
   * @param sec  Sector number 
   * @param str  Strip number 
   * 
   * @return  Index or -1 if the coordinates are invalid
   */
  Int_t  CheckIndex(UShort_t det, Char_t ring, 
		    UShort_t sec, UShort_t str) const;
  /** 
   * Check if we need UShort_t hack 
   * 
   * @param file File this object was read from 
   */
  void CheckNeedUShort(TFile* file);
  /** 
   * Right multiplication operator
   * 
   * @param o Other map to multuiply with
   * 
   * @return Reference to this object
   */
  AliFMDMap& operator*=(const AliFMDMap& o);
  /** 
   * Right division operator
   * 
   * @param o Other map to divide with
   * 
   * @return Reference to this object
   */
  AliFMDMap& operator/=(const AliFMDMap& o);
  /** 
   * Right addision operator
   * 
   * @param o Other map to add to this
   * 
   * @return Reference to this object
   */
  AliFMDMap& operator+=(const AliFMDMap& o);
  /** 
   * Right subtraction operator
   * 
   * @param o Other map to substract from this
   * 
   * @return Reference to this object
   */
  AliFMDMap& operator-=(const AliFMDMap& o);
  /** 
   * For each element of the map, call the user defined overloaded
   * @c ForOne::operator() 
   * 
   * @param algo Algorithm to use
   * 
   * @return @c true on success, @c false if for any element, @c
   * algo(d,r,s,t,v) returns false. 
   */
  virtual Bool_t  ForEach(ForOne& algo) const;
  /** 
   * Get the total size of the internal array - that is the maximum
   * index possible plus one. 
   * 
   * @return maximum index, plus 1
   */
  virtual Int_t MaxIndex() const = 0;
  /** 
   * Print content of the map 
   * 
   * @param option If not null or empty string, print map 
   */
  virtual void Print(Option_t* option="") const;
  /** 
   * Virtal function to get the value at index @a idx as a floating
   * point number.  
   * 
   * @note 
   * Even if the map is not floating point valued the
   * sub-class can define this member function to allow non l-value
   * usage of the data in the map.   That is, if @c a is a floating
   * point valued map, and @c b is not, then if the class @c B of @c b 
   * implements this member function, expression like 
   *
   * @verbatim
   *   a += b;
   * @endverbatim 
   *
   * multiplies each element of @c a (floating points) with each
   * element of @c c according to the definition of @c B::AtAsFloat
   * (@c const version).   
   *
   * @param idx Index number
   * 
   * @return Value at index as a floating point number 
   */
  virtual Float_t AtAsFloat(Int_t idx) const;
  /** 
   * Virtal function to get the value at index @a idx as a floating
   * point number.  
   * 
   * This member function should only be defined if the map is a
   * floating point valued, and can be assigned floating point valued
   * values.  That is, if the map's elements are anything but @c
   * float then this member function should not be defined in the
   * derived class.  That will prevent expressions like 
   *
   * @code 
   *   a += b;
   * @endcode
   * 
   * where @c a is non-floating point valued, and @c b is floating
   * point valued (only).
   *
   * @param idx Index number
   * 
   * @return Value at index as a floating point number 
   */
  virtual Float_t& AtAsFloat(Int_t idx);
  /** 
   * Virtal function to get the value at index @a idx as an integer
   * 
   * @see Float_t AliFMDMap::AtAsFloat(Int_t) const
   *
   * @param idx Index number
   * 
   * @return Value at index as an integer
   */
  virtual Int_t   AtAsInt(Int_t idx) const;
  /** 
   * Virtal function to get the value at index @a idx as an integer
   * 
   * @see Float_t& AliFMDMap::AtAsFloat(Int_t)
   *
   * @param idx Index number
   * 
   * @return Value at index as an integer
   */
  virtual Int_t&   AtAsInt(Int_t idx);
  /** 
   * Virtal function to get the value at index @a idx as an boolean  
   * 
   * @see Float_t AliFMDMap::AtAsFloat(Int_t) const
   *
   * @param idx Index number
   * 
   * @return Value at index as a boolean
   */
  virtual UShort_t   AtAsUShort(Int_t idx) const;
  /** 
   * Virtal function to get the value at index @a idx as an boolean
   * 
   * @see Float_t& AliFMDMap::AtAsFloat(Int_t)
   *
   * @param idx Index number
   * 
   * @return Value at index as a boolean
   */
  virtual UShort_t&   AtAsUShort(Int_t idx);
  /** 
   * Virtal function to get the value at index @a idx as an unsigned
   * short integer  
   * 
   * @see Float_t AliFMDMap::AtAsFloat(Int_t) const
   *
   * @param idx Index number
   * 
   * @return Value at index as an unsigned short integer
   */
  virtual Bool_t   AtAsBool(Int_t idx) const;
  /** 
   * Virtal function to get the value at index @a idx as an unsigned
   * short integer
   * 
   * @see Float_t& AliFMDMap::AtAsFloat(Int_t)
   *
   * @param idx Index number
   * 
   * @return Value at index as an unsigned short integer
   */
  virtual Bool_t&   AtAsBool(Int_t idx);
  /**
   * Whether this map is floating point valued - that is, it can be
   * assigned floating point valued values.
   * 
   * @return @c true if the map is floating point valued 
   */
  virtual Bool_t IsFloat() const { return kFALSE; }
  /**
   * Whether this map is floating point valued or integer valued -
   * that is, it can be assigned integer valued values
   * 
   * @return @c true if the map is integer valued 
   */
  virtual Bool_t IsInt() const { return kFALSE; }
  /**
   * Whether this map is unsigned short integer valued - that is it
   * can be assigned unsigned short integer values
   * 
   * @return @c true if the map is unsigned short integer valued 
   */
  virtual Bool_t IsUShort() const { return kFALSE; }
  /**
   * Whether this map is boolean valued - that is it can be assigned
   * boolean values
   * 
   * @return @c true if the map is boolean valued 
   */
  virtual Bool_t IsBool() const { return kFALSE; }
  /** 
   * Get raw data pointer. 
   * 
   * @return Raw data pointer 
   */
  virtual void* Ptr() const = 0;
  enum {
    /** In case of version 2 of this class, this bit should be set. */
    kNeedUShort = 14
  };
protected:
  /** 
   * Calculate, check, and return index for strip.  If the index is
   * invalid, -1 is returned
   * 
   * This is only used when a full map is used, signalled by
   * fMaxDetector = 0
   * 
   * @param det   Detector number
   * @param ring  Ring identifier
   * @param sec   Sector number 
   * @param str   Strip number
   * 
   * @return  Unique index, or -1 in case of errors 
   */
  Int_t  Coords2Index(UShort_t det, Char_t ring, 
		  UShort_t sec, UShort_t str) const;
  /** 
   * Calculate, check, and return index for strip.  If the index is
   * invalid, -1 is returned
   * 
   * This is for back-ward compatibility and for when a map does not
   * cover all of the FMD strips
   * 
   * @param det   Detector number
   * @param ring  Ring identifier
   * @param sec   Sector number 
   * @param str   Strip number
   *
   * @return  Unique index, or -1 in case of errors 
   */
  Int_t  Coords2IndexOld(UShort_t det, Char_t ring, 
			 UShort_t sec, UShort_t str) const;
  /** 
   * Calculate the detector coordinates from an array index. 
   * 
   * This is used for backward compatibility and for when a map does not
   * cover all of the FMD strips
   * 
   * @param idx  Index to convert
   * @param det  Detector number on return
   * @param ring Ring identifier on return
   * @param sec  Sector number on return
   * @param str  Strip number on return
   */
  void Index2CoordsOld(Int_t     idx, 
		       UShort_t& det, 
		       Char_t&   ring, 
		       UShort_t& sec, 
		       UShort_t& str) const;
  /** 
   * Calculate the detector coordinates from an array index. 
   * 
   * This is used for a full map only, signalled by fMaxDetector = 0
   * 
   * @param idx  Index to convert
   * @param det  Detector number on return
   * @param ring Ring identifier on return
   * @param sec  Sector number on return
   * @param str  Strip number on return
   */
  void Index2Coords(Int_t     idx, 
		    UShort_t& det, 
		    Char_t&   ring, 
		    UShort_t& sec, 
		    UShort_t& str) const;
  UShort_t fMaxDetectors;             // Maximum # of detectors
  UShort_t fMaxRings;                 // Maximum # of rings
  UShort_t fMaxSectors;               // Maximum # of sectors
  UShort_t fMaxStrips;                // Maximum # of strips

  ClassDef(AliFMDMap, 4) // Cache of per strip information
};

inline Float_t
AliFMDMap::AtAsFloat(Int_t) const
{
  return 0;
}
inline Float_t&
AliFMDMap::AtAsFloat(Int_t)
{
  static Float_t d;
  return d;
}
inline Int_t
AliFMDMap::AtAsInt(Int_t) const
{
  return 0;
}
inline Int_t&
AliFMDMap::AtAsInt(Int_t)
{
  static Int_t d;
  return d;
}
inline UShort_t
AliFMDMap::AtAsUShort(Int_t) const
{
  return 0;
}
inline UShort_t&
AliFMDMap::AtAsUShort(Int_t)
{
  static UShort_t d;
  return d;
}
inline Bool_t
AliFMDMap::AtAsBool(Int_t) const
{
  return kFALSE;
}
inline Bool_t&
AliFMDMap::AtAsBool(Int_t)
{
  static Bool_t d;
  return d;
}
inline Bool_t
AliFMDMap::ForOne::operator()(UShort_t, Char_t, UShort_t, UShort_t, Float_t)
{
  return kTRUE;
}
inline Bool_t
AliFMDMap::ForOne::operator()(UShort_t, Char_t, UShort_t, UShort_t, Int_t)
{
  return kTRUE;
}
inline Bool_t
AliFMDMap::ForOne::operator()(UShort_t, Char_t, UShort_t, UShort_t, UShort_t)
{
  return kTRUE;
}
inline Bool_t
AliFMDMap::ForOne::operator()(UShort_t, Char_t, UShort_t, UShort_t, Bool_t)
{
  return kTRUE;
}



#endif 
//____________________________________________________________________
//
// Local Variables:
//   mode: C++
// End:
//
// EOF
//


