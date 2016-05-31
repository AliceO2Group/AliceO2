//
//  Chip.h
//  ALICEO2
//
//  Created by Markus Fasel on 23.07.15.
//  Adapted from AliITSUChip by Massimo Masersa
//

#ifndef ALICEO2_ITS_CHIP_
#define ALICEO2_ITS_CHIP_

#include <exception>
#include <sstream>
#include <TObjArray.h>  // for TObjArray
#include "Rtypes.h"     // for Double_t, Int_t, ULong_t, Bool_t, etc
#include "TObject.h"    // for TObject

namespace AliceO2 { namespace ITS { class Point; }}  // lines 22-22
namespace AliceO2 { namespace ITS { class UpgradeGeometryTGeo; }}  // lines 23-23

namespace AliceO2 {

namespace ITS {

class Point;

class UpgradeGeometryTGeo;

/// @class Chip
/// @brief Container for similated points connected to a given chip
///
/// This class contains all points of an event connected to a given
/// chip identified by the chip index.
class Chip : public TObject
{
  public:

    /// @class IndexException
    /// @brief Handling discrepancies between Chip index stored in the hit
    /// and Chip index stored in the chip
    class IndexException : public std::exception
    {
      public:

        /// Default constructor
        /// Initializes indices with -1. Not to be used when throwing the
        /// exception. Use other constructor instead
        IndexException() : fDetIdChip(-1), fDetIdHit(-1)
        { }

        /// Constructor
        /// Initializing indices from chip and from hit
        /// @param indexChip Chip index stored in chip itself
        /// @param indexHit Chip index stored in the hit
        IndexException(ULong_t indexChip, ULong_t indexHit) :
          fDetIdChip(indexChip), fDetIdHit(indexHit)
        { }

        /// Destructor
        virtual ~IndexException() throw()
        { }

        /// Build error message
        /// The error message contains the indices stored in the chip and in the hit
        /// @return Error message connected to this exception
        const char *what() const throw()
        {
          std::stringstream message;
          message << "Chip ID " << fDetIdHit << " from hit different compared to this ID " << fDetIdChip;
          return message.str().c_str();
        }

        /// Get the chip index stored in the chip
        /// @return Chip index stored in the chip
        ULong_t GetChipIndexChip() const
        { return fDetIdChip; }

        /// Fet the chip index stored in the hit
        /// @return Chip index stored in the hit
        ULong_t GetChipIndexHit() const
        { return fDetIdHit; }

      private:
        ULong_t fDetIdChip;               ///< Index of the chip stored in the chip
        ULong_t fDetIdHit;                ///< Index of the chip stored in the hit
    };

    /// Default constructor
    Chip();

    /// Main constructor
    /// @param chipindex Index of the chip
    /// @param geometry Geometry of the ITS
    Chip(Int_t index, UpgradeGeometryTGeo *geo);

    /// Copy constructor
    /// @param ref Reference for the copy
    Chip(const Chip &ref);

    /// Assignment operator
    /// @param ref Reference for assignment
    /// @return Chip after assignment
    Chip &operator=(const Chip &ref);

    /// Access operator for point inside chip
    /// @param index Index of the point within the chip
    /// @return Point at the given index (nullptr if out of bounds)
    Point *operator[](Int_t index) const;

    /// Comparison operator, checking for equalness
    /// Comparison done on chip index
    /// @param other Chip to compare with
    /// @return True if chip indices are the same
    Bool_t operator==(const Chip &other) const;

    /// Coparison operator, checking for uneqalness
    /// Comparison done on chip index
    /// @param other Chip to compare with
    /// @return True if chip indices are the different
    Bool_t operator!=(const Chip &other) const;

    /// Comparison operator, checking whether this chip is
    /// smaller based on the chip index
    /// @param other Chip to compare with
    /// @return True if this chip index is smaller than the other chip index
    Bool_t operator<(const Chip &other) const;

    /// Destructor
    virtual ~Chip();

    /// Empties the point container
    /// @param option unused
    virtual void Clear(Option_t *opt = "");

    /// Change the chip index
    /// @param index New chip index
    void SetChipIndex(Int_t index)
    { fChipIndex = index; }

    /// Get the chip index
    /// @return Index of the chip
    Int_t GetChipIndex() const
    { return fChipIndex; }

    /// Insert new ITS point into the Chip
    /// @param p Point to be added
    void InsertPoint(Point *p);

    /// Get the number of point assigned to the chip
    /// @return Number of points assigned to the chip
    Int_t GetNumberOfPoints() const
    { return fPoints.GetEntriesFast(); }

    /// Get the list of points assigned to this chip
    /// @return TObjArray of points
    TObjArray *GetListOfPoints()
    { return &fPoints; }

    /// Access Point assigned to chip at a given index
    /// @param index Index of the point
    /// @return Point at given index (nullptr if index is out of bounds)
    Point *GetPointAt(Int_t index) const;

    /// Get the line segment of a given point (from start to current position)
    /// in local coordinates.
    /// Function derived from AliITSUChip
    /// @param hitindex Index of the hit
    /// @param xstart X-coordinate of the start of the hit (in local coordinates)
    /// @param xpoint X-coordinate of the hit (in local coordinates)
    /// @param ystart Y-coordinate of the start of the hit (in local coordinates)
    /// @param ypoint Y-coordinate of the hit (in local coordinates)
    /// @param zstart Z-coordinate of the start of the hit (in local coordinates)
    /// @param zpoint Z-coordinate of the hit (in local coordinates)
    /// @param timestart Start time of the hit
    /// @param eloss Energy loss during the hit
    Bool_t LineSegmentLocal(Int_t hitindex, Double_t &xstart, Double_t &xpoint, Double_t &ystart, Double_t &ypoint,
                            Double_t &zstart, Double_t &zpoint, Double_t &timestart, Double_t &eloss) const;

    /// Get the line segment of a given point (from start to current position)
    /// in global coordinates.
    /// Function derived from AliITSUChip
    /// @TODO: Change function to return a complete space point
    /// @param hitindex Index of the hit
    /// @param xstart X-coordinate of the start of the hit (in global coordinates)
    /// @param xpoint X-coordinate of the hit (in global coordinates)
    /// @param ystart Y-coordinate of the start of the hit (in global coordinates)
    /// @param ypoint Y-coordinate of the hit (in global coordinates)
    /// @param zstart Z-coordinate of the start of the hit (in global coordinates)
    /// @param zpoint Z-coordinate of the hit (in global coordinates)
    /// @param timestart Start time of the hit
    /// @param eloss Energy loss during the hit
    Bool_t LineSegmentGlobal(Int_t hitindex, Double_t &xstart, Double_t &xpoint, Double_t &ystart, Double_t &ypoint,
                             Double_t &zstart, Double_t &zpoint, Double_t &timestart, Double_t &eloss) const;

    /// Calculate median position of two hits
    /// @param p1 First point in the median calculation
    /// @param p2 Second point in the median calculation
    /// @param x Median x-position of the two hits in local coordinates
    /// @param y Median y-position of the two hits in local coordinates
    /// @param z Median z-position of the two hits in local coordinates
    void MedianHitLocal(const Point *p1, const Point *p2, Double_t &x, Double_t &y, Double_t &z) const;

    /// Calculate median positoin of two hits
    /// @param p1 First point in the median calculation
    /// @param p2 Second point in the median calculation
    /// @param x Median xposition of the two hits in global coordinates
    /// @param y Median xposition of the two hits in global coordinates
    /// @param z Median xposition of the two hits in global coordinates
    void MedianHitGlobal(const Point *p1, const Point *p2, Double_t &x, Double_t &y, Double_t &z) const;

    /// Calculate path length between two its points
    /// @param p1 First point for the path length calculation
    /// @param p2 Second point for the path length calculation
    /// @return path length between points
    Double_t PathLength(const Point *p1, const Point *p2) const;

  protected:

    Int_t fChipIndex;     ///< Chip ID
    TObjArray fPoints;        ///< Hits connnected to the given chip
    UpgradeGeometryTGeo *fGeometry;     ///< Geometry of the ITS

  ClassDef(Chip, 1);
};
}
}

#endif /* defined(ALICEO2_ITS_CHIP_) */
