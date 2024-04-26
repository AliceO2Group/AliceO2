// Copyright 2023-2099 CERN and copyright holders of ALICE O2.
// See https://alice-o2.web.cern.ch/copyright for details of the copyright holders.
// All rights not expressly granted are reserved.
//
// This software is distributed under the terms of the GNU General Public
// License v3 (GPL Version 3), copied verbatim in the file "COPYING".
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

/// @file   AODMcProducerHelpers.h
/// @author Christian Holm Christensen <cholm@nbi.dk>
/// common helpers for AOD MC producers

#ifndef O2_AODMCPRODUCER_HELPERS
#define O2_AODMCPRODUCER_HELPERS
#include <SimulationDataFormat/MCEventHeader.h>
#include <SimulationDataFormat/MCTrack.h>
#include <Framework/AnalysisDataModel.h>
#include <Framework/FunctionalHelpers.h>
#include <Framework/TableBuilder.h>
#include <string>
#include <vector>
#include <unordered_map>

/**
 * Utilities to transform simulated data into AO2D tables.
 *
 * The function templates below are templated on the cursor type over
 * the relevant AOD tables.  Such a table can be obtained from the
 * ProcessingContext @c pc
 *
 * @code
 * auto builder = pc.make<TableBulder>(OutputForTable<Table>::ref());
 * auto cursor  = builder->cursor<Table>();
 * @endcode
 *
 * If the task uses the @c Produces<Table> template,
 *
 * @code
 * Produces<Table> mTable;
 * @endcode
 *
 * then a cursor is obtained via,
 *
 * @code
 * auto cursor = mTable.cursor;
 * @endcode
 *
 * Note that these functions cannot be moved into a compilation unit,
 * because that would require deducing the table cursor type, by
 * f.ex.
 *
 * @code
 * template <typename Table>
 * struct TableCursor {
 *    using cursor_t = decltype(std::declval<framework::TableBuilder>()
 *                              .cursor<Table>());
 * };
 * using CollisionCursor = TableCursor<aod::McCollisions>:cursor_t;
 * @endcode
 *
 * but since cursors are really Lambdas and Lambda types are specific
 * to the compilation unit, then the implementation file (compilation
 * unit) of these functions definitions and their use (another
 * compilation unit) would have different types of the the cursers,
 * and thus not be able to link.  More information is given at
 * https://stackoverflow.com/questions/50033797.
 */
namespace o2::aodmchelpers
{
//==================================================================
/**
 * Deduce cursor type and wrap in std::function
 */
template <typename Table>
struct TableCursor {
  using type = decltype(framework::FFL(std::declval<framework::TableBuilder>()
                                         .cursor<Table>()));
};
//==================================================================
/** Cursor over aod::McCollisions */
using CollisionCursor = TableCursor<aod::McCollisions>::type;
/** Cursor over aod::McParticles */
using ParticleCursor = TableCursor<aod::StoredMcParticles_001>::type;
/** Cursor over aod::HepMCXSections */
using XSectionCursor = TableCursor<aod::HepMCXSections>::type;
/** Cursor over aod::HepMCPdfInfos */
using PdfInfoCursor = TableCursor<aod::HepMCPdfInfos>::type;
/** Cursor over aod::HepMCHeavyIons */
using HeavyIonCursor = TableCursor<aod::HepMCHeavyIons>::type;
//==================================================================
/** Types of updates on HepMC tables. */
enum HepMCUpdate {
  never,
  always,
  anyKey,
  allKeys
};

//==================================================================
/**
 * Check if header has keys.  If the argument @a anyNotAll is true,
 * then this member function returns true if @e any of the keys
 * were found.  If @a anyNotAll is false, then return true only if
 * @a all keys were found.
 *
 * @param header    MC event header
 * @param keys      Keys to look for
 * @param anyNotAll If true, return true if @e any key was found.
 *                  If false, return true only if @a all keys were found
 *
 * @return true if any or all keys were found
 */
bool hasKeys(o2::dataformats::MCEventHeader const& header,
             const std::vector<std::string>& keys,
             bool anyNotall = true);
//--------------------------------------------------------------------
/**
 * Get a property from the header, or if not set or not valid, a
 * default value.
 *
 * @param header  The MC event header
 * @param key     Key to look for
 * @param def     Value to return if key is not found
 *
 * @return Value of key or def if key is not found
 */
template <typename T>
const T getEventInfo(o2::dataformats::MCEventHeader const& header,
                     std::string const& key,
                     T const& def)
{
  if (not header.hasInfo(key))
    return def;

  bool isValid = false;
  const T& val = header.getInfo<T>(key, isValid);
  if (not isValid)
    return def;

  return val;
}
//====================================================================
/**
 * Fill in collision information.  This is read from the passed MC
 * header and stored in the MCCollision table.  The member function
 * returns the encoded generator ID.
 *
 * @param cursor      Cursor over o2::aod::McCollisions table
 * @param bcId        Bunch-crossing Identifier
 * @param time        Time of collisions
 * @param header      Event header from generator
 * @param generatorId Default generator
 * @param sourceId    Identifier of source
 *
 * @return encoded generator ID
 */
short updateMCCollisions(const CollisionCursor& cursor,
                         int bcId,
                         float time,
                         o2::dataformats::MCEventHeader const& header,
                         short generatorId = 0,
                         int sourceId = 0,
                         unsigned int mask = 0xFFFFFFF0);
//--------------------------------------------------------------------
/**
 * Fill in HepMC cross-section table from event generator header.
 *
 * @param cursor      Cursor over o2::aod::HepMCXSections table
 * @param collisionID Identifier of collision (as given updateMCCollision)
 * @param generatorID Encoded generator ID
 * @param header      Event header from generator
 * @param anyNotAll   If true, then any key present trigger and update.
 *                    If false, then all keys must be present to update
 *                    the table.
 *
 * @return true if table was updated
 */
bool updateHepMCXSection(const XSectionCursor& cursor,
                         int collisionID,
                         short generatorID,
                         o2::dataformats::MCEventHeader const& header,
                         HepMCUpdate when = HepMCUpdate::anyKey);
//--------------------------------------------------------------------
/**
 * Fill in HepMC parton distribution function table from event
 * generator header
 *
 * @param cursor      Cursor over o2::aod::HepMCXSections table
 * @param collisionID Identifier of collision (as given updateMCCollision)
 * @param generatorID Encoded generator ID
 * @param header      Event header from generator
 * @param anyNotAll   If true, then any key present trigger and update.
 *                    If false, then all keys must be present to update
 *                    the table.
 *
 * @return true if table was updated
 */
bool updateHepMCPdfInfo(const PdfInfoCursor& cursor,
                        int collisionID,
                        short generatorID,
                        o2::dataformats::MCEventHeader const& header,
                        HepMCUpdate when = HepMCUpdate::anyKey);
//--------------------------------------------------------------------
/**
 * Fill in HepMC heavy-ion table from generator event header.
 *
 * @param cursor      Cursor over o2::aod::HepMCXSections table
 * @param collisionID Identifier of collision (as given updateMCCollision)
 * @param generatorID Encoded generator ID
 * @param header      Event header from generator
 * @param anyNotAll   If true, then any key present trigger and update.
 *                    If false, then all keys must be present to update
 *                    the table.
 *
 * @return true if table was updated
 */
bool updateHepMCHeavyIon(const HeavyIonCursor& cursor,
                         int collisionID,
                         short generatorID,
                         o2::dataformats::MCEventHeader const& header,
                         HepMCUpdate when = HepMCUpdate::anyKey);
//--------------------------------------------------------------------
/**
 * Type of mapping from track number to row index
 */
using TrackToIndex = std::unordered_map<int, int>;
//--------------------------------------------------------------------
/**
 * Update aod::McParticles table with information from an MC track.
 *
 * @param cursor       Cursor over aod::McParticles table
 * @param mapping      Maps track number to index in table
 * @param collisionID  Collision identifier
 * @param track        Track to update table with
 * @param tracks       List of all tracks of current collision
 * @param flags        Base flags of this track
 * @param weightMask   Mask on weight floating point value
 * @param momentumMask Mask on momentum floating point values
 * @param positionMask Mask on position floating point values
 */
void updateParticle(const ParticleCursor& cursor,
                    const TrackToIndex& toStore,
                    int collisionID,
                    o2::MCTrack const& track,
                    std::vector<MCTrack> const& tracks,
                    uint8_t flags = 0,
                    uint32_t weightMask = 0xFFFFFFF0,
                    uint32_t momentumMask = 0xFFFFFFF0,
                    uint32_t positionMask = 0xFFFFFFF0);
//--------------------------------------------------------------------
/**
 * Update aod::McParticles table with tracks from MC.
 *
 * To add particles from many events, one will do
 *
 * @code
 * TrackToIndex preselect = findMcTracksToStore(...);
 *
 * size_t offset = 0;
 * for (auto event : events)
 *    offset = updateParticles(cursor,
 *                             event.getCollisionID(),
 *                             event.getTracks(),
 *                             offset,
 *                             filter,
 *                             event.isBackground(),
 *                             preselect);
 * @endcode
 *
 * Here @a preselect must be a map from track number to a positive
 * value.  Tracks that are mapped as such in @a preselect are stored
 * in addition to other tracks selected by the function.  Note that @a
 * preselect may be empty.
 *
 * If @a filter is false, then @a all tracks will be stored.
 *
 * If @a filter is true, then tracks that are
 *
 * - generated by the generator,
 * - physical primaries
 *   (MCTrackNavigator::isPhysicalPrimary),
 * - to be kept for physics
 *   (MCTrackNavigator::isKeepPhysics), or
 * - is listed with a positive value in @a preselect, or
 * - either a mother or daughter of one such track, then
 *
 * that track is kept
 *
 * On return, the @a preselect will map from track number (index in
 * the @a tracks container) to the table row index (including offset
 * from previous events in the same time-frame).
 *
 * @param cursor       Cursor over aod::McParticles
 * @param int          Collision identifier
 * @param tracks       List of all tracks of current collision
 * @param offset       Index just beyond last table entry
 * @param filter       Filter tracks
 * @param background   True of from background event
 * @param preselect    Mapping of preselected tracks
 * @param weightMask   Mask on weight floating point value
 * @param momentumMask Mask on momentum floating point values
 * @param positionMask Mask on position floating point values
 *
 * @return Index beyond the last particle added to table
 */
uint32_t updateParticles(const ParticleCursor& cursor,
                         int collisionID,
                         std::vector<MCTrack> const& tracks,
                         TrackToIndex& preselect,
                         uint32_t offset = 0,
                         bool filter = false,
                         bool background = false,
                         uint32_t weightMask = 0xFFFFFFF0,
                         uint32_t momentumMask = 0xFFFFFFF0,
                         uint32_t positionMask = 0xFFFFFFF0);
} // namespace o2::aodmchelpers

#endif /* O2_AODMCPRODUCER_HELPERS */
// Local Variables:
//   mode: C++
// End:
