<!-- doxy
\page refDetectorsMUONMCHROFFiltering ROF Filtering
/doxy -->

# MCH ROF Filtering

Utilities to filter out MCH [ROFs](/DataFormatsMCH/include/DataFormatsMCH/ROFRecord.h)

So far only one filter is defined : `isTrackable` which determines if a
 (e.g. digit) ROF has the minimum required information to get any chance of
 producing tracks, depending on the number of items (e.g. digits) per chamber.
