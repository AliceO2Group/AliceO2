<!-- doxy
\page refDetectorsMUONMCHRawElecMap Electronic Mapping
/doxy -->

# MCH Raw Electronic Mapping

The electronic mapping offers a few methods to convert electronic-oriented
identifiers (that are found in the raw data) into detector-oriented identifiers.

The raw data references the dual sampa chips relative to a solar board, while
for the rest of the processing we are used to reference them relative to
detection elements.

The solar board based id is represented by the class [DsElecId](include/MCHRawElecMap/DsElecId.h)
while the detection element based one is represented by the class [DsDetId](include/MCHRawElecMap/DsDetId.h)

In addition, one must know to which CRU a given solar board is connected.  This
is currently handled with the help of the
[FeeLinkId](include/MCHRawElecMap/FeeLinkId.h) class which store a pair
`(FeeId,LinkId)`, where the `FeeId` is basically `cruId * 2 + endpoint`
(remember here that each CRU has two endpoints, each handling 12 links out of
the 24 of a CRU).

As the time of this writing the electronic mapping is still a bit in flux, as :

-   the detector electronic is still being (re)installed
-   the FeeId,LinkId to Solar part is still to be implemented at Pt2

so things might evolve...

Nevertheless the current mapping is generated from a "master" (google) sheet
that is used at Pt2 for commissionning.
Another mapping (called Dummy) is also provided for testing only.

The API is to be found in [Mapper.h](include/MCHRawElecMap/Mapper.h) file.

## Generation of electronic mapping

The code generation uses the [gen.sh](src/gen.sh) script which basically loops 
on all chambers and call `elecmap.py` for each one, e.g.

```bash
./elecmap.py -gs "MCH Electronic Mapping" -s CH6R --credentials=cred.json -c CH6R
```

(for the moment a credential JSON file is required, we'll try to remove that
constraint as soon as possible)
