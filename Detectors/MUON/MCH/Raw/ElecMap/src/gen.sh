#!/bin/sh

CRED="/Users/laurent/MCH Mapping-0a71181801a9.json"
CHAMBERS="CH1R CH1L CH5R CH5L CH6R CH7R CH7L CH8L CH8R CH9L CH9R CH10R"

for chamber in $CHAMBERS; do
   echo "Generating code for ${chamber}"
   ./elecmap.py -gs "MCH Electronic Mapping" -s ${chamber} \
   --credentials="$CRED" -c \
   ${chamber}
done

rm cru.map
for chamber in $CHAMBERS; do
  echo "Generating cru map for ${chamber}"
  ./elecmap.py -gs "MCH Electronic Mapping" -s ${chamber} \
  --credentials="$CRED" \
  --cru_map ${chamber}.cru.map
  cat ${chamber}.cru.map >> cru.map
done

rm fec.map
for chamber in $CHAMBERS; do
  echo "Generating fec map for ${chamber}"
  ./elecmap.py -gs "MCH Electronic Mapping" -s ${chamber} \
  --credentials="$CRED" \
  --fec_map ${chamber}.fec.map
  cat ${chamber}.fec.map >> fec.map
done





