echo "Compiling using geant4-config..."

g++ -std=c++20 nist_export_all.cxx \
    $(geant4-config --cflags) \
    $(geant4-config --libs) \
    -O2 -o nist_export_all

echo ""
echo "Build complete."
echo "Run with:"
echo "  ./nist_export_all nist_db_all.json"