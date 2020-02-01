# Filtering out the HTML comments hiding doxygen keywords from Markdown
# I. Hrivnacova 25/03/2019
#
sed -e '/<!-- doxy/d' -e '/\/doxy -->/d' -e 's/```c++/~~~{.cpp}/g; s/```bash/~~~{.sh}/g; s/```/~~~/g;' "$1"

# Previous instructions applied in .travis.yml
# git grep -l '^```[a-zA-Z]' | xargs sed -i .old -e 's|```\([a-zA-Z][a-zA-Z]*\)|\n```{.\1}\n|g;s|[.]bash|.sh|g;s|```|~~~~~~~|g'
# find . -name "*.old" -delete
