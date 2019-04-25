# Filtering out the HTML comments hiding doxygen keywords from Markdown
# I. Hrivnacova 25/03/2019
#
sed -e '/<!-- doxy/d' -e '/\/doxy -->/d' "$1"
