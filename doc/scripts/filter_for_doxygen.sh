# Filtering out the HTML comments hiding doxygen keywords from Markdown
# I. Hrivnacova 25/03/2019
#
sed -e '/<!-- For Doxygen/d' -e '/---- end Doxygen -->/d' $1
