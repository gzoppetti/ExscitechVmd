
echo "Be sure you've installed tcl8.5-dev"
echo "  and tk8.5-dev"
echo "  and libnetcdf-dev"
echo "  and libfltk-dev"

export TCL_INCLUDE_DIR="/usr/include/"

currentDirectory=`pwd`
export PLUGINDIR="${currentDirectory}/../vmd-1.8.7/plugins"

cd ../plugins
make LINUXAMD64 TCLINC=-I${TCL_INCLUDE_DIR}
make distrib

cd ../vmd-1.8.7
cp ./src/Makefile ./src/MakefileCOPY

./configure LINUXAMD64 OPENGL TK FLTK IMD SILENT TCL PTHREADS NETCDF

mkdir ../vmd-1.8.7/LINUXAMD64/Exscitech
mkdir ../vmd-1.8.7/LINUXAMD64/Exscitech/Display
mkdir ../vmd-1.8.7/LINUXAMD64/Exscitech/Games
mkdir ../vmd-1.8.7/LINUXAMD64/Exscitech/Games/JobSubmitGame
mkdir ../vmd-1.8.7/LINUXAMD64/Exscitech/Games/LindseyGame
mkdir ../vmd-1.8.7/LINUXAMD64/Exscitech/Games/IdentificationGame
mkdir ../vmd-1.8.7/LINUXAMD64/Exscitech/Graphics
mkdir ../vmd-1.8.7/LINUXAMD64/Exscitech/Graphics/Animation
mkdir ../vmd-1.8.7/LINUXAMD64/Exscitech/Graphics/Instancing
mkdir ../vmd-1.8.7/LINUXAMD64/Exscitech/Graphics/Lighting
mkdir ../vmd-1.8.7/LINUXAMD64/Exscitech/Graphics/Mesh
mkdir ../vmd-1.8.7/LINUXAMD64/Exscitech/Graphics/Shaders
mkdir ../vmd-1.8.7/LINUXAMD64/Exscitech/Math
mkdir ../vmd-1.8.7/LINUXAMD64/Exscitech/Physics
mkdir ../vmd-1.8.7/LINUXAMD64/Exscitech/Physics/Bullet
mkdir ../vmd-1.8.7/LINUXAMD64/Exscitech/Utilities

touch ../vmd-1.8.7/src/Makedata.depend

rm ../vmd-1.8.7/src/Makefile
mv ../vmd-1.8.7/src/MakefileCOPY ../vmd-1.8.7/src/Makefile

echo "The original Makefile in vmd-<version> may have been replaced by configure."
echo "Please ensure that your makefile is indeed the ExSciTech makefile.  Replace it from the latest in the repository if you are unsure."

