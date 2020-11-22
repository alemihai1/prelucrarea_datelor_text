@echo off
echo.
echo parseaza toate documentele din folderul "input xml" si le salveaza in "output" cu acelasi nume
set classpath=data/FdgParserRo.jar
java -Xmx1000m -Dfile.encoding=utf-8 MaltWrapperRo.MaltParseTaggedXml "input xml" data/maltmodel.mco "output"