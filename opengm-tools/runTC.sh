echo "Running test cases"
echo "Deleting old results..."
rm -rf */*.lf */*.trws */*.qpbo */*.dd 
make
echo "tc1"
time ./map_solver tc1/N3Tr1.gm tc1/N3Tr1.lf lazyflipper tc1/N3Tr1.lf_init 2 
time ./map_solver tc1/N3Tr1.gm tc1/N3Tr1.trws trws 
time ./map_solver tc1/N3Tr1.gm tc1/N3Tr1.qpbo qpbo 
time ./map_solver tc1/N3Tr1.gm tc1/N3Tr1.dd dualdecomposition
echo "tc2"
time ./map_solver tc2/1.gm tc2/1.lf lazyflipper tc2/1.lf_init 2 
time ./map_solver tc2/1.gm tc2/1.trws trws 
time ./map_solver tc2/1.gm tc2/1.qpbo qpbo 
time ./map_solver tc2/1.gm tc2/1.dd dualdecomposition
echo "tc3"
time ./map_solver tc3/gridWi5Tr1.gm tc3/gridWi5Tr1.lf lazyflipper tc3/gridWi5Tr1.lf_init 2 
time ./map_solver tc3/gridWi5Tr1.gm tc3/gridWi5Tr1.trws trws 
time ./map_solver tc3/gridWi5Tr1.gm tc3/gridWi5Tr1.qpbo qpbo 
time ./map_solver tc3/gridWi5Tr1.gm tc3/gridWi5Tr1.dd dualdecomposition
