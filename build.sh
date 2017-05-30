cython3 bb.pyx --embed
export CFLAGS="/usr/lib/python3.6/site-packages/numpy/core/include $CFLAGS"
# gcc -I /usr/include/python3.6m/ *.c -lpython3.6m
