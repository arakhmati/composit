### Get System Cache Sizes
```bash
getconf -a | grep CACHE
```

### Profile Cache
```bash
valgrind --tool=cachegrind <executable> [ARGS]
```