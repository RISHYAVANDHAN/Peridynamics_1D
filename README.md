# Peridynamics_1D
1D Peridynamics code

## To run the simulation

### Linux/ WSL

```bash
git clone https://github.com/RISHYAVANDHAN/Peridynamics_1D.git
cd Peridynamics_1D
chmod +x build.sh
chmod +x peridynamics_1d.sh
./build.sh
./peridynamics_1d.sh
```

Please wait a minute after running the ./build.sh, as it also installs Eigen for you in case you don't have it.
If you have, please feel free to edit the CMakeLists.txt accordingly.

The 

```bash
chmod +x build.sh
chmod +x peridynamics_1d.sh
```
is only required the first time you run it, for the subsequent runs, you can directly use

```bash
./build.sh
./peridynamics_1d.sh
```

to run the simulation.
