#include <cmath>
#include <mpi.h>
#include <iostream>
#include <fstream>
#include <cstdlib>
#include <omp.h>
#include <assert.h>
#include <cstring>


using namespace std;

#define getOffset2D(x, y, ySize) (x) * (ySize) + (y)
#define getOffset3D(x, y, z, ySize, zSize) (x) * (ySize) * (zSize) + (y) * (zSize) + (z)
#define getOffset4D(t, x, y, z, xSize, ySize, zSize) (t) * (xSize) * (ySize) * (zSize) + (x) * (ySize) * (zSize) + (y) * (zSize) + (z)

#define xFromOffset3D(offset, ySize, zSize) (offset) / ((ySize) * (zSize))
#define yFromOffset3D(offset, ySize, zSize) ((offset) % ((ySize) * (zSize))) / (zSize)
#define zFromOffset3D(offset, ySize, zSize) ((offset) % ((ySize) * (zSize))) % (zSize)

enum Boundaries {
    BOUNDARY_UPPER,
    BOUNDARY_LOWER,
    BOUNDARY_FRONT,
    BOUNDARY_BACK,
    BOUNDARY_LEFT,
    BOUNDARY_RIGHT
};

double analytical_solution(double x, double y, double z, double t, double Lx, double Ly, double Lz) {
    double at = M_PI * sqrt(1 / (Lx * Lx) + 1 / (Ly * Ly) + 1 / (Lz * Lz));
    return sin((M_PI / Lx) * x) * sin((M_PI / Ly) * y) * sin((M_PI / Lz) * z) * cos(at * t);
}


class HyperbolicPDEApproximator {
public:
    double Lx, Ly, Lz, Lt;
    double deltaX, deltaY, deltaZ, deltaT;
    double deltaX2, deltaY2, deltaZ2, deltaT2;
    int locXNumPoints, locYNumPoints, locZNumPoints;
    int xx, yy, zz;
    bool finalized, buffersInitialized, buffersAllocated;
    int numProcs, numPoints, xNumProcs, yNumProcs, zNumProcs;
    int numTSteps;
    int worldRank;
    int decart3DRank;
    int blockSize;
    double *u, *prevU, *prevPrevU;
    double *u_gold;
    double *maxErrors, maxError;

    char *prefix;

    double *upperPart, *lowerPart, *frontPart, *backPart, *leftPart, *rightPart;
    double *tempPart;
    MPI_Comm decart_grid_comm;
    int ndims, dims[3], periods[3], coords[3];
    int curTStep;

    HyperbolicPDEApproximator(
        double Lx, double Ly, double Lz, double Lt,
        int numPoints,int numTSteps, int numProcs, int worldRank, char *prefix) {

        this -> Lx = Lx;
        this -> Ly = Ly;
        this -> Lz = Lz;
        this -> Lt = Lt;
        this -> numPoints = numPoints; // number of inner points in grid at each dimention
        this -> numTSteps = numTSteps;
        this -> numProcs = numProcs; // total number of processes to calculate inner points of the grid
        this -> worldRank = worldRank;
        this -> prefix = prefix;

        u = NULL, prevU = NULL, prevPrevU = NULL;
        u_gold = NULL;
        maxErrors = NULL;
        ndims = 3;
        upperPart = NULL, lowerPart = NULL, leftPart = NULL, rightPart = NULL;
        frontPart = NULL, backPart = NULL;
        periods[0] = 0, periods[1] = 0, periods[2] = 0;

        distribAreasToProcs();
        if (!worldRank) {
            cout << "Lx: " << Lx << ", Ly: " << Ly <<
                ", Lz: " << Lz << ", Lt: " << Lt <<  endl;
            cout << "numProcs: " << numProcs << ", numPoints: " <<
                numPoints << ", numTSteps: " << numTSteps << endl;
        }
        initDeltas();
        finalized = true;
        buffersInitialized = false;
        buffersAllocated = false;
    }

    ~HyperbolicPDEApproximator() {
        if (!finalized) {
            finalize();
        }
    }

    void initDeltas() {
        deltaX = Lx / (numPoints + 1);
        deltaY = Ly / (numPoints + 1);
        deltaZ = Lz / (numPoints + 1);
        deltaT = Lt / (numTSteps - 1);
        deltaX2 = pow(deltaX, 2);
        deltaY2 = pow(deltaY, 2);
        deltaZ2 = pow(deltaZ, 2);
        deltaT2 = pow(deltaT, 2);

        if (!worldRank) {
            cout << "deltaX: " << deltaX << ", deltaY: " << deltaY <<
                ", deltaZ: " << deltaZ << ", deltaT: " << deltaT <<  endl;
            cout << "deltaX^2: " << deltaX2 << ", deltaY^2: " << deltaY2 <<
                ", deltaZ^2: " << deltaZ2 << ", deltaT^2: " << deltaT2 <<  endl;
        }
    }

    void extrPreBoundPart(double *src, double *dest, Boundaries part) {

        if (part == BOUNDARY_UPPER) {
            #pragma omp parallel for
            for (int x = 1; x < xx - 1; ++x) {
                #pragma omp parallel for
                for (int y = 1; y < yy - 1; ++y) {
                    *(dest + getOffset2D(x - 1, y - 1, locYNumPoints)) =
                        *(src + getOffset3D(x, y, zz - 2, yy, zz));
                }
            }
        } else if (part == BOUNDARY_LOWER)
        {
            #pragma omp parallel for
            for (int x = 1; x < xx - 1; ++x) {
                #pragma omp parallel for
                for (int y = 1; y < yy - 1; ++y) {
                    *(dest + getOffset2D(x - 1, y - 1, locYNumPoints)) =
                        *(src + getOffset3D(x, y, 1, yy, zz));
                }
            }
        } else if (part == BOUNDARY_LEFT)
        {
            #pragma omp parallel for
            for (int y = 1; y < yy - 1; ++y) {
                #pragma omp parallel for
                for (int z = 1; z < zz - 1; ++z) {
                    *(dest + getOffset2D(y - 1, z - 1, locZNumPoints)) =
                        *(src + getOffset3D(1, y, z, yy, zz));
                }
            }
        } else if (part == BOUNDARY_RIGHT)
        {
            #pragma omp parallel for
            for (int y = 1; y < yy - 1; ++y) {
                #pragma omp parallel for
                for (int z = 1; z < zz - 1; ++z) {
                    *(dest + getOffset2D(y - 1, z - 1, locZNumPoints)) =
                        *(src + getOffset3D(xx - 2, y, z, yy, zz));
                }
            }
        } else if (part == BOUNDARY_FRONT)
        {
            #pragma omp parallel for
            for (int x = 1; x < xx - 1; ++x) {
                #pragma omp parallel for
                for (int z = 1; z < zz - 1; ++z) {
                    *(dest + getOffset2D(x - 1, z - 1, locZNumPoints)) =
                        *(src + getOffset3D(x, yy - 2, z, yy, zz));
                }
            }
        } else {
            #pragma omp parallel for
            for (int x = 1; x < xx - 1; ++x) {
                #pragma omp parallel for
                for (int z = 1; z < zz - 1; ++z) {
                    *(dest + getOffset2D(x - 1, z - 1, locZNumPoints)) =
                        *(src + getOffset3D(x, 1, z, yy, zz));
                }
            }
        }
    }

    void initBoundPart(double *src, double *dest, Boundaries part) {
        if (part == BOUNDARY_UPPER) {
            #pragma omp parallel for
            for (int x = 1; x < xx - 1; ++x) {
                #pragma omp parallel for
                for (int y = 1; y < yy - 1; ++y) {
                    *(dest + getOffset3D(x, y, zz - 1, yy, zz)) =
                        *(src + getOffset2D(x - 1, y - 1, locYNumPoints));
                }
            }
        } else if (part == BOUNDARY_LOWER)
        {
            #pragma omp parallel for
            for (int x = 1; x < xx - 1; ++x) {
                #pragma omp parallel for
                for (int y = 1; y < yy - 1; ++y) {
                    *(dest + getOffset3D(x, y, 0, yy, zz)) =
                        *(src + getOffset2D(x - 1, y - 1, locYNumPoints));
                }
            }
        } else if (part == BOUNDARY_LEFT)
        {
            #pragma omp parallel for
            for (int y = 1; y < yy - 1; ++y) {
                #pragma omp parallel for
                for (int z = 1; z < zz - 1; ++z) {
                    *(dest + getOffset3D(0, y, z, yy, zz)) =
                        *(src + getOffset2D(y - 1, z - 1, locZNumPoints));
                }
            }
        } else if (part == BOUNDARY_RIGHT)
        {
            #pragma omp parallel for
            for (int y = 1; y < yy - 1; ++y) {
                #pragma omp parallel for
                for (int z = 1; z < zz - 1; ++z) {
                    *(dest + getOffset3D(xx - 1, y, z, yy, zz)) =
                        *(src + getOffset2D(y - 1, z - 1, locZNumPoints));
                }
            }
        } else if (part == BOUNDARY_FRONT)
        {
            #pragma omp parallel for
            for (int x = 1; x < xx - 1; ++x) {
                #pragma omp parallel for
                for (int z = 1; z < zz - 1; ++z) {
                    *(dest + getOffset3D(x, yy - 1, z, yy, zz)) =
                        *(src + getOffset2D(x - 1, z - 1, locZNumPoints));
                }
            }
        } else {
            #pragma omp parallel for
            for (int x = 1; x < xx - 1; ++x) {
                #pragma omp parallel for
                for (int z = 1; z < zz - 1; ++z) {
                    *(dest + getOffset3D(x, 0, z, yy, zz)) =
                    *(src + getOffset2D(x - 1, z - 1, locZNumPoints));
                }
            }
        }
    }

    void testBoundaryOps(double *src) {

        for (int x = 1; x < xx - 1; ++x) {
            #pragma omp parallel for
            for (int y = 1; y < yy - 1; ++y) {
                if (*(src + getOffset3D(x, y, 0, yy, zz)) != *(src + getOffset3D(x, y, 1, yy, zz))) {
                    cout << worldRank << " " << *(src + getOffset3D(x, y, 0, yy, zz)) << " " << *(src + getOffset3D(x, y, 1, yy, zz)) << endl;
                }
                if (*(src + getOffset3D(x, y, zz - 1, yy, zz)) != *(src + getOffset3D(x, y, zz - 2, yy, zz))) {
                    cout << worldRank << " " << *(src + getOffset3D(x, y, zz - 1, yy, zz)) << " " << *(src + getOffset3D(x, y, zz - 2, yy, zz)) << endl;
                }
            }
        }

        for (int x = 1; x < xx - 1; ++x) {
            #pragma omp parallel for
            for (int z = 1; z < zz - 1; ++z) {
                if (*(src + getOffset3D(x, 0, z, yy, zz)) != *(src + getOffset3D(x, 1, z, yy, zz))) {
                    cout << worldRank << " " << *(src + getOffset3D(x, 0, z, yy, zz)) << " " << *(src + getOffset3D(x, 1, z, yy, zz)) << endl;
                }
                if (*(src + getOffset3D(x, yy - 1, z, yy, zz)) != *(src + getOffset3D(x, yy - 2, z, yy, zz))) {
                    cout << worldRank << " " << *(src + getOffset3D(x, yy - 1, z, yy, zz)) << " " << *(src + getOffset3D(x, yy - 2, z, yy, zz)) << endl;
                }
            }
        }

        for (int y = 1; y < yy - 1; ++y) {
            #pragma omp parallel for
            for (int z = 1; z < zz - 1; ++z) {
                if (*(src + getOffset3D(0, y, z, yy, zz)) != *(src + getOffset3D(1, y, z, yy, zz))) {
                    cout << worldRank << " " << *(src + getOffset3D(0, y, z, yy, zz)) << " " << *(src + getOffset3D(1, y, z, yy, zz)) << endl;
                }
                if (*(src + getOffset3D(xx - 1, y, z, yy, zz)) != *(src + getOffset3D(xx - 2, y, z, yy, zz))) {
                    cout << worldRank << " " << *(src + getOffset3D(xx - 1, y, z, yy, zz)) << " " << *(src + getOffset3D(xx - 2, y, z, yy, zz)) << endl;
                }
            }
        }
    }

    void exchangeData(double *src) {
        MPI_Status status;
        int upperProc, lowerProc, leftProc, rightProc, frontProc, backProc;

        MPI_Cart_shift(decart_grid_comm, 0, 1, &leftProc, &rightProc);
        MPI_Cart_shift(decart_grid_comm, 1, 1, &backProc, &frontProc);
        MPI_Cart_shift(decart_grid_comm, 2, 1, &lowerProc, &upperProc);

        if (upperProc != MPI_PROC_NULL) {
            extrPreBoundPart(src, upperPart, BOUNDARY_UPPER);
        }
        if (lowerProc != MPI_PROC_NULL) {
            extrPreBoundPart(src, lowerPart, BOUNDARY_LOWER);
        }
        if (rightProc != MPI_PROC_NULL) {
            extrPreBoundPart(src, rightPart, BOUNDARY_RIGHT);
        }
        if (leftProc != MPI_PROC_NULL) {
            extrPreBoundPart(src, leftPart, BOUNDARY_LEFT);
        }
        if (frontProc != MPI_PROC_NULL) {
            extrPreBoundPart(src, frontPart, BOUNDARY_FRONT);
        }
        if (backProc != MPI_PROC_NULL) {
            extrPreBoundPart(src, backPart, BOUNDARY_BACK);
        }

        MPI_Sendrecv(
            rightPart, locYNumPoints * locZNumPoints, MPI_DOUBLE, rightProc, 0,
            tempPart, locYNumPoints * locZNumPoints, MPI_DOUBLE, leftProc, 0,
            decart_grid_comm, &status
        );
        MPI_Sendrecv(
            leftPart, locYNumPoints * locZNumPoints, MPI_DOUBLE, leftProc, 1,
            rightPart, locYNumPoints * locZNumPoints, MPI_DOUBLE, rightProc, 1,
            decart_grid_comm, &status
        );
        if (leftProc != MPI_PROC_NULL) {
            memcpy(leftPart, tempPart, locYNumPoints * locZNumPoints * sizeof(double));
        }

        MPI_Sendrecv(
            upperPart, locXNumPoints * locYNumPoints, MPI_DOUBLE, upperProc, 2,
            tempPart, locXNumPoints * locYNumPoints, MPI_DOUBLE, lowerProc, 2,
            decart_grid_comm, &status
        );
        MPI_Sendrecv(
            lowerPart, locXNumPoints * locYNumPoints, MPI_DOUBLE, lowerProc, 3,
            upperPart, locXNumPoints * locYNumPoints, MPI_DOUBLE, upperProc, 3,
            decart_grid_comm, &status
        );
        if (lowerProc != MPI_PROC_NULL) {
            memcpy(lowerPart, tempPart, locXNumPoints * locYNumPoints * sizeof(double));
        }

        MPI_Sendrecv(
            frontPart, locXNumPoints * locZNumPoints, MPI_DOUBLE, frontProc, 4,
            tempPart, locXNumPoints * locZNumPoints, MPI_DOUBLE, backProc, 4,
            decart_grid_comm, &status
        );
        MPI_Sendrecv(
            backPart, locXNumPoints * locZNumPoints, MPI_DOUBLE, backProc, 5,
            frontPart, locXNumPoints * locZNumPoints, MPI_DOUBLE, frontProc, 5,
            decart_grid_comm, &status
        );
        if (backProc != MPI_PROC_NULL) {
            memcpy(backPart, tempPart, locXNumPoints * locZNumPoints * sizeof(double));
        }

        if (upperProc != MPI_PROC_NULL) {
            initBoundPart(upperPart, src, BOUNDARY_UPPER);
        }
        if (lowerProc != MPI_PROC_NULL) {
            initBoundPart(lowerPart, src, BOUNDARY_LOWER);
        }
        if (rightProc != MPI_PROC_NULL) {
            initBoundPart(rightPart, src, BOUNDARY_RIGHT);
        }
        if (leftProc != MPI_PROC_NULL) {
            initBoundPart(leftPart, src, BOUNDARY_LEFT);
        }
        if (frontProc != MPI_PROC_NULL) {
            initBoundPart(frontPart, src, BOUNDARY_FRONT);
        }
        if (backProc != MPI_PROC_NULL) {
            initBoundPart(backPart, src, BOUNDARY_BACK);
        }
    }

    void initBuffers(
        double (*u_analytical)
            (double x, double y, double z, double t, double Lx, double Ly, double Lz)
                = analytical_solution) {

        if (!worldRank) {
            cout << "buffers initialization ..." << endl;
        }

        MPI_Cart_coords(decart_grid_comm, decart3DRank, ndims, coords);

        double xshift = coords[0] * locXNumPoints;
        double yshift = coords[1] * locYNumPoints;
        double zshift = coords[2] * locZNumPoints;

        #pragma omp parallel for
        for (int x = 0; x < xx; ++x) {
            #pragma omp parallel for
            for (int y = 0; y < yy; ++y) {
                #pragma omp parallel for
                for (int z = 0; z < zz; ++z) {
                    *(prevPrevU + getOffset3D(x, y, z, yy, zz)) =
                        u_analytical(
                            (xshift + x) * deltaX,
                            (yshift + y) * deltaY,
                            (zshift + z) * deltaZ, 0, Lx, Ly, Lz);
                    #pragma omp parallel for
                    for (int t = 0; t < numTSteps; ++t) {
                        *(u_gold + getOffset4D(t, x, y, z, xx, yy, zz)) = 
                            u_analytical(
                                (xshift + x) * deltaX,
                                (yshift + y) * deltaY,
                                (zshift + z) * deltaZ, t * deltaT, Lx, Ly, Lz);
                    }
                }
            }
        }
        #pragma omp parallel for
        for (int x = 1; x < xx - 1; ++x) {
            #pragma omp parallel for
            for (int y = 1; y < yy - 1; ++y) {
                #pragma omp parallel for
                for (int z = 1; z < zz - 1; ++z) {
                    *(prevU + getOffset3D(x, y, z, yy, zz)) =
                        *(prevPrevU + getOffset3D(x, y, z, yy, zz)) +
                        (deltaT2 / 2) *
                        sevenPointDiffScheme(
                            prevPrevU, x, y, z, yy, zz
                        );
                }
            }
        }

        MPI_Barrier(decart_grid_comm);
        exchangeData(prevU);
        curTStep = 2;
        finalized = false;
        buffersInitialized = true;
    }

    void allocateBuffers() {
        if (!worldRank) {
            cout << "memory allocation for buffers ..." << endl;
            cout << "local block size: " << blockSize << endl;
            cout << "allocating buffer for analytic solutuion for " << numTSteps << " steps" << endl;
        }
        u_gold = new double [blockSize * numTSteps]();
        u = new double [blockSize]();
        prevU = new double [blockSize]();
        prevPrevU = new double [blockSize]();
        maxErrors = new double [numProcs]();

        upperPart = new double [locXNumPoints * locYNumPoints]();
        lowerPart = new double [locXNumPoints * locYNumPoints]();
        frontPart = new double [locXNumPoints * locZNumPoints]();
        backPart = new double [locXNumPoints * locZNumPoints]();
        leftPart = new double [locYNumPoints * locZNumPoints]();
        rightPart = new double [locYNumPoints * locZNumPoints]();
        tempPart = new double [locXNumPoints * locXNumPoints]();

        buffersAllocated = true;
    }

    void initialize() {
        allocateBuffers();
        initBuffers();
    }

    void finalize() {
        if (!worldRank) {
            cout << "memory dismission ..." << endl;
        }
        MPI_Barrier(decart_grid_comm);
        delete [] u_gold;
        delete [] u;
        delete [] prevU;
        delete [] prevPrevU;
        delete [] maxErrors;
        delete [] upperPart;
        delete [] lowerPart;
        delete [] leftPart;
        delete [] rightPart;
        delete [] frontPart;
        delete [] backPart;
        delete [] tempPart;

        finalized = true;
    }

    void calcError(int t = -1) {

        int atTStep = t == -1 ? curTStep : t;
        double localMaxError = 0.0;
        double *uAtT = atTStep == 0 ? prevPrevU : (atTStep == 1 ? prevU : u);
        double error = 0.0;

        #pragma omp parallel for
        for (int x = 0; x < xx; ++x) {
            #pragma omp parallel for
            for (int y = 0; y < yy; ++y) {
                #pragma omp parallel for
                for (int z = 0; z < zz; ++z) {
                    if (
                        ((x == 0) && (y == 0)) ||
                        ((x == 0) && (z == 0)) ||
                        ((y == 0) && (z == 0)) ||
                        ((x == 0) && (y == yy - 1)) ||
                        ((x == 0) && (z == zz - 1)) ||
                        ((y == 0) && (x == xx - 1)) ||
                        ((y == 0) && (z == zz - 1)) ||
                        ((z == 0) && (x == xx - 1)) ||
                        ((z == 0) && (y == yy - 1)) ||
                        ((x == xx - 1) && (y == yy - 1)) ||
                        ((x == xx - 1) && (z == zz - 1)) ||
                        ((y == yy - 1) && (z == zz - 1))
                    ) {
                        continue;
                    }
                    #pragma omp critical
                    {
                        error = abs(
                            *(uAtT + getOffset3D(x, y, z, yy, zz)) -
                            *(u_gold + getOffset4D(atTStep, x, y, z, xx, yy, zz))
                        );
                        localMaxError = max(localMaxError, error);
                    }
                }
            }
        }

        MPI_Gather(&localMaxError, 1, MPI_DOUBLE,
            maxErrors, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
        if (!worldRank) {
            for (int i = 0; i < numProcs; ++i) {
                localMaxError = max(localMaxError, maxErrors[i]);
            }
            cout << "max error at timestep " << atTStep << " is " << localMaxError << endl;
        }
        MPI_Bcast(&maxError, 1, MPI_INT, 0, MPI_COMM_WORLD);
    }

    void aggregate_results() {
        // aggregate u from all blocks at current timestep and dump it to file
        int *procsCoords = new int [3 * numProcs];
        ofstream file;

        if (!worldRank) {
            double *buff = new double [blockSize * numProcs];

            MPI_Gather(coords, 3, MPI_INT, procsCoords, 3, MPI_INT, 0, MPI_COMM_WORLD);
            MPI_Gather(
                u_gold + getOffset4D(curTStep, 0, 0, 0, xx, yy, zz), blockSize, MPI_DOUBLE,
                buff, blockSize, MPI_DOUBLE,
                0, MPI_COMM_WORLD
            );

            file.open(strcat(prefix, "coords.txt"));
            for (int i = 0; i < numProcs; ++i) {
                file << procsCoords[i * 3] << " " << procsCoords[i * 3 + 1] <<
                    " " << procsCoords[i * 3 + 2] << endl;
            }
            file.close();

            file.open(strcat(prefix, "u_analytical.txt"));
            for (int i = 0; i < numProcs; ++i) {
                for (int j = 0; j < blockSize; ++j) {
                    file << buff[i * blockSize + j] << " ";
                }
                file << endl;
            }
            file.close();
            MPI_Barrier(MPI_COMM_WORLD);
            MPI_Gather(
                u, blockSize, MPI_DOUBLE,
                buff, blockSize, MPI_DOUBLE,
                0, MPI_COMM_WORLD
            );

            file.open(strcat(prefix, "u_last_step.txt"));
            for (int i = 0; i < numProcs; ++i) {
                for (int j = 0; j < blockSize; ++j) {
                    file << buff[i * blockSize + j] << " ";
                }
                file << endl;
            }
            file.close();

            delete [] buff;

        } else {
            MPI_Gather(coords, 3, MPI_INT, procsCoords, 3, MPI_INT, 0, MPI_COMM_WORLD);
            MPI_Gather(
                u_gold + getOffset4D(curTStep, 0, 0, 0, xx, yy, zz), blockSize, MPI_DOUBLE,
                prevPrevU, blockSize, MPI_DOUBLE,
                0, MPI_COMM_WORLD
            );
            MPI_Barrier(MPI_COMM_WORLD);
            MPI_Gather(
                u, blockSize, MPI_DOUBLE,
                prevPrevU, blockSize, MPI_DOUBLE,
                0, MPI_COMM_WORLD
            );
        }
        delete [] procsCoords;
    }

    void distribAreasToProcs() {
        if (!(worldRank)) {
            int numProcs = this -> numProcs;
            int numPoints = this -> numPoints;
            bool success = false;
            while (!success) {
                for (int i = 1; i < numProcs; ++i) {
                    for (int j = i; j < numProcs; ++j) {
                        for (int k = j; k < numProcs; ++k) {
                            if (
                                numPoints % i == 0 && numPoints % j == 0
                                && numPoints % k == 0 && i * j * k == numProcs
                            ) {
                                success = true;
                                xNumProcs = i;
                                yNumProcs = j;
                                zNumProcs = k;
                            }
                        }
                    }
                }
                numPoints++;
            }
            numPoints--;

            cout << "worldRank: " << worldRank << endl;
            if (this -> numPoints != numPoints) {
                cout << "number of points in grid changed from " <<
                    this -> numPoints << " to " << numPoints << endl;
            }
            this -> numPoints = numPoints;
            locXNumPoints = numPoints / xNumProcs;
            locYNumPoints = numPoints / yNumProcs;
            locZNumPoints = numPoints / zNumProcs;
            cout << "locXNumPoints: " << locXNumPoints << endl;
            cout << "locYNumPoints: " << locYNumPoints << endl;
            cout << "locZNumPoints: " << locZNumPoints << endl;
            cout << "xNumProcs: " << xNumProcs << endl;
            cout << "yNumProcs: " << yNumProcs << endl;
            cout << "zNumProcs: " << zNumProcs << endl;
        }
        MPI_Bcast(&(this -> numPoints), 1, MPI_INT, 0, MPI_COMM_WORLD);
        MPI_Bcast(&(this -> xNumProcs), 1, MPI_INT, 0, MPI_COMM_WORLD);
        MPI_Bcast(&(this -> yNumProcs), 1, MPI_INT, 0, MPI_COMM_WORLD);
        MPI_Bcast(&(this -> zNumProcs), 1, MPI_INT, 0, MPI_COMM_WORLD);
        MPI_Bcast(&(this -> locXNumPoints), 1, MPI_INT, 0, MPI_COMM_WORLD);
        MPI_Bcast(&(this -> locYNumPoints), 1, MPI_INT, 0, MPI_COMM_WORLD);
        MPI_Bcast(&(this -> locZNumPoints), 1, MPI_INT, 0, MPI_COMM_WORLD);
        xx = locXNumPoints + 2; yy = locYNumPoints + 2; zz = locZNumPoints + 2;
        blockSize = xx * yy * zz;

        dims[0] = xNumProcs;
        dims[1] = yNumProcs;
        dims[2] = zNumProcs;

        MPI_Cart_create(MPI_COMM_WORLD, ndims, dims, periods, 1, &decart_grid_comm);
        MPI_Comm_rank(decart_grid_comm, &decart3DRank);
        MPI_Barrier(MPI_COMM_WORLD);
    }

    void calcCurTStepLocally() {
        if (!buffersInitialized) {
            throw "Buffers are not initialized. Initialize them first.";
        }
        #pragma omp parallel for
        for (int x = 1; x < xx - 1; ++x) {
            #pragma omp parallel for
            for (int y = 1; y < yy - 1; ++y) {
                #pragma omp parallel for
                for (int z = 1; z < zz - 1; ++z) {
                    *(u + getOffset3D(x, y, z, yy, zz)) =
                        deltaT2 * sevenPointDiffScheme(prevU, x, y, z, yy, zz) +
                        2 * *(prevU + getOffset3D(x, y, z, yy, zz)) -
                        *(prevPrevU + getOffset3D(x, y, z, yy, zz));
                }
            }
        }
    }

    double sevenPointDiffScheme(
        double *src, int x, int y, int z, int ySize, int zSize) {
        double res = 
            (1 / deltaX2) * (
                *(src + getOffset3D(x - 1, y, z, ySize, zSize)) -
                2 * *(src + getOffset3D(x, y, z, ySize, zSize)) +
                *(src + getOffset3D(x + 1, y, z, ySize, zSize))
            ) +
            (1 / deltaY2) * (
                *(src + getOffset3D(x, y - 1, z, ySize, zSize)) -
                2 * *(src + getOffset3D(x, y, z, ySize, zSize)) +
                *(src + getOffset3D(x, y + 1, z, ySize, zSize))
            ) +
            (1 / deltaZ2) * (
                *(src + getOffset3D(x, y, z - 1, ySize, zSize)) -
                2 * *(src + getOffset3D(x, y, z, ySize, zSize)) +
                *(src + getOffset3D(x, y, z + 1, ySize, zSize))
            );
        return res;
    }

    double calculate() {
        double elapsed_time = 0, intermediate_time;
        calcError(0);
        calcError(1);
        if (!worldRank) {
         cout << "calculation ..." << endl;
        }
        for (int timestep = 2; timestep <  numTSteps; timestep++) {
            intermediate_time = MPI_Wtime();
            calcCurTStepLocally();
            MPI_Barrier(decart_grid_comm);
            elapsed_time += MPI_Wtime() - intermediate_time;
            intermediate_time = MPI_Wtime();
            exchangeData(u);
            elapsed_time += MPI_Wtime() - intermediate_time;
            calcError();
            // if (timestep == numTSteps - 1) {
            //     aggregate_results();
            // }
            intermediate_time = MPI_Wtime();
            performStep();
            MPI_Barrier(decart_grid_comm);
            elapsed_time += MPI_Wtime() - intermediate_time;
        }

        return elapsed_time;
    }

    void performStep() {
        curTStep++;
        memcpy(prevPrevU, prevU, blockSize * sizeof(double));
        memcpy(prevU, u, blockSize * sizeof(double));
    }
};

int main(int argc, char **argv)
{
    int numProcs;
    int worldRank;
    MPI_Init(&argc, &argv);
    MPI_Comm_size(MPI_COMM_WORLD, &numProcs);
    MPI_Comm_rank(MPI_COMM_WORLD, &worldRank);

    if (argc != 8 && !worldRank) {
        cout << "Usage: ./" << argv[0] << " numPoints numTSteps Lx Ly Lz Lt savePrefix" << endl;
    } else if (argc == 8) {
        int numPoints, numTSteps;
        double Lx, Ly, Lz, Lt;
        numPoints = atoi(argv[1]);
        numTSteps = atoi(argv[2]);
        Lx = atof(argv[3]);
        Ly = atof(argv[4]);
        Lz = atof(argv[5]);
        Lt = atof(argv[6]);
        char *prefix = argv[7];

        HyperbolicPDEApproximator process = HyperbolicPDEApproximator(
            Lx, Ly, Lz, Lt, numPoints, numTSteps, numProcs, worldRank, prefix);

        double elapsed_time;
        elapsed_time = MPI_Wtime();
        process.initialize();
        elapsed_time = MPI_Wtime() - elapsed_time;
        elapsed_time += process.calculate();

        if (!worldRank) {
            cout << "elapsed time: " << elapsed_time << " seconds" << endl;
        }
        process.finalize();
    }

    MPI_Finalize();
    return 0;
}
