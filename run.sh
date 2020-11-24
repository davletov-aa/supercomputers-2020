#!/bin/bash -x

# for bluegene
for numthreads in 1 4;
do
    for L in 1 3.14159265358979323846;
    do
        for nprocs in 64 128 256;
        do
            for npoints in 128 256 512;
            do
                if test -f "prog_nt_"$numthreads"_L_"$L"_npr_"$nprocs"_npt_"$npoints"_nsteps_20_LT_0.001.txt";
                then
                    echo skipping;
                else
                    mpisubmit.bg -n $nprocs -w 00:05:00 -m smp -e "OMP_NUM_THREADS="$numthreads --stdout "prog_nt_"$numthreads"_L_"$L"_npr_"$nprocs"_npt_"$npoints"_nsteps_20_LT_0.001.txt" prog -- $npoints 20 $L $L $L 0.001 just_prefix;
                fi
            done
        done
    done
done


# for polus
# for L in 1 3.14159265358979323846;
# do
#     for nprocs in 10 20 40;
#     do
#         for npoints in 128 256 512;
#         do
#             if test -f "prog_nt_1_L_"$L"_npr_"$nprocs"_npt_"$npoints"_nsteps_20_LT_0.001.txt";
#             then
#                 echo skipping;
#             else
#                 mpisubmit.pl -p $nprocs -w 00:05 --stdout "prog_nt_1_L_"$L"_npr_"$nprocs"_npt_"$npoints"_nsteps_20_LT_0.001.txt" prog -- $npoints 20 $L $L $L 0.001 just_prefix;
#             fi
#         done
#     done
# done
