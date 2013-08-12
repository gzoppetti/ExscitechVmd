/***************************************************************************
 *cr
 *cr            (C) Copyright 1995-2009 The Board of Trustees of the
 *cr                        University of Illinois
 *cr                         All Rights Reserved
 *cr
 ***************************************************************************/

/***************************************************************************
 * RCS INFORMATION:
 *
 *      $RCSfile: Benchmark.h,v $
 *      $Author: johns $        $Locker:  $             $State: Exp $
 *      $Revision: 1.3 $      $Date: 2009/04/29 15:42:52 $
 *
 ***************************************************************************
 * DESCRIPTION:
 *
 * Various CPU/memory subsystem benchmarking routines.
 * The peak performance numbers achieved within a VMD build can be 
 * used to determine how well the VMD build was optimized, the 
 * performance of the host CPU/memory systems, SMP scaling efficiency, etc.
 *
 * The streaming memory bandwidth tests are an alternative implementation 
 * of McCalpin's STREAM benchmark.
 *
 ***************************************************************************/


/// built-in minor variation of the STREAM benchmark
int stream_bench(int N, double *time, double *mbsec);



