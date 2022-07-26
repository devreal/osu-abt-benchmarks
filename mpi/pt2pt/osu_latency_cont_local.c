#define BENCHMARK "OSU MPI%s Latency Test"
/*
 * Copyright (C) 2002-2019 the Network-Based Computing Laboratory
 * (NBCL), The Ohio State University. 
 *
 * Contact: Dr. D. K. Panda (panda@cse.ohio-state.edu)
 *
 * For detailed copyright and licensing information, please refer to the
 * copyright file COPYRIGHT in the top level OMB directory.
 */

#define USE_PAPI 1

#include <osu_util_mpi.h>
#include <mpi-ext.h>

#ifdef USE_PAPI
#include <papi.h>

typedef
struct hwc_cnt {
  long_long ins; // instructions
  long_long cyc; // cycles
} hwc_cnt_t;

static int EventSet=PAPI_NULL;
static void hwc_init()
{
  int retval = PAPI_library_init(PAPI_VER_CURRENT);
  if (retval != PAPI_VER_CURRENT && retval > 0) {
    fprintf(stderr, "PAPI version mismatch\n");
  }
  else if (retval < 0) {
    fprintf(stderr, "PAPI init failed\n");
  }
  retval = PAPI_create_eventset(&EventSet);
  printf("PAPI_create_eventset returned %d\n", retval);
  if (retval != PAPI_OK) fprintf(stderr, "Failed to create event set!\n");
  if (PAPI_OK != (retval = PAPI_add_event(EventSet, PAPI_TOT_INS))) {
    fprintf(stderr, "Could not add PAPI_TOT_INS to event set (%d)!\n", retval);
  }
  if (PAPI_OK != (retval = PAPI_add_event(EventSet, PAPI_TOT_CYC))) {
    fprintf(stderr, "Could not add PAPI_TOT_CYC to event set (%d)!\n", retval);
  }
  if (PAPI_start(EventSet) != PAPI_OK) {
    fprintf(stderr, "Could not start event set!\n");
  }
}

static void hwc_fini()
{
  long_long values[2];
  PAPI_stop(EventSet, values);
}

// return the current instruction counter value
static inline void hwc_ins(hwc_cnt_t *cnt)
{
  long long vals[2];
  PAPI_read(EventSet, vals);
  cnt->ins = vals[0]; cnt->cyc = vals[1];
  //printf("Read: %lld instructions, %lld cycles\n", vals[0], vals[1]);
}

#endif // USE_PAPI

static char *s_buf, *r_buf;
static int size;

typedef enum cont_cb_dir_t {
  CONT_CB_SEND,
  CONT_CB_RECV
} cont_cb_dir_t;

static int cont_cb(int status, void *data)
{
	return MPI_SUCCESS;
}

int
main (int argc, char *argv[])
{
    int myid, numprocs, i;
    MPI_Status reqstat;
    MPI_Request cont_req;
    double t_start = 0.0, t_end = 0.0;
    int po_ret = 0;
    options.bench = PT2PT;
    options.subtype = LAT;

    set_header(HEADER);
    set_benchmark_name("osu_latency");
#ifdef USE_PAPI
    hwc_cnt_t hwc_start, hwc_end;
    hwc_init();
#endif // USE_PAPI

    po_ret = process_options(argc, argv);

    if (PO_OKAY == po_ret && NONE != options.accel) {
        if (init_accel()) {
            fprintf(stderr, "Error initializing device\n");
            exit(EXIT_FAILURE);
        }
    }

    if (options.thread_multiple) {
        int provided;
        MPI_CHECK(MPI_Init_thread(&argc, &argv, MPI_THREAD_MULTIPLE, &provided));
        if (provided != MPI_THREAD_MULTIPLE) fprintf(stderr, "warn: MPI_THREAD_MULTIPLE not supported!\n");
    } else {
        MPI_CHECK(MPI_Init(&argc, &argv));
    }

    MPI_CHECK(MPI_Comm_size(MPI_COMM_WORLD, &numprocs));
    MPI_CHECK(MPI_Comm_rank(MPI_COMM_WORLD, &myid));

    //MPI_CHECK(MPIX_Continue_init(0, MPIX_CONT_POLL_ONLY, &cont_req, MPI_INFO_NULL));
    MPI_CHECK(MPIX_Continue_init(0, 0, &cont_req, MPI_INFO_NULL));

    if (0 == myid) {
        switch (po_ret) {
            case PO_CUDA_NOT_AVAIL:
                fprintf(stderr, "CUDA support not enabled.  Please recompile "
                        "benchmark with CUDA support.\n");
                break;
            case PO_OPENACC_NOT_AVAIL:
                fprintf(stderr, "OPENACC support not enabled.  Please "
                        "recompile benchmark with OPENACC support.\n");
                break;
            case PO_BAD_USAGE:
                print_bad_usage_message(myid);
                break;
            case PO_HELP_MESSAGE:
                print_help_message(myid);
                break;
            case PO_VERSION_MESSAGE:
                print_version_message(myid);
                MPI_CHECK(MPI_Finalize());
                exit(EXIT_SUCCESS);
            case PO_OKAY:
                break;
        }
    }

    switch (po_ret) {
        case PO_CUDA_NOT_AVAIL:
        case PO_OPENACC_NOT_AVAIL:
        case PO_BAD_USAGE:
            MPI_CHECK(MPI_Finalize());
            exit(EXIT_FAILURE);
        case PO_HELP_MESSAGE:
        case PO_VERSION_MESSAGE:
            MPI_CHECK(MPI_Finalize());
            exit(EXIT_SUCCESS);
        case PO_OKAY:
            break;
    }

    if(numprocs != 1) {
        if(myid == 0) {
            fprintf(stderr, "This test requires exactly one process\n");
        }

        MPI_CHECK(MPI_Finalize());
        exit(EXIT_FAILURE);
    }

    if (allocate_memory_pt2pt(&s_buf, &r_buf, myid)) {
        /* Error allocating memory */
        MPI_CHECK(MPI_Finalize());
        exit(EXIT_FAILURE);
    }

    print_header(myid, LAT);

    /* Latency test */
    for(size = options.min_message_size; size <= options.max_message_size; size = (size ? size * 2 : 1)) {
        set_buffer_pt2pt(s_buf, myid, options.accel, 'a', size);
        set_buffer_pt2pt(r_buf, myid, options.accel, 'b', size);

        if(size > LARGE_MESSAGE_SIZE) {
            options.iterations = options.iterations_large;
            options.skip = options.skip_large;
        }

        MPI_CHECK(MPI_Barrier(MPI_COMM_WORLD));

        if(myid == 0) {
            for(i = 0; i < options.skip; i++) {
                MPI_Request req;
                cont_cb_dir_t dir = CONT_CB_RECV;
                int flag;
				MPI_CHECK(MPI_Start(&cont_req));
                MPI_CHECK(MPI_Irecv(r_buf, size, MPI_CHAR, 0, 1, MPI_COMM_WORLD, &req));
                MPI_CHECK(MPIX_Continue(&req, &cont_cb, &dir, 0, &reqstat, cont_req));
                MPI_CHECK(MPI_Send(s_buf, size, MPI_CHAR, 0, 1, MPI_COMM_WORLD));
                MPI_CHECK(MPI_Wait(&cont_req, MPI_STATUS_IGNORE));
            }
#ifdef USE_PAPI
            hwc_ins(&hwc_start);
#endif // USE_PAPI
            t_start = MPI_Wtime();
            for(i = 0; i < options.iterations; i++) {
                MPI_Request req;
                cont_cb_dir_t dir = CONT_CB_RECV;
                int flag;
				MPI_CHECK(MPI_Start(&cont_req));
                MPI_CHECK(MPI_Irecv(r_buf, size, MPI_CHAR, 0, 1, MPI_COMM_WORLD, &req));
                //MPI_CHECK(MPIX_Continue(&req, &cont_cb, &dir, 0, &reqstat, cont_req));
                MPI_CHECK(MPIX_Continue(&req, &cont_cb, &dir, MPIX_CONT_REQBUF_VOLATILE, &reqstat, cont_req));
                MPI_CHECK(MPI_Send(s_buf, size, MPI_CHAR, 0, 1, MPI_COMM_WORLD));
                MPI_CHECK(MPI_Wait(&cont_req, MPI_STATUS_IGNORE));
            }
            t_end = MPI_Wtime();
#ifdef USE_PAPI
            hwc_ins(&hwc_end);
#endif // USE_PAPI
        }

        if(myid == 0) {
            double latency = (t_end - t_start) * 1e6 / (options.iterations);
#ifdef USE_PAPI
            uint64_t instructions = (hwc_end.ins - hwc_start.ins) / (options.iterations);
            uint64_t cycles       = (hwc_end.cyc - hwc_start.cyc) / (options.iterations);

            fprintf(stdout, "%-*d%*.*f %*llu %*llu\n", 10, size, FIELD_WIDTH,
                    FLOAT_PRECISION, latency, FIELD_WIDTH, instructions, FIELD_WIDTH, cycles);
#else
            fprintf(stdout, "%-*d%*.*f\n", 10, size, FIELD_WIDTH,
                    FLOAT_PRECISION, latency);
#endif // USE_PAPI
            fflush(stdout);
        }
    }

    free_memory(s_buf, r_buf, myid);
    MPI_CHECK(MPI_Finalize());

    if (NONE != options.accel) {
        if (cleanup_accel()) {
            fprintf(stderr, "Error cleaning up device\n");
            exit(EXIT_FAILURE);
        }
    }

    return EXIT_SUCCESS;
}

