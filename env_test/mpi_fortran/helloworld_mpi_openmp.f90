! helloworld_mpi_openmp.f90
program helloworld
  use mpi
  use omp_lib
  implicit none

  integer :: ierr, rank, size, num_threads, thread_num

  ! Initialize MPI
  call MPI_Init(ierr)
  call MPI_Comm_rank(MPI_COMM_WORLD, rank, ierr)
  call MPI_Comm_size(MPI_COMM_WORLD, size, ierr)

  ! Set the number of threads
  num_threads = 8
  call omp_set_num_threads(num_threads)

  !$OMP PARALLEL PRIVATE(thread_num)
    thread_num = omp_get_thread_num()
    print *, 'Hello from processor ', rank, ' and thread ', thread_num
  !$OMP END PARALLEL

  ! Finalize MPI
  call MPI_Finalize(ierr)

end program helloworld

