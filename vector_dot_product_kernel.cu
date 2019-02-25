#ifndef _VECTOR_DOT_PRODUCT_KERNEL_H_
#define _VECTOR_DOT_PRODUCT_KERNEL_H_


#define BLOCK_SIZE 256
#define GRID_SIZE 240

/* prototypes */
__device__ void lock(int *mutex);
__device__ void unlock(int *mutex);

__global__ void vector_dot_product_kernel( float *A, float *B, float *C, unsigned int numElements, int *mutex) {

	__shared__ float thread_sums[ BLOCK_SIZE ];

	/* thread ID and stride lengths (for coalescing memory) */
	unsigned int tID = blockDim.x * blockIdx.x + threadIdx.x;
	unsigned int stride_length = blockDim.x * gridDim.x;

	/* initialize local thread sum and starting location for thread*/
	float local_thread_sum = 0.0f;
	unsigned int i = tID;

	/* perform multiplication and add stride_length continuously until max number of elements reached -->*/
	while( i < numElements ) {

		/* multiply, increment by stride */
		local_thread_sum += A[i] * B[i];
		i += stride_length;
	}

	/* Put thread sum in shared mem accessible to all thread blocks */
	thread_sums[threadIdx.x] = local_thread_sum;
	__syncthreads();


	/* REDUCTION -- Reduce thread sums on a per-block basis (so result in one sum per block) */
	i = BLOCK_SIZE / 2; 	
	while ( i != 0 ) {

		/* threads where i < 0 are threads on the second "half" which don't need to execute */
		if ( threadIdx.x < i ) {

			/* sum the calculating threads partial value with its second "half" counterpart */
			thread_sums[threadIdx.x] += thread_sums[ threadIdx.x + i ];
		}
		__syncthreads();

		/* reduces the threads by 2 each iteration */
		i = i / 2;
	}

	/* first thread in each block adds block-wide value to global mem location*/
	if (threadIdx.x == 0) {
		// define a lock
		lock(mutex);
		// add sums to global memory which is my critical section
		C[0] += thread_sums[0] ;
		// unlock the mutex/critical section
		unlock(mutex);
	}
}

__device__ void lock(int *mutex){
       	// if my mutex is 0 swap and set it to 1 indication locking of critical section
	while(atomicCAS(mutex, 0, 1) != 0);
}

/* Using exchange to release mutex. */
__device__ void unlock(int *mutex)
{      // perform an atomic exchange in which the pointer of the mutex now becomes 0	
       atomicExch(mutex, 0);
}



#endif // #ifndef _VECTOR_DOT_PRODUCT_KERNEL_H
