#include "RNN.cuh"

RNN::RNN()
{ }

RNN::RNN(
	int i, 
	int o, 
	int n, 
	double lr, 
	int m_d, 
	int bat, 
	std::vector<activation_function> &a_f,  
	COO_matrix &w, 
	cusparseHandle_t&  hand
)
{
	handle = &hand;

	time_step = 0;
	
	dev_ini = false;
	ini = i;
	out = o;
	nodes = n;
	learning_rate = lr;
	bias_learning_rate = lr;
	max_iters = m_d;
	max_depth = m_d;
	batch = bat;
	
	act_f = a_f;
	current_values = Vector(nodes, 0);
	current_dx = Vector(nodes, 0);
	
	biases = Vector(nodes, 0);
	biases_update = Vector(nodes, 0);

	weights = w;
	weights_update = Vector(w.nnz, 0);

	values_through_time = Matrix(max_depth, Vector(nodes, 0));
	dx_through_time = Matrix(max_depth, Vector(nodes, 0));

	mask = std::vector<int>(nodes, 0);
}

RNN::~RNN(){
	if (dev_ini) {
		//std::cout << "Initialized" << std::endl;
		empty_device();
	}
}

int RNN::n_offset_by_time()
{
	return nodes * (time_step % max_depth);
}

void RNN::initialize_device_public()
{
	initialize_device();
	copy_weights_to_device();
	copy_vect_to_device(act_f, d_act_f, nodes);
	reset_vect(dX, nodes);
	reset_vect(d_m, weights.nnz);
	reset_vect(d_v, weights.nnz);
}

//////////////////////////DEVICE////////////////////////////
void RNN::initialize_device() 
{
	dev_ini = true;
	//Weights
	cudaMalloc((void**)&dA_rows, weights.nnz * sizeof(int));
	cudaMalloc((void**)&dA_columns, weights.nnz * sizeof(int));
	cudaMalloc((void**)&dA_values, weights.nnz * sizeof(double));

	//Biases
	cudaMalloc((void**)&d_biases, nodes * sizeof(double));

	// Weights and biases update
	cudaMalloc((void**)&d_weights_update, weights.nnz * sizeof(double));
	cudaMalloc((void**)&d_biases_update, nodes * sizeof(double));

	//Adam
	cudaMalloc((void**)&d_m, weights.nnz * sizeof(double));
	cudaMalloc((void**)&d_v, weights.nnz * sizeof(double));
	cudaMalloc((void**)&d_m_corr, weights.nnz * sizeof(double));
	cudaMalloc((void**)&d_v_corr, weights.nnz * sizeof(double));

	//Values through time
	cudaMalloc((void**)&d_values_through_time, nodes * max_depth * sizeof(double));
	cudaMalloc((void**)&d_dx_through_time, nodes * max_depth * sizeof(double));

	//Input / Output vectors
	cudaMalloc((void**)&dX, nodes * sizeof(double));
	cudaMalloc((void**)&dY, nodes * sizeof(double));

	// Gradient
	cudaMalloc((void**)&d_gradient, nodes * sizeof(double));
	cudaMalloc((void**)&d_dx, nodes * sizeof(double));

	// Input
	cudaMalloc((void**)&d_input, nodes * sizeof(double));

	// Mask
	cudaMalloc((void**)&d_mask, nodes * sizeof(int));

	//Activation functions
	cudaMalloc((void**)&d_act_f, nodes * sizeof(int));


	// Create sparse matrix A in COO format
	cusparseCreateCoo(&matA, nodes, nodes, weights.nnz,
		dA_rows, dA_columns, dA_values,
		CUSPARSE_INDEX_32I,
		CUSPARSE_INDEX_BASE_ZERO, CUDA_R_64F);
	// Create dense vector X
	cusparseCreateDnVec(&vecX, nodes, dX, CUDA_R_64F);
	// Create dense vector y
	cusparseCreateDnVec(&vecY, nodes, dY, CUDA_R_64F);
	// Create dense vector grad
	cusparseCreateDnVec(&gradX, nodes, d_gradient, CUDA_R_64F);


	// allocate an external buffer if needed
	cusparseSpMV_bufferSize(
		*handle, CUSPARSE_OPERATION_NON_TRANSPOSE,
		&alpha, matA, vecX, &beta, vecY, CUDA_R_64F,
		CUSPARSE_SPMV_ALG_DEFAULT, &bufferSize);
	cudaMalloc(&dBuffer, bufferSize);

}

void RNN::copy_weights_to_device()
{
	if (weights.nnz == 0) return;
	cudaMemcpy(dA_rows, weights.cooRowInd.data(), weights.nnz * sizeof(int),
		cudaMemcpyHostToDevice);
	cudaMemcpy(dA_columns, weights.cooColInd.data(), weights.nnz * sizeof(int),
		cudaMemcpyHostToDevice);
	cudaMemcpy(dA_values, weights.cooValues.data(), weights.nnz * sizeof(double),
		cudaMemcpyHostToDevice);
}

void RNN::copy_weights_and_biases_to_device()
{
	if (weights.nnz == 0) return;
	cudaMemcpy(dA_rows, weights.cooRowInd.data(), weights.nnz * sizeof(int),
		cudaMemcpyHostToDevice);
	cudaMemcpy(dA_columns, weights.cooColInd.data(), weights.nnz * sizeof(int),
		cudaMemcpyHostToDevice);
	cudaMemcpy(dA_values, weights.cooValues.data(), weights.nnz * sizeof(double),
		cudaMemcpyHostToDevice);
	 
	cudaMemcpy(d_biases, biases.data(), nodes * sizeof(double),
		cudaMemcpyHostToDevice);
}


void RNN::execute_matrix_vector_prod( 
	cusparseOperation_t transpose, 
	cusparseDnVecDescr_t vect, 
	cusparseDnVecDescr_t output
){
	if (weights.nnz == 0) return;
	cusparseSpMV(
		*handle, 
		transpose,
		&alpha, 
		matA, 
		vect, 
		&beta, 
		output, 
		CUDA_R_64F,
		CUSPARSE_SPMV_ALG_DEFAULT, 
		dBuffer
	);
}

int RNN::get_blocks(int size)
{
	if (size == 0) return 0;
	int val = size / block_size;
	if (size % block_size > 0) val++;
	return val;
}

int RNN::get_threads(int blocks, int size)
{
	if (size == 0) return 0;
	int val = size / blocks;
	if (size % blocks > 0) val++;
	return val;
}

void RNN::execute_update_gradient()
{
	int size = nodes;
	if (size == 0) return;
	int blocks = get_blocks(size);
	int threads = get_threads(blocks, size);
	multKernel<<< blocks, threads >>>(d_gradient, 1, d_dx, size);

	/*
	// cudaDeviceSynchronize waits for the kernel to finish, and returns
    // any errors encountered during the launch.
	cudaError_t cudaStatus = cudaGetLastError();
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "Update gradient launch failed: %s\n", cudaGetErrorString(cudaStatus));
	}

	// cudaDeviceSynchronize waits for the kernel to finish, and returns
	// any errors encountered during the launch.
	cudaStatus = cudaDeviceSynchronize();
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaDeviceSynchronize returned error code %d after launching addKernel!\n", cudaStatus);
	}
	*/
}

void RNN::execute_sub_biases_update()
{
	int size = nodes;
	if (size == 0) return;
	int blocks = get_blocks(size);
	int threads = get_threads(blocks, size);
	addKernel<<< blocks, threads >>>(d_biases_update, -1, d_gradient, size);
	
	/*
	cudaDeviceSynchronize();

	cudaError_t cudaStatus = cudaGetLastError();
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "Sub biases launch failed: %s\n", cudaGetErrorString(cudaStatus));
	}*/

}

void RNN::execute_add_biases()
{
	int size = nodes;
	if (size == 0) return;
	int blocks = get_blocks(size);
	int threads = get_threads(blocks, size);
	addKernel<<< blocks, threads >>>(dX, 1, d_biases, size);
	
	/*
	cudaDeviceSynchronize();

	cudaError_t cudaStatus = cudaGetLastError();
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "Add biases launch failed: %s\n", cudaGetErrorString(cudaStatus));
	}
	*/
}

void RNN::execute_set_input_values(Vector& input_values)
{
	int size = input_values.size();
	if (size == 0) return;
	copy_vect_to_device(input_values, d_input, size);
}

void RNN::execute_add_input_values(int how_many_input_values)
{
	if (how_many_input_values == 0) return;
	int blocks = get_blocks(how_many_input_values);
	int threads = get_threads(blocks, how_many_input_values);
	addKernel<<< blocks, threads >>>(dX, 1, d_input, how_many_input_values);

	/*
	cudaDeviceSynchronize();

	cudaError_t cudaStatus = cudaGetLastError();
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "Add input launch failed: %s\n", cudaGetErrorString(cudaStatus));
	}
	*/
}

void RNN::execute_sub_weights_update()
{
	int size = weights.nnz;
	if (size == 0) return;
	//std::cout << "yes its here" << std::endl;
	int blocks = get_blocks(size);
	int threads = get_threads(blocks, size);
	TmultVectsKernel<<<blocks, threads>>>(d_weights_update, dA_rows, dA_columns, d_gradient, -1, dX, size);
	
	/*
	cudaDeviceSynchronize();
	cudaError_t cudaStatus = cudaGetLastError();
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "Update gradient launch failed: %s\n", cudaGetErrorString(cudaStatus));
	}*/

}

void RNN::execute_update_weights_and_biases()
{
	int size = weights.nnz;
	if (size == 0) return;
	int blocks = get_blocks(size);
	int threads = get_threads(blocks, size);
	addKernel<<< blocks, threads >>>(dA_values, learning_rate/batch, d_weights_update, size);
	//cudaDeviceSynchronize();

	size = nodes;
	blocks = get_blocks(size);
	threads = get_threads(blocks, size);
	addKernel<<< blocks, threads >>>(d_biases, learning_rate/batch, d_biases_update, size);
	//cudaDeviceSynchronize();
}

void RNN::execute_update_weights_and_biases_Adam()
{
	// We divide by batch size to average the gradients
	int size = weights.nnz;
	if (size == 0) return;
	int blocks = get_blocks(size);
	int threads = get_threads(blocks, size);
	powBeta1 *= beta1;
	powBeta2 *= beta2;
	updateAdamKernel <<<blocks, threads>>> (beta1, beta2, d_m, d_v, d_m_corr, d_v_corr, d_weights_update, size, powBeta1, powBeta2);
	//cudaDeviceSynchronize();
	addWithAdamKernel<<<blocks, threads>>> (dA_values, learning_rate/batch, d_m_corr, d_v_corr, eps, size);
	//cudaDeviceSynchronize();
	/*
	cudaError_t cudaStatus = cudaGetLastError();
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "Adam launch failed: %s\n", cudaGetErrorString(cudaStatus));
	}*/

	size = nodes;
	blocks = get_blocks(size);
	threads = get_threads(blocks, size);
	addKernel <<< blocks, threads >>> (d_biases, learning_rate/ batch, d_biases_update, size);
	//cudaDeviceSynchronize();
	/*
	cudaStatus = cudaGetLastError();
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "Biases update launch failed: %s\n", cudaGetErrorString(cudaStatus));
	}
	*/
}

void RNN::execute_use_activation_functions()
{
	int size = nodes;
	if (size == 0) return;
	int blocks = get_blocks(size);
	int threads = get_threads(blocks, size);
	/*
	if (training) {
		for (int i = ini+out; i < nodes; ++i)
			mask[i] = int(double(rand() % 1000) / 1000 < inp_dropout);
		copy_vect_to_device(mask, d_mask, nodes);
	}*/
	useActKernel <<< blocks, threads >>> (dX, d_dx, d_act_f, size, d_mask);
	move_vect_of_device(d_dx_through_time + n_offset_by_time(), d_dx, nodes);
	//cudaDeviceSynchronize();
	/*
	cudaError_t cudaStatus = cudaGetLastError();
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "Use activation func launch failed: %s\n", cudaGetErrorString(cudaStatus));
	}
	*/
}

void RNN::copy_vect_to_device( Vector &values, double* device_vect, int size )
{
	if (size == 0) return;
	cudaMemcpy(device_vect, values.data(), size * sizeof(double),
		cudaMemcpyHostToDevice);
}

void RNN::copy_vect_to_device(std::vector<activation_function>& values, int* device_vect, int size)
{
	if (size == 0) return;
	cudaMemcpy(device_vect, values.data(), size * sizeof(int),
		cudaMemcpyHostToDevice);
}

void RNN::copy_vect_to_device(std::vector<int>& values, int* device_vect, int size)
{
	if (size == 0) return;
	cudaMemcpy(device_vect, values.data(), size * sizeof(int),
		cudaMemcpyHostToDevice);
}


void RNN::move_vect_of_device(double* device_vect1, double* device_vect2, int size)
{
	if (size == 0) return;
	cudaMemcpy(device_vect1, device_vect2, size * sizeof(double),
		cudaMemcpyHostToDevice);
}

void RNN::reset_vect(double* device_vect, int size)
{
	if (size == 0) return;
	int blocks = get_blocks(size);
	int threads = get_threads(blocks, size);
	resetVectKernel<<<blocks, threads>>>(device_vect, size);
	/*
	cudaDeviceSynchronize();
	cudaError_t cudaStatus = cudaGetLastError();
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "Reset vect launch failed: %s\n", cudaGetErrorString(cudaStatus));
	}
	*/
}

void RNN::reset_vect(int* device_vect, int size)
{
	if (size == 0) return;
	int blocks = get_blocks(size);
	int threads = get_threads(blocks, size);
	resetVectKernel << <blocks, threads >> > (device_vect, size);
	/*
	cudaDeviceSynchronize();
	cudaError_t cudaStatus = cudaGetLastError();
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "Reset vect launch failed: %s\n", cudaGetErrorString(cudaStatus));
	}
	*/
}

void RNN::empty_device()
{
	// destroy matrix/vector descriptors
	cusparseDestroySpMat(matA);
	cusparseDestroyDnVec(vecX);
	cusparseDestroyDnVec(gradX);
	cusparseDestroyDnVec(vecY);


	cudaFree(dBuffer); 
	cudaFree(dA_rows);
	cudaFree(dA_columns);
	cudaFree(dA_values);
	cudaFree(dX);
	cudaFree(dY);
	cudaFree(d_m);
	cudaFree(d_v);
	cudaFree(d_m_corr);
	cudaFree(d_v_corr);
	cudaFree(d_weights_update);
	cudaFree(d_biases_update);
	cudaFree(d_gradient);
	cudaFree(d_input);
	cudaFree(d_mask);
	cudaFree(d_act_f);
	cudaFree(d_dx);
	cudaFree(d_values_through_time);
	cudaFree(d_dx_through_time);
}

void RNN::read_weights_of_device()
{
	if (weights.nnz == 0) return;
	cudaMemcpy(weights.cooValues.data(), dA_values, weights.nnz * sizeof(double),
		cudaMemcpyDeviceToHost);
}

void RNN::read_vect_of_device(Vector& result, double* device_vect, int size)
{
	if (size == 0) return;
	cudaMemcpy(result.data(), device_vect, size * sizeof(double),
		cudaMemcpyDeviceToHost);
 }

void RNN::read_weights_and_biases_from_device(Vector& weights_, Vector& biases_)
{
	if (weights.nnz == 0) return;
	cudaMemcpy(weights_.data(), dA_values, weights.nnz * sizeof(double),
		cudaMemcpyDeviceToHost);

	cudaError_t cudaStatus = cudaGetLastError();
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "Read public weights failed: %s\n", cudaGetErrorString(cudaStatus));
	}

	cudaMemcpy(biases_.data(), d_biases, nodes * sizeof(double),
		cudaMemcpyDeviceToHost);

	cudaStatus = cudaGetLastError();
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "Read public biases failed: %s\n", cudaGetErrorString(cudaStatus));
	}
}

////////////////////////////////////////////////////////////

void RNN::copy_wb_to_dev()
{
	copy_weights_and_biases_to_device();
}
void RNN::add_connection(int node_i, int node_j, double value)
{
	weights.insert_elm(node_i, node_j, value);
	nodes++;
}

void RNN::full_reset()
{
	current_values = Vector(nodes, 0);
	current_dx = Vector(nodes, 0);

	values_through_time = Matrix(max_depth, Vector(nodes, 0));
	dx_through_time = Matrix(max_depth, Vector(nodes, 0));

	biases_update = Vector(nodes, 0);
	weights_update = Vector(weights.nnz, 0);

	reset_vect(dX, nodes);
	reset_vect(dY, nodes);
	reset_vect(d_biases_update, nodes);
	reset_vect(d_weights_update, weights.nnz);
	reset_vect(d_m, weights.nnz);
	reset_vect(d_v, weights.nnz);
	reset_vect(d_m_corr, weights.nnz);
	reset_vect(d_v_corr, weights.nnz);
	reset_vect(d_gradient, nodes);
	reset_vect(d_input, nodes);
	reset_vect(d_dx, nodes);
	reset_vect(d_mask, nodes);
	reset_vect(d_values_through_time, nodes * max_depth);
	reset_vect(d_dx_through_time, nodes * max_depth);

	powBeta1 = 0.9;
	powBeta2 = 0.999;

}

void RNN::reset()
{
	current_values = Vector(nodes, 0);
	current_dx = Vector(nodes, 0);

	values_through_time = Matrix(max_depth, Vector(nodes, 0));
	dx_through_time = Matrix(max_depth, Vector(nodes, 0));
	reset_vect(dX, nodes);
	reset_vect(d_mask, nodes);
	reset_vect(d_values_through_time, nodes*max_depth);
	reset_vect(d_dx_through_time, nodes*max_depth);

}

void RNN::forward_prop()
{
	execute_add_biases();
	execute_use_activation_functions();
	// GET RID OF THIS!!
	/*read_vect_of_device(current_values, dX, nodes);
	read_vect_of_device(current_dx,   d_dx, nodes);

	values_through_time.push_back(current_values);
	dx_through_time.push_back(current_dx);

	if (max_depth < values_through_time.size())
	{
		values_through_time.erase(values_through_time.begin());
		dx_through_time.erase(dx_through_time.begin());
	}*/
	//auto values = values_through_time[values_through_time.size()-1];
	//std::cout << "Before"<< std::endl;
	//copy_vect_to_device(values, dX, nodes);
	//std::cout << values[0] << " " << values[1] << std::endl;*/
	execute_matrix_vector_prod(CUSPARSE_OPERATION_NON_TRANSPOSE, vecX, vecY);
	//read_vect_of_device(values, dY, nodes);
	//std::cout << "After" << std::endl;
	//std::cout << values[0] << " " << values[1] << std::endl;
	move_vect_of_device(d_values_through_time + n_offset_by_time(), dX, nodes);

	move_vect_of_device(dX, dY, nodes);

	time_step++;
}

double mean(Vector &values)
{
	double sum = 0;
	for (double val : values)
		sum += val;
	return sum / values.size();
}

void RNN::softmax(Vector& values, int ini, int fi)
{
	int s = values.size();
	double max = values[0];
	for (int i = ini; i < fi; ++i)
		if (values[i] > max)
			max = values[i];

	double sum = 0;
	for (int i = ini; i < fi; ++i)
	{
		values[i] = exp(values[i] - max);
		sum += values[i];
	}

	for (int i = ini; i < fi; ++i) 
		values[i] /= sum;
}
double RNN::backward_prop(Vector &correct_vals, bool classif)
{
	auto layer = values_through_time.size()-1;

	auto values = values_through_time[layer];

	Vector error(out,0);
	Vector gradient(nodes,0);
    
	int max_i = 0;
	double max_val = values[max_i];

	for (int i = 0; i < out; ++i) {
		if (values[ini + i] > max_val) {
			max_val = values[ini + i];
			max_i = i;
		}
		//Cross Entropy-gradient
		/*if (classif) error[i] = (values[ini + i] - correct_vals[i]) / ((values[ini + i] * (1 - values[ini + i])));
		//MSE-gradient
		else */
		error[i] = values[ini + i] - correct_vals[i];
		//std::cout << error[i] << std::endl;

		gradient[ini + i] = error[i];
		error[i] = abs(error[i]);		

		/*if (correct_vals[i] == 1)
		{
			std::cout << values[ini + i] << " " << i << std::endl;
		}*/
	}
	
	auto aux_values = values_through_time[layer];

	read_vect_of_device(aux_values, dX, nodes);
	copy_vect_to_device(gradient, d_gradient, nodes);
	while (layer > 0)
	{

		auto node_dx = dx_through_time[layer];

		values = values_through_time[layer-1];

		copy_vect_to_device(node_dx, d_dx, nodes);
		execute_update_gradient();
		
		//gradient clipping
		/*
		read_vect_of_device(gradient, d_gradient, nodes);
		
		double norm = 0;
		for (int i = 0; i < nodes; ++i)
			norm += gradient[i]*gradient[i];
		norm = sqrt(norm);
		if (norm > 5)
		{
			for (int i = 0; i < nodes; ++i)
				gradient[i] *= 5/norm;
			copy_vect_to_device(gradient, d_gradient, nodes);
		}
		else if (norm < 0.001)
		{
			for (int i = 0; i < nodes; ++i)
				gradient[i] *= 0.001/(norm+eps);
			copy_vect_to_device(gradient, d_gradient, nodes);
		}
		*/
		/*
		double norm2 = 0;
		for (int i = 0; i < nodes; ++i)
			norm2 += gradient[i] * gradient[i];
		norm2 = sqrt(norm2);
		std::cout << norm << " "<<norm2 << std::endl;*/

		execute_sub_biases_update();

		copy_vect_to_device(values, dX, nodes);
		execute_sub_weights_update();

		execute_matrix_vector_prod(CUSPARSE_OPERATION_TRANSPOSE, gradX, vecY);
		move_vect_of_device(d_gradient, dY, nodes);
		
		layer--;
	}
	copy_vect_to_device(aux_values, dX, nodes);
	if (classif && !std::isnan(mean(error))) return (double)(correct_vals[max_i] != 1);
	else return mean(error);
}

//synflow///////////////////////////////////////////////////////////////////////////
Vector RNN::synflow_cycle( bool classif, Vector &ones, int wich )
{
	training = true;
	read_weights_of_device();

	auto ant_weights = weights.cooValues;

	for (int i = 0; i < weights.nnz; ++i)
		weights.cooValues[i] = abs(weights.cooValues[i]);

	copy_weights_to_device();

	double error = 0;
	//set_input_values(ones);
	for (int i = 0; i < max_iters; ++i)
	{
		set_input_values(ones);
		//for (int i = 0; i < ones.size(); ++i)
			//ones[i] /= 10;

		forward_prop();
	}

	if (classif)
		softmax(values_through_time[values_through_time.size() - 1], ini, ini + out);

	double sum_val = sum(values_through_time[values_through_time.size() - 1]);
	Vector correct_values(out, sum_val);
	if (wich != -1)
		for (int i = 0; i < out; ++i)
			if (wich != i) 
				correct_values[i] = values_through_time[values_through_time.size() - 1][ini + wich];
			else correct_values[i] = sum_val;
	else 
		for(int i = 0; i < out; ++i)
		correct_values[i] += values_through_time[values_through_time.size() - 1][ini + wich];


	backward_prop(correct_values, classif);
	
	read_vect_of_device(weights_update, d_weights_update, weights.nnz);

	for(int i = 0; i < weights.nnz; ++i)
		weights.cooValues[i] = abs(weights_update[i]*weights.cooValues[i]);

	auto ret = weights.cooValues;
	weights.cooValues = ant_weights;
	copy_weights_to_device();
	full_reset();
	training = false;

	return ret;
}

Vector RNN::synflow_cycle(bool classif, std::vector<std::pair<Vector, double>>& dataset, int samples)
{
	training = true;
	read_weights_of_device();

	auto ant_weights = weights.cooValues;

	for (int i = 0; i < weights.nnz; ++i)
		weights.cooValues[i] = abs(weights.cooValues[i]);

	copy_weights_to_device();

	double error = 0;
	for (int i = 0; i < out; ++i) {
		int index;
		Vector inp(ini, 0);

		for (int j = 0; j < samples; ++j)
		{
			index = rand() % dataset.size();
			while (int(dataset[index].second) != i)
				index = rand() % dataset.size();

			auto aux_inp = dataset[index].first;

			for (int j = 0; j < ini; ++j)
				inp[j] += aux_inp[j] / samples;
		}

		double suma = 0;
		for (int j = 0; j < inp.size(); j++)
			suma += inp[j];
		suma /= inp.size();

		for (int j = 0; j < inp.size(); j++)
			inp[j] /= suma;

		/*double max = 0;
		for (int j = 0; j < inp.size(); j++) {
			if (inp[j] > max)
				max = inp[j];
		}

		//for (int j = 0; j < ones.size(); j++)
			//ones[j] = 1;

		std::cout << "max: " << max << " suma: " << suma << std::endl;
		for (int j = 0; j < 28; j++) {
			for (int u = 0; u < 28; u++) {
				if (inp[j * 28 + u] > max * 0.75) std::cout << "X ";
				else if (inp[j * 28 + u] > max * 0.5) std::cout << "x ";
				else if (inp[j * 28 + u] > max * 0.25) std::cout << "+ ";
				else if (inp[j * 28 + u] > max * 0) std::cout << "- ";
				else std::cout << "_ ";
			}
			std::cout << std::endl;
		}
		std::cout << std::endl;
        */
		for (int i = 0; i < max_iters; ++i)
		{
			set_input_values(inp);
			forward_prop();
		}
		if (classif)
			softmax(values_through_time[values_through_time.size() - 1], ini, ini + out);

		double sum_val = sum(values_through_time[values_through_time.size() - 1]);
		Vector correct_values(out, sum_val);
		for (int j = 0; j < out; ++j)
			if (j != i)
				correct_values[j] = values_through_time[values_through_time.size() - 1][ini + j];
			else correct_values[j] = values_through_time[values_through_time.size() - 1][ ini + j ]+sum_val;

		backward_prop(correct_values, classif);
		reset();
	}
	
	read_vect_of_device(weights_update, d_weights_update, weights.nnz);

	for (int i = 0; i < weights.nnz; ++i)
		weights.cooValues[i] *= weights_update[i] / out;


	auto ret = weights.cooValues;
	weights.cooValues = ant_weights;
	copy_weights_to_device();
	full_reset();
	training = false;
	return ret;
}
//////////////////////////////////////////////////////////////////////////////////

void RNN::partition(Matrix& inp, Vector& sol_list, double train_val_part, double val_test_part,
	std::vector < std::pair<Vector, double> >& dataset, std::vector < std::pair<Vector, double> >&  test_data,
	std::vector < std::pair<Vector, double> >& val_data, double div, bool shuffle) {

	srand(time(0));

	for (int i = 0; i < inp.size(); i++) {
		Vector in(inp[i].size());
		for (int j = 0; j < inp[i].size(); ++j)
			in[j] = double(inp[i][j]) / div;
		dataset.push_back(std::pair<Vector, double>(in, sol_list[i]));
	}

	if(shuffle)
		std::random_shuffle(dataset.begin(), dataset.end());

	test_data = std::vector< std::pair<Vector, double> >(dataset.begin() + dataset.size() * train_val_part, dataset.end());

	dataset.resize(dataset.size() * train_val_part);

	val_data = std::vector < std::pair<Vector, double> >(test_data.begin() + test_data.size() * val_test_part, test_data.end());

	test_data.resize(test_data.size() * val_test_part);
}

void RNN::partition(std::vector<std::vector<int>>& inp, std::vector<int>& sol_list, double train_val_part, double val_test_part,
	std::vector < std::pair<Vector, double> >& dataset, std::vector < std::pair<Vector, double> >& test_data,
	std::vector < std::pair<Vector, double> >& val_data, double div, bool shuffle) {

	srand(time(0));

	for (int i = 0; i < inp.size(); i++) {
		Vector in(inp[i].size());
		for (int j = 0; j < inp[i].size(); ++j)
			in[j] = double(inp[i][j])/div;
		dataset.push_back(std::pair<Vector, double>(in, sol_list[i]));
	}

	if (shuffle)
		std::random_shuffle(dataset.begin(), dataset.end());

	test_data = std::vector< std::pair<Vector, double> >(dataset.begin() + dataset.size() * train_val_part, dataset.end());

	dataset.resize(dataset.size() * train_val_part);

	val_data = std::vector < std::pair<Vector, double> >(test_data.begin() + test_data.size() * val_test_part, test_data.end());

	test_data.resize(test_data.size() * val_test_part);
}


void RNN::partition(std::vector<std::vector<uint8_t>>& inp, std::vector<uint8_t>& sol_list, double train_val_part, double val_test_part,
	std::vector < std::pair<Vector, double> >& dataset, std::vector < std::pair<Vector, double> >& test_data,
	std::vector < std::pair<Vector, double> >& val_data, double div, bool shuffle) {

	srand(time(0));

	for (int i = 0; i < inp.size(); i++) {
		Vector in(inp[i].size());
		for (int j = 0; j < inp[i].size(); ++j)
			in[j] = double(inp[i][j]) / div;
		dataset.push_back(std::pair<Vector, double>(in, sol_list[i]));
	}

	if (shuffle)
		std::random_shuffle(dataset.begin(), dataset.end());

	test_data = std::vector< std::pair<Vector, double> >(dataset.begin() + dataset.size() * train_val_part, dataset.end());

	dataset.resize(dataset.size() * train_val_part);

	val_data = std::vector < std::pair<Vector, double> >(test_data.begin() + test_data.size() * val_test_part, test_data.end());

	test_data.resize(test_data.size() * val_test_part);
}

void RNN::set_input_values(Vector& input_values)
{
	execute_set_input_values(input_values);
}

void RNN::add_input_values(int how_many_input_values)
{
	execute_add_input_values(how_many_input_values);
}

void RNN::read_values_through_time_from_device()
{
	//Matrix previous_values_through_time = values_through_time;
	
	auto aux_vals_vect = Vector(max_depth * nodes);
	auto aux_dx_vect = Vector(max_depth * nodes);
	read_vect_of_device(aux_vals_vect, d_values_through_time, max_depth * nodes);
	read_vect_of_device(aux_dx_vect, d_dx_through_time, max_depth * nodes);
	for (int i = max_depth - 1; i >= 0; i--) {
		int index = (time_step + i) % max_depth;
		//std::cout << "index: " << index << ", i: "<<i<<std::endl;
		//std::cout << "nodes: " << nodes << ", index * nodes: " << index * nodes << std::endl;
		//std::cout << "max depth * nodes: " << max_depth * nodes << ", size of values through time " << values_through_time.size() * values_through_time[0].size() << std::endl;
		//read_vect_of_device(values_through_time[i], d_values_through_time + index * nodes, nodes);
		//read_vect_of_device(dx_through_time[i], d_dx_through_time + index * nodes, nodes);
		for (int n = 0; n < nodes; ++n) {
			values_through_time[i][n] = aux_vals_vect[i * nodes + n];
			dx_through_time[i][n] = aux_dx_vect[i * nodes + n];
		}
	}
	/*
	for (int d = 0; d < max_depth; ++d) {
		for (int i = 0; i < nodes; ++i) {
			std::cout << "(" << previous_values_through_time[d][i] << ", " << values_through_time[d][i] << ") ";
		}
		std::cout << std::endl;
	}
	std::cout << std::endl << std::endl << std::endl;
	*/
}
double RNN::forward_prop_cycle(Vector& input_values, Vector& correct_values, bool classif)
{
	double error = 0;
	execute_set_input_values(input_values);
	for (int i = 0; i < max_iters; ++i) {
		execute_add_input_values(input_values.size());
		forward_prop();
	}
	read_values_through_time_from_device();
	if (classif)
		softmax(values_through_time[values_through_time.size() - 1], ini, ini + out);

	error = get_error(correct_values, classif);
	return error;
}
double RNN::forward_prop_one_inp_cycle(Vector& input_values, Vector& correct_values, bool classif)
{
	double error = 0;

	execute_set_input_values(input_values);
	execute_add_input_values(input_values.size());
	for (int i = 0; i < max_iters; ++i)
		forward_prop();
	read_values_through_time_from_device();
	if (classif)
		softmax(values_through_time[values_through_time.size() - 1], ini, ini + out);

	error = get_error(correct_values, classif);
	return error;
}

double RNN::get_error(Vector& correct_values, bool classif)
{
	auto layer = values_through_time.size() - 1;

	auto values = values_through_time[layer];

	Vector error(out, 0);
	Vector gradient(nodes, 0);

	int max_i = 0;
	double max_val = values[max_i];

	for (int i = 0; i < out; ++i) {
		if (values[ini + i] > max_val) {
			max_val = values[ini + i];
			max_i = i;
		}
		//Cross Entropy-gradient
		/*if (classif) error[i] = (values[ini + i] - correct_vals[i]) / ((values[ini + i] * (1 - values[ini + i])));
		//MSE-gradient
		else */
		error[i] = abs(values[ini + i] - correct_values[i]);
		/*if (correct_vals[i] == 1)
		{
			std::cout << values[ini + i] << " " << i << std::endl;
		}*/
	}

	if (classif && !std::isnan(mean(error))) return (double)(correct_values[max_i] != 1);
	else return mean(error);
}

double RNN::train_step(
	Vector &input_values, 
	Vector &correct_values, 
	bool classif
){
	double error = 0;
	training = true;
	set_input_values(input_values);
	for (int i = 0; i < max_iters; ++i)
	{
		add_input_values(input_values.size());
		forward_prop();
	}
	read_values_through_time_from_device();
	if (classif);
		softmax(values_through_time[values_through_time.size()-1], ini, ini+out);

	error = backward_prop(correct_values, classif); // 35% of the time is spent here
	training = false;
	return error;
}

double RNN::train_step_one_inp_set(
	Vector& input_values, 
	Vector& correct_values, 
	bool classif
)
{
	double error = 0;
	training = true;
	set_input_values(input_values);
    for (int i = 0; i < max_iters; ++i)
		forward_prop();

	if (classif)
		softmax(values_through_time[values_through_time.size() - 1], ini, ini+out);

	error = backward_prop(correct_values, classif);
	training = false;
	return error;
}

double RNN::time_sens_train_step(std::vector< Vector>& input_values, std::vector< Vector>& correct_values, bool classif)
{
	double error = 0;
	training = true;
	std::string alphabet = "abcdefghijklmopqrstuvwxyz1234567890 ,";
	int iters = 0;
	for (int i = 0; i < delay_iters; ++i)
	{
		set_input_values(input_values[i]);
		forward_prop();
		iters++;
	}
	for (int i = delay_iters; i < input_values.size(); ++i) {
	    error += train_step(input_values[i], correct_values[i-delay_iters], classif);
		iters++;

		std::cout << alphabet[max_pos(ini, out+ini, values_through_time[values_through_time.size() - 1])-ini];
	}
	for (int i = iters; i < iters+delay_iters; ++i) {
		error += train_step(input_values[iters-1], correct_values[i - delay_iters], classif);

		std::cout << alphabet[max_pos(ini, out+ini, values_through_time[values_through_time.size() - 1])-ini];
	}
	std::cout << std::endl << std::endl << std::endl;

	std::cout <<"correct values"<< std::endl;
	for ( auto solution : correct_values )
		std::cout << alphabet[max_pos(0, alphabet.size(), solution)];
	std::cout << std::endl << std::endl << std::endl;
	training = false;
	return error / input_values.size();
}

void RNN::update_weights_and_biases()
{
	if(adam)execute_update_weights_and_biases_Adam();
	else execute_update_weights_and_biases();
	reset_vect(d_biases_update, nodes);
	reset_vect(d_weights_update, weights.nnz);
}

void RNN::print_matrix()
{
	read_vect_of_device(weights.cooValues, dA_values, weights.nnz);

	read_vect_of_device(biases, d_biases, nodes);

	printf("-------------------------------\n");
	for (int i = 0; i < nodes; ++i)
		std::cout<<i<< " value: "<< current_values[i]<< "| ";
	std::cout << std::endl;

	printf("-----------weights--------------\n");
	for (int i = 0; i < weights.nnz; ++i)
		std::cout<<weights.cooColInd[i]<<" "<< weights.cooRowInd[i]<<" " << weights.cooValues[i] << std::endl;

	printf("---------bias_update---------\n");
	for (int i = 0; i < nodes; ++i)
		std::cout<<i<< " bias: "<<biases_update[i]<<"| ";
	std::cout << std::endl;
	printf("---------weight_update----------\n");
	for (int i = 0; i < weights_update.size(); ++i)
    std::cout << weights.cooColInd[i] <<" "<< weights.cooRowInd[i]<<" " << weights_update[i] << std::endl;
}

void RNN::print_rnn_to_file()
{
	std::fstream strm;
	strm.open("./RNN.txt", std::ios_base::in);

	read_vect_of_device(weights.cooValues, dA_values, weights.nnz);

	read_vect_of_device(biases, d_biases, nodes);

	strm<< max_iters<<std::endl;
	strm << max_depth << std::endl;
	strm << delay_iters << std::endl;
	strm << batch << std::endl;
	strm << learning_rate << std::endl;
	strm << bias_learning_rate << std::endl;

	strm << ini << std::endl;
	strm << out << std::endl;
	strm << nodes << std::endl;

	for (int i = 0; i < nodes; ++i)
		strm << biases[i] << " " << act_f[i] << std::endl;

	COO_matrix weights;

	//printf("-----------weights--------------\n");
	for (int i = 0; i < weights.nnz; ++i)
		strm << weights.cooRowInd[i] << " " << weights.cooColInd[i] << " " << weights.cooValues[i] << std::endl;

	strm.close();
}

void RNN_xor_problem()
{
	cusparseHandle_t handle;
	cusparseCreate(&handle);
	int ini_nodes = 2;
	int out_nodes = 1;
	int nodes = 4;
	int batch = 4;
	int depth = 3;
	double learning_rate = 1.0;
	double error_sum = 0;
	std::vector<activation_function> act_f(nodes, RELU);

	for (int i = ini_nodes; i < ini_nodes + out_nodes; ++i)
		act_f[i] = SIGMOID;
	for (int i = ini_nodes + out_nodes; i < nodes; ++i)
		act_f[i] = SIGMOID;


	COO_matrix mat;

	/*mat.insert_elm(0, 2, randomdouble());
	mat.insert_elm(1, 2, randomdouble());

    mat.insert_elm(0, 4, randomdouble());
	mat.insert_elm(1, 4, randomdouble());
	mat.insert_elm(0, 3, randomdouble());
	mat.insert_elm(1, 3, randomdouble());
	mat.insert_elm(3, 5, randomdouble());
	mat.insert_elm(4, 5, randomdouble());
	mat.insert_elm(3, 6, randomdouble());
	mat.insert_elm(4, 6, randomdouble());

	mat.insert_elm(6, 4, randomdouble());
	mat.insert_elm(6, 3, randomdouble());

	mat.insert_elm(5, 4, randomdouble());
	mat.insert_elm(5, 3, randomdouble());

	mat.insert_elm(5, 2, randomdouble());
	mat.insert_elm(6, 2, randomdouble());
	*/
	mat.insert_elm(0, 2, randomdouble());
	mat.insert_elm(1, 2, randomdouble());


	mat.insert_elm(0, 3, randomdouble());
	mat.insert_elm(1, 3, randomdouble());


	mat.insert_elm(3, 2, randomdouble());



	RNN rnn(ini_nodes, out_nodes, nodes, learning_rate, depth, batch,  act_f, mat, handle);
	rnn.initialize_device_public();

	std::vector<Vector > inp = { {0, 0}, { 0,1 }, { 1,0 }, { 1,1 } };

	Vector sol = { 0 };

	for (int it = 0; it < 30000; it++) {
		error_sum = 0;
		for (auto input : inp )
		{
			sol[0] = 0;
			if (input[0] != input[1])
				sol[0] = 1;

			//rnn.print_matrix();

			error_sum += rnn.train_step(input, sol, false);

			rnn.reset();
			//std::cout << "---------------------\n";
		}

		if (it % 100 == 0) {
			std::cout << "Iteration " << it << ": " << error_sum / batch << std::endl;
			if(it%1000 == 0)rnn.print_matrix();
		}

		rnn.update_weights_and_biases();
	}

	for (auto input : inp)
	{
		sol[0] = 0;
		if (input[0] != input[1]) {
			//printf("%f %f\n",input[0], input[1]);
			sol[0] = 1;
		}


		//rnn.print_matrix();

		for (int i = 0; i < rnn.max_depth; ++i)
		{
			//std::cout << "Time " << i << " : " << time<<std::endl;
			rnn.set_input_values(input);
			rnn.forward_prop();
		}

		std::cout << "input: " << input[0] << " " << input[1] << " res: " << rnn.values_through_time[rnn.values_through_time.size()-1][2] << std::endl;

		rnn.reset();
	}
	
} 

void RNN_mnist_problem()
{
	cusparseHandle_t handle;
	cusparseCreate(&handle);
	int ini_nodes = 28*28;
	int out_nodes = 10;
	int nodes = ini_nodes+out_nodes+100;
	int batch = 10;
	int depth = 5;
	double learning_rate = 0.01;
	double error_sum = 0;
	std::vector<activation_function> act_f(nodes, RELU);

	for (int i = ini_nodes; i < ini_nodes + out_nodes; ++i)
		act_f[i] = LINEAL;

	for (int i = ini_nodes + out_nodes; i < nodes; ++i)
		act_f[i] = RELU;


	COO_matrix mat;

	for (int i = 0; i < ini_nodes; ++i) {

		for (int j = ini_nodes; j < ini_nodes + out_nodes; ++j)
			mat.insert_elm(i, j, randomdouble()/10);

		for (int j = ini_nodes + out_nodes; j < nodes; ++j)
			mat.insert_elm(i, j, randomdouble()/10);
	}

	for (int i = ini_nodes + out_nodes; i < nodes; ++i) {
		for (int j = ini_nodes; j < ini_nodes + out_nodes; ++j)
			mat.insert_elm(i, j, randomdouble()/10);
		if (i < nodes-50)
			for (int j = ini_nodes + out_nodes+50; j < nodes; ++j)
				mat.insert_elm(i, j, randomdouble()/10);
		else
			for (int j = ini_nodes; j < ini_nodes + out_nodes; ++j)
				mat.insert_elm(i, j, randomdouble()/10);
	}

	for (int i = 0; i < 10000; ++i)
	{
		mat.insert_elm(rand() % nodes, rand() % nodes, randomdouble()/10);
	}
	
	
	RNN rnn(ini_nodes, out_nodes, nodes, learning_rate, depth, batch, act_f, mat, handle);
	rnn.initialize_device_public();
	rnn.adam = true;
	auto inp = mnist::read_training_images();

	auto sol_list = mnist::read_training_labels();

	std::vector<int> indices(inp.size(), 0);
	for (int i = 0; i < inp.size(); ++i)
		indices[i] = i;

	int examples = inp.size();
	Vector input(28 * 28);
	Vector sol(10);
	for (int it = 0; it < 10; it++) {
		
		int print_stop = examples / 10;
		for (int image = 0; image < examples; ++image)
		{
			int image_ind = indices[image];

			//rnn.print_matrix();
			auto raw_input = inp[image_ind];
			int i_sol = sol_list[image_ind];

			for (int i = 0; i < raw_input.size(); ++i)
				input[i] = ((double)raw_input[i])/255;

			for (int i = 0; i < sol.size(); ++i)
				sol[i] = 0;
			sol[i_sol] = 1;

			error_sum += rnn.train_step(input, sol, true);
			//::cout << "hmmmmm" << std::endl;
			if (image % batch == 0) {
				//if (it % 1000 == 0)rnn.print_matrix();
				rnn.update_weights_and_biases();
			}
			if (examples > 1000 && image > print_stop)
			{
				std::cout<<"%"<<100 * image / examples<< " error = "<<100*error_sum / image<<std::endl;
				print_stop += examples / 10;
			}
			rnn.reset();
			//std::cout << "---------------------\n";
		}
		//rnn.print_matrix();
		std::random_shuffle(indices.begin(), indices.end());
		std::cout << "Iteration " << it << ": " << 100*error_sum / examples << std::endl;

		error_sum = 0;

	}

	auto test_inp = mnist::read_test_images();

	auto test_sol_list = mnist::read_test_labels();
	examples = test_inp.size();

	int print_stop = examples / 10;
	for (int image = 0; image < examples; ++image)
	{
		//rnn.print_matrix();
		auto raw_input = test_inp[image];
		int i_sol = test_sol_list[image];

		for (int i = 0; i < raw_input.size(); ++i)
			input[i] = ((double)raw_input[i]) / 255;

		for (int i = 0; i < sol.size(); ++i)
			sol[i] = 0;
		sol[i_sol] = 1;

		error_sum += rnn.train_step(input, sol, true);
		
		if (examples > 1000 && image > print_stop)
		{
			std::cout << "%" << 100 * image / examples << " error = " << 100*error_sum / image << std::endl;
			print_stop += examples / 10;
		}
		rnn.reset();
		//std::cout << "---------------------\n";
	}
	//rnn.print_matrix();
	std::cout << "Test: " << 100*error_sum / examples << std::endl;

	for (int image = 0; image < 10; ++image) {
		for (int j = 0; j < 28; j++) {
			for (int u = 0; u < 28; u++) {
				if (test_inp[image][j * 28 + u] > 0) std::cout << "X ";
				else std::cout << "_ ";
			}
			std::cout << std::endl;
		}
		std::cout << std::endl;

		auto raw_input = test_inp[image];
		std::cout << "solution: " << (int)test_sol_list[image] << std::endl;

		for (int i = 0; i < raw_input.size(); ++i) {
			input[i] = ((double)raw_input[i]) / 255;
		}

		rnn.train_step(input, sol, true);

		for (int i = 0; i < 10; ++i)
			std::cout << "i: " << i << " value: " << rnn.values_through_time[rnn.values_through_time.size()- 1][ini_nodes + i] << std::endl;
		rnn.reset();
	}

}

void RNN_mnist_problem( RNN& rnn, int exampl, int iters, bool test)
{
	double error_sum = 0;
	int ini_nodes = rnn.ini;
	int batch = rnn.batch;
	auto inp = mnist::read_training_images();

	auto sol_list = mnist::read_training_labels();

	std::vector<int> indices(inp.size(), 0);
	for (int i = 0; i < inp.size(); ++i)
		indices[i] = i;

	int examples = exampl;
	if (examples == -1)
		examples = inp.size();

	Vector input(28 * 28);
	Vector sol(10);
	for (int it = 0; it < iters; it++) {
		std::random_shuffle(indices.begin(), indices.end());
		int print_stop = examples / 10;
		for (int image = 0; image < examples; ++image)
		{
			int image_ind = indices[image];

			//rnn.print_matrix();
			auto raw_input = inp[image_ind];
			int i_sol = sol_list[image_ind];

			for (int i = 0; i < raw_input.size(); ++i)
				input[i] = ((double)raw_input[i]) / 255;

			for (int i = 0; i < sol.size(); ++i)
				sol[i] = 0;
			sol[i_sol] = 1;

			error_sum += rnn.train_step(input, sol, true);
			//::cout << "hmmmmm" << std::endl;
			if (image % batch == 0) {
				//if (it % 1000 == 0)rnn.print_matrix();
				rnn.update_weights_and_biases();
			}
			if (examples > 1000 && image > print_stop)
			{
				std::cout << "%" << 100 * image / examples << " error = " << 100*error_sum / image << std::endl;
				print_stop += examples / 10;
			}
			rnn.reset();
			//std::cout << "---------------------\n";
		}
		//rnn.print_matrix();
		std::cout << "Iteration " << it << ": " << 100*error_sum / examples << std::endl;

		error_sum = 0;

	}
	if (!test) return;

	auto test_inp = mnist::read_test_images();

	auto test_sol_list = mnist::read_test_labels();
	examples = test_inp.size();

	int print_stop = examples / 10;
	for (int image = 0; image < examples; ++image)
	{
		//rnn.print_matrix();
		auto raw_input = test_inp[image];
		int i_sol = test_sol_list[image];

		for (int i = 0; i < raw_input.size(); ++i)
			input[i] = ((double)raw_input[i]) / 255;

		for (int i = 0; i < sol.size(); ++i)
			sol[i] = 0;
		sol[i_sol] = 1;

		error_sum += rnn.train_step(input, sol, true);

		if (examples > 1000 && image > print_stop)
		{
			std::cout << "%" << 100 * image / examples << " error = " << 100*error_sum / image << std::endl;
			print_stop += examples / 10;
		}
		rnn.reset();
		//std::cout << "---------------------\n";
	}
	//rnn.print_matrix();
	std::cout << "Test: " << 100*error_sum / examples << std::endl;

	for (int image = 0; image < 10; ++image) {
		for (int j = 0; j < 28; j++) {
			for (int u = 0; u < 28; u++) {
				if (test_inp[image][j * 28 + u] > 0) std::cout << "X ";
				else std::cout << "_ ";
			}
			std::cout << std::endl;
		}
		std::cout << std::endl;

		auto raw_input = test_inp[image];
		std::cout << "solution: " << (int)test_sol_list[image] << std::endl;

		for (int i = 0; i < raw_input.size(); ++i) {
			input[i] = ((double)raw_input[i]) / 255;
		}

		rnn.train_step(input, sol, true);

		for (int i = 0; i < 10; ++i)
			std::cout << "i: " << i << " value: " << rnn.values_through_time[rnn.values_through_time.size() - 1][ini_nodes + i] << std::endl;
		rnn.reset();
	}

}

void RNN_mnist_autoencode_problem()
{
	int init_time = time(0);
	cusparseHandle_t handle;
	cusparseCreate(&handle);
	int ini_nodes = 28 * 28;
	int out_nodes = 28 * 28;
	int nodes = ini_nodes + out_nodes+50;
	int batch = 10;
	int depth = 5;
	double learning_rate = 0.001;
	double error_sum = 0;
	std::vector<activation_function> act_f(nodes, RELU);

	for (int i = ini_nodes; i < ini_nodes + out_nodes; ++i)
		act_f[i] = RELU;

	for (int i = ini_nodes + out_nodes; i < nodes; ++i)
		act_f[i] = RELU;


	COO_matrix mat;

	for (int i = 0; i < ini_nodes; ++i) {

		for (int j = ini_nodes + out_nodes; j < nodes; ++j)
			mat.insert_elm(i, j, randomdouble()/10);

	//	mat.insert_elm(i, i+ini_nodes, 1);
	}
	std::cout<<"from ini to out all connected"<<std::endl;

	for (int i = ini_nodes + out_nodes; i < nodes; ++i) {
		for (int j = ini_nodes; j < ini_nodes + out_nodes; ++j)
			mat.insert_elm(i, j, randomdouble() / 10);

		for (int j = ini_nodes + out_nodes; j < nodes; ++j)
			mat.insert_elm(i, j, randomdouble() / 10);
		/*
		if (i < nodes - 50)
			for (int j = ini_nodes + out_nodes + 50; j < nodes; ++j)
				mat.insert_elm(i, j, randomdouble() / 10);
		else
			for (int j = ini_nodes; j < ini_nodes + out_nodes; ++j)
				mat.insert_elm(i, j, randomdouble() / 10);*/
	}
	std::cout<<"others connected"<<std::endl;

	/*for (int i = 0; i < 10000; ++i)
	{
		mat.insert_elm(rand() % nodes, rand() % nodes, randomdouble());
	}
	*/

	RNN rnn(ini_nodes, out_nodes, nodes, learning_rate, depth, batch, act_f, mat, handle);
	rnn.initialize_device_public();

	auto inp = mnist::read_training_images();

	std::vector<int> indices(inp.size(), 0);
	for (int i = 0; i < inp.size(); ++i)
		indices[i] = i;

	int examples = inp.size();
	int resets = 0;
	Vector input(28 * 28);
	for (int it = 0; it < 1; it++) {

		int print_stop = examples / 10;
		for (int image = 0; image < examples; ++image)
		{
			int image_ind = indices[image];

			//rnn.print_matrix();
			auto raw_input = inp[image];

			for (int i = 0; i < raw_input.size(); ++i)
				input[i] = ((double)raw_input[i]);

			error_sum += rnn.train_step(input, input, false);
			//::cout << "hmmmmm" << std::endl;
			if (false || image % batch == 0) {
				//if (it % 1000 == 0)rnn.print_matrix();
				rnn.update_weights_and_biases();
			}
			if (examples > 1000 && image > print_stop)
			{
				std::cout << "%" << 100 * image / examples << " error = " << error_sum / image << std::endl;
				print_stop += examples / 10;
			}
			resets += 1;
			rnn.reset();
			//std::cout << "---------------------\n";
		}
		//rnn.print_matrix();
		std::random_shuffle(indices.begin(), indices.end());
		std::cout << "Iteration " << it << ": " << error_sum / examples << std::endl;

		error_sum = 0;

	}

	std::cout << "Time: " << time(0) - init_time << std::endl;
	std::cout << "Resets: " << resets << std::endl;
	return;
	/*
	auto test_inp = mnist::read_test_images();

	examples = test_inp.size();

	int print_stop = examples / 10;
	for (int image = 0; image < examples; ++image)
	{
		//rnn.print_matrix();
		auto raw_input = test_inp[image];

		for (int i = 0; i < raw_input.size(); ++i)
			input[i] = ((double)raw_input[i]);

		error_sum += rnn.train_step(input, input, false);

		if (examples > 1000 && image > print_stop)
		{
			std::cout << "%" << 100 * image / examples << " error = " << error_sum / image << std::endl;
			print_stop += examples / 10;
		}
		rnn.reset();
		//std::cout << "---------------------\n";
	}
	//rnn.print_matrix();
	std::cout << "Test: " << error_sum / examples << std::endl;

	for (int image = 0; image < 10; ++image) {
		for (int j = 0; j < 28; j++) {
			for (int u = 0; u < 28; u++) {
				if (test_inp[image][j * 28 + u] > 0) std::cout << "X ";
				else std::cout << "_ ";
			}
			std::cout << std::endl;
		}
		std::cout << std::endl;

		auto raw_input = test_inp[image];
		for (int i = 0; i < raw_input.size(); ++i) {
			input[i] = ((double)raw_input[i]);
		}

		rnn.train_step(input, input, false);

		for (int j = 0; j < 28; j++) {
			for (int u = 0; u < 28; u++) {
				if (rnn.values_through_time[rnn.values_through_time.size() - 1][ini_nodes + j * 28 + u] > 20) std::cout << "X ";
				else std::cout << "_ ";
			}
			std::cout << std::endl;
		}
		std::cout << std::endl;
		rnn.reset();
	}*/

}

void RNN_sum_vect_problem()
{
	cusparseHandle_t handle;
	cusparseCreate(&handle);
	int ini_nodes = 1;
	int out_nodes = 1;
	int nodes = ini_nodes + out_nodes;
	int batch = 100;
	int iters = 10;
	double learning_rate = 1;
	double error_sum = 0;

	std::vector<activation_function> act_f(nodes, LINEAL);

	for (int i = ini_nodes; i < ini_nodes + out_nodes; ++i)
		act_f[i] = LINEAL;
	for (int i = ini_nodes + out_nodes; i < nodes; ++i)
		act_f[i] = LINEAL;

	COO_matrix mat;

	/*mat.insert_elm(0, 0, -1);
	mat.insert_elm(0, 1, 1);
	mat.insert_elm(1, 1, 1);*/
	mat.insert_elm(0, 0, randomdouble());
	mat.insert_elm(0, 1, randomdouble());
	mat.insert_elm(1, 1, randomdouble());
	/*mat.insert_elm(0, 0, randomdouble());
	mat.insert_elm(0, 1, randomdouble());
	mat.insert_elm(1, 1, randomdouble());*/
	/*for (int i = 0; i < nodes*nodes*0.1; ++i)
	{
		mat.insert_elm(rand() % nodes, rand() % nodes, randomdouble());
	}*/



	RNN rnn(ini_nodes, out_nodes, nodes, learning_rate, iters, batch, act_f, mat, handle);
	rnn.initialize_device_public();
	
	rnn.max_depth = iters*2;
	rnn.bias_learning_rate = 0;
	rnn.delay_iters = 0;
	int examples = 10;
	Matrix inp(examples, Vector(1, 0));

	Matrix sol(examples, Vector(1, 0));
	
	for (int i = 0; i < examples; ++i)
		inp[i][0] = randomdouble();

	sol[0][0] = inp[0][0];
	for (int i = 1; i < examples; ++i) {
		sol[i][0] = inp[i][0] + sol[i - 1][0];
	}

	error_sum = 0;
	for (int it = 0; it < 50000; it++) {

		error_sum += rnn.time_sens_train_step(inp, sol, false);

		if (it * examples % batch == 0) {
			std::cout << "Iteration " << it << ": " << error_sum * examples / batch << std::endl;
			rnn.update_weights_and_biases();
			//rnn.print_matrix();
			error_sum = 0;
		}
		rnn.reset();
	}
	rnn.print_matrix();
}

void RNN_previous_sign_problem()
{
	cusparseHandle_t handle;
	cusparseCreate(&handle);
	int ini_nodes = 1;
	int out_nodes = 1;
	int nodes = ini_nodes+out_nodes;
	int batch = 100;
	int iters = 1;
	double learning_rate = 1;
	double error_sum = 0;

	std::vector<activation_function> act_f(nodes, LINEAL);

	for (int i = ini_nodes; i < ini_nodes + out_nodes; ++i)
		act_f[i] = LINEAL;
	for (int i = ini_nodes + out_nodes; i < nodes; ++i)
		act_f[i] = SIGMOID;
   
	COO_matrix mat;

	mat.insert_elm(0, 1, randomdouble());
	mat.insert_elm(1, 1, randomdouble());

	//mat.insert_elm(0, 1, 1000);
	//mat.insert_elm(1, 1, 1);
	/*for (int i = 0; i < nodes*nodes*0.1; ++i)
	{
		mat.insert_elm(rand() % nodes, rand() % nodes, randomdouble());
	}*/



	RNN rnn(ini_nodes, out_nodes, nodes, learning_rate, iters, batch, act_f, mat, handle);
	rnn.initialize_device_public();

	rnn.max_depth = 2;

	int examples = 100;
	Matrix inp(examples, Vector(1,0));

	Matrix sol(examples, Vector(1,0));
	error_sum = 0;
	for (int it = 0; it < 1000; it++) {
		for (int i = 0; i < examples; ++i)
			inp[i][0] = randomdouble();

		for (int i = 1; i < examples; ++i) {
			sol[i][0] = sign(inp[i-1][0]);
			if (sol[i][0] < 0)
				sol[i][0] = 0;
		}

		error_sum += rnn.time_sens_train_step(inp, sol, false);

		if (it*examples % batch == 0) {
			std::cout << "Iteration " << it << ": " << error_sum*examples/batch << std::endl;
			rnn.update_weights_and_biases();

			error_sum = 0;
		}
		rnn.reset();
	}
	rnn.print_matrix();
}

void RNN_shakespeare_problem() {

	cusparseHandle_t handle;
	cusparseCreate(&handle);
	int ini_nodes = 255;
	int out_nodes = 255;
	int nodes = ini_nodes + out_nodes;
	int batch = 100;
	int iters = 2;
	double learning_rate = 1;
	double error_sum = 0;

	std::vector<activation_function> act_f(nodes, LINEAL);

	for (int i = ini_nodes; i < ini_nodes + out_nodes; ++i)
		act_f[i] = LINEAL;
	for (int i = ini_nodes + out_nodes; i < nodes; ++i)
		act_f[i] = LINEAL;

	COO_matrix mat;

	/*mat.insert_elm(0, 0, -1);
	mat.insert_elm(0, 1, 1);
	mat.insert_elm(1, 1, 1);*/
	mat.insert_elm(0, 0, randomdouble());
	mat.insert_elm(0, 1, randomdouble());
	mat.insert_elm(1, 1, randomdouble());
	/*mat.insert_elm(0, 0, randomdouble());
	mat.insert_elm(0, 1, randomdouble());
	mat.insert_elm(1, 1, randomdouble());*/
	/*for (int i = 0; i < nodes*nodes*0.1; ++i)
	{
		mat.insert_elm(rand() % nodes, rand() % nodes, randomdouble());
	}*/



	RNN rnn(ini_nodes, out_nodes, nodes, learning_rate, iters, batch, act_f, mat, handle);
	rnn.initialize_device_public();

	rnn.max_depth = iters * 2;
	rnn.bias_learning_rate = 0;

	int examples = 3;
	Matrix inp(examples, Vector(1, 0));

	Matrix sol(examples, Vector(1, 0));
	for (int i = 0; i < examples; ++i)
		inp[i][0] = randomdouble();

	sol[0][0] = inp[0][0];
	for (int i = 1; i < examples; ++i) {
		sol[i][0] = inp[i][0] + sol[i - 1][0];
	}

	error_sum = 0;
	for (int it = 0; it < 50000; it++) {

		error_sum += rnn.time_sens_train_step(inp, sol, false);

		if (it * examples % batch == 0) {
			std::cout << "Iteration " << it << ": " << error_sum * examples / batch << std::endl;
			rnn.update_weights_and_biases();
			rnn.print_matrix();
			error_sum = 0;
		}
		rnn.reset();
	}
	rnn.print_matrix();
}

void RNN_delayed_str_problem()     {
	cusparseHandle_t handle;
	cusparseCreate(&handle);
	std::string alphabet = "abcdefghijklmopqrstuvwxyz1234567890 ,";
	std::vector<int> s = { 1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1 };
	int ini_nodes = alphabet.size();
	int out_nodes = alphabet.size();
	int nodes = ini_nodes + out_nodes + 200;
	int batch = s.size();
	int iters = 1;
	double learning_rate = 0.1;
	double error_sum = 0;

	std::vector<activation_function> act_f(nodes, RELU);

	for (int i = ini_nodes; i < ini_nodes + out_nodes; ++i)
		act_f[i] = LINEAL;
	for (int i = ini_nodes + out_nodes; i < nodes; ++i)
		act_f[i] = SIGMOID;

	COO_matrix mat;

	for (int i = 0; i < ini_nodes; ++i) {

		for (int j = ini_nodes + out_nodes; j < nodes; ++j)
			mat.insert_elm(i, j, randomdouble() / 10);

		/*for (int j = ini_nodes; j < ini_nodes+out_nodes; ++j)
			mat.insert_elm(i, j, randomdouble() / 10);*/
	}

	for (int i = ini_nodes + out_nodes; i < nodes; ++i) {

		for (int j = ini_nodes; j < ini_nodes + out_nodes; ++j)
			mat.insert_elm(i, j, randomdouble() / 10);

		for (int j = ini_nodes + out_nodes; j < nodes; ++j)
			mat.insert_elm(i, j, randomdouble() / 10);
	}

	RNN rnn(ini_nodes, out_nodes, nodes, learning_rate, iters, batch, act_f, mat, handle);
	rnn.initialize_device_public();
	rnn.adam = true;
	//std::string s = "I am a delayed string i will appear some time after being read, the generator must remember my characters in order to output the correct values.";

	//s = "I am a delayed string";
	//s = "aa";
	int window_size = 7;
	rnn.max_depth = window_size+1;
	rnn.delay_iters = window_size;
	batch = s.size();




	for (int it = 0; it < 30000; it++) {


		for (int i = 0; i < 36; ++i)
			s[i] = rand() % alphabet.size();


		Matrix inp(s.size(), Vector(alphabet.size(), 0));

		Matrix sol(s.size(), Vector(alphabet.size(), 0));

		for (int i = 0; i < s.size(); ++i) {
			inp[i][s[i]] = 1;
		}
		for (int i = 0; i < s.size(); ++i)
			sol[i][s[i]] = 1;
		error_sum = 0;
		error_sum += rnn.time_sens_train_step(inp, sol, true);

		if (it*s.size()  % batch == 0) {
			std::cout << "Iteration " << it << ": " << error_sum * s.size() / batch << std::endl;
			rnn.update_weights_and_biases();
			error_sum = 0;
		}
		rnn.reset();
	}
}

bool RNN_caltech101_sil_problem(RNN& rnn, int exampl, int iters, bool test, std::vector<std::vector<int>> &inp, std::vector<int> &sol_list, 
																			std::vector<std::vector<int>> &test_inp, std::vector<int> &test_sol_list,
																			std::vector<std::vector<int>> &val_inp, std::vector<int>  &val_sol_list)
{
	srand(time(0));
	double val_error_sum = 0;
	double error_sum = 0;
	int ini_nodes = rnn.ini;
	int batch = rnn.batch;



	int examples = exampl;
	if (examples == -1)
		examples = inp.size();

	int validation_examples = val_inp.size();

	if (examples < 1000) 
		validation_examples = 0;
	

	std::vector<int> indices(inp.size(), 0);
	for (int i = 0; i < inp.size(); ++i)
		indices[i] = i;


	Vector input(28 * 28);
	Vector sol(101);

	int it = 0;
	val_error_sum = validation_examples;
	error_sum = examples;
	while (it < iters && ( validation_examples == 0 || val_error_sum/validation_examples <= error_sum/examples*1.1) ) {
		error_sum = 0;
		val_error_sum = 0;
		std::random_shuffle(indices.begin(), indices.end());
		int print_stop = examples / 10;
		for (int image = 0; image < examples; ++image)
		{
			int image_ind = indices[image];
			//rnn.print_matrix();
			auto raw_input = inp[image_ind];
			int i_sol = sol_list[image_ind];

			for (int i = 0; i < raw_input.size(); ++i)
				input[i] = ((double)raw_input[i]);

			for (int i = 0; i < sol.size(); ++i)
				sol[i] = 0;
			sol[i_sol-1] = 1;

			error_sum += rnn.train_step(input, sol, true);
			//::cout << "hmmmmm" << std::endl;
			if (image % batch == 0) {
				//if (it % 1000 == 0)rnn.print_matrix();
				rnn.update_weights_and_biases();
			}
			if (examples > 1000 && image > print_stop)
			{
				std::cout << "%" << 100 * image / examples << " error = " << 100 * error_sum / image << std::endl;
				print_stop += examples / 10;
			}
			rnn.reset();
			//std::cout << "---------------------\n";
		}
		//rnn.print_matrix();



		//Validation
		if(validation_examples > 0)
		{
			for (int image = 0; image < validation_examples; ++image)
			{

				//rnn.print_matrix();
				auto raw_input = val_inp[image];
				int i_sol = val_sol_list[image];

				for (int i = 0; i < raw_input.size(); ++i)
					input[i] = ((double)raw_input[i]);

				for (int i = 0; i < sol.size(); ++i)
					sol[i] = 0;
				sol[i_sol - 1] = 1;

				val_error_sum += rnn.forward_prop_cycle(input, sol, true);
				rnn.reset();
				//std::cout << "---------------------\n";
			}
		}

		std::cout << "Iteration " << it << ": " << 100 * error_sum / examples << ", Validation: " << 100 * val_error_sum / validation_examples <<std::endl;
		it++;
	}
	if (!test) return val_error_sum / validation_examples <= error_sum / examples * 1.1;
	
	examples = test_inp.size();
	double test_error_sum = 0;

	int print_stop = examples / 10;
	for (int image = 0; image < examples; ++image)
	{
		//rnn.print_matrix();
		auto raw_input = test_inp[image];
		int i_sol = test_sol_list[image];

		for (int i = 0; i < raw_input.size(); ++i)
			input[i] = ((double)raw_input[i]);

		for (int i = 0; i < sol.size(); ++i)
			sol[i] = 0;
	    sol[i_sol-1] = 1;

		test_error_sum += rnn.forward_prop_cycle(input, sol, true);

		if (examples > 1000 && image > print_stop)
		{
			std::cout << "%" << 100 * image / examples << " error = " << 100 * test_error_sum / image << std::endl;
			print_stop += examples / 10;
		}
		rnn.reset();
		//std::cout << "---------------------\n";
	}
	//rnn.print_matrix();
	std::cout << "Test: " << 100 * test_error_sum / examples << std::endl;

	for (int image = 0; image < 100; ++image) {
		for (int j = 0; j < 28; j++) {
			for (int u = 0; u < 28; u++) {
				if (test_inp[image][u * 28 + j] > 0) std::cout << "X ";
				else std::cout << "_ ";
			}
			std::cout << std::endl;
		}
		std::cout << std::endl;

		auto raw_input = test_inp[image];
		std::cout << "solution: " << (int)test_sol_list[image]-1 << std::endl;

		for (int i = 0; i < raw_input.size(); ++i) {
			input[i] = ((double)raw_input[i]);
		}

		rnn.forward_prop_cycle(input, sol, true);

		for (int i = 0; i < 101; ++i)
			if(rnn.values_through_time[rnn.values_through_time.size() - 1][ini_nodes + i] > 0.1)
				std::cout << "i: " << i << " value: " << rnn.values_through_time[rnn.values_through_time.size() - 1][ini_nodes + i] << std::endl;
		rnn.reset();
	}
	
	return val_error_sum / validation_examples <= error_sum / examples * 1.1;
}



void RNN_chess_problem(RNN& rnn, int exampl, int iters, Matrix &inp, Vector &sol_list)
{
	double error_sum = 0;
	int ini_nodes = rnn.ini;
	int batch = rnn.batch;
	
	std::vector<int> indices(inp.size(), 0);
	for (int i = 0; i < inp.size(); ++i)
		indices[i] = i;

	int examples = exampl;
	if (examples == -1)
		examples = inp.size();

	Vector input(rnn.ini);
	Vector sol(rnn.out);

	for (int it = 0; it < iters; it++) {
		std::random_shuffle(indices.begin(), indices.end());
		int print_stop = examples / 10;
		for (int image = 0; image < examples; ++image)
		{
			int image_ind = indices[image];

			//rnn.print_matrix();
			auto raw_input = inp[image_ind];

			for (int i = 0; i < raw_input.size(); ++i)
				input[i] = ((double)raw_input[i]);

			sol[0] = sol_list[image_ind];

			error_sum += rnn.train_step(input, sol, false);
			//::cout << "hmmmmm" << std::endl;
			if (image % batch == 0) {
				//if (it % 1000 == 0)rnn.print_matrix();
				rnn.update_weights_and_biases();
			}
			if (examples > 1000 && image > print_stop)
			{
				std::cout << "%" << 100 * image / examples << " error = " << error_sum / image << std::endl;
				print_stop += examples / 10;
			}
			rnn.reset();
			//std::cout << "---------------------\n";
		}
		//rnn.print_matrix();

		std::cout << "Iteration " << it << ": " << error_sum / examples << std::endl;

		error_sum = 0;

	}

}



bool generic_classif_RNN_problem(
	RNN& rnn, 
	int exampl, 
	int iters, 
	bool test,
	std::vector < std::pair<Vector, double> >& dataset, 
	std::vector < std::pair<Vector, double> >& test_data,
	std::vector < std::pair<Vector, double> >& val_data
){

	double aux_drop = rnn.inp_dropout;
	rnn.inp_dropout = 0;
	double val_error_sum = 0;
	double error_sum = 0;
	int ini_nodes = rnn.ini;
	int batch = rnn.batch;

	int examples = exampl;
	if (examples == -1)
		examples = dataset.size();

	int validation_examples = val_data.size();
	auto best_weights = rnn.weights;
	auto best_biases = rnn.biases;
	double best_val_score = validation_examples*100;

	std::vector<int> indices(dataset.size(), 0);
	for (int i = 0; i < dataset.size(); ++i)
		indices[i] = i;


	Vector input(rnn.ini);
	Vector sol(rnn.out);

	int it = 0;
	val_error_sum = validation_examples;
	error_sum = examples;
	double ant_val_error = val_error_sum;
	bool first_dip = true;
	int didnt_do_better = 0;
	while (it < iters) {
		error_sum = 0;
		first_dip = (ant_val_error > val_error_sum);
		if (val_error_sum < best_val_score) {
			std::cout << "Better parameters found!" << std::endl;
			rnn.read_weights_and_biases_from_device(best_weights.cooValues, best_biases);
			best_val_score = val_error_sum;
			didnt_do_better = 0;
		}
		else didnt_do_better++;

		if (didnt_do_better > 7) break;

		ant_val_error = val_error_sum;
		val_error_sum = 0;
		std::random_shuffle(indices.begin(), indices.end());
		int print_stop = examples / 10;
		for (int image = 0; image < examples; ++image)
		{
			int image_ind = indices[image];
			//rnn.print_matrix();
			auto raw_input = dataset[image_ind].first;
			
			int i_sol = dataset[image_ind].second;

			for (int i = 0; i < raw_input.size(); ++i)
				input[i] = ((double)raw_input[i]);

			for (int i = 0; i < sol.size(); ++i)
				sol[i] = 0;
			sol[i_sol] = 1;
			

			error_sum += rnn.train_step(input, sol, true);
			//::cout << "hmmmmm" << std::endl;
			if (image % batch == 0) {
				//if (it % 1000 == 0)rnn.print_matrix();
				rnn.update_weights_and_biases();
			}
			if (examples > 1000 && image > print_stop)
			{
				std::cout << "%" << 100 * image / examples << " error = " << 100 * error_sum / image << std::endl;
				print_stop += examples / 10;
			}
			rnn.reset();
			//std::cout << "---------------------\n";
		}
		//rnn.print_matrix();

		//Validation
		if (validation_examples > 0)
		{
			for (int image = 0; image < validation_examples; ++image)
			{

				//rnn.print_matrix();
				auto raw_input = val_data[image].first;
				int i_sol = val_data[image].second;

				for (int i = 0; i < raw_input.size(); ++i)
					input[i] = ((double)raw_input[i]);

				for (int i = 0; i < sol.size(); ++i)
					sol[i] = 0;
				sol[i_sol] = 1;

				val_error_sum += rnn.forward_prop_cycle(input, sol, true);
				rnn.reset();
				//std::cout << "---------------------\n";
			}
		}
		rnn.inp_dropout = aux_drop;
		std::cout << "Iteration " << it << ": " << 100 * error_sum / examples << ", Validation: " << 100 * val_error_sum / validation_examples << std::endl;
		it++;
	}

	rnn.weights = best_weights;
	rnn.biases = best_biases;
	rnn.copy_wb_to_dev();

	if (!test) return val_error_sum / validation_examples <= error_sum / examples *1.1;

	examples = test_data.size();
	double test_error_sum = 0;

	int print_stop = examples / 10;
	for (int image = 0; image < examples; ++image)
	{
		//rnn.print_matrix();
		auto raw_input = test_data[image].first;
		int i_sol = test_data[image].second;

		for (int i = 0; i < raw_input.size(); ++i)
			input[i] = ((double)raw_input[i]);

		for (int i = 0; i < sol.size(); ++i)
			sol[i] = 0;
		sol[i_sol] = 1;

		test_error_sum += rnn.forward_prop_cycle(input, sol, true);

		if (examples > 1000 && image > print_stop)
		{
			std::cout << "%" << 100 * image / examples << " error = " << 100 * test_error_sum / image << std::endl;
			print_stop += examples / 10;
		}
		rnn.reset();
		//std::cout << "---------------------\n";
	}
	//rnn.print_matrix();
	std::cout << "Test: " << 100 * test_error_sum / examples << std::endl;

	if (rnn.ini == 28 * 28) {
		int point = rand() % (test_data.size()-10);
		for (int i = point; i < point+10; ++i){
			for (int j = 0; j < 28; j++) {
				for (int u = 0; u < 28; u++) {
					if (test_data[i].first[j * 28 + u] > 0.75) std::cout << "X ";
					else if (test_data[i].first[j * 28 + u] > 0.5) std::cout << "x ";
					else if (test_data[i].first[j * 28 + u] > 0.25) std::cout << "+ ";
					else if (test_data[i].first[j * 28 + u] > 0) std::cout << "- ";
					else std::cout << "_ ";
				}
				std::cout << std::endl;
			}
			std::cout << std::endl;

			auto raw_input = test_data[i].first;
			std::cout << "solution: " << (int)test_data[i].second << std::endl;

			for (int i = 0; i < raw_input.size(); ++i) {
				input[i] = ((double)raw_input[i]);
			}

			rnn.forward_prop_cycle(input, sol, true);

			for (int i = 0; i < rnn.out; ++i)
				if (rnn.values_through_time[rnn.values_through_time.size() - 1][ini_nodes + i] > 0.1)
					std::cout << "i: " << i << " value: " << rnn.values_through_time[rnn.values_through_time.size() - 1][ini_nodes + i] << std::endl;
			rnn.reset();
		}
	}

	return val_error_sum / validation_examples <= error_sum / examples * 1.1;
}



bool generic_regress_RNN_problem(RNN& rnn, int exampl, int iters, bool test,
	std::vector < std::pair<Vector, double> >& dataset, std::vector < std::pair<Vector, double> >& test_data,
	std::vector < std::pair<Vector, double> >& val_data) {

	double aux_drop = rnn.inp_dropout;
	rnn.inp_dropout = 0;
	double val_error_sum = 0;
	double error_sum = 0;
	int ini_nodes = rnn.ini;
	int batch = rnn.batch;

	int examples = exampl;
	if (examples == -1)
		examples = dataset.size();

	int validation_examples = val_data.size();
	auto best_weights = rnn.weights;
	auto best_biases = rnn.biases;
	double best_val_score = INFINITY;

	std::vector<int> indices(dataset.size(), 0);
	for (int i = 0; i < dataset.size(); ++i)
		indices[i] = i;


	Vector input(rnn.ini);
	Vector sol(rnn.out);

	int it = 0;
	val_error_sum = INFINITY;
	error_sum = examples;
	double ant_val_error = val_error_sum;
	bool first_dip = true;
	int didnt_do_better = 0;
	double error_sum_print = 0;
	while (it < iters) {
		error_sum = 0;
		error_sum_print = 0;
		first_dip = (ant_val_error > val_error_sum);
		if (val_error_sum < best_val_score) {
			std::cout << "Better parameters found!" << std::endl;
			rnn.read_weights_and_biases_from_device(best_weights.cooValues, best_biases);
			best_val_score = val_error_sum;
			didnt_do_better = 0;
		}
		else didnt_do_better++;

		if (didnt_do_better > 1) break;

		ant_val_error = val_error_sum;
		val_error_sum = 0;
		std::random_shuffle(indices.begin(), indices.end());
		int print_stop = examples / 10;
		for (int image = 0; image < examples; ++image)
		{
			int image_ind = indices[image];
			//rnn.print_matrix();
			auto raw_input = dataset[image_ind].first;

			sol[0] = dataset[image_ind].second;

			for (int i = 0; i < raw_input.size(); ++i)
				input[i] = ((double)raw_input[i]);

			double e = rnn.train_step(input, sol, false);
			error_sum += e;
			error_sum_print += e;

			//::cout << "hmmmmm" << std::endl;
			if (image % batch == 0) {
				//if (it % 1000 == 0)rnn.print_matrix();
				rnn.update_weights_and_biases();
			}
			if (examples > 1000 && image > print_stop)
			{
				std::cout <<"%" << double(it - 1) + double(image) / examples << " error = " << error_sum_print / (examples / 100) << std::endl;
				print_stop += examples / 100;
				error_sum_print = 0;
			}
			rnn.reset();
			//std::cout << "---------------------\n";
		}
		//rnn.print_matrix();

		//Validation
		if (validation_examples > 0)
		{
			for (int image = 0; image < validation_examples; ++image)
			{

				//rnn.print_matrix();
				auto raw_input = val_data[image].first;

				for (int i = 0; i < raw_input.size(); ++i)
					input[i] = ((double)raw_input[i]);

				sol[0] = val_data[image].second;

				val_error_sum += rnn.forward_prop_cycle(input, sol, false);
				rnn.reset();
				//std::cout << "---------------------\n";
			}
		}
		rnn.inp_dropout = aux_drop;
		std::cout << "Iteration " << it << ": " << error_sum / examples << ", Validation: " << val_error_sum / validation_examples << std::endl;
		it++;
	}

	rnn.weights = best_weights;
	rnn.biases = best_biases;
	rnn.copy_wb_to_dev();

	if (!test) return false;

	examples = test_data.size();
	double test_error_sum = 0;
	double test_error_sum_print = 0;

	int print_stop = examples / 10;
	for (int image = 0; image < examples; ++image)
	{
		//rnn.print_matrix();�
		auto raw_input = test_data[image].first;

		sol[0] = test_data[image].second;

		for (int i = 0; i < raw_input.size(); ++i)
			input[i] = ((double)raw_input[i]);

		double  e = rnn.forward_prop_cycle(input, sol, false);
		test_error_sum += e;
		test_error_sum_print += e;

		if (examples > 1000 && image > print_stop)
		{
			std::cout << "%" << 100*image / examples << " error = " << test_error_sum_print / (examples / 10) << std::endl;
			print_stop += examples / 10;
			test_error_sum_print = 0;
		}
		rnn.reset();
		//std::cout << "---------------------\n";
	}
	//rnn.print_matrix();
	std::cout << "Test: " << test_error_sum / examples << std::endl;

	return false;
}
