#include "RNN.cuh"


RNN::RNN()
{ }

RNN::RNN(int i, int o, int n, double lr, int m_d, int bat, std::vector<activation_function> &a_f,  COO_matrix &w, cusparseHandle_t&  hand)
{
	handle = &hand;

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
	current_values = std::vector<double>(nodes, 0);
	current_dx = std::vector<double>(nodes, 0);
	
	biases = std::vector<double>(nodes, 0);
	biases_update = std::vector<double>(nodes, 0);

	weights = w;
	weights_update = std::vector<double>(w.nnz, 0);

	values_through_time = std::vector<std::vector<double>>();
	dx_through_time = std::vector<std::vector<double>>();
}

RNN::~RNN(){
	if (dev_ini) {
		//std::cout << "Initialized" << std::endl;
		empty_device();
	}
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
	//Weights and input/output vectors
	cudaMalloc((void**)&dA_rows, weights.nnz * sizeof(int));
	cudaMalloc((void**)&dA_columns, weights.nnz * sizeof(int));
	cudaMalloc((void**)&dA_values, weights.nnz * sizeof(double));
	cudaMalloc((void**)&dX, nodes * sizeof(double));
	cudaMalloc((void**)&dY, nodes * sizeof(double));

	//Biases
	cudaMalloc((void**)&d_biases, nodes * sizeof(double));

	// Weights and biases update
	cudaMalloc((void**)&d_weights_update, weights.nnz * sizeof(double));
	cudaMalloc((void**)&d_biases_update, nodes * sizeof(double));

	// Gradient
	cudaMalloc((void**)&d_gradient, nodes * sizeof(double));
	cudaMalloc((void**)&d_dx, nodes * sizeof(double));

	//Adam
	cudaMalloc((void**)&d_m, weights.nnz * sizeof(double));
	cudaMalloc((void**)&d_v, weights.nnz * sizeof(double));
	cudaMalloc((void**)&d_m_corr, weights.nnz * sizeof(double));
	cudaMalloc((void**)&d_v_corr, weights.nnz * sizeof(double));

	// Input
	cudaMalloc((void**)&d_input, nodes * sizeof(double));

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
		CUSPARSE_MV_ALG_DEFAULT, &bufferSize);
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

void RNN::execute_matrix_vector_prod( cusparseOperation_t transpose , cusparseDnVecDescr_t vect, cusparseDnVecDescr_t output)
{
	if (weights.nnz == 0) return;
	cusparseSpMV(*handle, transpose,
		&alpha, matA, vect, &beta, output, CUDA_R_64F,
		CUSPARSE_MV_ALG_DEFAULT, dBuffer);
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

}

void RNN::execute_sub_biases_update()
{
	int size = nodes;
	if (size == 0) return;
	int blocks = get_blocks(size);
	int threads = get_threads(blocks, size);
	addKernel<<< blocks, threads >>>(d_biases_update, -1, d_gradient, size);
	cudaDeviceSynchronize();

	cudaError_t cudaStatus = cudaGetLastError();
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "Sub biases launch failed: %s\n", cudaGetErrorString(cudaStatus));
	}

}

void RNN::execute_add_biases()
{
	int size = nodes;
	if (size == 0) return;
	int blocks = get_blocks(size);
	int threads = get_threads(blocks, size);
	addKernel<<< blocks, threads >>>(dX, 1, d_biases, size);
	cudaDeviceSynchronize();

	cudaError_t cudaStatus = cudaGetLastError();
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "Add biases launch failed: %s\n", cudaGetErrorString(cudaStatus));
	}

}

void RNN::execute_add_input_values(std::vector<double>& input_values)
{
	int size = input_values.size();

	if (size == 0) return;
	copy_vect_to_device(input_values, d_input, size);
	int blocks = get_blocks(size);
	int threads = get_threads(blocks, size);
	addKernel<<< blocks, threads >>>(dX, 1, d_input, size);
	cudaDeviceSynchronize();

	cudaError_t cudaStatus = cudaGetLastError();
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "Add input launch failed: %s\n", cudaGetErrorString(cudaStatus));
	}

}

void RNN::execute_sub_weights_update()
{
	int size = weights.nnz;
	if (size == 0) return;
	//std::cout << "yes its here" << std::endl;
	int blocks = get_blocks(size);
	int threads = get_threads(blocks, size);
	TmultVectsKernel<<<blocks, threads>>>(d_weights_update, dA_rows, dA_columns, d_gradient, -1, dX, size);
	cudaDeviceSynchronize();
	cudaError_t cudaStatus = cudaGetLastError();
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "Update gradient launch failed: %s\n", cudaGetErrorString(cudaStatus));
	}

}

void RNN::execute_update_weights_and_biases()
{
	int size = weights.nnz;
	if (size == 0) return;
	int blocks = get_blocks(size);
	int threads = get_threads(blocks, size);
	addKernel<<< blocks, threads >>>(dA_values, learning_rate/batch, d_weights_update, size);
	cudaDeviceSynchronize();

	size = nodes;
	blocks = get_blocks(size);
	threads = get_threads(blocks, size);
	addKernel<<< blocks, threads >>>(d_biases, learning_rate/batch, d_biases_update, size);
	cudaDeviceSynchronize();
}

void RNN::execute_update_weights_and_biases_Adam()
{
	int size = weights.nnz;
	if (size == 0) return;
	int blocks = get_blocks(size);
	int threads = get_threads(blocks, size);
	powBeta1 *= beta1;
	powBeta2 *= beta2;
	updateAdamKernel <<<blocks, threads>>> (beta1, beta2, d_m, d_v, d_m_corr, d_v_corr, d_weights_update, size, powBeta1, powBeta2);
	cudaDeviceSynchronize();
	addWithAdamKernel<<<blocks, threads>>> (dA_values, learning_rate/batch, d_m_corr, d_v_corr, eps, size);
	cudaDeviceSynchronize();
	cudaError_t cudaStatus = cudaGetLastError();
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "Adam launch failed: %s\n", cudaGetErrorString(cudaStatus));
	}

	size = nodes;
	blocks = get_blocks(size);
	threads = get_threads(blocks, size);
	addKernel << < blocks, threads >> > (d_biases, learning_rate/ batch, d_biases_update, size);
	cudaDeviceSynchronize();
	cudaStatus = cudaGetLastError();
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "Biases update launch failed: %s\n", cudaGetErrorString(cudaStatus));
	}

}

void RNN::execute_use_activation_functions()
{
	int size = nodes;
	if (size == 0) return;
	int blocks = get_blocks(size);
	int threads = get_threads(blocks, size);
	useActKernel<<< blocks, threads >>>(dX, d_dx, d_act_f, size);
	cudaDeviceSynchronize();
	cudaError_t cudaStatus = cudaGetLastError();
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "Use activation func launch failed: %s\n", cudaGetErrorString(cudaStatus));
	}

}

void RNN::copy_vect_to_device( std::vector<double> &values, double* device_vect, int size )
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
	cudaDeviceSynchronize();
	cudaError_t cudaStatus = cudaGetLastError();
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "Reset vect launch failed: %s\n", cudaGetErrorString(cudaStatus));
	}

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
	cudaFree(d_act_f);
	cudaFree(d_dx);
}

void RNN::read_weights_of_device()
{
	if (weights.nnz == 0) return;
	cudaMemcpy(weights.cooValues.data(), dA_values, weights.nnz * sizeof(double),
		cudaMemcpyDeviceToHost);
}

void RNN::read_vect_of_device(std::vector<double>& result, double* device_vect, int size)
{
	if (size == 0) return;
	cudaMemcpy(result.data(), device_vect, size * sizeof(double),
		cudaMemcpyDeviceToHost);
 }

void RNN::read_weights_and_biases_from_device(std::vector<double>& weights_, std::vector<double>& biases_)
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

void RNN::add_connection(int node_i, int node_j, double value)
{
	weights.insert_elm(node_i, node_j, value);
	nodes++;
}

void RNN::reset()
{
	current_values = std::vector<double>(nodes, 0);
	current_dx = std::vector<double>(nodes, 0);

	values_through_time = std::vector<std::vector<double>>();
	dx_through_time = std::vector<std::vector<double>>();
	reset_vect(dX, nodes);

}

void RNN::use_activation_functions()
{
	execute_use_activation_functions();

	
	read_vect_of_device(current_values,            dX, nodes);
	read_vect_of_device(current_dx,              d_dx, nodes);

	values_through_time.push_back(current_values);
	dx_through_time.push_back(current_dx);
	
	if (max_depth < values_through_time.size())
	{
		values_through_time.erase(values_through_time.begin());
		dx_through_time.erase(dx_through_time.begin());
	}
}

void RNN::forward_prop()
{
	execute_add_biases();
	use_activation_functions();
	
	auto values = values_through_time[values_through_time.size()-1];
	
	//std::cout << "Before"<< std::endl;
	copy_vect_to_device(values, dX, nodes);
	//std::cout << values[0] << " " << values[1] << std::endl;*/
	execute_matrix_vector_prod(CUSPARSE_OPERATION_NON_TRANSPOSE, vecX, vecY);
	move_vect_of_device(dX, dY, nodes);
	//read_vect_of_device(values, dY, nodes);
	//std::cout << "After" << std::endl;
	//std::cout << values[0] << " " << values[1] << std::endl;

}

double mean(std::vector<double> &values)
{
	double sum = 0;
	for (double val : values)
		sum += val;
	return sum / values.size();
}

void RNN::softmax(std::vector<double>& values, int ini, int fi)
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
double RNN::backward_prop(std::vector<double> &correct_vals, bool classif)
{
	auto layer = values_through_time.size()-1;

	auto values = values_through_time[layer];

	std::vector<double> error(out,0);
	std::vector<double> gradient(nodes,0);
    
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

		read_vect_of_device(gradient, d_gradient, nodes);
		//gradient clipping
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
		/*double norm2 = 0;
		for (int i = 0; i < nodes; ++i)
			norm2 += gradient[i] * gradient[i];
		norm2 = sqrt(norm2);
		std::cout << norm << " "<<norm2 << std::endl;*/
		//

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

void RNN::set_input_values(std::vector<double>& input_values)
{
	execute_add_input_values(input_values);
}

double RNN::train_step(std::vector<double> &input_values, std::vector<double> &correct_values, bool classif)
{
	double error = 0;
	for (int i = 0; i < max_iters; ++i)
	{
		set_input_values(input_values);
		forward_prop();
	}
	if (classif)
		softmax(values_through_time[values_through_time.size()-1], ini, ini+out);

	error = backward_prop(correct_values, classif);
	return error;
}

double RNN::train_step_one_inp_set(std::vector<double>& input_values, std::vector<double>& correct_values, bool classif)
{
	double error = 0;
	set_input_values(input_values);
    for (int i = 0; i < max_iters; ++i)
		forward_prop();

	if (classif)
		softmax(values_through_time[values_through_time.size() - 1], ini, ini+out);

	error = backward_prop(correct_values, classif);
	return error;
}

double RNN::time_sens_train_step(std::vector< std::vector<double>>& input_values, std::vector< std::vector<double>>& correct_values, bool classif)
{
	double error = 0;
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
		std::cout<<weights.cooRowInd[i]<<" "<< weights.cooColInd[i]<<" " << weights.cooValues[i] << std::endl;

	printf("---------bias_update---------\n");
	for (int i = 0; i < nodes; ++i)
		std::cout<<i<< " bias: "<<biases_update[i]<<"| ";
	std::cout << std::endl;
	printf("---------weight_update----------\n");
	for (int i = 0; i < weights_update.size(); ++i)
    std::cout << weights.cooRowInd[i] <<" "<< weights.cooColInd[i]<<" " << weights_update[i] << std::endl;
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

	std::vector<std::vector<double> > inp = { {0, 0}, { 0,1 }, { 1,0 }, { 1,1 } };

	std::vector<double> sol = { 0 };

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
	std::vector<double> input(28 * 28);
	std::vector<double> sol(10);
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

	std::vector<double> input(28 * 28);
	std::vector<double> sol(10);
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
	std::vector<double> input(28 * 28);
	for (int it = 0; it < 100; it++) {

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
		std::random_shuffle(indices.begin(), indices.end());
		std::cout << "Iteration " << it << ": " << error_sum / examples << std::endl;

		error_sum = 0;

	}

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
	}

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
	std::vector<std::vector<double>> inp(examples, std::vector<double>(1, 0));

	std::vector<std::vector<double>> sol(examples, std::vector<double>(1, 0));
	
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
	std::vector<std::vector<double>> inp(examples, std::vector<double>(1,0));

	std::vector<std::vector<double>> sol(examples, std::vector<double>(1,0));
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
	std::vector<std::vector<double>> inp(examples, std::vector<double>(1, 0));

	std::vector<std::vector<double>> sol(examples, std::vector<double>(1, 0));
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


		std::vector<std::vector<double>> inp(s.size(), std::vector<double>(alphabet.size(), 0));

		std::vector<std::vector<double>> sol(s.size(), std::vector<double>(alphabet.size(), 0));

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

void RNN_caltech101_sil_problem(RNN& rnn, int exampl, int iters, bool test, std::vector<std::vector<int>> inp, std::vector<int> sol_list, std::vector<std::vector<int>> test_inp, std::vector<int> test_sol_list)
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

	std::vector<double> input(28 * 28);
	std::vector<double> sol(101);
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

		std::cout << "Iteration " << it << ": " << 100 * error_sum / examples << std::endl;

		error_sum = 0;

	}
	if (!test) return;
	
	examples = test_inp.size();

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

		error_sum += rnn.train_step(input, sol, true);

		if (examples > 1000 && image > print_stop)
		{
			std::cout << "%" << 100 * image / examples << " error = " << 100 * error_sum / image << std::endl;
			print_stop += examples / 10;
		}
		rnn.reset();
		//std::cout << "---------------------\n";
	}
	//rnn.print_matrix();
	std::cout << "Test: " << 100 * error_sum / examples << std::endl;

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

		rnn.train_step(input, sol, true);

		for (int i = 0; i < 101; ++i)
			if(rnn.values_through_time[rnn.values_through_time.size() - 1][ini_nodes + i] > 0.1)
				std::cout << "i: " << i << " value: " << rnn.values_through_time[rnn.values_through_time.size() - 1][ini_nodes + i] << std::endl;
		rnn.reset();
	}
	
	
}

