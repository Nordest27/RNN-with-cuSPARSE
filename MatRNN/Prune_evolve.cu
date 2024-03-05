#include "Prune_evolve.cuh"

Prune_evolver::Prune_evolver()
{

}
Prune_evolver::Prune_evolver(int i_nod, int o_nod, int nod, int m_i, int m_d, int d_it, int bat, double lr, double b_lr, std::vector<activation_function>& a_f, bool adamb, double inp_drop, cusparseHandle_t& hand)
{
	handle = &hand;

	max_iters = m_i;
	max_depth = m_d;
	delay_iters = d_it;
	batch = bat;
	learning_rate = lr;
	bias_learning_rate = lr;
	eval = 0;

	ini = i_nod;
	out = o_nod;
	nodes = nod;

	act_f = a_f;
	how_many_active_w = 0;
	biases = std::vector<double>(nodes, 0);
	adam = adamb;
	updated = false;
	inp_dropout = inp_drop;
}

Prune_evolver::Prune_evolver(const Prune_evolver& ng)
{
	handle = ng.handle;

	max_iters = ng.max_iters;
	max_depth = ng.max_depth;
	delay_iters = ng.delay_iters;
	batch = ng.batch;
	learning_rate = ng.learning_rate;
	bias_learning_rate = ng.bias_learning_rate;
	eval = 0;

	ini = ng.ini;
	out = ng.out;
	nodes = ng.nodes;

	act_f = ng.act_f;
	weights = ng.weights;
	how_many_active_w = ng.how_many_active_w;
	biases = ng.biases;
	adam = ng.adam;
	inp_dropout = ng.inp_dropout;

	updated = false;
}

void Prune_evolver::connect_input_output()
{

	for (int i = 0; i < ini; i++)
		for (int o = 0; o < out; o++)
			add_weight(i, ini + o, randomdouble()/10);
}

bool Prune_evolver::add_weight(int i, int j, double value)
{
	if (!weights.insert_elm(i, j, value)) return false;
	how_many_active_w++;
	return true;
}

void Prune_evolver::add_node(activation_function activ_f)
{
	nodes++;

	weights.cols = nodes;
	weights.rows = nodes;
	biases.push_back(0);

	act_f.push_back(activ_f);
}

void Prune_evolver::set_ini_weights()
{
	ini_weights = weights.cooValues;
}

void Prune_evolver::reset_rnn()
{
	rnn.reset();
}

void Prune_evolver::connect_to_all(int i)
{
	for (int n = 0; n < nodes; n++) {
		add_weight(i, n, randomdouble() / 10);

		if(n > ini)
			add_weight(n, i, randomdouble() / 10);
	}
}

void Prune_evolver::forward_prop(std::vector<double>& input_values) {

	for (int i = 0; i < max_iters; ++i)
	{
		rnn.set_input_values(input_values);
		rnn.forward_prop();
	}
}

double Prune_evolver::evaluate(std::vector<double>& correct_values, bool classif)
{
	double error = rnn.backward_prop(correct_values, classif);
	eval += error;
	return error;
}

void Prune_evolver::prune_weights(int how_many)
{
	if (how_many == 0) return;
	std::vector<double> weights_aux = weights.cooValues;

	for (int i = 0; i < weights_aux.size(); ++i)
		weights_aux[i] = abs(weights_aux[i]);

	std::sort(weights_aux.begin(), weights_aux.end());
	
	double threshold = weights_aux[how_many-1];
	
	prune_weights(threshold);
}

void Prune_evolver::prune_weights(double threshold)
{
	int last = 0;
	for (int i = 0; i < weights.nnz; ++i, ++last)
	{
		while (i < weights.nnz && abs(weights.cooValues[i]) <= threshold)
			++i;
		if (i >= weights.nnz) break;

		weights.cooColInd[last] = weights.cooColInd[i];
		weights.cooRowInd[last] = weights.cooRowInd[i];
		weights.cooValues[last] = weights.cooValues[i];

		//ini_weights[last] = ini_weights[i];
	}

	weights.cooColInd.resize(last);
	weights.cooRowInd.resize(last);
	weights.cooValues.resize(last);
	weights.nnz = weights.cooValues.size();
	//ini_weights.resize(last);
}

double Prune_evolver::prune_weights(int how_many, std::vector<double> &scores)
{
	if (how_many == 0) return 0;

	double suma = sum(scores);

	std::vector<double> scores_aux = scores;

	std::sort(scores_aux.begin(), scores_aux.end());
	//for (auto val : scores_aux)
	//	std::cout << val << std::endl;
	double threshold = scores_aux[how_many - 1];
	
	//if (threshold > 10)
	//	return;
	//std::cout<<how_many<<" " << threshold << " " << scores_aux[how_many-1] << std::endl;

	if (threshold == 0) {
		std::cout <<how_many << " Zero found :( " << suma << " " << scores_aux[how_many - 1] << std::endl;
		//for (int i = 0; i < scores_aux.size(); ++i)
		//	std::cout << scores_aux[i] << " ";
		return suma;
	}
	prune_weights(threshold, scores);
	return suma;
}

void Prune_evolver::prune_weights(double threshold, std::vector<double>& scores)
{
	
	int last = 0;
	for (int i = 0; i < weights.nnz; ++i, ++last)
	{
		while (i < weights.nnz && scores[i] <= threshold)
			++i;

		if (i >= weights.nnz) break;

		weights.cooColInd[last] = weights.cooColInd[i];
		weights.cooRowInd[last] = weights.cooRowInd[i];
		weights.cooValues[last] = weights.cooValues[i];

		//ini_weights[last] = ini_weights[i];
	}

	weights.cooColInd.resize(last);
	weights.cooRowInd.resize(last);
	weights.cooValues.resize(last);
	weights.nnz = weights.cooValues.size();
	//ini_weights.resize(last);

//	std::cout << scores.size()-last<< std::endl;
}

void Prune_evolver::update_weights_and_biases()
{
	rnn.update_weights_and_biases();
	//rnn.read_weights_and_biases_from_device(weights.cooValues, biases);
}

void Prune_evolver::add_fully_connected_nodes(int n)
{

	activation_function act_f[] = { SIGMOID, RELU };
	for (int i = 0; i < n; ++i)
	{
		add_node(act_f[rand() % 2]);
		connect_to_all(nodes);
	}
}

void Prune_evolver::fully_connect_n_nodes(int n)
{

	for (int i = 0; i < n; ++i)
	{
		int nodi = rand() % nodes;
		connect_to_all(nodi);
	}
}

void Prune_evolver::connect_everything( double sparsity, bool to_inp, bool from_out )
{
	if(to_inp && from_out)
		weights.connect_layers(0, nodes, 0, nodes, sparsity, nodes);
	else if(!to_inp && from_out)
		weights.connect_layers(0, nodes, ini, nodes, sparsity, nodes);
	else if (to_inp && !from_out) {
		weights.connect_layers(        0,   ini, 0, nodes, sparsity, nodes);
		weights.connect_layers(ini + out, nodes, 0, nodes, sparsity, nodes);
	}
	else {
		weights.connect_layers(        0,   ini, ini, nodes, sparsity, nodes);
		weights.connect_layers(ini + out, nodes, ini, nodes, sparsity, nodes);
	}

}

void Prune_evolver::add_n_random_connections(int n)
{

	//std::default_random_engine generator;
	//std::normal_distribution<double> distribution(0, nodes / 4);

	for (int i = 0; i < n; ++i)
		while (!add_weight(rand() % nodes, rand() % nodes, randomdouble() / sqrt(nodes)));
		
	
	
}

void Prune_evolver::add_n_random_connections_weight_strat(int n)
{
	std::vector< std::pair<double, int> > weights_aux(weights.nnz);

	for (int i = 0; i < weights.nnz; ++i)
		weights_aux[i] = std::pair<double, int>(abs(weights.cooValues[i]), i);

	std::sort(weights_aux.begin(), weights_aux.end());

	int ws = weights.nnz;
	for (int i = 0; i < n/2; ++i) {
		int weight_i = weights_aux[ws-1-i].second;
		int nodi = weights.cooColInd[weight_i];
		int nodj = weights.cooRowInd[weight_i];
		int nodu = rand() % nodes;

		//std::cout << nodi << " " << nodj << " " << nodu << " " << nodes << std::endl;
		while (!add_weight(nodi, nodu, randomdouble() / 10))
			nodu = rand() % nodes;

		//add_weight(nodi, nodu, randomdouble() / 10);
		add_weight(nodu, nodj, randomdouble() / 10);

	}

}

void Prune_evolver::connect_like_mlp(std::vector<int>& hid_layers, bool b)
{
	std::vector<double> offset(hid_layers.size(), 0);

	for (int i = 1; i < hid_layers.size(); ++i) {
		offset[i] += hid_layers[i - 1] + offset[i - 1];
		std::cout << offset[i] << " ";
	}
	std::cout << std::endl;
	if (b){

		std::cout<<0 <<" " << ini << " " << ini + out << " " << ini + out + hid_layers[0] << std::endl;
		weights.connect_layers(0, ini, ini + out, ini + out + hid_layers[0], 0, nodes);

		for (int i = 0; i < hid_layers.size() - 1; ++i)
			weights.connect_layers(
				ini + out + offset[i],
				ini + out + offset[i+1],
				ini + out + offset[i+1],
				ini + out + offset[i+1]+hid_layers[i+1], 0, nodes);
		
		weights.connect_layers(offset[hid_layers.size() - 1] + ini + out,
							   offset[hid_layers.size() - 1] + ini + out + hid_layers[hid_layers.size() - 1], 
							   ini, ini + out, 0, nodes);
		return;
	}
	
	for (int j = 0; j < ini; ++j)
		for (int u = 0; u < hid_layers[0]; ++u)
			add_weight(j, ini+out+u, randomdouble() / 10);

	for (int i = 0; i < hid_layers.size() - 1; ++i)
		for (int j = 0; j < hid_layers[i]; ++j)
			for (int u = 0; u < hid_layers[i + 1]; ++u)
				add_weight(ini + out + j+offset[i], ini + out + u+offset[i+1], randomdouble() / 10);
	
	for (int j = 0; j < hid_layers[hid_layers.size()-1]; ++j)
		for (int u = 0; u < out; ++u)
			add_weight(offset[hid_layers.size() - 1]+ini + out + j, ini + u, randomdouble() / 10);
 
	
	//hid nodes = 
}

void Prune_evolver::read_weights_and_biases_from_device()
{
	rnn.read_weights_and_biases_from_device(weights.cooValues, biases);
}

void Prune_evolver::update_rnn()
{
	if(updated)
		rnn.empty_device();

	rnn = RNN(ini, out, nodes, learning_rate, max_depth, batch, act_f, weights, *handle);
	rnn.bias_learning_rate = bias_learning_rate;
	rnn.delay_iters = delay_iters;
	rnn.max_iters = max_iters;
	rnn.initialize_device_public();
	rnn.adam = adam;
	rnn.inp_dropout = inp_dropout;
	updated = true;
}

void Prune_evolver::reinitialize_weights_and_reset_biases()
{
	weights.cooValues = ini_weights;

	for (int i = 0; i < nodes; ++i)
		biases[i] = 0;
}

void Prune_evolver::divide_weights_and_biases(double d)
{
	for (int i = 0; i < weights.nnz; ++i)
		weights.cooValues[i] /= d;

	for (int i = 0; i < nodes; ++i)
		biases[i] /= d;
}

double Prune_evolver::synflow_prune(int how_many, bool classif, std::vector<double> &ones)
{
	auto scores = rnn.synflow_cycle(classif, ones, -1);

	return prune_weights(how_many, scores);
}


double Prune_evolver::synflow_prune(int how_many, bool classif, std::vector<std::pair<std::vector<double>, double>>& dataset, int samples)
{
	auto scores = rnn.synflow_cycle(classif, dataset, samples);

	return prune_weights(how_many, scores);
	/*std::vector<double> scores(weights.nnz);
	int ant_res = -1;
	for (int i = 0; i < out; ++i) {

		int example;
		std::vector<double> inp(ini, 0);

		for (int j = 0; j < samples; ++j)
		{
			example = rand() % dataset.size();
			while (int(dataset[example].second) != i)
				example = rand() % dataset.size();

			auto aux_inp = dataset[example].first;

			for (int j = 0; j < ini; ++j)
				inp[j] += aux_inp[j]/samples;
		}

		double suma = 0;
		for (int j = 0; j < inp.size(); j++)
			suma += inp[j];
		suma /= inp.size();

		for (int j = 0; j < inp.size(); j++)
			inp[j] /= suma;
		std::cout << suma << std::endl;
		for (int j = 0; j < 28; j++) {
			for (int u = 0; u < 28; u++) {
				if (inp[j * 28 + u] > 0.75) std::cout << "X ";
				else if (inp[j * 28 + u] > 0.5) std::cout << "x ";
				else if (inp[j * 28 + u] > 0.25) std::cout << "+ ";
				else if (inp[j * 28 + u] > 0) std::cout << "- ";
				else std::cout << "_ ";
			}
			std::cout << std::endl;
		}
		std::cout << std::endl;
		//std::cout << dataset[example].second << std::endl;
		auto aux_score = rnn.synflow_cycle(classif, inp, i);
		for (int j = 0; j < weights.nnz; ++j)
			scores[j] += aux_score[j];

	}
	for (int j = 0; j < weights.nnz; ++j)
		scores[j] /= samples;

	return prune_weights(how_many, scores);*/
}

void Prune_evolver::synflow_loop(int iters, double new_conn_thresh, int max_conn, bool to_inp, bool from_out, std::vector<double> &ones, bool classif)
{
	double suma = 0;
	for (int i = 0; i < iters; i++) {

		connect_everything(new_conn_thresh, to_inp, from_out);
	    //std::vector<int> hid_layers = { 1000, 500, 300 };
       // connect_like_mlp(hid_layers, true);
		//connect_input_output();
		update_rnn();

		std::cout << "it " << i << " Number of Weights: " << weights.nnz << std::endl;
		//pe.synflow_prune(pe.rnn.weights.nnz-max_connections, true);
		double sparsity = double(max_conn) / weights.nnz;
		int it = 100;
		int total_params = weights.nnz;
		if (weights.nnz > max_conn)
			for (int k = 1; k <= it; k++) {
				int connections_to_prune_k = weights.nnz - total_params * pow(sparsity, double(k) / it);
				suma = synflow_prune(connections_to_prune_k, classif, ones);
				update_rnn();
			}

		int how_many = 0, how_many_inp_out = 0, how_many_inp_inp = 0, how_many_inp = 0, how_many_out = 0, how_many_out_out = 0;
		for (int i = 0; i < weights.nnz; ++i)
		{
			if (weights.cooRowInd[i] >= ini && weights.cooRowInd[i] < ini + out) {
				how_many++;

				if (weights.cooColInd[i] < ini)
					how_many_inp_out++;
				else if (weights.cooColInd[i] >= ini && weights.cooColInd[i] < ini + out)
				{
					how_many_out++;
					how_many_out_out++;
				}

				//std::cout << pe.weights.cooColInd[i] << " " << pe.weights.cooRowInd[i] << std::endl;
			}
			else if (weights.cooColInd[i] >= ini && weights.cooColInd[i] < ini + out)
			{
				how_many_out++;
			}
			if ( weights.cooColInd[i] < ini) {
				 if ( weights.cooRowInd[i] < ini )
					how_many_inp_inp++;

				 how_many_inp++;
			}
		}

		if (weights.nnz != weights.cooColInd.size() || weights.nnz != weights.cooRowInd.size())
			std::cout << "BIG ERROR!" << std::endl;

		std::cout << "How many begin in output: " << how_many_out << std::endl;
		std::cout << "How many end in output: " << how_many << std::endl;

		std::cout << "How many begin in output end in output: " << how_many_out_out << std::endl;
		std::cout << "How many begin in input end in output: " << how_many_inp_out << std::endl;

		std::cout << "How many begin in input: " << how_many_inp << std::endl;
		std::cout << "How many begin in input end in input: " << how_many_inp_inp << std::endl;

		std::cout << "Sum of last scores: " << suma << std::endl;
		std::cout << "Number of Weights: " << rnn.weights.nnz << std::endl;
	}
}

void Prune_evolver::synflow_loop(int iters, double new_conn_thresh, int max_conn, bool to_inp, bool from_out, bool classif, std::vector<std::pair<std::vector<double>, double>>& dataset, int samples)
{
	double suma = 0;
	for (int i = 0; i < iters; i++) {

		connect_everything(new_conn_thresh, to_inp, from_out);
		//connect_input_output();
		update_rnn();

		std::cout << "it " << i << " Number of Weights: " << weights.nnz << std::endl;
		//pe.synflow_prune(pe.rnn.weights.nnz-max_connections, true);
		double sparsity = double(max_conn) / weights.nnz;
		int it = 100;
		int total_params = weights.nnz;
		if (weights.nnz > max_conn)
			for (int k = 1; k <= it; k++) {
				int connections_to_prune_k = weights.nnz - total_params * pow(sparsity, double(k) / it);
				suma = synflow_prune(connections_to_prune_k, classif, dataset, samples);
				update_rnn();
			}

		int how_many = 0, how_many_inp_out = 0, how_many_inp_inp = 0, how_many_inp = 0, how_many_out = 0, how_many_out_out = 0;
		for (int i = 0; i < weights.nnz; ++i)
		{
			if (weights.cooRowInd[i] >= ini && weights.cooRowInd[i] < ini + out) {
				how_many++;

				if (weights.cooColInd[i] < ini)
					how_many_inp_out++;
				else if (weights.cooColInd[i] >= ini && weights.cooColInd[i] < ini + out)
				{
					how_many_out++;
					how_many_out_out++;
				}

				//std::cout << pe.weights.cooColInd[i] << " " << pe.weights.cooRowInd[i] << std::endl;
			}
			else if (weights.cooColInd[i] >= ini && weights.cooColInd[i] < ini + out)
			{
				how_many_out++;
			}
			if (weights.cooColInd[i] < ini) {
				if (weights.cooRowInd[i] < ini)
					how_many_inp_inp++;

				how_many_inp++;
			}
		}

		std::cout << "How many begin in output: " << how_many_out << std::endl;
		std::cout << "How many end in output: " << how_many << std::endl;

		std::cout << "How many begin in output end in output: " << how_many_out_out << std::endl;
		std::cout << "How many begin in input end in output: " << how_many_inp_out << std::endl;

		std::cout << "How many begin in input: " << how_many_inp << std::endl;
		std::cout << "How many begin in input end in input: " << how_many_inp_inp << std::endl;

		std::cout << "Sum of last scores: " << suma << std::endl;
		std::cout << "Number of Weights: " << rnn.weights.nnz << std::endl;
	}
}

void prune_test()
{
	cusparseHandle_t handle;
	cusparseCreate(&handle);
	int ini_nodes = 1;
	int out_nodes = 1;
	int extra_nodes = 1;
	int nodes = ini_nodes + out_nodes + extra_nodes;
	int batch = 10;
	int depth = 5;
	int delay_iteration = 0;
	double learning_rate = 0.01;
	double error_sum = 0;
	bool adam = true;
	double inp_dropout = 0;

	std::vector<activation_function> act_f(nodes, RELU);

	for (int i = ini_nodes; i < ini_nodes + out_nodes; ++i)
		act_f[i] = LINEAL;


	for (int i = ini_nodes + out_nodes; i < nodes; ++i)
		act_f[i] = LINEAL;

	Prune_evolver pe(ini_nodes, out_nodes, nodes, depth, depth, delay_iteration, batch, learning_rate, learning_rate, act_f, adam, inp_dropout, handle);
	//pe.connect_input_output();
	//std::vector<int> hid_layers = { 250, 150, 50 };
	//pe.connect_like_mlp(hid_layers);
	pe.update_rnn();

	std::cout << "Number of Weights: " << pe.rnn.weights.nnz << std::endl;
	int max_connections = 200;

	std::vector<double> ones(pe.ini, 1);

	pe.synflow_loop(10, 0, max_connections, false, false, ones, true);
	
	pe.update_rnn();
	std::cout << "Number of Weights: " << pe.rnn.weights.nnz << std::endl;

}

void mlp_test() 
{
	cusparseHandle_t handle;
	cusparseCreate(&handle);
	int ini_nodes = 28*28;
	int out_nodes = 10;
	int extra_nodes = 400 +300 +200 +100 +10;
	int nodes = ini_nodes + out_nodes + extra_nodes;
	int batch = 10;
	int depth = 7;
	int delay_iteration = 0;
	double learning_rate = 0.01;
	double error_sum = 0;
	bool adam = true;

	double inp_dropout = 0;

	std::vector<activation_function> act_f(nodes, RELU);

	for (int i = ini_nodes; i < ini_nodes + out_nodes; ++i)
		act_f[i] = LINEAL;


	for (int i = ini_nodes + out_nodes; i < nodes; ++i)
		act_f[i] = RELU;

	Prune_evolver pe(ini_nodes, out_nodes, nodes, depth, depth, delay_iteration, batch, learning_rate, learning_rate, act_f, inp_dropout, adam, handle);

	std::vector<int> hid_layers = { 400, 300, 200, 100, 10 };
	pe.connect_like_mlp( hid_layers, true);


	std::cout << "Number of Weights: " << pe.weights.nnz << " " <<pe.weights.rows<< " " << pe.weights.cols << std::endl;

	

	/*for (int i = 0; i < pe.weights.nnz; ++i)
	{
		if (pe.weights.cooColInd[i] != pe2.weights.cooColInd[i] || pe.weights.cooRowInd[i] != pe2.weights.cooRowInd[i])
			std::cout << "DIF! "<< pe.weights.cooColInd[i] <<" != "<< pe2.weights.cooColInd[i] <<" "
			<< pe.weights.cooRowInd[i] << " != " << pe2.weights.cooRowInd[i] << std::endl;
	}
	*/
	pe.update_rnn();
	
	std::cout << "Number of Weights: " << pe.rnn.weights.nnz << std::endl;
	RNN_mnist_problem(pe.rnn, -1, 5, true);

}

void prune_mnist_problem()
{
	cusparseHandle_t handle;
	cusparseCreate(&handle);
	int ini_nodes = 28 * 28;
	int out_nodes = 10;
	int extra_nodes = 500;
	int nodes = ini_nodes + out_nodes + extra_nodes;
	int batch = 1;
	int depth = 5;
	int delay_iteration = 0;
	double learning_rate = 0.0001;
	double error_sum = 0;
	bool adam = true;

	double inp_dropout = 0;
	std::vector<activation_function> act_f(nodes, RELU);

	for (int i = ini_nodes; i < ini_nodes + out_nodes; ++i)
		act_f[i] = LINEAL;


	for (int i = ini_nodes + out_nodes; i < nodes; ++i)
		act_f[i] = RELU;

	Prune_evolver pe(ini_nodes, out_nodes, nodes, depth, depth, delay_iteration, batch, learning_rate, learning_rate, act_f, adam, inp_dropout, handle);
	//pe.connect_input_output();
	//std::vector<int> hid_layers = { 13 };
	//pe.connect_like_mlp(hid_layers, true);
	//pe.update_rnn();

	std::cout << "Number of Weights: " << pe.rnn.weights.nnz << std::endl;
	int max_connections = 10000;

	std::vector<double> ones(pe.ini, 0);

	auto inp = mnist::read_training_images();
	auto sol_list = mnist::read_training_labels();

	auto test_inp = mnist::read_test_images();
	inp.insert(inp.end(), test_inp.begin(), test_inp.end());

	auto test_sol_list = mnist::read_test_labels();
	sol_list.insert(sol_list.end(), test_sol_list.begin(), test_sol_list.end());

	std::vector < std::pair<std::vector<double>, double> > dataset;
	std::vector < std::pair<std::vector<double>, double> > test_data;
	std::vector < std::pair<std::vector<double>, double> > val_data;

	pe.rnn.partition(inp, sol_list, 0.71428571428, 0.5, dataset, test_data, val_data, 255, true);
	//pe.rnn.partition(inp, sol_list, 0.8, 0.75, dataset, test_data, val_data, 255);
	std::cout << "Train size: " << dataset.size() << std::endl;
	std::cout << "Valid size: " << val_data.size() << std::endl;
	std::cout << "Test  size: " << test_data.size() << std::endl;
	double suma = 0;
	for (int i = 0; i < dataset.size(); ++i)
		for (int j = 0; j < dataset[i].first.size(); j++) 
			ones[j] += dataset[i].first[j];
		
	for (int j = 0; j < ones.size(); j++)
		ones[j] /= dataset.size();

	suma = 0;
	for (int j = 0; j < ones.size(); j++)
		suma += ones[j];
	suma /= ones.size();

	for (int j = 0; j < ones.size(); j++)
		ones[j] /= suma;

	suma = 0;
	for (int j = 0; j < ones.size(); j++)
		suma += ones[j];
	suma /= ones.size();

	//for (int j = 0; j < ones.size(); j++)
		//ones[j] = 1;

	std::cout << suma << std::endl;
	for (int j = 0; j < 28; j++) {
		for (int u = 0; u < 28; u++) {
			if (ones[j * 28 + u] > 0.75) std::cout << "X ";
			else if (ones[j * 28 + u] > 0.5) std::cout << "x ";
			else if (ones[j * 28 + u] > 0.25) std::cout << "+ ";
			else if (ones[j * 28 + u] > 0) std::cout << "- ";
			else std::cout << "_ ";
		}
		std::cout << std::endl;
	}
	std::cout << std::endl;
	//pe.connect_everything(0, true, true);
	//pe.prune_weights(pe.weights.nnz - max_connections);
	//std::cout << "Number of Weights: " << pe.weights.nnz << std::endl;
	//ones = std::vector<double>(pe.ini, 1);
	//pe.synflow_loop(20, 0, max_connections, true, true, ones, true);
    pe.synflow_loop(20, 0, max_connections, true, true, true, dataset, 100);
	//pe.synflow_loop(50, 0, max_connections, false, false, true, dataset, 100);
	//pe.add_n_random_connections(max_connections);

	pe.inp_dropout = 0;
	pe.update_rnn();
	//pe.rnn.print_matrix();
	//RNN_mnist_problem(pe.rnn, -1, 5, true);
	generic_classif_RNN_problem(pe.rnn, -1, 100, true, dataset, test_data, val_data);
	std::cout << "Number of Weights: " << pe.rnn.weights.nnz << std::endl;
}

std::vector<std::vector<int>> read_caltech_training_images()
{
	std::ifstream stream("train_caltech101.txt");
	std::vector<std::vector<int>> result;
	int n;
	stream >> n;
	std::cout << n << std::endl;
	for (int i = 0; i < n; i++) {
		std::vector<int> aux(28 * 28, 0);
		for (int j = 0; j < 28 * 28; ++j)
			stream >> aux[j];
		result.push_back(aux);
	}
	for (int j = 0; j < 28 * 28; ++j) {
		std::cout << result[0][j] << " ";
		if ((j + 1) % 28 == 0)
			std::cout << std::endl;
	}
	return result;
}

std::vector<int> read_caltech_training_labels()
{
	std::ifstream stream("train_labels_caltech101.txt");

	int n;
	stream >> n;
	std::vector<int> result(n, 0);

	for (int i = 0; i < n; i++)
		stream >> result[i];

	return result;

}

std::vector<std::vector<int>> read_caltech_test_images()
{
	std::ifstream stream("test_caltech101.txt");
	std::vector<std::vector<int>> result;
	int n;
	stream >> n;
	std::cout << n << std::endl;
	for (int i = 0; i < n; i++) {
		std::vector<int> aux(28 * 28, 0);
		for (int j = 0; j < 28 * 28; ++j)
			stream >> aux[j];
		result.push_back(aux);
	}
	for (int j = 0; j < 28 * 28; ++j) {
		std::cout << result[0][j] << " ";
		if ((j + 1) % 28 == 0)
			std::cout << std::endl;
	}
	return result;
}

std::vector<int> read_caltech_test_labels()
{
	std::ifstream stream("test_labels_caltech101.txt");

	int n;
	stream >> n;
	std::vector<int> result(n, 0);

	for (int i = 0; i < n; i++)
		stream >> result[i];
	std::cout << result[0] << std::endl;

	return result;

}

void prune_caltech_problem()
{
	auto inp = read_caltech_training_images();

	auto sol_list = read_caltech_training_labels();

	auto test_inp = read_caltech_test_images();
	inp.insert(inp.end(), test_inp.begin(), test_inp.end());

	auto test_sol_list = read_caltech_test_labels();
	sol_list.insert(sol_list.end(), test_sol_list.begin(), test_sol_list.end());
	/*
	test_inp = std::vector< std::vector<int> >(inp.begin()+inp.size()*0.6, inp.end());
	test_sol_list = std::vector< int >(sol_list.begin()+sol_list.size() * 0.6, sol_list.end() );

	inp.resize(inp.size() * 0.6);
	sol_list.resize(sol_list.size() * 0.6);

	auto val_inp = std::vector< std::vector<int> >(test_inp.begin() + test_inp.size() * 0.5, test_inp.end());
	auto val_sol_list = std::vector< int >(test_sol_list.begin() + test_sol_list.size() * 0.5, test_sol_list.end());

	test_inp.resize(test_inp.size() * 0.5);
	test_sol_list.resize(test_sol_list.size() * 0.5);

	std::cout << inp.size() <<" " << test_inp.size() << " " << val_inp.size() <<std::endl;
	
	std::cout << sol_list.size() <<" " << test_sol_list.size() << " " << val_sol_list.size() << std::endl;*/

	cusparseHandle_t handle;
	cusparseCreate(&handle);
	int ini_nodes = 28 * 28;
	int out_nodes = 101;
	int extra_nodes = 500;
	int nodes = ini_nodes + out_nodes + extra_nodes;
	int batch = 10;
	int depth = 20;
	int delay_iteration = 0;
	double learning_rate = 0.01;
	double error_sum = 0;
	bool adam = true;

	double inp_dropout = 0;

	std::vector<activation_function> act_f(nodes, RELU);

	Prune_evolver pe(ini_nodes, out_nodes, nodes, depth, depth, delay_iteration, batch, learning_rate, learning_rate, act_f, adam, inp_dropout, handle);

	for (int i = ini_nodes; i < ini_nodes + out_nodes; ++i)
		pe.act_f[i] = LINEAL;


	for (int i = ini_nodes + out_nodes; i < nodes; ++i)
		pe.act_f[i] = RELU;

	std::vector < std::pair<std::vector<double>, double> > dataset;
	std::vector < std::pair<std::vector<double>, double> > test_data;
	std::vector < std::pair<std::vector<double>, double> > val_data;

	for (int i = 0; i < sol_list.size(); ++i)
		sol_list[i]--;

	//pe.rnn.partition(inp, sol_list, 0.71428571428, 0.5, dataset, test_data, val_data, 255);
	pe.rnn.partition(inp, sol_list, 0.47284050282, 0.50470356595, dataset, test_data, val_data, 1, false);
	std::cout << "Train size: " << dataset.size() << std::endl;
	std::cout << "Valid size: " << val_data.size() << std::endl;
	std::cout << "Test  size: " << test_data.size() << std::endl;

	std::vector<double> ones(pe.ini, 0);
	double suma = 0;
	for (int i = 0; i < dataset.size(); ++i)
		for (int j = 0; j < dataset[i].first.size(); j++)
			ones[j] += dataset[i].first[j];

	for (int j = 0; j < ones.size(); j++)
		ones[j] /= dataset.size();

	suma = 0;
	for (int j = 0; j < ones.size(); j++)
		suma += ones[j];
	suma /= ones.size();

	double max = 0;
	for (int j = 0; j < ones.size(); j++) {
		ones[j] /= suma;
		if (ones[j] > max)
			max = ones[j];
	}

	suma = 0;
	for (int j = 0; j < ones.size(); j++)
		suma += ones[j];
	suma /= ones.size();

	//for (int j = 0; j < ones.size(); j++)
		//ones[j] = 1;

	std::cout <<"max: "<<max<<" suma: " << suma << std::endl;
	for (int j = 0; j < 28; j++) {
		for (int u = 0; u < 28; u++) {
			if (ones[j * 28 + u] > max*0.75) std::cout << "X ";
			else if (ones[j * 28 + u] > max*0.5) std::cout << "x ";
			else if (ones[j * 28 + u] > max*0.25) std::cout << "+ ";
			else if (ones[j * 28 + u] > max*0) std::cout << "- ";
			else std::cout << "_ ";
		}
		std::cout << std::endl;
	}
	std::cout << std::endl;
	int max_connections = 100000;
	pe.connect_everything(0, true, true);
	std::vector<double> randCrit(pe.weights.nnz);
	for (int i = 0; i < pe.weights.nnz; ++i)
		randCrit[i] = randomdouble();

	pe.prune_weights(pe.weights.nnz - max_connections, randCrit);
	pe.update_rnn();
    //pe.connect_input_output();
	//std::vector<int> hid_layers = { 1000, 200, 100 };
	//pe.connect_like_mlp(hid_layers, true);
	//pe.update_rnn();
	//pe.add_n_random_connections(max_connections);
	//ones = std::vector<double>(pe.ini, 1);
	//pe.synflow_loop(20, 0, max_connections, true, true, ones, true);
    //pe.synflow_loop(20, 0, max_connections, true, true, true, dataset, 1);
	//pe.add_n_random_connections(max_connections);

	std::cout << "Number of Weights: " << pe.rnn.weights.nnz << std::endl;
	pe.inp_dropout = 0;
	pe.update_rnn();
	//pe.rnn.print_matrix();
	generic_classif_RNN_problem(pe.rnn, -1, 100, true, dataset, test_data, val_data);
	std::cout << "Number of Weights: " << pe.rnn.weights.nnz << std::endl;

}



void prune_chess_problem(std::vector<std::vector<double>>& inp, std::vector<double>& sol_list)
{
	cusparseHandle_t handle;
	cusparseCreate(&handle);
	int ini_nodes = inp[0].size();
	int out_nodes = 1;
	int extra_nodes = 2000;
	int nodes = ini_nodes + out_nodes + extra_nodes;
	int batch = 10;
	int depth = 10;
	int delay_iteration = 0;
	double learning_rate = 0.001;
	double error_sum = 0;

	double inp_dropout = 0;
	bool adam = true;
	int max_connections = 1000000;

	std::vector<activation_function> act_f(nodes, LINEAL);

	for (int i = ini_nodes; i < ini_nodes + out_nodes; ++i)
		act_f[i] = LINEAL;


	for (int i = ini_nodes + out_nodes; i < nodes; ++i)
		act_f[i] = RELU;


	Prune_evolver pe(ini_nodes, out_nodes, nodes, depth, depth, delay_iteration, batch, learning_rate, learning_rate, act_f, adam, inp_dropout, handle);
	std::vector<int> hid_layers = { 1000, 300, 300, 300, 300, 100, 50, 10 };
	pe.connect_like_mlp(hid_layers, false);
	//pe.connect_input_output();
	pe.update_rnn();

	std::vector < std::pair<std::vector<double>, double> > dataset;
	std::vector < std::pair<std::vector<double>, double> > test_data;
	std::vector < std::pair<std::vector<double>, double> > val_data;

	pe.rnn.partition(inp, sol_list, 0.9, 0.5, dataset, test_data, val_data, 1, true);

	auto ones = std::vector<double>(pe.ini, 0);
	double suma = 0;

	for (int i = 0; i < dataset.size(); ++i)
		for (int j = 0; j < dataset[i].first.size(); j++)
			ones[j] += dataset[i].first[j];

	for (int j = 0; j < ones.size(); j++)
		ones[j] /= dataset.size();

	suma = 0;
	for (int j = 0; j < ones.size(); j++)
		suma += ones[j];
	suma /= ones.size();

	for (int j = 0; j < ones.size(); j++)
		ones[j] /= suma;

	suma = 0;
	for (int j = 0; j < ones.size(); j++)
		suma += ones[j];
	suma /= ones.size();

	//pe.connect_everything(0, true, true);
	//pe.prune_weights(pe.weights.nnz - max_connections);
	//std::cout << "Number of Weights: " << pe.weights.nnz << std::endl;
	//ones = std::vector<double>(pe.ini, 1);
	//pe.synflow_loop(20, 0, max_connections, true, true, ones, true);
	

	pe.update_rnn();
	std::cout << "Number of Weights: " << pe.rnn.weights.nnz << std::endl;
	pe.inp_dropout = 0;
	pe.update_rnn();
	//pe.rnn.print_matrix();
	generic_regress_RNN_problem(pe.rnn, -1, 100, true, dataset, test_data, val_data);
	//RNN_chess_problem(pe.rnn, -1, 5, inp, sol_list);
	std::cout << "Number of Weights: " << pe.rnn.weights.nnz << std::endl;

}


