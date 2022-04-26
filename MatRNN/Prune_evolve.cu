#include "Prune_evolve.cuh"

Prune_evolver::Prune_evolver(int i_nod, int o_nod, int nod, int m_i, int m_d, int d_it, int bat, double lr, double b_lr, std::vector<activation_function>& a_f, bool adamb, cusparseHandle_t& hand)
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

void Prune_evolver::connect_to_all(int i)
{
	for (int n = 0; n < nodes; n++)
		add_weight(i, n, randomdouble() / 10);
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
	}

	weights.cooColInd.resize(last);
	weights.cooRowInd.resize(last);
	weights.cooValues.resize(last);
	weights.nnz = weights.cooValues.size();
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

void Prune_evolver::add_n_random_connections(int n)
{

	std::default_random_engine generator;
	std::normal_distribution<double> distribution(0, nodes / 4);

	for (int i = 0; i < n; ++i) {
		int nodi = rand() % nodes;
		//std::cout << (int)distribution(generator) << std::endl;

		while (!add_weight(nodi, /*(nodi + (int)distribution(generator))*/rand() % nodes, randomdouble() / 10));
		
	}
	
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

void Prune_evolver::connect_like_mlp(std::vector<int>& hid_layers)
{
	for (int j = 0; j < ini; ++j)
		for (int u = 0; u < hid_layers[0]; ++u)
			add_weight(j, ini+out+u, randomdouble() / 10);

	for (int i = 0; i < hid_layers.size() - 1; ++i)
		for (int j = 0; j < hid_layers[i]; ++j)
			for (int u = 0; u < hid_layers[i + 1]; ++u)
				add_weight(ini + out + j, ini + out + u, randomdouble() / 10);

	for (int j = 0; j < hid_layers[hid_layers.size()-1]; ++j)
		for (int u = 0; u < out; ++u)
			add_weight(ini + out + j, ini + u, randomdouble() / 10);

	
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
	updated = true;
}

void Prune_evolver::randomize_weights_and_reset_biases()
{
	for (int i = 0; i < weights.nnz; ++i)
		weights.cooValues[i] = randomdouble()/10;

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

void prune_test()
{
	cusparseHandle_t handle;
	cusparseCreate(&handle);
	int ini_nodes = 28 * 28;
	int out_nodes = 10;
	int nodes = ini_nodes + out_nodes + 100;
	int batch = 10;
	int depth = 5;
	int delay_iteration = 0;
	double learning_rate = 0.01;
	double error_sum = 0;
	bool adam = true;

	std::vector<activation_function> act_f(nodes, RELU);

	for (int i = ini_nodes; i < ini_nodes + out_nodes; ++i)
		act_f[i] = LINEAL;

	for (int i = ini_nodes + out_nodes; i < nodes; ++i)
		act_f[i] = SIGMOID;


	Prune_evolver pe(ini_nodes, out_nodes, nodes, depth, depth, delay_iteration, batch, learning_rate, learning_rate, act_f, adam, handle);
	pe.connect_input_output();

	for (int i = 0; i < ini_nodes; ++i) {

		for (int j = ini_nodes; j < ini_nodes + out_nodes; ++j)
			pe.add_weight(i, j, randomdouble()/10);

		for (int j = ini_nodes + out_nodes; j < nodes; ++j)
			pe.add_weight(i, j, randomdouble()/10);
	}

	for (int i = ini_nodes + out_nodes; i < nodes; ++i) {
		for (int j = ini_nodes; j < ini_nodes + out_nodes; ++j)
			pe.add_weight(i, j, randomdouble()/10);
		if (i < nodes - 50)
			for (int j = ini_nodes + out_nodes + 50; j < nodes; ++j)
				pe.add_weight(i, j, randomdouble()/10);
		else
			for (int j = ini_nodes; j < ini_nodes + out_nodes; ++j)
				pe.add_weight(i, j, randomdouble()/10);
	}

	for (int i = 0; i < 10000; ++i)
	{
		pe.add_weight(rand() % nodes, rand() % nodes, randomdouble()/10);
	}
	
	pe.update_rnn();

	std::cout << "Number of Weights: " << pe.rnn.weights.nnz << std::endl;

	RNN_mnist_problem(pe.rnn, -1, 1, true);
	pe.read_weights_and_biases_from_device();
	pe.prune_weights(int(pe.rnn.weights.nnz *0.95));

	pe.randomize_weights_and_reset_biases();
	pe.update_rnn();

	RNN_mnist_problem(pe.rnn, -1, 2, true);
	std::cout << "Number of Weights: " << pe.rnn.weights.nnz << std::endl;

}


void prune_mnist_problem()
{
	cusparseHandle_t handle;
	cusparseCreate(&handle);
	int ini_nodes = 28 * 28;
	int out_nodes = 10;
	int extra_nodes = 250+150+50;
	int nodes = ini_nodes + out_nodes + extra_nodes;
	int batch = 10;
	int depth = 5;
	int delay_iteration = 0;
	double learning_rate = 0.01;
	double error_sum = 0;
	bool adam = true;

	std::vector<activation_function> act_f(nodes, RELU);

	for (int i = ini_nodes; i < ini_nodes + out_nodes; ++i)
		act_f[i] = LINEAL;


	for (int i = ini_nodes + out_nodes; i < nodes - extra_nodes / 2; ++i)
		act_f[i] = SIGMOID;

	Prune_evolver pe(ini_nodes, out_nodes, nodes, depth, depth, delay_iteration, batch, learning_rate, learning_rate, act_f, adam, handle);
	//pe.connect_input_output();
	std::vector<int> hid_layers = { 250, 150, 50 };
	pe.connect_like_mlp(hid_layers);
	pe.update_rnn();

	std::cout << "Number of Weights: " << pe.rnn.weights.nnz << std::endl;
	int max_weights = 25000;
	RNN_mnist_problem(pe.rnn, -1, 10, true);
	pe.read_weights_and_biases_from_device();

	for (int i = 0; i < 10; i++) {
		//pe.add_fully_connected_nodes(10);
		//pe.fully_connect_n_nodes(2);
		pe.add_n_random_connections(pe.rnn.weights.nnz *0.2);

		pe.update_rnn();

		RNN_mnist_problem(pe.rnn, -1, 1, false);

		pe.read_weights_and_biases_from_device();

		int conn_to_prune = std::max((pe.rnn.weights.nnz - max_weights), (int)(pe.rnn.weights.nnz * 0.05));

		int steps = 20;
		for (int p_it = 0; p_it < steps; ++p_it) {
			RNN_mnist_problem(pe.rnn, 1000, 1, false);

			pe.read_weights_and_biases_from_device();

			pe.prune_weights(conn_to_prune / steps);
			pe.update_rnn();

		}
		
		//pe.randomize_weights_and_reset_biases();

		std::cout <<"Iteration: "<<i<< " Number of Weights: " << pe.rnn.weights.nnz << " exces: "<< pe.rnn.weights.nnz - max_weights <<std::endl;
	}
	RNN_mnist_problem(pe.rnn, -1, 1, true);
	pe.randomize_weights_and_reset_biases();
	pe.update_rnn();
	pe.rnn.learning_rate = 0.01;
	pe.rnn.bias_learning_rate = 0.01;
	RNN_mnist_problem(pe.rnn, -1, 10, true);
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

	auto test_sol_list = read_caltech_test_labels();

	cusparseHandle_t handle;
	cusparseCreate(&handle);
	int ini_nodes = 28 * 28;
	int out_nodes = 101;
	int extra_nodes = 500 + 250 + 200 + 100 + 50;
	int nodes = ini_nodes + out_nodes + extra_nodes;
	int batch = 10;
	int depth = 7;
	int delay_iteration = 0;
	double learning_rate = 0.01;
	double error_sum = 0;
	bool adam = true;

	std::vector<activation_function> act_f(nodes, RELU);

	for (int i = ini_nodes; i < ini_nodes + out_nodes; ++i)
		act_f[i] = LINEAL;


	for (int i = ini_nodes + out_nodes; i < nodes; ++i)
		act_f[i] = SIGMOID;

	Prune_evolver pe(ini_nodes, out_nodes, nodes, depth, depth, delay_iteration, batch, learning_rate, learning_rate, act_f, adam, handle);
	std::vector<int> hid_layers= {500, 250, 200, 100, 50};
	pe.connect_like_mlp(hid_layers);
	//pe.connect_input_output();
	

	pe.update_rnn();
	std::cout << "Number of Weights: " << pe.rnn.weights.nnz << std::endl;

	int max_weights = pe.rnn.weights.nnz*0.7;
	//pe.add_n_random_connections(max_weights);

    RNN_caltech101_sil_problem(pe.rnn, -1, 15, true, inp, sol_list, test_inp, test_sol_list);
	for (int p_it = 0; p_it < 20; ++p_it) {
		RNN_caltech101_sil_problem(pe.rnn, 1000, 1, false, inp, sol_list, test_inp, test_sol_list);

		pe.read_weights_and_biases_from_device();

		pe.prune_weights((pe.rnn.weights.nnz - max_weights) / 20);
		pe.update_rnn();

	}
	pe.update_rnn();
	for (int i = 0; i < 10; i++) {
		//pe.add_fully_connected_nodes(10);
		//pe.fully_connect_n_nodes(2);
		//pe.divide_weights_and_biases(1.25);
		//pe.add_n_random_connections_weight_strat(pe.rnn.weights.nnz * 0.1);
		pe.add_n_random_connections(pe.rnn.weights.nnz * 0.1);
		pe.update_rnn();

		RNN_caltech101_sil_problem(pe.rnn, -1, 1, false, inp, sol_list, test_inp, test_sol_list);

		int conn_to_prune = std::max((pe.rnn.weights.nnz - max_weights), (int)(pe.rnn.weights.nnz * 0.05) );

		int steps = 20;
		for (int p_it = 0; p_it < steps; ++p_it){
			RNN_caltech101_sil_problem(pe.rnn, 1000, 1, false, inp, sol_list, test_inp, test_sol_list);

			pe.read_weights_and_biases_from_device();
			
			pe.prune_weights(conn_to_prune/steps);
			pe.update_rnn();

		}
		/*if (pe.rnn.weights.nnz > max_weights)
			pe.prune_weights((pe.rnn.weights.nnz - max_weights));
		else
			pe.prune_weights((int)(pe.rnn.weights.nnz * 0.05));*/



		std::cout << "Iteration: " << i << " Number of Weights: " << pe.rnn.weights.nnz << " exces: " << pe.rnn.weights.nnz - max_weights << std::endl;
	}
	pe.read_weights_and_biases_from_device();
	pe.update_rnn();
	pe.rnn.learning_rate = 0.001;
	pe.rnn.bias_learning_rate = 0.001;
	RNN_caltech101_sil_problem(pe.rnn, -1, 2, true, inp, sol_list, test_inp, test_sol_list);
	std::cout << "Number of Weights: " << pe.rnn.weights.nnz << std::endl;
}

