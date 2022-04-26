#include "NEAT.cuh"

NEAT_genotype::NEAT_genotype(int i_nod, int o_nod, int nod, int m_i, int m_d, int d_it, int bat, double lr, double b_lr, std::vector<activation_function> &a_f, cusparseHandle_t& hand)
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

	updated = false;
}
NEAT_genotype::NEAT_genotype(const NEAT_genotype &ng)
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
	active_weights = ng.active_weights;
	how_many_active_w = ng.how_many_active_w;
	biases = ng.biases;


	updated = false;
}

void NEAT_genotype::connect_input_output()
{

	for (int i = 0; i < ini; i++)
		for (int o = 0; o < out; o++)
			add_weight(i, ini+o, randomdouble());
}

bool NEAT_genotype::add_weight(int i, int j, double value)
{
	if (!weights.insert_elm(i, j, value)) return false;
	active_weights.push_back(true);
	how_many_active_w++;
	return true;
}

void NEAT_genotype::add_node( activation_function activ_f )
{
	nodes++;
	weights.cols = nodes;
	weights.rows = nodes;
	biases.push_back(0);
	act_f.push_back(activ_f);
}

void NEAT_genotype::forward_prop(std::vector<double>& input_values ) {
	for (int i = 0; i < max_iters; ++i)
	{
		rnn.set_input_values(input_values);
		rnn.forward_prop();
	}
}

double NEAT_genotype::evaluate(std::vector<double>& correct_values, bool classif)
{
	double error = rnn.backward_prop(correct_values, classif);
	eval += error;
	return error;
}

void NEAT_genotype::update_weights_and_biases()
{
	rnn.update_weights_and_biases();
	//rnn.read_weights_and_biases_from_device(weights.cooValues, biases);
}

void NEAT_genotype::update_rnn()
{
	if (!updated) {
		COO_matrix aux_weights;
		for (int i = 0; i < weights.nnz; ++i)
			if (active_weights[i])
				aux_weights.insert_elm(weights.cooColInd[i], weights.cooRowInd[i], weights.cooValues[i]);
		rnn = RNN(ini, out, nodes, learning_rate, max_depth, batch, act_f, aux_weights, *handle);
		rnn.bias_learning_rate = bias_learning_rate;
		rnn.delay_iters = delay_iters;
		rnn.max_iters = max_iters;
		rnn.initialize_device_public();
		updated = true;
	}
	else {

		//std::vector<double> lmao(nodes);
		//std::cout <<"bisize: "<< biases.size() << std::endl;
		rnn.read_weights_and_biases_from_device(weights.cooValues, biases);
	}
}

////////////////////NEAT POOL/////////////////////

NEAT_pool::NEAT_pool(int md, int mi, float lr, int ini_n, int out_n, int pool_s, activation_function output_act_f, bool inp_out_con)
{
    max_pool_size = pool_s;


	cusparseCreate(&handle);


	std::vector<activation_function> act_f(ini_n+out_n, RELU);

	for (int i = ini_n; i < ini_n + out_n; ++i)
		act_f[i] = output_act_f;

	
	genotypes.push_back(std::pair<float, std::unique_ptr<NEAT_genotype>>(0, new NEAT_genotype (ini_n, out_n, ini_n + out_n, mi, md, 0, 10, lr, lr, act_f, handle) ));
	if (inp_out_con)
		genotypes[0].second->connect_input_output();
	cudaError_t cudaStatus = cudaGetLastError();
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "Something failed: %s\n", cudaGetErrorString(cudaStatus));
	}

}

void   NEAT_pool::forward_prop(std::vector<double>& input_values)
{
	for (auto& gen : genotypes)
		gen.second->forward_prop(input_values);
}

double NEAT_pool::evaluate(std::vector<double>& correct_values, bool classif)
{
	double error = -1;

	for (auto& gen : genotypes) {
		if (error == -1) error = gen.second->evaluate(correct_values, classif);
		else gen.second->evaluate(correct_values, classif);
		gen.first = gen.second->eval*(1/*(gen.second->max_depth+gen.second->max_iters)*
										(gen.second->nodes*(0.005) + (0.0005)*gen.second->how_many_active_w)*/);
		if (gen.first == NAN || gen.first == -NAN) {
			//std::cout << "FUCK" << std::endl;
			gen.first = INT_MAX;
		}
	}
	
	return error;
}
void NEAT_pool::reset_evals()
{
	for (auto& gen : genotypes) {
		gen.second->eval = 0;
		gen.first = 0;
	}
}

void   NEAT_pool::update_weights_and_biases()
{
	for (auto& gen : genotypes)
		gen.second->update_weights_and_biases();
}

void   NEAT_pool::update_rnns()
{
	for (auto& gen : genotypes)
		gen.second->update_rnn();
}

void   NEAT_pool::reset_rnns()
{
	for (auto& gen : genotypes)
		gen.second->rnn.reset();
}

void   NEAT_pool::mutate(int mutations)
{
	int pool_size = genotypes.size();
	update_rnns();
	for (int i = 0; i < mutations; ++i) {

		int genotype = 0;
		if (i > pool_size / 10) 
			genotype = (int)sqrt(rand() % (pool_size * pool_size));

		while (std::isnan(genotypes[genotype].first))
			genotype = (int)sqrt(rand() % (pool_size * pool_size));

		double selector = abs(randomdouble()) * 2;
		//60% add multiple local connections + nodes
		if (selector > 0.4 && genotypes[0].second->nodes > 100)
			genotypes.push_back(std::pair<float, std::unique_ptr<NEAT_genotype>>(0, new NEAT_genotype(add_multiple_weights_and_nodes(*(genotypes[rand() % pool_size].second), rand()%(10)+1, rand() % 10 + 1))));
		//10% add node
		//if (selector > 0.7)
			//genotypes.push_back(std::pair<float, std::unique_ptr<NEAT_genotype>>(0, new NEAT_genotype(add_random_node(*(genotypes[genotype].second)))));
		//70% add link
		else if (selector > 0)
			genotypes.push_back(std::pair<float, std::unique_ptr<NEAT_genotype>>(0, new NEAT_genotype(add_random_weight(*(genotypes[genotype].second)))));
		
		//std::cout << selector << std::endl;
		//7.5% disable weight
		/*else if (selector > 0.025)
			genotypes.push_back(std::pair<float, std::unique_ptr<NEAT_genotype>>(0, new NEAT_genotype(mutate_gene_disable(*(genotypes[rand() % pool_size].second)))));
		//2.5% enable weight
		else
			genotypes.push_back(std::pair<float, std::unique_ptr<NEAT_genotype>>(0, new NEAT_genotype(mutate_gene_reenable(*(genotypes[rand() % pool_size].second)))));*/
		
		//5% probability of changing the depth
		//if (randomdouble() > 0.95)
			//genotypes[genotypes.size() - 1].second->max_depth--;

		//5% probability of changing the iters
		/*if (randomdouble() > 0.95) {
			genotypes[genotypes.size() - 1].second->max_iters++;
			genotypes[genotypes.size() - 1].second->max_depth++;
		}*/

	}

}
NEAT_genotype NEAT_pool::add_multiple_weights_and_nodes(const NEAT_genotype& ng, int how_many_first, int how_many_last)
{
	int nodes = ng.nodes;
	int extra_nodes = sqrt(rand() % (100))/2;
	activation_function act_f[] = { SIGMOID, RELU };
	NEAT_genotype res_ng(ng);
	for(int i = 0; i < extra_nodes; ++i)
		res_ng.add_node(act_f[rand() % 2]);

	int i = rand() % (nodes-how_many_first);
	int j = rand() % (nodes-how_many_last);

	for (; i < how_many_first; ++i) {
		if(extra_nodes == 0)
			for (; j < how_many_last; ++j)
				res_ng.add_weight(i, j, randomdouble());

		for (int u = nodes; u < nodes + extra_nodes; ++u)
			res_ng.add_weight(i, u, randomdouble());
	}

	for (int u = nodes; u < nodes + extra_nodes; ++u)
		for (; j < how_many_last; ++j)
			res_ng.add_weight(u, j, randomdouble());

	for (int u = nodes; u < nodes + extra_nodes; ++u)
		for (int t = nodes; t < nodes + extra_nodes; ++t)
			res_ng.add_weight(u, t, randomdouble());

	return res_ng;
}
NEAT_genotype NEAT_pool::add_random_weight(const NEAT_genotype& ng)
{
	int nodes = ng.nodes;
	if (ng.weights.nnz > nodes * nodes / 2) return add_random_node(ng);
	int i = rand() % nodes;
	int j = rand() % nodes;
	double value = randomdouble();
	NEAT_genotype res_ng(ng);
	while (!res_ng.add_weight(i, j, value))
	{
		i = rand() % nodes;
		j = rand() % nodes;
	}
	return res_ng;
}
NEAT_genotype NEAT_pool::add_random_node(const NEAT_genotype& ng)
{
	int weights = ng.weights.nnz;
	if (ng.how_many_active_w == 0) return add_random_weight(ng);
	int chosen_weight = rand() % weights;
	while (!ng.active_weights[chosen_weight]) 
		chosen_weight = rand() % weights;
	int nodes = ng.nodes;
	NEAT_genotype res_ng(ng);
	activation_function act_f[] = {SIGMOID, RELU};

	res_ng.add_node(act_f[rand()%2]);
	
	res_ng.active_weights[chosen_weight] = false;
	res_ng.how_many_active_w--;

	res_ng.add_weight(res_ng.weights.cooColInd[chosen_weight], nodes, randomdouble());

	res_ng.add_weight(nodes, res_ng.weights.cooRowInd[chosen_weight], randomdouble());

	return res_ng;
}

NEAT_genotype NEAT_pool::mutate_gene_reenable(const NEAT_genotype& ng)
{

	if (ng.weights.nnz == 0) return add_random_weight(ng);

	std::vector<int> disabled_links_indices;
	for (int i = 0; i < ng.weights.nnz; ++i)
		if (!ng.active_weights[i])
			disabled_links_indices.push_back(i);

	if (disabled_links_indices.size() == 0) return mutate_gene_disable(ng);
	
	NEAT_genotype res_ng(ng);
	res_ng.active_weights[disabled_links_indices[rand() % disabled_links_indices.size()]] = true;
	res_ng.how_many_active_w++;
	return res_ng;
}

NEAT_genotype NEAT_pool::mutate_gene_disable(const NEAT_genotype& ng)
{
	if (ng.weights.nnz == 0) return add_random_weight(ng);
	std::vector<int> disabled_links_indices;
	for (int i = 0; i < ng.weights.nnz; ++i)
		if (ng.active_weights[i])
			disabled_links_indices.push_back(i);

	if (disabled_links_indices.size() == 0) return mutate_gene_reenable(ng);

	NEAT_genotype res_ng(ng);
	res_ng.active_weights[disabled_links_indices[rand() % disabled_links_indices.size()]] = false;
	res_ng.how_many_active_w--;

	return res_ng;
}


void  NEAT_pool::breed(int children) 
{

}

NEAT_genotype NEAT_pool::mate(int p1, int p2) 
{
	auto& gen1 = genotypes[p1].second;
	double error_p1 = genotypes[p1].first;
	auto& gen2 = genotypes[p2].second;
	double error_p2 = genotypes[p2].first;
	bool p1Better = true;
	/*
	if (error_p1 > error_p2)
		p1Better = false;

	int g1_nnz = gen1->weights.nnz;
	int g2_nnz = gen2->weights.nnz;
	int i = 0; 
	int j = 0;
	while (i < g1_nnz || j < g2_nnz)
	{
		if( gen1->weights.cooRowInd[i] == gen2->weights.cooRowInd[j] &&
			gen1->weights.cooRowInd[i] == gen2->weights.cooRowInd[j] )

	}
	*/
	return *gen1;
}

/*bool operator<(const NEAT_genotype& ng1, const NEAT_genotype& ng2)
{
	return ng1.eval < ng2.eval;
}
*/
void NEAT_pool::sort_pool()
{
	sort(genotypes.begin(), genotypes.end());
}
void   NEAT_pool::trim_pool()
{
	sort_pool();
	std::cout << "Evals: \n";
	for (auto& gen : genotypes)
		std::cout << gen.first << " ";
	std::cout << "\n";

	while (genotypes.size() > max_pool_size)
		genotypes.erase(genotypes.end()-1);
}

void  NEAT_pool::next_gen()
{
	trim_pool();
	mutate(max_pool_size*2);
    for(int i = 0; i < max_pool_size; ++i)
		genotypes.erase(genotypes.begin());
}

void NEAT_xor_problem()
{
	int ini_nodes = 2;
	int out_nodes = 1;
	int nodes = 4;
	int batch = 4;
	int depth = 3;
	int pool_s = 10;
	bool inp_out_con = true;
	double learning_rate = 1.0;
	double error_sum = 0;

	NEAT_pool np(depth, depth, learning_rate, ini_nodes, out_nodes, pool_s, SIGMOID, inp_out_con);
	
	/*np.genotypes[0].second->add_node(SIGMOID);
	np.genotypes[0].second->add_weight(0, 2, 1);
	np.genotypes[0].second->add_weight(1, 2, 1);
	np.genotypes[0].second->add_weight(1, 3, 1);
	np.genotypes[0].second->add_weight(0, 3, 1);
	np.genotypes[0].second->add_weight(3, 2, 1);*/

	std::vector<std::vector<double> > inp = { {0,0},{ 0,1 },{ 1,0 }, { 1,1 } };

    np.mutate(10);

	np.update_rnns();

	std::vector<double> sol = { 0 };

	for (int it = 0; it < 10000; it++) {
		error_sum = 0;
		for (auto input : inp)
		{
			sol[0] = 0;
			if (input[0] != input[1])
				sol[0] = 1;

		    np.forward_prop(input);
			error_sum += np.evaluate(sol, false);


			np.reset_rnns();
			//std::cout << "---------------------\n";
		}

		if (it % 100 == 0) {
			//np.genotypes[0].second->rnn.print_matrix();
			/*for (auto& gen : np.genotypes) {

				std::cout << "result: " << gen.first << std::endl;
				std::cout << "\n matrix print: \n";
				gen.second->rnn.print_matrix();

				for (int i = 0; i < gen.second->biases.size(); ++i)
					std::cout << gen.second->biases[i] << " ";
				std::cout << "\n";
			}*/
			np.genotypes[0].second->rnn.print_matrix();
			std::cout << "Iteration " << it << ": " << error_sum / batch << std::endl;

		}

		np.update_weights_and_biases();
		if (it % 10 == 0)
		{
			np.trim_pool();
			np.mutate(10);
			np.update_rnns();
		}

		np.reset_evals();
	}

	/*for (auto input : inp)
	{
		sol[0] = 0;
		if (input[0] != input[1]) {
			//printf("%f %f\n",input[0], input[1]);
			sol[0] = 1;
		}


		//rnn.print_matrix();

		for (int i = 0; i < np.max_depth; ++i)
		{
			//std::cout << "Time " << i << " : " << time<<std::endl;
			rnn.set_input_values(input);
			rnn.forward_prop();
		}

		std::cout << "input: " << input[0] << " " << input[1] << " res: " << rnn.values_through_time[rnn.values_through_time.size() - 1][2] << std::endl;

		rnn.reset();
	}*/

}


void NEAT_mnist_problem()
{
	int ini_nodes = 28*28;
	int out_nodes = 10;
	int nodes = 4;
	int batch = 1;
	int depth = 5;
	int pool_s = 25;
	bool inp_out_con = true;
	double learning_rate = 0.1;
	double error_sum = 0;

	NEAT_pool np(depth, depth, learning_rate, ini_nodes, out_nodes, pool_s, SIGMOID, inp_out_con);
	//get first rnn
	//train previously to get to sub obtimum
	np.update_rnns();
	/*np.genotypes[0].second->rnn.adam = true;
	RNN_mnist_problem(np.genotypes[0].second->rnn, 1);*/
	np.genotypes[0].second->rnn.adam = true;

	//np.next_gen();
	//np.update_rnns();
	int NEAT_batch = 10;
	int NEAT_counter = 0;
	int generations = 100;

	//for (int i = 0; i < pool_s; ++i)
		//np.genotypes.push_back(std::pair< double, std::unique_ptr<NEAT_genotype>>(0, new NEAT_genotype( *np.genotypes[0].second ) ));
	//np.update_rnns();
	auto inp = mnist::read_training_images();

	auto sol_list = mnist::read_training_labels();

	std::vector<int> indices(inp.size(), 0);
	for (int i = 0; i < inp.size(); ++i)
		indices[i] = i;

	int examples = inp.size();
	/*examples = 1001;
	std::vector<double> input(28 * 28);
	std::vector<double> sol(10);
	int it = 0;
	while( generations > 0  )
	{
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

			np.forward_prop(input);
			error_sum += np.evaluate(sol, true);
			//std::cout << error_sum << std::endl;
			if (image % batch == 0) {
				np.update_weights_and_biases();
			}

			if (NEAT_batch < NEAT_counter &&  error_sum/image < 0.2)
			{
				std::cout << "Generation: " << generations << std::endl;
				np.next_gen();
				np.update_rnns();
				np.reset_evals();
				NEAT_counter = 0;
				generations--;
			}
			NEAT_counter++;

			if (examples > 1000 && image > print_stop)
			{
				std::cout << "%" << 100 * image / examples << " error = " << error_sum / image << std::endl;
				print_stop += examples / 10;
			}

			np.reset_rnns();
		}
		std::random_shuffle(indices.begin(), indices.begin()+examples);
		std::cout << "Iteration " << it << ": " << error_sum / examples << std::endl;
		it++;
		error_sum = 0;
		NEAT_counter = 0;
		np.sort_pool();


	}*/

	NEAT_genotype gen(*np.genotypes[0].second);
	gen.update_rnn();
	for (int i = 0; i < 100; i++) {
		gen = NEAT_genotype(np.add_multiple_weights_and_nodes(gen, rand() % (10) + 1, rand() % 10 + 1));

		gen.update_rnn();
	}

	//get winner
	RNN& rnn = np.genotypes[0].second->rnn;
	rnn = gen.rnn;
	rnn.adam = true;
	rnn.learning_rate = 0.01;
	rnn.bias_learning_rate = 0.01;
	rnn.batch = 10;
	std::cout<< "max_iters: "<< rnn.max_iters << std::endl;
	std::cout<< "new nodes: "<< rnn.nodes - ini_nodes - out_nodes<<std::endl;
	std::cout<< "new weights: " << rnn.weights.nnz - ini_nodes*out_nodes << std::endl;
	//train further and test winner
	RNN_mnist_problem(rnn, -1, 10, true);
}