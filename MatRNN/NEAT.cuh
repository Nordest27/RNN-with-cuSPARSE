#include "RNN.cuh"

class NEAT_genotype {

	cusparseHandle_t *handle;
public:
	bool updated = false;
	double eval;

	int max_iters;
	int max_depth;
	int delay_iters;
	int batch;
	double learning_rate;
	double bias_learning_rate;

	int ini;
	int out;
	int nodes;

	COO_matrix weights;
	std::vector<bool> active_weights;
	int how_many_active_w = 0;
	std::vector<double> biases;
	std::vector<activation_function> act_f;
	RNN rnn;

	NEAT_genotype(int i_nod, int o_nod, int nod, int m_i, int m_d, int d_it, int bat, double lr, double b_lr, std::vector<activation_function> &a_f, cusparseHandle_t& hand);
	NEAT_genotype(const NEAT_genotype &ng);

	void connect_input_output();
	bool add_weight(int i, int j, double value);
	void add_node(activation_function activ_f);
	void forward_prop(std::vector<double>& input_values);
	double evaluate(std::vector<double>& correct_values, bool classif);
	void update_weights_and_biases();
	void update_rnn();
};

//bool operator<(const NEAT_genotype& ng1, const NEAT_genotype& ng2);

class NEAT_pool {
	cusparseHandle_t handle;
public:
	std::vector<std::pair<float, std::unique_ptr<NEAT_genotype>>> genotypes;

	int max_pool_size;

	NEAT_pool(int md, int mi, float lr, int ini_n, int out_n, int pool_s, activation_function output_act_f, bool inp_out_con);

	void   forward_prop(std::vector<double>& input_values);

	double evaluate(std::vector<double>& correct_values, bool classif);
		void reset_evals();

	void   update_weights_and_biases();

	void   update_rnns();

	void   reset_rnns();

	void   mutate( int mutations );
		NEAT_genotype add_random_weight( const NEAT_genotype &ng );
		NEAT_genotype add_random_node( const NEAT_genotype& ng );
		NEAT_genotype mutate_gene_disable( const NEAT_genotype& ng );
		NEAT_genotype mutate_gene_reenable( const NEAT_genotype& ng );
		NEAT_genotype add_multiple_weights_and_nodes(const NEAT_genotype& ng, int how_many_first, int how_many_last);


	void   breed( int children );
		NEAT_genotype mate(int p1, int p2);

	double compatibility(const NEAT_genotype& gen1, const NEAT_genotype& gen2);

	void   trim_pool();
	void   sort_pool();
	void   next_gen();

};

void NEAT_xor_problem();
void NEAT_mnist_problem();
