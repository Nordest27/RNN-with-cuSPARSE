#include "RNN.cuh"

class Prune_evolver {

	cusparseHandle_t* handle;

public:
	bool updated = false;
	bool adam = false;
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
	int how_many_active_w = 0;
	std::vector<double> biases;
	std::vector<activation_function> act_f;
	RNN rnn;

	Prune_evolver(int i_nod, int o_nod, int nod, int m_i, int m_d, int d_it, int bat, double lr, double b_lr, std::vector<activation_function>& a_f, bool adamb, cusparseHandle_t& hand);
	Prune_evolver(const Prune_evolver& ng);

	void connect_input_output();

	bool add_weight(int i, int j, double value);
	void add_node(activation_function activ_f);
	void forward_prop(std::vector<double>& input_values);
	double evaluate(std::vector<double>& correct_values, bool classif);
	void update_weights_and_biases();

	void prune_weights( int how_many );
	void prune_weights( double threshold );

	void connect_to_all(int i);

	void connect_like_mlp(std::vector<int>& hid_layers);

	void add_fully_connected_nodes(int n);
	void fully_connect_n_nodes(int n);
	void add_n_random_connections(int n);
	void add_n_random_connections_weight_strat(int n);

	void read_weights_and_biases_from_device();

	void update_rnn();

	void randomize_weights_and_reset_biases();
	void divide_weights_and_biases(double d);
};

void prune_test();
void prune_mnist_problem();
void prune_caltech_problem();