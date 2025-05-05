#include "kernel.cuh"

class RNN
{
    public:

	int time_step = 0;
    
	int max_iters;
	int max_depth;
	int delay_iters;
	int batch;

	double learning_rate;
	double bias_learning_rate;

	double beta1 = 0.9, beta2 = 0.999;
	double powBeta1 = 0.9, powBeta2 = 0.999;
	double eps = 1e-8;
	bool adam = false;
	bool training = true;
	double inp_dropout = 0;

	int ini;
	int out;
	int nodes;

	Vector current_values;

	std::vector<int> mask;
	std::vector<activation_function> act_f;

	Vector biases;
	Vector biases_update;

	COO_matrix weights;
	Vector weights_update;

	Matrix values_through_time;
	Matrix dx_through_time;
	Vector current_dx;

	private:
	//////////////////////////DEVICE////////////////////////////

	bool dev_ini = false;
	
	// Weights and input/output vectors 
	int* dA_rows, * dA_columns;
	double* dA_values, * dX, * dY;

	// Values through time
	double* d_values_through_time;
	double* d_dx_through_time;

	//biases
	double* d_biases;

	// Weights and biases update
	double* d_weights_update;
	double* d_biases_update;

	// Gradient
	double* d_gradient;
	double* d_dx;

	// Adam
	double* d_m;
	double* d_v;
	double* d_m_corr;
	double* d_v_corr;

	// Input
	double* d_input;

	// Mask
	int* d_mask;

	// Activation functions
	int* d_act_f;

	// CUSPARSE APIs
	cusparseHandle_t  *handle;
	cusparseSpMatDescr_t matA;
	cusparseDnVecDescr_t vecX, vecY, gradX;
	void* dBuffer = NULL;
	size_t bufferSize = 0;

	double alpha = 1.0f;
	double beta = 0.0f;
	int block_size = 1024;

	int get_blocks(int size);

	int get_threads(int size, int blocks);

	void initialize_device();

	void copy_weights_to_device();

	void copy_weights_and_biases_to_device();

	void copy_vect_to_device(Vector& values, double* device_vect, int size);

	void copy_vect_to_device(std::vector<activation_function>& values, int* device_vect, int size);

	void copy_vect_to_device(std::vector<int>& values, int* device_vect, int size);

	void execute_matrix_vector_prod(cusparseOperation_t transpose, cusparseDnVecDescr_t vect, cusparseDnVecDescr_t output);
	
	void execute_add_biases();

	void execute_set_input_values(Vector& input_values);

	void execute_add_input_values(int how_many_input_values);
	
	void execute_update_gradient();

	void execute_sub_biases_update();

	void execute_sub_weights_update();

	void execute_update_weights_and_biases();

	void execute_update_weights_and_biases_Adam();

	void execute_use_activation_functions();

	void read_weights_of_device();

	void read_vect_of_device(Vector& result, double* device_vect, int size);

	void move_vect_of_device(double* device_vect1, double* device_vect2, int size);

	void reset_vect(double* device_vect, int size);

	void reset_vect(int* device_vect, int size);

	int n_offset_by_time();

    ////////////////////////////////////////////////////////////

    public:
	
	void read_weights_and_biases_from_device(Vector& weights, Vector& biases);

	void initialize_device_public();

	void empty_device();

	RNN();

	RNN(int i, int o, int n, double lr, int m_d, int bat,  std::vector<activation_function> &a_f, COO_matrix &w, cusparseHandle_t& hand);

	~RNN();

	void add_connection(int node_i, int node_j, double value);

	void reset();

	void full_reset();

	void set_input_values(Vector& input_values);

	void add_input_values(int how_many_input_values);

	void forward_prop();

	//synflow
	Vector synflow_cycle(bool classif, Vector &ones, int wich);
	Vector synflow_cycle(bool classif, std::vector<std::pair<Vector, double>>& dataset, int samples);
	//

	void partition(std::vector<Vector>& inp, Vector& sol_list, double train_val_part, double val_test_part,
				   std::vector < std::pair<Vector, double> >&     dataset, std::vector < std::pair<Vector, double> >  &test_data,
				   std::vector < std::pair<Vector, double> >&     val_data, double div, bool shuffle);

	void partition(std::vector<std::vector<int>>& inp, std::vector<int>& sol_list, double train_val_part, double val_test_part,
		std::vector < std::pair<Vector, double> >& dataset, std::vector < std::pair<Vector, double> >& test_data,
		std::vector < std::pair<Vector, double> >& val_data, double div, bool shuffle);

	void partition(std::vector<std::vector<uint8_t>>& inp, std::vector<uint8_t>& sol_list, double train_val_part, double val_test_part,
		std::vector < std::pair<Vector, double> >& dataset, std::vector < std::pair<Vector, double> >& test_data,
		std::vector < std::pair<Vector, double> >& val_data, double div, bool shuffle);

	void copy_wb_to_dev();

	void read_values_through_time_from_device();

	double forward_prop_cycle(Vector& input_values, Vector& correct_values, bool classif);

	double forward_prop_one_inp_cycle(Vector& input_values, Vector& correct_values, bool classif);

	double get_error(Vector& correct_values, bool classif);

	double backward_prop(Vector &correct_values, bool classif);

	void softmax(Vector& values, int ini, int fi);

	double train_step(Vector &input_values, Vector &correct_values, bool classif);

	double train_step_one_inp_set(Vector& input_values, Vector& correct_values, bool classif);

	double time_sens_train_step(std::vector< Vector>& input_values, std::vector< Vector>& correct_values, bool classif);

	void update_weights_and_biases();

	void print_matrix();

	void print_rnn_to_file();


};

void RNN_xor_problem();

void RNN_mnist_problem();

void RNN_mnist_problem(RNN& rnn, int examples, int iters, bool test);

void RNN_mnist_autoencode_problem();

void RNN_previous_sign_problem();

void RNN_sum_vect_problem();

void RNN_shakespeare_problem();

void RNN_delayed_str_problem();

bool RNN_caltech101_sil_problem(RNN& rnn, int exampl, int iters, bool test, std::vector<std::vector<int>>& inp, std::vector<int>& sol_list,
																			std::vector<std::vector<int>>& test_inp, std::vector<int>& test_sol_list,
																			std::vector<std::vector<int>>& val_inp, std::vector<int>& val_sol_list);

void RNN_chess_problem(RNN& rnn, int exampl, int iters, std::vector<Vector> &inp, Vector &sol_list);

bool generic_classif_RNN_problem(RNN& rnn, int exampl, int iters, bool test,
	std::vector < std::pair<Vector, double> >& dataset, std::vector < std::pair<Vector, double> >& test_data,
	std::vector < std::pair<Vector, double> >& val_data);

bool generic_regress_RNN_problem(RNN& rnn, int exampl, int iters, bool test,
	std::vector < std::pair<Vector, double> >& dataset, std::vector < std::pair<Vector, double> >& test_data,
	std::vector < std::pair<Vector, double> >& val_data);
